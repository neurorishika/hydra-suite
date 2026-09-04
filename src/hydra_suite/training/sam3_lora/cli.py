"""SAM3 LoRA training entry point -- runs INSIDE the ``hydra-sam3`` sidecar
conda env, launched by ``train.py`` as
``python -m hydra_suite.training.sam3_lora.cli --spec ... --run-dir ...``.

This module (and only this module, in this package) may import ``sam3`` and
``torch`` -- but only inside function bodies, never at module scope, so a
syntax check or a stray import elsewhere in the package never requires
``sam3`` to be installed.

Progress and logging cross the process boundary as single-line JSON records
via ``protocol.emit_log``/``emit_progress``, printed to stdout with a
sentinel prefix; `train.py` parses them back out on the other end.

Cancellation is NOT plumbed into this process: the launcher cancels a run by
`terminate()`/`kill()`-ing this whole process, so there is no
`should_cancel` callback here. If this process is killed mid-epoch, it exits
without having written `adapters.pt`, which the launcher already treats as
a failed/canceled run -- see `train.py`'s artifact-existence check.

Checkpoint selection is always the LAST epoch's weights, never the epoch
with the best validation loss: the spike found val loss anti-correlated with
held-out AP (the fold with the worst val loss had the best held-out AP75).
Do not add best-checkpoint selection or early stopping on val loss here.

The training set is built and checked for emptiness BEFORE any `sam3` model
is loaded: an empty dataloader must exit nonzero, never silently train
nothing and exit 0 -- a zero-initialised LoRA `lora_B` makes an untrained
adapter a mathematical no-op, so a fake-success run would publish a
"finetuned" checkpoint byte-identical to stock SAM3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams, sam3_prompt_text_error

from .artifacts import write_completion_marker
from .dataloader import (
    batch_count,
    build_descriptors,
    collate_batches,
    collate_epoch_batches,
    query_count,
    try_build_descriptors,
)
from .lora import adapter_state_dict, inject_adapters, lora_config_from_params
from .perflib_compat import install_grad_safe_addmm_act
from .protocol import emit_log, emit_progress
from .sizing import expected_lora_trainable_params


class _SidecarSpec:
    """Minimal stand-in for `TrainingRunSpec` inside the sidecar.

    The launcher serialises the FULL spec to `spec.json` via
    `TrainingRunSpec.to_dict()`; this only reconstructs the fields the
    training loop below actually reads (`seed`, `derived_dataset_dir`,
    `sam3_params`), via the real `Sam3LoraParams` dataclass rather than a
    duplicated schema, so new params fields need no change here.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.seed: int = int(data.get("seed", 42))
        self.derived_dataset_dir: str = data["derived_dataset_dir"]
        self.device: str = str(data.get("device", "cuda"))
        sam3_data = data.get("sam3_params")
        self.sam3_params = Sam3LoraParams(**sam3_data) if sam3_data else None


def _load_spec(spec_path: Path) -> _SidecarSpec:
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    return _SidecarSpec(data)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_with_warmup(warmup_steps: int, total_steps: int):
    def _fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        span = max(1, total_steps - warmup_steps)
        progress = min(max(float(step - warmup_steps) / float(span), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _fn


def _forward_batch(
    batch: Any,
    model: Any,
    device: Any,
    *,
    copy_to_device: Any | None = None,
) -> tuple[Any, list[Any], Any]:
    """Move one collated batch to ``device``, convert targets, then forward.

    Meta's collator constructs a CPU ``BatchedDatapoint``. Its trainer uses
    ``copy_data_to_device`` on that unwrapped object before both
    ``back_convert`` and model forward; keep this sidecar on the same seam so
    every nested tensor (images, boxes, masks, and validity arrays) moves
    together.
    """
    if copy_to_device is None:
        from sam3.model.utils.misc import copy_data_to_device

        copy_to_device = copy_data_to_device

    model_input = batch["input"] if isinstance(batch, dict) else batch
    model_input = copy_to_device(model_input, device, non_blocking=True)
    targets = [model.back_convert(target) for target in model_input.find_targets]
    outputs = model(model_input)
    return model_input, targets, outputs


def _build_loss_wrapper(
    loss_wrapper_type: Any,
    *,
    loss_fns_find: list[Any],
    matcher: Any,
    o2m_matcher: Any,
) -> Any:
    """Build Meta's loss without distributed collectives in this sidecar."""
    return loss_wrapper_type(
        loss_fns_find=loss_fns_find,
        normalization="local",
        matcher=matcher,
        o2m_weight=2.0,
        o2m_matcher=o2m_matcher,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fn_semantic_seg=None,
    )


def _attach_matcher_indices(outputs: Any, targets: list[Any], matcher: Any) -> None:
    """Attach matcher results when the model is in validation/eval mode.

    ``Sam3Image`` does this itself while ``model.training`` is true. Evaluation
    deliberately disables training behavior, so its SAM3Output dictionaries
    need the same indices before ``Sam3LossWrapper`` can score them.
    """
    for stage_outputs, target in zip(outputs.output, targets):
        for output in stage_outputs:
            output["indices"] = matcher(output, target)
            for auxiliary in output.get("aux_outputs", []):
                auxiliary["indices"] = matcher(auxiliary, target)


def _core_loss(loss_result: Any) -> Any:
    """Extract Meta's ``CORE_LOSS_KEY`` without importing SAM3 at module load."""
    return loss_result["core_loss"] if isinstance(loss_result, dict) else loss_result


def _validate_adapter_state(adapters: Any, torch_module: Any) -> None:
    """Reject corrupt, incomplete, non-finite, and mathematical no-op adapters."""

    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("SAM3 adapter state must be a non-empty mapping")
    paths: dict[str, set[str]] = {}
    tensors: dict[str, Any] = {}
    for key, tensor in adapters.items():
        if not isinstance(key, str) or not torch_module.is_tensor(tensor):
            raise ValueError("SAM3 adapter state contains a non-tensor entry")
        path, separator, suffix = key.rpartition(".")
        if not separator or suffix not in {"lora_A", "lora_B"}:
            raise ValueError(f"unexpected SAM3 adapter key {key!r}")
        if tensor.ndim != 2 or tensor.numel() == 0:
            raise ValueError(f"SAM3 adapter tensor {key!r} must be non-empty 2-D")
        if not bool(torch_module.isfinite(tensor).all().item()):
            raise ValueError(f"SAM3 adapter tensor {key!r} contains non-finite values")
        paths.setdefault(path, set()).add(suffix)
        tensors[key] = tensor
    if any(suffixes != {"lora_A", "lora_B"} for suffixes in paths.values()):
        raise ValueError("SAM3 adapter state has an incomplete LoRA A/B pair")
    for path in paths:
        matrix_a = tensors[f"{path}.lora_A"]
        matrix_b = tensors[f"{path}.lora_B"]
        if matrix_a.shape[0] != matrix_b.shape[1]:
            raise ValueError(f"SAM3 adapter pair {path!r} has incompatible rank")
    has_nonzero_delta = False
    max_delta_elements = 1_048_576
    for path in paths:
        matrix_a = tensors[f"{path}.lora_A"].float()
        matrix_b = tensors[f"{path}.lora_B"]
        if not bool(torch_module.count_nonzero(matrix_a).item()) or not bool(
            torch_module.count_nonzero(matrix_b).item()
        ):
            continue
        rows_per_chunk = max(1, max_delta_elements // max(1, matrix_a.shape[1]))
        for start in range(0, matrix_b.shape[0], rows_per_chunk):
            delta = torch_module.matmul(
                matrix_b[start : start + rows_per_chunk].float(), matrix_a
            )
            if bool(torch_module.count_nonzero(delta).item()):
                has_nonzero_delta = True
                break
        if has_nonzero_delta:
            break
    if not has_nonzero_delta:
        raise ValueError(
            "SAM3 adapter is a mathematical no-op (all LoRA deltas are zero)"
        )


def _write_validated_adapter_artifact(
    adapters: Any, artifact_path: Path, torch_module: Any
) -> None:
    """Serialize, reload, validate, and atomically promote one adapter state."""

    _validate_adapter_state(adapters, torch_module)
    temporary = artifact_path.with_name(
        f".{artifact_path.name}.{os.getpid()}.validated.tmp"
    )
    try:
        with temporary.open("wb") as artifact_file:
            torch_module.save(adapters, artifact_file)
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        reloaded = torch_module.load(temporary, map_location="cpu", weights_only=True)
        _validate_adapter_state(reloaded, torch_module)
        os.replace(temporary, artifact_path)
        directory_fd = os.open(artifact_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        write_completion_marker(artifact_path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_dataloader(spec: Any, params: Any, *, split: str) -> list:
    """Read the built COCO split into lightweight tile descriptors.

    No image is decoded, transformed, or rasterized until its epoch iterator
    reaches the corresponding batch.
    """
    return build_descriptors(spec.derived_dataset_dir, params, split, seed=spec.seed)


def _runtime_admission_refusal(torch_module: Any, params: Any) -> str | None:
    """Repeat the parent precision/hardware gate before importing SAM3."""
    prompt_error = sam3_prompt_text_error(getattr(params, "prompt", None))
    if prompt_error is not None:
        return f"SAM3 prompt {prompt_error}."
    for index, negative_prompt in enumerate(
        getattr(params, "negative_prompts", ()) or ()
    ):
        prompt_error = sam3_prompt_text_error(negative_prompt)
        if prompt_error is not None:
            return f"SAM3 negative prompt {index} {prompt_error}."
    if getattr(params, "mixed_precision", None) != "bf16":
        return (
            "SAM3 training supports only CUDA BF16; fp16/fp32 modes fail "
            "against SAM3's BF16 activation path and are disabled."
        )
    if not torch_module.cuda.is_available():
        return "SAM3 training requires a CUDA device; CPU and MPS are disabled."
    major, minor = torch_module.cuda.get_device_capability()
    if major < 8:
        return (
            "SAM3 training requires CUDA BF16 on compute capability >= 8.0; "
            f"the selected GPU reports {major}.{minor}. FP32 fallback is disabled."
        )
    if not bool(torch_module.cuda.is_bf16_supported()):
        return (
            "The selected CUDA runtime reports that BF16 operations are not "
            "supported; SAM3 training has no safe FP32 fallback."
        )
    if not any(
        bool(getattr(params, flag, False))
        for flag in (
            "adapt_vision_encoder",
            "adapt_text_encoder",
            "adapt_geometry_encoder",
            "adapt_detr_encoder",
            "adapt_detr_decoder",
            "adapt_mask_decoder",
        )
    ):
        return (
            "SAM3 training requires at least one enabled adapter scope; all "
            "adapt_* flags are disabled."
        )
    return None


def _validated_lora_trainables(
    model: Any, *, adapted_modules: int, expected_parameters: int
) -> tuple[list[Any], int]:
    """Return LoRA-only tensors or refuse estimator/runtime shape drift."""

    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected_trainable = [
        name
        for name, _parameter in trainable_named
        if not name.endswith((".lora_A", ".lora_B"))
    ]
    actual_parameters = sum(
        int(parameter.numel()) for _name, parameter in trainable_named
    )
    if (
        adapted_modules < 1
        or len(trainable_named) != 2 * adapted_modules
        or unexpected_trainable
        or actual_parameters != expected_parameters
    ):
        unexpected = ", ".join(unexpected_trainable[:5]) or "none"
        raise RuntimeError(
            "SAM3 LoRA trainable-parameter invariant failed: "
            f"adapters={adapted_modules}, trainable={len(trainable_named)}, "
            f"parameters={actual_parameters}, "
            f"expected_parameters={expected_parameters}, "
            f"unexpected={unexpected}"
        )
    return [parameter for _name, parameter in trainable_named], actual_parameters


def run_training(spec: Any, run_dir_path: Path) -> bool:
    """Run the SAM3 LoRA training loop and write `adapters.pt`.

    Returns True on a completed run that wrote the artifact, False on the
    zero-datapoint refusal (the only failure mode this function itself
    reports -- everything past this point either succeeds or raises, and an
    uncaught exception is `main()`'s cue to exit nonzero).
    """
    params = spec.sam3_params

    train_descriptors = _build_dataloader(spec, params, split="train")
    if not train_descriptors:
        emit_log(
            "Training set produced zero datapoints; refusing to report "
            "success for a run that trained nothing."
        )
        return False

    # --- Lazy, training-only imports -----------------------------------
    import torch

    refusal = _runtime_admission_refusal(torch, params)
    if refusal:
        emit_log(refusal)
        return False
    _seed_everything(spec.seed)

    # NOTE: verified against the real Meta sam3 source on the CUDA box
    # (2026-08-31): sam3/build_sam.py does not exist. The builder lives in
    # sam3/model_builder.py. Do not "correct" this back.
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    # SAM3's vision trunk MLP is inference-only as shipped (it refuses to run
    # with grad enabled, and detaches its weights). Swap in an eager,
    # differentiable equivalent before the model is built. See perflib_compat.
    if install_grad_safe_addmm_act():
        emit_log("Patched vitdet.addmm_act with a grad-safe eager equivalent.")

    device = torch.device("cuda")
    model = build_sam3_image_model(eval_mode=False)

    lora_cfg = lora_config_from_params(params)
    # The preflight estimate intentionally budgets optimizer and gradient
    # state for LoRA parameters only.  SAM3 builders do not promise frozen
    # defaults, so establish that invariant here before adapters are created.
    model.requires_grad_(False)
    n_adapted = inject_adapters(model, lora_cfg)
    emit_log(f"Injected LoRA adapters into {n_adapted} Linear modules.")
    expected_trainable_params = expected_lora_trainable_params(params)
    trainable_params, actual_trainable_params = _validated_lora_trainables(
        model,
        adapted_modules=n_adapted,
        expected_parameters=expected_trainable_params,
    )
    # Validate exact estimator parity before moving any unexpectedly large
    # upstream model drift onto VRAM. `.to(device)` remains after injection so
    # the newly created adapter tensors move with the frozen base.
    model.to(device)
    emit_log(
        "Verified LoRA-only optimizer scope: "
        f"{actual_trainable_params:,} parameters in {len(trainable_params)} "
        f"tensors across {n_adapted} adapters."
    )

    # Verified against the real Meta sam3 source on the CUDA box (2026-08-31,
    # sam3-lora env): `inspect.signature` on `Sam3LossWrapper.__init__`,
    # `Boxes.__init__`, `IABCEMdetr.__init__`, `Masks.__init__`,
    # `BinaryHungarianMatcherV2.__init__`, `BinaryOneToManyMatcher.__init__`.
    # `Sam3LossWrapper` has no zero-arg form -- `loss_fns_find` is required
    # and positional-first. Values below are translated from Meta's own
    # reference finetuning config,
    # `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`
    # (segmentation variant, since this trains masks). ONE matcher instance
    # is built and shared between the loss wrapper and the training loop's
    # own `outputs["indices"] = matcher(...)` calls below, rather than two
    # diverging matchers.
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    )
    # The image model computes matching internally in training mode. Use the
    # exact same matcher instance as the explicit validation path and loss.
    model.matcher = matcher
    loss_fn = _build_loss_wrapper(
        Sam3LossWrapper,
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                weak_loss=False,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_weight=10.0,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pos_focal=False,
                pad_n_queries=200,
                pad_scale_pos=1.0,
            ),
            Masks(
                focal_alpha=0.25,
                focal_gamma=2.0,
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                compute_aux=False,
            ),
        ],
        matcher=matcher,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
    )

    grad_accum = max(1, int(params.grad_accum))
    batch_size = max(1, int(params.batch))
    n_batches = batch_count(query_count(train_descriptors), batch_size)
    steps_per_epoch = -(-n_batches // grad_accum)  # ceil division
    total_steps = max(1, steps_per_epoch * params.epochs)
    warmup_steps = min(50, total_steps // 4)

    optimizer = torch.optim.AdamW(trainable_params, lr=params.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_cosine_with_warmup(warmup_steps, total_steps)
    )

    autocast_dtype = torch.bfloat16

    global_step = 0
    logging_steps = 10
    optimizer.zero_grad()

    for epoch in range(params.epochs):
        model.train()
        # Tile descriptors reshuffle every epoch (seeded from spec.seed +
        # epoch, so runs stay reproducible). Queries remain tile-grouped to
        # share one transformed image without a dataset-sized tensor cache.
        epoch_batches = collate_epoch_batches(
            train_descriptors, batch_size, seed=spec.seed + epoch
        )
        n_epoch_batches = n_batches
        for micro_idx, batch in enumerate(epoch_batches):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
                model_input, targets, outputs = _forward_batch(batch, model, device)
                loss_dict = loss_fn(outputs, targets)
                loss = _core_loss(loss_dict)

            (loss / grad_accum).backward()

            is_boundary = (micro_idx + 1) % grad_accum == 0
            is_final_micro_batch = (micro_idx + 1) == n_epoch_batches
            if is_boundary or is_final_micro_batch:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % logging_steps == 0:
                    emit_log(f"epoch {epoch} step {global_step} loss {float(loss):.4f}")
            del batch, model_input, targets, outputs, loss_dict, loss

        emit_progress(epoch + 1, params.epochs)

    # Keep adapters in memory until validation completes. A failed evaluation,
    # kill, or parent death must not expose a seemingly completed artifact.
    adapters = adapter_state_dict(model)
    _evaluate_and_write(
        model,
        spec,
        params,
        matcher,
        loss_fn,
        device,
        autocast_dtype,
        True,
        run_dir_path,
    )
    artifact_path = run_dir_path / "adapters.pt"
    _write_validated_adapter_artifact(adapters, artifact_path, torch)
    return True


def _evaluate_and_write(
    model: Any,
    spec: Any,
    params: Any,
    matcher: Any,
    loss_fn: Any,
    device: Any,
    autocast_dtype: Any,
    use_bf16: bool,
    run_dir_path: Path,
) -> Path | None:
    """Compute real validation-set loss, for reporting ONLY.

    Never influences checkpoint selection (see module docstring) -- this runs
    strictly after the `adapters.pt` save above. If there is no validation
    split (small datasets skip it -- see `dataset_build.py`'s `validation:
    "none"` case), no file is written and `None` is returned rather than
    fabricating a placeholder.
    """
    import torch

    val_descriptors = try_build_descriptors(
        spec.derived_dataset_dir, params, "valid", seed=spec.seed
    )
    if not val_descriptors:
        return None
    n_val_batches = batch_count(query_count(val_descriptors), params.batch)
    val_batches = collate_batches(val_descriptors, params.batch)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_batches:
            with torch.autocast(
                device_type="cuda", dtype=autocast_dtype, enabled=use_bf16
            ):
                model_input, targets, outputs = _forward_batch(batch, model, device)
                _attach_matcher_indices(outputs, targets, matcher)
                loss_dict = loss_fn(outputs, targets)
                loss = _core_loss(loss_dict)
            total_loss += float(loss)
            del batch, model_input, targets, outputs, loss_dict, loss

    val_stats = {
        "val_loss_mean": total_loss / n_val_batches,
        "val_batches": n_val_batches,
        "note": "informational only; checkpoint selection is always 'last'",
    }
    metrics_path = run_dir_path / "val_stats.json"
    metrics_path.write_text(json.dumps(val_stats, indent=2), encoding="utf-8")
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", required=True, help="Path to the serialised spec.json"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Run directory to write artifacts into"
    )
    args = parser.parse_args(argv)

    run_dir_path = Path(args.run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(Path(args.spec).expanduser().resolve())

    ok = run_training(spec, run_dir_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
