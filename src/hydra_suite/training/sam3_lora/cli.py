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
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams

from .dataloader import (
    build_datapoints,
    collate_batches,
    collate_epoch_batches,
    try_build_datapoints,
)
from .lora import adapter_state_dict, inject_adapters, lora_config_from_params
from .protocol import emit_log, emit_progress


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


def _targets_from_batch(batch: Any) -> Any:
    return batch.get("targets") if isinstance(batch, dict) else batch


def _build_dataloader(spec: Any, params: Any, *, split: str) -> list:
    """Read the built COCO split into its (unshuffled) Datapoints.

    Never returns `[]` itself -- it raises instead; the real emptiness
    refusal lives in `run_training` below. Per-epoch shuffling into batches
    happens later, inside the training loop, via `collate_epoch_batches`.
    """
    return build_datapoints(spec.derived_dataset_dir, params, split, seed=spec.seed)


def run_training(spec: Any, run_dir_path: Path) -> bool:
    """Run the SAM3 LoRA training loop and write `adapters.pt`.

    Returns True on a completed run that wrote the artifact, False on the
    zero-datapoint refusal (the only failure mode this function itself
    reports -- everything past this point either succeeds or raises, and an
    uncaught exception is `main()`'s cue to exit nonzero).
    """
    params = spec.sam3_params
    _seed_everything(spec.seed)

    train_datapoints = _build_dataloader(spec, params, split="train")
    if not train_datapoints:
        emit_log(
            "Training set produced zero datapoints; refusing to report "
            "success for a run that trained nothing."
        )
        return False

    # --- Lazy, training-only imports -----------------------------------
    import torch

    # NOTE: verified against the real Meta sam3 source on the CUDA box
    # (2026-08-31): sam3/build_sam.py does not exist. The builder lives in
    # sam3/model_builder.py. Do not "correct" this back.
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2

    device = torch.device("cuda")
    model = build_sam3_image_model(eval_mode=False)
    model.to(device)

    lora_cfg = lora_config_from_params(params)
    n_adapted = inject_adapters(model, lora_cfg)
    emit_log(f"Injected LoRA adapters into {n_adapted} Linear modules.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    matcher = BinaryHungarianMatcherV2(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
    )
    loss_fn = Sam3LossWrapper()

    grad_accum = max(1, int(params.grad_accum))
    batch_size = max(1, int(params.batch))
    n_batches = -(-len(train_datapoints) // batch_size)  # ceil division
    steps_per_epoch = -(-n_batches // grad_accum)  # ceil division
    total_steps = max(1, steps_per_epoch * params.epochs)
    warmup_steps = min(50, total_steps // 4)

    optimizer = torch.optim.AdamW(trainable_params, lr=params.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_cosine_with_warmup(warmup_steps, total_steps)
    )

    major, _minor = torch.cuda.get_device_capability()
    use_bf16 = major >= 8 and params.mixed_precision == "bf16"
    if params.mixed_precision == "bf16" and not use_bf16:
        emit_log(
            "GPU compute capability < 8.0; falling back to fp32 "
            "(bf16 autocast is not supported)."
        )
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    global_step = 0
    logging_steps = 10
    optimizer.zero_grad()

    for epoch in range(params.epochs):
        model.train()
        # Reshuffled every epoch (seeded from spec.seed + epoch, so runs stay
        # reproducible) -- a fixed order would put each tile's negatives in
        # the same accumulation window as its positive on every epoch.
        epoch_batches = collate_epoch_batches(
            train_datapoints, batch_size, seed=spec.seed + epoch
        )
        n_epoch_batches = len(epoch_batches)
        for micro_idx, batch in enumerate(epoch_batches):
            targets = _targets_from_batch(batch)
            with torch.autocast(
                device_type="cuda", dtype=autocast_dtype, enabled=use_bf16
            ):
                outputs = model(batch)
                outputs["indices"] = matcher(outputs, targets)
                if "aux_outputs" in outputs:
                    for aux in outputs["aux_outputs"]:
                        aux["indices"] = matcher(aux, targets)
                loss_dict = loss_fn(outputs, targets)
                loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict

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

        emit_progress(epoch + 1, params.epochs)

    # --- Persist artifacts (LAST checkpoint, never best-on-val-loss) -----
    artifact_path = run_dir_path / "adapters.pt"
    torch.save(adapter_state_dict(model), artifact_path)

    _evaluate_and_write(
        model, spec, params, matcher, loss_fn, autocast_dtype, use_bf16, run_dir_path
    )
    return True


def _evaluate_and_write(
    model: Any,
    spec: Any,
    params: Any,
    matcher: Any,
    loss_fn: Any,
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

    val_datapoints = try_build_datapoints(
        spec.derived_dataset_dir, params, "valid", seed=spec.seed
    )
    if not val_datapoints:
        return None
    val_batches = collate_batches(val_datapoints, params.batch)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_batches:
            targets = _targets_from_batch(batch)
            with torch.autocast(
                device_type="cuda", dtype=autocast_dtype, enabled=use_bf16
            ):
                outputs = model(batch)
                outputs["indices"] = matcher(outputs, targets)
                if "aux_outputs" in outputs:
                    for aux in outputs["aux_outputs"]:
                        aux["indices"] = matcher(aux, targets)
                loss_dict = loss_fn(outputs, targets)
                loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
            total_loss += float(loss)

    val_stats = {
        "val_loss_mean": total_loss / len(val_batches),
        "val_batches": len(val_batches),
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
