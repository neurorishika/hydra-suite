"""SAM3 LoRA training loop.

Qt-free. ``sam3`` is training-only and not installed on every machine that
imports `hydra_suite.training`; every reference to it lives inside function
bodies so this module (and the dispatch that lazily imports it) loads
cleanly without the package.

Checkpoint selection is always the LAST epoch's weights, never the epoch
with the best validation loss: the spike found val loss anti-correlated with
held-out AP (the fold with the worst val loss had the best held-out AP75).
Do not add best-checkpoint selection or early stopping on val loss here.

The training set is built and checked for emptiness BEFORE any `sam3` model
is loaded: an empty dataloader must return `success: False`, never silently
train nothing and report success (zero-initialised LoRA `lora_B` makes an
untrained adapter a mathematical no-op, so a fake-success run would publish
a "finetuned" checkpoint byte-identical to stock SAM3).
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .dataloader import (
    build_datapoints,
    collate_batches,
    collate_epoch_batches,
    try_build_datapoints,
)
from .lora import adapter_state_dict, inject_adapters, lora_config_from_params
from .preflight import preflight

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_with_warmup(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def _fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        span = max(1, total_steps - warmup_steps)
        progress = min(max(float(step - warmup_steps) / float(span), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _fn


def _targets_from_batch(batch: Any) -> Any:
    return batch.get("targets") if isinstance(batch, dict) else batch


def train_sam3_lora(
    spec: Any,
    run_dir: str,
    *,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Finetune SAM3 with LoRA adapters against `spec`.

    Returns a dict with keys `success`, `artifact_path`, `metrics_path`,
    `canceled` (and `error_message` on refusal/failure).
    """
    log_cb = log_cb or (lambda msg: None)
    progress_cb = progress_cb or (lambda epoch, total: None)
    should_cancel = should_cancel or (lambda: False)

    def _refuse(message: str) -> dict:
        return {
            "success": False,
            "error_message": message,
            "artifact_path": None,
            "metrics_path": None,
            "canceled": False,
        }

    refusals = preflight(spec)
    if refusals:
        return _refuse("; ".join(refusals))

    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    params = spec.sam3_params
    _seed_everything(spec.seed)

    # Build (and validate) the training set BEFORE touching `sam3` or the GPU.
    train_datapoints = _build_dataloader(spec, params, split="train")
    if not train_datapoints:
        return _refuse(
            "Training set produced zero datapoints; refusing to report "
            "success for a run that trained nothing."
        )

    # --- Lazy, training-only imports -----------------------------------
    import torch
    from sam3.build_sam import build_sam3_image_model
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2

    device = torch.device("cuda")
    model = build_sam3_image_model(eval_mode=False)
    model.to(device)

    lora_cfg = lora_config_from_params(params)
    n_adapted = inject_adapters(model, lora_cfg)
    log_cb(f"Injected LoRA adapters into {n_adapted} Linear modules.")

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
        log_cb(
            "GPU compute capability < 8.0; falling back to fp32 "
            "(bf16 autocast is not supported)."
        )
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    global_step = 0
    logging_steps = 10
    canceled = False
    optimizer.zero_grad()

    for epoch in range(params.epochs):
        if should_cancel():
            canceled = True
            break
        model.train()
        # Reshuffled every epoch (seeded from spec.seed + epoch, so runs stay
        # reproducible) -- a fixed order would put each tile's negatives in
        # the same accumulation window as its positive on every epoch.
        epoch_batches = collate_epoch_batches(
            train_datapoints, batch_size, seed=spec.seed + epoch
        )
        n_epoch_batches = len(epoch_batches)
        for micro_idx, batch in enumerate(epoch_batches):
            if should_cancel():
                canceled = True
                break

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
                    log_cb(f"epoch {epoch} step {global_step} loss {float(loss):.4f}")

        if canceled:
            break
        progress_cb(epoch + 1, params.epochs)

    if canceled:
        return {
            "success": False,
            "canceled": True,
            "artifact_path": None,
            "metrics_path": None,
        }

    # --- Persist artifacts (LAST checkpoint, never best-on-val-loss) -----
    artifact_path = run_dir_path / "adapters.pt"
    torch.save(adapter_state_dict(model), artifact_path)

    metrics_path = _evaluate_and_write(
        model, spec, params, matcher, loss_fn, autocast_dtype, use_bf16, run_dir_path
    )

    return {
        "success": True,
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "canceled": False,
    }


def _build_dataloader(spec: Any, params: Any, *, split: str) -> list:
    """Read the built COCO split into its (unshuffled) Datapoints. Kept as a
    thin, separately-mockable seam so tests can force a zero-datapoint run
    without a real dataset or `sam3` install; the real implementation lives
    in `dataloader.py` and never returns `[]` itself -- it raises instead.
    Per-epoch shuffling into batches happens later, inside the training
    loop, via `collate_epoch_batches`.
    """
    return build_datapoints(spec.derived_dataset_dir, params, split, seed=spec.seed)


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
