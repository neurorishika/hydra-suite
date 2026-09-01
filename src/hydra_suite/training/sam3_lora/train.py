"""SAM3 LoRA training loop.

Qt-free. ``sam3`` is training-only and not installed on every machine that
imports `hydra_suite.training`; every reference to it lives inside function
bodies so this module (and the dispatch that lazily imports it) loads
cleanly without the package.

Checkpoint selection is always the LAST epoch's weights, never the epoch
with the best validation loss: the spike found val loss anti-correlated with
held-out AP (the fold with the worst val loss had the best held-out AP75).
Do not add best-checkpoint selection or early stopping on val loss here.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

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

    refusals = preflight(spec)
    if refusals:
        return {
            "success": False,
            "error_message": "; ".join(refusals),
            "artifact_path": None,
            "metrics_path": None,
            "canceled": False,
        }

    run_dir_path = Path(run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    params = spec.sam3_params
    _seed_everything(spec.seed)

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

    # --- Data ------------------------------------------------------------
    train_batches = _build_dataloader(spec, params, split="train")
    total_steps = max(1, len(train_batches) * params.epochs)
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

    for epoch in range(params.epochs):
        if should_cancel():
            canceled = True
            break
        model.train()
        for batch in train_batches:
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if global_step % logging_steps == 0:
                log_cb(f"epoch {epoch} step {global_step} loss {float(loss):.4f}")
            global_step += 1

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

    val_stats = _evaluate(model, spec, params)
    metrics_path = run_dir_path / "val_stats.json"
    metrics_path.write_text(json.dumps(val_stats, indent=2), encoding="utf-8")

    return {
        "success": True,
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "canceled": False,
    }


def _cosine_with_warmup(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    import math

    def _fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _fn


def _build_dataloader(spec: Any, params: Any, *, split: str) -> list:
    """Build the batched Datapoint list for `split` from the built COCO dataset.

    Kept separate so it can be replaced/mocked without touching the training
    loop's control flow.
    """
    from .datapoints import collate_datapoints

    del spec, params, split, collate_datapoints  # placeholder wiring point
    return []


def _targets_from_batch(batch: Any) -> Any:
    return batch.get("targets") if isinstance(batch, dict) else batch


def _evaluate(model: Any, spec: Any, params: Any) -> dict:
    """Compute validation-set stats for reporting ONLY.

    These numbers must never feed checkpoint selection (see module docstring)
    -- they are written for the user/report, not consulted here to pick a
    different epoch's weights.
    """
    del model, spec, params
    return {"note": "validation stats are informational only; checkpoint is 'last'"}
