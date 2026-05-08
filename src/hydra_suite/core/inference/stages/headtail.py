from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import HeadTailConfig
from ..result import HeadTailResult, OBBResult
from ..runtime import RuntimeContext

_DIRECTION_OFFSET: dict[str, float] = {
    "right": 0.0,
    "left": math.pi,
    "up": -math.pi / 2,
    "down": math.pi / 2,
}


@dataclass
class HeadTailModel:
    backend: Any  # ClassifierBackend instance
    input_size: tuple[int, int]  # (H, W) expected by the model
    class_names: list[str]

    def close(self) -> None:
        pass


def load_headtail_model(
    config: HeadTailConfig, runtime: RuntimeContext
) -> HeadTailModel:
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(config.model_path, config.compute_runtime)
    meta = backend.metadata
    input_size = (meta.input_size[0], meta.input_size[1])
    return HeadTailModel(
        backend=backend,
        input_size=input_size,
        class_names=list(meta.class_names_per_factor[0]),
    )


def run_headtail(
    crops: torch.Tensor,
    obb_result: OBBResult,
    model: HeadTailModel,
    config: HeadTailConfig,
    runtime: RuntimeContext,
) -> HeadTailResult:
    """Classify head-tail orientation per crop. No I/O, no mode branching.

    Per Correction 15: canonical_affines is None — affines belong to the crops
    stage, not headtail. Downstream consumers must check for None and recompute
    if needed.
    """
    n = obb_result.num_detections
    hints = np.full(n, float("nan"), dtype=np.float32)
    confs = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.uint8)

    if crops.shape[0] == 0 or n == 0:
        return HeadTailResult(
            heading_hints=hints,
            heading_confidences=confs,
            directed_mask=mask,
            canonical_affines=None,
        )

    resized = _resize_crops(crops, model.input_size)
    np_crops = [
        resized[i].permute(1, 2, 0).cpu().numpy() for i in range(resized.shape[0])
    ]

    all_probs = model.backend.predict_batch(np_crops)

    for i, probs_per_factor in enumerate(all_probs):
        factor_probs = probs_per_factor[0]
        winning_idx = int(np.argmax(factor_probs))
        winning_conf = float(factor_probs[winning_idx])
        if winning_conf < config.confidence_threshold:
            continue
        label = model.class_names[winning_idx]
        offset = _label_to_heading_offset(label)
        if offset is None:
            continue
        hints[i] = obb_result.angles[i] + offset
        confs[i] = winning_conf
        mask[i] = 1

    return HeadTailResult(
        heading_hints=hints,
        heading_confidences=confs,
        directed_mask=mask,
        canonical_affines=None,
    )


def _label_to_heading_offset(label: str) -> float | None:
    """Map direction label to angle offset relative to OBB major axis."""
    return _DIRECTION_OFFSET.get(label)


def _resize_crops(crops: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize (N, C, H, W) tensor to (N, C, target_H, target_W)."""
    th, tw = target_size
    if crops.shape[2] == th and crops.shape[3] == tw:
        return crops
    return F.interpolate(crops, size=(th, tw), mode="bilinear", align_corners=False)
