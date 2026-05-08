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

    # axis_theta for the head-tail offset map is derived empirically from the
    # OBB corners with atan2(c[1]-c[0]). Reasoning: the new pipeline's crop
    # builder rotates by `obb.angles` (folded to [0, pi)), but YOLO's native
    # xyxyxyxy corners are in the *un-folded* axis ordering, so the canonical
    # crop the classifier sees is rotated 180 degrees relative to the legacy
    # detector's crop for ~94 percent of detections. Using atan2 of YOLO's
    # native corners as axis_theta cancels that flip and recovers legacy parity
    # for those detections. Falls back to obb.angles if corners are degenerate.
    signed_axes = _signed_major_axis_from_corners(obb_result.corners)

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
        axis_theta = float(obb_result.angles[i])
        if signed_axes is not None and math.isfinite(float(signed_axes[i])):
            axis_theta = float(signed_axes[i])
        hints[i] = (axis_theta + offset) % (2.0 * math.pi)
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


def _signed_major_axis_from_corners(corners: np.ndarray) -> np.ndarray | None:
    """Return per-detection major-axis angle from OBB corners, in [-pi, pi].

    Mirrors ``compute_alignment_affine`` in core.canonicalization.crop: picks
    the longer of the first two edges (c[1]-c[0] vs c[2]-c[1]) and returns
    ``atan2(major_vec_y, major_vec_x)``. NaN for degenerate boxes.
    """
    if corners is None or corners.size == 0 or corners.ndim != 3:
        return None
    n = corners.shape[0]
    out = np.full(n, float("nan"), dtype=np.float32)
    for i in range(n):
        c = corners[i].reshape(4, 2)
        e01 = float(np.linalg.norm(c[1] - c[0]))
        e12 = float(np.linalg.norm(c[2] - c[1]))
        if e01 < 1e-3 or e12 < 1e-3:
            continue
        major_vec = c[1] - c[0] if e01 >= e12 else c[2] - c[1]
        out[i] = float(math.atan2(float(major_vec[1]), float(major_vec[0])))
    return out


def _resize_crops(crops: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Resize (N, C, H, W) tensor to (N, C, target_H, target_W)."""
    th, tw = target_size
    if crops.shape[2] == th and crops.shape[3] == tw:
        return crops
    return F.interpolate(crops, size=(th, tw), mode="bilinear", align_corners=False)
