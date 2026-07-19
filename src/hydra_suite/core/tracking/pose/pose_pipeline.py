"""Pose crop-extraction and letterbox utilities.

Shared helpers used by the crops worker's pose-precompute path
(:func:`extract_one_crop`, :func:`letterbox_crop`,
:func:`invert_letterbox_keypoints`) and their supporting data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CropTransform:
    """Letterbox transform applied to one crop (for inverse coordinate mapping)."""

    scale: float = 1.0
    pad_x: int = 0
    pad_y: int = 0


@dataclass
class FrameCropResult:
    """Aggregated crop-extraction output for a single video frame."""

    frame_idx: int
    det_ids: List[int]
    n_dets: int
    crops: list  # List[np.ndarray] for CPU path; List[Tensor] for CUDA path
    crop_to_det: List[int]
    crop_offsets: Dict[int, Tuple[int, int]]
    all_obb_corners: List[np.ndarray]
    crop_transforms: Dict[int, CropTransform] = field(default_factory=dict)
    crop_M_inverses: Dict[int, np.ndarray] = field(default_factory=dict)


def _require_detection_index(det_idx: int, n_dets: int) -> int:
    """Return a validated detection-slot index for per-frame lists."""
    if not isinstance(det_idx, (int, np.integer)):
        raise TypeError(
            f"Detection slot index must be an integer, got {type(det_idx).__name__}"
        )
    normalized = int(det_idx)
    if normalized < 0 or normalized >= int(n_dets):
        raise IndexError(
            f"Detection slot index {normalized} out of range for {int(n_dets)} detections"
        )
    return normalized


# ---------------------------------------------------------------------------
# Crop extraction helpers  (thread-safe — read-only on *frame*)
# ---------------------------------------------------------------------------


def _expand_obb_to_aabb(
    corners: np.ndarray,
    padding_fraction: float,
    frame_h: int,
    frame_w: int,
) -> Tuple[int, int, int, int]:
    """Expand OBB corners and return axis-aligned bounding box ``(x0, y0, x1, y1)``."""
    centroid = corners.mean(axis=0)
    expanded = corners.copy()
    for i in range(4):
        direction = corners[i] - centroid
        expanded[i] = centroid + direction * (1.0 + padding_fraction)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, frame_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, frame_h - 1)
    x0 = max(0, int(np.floor(expanded[:, 0].min())))
    x1 = min(frame_w, int(np.ceil(expanded[:, 0].max())) + 1)
    y0 = max(0, int(np.floor(expanded[:, 1].min())))
    y1 = min(frame_h, int(np.ceil(expanded[:, 1].max())) + 1)
    return x0, y0, x1, y1


def extract_one_crop(
    frame: np.ndarray,
    corners: np.ndarray,
    det_idx: int,
    padding_fraction: float,
    all_obb_corners: List[np.ndarray],
    suppress_foreign: bool,
    bg_color: Tuple[int, int, int],
) -> Optional[Tuple[np.ndarray, Tuple[int, int], int]]:
    """Extract a single crop from *frame*.

    Returns ``(crop, (x0, y0), det_idx)`` on success, or ``None`` when the
    detection cannot produce a valid crop.

    This function only *reads* from *frame* so it is safe to call from
    multiple threads concurrently.
    """
    if frame is None or corners is None or corners.shape[0] < 4:
        return None

    frame_h, frame_w = frame.shape[:2]
    x0, y0, x1, y1 = _expand_obb_to_aabb(corners, padding_fraction, frame_h, frame_w)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None

    if suppress_foreign and len(all_obb_corners) > 1:
        from hydra_suite.utils.geometry import apply_foreign_obb_mask

        other = [
            all_obb_corners[j] for j in range(len(all_obb_corners)) if j != det_idx
        ]
        crop = apply_foreign_obb_mask(crop, x0, y0, other, background_color=bg_color)

    return crop, (x0, y0), det_idx


# ---------------------------------------------------------------------------
# Letterbox helpers
# ---------------------------------------------------------------------------


def letterbox_crop(
    crop: np.ndarray,
    target_size: int,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, CropTransform]:
    """Resize *crop* so its longest edge fits *target_size*, pad to square.

    Only down-scales; crops that already fit are padded without up-scaling.

    Returns ``(letterboxed_image, CropTransform)``.
    """
    h, w = crop.shape[:2]
    scale = min(target_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    if new_w == target_size and new_h == target_size:
        return crop, CropTransform(scale=scale, pad_x=0, pad_y=0)

    n_channels = crop.shape[2] if crop.ndim == 3 else 1
    if crop.ndim == 3:
        canvas = np.full(
            (target_size, target_size, n_channels),
            bg_color[:n_channels],
            dtype=np.uint8,
        )
    else:
        canvas = np.full((target_size, target_size), bg_color[0], dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = crop
    return canvas, CropTransform(scale=scale, pad_x=pad_x, pad_y=pad_y)


def invert_letterbox_keypoints(
    kpts: np.ndarray, transform: CropTransform
) -> np.ndarray:
    """Map keypoints from letterboxed space back to original crop space."""
    out = kpts.copy()
    out[:, 0] = (out[:, 0] - transform.pad_x) / max(transform.scale, 1e-9)
    out[:, 1] = (out[:, 1] - transform.pad_y) / max(transform.scale, 1e-9)
    return out
