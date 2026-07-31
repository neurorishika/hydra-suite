"""Pure crop -> keypoints composition for ViTPose. Runtime-agnostic: the caller
injects a forward function (torch module, ONNX session, TRT engine, or CoreML).

PoseKit-free leaf module.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import HEATMAP_SIZE_WH
from .decode import decode_udp_torch
from .transforms import box2cs, normalize, top_down_affine, transform_preds


def preprocess_crop(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A crop (already the animal's bbox) -> (CHW float32, center, scale).

    The crop's own extent is the box; box2cs applies PADDING_FACTOR + aspect fix.
    """
    h, w = crop_bgr.shape[:2]
    box_xywh = np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
    center, scale = box2cs(box_xywh)
    warped = top_down_affine(crop_bgr, center, scale, rot=0.0)
    chw = normalize(warped)
    return chw, center, scale


def decode_and_project(
    heatmaps: torch.Tensor,
    centers: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode heatmaps on-device, then project each set back to image coords.

    heatmaps: (B, K, 64, 48). centers/scales: (B, 2). Returns
    coords (B, K, 2) in image space and maxvals (B, K, 1).
    """
    coords_t, maxvals_t = decode_udp_torch(heatmaps)  # on heatmaps.device
    coords = coords_t.detach().cpu().numpy()
    maxvals = maxvals_t.detach().cpu().numpy()
    out = np.empty_like(coords)
    for i in range(coords.shape[0]):
        out[i] = transform_preds(coords[i], centers[i], scales[i], HEATMAP_SIZE_WH)
    return out, maxvals
