from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hydra_suite.core.canonicalization.crop import (
    compute_native_crop_dimensions,
    compute_alignment_affine,
    gpu_canonical_crop_batch,
)

from ..result import OBBResult
from ..runtime import RuntimeContext


def extract_canonical_crops(
    frame: np.ndarray | torch.Tensor,
    obb_result: OBBResult,
    canonical_aspect_ratio: float,
    canonical_margin: float,
    runtime: RuntimeContext,
) -> torch.Tensor:
    """Extract OBB-aligned canonical crops. Returns (N, C, H, W) tensor on runtime.device.

    GPU path (tensor_on_cuda only): single batched affine_grid + grid_sample call.
    CPU path: cv2.warpAffine per crop -> stacked CPU tensor.
    onnx_cuda/tensorrt use CPU path even though cuda_mode=True; their downstream
    models take CPU numpy, so GPU crop upload+download would be pure waste.
    """
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros((0, 3, 64, 64), dtype=torch.float32)

    if runtime.tensor_on_cuda:
        return _extract_canonical_gpu(
            frame,
            obb_result,
            canonical_aspect_ratio,
            canonical_margin,
            runtime.device,
        )
    return _extract_canonical_cpu(
        frame, obb_result, canonical_aspect_ratio, canonical_margin
    )


def extract_aabb_crops(
    frame: np.ndarray,
    obb_result: OBBResult,
    padding: float,
) -> list[np.ndarray]:
    """Extract axis-aligned bounding box crops for AprilTag detection.

    Always CPU numpy. frame must be a numpy array (already .cpu().numpy() on CUDA path).
    """
    if obb_result.num_detections == 0:
        return []
    h, w = frame.shape[:2]
    crops: list[np.ndarray] = []
    for i in range(obb_result.num_detections):
        corners = obb_result.corners[i]
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        bw, bh = x2 - x1, y2 - y1
        pad = padding * max(bw, bh)
        ox1 = max(0, int(x1 - pad))
        oy1 = max(0, int(y1 - pad))
        ox2 = min(w, int(x2 + pad))
        oy2 = min(h, int(y2 + pad))
        crop = frame[oy1:oy2, ox1:ox2]
        crops.append(crop if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8))
    return crops


def _extract_canonical_cpu(
    frame: np.ndarray | torch.Tensor,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
    else:
        arr = frame

    padding_fraction = max(0.0, float(margin) - 1.0)
    crops: list[np.ndarray] = []
    for i in range(obb.num_detections):
        crop = _warp_canonical_crop(
            arr,
            obb.corners[i],
            aspect_ratio,
            padding_fraction,
        )
        crops.append(crop)

    max_h = max(c.shape[0] for c in crops)
    max_w = max(c.shape[1] for c in crops)
    padded: list[np.ndarray] = []
    for c in crops:
        if c.shape[0] == max_h and c.shape[1] == max_w:
            padded.append(c)
        else:
            pad_h = max_h - c.shape[0]
            pad_w = max_w - c.shape[1]
            padded.append(np.pad(c, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"))

    stacked = np.stack(padded, axis=0)  # (N, H, W, C)
    t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    return t


def _warp_canonical_crop(
    frame: np.ndarray,
    corners: np.ndarray,
    aspect_ratio: float,
    padding_fraction: float,
) -> np.ndarray:
    """Extract canonical crop using main's corner-triangle affine.

    Delegates to ``compute_native_crop_dimensions`` + ``compute_alignment_affine``
    from ``core.canonicalization.crop`` so the new pipeline produces canvases
    with native major-axis pixel extent matching main exactly.
    """
    canvas_w, canvas_h = compute_native_crop_dimensions(
        corners, aspect_ratio, padding_fraction
    )
    try:
        M, _ = compute_alignment_affine(
            corners, canvas_w, canvas_h, padding_fraction
        )
    except ValueError:
        n_ch = frame.shape[2] if frame.ndim == 3 else 1
        return np.zeros((canvas_h, canvas_w, n_ch), dtype=frame.dtype)
    return cv2.warpAffine(
        frame,
        M,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _extract_canonical_gpu(
    frame: torch.Tensor | np.ndarray,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
    device: str,
) -> torch.Tensor:
    """Batched corner-affine crop extraction on CUDA via gpu_canonical_crop_batch.

    Mirrors main's headtail GPU path: each detection's M_align is computed via
    ``compute_alignment_affine`` (corner-triangle), then a single batched
    ``F.affine_grid`` + ``F.grid_sample`` warp produces all crops at a uniform
    canvas size = max(native_dims) so smaller OBBs are border-replicated.
    """
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3:
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        frame = frame.to(device)

    if frame.ndim == 4:
        frame = frame.squeeze(0)  # (C, H, W) — gpu_canonical_crop_batch expects CHW

    n = obb.num_detections
    padding_fraction = max(0.0, float(margin) - 1.0)

    canvas_dims: list[tuple[int, int]] = []
    M_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            cw, ch = compute_native_crop_dimensions(
                obb.corners[i], aspect_ratio, padding_fraction
            )
            M, _ = compute_alignment_affine(
                obb.corners[i], cw, ch, padding_fraction
            )
        except ValueError:
            cw, ch = 8, 8
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        canvas_dims.append((cw, ch))
        M_aligns.append(M)

    out_w = max(cd[0] for cd in canvas_dims) if canvas_dims else 8
    out_h = max(cd[1] for cd in canvas_dims) if canvas_dims else 8

    crops = gpu_canonical_crop_batch(frame, M_aligns, out_w, out_h)
    return crops


# Legacy GPU path kept for reference (no longer used).
def _extract_canonical_gpu_legacy(
    frame: torch.Tensor | np.ndarray,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
    device: str,
) -> torch.Tensor:
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3:
            frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        frame = frame.to(device)

    if frame.ndim == 3:
        frame = frame.unsqueeze(0)

    _, C, H, W = frame.shape
    n = obb.num_detections

    sides = [math.sqrt(float(obb.sizes[i])) * margin for i in range(n)]
    out_ws = [max(int(s * aspect_ratio), 4) for s in sides]
    out_hs = [max(int(s), 4) for s in sides]
    out_w = max(out_ws)
    out_h = max(out_hs)

    thetas = []
    for i in range(n):
        cx = float(obb.centroids[i, 0])
        cy = float(obb.centroids[i, 1])
        angle = float(obb.angles[i])
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        ncx = 2.0 * cx / W - 1.0
        ncy = 2.0 * cy / H - 1.0
        thetas.append(
            [
                [cos_a * (out_w / W), -sin_a * (out_h / W), ncx],
                [sin_a * (out_w / H), cos_a * (out_h / H), ncy],
            ]
        )

    theta_t = torch.tensor(thetas, dtype=torch.float32, device=device)
    frame_batch = frame.expand(n, -1, -1, -1)
    grid = F.affine_grid(theta_t, (n, C, out_h, out_w), align_corners=False)
    crops = F.grid_sample(
        frame_batch,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return crops
