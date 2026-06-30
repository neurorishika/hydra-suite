from __future__ import annotations

import cv2
import numpy as np
import torch

from hydra_suite.core.canonicalization.crop import (
    compute_alignment_affine,
    compute_native_crop_dimensions,
    gpu_canonical_crop_batch,
)

from ..result import CropBatch, OBBResult
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


def _frame_as_hwc_numpy(frame: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a frame (numpy HWC or torch CHW/HWC) to a HWC uint8/float numpy array."""
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr
    return frame


def _warp_crops_for_obb(
    arr: np.ndarray,
    obb: OBBResult,
    aspect_ratio: float,
    padding_fraction: float,
) -> list[np.ndarray]:
    """Warp each detection in *obb* to its native canonical extent.

    Returns a list of HWC numpy arrays, one per detection, at the native
    (un-resized) crop size produced by :func:`_warp_canonical_crop`.
    """
    crops: list[np.ndarray] = []
    for i in range(obb.num_detections):
        crop = _warp_canonical_crop(arr, obb.corners[i], aspect_ratio, padding_fraction)
        crops.append(crop)
    return crops


def _extract_canonical_cpu(
    frame: np.ndarray | torch.Tensor,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
) -> torch.Tensor:
    arr = _frame_as_hwc_numpy(frame)
    padding_fraction = max(0.0, float(margin) - 1.0)
    crops = _warp_crops_for_obb(arr, obb, aspect_ratio, padding_fraction)

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


def extract_classifier_crops(
    frame: np.ndarray | torch.Tensor,
    obb_result: OBBResult,
    target_size: tuple[int, int],
    aspect_ratio: float,
    margin: float,
) -> list[np.ndarray]:
    """Warp each OBB directly to the classifier's input size (BGR uint8).

    Bit-identical to the legacy head-tail / CNN crop path
    (HeadTailAnalyzer._canonicalize_obb + extract_canonical_crop): a SINGLE
    ``cv2.warpAffine`` maps the padded OBB straight to (target_w, target_h) with
    INTER_LINEAR + BORDER_REPLICATE. This avoids the double resample of going
    through the shared native-extent crop tensor + a second (torch) interpolate,
    which left ~1-2% of head-tail direction decisions flipping vs legacy near the
    classifier's decision boundary. ``target_size`` is the model's (out_w, out_h)
    using legacy's index convention (input_size[0], input_size[1]).
    """
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = (
                (arr * 255.0).clip(0, 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.astype(np.uint8)
            )
    else:
        arr = frame
    out_w, out_h = int(target_size[0]), int(target_size[1])
    pad = max(0.0, float(margin) - 1.0)
    n_ch = arr.shape[2] if arr.ndim == 3 else 1
    crops: list[np.ndarray] = []
    for i in range(obb_result.num_detections):
        corners = obb_result.corners[i]
        try:
            m_align, _ = compute_alignment_affine(corners, out_w, out_h, pad)
        except ValueError:
            crops.append(np.zeros((out_h, out_w, n_ch), dtype=np.uint8))
            continue
        crop = cv2.warpAffine(
            arr,
            m_align,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        crops.append(np.ascontiguousarray(crop))
    return crops


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
        M, _ = compute_alignment_affine(corners, canvas_w, canvas_h, padding_fraction)
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
            M, _ = compute_alignment_affine(obb.corners[i], cw, ch, padding_fraction)
        except ValueError:
            cw, ch = 8, 8
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        canvas_dims.append((cw, ch))
        M_aligns.append(M)

    out_w = max(cd[0] for cd in canvas_dims) if canvas_dims else 8
    out_h = max(cd[1] for cd in canvas_dims) if canvas_dims else 8

    crops = gpu_canonical_crop_batch(frame, M_aligns, out_w, out_h)
    return crops


def _extract_canonical_window(
    frame: np.ndarray | torch.Tensor,
    obb: OBBResult,
    margin: float,
    aspect_ratio: float,
    out_size: tuple[int, int],
    runtime: RuntimeContext,
) -> tuple[torch.Tensor, np.ndarray]:
    """Extract canonical crops for all detections in one frame.

    Delegates to the existing GPU or CPU canonical routine and returns
    ``(crops_tensor, native_sizes)`` where ``native_sizes`` is ``(n, 2)``
    int64 array of ``[h, w]`` before padding to ``out_size``.

    GPU path (tensor_on_cuda): calls ``gpu_canonical_crop_batch`` with
    ``out_size`` so crops are already at the target resolution.
    CPU/MPS path: calls ``_warp_canonical_crop`` per detection, records
    native sizes, then pads each crop to ``out_size``.
    """
    out_w, out_h = int(out_size[0]), int(out_size[1])
    n = obb.num_detections
    padding_fraction = max(0.0, float(margin) - 1.0)

    if runtime.tensor_on_cuda:
        # --- GPU path ---
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
            frame = frame.to(runtime.device)
        if frame.ndim == 4:
            frame = frame.squeeze(0)

        M_aligns: list[np.ndarray] = []
        native_hw: list[tuple[int, int]] = []
        for i in range(n):
            try:
                cw, ch = compute_native_crop_dimensions(
                    obb.corners[i], aspect_ratio, padding_fraction
                )
                M, _ = compute_alignment_affine(
                    obb.corners[i], cw, ch, padding_fraction
                )
            except ValueError:
                cw, ch = out_w, out_h
                M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            native_hw.append((ch, cw))
            M_aligns.append(M)

        crops = gpu_canonical_crop_batch(frame, M_aligns, out_w, out_h)
        sizes = np.array([[h, w] for h, w in native_hw], dtype=np.int64)
        return crops, sizes

    # --- CPU / MPS path ---
    # Use shared helpers: frame conversion + per-detection warp loop.
    arr = _frame_as_hwc_numpy(frame)
    raw_crops = _warp_crops_for_obb(arr, obb, aspect_ratio, padding_fraction)

    native_hw_list: list[tuple[int, int]] = [
        (c.shape[0], c.shape[1]) for c in raw_crops
    ]

    # Resize each native crop to out_size, matching the GPU path which warps
    # directly to (out_w, out_h) via affine_grid+grid_sample (bilinear,
    # padding_mode="border").  For crops smaller than out_size this replicates
    # the border; for crops larger than out_size this down-scales — both agree
    # with the GPU path and avoid the previous silent pixel-loss hard-crop.
    resized: list[np.ndarray] = []
    for crop in raw_crops:
        if crop.shape[0] == out_h and crop.shape[1] == out_w:
            resized.append(crop)
        else:
            resized.append(
                cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            )

    stacked = np.stack(resized, axis=0)  # (N, H, W, C)
    crops_t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    sizes = np.array([[h, w] for h, w in native_hw_list], dtype=np.int64)
    return crops_t, sizes


def extract_crops(
    frames: list,
    obb_results: list[OBBResult],
    *,
    canonical_margin: float,
    canonical_aspect_ratio: float,
    out_size: tuple[int, int],
    runtime: RuntimeContext,
) -> CropBatch:
    """Extract canonical crops across a window of frames into a single CropBatch.

    Concatenates detections in input order (frames ascending, detections in
    OBBResult order = detection-id order) — the reproducibility invariant.

    GPU path (tensor_on_cuda): crops are device-resident CUDA tensors.
    CPU/MPS path: crops are CPU tensors.
    """
    crops_list: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        frame_crops, sizes = _extract_canonical_window(
            frame, obb, canonical_margin, canonical_aspect_ratio, out_size, runtime
        )
        crops_list.append(frame_crops)  # (n_i, C, H, W)
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(
            np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64)
        )
        native_sizes.append(sizes)

    if not crops_list:
        device = runtime.device if runtime.cuda_mode else "cpu"
        empty = torch.zeros((0, 3, out_size[1], out_size[0]), device=device)
        return CropBatch(
            empty,
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            {o.frame_idx: o for o in obb_results},
            np.zeros((0, 2), np.int64),
        )

    return CropBatch(
        crops=torch.cat(crops_list, dim=0),
        detection_ids=np.concatenate(det_ids),
        frame_index=np.concatenate(frame_idx_list),
        obb_by_frame={o.frame_idx: o for o in obb_results},
        native_sizes=np.concatenate(native_sizes),
    )
