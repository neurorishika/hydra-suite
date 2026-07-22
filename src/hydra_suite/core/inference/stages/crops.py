from __future__ import annotations

import cv2
import numpy as np
import torch

from hydra_suite.core.canonicalization.crop import (
    _apply_foreign_mask_canonical,
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
    suppress_foreign: bool = False,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Extract OBB-aligned canonical crops. Returns (N, C, H, W) tensor on runtime.device.

    GPU path (tensor_on_cuda only): single batched affine_grid + grid_sample call.
    CPU path: cv2.warpAffine per crop -> stacked CPU tensor.
    onnx_cuda/tensorrt use CPU path even though cuda_mode=True; their downstream
    models take CPU numpy, so GPU crop upload+download would be pure waste.

    ``suppress_foreign=True`` blacks out the OTHER detections' OBB polygons in
    each crop (matching legacy's ``suppress_foreign_obb``, applied
    unconditionally there) via the same ``_apply_foreign_mask_canonical_batch``
    helper ``extract_canonical_crops_batch`` uses — this single-frame entry
    point previously had no masking support at all, a real (unintentional)
    legacy parity gap for any realtime/streaming caller.
    """
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros((0, 3, 64, 64), dtype=torch.float32)

    if runtime.tensor_on_cuda:
        crops = _extract_canonical_gpu(
            frame,
            obb_result,
            canonical_aspect_ratio,
            canonical_margin,
            runtime.device,
        )
    else:
        crops = _extract_canonical_cpu(
            frame, obb_result, canonical_aspect_ratio, canonical_margin
        )

    if suppress_foreign and n > 1:
        crops = _apply_foreign_mask_canonical_batch(
            crops,
            obb_result,
            canonical_aspect_ratio,
            canonical_margin,
            background_color,
        )
    return crops


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


def extract_classifier_crops_batch(
    frames: list,
    obb_results: list[OBBResult],
    target_size: tuple[int, int],
    aspect_ratio: float,
    margin: float,
) -> CropBatch:
    """Extract classifier crops across a window of frames into a single CropBatch.

    For each frame calls extract_classifier_crops (single warpAffine to model
    input size, BGR uint8), then stacks results in detection-id order. HT and
    CNN models may have different input sizes, so each calls this independently.
    target_size is (out_w, out_h) per legacy convention (index 0 = width).
    native_sizes rows are [out_h, out_w], reflecting the classifier crop dimensions.
    """
    out_w, out_h = int(target_size[0]), int(target_size[1])
    crops_list: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        np_crops = extract_classifier_crops(
            frame, obb, target_size, aspect_ratio, margin
        )
        # Convert list of HWC uint8 numpy arrays -> (N, C, H, W) float [0,1] tensor
        stacked = np.stack(np_crops, axis=0)  # (N, H, W, C)
        crops_t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
        crops_list.append(crops_t)
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(
            np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64)
        )
        native_sizes_list.append(
            np.full((obb.detection_ids.shape[0], 2), [out_h, out_w], np.int64)
        )

    if not crops_list:
        empty = torch.zeros((0, 3, out_h, out_w))
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
        native_sizes=np.concatenate(native_sizes_list),
    )


def frames_on_cuda(runtime, frames) -> bool:
    """Whether the GPU classifier crop path should run for this window.

    Requires BOTH a gpu tier (``runtime.tensor_on_cuda``) AND frames that are
    genuinely CUDA tensors. ``tensor_on_cuda`` only reflects NVDEC *availability*
    -- NVDEC can fall back to ``CpuFrameReader`` per clip (e.g. the MBCount limit
    on high-resolution video), in which case the frames are CPU numpy/tensors and
    uploading a whole frame to the GPU just to crop it is SLOWER than a CPU cv2
    warp. Gate on the real device so the GPU path only runs when it saves an
    actual device->host round-trip.
    """
    if not getattr(runtime, "tensor_on_cuda", False):
        return False
    for frame in frames:
        if frame is not None:
            return bool(torch.is_tensor(frame) and frame.is_cuda)
    return False


def extract_classifier_crops_gpu(
    frame: "torch.Tensor | np.ndarray",
    obb_result: OBBResult,
    target_size: tuple[int, int],
    aspect_ratio: float,
    margin: float,
    device: str,
) -> "torch.Tensor":
    """GPU-native analogue of :func:`extract_classifier_crops`.

    Warps each OBB directly to the classifier input size ``(out_w, out_h)`` with a
    single batched ``grid_sample`` on-device, using the SAME alignment affine the
    CPU path feeds to ``cv2.warpAffine`` (``compute_alignment_affine(corners,
    out_w, out_h, pad)`` — note ``aspect_ratio`` is unused here, matching the CPU
    entry point). Returns ``(N, C, out_h, out_w)`` float32 on ``device`` in the
    same BGR, ``[0, 1]`` convention as ``extract_classifier_crops_batch``'s tensor
    (``crops.py`` ``/255`` path). Used only when the frame is a CUDA tensor (NVDEC
    path); ``grid_sample`` != ``cv2`` bit-for-bit, so the CUDA pipeline's
    acceptance gate is identity agreement, not byte-identity (see the design spec).
    """
    out_w, out_h = int(target_size[0]), int(target_size[1])
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame.transpose(2, 0, 1))
    frame = frame.to(device)
    if frame.ndim == 4:
        frame = frame.squeeze(0)
    # NvdecFrameReader yields (H, W, 3) HWC tensors; gpu_canonical_crop_batch wants
    # (C, H, W). Detect channels-last and permute (mirrors obb.py's frame handling).
    if frame.ndim == 3 and frame.shape[-1] == 3 and frame.shape[0] != 3:
        frame = frame.permute(2, 0, 1)
    if frame.dtype == torch.uint8:
        frame = frame.float().div(255.0)
    frame = frame.contiguous()

    n = obb_result.num_detections
    n_ch = int(frame.shape[0])
    if n == 0:
        return torch.zeros((0, n_ch, out_h, out_w), dtype=torch.float32, device=device)

    pad = max(0.0, float(margin) - 1.0)
    m_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            m_align, _ = compute_alignment_affine(
                obb_result.corners[i], out_w, out_h, pad
            )
        except ValueError:
            m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        m_aligns.append(m_align)

    return gpu_canonical_crop_batch(frame, m_aligns, out_w, out_h)


def extract_classifier_crops_batch_gpu(
    frames: list,
    obb_results: list[OBBResult],
    target_size: tuple[int, int],
    aspect_ratio: float,
    margin: float,
    device: str,
) -> CropBatch:
    """GPU-native analogue of :func:`extract_classifier_crops_batch`.

    Per-frame :func:`extract_classifier_crops_gpu`, concatenated into a
    :class:`CropBatch` whose ``crops`` tensor stays on ``device`` (no host
    round-trip). Field layout is identical to the CPU batch builder so downstream
    ``select_frame`` / assembly is unchanged. ``target_size`` is ``(out_w, out_h)``.
    """
    out_w, out_h = int(target_size[0]), int(target_size[1])
    crops_list: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        crops_list.append(
            extract_classifier_crops_gpu(
                frame, obb, target_size, aspect_ratio, margin, device
            )
        )
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(
            np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64)
        )
        native_sizes_list.append(
            np.full((obb.detection_ids.shape[0], 2), [out_h, out_w], np.int64)
        )

    if not crops_list:
        empty = torch.zeros((0, 3, out_h, out_w), device=device)
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
        native_sizes=np.concatenate(native_sizes_list),
    )


def _apply_foreign_mask_canonical_batch(
    crops: torch.Tensor,
    obb: OBBResult,
    aspect_ratio: float,
    margin: float,
    background_color: tuple[int, int, int],
) -> torch.Tensor:
    """Black out foreign OBB polygons in each crop of one frame's crop tensor.

    ``crops`` is ``(N, C, H, W)`` float [0, 1] (the per-frame tensor padded to that
    frame's max canvas, top-left origin). For each detection ``i`` the OTHER
    detections (detection-id order) are projected into ``i``'s canonical space via
    its ``M_align`` and filled with ``background_color`` using the shared
    ``_apply_foreign_mask_canonical`` helper (cv2.fillPoly on a HWC uint8 view).

    The crop tensor may be CUDA-resident; masking uses a CPU round-trip
    (on-device polygon rasterisation is non-trivial) — same documented approach
    the old resize-based ``extract_crops`` used.
    """
    n = obb.num_detections
    padding_fraction = max(0.0, float(margin) - 1.0)

    m_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            cw, ch = compute_native_crop_dimensions(
                obb.corners[i], aspect_ratio, padding_fraction
            )
            M, _ = compute_alignment_affine(obb.corners[i], cw, ch, padding_fraction)
        except ValueError:
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        m_aligns.append(M)

    device = crops.device
    crops_np = (crops.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # crops_np is (N, C, H, W); operate per crop as a HWC view for fillPoly.
    for i in range(n):
        crop_hwc = np.ascontiguousarray(crops_np[i].transpose(1, 2, 0))
        foreign = [obb.corners[j] for j in range(n) if j != i]
        _apply_foreign_mask_canonical(
            crop_hwc,
            m_aligns[i],
            foreign,
            background_color,
            own_corners=obb.corners[i],
        )
        crops_np[i] = crop_hwc.transpose(2, 0, 1)

    return torch.from_numpy(crops_np).float().to(device) / 255.0


def extract_canonical_crops_batch(
    frames: list,
    obb_results: list[OBBResult],
    canonical_aspect_ratio: float,
    canonical_margin: float,
    runtime: RuntimeContext,
    suppress_foreign: bool = False,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> CropBatch:
    """Window-level canonical pose crops, bit-identical to ``extract_canonical_crops``.

    The per-frame batch-pass path builds pose crops via ``extract_canonical_crops``
    (native-extent warp, padded to the per-frame max, never resized). ``run_pose``
    then recovers each detection's native crop with
    ``compute_native_crop_dimensions`` and slices ``[:ch, :cw]``.

    The resize-based ``extract_crops`` (now removed) could not reproduce that
    numerically: it warped to native extent then *resized* every crop to a fixed
    ``out_size``, so ``run_pose_batch``'s ``native_sizes`` slice recovered a
    resampled (not bit-identical) crop. That would break the depth=1 correctness
    contract (pose keypoints must match the old ``_run_batch`` exactly).

    This builder instead warps each detection to its native extent (via the same
    ``extract_canonical_crops`` CPU/GPU routine) and pads — never resizes — to the
    WINDOW-wide max canvas, recording each crop's native ``[h, w]`` in
    ``native_sizes``. ``run_pose_batch`` slices back to native, recovering the
    exact pixels ``run_pose`` would have seen.

    Foreign-region suppression (``suppress_foreign=True``): to match legacy, each
    pose crop has the OTHER detections' OBB polygons in the SAME frame blacked out
    (filled with ``background_color``) in canonical space, via
    ``_apply_foreign_mask_canonical`` (same approach the old resize-based path
    used). The foreign set is the other detections in detection-id order
    (deterministic). Masking is a no-op for single-detection frames. The masking
    is applied to each detection's native crop (sliced out of the per-frame padded
    tensor) using its ``M_align``; padding is bottom/right zeros so ``M_align``
    still maps frame coords into the crop's top-left origin correctly. On CUDA the
    masking uses a CPU round-trip (on-device polygon rasterisation is non-trivial),
    documented as a follow-up optimisation — same as the old path.
    """
    per_frame: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        # extract_canonical_crops pads each frame's crops to that frame's own max.
        crops = extract_canonical_crops(
            frame, obb, canonical_aspect_ratio, canonical_margin, runtime
        )
        if suppress_foreign and obb.num_detections > 1:
            crops = _apply_foreign_mask_canonical_batch(
                crops,
                obb,
                canonical_aspect_ratio,
                canonical_margin,
                background_color,
            )
        per_frame.append(crops)
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(
            np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64)
        )
        # Native pre-pad dims per detection (mirrors run_pose recovery).
        pad = max(0.0, float(canonical_margin) - 1.0)
        sizes = np.zeros((obb.num_detections, 2), np.int64)
        for i in range(obb.num_detections):
            cw, ch = compute_native_crop_dimensions(
                obb.corners[i], canonical_aspect_ratio, pad
            )
            sizes[i] = (ch, cw)
        native_sizes_list.append(sizes)

    if not per_frame:
        device = runtime.device if runtime.cuda_mode else "cpu"
        empty = torch.zeros((0, 3, 8, 8), device=device)
        return CropBatch(
            empty,
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            {o.frame_idx: o for o in obb_results},
            np.zeros((0, 2), np.int64),
        )

    # Each frame's crop tensor is padded to that frame's max; concatenating
    # across frames needs a single window-wide canvas, so pad every frame's
    # tensor up to the window max (bottom/right zeros, same convention as
    # extract_canonical_crops).
    max_h = max(t.shape[2] for t in per_frame)
    max_w = max(t.shape[3] for t in per_frame)
    padded: list[torch.Tensor] = []
    for t in per_frame:
        ph = max_h - t.shape[2]
        pw = max_w - t.shape[3]
        if ph or pw:
            t = torch.nn.functional.pad(t, (0, pw, 0, ph))
        padded.append(t)

    return CropBatch(
        crops=torch.cat(padded, dim=0),
        detection_ids=np.concatenate(det_ids),
        frame_index=np.concatenate(frame_idx_list),
        obb_by_frame={o.frame_idx: o for o in obb_results},
        native_sizes=np.concatenate(native_sizes_list),
    )
