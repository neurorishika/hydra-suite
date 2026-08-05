from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from hydra_suite.core.canonicalization.crop import (
    _apply_foreign_mask_canonical,
    gpu_canonical_crop_batch,
)
from hydra_suite.core.canonicalization.fit import FitResult
from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)

from ..result import CropBatch, OBBResult
from ..runtime import RuntimeContext


def apply_fit_gpu(batch: torch.Tensor, fit: FitResult) -> torch.Tensor:
    """On-device analogue of Layer 2's ``apply_fit``: the SAME isotropic
    centred letterbox (one scale, both axes; zero pad) computed by ``fit``,
    applied to a whole ``(N, C, H, W)`` batch with a single ``F.interpolate``
    + zero-canvas paste instead of per-crop ``cv2.resize``.

    This exists so the GPU (on-device, no-host-round-trip) crop path produces
    the SAME crop geometry as the CPU path -- same scale, same offset, same
    padded canvas size -- for a given ``fit`` (itself pure arithmetic over
    ``geometry.canvas_wh`` and the model's input size, identical on both
    paths). Only the resampling KERNEL differs: ``F.interpolate(...,
    mode="bilinear")`` has no on-device equivalent of cv2's ``INTER_AREA``
    downscale filter, so this is NOT bit-identical to ``apply_fit`` -- the
    acceptance gate here is identity agreement, not byte-identity, matching
    every other GPU/CPU crop divergence in this module (grid_sample != cv2).
    """
    import torch.nn.functional as F

    n, c = batch.shape[0], batch.shape[1]
    inner_w, inner_h = fit.inner_wh
    model_w, model_h = fit.model_wh
    resized = F.interpolate(
        batch, size=(inner_h, inner_w), mode="bilinear", align_corners=False
    )
    if (inner_h, inner_w) == (model_h, model_w):
        return resized
    canvas = torch.zeros(
        (n, c, model_h, model_w), dtype=batch.dtype, device=batch.device
    )
    ox, oy = fit.offset_xy
    canvas[:, :, oy : oy + inner_h, ox : ox + inner_w] = resized
    return canvas


def _frame_to_chw_float(
    frame: "torch.Tensor | np.ndarray", device: str
) -> "torch.Tensor":
    """Normalize any frame to a contiguous ``(C, H, W)`` float32 [0,1] tensor.

    Handles all three frame sources the GPU crop paths can receive:
    - numpy ``(H, W, 3)`` uint8 (cv2 CPU decode) -> transpose + /255,
    - torch ``(3, H, W)`` (already CHW),
    - torch ``(H, W, 3)`` uint8 (``NvdecFrameReader``, RGB) -> permute + /255.

    ``gpu_canonical_crop_batch``'s ``F.grid_sample`` requires float, so a raw
    NVDEC uint8 HWC frame must be converted here -- otherwise grid_sample raises
    ``"grid_sampler_2d_cuda" not implemented for 'Byte'`` (the bug that surfaced
    the first time NVDEC actually decoded to the GPU).
    """
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame.transpose(2, 0, 1))
    frame = frame.to(device)
    if frame.ndim == 4:
        frame = frame.squeeze(0)
    if frame.ndim == 3 and frame.shape[-1] == 3 and frame.shape[0] != 3:
        frame = frame.permute(2, 0, 1)  # HWC -> CHW (NVDEC)
    if frame.dtype == torch.uint8:
        frame = frame.float().div(255.0)
    elif frame.dtype != torch.float32:
        frame = frame.float()
    return frame.contiguous()


def extract_canonical_crops(
    frame: np.ndarray | torch.Tensor,
    obb_result: OBBResult,
    geometry: CanonicalGeometry,
    runtime: RuntimeContext,
    suppress_foreign: bool = False,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Extract OBB-aligned canonical crops. Returns (N, C, canvas_h, canvas_w) tensor on runtime.device.

    GPU path (tensor_on_cuda only): single batched affine_grid + grid_sample call.
    CPU path: cv2.warpAffine per crop -> stacked CPU tensor.
    onnx_cuda/tensorrt use CPU path even though cuda_mode=True; their downstream
    models take CPU numpy, so GPU crop upload+download would be pure waste.

    Every crop is warped straight to ``geometry``'s fixed canvas: the rigid
    Layer 1 transform (rotation + translation only, no scale), so there is no
    per-detection canvas size to reconcile downstream.

    ``suppress_foreign=True`` blacks out the OTHER detections' OBB polygons in
    each crop (matching legacy's ``suppress_foreign_obb``, applied
    unconditionally there) via the same ``_apply_foreign_mask_canonical_batch``
    helper ``extract_canonical_crops_batch`` uses — this single-frame entry
    point previously had no masking support at all, a real (unintentional)
    legacy parity gap for any realtime/streaming caller.
    """
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros(
            (0, 3, geometry.canvas_h, geometry.canvas_w), dtype=torch.float32
        )

    if runtime.tensor_on_cuda:
        crops = _extract_canonical_gpu(frame, obb_result, geometry, runtime.device)
    else:
        crops = _extract_canonical_cpu(frame, obb_result, geometry)

    if suppress_foreign and n > 1:
        crops = _apply_foreign_mask_canonical_batch(
            crops, obb_result, geometry, background_color
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


# Canonical pose-crop warping is an embarrassingly parallel batch of independent
# cv2.warpAffine calls: each reads the shared frame read-only and writes its own
# output buffer, and cv2 releases the GIL during warpAffine. Running the batch
# across a thread pool is therefore BYTE-IDENTICAL to the serial loop (order is
# preserved by ``pool.map``) but scales with cores. For a dense colony (~16
# crops/frame x hundreds of frames) this pose crop-warp was ~10s of otherwise
# serial CPU work whenever the frame stays on CPU (e.g. NVDEC-undecodable 4512^2
# H.264, so the on-GPU grid_sample crop path never engages). Env-tunable via
# ``HYDRA_CROP_WARP_THREADS``; set it to 1 to force the serial path.
_WARP_MIN_PARALLEL = 4  # below this, the serial loop beats pool-submit overhead
_warp_pool_lock = threading.Lock()
_warp_pool: ThreadPoolExecutor | None = None
_warp_pool_size = 0


def _crop_warp_threads() -> int:
    try:
        v = os.environ.get("HYDRA_CROP_WARP_THREADS")
        if v is not None:
            return max(1, int(v))
    except Exception:
        pass
    return max(1, min(8, os.cpu_count() or 1))


def _get_warp_pool(n_workers: int) -> ThreadPoolExecutor | None:
    """Lazily create (and cache) a shared warp thread pool; None when serial."""
    global _warp_pool, _warp_pool_size
    if n_workers <= 1:
        return None
    with _warp_pool_lock:
        if _warp_pool is None or _warp_pool_size != n_workers:
            if _warp_pool is not None:
                _warp_pool.shutdown(wait=False)
            _warp_pool = ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="cropwarp"
            )
            _warp_pool_size = n_workers
        return _warp_pool


def _warp_crops_for_obb(
    arr: np.ndarray,
    obb: OBBResult,
    geometry: CanonicalGeometry,
) -> list[np.ndarray]:
    """Warp each detection in *obb* onto the shared canonical canvas.

    Returns a list of HWC numpy arrays, one per detection, all sized
    ``(geometry.canvas_h, geometry.canvas_w)``. When the detection count is
    large enough to amortise pool overhead, the independent warps run across a
    shared thread pool (byte-identical; see module note above).
    """
    n = obb.num_detections

    def _one(i: int) -> np.ndarray:
        return _warp_canonical_crop(arr, obb.corners[i], geometry)

    pool = _get_warp_pool(_crop_warp_threads()) if n >= _WARP_MIN_PARALLEL else None
    if pool is not None:
        return list(pool.map(_one, range(n)))
    return [_one(i) for i in range(n)]


def _extract_canonical_cpu(
    frame: np.ndarray | torch.Tensor,
    obb: OBBResult,
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    arr = _frame_as_hwc_numpy(frame)
    crops = _warp_crops_for_obb(arr, obb, geometry)

    # Every crop is already the fixed geometry canvas size, so this is a
    # plain stack -- no batch-max zero-pad reconciliation needed.
    stacked = np.stack(crops, axis=0)  # (N, H, W, C)
    t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    return t


def extract_classifier_crops(
    frame: np.ndarray | torch.Tensor,
    obb_result: OBBResult,
    geometry: CanonicalGeometry,
) -> list[np.ndarray]:
    """Warp each OBB onto the shared canonical canvas (BGR uint8).

    Produces a canonical crop at ``geometry``'s fixed canvas size, matching
    every other crop entry point. The consumer is responsible for fitting the
    crop to its model's input size via Layer 2 (``fit_to_model_input`` /
    ``apply_fit``); this deliberately reinstates a double resample the old
    single-warp-straight-to-model-input path avoided, in exchange for one
    rigid Layer 1 transform shared by every consumer.
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
    out_h, out_w = geometry.canvas_h, geometry.canvas_w
    n_ch = arr.shape[2] if arr.ndim == 3 else 1
    crops: list[np.ndarray] = []
    for i in range(obb_result.num_detections):
        corners = obb_result.corners[i]
        try:
            m_align, _theta, _clipped = canonical_affine(corners, geometry)
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
    geometry: CanonicalGeometry,
) -> np.ndarray:
    """Extract one canonical crop on ``geometry``'s fixed canvas.

    Delegates to ``canonical_affine`` (Layer 1: rotation + translation only,
    no scale) so every detection lands on the same canvas size regardless of
    the animal's native pixel extent.
    """
    try:
        m_align, _theta, _clipped = canonical_affine(corners, geometry)
    except ValueError:
        n_ch = frame.shape[2] if frame.ndim == 3 else 1
        return np.zeros((geometry.canvas_h, geometry.canvas_w, n_ch), dtype=frame.dtype)
    return cv2.warpAffine(
        frame,
        m_align,
        (geometry.canvas_w, geometry.canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _extract_canonical_gpu(
    frame: torch.Tensor | np.ndarray,
    obb: OBBResult,
    geometry: CanonicalGeometry,
    device: str,
) -> torch.Tensor:
    """Batched corner-affine crop extraction on CUDA via gpu_canonical_crop_batch.

    Each detection's ``M_align`` is computed via ``canonical_affine`` (Layer 1:
    rotation + translation, no scale), then a single batched ``F.affine_grid``
    + ``F.grid_sample`` warp produces all crops at ``geometry``'s fixed canvas
    size.
    """
    frame = _frame_to_chw_float(frame, device)

    n = obb.num_detections
    m_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            m_align, _theta, _clipped = canonical_affine(obb.corners[i], geometry)
        except ValueError:
            m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        m_aligns.append(m_align)

    return gpu_canonical_crop_batch(frame, m_aligns, geometry=geometry)


def extract_classifier_crops_batch(
    frames: list,
    obb_results: list[OBBResult],
    geometry: CanonicalGeometry,
) -> CropBatch:
    """Extract classifier crops across a window of frames into a single CropBatch.

    For each frame calls extract_classifier_crops (single warpAffine onto the
    shared canonical canvas, BGR uint8), then stacks results in detection-id
    order. HT and CNN models may have different geometries, so each calls this
    independently. native_sizes rows are ``[canvas_h, canvas_w]`` -- every crop
    is already uniform, so there is nothing to slice back to.
    """
    out_h, out_w = geometry.canvas_h, geometry.canvas_w
    crops_list: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        np_crops = extract_classifier_crops(frame, obb, geometry)
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

    Requires BOTH a GPU tier (``runtime.requested_gpu`` — True on ``gpu`` and
    ``gpu_fast``) AND frames that are genuinely CUDA tensors. It deliberately does
    NOT key off ``tensor_on_cuda``: on ``gpu_fast`` the OBB backend is ``tensorrt``
    so ``tensor_on_cuda`` is False, yet NVDEC frames are real CUDA tensors that
    belong on the on-GPU crop path. NVDEC can also fall back to ``CpuFrameReader``
    per clip (e.g. the H.264 4096 / MBCount limit), in which case the frames are
    CPU numpy/tensors and uploading a whole frame to the GPU just to crop it is
    SLOWER than a CPU cv2 warp -- so we gate on the real frame device too.
    """
    if not getattr(runtime, "requested_gpu", False):
        return False
    for frame in frames:
        if frame is not None:
            return bool(torch.is_tensor(frame) and frame.is_cuda)
    return False


def extract_classifier_crops_gpu(
    frame: "torch.Tensor | np.ndarray",
    obb_result: OBBResult,
    geometry: CanonicalGeometry,
    device: str,
) -> "torch.Tensor":
    """GPU-native analogue of :func:`extract_classifier_crops`.

    Warps each OBB onto ``geometry``'s fixed canonical canvas with a single
    batched ``grid_sample`` on-device, using the SAME alignment affine
    (``canonical_affine``) the CPU path feeds to ``cv2.warpAffine``. Returns
    ``(N, C, canvas_h, canvas_w)`` float32 on ``device`` in the same BGR,
    ``[0, 1]`` convention as ``extract_classifier_crops_batch``'s tensor
    (``crops.py`` ``/255`` path). Used only when the frame is a CUDA tensor
    (NVDEC path); ``grid_sample`` != ``cv2`` bit-for-bit, so the CUDA
    pipeline's acceptance gate is identity agreement, not byte-identity (see
    the design spec).
    """
    out_h, out_w = geometry.canvas_h, geometry.canvas_w
    frame = _frame_to_chw_float(frame, device)

    n = obb_result.num_detections
    n_ch = int(frame.shape[0])
    if n == 0:
        return torch.zeros((0, n_ch, out_h, out_w), dtype=torch.float32, device=device)

    m_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            m_align, _theta, _clipped = canonical_affine(
                obb_result.corners[i], geometry
            )
        except ValueError:
            m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        m_aligns.append(m_align)

    return gpu_canonical_crop_batch(frame, m_aligns, geometry=geometry)


def extract_classifier_crops_batch_gpu(
    frames: list,
    obb_results: list[OBBResult],
    geometry: CanonicalGeometry,
    device: str,
) -> CropBatch:
    """GPU-native analogue of :func:`extract_classifier_crops_batch`.

    Per-frame :func:`extract_classifier_crops_gpu`, concatenated into a
    :class:`CropBatch` whose ``crops`` tensor stays on ``device`` (no host
    round-trip). Field layout is identical to the CPU batch builder so downstream
    ``select_frame`` / assembly is unchanged.
    """
    out_h, out_w = geometry.canvas_h, geometry.canvas_w
    crops_list: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        crops_list.append(extract_classifier_crops_gpu(frame, obb, geometry, device))
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
    geometry: CanonicalGeometry,
    background_color: tuple[int, int, int],
) -> torch.Tensor:
    """Black out foreign OBB polygons in each crop of one frame's crop tensor.

    ``crops`` is ``(N, C, canvas_h, canvas_w)`` float [0, 1]. For each
    detection ``i`` the OTHER detections (detection-id order) are projected
    into ``i``'s canonical space via its ``M_align`` (``canonical_affine``)
    and filled with ``background_color`` using the shared
    ``_apply_foreign_mask_canonical`` helper (cv2.fillPoly on a HWC uint8
    view).

    The crop tensor may be CUDA-resident; masking uses a CPU round-trip
    (on-device polygon rasterisation is non-trivial) — same documented approach
    the old resize-based ``extract_crops`` used.
    """
    n = obb.num_detections

    m_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            m_align, _theta, _clipped = canonical_affine(obb.corners[i], geometry)
        except ValueError:
            m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        m_aligns.append(m_align)

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
    geometry: CanonicalGeometry,
    runtime: RuntimeContext,
    suppress_foreign: bool = False,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> CropBatch:
    """Window-level canonical pose crops, bit-identical to ``extract_canonical_crops``.

    Every crop lands on ``geometry``'s fixed canvas, so unlike the old
    native-extent-then-pad-to-window-max scheme, frames concatenate directly:
    no per-frame max tracking, no window-wide re-pad, and nothing for
    ``run_pose_batch`` to slice back to native (``native_sizes`` rows are all
    ``[canvas_h, canvas_w]``, i.e. the full crop).

    Foreign-region suppression (``suppress_foreign=True``): to match legacy, each
    pose crop has the OTHER detections' OBB polygons in the SAME frame blacked out
    (filled with ``background_color``) in canonical space, via
    ``_apply_foreign_mask_canonical_batch``. The foreign set is the other
    detections in detection-id order (deterministic). Masking is a no-op for
    single-detection frames. On CUDA the masking uses a CPU round-trip
    (on-device polygon rasterisation is non-trivial), documented as a
    follow-up optimisation — same as the old path.
    """
    per_frame: list[torch.Tensor] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        crops = extract_canonical_crops(frame, obb, geometry, runtime)
        if suppress_foreign and obb.num_detections > 1:
            crops = _apply_foreign_mask_canonical_batch(
                crops, obb, geometry, background_color
            )
        per_frame.append(crops)
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(
            np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64)
        )
        native_sizes_list.append(
            np.full(
                (obb.num_detections, 2),
                [geometry.canvas_h, geometry.canvas_w],
                np.int64,
            )
        )

    if not per_frame:
        device = runtime.device if runtime.cuda_mode else "cpu"
        empty = torch.zeros((0, 3, geometry.canvas_h, geometry.canvas_w), device=device)
        return CropBatch(
            empty,
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            {o.frame_idx: o for o in obb_results},
            np.zeros((0, 2), np.int64),
        )

    return CropBatch(
        crops=torch.cat(per_frame, dim=0),
        detection_ids=np.concatenate(det_ids),
        frame_index=np.concatenate(frame_idx_list),
        obb_by_frame={o.frame_idx: o for o in obb_results},
        native_sizes=np.concatenate(native_sizes_list),
    )
