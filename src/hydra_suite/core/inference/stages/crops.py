from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from hydra_suite.core.canonicalization.crop import _apply_foreign_mask_canonical
from hydra_suite.core.canonicalization.fit import FitResult
from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.canonicalization.resample import canonical_warp_batch_from_frame
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

from ..result import CropBatch, NumpyCropBatch, OBBResult
from ..runtime import RuntimeContext


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
    """Extract OBB-aligned canonical crops. Returns (N, C, canvas_h, canvas_w) tensor.

    One device-agnostic path for every runtime: the frame is normalised to a
    CHW float tensor on ITS OWN device (numpy -> CPU tensor; a CUDA/MPS tensor
    stays put) and warped via the single torch resampler seam
    (``canonical_warp_batch_from_frame``, ``F.grid_sample``), which slices each
    detection's AABB footprint out of the RAW frame and converts only those
    sub-regions -- the full frame is never materialised as a float32 tensor.
    There is no separate cv2
    CPU kernel any more -- ``runtime`` is accepted for signature/call-site
    compatibility but no longer selects a code path; the OUTPUT device always
    matches the frame's device, which downstream consumers (e.g.
    ``run_pose_batch``) already branch on via ``batch.crops.is_cuda``.

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
    del runtime  # kept for signature compatibility; device now follows frame
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros(
            (0, 3, geometry.canvas_h, geometry.canvas_w), dtype=torch.float32
        )

    device = frame.device if isinstance(frame, torch.Tensor) else "cpu"

    m_aligns: list[np.ndarray] = []
    with span(N.AFFINE_LOOP):
        for i in range(n):
            try:
                m_align, _theta, _clipped = canonical_affine(
                    obb_result.corners[i], geometry
                )
            except ValueError:
                m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            m_aligns.append(m_align)

    with span(N.WARP_BATCH, units=n, gpu=True):
        crops = canonical_warp_batch_from_frame(
            frame, m_aligns, geometry, lambda sub: _frame_to_chw_float(sub, device)
        )

    if suppress_foreign and n > 1:
        with span(N.FOREIGN_MASK, units=n):
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


# Layer 2 (``apply_fit``) over a whole window's crops is an embarrassingly
# parallel batch of independent per-crop resamples, each writing its own
# output buffer -- running the batch across a thread pool is therefore
# BYTE-IDENTICAL to the serial loop (order is preserved by ``pool.map``) but
# scales with cores. Env-tunable via ``HYDRA_CROP_WARP_THREADS``; set it to 1
# to force the serial path.
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

    Device-agnostic, same as :func:`extract_canonical_crops`: the frame is
    normalised to a CHW float tensor on its own device and warped in one
    batched call via the shared torch resampler seam
    (``canonical_warp_batch_from_frame``, which converts only each detection's
    AABB footprint rather than the whole frame), then quantised back to HWC
    uint8 -- there is
    no separate per-crop CPU kernel any more.
    """
    n = obb_result.num_detections
    if n == 0:
        return []

    device = frame.device if isinstance(frame, torch.Tensor) else "cpu"

    m_aligns: list[np.ndarray] = []
    with span(N.AFFINE_LOOP):
        for i in range(n):
            try:
                m_align, _theta, _clipped = canonical_affine(
                    obb_result.corners[i], geometry
                )
            except ValueError:
                m_align = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            m_aligns.append(m_align)

    with span(N.WARP_BATCH, units=n, gpu=True):
        crops_t = canonical_warp_batch_from_frame(
            frame, m_aligns, geometry, lambda sub: _frame_to_chw_float(sub, device)
        )
    crops_u8 = (
        (crops_t * 255.0)
        .round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [np.ascontiguousarray(crops_u8[i]) for i in range(n)]


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


def extract_classifier_crops_batch_np(
    frames: list,
    obb_results: list[OBBResult],
    geometry: CanonicalGeometry,
) -> NumpyCropBatch:
    """uint8 sibling of :func:`extract_classifier_crops_batch`.

    Same crops, same row order, same metadata — but the HWC uint8 BGR arrays
    ``extract_classifier_crops`` already produced are handed through as-is
    instead of being stacked into a ``(N, C, H, W)`` float32 ``[0, 1]`` torch
    tensor that the only CPU consumer immediately quantises back to uint8.
    That round trip is value-preserving (see :class:`NumpyCropBatch`), so this
    is byte-identical to ``extract_classifier_crops_batch`` followed by the
    ``permute/cpu/*255/clip/astype`` unpack, minus ~4 full-batch float32
    passes and two N-crop allocations per window.
    """
    out_h, out_w = geometry.canvas_h, geometry.canvas_w
    crops: list[np.ndarray] = []
    det_ids: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    native_sizes_list: list[np.ndarray] = []

    for frame, obb in zip(frames, obb_results):
        if obb.detection_ids.shape[0] == 0:
            continue
        crops.extend(extract_classifier_crops(frame, obb, geometry))
        det_ids.append(obb.detection_ids)
        frame_idx_list.append(
            np.full(obb.detection_ids.shape[0], obb.frame_idx, np.int64)
        )
        native_sizes_list.append(
            np.full((obb.detection_ids.shape[0], 2), [out_h, out_w], np.int64)
        )

    if not crops:
        return NumpyCropBatch(
            [],
            np.zeros(0, np.int64),
            np.zeros(0, np.int64),
            {o.frame_idx: o for o in obb_results},
            np.zeros((0, 2), np.int64),
        )

    return NumpyCropBatch(
        crops=crops,
        detection_ids=np.concatenate(det_ids),
        frame_index=np.concatenate(frame_idx_list),
        obb_by_frame={o.frame_idx: o for o in obb_results},
        native_sizes=np.concatenate(native_sizes_list),
    )


@N.spanned(N.APPLY_FIT)
def apply_fit_batch(crops: list, fit: FitResult) -> list:
    """Layer 2 over a whole window's crops, across the shared warp pool.

    ``apply_fit`` is one antialiased-bilinear resample plus a zero-canvas
    paste per crop, all independent -- each call writes its own output
    buffer, and ``pool.map`` preserves order, so this is byte-identical to
    ``[apply_fit(c, fit) for c in crops]``.
    """
    from hydra_suite.core.canonicalization.fit import apply_fit

    n = len(crops)
    pool = _get_warp_pool(_crop_warp_threads()) if n >= _WARP_MIN_PARALLEL else None
    if pool is None:
        return [apply_fit(c, fit) for c in crops]
    return list(pool.map(lambda c: apply_fit(c, fit), crops))


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
