"""Pipelined pose inference orchestrator.

Optimizations over the baseline sequential loop in TrackingWorker:

1. **Parallel crop extraction** — ``ThreadPoolExecutor`` fans out per-detection
   crop slicing and foreign-OBB masking across CPU cores.
2. **Pre-resize / letterbox** — crops are optionally down-scaled to a uniform
   target size, reducing memory pressure in the accumulator and data volume
   transferred to the backend.
3. **Double-buffered inference** — while the GPU processes batch *N*, the main
   thread reads frames and extracts crops for batch *N+1*.
4. **Async cache writes** — a background thread serialises pose results to the
   properties cache, decoupling disk I/O from the inference hot-path.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Async cache writer
# ---------------------------------------------------------------------------

_SENTINEL = object()


class AsyncCacheWriter:
    """Drains pose results to an ``IndividualPropertiesCache`` on a daemon thread."""

    def __init__(self, cache_writer):
        self._writer = cache_writer
        self._queue: Queue = Queue()
        self._error: Optional[Exception] = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="pose-cache-writer"
        )
        self._thread.start()

    # -- public --

    def submit(
        self,
        frame_idx: int,
        det_ids: List[int],
        kp_list: List[Optional[np.ndarray]],
    ) -> None:
        """Enqueue a frame of pose results for background writing."""
        self._queue.put((frame_idx, det_ids, kp_list))

    def flush_and_close(self) -> None:
        """Block until all pending writes are done, then stop the thread."""
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._error is not None:
            raise self._error

    # -- internal --

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            try:
                frame_idx, det_ids, kp_list = item
                self._writer.add_frame(frame_idx, det_ids, pose_keypoints=kp_list)
            except Exception as exc:
                logger.error("AsyncCacheWriter error on frame %s: %s", frame_idx, exc)
                self._error = exc


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class PosePipeline:
    """Double-buffered pose inference with parallel extraction and async writes.

    Usage::

        pipeline = PosePipeline(backend, cache_writer, ...)
        completed = pipeline.run(video_cap, det_cache, detector, ...)
        pipeline.close()
    """

    def __init__(
        self,
        pose_backend,
        cache_writer,
        *,
        cross_frame_batch: int = 64,
        crop_workers: int = 4,
        pre_resize_target: int = 0,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        suppress_foreign_obb: bool = True,
        padding_fraction: float = 0.1,
        # PrecomputePhase protocol support:
        cache_hit: bool = False,
        cache_path: Optional[str] = None,
        finalize_metadata: Optional[dict] = None,
        frame_result_callback: Optional[
            Callable[[int, List[int], List[Optional[np.ndarray]]], None]
        ] = None,
    ):
        self._backend = pose_backend
        self._batch_size = max(1, cross_frame_batch)
        self._pre_resize = max(0, pre_resize_target)
        self._bg_color = bg_color
        self._suppress_foreign = suppress_foreign_obb
        self._padding = padding_fraction

        self._crop_pool = ThreadPoolExecutor(
            max_workers=max(1, crop_workers),
            thread_name_prefix="pose-crop",
        )
        self._infer_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pose-infer"
        )
        self._cache_writer = cache_writer
        self._async_cache = AsyncCacheWriter(cache_writer) if cache_writer else None

        # Batch accumulators
        self._pending: List[FrameCropResult] = []
        self._flat_crops: list = []  # np.ndarray (CPU) or Tensor (CUDA)
        self._inflight: Optional[Future] = None

        # PrecomputePhase protocol state
        self._cache_hit = cache_hit
        self._cache_path = cache_path
        self._finalize_metadata = finalize_metadata or {}
        self._frame_result_callback = frame_result_callback
        self._closed = False
        self._async_cache_closed = False
        self._completed_profile: Dict[str, float] = {}

    def set_frame_result_callback(
        self,
        callback: Optional[
            Callable[[int, List[int], List[Optional[np.ndarray]]], None]
        ],
    ) -> None:
        """Register a callback that receives per-frame pose outputs."""
        self._frame_result_callback = callback

    def sync(self, profiler=None) -> None:
        """Wait until any in-flight inference has completed."""
        self._wait_inflight(profiler=profiler)

    def _filter_canonical_affines(self, raw_affines, raw_ids, det_ids):
        """Map raw canonical affines to filtered detection indices."""
        if raw_affines is None or not raw_ids:
            return None
        raw_id_map = {int(rid): idx for idx, rid in enumerate(raw_ids)}
        filtered = {}
        for di, did in enumerate(det_ids):
            raw_idx = raw_id_map.get(int(did))
            if (
                raw_idx is not None
                and raw_idx < len(raw_affines)
                and raw_affines[raw_idx] is not None
            ):
                M_inv = cv2.invertAffineTransform(
                    np.asarray(raw_affines[raw_idx], dtype=np.float64)
                )
                filtered[di] = M_inv.astype(np.float32)
        return filtered if filtered else None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        video_cap,
        detection_cache,
        detector,
        start_frame: int,
        end_frame: int,
        resize_factor: float,
        roi_mask,
        progress_cb: Optional[Callable] = None,
        stats_cb: Optional[Callable] = None,
        stop_check: Optional[Callable] = None,
    ) -> bool:
        """Execute the pipeline over ``[start_frame, end_frame]``.

        Returns ``True`` if the run completed, ``False`` if cancelled.
        """
        total = max(1, end_frame - start_frame + 1)
        t0 = time.time()

        for rel_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if stop_check and stop_check():
                self._drain()
                return False

            # --- read frame (must be sequential) ---
            ret, frame = video_cap.read()
            if ret and resize_factor < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_factor,
                    fy=resize_factor,
                    interpolation=cv2.INTER_AREA,
                )

            # --- get filtered detections ---
            (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confs,
                raw_obb,
                raw_ids,
                raw_headings,
                raw_heading_confidences,
                raw_directed,
                raw_canonical_affines,
                _raw_canvas_dims,
                _raw_M_inverse,
            ) = detection_cache.get_frame(frame_idx)
            (
                meas,
                _sizes,
                _shapes,
                _confs,
                filt_obb,
                det_ids,
                _headings,
                _heading_confidences,
                _directed,
            ) = detector.filter_raw_detections(
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confs,
                raw_obb,
                roi_mask=roi_mask,
                detection_ids=raw_ids,
                heading_hints=raw_headings,
                heading_confidences=raw_heading_confidences,
                directed_mask=raw_directed,
            )

            all_obb = [np.asarray(c, dtype=np.float32) for c in (filt_obb or [])]

            # --- filter canonical affines to match filtered detections ---
            filtered_M_inv = self._filter_canonical_affines(
                raw_canonical_affines,
                raw_ids,
                det_ids,
            )

            # --- extract crops (parallel across detections) ---
            fcr = self._extract_frame_crops(
                frame if ret else None, meas, det_ids, filt_obb, all_obb, frame_idx
            )
            # Attach canonical M_inverse to FrameCropResult
            if filtered_M_inv:
                fcr.crop_M_inverses = filtered_M_inv
            self._pending.append(fcr)
            self._flat_crops.extend(fcr.crops)

            # --- flush when batch full or last frame ---
            is_last = rel_idx == total - 1
            if len(self._flat_crops) >= self._batch_size or is_last:
                if stop_check and stop_check():
                    self._drain()
                    return False
                self._flush()

            # --- progress reporting ---
            done = rel_idx + 1
            if rel_idx % 10 == 0 or is_last:
                elapsed = max(1e-6, time.time() - t0)
                fps = done / elapsed
                remaining = max(0, total - done)
                eta = remaining / fps if fps > 1e-9 else 0.0
                pct = int(done * 100 / total)
                if progress_cb:
                    progress_cb(pct, f"Pose precompute: {done}/{total}")
                if stats_cb:
                    stats_cb(
                        {
                            "phase": "pose_precompute",
                            "fps": fps,
                            "elapsed": elapsed,
                            "eta": eta,
                        }
                    )

        # wait for last in-flight inference
        self._wait_inflight()
        return True

    # ------------------------------------------------------------------ #
    # PrecomputePhase protocol                                            #
    # ------------------------------------------------------------------ #

    name = "pose"
    is_fatal = True

    def has_cache_hit(self) -> bool:
        """Return True if the cache was pre-built and the loop can be skipped."""
        return self._cache_hit

    def process_frame(
        self,
        frame_idx: int,
        crops: list,
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Accept pre-extracted crops and feed them into the inference pipeline.

        *crops* may contain either ``np.ndarray`` (CPU path) or CUDA
        ``torch.Tensor`` in ``C×H×W`` layout (GPU path from NVDec +
        :func:`~hydra_suite.core.canonicalization.crop.gpu_canonical_crop`).

        Letterboxing via :func:`letterbox_crop` is applied only to CPU
        (numpy) crops; GPU tensor crops skip this step because the backend's
        :meth:`~hydra_suite.core.identity.pose.backends.sleap.SleapExportedBackend._prepare_batch_cuda`
        handles letterboxing on-device.
        """
        if self._cache_hit:
            return
        if not crops:
            return  # sparse cache — no detections, nothing to infer

        fcr = FrameCropResult(
            frame_idx=frame_idx,
            det_ids=[int(det_id) for det_id in detection_ids],
            n_dets=len(all_obb),
            crops=[],
            crop_to_det=[],
            crop_offsets={},
            all_obb_corners=list(all_obb),
            crop_transforms={},
            crop_M_inverses={},
        )

        for ci, (crop, offset, det_idx) in enumerate(
            zip(crops, crop_offsets, crop_det_indices)
        ):
            slot_idx = _require_detection_index(det_idx, fcr.n_dets)
            processed = crop
            # Skip CPU letterbox for CUDA tensors — _prepare_batch_cuda handles
            # letterboxing on-device, so no transform to invert later.
            _is_cuda_tensor = hasattr(crop, "is_cuda") and crop.is_cuda
            if self._pre_resize > 0 and not _is_cuda_tensor:
                processed, transform = letterbox_crop(
                    processed, self._pre_resize, self._bg_color
                )
                fcr.crop_transforms[slot_idx] = transform
            fcr.crops.append(processed)
            fcr.crop_to_det.append(slot_idx)
            fcr.crop_offsets[slot_idx] = offset
            # Store M_inverse when canonical affines are available
            if (
                canonical_affines
                and ci < len(canonical_affines)
                and canonical_affines[ci] is not None
            ):
                fcr.crop_M_inverses[slot_idx] = canonical_affines[ci]

        self._pending.append(fcr)
        self._flat_crops.extend(fcr.crops)

        if len(self._flat_crops) >= self._batch_size:
            self._flush()

    def _close_async_cache(self) -> None:
        """Flush and close the async cache writer (idempotent)."""
        if self._async_cache is not None and not self._async_cache_closed:
            self._async_cache_closed = True
            try:
                self._async_cache.flush_and_close()
            except Exception:
                pass

    def finalize(self) -> Optional[str]:
        """Flush in-flight inference, write cache, return path."""
        if self._cache_hit:
            return self._cache_path
        if self._pending or self._flat_crops:
            self._flush()
        self._wait_inflight()
        self._close_async_cache()
        # Persist accumulated frames to disk with metadata.
        if self._cache_writer is not None:
            self._cache_writer.save(metadata=self._finalize_metadata)
        logger.info("Pose properties cache saved: %s", self._cache_path)
        return self._cache_path

    def close(self) -> None:
        """Shut down thread pools and release backend. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._wait_inflight()
        self._close_async_cache()
        self._crop_pool.shutdown(wait=False)
        self._infer_pool.shutdown(wait=False)
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Crop extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_frame_crops(
        self,
        frame: Optional[np.ndarray],
        meas,
        det_ids,
        filt_obb,
        all_obb: List[np.ndarray],
        frame_idx: int,
    ) -> FrameCropResult:
        """Extract crops for every detection in one frame using the thread pool."""
        fcr = FrameCropResult(
            frame_idx=frame_idx,
            det_ids=[int(det_id) for det_id in det_ids],
            n_dets=len(meas),
            crops=[],
            crop_to_det=[],
            crop_offsets={},
            all_obb_corners=all_obb,
            crop_transforms={},
        )
        if frame is None or not meas or not filt_obb:
            return fcr

        # Fan out crop extraction across the thread pool
        futures = []
        for det_idx, corners in enumerate(filt_obb):
            corners_arr = np.asarray(corners, dtype=np.float32)
            fut = self._crop_pool.submit(
                extract_one_crop,
                frame,
                corners_arr,
                det_idx,
                self._padding,
                all_obb,
                self._suppress_foreign,
                self._bg_color,
            )
            futures.append(fut)

        # Gather results (preserves deterministic det_idx ordering)
        for fut in futures:
            result = fut.result()
            if result is None:
                continue
            crop, offset, det_idx = result
            slot_idx = _require_detection_index(det_idx, fcr.n_dets)

            if self._pre_resize > 0:
                crop, transform = letterbox_crop(crop, self._pre_resize, self._bg_color)
                fcr.crop_transforms[slot_idx] = transform

            fcr.crops.append(crop)
            fcr.crop_to_det.append(slot_idx)
            fcr.crop_offsets[slot_idx] = offset

        return fcr

    # ------------------------------------------------------------------ #
    # Inference + cache (runs on the inference thread)                     #
    # ------------------------------------------------------------------ #

    def _flush(self) -> None:
        """Submit the accumulated batch for inference (double-buffered)."""
        # Block until the *previous* batch finishes
        self._wait_inflight()

        pending = list(self._pending)
        flat = list(self._flat_crops)
        self._pending.clear()
        self._flat_crops.clear()

        if not flat:
            # No crops, but still write empty frames so the cache is complete
            for pf in pending:
                if self._async_cache:
                    self._async_cache.submit(
                        pf.frame_idx, pf.det_ids, [None] * pf.n_dets
                    )
            return

        self._inflight = self._infer_pool.submit(self._infer_and_cache, pending, flat)

    def _infer_and_cache(
        self,
        pending: List[FrameCropResult],
        flat_crops: List[np.ndarray],
    ) -> None:
        """Run backend inference and enqueue results for cache writing.

        Executed on the single-worker inference thread so GPU work does not
        block the main thread's frame reading and crop extraction.
        """
        stage_profile: Dict[str, float] = {}
        stage_profile["pose_individuals"] = float(len(flat_crops))
        _use_cuda = (
            bool(flat_crops)
            and hasattr(flat_crops[0], "is_cuda")
            and flat_crops[0].is_cuda  # type: ignore[union-attr]
            and hasattr(self._backend, "predict_batch_cuda")
        )
        if _use_cuda:
            all_pred = self._backend.predict_batch_cuda(flat_crops)  # type: ignore[union-attr]
        else:
            all_pred = self._backend.predict_batch(flat_crops)
        consume_profile = getattr(self._backend, "consume_last_profile", None)
        if callable(consume_profile):
            try:
                backend_profile = consume_profile()
            except Exception:
                backend_profile = {}
            if backend_profile:
                for key, value in dict(backend_profile).items():
                    try:
                        stage_profile[str(key)] = float(value)
                    except Exception:
                        continue

        offset = 0
        postprocess_start = time.perf_counter()
        for pf in pending:
            n_crops = len(pf.crops)
            batch_slice = all_pred[offset : offset + n_crops]
            offset += n_crops

            pose_outputs: List[Optional[np.ndarray]] = [None] * pf.n_dets
            for ci, det_idx in enumerate(pf.crop_to_det):
                if ci >= len(batch_slice):
                    break
                slot_idx = _require_detection_index(det_idx, pf.n_dets)
                out = batch_slice[ci]
                kpts = out.keypoints
                crop_offset = pf.crop_offsets.get(slot_idx)
                M_inverse = pf.crop_M_inverses.get(slot_idx)
                if kpts is not None and crop_offset is not None and len(kpts) > 0:
                    gkpts = np.asarray(kpts, dtype=np.float32).copy()
                    # Undo letterbox if pre-resized
                    transform = pf.crop_transforms.get(slot_idx)
                    if transform is not None:
                        gkpts = invert_letterbox_keypoints(gkpts, transform)
                    # Map crop-local → frame-global coordinates
                    if M_inverse is not None:
                        # Canonical crop: use affine inverse for accurate mapping
                        from hydra_suite.core.canonicalization.crop import (
                            invert_keypoints,
                        )

                        gkpts = invert_keypoints(gkpts, M_inverse).astype(np.float32)
                    else:
                        # Legacy AABB crop: simple offset addition
                        x0, y0 = crop_offset
                        gkpts[:, 0] += float(x0)
                        gkpts[:, 1] += float(y0)
                    # Suppress keypoints landing inside other animals' OBBs
                    if self._suppress_foreign and len(pf.all_obb_corners) > 1:
                        from hydra_suite.utils.geometry import (
                            filter_keypoints_by_foreign_obbs,
                        )

                        gkpts = filter_keypoints_by_foreign_obbs(
                            gkpts, pf.all_obb_corners, target_idx=slot_idx
                        )
                    pose_outputs[slot_idx] = gkpts
                elif kpts is not None:
                    pose_outputs[slot_idx] = kpts

            if self._async_cache:
                self._async_cache.submit(pf.frame_idx, pf.det_ids, pose_outputs)
            if self._frame_result_callback is not None:
                self._frame_result_callback(pf.frame_idx, pf.det_ids, pose_outputs)
        stage_profile["pose_postprocess_s"] = stage_profile.get(
            "pose_postprocess_s", 0.0
        ) + (time.perf_counter() - postprocess_start)
        self._completed_profile = stage_profile

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _wait_inflight(self, profiler=None) -> None:
        """Block until the in-flight inference future completes."""
        if self._inflight is not None:
            self._inflight.result()
            self._inflight = None
        if profiler is not None and self._completed_profile:
            profile = dict(self._completed_profile)
            profiler.add_sample(
                "live_pose_transport",
                profile.get("pose_transport_s", 0.0),
                work_units=profile.get("pose_individuals", 0.0),
            )
            profiler.add_sample(
                "live_pose_inference",
                profile.get("pose_inference_s", 0.0),
                work_units=profile.get("pose_individuals", 0.0),
            )
            profiler.add_sample(
                "live_pose_postprocess",
                profile.get("pose_postprocess_s", 0.0),
                work_units=profile.get("pose_individuals", 0.0),
            )
            profiler.add_phase_time(
                "pose_transport",
                profile.get("pose_transport_s", 0.0),
                work_units=profile.get("pose_individuals", 0.0),
            )
            profiler.add_phase_time(
                "pose_inference",
                profile.get("pose_inference_s", 0.0),
                work_units=profile.get("pose_individuals", 0.0),
            )
            profiler.add_phase_time(
                "pose_postprocess",
                profile.get("pose_postprocess_s", 0.0),
                work_units=profile.get("pose_individuals", 0.0),
            )
        self._completed_profile = {}

    def _drain(self) -> None:
        """Clean up accumulators on cancellation."""
        self._wait_inflight()
        self._pending.clear()
        self._flat_crops.clear()
