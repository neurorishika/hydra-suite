"""Unified precompute pipeline.

All phases (pose, AprilTag, CNN identity, ...) implement PrecomputePhase and
receive identical pre-extracted crops each frame via process_frame().

Single video read per tracking run regardless of how many phases are enabled.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from hydra_suite.core.identity.classification.apriltag import AprilTagDetector
from hydra_suite.core.identity.classification.cnn import (
    ClassPrediction,
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
)
from hydra_suite.core.tracking.pose_pipeline import extract_one_crop
from hydra_suite.data.tag_observation_cache import TagObservationCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PrecomputePhase:
    """Protocol for a precompute phase.

    Implement all four methods plus ``name`` and ``is_fatal`` attributes.
    """

    name: str  # e.g. "pose", "apriltag", "cnn_identity"
    is_fatal: bool  # True → failure in finalize() aborts tracking

    def has_cache_hit(self) -> bool:
        """Return True if a valid existing cache covers this run.

        Called before the frame loop. If all phases return True, the video
        read is skipped entirely.
        """
        raise NotImplementedError

    def process_frame(
        self,
        frame_idx: int,
        crops: List[np.ndarray],
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Process one frame.

        Called with empty lists when no detections exist for the frame.
        Must be a no-op when has_cache_hit() returned True.

        Note: len(crops) <= len(all_obb). Not every OBB produces a crop (zero-size
        or invalid bounding boxes yield None from extract_one_crop and are skipped).
        detection_ids gives the full detection-ID list for the frame, aligned to
        all_obb. crop_det_indices[i] gives the detection-slot index into all_obb
        for crops[i].

        canonical_affines: When canonical crops are enabled, parallel list of
        M_inverse (2x3 float32) arrays for each crop, or None per entry if
        unavailable.  Phases should use ``invert_keypoints(kpts, M_inverse)``
        instead of offset addition when an affine is present.
        """
        raise NotImplementedError

    def finalize(self) -> Optional[str]:
        """Flush caches and return artifact path, or None on failure.

        Runs to completion — stop_check is NOT called inside finalize().
        Non-fatal phases should catch their own exceptions and return None.
        Fatal phases should re-raise; UnifiedPrecompute propagates the exception.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release backend/model resources.

        Called after finalize(), or after cancellation (no finalize() in that case).
        Must be idempotent.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CropConfig
# ---------------------------------------------------------------------------


@dataclass
class CropConfig:
    """Controls shared crop extraction in UnifiedPrecompute."""

    padding_fraction: float = 0.1  # maps to INDIVIDUAL_CROP_PADDING
    suppress_foreign: bool = True  # maps to SUPPRESS_FOREIGN_OBB_REGIONS
    bg_color: Tuple[int, int, int] = (0, 0, 0)  # maps to INDIVIDUAL_BACKGROUND_COLOR
    reference_aspect_ratio: float = 2.0  # used to derive native-scale crop dims


# ---------------------------------------------------------------------------
# UnifiedPrecompute
# ---------------------------------------------------------------------------


class UnifiedPrecompute:
    """Single-pass precompute orchestrator.

    Reads each video frame once, extracts crops once, and dispatches to all
    registered phases. Phases are responsible for their own batching, cache
    writes, and resource lifecycle.
    """

    _CROP_WORKERS = 4  # parallel crop extraction threads

    def __init__(
        self,
        phases: List[PrecomputePhase],
        crop_config: CropConfig,
    ) -> None:
        self._phases = phases
        self._crop_config = crop_config
        # Pre-compute whether any phase needs AABB crops (e.g. AprilTag).
        self._needs_aabb = any(getattr(p, "_prefer_aabb_crops", False) for p in phases)

    # ------------------------------------------------------------------
    # Finalize / close helpers
    # ------------------------------------------------------------------

    def _finalize_phases(
        self,
        warning_cb: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Optional[str]]:
        """Run finalize() on all phases, then close() all phases.

        Fatal phases re-raise; non-fatal failures are logged and yield None.
        """
        results: Dict[str, Optional[str]] = {}
        try:
            for p in self._phases:
                try:
                    results[p.name] = p.finalize()
                except Exception as exc:
                    if p.is_fatal:
                        raise
                    logger.warning(
                        "Precompute phase '%s' finalize failed: %s", p.name, exc
                    )
                    if warning_cb:
                        warning_cb(f"Precompute Warning ({p.name})", str(exc))
                    results[p.name] = None
        finally:
            self._close_all_phases()
        return results

    def _close_all_phases(self) -> None:
        """Call close() on every phase, swallowing exceptions."""
        for p in self._phases:
            try:
                p.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Frame reading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_and_resize_frame(prefetcher, resize_factor: float, profiler):
        """Read a frame from the prefetcher and optionally resize it.

        Returns (ok, frame) where ok is False when the read failed.
        """
        if profiler:
            profiler.phase_start("precompute_frame_read")
        ret, frame = prefetcher.read()
        if not ret:
            if profiler:
                profiler.phase_end("precompute_frame_read")
            return False, None
        if resize_factor < 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=resize_factor,
                fy=resize_factor,
                interpolation=cv2.INTER_AREA,
            )
        if profiler:
            profiler.phase_end("precompute_frame_read")
        return True, frame

    @staticmethod
    def _log_prefetcher_failure(prefetcher, frame_idx, rel_idx, total):
        """Emit a detailed warning when the frame prefetcher fails."""
        _thread_alive = prefetcher.thread.is_alive() if prefetcher.thread else False
        logger.warning(
            "cap.read() failed at frame %d (frame %d/%d) — stopping precompute early. "
            "Prefetcher diagnostics: bg_thread_alive=%s, "
            "frames_read_by_thread=%d, frames_consumed=%d, "
            "queue_size=%d/%d, stopped_reason=%s, "
            "last_read_ok=%s, exception=%s",
            frame_idx,
            rel_idx + 1,
            total,
            _thread_alive,
            prefetcher._frames_read,
            prefetcher._frames_consumed,
            prefetcher.frame_queue.qsize(),
            prefetcher.buffer_size,
            prefetcher._stopped_reason,
            prefetcher._last_read_ok,
            prefetcher.exception,
        )

    # ------------------------------------------------------------------
    # Detection loading / filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw_detections(detection_cache, frame_idx, profiler):
        """Load raw detections from the cache for a single frame."""
        if profiler:
            profiler.phase_start("precompute_cache_load")
        try:
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
        except Exception:
            raw_meas = raw_sizes = raw_shapes = raw_confs = []
            raw_obb = raw_ids = raw_headings = raw_heading_confidences = (
                raw_directed
            ) = []
            raw_canonical_affines = None
        if profiler:
            profiler.phase_end("precompute_cache_load")
        return (
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
        )

    @staticmethod
    def _filter_detections(
        detector,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confs,
        raw_obb,
        raw_ids,
        raw_headings,
        raw_heading_confidences,
        raw_directed,
        roi_mask,
        profiler,
    ):
        """Apply detection filter and return (filt_obb, det_ids)."""
        if profiler:
            profiler.phase_start("precompute_filter")
        (
            _meas,
            _sz,
            _sh,
            _cf,
            filt_obb,
            det_ids,
            _hd,
            _hc,
            _dm,
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
        if profiler:
            profiler.phase_end("precompute_filter")
        return filt_obb, det_ids

    # ------------------------------------------------------------------
    # Canonical affine filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_canonical_affines(raw_canonical_affines, raw_ids, det_ids):
        """Map filtered detection IDs back to their canonical affines.

        Returns (use_canonical, filtered_affines).
        """
        use_canonical = raw_canonical_affines is not None
        filtered_affines: Optional[List[Optional[np.ndarray]]] = None
        if use_canonical and raw_ids:
            raw_id_to_idx: Dict[int, int] = {}
            for idx, rid in enumerate(raw_ids):
                raw_id_to_idx[int(rid)] = idx
            filtered_affines = []
            for did in det_ids:
                raw_idx = raw_id_to_idx.get(int(did))
                if (
                    raw_idx is not None
                    and raw_idx < len(raw_canonical_affines)
                    and raw_canonical_affines[raw_idx] is not None
                ):
                    filtered_affines.append(raw_canonical_affines[raw_idx])
                else:
                    filtered_affines.append(None)
        return use_canonical, filtered_affines

    # ------------------------------------------------------------------
    # Crop extraction
    # ------------------------------------------------------------------

    def _extract_crops_for_frame(
        self,
        frame,
        all_obb,
        filtered_affines,
        use_canonical: bool,
        profiler,
    ):
        """Extract AABB and canonical crops for all detections in one frame.

        Returns (aabb_crops, aabb_offsets, canonical_crops, canonical_M_inv,
                 crop_det_indices).
        """
        aabb_crops: List[np.ndarray] = []
        aabb_offsets: List[Tuple[int, int]] = []
        canonical_crops: List[np.ndarray] = []
        canonical_M_inv: List[Optional[np.ndarray]] = []
        crop_det_indices: List[int] = []

        if not all_obb:
            return (
                aabb_crops,
                aabb_offsets,
                canonical_crops,
                canonical_M_inv,
                crop_det_indices,
            )

        if profiler:
            profiler.phase_start("precompute_crop_extraction")

        from hydra_suite.core.canonicalization.crop import (
            compute_native_crop_dimensions,
            extract_canonical_crop,
        )

        cfg = self._crop_config

        def _extract_single(
            di_corners,
            frame=frame,
            all_obb=all_obb,
            filtered_affines=filtered_affines,
            use_canonical=use_canonical,
        ):
            """Extract AABB and/or canonical crop for one detection."""
            di, corners = di_corners
            aabb_result = self._extract_aabb_if_needed(
                frame,
                corners,
                di,
                cfg,
                all_obb,
            )
            if self._needs_aabb and aabb_result is None:
                return None  # skip detection entirely

            M_can = (
                filtered_affines[di]
                if (filtered_affines and di < len(filtered_affines))
                else None
            )
            c_crop, M_inv = self._extract_canonical_if_available(
                frame,
                corners,
                M_can,
                use_canonical,
                cfg,
                compute_native_crop_dimensions,
                extract_canonical_crop,
            )

            # When AABB was skipped (no phase needs it), we still need
            # a validity check -- canonical crop must succeed.
            if not self._needs_aabb:
                aabb_result = self._ensure_aabb_fallback(
                    frame,
                    corners,
                    di,
                    cfg,
                    all_obb,
                    c_crop,
                )
                if aabb_result is None:
                    return None

            return (di, aabb_result, c_crop, M_inv)

        # Fan out crop extraction across threads
        with ThreadPoolExecutor(
            max_workers=min(self._CROP_WORKERS, len(all_obb)),
            thread_name_prefix="precompute-crop",
        ) as pool:
            results = list(
                pool.map(
                    _extract_single,
                    [(di, corners) for di, corners in enumerate(all_obb)],
                )
            )

        # Gather results in deterministic order
        for result in results:
            if result is None:
                continue
            di, aabb_result, c_crop, M_inv = result
            aabb_crop, offset, _ = aabb_result
            aabb_crops.append(aabb_crop)
            aabb_offsets.append((int(offset[0]), int(offset[1])))
            crop_det_indices.append(di)

            if c_crop is not None:
                canonical_crops.append(c_crop)
                canonical_M_inv.append(M_inv)
            else:
                canonical_crops.append(aabb_crop)
                canonical_M_inv.append(None)

        if profiler:
            profiler.phase_end("precompute_crop_extraction")

        return (
            aabb_crops,
            aabb_offsets,
            canonical_crops,
            canonical_M_inv,
            crop_det_indices,
        )

    def _extract_aabb_if_needed(self, frame, corners, di, cfg, all_obb):
        """Extract an AABB crop if any phase requires it, else return None."""
        if not self._needs_aabb:
            return None
        return extract_one_crop(
            frame,
            corners,
            di,
            cfg.padding_fraction,
            all_obb,
            cfg.suppress_foreign,
            cfg.bg_color,
        )

    @staticmethod
    def _extract_canonical_if_available(
        frame,
        corners,
        M_can,
        use_canonical,
        cfg,
        compute_native_crop_dimensions,
        extract_canonical_crop,
    ):
        """Extract a canonical crop when affine data is available.

        Returns (c_crop, M_inv) or (None, None).
        """
        if not (use_canonical and M_can is not None):
            return None, None
        _cw, _ch = compute_native_crop_dimensions(
            corners,
            cfg.reference_aspect_ratio,
            cfg.padding_fraction,
        )
        c_crop = extract_canonical_crop(
            frame,
            M_can,
            _cw,
            _ch,
            bg_color=cfg.bg_color,
        )
        M_inv = cv2.invertAffineTransform(np.asarray(M_can, dtype=np.float64)).astype(
            np.float32
        )
        return c_crop, M_inv

    def _ensure_aabb_fallback(self, frame, corners, di, cfg, all_obb, c_crop):
        """When no phase needs AABB, ensure we still have a valid crop result.

        Falls back to AABB extraction if canonical failed, or synthesises a
        minimal result tuple from the canonical crop.
        """
        if c_crop is None:
            return extract_one_crop(
                frame,
                corners,
                di,
                cfg.padding_fraction,
                all_obb,
                cfg.suppress_foreign,
                cfg.bg_color,
            )
        # Synthesise a minimal aabb_result for offset tracking
        c = np.asarray(corners, dtype=np.float32)
        x0 = max(0, int(c[:, 0].min()))
        y0 = max(0, int(c[:, 1].min()))
        return (c_crop, (x0, y0), di)

    # ------------------------------------------------------------------
    # Phase dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch_to_phases(
        phases,
        frame_idx,
        aabb_crops,
        canonical_crops,
        frame_detection_ids,
        crop_det_indices,
        all_obb,
        aabb_offsets,
        canonical_M_inv,
        use_canonical,
        profiler,
        live_profile: bool = False,
    ):
        """Send extracted crops to every registered phase."""
        for phase in phases:
            _phase_prof_name = f"precompute_{phase.name}"
            live_category = None
            work_units = len(crop_det_indices)
            if live_profile:
                if phase.name == "apriltag":
                    live_category = "live_apriltag"
                elif str(phase.name).startswith("cnn"):
                    live_category = "live_cnn_identity"
            if profiler:
                profiler.phase_start(_phase_prof_name)
                if live_category:
                    profiler.tick(live_category)
            if getattr(phase, "_prefer_aabb_crops", False):
                phase.process_frame(
                    frame_idx,
                    aabb_crops,
                    frame_detection_ids,
                    crop_det_indices,
                    all_obb,
                    aabb_offsets,
                )
            else:
                phase.process_frame(
                    frame_idx,
                    canonical_crops if use_canonical else aabb_crops,
                    frame_detection_ids,
                    crop_det_indices,
                    all_obb,
                    aabb_offsets,
                    canonical_affines=(canonical_M_inv if use_canonical else None),
                )
            if profiler:
                if live_category:
                    profiler.tock(live_category, work_units=work_units)
                profiler.phase_end(_phase_prof_name, work_units=work_units)

    def process_live_frame(
        self,
        frame_idx: int,
        frame,
        detector,
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
        roi_mask,
        profiler=None,
    ) -> None:
        """Run all registered phases for a single live frame.

        This is the realtime counterpart to run(): detections are already
        available for the current frame, so the phases only need filtering,
        crop extraction, and per-frame dispatch.
        """
        if not self._phases:
            return

        if profiler:
            profiler.tick("live_filter")
        filt_obb, det_ids = self._filter_detections(
            detector,
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confs,
            raw_obb,
            raw_ids,
            raw_headings,
            raw_heading_confidences,
            raw_directed,
            roi_mask,
            profiler,
        )
        if profiler:
            profiler.tock("live_filter")
        all_obb = [np.asarray(c, dtype=np.float32) for c in (filt_obb or [])]
        use_canonical, filtered_affines = self._filter_canonical_affines(
            raw_canonical_affines,
            raw_ids,
            det_ids,
        )
        if profiler:
            profiler.tick("live_crop_extraction")
        (
            aabb_crops,
            aabb_offsets,
            canonical_crops,
            canonical_M_inv,
            crop_det_indices,
        ) = self._extract_crops_for_frame(
            frame,
            all_obb,
            filtered_affines,
            use_canonical,
            profiler,
        )
        if profiler:
            profiler.tock("live_crop_extraction")
        frame_detection_ids = [
            int(det_ids[i]) if det_ids and i < len(det_ids) else i
            for i in range(len(all_obb))
        ]
        self._dispatch_to_phases(
            self._phases,
            frame_idx,
            aabb_crops,
            canonical_crops,
            frame_detection_ids,
            crop_det_indices,
            all_obb,
            aabb_offsets,
            canonical_M_inv,
            use_canonical,
            profiler,
            live_profile=True,
        )

    def sync_live_frame(self, profiler=None) -> None:
        """Wait for any live phase that needs per-frame synchronization."""
        for phase in self._phases:
            sync = getattr(phase, "sync", None)
            if callable(sync):
                if profiler is not None:
                    try:
                        sync(profiler=profiler)
                        continue
                    except TypeError:
                        pass
                sync()

    def finalize_live(self, warning_cb: Optional[Callable[[str, str], None]] = None):
        """Finalize all live phases and persist their artifacts."""
        return self._finalize_phases(warning_cb)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        cap: cv2.VideoCapture,
        detection_cache,
        detector,
        start_frame: int,
        end_frame: int,
        resize_factor: float,
        roi_mask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        warning_cb: Optional[Callable[[str, str], None]] = None,
        profiler=None,
    ) -> Dict[str, Optional[str]]:
        """Run all phases over [start_frame, end_frame].

        Returns {phase.name: cache_path_or_none} for every phase.
        """
        if not self._phases:
            return {}

        # --- cache-hit short-circuit ---
        hits = [p.has_cache_hit() for p in self._phases]
        if all(hits):
            return self._finalize_phases(warning_cb)

        total = max(1, end_frame - start_frame + 1)

        # --- frame loop ---
        cancelled = False

        # Prefetch frames in a background thread to overlap I/O with
        # crop-extraction / inference work.
        # Use a generous timeout: when GPU inference (MPS / CUDA) runs
        # concurrently with hardware video decoding (VideoToolbox / NVDEC),
        # the decoder can stall for tens of seconds under resource contention.
        from hydra_suite.utils.frame_prefetcher import FramePrefetcher

        _prefetcher = FramePrefetcher(cap, buffer_size=8, read_timeout=120.0)
        _prefetcher.start()
        for rel_idx in range(total):
            frame_idx = start_frame + rel_idx

            # read + optional resize
            ok, frame = self._read_and_resize_frame(
                _prefetcher, resize_factor, profiler
            )
            if not ok:
                self._log_prefetcher_failure(_prefetcher, frame_idx, rel_idx, total)
                break

            # get raw detections
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
            ) = self._load_raw_detections(detection_cache, frame_idx, profiler)

            # filter detections (ROI mask, size/confidence gates)
            filt_obb, det_ids = self._filter_detections(
                detector,
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confs,
                raw_obb,
                raw_ids,
                raw_headings,
                raw_heading_confidences,
                raw_directed,
                roi_mask,
                profiler,
            )

            all_obb = [np.asarray(c, dtype=np.float32) for c in (filt_obb or [])]

            # --- Determine canonical extraction availability ---
            use_canonical, filtered_affines = self._filter_canonical_affines(
                raw_canonical_affines,
                raw_ids,
                det_ids,
            )

            # --- Extract crops (parallelised across detections) ---
            (
                aabb_crops,
                aabb_offsets,
                canonical_crops,
                canonical_M_inv,
                crop_det_indices,
            ) = self._extract_crops_for_frame(
                frame,
                all_obb,
                filtered_affines,
                use_canonical,
                profiler,
            )

            frame_detection_ids = [
                int(det_ids[i]) if det_ids and i < len(det_ids) else i
                for i in range(len(all_obb))
            ]

            # --- Fan-out to all phases ---
            self._dispatch_to_phases(
                self._phases,
                frame_idx,
                aabb_crops,
                canonical_crops,
                frame_detection_ids,
                crop_det_indices,
                all_obb,
                aabb_offsets,
                canonical_M_inv,
                use_canonical,
                profiler,
            )

            # cancellation check -- AFTER processing the frame
            if stop_check and stop_check():
                cancelled = True
                break

            # progress
            if progress_cb and (rel_idx % 50 == 0 or rel_idx == total - 1):
                pct = int((rel_idx + 1) * 100 / total)
                progress_cb(pct, f"Precompute: {rel_idx + 1}/{total} frames")
        _prefetcher.stop()

        # --- cancellation: skip finalize, just close ---
        if cancelled:
            self._close_all_phases()
            return {p.name: None for p in self._phases}

        # --- finalize all phases ---
        return self._finalize_phases(warning_cb)


# ---------------------------------------------------------------------------
# AprilTagPrecomputePhase
# ---------------------------------------------------------------------------


class AprilTagPrecomputePhase(PrecomputePhase):
    """Precompute phase that runs AprilTag detection on per-frame crops.

    Non-fatal: a detection failure will not abort the tracking run.
    If a compatible cache already exists for the requested frame range, the
    detector is never created and the video read is skipped.
    """

    name = "apriltag"
    is_fatal = False
    _prefer_aabb_crops = True  # AprilTag needs AABB crops for precise edge detection

    def __init__(
        self,
        detector_config,
        cache_path,
        start_frame: int,
        end_frame: int,
        video_path: str = "",
        frame_result_callback: Optional[Callable[..., None]] = None,
        ignore_existing_cache: bool = False,
    ) -> None:
        self._cache_path = Path(cache_path)
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._video_path = video_path
        self._hit = False
        self._detector: Optional[AprilTagDetector] = None
        self._tag_cache: Optional[TagObservationCache] = None
        self._frame_result_callback = frame_result_callback

        # Check for a compatible existing cache.
        if ignore_existing_cache and self._cache_path.exists():
            logger.info(
                "Realtime workflow ignoring existing AprilTag cache: %s",
                self._cache_path,
            )
        elif self._cache_path.exists():
            probe = TagObservationCache(
                self._cache_path, mode="r", start_frame=start_frame, end_frame=end_frame
            )
            try:
                if probe.is_compatible() and probe.covers_frame_range(
                    start_frame, end_frame
                ):
                    self._hit = True
                    logger.info(
                        "AprilTag cache hit: %s covers frames %d-%d",
                        self._cache_path,
                        start_frame,
                        end_frame,
                    )
                    return
                # fall through to create detector + write-mode cache
                logger.info(
                    "AprilTag cache miss or incompatible at %s — will regenerate",
                    self._cache_path,
                )
            finally:
                probe.close()

        # Cache miss: create detector and write-mode cache.
        self._detector = AprilTagDetector(detector_config)
        self._tag_cache = TagObservationCache(
            self._cache_path, mode="w", start_frame=start_frame, end_frame=end_frame
        )

    # ------------------------------------------------------------------
    # PrecomputePhase interface
    # ------------------------------------------------------------------

    def has_cache_hit(self) -> bool:
        return self._hit

    def set_frame_result_callback(
        self, callback: Optional[Callable[..., None]]
    ) -> None:
        """Register a callback for live per-frame AprilTag outputs."""
        self._frame_result_callback = callback

    def process_frame(
        self,
        frame_idx: int,
        crops: List[np.ndarray],
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        if self._hit:
            return
        if self._tag_cache is None:
            return

        if not crops:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[],
                centers_xy=[],
                corners=[],
                det_indices=[],
                hammings=[],
            )
            return

        observations = self._detector.detect_in_crops(
            crops, crop_offsets, det_indices=crop_det_indices
        )

        if self._frame_result_callback is not None:
            self._frame_result_callback(
                frame_idx,
                [obs.tag_id for obs in observations],
                [obs.center_xy for obs in observations],
                [obs.corners for obs in observations],
                [obs.det_index for obs in observations],
                [obs.hamming for obs in observations],
            )

        if not observations:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[],
                centers_xy=[],
                corners=[],
                det_indices=[],
                hammings=[],
            )
        else:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[obs.tag_id for obs in observations],
                centers_xy=[obs.center_xy for obs in observations],
                corners=[obs.corners for obs in observations],
                det_indices=[obs.det_index for obs in observations],
                hammings=[obs.hamming for obs in observations],
            )

    def finalize(self) -> Optional[str]:
        if self._hit:
            return str(self._cache_path)
        if self._tag_cache is None:
            return None
        try:
            self._tag_cache.save(
                metadata={
                    "start_frame": self._start_frame,
                    "end_frame": self._end_frame,
                    "video_path": self._video_path,
                }
            )
            self._tag_cache.close()
            self._tag_cache = None
            return str(self._cache_path)
        except Exception:
            logger.exception("AprilTagPrecomputePhase.finalize() failed")
            return None

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._tag_cache is not None:
            try:
                self._tag_cache.close()
            except Exception:
                pass
            self._tag_cache = None


# ---------------------------------------------------------------------------
# CNNPrecomputePhase
# ---------------------------------------------------------------------------


class CNNPrecomputePhase(PrecomputePhase):
    """Precompute phase that runs CNN identity classification on OBB crops.

    Supports multiple instances with different model configs and names (e.g.
    one for individual identity, another for age classification).
    """

    is_fatal = False

    def __init__(
        self,
        config: CNNIdentityConfig,
        model_path: str,
        cache_path,
        compute_runtime: str = "cpu",
        name: str = "cnn_identity",
        frame_result_callback: Optional[
            Callable[[int, List[ClassPrediction]], None]
        ] = None,
        ignore_existing_cache: bool = False,
    ) -> None:
        self.name = name
        self._cache_path = Path(cache_path)
        self._cfg = config
        self._hit = self._cache_path.exists() and not ignore_existing_cache
        self._closed = False
        self._frame_result_callback = frame_result_callback

        # accumulator for batching
        self._pending_crops: List[np.ndarray] = []
        self._pending_frame_idx: List[int] = []
        self._pending_det_ids: List[int] = []

        if not self._hit:
            if ignore_existing_cache and self._cache_path.exists():
                logger.info(
                    "Realtime workflow ignoring existing CNN cache (%s): %s",
                    self.name,
                    self._cache_path,
                )
                try:
                    self._cache_path.unlink()
                except OSError:
                    logger.warning(
                        "Failed to remove existing CNN cache before realtime regeneration: %s",
                        self._cache_path,
                    )
            self._backend = CNNIdentityBackend(
                config, model_path=model_path, compute_runtime=compute_runtime
            )
            self._cache = CNNIdentityCache(str(self._cache_path))
        else:
            self._backend = None
            self._cache = None

    def has_cache_hit(self) -> bool:
        return self._hit

    def set_frame_result_callback(
        self,
        callback: Optional[Callable[[int, List[ClassPrediction]], None]],
    ) -> None:
        """Register a callback for live per-frame CNN outputs."""
        self._frame_result_callback = callback

    def _flush_frame_batch(
        self,
        frame_idx: int,
        frame_crops: List[np.ndarray],
        frame_det_ids: List[int],
    ) -> None:
        """Run one frame's CNN crops immediately for realtime/live consumers."""
        if not frame_crops or self._backend is None or self._cache is None:
            return

        frame_preds: List[ClassPrediction] = []
        batch_size = max(1, int(self._cfg.batch_size))
        for chunk_start in range(0, len(frame_crops), batch_size):
            chunk_end = min(chunk_start + batch_size, len(frame_crops))
            chunk_crops = frame_crops[chunk_start:chunk_end]
            chunk_det_ids = frame_det_ids[chunk_start:chunk_end]
            preds = self._backend.predict_batch(chunk_crops)
            for pred, det_id in zip(preds, chunk_det_ids):
                frame_preds.append(
                    ClassPrediction(
                        det_index=int(det_id),
                        factor_names=pred.factor_names,
                        class_names=pred.class_names,
                        confidences=pred.confidences,
                    )
                )

        self._cache.save(frame_idx, frame_preds)
        if self._frame_result_callback is not None:
            self._frame_result_callback(frame_idx, frame_preds)

    def process_frame(
        self,
        frame_idx: int,
        crops: List[np.ndarray],
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        if self._hit or self._cache is None:
            return
        if not crops:
            self._cache.save(frame_idx, [])
            if self._frame_result_callback is not None:
                self._frame_result_callback(frame_idx, [])
            return

        if self._frame_result_callback is not None:
            self._flush_frame_batch(
                frame_idx,
                list(crops),
                [int(det_idx) for det_idx in crop_det_indices],
            )
            return

        for crop, det_idx in zip(crops, crop_det_indices):
            self._pending_crops.append(crop)
            self._pending_frame_idx.append(frame_idx)
            self._pending_det_ids.append(det_idx)

        if len(self._pending_crops) >= self._cfg.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        if not self._pending_crops or self._backend is None or self._cache is None:
            return
        preds = self._backend.predict_batch(self._pending_crops)
        # group by frame
        frame_preds: Dict[int, List[ClassPrediction]] = {}
        for pred, frame_idx, det_id in zip(
            preds, self._pending_frame_idx, self._pending_det_ids
        ):
            fixed_pred = ClassPrediction(
                det_index=int(det_id),
                factor_names=pred.factor_names,
                class_names=pred.class_names,
                confidences=pred.confidences,
            )
            frame_preds.setdefault(frame_idx, []).append(fixed_pred)
        for fid, fps in frame_preds.items():
            self._cache.save(fid, fps)
            if self._frame_result_callback is not None:
                self._frame_result_callback(fid, fps)
        self._pending_crops.clear()
        self._pending_frame_idx.clear()
        self._pending_det_ids.clear()

    def finalize(self) -> Optional[str]:
        if self._hit:
            return str(self._cache_path)
        if self._backend is None or self._cache is None:
            return None
        self._flush_batch()  # flush any remaining partial batch
        self._cache.flush()  # single np.savez_compressed write
        logger.info("CNN identity cache written: %s", self._cache_path)
        return str(self._cache_path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None
