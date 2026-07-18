"""
Core tracking engine running in separate thread for real-time performance.
This is the main orchestrator, functionally identical to the original.
"""

import gc
import logging
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal, Slot

from hydra_suite.core.assigners.hungarian import TrackAssigner
from hydra_suite.core.filters.kalman import KalmanFilterManager
from hydra_suite.core.identity.geometry import (
    build_detection_direction_overrides as _pf_build_direction_overrides,
)
from hydra_suite.core.identity.geometry import (
    resolve_detection_tracking_theta as _pf_resolve_detection_tracking_theta,
)
from hydra_suite.core.identity.geometry import (
    resolve_tracking_theta as _pf_resolve_tracking_theta,
)
from hydra_suite.core.identity.pose.features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from hydra_suite.core.identity.pose.features import (
    compute_pose_geometry_from_keypoints as _pf_compute_geometry,
)
from hydra_suite.core.identity.pose.features import (
    is_pose_heading_reliable as _pf_heading_reliable,
)
from hydra_suite.core.identity.pose.features import (
    normalize_pose_keypoints as _pf_normalize_keypoints,
)
from hydra_suite.core.identity.pose.features import (
    resolve_pose_group_indices as _pf_resolve_indices,
)
from hydra_suite.core.tracking.confidence.density import get_density_region_flags
from hydra_suite.core.tracking.features.live_features import (
    LiveCNNIdentityStore,
    LivePosePropertiesStore,
    LiveTagObservationStore,
)
from hydra_suite.core.tracking.features.tag_features import (
    NO_TAG,
    build_detection_tag_id_list,
    build_tag_detection_hamming_map,
    build_tag_detection_map,
    get_detection_tag_csv_values,
)
from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.data.tag_observation_cache import TagObservationCache
from hydra_suite.utils.frame_prefetcher import FramePrefetcher
from hydra_suite.utils.geometry import estimate_detection_crop_quality
from hydra_suite.utils.video_artifacts import (
    build_detected_properties_cache_path,
    build_individual_properties_cache_path,
)
from hydra_suite.utils.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)

from hydra_suite.core.inference.cache.keys import (  # noqa: E402
    bgsub_detection_cache_key,
    video_signature,
    with_video_signature,
)
from hydra_suite.core.inference.cache.store import DetectionCacheHandle  # noqa: E402

# Task 18: USE_NEW_INFERENCE_PIPELINE feature flag removed — new InferenceRunner
# pipeline is now the permanent path.  The legacy env-var toggle has been dropped.
from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig  # noqa: E402
from hydra_suite.core.inference.runner import InferenceRunner  # noqa: E402
from hydra_suite.core.tracking.ingest.frame_result_bridge import (  # noqa: E402
    build_density_cache_dict,
    frame_result_to_meas,
    populate_live_cnn_store,
    populate_live_pose_store,
    populate_live_tag_store,
)


def should_build_bgsub_detection_cache(
    *, preview_mode: bool, backward_mode: bool
) -> bool:
    """Return True if a forward bg-sub run should read/write the shared detection cache.

    Preview Mode must not touch it: the cache file is a single fixed path per
    video (not qualified by frame range), and closing a handle always
    overwrites it with only the current run's frames — so a short preview
    range would silently truncate a full-range cache used by backward mode.
    """
    return not preview_mode


class TrackingWorker(QThread):
    """
    Core tracking engine. Orchestrates tracking components to be functionally
    identical to the original monolithic implementation.
    """

    frame_signal = Signal(np.ndarray)
    finished_signal = Signal(bool, list, list)
    progress_signal = Signal(int, str)
    stats_signal = Signal(dict)  # Real-time FPS/ETA stats
    warning_signal = Signal(str, str)  # (title, message) for UI warnings
    pose_exported_model_resolved_signal = Signal(str)

    def __init__(
        self,
        video_path,
        csv_writer_thread=None,
        video_output_path=None,
        backward_mode=False,
        detection_cache_path=None,
        preview_mode=False,
        use_cached_detections=False,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.csv_writer_thread = csv_writer_thread
        self.video_output_path = video_output_path
        self.backward_mode = backward_mode
        self.detection_cache_path = detection_cache_path
        self.preview_mode = preview_mode
        self.use_cached_detections = use_cached_detections
        self.video_writer = None
        self.params_mutex = QMutex()
        self.parameters = {}
        self.individual_properties_cache_path = None
        self.detected_properties_cache_path = None
        self.detected_cnn_cache_paths = {}
        # Stats tracking for FPS/ETA
        self.start_time = None
        self.frame_times = deque(maxlen=30)  # Keep last 30 frames for FPS calculation
        self._stop_requested = False

        # Internal state variables that helper methods depend on
        self.frame_count = 0
        self.trajectories_full = []

        # Confidence density regions (computed after pre-detection phase)
        self._density_regions = []

        # Frame prefetcher for async I/O
        self.frame_prefetcher = None
        self.frame_prefetcher = None

    def set_parameters(self: object, p: dict) -> object:
        """Set full tracking parameter dictionary in a thread-safe way."""
        self.params_mutex.lock()
        self.parameters = p
        self.params_mutex.unlock()

    @Slot(dict)
    def update_parameters(self: object, new_params: dict) -> object:
        """Slot to safely update parameters from the GUI thread."""
        self.params_mutex.lock()
        self.parameters = new_params
        self.params_mutex.unlock()
        logger.info("Tracking worker parameters updated.")

    def get_current_params(self: object) -> object:
        """Return a shallow copy of current tracking parameters."""
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p

    def _confidence_density_enabled(self, params=None) -> bool:
        """Return whether confidence-density mapping should be active for this run."""
        p = self.get_current_params() if params is None else params
        return bool(p.get("ENABLE_CONFIDENCE_DENSITY_MAP", True))

    def _confidence_density_video_export_enabled(self, params=None) -> bool:
        """Return whether density-map diagnostic video export should run."""
        p = self.get_current_params() if params is None else params
        return bool(p.get("EXPORT_CONFIDENCE_DENSITY_VIDEO", False))

    @staticmethod
    def _resolve_resized_roi_mask(
        roi_mask,
        target_w: int,
        target_h: int,
        cache_key=None,
        cached_mask=None,
    ):
        """Return a ROI mask resized for the current frame geometry.

        ROI geometry is static for a given run unless the user updates the ROI or
        the effective frame size changes. Cache the resized mask so the tracking
        loop does not repeatedly resample the same binary mask every frame.
        """
        if roi_mask is None:
            return None, None, False

        resolved_target_w = max(1, int(target_w))
        resolved_target_h = max(1, int(target_h))
        resolved_key = (id(roi_mask), resolved_target_w, resolved_target_h)
        if cache_key == resolved_key and cached_mask is not None:
            return cached_mask, cache_key, False

        if (
            roi_mask.shape[1] != resolved_target_w
            or roi_mask.shape[0] != resolved_target_h
        ):
            resized_mask = cv2.resize(
                roi_mask,
                (resolved_target_w, resolved_target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            resized_mask = roi_mask
        return resized_mask, resolved_key, True

    @staticmethod
    def _tracking_frame_resize_interpolation(
        detection_method: str,
        resize_factor: float,
    ) -> int:
        """Choose the live tracking resize interpolation policy.

        Realtime YOLO tracking downscales very large frames before the model does
        its own preprocessing and letterboxing. ``INTER_LINEAR`` is materially
        faster than ``INTER_AREA`` for this path and provides sufficient quality
        for detector input preparation.
        """
        if (
            str(detection_method or "").strip().lower() == "yolo_obb"
            and float(resize_factor) < 1.0
        ):
            return cv2.INTER_LINEAR
        return cv2.INTER_AREA

    @classmethod
    def _resize_tracking_frame(
        cls,
        frame,
        resize_factor: float,
        detection_method: str,
    ):
        """Resize one tracking frame using the live tracking interpolation policy."""
        if frame is None or float(resize_factor) >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (0, 0),
            fx=float(resize_factor),
            fy=float(resize_factor),
            interpolation=cls._tracking_frame_resize_interpolation(
                detection_method,
                resize_factor,
            ),
        )

    @staticmethod
    def _visualization_emit_stride(params: dict) -> int:
        """Return the GUI visualization stride for this run."""
        advanced = params.get("ADVANCED_CONFIG", {}) or {}
        default_stride = 1
        if bool(params.get("TRACKING_REALTIME_MODE", False)):
            default_stride = int(advanced.get("realtime_visualization_emit_stride", 1))
        else:
            default_stride = int(advanced.get("visualization_emit_stride", 1))
        return max(1, default_stride)

    @classmethod
    def _should_emit_visualization_frame(cls, frame_count: int, params: dict) -> bool:
        stride = cls._visualization_emit_stride(params)
        return int(frame_count) % stride == 0

    def _actual_frame_index_for_count(
        self,
        frame_count: int,
        start_frame: int,
        end_frame: int,
    ) -> int:
        if self.backward_mode:
            return int(end_frame) - (int(frame_count) - 1)
        return int(start_frame) + (int(frame_count) - 1)

    def _prepare_tracking_frame_entry(
        self,
        frame,
        frame_count: int,
        start_frame: int,
        end_frame: int,
        params: dict,
        individual_generator,
        roi_fill_color,
        cache_key=None,
        cached_mask=None,
        profiler=None,
    ):
        actual_frame_index = self._actual_frame_index_for_count(
            frame_count,
            start_frame,
            end_frame,
        )
        if self.backward_mode:
            if actual_frame_index < start_frame:
                return None, cache_key, cached_mask, roi_fill_color, False
        elif actual_frame_index > end_frame:
            return None, cache_key, cached_mask, roi_fill_color, False

        resize_f = params["RESIZE_FACTOR"]
        detection_method = params.get("DETECTION_METHOD", "background_subtraction")

        preprocessing_started = time.perf_counter()
        if frame is not None:
            if individual_generator:
                original_frame = frame if resize_f >= 1.0 else frame.copy()
            else:
                original_frame = None
        else:
            original_frame = None
        if profiler is not None:
            profiler.add_sample(
                "preprocessing",
                time.perf_counter() - preprocessing_started,
            )

        if frame is not None and resize_f < 1.0:
            resize_started = time.perf_counter()
            frame = self._resize_tracking_frame(frame, resize_f, detection_method)
            if profiler is not None:
                profiler.add_sample(
                    "frame_resize",
                    time.perf_counter() - resize_started,
                )

        roi_prepare_started = time.perf_counter()
        roi_mask_current = None
        roi_mask_changed = False
        roi_mask = params.get("ROI_MASK", None)
        if roi_mask is not None and frame is not None:
            target_w, target_h = frame.shape[1], frame.shape[0]
            (
                roi_mask_current,
                cache_key,
                roi_mask_changed,
            ) = self._resolve_resized_roi_mask(
                roi_mask,
                target_w,
                target_h,
                cache_key=cache_key,
                cached_mask=cached_mask,
            )
            cached_mask = roi_mask_current
            if roi_fill_color is None:
                mask_inv = roi_mask_current == 0
                outside_pixels = frame[mask_inv]
                if len(outside_pixels) > 0:
                    roi_fill_color = np.mean(outside_pixels, axis=0).astype(np.uint8)
                else:
                    roi_fill_color = np.array([0, 0, 0], dtype=np.uint8)
        if profiler is not None:
            profiler.add_sample(
                "roi_prepare",
                time.perf_counter() - roi_prepare_started,
            )

        return (
            {
                "frame": frame,
                "original_frame": original_frame,
                "frame_count": int(frame_count),
                "actual_frame_index": int(actual_frame_index),
                "ROI_mask_current": roi_mask_current,
            },
            cache_key,
            cached_mask,
            roi_fill_color,
            roi_mask_changed,
        )

    def stop(self: object) -> object:
        """Request cooperative stop for current processing loop."""
        self._stop_requested = True
        prefetcher = getattr(self, "frame_prefetcher", None)
        if prefetcher is not None:
            try:
                prefetcher.stop()
            except Exception:
                logger.debug("Failed to stop frame prefetcher", exc_info=True)

    def _forward_frame_iterator(self, cap, use_prefetcher=False):
        """Iterate through frames in forward direction.

        Args:
            cap: OpenCV VideoCapture object
            use_prefetcher (bool): Use frame prefetching for better I/O performance
        """
        frame_num = 0

        if use_prefetcher:
            # Use async frame prefetching for better performance
            self.frame_prefetcher = FramePrefetcher(cap, buffer_size=2)
            self.frame_prefetcher.start()

            while not self._stop_requested:
                ret, frame = self.frame_prefetcher.read()
                if not ret:
                    break
                frame_num += 1
                yield frame, frame_num

            self.frame_prefetcher.stop()
            self.frame_prefetcher = None
        else:
            # Standard synchronous frame reading
            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                yield frame, frame_num

    def _cached_detection_iterator(
        self, total_frames, start_frame=0, end_frame=None, backward=False
    ):
        """Iterate through frame indices for cached detection mode (no actual frames needed).

        Args:
            total_frames: Total number of frames to process
            start_frame: Starting frame index (0-based, actual video frame)
            end_frame: Ending frame index (0-based, actual video frame)
            backward: If True, iterate in reverse order (for backward tracking)
        """
        if end_frame is None:
            end_frame = start_frame + total_frames - 1

        if backward:
            # Backward mode: iterate from end_frame down to start_frame
            # This matches the cache keys which are actual video frame indices
            for relative_idx in range(total_frames):
                if self._stop_requested:
                    break
                yield None, relative_idx + 1  # Return None for frame, 1-indexed count
        else:
            # Forward cached mode: iterate from start_frame to end_frame
            for relative_idx in range(total_frames):
                if self._stop_requested:
                    break
                yield None, relative_idx + 1  # Return None for frame, 1-indexed count

    def emit_frame(self: object, bgr: object) -> object:
        """Emit current frame to GUI in RGB format."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb)

    def _build_individual_properties_cache_path(
        self, properties_id: str, start_frame: int, end_frame: int
    ) -> Path:
        """Build deterministic path for individual-properties cache artifact."""
        return build_individual_properties_cache_path(
            self.video_path,
            properties_id,
            start_frame,
            end_frame,
            detection_cache_path=self.detection_cache_path,
        )

    def _flush_live_pose_cache(
        self,
        live_store,
        keypoint_names,
        params,
        start_frame,
        end_frame,
    ) -> None:
        """Persist an in-memory pose store to a file-backed properties cache.

        The InferenceRunner path populates ``LivePosePropertiesStore`` per frame
        for the in-loop direction override but, unlike detected-properties, never
        wrote it to disk — so the rich-export merge (which reads
        ``individual_properties_cache_path``) had no source and the final CSV
        carried no pose columns. This flush mirrors the detected-properties save:
        it writes one ``IndividualPropertiesCache`` keyed by frame + detection ID
        and records ``individual_properties_cache_path`` for the GUI handoff.
        """
        from hydra_suite.core.identity.properties.cache import (
            IndividualPropertiesCache,
            compute_detection_hash,
            compute_extractor_hash,
            compute_filter_settings_hash,
            compute_individual_properties_id,
        )

        frames = list(live_store.get_cached_frames())
        if not frames:
            logger.info("No live pose frames to persist; skipping pose cache flush.")
            return

        detection_hash = compute_detection_hash(
            params.get("INFERENCE_MODEL_ID", ""),
            self.video_path,
            start_frame,
            end_frame,
            detection_cache_version="2.0",
        )
        filter_hash = compute_filter_settings_hash(params)
        extractor_hash = compute_extractor_hash(params)
        properties_id = compute_individual_properties_id(
            detection_hash, filter_hash, extractor_hash
        )
        pose_cache_path = self._build_individual_properties_cache_path(
            properties_id, start_frame, end_frame
        )
        cache = IndividualPropertiesCache(str(pose_cache_path), mode="w")
        try:
            for frame_idx in frames:
                raw = live_store.get_raw_frame(frame_idx)
                if raw is None:
                    continue
                cache.add_frame(
                    int(frame_idx),
                    raw.get("detection_ids", []),
                    pose_keypoints=raw.get("pose_keypoints", []),
                )
            cache.save(
                metadata={
                    "individual_properties_id": properties_id,
                    "detection_hash": detection_hash,
                    "filter_settings_hash": filter_hash,
                    "extractor_hash": extractor_hash,
                    "pose_keypoint_names": [str(k) for k in (keypoint_names or [])],
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "video_path": str(Path(self.video_path).expanduser().resolve()),
                }
            )
        finally:
            cache.close()

        self.individual_properties_cache_path = str(pose_cache_path)
        params["INDIVIDUAL_PROPERTIES_ID"] = properties_id
        params["INDIVIDUAL_PROPERTIES_CACHE_PATH"] = str(pose_cache_path)
        logger.info(
            "Persisted %d live pose frames to properties cache: %s",
            len(frames),
            pose_cache_path,
        )

    def _build_detected_properties_cache_path(
        self, properties_id: str, start_frame: int, end_frame: int
    ) -> Path:
        """Build deterministic path for detected-properties export cache."""
        return build_detected_properties_cache_path(
            self.video_path,
            properties_id,
            start_frame,
            end_frame,
            detection_cache_path=self.detection_cache_path,
        )

    @staticmethod
    def _should_precompute_individual_data(params: dict, detection_method: str) -> bool:
        """Return True when individual-data precompute should run."""
        if detection_method != "yolo_obb":
            return False
        return bool(params.get("ENABLE_POSE_EXTRACTOR", False))

    _estimate_detection_crop_quality = staticmethod(estimate_detection_crop_quality)

    @staticmethod
    def _resolve_tracking_theta(
        track_idx,
        measured_theta,
        pose_directed,
        orientation_last,
        fallback_theta=None,
    ):
        """Resolve directed vs axis-aligned orientation consistently for one track."""
        return _pf_resolve_tracking_theta(
            track_idx, measured_theta, pose_directed, orientation_last, fallback_theta
        )

    @staticmethod
    def _heading_source_for_detection(
        pose_is_directed: bool,
        headtail_is_directed: bool,
        pose_overrides_headtail: bool,
    ) -> str:
        if pose_overrides_headtail:
            if pose_is_directed:
                return "pose"
            if headtail_is_directed:
                return "headtail"
            return "obb_axis"
        if headtail_is_directed:
            return "headtail"
        if pose_is_directed:
            return "pose"
        return "obb_axis"

    @staticmethod
    def _normalize_theta(theta):
        """Compatibility wrapper for legacy tests and call sites."""
        from hydra_suite.core.identity import geometry as _geom

        normalize = getattr(_geom, "normalize_theta", None)
        if normalize is None:
            return float(theta) % (2 * math.pi)
        return normalize(theta)

    @staticmethod
    def _circular_abs_diff_rad(a, b):
        """Compatibility wrapper for legacy tests and call sites."""
        from hydra_suite.core.identity import geometry as _geom

        diff = getattr(_geom, "circular_abs_diff_rad", None)
        if diff is not None:
            return diff(a, b)
        delta = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
        return abs(delta)

    @staticmethod
    def _collapse_obb_axis_theta(theta_axis, reference_theta):
        """Compatibility wrapper for legacy tests and call sites."""
        from hydra_suite.core.identity import geometry as _geom

        collapse = getattr(_geom, "collapse_obb_axis_theta", None)
        if collapse is not None:
            return collapse(theta_axis, reference_theta)

        theta0 = TrackingWorker._normalize_theta(theta_axis)
        theta1 = TrackingWorker._normalize_theta(theta0 + math.pi)
        if reference_theta is None:
            return theta0
        try:
            ref = TrackingWorker._normalize_theta(float(reference_theta))
        except Exception:
            return theta0
        d0 = TrackingWorker._circular_abs_diff_rad(theta0, ref)
        d1 = TrackingWorker._circular_abs_diff_rad(theta1, ref)
        return theta0 if d0 <= d1 else theta1

    def _build_cnn_identity_cache_path(
        self, label: str, classify_id: str, start_frame: int, end_frame: int
    ):
        """Build an independent, hash-keyed classify cache path."""
        from hydra_suite.utils.video_artifacts import build_classify_cache_path

        return str(
            build_classify_cache_path(
                self.video_path,
                classify_id,
                label,
                start_frame,
                end_frame,
                artifact_base_dir=(
                    Path(self.detection_cache_path).parent
                    if self.detection_cache_path
                    else None
                ),
                create_dir=True,
            )
        )

    def _run_batched_detection_phase(
        self,
        cap,
        detection_cache,
        detector,
        params,
        start_frame,
        end_frame,
        profiler=None,
    ):
        """Phase 1: Run batched YOLO detection and cache results."""
        from hydra_suite.core.tracking.ingest.detection_phase import (
            run_batched_detection_phase,
        )

        return run_batched_detection_phase(
            cap,
            detection_cache,
            detector,
            params,
            start_frame,
            end_frame,
            is_stop_requested=lambda: self._stop_requested,
            on_progress=lambda pct, msg: self.progress_signal.emit(pct, msg),
            on_stats=lambda stats: self.stats_signal.emit(stats),
            profiler=profiler,
            video_path=self.video_path,
        )

    def run(self: object) -> object:
        """QThread entry point: delegate to ``_run_impl``, guaranteeing ``finished_signal``.

        ``_run_impl`` is a large, deeply nested pipeline (video decode,
        detection, tracking, post-processing) with many early-return error
        paths, but no blanket exception handler. PySide6 silently swallows
        any exception that escapes a QThread's ``run()`` override (it prints
        "Error calling Python override of QThread::run()" and returns), so
        without this wrapper an unexpected exception anywhere in
        ``_run_impl`` — e.g. an NVDec hardware-decode failure raised lazily
        on the first frame read for a clip whose resolution exceeds the
        GPU's macroblock limit — leaves ``finished_signal`` never emitted.
        Callers that block on it (e.g. headless_tracking.py's
        ``QEventLoop.exec()``) then hang forever instead of surfacing a
        clean error.
        """
        try:
            self._run_impl()
        except Exception:
            logger.exception(
                "Unhandled exception in TrackingWorker.run(); emitting "
                "finished_signal(False, ...) so callers waiting on it "
                "(e.g. the headless CLI's QEventLoop) don't hang forever."
            )
            self.finished_signal.emit(False, [], [])

    def _run_impl(self: object) -> object:  # noqa: C901
        """Execute tracking pipeline for the configured video and parameters."""
        # === 1. INITIALIZATION (Identical to Original) ===
        gc.collect()
        self._stop_requested = False
        p = self.get_current_params()

        # Create profiler early so initialization timing is captured.
        # The profiler is configured with metadata later, once all params are known.
        _profiling_enabled = bool(p.get("ENABLE_PROFILING", False))
        profiler = TrackingProfiler(enabled=_profiling_enabled)
        profiler.phase_start("initialization")

        density_map_enabled = self._confidence_density_enabled(p)
        if not density_map_enabled:
            self._density_regions = []

        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            self.finished_signal.emit(True, [], [])
            return

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_video_frames <= 0:
            total_video_frames = None

        # Get frame range parameters early (before video writer init)
        start_frame = p.get("START_FRAME", 0)
        end_frame = p.get("END_FRAME", None)
        if end_frame is None:
            end_frame = total_video_frames - 1 if total_video_frames else 0

        # Validate frame range
        if total_video_frames:
            start_frame = max(0, min(start_frame, total_video_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_video_frames - 1))

        # Set total_frames to the range we'll actually process
        total_frames = end_frame - start_frame + 1

        logger.info(f"Video has {total_video_frames} frames total")
        logger.info(
            f"Processing frame range: {start_frame} to {end_frame} ({total_frames} frames)"
        )

        if self.video_output_path:
            fps, resize_f = cap.get(cv2.CAP_PROP_FPS), p.get("RESIZE_FACTOR", 1.0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_f), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_f
            )
            out_path = self.video_output_path
            if self.backward_mode:
                base, ext = os.path.splitext(out_path)
                out_path = f"{base}_backward{ext}"
            self.video_writer = VideoEncoder(out_path, fps=fps, width=w, height=h)

        # Determine if we should use batched detection
        # Batching is only used for YOLO in full tracking mode (not preview, not backward)
        detection_method = p.get("DETECTION_METHOD", "background_subtraction")
        advanced_config = p.get("ADVANCED_CONFIG", {})
        realtime_tracking_mode_requested = bool(
            p.get(
                "TRACKING_REALTIME_MODE",
                str(p.get("TRACKING_WORKFLOW_MODE", "non_realtime")).strip().lower()
                == "realtime",
            )
        )
        use_batched_detection = (
            not self.preview_mode  # Not preview mode
            and not self.backward_mode  # Not backward mode (uses cache)
            and detection_method == "yolo_obb"  # Only YOLO benefits from batching
            and not realtime_tracking_mode_requested
            and advanced_config.get(
                "enable_yolo_batching", True
            )  # Batching enabled in config
            and self.detection_cache_path
            is not None  # Need cache path for two-phase approach
        )

        if use_batched_detection:
            logger.info("Using batched YOLO detection (two-phase approach)")
        elif detection_method == "yolo_obb" and not self.preview_mode:
            logger.info("Using frame-by-frame YOLO detection")

        # Background model priming (formerly done here) now lives inside the
        # InferenceRunner bg-sub stage: load_bgsub_model primes the model from
        # the video when the runner is constructed below. Backward mode replays
        # cached detections and never constructs a runner.

        # Seek to start frame if not at beginning
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Seeking to start frame {start_frame}")

        individual_pipeline_enabled = bool(
            p.get(
                "ENABLE_INDIVIDUAL_PIPELINE", p.get("ENABLE_IDENTITY_ANALYSIS", False)
            )
        )
        # Individual analysis is YOLO-only in this phase. Keep tracking behavior
        # unchanged for background subtraction and skip analysis outputs there.
        if individual_pipeline_enabled and detection_method != "yolo_obb":
            msg = (
                "Individual analysis requires YOLO OBB mode. "
                "Background subtraction mode runs tracking without individual-analysis outputs."
            )
            logger.info(msg)
            self.warning_signal.emit("Individual Analysis Disabled", msg)
            individual_pipeline_enabled = False

        # Whether any precompute phase will be needed (pose, AprilTag, CNN identity).
        # Used to gate detection-cache requirements and force two-phase YOLO detection.
        individual_data_precompute_enabled = bool(
            not self.backward_mode
            and not self.preview_mode
            and detection_method == "yolo_obb"
            and (
                bool(p.get("ENABLE_POSE_EXTRACTOR", False))
                or bool(p.get("USE_APRILTAGS", False))
                or bool(p.get("CNN_CLASSIFIERS", []))
            )
        )
        # === Streaming Phase 5/6 ===
        # Forward YOLO runs perform individual analysis (pose, AprilTag, CNN
        # identity) inline via the InferenceRunner streaming path.
        _streaming_explicitly_requested = bool(
            p.get("ENABLE_STREAMING_INDIVIDUAL_ANALYSIS", False)
        )
        streaming_precompute_enabled = bool(
            individual_data_precompute_enabled
            and not self.preview_mode
            and not self.backward_mode
        )
        effective_realtime_tracking_mode = bool(
            realtime_tracking_mode_requested
            and not self.preview_mode
            and not self.backward_mode
        )
        if streaming_precompute_enabled:
            if use_batched_detection:
                use_batched_detection = False
                logger.info(
                    "Disabling batched YOLO prepass because streaming individual analysis is active."
                )
            logger.info(
                "Streaming-first individual analysis enabled inside the forward loop."
            )
            if (
                not effective_realtime_tracking_mode
                and not _streaming_explicitly_requested
            ):
                logger.info(
                    "Non-realtime YOLO forward runs now default to streaming individual analysis."
                )
        elif effective_realtime_tracking_mode:
            logger.info(
                "Realtime workflow enabled: using streaming forward detection/tracking."
            )

        # For yolo_obb, InferenceRunner manages its own cache dir via _resolve_cache_dir()
        # so individual precompute can proceed without a legacy detection_cache_path.
        if (
            individual_data_precompute_enabled
            and not self.detection_cache_path
            and detection_method != "yolo_obb"
        ):
            logger.error(
                "Individual precompute requires detection caching, but no detection cache path is configured."
            )
            cap.release()
            self.finished_signal.emit(False, [], [])
            return

        # Replay/precompute fallback needs full raw detections before tracking
        # starts. Fresh forward runs stay on the streaming path by default.
        if (
            individual_data_precompute_enabled
            and not use_batched_detection
            and not streaming_precompute_enabled
        ):
            use_batched_detection = True
            logger.info("Enabling batched YOLO prepass for replay precompute fallback.")

        # Final canonical stills and oriented videos are exported only after
        # backward tracking and post-processing complete.
        individual_generator = None

        self.kf_manager = KalmanFilterManager(p["MAX_TARGETS"], p)
        assigner = TrackAssigner(p, worker=self)

        N = p["MAX_TARGETS"]
        # Start all tracks as "lost" so the free_dets loop bootstraps each slot
        # from the first frame's real detections via initialize_filter.  Starting
        # as "active" with a zero-initialised KF state causes every track to sit
        # at (0, 0) = top-left corner for LOST_THRESHOLD_FRAMES frames before it
        # can be properly placed — producing a warm-up gap that the optimizer and
        # TrackingPreviewWorker do not have, and causing parameter divergence.
        track_states, missed_frames = ["lost"] * N, [0] * N
        self.trajectories_full = [[] for _ in range(N)]
        trajectories_pruned = [[] for _ in range(N)]
        position_deques = [
            deque(maxlen=2) for _ in range(N)
        ]  # entries: (x, y, frame_count)
        orientation_last, last_shape_info = [None] * N, [None] * N
        track_pose_prototypes = [None] * N
        track_avg_step = [0.0] * N
        tracking_continuity = [0] * N
        trajectory_ids, next_trajectory_id = list(range(N)), N

        detection_initialized = False
        detection_counts = 0

        # Diagnostic: log gate parameters for debugging jumps
        _diag_body = float(
            p.get("REFERENCE_BODY_SIZE", 20.0) * p.get("RESIZE_FACTOR", 1.0)
        )
        _diag_max_dist = float(p.get("MAX_DISTANCE_THRESHOLD", 1000.0))
        _diag_vel_gate = (
            float(p.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)) * _diag_body
        )
        _diag_young_mult = float(p.get("KALMAN_YOUNG_GATE_MULTIPLIER", 1.0))
        _diag_maturity = int(p.get("KALMAN_MATURITY_AGE", 10))
        _diag_lost_thresh = int(p.get("LOST_THRESHOLD_FRAMES", 5))
        logger.info(
            f"Assignment gates: body={_diag_body:.1f}px "
            f"MAX_DIST={_diag_max_dist:.1f}px "
            f"VEL_GATE={_diag_vel_gate:.1f}px "
            f"young_mult={_diag_young_mult:.1f} "
            f"maturity={_diag_maturity} "
            f"lost_thresh={_diag_lost_thresh} "
            f"density_regions={len(self._density_regions)}"
        )

        start_time, self.frame_count, fps_list = time.time(), 0, []
        # Lighting stabilization state (intensity_history / lighting_state) now
        # lives on BackgroundModel inside the bg-sub stage, so the worker no
        # longer tracks it here.
        local_counts = [0] * N
        roi_fill_color = None  # Average color outside ROI for visualization overlay

        # Pipeline profiler — store run metadata now that all params are known,
        # and close the initialization phase.
        profiler.set_config(
            detection_method=detection_method,
            n_targets=N,
            resize_factor=float(p.get("RESIZE_FACTOR", 1.0)),
            compute_runtime=str(p.get("COMPUTE_RUNTIME", "cpu")),
            start_frame=start_frame,
            end_frame=end_frame,
            backward_mode=self.backward_mode,
            preview_mode=self.preview_mode,
            batched_detection=use_batched_detection,
            density_map_enabled=density_map_enabled,
            precompute_enabled=individual_data_precompute_enabled,
            pass_name="backward" if self.backward_mode else "forward",
        )
        profiler.phase_end("initialization")

        # Initialize detection.
        # For YOLO OBB: InferenceRunner owns detection, caching, and all per-frame
        # inference (headtail, CNN, pose, AprilTag).  The legacy DetectionCache is
        # not used for this path.
        # For background subtraction: InferenceRunner bg-sub stage drives forward
        # detection live (constructed below); backward replays a cache.
        inference_runner = None  # InferenceRunner for yolo_obb mode
        bgsub_runner = None  # InferenceRunner for background-subtraction mode
        detection_cache = None  # Legacy cache — only used for background subtraction
        # Refactor-native detection cache for background subtraction: forward writes
        # per-frame detections, backward replays them (parity with the OBB path).
        bgsub_detection_cache = None
        use_cached_detections = False
        cached_frame_indices = set()

        if detection_method == "yolo_obb":
            # ── YOLO OBB: InferenceRunner path ──────────────────────────────
            if self.backward_mode and not self.detection_cache_path:
                logger.error(
                    "Backward tracking requires a configured forward detection cache path. "
                    "Please run forward tracking first."
                )
                cap.release()
                self.finished_signal.emit(False, [], [])
                return

            try:
                _inference_cfg = self._build_inference_config_from_params(p)
            except Exception as _cfg_err:
                logger.error(
                    "Failed to build InferenceConfig from params: %s", _cfg_err
                )
                cap.release()
                self.finished_signal.emit(False, [], [])
                return

            _cache_dir = self._resolve_cache_dir()
            _cache_dir.mkdir(parents=True, exist_ok=True)
            # Backward (replay) passes only call load_frame / caches_all_valid —
            # they never invoke run_realtime or run_batch_pass.  Skip loading
            # HeadTail, CNN, Pose (incl. SLEAP), and AprilTag backends in that
            # mode to avoid the ~8 s per-session SLEAP/ORT-TRT-EP init cost.
            inference_runner = InferenceRunner(
                _inference_cfg,
                cache_dir=_cache_dir,
                video_path=self.video_path,
                cache_only=self.backward_mode,
            )

            if self.backward_mode:
                if not inference_runner.caches_all_valid():
                    logger.error(
                        "Backward tracking requires valid forward-pass inference caches. "
                        "Please run forward tracking first."
                    )
                    inference_runner.close()
                    cap.release()
                    self.finished_signal.emit(False, [], [])
                    return
                if not inference_runner.detection_cache_covers_range(
                    start_frame, end_frame
                ):
                    _missing = inference_runner.detection_cache_missing_frames(
                        start_frame, end_frame
                    )
                    logger.error(
                        "Backward tracking requires a forward-pass cache covering "
                        "frames %d-%d, but it is incomplete (missing e.g. %s). "
                        "Please re-run forward tracking over the full range.",
                        start_frame,
                        end_frame,
                        _missing,
                    )
                    inference_runner.close()
                    cap.release()
                    self.finished_signal.emit(False, [], [])
                    return
                use_cached_detections = True
                logger.info(
                    "Backward pass: using pre-computed InferenceRunner caches from %s",
                    _cache_dir,
                )
            elif (
                not effective_realtime_tracking_mode
                and self.use_cached_detections
                and inference_runner.caches_all_valid()
                and inference_runner.detection_cache_covers_range(
                    start_frame, end_frame
                )
            ):
                use_cached_detections = True
                logger.info(
                    "Reusing pre-computed InferenceRunner caches from %s", _cache_dir
                )
            else:
                use_cached_detections = False
                if not self.use_cached_detections:
                    logger.info(
                        "Cache reuse disabled by user; InferenceRunner will recompute "
                        "detections (realtime=%s)",
                        effective_realtime_tracking_mode,
                    )
                else:
                    logger.info(
                        "InferenceRunner will compute detections (realtime=%s)",
                        effective_realtime_tracking_mode,
                    )

            # Load density regions sidecar for backward pass
            if density_map_enabled and self.backward_mode and not self._density_regions:
                try:
                    from hydra_suite.core.tracking.confidence.confidence_density import (
                        load_regions as _load_regions,
                    )

                    _regions_path = _cache_dir / "confidence_regions.json"
                    if _regions_path.exists():
                        self._density_regions = _load_regions(_regions_path)
                        logger.info(
                            "Backward pass: loaded %d density regions from %s",
                            len(self._density_regions),
                            _regions_path,
                        )
                    else:
                        logger.info(
                            "Backward pass: no density regions sidecar at %s; "
                            "density-aware assignment disabled.",
                            _regions_path,
                        )
                except Exception:
                    logger.exception(
                        "Failed to load density regions for backward pass (non-fatal)"
                    )
                    self._density_regions = []

        else:
            # ── Background subtraction ───────────────────────────────────────
            # Detections are produced live in the forward loop (the adaptive
            # background model needs sequential frames, so there is no separate
            # batch pass), and cached via the refactor-native DetectionCacheHandle
            # so the backward pass can replay them — parity with the OBB path,
            # which caches through InferenceRunner.
            if should_build_bgsub_detection_cache(
                preview_mode=self.preview_mode, backward_mode=self.backward_mode
            ):
                _cache_dir = self._resolve_cache_dir()
                _cache_dir.mkdir(parents=True, exist_ok=True)
                _bgsub_cache_path = _cache_dir / "bgsub_detection.npz"
                _bgsub_key = with_video_signature(
                    bgsub_detection_cache_key(BgSubConfig.from_params(p)),
                    video_signature(self.video_path),
                )
                bgsub_detection_cache = DetectionCacheHandle(
                    path=_bgsub_cache_path, key=_bgsub_key
                )
            if self.backward_mode:
                if (
                    bgsub_detection_cache is None
                    or not bgsub_detection_cache.is_valid()
                ):
                    logger.error(
                        "Backward tracking requires valid forward bg-sub detections "
                        "at %s. Please run forward tracking first.",
                        self._resolve_cache_dir() / "bgsub_detection.npz",
                    )
                    cap.release()
                    self.finished_signal.emit(False, [], [])
                    return
                use_cached_detections = True
                logger.info(
                    "Backward pass: replaying cached bg-sub detections from %s",
                    _bgsub_cache_path,
                )
            else:
                # Forward bg-sub detection now runs through InferenceRunner's
                # bgsub stage (owns lighting stabilization + background update +
                # foreground mask + measure). Mirrors the yolo_obb branch's
                # construction/tier resolution, but with cache_dir=None: the
                # worker keeps its own DetectionCacheHandle (bgsub_detection_cache)
                # for the backward replay, so the runner must NOT also open a
                # detection cache. bg-sub is sequential, so run_realtime is driven
                # in frame order below — never run_batch_pass.
                from hydra_suite.core.inference.config import migrate_runtime_to_tier

                _compute_runtime = str(p.get("COMPUTE_RUNTIME", "cpu"))
                _raw_tier = str(p.get("RUNTIME_TIER", "") or "").strip().lower()
                _runtime_tier = (
                    _raw_tier
                    if _raw_tier in {"cpu", "gpu", "gpu_fast"}
                    else migrate_runtime_to_tier({_compute_runtime})
                )
                bgsub_inference_config = InferenceConfig(
                    obb=None,
                    bgsub=BgSubConfig.from_params(p),
                    runtime_tier=_runtime_tier,
                    detection_batch_size=int(p.get("DETECTION_BATCH_SIZE", 1) or 1),
                )
                bgsub_runner = InferenceRunner(
                    bgsub_inference_config,
                    cache_dir=None,
                    video_path=self.video_path,
                    cache_only=False,
                )
                if bgsub_detection_cache is not None:
                    logger.info(
                        "Forward pass caching bg-sub detections to %s",
                        _bgsub_cache_path,
                    )
                else:
                    logger.info(
                        "Preview mode: skipping bg-sub detection cache to avoid "
                        "truncating the full-range cache."
                    )

        # === RUN BATCHED INFERENCE PHASE (if applicable) ===
        # For YOLO OBB: InferenceRunner.run_batch_pass() when caches are not yet valid.
        # For background subtraction: no batched phase; legacy detector runs per frame.
        if (
            inference_runner is not None
            and not use_cached_detections
            and not effective_realtime_tracking_mode
        ):
            profiler.phase_start("batched_detection")
            logger.info("=" * 80)
            logger.info("PHASE 1: InferenceRunner batch pass")
            logger.info("=" * 80)
            try:
                inference_runner.run_batch_pass(
                    Path(self.video_path),
                    progress_cb=self._emit_inference_progress,
                    start_frame=int(p.get("START_FRAME", 0)),
                    end_frame=(
                        int(p.get("END_FRAME", -1))
                        if int(p.get("END_FRAME", -1)) >= 0
                        else None
                    ),
                    should_stop=lambda: self._stop_requested,
                )
            except Exception as _bp_err:
                profiler.phase_end("batched_detection")
                logger.exception(
                    "InferenceRunner batch pass failed (fatal): %s", _bp_err
                )
                self.warning_signal.emit(
                    "Inference Failed",
                    f"Batch detection pass failed:\n{_bp_err}",
                )
                inference_runner.close()
                cap.release()
                if self.video_writer:
                    self.video_writer.release()
                self.finished_signal.emit(False, [], [])
                return
            profiler.phase_end("batched_detection")
            use_cached_detections = True
            logger.info(
                "InferenceRunner batch pass complete; caches written to %s", _cache_dir
            )

            # Reset video capture for the tracking loop phase.
            cap.release()
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info("Reset video to start frame %d for tracking loop", start_frame)

            logger.info("=" * 80)
            logger.info("PHASE 2: Tracking and Visualization")
            logger.info("=" * 80)

        # === COMPUTE CONFIDENCE DENSITY MAP ===
        # Runs for BOTH fresh and cached detections (forward pass only).
        # Backward pass loads regions from the sidecar JSON instead.
        # For YOLO OBB: uses InferenceRunner detection cache via build_density_cache_dict.
        # For background subtraction: no cached detections → skip density map.
        if (
            density_map_enabled
            and not self.backward_mode
            and inference_runner is not None
            and use_cached_detections
        ):
            profiler.phase_start("confidence_density")

            _regions_path = _cache_dir / "confidence_regions.json"
            if _regions_path.exists():
                # Regions already computed — just load them.
                try:
                    from hydra_suite.core.tracking.confidence.confidence_density import (
                        load_regions as _load_regions,
                    )

                    self._density_regions = _load_regions(_regions_path)
                    logger.info(
                        "Loaded %d existing density regions from %s",
                        len(self._density_regions),
                        _regions_path,
                    )
                except Exception:
                    logger.exception(
                        "Failed to load existing density regions (non-fatal)"
                    )
                    self._density_regions = []
            else:
                # Compute density map from InferenceRunner detection cache.
                try:
                    import cv2 as _cv2

                    from hydra_suite.core.tracking.confidence.confidence_density import (
                        compute_density_map_from_cache,
                        export_diagnostic_video,
                        save_regions,
                    )

                    _frame_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                    _frame_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))

                    # Build {frame_idx: (meas_arr, confs_arr, sizes_arr)} from runner caches
                    _cache_dict = build_density_cache_dict(
                        inference_runner, start_frame, end_frame
                    )

                    def _density_progress(pct, msg):
                        logger.info(msg)
                        self.progress_signal.emit(pct, msg)

                    logger.info("Computing confidence density map...")
                    self.progress_signal.emit(0, "Computing confidence density map...")

                    # Compute min_area_px in grid-pixel units from body-size fraction.
                    _density_ds = int(p.get("DENSITY_DOWNSAMPLE_FACTOR", 8))
                    _body_size_px = float(p.get("REFERENCE_BODY_SIZE", 20.0)) * float(
                        p.get("RESIZE_FACTOR", 1.0)
                    )
                    _body_size_grid = _body_size_px / max(1, _density_ds)
                    _density_min_area_px = int(
                        float(p.get("DENSITY_MIN_AREA_BODIES", 0.25))
                        * _body_size_grid**2
                    )

                    _dm, _raw_grids = compute_density_map_from_cache(
                        detection_cache=_cache_dict,
                        frame_h=_frame_h,
                        frame_w=_frame_w,
                        sigma_scale=float(p.get("DENSITY_GAUSSIAN_SIGMA_SCALE", 1.0)),
                        temporal_sigma=float(p.get("DENSITY_TEMPORAL_SIGMA", 2.0)),
                        threshold=float(p.get("DENSITY_BINARIZE_THRESHOLD", 0.3)),
                        downsample_factor=int(p.get("DENSITY_DOWNSAMPLE_FACTOR", 8)),
                        min_frame_duration=int(p.get("DENSITY_MIN_FRAME_DURATION", 3)),
                        min_area_px=_density_min_area_px,
                        progress_callback=_density_progress,
                    )
                    self._density_regions = _dm.regions

                    save_regions(self._density_regions, _regions_path)
                    logger.info(
                        f"Density map: {len(self._density_regions)} regions, "
                        f"saved to {_regions_path}"
                    )

                    # Export diagnostic video at reduced resolution with
                    # sequential frame reading (avoids expensive random seeks
                    # on large videos).  Saved next to the source video so it
                    # is easy to find regardless of where the cache lives.
                    _diag_path = Path(self.video_path).parent / (
                        Path(self.video_path).stem + "_confidence_map.mp4"
                    )
                    _fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0

                    # Use a sequential reader: seek to start_frame so the
                    # diagnostic video only covers the selected subset.
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

                    def _diag_reader(_fidx, _cap=cap):
                        # Sequential read — _fidx is expected to increase
                        # monotonically.  Just grab the next frame.
                        _ok, _fr = _cap.read()
                        return _fr if _ok else None

                    if self._confidence_density_video_export_enabled(p):
                        logger.info("Exporting confidence density diagnostic video...")
                        self.progress_signal.emit(
                            50, "Exporting confidence density video..."
                        )

                        # Output at reduced resolution for speed.
                        _diag_ds = 4  # 4× downsample for diagnostic video (independent of density grid ds)
                        _out_w = max(1, _frame_w // _diag_ds)
                        _out_h = max(1, _frame_h // _diag_ds)

                        export_diagnostic_video(
                            frame_reader=_diag_reader,
                            n_frames=total_frames,
                            frame_h=_out_h,
                            frame_w=_out_w,
                            density_grids=_raw_grids,
                            regions=self._density_regions,
                            output_path=_diag_path,
                            fps=_fps,
                            output_scale=1.0 / _diag_ds,
                            binary_volume=_dm.binary_volume,
                            progress_callback=_density_progress,
                        )
                        logger.info(f"Diagnostic video exported: {_diag_path}")
                    else:
                        logger.info(
                            "Skipping confidence density diagnostic video export by configuration."
                        )
                    self.progress_signal.emit(100, "Density map complete")

                    # Reopen video capture for subsequent phases.
                    # CAP_PROP_POS_FRAMES seek is unreliable with some
                    # codecs after reading to EOF, so reopen instead.
                    cap.release()
                    cap = _cv2.VideoCapture(self.video_path, _cv2.CAP_FFMPEG)
                    if start_frame > 0:
                        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

                except Exception:
                    logger.exception(
                        "Confidence density map generation failed (non-fatal)"
                    )
                    self._density_regions = []
                    # Reopen video capture to guarantee clean state.
                    cap.release()
                    cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
                    if start_frame > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            profiler.phase_end("confidence_density")

        # === PER-FRAME INFERENCE LIVE STORES ===
        # For YOLO OBB with InferenceRunner: live stores are populated per-frame
        # inside the tracking loop from FrameResult.cnn / .pose / .apriltag.
        # The runner's batch pass has already cached all inference results.
        props_path = None
        tag_observation_cache_path = None
        live_feature_precompute = None  # kept for legacy bg-subtraction paths
        live_pose_props_cache = None
        live_pose_keypoint_names = []
        live_tag_obs_cache = None
        live_cnn_caches = {}

        # Live pose store: created whenever the InferenceRunner drives detections
        # (forward run_realtime OR backward load_frame). Backward mode sets
        # individual_data_precompute_enabled=False, but it still reads per-frame
        # keypoints from FrameResult.pose via load_frame and must populate this
        # store too. Otherwise pose-direction override is dead on the backward
        # pass and the merged final falls back to head-tail headings for every
        # detection (legacy reuses its on-disk pose cache on the backward pass,
        # so ~87% of its headings are pose-derived).
        _pose_live_store_enabled = (
            inference_runner is not None
            and not self.preview_mode
            and detection_method == "yolo_obb"
            and bool(p.get("ENABLE_POSE_EXTRACTOR", False))
        )
        if _pose_live_store_enabled:
            live_pose_props_cache = LivePosePropertiesStore()
            # Use the canonical skeleton loader (same as the pose stage) so both
            # the modern "keypoint_names"/"skeleton_edges" schema and the legacy
            # "keypoints"/"edges" aliases resolve. A brittle
            # `_skel.get("keypoints")` here left the names empty for the modern
            # schema, which silently disabled pose-direction override.
            if _inference_cfg.pose and _inference_cfg.pose.skeleton_file:
                try:
                    from hydra_suite.core.identity.pose.utils import (
                        load_skeleton_from_json,
                    )

                    _names, _ = load_skeleton_from_json(
                        _inference_cfg.pose.skeleton_file
                    )
                    live_pose_keypoint_names = [str(k) for k in _names]
                except Exception:
                    live_pose_keypoint_names = []
            logger.info(
                "Live pose store ready for InferenceRunner per-frame population."
            )

        if inference_runner is not None and individual_data_precompute_enabled:
            # Instantiate remaining live stores; populated per-frame from FrameResult.
            if bool(p.get("USE_APRILTAGS", False)):
                live_tag_obs_cache = LiveTagObservationStore()
                logger.info(
                    "Live tag store ready for InferenceRunner per-frame population."
                )

            # Build IdentityEvidenceEmitter per CNN phase so populate_live_cnn_store
            # can push calibrated full-catalog log_probs into the live store. The
            # online identity decoder reads these via load_evidences() to do
            # Bayesian updates; without them it can only commit identity on top-1
            # confidence and dramatically under-commits (regression observed:
            # 4/26 labels committed, 80/324 rows vs main's 14/26 and 267/329).
            cnn_evidence_emitters: dict = {}
            for cnn_cfg_dict in p.get("CNN_CLASSIFIERS", []):
                _cnn_label = str(cnn_cfg_dict.get("label", "cnn_identity"))
                _live_cnn_store = LiveCNNIdentityStore()
                live_cnn_caches[_cnn_label] = _live_cnn_store
                _ev_emitter = self._build_cnn_evidence_emitter(
                    cnn_cfg_dict, _live_cnn_store, p
                )
                if _ev_emitter is not None:
                    cnn_evidence_emitters[_cnn_label] = _ev_emitter
                    if not hasattr(self, "_evidence_emitters"):
                        self._evidence_emitters = []
                    self._evidence_emitters.append(_ev_emitter)
                logger.info(
                    "Live CNN store ready for InferenceRunner per-frame population (%s)",
                    _cnn_label,
                )

        # Open tag observation cache for reading during tracking loop.
        tag_obs_cache = live_tag_obs_cache
        if tag_obs_cache is not None:
            logger.info("Using live AprilTag observations for realtime tracking.")
        elif (
            not effective_realtime_tracking_mode
            and tag_observation_cache_path
            and os.path.exists(tag_observation_cache_path)
        ):
            tag_obs_cache = TagObservationCache(tag_observation_cache_path, mode="r")
            if not tag_obs_cache.is_compatible():
                logger.warning("Tag observation cache incompatible; ignoring.")
                tag_obs_cache.close()
                tag_obs_cache = None
            else:
                logger.info(
                    "Tag observation cache loaded for tracking: %s",
                    tag_observation_cache_path,
                )

        # Label map for detection-level tag CSV columns (AprilTag ID → label string).
        _tag_label_map_csv: dict = {}
        if tag_obs_cache is not None:
            _tag_label_map_csv = {
                idx: str(_lbl)
                for idx, _lbl in enumerate(p.get("TAG_IDENTITY_LABELS", []) or [])
                if str(_lbl).strip()
            }

        # Open CNN identity caches for reading during tracking loop (multi-phase).
        _cnn_phase_states = []
        for cnn_cfg_dict in p.get("CNN_CLASSIFIERS", []):
            label = str(cnn_cfg_dict.get("label", "cnn_identity"))
            model_path = str(cnn_cfg_dict.get("model_path", ""))
            model_labels = {
                str(_lbl)
                for _lbl in (cnn_cfg_dict.get("labels", []) or [])
                if str(_lbl).strip()
            }
            _cnpf_phase = cnn_cfg_dict.get("class_names_per_factor") or []
            _is_identity_provider = bool(cnn_cfg_dict.get("unique_identifier", False))
            live_or_path = live_cnn_caches.get(label)
            if isinstance(live_or_path, LiveCNNIdentityStore):
                _cnn_phase_states.append(
                    {
                        "label": label,
                        "cache": live_or_path,
                        "model_labels": model_labels,
                        "class_names_per_factor": _cnpf_phase,
                        "evidence_cache": None,
                        "is_identity_provider": _is_identity_provider,
                    }
                )
                logger.info(
                    "Using live CNN identity outputs for realtime tracking (%s)", label
                )
            elif not effective_realtime_tracking_mode:
                from hydra_suite.core.identity.calibration import CalibrationModel
                from hydra_suite.core.identity.properties.cache import (
                    compute_classify_cache_id,
                )

                _calibration_temperature = float(
                    cnn_cfg_dict.get(
                        "calibration_temperature",
                        cnn_cfg_dict.get("temperature", 1.0),
                    )
                )
                _calibration_signature = (
                    CalibrationModel(temperature=_calibration_temperature).signature
                    if abs(_calibration_temperature - 1.0) > 1e-6
                    else ""
                )

                classify_id = compute_classify_cache_id(
                    model_path=model_path,
                    compute_runtime=str(
                        p.get("CNN_COMPUTE_RUNTIME", p.get("COMPUTE_RUNTIME", "cpu"))
                    ),
                    inference_model_id=str(p.get("INFERENCE_MODEL_ID", "")),
                    calibration_signature=_calibration_signature,
                )
                _path = self._build_cnn_identity_cache_path(
                    label, classify_id, start_frame, end_frame
                )
                if _path and os.path.exists(_path):
                    from hydra_suite.core.identity.classification.cnn import (
                        CNNIdentityCache,
                    )
                    from hydra_suite.core.tracking.identity.evidence_emitter import (
                        build_evidence_cache_path,
                    )

                    _cache = CNNIdentityCache(_path)
                    _evidence_cache = None
                    try:
                        from hydra_suite.core.identity.cache import (
                            IdentityEvidenceCache,
                        )

                        for _signature in ("batch", "live", ""):
                            _ev_path = build_evidence_cache_path(
                                _path, label, _signature
                            )
                            if os.path.exists(str(_ev_path)):
                                _evidence_cache = IdentityEvidenceCache(
                                    str(_ev_path), mode="r"
                                )
                                break
                    except Exception:
                        logger.debug(
                            "Posterior sidecar unavailable for CNN phase '%s'",
                            label,
                            exc_info=True,
                        )
                    _cnn_phase_states.append(
                        {
                            "label": label,
                            "cache": _cache,
                            "model_labels": model_labels,
                            "class_names_per_factor": _cnpf_phase,
                            "evidence_cache": _evidence_cache,
                            "is_identity_provider": _is_identity_provider,
                        }
                    )
                    logger.info("CNN identity cache loaded (%s): %s", label, _path)

        # Optional pose-properties reader for directional orientation override.
        pose_props_cache = live_pose_props_cache
        pose_direction_enabled = False
        pose_direction_applied_count = 0
        pose_direction_fallback_count = 0
        pose_direction_anterior_indices = []
        pose_direction_posterior_indices = []
        pose_ignore_indices = []
        pose_keypoint_names = []
        pose_min_valid_conf = float(p.get("POSE_MIN_KPT_CONF_VALID", 0.2))
        pose_direction_min_visibility = float(
            np.clip(
                p.get(
                    "POSE_DIRECTION_MIN_VISIBILITY",
                    max(0.6, p.get("POSE_REJECTION_MIN_VISIBILITY", 0.5)),
                ),
                0.0,
                1.0,
            )
        )
        pose_direction_min_keypoints = max(
            1, int(p.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 3))
        )
        pose_frame_keypoints_map = {}
        pose_frame_keypoints_map_frame = None

        pose_cache_candidate = str(
            self.individual_properties_cache_path
            or p.get("INDIVIDUAL_PROPERTIES_CACHE_PATH", "")
            or ""
        ).strip()
        pose_extractor_enabled = bool(p.get("ENABLE_POSE_EXTRACTOR", False))
        if pose_props_cache is not None:
            pose_keypoint_names = [str(v) for v in live_pose_keypoint_names]
            pose_ignore_indices = _pf_resolve_indices(
                p.get("POSE_IGNORE_KEYPOINTS", []), pose_keypoint_names
            )
            pose_direction_anterior_indices = _pf_resolve_indices(
                p.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), pose_keypoint_names
            )
            pose_direction_posterior_indices = _pf_resolve_indices(
                p.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), pose_keypoint_names
            )
            if (
                len(pose_direction_anterior_indices) > 0
                and len(pose_direction_posterior_indices) > 0
            ):
                pose_direction_enabled = True
                logger.info(
                    "Pose direction override enabled from live pose outputs: anterior=%s, posterior=%s",
                    pose_direction_anterior_indices,
                    pose_direction_posterior_indices,
                )
        elif (
            not effective_realtime_tracking_mode
            and pose_extractor_enabled
            and detection_method == "yolo_obb"
            and pose_cache_candidate
            and os.path.exists(pose_cache_candidate)
        ):
            from hydra_suite.core.identity.properties.cache import (
                IndividualPropertiesCache,
            )

            pose_props_cache = IndividualPropertiesCache(pose_cache_candidate, mode="r")
            if not pose_props_cache.is_compatible():
                logger.warning(
                    "Pose direction override disabled: incompatible properties cache: %s",
                    pose_cache_candidate,
                )
                pose_props_cache.close()
                pose_props_cache = None
            else:
                names = pose_props_cache.metadata.get("pose_keypoint_names", [])
                if isinstance(names, (list, tuple)):
                    pose_keypoint_names = [str(v) for v in names]
                pose_ignore_indices = _pf_resolve_indices(
                    p.get("POSE_IGNORE_KEYPOINTS", []), pose_keypoint_names
                )
                pose_direction_anterior_indices = _pf_resolve_indices(
                    p.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), pose_keypoint_names
                )
                pose_direction_posterior_indices = _pf_resolve_indices(
                    p.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), pose_keypoint_names
                )
                if (
                    len(pose_direction_anterior_indices) > 0
                    and len(pose_direction_posterior_indices) > 0
                ):
                    pose_direction_enabled = True
                    logger.info(
                        "Pose direction override enabled: anterior=%s, posterior=%s",
                        pose_direction_anterior_indices,
                        pose_direction_posterior_indices,
                    )
                else:
                    logger.info(
                        "Pose direction override disabled: define both anterior/posterior keypoint groups."
                    )

        from hydra_suite.core.identity.properties.cache import (
            compute_detection_hash,
            compute_extractor_hash,
            compute_filter_settings_hash,
            compute_individual_properties_id,
        )
        from hydra_suite.core.identity.properties.detected_cache import (
            DetectedPropertiesCache,
        )

        detected_props_id = compute_individual_properties_id(
            compute_detection_hash(
                p.get("INFERENCE_MODEL_ID", ""),
                self.video_path,
                start_frame,
                end_frame,
                detection_cache_version="2.4",
            ),
            compute_filter_settings_hash(p),
            compute_extractor_hash(p),
        )
        if not self.preview_mode:
            self.detected_properties_cache_path = str(
                self._build_detected_properties_cache_path(
                    detected_props_id, start_frame, end_frame
                )
            )
        detected_props_cache = (
            None
            if self.preview_mode
            else DetectedPropertiesCache(self.detected_properties_cache_path, mode="w")
        )

        # === 2. FRAME PROCESSING LOOP ===
        # Determine whether to use frame prefetcher
        # Enable for forward passes where we're not batching detection (to avoid double buffering)
        # Prefetching is most beneficial when frame I/O competes with processing time
        use_prefetcher = (
            not use_batched_detection  # Not in batched detection phase 1
            and not self.backward_mode  # Not backward mode (uses cache iterator)
            and not self.preview_mode  # Not preview (latency-sensitive)
            and p.get("ENABLE_FRAME_PREFETCH", True)  # User hasn't disabled it
        )

        # Choose appropriate frame iterator
        if use_cached_detections:
            # Check if we are in forward mode (either Reuse or Batched Phase 2) or backward mode
            # If forward mode, we might need frames for visualization/video/dataset
            if not self.backward_mode:
                # Phase 2 of batched detection OR Cached Reuse: only read frames if we need visualization OR individual analysis
                # Update condition: Check for NOT visualization_free_mode (since ENABLE_VISUALIZATION isn't used)
                # Also check self.video_output_path (since ENABLE_VIDEO_OUTPUT isn't reliably in params)
                needs_frames = (
                    not p.get("VISUALIZATION_FREE_MODE", False)
                    or (self.video_output_path is not None)
                    or individual_generator
                    is not None  # Need frames for cropping individuals
                )

                if needs_frames:
                    frame_iterator = self._forward_frame_iterator(
                        cap, use_prefetcher=use_prefetcher
                    )
                    skip_visualization = False
                    logger.info(
                        "Forward Cached: Using cached detections with frame reading"
                    )
                else:
                    # No visualization or individual analysis - skip frame reading entirely
                    frame_iterator = self._cached_detection_iterator(
                        total_frames, start_frame, end_frame, backward=False
                    )
                    skip_visualization = True
                    use_prefetcher = False
                    logger.info(
                        "Forward Cached: Skipping frame reading (no visualization/analysis needed, using cached detections)"
                    )
            else:
                # Backward pass: no frames needed, skip visualization
                frame_iterator = self._cached_detection_iterator(
                    total_frames, start_frame, end_frame, backward=True
                )
                skip_visualization = True
                use_prefetcher = False  # No frames to prefetch
                logger.info(
                    "Backward pass: Skipping frame reading and visualization for maximum speed"
                )
        else:
            # Standard frame-by-frame with detection
            frame_iterator = self._forward_frame_iterator(
                cap, use_prefetcher=use_prefetcher
            )
            skip_visualization = False

        if use_prefetcher:
            logger.info("Frame prefetching ENABLED (background I/O buffering)")
        else:
            logger.info("Frame prefetching disabled")

        # Pre-compute ROI contours once (the mask is static for the entire run).
        _roi_contours_cache = None
        _roi_mask_cache_key = None
        _roi_mask_resized = None

        # === Identity Overhaul Phase 1: Online Identity Decoder ===
        # Instantiate the decoder whenever individual_pipeline_enabled is set and labels exist.
        # The decoder runs after each frame's geometric assignment to maintain
        # per-slot probabilistic identity beliefs and enforce the uniqueness
        # constraint across visible tracks.  The catalog is built from the
        # union of all configured CNN classifier label sets.
        _identity_online_decoder = None
        _identity_online_assignments = {}  # slot_index → IdentityAssignment
        _identity_catalog = None
        _identity_in_tracking_enabled = bool(p.get("ENABLE_IDENTITY_IN_TRACKING", True))
        if individual_pipeline_enabled and _identity_in_tracking_enabled:
            try:
                # Collect all known label candidates from CNN + tag configurations.
                # For multihead (multi-factor) classifiers, build composite catalog
                # entries from the cartesian product of factor class names so the
                # uniqueness constraint operates on whole-individual identities rather
                # than single-factor labels.
                import itertools as _itertools

                from hydra_suite.core.identity.catalog import IdentityCatalog
                from hydra_suite.core.identity.online import OnlineIdentityDecoder

                _known_labels_set: list[str] = []
                for _cnn_cfg in p.get("CNN_CLASSIFIERS", []):
                    if not bool(_cnn_cfg.get("unique_identifier", False)):
                        continue
                    # Resolve per-factor class names: prefer stored field, else read model file.
                    _cnpf_cfg: list[list[str]] = (
                        _cnn_cfg.get("class_names_per_factor") or []
                    )
                    if not _cnpf_cfg:
                        # Try to read from the model file.
                        try:
                            import json as _json

                            _mp = str(_cnn_cfg.get("model_path", ""))
                            if _mp and os.path.exists(_mp):
                                with open(_mp) as _mf:
                                    _mmeta = _json.load(_mf)
                                _cnpf_cfg = _mmeta.get("class_names_per_factor") or []
                                if not _cnpf_cfg:
                                    _flat = _mmeta.get("class_names") or []
                                    if _flat:
                                        _cnpf_cfg = [_flat]
                                if not _cnpf_cfg:
                                    for _fe in _mmeta.get("factor_models") or []:
                                        _fl = _fe.get("class_names") or []
                                        if _fl:
                                            _cnpf_cfg.append(_fl)
                        except Exception:
                            pass

                    _non_empty = [fl for fl in _cnpf_cfg if fl]
                    if len(_non_empty) > 1:
                        # Multi-factor: composite labels (cartesian product).
                        for _combo in _itertools.product(*_non_empty):
                            _composite = "_".join(str(c) for c in _combo if c)
                            if _composite and _composite not in _known_labels_set:
                                _known_labels_set.append(_composite)
                    else:
                        # Single factor or flat labels list.
                        _clf_labels: list[str] = []
                        for _fl in _non_empty:
                            _clf_labels.extend([str(l) for l in _fl if l])
                        if not _clf_labels:
                            _clf_labels = [
                                str(l) for l in (_cnn_cfg.get("labels", []) or []) if l
                            ]
                        for _lbl in _clf_labels:
                            if _lbl and _lbl not in _known_labels_set:
                                _known_labels_set.append(_lbl)
                # Tag identities may also be provided via TAG_IDENTITY_LABELS.
                # Only accept tag labels that match CNN-derived classes when CNN
                # is configured — this blocks garbage composites (e.g. phase
                # names) from polluting the identity catalog and assignment.
                _cnn_derived = set(_known_labels_set)
                for _lbl in p.get("TAG_IDENTITY_LABELS", []):
                    _s = str(_lbl).strip() if _lbl else ""
                    if not _s:
                        continue
                    if _cnn_derived and _s not in _cnn_derived:
                        continue
                    if _s not in _known_labels_set:
                        _known_labels_set.append(_s)

                if _known_labels_set:
                    _identity_catalog = IdentityCatalog.from_labels(_known_labels_set)
                    _identity_online_decoder = OnlineIdentityDecoder(
                        _identity_catalog, p
                    )
                    logger.info(
                        "Identity online decoder enabled: catalog size=%d labels=%s",
                        _identity_catalog.size,
                        _identity_catalog.labels,
                    )
                else:
                    logger.info(
                        "Identity online decoder: no known labels configured; decoder disabled."
                    )
            except Exception as _dec_init_err:
                logger.warning(
                    "Identity online decoder init failed (non-fatal): %s",
                    _dec_init_err,
                )
                _identity_online_decoder = None

        def _online_identity_row_values(track_idx: int) -> list[object]:
            assignment = _identity_online_assignments.get(track_idx)
            belief = (
                _identity_online_decoder.get_belief(track_idx)
                if _identity_online_decoder is not None
                else None
            )

            label = ""
            catalog_index = float("nan")
            confidence = float("nan")
            margin = float("nan")
            entropy = float("nan")
            committed = 0
            evidence_sources = ""
            conflict_flag = 0
            slot_lock_label = ""

            if belief is not None:
                evidence_sources = ",".join(belief.last_evidence_sources)
                conflict_flag = 1 if belief.last_conflict_flag else 0
                slot_lock_label = belief.slot_lock_label or ""
                committed = 1 if belief.committed else 0
                if belief.committed_label:
                    label = belief.committed_label
                    catalog_index = float(belief.committed_index)
                    probs = np.exp(belief.log_posterior - np.max(belief.log_posterior))
                    probs /= np.clip(probs.sum(), 1e-300, None)
                    confidence = float(probs[int(belief.committed_index)])
                    entropy = float(
                        -np.sum(probs * np.log(np.clip(probs, 1e-300, None)))
                    )
                    known_probs = probs[1:]
                    if len(known_probs) >= 2:
                        top2 = np.partition(known_probs, -2)[-2:]
                        margin = float(top2[1] - top2[0])

            if assignment is not None:
                if assignment.label:
                    label = assignment.label
                    catalog_index = float(assignment.catalog_index)
                    confidence = float(assignment.confidence)
                margin = float(assignment.margin)
                entropy = float(assignment.entropy)
                committed = 1 if assignment.committed else committed

            return [
                catalog_index,
                label,
                confidence,
                margin,
                entropy,
                committed,
                evidence_sources,
                conflict_flag,
                slot_lock_label,
            ]

        def _remap_source_log_probs_to_catalog(
            log_probs: np.ndarray,
            source_labels: list[str] | tuple[str, ...] | None,
        ) -> np.ndarray:
            if _identity_catalog is None:
                return np.asarray(log_probs, dtype=np.float64)
            arr = np.asarray(log_probs, dtype=np.float64)
            if source_labels is None:
                if len(arr) == _identity_catalog.size:
                    out = arr.copy()
                    out -= np.logaddexp.reduce(out)
                    return out
                return _identity_catalog.known_uniform_log_prior()

            labels = tuple(str(label) for label in source_labels)
            if len(labels) != len(arr):
                return _identity_catalog.known_uniform_log_prior()

            probs = np.exp(arr - np.max(arr))
            probs /= np.clip(probs.sum(), 1e-300, None)
            remapped = np.full(_identity_catalog.size, 1e-300, dtype=np.float64)
            for src_idx, label in enumerate(labels):
                if not _identity_catalog.contains(label):
                    continue
                remapped[_identity_catalog.index_of(label)] += float(probs[src_idx])
            remapped /= np.clip(remapped.sum(), 1e-300, None)
            return np.log(np.clip(remapped, 1e-300, None))

        def _decoder_track_label(
            track_idx: int,
            allowed_labels: set[str] | None = None,
        ) -> str | None:
            if _identity_online_decoder is None:
                return None
            belief = _identity_online_decoder.get_belief(track_idx)
            if belief is None:
                return None
            label = belief.committed_label
            if not label:
                probs = np.exp(belief.log_posterior - np.max(belief.log_posterior))
                probs /= np.clip(probs.sum(), 1e-300, None)
                known_probs = probs[1:]
                if len(known_probs) > 0:
                    best_idx = int(np.argmax(known_probs)) + 1
                    if float(probs[best_idx]) >= float(
                        p.get("IDENTITY_DISPLAY_THRESHOLD", 0.6)
                    ):
                        label = _identity_catalog.label_of(best_idx)
            if not label:
                return None
            if allowed_labels and label not in allowed_labels:
                return None
            return label

        # Discard any frame-level state accumulated during Phase 1 (batched
        # detection uses tick/tock without end_frame), and clear interval
        # accumulators so pre-loop phase timings don't bleed into the first
        # periodic tracking-loop summary window.
        profiler.discard_frame_state()
        profiler.reset_interval()

        profiler.phase_start("tracking_loop")
        frame_iterator = iter(frame_iterator)

        while True:

            params = self.get_current_params()
            detection_method = params.get("DETECTION_METHOD", "background_subtraction")
            resize_f = params["RESIZE_FACTOR"]

            try:
                frame, _ = next(frame_iterator)
            except StopIteration:
                break

            self.frame_count += 1
            actual_frame_index = self._actual_frame_index_for_count(
                self.frame_count,
                start_frame,
                end_frame,
            )
            profiler.notify_frame_index(actual_frame_index)

            if self.backward_mode:
                if actual_frame_index < start_frame:
                    logger.info(
                        f"Reached start frame {start_frame}, stopping backward tracking"
                    )
                    break
            else:
                if actual_frame_index > end_frame:
                    logger.info(f"Reached end frame {end_frame}, stopping tracking")
                    break

            preprocessing_started = time.perf_counter()
            if frame is not None:
                if individual_generator:
                    original_frame = frame if resize_f >= 1.0 else frame.copy()
                else:
                    original_frame = None
            else:
                original_frame = None
            profiler.add_sample(
                "preprocessing",
                time.perf_counter() - preprocessing_started,
            )

            if frame is not None and resize_f < 1.0:
                resize_started = time.perf_counter()
                frame = self._resize_tracking_frame(
                    frame,
                    resize_f,
                    detection_method,
                )
                profiler.add_sample(
                    "frame_resize", time.perf_counter() - resize_started
                )

            ROI_mask = params.get("ROI_MASK", None)
            ROI_mask_current = None

            roi_prepare_started = time.perf_counter()
            if ROI_mask is not None:
                if frame is not None:
                    target_w, target_h = frame.shape[1], frame.shape[0]
                else:
                    base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    target_w = max(1, int(base_w * resize_f))
                    target_h = max(1, int(base_h * resize_f))

                (
                    ROI_mask_current,
                    _roi_mask_cache_key,
                    roi_mask_changed,
                ) = self._resolve_resized_roi_mask(
                    ROI_mask,
                    target_w,
                    target_h,
                    cache_key=_roi_mask_cache_key,
                    cached_mask=_roi_mask_resized,
                )
                _roi_mask_resized = ROI_mask_current
                if roi_mask_changed:
                    _roi_contours_cache = None

                if frame is not None and roi_fill_color is None:
                    mask_inv = ROI_mask_current == 0
                    outside_pixels = frame[mask_inv]
                    if len(outside_pixels) > 0:
                        roi_fill_color = np.mean(outside_pixels, axis=0).astype(
                            np.uint8
                        )
                    else:
                        roi_fill_color = np.array([0, 0, 0], dtype=np.uint8)
            profiler.add_sample(
                "roi_prepare",
                time.perf_counter() - roi_prepare_started,
            )

            detection_sample_started = time.perf_counter()
            profiler.tick("detection")

            # Initialize detection-related variables (in case no detection occurs)
            detection_ids = []
            raw_detection_ids = []
            filtered_obb_corners = []
            detection_confidences = []
            pose_directed_mask = np.zeros(0, dtype=np.uint8)
            detection_headtail_heading = np.full(0, np.nan, dtype=np.float32)
            headtail_directed_mask = np.zeros(0, dtype=np.uint8)
            raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
                [],
                [],
                [],
                [],
                [],
            )
            raw_heading_hints = []
            raw_heading_confidences = []
            raw_directed_mask = []
            # Initialized here so the streaming-payload block can reference them
            # via the "x in locals()" guard on the background-subtraction path.
            filtered_heading_hints: list = []
            filtered_heading_confidences: list = []
            filtered_directed_mask: list = []
            yolo_results = None
            fg_mask = None
            bg_u8 = None
            _current_frame_result = (
                None  # FrameResult from InferenceRunner (yolo_obb path)
            )

            # Get detections either from cache or by detection
            if use_cached_detections and inference_runner is not None:
                # Load per-frame results from InferenceRunner caches (YOLO OBB path).
                # All filtering (ROI, confidence, IOU) was applied during the batch pass.
                _frame_result = inference_runner.load_frame(actual_frame_index)
                if _frame_result is not None and _frame_result.obb.num_detections > 0:
                    _obb = _frame_result.obb
                    # meas carries the OBB axis angle (legacy convention, [0, pi));
                    # downstream resolve_tracking_theta picks between theta and
                    # theta+pi using motion history + headtail heading_hints.
                    meas = frame_result_to_meas(_obb.centroids, _obb.angles)
                    sizes = [float(_obb.sizes[i]) for i in range(_obb.num_detections)]
                    shapes = [
                        (float(_obb.shapes[i, 0]), float(_obb.shapes[i, 1]))
                        for i in range(_obb.num_detections)
                    ]
                    detection_confidences = [
                        float(_obb.confidences[i]) for i in range(_obb.num_detections)
                    ]
                    filtered_obb_corners = [
                        _obb.corners[i] for i in range(_obb.num_detections)
                    ]
                    detection_ids = [
                        int(_obb.detection_ids[i]) for i in range(_obb.num_detections)
                    ]
                    raw_detection_ids = detection_ids
                    raw_meas = meas
                    raw_sizes = sizes
                    raw_shapes = shapes
                    raw_confidences = detection_confidences
                    raw_obb_corners = filtered_obb_corners
                    if _frame_result.headtail is not None:
                        detection_headtail_heading = np.asarray(
                            _frame_result.headtail.heading_hints, dtype=np.float32
                        )
                        detection_headtail_confidence = np.asarray(
                            _frame_result.headtail.heading_confidences, dtype=np.float32
                        )
                        headtail_directed_mask = np.asarray(
                            _frame_result.headtail.directed_mask, dtype=np.uint8
                        )
                        raw_heading_hints = list(detection_headtail_heading)
                        raw_heading_confidences = list(detection_headtail_confidence)
                        raw_directed_mask = list(headtail_directed_mask)
                    else:
                        detection_headtail_heading = np.asarray([], dtype=np.float32)
                        detection_headtail_confidence = np.asarray([], dtype=np.float32)
                        headtail_directed_mask = np.asarray([], dtype=np.uint8)
                        raw_heading_hints = []
                        raw_heading_confidences = []
                        raw_directed_mask = []
                    raw_canonical_affines = None
                    # Store FrameResult for Site F (live store population)
                    _current_frame_result = _frame_result
                else:
                    # Empty frame — no detections this frame
                    meas = []
                    sizes = []
                    shapes = []
                    detection_confidences = []
                    filtered_obb_corners = []
                    detection_ids = []
                    raw_detection_ids = []
                    raw_meas = []
                    raw_sizes = []
                    raw_shapes = []
                    raw_confidences = []
                    raw_obb_corners = []
                    detection_headtail_heading = np.asarray([], dtype=np.float32)
                    detection_headtail_confidence = np.asarray([], dtype=np.float32)
                    headtail_directed_mask = np.asarray([], dtype=np.uint8)
                    raw_heading_hints = []
                    raw_heading_confidences = []
                    raw_directed_mask = []
                    raw_canonical_affines = None
                    _current_frame_result = _frame_result

            elif (
                use_cached_detections
                and detection_method == "background_subtraction"
                and bgsub_detection_cache is not None
            ):
                # Backward pass: replay cached bg-sub detections (no live frame).
                _obb = bgsub_detection_cache.read_frame(actual_frame_index)
                if _obb is not None and _obb.num_detections > 0:
                    meas = frame_result_to_meas(_obb.centroids, _obb.angles)
                    sizes = [float(_obb.sizes[i]) for i in range(_obb.num_detections)]
                    shapes = [
                        (float(_obb.shapes[i, 0]), float(_obb.shapes[i, 1]))
                        for i in range(_obb.num_detections)
                    ]
                    detection_confidences = [
                        float(_obb.confidences[i]) for i in range(_obb.num_detections)
                    ]
                    detection_ids = [
                        int(_obb.detection_ids[i]) for i in range(_obb.num_detections)
                    ]
                else:
                    meas, sizes, shapes, detection_confidences, detection_ids = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                filtered_obb_corners = []
                raw_meas = meas
                raw_sizes = sizes
                raw_shapes = shapes
                raw_confidences = detection_confidences
                raw_obb_corners = filtered_obb_corners
                raw_detection_ids = detection_ids
                raw_heading_hints = []
                raw_heading_confidences = []
                raw_directed_mask = []
                raw_canonical_affines = None

            elif detection_method == "background_subtraction" and frame is not None:
                # Background subtraction detection pipeline. The InferenceRunner
                # bg-sub stage owns the whole sequence the worker used to inline:
                # grayscale + image adjustments, lighting stabilization, adaptive
                # background update, foreground mask, ROI intersection,
                # conservative split, and contour measurement. The frame MUST
                # already be scaled by RESIZE_FACTOR (done at frame_resize above),
                # and ROI_mask_current is already in that resized space, so the
                # stage's ROI resolver is a shape no-op. bg-sub is strictly
                # sequential; this loop feeds frames in ascending order.
                _bgsub_result = bgsub_runner.run_realtime(
                    frame, actual_frame_index, roi_mask=ROI_mask_current
                )
                # SHOW_FG / SHOW_BG overlays read these (realtime-only; the stage
                # stashes the exact mask detection ran on).
                fg_mask = _bgsub_result.fg_mask
                bg_u8 = _bgsub_result.bg_u8
                if bg_u8 is None:
                    # First frame(s): the background model has no history yet, so
                    # there are no detections. Emit the raw frame and skip the
                    # rest of the loop, matching the legacy warmup behavior.
                    if frame is not None:
                        self.emit_frame(frame)
                    continue

                _obb = _bgsub_result.obb
                # meas carries the OBB axis angle in [0, pi); downstream
                # resolve_tracking_theta disambiguates theta vs theta+pi.
                meas = frame_result_to_meas(_obb.centroids, _obb.angles)
                sizes = [float(_obb.sizes[i]) for i in range(_obb.num_detections)]
                shapes = [
                    (float(_obb.shapes[i, 0]), float(_obb.shapes[i, 1]))
                    for i in range(_obb.num_detections)
                ]
                # bg-sub confidences are NaN by design (legacy); do NOT gate on them.
                detection_confidences = [
                    float(_obb.confidences[i]) for i in range(_obb.num_detections)
                ]
                # No OBB corners are drawn for background subtraction.
                filtered_obb_corners = []
                # detection_ids match legacy (frame_idx * STRIDE + slot).
                detection_ids = [int(did) for did in _obb.detection_ids]
                raw_meas = meas
                raw_sizes = sizes
                raw_shapes = shapes
                raw_confidences = detection_confidences
                raw_obb_corners = filtered_obb_corners
                raw_detection_ids = detection_ids
                raw_heading_hints = []
                raw_heading_confidences = []
                raw_directed_mask = []
                raw_canonical_affines = None

                # Cache this frame's detections so the backward pass can replay
                # them. Every frame is written (even empty) so the cache covers the
                # full range. The stage's OBBResult is written directly; its
                # ellipse corners are unused by the replay path (which reads only
                # meas/sizes/shapes/confidences/detection_ids).
                if bgsub_detection_cache is not None:
                    bgsub_detection_cache.write_frame(
                        actual_frame_index,
                        result=_obb,
                    )

            elif (
                detection_method == "yolo_obb" and frame is not None
            ):  # YOLO OBB realtime detection (non-cached)
                # InferenceRunner.run_realtime() runs the full inference stack
                # (OBB + headtail + CNN + pose + AprilTag) on a single frame and
                # returns a FrameResult.  No legacy detector is used here.  The
                # frame index is passed so detections are cached per-frame for the
                # backward pass to replay (realtime + backward support).
                _frame_result = inference_runner.run_realtime(frame, actual_frame_index)
                _current_frame_result = _frame_result
                if _frame_result is not None and _frame_result.obb.num_detections > 0:
                    _obb = _frame_result.obb
                    # meas carries the OBB axis angle (see cached path above).
                    meas = frame_result_to_meas(_obb.centroids, _obb.angles)
                    sizes = [float(_obb.sizes[i]) for i in range(_obb.num_detections)]
                    shapes = [
                        (float(_obb.shapes[i, 0]), float(_obb.shapes[i, 1]))
                        for i in range(_obb.num_detections)
                    ]
                    detection_confidences = [
                        float(_obb.confidences[i]) for i in range(_obb.num_detections)
                    ]
                    filtered_obb_corners = [
                        _obb.corners[i] for i in range(_obb.num_detections)
                    ]
                    detection_ids = [
                        int(_obb.detection_ids[i]) for i in range(_obb.num_detections)
                    ]
                    raw_detection_ids = detection_ids
                    raw_meas = meas
                    raw_sizes = sizes
                    raw_shapes = shapes
                    raw_confidences = detection_confidences
                    raw_obb_corners = filtered_obb_corners
                    if _frame_result.headtail is not None:
                        detection_headtail_heading = np.asarray(
                            _frame_result.headtail.heading_hints, dtype=np.float32
                        )
                        detection_headtail_confidence = np.asarray(
                            _frame_result.headtail.heading_confidences, dtype=np.float32
                        )
                        headtail_directed_mask = np.asarray(
                            _frame_result.headtail.directed_mask, dtype=np.uint8
                        )
                        raw_heading_hints = list(detection_headtail_heading)
                        raw_heading_confidences = list(detection_headtail_confidence)
                        raw_directed_mask = list(headtail_directed_mask)
                    else:
                        detection_headtail_heading = np.asarray([], dtype=np.float32)
                        detection_headtail_confidence = np.asarray([], dtype=np.float32)
                        headtail_directed_mask = np.asarray([], dtype=np.uint8)
                        raw_heading_hints = []
                        raw_heading_confidences = []
                        raw_directed_mask = []
                    raw_canonical_affines = None
                else:
                    # Empty frame — no detections this frame
                    meas = []
                    sizes = []
                    shapes = []
                    detection_confidences = []
                    filtered_obb_corners = []
                    detection_ids = []
                    raw_detection_ids = []
                    raw_meas = []
                    raw_sizes = []
                    raw_shapes = []
                    raw_confidences = []
                    raw_obb_corners = []
                    detection_headtail_heading = np.asarray([], dtype=np.float32)
                    detection_headtail_confidence = np.asarray([], dtype=np.float32)
                    headtail_directed_mask = np.asarray([], dtype=np.uint8)
                    raw_heading_hints = []
                    raw_heading_confidences = []
                    raw_directed_mask = []
                    raw_canonical_affines = None

            else:
                # No frame and no cached detections - skip this iteration
                if not use_cached_detections:
                    logger.warning(
                        f"Frame {self.frame_count}: No frame available and no cached detections"
                    )
                    continue

            # InferenceRunner writes its own caches during run_batch_pass / run_realtime.
            # Legacy detection_cache.add_frame() is not used for yolo_obb mode.
            profiler.tock("detection")

            # === Streaming Phase 1: Build shared analysis payload ===
            # Constructed from filtered detections + head-tail results so
            # downstream pose and CNN dispatchers share a single stable index.
            if (
                streaming_precompute_enabled
                and detection_ids
                and not self.backward_mode
            ):
                try:
                    from hydra_suite.core.tracking.ingest.streaming_payload import (
                        build_streaming_payload,
                    )

                    profiler.tick("streaming_payload_build")
                    _streaming_payload = build_streaming_payload(
                        frame_idx=actual_frame_index,
                        raw_meas=meas,
                        raw_obb_corners=(
                            filtered_obb_corners
                            if "filtered_obb_corners" in locals()
                            else []
                        ),
                        raw_heading_hints=(
                            filtered_heading_hints
                            if "filtered_heading_hints" in locals()
                            else []
                        ),
                        raw_heading_confidences=(
                            filtered_heading_confidences
                            if "filtered_heading_confidences" in locals()
                            else []
                        ),
                        raw_directed_mask=(
                            filtered_directed_mask
                            if "filtered_directed_mask" in locals()
                            else []
                        ),
                        raw_canonical_affines=(
                            raw_canonical_affines
                            if "raw_canonical_affines" in locals()
                            else []
                        ),
                        detection_ids=detection_ids,
                        input_is_bgr=True,
                        runtime_family=str(p.get("COMPUTE_RUNTIME", "cpu")),
                    )
                    profiler.tock("streaming_payload_build")
                except Exception as _spay_err:
                    logger.debug(
                        "StreamingAnalysisPayload build failed (non-fatal): %s",
                        _spay_err,
                    )
                    _streaming_payload = None
            else:
                _streaming_payload = None

            # === Site F: Populate live stores from FrameResult (YOLO OBB path) ===
            # For YOLO OBB: push CNN, pose, and AprilTag results from the FrameResult
            # produced by InferenceRunner (load_frame or run_realtime) into the
            # corresponding live stores so the tracking loop can look them up by
            # frame index and detection ID.
            if inference_runner is not None and _current_frame_result is not None:
                _fr = _current_frame_result
                _det_ids_arr = (
                    np.asarray(detection_ids, dtype=np.int64)
                    if detection_ids
                    else np.zeros(0, dtype=np.int64)
                )
                if live_pose_props_cache is not None:
                    populate_live_pose_store(
                        live_pose_props_cache,
                        _fr.pose,
                        _det_ids_arr,
                        actual_frame_index,
                    )
                if live_tag_obs_cache is not None:
                    populate_live_tag_store(
                        live_tag_obs_cache,
                        _fr.apriltag,
                        _det_ids_arr,
                        actual_frame_index,
                    )
                for _cnn_label, _cnn_store in live_cnn_caches.items():
                    populate_live_cnn_store(
                        _cnn_store,
                        _fr.cnn,
                        _det_ids_arr,
                        actual_frame_index,
                        _cnn_label,
                        evidence_emitter=(
                            cnn_evidence_emitters.get(_cnn_label)
                            if "cnn_evidence_emitters" in locals()
                            else None
                        ),
                    )

            profiler.tick("features")
            detection_crop_quality = np.zeros(len(meas), dtype=np.float32)
            detection_pose_heading = np.full(len(meas), np.nan, dtype=np.float32)
            detection_pose_keypoints = [None] * len(meas)
            detection_pose_visibility = np.zeros(len(meas), dtype=np.float32)
            detection_directed_heading = np.full(len(meas), np.nan, dtype=np.float32)
            detection_directed_mask = np.zeros(len(meas), dtype=np.uint8)
            detection_headtail_confidence = np.asarray(
                (
                    detection_headtail_confidence
                    if "detection_headtail_confidence" in locals()
                    else np.zeros(len(meas), dtype=np.float32)
                ),
                dtype=np.float32,
            )

            if meas and shapes:
                reference_body_size = float(params.get("REFERENCE_BODY_SIZE", 20.0))
                for det_idx in range(min(len(meas), len(shapes))):
                    detection_crop_quality[det_idx] = (
                        self._estimate_detection_crop_quality(
                            shapes[det_idx], reference_body_size
                        )
                    )

            # Optional pose-based geometry features and direction override.
            if pose_direction_enabled and meas and detection_ids:
                if pose_frame_keypoints_map_frame != actual_frame_index:
                    pose_frame_keypoints_map = _pf_build_keypoint_map(
                        pose_props_cache, actual_frame_index
                    )
                    pose_frame_keypoints_map_frame = actual_frame_index

                pose_directed_mask = np.zeros(len(meas), dtype=np.uint8)
                n_det = min(len(meas), len(detection_ids))
                for det_idx in range(n_det):
                    try:
                        det_id = int(detection_ids[det_idx])
                    except Exception:
                        continue
                    keypoints = pose_frame_keypoints_map.get(det_id)
                    pose_features = _pf_compute_geometry(
                        keypoints,
                        pose_direction_anterior_indices,
                        pose_direction_posterior_indices,
                        pose_min_valid_conf,
                        ignore_indices=pose_ignore_indices,
                    )
                    if pose_features is None:
                        continue
                    visibility = float(pose_features.get("visibility", 0.0) or 0.0)
                    detection_pose_visibility[det_idx] = visibility
                    detection_pose_keypoints[det_idx] = _pf_normalize_keypoints(
                        keypoints,
                        pose_min_valid_conf,
                        ignore_indices=pose_ignore_indices,
                    )
                    pose_theta = pose_features.get("heading")
                    if pose_theta is None:
                        continue
                    detection_pose_heading[det_idx] = np.float32(pose_theta)
                    if _pf_heading_reliable(
                        detection_pose_keypoints[det_idx],
                        visibility,
                        min_visibility=pose_direction_min_visibility,
                        min_valid_keypoints=pose_direction_min_keypoints,
                    ):
                        pose_directed_mask[det_idx] = 1

            pose_overrides_headtail = bool(params.get("POSE_OVERRIDES_HEADTAIL", True))
            if len(meas) > 0:
                detection_directed_heading, detection_directed_mask = (
                    _pf_build_direction_overrides(
                        len(meas),
                        detection_pose_heading,
                        pose_directed_mask,
                        detection_headtail_heading,
                        headtail_directed_mask,
                        pose_overrides_headtail=pose_overrides_headtail,
                    )
                )

            detection_theta_raw = (
                np.array([float(m[2]) for m in meas], dtype=np.float32)
                if meas
                else np.zeros(0, dtype=np.float32)
            )
            detection_theta_resolved = detection_theta_raw.copy()
            detection_heading_source = ["obb_axis"] * len(meas)
            for det_idx in range(len(meas)):
                pose_is_directed = bool(
                    det_idx < len(pose_directed_mask) and pose_directed_mask[det_idx]
                )
                headtail_is_directed = bool(
                    det_idx < len(headtail_directed_mask)
                    and headtail_directed_mask[det_idx]
                )
                detection_heading_source[det_idx] = self._heading_source_for_detection(
                    pose_is_directed,
                    headtail_is_directed,
                    pose_overrides_headtail,
                )
                if (
                    det_idx < len(detection_directed_mask)
                    and detection_directed_mask[det_idx]
                    and det_idx < len(detection_directed_heading)
                    and np.isfinite(detection_directed_heading[det_idx])
                ):
                    detection_theta_resolved[det_idx] = np.float32(
                        detection_directed_heading[det_idx]
                    )

            if detected_props_cache is not None and detection_ids:
                n_dets = len(detection_ids)

                def _pad_to_n(arr, n, fill_value, dtype):
                    a = np.asarray(arr, dtype=dtype).reshape(-1)
                    if a.size == n:
                        return a
                    out = np.full(n, fill_value, dtype=dtype)
                    out[: min(a.size, n)] = a[: min(a.size, n)]
                    return out

                ht_directed = _pad_to_n(headtail_directed_mask, n_dets, 0, np.uint8)
                ht_heading = _pad_to_n(
                    detection_headtail_heading, n_dets, np.nan, np.float32
                )
                ht_conf = _pad_to_n(
                    detection_headtail_confidence, n_dets, 0.0, np.float32
                )
                detected_props_cache.add_frame(
                    actual_frame_index,
                    detection_ids=detection_ids,
                    theta_raw=detection_theta_raw,
                    theta_resolved=detection_theta_resolved,
                    heading_source=detection_heading_source,
                    heading_directed=detection_directed_mask,
                    headtail_heading=ht_heading,
                    headtail_confidence=ht_conf,
                    headtail_directed=ht_directed,
                )

            if len(meas) >= params.get("MIN_DETECTIONS_TO_START", 1):
                detection_counts += 1
            else:
                detection_counts = 0
            if (
                detection_counts >= max(1, params["MIN_DETECTION_COUNTS"] // 2)
                and not detection_initialized
            ):
                detection_initialized = True
                logger.info(f"Tracking initialized with {len(meas)} detections.")

            # === VISUALIZATION (Skip in cached detection mode) ===
            emit_visualization_frame = self._should_emit_visualization_frame(
                self.frame_count,
                params,
            )
            needs_overlay = (
                not skip_visualization
                and frame is not None
                and (emit_visualization_frame or self.video_writer is not None)
            )

            # === VISUALIZATION (Skip in cached detection mode) ===
            if needs_overlay:
                overlay = frame.copy()

                # Apply ROI visualization - draw cyan boundary for all detection methods
                # The actual masking for background subtraction happens earlier in the pipeline
                if ROI_mask_current is not None:
                    # Compute ROI contours once and cache (mask is static).
                    if _roi_contours_cache is None:
                        _roi_contours_cache, _ = cv2.findContours(
                            ROI_mask_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                    if _roi_contours_cache:
                        # Draw cyan boundary (BGR: 255, 255, 0)
                        cv2.drawContours(
                            overlay, _roi_contours_cache, -1, (255, 255, 0), 2
                        )
            else:
                overlay = None

            profiler.tock("features")

            if detection_initialized and meas:
                # --- Assignment ---
                profiler.tick("kf_predict")
                preds = self.kf_manager.get_predictions()
                profiler.tock("kf_predict")

                profiler.tick("cost_matrix")

                # --- AprilTag detection map for this frame ---
                _tag_det_map = build_tag_detection_map(
                    tag_obs_cache, actual_frame_index
                )
                _tag_hamming_map = build_tag_detection_hamming_map(
                    tag_obs_cache, actual_frame_index
                )
                _det_tag_ids = build_detection_tag_id_list(_tag_det_map, len(meas))

                # The Assigner now takes the kf_manager directly to access X and S_inv
                association_data = {
                    "detection_confidences": detection_confidences,
                    "detection_crop_quality": detection_crop_quality,
                    "detection_pose_heading": detection_directed_heading,
                    "detection_pose_keypoints": detection_pose_keypoints,
                    "detection_pose_visibility": detection_pose_visibility,
                    "track_pose_prototypes": track_pose_prototypes,
                    "track_avg_step": track_avg_step,
                }

                # Load CNN frame predictions for each phase (used for Bayesian cost
                # term and post-assignment decoder evidence).
                _cnn_frame_preds_all = {}
                for _state in _cnn_phase_states:
                    _label = _state["label"]
                    _cache = _state["cache"]
                    try:
                        _cnn_frame_preds_all[_label] = _cache.load(actual_frame_index)
                    except Exception:
                        _cnn_frame_preds_all[_label] = []

                # Build Bayesian identity cost terms when the online decoder is active.
                if _identity_online_decoder is not None:
                    try:
                        _cat = _identity_online_decoder._catalog
                        _tag_label_map = {
                            idx: str(_lbl)
                            for idx, _lbl in enumerate(
                                p.get("TAG_IDENTITY_LABELS", []) or []
                            )
                            if str(_lbl).strip()
                        }
                        _n_dets = len(meas)
                        _det_log_likes: list = []
                        for _j in range(_n_dets):
                            _log_like = None
                            # AprilTag contribution
                            _tid = (
                                _det_tag_ids[_j] if _j < len(_det_tag_ids) else NO_TAG
                            )
                            if _tid >= 0 and hasattr(_cat, "apriltag_log_prior"):
                                _log_like = _cat.apriltag_log_prior(
                                    _tid, _tag_label_map
                                )
                            # CNN contribution (identity-providing phases only, summed in log-space)
                            for _state in _cnn_phase_states:
                                if not _state.get("is_identity_provider", False):
                                    continue
                                _fps = _cnn_frame_preds_all.get(_state["label"], [])
                                _pred = _fps[_j] if _j < len(_fps) else None
                                if _pred is not None and _pred.class_names:
                                    _cn = _pred.class_names[0]
                                    if _cn and _cat.contains(_cn):
                                        _conf = (
                                            float(_pred.confidences[0])
                                            if _pred.confidences
                                            else 0.5
                                        )
                                        _known = list(_cat.labels[1:])
                                        _n_known = max(_cat.num_known - 1, 1)
                                        _cnn_lp = _cat.cnn_log_prior(
                                            [
                                                (
                                                    _conf
                                                    if _l == _cn
                                                    else (1.0 - _conf) / _n_known
                                                )
                                                for _l in _known
                                            ],
                                            _known,
                                        )
                                        _log_like = (
                                            _cnn_lp
                                            if _log_like is None
                                            else _log_like + _cnn_lp
                                        )
                            _det_log_likes.append(_log_like)
                        association_data["identity_detection_log_likelihoods"] = (
                            _det_log_likes
                        )
                        association_data["identity_track_log_posteriors"] = (
                            _identity_online_decoder.get_slot_log_posteriors(
                                list(range(N))
                            )
                        )
                    except Exception:
                        logger.debug(
                            "Bayesian identity cost term build failed (non-fatal)",
                            exc_info=True,
                        )

                # --- Density-aware pre-gate ---
                # For detections inside a high-density region, apply a tighter
                # distance threshold in the cost matrix.  This blocks long-range
                # matches into crowded zones (the track goes occluded instead)
                # while leaving short-range matches intact.
                _density_flags = None
                if self._density_regions and len(meas) > 0:
                    try:
                        _density_flags = get_density_region_flags(
                            meas,
                            self._density_regions,
                            frame_idx=actual_frame_index,
                        )
                    except Exception:
                        logger.debug(
                            "Density region flag computation failed (non-fatal)",
                            exc_info=True,
                        )

                cost, spatial_candidates = assigner.compute_cost_matrix(
                    N,
                    meas,
                    preds,
                    shapes,
                    self.kf_manager,
                    last_shape_info,
                    meas_ori_directed=(
                        detection_directed_mask
                        if len(detection_directed_mask) == len(meas)
                        else None
                    ),
                    association_data=association_data,
                )

                # Tighter distance gate for density-region detections.
                # Block (track, detection) pairs where the detection is in a
                # density region AND the raw Euclidean distance from the track's
                # predicted position exceeds the tighter threshold.  This
                # prevents long-range jumps into crowded zones without
                # distorting the cost matrix (which can push assignments to
                # wrong detections farther away).
                if _density_flags is not None and np.any(_density_flags):
                    _density_factor = float(
                        params.get("DENSITY_CONSERVATIVE_FACTOR", 0.7)
                    )
                    if _density_factor < 1.0:
                        _base_max_dist = float(
                            assigner.params.get("MAX_DISTANCE_THRESHOLD", 1000.0)
                        )
                        _density_max_dist = _base_max_dist * _density_factor
                        _pred_xy = np.asarray(
                            self.kf_manager.X[:N, :2], dtype=np.float32
                        )
                        _meas_xy = np.array(
                            [meas[j][:2] for j in range(len(meas))],
                            dtype=np.float32,
                        )
                        _raw_dist = np.linalg.norm(
                            _pred_xy[:, None, :] - _meas_xy[None, :, :], axis=2
                        )
                        # Block long-range matches to density-region detections.
                        _flagged_cols = np.where(_density_flags)[0]
                        for _c in _flagged_cols:
                            cost[_raw_dist[:, _c] >= _density_max_dist, _c] = 1e9

                profiler.tock("cost_matrix")
                profiler.tick("hungarian")

                # Build committed slot map for identity-first rejoining.
                # Gated by ASSOCIATION_IDENTITY_HINT_SCALE so that weight=0
                # means "decoder labels tracks but does not influence which
                # detections are assigned to which slots" — geometry alone
                # decides associations, matching the same gate used by the
                # Bayesian cost term.
                _committed_slot_identities: "dict | None" = None
                if (
                    _identity_online_decoder is not None
                    and params.get("ENABLE_IDENTITY_ONLINE_DECODER", False)
                    and float(params.get("ASSOCIATION_IDENTITY_HINT_SCALE", 0.3)) > 0.0
                ):
                    _csmap: dict = {}
                    for _s in range(N):
                        _bel = _identity_online_decoder.get_belief(_s)
                        if _bel is not None and _bel.committed_label:
                            _csmap[_s] = _bel.committed_label
                    _committed_slot_identities = _csmap if _csmap else None

                rows, cols, free_dets, identity_rejoin_pairs = assigner.assign_tracks(
                    cost,
                    N,
                    len(meas),
                    meas,
                    track_states,
                    tracking_continuity,
                    self.kf_manager,
                    spatial_candidates,
                    association_data=association_data,
                    committed_slot_identities=_committed_slot_identities,
                    missed_frames=missed_frames,
                )
                respawned_matches = {r for r in rows if track_states[r] == "lost"}
                _identity_rejoin_slots = {s for s, _ in identity_rejoin_pairs}
                profiler.tock("hungarian")

                # --- Diagnostic: log large-distance assignments ---
                if rows and self.frame_count % 10 == 0:
                    _body = float(
                        params.get("REFERENCE_BODY_SIZE", 20.0)
                        * params.get("RESIZE_FACTOR", 1.0)
                    )
                    _max_d = float(
                        assigner.params.get("MAX_DISTANCE_THRESHOLD", 1000.0)
                    )
                    _vel_gate = (
                        float(params.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)) * _body
                    )
                    for _r, _c in zip(rows, cols):
                        _det_xy = np.array(meas[_c][:2], dtype=np.float32)
                        _pred_xy_diag = self.kf_manager.X[_r, :2].copy()
                        _raw_d = float(np.linalg.norm(_det_xy - _pred_xy_diag))
                        _last_d = float("nan")
                        if self.trajectories_full[_r]:
                            _lp = self.trajectories_full[_r][-1]
                            _last_d = float(
                                np.linalg.norm(
                                    _det_xy
                                    - np.array([_lp[0], _lp[1]], dtype=np.float32)
                                )
                            )
                        _state = track_states[_r]
                        _cont = tracking_continuity[_r]
                        _is_respawn = _r in respawned_matches
                        if _raw_d > _body * 1.5 or _last_d > _body * 2.0:
                            logger.warning(
                                f"JUMP? frame={actual_frame_index} slot={_r} "
                                f"traj={trajectory_ids[_r]} state={_state} "
                                f"cont={_cont} respawn={_is_respawn} "
                                f"raw_dist={_raw_d:.1f} last_dist={_last_d:.1f} "
                                f"body={_body:.1f} MAX_DIST={_max_d:.1f} "
                                f"VEL_GATE={_vel_gate:.1f}"
                            )

                # Conditionally compute confidence metrics (for performance)
                save_confidence = params.get("SAVE_CONFIDENCE_METRICS", True)
                if save_confidence:
                    # Compute assignment confidence for matched pairs
                    matched_pairs = list(zip(rows, cols))
                    assignment_confidences = assigner.compute_assignment_confidence(
                        cost, matched_pairs
                    )

                    # Get Kalman filter position uncertainties
                    position_uncertainties = (
                        self.kf_manager.get_position_uncertainties()
                    )
                else:
                    assignment_confidences = {}
                    position_uncertainties = []

                # --- State Management ---
                profiler.tick("state_update")
                # Identity-rejoin slots count as matched (no trajectory reset)
                matched = set(rows) | _identity_rejoin_slots
                unmatched = list(set(range(N)) - matched)
                for r in matched:
                    missed_frames[r], track_states[r] = 0, "active"
                for r in unmatched:
                    missed_frames[r] += 1
                    if missed_frames[r] >= params["LOST_THRESHOLD_FRAMES"]:
                        track_states[r], tracking_continuity[r] = "lost", 0
                    elif track_states[r] != "lost":
                        track_states[r] = "occluded"

                # === Identity Overhaul Phase 1: Online decoder per-frame update ===
                # Runs after history updates so head-tail and CNN histories are
                # current.  Consumes posterior-aware CNN evidence when available,
                # falling back to compatibility top-1 priors only when needed.
                if _identity_online_decoder is not None:
                    try:
                        from hydra_suite.core.identity.evidence import IdentityEvidence

                        # Only clear uncommitted respawns; identity-rejoin slots keep beliefs
                        for _r in respawned_matches:
                            _identity_online_decoder.clear_slot(
                                _r,
                                reason=f"respawn at frame {actual_frame_index}",
                                respawn_frame_idx=actual_frame_index,
                            )

                        # Decay committed beliefs for slots that are still absent
                        if _committed_slot_identities:
                            _still_absent_committed = [
                                _s
                                for _s in _committed_slot_identities
                                if _s not in _identity_rejoin_slots
                                and track_states[_s] == "lost"
                            ]
                            if _still_absent_committed:
                                _identity_online_decoder.decay_absent_slot_beliefs(
                                    _still_absent_committed
                                )

                        _visible_slots = [r for r in matched]
                        _slot_evs: dict = {}

                        # Build AprilTag evidence for matched detections
                        # (includes identity-rejoin pairs so decoder gets updated)
                        _all_matched_pairs = list(zip(rows, cols)) + list(
                            identity_rejoin_pairs
                        )
                        if tag_obs_cache is not None:
                            _tag_label_map = {
                                idx: str(label)
                                for idx, label in enumerate(
                                    p.get("TAG_IDENTITY_LABELS", []) or []
                                )
                                if str(label).strip()
                            }
                            for _r, _c in _all_matched_pairs:
                                _tid = (
                                    _det_tag_ids[_c] if _c < len(_det_tag_ids) else -1
                                )
                                if _tid >= 0 and hasattr(
                                    _identity_online_decoder, "_catalog"
                                ):
                                    _cat = _identity_online_decoder._catalog
                                    if hasattr(_cat, "apriltag_log_prior"):
                                        _lp = _cat.apriltag_log_prior(
                                            _tid,
                                            _tag_label_map,
                                        )
                                        _det_id = (
                                            int(detection_ids[_c])
                                            if _c < len(detection_ids)
                                            else int(_c)
                                        )
                                        _ev = IdentityEvidence.from_apriltag(
                                            actual_frame_index, _det_id, _lp
                                        )
                                        _slot_evs.setdefault(_r, []).append(_ev)

                        # Build CNN evidence for matched detections (identity-providing phases only)
                        for _state in _cnn_phase_states:
                            if not _state.get("is_identity_provider", False):
                                continue
                            _label = _state["label"]
                            _cache = _state["cache"]
                            _evidence_cache = _state.get("evidence_cache")
                            _fps = _cnn_frame_preds_all.get(_label)
                            if _fps is None:
                                continue
                            _live_evidence_map = {}
                            _source_labels = None
                            if isinstance(_cache, LiveCNNIdentityStore):
                                _live_evidence_map = {
                                    int(_ev.detection_id): _ev
                                    for _ev in _cache.load_evidences(actual_frame_index)
                                }
                                _source_labels = _cache.catalog_labels
                            elif _evidence_cache is not None:
                                _live_evidence_map = {
                                    int(_ev.detection_id): _ev
                                    for _ev in _evidence_cache.load_frame(
                                        actual_frame_index
                                    )
                                }
                                _source_labels = _evidence_cache.catalog_labels

                            for _r, _c in _all_matched_pairs:
                                _det_id = (
                                    int(detection_ids[_c])
                                    if _c < len(detection_ids)
                                    else int(_c)
                                )
                                _cached_ev = _live_evidence_map.get(_det_id)
                                if _cached_ev is not None and hasattr(
                                    _identity_online_decoder, "_catalog"
                                ):
                                    _mapped_lp = _remap_source_log_probs_to_catalog(
                                        _cached_ev.log_probs,
                                        _source_labels,
                                    )
                                    _slot_evs.setdefault(_r, []).append(
                                        IdentityEvidence.from_cnn(
                                            actual_frame_index,
                                            _det_id,
                                            _label,
                                            _mapped_lp,
                                            calibration_signature=_cached_ev.calibration_signature,
                                            runtime_signature=_cached_ev.runtime_signature,
                                            observed_mask=_cached_ev.observed_mask,
                                        )
                                    )
                                    continue

                                _pred = _fps[_c] if _c < len(_fps) else None
                                if _pred is None:
                                    continue
                                if hasattr(_identity_online_decoder, "_catalog"):
                                    _cat = _identity_online_decoder._catalog
                                    _phase_cnpf = (
                                        _state.get("class_names_per_factor") or []
                                    )
                                    _non_empty_factors = [
                                        fl for fl in _phase_cnpf if fl
                                    ]
                                    if len(_non_empty_factors) > 1:
                                        # Multi-factor: build joint log-prior over composite catalog.
                                        # Reconstruct per-factor soft distributions from top-1 and
                                        # compute P(composite) = product of per-factor probabilities.
                                        _per_factor_dist: list[dict[str, float]] = []
                                        for _k, (_cn, _conf_raw) in enumerate(
                                            zip(_pred.class_names, _pred.confidences)
                                        ):
                                            _fk_labels = (
                                                _non_empty_factors[_k]
                                                if _k < len(_non_empty_factors)
                                                else []
                                            )
                                            _nk = len(_fk_labels)
                                            if _nk == 0:
                                                _per_factor_dist.append({})
                                                continue
                                            _conf_k = max(float(_conf_raw), 1e-9)
                                            _other_p = max(
                                                1e-9, (1.0 - _conf_k) / max(_nk - 1, 1)
                                            )
                                            _per_factor_dist.append(
                                                {
                                                    lbl: (
                                                        _conf_k
                                                        if lbl == _cn
                                                        else _other_p
                                                    )
                                                    for lbl in _fk_labels
                                                }
                                            )
                                        _n_factors_local = len(_non_empty_factors)
                                        _n_cat = _cat.size
                                        _lp_arr = np.full(
                                            _n_cat, np.log(1e-9), dtype=np.float64
                                        )
                                        for _ci in range(1, _n_cat):
                                            _cl = _cat.label_of(_ci)
                                            _parts = _cl.split("_")
                                            if len(_parts) != _n_factors_local:
                                                continue
                                            _joint = 0.0
                                            for _k, _part in enumerate(_parts):
                                                _fk_p = (
                                                    _per_factor_dist[_k].get(
                                                        _part, 1e-9
                                                    )
                                                    if _k < len(_per_factor_dist)
                                                    else 1e-9
                                                )
                                                _joint += np.log(max(_fk_p, 1e-300))
                                            _lp_arr[_ci] = _joint
                                        _lp_arr -= np.logaddexp.reduce(_lp_arr)
                                        _ev = IdentityEvidence.from_cnn(
                                            actual_frame_index,
                                            _det_id,
                                            _label,
                                            _lp_arr,
                                        )
                                        _slot_evs.setdefault(_r, []).append(_ev)
                                    else:
                                        # Single factor: original per-label top-1 logic.
                                        for _k, _cn in enumerate(_pred.class_names):
                                            if _cn and _cat.contains(_cn):
                                                _conf = (
                                                    float(_pred.confidences[_k])
                                                    if _k < len(_pred.confidences)
                                                    else 0.5
                                                )
                                                _lp = _cat.cnn_log_prior(
                                                    [
                                                        (
                                                            _conf
                                                            if _l == _cn
                                                            else (1.0 - _conf)
                                                            / max(_cat.num_known - 1, 1)
                                                        )
                                                        for _l in list(_cat.labels[1:])
                                                    ],
                                                    list(_cat.labels[1:]),
                                                )
                                                _ev = IdentityEvidence.from_cnn(
                                                    actual_frame_index,
                                                    _det_id,
                                                    _label,
                                                    _lp,
                                                )
                                                _slot_evs.setdefault(_r, []).append(_ev)

                        _online_assignments = _identity_online_decoder.update_frame(
                            actual_frame_index, _visible_slots, _slot_evs
                        )
                        _identity_online_assignments = {
                            a.slot_index: a for a in _online_assignments
                        }
                    except Exception as _dec_err:
                        logger.debug(
                            "Online identity decoder update failed (non-fatal): %s",
                            _dec_err,
                        )

                # --- KF Update & State Update ---
                profiler.tock("state_update")
                profiler.tick("kf_update")
                # Identity-rejoin pairs use the same KF correct path but skip hard reset
                for r, c in list(zip(rows, cols)) + list(identity_rejoin_pairs):
                    meas_x = float(meas[c][0])
                    meas_y = float(meas[c][1])
                    measured_theta = float(meas[c][2])
                    directed_heading = bool(
                        c < len(detection_directed_mask)
                        and detection_directed_mask[c] == 1
                    )
                    theta_for_tracking = _pf_resolve_detection_tracking_theta(
                        r,
                        measured_theta,
                        (
                            detection_directed_heading[c]
                            if c < len(detection_directed_heading)
                            else float("nan")
                        ),
                        directed_heading,
                        orientation_last,
                        fallback_theta=preds[r, 2] if r < len(preds) else None,
                    )
                    if directed_heading:
                        pose_direction_applied_count += 1
                    else:
                        pose_direction_fallback_count += 1

                    if r in respawned_matches:
                        # Hard KF reset for Phase-3 respawns.  All per-track state
                        # is cleared here so the new trajectory starts clean.
                        # Trajectory-ID assignment is done here (single code path with
                        # the free_dets loop below) — never inside the assigner.
                        trajectory_ids[r] = next_trajectory_id
                        next_trajectory_id += 1
                        self.trajectories_full[r].clear()
                        trajectories_pruned[r].clear()
                        position_deques[r].clear()
                        track_avg_step[r] = 0.0
                        local_counts[r] = 0
                        orientation_last[r] = theta_for_tracking
                        track_pose_prototypes[r] = None
                        self.kf_manager.initialize_filter(
                            r,
                            np.array(
                                [meas_x, meas_y, theta_for_tracking, 0.0, 0.0],
                                dtype=np.float32,
                            ),
                        )
                    elif r in _identity_rejoin_slots:
                        # Soft KF reset for identity-rejoin: snap position to the
                        # matched detection and clear stale covariance from coasting.
                        # Trajectory state is preserved (no trajectory-ID reset) since
                        # identity confirms this is the same animal.  Without this reset
                        # the large P accumulated during the gap makes S near-singular
                        # in float32, causing LinAlgError in the correction step.
                        self.kf_manager.initialize_filter(
                            r,
                            np.array(
                                [meas_x, meas_y, theta_for_tracking, 0.0, 0.0],
                                dtype=np.float32,
                            ),
                        )

                    corrected_meas = np.asarray(
                        [meas_x, meas_y, theta_for_tracking], dtype=np.float32
                    )
                    # Scale theta measurement noise by heading confidence:
                    # low-confidence headings get inflated R[2,2] so the KF
                    # trusts its own prediction more than a noisy measurement.
                    _orient_conf_for_r = 1.0
                    if directed_heading:
                        if c < len(pose_directed_mask) and pose_directed_mask[c]:
                            _orient_conf_for_r = (
                                float(detection_pose_visibility[c])
                                if c < len(detection_pose_visibility)
                                else 1.0
                            )
                        else:
                            _orient_conf_for_r = (
                                float(detection_headtail_confidence[c])
                                if c < len(detection_headtail_confidence)
                                else 1.0
                            )
                    _theta_r_scale = 1.0 / max(_orient_conf_for_r, 0.1)
                    self.kf_manager.correct(
                        r, corrected_meas, theta_r_scale=_theta_r_scale
                    )
                    track_x = float(self.kf_manager.X[r, 0])
                    track_y = float(self.kf_manager.X[r, 1])
                    if not (np.isfinite(track_x) and np.isfinite(track_y)):
                        track_x, track_y = meas_x, meas_y

                    tracking_continuity[r] += 1
                    # Use raw detection position for the motion-gate reference so
                    # that hysteresis / flip detection compares against what was
                    # actually written to output (meas_x/y), not the KF posterior.
                    position_deques[r].append((meas_x, meas_y, self.frame_count))
                    if len(position_deques[r]) == 2:
                        (px1, py1, pf1), (px2, py2, pf2) = position_deques[r]
                        speed = math.hypot(px2 - px1, py2 - py1) / max(1, pf2 - pf1)
                    else:
                        speed = 0
                    orientation_last[r] = self._smooth_orientation(
                        r,
                        theta_for_tracking,
                        speed,
                        params,
                        orientation_last,
                        position_deques,
                        directed_heading=directed_heading,
                    )
                    # Feed smoothed heading back into the Kalman state so that
                    # the KF prediction stays consistent with the orientation
                    # actually used for tracking/display.  Without this, the KF
                    # state diverges from orientation_last and subsequent
                    # innovations are computed against a stale reference.
                    if orientation_last[r] is not None and r < len(self.kf_manager.X):
                        self.kf_manager.X[r, 2] = float(orientation_last[r])
                    last_shape_info[r] = shapes[c]
                    feature_alpha = float(
                        np.clip(params.get("TRACK_FEATURE_EMA_ALPHA", 0.85), 0.0, 0.999)
                    )
                    high_conf_thresh = float(
                        np.clip(
                            params.get("ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD", 0.7),
                            0.0,
                            1.0,
                        )
                    )
                    det_conf_for_track = (
                        float(detection_confidences[c])
                        if c < len(detection_confidences)
                        else 0.0
                    )
                    if det_conf_for_track >= high_conf_thresh:
                        prev_avg = float(track_avg_step[r])
                        track_avg_step[r] = (
                            feature_alpha * prev_avg + (1.0 - feature_alpha) * speed
                        )

                    det_pose_proto = (
                        detection_pose_keypoints[c]
                        if c < len(detection_pose_keypoints)
                        else None
                    )
                    if det_pose_proto is not None:
                        det_pose_proto = np.asarray(det_pose_proto, dtype=np.float32)
                        prev_pose_proto = track_pose_prototypes[r]
                        if prev_pose_proto is None or np.shape(
                            prev_pose_proto
                        ) != np.shape(det_pose_proto):
                            track_pose_prototypes[r] = det_pose_proto.copy()
                        else:
                            prev_pose_proto = np.asarray(
                                prev_pose_proto, dtype=np.float32
                            )
                            updated = prev_pose_proto.copy()
                            for kp_idx in range(len(det_pose_proto)):
                                det_valid = np.isfinite(
                                    det_pose_proto[kp_idx, 0]
                                ) and np.isfinite(det_pose_proto[kp_idx, 1])
                                prev_valid = np.isfinite(
                                    updated[kp_idx, 0]
                                ) and np.isfinite(updated[kp_idx, 1])
                                if det_valid and prev_valid:
                                    updated[kp_idx, :2] = (
                                        feature_alpha * updated[kp_idx, :2]
                                        + (1.0 - feature_alpha)
                                        * det_pose_proto[kp_idx, :2]
                                    )
                                    updated[kp_idx, 2] = max(
                                        float(updated[kp_idx, 2]),
                                        float(det_pose_proto[kp_idx, 2]),
                                    )
                                elif det_valid:
                                    updated[kp_idx] = det_pose_proto[kp_idx]
                            track_pose_prototypes[r] = updated

                    # Report the actual detection position, not the Kalman posterior or
                    # smoothed orientation.  The KF state is preserved for internal prediction
                    # and cost-matrix use, but every x,y,theta written to trajectories and CSV
                    # must correspond to a real detection measurement.
                    #
                    # In backward+undirected mode we expose det_theta_out flipped by pi
                    # (so the OUTPUT represents head-direction in forward time), but we
                    # MUST keep orientation_last and the KF theta state aligned with the
                    # internal theta_for_tracking — feeding the flipped output back into
                    # orientation_last makes next-frame OBB-axis disambiguation pick the
                    # opposite representative, producing a 180° per-frame oscillation in
                    # both stored state and emitted output.
                    det_theta_out = theta_for_tracking
                    if self.backward_mode and not directed_heading:
                        det_theta_out = (det_theta_out + np.pi) % (2 * np.pi)

                    # Update trajectory with actual frame index
                    pt = (meas_x, meas_y, det_theta_out, actual_frame_index)
                    self.trajectories_full[r].append(pt)
                    trajectories_pruned[r].append(pt)

                    # Synchronise orientation_last and the KF theta state to the
                    # internal theta_for_tracking (the value passed to KF correct()),
                    # NOT to the emitted det_theta_out — see comment above.
                    orientation_last[r] = float(theta_for_tracking)
                    if r < len(self.kf_manager.X):
                        self.kf_manager.X[r, 2] = float(theta_for_tracking)

                    if self.csv_writer_thread:
                        # Build base data row with actual frame index
                        row_data = [
                            r,
                            trajectory_ids[r],
                            local_counts[r],
                            pt[0],
                            pt[1],
                            pt[2],
                            pt[3],
                            track_states[r],
                        ]

                        # Add confidence values if enabled
                        if save_confidence:
                            det_conf = (
                                detection_confidences[c]
                                if c < len(detection_confidences)
                                else 0.0
                            )
                            assign_conf = assignment_confidences.get(r, 0.0)
                            pos_uncertainty = (
                                position_uncertainties[r]
                                if r < len(position_uncertainties)
                                else 0.0
                            )
                            row_data.extend([det_conf, assign_conf, pos_uncertainty])

                        # Add DetectionID (can be NaN for unmatched)
                        det_id = (
                            detection_ids[c] if c < len(detection_ids) else float("nan")
                        )
                        row_data.append(det_id)
                        row_data.extend(_online_identity_row_values(r))

                        # Add detection-level AprilTag columns (parallel to CNN)
                        if tag_obs_cache is not None:
                            row_data.extend(
                                get_detection_tag_csv_values(
                                    c,
                                    _tag_det_map,
                                    _tag_hamming_map,
                                    _tag_label_map_csv,
                                )
                            )

                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1

                # --- CSV for Unmatched & Final Respawn (Identical to Original) ---
                if self.csv_writer_thread:
                    for r in unmatched:
                        # No detection for this track this frame — write NaN for
                        # position/orientation so the raw CSV never contains values
                        # that do not correspond to an actual detection.
                        # Post-processing will interpolate or gap-fill these frames.
                        row_data = [
                            r,
                            trajectory_ids[r],
                            local_counts[r],
                            float("nan"),
                            float("nan"),
                            float("nan"),
                            actual_frame_index,
                            track_states[r],
                        ]

                        # Add confidence values if enabled (unmatched = 0)
                        if save_confidence:
                            det_conf = 0.0
                            assign_conf = 0.0
                            pos_uncertainty = (
                                position_uncertainties[r]
                                if r < len(position_uncertainties)
                                else 0.0
                            )
                            row_data.extend([det_conf, assign_conf, pos_uncertainty])

                        # Add DetectionID (NaN for unmatched tracks)
                        row_data.append(float("nan"))
                        row_data.extend(_online_identity_row_values(r))

                        # Add detection-level AprilTag columns (NaN — no detection)
                        if tag_obs_cache is not None:
                            row_data.extend([float("nan")] * 4)

                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1

                _committed_slots_set = (
                    set(_committed_slot_identities.keys())
                    if _committed_slot_identities
                    else set()
                )
                for d_idx in free_dets:
                    for track_idx in range(N):
                        if (
                            track_states[track_idx] == "lost"
                            and track_idx not in _committed_slots_set
                        ):
                            # Diagnostic: log slot reuse distance
                            if self.trajectories_full[track_idx]:
                                _old = self.trajectories_full[track_idx][-1]
                                _new_xy = meas[d_idx][:2]
                                _reuse_d = float(
                                    np.linalg.norm(
                                        np.array([_old[0], _old[1]])
                                        - np.array(_new_xy[:2])
                                    )
                                )
                                _body_d = float(
                                    params.get("REFERENCE_BODY_SIZE", 20.0)
                                    * params.get("RESIZE_FACTOR", 1.0)
                                )
                                if _reuse_d > _body_d * 3.0:
                                    logger.warning(
                                        f"SLOT_REUSE frame={actual_frame_index} "
                                        f"slot={track_idx} old_traj="
                                        f"{trajectory_ids[track_idx]} "
                                        f"new_traj={next_trajectory_id} "
                                        f"reuse_dist={_reuse_d:.1f} "
                                        f"old=({_old[0]:.0f},{_old[1]:.0f}) "
                                        f"new=({_new_xy[0]:.0f},"
                                        f"{_new_xy[1]:.0f})"
                                    )
                            directed_heading = bool(
                                d_idx < len(detection_directed_mask)
                                and detection_directed_mask[d_idx] == 1
                            )
                            theta_measurement = float(meas[d_idx][2])
                            if directed_heading and d_idx < len(
                                detection_directed_heading
                            ):
                                directed_theta = float(
                                    detection_directed_heading[d_idx]
                                )
                                if np.isfinite(directed_theta):
                                    theta_measurement = directed_theta
                            theta_init = self._resolve_tracking_theta(
                                track_idx,
                                theta_measurement,
                                pose_directed=directed_heading,
                                orientation_last=orientation_last,
                                fallback_theta=(
                                    self.kf_manager.X[track_idx, 2]
                                    if track_idx < len(self.kf_manager.X)
                                    else None
                                ),
                            )
                            self.kf_manager.initialize_filter(
                                track_idx,
                                np.array(
                                    [
                                        meas[d_idx][0],
                                        meas[d_idx][1],
                                        theta_init,
                                        0,
                                        0,
                                    ],
                                    np.float32,
                                ),
                            )
                            (
                                track_states[track_idx],
                                missed_frames[track_idx],
                                tracking_continuity[track_idx],
                            ) = ("active", 0, 0)
                            self.trajectories_full[track_idx].clear()
                            trajectories_pruned[track_idx].clear()
                            trajectory_ids[track_idx] = next_trajectory_id
                            orientation_last[track_idx] = theta_init
                            last_shape_info[track_idx] = (
                                shapes[d_idx] if d_idx < len(shapes) else None
                            )
                            track_pose_prototypes[track_idx] = (
                                np.asarray(
                                    detection_pose_keypoints[d_idx], dtype=np.float32
                                ).copy()
                                if (
                                    d_idx < len(detection_pose_keypoints)
                                    and detection_pose_keypoints[d_idx] is not None
                                )
                                else None
                            )
                            track_avg_step[track_idx] = 0.0
                            position_deques[track_idx].clear()
                            position_deques[track_idx].append(
                                (
                                    float(meas[d_idx][0]),
                                    float(meas[d_idx][1]),
                                    self.frame_count,
                                )
                            )
                            local_counts[track_idx] = 0
                            next_trajectory_id += 1
                            break

                profiler.tock("kf_update")

            elif detection_initialized:
                # No detections this frame — still advance the KF so predictions
                # don't freeze and costs are computed from a fresh prior on the next
                # detection frame.  Mark all tracks as occluded/lost and write NaN
                # CSV rows so every frame has an entry (required by post-processing).
                profiler.tick("kf_predict")
                self.kf_manager.predict()
                profiler.tock("kf_predict")

                for r in range(N):
                    missed_frames[r] += 1
                    if missed_frames[r] >= params["LOST_THRESHOLD_FRAMES"]:
                        track_states[r], tracking_continuity[r] = "lost", 0
                    elif track_states[r] != "lost":
                        track_states[r] = "occluded"

                if self.csv_writer_thread:
                    _save_conf_zero = params.get("SAVE_CONFIDENCE_METRICS", True)
                    _pos_unc_zero = (
                        self.kf_manager.get_position_uncertainties()
                        if _save_conf_zero
                        else []
                    )
                    for r in range(N):
                        row_data = [
                            r,
                            trajectory_ids[r],
                            local_counts[r],
                            float("nan"),
                            float("nan"),
                            float("nan"),
                            actual_frame_index,
                            track_states[r],
                        ]
                        if _save_conf_zero:
                            pos_uncertainty = (
                                _pos_unc_zero[r] if r < len(_pos_unc_zero) else 0.0
                            )
                            row_data.extend([0.0, 0.0, pos_uncertainty])
                        row_data.append(float("nan"))  # DetectionID
                        row_data.extend(_online_identity_row_values(r))
                        # Add detection-level AprilTag columns (NaN — no detection)
                        if tag_obs_cache is not None:
                            row_data.extend([float("nan")] * 4)
                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1

            # --- Individual Dataset Generation (supports YOLO OBB and BG subtraction) ---
            profiler.tick("individual_dataset")
            if individual_generator is not None and meas:
                # Get track and trajectory IDs for matched detections
                # cols contains the detection indices that were matched to tracks (rows)
                matched_track_ids = []
                matched_traj_ids = []

                if (
                    detection_initialized
                    and meas
                    and "cols" in locals()
                    and "rows" in locals()
                ):
                    # Create mapping from detection index to track info
                    det_to_track = {}
                    for r, c in zip(rows, cols):
                        det_to_track[c] = (r, trajectory_ids[r])

                    # Build lists in detection order
                    for det_idx in range(len(meas)):
                        if det_idx in det_to_track:
                            track_id, traj_id = det_to_track[det_idx]
                            matched_track_ids.append(track_id)
                            matched_traj_ids.append(traj_id)
                        else:
                            matched_track_ids.append(-1)
                            matched_traj_ids.append(-1)

                # Export all detections that already passed filtering
                # (confidence/IOU/ROI/size), regardless of assignment state.
                # Track/trajectory IDs are attached when available.
                track_ids_for_dataset = (
                    matched_track_ids if len(matched_track_ids) == len(meas) else None
                )
                traj_ids_for_dataset = (
                    matched_traj_ids if len(matched_traj_ids) == len(meas) else None
                )
                conf_for_dataset = (
                    detection_confidences if detection_confidences else None
                )
                detection_ids_for_dataset = detection_ids if detection_ids else None

                # Heading hints from head-tail model for directed canonicalization.
                _ht_hints = (
                    list(filtered_heading_hints)
                    if (
                        "filtered_heading_hints" in locals()
                        and len(filtered_heading_hints) == len(meas)
                    )
                    else None
                )
                _ht_directed = (
                    list(headtail_directed_mask)
                    if (
                        "headtail_directed_mask" in locals()
                        and len(headtail_directed_mask) == len(meas)
                    )
                    else None
                )

                # Motion-based velocity fallback: derive (vx, vy) per detection from
                # the two most-recent positions stored in each track's position_deque.
                _velocities_for_dataset = None
                if matched_track_ids and "position_deques" in locals():
                    _vel_list = []
                    for _tid in matched_track_ids:
                        if (
                            _tid >= 0
                            and _tid < len(position_deques)
                            and len(position_deques[_tid]) == 2
                        ):
                            (_x1, _y1, _f1), (_x2, _y2, _f2) = position_deques[_tid]
                            _dt = _f2 - _f1
                            if _dt != 0:
                                _vel_list.append(((_x2 - _x1) / _dt, (_y2 - _y1) / _dt))
                            else:
                                _vel_list.append(None)
                        else:
                            _vel_list.append(None)
                    if any(v is not None for v in _vel_list):
                        _velocities_for_dataset = _vel_list

                # Use original-frame coordinates for crop extraction.
                coord_scale_factor = 1.0 / resize_f

                # Filter canonical affines to match filtered detections.
                _canon_for_dataset = None
                if (
                    raw_canonical_affines is not None
                    and raw_detection_ids
                    and detection_ids
                ):
                    _raw_id_map = {}
                    for _ri, _rid in enumerate(raw_detection_ids):
                        _raw_id_map[int(_rid)] = _ri
                    _canon_for_dataset = []
                    for _did in detection_ids:
                        _ri2 = _raw_id_map.get(int(_did))
                        if (
                            _ri2 is not None
                            and _ri2 < len(raw_canonical_affines)
                            and raw_canonical_affines[_ri2] is not None
                        ):
                            _canon_for_dataset.append(raw_canonical_affines[_ri2])
                        else:
                            _canon_for_dataset.append(None)

                if filtered_obb_corners:
                    # YOLO OBB detection - use filtered OBB corners directly
                    individual_generator.process_frame(
                        frame=original_frame,
                        frame_id=actual_frame_index,
                        meas=meas,
                        obb_corners=filtered_obb_corners,
                        ellipse_params=None,
                        confidences=conf_for_dataset,
                        track_ids=track_ids_for_dataset,
                        trajectory_ids=traj_ids_for_dataset,
                        coord_scale_factor=coord_scale_factor,
                        detection_ids=detection_ids_for_dataset,
                        heading_hints=_ht_hints,
                        directed_mask=_ht_directed,
                        velocities=_velocities_for_dataset,
                        canonical_affines=_canon_for_dataset,
                    )
                elif shapes:
                    # Background subtraction - compute ellipse params from filtered shapes
                    ellipse_params = []
                    for shape in shapes:
                        area, aspect_ratio = shape[0], shape[1]
                        if aspect_ratio > 0 and area > 0:
                            ax2 = np.sqrt(4 * area / (np.pi * aspect_ratio))
                            ax1 = aspect_ratio * ax2
                            ellipse_params.append(
                                [ax1, ax2]
                            )  # [major_axis, minor_axis]
                        else:
                            # Fallback to small circle if invalid
                            ellipse_params.append([10.0, 10.0])

                    individual_generator.process_frame(
                        frame=original_frame,
                        frame_id=actual_frame_index,
                        meas=meas,
                        obb_corners=None,
                        ellipse_params=ellipse_params,
                        confidences=conf_for_dataset,
                        track_ids=track_ids_for_dataset,
                        trajectory_ids=traj_ids_for_dataset,
                        coord_scale_factor=coord_scale_factor,
                        detection_ids=detection_ids_for_dataset,
                        heading_hints=_ht_hints,
                        directed_mask=_ht_directed,
                        velocities=_velocities_for_dataset,
                        canonical_affines=_canon_for_dataset,
                    )

            profiler.tock("individual_dataset")

            # Emit progress signal periodically to avoid overwhelming the GUI thread
            # We also check that total_frames is valid
            if total_frames and total_frames > 0:
                # Emit more frequently (every 10 frames) to show better progress feedback
                # Especially important for batched detection where users need ETA
                if self.frame_count % 10 == 0:
                    percentage = int((self.frame_count * 100) / total_frames)

                    # Add mode information to status text
                    if use_cached_detections:
                        if use_batched_detection:
                            status_text = (
                                f"Tracking (batched): Frame {self.frame_count}/{total_frames} "
                                f"(abs {actual_frame_index})"
                            )
                        else:
                            status_text = (
                                f"Tracking (cached): Frame {self.frame_count}/{total_frames} "
                                f"(abs {actual_frame_index})"
                            )
                    else:
                        status_text = (
                            f"Processing: Frame {self.frame_count}/{total_frames} "
                            f"(abs {actual_frame_index})"
                        )

                    self.progress_signal.emit(percentage, status_text)

            # --- Visualization, Output & Loop Maintenance ---
            viz_free_mode = params.get("VISUALIZATION_FREE_MODE", False)

            if not viz_free_mode and overlay is not None:
                profiler.tick("visualization")
                # Incrementally prune old trajectory points instead of
                # rebuilding from trajectories_full every frame.
                _traj_horizon = int(params["TRAJECTORY_HISTORY_SECONDS"])
                if _traj_horizon >= 0:
                    _cutoff = self.frame_count - _traj_horizon
                    for _tp_list in trajectories_pruned:
                        while _tp_list and _tp_list[0][3] < _cutoff:
                            _tp_list.pop(0)

                self._draw_overlays(
                    overlay,
                    params,
                    trajectories_pruned,
                    track_states,
                    trajectory_ids,
                    tracking_continuity,
                    fg_mask,
                    bg_u8,
                    yolo_results,
                    filtered_obb_corners,  # Pass OBB corners for visualization
                    # Derive per-slot identity labels for visualization: prefer the
                    # committed belief label, then the current-frame assignment label.
                    # Falls back to empty string (→ trajectory ID display) when the
                    # online decoder is not active or no label has been assigned yet.
                    identity_labels=(
                        [
                            (
                                (
                                    _identity_online_decoder.get_belief(
                                        r
                                    ).committed_label
                                    if _identity_online_decoder.get_belief(r)
                                    is not None
                                    and _identity_online_decoder.get_belief(
                                        r
                                    ).committed_label
                                    else None
                                )
                                or (
                                    _identity_online_assignments[r].label
                                    if r in _identity_online_assignments
                                    and _identity_online_assignments[r].label
                                    else ""
                                )
                            )
                            for r in range(len(trajectory_ids))
                        ]
                        if _identity_online_decoder is not None
                        else None
                    ),
                )
                profiler.tock("visualization")

                profiler.tick("video_write")
                if self.video_writer:
                    self.video_writer.write(overlay)
                profiler.tock("video_write")

                if emit_visualization_frame:
                    profiler.tick("gui_emit")
                    # For YOLO with ROI, draw boundary overlay before emitting
                    if (
                        detection_method != "background_subtraction"
                        and ROI_mask_current is not None
                    ):
                        # Reuse cached ROI contours (computed once above).
                        if _roi_contours_cache is None:
                            _roi_contours_cache, _ = cv2.findContours(
                                ROI_mask_current,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                        if _roi_contours_cache:
                            # Draw cyan boundary
                            cv2.drawContours(
                                overlay, _roi_contours_cache, -1, (0, 255, 255), 2
                            )

                    self.emit_frame(overlay)
                    profiler.tock("gui_emit")

            # === Real-time Stats (Always emit, even in viz-free mode) ===
            current_time = time.time()
            self.frame_times.append(current_time)

            # Calculate FPS from recent frames
            if len(self.frame_times) >= 2:
                time_span = self.frame_times[-1] - self.frame_times[0]
                current_fps = (
                    (len(self.frame_times) - 1) / time_span if time_span > 0 else 0
                )
            else:
                current_fps = 0

            # Calculate elapsed and ETA
            if self.start_time is None:
                self.start_time = start_time
            elapsed = current_time - self.start_time

            if total_frames and self.frame_count > 0:
                frames_remaining = total_frames - self.frame_count
                eta = (
                    (frames_remaining / self.frame_count) * elapsed
                    if self.frame_count > 0
                    else 0
                )
            else:
                eta = 0

            # Emit stats every 10 frames to avoid overwhelming the UI
            if self.frame_count % 10 == 0:
                self.stats_signal.emit(
                    {"fps": current_fps, "elapsed": elapsed, "eta": eta}
                )

            # Finalize profiling for this frame and log periodically
            profiler.end_frame()
            profiler.log_periodic(100)

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_list.append(self.frame_count / elapsed)

        profiler.phase_end("tracking_loop")

        # Ensure cache has entries for all frames in the requested range (forward pass)
        if detection_cache and not self.backward_mode and not use_cached_detections:
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx not in cached_frame_indices:
                    detection_cache.add_frame(
                        frame_idx,
                        [],
                        [],
                        [],
                        [],
                        None,
                        [],
                        [],
                        [],
                    )

        # === 3. CLEANUP (Identical to Original) ===
        profiler.phase_start("cleanup")
        stop_requested = bool(self._stop_requested)
        # Stop frame prefetcher if still running
        if self.frame_prefetcher is not None:
            self.frame_prefetcher.stop()
            self.frame_prefetcher = None

        if live_feature_precompute is not None and not stop_requested:
            try:
                live_results = live_feature_precompute.finalize_live(
                    warning_cb=lambda title, msg: self.warning_signal.emit(title, msg)
                )
                props_path = props_path or live_results.get("pose")
                tag_observation_cache_path = (
                    tag_observation_cache_path or live_results.get("apriltag")
                )
            except Exception as exc:
                logger.exception("Realtime feature finalization failed")
                self.warning_signal.emit(
                    "Realtime Analysis Finalization Failed",
                    f"Tracking finished, but finalizing realtime analysis artifacts failed:\n{exc}",
                )
        elif live_feature_precompute is not None:
            logger.info(
                "Skipping realtime analysis finalization because stop was requested."
            )

        # === Streaming Phase 3+4 / Identity Phase 0 ===
        # Flush any pending evidence emitters so all buffered frames are written.
        if (
            hasattr(self, "_evidence_emitters")
            and self._evidence_emitters
            and not stop_requested
        ):
            for _emitter in self._evidence_emitters:
                try:
                    _emitter.flush()
                except Exception as _flush_exc:
                    logger.warning("Evidence emitter flush failed: %s", _flush_exc)
            self._evidence_emitters.clear()
        elif hasattr(self, "_evidence_emitters") and self._evidence_emitters:
            logger.info("Skipping evidence cache flush because stop was requested.")
            self._evidence_emitters.clear()

        cap.release()
        if self.video_writer:
            self.video_writer.release()

        # Persist live pose keypoints to a file-backed properties cache so the
        # rich-export merge can attach pose columns to the final CSV. Only the
        # in-memory live store carries keypoints in the InferenceRunner path;
        # without this flush the final output had no pose data even though pose
        # ran during inference. Skipped on stop/preview and when no frames exist.
        if (
            not stop_requested
            and not self.preview_mode
            and isinstance(live_pose_props_cache, LivePosePropertiesStore)
        ):
            try:
                self._flush_live_pose_cache(
                    live_pose_props_cache,
                    live_pose_keypoint_names,
                    p,
                    start_frame,
                    end_frame,
                )
            except Exception:
                logger.exception("Failed to persist live pose cache for export.")
        if pose_props_cache is not None:
            try:
                pose_props_cache.close()
            except Exception:
                pass
        if detected_props_cache is not None:
            try:
                if not stop_requested:
                    detected_props_cache.save(
                        metadata={
                            "cache_id": detected_props_id,
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "video_path": str(
                                Path(self.video_path).expanduser().resolve()
                            ),
                        }
                    )
            finally:
                detected_props_cache.close()
        # Save or close detection cache
        if detection_cache:
            if stop_requested:
                detection_cache.close()
            elif not self.backward_mode and not use_cached_detections:
                # Forward pass Phase 1 (detection phase): save cache to disk
                # Note: In batched detection, cache is already saved after Phase 1
                detection_cache.save()
                logger.info("Detection cache saved successfully")
            else:
                # Backward pass or Phase 2: just close cache (read-only mode)
                detection_cache.close()

        # Flush the refactor-native bg-sub detection cache on the forward pass:
        # close() writes the buffered per-frame detections to disk. We deliberately
        # do NOT close on the backward pass — that handle was read-only, and close()
        # would flush its empty buffer and overwrite the cache the forward pass wrote.
        if bgsub_detection_cache is not None and not self.backward_mode:
            bgsub_detection_cache.close()
            logger.info("Background-subtraction detection cache saved")

        # Flush the InferenceRunner. On a realtime forward pass this writes the
        # per-frame detection/headtail/cnn/pose caches to disk so the backward pass
        # can replay them; close() is a no-op flush for read-only (backward) handles.
        if inference_runner is not None:
            inference_runner.close()

        # Release the bg-sub runner's background model (numpy/CuPy arrays). It has
        # no cache of its own (cache_dir=None), so close() is a resource release,
        # not a flush — the worker's bgsub_detection_cache above owns persistence.
        if bgsub_runner is not None:
            bgsub_runner.close()

        # Finalize individual dataset if enabled
        if individual_generator is not None and not stop_requested:
            dataset_path = individual_generator.finalize()
            if dataset_path:
                logger.info(f"Individual dataset saved to: {dataset_path}")

        if pose_direction_applied_count > 0 or pose_direction_fallback_count > 0:
            logger.info(
                "Directed heading summary: applied=%d, fallback=%d",
                int(pose_direction_applied_count),
                int(pose_direction_fallback_count),
            )

        # --- Profiling: final summary and JSON export ---
        profiler.phase_end("cleanup")
        profiler.log_final_summary()
        # Export JSON next to the video output or detection cache, whichever is available.
        # Use a direction suffix so forward and backward profiles are kept separate.
        _dir_tag = "backward" if self.backward_mode else "forward"
        profile_export_path = None
        if self.video_output_path:
            _pbase = Path(self.video_output_path).with_suffix("")
            profile_export_path = Path(f"{_pbase}_{_dir_tag}.profile.json")
        elif self.detection_cache_path:
            profile_export_path = (
                Path(self.detection_cache_path).parent
                / f"tracking_profile_{_dir_tag}.json"
            )
        elif self.video_path:
            profile_export_path = (
                Path(self.video_path).parent / f"tracking_profile_{_dir_tag}.json"
            )
        if profile_export_path is not None and not stop_requested:
            profiler.export_summary(profile_export_path)

        logger.info("Tracking worker finished. Emitting raw trajectory data.")

        self.finished_signal.emit(
            not self._stop_requested, fps_list, self.trajectories_full
        )

    def _smooth_orientation(
        self,
        r,
        theta,
        speed,
        p,
        orientation_last,
        position_deques,
        directed_heading=False,
    ):
        from hydra_suite.core.tracking.features.orientation import smooth_orientation

        return smooth_orientation(
            r,
            theta,
            speed,
            p,
            orientation_last,
            position_deques,
            directed_heading=directed_heading,
            motion_is_reversed=bool(self.backward_mode),
        )

    def _draw_overlays(
        self,
        overlay,
        p,
        trajectories,
        track_states,
        ids,
        continuity,
        fg,
        bg,
        yolo_results=None,
        obb_corners=None,
        identity_labels=None,
    ):
        from hydra_suite.core.tracking.visualization import draw_overlays

        draw_overlays(
            overlay,
            p,
            trajectories,
            track_states,
            ids,
            continuity,
            fg,
            bg,
            kf_manager=getattr(self, "kf_manager", None),
            yolo_results=yolo_results,
            obb_corners=obb_corners,
            identity_labels=identity_labels,
        )

    # ── InferenceRunner-based pipeline helpers ────────────────────────────────

    def _resolve_cache_dir(self) -> Path:
        """Return the per-video cache directory for InferenceRunner caches."""
        video_path = Path(self.video_path)
        return video_path.parent / f".inference_cache_{video_path.stem}"

    def _build_cnn_evidence_emitter(
        self,
        cnn_cfg_dict: dict,
        live_store,
        params: dict,
    ):
        """Construct an IdentityEvidenceEmitter for one CNN phase, or None on failure.

        The emitter converts per-detection per-factor full posteriors into
        catalog-level ``IdentityEvidence`` rows (calibrated log_probs over
        catalog_size). populate_live_cnn_store calls
        ``emitter.build_frame_evidences`` and pushes the result into the live
        store via ``update_frame(..., evidences=...)`` so the online identity
        decoder can read them through ``live_store.load_evidences()``.

        Returns None if CNN classifier metadata can't be resolved (degraded
        mode: top-1 predictions only — the online decoder will under-commit).
        """
        try:
            from hydra_suite.core.tracking.identity.evidence_emitter import (
                IdentityEvidenceEmitter,
                build_evidence_cache_path,
            )
        except Exception:
            return None

        label = str(cnn_cfg_dict.get("label", "cnn_identity"))
        model_path = str(cnn_cfg_dict.get("model_path", "")).strip()
        if not model_path:
            return None
        # Prefer per-factor labels stored on the config; fall back to loading
        # the artifact metadata (cheap; reads schema-version header only).
        factor_labels = cnn_cfg_dict.get("class_names_per_factor") or []
        factor_labels = [list(f) for f in factor_labels if f]
        if not factor_labels:
            try:
                from hydra_suite.core.identity.classification.backend import (
                    ClassifierBackend,
                )

                _backend = ClassifierBackend(
                    model_path,
                    str(
                        cnn_cfg_dict.get(
                            "compute_runtime",
                            params.get("CNN_COMPUTE_RUNTIME", "cpu"),
                        )
                    ),
                )
                _meta = getattr(_backend, "metadata", None)
                if _meta is not None and hasattr(_meta, "class_names_per_factor"):
                    factor_labels = [list(f) for f in _meta.class_names_per_factor]
            except Exception:
                logger.debug(
                    "Could not resolve class_names_per_factor for CNN '%s'; "
                    "evidence emitter disabled.",
                    label,
                    exc_info=True,
                )
                return None
        if not factor_labels:
            return None

        from hydra_suite.core.identity.calibration import CalibrationModel
        from hydra_suite.core.identity.properties.cache import compute_classify_cache_id

        _calibration_temperature = float(
            cnn_cfg_dict.get(
                "calibration_temperature",
                cnn_cfg_dict.get("temperature", 1.0),
            )
        )
        _calibration_model = (
            CalibrationModel(temperature=_calibration_temperature)
            if abs(_calibration_temperature - 1.0) > 1e-6
            else None
        )
        _calibration_signature = (
            _calibration_model.signature if _calibration_model is not None else ""
        )

        try:
            classify_id = compute_classify_cache_id(
                model_path=model_path,
                compute_runtime=str(
                    params.get(
                        "CNN_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu")
                    )
                ),
                inference_model_id=str(params.get("INFERENCE_MODEL_ID", "")),
                calibration_signature=_calibration_signature,
            )
            start_frame = int(params.get("START_FRAME", 0))
            end_frame = int(params.get("END_FRAME", -1))
            if end_frame < 0:
                end_frame = start_frame
            cnn_cache_path = self._build_cnn_identity_cache_path(
                label, classify_id, start_frame, end_frame
            )
            ev_path = build_evidence_cache_path(cnn_cache_path, label, "live")
        except Exception:
            return None

        try:
            emitter = IdentityEvidenceEmitter(
                cache_path=ev_path,
                source_name=label,
                class_labels_per_factor=factor_labels,
                runtime_signature=str(params.get("COMPUTE_RUNTIME", "cpu")),
                calibration_signature=_calibration_signature,
                calibration=_calibration_model,
            )
        except Exception:
            logger.debug(
                "IdentityEvidenceEmitter construction failed for '%s'",
                label,
                exc_info=True,
            )
            return None

        live_store.set_catalog_labels(emitter.catalog_labels)
        logger.info(
            "Identity evidence emitter enabled for '%s': %s",
            label,
            ev_path,
        )
        return emitter

    def _build_inference_config_from_params(self, params: dict) -> InferenceConfig:
        """Build an InferenceConfig from tracking worker params dict.

        Maps legacy YOLO/headtail/CNN/pose/AprilTag params to the structured
        InferenceConfig dataclasses consumed by InferenceRunner.
        """
        from hydra_suite.core.inference.config import (
            AprilTagConfig,
            CNNConfig,
            HeadTailConfig,
            OBBConfig,
            OBBDirectConfig,
            OBBSequentialConfig,
            PoseConfig,
            PoseSLEAPConfig,
            PoseYOLOConfig,
        )

        compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
        # Pipeline-wide compute tier drives backend/device selection in the
        # redesign (per-stage compute_runtime fields are inert). Prefer an
        # explicit RUNTIME_TIER param; otherwise derive it from the legacy
        # per-stage runtime so old params still take effect.
        from hydra_suite.core.inference.config import migrate_runtime_to_tier

        _raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
        runtime_tier = (
            _raw_tier
            if _raw_tier in {"cpu", "gpu", "gpu_fast"}
            else migrate_runtime_to_tier({compute_runtime})
        )
        obb_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        if obb_mode not in {"direct", "sequential"}:
            obb_mode = "direct"

        direct_model_path = str(
            params.get(
                "YOLO_OBB_DIRECT_MODEL_PATH",
                params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        yolo_conf = float(params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
        yolo_iou = float(params.get("YOLO_IOU_THRESHOLD", 0.7))
        min_obj = float(params.get("MIN_OBJECT_SIZE", 0.0))
        max_obj = float(params.get("MAX_OBJECT_SIZE", float("inf")) or float("inf"))
        # Detection caps mirror legacy core/detectors/_obb_geometry:
        #   * RAW cap = 2 * MAX_TARGETS, applied at OBB extraction sorted by
        #     confidence, BEFORE size/aspect/IoU filtering.
        #   * FINAL cap = MAX_TARGETS, applied AFTER filtering, keeping the
        #     LARGEST detections (filtering sorts the cap by size, not conf).
        # Setting max_detections = MAX_TARGETS (not 2*MAX_TARGETS) restores the
        # legacy post-filter count cap (`_obb_geometry:587`) the redesign dropped.
        max_targets = max(1, int(params.get("MAX_TARGETS", 8)))
        raw_cap = 2 * max_targets
        max_dets = max_targets

        # Restrict detections to specific class IDs (legacy YOLO_TARGET_CLASSES;
        # None/empty == all classes). Threaded into OBBConfig.target_classes and
        # passed to every model.predict() (legacy yolo_detector.py:489,1078,1665).
        _target_classes_raw = params.get("YOLO_TARGET_CLASSES", None)
        target_classes = (
            [int(c) for c in _target_classes_raw] if _target_classes_raw else []
        )

        # Aspect-ratio gate (major/minor), mirroring legacy _obb_geometry: only
        # applied when enabled; bounds = ref_ar * mult. These are power-user
        # settings stored under ADVANCED_CONFIG (lowercase keys), matching legacy
        # _advanced_config_value access in core/detectors/_obb_geometry.py.
        _adv = params.get("ADVANCED_CONFIG", {}) or {}
        if _adv.get("enable_aspect_ratio_filtering", False):
            ref_ar = float(_adv.get("reference_aspect_ratio", 2.0))
            min_ar = ref_ar * float(_adv.get("min_aspect_ratio_multiplier", 0.5))
            max_ar = ref_ar * float(_adv.get("max_aspect_ratio_multiplier", 2.0))
        else:
            min_ar, max_ar = 0.0, float("inf")

        if obb_mode == "sequential":
            detect_path = str(params.get("YOLO_DETECT_MODEL_PATH", "") or "")
            crop_path = str(
                params.get("YOLO_CROP_OBB_MODEL_PATH", "") or direct_model_path
            )
            # YOLO_SEQ_* keys mirror the legacy per-stage sequential-OBB knobs
            # (yolo_detector.py:_seq_*); threading them through here keeps the
            # redesign's sequential pipeline config-driven instead of silently
            # falling back to OBBSequentialConfig's dataclass defaults.
            obb_cfg = OBBConfig(
                mode="sequential",
                sequential=OBBSequentialConfig(
                    detect_model_path=detect_path,
                    obb_model_path=crop_path,
                    detect_compute_runtime=compute_runtime,
                    obb_compute_runtime=compute_runtime,
                    detect_confidence_threshold=float(
                        params.get("YOLO_SEQ_DETECT_CONF_THRESHOLD", 0.25)
                    ),
                    obb_confidence_threshold=yolo_conf,
                    detect_image_size=int(params.get("YOLO_SEQ_DETECT_IMGSZ", 0)),
                    crop_pad_ratio=float(params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15)),
                    min_crop_size_px=float(
                        params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64.0)
                    ),
                    enforce_square_crop=bool(
                        params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True)
                    ),
                    stage2_image_size=int(params.get("YOLO_SEQ_STAGE2_IMGSZ", 160)),
                    stage2_batch_size=(
                        int(params["YOLO_SEQ_INDIVIDUAL_BATCH_SIZE"])
                        if params.get("YOLO_SEQ_INDIVIDUAL_BATCH_SIZE")
                        else None
                    ),
                ),
                target_classes=target_classes,
                confidence_threshold=yolo_conf,
                iou_threshold=yolo_iou,
                min_object_size=min_obj,
                max_object_size=max_obj,
                min_aspect_ratio=min_ar,
                max_aspect_ratio=max_ar,
                max_detections=max_dets,
                raw_detection_cap=raw_cap,
            )
        else:
            obb_cfg = OBBConfig(
                mode="direct",
                direct=OBBDirectConfig(
                    model_path=direct_model_path,
                    compute_runtime=compute_runtime,
                    confidence_floor=1e-3,
                    confidence_threshold=yolo_conf,
                ),
                target_classes=target_classes,
                confidence_threshold=yolo_conf,
                iou_threshold=yolo_iou,
                min_object_size=min_obj,
                max_object_size=max_obj,
                min_aspect_ratio=min_ar,
                max_aspect_ratio=max_ar,
                max_detections=max_dets,
                raw_detection_cap=raw_cap,
            )

        # HeadTail
        headtail_model_path = str(
            params.get("YOLO_HEADTAIL_MODEL_PATH", "") or ""
        ).strip()
        headtail_cfg = None
        if headtail_model_path and os.path.exists(headtail_model_path):
            ht_runtime = str(
                params.get(
                    "HEADTAIL_COMPUTE_RUNTIME",
                    params.get("COMPUTE_RUNTIME", "cpu"),
                )
            )
            headtail_cfg = HeadTailConfig(
                model_path=headtail_model_path,
                compute_runtime=ht_runtime,
                confidence_threshold=float(
                    params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.5)
                ),
                # Mirrors legacy's separate, stricter head-tail candidate gate
                # (_select_headtail_candidate_indices): detections below this
                # confidence never get classified at all (stay undirected),
                # independent of the main OBB filter's own confidence_threshold.
                candidate_confidence_threshold=float(
                    params.get(
                        "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD",
                        params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
                    )
                ),
                batch_size=int(params.get("HEADTAIL_BATCH_SIZE", 64)),
                canonical_aspect_ratio=float(
                    params.get("ADVANCED_CONFIG", {}).get("reference_aspect_ratio", 2.0)
                ),
                canonical_margin=float(
                    params.get("ADVANCED_CONFIG", {}).get(
                        "yolo_headtail_canonical_margin", 1.3
                    )
                ),
            )

        # CNN phases
        cnn_phases: list[CNNConfig] = []
        cnn_runtime = str(
            params.get("CNN_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
        )
        for cnn_cfg_dict in params.get("CNN_CLASSIFIERS", []):
            cnn_model_path = str(cnn_cfg_dict.get("model_path", "")).strip()
            if not cnn_model_path or not os.path.exists(cnn_model_path):
                continue
            cnn_label = str(cnn_cfg_dict.get("label", "cnn_identity"))
            cnn_phases.append(
                CNNConfig(
                    label=cnn_label,
                    model_path=cnn_model_path,
                    compute_runtime=cnn_runtime,
                    confidence_threshold=float(cnn_cfg_dict.get("confidence", 0.5)),
                    batch_size=int(cnn_cfg_dict.get("batch_size", 64)),
                    scoring_mode=str(cnn_cfg_dict.get("scoring_mode", "atomic")),
                    match_bonus=float(cnn_cfg_dict.get("match_bonus", 0.1)),
                    mismatch_penalty=float(cnn_cfg_dict.get("mismatch_penalty", 0.3)),
                    calibration_temperature=float(
                        cnn_cfg_dict.get(
                            "calibration_temperature",
                            cnn_cfg_dict.get("temperature", 1.0),
                        )
                    ),
                )
            )

        # Pose — supports both YOLO-pose and SLEAP backends.
        pose_cfg = None
        if bool(params.get("ENABLE_POSE_EXTRACTOR", False)):
            pose_model_type = str(params.get("POSE_MODEL_TYPE", "")).strip().lower()
            pose_runtime = str(
                params.get("POSE_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
            )
            common_pose_kwargs = dict(
                skeleton_file=str(params.get("POSE_SKELETON_FILE", "") or "").strip(),
                crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
                suppress_foreign_regions=bool(
                    params.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)
                ),
                min_keypoint_confidence=float(
                    params.get("POSE_MIN_KPT_CONF_VALID", 0.2)
                ),
                min_valid_keypoints=int(
                    params.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 1)
                ),
                anterior_keypoints=list(
                    params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []) or []
                ),
                posterior_keypoints=list(
                    params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []) or []
                ),
                ignore_keypoints=list(params.get("POSE_IGNORE_KEYPOINTS", []) or []),
                overrides_headtail=bool(params.get("POSE_OVERRIDES_HEADTAIL", True)),
            )
            sleap_model_path = str(
                params.get("POSE_SLEAP_MODEL_DIR", params.get("POSE_MODEL_DIR", ""))
                or ""
            ).strip()
            yolo_model_path = str(
                params.get(
                    "POSE_YOLO_MODEL_DIR",
                    params.get(
                        "POSE_MODEL_PATH",
                        params.get("YOLO_POSE_MODEL_PATH", ""),
                    ),
                )
                or ""
            ).strip()
            if pose_model_type == "sleap" and sleap_model_path:
                pose_cfg = PoseConfig(
                    backend="sleap",
                    sleap=PoseSLEAPConfig(
                        model_path=sleap_model_path,
                        compute_runtime=pose_runtime,
                        batch_size=int(params.get("POSE_BATCH_SIZE", 4)),
                    ),
                    **common_pose_kwargs,
                )
            elif yolo_model_path and os.path.exists(yolo_model_path):
                pose_cfg = PoseConfig(
                    backend="yolo",
                    yolo=PoseYOLOConfig(
                        model_path=yolo_model_path,
                        compute_runtime=pose_runtime,
                        confidence_threshold=float(
                            params.get("POSE_CONFIDENCE_THRESHOLD", 1e-4)
                        ),
                        iou_threshold=float(params.get("POSE_IOU_THRESHOLD", 0.7)),
                        max_detections_per_crop=1,
                        batch_size=int(params.get("POSE_BATCH_SIZE", 64)),
                    ),
                    **common_pose_kwargs,
                )

        # AprilTag
        apriltag_cfg = AprilTagConfig(
            enabled=bool(params.get("USE_APRILTAGS", False)),
            tag_family=str(params.get("APRILTAG_FAMILY", "tag36h11")),
            threads=int(params.get("APRILTAG_THREADS", 4)),
            max_hamming=int(params.get("APRILTAG_MAX_HAMMING", 1)),
            decimate=float(params.get("APRILTAG_DECIMATE", 1.0)),
            blur=float(params.get("APRILTAG_BLUR", 0.8)),
            crop_padding=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
        )

        batch_size = int(params.get("YOLO_BATCH_SIZE", params.get("BATCH_SIZE", 1)))

        return InferenceConfig(
            obb=obb_cfg,
            headtail=headtail_cfg,
            cnn_phases=cnn_phases,
            pose=pose_cfg,
            apriltag=apriltag_cfg,
            detection_batch_size=batch_size,
            realtime=False,
            use_cache=True,
            runtime_tier=runtime_tier,
        )

    def _emit_inference_progress(self, done: int, total: int) -> None:
        """Translate batch-pass progress to the existing progress_signal."""
        if total > 0:
            pct = int(done * 100 / total)
            self.progress_signal.emit(pct, f"Inference pass: {done}/{total} frames")
