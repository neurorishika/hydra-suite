"""QThread workers for the parameter optimizer UI.

DetectionCacheBuilderWorker — builds a detection cache for a frame range.
TrackingPreviewWorker — emits preview frames using cached detections.
"""

import logging
import math
from collections import deque
from typing import Any, Dict

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from hydra_suite.core.assigners.hungarian import TrackAssigner
from hydra_suite.core.filters.kalman import KalmanFilterManager
from hydra_suite.core.identity.geometry import (
    build_detection_direction_overrides as _pf_build_direction_overrides,
)
from hydra_suite.core.identity.geometry import normalize_theta as _pf_normalize_theta
from hydra_suite.core.identity.geometry import (
    resolve_detection_tracking_theta as _pf_resolve_detection_tracking_theta,
)
from hydra_suite.core.identity.pose.features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from hydra_suite.core.identity.pose.features import (
    compute_detection_pose_features as _pf_compute_det_features,
)
from hydra_suite.core.identity.pose.features import (
    is_pose_heading_reliable as _pf_heading_reliable,
)
from hydra_suite.core.identity.pose.features import (
    load_pose_context_from_params as _pf_load_pose_context,
)
from hydra_suite.core.inference.api import (
    apply_detection_filter as _apply_detection_filter,
)

# Correction 21: use new DetectionCacheHandle; fall back to legacy if present.
try:
    from hydra_suite.core.inference.cache.detection import (
        DetectionCacheHandle as DetectionCache,
    )
except ImportError:
    from hydra_suite.data.detection_cache import (
        DetectionCache,  # type: ignore[no-redef]
    )

logger = logging.getLogger(__name__)


def _preview_filter_cached_detections(det_filter, cache, f_idx, roi_mask):
    """Read a frame from cache and apply detection filtering for preview.

    Supports both new OBBResult API (Correction 21) and legacy 12-tuple.
    """
    frame_data = cache.get_frame(f_idx)

    from hydra_suite.core.inference.result import OBBResult as _OBBResult

    if isinstance(frame_data, _OBBResult):
        from hydra_suite.core.inference.config import OBBConfig

        conf_threshold = 0.0
        if hasattr(det_filter, "params"):
            conf_threshold = float(det_filter.params.get("DETECTION_CONFIDENCE", 0.0))
        elif hasattr(det_filter, "confidence_threshold"):
            conf_threshold = float(det_filter.confidence_threshold)

        _cfg = OBBConfig(confidence_threshold=conf_threshold)
        filtered_obb = _apply_detection_filter(frame_data, _cfg)
        meas = np.concatenate(
            [filtered_obb.centroids, filtered_obb.angles[:, None]], axis=1
        ).tolist()
        shapes = filtered_obb.shapes.tolist()
        _confs = filtered_obb.confidences.tolist()
        detection_ids = filtered_obb.detection_ids.tolist()
        return meas, shapes, _confs, detection_ids, [], []

    # Legacy 12-tuple path
    (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confs,
        raw_obb,
        raw_det_ids,
        raw_heading_hints,
        raw_heading_confidences,
        raw_directed_mask,
        _raw_canonical_affines,
        _raw_canvas_dims,
        _raw_M_inverse,
    ) = frame_data
    if raw_heading_hints:
        filtered = det_filter.filter_raw_detections(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confs,
            raw_obb,
            roi_mask=roi_mask,
            detection_ids=raw_det_ids,
            heading_hints=raw_heading_hints,
            heading_confidences=raw_heading_confidences,
            directed_mask=raw_directed_mask,
        )
        (
            meas,
            _,
            shapes,
            _confs,
            _obb_out,
            detection_ids,
            _headtail_hints,
            _,
            _headtail_directed,
        ) = filtered
    else:
        filtered = det_filter.filter_raw_detections(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confs,
            raw_obb,
            roi_mask=roi_mask,
            detection_ids=raw_det_ids,
        )
        meas, _, shapes, _confs, _obb_out, detection_ids = filtered
        _headtail_hints, _, _headtail_directed = [], [], []
    return meas, shapes, _confs, detection_ids, _headtail_hints, _headtail_directed


def _preview_compute_pose_features(
    meas,
    detection_ids,
    f_idx,
    pose_enabled,
    pose_cache,
    pose_kpt_map,
    pose_kpt_map_frame,
    pose_anterior,
    pose_posterior,
    pose_ignore,
    pose_min_conf,
):
    """Compute per-detection pose features for preview; returns updated kpt_map state."""
    _det_pose_kpts: list = [None] * len(meas)
    _det_pose_vis = np.zeros(len(meas), dtype=np.float32)
    _det_pose_headings: list = [None] * len(meas)
    if pose_enabled and meas and detection_ids:
        if pose_kpt_map_frame != f_idx:
            pose_kpt_map = _pf_build_keypoint_map(pose_cache, f_idx)
            pose_kpt_map_frame = f_idx
        _det_pose_kpts, _det_pose_vis, _det_pose_headings = _pf_compute_det_features(
            [int(d) for d in detection_ids],
            pose_kpt_map,
            pose_anterior,
            pose_posterior,
            pose_ignore,
            pose_min_conf,
            return_headings=True,
        )
    return (
        _det_pose_kpts,
        _det_pose_vis,
        _det_pose_headings,
        pose_kpt_map,
        pose_kpt_map_frame,
    )


def _preview_process_matched_tracks(
    matched_r,
    matched_c,
    meas,
    detection_directed_mask,
    detection_directed_heading,
    kf_manager,
    orientation_last,
    track_states,
    trail,
    _det_pose_kpts,
    track_pose_prototypes,
):
    """Correct KF state for matched tracks and update prototypes (preview)."""
    for r, c in zip(matched_r, matched_c):
        m = np.asarray(meas[c], dtype=np.float32)
        _pose_d = (
            bool(detection_directed_mask[c])
            if c < len(detection_directed_mask)
            else False
        )
        theta_cor = _pf_resolve_detection_tracking_theta(
            r,
            float(m[2]),
            (
                detection_directed_heading[c]
                if c < len(detection_directed_heading)
                else math.nan
            ),
            _pose_d,
            orientation_last,
        )
        m_cor = np.array([m[0], m[1], theta_cor], dtype=np.float32)
        if track_states[r] == "lost":
            trail[r].clear()
            kf_manager.initialize_filter(
                r,
                np.array([m_cor[0], m_cor[1], theta_cor, 0.0, 0.0], dtype=np.float32),
            )
        kf_manager.correct(r, m_cor)
        orientation_last[r] = _pf_normalize_theta(float(kf_manager.X[r, 2]))

    for r, c in zip(matched_r, matched_c):
        proto = _det_pose_kpts[c] if c < len(_det_pose_kpts) else None
        if proto is not None:
            track_pose_prototypes[r] = np.asarray(proto, dtype=np.float32).copy()


def _preview_init_free_detections(
    free_dets,
    N,
    meas,
    detection_directed_mask,
    detection_directed_heading,
    kf_manager,
    orientation_last,
    track_states,
    trail,
    matched_r,
    _det_pose_kpts,
    track_pose_prototypes,
    missed_frames,
    tracking_continuity,
    trajectory_ids,
    next_trajectory_id,
):
    """Assign free detections to lost track slots (preview worker)."""
    newly_initialized: set = set()
    existing_matched = set(matched_r)
    for d_idx in free_dets:
        for r in range(N):
            if (
                r not in existing_matched | newly_initialized
                and track_states[r] == "lost"
            ):
                m = np.asarray(meas[d_idx], dtype=np.float32)
                _pose_d = (
                    bool(detection_directed_mask[d_idx])
                    if d_idx < len(detection_directed_mask)
                    else False
                )
                theta_cor = _pf_resolve_detection_tracking_theta(
                    r,
                    float(m[2]),
                    (
                        detection_directed_heading[d_idx]
                        if d_idx < len(detection_directed_heading)
                        else math.nan
                    ),
                    _pose_d,
                    orientation_last,
                )
                kf_manager.initialize_filter(
                    r,
                    np.array([m[0], m[1], theta_cor, 0.0, 0.0], dtype=np.float32),
                )
                trail[r].clear()
                orientation_last[r] = _pf_normalize_theta(theta_cor)
                track_states[r] = "active"
                missed_frames[r] = 0
                tracking_continuity[r] = 0
                trajectory_ids[r] = next_trajectory_id
                next_trajectory_id += 1
                newly_initialized.add(r)
                proto = _det_pose_kpts[d_idx] if d_idx < len(_det_pose_kpts) else None
                if proto is not None:
                    track_pose_prototypes[r] = np.asarray(
                        proto, dtype=np.float32
                    ).copy()
                break
    return newly_initialized, next_trajectory_id


def _preview_render_tracks(
    display,
    N,
    track_states,
    kf_manager,
    trail,
    trajectory_ids,
    traj_colors,
    show_circles,
    show_orientation,
    show_trails,
    show_labels,
):
    """Render tracking overlay on the display frame."""
    for r in range(N):
        if track_states[r] == "lost":
            continue
        col = traj_colors[r % len(traj_colors)]
        x, y = float(kf_manager.X[r, 0]), float(kf_manager.X[r, 1])
        theta = float(kf_manager.X[r, 2])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        pt = (int(x), int(y))
        if show_trails and len(trail[r]) > 1:
            pts = np.array(list(trail[r]), dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], isClosed=False, color=col, thickness=2)
        if show_circles:
            cv2.circle(display, pt, 7, col, -1)
        if show_orientation:
            ex = int(x + 18 * math.cos(theta))
            ey = int(y + 18 * math.sin(theta))
            cv2.arrowedLine(display, pt, (ex, ey), col, 2, tipLength=0.4)
        if show_labels:
            state_tag = "" if track_states[r] == "active" else f" ({track_states[r]})"
            cv2.putText(
                display,
                f"T{trajectory_ids[r]}{state_tag}",
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )


class DetectionCacheBuilderWorker(QThread):
    """
    Phase-1-only worker: runs YOLO detection on a frame range and writes a
    DetectionCache.  Does NOT run Kalman tracking, pose precompute, CSV
    writing, interpolation, or any other production-pipeline stage.

    Used by the Parameter Helper to build a minimal detection cache so the
    Bayesian optimizer can run without triggering the full tracking pipeline.
    """

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)  # (success, cache_path)

    def __init__(
        self,
        video_path: str,
        cache_path: str,
        params: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_path = cache_path
        self.params = params.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        import time
        from collections import deque

        from hydra_suite.core.detectors import create_detector
        from hydra_suite.utils.batch_optimizer import BatchOptimizer

        # --- Load detector (YOLO model) ---
        try:
            detector = create_detector(self.params)
        except Exception as e:
            logger.error("DetectionCacheBuilder: could not create detector: %s", e)
            self.finished_signal.emit(False, "")
            return

        cap = cv2.VideoCapture(self.video_path)
        cache = None
        try:
            if not cap.isOpened():
                logger.error(
                    "DetectionCacheBuilder: could not open video: %s", self.video_path
                )
                self.finished_signal.emit(False, "")
                return

            resize_f = self.params.get("RESIZE_FACTOR", 1.0)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_f)
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_f)

            advanced = self.params.get("ADVANCED_CONFIG", {}).copy()
            advanced["enable_tensorrt"] = self.params.get("ENABLE_TENSORRT", False)
            advanced["tensorrt_max_batch_size"] = self.params.get(
                "TENSORRT_MAX_BATCH_SIZE", 16
            )
            advanced["tracking_realtime_mode"] = self.params.get(
                "TRACKING_REALTIME_MODE", False
            )
            advanced["tracking_workflow_mode"] = self.params.get(
                "TRACKING_WORKFLOW_MODE", "non_realtime"
            )
            batch_size = BatchOptimizer(advanced).estimate_batch_size(
                fw, fh, self.params.get("YOLO_MODEL_PATH", "")
            )

            cache = DetectionCache(
                self.cache_path,
                mode="w",
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            total_frames = self.end_frame - self.start_frame + 1
            rel_idx = 0  # 0-based position relative to start_frame
            batch_times: deque = deque(maxlen=30)

            while rel_idx < total_frames and not self._stop_requested:
                batch_start = rel_idx
                batch_frames = []
                while len(batch_frames) < batch_size and rel_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if resize_f != 1.0:
                        frame = cv2.resize(
                            frame,
                            (0, 0),
                            fx=resize_f,
                            fy=resize_f,
                            interpolation=cv2.INTER_AREA,
                        )
                    batch_frames.append(frame)
                    rel_idx += 1

                if not batch_frames:
                    break

                bt0 = time.time()
                results = detector.detect_objects_batched(
                    batch_frames, batch_start, None, return_raw=True
                )
                batch_times.append(time.time() - bt0)

                for local_i, (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    raw_hints,
                    raw_hint_confidences,
                    raw_directed,
                ) in enumerate(results):
                    actual_f = self.start_frame + batch_start + local_i
                    det_ids = [actual_f * 10000 + i for i in range(len(raw_meas))]
                    cache.add_frame(
                        actual_f,
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        det_ids,
                        raw_hints,
                        raw_hint_confidences,
                        raw_directed,
                    )

                pct = int(rel_idx * 100 / total_frames)
                avg = sum(batch_times) / len(batch_times) if batch_times else 0
                fps = (len(batch_frames) / avg) if avg > 0 else 0
                eta_s = int((total_frames - rel_idx) / fps) if fps > 0 else -1
                eta_str = f"  ETA {eta_s}s" if eta_s >= 0 else ""
                self.progress_signal.emit(
                    pct, f"Building detection cache: {pct}%{eta_str}"
                )

            if self._stop_requested:
                self.progress_signal.emit(0, "Cancelled.")
                self.finished_signal.emit(False, "")
                return

            cache.save()
            cache.close()
            cache = None
            logger.info("DetectionCacheBuilder: cache saved to %s", self.cache_path)
            self.finished_signal.emit(True, self.cache_path)

        except Exception:
            logger.exception("DetectionCacheBuilder error")
            self.finished_signal.emit(False, "")
        finally:
            cap.release()
            if cache is not None:
                try:
                    cache.close()
                except Exception:
                    pass


class TrackingPreviewWorker(QThread):
    """
    Emits visualization frames for previewing optimization results.
    """

    frame_signal = Signal(np.ndarray)
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        cache = DetectionCache(self.detection_cache_path, mode="r")
        try:
            if not cap.isOpened():
                logger.error("PreviewWorker: could not open video: %s", self.video_path)
                return
            if not cache.is_compatible():
                logger.error("PreviewWorker: incompatible detection cache.")
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            kf_manager = KalmanFilterManager(self.params["MAX_TARGETS"], self.params)
            assigner = TrackAssigner(self.params)

            # Correction 21: _ParamsFilter is a lightweight shim that exposes .params
            # so _preview_filter_cached_detections can read DETECTION_CONFIDENCE from
            # it regardless of whether the cache returns OBBResult or a legacy 12-tuple.
            class _ParamsFilter:
                def __init__(self, p):
                    self.params = p

            det_filter = _ParamsFilter(self.params)
            _roi_mask = self.params.get("ROI_MASK", None)

            N = self.params["MAX_TARGETS"]

            (
                _pose_cache,
                _pose_anterior,
                _pose_posterior,
                _pose_ignore,
                _pose_enabled,
            ) = _pf_load_pose_context(self.params)
            _pose_min_conf = float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
            _pose_kpt_map: dict = {}
            _pose_kpt_map_frame = None
            track_pose_prototypes: list = [None] * N
            track_states, tracking_continuity = ["lost"] * N, [0] * N
            missed_frames = [0] * N
            trajectory_ids, next_trajectory_id = list(range(N)), N
            orientation_last: list = [None] * N
            last_shape_info = [None] * N
            lost_threshold = self.params.get("LOST_THRESHOLD_FRAMES", 5)
            resize_f = self.params.get("RESIZE_FACTOR", 1.0)

            traj_colors = self.params.get("TRAJECTORY_COLORS", [])
            if not traj_colors:
                np.random.seed(42)
                traj_colors = [
                    tuple(int(c) for c in row)
                    for row in np.random.randint(0, 255, (max(N, 32), 3))
                ]

            _TRAIL_LEN = int(self.params.get("TRAJECTORY_HISTORY_SECONDS", 5))
            _trail_maxlen = None if _TRAIL_LEN < 0 else max(_TRAIL_LEN, 1)
            trail: list[deque] = [deque(maxlen=_trail_maxlen) for _ in range(N)]

            show_circles = self.params.get("SHOW_CIRCLES", True)
            show_orientation = self.params.get("SHOW_ORIENTATION", True)
            show_trails = self.params.get("SHOW_TRAJECTORIES", True)
            show_labels = self.params.get("SHOW_LABELS", True)

            for f_idx in range(self.start_frame, self.end_frame + 1):
                if self._stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                (
                    meas,
                    shapes,
                    _confs,
                    detection_ids,
                    _headtail_hints,
                    _headtail_directed,
                ) = _preview_filter_cached_detections(
                    det_filter, cache, f_idx, _roi_mask
                )

                kf_manager.predict()

                (
                    _det_pose_kpts,
                    _det_pose_vis,
                    _det_pose_headings,
                    _pose_kpt_map,
                    _pose_kpt_map_frame,
                ) = _preview_compute_pose_features(
                    meas,
                    detection_ids,
                    f_idx,
                    _pose_enabled,
                    _pose_cache,
                    _pose_kpt_map,
                    _pose_kpt_map_frame,
                    _pose_anterior,
                    _pose_posterior,
                    _pose_ignore,
                    _pose_min_conf,
                )
                _pose_direction_min_visibility = float(
                    np.clip(
                        self.params.get(
                            "POSE_DIRECTION_MIN_VISIBILITY",
                            max(
                                0.6,
                                self.params.get("POSE_REJECTION_MIN_VISIBILITY", 0.5),
                            ),
                        ),
                        0.0,
                        1.0,
                    )
                )
                _pose_direction_min_keypoints = max(
                    1,
                    int(self.params.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 3)),
                )
                _pose_heading_mask = [
                    (
                        1
                        if _pf_heading_reliable(
                            _det_pose_kpts[idx] if idx < len(_det_pose_kpts) else None,
                            (
                                float(_det_pose_vis[idx])
                                if idx < len(_det_pose_vis)
                                else 0.0
                            ),
                            min_visibility=_pose_direction_min_visibility,
                            min_valid_keypoints=_pose_direction_min_keypoints,
                        )
                        else 0
                    )
                    for idx in range(len(meas))
                ]
                detection_directed_heading, detection_directed_mask = (
                    _pf_build_direction_overrides(
                        len(meas),
                        _det_pose_headings,
                        _pose_heading_mask,
                        _headtail_hints,
                        _headtail_directed,
                        pose_overrides_headtail=bool(
                            self.params.get("POSE_OVERRIDES_HEADTAIL", True)
                        ),
                    )
                )
                _association_data: dict = {
                    "detection_pose_heading": detection_directed_heading,
                    "detection_pose_keypoints": _det_pose_kpts,
                    "detection_pose_visibility": _det_pose_vis,
                    "track_pose_prototypes": track_pose_prototypes,
                    "track_avg_step": np.zeros(N, dtype=np.float32),
                }

                if meas:
                    cost, _ = assigner.compute_cost_matrix(
                        N,
                        meas,
                        kf_manager.X,
                        shapes,
                        kf_manager,
                        last_shape_info,
                        meas_ori_directed=(
                            detection_directed_mask
                            if len(detection_directed_mask) == len(meas)
                            else None
                        ),
                        association_data=_association_data,
                    )
                    matched_r, matched_c, free_dets, _identity_rejoin_pairs = (
                        assigner.assign_tracks(
                            cost,
                            N,
                            len(meas),
                            meas,
                            track_states,
                            tracking_continuity,
                            kf_manager,
                            association_data=_association_data,
                            missed_frames=missed_frames,
                        )
                    )
                    _preview_process_matched_tracks(
                        matched_r,
                        matched_c,
                        meas,
                        detection_directed_mask,
                        detection_directed_heading,
                        kf_manager,
                        orientation_last,
                        track_states,
                        trail,
                        _det_pose_kpts,
                        track_pose_prototypes,
                    )
                    newly_initialized, next_trajectory_id = (
                        _preview_init_free_detections(
                            free_dets,
                            N,
                            meas,
                            detection_directed_mask,
                            detection_directed_heading,
                            kf_manager,
                            orientation_last,
                            track_states,
                            trail,
                            matched_r,
                            _det_pose_kpts,
                            track_pose_prototypes,
                            missed_frames,
                            tracking_continuity,
                            trajectory_ids,
                            next_trajectory_id,
                        )
                    )
                else:
                    matched_r, matched_c, newly_initialized = [], [], set()

                # --- State management ---
                matched_r_set = set(matched_r) | newly_initialized
                for r in matched_r:
                    missed_frames[r] = 0
                    track_states[r] = "active"
                    tracking_continuity[r] += 1
                for r in range(N):
                    if r not in matched_r_set and track_states[r] != "lost":
                        missed_frames[r] += 1
                        if missed_frames[r] >= lost_threshold:
                            track_states[r] = "lost"
                            tracking_continuity[r] = 0
                        else:
                            track_states[r] = "occluded"
                for r, c in zip(matched_r, matched_c):
                    last_shape_info[r] = shapes[c]

                # Update trails
                for r in range(N):
                    if track_states[r] != "lost":
                        x, y = float(kf_manager.X[r, 0]), float(kf_manager.X[r, 1])
                        if math.isfinite(x) and math.isfinite(y):
                            trail[r].append((int(x), int(y)))
                    else:
                        trail[r].clear()

                display = cv2.resize(frame, (0, 0), fx=resize_f, fy=resize_f)
                _preview_render_tracks(
                    display,
                    N,
                    track_states,
                    kf_manager,
                    trail,
                    trajectory_ids,
                    traj_colors,
                    show_circles,
                    show_orientation,
                    show_trails,
                    show_labels,
                )

                self.frame_signal.emit(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                self.msleep(20)
        except Exception:
            logger.exception("PreviewWorker encountered an error.")
        finally:
            cap.release()
            cache.close()
            if _pose_cache is not None:
                try:
                    _pose_cache.close()
                except Exception:
                    pass
            self.finished_signal.emit()
