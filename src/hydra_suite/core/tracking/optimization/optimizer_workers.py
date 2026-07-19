"""Pure tracking-preview and detection-cache helpers for the parameter
optimizer UI.

``run_tracking_preview`` runs the tracking/assignment/rendering loop over
cached detections and reports each rendered frame via a callback; the
``_preview_*`` helpers are its building blocks. This module has no Qt
dependency — the background-thread wrappers that drive it from the GUI
(``DetectionCacheBuildWorker``, ``TrackingPreviewWorker``) live in the
trackerkit app layer.
"""

import logging
import math
from collections import deque
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

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
from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.runner import _open_caches, video_signature

logger = logging.getLogger(__name__)


def _preview_filter_cached_detections(det_filter, cache, f_idx, roi_mask):
    """Read a frame from cache and apply detection filtering for preview.

    Detection caches are always ``OBBResult`` (InferenceRunner-based
    builder); filtering is applied via ``apply_detection_filter`` from
    ``core/inference/api``.
    """
    frame_data = cache.read_frame(f_idx)

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

    raise TypeError(
        "detection cache must contain OBBResult frames "
        f"(got {type(frame_data).__name__}); rebuild the cache with the "
        "current InferenceRunner-based builder."
    )


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


def run_tracking_preview(
    video_path: str,
    detection_cache_path: str,
    start_frame: int,
    end_frame: int,
    params: Dict[str, Any],
    *,
    frame_cb: Optional[Callable[[np.ndarray], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Pure tracking-preview loop, extracted from ``TrackingPreviewWorker.run``.

    Reads cached detections + video frames over ``[start_frame, end_frame]``,
    runs the same Kalman/assignment/rendering pipeline as the live preview
    worker, and reports each rendered RGB frame via ``frame_cb`` (single
    ``np.ndarray`` argument, matching ``frame_signal.emit(rgb)``). The loop
    stops early if ``stop_check()`` returns True.

    No Qt types are referenced here; ``TrackingPreviewWorker.run`` delegates
    to this function and wires ``frame_cb``/``stop_check`` to its own
    signal/flag.
    """
    from pathlib import Path

    cap = cv2.VideoCapture(video_path)
    # Open the InferenceRunner detection cache read-only. This handle
    # must never have close() called on it: DetectionCacheHandle.close()
    # flushes its (empty, since we never write) buffer and would clobber
    # the on-disk cache with zero frames (see optimizer._open_and_validate_cache).
    cfg = build_inference_config_from_params(params)
    cache = _open_caches(
        cfg,
        Path(detection_cache_path),
        video_signature(video_path),
    ).detection
    _pose_cache = None
    try:
        if not cap.isOpened():
            logger.error("PreviewWorker: could not open video: %s", video_path)
            return
        if cache is None or not cache.is_valid():
            logger.error("PreviewWorker: incompatible detection cache.")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        kf_manager = KalmanFilterManager(params["MAX_TARGETS"], params)
        assigner = TrackAssigner(params)

        # Correction 21: _ParamsFilter is a lightweight shim that exposes .params
        # so _preview_filter_cached_detections can read DETECTION_CONFIDENCE from
        # it regardless of whether the cache returns OBBResult or a legacy 12-tuple.
        class _ParamsFilter:
            def __init__(self, p):
                self.params = p

        det_filter = _ParamsFilter(params)
        _roi_mask = params.get("ROI_MASK", None)

        N = params["MAX_TARGETS"]

        (
            _pose_cache,
            _pose_anterior,
            _pose_posterior,
            _pose_ignore,
            _pose_enabled,
        ) = _pf_load_pose_context(params)
        _pose_min_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
        _pose_kpt_map: dict = {}
        _pose_kpt_map_frame = None
        track_pose_prototypes: list = [None] * N
        track_states, tracking_continuity = ["lost"] * N, [0] * N
        missed_frames = [0] * N
        trajectory_ids, next_trajectory_id = list(range(N)), N
        orientation_last: list = [None] * N
        last_shape_info = [None] * N
        lost_threshold = params.get("LOST_THRESHOLD_FRAMES", 5)
        resize_f = params.get("RESIZE_FACTOR", 1.0)

        traj_colors = params.get("TRAJECTORY_COLORS", [])
        if not traj_colors:
            np.random.seed(42)
            traj_colors = [
                tuple(int(c) for c in row)
                for row in np.random.randint(0, 255, (max(N, 32), 3))
            ]

        _TRAIL_LEN = int(params.get("TRAJECTORY_HISTORY_SECONDS", 5))
        _trail_maxlen = None if _TRAIL_LEN < 0 else max(_TRAIL_LEN, 1)
        trail: list[deque] = [deque(maxlen=_trail_maxlen) for _ in range(N)]

        show_circles = params.get("SHOW_CIRCLES", True)
        show_orientation = params.get("SHOW_ORIENTATION", True)
        show_trails = params.get("SHOW_TRAJECTORIES", True)
        show_labels = params.get("SHOW_LABELS", True)

        for f_idx in range(start_frame, end_frame + 1):
            if stop_check() if stop_check is not None else False:
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
            ) = _preview_filter_cached_detections(det_filter, cache, f_idx, _roi_mask)

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
                    params.get(
                        "POSE_DIRECTION_MIN_VISIBILITY",
                        max(
                            0.6,
                            params.get("POSE_REJECTION_MIN_VISIBILITY", 0.5),
                        ),
                    ),
                    0.0,
                    1.0,
                )
            )
            _pose_direction_min_keypoints = max(
                1,
                int(params.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 3)),
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
                        params.get("POSE_OVERRIDES_HEADTAIL", True)
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
                newly_initialized, next_trajectory_id = _preview_init_free_detections(
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

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            if frame_cb is not None:
                frame_cb(rgb)
    except Exception:
        logger.exception("PreviewWorker encountered an error.")
    finally:
        cap.release()
        # Do NOT call cache.close() here: this is a read-only
        # DetectionCacheHandle and close() would clobber the on-disk
        # cache (see the comment where it is opened, above).
        if _pose_cache is not None:
            try:
                _pose_cache.close()
            except Exception:
                pass
