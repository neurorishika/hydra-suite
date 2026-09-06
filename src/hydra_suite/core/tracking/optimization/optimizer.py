"""
Tracking Parameter Optimizer and Previewer.
Enhanced with Dynamic Bayesian Optimization.
"""

import logging
import math
from collections.abc import Mapping
from typing import Any, Dict

import numpy as np
import optuna

from hydra_suite.core.assigners.hungarian import TrackAssigner
from hydra_suite.core.filters.kalman import KalmanFilterManager
from hydra_suite.core.individual.geometry import (
    build_detection_direction_overrides as _pf_build_direction_overrides,
)
from hydra_suite.core.individual.geometry import normalize_theta as _pf_normalize_theta
from hydra_suite.core.individual.geometry import (
    resolve_detection_tracking_theta as _pf_resolve_detection_tracking_theta,
)
from hydra_suite.core.individual.pose.features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from hydra_suite.core.individual.pose.features import (
    compute_detection_pose_features as _pf_compute_det_features,
)
from hydra_suite.core.individual.pose.features import (
    is_pose_heading_reliable as _pf_heading_reliable,
)
from hydra_suite.core.individual.pose.features import (
    load_pose_context_from_params as _pf_load_pose_context,
)

# The optimizer reads a detection cache built by DetectionCacheBuildWorker
# (core/tracking/optimization/optimizer_workers.py), which now writes an
# InferenceRunner-format cache (key-constructed DetectionCacheHandle) instead
# of the legacy flat DetectionCache. Building and reading must derive their
# InferenceConfig from the SAME params so the cache key matches; a mismatch
# makes the handle read back as empty.
from hydra_suite.core.inference.runner import (
    _open_caches,
    frame_space_roi_mask,
    video_signature,
)
from hydra_suite.core.inference.stages.filtering import filter_for_source
from hydra_suite.core.tracking.arenas import arena_ids_for_meas as _meas_arena_ids
from hydra_suite.core.tracking.arenas import (
    arena_layout_from_params,
    check_slot_arena_covers_all_slots,
    tracking_frame_size,
)
from hydra_suite.core.tracking.optimization.detection_config import (
    inference_config_for_optimizer_params,
)
from hydra_suite.core.tracking.optimization.parameter_contract import (
    PARAM_RANGES,
    quantize_tracking_autotune_params,
    quantize_tracking_autotune_value,
)
from hydra_suite.core.tracking.optimization.production_replay import (
    ProductionReplayEvaluator,
    cache_directory,
    sanitize_replay_tuning_config,
)
from hydra_suite.core.tracking.optimization.unlabeled_scoring import (
    CandidateEvaluation,
    MetricDirection,
    MetricSpec,
    aggregate_candidate_evaluations,
    build_train_validation_split,
    decide_heldout_baseline_protection,
    forward_backward_cycle_consistency,
    global_slot_alignment,
    output_sanity_metrics,
    pareto_ranks,
    partition_temporal_segments,
    recommend_dominating_candidate,
    trajectory_quality_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions extracted from _run_tracking_loop to reduce complexity
# ---------------------------------------------------------------------------


def _load_pose_context_for_loop(self_obj, params):
    """Load pose context, preferring pre-loaded in-memory data when available."""
    if (
        hasattr(self_obj, "_pose_run_context")
        and self_obj._pose_run_context is not None
    ):
        _, _pose_anterior, _pose_posterior, _pose_ignore, _pose_enabled = (
            self_obj._pose_run_context
        )
        _pose_frame_data = getattr(self_obj, "_pose_frame_cache", {})
    else:
        (
            _tmp_pose_cache,
            _pose_anterior,
            _pose_posterior,
            _pose_ignore,
            _pose_enabled,
        ) = _pf_load_pose_context(params)
        _pose_frame_data = {}
        if _tmp_pose_cache is not None:
            for _fi in range(self_obj.start_frame, self_obj.end_frame + 1):
                _pose_frame_data[_fi] = _pf_build_keypoint_map(_tmp_pose_cache, _fi)
            try:
                _tmp_pose_cache.close()
            except Exception:
                pass
    return (
        _pose_anterior,
        _pose_posterior,
        _pose_ignore,
        _pose_enabled,
        _pose_frame_data,
    )


def _normalise_scoring_weights(params):
    """Read and normalise the six scoring-weight parameters to sum=1."""
    _w_cov = max(float(params.get("SCORE_WEIGHT_COVERAGE", 0.25)), 0.0)
    _w_asn = max(float(params.get("SCORE_WEIGHT_ASSIGNMENT", 0.15)), 0.0)
    _w_frg = max(float(params.get("SCORE_WEIGHT_FRAGMENTATION", 0.20)), 0.0)
    _w_occ = max(float(params.get("SCORE_WEIGHT_OCCLUSION", 0.10)), 0.0)
    _w_vel = max(float(params.get("SCORE_WEIGHT_VELOCITY", 0.20)), 0.0)
    _w_crd = max(float(params.get("SCORE_WEIGHT_CROWDING", 0.10)), 0.0)
    _w_sum = _w_cov + _w_asn + _w_frg + _w_occ + _w_vel + _w_crd
    if _w_sum < 1e-9:
        _w_cov = _w_asn = _w_frg = _w_occ = _w_vel = 0.2
        _w_crd = 0.0
        _w_sum = 1.0
    _w_cov /= _w_sum
    _w_asn /= _w_sum
    _w_frg /= _w_sum
    _w_occ /= _w_sum
    _w_vel /= _w_sum
    _w_crd /= _w_sum
    return _w_cov, _w_asn, _w_frg, _w_occ, _w_vel, _w_crd


def _optimizer_frame_size(video_path, params):
    """Coordinate space of the cached detections, for the arena lookup.

    The optimizer never reads a live tracking frame (it replays a detection
    cache), so it mirrors worker.py's cached-detection fallback: the capture's
    native size scaled by ``RESIZE_FACTOR``. ``None`` (unopenable video) means
    "assume the label image's own resolution", the same fallback worker.py's
    `target_w is None` sites take.
    """
    import cv2 as _cv2

    cap = _cv2.VideoCapture(str(video_path))
    try:
        base_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        base_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception:  # pragma: no cover - defensive
        return None
    finally:
        cap.release()
    return tracking_frame_size(params, base_w, base_h)


def _filter_cached_detections(
    det_filter,
    cache,
    f_idx,
    roi_mask,
    *,
    apply_max_detections: bool = True,
):
    """Read a frame from detection cache and apply filtering.

    Detection caches are always ``OBBResult`` (InferenceRunner-based
    builder).  Replay filters through the same source-aware production
    dispatcher used by ``InferenceRunner.load_frame``.
    """
    frame_data = cache.read_frame(f_idx)

    from hydra_suite.core.inference.result import OBBResult as _OBBResult

    if isinstance(frame_data, _OBBResult):
        inference_config = getattr(det_filter, "inference_config", None)
        if inference_config is None:
            inference_config = inference_config_for_optimizer_params(det_filter.params)
        if apply_max_detections:
            # Preserve the ordinary production call shape exactly; the
            # diagnostic-only pre-cap mode below is intentionally opt-in.
            filtered_obb, _ = filter_for_source(inference_config, frame_data, roi_mask)
        else:
            filtered_obb, _ = filter_for_source(
                inference_config,
                frame_data,
                roi_mask,
                apply_max_detections=False,
            )

        # Convert OBBResult back to the legacy format expected by the tracking loop.
        meas = np.concatenate(
            [filtered_obb.centroids, filtered_obb.angles[:, None]], axis=1
        ).tolist()
        shapes = filtered_obb.shapes.tolist()
        _confs = filtered_obb.confidences.tolist()
        detection_ids = filtered_obb.detection_ids.tolist()
        _headtail_hints: list = []
        _headtail_directed: list = []
        return meas, shapes, _confs, detection_ids, _headtail_hints, _headtail_directed

    raise TypeError(
        "detection cache must contain OBBResult frames "
        f"(got {type(frame_data).__name__}); rebuild the cache with the "
        "current InferenceRunner-based builder."
    )


def _compute_pose_features_for_frame(
    meas,
    detection_ids,
    f_idx,
    pose_enabled,
    pose_frame_data,
    pose_anterior,
    pose_posterior,
    pose_ignore,
    pose_min_conf,
):
    """Compute per-detection pose features for a single frame."""
    _det_pose_kpts: list = [None] * len(meas)
    _det_pose_vis = np.zeros(len(meas), dtype=np.float32)
    _det_pose_headings: list = [None] * len(meas)
    if pose_enabled and meas and detection_ids:
        _frame_kpt_map = pose_frame_data.get(f_idx, {})
        _det_pose_kpts, _det_pose_vis, _det_pose_headings = _pf_compute_det_features(
            [int(d) for d in detection_ids],
            _frame_kpt_map,
            pose_anterior,
            pose_posterior,
            pose_ignore,
            pose_min_conf,
            return_headings=True,
        )
    return _det_pose_kpts, _det_pose_vis, _det_pose_headings


def _respawn_free_detections(
    free_dets,
    N,
    meas,
    shapes,
    track_states,
    missed_frames,
    tracking_continuity,
    trajectory_ids,
    next_trajectory_id,
    orientation_last,
    last_shape_info,
    track_pose_prototypes,
    track_avg_step,
    kf_manager,
    detection_directed_mask,
    detection_directed_heading,
    _det_pose_kpts,
):
    """Assign free detections to lost track slots (Phase-3 respawn)."""
    for d_idx in free_dets:
        for track_idx in range(N):
            if track_states[track_idx] == "lost":
                _pose_d_f = (
                    bool(detection_directed_mask[d_idx])
                    if d_idx < len(detection_directed_mask)
                    else False
                )
                theta_init = _pf_resolve_detection_tracking_theta(
                    track_idx,
                    float(meas[d_idx][2]),
                    (
                        detection_directed_heading[d_idx]
                        if d_idx < len(detection_directed_heading)
                        else math.nan
                    ),
                    _pose_d_f,
                    orientation_last,
                    fallback_theta=(
                        float(kf_manager.X[track_idx, 2])
                        if track_idx < len(kf_manager.X)
                        else None
                    ),
                )
                kf_manager.initialize_filter(
                    track_idx,
                    np.array(
                        [meas[d_idx][0], meas[d_idx][1], theta_init, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                )
                track_states[track_idx] = "active"
                missed_frames[track_idx] = 0
                tracking_continuity[track_idx] = 0
                trajectory_ids[track_idx] = next_trajectory_id
                next_trajectory_id += 1
                orientation_last[track_idx] = theta_init
                last_shape_info[track_idx] = (
                    shapes[d_idx] if d_idx < len(shapes) else None
                )
                track_pose_prototypes[track_idx] = (
                    np.asarray(_det_pose_kpts[d_idx], dtype=np.float32).copy()
                    if (
                        d_idx < len(_det_pose_kpts)
                        and _det_pose_kpts[d_idx] is not None
                    )
                    else None
                )
                track_avg_step[track_idx] = 0.0
                break
    return next_trajectory_id


def _update_unmatched_track_states(
    N,
    matched_r_set,
    track_states,
    missed_frames,
    tracking_continuity,
    lost_threshold,
):
    """Age unmatched tracks toward 'occluded' then 'lost'."""
    for r in range(N):
        if r not in matched_r_set and track_states[r] != "lost":
            missed_frames[r] += 1
            if missed_frames[r] >= lost_threshold:
                track_states[r] = "lost"
                tracking_continuity[r] = 0
            else:
                track_states[r] = "occluded"


def _accumulate_velocity_metrics(
    N,
    track_states,
    kf_manager,
    _prev_positions,
    _prev_vecs,
    _step_norms,
    _direction_reversals,
    _body_size,
):
    """Accumulate inter-frame velocity and direction-reversal metrics."""
    _MOVE_MIN = 0.2 * _body_size
    for r in range(N):
        if track_states[r] != "lost":
            curr = kf_manager.X[r, :2].copy()
            if r in _prev_positions:
                step_vec = curr - _prev_positions[r]
                step = float(np.linalg.norm(step_vec))
                _step_norms.append(step / max(_body_size, 1e-6))
                if step > _MOVE_MIN and r in _prev_vecs:
                    prev_v = _prev_vecs[r]
                    prev_norm = float(np.linalg.norm(prev_v))
                    if prev_norm > _MOVE_MIN:
                        cos_a = float(np.dot(step_vec, prev_v) / (step * prev_norm))
                        _direction_reversals.append(1.0 if cos_a < -0.3 else 0.0)
                if step > _MOVE_MIN:
                    _prev_vecs[r] = step_vec
            _prev_positions[r] = curr
        else:
            _prev_positions.pop(r, None)
            _prev_vecs.pop(r, None)


def _accumulate_crowding_metric(
    N,
    track_states,
    kf_manager,
    _body_size,
    _n_pairs,
):
    """Compute per-frame crowding violation for active track pairs."""
    active_slots = [r for r in range(N) if track_states[r] == "active"]
    if len(active_slots) < 2:
        return 0.0
    frame_crowd = 0.0
    for _ci in range(len(active_slots)):
        for _cj in range(_ci + 1, len(active_slots)):
            _d = float(
                np.linalg.norm(
                    kf_manager.X[active_slots[_ci], :2]
                    - kf_manager.X[active_slots[_cj], :2]
                )
            )
            if _d < _body_size:
                frame_crowd += 1.0 - _d / max(_body_size, 1e-6)
    return frame_crowd / _n_pairs


def _compute_composite_score(
    n_frames,
    N,
    _coverage_sum,
    _occlusion_sum,
    _assign_cost_sum,
    _assign_count,
    _suspicious_assign_count,
    _det_count_sum,
    _max_continuity,
    _step_norms,
    _direction_reversals,
    _crowding_sum,
    _crowding_frames,
    _w_cov,
    _w_asn,
    _w_frg,
    _w_occ,
    _w_vel,
    _w_crd,
):
    """Compute the composite multi-objective score from accumulated metrics."""
    _cov_frac = _coverage_sum / max(n_frames, 1)
    _mean_dets = _det_count_sum / max(n_frames, 1)
    # Normalize excess detections as a fraction of the expected target count.
    # The previous expression divided by N twice, making this penalty vanish as
    # MAX_TARGETS increased.
    _det_excess = min(max(_mean_dets / max(N, 1.0) - 1.0, 0.0) / 0.5, 1.0)
    coverage_cost = 0.80 * (1.0 - _cov_frac) + 0.20 * _det_excess
    assign_cost = _assign_cost_sum / _assign_count if _assign_count > 0 else 1.0
    _cont_frac = min(sum(_max_continuity) / max(N, 1) / max(n_frames, 1), 1.0)
    _suspicious_rate = _suspicious_assign_count / max(_assign_count, 1)
    frag_cost = 0.70 * (1.0 - _cont_frac) + 0.30 * _suspicious_rate
    occlusion_cost = _occlusion_sum / max(n_frames, 1)

    if _step_norms:
        _steps_arr = np.asarray(_step_norms, dtype=np.float32)
        _vel_median = float(np.median(_steps_arr))
        _vel_p95 = float(np.percentile(_steps_arr, 95))
        magnitude_cost = min(
            0.5 * min(_vel_median / 3.0, 1.0) + 0.5 * min(_vel_p95 / 8.0, 1.0),
            1.0,
        )
    else:
        magnitude_cost = 1.0
    direction_cost = (
        float(np.mean(_direction_reversals)) if _direction_reversals else 0.0
    )
    velocity_cost = 0.4 * magnitude_cost + 0.6 * direction_cost
    crowding_cost = _crowding_sum / max(_crowding_frames, 1)

    sub_scores = {
        "coverage": coverage_cost,
        "assignment": assign_cost,
        "fragmentation": frag_cost,
        "occlusion": occlusion_cost,
        "velocity": velocity_cost,
        "crowding": crowding_cost,
    }
    sub_scores_arr = np.array(
        [
            coverage_cost,
            assign_cost,
            frag_cost,
            occlusion_cost,
            velocity_cost,
            crowding_cost,
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [_w_cov, _w_asn, _w_frg, _w_occ, _w_vel, _w_crd], dtype=np.float64
    )
    composite = float(np.dot(weights, sub_scores_arr))
    return composite, sub_scores


# Backwards-compatible module export. The typed lower-layer parameter contract
# is the source of truth for bounds and Qt-representable precision.
_PARAM_RANGES: Dict[str, tuple] = dict(PARAM_RANGES)

_UNLABELED_METRIC_SPECS = (
    MetricSpec("cycle_loss", MetricDirection.MINIMIZE),
    MetricSpec("cycle_observation_coverage", MetricDirection.MAXIMIZE),
    MetricSpec("coverage_loss", MetricDirection.MINIMIZE),
    MetricSpec("worst_track_coverage_loss", MetricDirection.MINIMIZE),
    MetricSpec("fragmentation_loss", MetricDirection.MINIMIZE),
    MetricSpec("motion_roughness_loss", MetricDirection.MINIMIZE),
    MetricSpec("collision_loss", MetricDirection.MINIMIZE),
    MetricSpec("detection_excess_loss", MetricDirection.MINIMIZE),
)

_VALIDATION_REGION_COUNT = 4
_MIN_REGION_FRAMES = 3
_MIN_CYCLE_SHARED_OBSERVATIONS = 3
_MIN_CYCLE_SHARED_COVERAGE = 0.5
_TEMPORAL_HORIZON_PARAMETERS = (
    "LOST_THRESHOLD_FRAMES",
    "KALMAN_MATURITY_AGE",
)
_PROPOSAL_METRICS = (
    "cycle_loss",
    "coverage",
    "assignment",
    "fragmentation",
    "occlusion",
    "velocity",
    "crowding",
)

# This is a search-allocation term only, not a held-out promotion metric.  The
# saturating transform prevents an outlying cycle loss from dominating the
# native replay composite before the diverse production shortlist can inspect
# it.
_PROPOSAL_CYCLE_TERM_WEIGHT = 0.25


def _bounded_cycle_proposal_penalty(cycle_loss: float) -> float:
    """Map a non-negative cycle loss monotonically into ``[0, weight)``."""

    if not np.isfinite(cycle_loss) or cycle_loss < 0:
        raise ValueError("cycle_loss must be finite and non-negative")
    return float(_PROPOSAL_CYCLE_TERM_WEIGHT * cycle_loss / (1.0 + cycle_loss))


# Upper real-time limits mirror TrackerKit's seconds-based controls.  Dynamic
# frame ranges are capped here so every generated candidate remains applyable.
_LIFECYCLE_MAX_SECONDS = {
    "KALMAN_MATURITY_AGE": 8.0,
    "LOST_THRESHOLD_FRAMES": 40.0,
}


def _dense_frame_positions(
    frame_positions: Dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    n_tracks: int,
) -> np.ndarray:
    """Convert a replay's frame map to chronological ``(F, N, 2)`` form."""

    dense = np.full((end_frame - start_frame + 1, n_tracks, 2), np.nan, np.float32)
    for frame_idx, positions in frame_positions.items():
        if start_frame <= int(frame_idx) <= end_frame:
            value = np.asarray(positions, dtype=np.float32)
            if value.shape == (n_tracks, 2):
                dense[int(frame_idx) - start_frame] = value
    return dense


class OptimizationResult:
    def __init__(
        self,
        params: Dict[str, Any],
        score: float,
        trial_number: int,
        sub_scores: Dict[str, float] | None = None,
        *,
        candidate_id: str | None = None,
        is_baseline: bool = False,
    ):
        self.params = params
        self.score = score
        self.trial_number = trial_number
        self.sub_scores: Dict[str, float] = sub_scores or {}
        self.candidate_id = candidate_id or (
            "baseline" if is_baseline else f"trial-{trial_number}"
        )
        self.is_baseline = bool(is_baseline)
        self.pareto_rank: int | None = None
        self.recommended = False
        self.recommendation_reason = "not production-validated"
        self.validation_metrics: Dict[str, Any] = {}
        self.mean_relative_spread: float | None = None


class TrackingOptimizerCore:
    """
    Runs Bayesian optimization on a video slice with different parameter sets.
    Requires a pre-populated DetectionCache for speed.

    This is a plain (non-Qt) core class. Progress and results are surfaced via
    optional callbacks (``progress_cb`` / ``result_cb``); the thin Qt wrapper
    ``TrackingOptimizer`` (trackerkit/gui/workers/param_optimizer_worker.py)
    forwards these to Qt signals. Cooperative cancellation is via
    ``request_stop()`` setting ``self._stop_requested``.
    """

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        base_params: Dict[str, Any],
        tuning_config: Dict[str, bool],
        n_trials: int = 50,
        n_seeds: int = 3,
        on_plateau: str = "restart",
        sampler_type: str = "auto",
        progress_cb=None,
        result_cb=None,
        error_cb=None,
    ):
        self._progress_cb = progress_cb
        self._result_cb = result_cb
        self._error_cb = error_cb
        self.video_path = video_path
        # TrackerKit historically stores the member file from a production
        # cache (``.../detection.npz``), whereas optimizer/preview readers
        # consume the cache directory. Normalize the public boundary once so
        # all core callers support either valid form.
        self.detection_cache_path = str(cache_directory(detection_cache_path))
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.base_params = base_params
        (
            self.tuning_config,
            self.disabled_tuning_dimensions,
        ) = sanitize_replay_tuning_config(tuning_config, base_params)
        self.n_trials = n_trials
        self.n_seeds = max(1, n_seeds)
        self.on_plateau = on_plateau  # "restart" | "stop"
        self.sampler_type = sampler_type  # "auto" | "gp" | "tpe"
        self._stop_requested = False
        self._search_converged = False
        self.cache = None
        # ROI is fixed over an optimization run, while every proposal repeatedly
        # filters cached native-frame detections. Resolve it lazily once rather
        # than reopening the video for each forward/backward proposal replay.
        self._native_roi_mask_source_id: int | None = None
        self._native_roi_mask: np.ndarray | None = None
        self._native_roi_mask_initialized = False

    # Alias to module-level constant so existing internal references keep working.
    _PARAM_RANGES = _PARAM_RANGES  # type: ignore[assignment]

    def _build_sampler(self, n_active: int):
        """
        Construct the Optuna sampler based on ``self.sampler_type``.

        "auto"  — OptunaHub AutoSampler: uses GPSampler for early trials then
                  falls back to TPE.  Best overall choice.
        "gp"    — GPSampler with Matérn-2.5 + ARD + log-EI.  Fastest convergence
                  for budgets ≤ 500 trials; needs scipy + torch.
        "tpe"   — Multivariate TPE.  Robust fallback that needs no extra deps.
        """
        stype = self.sampler_type

        if stype == "auto":
            try:
                import optunahub  # type: ignore[import-untyped]

                return optunahub.load_module("samplers/auto_sampler").AutoSampler()
            except Exception as e:
                logger.warning(
                    "AutoSampler unavailable (%s), falling back to GPSampler.", e
                )
                stype = "gp"  # fall through to GP

        if stype == "gp":
            try:
                qmc = optuna.samplers.QMCSampler(
                    qmc_type="sobol",
                    seed=42,
                    independent_sampler=optuna.samplers.RandomSampler(seed=42),
                )
                return optuna.samplers.GPSampler(
                    seed=42,
                    n_startup_trials=max(10, n_active),
                    deterministic_objective=True,  # tracking is fully deterministic
                    independent_sampler=qmc,
                )
            except Exception as e:
                logger.warning(
                    "GPSampler unavailable (%s), falling back to multivariate TPE.", e
                )
                # fall through to tpe

        # "tpe" (default / fallback)
        return optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=max(20, n_active * 2),
            seed=42,
        )

    def request_stop(self):
        self._stop_requested = True

    def _frame_space_roi_mask_for_params(
        self, params: Dict[str, Any]
    ) -> np.ndarray | None:
        """Return the cached native-frame ROI for a fixed evaluation contract."""

        raw_mask = params.get("ROI_MASK", None)
        source_id = id(raw_mask)
        if (
            not self._native_roi_mask_initialized
            or self._native_roi_mask_source_id != source_id
        ):
            self._native_roi_mask = frame_space_roi_mask(raw_mask, self.video_path)
            self._native_roi_mask_source_id = source_id
            self._native_roi_mask_initialized = True
        return self._native_roi_mask

    # ------------------------------------------------------------------
    # run() helpers
    # ------------------------------------------------------------------

    def _open_and_validate_cache(self) -> bool:
        """Open the InferenceRunner detection cache and validate compatibility.

        ``self.detection_cache_path`` is normalized to the cache **directory**
        written by ``DetectionCacheBuildWorker``. The handle is opened
        read-only here: it must never have ``close()`` called on it, because
        ``DetectionCacheHandle.close()`` flushes its (empty, since we never
        write) buffer and would clobber the on-disk cache with zero frames.

        Returns True if the cache is ready for use, False on failure
        (after emitting appropriate error signals).
        """
        try:
            from pathlib import Path

            _cfg = inference_config_for_optimizer_params(self.base_params)
            _roi_mask = self.base_params.get("ROI_MASK", None)
            caches = _open_caches(
                _cfg,
                Path(self.detection_cache_path),
                video_signature(self.video_path),
                _roi_mask,
                read_only=True,
            )
            self.cache = caches.detection
            if (
                not caches.set_manifest_valid
                or self.cache is None
                or not self.cache.is_valid()
            ):
                msg = (
                    "Detection cache is incompatible with the current parameters "
                    "(e.g. detection method/model or ROI mask changed since it was "
                    "built). Rebuild the cache and try again."
                )
                if self._progress_cb is not None:
                    self._progress_cb(0, f"Error: {msg}")
                if self._error_cb is not None:
                    self._error_cb(msg)
                return False
        except Exception as e:
            logger.exception("Optimizer: failed to open detection cache")
            msg = f"Error loading detection cache: {e}"
            if self._progress_cb is not None:
                self._progress_cb(0, msg)
            if self._error_cb is not None:
                self._error_cb(msg)
            return False

        if not self.cache.covers_frame_range(self.start_frame, self.end_frame):
            missing = self.cache.get_missing_frames(self.start_frame, self.end_frame)
            msg = (
                f"Detection cache does not cover frames "
                f"{self.start_frame}-{self.end_frame}. Missing: {missing}. "
                "Rebuild the detection cache for this frame range and try again."
            )
            if self._progress_cb is not None:
                self._progress_cb(0, f"Error: {msg}")
            if self._error_cb is not None:
                self._error_cb(msg)
            return False
        return True

    def _preload_pose_data(self) -> None:
        """Pre-load pose keypoints into memory so per-trial loops stay fast."""
        self._pose_run_context = (None, [], [], [], False)
        self._pose_frame_cache: dict = {}
        _pose_ctx = _pf_load_pose_context(self.base_params)
        if _pose_ctx[0] is not None and _pose_ctx[4]:
            _tmp_cache, _ant, _post, _ign, _enabled = _pose_ctx
            self._pose_run_context = (None, _ant, _post, _ign, _enabled)
            for _fi in range(self.start_frame, self.end_frame + 1):
                self._pose_frame_cache[_fi] = _pf_build_keypoint_map(_tmp_cache, _fi)
            try:
                _tmp_cache.close()
            except Exception:
                pass
            logger.info(
                "Optimizer: pre-loaded pose keypoints for %d frames into memory.",
                len(self._pose_frame_cache),
            )
        else:
            if _pose_ctx[0] is not None:
                try:
                    _pose_ctx[0].close()
                except Exception:
                    pass

    _SEED_DEFAULTS: Dict[str, Any] = {
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_DISTANCE_MULTIPLIER": 3.0,
        "KALMAN_NOISE_COVARIANCE": 0.03,
        "KALMAN_MEASUREMENT_NOISE_COVARIANCE": 0.1,
        "W_POSITION": 1.0,
        "W_ORIENTATION": 0.5,
        "W_AREA": 0.2,
        "W_ASPECT": 0.2,
        "KALMAN_DAMPING": 0.95,
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.0,
        "KALMAN_INITIAL_VELOCITY_RETENTION": 0.2,
        "LOST_THRESHOLD_FRAMES": 10,
        "KALMAN_MATURITY_AGE": 5,
    }

    def _search_range(self, key: str) -> tuple[str, float, float]:
        """Return a search range in the parameter's production units.

        Lifecycle parameters are stored as frames, but fixed frame bounds have
        different real-time meanings at different FPS. Explore dimensionless
        factors around the user's current duration instead, which preserves the
        same seconds-scale neighborhood across videos.
        """

        ptype, low, high = self._PARAM_RANGES[key]
        if key not in {"LOST_THRESHOLD_FRAMES", "KALMAN_MATURITY_AGE"}:
            return ptype, float(low), float(high)
        raw_base = self.base_params.get(key, self._SEED_DEFAULTS[key])
        try:
            base = max(float(raw_base), float(low))
        except (TypeError, ValueError):
            base = float(self._SEED_DEFAULTS[key])
        fps = max(float(self.base_params.get("FPS", 1.0)), 1e-6)
        representable_high = max(
            float(low), math.floor(_LIFECYCLE_MAX_SECONDS[key] * fps)
        )
        search_high = min(representable_high, max(float(low), math.ceil(base * 4.0)))
        search_low = min(search_high, max(float(low), math.floor(base * 0.25)))
        return ptype, search_low, search_high

    def _build_seed_trial(self) -> Dict[str, Any]:
        """Build the initial seed trial from the user's current parameters.

        Only representable values are enqueued.  The exact, unclamped baseline
        is evaluated separately before search, so an out-of-range production
        setting is never silently relabelled as the user's baseline.
        """
        seed_params: Dict[str, Any] = {}
        for key in self._PARAM_RANGES:
            if not self.tuning_config.get(key):
                continue
            ptype, low, high = self._search_range(key)
            raw = self.base_params.get(key, self._SEED_DEFAULTS.get(key, low))
            try:
                value = quantize_tracking_autotune_value(key, raw)
            except ValueError:
                # The exact baseline is evaluated separately. An invalid
                # external production value must not be silently rounded into
                # a different seed candidate.
                continue
            if low <= value <= high and not (ptype == "log_float" and value <= 0):
                seed_params[key] = value
        return seed_params

    def _perturb_near_base(self, rng, scale: float) -> dict:
        """Sample a parameter point perturbed around base_params.

        Works in the natural space for each parameter type:
        - log_float: Normal(log(base), scale*(log(high)-log(low))), then exp.
        - float    : Normal(base, scale*(high-low)), clipped to [low, high].
        - int      : same as float, then rounded and clipped.

        ``scale`` is the fraction of the total range used as sigma.  A scale of 0.1
        stays close to the current settings; 0.3 allows more exploration.
        """
        pt: Dict[str, Any] = {}
        for key in self._PARAM_RANGES:
            if not self.tuning_config.get(key):
                continue
            ptype, low, high = self._search_range(key)
            base_val = self.base_params.get(key)
            if ptype == "log_float":
                log_low, log_high = np.log(low), np.log(high)
                log_center = (
                    np.log(float(base_val))
                    if base_val is not None
                    else (log_low + log_high) / 2.0
                )
                sigma = scale * (log_high - log_low)
                pt[key] = float(
                    np.exp(np.clip(rng.normal(log_center, sigma), log_low, log_high))
                )
            elif ptype == "float":
                center = float(base_val) if base_val is not None else (low + high) / 2.0
                sigma = scale * (high - low)
                pt[key] = float(np.clip(rng.normal(center, sigma), low, high))
            else:  # int
                center = float(base_val) if base_val is not None else (low + high) / 2.0
                sigma = scale * (high - low)
                pt[key] = int(np.clip(round(rng.normal(center, sigma)), low, high))
        return quantize_tracking_autotune_params(pt)

    def _random_from_ranges(self, rng) -> dict:
        """Uniform-random point across the full search space (used for plateau restarts)."""
        pt: Dict[str, Any] = {}
        for key in self._PARAM_RANGES:
            if not self.tuning_config.get(key):
                continue
            ptype, low, high = self._search_range(key)
            if ptype == "log_float":
                pt[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            elif ptype == "float":
                pt[key] = float(rng.uniform(low, high))
            else:  # int
                pt[key] = int(rng.integers(low, high + 1))
        return quantize_tracking_autotune_params(pt)

    def _suggest_trial_params(self, trial, scaled_body_size: float) -> Dict[str, Any]:
        """Use the Optuna trial to suggest values for all enabled parameters."""
        trial_params: Dict[str, Any] = {}

        # One canonical range table drives seeds, random restarts, and Optuna
        # suggestions.  This prevents the three paths from silently exploring
        # different contracts.
        for name in self._PARAM_RANGES:
            if not self.tuning_config.get(name):
                continue
            ptype, low, high = self._search_range(name)
            if ptype == "int":
                trial_params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                trial_params[name] = trial.suggest_float(
                    name, float(low), float(high), log=ptype == "log_float"
                )

        # Every proposal is evaluated at the same decimal precision TrackerKit
        # can later write into its QDoubleSpinBoxes. Derive the pixel threshold
        # only after canonicalizing the body-length multiplier.
        trial_params = quantize_tracking_autotune_params(trial_params)

        # Derived parameter: MAX_DISTANCE_THRESHOLD from multiplier
        if "MAX_DISTANCE_MULTIPLIER" in trial_params:
            trial_params["MAX_DISTANCE_THRESHOLD"] = (
                trial_params["MAX_DISTANCE_MULTIPLIER"] * scaled_body_size
            )

        return trial_params

    def _temporal_validation_horizon(self) -> int:
        """Largest temporal effect horizon represented by the active search."""

        horizon = _MIN_REGION_FRAMES
        for name in _TEMPORAL_HORIZON_PARAMETERS:
            if not self.tuning_config.get(name):
                continue
            _, _, high = self._search_range(name)
            horizon = max(horizon, int(math.ceil(high)))
        return horizon

    def _minimum_validation_frames(self) -> int:
        """Frames needed for four paired regions with usable temporal evidence."""

        return _VALIDATION_REGION_COUNT * self._temporal_validation_horizon()

    def _validation_support_reason(
        self, validation_bounds: tuple[int, int]
    ) -> str | None:
        """Return why a held-out tail is insufficient for temporal promotion."""

        start_frame, end_frame = validation_bounds
        frame_count = end_frame - start_frame + 1
        required = self._minimum_validation_frames()
        if frame_count < required:
            return (
                "held-out temporal support is inadequate: requires at least "
                f"{required} frames for {_VALIDATION_REGION_COUNT} paired regions "
                f"at a {self._temporal_validation_horizon()}-frame horizon "
                f"(found {frame_count})"
            )
        return None

    def _search_and_validation_bounds(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int] | None]:
        """Reserve a chronological tail for held-out production replay."""

        frame_count = self.end_frame - self.start_frame + 1
        minimum_validation = self._minimum_validation_frames()
        minimum_train = max(8, self._temporal_validation_horizon())
        gap_frames = 1 if frame_count >= 12 else 0
        if frame_count < minimum_train + gap_frames + minimum_validation:
            return (self.start_frame, self.end_frame), None
        validation_fraction = max(0.25, minimum_validation / frame_count)
        split = build_train_validation_split(
            frame_count, validation_fraction=validation_fraction, gap_frames=gap_frames
        )
        train = split.train[0]
        validation = split.validation[0]
        return (
            (self.start_frame + train.start, self.start_frame + train.stop - 1),
            (
                self.start_frame + validation.start,
                self.start_frame + validation.stop - 1,
            ),
        )

    def _proposal_score(
        self,
        params: Dict[str, Any],
        start_frame: int,
        end_frame: int,
    ) -> tuple[float, Dict[str, float]]:
        """Fast search-only score with a forward/backward consistency signal.

        This deliberately remains a proposal heuristic.  Final recommendation
        is decided by the production loop on held-out frames below.
        """

        forward_score, sub_scores, forward_map = self._run_tracking_loop(
            params, start_frame=start_frame, end_frame=end_frame
        )
        _, _, backward_map = self._run_tracking_loop(
            params, reverse=True, start_frame=start_frame, end_frame=end_frame
        )
        n_tracks = int(params["MAX_TARGETS"])
        forward = _dense_frame_positions(forward_map, start_frame, end_frame, n_tracks)
        backward = _dense_frame_positions(
            backward_map, start_frame, end_frame, n_tracks
        )
        body_scale = max(
            float(params.get("REFERENCE_BODY_SIZE", 20.0))
            * float(params.get("RESIZE_FACTOR", 1.0)),
            1e-6,
        )
        try:
            cycle = forward_backward_cycle_consistency(
                forward,
                backward,
                backward_is_reverse_chronological=False,
                spatial_scale=body_scale,
            )
            cycle_loss = min(float(cycle.mean_normalized_error), 10.0)
        except ValueError:
            cycle_loss = 10.0
        sub_scores = dict(sub_scores)
        sub_scores["cycle_loss"] = cycle_loss
        # This bounded, monotonic penalty only guides Optuna.  Unlike the old
        # unbounded weighted term, a pathological cycle result cannot dominate
        # the native replay composite or exclude diverse candidates upstream of
        # held-out production validation.
        return (
            float(forward_score + _bounded_cycle_proposal_penalty(cycle_loss)),
            sub_scores,
        )

    @staticmethod
    def _validation_evaluations(
        candidate_id: str,
        forward: np.ndarray,
        backward: np.ndarray,
        body_scale: float,
        *,
        detection_counts: np.ndarray | None = None,
    ) -> list[CandidateEvaluation]:
        """Build paired region measurements using one global slot alignment.

        Temporal events are assigned to the region containing their final
        frame, so transition/triplet evidence at region boundaries remains in
        the validation data rather than disappearing with a bare array slice.
        """

        try:
            slot_alignment = global_slot_alignment(
                forward,
                backward,
                backward_is_reverse_chronological=False,
                spatial_scale=body_scale,
                minimum_shared_observations=_MIN_CYCLE_SHARED_OBSERVATIONS,
            )
        except ValueError:
            slot_alignment = None

        segment_count = min(_VALIDATION_REGION_COUNT, len(forward))
        evaluations: list[CandidateEvaluation] = []
        for index, segment in enumerate(
            partition_temporal_segments(len(forward), segment_count)
        ):
            forward_quality = trajectory_quality_metrics(
                forward, spatial_scale=body_scale, segment=segment
            )
            backward_quality = trajectory_quality_metrics(
                backward, spatial_scale=body_scale, segment=segment
            )
            forward_sanity = output_sanity_metrics(
                forward,
                spatial_scale=body_scale,
                detection_counts=detection_counts,
                segment=segment,
            )
            backward_sanity = output_sanity_metrics(
                backward,
                spatial_scale=body_scale,
                detection_counts=detection_counts,
                segment=segment,
            )
            if slot_alignment is None:
                cycle_loss = 10.0
                cycle_observation_coverage = 0.0
            else:
                try:
                    cycle = forward_backward_cycle_consistency(
                        forward[segment.start : segment.stop],
                        backward[segment.start : segment.stop],
                        backward_is_reverse_chronological=False,
                        spatial_scale=body_scale,
                        slot_alignment=slot_alignment,
                        minimum_shared_observations=_MIN_CYCLE_SHARED_OBSERVATIONS,
                    )
                    cycle_loss = min(float(cycle.mean_normalized_error), 10.0)
                    cycle_observation_coverage = float(
                        cycle.shared_observation_coverage
                    )
                except ValueError:
                    # No region-level shared evidence is strong evidence
                    # against automatic promotion, while remaining finite for
                    # deterministic ranking and diagnostics.
                    cycle_loss = 10.0
                    cycle_observation_coverage = 0.0
            evaluations.append(
                CandidateEvaluation(
                    candidate_id,
                    {
                        "cycle_loss": min(float(cycle_loss), 10.0),
                        "cycle_observation_coverage": cycle_observation_coverage,
                        "coverage_loss": max(
                            float(forward_quality.coverage_loss),
                            float(backward_quality.coverage_loss),
                        ),
                        "worst_track_coverage_loss": max(
                            float(forward_sanity.worst_track_coverage_loss),
                            float(backward_sanity.worst_track_coverage_loss),
                        ),
                        "fragmentation_loss": max(
                            float(forward_quality.fragmentation_loss),
                            float(backward_quality.fragmentation_loss),
                        ),
                        "motion_roughness_loss": min(
                            max(
                                float(forward_quality.motion_roughness_loss),
                                float(backward_quality.motion_roughness_loss),
                            ),
                            10.0,
                        ),
                        "collision_loss": max(
                            float(forward_sanity.collision_loss),
                            float(backward_sanity.collision_loss),
                        ),
                        "detection_excess_loss": max(
                            float(forward_sanity.detection_excess_loss),
                            float(backward_sanity.detection_excess_loss),
                        ),
                    },
                    perturbation_id=f"temporal-region-{index + 1}",
                )
            )
        return evaluations

    @staticmethod
    def _longest_consecutive_observed_runs(positions: np.ndarray) -> np.ndarray:
        """Return each slot's longest run of finite exported observations."""

        observed = np.isfinite(positions).all(axis=2)
        longest = np.zeros(observed.shape[1], dtype=np.int64)
        current = np.zeros(observed.shape[1], dtype=np.int64)
        for frame_observed in observed:
            current = np.where(frame_observed, current + 1, 0)
            longest = np.maximum(longest, current)
        return longest

    @staticmethod
    def _longest_bracketed_missing_runs(positions: np.ndarray) -> np.ndarray:
        """Return each slot's longest missing run with observations on both sides.

        A trailing or leading absence cannot prove that the live lifecycle
        transition and its post-loss rejoin were exercised, so it deliberately
        does not count here.
        """

        observed = np.isfinite(positions).all(axis=2)
        longest = np.zeros(observed.shape[1], dtype=np.int64)
        for track_index in range(observed.shape[1]):
            track_observed = observed[:, track_index]
            index = 0
            while index < len(track_observed):
                if track_observed[index]:
                    index += 1
                    continue
                start = index
                while index < len(track_observed) and not track_observed[index]:
                    index += 1
                if (
                    start > 0
                    and index < len(track_observed)
                    and track_observed[start - 1]
                    and track_observed[index]
                ):
                    longest[track_index] = max(longest[track_index], index - start)
        return longest

    def _changed_lifecycle_thresholds(
        self, candidate_params: Mapping[str, Any] | None
    ) -> dict[str, tuple[int, int]]:
        """Return ``(baseline, candidate)`` thresholds for changed lifecycle values.

        A candidate-specific dimension needs evidence that distinguishes the two
        policies, not necessarily evidence that reaches the more conservative
        one. The caller therefore requires exposure to the lower of these two
        values, while preserving the exact candidate/baseline values for
        diagnostics.
        """

        changed: dict[str, tuple[int, int]] = {}
        for name in _TEMPORAL_HORIZON_PARAMETERS:
            if candidate_params is None or name not in candidate_params:
                continue
            try:
                candidate_value = float(candidate_params[name])
                baseline_value = float(
                    self.base_params.get(name, self._SEED_DEFAULTS[name])
                )
            except (TypeError, ValueError):
                continue
            if np.isclose(candidate_value, baseline_value, rtol=0.0, atol=1e-9):
                continue
            baseline_threshold = max(1, int(math.ceil(baseline_value)))
            candidate_threshold = max(1, int(math.ceil(candidate_value)))
            changed[name] = (baseline_threshold, candidate_threshold)
        return changed

    def _minimum_region_cycle_observations(
        self, segment_frames: int, track_slots: int
    ) -> int:
        """Require both an absolute overlap floor and horizon-scaled support."""

        horizon_frames = min(segment_frames, self._temporal_validation_horizon())
        scaled_support = math.ceil(
            horizon_frames * max(1, track_slots) * _MIN_CYCLE_SHARED_COVERAGE
        )
        return max(_MIN_CYCLE_SHARED_OBSERVATIONS, scaled_support)

    def _validation_temporal_evidence_reason(
        self,
        forward: np.ndarray,
        backward: np.ndarray,
        body_scale: float,
        evaluations: list[CandidateEvaluation],
        *,
        candidate_params: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Return why regional production metrics lack real temporal evidence.

        Frame count establishes only the *opportunity* to observe transitions.
        Promotion also needs sufficient fully observed motion-triplet support
        in both replay directions and robust cycle overlap in every paired
        region.  The corresponding candidate evaluations stay finite for
        diagnostics, but their neutral/sentinel values must not become
        statistical evidence.
        """

        segments = partition_temporal_segments(len(forward), len(evaluations))
        # A 3-frame metric needs one triplet.  For a larger tuned lifecycle
        # horizon, require enough consecutive observations to actually cover
        # that horizon rather than accepting one isolated smooth blip.
        required_motion_triplets = max(1, self._temporal_validation_horizon() - 2)
        for index, (segment, evaluation) in enumerate(
            zip(segments, evaluations, strict=True), start=1
        ):
            for direction, positions in (
                ("forward", forward),
                ("backward", backward),
            ):
                quality = trajectory_quality_metrics(
                    positions, spatial_scale=body_scale, segment=segment
                )
                if quality.valid_motion_triplets < required_motion_triplets:
                    return (
                        "held-out temporal support is inadequate: validation "
                        f"region {index} requires at least {required_motion_triplets} "
                        "observed motion-triplet(s) but found "
                        f"{quality.valid_motion_triplets} in the {direction} replay"
                    )
            if evaluation.metrics["cycle_observation_coverage"] <= 0.0:
                return (
                    "held-out temporal support is inadequate: validation "
                    f"region {index} has no robust shared forward/backward "
                    "cycle observations"
                )

        # A positive fractional overlap alone can be two points. That produces
        # a zero standard error across four regions while providing almost no
        # evidence that a forward/backward agreement is stable. Reuse one
        # full-window mapping, then require both an absolute and a
        # horizon/slot-scaled amount of paired evidence in every region.
        try:
            slot_alignment = global_slot_alignment(
                forward,
                backward,
                backward_is_reverse_chronological=False,
                spatial_scale=body_scale,
                minimum_shared_observations=_MIN_CYCLE_SHARED_OBSERVATIONS,
            )
        except ValueError:
            return (
                "held-out temporal support is inadequate: no robust full-window "
                "forward/backward cycle alignment"
            )
        for index, segment in enumerate(segments, start=1):
            required_shared = self._minimum_region_cycle_observations(
                segment.frame_count, forward.shape[1]
            )
            try:
                cycle = forward_backward_cycle_consistency(
                    forward[segment.start : segment.stop],
                    backward[segment.start : segment.stop],
                    backward_is_reverse_chronological=False,
                    spatial_scale=body_scale,
                    slot_alignment=slot_alignment,
                    minimum_shared_observations=_MIN_CYCLE_SHARED_OBSERVATIONS,
                )
            except ValueError:
                return (
                    "held-out temporal support is inadequate: validation "
                    f"region {index} has no robust shared forward/backward "
                    "cycle observations"
                )
            if cycle.valid_observations < required_shared:
                return (
                    "held-out temporal support is inadequate: validation "
                    f"region {index} requires at least {required_shared} shared "
                    "forward/backward cycle observations but found "
                    f"{cycle.valid_observations}"
                )
            if cycle.shared_observation_coverage < _MIN_CYCLE_SHARED_COVERAGE:
                return (
                    "held-out temporal support is inadequate: validation "
                    f"region {index} requires at least "
                    f"{_MIN_CYCLE_SHARED_COVERAGE:.0%} shared forward/backward "
                    "cycle coverage"
                )

        # A candidate-specific lifecycle value is evidence-bearing only if the
        # held-out export reaches a threshold on which baseline and candidate
        # can differ. That is the lower of their two values: a run/gap between
        # them is precisely the case that exercises one policy while leaving
        # the other unchanged. This remains separate from generic motion
        # support because many disjoint three-frame fragments can produce
        # plenty of triplets without ever maturing a track, and a boundary gap
        # cannot prove a loss/rejoin transition.
        lifecycle_thresholds = self._changed_lifecycle_thresholds(candidate_params)
        maturity_thresholds = lifecycle_thresholds.get("KALMAN_MATURITY_AGE")
        if maturity_thresholds is not None:
            baseline_maturity, candidate_maturity = maturity_thresholds
            required_maturity = min(baseline_maturity, candidate_maturity)
            for direction, positions in (("forward", forward), ("backward", backward)):
                longest_run = int(
                    np.max(self._longest_consecutive_observed_runs(positions))
                )
                if longest_run < required_maturity:
                    return (
                        "held-out lifecycle support is inadequate: candidate "
                        f"KALMAN_MATURITY_AGE={candidate_maturity} differs from "
                        f"baseline {baseline_maturity} but has no "
                        f"{required_maturity}-frame consecutive observed run in "
                        f"the {direction} replay (longest {longest_run})"
                    )

        lost_thresholds = lifecycle_thresholds.get("LOST_THRESHOLD_FRAMES")
        if lost_thresholds is not None:
            baseline_lost, candidate_lost = lost_thresholds
            required_lost = min(baseline_lost, candidate_lost)
            for direction, positions in (("forward", forward), ("backward", backward)):
                longest_gap = int(
                    np.max(self._longest_bracketed_missing_runs(positions))
                )
                if longest_gap < required_lost:
                    return (
                        "held-out lifecycle support is inadequate: candidate "
                        f"LOST_THRESHOLD_FRAMES={candidate_lost} differs from "
                        f"baseline {baseline_lost} but has no bracketed "
                        f"{required_lost}-frame missing run in the {direction} "
                        f"replay (longest {longest_gap})"
                    )
        return None

    def _validation_detection_counts(
        self, params: Dict[str, Any], start_frame: int, end_frame: int
    ) -> np.ndarray | None:
        """Return candidate-filtered source counts as a false-positive risk signal.

        These counts are not labelled false positives.  They are only a
        conservative guard against promoting a candidate that admits a large
        excess of source detections.  A diagnostic failure remains neutral so
        it cannot fabricate a source-level claim when the cache is unavailable
        in a test or an injected replay implementation.
        """

        if self.cache is None:
            return None

        class _ParamsFilter:
            def __init__(self, values: Dict[str, Any]) -> None:
                self.params = values
                self.inference_config = inference_config_for_optimizer_params(values)

        try:
            detector = _ParamsFilter(params)
            roi_mask = self._frame_space_roi_mask_for_params(params)
            return np.asarray(
                [
                    len(
                        _filter_cached_detections(
                            detector,
                            self.cache,
                            frame,
                            roi_mask,
                            apply_max_detections=False,
                        )[0]
                    )
                    for frame in range(start_frame, end_frame + 1)
                ],
                dtype=np.int64,
            )
        except Exception:
            logger.warning(
                "Optimizer: unable to calculate source detection-excess safeguard",
                exc_info=True,
            )
            return None

    def _candidate_change_cost(self, result: OptimizationResult) -> float:
        """Normalized distance from current settings for deterministic tie breaks."""

        distances: list[float] = []
        for name, value in result.params.items():
            if name not in self._PARAM_RANGES:
                continue
            ptype, low, high = self._search_range(name)
            baseline = self.base_params.get(name, self._SEED_DEFAULTS.get(name, low))
            try:
                if ptype == "log_float":
                    if float(value) <= 0 or float(baseline) <= 0:
                        continue
                    distance = abs(
                        np.log(float(value)) - np.log(float(baseline))
                    ) / max(np.log(float(high)) - np.log(float(low)), 1e-12)
                else:
                    distance = abs(float(value) - float(baseline)) / max(
                        float(high) - float(low), 1e-12
                    )
            except (TypeError, ValueError):
                continue
            distances.append(float(np.clip(distance, 0.0, 1.0)))
        return float(np.mean(distances)) if distances else 0.0

    def _select_validation_shortlist(
        self, candidates: list[OptimizationResult], *, limit: int = 5
    ) -> list[OptimizationResult]:
        """Select a normalized, diverse proposal set for expensive replay.

        Proposal composite scores mix quantities with unlike scales, especially
        cycle loss.  Replay selection therefore normalizes each proposal signal,
        includes per-metric extremes, then fills remaining slots by metric and
        parameter-space diversity.  This is a search allocation heuristic, not
        an accuracy oracle.
        """

        if limit < 1:
            raise ValueError("shortlist limit must be positive")
        if len(candidates) <= limit:
            return sorted(candidates, key=lambda item: item.candidate_id)

        values = np.asarray(
            [
                [
                    float(item.sub_scores.get(name, item.score))
                    for name in _PROPOSAL_METRICS
                ]
                for item in candidates
            ],
            dtype=float,
        )
        values[~np.isfinite(values)] = np.nan
        normalized = np.zeros_like(values)
        for column in range(values.shape[1]):
            column_values = values[:, column]
            finite = column_values[np.isfinite(column_values)]
            if not finite.size:
                normalized[:, column] = 1.0
                continue
            low = float(np.min(finite))
            high = float(np.max(finite))
            if np.isclose(low, high):
                normalized[:, column] = 0.0
            else:
                normalized[:, column] = np.where(
                    np.isfinite(column_values),
                    (column_values - low) / (high - low),
                    1.0,
                )
        mean_normalized = np.mean(normalized, axis=1)

        def _best_index(indices: list[int]) -> int:
            return min(
                indices,
                key=lambda index: (
                    float(mean_normalized[index]),
                    float(candidates[index].score),
                    candidates[index].candidate_id,
                ),
            )

        selected: list[int] = [_best_index(list(range(len(candidates))))]
        # Preserve each diagnostic's best observed proposal instead of letting
        # the historical unnormalized scalar exclude it before replay.
        for column in range(normalized.shape[1]):
            if len(selected) >= limit:
                break
            best_value = float(np.min(normalized[:, column]))
            contenders = [
                index
                for index in range(len(candidates))
                if np.isclose(normalized[index, column], best_value)
            ]
            choice = _best_index(contenders)
            if choice not in selected:
                selected.append(choice)

        while len(selected) < limit:
            remaining = [
                index for index in range(len(candidates)) if index not in selected
            ]
            if not remaining:
                break

            def _diversity_key(index: int) -> tuple[float, float, float, str]:
                metric_distance = min(
                    float(np.linalg.norm(normalized[index] - normalized[chosen]))
                    for chosen in selected
                )
                parameter_distance = min(
                    self._proposal_parameter_distance(
                        candidates[index], candidates[chosen]
                    )
                    for chosen in selected
                )
                return (
                    metric_distance + 0.25 * parameter_distance,
                    -float(mean_normalized[index]),
                    -float(candidates[index].score),
                    candidates[index].candidate_id,
                )

            # ``max`` keeps larger diversity first; reverse the remaining
            # quality fields so lower normalized loss wins exact ties.
            selected.append(max(remaining, key=_diversity_key))
        return [candidates[index] for index in selected]

    def _proposal_parameter_distance(
        self, left: OptimizationResult, right: OptimizationResult
    ) -> float:
        distances: list[float] = []
        for name in sorted(set(left.params) | set(right.params)):
            if name not in self._PARAM_RANGES:
                continue
            ptype, low, high = self._search_range(name)
            left_value = left.params.get(
                name, self.base_params.get(name, self._SEED_DEFAULTS.get(name, low))
            )
            right_value = right.params.get(
                name, self.base_params.get(name, self._SEED_DEFAULTS.get(name, low))
            )
            try:
                if ptype == "log_float":
                    if float(left_value) <= 0 or float(right_value) <= 0:
                        continue
                    distance = abs(
                        np.log(float(left_value)) - np.log(float(right_value))
                    ) / max(np.log(float(high)) - np.log(float(low)), 1e-12)
                else:
                    distance = abs(float(left_value) - float(right_value)) / max(
                        float(high) - float(low), 1e-12
                    )
            except (TypeError, ValueError):
                continue
            distances.append(float(np.clip(distance, 0.0, 1.0)))
        return float(np.mean(distances)) if distances else 0.0

    def _production_validate_shortlist(
        self,
        results: list[OptimizationResult],
        validation_bounds: tuple[int, int] | None,
    ) -> None:
        """Annotate results and choose a conservatively safe recommendation."""

        baseline = next(result for result in results if result.is_baseline)
        baseline.recommended = True
        baseline.recommendation_reason = "current settings retained by default"
        if validation_bounds is None:
            baseline.recommendation_reason = (
                "current settings retained: frame range is too short for "
                "adequate held-out temporal validation"
            )
            return
        if self._stop_requested:
            baseline.recommendation_reason = (
                "current settings retained: held-out validation was cancelled"
            )
            return
        support_reason = self._validation_support_reason(validation_bounds)
        if support_reason is not None:
            baseline.recommendation_reason = (
                "current settings retained: " + support_reason
            )
            return

        # Validate only a bounded shortlist through the much heavier production
        # loop.  The baseline is always included exactly as configured.
        candidates = [result for result in results if not result.is_baseline]
        shortlist = [baseline, *self._select_validation_shortlist(candidates)]
        start_frame, end_frame = validation_bounds
        evaluator = ProductionReplayEvaluator(
            self.video_path,
            self.detection_cache_path,
            start_frame,
            end_frame,
            pre_roll_start=self.start_frame,
            cache_provenance_params=self.base_params,
            should_stop=lambda: self._stop_requested,
        )
        all_evaluations: list[CandidateEvaluation] = []
        validated_results: list[OptimizationResult] = []
        for index, result in enumerate(shortlist):
            if self._stop_requested:
                break
            if self._progress_cb is not None:
                self._progress_cb(
                    85 + int(15 * index / max(len(shortlist), 1)),
                    f"Production-validating candidate {index + 1}/{len(shortlist)}",
                )
            params = dict(self.base_params)
            params.update(result.params)
            detection_counts = self._validation_detection_counts(
                params, start_frame, end_frame
            )
            forward = evaluator.run(params, reverse=False)
            if self._stop_requested:
                break
            if not forward.success:
                result.recommendation_reason = "production validation failed" + (
                    f": {forward.error}" if forward.error else ""
                )
                continue
            backward = evaluator.run(params, reverse=True)
            if not backward.success:
                messages = [value for value in (forward.error, backward.error) if value]
                result.recommendation_reason = "production validation failed" + (
                    f": {'; '.join(messages)}" if messages else ""
                )
                continue
            body_scale = max(
                float(params.get("REFERENCE_BODY_SIZE", 20.0))
                * float(params.get("RESIZE_FACTOR", 1.0)),
                1e-6,
            )
            evaluations = self._validation_evaluations(
                result.candidate_id,
                forward.positions,
                backward.positions,
                body_scale,
                detection_counts=detection_counts,
            )
            temporal_evidence_reason = self._validation_temporal_evidence_reason(
                forward.positions,
                backward.positions,
                body_scale,
                evaluations,
                candidate_params=result.params,
            )
            if temporal_evidence_reason is not None:
                result.recommendation_reason = "production validation skipped: " + (
                    temporal_evidence_reason
                )
                if result.is_baseline:
                    baseline.recommendation_reason = "current settings retained: " + (
                        temporal_evidence_reason
                    )
                    return
                continue
            all_evaluations.extend(evaluations)
            validated_results.append(result)

        if self._progress_cb is not None:
            self._progress_cb(100, "Held-out production validation complete")

        if self._stop_requested:
            baseline.recommendation_reason = (
                "current settings retained: held-out validation was cancelled"
            )
            return
        if baseline not in validated_results:
            baseline.recommendation_reason = (
                "current settings retained: baseline production validation failed"
            )
            return

        scores = aggregate_candidate_evaluations(
            all_evaluations, _UNLABELED_METRIC_SPECS
        )
        score_by_id = {score.candidate_id: score for score in scores}
        result_by_id = {result.candidate_id: result for result in validated_results}
        for rank in pareto_ranks(scores, _UNLABELED_METRIC_SPECS):
            result = result_by_id[rank.candidate_id]
            result.pareto_rank = rank.front
            score = score_by_id[rank.candidate_id]
            result.mean_relative_spread = score.mean_relative_spread
            result.validation_metrics = {
                name: {
                    "mean": estimate.mean,
                    "std": estimate.std,
                    "sample_count": estimate.sample_count,
                }
                for name, estimate in score.metrics.items()
            }
            result.sub_scores.update(
                {name: estimate.mean for name, estimate in score.metrics.items()}
            )

        recommendation = recommend_dominating_candidate(
            scores,
            baseline.candidate_id,
            _UNLABELED_METRIC_SPECS,
            candidate_change_costs={
                result.candidate_id: self._candidate_change_cost(result)
                for result in validated_results
                if not result.is_baseline
            },
        )
        if recommendation.candidate_id is None:
            baseline.recommendation_reason = (
                "current settings retained: " + recommendation.reason
            )
            return

        candidate = result_by_id[recommendation.candidate_id]
        decision = decide_heldout_baseline_protection(
            score_by_id[baseline.candidate_id],
            score_by_id[candidate.candidate_id],
            _UNLABELED_METRIC_SPECS,
            primary_metric="cycle_loss",
            minimum_improvement=float(
                self.base_params.get("AUTOTUNE_MIN_HELDOUT_IMPROVEMENT", 0.01)
            ),
            selection_family_size=max(1, len(validated_results) - 1),
        )
        if not decision.accepted:
            baseline.recommendation_reason = (
                "current settings retained: " + decision.reason
            )
            return
        baseline.recommended = False
        candidate.recommended = True
        candidate.recommendation_reason = decision.reason

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(self):
        if not self._open_and_validate_cache():
            # Explicitly push an empty result set so a stale result list from a
            # previous successful run in the same dialog session isn't left
            # showing as if this run had produced (or reused) results.
            if self._result_cb is not None:
                self._result_cb([])
            return []

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_active = sum(1 for v in self.tuning_config.values() if v)

        sampler = self._build_sampler(n_active)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        results: list[OptimizationResult] = []

        # Pre-calculated scaled body size for pixel conversions
        ref_size = self.base_params.get("REFERENCE_BODY_SIZE", 20.0)
        resize_f = self.base_params.get("RESIZE_FACTOR", 1.0)
        scaled_body_size = ref_size * resize_f

        self._preload_pose_data()

        search_bounds, validation_bounds = self._search_and_validation_bounds()
        search_start, search_end = search_bounds

        # The current production settings are a first-class candidate and are
        # evaluated exactly, even when a value sits outside the exploration
        # range.  They are never replaced by a clamped approximation.
        try:
            baseline_score, baseline_sub_scores = self._proposal_score(
                dict(self.base_params), search_start, search_end
            )
        except Exception as exc:
            logger.exception("Optimizer: exact baseline evaluation failed")
            if self._error_cb is not None:
                self._error_cb(f"Baseline evaluation failed: {exc}")
            self.cache = None
            self._pose_frame_cache = None
            if self._result_cb is not None:
                self._result_cb([])
            return []
        results.append(
            OptimizationResult(
                {},
                baseline_score,
                -1,
                baseline_sub_scores,
                candidate_id="baseline",
                is_baseline=True,
            )
        )

        # Seed search near current settings when every selected value is inside
        # the exploration contract.  Exact baseline evidence is already stored
        # above, so a partial/out-of-range seed is not misrepresented as it.
        seed_params = self._build_seed_trial()
        if seed_params:
            study.enqueue_trial(seed_params)

        # Additional diversity seeds: random points spread across the full search space.
        _rng_seeds = np.random.default_rng(42)

        # Diversity seeds fan out from base_params: tight for early seeds, wider for
        # later ones.  sigma grows linearly from 10 % to 30 % of each parameter's range.
        n_extra = self.n_seeds - 1
        for i in range(n_extra):
            scale = 0.10 + 0.20 * (i / max(1, n_extra - 1)) if n_extra > 1 else 0.15
            extra_seed = self._perturb_near_base(_rng_seeds, scale)
            if extra_seed:
                study.enqueue_trial(extra_seed)

        # A separate unseeded RNG for plateau restarts so each restart is genuinely
        # different (not deterministically replaying the diversity seeds).
        _rng_restart = np.random.default_rng()

        # Plateau detection: patience counted in consecutive non-improving trials.
        _PLATEAU_PATIENCE = max(15, self.n_trials // 5)
        _no_improve_count = 0
        _best_score_seen = float("inf")

        def objective(trial):
            nonlocal _no_improve_count, _best_score_seen
            if self._stop_requested:
                study.stop()
                raise optuna.TrialPruned()

            trial_params = self._suggest_trial_params(trial, scaled_body_size)

            # Merge into full param set
            current_params = self.base_params.copy()
            current_params.update(trial_params)

            # Fast replay only proposes candidates on the training slice.
            # Production replay on held-out frames decides whether one is safe
            # to recommend after the study.
            score, sub_scores = self._proposal_score(
                current_params, search_start, search_end
            )

            results.append(
                OptimizationResult(trial_params, score, trial.number, sub_scores)
            )
            pct = int(((trial.number + 1) / max(self.n_trials, 1)) * 85)
            if self._progress_cb is not None:
                self._progress_cb(
                    int(pct),
                    f"Proposal {trial.number + 1}/{self.n_trials} "
                    f"(search loss: {score:.3f})",
                )

            # Plateau detection
            if score < _best_score_seen:
                _best_score_seen = score
                _no_improve_count = 0
            else:
                _no_improve_count += 1
                if _no_improve_count >= _PLATEAU_PATIENCE:
                    if self.on_plateau == "restart":
                        restart_pt = self._random_from_ranges(_rng_restart)
                        if restart_pt:
                            study.enqueue_trial(restart_pt)
                        _no_improve_count = 0
                    else:
                        # Convergence is a normal search terminal condition;
                        # it must not masquerade as an explicit user cancel
                        # and skip the held-out production validation below.
                        self._search_converged = True
                        study.stop()

            return score

        try:
            study.optimize(objective, n_trials=self.n_trials)
        except Exception as e:
            # Optuna's default catch=() means ANY exception raised inside a
            # trial (e.g. the very first one, before any result is appended)
            # aborts the whole study immediately. Surface this loudly instead
            # of just logging it -- otherwise the GUI reports a plain "no
            # results" with no indication anything actually went wrong.
            logger.exception("Optimization trial failed")
            if self._error_cb is not None:
                self._error_cb(f"Optimization trial failed: {e}")

        self._production_validate_shortlist(results, validation_bounds)
        # Put the conservative recommendation first.  Remaining rows are a
        # transparent shortlist: production-validated Pareto fronts first,
        # then the search heuristic used only to generate proposals.
        results.sort(
            key=lambda item: (
                not item.recommended,
                item.pareto_rank is None,
                item.pareto_rank if item.pareto_rank is not None else 10**9,
                item.score,
            )
        )
        # Do not call self.cache.close(): it was opened read-only and
        # DetectionCacheHandle.close() flushes its (empty) write buffer,
        # which would clobber the on-disk cache with zero frames.
        self.cache = None
        self._pose_frame_cache = None  # free memory after optimization
        if self._result_cb is not None:
            self._result_cb(results)
        return results

    def _run_tracking_loop(
        self,
        params: Dict[str, Any],
        reverse: bool = False,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ):
        """
        Core tracking simulation shared by quality scoring and the FB consistency metric.

        Returns
        -------
        composite : float
            Lower-is-better composite score balancing four orthogonal objectives.
        frame_positions : Dict[int, np.ndarray]
            Maps frame_idx -> (N, 2) float32 array of current detection
            observations. Missing/unmatched rows contain NaN.
        """

        # Correction 21: create a lightweight params holder that _filter_cached_detections
        # reads for confidence threshold when using new OBBResult cache.
        class _ParamsFilter:
            def __init__(self, p):
                self.params = p
                self.inference_config = inference_config_for_optimizer_params(p)

        start_frame = self.start_frame if start_frame is None else int(start_frame)
        end_frame = self.end_frame if end_frame is None else int(end_frame)
        if end_frame < start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")

        det_filter = _ParamsFilter(params)
        kf_manager = KalmanFilterManager(params["MAX_TARGETS"], params)
        assigner = TrackAssigner(params)
        # Arena gating must be identical to the live run's (worker.py): tuning
        # against an UNGATED simulation hands the real, gated run parameters
        # chosen under cross-arena assignment it will never be allowed to make.
        # `None` for single-arena takes the assigner's ungated path
        # STRUCTURALLY, so single-arena optimizer runs are unchanged.
        _arena_layout = arena_layout_from_params(params)
        check_slot_arena_covers_all_slots(_arena_layout, params["MAX_TARGETS"])
        assigner.set_track_arena(
            None if _arena_layout.is_single_arena else _arena_layout.slot_arena
        )
        _arena_frame_size = None
        if not _arena_layout.is_single_arena:
            _arena_frame_size = _optimizer_frame_size(self.video_path, params)
        _roi_mask = self._frame_space_roi_mask_for_params(params)

        (
            _pose_anterior,
            _pose_posterior,
            _pose_ignore,
            _pose_enabled,
            _pose_frame_data,
        ) = _load_pose_context_for_loop(self, params)
        _pose_min_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))

        N = params["MAX_TARGETS"]
        track_pose_prototypes: list = [None] * N

        # ── Composite objective accumulators ──────────────────────────────────
        _coverage_sum = 0.0
        _occlusion_sum = 0.0
        _assign_cost_sum = 0.0
        _assign_count = 0
        _suspicious_assign_count = 0
        _det_count_sum = 0
        _max_continuity = [0] * N
        track_states = ["lost"] * N
        tracking_continuity = [0] * N

        _body_size = max(
            params.get("REFERENCE_BODY_SIZE", 20.0) * params.get("RESIZE_FACTOR", 1.0),
            5.0,
        )
        _n_pairs = max(N * (N - 1) // 2, 1)

        _w_cov, _w_asn, _w_frg, _w_occ, _w_vel, _w_crd = _normalise_scoring_weights(
            params
        )

        _step_norms: list = []
        _direction_reversals: list = []
        _crowding_sum = 0.0
        _crowding_frames = 0
        _prev_positions: Dict[int, np.ndarray] = {}
        _prev_vecs: Dict[int, np.ndarray] = {}
        missed_frames = [0] * N
        trajectory_ids = list(range(N))
        next_trajectory_id = N
        last_shape_info = [None] * N
        orientation_last: list = [None] * N
        track_avg_step = np.zeros(N, dtype=np.float32)
        lost_threshold = params.get("LOST_THRESHOLD_FRAMES", 5)
        n_frames = end_frame - start_frame + 1
        detection_initialized = False
        detection_counts = 0

        frame_order = (
            range(end_frame, start_frame - 1, -1)
            if reverse
            else range(start_frame, end_frame + 1)
        )
        frame_positions: Dict[int, np.ndarray] = {}

        for f_idx in frame_order:
            if self._stop_requested:
                break

            meas, shapes, _confs, detection_ids, _headtail_hints, _headtail_directed = (
                _filter_cached_detections(det_filter, self.cache, f_idx, _roi_mask)
            )

            _det_pose_kpts, _det_pose_vis, _det_pose_headings = (
                _compute_pose_features_for_frame(
                    meas,
                    detection_ids,
                    f_idx,
                    _pose_enabled,
                    _pose_frame_data,
                    _pose_anterior,
                    _pose_posterior,
                    _pose_ignore,
                    _pose_min_conf,
                )
            )
            _pose_direction_min_visibility = float(
                np.clip(
                    params.get(
                        "POSE_DIRECTION_MIN_VISIBILITY",
                        max(0.6, params.get("POSE_REJECTION_MIN_VISIBILITY", 0.5)),
                    ),
                    0.0,
                    1.0,
                )
            )
            _pose_direction_min_keypoints = max(
                1, int(params.get("POSE_DIRECTION_MIN_VALID_KEYPOINTS", 3))
            )
            _pose_heading_mask = [
                (
                    1
                    if _pf_heading_reliable(
                        _det_pose_kpts[idx] if idx < len(_det_pose_kpts) else None,
                        float(_det_pose_vis[idx]) if idx < len(_det_pose_vis) else 0.0,
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
                "track_avg_step": track_avg_step.copy(),
            }
            current_observations: Dict[int, np.ndarray] = {}
            if len(meas) >= int(params.get("MIN_DETECTIONS_TO_START", 1)):
                detection_counts += 1
            else:
                detection_counts = 0
            if (
                detection_counts
                >= max(1, int(params.get("MIN_DETECTION_COUNTS", 1)) // 2)
                and not detection_initialized
            ):
                detection_initialized = True

            if detection_initialized and meas:
                kf_manager.predict()

                _meas_arena = _meas_arena_ids(_arena_layout, meas, _arena_frame_size)
                cost, _spatial_candidates = assigner.compute_cost_matrix(
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
                    meas_arena=_meas_arena,
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
                        spatial_candidates=_spatial_candidates,
                        association_data=_association_data,
                        missed_frames=missed_frames,
                        meas_arena=_meas_arena,
                    )
                )
                respawned_matches = {r for r in matched_r if track_states[r] == "lost"}
                _pre_correct: Dict[int, np.ndarray] = {
                    r: kf_manager.X[r, :2].copy() for r in matched_r
                }

                # Correct matched tracks and update orientation/speed EMA.
                _feature_alpha = float(params.get("TRACK_FEATURE_EMA_ALPHA", 0.85))
                _high_conf_thresh = float(
                    params.get("ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD", 0.7)
                )
                self._correct_matched_tracks(
                    matched_r,
                    matched_c,
                    meas,
                    _confs,
                    detection_directed_mask,
                    detection_directed_heading,
                    respawned_matches,
                    kf_manager,
                    orientation_last,
                    _prev_positions,
                    _prev_vecs,
                    track_avg_step,
                    _feature_alpha,
                    _high_conf_thresh,
                )

                # Innovation cost accumulation.
                _ac_sum, _ac_cnt, _ac_sus = self._accumulate_innovation_costs(
                    matched_r,
                    matched_c,
                    meas,
                    track_states,
                    tracking_continuity,
                    _pre_correct,
                    _body_size,
                    params,
                )
                _assign_cost_sum += _ac_sum
                _assign_count += _ac_cnt
                _suspicious_assign_count += _ac_sus

                # Keep track pose prototypes current.
                for r, c in zip(matched_r, matched_c):
                    proto = _det_pose_kpts[c] if c < len(_det_pose_kpts) else None
                    if proto is not None:
                        track_pose_prototypes[r] = np.asarray(
                            proto, dtype=np.float32
                        ).copy()

                matched_r_set = set(matched_r)
                for r in matched_r:
                    missed_frames[r] = 0
                    track_states[r] = "active"
                    tracking_continuity[r] += 1
                _update_unmatched_track_states(
                    N,
                    matched_r_set,
                    track_states,
                    missed_frames,
                    tracking_continuity,
                    lost_threshold,
                )
                for r, c in zip(matched_r, matched_c):
                    last_shape_info[r] = shapes[c]
                    current_observations[r] = np.asarray(meas[c][:2], dtype=np.float32)

                next_trajectory_id = _respawn_free_detections(
                    free_dets,
                    N,
                    meas,
                    shapes,
                    track_states,
                    missed_frames,
                    tracking_continuity,
                    trajectory_ids,
                    next_trajectory_id,
                    orientation_last,
                    last_shape_info,
                    track_pose_prototypes,
                    track_avg_step,
                    kf_manager,
                    detection_directed_mask,
                    detection_directed_heading,
                    _det_pose_kpts,
                )
            elif detection_initialized:
                kf_manager.predict()
                _update_unmatched_track_states(
                    N,
                    set(),
                    track_states,
                    missed_frames,
                    tracking_continuity,
                    lost_threshold,
                )

            # Per-frame coverage accounting
            for r in range(N):
                if (
                    track_states[r] == "active"
                    and tracking_continuity[r] > _max_continuity[r]
                ):
                    _max_continuity[r] = tracking_continuity[r]
            _coverage_sum += sum(1 for r in range(N) if track_states[r] == "active") / N
            _occlusion_sum += (
                sum(1 for r in range(N) if track_states[r] == "occluded") / N
            )
            _det_count_sum += len(meas)

            _accumulate_velocity_metrics(
                N,
                track_states,
                kf_manager,
                _prev_positions,
                _prev_vecs,
                _step_norms,
                _direction_reversals,
                _body_size,
            )

            _crowding_frames += 1
            _crowding_sum += _accumulate_crowding_metric(
                N,
                track_states,
                kf_manager,
                _body_size,
                _n_pairs,
            )

            # Record only current-frame observations.  Production output uses
            # NaN for coasted Kalman states, so hidden posterior positions must
            # not make a candidate look artificially smooth.
            pos = np.full((N, 2), np.nan, dtype=np.float32)
            for r, observed in current_observations.items():
                pos[r] = observed
            frame_positions[f_idx] = pos

        composite, sub_scores = _compute_composite_score(
            n_frames,
            N,
            _coverage_sum,
            _occlusion_sum,
            _assign_cost_sum,
            _assign_count,
            _suspicious_assign_count,
            _det_count_sum,
            _max_continuity,
            _step_norms,
            _direction_reversals,
            _crowding_sum,
            _crowding_frames,
            _w_cov,
            _w_asn,
            _w_frg,
            _w_occ,
            _w_vel,
            _w_crd,
        )
        return composite, sub_scores, frame_positions

    def _correct_matched_tracks(
        self,
        matched_r,
        matched_c,
        meas,
        _confs,
        detection_directed_mask,
        detection_directed_heading,
        respawned_matches,
        kf_manager,
        orientation_last,
        _prev_positions,
        _prev_vecs,
        track_avg_step,
        _feature_alpha,
        _high_conf_thresh,
    ):
        """Correct KF state for matched tracks and update orientation/speed EMA."""
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
                fallback_theta=(
                    float(kf_manager.X[r, 2]) if r < len(kf_manager.X) else None
                ),
            )
            m_cor = np.array([m[0], m[1], theta_cor], dtype=np.float32)
            if r in respawned_matches:
                _prev_positions.pop(r, None)
                _prev_vecs.pop(r, None)
                track_avg_step[r] = 0.0
                kf_manager.initialize_filter(
                    r,
                    np.array(
                        [m_cor[0], m_cor[1], theta_cor, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                )
            kf_manager.correct(r, m_cor)
            curr = kf_manager.X[r, :2].copy()
            orientation_last[r] = _pf_normalize_theta(float(kf_manager.X[r, 2]))
            _det_conf = float(_confs[c]) if _confs and c < len(_confs) else 0.0
            if r in _prev_positions and _det_conf >= _high_conf_thresh:
                _step = float(np.linalg.norm(curr - _prev_positions[r]))
                track_avg_step[r] = (
                    _feature_alpha * float(track_avg_step[r])
                    + (1.0 - _feature_alpha) * _step
                )

    def _accumulate_innovation_costs(
        self,
        matched_r,
        matched_c,
        meas,
        track_states,
        tracking_continuity,
        _pre_correct,
        _body_size,
        params,
    ):
        """Return (cost_sum, count, suspicious_count) for matched tracks."""
        cost_sum = 0.0
        count = 0
        suspicious = 0
        maturity_age = params.get("KALMAN_MATURITY_AGE", 5)
        for r, c in zip(matched_r, matched_c):
            if track_states[r] == "lost":
                continue
            pixel_dist = float(
                np.linalg.norm(
                    np.asarray(meas[c][:2], dtype=np.float32) - _pre_correct[r]
                )
            )
            cost_sum += min(pixel_dist / max(2.0 * _body_size, 1e-6), 1.0)
            if pixel_dist > 2.0 * _body_size and tracking_continuity[r] >= maturity_age:
                suspicious += 1
            count += 1
        return cost_sum, count, suspicious
