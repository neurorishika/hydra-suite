"""
Tracking Parameter Optimizer and Previewer.
Enhanced with Dynamic Bayesian Optimization.
"""

import logging
import math
from typing import Any, Dict

import numpy as np
import optuna
from PySide6.QtCore import QThread, Signal

from hydra_suite.core.assigners.hungarian import TrackAssigner
from hydra_suite.core.detectors import DetectionFilter
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
from hydra_suite.data.detection_cache import DetectionCache

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


def _filter_cached_detections(det_filter, cache, f_idx, roi_mask):
    """Read a frame from detection cache and apply filtering."""
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
    ) = cache.get_frame(f_idx)
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
    _det_excess = min(max(_mean_dets / max(N, 1.0) - 1.2, 0.0) / max(N * 0.5, 1.0), 1.0)
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
    active_mask = weights > 1e-9
    if active_mask.sum() > 1:
        composite += 0.3 * float(np.std(sub_scores_arr[active_mask]))

    return composite, sub_scores


# Canonical search-space bounds — module-level so they can be imported externally.
_PARAM_RANGES: Dict[str, tuple] = {
    "YOLO_CONFIDENCE_THRESHOLD": ("float", 0.05, 0.8),
    "YOLO_IOU_THRESHOLD": ("float", 0.1, 0.9),
    "MAX_DISTANCE_MULTIPLIER": ("float", 0.5, 3.0),
    "KALMAN_NOISE_COVARIANCE": ("log_float", 0.001, 0.2),
    "KALMAN_MEASUREMENT_NOISE_COVARIANCE": ("log_float", 0.01, 1.0),
    "W_POSITION": ("float", 0.1, 5.0),
    "W_ORIENTATION": ("float", 0.0, 5.0),
    "W_AREA": ("float", 0.0, 2.0),
    "W_ASPECT": ("float", 0.0, 2.0),
    "KALMAN_DAMPING": ("float", 0.70, 0.999),
    "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": ("float", 1.0, 20.0),
    "KALMAN_INITIAL_VELOCITY_RETENTION": ("float", 0.0, 1.0),
    "KALMAN_YOUNG_GATE_MULTIPLIER": ("float", 1.0, 4.0),
    "LOST_THRESHOLD_FRAMES": ("int", 2, 25),
    "KALMAN_MATURITY_AGE": ("int", 1, 20),
}


class OptimizationResult:
    def __init__(
        self,
        params: Dict[str, Any],
        score: float,
        trial_number: int,
        sub_scores: Dict[str, float] | None = None,
    ):
        self.params = params
        self.score = score
        self.trial_number = trial_number
        self.sub_scores: Dict[str, float] = sub_scores or {}


class TrackingOptimizer(QThread):
    """
    Runs Bayesian optimization on a video slice with different parameter sets.
    Requires a pre-populated DetectionCache for speed.
    """

    progress_signal = Signal(int, str)
    result_signal = Signal(list)  # List of OptimizationResult
    finished_signal = Signal()

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
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.base_params = base_params
        self.tuning_config = tuning_config
        self.n_trials = n_trials
        self.n_seeds = max(1, n_seeds)
        self.on_plateau = on_plateau  # "restart" | "stop"
        self.sampler_type = sampler_type  # "auto" | "gp" | "tpe"
        self._stop_requested = False
        self.cache = None

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

    def stop(self):
        self._stop_requested = True

    # ------------------------------------------------------------------
    # run() helpers
    # ------------------------------------------------------------------

    def _open_and_validate_cache(self) -> bool:
        """Open the detection cache and validate compatibility.

        Returns True if the cache is ready for use, False on failure
        (after emitting appropriate error signals).
        """
        try:
            self.cache = DetectionCache(self.detection_cache_path, mode="r")
            if not self.cache.is_compatible():
                self.progress_signal.emit(0, "Error: Incompatible detection cache.")
                self.cache.close()
                return False
        except Exception as e:
            self.progress_signal.emit(0, f"Error loading cache: {str(e)}")
            return False

        if not self.cache.covers_frame_range(self.start_frame, self.end_frame):
            missing = self.cache.get_missing_frames(self.start_frame, self.end_frame)
            msg = (
                f"Error: Cache does not cover frames "
                f"{self.start_frame}-{self.end_frame}. Missing: {missing}"
            )
            self.progress_signal.emit(0, msg)
            self.cache.close()
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
        "KALMAN_YOUNG_GATE_MULTIPLIER": 1.5,
        "LOST_THRESHOLD_FRAMES": 10,
        "KALMAN_MATURITY_AGE": 5,
    }

    def _build_seed_trial(self) -> Dict[str, Any]:
        """Build the initial seed trial from the user's current parameters.

        Values are clamped to each parameter's valid range so that base_params
        with zero or out-of-range entries never produce an invalid enqueued
        trial -- especially critical for log-scale params where the value must
        be strictly positive.
        """
        seed_params: Dict[str, Any] = {}
        for key, (ptype, low, high) in self._PARAM_RANGES.items():
            if not self.tuning_config.get(key):
                continue
            raw = self.base_params.get(key, self._SEED_DEFAULTS.get(key, low))
            if ptype == "int":
                seed_params[key] = int(np.clip(int(raw), low, high))
            else:
                seed_params[key] = float(np.clip(float(raw), low, high))
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
        for key, (ptype, low, high) in self._PARAM_RANGES.items():
            if not self.tuning_config.get(key):
                continue
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
        return pt

    def _random_from_ranges(self, rng) -> dict:
        """Uniform-random point across the full search space (used for plateau restarts)."""
        pt: Dict[str, Any] = {}
        for key, (ptype, low, high) in self._PARAM_RANGES.items():
            if not self.tuning_config.get(key):
                continue
            if ptype == "log_float":
                pt[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            elif ptype == "float":
                pt[key] = float(rng.uniform(low, high))
            else:  # int
                pt[key] = int(rng.integers(low, high + 1))
        return pt

    def _suggest_trial_params(self, trial, scaled_body_size: float) -> Dict[str, Any]:
        """Use the Optuna trial to suggest values for all enabled parameters."""
        trial_params: Dict[str, Any] = {}

        # Map of param_name -> (suggest_method, args, kwargs)
        _SUGGEST_SPEC = {
            "YOLO_CONFIDENCE_THRESHOLD": ("suggest_float", (0.05, 0.8), {}),
            "YOLO_IOU_THRESHOLD": ("suggest_float", (0.1, 0.9), {}),
            "MAX_DISTANCE_MULTIPLIER": ("suggest_float", (0.5, 3.0), {}),
            "KALMAN_NOISE_COVARIANCE": ("suggest_float", (0.001, 0.2), {"log": True}),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": (
                "suggest_float",
                (0.01, 1.0),
                {"log": True},
            ),
            "W_POSITION": ("suggest_float", (0.1, 5.0), {}),
            "W_ORIENTATION": ("suggest_float", (0.0, 5.0), {}),
            "W_AREA": ("suggest_float", (0.0, 2.0), {}),
            "W_ASPECT": ("suggest_float", (0.0, 2.0), {}),
            "KALMAN_DAMPING": ("suggest_float", (0.70, 0.999), {}),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": ("suggest_float", (1.0, 20.0), {}),
            "KALMAN_INITIAL_VELOCITY_RETENTION": ("suggest_float", (0.0, 1.0), {}),
            "KALMAN_YOUNG_GATE_MULTIPLIER": ("suggest_float", (1.0, 4.0), {}),
            "LOST_THRESHOLD_FRAMES": ("suggest_int", (2, 25), {}),
            "KALMAN_MATURITY_AGE": ("suggest_int", (1, 20), {}),
        }

        for name, (method, args, kwargs) in _SUGGEST_SPEC.items():
            if self.tuning_config.get(name):
                trial_params[name] = getattr(trial, method)(name, *args, **kwargs)

        # Derived parameter: MAX_DISTANCE_THRESHOLD from multiplier
        if "MAX_DISTANCE_MULTIPLIER" in trial_params:
            trial_params["MAX_DISTANCE_THRESHOLD"] = (
                trial_params["MAX_DISTANCE_MULTIPLIER"] * scaled_body_size
            )

        return trial_params

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        if not self._open_and_validate_cache():
            return

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_active = sum(1 for v in self.tuning_config.values() if v)

        sampler = self._build_sampler(n_active)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        results = []

        # Pre-calculated scaled body size for pixel conversions
        ref_size = self.base_params.get("REFERENCE_BODY_SIZE", 20.0)
        resize_f = self.base_params.get("RESIZE_FACTOR", 1.0)
        scaled_body_size = ref_size * resize_f

        self._preload_pose_data()

        # Seed the first trial with the user's current production settings.
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
                raise optuna.TrialPruned()

            trial_params = self._suggest_trial_params(trial, scaled_body_size)

            # Merge into full param set
            current_params = self.base_params.copy()
            current_params.update(trial_params)

            # Run tracking simulation and compute composite score
            composite, sub_scores, _ = self._run_tracking_loop(current_params)
            score = composite

            results.append(
                OptimizationResult(trial_params, score, trial.number, sub_scores)
            )
            pct = int(((trial.number + 1) / self.n_trials) * 100)
            self.progress_signal.emit(
                pct, f"Trial {trial.number + 1}/{self.n_trials} (Score: {score:.3f})"
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
                        self._stop_requested = True

            return score

        try:
            study.optimize(objective, n_trials=self.n_trials)
        except Exception as e:
            logger.error(f"Optimization trial failed: {e}")

        results.sort(key=lambda x: x.score)
        self.cache.close()
        self._pose_frame_cache = None  # free memory after optimization
        try:
            self.result_signal.emit(results)
            self.finished_signal.emit()
        except RuntimeError as e:
            # Dialog was closed before the thread finished; ignore orphaned signals.
            logger.warning("Optimizer signal emission skipped (dialog closed?): %s", e)

    def _run_tracking_loop(self, params: Dict[str, Any], reverse: bool = False):
        """
        Core tracking simulation shared by quality scoring and the FB consistency metric.

        Returns
        -------
        composite : float
            Lower-is-better composite score balancing four orthogonal objectives.
        frame_positions : Dict[int, np.ndarray]
            Maps frame_idx -> (N, 2) float32 array of KF-estimated positions.
            Rows for lost tracks contain NaN.
        """
        det_filter = DetectionFilter(params)
        kf_manager = KalmanFilterManager(params["MAX_TARGETS"], params)
        assigner = TrackAssigner(params)
        _roi_mask = params.get("ROI_MASK", None)

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
        n_frames = self.end_frame - self.start_frame + 1

        frame_order = (
            range(self.end_frame, self.start_frame - 1, -1)
            if reverse
            else range(self.start_frame, self.end_frame + 1)
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

            if meas:
                kf_manager.predict()

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
            else:
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

            # Record estimated positions (NaN = lost track)
            pos = np.full((N, 2), np.nan, dtype=np.float32)
            for r in range(N):
                if track_states[r] != "lost":
                    pos[r] = kf_manager.X[r, :2]
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
