"""
Optimized Track Assigner.
Compatible with Vectorized Kalman Filter.
Uses batch Mahalanobis distance and Numba-accelerated spatial assignment.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True)
def _compute_cost_matrix_numba(
    N,
    M,
    meas_pos,
    meas_ori,
    pred_pos,
    pred_ori,
    shapes_area,
    shapes_asp,
    prev_areas,
    prev_asps,
    S_inv_batch,
    use_maha,
    Wp,
    Wo,
    Wa,
    Wasp,
    per_track_gates,
    meas_ori_directed,
):
    """Numba kernel using pre-calculated batch Inverse Covariances.

    ``per_track_gates`` is a float32 array of shape (N,) providing each
    track's individual spatial cull distance.  This replaces the former
    scalar ``cull_threshold`` so that young/uncertain tracks get an
    appropriately expanded gate while established tracks keep a tight one.
    """
    cost = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        # Extract pre-calculated 2x2 inverse position covariance from the 3x3 S_inv
        # (This avoids N*M matrix inversions inside the loop)
        inv_S_pos = S_inv_batch[i, :2, :2]
        gate_i = per_track_gates[i]

        for j in range(M):
            diff = meas_pos[j] - pred_pos[i]

            # 1. Position Cost
            if use_maha:
                # Mahalanobis: sqrt(d^T * S_inv * d)
                maha_sq = (
                    diff[0] * (diff[0] * inv_S_pos[0, 0] + diff[1] * inv_S_pos[1, 0])
                    + diff[1] * (diff[0] * inv_S_pos[0, 1] + diff[1] * inv_S_pos[1, 1])
                )
                if maha_sq < 0.0:
                    maha_sq = 0.0
                pos_dist = np.sqrt(maha_sq)
            else:
                pos_dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

            # Spatial culling: each track uses its own adaptive gate so that
            # young/uncertain tracks are not unfairly blocked by the smallest
            # established-track gate.
            if pos_dist > gate_i:
                cost[i, j] = 1e6  # Large penalty
                continue

            # 2. Orientation Cost (Circular wrap)
            odiff = abs(pred_ori[i] - meas_ori[j])
            if odiff > np.pi:
                odiff = 2 * np.pi - odiff
            # OBB theta is an axis (0/180 equivalent) unless pose provides directed heading.
            if meas_ori_directed[j] == 0:
                alt = np.pi - odiff
                if alt < odiff:
                    odiff = alt

            # 3. Shape Costs
            area_diff = abs(shapes_area[j] - prev_areas[i])
            asp_diff = abs(shapes_asp[j] - prev_asps[i])

            cost[i, j] = Wp * pos_dist + Wo * odiff + Wa * area_diff + Wasp * asp_diff

    return cost


class TrackAssigner:
    """Handles assignment of detections to tracks with optimizations."""

    def __init__(self, params, worker=None):
        self.params = params
        self.worker = worker
        self._large_n_warning_shown = False  # Track if we've shown the warning

    def _spatial_optimization_enabled(self) -> bool:
        """Support both the current flag and the legacy alias."""
        return bool(
            self.params.get(
                "ENABLE_SPATIAL_OPTIMIZATION",
                self.params.get("USE_SPATIAL_PRUNING", False),
            )
        )

    def _get_spatial_candidates(self, N, M, pred_pos, meas_pos, max_dist):
        """Use KD-tree to find candidate matches within max_dist for large N."""
        if M == 0 or N == 0:
            return {}
        tree = cKDTree(meas_pos)
        candidates = {}
        for i in range(N):
            indices = tree.query_ball_point(pred_pos[i], max_dist)
            if indices:
                candidates[i] = indices
        return candidates

    def _compute_local_motion_gates(
        self,
        track_uncertainty: np.ndarray,
        track_avg_step: np.ndarray,
        cull_threshold: float,
    ) -> np.ndarray:
        p = self.params
        reference_body_size = max(1.0, float(p.get("REFERENCE_BODY_SIZE", 20.0)))
        gate_multiplier = float(p.get("ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER", 1.4))
        uncertainty_ref = max(1.0, reference_body_size**2)
        unc_scale = np.minimum(2.0, track_uncertainty / uncertainty_ref)
        mot_scale = np.minimum(2.0, track_avg_step / reference_body_size)
        return (
            cull_threshold
            * gate_multiplier
            * (1.0 + 0.5 * unc_scale + 0.35 * mot_scale)
        ).astype(np.float32, copy=False)

    @staticmethod
    def _orientation_diff(pred_theta, meas_theta, directed: bool) -> float:
        odiff = abs(float(pred_theta) - float(meas_theta))
        if odiff > np.pi:
            odiff = 2 * np.pi - odiff
        if not directed:
            alt = np.pi - odiff
            if alt < odiff:
                odiff = alt
        return float(max(0.0, odiff))

    @staticmethod
    def _pose_paired_stats(
        det_pose, track_pose, min_shared: int = 3
    ) -> tuple[float | None, int]:
        if det_pose is None or track_pose is None:
            return None, 0
        det_arr = np.asarray(det_pose, dtype=np.float32)
        track_arr = np.asarray(track_pose, dtype=np.float32)
        if (
            det_arr.shape != track_arr.shape
            or det_arr.ndim != 2
            or det_arr.shape[1] < 2
        ):
            return None, 0

        dists = []
        for kp_idx in range(len(det_arr)):
            det_valid = np.isfinite(det_arr[kp_idx, 0]) and np.isfinite(
                det_arr[kp_idx, 1]
            )
            track_valid = np.isfinite(track_arr[kp_idx, 0]) and np.isfinite(
                track_arr[kp_idx, 1]
            )
            if not (det_valid and track_valid):
                continue
            dist = float(np.linalg.norm(det_arr[kp_idx, :2] - track_arr[kp_idx, :2]))
            if np.isfinite(dist):
                dists.append(dist)

        if len(dists) < min_shared:
            return None, len(dists)

        dists_arr = np.asarray(dists, dtype=np.float32)
        med = float(np.median(dists_arr))
        abs_dev = np.abs(dists_arr - med)
        mad = float(np.median(abs_dev))
        if mad > 1e-6:
            keep = abs_dev <= (2.5 * mad)
            filtered = dists_arr[keep]
            if len(filtered) >= min_shared:
                dists_arr = filtered
        if len(dists_arr) >= 5:
            cutoff = max(1, int(np.floor(len(dists_arr) * 0.2)))
            dists_arr = (
                np.sort(dists_arr)[:-cutoff] if cutoff < len(dists_arr) else dists_arr
            )
        return float(np.mean(dists_arr)), int(len(dists_arr))

    def _has_pose_association_data(self, association_data) -> bool:
        if not association_data:
            return False
        kpts = association_data.get("detection_pose_keypoints")
        protos = association_data.get("track_pose_prototypes")
        has_kpts = kpts is not None and any(k is not None for k in kpts)
        has_protos = protos is not None and any(p is not None for p in protos)
        return has_kpts or has_protos

    def _normalize_cnn_phases(self, association_data) -> list[dict[str, Any]]:
        if not association_data:
            return []
        cnn_phases = association_data.get("cnn_phases", None)
        if cnn_phases is not None:
            return list(cnn_phases)
        return []

    def _identity_scale(self, association_data, cnn_phases) -> float:
        use_legacy_tag_hints = not self.params.get(
            "ENABLE_IDENTITY_ONLINE_DECODER", False
        ) or bool(self.params.get("ASSOCIATION_USE_LEGACY_TAG_HINTS", False))
        use_legacy_cnn_hints = not self.params.get(
            "ENABLE_IDENTITY_ONLINE_DECODER", False
        ) or bool(self.params.get("ASSOCIATION_USE_LEGACY_CNN_HINTS", False))
        det_tag_ids = (
            association_data.get("detection_tag_ids", []) if association_data else []
        )
        track_tag_ids = (
            association_data.get("track_last_tag_ids", []) if association_data else []
        )

        def _has_valid_tag(values) -> bool:
            return any(value is not None and int(value) != -1 for value in values)

        has_tag_factor = (
            use_legacy_tag_hints
            and _has_valid_tag(det_tag_ids)
            and _has_valid_tag(track_tag_ids)
        )
        n_cnn_factors = len(cnn_phases) if use_legacy_cnn_hints else 0
        n_identity_factors = (1 if has_tag_factor else 0) + n_cnn_factors
        base = 1.0 / n_identity_factors if n_identity_factors > 1 else 1.0
        hint_scale = float(
            self.params.get(
                "ASSOCIATION_IDENTITY_HINT_SCALE",
                0.35
                if self.params.get("ENABLE_IDENTITY_ONLINE_DECODER", False)
                else 1.0,
            )
        )
        return base * max(0.0, hint_scale)

    def _apply_tag_identity_overlay(
        self,
        cost: np.ndarray,
        association_data: Dict[str, Any] | None,
        identity_scale: float,
    ) -> None:
        if self.params.get("ENABLE_IDENTITY_ONLINE_DECODER", False) and not bool(
            self.params.get("ASSOCIATION_USE_LEGACY_TAG_HINTS", False)
        ):
            return
        if not association_data:
            return
        det_tag_ids = association_data.get("detection_tag_ids", [])
        track_tag_ids = association_data.get("track_last_tag_ids", [])
        if not det_tag_ids or not track_tag_ids:
            return

        n_tracks, n_dets = cost.shape
        det_arr = np.full(n_dets, -1, dtype=np.int32)
        trk_arr = np.full(n_tracks, -1, dtype=np.int32)
        det_limit = min(n_dets, len(det_tag_ids))
        trk_limit = min(n_tracks, len(track_tag_ids))
        if det_limit > 0:
            det_arr[:det_limit] = np.asarray(det_tag_ids[:det_limit], dtype=np.int32)
        if trk_limit > 0:
            trk_arr[:trk_limit] = np.asarray(track_tag_ids[:trk_limit], dtype=np.int32)

        valid = (trk_arr[:, None] != -1) & (det_arr[None, :] != -1)
        if not np.any(valid):
            return
        match = valid & (trk_arr[:, None] == det_arr[None, :])
        mismatch = valid & ~match
        cost[match] -= float(self.params.get("TAG_MATCH_BONUS", 20.0)) * identity_scale
        cost[mismatch] += (
            float(self.params.get("TAG_MISMATCH_PENALTY", 50.0)) * identity_scale
        )

    def _apply_cnn_identity_overlays(
        self,
        cost: np.ndarray,
        cnn_phases: list[dict[str, Any]],
        identity_scale: float,
    ) -> None:
        if self.params.get("ENABLE_IDENTITY_ONLINE_DECODER", False) and not bool(
            self.params.get("ASSOCIATION_USE_LEGACY_CNN_HINTS", False)
        ):
            return
        if not cnn_phases:
            return

        n_tracks, n_dets = cost.shape
        for phase in cnn_phases:
            det_classes = np.asarray(
                list(phase.get("detection_classes", []))[:n_dets], dtype=object
            )
            track_classes = np.asarray(
                list(phase.get("track_identities", []))[:n_tracks], dtype=object
            )
            if det_classes.size == 0 or track_classes.size == 0:
                continue

            track_valid = np.not_equal(track_classes[:, None], None)
            det_valid = np.not_equal(det_classes[None, :], None)
            valid = track_valid & det_valid
            if not np.any(valid):
                continue
            match = valid & (track_classes[:, None] == det_classes[None, :])
            mismatch = valid & ~match
            cost[match] -= float(phase.get("match_bonus", 20.0)) * identity_scale
            cost[mismatch] += (
                float(phase.get("mismatch_penalty", 50.0)) * identity_scale
            )

    @staticmethod
    def _apply_candidate_gate(
        cost: np.ndarray, candidates: Dict[int, List[int]]
    ) -> None:
        if not candidates:
            return
        allowed = np.zeros(cost.shape, dtype=bool)
        for track_idx, det_indices in candidates.items():
            if det_indices:
                allowed[track_idx, det_indices] = True
        cost[~allowed] = 1e6

    def _apply_pose_rejection_overlay(
        self,
        cost: np.ndarray,
        candidates: Dict[int, List[int]],
        association_data: Dict[str, Any],
    ) -> None:
        detection_pose_keypoints = list(
            association_data.get("detection_pose_keypoints", [None] * cost.shape[1])
        )
        detection_pose_visibility = np.asarray(
            association_data.get(
                "detection_pose_visibility", np.zeros(cost.shape[1], dtype=np.float32)
            ),
            dtype=np.float32,
        )
        track_pose_prototypes = list(
            association_data.get("track_pose_prototypes", [None] * cost.shape[0])
        )
        pose_rejection_enabled = bool(self.params.get("ENABLE_POSE_REJECTION", True))
        if not pose_rejection_enabled:
            return

        pose_veto_threshold = float(self.params.get("POSE_REJECTION_THRESHOLD", 0.5))
        pose_min_visibility = float(
            self.params.get("POSE_REJECTION_MIN_VISIBILITY", 0.5)
        )

        for track_idx, det_indices in candidates.items():
            track_pose_proto = (
                track_pose_prototypes[track_idx]
                if track_idx < len(track_pose_prototypes)
                else None
            )
            if track_pose_proto is None:
                continue
            for det_idx in det_indices:
                if cost[track_idx, det_idx] >= 1e6:
                    continue
                visibility = (
                    float(detection_pose_visibility[det_idx])
                    if det_idx < len(detection_pose_visibility)
                    else 0.0
                )
                visibility = float(np.clip(visibility, 0.0, 1.0))
                det_pose_proto = (
                    detection_pose_keypoints[det_idx]
                    if det_idx < len(detection_pose_keypoints)
                    else None
                )
                pose_dist, shared_keypoints = self._pose_paired_stats(
                    det_pose_proto, track_pose_proto
                )
                adaptive_pose_threshold = pose_veto_threshold
                if shared_keypoints > 0 and (
                    shared_keypoints <= 3
                    or visibility < min(1.0, pose_min_visibility + 0.15)
                ):
                    adaptive_pose_threshold *= 1.2
                if (
                    pose_dist is not None
                    and visibility >= pose_min_visibility
                    and pose_dist > adaptive_pose_threshold
                ):
                    cost[track_idx, det_idx] = 1e6

    def _compute_stage1_gate(
        self,
        N,
        M,
        meas_pos,
        pred_pos,
        shapes_area,
        shapes_asp,
        prev_areas,
        prev_asps,
        S_inv_batch,
        track_uncertainty,
        track_avg_step,
        cull_threshold,
        local_gates: np.ndarray | None = None,
    ):
        p = self.params
        max_area_ratio = float(p.get("ASSOCIATION_STAGE1_MAX_AREA_RATIO", 2.5))
        max_aspect_diff = float(p.get("ASSOCIATION_STAGE1_MAX_ASPECT_DIFF", 0.8))

        # --- Vectorized position distances (N × M) ---
        diff = meas_pos[None, :, :] - pred_pos[:, None, :]  # (N, M, 2)
        if p["USE_MAHALANOBIS"]:
            S_inv_2x2 = S_inv_batch[:, :2, :2]  # (N, 2, 2)
            maha_sq = np.einsum("nmd,nde,nme->nm", diff, S_inv_2x2, diff)
            np.maximum(maha_sq, 0.0, out=maha_sq)
            pos_dist = np.sqrt(maha_sq)
        else:
            pos_dist = np.linalg.norm(diff, axis=2)  # (N, M)

        # --- Per-track adaptive gate threshold ---
        if local_gates is None:
            local_gates = self._compute_local_motion_gates(
                np.asarray(track_uncertainty, dtype=np.float32),
                np.asarray(track_avg_step, dtype=np.float32),
                cull_threshold,
            )

        # --- Vectorized area ratio and aspect diff ---
        _prev = np.maximum(prev_areas, 1e-6)[:, None]  # (N, 1)
        _curr = np.maximum(shapes_area, 1e-6)[None, :]  # (1, M)
        area_ratio = np.maximum(_prev, _curr) / np.maximum(
            np.minimum(_prev, _curr), 1e-6
        )
        asp_diff = np.abs(shapes_asp[None, :] - prev_asps[:, None])

        # --- Build boolean pass mask and extract candidates ---
        mask = (
            (pos_dist <= local_gates[:, None])
            & (area_ratio <= max_area_ratio)
            & (asp_diff <= max_aspect_diff)
        )

        candidates = {}
        for i in range(N):
            indices = np.where(mask[i])[0]
            if len(indices) > 0:
                candidates[i] = indices.tolist()
        return candidates

    def compute_cost_matrix(
        self,
        N: int,
        measurements: List[np.ndarray],
        predictions: np.ndarray,
        shapes: List[Tuple[float, float]],
        kf_manager: Any,
        last_shape_info: List[Any],
        meas_ori_directed: np.ndarray | None = None,
        association_data: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Computes cost matrix. Compatible with Vectorized Kalman Filter.
        """
        p = self.params
        M = len(measurements)
        if M == 0:
            return np.zeros((N, 0), np.float32), {}

        # Warn about spatial indexing for large N
        if (
            N > 25
            and not self._spatial_optimization_enabled()
            and not self._large_n_warning_shown
        ):
            warning_msg = (
                f"Tracking {N} objects without spatial indexing may be slow.\n\n"
                f"Consider enabling these optimizations in tracking_config.json:\n"
                f"  • ENABLE_SPATIAL_OPTIMIZATION: true\n"
                f"  • ENABLE_GREEDY_ASSIGNMENT: true\n\n"
                f"Expected performance improvement: 10-30% for {N}+ objects."
            )
            logger.warning(warning_msg.replace("\n", " "))
            if self.worker is not None:
                self.worker.warning_signal.emit(
                    "Performance Optimization Available", warning_msg
                )
            self._large_n_warning_shown = True

        # Get pre-calculated Inverse Innovation Covariances from Manager
        S_inv_batch = kf_manager.get_mahalanobis_matrices()

        # Diagnostic guard: assignment requires finite numeric inputs.
        if not np.isfinite(S_inv_batch).all():
            bad = int(np.size(S_inv_batch) - np.count_nonzero(np.isfinite(S_inv_batch)))
            raise ValueError(
                f"non-finite Kalman S_inv entries ({bad}) before cost construction"
            )

        # Pre-extract arrays for Numba (Avoids attribute access in loop)
        meas_pos = np.array([m[:2] for m in measurements], dtype=np.float32)
        meas_ori = np.array([m[2] for m in measurements], dtype=np.float32)
        if meas_ori_directed is None:
            meas_ori_directed_arr = np.zeros(M, dtype=np.uint8)
        else:
            meas_ori_directed_arr = np.asarray(meas_ori_directed, dtype=np.uint8)
            if len(meas_ori_directed_arr) != M:
                logger.warning(
                    "meas_ori_directed length mismatch (%d != %d); falling back to axis mode.",
                    len(meas_ori_directed_arr),
                    M,
                )
                meas_ori_directed_arr = np.zeros(M, dtype=np.uint8)
        pred_pos = predictions[:, :2]  # Predictions are already (N, 3)
        pred_ori = predictions[:, 2]

        if not np.isfinite(meas_pos).all() or not np.isfinite(meas_ori).all():
            bad_pos = int(np.size(meas_pos) - np.count_nonzero(np.isfinite(meas_pos)))
            bad_ori = int(np.size(meas_ori) - np.count_nonzero(np.isfinite(meas_ori)))
            raise ValueError(
                f"non-finite detection measurement entries (pos={bad_pos}, ori={bad_ori})"
            )
        if not np.isfinite(pred_pos).all() or not np.isfinite(pred_ori).all():
            bad_pos = int(np.size(pred_pos) - np.count_nonzero(np.isfinite(pred_pos)))
            bad_ori = int(np.size(pred_ori) - np.count_nonzero(np.isfinite(pred_ori)))
            raise ValueError(
                f"non-finite Kalman prediction entries (pos={bad_pos}, ori={bad_ori})"
            )

        # Override meas_ori with the directed heading where headtail or
        # high-confidence pose supplies a reliable direction.
        if association_data is not None:
            _dh = association_data.get("detection_pose_heading")
            if _dh is not None:
                _dh_arr = np.asarray(_dh, dtype=np.float32)
                for _j in range(min(M, len(_dh_arr))):
                    if meas_ori_directed_arr[_j] == 1 and np.isfinite(_dh_arr[_j]):
                        meas_ori[_j] = _dh_arr[_j]

        shapes_area = np.array([s[0] for s in shapes], dtype=np.float32)
        shapes_asp = np.array([s[1] for s in shapes], dtype=np.float32)

        # Optimized fill for previous shape info
        prev_areas = np.zeros(N, dtype=np.float32)
        prev_asps = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if last_shape_info[i] is not None:
                prev_areas[i], prev_asps[i] = last_shape_info[i]
            else:
                prev_areas[i], prev_asps[i] = shapes_area[0], shapes_asp[0]

        MAX_DIST = p.get("MAX_DISTANCE_THRESHOLD", 1000.0)
        cull_threshold = (
            min(
                max(MAX_DIST / max(p.get("W_POSITION", 1.0), 1e-6), 50.0),
                MAX_DIST * 3.0,  # never search beyond 3× the hard distance limit
            )
            if p.get("W_POSITION", 1.0) > 0
            else 1e6
        )

        has_pose_data = self._has_pose_association_data(association_data)
        pose_candidates = {}
        local_gates = None
        track_uncertainty = None
        track_avg_step = None
        # Always compute per-track adaptive gates (not only for pose data).
        # Young and high-uncertainty tracks get an expanded search radius so
        # they are not incorrectly blocked by the established-track gate.
        track_uncertainty = (
            np.asarray(kf_manager.get_position_uncertainties(), dtype=np.float32)
            if hasattr(kf_manager, "get_position_uncertainties")
            else np.trace(kf_manager.P[:N, :2, :2], axis1=1, axis2=2).astype(np.float32)
        )
        track_avg_step_arr = np.asarray(
            (
                association_data.get("track_avg_step", np.zeros(N))
                if association_data is not None
                else np.zeros(N)
            ),
            dtype=np.float32,
        )
        local_gates = self._compute_local_motion_gates(
            track_uncertainty,
            track_avg_step_arr,
            cull_threshold,
        )
        if has_pose_data:
            # local_gates and track_uncertainty are already computed above.
            # track_avg_step_arr is also available; alias it for _compute_stage1_gate.
            track_avg_step = track_avg_step_arr
            pose_candidates = self._compute_stage1_gate(
                N,
                M,
                meas_pos,
                pred_pos,
                shapes_area,
                shapes_asp,
                prev_areas,
                prev_asps,
                S_inv_batch,
                track_uncertainty,
                track_avg_step,
                cull_threshold,
                local_gates=local_gates,
            )

        spatial_candidates = {}
        if has_pose_data and self._spatial_optimization_enabled() and N > 50:
            spatial_candidates = pose_candidates
            # KD-Tree mode uses a hybrid approach
            cost = self._compute_cost_python_fallback(
                N,
                M,
                meas_pos,
                meas_ori,
                pred_pos,
                pred_ori,
                shapes_area,
                shapes_asp,
                prev_areas,
                prev_asps,
                S_inv_batch,
                p,
                spatial_candidates,
                meas_ori_directed_arr,
            )
        elif self._spatial_optimization_enabled() and N > 50:
            spatial_candidates = self._get_spatial_candidates(
                N, M, pred_pos, meas_pos, cull_threshold
            )
            # KD-Tree mode uses a hybrid approach
            cost = self._compute_cost_python_fallback(
                N,
                M,
                meas_pos,
                meas_ori,
                pred_pos,
                pred_ori,
                shapes_area,
                shapes_asp,
                prev_areas,
                prev_asps,
                S_inv_batch,
                p,
                spatial_candidates,
                meas_ori_directed_arr,
            )
        else:
            cost = _compute_cost_matrix_numba(
                N,
                M,
                meas_pos,
                meas_ori,
                pred_pos,
                pred_ori,
                shapes_area,
                shapes_asp,
                prev_areas,
                prev_asps,
                S_inv_batch,
                p["USE_MAHALANOBIS"],
                p["W_POSITION"],
                p["W_ORIENTATION"],
                p["W_AREA"],
                p["W_ASPECT"],
                local_gates,
                meas_ori_directed_arr,
            )

        if association_data:
            cnn_phases = self._normalize_cnn_phases(association_data)
            identity_scale = self._identity_scale(association_data, cnn_phases)
            self._apply_tag_identity_overlay(cost, association_data, identity_scale)
            self._apply_cnn_identity_overlays(cost, cnn_phases, identity_scale)

            if has_pose_data:
                self._apply_candidate_gate(cost, pose_candidates)
                self._apply_pose_rejection_overlay(
                    cost, pose_candidates, association_data
                )
            elif spatial_candidates:
                pose_candidates = spatial_candidates

            return cost, pose_candidates

        return cost, spatial_candidates

    def compute_assignment_confidence(
        self: object, cost: object, matched_pairs: object
    ) -> object:
        """Compute confidence scores for assignments."""
        if not matched_pairs:
            return {}
        scale = self.params.get("MAX_DISTANCE_THRESHOLD", 100.0) * 0.5
        return {r: 1.0 / (1.0 + cost[r, c] / scale) for r, c in matched_pairs}

    def _compute_distance_gates(self, N, M, meas, tracking_continuity, kf_manager):
        """Compute per-track distance gates and the raw Euclidean distance matrix.

        Returns ``(per_track_gate, raw_dist_mat, meas_xy)``.
        """
        p = self.params
        THRESH = p.get("KALMAN_MATURITY_AGE", 10)
        MAX_DIST = p["MAX_DISTANCE_THRESHOLD"]
        _young_mult = max(1.0, float(p.get("KALMAN_YOUNG_GATE_MULTIPLIER", 1.0)))
        per_track_gate = np.where(
            np.array([tracking_continuity[r] for r in range(N)], dtype=np.float32)
            < THRESH,
            MAX_DIST * _young_mult,
            MAX_DIST,
        )
        meas_xy = np.array([meas[j][:2] for j in range(M)], dtype=np.float32)
        raw_dist_mat = np.linalg.norm(
            np.asarray(kf_manager.X[:N, :2], dtype=np.float32)[:, None, :]
            - meas_xy[None, :, :],
            axis=2,
        )
        return per_track_gate, raw_dist_mat, meas_xy

    def _assign_established_greedy(
        self, est, M, cost, raw_dist_mat, MAX_DIST, VEL_GATE
    ):
        """Phase 1 greedy assignment for established tracks."""
        track_det_costs = []
        for r in est:
            for c in range(M):
                if cost[r, c] < MAX_DIST and raw_dist_mat[r, c] < VEL_GATE:
                    track_det_costs.append((cost[r, c], r, c))
        track_det_costs.sort()
        assignments = []
        assigned_dets = set()
        assigned_r = set()
        for _, r, c in track_det_costs:
            if r not in assigned_r and c not in assigned_dets:
                assignments.append((r, c))
                assigned_dets.add(c)
                assigned_r.add(r)
        return assignments, assigned_dets

    def _assign_established_hungarian(
        self, est, cost, raw_dist_mat, MAX_DIST, VEL_GATE
    ):
        """Phase 1 Hungarian assignment for established tracks.

        ``est`` is always built from ``for i in range(N) if ...`` so it is
        monotonically increasing.  ``linear_sum_assignment(cost[est, :])``
        returns row indices 0..len(est)-1 into the submatrix; ``est[r_idx]``
        maps each back to the original track index.  Making the sort explicit
        here documents and enforces this invariant so that the mapping is safe
        even if the calling code ever builds ``est`` differently.
        """
        if not est:
            return [], set()
        est_sorted = sorted(est)
        assignments = []
        assigned_dets = set()
        cost_sub = cost[est_sorted, :]
        if not np.isfinite(cost_sub).all():
            bad = int(np.size(cost_sub) - np.count_nonzero(np.isfinite(cost_sub)))
            raise ValueError(
                f"assignment submatrix contains non-finite values (bad={bad}, tracks={len(est_sorted)}, dets={cost_sub.shape[1]})"
            )
        rows, cols = linear_sum_assignment(cost_sub)
        for r_idx, c in zip(rows, cols):
            r = est_sorted[r_idx]
            if cost[r, c] < MAX_DIST and raw_dist_mat[r, c] < VEL_GATE:
                assignments.append((r, c))
                assigned_dets.add(c)
        return assignments, assigned_dets

    def _assign_unstable(
        self,
        unst,
        M,
        cost,
        meas,
        kf_manager,
        tracking_continuity,
        per_track_gate,
        MAX_DIST,
        assigned_dets,
    ):
        """Phase 2: greedily assign unstable (young) tracks."""
        assignments = []
        for r in sorted(unst, key=lambda i: tracking_continuity[i], reverse=True):
            avail = [j for j in range(M) if j not in assigned_dets]
            if not avail:
                break
            best_c = avail[np.argmin(cost[r, avail])]
            # Skip if the cheapest candidate is still beyond the cost sentinel
            # (all remaining detections are blocked by a hard gate).
            if cost[r, best_c] >= 1e6:
                continue
            raw_dist = float(
                np.linalg.norm(np.asarray(meas[best_c][:2]) - kf_manager.X[r, :2])
            )
            if cost[r, best_c] < MAX_DIST and raw_dist < float(per_track_gate[r]):
                assignments.append((r, best_c))
                assigned_dets.add(best_c)
        return assignments

    def _assign_respawn(
        self,
        lost,
        M,
        meas,
        kf_manager,
        track_states,
        N,
        MAX_DIST,
        assigned_dets,
    ):
        """Phase 3: match lost tracks with unassigned detections.

        Trajectory-ID management is intentionally NOT done here — it is the
        worker's responsibility to assign new IDs when it processes
        ``respawned_matches``.  This keeps all trajectory-ID amendments in a
        single code path (worker KF-update loop + worker free_dets loop).
        """
        p = self.params
        unassigned = [j for j in range(M) if j not in assigned_dets]
        respawn_dist_limit = p.get("MIN_RESPAWN_DISTANCE", MAX_DIST * 0.8)
        non_lost_positions = [
            np.asarray(kf_manager.X[r, :2], dtype=np.float32)
            for r in range(N)
            if track_states[r] != "lost"
        ]
        assignments = []
        for c in unassigned:
            if not lost:
                break
            min_dist_non_lost = (
                min(
                    np.linalg.norm(meas[c][:2] - track_pos)
                    for track_pos in non_lost_positions
                )
                if non_lost_positions
                else 1e6
            )
            if min_dist_non_lost < respawn_dist_limit:
                continue
            best_r, best_c_val = None, 1e6
            for r in lost:
                last_pos = kf_manager.X[r, :2]
                dist = np.linalg.norm(meas[c][:2] - last_pos)
                if dist < best_c_val:
                    best_c_val, best_r = dist, r
            if best_r is not None and best_c_val < MAX_DIST:
                assignments.append((best_r, c))
                assigned_dets.add(c)
                lost.remove(best_r)
        return assignments

    def assign_tracks(
        self: object,
        cost: object,
        N: object,
        M: object,
        meas: object,
        track_states: object,
        tracking_continuity: object,
        kf_manager: object,
        spatial_candidates: object = None,
        association_data: Dict[str, Any] | None = None,
    ) -> object:
        """
        Drop-in replacement for track assignment logic.
        Compatible with kf_manager.X state access.

        Trajectory-ID management has been removed from this method — the
        worker is solely responsible for assigning new IDs so that all
        amendments happen in one place.
        """
        p = self.params
        if M == 0:
            return [], [], [], []

        THRESH = p.get("KALMAN_MATURITY_AGE", 10)
        MAX_DIST = p["MAX_DISTANCE_THRESHOLD"]
        USE_GREEDY = p.get("ENABLE_GREEDY_ASSIGNMENT", False)
        _body_size = p.get("REFERENCE_BODY_SIZE", 20.0) * p.get("RESIZE_FACTOR", 1.0)
        VEL_GATE = p.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0) * _body_size

        # Pre-gate: block physically impossible (track, detection) pairs.
        per_track_gate, raw_dist_mat, _ = self._compute_distance_gates(
            N,
            M,
            meas,
            tracking_continuity,
            kf_manager,
        )
        cost[raw_dist_mat >= per_track_gate[:, None]] = 1e9

        # Split tracks by state
        est = [
            i
            for i in range(N)
            if tracking_continuity[i] >= THRESH and track_states[i] != "lost"
        ]
        unst = [
            i
            for i in range(N)
            if tracking_continuity[i] < THRESH and track_states[i] != "lost"
        ]
        lost = [i for i in range(N) if track_states[i] == "lost"]
        all_assignments = []
        assigned_dets = set()

        # Phase 1: Established Tracks
        if est:
            if USE_GREEDY:
                ph1, ph1_dets = self._assign_established_greedy(
                    est,
                    M,
                    cost,
                    raw_dist_mat,
                    MAX_DIST,
                    VEL_GATE,
                )
            else:
                ph1, ph1_dets = self._assign_established_hungarian(
                    est,
                    cost,
                    raw_dist_mat,
                    MAX_DIST,
                    VEL_GATE,
                )
            all_assignments.extend(ph1)
            assigned_dets.update(ph1_dets)

        # Phase 2: Unstable Tracks
        ph2 = self._assign_unstable(
            unst,
            M,
            cost,
            meas,
            kf_manager,
            tracking_continuity,
            per_track_gate,
            MAX_DIST,
            assigned_dets,
        )
        all_assignments.extend(ph2)

        # Phase 3: Respawn Lost Tracks
        ph3 = self._assign_respawn(
            lost,
            M,
            meas,
            kf_manager,
            track_states,
            N,
            MAX_DIST,
            assigned_dets,
        )
        all_assignments.extend(ph3)

        if not all_assignments:
            return [], [], list(range(M)), []

        final_r, final_c = zip(*all_assignments)
        free_dets = list(set(range(M)) - set(final_c))
        return list(final_r), list(final_c), free_dets, []

    def _compute_cost_python_fallback(
        self,
        N,
        M,
        meas_pos,
        meas_ori,
        pred_pos,
        pred_ori,
        sh_area,
        sh_asp,
        pr_area,
        pr_asp,
        S_inv,
        p,
        candidates,
        meas_ori_directed,
    ):
        """Python fallback for spatial optimization."""
        cost = np.full((N, M), 1e6, dtype=np.float32)
        Wp, Wo, Wa, Wasp = (
            p["W_POSITION"],
            p["W_ORIENTATION"],
            p["W_AREA"],
            p["W_ASPECT"],
        )

        for r, det_indices in candidates.items():
            inv_S = S_inv[r, :2, :2]
            for c in det_indices:
                diff = meas_pos[c] - pred_pos[r]
                if p["USE_MAHALANOBIS"]:
                    maha_sq = float(diff @ inv_S @ diff)
                    maha_sq = max(maha_sq, 0.0)
                    pos_c = np.sqrt(maha_sq)
                else:
                    pos_c = np.linalg.norm(diff)

                odiff = abs(pred_ori[r] - meas_ori[c])
                if odiff > np.pi:
                    odiff = 2 * np.pi - odiff
                if meas_ori_directed[c] == 0:
                    odiff = min(odiff, np.pi - odiff)

                cost[r, c] = (
                    Wp * pos_c
                    + Wo * odiff
                    + Wa * abs(sh_area[c] - pr_area[r])
                    + Wasp * abs(sh_asp[c] - pr_asp[r])
                )
        return cost
