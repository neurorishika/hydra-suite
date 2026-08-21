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
def _compute_cost_matrix_numba_core(
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
    track_arena,
    meas_arena,
):
    """Numba kernel using pre-calculated batch Inverse Covariances.

    ``per_track_gates`` is a float32 array of shape (N,) providing each
    track's individual spatial cull distance.  This replaces the former
    scalar ``cull_threshold`` so that young/uncertain tracks get an
    appropriately expanded gate while established tracks keep a tight one.

    ``track_arena``/``meas_arena`` are int32 arrays of length N and M. When
    both are length-0 sentinels (or otherwise mismatched in length), no arena
    gating is applied and the result is bit-identical to the
    pre-multi-arena kernel. Cross-arena pairs get the same ``1e6``
    hard-reject sentinel used for distance-gated pairs, so no cross-arena
    pair is ever *accepted*.

    That sentinel does NOT, on its own, decompose the downstream Hungarian
    solve into independent per-arena problems: the reject sentinels are not
    all the same number (``1e6`` here, ``1e9`` for the raw distance/velocity
    gate in ``assign_tracks``), and a square solve that has to park surplus
    rows somewhere is therefore not indifferent between them. Independence is
    obtained structurally instead, by solving one sub-block per arena --
    see ``_assign_established_hungarian``.
    """
    cost = np.zeros((N, M), dtype=np.float32)
    gate_arenas = track_arena.shape[0] == N and meas_arena.shape[0] == M

    for i in range(N):
        # Extract pre-calculated 2x2 inverse position covariance from the 3x3 S_inv
        # (This avoids N*M matrix inversions inside the loop)
        inv_S_pos = S_inv_batch[i, :2, :2]
        gate_i = per_track_gates[i]
        arena_i = track_arena[i] if gate_arenas else 0

        for j in range(M):
            # Arena gating first: skips all downstream work for blocked pairs,
            # which is what keeps 100 arenas tractable.
            if gate_arenas and meas_arena[j] != arena_i:
                cost[i, j] = 1e6
                continue

            diff = meas_pos[j] - pred_pos[i]

            # 1. Position Cost
            if use_maha:
                # Mahalanobis: sqrt(d^T * S_inv * d)
                maha_sq = diff[0] * (
                    diff[0] * inv_S_pos[0, 0] + diff[1] * inv_S_pos[1, 0]
                ) + diff[1] * (diff[0] * inv_S_pos[0, 1] + diff[1] * inv_S_pos[1, 1])
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


_NO_ARENA = np.zeros(0, dtype=np.int32)


def _arena_arrays(track_arena, meas_arena, N, M):
    """Normalize optional arena arrays to numba-safe int32 arrays.

    Returns the length-0 sentinel pair when gating is off, which the kernel
    detects by shape and skips entirely.
    """
    if track_arena is None or meas_arena is None:
        return _NO_ARENA, _NO_ARENA
    ta = np.ascontiguousarray(track_arena, dtype=np.int32)
    ma = np.ascontiguousarray(meas_arena, dtype=np.int32)
    if ta.shape[0] != N or ma.shape[0] != M:
        return _NO_ARENA, _NO_ARENA
    return ta, ma


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
    track_arena=None,
    meas_arena=None,
):
    """Thin, non-jitted dispatch wrapper around ``_compute_cost_matrix_numba_core``.

    Normalizes ``track_arena``/``meas_arena`` (``None`` or mismatched-length
    inputs both mean "no gating") to the ``_NO_ARENA`` sentinel via
    ``_arena_arrays`` before calling the cached numba kernel, so the compiled
    kernel itself never has to type-infer over an ``Optional`` argument --
    it always sees concrete int32 arrays. ``None``/``None`` (the default)
    reproduces the pre-multi-arena kernel bit-for-bit.
    """
    ta, ma = _arena_arrays(track_arena, meas_arena, N, M)
    return _compute_cost_matrix_numba_core(
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
        ta,
        ma,
    )


def _pairwise_log_compat(posts, likes):
    """Log-compatibility matrix between track posteriors and detection likelihoods.

    Element ``[i, j]`` equals ``np.logaddexp.reduce(posts[i] + likes[j])`` --
    the log-domain inner product of a track's identity belief with a
    detection's identity evidence.  The reduction is accumulated with the same
    sequential ``logaddexp`` order as the per-pair scalar form, so the result is
    bit-identical to computing each pair on its own.

    Falls back to the per-pair loop when the inputs are ragged or of mixed
    dtype, where stacking would change the arithmetic.
    """
    post_arrs = [np.asarray(a) for a in posts]
    like_arrs = [np.asarray(a) for a in likes]
    if not post_arrs or not like_arrs:
        return np.zeros((len(post_arrs), len(like_arrs)), dtype=np.float64)

    def _uniform(arrs):
        first = arrs[0]
        return first.ndim == 1 and all(
            a.shape == first.shape and a.dtype == first.dtype for a in arrs
        )

    def _per_pair():
        return np.array(
            [[float(np.logaddexp.reduce(a + b)) for b in like_arrs] for a in post_arrs],
            dtype=np.float64,
        )

    n_classes = post_arrs[0].shape[0] if post_arrs[0].ndim == 1 else 0
    if not (_uniform(post_arrs) and _uniform(like_arrs)) or n_classes == 0:
        return _per_pair()

    P = np.stack(post_arrs)
    L = np.stack(like_arrs)
    if L.shape[1] != n_classes:
        return _per_pair()
    acc = P[:, None, 0] + L[None, :, 0]
    for k in range(1, n_classes):
        acc = np.logaddexp(acc, P[:, None, k] + L[None, :, k])
    return acc.astype(np.float64, copy=False)


class TrackAssigner:
    """Handles assignment of detections to tracks with optimizations."""

    def __init__(self, params, worker=None):
        self.params = params
        self.worker = worker
        self._large_n_warning_shown = False  # Track if we've shown the warning
        self.track_arena = None  # set by the worker via set_track_arena()

    def set_track_arena(self, track_arena) -> None:
        """Install the static per-slot arena mapping (None disables gating)."""
        self.track_arena = (
            None
            if track_arena is None
            else np.ascontiguousarray(track_arena, dtype=np.int32)
        )

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

    def _apply_bayesian_identity_cost(
        self,
        cost: np.ndarray,
        association_data: Dict[str, Any] | None,
        meas_arena: np.ndarray | None = None,
    ) -> None:
        """Add a soft Bayesian identity cost term to the assignment cost matrix.

        For each (track i, detection j) pair:
            identity_cost[i,j] = -logsumexp(track_log_posterior[i] + det_log_likelihood[j])

        This is the log-compatibility between the track's identity belief and
        the detection's identity evidence.  When the track is uncertain (uniform
        posterior), the term is constant across all detection columns and
        contributes nothing to the cost differential — giving a natural cold-start
        fallback without any special-casing.
        """
        if not self.params.get("ENABLE_IDENTITY_ONLINE_DECODER", False):
            return
        alpha = float(self.params.get("ASSOCIATION_IDENTITY_HINT_SCALE", 0.3))
        if alpha <= 0.0 or not association_data:
            return

        track_log_posts: dict = association_data.get(
            "identity_track_log_posteriors", {}
        )
        det_log_likes: list = association_data.get(
            "identity_detection_log_likelihoods", []
        )
        if not track_log_posts or not det_log_likes:
            return

        max_dist = float(self.params.get("MAX_DISTANCE_THRESHOLD", 1000.0))
        n_tracks, n_dets = cost.shape
        rows = [i for i in range(n_tracks) if track_log_posts.get(i) is not None]
        cols = [
            j
            for j in range(min(n_dets, len(det_log_likes)))
            if det_log_likes[j] is not None
        ]
        if not rows or not cols:
            return

        ta = self.track_arena
        if (
            ta is not None
            and meas_arena is not None
            and len(ta) >= n_tracks
            and len(meas_arena) >= n_dets
        ):
            # Arena mismatch is the ONLY legal predicate. Skipping on `cost >= 1e6`
            # would also skip distance-gated cells that exist on main today and
            # change compute_assignment_confidence's inputs.
            blocked = ta[np.ix_(rows)][:, None] != meas_arena[np.ix_(cols)][None, :]
        else:
            blocked = None

        log_compat = _pairwise_log_compat(
            [track_log_posts[i] for i in rows], [det_log_likes[j] for j in cols]
        )
        # Every step below is done in the cost matrix's own dtype, because the
        # scalar path it replaces added a Python float to a float32 element and
        # compared float32 against float64 thresholds -- NumPy's weak scalar
        # promotion means both happened in float32.  Widening here would change
        # results by an ulp at the cap boundary.
        dt = cost.dtype.type
        addon = (alpha * (-log_compat)).astype(cost.dtype)
        block = cost[np.ix_(rows, cols)]
        summed = block + addon
        # Cap: identity can reorder preferences but must never block a
        # geometrically-valid match by pushing cost above max_dist.
        capped = np.where(summed <= dt(max_dist - 1e-3), summed, dt(max_dist - 1e-3))
        new_block = np.where(block < dt(max_dist), capped, summed)
        if blocked is not None:
            new_block = np.where(blocked, block, new_block)
        cost[np.ix_(rows, cols)] = new_block

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
        meas_arena: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Computes cost matrix. Compatible with Vectorized Kalman Filter.

        ``meas_arena`` (optional) is paired with ``self.track_arena`` (set via
        ``set_track_arena``) to block cross-arena track/detection pairs with
        the same ``1e6`` hard-reject sentinel used for distance gating. Both
        default to ``None``, which reproduces the pre-multi-arena, ungated
        behaviour exactly.
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
                self.worker._emit_warning(
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

        track_arena_arr, meas_arena_arr = _arena_arrays(
            self.track_arena, meas_arena, N, M
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
                track_arena_arr,
                meas_arena_arr,
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
                track_arena_arr,
                meas_arena_arr,
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
                track_arena_arr,
                meas_arena_arr,
            )

        if association_data:
            # Pass the raw `meas_arena` argument, not the numba-kernel-normalized
            # `meas_arena_arr` -- the latter collapses to the `_NO_ARENA` empty
            # sentinel (a real, non-None ndarray) on a length mismatch, which
            # would masquerade as "gating requested but empty" to the overlay's
            # `meas_arena is not None` check below and raise on indexing.
            self._apply_bayesian_identity_cost(cost, association_data, meas_arena)

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

    @staticmethod
    def _solve_established_block(
        est_sorted, det_cols, cost, raw_dist_mat, MAX_DIST, VEL_GATE
    ):
        """Solve ONE Hungarian block and keep the pairs that pass the gates.

        ``det_cols`` is ``None`` for the whole-matrix (single-arena / ungated)
        solve -- that branch runs ``linear_sum_assignment(cost[est, :])``
        exactly as the pre-multi-arena code did, including leaving the column
        index as the numpy integer scipy returned.  Otherwise ``det_cols`` is
        the arena's own detection columns and the solve sees only that
        sub-block.

        The finiteness check runs on this same ``cost_sub`` slice -- the one
        slice that is actually handed to ``linear_sum_assignment`` -- instead
        of a separate whole-row slice computed by the caller and discarded.
        For the ungated path that is the same cells the old pre-check
        covered (identical error semantics, one slice instead of two); for a
        per-arena block it means only the cells that block can actually solve
        are checked -- cross-arena cells the solver never sees no longer need
        to be finite either, matching the "never handed to the solver at all"
        invariant already documented on the caller.
        """
        assignments = []
        assigned_dets = set()
        if det_cols is None:
            cost_sub = cost[est_sorted, :]
        else:
            cost_sub = cost[np.ix_(est_sorted, det_cols)]
        if not np.isfinite(cost_sub).all():
            bad = int(np.size(cost_sub) - np.count_nonzero(np.isfinite(cost_sub)))
            raise ValueError(
                f"assignment submatrix contains non-finite values (bad={bad}, tracks={len(est_sorted)}, dets={cost_sub.shape[1]})"
            )
        rows, cols = linear_sum_assignment(cost_sub)
        for r_idx, c_idx in zip(rows, cols):
            r = est_sorted[r_idx]
            c = c_idx if det_cols is None else det_cols[c_idx]
            if cost[r, c] < MAX_DIST and raw_dist_mat[r, c] < VEL_GATE:
                assignments.append((r, c))
                assigned_dets.add(c)
        return assignments, assigned_dets

    def _assign_established_hungarian(
        self,
        est,
        cost,
        raw_dist_mat,
        MAX_DIST,
        VEL_GATE,
        track_arena=None,
        meas_arena=None,
    ):
        """Phase 1 Hungarian assignment for established tracks.

        ``est`` is always built from ``for i in range(N) if ...`` so it is
        monotonically increasing.  ``linear_sum_assignment(cost[est, :])``
        returns row indices 0..len(est)-1 into the submatrix; ``est[r_idx]``
        maps each back to the original track index.  Making the sort explicit
        here documents and enforces this invariant so that the mapping is safe
        even if the calling code ever builds ``est`` differently.

        Multi-arena: when both arena arrays are supplied (the caller only
        passes them when they are long enough to gate every row and column),
        rows and columns are partitioned by arena id and ONE
        ``linear_sum_assignment`` is run per arena over that arena's own tracks
        and detections.  Cross-arena cells are never handed to the solver at
        all, so the reject sentinels (``1e6`` for arena/spatial blocking,
        ``1e9`` for the raw distance gate) can no longer influence which of an
        arena's tracks keeps its detection: a surplus row parks inside its own
        arena or nowhere.  Independence is structural, not emergent.

        Arena ``-1`` (a detection that fell outside every arena) is treated
        the same way the cost kernel already treats it -- as its own arena id
        under plain equality.  In practice track slots are never ``-1``, so
        that block has no rows, no solve is run for it and those detections
        simply stay unassigned here; they are neither matched nor lost,
        falling through to the later phases (which carry their own arena
        gates) and finally into ``free_dets``.

        ``track_arena``/``meas_arena`` default to ``None``, which takes the
        original whole-matrix solve unchanged -- single-arena runs never enter
        the partitioning code.
        """
        if not est:
            return [], set()
        est_sorted = sorted(est)
        if track_arena is None or meas_arena is None:
            return TrackAssigner._solve_established_block(
                est_sorted, None, cost, raw_dist_mat, MAX_DIST, VEL_GATE
            )

        M = cost.shape[1]
        ta = np.asarray(track_arena, dtype=np.int32)
        ma = np.asarray(meas_arena, dtype=np.int32)[:M]
        est_arr = np.asarray(est_sorted, dtype=np.intp)
        row_arena = ta[est_arr]

        # Vectorized grouping: one stable argsort per side turns "A full
        # scans over N (rows) / M (cols)" into "sort once, then a single
        # np.unique pass for group boundaries" -- O((N+M) log(N+M) + A)
        # instead of O(A*(N+M)). ``kind="stable"`` on both sides preserves
        # each side's original ascending order *within* a group, so the rows
        # and columns handed to each block are in exactly the order the old
        # per-arena list-comprehension / ``np.flatnonzero`` scan produced --
        # required for byte-identical results, not just equivalent ones.
        row_order = np.argsort(row_arena, kind="stable")
        row_arena_sorted = row_arena[row_order]
        row_ids_sorted = est_arr[row_order]
        uniq_row_arenas, row_start, row_count = np.unique(
            row_arena_sorted, return_index=True, return_counts=True
        )

        col_order = np.argsort(ma, kind="stable")
        ma_sorted = ma[col_order]
        uniq_col_arenas, col_start, col_count = np.unique(
            ma_sorted, return_index=True, return_counts=True
        )
        col_index_of_arena = {int(arena): i for i, arena in enumerate(uniq_col_arenas)}

        assignments = []
        assigned_dets = set()
        # Sorted arena order (np.unique already returns ascending) keeps the
        # emitted pair order deterministic and independent of dict/set
        # iteration, matching the previous ``sorted({...})`` loop.
        for i, arena in enumerate(uniq_row_arenas):
            j = col_index_of_arena.get(int(arena))
            if j is None:
                continue
            r0 = int(row_start[i])
            rc = int(row_count[i])
            block_rows = row_ids_sorted[r0 : r0 + rc]
            c0 = int(col_start[j])
            cc = int(col_count[j])
            block_cols = col_order[c0 : c0 + cc]
            pairs, dets = TrackAssigner._solve_established_block(
                block_rows, block_cols, cost, raw_dist_mat, MAX_DIST, VEL_GATE
            )
            assignments.extend(pairs)
            assigned_dets.update(dets)
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
        cost: np.ndarray,
        N: int,
        meas: list,
        track_states: list,
        tracking_continuity: list,
        kf_manager,
        spatial_candidates: dict | None = None,
        association_data: dict | None = None,
        committed_slot_identities: dict | None = None,
        missed_frames: list | None = None,
        _lost=None,
        _M=None,
        _MAX_DIST=None,
        _assigned_dets=None,
        meas_arena: np.ndarray | None = None,
    ) -> tuple:
        """Phase 3: respawn lost tracks with unassigned detections.

        Returns ``(rows, cols, identity_rejoin_pairs)`` where
        ``identity_rejoin_pairs`` is a list of ``(slot_index, det_index)``
        tuples matched via identity evidence for committed-lost slots.
        """
        p = self.params

        lost = (
            list(_lost)
            if _lost is not None
            else [i for i in range(N) if track_states[i] == "lost"]
        )
        M = _M if _M is not None else cost.shape[1]
        MAX_DIST = _MAX_DIST if _MAX_DIST is not None else p["MAX_DISTANCE_THRESHOLD"]
        assigned_dets: set = _assigned_dets if _assigned_dets is not None else set()

        ta = self.track_arena
        gate = (
            ta is not None
            and meas_arena is not None
            and len(ta) >= N
            and len(meas_arena) >= M
        )

        # Split lost slots into committed vs. uncommitted
        if committed_slot_identities:
            committed_lost = [s for s in lost if s in committed_slot_identities]
            uncommitted_lost = [s for s in lost if s not in committed_slot_identities]
        else:
            committed_lost = []
            uncommitted_lost = lost

        # Identity-only rejoin for committed lost slots
        identity_rejoin_pairs: list = []
        identity_claimed_dets: set = set()
        if committed_lost and association_data:
            det_log_likes = association_data.get(
                "identity_detection_log_likelihoods", []
            )
            track_log_posts = association_data.get("identity_track_log_posteriors", {})
            rejoin_threshold = float(p.get("IDENTITY_REJOIN_THRESHOLD", 0.5))
            log_threshold = np.log(max(rejoin_threshold, 1e-10))

            # Motion-budget gate: short occlusions can only rejoin nearby; long
            # occlusions retain long-range re-ID.  If missed_frames isn't
            # supplied (e.g. legacy callers / unit tests) the gate is disabled.
            body_size = float(p.get("REFERENCE_BODY_SIZE", 20.0)) * float(
                p.get("RESIZE_FACTOR", 1.0)
            )
            v_max_per_frame = (
                float(p.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)) * body_size
            )
            budget_safety = float(p.get("IDENTITY_REJOIN_VELOCITY_BUDGET", 1.5))
            _floor_raw = p.get("IDENTITY_REJOIN_DIST_FLOOR", None)
            budget_floor = 2.0 * body_size if _floor_raw is None else float(_floor_raw)

            def _within_budget(slot_idx: int, det_xy: np.ndarray) -> bool:
                if missed_frames is None:
                    return True
                last_pos = kf_manager.X[slot_idx, :2]
                dist = float(np.linalg.norm(det_xy - last_pos))
                lost_n = int(missed_frames[slot_idx])
                budget = max(budget_floor, lost_n * v_max_per_frame * budget_safety)
                return dist <= budget

            # Build best (score, det_idx) for each committed slot.  The score is
            # a pure function of (slot, det), so the whole candidate block is
            # scored at once; the per-pair motion-budget gate is applied after,
            # only to pairs that clear the threshold.
            slot_best: dict = {}
            cand_slots = [
                s for s in committed_lost if track_log_posts.get(s) is not None
            ]
            cand_dets = [
                j
                for j, log_like in enumerate(det_log_likes)
                if j not in assigned_dets and log_like is not None
            ]
            if cand_slots and cand_dets:
                scores = _pairwise_log_compat(
                    [
                        np.asarray(track_log_posts[s], dtype=np.float64)
                        for s in cand_slots
                    ],
                    [np.asarray(det_log_likes[j], dtype=np.float64) for j in cand_dets],
                )
                for si, slot in enumerate(cand_slots):
                    row = scores[si]
                    for dj in np.flatnonzero(row > log_threshold):
                        j = cand_dets[dj]
                        if gate and meas_arena[j] != ta[slot]:
                            continue
                        det_xy = np.asarray(meas[j][:2], dtype=np.float64)
                        if not _within_budget(slot, det_xy):
                            continue
                        score = float(row[dj])
                        if slot not in slot_best or score > slot_best[slot][0]:
                            slot_best[slot] = (score, j)

            # Resolve conflicts: highest score wins when two slots want same det
            det_best: dict = {}
            for slot, (score, det_j) in slot_best.items():
                if det_j not in det_best or score > det_best[det_j][0]:
                    det_best[det_j] = (score, slot)

            for det_j, (score, slot) in det_best.items():
                identity_rejoin_pairs.append((slot, det_j))
                identity_claimed_dets.add(det_j)

            # Committed-lost slots that got no identity match fall back to the
            # proximity path so they are not permanently stranded.
            identity_rejoined_slots = {s for s, _ in identity_rejoin_pairs}
            for slot in committed_lost:
                if slot not in identity_rejoined_slots:
                    uncommitted_lost.append(slot)

        # Proximity-based respawn for uncommitted lost slots.
        # No proximity-to-active guard: in dense colonies every detection is
        # near some active track, so any such guard would silently block all
        # phase-3 respawns.  The MAX_DIST ceiling on best_c_val below is the
        # only gate needed — if the detection is genuinely close to an active
        # track it will have been matched in phases 1-2 and won't appear here.
        unassigned = [
            j
            for j in range(M)
            if j not in assigned_dets and j not in identity_claimed_dets
        ]
        rows: list = []
        cols: list = []
        remaining_uncommitted = list(uncommitted_lost)
        for c in unassigned:
            if not remaining_uncommitted:
                break
            best_r, best_c_val = None, 1e6
            for r in remaining_uncommitted:
                if gate and meas_arena[c] != ta[r]:
                    continue
                last_pos = kf_manager.X[r, :2]
                dist = float(np.linalg.norm(meas[c][:2] - last_pos))
                if dist < best_c_val:
                    best_c_val, best_r = dist, r
            if best_r is not None and best_c_val < MAX_DIST:
                rows.append(best_r)
                cols.append(c)
                assigned_dets.add(c)
                remaining_uncommitted.remove(best_r)

        return rows, cols, identity_rejoin_pairs

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
        committed_slot_identities: Dict[int, str] | None = None,
        missed_frames: list | None = None,
        meas_arena: np.ndarray | None = None,
    ) -> object:
        """
        Drop-in replacement for track assignment logic.
        Compatible with kf_manager.X state access.

        Returns ``(rows, cols, free_dets, identity_rejoin_pairs)`` where
        ``identity_rejoin_pairs`` is a list of ``(slot_index, det_index)``
        tuples from the identity-only rejoin path for committed-lost slots.
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
                # Only hand the arena arrays down when they are long enough to
                # label EVERY row and column -- the same fail-open ``>=``
                # predicate the identity overlay uses (line ~1111). NOTE:
                # `_arena_arrays` (the cost-kernel gate) requires an exact
                # ``==`` length match instead and falls back to "no gating"
                # on any mismatch, short or long -- do not cite it as sharing
                # this predicate. A short array here would silently mislabel
                # slots; `None` keeps the original whole-matrix solve, which
                # is what single-arena runs take.
                _ta = self.track_arena
                _arena_gated = (
                    _ta is not None
                    and meas_arena is not None
                    and len(_ta) >= N
                    and len(meas_arena) >= M
                )
                ph1, ph1_dets = self._assign_established_hungarian(
                    est,
                    cost,
                    raw_dist_mat,
                    MAX_DIST,
                    VEL_GATE,
                    track_arena=_ta if _arena_gated else None,
                    meas_arena=meas_arena if _arena_gated else None,
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

        # Phase 3: Respawn Lost Tracks (split-path: committed vs. uncommitted)
        ph3_rows, ph3_cols, identity_rejoin_pairs = self._assign_respawn(
            cost=cost,
            N=N,
            meas=meas,
            track_states=track_states,
            tracking_continuity=tracking_continuity,
            kf_manager=kf_manager,
            spatial_candidates=spatial_candidates,
            association_data=association_data,
            committed_slot_identities=committed_slot_identities,
            missed_frames=missed_frames,
            _lost=lost,
            _M=M,
            _MAX_DIST=MAX_DIST,
            _assigned_dets=assigned_dets,
            meas_arena=meas_arena,
        )
        all_assignments.extend(zip(ph3_rows, ph3_cols))

        if not all_assignments:
            return [], [], list(range(M)), identity_rejoin_pairs

        final_r, final_c = zip(*all_assignments)
        free_dets = list(set(range(M)) - set(final_c))
        return list(final_r), list(final_c), free_dets, identity_rejoin_pairs

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
        track_arena=_NO_ARENA,
        meas_arena=_NO_ARENA,
    ):
        """Python fallback for spatial optimization."""
        cost = np.full((N, M), 1e6, dtype=np.float32)
        Wp, Wo, Wa, Wasp = (
            p["W_POSITION"],
            p["W_ORIENTATION"],
            p["W_AREA"],
            p["W_ASPECT"],
        )
        ta, ma = track_arena, meas_arena
        gate_arenas = ta.shape[0] == N and ma.shape[0] == M

        for r, det_indices in candidates.items():
            inv_S = S_inv[r, :2, :2]
            arena_r = ta[r] if gate_arenas else None
            for c in det_indices:
                if arena_r is not None and ma[c] != arena_r:
                    continue  # stays at the 1e6 initialization value
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
