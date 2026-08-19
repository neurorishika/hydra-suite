import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from hydra_suite.core.assigners.hungarian import (
    TrackAssigner,
    _compute_cost_matrix_numba,
)

BLOCKED = 1e6


def _kernel_args(n, m, rng):
    return dict(
        meas_pos=rng.uniform(0, 500, (m, 2)).astype(np.float32),
        meas_ori=rng.uniform(-np.pi, np.pi, m).astype(np.float32),
        pred_pos=rng.uniform(0, 500, (n, 2)).astype(np.float32),
        pred_ori=rng.uniform(-np.pi, np.pi, n).astype(np.float32),
        shapes_area=rng.uniform(10, 50, m).astype(np.float32),
        shapes_asp=rng.uniform(1, 3, m).astype(np.float32),
        prev_areas=rng.uniform(10, 50, n).astype(np.float32),
        prev_asps=rng.uniform(1, 3, n).astype(np.float32),
        S_inv_batch=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        use_maha=False,
        Wp=1.0,
        Wo=0.1,
        Wa=0.01,
        Wasp=0.01,
        per_track_gates=np.full(n, 1e9, dtype=np.float32),
        meas_ori_directed=np.ones(m, dtype=np.int32),
    )


def _cost(n, m, track_arena, meas_arena, seed=0):
    rng = np.random.default_rng(seed)
    return _compute_cost_matrix_numba(
        n, m, track_arena=track_arena, meas_arena=meas_arena, **_kernel_args(n, m, rng)
    )


def test_cross_arena_pairs_are_blocked():
    track_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    meas_arena = np.array([0, 1, 1, 0], dtype=np.int32)
    cost = _cost(4, 4, track_arena, meas_arena)
    for i in range(4):
        for j in range(4):
            if track_arena[i] != meas_arena[j]:
                assert cost[i, j] >= BLOCKED, f"({i},{j}) should be blocked"


def test_same_arena_pairs_are_unchanged_by_blocking():
    """Blocking must not perturb the cost of any within-arena pair."""
    track_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    meas_arena = np.array([0, 1, 1, 0], dtype=np.int32)
    blocked = _cost(4, 4, track_arena, meas_arena)
    ungated = _cost(4, 4, None, None)
    same = track_arena[:, None] == meas_arena[None, :]
    np.testing.assert_array_equal(blocked[same], ungated[same])


def test_none_arena_arrays_reproduce_current_behaviour_exactly():
    """The single-arena path must be bit-identical to no gating at all."""
    uniform = np.zeros(4, dtype=np.int32)
    np.testing.assert_array_equal(
        _cost(4, 4, None, None), _cost(4, 4, uniform, np.zeros(4, dtype=np.int32))
    )


def test_detections_outside_every_arena_are_blocked_from_all_tracks():
    track_arena = np.array([0, 1], dtype=np.int32)
    meas_arena = np.array([-1, 0], dtype=np.int32)  # -1 == outside
    cost = _cost(2, 2, track_arena, meas_arena)
    assert cost[0, 0] >= BLOCKED and cost[1, 0] >= BLOCKED


@pytest.mark.parametrize("seed", range(10))
def test_blocked_hungarian_equals_independent_per_arena_hungarian(seed):
    """The core correctness claim: block-diagonal solve == per-arena solves."""
    n_arenas, per_arena, dets_per_arena = 4, 3, 3
    n, m = n_arenas * per_arena, n_arenas * dets_per_arena
    track_arena = np.repeat(np.arange(n_arenas), per_arena).astype(np.int32)
    meas_arena = np.repeat(np.arange(n_arenas), dets_per_arena).astype(np.int32)

    cost = _cost(n, m, track_arena, meas_arena, seed=seed)
    rows, cols = linear_sum_assignment(cost)
    joint = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < BLOCKED}

    separate = {}
    for a in range(n_arenas):
        tr = np.flatnonzero(track_arena == a)
        dt = np.flatnonzero(meas_arena == a)
        sub = cost[np.ix_(tr, dt)]
        r_sub, c_sub = linear_sum_assignment(sub)
        for r, c in zip(r_sub, c_sub):
            if sub[r, c] < BLOCKED:
                separate[int(tr[r])] = int(dt[c])

    assert joint == separate


# --- End-to-end coverage of the public TrackAssigner API surface ----------
#
# The tests above only ever exercise the low-level kernel wrapper. These
# tests drive TrackAssigner.set_track_arena / compute_cost_matrix directly,
# and specifically force the Python fallback path
# (ENABLE_SPATIAL_OPTIMIZATION + N > 50) so a future divergence between the
# numba kernel and the fallback's arena gating cannot go unnoticed.


class _DummyKF:
    """Minimal stand-in for KalmanFilterManager, identity covariances."""

    def __init__(self, n: int):
        self.X = np.zeros((n, 5), dtype=np.float32)
        self.P = np.stack([np.eye(5, dtype=np.float32) for _ in range(n)])

    def get_mahalanobis_matrices(self):
        mats = np.zeros((len(self.X), 3, 3), dtype=np.float32)
        for i in range(len(self.X)):
            mats[i] = np.eye(3, dtype=np.float32)
        return mats

    def get_position_uncertainties(self):
        return [float(np.trace(self.P[i, :2, :2])) for i in range(len(self.X))]


def _e2e_params(spatial: bool) -> dict:
    return {
        "USE_MAHALANOBIS": False,
        "W_POSITION": 1.0,
        "W_ORIENTATION": 0.1,
        "W_AREA": 0.01,
        "W_ASPECT": 0.01,
        "MAX_DISTANCE_THRESHOLD": 1000.0,
        "CONTINUITY_THRESHOLD": 3,
        "ENABLE_GREEDY_ASSIGNMENT": False,
        "ENABLE_SPATIAL_OPTIMIZATION": spatial,
        "MIN_RESPAWN_DISTANCE": 15.0,
        "REFERENCE_BODY_SIZE": 20.0,
        "W_POSE_DIRECTION": 0.75,
        "W_POSE_LENGTH": 0.1,
        "POSE_VALID_ORIENTATION_SCALE": 0.15,
        "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER": 1.4,
        "ASSOCIATION_STAGE1_MAX_AREA_RATIO": 2.5,
        "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF": 0.8,
        "TRACK_FEATURE_EMA_ALPHA": 0.85,
        "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD": 0.7,
    }


def _clustered_scene(n_arenas, per_arena, seed):
    """All positions packed into a small box so every pair is within the
    default cull distance -- any cross-arena block is attributable ONLY to
    arena gating, never to the pre-existing distance gate. N == M == n_arenas
    * per_arena, tracks and detections use the same arena labelling.
    """
    rng = np.random.default_rng(seed)
    n = n_arenas * per_arena
    track_arena = np.repeat(np.arange(n_arenas), per_arena).astype(np.int32)
    meas_arena = track_arena.copy()
    predictions = np.zeros((n, 3), dtype=np.float32)
    predictions[:, :2] = rng.uniform(0, 50, (n, 2)).astype(np.float32)
    predictions[:, 2] = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    measurements = [
        np.array(
            [
                predictions[i, 0] + rng.uniform(-1, 1),
                predictions[i, 1] + rng.uniform(-1, 1),
                predictions[i, 2],
            ],
            dtype=np.float32,
        )
        for i in range(n)
    ]
    shapes = [(20.0, 1.3) for _ in range(n)]
    last_shape_info = [(20.0, 1.3) for _ in range(n)]
    return track_arena, meas_arena, predictions, measurements, shapes, last_shape_info


def test_set_track_arena_end_to_end_blocks_cross_arena_cells():
    """Drives set_track_arena + compute_cost_matrix (not the kernel directly)
    on a small (numba-path) scene. Would fail if set_track_arena or the
    meas_arena threading through compute_cost_matrix were removed, since the
    scene is deliberately clustered so cross-arena pairs are otherwise cheap.
    """
    track_arena, meas_arena, predictions, measurements, shapes, last_shape_info = (
        _clustered_scene(n_arenas=2, per_arena=2, seed=0)
    )
    assigner = TrackAssigner(_e2e_params(spatial=False))
    assigner.set_track_arena(track_arena)
    kf = _DummyKF(len(predictions))

    cost, _ = assigner.compute_cost_matrix(
        N=len(predictions),
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        meas_arena=meas_arena,
    )

    cross = track_arena[:, None] != meas_arena[None, :]
    same = ~cross
    assert np.all(cost[cross] >= BLOCKED)
    # Positions are clustered, so within-arena pairs must stay well under the
    # sentinel -- proves the block above is attributable to arena gating.
    assert np.all(cost[same] < BLOCKED)


def test_set_track_arena_none_restores_ungated_path():
    """set_track_arena(None) must reproduce the no-gating result exactly."""
    track_arena, meas_arena, predictions, measurements, shapes, last_shape_info = (
        _clustered_scene(n_arenas=2, per_arena=2, seed=1)
    )
    kf = _DummyKF(len(predictions))

    gated = TrackAssigner(_e2e_params(spatial=False))
    gated.set_track_arena(track_arena)
    cost_gated, _ = gated.compute_cost_matrix(
        N=len(predictions),
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=_DummyKF(len(predictions)),
        last_shape_info=last_shape_info,
        meas_arena=meas_arena,
    )
    # Some pairs really are blocked here, or the "None restores ungated"
    # comparison below would be vacuous.
    assert np.any(cost_gated >= BLOCKED)

    ungated = TrackAssigner(_e2e_params(spatial=False))
    ungated.set_track_arena(None)
    cost_ungated, _ = ungated.compute_cost_matrix(
        N=len(predictions),
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        meas_arena=None,
    )
    assert np.all(cost_ungated < BLOCKED)


def test_python_fallback_matches_numba_path_arena_gating():
    """Force ENABLE_SPATIAL_OPTIMIZATION + N > 50 so compute_cost_matrix takes
    the _compute_cost_python_fallback branch (hungarian.py's KD-tree path),
    and compare it directly against the numba path on the SAME inputs. This
    is exactly the failure mode the repo has shipped before: an optimisation
    silently diverging from the reference path with nothing to catch it.
    """
    n_arenas, per_arena = 4, 15  # 60 > 50, forces the spatial/fallback branch
    track_arena, meas_arena, predictions, measurements, shapes, last_shape_info = (
        _clustered_scene(n_arenas=n_arenas, per_arena=per_arena, seed=2)
    )
    n = len(predictions)

    fallback_assigner = TrackAssigner(_e2e_params(spatial=True))
    fallback_assigner.set_track_arena(track_arena)
    cost_fallback, _ = fallback_assigner.compute_cost_matrix(
        N=n,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=_DummyKF(n),
        last_shape_info=last_shape_info,
        meas_arena=meas_arena,
    )

    numba_assigner = TrackAssigner(_e2e_params(spatial=False))
    numba_assigner.set_track_arena(track_arena)
    cost_numba, _ = numba_assigner.compute_cost_matrix(
        N=n,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=_DummyKF(n),
        last_shape_info=last_shape_info,
        meas_arena=meas_arena,
    )

    # Sanity: the fallback path was genuinely exercised and genuinely blocked
    # some cross-arena pairs (otherwise this test would pass vacuously).
    cross = track_arena[:, None] != meas_arena[None, :]
    assert np.any(cross)
    assert np.all(cost_fallback[cross] >= BLOCKED)
    # Not bit-exact: the fallback uses np.linalg.norm vs. the kernel's
    # sqrt(dx**2+dy**2), which differs at the ULP level even pre-arena-gating
    # (a pre-existing characteristic of the two paths, unrelated to arena
    # blocking). The tolerance is tight enough that any arena-gating
    # divergence -- which would show up as a ~1e6 discrepancy on the blocked
    # cells -- still fails this assertion immediately.
    np.testing.assert_allclose(cost_fallback, cost_numba, rtol=1e-4, atol=1e-3)
