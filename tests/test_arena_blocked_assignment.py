import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from hydra_suite.core.assigners.hungarian import _compute_cost_matrix_numba

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
