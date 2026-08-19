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


# --- Task 4: identity-cost overlay + respawn arena gating -----------------


def test_respawn_never_crosses_arenas():
    """Proximity respawn must not pull an arena-0 slot onto an arena-1 detection."""
    from hydra_suite.core.assigners.hungarian import TrackAssigner

    class _KF:
        # slot 0 in arena 0 at (390,10) -- nearest to the detection; slot 1 in
        # arena 1 at (410,10). Slot 0 must win on pure distance (5px vs 15px)
        # unless the arena gate blocks it -- so this fixture can actually
        # distinguish "gated" from "ungated" behaviourally, not just via a
        # TypeError on a missing kwarg.
        X = np.array([[390.0, 10.0, 0.0, 0.0, 0.0], [410.0, 10.0, 0.0, 0.0, 0.0]])

    params = {
        "MAX_DISTANCE_THRESHOLD": 1000.0,
        "KALMAN_MATURITY_AGE": 10,
        "W_POSITION": 1.0,
        "W_ORIENTATION": 0.1,
        "W_AREA": 0.01,
        "W_ASPECT": 0.01,
        "USE_MAHALANOBIS": False,
    }
    assigner = TrackAssigner(params)
    assigner.set_track_arena(np.array([0, 1], dtype=np.int32))

    # One detection, sitting in arena 1, 5px from slot 0 (arena 0, nearer) and
    # 15px from slot 1 (arena 1, farther) -- only the arena gate can produce
    # (1, 0); an ungated nearest-neighbour loop would produce (0, 0).
    meas = [np.array([395.0, 10.0, 0.0])]
    cost = np.full((2, 1), 1e6, dtype=np.float32)
    rows, cols, _ = assigner._assign_respawn(
        cost=cost,
        N=2,
        meas=meas,
        track_states=["lost", "lost"],
        tracking_continuity=[0, 0],
        kf_manager=_KF(),
        spatial_candidates=None,
        association_data=None,
        committed_slot_identities=None,
        missed_frames=[5, 5],
        _lost=[0, 1],
        _M=1,
        _MAX_DIST=1000.0,
        _assigned_dets=set(),
        meas_arena=np.array([1], dtype=np.int32),
    )
    assert 0 not in rows, "arena-0 slot must not respawn on an arena-1 detection"
    assert list(zip(rows, cols)) == [(1, 0)], "arena-1 slot should take the detection"


def test_identity_rejoin_never_crosses_arenas():
    """The committed-lost identity rejoin scores a dense block; the arena test must
    be applied where the detection index is known, not inside the budget check."""
    from hydra_suite.core.assigners.hungarian import TrackAssigner

    class _KF:
        X = np.array([[10.0, 10.0, 0.0, 0.0, 0.0], [410.0, 10.0, 0.0, 0.0, 0.0]])

    params = {
        "MAX_DISTANCE_THRESHOLD": 1000.0,
        "IDENTITY_REJOIN_THRESHOLD": 0.1,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
    }
    assigner = TrackAssigner(params)
    assigner.set_track_arena(np.array([0, 1], dtype=np.int32))

    # Slot 0 (arena 0) is certain of label 0; the only detection is also label 0
    # but sits in arena 1.  Without gating it would be rejoined across arenas.
    certain = np.log(np.array([0.99, 0.01]))
    association = {
        "identity_track_log_posteriors": {0: certain},
        "identity_detection_log_likelihoods": [certain],
    }
    cost = np.full((2, 1), 1e6, dtype=np.float32)
    _rows, _cols, rejoin = assigner._assign_respawn(
        cost=cost,
        N=2,
        meas=[np.array([400.0, 10.0, 0.0])],
        track_states=["lost", "lost"],
        tracking_continuity=[0, 0],
        kf_manager=_KF(),
        spatial_candidates=None,
        association_data=association,
        committed_slot_identities={0: "antA"},
        missed_frames=None,
        _lost=[0, 1],
        _M=1,
        _MAX_DIST=1000.0,
        _assigned_dets=set(),
        meas_arena=np.array([1], dtype=np.int32),
    )
    assert rejoin == [], "committed arena-0 slot must not rejoin an arena-1 detection"


def test_oversubscribed_arena_pairing_invariant_to_other_arenas():
    """Pins Hole 1 at the compute_cost_matrix + overlay level.

    Arena 0 holds two tracks (T0, T2) competing for one detection (D0); arena 1
    holds one track (T1) and two detections (D1, D2). T2 sits exactly on D0
    while T0 is 0.2 px off, so the correct (and single-arena-equivalent)
    winner of D0 is T2. The identity overlay is rigged so that T0's fallback
    onto a blocked (cross-arena) cell is *expensive* and T2's is *cheap* --
    if the overlay's addon leaks onto blocked cells (the un-gated bug), the
    solver's global optimization flips the decision and awards D0 to T0
    instead, exactly the counter-example in the task brief. With the arena
    gate applied to the overlay, blocked cells stay a flat 1e6 regardless of
    identity, and the within-arena outcome must match what a standalone,
    single-arena solve of {T0, T2} vs {D0} would produce.
    """
    from hydra_suite.core.assigners.hungarian import TrackAssigner

    class _KF:
        def __init__(self, X):
            self.X = np.asarray(X, dtype=np.float32)
            self.P = np.stack([np.eye(5, dtype=np.float32) for _ in range(len(self.X))])

        def get_mahalanobis_matrices(self):
            return np.stack([np.eye(3, dtype=np.float32) for _ in range(len(self.X))])

        def get_position_uncertainties(self):
            return [float(np.trace(self.P[i, :2, :2])) for i in range(len(self.X))]

    params = _e2e_params(spatial=False)
    params["ENABLE_IDENTITY_ONLINE_DECODER"] = True
    params["ASSOCIATION_IDENTITY_HINT_SCALE"] = 1.0

    # T0=arena0 (slot 0), T1=arena1 (slot 1), T2=arena0 (slot 2).
    track_arena = np.array([0, 1, 0], dtype=np.int32)
    # D0=arena0 (det 0), D1=arena1 (det 1), D2=arena1 (det 2).
    meas_arena = np.array([0, 1, 1], dtype=np.int32)

    predictions = np.array(
        [
            [10.2, 10.0, 0.0],  # T0: 0.2px from D0
            [410.0, 10.0, 0.0],  # T1
            [10.0, 10.0, 0.0],  # T2: exactly on D0 -> the natural winner
        ],
        dtype=np.float32,
    )
    measurements = [
        np.array([10.0, 10.0, 0.0], dtype=np.float32),  # D0 (arena 0)
        np.array([410.0, 10.0, 0.0], dtype=np.float32),  # D1 (arena 1)
        np.array([415.0, 10.0, 0.0], dtype=np.float32),  # D2 (arena 1)
    ]
    shapes = [(20.0, 1.3)] * 3
    last_shape_info = [(20.0, 1.3)] * 3

    class_a = np.log(np.array([0.99, 0.01]))
    class_b = np.log(np.array([0.01, 0.99]))
    association_data = {
        # T0 (slot 0) believes class B; T2 (slot 2) believes class A. T1
        # (slot 1) is deliberately left out of the posterior dict.
        "identity_track_log_posteriors": {0: class_b, 2: class_a},
        # D0 (det 0) is deliberately left with no likelihood -- it must never
        # be touched by the overlay, so the D0 column stays pure distance
        # cost and the "natural winner" comparison is uncontaminated. D1/D2
        # both carry class-A evidence, matching T2 (cheap fallback) and
        # mismatching T0 (expensive fallback).
        "identity_detection_log_likelihoods": [None, class_a, class_a],
    }

    assigner = TrackAssigner(params)
    assigner.set_track_arena(track_arena)
    kf = _KF(
        np.pad(predictions, ((0, 0), (0, 2))),  # X = [x, y, theta, vx, vy]
    )
    cost, _ = assigner.compute_cost_matrix(
        N=3,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        association_data=association_data,
        meas_arena=meas_arena,
    )

    # D0 must stay unblocked and the addon must genuinely be biased the way
    # the docstring claims, or this test would pass vacuously.
    assert cost[0, 0] < BLOCKED and cost[2, 0] < BLOCKED
    assert cost[0, 0] > cost[2, 0]  # T0 really is farther from D0 than T2

    rows, cols = linear_sum_assignment(cost)
    combined_pairs = {
        int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < BLOCKED
    }
    assert combined_pairs.get(2) == 0, (
        "T2 (the geometrically closer arena-0 track) must win D0 even with "
        "arena 1 present -- a value leaking onto blocked cells would flip "
        "this to T0"
    )
    assert combined_pairs.get(0) != 0

    # Standalone control: solve arena 0 in isolation (no arena 1 at all) with
    # the identical identity data restricted to slots {T0, T2} and det {D0}.
    # This is the "no other arenas present" baseline the combined run must match.
    solo_params = _e2e_params(spatial=False)
    solo_params["ENABLE_IDENTITY_ONLINE_DECODER"] = True
    solo_params["ASSOCIATION_IDENTITY_HINT_SCALE"] = 1.0
    solo_assigner = TrackAssigner(solo_params)
    solo_predictions = predictions[[0, 2]]
    solo_kf = _KF(np.pad(solo_predictions, ((0, 0), (0, 2))))
    solo_association = {
        "identity_track_log_posteriors": {0: class_b, 1: class_a},
        "identity_detection_log_likelihoods": [None],
    }
    solo_cost, _ = solo_assigner.compute_cost_matrix(
        N=2,
        measurements=[measurements[0]],
        predictions=solo_predictions,
        shapes=shapes[:2],
        kf_manager=solo_kf,
        last_shape_info=last_shape_info[:2],
        association_data=solo_association,
    )
    solo_rows, solo_cols = linear_sum_assignment(solo_cost)
    solo_pairs = {int(r): int(c) for r, c in zip(solo_rows, solo_cols)}
    # Slot 1 in the solo problem is T2 -- must win D0, matching the combined run.
    assert solo_pairs.get(1) == 0
