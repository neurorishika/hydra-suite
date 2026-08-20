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


# --- Task 11: the solve itself must decompose, not just the accepted pairs ---


def _unbalanced_scene():
    """Geometry for the parking counter-example, in one place.

    Arena 0 holds three established tracks A/B/C but only two detections of
    its own, so in a square solve exactly one arena-0 row must park on a
    foreign column. Arena 1 holds three tracks and FOUR detections, which is
    what makes the joint problem square (6 established rows, 6 columns) and
    therefore forces the parking -- with fewer columns than rows the solver
    could just leave a row out for free and the coupling would never appear.

    The foreign column left over for arena 0 is the arena-1 detection at
    x=-850. It is 850 px from A (inside MAX_DIST=900, so its cell keeps the
    ``1e6`` arena-block sentinel) but 950/1050 px from B/C (outside, so those
    cells are overwritten with the ``1e9`` distance sentinel). Parking A is
    therefore a thousand times cheaper than parking C, and a joint solve buys
    that discount by taking D0/D1 away from A and giving them to B and C.
    Nothing about arena 0's own geometry changed; arena 1's detections decided
    it. That is the leak.
    """
    # slots 0,1,2 -> arena 0 (A, B, C); slots 3,4,5 -> arena 1
    track_xy = [
        (0.0, 0.0),  # A  -- owns D0
        (100.0, 0.0),  # B  -- owns D1
        (200.0, 0.0),  # C  -- surplus, no detection of its own
        (-1000.0, 0.0),
        (-1100.0, 0.0),
        (-1200.0, 0.0),
    ]
    track_arena = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    det_xy = [
        (5.0, 0.0),  # D0 (arena 0), next to A
        (105.0, 0.0),  # D1 (arena 0), next to B
        (-995.0, 0.0),
        (-1105.0, 0.0),
        (-1205.0, 0.0),
        (-850.0, 0.0),  # arena 1's spare -- the cheap parking spot for A
    ]
    meas_arena = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    return track_xy, track_arena, det_xy, meas_arena


def _unbalanced_params() -> dict:
    params = _e2e_params(spatial=False)
    # 900 is load-bearing: it is the per-track distance gate, and it is what
    # splits the x=-850 column into "1e6 for A" and "1e9 for B and C".
    params["MAX_DISTANCE_THRESHOLD"] = 900.0
    params["KALMAN_MATURITY_AGE"] = 10
    params["KALMAN_MAX_VELOCITY_MULTIPLIER"] = 100.0  # VEL_GATE = 2000 px
    params["REFERENCE_BODY_SIZE"] = 20.0
    return params


def _run_assignment(
    track_xy, det_xy, track_arena, meas_arena, tracking_continuity=None
):
    """Drive the real cost matrix + assign_tracks for one scene.

    Returns ``{track_index: (det_x, det_y)}`` -- pairs are reported by
    detection POSITION, not column index, so the combined and solo runs stay
    comparable even though their column numbering differs.

    ``tracking_continuity`` defaults to "every track established" (100 for
    all slots); callers that need a non-contiguous ``est`` (some slot below
    ``KALMAN_MATURITY_AGE``, so excluded from the established phase) pass
    their own list.
    """
    n = len(track_xy)
    m = len(det_xy)
    predictions = np.zeros((n, 3), dtype=np.float32)
    predictions[:, :2] = np.array(track_xy, dtype=np.float32)
    measurements = [np.array([x, y, 0.0], dtype=np.float32) for (x, y) in det_xy]
    kf = _DummyKF(n)
    kf.X[:, :2] = np.array(track_xy, dtype=np.float32)
    if tracking_continuity is None:
        tracking_continuity = [100] * n

    assigner = TrackAssigner(_unbalanced_params())
    assigner.set_track_arena(track_arena)
    cost, _ = assigner.compute_cost_matrix(
        N=n,
        measurements=measurements,
        predictions=predictions,
        shapes=[(20.0, 1.3)] * m,
        kf_manager=kf,
        last_shape_info=[(20.0, 1.3)] * n,
        meas_arena=meas_arena,
    )
    rows, cols, free_dets, _ = assigner.assign_tracks(
        cost=cost,
        N=n,
        M=m,
        meas=measurements,
        track_states=["active"] * n,
        tracking_continuity=tracking_continuity,
        kf_manager=kf,
        meas_arena=meas_arena,
    )
    pairs = {int(r): det_xy[int(c)] for r, c in zip(rows, cols)}
    return pairs, cost, sorted(int(d) for d in free_dets)


def test_unbalanced_arena_pairing_is_invariant_to_other_arenas():
    """An arena with more established tracks than detections must pair the
    same way whether or not other arenas exist.

    This is the property the ``1e6``/``1e9`` sentinels could not deliver and
    that per-arena sub-block solving delivers structurally. The balanced
    3-tracks/3-detections test above cannot exhibit it: with a perfect
    within-arena matching available, nothing has to park.
    """
    track_xy, track_arena, det_xy, meas_arena = _unbalanced_scene()

    combined, cost, free_dets = _run_assignment(
        track_xy, det_xy, track_arena, meas_arena
    )

    # --- the scene really is the counter-example, not a vacuous pass -------
    # Arena 0 is genuinely oversubscribed, and the whole problem is square so
    # some arena-0 row is forced onto a foreign column in a joint solve.
    assert int((track_arena == 0).sum()) == 3
    assert int((meas_arena == 0).sum()) == 2
    assert len(track_xy) == len(det_xy)
    # The two sentinels really are both present on arena 0's foreign cells.
    assert cost[0, 5] == pytest.approx(1e6), "A's parking cell lost its 1e6"
    assert cost[1, 5] == pytest.approx(1e9), "B's parking cell lost its 1e9"
    assert cost[2, 5] == pytest.approx(1e9), "C's parking cell lost its 1e9"

    # --- the solo run: arena 0 alone, single-arena (ungated) path ----------
    solo, _, _ = _run_assignment(track_xy[:3], det_xy[:2], None, None)
    assert solo == {0: det_xy[0], 1: det_xy[1]}, (
        "control failed: with no other arena present A and B keep their own "
        "detections and C goes unassigned"
    )

    # --- the property ------------------------------------------------------
    arena0 = {slot: det for slot, det in combined.items() if slot < 3}
    assert arena0 == solo, (
        "arena 0's pairing changed because arena 1 exists: "
        f"{arena0} with arena 1 present vs {solo} alone"
    )
    # No cross-arena pair was accepted anywhere, in either direction.
    for slot, det in combined.items():
        assert (det in det_xy[:2]) == (track_arena[slot] == 0)
    # Arena 1's spare detection is neither matched nor lost.
    assert det_xy.index((-850.0, 0.0)) in free_dets
    # Arena 1 is not vacuously untested: its own three tracks must pair to
    # their own three (non-spare) detections, not silently drop to zero pairs
    # or get mapped through the wrong (block-local vs. global) column index.
    assert combined.get(3) == (-995.0, 0.0)
    assert combined.get(4) == (-1105.0, 0.0)
    assert combined.get(5) == (-1205.0, 0.0)


def test_joint_solve_on_the_same_matrix_would_get_it_wrong():
    """Guards the test above from becoming vacuous.

    Feeds the identical cost matrix to the pre-Task-11 whole-matrix solve (the
    same code path single-arena runs still take, reached by passing no arena
    arrays) and asserts it produces a DIFFERENT within-arena pairing. If a
    future change to the geometry, gates or cost weights makes the joint solve
    right by accident, this fails and tells you the invariance test above is
    no longer proving anything.
    """
    track_xy, track_arena, det_xy, meas_arena = _unbalanced_scene()
    combined, cost, _ = _run_assignment(track_xy, det_xy, track_arena, meas_arena)

    params = _unbalanced_params()
    assigner = TrackAssigner(params)
    n, m = len(track_xy), len(det_xy)
    kf = _DummyKF(n)
    kf.X[:, :2] = np.array(track_xy, dtype=np.float32)
    _, raw_dist_mat, _ = assigner._compute_distance_gates(
        n,
        m,
        [np.array([x, y, 0.0], dtype=np.float32) for (x, y) in det_xy],
        [100] * n,
        kf,
    )
    joint_pairs, _ = assigner._assign_established_hungarian(
        list(range(n)),
        cost,
        raw_dist_mat,
        params["MAX_DISTANCE_THRESHOLD"],
        params["KALMAN_MAX_VELOCITY_MULTIPLIER"] * params["REFERENCE_BODY_SIZE"],
    )
    joint = {int(r): det_xy[int(c)] for r, c in joint_pairs if int(r) < 3}
    assert joint != {slot: d for slot, d in combined.items() if slot < 3}, (
        "the joint solve now agrees with the per-arena solve on this scene -- "
        "the invariance test has lost its teeth; re-tune the geometry"
    )


def test_arena_partition_uses_true_track_index_not_position_in_est():
    """The per-arena row partition must key off each track's real index, not
    its position within ``est``.

    ``est`` is only every established (non-young, non-lost) track, so it can
    be a non-contiguous subset of ``range(N)``. Slot 0 here is young
    (``tracking_continuity`` below ``KALMAN_MATURITY_AGE``) and excluded from
    ``est``, so ``est_sorted == [1..6]`` -- NOT ``range(6)``. A partition that
    slices the arena-id array by *position in est* (e.g.
    ``track_arena[:len(est_sorted)]``) rather than by *true track index*
    (``track_arena[est_sorted]``) silently shifts every later arena lookup by
    one, exactly the failure mode this guards against.
    """
    # slot 0: young/unstable, own (unused) arena id, far from everything so
    # it cannot accidentally win a detection through phase 2.
    # slots 1,2,3 -> arena 0 (A, B, C); slots 4,5,6 -> arena 1.
    track_xy = [
        (9999.0, 9999.0),
        (0.0, 0.0),
        (100.0, 0.0),
        (200.0, 0.0),
        (-1000.0, 0.0),
        (-1100.0, 0.0),
        (-1200.0, 0.0),
    ]
    track_arena = np.array([2, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    det_xy = [
        (5.0, 0.0),  # D0 (arena 0), next to A (slot 1)
        (105.0, 0.0),  # D1 (arena 0), next to B (slot 2)
        (-995.0, 0.0),
        (-1105.0, 0.0),
        (-1205.0, 0.0),
        (-850.0, 0.0),  # arena 1's spare
    ]
    meas_arena = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)
    tracking_continuity = [1] + [100] * (len(track_xy) - 1)

    pairs, _, free_dets = _run_assignment(
        track_xy,
        det_xy,
        track_arena,
        meas_arena,
        tracking_continuity=tracking_continuity,
    )

    assert pairs.get(1) == det_xy[0], "slot 1 (A) should keep its own D0"
    assert pairs.get(2) == det_xy[1], "slot 2 (B) should keep its own D1"
    assert pairs.get(4) == det_xy[2]
    assert pairs.get(5) == det_xy[3]
    assert pairs.get(6) == det_xy[4]
    assert det_xy.index((-850.0, 0.0)) in free_dets
    # Slot 0 (young, far away, orphan arena id) must not have stolen anything.
    assert 0 not in pairs


def test_detection_outside_every_arena_survives_the_per_arena_solve():
    """A detection in no arena (``arena_of_points`` -> -1) must not vanish.

    The per-arena solve builds one block per arena id present among the
    TRACKS, and track slots are never -1, so a -1 detection belongs to no
    block and is never offered to the established-track solve -- exactly as
    the cost kernel's equality test already rejected it. It must come back in
    ``free_dets`` (available to the later phases and to bootstrapping), and it
    must not perturb any arena's own pairing.

    Folding -1 columns into every arena's block instead would re-create the
    very coupling Task 11 removed, one arena at a time: here the stray sits
    850 px from A but 950/1050 px from B/C, so it carries the same 1e6-vs-1e9
    split, and admitting it into arena 0's block makes that block square and
    hands D0/D1 to the wrong tracks. The final assertion below fails if you
    do that.
    """
    track_xy, track_arena, det_xy, meas_arena = _unbalanced_scene()
    baseline, _, _ = _run_assignment(track_xy, det_xy, track_arena, meas_arena)

    # A stray detection inside neither arena, at the SAME x as arena 1's
    # spare so it inherits the 1e6/1e9 split across arena 0's three tracks.
    stray = (-850.0, 1.0)
    det_xy_stray = det_xy + [stray]
    meas_arena_stray = np.append(meas_arena, np.int32(-1))
    with_stray, cost, free_dets = _run_assignment(
        track_xy, det_xy_stray, track_arena, meas_arena_stray
    )

    # Non-vacuity: only the arena label keeps the stray out of arena 0's
    # reach -- it is inside A's distance gate, so its cell is arena-blocked
    # (1e6), not distance-gated (1e9).
    assert cost[0, 6] == pytest.approx(
        1e6
    ), "stray cell is distance-gated, not arena-gated"
    assert cost[2, 6] == pytest.approx(1e9)
    assert stray not in with_stray.values(), "a -1 detection was matched to a track"
    assert det_xy_stray.index(stray) in free_dets, "the -1 detection vanished"
    assert with_stray == baseline, "a -1 detection perturbed the per-arena pairings"
