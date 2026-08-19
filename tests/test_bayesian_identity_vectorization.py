"""Differential gate for the vectorized Bayesian identity cost / rejoin scoring.

Both call sites in ``hungarian.py`` used to be Python double loops over
(track, detection) with a per-pair ``np.logaddexp.reduce``.  They are now
computed blockwise.  These tests pin the vectorized form to the exact scalar
form it replaced -- bit-identical, not merely close -- because tracking output
is gated on byte-identity.

Cases are chosen where a divergence is actually plausible: float32 rounding of
the cost matrix, values straddling the ``max_dist`` cap, ragged/mixed-dtype
identity vectors, ``None`` holes, ties in the rejoin score, and the motion
budget gate.
"""

import numpy as np
import pytest

from hydra_suite.core.assigners.hungarian import TrackAssigner


# --------------------------------------------------------------------------
# Reference implementations: verbatim copies of the pre-vectorization loops.
# --------------------------------------------------------------------------
def ref_apply_bayesian_identity_cost(
    cost, alpha, max_dist, track_log_posts, det_log_likes
):
    n_tracks, n_dets = cost.shape
    for i in range(n_tracks):
        log_post_i = track_log_posts.get(i)
        if log_post_i is None:
            continue
        for j in range(min(n_dets, len(det_log_likes))):
            log_like_j = det_log_likes[j]
            if log_like_j is None:
                continue
            log_compat = float(np.logaddexp.reduce(log_post_i + log_like_j))
            addon = alpha * (-log_compat)
            if cost[i, j] < max_dist:
                cost[i, j] = min(cost[i, j] + addon, max_dist - 1e-3)
            else:
                cost[i, j] += addon
    return cost


def ref_slot_best(
    committed_lost,
    track_log_posts,
    det_log_likes,
    assigned_dets,
    log_threshold,
    within_budget,
    meas,
):
    slot_best = {}
    for slot in committed_lost:
        log_post = track_log_posts.get(slot)
        if log_post is None:
            continue
        log_post_arr = np.asarray(log_post, dtype=np.float64)
        for j, log_like in enumerate(det_log_likes):
            if j in assigned_dets or log_like is None:
                continue
            det_xy = np.asarray(meas[j][:2], dtype=np.float64)
            if not within_budget(slot, det_xy):
                continue
            log_like_arr = np.asarray(log_like, dtype=np.float64)
            score = float(np.logaddexp.reduce(log_post_arr + log_like_arr))
            if score > log_threshold:
                if slot not in slot_best or score > slot_best[slot][0]:
                    slot_best[slot] = (score, j)
    return slot_best


# --------------------------------------------------------------------------
# Site 1: _apply_bayesian_identity_cost
# --------------------------------------------------------------------------
def make_assigner(alpha=0.3, max_dist=1000.0, enabled=True):
    return TrackAssigner(
        {
            "ENABLE_IDENTITY_ONLINE_DECODER": enabled,
            "ASSOCIATION_IDENTITY_HINT_SCALE": alpha,
            "MAX_DISTANCE_THRESHOLD": max_dist,
        }
    )


def log_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x - np.logaddexp.reduce(x)


@pytest.mark.parametrize("seed", range(25))
def test_cost_term_bit_identical_random(seed):
    rng = np.random.default_rng(seed)
    n_tracks = int(rng.integers(1, 9))
    n_dets = int(rng.integers(1, 9))
    n_classes = int(rng.integers(1, 12))
    max_dist = float(rng.choice([1000.0, 137.37, 42.5, 9.87654321]))
    alpha = float(rng.choice([0.3, 1.0, 0.07]))

    # Costs deliberately straddle the cap so the min()/where() branch is hit
    # from both sides, including values within a float32 ulp of it.
    cost = rng.uniform(0.0, max_dist * 1.2, size=(n_tracks, n_dets)).astype(np.float32)
    cost.flat[0] = np.float32(max_dist)
    cost.flat[-1] = np.nextafter(np.float32(max_dist), np.float32(0.0))

    dtype = rng.choice([np.float32, np.float64])
    posts = {}
    for i in range(n_tracks):
        posts[i] = (
            None
            if rng.random() < 0.2
            else log_normalize(rng.normal(size=n_classes)).astype(dtype)
        )
    likes = [
        (
            None
            if rng.random() < 0.2
            else log_normalize(rng.normal(scale=3.0, size=n_classes)).astype(dtype)
        )
        for _ in range(n_dets)
    ]

    data = {
        "identity_track_log_posteriors": posts,
        "identity_detection_log_likelihoods": likes,
    }
    got = cost.copy()
    make_assigner(alpha, max_dist)._apply_bayesian_identity_cost(got, data)
    want = ref_apply_bayesian_identity_cost(cost.copy(), alpha, max_dist, posts, likes)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_cost_term_saturating_near_cap():
    """Costs engineered to land exactly on the cap boundary after the addon."""
    max_dist, alpha = 100.0, 1.0
    cap = max_dist - 1e-3
    n_classes = 4
    posts = {i: log_normalize(np.full(n_classes, 0.0)) for i in range(3)}
    likes = [log_normalize(np.full(n_classes, 0.0)) for _ in range(3)]
    addon = alpha * (-float(np.logaddexp.reduce(posts[0] + likes[0])))
    cost = np.array(
        [
            [cap - addon, cap - addon - 1e-4, cap - addon + 1e-4],
            [max_dist, max_dist + 5.0, max_dist - 1e-6],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    data = {
        "identity_track_log_posteriors": posts,
        "identity_detection_log_likelihoods": likes,
    }
    got = cost.copy()
    make_assigner(alpha, max_dist)._apply_bayesian_identity_cost(got, data)
    want = ref_apply_bayesian_identity_cost(cost.copy(), alpha, max_dist, posts, likes)
    np.testing.assert_array_equal(got, want)


def test_cost_term_ragged_and_mixed_dtype_falls_back():
    """Heterogeneous identity vectors must not be silently stacked."""
    max_dist, alpha = 50.0, 0.5
    posts = {
        0: log_normalize([0.1, -2.0, -3.0]).astype(np.float32),
        1: log_normalize([0.5, -1.0, -0.2]).astype(np.float64),
    }
    likes = [
        log_normalize([1.0, 0.0, -1.0]).astype(np.float32),
        log_normalize([-1.0, 2.0, 0.3]).astype(np.float64),
    ]
    cost = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data = {
        "identity_track_log_posteriors": posts,
        "identity_detection_log_likelihoods": likes,
    }
    got = cost.copy()
    make_assigner(alpha, max_dist)._apply_bayesian_identity_cost(got, data)
    want = ref_apply_bayesian_identity_cost(cost.copy(), alpha, max_dist, posts, likes)
    np.testing.assert_array_equal(got, want)


def test_cost_term_all_none_and_empty_are_noops():
    cost = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a = make_assigner()
    for data in (
        {
            "identity_track_log_posteriors": {0: None, 1: None},
            "identity_detection_log_likelihoods": [np.zeros(3), np.zeros(3)],
        },
        {
            "identity_track_log_posteriors": {0: np.zeros(3)},
            "identity_detection_log_likelihoods": [None, None],
        },
    ):
        got = cost.copy()
        a._apply_bayesian_identity_cost(got, data)
        np.testing.assert_array_equal(got, cost)


def test_cost_term_more_dets_than_likelihoods():
    """The column loop is bounded by len(det_log_likes), not by cost width."""
    max_dist, alpha = 80.0, 0.4
    posts = {0: log_normalize([0.0, -1.0]), 1: log_normalize([-0.5, 0.5])}
    likes = [log_normalize([0.2, -0.2])]  # only one, cost has three columns
    cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data = {
        "identity_track_log_posteriors": posts,
        "identity_detection_log_likelihoods": likes,
    }
    got = cost.copy()
    make_assigner(alpha, max_dist)._apply_bayesian_identity_cost(got, data)
    want = ref_apply_bayesian_identity_cost(cost.copy(), alpha, max_dist, posts, likes)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got[:, 1:], cost[:, 1:])


def test_cost_term_disabled_or_zero_alpha_is_noop():
    cost = np.array([[1.0, 2.0]], dtype=np.float32)
    data = {
        "identity_track_log_posteriors": {0: log_normalize([0.0, -1.0])},
        "identity_detection_log_likelihoods": [log_normalize([0.0, -1.0])] * 2,
    }
    for a in (make_assigner(enabled=False), make_assigner(alpha=0.0)):
        got = cost.copy()
        a._apply_bayesian_identity_cost(got, data)
        np.testing.assert_array_equal(got, cost)


# --------------------------------------------------------------------------
# Site 2: identity rejoin scoring inside _assign_respawn
# --------------------------------------------------------------------------
class FakeKF:
    def __init__(self, X):
        self.X = X


def run_respawn(
    assigner,
    params,
    N,
    M,
    meas,
    lost,
    kf,
    committed,
    posts,
    likes,
    missed_frames,
    assigned_dets,
):
    cost = np.full((N, M), 1e6, dtype=np.float32)
    return assigner._assign_respawn(
        cost=cost,
        N=N,
        meas=meas,
        track_states=["lost"] * N,
        tracking_continuity=[0] * N,
        kf_manager=kf,
        association_data={
            "identity_track_log_posteriors": posts,
            "identity_detection_log_likelihoods": likes,
        },
        committed_slot_identities=committed,
        missed_frames=missed_frames,
        _lost=lost,
        _M=M,
        _MAX_DIST=params["MAX_DISTANCE_THRESHOLD"],
        _assigned_dets=set(assigned_dets),
    )


@pytest.mark.parametrize("seed", range(25))
def test_rejoin_pairs_match_scalar_reference(seed):
    rng = np.random.default_rng(seed)
    n_slots = int(rng.integers(2, 7))
    n_dets = int(rng.integers(2, 7))
    n_classes = int(rng.integers(1, 8))

    posts = {
        i: (
            None
            if rng.random() < 0.15
            else log_normalize(rng.normal(scale=2.0, size=n_classes))
        )
        for i in range(n_slots)
    }
    likes = [
        (
            None
            if rng.random() < 0.15
            else log_normalize(rng.normal(scale=2.0, size=n_classes))
        )
        for _ in range(n_dets)
    ]
    meas = [
        np.array([rng.uniform(0, 200), rng.uniform(0, 200), 0.0]) for _ in range(n_dets)
    ]
    X = np.zeros((n_slots, 5))
    X[:, :2] = rng.uniform(0, 200, size=(n_slots, 2))
    kf = FakeKF(X)
    missed = rng.integers(1, 20, size=n_slots)
    assigned = {j for j in range(n_dets) if rng.random() < 0.2}
    committed = {i: f"id{i}" for i in range(n_slots) if rng.random() < 0.8}
    lost = list(range(n_slots))

    params = {
        "IDENTITY_REJOIN_THRESHOLD": float(rng.choice([0.5, 0.05, 0.9])),
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "MAX_DISTANCE_THRESHOLD": 1e-9,  # kill the proximity fallback path
        "ENABLE_IDENTITY_ONLINE_DECODER": True,
    }
    assigner = TrackAssigner(params)
    _, _, pairs = run_respawn(
        assigner,
        params,
        n_slots,
        n_dets,
        meas,
        lost,
        kf,
        committed,
        posts,
        likes,
        missed,
        assigned,
    )

    # Scalar reference for the same inputs.
    committed_lost = [s for s in lost if s in committed]
    log_threshold = float(np.log(max(params["IDENTITY_REJOIN_THRESHOLD"], 1e-10)))
    body = 20.0
    v_max = 2.0 * body
    floor = 2.0 * body

    def within_budget(slot, det_xy):
        dist = float(np.linalg.norm(det_xy - X[slot, :2]))
        budget = max(floor, int(missed[slot]) * v_max * 1.5)
        return dist <= budget

    slot_best = ref_slot_best(
        committed_lost, posts, likes, assigned, log_threshold, within_budget, meas
    )
    det_best = {}
    for slot, (score, det_j) in slot_best.items():
        if det_j not in det_best or score > det_best[det_j][0]:
            det_best[det_j] = (score, slot)
    want = [(slot, det_j) for det_j, (score, slot) in det_best.items()]
    assert sorted(pairs) == sorted(want)


def test_rejoin_tie_keeps_lowest_detection_index():
    """Identical scores must resolve to the first detection, as before."""
    n_classes = 3
    post = log_normalize(np.zeros(n_classes))
    like = log_normalize(np.zeros(n_classes))
    likes = [like.copy(), like.copy(), like.copy()]
    posts = {0: post}
    meas = [
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    kf = FakeKF(np.zeros((1, 5)))
    params = {
        "IDENTITY_REJOIN_THRESHOLD": 1e-6,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "MAX_DISTANCE_THRESHOLD": 1e-9,
    }
    _, _, pairs = run_respawn(
        TrackAssigner(params),
        params,
        1,
        3,
        meas,
        [0],
        kf,
        {0: "a"},
        posts,
        likes,
        np.array([3]),
        set(),
    )
    assert pairs == [(0, 0)]


def test_rejoin_budget_gate_still_blocks_far_detections():
    """A perfect identity match outside the motion budget must not rejoin."""
    post = log_normalize([10.0, -10.0])
    like = log_normalize([10.0, -10.0])
    kf = FakeKF(np.zeros((1, 5)))
    params = {
        "IDENTITY_REJOIN_THRESHOLD": 0.5,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "KALMAN_MAX_VELOCITY_MULTIPLIER": 2.0,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "MAX_DISTANCE_THRESHOLD": 1e-9,
    }
    near = [np.array([10.0, 0.0, 0.0])]
    far = [np.array([100000.0, 0.0, 0.0])]
    _, _, near_pairs = run_respawn(
        TrackAssigner(params),
        params,
        1,
        1,
        near,
        [0],
        kf,
        {0: "a"},
        {0: post},
        [like],
        np.array([1]),
        set(),
    )
    _, _, far_pairs = run_respawn(
        TrackAssigner(params),
        params,
        1,
        1,
        far,
        [0],
        kf,
        {0: "a"},
        {0: post},
        [like],
        np.array([1]),
        set(),
    )
    assert near_pairs == [(0, 0)]
    assert far_pairs == []
