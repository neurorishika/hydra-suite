"""Equivalence + performance guard for the merge-overlap pair enumeration.

``_merge_overlapping_agreeing_trajectories`` used to consider every ordered
pair ``(i, j)`` of trajectories on every iteration, even though
``_try_merge_trajectory_pair`` rejects non-overlapping frame ranges on its
first line.  On a real run (41,450 frames, ~60k trajectories) that all-pairs
loop took 2h45m -- 91% of the whole post-processing ``resolve`` span -- and
froze the GUI for the duration.

The optimisation restricts enumeration to bounds-overlapping ``j`` via a
start-frame-sorted index.  Every pair it skips is a pair the callee would have
rejected without side effects, so the result must be *identical*, not merely
similar.  ``_reference_merge_overlapping`` below is the original all-pairs body
kept verbatim as the oracle.
"""

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post import processing as P


def _reference_merge_overlapping(
    trajectories,
    agreement_distance,
    min_overlap,
    min_length,
    identity_disagree_min_run=5,
    identity_drives_splits: bool = True,
):
    """The original O(n^2) all-pairs body, verbatim, as the equivalence oracle."""
    if not trajectories:
        return trajectories

    has_detection_id = (
        "DetectionID" in trajectories[0].columns if trajectories else False
    )
    max_spatial_jump = agreement_distance * 5

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        merged_any = False
        used = set()
        new_trajectories = []

        traj_lookups = []
        traj_frame_sets = []
        traj_bounds = []
        for traj in trajectories:
            lookup = P._build_frame_lookup(traj, require_valid_x=True)
            frame_set = set(lookup.keys())
            traj_lookups.append(lookup)
            traj_frame_sets.append(frame_set)
            if frame_set:
                traj_bounds.append((min(frame_set), max(frame_set)))
            else:
                traj_bounds.append((np.inf, -np.inf))

        for i in range(len(trajectories)):
            if i in used:
                continue
            traj_a = trajectories[i]
            for j in range(i + 1, len(trajectories)):
                if j in used:
                    continue
                result_segments = P._try_merge_trajectory_pair(
                    i,
                    j,
                    trajectories,
                    traj_lookups,
                    traj_frame_sets,
                    traj_bounds,
                    has_detection_id,
                    agreement_distance,
                    min_overlap,
                    min_length,
                    max_spatial_jump,
                    identity_disagree_min_run=identity_disagree_min_run,
                    identity_drives_splits=identity_drives_splits,
                )
                if result_segments is not None:
                    used.add(i)
                    used.add(j)
                    new_trajectories.extend(result_segments)
                    merged_any = True
                    break
            if i not in used:
                new_trajectories.append(traj_a)
                used.add(i)

        trajectories = new_trajectories

        if not merged_any:
            break

    return trajectories


def _make_traj(traj_id, start, length, x0, y0, *, rng, nan_x=False, detection_id=None):
    frames = np.arange(start, start + length, dtype=np.int64)
    x = x0 + np.cumsum(rng.normal(0.0, 1.5, size=length))
    y = y0 + np.cumsum(rng.normal(0.0, 1.5, size=length))
    if nan_x:
        x = x.copy()
        x[: max(1, length // 2)] = np.nan
    data = {
        "TrajectoryID": np.full(length, traj_id, dtype=np.int64),
        "FrameID": frames,
        "X": x,
        "Y": y,
        "Theta": rng.uniform(0.0, 2 * np.pi, size=length),
    }
    if detection_id is not None:
        data["DetectionID"] = np.full(length, detection_id, dtype=np.int64)
    return pd.DataFrame(data)


def _make_population(n, *, seed, with_detection_id=False, n_frames=4000):
    """Trajectories spread over a long frame range: most pairs are disjoint in
    time (the real-data shape), a minority genuinely overlap, and a few have
    all-NaN-X prefixes so empty/short lookups occur."""
    rng = np.random.default_rng(seed)
    trajs = []
    for k in range(n):
        start = int(rng.integers(0, n_frames))
        length = int(rng.integers(6, 40))
        # Cluster positions so a real fraction of the time-overlapping pairs
        # also agree spatially and actually merge.
        cluster = int(rng.integers(0, max(2, n // 8)))
        x0 = 50.0 * cluster + rng.normal(0.0, 3.0)
        y0 = 50.0 * cluster + rng.normal(0.0, 3.0)
        trajs.append(
            _make_traj(
                k,
                start,
                length,
                x0,
                y0,
                rng=rng,
                nan_x=(k % 37 == 0),
                detection_id=(cluster if with_detection_id else None),
            )
        )
    return trajs


def _assert_same(actual, expected):
    """Exact, order-preserving equality of two result lists.

    ``check_exact=True`` because the claim is byte-identity, not closeness;
    ``assert_frame_equal`` treats NaN in the same position as equal, which a
    plain ``==`` on the values would not.
    """
    assert len(actual) == len(expected)
    for k, (a, e) in enumerate(zip(actual, expected)):
        pd.testing.assert_frame_equal(a, e, check_exact=True, obj=f"trajectory[{k}]")


@pytest.mark.parametrize("with_detection_id", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interval_index_matches_all_pairs_reference(seed, with_detection_id):
    trajs = _make_population(400, seed=seed, with_detection_id=with_detection_id)

    expected = _reference_merge_overlapping(
        [df.copy() for df in trajs],
        agreement_distance=15.0,
        min_overlap=5,
        min_length=5,
    )
    actual = P._merge_overlapping_agreeing_trajectories(
        [df.copy() for df in trajs], 15.0, 5, 5
    )

    _assert_same(actual, expected)


def test_empty_and_singleton_inputs():
    assert P._merge_overlapping_agreeing_trajectories([], 15.0, 5, 5) == []
    rng = np.random.default_rng(7)
    one = [_make_traj(0, 0, 20, 0.0, 0.0, rng=rng)]
    out = P._merge_overlapping_agreeing_trajectories(
        [df.copy() for df in one], 15.0, 5, 5
    )
    _assert_same(out, one)


def test_all_nan_x_trajectories_survive_untouched():
    """Trajectories whose lookup is empty get bounds (inf, -inf); they must be
    excluded from the interval index without being dropped from the output."""
    rng = np.random.default_rng(11)
    empty = pd.DataFrame(
        {
            "TrajectoryID": np.zeros(10, dtype=np.int64),
            "FrameID": np.arange(10, dtype=np.int64),
            "X": np.full(10, np.nan),
            "Y": np.full(10, np.nan),
            "Theta": np.zeros(10),
        }
    )
    trajs = [empty, _make_traj(1, 0, 20, 0.0, 0.0, rng=rng)]
    expected = _reference_merge_overlapping([df.copy() for df in trajs], 15.0, 5, 5)
    actual = P._merge_overlapping_agreeing_trajectories(
        [df.copy() for df in trajs], 15.0, 5, 5
    )
    _assert_same(actual, expected)


def test_considers_only_temporally_overlapping_pairs(monkeypatch):
    """The whole point, stated as an invariant rather than a stopwatch.

    On a realistically sparse population (short trajectories spread over a long
    video) the indexed enumeration must reach ``_try_merge_trajectory_pair``
    for only a small fraction of the pairs the all-pairs form did -- and still
    return the identical result.  Counting calls rather than timing keeps the
    guard deterministic and machine-independent.
    """
    trajs = _make_population(2000, seed=5, n_frames=40000)

    counter = {"n": 0}
    real = P._try_merge_trajectory_pair

    def counting(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(P, "_try_merge_trajectory_pair", counting)

    expected = _reference_merge_overlapping([df.copy() for df in trajs], 15.0, 5, 5)
    ref_calls = counter["n"]

    counter["n"] = 0
    actual = P._merge_overlapping_agreeing_trajectories(
        [df.copy() for df in trajs], 15.0, 5, 5
    )
    new_calls = counter["n"]

    _assert_same(actual, expected)
    assert ref_calls > 100_000, f"reference should be quadratic, got {ref_calls} calls"
    assert new_calls < ref_calls / 50, (
        f"expected the indexed enumeration to consider <2% of the pairs; "
        f"reference={ref_calls} new={new_calls}"
    )


def test_stop_is_observed_mid_pass():
    """A single pass can run for minutes; Stop must be heard inside it, and the
    partially-built pass must not be adopted (that would drop every trajectory
    after the stop point)."""
    trajs = _make_population(300, seed=3)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 25

    out = P._merge_overlapping_agreeing_trajectories(
        [df.copy() for df in trajs], 15.0, 5, 5, should_stop=should_stop
    )
    # Stopped mid-first-pass -> the untouched input generation comes back,
    # not a truncated prefix of it.
    assert len(out) == len(trajs)
