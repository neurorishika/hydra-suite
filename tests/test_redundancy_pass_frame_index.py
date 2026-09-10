"""Equivalence + enumeration guard for the spatial-redundancy pass.

``_remove_spatially_redundant_trajectories`` compared every ordered pair of
trajectories, but ``_find_agreeing_frames`` returns an empty set for any pair
with no frame in common, and an empty set always fails the
``agreeing_frames < min(min_overlap, total_b_frames)`` test -- so those pairs
cost a call and change nothing.  On the real X1_d1 dataset the two passes cost
4m41 + 4m30.

Same argument as ``test_merge_overlap_interval_index``: restricting
enumeration to co-occurring trajectories is a pure narrowing, so the result
must be identical.  The pre-change body is kept verbatim below as the oracle.
"""

import numpy as np
import pandas as pd

from hydra_suite.core.post import processing as P
from tests.test_merge_overlap_interval_index import _make_population


def _reference_remove_redundant(trajectories, agreement_distance, min_overlap):
    """The original all-pairs body, verbatim, as the equivalence oracle."""
    if not trajectories:
        return trajectories

    sorted_trajs = sorted(enumerate(trajectories), key=lambda x: -len(x[1]))
    redundant_indices = set()

    traj_arrays = []
    for idx, traj in sorted_trajs:
        frames = traj["FrameID"].values
        x = traj["X"].values
        y = traj["Y"].values
        valid_mask = ~np.isnan(x)
        frame_to_pos = {
            frames[i]: (x[i], y[i]) for i in range(len(frames)) if valid_mask[i]
        }
        traj_arrays.append((idx, frame_to_pos, np.sum(valid_mask)))

    trimmed_replacements = {}

    for i, (idx_a, a_by_frame, _) in enumerate(traj_arrays):
        if idx_a in redundant_indices:
            continue
        for idx_b, b_by_frame, total_b_frames in traj_arrays[i + 1 :]:
            if idx_b in redundant_indices or total_b_frames == 0:
                continue
            agreeing_frame_set = P._find_agreeing_frames(
                a_by_frame, b_by_frame, agreement_distance
            )
            agreeing_frames = len(agreeing_frame_set)
            if agreeing_frames < min(min_overlap, total_b_frames):
                continue
            agreement_ratio = agreeing_frames / total_b_frames
            if agreement_ratio < 0.7:
                continue
            if agreement_ratio >= 0.95:
                redundant_indices.add(idx_b)
            else:
                P._trim_or_remove_trajectory(
                    idx_b,
                    trajectories,
                    agreeing_frame_set,
                    agreeing_frames,
                    total_b_frames,
                    agreement_ratio,
                    trimmed_replacements,
                    redundant_indices,
                )

    result = []
    for i, t in enumerate(trajectories):
        if i in redundant_indices:
            continue
        if i in trimmed_replacements:
            result.append(trimmed_replacements[i])
        else:
            result.append(t)
    return result


def _assert_same(actual, expected):
    assert len(actual) == len(expected)
    for k, (a, e) in enumerate(zip(actual, expected)):
        pd.testing.assert_frame_equal(a, e, check_exact=True, obj=f"trajectory[{k}]")


def test_matches_all_pairs_reference():
    for seed in (0, 1, 2):
        trajs = _make_population(400, seed=seed)
        expected = _reference_remove_redundant([d.copy() for d in trajs], 15.0, 5)
        actual = P._remove_spatially_redundant_trajectories(
            [d.copy() for d in trajs], 15.0, 5
        )
        _assert_same(actual, expected)


def test_duplicates_are_still_removed():
    """Guard against a prune so aggressive it stops finding real redundancy."""
    rng = np.random.default_rng(4)
    base = pd.DataFrame(
        {
            "TrajectoryID": np.zeros(40, dtype=np.int64),
            "FrameID": np.arange(40, dtype=np.int64),
            "X": np.arange(40, dtype=float),
            "Y": np.zeros(40),
            "Theta": rng.uniform(0, 2 * np.pi, 40),
        }
    )
    dup = base.iloc[5:35].copy()
    dup["TrajectoryID"] = 1
    far = base.copy()
    far["TrajectoryID"] = 2
    far["Y"] = 5000.0

    out = P._remove_spatially_redundant_trajectories([base, dup, far], 15.0, 5)
    expected = _reference_remove_redundant([base, dup, far], 15.0, 5)
    _assert_same(out, expected)
    assert len(out) == 2, "the contained duplicate should have been removed"


def test_all_nan_x_trajectories_are_preserved():
    empty = pd.DataFrame(
        {
            "TrajectoryID": np.zeros(12, dtype=np.int64),
            "FrameID": np.arange(12, dtype=np.int64),
            "X": np.full(12, np.nan),
            "Y": np.full(12, np.nan),
            "Theta": np.zeros(12),
        }
    )
    trajs = [empty] + _make_population(30, seed=6)
    expected = _reference_remove_redundant([d.copy() for d in trajs], 15.0, 5)
    actual = P._remove_spatially_redundant_trajectories(
        [d.copy() for d in trajs], 15.0, 5
    )
    _assert_same(actual, expected)


def test_considers_only_co_occurring_pairs(monkeypatch):
    trajs = _make_population(2000, seed=5, n_frames=40000)

    counter = {"n": 0}
    real = P._find_agreeing_frames

    def counting(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(P, "_find_agreeing_frames", counting)

    expected = _reference_remove_redundant([d.copy() for d in trajs], 15.0, 5)
    ref_calls = counter["n"]
    counter["n"] = 0
    actual = P._remove_spatially_redundant_trajectories(
        [d.copy() for d in trajs], 15.0, 5
    )
    new_calls = counter["n"]

    _assert_same(actual, expected)
    assert ref_calls > 100_000, f"reference should be quadratic, got {ref_calls}"
    assert new_calls < ref_calls / 50, (
        f"expected <2% of the pairs to be scored; "
        f"reference={ref_calls} new={new_calls}"
    )
