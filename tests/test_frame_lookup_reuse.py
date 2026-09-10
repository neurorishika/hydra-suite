"""The overlap-merge loop must not rebuild every frame lookup on every pass.

Profiling the real X1_d1 resolve showed `_build_frame_lookup` called 1,709,520
times for 811s cumulative -- 56% of the overlap-merge stage -- because the loop
rebuilt a lookup for *every* trajectory on each of its 33 passes even though a
pass only replaces the trajectories it actually merged. Most of that cost is
not the function's own code: extracting one column at a time boxes a fresh
pandas Series per column (38.8M `DataFrame.__getitem__` calls, 700s).

Reuse is only sound because nothing in the loop mutates a passed-through
trajectory or the row dicts in its lookup; these tests pin both that property
and the resulting call-count drop.
"""

import numpy as np
import pandas as pd

from hydra_suite.core.post import processing as P
from tests.test_merge_overlap_interval_index import (
    _make_population,
    _reference_merge_overlapping,
)


def _assert_same(actual, expected):
    assert len(actual) == len(expected)
    for k, (a, e) in enumerate(zip(actual, expected)):
        pd.testing.assert_frame_equal(a, e, check_exact=True, obj=f"trajectory[{k}]")


def test_result_is_unchanged_by_reuse():
    for seed in (0, 1, 2):
        trajs = _make_population(400, seed=seed)
        expected = _reference_merge_overlapping([d.copy() for d in trajs], 15.0, 5, 5)
        actual = P._merge_overlapping_agreeing_trajectories(
            [d.copy() for d in trajs], 15.0, 5, 5
        )
        _assert_same(actual, expected)


def test_lookups_are_not_rebuilt_every_pass(monkeypatch):
    """Must be measured on a population that needs several passes -- on a
    one-pass population every implementation trivially builds n lookups."""
    trajs = _make_population(600, seed=2, n_frames=1500)

    counter = {"n": 0}
    passes = []
    real = P._build_frame_lookup

    def counting(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(P, "_build_frame_lookup", counting)
    P._merge_overlapping_agreeing_trajectories(
        [d.copy() for d in trajs],
        15.0,
        5,
        5,
        progress=lambda frac, msg: passes.append(msg),
    )
    builds = counter["n"]

    assert len(passes) >= 3, f"population converged too fast to be a guard: {passes}"
    # Without reuse this would be ~len(passes) * n. One build per input
    # trajectory is unavoidable; the rest are the newly merged segments.
    assert builds < len(passes) * len(trajs) / 2, (
        f"{builds} builds over {len(passes)} passes of ~{len(trajs)} "
        f"trajectories looks like a full rebuild per pass"
    )


def test_reused_lookup_is_not_stale_after_a_merge():
    """A merged pair is replaced by new DataFrames whose lookups must be built
    fresh -- reusing a lookup keyed on a dead object would silently resurrect
    pre-merge rows."""
    frames = np.arange(0, 60, dtype=np.int64)
    a = pd.DataFrame(
        {
            "TrajectoryID": np.zeros(60, dtype=np.int64),
            "FrameID": frames,
            "X": np.arange(60, dtype=float),
            "Y": np.zeros(60),
            "Theta": np.zeros(60),
        }
    )
    b = a.copy()
    b["TrajectoryID"] = 1
    b["X"] = a["X"] + 1.0  # within agreement distance -> merges with a
    far = a.copy()
    far["TrajectoryID"] = 2
    far["Y"] = 9000.0

    expected = _reference_merge_overlapping(
        [a.copy(), b.copy(), far.copy()], 15.0, 5, 5
    )
    actual = P._merge_overlapping_agreeing_trajectories(
        [a.copy(), b.copy(), far.copy()], 15.0, 5, 5
    )
    _assert_same(actual, expected)


def test_build_frame_lookup_output_is_unchanged():
    """The bulk column extraction must produce exactly the old structure and
    the same scalar types, since downstream code compares and averages them."""
    df = pd.DataFrame(
        {
            "TrajectoryID": np.arange(6, dtype=np.int64),
            "FrameID": np.arange(6, dtype=np.int64),
            "X": np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0]),
            "Y": np.arange(6, dtype=float),
            "Theta": np.arange(6, dtype=np.float32),
            "State": ["active"] * 6,
            "DetectionID": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        }
    )
    lookup = P._build_frame_lookup(df, require_valid_x=True)
    assert set(lookup) == {0, 2, 3, 5}
    row = lookup[2]
    assert set(row) == set(df.columns)
    assert row["X"] == 3.0 and row["State"] == "active"
    for col in df.columns:
        assert type(row[col]) is type(df[col].to_numpy()[2]), col

    every = P._build_frame_lookup(df, require_valid_x=False)
    assert set(every) == set(range(6))
    assert np.isnan(every[1]["X"])


def test_is_missing_matches_pandas():
    values = [
        None,
        float("nan"),
        np.nan,
        np.float64("nan"),
        np.float32("nan"),
        0.0,
        1.5,
        np.float64(2.5),
        np.float32(3.5),
        0,
        7,
        np.int64(9),
        np.int32(11),
        True,
        np.bool_(False),
        "",
        "x",
        pd.NaT,
    ]
    for value in values:
        assert P._is_missing(value) == bool(pd.isna(value)), repr(value)
