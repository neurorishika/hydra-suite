import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import (
    merge_same_label_neighbours,
    split_trajectories_at_changepoints,
)


def _traj(tid, frames):
    return pd.DataFrame({"TrajectoryID": tid, "FrameID": frames, "X": 0.0, "Y": 0.0})


def test_short_remnant_is_merged_not_dropped():
    df = _traj(1, range(0, 100))
    out = split_trajectories_at_changepoints(df, {1: [97]}, {"MIN_FRAGMENT_FRAMES": 5})
    assert len(out) == 100  # no rows lost
    assert out["TrajectoryID"].nunique() == 1  # 2-frame remnant folded back


def test_leading_short_remnant_merges_forward():
    df = _traj(1, range(0, 100))
    out = split_trajectories_at_changepoints(
        df, {1: [2, 60]}, {"MIN_FRAGMENT_FRAMES": 5}
    )
    assert len(out) == 100 and out["TrajectoryID"].nunique() == 2


def test_merge_same_label_neighbours_undoes_needless_cut():
    df = pd.concat(
        [
            _traj(10, range(0, 50)),
            _traj(11, range(50, 100)),
            _traj(12, range(100, 150)),
        ]
    )
    df["OriginalTrajectoryID"] = 1
    df[C.FINAL_LABEL] = np.where(df["TrajectoryID"] == 12, "b", "a")
    out = merge_same_label_neighbours(df)
    assert out["TrajectoryID"].nunique() == 2
    assert out.loc[out.FrameID < 100, "TrajectoryID"].nunique() == 1


def test_merge_respects_different_originals():
    df = pd.concat([_traj(10, range(0, 50)), _traj(11, range(50, 100))])
    df["OriginalTrajectoryID"] = [1] * 50 + [2] * 50
    df[C.FINAL_LABEL] = "a"
    assert merge_same_label_neighbours(df)["TrajectoryID"].nunique() == 2
