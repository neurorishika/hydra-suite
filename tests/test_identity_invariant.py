import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.identity_postprocess import (
    assert_one_identity_per_trajectory,
    collapse_to_majority_identity,
)


def _df():
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0, 1, 1],
            "FrameID": [1, 2, 3, 1, 2],
            C.FINAL_LABEL: ["a", "a", "b", "c", "c"],
            C.FINAL_ID: [1, 1, 2, 3, 3],
            C.FINAL_SOURCE: ["offline", "offline", "offline", "offline", "offline"],
            C.FINAL_CONFIDENCE: [0.9, 0.9, 0.2, 0.8, 0.8],
        }
    )


def test_offenders_are_reported():
    assert assert_one_identity_per_trajectory(_df()) == [0]


def test_collapse_uses_majority_and_min_confidence():
    out = collapse_to_majority_identity(_df(), [0])
    t0 = out[out.TrajectoryID == 0]
    assert t0[C.FINAL_LABEL].unique().tolist() == ["a"]
    assert t0[C.FINAL_ID].unique().tolist() == [1]
    assert (t0[C.FINAL_CONFIDENCE] == 0.2).all()
    assert assert_one_identity_per_trajectory(out) == []


def test_no_offenders_when_columns_missing():
    df = pd.DataFrame({"TrajectoryID": [0, 0], "FrameID": [1, 2]})
    assert assert_one_identity_per_trajectory(df) == []


def test_no_offenders_on_clean_frame():
    df = _df()
    df.loc[df.TrajectoryID == 0, C.FINAL_LABEL] = "a"
    df.loc[df.TrajectoryID == 0, C.FINAL_ID] = 1
    assert assert_one_identity_per_trajectory(df) == []


def test_collapse_is_noop_for_untouched_trajectories():
    out = collapse_to_majority_identity(_df(), [0])
    t1 = out[out.TrajectoryID == 1]
    assert t1[C.FINAL_LABEL].tolist() == ["c", "c"]
    assert t1[C.FINAL_CONFIDENCE].tolist() == [0.8, 0.8]
