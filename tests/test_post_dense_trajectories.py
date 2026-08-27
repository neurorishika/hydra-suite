import numpy as np
import pandas as pd

from hydra_suite.core.post.processing import (
    densify_trajectory_frames,
    final_interpolation_max_gap,
    interpolate_trajectories,
    trim_positionless_ends,
)


def _traj(frames, xs):
    return pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": frames,
            "X": xs,
            "Y": xs,
            "Theta": 0.0,
            "State": ["active" if not np.isnan(x) else "occluded" for x in xs],
            "DetectionID": [i if not np.isnan(x) else np.nan for i, x in enumerate(xs)],
            "DetectionConfidence": [0.9 if not np.isnan(x) else np.nan for x in xs],
        }
    )


def test_densify_inserts_missing_frames_as_occluded():
    df = _traj([1, 2, 5, 6], [1.0, 2.0, 5.0, 6.0])
    out = densify_trajectory_frames(df)
    assert out["FrameID"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out.loc[out.FrameID == 3, "State"].iloc[0] == "occluded"
    assert np.isnan(out.loc[out.FrameID == 3, "DetectionID"].iloc[0])
    assert out.loc[out.FrameID == 3, "DetectionConfidence"].iloc[0] == 0.0


def test_interpolate_fills_gaps_longer_than_max_gap_when_fill_all_interior():
    frames = list(range(1, 12))
    xs = [1.0] + [np.nan] * 9 + [11.0]
    df = _traj(frames, xs)
    capped = interpolate_trajectories(df, method="linear", max_gap=5)
    assert capped["X"].isna().sum() == 9
    full = interpolate_trajectories(
        df, method="linear", max_gap=5, fill_all_interior=True
    )
    assert full["X"].isna().sum() == 0
    assert abs(full.loc[full.FrameID == 6, "X"].iloc[0] - 6.0) < 1e-9


def test_trim_drops_leading_and_trailing_positionless_rows_only():
    df = _traj([1, 2, 3, 4, 5], [np.nan, np.nan, 3.0, np.nan, 5.0])
    out = trim_positionless_ends(df)
    assert out["FrameID"].tolist() == [3, 4, 5]


def test_final_interpolation_max_gap_never_below_user_knob():
    assert (
        final_interpolation_max_gap(
            {"interpolation_max_gap_seconds": 0.5}, {"FPS": 10, "MAX_OCCLUSION_GAP": 10}
        )
        == 11
    )
    assert (
        final_interpolation_max_gap(
            {"interpolation_max_gap_seconds": 5.0}, {"FPS": 10, "MAX_OCCLUSION_GAP": 10}
        )
        == 50
    )
