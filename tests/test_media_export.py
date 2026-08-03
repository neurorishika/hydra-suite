import pandas as pd
import pytest

from hydra_suite.core.post import media_export


def test_scale_trajectories_noop_when_factor_is_one():
    df = pd.DataFrame({"X": [10.0], "Y": [20.0], "Theta": [0.5], "FrameID": [0]})
    out = media_export.scale_trajectories_to_original_space(df, 1.0)
    assert out is df  # unchanged object when no scaling needed


def test_scale_trajectories_scales_xy_only():
    df = pd.DataFrame({"X": [10.0], "Y": [20.0], "Theta": [0.5], "FrameID": [3]})
    out = media_export.scale_trajectories_to_original_space(df, 0.5)
    assert out["X"].iloc[0] == pytest.approx(20.0)
    assert out["Y"].iloc[0] == pytest.approx(40.0)
    assert out["Theta"].iloc[0] == pytest.approx(0.5)  # angle not scaled
    assert out["FrameID"].iloc[0] == 3


def test_save_trajectories_to_csv_writes_ordered_columns(tmp_path):
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "X": [10.4, 11.6],
            "Y": [20.0, 21.0],
            "Theta": [0.1, 0.2],
            "FrameID": [0, 1],
            "TrackID": [5, 5],
            "Extra": ["a", "b"],
        }
    )
    out = tmp_path / "traj.csv"
    assert media_export.save_trajectories_to_csv(df, str(out)) is True
    written = pd.read_csv(out)
    assert list(written.columns)[:5] == ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    assert "TrackID" not in written.columns  # unwanted dropped
    assert written["X"].iloc[0] == 10  # rounded to Int64


def test_save_trajectories_to_csv_none_returns_false(tmp_path):
    assert media_export.save_trajectories_to_csv(None, str(tmp_path / "x.csv")) is False
