import pandas as pd

from hydra_suite.core.post.trajectory_writer import write_base_final_csv


def test_base_final_rounds_reorders_and_drops(tmp_path):
    df = pd.DataFrame(
        {
            "TrackID": [7],
            "Index": [0],
            "TrajectoryID": [2],
            "X": [1.6],
            "Y": [3.4],
            "Theta": [0.5],
            "FrameID": [10.0],
            "State": ["active"],
        }
    )
    out = tmp_path / "clip_final.csv"
    assert write_base_final_csv(df, str(out)) is True
    got = pd.read_csv(out)
    assert list(got.columns)[:5] == ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    assert "TrackID" not in got.columns and "Index" not in got.columns
    assert got["X"].tolist() == [2] and got["Y"].tolist() == [3]  # rounded


def test_base_final_empty_returns_false(tmp_path):
    assert write_base_final_csv(pd.DataFrame(), str(tmp_path / "x.csv")) is False
