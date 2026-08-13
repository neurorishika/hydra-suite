import os

import pandas as pd

from hydra_suite.core.post.trajectory_writer import write_final_trajectories


def _rich_df():
    return pd.DataFrame(
        {
            "TrajectoryID": [1, 1],
            "FrameID": [0, 1],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Theta": [0.0, 0.0],
            "State": ["active", "active"],
            "DetectionConfidence": [0.9, 0.8],
        }
    )


def test_user_mode_writes_tracks_csv_only(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    path = write_final_trajectories(_rich_df(), final_csv, debug_mode=False, fps=10.0)
    assert path.endswith("_tracks.csv")
    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "clip_final_with_individual.csv"))
    cols = pd.read_csv(path).columns.tolist()
    assert cols == [
        "id",
        "frame",
        "time_s",
        "x",
        "y",
        "heading_deg",
        "state",
        "detection_confidence",
    ]


def test_debug_mode_writes_with_individual(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    path = write_final_trajectories(_rich_df(), final_csv, debug_mode=True, fps=10.0)
    assert path.endswith("_with_individual.csv")
    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "clip_tracks.csv"))
