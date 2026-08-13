import math

import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.trajectory_writer import project_user_tracks


def _base_df():
    return pd.DataFrame(
        {
            "TrackID": [0, 0],
            "Index": [0, 1],
            "TrajectoryID": [3, 3],
            "FrameID": [0, 10],
            "X": [1.4, 2.6],
            "Y": [5.0, 6.0],
            "Theta": [0.0, -math.pi / 2],  # 0 rad, -90 deg
            "State": ["active", "occluded"],
            "DetectionConfidence": [0.9, 0.8],
            "AssignmentConfidence": [0.5, 0.4],
            "PositionUncertainty": [1.1, 1.2],
        }
    )


def test_core_columns_and_conversions():
    out = project_user_tracks(_base_df(), fps=10.0)
    assert list(out.columns) == [
        "id",
        "frame",
        "time_s",
        "x",
        "y",
        "heading_deg",
        "state",
        "detection_confidence",
    ]
    assert out["id"].tolist() == [3, 3]
    assert out["time_s"].tolist() == [0.0, 1.0]  # frame/fps
    # -pi/2 rad -> 270 deg (normalized to [0,360))
    assert out["heading_deg"].round(3).tolist() == [0.0, 270.0]
    # tracer-only confidences dropped
    assert "AssignmentConfidence" not in out.columns
    assert "PositionUncertainty" not in out.columns


def test_identity_columns_appear_only_when_final_present():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", ""]
    df[C.FINAL_SMOOTHED_LABEL] = ["antA", "antB"]
    df[C.FINAL_CONFIDENCE] = [0.7, 0.6]
    df[C.FINAL_SOURCE] = ["realtime", "offline"]
    out = project_user_tracks(df, fps=10.0)
    assert out["identity"].tolist() == [
        "antA",
        "antB",
    ]  # empty Final falls back to Smoothed
    assert out["identity_confidence"].tolist() == [0.7, 0.6]
    assert out["identity_source"].tolist() == ["realtime", "offline"]


def test_pose_triples_appear_only_when_pose_present():
    df = _base_df()
    df["PoseKpt_head_X"] = [1.0, 2.0]
    df["PoseKpt_head_Y"] = [3.0, 4.0]
    df["PoseKpt_head_Conf"] = [0.9, 0.8]
    out = project_user_tracks(df, fps=10.0)
    assert out["head_x"].tolist() == [1.0, 2.0]
    assert out["head_y"].tolist() == [3.0, 4.0]
    assert out["head_conf"].tolist() == [0.9, 0.8]


def test_no_fps_yields_nan_time():
    out = project_user_tracks(_base_df(), fps=None)
    assert out["time_s"].isna().all()
