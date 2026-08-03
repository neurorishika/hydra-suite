import numpy as np
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


def test_normalize_identity_key_treats_unknown_as_empty():
    assert media_export.normalize_video_identity_color_key("unknown") == ""
    assert media_export.normalize_video_identity_color_key(np.nan) == ""
    assert media_export.normalize_video_identity_color_key(None) == ""
    assert media_export.normalize_video_identity_color_key("apriltag=3") == "apriltag=3"


def test_format_label_falls_back_to_track_id():
    assert media_export.format_video_track_label(7, None) == "ID7"
    assert media_export.format_video_track_label(7, "") == "ID7"


def test_color_key_array_prefers_identity_then_trajectory():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 1],
            "UniqueIdentityKey": ["apriltag=3", "unknown"],
        }
    )
    keys = media_export.build_video_track_color_key_array(df)
    assert keys[0] == "identity:apriltag=3"
    assert keys[1] == "trajectory:1"


def test_precomputed_palette_uses_trajectory_colors_for_plain_tracks():
    colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    track_ids = np.asarray([0, 1, 2], dtype=np.int32)
    color_keys = np.asarray(
        ["trajectory:0", "trajectory:1", "trajectory:2"], dtype=object
    )
    row_colors = media_export.build_precomputed_color_palette(
        colors, track_ids, color_keys
    )
    assert row_colors == [(10, 20, 30), (40, 50, 60), (70, 80, 90)]


def test_media_export_palette_matches_unified_trajectory_colors():
    """media_export must color plain tracks with the Slice-1 unified palette,
    not a locally-generated one — this pins the GUI/CLI color-drift fix."""
    from hydra_suite.core.tracking.session_policy import build_trajectory_colors

    n = 5
    colors = build_trajectory_colors(n)

    # Reference values from the GUI's legacy RNG (np.random.seed(42)+randint).
    np.random.seed(42)
    expected = [tuple(int(v) for v in c) for c in np.random.randint(0, 255, (n, 3))]
    assert colors == expected

    df = pd.DataFrame({"TrajectoryID": [0, 1, 2, 3, 4]})
    color_keys = media_export.build_video_track_color_key_array(df)
    track_ids = df["TrajectoryID"].to_numpy(dtype=np.int32)
    row_colors = media_export.build_precomputed_color_palette(
        colors, track_ids, color_keys
    )
    assert row_colors == expected
