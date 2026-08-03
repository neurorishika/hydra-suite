import os

import cv2
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


def _draw_config():
    return {
        "video_show_labels": True,
        "video_show_orientation": True,
        "video_show_trails": False,
        "video_trail_duration": 1.0,
        "video_marker_size": 0.3,
        "video_text_scale": 0.5,
        "video_arrow_length": 0.7,
    }


def test_build_video_draw_params_reads_config_keys():
    params = {
        "TRAJECTORY_COLORS": [(1, 2, 3)],
        "REFERENCE_BODY_SIZE": 40.0,
        "ADVANCED_CONFIG": {},
        "POSE_MIN_KPT_CONF_VALID": 0.2,
    }
    df = pd.DataFrame({"TrajectoryID": [0], "X": [1.0], "Y": [2.0]})
    draw_p = media_export.build_video_draw_params(params, _draw_config(), 30.0, df)
    assert draw_p["show_labels"] is True
    assert draw_p["show_trails"] is False
    assert draw_p["marker_radius"] == int(0.3 * 40.0)
    assert draw_p["arrow_len"] == int(0.7 * 40.0)
    assert draw_p["colors"] == [(1, 2, 3)]


def test_get_pose_column_info_false_without_pose_columns():
    df = pd.DataFrame({"TrajectoryID": [0], "X": [1.0], "Y": [2.0]})
    edges, triplets, show_pose = media_export.get_pose_column_info({}, {}, df)
    assert show_pose is False
    assert triplets == []


def test_preextract_traj_arrays_indexes_by_frame():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Theta": [0.0, 0.1],
        }
    )
    arrays = media_export.preextract_traj_arrays(df, False, [], False)
    traj_indices_by_frame = arrays[7]
    assert traj_indices_by_frame[0] == [0]
    assert traj_indices_by_frame[1] == [1]


def _write_black_clip(path, n_frames=10, w=64, h=48, fps=10.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


def _simple_traj_df(n_frames=10):
    return pd.DataFrame(
        {
            "TrajectoryID": [0] * n_frames,
            "FrameID": list(range(n_frames)),
            "X": [30.0] * n_frames,
            "Y": [24.0] * n_frames,
            "Theta": [0.0] * n_frames,
        }
    )


def _render_params():
    return {
        "TRAJECTORY_COLORS": [(0, 255, 0)],
        "REFERENCE_BODY_SIZE": 10.0,
        "ADVANCED_CONFIG": {},
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "START_FRAME": 0,
        "END_FRAME": None,
    }


def test_render_annotated_video_writes_output(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_black_clip(src, n_frames=10)
    result = media_export.render_annotated_video(
        trajectories_df=_simple_traj_df(10),
        video_path=str(src),
        output_path=str(out),
        params=_render_params(),
        config=_draw_config(),
    )
    assert result == str(out)
    assert os.path.exists(out)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_render_annotated_video_cancel_deletes_partial(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_black_clip(src, n_frames=30)
    calls = {"n": 0}

    def _stop():
        calls["n"] += 1
        return calls["n"] > 3  # stop after a few frames

    result = media_export.render_annotated_video(
        trajectories_df=_simple_traj_df(30),
        video_path=str(src),
        output_path=str(out),
        params=_render_params(),
        config=_draw_config(),
        should_stop=_stop,
    )
    assert result is None
    assert not os.path.exists(out)  # partial file removed


def test_export_final_media_returns_none_when_nothing_requested(tmp_path):
    result = media_export.export_final_media(
        final_csv_path=str(tmp_path / "final.csv"),
        config={},
        video_path=str(tmp_path / "in.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
        interpolated_roi_npz_path=None,
        fps=30.0,
        image_root=None,
        video_root=None,
        export_images=False,
        export_videos=False,
        padding_fraction=0.1,
        background_color=(0, 0, 0),
    )
    assert result is None


def test_export_final_media_delegates_to_exporter(tmp_path, monkeypatch):
    captured = {}
    # Create dummy files for the existence checks
    (tmp_path / "final.csv").write_text("")
    (tmp_path / "cache.npz").write_bytes(b"")

    class _FakeResult:
        def to_dict(self):
            return {"exported_videos": 2, "exported_images": 0, "output_dir": "vids"}

    class _FakeExporter:
        def __init__(self, dataset_dir, final_csv_path, **kwargs):
            captured["dataset_dir"] = str(dataset_dir)
            captured["kwargs"] = kwargs

        def export(self, progress_callback=None, should_stop=None):
            return _FakeResult()

    monkeypatch.setattr(media_export, "OrientedTrackVideoExporter", _FakeExporter)
    result = media_export.export_final_media(
        final_csv_path=str(tmp_path / "final.csv"),
        config={
            "individual_save_interval": 2,
            "individual_output_format": "png",
            "final_media_export_heading_flip_burst": 5,
            "final_media_export_stabilization_window": 5,
        },
        video_path=str(tmp_path / "in.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
        interpolated_roi_npz_path=None,
        fps=30.0,
        image_root=None,
        video_root=tmp_path / "vroot",
        export_images=False,
        export_videos=True,
        padding_fraction=0.1,
        background_color=(0, 0, 0),
    )
    assert result == {"exported_videos": 2, "exported_images": 0, "output_dir": "vids"}
    assert captured["kwargs"]["export_videos"] is True
    assert captured["kwargs"]["image_interval"] == 2
