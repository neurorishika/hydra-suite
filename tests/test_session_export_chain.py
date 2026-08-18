import cv2
import numpy as np
import pandas as pd

import hydra_suite.core.tracking.session as session_mod
from hydra_suite.core.post import media_export as _mx
from hydra_suite.core.tracking.session import SessionCallbacks, TrackingSessionCore


def _make_service(config, params, tmp_path):
    paths = {
        "detection_cache_path": str(tmp_path / "cache.npz"),
        "interpolated_roi_npz_path": None,
        "individual_dataset_dir": tmp_path / "ds",
        "final_media_video_dir": tmp_path / "vids",
        "source_video_fps": 30.0,
    }
    return TrackingSessionCore(
        video_path=str(tmp_path / "in.mp4"),
        config=config,
        params=params,
        paths=paths,
        callbacks=SessionCallbacks(),
    )


def test_dataset_stage_fills_dataset_result(monkeypatch, tmp_path):
    svc = _make_service(
        {
            "enable_dataset_generation": True,
            "dataset_class_name": "ant",
            "dataset_max_frames": 5,
            "dataset_diversity_window": 30,
            "dataset_include_context": True,
            "dataset_probabilistic_sampling": False,
        },
        {},
        tmp_path,
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")
    (tmp_path / "in.mp4").write_bytes(b"x")

    monkeypatch.setattr(
        session_mod.dataset_export,
        "generate_active_learning_dataset",
        lambda **k: {"success": True, "num_frames": 3, "dir": "d"},
    )
    result = svc._run_dataset_generation(str(final_csv))
    assert result == {"success": True, "num_frames": 3, "dir": "d"}


def test_dataset_stage_skipped_when_disabled(tmp_path):
    svc = _make_service({"enable_dataset_generation": False}, {}, tmp_path)
    assert svc._run_dataset_generation(str(tmp_path / "final.csv")) is None


def test_dataset_stage_threads_export_levels_dedup_and_class_names(
    monkeypatch, tmp_path
):
    """Task 15: all four export-related knobs reach generate_active_learning_dataset."""
    svc = _make_service(
        {"enable_dataset_generation": True, "dataset_class_name": "ant"},
        {
            "DETECTION_METHOD": "background_subtraction",  # native level = POLYGON
            "DATASET_EXPORT_LEVELS": ["polygon", "obb"],
            "DATASET_CLASS_NAMES": ["ant", "larva"],
            "DATASET_DEDUP_METHOD": "dhash",
            "DATASET_DEDUP_THRESHOLD": 12,
        },
        tmp_path,
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")
    (tmp_path / "in.mp4").write_bytes(b"x")

    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return {"success": True, "num_frames": 1, "dir": "d"}

    monkeypatch.setattr(
        session_mod.dataset_export, "generate_active_learning_dataset", _fake_generate
    )
    svc._run_dataset_generation(str(final_csv))

    from hydra_suite.utils.geometry_levels import GeometryLevel

    assert captured["export_levels"] == [GeometryLevel.POLYGON, GeometryLevel.OBB]
    assert captured["class_names"] == ["ant", "larva"]
    assert captured["dedup_method"] == "dhash"
    assert captured["dedup_threshold"] == 12


def test_dataset_stage_clamps_stored_level_to_achievable(monkeypatch, tmp_path):
    """A stale stored 'polygon' preference against an OBB-only detector is
    clamped down to [obb, aabb] rather than raised or emptied."""
    svc = _make_service(
        {"enable_dataset_generation": True, "dataset_class_name": "ant"},
        {
            "DETECTION_METHOD": "yolo_obb",
            "YOLO_OBB_MODE": "direct",
            "YOLO_OBB_DIRECT_TASK": "obb",
            "DATASET_EXPORT_LEVELS": ["polygon"],
        },
        tmp_path,
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")
    (tmp_path / "in.mp4").write_bytes(b"x")

    captured = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return {"success": True, "num_frames": 1, "dir": "d"}

    monkeypatch.setattr(
        session_mod.dataset_export, "generate_active_learning_dataset", _fake_generate
    )
    svc._run_dataset_generation(str(final_csv))

    from hydra_suite.utils.geometry_levels import GeometryLevel

    assert captured["export_levels"] == [GeometryLevel.OBB, GeometryLevel.AABB]


def test_annotated_video_stage_returns_path(monkeypatch, tmp_path):
    svc = _make_service(
        {"video_output_enabled": True, "video_output_path": str(tmp_path / "out.mp4")},
        {},
        tmp_path,
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")

    monkeypatch.setattr(
        session_mod.media_export,
        "load_video_trajectories",
        lambda p: (pd.read_csv(final_csv), str(final_csv)),
    )
    monkeypatch.setattr(
        session_mod.media_export,
        "render_annotated_video",
        lambda **k: k["output_path"],
    )
    out = svc._run_annotated_video(str(final_csv))
    assert out == str(tmp_path / "out.mp4")


def _write_black_clip(path, n_frames=12, w=64, h=48, fps=12.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n_frames):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


def test_media_parity_frame_count_matches_input(tmp_path):
    """Media output is not covered by the CSV harness — assert the rendered
    video exists, is non-empty, and has the same frame count as the input clip."""
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_black_clip(src, n_frames=12)
    df = pd.DataFrame(
        {
            "TrajectoryID": [0] * 12,
            "FrameID": list(range(12)),
            "X": [30.0] * 12,
            "Y": [24.0] * 12,
            "Theta": [0.0] * 12,
        }
    )
    result = _mx.render_annotated_video(
        trajectories_df=df,
        video_path=str(src),
        output_path=str(out),
        params={
            "TRAJECTORY_COLORS": [(0, 255, 0)],
            "REFERENCE_BODY_SIZE": 10.0,
            "ADVANCED_CONFIG": {},
            "POSE_MIN_KPT_CONF_VALID": 0.2,
            "START_FRAME": 0,
            "END_FRAME": None,
        },
        config={
            "video_show_labels": True,
            "video_show_orientation": True,
            "video_show_trails": False,
            "video_trail_duration": 1.0,
            "video_marker_size": 0.3,
            "video_text_scale": 0.5,
            "video_arrow_length": 0.7,
        },
    )
    assert result == str(out)
    assert out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n_out == 12
