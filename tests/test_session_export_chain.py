import pandas as pd

import hydra_suite.core.tracking.session as session_mod
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
