from hydra_suite.core.tracking.session_summary import build_session_summary_lines


def test_summary_basic_lines():
    config = {
        "detection_method": "yolo_obb",
        "enable_pose_extractor": True,
        "enable_postprocessing": True,
        "enable_backward_tracking": False,
    }
    result = {
        "wall_seconds": 75.0,
        "frames_processed": 500,
        "fps_list": [20.0, 30.0],
        "video_path": "/data/clip.mp4",
        "csv_path": "/out/clip_tracking_final.csv",
        "trajectory_count": 7,
        "dataset": None,
    }
    lines = build_session_summary_lines(config, result)
    assert "Duration: 01:15" in lines
    assert "Frames processed: 500" in lines
    assert "Average FPS: 25.0" in lines
    assert "Video: clip.mp4" in lines
    assert "Output CSV: clip_tracking_final.csv" in lines
    assert "Trajectories: 7" in lines
    assert any(
        line.startswith("Pipelines:") and "Pose extraction" in line for line in lines
    )


def test_summary_dataset_success():
    lines = build_session_summary_lines(
        {"detection_method": "background_subtraction"},
        {
            "wall_seconds": None,
            "frames_processed": 0,
            "fps_list": [],
            "video_path": None,
            "csv_path": None,
            "trajectory_count": None,
            "dataset": {"success": True, "num_frames": 42, "dir": "/out/ds"},
        },
    )
    assert any("Dataset generated: 42 frame(s)" in line for line in lines)
