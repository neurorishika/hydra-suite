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


def test_summary_dataset_success_reports_manifest_drop_accounting():
    """The manifest's honest drop accounting (dropped_lost/dropped_unmatched)
    must actually reach the user, not just live in the manifest dict."""
    lines = build_session_summary_lines(
        {"detection_method": "yolo_obb"},
        {
            "wall_seconds": None,
            "frames_processed": 0,
            "fps_list": [],
            "video_path": None,
            "csv_path": None,
            "trajectory_count": None,
            "dataset": {
                "success": True,
                "num_frames": 42,
                "dir": "/out/ds",
                "manifest": {
                    "roots": [
                        {"level": "obb", "authoritative": True},
                        {"level": "aabb", "authoritative": False},
                    ],
                    "totals": {
                        "objects": 210,
                        "dropped_lost": 3,
                        "dropped_unmatched": 7,
                    },
                },
            },
        },
    )
    joined = "\n".join(lines)
    assert "Label levels: obb, aabb" in joined
    assert "Objects labelled: 210" in joined
    assert "Dropped (lost/interpolated tracks): 3" in joined
    assert "Dropped (no matching detection): 7" in joined


def test_summary_dataset_success_no_drops_stays_quiet():
    """No spurious drop lines when nothing was dropped."""
    lines = build_session_summary_lines(
        {"detection_method": "yolo_obb"},
        {
            "wall_seconds": None,
            "frames_processed": 0,
            "fps_list": [],
            "video_path": None,
            "csv_path": None,
            "trajectory_count": None,
            "dataset": {
                "success": True,
                "num_frames": 10,
                "dir": "/out/ds",
                "manifest": {
                    "roots": [{"level": "polygon", "authoritative": True}],
                    "totals": {
                        "objects": 40,
                        "dropped_lost": 0,
                        "dropped_unmatched": 0,
                    },
                },
            },
        },
    )
    joined = "\n".join(lines)
    assert "Dropped" not in joined


def test_summary_reports_skipped_and_failed_frames():
    lines = build_session_summary_lines(
        {"detection_method": "yolo_obb"},
        {
            "wall_seconds": 1.0,
            "frames_processed": 1,
            "fps_list": [1.0],
            "video_path": "/data/clip.mp4",
            "csv_path": "/out/clip.csv",
            "trajectory_count": 1,
            "dataset": {
                "success": True,
                "num_frames": 2,
                "dir": "/tmp/round",
                "manifest": {
                    "roots": [{"level": "polygon"}],
                    "totals": {
                        "objects": 4,
                        "frames_skipped_no_records": 3,
                        "detection_failed": 1,
                    },
                },
            },
        },
    )
    text = "\n".join(lines)
    assert "Frames skipped (no detection survived): 3" in text
    assert "Frames dropped (detection failed): 1" in text
