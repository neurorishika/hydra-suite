from __future__ import annotations

import math

from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    legacy_detection_runtime_fields,
    load_tracker_cli_session,
)


def test_cli_session_preserves_current_video_output_paths_for_foreign_config(tmp_path):
    video_path = tmp_path / "batch_video.mp4"
    config = {
        "file_path": "/elsewhere/original_video.mp4",
        "csv_path": "/elsewhere/original_tracking.csv",
        "video_output_path": "/elsewhere/original_tracking.mp4",
        "fps": 12.0,
        "start_frame": 7,
        "end_frame": 33,
        "enable_postprocessing": False,
        "save_confidence_metrics": False,
        "interpolation_method": "Linear",
    }

    session = load_tracker_cli_session(
        str(video_path),
        config_data=config,
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=120, width=640, height=480
        ),
    )

    assert session.raw_csv_path == str(tmp_path / "batch_video_tracking.csv")
    assert session.final_csv_path == str(
        tmp_path / "batch_video_tracking_forward_processed.csv"
    )
    assert session.params["FPS"] == 12.0
    assert session.params["START_FRAME"] == 7
    assert session.params["END_FRAME"] == 33
    assert session.enable_postprocessing is False
    assert session.save_confidence_metrics is False
    assert session.interpolation_method == "Linear"


def test_cli_session_builds_basic_tracking_conversions_and_roi_mask(tmp_path):
    session = load_tracker_cli_session(
        str(tmp_path / "subject.mp4"),
        config_data={
            "fps": 20.0,
            "max_targets": 3,
            "reference_body_size": 10.0,
            "resize_factor": 2.0,
            "min_object_size_multiplier": 0.5,
            "max_object_size_multiplier": 3.0,
            "max_assignment_distance_multiplier": 2.5,
            "min_respawn_distance_multiplier": 1.5,
            "velocity_threshold": 3.0,
            "max_velocity_break": 4.0,
            "lost_threshold_seconds": 1.5,
            "kalman_maturity_age_seconds": 0.5,
            "background_prime_seconds": 0.25,
            "min_detect_seconds": 0.2,
            "min_track_seconds": 0.3,
            "min_trajectory_length_seconds": 0.4,
            "max_occlusion_gap_seconds": 0.6,
            "velocity_zscore_window_seconds": 0.1,
            "stitch_max_gap_seconds": 0.0,
            "enable_greedy_assignment": True,
            "enable_spatial_optimization": False,
            "roi_shapes": [
                {
                    "type": "polygon",
                    "params": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    "mode": "include",
                },
                {"type": "circle", "params": [2, 2, 1], "mode": "exclude"},
            ],
        },
        video_probe=TrackerCliVideoProbe(fps=20.0, total_frames=50, width=5, height=5),
    )

    params = session.params
    scaled_body_size = 20.0
    scaled_body_area = math.pi * (10.0 / 2.0) ** 2 * (2.0**2)

    assert params["MAX_TARGETS"] == 3
    assert params["MIN_OBJECT_SIZE"] == int(0.5 * scaled_body_area)
    assert params["MAX_OBJECT_SIZE"] == int(3.0 * scaled_body_area)
    assert params["MAX_DISTANCE_THRESHOLD"] == 2.5 * scaled_body_size
    assert params["MIN_RESPAWN_DISTANCE"] == 1.5 * scaled_body_size
    assert params["VELOCITY_THRESHOLD"] == 3.0
    assert params["MAX_VELOCITY_BREAK"] == 4.0
    assert params["LOST_THRESHOLD_FRAMES"] == 30
    assert params["KALMAN_MATURITY_AGE"] == 10
    assert params["BACKGROUND_PRIME_FRAMES"] == 5
    assert params["MIN_DETECTION_COUNTS"] == 4
    assert params["MIN_TRACKING_COUNTS"] == 6
    assert params["MIN_TRAJECTORY_LENGTH"] == 8
    assert params["MAX_OCCLUSION_GAP"] == 12
    assert params["VELOCITY_ZSCORE_WINDOW"] == 5
    assert params["STITCH_MAX_GAP_FRAMES"] == 0
    assert params["ENABLE_GREEDY_ASSIGNMENT"] is True
    assert params["ENABLE_SPATIAL_OPTIMIZATION"] is False
    assert params["ROI_MASK"].shape == (5, 5)
    assert params["ROI_MASK"][0, 0] == 255
    assert params["ROI_MASK"][2, 2] == 0


def test_cli_session_direct_run_support_is_gated_by_gui_owned_features(tmp_path):
    probe = TrackerCliVideoProbe(fps=20.0, total_frames=20, width=32, height=32)

    backward_session = load_tracker_cli_session(
        str(tmp_path / "backward.mp4"),
        config_data={"enable_backward_tracking": True},
        video_probe=probe,
    )
    pose_session = load_tracker_cli_session(
        str(tmp_path / "pose.mp4"),
        config_data={"enable_pose_extractor": True},
        video_probe=probe,
    )
    identity_session = load_tracker_cli_session(
        str(tmp_path / "identity.mp4"),
        config_data={"identity_method": "apriltags"},
        video_probe=probe,
    )
    simple_session = load_tracker_cli_session(
        str(tmp_path / "simple.mp4"),
        config_data={},
        video_probe=probe,
    )

    assert backward_session.supports_direct_run() is True
    assert pose_session.supports_direct_run() is False
    assert identity_session.supports_direct_run() is False
    assert simple_session.supports_direct_run() is True


def test_cli_session_yolo_batch_size_reads_detection_batch_size_from_config(tmp_path):
    session = load_tracker_cli_session(
        str(tmp_path / "subject.mp4"),
        config_data={"detection_batch_size": 8},
        video_probe=TrackerCliVideoProbe(fps=20.0, total_frames=50, width=5, height=5),
    )

    assert session.params["YOLO_BATCH_SIZE"] == 8


def test_cli_session_yolo_batch_size_defaults_to_one_when_not_configured(tmp_path):
    session = load_tracker_cli_session(
        str(tmp_path / "subject.mp4"),
        config_data={},
        video_probe=TrackerCliVideoProbe(fps=20.0, total_frames=50, width=5, height=5),
    )

    assert session.params["YOLO_BATCH_SIZE"] == 1


def test_legacy_detection_runtime_fields_tensorrt():
    out = legacy_detection_runtime_fields("tensorrt")
    assert out == {
        "yolo_device": "cuda:0",
        "enable_tensorrt": True,
        "enable_onnx_runtime": False,
        "enable_gpu_background": True,
    }


def test_legacy_detection_runtime_fields_onnx_coreml():
    out = legacy_detection_runtime_fields("onnx_coreml")
    assert out == {
        "yolo_device": "mps",
        "enable_tensorrt": False,
        "enable_onnx_runtime": True,
        "enable_gpu_background": True,
    }


def test_legacy_detection_runtime_fields_onnx_cuda():
    out = legacy_detection_runtime_fields("onnx_cuda")
    assert out == {
        "yolo_device": "cuda:0",
        "enable_tensorrt": False,
        "enable_onnx_runtime": True,
        "enable_gpu_background": True,
    }


def test_legacy_detection_runtime_fields_coreml_is_not_collapsed_into_onnx_coreml():
    """Task 8: unlike the deleted derive_detection_runtime_settings, the native
    "coreml" tier-resolved backend must not be collapsed into "onnx_coreml" —
    it should behave like the plain "mps" device with no ONNX flag set."""
    out = legacy_detection_runtime_fields("coreml")
    assert out == {
        "yolo_device": "mps",
        "enable_tensorrt": False,
        "enable_onnx_runtime": False,
        "enable_gpu_background": True,
    }


def test_legacy_detection_runtime_fields_cpu_default():
    out = legacy_detection_runtime_fields("cpu")
    assert out == {
        "yolo_device": "cpu",
        "enable_tensorrt": False,
        "enable_onnx_runtime": False,
        "enable_gpu_background": False,
    }
