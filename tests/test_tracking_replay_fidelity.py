"""Contracts that keep autotuner replay conservative when evidence is cached."""

from hydra_suite.core.tracking.optimization.production_replay import (
    disabled_replay_tuning_dimensions,
    sanitize_replay_tuning_config,
)


def test_downstream_evidence_disables_threshold_tuning_with_explicit_reason():
    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_HEADTAIL_MODEL_PATH": "/models/headtail.pt",
        "CNN_CLASSIFIERS": [{"model_path": "/models/id.pt"}],
        "ENABLE_POSE_EXTRACTOR": True,
        "USE_APRILTAGS": True,
    }

    disabled = disabled_replay_tuning_dimensions(params)

    assert set(disabled) == {
        "YOLO_CONFIDENCE_THRESHOLD",
        "YOLO_IOU_THRESHOLD",
        "KALMAN_INITIAL_VELOCITY_RETENTION",
    }
    assert all(
        "downstream" in disabled[name].lower()
        for name in ("YOLO_CONFIDENCE_THRESHOLD", "YOLO_IOU_THRESHOLD")
    )
    assert "young" in disabled["KALMAN_INITIAL_VELOCITY_RETENTION"].lower()


def test_downstream_evidence_sanitizer_preserves_supported_dimensions():
    tuning, disabled = sanitize_replay_tuning_config(
        {
            "YOLO_CONFIDENCE_THRESHOLD": True,
            "YOLO_IOU_THRESHOLD": True,
            "W_POSITION": True,
            "KALMAN_INITIAL_VELOCITY_RETENTION": True,
        },
        {"DETECTION_METHOD": "yolo_obb", "ENABLE_POSE_EXTRACTOR": True},
    )

    assert tuning == {
        "YOLO_CONFIDENCE_THRESHOLD": False,
        "YOLO_IOU_THRESHOLD": False,
        "W_POSITION": True,
        "KALMAN_INITIAL_VELOCITY_RETENTION": False,
    }
    assert set(disabled) == {
        "YOLO_CONFIDENCE_THRESHOLD",
        "YOLO_IOU_THRESHOLD",
        "KALMAN_INITIAL_VELOCITY_RETENTION",
    }


def test_threshold_tuning_is_allowed_when_no_downstream_evidence_is_enabled():
    tuning, disabled = sanitize_replay_tuning_config(
        {"YOLO_CONFIDENCE_THRESHOLD": True, "YOLO_IOU_THRESHOLD": True},
        {"DETECTION_METHOD": "yolo_obb"},
    )

    assert tuning["YOLO_CONFIDENCE_THRESHOLD"] is True
    assert tuning["YOLO_IOU_THRESHOLD"] is True
    assert set(disabled) == {"KALMAN_INITIAL_VELOCITY_RETENTION"}
    assert "young" in disabled["KALMAN_INITIAL_VELOCITY_RETENTION"].lower()
