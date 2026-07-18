from hydra_suite.core.inference.config import (
    InferenceConfig,
    build_inference_config_from_params,
)


def test_direct_obb_minimal_params():
    cfg = build_inference_config_from_params(
        {
            "DETECTION_METHOD": "yolo_obb",
            "YOLO_OBB_MODE": "direct",
            "YOLO_OBB_DIRECT_MODEL_PATH": "some.pt",
            "COMPUTE_RUNTIME": "cpu",
            "MAX_TARGETS": 8,
        }
    )
    assert isinstance(cfg, InferenceConfig)
    assert cfg.obb is not None and cfg.obb.mode == "direct"
    assert cfg.obb.direct.model_path == "some.pt"
    # No headtail/cnn/pose enabled -> those stay unset (OBB-only by omission).
    assert cfg.headtail is None
    assert cfg.cnn_phases == []
    assert cfg.pose is None
    # raw cap = 2*MAX_TARGETS, final cap = MAX_TARGETS (legacy parity).
    assert cfg.obb.max_detections == 8
    assert cfg.obb.raw_detection_cap == 16
    assert cfg.runtime_tier == "cpu"


def test_build_obb_only_config_is_detection_only():
    from hydra_suite.core.inference.config import build_obb_only_config

    cfg = build_obb_only_config("m.pt", compute_runtime="cpu", confidence_threshold=0.3)
    assert cfg.obb is not None and cfg.obb.direct.model_path == "m.pt"
    assert cfg.obb.confidence_threshold == 0.3
    assert cfg.headtail is None and cfg.cnn_phases == [] and cfg.pose is None
    assert cfg.apriltag.enabled is False
