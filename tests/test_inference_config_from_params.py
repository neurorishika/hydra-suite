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


def test_direct_seg_params_flow_into_obbdirectconfig():
    from hydra_suite.core.inference.config import build_inference_config_from_params

    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_MODEL_PATH": "m.pt",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_SEG_NUM_ANGLES": 48,
        "YOLO_OBB_SEG_CROP_SIZE": 128,
        "YOLO_OBB_SEG_PAD_RATIO": 0.25,
        "YOLO_OBB_SEG_MASK_THRESHOLD": 0.6,
    }
    d = build_inference_config_from_params(params).obb.direct
    assert d.model_task == "segment"
    assert (d.seg_num_angles, d.seg_crop_size) == (48, 128)
    assert abs(d.seg_pad_ratio - 0.25) < 1e-9 and abs(d.seg_mask_threshold - 0.6) < 1e-9


def test_direct_seg_params_are_clamped_and_default():
    from hydra_suite.core.inference.config import build_inference_config_from_params

    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_MODEL_PATH": "m.pt",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_SEG_NUM_ANGLES": 9999,
        "YOLO_OBB_SEG_PAD_RATIO": "nope",
    }
    d = build_inference_config_from_params(params).obb.direct
    assert d.seg_num_angles == 24 and abs(d.seg_pad_ratio - 0.15) < 1e-9


def test_direct_task_defaults_to_obb_when_unset():
    from hydra_suite.core.inference.config import build_inference_config_from_params

    d = build_inference_config_from_params(
        {"YOLO_OBB_MODE": "direct", "YOLO_MODEL_PATH": "m.pt"}
    ).obb.direct
    assert d.model_task == "obb" and abs(d.fixed_angle_deg) < 1e-9


def test_direct_detect_task_fixed_angle_flows():
    from hydra_suite.core.inference.config import build_inference_config_from_params

    d = build_inference_config_from_params(
        {
            "YOLO_OBB_MODE": "direct",
            "YOLO_MODEL_PATH": "m.pt",
            "YOLO_OBB_DIRECT_TASK": "detect",
            "YOLO_OBB_FIXED_ANGLE_DEG": 42.5,
        }
    ).obb.direct
    assert d.model_task == "detect" and abs(d.fixed_angle_deg - 42.5) < 1e-9
