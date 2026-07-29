from hydra_suite.core.inference.config import build_inference_config_from_params


def _base(**extra):
    p = {
        "RUNTIME_TIER": "cpu",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 2.0,  # product -> 40.0
        "SLICE_ENABLED": True,
        "SLICE_GEOMETRY_MODE": "auto_object",
    }
    p.update(extra)
    return p


def test_trained_body_px_overrides_reference():
    cfg = build_inference_config_from_params(_base(SLICE_TRAINED_BODY_PX=560.0))
    assert cfg.obb.direct.slice.reference_body_px == 560.0


def test_absent_trained_body_px_uses_product():
    cfg = build_inference_config_from_params(_base())
    assert cfg.obb.direct.slice.reference_body_px == 40.0  # 20 * 2


def test_zero_trained_body_px_uses_product():
    cfg = build_inference_config_from_params(_base(SLICE_TRAINED_BODY_PX=0.0))
    assert cfg.obb.direct.slice.reference_body_px == 40.0
