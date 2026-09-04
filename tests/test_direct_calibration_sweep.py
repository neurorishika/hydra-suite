import pytest

from hydra_suite.core.inference.direct_calibration_sweep import build_calibration_config

SLICE_PARAMS = {
    "SLICE_ENABLED": True,
    "SLICE_GEOMETRY_MODE": "auto_object",
    "SLICE_OBJECT_TILE_FRACTION": 0.4,
    "SLICE_OVERLAP": 0.2,
    "SLICE_TRAINED_BODY_PX": 120.0,
}


def test_config_is_built_from_params_and_carries_every_claimed_field(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    config = build_calibration_config(
        str(model),
        slice_params=SLICE_PARAMS,
        max_targets=64,
        confidence=0.35,
        runtime_tier="cpu",
    )
    slice_cfg = config.obb.direct.slice
    assert slice_cfg.enabled is True
    assert slice_cfg.geometry_mode == "auto_object"
    assert slice_cfg.object_tile_fraction == pytest.approx(0.4)
    assert slice_cfg.overlap_width_ratio == pytest.approx(0.2)
    assert slice_cfg.reference_body_px == pytest.approx(120.0)
    assert config.obb.confidence_threshold == pytest.approx(0.35)
    assert config.obb.max_detections == 64
    assert config.obb.direct.model_path == str(model)


def test_slice_disabled_is_carried_through(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    params = dict(SLICE_PARAMS)
    params["SLICE_ENABLED"] = False
    config = build_calibration_config(
        str(model),
        slice_params=params,
        max_targets=64,
        confidence=0.35,
        runtime_tier="cpu",
    )
    assert config.obb.direct.slice.enabled is False
