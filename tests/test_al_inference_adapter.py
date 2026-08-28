import pytest

from hydra_suite.data.al.inference_adapter import build_obb_config_for_al


def test_build_obb_config_for_al_direct_mode():
    cfg = build_obb_config_for_al(
        "obb_direct",
        "/path/to/model.pt",
        None,
        crop_pad_ratio=0.15,
        confidence_threshold=0.05,
        iou_threshold=0.5,
    )
    assert cfg.obb.mode == "direct"
    assert cfg.obb.direct.model_path == "/path/to/model.pt"
    assert cfg.obb.confidence_threshold == 0.05
    assert cfg.obb.iou_threshold == 0.5


def test_build_obb_config_for_al_sequential_mode():
    cfg = build_obb_config_for_al(
        "sequential",
        "/path/to/detect.pt",
        "/path/to/obb.pt",
        crop_pad_ratio=0.2,
        confidence_threshold=0.05,
        iou_threshold=0.5,
    )
    assert cfg.obb.mode == "sequential"
    assert cfg.obb.sequential.detect_model_path == "/path/to/detect.pt"
    assert cfg.obb.sequential.obb_model_path == "/path/to/obb.pt"
    assert cfg.obb.sequential.crop_pad_ratio == 0.2
    assert cfg.obb.confidence_threshold == 0.05
    assert cfg.obb.iou_threshold == 0.5


def test_build_obb_config_for_al_sequential_missing_secondary_raises():
    with pytest.raises(ValueError):
        build_obb_config_for_al(
            "sequential",
            "/path/to/detect.pt",
            None,
            crop_pad_ratio=0.2,
            confidence_threshold=0.05,
            iou_threshold=0.5,
        )


def test_build_obb_config_for_al_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_obb_config_for_al(
            "unknown",
            "/path/to/model.pt",
            None,
            crop_pad_ratio=0.15,
            confidence_threshold=0.05,
            iou_threshold=0.5,
        )
