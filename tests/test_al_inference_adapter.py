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
    # Without an explicit `detect_confidence_threshold` override, stage-1
    # falls back to OBBSequentialConfig's own dataclass default (0.25) --
    # documenting this pre-existing default explicitly here so a future
    # change to it is a deliberate, reviewed decision, not a silent drift.
    assert cfg.obb.sequential.detect_confidence_threshold == 0.25


def test_build_obb_config_for_al_sequential_mode_detect_confidence_override():
    """Task 10 fix round (Critical finding C1): without this override, stage-1
    detect confidence was ALWAYS 0.25 regardless of the caller's requested
    `confidence_threshold` -- a real behavior change from the retired
    per-frame AL detector closure, which applied one caller `conf` to BOTH
    stages. `al_worker._build_detection_context` now always passes
    `detect_confidence_threshold=req.base_conf` to restore that parity."""
    cfg = build_obb_config_for_al(
        "sequential",
        "/path/to/detect.pt",
        "/path/to/obb.pt",
        crop_pad_ratio=0.2,
        confidence_threshold=0.05,
        iou_threshold=0.5,
        detect_confidence_threshold=0.05,
    )
    assert cfg.obb.sequential.detect_confidence_threshold == 0.05
    # Stage-2's own confidence gate is unaffected by this override -- it was
    # already `confidence_threshold` (matching OLD's actual per-stage-2
    # behavior), not something this fix round needed to touch.
    assert cfg.obb.sequential.obb_confidence_threshold == 0.05


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


@pytest.mark.parametrize(
    "kind,secondary",
    [("obb_direct", None), ("sequential", "/path/to/obb.pt")],
)
def test_al_config_does_not_inherit_the_8_detection_tracking_cap(kind, secondary):
    """DESIGN RULE: AL scoring must see everything the model proposes.

    `build_obb_only_config` defaults to `max_targets=8` (a tracking knob: the
    user's declared animal count), which becomes `raw_detection_cap=16` at RAW
    extraction and `max_detections=8` after filtering. Inheriting that here
    silently truncated every crowded frame, corrupting every AL signal AND --
    since the round exports labels straight from these detections -- writing a
    fabricated "only 8 animals here" ground truth for exactly the crowded
    frames AL exists to surface.
    """
    from hydra_suite.data.al.inference_adapter import AL_DEFAULT_MAX_TARGETS

    cfg = build_obb_config_for_al(
        kind,
        "/path/to/model.pt",
        secondary,
        crop_pad_ratio=0.15,
        confidence_threshold=0.05,
        iou_threshold=0.5,
    )
    # A frame with 15-20 real animals must survive both caps untouched.
    assert cfg.obb.max_detections == AL_DEFAULT_MAX_TARGETS
    assert cfg.obb.max_detections >= 20
    assert cfg.obb.raw_detection_cap >= 2 * 20


def test_al_config_max_targets_is_overridable():
    cfg = build_obb_config_for_al(
        "obb_direct",
        "/path/to/model.pt",
        None,
        crop_pad_ratio=0.15,
        confidence_threshold=0.05,
        iou_threshold=0.5,
        max_targets=1000,
    )
    assert cfg.obb.max_detections == 1000
    assert cfg.obb.raw_detection_cap == 2000
