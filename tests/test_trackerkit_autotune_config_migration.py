"""Legacy mode strings migrate to the apply boolean."""

import pytest

from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


@pytest.mark.parametrize(
    "legacy,expected",
    [("automatic", True), ("record", True), ("off", False)],
)
def test_legacy_mode_migrates_to_boolean(legacy, expected):
    cfg = TrackerConfig.from_dict({"inference_autotune_mode": legacy})
    assert cfg.apply_tuned_inference is expected


def test_absent_key_defaults_to_disabled():
    cfg = TrackerConfig.from_dict({})
    assert cfg.apply_tuned_inference is False


def test_new_key_wins_over_legacy():
    cfg = TrackerConfig.from_dict(
        {"inference_autotune_mode": "off", "apply_tuned_inference": True}
    )
    assert cfg.apply_tuned_inference is True


def test_roundtrip_emits_only_the_new_key():
    cfg = TrackerConfig.from_dict({"inference_autotune_mode": "automatic"})
    data = cfg.to_dict()
    assert data["apply_tuned_inference"] is True
    assert "inference_autotune_mode" not in data


@pytest.mark.parametrize("apply_tuned", [True, False])
def test_tile_batch_autotune_is_independent_of_the_apply_flag(apply_tuned):
    """The process-local SAHI tuner is disabled by the overlay when a coordinated
    tile value is actually applied -- never pre-emptively by platform or policy."""
    rt = RuntimeContext(fps=100.0, total_frames=500, frame_width=640, frame_height=480)
    params = build_engine_params({"apply_tuned_inference": apply_tuned}, runtime=rt)
    assert params["SLICE_TILE_BATCH_AUTOTUNE"] is False  # advanced default
