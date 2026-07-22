from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig
from hydra_suite.trackerkit.gui.workers.preview_worker import (
    _preview_build_bgsub_params,
)


def _ctx():
    return {
        "fps": 30.0,
        "bg_prime_frames": 30,
        "threshold_value": 25,
        "min_contour": 40,
        "max_targets": 7,
        "resize_factor": 0.5,
        "min_object_size": 0.3,
        "max_object_size": 3.0,
        "reference_body_size": 20.0,
        "runtime_tier": "cpu",
    }


def test_filters_off_disables_size_filtering():
    params = _preview_build_bgsub_params(_ctx(), use_detection_filters=False)
    assert params["ENABLE_SIZE_FILTERING"] is False
    # MIN_CONTOUR_AREA always survives (matches production + old preview).
    assert params["MIN_CONTOUR_AREA"] == 40


def test_filters_on_enables_size_filtering():
    params = _preview_build_bgsub_params(_ctx(), use_detection_filters=True)
    assert params["ENABLE_SIZE_FILTERING"] is True
    assert params["MAX_TARGETS"] == 7


def test_runtime_tier_defaults_to_cpu_when_unknown():
    ctx = _ctx()
    ctx["runtime_tier"] = "bogus"
    params = _preview_build_bgsub_params(ctx, use_detection_filters=True)
    assert params["RUNTIME_TIER"] == "cpu"


def test_builds_a_valid_bgsub_inference_config():
    params = _preview_build_bgsub_params(_ctx(), use_detection_filters=True)
    cfg = InferenceConfig(
        obb=None,
        bgsub=BgSubConfig.from_params(params),
        runtime_tier=params["RUNTIME_TIER"],
    )
    assert cfg.detection_source == "bgsub"
