"""Regression tests for background-subtraction cache keys."""

from hydra_suite.core.inference.cache.keys import bgsub_detection_cache_key
from hydra_suite.core.inference.config import BgSubConfig


def _base_params() -> dict:
    return {
        "THRESHOLD_VALUE": 20,
        "DARK_ON_LIGHT_BACKGROUND": True,
        "ENABLE_CONSERVATIVE_SPLIT": False,
        "ENABLE_ADAPTIVE_BACKGROUND": True,
        "BACKGROUND_LEARNING_RATE": 0.001,
        "BACKGROUND_PRIME_FRAMES": 30,
        "ENABLE_SIZE_FILTERING": False,
        "MIN_OBJECT_SIZE": 0,
        "MAX_OBJECT_SIZE": 10000,
        "ENABLE_ASPECT_RATIO_FILTERING": False,
        "BRIGHTNESS": 0,
        "CONTRAST": 1.0,
        "GAMMA": 1.0,
        "ENABLE_LIGHTING_STABILIZATION": False,
        "MORPH_KERNEL_SIZE": 5,
        "DILATION_KERNEL_SIZE": 3,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "START_FRAME": 0,
        "END_FRAME": 500,
        "RESIZE_FACTOR": 1.0,
        "BACKGROUND_CONVERGENCE_EPSILON": 1e-4,
        "BACKGROUND_CONVERGENCE_FRAMES": 30,
        "BACKGROUND_CONVERGENCE_PIXEL_DELTA": 5.0,
    }


def test_threshold_change_invalidates_cache_key():
    """THRESHOLD_VALUE is the most important bg-sub param; it must be keyed."""
    a = bgsub_detection_cache_key(BgSubConfig.from_params(_base_params()))
    p = _base_params()
    p["THRESHOLD_VALUE"] = 40
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_prime_frames_change_invalidates_cache_key():
    a = bgsub_detection_cache_key(BgSubConfig.from_params(_base_params()))
    p = _base_params()
    p["BACKGROUND_PRIME_FRAMES"] = 60
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_identical_params_produce_identical_key():
    assert (
        bgsub_detection_cache_key(BgSubConfig.from_params(_base_params())).config_hash
        == bgsub_detection_cache_key(
            BgSubConfig.from_params(_base_params())
        ).config_hash
    )


def test_key_params_all_exist_in_codebase_naming():
    """Guard against re-introducing param names nothing else uses."""
    from hydra_suite.core.inference.cache.keys import _BGSUB_KEY_PARAMS

    assert "SUBTRACTION_THRESHOLD" not in _BGSUB_KEY_PARAMS
    assert "BACKGROUND_PRIME_SECONDS" not in _BGSUB_KEY_PARAMS
    assert "THRESHOLD_VALUE" in _BGSUB_KEY_PARAMS
    assert "BACKGROUND_PRIME_FRAMES" in _BGSUB_KEY_PARAMS


def test_cache_key_accepts_bgsub_config():
    cfg = BgSubConfig.from_params(_base_params())
    key = bgsub_detection_cache_key(cfg)
    assert key.model_path == "background_subtraction"


def test_convergence_epsilon_is_keyed():
    p = _base_params()
    p["BACKGROUND_CONVERGENCE_EPSILON"] = 1e-4
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["BACKGROUND_CONVERGENCE_EPSILON"] = 5e-1
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_convergence_frames_are_keyed():
    p = _base_params()
    p["BACKGROUND_CONVERGENCE_FRAMES"] = 30
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["BACKGROUND_CONVERGENCE_FRAMES"] = 60
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_convergence_pixel_delta_is_keyed():
    p = _base_params()
    p["BACKGROUND_CONVERGENCE_PIXEL_DELTA"] = 5.0
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["BACKGROUND_CONVERGENCE_PIXEL_DELTA"] = 10.0
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_lighting_smooth_factor_is_keyed():
    p = _base_params()
    p["LIGHTING_SMOOTH_FACTOR"] = 0.95
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["LIGHTING_SMOOTH_FACTOR"] = 0.5
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash


def test_lighting_median_window_is_keyed():
    p = _base_params()
    p["LIGHTING_MEDIAN_WINDOW"] = 5
    a = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    p["LIGHTING_MEDIAN_WINDOW"] = 11
    b = bgsub_detection_cache_key(BgSubConfig.from_params(p))
    assert a.config_hash != b.config_hash
