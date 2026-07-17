"""Regression tests for background-subtraction cache keys."""

from hydra_suite.core.inference.cache.keys import bgsub_detection_cache_key


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
    }


def test_threshold_change_invalidates_cache_key():
    """THRESHOLD_VALUE is the most important bg-sub param; it must be keyed."""
    a = bgsub_detection_cache_key(_base_params())
    p = _base_params()
    p["THRESHOLD_VALUE"] = 40
    b = bgsub_detection_cache_key(p)
    assert a.config_hash != b.config_hash


def test_prime_frames_change_invalidates_cache_key():
    a = bgsub_detection_cache_key(_base_params())
    p = _base_params()
    p["BACKGROUND_PRIME_FRAMES"] = 60
    b = bgsub_detection_cache_key(p)
    assert a.config_hash != b.config_hash


def test_identical_params_produce_identical_key():
    assert (
        bgsub_detection_cache_key(_base_params()).config_hash
        == bgsub_detection_cache_key(_base_params()).config_hash
    )


def test_key_params_all_exist_in_codebase_naming():
    """Guard against re-introducing param names nothing else uses."""
    from hydra_suite.core.inference.cache.keys import _BGSUB_KEY_PARAMS

    assert "SUBTRACTION_THRESHOLD" not in _BGSUB_KEY_PARAMS
    assert "BACKGROUND_PRIME_SECONDS" not in _BGSUB_KEY_PARAMS
    assert "THRESHOLD_VALUE" in _BGSUB_KEY_PARAMS
    assert "BACKGROUND_PRIME_FRAMES" in _BGSUB_KEY_PARAMS
