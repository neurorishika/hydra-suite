"""Tests for the background-subtraction inference stage."""

import cv2
import numpy as np
import pytest

from hydra_suite.core.background.model import BackgroundModel


@pytest.fixture
def synthetic_video(tmp_path):
    """A 60-frame 64x64 video with a moving dark blob on a light background."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64), True
    )
    for i in range(60):
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)
        cx = 8 + i
        if cx < 56:
            cv2.circle(frame, (cx, 32), 4, (30, 30, 30), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def _params(**overrides) -> dict:
    p = {
        "BACKGROUND_PRIME_FRAMES": 20,
        "BRIGHTNESS": 0,
        "CONTRAST": 1.0,
        "GAMMA": 1.0,
        "RESIZE_FACTOR": 1.0,
        "THRESHOLD_VALUE": 20,
        "DARK_ON_LIGHT_BACKGROUND": True,
        "ENABLE_ADAPTIVE_BACKGROUND": True,
        "BACKGROUND_LEARNING_RATE": 0.001,
        "MORPH_KERNEL_SIZE": 3,
        "ENABLE_GPU_BACKGROUND": False,
    }
    p.update(overrides)
    return p


def test_priming_is_deterministic(synthetic_video):
    """Same video + same params must produce a byte-identical background.

    This is the property that makes the bg-sub cache key honest.
    """
    backgrounds = []
    for _ in range(2):
        model = BackgroundModel(_params())
        cap = cv2.VideoCapture(synthetic_video)
        model.prime_background(cap)
        cap.release()
        backgrounds.append(model.lightest_background.copy())

    np.testing.assert_array_equal(backgrounds[0], backgrounds[1])


def test_priming_covers_video_temporally(synthetic_video):
    """Evenly-spaced sampling must span the whole video, not cluster."""
    model = BackgroundModel(_params(BACKGROUND_PRIME_FRAMES=10))
    cap = cv2.VideoCapture(synthetic_video)
    model.prime_background(cap)
    cap.release()
    # The blob traverses the frame; a background spanning the video is the
    # light plate everywhere, so its minimum stays near the plate value.
    assert model.lightest_background is not None
    assert float(model.lightest_background.min()) > 150.0


def test_adaptive_disabled_never_switches_to_frozen_snapshot():
    """ENABLE_ADAPTIVE_BACKGROUND=False must mean 'do not switch', not
    'switch to a stale primed snapshot'."""
    model = BackgroundModel(
        _params(
            ENABLE_ADAPTIVE_BACKGROUND=False,
            BACKGROUND_CONVERGENCE_EPSILON=0.05,
            BACKGROUND_CONVERGENCE_FRAMES=1,
        )
    )
    gray = np.full((16, 16), 200, dtype=np.uint8)
    model.update_and_get_background(gray, None)
    model.update_and_get_background(gray, None)
    model.update_and_get_background(gray, None)
    assert model.stabilized

    result = model.update_and_get_background(gray, None)
    np.testing.assert_array_equal(
        result, cv2.convertScaleAbs(model.lightest_background)
    )


def test_convergence_latch_sets_when_lightest_stops_growing():
    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=0.05,
            BACKGROUND_CONVERGENCE_FRAMES=3,
        )
    )
    gray = np.full((16, 16), 200, dtype=np.uint8)
    model.update_and_get_background(gray, None)  # first frame primes, returns None
    assert not model.stabilized

    for _ in range(3):
        model.update_and_get_background(gray, None)
    assert model.stabilized


def test_convergence_latch_resets_counter_when_background_grows():
    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=0.05,
            BACKGROUND_CONVERGENCE_FRAMES=3,
        )
    )
    model.update_and_get_background(np.full((16, 16), 200, np.uint8), None)
    model.update_and_get_background(np.full((16, 16), 200, np.uint8), None)
    model.update_and_get_background(np.full((16, 16), 200, np.uint8), None)
    # A brighter frame grows the running max -> counter resets.
    model.update_and_get_background(np.full((16, 16), 250, np.uint8), None)
    assert not model.stabilized


def test_convergence_latch_is_monotonic():
    """Once latched, never un-latches, even if the background grows again."""
    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=0.05,
            BACKGROUND_CONVERGENCE_FRAMES=2,
        )
    )
    gray = np.full((16, 16), 200, dtype=np.uint8)
    for _ in range(4):
        model.update_and_get_background(gray, None)
    assert model.stabilized

    model.update_and_get_background(np.full((16, 16), 255, np.uint8), None)
    assert model.stabilized
