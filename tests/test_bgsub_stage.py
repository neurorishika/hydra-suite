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


def test_convergence_latch_scale_invariance_large_frame():
    """A mean-delta convergence metric is frame-size dependent: the same
    per-frame revealing event that swamps a small test frame's mean is far
    below any usable threshold at production resolutions, so a mean-based
    latch fires while the animal is still mid-reveal of its resting
    footprint. This test uses a realistically large 512x512 frame with an
    animal (~2000px resting patch, "a couple animal-widths") walking away in
    small steps, so the reveal is spread over many frames rather than
    completing in a single jump.

    This intentionally does NOT pass BACKGROUND_CONVERGENCE_EPSILON, relying
    on `_update_convergence`'s own default -- so this test tracks whichever
    metric (and whichever default) is actually wired up. Only
    BACKGROUND_CONVERGENCE_FRAMES is pinned, to bound how long the test runs.

    Under the old whole-frame mean-delta metric (default epsilon 0.05),
    delta stays ~0.028 every single frame -- below that epsilon -- so the
    old code latches after exactly `needed` frames (frame 8), less than
    half-way through the ~17-frame reveal. Under the scale-invariant
    changed-pixel-fraction metric (default epsilon 1e-4), the fraction of
    still-growing pixels (~5.6e-4) stays far above epsilon for the entire
    reveal, so the latch correctly does not fire until well after frame 20.
    """
    size = 512
    background_value = 200
    blob_value = 150
    radius = 25  # area ~ pi*25^2 =~ 1963px resting patch
    step = 3  # small per-frame translation: reveal spreads over ~17 frames

    model = BackgroundModel(_params(BACKGROUND_CONVERGENCE_FRAMES=8))

    def frame_at(cx):
        img = np.full((size, size), background_value, dtype=np.uint8)
        cv2.circle(img, (cx, size // 2), radius, blob_value, -1)
        return img

    # First call just primes the model: the animal has been resting here,
    # so its whole footprint is dark from frame zero.
    model.update_and_get_background(frame_at(80), None)
    assert not model.stabilized

    # Walk the animal away in small steps. Each step uncovers only a thin
    # trailing sliver of the resting footprint (never-before-revealed
    # background), so the reveal is still in progress through frame 20
    # (full reveal completes around frame ~17, at 3px/frame over a 50px
    # diameter). The latch must not fire while any of that is still
    # unrevealed.
    for i in range(1, 21):
        model.update_and_get_background(frame_at(80 + i * step), None)
        assert not model.stabilized, (
            f"latched after step {i} while the animal was still revealing "
            "its resting footprint -- convergence metric is not "
            "scale-invariant"
        )


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


def test_convergence_latch_survives_sensor_noise():
    """BACKGROUND_CONVERGENCE_PIXEL_DELTA must exceed the sensor noise floor.

    `lightest_background` is a running max, so under Gaussian sensor noise it
    never truly stops growing: every frame, noise pushes some pixels above
    the previous max. If PIXEL_DELTA sits inside the noise (e.g. 1.0 grey
    level for sd=2.0 noise), the "still growing" fraction plateaus at a
    noise-dependent floor above epsilon and the latch never fires -- the
    model then never switches to adaptive and silently loses lighting-drift
    tracking. PIXEL_DELTA=5.0 clears the noise floor while staying far below
    a genuine animal reveal (~50-150 grey levels), so the latch fires.
    """
    rng = np.random.default_rng(0)
    frames = [
        np.clip(rng.normal(200, 2.0, (256, 256)), 0, 255).astype(np.uint8)
        for _ in range(150)
    ]

    model = BackgroundModel(
        _params(
            BACKGROUND_CONVERGENCE_EPSILON=1e-4,
            BACKGROUND_CONVERGENCE_FRAMES=10,
            BACKGROUND_CONVERGENCE_PIXEL_DELTA=5.0,
        )
    )
    for gray in frames:
        model.update_and_get_background(gray, None)

    assert model.stabilized, (
        "background never latched under realistic sensor noise -- "
        "BACKGROUND_CONVERGENCE_PIXEL_DELTA is too close to the noise floor"
    )
