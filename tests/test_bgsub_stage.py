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


from hydra_suite.core.background.measure import BackgroundMeasurer


def _measure_params(**overrides) -> dict:
    p = {
        "MAX_TARGETS": 10,
        "MIN_CONTOUR_AREA": 5,
        "MAX_CONTOUR_MULTIPLIER": 20,
        "ENABLE_SIZE_FILTERING": False,
        "MIN_OBJECT_SIZE": 0,
        "MAX_OBJECT_SIZE": float("inf"),
        "THRESHOLD_VALUE": 20,
        "CONSERVATIVE_KERNEL_SIZE": 3,
        "CONSERVATIVE_ERODE_ITER": 1,
        "RESIZE_FACTOR": 1.0,
    }
    p.update(overrides)
    return p


def _mask_with_ellipse() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.ellipse(mask, (32, 32), (12, 6), 30, 0, 360, 255, -1)
    return mask


def test_detect_objects_returns_four_tuple_without_yolo_stub():
    measurer = BackgroundMeasurer(_measure_params())
    result = measurer.detect_objects(_mask_with_ellipse(), 0)
    assert len(result) == 4
    meas, sizes, shapes, confidences = result
    assert len(meas) == 1
    assert len(sizes) == 1
    assert len(shapes) == 1
    assert len(confidences) == 1


def test_detect_objects_confidence_is_nan():
    measurer = BackgroundMeasurer(_measure_params())
    _, _, _, confidences = measurer.detect_objects(_mask_with_ellipse(), 0)
    assert np.isnan(confidences[0])


def test_detect_objects_angle_is_radians():
    measurer = BackgroundMeasurer(_measure_params())
    meas, _, _, _ = measurer.detect_objects(_mask_with_ellipse(), 0)
    assert 0.0 <= float(meas[0][2]) <= np.pi


def test_too_many_contours_returns_empty_four_tuple():
    measurer = BackgroundMeasurer(
        _measure_params(MAX_TARGETS=1, MAX_CONTOUR_MULTIPLIER=1)
    )
    mask = np.zeros((64, 64), dtype=np.uint8)
    for x in range(4, 60, 8):
        for y in range(4, 60, 8):
            cv2.circle(mask, (x, y), 2, 255, -1)
    assert measurer.detect_objects(mask, 0) == ([], [], [], [])


from hydra_suite.core.background.measure import corners_from_ellipse


def test_corners_from_ellipse_axis_aligned_order_is_tl_tr_br_bl():
    """Order must match _corners_from_xywhr (stages/obb.py:249). Wrong order
    historically put SLEAP ~86 px off."""
    corners = corners_from_ellipse(10.0, 20.0, 8.0, 4.0, 0.0)
    assert corners.shape == (4, 2)
    expected = np.array(
        [[6.0, 18.0], [14.0, 18.0], [14.0, 22.0], [6.0, 22.0]], dtype=np.float32
    )
    np.testing.assert_allclose(corners, expected, atol=1e-4)


def test_corners_from_ellipse_rotated_90_degrees():
    corners = corners_from_ellipse(0.0, 0.0, 8.0, 4.0, np.pi / 2)
    # Major axis now vertical: bounding corners swap extents.
    assert np.isclose(np.abs(corners[:, 0]).max(), 2.0, atol=1e-4)
    assert np.isclose(np.abs(corners[:, 1]).max(), 4.0, atol=1e-4)


def test_corners_from_ellipse_centroid_is_mean_of_corners():
    corners = corners_from_ellipse(5.0, 7.0, 10.0, 3.0, 0.7)
    np.testing.assert_allclose(corners.mean(axis=0), [5.0, 7.0], atol=1e-4)


from hydra_suite.core.inference.config import BgSubConfig


def test_bgsub_config_from_params_reads_legacy_keys():
    cfg = BgSubConfig.from_params(
        {"THRESHOLD_VALUE": 42, "BACKGROUND_PRIME_FRAMES": 99}
    )
    assert cfg.threshold_value == 42.0
    assert cfg.background_prime_frames == 99
    assert cfg.convergence_epsilon == 0.05  # default


def test_bgsub_config_retains_raw_params():
    cfg = BgSubConfig.from_params({"THRESHOLD_VALUE": 42, "CUSTOM": "x"})
    assert cfg.params["CUSTOM"] == "x"


from hydra_suite.core.inference.config import (
    InferenceConfig,
    InferenceConfigError,
    OBBConfig,
)


def test_config_requires_exactly_one_detection_source():
    with pytest.raises(InferenceConfigError, match="exactly one"):
        InferenceConfig(obb=None, bgsub=None)

    with pytest.raises(InferenceConfigError, match="exactly one"):
        InferenceConfig(obb=OBBConfig(), bgsub=BgSubConfig.from_params({}))


def test_config_detection_source_reports_bgsub():
    cfg = InferenceConfig(obb=None, bgsub=BgSubConfig.from_params({}))
    assert cfg.detection_source == "bgsub"


def test_config_detection_source_reports_obb():
    cfg = InferenceConfig(obb=OBBConfig())
    assert cfg.detection_source == "obb"
