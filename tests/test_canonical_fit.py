import numpy as np
import pytest

from hydra_suite.core.canonicalization.fit import (
    apply_fit,
    fit_affine,
    fit_to_model_input,
)


def test_identity_when_source_matches_model():
    f = fit_to_model_input((128, 64), (128, 64))
    assert f.scale == 1.0
    assert f.offset_xy == (0, 0)
    assert f.inner_wh == (128, 64)


def test_scale_is_a_single_scalar_for_both_axes():
    f = fit_to_model_input((128, 64), (256, 256))
    assert f.scale == pytest.approx(2.0)
    assert f.inner_wh == (256, 128)


def test_fit_is_limited_by_the_tighter_axis():
    f = fit_to_model_input((100, 100), (256, 64))
    assert f.scale == pytest.approx(0.64)
    assert f.inner_wh == (64, 64)


def test_content_is_centred():
    f = fit_to_model_input((256, 128), (256, 256))
    assert f.offset_xy == (0, 64)


@pytest.mark.parametrize(
    "source_wh,model_wh",
    [((128, 64), (256, 256)), ((64, 128), (256, 256)), ((300, 50), (128, 128))],
)
def test_aspect_ratio_is_preserved(source_wh, model_wh):
    f = fit_to_model_input(source_wh, model_wh)
    src_ar = source_wh[0] / source_wh[1]
    out_ar = f.inner_wh[0] / f.inner_wh[1]
    assert out_ar == pytest.approx(src_ar, rel=0.02)


def test_apply_fit_pads_with_zeros_and_returns_uint8():
    img = np.full((64, 128, 3), 200, dtype=np.uint8)
    f = fit_to_model_input((128, 64), (256, 256))
    out = apply_fit(img, f)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 0  # padded band
    assert int(out[128, 128, 0]) > 100  # content


def test_apply_fit_is_deterministic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
    f = fit_to_model_input((128, 64), (96, 96))
    np.testing.assert_array_equal(apply_fit(img, f), apply_fit(img, f))


def test_fit_affine_round_trips_a_point():
    import cv2

    f = fit_to_model_input((128, 64), (256, 256))
    m = fit_affine(f)
    inv = cv2.invertAffineTransform(m)
    pt = np.array([37.0, 21.0, 1.0])
    mapped = m @ pt
    back = inv @ np.array([mapped[0], mapped[1], 1.0])
    np.testing.assert_allclose(back, pt[:2], atol=1e-6)


def test_apply_fit_nonsquare_shape_and_dtype():
    crop = np.random.default_rng(0).integers(0, 256, (56, 112, 3), np.uint8)
    fit = fit_to_model_input((112, 56), (64, 128))
    out = apply_fit(crop, fit)
    assert out.shape == (128, 64, 3) and out.dtype == np.uint8
