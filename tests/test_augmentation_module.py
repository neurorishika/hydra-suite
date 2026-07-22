"""Unit tests for the decode/resample training augmentations."""

import numpy as np

from hydra_suite.training.augmentation import (
    make_decode_resample_pil_transform,
    simulate_decode_color,
    simulate_resample,
)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(32, 24, 3), dtype=np.uint8)


def test_decode_color_shape_dtype_preserved():
    out = simulate_decode_color(_img(), 1.0, np.random.default_rng(1))
    assert out.shape == (32, 24, 3) and out.dtype == np.uint8


def test_decode_color_identity_at_zero_strength():
    img = _img()
    out = simulate_decode_color(img, 0.0, np.random.default_rng(1))
    assert np.array_equal(out, img)


def test_decode_color_changes_pixels_when_applied():
    img = _img()
    # strength 1.0 => always applies; a sampled non-BT601-limited convention shifts color.
    # Try several rng seeds; at least one must change pixels (identity convention is possible).
    changed = any(
        not np.array_equal(
            simulate_decode_color(img, 1.0, np.random.default_rng(s)), img
        )
        for s in range(8)
    )
    assert changed


def test_decode_color_deterministic_under_fixed_rng():
    img = _img()
    a = simulate_decode_color(img, 1.0, np.random.default_rng(42))
    b = simulate_decode_color(img, 1.0, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_decode_color_no_overflow():
    out = simulate_decode_color(
        np.full((8, 8, 3), 255, np.uint8), 1.0, np.random.default_rng(3)
    )
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255
    assert not np.isnan(out.astype(np.float32)).any()


def test_resample_shape_dtype_preserved():
    out = simulate_resample(_img(), 1.0, np.random.default_rng(1))
    assert out.shape == (32, 24, 3) and out.dtype == np.uint8


def test_resample_identity_at_zero_prob():
    img = _img()
    assert np.array_equal(simulate_resample(img, 0.0, np.random.default_rng(1)), img)


def test_resample_deterministic_under_fixed_rng():
    img = _img()
    a = simulate_resample(img, 1.0, np.random.default_rng(7))
    b = simulate_resample(img, 1.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_pil_transform_none_when_both_off():
    prof = type("P", (), {"decode_color_sim": 0.0, "resample_sim": 0.0})()
    assert make_decode_resample_pil_transform(prof, np.random.default_rng(0)) is None


def test_pil_transform_roundtrips_pil_image():
    from PIL import Image

    prof = type("P", (), {"decode_color_sim": 1.0, "resample_sim": 1.0})()
    tf = make_decode_resample_pil_transform(prof, np.random.default_rng(0))
    assert tf is not None
    img = Image.fromarray(_img())
    out = tf(img)
    assert isinstance(out, Image.Image) and out.size == img.size


def test_pil_transform_inserts_before_totensor_shape_ok():
    """The PIL transform must accept the post-Resize PIL image and return a PIL
    image ToTensor can consume."""
    from PIL import Image
    from torchvision import transforms

    prof = type("P", (), {"decode_color_sim": 1.0, "resample_sim": 1.0})()
    tf = make_decode_resample_pil_transform(prof, np.random.default_rng(0))
    pipeline = transforms.Compose(
        [transforms.Resize((16, 16)), tf, transforms.ToTensor()]
    )
    out = pipeline(Image.fromarray(np.zeros((20, 20, 3), np.uint8)))
    assert tuple(out.shape) == (3, 16, 16)
