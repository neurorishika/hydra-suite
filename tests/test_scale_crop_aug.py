from __future__ import annotations

import cv2
import numpy as np

from hydra_suite.training.contracts import AugmentationProfile
from hydra_suite.training.runner import (
    _build_scale_crop_aug,
    _prefit_yolo_classify_dataset,
)
from hydra_suite.training.scale_crop_aug import ScaleCropAug


def _img(h=40, w=90):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_zero_jitter_is_identity():
    img = _img()
    out = ScaleCropAug(0.0, 0.0, seed=1)(img)
    np.testing.assert_array_equal(out, img)


def test_dtype_preserved_and_shape_changes_only_when_enabled():
    img = _img()
    out = ScaleCropAug(0.3, 0.3, seed=1)(img)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3


def test_same_seed_is_deterministic_and_different_seed_differs():
    img = _img()
    a = ScaleCropAug(0.3, 0.3, seed=7)(img)
    b = ScaleCropAug(0.3, 0.3, seed=7)(img)
    c = ScaleCropAug(0.3, 0.3, seed=8)(img)
    np.testing.assert_array_equal(a, b)
    assert a.shape != c.shape or not np.array_equal(a, c)


def test_aspect_jitter_changes_aspect_ratio_at_constant_area():
    img = _img(60, 120)
    aug = ScaleCropAug(0.0, 0.3, seed=0)
    ratios, areas = [], []
    for _ in range(60):
        h, w = aug(img).shape[:2]
        ratios.append(w / h)
        areas.append(w * h)
    base = 120 / 60
    assert min(ratios) < base * 0.9 and max(ratios) > base * 1.1
    assert all(base / 1.31 <= r <= base * 1.31 for r in ratios)
    assert max(areas) / min(areas) < 1.03  # window area is held constant


def test_scale_jitter_keeps_aspect_and_resizes_window():
    img = _img(60, 120)
    aug = ScaleCropAug(0.4, 0.0, seed=0)
    shapes = [aug(img).shape[:2] for _ in range(40)]
    assert min(h for h, _ in shapes) < 60 < max(h for h, _ in shapes)
    assert all(abs(w / h - 2.0) < 0.1 for h, w in shapes)


def test_centre_is_never_moved_and_animal_not_stretched():
    img = np.zeros((60, 120, 3), np.uint8)
    img[25:35, 55:65] = 255  # 10x10 square centred at (59.5, 29.5)
    aug = ScaleCropAug(0.4, 0.4, seed=3)
    for _ in range(40):
        out = aug(img)
        m = out[..., 0] > 128
        assert m.sum() == 100  # pure crop/pad: pixels are not rescaled
        ys, xs = np.nonzero(m)
        h, w = out.shape[:2]
        assert abs(xs.mean() - (w - 1) / 2.0) <= 1.0
        assert abs(ys.mean() - (h - 1) / 2.0) <= 1.0


def test_builder_gating():
    assert _build_scale_crop_aug(None, 1) is None
    assert _build_scale_crop_aug(AugmentationProfile(enabled=True), 1) is None
    assert (
        _build_scale_crop_aug(AugmentationProfile(enabled=False, scale_jitter=0.2), 1)
        is None
    )
    assert (
        _build_scale_crop_aug(AugmentationProfile(enabled=True, aspect_jitter=0.1), 1)
        is not None
    )


def _make_src(root):
    for split in ("train", "val"):
        d = root / split / "cls0"
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "a.png"), _img())
    return root


def test_prefit_writes_augmented_copies_for_train_only(tmp_path):
    src = _make_src(tmp_path / "src")
    dest = tmp_path / "out"
    _prefit_yolo_classify_dataset(
        src,
        64,
        dest,
        profile=AugmentationProfile(scale_jitter=0.2, canonical_aug_copies=2),
        seed=3,
    )
    assert sorted(p.name for p in (dest / "train" / "cls0").iterdir()) == [
        "a.aug1.png",
        "a.aug2.png",
        "a.png",
    ]
    assert [p.name for p in (dest / "val" / "cls0").iterdir()] == ["a.png"]
