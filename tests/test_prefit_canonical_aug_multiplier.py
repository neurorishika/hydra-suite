"""Offline epoch-multiplied prefit for YOLO-classify CanonicalAug."""

import cv2
import numpy as np

from hydra_suite.training.contracts import AugmentationProfile
from hydra_suite.training.runner import _prefit_yolo_classify_dataset


def _make_src(root, n_classes=2, per_class=2, hw=(40, 90)):
    """Non-square crops so we can prove aug ran before the square letterbox."""
    h, w = hw
    for c in range(n_classes):
        d = root / "train" / f"cls{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            img = (
                np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3) + c * 7 + i
            ) % 255
            cv2.imwrite(str(d / f"img{i}.png"), img.astype(np.uint8))
    return root


def test_off_is_byte_identical_to_clean_prefit(tmp_path):
    src = _make_src(tmp_path / "src")
    dest_a = tmp_path / "out_none"
    dest_b = tmp_path / "out_off"
    # profile=None and an explicitly-off profile must both yield clean-only.
    _prefit_yolo_classify_dataset(src, 64, dest_a, profile=None, seed=42)
    _prefit_yolo_classify_dataset(
        src,
        64,
        dest_b,
        profile=AugmentationProfile(canonical_aug=False, canonical_aug_copies=3),
        seed=42,
    )
    files_a = sorted(p.name for p in (dest_a / "train" / "cls0").iterdir())
    files_b = sorted(p.name for p in (dest_b / "train" / "cls0").iterdir())
    assert files_a == files_b == ["img0.png", "img1.png"]  # no .aug* files
    a = cv2.imread(str(dest_a / "train" / "cls0" / "img0.png"))
    b = cv2.imread(str(dest_b / "train" / "cls0" / "img0.png"))
    np.testing.assert_array_equal(a, b)


def test_on_writes_clean_plus_k_augmented(tmp_path):
    src = _make_src(tmp_path / "src")
    dest = tmp_path / "out_on"
    _prefit_yolo_classify_dataset(
        src,
        64,
        dest,
        profile=AugmentationProfile(canonical_aug=True, canonical_aug_copies=3),
        seed=42,
    )
    names = sorted(p.name for p in (dest / "train" / "cls0").iterdir())
    # per source image: 1 clean + 3 augmented
    assert names == [
        "img0.aug1.png",
        "img0.aug2.png",
        "img0.aug3.png",
        "img0.png",
        "img1.aug1.png",
        "img1.aug2.png",
        "img1.aug3.png",
        "img1.png",
    ]
    # every output is the square model input (letterbox ran on all copies)
    for n in names:
        out = cv2.imread(str(dest / "train" / "cls0" / n))
        assert out.shape[:2] == (64, 64)
    # augmented differs from clean (aug had an effect)
    clean = cv2.imread(str(dest / "train" / "cls0" / "img0.png"))
    aug1 = cv2.imread(str(dest / "train" / "cls0" / "img0.aug1.png"))
    assert not np.array_equal(clean, aug1)


def test_prefit_is_reproducible_for_fixed_seed(tmp_path):
    src = _make_src(tmp_path / "src")
    d1 = tmp_path / "r1"
    d2 = tmp_path / "r2"
    prof = AugmentationProfile(canonical_aug=True, canonical_aug_copies=2)
    _prefit_yolo_classify_dataset(src, 64, d1, profile=prof, seed=7)
    _prefit_yolo_classify_dataset(src, 64, d2, profile=prof, seed=7)
    a = cv2.imread(str(d1 / "train" / "cls0" / "img0.aug1.png"))
    b = cv2.imread(str(d2 / "train" / "cls0" / "img0.aug1.png"))
    np.testing.assert_array_equal(a, b)


def test_aug_receives_prefit_crop_not_letterboxed(tmp_path, monkeypatch):
    """The aug must see the raw non-square crop (before Layer-2), proving order."""
    src = _make_src(tmp_path / "src", n_classes=1, per_class=1, hw=(40, 90))
    seen_shapes = []

    import hydra_suite.training.canonical_aug as canon_mod

    class _SpyAug:
        def __init__(self, *a, **k):
            pass

        def __call__(self, img):
            seen_shapes.append(img.shape[:2])
            return img  # passthrough

    monkeypatch.setattr(canon_mod, "CanonicalAug", _SpyAug)
    _prefit_yolo_classify_dataset(
        src,
        64,
        tmp_path / "out",
        profile=AugmentationProfile(canonical_aug=True, canonical_aug_copies=1),
        seed=1,
    )
    assert seen_shapes and all(s == (40, 90) for s in seen_shapes)
