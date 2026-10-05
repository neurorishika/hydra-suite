import cv2
import numpy as np
import pytest

from hydra_suite.training.contracts import AugmentationProfile
from hydra_suite.training.sam3_lora import augment as aug


def _mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    # edge-convention polygon -> raster: shift by -0.5 to pixel-index space
    cv2.fillPoly(m, [np.round((poly - 0.5) * 16).astype(np.int32)], 1, shift=4)
    return m


def _tile(h=40, w=64):
    img = np.zeros((h, w, 3), np.uint8)
    # +0.3 keeps vertices off exact .5 index ties: cv2.fillPoly rounds ties
    # asymmetrically, which makes mirrored masks differ by ~1px (IoU ~0.85) even
    # though the transform is exact.
    poly = np.array([[5, 4], [20, 6], [18, 25], [6, 20]], np.float32) + 0.3
    cv2.fillPoly(
        img, [np.round((poly - 0.5) * 16).astype(np.int32)], (0, 0, 255), shift=4
    )
    return img, poly


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union


@pytest.mark.parametrize(
    "profile",
    [
        AugmentationProfile(enabled=True, fliplr=1.0),
        AugmentationProfile(enabled=True, flipud=1.0),
        AugmentationProfile(enabled=True, rot90=1.0),
        AugmentationProfile(enabled=True, fliplr=1.0, flipud=1.0, rot90=1.0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_geometric_ops_keep_image_and_polygon_aligned(profile, seed):
    img, poly = _tile()
    out, inst = aug.augment_tile(
        img, [(poly, False)], profile, np.random.default_rng(seed), min_area_ratio=0.1
    )
    painted = out[..., 2] > 127
    assert _iou(painted, _mask(inst[0][0], out.shape).astype(bool)) > 0.95


def test_rot90_non_square_tile_masks_align():
    img, poly = _tile(h=40, w=64)
    out, inst = aug.augment_tile(
        img,
        [(poly, False)],
        AugmentationProfile(enabled=True, rot90=1.0),
        np.random.default_rng(0),
        min_area_ratio=0.1,
    )
    assert out.shape[:2] == (64, 40)
    assert _iou(out[..., 2] > 127, _mask(inst[0][0], out.shape).astype(bool)) > 0.95


def test_rotate_uses_constant_fill_not_reflection():
    img = np.full((50, 50, 3), 255, np.uint8)
    out, _ = aug._rotate(img, [], 45.0, 0.1)
    assert (out[0, 0] == (114, 114, 114)).all()


def test_rotate_keeps_alignment_for_interior_instance():
    img, poly = _tile(64, 64)
    poly = poly + 18
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.fillPoly(
        img, [np.round((poly - 0.5) * 16).astype(np.int32)], (0, 0, 255), shift=4
    )
    out, inst = aug.augment_tile(
        img,
        [(poly, False)],
        AugmentationProfile(enabled=True, rotate=30.0),
        np.random.default_rng(1),
        min_area_ratio=0.1,
    )
    assert _iou(out[..., 2] > 127, _mask(inst[0][0], out.shape).astype(bool)) > 0.9
    assert inst[0][1] is False


def test_rotate_marks_sub_floor_fragment_crowd_and_keeps_existing_crowd():
    img = np.zeros((40, 40, 3), np.uint8)
    corner = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    centre = np.array([[15, 15], [25, 15], [25, 25], [15, 25]], np.float32)
    out, inst = aug._rotate(img, [(corner, False), (centre, True)], 45.0, 0.9)
    assert len(inst) == 2
    assert all(flag for _, flag in inst)  # fragment -> crowd; crowd stays crowd
    _, inst_lo = aug._rotate(img, [(centre, False)], 45.0, 0.9)
    assert inst_lo[0][1] is False  # fully retained interior instance stays positive


def test_rotate_drops_instance_fully_outside():
    img = np.zeros((40, 40, 3), np.uint8)
    corner = np.array([[0, 0], [1.5, 0], [1.5, 1.5], [0, 1.5]], np.float32)
    _, inst = aug._rotate(img, [(corner, False)], 45.0, 0.1)
    assert inst == []


def test_photometric_ops_leave_polygons_bit_identical():
    img, poly = _tile()
    profile = AugmentationProfile(
        enabled=True,
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1,
        decode_color_sim=1.0,
        resample_sim=0.0,
    )
    _, inst = aug.augment_tile(
        img, [(poly, False)], profile, np.random.default_rng(0), min_area_ratio=0.1
    )
    np.testing.assert_array_equal(inst[0][0], poly)


def test_brightness_preserves_bgr_channel_order():
    img = np.zeros((8, 8, 3), np.uint8)
    img[..., 2] = 200  # pure red in BGR
    out, _ = aug.augment_tile(
        img,
        [],
        AugmentationProfile(enabled=True, brightness=0.5),
        np.random.default_rng(0),
        min_area_ratio=0.1,
    )
    assert out[..., 0].max() == 0 and out[..., 1].max() == 0 and out[..., 2].min() > 0


def test_same_seed_same_output_different_epoch_differs():
    img, poly = _tile()
    p = aug.recommended_sam3_augmentation()
    a = aug.make_tile_augmenter(p, epoch_seed=7, min_area_ratio=0.1)
    b = aug.make_tile_augmenter(p, epoch_seed=7, min_area_ratio=0.1)
    outs_a = [a(img, [(poly, False)], i)[0] for i in range(8)]
    outs_b = [b(img, [(poly, False)], i)[0] for i in reversed(range(8))][::-1]
    for x, y in zip(outs_a, outs_b):
        np.testing.assert_array_equal(x, y)  # order-independent
    c = aug.make_tile_augmenter(p, epoch_seed=8, min_area_ratio=0.1)
    assert any(
        not np.array_equal(c(img, [(poly, False)], i)[0], outs_a[i]) for i in range(8)
    )


@pytest.mark.parametrize(
    "profile",
    [
        AugmentationProfile(enabled=False, fliplr=1.0),
        AugmentationProfile(enabled=True),
        AugmentationProfile(enabled=False),
    ],
)
def test_inactive_profiles_yield_no_augmenter(profile):
    assert aug.make_tile_augmenter(profile, epoch_seed=0, min_area_ratio=0.1) is None
    assert aug.is_active(profile) is False


def test_recommended_profile_values():
    p = aug.recommended_sam3_augmentation()
    assert (p.enabled, p.fliplr, p.flipud, p.rot90) == (True, 0.5, 0.5, 0.5)
    assert (p.brightness, p.contrast, p.saturation) == (0.2, 0.2, 0.2)
    assert (p.hue, p.rotate, p.decode_color_sim, p.resample_sim, p.monochrome) == (
        0.0,
        0.0,
        0.0,
        0.0,
        False,
    )
    assert aug.recommended_sam3_augmentation() is not p


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"fliplr": 1.5}, "fliplr"),
        ({"rot90": -0.1}, "rot90"),
        ({"rotate": 181.0}, "rotate"),
        ({"hue": 0.6}, "hue"),
        ({"brightness": float("nan")}, "brightness"),
        ({"canonical_aug": True}, "canonical_aug"),
        ({"args": {"mosaic": 1.0}}, "args"),
        ({"label_expansion": {"fliplr": {"a": "b"}}}, "label_expansion"),
    ],
)
def test_validation_rejects(kwargs, fragment):
    errors = aug.validate_sam3_augmentation(AugmentationProfile(enabled=True, **kwargs))
    assert any(fragment in e for e in errors)


def test_validation_accepts_recommended_and_disabled():
    assert aug.validate_sam3_augmentation(aug.recommended_sam3_augmentation()) == []
    assert aug.validate_sam3_augmentation(AugmentationProfile(enabled=False)) == []


def test_stamp_both_arms(tmp_path):
    aug.write_sam3_augmentation_stamp(
        tmp_path / "on", aug.recommended_sam3_augmentation()
    )
    on = aug.read_sam3_augmentation_stamp(tmp_path / "on")
    assert on["applied"]["augmentation"] is True
    assert on["applied"]["ops"]["fliplr"] == 0.5
    aug.write_sam3_augmentation_stamp(
        tmp_path / "off", AugmentationProfile(enabled=False)
    )
    off = aug.read_sam3_augmentation_stamp(tmp_path / "off")
    assert off["applied"] == {"augmentation": False, "ops": {}, "reason": "disabled"}
    aug.write_sam3_augmentation_stamp(
        tmp_path / "zero", AugmentationProfile(enabled=True)
    )
    assert (
        aug.read_sam3_augmentation_stamp(tmp_path / "zero")["applied"]["reason"]
        == "no active ops"
    )
    assert aug.read_sam3_augmentation_stamp(tmp_path / "missing") is None
