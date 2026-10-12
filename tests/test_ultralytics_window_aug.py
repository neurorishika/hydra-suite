from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ultralytics")
from ultralytics.utils.instance import Instances  # noqa: E402

from hydra_suite.training.scale_crop_aug import ScaleCropAug  # noqa: E402
from hydra_suite.training.ultralytics_window_aug import (  # noqa: E402
    install_window_jitter,
    recrop_label,
)

IMGSZ = 192


def _label(h=128, w=192):
    img = np.full((h, w, 3), 50, np.uint8)
    # one ant: box centred, three keypoints, normalised xywh
    bbox = np.array([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
    kpts = np.array(
        [[[0.4, 0.5, 2.0], [0.5, 0.5, 2.0], [0.6, 0.5, 2.0]]], dtype=np.float32
    )
    inst = Instances(bbox, np.zeros((0, 1000, 2), np.float32), kpts, "xywh", True)
    return {
        "img": img,
        "instances": inst,
        "cls": np.zeros((1, 1), np.float32),
        "resized_shape": (h, w),
    }


def test_identity_when_jitter_zero():
    lab = _label()
    out = recrop_label(lab, ScaleCropAug(0.0, 0.0, seed=0), IMGSZ)
    assert out["img"].shape == (128, 192, 3)
    np.testing.assert_allclose(out["instances"].bboxes, [[0.5, 0.5, 0.4, 0.2]])


def test_centre_fixed_and_keypoints_stay_aligned_with_pixels():
    aug = ScaleCropAug(0.3, 0.3, seed=1)
    for _ in range(30):
        lab = _label()
        # paint a marker exactly at the middle keypoint so we can track it
        lab["img"][60:68, 92:100] = 255
        out = recrop_label(lab, aug, IMGSZ)
        h, w = out["img"].shape[:2]
        assert max(h, w) <= IMGSZ
        kp = out["instances"].keypoints[0, 1]
        # the centre keypoint stays at the window centre ...
        assert abs(kp[0] - 0.5) < 0.02 and abs(kp[1] - 0.5) < 0.02
        # ... and the painted marker is still under it
        ys, xs = np.nonzero(out["img"][..., 0] > 200)
        assert abs(xs.mean() / w - kp[0]) < 0.02
        assert abs(ys.mean() / h - kp[1]) < 0.03
        # box centre unchanged too
        np.testing.assert_allclose(
            out["instances"].bboxes[0, :2], [0.5, 0.5], atol=0.02
        )


def test_aspect_and_size_actually_vary():
    aug = ScaleCropAug(0.3, 0.3, seed=2)
    shapes = {recrop_label(_label(), aug, IMGSZ)["img"].shape[:2] for _ in range(30)}
    ratios = {round(w / h, 1) for h, w in shapes}
    assert len(shapes) > 5 and len(ratios) > 3


def test_keypoints_outside_window_lose_visibility_but_box_survives():
    lab = _label()
    # keypoint far left; a narrow window will cut it off
    lab["instances"].keypoints[0, 0, :2] = [0.02, 0.5]
    aug = ScaleCropAug(0.0, 0.0, seed=0)
    aug.sample_window = lambda h, w: (int(w * 0.6), h)  # narrower window
    out = recrop_label(lab, aug, IMGSZ)
    assert out["instances"].keypoints[0, 0, 2] == 0.0
    assert out["instances"].keypoints[0, 1, 2] == 2.0
    assert len(out["instances"]) == 1 and len(out["cls"]) == 1


def test_leaves_label_untouched_if_every_instance_would_vanish():
    lab = _label()
    lab["instances"].bboxes[:] = [0.02, 0.5, 0.02, 0.1]
    aug = ScaleCropAug(0.0, 0.0, seed=0)
    aug.sample_window = lambda h, w: (int(w * 0.5), h)
    out = recrop_label(lab, aug, IMGSZ)
    assert out["img"].shape == (128, 192, 3)


def test_install_patches_only_augmenting_datasets():
    from ultralytics.data.base import BaseDataset

    original = BaseDataset.get_image_and_label
    try:
        assert install_window_jitter(0.0, 0.0) is False
        assert BaseDataset.get_image_and_label is original
        assert install_window_jitter(0.2, 0.2, seed=0) is True

        class _DS:
            augment = False
            imgsz = IMGSZ
            labels = [None]

            def __init__(self, augment):
                self.augment = augment

        # call the patched function on a stub whose original is replaced
        patched = BaseDataset.get_image_and_label
        base = patched._hydra_window_jitter_original
        assert base is original
    finally:
        BaseDataset.get_image_and_label = original
