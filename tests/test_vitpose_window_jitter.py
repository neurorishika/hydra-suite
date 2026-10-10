from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from hydra_suite.core.individual.pose.vitpose.training.config import (
    validate_run_config,
)
from hydra_suite.core.individual.pose.vitpose.training.dataset import (
    CocoKeypointsDataset,
    load_coco_index,
)
from hydra_suite.core.individual.pose.vitpose.training.window_jitter import (
    WindowJitter,
)
from hydra_suite.training.scale_crop_aug import ScaleCropAug


def test_window_sampling_matches_classifier_augmenter_draw_for_draw():
    a = ScaleCropAug(0.3, 0.4, seed=11)
    b = WindowJitter(0.3, 0.4, seed=11)
    for h, w in [(128, 192), (60, 120), (200, 90)] * 20:
        assert a.sample_window(h, w) == b.sample_window(h, w)


def test_recrop_keeps_centre_keypoint_on_its_pixel_and_flags_outside():
    img = np.zeros((60, 120, 3), np.uint8)
    img[28:32, 58:62] = 255  # marker at (59.5, 29.5)
    kp = np.array([[59.5, 29.5, 2.0], [1.0, 30.0, 2.0]], np.float32)
    wj = WindowJitter(0.4, 0.4, seed=3)
    for _ in range(40):
        out, kp2 = wj.recrop(img, kp)
        h, w = out.shape[:2]
        ys, xs = np.nonzero(out[..., 0] > 200)
        if len(xs):
            assert abs(xs.mean() - kp2[0, 0]) < 1.5
            assert abs(ys.mean() - kp2[0, 1]) < 1.5
        assert abs(kp2[0, 0] - (w - 1) / 2) < 1.5  # centre never moved
        inside = 0 <= kp2[1, 0] <= w - 1 and 0 <= kp2[1, 1] <= h - 1
        assert (kp2[1, 2] == 2.0) == inside


def test_inactive_is_identity():
    img = np.zeros((10, 10, 3), np.uint8)
    kp = np.ones((2, 3), np.float32)
    out, kp2 = WindowJitter(0.0, 0.0).recrop(img, kp)
    assert out is img and kp2 is kp


def _make_ds_dir(tmp_path, k=3, hw=(60, 120)):
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "f0.png"), np.full((*hw, 3), 127, np.uint8))
    kpts = []
    for j in range(k):
        kpts += [40 + 15 * j, 30, 2]
    coco = {
        "images": [{"id": 1, "file_name": "f0.png", "width": hw[1], "height": hw[0]}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 90.0, 40.0],
                "area": 3600.0,
                "iscrowd": 0,
                "num_keypoints": k,
                "keypoints": kpts,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "a",
                "keypoints": [f"k{j}" for j in range(k)],
                "skeleton": [],
            }
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    return tmp_path


def test_dataset_window_mode_is_deterministic_per_seed_and_varies(tmp_path):
    d = _make_ds_dir(tmp_path)
    ids, _ = load_coco_index(d)

    def centres(seed):
        np.random.seed(0)
        ds = CocoKeypointsDataset(
            d, ids, 2.0, augment=True, scale_jitter=0.3, aspect_jitter=0.3, seed=seed
        )
        return [ds[0]["scale"].numpy().copy() for _ in range(8)]

    a, b, c = centres(5), centres(5), centres(6)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)
    assert any(not np.allclose(x, y) for x, y in zip(a, c))
    assert len({tuple(np.round(s, 4)) for s in a}) > 3


def test_dataset_default_and_eval_paths_unchanged(tmp_path):
    d = _make_ds_dir(tmp_path)
    ids, _ = load_coco_index(d)
    legacy = CocoKeypointsDataset(d, ids, 2.0, augment=True)
    assert legacy._window is None and legacy._legacy_scale is True
    ev = CocoKeypointsDataset(
        d, ids, 2.0, augment=False, scale_jitter=0.3, aspect_jitter=0.3
    )
    assert ev._window is None  # never jitter validation data


def test_run_config_validation():
    base = dict(
        init_checkpoint="x",
        variant="B",
        num_keypoints=3,
        dataset_dir="d",
        output_dir="o",
    )
    assert validate_run_config(base).scale_jitter is None
    cfg = validate_run_config({**base, "scale_jitter": 0.2, "aspect_jitter": 0.2})
    assert cfg.scale_jitter == 0.2 and cfg.aspect_jitter == 0.2
    with pytest.raises(ValueError):
        validate_run_config({**base, "scale_jitter": 1.5})
    with pytest.raises(ValueError):
        validate_run_config({**base, "aspect_jitter": -0.1})
