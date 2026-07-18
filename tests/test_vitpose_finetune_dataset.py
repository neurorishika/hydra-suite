import json

import cv2
import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.training.dataset import (
    FEAT_STRIDE,
    CocoKeypointsDataset,
    load_coco_index,
)


def _make_ds(tmp_path, k=3):
    (tmp_path / "images").mkdir()
    img = np.full((100, 80, 3), 127, np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "f0.png"), img)
    kpts = []
    for j in range(k):
        kpts += [20 + 5 * j, 30 + 5 * j, 2]
    coco = {
        "images": [{"id": 1, "file_name": "f0.png", "width": 80, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 40.0, 60.0],
                "area": 2400.0,
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


def test_getitem_shapes(tmp_path):
    ds_dir = _make_ds(tmp_path)
    ids, _ = load_coco_index(ds_dir)
    ds = CocoKeypointsDataset(ds_dir, ids, sigma=2.0, augment=False)
    s = ds[0]
    assert s["image"].shape == (3, 256, 192)
    assert s["target"].shape == (3, 64, 48)
    assert s["target_weight"].shape == (3, 1)
    assert torch.all(s["target_weight"] == 1.0)


def test_target_peak_matches_decoded_gt(tmp_path):
    # With no augmentation, decoding the GT heatmap and mapping back through
    # transform_preds must recover the annotated keypoints (sub-pixel).
    from hydra_suite.core.identity.pose.vitpose.decode import decode_udp_cv2
    from hydra_suite.core.identity.pose.vitpose.transforms import transform_preds

    ds_dir = _make_ds(tmp_path)
    ids, _ = load_coco_index(ds_dir)
    ds = CocoKeypointsDataset(ds_dir, ids, sigma=2.0, augment=False)
    s = ds[0]
    coords, _ = decode_udp_cv2(s["target"].numpy()[None], kernel=11)
    orig = transform_preds(coords[0], s["center"].numpy(), s["scale"].numpy(), (48, 64))
    gt = s["gt_joints"].numpy()[:, :2]
    assert np.allclose(orig, gt, atol=1.0)


def test_feat_stride_value():
    assert np.allclose(
        FEAT_STRIDE, (np.array([192, 256]) - 1.0) / (np.array([48, 64]) - 1.0)
    )
