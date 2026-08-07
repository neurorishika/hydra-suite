"""Pins that ViTPose training and inference derive their box2cs box the same
way: the crop's full extent, not the tight per-annotation COCO bbox.

PoseKit images are one-animal canonical crops (one class + one keypoint set
saved per image; see posekit/gui/main_window.py::save_current). The
per-annotation COCO bbox written by build_coco_keypoints_dataset
(compute_bbox_from_kpts, ~3% pad around keypoints) therefore does NOT
represent the crop the model should be centered on -- inference always uses
the crop's full extent (infer.py::preprocess_crop -> box_xywh = (0,0,w,h)).
Training must derive box2cs's input the same way, or the animal occupies a
systematically different fraction of the model input at train vs inference
time (see docs/superpowers/... deviation D). This test fails if that drifts
apart again.
"""

import json

import cv2
import numpy as np

from hydra_suite.core.individual.pose.vitpose.infer import preprocess_crop
from hydra_suite.core.individual.pose.vitpose.training.dataset import (
    CocoKeypointsDataset,
    load_coco_index,
)


def _make_ds(tmp_path, img_w=80, img_h=100, tight_bbox=(10.0, 10.0, 40.0, 60.0)):
    (tmp_path / "images").mkdir(parents=True)
    img = np.full((img_h, img_w, 3), 127, np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "f0.png"), img)
    k = 3
    kpts = []
    for j in range(k):
        kpts += [20 + 5 * j, 30 + 5 * j, 2]
    coco = {
        "images": [{"id": 1, "file_name": "f0.png", "width": img_w, "height": img_h}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                # Deliberately tight/off-center vs. the full crop extent, to
                # prove training does NOT key box2cs off this value.
                "bbox": list(tight_bbox),
                "area": tight_bbox[2] * tight_bbox[3],
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
    return tmp_path, img


def test_training_box2cs_matches_inference_full_extent(tmp_path):
    ds_dir, img = _make_ds(tmp_path)
    ids, _ = load_coco_index(ds_dir)
    ds = CocoKeypointsDataset(ds_dir, ids, sigma=2.0, augment=False)
    sample = ds[0]

    # Inference derives center/scale from the crop's own full extent.
    _, inf_center, inf_scale = preprocess_crop(img)

    assert np.allclose(sample["center"].numpy(), inf_center)
    assert np.allclose(sample["scale"].numpy(), inf_scale)


def test_training_box2cs_ignores_tight_annotation_bbox(tmp_path):
    # Two datasets sharing the same image size but different (deliberately
    # wrong) tight per-annotation COCO bboxes must yield IDENTICAL
    # center/scale, because training keys off the image's full extent, not
    # ann["bbox"].
    ds_dir_a, _ = _make_ds(tmp_path / "a", tight_bbox=(10.0, 10.0, 40.0, 60.0))
    ds_dir_b, _ = _make_ds(tmp_path / "b", tight_bbox=(0.0, 0.0, 5.0, 5.0))

    ids_a, _ = load_coco_index(ds_dir_a)
    ids_b, _ = load_coco_index(ds_dir_b)
    ds_a = CocoKeypointsDataset(ds_dir_a, ids_a, sigma=2.0, augment=False)
    ds_b = CocoKeypointsDataset(ds_dir_b, ids_b, sigma=2.0, augment=False)

    sample_a = ds_a[0]
    sample_b = ds_b[0]

    assert np.allclose(sample_a["center"].numpy(), sample_b["center"].numpy())
    assert np.allclose(sample_a["scale"].numpy(), sample_b["scale"].numpy())

    # The raw tight bbox is still carried through untouched, for PCK
    # normalization (training/validate.py), not for box2cs.
    assert np.allclose(sample_a["bbox"].numpy(), [10.0, 10.0, 40.0, 60.0])
    assert np.allclose(sample_b["bbox"].numpy(), [0.0, 0.0, 5.0, 5.0])
