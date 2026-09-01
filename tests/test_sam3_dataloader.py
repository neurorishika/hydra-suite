"""sam3-free tests for the COCO-split reading and batching helpers in
dataloader.py -- Task-8 fix round 2, findings 2/3/4 and the two minor fixes.
"""

import json

import numpy as np
import pytest

import hydra_suite.training.sam3_lora.dataloader as dl
from hydra_suite.training.contracts import Sam3LoraParams


def _write_coco(split_dir, images, annotations, categories=None):
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
        or [{"id": 1, "name": "ant", "supercategory": "object"}],
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )


def test_load_coco_split_groups_annotations_by_image(tmp_path):
    split_dir = tmp_path / "train"
    _write_coco(
        split_dir,
        images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "segmentation": [[0, 0, 1, 0, 1, 1]],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "segmentation": [[2, 2, 3, 2, 3, 3]],
                "iscrowd": 0,
            },
        ],
    )

    resolved_split_dir, coco, by_image = dl._load_coco_split(tmp_path, "train")

    assert resolved_split_dir == split_dir
    assert len(coco["images"]) == 1
    assert len(by_image[1]) == 2


def test_load_coco_split_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl._load_coco_split(tmp_path, "train")


def test_segmentation_to_polygons_keeps_crowd_flagged():
    # Crowd instances stay in the object list (flagged, not dropped) --
    # `Object.is_crowd` lets the loss handle them correctly, superseding an
    # earlier design that dropped them and forced is_exhaustive=False.
    annotations = [
        {"segmentation": [[0, 0, 1, 0, 1, 1]], "iscrowd": 0},
        {"segmentation": [[2, 2, 3, 2, 3, 3]], "iscrowd": 1},
    ]

    instances = dl._segmentation_to_polygons(annotations)

    assert len(instances) == 2
    (poly0, crowd0), (poly1, crowd1) = instances
    np.testing.assert_allclose(poly0, [[0, 0], [1, 0], [1, 1]])
    assert crowd0 is False
    np.testing.assert_allclose(poly1, [[2, 2], [3, 2], [3, 3]])
    assert crowd1 is True


def test_negative_prompts_not_required_when_num_negatives_zero(tmp_path):
    params = Sam3LoraParams(prompt="ant", num_negatives=0, negative_prompts=[])

    assert dl._negative_prompts_for(tmp_path, params) == []


def test_negative_prompts_raises_when_wanted_but_unavailable(tmp_path):
    params = Sam3LoraParams(prompt="ant", num_negatives=3, negative_prompts=[])

    with pytest.raises(RuntimeError):
        dl._negative_prompts_for(tmp_path, params)


def test_negative_prompts_prefers_manifest(tmp_path):
    (tmp_path / "build_manifest.json").write_text(
        json.dumps({"negative_prompts": ["background"]}), encoding="utf-8"
    )
    params = Sam3LoraParams(prompt="ant", num_negatives=1, negative_prompts=["ignored"])

    assert dl._negative_prompts_for(tmp_path, params) == ["background"]


def test_try_build_datapoints_returns_none_when_split_absent(tmp_path):
    (tmp_path / "valid").mkdir(parents=True)

    assert (
        dl.try_build_datapoints(str(tmp_path), Sam3LoraParams(prompt="ant"), "valid")
        is None
    )


def test_try_build_datapoints_raises_when_split_present_but_empty(tmp_path):
    _write_coco(tmp_path / "valid", images=[], annotations=[])

    with pytest.raises(RuntimeError):
        dl.try_build_datapoints(str(tmp_path), Sam3LoraParams(prompt="ant"), "valid")


def test_collate_epoch_batches_reshuffles_reproducibly(monkeypatch):
    # Fake collation (identity) so this test never needs a real `sam3`
    # install -- only the shuffle/batch bookkeeping is under test.
    monkeypatch.setattr(dl, "collate_datapoints", lambda lst: list(lst))
    datapoints = list(range(10))

    batches_seed1_a = dl.collate_epoch_batches(datapoints, batch_size=2, seed=1)
    batches_seed1_b = dl.collate_epoch_batches(datapoints, batch_size=2, seed=1)
    batches_seed2 = dl.collate_epoch_batches(datapoints, batch_size=2, seed=2)

    assert batches_seed1_a == batches_seed1_b
    assert batches_seed1_a != batches_seed2

    flat = sorted(x for batch in batches_seed1_a for x in batch)
    assert flat == datapoints
