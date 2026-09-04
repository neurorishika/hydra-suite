"""sam3-free tests for the COCO-split reading and batching helpers in
dataloader.py -- Task-8 fix round 2, findings 2/3/4 and the two minor fixes.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

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


@pytest.mark.parametrize(
    ("segmentation", "expected"),
    [
        ([0, 0, 4, 0, 4, 4], [[[0, 0], [4, 0], [4, 4]]]),
        ([[0, 0], [4, 0], [4, 4]], [[[0, 0], [4, 0], [4, 4]]]),
        (
            [[0, 0, 4, 0, 4, 4], [[10, 10], [12, 10], [12, 12]]],
            [
                [[0, 0], [4, 0], [4, 4]],
                [[10, 10], [12, 10], [12, 12]],
            ],
        ),
        (
            [
                [[0, 0], [4, 0], [4, 4]],
                [[10, 10], [12, 10], [12, 12]],
            ],
            [
                [[0, 0], [4, 0], [4, 4]],
                [[10, 10], [12, 10], [12, 12]],
            ],
        ),
    ],
)
def test_segmentation_polygon_normalization_accepts_flat_and_nested_nx2(
    segmentation, expected
):
    polygons = dl.validated_segmentation_polygons(segmentation)

    assert polygons == tuple(
        tuple((float(x), float(y)) for x, y in polygon) for polygon in expected
    )
    converted = dl._segmentation_to_polygons(
        [{"segmentation": segmentation, "iscrowd": 0}]
    )
    assert len(converted) == len(expected)
    for (points, _crowd), expected_points in zip(converted, expected):
        np.testing.assert_allclose(points, expected_points)


@pytest.mark.parametrize(
    "segmentation",
    [
        [0, 0, 1, 1],
        [0, 0, 1, 1, 2],
        [[0, 0], [1, 1]],
        [[0, 0], [1, float("nan")], [2, 2]],
        {"counts": "encoded-rle"},
    ],
)
def test_segmentation_polygon_normalization_rejects_invalid_shapes(segmentation):
    assert dl.validated_segmentation_polygons(segmentation) == ()
    assert (
        dl._segmentation_to_polygons([{"segmentation": segmentation, "iscrowd": 0}])
        == []
    )


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
    assert (
        dl.try_build_descriptors(str(tmp_path), Sam3LoraParams(prompt="ant"), "valid")
        is None
    )


def test_try_build_datapoints_raises_when_split_present_but_empty(tmp_path):
    _write_coco(tmp_path / "valid", images=[], annotations=[])

    with pytest.raises(RuntimeError):
        dl.try_build_datapoints(str(tmp_path), Sam3LoraParams(prompt="ant"), "valid")
    with pytest.raises(RuntimeError):
        dl.try_build_descriptors(str(tmp_path), Sam3LoraParams(prompt="ant"), "valid")


def test_collate_epoch_batches_reshuffles_reproducibly(monkeypatch):
    # Fake collation (identity) so this test never needs a real `sam3`
    # install -- only the shuffle/batch bookkeeping is under test.
    monkeypatch.setattr(dl, "collate_datapoints", lambda lst: list(lst))
    datapoints = list(range(10))

    monkeypatch.setattr(dl, "load_datapoints", lambda value, transform: [value])
    monkeypatch.setattr(dl, "_default_transform", lambda: object())
    batches_seed1_a = list(dl.collate_epoch_batches(datapoints, batch_size=2, seed=1))
    batches_seed1_b = list(dl.collate_epoch_batches(datapoints, batch_size=2, seed=1))
    batches_seed2 = list(dl.collate_epoch_batches(datapoints, batch_size=2, seed=2))

    assert batches_seed1_a == batches_seed1_b
    assert batches_seed1_a != batches_seed2

    flat = sorted(x for batch in batches_seed1_a for x in batch)
    assert flat == datapoints


def test_build_descriptors_does_not_decode_or_transform_tiles(tmp_path, monkeypatch):
    split_dir = tmp_path / "train"
    _write_coco(
        split_dir,
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 1008, "height": 1008},
            {"id": 2, "file_name": "b.jpg", "width": 1008, "height": 1008},
        ],
        annotations=[],
    )
    params = Sam3LoraParams(
        prompt="ant", num_negatives=1, negative_prompts=["background"]
    )
    monkeypatch.setattr(
        dl.cv2,
        "imread",
        lambda *_a, **_k: pytest.fail("descriptor construction decoded a tile"),
    )
    monkeypatch.setattr(
        dl,
        "_default_transform",
        lambda: pytest.fail("descriptor construction created a transform"),
    )

    descriptors = dl.build_descriptors(str(tmp_path), params, "train", seed=7)

    assert len(descriptors) == 2
    assert all(is_dataclass(descriptor) for descriptor in descriptors)
    json.dumps([asdict(descriptor) for descriptor in descriptors])
    assert [Path(descriptor.image_path).name for descriptor in descriptors] == [
        "a.jpg",
        "b.jpg",
    ]
    assert all(
        descriptor.negative_prompts == ("background",) for descriptor in descriptors
    )


def test_lazy_batches_decode_and_transform_once_per_tile(tmp_path, monkeypatch):
    split_dir = tmp_path / "train"
    images = [
        {"id": idx, "file_name": f"{idx}.jpg", "width": 1008, "height": 1008}
        for idx in range(5)
    ]
    _write_coco(split_dir, images=images, annotations=[])
    params = Sam3LoraParams(
        prompt="ant",
        num_negatives=3,
        negative_prompts=["floor", "wall", "food"],
    )
    descriptors = dl.build_descriptors(str(tmp_path), params, "train", seed=4)
    decoded = []
    transformed = []

    def fake_imread(path):
        decoded.append(path)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def transform(_image):
        transformed.append(object())
        return object()

    def fake_build(tile, prompt, instances, negative_prompts, transform_fn):
        transform_fn(tile)
        shared_id = id(tile)
        return [(query, shared_id) for query in (prompt, *tuple(negative_prompts))]

    monkeypatch.setattr(dl.cv2, "imread", fake_imread)
    monkeypatch.setattr(dl, "_default_transform", lambda: transform)
    monkeypatch.setattr(dl, "build_shared_query_datapoints", fake_build)
    monkeypatch.setattr(dl, "collate_datapoints", lambda values: list(values))

    batches = dl.collate_batches(descriptors, batch_size=2)
    assert decoded == []
    assert transformed == []

    first = next(batches)
    assert len(first) == 2
    assert len(decoded) == 1
    assert len(transformed) == 1
    assert first[0][1] == first[1][1]

    remaining = list(batches)
    assert [len(first), *(len(batch) for batch in remaining)] == [2] * 10
    assert len(decoded) == len(descriptors)
    assert len(transformed) == len(descriptors)


def test_descriptor_order_contains_every_tile_once_in_incomplete_final_batch(
    tmp_path, monkeypatch
):
    split_dir = tmp_path / "train"
    images = [
        {"id": idx, "file_name": f"{idx}.jpg", "width": 1008, "height": 1008}
        for idx in range(7)
    ]
    _write_coco(split_dir, images=images, annotations=[])
    descriptors = dl.build_descriptors(
        str(tmp_path), Sam3LoraParams(prompt="ant", num_negatives=0), "train"
    )
    monkeypatch.setattr(dl, "_default_transform", lambda: object())
    monkeypatch.setattr(
        dl, "load_datapoints", lambda descriptor, _transform: [descriptor]
    )
    monkeypatch.setattr(dl, "collate_datapoints", lambda values: list(values))

    batches = list(dl.collate_epoch_batches(descriptors, batch_size=3, seed=11))

    assert [len(batch) for batch in batches] == [3, 3, 1]
    assert sorted(d.image_id for batch in batches for d in batch) == list(range(7))


def test_build_descriptors_keeps_polygon_and_crowd_metadata(tmp_path):
    split_dir = tmp_path / "train"
    _write_coco(
        split_dir,
        images=[{"id": 3, "file_name": "a.jpg", "width": 20, "height": 10}],
        annotations=[
            {
                "id": 1,
                "image_id": 3,
                "segmentation": [[1, 2, 9, 2, 9, 8]],
                "iscrowd": 1,
            }
        ],
    )

    descriptors = dl.build_descriptors(
        str(tmp_path), Sam3LoraParams(prompt="ant", num_negatives=0), "train"
    )

    assert len(descriptors) == 1
    assert descriptors[0].instances[0].is_crowd is True
    assert descriptors[0].instances[0].polygon == ((1.0, 2.0), (9.0, 2.0), (9.0, 8.0))


def test_query_count_preserves_the_established_batch_and_step_unit(tmp_path):
    split_dir = tmp_path / "train"
    _write_coco(
        split_dir,
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 1008, "height": 1008},
            {"id": 2, "file_name": "b.jpg", "width": 1008, "height": 1008},
        ],
        annotations=[],
    )
    descriptors = dl.build_descriptors(
        str(tmp_path),
        Sam3LoraParams(
            prompt="ant",
            num_negatives=3,
            negative_prompts=["floor", "wall", "food"],
        ),
        "train",
    )

    assert dl.query_count(descriptors) == 8
    assert dl.batch_count(dl.query_count(descriptors), batch_size=3) == 3


def test_every_positive_and_negative_query_appears_once_per_epoch(monkeypatch):
    descriptors = [
        dl.TileDescriptor(
            image_id=1,
            image_path="one.jpg",
            positive_prompt="ant-1",
            negative_prompts=("floor-1", "wall-1"),
            instances=(),
        ),
        dl.TileDescriptor(
            image_id=2,
            image_path="two.jpg",
            positive_prompt="ant-2",
            negative_prompts=("floor-2", "wall-2"),
            instances=(),
        ),
    ]
    monkeypatch.setattr(dl, "_default_transform", lambda: object())
    monkeypatch.setattr(
        dl,
        "load_datapoints",
        lambda descriptor, _transform: [
            descriptor.positive_prompt,
            *descriptor.negative_prompts,
        ],
    )
    monkeypatch.setattr(dl, "collate_datapoints", lambda values: list(values))

    batches = list(dl.collate_epoch_batches(descriptors, batch_size=4, seed=9))

    assert [len(batch) for batch in batches] == [4, 2]
    assert sorted(query for batch in batches for query in batch) == [
        "ant-1",
        "ant-2",
        "floor-1",
        "floor-2",
        "wall-1",
        "wall-2",
    ]
