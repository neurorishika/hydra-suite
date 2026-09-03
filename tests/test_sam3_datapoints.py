"""sam3-free tests for the tile -> RES-space polygon scaling in datapoints.py.

Task-8 fix round 2, finding 1: polygons were passed through untouched while
the tile image was resized to RES x RES, silently mistraining every mask/box
target on any non-1008 tile. `_scale_polygons_to_res` is the pure helper that
fix extracted so this can be tested without a `sam3` install.
"""

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from hydra_suite.training.sam3_lora.datapoints import RES, _scale_polygons_to_res


def test_scale_polygons_to_res_non_square_non_1008_tile():
    # A non-square, non-RES tile -- the exact case `tile_size_for_mode`
    # produces in `auto_object` mode and at frame edges in any mode.
    w, h = 700, 500
    poly = np.array([[0, 0], [700, 0], [700, 500], [0, 500]], dtype=np.float32)

    scaled = _scale_polygons_to_res([poly], w, h)[0]

    expected = np.array([[0, 0], [RES, 0], [RES, RES], [0, RES]], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, atol=1e-3)


def test_scale_polygons_to_res_interior_point():
    w, h = 504, 1008  # scale x by 2, y by 1
    poly = np.array([[100.0, 200.0]], dtype=np.float32)

    scaled = _scale_polygons_to_res([poly], w, h)[0]

    np.testing.assert_allclose(scaled, [[200.0, 200.0]], atol=1e-3)


def test_scale_polygons_to_res_noop_when_already_res():
    # No resize occurs when the tile is already RES x RES, so no scaling
    # should occur either -- same object, unchanged.
    poly = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    scaled = _scale_polygons_to_res([poly], RES, RES)

    assert scaled[0] is poly


def test_scale_polygons_to_res_empty_list():
    assert _scale_polygons_to_res([], 700, 500) == []


def test_positive_datapoint_collates_nonempty_normalized_boxes():
    """Durable guard for two Critical fixes found by review (2026-08-31):

    1. `object_ids_output=[]` made every positive tile collate to
       `num_boxes=0` -- `collate_fn_api` (sam3/train/data/collator.py)
       builds find targets exclusively from that list, used as positional
       indices into `Image.objects`; `Image.objects` is never read any
       other way. A positive datapoint that collates to zero boxes is
       silently indistinguishable from a negative -- the worst outcome
       this feature can have (trains successfully, publishes a useless
       adapter).
    2. Boxes were never converted to normalized CxCyWH and the image never
       got SAM3's mean/std normalization -- both are `NormalizeAPI`'s job,
       confirmed against the installed `sam3.train.transforms.basic_for_api`
       source and every reference config's `train_norm_mean`/`std`.

    Every other test in this repo stops at the `Datapoint` boundary; none
    exercises the real `collate_fn_api` or a real transform, so neither bug
    could be caught without this. Requires a live `sam3` install -- skips
    cleanly on this Mac; must actually run on the CUDA box.
    """
    pytest.importorskip("sam3")
    import numpy as np

    from hydra_suite.training.sam3_lora.dataloader import _default_transform
    from hydra_suite.training.sam3_lora.datapoints import (
        build_datapoint,
        collate_datapoints,
    )

    tile = np.zeros((RES, RES, 3), dtype=np.uint8)
    polygon = np.array(
        [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        dtype=np.float32,
    )
    datapoint = build_datapoint(tile, "ant", [(polygon, False)], _default_transform())

    batched = collate_datapoints([datapoint])["input"]
    target0 = batched.find_targets[0]

    assert target0.num_boxes[0] > 0
    boxes = target0.boxes_padded[0][: target0.num_boxes[0]]
    assert boxes.min().item() >= 0.0
    assert boxes.max().item() <= 1.0


def test_tile_datapoint_shares_one_image_across_positive_and_negative_queries():
    """Meta's collator supports several query rows pointing at one image row."""
    pytest.importorskip("sam3", exc_type=ImportError)

    from hydra_suite.training.sam3_lora.dataloader import _default_transform
    from hydra_suite.training.sam3_lora.datapoints import (
        build_tile_datapoint,
        collate_datapoints,
    )

    tile = np.zeros((RES, RES, 3), dtype=np.uint8)
    polygon = np.array(
        [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        dtype=np.float32,
    )
    transforms = 0
    base_transform = _default_transform()

    def counting_transform(image):
        nonlocal transforms
        transforms += 1
        return base_transform(image)

    datapoint = build_tile_datapoint(
        tile,
        "ant",
        [(polygon, True)],
        ["floor", "food", "wall"],
        counting_transform,
    )
    assert transforms == 1
    assert len(datapoint.images) == 1
    assert datapoint.raw_images is None
    assert [query.image_id for query in datapoint.find_queries] == [0, 0, 0, 0]
    assert [query.object_ids_output for query in datapoint.find_queries] == [
        [0],
        [],
        [],
        [],
    ]

    batched = collate_datapoints([datapoint])["input"]
    stage = batched.find_inputs[0]
    targets = batched.find_targets[0]
    assert stage.img_ids.tolist() == [0, 0, 0, 0]
    assert targets.num_boxes.tolist() == [1, 0, 0, 0]
    assert targets.is_exhaustive.tolist() == [True, True, True, True]
    assert targets.is_valid_segment.tolist() == [1]


def test_shared_query_datapoints_preserve_independent_query_targets():
    """CUDA integration guard: shared owners collate like the former copies."""
    pytest.importorskip("sam3", exc_type=ImportError)

    from hydra_suite.training.sam3_lora.dataloader import _default_transform
    from hydra_suite.training.sam3_lora.datapoints import (
        build_datapoint,
        build_negative_datapoint,
        build_shared_query_datapoints,
        collate_datapoints,
    )

    tile = np.zeros((RES, RES, 3), dtype=np.uint8)
    polygon = np.array(
        [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        dtype=np.float32,
    )
    transform = _default_transform()
    copied = [build_datapoint(tile, "ant", [(polygon, True)], transform)]
    copied.extend(
        build_negative_datapoint(tile, prompt, transform)
        for prompt in ("floor", "wall")
    )
    shared = build_shared_query_datapoints(
        tile, "ant", [(polygon, True)], ["floor", "wall"], transform
    )

    copied_batch = collate_datapoints(copied)["input"]
    shared_batch = collate_datapoints(shared)["input"]
    assert copied_batch.find_text_batch == shared_batch.find_text_batch
    assert copied_batch.find_inputs[0].img_ids.tolist() == [0, 1, 2]
    assert shared_batch.find_inputs[0].img_ids.tolist() == [0, 1, 2]
    assert copied_batch.find_targets[0].num_boxes.tolist() == [1, 0, 0]
    assert (
        shared_batch.find_targets[0].num_boxes.tolist()
        == copied_batch.find_targets[0].num_boxes.tolist()
    )
    assert copied_batch.find_targets[0].is_exhaustive.tolist() == [True] * 3
    assert (
        shared_batch.find_targets[0].is_exhaustive.tolist()
        == copied_batch.find_targets[0].is_exhaustive.tolist()
    )
    assert shared[0].images is shared[1].images is shared[2].images


def test_tile_datapoint_native_shape_without_importing_full_sam3(monkeypatch):
    """Exercise our adapter against the Meta dataclass contract with fakes.

    The macOS training environment cannot import Meta's package root because
    its CUDA-only Triton module is unconditional. These fakes mirror the
    inspected dataclass signatures while the CUDA-only real-package test above
    remains the authoritative integration check.
    """

    @dataclass
    class InferenceMetadata:
        coco_image_id: int
        original_image_id: int
        original_category_id: int
        original_size: tuple[int, int]
        object_id: int
        frame_index: int

    @dataclass
    class FindQueryLoaded:
        query_text: str
        image_id: int
        object_ids_output: list[int]
        is_exhaustive: bool
        query_processing_order: int
        inference_metadata: InferenceMetadata

    @dataclass
    class Object:
        bbox: object
        area: float
        segment: object
        is_crowd: bool

    @dataclass
    class Image:
        data: object
        objects: list[Object]
        size: tuple[int, int]

    @dataclass
    class Datapoint:
        find_queries: list[FindQueryLoaded]
        images: list[Image]
        raw_images: object = None

    class NormalizeAPI:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, datapoint):
            return datapoint

    module_names = [
        "sam3",
        "sam3.train",
        "sam3.train.data",
        "sam3.train.transforms",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    data_module = types.ModuleType("sam3.train.data.sam3_image_dataset")
    for value in (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
        Object,
    ):
        setattr(data_module, value.__name__, value)
    monkeypatch.setitem(sys.modules, "sam3.train.data.sam3_image_dataset", data_module)
    normalize_module = types.ModuleType("sam3.train.transforms.basic_for_api")
    normalize_module.NormalizeAPI = NormalizeAPI
    monkeypatch.setitem(
        sys.modules, "sam3.train.transforms.basic_for_api", normalize_module
    )

    from hydra_suite.training.sam3_lora.datapoints import (
        build_shared_query_datapoints,
        build_tile_datapoint,
    )

    transform_calls = []
    tile = np.zeros((20, 10, 3), dtype=np.uint8)
    polygon = np.array([[1, 2], [8, 2], [8, 18]], dtype=np.float32)

    datapoint = build_tile_datapoint(
        tile,
        "ant",
        [(polygon, True)],
        ["floor", "wall"],
        lambda image: transform_calls.append(image) or "tensor",
    )

    assert len(transform_calls) == 1
    assert len(datapoint.images) == 1
    assert datapoint.images[0].data == "tensor"
    assert datapoint.images[0].objects[0].is_crowd is True
    assert datapoint.raw_images is None
    assert [query.query_text for query in datapoint.find_queries] == [
        "ant",
        "floor",
        "wall",
    ]
    assert [query.image_id for query in datapoint.find_queries] == [0, 0, 0]
    assert [query.object_ids_output for query in datapoint.find_queries] == [
        [0],
        [],
        [],
    ]

    shared_transform_calls = []
    shared = build_shared_query_datapoints(
        tile,
        "ant",
        [(polygon, True)],
        ["floor", "wall"],
        lambda image: shared_transform_calls.append(image) or "shared-tensor",
    )
    assert len(shared_transform_calls) == 1
    assert len(shared) == 3
    assert all(len(item.find_queries) == 1 for item in shared)
    assert all(item.images is shared[0].images for item in shared)
    assert all(item.images[0].data == "shared-tensor" for item in shared)
    assert all(item.images[0].objects is shared[0].images[0].objects for item in shared)
    assert [item.find_queries[0].object_ids_output for item in shared] == [
        [0],
        [],
        [],
    ]
