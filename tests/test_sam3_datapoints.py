"""sam3-free tests for the tile -> RES-space polygon scaling in datapoints.py.

Task-8 fix round 2, finding 1: polygons were passed through untouched while
the tile image was resized to RES x RES, silently mistraining every mask/box
target on any non-1008 tile. `_scale_polygons_to_res` is the pure helper that
fix extracted so this can be tested without a `sam3` install.
"""

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
