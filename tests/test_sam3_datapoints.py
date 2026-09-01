"""sam3-free tests for the tile -> RES-space polygon scaling in datapoints.py.

Task-8 fix round 2, finding 1: polygons were passed through untouched while
the tile image was resized to RES x RES, silently mistraining every mask/box
target on any non-1008 tile. `_scale_polygons_to_res` is the pure helper that
fix extracted so this can be tested without a `sam3` install.
"""

import numpy as np

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
