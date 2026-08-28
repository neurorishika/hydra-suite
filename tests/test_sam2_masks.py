import numpy as np

from hydra_suite.core.inference.sam2.masks import clip_mask_to_polygon, mask_to_contour


def test_square_mask_yields_rectangular_contour():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 15:35] = True
    poly = mask_to_contour(mask)
    assert poly is not None and poly.shape[1] == 2
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.min() <= 16 and xs.max() >= 33
    assert ys.min() <= 11 and ys.max() >= 38


def test_empty_mask_returns_none():
    assert mask_to_contour(np.zeros((20, 20), dtype=bool)) is None


def test_largest_contour_selected_over_speck():
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:55, 5:55] = True  # big blob
    mask[0:2, 0:2] = True  # tiny speck (separate)
    poly = mask_to_contour(mask)
    # bbox of returned contour must be the big blob, not the speck
    assert poly[:, 0].max() > 40 and poly[:, 1].max() > 40


def test_clip_mask_to_polygon_removes_pixels_outside_bounding_quad():
    # SAM2 predicted the whole frame (bleeds far past the OBB it was prompted with).
    mask = np.ones((100, 100), dtype=bool)
    # Axis-aligned OBB quad occupying [10, 40) x [10, 40).
    polygon_px = [(10, 10), (40, 10), (40, 40), (10, 40)]

    clipped = clip_mask_to_polygon(mask, polygon_px)

    assert clipped.dtype == bool
    assert clipped.shape == mask.shape
    # Inside the quad: preserved.
    assert clipped[20, 20]
    # Outside the quad: zeroed.
    assert not clipped[0, 0]
    assert not clipped[99, 99]
    assert not clipped[50, 50]


def test_clip_mask_to_polygon_respects_rotated_quad():
    # A rotated (non-axis-aligned) OBB quad -- a diamond centered at (50, 50).
    mask = np.ones((100, 100), dtype=bool)
    polygon_px = [(50, 20), (80, 50), (50, 80), (20, 50)]

    clipped = clip_mask_to_polygon(mask, polygon_px)

    assert clipped[50, 50]  # center of the diamond: inside
    assert not clipped[5, 5]  # corner of the frame: well outside the diamond


def test_clip_mask_to_polygon_empty_when_mask_and_polygon_disjoint():
    mask = np.zeros((100, 100), dtype=bool)
    mask[60:90, 60:90] = True  # mask lives entirely outside the polygon below
    polygon_px = [(0, 0), (20, 0), (20, 20), (0, 20)]

    clipped = clip_mask_to_polygon(mask, polygon_px)

    assert not clipped.any()
