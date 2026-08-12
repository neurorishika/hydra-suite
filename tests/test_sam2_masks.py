import numpy as np

from hydra_suite.core.inference.sam2.masks import mask_to_contour


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
