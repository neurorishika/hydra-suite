import numpy as np
import pytest

from hydra_suite.core.identity.pose.vitpose.config import UDP_BLUR_KERNEL
from hydra_suite.core.identity.pose.vitpose.decode import (
    decode_udp_cv2,
    flip_back,
    get_max_preds,
)


def _gaussian_heatmap(h=64, w=48, cx=20.0, cy=30.0, sigma=2.0):
    ys, xs = np.mgrid[0:h, 0:w]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return g.astype(np.float32)


def test_get_max_preds_argmax():
    hm = np.zeros((1, 1, 64, 48), np.float32)
    hm[0, 0, 30, 20] = 1.0
    coords, maxvals = get_max_preds(hm)
    assert coords.shape == (1, 1, 2)
    assert np.allclose(coords[0, 0], [20.0, 30.0])
    assert np.allclose(maxvals[0, 0], [1.0])


def test_decode_refines_to_subpixel_peak():
    """Peak at a non-integer location: integer argmax lands at (20,30) but the
    true peak is (20.4, 30.4). DARK/UDP refinement must move toward it."""
    hm = _gaussian_heatmap(cx=20.4, cy=30.4)[None, None]
    coords, _ = decode_udp_cv2(hm, kernel=UDP_BLUR_KERNEL)
    assert abs(coords[0, 0, 0] - 20.4) < 0.25
    assert abs(coords[0, 0, 1] - 30.4) < 0.25


def test_decode_does_not_mutate_input():
    hm = _gaussian_heatmap()[None, None]
    original = hm.copy()
    decode_udp_cv2(hm, kernel=UDP_BLUR_KERNEL)
    assert np.array_equal(hm, original), "decode must not mutate its input"


def test_flip_back_swaps_pairs_and_mirrors():
    hm = np.zeros((1, 2, 4, 4), np.float32)
    hm[0, 0, 1, 0] = 1.0
    out = flip_back(hm, [(0, 1)])
    assert out[0, 1, 1, 3] == pytest.approx(1.0)
