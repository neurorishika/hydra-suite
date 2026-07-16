import numpy as np
import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.config import UDP_BLUR_KERNEL
from hydra_suite.core.identity.pose.vitpose.decode import (
    decode_udp_cv2,
    decode_udp_torch,
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


def _random_peaky_heatmaps(n=2, k=17, h=64, w=48, seed=0):
    """Real forward-pass heatmaps are peaky but noisy and occasionally flat or
    multi-modal. Pure synthetic Gaussians are too well-conditioned to exercise
    the Hessian solve, so add noise and a second lobe."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, k, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for i in range(n):
        for j in range(k):
            cx, cy = rng.uniform(6, w - 6), rng.uniform(6, h - 6)
            g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 8.0)
            cx2, cy2 = rng.uniform(6, w - 6), rng.uniform(6, h - 6)
            g = g + 0.3 * np.exp(-((xs - cx2) ** 2 + (ys - cy2) ** 2) / 8.0)
            out[i, j] = g + rng.normal(0, 0.01, (h, w))
    return out.astype(np.float32)


def _max_coord_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def test_gate_b_torch_decode_matches_cv2_cpu():
    """GATE B. Tolerance 1e-2 heatmap units, per-keypoint (not averaged --
    averaging hides a single badly-decoded joint, which is the failure we care
    about). 1 heatmap unit is 4 image px, so 1e-2 is comfortably sub-pixel."""
    hm = _random_peaky_heatmaps()
    ref, ref_v = decode_udp_cv2(hm)
    got, got_v = decode_udp_torch(torch.from_numpy(hm))
    assert _max_coord_delta(ref, got.numpy()) < 1e-2
    assert _max_coord_delta(ref_v, got_v.numpy()) < 1e-4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_gate_b_torch_decode_matches_cv2_on_mps():
    """Same gate on MPS. NOTE: the oracle is float64 numpy on CPU; MPS has no
    float64, so this compares ACROSS DTYPES by design. float32 carries ~7
    decimal digits and we need ~2, so 1e-2 is still the right bound."""
    hm = _random_peaky_heatmaps()
    ref, _ = decode_udp_cv2(hm)
    got, _ = decode_udp_torch(torch.from_numpy(hm).to("mps"))
    assert _max_coord_delta(ref, got.cpu().numpy()) < 1e-2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_torch_decode_never_leaves_device():
    """The whole point: no GPU->CPU roundtrip."""
    hm = torch.from_numpy(_random_peaky_heatmaps()).to("mps")
    coords, maxvals = decode_udp_torch(hm)
    assert coords.device.type == "mps"
    assert maxvals.device.type == "mps"
