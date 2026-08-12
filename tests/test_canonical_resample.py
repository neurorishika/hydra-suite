import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.canonicalization.resample import (
    canonical_warp,
    canonical_warp_batch,
    letterbox_fit,
)

GEOM = CanonicalGeometry.from_reference(
    reference_body_px=60, aspect_ratio=2.0, margin=1.3
)


def _obb(cx, cy, L, W, deg):
    a = np.deg2rad(deg)
    dx = np.array([np.cos(a), np.sin(a)])
    dy = np.array([-np.sin(a), np.cos(a)])
    c = np.array([cx, cy], float)
    return np.array(
        [
            c - L / 2 * dx - W / 2 * dy,
            c + L / 2 * dx - W / 2 * dy,
            c + L / 2 * dx + W / 2 * dy,
            c - L / 2 * dx + W / 2 * dy,
        ]
    )


def test_warp_is_geometrically_exact():
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[210, 205] = 255
    m, _, _ = canonical_affine(_obb(200, 200, 80, 40, 0), GEOM)
    t = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
    crop = canonical_warp(t, m, GEOM).permute(1, 2, 0).numpy().sum(2)
    ys, xs = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
    s = crop.sum()
    got = (float((xs * crop).sum() / s), float((ys * crop).sum() / s))
    exp = m @ np.array([205, 210, 1.0])
    assert abs(got[0] - exp[0]) < 0.5 and abs(got[1] - exp[1]) < 0.5


def test_letterbox_is_isotropic_nonsquare():
    # a horizontal edge must not tilt under a non-square fit -> single scale
    crop = torch.zeros(3, 56, 112)
    crop[:, 28:, :] = 1.0
    out = letterbox_fit(crop, (64, 128))  # (W,H) -> tensor (C,128,64)
    assert out.shape == (3, 128, 64)
    col_means = out[0].mean(dim=0)  # variation across width
    assert float(col_means.std()) < 1e-3  # no horizontal gradient => no x/y anisotropy


def test_batch_matches_singleton():
    frame = torch.rand(3, 300, 300)
    ms = [canonical_affine(_obb(150, 150, 80, 40, d), GEOM)[0] for d in (0, 30, 60)]
    batch = canonical_warp_batch(frame, ms, GEOM)
    for i, m in enumerate(ms):
        assert torch.allclose(batch[i], canonical_warp(frame, m, GEOM), atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_device_parity_cpu_vs_mps():
    frame = torch.rand(3, 300, 300)
    m, _, _ = canonical_affine(_obb(150, 150, 80, 40, 30), GEOM)
    cpu = canonical_warp(frame, m, GEOM)
    mps = canonical_warp(frame.to("mps"), m, GEOM).cpu()
    assert torch.allclose(cpu, mps, atol=2e-2)  # sub-gray-level float noise only
