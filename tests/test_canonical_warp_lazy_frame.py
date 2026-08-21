"""``canonical_warp_batch_from_frame`` must be BIT-identical to converting whole.

The lazy entry point exists so a 4512x4512 frame is never turned into a 244 MB
float32 tensor just to sample a few dozen small crops (it was, once per crop
consumer per frame). It is only a legitimate substitution if slicing the raw
frame before the elementwise conversion produces exactly what slicing after it
produced -- hence ``torch.equal``, not ``allclose``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization import resample as R
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.stages.crops import _frame_to_chw_float


def _make_geometry(cw: int = 96, ch: int = 48) -> CanonicalGeometry:
    return CanonicalGeometry(canvas_wh=(cw, ch), margin=2.0, aspect_ratio=cw / ch)


def _m_aligns(rng, n, w, h):
    out = []
    for _ in range(n):
        ang = rng.uniform(-np.pi, np.pi)
        s = rng.uniform(0.5, 2.0)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ca, sa = np.cos(ang) * s, np.sin(ang) * s
        out.append(
            np.array([[ca, -sa, cx], [sa, ca, cy]], dtype=np.float64),
        )
    return out


@pytest.mark.parametrize("kind", ["numpy_hwc_u8", "torch_chw_f32", "torch_hwc_u8"])
def test_lazy_matches_full_frame_conversion_bitwise(kind):
    rng = np.random.default_rng(7)
    h, w = 240, 320
    geo = _make_geometry()

    if kind == "numpy_hwc_u8":
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif kind == "torch_chw_f32":
        frame = torch.rand(3, h, w, dtype=torch.float32)
    else:
        frame = torch.from_numpy(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    m_aligns = _m_aligns(rng, 12, w, h)

    def to_chw(sub):
        return _frame_to_chw_float(sub, "cpu")

    eager = R.canonical_warp_batch(to_chw(frame), m_aligns, geo)
    lazy = R.canonical_warp_batch_from_frame(frame, m_aligns, geo, to_chw)

    assert lazy.shape == eager.shape
    assert torch.equal(lazy, eager)


def test_lazy_handles_empty_and_out_of_frame():
    """No detections, and detections whose footprint misses the frame entirely."""
    rng = np.random.default_rng(3)
    h, w = 64, 64
    geo = _make_geometry(32, 16)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def to_chw(sub):
        return _frame_to_chw_float(sub, "cpu")

    empty = R.canonical_warp_batch_from_frame(frame, [], geo, to_chw)
    assert empty.shape == (0, 3, geo.canvas_h, geo.canvas_w)

    # Far off-frame: grid_sample must return the same zeros the full-frame
    # padding_mode="zeros" path returns.
    far = [np.array([[1.0, 0.0, -10_000.0], [0.0, 1.0, -10_000.0]], dtype=np.float64)]
    eager = R.canonical_warp_batch(to_chw(frame), far, geo)
    lazy = R.canonical_warp_batch_from_frame(frame, far, geo, to_chw)
    assert torch.equal(lazy, eager)
