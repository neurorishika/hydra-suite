# tests/test_canonical_warp_aabb.py
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hydra_suite.core.canonicalization import resample as R
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry


def _make_geometry(cw: int = 210, ch: int = 86) -> CanonicalGeometry:
    return CanonicalGeometry(canvas_wh=(cw, ch), margin=2.0, aspect_ratio=cw / ch)


def _theta_from_m_align_ref(m_align, canvas_w, canvas_h, w_in, h_in):
    """Verbatim copy of resample._theta_from_m_align (frozen oracle)."""
    m_inv = cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
    sw = float(canvas_w - 1)
    sh = float(canvas_h - 1)
    inv_win = 1.0 / max(float(w_in - 1), 1.0)
    inv_hin = 1.0 / max(float(h_in - 1), 1.0)
    t00 = m_inv[0, 0] * sw * inv_win
    t01 = m_inv[0, 1] * sh * inv_win
    t10 = m_inv[1, 0] * sw * inv_hin
    t11 = m_inv[1, 1] * sh * inv_hin
    return np.array(
        [
            [t00, t01, t00 + t01 + 2.0 * m_inv[0, 2] * inv_win - 1.0],
            [t10, t11, t10 + t11 + 2.0 * m_inv[1, 2] * inv_hin - 1.0],
        ],
        dtype=np.float32,
    )


def _ref_warp_batch(frame_chw, m_aligns, geometry):
    """Verbatim copy of the CURRENT expand-based canonical_warp_batch."""
    canvas_w, canvas_h = geometry.canvas_w, geometry.canvas_h
    c, h_in, w_in = frame_chw.shape
    n = len(m_aligns)
    if n == 0:
        return torch.zeros(
            0, c, canvas_h, canvas_w, dtype=frame_chw.dtype, device=frame_chw.device
        )
    thetas_np = np.empty((n, 2, 3), dtype=np.float32)
    for i, m in enumerate(m_aligns):
        thetas_np[i] = _theta_from_m_align_ref(m, canvas_w, canvas_h, w_in, h_in)
    thetas_t = torch.as_tensor(thetas_np, dtype=torch.float32, device=frame_chw.device)
    with torch.inference_mode():
        grid = F.affine_grid(thetas_t, (n, c, canvas_h, canvas_w), align_corners=True)
        frame_expanded = frame_chw.unsqueeze(0).expand(n, -1, -1, -1).float()
        return F.grid_sample(
            frame_expanded.contiguous(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def _rigid_m_inv(angle_deg, px, py, cw, ch):
    """canvas->frame rigid map placing the canvas CENTRE at frame (px,py)."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R2 = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    cx, cy = (cw - 1) / 2.0, (ch - 1) / 2.0
    t = np.array([px, py], dtype=np.float64) - R2 @ np.array([cx, cy])
    return np.array([[ca, -sa, t[0]], [sa, ca, t[1]]], dtype=np.float64)


def _make_m_aligns(kind, W, H, cw, ch):
    rng = np.random.default_rng(7)

    def frame_align(angle, px, py):
        return cv2.invertAffineTransform(_rigid_m_inv(angle, px, py, cw, ch))

    if kind == "centered":
        return [frame_align(0.0, W / 2, H / 2)]
    if kind == "rotated":
        return [frame_align(37.0, W / 2, H / 2)]
    if kind == "many":
        return [
            frame_align(
                rng.uniform(0, 360),
                rng.uniform(W * 0.2, W * 0.8),
                rng.uniform(H * 0.2, H * 0.8),
            )
            for _ in range(18)
        ]
    if kind == "near_border":
        return [frame_align(20.0, 3.0, 3.0), frame_align(90.0, W - 2.0, H - 2.0)]
    if kind == "out_of_frame":
        return [frame_align(10.0, -400.0, -400.0)]
    raise ValueError(kind)


@pytest.mark.parametrize(
    "kind", ["centered", "rotated", "many", "near_border", "out_of_frame"]
)
def test_canonical_warp_batch_matches_reference(kind):
    W, H = 640, 512
    geo = _make_geometry()
    frame = torch.rand(3, H, W, dtype=torch.float32)
    m_aligns = _make_m_aligns(kind, W, H, geo.canvas_w, geo.canvas_h)
    got = R.canonical_warp_batch(frame, m_aligns, geo)
    ref = _ref_warp_batch(frame, m_aligns, geo)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), f"{kind}: max|Δ|={(got - ref).abs().max().item()}"


def test_canonical_warp_single_matches_reference():
    W, H = 640, 512
    geo = _make_geometry()
    frame = torch.rand(3, H, W, dtype=torch.float32)
    m = _make_m_aligns("rotated", W, H, geo.canvas_w, geo.canvas_h)[0]
    got = R.canonical_warp(frame, m, geo)
    ref = _ref_warp_batch(frame, [m], geo).squeeze(0)
    assert torch.equal(got, ref)


def test_empty_returns_zeros():
    geo = _make_geometry()
    frame = torch.rand(3, 64, 64)
    out = R.canonical_warp_batch(frame, [], geo)
    assert out.shape == (0, 3, geo.canvas_h, geo.canvas_w)
