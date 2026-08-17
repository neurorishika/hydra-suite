# tools/equivalence/warp_aabb_microbench.py
"""Byte-identity + speedup check for the AABB canonical warp on CPU and CUDA.

PYTHONPATH=<wt>/src python tools/equivalence/warp_aabb_microbench.py --device cuda
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hydra_suite.core.canonicalization import resample as R
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry


def _theta_ref(m, cw, ch, w, h):
    mi = cv2.invertAffineTransform(np.asarray(m, np.float64))
    sw, sh = cw - 1.0, ch - 1.0
    iw, ih = 1.0 / max(w - 1.0, 1.0), 1.0 / max(h - 1.0, 1.0)
    t00, t01 = mi[0, 0] * sw * iw, mi[0, 1] * sh * iw
    t10, t11 = mi[1, 0] * sw * ih, mi[1, 1] * sh * ih
    return np.array(
        [
            [t00, t01, t00 + t01 + 2 * mi[0, 2] * iw - 1],
            [t10, t11, t10 + t11 + 2 * mi[1, 2] * ih - 1],
        ],
        np.float32,
    )


def _ref(frame, m_aligns, geo):
    cw, ch = geo.canvas_w, geo.canvas_h
    c, h, w = frame.shape
    th = np.stack([_theta_ref(m, cw, ch, w, h) for m in m_aligns])
    tt = torch.as_tensor(th, dtype=torch.float32, device=frame.device)
    with torch.inference_mode():
        grid = F.affine_grid(tt, (len(m_aligns), c, ch, cw), align_corners=True)
        fe = frame.unsqueeze(0).expand(len(m_aligns), -1, -1, -1).float()
        return F.grid_sample(
            fe.contiguous(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def _m_aligns(n, W, H, cw, ch, rng):
    out = []
    for _ in range(n):
        a = np.deg2rad(rng.uniform(0, 360))
        ca, sa = np.cos(a), np.sin(a)
        R2 = np.array([[ca, -sa], [sa, ca]])
        cx, cy = (cw - 1) / 2, (ch - 1) / 2
        px, py = rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.2, H * 0.8)
        t = np.array([px, py]) - R2 @ np.array([cx, cy])
        out.append(
            cv2.invertAffineTransform(
                np.array([[ca, -sa, t[0]], [sa, ca, t[1]]], np.float64)
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--W", type=int, default=2048)
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--frames", type=int, default=50)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    geo = CanonicalGeometry(canvas_wh=(210, 86), margin=2.0, aspect_ratio=210 / 86)
    frame = torch.rand(3, a.H, a.W, device=a.device)
    m = _m_aligns(a.n, a.W, a.H, geo.canvas_w, geo.canvas_h, rng)

    got, ref = R.canonical_warp_batch(frame, m, geo), _ref(frame, m, geo)
    print(
        "byte_identical:",
        torch.equal(got, ref),
        "max_abs_diff:",
        (got - ref).abs().max().item(),
    )

    def bench(fn):
        fn()
        if a.device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(a.frames):
            fn()
        if a.device == "cuda":
            torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t) / a.frames

    print(f"new  {bench(lambda: R.canonical_warp_batch(frame, m, geo)):.2f} ms/frame")
    print(f"old  {bench(lambda: _ref(frame, m, geo)):.2f} ms/frame")


if __name__ == "__main__":
    raise SystemExit(main())
