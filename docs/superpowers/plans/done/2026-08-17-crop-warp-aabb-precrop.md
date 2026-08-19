# Slice A — AABB Pre-Crop Canonical Warp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-frame ×N replication in `canonical_warp_batch` with per-detection AABB pre-cropping + one batched `grid_sample`, byte-identically, to remove the ~32% crop-warp cost and the 47 s `torch.contiguous()` hotspot.

**Architecture:** All changes are confined to `src/hydra_suite/core/canonicalization/resample.py`. Two new pure helpers (`_canvas_footprint_aabb`, `_theta_for_subregion`) plus rewritten internals of `canonical_warp` / `canonical_warp_batch`. Public signatures and outputs are unchanged, so callers in `stages/crops.py` are untouched. Correctness is pinned by a characterization test that reproduces the current expand-based algorithm and asserts bitwise equality.

**Tech Stack:** Python, PyTorch (`F.affine_grid` / `F.grid_sample`), NumPy, OpenCV (`cv2.invertAffineTransform`), pytest.

## Global Constraints

- **Byte-identical output.** New `canonical_warp` / `canonical_warp_batch` must return tensors bitwise-equal (`torch.equal`) to the current implementation for every input, on both CPU and CUDA tensors.
- **No signature changes.** `canonical_warp(frame_chw, m_align, geometry) -> (C,ch,cw)` and `canonical_warp_batch(frame_chw, m_aligns, geometry) -> (N,C,ch,cw)` keep their exact signatures; callers in `stages/crops.py` are NOT modified.
- **Device-agnostic.** Output device always equals `frame_chw.device`. No new host↔device transfers on the CUDA path.
- **Verification gate.** Equivalence harness must remain byte-identical vs `legacy/main` on MPS (this box) AND CUDA (mehek) before merge. Kill stale sleap/hydra procs before any heavy run.
- **`torch.inference_mode()`** wraps the `affine_grid`/`grid_sample` calls (as today).

---

### Task 1: Characterization test pinning current warp behavior

Locks the current output as the correctness oracle BEFORE any rewrite. The reference function is a verbatim copy of today's expand-based algorithm; the test asserts the live `canonical_warp_batch` matches it across representative geometries. It PASSES against current code (proving reference fidelity) and must keep passing through the rewrite.

**Files:**
- Test: `tests/test_canonical_warp_aabb.py` (create)

**Interfaces:**
- Consumes: `hydra_suite.core.canonicalization.resample.canonical_warp_batch`, `canonical_warp`; `hydra_suite.core.canonicalization.geometry.CanonicalGeometry`.
- Produces: `_ref_warp_batch(frame_chw, m_aligns, geometry)` (test-local oracle); `_make_geometry()`, `_make_m_aligns(kind, W, H, cw, ch)` test helpers reused by later tasks.

- [ ] **Step 1: Write the characterization test**

```python
# tests/test_canonical_warp_aabb.py
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.canonicalization import resample as R


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
            frame_align(rng.uniform(0, 360), rng.uniform(W * 0.2, W * 0.8),
                        rng.uniform(H * 0.2, H * 0.8))
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
    assert torch.equal(got, ref), f"{kind}: max|Δ|={(got-ref).abs().max().item()}"


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
```

- [ ] **Step 2: Run the test against current code**

Run: `python -m pytest tests/test_canonical_warp_aabb.py -q`
Expected: PASS (current impl equals the reference — this proves the oracle is faithful before we change anything).

- [ ] **Step 3: Commit**

```bash
git add tests/test_canonical_warp_aabb.py
git commit -m "test: characterize canonical_warp_batch output as byte-identical oracle"
```

---

### Task 2: `_canvas_footprint_aabb` helper (red → green)

Computes the clamped, padded frame-space AABB of the canonical canvas footprint for one affine.

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/resample.py`
- Test: `tests/test_canonical_warp_aabb.py`

**Interfaces:**
- Produces: `_canvas_footprint_aabb(m_align: np.ndarray, geometry: CanonicalGeometry, frame_hw: tuple[int,int]) -> tuple[int,int,int,int]` returning `(x0, y0, x1, y1)` clamped to `[0,W]×[0,H]`, padded by 1 px; a fully-out-of-frame footprint yields a degenerate box (`x1<=x0` or `y1<=y0`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_canonical_warp_aabb.py
def test_footprint_aabb_centered_is_small_and_inside():
    W, H = 640, 512
    geo = _make_geometry()
    m = _make_m_aligns("centered", W, H, geo.canvas_w, geo.canvas_h)[0]
    x0, y0, x1, y1 = R._canvas_footprint_aabb(m, geo, (H, W))
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
    # footprint of a ~210x86 canvas is far smaller than the full frame
    assert (x1 - x0) < 260 and (y1 - y0) < 260


def test_footprint_aabb_out_of_frame_is_degenerate_or_clamped():
    W, H = 640, 512
    geo = _make_geometry()
    m = _make_m_aligns("out_of_frame", W, H, geo.canvas_w, geo.canvas_h)[0]
    x0, y0, x1, y1 = R._canvas_footprint_aabb(m, geo, (H, W))
    assert x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H
    assert (x1 - x0) <= 0 or (y1 - y0) <= 0  # nothing in-frame
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_warp_aabb.py -k footprint -q`
Expected: FAIL with `AttributeError: module ... has no attribute '_canvas_footprint_aabb'`.

- [ ] **Step 3: Implement the helper**

Add to `resample.py` (after `_theta_from_m_align`):

```python
def _canvas_footprint_aabb(
    m_align: np.ndarray,
    geometry: CanonicalGeometry,
    frame_hw: tuple,
) -> tuple:
    """Clamped, +1px-padded frame-space AABB of the canonical canvas footprint.

    Maps the four canvas corners back through ``m_inv`` (canvas -> frame) and
    bounds them. The +1px pad guarantees bilinear neighbours of any in-frame
    sampled coordinate are inside the crop; clamping to the frame makes
    out-of-frame samples fall outside the sub-region (grid_sample zeros),
    exactly matching the full-frame ``padding_mode="zeros"`` behaviour.
    """
    h_in, w_in = int(frame_hw[0]), int(frame_hw[1])
    cw, ch = geometry.canvas_w, geometry.canvas_h
    m_inv = cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
    xs = np.array([0.0, cw - 1.0, 0.0, cw - 1.0])
    ys = np.array([0.0, 0.0, ch - 1.0, ch - 1.0])
    fx = m_inv[0, 0] * xs + m_inv[0, 1] * ys + m_inv[0, 2]
    fy = m_inv[1, 0] * xs + m_inv[1, 1] * ys + m_inv[1, 2]
    pad = 1
    x0 = max(0, int(np.floor(fx.min())) - pad)
    y0 = max(0, int(np.floor(fy.min())) - pad)
    x1 = min(w_in, int(np.ceil(fx.max())) + pad)
    y1 = min(h_in, int(np.ceil(fy.max())) + pad)
    return x0, y0, x1, y1
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_canonical_warp_aabb.py -k footprint -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/canonicalization/resample.py tests/test_canonical_warp_aabb.py
git commit -m "feat: add _canvas_footprint_aabb helper for canonical warp"
```

---

### Task 3: `_theta_for_subregion` helper (red → green)

Generalizes `_theta_from_m_align` to a sub-region origin and padded-input normalization; reduces exactly to `_theta_from_m_align` when the sub-region is the whole frame.

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/resample.py`
- Test: `tests/test_canonical_warp_aabb.py`

**Interfaces:**
- Produces: `_theta_for_subregion(m_align: np.ndarray, x0: int, y0: int, canvas_wh: tuple[int,int], pad_wh: tuple[int,int]) -> np.ndarray` shape `(2,3)` float32.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_canonical_warp_aabb.py
def test_theta_subregion_reduces_to_full_frame():
    W, H = 640, 512
    geo = _make_geometry()
    m = _make_m_aligns("rotated", W, H, geo.canvas_w, geo.canvas_h)[0]
    full = R._theta_from_m_align(m, geo.canvas_w, geo.canvas_h, W, H)
    sub = R._theta_for_subregion(m, 0, 0, (geo.canvas_w, geo.canvas_h), (W, H))
    assert np.allclose(full, sub, atol=0, rtol=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_warp_aabb.py -k theta_subregion -q`
Expected: FAIL with `AttributeError: ... '_theta_for_subregion'`.

- [ ] **Step 3: Implement the helper**

Add to `resample.py`:

```python
def _theta_for_subregion(
    m_align: np.ndarray,
    x0: int,
    y0: int,
    canvas_wh: tuple,
    pad_wh: tuple,
) -> np.ndarray:
    """``_theta_from_m_align`` for a sub-region input.

    Input pixel ``(u, v)`` of the (padded) sub-region corresponds to frame
    pixel ``(u + x0, v + y0)``, so the canvas->frame map ``m_inv`` has its
    translation shifted by ``-(x0, y0)`` and is normalised by the padded
    sub-region size ``pad_wh`` instead of the full frame. Equals
    ``_theta_from_m_align`` when ``x0 == y0 == 0`` and ``pad_wh == (w_in, h_in)``.
    """
    canvas_w, canvas_h = int(canvas_wh[0]), int(canvas_wh[1])
    pad_w, pad_h = int(pad_wh[0]), int(pad_wh[1])
    m_inv = cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
    tx = m_inv[0, 2] - float(x0)
    ty = m_inv[1, 2] - float(y0)

    sw = float(canvas_w - 1)
    sh = float(canvas_h - 1)
    inv_win = 1.0 / max(float(pad_w - 1), 1.0)
    inv_hin = 1.0 / max(float(pad_h - 1), 1.0)

    t00 = m_inv[0, 0] * sw * inv_win
    t01 = m_inv[0, 1] * sh * inv_win
    t10 = m_inv[1, 0] * sw * inv_hin
    t11 = m_inv[1, 1] * sh * inv_hin
    return np.array(
        [
            [t00, t01, t00 + t01 + 2.0 * tx * inv_win - 1.0],
            [t10, t11, t10 + t11 + 2.0 * ty * inv_hin - 1.0],
        ],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_canonical_warp_aabb.py -k theta_subregion -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/canonicalization/resample.py tests/test_canonical_warp_aabb.py
git commit -m "feat: add _theta_for_subregion helper for canonical warp"
```

---

### Task 4: Rewrite `canonical_warp_batch` / `canonical_warp` to AABB pre-crop

Replaces the full-frame replication with per-detection AABB slicing into a padded batch buffer + one `grid_sample`. Guarded by the Task 1 characterization test (must stay byte-identical) plus the whole existing canonicalization/crops suite.

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/resample.py:94-131` (`canonical_warp_batch`), `:63-91` (`canonical_warp`)
- Test: `tests/test_canonical_warp_aabb.py` (existing, must stay green)

**Interfaces:**
- Consumes: `_canvas_footprint_aabb`, `_theta_for_subregion` (Tasks 2–3).
- Produces: unchanged public `canonical_warp` / `canonical_warp_batch` signatures & outputs.

- [ ] **Step 1: Replace the body of `canonical_warp_batch`**

```python
def canonical_warp_batch(
    frame_chw: torch.Tensor,
    m_aligns: List[np.ndarray],
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    """Batch canonical warp via per-detection AABB pre-crop + one grid_sample.

    Byte-identical to the previous full-frame ``expand(N).contiguous()`` path,
    but samples only each detection's canvas footprint (a small frame region)
    instead of replicating the whole frame N times. See
    ``docs/superpowers/specs/2026-08-17-crop-warp-aabb-precrop-design.md``.
    """
    canvas_w, canvas_h = geometry.canvas_w, geometry.canvas_h
    c, h_in, w_in = frame_chw.shape
    n = len(m_aligns)
    if n == 0:
        return torch.zeros(
            0, c, canvas_h, canvas_w, dtype=frame_chw.dtype, device=frame_chw.device
        )

    boxes = [
        _canvas_footprint_aabb(m, geometry, (h_in, w_in)) for m in m_aligns
    ]
    sub_w = [max(0, x1 - x0) for (x0, _y0, x1, _y1) in boxes]
    sub_h = [max(0, y1 - y0) for (_x0, y0, _x1, y1) in boxes]
    pad_w = max(1, max(sub_w))
    pad_h = max(1, max(sub_h))

    batch = frame_chw.new_zeros((n, c, pad_h, pad_w))
    thetas_np = np.empty((n, 2, 3), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        sw_i, sh_i = sub_w[i], sub_h[i]
        if sw_i > 0 and sh_i > 0:
            batch[i, :, :sh_i, :sw_i] = frame_chw[:, y0:y1, x0:x1]
        thetas_np[i] = _theta_for_subregion(
            m_aligns[i], x0, y0, (canvas_w, canvas_h), (pad_w, pad_h)
        )

    thetas_t = torch.as_tensor(thetas_np, dtype=torch.float32, device=frame_chw.device)
    with torch.inference_mode():
        grid = F.affine_grid(thetas_t, (n, c, canvas_h, canvas_w), align_corners=True)
        crops = F.grid_sample(
            batch.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return crops
```

- [ ] **Step 2: Make `canonical_warp` delegate to the batch path**

Replace the body of `canonical_warp` with:

```python
def canonical_warp(
    frame_chw: torch.Tensor,
    m_align: np.ndarray,
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    """Single-crop canonical warp (see :func:`canonical_warp_batch`)."""
    return canonical_warp_batch(frame_chw, [m_align], geometry).squeeze(0)
```

Leave `_theta_from_m_align` in place (still imported/used by the Task-1 oracle and as documentation of the full-frame case).

- [ ] **Step 3: Run the characterization + helper tests**

Run: `python -m pytest tests/test_canonical_warp_aabb.py -q`
Expected: PASS — all parametrized `centered/rotated/many/near_border/out_of_frame`, single, empty, footprint, and theta cases green (byte-identical).

- [ ] **Step 4: Run the existing canonicalization + crops suites**

Run: `python -m pytest tests/ -k "canonical or crop or resample or warp" -q`
Expected: PASS (no regressions in dependent tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/canonicalization/resample.py
git commit -m "perf: canonical_warp_batch samples per-detection AABB, not full frame x N

Removes the full-frame expand(N).contiguous() replication (the 47s
torch.contiguous hotspot) and the N full-resolution grid_samples; each crop
now samples only its ~230px canvas footprint. Byte-identical (guarded by
tests/test_canonical_warp_aabb.py)."
```

---

### Task 5: CUDA byte-identity + speedup microbench

Confirms byte-identity on CUDA tensors (the unit test runs on CPU here) and quantifies the win. Runs on mehek (`hydra-cuda`).

**Files:**
- Create: `tools/equivalence/warp_aabb_microbench.py`

**Interfaces:**
- Consumes: `canonical_warp_batch`, and the `_ref_warp_batch` algorithm (inlined into the bench).

- [ ] **Step 1: Write the microbench**

```python
# tools/equivalence/warp_aabb_microbench.py
"""Byte-identity + speedup check for the AABB canonical warp on CPU and CUDA.

  PYTHONPATH=<wt>/src python tools/equivalence/warp_aabb_microbench.py --device cuda
"""
from __future__ import annotations
import argparse, time
import cv2, numpy as np, torch, torch.nn.functional as F
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.canonicalization import resample as R


def _theta_ref(m, cw, ch, w, h):
    mi = cv2.invertAffineTransform(np.asarray(m, np.float64))
    sw, sh = cw - 1.0, ch - 1.0
    iw, ih = 1.0 / max(w - 1.0, 1.0), 1.0 / max(h - 1.0, 1.0)
    t00, t01 = mi[0, 0] * sw * iw, mi[0, 1] * sh * iw
    t10, t11 = mi[1, 0] * sw * ih, mi[1, 1] * sh * ih
    return np.array([[t00, t01, t00 + t01 + 2 * mi[0, 2] * iw - 1],
                     [t10, t11, t10 + t11 + 2 * mi[1, 2] * ih - 1]], np.float32)


def _ref(frame, m_aligns, geo):
    cw, ch = geo.canvas_w, geo.canvas_h
    c, h, w = frame.shape
    th = np.stack([_theta_ref(m, cw, ch, w, h) for m in m_aligns])
    tt = torch.as_tensor(th, dtype=torch.float32, device=frame.device)
    with torch.inference_mode():
        grid = F.affine_grid(tt, (len(m_aligns), c, ch, cw), align_corners=True)
        fe = frame.unsqueeze(0).expand(len(m_aligns), -1, -1, -1).float()
        return F.grid_sample(fe.contiguous(), grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)


def _m_aligns(n, W, H, cw, ch, rng):
    out = []
    for _ in range(n):
        a = np.deg2rad(rng.uniform(0, 360))
        ca, sa = np.cos(a), np.sin(a)
        R2 = np.array([[ca, -sa], [sa, ca]])
        cx, cy = (cw - 1) / 2, (ch - 1) / 2
        px, py = rng.uniform(W * .2, W * .8), rng.uniform(H * .2, H * .8)
        t = np.array([px, py]) - R2 @ np.array([cx, cy])
        out.append(cv2.invertAffineTransform(
            np.array([[ca, -sa, t[0]], [sa, ca, t[1]]], np.float64)))
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
    print("byte_identical:", torch.equal(got, ref),
          "max_abs_diff:", (got - ref).abs().max().item())

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
```

- [ ] **Step 2: Run on CPU (this box)**

Run: `PYTHONPATH=$PWD/src python tools/equivalence/warp_aabb_microbench.py --device cpu`
Expected: `byte_identical: True`; `new` ms/frame markedly lower than `old`.

- [ ] **Step 3: Run on CUDA (mehek)**

Sync the branch to mehek, then:
`PYTHONPATH=<wt>/src python tools/equivalence/warp_aabb_microbench.py --device cuda`
Expected: `byte_identical: True` on CUDA; record both timings in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tools/equivalence/warp_aabb_microbench.py
git commit -m "test(bench): AABB canonical warp byte-identity + speedup (CPU+CUDA)"
```

---

### Task 6: Equivalence gate (MPS + CUDA) — merge blocker

Full byte-identical tracking-output gate vs `legacy/main`, the standing acceptance bar. Not a pytest — a documented verification run.

**Files:** none (verification only).

- [ ] **Step 1: Kill stale sleap/hydra procs, then run the MPS matrix (this box)**

```bash
conda activate hydra-mps
pkill -f "sleap_service_[0-9]" 2>/dev/null; pkill -f "conda run -n sleap" 2>/dev/null
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_sliceA RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh ant_pose_headtail ant_obb_sleap fly_obb
```
Expected: EQUIVALENCE at/near DETERMINISM floor for every clip (positions p99 ≈ 0, θ at floor, identical row counts), both `_forward.csv` and `_tracking_final.csv`. Verify CSV row counts > 0.

- [ ] **Step 2: Run the CUDA matrix on mehek**

Per CLAUDE.md's CUDA-box recipe (sync branch → `hydra-cuda` → `run_matrix.sh RUNTIME=cuda`), same clip subset. Expected: byte-identical.

- [ ] **Step 3: Record the gate result**

```bash
git commit --allow-empty -m "verify: Slice A byte-identical equivalence MPS+CUDA on pose subset"
```

---

## Self-Review

**Spec coverage:**
- AABB pre-crop + adjusted affine + batched grid_sample → Tasks 2, 3, 4. ✓
- Byte-identical guarantee → Task 1 oracle + Task 4 gate + Task 5 CUDA check. ✓
- Device-agnostic, unchanged signatures → Task 4 (delegation, `frame_chw.new_zeros`/`.device`). ✓
- Edge cases (out-of-frame, degenerate, n==0, single) → Tasks 1/2 parametrization. ✓
- Verification: unit test → microbench → equivalence harness → Tasks 1/4, 5, 6. ✓
- `suppress_foreign` untouched (operates on output in `stages/crops.py`) — no task needed, unchanged. ✓

**Placeholder scan:** none — every code/step block is concrete.

**Type consistency:** `_canvas_footprint_aabb(m_align, geometry, frame_hw) -> (x0,y0,x1,y1)`, `_theta_for_subregion(m_align,x0,y0,canvas_wh,pad_wh) -> (2,3)`, `canonical_warp_batch(frame_chw, m_aligns, geometry) -> (N,C,ch,cw)` used consistently across Tasks 2–5. `frame_hw` is `(H, W)` everywhere; `canvas_wh`/`pad_wh` are `(w, h)`. ✓
