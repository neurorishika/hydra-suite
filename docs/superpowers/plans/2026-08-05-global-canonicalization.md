# Global Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One canonicalization (rigid, fixed canvas, fixed scale) and one isotropic model fit, shared byte-for-byte by inference, crop-dataset generation, oriented-video generation, ClassKit training and PoseKit training.

**Architecture:** Two pure-geometry layers in `core/canonicalization/`. Layer 1 maps frame → canonical crop with a rotation and a translation only — the source rectangle is a fixed size derived from `REFERENCE_BODY_SIZE`, `reference_aspect_ratio` and `margin`, and does **not** depend on the animal's own dimensions. Layer 2 maps any image → a model's input tensor by isotropic centred letterbox, pinning dtype, channel order, resampler and pad fill. Every producer and every consumer calls these two functions and nothing else.

**Tech Stack:** Python 3.13, numpy, OpenCV (`cv2`), PyTorch, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-global-canonicalization-design.md`
**Survey:** `docs/superpowers/specs/notes/global-canonicalization-research.md`
**Branch:** `feat/global-canonicalization`, worktree `.worktrees/feat-global-canonicalization`, off `main` @ `e6882c0e`.

## Global Constraints

- **Environment:** `conda activate hydra-mps`; `export KMP_DUPLICATE_LIB_OK=TRUE`; `export PYTHONPATH=$PWD/src` from the worktree root. CUDA verification runs on `rutalab@mehek.taild08eb9.ts.net` with `hydra-cuda`.
- **Test gate is delta-based, never absolute.** `main` carries 19 pre-existing failures in the `-k "pose or crop or cli_config or headless or inference or canonical or vitpose or sleap"` selection, plus a collection error in `tests/test_identity_postprocess.py` (always pass `--ignore=tests/test_identity_postprocess.py`). A task passes when its failure set by *name* is a subset of the baseline's. Capture the baseline once with `-p no:randomly` and diff by name, never by count.
- **Never run the whole suite in one process** — `tests/` contains known modal-dialog hangs and a SIGABRT. Batch per file or per `-k` selection.
- **Layer 2 contract, fixed for every call site:** dtype uint8 `[0, 255]`; channel order BGR; resampler `cv2.INTER_LINEAR` with antialias on downscale; pad fill **zeros** (not the foreign-mask background colour).
- **Layer 1 contract:** the linear part of the affine is a pure rotation. Any change that makes `M[:, :2]` non-orthogonal is a defect, not a tradeoff.
- **`REFERENCE_BODY_SIZE` is not redefined.** Its meaning (median per-detection geometric mean of major and minor), its auto-set formula, and all existing consumers stay exactly as they are. The crop path only reads it.
- **No new user-facing config knobs.** Canvas geometry derives from `REFERENCE_BODY_SIZE`, `RESIZE_FACTOR`, `reference_aspect_ratio`, and the canonical margin.
- **Dependency direction:** `core/` must never import from an app layer (`trackerkit`, `posekit`, `classkit`, `detectkit`, `refinekit`, `filterkit`) or from `integrations/`.
- **Formatting:** run `make format` (autopep8 + black + isort; note CLAUDE.md documents a `make commit-prep` target that does not exist in the Makefile) before each commit. Pre-commit hooks run automatically.
- **Commit style:** conventional commits. Do **not** add a `Co-Authored-By: Claude` trailer.
- **This change is intentionally not equivalent.** Never "fix" a canonicalization difference to make the equivalence harness pass. The harness is used to re-baseline (Task 12).

---

### Task 1: Layer 1 — canonical geometry

**Files:**
- Create: `src/hydra_suite/core/canonicalization/geometry.py`
- Test: `tests/test_canonical_geometry.py`

**Interfaces:**
- Consumes: nothing (leaf module; numpy + cv2 only)
- Produces:
  - `CanonicalGeometry` frozen dataclass with fields `canvas_wh: tuple[int, int]`, `margin: float`, `aspect_ratio: float`; properties `canvas_w -> int`, `canvas_h -> int`; classmethod `from_reference(reference_body_px: float, aspect_ratio: float, margin: float) -> CanonicalGeometry`; method `to_dict() -> dict` and classmethod `from_dict(d: dict) -> CanonicalGeometry`
  - `canonical_affine(corners: np.ndarray, geometry: CanonicalGeometry) -> tuple[np.ndarray, float, bool]` returning `(M_align (2,3) float64, major_axis_theta_rad, clipped)`
  - `overflow_ratio(corners: np.ndarray, geometry: CanonicalGeometry) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_geometry.py
import math

import numpy as np
import pytest

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
    overflow_ratio,
)


def obb(cx, cy, major, minor, theta):
    """(4,2) OBB corners for a box centred at (cx, cy), rotated by theta."""
    hw, hh = major / 2.0, minor / 2.0
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (base @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(np.float32)


def test_canvas_derives_major_axis_from_geometric_mean():
    # REFERENCE_BODY_SIZE is sqrt(major*minor); major = body * sqrt(ar).
    g = CanonicalGeometry.from_reference(
        reference_body_px=20.0, aspect_ratio=4.0, margin=1.5
    )
    # major = 20 * 2 = 40; canvas_w = 40 * 1.5 = 60 -> even
    assert g.canvas_w == 60
    assert g.canvas_h == 60 // 4


def test_canvas_dimensions_are_even():
    g = CanonicalGeometry.from_reference(
        reference_body_px=17.3, aspect_ratio=2.44, margin=1.37
    )
    assert g.canvas_w % 2 == 0
    assert g.canvas_h % 2 == 0


def test_affine_linear_part_is_a_pure_rotation():
    """The defining property: Layer 1 scales nothing."""
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    for ar in (1.1, 1.8, 2.44, 3.5, 6.0):
        corners = obb(100.0, 80.0, 40.0, 40.0 / ar, math.radians(37.0))
        M, _, _ = canonical_affine(corners, g)
        A = np.asarray(M)[:, :2]
        sv = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(sv, [1.0, 1.0], atol=1e-6)


def test_affine_is_invariant_to_animal_size():
    """Same centre and angle, different extents -> identical affine."""
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    small = obb(100.0, 80.0, 20.0, 8.0, 0.4)
    large = obb(100.0, 80.0, 60.0, 24.0, 0.4)
    m_small, _, _ = canonical_affine(small, g)
    m_large, _, _ = canonical_affine(large, g)
    np.testing.assert_allclose(m_small, m_large, atol=1e-9)


def test_centroid_maps_to_canvas_centre():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    corners = obb(123.0, 45.0, 30.0, 12.0, 1.1)
    M, _, _ = canonical_affine(corners, g)
    centre = np.asarray(M) @ np.array([123.0, 45.0, 1.0])
    np.testing.assert_allclose(
        centre, [g.canvas_w / 2.0, g.canvas_h / 2.0], atol=1e-6
    )


def test_theta_recovers_the_major_axis_angle():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    for deg in (0.0, 30.0, 91.0, 179.0):
        corners = obb(50.0, 50.0, 40.0, 16.0, math.radians(deg))
        _, theta, _ = canonical_affine(corners, g)
        assert math.isclose(
            math.cos(2 * theta), math.cos(2 * math.radians(deg)), abs_tol=1e-5
        )


def test_clipping_is_reported_not_absorbed():
    g = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    fits = obb(80.0, 80.0, 20.0, 10.0, 0.0)
    _, _, clipped_small = canonical_affine(fits, g)
    assert clipped_small is False

    huge = obb(80.0, 80.0, 400.0, 200.0, 0.0)
    _, _, clipped_big = canonical_affine(huge, g)
    assert clipped_big is True
    assert overflow_ratio(huge, g) > 1.0
    assert overflow_ratio(fits, g) <= 1.0


def test_degenerate_obb_raises():
    g = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    degenerate = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        canonical_affine(degenerate, g)


def test_round_trips_through_dict():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    assert CanonicalGeometry.from_dict(g.to_dict()) == g
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_canonical_geometry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.canonicalization.geometry'`

- [ ] **Step 3: Implement the module**

```python
"""Canonical crop geometry: one rigid transform, one fixed canvas.

The canvas is a property of the project, not of the detection.  Its long edge
holds ``margin`` times the reference animal's major axis; the OBB supplies only
a centre and an angle.  Both axes therefore map at scale 1 -- the transform is a
rotation and a translation, nothing more -- so no animal is stretched, and body
size survives into the crop as signal instead of being normalised away.

``REFERENCE_BODY_SIZE`` is the geometric mean ``sqrt(major * minor)``; with
``ar = major / minor`` the major axis is ``body_px * sqrt(ar)``.  That recovers
the extent this module needs without redefining a knob that Kalman, Hungarian,
background subtraction and the detection cache key all depend on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_MIN_CANVAS_EDGE = 8


def _even(value: float) -> int:
    return max(_MIN_CANVAS_EDGE, int(math.ceil(float(value) / 2.0) * 2))


@dataclass(frozen=True)
class CanonicalGeometry:
    """Fixed canonical crop geometry for one project/session."""

    canvas_wh: tuple[int, int]
    margin: float
    aspect_ratio: float

    @classmethod
    def from_reference(
        cls,
        reference_body_px: float,
        aspect_ratio: float,
        margin: float,
    ) -> "CanonicalGeometry":
        body = max(1e-3, float(reference_body_px))
        ar = max(1.0, float(aspect_ratio))
        m = max(1.0, float(margin))
        canvas_w = _even(body * math.sqrt(ar) * m)
        canvas_h = _even(canvas_w / ar)
        return cls(canvas_wh=(canvas_w, canvas_h), margin=m, aspect_ratio=ar)

    @property
    def canvas_w(self) -> int:
        return int(self.canvas_wh[0])

    @property
    def canvas_h(self) -> int:
        return int(self.canvas_wh[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas_wh": [self.canvas_w, self.canvas_h],
            "margin": float(self.margin),
            "aspect_ratio": float(self.aspect_ratio),
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CanonicalGeometry":
        w, h = d["canvas_wh"]
        return cls(
            canvas_wh=(int(w), int(h)),
            margin=float(d["margin"]),
            aspect_ratio=float(d["aspect_ratio"]),
        )


def _axes(corners: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))
    if e01 < 1e-3 or e12 < 1e-3:
        raise ValueError("Degenerate OBB (zero-length edge)")
    major_vec = c[1] - c[0] if e01 >= e12 else c[2] - c[1]
    angle = float(math.atan2(float(major_vec[1]), float(major_vec[0])))
    return c, max(e01, e12), min(e01, e12), angle, 0.0


def overflow_ratio(corners: np.ndarray, geometry: CanonicalGeometry) -> float:
    """How far the padded OBB exceeds the canvas; <= 1.0 means it fits."""
    _, major, minor, _, _ = _axes(corners)
    return max(
        major * geometry.margin / geometry.canvas_w,
        minor * geometry.margin / geometry.canvas_h,
    )


def canonical_affine(
    corners: np.ndarray,
    geometry: CanonicalGeometry,
) -> tuple[np.ndarray, float, bool]:
    """Return ``(M_align, major_axis_theta, clipped)`` for one OBB.

    ``M_align`` is a 2x3 affine mapping frame pixels to canvas pixels: a
    rotation that puts the major axis horizontal, then a translation that puts
    the centroid at the canvas centre.  Its linear part is orthonormal by
    construction -- there is no scale term.
    """
    c, major, minor, angle, _ = _axes(corners)
    cx = float(np.mean(c[:, 0]))
    cy = float(np.mean(c[:, 1]))

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    half_w = geometry.canvas_w / 2.0
    half_h = geometry.canvas_h / 2.0

    m_align = np.array(
        [
            [cos_a, -sin_a, half_w - (cos_a * cx - sin_a * cy)],
            [sin_a, cos_a, half_h - (sin_a * cx + cos_a * cy)],
        ],
        dtype=np.float64,
    )

    clipped = (
        major * geometry.margin > geometry.canvas_w
        or minor * geometry.margin > geometry.canvas_h
    )
    return m_align, angle, bool(clipped)


def invert_affine(m_align: np.ndarray) -> np.ndarray:
    """Canvas -> frame, for back-projecting predictions."""
    return cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_canonical_geometry.py -q`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/canonicalization/geometry.py tests/test_canonical_geometry.py
git commit -m "feat(canonicalization): rigid fixed-canvas Layer 1 geometry"
```

---

### Task 2: Layer 2 — model fit

**Files:**
- Create: `src/hydra_suite/core/canonicalization/fit.py`
- Test: `tests/test_canonical_fit.py`

**Interfaces:**
- Consumes: nothing (leaf module)
- Produces:
  - `FitResult` frozen dataclass: `model_wh: tuple[int, int]`, `inner_wh: tuple[int, int]`, `offset_xy: tuple[int, int]`, `scale: float`
  - `fit_to_model_input(source_wh: tuple[int, int], model_wh: tuple[int, int]) -> FitResult`
  - `apply_fit(image: np.ndarray, fit: FitResult) -> np.ndarray` — uint8 BGR in, uint8 BGR out, zero pad
  - `fit_affine(fit: FitResult) -> np.ndarray` — 2x3, for composing into the full frame→model transform

`apply_fit` uses `cv2.INTER_AREA` when `fit.scale < 1.0` (antialiased downscale) and `cv2.INTER_LINEAR` otherwise. This is the single resampler decision for the whole repo.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_fit.py
import numpy as np
import pytest

from hydra_suite.core.canonicalization.fit import (
    apply_fit,
    fit_affine,
    fit_to_model_input,
)


def test_identity_when_source_matches_model():
    f = fit_to_model_input((128, 64), (128, 64))
    assert f.scale == 1.0
    assert f.offset_xy == (0, 0)
    assert f.inner_wh == (128, 64)


def test_scale_is_a_single_scalar_for_both_axes():
    f = fit_to_model_input((128, 64), (256, 256))
    assert f.scale == pytest.approx(2.0)
    assert f.inner_wh == (256, 128)


def test_fit_is_limited_by_the_tighter_axis():
    f = fit_to_model_input((100, 100), (256, 64))
    assert f.scale == pytest.approx(0.64)
    assert f.inner_wh == (64, 64)


def test_content_is_centred():
    f = fit_to_model_input((256, 128), (256, 256))
    assert f.offset_xy == (0, 64)


@pytest.mark.parametrize(
    "source_wh,model_wh",
    [((128, 64), (256, 256)), ((64, 128), (256, 256)), ((300, 50), (128, 128))],
)
def test_aspect_ratio_is_preserved(source_wh, model_wh):
    f = fit_to_model_input(source_wh, model_wh)
    src_ar = source_wh[0] / source_wh[1]
    out_ar = f.inner_wh[0] / f.inner_wh[1]
    assert out_ar == pytest.approx(src_ar, rel=0.02)


def test_apply_fit_pads_with_zeros_and_returns_uint8():
    img = np.full((64, 128, 3), 200, dtype=np.uint8)
    f = fit_to_model_input((128, 64), (256, 256))
    out = apply_fit(img, f)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 0          # padded band
    assert int(out[128, 128, 0]) > 100     # content


def test_apply_fit_is_deterministic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
    f = fit_to_model_input((128, 64), (96, 96))
    np.testing.assert_array_equal(apply_fit(img, f), apply_fit(img, f))


def test_fit_affine_round_trips_a_point():
    import cv2

    f = fit_to_model_input((128, 64), (256, 256))
    m = fit_affine(f)
    inv = cv2.invertAffineTransform(m)
    pt = np.array([37.0, 21.0, 1.0])
    mapped = m @ pt
    back = inv @ np.array([mapped[0], mapped[1], 1.0])
    np.testing.assert_allclose(back, pt[:2], atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_canonical_fit.py -q`
Expected: FAIL — `ModuleNotFoundError: ... canonicalization.fit`

- [ ] **Step 3: Implement the module**

```python
"""Layer 2: fit any image into a model's input tensor, identically everywhere.

This is the only step between a canonical crop and a model, so it pins every
property a second implementation could get wrong: dtype uint8 [0, 255], BGR
channel order, one resampler (INTER_AREA down, INTER_LINEAR up), and zero pad.

The pad value is deliberately NOT the foreign-mask background colour: masking
hides a neighbouring animal inside the crop, padding fills canvas outside the
source image.  Pose already pads zeros at both training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FitResult:
    model_wh: tuple[int, int]
    inner_wh: tuple[int, int]
    offset_xy: tuple[int, int]
    scale: float


def fit_to_model_input(
    source_wh: tuple[int, int],
    model_wh: tuple[int, int],
) -> FitResult:
    """Isotropic centred letterbox parameters. Pure arithmetic."""
    sw, sh = int(source_wh[0]), int(source_wh[1])
    mw, mh = int(model_wh[0]), int(model_wh[1])
    if sw <= 0 or sh <= 0 or mw <= 0 or mh <= 0:
        raise ValueError(f"non-positive dimensions: {source_wh} -> {model_wh}")

    scale = min(mw / sw, mh / sh)
    inner_w = max(1, min(mw, int(round(sw * scale))))
    inner_h = max(1, min(mh, int(round(sh * scale))))
    return FitResult(
        model_wh=(mw, mh),
        inner_wh=(inner_w, inner_h),
        offset_xy=((mw - inner_w) // 2, (mh - inner_h) // 2),
        scale=float(scale),
    )


def apply_fit(image: np.ndarray, fit: FitResult) -> np.ndarray:
    """Resize *image* by ``fit`` and paste it centred on a zero canvas."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"Layer 2 requires uint8 [0,255] input, got {arr.dtype}. "
            "Convert at the producer, not here."
        )
    if arr.ndim == 2:
        arr = arr[:, :, None]
    channels = arr.shape[2]

    interp = cv2.INTER_AREA if fit.scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, fit.inner_wh, interpolation=interp)
    if resized.ndim == 2:
        resized = resized[:, :, None]

    mw, mh = fit.model_wh
    canvas = np.zeros((mh, mw, channels), dtype=np.uint8)
    ox, oy = fit.offset_xy
    canvas[oy : oy + fit.inner_wh[1], ox : ox + fit.inner_wh[0]] = resized
    return canvas


def fit_affine(fit: FitResult) -> np.ndarray:
    """2x3 affine mapping source pixels to model-input pixels."""
    return np.array(
        [[fit.scale, 0.0, fit.offset_xy[0]], [0.0, fit.scale, fit.offset_xy[1]]],
        dtype=np.float64,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_canonical_fit.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/canonicalization/fit.py tests/test_canonical_fit.py
git commit -m "feat(canonicalization): isotropic letterbox Layer 2 with a pinned contract"
```

---

### Task 3: Fix the `(H, W)` transpose and widen the classifier input size

Prerequisite for Layer 2 wiring. Today `ClassifierMetadata.input_size` is documented `(H, W)` but read as `(W, H)` at nine sites; that is harmless only because two anisotropic stretches cancel. An isotropic Layer 2 breaks the cancellation, so this must be correct **before** Task 6 wires it.

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/cnn.py:55`
- Modify: `src/hydra_suite/core/inference/stages/headtail.py:84`
- Modify: `src/hydra_suite/core/inference/stages/crops.py:261, 367, 388, 450, 487, 506`
- Modify: `src/hydra_suite/core/identity/classification/headtail.py:522-523, 599-601, 753-755`
- Modify: `src/hydra_suite/training/contracts.py:104`
- Modify: `src/hydra_suite/training/runner.py:1170, 1195, 1832, 1855`
- Test: `tests/test_classifier_input_size_orientation.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `CustomCNNParams.input_size: tuple[int, int]` in `(H, W)` order (was `int`). `_normalize_input_size` in `core/identity/classification/backend.py:82` remains the single parser and is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_input_size_orientation.py
"""input_size is (H, W) everywhere. Non-square models make this observable."""

import numpy as np
import pytest

from hydra_suite.core.identity.classification.backend import _normalize_input_size


def test_normalize_returns_h_w():
    assert _normalize_input_size([64, 128]) == (64, 128)


def test_crop_target_uses_width_second():
    """extract_classifier_crops must produce (H, W) == metadata.input_size."""
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.stages.crops import extract_classifier_crops

    corners = np.array(
        [[10.0, 10.0], [42.0, 10.0], [42.0, 26.0], [10.0, 26.0]], dtype=np.float32
    )
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[26.0, 18.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([512.0], dtype=np.float32),
        shapes=np.array([[512.0, 2.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=np.stack([corners]),
        detection_ids=np.array([0], dtype=np.int64),
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    input_size = (64, 128)  # (H, W) -- deliberately non-square
    crops = extract_classifier_crops(frame, obb, input_size, 2.0, 1.3)
    assert crops[0].shape[:2] == input_size


def test_custom_cnn_params_accept_a_pair():
    from hydra_suite.training.contracts import CustomCNNParams

    p = CustomCNNParams(input_size=(64, 128))
    assert tuple(p.input_size) == (64, 128)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_classifier_input_size_orientation.py -q`
Expected: FAIL — `test_crop_target_uses_width_second` gets `(128, 64)`; `test_custom_cnn_params_accept_a_pair` fails on the `int` field.

- [ ] **Step 3: Fix each site**

At every listed call site, replace the width-first read with an explicit unpack, so the orientation is stated rather than positional:

```python
# stages/cnn.py, stages/headtail.py -- was: (meta.input_size[0], meta.input_size[1])
in_h, in_w = model.input_size          # ClassifierMetadata documents (H, W)
target_size = (in_h, in_w)
```

```python
# stages/crops.py extract_classifier_crops -- was: out_w, out_h = target_size[0], target_size[1]
out_h, out_w = int(target_size[0]), int(target_size[1])
```

Apply the same unpack at `crops.py:367, 450, 487`, and write `native_sizes` rows
as `[out_h, out_w]` consistently at `:388` and `:506`.

```python
# identity/classification/headtail.py -- three sites, was: out_w = input_size[0]
out_h, out_w = int(self._input_size[0]), int(self._input_size[1])
```

```python
# training/contracts.py
input_size: tuple[int, int] = (224, 224)   # (H, W); was: int = 224
```

At `runner.py:1170, 1195, 1832, 1855`, stamp the pair directly instead of
`(sz, sz)`:

```python
in_h, in_w = params.input_size
metadata["input_size"] = [int(in_h), int(in_w)]
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_classifier_input_size_orientation.py -q`
Expected: PASS, 3 tests.

Run the delta gate:
`python -m pytest tests/ -q -p no:randomly --ignore=tests/test_identity_postprocess.py -k "classifier or headtail or cnn or crops or training"`
Expected: no failure names absent from the baseline.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "fix(identity): read classifier input_size as (H, W) everywhere

Documented (H, W) at backend.py:47 but read width-first at nine sites. Harmless
only because two anisotropic stretches cancelled; an isotropic Layer 2 breaks
that cancellation. Widens CustomCNNParams.input_size to a pair so the
torchvision backbone can express a non-square input at all."
```

---

### Task 4: Make the canonical margin a real knob (D4, D2)

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:807-812`
- Modify: `src/hydra_suite/trackerkit/cli_config.py:153-175, 295-300`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:2072, 2728-2760`
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:985, 1000-1015`
- Create: `src/hydra_suite/trackerkit/advanced_defaults.py`
- Test: `tests/test_canonical_margin_wiring.py`

**Interfaces:**
- Consumes: `CanonicalGeometry` (Task 1)
- Produces: `DEFAULT_ADVANCED_CONFIG: dict[str, Any]` in the new `advanced_defaults.py`, imported by both `cli_config._default_advanced_config()` and `gui/orchestrators/config._load_advanced_config()`; new advanced key `canonical_margin` (default `1.3`); GUI widget `detection_panel.spin_canonical_margin` and button `btn_auto_set_margin`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_margin_wiring.py
"""The canonical margin must be settable, and identical on both entry points."""

import ast
from pathlib import Path

from hydra_suite.trackerkit.advanced_defaults import DEFAULT_ADVANCED_CONFIG
from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    load_tracker_cli_session,
)


def test_margin_has_one_default_table():
    assert "canonical_margin" in DEFAULT_ADVANCED_CONFIG
    assert "reference_aspect_ratio" in DEFAULT_ADVANCED_CONFIG


def test_both_builders_share_the_table():
    src = Path(__file__).resolve().parents[1] / "src" / "hydra_suite"
    for rel in ("trackerkit/cli_config.py", "trackerkit/gui/orchestrators/config.py"):
        text = (src / rel).read_text(encoding="utf-8")
        assert "DEFAULT_ADVANCED_CONFIG" in text, rel


def test_cli_honours_a_configured_margin(tmp_path):
    session = load_tracker_cli_session(
        str(tmp_path / "clip.mp4"),
        config_data={
            "file_path": str(tmp_path / "clip.mp4"),
            "fps": 30.0,
            "canonical_margin": 1.6,
        },
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=60, width=640, height=480
        ),
    )
    assert session.params["ADVANCED_CONFIG"]["canonical_margin"] == 1.6


def test_dead_key_is_gone():
    src = Path(__file__).resolve().parents[1] / "src"
    hits = [
        p
        for p in src.rglob("*.py")
        if "yolo_headtail_canonical_margin" in p.read_text(encoding="utf-8")
    ]
    assert hits == [], f"dead margin key still read in {hits}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_canonical_margin_wiring.py -q`
Expected: FAIL — `advanced_defaults` does not exist; `yolo_headtail_canonical_margin` still present at `config.py:812`.

- [ ] **Step 3: Implement**

Create `src/hydra_suite/trackerkit/advanced_defaults.py` holding the union of
the two existing tables (25 keys from `gui/orchestrators/config._load_advanced_config()`
plus anything only `cli_config._default_advanced_config()` had), and add
`"canonical_margin": 1.3` and `"reference_aspect_ratio": 2.0`. Both loaders
import it; delete both literal tables.

In `core/inference/config.py`, replace the `yolo_headtail_canonical_margin`
read with `canonical_margin`, and build the geometry once:

```python
_adv = params.get("ADVANCED_CONFIG", {})
canonical = CanonicalGeometry.from_reference(
    reference_body_px=float(params.get("REFERENCE_BODY_SIZE", 20.0))
    * float(params.get("RESIZE_FACTOR", 1.0)),
    aspect_ratio=float(_adv.get("reference_aspect_ratio", 2.0)),
    margin=float(_adv.get("canonical_margin", 1.3)),
)
```

In `detection_panel.py`, add a `QDoubleSpinBox` `spin_canonical_margin`
(range 1.0-3.0, step 0.05, default 1.3) beside the aspect-ratio spin, and an
`Auto-Set Margin from Max` button next to the two existing auto-set buttons:

```python
def _auto_set_margin_from_detection(self):
    if self.detected_sizes is None:
        return
    stats = self.detected_sizes["stats"]
    body = self._detection_panel.spin_reference_body_size.value()
    ar = self._detection_panel.spin_reference_aspect_ratio.value()
    suggested = stats["major"]["max"] / max(1e-6, body * math.sqrt(ar))
    self._detection_panel.spin_canonical_margin.setValue(
        min(3.0, math.ceil(suggested * 20.0) / 20.0)
    )
```

Also fix the uppercase read at `gui/workers/crops_worker.py:306` to take the
aspect ratio from `ADVANCED_CONFIG` (defect D1).

Finally, document the surviving coupling in the aspect-ratio tooltip at
`detection_panel.py:980-984`. `reference_aspect_ratio` also centres the
detection aspect-ratio filter (`core/inference/config.py:641-644` derives
`min/max_aspect_ratio` as `ref_ar x multiplier`). That filter is off by default,
but an operator who retunes the AR for canvas efficiency *with filtering on*
silently changes which detections survive:

```python
self.spin_reference_aspect_ratio.setToolTip(
    "Species-typical major/minor axis ratio.\n"
    "Sets the canonical crop canvas shape (a poor match costs background\n"
    "pixels, not accuracy).\n"
    "ALSO centres the detection aspect-ratio filter when that filter is\n"
    "enabled — changing this with filtering on changes which detections\n"
    "survive.\n"
    "Click 'Auto-Set Aspect Ratio' to measure it from sample frames."
)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_canonical_margin_wiring.py -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "fix(trackerkit): make the canonical margin settable; one advanced-config table

config.py read yolo_headtail_canonical_margin -- a key nothing writes -- pinning
the inference margin at 1.3 while the crop exporter used INDIVIDUAL_CROP_PADDING.
Margin is the operator's clipping dial under global canonicalization, so it gains
a wired spin box and an Auto-Set from the measured maximum. The two diverged
advanced-config default tables collapse into one, which is what let the CLI
invent a 4.0 aspect-ratio default in the first place."
```

---

### Task 5: Rewrite `canonicalization/crop.py` onto Layer 1

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/crop.py`
- Modify: `src/hydra_suite/core/canonicalization/__init__.py`
- Delete: `src/hydra_suite/core/canonicalization/dataset.py`
- Delete: `tests/test_canonicalization.py`, `tests/test_canonicalization_flexible.py`
- Modify: `tests/test_canonical_crop.py`

**Interfaces:**
- Consumes: `CanonicalGeometry`, `canonical_affine`, `invert_affine` (Task 1)
- Produces: `extract_canonical_crop(frame, m_align, geometry, foreign_corners=None, own_corners=None, bg_color=(0,0,0))`; `gpu_canonical_crop(frame_chw, m_align, geometry)`; `gpu_canonical_crop_batch(frame_chw, m_aligns, geometry)`; `apply_headtail_rotation(crop, m_align, direction, geometry)`; `invert_keypoints` unchanged.

Deleted: `compute_crop_dimensions`, `compute_native_crop_dimensions`, `compute_native_scale_affine`, `compute_alignment_affine`. Canvas dimensions are no longer a function of the detection, so a function that computes them from corners has no meaning.

- [ ] **Step 1: Update `tests/test_canonical_crop.py` to the new contract**

Replace every `compute_native_crop_dimensions(...)` / `compute_alignment_affine(...)` call with a `CanonicalGeometry` plus `canonical_affine`. Keep the foreign-mask tests as-is — masking is unchanged — and add:

```python
def test_all_crops_from_one_geometry_share_dimensions():
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    shapes = set()
    for major, minor, theta in [(20, 8, 0.0), (45, 30, 1.2), (12, 10, 2.9)]:
        corners = obb(100.0, 100.0, major, minor, theta)
        m, _, _ = canonical_affine(corners, g)
        shapes.add(extract_canonical_crop(frame, m, g).shape)
    assert len(shapes) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_crop.py -q`
Expected: FAIL — the new signatures do not exist yet.

- [ ] **Step 3: Rewrite the module**

`extract_canonical_crop` takes the geometry instead of `(canvas_w, canvas_h)`:

```python
def extract_canonical_crop(
    frame: np.ndarray,
    m_align: np.ndarray,
    geometry: CanonicalGeometry,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    foreign_corners: Optional[List[np.ndarray]] = None,
    own_corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    crop = cv2.warpAffine(
        frame,
        m_align,
        (geometry.canvas_w, geometry.canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if foreign_corners:
        _apply_foreign_mask_canonical(
            crop, m_align, foreign_corners, bg_color, own_corners=own_corners
        )
    return crop
```

`gpu_canonical_crop` and `gpu_canonical_crop_batch` take the geometry in place of `canvas_w, canvas_h`; their theta derivation is unchanged. `apply_headtail_rotation` keeps its 0/90/180/270 behaviour, reading dimensions from the geometry, and still returns `(rotated, M_canonical, M_inverse, offset_rad)`.

Delete `canonicalization/dataset.py` and its two test files; drop the corresponding exports from `__init__.py`. Delete `_find_metadata_path`, `_load_metadata_index`, `_resolve_annotations_for_image`, `MatMetadataCanonicalizer`, `get_canon_transform` with it.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_canonical_crop.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(canonicalization): crop.py onto Layer 1; delete the dead metadata canonicalizer

Canvas dimensions are no longer a function of the detection, so the four
functions that derived them from corners are removed. MatMetadataCanonicalizer
and get_canon_transform had no production callers -- only tests -- and the
per-image metadata path they implemented is subsumed by Layer 1 + Layer 2."
```

---

### Task 6: Inference stages onto the new contract

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py` (all nine entry points)
- Modify: `src/hydra_suite/core/inference/stages/pose.py:245-275, 349-412`
- Modify: `src/hydra_suite/core/inference/stages/cnn.py:67-68, 128-129`
- Modify: `src/hydra_suite/core/inference/stages/headtail.py:98-99, 246-247`
- Modify: `src/hydra_suite/core/inference/config.py:317-318, 807-812`
- Modify: `src/hydra_suite/core/inference/cache/keys.py`
- Modify: `src/hydra_suite/core/inference/pipeline.py`, `api.py`, `runner.py`
- Modify: `src/hydra_suite/core/tracking/ingest/streaming_payload.py`
- Test: `tests/test_inference_canonical_contract.py`; update `tests/test_inference_crops.py`, `test_inference_stages_crops.py`, `test_inference_extract_crops_batch.py`, `test_inference_foreign_mask.py`, `test_gpu_classifier_crop.py`, `test_inference_api_pose.py`, `test_inference_cache_keys.py`, `test_pipeline_pose_batch_canonical_geometry.py`, `tests/helpers/tiny_clip.py`

**Interfaces:**
- Consumes: `CanonicalGeometry`, `canonical_affine` (Task 1); `fit_to_model_input`, `apply_fit`, `fit_affine` (Task 2); corrected `(H, W)` reads (Task 3); `InferenceConfig.canonical` (Task 4)
- Produces: `InferenceConfig.canonical: CanonicalGeometry` replacing `canonical_aspect_ratio` and `canonical_margin`. Every crop entry point takes `geometry: CanonicalGeometry` instead of `(aspect_ratio, margin)` floats and returns uniformly shaped crops.

**Dispatch this task in two halves, committed and reviewed separately** — it is
the largest diff in the plan and a ten-file review is a poor gate:

- **6a** — `crops.py` alone: all nine entry points onto `CanonicalGeometry`,
  the batch-max pad deleted, both unused `aspect_ratio` parameters removed.
  Commit; review; then proceed.
- **6b** — the consumers: `pose.py`, `cnn.py`, `headtail.py`, `config.py`,
  `cache/keys.py`, `pipeline.py`, `api.py`, `runner.py`, `streaming_payload.py`.

The tests in Step 1 belong to 6b; 6a is gated by the updated existing crop tests.

Deletions this task must make, because a fixed canvas renders them meaningless:
the batch-max zero-pad in `_extract_canonical_cpu` (`crops.py:215-227`), the
slice-back in `run_pose_batch` (`pose.py:269` and `:402`), and the unused
`aspect_ratio` parameter on both `extract_classifier_crops` and
`extract_classifier_crops_gpu`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_canonical_contract.py
"""Every inference crop path obeys the Layer 1 + Layer 2 contract."""

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult


def _obb(n, rng):
    corners = []
    for i in range(n):
        major, minor = 20.0 + 10.0 * i, 8.0 + 3.0 * i
        hw, hh = major / 2, minor / 2
        base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        corners.append((base + np.array([100.0, 100.0])).astype(np.float32))
    corners = np.stack(corners)
    return OBBResult(
        frame_idx=0,
        centroids=np.full((n, 2), 100.0, dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 512.0, dtype=np.float32),
        shapes=np.full((n, 2), 2.0, dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=corners,
        detection_ids=np.arange(n, dtype=np.int64),
    )


def test_crops_are_uniform_regardless_of_animal_size():
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops

    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    runtime = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False)
    crops = extract_canonical_crops(frame, _obb(3, None), g, runtime)
    assert crops.shape[0] == 3
    assert crops.shape[2] == g.canvas_h
    assert crops.shape[3] == g.canvas_w


def test_cache_key_includes_the_canonical_geometry():
    from hydra_suite.core.inference.cache.keys import canonical_geometry_key

    a = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    b = CanonicalGeometry.from_reference(20.0, 2.44, 1.6)
    assert canonical_geometry_key(a) != canonical_geometry_key(b)


def test_every_cache_key_param_is_actually_written():
    """ENABLE_ASPECT_RATIO_FILTERING was a phantom key hashing None forever."""
    from hydra_suite.core.inference.cache.keys import _BGSUB_KEY_PARAMS
    from hydra_suite.paths import get_default_config

    defaults = get_default_config("default") or {}
    advanced = defaults.get("ADVANCED_CONFIG", {})
    for key in _BGSUB_KEY_PARAMS:
        assert (
            key in defaults or key.lower() in defaults or key.lower() in advanced
        ), f"{key} is hashed into the cache key but nothing writes it"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_inference_canonical_contract.py -q`
Expected: FAIL on all three.

- [ ] **Step 3: Convert the stages**

Thread `geometry: CanonicalGeometry` through every entry point in `crops.py`,
replacing per-detection `compute_native_crop_dimensions` calls with the shared
geometry. `_extract_canonical_cpu` becomes a plain `np.stack` — every crop is
already the same size. `extract_classifier_crops*` produce canonical crops; the
consumer fits them via Layer 2 rather than warping straight to the model input.

In `pose.py`, drop both slice-back blocks and build the inverse from the
composite:

```python
m_align, _, clipped = canonical_affine(corners, geometry)
fit = fit_to_model_input(geometry.canvas_wh, model_input_wh)
m_total = compose_affine(fit_affine(fit), m_align)
m_inv = cv2.invertAffineTransform(m_total)
```

Add `canonical_geometry_key(geometry) -> str` in `cache/keys.py` and include it
in the detection-cache key. Remove the phantom `ENABLE_ASPECT_RATIO_FILTERING`
entry from `_BGSUB_KEY_PARAMS` — nothing writes it, so it has always hashed
`None`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_inference_canonical_contract.py -q`
Expected: PASS, 3 tests.

Run the delta gate:
`python -m pytest tests/ -q -p no:randomly --ignore=tests/test_identity_postprocess.py -k "inference or crop or pose or canonical"`
Expected: no failure names absent from the baseline.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): every crop path onto the shared canonical geometry

Fixed canvas dimensions delete the batch-max zero-pad and the slice-back that
recovered native extents from it. Classifier crops stop warping straight to the
model input and go through Layer 2 like everything else. The canonical geometry
enters the detection-cache key; the phantom ENABLE_ASPECT_RATIO_FILTERING entry,
which hashed None forever, is removed."
```

---

### Task 7: Identity — head-tail, crop dataset, oriented video

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/headtail.py:517-560, 594-615, 748-770`
- Modify: `src/hydra_suite/core/identity/dataset/generator.py:75, 92-93, 152, 323-350, 434-448, 495-525, 776-791`
- Modify: `src/hydra_suite/core/identity/dataset/oriented_video.py:180, 207, 668, 883, 1198-1220`
- Modify: `src/hydra_suite/trackerkit/gui/workers/crops_worker.py:306, 657, 1456`
- Test: `tests/test_canonical_dataset_provenance.py`

**Interfaces:**
- Consumes: `CanonicalGeometry`, `canonical_affine`, `extract_canonical_crop` (Tasks 1, 5); Layer 2 (Task 2)
- Produces: `metadata.json` gains `parameters.canonical` = `CanonicalGeometry.to_dict()` plus `clipped_count` and `worst_overflow_ratio`; `read_canonical_provenance(dataset_dir: Path) -> CanonicalGeometry | None` in `core/identity/dataset/naming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_dataset_provenance.py
import json

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.identity.dataset.naming import read_canonical_provenance


def test_provenance_round_trips(tmp_path):
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "canonical": {
                        **g.to_dict(),
                        "clipped_count": 3,
                        "worst_overflow_ratio": 1.08,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert read_canonical_provenance(tmp_path) == g


def test_missing_provenance_is_none_not_a_guess(tmp_path):
    assert read_canonical_provenance(tmp_path) is None


def test_legacy_metadata_without_the_block_is_none(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"parameters": {"padding_fraction": 0.1}}), encoding="utf-8"
    )
    assert read_canonical_provenance(tmp_path) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_dataset_provenance.py -q`
Expected: FAIL — `read_canonical_provenance` does not exist.

- [ ] **Step 3: Implement**

Collapse the three `HeadTailAnalyzer` crop sites onto one Layer 1 call plus
Layer 2; delete the `128 x 128/ref_AR` fallback canvas at `headtail.py:600-605`
and `:755-760` — there is no per-analyzer canvas any more.

In `generator.py`, replace `compute_native_crop_dimensions` at both sites with
the session geometry, accumulate `clipped_count` and `worst_overflow_ratio`
across the run, and write the `parameters.canonical` block in the existing
`metadata.json` writer at `:776`.

In `oriented_video.py`, delete `_compute_affine` (`:1198-1220`) and call
`canonical_affine`; write the same block into its own export metadata.

Add `read_canonical_provenance` to `naming.py`. Return `None` — never a
default — when the block is absent, so an unknown-provenance dataset is
visibly unknown.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_canonical_dataset_provenance.py -q`
Expected: PASS, 3 tests.

Run the delta gate: `-k "headtail or dataset or oriented or generator"`.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(identity): head-tail, crop export and oriented video onto Layer 1

All three canonicalization implementations are now one. Crop datasets record
their geometry in the metadata.json parameters block that already existed,
rather than a new sidecar next to it -- the repo already has two incompatible
sidecar naming conventions and does not need a third."
```

---

### Task 8: Training — one fit for ClassKit and PoseKit

**Files:**
- Modify: `src/hydra_suite/training/runner.py:1297, 1337, 1635, 1664`
- Modify: `src/hydra_suite/core/identity/pose/vitpose/training/dataset.py`
- Modify: `src/hydra_suite/posekit/core/vitpose_training.py`
- Create: `src/hydra_suite/training/canonical_transform.py`
- Test: `tests/test_train_inference_fit_identity.py`

**Interfaces:**
- Consumes: Layer 2 (Task 2); `(H, W)` `input_size` (Task 3)
- Produces: `CanonicalFitTransform` — a callable torchvision-compatible transform wrapping `fit_to_model_input` + `apply_fit`, used by both training data loading and inference preprocessing

**This is the task the whole spec exists for.** The test below is the structural guard: if it passes, train and inference cannot silently drift.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_inference_fit_identity.py
"""The same image, fitted by training and by inference, is byte-identical."""

import numpy as np
import pytest

from hydra_suite.training.canonical_transform import CanonicalFitTransform


@pytest.fixture
def crop():
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)


@pytest.mark.parametrize("model_hw", [(224, 224), (64, 128), (96, 160)])
def test_classkit_train_matches_inference(crop, model_hw):
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    train_out = CanonicalFitTransform(model_hw)(crop)
    fit = fit_to_model_input((crop.shape[1], crop.shape[0]), (model_hw[1], model_hw[0]))
    infer_out = apply_fit(crop, fit)
    np.testing.assert_array_equal(np.asarray(train_out), infer_out)


@pytest.mark.parametrize("model_hw", [(256, 192), (256, 256)])
def test_posekit_train_matches_inference(crop, model_hw):
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    train_out = CanonicalFitTransform(model_hw)(crop)
    fit = fit_to_model_input((crop.shape[1], crop.shape[0]), (model_hw[1], model_hw[0]))
    np.testing.assert_array_equal(np.asarray(train_out), apply_fit(crop, fit))


def test_transform_rejects_float_input(crop):
    with pytest.raises(TypeError):
        CanonicalFitTransform((224, 224))(crop.astype(np.float32) / 255.0)


def test_no_resize_call_survives_in_the_runner():
    from pathlib import Path

    runner = (
        Path(__file__).resolve().parents[1]
        / "src/hydra_suite/training/runner.py"
    )
    assert "Resize((sz, sz))" not in runner.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_train_inference_fit_identity.py -q`
Expected: FAIL — `training.canonical_transform` does not exist; `Resize((sz, sz))` still present.

- [ ] **Step 3: Implement**

```python
# src/hydra_suite/training/canonical_transform.py
"""The one transform training and inference share.

A torchvision Resize here and a cv2.resize there is exactly how train and
inference drift apart: PIL antialiases on downscale, cv2.INTER_LINEAR does not.
Both ends call this.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input


class CanonicalFitTransform:
    """Fit a uint8 BGR image into ``model_hw`` (H, W) by isotropic letterbox."""

    def __init__(self, model_hw: tuple[int, int]) -> None:
        self.model_hw = (int(model_hw[0]), int(model_hw[1]))

    def __call__(self, image) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            raise TypeError(
                f"CanonicalFitTransform requires uint8 input, got {arr.dtype}"
            )
        h, w = arr.shape[:2]
        fit = fit_to_model_input((w, h), (self.model_hw[1], self.model_hw[0]))
        return apply_fit(arr, fit)
```

Replace all four `transforms.Resize((sz, sz))` occurrences with
`CanonicalFitTransform(params.input_size)`, in both train and eval transform
lists. Decode with `cv2.imread` on both paths so the PIL/EXIF difference
between the torchvision and tiny loaders disappears. Point the ViTPose training
dataset at the same transform.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_train_inference_fit_identity.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(training): one fit shared by training and inference in every kit

transforms.Resize((sz,sz)) squared every image and antialiased on downscale
while inference used cv2.INTER_LINEAR and did not. Both ends now call
CanonicalFitTransform, and a per-kit byte-identity test makes drift a test
failure rather than a silent accuracy loss."
```

---

### Task 9: YOLO-classify — close the hole in the guard

`_forward_yolo` hands raw crops to Ultralytics, which applies `Resize(shortest_edge)` + `CenterCrop` at inference and `RandomResizedCrop(scale=0.08-1.0)` at training. A 128x64 canonical crop is upscaled and centre-cropped to 224x224, discarding half the animal's length, and `input_size` is ignored. Task 8's guard cannot be written for this family until that is fixed.

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/backend.py:950-953`
- Modify: `src/hydra_suite/training/runner.py:88-147`
- Test: `tests/test_yolo_classify_canonical_fit.py`

**Interfaces:**
- Consumes: Layer 2 (Task 2), `CanonicalFitTransform` (Task 8)
- Produces: no new public API; `_forward_yolo` pre-fits to a square so Ultralytics' centre-crop is a no-op

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yolo_classify_canonical_fit.py
"""YOLO-classify must see the whole animal, not ultralytics' centre crop."""

import numpy as np


def test_prefit_makes_centre_crop_a_noop():
    from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input

    # A 128x64 canonical crop pre-fitted to a square is already square, so
    # ultralytics' Resize(shortest_edge) + CenterCrop(size) cannot remove
    # anything: shortest edge == longest edge.
    fit = fit_to_model_input((128, 64), (224, 224))
    out = apply_fit(np.full((64, 128, 3), 200, dtype=np.uint8), fit)
    assert out.shape[0] == out.shape[1]


def test_forward_yolo_prefits(monkeypatch):
    from hydra_suite.core.identity.classification import backend as backend_mod

    seen = []

    class _StubYolo:
        def __call__(self, crops, **kwargs):
            seen.extend(np.asarray(c) for c in crops)
            return []

    obj = object.__new__(backend_mod.ClassifierBackend)
    obj._model = _StubYolo()
    obj._metadata = type("M", (), {"input_size": (224, 224), "monochrome": False})()
    backend_mod.ClassifierBackend._forward_yolo(
        obj, [np.full((64, 128, 3), 200, dtype=np.uint8)]
    )
    assert seen[0].shape[:2] == (224, 224)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_yolo_classify_canonical_fit.py -q`
Expected: FAIL — `_forward_yolo` passes crops through unchanged.

- [ ] **Step 3: Implement**

In `_forward_yolo`, fit every crop to `self._metadata.input_size` before the
call. For training, pass the pre-fitted dataset to the Ultralytics CLI and set
`scale=1.0` so `RandomResizedCrop` degenerates to a centre crop of the whole
already-square image.

**If Ultralytics still alters the pre-fitted image, that is accepted, not a
blocker.** YOLO-classify remains a supported but **known-lossy** path: the
pre-fit gets it as close as the vendor's pipeline allows, and the geometry
guarantee explicitly does not extend to it. Do not stop, and do not declare it
unsupported. Instead:

1. Measure what survives — round-trip a pre-fitted image through the vendor
   transform and record the actual difference, so the loss is quantified rather
   than assumed.
2. Exempt the family from the byte-identity guard with an explicit reason, never
   a silent gap:

```python
@pytest.mark.xfail(
    reason="YOLO-classify runs ultralytics' own Resize+CenterCrop; the canonical "
    "geometry guarantee does not extend to it. Known-lossy by decision, "
    "2026-08-05. Follow-up: replace or bypass the vendor transform.",
    strict=False,
)
def test_yolo_classify_train_matches_inference():
    ...
```

3. Log a one-line warning when a YOLO-classify backend loads, naming the
   limitation, so an operator choosing it knows what they are choosing.
4. Record the follow-up in the "Deliberately out of scope" section.

The two tests in Step 1 still apply: pre-fitting must happen, and must produce a
square. Only the *train/inference byte-identity* property is waived.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_yolo_classify_canonical_fit.py -q`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "fix(identity): pre-fit YOLO-classify crops so ultralytics cannot re-crop them"
```

---

### Task 10: Stamp the geometry on the model

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py:786-800, 821-826`
- Create: `src/hydra_suite/core/inference/canonical_meta.py`
- Modify: `src/hydra_suite/core/inference/config.py`
- Test: `tests/test_canonical_model_stamp.py`

**Interfaces:**
- Consumes: `CanonicalGeometry` (Task 1)
- Produces: `<model>.canonical_meta.json` written at publish (append convention, matching `.slice_meta.json` and `.runtime_meta.json`); `read_canonical_meta(model_path: Path) -> CanonicalGeometry | None`; `warn_on_geometry_mismatch(model_path, session_geometry) -> str | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_model_stamp.py
import json

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.canonical_meta import (
    read_canonical_meta,
    warn_on_geometry_mismatch,
)


def test_stamp_round_trips(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    (tmp_path / "m.pt.canonical_meta.json").write_text(
        json.dumps(g.to_dict()), encoding="utf-8"
    )
    assert read_canonical_meta(model) == g


def test_unstamped_model_is_none(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    assert read_canonical_meta(model) is None
    assert warn_on_geometry_mismatch(model, CanonicalGeometry.from_reference(20.0, 2.0, 1.3)) is None


def test_mismatch_is_reported(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    trained = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    (tmp_path / "m.pt.canonical_meta.json").write_text(
        json.dumps(trained.to_dict()), encoding="utf-8"
    )
    session = CanonicalGeometry.from_reference(20.0, 2.44, 2.0)
    msg = warn_on_geometry_mismatch(model, session)
    assert msg is not None and "margin" in msg
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_model_stamp.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Mirror `core/inference/slice_meta.py` exactly: same append naming, same
tolerant read, same mirroring into registry metadata at
`model_publish.py:821-826`. `warn_on_geometry_mismatch` returns `None` for an
unstamped model — pre-existing checkpoints must keep loading — and a
human-readable diff naming the fields that differ otherwise.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_canonical_model_stamp.py -q`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(training): stamp canonical geometry on published models

Every model is being retrained under a new convention; without a stamp a
checkpoint carries no record of which one it was trained under. Follows the
.slice_meta.json pattern, including the registry mirror."
```

---

### Task 11: Sweep the dead config surface

**Files:**
- Modify: `src/hydra_suite/resources/configs/default.json`
- Modify: `src/hydra_suite/core/inference/cache/keys.py`
- Test: `tests/test_config_surface_is_live.py`

**Interfaces:**
- Consumes: nothing
- Produces: a regression test asserting no shipped config key is unreferenced

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_surface_is_live.py
"""A shipped config key that nothing reads is a lie to the operator."""

import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hydra_suite"

# Keys that no longer exist under global canonicalization.
RETIRED = {
    "identity_crop_size_multiplier",
    "identity_crop_min_size",
    "identity_crop_max_size",
}


def test_retired_crop_keys_are_gone():
    defaults = json.loads(
        (SRC / "resources/configs/default.json").read_text(encoding="utf-8")
    )
    assert RETIRED.isdisjoint(defaults.keys())


def test_no_shipped_key_is_unreferenced():
    defaults = json.loads(
        (SRC / "resources/configs/default.json").read_text(encoding="utf-8")
    )
    sources = "\n".join(
        p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
    )
    unreferenced = [
        k for k in defaults if k not in sources and k.upper() not in sources
    ]
    assert unreferenced == [], f"shipped but never read: {unreferenced}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_config_surface_is_live.py -q`
Expected: FAIL — the three `identity_crop_*` keys plus other unreferenced keys.

- [ ] **Step 3: Delete the retired keys**

Remove the three `identity_crop_*` keys — old crop-geometry names that would be
mistaken for the new contract. For any *other* key the second test surfaces,
do **not** delete it: add it to an explicit `KNOWN_UNWIRED` allow-list in the
test with a one-line reason each, and report the list. Those are the backlog
items from the config review (`MIN_RESPAWN_DISTANCE`, `W_POSE_*`,
`ENABLE_FRAGMENT_SCORING`, ...), each of which is a separate decision about
tracking behaviour, not about crops.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_config_surface_is_live.py -q`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "chore(config): retire the dead identity_crop_* keys; guard the surface"
```

---

### Task 12: Re-baseline and verify on both platforms

This change is **intentionally not equivalent**. The harness is used to record a new baseline and to confirm the change is confined to crop consumers — not to pass.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-global-canonicalization-design.md` (record the measured re-baseline)
- Create: `docs/superpowers/specs/notes/global-canonicalization-rebaseline.md`

- [ ] **Step 1: Capture the delta gate on both trees**

The whole suite cannot run in one process (modal-dialog hangs and a SIGABRT), so
the gate runs as the same `-k` selection on both trees. Use this selection, which
covers every area this plan touches:

```bash
SEL="pose or crop or cli_config or headless or inference or canonical or vitpose or sleap or classifier or training or dataset or oriented or margin"
export KMP_DUPLICATE_LIB_OK=TRUE

# baseline: current main
cd <repo-root> && PYTHONPATH=$PWD/src python -m pytest tests/ -q -p no:randomly \
  --ignore=tests/test_identity_postprocess.py -k "$SEL" > /tmp/gate_main.txt 2>&1

# branch
cd .worktrees/feat-global-canonicalization && PYTHONPATH=$PWD/src python -m pytest tests/ \
  -q -p no:randomly --ignore=tests/test_identity_postprocess.py -k "$SEL" \
  > /tmp/gate_branch.txt 2>&1

grep '^FAILED' /tmp/gate_main.txt   | sed 's/ - .*//' | sort > /tmp/f_main.txt
grep '^FAILED' /tmp/gate_branch.txt | sed 's/ - .*//' | sort > /tmp/f_branch.txt
comm -23 /tmp/f_branch.txt /tmp/f_main.txt   # must be empty
```

Expected: empty. Any name here is a regression, not a re-baseline.

Capture the baseline **fresh from current `main`** at the start of execution, not
from an earlier run — `main` has moved since this plan was written (`e6882c0e`
added the pose crop-dtype fix and two test files). Baselines age.

Then sweep the files the `-k` selection misses, per file, to catch collateral:

```bash
for f in tests/test_main_window_config_persistence.py tests/test_detectkit_main_window.py; do
  PYTHONPATH=$PWD/src python -m pytest "$f" -q -p no:randomly 2>&1 | tail -3
done
```

- [ ] **Step 2: Fetch fixtures and run the matrix on MPS**

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_canon RUNTIME=mps bash tools/equivalence/run_matrix.sh
```

Expected, and each must be checked rather than assumed:
- `fly_obb` and `worm_bgsub` **byte-identical** — they run no crop-consuming stage, so they are the control proving the change is confined.
- `emi_obb_identity`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`, `ant_cnn_identity` differ. Record the magnitude per clip.
- Verify `wc -l` > 1 on every CSV before trusting any comparison — an inactive conda env yields empty CSVs that falsely compare equivalent.

- [ ] **Step 3: Repeat on CUDA**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch && git checkout <branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_canon RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh \
  > /tmp/equiv_cuda.log 2>&1 &
```

- [ ] **Step 4: Record the new baseline**

Write `notes/global-canonicalization-rebaseline.md` with per-clip, per-platform numbers, the two control clips' byte-identity, and the clipped-detection counts each clip reported. This document is what a future change compares against.

- [ ] **Step 5: Commit**

```bash
git add docs/
git commit -m "docs(specs): record the global-canonicalization re-baseline on MPS and CUDA"
```

---

## Retraining (operator, after merge)

Every model was trained on old-convention crops: head-tail, CNN identity, ViTPose, SLEAP. Head-tail warrants explicit measurement rather than assumption — `stages/crops.py:238-247` records that merely adding a resample step flipped 1-2% of direction decisions, and head-tail sits upstream of tracking identity. After retraining, measure direction agreement against the current model on a held-out clip and report it, rather than inferring health from tracking output.

## Follow-up work this plan creates

**YOLO-classify geometry.** Task 9 leaves it a supported but known-lossy path:
pre-fitted as close as Ultralytics allows, with the byte-identity guarantee
explicitly not extending to it. The follow-up is to replace or bypass the vendor
transform — either by calling the underlying model directly with an
already-preprocessed tensor, or by exporting it to a runtime we preprocess for —
so it rejoins the single geometry contract. Decided 2026-08-05; deliberately
deferred until global canonicalization has landed, so the two changes can be
attributed separately.

## Deliberately out of scope

Backlog items confirmed during review, each a separate decision about tracking behaviour rather than about crops: `SAVE_CONFIDENCE_METRICS` (fixed already, `8900191d`), `MIN_RESPAWN_DISTANCE` (dead UI dial), `POSE_DIRECTION_MIN_VALID_KEYPOINTS` (1 in inference vs 3 in the worker), `DETECTION_BATCH_SIZE` (bgsub batching pinned at 1), sequential-OBB stage-2 knobs (unreachable), `W_POSE_*` (written, never read), `ENABLE_FRAGMENT_SCORING` (inert checkbox; the fragment solver cannot run headless at all), YOLO-pose conf/IoU (hardwired), tiny monochrome models validating on colour, `predict_batch(crops, input_is_bgr=...)` `TypeError` on two fallback paths, and `fallback_input_size` never passed at publish. Also out of scope, owned by the deferred registry spec: `TrainingRole` has no pose member, and picker surface.
