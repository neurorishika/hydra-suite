# ClassKit Decode/Resample Augmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in, decode/resample-robustness augmentations to the ClassKit classifier training path so colortag/identity models trained with them tolerate GPU-Fast/NVDEC and cross-codec decode differences, without collecting decode-specific data.

**Architecture:** One shared numpy module `training/augmentation.py` with two pure, `rng`-seeded functions (`simulate_decode_color`, `simulate_resample`) operating on RGB uint8 `(H,W,3)` arrays, plus a PIL wrapper for the torchvision path. Two new `AugmentationProfile` fields (`decode_color_sim`, `resample_sim`, default `0.0`) carry the settings. The functions are injected into all three in-process classifier trainers and surfaced as two GUI controls. Both augmentations default **off** (backward-compatible).

**Tech Stack:** Python, NumPy, OpenCV (cv2), PyTorch (`grid_sample` for the resampler + torchvision transforms), PyQt (classkit dialog), pytest.

## Global Constraints

- **Branch:** all work lands on `classkit-decode-augmentation` (off merged `main` `8c0f3277`).
- **Backward-compatible:** both new fields default `0.0` (= exactly current behavior). A profile/spec with the new fields absent must still construct and train identically to today. Never change behavior when both fields are `0.0`.
- **RGB uint8 `(H,W,3)` is the common currency** for both augmentation functions: input and output are `np.uint8`, shape `(H,W,3)`, RGB order. No normalization inside them (callers normalize later).
- **Determinism:** both functions take an explicit `rng` (a `numpy.random.Generator`). Given the same `rng` state + input they produce identical output. Do NOT use the global `random`/`np.random`/torch RNG inside them. (The rest of the trainer's flips/jitter remain on their existing unseeded global RNG — out of scope to change.)
- **Scope = ClassKit classifier training only.** The three in-process trainers in `training/runner.py` and the ClassKit training dialog. Do NOT touch YOLO-classify (Ultralytics subprocess), the trackerkit/detectkit training dialogs, or inference code.
- **Tests run inside the `hydra-mps` conda env** (base miniforge `python` has a broken torch — `torch.utils._pytree has no attribute tree_flatten`). Every test command: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps` then `PYTHONPATH=$PWD/src python -m pytest ...`. If you see the `tree_flatten` error you forgot to activate conda.
- **Pre-commit hook reformats then ABORTS the first commit.** Always commit twice: `git add -A && git commit -m "..."` then `git add -A && git commit -m "..."` again. Verify `git status` clean afterward. flake8 flags unused imports (F401) — only import what you use.
- **Commit as the configured git user** — NO `Co-Authored-By: Claude` trailer.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/hydra_suite/training/augmentation.py` | Shared decode/resample augmentation | **Create**: 2 functions + YCbCr helpers + PIL wrapper |
| `src/hydra_suite/training/contracts.py` | `AugmentationProfile` (slots dataclass, line 124) | Add `decode_color_sim`, `resample_sim` fields |
| `src/hydra_suite/training/runner.py` | 3 in-process trainers | Inject aug into tiny (`_apply_tiny_augmentation`) + 2 torchvision pipelines |
| `src/hydra_suite/classkit/gui/dialogs/training.py` | Aug tab (line 1566), `get_settings` (2295), summary (2044) | 2 spinboxes + 2 dict keys + summary text |
| `src/hydra_suite/classkit/gui/main_window.py` | `_make_training_spec` build site (7222) | Pass 2 new fields into `AugmentationProfile(...)` |
| `tests/test_training_augmentation.py` | Existing tiny-aug tests | Extend: tiny decode/resample injection |
| `tests/test_augmentation_module.py` | New | Unit tests for the 2 functions + PIL wrapper |
| `tests/test_classkit_main_window.py` | Existing (line 1962) | Extend: 2 new fields map through `_make_training_spec` |

---

## Task 1: Shared augmentation module

**Files:**
- Create: `src/hydra_suite/training/augmentation.py`
- Create: `tests/test_augmentation_module.py`

**Interfaces:**
- Produces:
  - `simulate_decode_color(rgb_u8: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray` — RGB uint8 `(H,W,3)` in/out. With probability `strength`, re-decodes the crop through a randomly sampled `{BT.601,BT.709}×{limited,full}` YCbCr convention (blended by a random severity) + mild brightness/contrast jitter. Identity when `strength<=0` or the rng roll skips.
  - `simulate_resample(rgb_u8: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray` — RGB uint8 in/out. With probability `prob`, re-warps through either cv2 `warpAffine` (random interp + sub-pixel shift) or torch `grid_sample` (bilinear). Identity when `prob<=0` or skipped.
  - `make_decode_resample_pil_transform(profile, rng: np.random.Generator)` — returns a callable `PIL.Image -> PIL.Image` applying both functions (gated on `profile.decode_color_sim`/`profile.resample_sim`), for the torchvision pipeline. Returns `None` if both are `<=0` (nothing to insert).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_augmentation_module.py`:

```python
"""Unit tests for the decode/resample training augmentations."""

import numpy as np
import pytest

from hydra_suite.training.augmentation import (
    simulate_decode_color,
    simulate_resample,
    make_decode_resample_pil_transform,
)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(32, 24, 3), dtype=np.uint8)


def test_decode_color_shape_dtype_preserved():
    out = simulate_decode_color(_img(), 1.0, np.random.default_rng(1))
    assert out.shape == (32, 24, 3) and out.dtype == np.uint8


def test_decode_color_identity_at_zero_strength():
    img = _img()
    out = simulate_decode_color(img, 0.0, np.random.default_rng(1))
    assert np.array_equal(out, img)


def test_decode_color_changes_pixels_when_applied():
    img = _img()
    # strength 1.0 => always applies; a sampled non-BT601-limited convention shifts color.
    # Try several rng seeds; at least one must change pixels (identity convention is possible).
    changed = any(
        not np.array_equal(simulate_decode_color(img, 1.0, np.random.default_rng(s)), img)
        for s in range(8)
    )
    assert changed


def test_decode_color_deterministic_under_fixed_rng():
    img = _img()
    a = simulate_decode_color(img, 1.0, np.random.default_rng(42))
    b = simulate_decode_color(img, 1.0, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_decode_color_no_overflow():
    out = simulate_decode_color(np.full((8, 8, 3), 255, np.uint8), 1.0, np.random.default_rng(3))
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255
    assert not np.isnan(out.astype(np.float32)).any()


def test_resample_shape_dtype_preserved():
    out = simulate_resample(_img(), 1.0, np.random.default_rng(1))
    assert out.shape == (32, 24, 3) and out.dtype == np.uint8


def test_resample_identity_at_zero_prob():
    img = _img()
    assert np.array_equal(simulate_resample(img, 0.0, np.random.default_rng(1)), img)


def test_resample_deterministic_under_fixed_rng():
    img = _img()
    a = simulate_resample(img, 1.0, np.random.default_rng(7))
    b = simulate_resample(img, 1.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_pil_transform_none_when_both_off():
    prof = type("P", (), {"decode_color_sim": 0.0, "resample_sim": 0.0})()
    assert make_decode_resample_pil_transform(prof, np.random.default_rng(0)) is None


def test_pil_transform_roundtrips_pil_image():
    from PIL import Image

    prof = type("P", (), {"decode_color_sim": 1.0, "resample_sim": 1.0})()
    tf = make_decode_resample_pil_transform(prof, np.random.default_rng(0))
    assert tf is not None
    img = Image.fromarray(_img())
    out = tf(img)
    assert isinstance(out, Image.Image) and out.size == img.size
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_augmentation_module.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.training.augmentation'`.

- [ ] **Step 3: Create the module**

Create `src/hydra_suite/training/augmentation.py`:

```python
"""Decode/resample-robustness augmentations for classifier training.

Colortag/identity classifiers read color as their signal and are trained on ONE
decode+resample convention (cv2 CPU decode, BT.601 limited range, cv2.warpAffine
crops). At inference the GPU-Fast/NVDEC path decodes YCbCr->RGB with a different
matrix/range and warps with torch grid_sample, leaving a small but systematic
shift that tips borderline predictions. Rather than chase bit-exact matching per
codec/GPU/driver, these augmentations train the model to be invariant to the
*space* of decode and resample transforms. See
docs/superpowers/specs/2026-07-22-classkit-decode-robust-augmentation-design.md.

Both functions operate on RGB uint8 (H, W, 3) arrays and take an explicit
``numpy.random.Generator`` so training augmentation is reproducible given a seed.
"""

from __future__ import annotations

import numpy as np

# (Kr, Kb) luma coefficients for the two common matrices.
_BT601 = (0.299, 0.114)
_BT709 = (0.2126, 0.0722)


def _rgb_to_ycbcr(rgb: np.ndarray, kr: float, kb: float, full: bool) -> np.ndarray:
    """RGB float [0,255] -> YCbCr float, given luma coeffs and range."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    kg = 1.0 - kr - kb
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb))
    cr = (r - y) / (2.0 * (1.0 - kr))
    if full:
        return np.stack([y, cb + 128.0, cr + 128.0], axis=-1)
    return np.stack(
        [16.0 + y * (219.0 / 255.0),
         128.0 + cb * (224.0 / 255.0),
         128.0 + cr * (224.0 / 255.0)],
        axis=-1,
    )


def _ycbcr_to_rgb(ycc: np.ndarray, kr: float, kb: float, full: bool) -> np.ndarray:
    """YCbCr float -> RGB float [0,255], given luma coeffs and range."""
    Y, Cb, Cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    kg = 1.0 - kr - kb
    if full:
        y = Y
        cb = Cb - 128.0
        cr = Cr - 128.0
    else:
        y = (Y - 16.0) * (255.0 / 219.0)
        cb = (Cb - 128.0) * (255.0 / 224.0)
        cr = (Cr - 128.0) * (255.0 / 224.0)
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.stack([r, g, b], axis=-1)


def simulate_decode_color(rgb_u8: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Re-decode the crop through a randomly sampled YCbCr<->RGB convention.

    With probability ``strength``: recover YCbCr using the training origin
    (BT.601 limited), then re-decode with a randomly sampled matrix in
    {BT.601, BT.709} and range in {limited, full}, blended toward the original by
    a random severity, followed by mild brightness/contrast jitter. Reproduces the
    decode variation (matrix + range) the model meets at inference without NVDEC
    data. Identity RGB uint8 in / RGB uint8 out.
    """
    if strength <= 0.0 or rng.random() >= strength:
        return rgb_u8
    rgb = rgb_u8.astype(np.float32)
    ycc = _rgb_to_ycbcr(rgb, *_BT601, full=False)  # recover origin YCbCr
    kr, kb = _BT601 if rng.random() < 0.5 else _BT709
    full = rng.random() < 0.5
    shifted = _ycbcr_to_rgb(ycc, kr, kb, full)
    blend = float(rng.uniform(0.25, 1.0))
    out = rgb + (shifted - rgb) * blend
    # mild photometric jitter
    out *= float(rng.uniform(0.95, 1.05))                 # brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * float(rng.uniform(0.95, 1.05)) + mean  # contrast
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def simulate_resample(rgb_u8: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Re-warp the crop through an alternate resampler with probability ``prob``.

    Randomly chooses cv2.warpAffine (random INTER_LINEAR/CUBIC/AREA + sub-pixel
    translation) or torch grid_sample (bilinear, align_corners=False), simulating
    the cv2.warpAffine-vs-grid_sample geometry gap between training and the GPU
    inference crop path. RGB uint8 in / RGB uint8 out.
    """
    if prob <= 0.0 or rng.random() >= prob:
        return rgb_u8
    import cv2

    h, w = rgb_u8.shape[:2]
    dx = float(rng.uniform(-0.5, 0.5))
    dy = float(rng.uniform(-0.5, 0.5))
    if rng.random() < 0.5:
        interp = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(rgb_u8, M, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float()
    theta = torch.tensor([[[1.0, 0.0, 2.0 * dx / w], [0.0, 1.0, 2.0 * dy / h]]], dtype=torch.float32)
    grid = F.affine_grid(theta, list(t.shape), align_corners=False)
    warped = F.grid_sample(t, grid, mode="bilinear", align_corners=False)
    return warped[0].permute(1, 2, 0).clamp(0.0, 255.0).round().byte().numpy()


def make_decode_resample_pil_transform(profile, rng: np.random.Generator):
    """Return a callable ``PIL.Image -> PIL.Image`` applying the two augmentations,
    or ``None`` when both are off. For the torchvision Compose pipeline: insert it
    BEFORE ToTensor()/Normalize (train transform only).
    """
    decode = float(getattr(profile, "decode_color_sim", 0.0) or 0.0)
    resample = float(getattr(profile, "resample_sim", 0.0) or 0.0)
    if decode <= 0.0 and resample <= 0.0:
        return None

    from PIL import Image

    def _tf(pil_img):
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        if decode > 0.0:
            arr = simulate_decode_color(arr, decode, rng)
        if resample > 0.0:
            arr = simulate_resample(arr, resample, rng)
        return Image.fromarray(arr)

    return _tf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_augmentation_module.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(training): decode/resample robustness augmentation module"
git add -A && git commit -m "feat(training): decode/resample robustness augmentation module"
git status
```

---

## Task 2: `AugmentationProfile` config fields

**Files:**
- Modify: `src/hydra_suite/training/contracts.py:124-145` (`AugmentationProfile`)
- Test: `tests/test_training_augmentation.py` (add a serialization test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `AugmentationProfile.decode_color_sim: float = 0.0`, `AugmentationProfile.resample_sim: float = 0.0`. Serialized transitively via `TrainingRunSpec.to_dict()` → `dataclasses.asdict`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_training_augmentation.py`:

```python
def test_augmentation_profile_new_decode_fields_default_off_and_serialize():
    from dataclasses import asdict
    from hydra_suite.training.contracts import AugmentationProfile

    p = AugmentationProfile()
    assert p.decode_color_sim == 0.0
    assert p.resample_sim == 0.0
    d = asdict(p)
    assert d["decode_color_sim"] == 0.0 and d["resample_sim"] == 0.0

    p2 = AugmentationProfile(decode_color_sim=0.5, resample_sim=0.3)
    assert asdict(p2)["decode_color_sim"] == 0.5
    assert asdict(p2)["resample_sim"] == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_training_augmentation.py::test_augmentation_profile_new_decode_fields_default_off_and_serialize -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'decode_color_sim'`.

- [ ] **Step 3: Add the fields**

In `src/hydra_suite/training/contracts.py`, in the `AugmentationProfile` dataclass (after `contrast: float = 0.0`, before `monochrome: bool = False`), add:

```python
    contrast: float = 0.0
    decode_color_sim: float = 0.0  # 0=off; ~0.5 recommended. P(apply) decode-color re-sim.
    resample_sim: float = 0.0      # 0=off; ~0.3 recommended. P(apply) alternate resampler.
    monochrome: bool = False
```

(Insert the two lines; keep `monochrome`, `args`, `label_expansion` as they are. New fields go before the non-default... note all fields here have defaults, so ordering is safe.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_training_augmentation.py::test_augmentation_profile_new_decode_fields_default_off_and_serialize -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(training): AugmentationProfile decode_color_sim/resample_sim fields"
git add -A && git commit -m "feat(training): AugmentationProfile decode_color_sim/resample_sim fields"
git status
```

---

## Task 3: Inject into the tiny trainer

**Files:**
- Modify: `src/hydra_suite/training/runner.py` — `_apply_tiny_augmentation` (327-365) + `TinyDataset` (around 305-320, `__init__`/`__getitem__`) + its construction (383-388)
- Test: `tests/test_training_augmentation.py` (add tiny decode/resample tests)

**Interfaces:**
- Consumes: `simulate_decode_color`, `simulate_resample` (Task 1); `profile.decode_color_sim`, `profile.resample_sim` (Task 2).
- Produces: `_apply_tiny_augmentation(img, augment, profile, rng=None)` — new optional `rng` param (defaults to a fresh `np.random.default_rng()` when `None`, preserving the existing 3-arg callers). Applies the two augmentations (gated on the profile fields) after the existing flips/jitter. `TinyDataset` gains `self._rng = np.random.default_rng(seed)` and passes it in.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training_augmentation.py`:

```python
def test_tiny_augmentation_decode_color_changes_image_when_on():
    import numpy as np
    from hydra_suite.training.contracts import AugmentationProfile
    from hydra_suite.training.runner import _apply_tiny_augmentation

    rng_img = np.random.default_rng(0)
    img = rng_img.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    prof = AugmentationProfile(enabled=True, decode_color_sim=1.0)
    changed = any(
        not np.array_equal(
            _apply_tiny_augmentation(img, True, prof, rng=np.random.default_rng(s)), img
        )
        for s in range(8)
    )
    assert changed


def test_tiny_augmentation_off_by_default_leaves_decode_untouched():
    import numpy as np
    from hydra_suite.training.contracts import AugmentationProfile
    from hydra_suite.training.runner import _apply_tiny_augmentation

    img = np.full((16, 16, 3), 120, np.uint8)
    prof = AugmentationProfile(enabled=True)  # decode_color_sim/resample_sim default 0.0
    # No flips/jitter either -> fully identity.
    out = _apply_tiny_augmentation(img, True, prof, rng=np.random.default_rng(0))
    assert np.array_equal(out, img)


def test_tiny_augmentation_deterministic_under_fixed_rng():
    import numpy as np
    from hydra_suite.training.contracts import AugmentationProfile
    from hydra_suite.training.runner import _apply_tiny_augmentation

    img = np.random.default_rng(1).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    prof = AugmentationProfile(enabled=True, decode_color_sim=1.0, resample_sim=1.0)
    a = _apply_tiny_augmentation(img, True, prof, rng=np.random.default_rng(5))
    b = _apply_tiny_augmentation(img, True, prof, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_training_augmentation.py -k "tiny_augmentation_decode or off_by_default_leaves or tiny_augmentation_deterministic" -v`
Expected: FAIL — `_apply_tiny_augmentation()` currently takes no `rng` kwarg / does not apply decode/resample.

- [ ] **Step 3: Add the injection to `_apply_tiny_augmentation`**

In `src/hydra_suite/training/runner.py`, change the signature and add the new augmentations just before `return img` (after the existing monochrome block at ~365):

```python
def _apply_tiny_augmentation(img, augment, profile, rng=None):
    """Apply optional canonical-pose-safe augmentations to an image."""
    import cv2
    import numpy as np

    if not (augment and profile and profile.enabled):
        return img
    # ... existing flips / brightness / contrast / saturation / hue / monochrome ...
```

and immediately before the final `return img`:

```python
    decode = float(getattr(profile, "decode_color_sim", 0.0) or 0.0)
    resample = float(getattr(profile, "resample_sim", 0.0) or 0.0)
    if decode > 0.0 or resample > 0.0:
        from hydra_suite.training.augmentation import (
            simulate_decode_color,
            simulate_resample,
        )

        if rng is None:
            rng = np.random.default_rng()
        if decode > 0.0:
            img = simulate_decode_color(img, decode, rng)
        if resample > 0.0:
            img = simulate_resample(img, resample, rng)
    return img
```

- [ ] **Step 4: Seed and thread the rng through `TinyDataset`**

In `src/hydra_suite/training/runner.py`, `TinyDataset.__init__` (~305): store a seeded generator. Add to the end of `__init__`:

```python
        import numpy as np
        self._rng = np.random.default_rng(getattr(spec, "seed", 42) if "spec" in dir() else 42)
```

NOTE: `TinyDataset.__init__` may not receive `spec`. Read the actual `__init__` signature; if it does not have `spec`/`seed`, add a `seed: int = 42` parameter to `__init__`, store `self._rng = np.random.default_rng(seed)`, and pass `seed=spec.seed` at the construction site (383-388): `TinyDataset(train_samples, augment=True, profile=spec.augmentation_profile, seed=spec.seed)`. The val loader (`augment=False`) does not need a seed (aug is skipped).

Then in `TinyDataset.__getitem__` (~315), change the augmentation call to pass the rng:

```python
        img = _apply_tiny_augmentation(img, self.augment, self.profile, rng=getattr(self, "_rng", None))
```

- [ ] **Step 5: Run the tiny tests + the existing tiny test to verify green (no regression)**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_training_augmentation.py -v`
Expected: all PASS, including the pre-existing `test_tiny_augmentation_monochrome_enforces_equal_channels` (which calls `_apply_tiny_augmentation(img, True, profile=...)` with 3 args — the new `rng=None` default keeps it working).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(training): apply decode/resample aug in the tiny classifier trainer"
git add -A && git commit -m "feat(training): apply decode/resample aug in the tiny classifier trainer"
git status
```

---

## Task 4: Inject into the two torchvision trainers

**Files:**
- Modify: `src/hydra_suite/training/runner.py` — `_train_custom_classify` (train transforms 1274-1301) and `_train_multihead_shared_classify` (train transforms 1602-1619)
- Test: `tests/test_augmentation_module.py` (the PIL-transform tests from Task 1 already cover the mechanism; add one asserting insertion position via a small helper)

**Interfaces:**
- Consumes: `make_decode_resample_pil_transform(profile, rng)` (Task 1); `profile.decode_color_sim`/`resample_sim` (Task 2).
- Produces: both train pipelines insert the PIL transform (when non-`None`) immediately BEFORE `transforms.ToTensor()`. Val pipelines unchanged (evaluate on the clean convention).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_augmentation_module.py`:

```python
def test_pil_transform_inserts_before_totensor_shape_ok():
    """The PIL transform must accept the post-Resize PIL image and return a PIL
    image ToTensor can consume."""
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    from hydra_suite.training.augmentation import make_decode_resample_pil_transform

    prof = type("P", (), {"decode_color_sim": 1.0, "resample_sim": 1.0})()
    tf = make_decode_resample_pil_transform(prof, np.random.default_rng(0))
    pipeline = transforms.Compose([transforms.Resize((16, 16)), tf, transforms.ToTensor()])
    out = pipeline(Image.fromarray(np.zeros((20, 20, 3), np.uint8)))
    assert tuple(out.shape) == (3, 16, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_augmentation_module.py::test_pil_transform_inserts_before_totensor_shape_ok -v`
Expected: PASS already if Task 1 is done (the helper exists). If it fails, fix `make_decode_resample_pil_transform`. (This test guards the torchvision integration contract; Steps 3-4 wire it into the trainers.)

- [ ] **Step 3: Wire into `_train_custom_classify`**

In `src/hydra_suite/training/runner.py`, in `_train_custom_classify`, after `train_transforms` is built and BEFORE the `train_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])` line (~1295), insert:

```python
    import numpy as np
    from hydra_suite.training.augmentation import make_decode_resample_pil_transform

    _decode_tf = make_decode_resample_pil_transform(
        profile, np.random.default_rng(getattr(spec, "seed", 42))
    )
    if _decode_tf is not None:
        train_transforms.append(_decode_tf)
    train_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
```

(The insert appends the PIL transform right before ToTensor/Normalize. Val pipeline `val_tf` is untouched.)

- [ ] **Step 4: Wire into `_train_multihead_shared_classify`**

In `src/hydra_suite/training/runner.py`, in `_train_multihead_shared_classify`, before the `train_tf_steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]` line (~1618), insert:

```python
    import numpy as np
    from hydra_suite.training.augmentation import make_decode_resample_pil_transform

    _decode_tf = make_decode_resample_pil_transform(
        profile, np.random.default_rng(getattr(spec, "seed", 42))
    )
    if _decode_tf is not None:
        train_tf_steps.append(_decode_tf)
    train_tf_steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
```

(Val `val_tf` untouched.)

- [ ] **Step 5: Run tests + a smoke import to verify no syntax/name errors**

Run:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps
PYTHONPATH=$PWD/src python -c "import hydra_suite.training.runner"   # import must succeed
PYTHONPATH=$PWD/src python -m pytest tests/test_augmentation_module.py -v
```
Expected: import OK; all PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(training): decode/resample aug in torchvision custom + multihead trainers"
git add -A && git commit -m "feat(training): decode/resample aug in torchvision custom + multihead trainers"
git status
```

---

## Task 5: GUI controls (ClassKit training dialog)

**Files:**
- Modify: `src/hydra_suite/classkit/gui/dialogs/training.py` — aug tab widgets (~1617, next to `hue_spin`), `get_settings()` dict (~2378-2385), summary text (~2044-2068)
- Modify: `src/hydra_suite/classkit/gui/main_window.py` — `AugmentationProfile(...)` build at 7222-7233
- Test: `tests/test_classkit_main_window.py` (extend `test_make_training_spec_maps_color_augmentations` region, ~1962)

**Interfaces:**
- Consumes: the two `AugmentationProfile` fields (Task 2).
- Produces: two `QDoubleSpinBox` controls (`self.decode_color_spin`, `self.resample_spin`) whose values flow through `get_settings()` (dict keys `decode_color_sim`, `resample_sim`) into `AugmentationProfile` at the `_make_training_spec` build site.

- [ ] **Step 1: Write the failing test**

In `tests/test_classkit_main_window.py`, add near the existing `test_make_training_spec_maps_color_augmentations` (extend it or add a sibling). Follow the existing test's setup (it stubs the dialog's `get_settings()` return). Add `decode_color_sim`/`resample_sim` to the stubbed settings and assert they land on the profile:

```python
def test_make_training_spec_maps_decode_resample_augmentations(qtbot, ...):
    # (mirror the setup of test_make_training_spec_maps_color_augmentations:
    #  build the main window / training spec with a settings dict that now includes
    #  decode_color_sim and resample_sim)
    settings = {..., "decode_color_sim": 0.5, "resample_sim": 0.3}
    # ... invoke the same _make_training_spec path the sibling test uses ...
    assert spec.augmentation_profile.decode_color_sim == 0.5
    assert spec.augmentation_profile.resample_sim == 0.3
```

Read `test_make_training_spec_maps_color_augmentations` (line ~1962) first and copy its exact fixture/stub wiring — reuse its harness rather than inventing a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_classkit_main_window.py::test_make_training_spec_maps_decode_resample_augmentations -v`
Expected: FAIL — the fields are not read/mapped yet (`AttributeError`/`0.0` mismatch).

- [ ] **Step 3: Add the two spinboxes to the aug tab**

In `src/hydra_suite/classkit/gui/dialogs/training.py`, `_build_space_and_augmentations_tab`, after the `self.hue_spin` row (~1617-1625) and before the monochrome checkbox, add (mirror the existing spinbox idiom exactly — `QDoubleSpinBox`, `setRange`, `setSingleStep`, `setDecimals`, `aug_form.addRow`):

```python
        self.decode_color_spin = QDoubleSpinBox()
        self.decode_color_spin.setRange(0.0, 1.0)
        self.decode_color_spin.setSingleStep(0.05)
        self.decode_color_spin.setDecimals(2)
        self.decode_color_spin.setValue(0.0)
        self.decode_color_spin.setToolTip(
            "Decode-color robustness: probability of re-decoding each crop through a "
            "random YCbCr matrix/range convention. Hardens colortag/identity models "
            "against GPU-Fast/NVDEC and cross-codec decode differences. 0 = off."
        )
        aug_form.addRow("Decode-color Robustness:", self.decode_color_spin)

        self.resample_spin = QDoubleSpinBox()
        self.resample_spin.setRange(0.0, 1.0)
        self.resample_spin.setSingleStep(0.05)
        self.resample_spin.setDecimals(2)
        self.resample_spin.setValue(0.0)
        self.resample_spin.setToolTip(
            "Resample robustness: probability of re-warping each crop through an "
            "alternate resampler (cv2 / torch grid_sample). Hardens against the "
            "training-vs-inference crop geometry gap. 0 = off."
        )
        aug_form.addRow("Resample Robustness:", self.resample_spin)
```

(Confirm `QDoubleSpinBox` is imported in this file — the existing spins use it, so it is.)

- [ ] **Step 4: Emit the two keys from `get_settings()`**

In `src/hydra_suite/classkit/gui/dialogs/training.py`, `get_settings()` (~2295), where the aug keys are emitted (~2378-2385, alongside `brightness`/`contrast`/`monochrome`), add:

```python
            "decode_color_sim": self.decode_color_spin.value(),
            "resample_sim": self.resample_spin.value(),
```

- [ ] **Step 5: Reflect in the augmentation summary text**

In `src/hydra_suite/classkit/gui/dialogs/training.py`, the summary builder (~2044-2068), where it appends `f"brightness {..:.2f}"` etc., add (mirroring the existing pattern):

```python
        if self.decode_color_spin.value() > 0:
            parts.append(f"decode-color {self.decode_color_spin.value():.2f}")
        if self.resample_spin.value() > 0:
            parts.append(f"resample {self.resample_spin.value():.2f}")
```

(Use the actual local list variable name from that block — read it; it may be `parts`/`items`/similar.)

- [ ] **Step 6: Map the fields into `AugmentationProfile` at the build site**

In `src/hydra_suite/classkit/gui/main_window.py`, `_make_training_spec` (~7222-7233), add to the `AugmentationProfile(...)` constructor call:

```python
        aug = AugmentationProfile(
            enabled=True,
            flipud=settings.get("flipud", 0.0),
            fliplr=settings.get("fliplr", 0.0),
            hue=settings.get("hue", 0.0),
            saturation=settings.get("saturation", 0.0),
            brightness=settings.get("brightness", 0.0),
            contrast=settings.get("contrast", 0.0),
            decode_color_sim=settings.get("decode_color_sim", 0.0),
            resample_sim=settings.get("resample_sim", 0.0),
            monochrome=bool(settings.get("monochrome", False)),
            args=aug_args,
            label_expansion=settings.get("label_expansion") or {},
        )
```

- [ ] **Step 7: Run the GUI test + the existing color-aug test (no regression)**

Run: `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps && PYTHONPATH=$PWD/src python -m pytest tests/test_classkit_main_window.py -k "make_training_spec_maps" -v`
Expected: both the new decode/resample test and the existing `test_make_training_spec_maps_color_augmentations` PASS.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(classkit): decode-color + resample robustness controls in training dialog"
git add -A && git commit -m "feat(classkit): decode-color + resample robustness controls in training dialog"
git status
```

---

## Task 6: Full-suite delta gate + empirical validation procedure (documented)

**Files:** none (validation only).

The unit tests above run without training. The spec's headline acceptance — a small colortag model trained WITH the augmentation reaching ≥99% cv2-vs-NVDEC crop label agreement (up from ~90%) with no clean-val regression — requires an actual training run + the crop-agreement harness on the CUDA box. That is a heavier empirical gate (analogous to Part 1's hardware step), run by the user, not an automated unit test.

- [ ] **Step 1: Full touched-suite sanity (this box)**

Run the directly-touched test files together to confirm no regression:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps
PYTHONPATH=$PWD/src python -m pytest tests/test_augmentation_module.py tests/test_training_augmentation.py tests/test_classkit_main_window.py -q
```
Expected: all green. (The base suite has ~pre-existing failures unrelated to this work — compare against a baseline run if any failure appears in an untouched area; only touched-area failures block.)

- [ ] **Step 2: Document the empirical validation procedure (for the user to run when retraining)**

Record in the PR description / memory the procedure to close the spec's ≥99% acceptance when convenient:
1. Train a small colortag/tiny classifier on the existing training set twice: once with `decode_color_sim=0.5, resample_sim=0.3`, once as an un-augmented control (both fields `0.0`), same seed.
2. Run the crop-level agreement harness (the `crop_isolate.sh`-style probe from the Part-1 investigation, in the session scratchpad) on held-out detections: run each model on the same detections cropped via cv2-decode+cv2-warp vs NVDEC-decode+grid_sample; report colortag label agreement.
3. Acceptance: augmented model ≥99% cv2-vs-NVDEC agreement (control ~90%); augmented model's clean cv2 val accuracy within noise of the control (no in-distribution regression).

This step is **not a code gate for landing** — it is the retraining validation to run on the CUDA box when a model is actually retrained. Landing = Tasks 1-5 + Step 1 green.

---

## Self-Review

**Spec coverage:**
- Spec §1 (shared module, 2 functions, RGB uint8, explicit rng) → **Task 1**.
- Spec §2 (`AugmentationProfile` fields, default off, serialize) → **Task 2**.
- Spec §3 (inject into 3 trainers; val excluded) → **Task 3** (tiny) + **Task 4** (2 torchvision).
- Spec §4 (GUI two controls + build site + summary) → **Task 5** (corrected: build site is `main_window.py:7222`, not `training.py`).
- Spec Acceptance (≥99% crop agreement, no clean-val regression, reproducible-under-seed) → **Task 6** (documented empirical procedure) + determinism unit tests in Tasks 1/3.
- Spec Testing (unit: shape/dtype, identity@0, bounded, deterministic, profile round-trip; integration: fast tiny train + harness) → Tasks 1-3 (unit) + Task 6 (integration procedure).

**Placeholder scan:** the only intentionally-open items are the GUI test's fixture wiring (Task 5 Step 1 — must copy the existing sibling test's harness, which the implementer reads) and `TinyDataset.__init__`'s actual `seed` plumbing (Task 3 Step 4 — depends on the real signature the implementer reads). Both are explicitly flagged to read-then-mirror, not guess. All augmentation logic is concrete code.

**Type consistency:** `simulate_decode_color(rgb_u8, strength, rng) -> np.ndarray`, `simulate_resample(rgb_u8, prob, rng) -> np.ndarray`, `make_decode_resample_pil_transform(profile, rng) -> callable|None`, `_apply_tiny_augmentation(img, augment, profile, rng=None)`, profile fields `decode_color_sim`/`resample_sim: float`. All consistent across tasks. Profile accessed as `spec.augmentation_profile` (verified), serialized via `asdict` (verified, no `from_dict`).

**Backward-compat:** every injection is gated on `> 0.0`; both fields default `0.0`; `_apply_tiny_augmentation`'s new `rng` is optional — the pre-existing 3-arg test caller and the existing flips/jitter behavior are unchanged when the fields are off.
