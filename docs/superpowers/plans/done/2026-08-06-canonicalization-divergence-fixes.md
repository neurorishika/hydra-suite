# Canonicalization Divergence Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire `cv2` from the animal-crop path in favour of one shared torch resampler used byte-identically by training-data generation and by inference on every device, and close the device/format/guard divergences the adversarial review found (F1–F7).

**Architecture:** A single module `core/canonicalization/resample.py` provides `canonical_warp*` (Layer 1, `F.grid_sample`) and `letterbox_fit` (Layer 2, antialiased `F.interpolate`). Every producer — inference stages, training transform, dataset export, interpolated crops — calls these, so there is exactly one implementation, parameterised only by the input tensor's device. ViTPose (and SLEAP-native) take an identity Layer-2 fit because they do their own model-internal resample. Models are retrained against the unified path with Moderate augmentation.

**Tech Stack:** Python, PyTorch (`torch.nn.functional.grid_sample` / `interpolate`), NumPy, OpenCV (retained only for polygon rasterisation and image I/O, not resampling), pytest.

## Global Constraints

- Retain the existing `FitResult` arithmetic (`fit_to_model_input`, `fit_affine`) — kernel-independent, do not change offsets/scale math.
- `letterbox_fit` uses `mode="bilinear", align_corners=False, antialias=True`; `grid_sample` uses `mode="bilinear", padding_mode="zeros", align_corners=True` (matches the verified-exact existing theta derivation).
- No `cv2.warpAffine` / `cv2.resize` may remain on the crop path after Phase 1 (grep gate in Task 3/final).
- `geometry` is a **required** argument on crop/stage helpers — no hardcoded `from_reference(20,2,1.3)` fallback survives (F7b).
- Crop-dataset export (model input) is lossless-only, default PNG (F5). Oriented-video export keeps jpg.
- Within-device determinism must hold (reproducible reruns). Cross-device parity is delivered by augmentation, not bit-equality.
- Do NOT append `Co-Authored-By: Claude` to commits (commit as the configured git user).
- Run tests per-file (`python -m pytest tests/test_x.py`), never the whole suite in one process (modal-dialog hangs + SIGABRT).
- Environment for any run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD/src`.
- CUDA-only numeric checks run on mehek (`hydra-cuda`); mark such tests to skip when `torch.cuda.is_available()` is False.

---

## File Structure

- **Create** `src/hydra_suite/core/canonicalization/resample.py` — the one torch resampler (Layer 1 + Layer 2).
- **Create** `src/hydra_suite/training/canonical_aug.py` — Moderate training augmentation.
- **Modify** `src/hydra_suite/core/canonicalization/fit.py` — `apply_fit` delegates to `letterbox_fit`.
- **Modify** `src/hydra_suite/core/canonicalization/crop.py` — warps delegate to the seam.
- **Modify** `src/hydra_suite/core/inference/stages/crops.py` — collapse cpu/gpu twins; drop `apply_fit_gpu`.
- **Modify** `src/hydra_suite/core/inference/stages/{cnn,headtail,pose}.py` — required `geometry`; ViTPose identity fit.
- **Modify** `src/hydra_suite/training/canonical_transform.py` — torch `letterbox_fit`.
- **Modify** `src/hydra_suite/core/individual/classification/backend.py` — `predict_batch(input_is_bgr=...)` (F2).
- **Modify** `src/hydra_suite/core/individual/pose/backends/vitpose.py` — `does_own_letterbox` (F3).
- **Modify** `src/hydra_suite/core/post/interpolated_crops.py` — skip on fit failure (F6).
- **Modify** `src/hydra_suite/core/inference/config.py` + `core/canonicalization/geometry.py` — single geometry derivation (F7a).
- **Modify** `src/hydra_suite/core/individual/dataset/{generator,oriented_video}.py` — shared ClippingStats + lossless export (F4/F5).
- **Modify** `src/hydra_suite/classkit/jobs/task_workers.py` — non-square eval transform (F7c).
- **Modify** `src/hydra_suite/resources/configs/default.json` (+ presets) — `canonical_margin`, `reference_body_size` (F7d).

---

## Phase 1 — Seam + safe fixes (no retrain; re-baseline harness after)

### Task 1: The torch resampler seam

**Files:**
- Create: `src/hydra_suite/core/canonicalization/resample.py`
- Test: `tests/test_canonical_resample.py`

**Interfaces:**
- Consumes: `CanonicalGeometry`, `canonical_affine` (`core/canonicalization/geometry.py`); `FitResult`, `fit_to_model_input` (`core/canonicalization/fit.py`).
- Produces:
  - `canonical_warp(frame_chw: torch.Tensor, m_align: np.ndarray, geometry: CanonicalGeometry) -> torch.Tensor` — `(C, canvas_h, canvas_w)` on the input device.
  - `canonical_warp_batch(frame_chw: torch.Tensor, m_aligns: list[np.ndarray], geometry: CanonicalGeometry) -> torch.Tensor` — `(N, C, canvas_h, canvas_w)`.
  - `letterbox_fit(crop_chw: torch.Tensor, model_wh: tuple[int,int]) -> torch.Tensor` — `(C, model_h, model_w)` (or `(N, C, ...)` for a batched input).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_canonical_resample.py
import numpy as np, torch, pytest
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, canonical_affine
from hydra_suite.core.canonicalization.resample import (
    canonical_warp, canonical_warp_batch, letterbox_fit,
)

GEOM = CanonicalGeometry.from_reference(reference_body_px=60, aspect_ratio=2.0, margin=1.3)

def _obb(cx, cy, L, W, deg):
    a = np.deg2rad(deg)
    dx = np.array([np.cos(a), np.sin(a)]); dy = np.array([-np.sin(a), np.cos(a)])
    c = np.array([cx, cy], float)
    return np.array([c-L/2*dx-W/2*dy, c+L/2*dx-W/2*dy, c+L/2*dx+W/2*dy, c-L/2*dx+W/2*dy])

def test_warp_is_geometrically_exact():
    frame = np.zeros((400, 400, 3), np.uint8); frame[210, 205] = 255
    m, _, _ = canonical_affine(_obb(200, 200, 80, 40, 0), GEOM)
    t = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
    crop = canonical_warp(t, m, GEOM).permute(1, 2, 0).numpy().sum(2)
    ys, xs = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]; s = crop.sum()
    got = (float((xs*crop).sum()/s), float((ys*crop).sum()/s))
    exp = m @ np.array([205, 210, 1.0])
    assert abs(got[0]-exp[0]) < 0.5 and abs(got[1]-exp[1]) < 0.5

def test_letterbox_is_isotropic_nonsquare():
    # a horizontal edge must not tilt under a non-square fit -> single scale
    crop = torch.zeros(3, 56, 112); crop[:, 28:, :] = 1.0
    out = letterbox_fit(crop, (64, 128))  # (W,H) -> tensor (C,128,64)
    assert out.shape == (3, 128, 64)
    col_means = out[0].mean(dim=0)          # variation across width
    assert float(col_means.std()) < 1e-3    # no horizontal gradient => no x/y anisotropy

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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_canonical_resample.py -v`
Expected: FAIL (`ModuleNotFoundError: resample`).

- [ ] **Step 3: Implement the seam**

Port the verified theta derivation from `crop.py:gpu_canonical_crop*` verbatim (it is geometrically exact) into `canonical_warp` / `canonical_warp_batch`. Implement `letterbox_fit` from the current `apply_fit`/`apply_fit_gpu` arithmetic using `fit_to_model_input`:

```python
# resample.py (letterbox_fit core)
import torch, torch.nn.functional as F
from hydra_suite.core.canonicalization.fit import fit_to_model_input

def letterbox_fit(crop_chw, model_wh):
    single = crop_chw.dim() == 3
    x = crop_chw.unsqueeze(0) if single else crop_chw
    n, c, sh, sw = x.shape
    fit = fit_to_model_input((sw, sh), model_wh)
    iw, ih = fit.inner_wh; mw, mh = fit.model_wh
    resized = F.interpolate(x, size=(ih, iw), mode="bilinear",
                            align_corners=False, antialias=True)
    if (ih, iw) == (mh, mw):
        out = resized
    else:
        out = x.new_zeros((n, c, mh, mw))
        ox, oy = fit.offset_xy
        out[:, :, oy:oy+ih, ox:ox+iw] = resized
    return out.squeeze(0) if single else out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_canonical_resample.py -v`
Expected: PASS (MPS test passes on this box; skipped elsewhere).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/canonicalization/resample.py tests/test_canonical_resample.py
git commit -m "feat(canonicalization): one torch resampler seam (grid_sample + antialiased letterbox)"
```

---

### Task 2: Route `fit.py` / `crop.py` onto the seam (retire cv2 there)

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/fit.py` (`apply_fit`)
- Modify: `src/hydra_suite/core/canonicalization/crop.py` (`extract_canonical_crop`, `gpu_canonical_crop`, `gpu_canonical_crop_batch`)
- Test: `tests/test_canonical_crop.py` (existing), `tests/test_canonical_fit.py` (existing)

**Interfaces:**
- Consumes: `canonical_warp*`, `letterbox_fit` (Task 1).
- Produces: unchanged public signatures; numpy variants convert once (`torch.from_numpy` → seam → `.numpy()`), preserving uint8/BGR/HWC I/O contracts.

- [ ] **Step 1: Add a numpy-parity test**

```python
# tests/test_canonical_fit.py (append)
import numpy as np
from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input
def test_apply_fit_nonsquare_shape_and_dtype():
    crop = (np.random.default_rng(0).integers(0,256,(56,112,3),np.uint8))
    fit = fit_to_model_input((112,56),(64,128))
    out = apply_fit(crop, fit)
    assert out.shape == (128,64,3) and out.dtype == np.uint8
```

- [ ] **Step 2: Run existing crop/fit suites to capture the pre-change baseline**

Run: `python -m pytest tests/test_canonical_fit.py tests/test_canonical_crop.py -v`
Expected: PASS (records current behaviour before rewiring).

- [ ] **Step 3: Rewire**

`apply_fit(image, fit)` → convert to CHW tensor, call `letterbox_fit(t, fit.model_wh)`, convert back to HWC uint8. `extract_canonical_crop` / `gpu_canonical_crop*` → delegate to `canonical_warp*` (numpy variant converts once). Keep the uint8 guard and the foreign-mask call sequence intact.

- [ ] **Step 4: Run suites**

Run: `python -m pytest tests/test_canonical_fit.py tests/test_canonical_crop.py -v`
Expected: PASS (geometry unchanged; kernel now torch — update any byte-exact asserts to `atol` tolerances where they compared against cv2 output, noting the change in the test docstring).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/canonicalization/fit.py src/hydra_suite/core/canonicalization/crop.py tests/test_canonical_fit.py
git commit -m "refactor(canonicalization): fit.py/crop.py delegate to the torch seam"
```

---

### Task 3: Collapse the cpu/gpu twins in `stages/crops.py`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py`
- Test: `tests/test_inference_canonical_contract.py` (existing), `tests/test_gpu_classifier_crop.py` (rewrite in Task 14)

**Interfaces:**
- Consumes: `canonical_warp*`, `letterbox_fit` (Task 1).
- Produces: `extract_canonical_crops`, `extract_canonical_crops_batch`, `extract_classifier_crops*` — same signatures, single device-agnostic body; `apply_fit_gpu`, `_extract_canonical_cpu`, `_extract_canonical_gpu`, `extract_classifier_crops_gpu`, `extract_classifier_crops_batch_gpu` removed (callers use the unified functions).

- [ ] **Step 1: Write a CPU/GPU-parity test (device-agnostic seam)**

```python
# tests/test_inference_canonical_contract.py (append)
import numpy as np, torch
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.stages.crops import extract_canonical_crops
# build a minimal OBBResult + RuntimeContext per the existing helpers in this file
def test_crops_nonsquare_canvas_uniform(make_obb, cpu_runtime):
    geom = CanonicalGeometry.from_reference(60, 2.0, 1.3)  # non-square canvas
    frame = np.random.default_rng(0).integers(0,256,(300,300,3),np.uint8)
    crops = extract_canonical_crops(frame, make_obb(3), geom, cpu_runtime)
    assert crops.shape == (3, 3, geom.canvas_h, geom.canvas_w)
    assert geom.canvas_h != geom.canvas_w
```

- [ ] **Step 2: Run to verify (uses existing fixtures)**

Run: `python -m pytest tests/test_inference_canonical_contract.py -v`
Expected: PASS pre-refactor (baseline), then re-run after Step 3.

- [ ] **Step 3: Unify**

Replace the `if runtime.tensor_on_cuda: _extract_canonical_gpu else _extract_canonical_cpu` fork with a single path: normalise `frame` to a CHW float tensor on its own device (numpy → CPU tensor), call `canonical_warp_batch`. Delete `apply_fit_gpu` and the `_gpu`/`_cpu`/`_batch_gpu` twins; classifier stages call the unified `extract_classifier_crops*` then `letterbox_fit`.

- [ ] **Step 4: Grep gate + tests**

Run:
```bash
! grep -nE "cv2\.(warpAffine|resize)" src/hydra_suite/core/inference/stages/crops.py src/hydra_suite/core/canonicalization/crop.py src/hydra_suite/core/canonicalization/fit.py
python -m pytest tests/test_inference_canonical_contract.py tests/test_classifier_crop_batch_np_identity.py -v
```
Expected: grep prints nothing (exit 0 via `!`); tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/crops.py tests/test_inference_canonical_contract.py
git commit -m "refactor(inference): collapse cpu/gpu crop twins onto the torch seam"
```

---

### Task 4: Training transform on the torch seam (train==infer clean-path identity)

**Files:**
- Modify: `src/hydra_suite/training/canonical_transform.py`
- Test: `tests/test_train_inference_fit_identity.py` (existing; strengthen)

**Interfaces:**
- Consumes: `letterbox_fit` (Task 1).
- Produces: `CanonicalFitTransform.__call__(image) -> np.ndarray` (unchanged signature; kernel now torch).

- [ ] **Step 1: Add a non-square clean-path identity test**

```python
# tests/test_train_inference_fit_identity.py (append)
import numpy as np
from hydra_suite.training.canonical_transform import CanonicalFitTransform
from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input
def test_transform_equals_inference_letterbox_nonsquare():
    crop = np.random.default_rng(1).integers(0,256,(56,112,3),np.uint8)
    t = CanonicalFitTransform((128, 64))            # (H, W)
    fit = fit_to_model_input((112, 56), (64, 128))  # inference side (W,H)
    np.testing.assert_array_equal(t(crop), apply_fit(crop, fit))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_train_inference_fit_identity.py -v`
Expected: FAIL (transform still cv2, inference now torch — outputs differ).

- [ ] **Step 3: Rewire the transform**

`CanonicalFitTransform.__call__` builds a CHW tensor, calls `letterbox_fit(t, (W, H))`, returns HWC uint8 — the exact same call `apply_fit` now makes.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_train_inference_fit_identity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/canonical_transform.py tests/test_train_inference_fit_identity.py
git commit -m "feat(training): CanonicalFitTransform uses the torch seam (train==infer clean path)"
```

---

### Task 5: Make `geometry` required (F7b — delete fallback geometries)

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/{cnn,headtail,pose}.py`, `src/hydra_suite/core/individual/classification/headtail.py`
- Test: `tests/test_canonical_geometry_required.py` (new)

**Interfaces:**
- Produces: `run_cnn(..., geometry: CanonicalGeometry)`, `run_headtail(..., geometry: CanonicalGeometry)` (no default); `_DEFAULT_CANONICAL_GEOMETRY` removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_geometry_required.py
import inspect, pytest
from hydra_suite.core.inference.stages import cnn, headtail, pose
@pytest.mark.parametrize("fn", [cnn.run_cnn, headtail.run_headtail])
def test_geometry_has_no_default(fn):
    assert inspect.signature(fn).parameters["geometry"].default is inspect.Parameter.empty
def test_no_module_default_geometry():
    assert not hasattr(cnn, "_DEFAULT_CANONICAL_GEOMETRY")
    assert not hasattr(headtail, "_DEFAULT_CANONICAL_GEOMETRY")
    assert not hasattr(pose, "_DEFAULT_CANONICAL_GEOMETRY")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_geometry_required.py -v`
Expected: FAIL.

- [ ] **Step 3: Remove the defaults**

Delete `_DEFAULT_CANONICAL_GEOMETRY` and make `geometry` a required parameter in the three stages and in `HeadTailAnalyzer.__init__` (drop the `reference_aspect_ratio=2.0`/`canonical_margin=1.3` self-built geometry; require the caller's geometry). Confirm live callers already pass it (`pipeline.py`, `runner.py`, `interpolated_crops.py` — verified in review).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_canonical_geometry_required.py tests/test_inference_canonical_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/cnn.py src/hydra_suite/core/inference/stages/headtail.py src/hydra_suite/core/inference/stages/pose.py src/hydra_suite/core/individual/classification/headtail.py tests/test_canonical_geometry_required.py
git commit -m "fix(inference): require geometry on crop stages; delete fallback geometries (F7b)"
```

---

### Task 6: Single geometry derivation (F7a)

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (~660–690), `src/hydra_suite/core/canonicalization/geometry.py`
- Test: `tests/test_canonical_geometry_single_derivation.py` (new)

**Interfaces:**
- Consumes/Produces: `canonical_geometry_from_params(params) -> CanonicalGeometry` becomes the sole derivation; `config.py` calls it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_geometry_single_derivation.py
from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params
from hydra_suite.core.inference import config as cfg
def test_config_derivation_matches_helper():
    params = {"REFERENCE_BODY_SIZE": 33.0, "RESIZE_FACTOR": 1.5,
              "ADVANCED_CONFIG": {"reference_aspect_ratio": 2.4, "canonical_margin": 1.5}}
    g_helper = canonical_geometry_from_params(params)
    g_config = cfg._canonical_geometry_from_params(params)  # whatever config.py exposes
    assert g_helper.to_dict() == g_config.to_dict()
```

- [ ] **Step 2: Run to verify (may fail on drift or on missing symbol)**

Run: `python -m pytest tests/test_canonical_geometry_single_derivation.py -v`
Expected: FAIL until config.py routes through the helper.

- [ ] **Step 3: Collapse**

Replace the re-inlined `CanonicalGeometry.from_reference(...)` at `config.py:679` and the dataclass fallback `_default_canonical_geometry()` (`config.py:403`) with calls to `canonical_geometry_from_params` (the fallback uses an empty-params dict → documented defaults). Keep one copy of the magic defaults, inside `canonical_geometry_from_params`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_canonical_geometry_single_derivation.py tests/test_canonical_margin_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py src/hydra_suite/core/canonicalization/geometry.py tests/test_canonical_geometry_single_derivation.py
git commit -m "refactor(canonicalization): single params->geometry derivation (F7a)"
```

---

### Task 7: Fix the CUDA classifier fallback crash (F2)

**Files:**
- Modify: `src/hydra_suite/core/individual/classification/backend.py` (`predict_batch`, ~1338)
- Test: `tests/test_classifier_predict_batch_bgr.py` (new)

**Interfaces:**
- Produces: `predict_batch(self, crops: list[np.ndarray], input_is_bgr: bool = True) -> list[list[np.ndarray]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_predict_batch_bgr.py
import inspect
from hydra_suite.core.individual.classification import backend
def test_predict_batch_accepts_input_is_bgr():
    sig = inspect.signature(backend.ClassifierBackend.predict_batch)  # actual class name
    assert "input_is_bgr" in sig.parameters
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_classifier_predict_batch_bgr.py -v`
Expected: FAIL (`assert 'input_is_bgr' in ...`).

- [ ] **Step 3: Implement**

Add `input_is_bgr: bool = True` to `predict_batch`; in `_preprocess`, flip BGR→RGB only when `input_is_bgr` (mirror `_preprocess_cuda`'s conditional at backend.py:1163). The two CUDA fallback call sites (backend.py:1264/1284) now pass a valid kwarg.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_classifier_predict_batch_bgr.py -v`
Expected: PASS. (A functional both-branches test is added in Task 14 where a stub factor backend without a CUDA forward is available.)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/classification/backend.py tests/test_classifier_predict_batch_bgr.py
git commit -m "fix(classification): predict_batch accepts input_is_bgr; unblock CUDA fallback (F2)"
```

---

### Task 8: Interpolated-crops fit-failure skips (F6)

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py` (~564–603)
- Test: `tests/test_interpolated_crops_worker.py` (existing; append)

**Interfaces:**
- Produces: `_flush_pose_batch` drops a crop whose `apply_fit`/`letterbox_fit` raises (append `None`, skip back-projection) instead of feeding a raw canvas.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpolated_crops_worker.py (append)
def test_pose_fit_failure_skips_detection(monkeypatch):
    import hydra_suite.core.post.interpolated_crops as ic
    monkeypatch.setattr(ic, "apply_fit", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    # drive _flush_pose_batch with one canonical crop; assert result is None and
    # pose_backend.predict_batch was NOT called with a raw canvas crop
    ...
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_interpolated_crops_worker.py -k fit_failure -v`
Expected: FAIL (current code appends the raw crop and calls predict).

- [ ] **Step 3: Implement**

In the `except` around the fit call, append `None`, `continue` (do not append raw crop, do not compose `fit_m`), matching the degenerate-OBB skip at `interpolated_crops.py:1090-1098`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_interpolated_crops_worker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py tests/test_interpolated_crops_worker.py
git commit -m "fix(post): interpolated pose skips on fit failure instead of raw-crop fallback (F6)"
```

---

### Task 9: ClassKit non-square eval + config keys (F7c/d)

**Files:**
- Modify: `src/hydra_suite/classkit/jobs/task_workers.py` (~1591–1594)
- Modify: `src/hydra_suite/resources/configs/default.json` (+ `ooceraea_biroi.json`)
- Test: `tests/test_classkit_eval_transform_nonsquare.py` (new), `tests/test_default_config_canonical_keys.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_default_config_canonical_keys.py
import json
from hydra_suite.paths import get_presets_dir  # or the resources loader used elsewhere
def test_default_has_canonical_margin():
    d = json.loads((get_presets_dir()/"default.json").read_text())  # adjust to real loader
    assert "canonical_margin" in d and "reference_body_size" in d
```

```python
# tests/test_classkit_eval_transform_nonsquare.py
# assert _build_transform for a classifier whose input_size is (H!=W)
# constructs CanonicalFitTransform((input_h, input_w)), not (sz, sz).
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_default_config_canonical_keys.py tests/test_classkit_eval_transform_nonsquare.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`_build_transform` uses `CanonicalFitTransform((input_h, input_w))`. Add `"canonical_margin": 1.3` and `"reference_body_size": 20.0` to `default.json` and `ooceraea_biroi.json` next to `reference_aspect_ratio`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_default_config_canonical_keys.py tests/test_classkit_eval_transform_nonsquare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/classkit/jobs/task_workers.py src/hydra_suite/resources/configs/default.json src/hydra_suite/resources/configs/ooceraea_biroi.json tests/test_classkit_eval_transform_nonsquare.py tests/test_default_config_canonical_keys.py
git commit -m "fix(classkit/config): non-square eval transform; add canonical_margin/reference_body_size (F7c/d)"
```

- [ ] **Step 6: Phase-1 re-baseline (manual gate, both platforms)**

Run the equivalence matrix (CLAUDE.md fast path) on MPS and CUDA. Expected: `fly_obb`, `worm_bgsub` byte-identical; DETERMINISM exact; crop clips changed-by-design (record the re-baseline in `docs/superpowers/specs/notes/`). PERFORMANCE within `PERF_TOLERANCE`; if CPU-tier crop warp regresses, confirm the batched `canonical_warp_batch` path is used.

---

## Phase 2 — ViTPose identity fit (F3)

### Task 10: ViTPose takes the identity Layer-2 fit on all devices

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/pose.py` (`model_input_wh`, `run_pose`, `run_pose_batch`)
- Modify: `src/hydra_suite/core/individual/pose/backends/vitpose.py`
- Test: `tests/test_pose_model_input_wh_non_square.py` (rewrite), `tests/test_vitpose_backend_geometry.py` (rewrite), `tests/test_vitpose_identity_fit.py` (new)

**Interfaces:**
- Produces: backend property `does_own_letterbox: bool` (default `False`); `model_input_wh` returns `geometry.canvas_wh` (identity fit) when the backend `does_own_letterbox`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vitpose_identity_fit.py
from hydra_suite.core.inference.stages.pose import model_input_wh
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
class _Backend:  # stand-in for ViTPose
    does_own_letterbox = True
    preferred_input_wh = (192, 256)
    preferred_input_size = 256
class _Model: backend = _Backend()
def test_vitpose_gets_identity_fit():
    g = CanonicalGeometry.from_reference(60, 2.0, 1.3)
    assert model_input_wh(_Model(), g) == g.canvas_wh   # not (192,256)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_vitpose_identity_fit.py -v`
Expected: FAIL (returns (192,256)).

- [ ] **Step 3: Implement**

In `model_input_wh`, check `getattr(model.backend, "does_own_letterbox", False)` first → return `geometry.canvas_wh`. Set `does_own_letterbox = True` on `ViTPoseBackend`. Because the fit becomes canvas→canvas (scale 1), the `apply_fit` in `run_pose`/`run_pose_batch` is a no-op; keep the code uniform (both branches feed the canvas), and drop the on_cuda/non_cuda asymmetry so back-projection composes `fit_m` (identity) consistently. Rewrite the two pinning tests to assert the identity-fit contract.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_vitpose_identity_fit.py tests/test_pose_model_input_wh_non_square.py tests/test_vitpose_backend_geometry.py tests/test_vitpose_train_infer_box_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/pose.py src/hydra_suite/core/individual/pose/backends/vitpose.py tests/test_vitpose_identity_fit.py tests/test_pose_model_input_wh_non_square.py tests/test_vitpose_backend_geometry.py
git commit -m "fix(pose): ViTPose takes identity Layer-2 fit on all devices (F3)"
```

- [ ] **Step 6: Re-gate** the pose clips on MPS + CUDA (equivalence matrix `ant_pose_headtail`); MPS ViTPose now matches CUDA and training. Record.

---

## Phase 3 — Exports + augmentation

### Task 11: Export clip guard surfacing (F4)

**Files:**
- Modify: `src/hydra_suite/core/individual/dataset/generator.py`, `oriented_video.py`
- Test: `tests/test_export_clipping_surfaced.py` (new)

**Interfaces:**
- Consumes: `ClippingStats` (`core/canonicalization/geometry.py`).
- Produces: both exporters hold a `ClippingStats`, call `.record` per detection, and `.summary()` → `logger.warning` (+ export-summary line) at finalize.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_clipping_surfaced.py
import logging
from hydra_suite.core.individual.dataset.generator import IndividualDatasetGenerator
def test_overflowing_obb_warns(caplog, tmp_path):
    # configure a geometry whose canvas is smaller than the OBB so overflow_ratio>1,
    # export one detection, assert a "CLIPPED" warning is emitted at finalize()
    with caplog.at_level(logging.WARNING):
        ...  # drive export
    assert any("CLIPPED" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_export_clipping_surfaced.py -v`
Expected: FAIL (counters only written to JSON).

- [ ] **Step 3: Implement**

Replace the ad-hoc `self._clipped_count`/`self._worst_overflow_ratio` bookkeeping with a `ClippingStats` instance; call `record` where corners are canonicalised; in `finalize()`/`_write_canonical_metadata`, emit `summary()` via `logger.warning` and include it in the returned/written summary. Keep the JSON keys for provenance.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_export_clipping_surfaced.py tests/test_canonical_dataset_provenance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/dataset/generator.py src/hydra_suite/core/individual/dataset/oriented_video.py tests/test_export_clipping_surfaced.py
git commit -m "fix(dataset): surface canonical-crop clipping on export via ClippingStats (F4)"
```

---

### Task 12: Lossless-only crop-dataset export (F5)

**Files:**
- Modify: `src/hydra_suite/core/individual/dataset/generator.py` (output-format handling ~137, 224–229)
- Test: `tests/test_crop_export_lossless.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crop_export_lossless.py
import pytest
from hydra_suite.core.individual.dataset.generator import IndividualDatasetGenerator
def test_crop_export_rejects_jpg():
    with pytest.raises((ValueError, AssertionError)):
        IndividualDatasetGenerator(..., image_format="jpg")  # crop dataset is model input
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_crop_export_lossless.py -v`
Expected: FAIL (jpg currently accepted).

- [ ] **Step 3: Implement**

Constrain crop-dataset export to lossless (default `png`); reject `jpg`/`jpeg` with a clear error pointing at the train/infer-consistency requirement. Leave `oriented_video.py` jpg support untouched.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_crop_export_lossless.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/dataset/generator.py tests/test_crop_export_lossless.py
git commit -m "fix(dataset): crop-dataset export is lossless-only (F5)"
```

---

### Task 13: Moderate robustness augmentation

**Files:**
- Create: `src/hydra_suite/training/canonical_aug.py`
- Modify: `src/hydra_suite/training/runner.py` (inject `CanonicalAug` when aug enabled)
- Test: `tests/test_canonical_aug.py` (new)

**Interfaces:**
- Produces: `CanonicalAug(seed: int | None = None, p_degrade: float = 0.3)` — callable `np.ndarray -> np.ndarray`, applied to the canonical crop before `letterbox_fit`. Kernel ∼ `U{torch-bilinear, torch-antialias, cv2-AREA, cv2-LINEAR}`; warp jitter `dx,dy∼U(-0.5,0.5)px`, `dθ∼U(-1,1)°`; `p=0.3` gaussian-blur / JPEG `q∈[85,100]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonical_aug.py
import numpy as np
from hydra_suite.training.canonical_aug import CanonicalAug
def test_aug_is_deterministic_with_seed_and_shape_preserving():
    crop = np.random.default_rng(0).integers(0,256,(56,112,3),np.uint8)
    a1 = CanonicalAug(seed=7)(crop.copy()); a2 = CanonicalAug(seed=7)(crop.copy())
    np.testing.assert_array_equal(a1, a2)
    assert a1.shape == crop.shape and a1.dtype == np.uint8
def test_aug_changes_pixels():
    crop = np.random.default_rng(0).integers(0,256,(56,112,3),np.uint8)
    assert not np.array_equal(CanonicalAug(seed=1)(crop.copy()), crop)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_canonical_aug.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Write `CanonicalAug` with a seeded `np.random.Generator` (no global RNG — respects the sandbox ban on `Math.random`-style nondeterminism and keeps training reproducible). Wire it into the classifier/pose trainers behind the existing augmentation config flag.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_canonical_aug.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/canonical_aug.py src/hydra_suite/training/runner.py tests/test_canonical_aug.py
git commit -m "feat(training): Moderate canonical-crop augmentation for compute-variation robustness"
```

---

### Task 14: Close the test blind spots (non-square GPU parity, CUDA train==infer, F2 functional)

**Files:**
- Rewrite: `tests/test_gpu_classifier_crop.py`
- Create: `tests/test_cuda_train_infer_tensor.py`, extend `tests/test_classifier_predict_batch_bgr.py`

- [ ] **Step 1: Rewrite the GPU pixel-parity test — non-square + tight**

```python
# tests/test_gpu_classifier_crop.py (key change)
_GEOM = CanonicalGeometry.from_reference(60, 2.0, 1.3)  # NON-square canvas
# assert CPU-torch vs device-torch crop agree to atol ~2e-2 AND a landmark
# (bright dot) lands within 0.5px of the analytic canonical location on both.
```

- [ ] **Step 2: CUDA train==infer tensor test (skipped off-CUDA)**

```python
# tests/test_cuda_train_infer_tensor.py
import torch, pytest
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA box only")
def test_cuda_infer_equals_training_transform():
    # same crop through letterbox_fit on cuda vs CanonicalFitTransform on cpu
    # assert allclose within float tolerance (un-augmented clean path)
    ...
```

- [ ] **Step 3: F2 functional both-branches**

```python
# tests/test_classifier_predict_batch_bgr.py (append)
def test_cuda_fallback_runs_without_typeerror(monkeypatch):
    # stub a factor backend lacking a CUDA forward; drive predict_batch_cuda;
    # assert it returns results (no TypeError) and RGB order is honored.
    ...
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/test_gpu_classifier_crop.py tests/test_classifier_predict_batch_bgr.py -v` (locally); `tests/test_cuda_train_infer_tensor.py` on mehek.
Expected: PASS locally (CUDA test skipped); PASS on mehek.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gpu_classifier_crop.py tests/test_cuda_train_infer_tensor.py tests/test_classifier_predict_batch_bgr.py
git commit -m "test(canonicalization): non-square GPU parity, CUDA train==infer, F2 fallback"
```

---

## Phase 4 — Retrain (operational, not TDD)

Not code tasks; run after Phases 1–3 merge and re-gate green on both platforms.

1. Regenerate every crop dataset (ClassKit identity, head-tail, PoseKit) — do **not** reuse the old ones; they were cv2-baked and possibly jpg-lossy.
2. Retrain CNN identity, head-tail, ViTPose, SLEAP, YOLO-pose/classify with `CanonicalAug` enabled.
3. Publish retrained models; verify `canonical_meta.py` geometry stamps.
4. Evaluate on held-out data: no accuracy regression vs the pre-refactor baseline on MPS and CUDA.

---

## Self-Review Notes

- **Spec coverage:** F1→Tasks 1–4; F2→Tasks 7,14; F3→Task 10; F4→Task 11; F5→Task 12; F6→Task 8; F7a→Task 6; F7b→Task 5; F7c/d→Task 9; augmentation→Task 13; test blind spots→Task 14; retrain→Phase 4. Equivalence re-baseline→Tasks 9,10 gates.
- **Placeholders:** the few `...` bodies are in tests that must be fleshed against existing in-file fixtures (`make_obb`/`cpu_runtime` helpers already present in the referenced test modules); each states its exact assertion. Implementers fill the driver using the neighbouring tests as templates — no logic is left unspecified.
- **Type consistency:** `canonical_warp`/`canonical_warp_batch`/`letterbox_fit` signatures are used identically in Tasks 2–4,10; `predict_batch(..., input_is_bgr=True)` in Tasks 7,14; `does_own_letterbox` in Task 10.
