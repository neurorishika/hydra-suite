# GPU-Native Classifier Crop Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the CUDA/NVDEC path, keep classifier (head/tail + CNN identity) crops and forward on the GPU end-to-end — NVDEC frame → `grid_sample` crop at the model input size → `predict_batch_cuda` — eliminating the per-frame frame device→host copy and the per-crop CPU `cv2.warpAffine` that dominate the dense-colony 1.49× CUDA regression.

**Architecture:** Add a GPU classifier-crop extractor (parallel to the existing pose/OBB `_extract_canonical_gpu`), a GPU factor-bundle forward (`_forward_multi_cuda`) so `predict_batch_cuda` stops falling back to numpy for multihead bundles, and stage routing in `run_headtail_batch`/`run_cnn_batch` that takes the GPU path when `runtime.tensor_on_cuda`. A strict load-time capability check makes an unsupported gpu-tier classifier fail loud. The CPU/MPS path is untouched.

**Tech Stack:** Python, PyTorch (`torch.nn.functional.grid_sample`/`affine_grid`), NumPy, pytest. Conda env `hydra-mps` (local) / `hydra-cuda` (mehek). `PYTHONPATH=src`, `KMP_DUPLICATE_LIB_OK=TRUE`.

## Global Constraints

- **Platform gate:** all new GPU behavior is conditional on `runtime.tensor_on_cuda`. When `False` (MPS / CPU / `HYDRA_DISABLE_NVDEC` / onnx-cpu), the existing `cv2` CPU path runs **unchanged**. MPS output must stay byte-identical to pre-change.
- **Strict on gpu tier:** with `tensor_on_cuda` True, a classifier (and every factor of a bundle) that lacks a CUDA-native forward must raise at **load** time — never a silent CPU fallback, never a mid-batch error.
- **Acceptance vs current CUDA (not legacy):** determinism (`new_a == new_b` byte-identical); positions within determinism floor; identity-label agreement ≥ 99% vs the current CUDA pipeline on `ant_cnn_identity`.
- **Base suite is NOT green** (~24 pre-existing failures). Use a **delta gate** ("no new failures"), not "all green".
- **Worktree testing:** run tests with `PYTHONPATH=src` from `.worktree/gpu-classifier` (editable install points elsewhere). `make format` is broken — run `black`/`isort` directly.
- **Reference files:** design spec `docs/superpowers/specs/2026-07-22-gpu-native-classifier-crop-design.md`. Existing GPU crop machinery: `src/hydra_suite/core/canonicalization/crop.py:gpu_canonical_crop_batch`, `src/hydra_suite/core/inference/stages/crops.py:_extract_canonical_gpu`. Classifier backend: `src/hydra_suite/core/identity/classification/backend.py`.

---

## File Structure

- `src/hydra_suite/core/inference/stages/crops.py` — add `extract_classifier_crops_gpu` (single-frame) + `extract_classifier_crops_batch_gpu` (window). Mirrors the CPU `extract_classifier_crops[_batch]` but stays on device.
- `src/hydra_suite/core/identity/classification/backend.py` — add `_forward_multi_cuda`; route factor bundles in `predict_batch_cuda`; add `supports_cuda_forward()` capability probe.
- `src/hydra_suite/core/inference/stages/cnn.py` — GPU routing in `run_cnn_batch` + strict check in `load_cnn_model`.
- `src/hydra_suite/core/inference/stages/headtail.py` — GPU routing in `run_headtail_batch` + strict check in `load_headtail_model`.
- `tests/test_gpu_classifier_crop.py` — new unit tests (crop shape/device, CPU/GPU numeric closeness on CPU-simulated grid_sample, factor-forward shape, strict check).

---

## Task 1: GPU classifier crop extractor

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py` (add two functions near `_extract_canonical_gpu`)
- Test: `tests/test_gpu_classifier_crop.py` (create)

**Interfaces:**
- Consumes: `gpu_canonical_crop_batch(frame_chw, M_aligns, canvas_w, canvas_h) -> (N,C,canvas_h,canvas_w)` from `core/canonicalization/crop.py`; `compute_native_crop_dimensions`, `compute_alignment_affine` from the same module; `OBBResult` with `.num_detections`, `.corners` (N,4,2), `.frame_idx`, `.detection_ids`.
- Produces:
  - `extract_classifier_crops_gpu(frame, obb_result, target_size, aspect_ratio, margin, device) -> torch.Tensor` shape `(N, C, out_h, out_w)` float32 on `device`, BGR channel order, values in `[0,1]` (matching `extract_classifier_crops_batch`'s CPU tensor convention, `crops.py:307`).
  - `extract_classifier_crops_batch_gpu(frames, obb_results, target_size, aspect_ratio, margin, device) -> CropBatch` with `.crops` a `(ΣN, C, out_h, out_w)` CUDA tensor; same `detection_ids`/`frame_index`/`obb_by_frame`/`native_sizes` fields as the CPU `extract_classifier_crops_batch`.
- `target_size` is `(out_w, out_h)` per legacy convention (index 0 = width), same as the CPU function.

- [ ] **Step 1: Write the failing test (crop shape/device/dtype)**

```python
# tests/test_gpu_classifier_crop.py
import numpy as np
import pytest
import torch

from hydra_suite.core.inference.result import OBBResult


def _toy_obb(n=3, frame_idx=0):
    # axis-aligned-ish boxes with valid 4x2 corners
    corners = np.zeros((n, 4, 2), np.float32)
    for i in range(n):
        x0, y0 = 10 + 40 * i, 12
        corners[i] = [[x0, y0], [x0 + 30, y0], [x0 + 30, y0 + 15], [x0, y0 + 15]]
    return OBBResult(
        corners=corners,
        detection_ids=np.arange(n, dtype=np.int64) + frame_idx * 10000,
        frame_idx=frame_idx,
    )


def test_gpu_classifier_crop_shape_device():
    from hydra_suite.core.inference.stages.crops import extract_classifier_crops_gpu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frame = torch.randint(0, 256, (3, 200, 300), dtype=torch.uint8).float().div(255).to(dev)
    obb = _toy_obb(3)
    crops = extract_classifier_crops_gpu(frame, obb, (128, 128), 2.0, 1.3, dev)
    assert crops.shape == (3, 3, 128, 128)
    assert str(crops.device).startswith(dev)
    assert crops.dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .worktree/gpu-classifier && PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_gpu_classifier_crop_shape_device -v`
Expected: FAIL with `ImportError: cannot import name 'extract_classifier_crops_gpu'`
(Adjust `OBBResult(...)` kwargs in the test if construction differs — read `core/inference/result.py:OBBResult` first and match its actual required fields; keep the assertion identical.)

- [ ] **Step 3: Implement `extract_classifier_crops_gpu`**

Add to `crops.py`. It mirrors `_extract_canonical_gpu` but the canvas is the **classifier input size** (fixed) rather than `max(native_dims)`, and it scales the alignment affine so the OBB fills that fixed canvas — matching what the CPU `extract_classifier_crops` does (which warps directly to `target_size`). Read `_extract_canonical_cpu` / `_warp_canonical_crop` to confirm the exact affine the CPU path builds (native crop dims → resized to `target_size`), and compose the equivalent `M_align` that maps the OBB corners straight to the `(out_w, out_h)` canvas so `gpu_canonical_crop_batch` reproduces it.

```python
def extract_classifier_crops_gpu(
    frame: "torch.Tensor | np.ndarray",
    obb_result: OBBResult,
    target_size: tuple[int, int],
    aspect_ratio: float,
    margin: float,
    device: str,
) -> "torch.Tensor":
    """GPU-native classifier crop: warp the on-device frame directly to the
    classifier input size via a single batched grid_sample. Returns
    ``(N, C, out_h, out_w)`` float32 on ``device`` (BGR, [0,1]) — the on-GPU
    analogue of ``extract_classifier_crops`` (cv2), used when the frame is a
    CUDA tensor (NVDEC path)."""
    import torch

    out_w, out_h = int(target_size[0]), int(target_size[1])
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).to(device)
    if frame.ndim == 4:
        frame = frame.squeeze(0)
    n = obb_result.num_detections
    if n == 0:
        return torch.zeros((0, 3, out_h, out_w), dtype=torch.float32, device=device)

    padding_fraction = max(0.0, float(margin) - 1.0)
    M_aligns: list[np.ndarray] = []
    for i in range(n):
        try:
            cw, ch = compute_native_crop_dimensions(
                obb_result.corners[i], aspect_ratio, padding_fraction
            )
            M, _ = compute_alignment_affine(
                obb_result.corners[i], cw, ch, padding_fraction
            )
            # Scale the native-canvas affine to the fixed classifier canvas so the
            # OBB fills (out_w, out_h) exactly (cv2 warps native then resizes; the
            # composed affine warps straight to target — same mapping).
            sx, sy = out_w / float(cw), out_h / float(ch)
            S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]], dtype=np.float64)
            M = S @ np.vstack([M, [0.0, 0.0, 1.0]])
        except ValueError:
            M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        M_aligns.append(M)

    return gpu_canonical_crop_batch(frame, M_aligns, out_w, out_h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_gpu_classifier_crop_shape_device -v`
Expected: PASS (runs on CPU device where grid_sample also works — CUDA not required for the shape test).

- [ ] **Step 5: Add the batch wrapper + numeric-closeness test**

```python
def test_gpu_vs_cpu_classifier_crop_close():
    # grid_sample != cv2 exactly, but on a synthetic frame the crops must be
    # close (mean abs diff small); this guards gross affine mistakes.
    from hydra_suite.core.inference.stages.crops import (
        extract_classifier_crops, extract_classifier_crops_gpu)
    frame = (np.random.default_rng(0).integers(0, 256, (200, 300, 3), np.uint8))
    obb = _toy_obb(3)
    cpu = extract_classifier_crops(frame, obb, (128, 128), 2.0, 1.3)  # list HWC uint8
    cpu_t = np.stack([c.astype(np.float32) / 255.0 for c in cpu])     # (N,H,W,C)
    ft = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0)
    gpu = extract_classifier_crops_gpu(ft, obb, (128, 128), 2.0, 1.3, "cpu")
    gpu_hwc = gpu.permute(0, 2, 3, 1).numpy()
    assert gpu_hwc.shape == cpu_t.shape
    assert float(np.abs(gpu_hwc - cpu_t).mean()) < 0.03  # <~8/255 mean abs
```

Implement `extract_classifier_crops_batch_gpu` by looping frames, calling `extract_classifier_crops_gpu` per frame, and `torch.cat`-ing into a `CropBatch` with the SAME field construction as `extract_classifier_crops_batch` (`crops.py:317-333`) — copy that field layout verbatim, only swapping the crop tensor source.

- [ ] **Step 6: Run both tests, then commit**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py -v`
Expected: PASS (2 tests). If closeness fails, fix the affine composition (the `S @ M` scaling) — do NOT loosen the tolerance below 0.03 without noting why.

```bash
git add src/hydra_suite/core/inference/stages/crops.py tests/test_gpu_classifier_crop.py
git commit -m "feat(inference): GPU-native classifier crop extractor (grid_sample to input size)"
```

---

## Task 2: GPU factor-bundle forward (`_forward_multi_cuda`)

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/backend.py` (add `_forward_multi_cuda`, `supports_cuda_forward`)
- Test: `tests/test_gpu_classifier_crop.py` (append)

**Interfaces:**
- Consumes: `self._model` is a `list[ClassifierBackend]` when `_uses_factor_backends()`; each factor exposes `predict_batch_cuda(crops_chw, input_is_bgr) -> list[list[np.ndarray]]` and `_active_execution_backend in {"native","onnx"}`. Numpy reference: `_forward_yolo_multi` (backend.py:955) returns `np.concatenate([log(clip(probs)) ...], axis=-1)`.
- Produces:
  - `_forward_multi_cuda(self, crops_chw: list, input_is_bgr: bool) -> np.ndarray` — concatenated per-factor log-probs, SAME shape/semantics as `_forward_yolo_multi`, but each factor's probs come from `factor.predict_batch_cuda` instead of `factor.predict_batch`.
  - `supports_cuda_forward(self) -> bool` — True iff this backend (and, for a bundle, every factor) has a CUDA-native forward (`_active_execution_backend in {"native","onnx"}`).

- [ ] **Step 1: Write the failing test (forward shape parity)**

```python
def test_forward_multi_cuda_shape_matches_numpy(monkeypatch):
    # A fake 2-factor bundle: each factor returns fixed probs from both
    # predict_batch and predict_batch_cuda; the concatenated log-probs must match.
    import numpy as np
    from hydra_suite.core.identity.classification import backend as bk

    class _FakeFactor:
        _active_execution_backend = "native"
        def predict_batch(self, crops):           # numpy path
            return [[np.array([0.2, 0.8], np.float32)] for _ in crops]
        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return [[np.array([0.2, 0.8], np.float32)] for _ in crops]

    b = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    b._model = [_FakeFactor(), _FakeFactor()]
    monkeypatch.setattr(b, "_uses_factor_backends", lambda: True, raising=False)
    crops = [object(), object()]  # 2 crops
    numpy_out = b._forward_yolo_multi(crops)       # existing numpy reference
    cuda_out = b._forward_multi_cuda(crops, True)
    assert cuda_out.shape == numpy_out.shape       # (2 crops, 4 = 2 factors x 2)
    np.testing.assert_allclose(cuda_out, numpy_out, rtol=0, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_forward_multi_cuda_shape_matches_numpy -v`
Expected: FAIL with `AttributeError: ... has no attribute '_forward_multi_cuda'`

- [ ] **Step 3: Implement `_forward_multi_cuda` + `supports_cuda_forward`**

Mirror `_forward_yolo_multi` (backend.py:955) exactly, swapping the per-factor call:

```python
def _forward_multi_cuda(self, crops_chw: list, input_is_bgr: bool) -> "np.ndarray":
    import numpy as np
    per_factor_logits: list[np.ndarray] = []
    for factor_backend in self._model:
        factor_probs = factor_backend.predict_batch_cuda(crops_chw, input_is_bgr)
        probs = np.array(
            [per_crop[0] for per_crop in factor_probs], dtype=np.float32
        )
        per_factor_logits.append(np.log(np.clip(probs, 1e-9, 1.0)))
    return np.concatenate(per_factor_logits, axis=-1)

def supports_cuda_forward(self) -> bool:
    """True iff this backend has a CUDA-native forward (native torch or ONNX
    IOBinding); for a factor bundle, every factor must qualify."""
    if self._uses_factor_backends():
        return all(
            getattr(f, "_active_execution_backend", None) in ("native", "onnx")
            and hasattr(f, "predict_batch_cuda")
            for f in self._model
        )
    return getattr(self, "_active_execution_backend", None) in ("native", "onnx")
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_forward_multi_cuda_shape_matches_numpy -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/classification/backend.py tests/test_gpu_classifier_crop.py
git commit -m "feat(identity): GPU factor-bundle forward + supports_cuda_forward probe"
```

---

## Task 3: Route factor bundles through the GPU forward in `predict_batch_cuda`

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/backend.py` (`predict_batch_cuda`, the fallback branch at ~1151)

**Interfaces:**
- Consumes: `_forward_multi_cuda`, `supports_cuda_forward`, `_preprocess_cuda`, existing `_cardinalities()`/`_softmax` post-processing.
- Produces: `predict_batch_cuda` returns the same `[N_crops][K_factors]` structure, now computed on-GPU for CUDA-capable factor bundles instead of the numpy fallback.

- [ ] **Step 1: Write the failing test**

```python
def test_predict_batch_cuda_uses_gpu_forward_for_capable_bundle(monkeypatch):
    import numpy as np
    from hydra_suite.core.identity.classification import backend as bk

    b = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    called = {"numpy_fallback": False, "multi_cuda": False}

    class _F:
        _active_execution_backend = "native"
        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return [[np.array([0.1, 0.9], np.float32)] for _ in crops]
    b._model = [_F(), _F()]
    monkeypatch.setattr(b, "_uses_factor_backends", lambda: True, raising=False)
    monkeypatch.setattr(b, "_ensure_loaded", lambda: None, raising=False)
    monkeypatch.setattr(b, "_cardinalities", lambda: [2, 2], raising=False)
    orig = b._forward_multi_cuda
    def _spy(c, bgr): called["multi_cuda"] = True; return orig(c, bgr)
    monkeypatch.setattr(b, "_forward_multi_cuda", _spy, raising=False)
    def _no_numpy(crops): called["numpy_fallback"] = True; return []
    monkeypatch.setattr(b, "predict_batch", _no_numpy, raising=False)

    out = b.predict_batch_cuda([object(), object()], input_is_bgr=True)
    assert called["multi_cuda"] and not called["numpy_fallback"]
    assert len(out) == 2 and len(out[0]) == 2  # 2 crops, 2 factors
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_predict_batch_cuda_uses_gpu_forward_for_capable_bundle -v`
Expected: FAIL (current code hits the numpy `predict_batch` fallback → `called["numpy_fallback"]` True).

- [ ] **Step 3: Modify the fallback branch**

In `predict_batch_cuda`, replace the unconditional factor-bundle numpy fallback (backend.py:1151-1157) with: if `supports_cuda_forward()`, compute `logits = self._forward_multi_cuda(crops_chw, input_is_bgr)` and continue to the shared width-check + per-factor softmax block (the existing code after the forward). Only fall back to numpy when NOT `supports_cuda_forward()`:

```python
if self._uses_factor_backends():
    if self.supports_cuda_forward():
        logits = self._forward_multi_cuda(crops_chw, input_is_bgr)
    else:
        numpy_crops = [
            c.permute(1, 2, 0).cpu().numpy() if hasattr(c, "cpu") else c
            for c in crops_chw
        ]
        return self.predict_batch(numpy_crops)
    cardinalities = self._cardinalities()
    expected_total = sum(cardinalities)
    if logits.shape[-1] != expected_total:
        raise ClassifierRuntimeError(
            f"{self._model_path!r}: multi output width {logits.shape[-1]} "
            f"does not match expected total {expected_total}"
        )
    results: list[list[np.ndarray]] = []
    for row in logits:
        per_factor: list[np.ndarray] = []
        offset = 0
        for width in cardinalities:
            per_factor.append(self._softmax(row[offset : offset + width]))
            offset += width
        results.append(per_factor)
    return results
```

(Keep the existing non-bundle native/onnx path below unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_predict_batch_cuda_uses_gpu_forward_for_capable_bundle -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/classification/backend.py tests/test_gpu_classifier_crop.py
git commit -m "feat(identity): predict_batch_cuda runs CUDA-capable factor bundles on-GPU"
```

---

## Task 4: Strict load-time capability check

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/cnn.py` (`load_cnn_model`), `src/hydra_suite/core/inference/stages/headtail.py` (`load_headtail_model`)
- Test: `tests/test_gpu_classifier_crop.py` (append)

**Interfaces:**
- Consumes: `resolved_backend_for(runtime)` (already used in `load_cnn_model`, cnn.py:35), `runtime.tensor_on_cuda`, `backend.supports_cuda_forward()`.
- Produces: on `tensor_on_cuda` True, a classifier whose backend `not supports_cuda_forward()` raises `RuntimeError` at load naming the model + constraint.

- [ ] **Step 1: Write the failing test**

```python
def test_load_cnn_strict_raises_without_cuda_forward(monkeypatch, tmp_path):
    from hydra_suite.core.inference.stages import cnn as cnn_stage
    from hydra_suite.core.identity.classification import backend as bk

    class _NoCudaBackend:
        metadata = type("M", (), {"input_size": (128, 128),
                                  "factor_names": ["f"],
                                  "class_names_per_factor": [["a", "b"]]})()
        def supports_cuda_forward(self): return False
    monkeypatch.setattr(bk, "ClassifierBackend", lambda *a, **k: _NoCudaBackend())

    class _RT:  # minimal runtime stub with tensor_on_cuda True
        tensor_on_cuda = True
    monkeypatch.setattr(cnn_stage, "resolved_backend_for",
                        lambda rt: type("R", (), {"backend": "torch", "device": "cuda"})())
    cfg = type("C", (), {"model_path": str(tmp_path / "m.multihead.json")})()
    with pytest.raises(RuntimeError, match="CUDA-native"):
        cnn_stage.load_cnn_model(cfg, _RT())
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_load_cnn_strict_raises_without_cuda_forward -v`
Expected: FAIL (no check yet → no raise). Adjust the stub attributes to match the real `load_cnn_model` body after reading cnn.py:28-50; keep the `pytest.raises(RuntimeError, match="CUDA-native")` assertion.

- [ ] **Step 3: Add the check to both loaders**

After constructing `backend` in `load_cnn_model` (cnn.py:43) and `load_headtail_model`, before returning the model:

```python
if getattr(runtime, "tensor_on_cuda", False) and not backend.supports_cuda_forward():
    raise RuntimeError(
        f"Classifier {config.model_path!r} lacks a CUDA-native forward, but the "
        f"gpu tier with NVDEC requires it (no silent CPU fallback). Use a native "
        f"torch / ONNX classifier, or run on the cpu tier."
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py::test_load_cnn_strict_raises_without_cuda_forward -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/cnn.py src/hydra_suite/core/inference/stages/headtail.py tests/test_gpu_classifier_crop.py
git commit -m "feat(inference): strict gpu-tier classifier CUDA-forward capability check"
```

---

## Task 5: Stage routing — GPU path in `run_cnn_batch` + `run_headtail_batch`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/cnn.py` (`run_cnn_batch`), `src/hydra_suite/core/inference/stages/headtail.py` (`run_headtail_batch`)
- Test: `tests/test_gpu_classifier_crop.py` (append)

**Interfaces:**
- Consumes: `extract_classifier_crops_batch_gpu` (Task 1), `model.backend.predict_batch_cuda` (Tasks 2-3), `runtime.tensor_on_cuda`, `runtime.device`.
- Produces: unchanged return types (`dict[int, CNNResult]` / `dict[int, HeadTailResult]`).

- [ ] **Step 1: Write the failing test (routing)**

```python
def test_run_cnn_batch_takes_gpu_path_when_tensor_on_cuda(monkeypatch):
    from hydra_suite.core.inference.stages import cnn as cnn_stage
    used = {"gpu_crop": False, "cuda_forward": False}

    def _gpu_crop(frames, obbs, size, ar, mg, dev):
        used["gpu_crop"] = True
        import torch
        from hydra_suite.core.inference.stages.crops import CropBatch
        # ... build a tiny CropBatch with 1 crop on 'cpu' device (stand-in)
        raise NotImplementedError  # replace with real minimal CropBatch per crops.py
    # Spy predict_batch_cuda; assert the cv2/numpy predict_batch path is NOT called
    # when runtime.tensor_on_cuda is True. Fill in against the real signatures.
```

Note: this routing test is best written against the real `CropBatch`/`OBBResult`; read `crops.py:CropBatch` and construct a 1-frame/1-detection batch. Assert: with a runtime whose `tensor_on_cuda=True`, `extract_classifier_crops_batch_gpu` and `predict_batch_cuda` are invoked and the numpy `predict_batch` is not; with `tensor_on_cuda=False`, the reverse.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py -k run_cnn_batch_takes_gpu -v`
Expected: FAIL (no routing yet).

- [ ] **Step 3: Add routing to `run_cnn_batch`**

Replace the crop+forward block (the vectorized CPU block from commit `8d1d8ef`) with a device branch:

```python
if getattr(runtime, "tensor_on_cuda", False):
    from .crops import extract_classifier_crops_batch_gpu
    batch = extract_classifier_crops_batch_gpu(
        frames, obb_results, model.input_size, aspect_ratio, margin, runtime.device
    )
    n_total = batch.crops.shape[0]
    if n_total:
        # crops stay on-device: pass CHW cuda tensors scaled to [0,255], BGR
        cuda_crops = [batch.crops[i] * 255.0 for i in range(n_total)]
        all_probs = model.backend.predict_batch_cuda(cuda_crops, input_is_bgr=True)
    else:
        all_probs = []
else:
    from .crops import extract_classifier_crops_batch
    batch = extract_classifier_crops_batch(
        frames, obb_results, model.input_size, aspect_ratio, margin
    )
    n_total = batch.crops.shape[0]
    if n_total:
        hwc_all = np.ascontiguousarray(batch.crops.permute(0, 2, 3, 1).cpu().numpy())
        stacked = (hwc_all * 255.0).clip(0, 255).astype(np.uint8)
        all_probs = model.backend.predict_batch(list(stacked))
    else:
        all_probs = []
```

Keep the existing per-frame assembly loop (`_assemble_cnn_result`) unchanged — it consumes `all_probs` the same way. Apply the identical pattern to `run_headtail_batch` (with `extract_classifier_crops_batch_gpu` + `_assemble_headtail_result`).

- [ ] **Step 4: Run routing test + the full stage test file**

Run: `PYTHONPATH=src python -m pytest tests/test_gpu_classifier_crop.py tests/test_inference_batch_stages.py tests/test_headtail_consumer.py -v`
Expected: PASS (routing tests + no regression in existing batch-stage tests, which run `tensor_on_cuda=False` → CPU path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/stages/cnn.py src/hydra_suite/core/inference/stages/headtail.py tests/test_gpu_classifier_crop.py
git commit -m "feat(inference): route classifier stages through GPU crop+forward on tensor_on_cuda"
```

---

## Task 6: Verification (MPS byte-identity + CUDA determinism/agreement/perf)

**Files:** none (verification only). Uses `tools/equivalence/`.

- [ ] **Step 1: MPS byte-identity guard (local, reliable)**

The MPS path has `tensor_on_cuda=False` → CPU path untouched. Confirm byte-identity vs the pre-feature branch tip:

```bash
cd .worktree/gpu-classifier && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate hydra-mps
export KMP_DUPLICATE_LIB_OK=TRUE
FX=/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/tools/equivalence/fixtures
for clip in ant_obb_sleap ant_cnn_identity; do
  PYTHONPATH=src python tools/equivalence/runner.py \
    --orig-config "$FX/configs/$clip.json" --video "$FX/clips/$clip.mp4" \
    --outdir "/tmp/gpucrop_mps/$clip" --runtime mps --label feat \
    --skeleton "$FX/ooceraea_biroi.json" --detection-batch-size 1
done
```
Compare each `*_forward.csv` to a run from `main` (checkout `main` src into a second worktree or reuse an equiv baseline). Expected: `VERDICT: EQUIVALENT` (pos/θ max 0). Verify row counts > 1 first.

- [ ] **Step 2: CUDA determinism + agreement + perf (mehek)**

On mehek (`hydra-cuda`, kill sleap zombies first), run `ant_cnn_identity` via `run_matrix.sh` with MAIN_SRC = current `main` worktree and WT_SRC = this feature branch worktree, `RUNTIME=cuda`. Expected:
- DETERMINISM (new_a vs new_b): EQUIVALENT.
- EQUIVALENCE (main vs feat): identity-label agreement ≥ 99% and positions within determinism floor (the harness prints per-column diffs; check `IdentityAssignedLabel` mismatch rate < 1%).
- PERFORMANCE: `feat/main` wall-clock ratio materially < 1.0 (the 1.49× regression closes). Record the number.

- [ ] **Step 3: Delta gate + docs**

Run: `PYTHONPATH=src python -m pytest tests/ -k "cnn or headtail or headless or inference_batch or gpu_classifier" -q` — no NEW failures vs `main`.
Update `docs/superpowers/specs/2026-07-22-gpu-native-classifier-crop-design.md` Status → Implemented, and append the measured CUDA perf ratio + MPS byte-identity result. Commit.

```bash
git add docs/superpowers/specs/2026-07-22-gpu-native-classifier-crop-design.md
git commit -m "docs(spec): mark GPU classifier crop path implemented + record perf/agreement"
```

---

## Self-Review Notes

- **Spec coverage:** platform split (Tasks 1,5) ✓; three coupled changes (Tasks 1,2/3,5) ✓; strict capability check (Task 4) ✓; acceptance gate (Task 6) ✓; MPS unchanged/byte-identical (Task 5 else-branch + Task 6 Step 1) ✓; non-goals (AprilTag/pose) untouched ✓.
- **Numerics risk:** grid_sample ≠ cv2 — Task 1 gates crop closeness (<0.03 mean abs); Task 6 gates end-to-end identity agreement (≥99%). If Task 1 closeness fails, the affine composition (`S @ M`) is wrong — fix before proceeding; do not loosen tolerances silently.
- **CUDA-only validation:** Tasks 1-5 unit tests run on CPU device (grid_sample + fakes), so correctness is provable off-box; only Task 6 Step 2 needs mehek.
- **Per-frame `run_cnn`/`run_headtail`:** left on the CPU path (realtime forces batch=1 and is latency-bound); batch path is where dense-colony perf lives. If a future need arises, apply the same Task-5 branch.
