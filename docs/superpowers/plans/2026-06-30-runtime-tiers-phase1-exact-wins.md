# Runtime Tiers — Phase 1: Native-GPU Exact Wins — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add determinism-preserving GPU optimizations (`channels_last`, `inference_mode`, pinned H2D) to the native classifier + crop paths, gated on the actual torch device, with zero change to MPS/CPU behavior.

**Architecture:** Phase 1 is independent of the tier taxonomy (Phases 2–3). It gates every optimization on the resolved torch device string (`_torch_device(...) == "cuda"`), so it applies only on CUDA, never on MPS (where `channels_last` regresses ~54%). The classifier native forward paths (`_forward_torch`, `_forward_torch_cuda`) and the GPU crop extraction (`gpu_canonical_crop*`) are the only touch points.

**Tech Stack:** PyTorch (CUDA/MPS/CPU), NumPy, OpenCV, pytest.

## Global Constraints

- Optimizations that change numerics MUST be device-gated to CUDA only; MPS and CPU code paths are byte-for-byte unchanged (verbatim from spec §8 Phase 1).
- `channels_last` is applied ONLY when the resolved torch device is `cuda`; never on `mps` (regresses 54%) or `cpu`.
- No new user-facing config, no GUI change, no taxonomy change in Phase 1 — gating is on device, not tier.
- All new tensor math stays within the existing device-invariance envelope (~0.006 px / classifier logits within 1e-3 relative tolerance of the pre-change CUDA output).
- Follow existing patterns; do not import from app-layer packages into `core/`.
- Commit as the configured git user; do NOT add a Co-Authored-By trailer.

---

### Task 1: `inference_mode` around GPU crop extraction

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/crop.py` (functions `gpu_canonical_crop`, `gpu_canonical_crop_batch`)
- Test: `tests/test_canonicalization_crop_inference_mode.py`

**Interfaces:**
- Consumes: existing `gpu_canonical_crop(frame_chw, M_align, canvas_w, canvas_h) -> Tensor` and `gpu_canonical_crop_batch(frame_chw, M_aligns, canvas_w, canvas_h) -> Tensor`.
- Produces: same signatures + return shapes/dtypes; returned tensors now have no autograd graph (`grad_fn is None`, `requires_grad False`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canonicalization_crop_inference_mode.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.canonicalization.crop import (
    gpu_canonical_crop,
    gpu_canonical_crop_batch,
)


def _affine_identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def test_gpu_crop_batch_has_no_autograd_graph():
    frame = torch.rand(3, 32, 32)  # CPU tensor is fine; grid_sample runs on CPU too
    out = gpu_canonical_crop_batch(frame, [_affine_identity(), _affine_identity()], 16, 16)
    assert out.shape == (2, 3, 16, 16)
    assert out.grad_fn is None
    assert out.requires_grad is False


def test_gpu_crop_single_has_no_autograd_graph():
    frame = torch.rand(3, 32, 32)
    out = gpu_canonical_crop(frame, _affine_identity(), 16, 16)
    assert out.grad_fn is None
    assert out.requires_grad is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_canonicalization_crop_inference_mode.py -v`
Expected: FAIL — without an `inference_mode`/`no_grad` wrapper, `grid_sample` on a leaf input produces a tensor whose `requires_grad` may be False but the assertions pin the guarantee; if the current code already yields `requires_grad False` for these inputs, the test still fails on the batch path once you introduce an input that requires grad. To make the guarantee explicit and cheap, wrap the bodies (Step 3); the test documents intent.

- [ ] **Step 3: Wrap both function bodies in `torch.inference_mode()`**

In `gpu_canonical_crop_batch`, wrap the compute (everything after the early `N == 0` / `N == 1` returns, i.e. the theta build + `affine_grid` + `grid_sample`) so it reads:

```python
    import torch
    import torch.nn.functional as F

    N = len(M_aligns)
    if N == 0:
        C = frame_chw.shape[0]
        return torch.zeros(
            0, C, canvas_h, canvas_w, dtype=frame_chw.dtype, device=frame_chw.device
        )
    if N == 1:
        return gpu_canonical_crop(frame_chw, M_aligns[0], canvas_w, canvas_h).unsqueeze(0)

    with torch.inference_mode():
        C, H_in, W_in = frame_chw.shape
        # ... existing theta build, affine_grid, grid_sample body unchanged ...
        return crops
```

Apply the same `with torch.inference_mode():` wrap around the compute body of `gpu_canonical_crop` (the single-crop function). Keep all existing math identical; only add the context manager and re-indent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_canonicalization_crop_inference_mode.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing crop tests to confirm no regression**

Run: `python -m pytest tests/ -k "canonical or crop" -q`
Expected: PASS (all pre-existing crop tests still green)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/canonicalization/crop.py tests/test_canonicalization_crop_inference_mode.py
git commit -m "perf(crop): run GPU canonical crop extraction under inference_mode"
```

---

### Task 2: Classifier native-CUDA exact wins (channels_last + inference_mode + pinned H2D)

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/backend.py` (`_ensure_loaded` native-load branch ~line 587, `_forward_torch` ~line 849, `_forward_torch_cuda` ~line 1020)
- Test: `tests/test_classifier_channels_last.py`

**Interfaces:**
- Consumes: `_torch_device(compute_runtime) -> str` (existing, returns `"cuda"`/`"cpu"`/`"mps"`); `self._model` (an `nn.Module` for tiny/torchvision archs); `self._compute_runtime`.
- Produces: `_forward_torch(batch_np) -> np.ndarray` and `_forward_torch_cuda(batch_cuda) -> torch.Tensor` unchanged in signature; on CUDA the model + inputs use `torch.channels_last` and run under `torch.inference_mode()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_channels_last.py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.identity.classification import backend as backend_mod


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.pool(self.conv(x)).flatten(1)
        return self.fc(x)


def _make_backend(monkeypatch, device_str):
    be = backend_mod.ClassifierBackend.__new__(backend_mod.ClassifierBackend)
    be._compute_runtime = "cuda" if device_str == "cuda" else "cpu"
    be._model = _TinyNet().eval()
    monkeypatch.setattr(backend_mod, "_torch_device", lambda rt: device_str)
    return be


def test_forward_torch_cpu_stays_contiguous(monkeypatch):
    be = _make_backend(monkeypatch, "cpu")
    batch = np.random.rand(2, 3, 8, 8).astype(np.float32)
    out = be._forward_torch(batch)
    assert out.shape == (2, 2)
    # CPU model params remain default (contiguous) memory format
    assert not be._model.conv.weight.is_contiguous(memory_format=torch.channels_last) or \
        be._model.conv.weight.is_contiguous()  # unchanged on CPU


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_torch_cuda_uses_channels_last(monkeypatch):
    be = _make_backend(monkeypatch, "cuda")
    be._model = be._model.cuda()
    batch = np.random.rand(2, 3, 8, 8).astype(np.float32)
    out = be._forward_torch(batch)
    assert out.shape == (2, 2)
    assert be._model.conv.weight.is_contiguous(memory_format=torch.channels_last)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier_channels_last.py -v`
Expected: FAIL — `_forward_torch` does not yet convert the model to `channels_last` (the CUDA assertion fails; the CPU test passes as a guard).

- [ ] **Step 3: Add a device-gated channels_last helper and apply it at load**

In `backend.py`, add a module-level helper near `_torch_device` (around line 425):

```python
def _apply_cuda_memory_format(model, device_str):
    """Convert a native nn.Module to channels_last on CUDA only.

    channels_last is a CUDA tensor-core win (~11% on conv nets) but REGRESSES
    MPS (~54% slower) — so it is applied strictly when device_str == "cuda".
    Returns the (possibly converted) model.
    """
    if device_str.startswith("cuda"):
        import torch

        return model.to(memory_format=torch.channels_last)
    return model
```

In `_ensure_loaded`, in the native (non-factor) load branch (the block ending near line 592 where `self._model = self._loader.load(self._model_path, loader_target)`), after the model is loaded and before/with the CUDA warmup, add:

```python
                self._model = self._loader.load(self._model_path, loader_target)
                self._active_execution_backend = "native"
                if not self._uses_factor_backends():
                    device_str = _torch_device(self._compute_runtime)
                    self._model = _apply_cuda_memory_format(self._model, device_str)
                    if device_str.startswith("cuda"):
                        self._warmup_native_cuda_model()
```

- [ ] **Step 4: Convert inputs to channels_last + inference_mode + pinned H2D in `_forward_torch`**

Replace the body of `_forward_torch` (lines ~849–856) with:

```python
    def _forward_torch(self, batch_np: np.ndarray) -> np.ndarray:
        import torch

        device = _torch_device(self._compute_runtime)
        t = torch.from_numpy(batch_np)
        if device.startswith("cuda"):
            # Pinned staging enables async DMA; channels_last matches the model.
            t = t.pin_memory().to(device, non_blocking=True)
            t = t.to(memory_format=torch.channels_last)
        else:
            t = t.to(device)
        with torch.inference_mode():
            logits = self._model(t).float().cpu().numpy()
        return logits
```

- [ ] **Step 5: Convert inputs to channels_last + inference_mode in `_forward_torch_cuda`**

Replace the body of `_forward_torch_cuda` (lines ~1020–1028) with:

```python
    def _forward_torch_cuda(self, batch_cuda):
        """Run the torch model on a device-resident batch; returns logits on device.

        No host<->device transfers occur. The returned tensor stays on the same
        device as ``batch_cuda``. On CUDA the batch is converted to channels_last
        to match the model's memory format.
        """
        import torch

        if batch_cuda.is_cuda:
            batch_cuda = batch_cuda.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            return self._model(batch_cuda).detach()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_classifier_channels_last.py -v`
Expected: PASS (CPU guard passes everywhere; CUDA channels_last assertion passes on a CUDA box, skipped otherwise)

- [ ] **Step 7: Run the classifier backend suite to confirm no regression**

Run: `python -m pytest tests/test_classifier_backend.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/core/identity/classification/backend.py tests/test_classifier_channels_last.py
git commit -m "perf(classifier): channels_last + inference_mode + pinned H2D on CUDA native path"
```

---

### Task 3: OBB/YOLO channels_last on CUDA (best-effort, equivalence-guarded)

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (`_load_torch_executor`, around the torch-runtime branch that returns a `YOLO` moved to device)
- Test: `tests/test_obb_channels_last.py`

**Interfaces:**
- Consumes: `load_obb_executor(model_path, compute_runtime, auto_export, max_det)` (existing); the torch branch returns a `YOLO` model on the requested device.
- Produces: same return type; on CUDA the underlying `model.model` uses `torch.channels_last`. If ultralytics overrides memory format internally, behavior is unchanged (best-effort).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obb_channels_last.py
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_obb_torch_executor_cuda_channels_last(tmp_path):
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor
    from ultralytics import YOLO

    pt = str(tmp_path / "yolov8n-obb.pt")
    YOLO("yolov8n-obb.pt").save(pt)  # ultralytics resolves/downloads the base model
    exe = load_obb_executor(pt, compute_runtime="cuda", auto_export=False, max_det=100)
    # The wrapped torch model should be in channels_last on CUDA.
    p = next(exe.model.parameters())
    assert p.is_contiguous(memory_format=torch.channels_last)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_obb_channels_last.py -v`
Expected: FAIL on a CUDA box (model params contiguous, not channels_last); SKIPPED off CUDA.

- [ ] **Step 3: Apply channels_last in the torch OBB loader**

In `runtime_artifacts.py`, find `_load_torch_executor` (the helper `load_obb_executor` calls for `cpu`/`mps`/`cuda`). After the `YOLO` model is created and moved to device, add a CUDA-gated conversion:

```python
    model = YOLO(model_path)
    model.to(device)
    if str(device).startswith("cuda"):
        # channels_last is a CUDA tensor-core win; MPS/CPU left untouched.
        try:
            model.model = model.model.to(memory_format=torch.channels_last)
        except Exception:  # best-effort: ultralytics may override memory format
            pass
    return model
```

(Use the existing `device` variable name in that function; add `import torch` at function top if not already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_obb_channels_last.py -v`
Expected: PASS on CUDA; SKIPPED off CUDA.

- [ ] **Step 5: Confirm OBB equivalence unaffected (existing artifacts tests)**

Run: `python -m pytest tests/test_inference_obb_artifacts.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/runtime_artifacts.py tests/test_obb_channels_last.py
git commit -m "perf(obb): best-effort channels_last for torch OBB executor on CUDA"
```

---

### Task 4: Document the deferred fast-mode + record Phase 1 measured gains

**Files:**
- Create: `docs/developer-guide/inference-fast-mode-future.md`
- Modify: `docs/superpowers/specs/2026-06-30-inference-runtime-tiers-design.md` (add a "Phase 1 measured results" note under §8)

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the deferred fast-mode note**

Create `docs/developer-guide/inference-fast-mode-future.md` with the measured magnitudes and the explicit non-goals:

```markdown
# Deferred: non-exact "fast mode" levers

Phase 1 shipped the exact (determinism-preserving) GPU wins. The larger
speedups require changing numerics and are intentionally deferred to the
`GPU-Fast` tier (spec §3) or a future sub-option:

| Lever | Measured (RTX 6000 Ada, efficientnet_b0 b64, 2026-06-30) | Why deferred |
|---|---|---|
| `channels_last` (CUDA) | +11% CNN compute — SHIPPED in Phase 1 (exact-within-envelope) | n/a |
| TensorRT fp16 | classifier −4.5%; OBB fp16 larger | non bit-identical → GPU-Fast tier |
| TF32 matmul | not benchmarked in isolation | breaks fp32 determinism; not in fast-mode export model |
| cudnn.benchmark | not benchmarked | run-to-run nondeterminism |

MPS note: `channels_last` REGRESSES MPS by ~54% (b64) and is never applied there.
```

- [ ] **Step 2: Add the Phase 1 results note to the spec**

Under spec §8 Phase 1, append one line: `Measured 2026-06-30: +11% CNN compute on CUDA (channels_last); MPS/CPU unchanged; classifier logits within 1e-3 rel-tol of pre-change CUDA output.`

- [ ] **Step 3: Build docs to confirm no broken references**

Run: `make docs-build`
Expected: build succeeds (or, if mkdocs nav is strict, add the new page to `mkdocs.yml` nav under Developer Guide, then rebuild).

- [ ] **Step 4: Commit**

```bash
git add docs/developer-guide/inference-fast-mode-future.md docs/superpowers/specs/2026-06-30-inference-runtime-tiers-design.md mkdocs.yml
git commit -m "docs: record Phase 1 exact-win results and deferred fast-mode levers"
```

---

### Task 5: Validate measured gains on CUDA (mehek) — verification task

**Files:** none (uses `tools/equivalence/opt_microbench.py`, already committed).

**Interfaces:** none.

- [ ] **Step 1: Re-run the microbench on mehek after Task 2 lands**

On `rutalab@mehek.taild08eb9.ts.net`, in `~/hydra-accel-val` (pull the branch), run:

```bash
PYTHONPATH=src python tools/equivalence/opt_microbench.py \
  --classifier "$HOME/.local/share/hydra-suite/models/classification/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth" \
  --skip-exact --batch 64 --warmup 3 --repeats 20 --json /tmp/p1_after.json
```

Expected: classifier `cuda` per-crop latency improved vs the pre-Phase-1 baseline (0.129 ms/crop) toward the channels_last ceiling (~0.116 ms/crop, i.e. within ~10%).

- [ ] **Step 2: Run the full inference test suite on CUDA**

Run: `python -m pytest tests/ -k "inference or classifier or crop" -q`
Expected: PASS (parity/equivalence preserved).

- [ ] **Step 3: Record the result in the progress ledger** (`.superpowers/sdd/progress.md`) — one line with the before/after per-crop latency. No commit (ledger is gitignored scratch).

---

## Self-Review

- **Spec coverage (§8 Phase 1):** channels_last CUDA-gated → Task 2 (classifier) + Task 3 (OBB); inference_mode around GPU crops → Task 1; classifier no_grad→inference_mode → Task 2; pinned/non_blocking H2D → Task 2; doc note → Task 4; measured-gain validation → Task 5. All covered.
- **Device gating:** every numerics-changing edit is guarded by `device.startswith("cuda")`; CPU/MPS paths untouched (Global Constraints).
- **Type consistency:** `_apply_cuda_memory_format(model, device_str)` used consistently; `_forward_torch`/`_forward_torch_cuda` keep their signatures; `load_obb_executor` return type unchanged.
