# NVDEC Confined to the GPU-Fast Tier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confine NVDEC hardware decode to the `gpu_fast` tier so the `gpu` tier stays byte-identical to CPU/MPS, while NVDEC frames on `gpu_fast` flow end-to-end through the on-GPU crop + classifier path.

**Architecture:** Three small, surgical gate changes on the feature branch `feature/gpu-native-classifier-crop`: (1) `RuntimeContext.from_config` only enables `use_nvdec` on `gpu_fast`; (2) the classifier GPU-crop gate `frames_on_cuda` keys off `requested_gpu` + an actual CUDA frame instead of `tensor_on_cuda` (which is False on `gpu_fast`); (3) the strict classifier load-check fires on `cuda_mode` (both GPU tiers on a CUDA box) so a `gpu_fast` NVDEC frame can never reach the RGB-unaware numpy fallback. The OBB TensorRT path already accepts CUDA HWC frames — that is locked in with a regression test, not changed.

**Tech Stack:** Python, PyTorch (`grid_sample`, CUDA tensors), Ultralytics YOLO, ONNX Runtime / TensorRT EP, PyNvVideoCodec + cupy (NVDEC), pytest.

## Global Constraints

- **Branch:** all work lands on `feature/gpu-native-classifier-crop` (current `HEAD` = `95ead089`). Do not branch off `main`.
- **`gpu` tier stays byte-identical** to CPU/MPS: it must always CPU-decode (never NVDEC). This is the primary correctness gate.
- **`gpu_fast` accepts the small color residual** (max ~2 / mean ~0.48 per channel). It is *reported, not gated* — never add a byte-identity assertion for `gpu_fast`.
- **CPU / MPS tiers unchanged** — byte-identical before/after.
- **Tests must run on the Mac dev box** (CPU torch device; `grid_sample` works there). Any assertion needing a real CUDA tensor MUST be guarded by `if torch.cuda.is_available():` (skip otherwise). The end-to-end hardware gate is Task 6 (mehek).
- **Run tests with the worktree src on the path:** `PYTHONPATH=$PWD/src python -m pytest tests/<file> -v` (editable/worktree install).
- **Pre-commit hook reformats then aborts the first commit.** Always stage-and-commit twice: `git add -A && git commit -m "..." ; git add -A && git commit -m "..."` (second call succeeds with the reformatted tree). Verify with `git status` that the tree is clean after.
- **Commit as the configured git user** — do NOT add a `Co-Authored-By: Claude` trailer (per user preference / memory `feedback_git_commit_identity`).
- **No `compute_runtime` strings** — the runtime vocabulary is `ResolvedBackend` + tier flags only (Runtime Gen-2). Do not reintroduce string runtime names.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/hydra_suite/core/inference/runtime.py` | Builds `RuntimeContext` from tier | Add `_should_use_nvdec(tier)`; gate `use_nvdec` on it; update `use_nvdec` field docstring |
| `src/hydra_suite/core/inference/stages/crops.py` | Classifier GPU-crop gate | `frames_on_cuda` keys off `requested_gpu`, not `tensor_on_cuda` |
| `src/hydra_suite/core/inference/stages/cnn.py` | CNN load + batch | Strict load-check gate `tensor_on_cuda` → `cuda_mode` |
| `src/hydra_suite/core/inference/stages/headtail.py` | Head/tail load + batch | Strict load-check gate `tensor_on_cuda` → `cuda_mode` |
| `src/hydra_suite/core/identity/classification/backend.py` | Classifier forward | Numpy fallback in `predict_batch_cuda` must forward `input_is_bgr` (defensive) |
| `src/hydra_suite/core/inference/sources.py` | Frame-source selection | Promote reader-selection log to `info` (acceptance visibility) |
| `tests/test_nvdec_gpu_fast_tier.py` | New unit tests | Create: NVDEC tiering, OBB CUDA-frame routing, reader-log |
| `tests/test_gpu_classifier_crop.py` | Existing crop/forward tests | Update gate references (`tensor_on_cuda` → `requested_gpu`/`cuda_mode`) |

---

## Task 1: Gate NVDEC on the `gpu_fast` tier

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime.py:31` (field docstring), `:140-145` (nvdec gate), and add `_should_use_nvdec` near `_nvdec_available` (`:205`)
- Test: `tests/test_nvdec_gpu_fast_tier.py` (create)

**Interfaces:**
- Consumes: existing `_nvdec_available() -> bool`, `RuntimeContext.from_config`.
- Produces: `_should_use_nvdec(runtime_tier: str) -> bool` — returns `True` only when `runtime_tier == "gpu_fast"` **and** `_nvdec_available()`. `RuntimeContext.use_nvdec` is now `True` only on `gpu_fast` (CUDA box, NVDEC libs present).

- [ ] **Step 1: Write the failing test**

Create `tests/test_nvdec_gpu_fast_tier.py`:

```python
"""Unit tests for NVDEC confined to the gpu_fast tier (spec 2026-07-22-nvdec-gpu-fast-tier).

Runs on the Mac dev box: the tier-gating logic is pure and does not need a CUDA
device. The end-to-end NVDEC-engaged gate is Task 6 (mehek).
"""

import numpy as np
import pytest
import torch


def test_should_use_nvdec_gpu_fast_only(monkeypatch):
    from hydra_suite.core.inference import runtime as rt_mod

    # NVDEC libraries present: only gpu_fast enables it.
    monkeypatch.setattr(rt_mod, "_nvdec_available", lambda: True)
    assert rt_mod._should_use_nvdec("gpu_fast") is True
    assert rt_mod._should_use_nvdec("gpu") is False
    assert rt_mod._should_use_nvdec("cpu") is False

    # NVDEC libraries absent: never, even on gpu_fast.
    monkeypatch.setattr(rt_mod, "_nvdec_available", lambda: False)
    assert rt_mod._should_use_nvdec("gpu_fast") is False
    assert rt_mod._should_use_nvdec("gpu") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_nvdec_gpu_fast_tier.py::test_should_use_nvdec_gpu_fast_only -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_should_use_nvdec'`.

- [ ] **Step 3: Add the helper**

In `src/hydra_suite/core/inference/runtime.py`, immediately after `_nvdec_available()` (ends ~`:220`), add:

```python
def _should_use_nvdec(runtime_tier: str) -> bool:
    """NVDEC is confined to the gpu_fast tier.

    The gpu tier is contracted to be byte-identical to CPU/MPS, so it must always
    CPU-decode; only gpu_fast (the "fastest path, accepts small numerical
    differences" tier) may use NVDEC. See
    docs/superpowers/specs/2026-07-22-nvdec-gpu-fast-tier-design.md.
    """
    return runtime_tier == "gpu_fast" and _nvdec_available()
```

- [ ] **Step 4: Wire it into `from_config`**

In `src/hydra_suite/core/inference/runtime.py`, change the `cuda_mode` branch (currently `:140-145`):

```python
        if cuda_mode:
            device = _cuda_device_available()
            nvdec = _should_use_nvdec(config.runtime_tier)
        else:
            device = _cpu_or_mps_device()
            nvdec = False
```

(was `nvdec = _nvdec_available()`). `cuda_mode` already implies `has_cuda` + tier ∈ {gpu, gpu_fast}; `_should_use_nvdec` narrows it to `gpu_fast`.

- [ ] **Step 5: Update the `use_nvdec` field docstring**

In `src/hydra_suite/core/inference/runtime.py:31`, change the inline comment on the `use_nvdec` field:

```python
    use_nvdec: bool  # True only on the gpu_fast tier with NVDEC available
```

(was `# cuda_mode AND NVDEC available`).

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_nvdec_gpu_fast_tier.py::test_should_use_nvdec_gpu_fast_only -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(inference): confine NVDEC to the gpu_fast tier"
git add -A && git commit -m "feat(inference): confine NVDEC to the gpu_fast tier"
git status   # must be clean
```

---

## Task 2: Decouple `frames_on_cuda` from `tensor_on_cuda`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py:359-375` (`frames_on_cuda`)
- Test: `tests/test_gpu_classifier_crop.py:229` (`test_frames_on_cuda_gate` — update)

**Interfaces:**
- Consumes: `runtime.requested_gpu: bool` (True on both `gpu`/`gpu_fast`), `torch.is_tensor`, `frame.is_cuda`.
- Produces: `frames_on_cuda(runtime, frames) -> bool` — `True` iff `runtime.requested_gpu` **and** the first non-`None` frame is a CUDA `torch.Tensor`. This is what routes the classifier crop through `extract_classifier_crops_batch_gpu` + `predict_batch_cuda`.

**Why:** on `gpu_fast` the OBB backend is `tensorrt`, so `tensor_on_cuda` is `False` — but NVDEC frames are genuine CUDA tensors that should take the on-GPU crop path. Keying off `requested_gpu` (True on both GPU tiers) + the actual frame device fixes this without re-engaging the GPU path when NVDEC has fallen back to CPU frames (numpy → `False`).

- [ ] **Step 1: Update the failing test**

In `tests/test_gpu_classifier_crop.py`, replace `test_frames_on_cuda_gate` (currently `:229-245`) with:

```python
def test_frames_on_cuda_gate():
    from hydra_suite.core.inference.stages.crops import frames_on_cuda

    # requested_gpu gates the path (True on gpu AND gpu_fast); tensor_on_cuda is
    # irrelevant here (it is False on gpu_fast, where NVDEC frames still belong
    # on the GPU crop path).
    rt_gpu = type("RT", (), {"requested_gpu": True})()
    rt_cpu = type("RT", (), {"requested_gpu": False})()
    cpu_tensor = torch.zeros((3, 8, 8))
    np_frame = np.zeros((8, 8, 3), np.uint8)

    # Not a gpu tier -> never.
    assert frames_on_cuda(rt_cpu, [cpu_tensor]) is False
    # gpu tier but the frame is CPU (NVDEC fell back to CpuFrameReader) -> False:
    # uploading a CPU frame to GPU just to crop is slower than cv2.
    assert frames_on_cuda(rt_gpu, [cpu_tensor]) is False
    assert frames_on_cuda(rt_gpu, [np_frame]) is False
    assert frames_on_cuda(rt_gpu, []) is False
    if torch.cuda.is_available():
        assert frames_on_cuda(rt_gpu, [cpu_tensor.cuda()]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_gpu_classifier_crop.py::test_frames_on_cuda_gate -v`
Expected: FAIL — `frames_on_cuda` still reads `tensor_on_cuda`, which the new `rt_gpu` no longer defines (`getattr(..., False)` → returns `False` for the CPU-tensor branch too, but the CUDA branch under `torch.cuda.is_available()` would fail on a CUDA box; on the Mac the assertion `frames_on_cuda(rt_gpu, [cpu_tensor]) is False` passes for the wrong reason). To make the failure explicit on any box, the implementation change in Step 3 is what the test validates. (On the Mac, confirm the test at least imports and the non-CUDA assertions hold; the CUDA assertion is the real gate on mehek.)

- [ ] **Step 3: Change the gate**

In `src/hydra_suite/core/inference/stages/crops.py`, replace `frames_on_cuda` (`:359-375`) body + docstring:

```python
def frames_on_cuda(runtime, frames) -> bool:
    """Whether the GPU classifier crop path should run for this window.

    Requires BOTH a GPU tier (``runtime.requested_gpu`` — True on ``gpu`` and
    ``gpu_fast``) AND frames that are genuinely CUDA tensors. It deliberately does
    NOT key off ``tensor_on_cuda``: on ``gpu_fast`` the OBB backend is ``tensorrt``
    so ``tensor_on_cuda`` is False, yet NVDEC frames are real CUDA tensors that
    belong on the on-GPU crop path. NVDEC can also fall back to ``CpuFrameReader``
    per clip (e.g. the H.264 4096 / MBCount limit), in which case the frames are
    CPU numpy/tensors and uploading a whole frame to the GPU just to crop it is
    SLOWER than a CPU cv2 warp -- so we gate on the real frame device too.
    """
    if not getattr(runtime, "requested_gpu", False):
        return False
    for frame in frames:
        if frame is not None:
            return bool(torch.is_tensor(frame) and frame.is_cuda)
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_gpu_classifier_crop.py::test_frames_on_cuda_gate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix(inference): gate GPU classifier crop on requested_gpu (covers gpu_fast NVDEC)"
git add -A && git commit -m "fix(inference): gate GPU classifier crop on requested_gpu (covers gpu_fast NVDEC)"
git status
```

---

## Task 3: Strict classifier load-check covers `gpu_fast` + fix numpy-fallback channel order

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/cnn.py:44-53` (strict check)
- Modify: `src/hydra_suite/core/inference/stages/headtail.py:76-88` (strict check)
- Modify: `src/hydra_suite/core/identity/classification/backend.py:1218-1223` and `:1237-1243` (numpy fallback `input_is_bgr`)
- Test: `tests/test_gpu_classifier_crop.py:167` and `:192` (update the two strict-check tests)

**Interfaces:**
- Consumes: `runtime.cuda_mode: bool` (True on `gpu`/`gpu_fast` on a CUDA box; False on MPS/CPU), `backend.supports_cuda_forward() -> bool`, `ClassifierBackend.predict_batch(crops, input_is_bgr=...)`.
- Produces: `load_cnn_model` / `load_headtail_model` raise `RuntimeError` when `cuda_mode` is set and the classifier lacks a CUDA-native forward. `predict_batch_cuda`'s numpy fallback now forwards `input_is_bgr` so an RGB crop is never silently treated as BGR.

**Why:** with Task 2, `gpu_fast` NVDEC frames route through `predict_batch_cuda`. If the classifier lacks a CUDA forward, `predict_batch_cuda` currently falls back to `self.predict_batch(numpy_crops)` **without** `input_is_bgr` (backend.py:1219-1223, 1239-1243), which defaults `input_is_bgr=True` — channel-swapping the RGB NVDEC crops. Two independent defenses: (a) extend the strict load-check from `tensor_on_cuda` (gpu-native only) to `cuda_mode` (both GPU tiers on CUDA) so an incapable classifier fails loudly at load; (b) fix the fallback to forward `input_is_bgr` so it is correct even if reached. On MPS/CPU `cuda_mode` is False, so neither GPU tier there is affected.

- [ ] **Step 1: Update the two strict-check tests**

In `tests/test_gpu_classifier_crop.py`, change `test_load_cnn_strict_raises_without_cuda_forward` (`:167`): replace its runtime stub line

```python
    rt = type("RT", (), {"tensor_on_cuda": True})()
```

with

```python
    rt = type("RT", (), {"cuda_mode": True})()
```

Then rename `test_load_cnn_no_raise_when_not_tensor_on_cuda` (`:192`) to `test_load_cnn_no_raise_when_not_cuda_mode`, update its docstring to "On MPS/CPU (cuda_mode False), a non-CUDA classifier loads fine.", and change its stub line

```python
    rt = type("RT", (), {"tensor_on_cuda": False})()
```

to

```python
    rt = type("RT", (), {"cuda_mode": False})()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_gpu_classifier_crop.py::test_load_cnn_strict_raises_without_cuda_forward tests/test_gpu_classifier_crop.py::test_load_cnn_no_raise_when_not_cuda_mode -v`
Expected: `test_load_cnn_strict_raises_without_cuda_forward` FAILS (`cnn.py` still reads `tensor_on_cuda`; the `cuda_mode=True` stub has no `tensor_on_cuda`, so `getattr(..., False)` → the strict check does not fire → no `RuntimeError` raised → test fails).

- [ ] **Step 3: Extend the CNN strict check**

In `src/hydra_suite/core/inference/stages/cnn.py`, change the condition (`:44-47`) and message (`:49-53`):

```python
    if getattr(runtime, "cuda_mode", False) and not backend.supports_cuda_forward():
        backend.close()
        raise RuntimeError(
            f"CNN classifier {config.model_path!r} lacks a CUDA-native forward, "
            "but a GPU tier (gpu/gpu_fast) on CUDA routes NVDEC/GPU crops through "
            "predict_batch_cuda (no silent CPU fallback). Use a native-torch / "
            "ONNX classifier, or run on the cpu tier."
        )
```

- [ ] **Step 4: Extend the head/tail strict check**

In `src/hydra_suite/core/inference/stages/headtail.py`, change the analogous condition (`:76-88`). Replace:

```python
        getattr(runtime, "tensor_on_cuda", False)
        and not backend.supports_cuda_forward()
```

with:

```python
        getattr(runtime, "cuda_mode", False)
        and not backend.supports_cuda_forward()
```

and update the surrounding `raise RuntimeError(...)` message to match cnn.py's wording (mention "a GPU tier (gpu/gpu_fast) on CUDA ... predict_batch_cuda", not "the gpu tier with NVDEC"). Keep it a head/tail-specific message (reference the head/tail classifier path).

- [ ] **Step 5: Fix the numpy-fallback channel order**

In `src/hydra_suite/core/identity/classification/backend.py`, the factor-bundle fallback (`:1218-1223`), change:

```python
            return self.predict_batch(numpy_crops)
```

to:

```python
            return self.predict_batch(numpy_crops, input_is_bgr=input_is_bgr)
```

And the "unknown backend" fallback (`:1237-1243`), change the same call the same way:

```python
                return self.predict_batch(numpy_crops, input_is_bgr=input_is_bgr)
```

(Both currently drop `input_is_bgr`, defaulting it to `True` and channel-swapping RGB crops.)

- [ ] **Step 6: Add a regression test for the fallback channel order**

In `tests/test_gpu_classifier_crop.py`, add:

```python
def test_predict_batch_cuda_fallback_forwards_input_is_bgr(monkeypatch):
    """When a factor lacks CUDA forward, the numpy fallback must NOT re-flip RGB."""
    from hydra_suite.core.identity.classification import backend as bk

    be = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    be._model_path = "x.multihead.json"
    seen = {}

    monkeypatch.setattr(be, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(be, "_uses_factor_backends", lambda: True)
    monkeypatch.setattr(be, "supports_cuda_forward", lambda: False)  # force fallback

    def _fake_predict_batch(crops, input_is_bgr=True):
        seen["input_is_bgr"] = input_is_bgr
        return [[np.array([1.0], np.float32)]]

    monkeypatch.setattr(be, "predict_batch", _fake_predict_batch)

    crop = torch.zeros((3, 4, 4))
    be.predict_batch_cuda([crop], input_is_bgr=False)
    assert seen["input_is_bgr"] is False   # RGB stays RGB through the fallback
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_gpu_classifier_crop.py -k "strict_raises or not_cuda_mode or fallback_forwards" -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "fix(identity): strict CUDA-forward check on both GPU tiers; fallback keeps RGB"
git add -A && git commit -m "fix(identity): strict CUDA-forward check on both GPU tiers; fallback keeps RGB"
git status
```

---

## Task 4: Lock in OBB TensorRT consuming CUDA HWC frames on `gpu_fast`

**Files:**
- Test: `tests/test_nvdec_gpu_fast_tier.py` (add)
- (No production change expected — obb.py:415-419 already routes `DirectExecutorAdapter` through the frames-list path, which accepts a raw list of CUDA HWC frames and does its own letterbox.)

**Interfaces:**
- Consumes: `hydra_suite.core.inference.stages.obb._run_direct`, `hydra_suite.core.inference.runtime_artifacts.DirectExecutorAdapter`.
- Produces: a regression test asserting that on a `gpu_fast` context (`tensor_on_cuda=False`) with CUDA-tensor frames + a `DirectExecutorAdapter` model, `_run_direct` does NOT call `_gpu_letterbox_batch` (it hands the raw frame list to `model.predict`) and returns `OBBResult`s (not raw tensors).

**Why:** spec §3 flags this as the one path to verify. Reading obb.py:405-455 shows it is already correct: the manual GPU-letterbox pre-batch is explicitly skipped for `DirectExecutorAdapter` (`and not isinstance(model, DirectExecutorAdapter)`), and `tensor_on_cuda=False` on `gpu_fast` routes the return through `extract_obb_result` (OBBResult), keeping the frame on-GPU for the downstream crop stage. This task locks that behavior with a test so a future refactor cannot silently break it.

- [ ] **Step 1: Write the test**

Add to `tests/test_nvdec_gpu_fast_tier.py`:

```python
def test_run_direct_gpu_fast_tensorrt_takes_frames_list_path(monkeypatch):
    """gpu_fast: a DirectExecutorAdapter (TensorRT) must receive the raw CUDA
    frame list (its predict does its own letterbox); the manual GPU-letterbox
    pre-batch must be skipped, and the return must be OBBResults (tensor_on_cuda
    False), not raw tensors."""
    from unittest.mock import MagicMock

    from hydra_suite.core.inference.runtime_artifacts import DirectExecutorAdapter
    from hydra_suite.core.inference.stages import obb as obb_stage

    # A frame that passes isinstance(_, torch.Tensor) and reports is_cuda True.
    fake_frame = MagicMock(spec=torch.Tensor)
    fake_frame.is_cuda = True

    model = MagicMock(spec=DirectExecutorAdapter)
    model.predict.return_value = ["raw_result_0"]

    letterbox_spy = {"called": False}
    monkeypatch.setattr(
        obb_stage,
        "_gpu_letterbox_batch",
        lambda *a, **k: letterbox_spy.__setitem__("called", True) or (None, []),
    )
    # Bypass real Ultralytics Results parsing.
    monkeypatch.setattr(obb_stage, "extract_obb_result", lambda r, idx: f"obb::{r}")
    monkeypatch.setattr(obb_stage, "_apply_raw_detection_cap", lambda res, cap: res)

    cfg = type(
        "C",
        (),
        {"target_classes": None, "raw_detection_cap": 0,
         "direct": type("D", (), {"confidence_floor": 1e-3})()},
    )()
    rt = type("RT", (), {"tensor_on_cuda": False, "device": "cuda"})()

    out = obb_stage._run_direct([fake_frame], model, cfg, rt)

    assert letterbox_spy["called"] is False        # frames-list path, no pre-batch
    passed_frames = model.predict.call_args.args[0]
    assert passed_frames == [fake_frame]            # raw CUDA frame list handed through
    assert out == ["obb::raw_result_0"]             # OBBResult path, not raw tensors
```

- [ ] **Step 2: Run test to verify it passes as-is (verification, not TDD)**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_nvdec_gpu_fast_tier.py::test_run_direct_gpu_fast_tensorrt_takes_frames_list_path -v`
Expected: PASS **without any production change** — this locks in existing correct behavior. If it FAILS, the routing is not what the spec assumed: STOP and reconcile (do not force the test green by weakening it); the divergence is a real finding to report before proceeding.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test(inference): lock gpu_fast TensorRT OBB consuming CUDA HWC frame list"
git add -A && git commit -m "test(inference): lock gpu_fast TensorRT OBB consuming CUDA HWC frame list"
git status
```

---

## Task 5: Log which frame reader engaged

**Files:**
- Modify: `src/hydra_suite/core/inference/sources.py:475-492` (`make_frame_source`)
- Test: `tests/test_nvdec_gpu_fast_tier.py` (add)

**Interfaces:**
- Consumes: `runtime.use_nvdec`, existing `logger`.
- Produces: `make_frame_source` emits an `info`-level line naming the reader actually selected (NvdecFrameReader vs CpuFrameReader) so the Task 6 acceptance check ("NVDEC actually engaged, no fallback") is visible in normal logs.

**Why:** spec acceptance requires confirming NVDEC engaged (no silent `CpuFrameReader` fallback) for a `gpu_fast` HEVC clip. The success path currently logs at `debug`; the fallback logs a `warning`. Promote the success log to `info` so both branches are visible at default verbosity.

- [ ] **Step 1: Write the test**

Add to `tests/test_nvdec_gpu_fast_tier.py`:

```python
def test_make_frame_source_cpu_reader_when_nvdec_off(monkeypatch, tmp_path, caplog):
    """use_nvdec False -> CpuFrameReader, no NVDEC construction attempted."""
    import logging

    from hydra_suite.core.inference import sources as src

    monkeypatch.setattr(
        src, "NvdecFrameReader",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not construct NVDEC")),
    )
    rt = type("RT", (), {"use_nvdec": False})()
    fake_video = str(tmp_path / "clip.mp4")

    with caplog.at_level(logging.INFO):
        reader = src.make_frame_source(fake_video, rt, start_frame=0, end_frame=1)
    assert isinstance(reader, src.CpuFrameReader)
    assert any("CpuFrameReader" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_nvdec_gpu_fast_tier.py::test_make_frame_source_cpu_reader_when_nvdec_off -v`
Expected: FAIL — no `info` line naming `CpuFrameReader` is currently emitted on the `use_nvdec=False` path.

- [ ] **Step 3: Add the info logs**

In `src/hydra_suite/core/inference/sources.py`, in `make_frame_source`:
- On the successful NVDEC branch (after `reader = NvdecFrameReader(...)` succeeds, near `:483`), change the existing `logger.debug("make_frame_source: using NvdecFrameReader for %s", ...)` to `logger.info(...)`.
- On the final `return CpuFrameReader(...)` (`:492`), add immediately before it:

```python
    logger.info("make_frame_source: using CpuFrameReader for %s", video_path)
    return CpuFrameReader(video_path, start_frame=start_frame, end_frame=end_frame)
```

(Leave the existing fallback `warning` when NVDEC was requested but unavailable — it is complementary.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_nvdec_gpu_fast_tier.py::test_make_frame_source_cpu_reader_when_nvdec_off -v`
Expected: PASS.

- [ ] **Step 5: Full suite sanity + commit**

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_nvdec_gpu_fast_tier.py tests/test_gpu_classifier_crop.py -v
git add -A && git commit -m "feat(inference): log the frame reader selected (NVDEC vs CPU) at info level"
git add -A && git commit -m "feat(inference): log the frame reader selected (NVDEC vs CPU) at info level"
git status
```

---

## Task 6: Hardware verification (mehek CUDA) + MPS byte-identity (local)

**Files:** none (verification only). This is the real correctness gate; the unit tests above run without a GPU.

**Goal:** prove the three acceptance criteria from the spec: (1) **GPU tier byte-identical** on CUDA (NVDEC no longer engages there); (2) **GPU-Fast NVDEC end-to-end** on an NVDEC-decodable clip (determinism + NVDEC engaged + perf); (3) **MPS/CPU unchanged**.

- [ ] **Step 1: Local MPS byte-identity** (this Mac)

The equivalence harness proves nothing changed on MPS (NVDEC is CUDA-only; these edits are gated on `cuda_mode`/`requested_gpu`, but confirm no accidental MPS regression). Use the current worktree as both trees is not meaningful (baseline must be pre-change) — instead compare `HEAD` against the commit before Task 1 on MPS:

```bash
conda activate hydra-mps
# baseline worktree at the pre-change commit (the tip before Task 1):
git worktree add --detach .worktrees/equiv-prechange <sha-before-task-1>
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-prechange/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_nvdec_mps RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub ant_cnn_identity
git worktree remove --force .worktrees/equiv-prechange && git worktree prune
```

Expected: EQUIVALENCE at the DETERMINISM floor (pos p99 ≈ 0, θ max ≈ 0, identical row counts) for every clip. **Verify `wc -l` on the CSVs > 1** before trusting an EQUIVALENT (empty CSVs falsely pass — conda must be active for SLEAP clips).

- [ ] **Step 2: Push the branch so mehek can fetch it**

```bash
git push origin feature/gpu-native-classifier-crop
```

- [ ] **Step 3: mehek — GPU-tier byte-identity vs `main`** (the primary gate)

On mehek, the `gpu` tier must now CPU-decode (no NVDEC), so it stays byte-identical to `main`. Compare current branch (`gpu` tier) against `main` (`gpu` tier), same clips:

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
export KMP_DUPLICATE_LIB_OK=TRUE
# baseline = main; current = feature branch tip
git worktree add --detach .worktrees/equiv-main origin/main
git worktree add --detach .worktrees/equiv-feat origin/feature/gpu-native-classifier-crop
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-main/src WT_SRC=$PWD/.worktrees/equiv-feat/src \
  OUT=/tmp/equiv_nvdec_gpu RUNTIME=gpu \
  nohup bash tools/equivalence/run_matrix.sh ant_cnn_identity ant_obb_sleap fly_obb > /tmp/equiv_gpu.log 2>&1 &
```

Expected: **byte-identical** (pos/θ max 0, identical row counts) on the `gpu` tier for every clip — confirms NVDEC no longer engages on `gpu`. Grep `/tmp/equiv_gpu.log` for `make_frame_source: using` and confirm **CpuFrameReader** on the `gpu` tier. If any clip diverges on `gpu`, that is a **release-blocking regression** — stop and report.

- [ ] **Step 4: mehek — GPU-Fast NVDEC end-to-end**

Produce an NVDEC-decodable clip (H.264 4512² exceeds the 4096 limit; transcode `ant_cnn_identity` to lossless HEVC, which allows up to 8192²). Then run it twice on `gpu_fast` to check determinism, and confirm NVDEC actually engaged:

```bash
# one-time: lossless HEVC transcode of the fixture
mkdir -p /tmp/hevc_test
ffmpeg -y -i tools/equivalence/fixtures/clips/ant_cnn_identity.mp4 \
  -c:v libx265 -x265-params lossless=1 /tmp/hevc_test/ant_cnn_identity_hevc.mp4

# run gpu_fast twice (determinism) on the HEVC clip; confirm NVDEC engaged.
# (Use runner.py directly on the HEVC clip with RUNTIME=gpu_fast, or the matrix if
#  the fixture set is pointed at the HEVC clip. Capture the log.)
```

Expected:
- The run log shows `make_frame_source: using NvdecFrameReader` for the HEVC clip (NVDEC engaged, **not** a fallback warning).
- `new_a == new_b` (determinism): two `gpu_fast` runs of the same clip produce identical CSVs.
- Non-empty output (`wc -l` > 1).
- Measure and record the decode/wall-clock speedup vs the same clip on the `gpu` tier (CPU decode). The colortag identity residual vs CPU decode is **reported, not gated** (Part 2 closes it) — record the number, do not fail on it.

- [ ] **Step 5: Cleanup + record results**

```bash
# on mehek
git worktree remove --force .worktrees/equiv-main
git worktree remove --force .worktrees/equiv-feat && git worktree prune
```

Record the outcomes (gpu byte-identity PASS/FAIL, gpu_fast NVDEC-engaged + determinism + speedup, MPS byte-identity) in the PR description and update memory `project_runtime_gen2_core_done` with the verification numbers.

- [ ] **Step 6: Final commit (if any doc/memory updates)**

```bash
git add -A && git commit -m "docs: record NVDEC gpu_fast-tier verification results"
git add -A && git commit -m "docs: record NVDEC gpu_fast-tier verification results"
git status
```

---

## Self-Review

**Spec coverage:**
- Spec §1 (gate NVDEC on tier) → **Task 1**.
- Spec §2 (decouple frame-on-GPU from native-torch for crop routing) → **Task 2**; the `predict_batch_cuda` coverage claim + strict-check guarantee → **Task 3**.
- Spec §3 (confirm OBB/pose stages consume NVDEC frames on gpu_fast) → **Task 4** (verified already-correct + locked with a test).
- Spec §4 (color handling already implemented, keep) → no change; exercised end-to-end in **Task 6 Step 4**.
- Spec Acceptance (GPU byte-identical; GPU-Fast NVDEC determinism + engaged + perf; MPS/CPU unchanged) → **Task 6**.
- Spec Testing (unit: `use_nvdec` gating, `frames_on_cuda`; integration: mehek) → **Tasks 1, 2, 6**.
- Spec Risk "log which reader was used" → **Task 5**.

**Placeholder scan:** none — every code step shows the exact edit; the one intentionally open value is `<sha-before-task-1>` in Task 6 Step 1 (resolved at execution time from the branch log).

**Type consistency:** `frames_on_cuda(runtime, frames) -> bool`, `_should_use_nvdec(tier: str) -> bool`, `supports_cuda_forward() -> bool`, `predict_batch(crops, input_is_bgr=...)`, `predict_batch_cuda(crops, input_is_bgr=...)` — names/signatures match the live code read during planning. Runtime flags used: `use_nvdec`, `requested_gpu`, `cuda_mode`, `tensor_on_cuda` — all existing `RuntimeContext` fields.

**Note on Task 3 breadth:** extending the strict check from `tensor_on_cuda` to `cuda_mode` also makes it fire on the `gpu` tier (which now CPU-decodes and no longer strictly needs a CUDA forward). This is not a regression — the check already fired on `gpu` before (via `tensor_on_cuda`, which is True there), and real `gpu`-tier classifiers (native torch / ONNX) satisfy `supports_cuda_forward()`. The change only *adds* `gpu_fast` to the enforced set, per spec §2.
