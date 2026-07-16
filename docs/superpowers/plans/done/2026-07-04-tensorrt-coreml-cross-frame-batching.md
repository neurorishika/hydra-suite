# TensorRT + CoreML Cross-Frame Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OBB detection (direct + sequential) genuinely batch across frames on the `gpu_fast` tier by switching a TensorRT OBB artifact between a static batch=1 engine (realtime / batch_size==1) and a dynamic-batch engine (batch_size>=2), fix the CoreML classifier's per-crop-loop throughput bug, and surface CoreML OBB's permanent batch=1 limitation in the GUI.

**Architecture:** `runtime_artifacts.py`'s existing export/cache machinery already has all the plumbing needed (`_export_artifact`, `_artifact_path_for`, freshness markers) — the change is to make the requested batch size flow into that machinery and flip one `dynamic=` flag. `_direct_obb_runtime.py`'s `DirectTensorRTOBBExecutor` requires **zero changes**: it already detects a dynamic-profile engine (`engine.get_tensor_shape(...)` returns a negative batch dim) and already calls `context.set_input_shape(...)` on every inference call — this generic mechanism was built for ONNX dynamic axes and works identically for TensorRT. CoreML's classifier export already supports batching (`RangeDim(1, 512)`); only the call site loops needlessly.

**Tech Stack:** Python, PyTorch, TensorRT 10.x (Python API), ultralytics `YOLO.export()`, onnxruntime, coremltools, PySide6 (GUI), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-03-tensorrt-coreml-cross-frame-batching-design.md` (this plan implements its "Decision" section, dated 2026-07-04).
- Branch: `feature/inference-pipeline-redesign`. All commits land there.
- TensorRT OBB routing rule (from spec decision): batch_size == 1 (realtime, or any stage configured for single-frame windows) → static batch=1 engine (unchanged today's behavior). batch_size >= 2 → a single dynamic-profile engine (min=1, opt=batch_size, max=batch_size) used for all calls from that executor instance, including any undersized final chunk.
- CoreML OBB stays permanently batch=1 — do not attempt dynamic CoreML OBB export (Phase A/B proved this hard-crashes at CoreML compile time: `E5RT: TopK k=300 not within range [1,21]`).
- CoreML classifier batching: fix the call site only. Do not change `export_tiny_to_coreml`/`export_torchvision_to_coreml` — they already export with `RangeDim(1, 512)`.
- No changes to `core/detectors/` (legacy) — out of scope per spec.
- Every task must leave `make pytest` green before moving to the next task.

---

### Task 0: Fix pre-existing broken test fixtures (prerequisite)

A sanity run of the existing suite (`python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py -v`) on 2026-07-04 found 4 pre-existing failures, unrelated to batching, that must be fixed first since Task 1's new tests reuse the same `fake_loader` fixture:

- `_create_direct_executor` is called with a `task=task` kwarg at `runtime_artifacts.py:600` and `:566`, but `tests/test_inference_obb_artifacts.py`'s `fake_executor_factory` (in the `fake_loader` fixture) and `tests/test_inference_stages_obb.py`'s local `fake_executor` (in `test_load_yolo_routes_onnx_trt_to_direct_executor`) don't accept it — `TypeError: got an unexpected keyword argument 'task'`. Breaks `test_tensorrt_auto_export_triggers_export_exactly_once`, `test_tensorrt_existing_engine_skips_export`, `test_explicit_engine_path_used_directly`.
- `test_load_yolo_routes_onnx_trt_to_direct_executor` asserts `_load_yolo(str(onnx), "onnx_cuda", auto_export=False)` routes to a direct executor — this behavior was removed when ONNX support for OBB was dropped (see `runtime_artifacts.py`'s module docstring: "onnx_* runtimes are NOT supported for OBB"). The correct behavior (raising `ArtifactExportError`) is already covered by the passing `test_onnx_cuda_raises_unsupported` in `test_inference_obb_artifacts.py`. This test is obsolete and must be deleted, not fixed.

**Files:**
- Modify: `tests/test_inference_obb_artifacts.py` (`fake_loader` fixture's `fake_executor_factory`)
- Modify: `tests/test_inference_stages_obb.py` (delete `test_load_yolo_routes_onnx_trt_to_direct_executor`)

- [ ] **Step 1: Confirm the failures exist**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py -v`
Expected: 4 FAILED (as listed above), rest PASS.

- [ ] **Step 2: Fix `fake_executor_factory` in `test_inference_obb_artifacts.py`**

Replace, in the `fake_loader` fixture:

```python
    def fake_executor_factory(*, runtime, artifact_path, imgsz, class_names=None):
        counters["executor"] += 1
        return _FakeExecutor(runtime, str(artifact_path), int(imgsz))
```

with:

```python
    def fake_executor_factory(
        *, runtime, artifact_path, imgsz, class_names=None, task="obb"
    ):
        counters["executor"] += 1
        return _FakeExecutor(runtime, str(artifact_path), int(imgsz))
```

- [ ] **Step 3: Delete the obsolete test in `test_inference_stages_obb.py`**

Remove the entire `test_load_yolo_routes_onnx_trt_to_direct_executor` function (currently lines 225-251) — its assertion (`onnx_cuda`/`onnx_coreml` route to a direct executor) contradicts the current, intentional, already-tested behavior that ONNX runtimes raise `ArtifactExportError` for OBB.

- [ ] **Step 4: Run tests to verify all 4 failures are resolved**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py -v`
Expected: all PASS (0 failed).

- [ ] **Step 5: Commit**

```bash
cd .worktrees/inference-pipeline-redesign
git add tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py
git commit -m "test(inference): fix stale OBB-artifact test fixtures (missing task kwarg, obsolete onnx_cuda test)"
```

---

### Task 1: TensorRT artifact export — dynamic vs. static batch selection

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py:123-200` (`_export_artifact`), `:276-283` (`_artifact_path_for`), `:537-607` (`_load_direct_executor`), `:395-468` (`load_obb_executor`)
- Test: `tests/test_inference_obb_artifacts.py`

**Interfaces:**
- Produces: `_artifact_path_for(pt_path: Path, runtime: str, batch_size: int = 1) -> Path` (new `batch_size` kwarg, default preserves today's `_b1.engine` naming)
- Produces: `load_obb_executor(model_path: str, compute_runtime: str, *, auto_export: bool = True, max_det: int = 20, imgsz_override: int | None = None, task: str = "obb", batch_size: int = 1) -> Any` (new `batch_size` kwarg)
- Consumes: `_export_artifact(*, pt_path, artifact_path, runtime, imgsz, batch_size)` (signature unchanged — only its internal `dynamic=` value changes)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_inference_obb_artifacts.py` (append after the existing `test_explicit_onnx_path_with_onnx_cpu_raises_unsupported` test, before the "Adapter wrapping" section comment):

```python
def test_artifact_path_embeds_requested_batch_size(tmp_path):
    """Different batch sizes must produce different cached artifact filenames,
    so a workflow requesting batch=8 never reuses a batch=1 (or batch=16)
    cached engine."""
    pt = tmp_path / "model.pt"
    assert ra._artifact_path_for(pt, "tensorrt", batch_size=1).name == "model_b1.engine"
    assert ra._artifact_path_for(pt, "tensorrt", batch_size=8).name == "model_b8.engine"
    # Default (no batch_size passed) preserves today's behaviour exactly.
    assert ra._artifact_path_for(pt, "tensorrt").name == "model_b1.engine"


def test_tensorrt_export_uses_dynamic_profile_when_batch_size_gt_one(
    tmp_path, monkeypatch
):
    """batch_size > 1 must export with dynamic=True (a real optimization
    profile covering 1..batch_size); batch_size == 1 must stay dynamic=False
    (today's static engine) -- this is the routing rule from Spec 1's
    Phase A/B decision (2026-07-04): realtime/batch=1 uses the un-taxed
    static engine, batch>=2 uses the dynamic engine."""
    import sys
    import types

    captured: dict = {}

    class _FakeExportYOLO:
        def __init__(self, path):
            self.path = path
            self.model = types.SimpleNamespace(
                model=[types.SimpleNamespace(end2end=False)]
            )

        def export(self, **kwargs):
            captured.update(kwargs)
            out = tmp_path / "exported.engine"
            out.write_bytes(b"fake-engine")
            return str(out)

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = _FakeExportYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setattr(ra, "_create_direct_executor", lambda **kw: object())

    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=8)
    assert captured["dynamic"] is True
    assert captured["batch"] == 8

    captured.clear()
    pt2 = tmp_path / "model2.pt"
    pt2.write_bytes(b"x")
    load_obb_executor(str(pt2), "tensorrt", auto_export=True, batch_size=1)
    assert captured["dynamic"] is False
    assert captured["batch"] == 1


def test_tensorrt_batch_size_two_and_eight_export_separate_cached_artifacts(
    fake_loader, tmp_path
):
    """Requesting batch_size=8 then batch_size=1 for the same .pt must export
    TWICE (two distinct cached files), not reuse/clobber one artifact."""
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")

    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=8)
    assert fake_loader["export"] == 1
    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=1)
    assert fake_loader["export"] == 2  # different artifact path -> re-export, not reuse

    # Requesting batch_size=8 again reuses the now-cached batch=8 artifact.
    load_obb_executor(str(pt), "tensorrt", auto_export=True, batch_size=8)
    assert fake_loader["export"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_inference_obb_artifacts.py -v -k "batch_size or dynamic_profile"`
Expected: FAIL — `_artifact_path_for() got an unexpected keyword argument 'batch_size'` (or similar `TypeError`), and `load_obb_executor() got an unexpected keyword argument 'batch_size'`.

- [ ] **Step 3: Implement `_artifact_path_for` with a `batch_size` parameter**

Replace the function at `runtime_artifacts.py:276-283`:

```python
def _artifact_path_for(
    pt_path: Path, runtime: str, batch_size: int = _DEFAULT_BATCH_SIZE
) -> Path:
    """Derive the artifact path for a ``.pt`` source + direct runtime name.

    ``batch_size`` is embedded in the filename (``_b1``, ``_b8``, ...) so a
    workflow requesting a different batch size never reuses a wrong-shaped
    cached engine. ``batch_size == 1`` exports a static batch=1 engine
    (unchanged from before); ``batch_size > 1`` exports a TensorRT engine
    with a dynamic batch profile (min=1, opt=batch_size, max=batch_size --
    see ``_export_artifact``).
    """
    pt_path = Path(pt_path)
    suffix = _artifact_suffix(runtime)
    if runtime == "coreml":
        # CoreML uses a bare stem (no batch suffix): OBB stays batch=1 on
        # CoreML permanently -- ultralytics' CoreML export hard-crashes at
        # compile time when both the batch and spatial dims are made
        # dynamic together for an OBB model (Spec 1 Phase A/B, 2026-07-04)
        # -- so there is only ever one CoreML OBB artifact.
        return pt_path.with_suffix(".mlpackage")
    return pt_path.with_name(f"{pt_path.stem}_b{int(batch_size)}{suffix}")
```

- [ ] **Step 4: Implement the `dynamic=` switch in `_export_artifact`**

Replace the `if runtime == "tensorrt":` block at `runtime_artifacts.py:150-164`:

```python
    if runtime == "tensorrt":
        dynamic = int(batch_size) > 1
        logger.info(
            "Building TensorRT OBB engine (imgsz=%d, batch=%d, dynamic=%s) — "
            "one-time export...",
            imgsz,
            batch_size,
            dynamic,
        )
        export_path = base_model.export(
            format="engine",
            device="cuda:0",
            half=True,
            dynamic=dynamic,
            batch=int(batch_size),
            imgsz=imgsz,
            verbose=False,
        )
```

- [ ] **Step 5: Thread `batch_size` through `_load_direct_executor` and `load_obb_executor`**

Replace `_load_direct_executor` at `runtime_artifacts.py:537-607`:

```python
def _load_direct_executor(
    model_path: str,
    compute_runtime: str,
    *,
    auto_export: bool,
    max_det: int,
    imgsz_override: int | None = None,
    task: str = "obb",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> DirectExecutorAdapter:
    """Resolve (or auto-export) an ONNX/TRT artifact and wrap a direct executor."""
    runtime = _direct_runtime_name(compute_runtime)
    resolved = Path(model_path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    # 1) Explicit artifact path supplied by the user: use as-is.
    if suffix in {".onnx", ".engine", ".trt"}:
        if not resolved.exists():
            raise ArtifactExportError(
                f"{runtime} artifact not found: {resolved}. "
                f"Provide a valid {_artifact_suffix(runtime)} file or use a .pt "
                f"source with auto_export=True."
            )
        imgsz = _DEFAULT_IMGSZ
        executor = _create_direct_executor(
            runtime=runtime,
            artifact_path=resolved,
            imgsz=imgsz,
            class_names=None,
            task=task,
        )
        return DirectExecutorAdapter(executor, max_det=max_det)

    # 2) Source .pt path: locate (or build) the derived artifact.
    artifact_path = _artifact_path_for(resolved, runtime, batch_size=batch_size)
    imgsz = (
        int(imgsz_override)
        if imgsz_override and imgsz_override > 0
        else _resolve_imgsz(resolved)
    )

    if _artifact_is_fresh(artifact_path, resolved, imgsz):
        logger.info("Reusing cached %s OBB artifact: %s", runtime, artifact_path.name)
    else:
        if not auto_export:
            raise ArtifactExportError(
                f"compute_runtime={compute_runtime!r} requested but no fresh "
                f"{_artifact_suffix(runtime)} artifact exists for {resolved.name} "
                f"and auto_export=False. Provide a prebuilt "
                f"{_artifact_suffix(runtime)} (point model_path at it) or enable "
                f"auto_export (CUDA box) — refusing to silently fall back to "
                f"PyTorch (H4)."
            )
        _export_artifact(
            pt_path=resolved,
            artifact_path=artifact_path,
            runtime=runtime,
            imgsz=imgsz,
            batch_size=batch_size,
        )
        _write_fresh_marker(artifact_path, resolved, imgsz)
        logger.info("Exported %s OBB artifact: %s", runtime, artifact_path)

    class_names = _model_class_names(resolved)
    executor = _create_direct_executor(
        runtime=runtime,
        artifact_path=artifact_path,
        imgsz=imgsz,
        class_names=class_names,
        task=task,
    )
    return DirectExecutorAdapter(executor, max_det=max_det)
```

Then in `load_obb_executor` (`runtime_artifacts.py:395-468`), add the `batch_size` parameter to the signature and thread it only into the TensorRT branch. Replace:

```python
def load_obb_executor(
    model_path: str,
    compute_runtime: str,
    *,
    auto_export: bool = True,
    max_det: int = _DEFAULT_MAX_DET,
    imgsz_override: int | None = None,
    task: str = "obb",
) -> Any:
```

with:

```python
def load_obb_executor(
    model_path: str,
    compute_runtime: str,
    *,
    auto_export: bool = True,
    max_det: int = _DEFAULT_MAX_DET,
    imgsz_override: int | None = None,
    task: str = "obb",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Any:
```

and add `batch_size` to the docstring's `Parameters` section (insert after the `task` entry):

```
    batch_size:
        The number of frames/crops this executor will typically be called
        with per ``predict()`` call. ``1`` (default) exports/loads a static
        batch=1 TensorRT engine (unchanged from before). ``>1`` exports a
        TensorRT engine with a dynamic batch profile (min=1, opt=batch_size,
        max=batch_size) so a single engine handles the whole configured
        window in one inference call. Ignored for cpu/mps/cuda (torch
        already batches natively) and for coreml (OBB stays batch=1
        permanently on CoreML -- see ``_load_coreml_executor``).
```

and update the `if runtime in _TENSORRT_RUNTIMES:` branch to pass it through:

```python
    if runtime in _TENSORRT_RUNTIMES:
        return _load_direct_executor(
            model_path,
            runtime,
            auto_export=auto_export,
            max_det=max_det,
            imgsz_override=imgsz_override,
            task=task,
            batch_size=batch_size,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_inference_obb_artifacts.py -v`
Expected: all tests PASS, including the 3 new ones.

- [ ] **Step 7: Run the full test file plus a broader sanity sweep**

Run: `python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py -v`
Expected: all PASS (no regression in existing selection-logic or stage tests).

- [ ] **Step 8: Commit**

```bash
cd .worktrees/inference-pipeline-redesign
git add src/hydra_suite/core/inference/runtime_artifacts.py tests/test_inference_obb_artifacts.py
git commit -m "feat(inference): export a dynamic-batch TensorRT OBB engine when batch_size>1"
```

---

### Task 2: Thread the configured batch size from `InferenceConfig` into OBB model loading

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py:284-328` (`load_obb_models`), `:687-755` (`_load_yolo`)
- Modify: `src/hydra_suite/core/inference/runner.py:130` (`_load_all_models`)
- Test: `tests/test_inference_stages_obb.py`

**Interfaces:**
- Consumes: `load_obb_executor(..., batch_size: int = 1)` from Task 1
- Produces: `load_obb_models(config: OBBConfig, runtime: RuntimeContext, *, batch_size: int = 1) -> OBBModels` (new `batch_size` kwarg)
- Produces: `_load_yolo(model_path, compute_runtime, *, auto_export=True, max_det=20, imgsz_override=None, task="obb", batch_size: int = 1) -> Any` (new `batch_size` kwarg)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_inference_stages_obb.py` (near the other `_load_yolo`/`load_obb_models` tests, after `test_load_yolo_calls_to_for_native_pt`):

```python
def test_load_yolo_forwards_batch_size_to_load_obb_executor(monkeypatch):
    """_load_yolo must forward its batch_size kwarg to load_obb_executor
    unchanged -- this is how the configured window size reaches the
    TensorRT dynamic-vs-static export decision (Task 1)."""
    import hydra_suite.core.inference.runtime_artifacts as ra_mod

    captured = {}

    def fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ra_mod, "load_obb_executor", fake_load_obb_executor)
    from hydra_suite.core.inference.stages.obb import _load_yolo

    _load_yolo("/m.pt", "tensorrt", auto_export=False, batch_size=8)
    assert captured["batch_size"] == 8


def test_load_obb_models_direct_mode_uses_detection_batch_size(monkeypatch):
    """Direct-mode OBB must be loaded with the caller's batch_size (the
    pipeline's configured detection_batch_size), not a hardcoded 1."""
    import hydra_suite.core.inference.stages.obb as obb_mod
    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.runtime import RuntimeContext

    captured = {}

    def fake_load_yolo(model_path, compute_runtime, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(obb_mod, "_load_yolo", fake_load_yolo)

    config = OBBConfig(
        mode="direct", direct=OBBDirectConfig(model_path="/m.pt")
    )
    runtime = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        default_runtime="tensorrt",
        tensor_on_cuda=False,
    )
    obb_mod.load_obb_models(config, runtime, batch_size=8)
    assert captured["batch_size"] == 8


def test_load_obb_models_sequential_mode_uses_stage2_batch_size_for_obb_model(
    monkeypatch,
):
    """Sequential mode: stage-1 (detect) uses the frame-window batch_size;
    stage-2 (obb/crop) uses OBBSequentialConfig.stage2_batch_size when set,
    falling back to the frame-window batch_size otherwise."""
    import hydra_suite.core.inference.stages.obb as obb_mod
    from hydra_suite.core.inference.config import OBBConfig, OBBSequentialConfig
    from hydra_suite.core.inference.runtime import RuntimeContext

    calls = []

    def fake_load_yolo(model_path, compute_runtime, **kwargs):
        calls.append((model_path, kwargs.get("batch_size")))
        return object()

    monkeypatch.setattr(obb_mod, "_load_yolo", fake_load_yolo)

    runtime = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        default_runtime="tensorrt",
        tensor_on_cuda=False,
    )

    # stage2_batch_size explicitly set -> obb model uses it, not batch_size.
    config = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/detect.pt",
            obb_model_path="/obb.pt",
            stage2_batch_size=16,
        ),
    )
    obb_mod.load_obb_models(config, runtime, batch_size=8)
    assert calls == [("/detect.pt", 8), ("/obb.pt", 16)]

    # stage2_batch_size unset (None) -> obb model falls back to batch_size.
    calls.clear()
    config2 = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/detect.pt", obb_model_path="/obb.pt"
        ),
    )
    obb_mod.load_obb_models(config2, runtime, batch_size=8)
    assert calls == [("/detect.pt", 8), ("/obb.pt", 8)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_inference_stages_obb.py -v -k "batch_size"`
Expected: FAIL — `load_obb_models() got an unexpected keyword argument 'batch_size'` / `captured["batch_size"]` raises `KeyError` (since `_load_yolo` doesn't forward it yet).

- [ ] **Step 3: Implement `_load_yolo`'s `batch_size` parameter**

Replace the function signature and body at `stages/obb.py:687-755` (the function currently reads as shown in the file; only the signature and the primary `load_obb_executor(...)` call change):

```python
def _load_yolo(
    model_path: str,
    compute_runtime: ComputeRuntime,
    *,
    auto_export: bool = True,
    max_det: int = 20,
    imgsz_override: int | None = None,
    task: str = "obb",
    batch_size: int = 1,
) -> Any:
    """Load the OBB executor for ``model_path`` under ``compute_runtime``.

    Thin delegator to :func:`load_obb_executor`:
      * cpu/mps/cuda → a plain ultralytics ``YOLO`` model (``.to()``-moved as
        before; CPU does not call ``.to()`` so CPU byte-parity is preserved).
      * tensorrt → a direct TensorRT executor (auto-exporting the ``.engine``
        from ``.pt`` on first load when ``auto_export``); coreml → the
        ``.mlpackage``. When no artifact exists and ``auto_export`` is False, a
        clear error is raised instead of silently running PyTorch (finding H4).

    ``imgsz_override``, when set, forces the ONNX/TRT export/load size instead
    of the checkpoint's own embedded default -- needed for the sequential-OBB
    stage-2 (crop) model, whose ``stage2_image_size`` config value may differ
    from the checkpoint's default (see :func:`load_obb_executor`). Ignored for
    the torch runtimes (cpu/mps/cuda), which take crops pre-resized by the
    caller and never re-export an artifact.

    ``task="detect"`` must be passed for the sequential pipeline's stage-1
    model under tensorrt/onnx (see :func:`load_obb_executor`) -- it is a plain
    detector, not an OBB model.

    ``batch_size`` is forwarded to :func:`load_obb_executor` and governs
    whether a TensorRT export uses a static batch=1 engine or a dynamic-batch
    engine (see Task 1) -- ignored for the torch runtimes and for coreml.

    For ``compute_runtime="tensorrt"`` (gpu_fast tier), if the TRT artifact is
    unavailable or the build fails, falls back to native ``"cuda"`` and logs a
    WARNING.  Never falls back to CPU — stays on GPU device.
    """
    try:
        return load_obb_executor(
            model_path,
            compute_runtime,
            auto_export=auto_export,
            max_det=max_det,
            imgsz_override=imgsz_override,
            task=task,
            batch_size=batch_size,
        )
    except (
        Exception
    ) as exc:  # best-effort GPU-Fast fallback (spec §3: never a hard crash)
        if str(compute_runtime) == "coreml":
            logger.warning(
                "GPU-Fast OBB CoreML load/build failed (%s); falling back to native MPS",
                exc,
            )
            return load_obb_executor(
                model_path,
                "mps",
                auto_export=auto_export,
                max_det=max_det,
            )
        if str(compute_runtime) != "tensorrt":
            raise
        logger.warning(
            "GPU-Fast OBB TensorRT load/build failed (%s); falling back to native CUDA",
            exc,
        )
        return load_obb_executor(
            model_path,
            "cuda",
            auto_export=auto_export,
            max_det=max_det,
        )
```

- [ ] **Step 4: Implement `load_obb_models`'s `batch_size` parameter**

Replace the function at `stages/obb.py:284-328`:

```python
def load_obb_models(
    config: OBBConfig, runtime: RuntimeContext, *, batch_size: int = 1
) -> OBBModels:
    # Derive backend from the RuntimeContext (which reflects runtime_tier via
    # from_config). Per-stage compute_runtime fields are deprecated in favor of
    # runtime_tier; they are kept in place for serialization only.
    compute_runtime = runtime_to_compute_runtime(runtime)
    if compute_runtime in ("tensorrt", "coreml"):
        logger.warning(
            "Runtime fallback may apply for OBB stage: "
            "gpu_fast (%s) requested — artifact availability governs actual backend.",
            compute_runtime,
        )
    if config.mode == "direct":
        assert config.direct is not None
        auto_export = config.direct.auto_export
        m = _load_yolo(
            config.direct.model_path,
            compute_runtime,
            auto_export=auto_export,
            max_det=config.max_detections,
            batch_size=batch_size,
        )
        return OBBModels(mode="direct", direct_model=m)
    assert config.sequential is not None
    auto_export = config.sequential.auto_export
    detect_imgsz = config.sequential.detect_image_size
    detect = _load_yolo(
        config.sequential.detect_model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=config.max_detections,
        imgsz_override=detect_imgsz if detect_imgsz > 0 else None,
        # Stage-1 is a plain detector (no angle head) -- must be parsed as
        # Results(boxes=...), not Results(obb=...), under tensorrt/onnx.
        task="detect",
        batch_size=batch_size,
    )
    # stage2_image_size is always the effective input size (the pipeline
    # pre-resizes every crop to it in _resize_crops_for_stage2), so the
    # artifact must be built at that size, not the checkpoint's own default.
    # stage2_batch_size, when set, is the number of crops stage-2 is called
    # with per chunk (see _run_sequential's `batch_size = seq.stage2_batch_size
    # or len(crops)`); falls back to the frame-window batch_size when unset so
    # the exported artifact still gets a dynamic profile sized reasonably.
    obb = _load_yolo(
        config.sequential.obb_model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=config.max_detections,
        imgsz_override=config.sequential.stage2_image_size,
        batch_size=config.sequential.stage2_batch_size or batch_size,
    )
    return OBBModels(mode="sequential", detect_model=detect, obb_model=obb)
```

- [ ] **Step 5: Thread `batch_size` from `_load_all_models` in `runner.py`**

Modify `runner.py:130`, changing:

```python
    obb = load_obb_models(config.obb, runtime)
```

to:

```python
    obb = load_obb_models(config.obb, runtime, batch_size=config.detection_batch_size)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_inference_stages_obb.py -v`
Expected: all PASS, including the 3 new tests.

- [ ] **Step 7: Run the broader inference test suite**

Run: `python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py tests/test_inference_batch_stages.py tests/test_gpu_fast_fallback.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
cd .worktrees/inference-pipeline-redesign
git add src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/runner.py tests/test_inference_stages_obb.py
git commit -m "feat(inference): thread detection_batch_size/stage2_batch_size into OBB model loading"
```

---

### Task 3: Fix CoreML classifier per-crop-loop throughput bug

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/backend.py:871-895` (`_forward_coreml`)
- Test: `tests/test_classifier_coreml_backend.py`

**Interfaces:**
- Produces: `ClassifierBackend._forward_coreml(self, batch_np: np.ndarray) -> np.ndarray` (same signature, now calls `predict()` once instead of N times)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_classifier_coreml_backend.py`, in the "Unit tests (no real model required)" section (after `test_backend_does_not_report_coreml_for_onnx`):

```python
def test_forward_coreml_calls_predict_once_for_whole_batch():
    """_forward_coreml must call predict() ONCE for the whole batch, not once
    per crop. Regression test for the throughput bug found in Spec 1 Phase
    A/B (2026-07-04): looping per-crop measured 7.8x slower per-frame at
    batch=32 than a single batched predict() call, even though the exported
    .mlpackage already supports batch via RangeDim(1, 512)."""
    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    be._coreml_output_name = "out"

    calls = []

    class _FakeCoreMLModel:
        def predict(self, feed):
            n = feed["input"].shape[0]
            calls.append(n)
            return {"out": np.zeros((n, 5), dtype=np.float32)}

    be._model = _FakeCoreMLModel()
    batch = np.random.rand(4, 3, 64, 128).astype(np.float32)
    out = be._forward_coreml(batch)

    assert calls == [4], (
        f"Expected exactly one predict() call covering the whole batch of 4, "
        f"got per-call batch sizes {calls}"
    )
    assert out.shape == (4, 5)


def test_forward_coreml_falls_back_to_positional_output_when_name_unset():
    """When _coreml_output_name is None (unresolved output name), fall back
    to the first value in the prediction dict, matching the old per-crop
    behaviour's fallback."""
    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    be._coreml_output_name = None

    class _FakeCoreMLModel:
        def predict(self, feed):
            n = feed["input"].shape[0]
            return {"var_23": np.ones((n, 3), dtype=np.float32)}

    be._model = _FakeCoreMLModel()
    batch = np.zeros((2, 3, 64, 128), dtype=np.float32)
    out = be._forward_coreml(batch)
    assert out.shape == (2, 3)
    assert np.all(out == 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_classifier_coreml_backend.py -v -k "calls_predict_once or falls_back_to_positional"`
Expected: FAIL — `calls == [1, 1, 1, 1]` (one call per crop) instead of `[4]`.

- [ ] **Step 3: Implement the batched `_forward_coreml`**

Replace the function at `backend.py:871-895`:

```python
    def _forward_coreml(self, batch_np: np.ndarray) -> np.ndarray:
        """Run a preprocessed (N, 3, H, W) float32 batch through the CoreML model.

        Calls predict() ONCE for the whole batch. The exported .mlpackage's
        input already uses a RangeDim(1, 512) batch axis (see
        export_tiny_to_coreml / export_torchvision_to_coreml) -- looping
        per-crop was a correctness-preserving but throughput-destroying bug:
        measured 7.8x slower per-frame at batch=32 than a single batched call
        (Spec 1 Phase A/B, 2026-07-04).

        The output feature name assigned by coremltools varies by model graph
        (e.g. ``'var_23'``). We therefore index the prediction dict by the
        cached name when known, falling back to the first value by position.

        The model was traced with an NCHW ``ct.TensorType(name="input", ...)``
        input, so we feed the preprocessed batch as-is in NCHW under the
        "input" key — no layout transpose is needed.
        """
        pred = self._model.predict({"input": batch_np})
        if (
            self._coreml_output_name is not None
            and self._coreml_output_name in pred
        ):
            logits = pred[self._coreml_output_name]
        else:
            logits = next(iter(pred.values()))
        return np.asarray(logits, dtype=np.float32).reshape(batch_np.shape[0], -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_classifier_coreml_backend.py -v`
Expected: all PASS. On this machine (Apple Silicon with `coremltools` installed and the real orientation model present at `~/Library/Application Support/hydra-suite/models/classification/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth`), the real end-to-end tests (`test_coreml_predict_batch_shape_and_probabilities`, `test_coreml_output_agrees_with_native_torch`, `test_coreml_peer_cached_on_second_load`) also run for real (not skipped) and must still PASS — they assert output shape/values/caching, which are unaffected by batching the call.

- [ ] **Step 5: Commit**

```bash
cd .worktrees/inference-pipeline-redesign
git add src/hydra_suite/core/identity/classification/backend.py tests/test_classifier_coreml_backend.py
git commit -m "fix(identity): batch CoreML classifier predict() calls instead of looping per-crop"
```

---

### Task 4: Surface CoreML OBB's permanent batch=1 limitation in the GUI

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/session.py:1027-1030` (`_runtime_requires_fixed_yolo_batch`, add a new helper after it)
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py:1272-1358` (`_sync_batch_policy_controls`)
- Test: new file `tests/test_session_gpu_fast_coreml_notice.py`

**Interfaces:**
- Produces: `SessionOrchestrator._gpu_fast_obb_is_coreml_only(self) -> bool`
- Consumes: `SessionOrchestrator._current_runtime_tier(self) -> str` (existing), `detect_platform()` from `hydra_suite.runtime.resolver` (already imported in `session.py`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_gpu_fast_coreml_notice.py`:

```python
"""Tests for the gpu_fast + CoreML-OBB-batch=1 UI-notice helper.

CoreML's OBB export cannot use a dynamic batch axis (Spec 1 Phase A/B,
2026-07-04: ultralytics' CoreML export hard-crashes at compile time for OBB
models when both the batch and spatial dims are dynamic together), so OBB
detection under gpu_fast on Apple Silicon is permanently batch=1 even though
CoreML classification (identity/head-tail/CNN) batches normally. These tests
verify the GUI helper that identifies this case so the batch-policy notice
can say so explicitly, distinct from the TensorRT/ONNX "fixed batch" message.
"""

from __future__ import annotations

import types

from hydra_suite.trackerkit.gui.orchestrators import session as session_mod


class _FakeMainWindow:
    """Minimal stand-in exposing only what SessionOrchestrator needs here."""


def _make_orchestrator(tier: str) -> session_mod.SessionOrchestrator:
    orch = session_mod.SessionOrchestrator.__new__(session_mod.SessionOrchestrator)
    orch._mw = _FakeMainWindow()
    orch._current_runtime_tier = lambda: tier  # type: ignore[method-assign]
    return orch


def test_gpu_fast_obb_is_coreml_only_true_on_apple_silicon(monkeypatch):
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    assert orch._gpu_fast_obb_is_coreml_only() is True


def test_gpu_fast_obb_is_coreml_only_false_on_cuda(monkeypatch):
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=False, has_cuda=True),
    )
    assert orch._gpu_fast_obb_is_coreml_only() is False


def test_gpu_fast_obb_is_coreml_only_false_when_tier_is_not_gpu_fast(monkeypatch):
    orch = _make_orchestrator("gpu")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    assert orch._gpu_fast_obb_is_coreml_only() is False


def test_runtime_requires_fixed_yolo_batch_true_for_apple_silicon_gpu_fast(
    monkeypatch,
):
    """The existing 'fixed batch' UI gate must also fire for Apple-Silicon
    gpu_fast (CoreML OBB), not just tensorrt/onnx, so the frame-batch
    controls get disabled there too."""
    orch = _make_orchestrator("gpu_fast")
    monkeypatch.setattr(
        session_mod,
        "detect_platform",
        lambda: types.SimpleNamespace(has_mps=True, has_cuda=False),
    )
    orch._selected_compute_runtime = lambda: "mps"  # type: ignore[method-assign]
    assert orch._runtime_requires_fixed_yolo_batch() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_session_gpu_fast_coreml_notice.py -v`
Expected: FAIL — `AttributeError: 'SessionOrchestrator' object has no attribute '_gpu_fast_obb_is_coreml_only'`.

- [ ] **Step 3: Implement `_gpu_fast_obb_is_coreml_only` and update `_runtime_requires_fixed_yolo_batch`**

Replace the method at `session.py:1027-1030`:

```python
    def _runtime_requires_fixed_yolo_batch(self, runtime=None) -> bool:
        """Return True when runtime mandates a fixed YOLO batch size."""
        rt = str(runtime or self._selected_compute_runtime() or "").strip().lower()
        if rt == "tensorrt" or rt.startswith("onnx"):
            return True
        return self._gpu_fast_obb_is_coreml_only()

    def _gpu_fast_obb_is_coreml_only(self) -> bool:
        """Return True when gpu_fast OBB detection will run on CoreML.

        ``_tier_to_compute_runtime("gpu_fast")`` reports "mps" on Apple
        Silicon (the GUI's per-tier compute_runtime label), but the OBB
        stage internally upgrades to a CoreML direct executor whenever the
        exported ``.mlpackage`` artifact is available (see
        ``core/inference/runtime.py:runtime_to_compute_runtime``). CoreML's
        OBB export cannot use a dynamic batch axis (Spec 1 Phase A/B,
        2026-07-04: ultralytics' CoreML export hard-crashes at compile time
        for OBB models when both the batch and spatial dims are dynamic
        together), so OBB detection under this path is permanently batch=1,
        even though CoreML classification (identity/head-tail/CNN) batches
        normally. This is a platform limitation, not a config choice.
        """
        if self._current_runtime_tier() != "gpu_fast":
            return False
        platform = detect_platform()
        return bool(platform.has_mps and not platform.has_cuda)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_session_gpu_fast_coreml_notice.py -v`
Expected: all PASS.

- [ ] **Step 5: Distinguish the notice message in `detection_panel.py`**

In `_sync_batch_policy_controls` at `detection_panel.py:1272-1358`, replace the `if fixed_runtime:` block (currently at line ~1346-1352):

```python
        if fixed_runtime:
            message = "The selected runtime uses a fixed exported batch. Manual batch size controls the non-realtime detector artifact size."
            if recommendation_text:
                message += "\n" + recommendation_text
            self.lbl_batch_policy_notice.setText(message)
            self.lbl_batch_policy_notice.setVisible(True)
```

with:

```python
        if fixed_runtime:
            if self._main_window._gpu_fast_obb_is_coreml_only():
                message = (
                    "On this platform, gpu_fast detection (OBB) runs on "
                    "CoreML, which supports only batch size 1 — one frame "
                    "at a time, regardless of this setting. CoreML "
                    "classification (identity/head-tail/CNN) is unaffected "
                    "and still batches normally."
                )
            else:
                message = "The selected runtime uses a fixed exported batch. Manual batch size controls the non-realtime detector artifact size."
            if recommendation_text:
                message += "\n" + recommendation_text
            self.lbl_batch_policy_notice.setText(message)
            self.lbl_batch_policy_notice.setVisible(True)
```

- [ ] **Step 6: Run the GUI test suite for regressions**

Run: `cd .worktrees/inference-pipeline-redesign && python -m pytest tests/test_main_window_config_persistence.py -v -k "batch"`
Expected: all PASS (existing TensorRT/ONNX batch-notice tests are unaffected since `_gpu_fast_obb_is_coreml_only()` returns `False` whenever `detect_platform()` reports CUDA or the tier isn't `gpu_fast`, which is the case in those tests' fixtures).

- [ ] **Step 7: Commit**

```bash
cd .worktrees/inference-pipeline-redesign
git add src/hydra_suite/trackerkit/gui/orchestrators/session.py src/hydra_suite/trackerkit/gui/panels/detection_panel.py tests/test_session_gpu_fast_coreml_notice.py
git commit -m "feat(trackerkit): surface CoreML OBB's permanent batch=1 limit in the batch-policy notice"
```

---

### Task 5: Real-hardware verification (equivalence + throughput, not synthetic)

This task has no new pytest file — it is a manual verification pass on real hardware, required by the spec's acceptance gate ("must pass the equivalence harness on real video fixtures, not just the synthetic throughput spike"). Do not skip it or claim Task 1–4 are "done" without running it.

**Files:** none modified — verification only.

- [ ] **Step 1: Confirm the fixtures are present on `mehek`**

Run (SSH):
```bash
ssh rutalab@mehek.taild08eb9.ts.net "bash -lc 'cd ~/hydra-fix-seq-obb-cuda && bash tools/equivalence/fixtures/fetch_fixtures.sh'"
```
Expected: fixtures already present (fast no-op) or freshly downloaded, no errors.

- [ ] **Step 2: Sync this branch's changes to a fresh checkout on mehek**

```bash
scp -r /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/inference-pipeline-redesign/src \
  rutalab@mehek.taild08eb9.ts.net:~/hydra-fix-seq-obb-cuda/src
scp /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/inference-pipeline-redesign/tests/test_inference_obb_artifacts.py \
    /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker/.worktrees/inference-pipeline-redesign/tests/test_inference_stages_obb.py \
  rutalab@mehek.taild08eb9.ts.net:~/hydra-fix-seq-obb-cuda/tests/
```

- [ ] **Step 3: Run the CPU-testable unit tests on mehek (sanity: no import errors in the real env)**

```bash
ssh rutalab@mehek.taild08eb9.ts.net "bash -lc '
source ~/mambaforge/etc/profile.d/conda.sh
conda activate hydra-cuda
cd ~/hydra-fix-seq-obb-cuda
python -m pytest tests/test_inference_obb_artifacts.py tests/test_inference_stages_obb.py -v
'"
```
Expected: all PASS.

- [ ] **Step 4: Run the equivalence + throughput matrix at `RUNTIME=tensorrt` with `detection_batch_size` set to 8, on the direct and sequential OBB fixtures**

```bash
ssh rutalab@mehek.taild08eb9.ts.net "bash -lc '
source ~/mambaforge/etc/profile.d/conda.sh
conda activate hydra-cuda
cd ~/hydra-fix-seq-obb-cuda
rm -f yolo26x-obb_b*.engine yolo26x-obb_dynamic.engine yolo26s-obb_b*.engine 2>/dev/null
DETECTION_BATCH_SIZE=8 RUNTIME=tensorrt ONLY=\"emi_obb_identity fly_obb ant_obb_sleap ant_obb_sequential\" bash tools/equivalence/run_matrix.sh
'"
```
Expected (per the existing harness's own pass/fail semantics — see `tools/equivalence/README.md` and `PARITY_AUDIT.md` for the tolerance definitions already established): the equivalence gate (new vs. legacy CSV) PASSES within existing tolerances for every listed clip, and the harness's own determinism check (new vs. new) also PASSES. Compare the printed PERFORMANCE line's fps against the `static-b1` baseline recorded in the spec (Phase A/B table) — expect a throughput improvement in the same +20–30% ballpark measured in the synthetic spike, not a regression. If `DETECTION_BATCH_SIZE` is not an environment variable `run_matrix.sh`/`runner.py` currently understands, instead set it via the fixture's config JSON (`tools/equivalence/fixtures/configs/<clip>.json`, field `detection_batch_size`) before running, or pass `--detection-batch-size 8` if `runner.py` exposes it as a CLI flag — check `runner.py --help` first: `ssh rutalab@mehek.taild08eb9.ts.net "bash -lc 'source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda && cd ~/hydra-fix-seq-obb-cuda && python tools/equivalence/runner.py --help'"`.

- [ ] **Step 5: Confirm a fresh `_b8` dynamic engine (not a stale `_b1`) was actually built**

```bash
ssh rutalab@mehek.taild08eb9.ts.net "ls -la ~/hydra-fix-seq-obb-cuda/**/*.engine 2>/dev/null; find ~/hydra-fix-seq-obb-cuda -name '*_b8.engine' -o -name '*_b1.engine'"
```
Expected: both a `*_b1.engine` (from any batch=1 realtime-mode calls in the matrix, if exercised) and a `*_b8.engine` exist, confirming the dynamic-vs-static filename split from Task 1 is working end-to-end with the real exporter (not just the fake in unit tests).

- [ ] **Step 6: Record the results**

No code changes in this step — capture the printed fps/parity output from Step 4 for use in Task 6's doc update. If any clip's equivalence gate FAILS, stop here and treat it as a bug in Tasks 1–2 (do not proceed to Task 6 or claim completion) — file the failure output verbatim for debugging in the next session.

---

### Task 6: Update the spec's Batching Capability Matrix with final Phase C results

**Files:**
- Modify: `docs/superpowers/specs/2026-07-03-tensorrt-coreml-cross-frame-batching-design.md`

- [ ] **Step 1: Append the final results**

Add a new `## Phase C results (implementation)` section at the end of the spec file (after the existing `## Risks` section), summarizing: the two-artifact-filename approach shipped as designed (`_b1.engine` / `_b{N}.engine` with `dynamic=(N>1)`); confirmation that `_direct_obb_runtime.py` required zero changes (the existing static/dynamic detection and `set_input_shape` call already handled it generically); the CoreML classifier fix; the CoreML OBB GUI notice; and the real-hardware equivalence/throughput numbers captured in Task 5, Step 4/6 (fill in the actual fps and parity pass/fail values observed — do not leave placeholder numbers).

- [ ] **Step 2: Commit**

```bash
cd .worktrees/inference-pipeline-redesign
git add docs/superpowers/specs/2026-07-03-tensorrt-coreml-cross-frame-batching-design.md
git commit -m "docs: record Phase C implementation results for TRT/CoreML batching"
```
