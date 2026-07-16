# Runtime Tiers — Phase 3: Native CoreML Fast-Mode (Apple) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GPU-Fast` on Apple resolve to native CoreML `.mlpackage` inference for YOLO (OBB/pose) and for tiny + torchvision/timm classifiers, with artifact auto-management mirroring TensorRT and best-effort native-MPS fallback.

**Architecture:** Add CoreML exporters mirroring the existing ONNX exporters, a `.mlpackage` load/run path in the classifier backend and OBB loader, and flip the Phase-2 Apple `gpu_fast` resolver branch from "native-MPS fallback" to "CoreML when a fresh `.mlpackage` exists, else native-MPS." Artifact freshness reuses the mtime-marker pattern from `runtime_artifacts.py`.

**Tech Stack:** coremltools, ultralytics (CoreML export), PyTorch, PySide6, pytest. macOS/Apple Silicon only for execution; export/build gated on `platform.has_mps`.

## Global Constraints

- CoreML applies ONLY on Apple hosts (`platform.has_mps`); never on CUDA/CPU (spec §3).
- Fast-mode stays best-effort: any family without a working `.mlpackage` runs native-MPS, logged, never CPU, never crash (spec §3, §7, §10).
- Export coverage is mandatory for every classifier family: `yolo`/`yolo_multihead`/`classifier_multihead` (ultralytics), `tinyclassifier`, torchvision+timm; multihead exports per-factor (spec §7).
- Fast-mode numerics are NOT bit-identical to native but MUST be deterministic run-to-run (spec §3, §9).
- `coremltools` is a new optional dependency under the `mps` extra; import lazily and guard availability.
- Commit as the configured git user; no Co-Authored-By trailer.

---

### Task 1: Add `coremltools` dependency + availability flag

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]` `mps` extra)
- Modify: `src/hydra_suite/utils/gpu_utils.py` (add `COREMLTOOLS_AVAILABLE`)
- Test: `tests/test_coremltools_availability.py`

**Interfaces:**
- Produces: `COREMLTOOLS_AVAILABLE: bool` in `gpu_utils`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coremltools_availability.py
from hydra_suite.utils import gpu_utils


def test_coremltools_flag_exists_and_is_bool():
    assert isinstance(gpu_utils.COREMLTOOLS_AVAILABLE, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coremltools_availability.py -v`
Expected: FAIL — `COREMLTOOLS_AVAILABLE` undefined.

- [ ] **Step 3: Add the flag + dependency**

In `gpu_utils.py`:

```python
try:
    import coremltools as _ct  # noqa: F401
    COREMLTOOLS_AVAILABLE = True
except Exception:
    COREMLTOOLS_AVAILABLE = False
```

In `pyproject.toml`, add to the `mps` optional-dependencies list: `"coremltools>=8.0"`.

- [ ] **Step 4: Run test + install**

Run: `pip install coremltools>=8.0` (on the Apple dev box), then `python -m pytest tests/test_coremltools_availability.py -v`
Expected: PASS (`True` on the Apple box with the dep installed; `False` elsewhere — still a bool).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/hydra_suite/utils/gpu_utils.py tests/test_coremltools_availability.py
git commit -m "build(mps): add coremltools optional dependency + availability flag"
```

---

### Task 2: `export_tiny_to_coreml` + `export_torchvision_to_coreml`

**Files:**
- Modify: `src/hydra_suite/training/tiny_model.py` (add `export_tiny_to_coreml`)
- Modify: `src/hydra_suite/training/torchvision_model.py` (add `export_torchvision_to_coreml`, covers timm)
- Test: `tests/test_classifier_coreml_export.py`

**Interfaces:**
- Produces:
  - `export_tiny_to_coreml(model, ckpt: dict, mlpackage_path: str | Path) -> Path`
  - `export_torchvision_to_coreml(model, ckpt: dict, mlpackage_path: str | Path) -> Path`
  Both mirror the ONNX exporter signatures; batch axis dynamic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_coreml_export.py
import pytest

pytest.importorskip("coremltools")
torch = pytest.importorskip("torch")

from hydra_suite.training.torchvision_model import export_torchvision_to_coreml


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.p = torch.nn.AdaptiveAvgPool2d(1)
        self.f = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.f(self.p(self.c(x)).flatten(1))


def test_export_torchvision_to_coreml_writes_mlpackage(tmp_path):
    out = tmp_path / "m.mlpackage"
    p = export_torchvision_to_coreml(_Net().eval(), {"input_size": (32, 32)}, out)
    assert p.exists()
    assert str(p).endswith(".mlpackage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier_coreml_export.py -v`
Expected: FAIL — `export_torchvision_to_coreml` undefined.

- [ ] **Step 3: Implement the exporters**

Add to `torchvision_model.py` (and the tiny analogue with `input_size` default `[64, 128]` in `tiny_model.py`):

```python
def export_torchvision_to_coreml(model, ckpt, mlpackage_path):
    """Export a torchvision/timm classifier to a CoreML .mlpackage.

    Batch axis is a RangeDim so any batch size works at inference.
    """
    from pathlib import Path
    import coremltools as ct
    import torch

    mlpackage_path = Path(mlpackage_path)
    h, w = ckpt.get("input_size", (224, 224))
    model.eval()
    dummy = torch.zeros(1, 3, int(h), int(w))
    traced = torch.jit.trace(model, dummy)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input",
            shape=ct.Shape(shape=(ct.RangeDim(1, 512), 3, int(h), int(w))),
        )],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.save(str(mlpackage_path))
    return mlpackage_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classifier_coreml_export.py -v`
Expected: PASS on the Apple box; SKIPPED where coremltools is absent.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/tiny_model.py src/hydra_suite/training/torchvision_model.py tests/test_classifier_coreml_export.py
git commit -m "feat(training): CoreML .mlpackage exporters for tiny + torchvision/timm classifiers"
```

---

### Task 3: YOLO CoreML export for OBB/pose

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (add `format="coreml"` branch in `_export_artifact`; `.mlpackage` artifact path + freshness)
- Test: `tests/test_obb_coreml_export.py`

**Interfaces:**
- Consumes: existing `_export_artifact(pt_path, artifact_path, runtime, imgsz, batch_size)` and `_artifact_path_for`.
- Produces: `runtime="coreml"` exports a `.mlpackage` via `YOLO(pt).export(format="coreml")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obb_coreml_export.py
import pytest

pytest.importorskip("coremltools")


def test_coreml_artifact_path_suffix():
    from hydra_suite.core.inference.runtime_artifacts import _artifact_path_for
    p = _artifact_path_for("model.pt", "coreml")
    assert str(p).endswith(".mlpackage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_obb_coreml_export.py -v`
Expected: FAIL — `_artifact_path_for` does not map `coreml` → `.mlpackage`.

- [ ] **Step 3: Implement the coreml artifact path + export branch**

In `runtime_artifacts.py`: extend `_artifact_path_for` to return `pt.with_suffix(".mlpackage")` for `runtime == "coreml"`; add a branch in `_export_artifact`:

```python
    elif runtime == "coreml":
        model.export(format="coreml", imgsz=imgsz, nms=False)
```

Add `"coreml"` to the runtime-classification sets (`_COREML_RUNTIMES = frozenset({"coreml"})`) and route `load_obb_executor(compute_runtime="coreml")` through a CoreML executor (ultralytics can load the `.mlpackage` back via `YOLO(mlpackage_path)`), mirroring the torch branch, on Apple only.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_obb_coreml_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/runtime_artifacts.py tests/test_obb_coreml_export.py
git commit -m "feat(obb): CoreML .mlpackage export + load for YOLO OBB/pose"
```

---

### Task 4: Classifier CoreML load/run path

**Files:**
- Modify: `src/hydra_suite/core/identity/classification/backend.py` (add `_derive_coreml_peer`, `_load_coreml`, `_forward_coreml`; route `compute_runtime == "coreml"` here)
- Test: `tests/test_classifier_coreml_backend.py`

**Interfaces:**
- Consumes: `export_tiny_to_coreml`, `export_torchvision_to_coreml`, ultralytics for yolo arch.
- Produces: classifier `predict_batch` runs via CoreML when `compute_runtime == "coreml"` and a `.mlpackage` peer exists; `_active_execution_backend == "coreml"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_coreml_backend.py
import pytest

pytest.importorskip("coremltools")

from hydra_suite.core.identity.classification import backend as bmod


def test_backend_reports_coreml_uses():
    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    be._compute_runtime = "coreml"
    assert be._uses_coreml() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classifier_coreml_backend.py -v`
Expected: FAIL — `_uses_coreml` undefined.

- [ ] **Step 3: Implement the CoreML path**

Add `_uses_coreml(self)` (returns `self._compute_runtime == "coreml"`); `_derive_coreml_peer` that dispatches by `self._metadata.arch` to `export_tiny_to_coreml` / `export_torchvision_to_coreml` / ultralytics coreml export (per-factor for multihead); `_load_coreml` loads the `.mlpackage` via `coremltools.models.MLModel`; `_forward_coreml(batch_np)` runs prediction and returns logits. Wire `_ensure_loaded` to select the coreml path for `_uses_coreml()`, and `predict_batch` to call `_forward_coreml`. Reuse `_ensure_loaded_best_effort` (Phase 2 Task 4) so a CoreML failure falls back to native-MPS.

- [ ] **Step 4: Run test + classifier suite**

Run: `python -m pytest tests/test_classifier_coreml_backend.py tests/test_classifier_backend.py -q`
Expected: PASS (coreml-specific tests skipped off-Apple)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/classification/backend.py tests/test_classifier_coreml_backend.py
git commit -m "feat(classifier): native CoreML .mlpackage load/run path"
```

---

### Task 5: Flip resolver Apple `gpu_fast` → CoreML; wire artifact-availability probe

**Files:**
- Modify: `src/hydra_suite/runtime/resolver.py` (Apple `gpu_fast` branch returns `coreml` when available)
- Modify: stage loaders (`stages/obb.py`, `headtail.py`, `cnn.py`, `pose.py`) to pass `compute_runtime="coreml"` when the resolver returns `backend == "coreml"`
- Test: update `tests/test_runtime_resolver.py`

**Interfaces:**
- Produces: `RuntimeResolver("gpu_fast", MAC).resolve("obb", artifact_available=lambda: True) == ResolvedBackend("coreml", "mps", False)`.

- [ ] **Step 1: Update the failing test**

Replace the Phase-2 `test_gpu_fast_mac_phase2_falls_back_to_native_mps` with:

```python
def test_gpu_fast_mac_with_artifact_is_coreml():
    r = RuntimeResolver("gpu_fast", MAC)
    assert r.resolve("obb", artifact_available=lambda: True) == ResolvedBackend("coreml", "mps", False)


def test_gpu_fast_mac_without_artifact_falls_back_to_native_mps():
    r = RuntimeResolver("gpu_fast", MAC)
    assert r.resolve("cnn", artifact_available=lambda: False) == ResolvedBackend("torch", "mps", True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime_resolver.py -v`
Expected: FAIL — Apple `gpu_fast` still returns native-MPS unconditionally.

- [ ] **Step 3: Implement the Apple CoreML branch**

In `resolver.py`, replace the Phase-2 Apple `gpu_fast` block:

```python
        if self.platform.has_mps:
            if artifact_available():
                return ResolvedBackend("coreml", "mps", False)
            return ResolvedBackend("torch", "mps", used_fallback=True)
```

Update stage loaders so `backend == "coreml"` maps to `compute_runtime="coreml"` for the classifier/OBB loaders (Task 3/4).

- [ ] **Step 4: Run resolver + stage tests**

Run: `python -m pytest tests/test_runtime_resolver.py tests/ -k "stage or runtime" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/runtime/resolver.py src/hydra_suite/core/inference/stages/ tests/test_runtime_resolver.py
git commit -m "feat(runtime): resolve Apple GPU-Fast to CoreML with native-MPS fallback"
```

---

### Task 6: `.mlpackage` artifact freshness + determinism test

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime_artifacts.py` (freshness for `.mlpackage` mirrors `.engine`)
- Test: `tests/test_coreml_determinism.py`

**Interfaces:** reuses `_artifact_is_fresh` / `_write_fresh_marker`.

- [ ] **Step 1: Write the failing/skip-guarded test**

```python
# tests/test_coreml_determinism.py
import numpy as np
import pytest

pytest.importorskip("coremltools")


@pytest.mark.skipif(
    not __import__("hydra_suite.utils.gpu_utils", fromlist=["MPS_AVAILABLE"]).MPS_AVAILABLE,
    reason="Apple MPS required",
)
def test_coreml_classifier_is_deterministic_run_to_run(tmp_path):
    # Build a tiny classifier, export to mlpackage, run twice, assert identical.
    from hydra_suite.core.identity.classification import backend as bmod  # noqa: F401
    # (Construct a ClassifierBackend on a fixture .pth with compute_runtime="coreml",
    #  run predict_batch twice on the same crops, assert np.array_equal.)
    crops = [np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(8)]
    # ... load backend, out1 = be.predict_batch(crops); out2 = be.predict_batch(crops)
    # assert np.array_equal(np.array(out1), np.array(out2))
```

Provide a real fixture classifier `.pth` under `tests/fixtures/` (or generate one in a `tmp_path` via the training helpers) so the test constructs a concrete backend; assert `np.array_equal(out1, out2)`.

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_coreml_determinism.py -v`
Expected: PASS on Apple; SKIPPED elsewhere.

- [ ] **Step 3: Implement freshness for `.mlpackage`**

Ensure `_artifact_is_fresh`/`_write_fresh_marker` handle the `.mlpackage` directory-style artifact (stat the package dir, not a single file).

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/inference/runtime_artifacts.py tests/test_coreml_determinism.py tests/fixtures/
git commit -m "feat(coreml): .mlpackage freshness tracking + determinism test"
```

---

### Task 7: GUI — make `GPU-Fast (CoreML)` live + fallback indicator

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py` / `main_window.py` (populate `lbl_runtime_fallback` from resolver `used_fallback`)
- Modify: `src/hydra_suite/posekit/gui/main_window.py` (same indicator, if applicable)
- Test: `tests/test_runtime_fallback_indicator.py`

**Interfaces:**
- Consumes: `RuntimeResolver.resolve(...).used_fallback`, `tier_label` (already emits `GPU-Fast (CoreML)` on Apple from Phase 2 Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime_fallback_indicator.py
from hydra_suite.runtime.resolver import PlatformInfo, RuntimeResolver


def test_fallback_message_when_no_coreml_artifact():
    r = RuntimeResolver("gpu_fast", PlatformInfo(has_cuda=False, has_mps=True))
    res = r.resolve("cnn", artifact_available=lambda: False)
    assert res.used_fallback is True  # panel renders "running native MPS (no fast artifact)"
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_runtime_fallback_indicator.py -v`
Expected: PASS (guards the signal the panel renders).

- [ ] **Step 3: Render the indicator**

When the pipeline resolves stages, aggregate any `used_fallback=True` and set `self.lbl_runtime_fallback` text (e.g. "GPU-Fast: 1 stage using native GPU (no fast artifact)"); clear it otherwise. Non-blocking, informational (spec §5.4).

- [ ] **Step 4: Run trackerkit/posekit UI tests**

Run: `python -m pytest tests/ -k "runtime and (indicator or trackerkit or posekit)" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/ src/hydra_suite/posekit/ tests/test_runtime_fallback_indicator.py
git commit -m "feat(gui): live GPU-Fast (CoreML) label + fallback indicator"
```

---

### Task 8: Coverage matrix validation + docs

**Files:**
- Modify: `docs/developer-guide/inference-fast-mode-future.md` → rename/repurpose to `inference-fast-mode.md` (document the shipped CoreML tier)
- Test: `tests/test_fastmode_coverage_matrix.py`

- [ ] **Step 1: Write the coverage test**

```python
# tests/test_fastmode_coverage_matrix.py
import pytest

pytest.importorskip("coremltools")

FAMILIES = ["tinyclassifier", "resnet18", "efficientnet_b0"]  # timm/torchvision reps


@pytest.mark.parametrize("arch", FAMILIES)
def test_every_classifier_family_has_a_coreml_exporter(arch, tmp_path):
    # Assert the backend can derive a .mlpackage peer for each family (or that a
    # documented native-MPS fallback path is taken) — no family is left unhandled.
    from hydra_suite.core.identity.classification import backend as bmod  # noqa: F401
    # Build a minimal ckpt per arch, call the appropriate export_* helper,
    # assert a .mlpackage is produced.
    assert True  # replace with concrete per-arch export assertions
```

Replace the placeholder body with concrete per-arch export assertions using the Task 2 helpers + ultralytics for the yolo family.

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_fastmode_coverage_matrix.py -v`
Expected: PASS on Apple; SKIPPED elsewhere.

- [ ] **Step 3: Update the fast-mode doc**

Rewrite `docs/developer-guide/inference-fast-mode.md` to describe the live tiers (CUDA→TensorRT, Apple→CoreML), coverage matrix, and the best-effort fallback contract. Add to `mkdocs.yml` nav if needed; run `make docs-build`.

- [ ] **Step 4: Commit**

```bash
git add docs/ mkdocs.yml tests/test_fastmode_coverage_matrix.py
git commit -m "docs+test: document shipped CoreML fast-mode + family coverage matrix"
```

---

## Self-Review

- **Spec coverage:** §3 CoreML tier on Apple → Tasks 3,4,5; §7 all classifier families (yolo/tiny/torchvision+timm, multihead per-factor) → Tasks 2,3,4,8; artifact auto-management → Tasks 3,6; §9 determinism → Task 6; §5.3/5.4 live CoreML label + fallback indicator → Task 7; new dependency → Task 1.
- **Type consistency:** `export_tiny_to_coreml` / `export_torchvision_to_coreml` mirror the ONNX exporter signatures; `ResolvedBackend("coreml","mps",...)` consistent with Phase 2 resolver types; `_uses_coreml`/`_load_coreml`/`_forward_coreml` parallel the existing `_uses_onnx`/`_load_onnx`/`_forward_onnx`.
- **Placeholder scan:** Two tests (Task 6 determinism, Task 8 coverage) carry explicit "replace with concrete assertions" bodies because they require a real fixture classifier `.pth`; each step names exactly what to assert (`np.array_equal` run-to-run; per-arch `.mlpackage` produced). Provide the fixture during execution — not a silent stub.
- **Platform gating:** every CoreML path guarded on `platform.has_mps` / `COREMLTOOLS_AVAILABLE`; skips cleanly on CUDA/CPU.
