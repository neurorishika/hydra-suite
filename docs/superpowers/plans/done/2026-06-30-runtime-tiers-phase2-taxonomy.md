# Runtime Tiers — Phase 2: Tier Taxonomy + RuntimeResolver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 7 mix-and-match runtimes + per-stage selection with a single pipeline-wide `runtime_tier` (CPU / GPU / GPU-Fast), resolved per platform+stage by a new `RuntimeResolver`, with hard-cutover config migration and a collapsed single-selector GUI in both kits.

**Architecture:** A pure `RuntimeResolver` in `runtime/` maps `(tier, platform, stage, artifact_available) -> ResolvedBackend(backend, device, used_fallback)`. `InferenceConfig` grows a `runtime_tier` field; per-stage `compute_runtime` fields are removed and derived from the tier + resolver. `GPU-Fast` on CUDA resolves to TensorRT (existing auto-export) with best-effort native-CUDA fallback. The trackerkit 3-combo panel and posekit combo collapse to one platform-labeled tier selector bound to the kit config schema.

**Tech Stack:** PyTorch, ONNX Runtime, TensorRT, PySide6/Qt, pytest.

## Global Constraints

- Single tier applies pipeline-wide; NO per-stage runtime field remains (spec §3).
- Fast-mode is best-effort: a stage without a fast artifact runs native-GPU on the SAME device, never CPU, logged, never crashes (spec §3, §10).
- AprilTag always runs on CPU, exempt from the tier (spec §3).
- `onnx_cpu`, `onnx_cuda`, `onnx_coreml`, `tensorrt` are removed as user-facing runtimes; the GUI change ships in THIS phase, not later (spec §5).
- Hard-cutover migration mapping: `cpu→CPU`; `cuda`/`mps`→`GPU`; `onnx_*`/`tensorrt`→`GPU-Fast`; mixed configs take the highest tier present (`GPU-Fast`>`GPU`>`CPU`) with a warning (spec §6).
- CoreML backends are NOT wired in Phase 2 (that is Phase 3); on Apple, `GPU-Fast` falls back to native-MPS in Phase 2.
- Commit as the configured git user; no Co-Authored-By trailer.

---

### Task 1: `RuntimeResolver` core (pure, unit-tested)

**Files:**
- Create: `src/hydra_suite/runtime/resolver.py`
- Test: `tests/test_runtime_resolver.py`

**Interfaces:**
- Produces:
  - `RuntimeTier = Literal["cpu", "gpu", "gpu_fast"]`
  - `@dataclass(frozen=True) PlatformInfo(has_cuda: bool, has_mps: bool)`
  - `@dataclass(frozen=True) ResolvedBackend(backend: Literal["torch","tensorrt","coreml"], device: Literal["cpu","cuda","mps"], used_fallback: bool)`
  - `class RuntimeResolver(tier: RuntimeTier, platform: PlatformInfo)` with `resolve(stage: str, artifact_available: Callable[[], bool] = lambda: True) -> ResolvedBackend`
  - `STAGES = ("obb", "head_tail", "cnn", "yolo_pose", "sleap_pose")` — `apriltag` is intentionally absent (always CPU).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime_resolver.py
from hydra_suite.runtime.resolver import (
    PlatformInfo, ResolvedBackend, RuntimeResolver,
)

CUDA = PlatformInfo(has_cuda=True, has_mps=False)
MAC = PlatformInfo(has_cuda=False, has_mps=True)
CPU_ONLY = PlatformInfo(has_cuda=False, has_mps=False)


def test_cpu_tier_always_torch_cpu():
    r = RuntimeResolver("cpu", CUDA)
    assert r.resolve("obb") == ResolvedBackend("torch", "cpu", False)


def test_gpu_tier_cuda_is_native_torch():
    r = RuntimeResolver("gpu", CUDA)
    assert r.resolve("cnn") == ResolvedBackend("torch", "cuda", False)


def test_gpu_tier_mac_is_native_mps():
    r = RuntimeResolver("gpu", MAC)
    assert r.resolve("cnn") == ResolvedBackend("torch", "mps", False)


def test_gpu_fast_cuda_with_artifact_is_tensorrt():
    r = RuntimeResolver("gpu_fast", CUDA)
    assert r.resolve("obb", artifact_available=lambda: True) == ResolvedBackend("tensorrt", "cuda", False)


def test_gpu_fast_cuda_without_artifact_falls_back_to_native_cuda():
    r = RuntimeResolver("gpu_fast", CUDA)
    assert r.resolve("cnn", artifact_available=lambda: False) == ResolvedBackend("torch", "cuda", True)


def test_gpu_fast_mac_phase2_falls_back_to_native_mps():
    # CoreML not wired until Phase 3 -> Phase 2 resolves Apple fast to native MPS.
    r = RuntimeResolver("gpu_fast", MAC)
    assert r.resolve("obb") == ResolvedBackend("torch", "mps", True)


def test_gpu_tier_on_cpu_only_host_degrades_to_cpu():
    r = RuntimeResolver("gpu", CPU_ONLY)
    assert r.resolve("obb") == ResolvedBackend("torch", "cpu", True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime_resolver.py -v`
Expected: FAIL — module `hydra_suite.runtime.resolver` does not exist.

- [ ] **Step 3: Implement the resolver**

```python
# src/hydra_suite/runtime/resolver.py
"""Single authority mapping a runtime tier + platform + stage to a concrete backend.

Replaces per-stage compute_runtime selection and the ONNX/TensorRT capability
tables. Pure and deterministic: no torch import, no I/O — availability is passed
in as a callable so callers own artifact discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

RuntimeTier = Literal["cpu", "gpu", "gpu_fast"]

STAGES = ("obb", "head_tail", "cnn", "yolo_pose", "sleap_pose")


@dataclass(frozen=True)
class PlatformInfo:
    has_cuda: bool
    has_mps: bool


@dataclass(frozen=True)
class ResolvedBackend:
    backend: Literal["torch", "tensorrt", "coreml"]
    device: Literal["cpu", "cuda", "mps"]
    used_fallback: bool


class RuntimeResolver:
    def __init__(self, tier: RuntimeTier, platform: PlatformInfo) -> None:
        self.tier = tier
        self.platform = platform

    def _native_gpu(self) -> tuple[str, str]:
        """Return (backend, device) for the native-GPU tier, or CPU degrade."""
        if self.platform.has_cuda:
            return ("torch", "cuda")
        if self.platform.has_mps:
            return ("torch", "mps")
        return ("torch", "cpu")

    def resolve(
        self,
        stage: str,
        artifact_available: Callable[[], bool] = lambda: True,
    ) -> ResolvedBackend:
        if self.tier == "cpu":
            return ResolvedBackend("torch", "cpu", False)

        if self.tier == "gpu":
            backend, device = self._native_gpu()
            return ResolvedBackend(backend, device, used_fallback=(device == "cpu"))

        # gpu_fast
        if self.platform.has_cuda:
            if artifact_available():
                return ResolvedBackend("tensorrt", "cuda", False)
            return ResolvedBackend("torch", "cuda", used_fallback=True)
        if self.platform.has_mps:
            # Phase 2: CoreML not wired yet -> native MPS fallback.
            return ResolvedBackend("torch", "mps", used_fallback=True)
        return ResolvedBackend("torch", "cpu", used_fallback=True)


def detect_platform() -> PlatformInfo:
    """Detect host acceleration via the existing gpu_utils availability flags."""
    from hydra_suite.utils.gpu_utils import CUDA_AVAILABLE, MPS_AVAILABLE

    return PlatformInfo(has_cuda=bool(CUDA_AVAILABLE), has_mps=bool(MPS_AVAILABLE))
```

(Verify `CUDA_AVAILABLE`/`MPS_AVAILABLE` names in `src/hydra_suite/utils/gpu_utils.py`; if the flag is named `_cuda_like_available()` use that instead. This is the only place `detect_platform` couples to gpu_utils.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runtime_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/runtime/resolver.py tests/test_runtime_resolver.py
git commit -m "feat(runtime): add RuntimeResolver mapping tier+platform+stage to backend"
```

---

### Task 2: `InferenceConfig.runtime_tier` + hard-cutover migration

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (add `runtime_tier` field ~line 157; add `_migrate_legacy_runtimes` used by `from_dict` ~line 283; keep `_collect_all_runtimes` for back-compat internal use but no longer user-set)
- Test: `tests/test_inference_config_tier_migration.py`

**Interfaces:**
- Consumes: `RuntimeTier` from `hydra_suite.runtime.resolver`.
- Produces: `InferenceConfig.runtime_tier: RuntimeTier = "gpu"`; module function `migrate_runtime_to_tier(runtimes: set[str]) -> RuntimeTier`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_config_tier_migration.py
from hydra_suite.core.inference.config import migrate_runtime_to_tier


def test_cpu_maps_to_cpu():
    assert migrate_runtime_to_tier({"cpu"}) == "cpu"


def test_cuda_and_mps_map_to_gpu():
    assert migrate_runtime_to_tier({"cuda"}) == "gpu"
    assert migrate_runtime_to_tier({"mps"}) == "gpu"


def test_onnx_and_tensorrt_map_to_gpu_fast():
    for rt in ("onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"):
        assert migrate_runtime_to_tier({rt}) == "gpu_fast"


def test_mixed_takes_highest_tier():
    assert migrate_runtime_to_tier({"cpu", "cuda", "tensorrt"}) == "gpu_fast"
    assert migrate_runtime_to_tier({"cpu", "mps"}) == "gpu"


def test_empty_defaults_to_gpu():
    assert migrate_runtime_to_tier(set()) == "gpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_config_tier_migration.py -v`
Expected: FAIL — `migrate_runtime_to_tier` undefined.

- [ ] **Step 3: Implement the field + migration**

In `config.py`, add near the top imports: `from hydra_suite.runtime.resolver import RuntimeTier`. Add the migration function:

```python
def migrate_runtime_to_tier(runtimes: set[str]) -> "RuntimeTier":
    """Map legacy per-stage runtime strings to a single pipeline tier.

    cpu -> "cpu"; cuda/mps -> "gpu"; onnx_*/tensorrt -> "gpu_fast".
    Mixed sets take the highest tier present (gpu_fast > gpu > cpu).
    """
    fast = {"onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"}
    gpu = {"cuda", "mps"}
    if runtimes & fast:
        return "gpu_fast"
    if runtimes & gpu:
        return "gpu"
    return "cpu"
```

Add `runtime_tier: RuntimeTier = "gpu"` to the `InferenceConfig` dataclass (after `pipeline_depth`, ~line 157). In `from_dict` (~line 283), derive the tier when a legacy config omits `runtime_tier`:

```python
        raw_tier = d.get("runtime_tier")
        if raw_tier is None:
            # Legacy config: derive from any per-stage runtime strings present.
            legacy = _collect_legacy_runtime_strings(d)
            raw_tier = migrate_runtime_to_tier(legacy)
            if legacy:
                logging.getLogger(__name__).warning(
                    "Migrated legacy per-stage runtimes %s -> runtime_tier=%r",
                    legacy, raw_tier,
                )
        # ... pass runtime_tier=raw_tier into the constructor ...
```

Add helper `_collect_legacy_runtime_strings(d: dict) -> set[str]` that pulls `compute_runtime`/`detect_compute_runtime`/`obb_compute_runtime` from the raw `obb`/`headtail`/`cnn_phases`/`pose` sub-dicts (mirrors `_collect_all_runtimes` but reads the dict, not the built object).

- [ ] **Step 4: Run test + existing config tests**

Run: `python -m pytest tests/test_inference_config_tier_migration.py tests/test_inference_config.py -v`
Expected: PASS (add/adjust any `test_inference_config.py` cases that asserted per-stage runtime round-trips to assert `runtime_tier` instead).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_inference_config_tier_migration.py tests/test_inference_config.py
git commit -m "feat(config): add runtime_tier with hard-cutover legacy migration"
```

---

### Task 3: Derive `RuntimeContext` + per-stage backends from the tier

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime.py` (`RuntimeContext.from_config` ~line 86)
- Modify: `src/hydra_suite/core/inference/stages/obb.py`, `stages/headtail.py`, `stages/cnn.py`, `stages/pose.py` (replace `config.<stage>.compute_runtime` reads with resolver output)
- Test: `tests/test_runtime_context_from_tier.py`

**Interfaces:**
- Consumes: `RuntimeResolver`, `detect_platform`, `InferenceConfig.runtime_tier`.
- Produces: `RuntimeContext.from_config` derives `cuda_mode`/`device`/`tensor_on_cuda` from `runtime_tier` + platform; each stage loader takes a `ResolvedBackend` (backend+device) instead of a runtime string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime_context_from_tier.py
from hydra_suite.core.inference.config import InferenceConfig
from hydra_suite.core.inference.runtime import RuntimeContext


def test_cpu_tier_context_is_not_cuda(monkeypatch):
    import hydra_suite.core.inference.runtime as rt
    cfg = InferenceConfig(runtime_tier="cpu")  # minimal; other fields default/None
    ctx = RuntimeContext.from_config(cfg)
    assert ctx.cuda_mode is False
    assert ctx.tensor_on_cuda is False


def test_gpu_tier_on_cuda_host_is_cuda(monkeypatch):
    import hydra_suite.core.inference.runtime as rt
    monkeypatch.setattr(rt, "_cuda_device_available", lambda: "cuda")
    from hydra_suite.runtime import resolver
    monkeypatch.setattr(resolver, "detect_platform",
                        lambda: resolver.PlatformInfo(has_cuda=True, has_mps=False))
    cfg = InferenceConfig(runtime_tier="gpu")
    ctx = RuntimeContext.from_config(cfg)
    assert ctx.cuda_mode is True
    assert ctx.tensor_on_cuda is True  # native GPU tier keeps tensors on CUDA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime_context_from_tier.py -v`
Expected: FAIL — `from_config` still reads per-stage runtimes.

- [ ] **Step 3: Rewrite `RuntimeContext.from_config` to use the tier**

```python
    @staticmethod
    def from_config(config: InferenceConfig) -> "RuntimeContext":
        from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform

        platform = detect_platform()
        resolver = RuntimeResolver(config.runtime_tier, platform)
        # tensor_on_cuda: only the native-GPU tier on a CUDA host keeps model
        # outputs as live CUDA tensors. gpu_fast (TensorRT) returns CPU numpy.
        gpu_native = resolver.resolve("obb").backend == "torch"
        cuda_mode = config.runtime_tier in ("gpu", "gpu_fast") and platform.has_cuda
        tensor_on_cuda = cuda_mode and gpu_native
        if cuda_mode:
            device = _cuda_device_available()
            nvdec = _nvdec_available()
        else:
            device = _cpu_or_mps_device()
            nvdec = False
        default: ComputeRuntime = "cuda" if cuda_mode else "cpu"
        return RuntimeContext(
            cuda_mode=cuda_mode,
            device=device,
            use_nvdec=nvdec,
            default_runtime=default,
            tensor_on_cuda=tensor_on_cuda,
        )
```

- [ ] **Step 4: Update each stage loader to consult the resolver**

For each of `stages/obb.py`, `stages/headtail.py`, `stages/cnn.py`, `stages/pose.py`: replace the read of `config.<stage>.compute_runtime` with a `RuntimeResolver(config.runtime_tier, detect_platform()).resolve("<stage_key>", artifact_available=<probe>)` call, and pass the resulting `ResolvedBackend.device`/`.backend` to the existing loader (`_load_yolo`, `ClassifierBackend`, pose backend factory). The classifier/OBB loaders already accept a device/runtime string — translate `ResolvedBackend` to that: `torch`→device string; `tensorrt`→`"tensorrt"`. Log `used_fallback=True` at WARNING.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_runtime_context_from_tier.py tests/ -k "inference and (stage or runtime)" -q`
Expected: PASS (adjust stage tests that constructed configs with per-stage runtime strings to set `runtime_tier`).

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/inference/runtime.py src/hydra_suite/core/inference/stages/ tests/test_runtime_context_from_tier.py
git commit -m "feat(inference): derive runtime context + per-stage backends from runtime_tier"
```

---

### Task 4: TensorRT as the CUDA GPU-Fast backend (best-effort fallback)

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py` and `runtime_artifacts.py` (gate TRT auto-export on `tier == gpu_fast` + resolver; fall back to native-CUDA when export/build fails)
- Modify: `src/hydra_suite/core/identity/classification/backend.py` (fast tier → derive+run ONNX/TRT peer; on failure fall back to native CUDA — the `_should_fallback_to_native_runtime` path already exists, extend it)
- Test: `tests/test_gpu_fast_fallback.py`

**Interfaces:**
- Consumes: `ResolvedBackend(backend="tensorrt", device="cuda")`.
- Produces: on `gpu_fast`+CUDA, OBB/classifier attempt TensorRT and fall back to native-CUDA with a logged warning; `used_fallback` reflected upstream.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_fast_fallback.py
import logging
import pytest


def test_classifier_gpu_fast_falls_back_to_native_when_export_fails(monkeypatch, caplog):
    from hydra_suite.core.identity.classification import backend as bmod

    be = bmod.ClassifierBackend.__new__(bmod.ClassifierBackend)
    be._compute_runtime = "tensorrt"  # gpu_fast resolves classifier to TRT on cuda
    # Force ONNX peer derivation to fail so best-effort fallback triggers.
    monkeypatch.setattr(bmod.ClassifierBackend, "_load_onnx",
                        lambda self: (_ for _ in ()).throw(RuntimeError("no engine")))
    monkeypatch.setattr(bmod, "_torch_device", lambda rt: "cuda")
    # native loader is stubbed to a sentinel so we can detect the fallback path
    monkeypatch.setattr(be, "_loader", type("L", (), {"load": staticmethod(lambda p, d: "NATIVE")})())
    with caplog.at_level(logging.WARNING):
        be._ensure_loaded_best_effort()  # new wrapper (Step 3)
    assert be._active_execution_backend == "native"
    assert any("fall" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_fast_fallback.py -v`
Expected: FAIL — `_ensure_loaded_best_effort` undefined.

- [ ] **Step 3: Add a best-effort load wrapper**

In `backend.py`, wrap `_ensure_loaded` so that when the fast (ONNX/TRT) path raises, it logs and retries on the native device:

```python
    def _ensure_loaded_best_effort(self) -> None:
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            if not self._uses_onnx():
                raise
            native_device = _torch_device(self._compute_runtime)
            logger.warning(
                "GPU-Fast classifier backend failed (%s); falling back to native %s",
                exc, native_device,
            )
            self._model = self._loader.load(self._model_path, native_device)
            self._active_execution_backend = "native"
            self._loaded = True
```

Route `predict_batch` to call `_ensure_loaded_best_effort()` instead of `_ensure_loaded()`. Mirror the same best-effort pattern for OBB in `stages/obb.py`: on `load_obb_executor(..., compute_runtime="tensorrt")` raising `ArtifactExportError`/build error, retry with `compute_runtime="cuda"` and log.

- [ ] **Step 4: Run test + classifier + obb suites**

Run: `python -m pytest tests/test_gpu_fast_fallback.py tests/test_classifier_backend.py tests/test_inference_obb_artifacts.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/identity/classification/backend.py src/hydra_suite/core/inference/stages/obb.py src/hydra_suite/core/inference/runtime_artifacts.py tests/test_gpu_fast_fallback.py
git commit -m "feat(inference): best-effort native-GPU fallback for GPU-Fast tier"
```

---

### Task 5: Remove ONNX/TensorRT as user-facing runtimes (backend enumeration)

**Files:**
- Modify: `src/hydra_suite/runtime/compute_runtime.py` (add `available_tiers(platform) -> list[RuntimeTier]` + tier labels; deprecate `allowed_runtimes_for_pipelines` for GUI use)
- Modify: `src/hydra_suite/utils/gpu_utils.py` (runtime-option enumerators used by GUIs)
- Test: `tests/test_available_tiers.py`

**Interfaces:**
- Produces: `available_tiers(platform: PlatformInfo) -> list[RuntimeTier]`; `tier_label(tier, platform) -> str` (e.g. `"GPU-Fast (TensorRT)"` on CUDA, `"GPU-Fast (CoreML)"` on Apple, `"GPU (Metal)"` on Apple, `"GPU (CUDA)"` on CUDA).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_available_tiers.py
from hydra_suite.runtime.resolver import PlatformInfo
from hydra_suite.runtime.compute_runtime import available_tiers, tier_label


def test_cuda_host_tiers_and_labels():
    p = PlatformInfo(has_cuda=True, has_mps=False)
    assert available_tiers(p) == ["cpu", "gpu", "gpu_fast"]
    assert tier_label("gpu", p) == "GPU (CUDA)"
    assert tier_label("gpu_fast", p) == "GPU-Fast (TensorRT)"


def test_mac_host_tiers_and_labels():
    p = PlatformInfo(has_cuda=False, has_mps=True)
    assert available_tiers(p) == ["cpu", "gpu", "gpu_fast"]
    assert tier_label("gpu", p) == "GPU (Metal)"
    assert tier_label("gpu_fast", p) == "GPU-Fast (CoreML)"


def test_cpu_only_host_only_cpu():
    p = PlatformInfo(has_cuda=False, has_mps=False)
    assert available_tiers(p) == ["cpu"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_available_tiers.py -v`
Expected: FAIL — functions undefined.

- [ ] **Step 3: Implement `available_tiers` + `tier_label`**

```python
def available_tiers(platform) -> list:
    if not (platform.has_cuda or platform.has_mps):
        return ["cpu"]
    return ["cpu", "gpu", "gpu_fast"]


def tier_label(tier: str, platform) -> str:
    if tier == "cpu":
        return "CPU"
    accel = "CUDA" if platform.has_cuda else ("Metal" if platform.has_mps else "CPU")
    fast = "TensorRT" if platform.has_cuda else ("CoreML" if platform.has_mps else "CPU")
    return {"gpu": f"GPU ({accel})", "gpu_fast": f"GPU-Fast ({fast})"}[tier]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_available_tiers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/runtime/compute_runtime.py tests/test_available_tiers.py
git commit -m "feat(runtime): available_tiers + platform-aware tier labels"
```

---

### Task 6: trackerkit GUI — collapse three combos to one tier selector

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py` (remove `combo_compute_runtime`, `combo_headtail_runtime`, `combo_cnn_runtime`; add `combo_runtime_tier` + a fallback status label)
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py` (remove `_on_runtime_context_changed`/`_sync_headtail_runtime_selection`/`_populate_*runtime*` wiring for the 3 combos; add tier-change handler binding to config schema)
- Modify: `src/hydra_suite/trackerkit/config/schemas.py` (add `runtime_tier` to the kit config schema; drop per-stage runtime fields)
- Test: `tests/test_trackerkit_runtime_tier_ui.py`

**Interfaces:**
- Consumes: `available_tiers`, `tier_label`, `detect_platform`.
- Produces: `SetupPanel.combo_runtime_tier` (one QComboBox holding tier ids as itemData, platform labels as text); config schema field `runtime_tier`.

- [ ] **Step 1: Write the failing test** (Qt offscreen)

```python
# tests/test_trackerkit_runtime_tier_ui.py
import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from hydra_suite.runtime.resolver import PlatformInfo
from hydra_suite.runtime.compute_runtime import available_tiers, tier_label


def test_tier_combo_lists_platform_tiers():
    # Unit-level guard for the data the panel will render (keeps Qt light).
    p = PlatformInfo(has_cuda=True, has_mps=False)
    labels = [tier_label(t, p) for t in available_tiers(p)]
    assert labels == ["CPU", "GPU (CUDA)", "GPU-Fast (TensorRT)"]
```

(A fuller widget test that instantiates `SetupPanel` and asserts `combo_runtime_tier` item count/text may be added if the panel can be constructed headless; keep it optional if `SetupPanel` requires a full MainWindow.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trackerkit_runtime_tier_ui.py -v`
Expected: PASS for the data guard once Task 5 lands; FAIL earlier. (The GUI wiring below has no pure-python assertion; verify via Step 4 manual/smoke + existing panel tests.)

- [ ] **Step 3: Replace the three combos with one tier selector**

In `setup_panel.py`, delete the `combo_compute_runtime`, `combo_headtail_runtime`, `combo_cnn_runtime` blocks (lines ~609–670 and their `_register_optional_performance_control` / `_set_form_row_visible` calls) and add:

```python
        from hydra_suite.runtime.resolver import detect_platform
        from hydra_suite.runtime.compute_runtime import available_tiers, tier_label

        self.combo_runtime_tier = QComboBox()
        self.combo_runtime_tier.setFixedHeight(30)
        self.combo_runtime_tier.setToolTip(
            "Compute tier for the whole pipeline.\n"
            "CPU · GPU (exact) · GPU-Fast (max speed, some accuracy loss)."
        )
        _platform = detect_platform()
        for _tier in available_tiers(_platform):
            self.combo_runtime_tier.addItem(tier_label(_tier, _platform), _tier)
        self.combo_runtime_tier.currentIndexChanged.connect(
            self._main_window._on_runtime_tier_changed
        )
        self.lbl_runtime_fallback = QLabel("")  # §5.4 fallback indicator
        self.lbl_runtime_fallback.setWordWrap(True)
        runtime_card = self._create_performance_control_card(
            "Compute tier", self.combo_runtime_tier
        )
        self._performance_base_control_cards.append(runtime_card)
        self._performance_control_cards.append(runtime_card)
```

- [ ] **Step 4: Rewire MainWindow + schema**

In `main_window.py`: delete `_on_runtime_context_changed`, `_sync_headtail_runtime_selection`, and the per-combo populate methods; add `_on_runtime_tier_changed(self, index)` that reads `combo_runtime_tier.currentData()` and writes `self.config.runtime_tier`. In `trackerkit/config/schemas.py` add `runtime_tier: str = "gpu"` and remove the per-stage runtime fields; ensure `to_dict`/`from_dict` round-trip it and map any legacy per-stage keys via `migrate_runtime_to_tier`.

Run: `python -m pytest tests/ -k "trackerkit and (setup or panel or config or schema)" -q`
Expected: PASS (update panel/config tests that referenced the removed combos).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/ tests/test_trackerkit_runtime_tier_ui.py
git commit -m "feat(trackerkit): collapse runtime combos into one compute-tier selector"
```

---

### Task 7: posekit GUI — single tier selector

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py` (replace `combo_pred_runtime` population `_populate_pred_runtime_options` with tier enumeration)
- Modify: `src/hydra_suite/posekit/config/schemas.py` (add `runtime_tier`)
- Test: `tests/test_posekit_runtime_tier_ui.py`

**Interfaces:** same `available_tiers`/`tier_label`/`detect_platform` as Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_posekit_runtime_tier_ui.py
from hydra_suite.runtime.resolver import PlatformInfo
from hydra_suite.runtime.compute_runtime import available_tiers, tier_label


def test_posekit_tier_labels_mac():
    p = PlatformInfo(has_cuda=False, has_mps=True)
    assert [tier_label(t, p) for t in available_tiers(p)] == ["CPU", "GPU (Metal)", "GPU-Fast (CoreML)"]
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `python -m pytest tests/test_posekit_runtime_tier_ui.py -v`
Expected: PASS once Task 5 lands (guards the data the combo renders).

- [ ] **Step 3: Replace `_populate_pred_runtime_options`**

In `posekit/gui/main_window.py`, change `_populate_pred_runtime_options` (~line 4236) to enumerate `available_tiers(detect_platform())` with `tier_label`, storing the tier id as `itemData`; rename the widget to `combo_runtime_tier` (or keep `combo_pred_runtime` but populate with tiers) and bind selection to `posekit` config `runtime_tier`. Remove the `allowed_runtimes_for_pipelines`-based population.

- [ ] **Step 4: Run posekit tests**

Run: `python -m pytest tests/ -k "posekit and (runtime or config or predict)" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/ tests/test_posekit_runtime_tier_ui.py
git commit -m "feat(posekit): single compute-tier selector replacing runtime dropdown"
```

---

### Task 8: End-to-end migration + equivalence integration test

**Files:**
- Test: `tests/test_runtime_tier_end_to_end.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_runtime_tier_end_to_end.py
import json
from hydra_suite.core.inference.config import InferenceConfig


def test_legacy_config_json_migrates_to_tier(tmp_path):
    legacy = {
        "obb": {"mode": "direct", "direct": {"model_path": "m.pt", "compute_runtime": "tensorrt"}},
        "pipeline_depth": 2,
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy))
    cfg = InferenceConfig.from_json(str(p))
    assert cfg.runtime_tier == "gpu_fast"
    assert not hasattr(cfg.obb.direct, "compute_runtime") or \
        cfg.obb.direct.compute_runtime in (None, "")  # per-stage field removed/ignored
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_runtime_tier_end_to_end.py -v`
Expected: PASS

- [ ] **Step 3: Run the full inference + equivalence suites**

Run: `python -m pytest tests/ -k "inference or runtime or config or classifier or obb or pose" -q`
Expected: PASS. Then, on mehek, run `tools/equivalence/run_matrix.sh` for the `gpu` (native) tier and confirm the CPU/GPU equivalence + device-invariance guarantees from Phase 0 still hold.

- [ ] **Step 4: Commit**

```bash
git add tests/test_runtime_tier_end_to_end.py
git commit -m "test(runtime): end-to-end legacy-config migration to runtime_tier"
```

---

## Self-Review

- **Spec coverage:** §2/§3 tiers → Tasks 1,3; §4 RuntimeResolver → Task 1; §5 GUI collapse + labels → Tasks 5,6,7; §6 migration → Tasks 2,8; §3 best-effort fallback → Task 4; AprilTag exemption → resolver `STAGES` omits it (Task 1). Phase-2 Apple `GPU-Fast`→native-MPS fallback → Task 1 test `test_gpu_fast_mac_phase2_falls_back_to_native_mps`.
- **Type consistency:** `RuntimeTier`, `PlatformInfo`, `ResolvedBackend`, `available_tiers`, `tier_label`, `migrate_runtime_to_tier`, `detect_platform` used identically across tasks.
- **Placeholder scan:** GUI wiring steps name exact widgets/methods to remove; where a full headless widget test is impractical (SetupPanel needs a MainWindow) the plan states so and falls back to a data-level guard test plus existing panel tests, rather than a fake assertion.
