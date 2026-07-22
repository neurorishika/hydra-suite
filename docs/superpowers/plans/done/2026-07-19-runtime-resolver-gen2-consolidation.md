# Runtime Resolver Gen-2 Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the two-generation runtime system (5 representations, 6 bridges) down to the single Gen-2 path: `runtime_tier` → `RuntimeResolver.resolve` → `ResolvedBackend`, consumed directly by backends, with all legacy string vocabulary and migration logic deleted from `src/`.

**Architecture:** `ResolvedBackend(backend, device, used_fallback)` becomes the one runtime value that flows. Backend predicates are rewritten from string-set membership (`rt in ("cuda","onnx_cuda","tensorrt")`) to field checks (`resolved.device == "cuda"`) — a faithful translation because the resolver only ever emits `{cpu,mps,cuda,tensorrt,coreml}`, so `onnx_cuda`/`onnx_cpu` string cases simply never occur. A one-shot offline script migrates stored configs; loading a config with no `runtime_tier` becomes a loud error.

**Tech Stack:** Python 3, dataclasses, pytest, PySide6 (GUI, untouched in behavior), ONNX Runtime execution providers, PyTorch / TensorRT / CoreML backends.

## Global Constraints

- Runtime tiers are exactly `{cpu, gpu, gpu_fast}` — do not add or rename.
- Resolver output backends are exactly `{torch, tensorrt, coreml}`; devices exactly `{cpu, cuda, mps}`. Do not introduce new values.
- Core/Runtime/Data/Training/Utils must never import from any app-layer package (trackerkit, posekit, etc.) or from Integrations. (CLAUDE.md dependency direction.)
- No change to public CLI entry points or inter-kit APIs.
- Tier semantics are preserved exactly: the Phase-0 golden table `(tier × platform × stage × artifact) → ResolvedBackend` must be byte-identical before and after every task in Tasks 2–13.
- Commit as the configured git user; do NOT add a `Co-Authored-By: Claude` trailer.
- `make pytest` must pass at the end of every task. Run `make format` before each commit.
- Platform is injected via `PlatformInfo(has_cuda, has_mps)` — all four host shapes (neither, cuda-only, mps-only, both) are testable on any machine; never gate tests on the real host.

---

## File Structure

**New files**
- `tests/runtime/test_resolver_golden.py` — the behavior-preservation golden table (Task 1).
- `scripts/migrate_runtime_config.py` — one-shot offline config/preset migrator; self-contained, imports nothing being deleted (Task 11).
- `tests/scripts/test_migrate_runtime_config.py` — migration-script fixtures (Task 11).
- `src/hydra_suite/runtime/onnx_providers.py` — the renamed, shrunken module: `execution_providers_for(resolved)`, CoreML options, TensorRT cache options, dedup helpers (Task 12).

**Modified — core/inference**
- `runtime.py` — add `RuntimeContext.resolved`; derive mode flags from it; delete `runtime_to_compute_runtime` (Tasks 3, 5).
- `config.py` — loud missing-tier error; drop per-stage `compute_runtime` fields, `ComputeRuntime` Literal, `CUDA_RUNTIMES`/`CPU_RUNTIMES` frozensets; `from_params` reads only `RUNTIME_TIER` (Tasks 8, 9, 10).
- `stages/obb.py`, `stages/pose.py`, `stages/cnn.py`, `stages/headtail.py`, `stages/crops.py` — consume `runtime.resolved`; rewrite string predicates to field checks (Task 4).

**Modified — core/identity backends**
- `classification/backend.py`, `classification/headtail.py`, `pose/backends/sleap.py` — accept `ResolvedBackend`; rewrite predicates; call `execution_providers_for` (Task 4).
- `properties/cache.py` — resolve tier at the param-bag boundary; write resolved device into payload (Task 7).

**Modified — GUI / params channel**
- `trackerkit/gui/orchestrators/{config,tracking,session}.py`, `panels/{detection_panel,identity_panel}.py`, `gui/workers/{preview_worker,crops_worker}.py`, `trackerkit/cli_config.py`, `trackerkit/tracking_cache.py`, `posekit/gui/main_window.py`, `posekit/gui/runtimes.py`, `core/tracking/worker.py`, `data/dataset_generation.py` — drop `COMPUTE_RUNTIME`/pose-flavor writes, keep `RUNTIME_TIER` (Tasks 6, 7, 13).

**Modified — runtime**
- `resolver.py` — delete `resolve_compute_runtime` display shim (Task 5).
- `compute_runtime.py` → renamed to `onnx_providers.py`; legacy surface deleted (Task 12).

---

## Task 1: Golden characterization test (safety net)

**Files:**
- Create: `tests/runtime/test_resolver_golden.py`

**Interfaces:**
- Consumes: `hydra_suite.runtime.resolver.{RuntimeResolver, PlatformInfo, ResolvedBackend}`.
- Produces: a frozen golden dict other tasks must not change; the canonical enumeration of platform shapes used across the plan.

- [ ] **Step 1: Write the golden table test**

```python
# tests/runtime/test_resolver_golden.py
"""Behavior-preservation net for the Gen-2 runtime consolidation.

Every (tier, platform, stage, artifact_available) -> ResolvedBackend mapping
is frozen here. No task in the consolidation may change these values.
"""
import itertools
import pytest
from hydra_suite.runtime.resolver import RuntimeResolver, PlatformInfo

TIERS = ("cpu", "gpu", "gpu_fast")
STAGES = ("obb", "head_tail", "cnn", "yolo_pose", "sleap_pose", "bgsub")
PLATFORMS = {
    "none": PlatformInfo(has_cuda=False, has_mps=False),
    "cuda": PlatformInfo(has_cuda=True, has_mps=False),
    "mps": PlatformInfo(has_cuda=False, has_mps=True),
    "both": PlatformInfo(has_cuda=True, has_mps=True),
}

def _key(tier, plat, stage, artifact):
    return f"{tier}|{plat}|{stage}|artifact={artifact}"

def _resolve(tier, plat_name, stage, artifact):
    r = RuntimeResolver(tier, PLATFORMS[plat_name]).resolve(
        stage, artifact_available=lambda: artifact
    )
    return (r.backend, r.device, r.used_fallback)

def test_golden_table_is_stable():
    table = {}
    for tier, plat, stage, artifact in itertools.product(
        TIERS, PLATFORMS, STAGES, (True, False)
    ):
        table[_key(tier, plat, stage, artifact)] = _resolve(tier, plat, stage, artifact)
    # Snapshot: values captured from current (pre-refactor) resolver.
    # Regenerate ONCE now, then treat as frozen for the rest of the plan.
    assert table == EXPECTED_GOLDEN
```

- [ ] **Step 2: Generate the frozen snapshot**

Run a throwaway print to capture current values, paste them into `EXPECTED_GOLDEN` as a literal dict at the top of the file:

Run: `python -c "import tests.runtime.test_resolver_golden as t; import itertools; print({t._key(a,b,c,d): t._resolve(a,b,c,d) for a,b,c,d in itertools.product(t.TIERS,t.PLATFORMS,t.STAGES,(True,False))})"`
Then define `EXPECTED_GOLDEN = { ... }` with that exact output.

- [ ] **Step 3: Run to verify it passes against current code**

Run: `python -m pytest tests/runtime/test_resolver_golden.py -v`
Expected: PASS (snapshot matches live resolver).

- [ ] **Step 4: Commit**

```bash
make format
git add tests/runtime/test_resolver_golden.py
git commit -m "test(runtime): freeze tier x platform x stage golden table"
```

---

## Task 2: `execution_providers_for(resolved)` alongside the string API

**Files:**
- Modify: `src/hydra_suite/runtime/compute_runtime.py`
- Test: `tests/runtime/test_execution_providers_for.py` (create)

**Interfaces:**
- Consumes: `ResolvedBackend` from `runtime.resolver`.
- Produces: `execution_providers_for(resolved: ResolvedBackend, include_cpu_fallback: bool = True) -> list[object]`. Later tasks (4, 12) call this; the old `derive_onnx_execution_providers(str)` stays until Task 12.

- [ ] **Step 1: Write the failing test** — assert TensorRT→`[Tensorrt…, CUDA…, CPU…]`, CoreML(on mps)→`[CoreML…, CPU…]`, torch/cpu→`[CPU…]`:

```python
# tests/runtime/test_execution_providers_for.py
from hydra_suite.runtime.resolver import ResolvedBackend
from hydra_suite.runtime.compute_runtime import (
    execution_providers_for, derive_onnx_execution_providers,
)

def _names(providers):
    return [p[0] if isinstance(p, tuple) else p for p in providers]

def test_tensorrt_matches_string_api():
    rb = ResolvedBackend("tensorrt", "cuda", False)
    assert _names(execution_providers_for(rb)) == _names(
        derive_onnx_execution_providers("tensorrt")
    )

def test_cpu_backend_is_cpu_only():
    rb = ResolvedBackend("torch", "cpu", False)
    assert _names(execution_providers_for(rb)) == ["CPUExecutionProvider"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/runtime/test_execution_providers_for.py -v`
Expected: FAIL — `execution_providers_for` not defined.

- [ ] **Step 3: Implement, delegating to the existing string logic for now**

In `compute_runtime.py`, add (keep `derive_onnx_execution_providers` untouched):

```python
def execution_providers_for(resolved, include_cpu_fallback: bool = True):
    """ONNX EP list keyed off a ResolvedBackend (the Gen-2 vocabulary).

    torch backends never run ONNX; tensorrt -> TRT-EP+CUDA-EP+cache,
    coreml -> CoreML-EP. CPU fallback appended per include_cpu_fallback.
    """
    if resolved.backend == "tensorrt":
        _compute_runtime = "tensorrt"
    elif resolved.backend == "coreml":
        _compute_runtime = "coreml"
    else:
        _compute_runtime = resolved.device  # cpu / cuda / mps -> no accel EP
    return derive_onnx_execution_providers(_compute_runtime, include_cpu_fallback)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/runtime/test_execution_providers_for.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/runtime/compute_runtime.py tests/runtime/test_execution_providers_for.py
git commit -m "feat(runtime): add ResolvedBackend-keyed execution_providers_for"
```

---

## Task 3: `RuntimeContext.resolved` field

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime.py` (dataclass ~lines 24-68; `from_config` ~lines 120-150)
- Test: `tests/core/inference/test_runtime_context_resolved.py` (create)

**Interfaces:**
- Consumes: `RuntimeResolver.resolve` (already called in `from_config`).
- Produces: `RuntimeContext.resolved: ResolvedBackend`. Tasks 4 and 5 read it.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/inference/test_runtime_context_resolved.py
from hydra_suite.core.inference.config import InferenceConfig
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.runtime.resolver import ResolvedBackend

def test_from_config_populates_resolved():
    cfg = InferenceConfig(runtime_tier="cpu")
    ctx = RuntimeContext.from_config(cfg)
    assert isinstance(ctx.resolved, ResolvedBackend)
    assert ctx.resolved.device == "cpu"
    assert ctx.device == "cpu"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/core/inference/test_runtime_context_resolved.py -v`
Expected: FAIL — `RuntimeContext` has no `resolved`.

- [ ] **Step 3: Add the field and populate it**

In the `RuntimeContext` dataclass add `resolved: "ResolvedBackend | None" = None` (import `ResolvedBackend` under `TYPE_CHECKING` + a runtime import in `from_config`). In `from_config`, capture the `resolved` object already produced by `resolver.resolve("obb")` and pass `resolved=resolved` into the constructor. Do NOT yet change any downstream reader.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/core/inference/test_runtime_context_resolved.py tests/runtime/test_resolver_golden.py -v`
Expected: PASS (golden unchanged).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/runtime.py tests/core/inference/test_runtime_context_resolved.py
git commit -m "feat(inference): carry ResolvedBackend on RuntimeContext"
```

---

## Task 4: Rewrite backend + stage predicates onto `ResolvedBackend`

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/obb.py:295`, `stages/pose.py:98,106,138`, `stages/cnn.py:35`, `stages/headtail.py:58`, `stages/crops.py:31`
- Modify: `src/hydra_suite/core/identity/classification/backend.py:425-562`, `classification/headtail.py:364,722,848`, `pose/backends/sleap.py:327-352`
- Test: `tests/core/inference/test_backend_runtime_predicates.py` (create)

**Interfaces:**
- Consumes: `RuntimeContext.resolved` (Task 3), `execution_providers_for` (Task 2).
- Produces: backend factories accept `ResolvedBackend`; internal device/accelerator predicates read `resolved.device` / `resolved.backend`.

**Transformation rules (apply verbatim at each site; the resolver never emits `onnx_cuda`/`onnx_cpu`/`onnx_coreml`, so these are faithful):**

| Current predicate | Replacement |
|---|---|
| `rt in ("cuda", "onnx_cuda", "tensorrt")` | `resolved.device == "cuda"` |
| `rt in ("mps", "onnx_coreml")` | `resolved.device == "mps"` |
| `self._compute_runtime == "coreml"` | `resolved.backend == "coreml"` |
| `derive_onnx_execution_providers(compute_runtime)` | `execution_providers_for(resolved)` |
| `_torch_device(compute_runtime)` returns cuda/mps/cpu | `resolved.device` |
| SLEAP `_normalize(req) -> onnx_*` then `derive_onnx_execution_providers` | `execution_providers_for(resolved)` |

**Boundary handling:** each stage currently does `compute_runtime = runtime_to_compute_runtime(runtime)`. Change to `resolved = runtime.resolved` and pass `resolved` into the backend factory (whose signature changes from `compute_runtime: str` to `resolved: ResolvedBackend`).

- [ ] **Step 1: Write the failing characterization test** — one assertion per backend that the resolved device/EP outcome matches the pre-refactor string outcome for all 5 producible ResolvedBackends:

```python
# tests/core/inference/test_backend_runtime_predicates.py
import pytest
from hydra_suite.runtime.resolver import ResolvedBackend
from hydra_suite.core.identity.classification.backend import _torch_device_for_resolved

FIVE = [
    (ResolvedBackend("torch", "cpu", False), "cpu"),
    (ResolvedBackend("torch", "mps", False), "mps"),
    (ResolvedBackend("torch", "cuda", False), "cuda"),
    (ResolvedBackend("tensorrt", "cuda", False), "cuda"),
    (ResolvedBackend("coreml", "mps", False), "mps"),
]

@pytest.mark.parametrize("resolved,expected_device", FIVE)
def test_torch_device_for_resolved(resolved, expected_device):
    assert _torch_device_for_resolved(resolved) == expected_device
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/core/inference/test_backend_runtime_predicates.py -v`
Expected: FAIL — `_torch_device_for_resolved` not defined.

- [ ] **Step 3: Apply the transformation rules** — rewrite `classification/backend.py` first: add `_torch_device_for_resolved(resolved) -> str: return resolved.device`; change `ClassifierBackend.__init__` to accept `resolved: ResolvedBackend`; replace the string predicates per the table; call `execution_providers_for(resolved)`. Repeat for `classification/headtail.py`, `pose/backends/sleap.py`. Then update the 5 stage files to pass `runtime.resolved`.

- [ ] **Step 4: Run backend + golden + full inference tests**

Run: `python -m pytest tests/core/inference tests/runtime/test_resolver_golden.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/stages src/hydra_suite/core/identity tests/core/inference/test_backend_runtime_predicates.py
git commit -m "refactor(inference): backends consume ResolvedBackend, not runtime strings"
```

---

## Task 5: Delete `runtime_to_compute_runtime` and `resolve_compute_runtime`

**Files:**
- Modify: `src/hydra_suite/core/inference/runtime.py` (delete `runtime_to_compute_runtime`; derive `cuda_mode`/`coreml_mode`/`tensor_on_cuda` from `resolved`)
- Modify: `src/hydra_suite/runtime/resolver.py:104-124` (delete `resolve_compute_runtime`)
- Modify: any residual importers (`stages/obb.py`, `pose.py`, `cnn.py`, `headtail.py` already switched in Task 4; confirm none remain)

**Interfaces:**
- Consumes: `RuntimeContext.resolved`.
- Produces: nothing new; removes two bridges.

- [ ] **Step 1: Confirm no live callers remain**

Run: `grep -rn "runtime_to_compute_runtime\|resolve_compute_runtime" src/ --include='*.py' | grep -v "def "`
Expected: only the definitions themselves (and doc comments). If any live caller remains, it was missed in Task 4 — fix it there first.

- [ ] **Step 2: Write/adjust the guard test** — assert mode flags derive from `resolved`:

```python
def test_mode_flags_from_resolved():
    from hydra_suite.core.inference.config import InferenceConfig
    from hydra_suite.core.inference.runtime import RuntimeContext
    ctx = RuntimeContext.from_config(InferenceConfig(runtime_tier="cpu"))
    assert ctx.cuda_mode is False and ctx.coreml_mode is False
```

Add to `tests/core/inference/test_runtime_context_resolved.py`.

- [ ] **Step 3: Delete both functions; make mode flags computed from `resolved`**

Replace the stored `cuda_mode`/`coreml_mode`/`tensor_on_cuda` construction so they are computed from `resolved` (e.g. `cuda_mode = resolved.device == "cuda" and resolved.backend == "torch"`), preserving the `__post_init__` guard. Delete `resolve_compute_runtime` from `resolver.py`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/inference tests/runtime -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/runtime.py src/hydra_suite/runtime/resolver.py tests/core/inference/test_runtime_context_resolved.py
git commit -m "refactor(runtime): delete runtime_to_compute_runtime + resolve_compute_runtime shims"
```

---

## Task 6: Retire the `COMPUTE_RUNTIME` params-dict write channel

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py`, `orchestrators/tracking.py`, `gui/workers/preview_worker.py`, `gui/workers/crops_worker.py`, `trackerkit/tracking_cache.py`, `data/dataset_generation.py`
- Test: `tests/core/inference/test_params_runtime_tier.py` (create)

**Interfaces:**
- Consumes: `RUNTIME_TIER` param (already emitted at `preview_worker.py:576`, read at `worker.py:1146`).
- Produces: params dicts carry only `RUNTIME_TIER`, no `COMPUTE_RUNTIME`.

- [ ] **Step 1: Write the failing test** — `build_inference_config_from_params` must honor `RUNTIME_TIER` with no `COMPUTE_RUNTIME` present:

```python
# tests/core/inference/test_params_runtime_tier.py
from hydra_suite.core.inference.config import build_inference_config_from_params

def test_runtime_tier_only_params():
    cfg = build_inference_config_from_params({"RUNTIME_TIER": "gpu_fast"})
    assert cfg.runtime_tier == "gpu_fast"

def test_missing_runtime_tier_defaults_cpu():
    cfg = build_inference_config_from_params({})
    assert cfg.runtime_tier == "cpu"
```

- [ ] **Step 2: Run to verify current behavior** (may already pass for line 1; line 2 depends on Task 10)

Run: `python -m pytest tests/core/inference/test_params_runtime_tier.py -v`
Expected: first test PASS, second may FAIL until Task 10 — mark `test_missing_runtime_tier_defaults_cpu` with `@pytest.mark.xfail(reason="Task 10")` for now.

- [ ] **Step 3: Remove `COMPUTE_RUNTIME` writes** — in each GUI/orchestrator/data file above, delete the line that sets `params["COMPUTE_RUNTIME"] = ...`. Keep the `RUNTIME_TIER` write. Do NOT touch `properties/cache.py` (Task 7) or `worker.py`'s reader yet.

- [ ] **Step 4: Run GUI-adjacent + inference tests**

Run: `python -m pytest tests/ -k "params or config or tracking" -v`
Expected: PASS (xfail stays xfail).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite tests/core/inference/test_params_runtime_tier.py
git commit -m "refactor(trackerkit): drop COMPUTE_RUNTIME param writes, keep RUNTIME_TIER"
```

---

## Task 7: Resolve tier at the `properties/cache.py` boundary; drop pose-flavor fields

**Files:**
- Modify: `src/hydra_suite/core/identity/properties/cache.py:173-196,256-271`
- Modify: pose-flavor threading in `trackerkit/gui/orchestrators/{session,tracking,config}.py`, `panels/{detection_panel,identity_panel}.py`, `trackerkit/gui/main_window.py`, `posekit/gui/main_window.py`, `trackerkit/cli_config.py`
- Test: `tests/core/identity/test_cache_runtime_payload.py` (create)

**Interfaces:**
- Consumes: `RUNTIME_TIER` param, `RuntimeResolver`, `detect_platform`.
- Produces: cache payload stores resolved `device`/`backend` (no `pose_runtime_flavor`/`pose_sleap_device`).

- [ ] **Step 1: Write the failing test**

```python
# tests/core/identity/test_cache_runtime_payload.py
from hydra_suite.core.identity.properties.cache import _runtime_payload_for_params

def test_payload_from_tier():
    payload = _runtime_payload_for_params({"RUNTIME_TIER": "cpu"})
    assert payload["device"] == "cpu"
    assert "pose_runtime_flavor" not in payload
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/core/identity/test_cache_runtime_payload.py -v`
Expected: FAIL — helper not defined.

- [ ] **Step 3: Implement** — add `_runtime_payload_for_params(params)` that reads `RUNTIME_TIER`, resolves via `RuntimeResolver(tier, detect_platform()).resolve("sleap_pose")`, and returns `{"device": resolved.device, "backend": resolved.backend}`. Replace the `COMPUTE_RUNTIME`/`POSE_SLEAP_DEVICE` reads at 173-196 and 256-271 with it. Delete `pose_runtime_flavor`/`pose_sleap_device` set/get in the listed GUI files.

- [ ] **Step 4: Run identity + GUI tests**

Run: `python -m pytest tests/core/identity tests/ -k "cache or pose or identity" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite tests/core/identity/test_cache_runtime_payload.py
git commit -m "refactor(identity): resolve runtime tier at cache boundary, drop pose-flavor strings"
```

---

## Task 8: Delete `derive_pose_runtime_settings` + `infer_compute_runtime_from_legacy` producers

**Files:**
- Modify: `src/hydra_suite/runtime/compute_runtime.py` (delete `derive_pose_runtime_settings`, `infer_compute_runtime_from_legacy`, `_runtime_from_pose_flavor`, `_best_auto_runtime`, `_best_explicit_onnx_runtime`)
- Modify: `posekit/gui/runtimes.py` (remove re-exports), `trackerkit/gui/orchestrators/{session,tracking,config}.py`, `panels/detection_panel.py`, `trackerkit/cli_config.py` (drop imports/uses)

**Interfaces:**
- Consumes: nothing new.
- Produces: removes 5 legacy functions. All uses were eliminated in Tasks 6–7; this task confirms and deletes.

- [ ] **Step 1: Confirm no live callers**

Run: `grep -rn "derive_pose_runtime_settings\|infer_compute_runtime_from_legacy\|_runtime_from_pose_flavor" src/ --include='*.py' | grep -v "def "`
Expected: empty (all removed in Tasks 6–7). If any remain, remove that call site first.

- [ ] **Step 2: Delete the five functions** from `compute_runtime.py` and their imports from `posekit/gui/runtimes.py` and the trackerkit files.

- [ ] **Step 3: Run full suite**

Run: `make pytest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
make format
git add src/hydra_suite
git commit -m "refactor(runtime): delete legacy pose-settings + legacy-config inference helpers"
```

---

## Task 9: Rewrite `CUDA_RUNTIMES`/`CPU_RUNTIMES` consumers onto `ResolvedBackend`

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:15-16,33`, `runtime.py:30,140`, `stages/obb.py:12`
- Test: extend `tests/core/inference/test_runtime_context_resolved.py`

**Interfaces:**
- Consumes: `ResolvedBackend.device`.
- Produces: removes the frozenset-membership runtime classification; `default_runtime: ComputeRuntime` field replaced by `resolved`.

- [ ] **Step 1: Locate every frozenset use**

Run: `grep -rn "CUDA_RUNTIMES\|CPU_RUNTIMES\|default_runtime" src/ --include='*.py'`
Record each site; each becomes a `resolved.device == "cuda"` / `== "cpu"` check.

- [ ] **Step 2: Write the guard test** — assert the replaced classification matches the frozensets for all 5 producible backends:

```python
def test_cuda_classification_matches_device():
    from hydra_suite.runtime.resolver import ResolvedBackend
    for rb in [ResolvedBackend("torch","cuda",False), ResolvedBackend("tensorrt","cuda",False)]:
        assert rb.device == "cuda"
    for rb in [ResolvedBackend("torch","cpu",False), ResolvedBackend("torch","mps",False), ResolvedBackend("coreml","mps",False)]:
        assert rb.device != "cuda"
```

- [ ] **Step 3: Replace each site** — swap `rt in CUDA_RUNTIMES` → `resolved.device == "cuda"`; remove the `default_runtime` field from `RuntimeContext`, using `resolved` instead. Delete the two frozensets and the `fast` set at `config.py:33`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/inference tests/runtime -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference tests/core/inference/test_runtime_context_resolved.py
git commit -m "refactor(inference): classify runtime by ResolvedBackend.device, drop frozensets"
```

---

## Task 10: Loud missing-tier error; drop per-stage `compute_runtime` fields + `ComputeRuntime` type

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (`_dict_to_config:437-447`, `build_inference_config_from_params:491-500`, per-stage dataclass fields at lines 89,105-106,217,231,245,257; the `ComputeRuntime` Literal at 11; `_collect_legacy_runtime_strings`; `migrate_runtime_to_tier`)
- Test: extend `tests/core/inference/test_params_runtime_tier.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `from_dict` raises `ValueError` on missing `runtime_tier`; `from_params` defaults absent tier to `"cpu"`; per-stage `compute_runtime` fields and `ComputeRuntime` gone.

- [ ] **Step 1: Write the tests** (replace the xfail from Task 6):

```python
def test_from_dict_missing_tier_raises():
    import pytest
    from hydra_suite.core.inference.config import _dict_to_config
    with pytest.raises(ValueError, match="migrate_runtime_config"):
        _dict_to_config({"obb": {}})  # no runtime_tier

def test_missing_runtime_tier_defaults_cpu():
    from hydra_suite.core.inference.config import build_inference_config_from_params
    assert build_inference_config_from_params({}).runtime_tier == "cpu"
```

(Remove the `xfail` marker added in Task 6.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/core/inference/test_params_runtime_tier.py -v`
Expected: FAIL — currently migrates instead of raising.

- [ ] **Step 3: Implement**

In `_dict_to_config`: replace the `migrate_runtime_to_tier(legacy)` fallback with
```python
raw_tier = d.get("runtime_tier")
if raw_tier is None:
    raise ValueError(
        "Config has no 'runtime_tier'. Run "
        "`python scripts/migrate_runtime_config.py <file>` to migrate legacy configs."
    )
```
In `build_inference_config_from_params`: replace the `migrate_runtime_to_tier({compute_runtime})` fallback with `runtime_tier = _raw_tier if _raw_tier in {"cpu","gpu","gpu_fast"} else "cpu"`. Delete every per-stage `compute_runtime` field and its `detect_compute_runtime`/`obb_compute_runtime` siblings, the `ComputeRuntime` Literal, `_collect_legacy_runtime_strings`, and `migrate_runtime_to_tier`. Remove now-dead `compute_runtime=...` kwargs in `from_params`'s config construction.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/core/inference tests/runtime -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/config.py tests/core/inference/test_params_runtime_tier.py
git commit -m "refactor(inference): loud missing-tier error, drop per-stage compute_runtime + ComputeRuntime"
```

---

## Task 11: One-shot migration script

**Files:**
- Create: `scripts/migrate_runtime_config.py`
- Create: `tests/scripts/test_migrate_runtime_config.py`
- Modify: `src/hydra_suite/resources/configs/ooceraea_biroi.json` (migrated output committed)

**Interfaces:**
- Consumes: nothing from `src/` runtime modules (self-contained; the deleted legacy logic lives here).
- Produces: CLI `python scripts/migrate_runtime_config.py <path...>` rewriting configs in place.

- [ ] **Step 1: Write the failing test** with representative legacy fixtures:

```python
# tests/scripts/test_migrate_runtime_config.py
import json
from scripts.migrate_runtime_config import migrate_config_dict

def test_per_stage_cuda_becomes_gpu():
    out = migrate_config_dict({"obb": {"direct": {"compute_runtime": "cuda"}}})
    assert out["runtime_tier"] == "gpu"
    assert "compute_runtime" not in out["obb"]["direct"]

def test_tensorrt_becomes_gpu_fast():
    out = migrate_config_dict({"pose": {"sleap": {"compute_runtime": "tensorrt"}}})
    assert out["runtime_tier"] == "gpu_fast"

def test_cpu_stays_cpu():
    out = migrate_config_dict({"obb": {"direct": {"compute_runtime": "cpu"}}})
    assert out["runtime_tier"] == "cpu"

def test_existing_tier_preserved():
    out = migrate_config_dict({"runtime_tier": "gpu_fast", "obb": {}})
    assert out["runtime_tier"] == "gpu_fast"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/scripts/test_migrate_runtime_config.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the script** — self-contained `migrate_config_dict(d) -> dict` embedding the old tier-inference rules (fast set `{onnx_cpu,onnx_cuda,onnx_coreml,tensorrt}` → `gpu_fast`; cuda-like/mps → `gpu`; cpu → `cpu`), collecting every per-stage `compute_runtime`, computing the tier, stripping the deprecated fields and `pose_runtime_flavor`/`pose_sleap_device`, and setting `runtime_tier`. `main()` reads JSON files from argv, applies it, writes back with a `.bak`.

- [ ] **Step 4: Run tests; migrate the bundled resource**

Run: `python -m pytest tests/scripts/test_migrate_runtime_config.py -v && python scripts/migrate_runtime_config.py src/hydra_suite/resources/configs/ooceraea_biroi.json`
Expected: PASS; the resource JSON gains `runtime_tier` and loses legacy fields.

- [ ] **Step 5: Commit**

```bash
make format
git add scripts/migrate_runtime_config.py tests/scripts/test_migrate_runtime_config.py src/hydra_suite/resources/configs/ooceraea_biroi.json
git commit -m "feat(scripts): one-shot legacy runtime-config migrator; migrate bundled config"
```

---

## Task 12: Delete legacy surface; rename module to `onnx_providers.py`

**Files:**
- Rename: `src/hydra_suite/runtime/compute_runtime.py` → `src/hydra_suite/runtime/onnx_providers.py`
- Modify importers: `core/identity/classification/backend.py`, `pose/backends/sleap.py`, `training/tiny_model.py`, `posekit/gui/runtimes.py`
- Delete from the module: `_normalize_runtime`, `runtime_label`, `CANONICAL_RUNTIMES`, `_pipeline_supports_runtime`, `supported_runtimes_for_pipeline`, `allowed_runtimes_for_pipelines`, `_onnx_available`, `_sleap_onnx_available`, `derive_onnx_execution_providers` (string API — now unused), `_best_explicit_onnx_runtime`

**Interfaces:**
- Consumes: nothing new.
- Produces: `onnx_providers.py` exposing only `execution_providers_for`, `COREML_PROVIDER_OPTIONS`, `_tensorrt_ep_cache_options`, dedup helpers.

- [ ] **Step 1: Confirm the string API + capability tables are unused**

Run: `grep -rn "derive_onnx_execution_providers\|CANONICAL_RUNTIMES\|runtime_label\|_normalize_runtime\|supported_runtimes_for_pipeline\|allowed_runtimes_for_pipelines" src/ --include='*.py' | grep -v "def \|compute_runtime.py"`
Expected: empty. Any hit is a missed migration — resolve before renaming.

- [ ] **Step 2: `git mv` and prune**

```bash
git mv src/hydra_suite/runtime/compute_runtime.py src/hydra_suite/runtime/onnx_providers.py
```
Delete the listed functions from the renamed file. Update the 4 importers to `from hydra_suite.runtime.onnx_providers import execution_providers_for`.

- [ ] **Step 3: Run full suite + dead-code check**

Run: `make pytest && make dead-code`
Expected: PASS; vulture reports no new unreferenced runtime helpers.

- [ ] **Step 4: Commit**

```bash
make format
git add -A src/hydra_suite/runtime src/hydra_suite/core src/hydra_suite/training src/hydra_suite/posekit
git commit -m "refactor(runtime): shrink+rename compute_runtime.py to onnx_providers.py, delete legacy surface"
```

---

## Task 13: GUI display cleanup; final verification

**Files:**
- Modify: `posekit/gui/runtimes.py`, any file still importing `runtime_label`
- Verify: `trackerkit/gui/panels/setup_panel.py`, `posekit/gui/main_window.py` selectors already use `tier_label`

**Interfaces:**
- Consumes: `runtime/resolver.py` `{available_tiers, tier_label}`.
- Produces: no runtime string vocabulary anywhere in GUI.

- [ ] **Step 1: Find residual string-vocab references**

Run: `grep -rn "runtime_label\|compute_runtime\|COMPUTE_RUNTIME\|pose_runtime_flavor\|pose_sleap_device" src/ --include='*.py'`
Expected: only genuinely-required survivors (e.g. `worker.py` reading `RUNTIME_TIER`). Investigate every hit; remove dead ones.

- [ ] **Step 2: Collapse `posekit/gui/runtimes.py`** to re-export only resolver symbols (`available_tiers`, `tier_label`, `RuntimeTier`, `PlatformInfo`, `detect_platform`).

- [ ] **Step 3: Full verification sweep**

Run: `make pytest && make lint-moderate && make dead-code`
Expected: all green; dead-code clean.

- [ ] **Step 4: Launch smoke test** — start each affected app to confirm runtime selectors populate:

Run: `trackerkit` and `posekit` (manual: confirm the runtime dropdown shows tier labels and tracking a short clip on CPU tier still runs).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(gui): resolver-only runtime vocabulary; final consolidation cleanup"
```

---

## Self-Review Notes

- **Spec coverage:** every spec phase maps to tasks — P0→T1, P1→T2/T3/T4, P2→T5, P3→T6/T7, P4→T9/T10, P5→T11, P6→T8/T12, P7→T13. The four critical-review additions are covered: missing-tier error (T10), `onnx_*`-in-backends predicate rewrite (T4), `COMPUTE_RUNTIME` params channel (T6), frozenset/type migration (T9).
- **Ordering:** producers of a value are retired only after all consumers are migrated (backends T4 before shim delete T5; params writes T6 before legacy-helper delete T8; per-stage fields T10 before script T11 before module prune T12).
- **Behavior preservation:** the Task-1 golden table is re-run in Tasks 3, 4, 5, 9 as the regression gate.
- **Risk residual:** the predicate-rewrite table in T4 is only faithful because the resolver cannot emit `onnx_cuda`/`onnx_cpu`; if a future resolver change adds an ONNX-EP backend, revisit T4's predicates.
