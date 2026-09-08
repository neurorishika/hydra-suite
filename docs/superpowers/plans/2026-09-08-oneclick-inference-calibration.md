# One-Click Inference Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make inference calibration a single explicit user action that produces a reusable profile, so every subsequent tracking run is pure inference that simply applies it — on any video, any backend (`cpu`/`gpu`/`gpu_fast`), forward or backward.

**Architecture:** Lift the existing in-run calibration entry point into a Qt-free `session.py` exposing `build_autotune_context` + `lookup` + `calibrate`. Split the conflated `eligible` ("may calibrate") and `allow_cached_reuse` ("may apply") permissions so applying works everywhere measuring cannot. Propagate the forward pass's effective vector to the backward pass through a file next to the inference cache, which is the only seam the GUI and headless paths share. The search, gates, and profile store are unchanged.

**Tech Stack:** Python 3.11+, PySide6 (`BaseWorker`/`BaseDialog`), pytest, conda env `hydra-mps` (this box) / `hydra-cuda` (mehek).

**Spec:** `docs/superpowers/specs/2026-09-08-inference-calibration-decoupling-design.md`

**Worktree:** `.worktrees/oneclick-calib` on branch `feat/oneclick-inference-calibration` (already created from `main` @ `501348c5`). All work happens there.

## Global Constraints

- **Baseline to protect:** 194 tests pass in `tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py`; 20 pass in `tests/test_get_parameters_dict_characterization.py tests/test_trackerkit_panels_smoke.py`. Measured on `hydra-mps` at `501348c5`. Any task that reduces these counts without an explicit, justified deletion is a regression.
- **Run the FULL autotune suite every task**, never a subset chosen by apparent relevance — the reflective contract guards break on any field change.
- **Never import from `legacy/`.** Core/Runtime/Data/Training/Utils must never import from an app layer (`trackerkit`, etc.). `session.py` lives in `core/` and must stay Qt-free.
- **`density_is_estimated` STAYS in the key.** It is load-bearing for the S2 two-record bridge (`store.py:243-300`). Do not remove it.
- **`TUNING_SCHEMA_VERSION` must NOT change.** Existing stored profiles must remain loadable.
- Format before every commit: `make format`. Lint gate: `make lint-moderate`.
- Activate the env first: `conda activate hydra-mps`. Kill stale `sleap`/`hydra` processes before any heavy run; never touch other processes.
- Commit after every task. Do not squash tasks together.

---

### Task 1: Extract the calibration entry point into a Qt-free session module

**Files:**
- Create: `src/hydra_suite/core/inference/autotune/session.py`
- Modify: `src/hydra_suite/core/inference/autotune/__init__.py`
- Test: `tests/test_inference_autotune_session.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `AutotuneContext` — frozen dataclass with fields `config: InferenceConfig`, `run_context: TrackingRunContext`, `params: dict`, `backend: str`, `probe`, `device_identity: tuple[str, str, str, int]`.
  - `build_autotune_context(config, params, *, video_path, frame_width, frame_height, start_frame, end_frame, realtime, cache_dir=None, use_cached_detections=False, cache_read_only_replay=False, should_cancel=lambda: False, status_callback=lambda _m: None) -> AutotuneContext`
  - `lookup(ctx: AutotuneContext) -> tuple[InferenceConfig, InferenceRuntimeOverlay, ResolveResult]`
  - `calibrate(ctx: AutotuneContext, *, budget_seconds: float) -> tuple[InferenceConfig, InferenceRuntimeOverlay, ResolveResult]`

This task is a **pure refactor**: byte-for-byte behaviour preservation. `worker.py` is not touched yet (Task 7 does that), so the suite must stay at 194 passing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_autotune_session.py`:

```python
"""Session-level entry points for autotune: shared context, lookup, calibrate."""

import pytest

from hydra_suite.core.inference.autotune import session


def test_module_exposes_the_three_entry_points():
    assert callable(session.build_autotune_context)
    assert callable(session.lookup)
    assert callable(session.calibrate)


def test_session_module_is_qt_free():
    """core/ must never import Qt. Guard it at the module source level."""
    from pathlib import Path

    src = Path(session.__file__).read_text(encoding="utf-8")
    assert "PySide6" not in src
    assert "QtCore" not in src


def test_build_context_is_pure_and_does_not_mutate_caller_params():
    """The ephemeral params dict must be a copy: worker.py:147 does dict(params)
    and then mutates. A caller's dict must never gain autotune-only keys."""
    from pathlib import Path
    src = Path(session.__file__).read_text(encoding="utf-8")
    assert "ephemeral_params = dict(params)" in src
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda activate hydra-mps
python -m pytest tests/test_inference_autotune_session.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.inference.autotune.session'`.

- [ ] **Step 3: Create the module by lifting `worker.py:104-248`**

Create `src/hydra_suite/core/inference/autotune/session.py`. Copy the body of `_resolve_inference_autotune_before_load` verbatim, with exactly these changes: drop the `if config.inference_autotune.mode == "off": return` early return (callers decide), split the tail into `lookup`/`calibrate`, and return a context object instead of resolving inline.

```python
"""Qt-free entry points for inference calibration and profile lookup.

Split out of ``core/tracking/worker.py`` so calibration is an explicit act a
user triggers, not a side effect of starting a tracking run. Both entry points
build their fingerprint through ``build_autotune_context`` so a profile
produced by ``calibrate`` is found by ``lookup`` -- key agreement between the
two is the whole correctness argument for this feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hydra_suite.core.inference.config import InferenceConfig


@dataclass(frozen=True, slots=True)
class AutotuneContext:
    """Everything both entry points need, derived exactly once."""

    config: InferenceConfig
    run_context: Any  # TrackingRunContext
    params: dict
    backend: str
    probe: Any  # RuntimeResourceProbe
    device_identity: tuple[str, str, str, int]


def build_autotune_context(
    config: InferenceConfig,
    params: dict,
    *,
    video_path: str,
    frame_width: int,
    frame_height: int,
    start_frame: int,
    end_frame: int,
    realtime: bool,
    cache_dir=None,
    use_cached_detections: bool = False,
    cache_read_only_replay: bool = False,
    should_cancel: Callable[[], bool] = lambda: False,
    status_callback: Callable[[str], None] = lambda _message: None,
) -> AutotuneContext:
    """Build the one context both ``lookup`` and ``calibrate`` fingerprint from."""

    from hydra_suite.core.inference.autotune.device import probe_runtime_resources
    from hydra_suite.core.inference.autotune.integration import (
        TrackingRunContext,
        sample_detection_workload,
    )
    from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform
    from hydra_suite.runtime.resource_budget import AcceleratorKind

    platform_info = detect_platform()
    resolved = RuntimeResolver(config.runtime_tier, platform_info).resolve("obb")
    kind = {
        "cuda": AcceleratorKind.CUDA,
        "mps": AcceleratorKind.MPS,
        "cpu": AcceleratorKind.CPU,
    }[resolved.device]
    probe = probe_runtime_resources(kind)
    ephemeral_params = dict(params)
    ephemeral_params["INFERENCE_AUTOTUNE_DRIVER_VERSION"] = probe.driver_version
    prior_counts = ()
    if cache_dir:
        prior_counts = sample_detection_workload(
            cache_dir, start_frame=start_frame, end_frame=end_frame
        )
        if prior_counts and "INFERENCE_AUTOTUNE_DETECTION_COUNTS" not in ephemeral_params:
            ephemeral_params["INFERENCE_AUTOTUNE_DETECTION_COUNTS"] = prior_counts
            ephemeral_params["INFERENCE_AUTOTUNE_CROP_COUNTS"] = prior_counts
    cached_fields = frozenset()
    if use_cached_detections and prior_counts:
        cached_fields = frozenset(("detection_batch_size", "slice_tile_batch_size"))
        ephemeral_params["RESULT_CACHE_STAGE_MASK"] = ("detector",)
    run_context = TrackingRunContext(
        video_path=video_path,
        params=ephemeral_params,
        frame_width=max(1, int(frame_width)),
        frame_height=max(1, int(frame_height)),
        channels=3,
        decoder_mode="nvdec" if config.runtime_tier == "gpu_fast" else "opencv",
        execution_mode=(
            "cache_replay"
            if cache_read_only_replay
            else ("realtime" if realtime else "batch")
        ),
        start_frame=start_frame,
        end_frame=end_frame,
        cached_fields=cached_fields,
        contention_detected=probe.contention_detected,
        thermal_throttled=probe.thermal_throttled,
        should_cancel=should_cancel,
        status_callback=status_callback,
    )
    return AutotuneContext(
        config=config,
        run_context=run_context,
        params=ephemeral_params,
        backend=resolved.backend,
        probe=probe,
        device_identity=(
            probe.device_uuid,
            probe.device_model,
            probe.compute_capability,
            int(probe.observation.total_accelerator_bytes or 0),
        ),
    )


def lookup(ctx: AutotuneContext):
    """Find and apply a validated profile. Never measures, never claims a lock."""

    from hydra_suite.core.inference.autotune.integration import (
        resolve_tracking_inference_config,
    )

    return resolve_tracking_inference_config(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
        trial_executor=None,
    )


def calibrate(ctx: AutotuneContext, *, budget_seconds: float):
    """Measure and persist a profile for this exact context."""

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
        resolve_tracking_inference_config,
    )
    from hydra_suite.core.inference.autotune.sidecar import (
        ARTIFACT_BUILD_ALLOWANCE_SECONDS,
        ContainedTrialExecutor,
        SidecarTrialSpec,
    )

    preflight = build_tracking_autotune_request(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )
    artifact_batch_size = max(
        (
            value
            for field in ("detection_batch_size", "slice_tile_batch_size")
            for value in preflight.planner.values_for(field, preflight.baseline)
        ),
        default=preflight.baseline.detection_batch_size,
    )
    if ctx.backend == "tensorrt":
        ctx.params["INFERENCE_AUTOTUNE_TENSORRT_PROFILE_BATCH_SIZE"] = artifact_batch_size
    executor = ContainedTrialExecutor(
        SidecarTrialSpec(
            video_path=ctx.run_context.video_path,
            params=ctx.params,
            observation=ctx.probe.observation,
            resource_probe=ctx.probe,
            start_frame=ctx.run_context.start_frame,
            end_frame=ctx.run_context.end_frame,
            budget_seconds=budget_seconds,
            runtime_artifact_batch_size=(
                artifact_batch_size if ctx.backend == "tensorrt" else None
            ),
            artifact_build_allowance_seconds=(
                ARTIFACT_BUILD_ALLOWANCE_SECONDS
                if ctx.backend in ("tensorrt", "coreml")
                else 0.0
            ),
        )
    )
    return resolve_tracking_inference_config(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
        trial_executor=executor,
    )
```

Add to `src/hydra_suite/core/inference/autotune/__init__.py`:

```python
from hydra_suite.core.inference.autotune.session import (  # noqa: F401
    AutotuneContext,
    build_autotune_context,
    calibrate,
    lookup,
)
```

- [ ] **Step 4: Run the new test and the full autotune suite**

```bash
python -m pytest tests/test_inference_autotune_session.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

Expected: new file PASS; suite **194 passed + 3 new = 197 passed**. `worker.py` is untouched, so nothing else may change.

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/autotune/session.py \
        src/hydra_suite/core/inference/autotune/__init__.py \
        tests/test_inference_autotune_session.py
git commit -m "refactor(autotune): extract Qt-free session entry points

build_autotune_context is the single fingerprint source for both lookup
and calibrate; key agreement between them is what makes a user-triggered
calibration findable by a later run. Pure lift from worker.py:104-248."
```

---

### Task 2: Split "may calibrate" from "may apply"

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/integration.py:601-635`
- Test: `tests/test_inference_autotune_eligibility_split.py`
- Modify: `tests/test_inference_autotune_integration.py:258` (delete the CUDA-only assertion)

**Interfaces:**
- Consumes: `build_tracking_autotune_request` from Task 1's context.
- Produces: `AutotuneRequest.eligible` now means *may calibrate* only; `AutotuneRequest.allow_cached_reuse` means *may apply* and is cleared only by `record` mode (removed in Task 3) and by baseline-admission failure.

This is the change that delivers "works on all backends". Today `integration.py:604-624` clears **both** flags for realtime, `cache_replay`, and non-CUDA, so applying is impossible in exactly the configurations the user runs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_autotune_eligibility_split.py`:

```python
"""`eligible` gates measuring; `allow_cached_reuse` gates applying. They differ."""

import pytest

from hydra_suite.core.inference.autotune.integration import (
    build_tracking_autotune_request,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind
from tests.autotune_helpers import make_request_inputs


@pytest.mark.parametrize(
    "kind", [AcceleratorKind.MPS, AcceleratorKind.CPU, AcceleratorKind.CUDA]
)
def test_apply_is_permitted_on_every_accelerator(kind):
    """A validated profile must be applicable on cpu/mps/cuda alike. The gain and
    equivalence gates -- not the device name -- establish that it is safe."""
    request = build_tracking_autotune_request(**make_request_inputs(kind=kind))
    assert request.allow_cached_reuse is True


def test_realtime_may_apply_but_may_not_calibrate():
    request = build_tracking_autotune_request(
        **make_request_inputs(execution_mode="realtime")
    )
    assert request.eligible is False
    assert request.allow_cached_reuse is True


def test_cache_replay_may_apply_but_may_not_calibrate():
    """A backward pass IS a cache_replay. It must apply the forward vector or the
    detection cache key will not match -- the measured courtship abort."""
    request = build_tracking_autotune_request(
        **make_request_inputs(execution_mode="cache_replay")
    )
    assert request.eligible is False
    assert request.allow_cached_reuse is True


def test_failed_baseline_admission_blocks_both():
    """If the baseline vector does not fit in memory, applying it is unsafe too."""
    request = build_tracking_autotune_request(
        **make_request_inputs(available_accelerator_bytes=1)
    )
    assert request.eligible is False
    assert request.allow_cached_reuse is False
```

Extend `tests/autotune_helpers.py` with a `make_request_inputs(**overrides) -> dict` factory returning the keyword arguments `build_tracking_autotune_request` needs (config, context, observation, backend, device_identity), defaulting to a CUDA batch context and applying overrides for `kind`, `execution_mode`, and `available_accelerator_bytes`. Read the existing helpers in that file and follow their construction pattern rather than inventing new fixtures.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_inference_autotune_eligibility_split.py -q
```

Expected: FAIL — MPS/CPU and realtime/cache_replay cases assert `allow_cached_reuse is True` but get `False`.

- [ ] **Step 3: Rewrite the eligibility block**

Replace `integration.py:601-635` with:

```python
    # Two DIFFERENT permissions, deliberately no longer computed together.
    #
    #   eligible            -- may we MEASURE? Expensive, environment-sensitive,
    #                          and meaningless under contention or replay.
    #   allow_cached_reuse  -- may we APPLY an already-validated profile? Nearly
    #                          free, and for a cache_replay (backward) pass it is
    #                          REQUIRED: the detection cache was written at the
    #                          forward pass's batch size, so refusing to apply
    #                          that same vector is what made backward runs abort.
    #
    # Only a baseline that does not fit in memory blocks both: a vector we cannot
    # admit is not one we can safely apply either.
    eligible = True
    allow_cached_reuse = True
    eligibility_reason = None
    if context.execution_mode == "realtime":
        eligible = False
        eligibility_reason = "realtime inference is not tunable"
    elif context.execution_mode == "cache_replay":
        eligible = False
        eligibility_reason = "all inference stages are satisfied by reusable caches"
    elif context.contention_detected:
        eligible = False
        eligibility_reason = "another accelerator job is active"
    elif context.thermal_throttled:
        eligible = False
        eligibility_reason = "accelerator is thermally throttled"
    else:
        admission = planner.admit(baseline)
        if not admission.admitted:
            eligible = False
            allow_cached_reuse = False
            eligibility_reason = f"baseline admission failed: {admission.reason}"
```

Note what is gone: the `policy.mode == "automatic" and not CUDA` branch (previously `:618-624`) and the `policy.mode == "record"` override (previously `:631-635`). Task 3 removes `record` from the vocabulary entirely.

- [ ] **Step 4: Delete the obsolete CUDA assertion and re-run**

Delete the test at `tests/test_inference_autotune_integration.py:258` that asserts the string `"automatic inference tuning is validated only for CUDA"`; that policy no longer exists.

```bash
python -m pytest tests/test_inference_autotune_eligibility_split.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

Expected: new file PASS (4 tests). Suite: 197 − 1 deleted + 4 new = **200 passed**. Investigate any other failure; do not adjust an assertion to make it green without understanding what it was protecting.

- [ ] **Step 5: Format and commit**

```bash
make format
git add src/hydra_suite/core/inference/autotune/integration.py \
        tests/test_inference_autotune_eligibility_split.py \
        tests/test_inference_autotune_integration.py tests/autotune_helpers.py
git commit -m "feat(autotune): separate 'may calibrate' from 'may apply'

Applying a validated profile is now permitted on cpu/mps/cuda, in
realtime, and on a cache_replay pass. Measuring stays guarded. This is
what makes a tuned profile usable outside forward-only CUDA runs, and it
is a precondition for backward-pass support."
```

---

### Task 3: Collapse the mode vocabulary to lookup/calibrate

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/coordinator.py:99-260`
- Modify: `src/hydra_suite/core/inference/config.py:94-103, 1313-1315`
- Modify: `src/hydra_suite/core/inference/autotune/integration.py` (`policy.mode` uses)
- Test: `tests/test_inference_autotune_intent.py`

**Interfaces:**
- Consumes: Task 2's `eligible`/`allow_cached_reuse` split.
- Produces: `AutotuneRequest.mode` ∈ `{"lookup", "calibrate"}`. `InferenceAutotunePolicy.mode` accepts the same two values. `"off"` is no longer a mode — the *caller* decides whether to call `lookup` at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_autotune_intent.py`:

```python
"""Intent replaces the off/record/automatic mode vocabulary."""

import time

import pytest

from hydra_suite.core.inference.autotune.coordinator import AutotuneCoordinator
from hydra_suite.core.inference.autotune.models import ProfileState
from tests.autotune_helpers import make_coordinator, make_incomplete_profile, make_request


def test_lookup_without_executor_reports_unavailable_on_a_miss():
    coordinator, store = make_coordinator(trial_executor=None)
    result = coordinator.resolve(make_request(mode="lookup"))
    assert result.overlay.status == "unavailable"
    assert store.saves == []
    assert store.claims == []


def test_lookup_serves_a_hit_even_when_ineligible():
    """Eligibility gates measuring only. A backward pass is ineligible and must
    still receive the forward pass's vector."""
    coordinator, _ = make_coordinator(trial_executor=None, cached_state=ProfileState.VALIDATED)
    result = coordinator.resolve(make_request(mode="lookup", eligible=False))
    assert result.overlay.status == "cache_hit"


def test_explicit_calibrate_bypasses_the_24h_negative_cache():
    """A human clicked the button. The negative cache exists to stop a RUN
    silently re-paying a failed budget, which does not apply here."""
    coordinator, _ = make_coordinator(cached_profile=make_incomplete_profile())
    result = coordinator.resolve(make_request(mode="calibrate"))
    assert result.overlay.status != "deferred_due_to_prior_failure"


def test_lookup_still_honours_the_negative_cache():
    coordinator, _ = make_coordinator(cached_profile=make_incomplete_profile())
    result = coordinator.resolve(make_request(mode="lookup"))
    assert result.overlay.status == "deferred_due_to_prior_failure"


def test_record_mode_is_gone():
    with pytest.raises((ValueError, KeyError)):
        make_request(mode="record")
```

Add the `make_coordinator` / `make_request` / `make_incomplete_profile` helpers to `tests/autotune_helpers.py`, with a fake store recording `saves` and `claims` lists.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_inference_autotune_intent.py -q
```

Expected: FAIL — `mode="lookup"` is rejected by the current validation.

- [ ] **Step 3: Rewrite `AutotuneCoordinator.resolve`**

In `coordinator.py`, replace the head of `resolve` (currently `:101-168`) with:

```python
    def resolve(self, request: AutotuneRequest) -> ResolveResult:
        """Apply a validated profile, or measure one when explicitly asked."""
        cached = self.store.load(request.key)
        if (
            request.allow_cached_reuse
            and cached is not None
            and cached.state is ProfileState.VALIDATED
        ):
            return self._reuse(request, cached, status="cache_hit")
        if request.mode == "lookup":
            # A run never measures. Terminating here -- BEFORE the eligibility
            # and negative-cache branches -- keeps "no profile yet" a single,
            # honest status instead of leaking calibration-only vocabulary
            # ("deferred_due_to_contention") into a run that was never going
            # to calibrate anyway.
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="unavailable",
                    reason="no validated profile for this configuration",
                )
            )
        if not request.eligible:
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="deferred_due_to_contention",
                    reason=request.eligibility_reason
                    or "live resource eligibility failed",
                )
            )
        if self.trial_executor is None:
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="unavailable",
                    reason="no contained calibration executor is configured",
                )
            )
```

Delete the two `record` blocks (`:117-129` and `:190-202`) and the `INCOMPLETE` early return (`:130-151`) from `resolve`. Move the `INCOMPLETE` check into the `lookup` terminal above, so it reads:

```python
        if request.mode == "lookup":
            if (
                cached is not None
                and cached.state is ProfileState.INCOMPLETE
                and (time.time_ns() - cached.last_validation_unix_ns)
                < INCOMPLETE_RETRY_SECONDS * 1e9
            ):
                reason = (
                    cached.invalidation_reason
                    or "a prior calibration attempt did not complete"
                )
                return ResolveResult(
                    InferenceRuntimeOverlay.baseline(
                        request.baseline,
                        status="deferred_due_to_prior_failure",
                        reason=reason,
                    ),
                    cached,
                )
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="unavailable",
                    reason="no validated profile for this configuration",
                )
            )
```

Inside the single-flight block, delete the `record` re-check (`:190-202`), keeping only the `cache_hit_after_wait` re-check.

Two validation sites must move together, or one will silently reject the new vocabulary:

- `coordinator.py:40` (`mode: str = "off"  # off, record, automatic`) and its `__post_init__` check at `:65-66`.
- `config.py:94` (`mode: Literal["off", "record", "automatic"] = "off"`) and its `__post_init__` check at `:100-103`.

Both become `{"lookup", "calibrate"}`, defaulting to `"lookup"`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_autotune_intent.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

Expected: new file PASS (5). Existing tests that construct `mode="automatic"`/`"record"` must be **updated to the new vocabulary**, not deleted — `tests/test_inference_autotune_integration.py:84,152,177,202,226,233` and `tests/test_inference_autotune_sidecar.py:60`. Report the final count.

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src/hydra_suite/core/inference tests/
git commit -m "feat(autotune): collapse off/record/automatic to lookup/calibrate

A run looks up; a user calibrates. 'record' was only ever 'calibrate
without applying', which the apply checkbox now expresses. Lookup
terminates in 'unavailable' rather than leaking calibration-only
statuses, and explicit calibration bypasses the 24h negative cache."
```

---

### Task 4: Make the TensorRT profile id independent of live free memory

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/session.py` (`calibrate`)
- Modify: `src/hydra_suite/core/inference/autotune/candidates.py` (add a static accessor)
- Test: `tests/test_inference_autotune_fingerprint_stability.py`

**Interfaces:**
- Consumes: Task 1's `calibrate`.
- Produces: `CandidatePlanner.static_max_for(field: str, baseline) -> int` — the candidate-space maximum ignoring live memory admission.

Today `session.calibrate` computes `artifact_batch_size` from `planner.values_for(...)`, which filters through `admit()` and therefore reads live `available_host_bytes`/`available_accelerator_bytes` (`candidates.py:199-227`). That value folds into `tensorrt_profile_id` (`integration.py:175-186`), so on `gpu_fast` the **profile key changes with how much memory happens to be free** — a calibration run and a later lookup on a busier machine will not match.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_autotune_fingerprint_stability.py`:

```python
"""The profile key must not depend on transient machine state."""

from hydra_suite.core.inference.autotune import session
from tests.autotune_helpers import make_tensorrt_context

LOW_VRAM = 2 * 1024**3
HIGH_VRAM = 48 * 1024**3


def test_tensorrt_digest_identical_under_memory_pressure():
    """gpu_fast folds an artifact batch size into tensorrt_profile_id. If that
    size is admission-filtered it tracks free VRAM, so the same machine running
    the same config yields a different profile key when it happens to be busy --
    and a calibrated profile becomes permanently unfindable."""
    low = session.calibration_key_digest(
        make_tensorrt_context(available_accelerator_bytes=LOW_VRAM)
    )
    high = session.calibration_key_digest(
        make_tensorrt_context(available_accelerator_bytes=HIGH_VRAM)
    )
    assert low == high


def test_non_tensorrt_digest_also_stable():
    low = session.calibration_key_digest(
        make_tensorrt_context(backend="torch", available_accelerator_bytes=LOW_VRAM)
    )
    high = session.calibration_key_digest(
        make_tensorrt_context(backend="torch", available_accelerator_bytes=HIGH_VRAM)
    )
    assert low == high
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_inference_autotune_fingerprint_stability.py -q
```

Expected: FAIL — `session.calibration_key_digest` does not exist, then (once added) the TensorRT digests differ.

- [ ] **Step 3: Add the static accessor and a digest helper**

In `candidates.py`, alongside `values_for`:

```python
    def static_max_for(self, field: str, baseline) -> int:
        """Largest candidate value for *field*, ignoring live memory admission.

        ``values_for`` filters through ``admit()``, which reads the live
        resource observation. Anything that feeds the PROFILE KEY must not,
        or the key drifts with free memory and a stored profile becomes
        unfindable on a busier machine.
        """
        return max(self.candidate_space(field, baseline), default=getattr(baseline, field))
```

Implement `candidate_space` if it does not already exist by extracting the unfiltered candidate enumeration that `values_for` currently filters; read `candidates.py:199-227` and reuse its generator rather than duplicating the value list.

In `session.py`, replace the `artifact_batch_size` computation in `calibrate` with:

```python
    artifact_batch_size = max(
        (
            preflight.planner.static_max_for(field, preflight.baseline)
            for field in ("detection_batch_size", "slice_tile_batch_size")
        ),
        default=preflight.baseline.detection_batch_size,
    )
```

and add:

```python
def calibration_key_digest(ctx: AutotuneContext) -> str:
    """The profile-key digest this context will calibrate and look up under."""

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )

    request = build_tracking_autotune_request(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )
    return request.key.digest
```

Add `make_tensorrt_context` to `tests/autotune_helpers.py`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_autotune_fingerprint_stability.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

Expected: new file PASS (2). Full suite green.

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src/hydra_suite/core/inference tests/
git commit -m "fix(autotune): keep the TensorRT profile key free of live memory

artifact_batch_size fed tensorrt_profile_id through an admission filter
that reads free VRAM, so a gpu_fast profile key drifted with machine
load and could not be found again. Derive it from the static candidate
space instead."
```

---

### Task 5: Rename the config/params surface and migrate

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/schemas.py:51-53, 89-94, 117-143`
- Modify: `src/hydra_suite/trackerkit/engine_params.py:541-575, 1240-1242`
- Modify: `src/hydra_suite/core/inference/config.py:1313-1315`
- Modify: `src/hydra_suite/trackerkit/cli_config.py:216-247`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py:311-325, 1720`
- Modify: `tools/equivalence/autotune_plumbing_probe.py:76`, `tools/equivalence/run_matrix.sh:127`
- Test: `tests/test_trackerkit_autotune_config_migration.py`
- Regenerate: `tests/data/get_parameters_dict_golden/{ant_cnn_identity,fly_obb}.json`

**Interfaces:**
- Produces: config field `apply_tuned_inference: bool`; engine param `APPLY_TUNED_INFERENCE: bool`. `inference_autotune_budget_seconds` and `inference_autotune_manual_fields` are retained; the budget now applies to calibration.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackerkit_autotune_config_migration.py`:

```python
"""Legacy mode strings migrate to the apply boolean."""

import pytest

from hydra_suite.trackerkit.config.schemas import TrackerConfig


@pytest.mark.parametrize(
    "legacy,expected",
    [("automatic", True), ("record", True), ("off", False)],
)
def test_legacy_mode_migrates_to_boolean(legacy, expected):
    cfg = TrackerConfig.from_dict({"inference_autotune_mode": legacy})
    assert cfg.apply_tuned_inference is expected


def test_absent_key_defaults_to_disabled():
    cfg = TrackerConfig.from_dict({})
    assert cfg.apply_tuned_inference is False


def test_new_key_wins_over_legacy():
    cfg = TrackerConfig.from_dict(
        {"inference_autotune_mode": "off", "apply_tuned_inference": True}
    )
    assert cfg.apply_tuned_inference is True


def test_roundtrip_emits_only_the_new_key():
    cfg = TrackerConfig.from_dict({"inference_autotune_mode": "automatic"})
    data = cfg.to_dict()
    assert data["apply_tuned_inference"] is True
    assert "inference_autotune_mode" not in data
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_trackerkit_autotune_config_migration.py -q
```

Expected: FAIL — `AttributeError: 'TrackerConfig' object has no attribute 'apply_tuned_inference'`.

- [ ] **Step 3: Apply the rename across every site**

The dataclass is `TrackerConfig`. Replace `inference_autotune_mode: str = "off"` at `schemas.py:51`; update the `to_dict` emission at `:89`; and in `from_dict`, replace the `inference_autotune_mode=str(...)` argument at `:135` with a migrating read:

```python
        legacy_mode = str(data.get("inference_autotune_mode", "off")).strip().lower()
        apply_tuned_inference = bool(
            data.get("apply_tuned_inference", legacy_mode in {"automatic", "record"})
        )
```

In `engine_params.py`, delete the `autotune_mode` derivation and its three-value validation (`:541-549`) and emit at `:1240`:

```python
        "APPLY_TUNED_INFERENCE": bool(
            _cfg_get(cfg, "apply_tuned_inference", default=False)
        ),
```

Keep the manual-field normalization (`:550-559`), the budget clamp (`:560-576`), and the `INFERENCE_AUTOTUNE_MANUAL_FIELDS` / `_BUDGET_SECONDS` / `_SINGLEFLIGHT_WAIT_SECONDS` / `_STAGE_SHARES` emissions at `:1241-1256` exactly as they are — the budget now funds calibration instead of an in-run search, but nothing about its plumbing changes.

In `core/inference/config.py:1313`, replace the mode-string read. The policy mode is now **always** `"lookup"` when built from params — `calibrate` is set only by `session.calibrate`, never derived from a config file. Whether a run consults the store at all is the caller's decision from `APPLY_TUNED_INFERENCE`, not a policy value:

```python
    autotune_mode = "lookup"
```

Delete the now-unused `raw_autotune_mode` local and any validation branch it fed.

Update `cli_config.apply_inference_autotune_override`, `gui/orchestrators/config.py`, and both `tools/equivalence` sites to the new key.

- [ ] **Step 4: Regenerate the goldens and run tests**

The goldens at `tests/data/get_parameters_dict_golden/{ant_cnn_identity,fly_obb}.json` contain `"INFERENCE_AUTOTUNE_MODE": "off"`.

```bash
python -m pytest tests/test_get_parameters_dict_characterization.py -q   # expect FAIL first
```

Regenerate them using whatever mechanism that test documents (read its docstring/header for the regeneration command — do not hand-edit the JSON), then **review the diff**: the only change may be `INFERENCE_AUTOTUNE_MODE: "off"` → `APPLY_TUNED_INFERENCE: false`. Any other key moving means something unintended changed.

```bash
python -m pytest tests/test_trackerkit_autotune_config_migration.py -q
python -m pytest tests/test_get_parameters_dict_characterization.py tests/test_trackerkit_panels_smoke.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src tests tools
git commit -m "feat(trackerkit): apply_tuned_inference replaces the mode string

One boolean: does this run apply a tuned profile? Legacy record and
automatic both migrate to true. Golden params regenerated; the diff is
the single renamed key."
```

---

### Task 6: Carry the sidecar recursion guard through the rename

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/sidecar.py:140`
- Modify: `src/hydra_suite/core/inference/autotune/sidecar_child.py:141`
- Modify: `tests/test_inference_autotune_sidecar.py:60,69`

**Interfaces:** consumes Task 5's `APPLY_TUNED_INFERENCE`.

A trial child must never apply a profile: if it did, a measurement would be taken against tuned settings rather than the vector under test, corrupting the baseline block and therefore the gain calculation. Today that is enforced by setting `INFERENCE_AUTOTUNE_MODE="off"`, which becomes a no-op string after Task 5.

- [ ] **Step 1: Write the failing test**

Extend `tests/test_inference_autotune_sidecar.py`:

```python
def test_trial_child_never_applies_a_tuned_profile():
    """A child that applied a profile would measure the profile, not the
    candidate -- silently invalidating every gain number."""
    params = {"APPLY_TUNED_INFERENCE": True, "MAX_TARGETS": 25}
    output = sidecar.apply_settings_to_params(params, _any_settings())
    assert output["APPLY_TUNED_INFERENCE"] is False
```

The real function is `sidecar.apply_settings_to_params(params, settings, *, runtime_artifact_batch_size=None)` (`sidecar.py:131`); build `_any_settings()` from `InferenceTuningSettings.from_config` or the literal constructor used elsewhere in that test file. The child-side twin is the `run_params` dict at `sidecar_child.py:135-149`; add a parallel assertion for it.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_inference_autotune_sidecar.py -q
```

Expected: FAIL — the child still emits `INFERENCE_AUTOTUNE_MODE`.

- [ ] **Step 3: Update both guards**

`sidecar.py:140`:

```python
    result["APPLY_TUNED_INFERENCE"] = False
```

`sidecar_child.py:141`:

```python
            "APPLY_TUNED_INFERENCE": False,
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_autotune_sidecar.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src/hydra_suite/core/inference/autotune tests/test_inference_autotune_sidecar.py
git commit -m "fix(autotune): keep the trial-child recursion guard alive

The guard was a mode string the rename would have made inert, letting a
trial apply a profile mid-measurement and corrupt its own baseline."
```

---

### Task 7: Make the tracking run lookup-only, preserving the throughput bridge

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:104-248` (delete), `:993-995`, `:726-728` (delete), `:1332-1415`, `:4964-5006`
- Modify: `src/hydra_suite/trackerkit/gui/workers/tracking_worker.py:71-73` (delete)
- Test: `tests/test_worker_inference_lookup_only.py`
- Modify: `tests/test_worker_real_inference_integration.py:496,551,742,802`

**Interfaces:** consumes `session.build_autotune_context` / `session.lookup`.

Two things must both hold: the run never constructs a trial executor, **and** `observe_production_throughput` keeps firing. The second is not optional — it writes the measured-key record that the S2 two-record bridge (`store.py:243-300`) depends on. It is currently gated on `profiler.enabled`, which `:993-995` derives from the mode string; the rename would silently kill it and every cached run would miss forever.

- [ ] **Step 1: Write the failing test**

Create `tests/test_worker_inference_lookup_only.py`:

```python
"""A tracking run applies profiles. It never measures them."""

import pathlib

import pytest


def test_worker_never_constructs_a_trial_executor():
    src = pathlib.Path(
        "src/hydra_suite/core/tracking/worker.py"
    ).read_text(encoding="utf-8")
    assert "ContainedTrialExecutor" not in src
    assert "SidecarTrialSpec" not in src


def test_worker_calls_session_lookup_not_a_local_resolver():
    src = pathlib.Path(
        "src/hydra_suite/core/tracking/worker.py"
    ).read_text(encoding="utf-8")
    assert "_resolve_inference_autotune_before_load" not in src
    assert "session.lookup" in src or "autotune_session.lookup" in src


def test_profiler_enabling_no_longer_depends_on_a_mode_string():
    """observe_production_throughput is gated on profiler.enabled and writes the
    measured-key record the S2 bridge needs. If the gate reads a key that no
    longer exists, the bridge dies silently."""
    src = pathlib.Path(
        "src/hydra_suite/core/tracking/worker.py"
    ).read_text(encoding="utf-8")
    assert "INFERENCE_AUTOTUNE_MODE" not in src
    assert "APPLY_TUNED_INFERENCE" in src


def test_cancel_inference_autotune_is_gone():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    assert not hasattr(TrackingEngineCore, "cancel_inference_autotune")
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_worker_inference_lookup_only.py -q
```

Expected: FAIL on all four.

- [ ] **Step 3: Rewire the worker**

Delete `_resolve_inference_autotune_before_load` (`:104-248`) and `cancel_inference_autotune` (`:726-728`) plus its GUI forwarder (`tracking_worker.py:71-73`) and the `_inference_autotune_cancel_requested` flag.

At `:993-995`, re-gate the profiler:

```python
        _profiling_export_enabled = bool(p.get("ENABLE_PROFILING", False))
        # The autotune path needs timing too: observe_production_throughput
        # (end of run) reads profiler.get_summary() to write the measured-key
        # record that the S2 two-record bridge depends on. Gating that on the
        # export flag alone would silently break profile reuse.
        _profiling_enabled = _profiling_export_enabled or bool(
            p.get("APPLY_TUNED_INFERENCE", False)
        )
```

At the call site (`:1376-1415`), replace the block with a lookup guarded by the new flag, keeping the existing preview and backward-pass conditions and the surrounding try/except envelope at `:1416-1441` exactly as they are:

```python
            if p.get("APPLY_TUNED_INFERENCE", False) and not self.preview_mode:
                from hydra_suite.core.inference.autotune import session as _autotune

                _ctx = _autotune.build_autotune_context(
                    _inference_cfg,
                    p,
                    video_path=self.video_path,
                    frame_width=_frame_width,
                    frame_height=_frame_height,
                    start_frame=int(p.get("START_FRAME", 0)),
                    end_frame=_end_frame,
                    realtime=_effective_realtime,
                    cache_dir=self._resolve_cache_dir(),
                    use_cached_detections=self.use_cached_detections,
                    cache_read_only_replay=_cache_read_only_replay,
                )
                _inference_cfg, self.inference_autotune_overlay, _result = (
                    _autotune.lookup(_ctx)
                )
```

Reuse the exact local names the current block already computes for `_frame_width`, `_frame_height`, `_end_frame`, `_effective_realtime`, and `_cache_read_only_replay`; do not re-derive them.

Delete the `_backward_enabled` decline (`:1358-1375`) — Task 8 replaces it with propagation. **Task 7 and Task 8 must land together before any real-video run**; if Task 8 is not yet done, keep the decline in place.

Update the four `_dispatch_params(INFERENCE_AUTOTUNE_MODE="automatic", ...)` calls in `tests/test_worker_real_inference_integration.py` to `APPLY_TUNED_INFERENCE=True`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_worker_inference_lookup_only.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py \
                tests/test_worker_real_inference_integration.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src tests
git commit -m "feat(tracking): a run looks up a profile and never measures one

Deletes the in-run calibration entry point and the cancel plumbing it
needed. Keeps observe_production_throughput alive by gating the profiler
on APPLY_TUNED_INFERENCE -- that write is the measured-key half of the
S2 bridge, and losing it would make every cached run miss forever."
```

---

### Task 8: Propagate the forward vector to the backward pass

**Files:**
- Create: `src/hydra_suite/core/inference/autotune/applied_vector.py`
- Modify: `src/hydra_suite/core/tracking/worker.py` (write after lookup; read when `backward_mode`)
- Test: `tests/test_inference_autotune_backward_propagation.py`

**Interfaces:**
- Produces:
  - `write_applied_vector(cache_dir: Path, settings: InferenceTuningSettings) -> None`
  - `read_applied_vector(cache_dir: Path) -> InferenceTuningSettings | None`
  - File: `<cache_dir>/applied_inference_vector.json`

**Precedent to follow:** `utils/video_artifacts.py:257-265` already defines `build_autotune_state_path`, which parks a `.autotune_state.json` sidecar next to the detection cache for the older tile-batch tuner. Use the same placement convention (a small JSON sidecar in the inference-cache directory) but a distinct filename — the two carry unrelated state and must not collide.

This is the task that makes the feature work on the default configuration. `src/hydra_suite/resources/configs/default.json:60` sets `enable_backward_tracking: true`, and **every** equivalence fixture config enables it, so without this the tuner is inert almost everywhere.

The failure being fixed is documented at `worker.py:1341-1349` and was measured on courtship: forward promotes `det=4` and writes its detection cache under a key containing that batch size; backward re-resolves at the project's untuned size, misses the key, and aborts the whole run.

A file is required rather than an in-memory hand-off because the two paths do not share one: headless drives both passes from one function with one `params` dict (`headless_tracking.py:229-257`), but the GUI issues backward as a **separate** `start_tracking_on_video(video_fp, backward_mode=True)` invocation (`gui/orchestrators/tracking.py:1110`) that rebuilds params from config. `_resolve_cache_dir()` (`worker.py:5090-5096`) returns the same directory for both passes of the same video, so it is the one seam they share.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_autotune_backward_propagation.py`:

```python
"""The backward pass must run at the forward pass's batch size."""

import json
from pathlib import Path

import pytest

from hydra_suite.core.inference.autotune.applied_vector import (
    read_applied_vector,
    write_applied_vector,
)
from hydra_suite.core.inference.autotune.models import InferenceTuningSettings


def test_roundtrip_preserves_every_field(tmp_path):
    settings = InferenceTuningSettings(
        detection_batch_size=4,
        slice_tile_batch_size=8,
        pose_batch_size=16,
        headtail_batch_size=None,
        identity_batch_sizes=(("ant", 32),),
        pipeline_depth=2,
    )
    write_applied_vector(tmp_path, settings)
    assert read_applied_vector(tmp_path) == settings


def test_missing_file_returns_none(tmp_path):
    """A cache written before this change must fall back to configured values,
    which is exactly today's behaviour for an untuned forward pass."""
    assert read_applied_vector(tmp_path) is None


def test_corrupt_file_returns_none_and_does_not_raise(tmp_path):
    (tmp_path / "applied_inference_vector.json").write_text("{ not json", encoding="utf-8")
    assert read_applied_vector(tmp_path) is None


def test_written_file_is_human_readable_json(tmp_path):
    settings = InferenceTuningSettings(
        detection_batch_size=4,
        slice_tile_batch_size=None,
        pose_batch_size=None,
        headtail_batch_size=None,
        identity_batch_sizes=(),
        pipeline_depth=1,
    )
    write_applied_vector(tmp_path, settings)
    data = json.loads((tmp_path / "applied_inference_vector.json").read_text())
    assert data["detection_batch_size"] == 4
```

Read `models.py:49-218` first and construct `InferenceTuningSettings` with its real field names and defaults; adjust the constructor calls above if the real signature differs.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_inference_autotune_backward_propagation.py -q
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the sidecar file and wire both passes**

Create `src/hydra_suite/core/inference/autotune/applied_vector.py`:

```python
"""The inference vector a forward pass actually ran at, persisted for backward.

The backward pass replays the forward pass's detection cache, and the cache key
includes the batch size. If backward re-resolves independently it can pick a
different size, miss the key, and abort the run -- measured on courtship. The
GUI and headless paths issue the two passes separately and share no in-memory
state, so the inference-cache directory is the only seam available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hydra_suite.core.inference.autotune.models import InferenceTuningSettings

logger = logging.getLogger(__name__)

APPLIED_VECTOR_FILENAME = "applied_inference_vector.json"


def write_applied_vector(cache_dir, settings: InferenceTuningSettings) -> None:
    """Record the vector this pass ran at, next to the cache it wrote."""
    path = Path(cache_dir) / APPLIED_VECTOR_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        # Never fail a run over provenance: the backward pass falls back to
        # configured values, which is the pre-existing untuned behaviour.
        logger.warning("Could not persist the applied inference vector", exc_info=True)


def read_applied_vector(cache_dir) -> InferenceTuningSettings | None:
    """Return the forward pass's vector, or None to fall back to config."""
    path = Path(cache_dir) / APPLIED_VECTOR_FILENAME
    if not path.is_file():
        return None
    try:
        return InferenceTuningSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("Ignoring an unreadable applied inference vector", exc_info=True)
        return None
```

In `worker.py`, immediately after a successful forward lookup, persist the effective vector:

```python
                if not self.backward_mode:
                    from hydra_suite.core.inference.autotune.applied_vector import (
                        write_applied_vector,
                    )

                    write_applied_vector(
                        self._resolve_cache_dir(),
                        self.inference_autotune_overlay.effective,
                    )
```

And on the backward pass, apply the recorded vector instead of performing a lookup:

```python
            elif self.backward_mode and p.get("APPLY_TUNED_INFERENCE", False):
                from hydra_suite.core.inference.autotune.applied_vector import (
                    read_applied_vector,
                )

                _forward_vector = read_applied_vector(self._resolve_cache_dir())
                if _forward_vector is not None:
                    _inference_cfg = _forward_vector.apply(_inference_cfg)
                    logger.info(
                        "Backward pass reusing the forward pass's inference vector: %s",
                        _forward_vector.to_dict(),
                    )
```

Remove the `_backward_enabled` decline block deleted in Task 7 if it is still present.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_autotune_backward_propagation.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py \
                tests/test_worker_inference_lookup_only.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src tests
git commit -m "feat(autotune): run the backward pass at the forward pass's vector

Backward replays the forward detection cache, whose key includes the
batch size; re-resolving independently missed the key and aborted the
run (measured on courtship). The GUI and headless paths share no
in-memory state between passes, so the vector rides in the cache dir.
This is what makes tuning work on the default config, which enables
backward tracking."
```

---

### Task 9: Close the two-record bridge during calibration

**Files:**
- Modify: `src/hydra_suite/core/inference/autotune/session.py` (`calibrate`)
- Test: `tests/test_inference_autotune_bridge.py`

**Interfaces:** consumes Task 1's `calibrate`, Task 3's intent.

Calibration runs before any detection cache exists, so its key carries `density_is_estimated=True` and `bucket(MAX_TARGETS)` density. A later run *with* a cache computes measured density and a different key. The store already bridges this over two runs (`store.py:243-300`): the first real sample writes a **second** record under the measured key and leaves the estimated one in place. Calibration can close that gap immediately, since its own trials measured real density.

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_autotune_bridge.py`:

```python
"""One click must serve both the cache-less first run and every cached run."""

import pytest

from hydra_suite.core.inference.autotune import session
from tests.autotune_helpers import (
    fake_store,
    make_calibration_context,
    make_search_result_with_density,
)


def test_calibration_writes_both_an_estimated_and_a_measured_record(monkeypatch):
    store = fake_store()
    ctx = make_calibration_context(cache_dir=None)  # no detection cache yet
    session.calibrate(ctx, budget_seconds=60.0)
    keys = {profile.key.workload.density_is_estimated for profile in store.saved}
    assert keys == {True, False}, "expected an estimated-key AND a measured-key record"


def test_a_cacheless_run_hits_the_estimated_record(monkeypatch):
    store = fake_store()
    ctx = make_calibration_context(cache_dir=None)
    session.calibrate(ctx, budget_seconds=60.0)
    _, overlay, _ = session.lookup(make_calibration_context(cache_dir=None))
    assert overlay.status == "cache_hit"


def test_a_cached_run_hits_the_measured_record(monkeypatch, tmp_path):
    store = fake_store()
    session.calibrate(make_calibration_context(cache_dir=None), budget_seconds=60.0)
    _, overlay, _ = session.lookup(
        make_calibration_context(cache_dir=tmp_path, measured_counts=(8, 9, 8))
    )
    assert overlay.status == "cache_hit"
```

Build `fake_store`, `make_calibration_context`, and `make_search_result_with_density` in `tests/autotune_helpers.py` following the fake-executor pattern already used there, so no real video or sidecar is needed.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_inference_autotune_bridge.py -q
```

Expected: FAIL — only the estimated-key record is written.

- [ ] **Step 3: Have calibrate observe its own measured density**

At the end of `session.calibrate`, after `resolve_tracking_inference_config` returns:

```python
    effective, overlay, result = resolve_tracking_inference_config(...)
    _close_density_bridge(result)
    return effective, overlay, result


def _close_density_bridge(result) -> None:
    """Write the measured-key twin of an estimate-keyed profile immediately.

    The store already re-keys an estimated-density profile on the first real
    production sample (``store.observe_production_throughput``), deliberately
    leaving the estimated record in place so the next brand-new video still
    gets a warm start. Calibration has already measured real density in its own
    trials, so there is no reason to make the user pay a whole extra tracking
    run before a cached run can find anything.
    """

    from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore

    profile = getattr(result, "profile", None)
    if profile is None or not profile.key.workload.density_is_estimated:
        return
    counts = _measured_density_from(profile)
    if not counts:
        return
    try:
        InferenceTuningProfileStore().observe_production_throughput(
            profile.profile_id,
            _median_throughput_of(profile),
            detection_counts=counts,
            crop_counts=counts,
        )
    except Exception:
        logger.warning("Could not close the calibration density bridge", exc_info=True)
```

Implement `_measured_density_from(profile)` by reading the per-candidate evidence the search already records — inspect `CandidateEvidence` in `models.py:264-360` for the field carrying observed detection counts, and if none exists, thread the counts out of `sidecar_child._run_window` into `CandidateEvidence` as a new `detection_counts: tuple[int, ...]` field in this task. Implement `_median_throughput_of(profile)` from the selected candidate's `median_throughput`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference_autotune_bridge.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src tests
git commit -m "feat(autotune): one calibration serves cached and cache-less runs

Calibration measured real density in its trials, so write the
measured-key twin immediately instead of making the user pay an extra
full tracking run before the S2 bridge closes."
```

---

### Task 10: CLI — apply flag and a `trackerkit calibrate` subcommand

**Files:**
- Modify: `src/hydra_suite/trackerkit/app.py:121-145, 300-340`
- Modify: `src/hydra_suite/trackerkit/cli.py:51-72`
- Modify: `src/hydra_suite/trackerkit/cli_config.py:216-247`
- Create: `src/hydra_suite/trackerkit/calibrate_cli.py`
- Test: `tests/test_trackerkit_calibrate_cli.py`

**Interfaces:**
- Produces: `run_calibrate_cli(video_path, config_path, *, budget_seconds) -> int` (0 success, non-zero failure).

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackerkit_calibrate_cli.py`:

```python
"""The headless equivalent of the Calibrate button."""

import pytest

from hydra_suite.trackerkit.app import build_parser


def test_track_exposes_the_apply_flag_and_not_the_old_modes():
    parser = build_parser()
    args = parser.parse_args(["track", "--video", "v.mp4", "--apply-tuned-inference"])
    assert args.apply_tuned_inference is True
    with pytest.raises(SystemExit):
        parser.parse_args(["track", "--video", "v.mp4", "--inference-autotune", "record"])


def test_no_apply_flag_disables():
    parser = build_parser()
    args = parser.parse_args(
        ["track", "--video", "v.mp4", "--no-apply-tuned-inference"]
    )
    assert args.apply_tuned_inference is False


def test_calibrate_subcommand_exists_with_a_budget():
    parser = build_parser()
    args = parser.parse_args(
        ["calibrate", "--video", "v.mp4", "--budget-seconds", "600"]
    )
    assert args.command == "calibrate"
    assert args.budget_seconds == 600.0


def test_calibrate_refuses_fanout_flags():
    """Concurrent calibration on one box measures contention, not throughput."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate", "--video", "v.mp4", "--gpus", "auto"])
```

If `app.py` has no `build_parser` factory, extract one in this task so the parser is testable without invoking `main`.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_trackerkit_calibrate_cli.py -q
```

- [ ] **Step 3: Implement**

Replace the mutually exclusive `--inference-autotune` group at `app.py:121-145` with:

```python
    autotune_group = track_parser.add_mutually_exclusive_group()
    autotune_group.add_argument(
        "--apply-tuned-inference",
        dest="apply_tuned_inference",
        action="store_true",
        default=None,
        help="Apply a validated inference profile if one exists for this "
        "configuration. Produce one with `trackerkit calibrate`.",
    )
    autotune_group.add_argument(
        "--no-apply-tuned-inference",
        dest="apply_tuned_inference",
        action="store_false",
        help="Ignore any stored inference profile and use configured values.",
    )
```

Keep `--inference-autotune-manual` as-is. Add the `calibrate` subparser with the same `--video`/`--config` arguments as `track` plus `--budget-seconds` (default from `core/inference/config.py`'s existing default, bounded by `MINIMUM_`/`MAXIMUM_CALIBRATION_BUDGET_SECONDS`), and deliberately **without** `--gpus`/`--jobs`.

Create `calibrate_cli.py` that builds params via the same shared builder `track` uses, calls `session.build_autotune_context` then `session.calibrate`, logs the resulting status/reason/effective vector, and returns 0 on a `calibrated`/`cache_hit` status and 1 otherwise.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_trackerkit_calibrate_cli.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src tests
git commit -m "feat(cli): trackerkit calibrate, and --apply-tuned-inference

Calibration is now something you ask for, on the box that will run the
work. Fan-out flags are deliberately rejected: concurrent calibration
measures contention rather than throughput."
```

---

### Task 11: GUI — one checkbox and one button

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py:48, 755-801, 807-817, 835-837, 991-1041`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:931-933`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:362-363, 365-373`
- Create: `src/hydra_suite/trackerkit/gui/dialogs/calibration.py`
- Create: `src/hydra_suite/trackerkit/gui/workers/calibration_worker.py`
- Test: `tests/test_trackerkit_calibration_gui.py`

**Interfaces:**
- Consumes: `session.build_autotune_context`, `session.calibrate`.
- Produces: `CalibrationWorker(BaseWorker)` with signals `progress(int)`, `status(str)`, `error(str)` (inherited) and `completed(dict)`; `CalibrationDialog(BaseDialog)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackerkit_calibration_gui.py`:

```python
"""Setup panel: an apply checkbox and a Calibrate button, nothing else."""

import pytest

pytest.importorskip("PySide6")


def test_old_calibration_widgets_are_gone(qtbot, setup_panel):
    assert not hasattr(setup_panel, "combo_inference_autotune")
    assert not hasattr(setup_panel, "spin_inference_autotune_budget")
    assert not hasattr(setup_panel, "btn_continue_inference_settings")


def test_apply_checkbox_and_calibrate_button_exist(qtbot, setup_panel):
    assert setup_panel.chk_apply_tuned_inference is not None
    assert setup_panel.btn_calibrate_inference is not None


def test_checkbox_writes_the_config_field(qtbot, setup_panel):
    setup_panel.chk_apply_tuned_inference.setChecked(True)
    assert setup_panel.config.apply_tuned_inference is True


def test_calibration_worker_subclasses_baseworker():
    from hydra_suite.trackerkit.gui.workers.calibration_worker import CalibrationWorker
    from hydra_suite.widgets.workers import BaseWorker

    assert issubclass(CalibrationWorker, BaseWorker)


def test_calibration_dialog_subclasses_basedialog():
    from hydra_suite.trackerkit.gui.dialogs.calibration import CalibrationDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    assert issubclass(CalibrationDialog, BaseDialog)
```

Reuse the `setup_panel` fixture pattern from `tests/test_trackerkit_panels_smoke.py`; if none exists there, add one that constructs the panel under `qtbot`.

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_trackerkit_calibration_gui.py -q
```

- [ ] **Step 3: Implement the widgets, worker, and dialog**

In `setup_panel.py`, delete the signal at `:48`, the combo (`:755-774`), the budget spinbox and its row (`:776-801`), the Continue button (`:807-817`), their grid placements (`:835-837`), and the handlers `_on_inference_autotune_mode_changed` / `_on_inference_autotune_budget_changed` / `_set_inference_autotune_combo_mode` / `set_inference_autotune_calibration_active` (`:991-1041`). Keep `lbl_inference_autotune_status` and its setters.

Add, in the same toggle grid as "Reuse cache":

```python
        self.chk_apply_tuned_inference = QCheckBox("Apply tuned inference profile if available")
        self.chk_apply_tuned_inference.setToolTip(
            "Use a validated performance profile measured for this machine, "
            "model, and workload. Produce one with Calibrate… . When no "
            "profile matches, the run uses your configured values unchanged."
        )
        self.chk_apply_tuned_inference.setChecked(bool(self.config.apply_tuned_inference))
        self.chk_apply_tuned_inference.toggled.connect(self._on_apply_tuned_inference_toggled)

        self.btn_calibrate_inference = QPushButton("Calibrate…")
        self.btn_calibrate_inference.setToolTip(
            "Measure the fastest inference settings for the current video and "
            "save them for reuse. Runs once; every later run just applies the "
            "result."
        )
        self.btn_calibrate_inference.clicked.connect(self.calibrate_inference_requested.emit)
```

with `calibrate_inference_requested = Signal()` replacing the removed signal, and:

```python
    def _on_apply_tuned_inference_toggled(self, checked: bool) -> None:
        self.config.apply_tuned_inference = bool(checked)
```

Create `gui/workers/calibration_worker.py`:

```python
"""Background worker for user-triggered inference calibration."""

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker


class CalibrationWorker(BaseWorker):
    """Run one calibration search off the GUI thread."""

    completed = Signal(dict)

    def __init__(self, params: dict, config, *, video_path: str, budget_seconds: float,
                 frame_width: int, frame_height: int, start_frame: int, end_frame: int,
                 cache_dir=None, use_cached_detections: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._params = dict(params)
        self._config = config
        self._video_path = video_path
        self._budget_seconds = float(budget_seconds)
        self._frame_width = int(frame_width)
        self._frame_height = int(frame_height)
        self._start_frame = int(start_frame)
        self._end_frame = int(end_frame)
        self._cache_dir = cache_dir
        self._use_cached_detections = bool(use_cached_detections)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def execute(self) -> None:
        from hydra_suite.core.inference.autotune import session

        ctx = session.build_autotune_context(
            self._config,
            self._params,
            video_path=self._video_path,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
            start_frame=self._start_frame,
            end_frame=self._end_frame,
            realtime=False,
            cache_dir=self._cache_dir,
            use_cached_detections=self._use_cached_detections,
            should_cancel=lambda: self._cancelled,
            status_callback=self.status.emit,
        )
        _effective, overlay, _result = session.calibrate(
            ctx, budget_seconds=self._budget_seconds
        )
        self.completed.emit(
            {
                "status": overlay.status,
                "reason": overlay.reason,
                "profile_id": overlay.profile_id,
                "effective": overlay.effective.to_dict(),
            }
        )
```

Create `gui/dialogs/calibration.py` with `CalibrationDialog(BaseDialog)` holding the budget `QDoubleSpinBox` (range `MINIMUM_`/`MAXIMUM_CALIBRATION_BUDGET_SECONDS`, suffix `" s"`), a `QLabel` status area fed by `CalibrationWorker.status`, and a Cancel button wired to `worker.cancel()`. Use `BaseDialog(title="Calibrate inference performance", buttons=QDialogButtonBox.Close)` and `add_content(...)`.

In `main_window.py:931-933`, replace the Continue connection with one from `setup.calibrate_inference_requested` to a `TrackingOrchestrator.open_calibration_dialog` slot. That slot must **refuse to open while a track is running** and, symmetrically, tracking must refuse to start while the dialog is open — on MPS/CPU the contention probe hardcodes `contention_detected=False` (`device.py:172-193`), so nothing below the GUI can detect the overlap and the measurement would be garbage.

In `orchestrators/tracking.py`, delete only lines `:362-363` and `:365-373`. **Lines `:356-361` drive the progress bar and must stay.**

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_trackerkit_calibration_gui.py tests/test_trackerkit_panels_smoke.py -q
python -m pytest tests/test_inference_autotune_*.py tests/test_trackerkit_inference_autotune_surfaces.py -q
```

- [ ] **Step 5: Format and commit**

```bash
make format
git add -A src tests
git commit -m "feat(trackerkit): one Calibrate button and one apply checkbox

Replaces the three-state dropdown, in-run budget spinbox, and the
Continue escape hatch that only existed to interrupt calibration the run
should never have been doing. Calibration and tracking are mutually
exclusive: the MPS contention probe cannot detect an overlap."
```

---

### Task 12: Documentation and the equivalence gate

**Files:**
- Modify: `docs/developer-guide/inference-autotuner.md`
- Modify: `docs/user-guide/trackerkit-cli.md`
- Verify: `tools/equivalence/run_matrix.sh`

- [ ] **Step 1: Rewrite the docs**

In `docs/developer-guide/inference-autotuner.md`, replace "The three modes" and "CUDA-only policy for `automatic`" with a "Calibrate once, apply everywhere" section covering: the Calibrate button and `trackerkit calibrate`; the apply checkbox; the `eligible` vs `allow_cached_reuse` split with the table from the spec; backward-pass propagation via `applied_inference_vector.json`; and the two-record density bridge. Retain the budget, correctness-gate, determinism-floor, and profile sections. Document that on MPS a no-op profile is a correct outcome, citing the 1.58× figure in `docs/developer-guide/performance-tuning.md:49`.

Add the `calibrate` subcommand to `docs/user-guide/trackerkit-cli.md`.

- [ ] **Step 2: Build the docs**

```bash
make docs-check
```

Expected: strict build passes, terminology check passes.

- [ ] **Step 3: Full test suite**

```bash
conda activate hydra-mps
python -m pytest tests/ -q -p no:randomly 2>&1 | tail -20
```

Compare the failure **set** (not count) against `main`: `git stash` is forbidden here — instead run the same command on a clean checkout of `main` in a separate worktree and diff the failing test IDs. Only newly-failing tests are regressions.

- [ ] **Step 4: Equivalence gate, both platforms**

Kill stale `sleap`/`hydra` processes first. With `apply_tuned_inference` **off**, output must be byte-identical to `main` — the run path changed, so the untuned path must be proven unaffected.

```bash
# MPS (this box)
conda activate hydra-mps
git worktree add --detach .worktrees/equiv-base main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_calib RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

Verify row counts > 1 in every CSV before trusting an `EQUIVALENT` — a bare shell yields empty CSVs that falsely compare equal. Then repeat on mehek with `RUNTIME=cuda` per `CLAUDE.md`.

Additionally run **one backward-enabled clip with `apply_tuned_inference` on** and confirm the run completes — that is the direct regression test for the courtship abort.

- [ ] **Step 5: Commit**

```bash
git add -A docs
git commit -m "docs(autotune): calibrate once, apply everywhere"
```

---

## Self-Review

**Spec coverage:** §1 eligibility split → Task 2. §2a density bridge → Tasks 7, 9. §2b cache mask → Task 1 (context builder). §2c TensorRT → Task 4. §3 session module → Tasks 1, 3. §4 backward propagation → Task 8. §5 worker + throughput observation → Task 7. §6 sidecar guard → Task 6. §7 GUI → Task 11. §8 config/params/CLI → Tasks 5, 10. §9 docs → Task 12. Testing section → distributed across every task, with the equivalence gate in Task 12.

**Known soft spots the executor must resolve rather than guess:**
- Task 2, 3, 4, 9 depend on helper factories in `tests/autotune_helpers.py` that do not exist yet. Each task says to follow the file's existing construction pattern; read it before writing.
- Task 9's `_measured_density_from` may require adding a `detection_counts` field to `CandidateEvidence` and threading it out of `sidecar_child._run_window`. Confirm against `models.py:264-360` first.
- Task 10 assumes a `build_parser` factory in `app.py`; extract one if absent.
- Task 4's `candidate_space` may need extracting from `values_for`.

**Ordering constraint:** Tasks 7 and 8 must land together before any real-video run. Task 7 deletes the backward decline that Task 8 replaces with propagation; shipping 7 alone re-exposes the courtship abort.
