# Runtime Resolver Gen-2 Consolidation

**Date:** 2026-07-19
**Status:** Design — approved, pending implementation plan
**Scope:** `hydra_suite.runtime`, `hydra_suite.core.inference`, GUI runtime plumbing

## Problem

Runtime selection currently spans two generations of code that never finished
merging. There are **five runtime representations** and **six bridges** in
flight simultaneously:

**Representations**

1. `runtime_tier` ∈ `{cpu, gpu, gpu_fast}` — Gen-2 intent (the desired end state), stored on `InferenceConfig.runtime_tier`.
2. `ComputeRuntime` string ∈ `{cpu, mps, cuda, onnx_cpu, onnx_cuda, onnx_coreml, tensorrt}` — Gen-1 canonical runtime, still present as per-stage config fields (marked deprecated).
3. `ResolvedBackend(backend, device, used_fallback)` — resolver output.
4. `RuntimeContext` — live per-run object (owns CUDA-event handoff).
5. Legacy pose settings `pose_runtime_flavor` / `pose_sleap_device`.

**Bridges**

- `migrate_runtime_to_tier` (string set → tier) — `core/inference/config.py`
- `runtime_to_compute_runtime` (RuntimeContext → string) — `core/inference/runtime.py`
- `resolve_compute_runtime` (tier → string, display/benchmark only) — `runtime/resolver.py`
- `infer_compute_runtime_from_legacy` (old fields → string) — `runtime/compute_runtime.py`
- `derive_pose_runtime_settings` (string → pose flavor dict) — `runtime/compute_runtime.py`
- `derive_onnx_execution_providers` (string → ONNX EP list) — `runtime/compute_runtime.py`
- plus `_normalize_runtime` + the per-pipeline capability tables

The live inference path already flows through the Gen-2 resolver
(`RuntimeContext.from_config` calls `RuntimeResolver.resolve`), and the GUI
runtime selectors already use `tier_label` / `available_tiers`. The string
vocabulary and the capability/legacy layers are now **internal shim and
legacy-migration weight** that the finished system does not need.

## Goal

Keep **only** the Gen-2 resolver path. One stored knob (`runtime_tier`), one
value that flows (`ResolvedBackend`), zero string shims.

### Key observation

The 5 values `runtime_to_compute_runtime` actually emits —
`{cpu, mps, cuda, tensorrt, coreml}` — are exactly `ResolvedBackend(backend, device)`
flattened:

| string   | backend  | device |
|----------|----------|--------|
| cpu      | torch    | cpu    |
| mps      | torch    | mps    |
| cuda     | torch    | cuda   |
| tensorrt | tensorrt | cuda   |
| coreml   | coreml   | mps    |

So the string encoding carries no information `ResolvedBackend` doesn't already
carry. It can be deleted, not just narrowed.

## Decisions

Two forks were resolved during design:

1. **Legacy configs — clean break + one-shot migration.** No read-time legacy
   migration survives in `src/`. A single offline script rewrites old
   config/preset files to `runtime_tier` once. Old files fail to load until
   migrated (acceptable; internal tool, migration script provided).
2. **Backend vocabulary — `ResolvedBackend` end-to-end.** Delete the
   `ComputeRuntime` string type and `runtime_to_compute_runtime`. Backend
   factories consume `ResolvedBackend` (backend + device) directly.

Two follow-up decisions locked for this spec:

3. **Rename** the shrunken `runtime/compute_runtime.py` → `runtime/onnx_providers.py`
   in the final phase (its surviving contents are purely ONNX-provider plumbing;
   the old name would mislead).
4. **Pose fields fully deleted.** `pose_runtime_flavor` / `pose_sleap_device`
   are removed as stored config; pose device is derived from `ResolvedBackend`
   at the pose stage. The `core/identity/properties/cache.py` param-bag boundary
   (which reads `COMPUTE_RUNTIME` / `POSE_SLEAP_DEVICE` string params, not typed
   config) is resolved from `runtime_tier` at that boundary — audited in Phase 3.

## Target End State

```
config.runtime_tier  (cpu | gpu | gpu_fast)          ← the ONLY stored runtime knob
   → RuntimeResolver(tier, platform).resolve(stage, artifact_available)
   → ResolvedBackend(backend, device, used_fallback)  ← the ONLY runtime value that flows
   → backends consume it directly:
        torch device  = resolved.device
        ONNX EP list   = execution_providers_for(resolved)   ← only surviving derivation
        pose device    = resolved.device / resolved.backend
```

- `RuntimeContext` retains ownership of the live CUDA-event handoff but holds a
  `ResolvedBackend` and derives `cuda_mode` / `coreml_mode` / `tensor_on_cuda`
  from it, rather than a string plus parallel bools.
- `runtime/onnx_providers.py` (renamed) contains only `execution_providers_for(resolved)`,
  `COREML_PROVIDER_OPTIONS`, `_tensorrt_ep_cache_options`, and the provider
  dedup helpers.
- `runtime/resolver.py` is the single authority for tier → backend and for UI
  gating (`available_tiers`, `tier_label`).

## What Gets Deleted

- `ComputeRuntime` Literal type and **every** per-stage `compute_runtime` config
  field (obb direct/sequential, headtail, phase, pose.yolo, pose.sleap).
- Bridges: `runtime_to_compute_runtime`, `resolve_compute_runtime`,
  `infer_compute_runtime_from_legacy`, `derive_pose_runtime_settings`.
- Normalization/auto: `_normalize_runtime` + all aliases, `_best_auto_runtime`,
  `_best_explicit_onnx_runtime`, `_runtime_from_pose_flavor`, all `onnx_*`
  canonical values.
- Capability tables: `_pipeline_supports_runtime`, `supported_runtimes_for_pipeline`,
  `allowed_runtimes_for_pipelines`, `CANONICAL_RUNTIMES`, `runtime_label`.
- `migrate_runtime_to_tier` — **relocated** into the one-shot migration script,
  deleted from `src/`.
- Legacy pose fields `pose_runtime_flavor` / `pose_sleap_device`.

## What Survives (re-keyed off `ResolvedBackend`)

- `execution_providers_for(resolved: ResolvedBackend)` (renamed from
  `derive_onnx_execution_providers`): TensorRT → TRT-EP + CUDA-EP + engine cache;
  CoreML → CoreML-EP; otherwise CPU-EP. Torch backends never call it.
- `COREML_PROVIDER_OPTIONS`, `_tensorrt_ep_cache_options`, provider dedup helpers.
- `RuntimeResolver`, `PlatformInfo`, `ResolvedBackend`, `available_tiers`,
  `tier_label`, `detect_platform` (all already Gen-2).

## Phase Sequence

Each phase keeps `make pytest` green and is independently shippable.

### Phase 0 — Golden characterization test
Snapshot `(tier × platform × stage × artifact_available) → ResolvedBackend` and
the current `runtime_to_compute_runtime` output as a golden table. This is the
behavior-preservation net for every later phase. Platform is injected
(`PlatformInfo`), so all four host shapes (no-accel, cuda, mps, cuda+mps) are
testable on any machine.

### Phase 1 — Backends accept `ResolvedBackend`
- Add `RuntimeContext.resolved: ResolvedBackend`.
- Change `ClassifierBackend`, the SLEAP pose backends, and the OBB backend
  factory to accept `ResolvedBackend` (backend + device) instead of a
  `compute_runtime` string.
- Update the 4 stage files (`stages/obb.py`, `pose.py`, `cnn.py`, `headtail.py`)
  to pass `runtime.resolved` instead of `runtime_to_compute_runtime(runtime)`.
- Add `execution_providers_for(resolved)` alongside the old
  `derive_onnx_execution_providers` (old one not yet removed).
- No behavior change; golden test unchanged.

### Phase 2 — Slim `RuntimeContext`
- Derive `cuda_mode` / `coreml_mode` / `tensor_on_cuda` / `device` from
  `.resolved`.
- Delete `runtime_to_compute_runtime` and its callers (now all migrated).

### Phase 3 — Pose settings onto `ResolvedBackend`
- Replace `derive_pose_runtime_settings`; derive pose device from
  `ResolvedBackend` at the pose stage.
- Remove `pose_runtime_flavor` / `pose_sleap_device` from stored config; update
  the ~8 GUI files that thread them (`posekit/gui/main_window.py`,
  `trackerkit/gui/main_window.py`, `panels/detection_panel.py`,
  `panels/identity_panel.py`, `orchestrators/config.py`, `orchestrators/tracking.py`,
  `orchestrators/session.py`, `trackerkit/cli_config.py`).
- **Audit `core/identity/properties/cache.py`:** it reads `COMPUTE_RUNTIME` /
  `POSE_SLEAP_DEVICE` from a string param-bag. Resolve `runtime_tier` →
  `ResolvedBackend` at that boundary and write the resolved device/backend into
  the cache payload, so cached property metadata stays meaningful without the
  legacy strings.

### Phase 4 — Drop per-stage config fields + `ComputeRuntime` type
- Remove per-stage `compute_runtime` fields from `InferenceConfig` and
  sub-configs; `from_dict` / `from_params` read only `runtime_tier`.
- Delete the `ComputeRuntime` Literal.

### Phase 5 — One-shot migration script
- `scripts/migrate_runtime_config.py`: absorbs the deleted legacy logic
  (`migrate_runtime_to_tier`, `infer_compute_runtime_from_legacy`,
  `_normalize_runtime`, `_runtime_from_pose_flavor`) as **self-contained** code
  inside the script — not imported from `src/`.
- Reads an old config/preset JSON, computes `runtime_tier`, strips deprecated
  per-stage `compute_runtime` and pose-flavor fields, writes back.
- Migrate the bundled `resources/configs/*.json` (e.g. `ooceraea_biroi.json`) in
  the same pass and commit the migrated files.

### Phase 6 — Delete legacy surface + rename module
- Delete capability tables, `_normalize_runtime`, `resolve_compute_runtime`,
  `runtime_label`, `CANONICAL_RUNTIMES`, and the remaining legacy helpers.
- Rename `runtime/compute_runtime.py` → `runtime/onnx_providers.py`; update
  imports (`core/identity/classification/backend.py`,
  `core/identity/pose/backends/sleap.py`, `training/tiny_model.py`).
- Collapse `posekit/gui/runtimes.py` re-export facade to resolver-only symbols.

### Phase 7 — GUI display cleanup
- Confirm every runtime selector uses `tier_label` / `available_tiers`
  (`trackerkit/gui/panels/setup_panel.py`, `posekit/gui/main_window.py` already do).
- Remove stray `runtime_label` references.

## Ordering Rationale

Backends move first (Phase 1) so the string producer
(`runtime_to_compute_runtime`) has no live consumers when it is deleted
(Phase 2). Config-field removal (Phase 4) waits until nothing reads the fields.
The legacy bulk-delete and rename (Phase 6) come last, after the one-shot script
(Phase 5) has rehomed the only legacy logic worth keeping. The golden test
(Phase 0) guards the whole sequence.

## Testing Strategy

- **Phase 0 golden table** is the primary regression guard: identical
  `(tier, platform, stage, artifact) → ResolvedBackend` mapping across all
  phases 1–7.
- Per phase, run `make pytest`; add focused unit tests where a boundary changes
  shape (backend factory signatures in Phase 1, pose-device derivation in
  Phase 3, `from_dict`/`from_params` in Phase 4).
- Migration script (Phase 5) gets its own test: a fixture of representative
  legacy JSONs → expected migrated JSONs, including the bundled resource config.
- Pre-PR: `make commit-prep`, `make lint-moderate`, and `make audit`
  (dead-code sweep should confirm the deleted legacy helpers are truly unreferenced).

## Non-Goals

- No change to public CLI entry points or inter-kit APIs.
- No change to the tier semantics themselves (`cpu`/`gpu`/`gpu_fast` behavior is
  preserved exactly, per the Phase 0 golden table).
- No new runtime tiers or backends.

## Risks

- **Pose path breadth (Phase 3).** Pose settings reach the most files and cross
  the `properties/cache.py` param-bag boundary. Mitigation: audit that boundary
  explicitly; keep Phase 3 isolated and independently reviewable.
- **Hand-built `RuntimeContext` sites** (tests, GUI workers, `api.py`) that set
  fields manually must switch to setting `.resolved`. The existing
  `__post_init__` guard catches some inconsistencies; extend it to validate
  `.resolved` presence.
- **Clean break blocks old configs** until the migration script is run.
  Mitigation: ship the script in Phase 5 (before the legacy delete in Phase 6)
  and migrate bundled resources in the same commit.
