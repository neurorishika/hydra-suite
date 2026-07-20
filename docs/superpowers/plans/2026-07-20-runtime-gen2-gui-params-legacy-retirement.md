# Runtime Gen-2: GUI Params + Legacy Vocabulary Retirement — Follow-up Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Status:** Follow-up to the landed core consolidation (branch `feature/runtime-gen2-consolidation`, 8 commits, off `main`). This plan supersedes Tasks 6–13 of `2026-07-19-runtime-resolver-gen2-consolidation.md`, which under-scoped the GUI params channel.

**Goal:** Remove the remaining `compute_runtime` STRING vocabulary — which now lives ENTIRELY outside core inference — so `runtime_tier`/`ResolvedBackend` is the only runtime representation anywhere.

**What the core slice already achieved (do not redo):** Core inference (`stages/*`, identity `classification/*` + `pose/backends/sleap.py`) consumes `ResolvedBackend` directly. `runtime_to_compute_runtime` is deleted. No `compute_runtime` string flows through core inference. Golden table (`tests/runtime/test_resolver_golden.py`) is the behavior gate.

## Sequencing

Do this plan **before** the deferred TrackingWorker Qt-in-core migration. Rationale: this plan makes only ~4 small surgical edits to `core/tracking/worker.py` (cache-key + telemetry string sources); doing it first leaves `worker.py`'s runtime handling string-free, so the later large Qt-in-core split carries less legacy. Front-loading the 4372-line Qt split to enable 4 edits is not worth the risk (and it is verification-gated).

## Global Constraints

- Runtime tiers exactly `{cpu, gpu, gpu_fast}`; resolver backends exactly `{torch, tensorrt, coreml}`; devices exactly `{cpu, cuda, mps}`.
- Core/Runtime/Data/Training/Utils never import from app layers or Integrations.
- The golden table `(tier × platform × stage × artifact) → ResolvedBackend` stays byte-identical.
- Base is NOT green: ~24 pre-existing classifier/GUI failures on main (`.superpowers/sdd/baseline-failures-t4-scope.txt`). Bar = zero NEW failures, not all-green.
- Test contract: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest … -p no:cacheprovider`; add `--ignore=tests/test_identity_postprocess.py` for broad runs. Never run SLEAP-subprocess integration tests in the iterating gate (they hang). Commit as the configured git user; NO `Co-Authored-By: Claude` trailer. `make format` is broken — run `black`/`isort` directly.

## Key discovery: the params channel is a per-stage FAMILY, not one key

The GUI→worker params dict carries a whole family of runtime strings, all now downstream of `RUNTIME_TIER`:
`COMPUTE_RUNTIME`, `HEADTAIL_COMPUTE_RUNTIME`, `CNN_COMPUTE_RUNTIME`, `OBB_COMPUTE_RUNTIME`, `POSE_COMPUTE_RUNTIME`.
Producers: `trackerkit/gui/orchestrators/config.py:2143-2153`, `orchestrators/tracking.py:3842-3970` (`_preview_safe_runtime`), `gui/workers/preview_worker.py:489-592`, `crops_worker.py`, `cli_config.py`, `tracking_cache.py`, `data/dataset_generation.py`.
Consumers, all now SOFT (none drive backend selection — that is `runtime_tier`→`resolved`):
- `core/inference/config.py:build_inference_config_from_params` — feeds only the **inert** per-stage config fields + the tier fallback (`migrate_runtime_to_tier({compute_runtime})`).
- `core/tracking/worker.py` — classify **cache-key** (`compute_classify_cache_id`, lines ~1546, ~4320) and **telemetry** (`profiler.set_config(compute_runtime=…)` ~949, `runtime_family=…` ~2471).
- `core/identity/properties/cache.py:174` — param-bag `COMPUTE_RUNTIME`/`POSE_SLEAP_DEVICE`.

## Open decision (resolve at FT1 execution): classify cache-key

`compute_classify_cache_id(compute_runtime=…)` currently keys on the runtime string. Removing the string changes what feeds the key. Two options:
- **Stable (recommended):** derive the SAME string the old path produced from `runtime_tier` (via a local tier→string map or the resolved device/backend), so existing identity caches stay valid. Slightly more code, no invalidation.
- **Change:** key on `runtime_tier` directly; simpler, but invalidates existing classify caches on next run.
Pick one before implementing FT1 and state it in the commit.

---

## FT1 — Retire the COMPUTE_RUNTIME params family; migrate worker soft-consumers

**Files:** `trackerkit/gui/orchestrators/config.py` (2143-2153), `orchestrators/tracking.py` (3842-3970), `gui/workers/preview_worker.py` (489-592), `crops_worker.py`, `tracking_cache.py`, `data/dataset_generation.py`, `core/tracking/worker.py` (~949, ~1546, ~2471, ~4320). Tests: `test_inference_config_from_params.py`, `test_worker_runtime_tier.py`, `test_interpolated_crops_worker.py`.

- [ ] Ensure every params producer writes `RUNTIME_TIER` (from the setup panel's `_selected_runtime_tier()`/`_current_runtime_tier()`); remove all `*COMPUTE_RUNTIME*` family writes.
- [ ] Migrate `worker.py` cache-key + telemetry off the string per the chosen cache-key decision above. `profiler.set_config`/`runtime_family` telemetry → the tier string (or resolved device).
- [ ] Retire `_preview_safe_runtime`'s per-stage string juggling (it exists only to sanitize the family).
- [ ] Leave `build_inference_config_from_params`'s READ of `COMPUTE_RUNTIME` for now (FT5 removes it with the per-stage fields). Keep `RUNTIME_TIER` as the sole live tier source.
- [ ] Gate: `test_inference_config_from_params.py`, `test_worker_runtime_tier.py`, delta vs baseline. Commit.

## FT2 — Resolve tier at `properties/cache.py`; drop pose-flavor fields

**Files:** `core/identity/properties/cache.py:173-196,256-271`; pose-flavor threading in `orchestrators/{session,tracking,config}.py`, `panels/{detection_panel,identity_panel}.py`, `trackerkit/gui/main_window.py`, `posekit/gui/main_window.py`, `cli_config.py`. Test: new `tests/test_cache_runtime_payload.py`.

- [ ] Add `_runtime_payload_for_params(params)`: read `RUNTIME_TIER`, resolve via `RuntimeResolver(tier, detect_platform()).resolve("sleap_pose")`, return `{"device": resolved.device, "backend": resolved.backend}`. Replace the `COMPUTE_RUNTIME`/`POSE_SLEAP_DEVICE` reads.
- [ ] Remove `pose_runtime_flavor`/`pose_sleap_device` from stored config; update the GUI threading sites.
- [ ] Gate + commit.

## FT3 — Delete legacy pose-settings + legacy-config inference helpers

**Files:** `runtime/compute_runtime.py` — delete `derive_pose_runtime_settings`, `infer_compute_runtime_from_legacy`, `_runtime_from_pose_flavor`, `_best_auto_runtime`, `_best_explicit_onnx_runtime`. Their callers were removed in FT1/FT2 + the landed core slice (`detection_panel.py` still imports `derive_pose_runtime_settings` — migrate it in FT2 first).

- [ ] `grep -rn "derive_pose_runtime_settings\|infer_compute_runtime_from_legacy\|_runtime_from_pose_flavor" src/` → zero non-def hits before deleting. Commit.

## FT4 — Rewrite `CUDA_RUNTIMES`/`CPU_RUNTIMES`/`default_runtime` onto ResolvedBackend

**Files:** `core/inference/config.py:15-16,33`; `runtime.py:30,153` (`default_runtime: ComputeRuntime` field); `runner.py:691` (`runtime_family=str(self.runtime.default_runtime)`); `stages/obb.py:12`; `api.py:185`.

- [ ] Replace `rt in CUDA_RUNTIMES` membership with `resolved.device == "cuda"` checks. Replace `default_runtime` usage (telemetry `runtime_family` in runner.py) with the resolved device/tier. Delete the two frozensets + the `fast` set.
- [ ] Gate: `tests/core/inference/`, golden. Commit.

## FT5 — Loud missing-tier error; drop per-stage `compute_runtime` fields + `ComputeRuntime` type

**Files:** `core/inference/config.py` — `_dict_to_config:437-447` (raise on missing `runtime_tier`), `build_inference_config_from_params:491-500` (default absent tier to `"cpu"`, drop the `COMPUTE_RUNTIME` read + `migrate_runtime_to_tier`), per-stage fields (89,105-106,217,231,245,257), the `ComputeRuntime` Literal (11), `_collect_legacy_runtime_strings`, `migrate_runtime_to_tier`. Tests: `test_inference_config_tier_migration.py` (rewrite/retire), `test_inference_config_from_params.py`.

- [ ] `_dict_to_config`: missing `runtime_tier` → `ValueError("… run scripts/migrate_runtime_config.py …")`.
- [ ] Remove per-stage `compute_runtime` fields + all their construction kwargs; delete the `ComputeRuntime` Literal; delete the migration helpers.
- [ ] Gate + commit.

## FT6 — One-shot migration script

**Files:** `scripts/migrate_runtime_config.py`, `tests/scripts/test_migrate_runtime_config.py`, migrate bundled `src/hydra_suite/resources/configs/ooceraea_biroi.json`.

- [ ] Self-contained `migrate_config_dict(d)`: fast set `{onnx_cpu,onnx_cuda,onnx_coreml,tensorrt}`→`gpu_fast`; cuda-like/mps→`gpu`; cpu→`cpu`; collect every per-stage `compute_runtime`, set `runtime_tier`, strip deprecated fields + `pose_runtime_flavor`/`pose_sleap_device`; preserve an existing `runtime_tier`. `main()` rewrites JSON files from argv with a `.bak`.
- [ ] Test fixtures + migrate the bundled resource. Commit.

## FT7 — Delete legacy surface; rename `compute_runtime.py` → `onnx_providers.py`

**Files:** `git mv src/hydra_suite/runtime/compute_runtime.py src/hydra_suite/runtime/onnx_providers.py`; update importers (`classification/backend.py`, `pose/backends/sleap.py`, `training/tiny_model.py`, `posekit/gui/runtimes.py`). Delete `_normalize_runtime`, `runtime_label`, `CANONICAL_RUNTIMES`, `_pipeline_supports_runtime`, `supported_runtimes_for_pipeline`, `allowed_runtimes_for_pipelines`, `_onnx_available`, `_sleap_onnx_available`, `derive_onnx_execution_providers` (string API — now unused), `_best_explicit_onnx_runtime`.

- [ ] Confirm the string API + capability tables have zero non-def callers first. Surviving module = `execution_providers_for`, `COREML_PROVIDER_OPTIONS`, `_tensorrt_ep_cache_options`, provider dedup helpers.
- [ ] `make dead-code` clean. Commit.

## FT8 — Delete `resolve_compute_runtime`; GUI display cleanup; stale-name fixes; final verification

**Files:** `runtime/resolver.py` (delete `resolve_compute_runtime` — its GUI callers were migrated in FT1/FT2); `posekit/gui/runtimes.py` (collapse to resolver-only re-exports); `detection_panel.py:1550`, `session.py:998,1140` (now off `resolve_compute_runtime`); rename the two stale tests (`test_backend_generic_multihead_bundle_preserves_onnx_runtime`, `test_interpolated_worker_uses_split_..._runtimes`).

- [ ] `grep -rn "resolve_compute_runtime\|runtime_label\|COMPUTE_RUNTIME\|pose_runtime_flavor" src/` → only genuinely-required survivors (e.g. `worker.py` reading `RUNTIME_TIER`).
- [ ] `make pytest && make lint-moderate && make dead-code` clean. Launch-smoke `trackerkit`/`posekit`: runtime dropdowns show tier labels; a short CPU-tier track runs.
- [ ] Commit.

## Residual findings carried from the core whole-branch review (address opportunistically)

- **(b)** OBB executor still string-based: `stages/obb.py` keeps a local resolved→string "TEMPORARY BOUNDARY" map for the out-of-scope `load_obb_executor`. Cutting `load_obb_executor` over to `ResolvedBackend` removes it — fold into FT4/FT7 if convenient, else a small dedicated task.
- **(c)** `HYDRA_SLEAP_FLAVOR=onnx_cuda` debug override now yields CPU providers (non-resolver-producible debug path). Documented; no action unless the debug flag is revived.
- `sleap.py:_resolved_from_canonical_export` is a bounded canonical-export-flavor→ResolvedBackend helper (SLEAP twin of the OBB boundary), not the forbidden general string bridge — leave unless the SLEAP export string API itself is cut over.

## Non-Goals
- No change to tier semantics or public CLI/inter-kit APIs. No TrackingWorker Qt-in-core structural refactor (separate, later, verification-gated effort).
