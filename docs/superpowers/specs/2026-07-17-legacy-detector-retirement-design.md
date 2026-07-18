# Design spec: retire `core/detectors`, converge on InferenceRunner

**Date:** 2026-07-17
**Status:** design — not yet a task-by-task plan.
**Runs after:** `plans/2026-07-16-bgsub-inference-stage.md` (with correction note
`plans/notes/2026-07-17-bgsub-task12-yolo-scope-correction.md`) **and**
`plans/2026-07-16-legacy-batching-vestige-removal.md` have both landed on `main`.

## Goal

Delete `src/hydra_suite/core/detectors/` in its entirety and make
`hydra_suite.core.inference` the single detection/inference path. No parallel
pipeline, no duplicate detect API. The one architectural rule this spec enforces:
**helpers may build configs and the runner may grow entrypoints; nothing
re-orchestrates the stages.** Every place that currently hand-rolls
`load_obb_executor` + private `stages.obb` helpers (preview, model-test) or
constructs `YOLOOBBDetector` is a violation to be removed, not a pattern to copy.

## Starting state (after the two prerequisite plans)

`core/detectors/` still contains, all YOLO/OBB:
`__init__.py`, `_direct_obb_runtime.py`, `_obb_geometry.py`, `_runtime_artifacts.py`,
`_utils.py`, `detection_filter.py`, `yolo_detector.py`.

Live consumers that still reach into `core/detectors` at that point:

| Consumer | What it uses | Notes |
|---|---|---|
| `core/inference/runtime_artifacts.py:241` | `create_direct_*_executor` from `_direct_obb_runtime` | **the hard blocker** — the new pipeline imports the legacy module on every TensorRT path |
| `data/dataset_generation.py:421` | `YOLOOBBDetector` (per correction note) | dimension extraction; batched detect |
| `optimization/optimizer_workers.py:372` | `YOLOOBBDetector` (per correction note) | `DetectionCacheBuilderWorker` |
| `optimization/optimizer_workers.py:94,118` + `optimizer.py:165,189` | `DetectionFilter.filter_raw_detections` | legacy-cache scoring branch |
| `core/tracking/pose/pose_pipeline.py:408` | `detector.filter_raw_detections` | |
| `trackerkit/.../preview_worker.py` | ~1000-line clone of `run_realtime`; `DetectionFilter` (:1161,:1774); `_utils._advanced_config_value` (:691,:762,:832) | fake `RuntimeContext` (:577); documented behavioral divergence (:1060) |
| `trackerkit/.../model_test_dialog.py` | private `stages.obb` helpers (`_extract_obb_result`, `_build_crops`, …) | already off `YOLOOBBDetector`, but bypasses the runner |
| `detectkit/gui/prediction_preview.py:40` | raw `ultralytics.YOLO` | torch-only; not even via `core/detectors` |
| `tools/benchmark_models.py` | `YOLOOBBDetector` ×10 incl. `__new__` tricks | to be deprecated (below) |

`_obb_geometry.OBBGeometryMixin` (~392 lines) is the shared geometry base of both
`YOLOOBBDetector` and `DetectionFilter`; `stages/obb.py` reimplements it and cites it
only as a parity reference.

## Decisions already made (this conversation)

1. **No duplicate detect API.** `InferenceRunner` already exposes the primitives:
   `run_realtime(frame) -> FrameResult` (single frame) and
   `run_batch_pass(range) -> None` (batched → cache). An OBB-only config is free to
   express (`headtail`/`cnn`/`pose`/`apriltag` are all optional in
   `config.py:_dict_to_config`), so eager model loading collapses to one model.
2. **`DetectionCacheBuilder` is deleted, not migrated.** Its job is exactly
   `run_batch_pass`. The optimizer scoring path already reads the runner's `OBBResult`
   cache (`optimizer.py:_filter_cached_detections`). This executes the DELETE finding
   that the legacy-batching plan's cache-builder-decision note defers as follow-up.
3. **Benchmarking is redesigned, not migrated.** Runtime selection is now three
   speed-ordered tiers (`resolver.py:13` `["cpu","gpu","gpu_fast"]`) resolved
   deterministically per platform, so the old per-model / per-runtime-string tool
   answers a question that no longer exists. **All in-UI benchmarking is removed**
   (the GUI twin `trackerkit/benchmarking.py` + its test). It is replaced by a **new
   external CLI tool** that:
   - takes a **single config file** and runs the **whole pipeline** (OBB + CNN + pose,
     end-to-end via `InferenceRunner`), not per-model, not one stage at a time;
   - runs it at each of the three **tiers** (cpu / gpu / gpu_fast) — a uniform tier per
     run, not per-stage runtime control;
   - reports **end-to-end timing**, and can include **engine-build-time diagnostics**
     (this absorbs the old `--compile-benchmark` capability).

   Because it drives the full runner from a config, it needs no `YOLOOBBDetector` and
   no low-level `load_obb_executor` entrypoint — it is a consumer of the one pipeline,
   which also kills the `benchmark_models.py` `__new__`-without-`__init__` internals
   poking outright.

## Open decisions (resolve before writing the plan)

- **D4 — filtering/geometry final home.** `apply_detection_filter` (over
  `stages/filtering`) is the public replacement for `filter_raw_detections`, but
  `OBBGeometryMixin` also computes geometry. Confirm `stages/obb`+`stages/filtering`
  cover the full mixin surface the optimizer and pose-pipeline consumers need before
  deleting it.

## Phases

Ordered by dependency. A blocks B blocks C/E; D is independent.

### Phase A — Port the runtime layer out of `core/detectors` (the blocker)

- Relocate `_direct_obb_runtime.py` into `core/inference` (e.g.
  `core/inference/direct_executors.py`), carrying `create_direct_obb_executor`,
  `create_direct_detect_executor`, and the `Direct*Executor` classes.
- Repoint `core/inference/runtime_artifacts.py:241` at the new location; fix the false
  module docstring (`:9`) that claims it never imports from `core/detectors`.
- Repoint the three `tools/diag_*`/`compare_runtimes.py` scripts (or let them die with
  the benchmarking deprecation).
- **Result:** `core/inference` no longer imports `core/detectors`. This is the single
  change that makes deletion possible at all.

### Phase B — Shared config + a batched detect entrypoint (no new pipeline)

- Promote `TrackingWorker._build_inference_config_from_params` (`worker.py:1014`) into
  a public `core/inference` helper; support building an OBB-only config.
- Fix `core/inference/api.py`: `predict_pose_for_image` imports a nonexistent
  `_load_pose_model` (real name `load_pose_model`, `stages/pose.py:63`) and has a
  wrong arity/return contract — it has never run. This is also a prerequisite the
  stale deletion-step brief already depends on.
- Add a **batched-in-memory detect method** to `InferenceRunner` (returns results
  directly, no cache on disk) for the dataset-generation throughput path. One method
  on the existing pipeline sharing `stages/obb` — not a second API. (Decided.)

### Phase C — Migrate the remaining YOLO consumers onto the runner

- `dataset_generation.py` — `YOLOOBBDetector` → runner (OBB-only config), via the
  batched-in-memory entrypoint from Phase B. Preserve the per-batch dataset conf/iou
  override
  (`DATASET_YOLO_CONFIDENCE_THRESHOLD` etc.) by baking it into the config.
- `optimizer_workers.py` — **delete `DetectionCacheBuilderWorker`**; callers use
  `run_batch_pass`. Confirm the Parameter-Helper GUI path that drives it is rewired.
- `preview_worker.py` — replace the `_preview_run_yolo_branch` clone with
  `run_realtime` (OBB + overlays config). This removes the fake `RuntimeContext`
  (:577), the three `_advanced_config_value` reach-ins, and the :1060 divergence.
  Keep only preview-specific drawing and the bg-sub measurer branch (already migrated
  by the bg-sub plan).
- `model_test_dialog.py` — replace private `stages.obb` imports with a single-frame
  `run_realtime`.
- `detectkit/gui/prediction_preview.py` — replace raw `ultralytics.YOLO` with
  `run_realtime` (OBB-only); gains ONNX/TRT/CoreML for free.
- Pose backend duplication ×4 (`posekit/gui/workers.py`, `preview_worker.py`,
  `crops_worker.py`, `trackerkit/benchmarking.py`) → one public shim over
  `stages/pose.py:load_pose_model` (unblocked by the api.py fix in Phase B).
- Filtering consumers (`optimizer.py`, `optimizer_workers.py:94/118`,
  `pose_pipeline.py:408`) → `api.apply_detection_filter`, per **D4**.

### Phase D — Replace benchmarking (independent)

- Delete the GUI benchmarking twin `trackerkit/benchmarking.py` and its test — all
  in-UI benchmarking goes away.
- Delete the old `tools/benchmark_models.py` (removes the ×10 `YOLOOBBDetector`
  constructions and the `__new__` internals poking).
- Build the **new external CLI benchmark** per Decision 3: config-file in → runs the
  whole `InferenceRunner` pipeline at each tier (cpu / gpu / gpu_fast) → end-to-end
  timing table, with an opt-in engine-build-time diagnostic (absorbs the old
  `--compile-benchmark`). Drives the runner only; no low-level executor access.

### Phase E — Retire `core/detectors`

Once Phases A/C/D land, no live consumer remains. In dependency order:

- Delete `yolo_detector.py` (its `RuntimeArtifactMixin` dependency
  `_runtime_artifacts.py` dies with it).
- Delete `_runtime_artifacts.py`.
- Resolve `_obb_geometry.py` / `detection_filter.py` per **D4** (delete, or fold the
  residual geometry into `stages/`).
- Move `_advanced_config_value` from `_utils.py` to a public home in
  `core/inference/config`; delete `_utils.py`.
- `git rm -r src/hydra_suite/core/detectors/` and drop remaining re-exports from
  `__init__.py` chains.

### Phase F — Tests + verification

- `tests/test_detectors_engine.py` importlib-loads `_direct_obb_runtime`,
  `_runtime_artifacts`, and `yolo_detector` by file path — migrate to the new
  locations or replace. (This file is absent from the prior plans' test-migration
  tables — a known accounting gap.)
- Keep the negative-assertion tests (`test_trackerkit_preview_worker.py`,
  `test_model_test_dialog.py`, `test_trackerkit_benchmarking.py`) that assert no
  `YOLOOBBDetector` is constructed — they now guard the migrated code.
- Final gates: `grep -rn "core.detectors\|core/detectors" src/ tests/` → empty;
  full `pytest -m "not benchmark"`; and a **frame-by-frame parity run** on a real
  video against a pre-migration baseline (mirroring the original redesign's Task 18
  gate: `assert (diff < 1e-4).all()`).

## Also closes

- **Retire `plans/2026-07-16-inference-redesign-deletion-step.md`.** It predates the
  bg-sub plan, would keep files that plan deletes, its parent spec's deletion list
  omits the four private modules that postdate it, and it warns its own final-scan
  greps "pass vacuously." This spec supersedes it; fold any live remnant (the
  `api.py` pose helper, pose-backend deletions) into Phases B/C/E.

### Deferred follow-ups inherited from the bg-sub stage work (merge 939e91d)

The bg-sub-as-first-class-stage branch (`docs/superpowers/specs/2026-07-16-bgsub-inference-stage-design.md`)
left three follow-ups whose natural home is this project. Two others from that
review were quick fixes, landed separately in the bg-sub follow-up branch
(GPU-vs-CPU lighting rounding parity in `utils/image_processing.py`; deletion of
the dead `tracking_counts`/`total_cost` chain in `worker.py`).

- **Move the Qt worker classes out of `core/background/optimizer.py` (Phase A/E).**
  `optimizer.py` (relocated from `core/detectors/bg_optimizer.py`, moved as-is)
  defines two classes that structurally inherit `QThread` —
  `BgSubtractionOptimizer(QThread)` (`:523`) and `BgDetectionPreviewWorker(QThread)`
  (`:846`), with `PySide6` `Signal`s. This is a `core`→GUI-framework
  dependency-direction violation baked into the class hierarchy, not a stray
  import. Fix: separate the pure Optuna/optimization logic (stays in `core`) from
  the Qt wrapper (moves to the `trackerkit` app layer), or rewrite the workers not
  to inherit `QThread`. This is the same class of "Qt worker sitting in `core`"
  cleanup Phase A already contemplates for the runtime layer — do them together.

- **Retire the now-orphaned `MIN_TRACKING_COUNTS` / `min_track_seconds` param
  (config-deprecation, pair with Phase C or F).** The bg-sub work deleted its only
  reader (the `tracking_stabilized` latch), so this user-facing knob is now a
  no-op. It still ships in `resources/configs/default.json`,
  `resources/configs/ooceraea_biroi.json`, `trackerkit/cli_config.py:712`, and
  `trackerkit/gui/orchestrators/config.py:2248`. Removing it is a user-facing
  config change: existing user configs carry the key and the GUI exposes a control,
  so it needs a deprecation path (accept-and-ignore, or a config migration), not a
  bare delete. (The internal dead variable it drove is already removed in the
  bg-sub follow-up branch; only the user-facing param remains.)

- **Re-cover the legacy detector-init path (Phase F — Tests + verification).** Task 13
  of the bg-sub work deleted `tests/test_tracking_worker_realtime_live_features.py`
  (210 lines) because it monkeypatched removed symbols (`create_detector`,
  `DetectionCache`). The behavior it exercised (backward-cached runs skipping
  runtime detector init; cache-write mode when reuse is disabled) now lives in
  `core/tracking/ingest/detection_phase.py`. Write fresh tests against that ingest
  path rather than restoring the old ones — the old assertions targeted internals
  that no longer exist.
