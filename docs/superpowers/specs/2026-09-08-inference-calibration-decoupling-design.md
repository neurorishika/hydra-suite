# Decoupling Inference Calibration from the Tracking Run

**Status:** pending implementation plan
**Date:** 2026-09-08
**Supersedes the UX of:** `docs/superpowers/specs/done/*inference-autotuner*`, `docs/developer-guide/inference-autotuner.md` (modes section)

## Problem

Inference calibration currently runs *inside* a tracking run. `TrackingEngineCore.run`
calls `_resolve_inference_autotune_before_load` (`core/tracking/worker.py:104-248`)
before `InferenceRunner` loads any model; in `automatic` or `record` mode that call may
spawn a sidecar search that spends up to 4500 s measuring, before a single production
frame is tracked.

Three consequences:

1. **A tracking run is not pure inference.** Its wall-clock time, its resource
   footprint, and its failure modes all depend on whether a profile happens to exist.
2. **Calibration is not user-triggered.** It is a side effect of a mode dropdown, so the
   only way to ask for measurement is to start a run you may not want.
3. **The GUI carries run-time calibration controls** — a three-state dropdown, a budget
   spinbox, a status label, and a "Continue with current settings" escape hatch — that
   exist solely to manage an activity that should not be happening during the run.

The profile *store* is already correct and needs no change in shape: profiles live in
`get_data_dir()/inference_tuning_profiles/` (`core/inference/autotune/store.py:46`) and
are keyed by hardware, model, and workload — never by project path. Cross-project reuse
already works. What is missing is a way to *produce* a profile without tracking, and a
way to *apply* one without risking calibration.

## Goal

- A tracking run performs **lookup only**: it either finds a validated profile and
  applies it, or runs with configured values. It never measures, never claims a lock,
  never writes to the store.
- Calibration is an **explicit user action**: a button in the Setup panel and a CLI
  subcommand. It calibrates on the currently configured video with the same parameters
  the run would use.
- Calibration and lookup **agree on the fingerprint**. This is the correctness crux of
  the whole design.

## Non-goals

- Changing what is tuned. The six-field `InferenceTuningSettings` vector, the coordinate
  search, the gain gate, and the CSV equivalence gate are unchanged.
- Changing where or how profiles are stored, beyond one schema-version bump.
- Making calibration work without a video. Each trial runs real tracking over striped
  frame windows and compares real CSV output; synthetic input would not measure the
  thing we care about.

## The fingerprint crux

The profile key is built from an *ephemeral* params dict that `worker.py:147-165`
mutates before constructing `TrackingRunContext`. Two of those mutations differ between
"calibrate now" and "track later":

| Field | Calibrate on a fresh video | Track later with cache reuse |
|---|---|---|
| `INFERENCE_AUTOTUNE_DETECTION_COUNTS` | absent → density falls back to `bucket(MAX_TARGETS)`, and `density_is_estimated=True` | sampled from the detection cache → `density_is_estimated=False` |
| `RESULT_CACHE_STAGE_MASK` | unset | `("detector",)` |

Either difference changes `key.digest`, so the run misses the profile the user just
paid for and the Calibrate button appears to do nothing. Density itself is bucketed at
powers of two (`count_bucket`, `fingerprint.py:274`), so a modest density change does not
by itself break the match — but `density_is_estimated` is a raw boolean and *always*
breaks it.

Resolution, in three parts:

1. **One shared context builder.** Extract `build_autotune_context(...)` from
   `worker.py:104-248`. Both the calibrate entry point and the run-time lookup call it
   with the same inputs, derived from the same `build_engine_params` output.
2. **Remove `density_is_estimated` from the key.** It is provenance, not workload
   (`fingerprint.py:137-142`). Keep it as a field on `InferenceTuningProfile` so the UI
   can report "measured against an estimated density". This bumps
   `TUNING_SCHEMA_VERSION`, invalidating existing records — acceptable, because
   `hydra_code_identity()` (`fingerprint.py:382`) already invalidates every profile on
   any source change.
3. **Keep `RESULT_CACHE_STAGE_MASK` in the key.** When the detector stage is replayed
   from cache it does not run, so the optimum genuinely differs. Calibration mirrors
   whatever the project's cache-reuse setting currently is, yielding one profile per
   cache mode. Both are cheap after the first.

Density buckets themselves stay in the key. Four of the six tuned fields
(`pose_batch_size`, `headtail_batch_size`, and the per-label `identity_batch_sizes`)
batch animals rather than frames, and the planner's memory admission is computed from
peak instance count — so crops-per-frame is load-bearing, not incidental.
`detection_batch_size` alone is frames-only.

## Design

### 1. `core/inference/autotune/session.py` (new)

Three public functions, all Qt-free:

```python
def build_autotune_context(config, params, *, video_path, frame_width, frame_height,
                           start_frame, end_frame, realtime, cache_dir,
                           use_cached_detections, cache_read_only_replay,
                           should_cancel, status_callback) -> AutotuneContext
```

Returns a value object carrying the resolved backend, the runtime probe, the ephemeral
params dict, and the `TrackingRunContext`. This is `worker.py:122-197` lifted verbatim
apart from the `mode == "off"` early return, which moves to the callers.

```python
def lookup(context) -> tuple[InferenceConfig, InferenceRuntimeOverlay, ResolveResult]
```

Calls `resolve_tracking_inference_config` with `trial_executor=None`. Returns a
`cache_hit` overlay when a validated profile exists, otherwise a baseline overlay with
`status="unavailable"`. Guarantees: no single-flight claim, no store write, no
measurement. The coordinator already returns the cache hit at `coordinator.py:109-116`,
before the eligibility check at `:152`, so a hit is served even when the live device
would be ineligible to calibrate.

```python
def calibrate(context, *, budget_seconds) -> ResolveResult
```

Constructs the `ContainedTrialExecutor` (`worker.py:198-234`) and drives the coordinator
to completion. Because the user asked explicitly, calibration **bypasses the 24 h
`INCOMPLETE` negative cache** (`coordinator.py:130-151`) — that guard exists to stop a
run silently re-paying a failed budget, which no longer applies when a human clicked a
button. The guard stays in force for lookup.

### 2. Coordinator and request changes

`AutotuneRequest.mode` collapses from `{off, record, automatic}` to an intent enum
`{lookup, calibrate}`. In `coordinator.resolve`:

- the `record`-mode branches (`:117-129`, `:190-202`) are deleted;
- `trial_executor is None` (`:161-168`) becomes the documented lookup terminal, returning
  `status="unavailable"` at INFO rather than implying misconfiguration;
- `calibrate` intent skips the `INCOMPLETE` early return.

`build_tracking_autotune_request` (`integration.py:601-664`) loses its `record` special
case and its non-CUDA branch (below).

### 3. Lift the CUDA-only apply policy

`integration.py:618-624` currently sets `eligible=False` **and**
`allow_cached_reuse=False` when the accelerator is not CUDA, so on MPS a validated
profile is never applied even on a direct hit. Delete this branch.

The guards that remain are the ones that actually establish correctness and benefit: the
CSV equivalence gate (`equivalence.py`, including the mandatory-exact
`TrackID`/`TrajectoryID`/`State`/`ArenaID` columns), the determinism floor, and the
acceptance gate requiring ≥2 % median throughput gain with a bootstrap lower bound above
zero. A candidate that is slower on MPS is rejected by construction.

Expect many MPS calibrations to select the baseline and store a no-op profile. That is a
correct and useful result — cross-frame batching has been measured at up to 1.58×
*slower* on MPS (`docs/developer-guide/performance-tuning.md`) — and the UI should report
"no improvement found" rather than treating it as a failure.

The realtime, `cache_replay`, contention, thermal-throttle, and baseline-admission
ineligibility branches are untouched.

### 4. TrackingEngineCore

`_resolve_inference_autotune_before_load` is deleted. The run path calls
`session.build_autotune_context(...)` then `session.lookup(...)`, guarded by
`config.apply_tuned_inference`. The surrounding fallback envelope
(`worker.py:1416-1441`) and the `_inference_autotune_stats` summary
(`worker.py:251-330`) are retained unchanged; `cancel_inference_autotune`
(`worker.py:726-728`) is deleted from the engine, since the engine no longer calibrates.

The preview-mode, backward-pass, and backward-enabled-project declines
(`worker.py:1332-1375`) collapse into the calibrate path: a lookup is harmless in all
three cases, but the Calibrate button must refuse a backward-enabled project with the
same explanation it gives today.

### 5. GUI

`trackerkit/gui/panels/setup_panel.py`, Performance section:

- **Removed:** `combo_inference_autotune` (`:755-774`), `spin_inference_autotune_budget`
  (`:776-801`), `btn_continue_inference_settings` (`:808-817`), and the
  `inference_autotune_continue_requested` signal (`:48`).
- **Added:** a `QCheckBox` "Apply tuned inference profile if available", sitting in the
  same toggle grid as "Reuse cache"; and a "Calibrate…" `QPushButton` beside it.
- **Retained:** `lbl_inference_autotune_status`, repurposed to report profile presence
  ("Tuned profile found — 1.16× measured" / "No profile for this configuration").

The button opens a modal `CalibrationDialog(BaseDialog)` in
`trackerkit/gui/dialogs/calibration.py` holding the budget spinbox
(`MINIMUM_`/`MAXIMUM_CALIBRATION_BUDGET_SECONDS`, default unchanged), a progress area fed
by the search's `status_callback`, and Cancel. Work runs in
`CalibrationWorker(BaseWorker)` under `trackerkit/gui/workers/`, whose `should_cancel`
closure is driven by the dialog's Cancel button — the same mechanism the retired
Continue button used.

Parameters come from `build_engine_params`, the identical call the run makes, which is
what makes the fingerprint agreement hold in practice rather than only in principle.

`gui/orchestrators/tracking.py:356-373` (Continue-button show/hide and
`continue_with_current_inference_settings`) is deleted; `:669-692` (stats rendering) is
retained.

### 6. Config and CLI

`trackerkit/config/schemas.py:51-53`: `inference_autotune_mode: str` becomes
`apply_tuned_inference: bool`. Load-time migration maps `automatic` and `record` → `True`
and `off` → `False`; `inference_autotune_budget_seconds` and
`inference_autotune_manual_fields` are retained (the budget now applies to calibration).

`engine_params.py:541-575` and `:1240-1247` emit `APPLY_TUNED_INFERENCE` in place of
`INFERENCE_AUTOTUNE_MODE`.

CLI (`trackerkit/app.py:121-145`):

- `--inference-autotune {off,record,automatic}` and `--no-inference-autotune` are
  replaced by `--apply-tuned-inference` / `--no-apply-tuned-inference`.
- A new `trackerkit calibrate` subcommand takes the same video/config arguments as
  `track`, plus `--budget-seconds`, and runs `session.calibrate` to completion. This is
  the headless equivalent of the button and the supported path on the CUDA box.

### 7. Documentation

`docs/developer-guide/inference-autotuner.md`: the "three modes" and "CUDA-only policy"
sections are rewritten as "calibrate, then apply". The budget, correctness gate,
determinism floor, and profile sections are retained.

## Error handling

- **Lookup failure of any kind** falls back to configured values and the run proceeds.
  The existing envelope at `worker.py:1416-1441` already guarantees this; the change only
  narrows what can go wrong inside it.
- **Calibration failure** is surfaced in the dialog with the coordinator's `reason`, and
  persists an `INCOMPLETE` record exactly as today, so a *run* will not retry it.
- **Cancellation** returns the partial search's `reason="cancelled"`, writes nothing, and
  leaves the project's configured settings in force.
- **A backward-enabled project** refuses calibration with the current explanation.

## Testing

- **Fingerprint agreement (the load-bearing test).** Given one params dict and video,
  `build_autotune_context` fed through the calibrate path and through the lookup path
  produce an identical `key.digest`. Includes the cache-present and cache-absent cases,
  asserting they agree once `density_is_estimated` leaves the key and differ when
  `RESULT_CACHE_STAGE_MASK` is set.
- **Lookup purity.** With `trial_executor=None` and a fake store, assert no `claim()`, no
  `save()`, and no executor construction, for both hit and miss.
- **Negative-cache asymmetry.** An `INCOMPLETE` record within 24 h defers a lookup and
  does not defer an explicit calibrate.
- **Non-CUDA apply.** A validated profile with an MPS-shaped key is applied by lookup.
- **Config migration.** Each legacy `inference_autotune_mode` value loads to the right
  boolean.
- **Full autotune suite** (~20 `tests/test_inference_autotune_*.py` plus
  `test_trackerkit_inference_autotune_surfaces.py`) — run in full, never a subset chosen
  by apparent relevance, because the reflective contract guards break on any field
  change.
- **Equivalence gate.** `tools/equivalence/run_matrix.sh` on MPS and on CUDA (mehek) with
  `apply_tuned_inference` off must stay byte-identical: the run path changed, so the
  no-tuning path must be proven unaffected.

## Consequences

- A tracking run's duration no longer depends on whether a profile exists.
- The schema bump invalidates existing stored profiles. Given that
  `hydra_code_identity()` already invalidates them on any source edit, this costs
  nothing in practice.
- Profiles remain sensitive to hydra's own source digest, video geometry, detector
  thresholds, and slice geometry. Reuse works for "one stable install, one rig, many
  videos" — the common lab case — and not for a developer editing this repository.
- The `record` concept disappears. Calibrating without applying is now expressed by
  clicking Calibrate with the apply checkbox off.
