# One-Click Inference Calibration, Decoupled from the Tracking Run

**Status:** pending implementation plan
**Date:** 2026-09-08
**Revised:** 2026-09-08 after adversarial review (14 findings, 2 critical) and a widened goal
**Supersedes the UX of:** `docs/developer-guide/inference-autotuner.md` (modes + CUDA-only sections)

## Problem

Inference calibration currently runs *inside* a tracking run.
`_resolve_inference_autotune_before_load` (`core/tracking/worker.py:104-248`) fires
before `InferenceRunner` loads any model, and in `automatic` or `record` mode may spend
up to 4500 s measuring before a single production frame is tracked.

The feature is also unavailable in most real configurations:

| Configuration | Today |
|---|---|
| CUDA, forward-only | works |
| MPS / CPU | calibrates in `record`, **never applies** (`integration.py:618-624`) |
| gpu_fast / TensorRT | applies, but the profile key varies with free memory (see below) |
| backward-enabled project | declines entirely (`worker.py:1363-1375`) |
| realtime | never applies (`integration.py:604-607`) |
| cached-detection replay | never applies (`integration.py:608-611`) |

So the shipped feature is a three-state dropdown that, on most projects and on this MPS
box, does nothing — while still being able to consume a 75-minute budget mid-run.

The profile *store* needs no structural change: profiles live in
`get_data_dir()/inference_tuning_profiles/` (`store.py:46`) and are keyed by hardware,
model, and workload, never by project path. Cross-project reuse already works. What is
missing is a way to produce a profile without tracking, and a way to apply one
everywhere.

## Goal

**One click calibrates; every run afterwards just applies the result** — on any video,
any backend (`cpu` / `gpu` / `gpu_fast`), forward or backward.

1. A tracking run performs **lookup only**. It never measures and never claims the
   calibration lock.
2. Calibration is an **explicit user action**: one "Calibrate…" button, plus a
   `trackerkit calibrate` subcommand for headless boxes.
3. Calibration and lookup **agree on the fingerprint** — the correctness crux.
4. A backward pass applies **the same vector the forward pass used**, by construction.

## Non-goals

- Changing what is tuned. The six-field `InferenceTuningSettings` vector, the coordinate
  search, the ≥2 % gain gate, and the CSV equivalence gate are unchanged.
- Calibrating without a video. Each trial runs real tracking over striped frame windows
  and compares real CSV output.
- Making every configuration *calibratable*. Realtime, contention, and thermal throttle
  remain reasons not to measure. They stop being reasons not to apply.

## Design

### 1. Separate "may calibrate" from "may apply"

This is the change that unlocks the widened goal. `integration.py:601-630` computes
`eligible` and `allow_cached_reuse` in one if/elif chain, so every branch that forbids
measuring also forbids applying. Split them:

| Condition | May calibrate | May apply | Change |
|---|---|---|---|
| non-CUDA (`mps`, `cpu`) | yes | **yes** | delete the branch at `:618-624` |
| backward pass (`cache_replay`) | no | **yes, from the forward vector** | §4 |
| realtime | no | **yes** | `:604-607` clears only `eligible` |
| contention / thermal throttle | no | yes | already `eligible`-only |
| preview | no | yes | preview declines calibration, not lookup |
| baseline admission failed | no | no | unchanged — the vector does not fit in memory |

Applying is nearly free and is what makes a replayed cache *consistent*. Calibrating is
the expensive, environment-sensitive act. Only the latter needs guarding.

Non-CUDA rationale: the guards that establish correctness and benefit are device-neutral
— the CSV equivalence gate (`equivalence.py`, with `TrackID`/`TrajectoryID`/`State`/
`ArenaID` mandatory-exact), the determinism floor, and the acceptance gate requiring
≥2 % median gain with a bootstrap lower bound above zero. A candidate that is slower on
MPS is rejected by construction. The sidecar's MPS plumbing is already exercised today,
since `record` mode calibrates on MPS
(`docs/developer-guide/inference-autotuner.md:55-56, 72-75`). Expect many MPS
calibrations to select the baseline and store a no-op profile — cross-frame batching has
measured up to 1.58× *slower* on MPS
(`docs/developer-guide/performance-tuning.md:49`). The UI must report that as "no
improvement available", not as failure.

Caveat to document, not fix: on MPS the accelerator admission gate is skipped
(`candidates.py:214`, because the MPS probe reports no accelerator byte total), leaving
the host cost model plus the child's `mps_high_watermark_ratio=0.7`
(`sidecar.py:582-584`) as the only limits. OOM is classified and recorded as a candidate
rejection (`process_supervisor.py:1420-1422`), so this degrades to a wasted trial rather
than a wrong answer.

### 2. Fingerprint agreement

The key is built from an *ephemeral* params dict mutated at `worker.py:147-165`. Four
things differ between "calibrate now" and "track later", not the one the first draft of
this spec identified:

**(a) `density_is_estimated`** (`fingerprint.py:142`) is a raw boolean inside the digest
(`:198-199`, `:238-242`). Fresh-video calibration stores `True`; a later cached run
computes `False`. Always a miss. Remove it from `WorkloadKey`; keep it as a field on
`InferenceTuningProfile`.

**(b) The density buckets themselves.** `integration.py:449-454`: with no cache,
`detection_counts == crop_counts == (MAX_TARGETS,)`, so all four p50/p95 buckets collapse
to `bucket(MAX_TARGETS)`. With a cache they are distinct measured values. `MAX_TARGETS=12`
(bucket 16) against a real p50 of 8 (bucket 8) misses even with (a) fixed. Removing the
flag is necessary but **not sufficient** — the fresh-then-track case is precisely the one
this feature exists to serve.

Resolution: calibration seeds the workload from its own measured first block rather than
from `MAX_TARGETS`, and persists under the *measured* key. Where no measurement exists
yet, lookup falls back to a `MAX_TARGETS`-keyed probe and then the measured key, applying
whichever hits.

**(c) The S2 rekey reads the flag off the key.** `store.py:254` is
`if current.key.workload.density_is_estimated:`; the rekey-on-first-real-sample block
(`:243-270`) must be rewired to read the profile field instead. This is not a field move,
it is a rewrite of that block.

**(d) TensorRT profile id depends on live free memory.** `worker.py:198-209` computes
`artifact_batch_size = max(planner.values_for(...))` **after** the preflight key, mutating
the same dict object, and `values_for` filters through `admit`, which reads live
`available_host_bytes` / `available_accelerator_bytes` (`candidates.py:199-227`). That
value folds into `tensorrt_profile_id` (`integration.py:175-186`,
`runtime_artifacts.py:725-737`). So on gpu_fast the digest varies with how much memory
happens to be free. Fix: derive the artifact batch size from the *static* candidate-space
maximum for those fields, independent of the live observation, so the key is
reproducible. Fixture-based tests will not catch this — they pin the observation — so it
needs a dedicated test that varies free memory and asserts digest stability.

Underpinning all four: **one shared context builder**, `build_autotune_context(...)`,
called by both the calibrate entry point and the run-time lookup, fed from the same
`build_engine_params` output. The caller must also reproduce worker's own derivations of
`end_frame`, `effective_realtime_tracking_mode` (`worker.py:1124-1128`) and
`_resolve_cache_dir()` (`worker.py:5090-5096`) — the context builder takes these as
inputs, so agreement depends on the caller, and the calibrate path must call the same
helpers rather than re-deriving them.

### 3. `core/inference/autotune/session.py` (new)

Qt-free. `build_autotune_context(...)` is `worker.py:122-197` lifted, minus the `off`
early return. Then:

- `lookup(context)` — `resolve_tracking_inference_config` with `trial_executor=None`.
  Returns a `cache_hit` overlay or a baseline overlay. No lock, no measurement.
  Note the coordinator's ordering: `:152-160` returns `deferred_due_to_contention`
  *before* the `trial_executor is None` terminal at `:161-168`, so an ineligible miss
  reports "deferred", not "unavailable". Since eligibility no longer blocks applying
  (§1), reorder so lookup terminates in `unavailable` on a genuine miss.
- `calibrate(context, *, budget_seconds)` — constructs the `ContainedTrialExecutor`
  (`worker.py:198-234`) and drives the coordinator. Bypasses the 24 h `INCOMPLETE`
  negative cache (`coordinator.py:130-151`): that guard stops a *run* silently re-paying
  a failed budget, which does not apply when a human clicked a button. The guard stays in
  force for lookup.

`AutotuneRequest.mode` collapses from `{off, record, automatic}` to `{lookup, calibrate}`;
the `record` branches (`coordinator.py:117-129`, `:190-202`) and
`InferenceAutotunePolicy.mode`'s validation (`core/inference/config.py:94-103`) change
accordingly.

### 4. Backward-pass propagation

Today's decline (`worker.py:1341-1375`) is not a tuner limitation. Forward gets tuned to
`det=4`, writes its detection cache under a key that *includes* the batch size, then the
backward pass re-resolves at the project's untuned batch size, misses the key, and the
run aborts — measured on courtship.

A second independent lookup on the backward pass is **not** an acceptable fix: it
re-derives a key from a `cache_replay` context and could legitimately differ, silently
reintroducing the abort. Instead, **the forward pass persists its effective
`InferenceTuningSettings` into the run's inference-cache directory**
(`.inference_cache_<stem>/`) alongside the cache it wrote, and the backward pass reads
that vector and applies it verbatim, with no fingerprint involved. The cache key matches
because it is the same vector that wrote the cache.

If the sidecar file is absent (a cache written before this change), the backward pass
falls back to the project's configured values, which is exactly today's behaviour for an
untuned forward pass.

This makes backward-enabled projects fully supported and removes the `_backward_enabled`
gate. **Until this propagation is implemented, lookup must decline on backward-enabled
projects** — a lookup *hit* applies the same vector that caused the measured abort, so
the two changes ship together or not at all.

### 5. TrackingEngineCore

`_resolve_inference_autotune_before_load` is deleted; the run calls
`session.build_autotune_context` then `session.lookup`, guarded by
`config.apply_tuned_inference`. The fallback envelope (`worker.py:1416-1441`) and the
`_inference_autotune_stats` summary (`worker.py:251-315`) are retained.
`cancel_inference_autotune` (`worker.py:726-728`) and its forwarder
(`gui/workers/tracking_worker.py:71-73`) are deleted — the engine no longer calibrates.

**Production-throughput observation must be preserved.** `worker.py:4964-5006` calls
`store.observe_production_throughput(...)` at the end of every run carrying an overlay
`profile_id`; that is a locked `save()` (`store.py:233-309`) implementing the 15 %
regression-demotion rule and the S2 rekey. It is gated on `profiler.enabled`, which
`worker.py:993-995` derives from `INFERENCE_AUTOTUNE_MODE in {automatic, record}` — after
the rename that silently becomes dead unless `ENABLE_PROFILING`. Re-gate it on
`APPLY_TUNED_INFERENCE or ENABLE_PROFILING`.

Consequently the goal's "a run never writes to the store" is **narrowed**: a run never
writes a *profile*, but it does append a throughput observation and may demote a
regressed profile. The purity test asserts no `claim()` for calibration and no profile
`save()` from `lookup()` itself — not that the run process never touches the store.

### 6. Sidecar recursion guard

`sidecar.py:140` and `sidecar_child.py:141` set `INFERENCE_AUTOTUNE_MODE="off"` so trial
children never resolve. After the rename the children must set
`APPLY_TUNED_INFERENCE=False`; otherwise a trial that hits the cache applies a profile
and corrupts its own measurement, including the baseline block.

### 7. GUI

`trackerkit/gui/panels/setup_panel.py`, Performance section:

- **Removed:** `combo_inference_autotune` (`:755-774`), `spin_inference_autotune_budget`
  (`:776-801`), `btn_continue_inference_settings` (`:807-817`), the
  `inference_autotune_continue_requested` signal (`:48`), and its connection in
  `gui/main_window.py:931-932`.
- **Added:** a `QCheckBox` "Apply tuned inference profile if available" in the same
  toggle grid as "Reuse cache", and a "Calibrate…" `QPushButton`.
- **Retained:** `lbl_inference_autotune_status`, repurposed to report profile presence —
  "Tuned profile found — 1.16× measured", "No profile for this configuration", or "No
  improvement available on this device".

The button opens `CalibrationDialog(BaseDialog)` in
`trackerkit/gui/dialogs/calibration.py` with the budget spinbox, progress fed by the
search's `status_callback`, and Cancel; work runs in `CalibrationWorker(BaseWorker)`.
Parameters come from `build_engine_params` — the same call the run makes.

In `gui/orchestrators/tracking.py`, delete only `:362-363` and `:365-373`; `:356-361`
drives the progress bar and must stay. `:670-692` (stats rendering) is retained.

The dialog must refuse to start while a track is running, and vice versa: on MPS/CPU the
contention probe hardcodes `contention_detected=False` (`device.py:172-193`), so nothing
in the tuner will detect the overlap and the measurement would be garbage. A GUI-level
mutual exclusion is the only guard available.

### 8. Config, params, CLI

`trackerkit/config/schemas.py:51-53`: `inference_autotune_mode: str` →
`apply_tuned_inference: bool`, migrating `automatic`/`record` → `True`, `off` → `False`.
`engine_params.py:541-575` and `:1240-1242` emit `APPLY_TUNED_INFERENCE`; the singleflight
and stage-share knobs at `:1243-1256` are retained.

Every remaining consumer of the retired key must move together:
`gui/orchestrators/config.py:311-325, :1720`; `cli_config.py:216-246`
(`apply_inference_autotune_override`); `cli.py:51-72`; `core/inference/config.py:94-103,
:1313-1315`; `tools/equivalence/autotune_plumbing_probe.py:76`;
`tools/equivalence/run_matrix.sh:127`.

CLI (`app.py:121-145`): `--inference-autotune {off,record,automatic}` and
`--no-inference-autotune` become `--apply-tuned-inference` /
`--no-apply-tuned-inference`. New `trackerkit calibrate` subcommand takes `track`'s
video/config arguments plus `--budget-seconds`. Its interaction with `--gpus`/`--jobs`
fan-out is **unresolved** and must be settled during planning: the simplest correct
answer is that `calibrate` accepts neither and runs single-process, since concurrent
calibration on one box measures contention rather than throughput.

### 9. Documentation

`docs/developer-guide/inference-autotuner.md`: rewrite "The three modes" and "CUDA-only
policy" as "calibrate once, apply everywhere". Retain the budget, correctness gate,
determinism floor, and profile sections.

## Error handling

- **Lookup failure of any kind** falls back to configured values; the run proceeds. The
  envelope at `worker.py:1416-1441` already guarantees this.
- **Calibration failure** surfaces the coordinator's `reason` in the dialog and persists
  an `INCOMPLETE` record, so a *run* will not retry it but the button will.
- **Cancellation** returns `reason="cancelled"` and writes nothing
  (`coordinator.py:239-253` does not save on cancel).
- **Backward pass with no persisted vector** falls back to configured values.

## Testing

- **Fingerprint agreement** — one params dict and video, run through the calibrate path
  and the lookup path, produce an identical `key.digest`. Cases: cache-absent vs
  cache-present (must agree once (a) and (b) are fixed), `RESULT_CACHE_STAGE_MASK` set
  (must differ), and **free memory varied on a TensorRT-backed context (must agree)**.
- **Lookup purity** — with `trial_executor=None` and a fake store: no `claim()`, no
  profile `save()`, no executor construction, for both hit and miss.
- **Backward propagation** — a forward pass that applies a non-baseline vector writes it
  to the cache dir; the backward pass reads it and resolves the same inference cache key.
  This is the regression test for the measured courtship abort.
- **Negative-cache asymmetry** — an `INCOMPLETE` record within 24 h defers lookup, does
  not defer an explicit calibrate.
- **Apply on every backend** — a validated profile with an MPS-shaped key is applied;
  same for cpu and for a realtime run.
- **Sidecar recursion** — a trial child resolves with `APPLY_TUNED_INFERENCE=False` even
  when a validated profile exists for its key.
- **Throughput observation still fires** with `apply_tuned_inference=True` and
  `ENABLE_PROFILING` unset.
- **Config migration** — each legacy mode value loads to the right boolean.
- **Full autotune suite** — all `tests/test_inference_autotune_*.py`,
  `test_trackerkit_inference_autotune_surfaces.py`, plus
  `tests/test_trackerkit_panels_smoke.py:106-150`. Run in full, never a subset chosen by
  apparent relevance: the reflective contract guards break on any field change.
- **Regenerate the characterization goldens** —
  `tests/data/get_parameters_dict_golden/{ant_cnn_identity,fly_obb}.json` contain
  `"INFERENCE_AUTOTUNE_MODE": "off"` and `test_get_parameters_dict_characterization.py`
  will fail until they are regenerated and the diff reviewed.
- **Delete** `tests/test_inference_autotune_integration.py:258`, which asserts the
  CUDA-only reason string.
- **Equivalence gate** — `tools/equivalence/run_matrix.sh` on MPS and CUDA (mehek) with
  `apply_tuned_inference` off must stay byte-identical, plus at least one
  backward-enabled clip with it on.

## Consequences

- A tracking run's duration no longer depends on whether a profile exists.
- The `WorkloadKey` change bumps `TUNING_SCHEMA_VERSION`. `store.py:128` returns `None`
  for a stale schema rather than crashing, but stale files are only reclaimed by the
  512-record mtime cap (`:311-326`); add a schema-mismatch unlink.
- For an editable dev checkout, `hydra_code_identity()` (`fingerprint.py:382`) already
  invalidates profiles on any source edit, so the bump costs nothing. For a **non-editable
  install** the identity comes from `HYDRA_BUILD_COMMIT` or `direct_url.json`
  (`:385-394`) and is stable — so for the "one stable install, one rig, many videos" case
  this design targets, the bump is a real one-time recalibration.
- Profiles remain sensitive to video geometry, detector thresholds, and slice geometry.
  Reuse works across videos on the same rig with the same configuration.
- The `record` concept disappears: calibrating without applying is Calibrate with the
  apply checkbox off.
