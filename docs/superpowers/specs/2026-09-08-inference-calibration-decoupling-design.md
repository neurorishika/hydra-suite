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

The key is built from an *ephemeral* params dict mutated at `worker.py:147-165`.

**(a) Density is already handled — do NOT remove `density_is_estimated` from the key.**
An earlier draft of this spec proposed exactly that. It would have deleted a working
mechanism. The S2 design is a deliberate **two-record bridge**:

1. With no detection cache, `_counts` falls back to `(MAX_TARGETS,)`
   (`integration.py:450-463`), so all four p50/p95 buckets collapse to
   `bucket(MAX_TARGETS)` and the key carries `density_is_estimated=True`.
2. At the end of a run, `observe_production_throughput` (`store.py:243-299`) sees real
   counts, builds the measured `WorkloadFingerprint`, and — *because* the key says the
   old one was an estimate — writes a **second** record under the measured key
   (`store.py:293-300`), deliberately leaving the estimated-key record in place so "the
   next brand-new video, which also has no cache yet, still gets a warm start from it"
   (`store.py:263-266`).
3. A later run that does have a cache computes the measured key and hits record 2.

So the flag is load-bearing *as a key component*: it is what distinguishes "re-key this,
it was never measured" from "demote this, reality changed". Removing it turns every
first-run density correction into a spurious regression demotion.

The consequence for this design is that the **bridge must keep running**, which makes
§5's profiler-gate fix load-bearing rather than incidental: if
`observe_production_throughput` stops firing, record 2 is never written and every cached
run misses forever.

Calibration should additionally **close the bridge in one shot** rather than waiting for
a subsequent run: after a successful search, the calibrate path calls
`observe_production_throughput` with the density its own trials measured, producing both
the estimated-key and measured-key records immediately. One click then serves the
cache-less first run *and* every cached run afterwards.

**(b) `RESULT_CACHE_STAGE_MASK`** genuinely changes the optimum — a replayed detector
stage does not run — and stays in the key. Calibration mirrors the project's current
cache-reuse setting, so a project yields one profile per cache mode.

**(c) TensorRT profile id depends on live free memory.** `worker.py:198-209` computes
`artifact_batch_size = max(planner.values_for(...))` **after** the preflight key, mutating
the same dict object, and `values_for` filters through `admit`, which reads live
`available_host_bytes` / `available_accelerator_bytes` (`candidates.py:199-227`). That
value folds into `tensorrt_profile_id` (`integration.py:175-186`,
`runtime_artifacts.py:725-737`). So on gpu_fast the digest varies with how much memory
happens to be free. Fix: derive the artifact batch size from the *static* candidate-space
maximum for those fields, independent of the live observation, so the key is
reproducible. Fixture-based tests will not catch this — they pin the observation — so it
needs a dedicated test that varies free memory and asserts digest stability.

Underpinning all three: **one shared context builder**, `build_autotune_context(...)`,
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
  and the lookup path with the same cache state, produce an identical `key.digest`.
  Cases: `RESULT_CACHE_STAGE_MASK` set (must *differ*, by design), and **free memory
  varied on a TensorRT-backed context (must agree)**.
- **The two-record bridge, end to end** — calibrate with no detection cache, assert both
  an estimated-key and a measured-key record exist; then assert a cache-less run and a
  cache-present run each hit one of them. This is the test that proves "one click, then
  every later run just works", and it is the whole feature.
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
- **No key-schema change is required.** Because `density_is_estimated` stays in the key
  (§2a), `WorkloadFingerprint` is untouched and `TUNING_SCHEMA_VERSION` does not move, so
  existing stored profiles remain valid. This is a direct improvement over the first
  draft, which would have invalidated every record for no gain.
- Stale-schema records are still only reclaimed by the 512-record mtime cap
  (`store.py:311-326`); `store.py:128` returns `None` rather than crashing. Adding a
  schema-mismatch unlink is a cheap independent tidy-up, not a requirement here.
- For an editable dev checkout, `hydra_code_identity()` (`fingerprint.py:382`)
  invalidates profiles on any source edit, so this branch's own code changes will
  invalidate local profiles once. For a **non-editable install** the identity comes from
  `HYDRA_BUILD_COMMIT` or `direct_url.json` (`:385-394`) and is stable across videos —
  which is the "one stable install, one rig, many videos" case this design targets.
- Profiles remain sensitive to video geometry, detector thresholds, and slice geometry.
  Reuse works across videos on the same rig with the same configuration.
- The `record` concept disappears: calibrating without applying is Calibrate with the
  apply checkbox off.
