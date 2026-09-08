# Inference Throughput Autotuner

The inference autotuner measures a project's own inference pipeline on the
project's own video and, when it can prove the result is output-equivalent,
selects faster batch sizes and pipeline depth than the configured defaults.

It is implemented in `src/hydra_suite/core/inference/autotune/`. Calibration
(measuring) and tracking (running) are separate actions: a tracking run only
ever performs a lookup against a stored profile — it never measures — via
`core/inference/autotune/session.py`, driven from `core/tracking/worker.py`.

## What it tunes

The tuned vector (`InferenceTuningSettings`, `models.py`) is:

| Field | Meaning |
| --- | --- |
| `detection_batch_size` | video frames per detector call |
| `slice_tile_batch_size` | SAHI tiles per call (sliced OBB only) |
| `pose_batch_size` | crops per pose backend call |
| `headtail_batch_size` | crops per head/tail classifier call |
| `identity_batch_sizes` | per-identity-model crops per call |
| `pipeline_depth` | inference pipeline overlap depth (1 = synchronous) |

Fields whose configured value is `None` (stage not in this project's pipeline)
are not tuned. Fields the user pinned in the GUI are passed as `manual_fields`
and are excluded from the search.

## Calibrate once, apply everywhere

Calibration is an explicit user action, not something a tracking run does to
itself. There is a "Calibrate…" button in TrackerKit's Setup panel (Performance
section), which opens `CalibrationDialog` and runs `CalibrationWorker` in the
background, and a headless `trackerkit calibrate` subcommand (see
`docs/user-guide/trackerkit-cli.md`) for boxes without a GUI. Both build engine
params through the same `build_engine_params` call `track` uses, then call
`core/inference/autotune/session.py`'s `calibrate(context, budget_seconds=...)`.
That is the correctness argument for cross-entry-point agreement: calibrate and
lookup fingerprint the same params dict through the same
`build_autotune_context(...)`, so a profile written by one entry point is found
by the other.

A tracking run never calibrates. It performs a **lookup only**
(`session.lookup(context)`, `trial_executor=None`) — no lock is claimed, no
measurement runs, and lookup terminates in `unavailable` on a genuine miss
rather than blocking. Whether a run applies a profile it finds is a single
boolean: `config.apply_tuned_inference`, emitted as the engine param
`APPLY_TUNED_INFERENCE`. The mode vocabulary `{off, record, automatic}` is
gone; `record` is expressed by calibrating with the GUI's apply checkbox
(or `--apply-tuned-inference`) off — a validated profile is written, but this
run does not apply it.

Because a run only ever looks up, it does not have the old modes' cost: a
run's duration no longer depends on whether a profile exists. It does still
call `store.observe_production_throughput(...)` at the end (gated on
`APPLY_TUNED_INFERENCE or ENABLE_PROFILING`), which can demote a profile
found to have regressed and is also half of the density bridge — see below.

### `eligible` vs `allow_cached_reuse`

The old CUDA-only restriction on *applying* a profile is gone. "May this
context calibrate" (measure) and "may this context apply" (use a stored
profile) are now independent:

| Condition | May calibrate (`eligible`) | May apply (`allow_cached_reuse`) |
| --- | --- | --- |
| non-CUDA (`mps`, `cpu`) | yes | yes |
| backward pass (`cache_replay`) | no | yes, from the forward pass's persisted vector (see below) |
| realtime | no | yes |
| contention / thermal throttle | no | yes |
| preview | no | yes |
| baseline admission failed (vector doesn't fit in memory) | no | no |

Applying a stored, validated profile is cheap and device-neutral — the same
correctness gate (CSV equivalence + determinism floor + the ≥2% gain gate, all
below) already proved the vector is safe before it was ever saved. Only
*measuring* is environment-sensitive, so only measuring is guarded by
realtime/contention/throttle/cache-replay.

On MPS and CPU, calibration now actually runs the full search (previously
only `record` mode did this) and can validate and apply a profile, exactly
like CUDA. Do not read a calibration that ends in "no improvement available"
as a bug: on MPS, cross-frame batching has been measured up to 1.58× *slower*
than the per-frame baseline (`docs/developer-guide/performance-tuning.md:49`),
so a candidate that loses on this device is expected, and the acceptance gate
(below) correctly rejects it and stores a no-op profile. The GUI status label
reports this case as "No improvement available on this device", distinct from
"No profile for this configuration" (never calibrated) and "Tuned profile
found — N.NNx measured" (a real win).

Calibration is declined, in any case, when:

- the run is realtime (`"realtime inference is not tunable"`);
- every stage is served from cache (`cache_replay`);
- another accelerator job is active (`contention_detected`);
- the accelerator is thermally throttled;
- the baseline vector itself fails the planner's memory admission (this also
  blocks applying — the vector doesn't fit regardless of who chose it).

### Backward-pass propagation

Backward-enabled projects (i.e. `resources/configs/default.json`, which
enables backward tracking by default) are fully supported. Previously a tuned
forward pass wrote its detection cache under a key that includes the tuned
batch size, then the backward pass re-resolved at the project's *untuned*
batch size, missed that key, and aborted — measured on a real courtship
project. A second independent lookup on the backward pass would not fix this:
re-deriving a key from a `cache_replay` context could legitimately differ from
the forward pass's key and silently reintroduce the abort.

Instead, the forward pass persists its **effective** `InferenceTuningSettings`
to `applied_inference_vector.json` in the run's inference-cache directory
(`.inference_cache_<stem>/`), alongside the detection cache it wrote. The
backward pass reads that file and applies the vector **verbatim, with no
fingerprint lookup involved** — the cache key matches because it is
byte-for-byte the same vector that wrote the cache. If the sidecar file is
absent (e.g. a cache written before this feature existed), the backward pass
falls back to the project's configured values, which is exactly the
pre-existing behaviour for an untuned forward pass.

### The two-record density bridge

One calibration writes **two** profile records, so both a cache-less first run
and later cached runs hit a profile without a second tracking run:

1. With no detection cache yet, the search has no real per-frame object
   counts to key on, so it falls back to `bucket(MAX_TARGETS)` for all
   density buckets and marks the key `density_is_estimated=True`.
2. After a successful calibration, the search's own trials measured real
   detection counts. The calibrate path calls
   `store.observe_production_throughput(...)` immediately with that measured
   density, which — because the first record's key says it was an estimate —
   writes a **second** record under the measured key, while deliberately
   leaving the estimated-key record in place so the next brand-new video
   (which also starts with no cache) still gets a warm start from it.
3. A later run that does have a detection cache computes the measured key and
   hits record 2; a cache-less run hits record 1.

`density_is_estimated` therefore stays a load-bearing key component — it
distinguishes "re-key this, it was never measured" from "demote this, reality
changed" — and is not something a future change should drop from the
fingerprint.

## The calibration budget

`InferenceAutotunePolicy.budget_seconds` must be between **5 and 7200 seconds**
(`MINIMUM_/MAXIMUM_CALIBRATION_BUDGET_SECONDS` in `core/inference/config.py`,
re-exported from `autotune/models.py`; default **4500**). The ceiling is arithmetic. At a measured ~20 s per measurement
block and five blocks per candidate vector, one batch-size field costs ~410 s to
screen; with screened losers short-circuited a completed `fly_obb` search
measured 885 s (baseline 101 s, detection screen 424 s, depth screen 273 s,
final validation 80 s) with **zero** confirmations, because on MPS every
candidate is decisively slower and short-circuits. That is the cheap case, not
the representative one: a field pays for a confirmation whenever it has a
candidate that is not *confidently* slower, and on CUDA batch effects measure
~1.000 -- dead in the noise -- so every field confirms. The default is set from
that worst case: 101 + 4x424 + 273 + 5x300 + 80 = **~3650 s**, so the default is
**4500 s** with ~23 % headroom. A second pass runs only if a field is accepted
(gain >= 2 %), which a within-noise field never is, so pass 0 binds. The 7200 s
maximum covers the two-pass case in which fields genuinely win (~5650 s), which
no tolerable default could. Per-trial cost is clip- and model-dependent, so these figures
are orders of magnitude, not guarantees.

An early stop is loud: the search logs the elapsed time and the fields it did
and did not reach at WARNING, and the INCOMPLETE record carries
`searched_fields`/`unsearched_fields` in its `calibration_summary`. A rejected
budget value is also reported at WARNING with the requested and effective
numbers -- it is not clamped but replaced by the default, so a silent fallback
once turned an explicit 3000 s into 600 s.

It is a wall-clock deadline on the whole search: when it expires
the search returns the baseline with reason `budget_expired` and writes an
`INCOMPLETE` profile so the next run does not re-burn the budget (retry after
`INCOMPLETE_RETRY_SECONDS` = 24 h; never written when contention was detected).

Within that budget each measurement obeys `MeasurementProtocol` (`measure.py`):
at least 3 warmup calls over ≥ 8 frames, ≥ 5 measured blocks, and a stage is
complete once it has run ≥ 2.0 s **or** ≥ 128 frames.

A screened candidate whose paired bootstrap gain interval against the incumbent
lies entirely **below zero** is rejected immediately with reason
`screened_slower_than_incumbent: <candidate> fps vs incumbent <incumbent> fps`
and never pays for a full-pipeline confirmation: the confirmation gate exists to
admit winners, and a candidate that measurably lost the screen can only ever be
rejected by it. Every candidate that could still win is confirmed as before.

A candidate is only accepted if its median throughput beats the incumbent by at
least **2 %** *and* the lower bound of the paired bootstrap gain interval is
above zero; otherwise it is recorded as rejected with reason
`gain_or_confidence_gate`. Candidates are never silently dropped — every
rejection carries a reason string (`search.py`, `_rejection_reason`).

## The correctness gate

A speedup is worthless if it changes tracking output, so every candidate is
compared against the baseline's own output with `compare_outputs`
(`equivalence.py`). Both the forward-rich CSV and the final CSV are gated.

`EquivalencePolicy` defaults:

| Knob | Value | Meaning |
| --- | --- | --- |
| `position_p99_tolerance` | `0.5` px | 99th percentile of per-row position distance |
| `angle_max_tolerance` | `0.05` rad | **per-row maximum** angular difference, not a mean |
| `match_gate` | `2.0` | maximum distance for a positional (Hungarian) row match |
| `exact_categorical` | `True` | heuristic categorical columns must match exactly |

On top of the tolerances, these columns are **always** compared exactly,
independent of `exact_categorical` — no caller may switch them off:

```
_MANDATORY_EXACT_COLUMNS = ("TrackID", "TrajectoryID", "State", "ArenaID")
```

That is what catches a Hungarian identity swap, which keeps row counts, keys and
XY coordinates identical and is therefore invisible to a geometric tolerance.

Rows with NaN positions — ordinary lost/coasting tracks — cannot be matched by
nearest-XY, so they are aligned by `(FrameID, TrackID)` key instead. They feed
the exact-column comparison but contribute nothing to the position percentile or
the angular maximum, so the geometric populations are unchanged by their
presence.

The gate is deliberately **not** byte-exact: the design records measured
evidence that a larger batch changes an identity row on CUDA, so a byte-exact
gate would admit nothing.

### The determinism floor

Before any candidate is evaluated, the baseline's **reference block is run a
second time at the same frame window**, and that duplicate is compared against
the original with `for_determinism_floor=True` (which additionally excludes
known-bistable head/tail rows from the angular statistic). The same-window
requirement is not incidental: the five measurement blocks are *striped* across
the clip (`sidecar_child._block_window`) so the timing is representative, which
means block 0 and block 1 cover different frames and comparing them would
measure the clip, not the pipeline. For the same reason every candidate block is
compared against the baseline block carrying the **same block index**. That
comparison is the measured noise floor: the candidate gate's limits are then
`max(policy tolerance, floor)`, so a project whose own output is noisier than
the tolerance is not judged against an unattainable bar.

If the floor comparison itself fails, the search aborts immediately with

```
status=fallback  reason=baseline_nondeterministic_beyond_contract
```

and the configured settings are used unchanged. That abort is negative-cached
(an `INCOMPLETE` record), so a project that cannot reproduce itself does not
re-pay the baseline measurement on every run.

## Profiles

Validated results are stored under
`get_data_dir()/inference_tuning_profiles/`, keyed by a fingerprint covering the
model set, video geometry, device identity, and package source digest — so a
code edit or a hardware change invalidates the profile rather than reusing a
stale one. States are `PROVISIONAL`, `VALIDATED`, and `INCOMPLETE` (the negative
cache described above). Only a `VALIDATED` profile can affect a production run,
and only when the run has `apply_tuned_inference` (`APPLY_TUNED_INFERENCE`) on.
