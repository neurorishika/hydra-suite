# Inference Throughput Autotuner

The inference autotuner measures a project's own inference pipeline on the
project's own video and, when it can prove the result is output-equivalent,
selects faster batch sizes and pipeline depth than the configured defaults.

It is implemented in `src/hydra_suite/core/inference/autotune/` and is driven
from `core/tracking/worker.py` on every offline tracking run.

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

## The three modes

`InferenceAutotunePolicy.mode` (`core/inference/config.py`) accepts exactly
`off`, `record`, or `automatic`. Anything else is rejected at construction; an
unrecognised value arriving through `engine_params` falls back to `off`.

- **`off`** (default) — no autotune code runs. The configured settings are used.
- **`record`** — calibration runs and a profile is persisted, but the tuned
  vector is **never applied**. `effective == baseline` on every run, including a
  run that hits an already-validated cached profile: `integration.py` forces
  `allow_cached_reuse = False` for record mode, and the coordinator
  short-circuits a validated cache hit to a baseline overlay with
  `status="recorded"`. This is rollout stage 1 — it lets you see what the tuner
  *would* choose without changing any output.
- **`automatic`** — calibration runs, the profile is persisted, and a
  **validated** profile is applied to the production run.

## The CUDA-only policy for `automatic`

`automatic` is refused on any non-CUDA accelerator
(`integration.py`, `AcceleratorKind.CUDA` check): the request comes back with
`eligible=False`, `allow_cached_reuse=False`, and

```
eligibility_reason = "automatic inference tuning is validated only for CUDA"
```

On Apple Silicon (MPS) and CPU, `automatic` therefore behaves as a no-op —
the overlay reports the baseline. `record` mode is **not** subject to this gate
and does calibrate on MPS.

Calibration is also declined, in any mode, when:

- the run is realtime (`"realtime inference is not tunable"`);
- every stage is served from cache (`cache_replay`);
- another accelerator job is active (`contention_detected`);
- the accelerator is thermally throttled;
- the baseline vector itself fails the planner's memory admission.

## The calibration budget

`InferenceAutotunePolicy.budget_seconds` must be between **5 and 600 seconds**
(default 600). It is a wall-clock deadline on the whole search: when it expires
the search returns the baseline with reason `budget_expired` and writes an
`INCOMPLETE` profile so the next run does not re-burn the budget (retry after
`INCOMPLETE_RETRY_SECONDS` = 24 h; never written when contention was detected).

Within that budget each measurement obeys `MeasurementProtocol` (`measure.py`):
at least 3 warmup calls over ≥ 8 frames, ≥ 5 measured blocks, and a stage is
complete once it has run ≥ 2.0 s **or** ≥ 128 frames.

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

Before any candidate is evaluated, the baseline is measured **twice** and the
two outputs are compared with `for_determinism_floor=True` (which additionally
excludes known-bistable head/tail rows from the angular statistic). That
comparison is the measured noise floor: the candidate gate's limits are then
`max(policy tolerance, floor)`, so a project whose own output is noisier than
the tolerance is not judged against an unattainable bar.

If the floor comparison itself fails, the search aborts immediately with

```
status=fallback  reason=baseline_nondeterministic_beyond_contract
```

and the configured settings are used unchanged.

## Profiles

Validated results are stored under
`get_data_dir()/inference_tuning_profiles/`, keyed by a fingerprint covering the
model set, video geometry, device identity, and package source digest — so a
code edit or a hardware change invalidates the profile rather than reusing a
stale one. States are `PROVISIONAL`, `VALIDATED`, and `INCOMPLETE` (the negative
cache described above). Only a `VALIDATED` profile can affect a production run,
and only in `automatic` mode.
