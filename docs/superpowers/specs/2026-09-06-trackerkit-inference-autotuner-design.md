# TrackerKit system-specific inference throughput autotuner

**Date:** 2026-09-06
**Status:** Proposed; experimentally specified, not implemented
**Scope:** TrackerKit full-run local inference
**Evidence:** [Mehek measurement study](notes/2026-09-06-trackerkit-inference-autotuner-mehek-study.md)

## Decision

Add one coordinated, persistent inference autotuner around `InferenceRunner`.
It owns these throughput controls when automatic mode is enabled:

- detector frames per call (`detection_batch_size`);
- sliced-detector tiles per call (`slice_tile_batch_size`);
- pose crops per call (`pose_batch_size`);
- head-tail crops per call (`headtail_batch_size`);
- each identity classifier's crops per call; and
- `pipeline_depth`, because it changes both overlap and the safe frame-batch
  envelope.

The tuner will not choose each field independently and will not maximize batch
size. It will analytically prune unsafe candidates, screen stage-local
candidates on representative real work, perform bounded coordinate search in
the complete pipeline, reject output changes beyond the determinism floor, and
persist only a fully validated profile for the exact system/model/workload
signature.

The optimization objective is lexicographic:

1. preserve output semantics and resource safety;
2. maximize median steady-state full-run throughput; then
3. among candidates within 2% of the fastest, choose the one with lower peak
   memory, lower batch/depth, and lower preparation cost, in that order.

"Keep the current settings" is a successful result. Automatic means no user
must choose batch values; it does not promise that every model combination has
a faster setting.

## Why this architecture

Mehek invalidated four simpler designs:

- largest-fit failed because 1200 px detector batch 4 beat 8 and 16, 2632 px
  batch 1 beat all larger choices, and SAHI batch 1 beat batch 2;
- independent stage tuning failed because the composed depth-1 ant profiles
  regressed 4-13% end to end;
- model-forward-only timing failed because the existing SAHI microtuner was
  11.5% slower and used 9.1x the VRAM of the best manual result; and
- output invariance failed because SAHI batching changed detection counts and
  identity batching changed a categorical identity beyond a zero-noise
  same-setting determinism floor.

The only ant change that survived the joint gate was pose batch 4 plus
head-tail batch 4 at the existing depth 2, improving median throughput about
2.3%. The identity pipeline correctly retained its manual settings.

## Goals

- On the first eligible full run for an unknown signature, select safe,
  near-optimal settings without manual batch experimentation.
- Reuse the selection across future processes on the same system.
- Preserve existing inference semantics within the repository's equivalence
  contract, with stricter exact gates for categorical outputs.
- Never use OOM as ordinary search control flow in the main runner.
- Bound calibration time, memory, candidate count, disk use, and retries.
- Explain the effective values, evidence, cache status, and fallback reason.

## Non-goals

- A universal batch constant transferable across systems.
- Changing model accuracy thresholds, slice geometry, crop geometry, target
  count, tracking gates, or postprocessing behavior.
- Tuning realtime preview; realtime keeps its existing batch-1 constraints.
- Tuning CoreML OBB batching; CoreML remains fixed at batch 1.
- Consuming calibration predictions as production results in v1.
- Making CPU/MPS policy claims from the Mehek CUDA measurements. The
  architecture supports backend-specific policies, but every backend needs its
  own evidence and profile.

## Relationship to the tracking auto-tuner

TrackerKit already has a tracking auto-tuner under
`hydra_suite.core.tracking.optimization`. It recommends semantic detection,
assignment, and Kalman parameters using held-out trajectory-quality evidence,
then applies representable values through
`trackerkit.gui.autotune_contract`. That is a separate product and objective.

This design is named the **Inference Throughput Autotuner** in code, logs, and
UI. It changes execution strategy only and must not:

- add batch/depth fields to `TRACKING_AUTOTUNE_CANDIDATE_KEYS`;
- write its result through the tracking candidate UI contract;
- claim an accuracy or trajectory-quality improvement; or
- search semantic thresholds to recover throughput or equivalence.

The two systems may share read-only inference/result caches and common runtime
profiling primitives, but their keys, evidence, stores, recommendations, and UI
actions remain separate. If both are requested, semantic tracking parameters
are resolved first; their resulting detection/count workload signature then
keys throughput tuning.

## Architecture

```text
TrackerKit / CLI
       |
       v
InferenceRunner -- explicit/manual ownership check
       |
       v
AutotuneCoordinator (Qt-free, core/inference)
       |
       +--> WorkloadFingerprint --> persistent profile lookup
       |                               |
       |                         hit --+--> live down-admission
       |                               |
       +--> CandidatePlanner <---------+ miss/stale
       |        |
       |        +--> analytic + measured memory admission
       |        +--> bounded candidate sets
       |
       +--> contained calibration sidecar under heavy-job lease
       |        +--> warmup and randomized stage screening
       |        +--> full-pipeline coordinate trials
       |        +--> determinism/equivalence gate
       |
       +--> atomic validated-profile promotion
       |
       v
InferenceRunner with an immutable runtime overlay
```

### Component boundaries

The implementation belongs under a focused
`hydra_suite.core.inference.autotune` package, not in TrackerKit's
`MainWindow`. Suggested modules are:

- `fingerprint.py`: stable hardware, software, model, and workload identity;
- `candidates.py`: candidate generation and analytical admission;
- `measure.py`: warmup, synchronized timing, resource telemetry, and robust
  statistics;
- `equivalence.py`: calibration-output and determinism comparison;
- `search.py`: bounded coordinate/beam search;
- `store.py`: schema validation, locking, atomic promotion, and invalidation;
- `sidecar.py`: contained child protocol; and
- `coordinator.py`: policy and the one public resolve operation.

TrackerKit supplies typed configuration and displays progress/results. Core
owns the algorithm. Runtime supplies backend resolution, resource leases,
process supervision, resource budgets, and measured memory profiles. No lower
layer imports from TrackerKit.

Use the existing `MemoryProfileStore` and measured-envelope functions for
memory evidence. Do not put throughput winners into that store: performance
trials need distributions, equivalence verdicts, and a wider fingerprint, so
they require a distinct `InferenceTuningProfileStore`.

## Ownership and compatibility

Resolution precedence is field-specific:

1. an explicit CLI override or a field marked manual in the project config;
2. a validated compatible automatic profile;
3. a newly validated automatic result; or
4. the existing safe configured/default value.

The tuner produces an immutable runtime overlay. It does not rewrite project
JSON, advanced config, or the user's batch widgets. Logs and UI show requested,
admitted, and effective values separately.

Existing configs retain their explicit settings until the user enables
automatic inference tuning. New TrackerKit configs may default to automatic
mode after shadow rollout passes. A single "Auto-optimize full inference"
control is preferable to six independent auto checkboxes; advanced users may
mark individual fields manual without disabling the remaining search.

## Persistent identity

Cache reuse requires an exact `TuningProfileKey`. Paths and Python object IDs
are not identities. Model files use content digests, preferably supplied by the
model registry and otherwise computed once and memoized by size/mtime.

| Key group | Required fields |
|---|---|
| Schema | tuning schema, search-policy version, memory-estimator version |
| System | stable host ID, OS/architecture, CPU model/logical count, host-memory class |
| Accelerator | physical device UUID, model, compute capability, total VRAM |
| Software | Hydra version/commit, Python ABI, backend, precision, driver, CUDA, cuDNN, TensorRT, PyTorch, Ultralytics, and SLEAP runtime versions when active |
| Models | ordered stage roles and SHA-256 digests for direct/stage-1/stage-2 detector, pose, head-tail, and every identity classifier; model input/crop sizes; TensorRT profile identity |
| Frame/decode | width, height, channels/pixel format, resize factor, decoder/backend mode |
| Detector semantics | detection method/task, class selection, confidence/IoU, maximum detections, direct versus sequential mode |
| Slice semantics | enabled flag, resolved tile width/height, overlap, full-frame mode, ROI-gating mode, reference-body geometry |
| Pipeline | ordered enabled stage set, tracking direction/mode, result-cache stage mask |
| Workload | configured target count, observed detections/crops p50 and p95 bucket, canonical crop geometries |

Do not key by video pathname: future videos with the same semantic workload
should reuse the profile. On a hit, compare the first production windows'
density with the stored bucket. A bucket mismatch makes the profile
provisional and triggers a bounded revalidation rather than silent reuse.

Current free VRAM, utilization, temperature, and other-process load are dynamic
admission inputs, not cache-key fields.

## Stored profile

A validated record contains:

- the full key and a stable profile ID;
- baseline, requested, admitted, and selected settings;
- candidate medians, median absolute deviations, confidence intervals,
  measurement counts, warmup counts, and stage/full-pipeline timing;
- host RSS, accelerator peak, queue/frame-buffer high-water marks, and thermal
  range;
- calibration frame/density summary without video paths or pixels;
- determinism-floor and candidate-equivalence verdicts;
- TensorRT preparation artifact IDs and preparation seconds;
- selection reason, rejected candidates and bounded failure classes;
- creation/last-validation timestamps and observed production throughput; and
- state: `provisional` or `validated`.

Only `validated` records can alter a production run. A provisional record may
reduce the next search space but never overrides the baseline.

Store records under `get_data_dir()`, with a small versioned size/record cap.
Writes use a per-key interprocess lock, write/fsync to a private temporary file,
and atomic replace. Corruption, an unknown schema, or partial data degrades to
a cache miss. Concurrent runners use single-flight tuning; a waiter may use the
current baseline rather than block indefinitely.

## Candidate admission

Admission happens before loading candidate-specific models or allocating
candidate buffers.

1. Resolve the backend and fixed constraints. CoreML/realtime OBB yields only
   batch 1. A cached detection/result stage removes its unused knob from the
   search.
2. Apply existing hard limits and compute the live decoded-frame requirement
   from frame bytes, detection batch, and retained windows implied by pipeline
   depth. Reject a candidate that exceeds the existing frame-buffer budget.
3. Estimate tile/crop inputs, preprocessing outputs, raw predictions, queues,
   and backend workspaces with all enabled models resident.
4. Combine analytical floors with exact-key `MemoryProfileStore` observations;
   never extrapolate above the largest successful observed setting.
5. Admit against live free resources once, with the runtime's canonical safety
   fraction. A cache hit repeats this step and may choose a smaller previously
   validated equivalent setting.

Candidate sets are geometric and bounded, always including the current value:

- frame/tile batches: `1, 2, 4, ... admitted_max` plus current and admitted
  maximum;
- crop batches: the same set, capped by the observed p95 crops per frame and
  including that p95/count boundary; and
- depth: legal values 1-4 that remain admitted with the incumbent frame batch.

Values above the actual crop-count ceiling are canonicalized to the smallest
equivalent value. Requested and effective values remain distinct in telemetry.

## Measurement protocol

### Eligibility

Do not tune while another accelerator job is active, when free memory is below
the baseline requirement, or when the device is already thermally throttled.
Sample utilization/free memory several times rather than trusting one instant.
If eligibility fails, use a compatible down-admitted cached profile or the
baseline and report `deferred_due_to_contention`.

The default calibration budget is 120 seconds per unknown signature. Preparing
the baseline artifact required by the already-selected runtime is reported
separately and does not consume that budget; optional candidate-only artifact
work does. Budget expiry keeps the incumbent; it never promotes the best
incomplete observation. Higher-value fields are measured first, ranked by
incumbent profiler wall-time share. This lets an expensive low-share identity
sweep be skipped without blocking detector tuning.

### Representative work

Use normal stage APIs on real frames from the input, including preprocessing,
crop/tile extraction, merge/NMS, transfers, and output conversion. Do not time
only backend `predict`. Use bounded early/middle/late windows when seeking is
available. Do not retain an unbounded decoded-frame set; reread or spool a
bounded window and account for decoder timing separately.

The calibration sidecar loads the full active model combination so residency
and scheduling match production. Candidate predictions are disposable and
never enter production caches in v1.

### Warmup and timing

- Warm every backend/candidate for at least three backend calls and eight
  representative frames before measurement.
- Synchronize CUDA around measured spans.
- Measure at least five randomized/interleaved blocks. Continue until each
  candidate accumulates at least two seconds in its affected stage or reaches
  the 128-frame/120-second cap.
- Record stage-inclusive time and full-pipeline frames/second separately from
  initialization, engine preparation, decode, and final export.
- Use paired medians, MAD, and a bootstrap 95% confidence interval. Do not use
  a single fastest sample.

Grouped candidate order is forbidden because compilation, allocator growth,
temperature, and clocks drift over a run. Use a deterministic seeded block
order so a failure is reproducible.

## Search and selection

The search is a bounded coordinate beam, not a Cartesian product:

1. Run the current settings twice to establish a determinism floor and an
   incumbent full-pipeline distribution.
2. Rank tunable fields by their profiler share.
3. For one field at a time, hold every other incumbent value fixed and run the
   admitted stage-inclusive screen.
4. Keep at most the two fastest equivalent mutations plus the incumbent.
5. Replay those mutations through the full forward pipeline from identical
   initial tracking state. Reject any correctness failure.
6. Accept a mutation only when its median throughput gain is at least 2% and
   the paired 95% confidence interval excludes regression. Otherwise keep the
   incumbent.
7. Continue with the next field against the updated incumbent.
8. Make one second pass over accepted fields and their nearest rejected
   neighbors to catch interactions. Stop after two passes or budget expiry.
9. Re-run the final vector in a fresh sidecar. Promote it only after that run
   passes admission, equivalence, and the full-pipeline performance gate.

Within a 2% near-optimal confidence band, prefer lower peak memory, then lower
batch/depth. This selected 1200 px detector batch 4 over larger values, pose
batch 4 over 8, and avoids meaningless values above a per-frame crop ceiling.

Pipeline depth is treated like every other coordinate against the full current
incumbent. It is never selected on a detector-only microbenchmark and then
blindly composed with pose/identity settings; Mehek showed that loses useful
overlap.

## Correctness gate

The baseline duplicate defines the backend's determinism floor on the exact
calibration sample. A candidate may not exceed that floor plus these public
limits:

- every output CSV has a non-zero row count;
- detection/tracking row counts match and positional matching has zero
  unmatched rows;
- position p99 is at most the existing 0.5 px tolerance and angular mean at
  most 0.05 rad, unless the measured determinism floor is larger;
- pose keypoint presence/NaN patterns and categorical head-tail results match;
- identity class, unique identity key, class-label, and NaN patterns match
  exactly by default; and
- both forward-rich output and final tracking output pass.

The exact categorical default is intentional: Mehek changed one identity row
at batch 8 while same-setting repeats were exact. A future user-selectable
"numerically equivalent" policy may relax geometry, but must never be the
silent default for identity.

If a faster candidate fails correctness, record the rejection and continue.
The tuner must not modify thresholds to make it pass.

## Failure containment and live reuse

Every unknown candidate runs in a fresh supervised child under the canonical
heavy-job lease. Recognized accelerator OOM or host soft-limit exits may reduce
one pressure field and retry at most twice using the existing bounded-adaptation
machinery. A crash, timeout, NaN output, corrupt output, or correctness failure
is not retried as an OOM and is never promoted.

The production runner never explores an unsafe setting. After a cache hit it:

1. resolves current free memory and contention;
2. checks the stored peak plus safety reserve;
3. uses the stored selected value if admitted;
4. otherwise steps down only through previously successful/equivalent settings;
5. otherwise falls back to the baseline; and
6. records the down-admission reason without overwriting the validated winner.

If production throughput is more than 15% below the stored distribution for
three comparable windows, mark the record provisional and revalidate on a
later idle run; do not start a disruptive search in the middle of tracking.

## TensorRT artifact policy

Engine preparation is not a timing candidate and is never hidden inside a
trial. The coordinator reports separate `prepare_seconds` and
`steady_state_seconds`.

- Reuse compatible immutable batch-profile artifacts when present.
- Serialize builds by source digest, TensorRT/runtime fingerprint, input size,
  task, precision, and optimization profile.
- Build in a private temporary directory and atomically promote the final
  profile. Candidate builds must not share generic intermediate `.onnx` or
  `.engine` paths.
- On an unknown profile range, build at most one wide dynamic calibration
  engine spanning admitted candidates, benchmark candidates within it, then
  build/validate a dedicated selected profile only if the predicted benefit
  justifies the second build.
- The dedicated engine's final full-pipeline result is authoritative because
  TensorRT optimization-profile choice can change both performance and
  numerics.
- Never compile all candidate profiles speculatively. Mehek measured 255-310
  seconds for a single cold profile versus about four seconds on a hot hit.

Engine artifacts and tuning profiles have separate identities and lifecycles.
Removing an engine invalidates only profiles that require it; changing a
tuning policy does not delete otherwise valid engines.

## First-run and future-run behavior

On a cache miss, TrackerKit shows a bounded "Optimizing inference" phase with
the incumbent, current field, elapsed/budget time, and an option to continue
with current settings. Cancellation leaves model/result caches untouched and
does not promote a partial profile.

After validation, the first run uses the winner for its production pass. The
UI reports one-time calibration and artifact preparation separately so a
steady-state gain is not misrepresented as a faster first completion. Future
processes perform a profile lookup and live admission only.

Target cache-hit overhead is under 50 ms excluding model hashing on first
registry discovery. A hit never repeats calibration. The current SAHI tuner's
process-local dictionary does not meet this requirement and should be replaced
or delegated to the coordinator when automatic mode is active.

## Observability

Every run records a bounded, path-free summary containing:

- cache hit/miss/stale/deferred status and profile ID;
- fingerprint digest and invalidation reason;
- per-field requested, admitted, and effective values;
- baseline/selected throughput, confidence, peak memory, and preparation time;
- candidate rejection reasons, including equivalence details; and
- live down-admission or fallback reason.

Detailed calibration logs remain in the normal run log directory. Profile
records never contain video paths, pixels, animal identities, or model paths.

## Verification requirements for implementation

### Unit and property tests

- stable fingerprint construction and one-field invalidation for every key
  category;
- manual/automatic precedence and immutable runtime overlays;
- bounded candidate generation, count canonicalization, and frame-window
  admission;
- measured-envelope use without double-applying safety fractions;
- paired statistics, tie-breaking, sequential stopping, and deterministic
  randomized order;
- output gates for row count, geometry, NaN pattern, pose, head-tail, and
  categorical identity;
- corrupt/oversized/old-schema stores degrade to misses;
- atomic promotion, concurrent single-flight locking, cancellation, and stale
  lock recovery;
- recognized OOM reduction and the two-retry cap; and
- cache-hit down-admission without mutating the stored winner.

### Integration tests

- fake stages whose independent optima conflict, proving the full-pipeline
  gate wins;
- a faster candidate with changed categorical output is rejected;
- cache miss writes one validated profile and a second fresh process uses it
  without trials;
- model digest, frame geometry, slice geometry, runtime version, or density
  bucket changes cause miss/revalidation;
- concurrent runners create one TensorRT build and one tuning profile;
- cold TensorRT preparation and hot inference are reported separately; and
- no candidate outputs contaminate production result caches.

### Real equivalence/performance gates

Use `tools/equivalence/` with non-empty row-count assertions. Compare both the
forward-rich and final tracking CSVs. At minimum cover detection-only, sliced
detection, pose/head-tail, CNN identity, sequential OBB, and a result-cache hit.
Risky implementation changes require both MPS and CUDA verification unless the
implementation plan explicitly narrows platform scope.

Mehek's results are regression evidence, not permanent absolute thresholds:

- the 1200 px native detector search should find batch 4's clear >=10% gain
  over batch 1 without selecting 8/16;
- the 4512 px SAHI search must not prefer batch 2 over batch 1;
- the 4512 px frame-batch 16/32 candidates must be pruned before inference;
- the pose pipeline must reject the depth-1 composition and may accept the
  depth-2 pose/head-tail batch-4 vector only if the confidence gate passes; and
- the identity pipeline must retain the baseline when alternatives regress or
  change a categorical identity.

## Rollout

1. **Record-only:** generate profiles and decisions but never alter runtime
   settings. Compare predicted winners with offline equivalence and throughput.
2. **Opt-in automatic:** apply validated profiles only when explicitly enabled;
   retain all manual controls and a per-run bypass.
3. **Default for new CUDA configs:** only after cache concurrency, artifact
   locking, cancellation, and equivalence gates pass on multiple CUDA systems.
4. **Backend expansion:** validate and version separate candidate policies for
   MPS and CPU before enabling them by default.

Do not ship a partial version that persists independent stage winners without
the full-pipeline and correctness gates; Mehek directly demonstrated that such
a tuner can be slower and can change results.
