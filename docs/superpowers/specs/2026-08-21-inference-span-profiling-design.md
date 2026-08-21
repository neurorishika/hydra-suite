# Extreme-Detail Inference Profiling (Span Profiler)

**Status:** pending implementation plan
**Date:** 2026-08-21
**Branch:** `feat/inference-span-profiling`

## Problem

A 1.7–2.1x tracking regression took hours to root-cause because
`batched_detection` is a single opaque bucket in `core/tracking/profiler.py`'s
`PHASE_ORDER`. The legacy profiler broke out `yolo_obb_inference`,
`pose_inference`, `precompute_cnn_identity` and `headtail_*` separately; the
Pipeline/InferenceRunner architecture that replaced it does not. The actual
defect — the whole frame converted to a float32 CHW tensor once per crop
consumer per frame — was invisible to every existing instrument and was found
only with ad-hoc timers (see `project_pose_cnn_batched_detection_slowdown`).

There are currently three unrelated timing systems:

1. `TrackingProfiler` — coarse phases + per-frame categories, JSON export.
2. `HYDRA_RT_PROFILE` (`runner.py:64-87`) — env-gated, realtime path only,
   logs `ms/f` every 100 frames, undocumented.
3. Ad-hoc timers added and deleted per investigation.

## Goal

When Debug Mode is on, a tracking run emits a profile detailed enough to
localize a performance defect to a single function in one run. When it is off,
behavior and output are exactly as today.

**Non-goals:** GUI surface for span depth, cross-run diff tooling, flamegraph
export, per-detection spans, CUDA-event device timing (named as a possible
future slice in Risk 2).

## Constraints

- Tracking output byte-identical with Debug Mode **on and off**.
- Zero measurable cost when Debug Mode is off.
- `core/` must not import from any app layer; the gate arrives as a param.
- CLAUDE.md design principles: no god objects, no copy-pasted boilerplate.

## Gate

**`ENABLE_PROFILING` is reused unchanged.** It is already the Debug Mode gate:
`trackerkit/engine_params.py:1471` derives it from `debug_mode` when that key is
present and falls back to the stored `enable_profiling` otherwise. It is
Qt-free and already threaded to `core`. No new knob, no new plumbing.

Consequence to keep in view: **all nine equivalence fixtures set
`"enable_profiling": true`** and none set `debug_mode`, so the instrumented
path is hot in every equivalence run. That is free coverage, and it makes the
instrumentation's own overhead a gated property (see Verification).

## Architecture

### New module: `src/hydra_suite/core/profiling/spans.py`

Qt-free, imports nothing above Core.

```
SpanRecorder
  .span(name, units=None, gpu=False) -> _Span   # context manager
  .snapshot() -> dict                           # nested tree
  .armed()                                      # ctx manager: arm/disarm, try/finally
  .bind_thread()                                # ctx manager: re-arm inside a worker thread

module level:
  span(name, units=None, gpu=False)             # what call sites use
  current() -> SpanRecorder | None
```

**Off-path cost.** `span()` reads one `ContextVar`; when unarmed it returns a
shared frozen `_NullSpan` singleton whose `__enter__`/`__exit__` are empty
`__slots__` methods. No allocation, no dict touch, no generator frame. The call
sites are identical in both modes — no `if profiling:` branching in stages.

**Tree, not flat dict.** The recorder keeps an explicit stack. Each node records
`total_s` (inclusive), `self_s` (inclusive minus direct children), `n_calls`,
summed `units`, and the name of the thread it ran on. Children are keyed by name
*within their parent*, so `crop_extract` under `cnn` and under `headtail` stay
distinct without callers hand-prefixing strings. `self_s` is what localizes a
defect: high inclusive with near-zero self time exonerates a stage and indicts
its child.

**Aggregate-only storage.** Totals per distinct span *path*, never per call. No
sample lists, therefore no percentiles. Memory is O(distinct paths) — the span
map is static at ~45 names — regardless of a 100k-frame run.

**`units` answers the batch-size question.** Every span may carry a work-unit
count (frames in window, detections, crops). With `n_calls` and `units` both
recorded the report prints `ms/call` *and* `ms/unit`. At the default
`detection_batch_size=1` a pass shows `pose/backend_forward n=150, 1 unit/call`;
at 25 it shows `n=6, 25 units/call`. Per-call overhead falls out of the
comparison directly. This is a property of the schema, not a special-cased
metric.

### GPU-boundary sync

Spans declared `gpu=True` synchronize on exit
(`torch.cuda.synchronize()` / `torch.mps.synchronize()`), via one helper that
reads the device off `RuntimeContext`; a no-op on CPU. Only real device
boundaries carry the flag: backend forwards, `canonical_warp_batch`,
`materialize_tensors`. Host-side spans never sync.

**Dual timestamps.** A `gpu=True` span takes `perf_counter()` both before and
after the sync, emitting `total_s` (host cost — what production pays) and
`gpu_wait_s` (the drain). One run yields both the production-faithful number and
the device-truth number, and the sync's own magnitude is printed rather than
hidden.

`HYDRA_PROFILE_SYNC=0` disables the syncs for a perf-faithful pass. Runs must
not be compared across this setting.

### Threading

Three sites run spans off the arming thread. A `ContextVar` armed on one thread
is **not** visible in another, so each is bound explicitly or its spans vanish
silently — the report would confidently show OBB costing zero:

| Site | Binding |
|---|---|
| `Pipeline._run_double_buffer` producer (`pipeline-obb-producer`) | wrap the producer target in `recorder.bind_thread()` |
| async `CacheWriter` worker thread | wrap the worker loop |
| `crops.py:164 _get_warp_pool` ThreadPoolExecutor | pool `initializer=`, not per-task |

Recorder counter updates take a `threading.Lock` — uncontended in the common
case, at most a few threads.

Each node is stamped with its thread name. **Percentages are computed within a
thread, never across**, and the top-level denominator is the pass's own
wall-clock. At depth>=2 producer and consumer overlap, so summed span time
legitimately exceeds wall-clock; a subtree that is 43% of its own thread but 4%
of the pass renders visibly as both. This is the distortion that produced the
refuted SLEAP-batching premise (`project_sleap_roundtrip_audit`: pose measured
4.6% of wall, batching returned ~0 end-to-end gain) and it must not be
reintroduced by a tree that folds off-thread work into one percentage base.

### Span map

~45 spans across three trees. Parents supply the prefix.

```
inference/                              (armed for the whole runner pass)
  batch_pass/
    open_caches
    window/                    units=frames
      detect/                            <- producer thread at depth>=2
        run_obb/  model_execute[gpu]  extract_raw  materialize[gpu]
        run_bgsub
      filter
      headtail/  crop_extract  apply_fit  backend_forward[gpu]     units=dets
      cnn/       crop_extract  apply_fit  backend_forward[gpu]     units=dets
      pose/      crop_extract/ frame_to_chw  affine_loop  warp_batch[gpu]
                               foreign_mask
                 prep_loop  transport  backend_forward[gpu]        units=crops
      apriltag
      cache_write/ enqueue  flush                                  <- writer thread
      assemble_scatter
  realtime/       (same child names; replaces HYDRA_RT_PROFILE)
post/
  prepare  resolve/ merge_candidates  enrich_identity  apply_merges  renumber
  interpolate  tag_identity  rescale
interp_crops/
  setup  gap_detection  crop_extraction/ read  warp  pose_inference  cnn_inference
  finalize
```

The four spans under `pose/crop_extract` are exactly where the last defect
lived: `frame_to_chw` would have shown one call *per crop consumer per frame*
with a large `self_s`.

**Placement rule: spans wrap loops, never loop bodies.** A span is cheap but not
free; inside a per-detection body at 50 detections x 100k frames that is 5M
calls measuring what the aggregate already reports. Per-detection cost is
recovered from `units` (`ms/unit`). `affine_loop` and `pose/prep_loop` follow
this.

### TrackingProfiler integration

`TrackingProfiler.__init__` constructs a `SpanRecorder` when `enabled`, exposes
it as `.spans`, and gains `armed()` — the context manager the three consumers
wrap their work in (`core/tracking/worker.py` around the runner pass,
`core/post/merge.py`, `core/post/interpolated_crops.py`).

`PHASE_ORDER`, `CATEGORY_ORDER`, and the existing `phases` / `categories` JSON
sections are **untouched**. The tree lands in a new sibling key of
`<video>_logs/tracking_profile_{forward,backward}.json`:

```json
"spans": {"name": "inference", "total_s": 41.2, "self_s": 0.3, "n_calls": 1,
          "thread": "MainThread", "children": [...]}
```

`log_final_summary` gains a `SPAN TREE` block after the existing phase table:
indented by depth, each line `name  total_s  (% of parent)  n=calls  ms/call
ms/unit`, sorted by `total_s` descending within each level, with a `concurrent`
marker on off-thread subtrees and `gpu_wait` shown where non-zero. Existing
output is strictly appended to; anything parsing today's JSON keeps working.

**Debug-off behavior is unchanged in every respect**: `enabled=False` builds no
recorder, `arm()` is never called, `span()` returns the null singleton, and the
JSON is not written — same as today.

### Re-entrancy and lifecycle

- No public bare `arm()`. `armed()` is a context manager with `try/finally`, so
  a leaked arm is unrepresentable rather than merely discouraged.
- `ContextVar.set()` / `.reset()` with the token saved, so nesting is a proper
  stack: forward pass, backward pass and `MergeWorker` coexist in one process,
  the inner arm restoring the outer on exit instead of clobbering it.
- `_Span.__exit__` pops by object identity, not by name, and never swallows
  (returns `False`). An exception mid-span unwinds the stack correctly instead
  of nesting every later span under a phantom parent. The recorder warns on an
  unbalanced stack at disarm, surfacing the bug rather than reporting quietly
  wrong numbers.

### Retiring `HYDRA_RT_PROFILE`

`runner.py:64-87` and its four `_rt_prof_*` call sites are deleted. Naive
deletion would be a capability regression, not an alias removal: `core/inference`
is also driven by DetectKit and PoseKit, which have **no `TrackingProfiler`**,
and the env var was the only way to profile those paths.

Replacement: **`HYDRA_PROFILE=1` arms a process-level recorder** that dumps the
same tree at exit — same recorder, same renderer, no `TrackingProfiler`
required. Any `core/inference` caller becomes profilable; TrackerKit merely
supplies a nicer host. `HYDRA_RT_PROFILE` is kept as a one-line alias.

## Verification

### Correctness

Byte-identical tracking output, Debug on **and** off, via
`tools/equivalence/run_matrix.sh` — MPS on this box, CUDA on mehek. Baseline
`legacy/main` per the standard recipe. The matrix is run **before** the change
with the same baseline so the slice's effect is attributable rather than
conflated. Fixtures already carry `enable_profiling: true`, so the default
matrix exercises the instrumented path; one additional run with
`debug_mode: false` injected proves the off-path is also identical. Row counts
verified `> 1` before trusting any `EQUIVALENT`; conda active throughout.

Known baseline noise: bistable head/tail pi-flips on head/tail clips
(`project_migration_verification`).

### Overhead

**Primary measurement: current-src vs current-src, `enable_profiling` true vs
false.** Same tree, same models, one variable — this isolates span cost exactly
and is the number the **<= 2% target** is judged against. The legacy-vs-current
`PERF_TOLERANCE=1.25` ratio remains only as the secondary regression gate; it
cannot attribute overhead, since legacy differs by every change since the tag.

If a clip exceeds tolerance, attribute it to a specific `gpu=True` span and drop
that span's sync rather than loosen the gate. Any single red perf line is re-run
on a quiet box before being believed — the ~30% wall-clock swing under load
previously produced a bogus `1.65x SLOWER` verdict on an untouched path.

### Self-proving run

`ant_cnn_identity` at stock `detection_batch_size=1` versus the same clip at 25,
comparing `n_calls` and `ms/unit` on `pose/backend_forward`,
`cnn/backend_forward` and `headtail/backend_forward`. If the per-call overhead
of a 1-frame window against 25-frame windows is not readable from those two JSON
files alone, the profiler has not earned its keep and that is reported as a
failure rather than papered over.

### Unit tests

- nesting and `self_s` arithmetic
- `units` aggregation; `ms/unit` derivation
- disarmed `span()` allocates nothing
- **thread propagation through `bind_thread`** — a test that fails if OBB spans
  go missing at depth>=2. This is the single most likely silent bug in the
  design.
- exception mid-span leaves a balanced stack
- nested `armed()` restores the outer recorder

## Implementation rules

1. **Take the timers from `instrumentation.patch`, take none of the
   memoization.** The shim also introduces `_CHW_MEMO` / `reset_chw_memo()` /
   `HYDRA_CHW_MEMO` and wires `reset_chw_memo()` into `_process_obb_results` —
   a *functional* change riding in the same diff. Porting it wholesale would
   smuggle a caching change into a pure-observation commit: the one thing that
   could break byte-identity, and the hardest kind to attribute because the diff
   reads as "just profiling." The real fix for that cost already shipped in
   `4db4e93a` / `542ce736`.
2. Every hunk in the final diff against `main` is a `with span(...)` wrapper, an
   import, or the new module. No logic edits. Verified by reading the diff
   before the gate is run.
3. Spans wrap loops, never loop bodies.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Concurrent spans double-count; an overlapped stage looks expensive but returns nothing when optimized | Thread stamping; percentages within a thread; pass wall-clock as top-level denominator; `concurrent` marker |
| 2 | `gpu=True` syncs slow the profiled run and overstate a stage's marginal cost | Dual timestamps (`total_s` + `gpu_wait_s`); flag on very few spans; `HYDRA_PROFILE_SYNC=0`. CUDA events named as a future slice |
| 3 | Ambient global state: leaked arm, nested arm, stack corruption, growth | `armed()` ctx manager only; ContextVar token stack; identity-based pop that never swallows; aggregate-only storage |
| 4 | Retiring `HYDRA_RT_PROFILE` blinds DetectKit/PoseKit | `HYDRA_PROFILE=1` process-level recorder; `HYDRA_RT_PROFILE` kept as alias |
| 5 | Perf gate cannot attribute span overhead | Profiling-on-vs-off on current src as the primary measurement |
| 6 | Porting the shim's memoization along with its timers | Implementation rules 1 and 2 |
| 7 | Span cost inside hot loops | Implementation rule 3 |
