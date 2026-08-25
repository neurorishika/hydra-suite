# Extreme-Detail Inference Profiling (Span Profiler)

**Status:** Shipped — merged to main (bb665956)
**Date:** 2026-08-21
**Branch:** `feat/inference-span-profiling`
**Revision:** 3 — plan-stage adversarial review corrected the self-proving
fixture (revision 2 had it inverted) and the Debug-OFF gate mechanism

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
2. `HYDRA_RT_PROFILE` (`runner.py:64-87`, seven `_rt_prof_*` call sites at
   `:777, :833, :840, :841, :914, :972, :1053`) — env-gated, realtime path only,
   logs `ms/f` every 100 frames, undocumented.
3. Ad-hoc timers added and deleted per investigation.

## Goal

When Debug Mode is on, a tracking run emits a profile detailed enough to
localize a **constant-cost** performance defect to a single function in one run.
When it is off, behavior and output are exactly as today.

**Explicitly out of reach** (stated because the Goal above would otherwise
overclaim): *diffuse self-time* defects — a cross-cutting tax executed inside
many small operations, such as the grad-mode toggling measured at 26.5s over 20k
calls in `project_sleap_roundtrip_audit`. That cost smears as slightly-elevated
`self_s` across dozens of spans and never aggregates into one line. No span
layout can catch it; `cProfile` is the right instrument and the docs say so.

**Other non-goals:** GUI surface for span depth, cross-run diff tooling,
flamegraph export, per-detection spans.

## Constraints

- Tracking output byte-identical with Debug Mode **on and off**.
- No measurable cost when Debug Mode is off (quantified in Verification).
- `core/` must not import from any app layer; the gate arrives as a param.
- CLAUDE.md design principles: no god objects, no copy-pasted boilerplate.

## Gate

**`ENABLE_PROFILING` is reused unchanged.** It is already the Debug Mode gate:
`trackerkit/engine_params.py:1471-1475` derives it from `debug_mode` when that
key is present (`_debug_present`/`_debug_mode` at `:856-857`) and falls back to
the stored `enable_profiling` otherwise. It is Qt-free and already threaded to
`core` (`session.py:243,:319`; `worker.py:698`). No new knob.

All nine fixture configs in `tools/equivalence/fixtures/configs/` set
`"enable_profiling": true` and none set `debug_mode`; the default matrix runs
**eight** of them (`ant_cnn_identity_marked` is excluded from the `VIDEOS` array
at `run_matrix.sh:56-75`). So the instrumented path is hot in the default gate.
That is free coverage, and it makes the instrumentation's own overhead a gated
property — which is why the default profiled path must not synchronize (see
GPU timing).

## Architecture

### New module: `src/hydra_suite/utils/profiling.py`

**Placement.** The timing primitive is a leaf with no dependencies, consumed by
Core (`inference`, `tracking`, `post`), Data (`dataset_generation.py`, which
already imports Core at `:332`), Integrations, and potentially every kit.
`utils/` is the bottom layer in CLAUDE.md's dependency table and is importable
by all of them; putting the primitive there avoids a second profiling home
*inside* `core/`. `TrackingProfiler` stays at `core/tracking/profiler.py` and
remains the lifecycle owner and renderer. One primitive, one host.

```
SpanRecorder
  .span(name, units=None, gpu=False) -> _Span   # context manager
  .snapshot() -> dict                           # nested tree
  .armed()                                      # ctx manager: arm/disarm, try/finally
  .bind_thread()                                # ctx manager: per-thread stack rooted here

module level:
  span(name, units=None, gpu=False)             # what call sites use
  current() -> SpanRecorder | None
```

**Off-path cost.** `span()` reads one `ContextVar`; when unarmed it returns a
shared frozen `_NullSpan` singleton with empty `__slots__` methods. Measured on
the deployment interpreter (3.13.12): `ContextVar.get()` 17.3 ns, `with
_NullSpan:` 117 ns, full `with span('x'):` **152 ns** against a 7.7 ns
unwrapped baseline. At ~30 span-enters per frame that is ~4.6 µs/frame — 0.02%
of a 25 ms frame. The claim is "not measurable at frame scale", not literally
zero, and it holds **only while implementation rule 3 (no spans in loop bodies)
is enforced**.

**Tree, not flat dict.** Each node records `total_s` (inclusive), `self_s`
(inclusive minus direct children), `n_calls`, summed `units`, `max_s`,
`first_call_s`, and its thread name. Children are keyed by name *within their
parent*, so `crop_extract` under `cnn` and under `headtail` stay distinct
without callers hand-prefixing strings. `self_s` localizes a defect: high
inclusive with near-zero self time exonerates a stage and indicts its child.

**`max_s` and `first_call_s` are not optional.** The adjacent `TrackingProfiler`
reports p50/p95/p99 from per-frame samples (`profiler.py:328, 466-468,
579-594`); a mean-only tree would be a regression beside it. A 5 s TensorRT
engine build or MPS shader compile inside `backend_forward` (n=500) inflates the
mean by 10 ms/call and is **indistinguishable from a uniform 10 ms slowdown** —
it would indict the wrong function. Two floats per node catch it at O(paths).
Full percentiles are still declined: they cost O(calls).

**Aggregate-only otherwise.** Totals per distinct span *path*, no sample lists.
Memory is O(distinct paths) — the map is static at ~60 names, enforced by the
registry below — regardless of a 100k-frame run.

**`units` answers the batch-size question.** Every span may carry a work-unit
count (frames in window, detections, crops). With `n_calls` and `units` both
recorded the report prints `ms/call` *and* `ms/unit`. At the default
`detection_batch_size=1` (`config.py:439`) each window is one frame; at 25 the
same total work arrives in 1/25th the calls. Per-call overhead falls out of the
comparison directly. `units` at a parent is supplied independently, never summed
from children (`window/` counts frames, `pose/` counts crops — one level apart,
different quantities).

**Span-name registry.** Names are module-level constants in
`utils/profiling_names.py`, not bare string literals duplicated across 10+
modules — that would be the copy-pasted boilerplate CLAUDE.md forbids, in string
form, and a refactor that moved a function would silently drop its row with
nothing failing. Function-boundary spans use a `@spanned(NAME, units=...)`
decorator; `with span(...)` is reserved for sub-function regions
(`frame_to_chw`, `affine_loop`). Dynamic/label-keyed span names are prohibited —
they would make memory O(labels).

### GPU timing: opt-in deep-GPU mode

**The default profiled path does not synchronize.** Spans are host wall-clock.

`torch.cuda.synchronize()` and `torch.mps.synchronize()` are **device-wide**,
not stream- or thread-scoped, and `pipeline_depth` defaults to 2
(`config.py:440`), so `_run_double_buffer` is the normal path. A sync taken on
the consumer thread therefore drains the *producer's* in-flight OBB kernels:
`cnn/backend_forward` would bill OBB's device time to CNN — cross-stage
misattribution, the exact failure this design exists to prevent. The codebase
already made this call deliberately: `runtime.py:75-105` uses per-tensor
`torch.cuda.Event` handoffs via `_HANDOFF_EVENTS` specifically to avoid a global
drain.

Measured cost on `hydra-mps` (torch 2.11.0): `torch.mps.synchronize()` is 0.1 µs
against an idle queue but **318 µs with one pending op**, versus 8.5 µs for the
same op unsynced — a ~37x penalty. At ~6 `gpu=True` spans per frame at
`detection_batch_size=1` that is ~1.8 ms/frame, which alone breaks the ≤2%
target on the fast clips and would ride in every future equivalence perf ratio
on every branch, since the fixtures profile hot.

**`HYDRA_PROFILE_GPU=1` opts into a deep-GPU diagnostic pass** which:

1. enables `synchronize()` on `gpu=True` span exit, and
2. **forces `pipeline_depth=1`**, so there is no producer thread to contaminate
   and the serialization is deliberate and known.

The run is explicitly not the production schedule; that is the price of device
attribution, paid only when asked for. The JSON stamps
`"gpu_mode": "off" | "deep"` so no one compares trees across the two.

An honest statement of what each mode yields, replacing the "dual timestamps
give you both" claim from revision 1 (which was wrong — a sync's effect is not
local: once span N drains the device, span N+1 starts from an empty queue and
its `total_s` describes a schedule production never runs):

| Mode | `total_s` means | GPU attribution |
|---|---|---|
| default | host cost under the production schedule | device work smears into whichever span later blocks |
| deep (`HYDRA_PROFILE_GPU=1`) | host cost under a serialized depth=1 schedule | per-span device time, uncontaminated |

Per-span CUDA events — true device time without serializing — are recorded as a
**named future slice**, not silently omitted. They are CUDA-only (no MPS
equivalent) and would let the deep pass keep depth=2.

### Threading

Spans taken on a thread that never armed the recorder read the `ContextVar`'s
default and **vanish silently** — the report would confidently show that work
costing zero. (Verified semantics: each thread gets its own top-level context;
`threading.Thread` targets and `ThreadPoolExecutor` workers do **not** inherit
the spawning thread's context, unlike asyncio tasks.) Every off-thread site must
therefore be bound explicitly:

| Site | Binding |
|---|---|
| `pipeline.py:502,541-543` `pipeline-obb-producer` | wrap the producer target |
| `cache/writer.py:63` async CacheWriter worker | wrap the worker loop |
| `crops.py:164` `_get_warp_pool` ThreadPoolExecutor | pool `initializer=` |
| `interpolated_crops.py:925-928` Sequential/SparseFramePrefetcher (`utils/frame_prefetcher.py:88-98`) | bind inside the prefetcher's decode thread |
| `media_export.py:605` writer thread | wrap the writer loop |
| `worker.py:448-452` realtime FramePrefetcher | bind inside the decode thread |

The last three were missed in revision 1. The prefetcher one is the most
dangerous: `interp_crops/crop_extraction/read` sits directly over it, so an
unbound version would report near-zero frame-read cost in interpolation —
against ~12s of video seek measured in `project_sleap_roundtrip_audit`. A span
wrapped around `.read()` instead of the decode measures **queue-wait, not
decode**, which is a different lie; the span goes inside the prefetcher thread.

**Stacks are thread-local; only the aggregated node table is shared.** The stack
lives in the `ContextVar` value, rooted at each `bind_thread()`; a single shared
stack would interleave pushes from concurrent threads and corrupt parentage, and
the lock protects counters, not stack coherence. Node-table updates take a
`threading.Lock` (39 ns uncontended; producer + CacheWriter + warp pool can
contend, so the contended case is measured in Verification, not assumed).

Each node is stamped with its thread. **Percentages are computed within a
thread, never across**; the top-level denominator is the pass's own wall-clock.
At depth≥2 summed span time legitimately exceeds wall-clock, and a subtree that
is 43% of its thread but 4% of the pass renders visibly as both. This is the
distortion behind the refuted SLEAP-batching premise (pose measured 4.6% of
wall, batching returned ~0 end-to-end gain) and must not be reintroduced.

### Span map

~60 spans across four trees. Parents supply the prefix.

```
session/                        <- armed at SessionRunner.run (session.py:530+)
  track_forward  track_backward
  postprocess/ pose_quality  temporal_pose  trajectory_postproc
  merge
  rich_export/ build_dataframe  relink  write
  interpolated_crops -> interp_crops tree
  dataset_generation/ seek  read  crop  write
  media_export  annotated_video

inference/                      (armed for the runner pass)
  batch_pass/
    open_caches
    window/                     units=frames
      decode                             <- producer thread at depth>=2
      detect/
        run_obb/  model_execute[gpu]  extract_raw
        run_bgsub_batch
      materialize[gpu]                    <- window/ level, sibling of detect/, not nested under run_obb/
      filter
      headtail/  crop_extract/ affine_loop  warp_batch[gpu]/ frame_to_chw
                 apply_fit  backend_forward[gpu]              units=dets
      cnn/       crop_extract/ affine_loop  warp_batch[gpu]/ frame_to_chw
                 apply_fit  backend_forward[gpu]              units=dets
      pose/      crop_extract/ affine_loop  warp_batch[gpu]/ frame_to_chw
                               foreign_mask
                 prep_loop  transport  backend_forward[gpu]   units=crops
      apriltag
      cache_write/ enqueue  flush                             <- writer thread
      assemble_scatter
  realtime/     obb/  crops/  individual/  cache/   (mirrors runner.py:760-1273
                structure, NOT the batch child names)

post/
  prepare  resolve/ merge_candidates  enrich_identity  apply_merges  renumber
  interpolate  tag_identity  rescale

interp_crops/
  setup  gap_detection  crop_extraction/ read  warp  pose_inference  cnn_inference
  finalize
```

**Head-tail and CNN get the same `crop_extract` children as pose.** Revision 1
decomposed only pose and claimed that was "exactly where the last defect lived".
It was not: of the 34.4s defect, **24.0s was in the head-tail + CNN consumers**
(n=300 via `extract_classifier_crops`) and 10.4s in pose (n=150). Both paths
route through the same seam — `crops.py:103-104` and `crops.py:218-219` both
call `canonical_warp_batch_from_frame(frame, m_aligns, geometry, lambda sub:
_frame_to_chw_float(sub, device))`. The smoking gun is that conversion cost is
O(frame area) and independent of detection count while the warp scales with
`units`; that signature is only visible when the two are separate spans. Under
revision 1's map, the *larger share* of the defect stayed opaque.

**The `session/` tree closes the second acid test.** Revision 1 covered only the
pipeline it had just finished debugging. `SessionRunner` (`session.py:541-584`)
runs `postprocess`, `rich_export`, `interpolated_crops`, `dataset_generation`,
`media_export` and `annotated_video` as stages that no armed consumer wrapped —
~28% of wall in pandas code plus the dataset-generation seeks, all invisible.
Arming at `SessionRunner.run` and adding the `session/` tree is what stops this
design from reproducing, at session scope, the "one opaque bucket" failure the
Problem section opens with.

**Placement rule: spans wrap loops, never loop bodies.** Inside a per-detection
body at 50 detections x 100k frames that is 5M calls measuring what the
aggregate already reports. Per-detection cost comes from `units` (`ms/unit`).

### TrackingProfiler integration

`TrackingProfiler.__init__` constructs a `SpanRecorder` when `enabled`, exposes
it as `.spans`, and gains `armed()` — the context manager the consumers wrap
their work in (`SessionRunner.run`, `worker.py` around the runner pass,
`post/merge.py`, `post/interpolated_crops.py`).

`PHASE_ORDER`, `CATEGORY_ORDER`, and the existing `phases` / `categories` JSON
sections are **untouched**. The tree lands in a new sibling key of
`<video>_logs/tracking_profile_{forward,backward}.json`. Verified safe: the only
consumers of that file are `tests/test_tracking_profiler.py` (no strict key-set
assertions) and the path-only tests in
`tests/core/tracking/test_profile_output_path.py`; nothing in `tools/` or
`scripts/` parses it.

```json
"gpu_mode": "off",
"spans": {"name": "session", "total_s": 41.2, "self_s": 0.3, "n_calls": 1,
          "max_s": 41.2, "first_call_s": 41.2, "thread": "MainThread",
          "children": [...]}
```

`log_final_summary` gains a `SPAN TREE` block after the existing phase table:
indented by depth, each line `name  total_s  (% of parent)  n=calls  ms/call
ms/unit  max`, sorted by `total_s` descending within each level, with a
`concurrent` marker on off-thread subtrees.

**Debug-off behavior is unchanged in every respect**: `enabled=False` builds no
recorder, `arm()` is never called, `span()` returns the null singleton, and the
JSON is not written.

### Re-entrancy and lifecycle

- No public bare `arm()`. `armed()` is a context manager with `try/finally`, so
  a leaked arm is unrepresentable.
- `ContextVar.set()` / `.reset()` with the token saved: nesting is a proper
  stack, so `SessionRunner`, forward/backward passes and `MergeWorker` coexist
  in one process, the inner arm restoring the outer on exit.
- `_Span.__exit__` pops by object identity, never by name, and never swallows
  (returns `False`). An exception mid-span unwinds correctly instead of nesting
  every later span under a phantom parent. The recorder warns on an unbalanced
  stack at disarm.

### `HYDRA_PROFILE=1` and the User-mode path

`runner.py:64-87` and its seven `_rt_prof_*` call sites are deleted. Naive
deletion would be a capability regression: `core/inference` is also driven by
DetectKit and PoseKit, which have no `TrackingProfiler`, and the env var was the
only way to profile those paths.

`HYDRA_PROFILE=1` arms a **process-level recorder** dumping the same tree at
exit — same recorder, same renderer, no `TrackingProfiler` required.
`HYDRA_RT_PROFILE` is kept as a one-line alias. Specified precisely, because
revision 1 left all three of these open:

- **Precedence:** a `TrackingProfiler` recorder wins while armed (its `armed()`
  nests over the process recorder via the token stack); the process recorder
  resumes outside. Spans go to exactly one recorder, never both.
- **Dump location:** `<video>_logs/` when a session supplies one, else
  `$HYDRA_DATA_DIR/profiles/span_profile_<pid>.json` plus a logged tree.
- **User mode:** this is the supported route to profile **without changing what
  the run does**. Debug Mode is not observation-only — it changes intermediate
  cleanup and CSV outputs (`session.py:614-621`) — so "turn on Debug and re-run"
  profiles a different run than the one that was slow. `HYDRA_PROFILE=1` is
  documented as the answer for a User-mode user hitting a performance problem.

## Verification

### Correctness

Byte-identical tracking output, Debug on **and** off, via
`tools/equivalence/run_matrix.sh` — MPS on this box, CUDA on mehek. Baseline
`legacy/main`. The matrix is run **before** the change with the same baseline so
the slice's effect is attributable. Fixtures already profile hot; one additional
run with **`enable_profiling: false`** injected proves the off-path is
identical. It must **not** be done by setting `debug_mode: false`: that
derives `DEBUG_MODE=False`, which triggers the User-mode cleanup at
`session.py:619-637` and **deletes `_forward.csv` and `_tracking_final.csv`** —
the two files the acceptance criterion compares. The code says so itself
("NO-OP in debug mode (and thus a no-op for the equivalence gate)"). Clearing
`enable_profiling` while leaving `debug_mode` absent keeps DEBUG_MODE at its
`True` default, so the debug CSVs are still written, and disarms every span —
which isolates exactly the variable under test. Row
counts verified `> 1`; conda active throughout. Known baseline noise: bistable
head/tail pi-flips on head/tail clips.

### Overhead

**Primary measurement: current-src vs current-src, `enable_profiling` true vs
false** — same tree, same models, one variable.

Revision 1 specified a single on/off pair, which cannot resolve the ≤2% target:
this box has a **measured ~30% wall-clock swing under load** that once produced
a bogus `1.65x SLOWER` verdict. The protocol is therefore **N=5 alternating
runs per condition on an otherwise-quiet box, reporting median and IQR**, with
the pass criterion being that the on/off median delta is ≤2% *and* smaller than
the within-condition IQR. If the noise floor exceeds the effect, the result is
reported as "below noise floor", not as a pass.

Host-side armed cost is already bounded by measurement: ~594 ns/span (2x
`perf_counter` @31.3 ns, `Lock` @39 ns, dict updates, stack push/pop) — at ~30
spans/frame, ~18 µs/frame. Contended lock cost is measured during this pass
rather than assumed. The legacy-vs-current `PERF_TOLERANCE=1.25` ratio remains
only as the secondary regression gate; it cannot attribute overhead.

`HYDRA_PROFILE_GPU=1` is **not** exercised by the gate — it is a diagnostic mode
that deliberately changes the schedule.

### Self-proving run

Revision 2 moved this run to `ant_obb_sleap` on the claim that
`ant_cnn_identity` "has no pose model". **That was backwards and is corrected
here.** Verified against the fixture configs:

| fixture | `enable_pose_extractor` | `pose_model_dir` | `cnn_classifiers` | headtail |
|---|---|---|---|---|
| `ant_obb_sleap` | `false` | — | `[]` | yes |
| `ant_cnn_identity` | `true` | SLEAP unet | 1 model | yes |

`is_pose_inference_enabled` (`session_policy.py:29`) requires both
`enable_pose_extractor` and a non-empty `pose_model_dir`, so **`ant_obb_sleap`
runs neither pose nor CNN** — two of the three `backend_forward` nodes the
criterion compares do not exist there. `ant_cnn_identity` is the only fixture
carrying all three consumers. Revision 1's clip was right; revision 2 broke it.

`ant_cnn_identity` at stock `detection_batch_size=1` versus the same
clip at 25 via `tools/equivalence/runner.py --detection-batch-size`
(`:143,156-157,240`, flowing to `window_size` at `pipeline.py:178-189`),
comparing `n_calls` and `ms/unit` on `pose/backend_forward`,
`cnn/backend_forward` and `headtail/backend_forward`. The batch-25 run is a
**profiling experiment, not a byte-identity gate**: changing the window size
changes decode and crop batching, so a tracking diff there is expected and is
not evidence of a profiler bug.

If the per-call overhead of a 1-frame window against 25-frame windows is not
readable from those two JSON files alone, the profiler has not earned its keep
and that is reported as a failure.

### Unit tests

- nesting and `self_s` arithmetic; `max_s` / `first_call_s` correctness
- `units` aggregation; `ms/unit` derivation; parent `units` not summed from
  children
- disarmed `span()` allocates nothing
- **thread propagation through `bind_thread`** — fails if OBB spans go missing
  at depth≥2. The single most likely silent bug in the design.
- **two-thread concurrent nesting** produces correctly-parented, uncorrupted
  trees (guards the thread-local-stack requirement, which is the part two
  implementers would otherwise build differently)
- percentages never computed across threads
- exception mid-span leaves a balanced stack
- nested `armed()` restores the outer recorder; `HYDRA_PROFILE` precedence
- **golden span-path set**: a tiny synthetic pipeline run armed, asserting the
  snapshot's span-path set against a checked-in expected set. This is the only
  test that catches "a span silently disappeared in a refactor" — the failure
  mode the whole feature exists to prevent — and revision 1 had no test of the
  instrumentation at all, only of the recorder.
- `HYDRA_PROFILE=1` exercised for DetectKit and PoseKit, so Risk 4's mitigation
  does not ship untested.

## Implementation rules

1. **Take the timers from `instrumentation.patch`, take none of the
   memoization.** The shim also introduces `_CHW_MEMO` / `reset_chw_memo()` /
   `HYDRA_CHW_MEMO` (patch lines 238-253) and wires `reset_chw_memo()` into
   `Pipeline._process_obb_results` (patch lines 82-84, 190) — a *functional*
   change riding in the same diff. Porting it wholesale would smuggle a caching
   change into a pure-observation commit: the one thing that could break
   byte-identity, and the hardest kind to attribute because the diff reads as
   "just profiling". The real fix for that cost already shipped in `4db4e93a` /
   `542ce736`.
2. Every hunk in the final diff against `main` is a `with span(...)` / `@spanned`
   wrapper, an import, or the new module. No logic edits. Verified by reading
   the diff before the gate is run. Sanctioned exception: a fix round on
   `pipeline.py`'s `_stream_windows` was a genuine, disclosed,
   output-equivalent restructure (needed to place the `DECODE` span
   correctly), not a wrapper insertion.
3. Spans wrap loops, never loop bodies. Checked during diff review — the
   "no measurable cost when off" claim depends on it. Sanctioned per-frame
   exception: `READ` (`frame_prefetcher.py`, all three prefetcher classes) and
   `FLUSH` (`cache/writer.py`) deliberately open one span per frame inside a
   loop body — wrapping from outside the loop would measure queue-wait
   instead of decode/flush cost.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Concurrent spans double-count; an overlapped stage looks expensive but returns nothing when optimized | Thread stamping; percentages within a thread; pass wall-clock as top-level denominator; `concurrent` marker |
| 2 | Device-wide sync drains the producer's queue and misattributes across stages | No sync on the default path; `HYDRA_PROFILE_GPU=1` syncs **and** forces depth=1; `gpu_mode` stamped in JSON; CUDA events as a named future slice |
| 3 | Ambient global state: leaked arm, nested arm, stack corruption, growth | `armed()` ctx manager only; ContextVar token stack; thread-local stacks; identity-based pop that never swallows; aggregate-only storage |
| 4 | Retiring `HYDRA_RT_PROFILE` blinds DetectKit/PoseKit | `HYDRA_PROFILE=1` process-level recorder with specified precedence + dump location; alias kept; both kits tested |
| 5 | Perf gate cannot attribute span overhead; 30% noise floor swamps a 2% effect | Profiling-on-vs-off on current src, N=5 alternating runs, median + IQR, "below noise floor" is a valid outcome |
| 6 | Porting the shim's memoization along with its timers | Implementation rules 1 and 2 |
| 7 | Span cost inside hot loops | Implementation rule 3, checked in diff review |
| 8 | Span names rot silently after a refactor | Name registry + golden span-path test |
| 9 | Warmup/tail costs folded into a mean | `max_s` + `first_call_s` per node |
| 10 | Diffuse self-time defects (grad-mode toggling) are undetectable | Stated as out of reach in the Goal; `cProfile` documented as the escape |

## Review Corrections (revision 1 → 2)

Three adversarial reviews ran against revision 1. Substantive corrections:

- **GPU sync reversed.** Revision 1 synced on `gpu=True` spans and claimed dual
  timestamps yielded both a production-faithful and a device-truth number. Both
  wrong: sync is device-wide and drains the producer at the default
  `pipeline_depth=2`, and a sync's effect contaminates every downstream span.
- **Acid test 1 was failed.** Only pose's `crop_extract` was decomposed; the
  head-tail + CNN share was the larger part of the defect (24.0s vs 10.4s).
- **Acid test 2 exposed a session-scope void** — ~28% of wall in `SessionRunner`
  stages that no armed consumer wrapped. Added the `session/` tree.
- **Three off-thread sites were missed**, including the prefetcher directly
  under `interp_crops/crop_extraction/read`.
- **`run_bgsub` → `run_bgsub_batch`** in the batch tree (`pipeline.py:227`).
- **Self-proving fixture** was moved to `ant_obb_sleap` on an inverted reading
  of the configs; **revision 3 moves it back to `ant_cnn_identity`**, the only
  fixture with pose + CNN + head-tail all enabled.
- **Overhead protocol** strengthened from one pair to N=5 median/IQR.
- **Added:** `max_s`/`first_call_s`, name registry + golden test, thread-local
  stacks made explicit, `HYDRA_PROFILE` precedence/dump/User-mode specified,
  diffuse-self-time limitation stated, module moved to `utils/`.

One reviewer claim was **rejected**: that the module must live in `utils/`
because Data has no Core imports. `data/dataset_generation.py:332` already does
`from ..core.inference.runner import InferenceRunner`, so that argument is
false. The module is in `utils/` on the independent grounds above (leaf
primitive, avoids two profiling homes inside `core/`).

## Gate results (Task 14, MPS)

Run on `hydra-mps`, `conda activate hydra-mps`, all 8 default equivalence
clips, baseline `legacy/main` (`157e1ae3`), current branch `378ff2d9`.

### 1. Profiling-ON equivalence (`enable_profiling: true`, the fixture default)

Every clip's `DETERMINISM (new_a vs new_b)` — this branch run twice — is
`EQUIVALENT ✅` with zero unmatched rows and zero position/theta deltas on
every clip's forward/final/final_with_individual CSV. The branch's own output
is fully reproducible.

The `EQUIVALENCE (legacy vs new_a)` comparison shows 7 of 8 clips as
"UNTRUSTWORTHY" per `run_matrix.sh`'s own guard. Diagnosed directly from the
log, not assumed: legacy CSVs carry `IdentityAssigned*` columns, current-branch
CSVs carry `IdentityRealtime*` columns — an already-merged, separately-verified
Identity overhaul (see memory `project_identity_overhaul_phase1_done`) renamed
these columns on `main` **after** the `legacy/main` tag was cut, so the tag is
stale relative to *any* current branch, not specific to span profiling. The
resulting positional-match theta deltas (mean up to ~0.8 rad, max exactly
3.142 ≈ π) are the already-documented "bistable head/tail π-flip" baseline
noise this spec's Verification section names. Neither is a span-profiler
regression: both patterns are identical between the profiling-ON and
profiling-OFF runs (below), which they could not be if either originated from
this branch's change.

### 2. Profiling-OFF equivalence (`enable_profiling: false` injected in place, `debug_mode` left absent)

Same 8-clip matrix, same baseline, only `enable_profiling` cleared. Shows the
identical vs-legacy pattern (same 7 clips, same reason) and the identical
clean `DETERMINISM` verdicts. No new divergence introduced by turning
profiling off.

### 3. The actual gate: Debug-ON vs Debug-OFF byte-identical

Since `run_matrix.sh` only ever compares against the (stale) legacy baseline,
the binding constraint — *this branch's own output does not change based on
`enable_profiling`* — was checked directly: every tracking CSV from the
profiling-ON run's `new_a` tree was diffed byte-for-byte (`cmp -s`) against
the same file from the profiling-OFF run's `new_a` tree, across all 8 clips
(`_tracking_forward.csv`, `_tracking_backward.csv`, `_tracking_final.csv`,
`_tracking_final_with_individual.csv`).

**Result: 32/32 files byte-identical. Zero differences, zero missing files.**
This is the constraint the whole feature depends on, confirmed directly
rather than inferred from the noisy vs-legacy comparison.

### 4. Overhead (`fly_obb`, N=5 alternating, current-src vs current-src)

| condition | run times (s) | median | IQR |
|---|---|---|---|
| profiling ON  | 21.855, 21.947, 22.161, 22.360, 24.764 | 22.161s | 1.661s |
| profiling OFF | 21.824, 22.159, 22.190, 24.601, 24.672 | 22.190s | 2.645s |

Median delta = 0.029s (**0.13%**) — well under the ≤2% target and well
under either condition's own IQR (1.66s / 2.65s). **Pass**, clearly inside
the noise floor rather than merely under a threshold.

### 5. Self-proving run (`ant_cnn_identity`, `detection_batch_size` 1 vs 25)

Non-vacuousness confirmed first: `headtail/backend_forward`, `cnn/backend_forward`,
`pose/backend_forward` all have `n_calls > 0` in both runs (this is the
fixture the spec's Review Corrections section fixed after revision 2 named
the wrong clip — `ant_cnn_identity` is the only fixture with pose + CNN +
head-tail all enabled, confirmed again here by these three nodes actually
existing with real call counts).

| span | batch=1: n_calls / ms\/call / ms\/unit | batch=25: n_calls / ms\/call / ms\/unit |
|---|---|---|
| `headtail/backend_forward` | 500 / 61.09ms / 3.91ms | 20 / 1814.39ms / 4.64ms |
| `cnn/backend_forward`      | 500 / 30.91ms / 1.98ms | 20 / 706.86ms / 1.81ms |
| `pose/backend_forward`     | 500 / 42.54ms / 2.72ms | 20 / 984.21ms / 2.52ms |

(`units` = total detections across the run, 7819, identical in both — same
video, same detections, only the window size differs.)

`n_calls` scales with window count (500 windows at batch=1 → 20 windows at
batch=25, matching 500/25); `ms/call` scales up 23-30x with it (fixed
per-call overhead amortized over more work per call). `ms/unit` — the actual
per-detection cost — is far more stable than `ms/call`: cnn and pose drift
7-9% (1.98→1.81, 2.72→2.52), while headtail drifts 19% (3.91→4.64), a real
divergence worth a closer look in a follow-up but small next to `ms/call`'s
23-30x swing. That contrast is the readable signal: the profiler shows
batching amortizes
*fixed per-call overhead*, not per-detection compute, which is exactly the
`detection_batch_size=1` defect class this feature was built to make visible.
**Pass** — the per-call overhead is directly readable from these two JSON
files alone, per the spec's stated bar for this check.

### Verdict

All four checks pass. The CUDA gate (Task 15, mehek) remains outstanding.

## Gate results (Task 15, CUDA — mehek)

Run on `mehek.taild08eb9.ts.net`, `conda activate hydra-cuda`, branch
`feat/inference-span-profiling` @ `29c7f7b9` transferred via `git bundle`
(no network access to this box's git remotes from the branch worktree), same
8-clip fixture set, baseline `legacy/main` @ `157e1ae3` (an existing
`.worktrees/equiv-legacy` on the shared box, already at the correct SHA —
reused rather than recreated; several other in-progress worktrees on this
box belong to other work and were left untouched).

### 1. Equivalence matrix (`RUNTIME=cuda`)

Every clip's `DETERMINISM (new_a vs new_b)` verdict is `EQUIVALENT ✅` — 24/24
checks (positional + final + final_with_individual per clip), zero unmatched,
zero deltas. Identical shape to the MPS run's own-branch reproducibility.

The `EQUIVALENCE (legacy vs new_a)` comparison shows the same
"UNTRUSTWORTHY" pattern as MPS — 7 of 8 clips, same reason (`IdentityAssigned*`
→ `IdentityRealtime*` column rename predating the `legacy/main` tag, per the
diagnosis in the MPS Gate results section above), 17/17 `EQUIVALENCE`
sub-verdicts reading `DIFFERENCES` for that same, pre-existing, cross-platform
reason. That the exact same clips fail for the exact same diagnosed reason on
a second, independent device confirms the cause is the stale baseline tag,
not anything platform- or profiler-specific.

All CSV row counts verified `> 1` (minimum observed: 1048 rows) before
trusting any verdict.

### 2. Deep-GPU mode crash check (`HYDRA_PROFILE_GPU=1`, `fly_obb`, CUDA)

Not a byte-identity gate — deliberately runs a different (`pipeline_depth=1`,
synchronized) schedule. Ran to completion, exit code 0, produced all four
tracking CSVs. Exported `tracking_profile_forward.json` has `"gpu_mode":
"deep"`, confirming the CUDA sync path (`torch.cuda.synchronize()`) executes
without error — the MPS gate could only exercise the MPS branch of
`_synchronize()`; this is the first real exercise of the CUDA branch.

(Gotcha hit and resolved: invoking `runner.py` directly, outside
`run_matrix.sh`, requires `PYTHONPATH=<repo>/src` set explicitly — an
unset `PYTHONPATH` silently imports whatever `hydra_suite` is on the
environment's default path rather than this branch's `src/`, and without it
the run fails outright with `ModuleNotFoundError` rather than silently using
the wrong tree. Documented here per the project's existing PYTHONPATH gotcha
memory.)

### Verdict

Both gates pass. MPS and CUDA together: the plan's binding constraint
(Debug-ON vs Debug-OFF byte-identical) is confirmed on MPS by direct CSV
diff; DETERMINISM is clean `EQUIVALENT` on both MPS and CUDA; the vs-legacy
noise is diagnosed, pre-existing, and now confirmed cross-platform (same
clips, same reason, both devices) rather than device- or profiler-specific;
the CUDA-only deep-GPU sync path runs without error. Span profiler
verification is complete.
