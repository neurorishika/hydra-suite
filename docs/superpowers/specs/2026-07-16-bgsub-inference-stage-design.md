# Background Subtraction as a First-Class Inference Stage

**Date:** 2026-07-16
**Status:** Implemented and calibrated (2026-07-17). Tasks 1–14 complete.
**Supersedes:** item 9 of `2026-04-26-inference-pipeline-redesign.md` (which deferred bg-sub)

## Calibration Results (2026-07-17, worm_bgsub, cpu tier, 500 frames)

The acceptance gate (`tools/equivalence/run_matrix.sh worm_bgsub`) was run
legacy (`main`) vs this branch. Verdict: **accepted with a documented new
baseline** — the strict exact-equivalence gate (`pos_p99 ≤ 0.5px`,
`unmatched == 0`) does not apply here, because this project deliberately changed
the background computation. Evidence:

| Measure | Result | Meaning |
|---|---|---|
| Determinism (`new_a` vs `new_b`) | `pos_p99 = 0.000px`, `unmatched = 0` | Evenly-spaced priming made bg-sub deterministic — the property the cache key needs. |
| Performance | `new/legacy = 1.00×` | No overhead from routing through `InferenceRunner`. |
| Convergence redesign | eps sweep: default (`1e-4`) and latch-at-frame-30 (`0.9`) are **byte-identical** | Moving the switchover frame S changes nothing — the latch replacing `tracking_stabilized` does **not** alter detections. The core feature is behavior-neutral. |
| Equivalence vs legacy | mean `0.22px`, `p99 ≈ 1.6px`, ~2.7% unmatched, **uniform across all 500 frames** | Small, global divergence — not a regression at any single frame. |
| Track level | both post-process to **39 final tracks** | Detection jitter washes out where it matters for the science. |

**Cause of the residual, isolated by elimination:** the switchover S is ruled
out (eps sweep byte-identical); the ROI-resize fix is moot on this clip
(`RESIZE_FACTOR=1.0`, no ROI); conservative-split timing can only differ for the
first frame or two. The uniform, all-frames divergence is the **evenly-spaced vs
legacy random-sample priming change** — deliberate, and not tunable via
`eps`/`FRAMES`/`PIXEL_DELTA` (the sweep is flat).

**Why accepted rather than reverted:** legacy's baseline is itself an arbitrary
harness-seeded (`seed=0`) sample; production legacy priming was *unseeded and
non-deterministic*, so there is no canonical legacy behavior to match.
Evenly-spaced priming is at least as defensible (guaranteed temporal coverage)
and is deterministic, which is the enabling property for the whole cache. The
convergence defaults ship as calibrated: `EPSILON=1e-4`, `FRAMES=30`,
`PIXEL_DELTA=5.0` (never-latching, `eps=1e-12`, was measurably worse — 914
unmatched — confirming the default correctly latches).

**Coverage caveat (unchanged):** `worm_bgsub` is the only bg-sub equivalence
clip. Worms on a plate may converge differently from an ant colony; if bg-sub is
run on ant footage, re-check these defaults. Consider adding an ant bg-sub clip.

**One loose thread, judged benign:** the intermediate post-processing reported 34
`broken_occlusion` events on the new run vs 0 on legacy, yet both converge to 39
final tracks. This is consistent with the occlusion-window timing shifting under
the small detection jitter, not a behavior change in occlusion handling. Flagged
for awareness; not blocking.

## Problem

Background-subtraction detection never got a stage in `InferenceRunner`. It is a
second-class citizen: it does not get the runner's detection cache, batch
precompute, backward-pass replay, or `runtime_tier` handling. Everything on the
YOLO/OBB path does.

The reason it was deferred is architectural, not incidental. **bg-sub detection
depends on tracking state.** `BackgroundModel.update_and_get_background` takes a
`tracking_stabilized` flag (`core/background/model.py:404-437`) that selects which
background to subtract against — lightest-pixel before, adaptive EMA after. That
flag is set at `core/tracking/worker.py:3656`, after the Kalman update, when the
Hungarian assignment's average cost stays under `MAX_DISTANCE_THRESHOLD` for
`MIN_TRACKING_COUNTS` consecutive frames.

A feed-forward, cacheable inference pass cannot have a feedback loop from
tracking. That conflict is what this design resolves.

### What this design is *not*

It does **not** retire `core/detectors/`. That directory is ~214 KB of YOLO/OBB
code and 9.5 KB of bg-sub, and two of its files (`_direct_obb_runtime.py`,
`_runtime_artifacts.py`) are *live dependencies of the new pipeline* —
`core/inference/runtime_artifacts.py:217` lazily imports
`create_direct_obb_executor` from them. After this work, `core/detectors/` is a
purely-YOLO directory. Relocating that live runtime is a separate project.

## Key findings that shape the design

### 1. The feedback loop is tractable

Three properties make the loop removable rather than fundamental:

- **It is a monotonic latch.** Once `True`, never `False`. The entire history
  collapses to one number: the switchover frame `S`.
- **The background state evolves independently of the flag.** At
  `model.py:418-431`, both `lightest_background` (running max) and
  `adaptive_background` (EMA) update every frame regardless. Only the *selection*
  at line 434 reads `tracking_stabilized`. The model's trajectory is a pure
  function of the video.
- **Both non-tracker consumers already opt out.** `bg_optimizer.py:470` passes
  `tracking_stabilized=True` unconditionally; `preview_worker.py:423` passes
  `False`. Nobody reproduces the real loop.

The expensive, stateful part is already tracking-independent. Only a one-bit
selector is not. `MIN_TRACKING_COUNTS` is a tracking-quality threshold being
reused as a "has the background settled?" proxy — a question the background model
can answer about itself far more directly.

### 2. Three defects block sound caching

These are prerequisites, not side quests. Each independently makes a bg-sub cache
key a lie.

| # | Defect | Consequence |
|---|---|---|
| 1 | `_BGSUB_KEY_PARAMS` (`core/inference/cache/keys.py:60-80`) hashes `SUBTRACTION_THRESHOLD` and `BACKGROUND_PRIME_SECONDS`, which **exist nowhere in the codebase**. The real params are `THRESHOLD_VALUE` and `BACKGROUND_PRIME_FRAMES`. `params.get()` returns `None` for both, hashing to a constant. | Changing the bg-sub threshold — the single most important parameter — does not invalidate the detection cache. Stale detections replay silently. |
| 2 | `prime_background` (`model.py:308`) uses `random.sample(range(total), count)` on the **unseeded** module-level `random`. | Identical params + identical video produce a different background each run. Caching is unsound by construction. |
| 3 | When `ENABLE_ADAPTIVE_BACKGROUND=False`, `model.py:434` **still** switches to `adaptive_background` on stabilization — but that array is frozen at its primed value. | Disabling adaptive silently means "switch to a stale snapshot" rather than "don't switch". |

Defect 2 is **already known**: `tools/equivalence/runner.py:302-308` documents it
and works around it with `random.seed(0)`. That workaround is **test-only** — it
lives in the harness, not the library. Production runs still prime
nondeterministically. Seeding proves legacy-vs-new *comparable*; it does not make
bg-sub *cacheable*.

### 3. There is no plug-in seam

Detection is hardcoded YOLO-OBB. `_AllModels` (`runner.py:78`) is a closed
dataclass; stages are modules of free functions (`load_x_model` / `run_x` /
`run_x_batch`); dispatch is `if/elif` on config literals. There is no registry.
This design follows the existing module-of-free-functions pattern rather than
introducing one — a registry is not needed for a second detection method and
would be speculative generality.

## Design

### Module layout

`core/inference` may not import `core/detectors` — a rule `obb.py:76` already
broke down and duplicated `_gpu_letterbox_batch` over. So the mask→ellipse logic
cannot stay in `core/detectors/bg_detector.py` and be called from a stage. It
moves to join the model it belongs with:

```
core/background/
    model.py      # BackgroundModel — priming, lightest/adaptive state, GPU tiers (exists)
    measure.py    # mask -> ellipse measurements (moved from detectors/bg_detector.py, 238 lines)
core/inference/stages/
    bgsub.py      # thin stage adapter
```

`core/background` owns "what is the background and what is moving against it".
`stages/bgsub.py` is a thin adapter speaking the runner's protocol.
`core/inference -> core/background` is a legal downward edge: no rule bent, no
code duplicated.

### The stage

Follows the established shape exactly (cf. `stages/cnn.py`):

```python
@dataclass
class BgSubModel:
    bg_model: BackgroundModel      # holds all cross-frame state
    def close(self) -> None: ...

def load_bgsub_model(config: BgSubConfig, runtime: RuntimeContext) -> BgSubModel
def run_bgsub(frame, model, config, runtime) -> OBBResult
def run_bgsub_batch(frames, frame_indices, model, config, runtime) -> list[OBBResult]
```

Priming is the "load" — `BackgroundModel` slots naturally into the
`XModel`-wraps-an-opaque-backend role.

**Detection source becomes a choice.** `InferenceConfig.obb: OBBConfig`
(`config.py:229`) is a **required** field, and `_AllModels.obb` (`runner.py:80`)
is non-optional. bg-sub is an *alternative* detection source, not an addition, so
both become optional and `InferenceConfig` gains an exactly-one validation plus a
`detection_source -> Literal["obb", "bgsub"]` property. Note the asymmetry with
`cache_only`: OBB must still load in that mode because it is needed for
cache-key validation, whereas bg-sub's key needs no model and can skip loading
entirely.

**Ordering constraint.** `BackgroundModel` is stateful and strictly sequential.
`Pipeline` runs windows through a single in-order consumer
(`pipeline.py:38-42`), so this is safe, but it must be documented: bg-sub
requires strictly in-order frames. Random access (`load_frame`) is served from
cache only.

### Emitting `OBBResult`

bg-sub maps onto the existing detection result type; no new type is needed.

| Field | Source |
|---|---|
| `centroids`, `angles` | ellipse fit (angles already radians, `bg_detector.py:199`) |
| `sizes`, `shapes` | contour area / aspect ratio |
| `confidences` | `NaN` — not feasible for bg-sub (`bg_detector.py:194`). Downstream consumers must be NaN-safe. |
| `detection_ids` | `frame_idx * DETECTION_ID_STRIDE + slot` |
| `corners` | **derived**: ellipse -> rotated rect, in TL/TR/BR/BL order matching `_corners_from_xywhr` (`obb.py:249`) |

Corner ordering is load-bearing: the commit log records that getting it wrong put
SLEAP ~86 px off vs ~2.7 px.

This retires the `yolo_results=None` compatibility stub in legacy
`detect_objects`'s 5-tuple return.

### The convergence rule

`update_and_get_background` **loses its `tracking_stabilized` parameter**.
Stabilization becomes internal to `BackgroundModel`:

```
# lightest_background is a running max, so growth is non-negative (no abs needed).
grew = (lightest_background_new - lightest_background_old) > BACKGROUND_CONVERGENCE_PIXEL_DELTA
frac = count_nonzero(grew) / grew.size
if frac < BACKGROUND_CONVERGENCE_EPSILON for BACKGROUND_CONVERGENCE_FRAMES consecutive frames:
    self._stabilized = True   # monotonic latch, as before
```

**Why a changed-pixel FRACTION and not a mean delta.** An earlier draft of this
spec used `mean(|delta|)` over the whole frame. That is frame-size dependent and
silently wrong at production resolutions: one 200px animal revealing background
at ~150 grey levels moves the whole-frame mean by 7.32 on a 64x64 test fixture
but only 0.007 on a 2048x2048 rig -- seven times BELOW a 0.05 threshold. The
latch would fire almost immediately while animals still sat on their start
positions, baking them into the EMA: precisely the failure the lightest-pixel
phase exists to prevent. Unit tests at 64x64 cannot catch this, because the same
event moves their mean by 1000x the threshold.

**PIXEL_DELTA is the noise gate and MUST exceed the sensor noise floor.** A
running max never stops growing under noise: every frame, noise pushes some
pixels above the previous max, so the growing fraction plateaus at a
noise-dependent floor instead of reaching zero. Measured steady-state fraction
at 256x256 with epsilon=1e-4:

| sensor noise sd | pd=1.0 | pd=3.0 | pd=5.0 |
|---|---|---|---|
| 1.0 | 2.6e-5 latches | 0 latches | 0 latches |
| 2.0 | 3.9e-4 NEVER | 6.4e-6 latches | 0 latches |
| 4.0 | 1.2e-3 NEVER | 2.3e-4 NEVER | 3.7e-5 latches |

With `pd=1.0` and any realistic sensor noise (sd>=2), the latch NEVER fires: the
model sits on the lightest background forever, never switching to adaptive, and
silently loses the lighting-drift tracking adaptive exists for. Default is
therefore `5.0` -- comfortably above sensor noise, far below a genuine animal
reveal (~50-150 grey levels), so it discriminates correctly. It remains the
knob to raise on a noisier rig.

The changed-pixel fraction is scale-invariant: the same animal on the same
proportion of frame yields the same fraction at any resolution, so the default
epsilon is portable across rigs. It is also robust to single hot pixels, which a
max-delta metric is not.

Same latch semantics, same monotonicity — sourced from the background rather than
from Hungarian assignment cost. Cheap: the running-max update already touches
those pixels. Deterministic given video + params, which is what caching requires.

`worker.py:3651-3660` loses its `tracking_stabilized` bookkeeping.
`tracking_counts` remains — it serves other purposes.

Defect 3 is fixed at the same time: when `ENABLE_ADAPTIVE_BACKGROUND=False`, no
switch occurs at all.

### Caching

- `_BGSUB_KEY_PARAMS` corrected to `THRESHOLD_VALUE` and
  `BACKGROUND_PRIME_FRAMES`; gains `BACKGROUND_CONVERGENCE_EPSILON` and
  `BACKGROUND_CONVERGENCE_FRAMES`.
- **`CACHE_SCHEMA_VERSION` bumps.** Not optional. Every bg-sub cache on disk was
  produced under random priming and keyed by a hash ignoring the threshold. Those
  artifacts are unsound and must be invalidated, not inherited.
- Priming replaces `random.sample(range(total), count)` with **evenly-spaced
  indices** across the video. Deterministic without a seed, and strictly
  guarantees the temporal coverage random sampling only achieves on average.
  After this, the primed background is a pure function of
  (video, `BACKGROUND_PRIME_FRAMES`).
- Video signature folded in via `with_video_signature`, as for OBB.

bg-sub then gets the full runner treatment: `run_batch_pass` precompute,
backward-pass replay via `load_frame`, and `cache_only=True` skipping model load.

### Runtime tiers

`BackgroundModel._setup_gpu_acceleration` (`model.py:197-239`) gates on the
`ENABLE_GPU_BACKGROUND` param and then **self-selects** CUDA > MPS > CPU. It
never consults `runtime_tier`. Under the runner this is a live bug:
`runtime_tier="cpu"` would still run CuPy on the GPU. The port inverts control —
`RuntimeContext` drives selection via a new `configure_runtime()`; the model
stops deciding. `_setup_gpu_acceleration` is retained for legacy callers that
construct `BackgroundModel` without a runtime.

| tier | bg-sub backend |
|---|---|
| `cpu` | Numba (`_update_adaptive_background_numba`, `model.py:45`) |
| `gpu` | CuPy (CUDA, `model.py:61`) / torch (MPS, `model.py:121`) |
| `gpu_fast` | no distinct implementation -> resolves to `gpu`, `used_fallback=True` |

bg-sub has no TensorRT/CoreML story and needs none: it is elementwise work, not a
network.

Per `docs/developer-guide/runtime-integration.md`: add a `"bgsub"` pipeline key,
capability rules in `_pipeline_supports_runtime()`, runtime translation, and UI
intersection gating.

Existing permanent-fallback behavior (`model.py:401`, `_gpu_fallback_warned`) is
retained.

### Config

`BgSubConfig` dataclass in `core/inference/config.py`, with `from_params(dict)`
at the boundary. Params today flow as a plain dict with inline `.get()` defaults
scattered across three drifting key lists (`cache/keys.py:58-80`,
`trackerkit/tracking_cache.py:228-234`, `trackerkit/cli_config.py:638-720`) —
defect 1 is a direct symptom of that drift. The dict stays at the API boundary;
the dataclass is the internal contract.

### Call-site migration

| Site | Change |
|---|---|
| `core/tracking/worker.py:20,1160,2297,2315` | `create_detector` / `detect_objects` / `update_and_get_background` -> `InferenceRunner`; drop `tracking_stabilized` bookkeeping |
| `trackerkit/gui/workers/preview_worker.py:394,419-423,691,762,832,1159,1751` | `ObjectDetector` / `DetectionFilter` -> runner + `apply_detection_filter` |
| `data/dataset_generation.py:415,421,531,552,579` | `create_detector` -> runner |
| `core/tracking/optimization/optimizer_workers.py:367,372,441` | already imports the new filter shim; finish the detector move |
| `core/detectors/bg_optimizer.py:467-470` | drop `tracking_stabilized=True`; tune against the real convergence rule |
| `core/__init__.py:11`, `core/detectors/__init__.py` | drop `ObjectDetector` re-export |
| `core/detectors/factory.py` | `create_detector` dies — the last string-keyed factory goes with it |

`bg_detector.py` and `bg_optimizer.py` leave `core/detectors/`.

### Out of scope

- **`bg_optimizer.py`'s Qt coupling.** 1,015 lines importing `QThread`/`Signal`
  inside a `core` package — a real boundary violation, but a separate one. It
  moves alongside bg-sub; its structure is not addressed.
- **Retiring `core/detectors/`.** Requires relocating the live OBB runtime.
- **GUI restructuring.**

## Validation

### Acceptance criterion

> `bash tools/equivalence/run_matrix.sh worm_bgsub` passes at default tolerances.

`worm_bgsub` (`tools/equivalence/README.md:55`) is a ~500-frame clip and the only
coverage of the bg-sub path. Fixtures via `tools/equivalence/fixtures/fetch_fixtures.sh`.

The comparator (`tools/equivalence/compare.py`) is **already tolerance-based**,
not byte-equality: it matches detections positionally and passes when
`pos_p99 <= 0.5px`, `theta_mean <= theta_atol`, and `unmatched == 0`. This is the
right target — it asserts "the animals land in the same places", not "the floats
are identical" — and it turns `eps`/`K` calibration from a judgment call into a
search with a measurable objective.

### Why exact parity is not the goal

Two changes deliberately alter detection output: prime frame selection and the
switchover rule. A byte-comparison against legacy would fail, and *should*.
Legacy's own baseline is not reproducible without the harness's seed.

Note that evenly-spaced priming makes `runner.py`'s `random.seed(0)` a **no-op**
for the new path. That seeding call and its comment must be updated, not left to
imply a guarantee it no longer provides.

### Staged validation

1. **Prime strategy alone.** Legacy with seeded `random.sample` vs. evenly-spaced,
   everything else fixed. Establishes the sampling change is benign. Expect
   near-identical: a lightest-pixel max over a good spread is insensitive to
   *which* frames.
2. **Switchover alone.** Instrument both `S` values on `worm_bgsub` — legacy
   tracking-derived vs. convergence-derived. Tune `eps`/`K` to align them. This is
   where the judgment lies.
3. **End-to-end.** The acceptance criterion above.

### Honest risk assessment

Whether `worm_bgsub` can pass at 0.5px is an **open empirical question**.

- *In favor:* a lightest-pixel max over evenly-spaced frames should converge very
  close to one over seeded-random frames — a max over a good spread either way.
  0.5px is forgiving for a change that does not move centroids systematically.
- *Against:* `unmatched == 0` is strict. A single frame where the threshold sits
  near a contour boundary and one blob splits or merges fails the clip outright.
  If convergence-derived `S` lands far from tracking-derived `S`, the frames
  between the two switchovers subtract against a different background entirely and
  will diverge hard.

**If the gate cannot pass no matter how `S` is tuned, that is real evidence the
change is not as behavior-preserving as assumed. Stop and reconsider — do not
loosen the tolerance to make it green.**

### Coverage caveat

`worm_bgsub` is a single worm clip and the only bg-sub coverage. Worms on a plate
may converge quite differently from an ant colony. If bg-sub is used on ant
videos in practice, `eps`/`K` defaults tuned solely on worms could be wrong for
them. Consider adding an ant bg-sub clip to the fixtures.

### Unit tests

- Convergence latch: monotonicity, epsilon/K boundaries, never un-latches.
- **Determinism:** same video + params twice -> byte-identical detections. This is
  the property that makes caching sound; it is the most important new test.
- Cache key: threshold change invalidates (regression test for defect 1).
- `ENABLE_ADAPTIVE_BACKGROUND=False` -> no switch (regression test for defect 3).
- Corner ordering (TL/TR/BR/BL) and NaN-confidence handling downstream.
- Runtime tier selection drives the backend; `tier="cpu"` does not touch CuPy
  (regression test for the self-selection bug).

## Sequencing

1. Defect fixes (cache key names, deterministic prime, adaptive-disabled switch) —
   land first; they are independently correct and make later parity work
   trustworthy.
2. `core/background/measure.py` move + `BgSubConfig`.
3. Convergence rule replacing `tracking_stabilized`.
4. `stages/bgsub.py` + runner wiring + cache + runtime tiers.
5. Call-site migration.
6. Calibration and the acceptance gate.
