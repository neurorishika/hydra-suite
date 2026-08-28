# Active-Learning Pipeline Optimization — Design

**Date:** 2026-08-27
**Status:** Pending implementation plan.
**Scope:** DetectKit + TrackerKit active-learning (AL) frame selection and export

## Problem

AL frame selection and dataset export are far too slow on long videos, in both kits,
for different reasons:

- **DetectKit selection** (`al_worker.py:run_active_learning` →
  `data/al/candidate_pool.py:build_candidate_pool`) scans the entire video and, for
  every frame, reopens a fresh `cv2.VideoCapture` and seeks
  (`data/al/frame_source.py:51-58`), runs the model **three separate times**
  (one base pass + two NMS-threshold perturbations for instability scoring,
  `al_worker.py:125-160` → `data/al/signals.py:203-224`) with no cache of any kind,
  and dedups each candidate's perceptual hash against **every previously kept
  frame** (`candidate_pool.py:67-70`), an O(n²) scan over the whole video.
- **TrackerKit selection** (`core/post/dataset_export.py:generate_active_learning_dataset`)
  already scores from the on-disk detection cache with no re-inference, but does a
  full linear `df[df["FrameID"] == frame_id]` scan of the tracking CSV per frame
  (`dataset_export.py:202`), also O(n²) over the video.
- **TrackerKit export** (`data/dataset_generation.py:_init_detection_runner` →
  `detect_batch`, `dataset_generation.py:322-416,610`) reruns detection from
  scratch, uncached, at export time — even though the on-disk detection cache
  already stores raw detections down to a `1e-3` confidence floor
  (`core/inference/config.py:159-160`, `core/inference/api.py:49`) and the cache
  key deliberately excludes `confidence_threshold`/`iou` specifically so it's valid
  across different consumer thresholds (`core/inference/cache/keys.py:100`).
  `_init_detection_runner` builds its `InferenceRunner` with no `cache_dir` at all
  (line 397), so this cache is never consulted. The prior design doc
  (`docs/superpowers/specs/done/2026-08-17-al-escalated-multi-format-export-design.md`,
  finding #15) called this rerun "defensible" without checking the cache-key
  exclusion — that justification does not hold up; this is a plain missed-reuse bug,
  not a genuine threshold mismatch.
- **DetectKit's detection calls bypass the shared, optimized inference stack
  entirely.** `al_worker.py`'s `_detector_fn`/`_detector_fn_seq`
  (`detectkit/gui/prediction_preview.py:268-384`, siblings of the AL worker's calls)
  hand-roll their own tiling and two-stage sequential logic and call a low-level
  executor directly, rather than going through `core/inference/runner.py`'s
  `InferenceRunner` — the shared component that resolves the runtime-tier backend
  (torch/TensorRT/CoreML), batches frames, and persists to the detection cache.
  DetectKit gets none of that today, in AL scoring or in its main preview flow.

## Current state

| | DetectKit AL | TrackerKit AL |
|---|---|---|
| Selection scoring | Live, uncached, 3x model calls/frame | Cached detections, no re-inference |
| Video decode | Per-frame `VideoCapture` reopen + seek | Per-frame seek (sparse, selected+context only) |
| Dedup | O(n²) global perceptual hash | N/A (scoped to selected+context, per prior AL design) |
| Frame→data lookup | N/A | O(n²) full-CSV scan per frame |
| Export detection | Reuses scoring-time in-memory detections — no rerun | Uncached rerun via `InferenceRunner(cache_dir=None)` |
| Uses `InferenceRunner`? | No — hand-rolled executor calls | Yes, but without `cache_dir` |

DetectKit already avoids double-inference at export (it writes labels from the
detections computed during scoring); its cost is concentrated entirely in
selection. TrackerKit's selection is already close to optimal in approach (cache-
based) but has its own O(n²) bug; its cost is concentrated in the export rerun.

## Design

### A. TrackerKit export: use the existing detection cache

`_init_detection_runner` currently takes only a flat `params` dict
(`dataset_generation.py:322-416`) — `video_path` isn't threaded into it at all, so
it needs a new `video_path` parameter, threaded from `export_dataset`
(`dataset_generation.py:712+`, which already has `video_path` in scope). With that,
`_detect_records_for_frames` (`dataset_generation.py:562-630`) calls the shared
`get_or_compute_raw` helper (section C.2) with
`cache_dir=build_inference_cache_dir(video_path)` (the same helper tracking itself
uses — `core/tracking/session.py:199`, `core/tracking/worker.py:4351-4353`) instead
of calling `runner.detect_batch(...)` directly. Do not simply point `detect_batch`
at the cache dir and hope — `detect_batch` never reads or writes cache by design
(its own docstring).

Since tracking's cache always covers the entire video, and export only ever runs
after a tracking run has already built it (`core/tracking/session.py:504`,
`trackerkit/gui/workers/dataset_worker.py:58`), this is a cache hit — zero new
inference — in the overwhelming common case. Delete the "double inference is
intentional" note (finding #15 in the 2026-08-17 design doc) once this lands — it
was based on a premise (the cache lacks low-enough-confidence detections) that is
false.

(Note: the `background_subtraction` branch of `_init_detection_runner` has a
pre-existing, separate bug — see the correction above — that means this fix only
applies to the YOLO-OBB branch until that's addressed independently.)

### B. TrackerKit selection: fix the O(n²) frame lookup

In `generate_active_learning_dataset` (`dataset_export.py:123-343`), replace the
per-frame `df[df["FrameID"] == frame_id]` (line 202) with a single upfront
`df.groupby("FrameID")` (or an equivalent frame→row-index map built once before the
loop). Mechanical, no behavior change.

### C. DetectKit: route detection through `InferenceRunner`

This is the largest piece. Several unknowns were checked before committing to this
design; findings below correct two premises this design started with.

- **SAM2 mask-priming escalation** (`core/.../sam2_escalation.py`) operates on
  already-written label files via `read_boxes_from_label`, not on live per-frame
  detection — it is a separate, user-triggered job
  (`detectkit/gui/dialogs/escalate_sam2_dialog.py`) that never runs during AL
  scoring. Zero interaction with this change.
- **`OBBSequentialConfig`** (`core/inference/config.py:154-177`) holds only static
  per-frame settings; `run_obb` (`core/inference/stages/obb.py:469-527`) processes
  each frame independently within a batch (`plan → execute → extract → merge`,
  indexed only by position in that call's list). "Sequential" refers to the
  two-stage detect→crop→OBB pipeline applied to one frame, not cross-frame state.
  A scattered, non-contiguous candidate-frame batch works with no special handling.
- **Correction: `OBBSource` is not a config carrier.** It's a dataset-registration
  record (path, level, provenance, `reviewed` flag — `detectkit/gui/models.py:27-54`)
  with no model path, mode, or threshold fields. The real detector-construction seam
  is `main_window.py:_load_active_detector_fn` (1409-1464), which resolves
  `detectkit_resolve_inference_models(project, model_path)` (`detectkit/gui/project.py:660-663`)
  to `kind ∈ {"obb_direct", "sequential", "unknown"}` plus primary/secondary model
  paths, then closes over `predict_obb_for_frame_export`/`predict_obb_for_frame_sequential`
  (`prediction_preview.py:473-531`) accordingly. The adapter (C.1) targets this seam.
- **Correction: sliced/SAHI mode is not wired into AL at all today.**
  `predict_sliced_obb_result` exists in `prediction_preview.py` but
  `_load_active_detector_fn` never calls it — AL scoring only ever runs in
  `obb_direct` or `sequential` mode. The adapter only needs to map to
  `OBBDirectConfig`/`OBBSequentialConfig`; `SliceConfig` is out of scope.
- **Correction: `predict_obb_for_frame_export`/`_sequential` do not floor at
  near-zero confidence.** Both bottom out in `executor.predict(frame, conf=raw_floor,
  iou=...)` where `raw_floor = max(1e-4, confidence_threshold)` — the *caller's*
  threshold, not a fixed floor — so NMS is baked into every forward pass at the
  requested `(conf, iou)`. This is exactly why the current 3 threshold variants cost
  3 real forward passes, and confirms the 3x→1 collapse requires swapping to
  `InferenceRunner`/`run_obb` (which floors near `1e-3` and filters downstream) —
  reusing `prediction_preview.py`'s existing functions cannot achieve it.
- **Correction: `DetectionCacheHandle.close()` overwrites rather than merges, and
  this is deliberate existing convention, not a bug to fix here.** `write_frame`/
  `close()` (`core/inference/cache/store.py:85-227`) write `self._buffer` alone to
  disk — pre-existing on-disk frames not touched in the current session are lost.
  Grepping every current consumer of `caches_all_valid`/`detection_cache_missing_frames`
  confirms **no code anywhere in this codebase does incremental cache merging** —
  the one existing consumer (`core/tracking/worker.py:1055-1091`, backward/replay
  pass) treats a cache as fully valid or refuses to run, matching this repo's own
  documented rule ("Backward mode refuses to run without a valid cache," per
  `CLAUDE.md`) rather than filling gaps. Adding real merge-on-close logic to a class
  the core tracking forward/backward passes also depend on would be a new pattern
  with its own risk surface, for a benefit (cross-run AL accumulation) this design
  does not require. Section C.2 below follows the existing convention instead:
  read-if-fully-covered, else recompute the whole currently-requested set fresh.
- **Correction: `detect_batch` computes a raw (pre-filter) `OBBResult` internally
  but discards it** (`runner.py:1157-1197`) — it calls `run_obb(...)` once per
  batch, then per frame calls `filter_for_source(...)` and keeps only the filtered
  result. Nothing today exposes the raw result, which the NMS-instability collapse
  (C.3) needs. `runner.py` needs a small, behavior-preserving refactor: extract the
  raw batched call into `detect_batch_raw(frames, frame_indices) -> list[OBBResult]`,
  with `detect_batch` becoming `detect_batch_raw(...)` + the existing filter step
  (regression-tested to confirm `detect_batch`'s output is unchanged).
- **Correction: filtering confirms the collapse is real and cheap.**
  `filter_with_indices(raw: OBBResult, config: OBBConfig, roi_mask) -> (OBBResult, indices)`
  (`core/inference/stages/filtering.py:269`) applies confidence gate → size/aspect
  gates → ROI gate → **Python-side NMS** (`_obb_nms`, line ~306) → `max_detections`
  cap, entirely as a post-process over an already-computed raw `OBBResult`. Calling
  it repeatedly with different `iou_threshold`/`confidence_threshold` values on the
  same cached raw result costs zero additional model calls — this is the exact
  mechanism C.3's NMS-instability collapse uses.
- **Correction: today's AL scoring costs 4 detector calls per frame, not 3.**
  `_frame_signals` (`al_worker.py:125-160`) calls `detector_fn` once directly
  (line 136) for base detections, then `score_nms_instability`
  (`data/al/signals.py:203-224`) redundantly recomputes an identical base call
  (line 214, same `base_conf`/`base_iou`) before its two perturbation calls. The
  redesigned flow in C.3 makes this moot — all four reads come from one cached raw
  result — so no separate fix is needed for this redundancy.
- **Pre-existing, unrelated bug spotted, not fixed here:** `_detect_records_for_frames`
  (`dataset_generation.py:562-630`) unconditionally calls `runner.detect_batch(...)`,
  which raises when `self._models.obb is None` — the case for the
  `background_subtraction` branch of `_init_detection_runner`. TrackerKit AL export
  for bgsub-tracked videos appears to already be broken today, independent of this
  optimization effort. Flagged for a separate fix, out of scope here.

#### C.1 Config adapter

Smaller than originally scoped: `core/inference/config.py:build_obb_only_config(...)`
(line 1107) already builds an `InferenceConfig` generically from flat params
(`model_path`, `mode="direct"|"sequential"`, thresholds, `extra_params` escape
hatch) — it's the exact helper TrackerKit's `_init_detection_runner` already uses.
DetectKit doesn't need a new `OBBConfig`-family mapper; it needs a small function,
`data/al/inference_adapter.py` (Qt-free), that reads the same fields DetectKit's
existing `_detector_fn`/`_detector_fn_seq` closures already read (model path,
`kind` from `detectkit_resolve_inference_models`, `crop_pad_ratio`, device) and
calls `build_obb_only_config` with them. Verify during implementation that
`extra_params` covers whatever DetectKit-specific knobs the sequential path needs
(`crop_pad_ratio` etc.) without extending `build_obb_only_config` itself.

#### C.2 Shared cache-reuse helper

No merge-on-close logic — this codebase's existing convention (see correction
above) is followed instead: for a `(cache_dir, requested frame indices)` pair,
either the cache already covers every requested index (pure read, zero inference)
or it doesn't (recompute the *whole currently-requested set* fresh in one batched
pass and write it as one new complete session). A single helper — living in
`core/inference/` (not duplicated per-kit, per this repo's cross-kit rule) —
implements this against the new `detect_batch_raw` (see correction above):

```
def get_or_compute_raw(runner: InferenceRunner, cache_dir, frames, frame_indices) -> dict[int, OBBResult]:
    reader = open_detection_cache_reader(cache_dir) if cache_exists(cache_dir) else None
    if reader is not None and all(reader.read_frame(i) is not None for i in frame_indices):
        return {i: reader.read_frame(i) for i in frame_indices}
    results = runner.detect_batch_raw(frames, frame_indices)
    handle = DetectionCacheHandle(cache_dir, ..., read_only=False)
    for idx, result in zip(frame_indices, results):
        handle.write_frame(idx, result=result)
    handle.close()
    return dict(zip(frame_indices, results))
```

(Exact signature to be finalized in the implementation plan against the real
`InferenceRunner`/`DetectionCacheHandle` APIs — note `open_detection_cache_reader`
and `DetectionCacheHandle` are the same class, confirmed during research.) Both
TrackerKit's export fix (section A) and DetectKit's AL worker (section C.3) call
this same helper — this is the one piece of genuinely shared logic between the two
kits' fixes. TrackerKit's case is the common-case fast path: tracking's cache
already covers the entire video, so export's requested subset is always already
satisfied → pure read, every time.

#### C.3 DetectKit AL worker restructure

`run_active_learning`'s current loop (`al_worker.py:186+`) is strictly
sequential — decode one frame, score it, decode the next — with no batching seam.
A closure-only swap (keep the per-frame loop, just back `detector_fn` with a
per-frame cached call) would get the caching and decode/dedup/prefilter wins but
**not** the genuine multi-frame GPU batching this design is meant to deliver. This
design commits to the restructure, in three phases:

1. **Candidate-selection pass** (sections D, E, F): one sequential video decode
   producing the final candidate frame list — frame-difference prefilter and
   windowed dedup both run inline here, before any model call.
2. **Batched detection pass**: the full candidate list goes through
   `get_or_compute_raw` (C.2) in one or a few batches (respecting
   `_get_detector_batch_size`-style sizing, already precedented in
   `dataset_generation.py:_detect_records_for_frames`), using the adapter (C.1) to
   build the `OBBConfig`. Cache dir uses the same `build_inference_cache_dir(video_path)`
   convention as tracking.
3. **Scoring pass**: for each candidate, `_frame_signals`/`score_nms_instability`
   read the one cached raw `OBBResult` per frame and call
   `filter_with_indices(raw, config, roi_mask)` (`core/inference/stages/filtering.py:269`)
   three times, once per `(conf, iou)` variant — no additional model calls. This
   also incidentally fixes the redundant 4th call (see correction above) since
   there is no longer a separate `detector_fn` closure for `score_nms_instability`
   to invoke.

This changes `DetectorFn`'s role for the AL path specifically: `al_worker.py`
no longer takes an opaque per-frame closure for its detection step (the closures
in `main_window.py:_load_active_detector_fn` are replaced for this call site by
the adapter+helper). Confirm during implementation whether `DetectorFn` is used
by any other caller before deciding whether to keep, narrow, or retire the type
alias itself.

This gets DetectKit AL scoring genuine GPU batching (multiple candidate frames per
`detect_batch_raw` call) and the same runtime-tier backend resolution
(TensorRT/CoreML/ONNX) the rest of the codebase already gets — not just a caching
fix.

### D. DetectKit: sequential single-pass video decode

Replace `frame_source.py`'s per-frame `VideoCapture` reopen + seek with one shared
capture read sequentially in frame order across the full candidate-pool scan.
Reopening a video container per frame is far more expensive than a sequential
`cap.read()` loop; this is likely the single largest wall-clock win on long videos,
independent of every other change in this design.

### E. DetectKit: frame-difference prefilter

Computed inline during the same sequential decode pass (D): a cheap grayscale
absolute-difference (or downsampled mean-absolute-difference) between each frame
and a rolling reference (the previous frame, or a periodically refreshed reference
to tolerate slow lighting drift). Frames whose delta falls below a configurable
motion threshold skip full model scoring entirely — a frame with no visible change
from its neighbor cannot be newly "challenging." A periodic-sampling floor (at
least one scored frame per configurable time window, even through static stretches)
guards against silently missing a rare, low-visual-delta-but-important frame and
preserves temporal coverage of the video. This changes the selection algorithm
(explicitly approved) and needs its own equivalence check — see Testing.

### F. DetectKit: windowed dedup

Replace the whole-video O(n²) perceptual-hash dedup (`candidate_pool.py:67-70`)
with comparison against only the last *K* kept frames (or last *T* seconds), not
every previously kept frame. Near-duplicate frames in a tracking video are
temporally local; a global scan buys nothing a generous window doesn't already
capture, at O(n) instead of O(n²) cost. Default the window generously so behavior
stays close to current output.

## Testing

None of this touches the Kalman filter, assignment, or CSV tracking output, so it
sits outside the MPS+CUDA byte-identical tracking-equivalence gate described in
`CLAUDE.md`. Verification instead:

- Regression test for the `detect_batch`/`detect_batch_raw` refactor (correction
  above): `detect_batch`'s output is byte-identical before and after extracting
  `detect_batch_raw`.
- Unit tests for the shared `get_or_compute_raw` helper (C.2): fully-covered cache
  short-circuits to a pure read with zero `detect_batch_raw` calls; a
  not-fully-covered cache triggers exactly one recompute of the whole requested
  set and persists it; both call sites (TrackerKit export, DetectKit AL worker)
  get correct results.
- Unit tests for TrackerKit's export cache wiring (A): export against a
  pre-existing tracking cache issues zero new inference calls; export against a
  missing/mismatched cache (different model) falls back correctly.
- Fixture-based before/after comparison of DetectKit's selected-frame sets and
  per-frame detection numerics across the `InferenceRunner` swap (C) — this one
  changes the actual detection computation path, not just caching, so needs a
  real numeric diff against the current hand-rolled executor path on at least one
  fixture per mode (direct, sequential — sliced is not part of AL scoring today).
- Fixture test for the frame-difference prefilter (E): a curated fixture with known
  static and motion segments, confirming no known-challenging frame is dropped and
  the periodic-sampling floor fires during static stretches.
- Fixture test comparing windowed dedup (F) against current global dedup's selected
  set on at least one long-video fixture — expect a close, not necessarily
  identical, match (approved algorithm change).
- A wall-clock benchmark on a long-video fixture, before and after, to demonstrate
  the actual goal (this is a performance effort — a passing test suite alone
  doesn't confirm it).

## Deferred / explicitly out of scope

- Migrating `prediction_preview.py`'s main (non-AL) detection flow onto
  `InferenceRunner`. The adapter in C.1 should not preclude this later, but it is
  not part of this effort.
- GPU-batching candidate frames beyond what `detect_batch` already provides
  (e.g. dynamic batch-size tuning), and a cheap non-visual pre-filter beyond frame
  differencing (e.g. optical flow). Revisit only if the above is insufficient.
