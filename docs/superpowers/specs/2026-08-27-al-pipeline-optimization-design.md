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

Wire `cache_dir=build_inference_cache_dir(video_path)` (the same helper tracking
itself uses — `core/tracking/session.py:199`, `core/tracking/worker.py:4351-4355`)
into the `InferenceRunner` built in `_init_detection_runner`
(`dataset_generation.py:397`). Do not simply point `detect_batch` at the cache dir
and hope — `detect_batch` never reads or writes cache by design (its own
docstring). Instead, `_detect_records_for_frames` (`dataset_generation.py:562-630`)
gets the shared cache-reuse wrapper described in section C.2 below: check the
cache for each requested frame, compute+persist only the misses, read the rest.

This makes export a cache hit in the overwhelming common case — export is only ever
invoked after a tracking run has already built the cache
(`core/tracking/session.py:504`, `trackerkit/gui/workers/dataset_worker.py:58`).
Delete the "double inference is intentional" note (finding #15 in the 2026-08-17
design doc) once this lands — it was based on a premise (the cache lacks
low-enough-confidence detections) that is false.

### B. TrackerKit selection: fix the O(n²) frame lookup

In `generate_active_learning_dataset` (`dataset_export.py:123-343`), replace the
per-frame `df[df["FrameID"] == frame_id]` (line 202) with a single upfront
`df.groupby("FrameID")` (or an equivalent frame→row-index map built once before the
loop). Mechanical, no behavior change.

### C. DetectKit: route detection through `InferenceRunner`

This is the largest piece. Three unknowns were checked before committing to this
design and all came back clean (no adapter-blocking issues):

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
- **Cache writes for a batch result.** `DetectionCacheHandle.write_frame(frame_idx,
  result=...)` (`core/inference/cache/store.py:85`) and `read_frame`
  (line 148, matched by `frame_indices == frame_idx`, not position) tolerate
  out-of-order writes. The higher-level `CacheWriter` (`cache/writer.py:31`)
  assumes a strictly-ascending single consumer and is the wrong class to reuse for
  a scattered candidate set — the adapter drives `DetectionCacheHandle` directly.

#### C.1 Config adapter

New module, `detectkit/gui/inference_adapter.py` (or `data/al/inference_adapter.py`
if it should be importable without Qt — prefer this, since `data/al/` is already
Qt-free and DetectKit's `OBBSource` config is a plain dataclass): translates a
DetectKit `OBBSource`'s existing config (model path, direct/sliced/sequential mode,
tiling params, thresholds) into the matching `OBBConfig`/`OBBDirectConfig`/
`SliceConfig`/`OBBSequentialConfig`. This is new code — no such mapping exists
today, since DetectKit's preview and AL paths both bypass `InferenceRunner`
entirely and hand-roll tiling/sequential logic instead (`prediction_preview.py`'s
own docstring says this is deliberate). This adapter is scoped to the AL path only;
migrating `prediction_preview.py` itself is out of scope for this effort, but the
adapter should not be written in a way that forecloses reusing it there later.

#### C.2 Shared cache-reuse wrapper

A single helper — living in `core/inference/` (not duplicated per-kit, per this
repo's cross-kit rule) — implementing "check cache, compute misses, read all":

```
def get_or_compute(runner: InferenceRunner, cache_dir, frames, frame_indices, obb_config) -> dict[int, OBBResult]:
    handle = DetectionCacheHandle(cache_dir, ...)
    missing = [i for i in frame_indices if not handle.is_valid(i)]
    if missing:
        results = runner.detect_batch(frames_for(missing), missing, obb_config)
        for idx, result in zip(missing, results):
            handle.write_frame(idx, result=result)
        handle.close()
    return {i: handle.read_frame(i) for i in frame_indices}
```

(Exact signature to be finalized in the implementation plan against the real
`InferenceRunner`/`DetectionCacheHandle` APIs.) Both TrackerKit's export fix
(section A) and DetectKit's AL worker (section C.3) call this same helper —
this is the one piece of genuinely shared logic between the two kits' fixes.

#### C.3 DetectKit AL worker swap

`al_worker.py`'s detector calls route through the adapter (C.1) + wrapper (C.2)
instead of `predict_obb_for_frame_export`/`_sequential`. Cache dir uses the same
`build_inference_cache_dir(video_path)` convention as tracking, so repeated
selection runs on the same video — and even a later TrackerKit tracking run on that
same video/model — reuse the same on-disk cache.

`score_nms_instability`'s three threshold variants (`signals.py:203-224`) become:
one `get_or_compute` call (populates/reads the cache once per frame), then the
NMS/confidence-threshold re-filter applied to the same cached raw detections for
each of the three thresholds — no second or third forward pass. Confirm during
implementation which existing filtering entry point (`core/inference/stages/filtering.py`)
reapplies threshold/NMS at read time, and use that rather than re-deriving it in
DetectKit-side code.

This also gets DetectKit AL scoring genuine GPU batching (multiple candidate frames
per `detect_batch` call) and the same runtime-tier backend resolution
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

- Unit tests for the shared cache-reuse wrapper (C.2): cache hit skips compute,
  cache miss computes and persists, mixed hit/miss batches return correct results
  for both call sites (TrackerKit export, DetectKit AL worker).
- Unit tests for TrackerKit's export cache wiring (A): export against a
  pre-existing tracking cache issues zero new inference calls; export against a
  missing/mismatched cache (different model) falls back correctly.
- Fixture-based before/after comparison of DetectKit's selected-frame sets and
  per-frame detection numerics across the `InferenceRunner` swap (C) — this one
  changes the actual detection computation path, not just caching, so needs a
  real numeric diff against the current hand-rolled executor path on at least one
  fixture per mode (direct, sliced, sequential).
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
