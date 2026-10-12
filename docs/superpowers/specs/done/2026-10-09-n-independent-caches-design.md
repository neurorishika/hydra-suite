# N-independent inference caches

**Status:** Shipped — merged to main (feat/n-independent-caches, 2026-10-10).

## Goal

The number of animals, N (`MAX_TARGETS`), becomes a replay-time knob. A video
whose inference caches exist can be re-tracked at any N in 1..1024 without
running any detector or per-animal model again. N is removed from every
inference-cache key and from every extraction-time decision.

Backward compatibility is explicitly **not** required: the cache schema bumps
and every existing cache is rebuilt once. Tracking outputs change (accepted
behaviour change); the equivalence harness is re-baselined, not held
byte-identical to `legacy/main`.

## One declared limit: `MAX_DETECTIONS_PER_FRAME = 1024`

A single named constant (new home: `core/inference/limits.py`) replaces
`MAX_RAW_CANDIDATES_PER_FRAME` (obb.py, 1024) and
`MAX_DOWNSTREAM_CROPS_PER_FRAME` (filtering.py, 128). It bounds:

1. detections stored per frame in the detection cache,
2. per-animal analyses per frame (head-tail, pose, CNN, AprilTag, identity
   evidence),
3. N itself.

It is LOUD:

- **N > 1024** is rejected when engine params are built (GUI and CLI share
  `trackerkit/engine_params.py`), with an error naming the limit.
- **A frame exceeding the limit** is truncated to the top 1024 by confidence;
  a `WARNING` is logged for each affected frame (frame index, candidate count)
  and a summary ("K frames hit the 1024-detection limit") is logged at the end
  of the inference pass and surfaced through the status callback.
- **No silent clamps remain.** Today `_effective_max_detections`
  (filtering.py) silently clamps N to 128 — every run with N > 128 since
  `893cde19` (2026-09-03) kept at most 128 detections per frame. That clamp is
  removed.

## 1. YOLO extraction (direct whole-frame / sliced / sequential)

- No N-derived cap anywhere in extraction: model `max_det`, sequential
  stage-1 crop count, tile reservoir, overlap-merge pre/post caps and
  `materialize_tensors` all use `MAX_DETECTIONS_PER_FRAME` instead of
  `effective_raw_detection_cap(config)`.
- New fixed `EXTRACTION_CONFIDENCE_FLOOR = 0.01` (raised from the 1e-3
  `confidence_floor` default) bounds extraction. Everything ≥ floor is kept
  (subject to the 1024 limit) and stored **sorted by confidence descending**,
  detection ids re-stamped in rank order (existing `_apply_raw_detection_cap`
  behaviour, now always applied with the 1024 limit).
- Memory admission (`estimated_prediction_job_bytes`) sizes jobs for the
  1024 limit, not 2N: the compact output term counts all 1025 candidate
  slots (limit + probe row); the dense segment-mask term assumes at most
  `DENSE_MASK_ESTIMATE_CANDIDATES` (64) full-resolution masks per item -- a
  memory-estimate assumption, never a detection cap (see `limits.py`).
- `OBBConfig.raw_detection_cap` is retained (ruling R4): DetectKit preview
  and active-learning export still use it; tracking extraction never reads
  N through it. `max_detections` remains the replay-time final cap (= N).
- Detection cache key: `max_detections` and `raw_detection_cap` removed from
  `_direct_raw_config_hash` / `_sequential_config_hash`; the floor and the
  limit are folded in. `CACHE_SCHEMA_VERSION` 5 → 6.

## 2. Background subtraction

- Extraction stores every contour passing the N-free size filters, sorted by
  area descending, up to the 1024 limit (loud, as above).
- The two N rules move to replay (`filter_for_source` bgsub branch):
  skip the frame when contour count > `N * MAX_CONTOUR_MULTIPLIER`; keep the
  top N by area (stable sort).
- `MAX_TARGETS` removed from `_BGSUB_KEY_PARAMS`.

## 3. Replay (the only place N lives)

`filter_with_indices` order, applied to the confidence-sorted cached frame:

1. **2N window:** take the first `min(2N, stored)` rows — a prefix slice of
   the already-ranked cache, never a re-sort (re-sorting flips tie order).
2. confidence / size / aspect / ROI / OBB-NMS filters (unchanged).
3. final cap N (largest-first, unchanged ordering helpers).

Changing N or any filter never re-runs a detector, provided the detection
cache covers the run's frame range under the current detection settings and
cache reuse is allowed (TrackerKit "Reuse cache"; non-realtime YOLO
workflow). With reuse disabled, in the realtime workflow, or for
background subtraction's forward pass (sequential, always re-detected; its
cache serves the backward pass) detection runs as before.

## 4. Per-animal stages

Head-tail, pose, CNN, AprilTag and identity evidence are computed for
**every detection surviving step 2 with N = 1024** (i.e. the window and final
cap at the limit), not the post-final-cap set, and stored keyed by **raw
cache index**.

The detected/individual-properties caches are NOT among them: they are
written in the tracking loop over the final-N detection ids, i.e. they are
tracking-level final-N artifacts. N is therefore part of their identity
(`compute_detection_hash(..., max_targets=N)` and a recorded `max_targets`
that `IndividualPropertiesCache.is_compatible(max_targets=N)` checks), so a
file written at one N is never opened by a pass at another N.

- Their cache keys gain a hash of the replay filter settings (confidence,
  size, aspect, ROI, NMS IoU) and never contain N. Changing N reuses them;
  changing a filter recomputes only these stages (never detection): the
  batch pass reads the stored, already-ranked detections back
  (`partial_reuse_plan` + `Pipeline.detection_reader`), recomputes and
  rewrites only the stale per-animal caches, and leaves the detection cache
  and every still-valid per-animal cache untouched. This needs the detection
  cache to cover exactly the run's frame range; otherwise the pass runs the
  detector.
- Replay selects the per-animal rows for the post-final-cap indices. A
  missing index is an error (cache incoherent), never a silent NaN/0 fill.
- Fixes the existing index-base bug: CNN (`stages/cnn.py`) and AprilTag
  (`stages/apriltag.py`) store positions in the filtered set while replay
  (`runner._load_cnn_for_indices`) looks them up by raw index.
- Crops are materialised and run in fixed-size chunks so 1024 detections per
  frame does not spike memory.
- `MAX_TARGETS` removed from `get_tracking_cache_model_ids`
  (`trackerkit/tracking_cache.py`), so the inference model id is N-free (N
  re-enters the props-cache identity only, see above). The unused duplicate
  id builder in `trackerkit/gui/orchestrators/tracking.py` is deleted.

## 5. Unchanged

Confidence-density sidecars and autotune fingerprints keep N (cheap,
tracking-level). DetectKit's own caches keep their fixed 300 cap, but
`detectkit/jobs/prediction_cache.py` keys on the shared
`CACHE_SCHEMA_VERSION`, so the 5 -> 6 bump invalidates existing DetectKit
prediction caches once (rebuilt on next use).

## Verification

- **Unit/integration tests**
  - Cache built at N=10; replay at N=5 and N=20 makes zero detector or
    per-animal model calls and equals a fresh run at each N.
  - Raising N yields no NaN pose/head-tail rows.
  - CNN/AprilTag replay returns the result of the right detection when
    filtering removed earlier rows.
  - N=1025 is rejected; N=200 keeps 200 detections (regression for the 128
    clamp).
  - A synthetic frame with >1024 candidates truncates and warns; summary
    line emitted.
  - Stress: 1024 detections in one frame through head-tail/pose/CNN; peak
    memory recorded on MPS and CUDA.
- **Equivalence harness** run twice on the new code (determinism floor) on
  MPS and CUDA; differences vs the previous baseline are attributed to the
  floor, the window semantics and the per-animal superset, and reported.
- **Cost** on the fixture clips: stored detections/frame, inference wall
  time, per-animal crop counts, before vs after.

## Risks

- More per-animal work when many low-confidence detections pass the filters
  (e.g. confidence threshold 0.05). Measured in Verification before merge.
- Concurrent SAHI-unification work touches `stages/slicing.py`/`obb.py`;
  rebase onto main before merge.
