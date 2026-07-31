# SAHI Sliced Inference for the InferenceRunner Pipeline — Design

**Date:** 2026-07-23
**Status:** Design approved; pending spec review → implementation plan
**Scope:** Direct-mode OBB detection, all three tasks (`obb` / `detect` / `segment`)
**Depends on:** `feature/obb-direct-detect-segment` merged into `main` (the three
task-specific direct extractors — `_extract_obb_from_boxes`, `_extract_obb_from_masks`,
native `obb` — and their `_RawOBBTensors` variants must be present).

## 1. Motivation

Two failure modes drive this, both the canonical SAHI use case:

- **Small objects in large frames.** After the model's letterbox downscale (e.g. a
  4K frame → 1024 `imgsz`), small animals occupy too few pixels and recall collapses.
- **Crowding.** Dense scenes exceed `max_det` / cause NMS crowding at full-frame scale.

[Sliced inference](https://obss.github.io/sahi/guides/sliced-inference/) tiles the frame,
runs detection per tile at native resolution, remaps detections to frame coordinates, and
merges overlaps. This design ports the *technique* natively (no `sahi` PyPI dependency),
reusing the existing batched predict path, the existing OBB-NMS/convex-hull machinery, and
the CUDA/TensorRT/CoreML executors — with slicing **off by default** and user-toggleable.

Non-goals for v1: sequential-mode stage-1 slicing, bgsub slicing (resolution-independent),
motion/track-gated sparse tiles (a later phase; recall risk for entering animals), and a
GPU rotated-IoU merge kernel (would diverge from the pipeline's cv2 polygon geometry).

## 2. Core architectural decision: slicing is a task-agnostic *wrapper*

Slicing is a frame transform + coordinate remap + merge. It is **orthogonal** to how
detections are extracted from a model result. Therefore it wraps `_run_direct`'s chunked
predict call rather than living inside any extractor. New module:
`core/inference/stages/slicing.py`.

```
run_obb(frames, models, config, runtime)
 └─ direct mode AND config.obb.direct.slice.enabled → run_direct_sliced(...)
      1. plan = plan_slices(frame_hw, slice_cfg, imgsz, roi_mask)   # memoized per video
      2. flatten: [(frame_idx, slice_idx, tile)] across the whole frame window
      3. chunk at the TILE-chunk size → existing predict + per-task extract path, UNCHANGED
      4. offset-remap each tile's detections into frame coordinates (+= tile x0,y0)
      5. group by frame_idx → concatenate → merge (policy × backend) → one result per frame
      6. existing _apply_raw_detection_cap; downstream filtering UNCHANGED
```

Step 3 is the entire compatibility story: `_extract_obb_from_masks` (segment),
`_extract_obb_from_boxes` (detect + fixed angle), and native `obb` each run per-tile,
untouched. `DirectExecutorAdapter` (TRT/CoreML) also works unchanged — it still receives a
plain list of images. **Segment gains recall**: the rotated-rect kernel sees mask crops at
tile resolution instead of a downscaled full-frame mask.

**Return type is conditional, not always `OBBResult`.** `filter_raw` already dispatches on
result type (`_RawOBBTensors` → on-device gates; `OBBResult` → numpy), so the sliced path
returns whichever preserves the most on-device work for the active runtime — see §5e. The
sync is minimized (band-only) rather than materializing every raw tile detection.

**Invariant:** `enabled=False` ⇒ `run_obb` behaves exactly as pre-feature `main`
(the feature is dead code behind the dispatch check), producing byte-identical output.

## 3. Slice planning — three geometry modes

`plan_slices` computes tile boxes once per video (frame size is constant) and memoizes.

- **`auto_model`** (default): tile = the model's own `imgsz` (via existing
  `_resolve_imgsz`). Zero letterbox downscale → max recall per forward pass, and enables
  the exact-tile fast path (§5).
- **`auto_object`**: tile sized so a reference object spans `object_tile_fraction` of the
  tile linearly, derived from `REFERENCE_BODY_SIZE` / object-size thresholds. Tiles then
  letterbox to `imgsz` normally (costs a resample).
- **`custom`**: explicit `slice_height` / `slice_width` / `overlap_*_ratio`.

Grid follows SAHI's `get_slice_bboxes`: step = `slice · (1 − overlap)`; the last tile in
each axis is shifted back to sit flush with the frame edge (no runt tile). Then:

- **ROI gating:** drop any tile whose box contains no live ROI-mask pixel (uses the
  pipeline's existing `roi_mask` / `roi_mask_cuda`). Zero recall change; typically drops
  ~20–25% of tiles for a circular arena in a rectangular frame.
- **Optional full-frame pass** (`perform_standard_pred`, default off): append one
  full-frame entry per frame to catch objects larger than a tile.

## 4. Merge — policy × metric × backend

Three orthogonal knobs:
- `merge_policy ∈ {nms, nmm, greedy_nmm}`
- `merge_metric ∈ {iou, ios}` — `iou = inter/union`; `ios = inter/min(area₁,area₂)`
- `merge_backend ∈ {cv2, gpu}` — *where* the geometry runs (see below)

Merge semantics (backend-independent):
- **`nms`**: greedy suppression, preserves raw model geometry.
- **`nmm` / `greedy_nmm`**: non-maximum *merging* — union overlapping members into one OBB.
  Confidence = max, class = top scorer.

**Default: `greedy_nmm` + `ios` + `0.5`** (SAHI's detection default). Rationale: an animal
straddling a tile boundary yields two *truncated* OBBs whose IoU is low (NMS keeps both as
duplicates) but whose IoS is high — merging unions them into one correct box. `nms` remains
available for users who want untouched model geometry. `overlap == 0` skips merge entirely.

### 4a. Pluggable merge backend

The merge is a dispatch seam (`merge(policy, metric, corners, conf, cls) → OBBResult`) with
two interchangeable implementations validated against each other:

- **`cv2`** (default, all six paths, correctness oracle): reuses the existing
  `_obb_iou_corners` / convex-hull machinery generalized over the metric; union =
  `cv2.minAreaRect` of stacked member corners, renormalized through existing
  `_normalize_obb_geometry` + `_corners_from_xywhr`. Geometry is consistent with the
  downstream tracking-time cv2 NMS.
- **`gpu`** (opt-in; auto-selected only on the native-cuda `_RawOBBTensors` path): torch-only,
  no cv2. Three parts:
  1. **Pairwise overlap matrix** (IoU/IoS) over band members — *new* vectorized rotated-box
     intersection-area (batched Sutherland–Hodgman polygon clipping), N×N on device.
  2. **Greedy grouping** — compute the N×N matrix on GPU, sync only that small float matrix,
     run the sequential absorption loop on CPU (pure index bookkeeping, no geometry crosses),
     then unions back on GPU.
  3. **Union** — reuse the branch's GPU angle-search kernel
     (`utils/obb_from_mask.py:rotated_rect_from_masks` core: project a point set onto
     `num_angles` axes → tightest rotated rect, sub-grid refined), fed member *corners*
     instead of mask pixels. Runs on CUDA and MPS.

**Scope honesty:** the `gpu` backend benefits only the native-cuda path (the other five are
already CPU-numpy), and the band pre-filter (§5d) already caps the merge to a small subset
while forward passes dominate ~20× — so its win is real but bounded. It is included in v1
because the branch already provides the union primitive; `cv2` stays the default and the
test oracle. Since sliced inference is a *new* feature with no legacy byte-parity contract,
the `gpu` backend need only match `cv2` within tolerance, not bit-for-bit.

## 5. Efficiency across all six paths (tier × device)

### 5a. Exact-tile fast path (largest single win, every path)

When `slice_size == imgsz` (the `auto_model` default), letterbox is the identity: `r = 1`,
zero padding. The design special-cases this:
- No `F.interpolate`, no `F.pad`, no `cv2.resize` — just a `stack`.
- No `_invert_letterbox_on_result`: ultralytics reports pre-batched-tensor coords in tensor
  space, which *is* tile space, so the offset remap `+= (x0, y0)` is the only transform.
- No resampling ⇒ merge operates on exact model geometry (`minAreaRect` unions are exact).

This is why `auto_model` is the default; `auto_object` / `custom` are documented as costing
a resample.

### 5b. Per-path tile delivery

| Tier / device | Frame source | Strategy |
|---|---|---|
| **cpu** | numpy | numpy slice views; `ascontiguousarray` only when the executor demands it. Cost dominated by forward passes. |
| **gpu / cuda** (nvdec) | CUDA uint8 HWC tensor | Tiles are zero-copy device views. One fused batch op `stack → permute → float → ÷255` over the chunk (replaces the per-frame Python loop in `_gpu_letterbox_batch`). No H2D traffic. |
| **gpu / cuda** (numpy frames) | numpy | Upload the frame **once**, tile on device (1 transfer of 1.0× vs N transfers of ~1.56× at 0.2 overlap). |
| **gpu / mps** | numpy | Upload-once-then-tile. Extends the device-tensor batch path (currently gated on `frames[0].is_cuda`) to any device tensor. Safe because §5a means no resampling ⇒ cannot diverge from ultralytics' cv2 letterbox. |
| **gpu_fast / tensorrt** | either | Tiles → `DirectExecutorAdapter.predict` as a plain list, unchanged; its internal `_preprocess_cuda_batch` also no-ops under §5a. |
| **gpu_fast / coreml** | numpy | Same list-of-tiles contract. |

### 5c. Batch-shape threading (TensorRT correctness)

`detection_batch_size` currently means both "frames per window" (pipeline.py:165) and
"engine batch profile" (runner.py:147 → `load_obb_models`). Slicing forks them:
- **Window depth stays frame-based** — cache windows, boundaries, byte-parity untouched.
- **Model batch = tiles per chunk.** `load_obb_models(..., batch_size=…)` must receive the
  *tile-chunk* size when slicing is enabled, or the exported TRT engine's dynamic profile
  won't cover the shapes it is fed (silent `setInputShape` failure). Chunking is tile-major
  over the flattened `(frame, tile)` list, so peak memory is bounded by one chunk,
  independent of `N_tiles × window_depth`.

### 5d. Merge-cost containment (CPU-bound on every path)

The merge is O(n²) convex-hull intersections; with crowding `n` reaches several hundred.
**Overlap-band pre-filter:** a detection whose AABB lies wholly inside a tile's *exclusive*
region (covered by no other tile) cannot have a cross-tile duplicate — no other tile saw
those pixels. Restrict the quadratic merge to detections whose AABB touches an overlap band;
everything else passes through untouched. Exact, not approximate. At 0.2 overlap, typically
<20% of detections enter the quadratic stage. The existing AABB pre-check inside `_obb_nms`
still applies on top. `overlap == 0` ⇒ merge skipped.

### 5e. Conditional return type & minimal sync (native-cuda path)

`_RawOBBTensors` exists **only** on the native-cuda (`gpu` tier, torch backend) path —
`tensor_on_cuda = cuda_mode and gpu_native` (`runtime.py:139`). TensorRT/CoreML (`gpu_fast`)
and cpu/mps already return CPU numpy, so there is no roundtrip to preserve on those five
paths: the sliced path returns `OBBResult` for them at no extra cost.

On the native-cuda path the sliced path preserves `_RawOBBTensors` as far as possible:
- **`overlap == 0` (or `merge_policy == nms` with no cross-tile union):** no merge is needed.
  Remap (`+= x0,y0`) and concat are pure on-device ops → return a concatenated
  `_RawOBBTensors`; the existing `filter_from_tensors` still gates on-device and syncs only
  survivors, exactly as the non-sliced path does today. Zero regression.
- **`overlap > 0` with merging:** keep remap + concat + the cheap gates (conf-floor,
  geometry-validity, ROI) + overlap-band classification **on device**. Detections in a
  tile's *exclusive* region cannot have a cross-tile duplicate, so they flow straight into
  `filter_from_tensors` untouched. Only the overlap-band members round-trip for the merge —
  and with `merge_backend == gpu` even those stay on device (only the small N×N matrix syncs,
  §4a). The merged band results are concatenated back with the on-device exclusive-region
  survivors. This is the minimal possible sync, not a full materialization.

The merge (whichever backend) runs **before caching** — its threshold is in the cache key
(§7) — so it is part of raw-detection production, not the re-tunable filtering stage.

## 6. Config schema

New dataclass in `core/inference/config.py`, nested on `OBBDirectConfig`:

```python
@dataclass
class SliceConfig:
    enabled: bool = False
    geometry_mode: Literal["auto_model", "auto_object", "custom"] = "auto_model"
    # custom mode
    slice_height: int = 0
    slice_width: int = 0
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    # auto_object mode
    object_tile_fraction: float = 0.15    # reference object linear span / tile
    # merge
    merge_policy: Literal["nms", "nmm", "greedy_nmm"] = "greedy_nmm"
    merge_metric: Literal["iou", "ios"] = "ios"
    merge_threshold: float = 0.5
    merge_backend: Literal["cv2", "gpu"] = "cv2"   # gpu auto-used only on native-cuda
    # cost
    perform_standard_pred: bool = False   # extra full-frame pass
```

`OBBDirectConfig` gains `slice: SliceConfig = field(default_factory=SliceConfig)`.
`from_dict`/`to_dict` round-trip it. `enabled=False` ⇒ the whole feature is dead code at the
`run_obb` dispatch check.

## 7. Cache key

Slicing changes *which raw detections exist*, so `detection_cache_key` must fold the slice
params into `config_hash` when `enabled=True`. When `enabled=False`, the key must be
**identical to today's** (empty slice contribution) so existing caches stay valid and the
byte-parity gate holds. `confidence_threshold` / `iou` stay excluded (re-applied at tracking
time), exactly as now.

## 8. GUI (TrackerKit)

Mirrors how `feature/obb-direct-detect-segment` wired `combo_yolo_direct_task`:

- **Detection panel:** checkbox `Enable sliced inference (SAHI)` + a geometry-mode dropdown,
  enabled only in direct mode. Persisted through `ConfigOrchestrator` (`get_cfg` load, save
  dict, and the `UPPER_SNAKE` runtime dict).
- **Advanced config (`advanced_config.json`):** `slice_overlap`, `slice_merge_policy`,
  `slice_merge_metric`, `slice_merge_threshold`, `slice_merge_backend`,
  `slice_object_tile_fraction`, `slice_perform_standard_pred`, `slice_height`,
  `slice_width` — threaded via `worker.py`'s param dict exactly like the branch's
  `obb_seg_*` kernel knobs.
- **Preview worker:** "Test Detection" already routes through `_run_direct`; because slicing
  lives inside `run_obb`, the preview shows sliced results with no extra wiring.

## 9. Testing

- **Planning:** grid coverage; flush-to-edge last tile; overlap math; ROI tile-drop count;
  `auto_model` → `imgsz`; `auto_object` sizing.
- **Remap / merge:** straddling detection → correct frame coords; `nms` vs `nmm` vs
  `greedy_nmm`; `iou` vs `ios` on a truncated-overlap pair; overlap-band pre-filter selects
  the right subset; union box == `minAreaRect` of members.
- **Merge backend equivalence:** `gpu` backend matches `cv2` (the oracle) within tolerance on
  the pairwise matrix, grouping, and union geometry, across randomized OBB sets; degenerate
  cases (single member, collinear corners, zero-area).
- **Task coverage:** `obb` / `detect`(fixed-angle) / `segment`(mask) each through the sliced
  path (segment via the CoreML/TRT smoke-test doubles the branch already ships).
- **Path coverage:** cpu numpy; a fake CUDA-tensor frame source; a `DirectExecutorAdapter`
  double — assert the exact-tile fast path takes no resample when `slice == imgsz`.
- **Parity gate:** `enabled=False` byte-identical to pre-feature `main` (unit + the
  equivalence harness clip on MPS, and CUDA on mehek per CLAUDE.md).
- **Cache:** toggling `enabled` changes the key; `enabled=False` key equals today's.

## 10. Acceptance

1. `enabled=False` is byte-identical to pre-feature `main` (equivalence harness, both
   platforms) and its detection cache key is unchanged.
2. Slicing works on all three tasks (`obb`/`detect`/`segment`) and all six tier×device
   paths, with the exact-tile fast path verified resample-free.
3. Merge policy / metric / threshold / backend, geometry mode, overlap, and full-frame
   toggle are all user-configurable (panel toggle + advanced config), defaulting to
   `greedy_nmm` / `ios` / `0.5` / `cv2` / `auto_model` / `0.2` / off.
4. TensorRT engine batch profile is sized from the tile-chunk size, not window depth.
5. Native-cuda path preserves `_RawOBBTensors` when `overlap == 0`; when merging, only the
   overlap band syncs (or, with `merge_backend == gpu`, only the N×N matrix). The `gpu`
   merge backend matches the `cv2` oracle within tolerance.
