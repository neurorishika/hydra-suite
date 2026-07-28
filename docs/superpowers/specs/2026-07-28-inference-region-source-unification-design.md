# Inference Region-Source Unification (Phase C) — Design

**Date:** 2026-07-28
**Status:** Design approved (ambition + key decisions locked); pending spec review → implementation plan
**Scope:** `src/hydra_suite/core/inference/stages/obb.py`, `slicing.py`, `slicing_cuda.py`, `merge.py`/`merge_gpu.py`, and the executor seam (`runtime_artifacts.py`/`direct_executors.py` consumed, mostly unchanged). No change to detection *results*, the filtering/tracking stages, config-visible behavior, or the `.npz` cache.
**Depends on:** Geometry-levels A + A.5 (merged). A.5 cleared C's three former blockers, so **C is now a pure maintainability refactor** — unify three region-generation modes onto one pipeline — plus one new capability (sequential-mode slicing) that falls out of the abstraction.

## 1. Motivation

The OBB detection stage has three region-generation modes — whole frame (`_run_direct`), grid tiles (SAHI `run_direct_sliced`), and stage-1 proposal crops (`_run_sequential`) — that are structurally identical (`regions → predict → extract-with-remap → merge`) but implemented as three parallel code paths. The duplication is now severe and has already hidden bugs (the `slicing_cuda` docstring cites two: tier dispatch keyed off the wrong flag; overlap gate double-counting):

- **Task→extractor dispatch appears in 5 places:** numpy (`_run_direct`, `slicing._extract_tile`, `_run_sequential`) and raw-tensor (`_run_direct` cuda branch, `slicing_cuda.extract_raw_tile`).
- **Region→frame remap appears in 3 sites with 2 incompatible mechanisms:** `slicing._offset_result` (numpy translate, post-extraction), `extract_obb_result(offset=, scale=)` (numpy affine, during extraction), `slicing_cuda._remap_raw` (raw translate, on-device).
- **Per-frame merge appears in 3 forms:** `slicing._merge_frame_obb_results`, `slicing_cuda.assemble_raw_frames`, and the plain `merge_obb_results`+cap in direct/sequential.

C unifies these onto one region-source pipeline while **preserving every tier's fast path exactly** and keeping tracking output **byte-identical** on both platforms.

## 2. Locked decisions

- **Full pipeline unification** (not a partial dedup): `run_obb` becomes `region_source → executor → extract_with_transform → merge_per_frame`, and direct/sliced/sequential become region-source configs.
- **Ship sequential-mode slicing** as a new region source (§7).
- **Per-source merge policy** (§6): Grid does cross-region overlap-band NMS; WholeFrame/Stage1Proposals do plain concat+cap (dedup deferred to the filtering stage — exactly as today). Byte-identity requires NOT adding merge-time NMS to sequential.
- **Affine invariant:** a region's transform is an affine `(offset, scale)`; **`scale ≠ 1` occurs only for sequential crops, which are always numpy/CPU.** The raw-tensor (on-device) universe therefore only ever applies pure translation — the fast path never grows scale support.
- **Preserve each mode's current extract universe** (numpy vs raw) so byte-identity is trivial; sequential's stage-2 stays numpy. Routing sequential-stage-2 through the raw path is a **one-line region-source flag attempted as a harness-gated final step** (§5.2) — enabled only if proven byte-identical, else shipped off.

## 3. Research findings: the two universes and honest device residency

`RuntimeResolver.resolve("obb")` + `RuntimeContext.from_config` produce, per tier:

| Tier | Platform | Backend/device | Executor | `tensor_on_cuda` | Extract universe |
|---|---|---|---|---|---|
| `cpu` | any | torch/cpu | ultralytics YOLO | False | **numpy** `Results` |
| `gpu` | CUDA | torch/cuda | ultralytics YOLO | **True** | **raw** `_RawOBBTensors` |
| `gpu` | Apple | torch/mps | ultralytics YOLO | False | **numpy** `Results` |
| `gpu_fast` | CUDA + TRT engine | tensorrt/cuda | `DirectTensorRTOBBExecutor` via `DirectExecutorAdapter` | False | **numpy** `Results` |
| `gpu_fast` | Apple + CoreML | coreml/mps | Direct CoreML via adapter | False | **numpy** `Results` |
| `gpu_fast` | CUDA, no engine | torch/cuda (fallback) | ultralytics YOLO | True | **raw** |

**The two universes are the load-bearing axis and MUST be preserved:**
- **numpy universe** (`Results` → `extract_obb_result`/`_extract_obb_from_boxes`/`_extract_obb_from_masks`): cpu, mps, and **all gpu_fast** backends. `DirectExecutorAdapter` deliberately returns CPU-resident ultralytics `Results` so gpu_fast rides the same numpy extract path — that unification already exists and is kept.
- **raw universe** (cuda tensors → `_extract_raw_tensors*` → `_RawOBBTensors`, deferring normalization/cap to `materialize_tensors`): native-torch-CUDA only (`tensor_on_cuda=True`). Zero CPU sync until an unavoidable materialize/merge.

**Honest device residency (this is what the abstraction must encode, not hide):**
- Only `tensor_on_cuda` is zero-CPU-sync. gpu_fast runs inference on GPU/ANE but returns CPU `Results`; NVDEC (decode-on-GPU) is gpu_fast-only, raw extraction is gpu-native-only — different tiers.
- **Grid (sliced):** tile image is a CPU cv2 slice, but predict+extract+remap+merge stay on-device on the gpu tier (`slicing_cuda`).
- **Stage1Proposals (sequential):** `crops.build_crops` forces a **CPU crop round-trip** (`.cpu().numpy()` on CUDA — GPU crop upload/download is "pure waste" for ultralytics stage-2 input), and `_run_sequential` extracts **numpy-only even on gpu** — it never touches raw. This is an inherent CPU boundary plus an unrealized gpu-tier acceleration.

## 4. Architecture

```
run_obb(frames, models, config, runtime, roi_mask):
    source  = select_region_source(config)               # §4.2
    regions = source.plan(frames, models, config, runtime)  # per-frame list[Region]; may run stage-1
    results = execute_regions(regions, source, model, runtime)  # §4.3 chunked, letterbox-aware, tier-native
    obbs    = [extract_with_transform(res, region, task, config, runtime)  # §4.4 task × universe, affine applied
               for res, region in zip(results, regions_flat)]
    return merge_per_frame(obbs, source.merge_policy, config, runtime)      # §4.5
```

### 4.1 `Region`
A dataclass carrying: `image` (the sub-image to predict on — the frame, a tile crop, or a resized proposal crop), `affine` (`offset: (float,float)`, `scale: (float,float)`, mapping region-space → frame-space), `frame_idx`, and the region's provenance (for merge/debug). Whole-frame regions have identity affine; tiles have translate-only; proposal crops have offset+scale.

### 4.2 `RegionSource` protocol
`plan(frames, models, config, runtime) -> list[list[Region]]` (per frame). Also exposes `merge_policy` and a `device_residency` marker (`on_device_capable` vs `cpu_crop_boundary`). Four implementations:
- **`WholeFrame`** — one region per frame, identity affine, `on_device_capable`. (was `_run_direct`)
- **`Grid`** — `plan_slices` geometry → tile crops, translate affines, `on_device_capable`, `merge_policy=overlap_band_nms`. Consumes `roi_mask` for tile gating. (was `run_direct_sliced`)
- **`Stage1Proposals`** — runs stage-1 detect, `build_crops` → resized crops + offset/scale affines, `cpu_crop_boundary`, `merge_policy=plain`. (was `_run_sequential`)
- **`SlicedStage1Proposals`** *(new, §7)* — Grid over stage-1, merge stage-1 boxes to frame, then proposals. `cpu_crop_boundary`.

`select_region_source(config)`: `direct` + `slice.enabled` → Grid; `direct` → WholeFrame; `sequential` + `stage1_slice.enabled` → SlicedStage1Proposals; `sequential` → Stage1Proposals.

### 4.3 Executor
`execute_regions` = the existing chunked-predict machinery, generalized: chunk region images to the tile-batch cap, run the **region-prediction model** (which the source supplies — the direct model for WholeFrame/Grid, the stage-2 obb/segment model for the Proposals sources; stage-1 detection runs earlier, inside `Stage1Proposals.plan`), and handle the CUDA-tensor-frame letterbox-invert (today split between `_run_direct`'s `_frames_are_cuda_tensors` branch and `slicing._predict_tiles`). Uses `DirectExecutorAdapter` for gpu_fast unchanged. The executor is **tier-native and universe-agnostic** — it returns whatever the model returns (`Results` or cuda-tensor `Results`); the universe split happens at extract.

### 4.4 `extract_with_transform` — the single dispatch
Dispatches on **two axes** and applies the region's affine in the matching universe:

```
if runtime.tensor_on_cuda:                      # raw universe (gpu-native)
    raw = { obb: _extract_raw_tensors,
            detect: _extract_raw_tensors_from_boxes,
            segment: _extract_raw_tensors_from_masks }[task](result, ...)
    return _translate_raw(raw, region.affine.offset)   # scale is always 1 here (invariant)
else:                                            # numpy universe (cpu/mps/gpu_fast)
    return { obb: extract_obb_result,
             detect: _extract_obb_from_boxes,
             segment: _extract_obb_from_masks }[task](
        result, ..., offset=region.affine.offset, scale=region.affine.scale)
```

This is the **one** place task dispatch lives (replacing all 5). To make numpy detect first-class, **`_extract_obb_from_boxes` gains `offset`/`scale`** (completing the trio — `extract_obb_result` and `_extract_obb_from_masks` already have it from A/A.5). `_translate_raw` is the generalized `slicing_cuda._remap_raw` (pure on-device translation). `slicing._offset_result` is retired (its translate becomes an affine with scale=1 applied during extraction).

### 4.5 `merge_per_frame`
Concatenate a frame's region OBBResults/`_RawOBBTensors`, apply `_apply_raw_detection_cap`, and — **only when `merge_policy=overlap_band_nms` AND regions actually overlap** (`tiles_overlap` predicate) — run the cross-region band NMS (`merge.merge_obb_detections` / `merge_gpu`, backend chosen by `runtime`). WholeFrame/Stage1Proposals use plain concat+cap (no NMS), byte-identical to today. This unifies `_merge_frame_obb_results` + `assemble_raw_frames` + the plain direct/sequential merge, preserving the numpy(cpu-oracle)/gpu backend split.

## 5. Preserving fast paths & the sequential opt-in

### 5.1 Fast-path preservation (non-negotiable)
The abstraction changes structure, not which universe a tier uses. On the gpu-native tier, WholeFrame/Grid stay raw end-to-end (zero-CPU-sync, gpu merge backend). cpu/mps/gpu_fast stay numpy `Results`. The `DirectExecutorAdapter`, NVDEC path, pinned-buffer executor internals, and merge backends are consumed unchanged.

### 5.2 Sequential raw opt-in (harness-gated final step)
Stage1Proposals keeps `cpu_crop_boundary` and numpy stage-2 extraction — **byte-identical to today**. Because `extract_with_transform` already has a raw branch, routing sequential-stage-2 through raw on the gpu tier is a one-line region-source capability flag. It is attempted as the **last implementation step, gated on the equivalence harness proving byte-identical vs the numpy baseline**. If raw-vs-numpy stage-2 extraction diverges even slightly, the flag ships **off** (documented), and sequential stays numpy. No mid-refactor tradeoff between speed and correctness.

## 6. Merge policy is a region-source property
Encoded on the source, not global: `Grid.merge_policy = overlap_band_nms`; `WholeFrame`/`Stage1Proposals`/`SlicedStage1Proposals.merge_policy = plain`. The `tiles_overlap` geometry predicate (never `overlap_*_ratio` — the documented bug) remains the runtime trigger for whether NMS actually runs. This guarantees sequential/direct output is unchanged (no new NMS).

## 7. Sequential-mode slicing (new capability, shipped)
`SlicedStage1Proposals`: for small-object videos, tile the frame for **stage-1** detection (reusing Grid geometry + stage-1 overlap-band NMS on the merged stage-1 boxes in frame space), then feed the merged boxes to `build_crops`/stage-2 as normal. Config: a `stage1_slice` `SliceConfig` on `OBBSequentialConfig` (off by default) + a `YOLO_SEQ_STAGE1_SLICE_*` builder param. Because it is **new behavior (off by default)** it has no parity baseline — it gets **correctness tests** (tiled stage-1 finds small objects a whole-frame stage-1 misses; stage-1 boxes are correctly merged to frame space and crop offsets are right) rather than an equivalence-vs-baseline gate. Existing sequential (flag off) stays byte-identical.

## 8. Byte-identity constraint & incremental execution order

**Constraint:** direct, sliced, `seq_crop_obb`, and `seq_crop_segment` produce **byte-identical** tracking output vs pre-C main, on **MPS and CUDA**, at every step. This is a hot-path rewrite; a big-bang cutover is forbidden.

**Implementation order (each step ends with its own parity gate before the next begins):**
1. **Affine model + complete the numpy trio:** add `offset`/`scale` to `_extract_obb_from_boxes`; introduce the `Region`/affine dataclass. Pure addition; unit tests. (No mode rerouted yet.)
2. **Unify numpy extract dispatch** into `extract_with_transform` (numpy branch only); route `_run_direct`'s numpy path through it. **Full equivalence matrix (all 7 clips) MPS + CUDA.**
3. **Unify raw extract dispatch** into the raw branch; route `_run_direct`'s cuda path + `slicing_cuda.extract_raw_tile` through it; retire `_remap_raw`/`_offset_result` in favor of the affine. **Gate.**
4. **`RegionSource` protocol + `WholeFrame`;** route `_run_direct` fully through the pipeline. **Gate.**
5. **`Grid` source + `merge_per_frame`;** route `run_direct_sliced` through the pipeline; retire `_merge_frame_obb_results`/`assemble_raw_frames` in favor of `merge_per_frame`. **Gate (sliced clips, both platforms).**
6. **`Stage1Proposals` source;** route `_run_sequential` through the pipeline (numpy stage-2, unchanged universe). **Gate (`ant_obb_sequential`, both platforms).**
7. **`SlicedStage1Proposals`** = ship sequential-mode slicing (§7). Correctness tests; existing sequential (flag off) still byte-identical.
8. **Delete** the dead orchestrators (`_run_direct`/`_run_sequential`/`run_direct_sliced` bodies now thin), dispatch copies, and remap functions. **Final full matrix.**
9. **(Opt-in) sequential raw** (§5.2): attempt the flag; keep only if the harness proves byte-identical.

## 9. Non-goals
- Changing detection *results*, merge *algorithms*, the filtering/tracking stages, or the `.npz` cache schema.
- Touching `DirectExecutorAdapter`/`direct_executors.py` internals, NVDEC, or `merge_gpu` kernels beyond consuming them.
- New runtime tiers or config-visible tier behavior.
- Phase B (SAM2 escalation).

## 10. Testing
- **Unit:** affine model; `extract_with_transform` task×universe dispatch (numpy + a raw fake); `_extract_obb_from_boxes` offset/scale; each `RegionSource.plan` (region count, affines, `merge_policy`, device_residency).
- **Parity (the gate that matters):** the equivalence harness at each step (2–8), all clips, **MPS + CUDA**, byte-identical for direct/sliced/`seq_crop_obb`/`seq_crop_segment`. `fly_obb`+`ant_obb_sequential`+a sliced clip are the fast smoke set; full matrix at steps 5, 6, 8.
- **New-capability:** `SlicedStage1Proposals` correctness (tiled stage-1 recovers small objects; frame-space box merge + crop offsets correct).
- **Opt-in:** sequential-raw equivalence (step 9) — pass ⇒ ship on; fail ⇒ ship off + document.
