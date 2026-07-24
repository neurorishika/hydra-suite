# Runtime Integration Guide

This guide defines the runtime contract for end-to-end integration of:

- New detection models
- New pose models
- New identity/individual-analysis methods (classifiers, embeddings, contrastive features, tag readers)

## Design Goal

All compute-heavy methods must be controlled by one canonical runtime setting:

- `compute_runtime`

No feature should require users to configure a separate runtime selector.

## Source of Truth

Runtime support and translation logic are centralized in:

- `src/hydra_suite/core/runtime/compute_runtime.py`
- `src/hydra_suite/utils/gpu_utils.py`

Core public helpers:

- `CANONICAL_RUNTIMES`
- `allowed_runtimes_for_pipelines(...)`
- `infer_compute_runtime_from_legacy(...)`
- `derive_detection_runtime_settings(...)`
- `derive_pose_runtime_settings(...)`

## Canonical Runtime Values

- `cpu`
- `mps`
- `cuda`
- `onnx_coreml`
- `onnx_cpu`
- `onnx_cuda`
- `tensorrt`

## Integration Checklist (Required)

### 1) Define a pipeline key

Add a stable pipeline name and use it in runtime gating.

Current examples:

- `yolo_obb_detection`
- `yolo_pose`
- `sleap_pose`
- `cnn_identity`
- `head_tail`

For future additions, use names like:

- `appearance_embedding`
- `contrastive_embedding`
- `apriltag_classifier`
- `colortag_classifier`

### 2) Add capability rules

Update `_pipeline_supports_runtime(...)` in `compute_runtime.py` so the new pipeline explicitly defines supported runtimes.

Rules must be strict:

- If unsupported, return `False`.
- Do not silently remap unsupported runtime to a different backend.

### 3) Add runtime translation

If the pipeline consumes legacy backend knobs, add mapping from `compute_runtime` to backend settings.

Examples already used:

- Detection: `yolo_device`, `enable_onnx_runtime`, `enable_tensorrt`
- Pose: `pose_runtime_flavor`, `pose_sleap_device`

### 4) Wire UI intersection gating

Ensure the UI includes the new pipeline in the runtime context set.

TrackerKit pattern:

- Gather enabled pipeline set.
- Call `allowed_runtimes_for_pipelines(...)`.
- Populate runtime dropdown from the intersection.

PoseKit uses the same pattern for active prediction backend scope.

### 5) Implement runtime lifecycle

If the integration has long-lived resources (service/subprocess/session), lifecycle must be run-scoped:

- Initialize once per run.
- Warmup once.
- Close on complete/error/cancel.

Use existing runtime manager/service patterns where possible.

### 6) Export artifacts automatically

If ONNX/TensorRT export is needed:

- Generate artifacts automatically.
- Store artifacts adjacent to model paths.
- Save runtime metadata signature for freshness checks.
- Never require a manual export path for normal operation.

### 7) Keep cache keys runtime-correct

Any cached output that depends on runtime/model/export shape must include those inputs in cache identity.

For new features:

- Include model fingerprint and runtime flavor in cache signatures.
- Include feature-specific shaping params (for example max instances, embedding dimension, preprocessing mode).

### 8) Lock controls during compute

UI controls that could invalidate active runtime sessions must be disabled while jobs are running.

This prevents mid-run backend switches and thread crashes.

### 9) Add tests (minimum bar)

Add/extend tests for:

- Capability matrix and intersection gating.
- Runtime translation determinism from `compute_runtime`.
- Migration from legacy config values.
- Lifecycle correctness (startup/teardown on success and failure).
- Artifact auto-export + freshness behavior.
- Failure fallback behavior with explicit logging.

## End-to-End Acceptance Criteria

A new model/method integration is complete only when:

1. It appears in runtime gating with explicit support rules.
2. It runs from canonical `compute_runtime` without extra runtime selectors.
3. Its ONNX/TensorRT artifacts are auto-managed (if applicable).
4. Caches remain valid and runtime/model-aware.
5. TrackerKit and PoseKit behavior is consistent where the feature exists.

## Common Anti-Patterns (Do Not Add)

- Hidden runtime remapping (`onnx_*` requested, CPU used without notice).
- Feature-specific runtime dropdowns when global runtime is available.
- Manual exported-model-path requirements for standard workflows.
- Runtime checks scattered across GUI/business logic without shared resolver usage.

## Sliced Inference (SAHI)

Direct-mode OBB detection supports optional SAHI-style sliced inference
(`SliceConfig`, off by default). `run_obb` dispatches to
`stages/slicing.py:run_direct_sliced` when `config.obb.direct.slice.enabled`.

- **Geometry:** `auto_model` (tile = model imgsz, no resample — the fast path),
  `auto_object` (tile from expected object size), `custom` (explicit size).
- **Merge:** `merge_policy` (nms/nmm/greedy_nmm) × `merge_metric` (iou/ios) ×
  `merge_backend` (cv2 default; gpu = native-cuda only, cv2-validated). Default
  `greedy_nmm` + `ios` + `0.5`.
- **Merge gate is tile geometry, never the configured ratio:** the merge step
  is gated on `tiles_overlap(plan.tiles)` — whether the *actual* planned tiles
  overlap — not on whether `overlap_height_ratio`/`overlap_width_ratio` is
  nonzero. `get_slice_bboxes` flushes the last tile on each axis to the frame
  edge, so tiles genuinely overlap even at a configured ratio of `0.0` (e.g. a
  300px frame with 256px tiles yields `[0,256)` and `[44,300)` — 212px of real
  overlap). Using the configured ratio to decide whether to merge was a real
  bug: it skipped merging in exactly the cases where cross-tile duplicates
  occur. Always derive the decision from tile geometry, never from the
  config ratio.
- **Cost:** tiles flatten into a predict batch that is chunked to at most
  tiles-per-frame images (`slicing.MAX_TILE_CHUNK`), so peak activation memory
  is bounded rather than `frames × tiles`; the overlap-band pre-filter caps the
  O(n²) merge; native-cuda preserves `_RawOBBTensors` (whole when the planned
  tiles do not overlap, band-only sync when merging). TRT engine batch is sized
  from the same tile-chunk bound, not window depth.
- **Tile-count ceiling:** `plan_slices` raises `ValueError` above
  `slicing.MAX_TILES_PER_FRAME` (4096). A reachable `advanced_config.json`
  combination (`SLICE_HEIGHT=SLICE_WIDTH=64`, `SLICE_OVERLAP=0.9`) would
  otherwise plan ~53k tiles per 1080p frame.
- **ROI gating is implemented but NOT wired:** `plan_slices` accepts a
  `roi_mask` and drops tiles that do not intersect it (falling back to the full
  grid if the mask would drop everything), and that behaviour is unit-tested —
  but every production call site currently passes `roi_mask=None`, so no tile
  is ROI-dropped in the shipped pipeline. Threading the mask through `run_obb`
  is a follow-up. ROI correctness does not depend on it: the filtering stage
  re-applies the mask per detection, so tile gating is purely a compute
  optimization.
- **Cache:** slice params fold into `detection_cache_key` only when enabled, so
  existing non-sliced caches stay valid.

### `gpu` merge-backend performance

The `gpu` merge backend (native CUDA kernel) exists only where it measurably
beats the `cv2` oracle. Measured on an RTX 6000 Ada, torch 2.11+cu130, cv2 as
the correctness oracle:

| N | cuda kernel | cv2 | speedup |
|---|---|---|---|
| 50 | 1.31 ms | 1.21 ms | 0.93x |
| 100 | 1.28 ms | 4.72 ms | 3.68x |
| 200 | 1.41 ms | 17.83 ms | 12.62x |
| 400 | 3.51 ms | 71.33 ms | 20.33x |

The CUDA crossover is around N=50 detections per merge call; below that, `cv2`
is faster or comparable. `cv2` remains the default `merge_backend` and the
correctness oracle everywhere — `gpu` is an opt-in, CUDA-only acceleration for
high-detection-count scenes.

## Related Docs

- [GPU Backends](gpu-backends.md)
- [Extending Detection](extending-detection.md)
- [Extending Identity](extending-identity.md)
- [Compute Runtimes (User Guide)](../user-guide/compute-runtimes.md)
