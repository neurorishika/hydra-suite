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
- `vitpose_pose`
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

## ViTPose Training CLI

ViTPose has a standalone training entry point (invoked by PoseKit's Training Runner
dialog, or directly for scripted/headless runs):

```bash
python -m hydra_suite.core.individual.pose.vitpose.training --config run.json
```

`run.json` is validated against the `RunConfig` schema in
`src/hydra_suite/core/individual/pose/vitpose/training/config.py`
(`validate_run_config` rejects unknown keys and bad ranges before training starts).
Fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `init_checkpoint` | `str` | required | Path to a pretrained ViTPose checkpoint (COCO catalog download or Browse-selected animal/local checkpoint). |
| `variant` | `str` | required | One of the uppercase `VARIANTS` keys (`S`, `B`, `L`, `H`) from `vitpose/config.py`. |
| `num_keypoints` | `int` | required | Must be positive; must match the project's keypoint schema. |
| `dataset_dir` | `str` | required | YOLO-pose dataset root (images + label files). |
| `output_dir` | `str` | required | Destination for checkpoints, logs, and the loss-curve plot. |
| `device` | `str` | `"cpu"` | Torch device string (`cpu`, `mps`, `cuda`). |
| `epochs` | `int` | `40` | Must be positive. |
| `batch_size` | `int` | `16` | |
| `lr` | `float` | `5e-4` | |
| `weight_decay` | `float` | `0.1` | |
| `drop_path` | `float` | `0.1` | Stochastic depth rate. |
| `sigma` | `float` | `2.0` | Heatmap target Gaussian sigma. |
| `grad_clip` | `float` | `1.0` | |
| `val_fraction` | `float` | `0.2` | Must be strictly between 0 and 1. |
| `seed` | `int` | `0` | |
| `resume_from` | `str \| None` | `None` | Optional checkpoint to resume training from. |

The CLI prints `DONE best_pck=<value> best_epoch=<n>` on completion and exits non-zero
on validation/training failure.

## SAM2 Escalation

DetectKit's segmentation escalation uses a standalone `Sam2SegmentExecutor` (pipeline key: `sam2_segment`)
that intentionally sits **outside** the tier/`InferenceRunner` system. Prompt-based models (SAM2 takes
point/box prompts and returns segmentation masks) do not fit the `predict(frame)` contract, which assumes
static inference over a frame's detections; SAM2 is interactive and iterative.

The shipped feature is an offline **batch "escalate-all"**, not an interactive per-detection tool: it
reads the existing `obb`/`aabb` labels for one or more selected sources and auto-primes a segmentation
mask for every detection, staging the result for the user to accept or reject in place on the same
source (see Workflow below) — it does not create a new source.

**Workflow:**

1. User picks one or more eligible sources (level `obb`/`aabb`; already-`polygon` sources are shown
   disabled) and a SAM2 variant in the "Escalate to Segment" dialog (`escalate_sam2_dialog.py`).
2. `run_escalation()` (`detectkit/jobs/sam2_escalation.py`) runs in a `Sam2EscalationWorker(BaseWorker)`
   **background `QThread`** — it does not run on the GUI thread. For each source, for every image: the
   existing box label is read (`read_boxes_from_label`), a prompt is auto-built from it
   (`build_prompts` — the box itself plus its center as a positive point, with other detections' centers
   as negative points), and `Sam2SegmentExecutor.set_image()` / `segment(box, points, negative_points)`
   returns a mask + IoU confidence. No user-drawn prompts are involved.
3. `Sam2SegmentExecutor.from_variant(variant)` loads the SAM2 model (torch-only, auto-downloads weights
   from the HF hub via `checkpoints.py`).
4. Each mask is converted to a polygon (`mask_to_contour`); an empty/low-quality mask falls back to the
   original box's rectangle as the polygon so no detection is ever dropped.
5. Results are written to a **per-source staging directory** under
   `artifacts/pending_escalations/<source>-<variant>-<hash>/` (`run_escalation()`,
   `detectkit/jobs/sam2_escalation.py`) and recorded on `OBBSource.pending_escalation`
   (`PendingEscalation`: `staged_path`, `target_level`, `sam2_variant`, `created_at`). The source's own
   canonical `labels/`/`classes.txt` are never touched during staging, and no new source is registered
   — the source list still shows exactly one entry for that source. Re-running escalation over a source
   that already has a pending escalation is skipped by default (recorded in `EscalationResult.skipped`);
   passing `overwrite=True` replaces the staged directory in place.
6. The user reviews and resolves the staged result via `accept_pending_escalation(source)` or
   `reject_pending_escalation(source)` (same module). `accept_pending_escalation` validates the staged
   directory is present and covers every label the source currently has, then promotes it into the
   source's canonical `labels/`/`classes.txt` in place, sets `level`/`sam2_variant` from the pending
   record, resets `reviewed=False` (same meaning as any other machine-derived, not-yet-confirmed
   result — just applied to the existing source instead of a new sibling), removes the staging
   directory, and clears `pending_escalation`. `reject_pending_escalation` discards the staged
   directory and clears `pending_escalation`, leaving the source untouched. Both raise if the source has
   no pending escalation.
7. This flow is driven from the GUI by the "Review escalations…" button in the Tools panel
   (`ToolsPanel.review_escalations_requested` → `MainWindow._on_review_escalations`), which opens
   `ReviewEscalationsDialog` (`gui/dialogs/review_escalations_dialog.py`) listing **every** source in
   the project with a non-`None` `pending_escalation`, not just sources from the escalation run that
   just finished — so a pending escalation left unresolved (e.g. the dialog was closed without acting
   on it, or the project was closed and reopened) is still found and reopened later. The dialog also
   opens automatically right after a run that staged at least one source.

**Key properties:**

- Runs in a background `QThread` (`Sam2EscalationWorker`, a `BaseWorker`), keeping the GUI responsive
  during a run; progress/status are reported via the standard `BaseWorker` signals.
- Model weights are auto-managed (variant catalog via `checkpoints.py`); weights download on first use.
- No caching, no ONNX/TensorRT export (SAM2 is torch-only).
- No per-tier gating; runs on whatever torch device is available.
- `sam2` is imported lazily, only inside `Sam2SegmentExecutor.from_variant` — importing the executor
  module (or `sam2_escalation.py`) does not require `sam2` to be installed.

## Related Docs

- [GPU Backends](gpu-backends.md)
- [Extending Detection](extending-detection.md)
- [Extending Identity](extending-identity.md)
- [Compute Runtimes (User Guide)](../user-guide/compute-runtimes.md)
