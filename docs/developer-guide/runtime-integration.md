# Runtime Integration Guide

This guide defines the runtime contract for end-to-end integration of:

- New detection models
- New pose models
- New identity/individual-analysis methods (classifiers, embeddings, contrastive features, tag readers)

## Design Goal

All compute-heavy methods are controlled by **one stored setting**, the
runtime tier:

```
config.runtime_tier ∈ {cpu, gpu, gpu_fast}
  → RuntimeResolver(tier, platform).resolve(stage)
  → ResolvedBackend(backend ∈ {torch, tensorrt, coreml},
                    device ∈ {cpu, cuda, mps}, used_fallback)
```

No feature may add its own runtime selector or runtime string vocabulary. The
pre-Gen-2 per-stage `compute_runtime` strings (`onnx_cuda`, `onnx_coreml`, …)
are gone. Old configs are migrated once with
`scripts/migrate_runtime_config.py`, and loading a config without
`runtime_tier` raises an error that points to that script.

## Source of Truth

- `src/hydra_suite/runtime/resolver.py` is the single authority. It holds
  `RuntimeResolver`, `ResolvedBackend`, `PlatformInfo`, `STAGES`,
  `available_tiers(platform)` and `tier_label(tier, platform)` (UI labels
  such as "GPU (Metal)" and "GPU-Fast (TensorRT)"), plus `detect_platform()`.
  It is pure: no torch import and no I/O, and artifact availability is
  injected as a callable.
- `src/hydra_suite/runtime/onnx_providers.py` provides
  `execution_providers_for(resolved)`, the ONNX Runtime provider list for a
  `ResolvedBackend`, together with the TensorRT engine-cache options and
  `preload_ort_cuda_libraries()`.
- `src/hydra_suite/core/inference/runtime.py` provides
  `RuntimeContext.from_config(config)`, which resolves the tier and carries
  the result as `RuntimeContext.resolved`. `resolved_backend_for(ctx)`
  derives a `ResolvedBackend` for hand-built contexts (tests, GUI workers).
- `src/hydra_suite/utils/gpu_utils.py` provides the host accelerator
  availability flags.

## Resolution Rules

| Tier | CUDA host | MPS host | CPU-only host |
|---|---|---|---|
| `cpu` | `torch`/`cpu` | `torch`/`cpu` | `torch`/`cpu` |
| `gpu` | `torch`/`cuda` | `torch`/`mps` | `torch`/`cpu`, `used_fallback=True` |
| `gpu_fast` | `tensorrt`/`cuda` if the artifact is available, else `torch`/`cuda` (fallback) | `coreml`/`mps` if available, else `torch`/`mps` (fallback) | `torch`/`cpu` (fallback) |

`bgsub` has no TensorRT/CoreML implementation, so on `gpu_fast` it resolves
like `gpu` and is always flagged `used_fallback=True`.

## Integration Checklist (Required)

### 1) Pick or add a stage key

Current stages (`resolver.STAGES`): `obb`, `head_tail`, `cnn`, `yolo_pose`,
`sleap_pose`, `vitpose_pose`, `bgsub`. A new kind of model gets a new stage
key. Resolve it with `RuntimeResolver.resolve(stage, artifact_available=...)`,
and pass an `artifact_available` callable that reports whether the stage's
fast artifact (TensorRT engine / CoreML package) exists or can be built.

### 2) Consume `ResolvedBackend`, nothing else

Branch on `resolved.backend` and `resolved.device` only. Do not re-derive
backend choices from the tier, the platform, or legacy knobs. If the stage
runs ONNX, take the providers from `execution_providers_for(resolved)`:

- `tensorrt` → TensorRT EP (persistent engine cache under the data dir) +
  CUDA EP
- `coreml` → CoreML EP, when ONNX Runtime's CoreML provider is present on an
  MPS host
- `torch` → CPU EP only (torch stages never run ONNX)

### 3) Report fallbacks

When a `gpu_fast` request cannot use the fast artifact, the resolver returns
`used_fallback=True`. Surface it (TrackerKit shows a note under the tier
selector) and log it. Never remap silently.

### 4) Implement runtime lifecycle

If the integration has long-lived resources (service/subprocess/session),
lifecycle must be run-scoped:

- Initialize once per run.
- Warmup once.
- Close on complete/error/cancel.

Use existing runtime manager/service patterns where possible.

### 5) Export artifacts automatically

If TensorRT/CoreML/ONNX export is needed:

- Generate artifacts automatically.
- Store artifacts adjacent to model paths.
- Save runtime metadata signature for freshness checks.
- Never require a manual export path for normal operation.

### 6) Keep cache keys runtime-correct

Any cached output that depends on runtime/model/export shape must include
those inputs in cache identity:

- Include the model fingerprint and the resolved backend in cache signatures.
- Include feature-specific shaping params (for example max instances,
  embedding dimension, preprocessing mode).

### 7) Lock controls during compute

UI controls that could invalidate active runtime sessions must be disabled
while jobs are running.

### 8) Add tests (minimum bar)

Add/extend tests for:

- Resolution for every tier × platform the stage supports, including
  fallbacks.
- Provider lists for each `ResolvedBackend` the stage can receive.
- Lifecycle correctness (startup/teardown on success and failure).
- Artifact auto-export + freshness behavior.
- Failure fallback behavior with explicit logging.

## End-to-End Acceptance Criteria

A new model/method integration is complete only when:

1. It resolves through `RuntimeResolver` under a stage key, with no extra
   runtime selector.
2. It consumes `ResolvedBackend` (and `execution_providers_for` for ONNX).
3. Its TensorRT/CoreML artifacts are auto-managed (if applicable).
4. Caches remain valid and runtime/model-aware.
5. TrackerKit and PoseKit behavior is consistent where the feature exists.

## Common Anti-Patterns (Do Not Add)

- Hidden runtime remapping (fast tier requested, a slower backend used
  without `used_fallback`).
- New runtime strings or feature-specific runtime dropdowns.
- Manual exported-model-path requirements for standard workflows.
- Backend checks scattered across GUI/business logic instead of resolver usage.

## Sliced Inference (SAHI)

Direct-mode OBB detection supports optional SAHI-style sliced inference
(`SliceConfig`, off by default). `run_obb` dispatches to
`stages/slicing.py:run_direct_sliced` when `config.obb.direct.slice.enabled`.

- **Geometry:** `auto_model` (tile = model imgsz, no resample — the fast path),
  `auto_object` (tile = body px / object scale), `custom` (explicit size).
- **Merge:** `merge_policy` (`nms`/`greedy_nmm`) × `merge_metric` (iou/ios) ×
  `merge_backend` (cv2 default; gpu = native-cuda only, cv2-validated). Default
  `greedy_nmm` + `ios` + `0.5`. A saved `nmm` runs the same code path as
  `greedy_nmm`: `canonicalize` reads it as `greedy_nmm` and no UI offers it,
  but TrackerKit and the engine pass the raw value through so the cache hash
  of old configs still hits.
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

### The SAHI contract (`TilingSpec`, aliases, resolution)

Every SAHI surface (TrackerKit inference, DetectKit YOLO training, preview and
inference dialog, SAM3 training, SAM3/SAM2 escalation) shares one vocabulary
and one set of rules. The user-facing description is
[SAHI Settings](../user-guide/sahi-settings.md). The pieces:

- **`utils/tiling_spec.py`** (pure, numpy only, importable from every layer):
  - `TilingSpec`, a frozen dataclass in TrackerKit's vocabulary: `enabled`,
    `geometry_mode`, `object_tile_fractions`, `reference_body_px`,
    `slice_width`/`slice_height`, `overlap` (`None` = unset),
    `min_area_ratio`, `fragment_policy` (`drop`/`crowd`/`mask`),
    `merge_policy`/`merge_metric`/`merge_threshold`. The constructor is
    **write-strict**: a wrong type or an out-of-range value raises
    `ValueError`. `TilingSpec.from_mapping(...)` is the **read-lenient**
    path: legacy or out-of-range values in old files load with a one-time
    warning and never raise. `TilingSpec.defaults(backend)` reads the
    table below. `training_tile_sizes()` fans out every scale and is for
    TRAINING only. Inference uses one operating scale.
  - `BACKEND_DEFAULTS` is the one per-backend defaults table (`yolo_train`
    `(0.05, 0.10, 0.15, 0.20)`, `yolo_infer` `(0.15,)`, `sam3` `(0.055,)`,
    `sam2` `()` = full frame until calibrated), plus `DEFAULT_OVERLAP`
    (0.2), `OVERLAP_MARGIN` (0.05), `OVERLAP_MAX` (0.9) and
    `FRACTION_MIN`/`FRACTION_MAX` (0.01/0.9). Widget ranges come from
    these constants. The SAM3 *escalation* seed
    (`semantic.tiling.SEMANTIC_TILE_FRACTION_SEED = 0.05`, on the
    calibration grid) is deliberately separate from the `sam3` *training*
    default.
  - `SLICE_ALIASES` plus `canonicalize(mapping)` is the **only** place
    legacy names are translated (for example `slice_overlap`/`tile_overlap`
    → `overlap`, `min_retained_area_frac` → `min_area_ratio`, `merge_iou` →
    `merge_threshold`, `trained_body_px` → `reference_body_px`, and
    `target_size_fractions`/`tile_fraction` → fractions). Pixel
    `target_sizes` are divided by their stated `imgsz` (640 for legacy YOLO
    only) and are never written. Writers emit canonical names.
  - `operating_fraction(fractions)` is the one operating-scale rule: the
    `np.median` of the set.
- **`utils/tiling_resolve.py`** holds the `(value, source)` resolvers
  (re-exported by `tiling_spec`). Each returns a `Sourced` whose `source` is
  the UI badge: `resolve_reference_body_px` (override → dataset → stamped →
  TrackerKit's own), `resolve_operating_fraction` (profile → stamped →
  default), `resolve_object_tile_fractions` (user → profile → stamped →
  default), `resolve_overlap` (override → saved → derived `max(fraction) +
  OVERLAP_MARGIN` → default), and `resolve_tile_size`.
- **Unchanged byte-for-byte:** `SLICE_*` engine keys, `SliceConfig` fields,
  `_slice_config_hash`, and the `get_parameters_dict` goldens. No `SLICE_*`
  default moved; the defaults are now named constants with the same values.

### The model sidecar (v3) and `read_tiling_meta`

- The file is `<model>.<ext>.slice_meta.json` (`slice_meta.sidecar_path`), now
  `schema_version: 3` with `model_family` (`"yolo"` | `"sam3"`), a
  `training_geometry` block, `primary_profile_id` and `profiles`.
- **YOLO v3 is additive** (`tiling_meta.training_geometry_from_yolo_manifest`):
  every v2 manifest key is kept verbatim (including `target_sizes` and the
  scalar `object_tile_fraction`), with canonical `object_tile_fractions`,
  `trained_body_px` and `fragment_policy` added beside them. The v2 reader
  (`slice_meta._training_values`), the drift guard and the calibration grid
  therefore read a v3 file bit-for-bit as they read v2.
- **SAM3 dual-writes**: publish keeps the geometry in
  `<artifact>.sam3_meta.json`, which existing readers use, and also writes a
  canonical-only `.slice_meta.json` v3
  (`training_geometry_from_sam3_manifest`). Builders stamp only values the
  build stated, never defaults. Removing geometry from `.sam3_meta.json` is
  deferred.
- `core/inference/tiling_meta.read_tiling_meta(model_path) -> TilingMeta |
  None` is the one normalizing reader for v1/v2/v3 and legacy
  `.sam3_meta.json` geometry. It returns `model_family`, `source`
  (`slice_meta`/`sam3_meta`), `training: TilingSpec`, `imgsz`, `tile_px_set`,
  `operating_fraction` and `profiles`, and it never raises. **It has no
  production caller yet**, only its round-trip tests. TrackerKit and the DetectKit
  preview still resolve through `slice_meta.slice_meta_to_panel_values` /
  `resolve_slice_profile_values`, TrackerKit's ladder: requested profile →
  primary profile → training geometry, or `slice_meta_values_from_settings`
  for a saved session. That ladder already reads v3 additively, so moving
  TrackerKit onto `read_tiling_meta` was deferred to protect byte-identity.
- The DetectKit preview resolves in `detectkit/jobs/preview_tiling.py`
  (`resolve_preview_tiling`). An open inference-settings override wins.
  Otherwise geometry, scale, overlap, tile size and merge come from the
  sidecar through the TrackerKit ladder, while the project keeps `enabled`,
  imgsz and the body (label-measured first, then stamped). The sidecar parse
  is cached per (path, mtime, size, inode), so a recalibration is picked up.
- SAM3 escalation's opening state and headless `detectkit escalate sam3`
  share `detectkit/jobs/semantic_escalation.default_semantic_tiling(project,
  variant)`: saved → calibration → stamp → seed, and full frame only when the
  body is 0. This mirrors SAM2's `sam2_escalation.default_geometry_tiling`.

### The shared settings widget

`hydra_suite/widgets/slice_settings.py` (`SliceSettingsWidget`, with the
control/row tables in `slice_settings_controls.py` and vocabulary/ranges in
`slice_settings_parts.py`) is the one SAHI UI. It imports only Qt and `utils`.
`tests/test_widgets_no_app_imports.py` enforces that it imports no app layer.

- **Roles** (`ROLES`): `infer_yolo` (TrackerKit detection panel, DetectKit
  inference dialog), `train_yolo` (DetectKit training dialog), `train_sam3`
  (SAM3 training panel), `escalate_sam3`, `escalate_sam2`. The role selects
  rows and the defaults row (`ROLE_BACKEND`).
- **Capabilities** (`SliceWidgetCapabilities`) cover host differences within
  a role: `body_override`, `body_display_only` (TrackerKit),
  `advanced_merge`, `merge_threshold_row`, `full_frame_pass`,
  `execution_knobs` (TrackerKit tiles per call / memory budget),
  `fixed_overlap` (SAM2), and `tile_label_formatter` (escalation wording).
- **State** is a `TilingSpec` plus a per-role `extras` dict (`ROLE_EXTRAS`):
  `set_spec(spec, extras=...)`, `spec()` and `extras()`. Geometry combo
  items show the label and store the enum as item data. Readers must use
  `currentData()`/`findData()`, never the text.
- **Ownership rule:** hosts own persistence. The widget writes its own
  controls ONLY inside `set_spec(...)` (signals blocked) and
  `set_reference_body(value, source)`. Derived values (tile size outside
  Custom, the resolved tile, the whole-animal overlap minimum) appear in
  LABELS and are never written into a host-persisted spin. A user edit
  emits `field_changed(field)`. Hosts keep their own per-spin wiring, since
  the spins are the widget's. TrackerKit never calls `set_spec`: it binds
  its old attribute names to the widget's controls and keeps its existing
  `setValue` + `_applying_slice_profile` logic.
- **F7 is a hint:** the whole-animal minimum (`max(scale) + OVERLAP_MARGIN`)
  is shown with a "Raise to X" button only when the user's own overlap is
  below it. It is info-only for profile, stamped or config overlaps, and it
  is never applied automatically. No host derives a saved overlap.
- `set_source(field, source)` / `source_badge(field)` carry the badge
  vocabulary in `SOURCE_DESCRIPTIONS` (`user`, `override`, `profile`,
  `stamped`, `project`, `dataset`, `config`, `derived`, `default`).
- DetectKit adapters (`detectkit/gui/panels/slice_settings_adapter.py`) map
  `SliceTrainingSettings` and the nine SAM3 tiling kwargs to and from
  `(TilingSpec, extras)`. `SliceSettingsGroup` keeps legacy attribute
  aliases for its tests.
- `tools/sahi_widget_gallery.py` renders every role and host offscreen, for
  visual checks after a layout change.

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
   `detectkit/jobs/sam2_escalation.py`) and recorded on `OBBSource.staged_review`
   (`StagedReview`: `staged_path`, `target_level`, `sam2_variant`, `created_at` — the field was renamed
   from `pending_escalation`/`PendingEscalation` because it now holds the staged output of any producer,
   not just SAM2/SAM3 escalations; old project files with the old key/names still load, see
   `docs/superpowers/specs/done/2026-08-31-detectkit-frame-granular-review-design.md`). The source's own
   canonical `labels/`/`classes.txt` are never touched during staging, and no new source is registered
   — the source list still shows exactly one entry for that source. Re-running escalation over a source
   that already has a staged review is skipped by default (recorded in `EscalationResult.skipped`);
   passing `overwrite=True` replaces the staged directory in place.
6. The user reviews the staged result **frame by frame**, via the review bar shown above the canvas
   whenever the current source has a staged review (`detectkit/gui/panels/review_bar.py`), backed by
   `detectkit/jobs/staged_review.py`. Per frame: **Replace** (`accept_frame(..., mode=MergeMode.OVERWRITE)`)
   overwrites the frame's label with the staged one; **Add New**
   (`accept_frame(..., mode=MergeMode.ADD_NEW)`) keeps the frame's existing labels and appends only
   non-overlapping staged instances via `merge_records`; **Reject** discards the staged frame. **Accept
   All**/**Reject All** apply the same over every undecided frame; **Next Undecided** jumps between them;
   a `decided`/`total` counter tracks progress. Every accept applies **immediately** to the source's
   canonical `labels/`/`classes.txt` — there is no separate "commit" step — and the first accept in a
   review snapshots the source's pre-review state so **Revert Review** can restore it (available only
   while the review is open; finishing the review deletes the staging directory and the snapshot with
   it). If the staged geometry level is above the source's, the accept promotes the source (existing
   labels lifted, never re-derived down); see the design spec's §3 for the promotion/lift distinction.
   Finishing a review (every frame decided) removes the staging directory and clears `staged_review`.
7. This flow is driven from the GUI by the "Review escalations…" button in the Tools panel
   (`ToolsPanel.review_escalations_requested` → `MainWindow._on_go_to_staged_review`), which switches to
   the first source with an unfinished `staged_review` and shows its review bar — there is no longer a
   separate review dialog (`ReviewEscalationsDialog` was retired; re-thresholding a staged run, previously
   a dialog action, is now a review-bar button). A source with an unresolved staged review is still found
   this way after the project is closed and reopened, since the state lives on `OBBSource.staged_review`,
   not in transient UI.

**Key properties:**

- Runs in a background `QThread` (`Sam2EscalationWorker`, a `BaseWorker`), keeping the GUI responsive
  during a run; progress/status are reported via the standard `BaseWorker` signals.
- Model weights are auto-managed (variant catalog via `checkpoints.py`); weights download on first use.
- No caching, no ONNX/TensorRT export (SAM2 is torch-only).
- No per-tier gating. The device comes from the dialog's **Run on** picker (or `detectkit escalate sam2 --device`), resolved by `core.inference.torch_device.resolve_torch_device`: Auto picks CUDA, then MPS, then CPU, and an unavailable saved choice falls through to the best available device.
- `sam2` is imported lazily, only inside `Sam2SegmentExecutor.from_variant` — importing the executor
  module (or `sam2_escalation.py`) does not require `sam2` to be installed.

## Related Docs

- [GPU Backends](gpu-backends.md)
- [Extending Detection](extending-detection.md)
- [Extending Identity](extending-identity.md)
- [Compute Runtimes (User Guide)](../user-guide/compute-runtimes.md)
