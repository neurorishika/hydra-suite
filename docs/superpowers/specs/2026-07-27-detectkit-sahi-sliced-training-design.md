# DetectKit SAHI Sliced Training + Inference — Design

**Date:** 2026-07-27
**Status:** Design approved; pending spec review → implementation plan
**Depends on:** `docs/superpowers/plans/2026-07-24-detectkit-geometry-levels.md` **merged first.**
This spec builds on the polygon-first label model, the level-aware dataset
builders, and `OBBResult.polygons` / `emit_native_geometry` that piece introduces.
**Also builds on** the merged SAHI sliced inference (`core/inference/stages/slicing.py`,
`SliceConfig`, `merge_obb_detections`, `band_membership`).

## 1. Motivation

SAHI sliced inference is live in TrackerKit, but direct-mode OBB models trained
by DetectKit fail under it. Measured on a 4512×4512 video (`imgsz=1024`,
`auto_model` → 36 native-resolution tiles): full-frame inference lands ants at
~127px apparent and yields 11 clean detections; SAHI feeds native tiles where
ants are ~560px and yields ~2 spurious fragments. A scale sweep confirms the
model's detection peaks at ~120–200px apparent and collapses outside it — an
artifact of training on full frames downscaled to `imgsz`, where objects only
ever appear small.

The driver is **crowding**: clustered ants merge at the full-frame downscale, so
slicing (which gives each tile more pixels per ant and splits the crowd across
tiles) is the right tool — but only if the model can detect at the sliced scale.
It can't, because it never trained there.

**Goal:** make DetectKit produce direct OBB models that are *usable under SAHI*,
by (a) generating training data at the sliced scale, (b) training size-robust so
one model spans full-frame and sliced geometries, (c) letting DetectKit preview
sliced predictions for labeling/validation, and (d) recording the training
geometry in the model manifest so inference can be matched.

**Non-goal (this spec):** TrackerKit reading the manifest to auto-configure
`SliceConfig`. This spec *writes* the metadata; a small follow-up spec teaches
TrackerKit to *read* it.

## 2. The scale mechanics this design is built on

`auto_object` tile sizing (verified in `_tile_size`): `tile_size =
reference_body_px / object_tile_fraction`. After a tile is letterboxed to
`imgsz`, an object's apparent size in the model input is:

```
apparent_size = object_tile_fraction × imgsz
reference_body_px = object size at FULL/native resolution
tile_size = reference_body_px × imgsz / apparent_size
```

The **apparent (target) size is the central knob**: it sets both how small the
tiles are (smaller target → smaller tiles → more crowd-splitting) *and* the scale
the model must learn. Pinning the target at the current model's ~150px peak
forces tiles ≈ the whole frame (no splitting); the fix is to *retrain* at a
larger target (e.g. ~300–400px), where tiles are ~1400–1900px on a 4512 frame
(real crowd-splitting) and the model learns to detect at that scale.

Because YOLO resizes every training image to `imgsz`, a tile of size
`reference_body_px / (target/imgsz)` presents its objects at `target` apparent
size after resize — identical to inference. **Training scale == inference scale
by construction**, provided both tile with the same geometry.

## 3. Architecture — shared tile geometry (Approach B)

The pure tile planner is extracted so inference, the training builder, and the
preview all consume ONE implementation — the only structural guarantee that
train-tiles == inference-tiles.

**New `utils/slice_geometry.py`:** move `SlicePlan`, `get_slice_bboxes`,
`tiles_overlap`, `_tile_size`, `plan_slices`, and `MAX_TILES_PER_FRAME` here.
These take plain geometry params (tile size / overlap / geometry-mode /
ref-object-px / target) — NOT `SliceConfig` — so `utils/` stays free of any
`core.inference` dependency (it sits beside `rotated_iou.py` and
`obb_from_mask.py`). Pure geometry, no Qt, no I/O.

**`core/inference/stages/slicing.py`:** keeps `SliceConfig`, builds the plain
params from it, and re-imports the planner from utils. Its inference
orchestration (`run_direct_sliced`, merge, executor calls) is otherwise
unchanged. A **byte-parity test** asserts the inference path emits identical tile
boxes before/after the extraction (same discipline as the disabled-slicing gate).

**Consumers:**
- `training/sliced_dataset.py` (new) — the dataset builder.
- `detectkit/gui/prediction_preview.py` (extended) — the preview slicer.

Both import `utils.slice_geometry` + reuse `merge_obb_detections`,
`band_membership`, `_offset_result` from `core.inference` for the inference-side
merge.

## 4. Sliced training-data builder (`training/sliced_dataset.py`)

A tiling front-end to the (post-geometry-levels) dataset builders — NOT a
parallel builder. Per labeled source frame:

1. `plan_slices(frame_hw, geom_params)` → tile boxes (the exact call inference
   makes).
2. Per tile: crop the image; for each labeled object, **clip its polygon to the
   tile rect** (Sutherland–Hodgman / `cv2`), compute `area_in_tile / area_full`,
   keep the clipped label iff `≥ min_area_ratio` (default 0.1, configurable),
   remap coords into tile space, and **re-derive the geometry level**
   (polygon→obb→aabb) via the geometry-levels downward-derivation so the tile
   label matches the source's level. Tile overlap guarantees a boundary object
   survives whole in the neighbour.
3. **Negative tiles** (zero kept labels) are sampled at a configurable fraction
   (default keep ~15%) so the model learns true background without dataset bloat.
4. **Full-frame mix** (default ON, ratio configurable): also emit the original
   full frames + labels, so the model spans both scales.
5. Write via the existing level-aware YOLO writer.

### 4a. Multi-scale / size-robust emission

The builder does not pin one target. It emits sliced tiles at each of a
**configurable list of target apparent sizes** (default a small set spanning the
robust range, e.g. three values), plus the full frames from §4-step-4. This flattens the detection-vs-scale curve — the
model becomes robust across the apparent-size range that full-frame (~127px),
`auto_model` (native), and `auto_object` (chosen target) all produce, rather than
just moving the narrow peak. A size-robust model largely dissolves the
train/inference mode-matching problem: objects land in the model's competent
range regardless of the geometry chosen at inference.

Honest bound: a fixed `imgsz` input cannot cover arbitrarily small AND large
objects (too small vanishes in downscale; too large does not fit). "Size-robust"
means a wide but bounded range (order several-fold), not literal invariance.

### 4b. `auto_object` reference size — measured from labels

`auto_object` needs `reference_body_px` (full-res object scale). TrackerKit gets
it from `REFERENCE_BODY_SIZE × RESIZE_FACTOR` (a tracking param DetectKit lacks).
DetectKit's advantage: it has the labels. The builder **measures
`reference_body_px` as the median OBB major axis** of the labeled objects (reuse
`training/dataset_inspector.py`'s object-size stats), which is more accurate than
a hand-set reference. The UI knob is **target apparent size**; the builder derives
`object_tile_fraction = target / imgsz` and the tile size from the measured
reference.

### 4c. Geometry modes

The builder supports all three inference geometries so training matches whichever
the collaborator runs: `auto_model` (native tiles), `custom` (explicit size),
`auto_object` (target-driven, §4b). Selected via the shared settings block (§6).

### 4d. Manifest (metadata written here)

Alongside the dataset, write a manifest recording the resolved geometry:
`reference_body_px`, the target-size range trained, `imgsz`, geometry mode,
overlap, `min_area_ratio`, full-frame-mix ratio. This is stamped through
`model_publish.py` into the model artifact's TrackerKit-readable manifest (the
existing sidecar pattern — cf. the classifier `.v2meta.json` and
`runtime_artifacts` `.runtime_meta.json`). TrackerKit *reading* this to
auto-configure `SliceConfig` is the follow-up spec; here we only ensure it is
written and travels with the model.

## 5. Sliced inference in DetectKit preview / active-learning

DetectKit predicts at the executor level (`prediction_preview.py:_predict_direct`
→ `executor.predict(...)`), NOT via `InferenceRunner`. Add a thin wrapper there:

```
predict_sliced(image, executor, geom_params, merge_params):
    plan   = plan_slices(image.shape, geom_params)     # utils.slice_geometry
    tiles  = [image[y0:y1, x0:x1] for box in plan.tiles]
    raw    = executor.predict(tiles, ...)              # existing batched call
    dets   = [_offset_result(r, x0, y0) ...]           # into frame coords
    merged = merge_obb_detections(concat, band_membership(...), backend="cv2")
```

Reuses the shipped helpers; cv2 merge only (preview is CPU/numpy). A toggle in
the preview/AL surface switches full-frame vs sliced, reading the **same geometry
params** as the builder. This is the labeler's loop: import crowded frames →
sliced predictions as suggestions → active-learning on the clusters → export →
retrain. It does not fix the model on its own; it is how the collaborator *sees*
whether a retrained model separates clusters. When slicing is off, this path must
reduce exactly to today's full-frame preview (parity test).

## 6. UI / config surface

One shared `SliceSettings` block — tile geometry mode, target apparent size (or
explicit/custom size), overlap, `min_area_ratio`, negative-tile fraction,
full-frame-mix ratio, multi-scale target range, merge threshold — surfaced in
BOTH:
- **Dataset/training dialog** — a "Sliced dataset" group feeding the builder.
- **Preview/AL panel** — an "Enable sliced inference" toggle reading the same
  block.

Persisted in the DetectKit project JSON (like geometry-levels' `OBBSource.level`)
so settings travel with the project to the collaborator. Sharing one block keeps
training geometry and preview geometry coupled by construction.

## 7. Testing

The real "does the retrained model detect crowded ants" outcome cannot be
validated locally (dataset is with a collaborator). Confidence is built where it
can be:

**Unit tests on synthetic labeled frames (deterministic, no ants needed):**
- boundary clipping yields the correct clipped polygon;
- `area_in_tile/area_full` thresholding keeps/drops correctly at the boundary;
- a straddling object appears whole in the overlapping neighbour tile;
- label coords remap into tile space exactly;
- negative-tile sampling hits the configured fraction;
- multi-scale emission produces tiles at each target in the range + full frames;
- `auto_object` derives `reference_body_px` = median object major axis from labels;
- level re-derivation matches the source level (polygon→obb→aabb).

**Regression / parity gates:**
- `utils/slice_geometry.py` emits byte-identical tile boxes to the pre-extraction
  inference path (guards Approach B);
- the preview slicer reduces to today's full-frame preview when slicing is off.

**Collaborator runbook (ships with the feature):** generate sliced dataset →
train → re-run the scale sweep → confirm the detection-vs-scale curve flattened
and clusters separate. We ship the machinery proven correct on synthetic data;
the collaborator proves the ML outcome on the real data.

## 8. File structure

**New**
- `src/hydra_suite/utils/slice_geometry.py` — extracted pure tile planner.
- `src/hydra_suite/training/sliced_dataset.py` — the tiling dataset builder.
- Tests: `tests/test_slice_geometry.py`, `tests/test_sliced_dataset.py`,
  `tests/test_detectkit_sliced_preview.py`, `tests/test_slice_geometry_parity.py`.

**Modified**
- `src/hydra_suite/core/inference/stages/slicing.py` — re-import planner from utils.
- `src/hydra_suite/training/model_publish.py` — stamp slice geometry into the manifest.
- `src/hydra_suite/detectkit/gui/prediction_preview.py` — `predict_sliced` wrapper.
- DetectKit dataset/training dialog + preview panel — the shared `SliceSettings` block.
- DetectKit project JSON schema — persist `SliceSettings`.

## 9. Acceptance

1. `utils/slice_geometry.py` extraction is byte-parity-verified against the
   inference tile boxes.
2. The builder tiles labeled data with `plan_slices`, clips + area-thresholds
   boundary labels, remaps into tile space, re-derives level, samples negatives,
   mixes full frames, and emits multi-scale targets — all covered by synthetic
   unit tests.
3. `auto_object` derives `reference_body_px` from the labels; target apparent size
   is the UI knob.
4. DetectKit preview/AL runs sliced predictions via the executor-level wrapper and
   reduces to full-frame parity when off.
5. Slice geometry is written into the model manifest (TrackerKit-readable),
   carrying the training geometry for the follow-up auto-config spec.
6. A collaborator runbook is included.
7. An existing non-sliced DetectKit project is unaffected (defaults off).
