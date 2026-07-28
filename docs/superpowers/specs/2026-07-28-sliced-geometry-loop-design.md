# Close the Sliced-Geometry Loop — Design

**Date:** 2026-07-28
**Status:** Design approved; pending spec review → implementation plan
**Depends on:** the merged DetectKit SAHI sliced training + inference feature
(`utils/slice_geometry.py`, `training/sliced_dataset.py`, `SliceTrainingSettings`,
`predict_sliced_obb_result`, `model_publish.py` slice-geometry stamping,
`service._slice_geometry_for_publish`). Geometry-levels already on main.

## 1. Motivation

The sliced-training feature shipped with three open follow-ups that together
prevent the trained slice geometry from flowing all the way to inference:

1. **DetectKit preview tiles at an untrained scale.** `predict_sliced_obb_result`
   derives its `auto_object` tile from the stored `object_tile_fraction`, ignoring
   the builder's `target_sizes` fan-out — so a "validate under slicing" preview can
   tile at a scale the model never trained at.
2. **`reference_body_px` is never measured into the record.** The builder measures
   it per frame internally, but `SliceTrainingSettings.reference_body_px` stays 0.0
   and the dataset manifest's `slice_geometry.reference_body_px` records the *input*
   (0.0 for the measured case). So the model sidecar carries `0.0`, and `auto_object`
   preview degrades to `imgsz` tiling.
3. **TrackerKit ignores the model sidecar.** `publish_trained_model` writes
   `<model>.slice_meta.json`, but TrackerKit's SAHI inference never reads it, so the
   user must re-enter the trained geometry by hand (and has no measured reference to
   enter).

These form a chain: the build must **measure and stamp** a real `reference_body_px`
(#2) so TrackerKit can **read it back** (#3), while the DetectKit preview shares the
same trained geometry (#1). This spec closes that loop.

**Non-goal:** authoritative/forced inference-time override. TrackerKit *pre-fills*
the SAHI panel from the sidecar; the user always sees and can edit the values.

## 2. Data flow

```
DetectKit build  →  sliced dataset manifest.json  →  model .slice_meta.json  →  TrackerKit SAHI
   measures ref        slice_geometry{ref,...}       (via existing publish)      pre-fills panel
        │                                                                            ▲
        └─→ SliceTrainingSettings.reference_body_px ──→ DetectKit preview (median target)
```

The single knob that keeps DetectKit preview and TrackerKit pre-fill consistent by
construction: **`object_tile_fraction = median(target_sizes) / imgsz`** is used in
BOTH the preview (Component B) and the TrackerKit mapping (Component C).

## 3. Component A — DetectKit: measure + stamp `reference_body_px` (#2)

`training/sliced_dataset.py`:
- `build_sliced_obb_dataset` accumulates every object's `cv2.minAreaRect` major axis
  (in frame pixels) across the whole dataset and computes the **dataset-level median**
  = `measured_reference_body_px`. Reuse the existing per-object measurement logic
  (the same math as `measure_reference_body_px`), aggregated over all objects rather
  than per frame.
- The manifest's `slice_geometry.reference_body_px` records the **actual** reference
  used: `params.reference_body_px if params.reference_body_px > 0 else
  measured_reference_body_px`. This is the value that travels to the model sidecar
  (unchanged publish path). Add `measured_reference_body_px` to the returned
  `DatasetBuildResult.stats` so the caller can read it.

`detectkit/gui/dialogs/training_dialog.py` (`_build_role_datasets`):
- After the sliced build, if `project.slice_settings.reference_body_px == 0.0`, set it
  to the measured median from the build stats and `save_project(...)`. A non-zero
  (user-set) value is **never** overwritten. Only runs when slicing is enabled (the
  build already gates on `slice_settings.enabled`).

## 4. Component B — DetectKit preview: median trained target (#1)

`detectkit/gui/main_window.py` (the preview routing that calls
`predict_sliced_obb_result`):
- Compute `object_tile_fraction = median(slice_settings.target_sizes) / imgsz` where
  `imgsz = project.imgsz_obb_direct`, falling back to `slice_settings.object_tile_fraction`
  when `target_sizes` is empty. Pass the (now-populated) `slice_settings.reference_body_px`.
- Extract the median computation into a small pure helper so it is unit-testable
  without Qt: `preview_object_tile_fraction(target_sizes, object_tile_fraction,
  imgsz) -> float` in `detectkit/gui/prediction_preview.py` (already Qt-free — imports
  cv2/math, not Qt).
- Everything else in the routing is unchanged; the disabled-slicing branch is untouched.

## 5. Component C — TrackerKit: read the sidecar, pre-fill the SAHI panel (#3)

**Key distinction (from review):** TrackerKit's `REFERENCE_BODY_SIZE` is the ant's size
in the **full tracking frame** (the user's tracking-scale value, `spin_reference_body_size`,
range 1–500) and inference derives `reference_body_px = REFERENCE_BODY_SIZE × RESIZE_FACTOR`.
The sidecar's `reference_body_px` is a **different, model-internal quantity** — the object
size measured in *training-image* pixels. These must not be conflated. So this design
introduces a **new, separately-named "trained body size" input** pre-filled from the
sidecar, and NEVER touches `REFERENCE_BODY_SIZE`.

**New param + override:** add `SLICE_TRAINED_BODY_PX` (advanced-config key
`slice_trained_body_px`, default `0.0`). In
`core/inference/config.py:build_inference_config_from_params`, when
`SLICE_TRAINED_BODY_PX > 0` it becomes `SliceConfig.reference_body_px` (the model-internal
trained scale); otherwise the existing `REFERENCE_BODY_SIZE × RESIZE_FACTOR` product is used
unchanged. This keeps the tracking reference and the trained reference as distinct sources,
with the trained value winning for a sliced model that carries one.

Two pure functions (unit-testable, no Qt) in a new `core/inference/slice_meta.py`
module (mirroring the `<artifact>.runtime_meta.json` sidecar convention in
`core/inference/runtime_artifacts.py`; kept in `core/inference` so both DetectKit
and TrackerKit could reuse them and so they stay Qt-free):

- `read_slice_meta(model_path) -> dict | None` — read `<model_path>.slice_meta.json`;
  return the parsed dict, or `None` on absent file / bad JSON / wrong shape. Never raises.
- `slice_meta_to_panel_values(meta) -> dict` — translate the sidecar into panel/config
  values: `enabled=True`; `geometry_mode ← meta["geometry_mode"]`;
  `overlap ← meta["overlap"]`; `trained_body_px ← meta["reference_body_px"]`;
  `object_tile_fraction ← median(meta["target_sizes"]) / meta["imgsz"]` (same median
  convention as Component B; fall back to `meta["object_tile_fraction"]` when
  `target_sizes` is empty/absent, and to `meta["object_tile_fraction"]` when `imgsz`
  is absent/zero). Missing keys fall back to current defaults.

TrackerKit detection-panel wiring:
- When the user selects an OBB model in the detection panel (the `combo_yolo_model`
  selection path), call `read_slice_meta` on the chosen model path. If it returns a dict,
  apply `slice_meta_to_panel_values`: set the SAHI **widgets** `chk_slice_enabled`=True and
  `combo_slice_geometry`=geometry_mode, and write the **advanced-config** keys
  `slice_overlap`, `slice_object_tile_fraction`, and `slice_trained_body_px`. Show a
  dismissible **"Matched trained SAHI geometry"** banner. **All values stay editable**
  (widgets directly; advanced-config via its existing editor); nothing is forced at
  config-build or inference time — `get_parameters_dict()` reads whatever the panel +
  advanced-config hold.
- The pre-fill transfers only **scale-independent** trained knobs plus the explicitly
  model-internal `slice_trained_body_px`; `REFERENCE_BODY_SIZE` is left as the user's
  tracking-scale value. Document at the pre-fill site that `slice_trained_body_px` is the
  training-image body scale and the user may adjust it if their tracking frames differ in
  resolution from the training frames.

## 6. Error handling

- Absent / malformed sidecar → `read_slice_meta` returns `None` → TrackerKit panel
  behavior byte-identical to today (no banner, no pre-fill).
- Empty `target_sizes` in the sidecar or settings → fall back to the stored
  `object_tile_fraction` (never divide by an empty median).
- `imgsz` absent/zero in the sidecar → fall back to `object_tile_fraction` (guard the
  division).
- User-set `reference_body_px` in DetectKit settings → build must not overwrite it.
- `SLICE_TRAINED_BODY_PX` absent/0 in TrackerKit params → `reference_body_px` computation
  is byte-identical to today (`REFERENCE_BODY_SIZE × RESIZE_FACTOR`).

## 7. Testing

Pure/unit (synthetic data, no ants, no models):
- dataset-level median measurement over multiple frames/objects;
- manifest records the **measured** median when `params.reference_body_px == 0`, and the
  **explicit** value when set;
- `preview_object_tile_fraction`: `median(target)/imgsz`, and the empty-`target_sizes`
  fallback;
- `read_slice_meta`: present dict / absent file / malformed JSON → `None`;
- `slice_meta_to_panel_values`: full mapping incl. median-target → fraction, `trained_body_px`
  from the sidecar reference, and each fallback (empty target_sizes, missing imgsz, missing keys);
- `build_inference_config_from_params`: `SLICE_TRAINED_BODY_PX > 0` overrides
  `SliceConfig.reference_body_px`; `0`/absent → `REFERENCE_BODY_SIZE × RESIZE_FACTOR` unchanged.

Guard / regression:
- No sidecar → TrackerKit panel behavior byte-identical (pre-fill is a no-op).
- `SLICE_TRAINED_BODY_PX` absent/0 → `reference_body_px` byte-identical to today.
- DetectKit build with a user-set `reference_body_px` leaves the settings value unchanged.
- `predict_sliced_obb_result` / non-sliced preview unaffected when slicing off.

Qt (headless, `pytest.importorskip("PySide6")`):
- selecting a model whose path has a `.slice_meta.json` sets `chk_slice_enabled`=True,
  `combo_slice_geometry`=trained mode, writes `slice_overlap` / `slice_object_tile_fraction`
  / `slice_trained_body_px` into advanced-config, raises the banner, and leaves
  `REFERENCE_BODY_SIZE` (`spin_reference_body_size`) untouched.

## 8. File structure

**Modified**
- `src/hydra_suite/training/sliced_dataset.py` — dataset-median measurement + manifest/stats.
- `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` — populate settings after build.
- `src/hydra_suite/detectkit/gui/main_window.py` — preview median-target fraction.
- `src/hydra_suite/detectkit/gui/prediction_preview.py` — the pure
  `preview_object_tile_fraction` helper (already Qt-free).
- `src/hydra_suite/core/inference/config.py` — `SLICE_TRAINED_BODY_PX` overrides
  `SliceConfig.reference_body_px` in `build_inference_config_from_params`.
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py` — emit `SLICE_TRAINED_BODY_PX`
  from advanced-config `slice_trained_body_px`.
- TrackerKit detection panel (`detection_panel.py`) + its model-selection wiring —
  sidecar pre-fill (widgets + advanced-config) + banner.

**New**
- `src/hydra_suite/core/inference/slice_meta.py` — pure `read_slice_meta` +
  `slice_meta_to_panel_values` (Qt-free, beside the existing sidecar/runtime-artifact code).
- Tests: `tests/test_sliced_dataset_reference.py`, `tests/test_detectkit_preview_target.py`,
  `tests/test_slice_meta_read.py`, `tests/test_trackerkit_slice_meta_prefill.py`.

## 9. Acceptance

1. `build_sliced_obb_dataset` measures a dataset-level median `reference_body_px` and
   stamps the actual reference (measured or explicit) into the manifest → model sidecar.
2. DetectKit populates `SliceTrainingSettings.reference_body_px` from the measured median
   only when unset; a user value is preserved.
3. DetectKit `auto_object` preview tiles at `median(target_sizes)/imgsz`.
4. `read_slice_meta` + `slice_meta_to_panel_values` are pure, tested, and tolerant of
   absent/malformed sidecars; the mapper emits `trained_body_px` from the sidecar reference.
5. `SLICE_TRAINED_BODY_PX > 0` overrides `SliceConfig.reference_body_px`; `0`/absent leaves
   the `REFERENCE_BODY_SIZE × RESIZE_FACTOR` computation byte-identical to today.
6. Selecting an OBB model with a sidecar pre-fills the TrackerKit SAHI panel — enable +
   geometry mode + advanced-config overlap/object_tile_fraction/slice_trained_body_px
   (editable) + banner — and leaves `REFERENCE_BODY_SIZE` untouched; no sidecar →
   byte-identical to today.
7. The trained-vs-tracking scale distinction is documented at the pre-fill site.
