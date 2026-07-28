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

Two pure functions (unit-testable, no Qt) in a new `core/inference/slice_meta.py`
module (mirroring the `<artifact>.runtime_meta.json` sidecar convention in
`core/inference/runtime_artifacts.py`; kept in `core/inference` so both DetectKit
and TrackerKit could reuse them and so they stay Qt-free):

- `read_slice_meta(model_path) -> dict | None` — read `<model_path>.slice_meta.json`;
  return the parsed dict, or `None` on absent file / bad JSON / wrong shape. Never raises.
- `slice_meta_to_panel_values(meta) -> dict` — translate the sidecar into SAHI panel
  values: `enabled=True`; `geometry_mode ← meta["geometry_mode"]`;
  `overlap ← meta["overlap"]`; `reference_body_px ← meta["reference_body_px"]`;
  `object_tile_fraction ← median(meta["target_sizes"]) / meta["imgsz"]` (same median
  convention as Component B; fall back to `meta["object_tile_fraction"]` when
  `target_sizes` is empty/absent). Missing keys fall back to current defaults.

TrackerKit detection-panel wiring:
- When the user selects an OBB model in the detection panel, call `read_slice_meta` on
  the chosen model path. If it returns a dict, apply `slice_meta_to_panel_values` to the
  SAHI widgets (enable checkbox, geometry-mode combo, reference-body / overlap /
  object-tile-fraction fields) and show a dismissible **"Matched trained SAHI geometry"**
  banner. **All fields stay editable**; nothing is forced at config-build or inference
  time — the existing `get_parameters_dict()` path reads whatever the widgets hold.
- Document at this site the **scale caveat**: the sidecar's `reference_body_px` is in
  training-image pixels; if the tracking video differs in resolution / `RESIZE_FACTOR`
  from the training frames, the pre-filled value is approximate and the user should
  adjust it. (This is exactly why pre-fill is editable rather than forced.)

## 6. Error handling

- Absent / malformed sidecar → `read_slice_meta` returns `None` → TrackerKit panel
  behavior byte-identical to today (no banner, no pre-fill).
- Empty `target_sizes` in the sidecar or settings → fall back to the stored
  `object_tile_fraction` (never divide by an empty median).
- `imgsz` absent/zero in the sidecar → fall back to `object_tile_fraction` (guard the
  division).
- User-set `reference_body_px` in DetectKit settings → build must not overwrite it.

## 7. Testing

Pure/unit (synthetic data, no ants, no models):
- dataset-level median measurement over multiple frames/objects;
- manifest records the **measured** median when `params.reference_body_px == 0`, and the
  **explicit** value when set;
- `preview_object_tile_fraction`: `median(target)/imgsz`, and the empty-`target_sizes`
  fallback;
- `read_slice_meta`: present dict / absent file / malformed JSON → `None`;
- `slice_meta_to_panel_values`: full mapping incl. median-target → fraction, and each
  fallback (empty target_sizes, missing imgsz, missing keys).

Guard / regression:
- No sidecar → TrackerKit panel behavior byte-identical (pre-fill is a no-op).
- DetectKit build with a user-set `reference_body_px` leaves the settings value unchanged.
- `predict_sliced_obb_result` / non-sliced preview unaffected when slicing off.

Qt (headless, `pytest.importorskip("PySide6")`):
- selecting a model whose path has a `.slice_meta.json` pre-fills the SAHI group widgets
  (enable + mode + reference + overlap + fraction) and raises the banner.

## 8. File structure

**Modified**
- `src/hydra_suite/training/sliced_dataset.py` — dataset-median measurement + manifest/stats.
- `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` — populate settings after build.
- `src/hydra_suite/detectkit/gui/main_window.py` — preview median-target fraction.
- `src/hydra_suite/detectkit/gui/prediction_preview.py` OR a small helper module — the
  pure `preview_object_tile_fraction` helper (place wherever keeps it Qt-free + testable).
- TrackerKit detection panel + its config/model-selection wiring — sidecar pre-fill + banner.

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
   absent/malformed sidecars.
5. Selecting an OBB model with a sidecar pre-fills the TrackerKit SAHI panel (editable) +
   shows a banner; no sidecar → byte-identical to today.
6. The scale caveat is documented at the pre-fill site.
