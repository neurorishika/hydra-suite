# SAHI Unification Across the Suite — Design

**Status:** Design approved in conversation (2026-10-09); pending written-spec review.
**Supersedes:** `docs/superpowers/specs/2026-09-06-unified-sahi-training-geometry-design.md`
(unimplemented, partly overtaken by multi-scale SAM3 @50cb5b94, geometry-escalation SAHI
@dba69a55, and calibration profiles). Its resolved decisions D7–D11 and the
"do not re-litigate" autonomy decisions (stored-pick + rule UX, R7 loud, D10 persist
semantic calibration on the sidecar, D11 warn-never-refuse) carry over unchanged.

## 1. Goal

TrackerKit's SAHI design (`SliceConfig`, `SLICE_*` engine keys, `slice_*` config keys,
`.slice_meta.json` v2 with calibration profiles) is the suite's framework. Every other
SAHI surface — DetectKit YOLO training / preview / calibration, SAM3 LoRA training,
SAM3 semantic escalation, SAM2 geometry escalation — adopts its vocabulary, its
parameter resolution, its sidecar, and one shared settings widget.

Success criteria:

- One canonical name per concept across code, configs, sidecars, CLI and UI.
- Every derived value is computed once, by one function, and shown read-only with its
  source; only genuinely free values are editable.
- TrackerKit tracking output is byte-identical before and after (equivalence matrix,
  MPS + CUDA). Existing caches, saved configs, project files and model sidecars load.
- The eight behavioral defects in §6 are fixed.

Non-goals: merging the three tile-merge algorithms (YOLO overlap-band NMS/NMM, SAM3
containment + polygon-IoU NMS, SAM2 owner tiles). They differ by design; only their
parameter vocabulary is unified. Multi-scale fan-out at inference time is out of scope
(see §3.3).

## 2. Current state (investigation summary)

The tile **grid** is already shared: `utils/slice_geometry.py` (`tile_size_for_mode`,
`plan_tiles`, `resolve_scales`, `get_slice_bboxes`) serves inference, YOLO training,
SAM3 training, DetectKit preview, and SAM3/SAM2 escalation. Everything around the grid
diverges:

| Concept | TrackerKit (canonical) | DetectKit YOLO train | DetectKit inference dialog | SAM3 train / sidecar | SAM3 / SAM2 escalation |
|---|---|---|---|---|---|
| Scale | `object_tile_fraction` | `target_size_fractions` + legacy `target_sizes` px | "Target object size (px)" ÷ hard 640 | `object_tile_fraction(s)`, `prefill_object_tile_fraction` | `tile_fraction` |
| Body px | `SLICE_TRAINED_BODY_PX` → `reference_body_px`; stamp `trained_body_px` | `reference_body_px`, manifest `measured_reference_body_px` | "Reference body (px)" | `reference_body_px` | "Body size (px)" |
| Tile px | `slice_width/height` | `slice_width/height` | same | manifest `tile_px(_set)`, sidecar `train_tile_px(_set)` | `tile_px` |
| Overlap | `slice_overlap` → `overlap_{h,w}_ratio` | `overlap` | `overlap` | `tile_overlap` | `overlap` |
| Full frame | `perform_standard_pred` | `full_frame_mix` | — | `full_frame_mix` | `tile_fraction=None`/`0.0` |
| Fragment floor | — | `min_area_ratio` (drop) | — | `min_area_ratio` / `min_retained_area_frac` (is_crowd) | — |
| Merge | `merge_policy/metric/threshold/backend` | `merge_threshold` (preview) | `merge_threshold` | — | `merge_iou` (polygon IoU), `seam_margin_px` |
| Mode labels | raw enum | "Fit labelled objects / Use model input / Custom tile size" | raw enum | hidden | none |
| Mode default | `auto_model` | `auto_object` | — | `auto_object` | — |
| Fraction default | 0.15 | 0.10 | — | 0.055 | 0.05 seed / SAM2 full frame |
| Overlap default | 0.2 | 0.2 | 0.2 | 0.25 | 0.5 |
| Sidecar | `<model>.pt.slice_meta.json` v2 | writes same | not read | `<artifact>.sam3_meta.json` | reads `.sam3_meta.json` |

Vocabulary is also mixed: "slice", "tile", "SAHI", "scale" are used interchangeably.

## 3. Canonical vocabulary and `TilingSpec`

### 3.1 Words

- **slice** — configuration and contract (`SliceConfig`, `SLICE_*`, `slice_meta`).
- **tile** — one runtime grid piece (`tile_px`, `tile_batch_size`).
- **scale** — one member of `object_tile_fractions`.
- **SAHI** — the user-facing feature name only (labels, `--sahi-profile`).
- Not used: patch, window, stride.

### 3.2 `TilingSpec`

A frozen, Qt-free dataclass in `utils/slice_geometry.py` (importable from core,
training, data and every kit). Field names are TrackerKit's.

| Field | Meaning |
|---|---|
| `enabled` | tiling on/off |
| `geometry_mode` | `auto_model` \| `auto_object` \| `custom` |
| `object_tile_fractions` | tuple of body÷tile fractions; one value = single scale |
| `reference_body_px` | body px (original frame) that sizes tiles at run time |
| `slice_width`, `slice_height` | tile px in original-frame pixels (0 = imgsz) |
| `overlap` | single ratio of tile size, both axes |
| `min_area_ratio` | training fragment floor: visible area ÷ full polygon area |
| `fragment_policy` | training: `drop` \| `crowd` \| `mask` (§6, F-mask) |
| `merge_policy`, `merge_metric`, `merge_threshold` | frame-level merge; `merge_metric` ∈ `iou` \| `ios` \| `polygon_iou` |

Backend extensions live beside the spec, not in it, because they mean different
things per backend: `perform_standard_pred` (YOLO inference extra full-frame pass),
`full_frame_mix` (training dataset composition), `seam_margin_px` (SAM3 merge),
`merge_backend` / `tile_batch_size` / `memory_budget_mib` (YOLO runtime admission).

`trained_body_px` survives only as the **provenance stamp** in the sidecar (the body
size a model was trained at). `REFERENCE_BODY_SIZE` is never written by any SAHI path.

### 3.3 Value resolution

Each derived field is resolved by one function returning `(value, source)`, where
`source ∈ {user, override, profile, stamped, dataset, derived, default}`. The UI shows
both (§5).

| Field | Resolution (first available) | Editable when |
|---|---|---|
| `enabled` | user | always |
| `geometry_mode` | user; prefilled from stamp | always |
| `object_tile_fractions` | calibration profile → stamp → per-backend default (§3.4) | `auto_object` |
| inference operating scale | profile scale → stamped median → default | via profile / `auto_object` |
| `reference_body_px` | override → dataset label median (DetectKit) → stamped `trained_body_px` → `REFERENCE_BODY_SIZE × RESIZE_FACTOR` (TrackerKit) | only via "Override" |
| `slice_width/height` | `tile_size_for_mode(...)` | `custom` only |
| `overlap` | override → saved value → derived `min(max(fractions) + margin, 0.9)` | only via "Override" |
| merge fields | profile → per-backend default | yes |
| `fragment_policy`, `min_area_ratio` | per-backend default | yes |

Inference always runs **one operating scale** (decided 2026-10-09). Multi-scale is a
training-robustness tool; serving cost stays 1×.

Overlap derivation rationale: an animal is guaranteed whole inside at least one tile
iff overlap px ≥ body px, i.e. `overlap ≥ object_tile_fraction`. The derived value is
used **only when no overlap is saved**; any saved overlap (including TrackerKit's 0.2
default persisted in existing configs) counts as set. The margin constant is chosen in
the plan and documented next to the default table.

### 3.4 Per-backend default table

One module-level table in `utils/slice_geometry.py`, read by GUI and headless alike,
with a comment justifying each value. Fraction defaults stay per backend because
fractions do not transfer between models (2026-10-03 spec); stamped or dataset values
override them.

| Backend | fractions | geometry_mode | fragment_policy | merge |
|---|---|---|---|---|
| YOLO (train + infer) | train set `(0.05, 0.10, 0.15, 0.20)`; infer default `0.15` | `auto_object` train / `auto_model` infer (unchanged) | `drop` | `greedy_nmm` / `ios` / 0.5 |
| SAM3 | `(0.055,)` | `auto_object` | `crowd` | `polygon_iou` / 0.5 |
| SAM2 | none (full frame until calibrated) | `auto_object` | n/a | n/a (owner tiles) |

### 3.5 Aliases and compatibility

- One `SLICE_ALIASES` map (legacy name → canonical) is the only translation point,
  applied on read of configs, project files, training plans and sidecars. Writers emit
  canonical names only.
- Aliases: `target_size_fractions`, `tile_fraction`, `prefill_object_tile_fraction` →
  `object_tile_fractions`; `target_sizes` → fractions using the **stated** imgsz
  (read-only, never written); `tile_overlap`, `slice_overlap` → `overlap`;
  `min_retained_area_frac` → `min_area_ratio`; `merge_iou` → `merge_threshold` with
  `merge_metric="polygon_iou"`; `tile_fraction=None`/`0.0` → `enabled=False`.
- `SLICE_*` engine keys, `SliceConfig` fields, and `_slice_config_hash` are unchanged
  byte-for-byte; `get_parameters_dict` goldens must not move.
- `nmm` is hidden in the UI and read as `greedy_nmm` (identical code path today); the
  cache hash keeps the raw stored value so old caches still hit.

## 4. Sidecar v3

- One file per model: `<model>.<ext>.slice_meta.json`, `schema_version: 3`, for both
  YOLO and SAM3:
  ```
  { schema_version: 3,
    model_family: "yolo" | "sam3",
    training_geometry: { geometry_mode, object_tile_fractions, object_tile_fraction,
                         overlap, min_area_ratio, fragment_policy, imgsz,
                         tile_px_set, trained_body_px, <family extras> },
    primary_profile_id, profiles: [ ...unchanged from v2... ] }
  ```
- v3 is additive over v2: it keeps a scalar `object_tile_fraction` (the operating-scale
  median) so older readers still work; it never writes `target_sizes`.
- `.sam3_meta.json` keeps only non-tiling metadata (augmentation, batching,
  checkpoint). SAM3 publish writes tiling geometry to `slice_meta`.
- SAM3 semantic calibration persists as sidecar profiles (D10). SAM2 is a stock
  pretrained model, so its geometry calibration stays per project + variant.
- One reader, `read_tiling_meta(model_path)`, normalizes v1, v2, v3 and legacy
  `.sam3_meta.json` geometry via `SLICE_ALIASES`. TrackerKit, DetectKit preview,
  calibration, the SAM3 dialog and the drift guard all read through it.
- Every writer/reader pair has one round-trip test per family (guards against a repeat
  of the `foo.slice_meta.json` vs `foo.pt.slice_meta.json` mismatch).

## 5. UI

### 5.1 One shared widget

`SliceSettingsGroup` moves from `detectkit/gui/panels/slice_settings_widget.py` to
`hydra_suite/widgets/slice_settings.py` (shared layer; imports no app layer). It is
constructed with a `role`:

| Role | Hosts |
|---|---|
| `infer_yolo` | TrackerKit Detection panel (direct mode); DetectKit inference settings dialog |
| `train_yolo` | DetectKit training dialog |
| `train_sam3` | SAM3 training panel |
| `escalate_sam3` | SAM3 semantic escalation dialog |
| `escalate_sam2` | SAM2 escalate dialog |

The role selects which rows appear. Labels, ranges, tooltips and resolution are
identical everywhere. It replaces TrackerKit's hand-built slice rows in
`detection_panel.py`, DetectKit's inference-dialog SAHI group (and its px÷640 field),
and the tiling rows in the SAM3/SAM2 dialogs and SAM3 training panel.

### 5.2 Layout (top to bottom)

1. **"Sliced inference (SAHI)"** toggle.
2. **Profile** combo (Training geometry / calibrated profiles / Custom) + status line;
   shown when a model sidecar exists.
3. **Tile strategy**: "Fit to animal size" / "Use model input size" / "Custom tile
   size"; enum value in the tooltip.
4. **Object scale**: single value (inference roles, shows the operating scale) or a
   scale-set editor (training roles).
5. **Derived rows** with source badge (`dataset` / `stamped` / `derived` /
   `override`): Body size [Override…], Tile size W×H (editable only in Custom),
   Overlap [Override…].
6. **Live tile-layout preview** (existing DetectKit schematic, now also in
   TrackerKit): real frame size, tiles per frame, warning near the 4096-tile ceiling.
7. **Advanced** (collapsed): merge policy/metric/threshold; fragment policy + min area
   (training roles); full-frame pass / full-frame mix; tiles per call and tile memory;
   seam margin (SAM3).

### 5.3 Rules

- Constrained fields are disabled, never hidden; the tooltip names their source.
- Widget ranges come from the contract validators.
- Editing a profile-owned value moves the profile to "Custom (based on X)"; derived
  values never do.
- "Calibrate…" is reachable from the widget wherever calibration exists (YOLO direct,
  SAM3, SAM2).
- CLI uses the same names: `detectkit escalate sam3` gains `--tile-fraction` and
  `--reference-body-px` (parity with `sam2`); `--sahi-profile` is unchanged.

## 6. Behavioral fixes

| # | Defect | Fix |
|---|---|---|
| F1 | px↔fraction anchored to 640 (`detectkit/gui/dialogs/inference_settings.py:220`, `slice_settings_widget.py:640` even after `set_model_input_size`) | Fractions only; px is a read-only display at the role's real imgsz; `target_sizes` read-only |
| F2 | DetectKit preview ignores the model sidecar and always runs the median scale | Preview resolves via `read_tiling_meta` + profile, the TrackerKit ladder (`resolve_slice_profile_values`) |
| F3 | Headless `escalate sam3` is always full-frame (`semantic_escalation.py:736`, no body/fraction flags) | Add flags; resolve from sidecar → dataset; full frame only when uncalibrated, as SAM2 |
| F4 | `SliceTrainingSettings.target_size_fractions` defaults empty, `SliceTrainingConfig` to 4 values | Both read §3.4 |
| F5 | SAM3 overlap spin allows 1.0, contract rejects it | Ranges from validators |
| F6 | `nmm` ≡ `greedy_nmm` code path | Hide `nmm`, read as `greedy_nmm`; hash raw value |
| F7 | Unexplained overlap defaults 0.2 / 0.25 / 0.5 | Derived overlap (§3.3), only where unset |
| F8 | Fraction defaults scattered as inline literals | §3.4 table |

**F-mask (flagged, default off):** a YOLO `fragment_policy="mask"` that drops a
sub-threshold fragment's label and paints out its pixels, avoiding teaching visible
animal parts as background. Ultralytics has no ignore regions, so this is the YOLO
analogue of SAM3's `crowd`. Default stays `drop` until a retrain A/B decides.

**Filtering rule (inference):** tiles predict at a confidence floor with no model-side
NMS; confidence thresholding, merge, and a dataset-derived `AreaBand` outlier filter run
on the merged frame result, shared by YOLO and SAM3. Training fragment handling is
necessarily tile-level (training has no frame level) and is governed by
`fragment_policy`. `min_area_ratio` stays a visibility ratio, not an outlier rule:
its distribution is pure grid geometry. Size-outlier detection belongs to the
frame-level `AreaBand`.

## 7. Migration and verification

Each slice is its own worktree branched from local HEAD, merged in order.

| Slice | Scope | Gate |
|---|---|---|
| S1 | `TilingSpec`, `SLICE_ALIASES`, resolution functions, §3.4 table (no callers) | Unit tests; `test_inference_slicing.py` byte-parity |
| S2 | Sidecar v3 writer + `read_tiling_meta`; SAM3 publish writes `slice_meta` | Round-trip per family; v1/v2/`.sam3_meta` fixtures read |
| S3 | All callers on `TilingSpec` (TrackerKit engine params, DetectKit train / preview / calibration, SAM3 / SAM2 jobs); F1–F4, F6, F8; frame-level `AreaBand` for YOLO behind default-off until measured | `get_parameters_dict` goldens unchanged; GUI/CLI parity tests; `test_sam3_gui_cli_training_parity.py`, `test_shared_scoring_primitives_identity.py`, `test_core_import_is_light.py`; drift-guard tests; **equivalence matrix byte-identical, MPS + CUDA** |
| S4 | Shared widget in `widgets/`, swapped into all five hosts; F5, F7; SAM3 CLI flags | Parameterized widget → `TilingSpec` test per role; GUI/CLI parity per role; manual GUI check in both kits |
| S5 | User-guide glossary + terminology pass; 09-06 spec marked superseded and moved to `done/` | `make docs-check` |

TrackerKit tracking must stay byte-identical: no `SLICE_*` default changes, derived
overlap never applies to saved configs. Any gate diff is a bug, not a new noise floor.

Existing tests retargeted rather than deleted: `test_detection_panel_slice_widgets.py`,
`test_detectkit_slice_ui.py`, `test_detectkit_slice_settings.py`,
`test_sam3_slice_settings_shared.py`, `test_slice_meta_read.py`,
`test_engine_params_slice_profile.py`, `test_gui_cli_profile_parity.py`.

## 8. Open items for the plan

- Overlap margin constant (§3.3).
- Whether YOLO frame-level `AreaBand` defaults on after measurement (S3 ships it off).
- F-mask A/B protocol and dataset (separate run; not a merge gate).
