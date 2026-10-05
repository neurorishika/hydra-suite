# SAM3 LoRA training augmentation — design

**Status:** Approved for implementation (goal-directed session, 2026-10-05).
**Branch:** `feat/sam3-lora-augmentation` (worktree `.worktrees/sam3-aug`).

## Why

SAM3 LoRA training (`src/hydra_suite/training/sam3_lora/`) applies **no**
augmentation. `dataloader._default_transform` is `to_tensor`;
`datapoints.build_tile_datapoint` only resizes to `RES` and applies Meta's
`NormalizeAPI`; `dataset_build` writes fixed tiles. Every epoch replays the same
pixels, which encourages overfitting on the small labeled sets SAM3 is meant
for. YOLO roles get augmentation through `AugmentationProfile`
(`training/contracts.py`) and Ultralytics. This design gives the SAM3 role a
YOLO-parallel augmentation path with GUI controls, CLI/JSON parity, and a
realised stamp.

## Decisions made without interactive review

The session ran under a "do not pause" goal, so these were decided by the
implementer (with an advisor pass) rather than by the user. Each one can be
reversed cheaply.

1. **Contract default OFF, GUI fresh-session default ON.**
   `Sam3LoraParams().augmentation` is a disabled profile, so every existing
   plan JSON, published spec and probe resolves to exactly today's training
   (the `patience` rule in `contracts.py`: "a plan written before this field
   existed must train exactly as it did"). The SAM3 panel's fresh default is
   the recommended profile below. This is a **documented intentional GUI/CLI
   default divergence**, asserted next to the existing `auto_import` one in
   `tests/test_sam3_gui_cli_training_parity.py`.
2. **Schema: nest the existing `AugmentationProfile`** as
   `Sam3LoraParams.augmentation`. No parallel `aug_*` fields. A
   `__post_init__` coerces a `dict` into `AugmentationProfile`, so all four
   `Sam3LoraParams(**dict)` reconstruction sites (sidecar child, publish CLI,
   dialog project load, dataset-prep sidecar) keep working unchanged.
3. **New typed field `AugmentationProfile.rot90: float = 0.0`**: the
   probability of a 90° rotation. Transpose-type rotations cannot be expressed
   with flips, and together with `fliplr`/`flipud` they cover all eight
   label-exact dihedral symmetries of a top-down view. SAM3 is the only
   consumer today. YOLO/classify ignore it, as they already ignore `rotate`.
4. **Small-angle `rotate` defaults to 0** (opt-in). It needs a border fill and
   polygon clipping, so the label-exact geometric ops are the default.
5. **BGR handling:** the tile arrives from `cv2.imread` as BGR. The
   photometric ops convert to RGB once, run in RGB (the convention of
   `_apply_tiny_augmentation` and `simulate_decode_color`), and convert back.
6. **Saved dialog state without an `augmentation` key** keeps the panel's
   current (fresh-default) augmentation rather than resetting it to the
   contract's OFF. Older per-machine UI state therefore picks up the
   recommended profile. Old *plan JSON* (CLI) does not.

## Vocabulary (reuse of `AugmentationProfile`)

| Field | SAM3 semantics | Fresh GUI default |
|---|---|---|
| `enabled` | master switch | `True` |
| `fliplr` | P(horizontal flip) | 0.5 |
| `flipud` | P(vertical flip) | 0.5 |
| `rot90` (new) | P(rotate 90°, direction 50/50 CW/CCW) | 0.5 |
| `rotate` | max \|angle\| in degrees, uniform; constant gray fill | 0.0 |
| `brightness` | multiplicative factor ~ U(1−b, 1+b) | 0.2 |
| `contrast` | about the per-channel mean, ~ U(1−c, 1+c) | 0.2 |
| `saturation` | HSV S factor ~ U(1−s, 1+s) | 0.2 |
| `hue` | HSV H shift ~ U(−h, h)·179 (Ultralytics `hsv_h` units) | 0.0 |
| `monochrome` | gray → 3-channel | `False` |
| `decode_color_sim` | P(apply), `training.augmentation.simulate_decode_color` | 0.0 |
| `resample_sim` | P(apply), `training.augmentation.simulate_resample` | 0.0 |

The photometric semantics match `runner._apply_tiny_augmentation` exactly,
except that draws come from an explicit generator. `hue` defaults to 0
because colour can be the concept's signal (painted tags), as with
`fliplr=0` guidance for asymmetric animals on the YOLO side.

**Unsupported for SAM3, rejected loudly** by the shared validator:
`canonical_aug=True`, non-empty `label_expansion`, non-empty `args`.
`canonical_aug_copies` is inert unless `canonical_aug` is set, and is ignored.
Ranges: probabilities in [0, 1]; `rotate` in [0, 180]; `hue` in [0, 0.5];
`brightness`/`contrast`/`saturation` in [0, 1].

Mosaic and mixup are deliberately not ported. Mosaic changes the tile scale
that the scale-grouped multi-scale build controls, and mixup blends unlabeled
pixels into exhaustive queries. Both would corrupt SAM3's exhaustive-query
supervision.

## Architecture

### New module `training/sam3_lora/augment.py` (pure, Qt-free, no `sam3` import)

- `is_active(profile) -> bool`: true when `enabled` and at least one op is
  non-zero (or `monochrome`).
- `validate_sam3_augmentation(profile) -> list[str]`: error strings for range
  or unsupported-field violations. It is called by
  `DetectTrainingPlan.validate()` (raising `TrainingPlanError`) and by the
  sidecar child at startup (raising `RuntimeError`).
- `active_ops(profile) -> dict`: the ops actually in effect (for the stamp).
- `augment_tile(tile_bgr, instances, profile, rng, *, min_area_ratio)
  -> (tile_bgr, instances)`. `instances` is `[(polygon Nx2 float32, is_crowd)]`
  in tile-pixel, edge-convention coordinates (the convention
  `_scale_polygons_to_res` assumes).
  - Geometric ops in order flipud → fliplr → rot90 → rotate, each applied to
    the image and every polygon together. Flip: `x' = w − x`. rot90 CW:
    `(x, y) → (h − y, x)` with new shape (w, h). CCW is the inverse.
  - `rotate`: `cv2.getRotationMatrix2D` about the pixel-index centre
    `((w−1)/2, (h−1)/2)`, polygons mapped as `index = edge − 0.5`, warp with
    `BORDER_CONSTANT` value (114, 114, 114) — **never reflect**, which would
    paste unlabeled mirrored animals into exhaustive queries. Polygons are
    clipped to the tile with `utils.slice_geometry.clip_polygon_to_tile`.
    Retained fraction = clipped area / pre-clip area (the tile polygon; the
    full-frame area that `dataset_build` used is unavailable here).
    0 → instance dropped (it is no longer in the image). Below
    `params.min_area_ratio` → `is_crowd=True`, so
    `select_output_objects` excludes it and marks the positive query
    non-exhaustive. Already-crowd stays crowd.
  - Photometric ops (BGR→RGB once): brightness → contrast → saturation/hue →
    monochrome → decode_color_sim → resample_sim → RGB→BGR. `resample_sim`'s
    ≤0.5 px sub-pixel shift is left unlabeled, as it is for classifiers.
- `tile_rng(epoch_seed, image_id) -> np.random.Generator`:
  `np.random.default_rng([epoch_seed, image_id, _SALT])`. Draws are therefore
  independent of shuffle and grouping order, and reproducible per epoch.
- `make_tile_augmenter(profile, *, epoch_seed, min_area_ratio)
  -> Callable | None`: returns `None` when not `is_active`. A no-op profile
  therefore takes literally the existing code path.

### Dataloader threading (`dataloader.py`)

`load_datapoints(descriptor, transform, augmenter=None)` applies
`augmenter(tile_bgr, instances, descriptor.image_id)` before
`build_shared_query_datapoints`. `collate_batches(..., augmenter=None)` and
`collate_epoch_batches(..., augmenter=None)` (both grouped and ungrouped arms)
forward it. **Only the training loop passes one.** Validation (`cli.py`
`collate_batches(val_descriptors, …)`), the autobatch probe
(`collate_batches(densest, …)`) and `detection_quality` keep `None`.

`TileDescriptor.width/height` stay the stored tile shape. `rot90` swaps the
array's shape, and `build_tile_datapoint` reads the shape from the array.
Descriptor dims are only read by consumers that skip decoding, and none of
those sit downstream of the augmenter (verified in the plan).

**The probe stays an upper bound.** Flips and rot90 preserve the instance
count and the RES×RES tensor; rotate can only drop instances. The collator's
max-instance padding (the superlinear VRAM driver) therefore never grows.

### Training loop (`cli.py`)

At startup: validate `params.augmentation` (fail loudly). Write
`hydra_sam3_augmentation.json` to the run dir on **both** arms:
`{"requested": asdict(profile), "applied": {"augmentation": bool, "ops": {...},
"reason": str}}`, with reason `"disabled"` / `"no active ops"` when the arm is
off. In each epoch, build `make_tile_augmenter(profile,
epoch_seed=spec.seed + epoch, min_area_ratio=params.min_area_ratio)` and pass
it to `collate_epoch_batches`. Log one line stating the arm and active ops.

### Publish (`publish_worker.py`)

`_augmentation_metadata(run_dir)` reads the stamp (`read_sam3_augmentation_stamp`)
and adds an `"augmentation"` block to the sidecar next to
`scale_grouped_batching`. When the stamp is absent it adds `{}`, so old runs
are unchanged.

### Config / CLI (`detectkit/config/training.py`)

A shared `_parse_augmentation_profile(values, label)` handles typing for both
`training.augmentation` (now also accepting `rot90`) and `sam3.augmentation`.
`_sam3_to_json` already recurses via `asdict`. `plan.validate()` calls
`validate_sam3_augmentation` when SAM3 is a role.

### GUI (`detectkit/gui/panels/sam3_training_panel.py`)

A new checkable `QGroupBox("Augmentation")` on the SAM3 tab with spin boxes
for every row of the vocabulary table and a monochrome checkbox, each with a
tooltip. `params()` emits `augmentation=AugmentationProfile(...)`;
`set_params()` round-trips it. Panel construction applies the fresh
recommended profile (`RECOMMENDED_SAM3_AUGMENTATION` in `augment.py`, the one
source for the defaults). The dialog's persisted-state loader keeps the
panel's augmentation when the saved state has none (decision 6).

## Error handling

- Invalid ranges or unsupported fields: `TrainingPlanError` at plan
  validation (GUI shows its existing warning dialog), and `RuntimeError` in
  the child if a hand-edited `spec.json` bypassed the plan.
- An augmenter exception inside a batch propagates (never swallowed). A silent
  fallback to unaugmented training would make the stamp lie.

## Testing

- `augment_tile` unit tests (no `sam3` needed): for each geometric op, rasterise
  the polygons to masks before and after, and check the mask transformed by
  the same op as the image equals the mask of the transformed polygons (IoU ≈
  1). Edge-clipped instances under `rotate`: drop at 0, crowd below
  `min_area_ratio`, already-crowd stays crowd. Constant fill (no reflected
  content). Photometric ops leave polygons bit-identical. BGR/RGB: a pure-red
  BGR tile stays red after brightness-only.
- Determinism: same `(epoch_seed, image_id)` gives identical output; a
  different epoch gives different output; the result is independent of
  descriptor order.
- Disabled identity (goal item 5): `collate_epoch_batches(..., augmenter=
  make_tile_augmenter(disabled))` yields batches equal to `augmenter=None`,
  and `make_tile_augmenter` returns `None` for disabled and all-zero profiles.
  Default `Sam3LoraParams()` is inactive.
- Config: `sam3.augmentation` round-trip through `to_dict`/`from_dict`;
  rejection of out-of-range and unsupported fields; `training.augmentation.rot90`
  accepted.
- `Sam3LoraParams(**asdict(p))` round-trip, i.e. `__post_init__` coercion.
- GUI: panel `params()`/`set_params()` round-trip; fresh default equals
  `RECOMMENDED_SAM3_AUGMENTATION`; parity test covers the nested field with a
  non-default profile; intentional default divergence asserted.
- Stamp written on both arms; publish sidecar carries it.
- Reflective guards updated: `_REFERENCE_KWARGS`, `_SAM3_PARAM_FIELDS`,
  slice-settings contract, geometry drift guard.
- Full SAM3 test files + contract guards on `hydra-mps`, with
  `hydra_suite.__file__` verified to resolve into the worktree.
- Visual: PNG grid of augmented fixture tiles with overlaid polygons.
- CUDA smoke on **courtship** (RTX 4090): at least 1 epoch with augmentation
  on, finite loss, no OOM, peak VRAM within the probe's admission.
