# Crop-Padding Retirement — one framing dial, plus an AprilTag-local one

**Date:** 2026-08-18
**Status:** Design approved, pending spec review
**Branch:** `refactor/crop-padding-retirement` (worktree from local HEAD)

## Context

Two crop-framing knobs are exposed to the operator today, and they are not
peers:

| Knob | Where | What it actually does |
|---|---|---|
| `canonical_margin` (Setup → Reference Scale) | `ADVANCED_CONFIG.canonical_margin` | Layer 1. With `REFERENCE_BODY_SIZE` and `reference_aspect_ratio` it defines the **fixed canonical canvas** for the whole project: `canvas_w = body_px·√ar·margin`, `canvas_h = canvas_w/ar` (`core/canonicalization/geometry.py:38-50`). The crop transform is rigid — rotation + translation at scale 1 — so margin is the only dial against a clipped animal. |
| `individual_crop_padding` (Identity → "Crop padding fraction (all phases)") | `INDIVIDUAL_CROP_PADDING` | The pre-canonicalization per-detection AABB expansion: grow the OBB's bounding box by a fraction of its size before cutting. |

The Global Canonicalization work (merged `0ca40789`) made the canonical
geometry the single fit for every crop consumer — training and inference, all
kits — which left `individual_crop_padding` in an incoherent state:

- **Pose:** fully dead. Crops come off the canonical canvas
  (`core/inference/stages/pose.py:305-320`); `PoseConfig.crop_padding` only
  perturbs the pose cache-key hash (`core/inference/cache/keys.py:255`).
- **Head-tail / CNN identity:** dead. `extract_and_classify_batch` treats a
  supplied `padding_fraction` as `geometry.margin - 1.0` and *raises* if the
  caller disagrees (`core/canonicalization/crop.py:373-385`) — the geometry
  always wins.
- **Crop-dataset export (class training):** effectively dead. The generator
  builds a `CanonicalGeometry` from `canonical_margin`
  (`core/individual/dataset/generator.py:99-124`); `padding_fraction` survives
  only in a legacy non-canonical AABB branch that fires when
  `reference_aspect_ratio <= 0`, plus two metadata fields.
- **Oriented-video export:** `padding_fraction` is a *fallback* margin used
  only when no `geometry` is passed (`core/individual/dataset/oriented_video.py:194-208`);
  the live TrackerKit path always passes one (`core/tracking/session.py:485`).
- **AprilTag:** the one genuinely live consumer. Tag crops are axis-aligned
  patches cut via `_expand_obb_to_aabb` (`core/tracking/pose/pose_pipeline.py:66`),
  reached from `core/inference/runner.py:906`, `core/inference/pipeline.py:394`,
  and `core/post/interpolated_crops.py:714`, and hashed into the AprilTag cache
  id (`core/individual/properties/cache.py:291`).

So a project-wide knob labelled "all phases" changes exactly one phase, and the
crops used for classifier *training* are framed by a different mechanism than
the operator's label implies. That is the same silent-mismatch class the
canonicalization module was built to remove.

An AprilTag is a rigid printed square: any rotation, rescale, or canvas fit
degrades decode. Its crop should be the plain axis-aligned extent of the
inference OBB — which is a property of the AprilTag stage, not a project-wide
crop policy.

## Goal

One framing dial for everything that feeds a model or a dataset —
`canonical_margin` — and one stage-local dial for AprilTag,
`apriltag_crop_padding`, defaulting to the bare AABB extent of the inference
OBB. `individual_crop_padding` ceases to exist.

Plus a GUI-text-only rename of the detection-batching control, which is
unrelated in mechanism but was the question that surfaced this and is cheap to
carry here.

## Non-goals

- `roi_crop_padding_fraction` (`advanced_config.json:5`) is a different knob —
  the ROI bounding-box padding for region selection. **Untouched.**
- The canonical geometry's own maths, `REFERENCE_BODY_SIZE`,
  `reference_aspect_ratio`, and the clipping-stats reporting. Untouched.
- `detection_batch_size` semantics or plumbing. Only its GUI labels change.

## Design

### Part 1 — Rename the detection-batching control (GUI text only)

`src/hydra_suite/trackerkit/gui/panels/detection_panel.py:1082-1112`:

- Group box `"Live Detection Batching"` → **`"Detection Frame Batching"`**.
- Row label `"Frame batch size"` → **`"Frames per detector call"`**.
- Help text and tooltip state explicitly that this is **stage-1 detection
  only** (the OBB / bg-sub model's input batch) and is *not* the stage-2 crop
  batching used by head-tail, CNN identity, pose, and AprilTag — those always
  run the backend once over every crop in the chunk
  (`core/inference/stages/cnn.py:142`, `stages/pose.py:395`,
  `stages/headtail.py:258`).

The word "Live" was ambiguous: it meant "during a real tracking run, as opposed
to the Test Detection preview", but reads as "realtime mode" — which is one of
the conditions that *locks the control to 1*
(`_sync_live_detection_batch_controls`, line 1424). The policy-notice strings
are re-worded to match the new labels. `detection_batch_size`,
`InferenceConfig.detection_batch_size`, and `YOLO_BATCH_SIZE` are unchanged, so
no saved config, cache key, or engine profile moves.

### Part 2 — Retire `individual_crop_padding`

**Invariant after this change:** every crop that feeds a model or a dataset is
extracted through the project-wide `CanonicalGeometry`. AprilTag is the sole
exception, and it is an exception by construction, not by configuration.

Deletions, by layer:

**Config / params**
- `individual_crop_padding` removed from `trackerkit/cli_config.py`,
  `trackerkit/engine_params.py:1332-1334` (`INDIVIDUAL_CROP_PADDING` stops
  being emitted), `resources/configs/default.json:138`, and
  `resources/configs/ooceraea_biroi.json:217`.
- `trackerkit/gui/orchestrators/config.py`: the load path (`:1393`) and save
  path (`:1978`) drop the key; the stale migration message at `:1230`
  ("All precompute phases now use 'individual_crop_padding'") is deleted.
- `trackerkit/gui/workers/preview_worker.py:562` drops its
  `INDIVIDUAL_CROP_PADDING` assignment.

**GUI**
- The `spin_individual_padding` widget and its
  `"Crop padding fraction (all phases)"` form row are removed from
  `trackerkit/gui/panels/identity_panel.py:260-272`, along with every reference
  to the widget.

**Core**
- `PoseConfig.crop_padding` (`core/inference/config.py:383`) deleted, and with
  it the `config.crop_padding` term in the pose cache-key hash
  (`core/inference/cache/keys.py:255`). The geometry key already in that hash
  covers real framing changes. Existing on-disk pose caches are invalidated and
  recompute once — acceptable, and the same class of change as any cache-key
  edit in this repo.
- `extract_and_classify_batch`'s `padding_fraction` parameter
  (`core/canonicalization/crop.py:339`) is removed; `geometry` becomes
  required-or-canvas-dims as before, and the disagreement guard at 373-385 goes
  away with the parameter it guarded. The `elif padding_fraction is None:
  padding_fraction = 0.1` default at 386-387 disappears; a caller passing bare
  `canvas_w`/`canvas_h` gets a synthesized geometry whose margin comes from the
  canvas itself.
- `IndividualDatasetGenerator`: `self.padding_fraction` and `_canonical_padding`
  deleted. **The legacy non-canonical AABB branch is deleted** — a
  `reference_aspect_ratio <= 0` now raises `ValueError` at construction instead
  of silently producing training crops that do not share the canvas inference
  uses. `_expand_corners`-style padding helpers that become unreachable go with
  it. `metadata.json` loses `parameters.padding_fraction` and
  `expansion_factor`; the existing `canonical` block (canvas, margin, aspect,
  clipping stats) is the sole record of framing.
- `OrientedVideoExporter`: the `padding_fraction` constructor argument is
  removed. The geometry-less fallback (`oriented_video.py:194-208`) uses the
  project default `canonical_margin = 1.3` and keeps its existing loud warning
  that the canvas may not match the project's. `FrameTask.expanded_corners`
  (`:985`) — the self-polygon used to build the keep-mask when foreign-OBB
  suppression is on (`:1236`) — is expanded by `self._geometry.margin - 1.0`
  instead of `self.padding_fraction`, matching the convention
  `core/canonicalization/crop.py` already uses for the same quantity. With a
  project geometry passed (the live path) that is `0.3` where it was
  `individual_crop_padding`, so **the exported oriented-video mask changes for
  runs that had `individual_crop_padding != 0.3`** — not a tracking-CSV
  difference (positions/angles/IDs are untouched), but a pixel difference in
  **every** exported oriented crop/video, since `expanded_corners` feeds the
  keep-mask applied to every rendered task (`_render_task`,
  `oriented_video.py:~1238-1262`), not just the foreign-suppression fill.
  Exported oriented images/videos can themselves feed a training set, so this
  is not purely cosmetic.
- `core/post/media_export.py:846,904` and `core/tracking/session.py:480` stop
  threading `padding_fraction`; the `geometry=` they already pass is the whole
  story.
- `core/post/interpolated_crops.py:714` reads the new AprilTag key (Part 3).

**Legacy configs**
`individual_crop_padding` present in a loaded config produces **one loud
warning** naming `canonical_margin` (for crop framing) and
`apriltag_crop_padding` (for tag crops), then is ignored. No hard error and no
migration script: unlike the `runtime_tier` cutover, ignoring this key cannot
select a wrong backend or silently change tracking output — the canonical path
was already what ran for every stage except AprilTag, and AprilTag's change is
called out below. The warning is emitted once per load, from the same place
that reads the rest of the identity config.

`ooceraea_biroi.json` currently sets `0.5`. That value has had no effect on
anything but AprilTag crops since the canonicalization merge; the key is
deleted from the bundled file and that lab's crop framing continues to be
`canonical_margin: 1.3`, which the file already sets.

### Part 3 — `apriltag_crop_padding`

New config key `apriltag_crop_padding` / param `APRILTAG_CROP_PADDING`,
**default `0.0`**, range `-0.5 … 2.0`, step `0.05`.

- `0.0` means the crop is exactly the axis-aligned bounding box of the
  inference OBB — no padding, no transformation. `_expand_obb_to_aabb`
  (`core/tracking/pose/pose_pipeline.py:66-106`) already produces exactly that
  at `0.0`, and shrinks symmetrically for negative values, so no new geometry
  code is needed.
- GUI: a `Crop padding` row in the AprilTag settings group of
  `trackerkit/gui/panels/identity_panel.py`, next to Downsampling / Blur /
  Sharpening (`:170-180`). Tooltip: padding as a fraction of the OBB's
  bounding-box size; `0.0` = the detection's exact extent; negative tightens.
- Plumbing: `trackerkit/gui/panels/detection_panel.py:1940`-style context dict,
  `engine_params.py` alongside the other `APRILTAG_*` keys,
  `AprilTagConfig.from_params` in both
  `core/individual/classification/apriltag.py:117` and
  `core/inference/config.py:1082`, and the AprilTag cache id payload
  (`core/individual/properties/cache.py:291`, where the `padding_fraction`
  entry is renamed to `apriltag_crop_padding` and re-sourced).
- `resources/configs/default.json` and `ooceraea_biroi.json` gain
  `"apriltag_crop_padding": 0.0`.

**Default choice.** `0.0` everywhere, including `ooceraea_biroi.json` — not a
`0.1` pin to preserve today's behavior. The design rationale is that the tag
crop should be the detection's extent; a lab that finds decode degraded has a
visible, stage-local dial to raise, whereas a hidden `0.1` pin reintroduces
exactly the "a number in a file frames my crops and I can't see why" problem
this change exists to remove. This is the one decision in this spec taken
without an explicit ruling; it is called out here so it can be overridden
cheaply before implementation.

## Behavior changes

| Surface | Change |
|---|---|
| Tracking CSVs (positions, angles, track IDs) | **None.** Byte-identical. Every crop stage except AprilTag already ran the canonical path. |
| AprilTag decode | **Changes.** Tag crops go from 10%-padded to bare OBB extent by default. Tag IDs/assignments on AprilTag clips may differ. |
| Pose detection cache | Invalidated once (cache-key hash loses a term); recomputes, same results. |
| AprilTag cache | Invalidated once (payload key renamed and re-sourced); recomputes. |
| Crop-dataset `metadata.json` | Loses `padding_fraction` and `expansion_factor`; `canonical` block unchanged. |
| Oriented-video keep-mask | Self-polygon expansion (`expanded_corners`, `oriented_video.py:989`) becomes `canonical_margin - 1.0` (0.3 by default) instead of `individual_crop_padding` (0.1 by default). This mask is applied in `_render_task` (`oriented_video.py:~1238-1262`) to **every** exported task, not just when foreign-OBB suppression is on — that flag only gates the separate foreign-fill step. So this changes pixels in every exported oriented crop/video, not just visuals gated behind suppression, and exported images can feed a training set. |
| `canonicalization/crop.py` bare-canvas geometry (reporting only) | Synthesized geometry for callers that pass bare `canvas_w`/`canvas_h` (`crop.py:381-384`) now uses `margin=1.0` instead of `1.0 + padding` (1.1 previously). `margin` feeds only `overflow_ratio`/`clipped` reporting, never the affine, so pixels are byte-identical — but such callers now report fewer clipped detections than before. |
| `reference_aspect_ratio <= 0` | Was a warning + non-canonical fallback; now a `ValueError`. |
| Configs with `individual_crop_padding` | Load fine, one warning, key ignored. |

## Testing

Unit (new, in `tests/`):
1. Loading a config carrying `individual_crop_padding` logs the warning once
   and the resulting params contain no `INDIVIDUAL_CROP_PADDING`.
2. `IndividualDatasetGenerator` with `reference_aspect_ratio <= 0` raises
   `ValueError`.
3. `apriltag_crop_padding = 0.0` yields a crop whose bounds equal
   `cv2.boundingRect` of the OBB corners, to the pixel.
4. Negative `apriltag_crop_padding` shrinks the crop symmetrically about the
   OBB centroid.
5. `extract_and_classify_batch` no longer accepts `padding_fraction` (the
   removed-parameter contract), and geometry-driven output is unchanged.

Regression:
6. Grep gate — no `INDIVIDUAL_CROP_PADDING` / `individual_crop_padding`
   remains in `src/` or `tests/` outside the legacy-warning site.

Equivalence (`tools/equivalence/run_matrix.sh`, baseline = pre-change HEAD, not
`legacy/main`, so this slice's effect is isolated):
7. **MPS** (`hydra-mps`, this box) and **CUDA** (`hydra-cuda`, mehek).
8. `fly_obb`, `worm_bgsub`, `ant_pose_headtail`, `ant_obb_sleap`,
   `ant_obb_sequential`, `ant_cnn_identity` → **byte-identical** on both
   `_forward.csv` and `_tracking_final.csv`, at the determinism floor (modulo
   the documented bistable head/tail π-flips).
9. `emi_obb_identity` (the AprilTag clip) → positions and angles byte-identical;
   **tag columns diffed column-wise and reported**, not asserted equal. A
   difference confined to tag columns confirms the change landed where intended;
   a difference anywhere else is a regression.

## Risks

- **AprilTag decode rate could drop** at `0.0` if the lab's tags sit close to
  the OBB edge. Mitigated by the new dial and by step 9 reporting the actual
  per-clip tag-column delta rather than hiding it behind an equivalence verdict.
- **A caller outside `src/` passing `padding_fraction`** to
  `extract_and_classify_batch` or `OrientedVideoExporter` breaks at import/call
  time rather than silently. Intended; the grep gate finds in-repo cases.
- **Cache invalidation** costs one recompute on the first run after merge for
  pose and AprilTag caches on existing projects.
