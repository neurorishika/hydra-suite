# Geometry-escalation SAHI + split calibration/escalation source selection

**Status:** Design — pending implementation plan
**Date:** 2026-10-03

## Problem

DetectKit's semantic escalation (SAM3) tiles frames (SAHI) and calibrates the
tile fraction against the user's labelled frames. Geometry escalation (SAM2,
`detectkit/jobs/sam2_escalation.py`) does neither: it calls `set_image` on the
whole frame, and `SAM2ImagePredictor` resizes that frame to 1024 px. On a
4512² frame a ~50 px animal becomes a ~11 px box prompt, which is the main
reason primed masks are poor or fall back to the OBB.

Separately, the semantic dialog uses ONE source list for both calibration and
escalation, and calibrates against labels at any geometry level. A box is not
ground truth for a mask: calibration is only meaningful against polygons.

## Decisions (agreed with the user)

1. **Unify at the tile-plan layer, not the collect/merge layer.** SAM3 is
   detect-then-merge (every tile runs, cross-seam duplicates merged,
   confidence is an axis). SAM2 is prompt-per-known-box (nothing to find,
   nothing to merge). Both share tile sizing/planning; SAM2 gets its own
   owner-tile execution. Routing SAM2 through `collect_candidates` /
   `merge_candidates` is rejected. Per-instance crop windows are rejected
   (encoder cost scales with instance count; shares nothing).
2. **Calibration needs real ground truth: polygon frames only — for BOTH
   semantic and geometry.** Box-only projects lose SAM3 calibration; that is
   accepted (scoring a mask against a box was never ground truth).
   Polygon sources are assumed user-reviewed; no provenance filtering.
3. **Both dialogs get two source selectors on the left side:** top =
   "Calibrate on" (only sources with polygon frames), bottom = "Escalate"
   (all sources). Each action reads its own list.
4. **Uncalibrated geometry default = full frame** (`tile_fraction=None`),
   byte-identical to today. SAM2 must NOT borrow
   `SEMANTIC_TILE_FRACTION_SEED` (`resolve_tile_px` documents that fractions
   do not transfer between models).

## Design

### 1. Shared tiling settings

New frozen dataclass `TilingSettings(reference_body_px, tile_fraction,
tile_px, overlap)` in `core/inference/semantic/tiling.py` (re-homing to a
neutral `core/inference/tiling.py` is a later, optional move). Embedded by
`EscalationRequest` (default `tile_fraction=None`) and
`SemanticEscalationRequest` (existing fields kept as properties/aliases so
positional construction and the request/params/JSON round trip keep working).
SAM3-only settings stay on the semantic request: `seam_margin_px`,
`merge_iou`, `confidence`, `max_instances`, `area_*`.

### 2. SAM2 owner-tile execution — `core/inference/sam2/tiling.py` (Qt-free)

```
assign_owner_tiles(boxes, plan) -> (dict[tile_index, list[box_index]], unowned: list[box_index])
segment_frame(executor, image, boxes, plan) -> list[(mask_frame_bool | None, iou)]
```

- A box is owned by the tile that fully contains its AABB with the largest
  minimum margin; ties → lowest tile index (deterministic).
- `set_image` runs once per tile that owns ≥1 box; empty tiles are skipped.
- Prompts (box, positive centre, negative neighbour centres from
  `build_prompts`) are offset into tile coordinates. Negative points that
  fall outside the owner tile are dropped (SAM2 cannot use them).
- The returned tile mask is pasted into a frame-sized mask before the
  existing `clip_mask_to_polygon` → `mask_to_contour` path, so staging,
  fallback and class-id handling in `run_escalation` are unchanged.
- **Seam fallback:** boxes no tile fully contains are segmented with ONE
  lazy full-frame `set_image` for that frame (today's behaviour), counted in
  a new `EscalationResult.seam_fallbacks`.
- `full_frame_plan` (no tiling) yields one tile owning every box → identical
  call sequence to today. This is the byte-identity guard.

`run_escalation` replaces its inline `set_image`/`segment` loop with
`segment_frame`; tile plan resolved per frame via `resolve_tile_px` +
`plan_for_frame`/`full_frame_plan`, exactly as the semantic job does.

### 3. Neutral home for shared calibration helpers

Move `measure_median_body_px`, `labelled_frames_for`,
`stratified_calibration_frames`, `has_labelled_frames`, `_label_path_for`
from `jobs/semantic_escalation.py` to `detectkit/jobs/calibration_frames.py`;
re-export from the old module (existing importers keep working).

Add the polygon criterion there:
- A label line is a polygon when its coordinate count is even, >= 6 and
  not 8 (AABB = 4, OBB = 8; the AL writer pads 4-point polygons so a
  polygon line never has 8).
- A frame is ground truth only if EVERY line in it is a polygon. Mixed
  frames are excluded: their box-labelled animals could not be scored.
- `has_polygon_frames(source) -> bool` — label-FILE scan (no image decode):
  true if the source has at least one such frame.
- `labelled_frames_for(..., polygon_only=True)` /
  `stratified_calibration_frames(..., polygon_only=True)` — only such
  frames. Default `False`, so direct/YOLO calibration is unchanged.

### 4. Geometry calibration — `core/inference/sam2/calibration.py` (Qt-free)

Per sampled polygon frame (from the "Calibrate on" list, via
`stratified_calibration_frames(polygon_only=True)`):

1. Derive the prompt from ground truth with `data/al/escalation.derive_down`
   (polygon → OBB), i.e. exactly what a box source would supply.
2. For each fraction in `candidate_tile_plans(frame_hw, reference_body_px)`
   (full frame always included) run `segment_frame`, then the same
   clip/contour step as escalation.
3. Score: pairing is 1:1 by construction (one box in, one mask out), no
   matcher. Per instance: `polygon_iou(contour, gt_polygon)`; contour `None`
   counts as fallback.

```
@dataclass(frozen=True)
class GeometryCalibrationPoint:
    tile_fraction: float | None
    tile_px: int | None
    owned_tiles_per_frame: float   # the real cost axis (empty tiles skipped)
    seconds_per_frame: float       # measured
    median_iou: float
    p10_iou: float
    fallback_rate: float
    seam_fallback_rate: float
    n_instances: int
```

`recommend_geometry(points)`: eligibility filters `n_instances ≥ MIN`,
`median_iou ≥ IOU_FLOOR`, `fallback_rate ≤ FALLBACK_CEIL`; among eligible,
fewest `owned_tiles_per_frame`, ties → higher `median_iou`. Refusal returns a
reason string, same contract as semantic `recommend`. **`IOU_FLOOR` and
`FALLBACK_CEIL` are set from one measurement on real polygon sources during
implementation**, not invented here; the measurement is recorded alongside.

Worker: `Sam2CalibrationWorker(BaseWorker)` in `detectkit/jobs/`, honours
cancellation between frames/fractions.

### 5. Persistence

SAM2 checkpoints are public HF weights with no sidecar, so the result is
PROJECT-only: `DetectKitProject.geometry_calibration: dict[variant, record]`
(new field, round-tripped like `semantic_calibration`). Record: `created_at`,
points, chosen point, `calibration_source_paths`, calibration
`median_body_px`, frames sampled/truncated. Semantic persistence (project +
SAM3 sidecar) is unchanged in shape.

Persisted dialog settings split `source_names` into
`calibration_source_names` + `escalation_source_names`; a legacy
`source_names` is read as `escalation_source_names`.

### 6. Shared dialog layout — `CalibrationSourceSelector`

New widget `detectkit/gui/widgets/calibration_source_selector.py` (first file
in a new `detectkit/gui/widgets/`), used by `SemanticEscalationDialog` and
`EscalateSam2Dialog`. Left column, two stacked groups:

- **Calibrate on** — only sources where `has_polygon_frames` is true; all
  checked by default. Empty → message "No polygon ground truth in this
  project. Label polygons on a few frames to calibrate." and the dialog's
  Calibrate button disabled.
- **Escalate** — every source. In the SAM2 dialog, polygon-level sources are
  shown disabled with tooltip "Already polygon" (run_escalation skips them).

API: `calibration_sources()`, `escalation_sources()`, signals
`calibration_changed` / `escalation_changed`, `restore(saved)`, `state()`.

Wiring per dialog:
- Semantic: `_run_calibration` reads `calibration_sources()` (replaces
  `selected_sources() or self._sources`); escalation run, frame count and
  ETA read `escalation_sources()`; the complete-frame preview samples
  escalation sources (it estimates run time on the targets) and shows GT
  overlays when the frame has polygons. `reference_body_px` prefill
  (project setting → median) measures the calibration sources.
- SAM2: right side gains a tile-fraction control (Full frame + grid
  values, prefilled from `geometry_calibration[variant]` if present), a
  "Calibrate against polygon frames…" button, and a compact results table
  (one row per fraction: tiles/frame, s/frame, median IoU, p10 IoU,
  fallback %). It reuses the calibration preview-frame rendering, not
  `CalibrationResultsDialog` (whose rows are confidence-sweep shaped).

**Scale-transfer warning (both dialogs):** tile fraction is relative to body
size, so when median body px of calibration vs escalation sources differs by
more than 1.5× (constant, tunable), show a non-blocking warning ("calibrated
on frames at a different scale"). Escalation sources without labels skip the
check.

### 7. Out of scope

Direct/YOLO calibration (`direct_calibration*`); moving semantic tiling
helpers to a neutral package; multi-scale SAM2; any change to SAM3
collect/merge.

## Error handling

- No polygon frames among calibration sources → Calibrate disabled (GUI);
  worker raises a clear error if invoked headless.
- `plan_for_frame` ValueError (tile ceiling) → that fraction is skipped in
  calibration (as `candidate_tile_plans` already does); at escalation time it
  falls back to full frame for that frame and is reported, never silent.
- Cancelled calibration → no record written; cancelled escalation keeps
  existing staging semantics.

## Testing

- `assign_owner_tiles`: interior box, seam box (unowned), margin tie-break,
  empty tiles skipped.
- `segment_frame` with a fake executor: coordinate round trip (tile mask
  pasted to correct frame location), negative points outside tile dropped,
  one `set_image` per owned tile, seam fallback issues exactly one
  full-frame `set_image`.
- **Byte-identity:** `run_escalation` with `tile_fraction=None` produces
  staged label files identical to current main and an identical executor
  call sequence.
- Geometry calibration on a synthetic polygon source with a fake executor
  returning the GT mask (IoU 1) vs a shrunken mask; `recommend_geometry`
  refusal paths.
- `has_polygon_frames` / `polygon_only` parsing: AABB, OBB, polygon, mixed
  files.
- Semantic calibration now refuses box-only sources (behaviour change test).
- Dialog: selector population/filtering, legacy `source_names` restore,
  each action reads its own list, scale warning threshold.
- Equivalence harness: unaffected (TrackerKit pipeline untouched); not
  required.
- One real measurement (GPU) on polygon sources to set `IOU_FLOOR` /
  `FALLBACK_CEIL` and confirm owner-tile SAM2 beats full frame on small
  animals.

## Measurement (2026-10-03, mehek, RTX 6000 Ada, `hydra-cuda`)

**Data:** the user-reviewed polygon source `merged_source_20260914`
(improved_ant_detection; 117 frames, all frames fully polygon-labelled,
4512×4512). 30 frames sampled with `stratified_calibration_frames(polygon_only=True)`,
697 instances. Median body 79.0 px. Overlap 0.5. Tool:
`tools/sam2_geometry_calibration/measure.py`; raw JSON in
`tools/sam2_geometry_calibration/results/` (commit 14bb02ea).

`sam2.1-hiera-base_plus` (dialog default):

| fraction | tile px | encodes/frame | s/frame | median IoU | p10 IoU | fallback | seam |
|---|---|---|---|---|---|---|---|
| full | — | 1.00 | 1.65 | 0.598 | 0.519 | 0 | 0 |
| 0.02 | 3950 | 1.73 | 1.34 | 0.601 | 0.524 | 0 | 0 |
| 0.03 | 2633 | 2.50 | 0.46 | 0.609 | 0.534 | 0 | 0 |
| 0.05 | 1580 | 5.17 | 0.37 | 0.635 | 0.559 | 0 | 0 |
| 0.10 | 790 | 6.67 | 0.38–0.41 | 0.683 | 0.599 | 0 | 0 |
| 0.20 | 395 | 10.30 | 0.51–0.55 | 0.729 | 0.633 | 0 | 0 |
| 0.30 | 263 | 14.67 | 0.74 | 0.756 | 0.636 | 0 | 0 |
| 0.40 | 197 | 17.70 | 0.90 | 0.761 | 0.623 | 0 | 0.009 |
| 0.50 | 158 | 18.13 | 1.18 | 0.753 | 0.561 | 0 | 0.145 |

(0.70 → 113 px tiles exceeds `MAX_TILES_PER_FRAME` on 4512 px frames and is
dropped.) `sam2.1-hiera-tiny` shows the same shape: full 0.598 → 0.752 at 0.20.

**Answer: owner-tile SAHI beats full frame on small animals.** Median IoU
0.598 → 0.756 (+0.16), p10 0.519 → 0.636, and it is also FASTER
(1.65 → 0.74 s/frame), because SAM2's per-pass cost scales with the pixels
it is handed, not with the number of passes.

**Deviations from the design, forced by the data:**

1. *Cost axis.* The design ranked by owned tiles per frame. Encode count
   anti-correlates with time here (full frame = 1 pass, slowest), so ranking
   by it recommended full frame — the worst and slowest configuration.
   `recommend_geometry` now ranks by quality first, then MEASURED
   `seconds_per_frame`: among eligible points within `IOU_TOLERANCE = 0.02`
   of the best median IoU, the fastest wins. On this sweep that is 0.30.
2. *Grid.* SAM3's `TILE_FRACTION_GRID` (0.03/0.05/0.10/full) never reaches
   the region where SAM2 wins. SAM2 gets `SAM2_TILE_FRACTION_GRID =
   (0.05, 0.1, 0.2, 0.3, 0.4, full)`; it stops at 0.4 because at 0.5
   (tile = 2× body) 14.5% of animals cross every seam and p10 IoU drops.

**Constants:**

- `IOU_FLOOR = 0.45` — a "SAM2 is broken here" gate, set below the worst
  working configuration measured (full frame, 0.598) with margin. Ranking,
  not the floor, picks the operating point.
- `FALLBACK_CEIL = 0.10` — **not measurable from this data**: no
  configuration fell back on a single animal. Kept as a generous sanity
  bound.
- `IOU_TOLERANCE = 0.02` — separates 0.729 (0.2) from the 0.756/0.761
  plateau (0.3/0.4).

**End-to-end check (same box, commit 5af2dc8c):** 8 held-out polygon frames
(indices 40-47) reduced to OBB labels and escalated through
`run_escalation` with the real `sam2.1-hiera-base_plus` executor; staged
polygons scored against the original polygons. Full frame: median IoU
0.597, 195/195 primed, 14.7 s. Fraction 0.3 (263 px tiles): median IoU
0.756, 195/195 primed, 0 seam fallbacks, 9.7 s.
