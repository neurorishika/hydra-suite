# Multi-Arena UI/UX Redesign — Design Spec

**Status:** Design approved, pending spec review
**Date:** 2026-08-21
**Supersedes UI portions of:** `docs/superpowers/specs/done/2026-08-18-multi-arena-tracking-design.md`

## Problem

Multi-arena tracking shipped (merge `a2c9838f`) with a UI grafted onto the
existing ROI toolbar. Three concrete failures:

1. **The ROI-role vocabulary leaks into the arena workflow.** The toolbar
   presents ten always-visible widgets (`main_window.py:575-673`), including an
   `Include Zone / Exclude Zone` combo that the user must set *before* drawing.
   Arena membership is a separate, invisible mode advanced by a `New Arena`
   button. Nothing tells the user what state they are in.
2. **Arena rendering is near-invisible.** Shapes draw as `QPen(Qt.cyan, 2)` in
   *image* pixel space (`session.py:1986` `update_roi_preview`,
   `session.py:2395` `_draw_roi_overlay`). There is no region fill and no arena
   number, so with more than a handful of arenas the display is unreadable.
3. **Zoom is disabled while drawing.** `start_roi_selection` calls
   `slider_zoom.setEnabled(False)` (`session.py:2100`) and
   `_handle_video_wheel` calls `evt.ignore()` when `roi_selection_active`
   (`session.py:1790`). Users must draw every ROI at 100% zoom.

### Root cause shared by (2) and (3)

All overlay drawing paints into the **image pixels**: a `QPainter` runs on the
frame `QImage`, and the result is scaled afterwards by the zoom slider.

- A 2-pixel pen is 2 *image* pixels, so apparent width scales with zoom — it
  vanishes when zoomed out and goes chunky when zoomed in.
- `record_roi_click` (`session.py:1950`) stores `evt.position()`, a *label*
  coordinate, directly as an *image* coordinate. Those agree only at 100% zoom,
  which is precisely why zoom is force-disabled during selection.

Zoom-invariant line weight and draw-while-zoomed are therefore the same defect,
not two features. Painting the overlay in **viewport space** through an explicit
image-to-viewport transform fixes both.

### Adjacent correctness hazard

`build_arena_labels` (`engine_params.py`) resolves overlapping arenas by
last-writer-wins in shape draw order, silently. Two arenas that overlap produce
a label image where the later one wins the shared pixels, with no warning. The
redesign makes overlap a visible, blocking condition.

## Global Constraints

- `roi_shapes` keeps its exact existing schema:
  `{"type": "circle"|"polygon", "params": ..., "mode": "include"|"exclude", "arena_id": int}`.
  No new shape type, no new field, no separate storage for generated shapes.
- Tracking output must be byte-identical. This work is display and shape
  authoring only; verified with the MPS equivalence gate, not asserted.
- Pure geometry and style logic must be importable without Qt. Some GUI tests
  abort the interpreter (memory `project_main_suite_blockers`), so logic that
  needs coverage cannot live behind a widget import.
- The single-arena case is unchanged end to end: one arena means
  `n_arenas == 1`, which means no `arena_id` CSV column, exactly as today.
- Implementation happens in a git worktree branched from local `HEAD`.

## Decisions (adjudicated with the user)

| Question | Decision |
|---|---|
| Overlap handling | Warn and block. No automatic geometry mutation. |
| Overlap lock scope | Block Previous/Next/Add-new only while the **current** arena overlaps. Block tracking whenever **any** pair overlaps. |
| `Clear Arena` | Empties the arena's shapes; the arena remains and numbering is untouched. |
| Displaced controls | Undo stays in the bar. Clear All, Crop Video to ROI, Add Grid move to an overflow menu. Include/Exclude combo is deleted. |
| Grid row/column cap | Every arena's **centre** must lie inside the frame. |
| Veil polarity and alpha | Inside the ROI, alpha `0.15`. |
| Line colour | Luminance-driven light/dark variants of a fixed per-role palette. |
| Arena number legibility | Contrasting halo, composited from an offscreen full-opacity layer. |
| Rendering architecture | Dedicated `ArenaCanvas(QWidget)` with a viewport-space overlay. |
| Extra scope accepted | Click an arena to select it; arena numbers in the grid preview. |
| Grid rotation | Slider + linked spinbox, −45°..+45°, 0.5° steps, pivoting on arena 1's centre. |
| Click mapping | Left-click adds, right-click removes the most recent point (today's mapping, retained). |

### Corrected from the original brief

The brief specified default grid spacing as "the minimum of radius/2, or
width/2, height/2, so that there can be NO overlap". That produces the opposite
result: circles of radius `r` avoid overlap only when centre-to-centre spacing
is at least `2r`; at `r/2` every arena overlaps its neighbour heavily. The
design implements the stated *intent* — default to the tightest spacing that
cannot overlap — with the correct values: `2*radius` for circles, `width` in x
and `height` in y for rectangles. These are also enforced as spinbox minimums,
so the grid generator cannot emit a layout the overlap lock would reject.

The brief also specified "right-click marks, left-click removes". The user
reverted this to the conventional mapping during review.

## Architecture

### New modules

**`trackerkit/arena_geometry.py`** — Qt-free, pure.

```
shape_centroid(shape) -> tuple[float, float]
point_in_shape(shape, x, y) -> bool
arena_at_point(shapes, x, y) -> int | None      # last-drawn include arena, minus excludes
overlapping_arena_pairs(shapes, width, height) -> list[tuple[int, int]]
generate_grid_shapes(...) -> list[dict]          # moved from arena_grid_dialog, + rotation
max_grid_extent(...) -> tuple[int, int]          # (max_rows, max_cols)
```

`generate_grid_shapes` moves here from `arena_grid_dialog.py` so grid maths and
overlap maths sit together and neither requires Qt to test.

**`trackerkit/gui/widgets/arena_style.py`** — Qt-free, returns plain RGBA tuples.

```
frame_palette(mean_luminance: float) -> ArenaPalette
line_width_px(viewport_min_dim: int) -> int
glyph_size_px(on_screen_radius: float) -> int    # clamped to [10, 64]
VEIL_ALPHA = 0.15
TEXT_ALPHA = 0.70
CLICK_DRAG_THRESHOLD_PX = 3
```

**`trackerkit/gui/widgets/arena_canvas.py`** — `ArenaCanvas(QWidget)`.

Owns the frame pixmap, zoom factor, shape list, current arena id, and
in-progress point list. Overrides `paintEvent`: paints the scaled frame, then
paints the overlay in widget coordinates. Reports `setFixedSize(scaled_size)` so
the existing `QScrollArea` continues to handle panning unchanged. Exposes
`to_image(pt)` / `to_viewport(pt)` and emits `arena_clicked(int)`,
`point_added(x, y)`, `point_removed()`.

**`trackerkit/gui/panels/arena_panel.py`** — the arena bar and its state machine.

### Modified

- `arena_grid_dialog.py` — shape/rotation/spacing controls, caps, shared renderer.
- `orchestrators/session.py` — ROI methods delegate to the canvas; the three
  `slider_zoom.setEnabled(False)` / `evt.ignore()` zoom locks are removed.
- `main_window.py` — `video_label` becomes an `ArenaCanvas`; the ten-widget ROI
  toolbar row is replaced by `ArenaPanel`.

## Arena bar

**Empty state** — the only things visible:

```
By default, the whole video is used.   [+ Add Single Arena]  [+ Add Grid of Arenas]
```

**Editing state:**

```
Currently labelling: Arena 3   [< Previous] [Next >] [+ Add new arena]
  |  Add Inclusion and Exclusion Zones (Left-click marks, Right-click removes the last point)
  |  [Clear Arena] [+ Circle] [- Circle] [+ Polygon] [- Polygon] [Finish Shape]
  |  [Undo]  [...]
```

`[...]` holds Add Grid, Clear All, and Crop Video to ROI.

`[Finish Shape]` is enabled only for a valid shape: at least 3 points, and for
circles a successful `fit_circle_to_points`. Previous/Next disable at the ends.
`[Clear Arena]` removes that arena's include and exclude shapes and keeps the
user on the now-empty arena.

Adding the first arena transitions Empty to Editing. Returning to Empty happens
only via Clear All.

One arena is exactly today's plain ROI: a user who only wants to mask out junk
draws one arena and never thinks about arena numbering, because `n_arenas == 1`
suppresses the `arena_id` column and every downstream per-arena code path
degenerates to the single-arena case. "Arena" and "ROI" stop being two concepts.

`+ Add new arena` is disabled while the current arena has no shapes, so the user
cannot accumulate a run of empty arenas that would inflate `MAX_TARGETS`
(`n_arenas * animals_per_arena`, `engine_params.py:477`) with slots no detection
can ever occupy.

## Interaction

| Gesture | Action |
|---|---|
| Left-click, displacement < `CLICK_DRAG_THRESHOLD_PX` | Add point |
| Left-drag, displacement >= threshold | Pan |
| Right-click | Remove most recent point |
| Middle-drag | Pan |
| Wheel / modifier-wheel | Zoom at cursor — remains live while drawing |
| Double-click (polygon mode) | Finish shape |
| Esc | Cancel the in-progress shape |
| Click inside an arena (not drawing) | Make that arena current |

Press-to-release displacement decides click versus pan, so one button serves
both: a deliberate point-click never pans, and a deliberate drag never drops a
stray point. The threshold is a named constant with a test — too small and a
shaky hand pans instead of marking; too large and short drags drop points.

The canvas suppresses its default context menu while a shape is in progress. On
macOS, Ctrl+left-click arrives as a right-click, so "remove last point" is
available on trackpads without configuration.

## Overlap detection and lock

`overlapping_arena_pairs` returns every pair of arena ids sharing at least one
pixel, after exclude zones are subtracted.

Performance matters: a 96-well plate is 4560 candidate pairs, and the reference
fixture is 4512x4512, so naive full-frame mask intersection is not viable. The
implementation is staged:

1. Analytic circle-circle test when both arenas are single circles
   (`dist < r1 + r2`).
2. Axis-aligned bounding-box rejection for every other pair.
3. Rasterized intersection only for surviving candidates, cropped to the
   intersection bounding box rather than the full frame.

`arena_at_point` resolves ties by draw order, matching `build_arena_labels`'
last-writer-wins rule, so selection agrees with the label image the tracker
actually uses. Ties can only arise while an overlap exists, which the lock
already surfaces.

Stage 1 and 2 are pure rejection filters; stage 3 is authoritative. A test
asserts the fast path agrees with brute-force full-frame rasterization on
randomized layouts, so an optimization bug cannot silently weaken the gate.

**Lock behaviour.** While the current arena overlaps another: Previous, Next,
and Add-new-arena are disabled, and a warning banner names the conflicting
arena. Whenever any pair overlaps anywhere: tracking is blocked and the banner
lists the offending pairs. Because navigation is blocked only by the *current*
arena's conflicts, the user can always reach a conflicting arena to fix it.

## Grid builder

- **Shape:** Circle (Radius) or Rectangle (Width, Height).
- **Top-Left Position X / Y:** the centre of the first arena.
- **Rows / Columns:** start at 1 x 1. X-spacing and Y-spacing spinboxes appear
  only once rows or columns exceed 1.
- **Spacing defaults and minimums:** `2*radius` for circles; `width` in x and
  `height` in y for rectangles. Enforced as spinbox minimums.
- **Rotation:** slider plus linked spinbox, −45°..+45°, 0.5° steps, pivoting on
  the first arena's centre. Sub-degree resolution matters because a small
  angular error compounds across a wide grid.
- **Caps:** rows and columns are capped so every arena's centre lies inside the
  frame. Under rotation the centres form a rotated lattice, so `max_grid_extent`
  finds the cap by direct search rather than a closed form.
- **Output:** ordinary `roi_shapes` with sequential row-major `arena_id`s,
  individually editable afterwards, indistinguishable from hand-drawn shapes.
  Rotated rectangles are emitted as 4-point polygons, an already-supported type.
- **Preview:** uses the shared renderer, including arena numbers — now
  load-bearing, since rotation pivots about arena 1.

## Rendering

The palette derives from the base frame's mean luminance, computed once per
frame and cached. Light footage gets a dark veil, dark glyph, and white halo;
dark footage gets the inverse. Hue is fixed per **role** — include one hue,
exclude red, in-progress green — so the meaning stays learnable across videos
rather than shifting with footage.

- **Veil:** alpha `0.15` over the include-minus-exclude region, inside the ROI.
- **Line weight and glyph size:** device-pixel constants derived from viewport
  size, never image size. This is what makes them zoom-invariant.
- **Arena number:** drawn at the arena's centroid. The glyph and its halo render
  into an offscreen full-opacity ARGB layer, which composites once at alpha
  `0.70`. Stroking and filling separately at partial alpha would double-composite
  where they overlap and let the halo bleed through the glyph edge.
- **Glyph size:** tracks on-screen arena radius, clamped to 10-64 px, so 96 wells
  stay readable and a single large arena does not get an absurd number.
- **Current arena:** heavier outline.

The grid dialog calls this same renderer, so the two previews cannot drift.

## Testing

**Qt-free unit tests** (`arena_geometry`, `arena_style`):
grid generation under rotation including the 0° identity case; `max_grid_extent`
caps at several rotations; the spacing floor; `overlapping_arena_pairs` fast path
versus brute-force rasterization on randomized layouts; touching-but-not-
overlapping arenas reported clean; `point_in_shape` and `arena_at_point`
including a point inside an exclude hole; luminance-to-palette on light and dark
frames; glyph-size clamping at both bounds.

**Canvas tests:** the image to viewport to image round trip is identity at
several zoom levels — the direct regression test for the defect underlying both
reported problems. Click/drag disambiguation at the threshold boundary.

**Panel tests:** Empty to Editing transitions; the lock's enable/disable matrix;
`Clear Arena` preserving the arena and its numbering.

**Equivalence gate:** `main_window.py` and `session.py` are both modified, so the
MPS matrix runs to prove tracking output is byte-identical rather than assuming
it. `roi_shapes` is unchanged in format, so no fixture changes are required.

## Out of scope

- Per-arena detection budgets (still global — a known multi-arena limitation).
- The pre-existing `main` bug where the batched YOLO path does not gate
  detections by ROI.
- Equivalence fixtures carrying real `roi_shapes`; the gates continue to prove
  only the no-op single-arena case.
