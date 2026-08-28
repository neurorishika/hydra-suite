# DetectKit Multi-Level Canvas Visualization — Design Spec (Part B)

> **Status:** APPROVED, ready for planning.
> **Decided:** 2026-08-27.
> **Scope:** Part B of the DetectKit source-unification effort (see
> `docs/superpowers/specs/done/2026-08-27-detectkit-source-unification-design.md`'s "Design
> note: Part B" for the original lighter-weight sketch this spec formalizes). Runs in parallel
> with Part C (`docs/superpowers/specs/2026-08-27-detectkit-clear-labels-design.md`).

## Goal

For the image currently displayed in DetectKit's canvas, render every geometry level from the
source's native level down to AABB — not just the native level, which is all the canvas shows
today — each level visually distinguished, so a polygon-native source shows its polygon *and*
its derived OBB *and* its derived AABB outline in one view.

## Motivation (verified this session)

- **The user's screenshot symptom ("colored dots, no box outlines") is a real, currently-live
  bug, root-caused before this spec was written** (per Part B's own investigation
  prerequisite): `gui/utils.py::parse_obb_label` hardcoded a strict `len(parts) == 9` field
  check, silently dropping every 5-field AABB line and every variable-length polygon line —
  the exact shapes `data/al/labels.py`'s exporter (and, after Part A, an *accepted* SAM2
  escalation promoted in place) writes. **Already fixed and merged to `main`** as a standalone
  hotfix (independent of this spec, per the brainstorming decision to decouple an active bug
  from this larger feature's timeline) — `parse_obb_label` now handles all three line shapes
  DetectKit's own sources and the AL exporter can produce. This spec's new rendering builds on
  top of that fix, not around it.
- **`OBBCanvas` has exactly two hardcoded layers today** — GT (solid line) and Pred (dashed
  line) — an orthogonal axis (ground-truth vs. model output), not a geometry-level axis. There
  is no per-detection level tag, no N-layer item-list pattern, and no derivation logic anywhere
  in `canvas.py`. This is new capability, not a generalization of an existing one.
- **Corrected during this session's adversarial review:** `source_import.py::_points_to_min_area_rect`
  is **dead code** as of Part A — `grep -rn "_points_to_min_area_rect" src/ tests/` finds only
  its own `def`, zero call sites. Part A's COCO segmentation-conversion path
  (`_coco_annotation_to_points`) now preserves the full contour and never calls minAreaRect. The
  min-area-rect math this spec needs (for deriving a polygon-native source's OBB outline) has no
  existing home to extract from — it's new capability for canvas rendering, not a reuse of
  existing logic. `_points_to_min_area_rect` is deleted outright as part of this spec's work
  (Part B's plan), and `min_area_rect_quad` becomes the sole implementation.

## Decisions (locked during brainstorming)

1. **Per-level styling:** polygon = translucent shaded/filled region; OBB = dashed outline;
   AABB = solid outline. If the *source* is unreviewed (`OBBSource.reviewed is False`), its
   **native-level** shape renders as a striped/hatched region instead of its normal per-level
   style — "needs review" is more urgent to see than level styling for the shape that's
   actually the reviewable data. Derived (non-native, lower) levels always render in their
   normal per-level style regardless of the source's reviewed state, since they're a
   visualization aid, not independently-reviewable data.
2. **One combined "Show derived levels" toggle**, not one checkbox per level. The native level
   is always shown (it's the real data); the toggle controls whether derived lower levels are
   drawn at all. Default on. This keeps the per-level styling from Decision 1 as the primary
   visual differentiator, and avoids a cluttered per-level checkbox UI for a first cut — the
   underlying architecture (an ordered per-level layer list) still supports finer-grained
   per-level toggling later without a redesign, it's just not exposed as separate UI controls
   yet.
3. **Derivation module lives in `hydra_suite/utils/`** (`utils/geometry_derivation.py`, alongside
   the existing `utils/geometry.py` and `utils/slice_geometry.py` — precedent for shared,
   Qt-free geometry math living there). Corrected during adversarial review: an earlier draft of
   this decision put the module in `detectkit/gui/`, which would have violated CLAUDE.md's
   dependency-direction rule (Core/Runtime/Data/Training/Utils must never import from an
   app-layer package) the moment any non-GUI code needed this math, and the "name collision"
   justification for that placement didn't hold up (`geometry_derivation.py` doesn't collide
   with `utils/geometry_levels.py` or `training/geometry_levels.py`, which classify line formats
   into a `GeometryLevel`, not derive points). `utils/` is importable from every app layer,
   including `detectkit/gui/canvas.py`.
4. **Only the native level's shape carries a text label** (class name); derived (non-native)
   levels render as pure outline/fill with no duplicate overlapping text. Locked during planning
   (not originally in this Decisions list) to avoid 2-3 overlapping text labels per detection
   when multiple levels draw for the same shape.

## Architecture

```
src/hydra_suite/utils/geometry_derivation.py              # NEW, Qt-free
    min_area_rect_quad(points) -> list[(x, y)] | None     # same coordinate space in and out
    axis_aligned_bbox_quad(points) -> list[(x, y)] | None  # plain bbox, same space in/out

src/hydra_suite/detectkit/gui/source_import.py
    _points_to_min_area_rect(points, width, height)       # DELETED: dead code (zero call sites
                                                            # as of Part A -- see Motivation)

src/hydra_suite/detectkit/gui/canvas.py
    OBBCanvas._draw_detections(...)                        # MODIFIED: takes a style descriptor
                                                            # (pen style, fill brush) instead of
                                                            # a hardcoded line_style, so it can
                                                            # draw filled/hatched shapes too
    OBBCanvas.set_gt_detections_multi_level(                # NEW: replaces the single-layer
        detections, class_names, *,                         # set_gt_detections for the one
        native_level, reviewed,                              # call site that renders a
    ) -> None                                                # source's labels (main_window.py)
    OBBCanvas.set_derived_levels_visible(bool) -> None       # NEW: the "Show derived levels"
                                                              # toggle's effect
    # Internal: per-level parallel item lists (native + each derived level below it),
    # generalizing the existing _gt_obb_items/_gt_label_items/_gt_class_ids triple into a
    # dict keyed by GeometryLevel. set_gt_detections (the old single-layer API) stays as a
    # thin backward-compatible wrapper for set_pred_detections's continued single-layer use
    # (predictions are model output, not stored labels -- they have no "native level" to
    # derive from, and are out of scope for this spec).

src/hydra_suite/detectkit/gui/main_window.py
    show_image(...)                                          # MODIFIED: look up the current
                                                              # image's OBBSource (by
                                                              # source_path, same pattern as
                                                              # dataset_panel.py's
                                                              # _selected_source_obj) to get
                                                              # level/reviewed, pass to
                                                              # set_gt_detections_multi_level

src/hydra_suite/detectkit/gui/panels/tools_panel.py
    # NEW: "Show derived levels" checkbox in the existing Overlay settings group (alongside
    # whatever GT/Pred visibility controls already live there), wired to
    # OBBCanvas.set_derived_levels_visible via the existing overlay_settings_changed signal
    # path (get_overlay_settings/set_overlay_visibility) -- no new signal needed, this is an
    # additional field on the existing overlay-settings plumbing.
```

## Derivation logic

Given a detection's `polygon_px` (from `parse_obb_label`, already pixel-space — an N-point
polygon if the source's native level is `polygon`, else a 4-point quad) and the source's native
`GeometryLevel`:

- **native = polygon:** native shape = the raw N-point polygon (drawn translucent-filled).
  Derived OBB = `min_area_rect_quad(polygon_px)` (dashed). Derived AABB =
  `axis_aligned_bbox_quad(polygon_px)` (solid).
- **native = obb:** native shape = the raw quad (drawn dashed — it's already an oriented box, no
  polygon fill to show). Derived AABB = `axis_aligned_bbox_quad(polygon_px)` (solid, the bbox of
  the OBB's own 4 corners).
- **native = aabb:** native shape = the raw quad (drawn solid). No derivation — AABB is the
  lowest level, nothing below it to show.

If the source is unreviewed, substitute the native-level shape's draw call with the
striped/hatched style (`Qt.BrushStyle.BDiagPattern` or similar), keeping derived levels in their
normal style per Decision 1.

## `min_area_rect_quad` / `axis_aligned_bbox_quad`

```python
def min_area_rect_quad(
    points: Sequence[tuple[float, float]]
) -> list[tuple[float, float]] | None:
    """4 corners of the minimum-area rotated rect enclosing *points*, in the
    same coordinate space as the input (pixel or normalized -- caller's
    choice, this function is unit-agnostic). None if fewer than 3 points."""

def axis_aligned_bbox_quad(
    points: Sequence[tuple[float, float]]
) -> list[tuple[float, float]] | None:
    """4 corners of the axis-aligned bounding box of *points*, same
    coordinate space as input. None if points is empty."""
```

`source_import.py::_points_to_min_area_rect` is deleted outright (dead code, see Motivation) —
there is no wrapper to preserve. `min_area_rect_quad`/`axis_aligned_bbox_quad` in
`utils/geometry_derivation.py` are the only implementation, consumed solely by `canvas.py` for
this spec.

## Testing

- Pure-function: `min_area_rect_quad`/`axis_aligned_bbox_quad` on known point sets (a rotated
  rectangle's corners round-trip through `min_area_rect_quad` close to themselves; an
  axis-aligned bbox is exact); `<3`/empty-point edge cases return `None`.
- `source_import.py`'s existing COCO-conversion tests must still pass unchanged after
  `_points_to_min_area_rect`'s deletion — they never called it (confirmed dead code), so this is
  a true no-op regression check, not exercise of the deleted function.
- Canvas: a new test constructing a polygon-native detection and asserting
  `set_gt_detections_multi_level` produces 3 layers (polygon/obb/aabb) with the right styles; an
  OBB-native detection produces 2 layers (obb/aabb); an AABB-native detection produces 1 layer.
  An unreviewed source's native-level item uses the hatched brush; a reviewed one doesn't.
  `set_derived_levels_visible(False)` hides derived layers' items but not the native one. The
  pre-existing single-layer `set_gt_detections`/`set_overlay_visibility` path (used by
  `test_detectkit_canvas_dual_layer.py`) must keep working unchanged — `_apply_visibility` must
  branch on whether per-level state was ever populated, not unconditionally assume it was.
- `main_window.py`'s `show_image` wiring: a smoke test that it resolves the current source's
  `level`/`reviewed` and calls `set_gt_detections_multi_level` with them (not the old
  single-layer `set_gt_detections`), including a behavioral (not just source-text) check that
  the resolved values are actually correct for a given `OBBSource`.
- `tests/test_detectkit_canvas.py` does not currently set `QT_QPA_PLATFORM=offscreen` (only
  `pytest.importorskip("PySide6")`) — the plan adds that guard as part of this work, matching
  `test_detectkit_canvas_dual_layer.py`'s existing convention, rather than assuming it already
  exists.

## Out of scope

- Per-level visibility toggles beyond the one combined "Show derived levels" switch (Decision
  2) — the architecture supports it, the UI doesn't expose it yet.
- Any change to the Pred (model-prediction) layer — predictions have no stored "native level"
  to derive from; multi-level rendering is a GT-only concept in this spec.
- User-configurable per-level colors/styles (considered and explicitly declined during
  brainstorming in favor of the fixed styling in Decision 1).
- Any change to how labels are stored on disk, imported, or escalated — this spec is
  visualization-only, reading whatever a source's canonical `labels/` already contains.
