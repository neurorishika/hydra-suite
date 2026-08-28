# DetectKit Multi-Level Canvas Visualization (Part B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** DetectKit's canvas renders every geometry level from a source's native level down to
AABB for the displayed image, each in its own style, instead of only the native level.

**Architecture:** A new Qt-free `geometry_derivation.py` module holds the point-derivation math
(extracted from `source_import.py`'s existing COCO-conversion helper). `OBBCanvas` generalizes
its single GT layer into a per-`GeometryLevel` layer dict, each drawn with a level-specific style
(and an unreviewed-native-shape override), gated by a new combined "Show derived levels" toggle
threaded through the existing overlay-settings plumbing.

**Tech Stack:** Python 3.11+, PySide6, pytest, `hydra-mps` conda env for all test runs.

**Spec:** `docs/superpowers/specs/2026-08-27-detectkit-canvas-multi-level-design.md`

## Global Constraints

- `GeometryLevel` is an `IntEnum` (`AABB=0 < OBB=1 < POLYGON=2`, `hydra_suite.utils.geometry_levels`,
  re-exported via `hydra_suite.training.geometry_levels` — this `gui/` package already imports it
  from the `training` re-export elsewhere, e.g. `dataset_panel.py`/`source_import.py`; match that
  convention).
- `OBBSource.level` is a **string** (`GeometryLevel.label`, e.g. `"obb"`), not an enum instance —
  convert with `GeometryLevel.from_str(...)` wherever a source's level needs comparing.
- `OBBCanvas` (`detectkit/gui/canvas.py`) has no consumers outside DetectKit — verified this
  session via repo-wide grep (`trackerkit`'s superficially-similar `ReferenceScalePreviewWidget`
  is an unrelated, independently-defined class). Internal restructuring here cannot break another
  kit.
- Only the **native** level's shape shows a text label (class name); derived (non-native) levels
  render as pure outline/fill with no duplicate overlapping text — avoids label clutter when 2-3
  levels draw for one detection.
- The Pred (model-prediction) layer is untouched by this plan — predictions have no stored
  "native level" to derive from.
- Commit as the configured git user — no `Co-Authored-By: Claude` trailer, no
  `Claude-Session:` line.
- All test runs: `conda activate hydra-mps` first, from this plan's dedicated worktree.

---

## Task 1: `geometry_derivation.py` — shared point-derivation math

**Files:**
- Create: `src/hydra_suite/utils/geometry_derivation.py`
- Test: Create `tests/test_geometry_derivation.py`

**Interfaces:**
- Produces: `min_area_rect_quad(points: Sequence[tuple[float, float]]) -> list[tuple[float,
  float]] | None` and `axis_aligned_bbox_quad(points: Sequence[tuple[float, float]]) ->
  list[tuple[float, float]] | None` — both coordinate-space-agnostic (same units in and out; no
  normalization). Task 3 calls both directly in pixel space from `canvas.py`. Placed in
  `hydra_suite/utils/` (not `detectkit/gui/`) — this repo's dependency-direction rule
  (CLAUDE.md: "Core, Runtime, Data, Training, and Utils must never import from any app-layer
  package") would make a `detectkit/gui/`-hosted module permanently unreusable outside DetectKit;
  `utils/` is importable from every app layer, including `detectkit/gui/canvas.py`. `cv2`/`numpy`
  are safe to import at module level here: `canvas.py` (Task 3's only consumer) already imports
  both at module level, so this module adds no new import cost on its one real call path.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_geometry_derivation.py`:

```python
"""Tests for shared geometry-derivation math (used by DetectKit's canvas rendering)."""

from __future__ import annotations

from hydra_suite.utils.geometry_derivation import (
    axis_aligned_bbox_quad,
    min_area_rect_quad,
)


def test_min_area_rect_quad_axis_aligned_square_returns_its_own_corners():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    box = min_area_rect_quad(points)
    assert box is not None
    assert len(box) == 4
    xs = sorted(p[0] for p in box)
    ys = sorted(p[1] for p in box)
    assert abs(xs[0] - 0.0) < 0.5 and abs(xs[-1] - 10.0) < 0.5
    assert abs(ys[0] - 0.0) < 0.5 and abs(ys[-1] - 10.0) < 0.5


def test_min_area_rect_quad_too_few_points_returns_none():
    assert min_area_rect_quad([(0.0, 0.0), (1.0, 1.0)]) is None
    assert min_area_rect_quad([]) is None


def test_axis_aligned_bbox_quad_returns_exact_bbox_corners():
    points = [(2.0, 5.0), (8.0, 3.0), (6.0, 9.0)]
    box = axis_aligned_bbox_quad(points)
    assert box is not None
    xs = sorted(p[0] for p in box)
    ys = sorted(p[1] for p in box)
    assert xs[0] == 2.0 and xs[-1] == 8.0
    assert ys[0] == 3.0 and ys[-1] == 9.0
    assert len(box) == 4


def test_axis_aligned_bbox_quad_empty_points_returns_none():
    assert axis_aligned_bbox_quad([]) is None


def test_axis_aligned_bbox_quad_of_a_single_point_is_degenerate_but_defined():
    box = axis_aligned_bbox_quad([(5.0, 5.0)])
    assert box is not None
    assert all(p == (5.0, 5.0) for p in box)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_geometry_derivation.py -v`
Expected: FAIL with `ModuleNotFoundError` (the module doesn't exist yet).

- [ ] **Step 3: Create the module**

Create `src/hydra_suite/utils/geometry_derivation.py`:

```python
"""Shared point-derivation math (minimum-area rect, axis-aligned bbox).

Coordinate-space-agnostic: every function here takes and returns points in
whatever space the caller is working in (pixel or normalized [0, 1]) -- no
normalization happens inside this module.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def min_area_rect_quad(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """4 corners of the minimum-area rotated rect enclosing *points*, in the
    same coordinate space as the input. None if fewer than 3 points."""
    if len(points) < 3:
        return None
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    box = cv2.boxPoints(rect).astype(float)
    return [(float(x), float(y)) for x, y in box]


def axis_aligned_bbox_quad(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """4 corners of the axis-aligned bounding box of *points*, same
    coordinate space as input. None if points is empty."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_geometry_derivation.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/utils/geometry_derivation.py tests/test_geometry_derivation.py && isort src/hydra_suite/utils/geometry_derivation.py tests/test_geometry_derivation.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/utils/geometry_derivation.py tests/test_geometry_derivation.py
git commit -m "feat(utils): add shared geometry-derivation module (min-area-rect, bbox)"
```

---

## Task 2: delete dead code (`source_import.py::_points_to_min_area_rect`)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/source_import.py`
- Test: `tests/test_detectkit_source_import.py` (no test changes needed — this task's own
  verification is that the existing suite still fully passes after the deletion)

**Interfaces:** none produced — this task removes a function, it doesn't add one.

**Why this task exists:** the original plan draft assumed `_points_to_min_area_rect` was live
code used by COCO-segmentation conversion, worth refactoring into a thin wrapper around Task 1's
new module. Adversarial review found this premise stale: `grep -rn
"_points_to_min_area_rect" src/ tests/` returns **only its own `def`** — zero call sites. Part A's
COCO-conversion path (`_coco_annotation_to_points`) now preserves the full contour and never
calls minAreaRect. There is nothing to refactor; the function is simply dead and should be
deleted, matching what `make dead-code` should already be flagging.

- [ ] **Step 1: Confirm it's genuinely dead before deleting**

Run: `grep -rn "_points_to_min_area_rect" /path/to/repo/src /path/to/repo/tests` (use this
plan's actual worktree path). Expected: exactly one match — the `def _points_to_min_area_rect(`
line itself, in `source_import.py`. If this turns up ANY other match, STOP — the premise for this
task is wrong, escalate rather than deleting live code.

- [ ] **Step 2: Confirm the baseline (existing tests pass before this change)**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_import.py -v`
Expected: all PASS (baseline — establishes what "no regression" means for Step 4, and confirms
none of these tests currently exercise the function being deleted).

- [ ] **Step 3: Delete the function**

In `src/hydra_suite/detectkit/gui/source_import.py`, delete the entire
`_points_to_min_area_rect` function (currently lines 454-468: the `def` line, its docstring-free
body, the inline `import cv2` / `import numpy as np`, and the `cv2.minAreaRect(...)` /
`cv2.boxPoints(...)` logic — everything from `def _points_to_min_area_rect(` through the final
`return coords` of that function). Leave the surrounding functions (`_coco_segmentation_points`
above it, `_coco_bbox_to_polygon` below it) untouched.

- [ ] **Step 4: Run tests to verify no regression**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_import.py -v`
Expected: all PASS, identical count to Step 2's baseline — proving the deletion touched nothing
any test depended on.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/source_import.py && isort src/hydra_suite/detectkit/gui/source_import.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/source_import.py
git commit -m "refactor(detectkit): delete dead _points_to_min_area_rect (zero call sites since Part A)"
```

---

## Task 3: `OBBCanvas` — per-level GT layers, styling, unreviewed override

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/canvas.py`
- Test: `tests/test_detectkit_canvas.py`

**Interfaces:**
- Consumes: `min_area_rect_quad`, `axis_aligned_bbox_quad` from `hydra_suite.utils.geometry_derivation`
  (Task 1); `GeometryLevel` (existing, `hydra_suite.training.geometry_levels`).
- Produces: `OBBCanvas.set_gt_detections_multi_level(detections, class_names=None, *,
  native_level: GeometryLevel, reviewed: bool = True) -> None` — the new entry point Task 4's
  `main_window.py` wiring calls. `OBBCanvas.set_derived_levels_visible(visible: bool) -> None` —
  the toggle Task 5's checkbox drives. `set_gt_detections` (existing single-layer API) stays as
  a backward-compatible wrapper — no other code in this repo calls it besides
  `main_window.py:1983` (verified this session), which Task 4 changes to call the new method
  instead, but the old method must keep working for any future/manual callers and for
  `set_detections`'s existing alias.

- [ ] **Step 1: Write the failing tests**

First, `tests/test_detectkit_canvas.py` does NOT currently set `QT_QPA_PLATFORM=offscreen`
(confirmed this session — it only has `pytest.importorskip("PySide6")`), unlike its sibling
`test_detectkit_canvas_dual_layer.py`, which does. Add the guard at the very top of the file,
before the `import sys` line: `import os` then `os.environ.setdefault("QT_QPA_PLATFORM",
"offscreen")`, matching the sibling file's convention — this repo has a documented history of
headed-Qt test runs hanging/crashing the suite.

Then add to `tests/test_detectkit_canvas.py`:

```python
def test_set_gt_detections_multi_level_polygon_native_draws_three_layers(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    polygon = [(10.0, 10.0), (50.0, 5.0), (90.0, 40.0), (60.0, 90.0), (20.0, 60.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": polygon}],
        native_level=GeometryLevel.POLYGON,
        reviewed=True,
    )
    assert set(canvas._gt_level_items.keys()) == {
        GeometryLevel.POLYGON,
        GeometryLevel.OBB,
        GeometryLevel.AABB,
    }
    for level in (GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB):
        assert len(canvas._gt_level_items[level]) == 1


def test_set_gt_detections_multi_level_obb_native_draws_two_layers(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.OBB,
        reviewed=True,
    )
    assert set(canvas._gt_level_items.keys()) == {GeometryLevel.OBB, GeometryLevel.AABB}


def test_set_gt_detections_multi_level_aabb_native_draws_one_layer(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.AABB,
        reviewed=True,
    )
    assert set(canvas._gt_level_items.keys()) == {GeometryLevel.AABB}


def test_set_gt_detections_multi_level_unreviewed_uses_hatched_brush_on_native_only(qapp):
    from PySide6.QtCore import Qt

    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}],
        native_level=GeometryLevel.OBB,
        reviewed=False,
    )
    native_item = canvas._gt_level_items[GeometryLevel.OBB][0]
    derived_item = canvas._gt_level_items[GeometryLevel.AABB][0]
    assert native_item.brush().style() == Qt.BrushStyle.BDiagPattern
    assert derived_item.brush().style() != Qt.BrushStyle.BDiagPattern


def test_set_derived_levels_visible_hides_non_native_layers_only(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    polygon = [(10.0, 10.0), (50.0, 5.0), (90.0, 40.0), (60.0, 90.0), (20.0, 60.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": polygon}],
        native_level=GeometryLevel.POLYGON,
        reviewed=True,
    )

    canvas.set_derived_levels_visible(False)
    assert canvas._gt_level_items[GeometryLevel.POLYGON][0].isVisible() is True
    assert canvas._gt_level_items[GeometryLevel.OBB][0].isVisible() is False
    assert canvas._gt_level_items[GeometryLevel.AABB][0].isVisible() is False

    canvas.set_derived_levels_visible(True)
    assert canvas._gt_level_items[GeometryLevel.OBB][0].isVisible() is True
    assert canvas._gt_level_items[GeometryLevel.AABB][0].isVisible() is True


def test_clear_gt_detections_clears_per_level_state(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}], native_level=GeometryLevel.OBB
    )
    canvas.clear_gt_detections()
    assert canvas._gt_level_items == {}
    assert canvas._gt_obb_items == []


def test_set_gt_detections_single_layer_still_works_unchanged(qapp):
    """Backward-compat: the old single-layer API is untouched for any
    caller that doesn't pass native_level/reviewed -- including its
    visibility wiring, which _apply_visibility must still drive via the
    flat _gt_obb_items list when _gt_level_items was never populated."""
    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    canvas.set_gt_detections(
        [{"class_id": 0, "polygon_px": [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]}]
    )
    assert len(canvas._gt_obb_items) == 1
    assert canvas._gt_level_items == {}  # single-layer path never touches this

    canvas.set_overlay_visibility(show_gt=False, show_pred=True)
    assert canvas._gt_obb_items[0].isVisible() is False

    canvas.set_overlay_visibility(show_gt=True, show_pred=True)
    assert canvas._gt_obb_items[0].isVisible() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_canvas.py -k multi_level -v`
Expected: FAIL — `AttributeError: 'OBBCanvas' object has no attribute 'set_gt_detections_multi_level'`.

- [ ] **Step 3: Implement the per-level layer system**

In `src/hydra_suite/detectkit/gui/canvas.py`:

Add imports (with the existing `from PySide6.QtGui import (...)` block and a new import):

```python
from dataclasses import dataclass

from hydra_suite.utils.geometry_derivation import axis_aligned_bbox_quad, min_area_rect_quad
```

(`GeometryLevel` is imported lazily inside the methods that need it, matching this file's
existing pattern of not importing `hydra_suite.training.*` at module level for a GUI-layer file
that otherwise only imports Qt/cv2/numpy — check whether that pattern actually holds in this
file before deciding; if `canvas.py` already imports non-Qt hydra_suite modules at top level
elsewhere, import `GeometryLevel` at the top instead for consistency. Either placement is
correct; match whatever this file already does.)

Add a per-level style descriptor and the style table, near the `_PALETTE` constant:

```python
@dataclass(frozen=True)
class _LevelStyle:
    pen_style: "Qt.PenStyle"
    brush_style: "Qt.BrushStyle"
    fill_alpha: int  # 0-255; only used when brush_style != NoBrush


def _level_styles():
    """Lazily built so importing GeometryLevel doesn't happen at module load."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    return {
        # POLYGON's pen is DotLine (not SolidLine) so a polygon-native
        # source's filled outline stays visually distinct from AABB's solid
        # outline when both draw for the same detection -- the fill alone
        # (translucent) is the primary differentiator per spec Decision 1,
        # but a same-style outline underneath it would still be confusable.
        GeometryLevel.POLYGON: _LevelStyle(
            Qt.PenStyle.DotLine, Qt.BrushStyle.SolidPattern, 90
        ),
        GeometryLevel.OBB: _LevelStyle(
            Qt.PenStyle.DashLine, Qt.BrushStyle.NoBrush, 0
        ),
        GeometryLevel.AABB: _LevelStyle(
            Qt.PenStyle.SolidLine, Qt.BrushStyle.NoBrush, 0
        ),
    }


_UNREVIEWED_NATIVE_STYLE = _LevelStyle(Qt.PenStyle.SolidLine, Qt.BrushStyle.BDiagPattern, 140)
```

Modify `__init__` to add the new state (alongside the existing `_gt_obb_items` etc. block):

```python
        # Per-geometry-level GT sub-layers (native level down to AABB); the
        # flat _gt_obb_items/_gt_label_items/_gt_class_ids lists below stay
        # as a concatenation across all drawn levels, for _apply_visibility
        # and clear_gt_detections' pre-existing flat-iteration callers.
        self._gt_level_items: dict = {}
        self._gt_level_label_items: dict = {}
        self._gt_level_class_ids: dict = {}
        self._gt_native_level = None
        self._show_derived_levels: bool = True
```

Modify `_draw_detections` to accept a brush/fill spec instead of always `NoBrush`, and a
`show_labels` flag:

```python
    def _draw_detections(
        self,
        detections: list[dict],
        obb_items: list,
        label_items: list,
        class_ids: list,
        class_names: list[str] | dict[int, str] | None,
        line_style: "Qt.PenStyle",
        show_confidence: bool = False,
        *,
        brush_style: "Qt.BrushStyle" = Qt.BrushStyle.NoBrush,
        fill_alpha: int = 255,
        show_labels: bool = True,
    ) -> None:
        """Render *detections* into the given item lists."""
        font = QFont()
        font.setPixelSize(DEFAULT_OBB_FONT_SIZE)
        lookup = self._build_class_lookup(class_names)

        for det in detections:
            class_id: int = det.get("class_id", 0)
            polygon_px = det.get("polygon_px", [])
            if len(polygon_px) < 3:
                continue
            confidence = det.get("confidence", None)

            colour = _PALETTE[class_id % len(_PALETTE)]
            qpoly = QPolygonF()
            for x, y in polygon_px:
                qpoly.append(QPointF(x, y))
            qpoly.append(QPointF(*polygon_px[0]))

            pen = QPen(colour, DEFAULT_OBB_LINE_WIDTH)
            pen.setCosmetic(True)
            pen.setStyle(line_style)

            if brush_style != Qt.BrushStyle.NoBrush:
                fill_colour = QColor(colour)
                fill_colour.setAlpha(fill_alpha)
                brush = QBrush(fill_colour, brush_style)
            else:
                brush = QBrush(Qt.BrushStyle.NoBrush)

            poly_item = self._scene.addPolygon(qpoly, pen, brush)
            obb_items.append(poly_item)
            class_ids.append(class_id)

            if not show_labels:
                label_items.append(None)
                continue

            label_name = lookup.get(class_id, f"class_{class_id}")
            if show_confidence and confidence is not None:
                label_text = f"{label_name} ({confidence:.2f})"
            else:
                label_text = f"{label_name} ({class_id})"
            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(font)
            txt_item.setDefaultTextColor(colour)
            txt_item.setPos(QPointF(*polygon_px[0]))
            self._scene.addItem(txt_item)
            label_items.append(txt_item)
```

`label_items.append(None)` for the `show_labels=False` case keeps `obb_items`/`label_items`/
`class_ids` the same length (needed for `_apply_visibility`'s `zip`-based iteration) without
creating an unused scene item — update `_apply_visibility`'s inner `_set_layer` helper to guard
against a `None` label item:

```python
        def _set_layer(obb_items, label_items, class_ids, layer_visible):
            for obb, lbl, cid in zip(obb_items, label_items, class_ids):
                visible = layer_visible and (
                    not self._visible_class_ids or cid in self._visible_class_ids
                )
                obb.setVisible(visible)
                if lbl is not None:
                    lbl.setVisible(visible)
```

**CRITICAL — do not simply delete the old GT `_set_layer` call.** `set_gt_detections` (the
old single-layer API, kept for backward compat) populates ONLY the flat
`_gt_obb_items`/`_gt_label_items`/`_gt_class_ids` lists — it never touches `_gt_level_items`.
If `_apply_visibility` unconditionally iterates `_gt_level_items` and drops the flat-list call,
then after any `set_gt_detections`/`set_detections` call `_gt_level_items == {}`, the GT loop
runs zero times, and `setVisible()` is never called on any GT item for that path — silently
breaking `set_overlay_visibility`/`set_class_filter` for every caller still using the single-layer
API. This is caught by two pre-existing tests in `tests/test_detectkit_canvas_dual_layer.py`
(`test_canvas_set_overlay_visibility_hides_gt`, `test_canvas_set_class_filter`) — if Step 4 shows
either of those failing, this is why.

Replace `_apply_visibility`'s body (below the `_set_layer` helper) with a branch: use the
per-level path when it's been populated, otherwise fall back to the old flat-list path — the two
are mutually exclusive per call (whichever of `set_gt_detections`/`set_gt_detections_multi_level`
was called last), so there's no double-`setVisible()` risk:

```python
        if self._gt_level_items:
            for level, items in self._gt_level_items.items():
                label_items = self._gt_level_label_items[level]
                class_ids = self._gt_level_class_ids[level]
                level_visible = self._show_gt and (
                    level == self._gt_native_level or self._show_derived_levels
                )
                _set_layer(items, label_items, class_ids, level_visible)
        else:
            _set_layer(
                self._gt_obb_items, self._gt_label_items, self._gt_class_ids, self._show_gt
            )

        _set_layer(
            self._pred_obb_items,
            self._pred_label_items,
            self._pred_class_ids,
            self._show_pred,
        )
```

Add the new multi-level entry point, right after the existing `set_gt_detections`:

```python
    def set_gt_detections_multi_level(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
        *,
        native_level,
        reviewed: bool = True,
    ) -> None:
        """Draw ground-truth detections at *native_level*, plus every
        derived level below it down to AABB, each in its own per-level
        style. Only the native level's shapes carry a text label."""
        from hydra_suite.training.geometry_levels import GeometryLevel

        self.clear_gt_detections()
        styles = _level_styles()

        for level in (GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB):
            if level > native_level:
                continue

            level_detections = []
            for det in detections:
                polygon_px = det.get("polygon_px", [])
                if level == native_level:
                    shape = polygon_px
                elif level == GeometryLevel.OBB:
                    shape = min_area_rect_quad(polygon_px)
                else:  # GeometryLevel.AABB
                    shape = axis_aligned_bbox_quad(polygon_px)
                if not shape:
                    continue
                level_detections.append({**det, "polygon_px": shape})

            if not level_detections:
                continue

            style = (
                _UNREVIEWED_NATIVE_STYLE
                if (level == native_level and not reviewed)
                else styles[level]
            )
            obb_items: list = []
            label_items: list = []
            class_ids: list = []
            self._draw_detections(
                level_detections,
                obb_items,
                label_items,
                class_ids,
                class_names,
                style.pen_style,
                show_confidence=False,
                brush_style=style.brush_style,
                fill_alpha=style.fill_alpha,
                show_labels=(level == native_level),
            )
            self._gt_level_items[level] = obb_items
            self._gt_level_label_items[level] = label_items
            self._gt_level_class_ids[level] = class_ids
            self._gt_obb_items.extend(obb_items)
            self._gt_label_items.extend(label_items)
            self._gt_class_ids.extend(class_ids)

        self._gt_native_level = native_level
        self._apply_visibility()

    def set_derived_levels_visible(self, visible: bool) -> None:
        """Toggle whether non-native (derived) GT levels are drawn."""
        self._show_derived_levels = visible
        self._apply_visibility()
```

Modify `clear_gt_detections` to also clear the new per-level state:

```python
    def clear_gt_detections(self) -> None:
        """Remove all GT polygon and label items from the scene."""
        for item in self._gt_obb_items:
            self._scene.removeItem(item)
        for item in self._gt_label_items:
            if item is not None:
                self._scene.removeItem(item)
        self._gt_obb_items.clear()
        self._gt_label_items.clear()
        self._gt_class_ids.clear()
        self._gt_level_items.clear()
        self._gt_level_label_items.clear()
        self._gt_level_class_ids.clear()
        self._gt_native_level = None
```

`clear_all` also needs the same new state reset. It currently duplicates
`clear_gt_detections`'s field list rather than calling it (it goes through `self._scene.clear()`,
which deletes the underlying Qt/C++ items directly, so it must NOT also call
`clear_gt_detections()` — that would try to `removeItem()` objects `scene.clear()` already
destroyed). **This is not optional cleanup:** if `_gt_level_items` is left populated after
`clear_all()`, the next `_apply_visibility()` call (e.g. from `_on_overlay_changed`) calls
`setVisible()` on already-deleted C++ objects and raises `RuntimeError: Internal C++ object
already deleted`. `clear_all` is reachable in production via `main_window.py`'s
`on_images_deleted` → `_apply_visibility` sequence. Add the reset to `clear_all`'s existing field
list:

```python
    def clear_all(self) -> None:
        """Remove everything from the scene."""
        self._scene.clear()
        self._pix_item = None
        self._gt_obb_items.clear()
        self._gt_label_items.clear()
        self._gt_class_ids.clear()
        self._gt_level_items.clear()
        self._gt_level_label_items.clear()
        self._gt_level_class_ids.clear()
        self._gt_native_level = None
        self._pred_obb_items.clear()
        self._pred_label_items.clear()
        self._pred_class_ids.clear()
        self._zoom = 1.0
        self._fit_mode = True
        self.setCursor(Qt.CursorShape.ArrowCursor)
```

(`_show_derived_levels` is deliberately NOT reset here — it's a user preference tracking the
"Show derived levels" checkbox state, not scene content; resetting it would desync the canvas
from the checkbox's actual UI state.)

Add a regression test for this exact crash to Task 3's Step 1 test list:

```python
def test_clear_all_then_apply_visibility_does_not_raise(qapp):
    from hydra_suite.training.geometry_levels import GeometryLevel

    canvas = OBBCanvas()
    canvas.set_image_array(np.zeros((100, 100, 3), dtype=np.uint8))
    quad = [(10.0, 10.0), (90.0, 20.0), (80.0, 90.0), (0.0, 80.0)]
    canvas.set_gt_detections_multi_level(
        [{"class_id": 0, "polygon_px": quad}], native_level=GeometryLevel.OBB
    )
    canvas.clear_all()
    canvas.set_overlay_visibility(show_gt=True, show_pred=True)  # must not raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_canvas.py -v`
Expected: all PASS, including every pre-existing test in the file (the backward-compat test
confirms `set_gt_detections`'s old single-layer behavior is unchanged).

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/canvas.py tests/test_detectkit_canvas.py && isort src/hydra_suite/detectkit/gui/canvas.py tests/test_detectkit_canvas.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/canvas.py tests/test_detectkit_canvas.py
git commit -m "feat(detectkit): OBBCanvas renders per-geometry-level GT layers"
```

---

## Task 4: `main_window.py` wiring — resolve source level/reviewed, call the new API

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Test: Create `tests/test_detectkit_show_image_multi_level.py` (no existing
  `test_detectkit_main_window*.py` file covers `show_image` — verified this session; this is a
  new file, following this repo's usual style for a source-inspection-only test, e.g.
  `tests/test_detectkit_sam2_escalation_wiring.py`'s `inspect.getsource(...)`-based assertions,
  which need no `qapp` fixture since they never instantiate a `MainWindow`)

**Interfaces:**
- Consumes: `OBBCanvas.set_gt_detections_multi_level` (Task 3).
- Produces: `_resolve_source_render_state(project, source_path) -> tuple[GeometryLevel, bool]` —
  a new MODULE-LEVEL (not a `MainWindow` method) pure function in `main_window.py`, so it's
  testable without instantiating a `QWidget`. Looks up the `OBBSource` whose `.path ==
  source_path` in `project.sources`; returns `(GeometryLevel.OBB, True)` if `project` is `None`,
  no source matches, or the matched source's `level` string doesn't parse via
  `GeometryLevel.from_str` (which raises `ValueError` on an unrecognized string — this function
  catches that and falls back rather than propagating it, since `OBBSource.level` is an
  unvalidated string loaded straight from project JSON). `MainWindow.show_image` calls this
  helper, then calls `set_gt_detections_multi_level` with its result, instead of the old
  single-layer `set_gt_detections`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_show_image_multi_level.py`:

```python
"""Regression: show_image must render every geometry level, not just native."""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def test_show_image_calls_multi_level_api_with_source_level_and_reviewed():
    """show_image must resolve the current source's level/reviewed and pass
    them to set_gt_detections_multi_level, not the old single-layer
    set_gt_detections."""
    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    assert "set_gt_detections_multi_level" in source
    assert "set_gt_detections(" not in source  # the old single-layer call is gone
    assert "native_level" in source
    assert "reviewed" in source
    assert "GeometryLevel.from_str" in source
    assert "except ValueError" in source  # from_str must be guarded, see Step 3


def test_resolve_native_level_and_reviewed_reads_the_matching_source():
    """Behavioral check (not just source-text): the level/reviewed resolved
    for a given source_path must actually come from the OBBSource whose
    .path matches, not some other source or a wrong default."""
    from hydra_suite.detectkit.gui.main_window import _resolve_source_render_state
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    proj.sources = [
        OBBSource(path="/a", name="a", level="obb", reviewed=True),
        OBBSource(path="/b", name="b", level="polygon", reviewed=False),
    ]

    native_level, reviewed = _resolve_source_render_state(proj, "/b")
    assert native_level == GeometryLevel.POLYGON
    assert reviewed is False


def test_resolve_native_level_and_reviewed_defaults_when_source_missing():
    from hydra_suite.detectkit.gui.main_window import _resolve_source_render_state
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    native_level, reviewed = _resolve_source_render_state(proj, "/nonexistent")
    assert native_level == GeometryLevel.OBB
    assert reviewed is True


def test_resolve_native_level_and_reviewed_falls_back_on_unknown_level_string():
    """A hand-edited/future-version project JSON could carry a level string
    GeometryLevel.from_str doesn't recognize -- this must degrade to OBB
    with a warning, not crash show_image on every image selection."""
    from hydra_suite.detectkit.gui.main_window import _resolve_source_render_state
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    proj.sources = [OBBSource(path="/c", name="c", level="not_a_real_level", reviewed=True)]

    native_level, reviewed = _resolve_source_render_state(proj, "/c")
    assert native_level == GeometryLevel.OBB  # fallback, not a raised ValueError
    assert reviewed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_show_image_multi_level.py -v`
Expected: FAIL — `set_gt_detections_multi_level` not yet in `show_image`'s source.

- [ ] **Step 3: Update `show_image`**

In `src/hydra_suite/detectkit/gui/main_window.py`, `show_image` currently does (around lines
1961-1983):

```python
        label_path = find_label_for_image(Path(image_path), source_path)
        if label_path is not None:
            import cv2

            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                class_names = self._project.class_names if self._project else ["object"]
                class_id_map = None
                if self._project is not None:
                    try:
                        class_id_map = source_class_id_map(
                            source_path, self._project.class_names
                        )
                    except Exception:
                        class_id_map = {}
                        logger.warning(
                            "Skipping incompatible source labels for preview: %s",
                            source_path,
                            exc_info=True,
                        )
                dets = parse_obb_label(label_path, w, h, class_id_map=class_id_map)
                self._canvas.set_gt_detections(dets, class_names=class_names)
```

Add a new module-level function (not a `MainWindow` method — this makes it testable without a
`QApplication`/widget instance), placed near the top of `main_window.py` after its imports,
alongside any other module-level helper functions already in the file:

```python
def _resolve_source_render_state(project, source_path):
    """Return (native_level, reviewed) for the OBBSource at *source_path* in
    *project*. Falls back to (GeometryLevel.OBB, True) if project is None,
    no source matches, or the matched source's level string doesn't parse --
    OBBSource.level is an unvalidated string loaded from project JSON, so a
    hand-edited or future-version file must degrade gracefully here rather
    than crashing show_image on every image selection."""
    from hydra_suite.training.geometry_levels import GeometryLevel

    if project is None:
        return GeometryLevel.OBB, True

    src_obj = next((s for s in project.sources if s.path == source_path), None)
    if src_obj is None:
        return GeometryLevel.OBB, True

    try:
        native_level = GeometryLevel.from_str(src_obj.level)
    except ValueError:
        logger.warning(
            "Unknown geometry level %r for source %r; rendering as OBB",
            src_obj.level,
            source_path,
        )
        native_level = GeometryLevel.OBB

    return native_level, src_obj.reviewed
```

(Check whether `main_window.py` already has a module-level `logger = logging.getLogger(...)` —
it almost certainly does, given `logger.warning(...)` calls already appear in `show_image` per
the quoted code above; reuse that same logger, don't create a second one.)

Then replace the final two lines of `show_image`'s existing block (the `dets = ...` /
`self._canvas.set_gt_detections(...)` pair) with:

```python
                dets = parse_obb_label(label_path, w, h, class_id_map=class_id_map)
                native_level, reviewed = _resolve_source_render_state(
                    self._project, source_path
                )
                self._canvas.set_gt_detections_multi_level(
                    dets,
                    class_names=class_names,
                    native_level=native_level,
                    reviewed=reviewed,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_show_image_multi_level.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/main_window.py && isort src/hydra_suite/detectkit/gui/main_window.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_show_image_multi_level.py
git commit -m "feat(detectkit): show_image renders all geometry levels via the new canvas API"
```

---

## Task 5: "Show derived levels" toggle in the Overlay settings group

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/tools_panel.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Test: `tests/test_detectkit_tools_panel.py`

**Interfaces:**
- Consumes: `OBBCanvas.set_derived_levels_visible` (Task 3).
- Produces: `OverlaySettings.show_derived_levels: bool` (new field), `ToolsPanel._chk_show_derived_levels`
  (new checkbox), wired through `get_overlay_settings()` to `MainWindow._on_overlay_changed`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_tools_panel.py`:

```python
def test_tools_panel_has_show_derived_levels_checkbox(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_chk_show_derived_levels")


def test_overlay_settings_includes_show_derived_levels_default_true(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    settings = panel.get_overlay_settings()
    assert settings.show_derived_levels is True


def test_overlay_settings_reflects_unchecked_show_derived_levels(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel._chk_show_derived_levels.setChecked(False)
    settings = panel.get_overlay_settings()
    assert settings.show_derived_levels is False


def test_main_window_on_overlay_changed_wires_derived_levels_to_canvas():
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._on_overlay_changed)
    assert "set_derived_levels_visible" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_tools_panel.py -k derived_levels -v`
Expected: FAIL — `AttributeError`/missing field.

- [ ] **Step 3: Add the field, checkbox, and wiring**

In `src/hydra_suite/detectkit/gui/panels/tools_panel.py`, add the new field to `OverlaySettings`:

```python
class OverlaySettings(NamedTuple):
    """Overlay display settings passed from ToolsPanel to MainWindow."""

    show_gt: bool
    show_pred: bool
    show_derived_levels: bool
    confidence_threshold: float
    visible_class_ids: set
    active_model_path: str
```

Add the checkbox in `_build_overlay_group`, right after `self._chk_show_pred`:

```python
        self._chk_show_derived_levels = QCheckBox("Show derived levels")
        self._chk_show_derived_levels.setChecked(True)
        self._chk_show_derived_levels.setToolTip(
            "Also show geometry levels derived from a source's native level "
            "(e.g. a polygon source's derived OBB and AABB outlines)."
        )
        self._chk_show_derived_levels.stateChanged.connect(self._emit_overlay_changed)
        v.addWidget(self._chk_show_derived_levels)
```

Update `get_overlay_settings` to populate the new field:

```python
        return OverlaySettings(
            show_gt=show_gt,
            show_pred=show_pred,
            show_derived_levels=self._chk_show_derived_levels.isChecked(),
            confidence_threshold=confidence,
            visible_class_ids=visible_ids,
            active_model_path=self._active_model_path,
        )
```

In `src/hydra_suite/detectkit/gui/main_window.py`'s `_on_overlay_changed`, add the new call right
after the existing `set_overlay_visibility`/`set_class_filter` calls:

```python
        self._canvas.set_overlay_visibility(settings.show_gt, settings.show_pred)
        self._canvas.set_class_filter(settings.visible_class_ids)
        self._canvas.set_derived_levels_visible(settings.show_derived_levels)
```

`tests/test_detectkit_tools_panel.py:38-49` (`test_overlay_settings_namedtuple`) constructs an
`OverlaySettings` by keyword but does not pass `show_derived_levels` — since the new field has no
default (matching every other field in this `NamedTuple`, none of which have defaults), this
existing test will fail with a `TypeError: missing required argument` once the field is added.
Fix it in the same edit, not as an afterthought:

```python
def test_overlay_settings_namedtuple():
    from hydra_suite.detectkit.gui.panels.tools_panel import OverlaySettings

    s = OverlaySettings(
        show_gt=True,
        show_pred=False,
        show_derived_levels=True,
        confidence_threshold=0.5,
        visible_class_ids=set(),
        active_model_path="",
    )
    assert s.show_gt is True
    assert s.show_pred is False
    assert s.show_derived_levels is True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_tools_panel.py -v`
Expected: all PASS, including the updated `test_overlay_settings_namedtuple` and every other
pre-existing test in the file.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/panels/tools_panel.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_tools_panel.py && isort src/hydra_suite/detectkit/gui/panels/tools_panel.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_tools_panel.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/tools_panel.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_tools_panel.py
git commit -m "feat(detectkit): Show derived levels toggle wired into overlay settings"
```

---

## Task 6: Full sweep + lint + GUI smoke test

**Files:** none new — verification-only.

- [ ] **Step 1: Run the full Part B test slice**

Run:
```bash
conda activate hydra-mps
python -m pytest tests/test_detectkit_geometry_derivation.py tests/test_detectkit_source_import.py \
  tests/test_detectkit_canvas.py tests/test_detectkit_canvas_dual_layer.py \
  tests/test_detectkit_tools_panel.py tests/test_detectkit_sam2_escalation_wiring.py -v
```
Expected: all PASS. `test_detectkit_canvas_dual_layer.py` in particular must show no regression
— it's a pre-existing file exercising the GT/Pred dual-layer behavior this plan's restructuring
touches.

- [ ] **Step 2: Run `make format-check` and `make lint`**

Run: `conda activate hydra-mps && make format-check && make lint`
Expected: no formatting diffs; no new lint findings in files this plan touched.

- [ ] **Step 3: Manual smoke test (GUI)**

Per CLAUDE.md: start `detectkit`, open a project with sources at different native levels (obb,
aabb, and — if available — an accepted-escalation polygon source), select each in turn, and
verify: a polygon-native source shows a translucent filled shape plus a dashed derived-OBB
outline plus a solid derived-AABB outline; an OBB-native source shows a dashed native outline
plus a solid derived-AABB outline; an AABB-native source shows only its solid outline; an
unreviewed source's native shape renders hatched/striped instead of its normal style; unchecking
"Show derived levels" in the Overlay settings hides everything except the native shape. If no
interactive display is available in this environment, say so explicitly rather than claiming
this step was completed.

- [ ] **Step 4: Commit (only if Steps 1-3 required fixes)**

```bash
git add -A
git commit -m "fix(detectkit): address lint/test findings from multi-level canvas sweep"
```
