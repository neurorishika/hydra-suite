# Multi-Arena UI/UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TrackerKit's ten-widget ROI toolbar with an arena-centric bar, add a rotation-capable grid builder, and render arenas through a viewport-space overlay so line weight is zoom-invariant and drawing works at any zoom.

**Architecture:** All geometry and styling logic moves into three Qt-free modules (`arena_geometry.py`, `arena_style.py`) that can be unit-tested without a display. A new `ArenaCanvas(QWidget)` replaces the monkeypatched `QLabel`, painting the scaled frame and then the overlay in *widget* coordinates through an explicit image-to-viewport transform. A new `ArenaPanel` owns the two-state arena bar and the overlap lock.

**Tech Stack:** Python 3.13, PySide6 (Qt 6), NumPy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-multi-arena-ux-redesign-design.md` — read it first; this plan argues from it.

## Global Constraints

- `roi_shapes` schema is frozen: `{"type": "circle"|"polygon", "params": ..., "mode": "include"|"exclude", "arena_id": int}`. No new shape type, no new field, no separate storage for generated shapes.
- Tracking output must be byte-identical. Verified with the MPS equivalence gate in Task 12, not asserted.
- Geometry and style logic must import without Qt. Some GUI tests abort the interpreter (memory `project_main_suite_blockers`), so anything needing coverage cannot live behind a widget import.
- Single-arena behaviour is unchanged end to end: `n_arenas == 1` means no `arena_id` CSV column.
- Veil alpha `0.15`; text alpha `0.70`; click/drag threshold `3` px; rotation range `-45.0` to `+45.0` degrees in `0.5` steps; glyph size clamped to `[10, 64]` px.
- Non-overlapping spacing minimums: `2 * radius` for circles; `width` in x and `height` in y for rectangles.
- Click mapping: left-click adds a point, right-click removes the most recent point.
- Work in a git worktree branched from local `HEAD`:
  `git worktree add .worktrees/arena-ux -b feat/arena-ux HEAD`
- Environment: `conda activate hydra-mps`. GUI tests need `QT_QPA_PLATFORM=offscreen`.
- Commit as the configured git user. Do **not** add a `Co-Authored-By: Claude` trailer.
- Never `exec()` a dialog in a test — construct it and read its state.

## File Structure

| File | Responsibility |
|---|---|
| `src/hydra_suite/trackerkit/arena_geometry.py` *(new, Qt-free)* | Shape maths: centroid, hit-testing, overlap detection, grid generation, grid caps |
| `src/hydra_suite/trackerkit/gui/widgets/arena_style.py` *(new, Qt-free)* | Luminance-driven palette, device-pixel sizing, alpha constants |
| `src/hydra_suite/trackerkit/gui/widgets/arena_canvas.py` *(new)* | `ArenaCanvas(QWidget)`: transform, `paintEvent`, mouse routing |
| `src/hydra_suite/trackerkit/gui/panels/arena_panel.py` *(new)* | Arena bar, two-state machine, overlap lock |
| `src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py` *(modify)* | Shape/rotation/spacing controls, caps, shared renderer |
| `src/hydra_suite/trackerkit/gui/orchestrators/session.py` *(modify)* | Delegate ROI drawing to the canvas; delete the zoom locks |
| `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` *(modify)* | Block tracking start on any arena overlap |
| `src/hydra_suite/trackerkit/gui/main_window.py` *(modify)* | `video_label` becomes `ArenaCanvas`; old ROI toolbar replaced by `ArenaPanel` |

---

### Task 1: Arena hit-testing geometry

**Files:**
- Create: `src/hydra_suite/trackerkit/arena_geometry.py`
- Test: `tests/test_arena_geometry_hittest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `shape_centroid(shape) -> tuple[float, float]`, `point_in_shape(shape, x, y) -> bool`, `arena_at_point(shapes, x, y) -> int | None`.

- [x] **Step 1: Write the failing test**

```python
"""Hit-testing for arena shapes. Pure geometry, no Qt import."""

from hydra_suite.trackerkit.arena_geometry import (
    arena_at_point,
    point_in_shape,
    shape_centroid,
)


def _circle(cx, cy, r, arena_id=0, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def _square(cx, cy, half, arena_id=0, mode="include"):
    return {
        "type": "polygon",
        "params": [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        "mode": mode,
        "arena_id": arena_id,
    }


def test_circle_centroid_is_its_centre():
    assert shape_centroid(_circle(100, 200, 30)) == (100.0, 200.0)


def test_polygon_centroid_is_vertex_mean():
    assert shape_centroid(_square(100, 200, 10)) == (100.0, 200.0)


def test_point_inside_circle():
    assert point_in_shape(_circle(100, 100, 20), 110, 100) is True


def test_point_outside_circle():
    assert point_in_shape(_circle(100, 100, 20), 130, 100) is False


def test_point_on_circle_edge_counts_as_inside():
    """Inclusive boundary matches cv2.circle's filled rasterization."""
    assert point_in_shape(_circle(100, 100, 20), 120, 100) is True


def test_point_inside_polygon():
    assert point_in_shape(_square(100, 100, 10), 100, 100) is True


def test_point_outside_polygon():
    assert point_in_shape(_square(100, 100, 10), 200, 200) is False


def test_arena_at_point_finds_the_arena():
    shapes = [_circle(50, 50, 20, arena_id=0), _circle(200, 50, 20, arena_id=1)]
    assert arena_at_point(shapes, 200, 50) == 1


def test_arena_at_point_returns_none_outside_every_arena():
    shapes = [_circle(50, 50, 20, arena_id=0)]
    assert arena_at_point(shapes, 500, 500) is None


def test_exclude_hole_is_not_part_of_the_arena():
    """A point inside an exclude zone belongs to no arena, even inside an include."""
    shapes = [
        _circle(100, 100, 50, arena_id=3),
        _circle(100, 100, 10, arena_id=3, mode="exclude"),
    ]
    assert arena_at_point(shapes, 100, 100) is None
    assert arena_at_point(shapes, 140, 100) == 3


def test_overlap_resolves_by_draw_order():
    """Last-writer-wins, matching engine_params.build_arena_labels."""
    shapes = [_circle(100, 100, 40, arena_id=0), _circle(110, 100, 40, arena_id=1)]
    assert arena_at_point(shapes, 105, 100) == 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_geometry_hittest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.trackerkit.arena_geometry'`

- [x] **Step 3: Write the implementation**

```python
"""Pure arena geometry: hit-testing, overlap detection, grid generation.

Qt-free by design. Some GUI tests abort the interpreter (see memory
``project_main_suite_blockers``), so every rule that needs coverage lives
here rather than behind a widget import.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def shape_centroid(shape: dict[str, Any]) -> tuple[float, float]:
    """Centre of a circle, or the vertex mean of a polygon."""
    if shape.get("type") == "circle":
        center_x, center_y, _radius = shape["params"]
        return (float(center_x), float(center_y))
    points = np.asarray(shape["params"], dtype=float)
    return (float(points[:, 0].mean()), float(points[:, 1].mean()))


def point_in_shape(shape: dict[str, Any], x: float, y: float) -> bool:
    """Whether (x, y) falls inside *shape*, boundary inclusive.

    The boundary is inclusive so hit-testing agrees with the filled
    rasterization ``cv2.circle`` / ``cv2.fillPoly`` produce in
    ``engine_params.build_arena_labels``.
    """
    if shape.get("type") == "circle":
        center_x, center_y, radius = shape["params"]
        return math.hypot(x - float(center_x), y - float(center_y)) <= float(radius)
    contour = np.asarray(shape["params"], dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def arena_at_point(
    shapes: list[dict[str, Any]] | None, x: float, y: float
) -> int | None:
    """The arena owning (x, y), or ``None`` if no arena does.

    Ties resolve by draw order -- the LAST include shape containing the point
    wins, matching ``build_arena_labels``' last-writer-wins rasterization, so
    clicking to select an arena agrees with the label image the tracker uses.
    Exclude zones are subtracted last, again mirroring the engine, so a point
    inside an exclude hole belongs to no arena.
    """
    if not shapes:
        return None
    for shape in shapes:
        if shape.get("mode", "include") == "exclude" and point_in_shape(shape, x, y):
            return None
    found: int | None = None
    for shape in shapes:
        if shape.get("mode", "include") == "include" and point_in_shape(shape, x, y):
            found = int(shape.get("arena_id", 0))
    return found
```

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_geometry_hittest.py -v`
Expected: PASS (11 tests)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/arena_geometry.py tests/test_arena_geometry_hittest.py
git commit -m "feat(arena): pure hit-testing geometry for arena shapes"
```

---

### Task 2: Overlap detection

**Files:**
- Modify: `src/hydra_suite/trackerkit/arena_geometry.py`
- Test: `tests/test_arena_overlap.py`

**Interfaces:**
- Consumes: nothing from Task 1 -- this rasterizes independently so it mirrors `build_arena_labels`' fill semantics exactly rather than its own hit-test.
- Produces: `overlapping_arena_pairs(shapes, width, height) -> list[tuple[int, int]]` returning sorted `(lower_id, higher_id)` pairs, ascending.

**Why staged:** a 96-well plate is 4560 candidate pairs and the reference fixture is 4512x4512, so full-frame mask intersection per pair is not viable. Analytic and bounding-box filters reject the vast majority; rasterization runs only on survivors, cropped to the intersection box.

- [x] **Step 1: Write the failing test**

```python
"""Overlap detection between arenas, including fast-path/brute-force agreement."""

import numpy as np
import pytest

from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs


def _circle(cx, cy, r, arena_id, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def _square(cx, cy, half, arena_id, mode="include"):
    return {
        "type": "polygon",
        "params": [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        "mode": mode,
        "arena_id": arena_id,
    }


def _brute_force_pairs(shapes, width, height):
    """Authoritative reference: rasterize every arena full-frame and intersect."""
    ids = sorted({int(s["arena_id"]) for s in shapes if s.get("mode") == "include"})
    import cv2

    masks = {}
    for arena_id in ids:
        canvas = np.zeros((height, width), np.uint8)
        for shape in shapes:
            if int(shape["arena_id"]) != arena_id:
                continue
            value = 255 if shape.get("mode", "include") == "include" else 0
            if shape["type"] == "circle":
                cx, cy, r = shape["params"]
                cv2.circle(canvas, (int(cx), int(cy)), int(r), value, -1)
            else:
                pts = np.asarray(shape["params"], np.int32)
                cv2.fillPoly(canvas, [pts], value)
        masks[arena_id] = canvas > 0
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if np.any(masks[a] & masks[b]):
                out.append((a, b))
    return out


def test_separate_circles_do_not_overlap():
    shapes = [_circle(50, 50, 20, 0), _circle(200, 50, 20, 1)]
    assert overlapping_arena_pairs(shapes, 400, 200) == []


def test_intersecting_circles_are_reported():
    shapes = [_circle(100, 100, 40, 0), _circle(150, 100, 40, 1)]
    assert overlapping_arena_pairs(shapes, 400, 300) == [(0, 1)]


def test_exactly_tangent_circles_do_not_overlap():
    """Centre distance == r1 + r2 touches but shares no interior pixel."""
    shapes = [_circle(100, 100, 30, 0), _circle(160, 100, 30, 1)]
    assert overlapping_arena_pairs(shapes, 400, 300) == []


def test_exclude_zone_can_resolve_an_overlap():
    """Punching the shared region out of one arena clears the conflict."""
    shapes = [
        _circle(100, 100, 40, 0),
        _circle(150, 100, 40, 1),
        _circle(150, 100, 40, 0, mode="exclude"),
    ]
    assert overlapping_arena_pairs(shapes, 400, 300) == []


def test_pairs_are_sorted_ascending():
    shapes = [_circle(100, 100, 60, 5), _circle(120, 100, 60, 2)]
    assert overlapping_arena_pairs(shapes, 400, 300) == [(2, 5)]


def test_mixed_circle_and_polygon_overlap():
    shapes = [_circle(100, 100, 30, 0), _square(115, 100, 20, 1)]
    assert overlapping_arena_pairs(shapes, 400, 300) == [(0, 1)]


def test_single_arena_never_overlaps_itself():
    shapes = [_circle(100, 100, 40, 0), _circle(110, 100, 40, 0)]
    assert overlapping_arena_pairs(shapes, 400, 300) == []


@pytest.mark.parametrize("seed", range(20))
def test_fast_path_agrees_with_brute_force(seed):
    """The analytic and bbox filters must never disagree with rasterization.

    Without this, an optimization bug would silently weaken the overlap gate
    -- the failure mode is a missed conflict, which is invisible in the UI.
    """
    rng = np.random.default_rng(seed)
    width = height = 200
    shapes = []
    for arena_id in range(5):
        cx, cy = rng.integers(20, 180, size=2)
        r = int(rng.integers(10, 45))
        if arena_id % 2:
            shapes.append(_circle(int(cx), int(cy), r, arena_id))
        else:
            shapes.append(_square(int(cx), int(cy), r, arena_id))
    assert overlapping_arena_pairs(shapes, width, height) == _brute_force_pairs(
        shapes, width, height
    )
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_overlap.py -v`
Expected: FAIL — `ImportError: cannot import name 'overlapping_arena_pairs'`

- [x] **Step 3: Write the implementation**

Append to `arena_geometry.py`:

```python
def _shape_bbox(shape: dict[str, Any]) -> tuple[float, float, float, float]:
    """Axis-aligned (x0, y0, x1, y1) bounding box of one shape."""
    if shape.get("type") == "circle":
        center_x, center_y, radius = (float(v) for v in shape["params"])
        return (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
    points = np.asarray(shape["params"], dtype=float)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _arena_bbox(shapes: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Bounding box of an arena's INCLUDE shapes only.

    Excludes can only shrink an arena, never grow it, so ignoring them keeps
    this a valid conservative bound.
    """
    boxes = [_shape_bbox(s) for s in shapes if s.get("mode", "include") == "include"]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _boxes_disjoint(a, b) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _rasterize(
    shapes: list[dict[str, Any]],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Boolean mask of one arena, cropped to the (x0, y0, width, height) box."""
    canvas = np.zeros((height, width), np.uint8)
    offset = np.asarray([x0, y0], dtype=np.int32)
    for shape in shapes:
        value = 255 if shape.get("mode", "include") == "include" else 0
        if shape.get("type") == "circle":
            center_x, center_y, radius = shape["params"]
            cv2.circle(
                canvas,
                (int(center_x) - x0, int(center_y) - y0),
                int(radius),
                value,
                -1,
            )
        else:
            points = np.asarray(shape["params"], dtype=np.int32) - offset
            cv2.fillPoly(canvas, [points], value)
    return canvas > 0


def overlapping_arena_pairs(
    shapes: list[dict[str, Any]] | None,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Every pair of arena ids sharing at least one pixel, ascending.

    Staged for cost: analytic circle-circle first, then bounding-box
    rejection, then rasterization of survivors cropped to the intersection
    box. A 96-well plate is 4560 candidate pairs and frames run to 4512x4512,
    so full-frame intersection per pair is not viable. The first two stages
    are pure rejection filters -- they may only prove NON-overlap; stage 3 is
    authoritative for everything that survives.
    """
    if not shapes:
        return []
    by_arena: dict[int, list[dict[str, Any]]] = {}
    for shape in shapes:
        by_arena.setdefault(int(shape.get("arena_id", 0)), []).append(shape)
    arena_ids = sorted(
        aid
        for aid, group in by_arena.items()
        if any(s.get("mode", "include") == "include" for s in group)
    )

    boxes = {aid: _arena_bbox(by_arena[aid]) for aid in arena_ids}
    pairs: list[tuple[int, int]] = []
    for index, first in enumerate(arena_ids):
        for second in arena_ids[index + 1 :]:
            box_a, box_b = boxes[first], boxes[second]
            if _boxes_disjoint(box_a, box_b):
                continue

            group_a, group_b = by_arena[first], by_arena[second]
            # Analytic fast path: two bare circles, no exclude zones.
            if len(group_a) == 1 and len(group_b) == 1:
                if group_a[0]["type"] == "circle" and group_b[0]["type"] == "circle":
                    ax, ay, ar = (float(v) for v in group_a[0]["params"])
                    bx, by_, br = (float(v) for v in group_b[0]["params"])
                    if math.hypot(ax - bx, ay - by_) >= ar + br:
                        continue

            x0 = int(math.floor(max(box_a[0], box_b[0], 0)))
            y0 = int(math.floor(max(box_a[1], box_b[1], 0)))
            x1 = int(math.ceil(min(box_a[2], box_b[2], width - 1)))
            y1 = int(math.ceil(min(box_a[3], box_b[3], height - 1)))
            crop_w, crop_h = x1 - x0 + 1, y1 - y0 + 1
            if crop_w <= 0 or crop_h <= 0:
                continue
            mask_a = _rasterize(group_a, x0, y0, crop_w, crop_h)
            mask_b = _rasterize(group_b, x0, y0, crop_w, crop_h)
            if np.any(mask_a & mask_b):
                pairs.append((first, second))
    return pairs
```

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_overlap.py -v`
Expected: PASS (27 tests, including 20 parametrized agreement cases)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/arena_geometry.py tests/test_arena_overlap.py
git commit -m "feat(arena): staged overlap detection with brute-force agreement test"
```

---

### Task 3: Grid generation with rotation and caps

**Files:**
- Modify: `src/hydra_suite/trackerkit/arena_geometry.py`
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py` (delete `generate_grid_shapes`, import from the new home)
- Modify: `tests/test_arena_grid_dialog.py` (update the import)
- Test: `tests/test_arena_grid_geometry.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `generate_grid_shapes(rows, cols, origin_x, origin_y, pitch_x, pitch_y, size, shape_type="circle", first_arena_id=0, *, size_y=None, rotation_deg=0.0) -> list[dict]`
  - `min_pitch(shape_type, size, size_y=None) -> tuple[int, int]`
  - `max_grid_extent(origin_x, origin_y, pitch_x, pitch_y, width, height, rotation_deg=0.0, limit=100) -> tuple[int, int]`

`size` is the circle diameter or the rectangle width; `size_y` is the rectangle height and defaults to `size` (a square), which keeps every existing positional call working unchanged. `shape_type` accepts `"circle"` or `"polygon"`.

**Cap rule:** all centres lie in the convex hull of the four *corner* centres — `(0,0)`, `(0, cols-1)`, `(rows-1, 0)`, `(rows-1, cols-1)` — because a rotated lattice is an affine image of a rectangular one. The frame is convex, so if the four corners are inside, every centre is. That makes the check O(1) per candidate instead of O(rows*cols).

- [x] **Step 1: Write the failing test**

```python
"""Grid generation under rotation, non-overlapping pitch floors, and extent caps."""

import math

from hydra_suite.trackerkit.arena_geometry import (
    generate_grid_shapes,
    max_grid_extent,
    min_pitch,
)


def test_zero_rotation_matches_the_unrotated_lattice():
    """Rotation must be a strict extension: 0 degrees changes nothing."""
    plain = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40)
    rotated = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40, rotation_deg=0.0)
    assert plain == rotated


def test_rotation_pivots_about_the_first_arena_centre():
    shapes = generate_grid_shapes(1, 2, 100, 100, 50, 50, 20, rotation_deg=90.0)
    assert (shapes[0]["params"][0], shapes[0]["params"][1]) == (100, 100)
    assert (shapes[1]["params"][0], shapes[1]["params"][1]) == (100, 150)


def test_rotation_preserves_centre_to_centre_distance():
    shapes = generate_grid_shapes(1, 2, 100, 100, 60, 60, 20, rotation_deg=37.0)
    ax, ay, _ = shapes[0]["params"]
    bx, by, _ = shapes[1]["params"]
    assert math.hypot(bx - ax, by - ay) == 60


def test_rectangle_uses_separate_width_and_height():
    shapes = generate_grid_shapes(
        1, 1, 100, 100, 200, 200, 40, shape_type="polygon", size_y=20
    )
    xs = [p[0] for p in shapes[0]["params"]]
    ys = [p[1] for p in shapes[0]["params"]]
    assert max(xs) - min(xs) == 40
    assert max(ys) - min(ys) == 20


def test_rotated_rectangle_is_a_four_point_polygon():
    shapes = generate_grid_shapes(
        1, 1, 100, 100, 200, 200, 40, shape_type="polygon", rotation_deg=30.0
    )
    assert shapes[0]["type"] == "polygon"
    assert len(shapes[0]["params"]) == 4


def test_rotated_rectangle_corners_are_actually_rotated():
    """A rotated square must not stay axis-aligned."""
    shapes = generate_grid_shapes(
        1, 1, 100, 100, 200, 200, 40, shape_type="polygon", rotation_deg=30.0
    )
    ys = sorted(p[1] for p in shapes[0]["params"])
    assert len(set(ys)) > 2


def test_circles_ignore_size_y():
    """A circle has one dimension; size_y must not silently deform it."""
    shapes = generate_grid_shapes(1, 1, 50, 50, 100, 100, 40, size_y=10)
    assert shapes[0]["params"][2] == 20


def test_min_pitch_for_circles_is_the_diameter():
    """radius/2 (the original brief) guarantees overlap; 2*radius is the floor."""
    assert min_pitch("circle", 40) == (40, 40)


def test_min_pitch_for_rectangles_is_width_and_height():
    assert min_pitch("polygon", 40, size_y=20) == (40, 20)


def test_min_pitch_grid_produces_no_overlap():
    from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs

    px, py = min_pitch("circle", 40)
    shapes = generate_grid_shapes(3, 3, 60, 60, px, py, 40)
    assert overlapping_arena_pairs(shapes, 400, 400) == []


def test_extent_cap_keeps_every_centre_inside():
    rows, cols = max_grid_extent(50, 50, 100, 100, 400, 300)
    assert (rows, cols) == (3, 4)


def test_extent_cap_shrinks_under_rotation():
    """Rotating a wide grid pushes far centres off-frame, so the cap tightens."""
    straight = max_grid_extent(20, 20, 100, 100, 400, 400, rotation_deg=0.0)
    tilted = max_grid_extent(20, 20, 100, 100, 400, 400, rotation_deg=45.0)
    assert tilted[0] * tilted[1] < straight[0] * straight[1]


def test_extent_cap_is_at_least_one_by_one():
    """An origin outside the frame must still yield a usable minimum."""
    assert max_grid_extent(9999, 9999, 100, 100, 400, 300) == (1, 1)


def test_capped_grid_has_every_centre_in_frame():
    rows, cols = max_grid_extent(30, 30, 90, 70, 640, 480, rotation_deg=22.0)
    shapes = generate_grid_shapes(
        rows, cols, 30, 30, 90, 70, 20, rotation_deg=22.0
    )
    for shape in shapes:
        cx, cy, _ = shape["params"]
        assert 0 <= cx < 640 and 0 <= cy < 480
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_grid_geometry.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_grid_shapes'`

- [x] **Step 3: Write the implementation**

Append to `arena_geometry.py`:

```python
def min_pitch(
    shape_type: str, size: int, size_y: int | None = None
) -> tuple[int, int]:
    """Tightest centre-to-centre pitch that cannot produce overlap.

    Circles of radius r avoid overlap only when spacing is at least 2r, i.e.
    the full diameter -- ``size`` already IS the diameter. Rectangles need the
    full width in x and the full height in y. (The original brief specified
    half these values, which guarantees overlap rather than preventing it.)
    """
    height = int(size if size_y is None else size_y)
    if shape_type == "circle":
        return (int(size), int(size))
    return (int(size), height)


def _rotate(dx: float, dy: float, cos_t: float, sin_t: float) -> tuple[float, float]:
    return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)


def generate_grid_shapes(
    rows: int,
    cols: int,
    origin_x: int,
    origin_y: int,
    pitch_x: int,
    pitch_y: int,
    size: int,
    shape_type: str = "circle",
    first_arena_id: int = 0,
    *,
    size_y: int | None = None,
    rotation_deg: float = 0.0,
) -> list[dict[str, Any]]:
    """Build a row-major grid of arena shapes, optionally rotated.

    ``origin_x``/``origin_y`` is the centre of the top-left (row 0, col 0)
    arena and is the pivot for ``rotation_deg``. ``size`` is the circle
    diameter or the rectangle width; ``size_y`` is the rectangle height and
    defaults to ``size``. Ids are row-major from ``first_arena_id``, matching
    well-plate naming (A1, A2, ..., B1, ...).

    Every shape is an ``include`` shape -- the grid generator only adds
    arenas, it never punches exclude holes. Rotated rectangles are emitted as
    ordinary 4-point polygons, an already-supported shape type, so nothing
    downstream distinguishes them from hand-drawn shapes.
    """
    theta = math.radians(float(rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half_x = int(size) // 2
    half_y = int(size if size_y is None else size_y) // 2

    shapes: list[dict[str, Any]] = []
    arena_id = int(first_arena_id)
    for row in range(int(rows)):
        for col in range(int(cols)):
            offset_x, offset_y = _rotate(
                col * int(pitch_x), row * int(pitch_y), cos_t, sin_t
            )
            center_x = int(round(int(origin_x) + offset_x))
            center_y = int(round(int(origin_y) + offset_y))
            if shape_type == "polygon":
                corners = [
                    (-half_x, -half_y),
                    (half_x, -half_y),
                    (half_x, half_y),
                    (-half_x, half_y),
                ]
                params: Any = [
                    [
                        int(round(center_x + rx)),
                        int(round(center_y + ry)),
                    ]
                    for rx, ry in (
                        _rotate(dx, dy, cos_t, sin_t) for dx, dy in corners
                    )
                ]
            else:
                params = [center_x, center_y, half_x]
            shapes.append(
                {
                    "type": "circle" if shape_type != "polygon" else "polygon",
                    "params": params,
                    "mode": "include",
                    "arena_id": arena_id,
                }
            )
            arena_id += 1
    return shapes


def max_grid_extent(
    origin_x: int,
    origin_y: int,
    pitch_x: int,
    pitch_y: int,
    width: int,
    height: int,
    rotation_deg: float = 0.0,
    limit: int = 100,
) -> tuple[int, int]:
    """Largest (rows, cols) keeping every arena CENTRE inside the frame.

    A rotated lattice is an affine image of a rectangular one, so every centre
    lies in the convex hull of the four corner centres. The frame is convex,
    so checking those four corners is exact -- and O(1) per candidate instead
    of O(rows*cols). Returns at least (1, 1) so the caller always has a usable
    minimum even when the origin itself is off-frame.
    """
    theta = math.radians(float(rotation_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def corners_inside(rows: int, cols: int) -> bool:
        for row in (0, rows - 1):
            for col in (0, cols - 1):
                offset_x, offset_y = _rotate(
                    col * int(pitch_x), row * int(pitch_y), cos_t, sin_t
                )
                center_x = int(origin_x) + offset_x
                center_y = int(origin_y) + offset_y
                if not (0 <= center_x < width and 0 <= center_y < height):
                    return False
        return True

    max_cols = 1
    while max_cols < limit and corners_inside(1, max_cols + 1):
        max_cols += 1
    max_rows = 1
    while max_rows < limit and corners_inside(max_rows + 1, max_cols):
        max_rows += 1
    return (max_rows, max_cols)
```

- [x] **Step 4: Move the old copy and repoint its importers**

Delete `generate_grid_shapes` from `arena_grid_dialog.py` entirely (the whole `def generate_grid_shapes(...)` block and its docstring). Replace it with an import at the top of the file:

```python
from hydra_suite.trackerkit.arena_geometry import generate_grid_shapes
```

In `tests/test_arena_grid_dialog.py`, change the import block to:

```python
from hydra_suite.trackerkit.arena_geometry import generate_grid_shapes
from hydra_suite.trackerkit.gui.dialogs.arena_grid_dialog import ArenaGridDialog
```

- [x] **Step 5: Run both test files to verify they pass**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_grid_geometry.py tests/test_arena_grid_dialog.py -v`
Expected: PASS. The pre-existing `test_arena_grid_dialog.py` cases must all still pass unchanged — the new keyword-only parameters default to the old behaviour, so the positional calls are unaffected.

- [x] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/arena_geometry.py \
        src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py \
        tests/test_arena_grid_geometry.py tests/test_arena_grid_dialog.py
git commit -m "feat(arena): grid rotation, rectangle sizing, pitch floors and extent caps"
```

---

### Task 4: Overlay style rules

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/widgets/arena_style.py`
- Test: `tests/test_arena_style.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ArenaPalette` dataclass with fields `line_include`, `line_exclude`, `line_preview`, `veil`, `glyph`, `halo` (each an `(r, g, b)` int tuple); `frame_palette(mean_luminance: float) -> ArenaPalette`; `line_width_px(viewport_min_dim: int) -> int`; `glyph_size_px(on_screen_radius: float) -> int`; constants `VEIL_ALPHA = 0.15`, `TEXT_ALPHA = 0.70`, `CLICK_DRAG_THRESHOLD_PX = 3`, `GLYPH_MIN_PX = 10`, `GLYPH_MAX_PX = 64`.

- [x] **Step 1: Write the failing test**

```python
"""Overlay styling rules. Qt-free: returns plain RGB tuples and ints."""

from hydra_suite.trackerkit.gui.widgets.arena_style import (
    CLICK_DRAG_THRESHOLD_PX,
    GLYPH_MAX_PX,
    GLYPH_MIN_PX,
    TEXT_ALPHA,
    VEIL_ALPHA,
    frame_palette,
    glyph_size_px,
    line_width_px,
)


def test_light_frame_gets_a_dark_veil():
    assert frame_palette(0.90).veil == (0, 0, 0)


def test_dark_frame_gets_a_light_veil():
    assert frame_palette(0.10).veil == (255, 255, 255)


def test_glyph_and_halo_are_opposite_poles_on_light_frames():
    palette = frame_palette(0.90)
    assert sum(palette.glyph) < sum(palette.halo)


def test_glyph_and_halo_are_opposite_poles_on_dark_frames():
    palette = frame_palette(0.10)
    assert sum(palette.glyph) > sum(palette.halo)


def test_role_hues_are_distinct():
    """Include, exclude and in-progress must never be confusable."""
    palette = frame_palette(0.50)
    hues = {palette.line_include, palette.line_exclude, palette.line_preview}
    assert len(hues) == 3


def test_exclude_stays_red_on_both_polarities():
    """Role meaning must be learnable, so hue cannot flip with the footage."""
    light = frame_palette(0.90).line_exclude
    dark = frame_palette(0.10).line_exclude
    assert light[0] == max(light) and dark[0] == max(dark)


def test_line_width_grows_with_viewport_but_never_vanishes():
    assert line_width_px(200) >= 2
    assert line_width_px(4000) > line_width_px(400)


def test_glyph_size_is_clamped_at_both_ends():
    assert glyph_size_px(1.0) == GLYPH_MIN_PX
    assert glyph_size_px(100000.0) == GLYPH_MAX_PX


def test_glyph_size_scales_between_the_clamps():
    small = glyph_size_px(30.0)
    large = glyph_size_px(60.0)
    assert GLYPH_MIN_PX <= small < large <= GLYPH_MAX_PX


def test_alpha_and_threshold_constants_match_the_spec():
    assert VEIL_ALPHA == 0.15
    assert TEXT_ALPHA == 0.70
    assert CLICK_DRAG_THRESHOLD_PX == 3
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_style.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.trackerkit.gui.widgets.arena_style'`

- [x] **Step 3: Write the implementation**

```python
"""Arena overlay styling: palette, sizing, alpha constants.

Deliberately Qt-free -- returns plain ``(r, g, b)`` tuples and ints. The
caller converts to ``QColor``. Keeping these rules importable without a
display is what makes them testable (see ``project_main_suite_blockers``).
"""

from __future__ import annotations

from dataclasses import dataclass

VEIL_ALPHA = 0.15
TEXT_ALPHA = 0.70
CLICK_DRAG_THRESHOLD_PX = 3
GLYPH_MIN_PX = 10
GLYPH_MAX_PX = 64

# Below this mean luminance the frame counts as dark and the polarity flips.
_LUMINANCE_MIDPOINT = 0.5
# Divisor turning the viewport's short edge into a line width. 260 gives 2 px
# on a 520 px viewport and 4 px on a 1080 px one -- visible without being fat.
_LINE_WIDTH_DIVISOR = 260


@dataclass(frozen=True)
class ArenaPalette:
    """Colours for one frame's overlay, all as (r, g, b) 0-255 tuples."""

    line_include: tuple[int, int, int]
    line_exclude: tuple[int, int, int]
    line_preview: tuple[int, int, int]
    veil: tuple[int, int, int]
    glyph: tuple[int, int, int]
    halo: tuple[int, int, int]


def frame_palette(mean_luminance: float) -> ArenaPalette:
    """Palette for a frame of the given mean luminance (0.0-1.0).

    Hue is fixed per ROLE -- include blue, exclude red, in-progress green --
    so the meaning stays learnable across videos. Only the light/dark variant
    changes with the footage, along with veil, glyph and halo polarity.
    """
    dark_frame = mean_luminance < _LUMINANCE_MIDPOINT
    if dark_frame:
        return ArenaPalette(
            line_include=(120, 190, 255),
            line_exclude=(255, 120, 110),
            line_preview=(140, 255, 150),
            veil=(255, 255, 255),
            glyph=(255, 255, 255),
            halo=(20, 20, 20),
        )
    return ArenaPalette(
        line_include=(0, 80, 200),
        line_exclude=(200, 25, 25),
        line_preview=(20, 130, 40),
        veil=(0, 0, 0),
        glyph=(20, 20, 20),
        halo=(255, 255, 255),
    )


def line_width_px(viewport_min_dim: int) -> int:
    """Outline width in DEVICE pixels for the given viewport short edge.

    Deriving this from the viewport rather than the image is the whole point:
    a width in image pixels scales with zoom, which is why the current 2 px
    cyan pen vanishes when zoomed out.
    """
    return max(2, int(round(int(viewport_min_dim) / _LINE_WIDTH_DIVISOR)))


def glyph_size_px(on_screen_radius: float) -> int:
    """Arena-number point size for an arena of the given on-screen radius.

    Clamped so 96 wells stay readable and a single large arena does not get an
    absurd number.
    """
    raw = float(on_screen_radius) * 0.8
    return int(max(GLYPH_MIN_PX, min(GLYPH_MAX_PX, raw)))
```

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_arena_style.py -v`
Expected: PASS (10 tests)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/widgets/arena_style.py tests/test_arena_style.py
git commit -m "feat(arena): luminance-driven overlay palette and device-pixel sizing"
```

---

### Task 5: ArenaCanvas coordinate transform and mouse routing

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/widgets/arena_canvas.py`
- Test: `tests/test_arena_canvas_transform.py`

**Interfaces:**
- Consumes: `CLICK_DRAG_THRESHOLD_PX` from `arena_style` (Task 4); `arena_at_point` from `arena_geometry` (Task 1).
- Produces: `ArenaCanvas(QWidget)` with `set_frame(qimage)`, `set_zoom(float)`, `set_shapes(list)`, `set_current_arena(int | None)`, `set_drawing(bool)`, `set_points(list)`, `to_image(QPointF) -> tuple[float, float]`, `to_viewport(x, y) -> QPointF`; signals `point_added = Signal(float, float)`, `point_removed = Signal()`, `arena_clicked = Signal(int)`, `pan_delta = Signal(int, int)`.

This is the task that fixes the reported bug. The round-trip test is the regression guard: if `to_image(to_viewport(p)) != p` at any zoom, clicks land in the wrong place, which is why zoom is currently force-disabled during selection.

- [x] **Step 1: Write the failing test**

```python
"""ArenaCanvas coordinate transform and click/drag disambiguation.

Constructed under offscreen Qt; never shown, never exec()'d.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    widget = ArenaCanvas()
    widget.set_frame(QImage(400, 300, QImage.Format_RGB888))
    return widget


@pytest.mark.parametrize("zoom", [0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
def test_image_viewport_round_trip_is_identity(canvas, zoom):
    """The regression test for the defect underlying both reported problems."""
    canvas.set_zoom(zoom)
    for point in [(0.0, 0.0), (123.0, 45.0), (399.0, 299.0)]:
        back = canvas.to_image(canvas.to_viewport(*point))
        assert back == pytest.approx(point, abs=1e-6)


def test_viewport_origin_maps_to_image_origin(canvas):
    canvas.set_zoom(3.0)
    assert canvas.to_image(QPointF(0.0, 0.0)) == pytest.approx((0.0, 0.0))


def test_zoom_scales_viewport_coordinates(canvas):
    canvas.set_zoom(2.0)
    point = canvas.to_viewport(50.0, 50.0)
    assert (point.x(), point.y()) == pytest.approx((100.0, 100.0))


def test_widget_size_tracks_zoomed_frame(canvas):
    canvas.set_zoom(2.0)
    assert (canvas.width(), canvas.height()) == (800, 600)


def test_small_left_movement_is_a_click(canvas):
    assert canvas._is_click(0, 0, 2, 1) is True


def test_large_left_movement_is_a_drag(canvas):
    assert canvas._is_click(0, 0, 40, 3) is False


def test_threshold_boundary_counts_as_a_drag(canvas):
    """At exactly the threshold the gesture is a drag, so a click is strictly under."""
    assert canvas._is_click(0, 0, 3, 0) is False
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_canvas_transform.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.trackerkit.gui.widgets.arena_canvas'`

- [x] **Step 3: Write the implementation**

```python
"""ArenaCanvas: the video preview widget that owns arena drawing.

Replaces the previous ``QLabel`` whose event handlers were monkeypatched
from ``MainWindow``. The decisive difference is WHERE the overlay is painted:
the old code painted into the frame's image pixels and scaled the result, so
a 2 px pen was 2 IMAGE pixels (apparent width scaled with zoom) and click
coordinates were only valid at 100% zoom -- which is why zoom had to be
force-disabled while drawing. Here the frame is painted scaled and the
overlay is painted afterwards in WIDGET coordinates, so pen widths are device
pixels and clicks map back through the inverse transform at any zoom.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from hydra_suite.trackerkit.arena_geometry import arena_at_point
from hydra_suite.trackerkit.gui.widgets.arena_style import CLICK_DRAG_THRESHOLD_PX


class ArenaCanvas(QWidget):
    """Frame display plus arena overlay, with an explicit image/viewport map."""

    point_added = Signal(float, float)
    point_removed = Signal()
    arena_clicked = Signal(int)
    pan_delta = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: QImage | None = None
        self._scaled: QPixmap | None = None
        self._zoom = 1.0
        self._shapes: list[dict[str, Any]] = []
        self._points: list[tuple[float, float]] = []
        self._current_arena: int | None = None
        self._drawing = False
        self._press_pos: QPointF | None = None
        self._panning = False
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # -- state ------------------------------------------------------------

    def set_frame(self, image: QImage | None) -> None:
        self._frame = image
        self._rescale()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.1, float(zoom))
        self._rescale()

    def set_shapes(self, shapes: list[dict[str, Any]] | None) -> None:
        self._shapes = list(shapes or [])
        self.update()

    def set_points(self, points: list[tuple[float, float]] | None) -> None:
        self._points = list(points or [])
        self.update()

    def set_current_arena(self, arena_id: int | None) -> None:
        self._current_arena = arena_id
        self.update()

    def set_drawing(self, drawing: bool) -> None:
        self._drawing = bool(drawing)
        self.setCursor(Qt.CrossCursor if drawing else Qt.OpenHandCursor)
        self.setContextMenuPolicy(
            Qt.PreventContextMenu if drawing else Qt.DefaultContextMenu
        )
        self.update()

    def _rescale(self) -> None:
        if self._frame is None:
            self._scaled = None
            return
        width = max(1, int(self._frame.width() * self._zoom))
        height = max(1, int(self._frame.height() * self._zoom))
        self._scaled = QPixmap.fromImage(
            self._frame.scaled(
                width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
        )
        self.setFixedSize(width, height)
        self.update()

    # -- transform --------------------------------------------------------

    def to_image(self, point: QPointF) -> tuple[float, float]:
        """Widget coordinates -> image coordinates."""
        return (point.x() / self._zoom, point.y() / self._zoom)

    def to_viewport(self, x: float, y: float) -> QPointF:
        """Image coordinates -> widget coordinates."""
        return QPointF(float(x) * self._zoom, float(y) * self._zoom)

    @staticmethod
    def _is_click(x0: float, y0: float, x1: float, y1: float) -> bool:
        """Whether a press/release pair was a click rather than a drag.

        One button must serve both marking and panning, so displacement
        decides. Strictly under the threshold, so a gesture exactly at the
        threshold is a drag.
        """
        return (abs(x1 - x0) < CLICK_DRAG_THRESHOLD_PX) and (
            abs(y1 - y0) < CLICK_DRAG_THRESHOLD_PX
        )

    # -- input ------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._press_pos = QPointF(event.position())
        if event.button() == Qt.MiddleButton:
            self._panning = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_pos is None:
            event.accept()
            return
        current = QPointF(event.position())
        if not self._is_click(
            self._press_pos.x(), self._press_pos.y(), current.x(), current.y()
        ):
            self._panning = True
            self.pan_delta.emit(
                int(current.x() - self._press_pos.x()),
                int(current.y() - self._press_pos.y()),
            )
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        press, self._press_pos = self._press_pos, None
        panning, self._panning = self._panning, False
        if press is None:
            event.accept()
            return
        release = QPointF(event.position())
        was_click = self._is_click(press.x(), press.y(), release.x(), release.y())

        if event.button() == Qt.RightButton and self._drawing:
            self.point_removed.emit()
        elif event.button() == Qt.LeftButton and was_click and not panning:
            image_x, image_y = self.to_image(release)
            if self._drawing:
                self.point_added.emit(image_x, image_y)
            else:
                arena_id = arena_at_point(self._shapes, image_x, image_y)
                if arena_id is not None:
                    self.arena_clicked.emit(arena_id)
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self._scaled is not None:
            painter.drawPixmap(0, 0, self._scaled)
        painter.end()
```

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_canvas_transform.py -v`
Expected: PASS (10 tests: 6 parametrized round-trips plus 4 others)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/widgets/arena_canvas.py tests/test_arena_canvas_transform.py
git commit -m "feat(arena): ArenaCanvas with viewport-space transform and click/drag routing"
```

---

### Task 6: Overlay painting

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/widgets/arena_canvas.py`
- Test: `tests/test_arena_canvas_paint.py`

**Interfaces:**
- Consumes: `frame_palette`, `line_width_px`, `glyph_size_px`, `VEIL_ALPHA`, `TEXT_ALPHA` (Task 4); `shape_centroid` (Task 1).
- Produces: `ArenaCanvas.render_overlay(painter)`, `ArenaCanvas.mean_luminance() -> float`, and a module-level `paint_arena_number(painter, text, center, size_px, glyph_rgb, halo_rgb, alpha)`.

The glyph and its halo render into an offscreen full-opacity ARGB layer that composites once. Stroking and filling separately at partial alpha would double-composite where they overlap and let the halo bleed through the glyph edge.

- [x] **Step 1: Write the failing test**

```python
"""Overlay painting: veil polarity, zoom-invariant weight, halo compositing."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.widgets.arena_canvas import (  # noqa: E402
    ArenaCanvas,
    paint_arena_number,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _frame(app, gray):
    image = QImage(200, 200, QImage.Format_RGB888)
    image.fill(QColor(gray, gray, gray))
    return image


def _canvas(app, gray=230):
    widget = ArenaCanvas()
    widget.set_frame(_frame(app, gray))
    widget.set_shapes(
        [
            {
                "type": "circle",
                "params": (100, 100, 40),
                "mode": "include",
                "arena_id": 0,
            }
        ]
    )
    return widget


def test_mean_luminance_reads_a_light_frame(app):
    assert _canvas(app, 230).mean_luminance() > 0.8


def test_mean_luminance_reads_a_dark_frame(app):
    assert _canvas(app, 20).mean_luminance() < 0.2


def _rendered(canvas):
    out = QImage(canvas.width(), canvas.height(), QImage.Format_RGB888)
    out.fill(QColor(255, 255, 255))
    painter = QPainter(out)
    painter.drawPixmap(0, 0, canvas._scaled)
    canvas.render_overlay(painter)
    painter.end()
    return out


def test_veil_darkens_the_arena_interior_on_light_footage(app):
    """Veil goes INSIDE the ROI, per the design decision."""
    canvas = _canvas(app, 230)
    out = _rendered(canvas)
    inside = QColor(out.pixel(100, 100)).lightness()
    outside = QColor(out.pixel(5, 5)).lightness()
    assert inside < outside


def test_veil_lightens_the_arena_interior_on_dark_footage(app):
    canvas = _canvas(app, 20)
    out = _rendered(canvas)
    inside = QColor(out.pixel(100, 100)).lightness()
    outside = QColor(out.pixel(5, 5)).lightness()
    assert inside > outside


def test_exclude_hole_is_not_veiled(app):
    """An exclude zone is outside the ROI, so it must not carry the veil."""
    canvas = _canvas(app, 230)
    canvas.set_shapes(
        [
            {
                "type": "circle",
                "params": (100, 100, 40),
                "mode": "include",
                "arena_id": 0,
            },
            {
                "type": "circle",
                "params": (100, 100, 15),
                "mode": "exclude",
                "arena_id": 0,
            },
        ]
    )
    out = _rendered(canvas)
    in_hole = QColor(out.pixel(100, 100)).lightness()
    in_ring = QColor(out.pixel(130, 100)).lightness()
    assert in_hole > in_ring


def test_outline_width_is_independent_of_zoom(app):
    """The core requirement: constant APPARENT thickness at any zoom."""
    canvas = _canvas(app, 230)
    canvas.set_zoom(1.0)
    width_at_1x = canvas._line_width()
    canvas.set_zoom(4.0)
    assert canvas._line_width() == width_at_1x


def test_paint_arena_number_draws_a_dark_glyph_over_a_light_halo(app):
    """Halo and glyph composite once, so their overlap is not double-darkened.

    A dark glyph on a white halo over mid grey must leave the glyph body
    DARKER than the surrounding halo ring, and the halo LIGHTER than the
    untouched background. Double-compositing would darken the halo/glyph
    overlap and invert the second relationship.
    """
    out = QImage(160, 160, QImage.Format_ARGB32)
    out.fill(QColor(128, 128, 128))
    painter = QPainter(out)
    paint_arena_number(
        painter, "1", QPointF(80, 80), 90, (20, 20, 20), (255, 255, 255), 0.70
    )
    painter.end()

    lightness = [
        [QColor(out.pixel(x, y)).lightness() for x in range(160)]
        for y in range(160)
    ]
    flat = [v for row in lightness for v in row]
    background = QColor(out.pixel(2, 2)).lightness()
    assert min(flat) < background, "no dark glyph body was drawn"
    assert max(flat) > background, "no light halo was drawn"


def test_paint_arena_number_respects_alpha(app):
    """At alpha 0 nothing is drawn; the whole layer is composited once."""
    out = QImage(160, 160, QImage.Format_ARGB32)
    out.fill(QColor(128, 128, 128))
    painter = QPainter(out)
    paint_arena_number(
        painter, "1", QPointF(80, 80), 90, (20, 20, 20), (255, 255, 255), 0.0
    )
    painter.end()
    assert QColor(out.pixel(80, 80)).lightness() == QColor(128, 128, 128).lightness()


def test_current_arena_outline_is_heavier(app):
    canvas = _canvas(app, 230)
    canvas.set_current_arena(None)
    plain = canvas._outline_width_for(0)
    canvas.set_current_arena(0)
    assert canvas._outline_width_for(0) > plain
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_canvas_paint.py -v`
Expected: FAIL — `ImportError: cannot import name 'paint_arena_number'`

- [x] **Step 3: Write the implementation**

Add these imports to `arena_canvas.py`:

```python
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPainterPathStroker, QPen, QPolygonF

from hydra_suite.trackerkit.arena_geometry import shape_centroid
from hydra_suite.trackerkit.gui.widgets.arena_style import (
    TEXT_ALPHA,
    VEIL_ALPHA,
    frame_palette,
    glyph_size_px,
    line_width_px,
)
```

Add the module-level helper:

```python
def paint_arena_number(
    painter: QPainter,
    text: str,
    center: QPointF,
    size_px: int,
    glyph_rgb: tuple[int, int, int],
    halo_rgb: tuple[int, int, int],
    alpha: float,
) -> None:
    """Draw a haloed arena number, composited ONCE at *alpha*.

    The glyph and its halo are rendered into an offscreen ARGB layer at full
    opacity and that layer is composited in one pass. Stroking and filling
    directly at partial alpha would composite twice where the halo underlies
    the glyph, letting the halo bleed through the glyph edge and making the
    number look doubled.
    """
    font = QFont()
    font.setPixelSize(int(size_px))
    font.setBold(True)

    path = QPainterPath()
    path.addText(0.0, 0.0, font, text)
    bounds = path.boundingRect()
    path.translate(-bounds.center().x(), -bounds.center().y())

    stroker = QPainterPathStroker()
    stroker.setWidth(max(2.0, size_px * 0.18))
    halo_path = stroker.createStroke(path)

    pad = int(max(4.0, size_px * 0.5))
    layer_rect = path.boundingRect().united(halo_path.boundingRect())
    layer = QImage(
        int(layer_rect.width()) + 2 * pad,
        int(layer_rect.height()) + 2 * pad,
        QImage.Format_ARGB32_Premultiplied,
    )
    layer.fill(Qt.transparent)

    layer_painter = QPainter(layer)
    layer_painter.setRenderHint(QPainter.Antialiasing, True)
    layer_painter.translate(
        pad - layer_rect.left(),
        pad - layer_rect.top(),
    )
    layer_painter.fillPath(halo_path, QBrush(QColor(*halo_rgb)))
    layer_painter.fillPath(path, QBrush(QColor(*glyph_rgb)))
    layer_painter.end()

    painter.save()
    painter.setOpacity(float(alpha))
    painter.drawImage(
        QPointF(
            center.x() - layer.width() / 2.0,
            center.y() - layer.height() / 2.0,
        ),
        layer,
    )
    painter.restore()
```

Add these methods to `ArenaCanvas`:

```python
    def mean_luminance(self) -> float:
        """Mean luminance of the base frame, 0.0-1.0. Cached per frame."""
        if self._frame is None:
            return 0.5
        if self._luminance is None:
            small = self._frame.scaled(
                64, 64, Qt.IgnoreAspectRatio, Qt.FastTransformation
            ).convertToFormat(QImage.Format_Grayscale8)
            buffer = np.frombuffer(
                small.constBits(), dtype=np.uint8, count=small.sizeInBytes()
            )
            self._luminance = float(buffer.mean()) / 255.0
        return self._luminance

    def _palette(self):
        return frame_palette(self.mean_luminance())

    def _line_width(self) -> int:
        """Device-pixel outline width -- independent of zoom by construction."""
        return line_width_px(min(self.parentWidth(), self.parentHeight()))

    def parentWidth(self) -> int:
        parent = self.parentWidget()
        return parent.width() if parent is not None else 800

    def parentHeight(self) -> int:
        parent = self.parentWidget()
        return parent.height() if parent is not None else 600

    def _outline_width_for(self, arena_id: int) -> int:
        base = self._line_width()
        return base * 2 if arena_id == self._current_arena else base

    def _shape_path(self, shape: dict[str, Any]) -> QPainterPath:
        """The shape as a viewport-space path."""
        path = QPainterPath()
        if shape.get("type") == "circle":
            cx, cy, radius = (float(v) for v in shape["params"])
            top_left = self.to_viewport(cx - radius, cy - radius)
            path.addEllipse(
                QRectF(
                    top_left.x(),
                    top_left.y(),
                    2.0 * radius * self._zoom,
                    2.0 * radius * self._zoom,
                )
            )
        else:
            polygon = QPolygonF([self.to_viewport(x, y) for x, y in shape["params"]])
            path.addPolygon(polygon)
            path.closeSubpath()
        return path

    def render_overlay(self, painter: QPainter) -> None:
        """Paint veil, outlines, in-progress points and arena numbers.

        Everything here is in WIDGET coordinates: pen widths and glyph sizes
        are device pixels, so apparent size does not change with zoom.
        """
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self._palette()

        include = [s for s in self._shapes if s.get("mode", "include") == "include"]
        exclude = [s for s in self._shapes if s.get("mode", "include") == "exclude"]

        # Veil: inside the include region, minus every exclude hole.
        if include:
            veil_path = QPainterPath()
            for shape in include:
                veil_path = veil_path.united(self._shape_path(shape))
            for shape in exclude:
                veil_path = veil_path.subtracted(self._shape_path(shape))
            painter.save()
            painter.setOpacity(VEIL_ALPHA)
            painter.fillPath(veil_path, QBrush(QColor(*palette.veil)))
            painter.restore()

        for shape in self._shapes:
            is_include = shape.get("mode", "include") == "include"
            colour = palette.line_include if is_include else palette.line_exclude
            width = self._outline_width_for(int(shape.get("arena_id", 0)))
            painter.setPen(QPen(QColor(*colour), width))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._shape_path(shape))

        # In-progress points and their preview outline.
        if self._points:
            preview = QColor(*palette.line_preview)
            painter.setPen(QPen(preview, self._line_width() * 2))
            for x, y in self._points:
                painter.drawPoint(self.to_viewport(x, y))

        for arena_id in sorted({int(s.get("arena_id", 0)) for s in include}):
            members = [s for s in include if int(s.get("arena_id", 0)) == arena_id]
            centroids = [shape_centroid(s) for s in members]
            center_x = sum(c[0] for c in centroids) / len(centroids)
            center_y = sum(c[1] for c in centroids) / len(centroids)
            box = self._shape_path(members[0]).boundingRect()
            paint_arena_number(
                painter,
                str(arena_id + 1),
                self.to_viewport(center_x, center_y),
                glyph_size_px(min(box.width(), box.height()) / 2.0),
                palette.glyph,
                palette.halo,
                TEXT_ALPHA,
            )
```

In `__init__`, add `self._luminance: float | None = None`, and in `set_frame` set `self._luminance = None` before calling `self._rescale()`.

In `paintEvent`, call `self.render_overlay(painter)` after `drawPixmap` and before `painter.end()`.

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_canvas_paint.py -v`
Expected: PASS (9 tests)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/widgets/arena_canvas.py tests/test_arena_canvas_paint.py
git commit -m "feat(arena): viewport-space overlay with veil, zoom-invariant outlines and haloed numbers"
```

---

### Task 7: Arena panel state machine

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/panels/arena_panel.py`
- Test: `tests/test_arena_panel.py`

**Interfaces:**
- Consumes: `overlapping_arena_pairs` (Task 2), `next_free_arena_id` from `hydra_suite.trackerkit.engine_params`.
- Produces: `ArenaPanel(QWidget)` with `set_shapes(list)`, `set_frame_size(width, height)`, `current_arena -> int`, `arena_ids() -> list[int]`, `refresh()`, `blocking_pairs() -> list[tuple[int,int]]`, `can_track() -> tuple[bool, str]`; signals `arena_changed = Signal(int)`, `add_single_requested = Signal()`, `add_grid_requested = Signal()`, `clear_arena_requested = Signal(int)`, `draw_requested = Signal(str, str)` (shape type, zone mode), `finish_requested = Signal()`, `undo_requested = Signal()`, `clear_all_requested = Signal()`, `crop_requested = Signal()`.

- [x] **Step 1: Write the failing test**

```python
"""Arena bar: two-state machine and the overlap lock's enable/disable matrix."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.panels.arena_panel import ArenaPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    widget = ArenaPanel()
    widget.set_frame_size(400, 400)
    return widget


def _circle(cx, cy, r, arena_id, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def test_empty_state_shows_only_the_two_add_buttons(panel):
    panel.set_shapes([])
    assert panel.stack.currentWidget() is panel.empty_widget
    assert panel.btn_add_single.isEnabled()
    assert panel.btn_add_grid.isEnabled()


def test_empty_state_message(panel):
    panel.set_shapes([])
    assert panel.lbl_default.text() == "By default, the whole video is used."


def test_adding_a_shape_enters_editing_state(panel):
    panel.set_shapes([_circle(100, 100, 30, 0)])
    assert panel.lbl_current.text() == "Currently labelling: Arena 1"


def test_previous_is_disabled_on_the_first_arena(panel):
    panel.set_shapes([_circle(50, 50, 20, 0), _circle(200, 200, 20, 1)])
    panel.set_current_arena(0)
    assert panel.btn_prev.isEnabled() is False
    assert panel.btn_next.isEnabled() is True


def test_next_is_disabled_on_the_last_arena(panel):
    panel.set_shapes([_circle(50, 50, 20, 0), _circle(200, 200, 20, 1)])
    panel.set_current_arena(1)
    assert panel.btn_next.isEnabled() is False


def test_add_new_arena_blocked_while_current_arena_is_empty(panel):
    """Empty arenas inflate MAX_TARGETS (n_arenas * animals_per_arena)."""
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.begin_new_arena()
    assert panel.btn_add_new.isEnabled() is False


def test_navigation_locked_while_the_current_arena_overlaps(panel):
    panel.set_shapes([_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)])
    panel.set_current_arena(0)
    assert panel.btn_next.isEnabled() is False
    assert panel.btn_add_new.isEnabled() is False
    assert "overlap" in panel.lbl_warning.text().lower()


def test_warning_names_the_conflicting_arena(panel):
    panel.set_shapes([_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)])
    panel.set_current_arena(0)
    assert "2" in panel.lbl_warning.text()


def test_navigation_free_when_a_distant_pair_overlaps(panel):
    """The lock must never strand the user away from the arenas they must fix."""
    shapes = [
        _circle(30, 30, 15, 0),
        _circle(200, 200, 50, 1),
        _circle(230, 200, 50, 2),
    ]
    panel.set_shapes(shapes)
    panel.set_current_arena(0)
    assert panel.btn_next.isEnabled() is True


def test_tracking_blocked_by_any_overlap_anywhere(panel):
    shapes = [
        _circle(30, 30, 15, 0),
        _circle(200, 200, 50, 1),
        _circle(230, 200, 50, 2),
    ]
    panel.set_shapes(shapes)
    panel.set_current_arena(0)
    allowed, reason = panel.can_track()
    assert allowed is False
    assert "2" in reason and "3" in reason


def test_tracking_allowed_when_nothing_overlaps(panel):
    panel.set_shapes([_circle(50, 50, 20, 0), _circle(300, 300, 20, 1)])
    allowed, _reason = panel.can_track()
    assert allowed is True


def test_clear_arena_keeps_the_arena_and_its_number(panel):
    shapes = [_circle(50, 50, 20, 0), _circle(300, 300, 20, 1)]
    panel.set_shapes(shapes)
    panel.set_current_arena(1)
    remaining = panel.shapes_after_clearing(1)
    assert remaining == [shapes[0]]
    assert panel.current_arena == 1


def test_finish_disabled_until_a_shape_is_valid(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.set_shape_valid(False)
    assert panel.btn_finish.isEnabled() is False
    panel.set_shape_valid(True)
    assert panel.btn_finish.isEnabled() is True
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.trackerkit.gui.panels.arena_panel'`

- [x] **Step 3: Write the implementation**

```python
"""The arena bar: an arena-centric replacement for the old ROI toolbar.

Two states. Empty shows only a sentence and two add buttons. Editing shows
arena navigation, the zone tools, and the overlap warning. The
include/exclude combo is gone -- zone role is chosen by pressing "+ Circle"
versus "- Circle", so the user never sets a mode before drawing.

One arena is exactly a plain ROI: with a single arena, ``n_arenas == 1``
suppresses the ``arena_id`` column and every per-arena path degenerates, so
a user who only wants to mask out junk never meets arena numbering.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs


class ArenaPanel(QWidget):
    """Arena navigation, zone tools and the overlap lock."""

    arena_changed = Signal(int)
    add_single_requested = Signal()
    add_grid_requested = Signal()
    clear_arena_requested = Signal(int)
    draw_requested = Signal(str, str)
    finish_requested = Signal()
    undo_requested = Signal()
    clear_all_requested = Signal()
    crop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shapes: list[dict[str, Any]] = []
        self._frame_size = (0, 0)
        self._current = 0
        self._pending_new = False
        self._shape_valid = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        self.stack = QStackedWidget()
        self.empty_widget = self._build_empty()
        self.editing_widget = self._build_editing()
        self.stack.addWidget(self.empty_widget)
        self.stack.addWidget(self.editing_widget)
        root.addWidget(self.stack)

        self.lbl_warning = QLabel("")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setStyleSheet(
            "color: #ffffff; font-weight: bold; padding: 6px; "
            "background-color: #8a1f1f; border-radius: 4px;"
        )
        self.lbl_warning.setVisible(False)
        root.addWidget(self.lbl_warning)

        self.refresh()

    def _build_empty(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_default = QLabel("By default, the whole video is used.")
        self.lbl_default.setStyleSheet("color: #cccccc;")
        self.btn_add_single = QPushButton("+ Add Single Arena")
        self.btn_add_grid = QPushButton("+ Add Grid of Arenas")
        self.btn_add_single.clicked.connect(self.add_single_requested.emit)
        self.btn_add_grid.clicked.connect(self.add_grid_requested.emit)
        layout.addWidget(self.lbl_default)
        layout.addWidget(self.btn_add_single)
        layout.addWidget(self.btn_add_grid)
        layout.addStretch()
        return widget

    def _build_editing(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_current = QLabel("Currently labelling: Arena 1")
        self.lbl_current.setStyleSheet("font-weight: bold; color: #cccccc;")
        self.btn_prev = QPushButton("< Previous")
        self.btn_next = QPushButton("Next >")
        self.btn_add_new = QPushButton("+ Add new arena")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_add_new.clicked.connect(self.begin_new_arena)

        self.lbl_hint = QLabel(
            "Add Inclusion and Exclusion Zones "
            "(Left-click marks, Right-click removes the last point)"
        )
        self.lbl_hint.setStyleSheet("color: #4fc1ff; font-size: 11px;")

        self.btn_clear_arena = QPushButton("Clear Arena")
        self.btn_clear_arena.clicked.connect(
            lambda: self.clear_arena_requested.emit(self._current)
        )
        self.btn_add_circle = QPushButton("+ Circle")
        self.btn_sub_circle = QPushButton("- Circle")
        self.btn_add_polygon = QPushButton("+ Polygon")
        self.btn_sub_polygon = QPushButton("- Polygon")
        self.btn_add_circle.clicked.connect(
            lambda: self.draw_requested.emit("circle", "include")
        )
        self.btn_sub_circle.clicked.connect(
            lambda: self.draw_requested.emit("circle", "exclude")
        )
        self.btn_add_polygon.clicked.connect(
            lambda: self.draw_requested.emit("polygon", "include")
        )
        self.btn_sub_polygon.clicked.connect(
            lambda: self.draw_requested.emit("polygon", "exclude")
        )
        self.btn_finish = QPushButton("Finish Shape")
        self.btn_finish.setEnabled(False)
        self.btn_finish.clicked.connect(self.finish_requested.emit)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo_requested.emit)

        self.btn_overflow = QToolButton()
        self.btn_overflow.setText("...")
        self.btn_overflow.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_overflow)
        menu.addAction("Add Grid of Arenas", self.add_grid_requested.emit)
        menu.addAction("Clear All", self.clear_all_requested.emit)
        menu.addAction("Crop Video to ROI", self.crop_requested.emit)
        self.btn_overflow.setMenu(menu)

        for w in (
            self.lbl_current,
            self.btn_prev,
            self.btn_next,
            self.btn_add_new,
            self._separator(),
            self.lbl_hint,
            self.btn_clear_arena,
            self.btn_add_circle,
            self.btn_sub_circle,
            self.btn_add_polygon,
            self.btn_sub_polygon,
            self.btn_finish,
            self.btn_undo,
            self.btn_overflow,
        ):
            layout.addWidget(w)
        layout.addStretch()
        return widget

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #3e3e42;")
        return line

    # -- state ------------------------------------------------------------

    @property
    def current_arena(self) -> int:
        return self._current

    def arena_ids(self) -> list[int]:
        return sorted(
            {
                int(s.get("arena_id", 0))
                for s in self._shapes
                if s.get("mode", "include") == "include"
            }
        )

    def set_shapes(self, shapes: list[dict[str, Any]] | None) -> None:
        self._shapes = list(shapes or [])
        if self._shapes:
            self._pending_new = False
        self.refresh()

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_size = (int(width), int(height))
        self.refresh()

    def set_current_arena(self, arena_id: int) -> None:
        self._current = int(arena_id)
        self.refresh()

    def set_shape_valid(self, valid: bool) -> None:
        self._shape_valid = bool(valid)
        self.btn_finish.setEnabled(self._shape_valid)

    def begin_new_arena(self) -> int:
        """Start a fresh arena; it holds no shapes until the user draws one."""
        from hydra_suite.trackerkit.engine_params import next_free_arena_id

        self._current = next_free_arena_id(self._shapes)
        self._pending_new = True
        self.refresh()
        self.arena_changed.emit(self._current)
        return self._current

    def shapes_after_clearing(self, arena_id: int) -> list[dict[str, Any]]:
        """The shape list with *arena_id*'s shapes removed; the arena remains.

        Numbering is untouched -- arena ids appear in the exported
        ``arena_id`` column, so renumbering would silently change what a
        number refers to.
        """
        return [s for s in self._shapes if int(s.get("arena_id", 0)) != int(arena_id)]

    def _step(self, delta: int) -> None:
        ids = self.arena_ids()
        if self._current not in ids:
            return
        index = ids.index(self._current) + delta
        if 0 <= index < len(ids):
            self._current = ids[index]
            self.refresh()
            self.arena_changed.emit(self._current)

    # -- lock -------------------------------------------------------------

    def blocking_pairs(self) -> list[tuple[int, int]]:
        width, height = self._frame_size
        if not width or not height:
            return []
        return overlapping_arena_pairs(self._shapes, width, height)

    def can_track(self) -> tuple[bool, str]:
        """Whether tracking may start, and why not if it may not."""
        pairs = self.blocking_pairs()
        if not pairs:
            return (True, "")
        listed = ", ".join(f"Arena {a + 1} and Arena {b + 1}" for a, b in pairs)
        return (
            False,
            f"Arenas overlap: {listed}. Each animal must belong to exactly one "
            "arena, so tracking cannot start until the overlaps are resolved.",
        )

    def refresh(self) -> None:
        ids = self.arena_ids()
        if not self._shapes and not self._pending_new:
            self.stack.setCurrentWidget(self.empty_widget)
            self.lbl_warning.setVisible(False)
            return
        self.stack.setCurrentWidget(self.editing_widget)
        self.lbl_current.setText(f"Currently labelling: Arena {self._current + 1}")

        pairs = self.blocking_pairs()
        conflicts = sorted(
            {
                (b if a == self._current else a)
                for a, b in pairs
                if self._current in (a, b)
            }
        )
        current_blocked = bool(conflicts)
        current_empty = self._current not in ids

        if current_blocked:
            listed = ", ".join(f"Arena {c + 1}" for c in conflicts)
            self.lbl_warning.setText(
                f"Arena {self._current + 1} overlaps {listed}. "
                "Resolve the overlap before moving on -- an animal in the shared "
                "region cannot be assigned to a single arena."
            )
        elif pairs:
            self.lbl_warning.setText(self.can_track()[1])
        self.lbl_warning.setVisible(bool(pairs))

        index = ids.index(self._current) if self._current in ids else -1
        self.btn_prev.setEnabled(not current_blocked and index > 0)
        self.btn_next.setEnabled(
            not current_blocked and index >= 0 and index < len(ids) - 1
        )
        self.btn_add_new.setEnabled(not current_blocked and not current_empty)
        self.btn_clear_arena.setEnabled(not current_empty)
        self.btn_undo.setEnabled(bool(self._shapes))
```

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_panel.py -v`
Expected: PASS (14 tests)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/arena_panel.py tests/test_arena_panel.py
git commit -m "feat(arena): arena bar state machine with non-stranding overlap lock"
```

---

### Task 8: Wire ArenaCanvas into the main window

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:553-570` (video label construction), `:2770-2776` (`_set_video_pixmap`)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/session.py:1734-1800`, `:1897-1913`, `:1950-2068`, `:2100`, `:2184`, `:2294`
- Test: `tests/test_arena_canvas_wiring.py`

**Interfaces:**
- Consumes: `ArenaCanvas` (Tasks 5-6).
- Produces: `MainWindow.video_label` is an `ArenaCanvas`; `roi_selection_active` no longer disables the zoom slider.

**The three zoom locks to delete** (they exist only because click coordinates were previously image coordinates):
1. `session.py:2100` — `self._mw.slider_zoom.setEnabled(False)` in `start_roi_selection`
2. `session.py:1790-1796` — the `if self._mw.roi_selection_active: evt.ignore(); return` guard in `_handle_video_wheel`
3. `session.py:1739-1741` — the `if self._mw.roi_selection_active: record_roi_click(evt); return` short-circuit in `_handle_video_mouse_press`, now handled by the canvas

Lines 2184 and 2294 re-enable the slider; those become unnecessary but are harmless to leave. Delete them for clarity.

- [x] **Step 1: Write the failing test**

```python
"""The main window's preview is an ArenaCanvas and zoom survives ROI drawing."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def test_preview_widget_is_an_arena_canvas(window):
    assert isinstance(window.video_label, ArenaCanvas)


def test_zoom_slider_stays_enabled_during_roi_drawing(window):
    """The bug: zoom used to be force-disabled because clicks were image coords."""
    window.roi_selection_active = True
    window._session_orch._sync_contextual_controls()
    assert window.slider_zoom.isEnabled() is True


def test_source_has_no_remaining_roi_zoom_locks():
    """Guards the three deletions -- a re-introduced lock silently regresses this."""
    import inspect

    from hydra_suite.trackerkit.gui.orchestrators import session

    source = inspect.getsource(session)
    assert "slider_zoom.setEnabled(False)" not in source
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_canvas_wiring.py -v`
Expected: FAIL — `assert isinstance(...)` fails, `video_label` is still a `QLabel`

- [x] **Step 3: Replace the widget in `main_window.py`**

Replace lines 553-570 (from `self.video_label = QLabel("")` through `self.video_label.event = self._handle_video_event`) with:

```python
        self.video_label = ArenaCanvas()
        self.scroll.setWidget(self.video_label)
        self._show_video_logo_placeholder()

        # Pan is driven by the canvas: it reports a delta only for gestures it
        # classified as drags, so a point-marking click never scrolls.
        self.video_label.pan_delta.connect(self._on_canvas_pan)
        self.video_label.point_added.connect(self._on_canvas_point_added)
        self.video_label.point_removed.connect(self._on_canvas_point_removed)
        self.video_label.arena_clicked.connect(self._on_canvas_arena_clicked)
        self.video_label.wheelEvent = self._handle_video_wheel
        self.video_label.mouseDoubleClickEvent = self._handle_video_double_click
        self.video_label.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.video_label.grabGesture(Qt.PinchGesture)
        self.video_label.event = self._handle_video_event
```

Add the import at the top of `main_window.py`:

```python
from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas
```

Add these handlers next to the other video handlers (near line 2182):

```python
    def _on_canvas_pan(self, dx: int, dy: int) -> None:
        """Scroll by a drag delta the canvas already classified as a pan."""
        self.scroll.horizontalScrollBar().setValue(
            self.scroll.horizontalScrollBar().value() - dx
        )
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().value() - dy
        )

    def _on_canvas_point_added(self, x: float, y: float) -> None:
        """A left-click while drawing: append an ROI point in IMAGE coordinates."""
        self._session_orch.add_roi_point(x, y)

    def _on_canvas_point_removed(self) -> None:
        """A right-click while drawing: drop the most recent ROI point."""
        self._session_orch.remove_last_roi_point()

    def _on_canvas_arena_clicked(self, arena_id: int) -> None:
        """A left-click outside drawing mode: make that arena current."""
        self._session_orch.set_current_arena(arena_id)
```

Replace `_set_video_pixmap` (line 2770) so it drives the canvas rather than a label:

```python
    def _set_video_pixmap(self, pixmap: QPixmap):
        """Display a pixmap on the canvas."""
        self.video_label.set_frame(pixmap.toImage())
```

- [x] **Step 4: Rework the drawing methods in `session.py`**

Replace `record_roi_click` (line 1950) with two coordinate-space-explicit methods, and drop the double-click-to-finish handling from it (the canvas forwards `mouseDoubleClickEvent` to `_handle_video_double_click`, which now finishes a polygon when drawing):

```python
    def add_roi_point(self, image_x: float, image_y: float) -> None:
        """Append an ROI point. Coordinates are already IMAGE coordinates.

        The canvas converts through its inverse transform, so this is valid at
        any zoom -- previously the raw label position was stored directly,
        which only agreed with image space at 100% zoom.
        """
        if not self._mw.roi_selection_active:
            return
        self._mw.roi_points.append((image_x, image_y))
        self.update_roi_preview()

    def remove_last_roi_point(self) -> None:
        """Drop the most recent in-progress point."""
        if not self._mw.roi_selection_active or not self._mw.roi_points:
            return
        removed = self._mw.roi_points.pop()
        logger.info(f"Undid last ROI point: ({removed[0]:.1f}, {removed[1]:.1f})")
        self.update_roi_preview()

    def set_current_arena(self, arena_id: int) -> None:
        """Make *arena_id* the arena new shapes join.

        Task 9 adds the ``self._panels.arena.set_current_arena(arena_id)``
        line here once the panel exists; do NOT add it now -- the panel is not
        registered yet and this task's tests would fail on it.
        """
        if self._mw.roi_selection_active:
            return
        self.current_arena_id = int(arena_id)
        self._mw.video_label.set_current_arena(arena_id)
```

Replace the body of `update_roi_preview` (line 1986) so it hands state to the canvas instead of rasterizing:

```python
    def update_roi_preview(self):
        """Push current shapes and in-progress points to the canvas.

        No rasterization here: the canvas paints the overlay in viewport space
        on its next paintEvent. The old implementation deep-copied and
        repainted the whole QImage on every click -- about 61 MB per click on
        a 4512x4512 frame.
        """
        canvas = self._mw.video_label
        canvas.set_shapes(self._mw.roi_shapes)
        canvas.set_points(self._mw.roi_points)
        canvas.set_current_arena(self.current_arena_id)
        canvas.set_drawing(self._mw.roi_selection_active)

        valid = False
        if self._mw.roi_current_mode == "circle" and len(self._mw.roi_points) >= 3:
            circle_fit = fit_circle_to_points(self._mw.roi_points)
            if circle_fit:
                self._mw.roi_fitted_circle = circle_fit
                valid = True
        elif self._mw.roi_current_mode == "polygon" and len(self._mw.roi_points) >= 3:
            valid = True
        # Task 9 replaces this with `self._panels.arena.set_shape_valid(valid)`.
        # The old button still exists at this point; the panel does not.
        self._mw.btn_finish_roi.setEnabled(valid)
```

Delete `_display_roi_with_zoom`'s rasterizing body (line 1897) and replace it with:

```python
    def _display_roi_with_zoom(self):
        """Apply the current zoom to the canvas; the overlay follows."""
        self._mw.video_label.set_zoom(max(self._mw.slider_zoom.value() / 100.0, 0.1))
```

Delete `_draw_roi_overlay` and `_apply_roi_mask_to_image` from `session.py` (lines 2395-2423) and their delegating wrappers in `main_window.py` (lines 2803-2810). Both are superseded by `ArenaCanvas.render_overlay`. Then remove the three zoom locks listed above.

- [x] **Step 5: Run test to verify it passes**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_canvas_wiring.py -v`
Expected: PASS (3 tests)

- [x] **Step 6: Run the whole arena suite for regressions**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/ -k arena -v`
Expected: PASS. The 166 arena tests that passed on `main` at `a2c9838f` must still pass, plus the new ones.

- [x] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/main_window.py \
        src/hydra_suite/trackerkit/gui/orchestrators/session.py \
        tests/test_arena_canvas_wiring.py
git commit -m "refactor(arena): drive the preview through ArenaCanvas and unlock zoom while drawing"
```

---

### Task 9: Replace the ROI toolbar with the arena panel

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:574-693` (the whole ROI toolbar block)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/session.py` (`start_roi_selection`, `finish_roi_selection`, `clear_roi`, `undo_last_roi_shape`)
- Test: `tests/test_arena_panel_wiring.py`

**Interfaces:**
- Consumes: `ArenaPanel` (Task 7).
- Produces: `MainWindow.arena_panel`; `self._panels.arena` resolves to it.

The deleted widgets are `combo_roi_mode`, `combo_roi_zone`, `btn_start_roi`, `btn_finish_roi`, `btn_undo_roi`, `btn_new_arena`, `btn_generate_grid`, `btn_clear_roi`, `btn_crop_video`, and `roi_instructions`. `roi_status_label` and `roi_optimization_label` stay — they carry the ROI efficiency readout, which is unrelated to arena editing. Every reference to a deleted widget must be repointed; find them with:

```bash
grep -rn "combo_roi_mode\|combo_roi_zone\|btn_start_roi\|btn_finish_roi\|btn_undo_roi\|btn_new_arena\|btn_generate_grid\|btn_clear_roi\|btn_crop_video\|roi_instructions" src/ tests/
```

- [x] **Step 1: Write the failing test**

```python
"""The main window hosts an ArenaPanel and no longer has the old ROI toolbar."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit.gui.panels.arena_panel import ArenaPanel  # noqa: E402


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def test_window_has_an_arena_panel(window):
    assert isinstance(window.arena_panel, ArenaPanel)


@pytest.mark.parametrize(
    "attr",
    [
        "combo_roi_mode",
        "combo_roi_zone",
        "btn_start_roi",
        "btn_finish_roi",
        "btn_undo_roi",
        "btn_new_arena",
        "btn_generate_grid",
        "btn_clear_roi",
        "btn_crop_video",
    ],
)
def test_old_roi_toolbar_widgets_are_gone(window, attr):
    assert not hasattr(window, attr)


def test_roi_efficiency_readout_survives(window):
    """Unrelated to arena editing; must not be collateral damage."""
    assert hasattr(window, "roi_status_label")
    assert hasattr(window, "roi_optimization_label")


def test_panel_starts_in_the_empty_state(window):
    window.roi_shapes = []
    window.arena_panel.set_shapes([])
    assert window.arena_panel.lbl_default.text() == (
        "By default, the whole video is used."
    )
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_panel_wiring.py -v`
Expected: FAIL — `AttributeError: 'MainWindow' object has no attribute 'arena_panel'`

- [x] **Step 3: Replace the toolbar block**

In `main_window.py`, replace everything from `roi_layout = QHBoxLayout()` (line 582) through `roi_main_layout.addLayout(roi_layout)` (line 673) with:

```python
        self.arena_panel = ArenaPanel()
        self.arena_panel.add_single_requested.connect(self._on_add_single_arena)
        self.arena_panel.add_grid_requested.connect(self._on_generate_grid_clicked)
        self.arena_panel.arena_changed.connect(self._on_arena_changed)
        self.arena_panel.clear_arena_requested.connect(self._on_clear_arena)
        self.arena_panel.draw_requested.connect(self._on_draw_requested)
        self.arena_panel.finish_requested.connect(self.finish_roi_selection)
        self.arena_panel.undo_requested.connect(self.undo_last_roi_shape)
        self.arena_panel.clear_all_requested.connect(self.clear_roi)
        self.arena_panel.crop_requested.connect(self.crop_video_to_roi)
        roi_main_layout.addWidget(self.arena_panel)
```

Also delete the `roi_label` / `roi_frame` header widgets built at lines 583-585 and the `roi_instructions` block at lines 686-693; the panel carries its own hint text.

Add the import:

```python
from hydra_suite.trackerkit.gui.panels.arena_panel import ArenaPanel
```

Add the new handlers:

```python
    def _on_add_single_arena(self):
        """Start the first (or a fresh) arena and immediately begin a circle."""
        self.arena_panel.begin_new_arena()
        self._session_orch.current_arena_id = self.arena_panel.current_arena
        self._on_draw_requested("circle", "include")

    def _on_arena_changed(self, arena_id: int):
        self._session_orch.set_current_arena(arena_id)

    def _on_clear_arena(self, arena_id: int):
        """Empty an arena's shapes; the arena and its number remain."""
        self.roi_shapes = self.arena_panel.shapes_after_clearing(arena_id)
        self.arena_panel.set_shapes(self.roi_shapes)
        if self.roi_base_frame:
            self._generate_combined_roi_mask(
                self.roi_base_frame.height(), self.roi_base_frame.width()
            )
        self._update_animals_per_arena_total_label()
        self.update_roi_preview()

    def _on_draw_requested(self, shape_type: str, zone_mode: str):
        """Begin drawing a shape of the requested type and zone role."""
        self.roi_current_mode = shape_type
        self.roi_current_zone_type = zone_mode
        self.start_roi_selection()
```

- [x] **Step 4: Repoint the orchestrator**

In `session.py`, delete every reference to a removed widget. Specifically:

- In `start_roi_selection`, delete the `btn_start_roi` / `btn_finish_roi` / `combo_roi_mode` / `combo_roi_zone` / `slider_zoom` disabling block and the `roi_instructions.setText(...)` calls; keep the frame-loading logic and the `roi_selection_active = True` assignment, then call `self.update_roi_preview()`.
- In `finish_roi_selection`, replace the `btn_*` state updates with `self._panels.arena.set_shapes(self._mw.roi_shapes)`.
- In `undo_last_roi_shape`, replace `self._mw.btn_undo_roi.setEnabled(...)` and the `_apply_roi_mask_to_image` redraw with `self._panels.arena.set_shapes(self._mw.roi_shapes)` followed by `self.update_roi_preview()`.
- In `clear_roi`, replace the five `btn_*` / `combo_*` / `slider_zoom` lines with `self._panels.arena.set_shapes([])`.

Register the panel so `self._panels.arena` resolves. In `_panels_bundle` (`main_window.py:993-1004`), add one line beside the existing entries:

```python
        ns.arena = self.arena_panel
```

Then apply the two deferred lines Task 8 left behind:

- In `set_current_arena`, add `self._panels.arena.set_current_arena(arena_id)` before the `self._mw.video_label.set_current_arena(arena_id)` call.
- In `update_roi_preview`, replace `self._mw.btn_finish_roi.setEnabled(valid)` with `self._panels.arena.set_shape_valid(valid)`.

- [x] **Step 5: Verify no dangling references remain**

Run:
```bash
grep -rn "combo_roi_mode\|combo_roi_zone\|btn_start_roi\|btn_finish_roi\|btn_undo_roi\|btn_new_arena\|btn_generate_grid\|btn_clear_roi\|btn_crop_video\|roi_instructions" src/ tests/
```
Expected: no output.

- [x] **Step 6: Run the tests**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_panel_wiring.py tests/ -k arena -v`
Expected: PASS

- [x] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/main_window.py \
        src/hydra_suite/trackerkit/gui/orchestrators/session.py \
        tests/test_arena_panel_wiring.py
git commit -m "feat(arena): replace the ROI toolbar with the arena-centric panel"
```

---

### Task 10: Rebuild the grid dialog

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py`
- Test: `tests/test_arena_grid_dialog.py` (extend)

**Interfaces:**
- Consumes: `generate_grid_shapes`, `min_pitch`, `max_grid_extent` (Task 3); `ArenaCanvas.render_overlay` (Task 6), used as a renderer so the dialog preview and the main preview cannot drift.
- Produces: `ArenaGridDialog` with `spin_rows`, `spin_cols`, `spin_origin_x`, `spin_origin_y`, `spin_pitch_x`, `spin_pitch_y`, `spin_radius`, `spin_width`, `spin_height`, `slider_rotation`, `spin_rotation`, `combo_shape_type`.

- [x] **Step 1: Write the failing test**

Append to `tests/test_arena_grid_dialog.py`:

```python
def _dialog(app, width=640, height=480):
    from PySide6.QtGui import QImage

    frame = QImage(width, height, QImage.Format_RGB888)
    frame.fill(0)
    return ArenaGridDialog(reference_frame=frame, first_arena_id=0)


def test_dialog_starts_at_one_by_one(qt_app):
    dialog = _dialog(qt_app)
    assert dialog.spin_rows.value() == 1
    assert dialog.spin_cols.value() == 1


def test_spacing_controls_hidden_at_one_by_one(qt_app):
    """Spacing is meaningless with a single arena, so it must not be shown."""
    dialog = _dialog(qt_app)
    assert dialog.spin_pitch_x.isVisibleTo(dialog) is False
    assert dialog.spin_pitch_y.isVisibleTo(dialog) is False


def test_spacing_controls_appear_once_a_column_is_added(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_cols.setValue(2)
    assert dialog.spin_pitch_x.isVisibleTo(dialog) is True


def test_spacing_defaults_to_the_non_overlapping_minimum(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    dialog.spin_radius.setValue(30)
    dialog.spin_cols.setValue(2)
    assert dialog.spin_pitch_x.value() == 60
    assert dialog.spin_pitch_x.minimum() == 60


def test_rectangle_spacing_uses_width_and_height_separately(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Rectangle")
    dialog.spin_width.setValue(40)
    dialog.spin_height.setValue(20)
    dialog.spin_rows.setValue(2)
    dialog.spin_cols.setValue(2)
    assert dialog.spin_pitch_x.minimum() == 40
    assert dialog.spin_pitch_y.minimum() == 20


def test_circle_shows_radius_only(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    assert dialog.spin_radius.isVisibleTo(dialog) is True
    assert dialog.spin_width.isVisibleTo(dialog) is False


def test_rectangle_shows_width_and_height(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Rectangle")
    assert dialog.spin_width.isVisibleTo(dialog) is True
    assert dialog.spin_height.isVisibleTo(dialog) is True
    assert dialog.spin_radius.isVisibleTo(dialog) is False


def test_rotation_range_and_step(qt_app):
    dialog = _dialog(qt_app)
    assert dialog.spin_rotation.minimum() == -45.0
    assert dialog.spin_rotation.maximum() == 45.0
    assert dialog.spin_rotation.singleStep() == 0.5


def test_rotation_slider_and_spinbox_stay_in_sync(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_rotation.setValue(12.5)
    assert dialog.slider_rotation.value() == 25  # half-degree ticks
    dialog.slider_rotation.setValue(-30)
    assert dialog.spin_rotation.value() == -15.0


def test_rows_and_cols_capped_so_centres_stay_in_frame(qt_app):
    dialog = _dialog(qt_app, width=400, height=300)
    dialog.spin_origin_x.setValue(50)
    dialog.spin_origin_y.setValue(50)
    dialog.spin_cols.setValue(2)
    dialog.spin_pitch_x.setValue(100)
    dialog.spin_pitch_y.setValue(100)
    assert dialog.spin_cols.maximum() == 4
    assert dialog.spin_rows.maximum() == 3


def test_generated_grid_never_overlaps_at_default_spacing(qt_app):
    from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs

    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    dialog.spin_radius.setValue(20)
    dialog.spin_rows.setValue(3)
    dialog.spin_cols.setValue(3)
    shapes = dialog.accepted_shapes()
    assert overlapping_arena_pairs(shapes, 640, 480) == []


def test_accepted_shapes_carry_the_rotation(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_cols.setValue(2)
    dialog.spin_rotation.setValue(45.0)
    shapes = dialog.accepted_shapes()
    assert shapes[0]["params"][1] != shapes[1]["params"][1]
```

Add the shared fixture at the top of the file:

```python
@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_grid_dialog.py -v`
Expected: FAIL — `AttributeError: 'ArenaGridDialog' object has no attribute 'spin_radius'`

- [x] **Step 3: Rebuild the dialog widgets**

Replace the `__init__` form-building block in `ArenaGridDialog` with:

```python
        form_group = QGroupBox("Grid layout")
        form = QFormLayout(form_group)

        self.combo_shape_type = QComboBox()
        self.combo_shape_type.addItems(["Circle", "Rectangle"])
        form.addRow("Shape:", self.combo_shape_type)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 100000)
        self.spin_radius.setValue(20)
        self.row_radius = QLabel("Radius:")
        form.addRow(self.row_radius, self.spin_radius)

        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 100000)
        self.spin_width.setValue(40)
        self.row_width = QLabel("Width:")
        form.addRow(self.row_width, self.spin_width)

        self.spin_height = QSpinBox()
        self.spin_height.setRange(1, 100000)
        self.spin_height.setValue(40)
        self.row_height = QLabel("Height:")
        form.addRow(self.row_height, self.spin_height)

        self.spin_origin_x = QSpinBox()
        self.spin_origin_x.setRange(0, 100000)
        self.spin_origin_x.setValue(50)
        form.addRow("Top-Left Position X:", self.spin_origin_x)

        self.spin_origin_y = QSpinBox()
        self.spin_origin_y.setRange(0, 100000)
        self.spin_origin_y.setValue(50)
        form.addRow("Top-Left Position Y:", self.spin_origin_y)

        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 100)
        self.spin_rows.setValue(1)
        form.addRow("Rows:", self.spin_rows)

        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 100)
        self.spin_cols.setValue(1)
        form.addRow("Columns:", self.spin_cols)

        self.spin_pitch_x = QSpinBox()
        self.spin_pitch_x.setRange(1, 100000)
        self.row_pitch_x = QLabel("X spacing:")
        form.addRow(self.row_pitch_x, self.spin_pitch_x)

        self.spin_pitch_y = QSpinBox()
        self.spin_pitch_y.setRange(1, 100000)
        self.row_pitch_y = QLabel("Y spacing:")
        form.addRow(self.row_pitch_y, self.spin_pitch_y)

        rotation_row = QWidget()
        rotation_layout = QHBoxLayout(rotation_row)
        rotation_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_rotation = QSlider(Qt.Horizontal)
        # Half-degree ticks: the slider is integer-valued, so it counts halves.
        self.slider_rotation.setRange(-90, 90)
        self.spin_rotation = QDoubleSpinBox()
        self.spin_rotation.setRange(-45.0, 45.0)
        self.spin_rotation.setSingleStep(0.5)
        self.spin_rotation.setDecimals(1)
        self.spin_rotation.setSuffix(" deg")
        rotation_layout.addWidget(self.slider_rotation)
        rotation_layout.addWidget(self.spin_rotation)
        form.addRow("Rotation (about arena 1):", rotation_row)
```

Add the imports `QDoubleSpinBox`, `QSlider` from `PySide6.QtWidgets`.

- [x] **Step 4: Add the coupling logic**

```python
    def _shape_key(self) -> str:
        """Internal shape id for the geometry helpers."""
        return "circle" if self.combo_shape_type.currentText() == "Circle" else "polygon"

    def _size_pair(self) -> tuple[int, int]:
        """(size_x, size_y): diameter/diameter for circles, width/height for rects."""
        if self._shape_key() == "circle":
            diameter = self.spin_radius.value() * 2
            return (diameter, diameter)
        return (self.spin_width.value(), self.spin_height.value())

    def _on_shape_changed(self, *_args) -> None:
        is_circle = self._shape_key() == "circle"
        for widget in (self.row_radius, self.spin_radius):
            widget.setVisible(is_circle)
        for widget in (
            self.row_width,
            self.spin_width,
            self.row_height,
            self.spin_height,
        ):
            widget.setVisible(not is_circle)
        self._sync_pitch_floors()

    def _sync_pitch_floors(self, *_args) -> None:
        """Clamp spacing to the tightest value that cannot overlap.

        Flooring here means the generator can never emit a layout the overlap
        lock would immediately reject.
        """
        size_x, size_y = self._size_pair()
        floor_x, floor_y = min_pitch(self._shape_key(), size_x, size_y=size_y)
        for spin, floor in (
            (self.spin_pitch_x, floor_x),
            (self.spin_pitch_y, floor_y),
        ):
            was_at_floor = spin.value() <= spin.minimum()
            spin.setMinimum(int(floor))
            if was_at_floor or spin.value() < floor:
                spin.setValue(int(floor))
        self._sync_spacing_visibility()
        self._sync_extent_caps()

    def _sync_spacing_visibility(self, *_args) -> None:
        """Spacing is meaningless with one row/column, so it stays hidden."""
        multi_col = self.spin_cols.value() > 1
        multi_row = self.spin_rows.value() > 1
        self.row_pitch_x.setVisible(multi_col)
        self.spin_pitch_x.setVisible(multi_col)
        self.row_pitch_y.setVisible(multi_row)
        self.spin_pitch_y.setVisible(multi_row)

    def _sync_extent_caps(self, *_args) -> None:
        """Cap rows/cols so every arena CENTRE stays inside the frame."""
        if self._reference_frame is None:
            return
        max_rows, max_cols = max_grid_extent(
            self.spin_origin_x.value(),
            self.spin_origin_y.value(),
            self.spin_pitch_x.value(),
            self.spin_pitch_y.value(),
            self._reference_frame.width(),
            self._reference_frame.height(),
            rotation_deg=self.spin_rotation.value(),
        )
        self.spin_rows.setMaximum(max_rows)
        self.spin_cols.setMaximum(max_cols)

    def _on_slider_rotation(self, ticks: int) -> None:
        self.spin_rotation.setValue(ticks / 2.0)

    def _on_spin_rotation(self, degrees: float) -> None:
        self.slider_rotation.blockSignals(True)
        self.slider_rotation.setValue(int(round(degrees * 2)))
        self.slider_rotation.blockSignals(False)
        self._sync_extent_caps()
        self._update_preview()
```

Wire the signals after the widgets are built:

```python
        self.combo_shape_type.currentTextChanged.connect(self._on_shape_changed)
        for spin in (self.spin_radius, self.spin_width, self.spin_height):
            spin.valueChanged.connect(self._sync_pitch_floors)
        for spin in (self.spin_rows, self.spin_cols):
            spin.valueChanged.connect(self._sync_spacing_visibility)
        for spin in (self.spin_origin_x, self.spin_origin_y):
            spin.valueChanged.connect(self._sync_extent_caps)
        for spin in (self.spin_pitch_x, self.spin_pitch_y):
            spin.valueChanged.connect(self._sync_extent_caps)
        self.slider_rotation.valueChanged.connect(self._on_slider_rotation)
        self.spin_rotation.valueChanged.connect(self._on_spin_rotation)
        for widget in (
            self.spin_rows,
            self.spin_cols,
            self.spin_origin_x,
            self.spin_origin_y,
            self.spin_pitch_x,
            self.spin_pitch_y,
            self.spin_radius,
            self.spin_width,
            self.spin_height,
        ):
            widget.valueChanged.connect(self._update_preview)
        self.combo_shape_type.currentTextChanged.connect(self._update_preview)

        self._on_shape_changed()
        self._update_preview()
```

Replace `_current_shapes`:

```python
    def _current_shapes(self) -> list[dict[str, Any]]:
        """The grid shapes for the dialog's current widget values."""
        size_x, size_y = self._size_pair()
        return generate_grid_shapes(
            self.spin_rows.value(),
            self.spin_cols.value(),
            self.spin_origin_x.value(),
            self.spin_origin_y.value(),
            self.spin_pitch_x.value(),
            self.spin_pitch_y.value(),
            size_x,
            shape_type=self._shape_key(),
            first_arena_id=self._first_arena_id,
            size_y=size_y,
            rotation_deg=self.spin_rotation.value(),
        )
```

- [x] **Step 5: Route the preview through the shared renderer**

Replace the painting block in `_update_preview` (the `painter.setPen(QPen(Qt.cyan, 2))` loop) with a call into an `ArenaCanvas` used purely as a renderer, so the dialog and the main preview cannot drift:

```python
        from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas

        renderer = ArenaCanvas()
        renderer.set_frame(image)
        renderer.set_shapes(shapes)
        target_w = self.preview_label.width() or 320
        renderer.set_zoom(min(1.0, target_w / max(1, image.width())))
        pixmap = QPixmap(renderer.width(), renderer.height())
        pixmap.fill(Qt.black)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, renderer._scaled)
        renderer.render_overlay(painter)
        painter.end()
        self.preview_label.setPixmap(pixmap)
```

- [x] **Step 6: Run the tests**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_grid_dialog.py -v`
Expected: PASS — the twelve new tests plus every pre-existing case in the file

- [x] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py tests/test_arena_grid_dialog.py
git commit -m "feat(arena): grid builder with rotation, shape sizing, pitch floors and shared preview"
```

---

### Task 11: Block tracking on any arena overlap

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:1032` (`start_tracking`)
- Test: `tests/test_arena_tracking_gate.py`

**Interfaces:**
- Consumes: `ArenaPanel.can_track()` (Task 7).
- Produces: `TrackingOrchestrator._validate_arena_overlaps(mode_label) -> bool`, following the existing `_validate_identity_requirements` / `_validate_yolo_model_requirements` pattern at lines 1584 and 1608.

- [x] **Step 1: Write the failing test**

```python
"""Tracking refuses to start while any two arenas overlap."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _circle(cx, cy, r, arena_id):
    return {
        "type": "circle",
        "params": (cx, cy, r),
        "mode": "include",
        "arena_id": arena_id,
    }


@pytest.fixture(scope="module")
def window():
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    QApplication.instance() or QApplication([])
    return MainWindow()


def test_gate_passes_with_no_overlap(window, monkeypatch):
    window.roi_shapes = [_circle(50, 50, 20, 0), _circle(300, 300, 20, 1)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None
    )
    assert window._tracking_orch._validate_arena_overlaps("Forward") is True


def test_gate_blocks_on_overlap(window, monkeypatch):
    window.roi_shapes = [_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None
    )
    assert window._tracking_orch._validate_arena_overlaps("Forward") is False


def test_gate_passes_for_a_single_arena(window, monkeypatch):
    """One arena cannot overlap itself; plain-ROI users must not be gated."""
    window.roi_shapes = [_circle(100, 100, 50, 0), _circle(130, 100, 50, 0)]
    window.arena_panel.set_frame_size(400, 400)
    window.arena_panel.set_shapes(window.roi_shapes)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None
    )
    assert window._tracking_orch._validate_arena_overlaps("Forward") is True


def test_gate_passes_with_no_arenas_at_all(window, monkeypatch):
    """No ROI means the whole video is used -- nothing to conflict."""
    window.roi_shapes = []
    window.arena_panel.set_shapes([])
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None
    )
    assert window._tracking_orch._validate_arena_overlaps("Forward") is True
```

- [x] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_tracking_gate.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_validate_arena_overlaps'`

- [x] **Step 3: Write the implementation**

Add to `tracking.py`, beside the other validators:

```python
    def _validate_arena_overlaps(self, mode_label: str) -> bool:
        """Refuse to start while any two arenas share a pixel.

        Overlapping arenas are not merely a UI blemish:
        ``engine_params.build_arena_labels`` resolves them by last-writer-wins
        in shape draw order, silently, so an animal in the shared region is
        assigned to whichever arena happened to rasterize last.
        """
        allowed, reason = self._mw.arena_panel.can_track()
        if allowed:
            return True
        QMessageBox.warning(self._mw, f"{mode_label}: Overlapping Arenas", reason)
        return False
```

Call it in `start_tracking` immediately after the existing validators run, returning early when it fails.

- [x] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/test_arena_tracking_gate.py -v`
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_arena_tracking_gate.py
git commit -m "feat(arena): refuse to start tracking while arenas overlap"
```

---

### Task 12: Docs, quality gates, and the equivalence proof

**Files:**
- Modify: `docs/user-guide/` — the TrackerKit ROI/arena page
- Test: the full arena suite plus the MPS equivalence matrix

- [x] **Step 1: Find and update the user-facing docs**

```bash
grep -rln "ROI\|arena" docs/user-guide/ | head
```

Update the page describing ROI setup to the new flow: the empty state, the two add buttons, arena navigation, the `+`/`-` zone buttons replacing the Include/Exclude dropdown, the overlap warning and what it blocks, the grid builder's controls, and that drawing now works at any zoom. State explicitly that one arena is exactly a plain ROI and produces no `arena_id` column.

- [x] **Step 2: Run formatting and lint**

```bash
make commit-prep
make lint-moderate
```
Expected: clean. Fix anything reported.

- [x] **Step 3: Run the full arena suite**

Run: `conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen python -m pytest tests/ -k arena -v`
Expected: PASS. Baseline on `main` at `a2c9838f` was 166 passing; the count must be higher and nothing previously passing may fail.

- [x] **Step 4: Run the batched wider suite**

Do NOT run `pytest tests/` in one go — a classkit modal-dialog hang and a SIGABRT make it never finish (memory `project_main_suite_blockers`). Run per-file batches and compare failures against `main`:

```bash
git stash list  # ensure a clean tree
for f in tests/test_*.py; do
  conda run -n hydra-mps env QT_QPA_PLATFORM=offscreen \
    python -m pytest "$f" -q --timeout=300 2>&1 | tail -2
done
```
Expected: the same failure set as `main`. A delta gate, not an absolute one — `main` carries pre-existing failures.

- [x] **Step 5: Kill stale processes, then run the equivalence matrix**

Per `CLAUDE.md`, kill dead/stale sleap/hydra processes first, and never touch anything else:

```bash
pgrep -fl "sleap|hydra" || true
```

Then, with conda active (a bare shell yields EMPTY CSVs that falsely compare EQUIVALENT):

```bash
conda activate hydra-mps
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_arena_ux RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

Verify row counts are non-zero before trusting any verdict, and count verdicts with `grep -c "^VERDICT"` — `grep -c "EQUIVALENT"` over-counts because that string also appears on PERFORMANCE lines.

Expected: every clip EQUIVALENT at its determinism floor. This work is display and shape authoring only, and `roi_shapes` is unchanged in format, so any difference is a real regression.

- [x] **Step 6: Clean up the equivalence worktree**

```bash
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
```

- [x] **Step 7: Commit**

```bash
git add docs/
git commit -m "docs(arena): document the arena-centric ROI workflow and grid builder"
```

---

## Completion

When every task is checked off and the equivalence gate passes, merge to local `main` with `--no-ff`, then `git mv` this plan to `docs/superpowers/plans/done/` and the spec to `docs/superpowers/specs/done/` in the same commit, updating the spec's `**Status:**` header to `Shipped — merged to main (<sha>)`. Then remove the worktree and delete the merged branch.
