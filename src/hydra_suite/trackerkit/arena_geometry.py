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

    Ties resolve by draw order -- the LAST include shape containing the
    point wins, matching ``build_arena_labels``' last-writer-wins
    rasterization. Excludes are scoped per-arena, exactly like
    ``build_arena_labels``: only an exclude shape whose OWN ``arena_id``
    matches the point's candidate arena can null it out -- a different
    arena's exclude zone that happens to geometrically overlap this point
    must not affect it, even if their raw shapes overlap.
    """
    if not shapes:
        return None
    found: int | None = None
    for shape in shapes:
        if shape.get("mode", "include") == "include" and point_in_shape(shape, x, y):
            found = int(shape.get("arena_id", 0))
    if found is None:
        return None
    for shape in shapes:
        if (
            shape.get("mode", "include") == "exclude"
            and int(shape.get("arena_id", 0)) == found
            and point_in_shape(shape, x, y)
        ):
            return None
    return found


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


def _paint_shape(
    canvas: np.ndarray, shape: dict[str, Any], x0: int, y0: int, value: int
) -> None:
    """Rasterize one shape onto *canvas* (already offset-cropped) with *value*."""
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
        offset = np.asarray([x0, y0], dtype=np.int32)
        points = np.asarray(shape["params"], dtype=np.int32) - offset
        cv2.fillPoly(canvas, [points], value)


def _rasterize(
    shapes: list[dict[str, Any]],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Boolean mask of one arena, cropped to the (x0, y0, width, height) box.

    Paints ALL include shapes first, then ALL exclude shapes second,
    regardless of the input list's order -- mirroring
    ``engine_params.build_arena_labels``'s own two-pass structure exactly,
    so this UI-facing geometry check can never disagree with the real
    engine rasterization about which pixels an arena actually owns. A
    single draw-order pass (paint whichever mode each shape happens to be,
    in list order) is WRONG: an exclude shape that appears before its
    arena's include shape in the list would paint 0 onto an already-zero
    canvas (no effect), then get overwritten entirely by the include's 255
    fill -- silently losing the hole. See Fix Wave 21 Finding B.
    """
    canvas = np.zeros((height, width), np.uint8)
    for shape in shapes:
        if shape.get("mode", "include") == "include":
            _paint_shape(canvas, shape, x0, y0, 255)
    for shape in shapes:
        if shape.get("mode", "include") != "include":
            _paint_shape(canvas, shape, x0, y0, 0)
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


def non_contiguous_arena_ids(
    shapes: list[dict[str, Any]] | None,
    width: int,
    height: int,
) -> list[int]:
    """Every arena id whose own net region (includes minus excludes) forms
    more than one disconnected piece.

    An arena made of two circles that don't touch is exactly the failure
    mode this catches: nothing else in the pipeline treats "one arena" as
    "one connected region" today, so a user can silently create two
    physically separate zones that share an arena id -- which the engine
    then tracks as ONE arena with no gating between the two pieces at all.

    Rasterized per-arena, cropped to that arena's own bounding box (same
    perf rationale as ``overlapping_arena_pairs``: full-frame rasterization
    per arena is not viable at 4512x4512 with many arenas).
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

    disconnected: list[int] = []
    for aid in arena_ids:
        group = by_arena[aid]
        box = _arena_bbox(group)
        x0 = int(math.floor(max(box[0], 0)))
        y0 = int(math.floor(max(box[1], 0)))
        x1 = int(math.ceil(min(box[2], width - 1)))
        y1 = int(math.ceil(min(box[3], height - 1)))
        crop_w, crop_h = x1 - x0 + 1, y1 - y0 + 1
        if crop_w <= 0 or crop_h <= 0:
            continue
        mask = _rasterize(group, x0, y0, crop_w, crop_h)
        n_components, _ = cv2.connectedComponents(mask.astype(np.uint8))
        # connectedComponents always counts background (label 0) as one
        # component, so >1 include region means n_components > 2.
        if n_components > 2:
            disconnected.append(aid)
    return disconnected


def min_pitch(shape_type: str, size: int, size_y: int | None = None) -> tuple[int, int]:
    """Tightest centre-to-centre pitch that cannot produce overlap.

    Circles of radius r avoid overlap only when spacing is at least 2r, i.e.
    the full diameter -- ``size`` already IS the diameter, and exact
    tangency is safe here because ``overlapping_arena_pairs``' analytic
    circle-circle fast path treats distance ``>= r_a + r_b`` as non-
    overlapping without ever rasterizing. Rectangles have no such fast
    path -- they always go through ``_rasterize``'s boundary-inclusive
    ``cv2.fillPoly``, where an EXACT width/height pitch still shares one
    column/row of pixels at the touching edge. Add a 1-pixel margin so a
    rectangle grid at its own floor/default spacing can never be flagged
    as overlapping by the exact rasterizer that checks it.
    """
    height = int(size if size_y is None else size_y)
    if shape_type == "circle":
        return (int(size), int(size))
    return (int(size) + 1, height + 1)


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
                    for rx, ry in (_rotate(dx, dy, cos_t, sin_t) for dx, dy in corners)
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
