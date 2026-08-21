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
