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
