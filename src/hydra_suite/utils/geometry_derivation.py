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
