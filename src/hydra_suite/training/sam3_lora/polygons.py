"""Dependency-free COCO polygon normalization shared by loading and admission."""

from __future__ import annotations

import math
from numbers import Real


def _polygon_points(value: object) -> tuple[tuple[float, float], ...] | None:
    """Normalize one flat or ``N x 2`` JSON polygon without reshaping it."""

    if not isinstance(value, (list, tuple)):
        return None
    if len(value) >= 3 and all(
        isinstance(point, (list, tuple)) and len(point) == 2 for point in value
    ):
        coordinates = [coordinate for point in value for coordinate in point]
    elif len(value) >= 6 and len(value) % 2 == 0:
        coordinates = list(value)
    else:
        return None
    if not all(
        isinstance(coordinate, Real)
        and not isinstance(coordinate, bool)
        and math.isfinite(float(coordinate))
        for coordinate in coordinates
    ):
        return None
    return tuple(
        (float(coordinates[index]), float(coordinates[index + 1]))
        for index in range(0, len(coordinates), 2)
    )


def validated_segmentation_polygons(
    segmentation: object,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return valid flat or nested ``N x 2`` polygons from segmentation."""

    direct = _polygon_points(segmentation)
    if direct is not None:
        return (direct,)
    if not isinstance(segmentation, (list, tuple)):
        return ()
    return tuple(
        polygon
        for candidate in segmentation
        if (polygon := _polygon_points(candidate)) is not None
    )
