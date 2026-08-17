"""Geometry-level vocabulary for polygon-first labels.

A label line stays ``class_id`` followed by a normalized point list. The
information content of a source is captured by a totally-ordered level:

    aabb  <  obb  <  polygon

Downward derivation (polygon -> minAreaRect -> obb -> aabb) is lossless to the
target; upward derivation needs new information.

This lives in ``utils`` (the bottom layer) so both ``data.al`` and ``training``
can import it without a lateral dependency.
"""

from __future__ import annotations

from enum import IntEnum


class GeometryLevel(IntEnum):
    """Information content of a source's geometry, totally ordered."""

    AABB = 0
    OBB = 1
    POLYGON = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_str(value: str) -> "GeometryLevel":
        key = str(value).strip().lower()
        for level in GeometryLevel:
            if level.name.lower() == key:
                return level
        raise ValueError(f"Unknown geometry level: {value!r}")


def classify_label_line(field_count: int) -> str:
    """Classify one label line by its whitespace field count.

    Returns:
        - "aabb"       for 5 fields (class + cx cy w h),
        - "four_point" for 9 fields (class + 8 coords: OBB or quad polygon),
        - "polygon"    for an odd field count >= 7 encoding 3 or >=5 points,
        - "invalid"    otherwise.
    """
    if field_count == 5:
        return "aabb"
    if field_count == 9:
        return "four_point"
    coords = field_count - 1
    if field_count >= 7 and coords % 2 == 0:
        points = coords // 2
        if points >= 3 and points != 4:
            return "polygon"
    return "invalid"
