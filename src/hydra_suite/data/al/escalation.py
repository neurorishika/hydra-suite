"""Geometry escalation for active-learning labels.

The single authority for converting one detection's geometry between levels.
Downward derivation only: polygon -> minAreaRect -> obb -> aabb. Upward
derivation would invent information and is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from hydra_suite.utils.geometry_levels import GeometryLevel


@dataclass
class LabelRecord:
    """One detection's exportable geometry, in frame pixel space."""

    class_id: int
    confidence: float
    points: np.ndarray  # (P, 2) float32, pixel space
    level: GeometryLevel


def achievable_levels(native_level: GeometryLevel) -> list[GeometryLevel]:
    """Levels derivable from `native_level`, highest first."""
    return [lvl for lvl in sorted(GeometryLevel, reverse=True) if lvl <= native_level]


def records_from_obb_result(
    obb,
    native_level: GeometryLevel,
    keep: Sequence[int] | None = None,
) -> list[LabelRecord]:
    """Build LabelRecords from an OBBResult at its native geometry level.

    `keep` optionally restricts output to those detection indices (used by the
    strict-label filter). Passing an empty sequence yields no records.
    """
    indices = range(obb.num_detections) if keep is None else [int(i) for i in keep]

    if native_level is GeometryLevel.POLYGON and obb.polygons is None:
        raise ValueError(
            "native polygons requested but OBBResult.polygons is None; the "
            "detection stage was not run with emit_native_geometry=True"
        )

    class_ids = obb.class_ids_or_zeros
    records: list[LabelRecord] = []
    for i in indices:
        if native_level is GeometryLevel.POLYGON:
            pts = np.asarray(obb.polygons[i], dtype=np.float32).reshape(-1, 2)
        else:
            pts = np.asarray(obb.corners[i], dtype=np.float32).reshape(-1, 2)
        records.append(
            LabelRecord(
                class_id=int(class_ids[i]),
                confidence=float(obb.confidences[i]),
                points=pts.copy(),
                level=native_level,
            )
        )
    return records


def _to_obb(points: np.ndarray) -> np.ndarray:
    """Minimum-area rotated rectangle enclosing `points`, as (4, 2) float32."""
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    return cv2.boxPoints(rect).astype(np.float32)


def _to_aabb(points: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding quad of `points`, as (4, 2) float32."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def derive_down(
    records: Sequence[LabelRecord],
    target: GeometryLevel,
) -> list[LabelRecord]:
    """Derive `records` down to `target`. Refuses upward derivation."""
    out: list[LabelRecord] = []
    for rec in records:
        if target > rec.level:
            raise ValueError(
                f"cannot derive upward from {rec.level.label} to {target.label}: "
                "upward derivation requires information the model did not produce"
            )
        if target is rec.level:
            out.append(rec)
            continue
        if target is GeometryLevel.OBB:
            pts = _to_obb(rec.points)
        else:
            pts = _to_aabb(rec.points)
        out.append(
            LabelRecord(
                class_id=rec.class_id,
                confidence=rec.confidence,
                points=pts,
                level=target,
            )
        )
    return out
