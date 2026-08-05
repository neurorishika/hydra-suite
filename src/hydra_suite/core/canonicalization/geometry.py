"""Canonical crop geometry: one rigid transform, one fixed canvas.

The canvas is a property of the project, not of the detection.  Its long edge
holds ``margin`` times the reference animal's major axis; the OBB supplies only
a centre and an angle.  Both axes therefore map at scale 1 -- the transform is a
rotation and a translation, nothing more -- so no animal is stretched, and body
size survives into the crop as signal instead of being normalised away.

``REFERENCE_BODY_SIZE`` is the geometric mean ``sqrt(major * minor)``; with
``ar = major / minor`` the major axis is ``body_px * sqrt(ar)``.  That recovers
the extent this module needs without redefining a knob that Kalman, Hungarian,
background subtraction and the detection cache key all depend on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_MIN_CANVAS_EDGE = 8


def _even(value: float) -> int:
    return max(_MIN_CANVAS_EDGE, int(math.ceil(float(value) / 2.0) * 2))


@dataclass(frozen=True)
class CanonicalGeometry:
    """Fixed canonical crop geometry for one project/session."""

    canvas_wh: tuple[int, int]
    margin: float
    aspect_ratio: float

    @classmethod
    def from_reference(
        cls,
        reference_body_px: float,
        aspect_ratio: float,
        margin: float,
    ) -> "CanonicalGeometry":
        body = max(1e-3, float(reference_body_px))
        ar = max(1.0, float(aspect_ratio))
        m = max(1.0, float(margin))
        canvas_w = _even(body * math.sqrt(ar) * m)
        canvas_h = _even(canvas_w / ar)
        return cls(canvas_wh=(canvas_w, canvas_h), margin=m, aspect_ratio=ar)

    @property
    def canvas_w(self) -> int:
        return int(self.canvas_wh[0])

    @property
    def canvas_h(self) -> int:
        return int(self.canvas_wh[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas_wh": [self.canvas_w, self.canvas_h],
            "margin": float(self.margin),
            "aspect_ratio": float(self.aspect_ratio),
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CanonicalGeometry":
        w, h = d["canvas_wh"]
        return cls(
            canvas_wh=(int(w), int(h)),
            margin=float(d["margin"]),
            aspect_ratio=float(d["aspect_ratio"]),
        )


def _axes(corners: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))
    if e01 < 1e-3 or e12 < 1e-3:
        raise ValueError("Degenerate OBB (zero-length edge)")
    major_vec = c[1] - c[0] if e01 >= e12 else c[2] - c[1]
    angle = float(math.atan2(float(major_vec[1]), float(major_vec[0])))
    return c, max(e01, e12), min(e01, e12), angle, 0.0


def overflow_ratio(corners: np.ndarray, geometry: CanonicalGeometry) -> float:
    """How far the padded OBB exceeds the canvas; <= 1.0 means it fits."""
    _, major, minor, _, _ = _axes(corners)
    return max(
        major * geometry.margin / geometry.canvas_w,
        minor * geometry.margin / geometry.canvas_h,
    )


def canonical_affine(
    corners: np.ndarray,
    geometry: CanonicalGeometry,
) -> tuple[np.ndarray, float, bool]:
    """Return ``(M_align, major_axis_theta, clipped)`` for one OBB.

    ``M_align`` is a 2x3 affine mapping frame pixels to canvas pixels: a
    rotation that puts the major axis horizontal, then a translation that puts
    the centroid at the canvas centre.  Its linear part is orthonormal by
    construction -- there is no scale term.
    """
    c, major, minor, angle, _ = _axes(corners)
    cx = float(np.mean(c[:, 0]))
    cy = float(np.mean(c[:, 1]))

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    half_w = geometry.canvas_w / 2.0
    half_h = geometry.canvas_h / 2.0

    m_align = np.array(
        [
            [cos_a, -sin_a, half_w - (cos_a * cx - sin_a * cy)],
            [sin_a, cos_a, half_h - (sin_a * cx + cos_a * cy)],
        ],
        dtype=np.float64,
    )

    clipped = (
        major * geometry.margin > geometry.canvas_w
        or minor * geometry.margin > geometry.canvas_h
    )
    return m_align, angle, bool(clipped)


def invert_affine(m_align: np.ndarray) -> np.ndarray:
    """Canvas -> frame, for back-projecting predictions."""
    return cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
