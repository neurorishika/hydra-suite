"""Geometry contracts for DetectKit inference results.

Inference caches contain only canvas-ready contours.  Their geometry level is
therefore carried by the inference kind that produced them, rather than by
each cached detection.
"""

from __future__ import annotations

from hydra_suite.utils.geometry_levels import GeometryLevel

_NATIVE_LEVEL_BY_INFERENCE_KIND = {
    "obb_direct": GeometryLevel.OBB,
    "sequential": GeometryLevel.OBB,
    "detect_direct": GeometryLevel.AABB,
    "segment_direct": GeometryLevel.POLYGON,
    "sequential_segment": GeometryLevel.POLYGON,
}


def native_prediction_level(inference_kind: str) -> GeometryLevel:
    """Return the highest geometry level emitted by an inference kind.

    Unknown and legacy models retain DetectKit's historical OBB fallback.
    """
    return _NATIVE_LEVEL_BY_INFERENCE_KIND.get(str(inference_kind), GeometryLevel.OBB)
