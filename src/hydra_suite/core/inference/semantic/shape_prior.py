"""Re-export shim: the size-and-shape prior lives at ``core/inference/shape_prior``.

Moved (2026-09-06) so the direct-calibration path can share it without
importing under ``core/inference/semantic/`` -- see
``docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md`` Task 1.
Every name below is the SAME object as in ``core.inference.shape_prior``
(see ``tests/test_shared_scoring_primitives_identity.py``); this module must
never define a second copy.
"""

from __future__ import annotations

from hydra_suite.core.inference.shape_prior import (
    HIGH_MULTIPLIER,
    LOW_MULTIPLIER,
    MIN_MATCH_QUALITY,
    SHAPE_TERM_FLOOR,
    AreaBand,
    aspect_ratio,
    fit_area_band,
    in_band,
    match_quality,
    polygon_area,
)

__all__ = [
    "AreaBand",
    "HIGH_MULTIPLIER",
    "LOW_MULTIPLIER",
    "MIN_MATCH_QUALITY",
    "SHAPE_TERM_FLOOR",
    "aspect_ratio",
    "fit_area_band",
    "in_band",
    "match_quality",
    "polygon_area",
]
