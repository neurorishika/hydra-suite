"""Shared Layer 1 canonical crop geometry derivation for TrackerKit.

Every TrackerKit worker/orchestrator that needs the project-wide
:class:`~hydra_suite.core.canonicalization.geometry.CanonicalGeometry` (one
fixed canvas shared by every crop-consuming stage) derives it the same way:
``REFERENCE_BODY_SIZE * RESIZE_FACTOR`` for the reference body extent, and
``ADVANCED_CONFIG`` (lowercase keys) for the species aspect ratio and crop
margin -- mirroring ``core.inference.config``'s derivation exactly. This
module is the one place that expression lives; callers must not re-derive it.
"""

from __future__ import annotations

from typing import Any, Mapping

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry


def canonical_geometry_from_params(params: Mapping[str, Any]) -> CanonicalGeometry:
    """Build the project-wide Layer 1 geometry from a tracking parameters dict.

    Mirrors ``core.inference.config``'s ``CanonicalGeometry.from_reference``
    call: ``REFERENCE_BODY_SIZE * RESIZE_FACTOR`` for the reference body
    extent, and ``ADVANCED_CONFIG.reference_aspect_ratio`` /
    ``ADVANCED_CONFIG.canonical_margin`` for the species aspect ratio and
    crop margin -- the same knobs every other canonical-crop consumer reads.
    """
    adv = params.get("ADVANCED_CONFIG", {}) or {}
    return CanonicalGeometry.from_reference(
        reference_body_px=float(params.get("REFERENCE_BODY_SIZE", 20.0))
        * float(params.get("RESIZE_FACTOR", 1.0)),
        aspect_ratio=float(adv.get("reference_aspect_ratio", 2.0)),
        margin=float(adv.get("canonical_margin", 1.3)),
    )
