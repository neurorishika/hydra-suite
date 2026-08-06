"""Shared Layer 1 canonical crop geometry derivation for TrackerKit.

Every TrackerKit worker/orchestrator that needs the project-wide
:class:`~hydra_suite.core.canonicalization.geometry.CanonicalGeometry` (one
fixed canvas shared by every crop-consuming stage) derives it the same way:
``REFERENCE_BODY_SIZE * RESIZE_FACTOR`` for the reference body extent, and
``ADVANCED_CONFIG`` (lowercase keys) for the species aspect ratio and crop
margin -- mirroring ``core.inference.config``'s derivation exactly.

The expression itself now lives in
:func:`hydra_suite.core.canonicalization.geometry.canonical_geometry_from_params`
so that Qt-free ``core`` consumers can share it without ``core`` importing
from an app layer.  This module is kept as the TrackerKit-facing name; it must
stay a pure re-export -- never re-derive the geometry here.
"""

from __future__ import annotations

from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params

__all__ = ["canonical_geometry_from_params"]
