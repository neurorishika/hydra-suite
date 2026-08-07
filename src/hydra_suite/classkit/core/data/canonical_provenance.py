"""Recover the CanonicalGeometry a ClassKit training run's crops were built under.

ClassKit ingests already-cropped images (``core/data/source_import.py`` records
each image's ``source_root`` -- the folder it was imported from -- in its
per-image metadata). When that folder still carries a ``metadata.json`` with
Layer 1 canonical-crop provenance (written by the identity dataset generator /
oriented-video exporter -- see ``core.identity.dataset.naming.read_canonical_provenance``),
this module recovers the geometry so a published classifier can be stamped
with what it actually trained on, instead of every model staying silently
unstamped.

Deliberately conservative: this is app-layer, best-effort recovery of
provenance ClassKit did not itself capture. It returns ``None`` -- never a
guess -- whenever any image lacks a recorded source root, any source root
lacks (or fails to parse) provenance, or source roots disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.identity.dataset.naming import read_canonical_provenance


def canonical_geometry_for_training_images(
    image_paths: Iterable[Path | str],
    metadata_by_path: Mapping[str, Mapping[str, Any]],
) -> CanonicalGeometry | None:
    """Return the single geometry every training image's source agrees on.

    Groups ``image_paths`` by their recorded ``source_root`` (from
    ``metadata_by_path``, e.g. ``ClassKitDB.get_image_metadata_by_path()``),
    reads each unique root's provenance once, and returns that geometry only
    if exactly one distinct geometry is found across ALL images. Returns
    ``None`` on any missing source root, any unreadable/unstamped source
    dataset, or disagreement between sources -- an ambiguous or unknown
    provenance must stay visibly unstamped rather than silently taking on the
    wrong geometry.
    """
    source_roots: set[str] = set()
    for path in image_paths:
        meta = metadata_by_path.get(str(path)) or {}
        source_root = str(meta.get("source_root") or "").strip()
        if not source_root:
            return None
        source_roots.add(source_root)

    if not source_roots:
        return None

    geometries: set[CanonicalGeometry] = set()
    for root in source_roots:
        geometry = read_canonical_provenance(Path(root))
        if geometry is None:
            return None
        geometries.add(geometry)

    if len(geometries) != 1:
        return None
    return next(iter(geometries))
