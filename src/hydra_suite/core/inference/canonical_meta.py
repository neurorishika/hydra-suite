"""Read a model's .canonical_meta.json sidecar: the crop geometry convention.

Pure (stdlib only). The sidecar is written by training/model_publish.py at
publish time so a checkpoint on disk carries a record of which
``CanonicalGeometry`` convention it was trained under. Mirrors the
``<artifact>.slice_meta.json`` / ``<artifact>.runtime_meta.json`` append
convention (``model.pt`` -> ``model.pt.canonical_meta.json``), NOT the
``.v2meta.json`` replace convention -- see core/inference/slice_meta.py.

Every pre-existing checkpoint predates this stamp and must keep loading:
``read_canonical_meta`` returns ``None`` for an unstamped model, never a
default or inferred geometry. ``warn_on_geometry_mismatch`` likewise returns
``None`` when there is nothing to compare, and otherwise returns a
human-readable diff -- it warns, it never raises.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry

logger = logging.getLogger(__name__)


def read_canonical_meta(model_path) -> "CanonicalGeometry | None":
    """Return the ``CanonicalGeometry`` stamped in <model_path>.canonical_meta.json.

    Returns ``None`` on a missing, unreadable, or malformed sidecar -- in
    particular for every pre-existing checkpoint, which was published before
    this stamp existed and carries no such file. Never infers or defaults a
    geometry: an unknown provenance must be visibly unknown.
    """
    try:
        sidecar = Path(model_path).with_suffix(
            Path(model_path).suffix + ".canonical_meta.json"
        )
        if not sidecar.exists():
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return CanonicalGeometry.from_dict(data)
    except Exception:
        logger.warning(
            "Failed to read canonical geometry sidecar for %s.",
            model_path,
            exc_info=True,
        )
        return None


def warn_on_geometry_mismatch(
    model_path,
    session_geometry: "CanonicalGeometry",
) -> "str | None":
    """Compare a model's stamped geometry against the session's, if any.

    Returns ``None`` when the model is unstamped (nothing to compare -- a
    pre-existing checkpoint must keep loading silently) or when the stamped
    geometry matches. Otherwise returns a human-readable message naming the
    fields that differ. This is a warning, not a gate: it never raises, so a
    mismatch never bricks a load.
    """
    trained = read_canonical_meta(model_path)
    if trained is None:
        return None

    diffs: list[str] = []
    if trained.canvas_wh != session_geometry.canvas_wh:
        diffs.append(
            f"canvas_wh: trained={trained.canvas_wh} session={session_geometry.canvas_wh}"
        )
    if trained.margin != session_geometry.margin:
        diffs.append(
            f"margin: trained={trained.margin} session={session_geometry.margin}"
        )
    if trained.aspect_ratio != session_geometry.aspect_ratio:
        diffs.append(
            "aspect_ratio: "
            f"trained={trained.aspect_ratio} session={session_geometry.aspect_ratio}"
        )
    if not diffs:
        return None

    return (
        f"Model {Path(model_path).name} was trained under a different canonical "
        f"crop geometry than this session's ({'; '.join(diffs)}). Predictions "
        "may be degraded until the model is retrained under the current "
        "geometry."
    )
