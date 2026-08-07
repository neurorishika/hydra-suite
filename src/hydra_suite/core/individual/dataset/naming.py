"""Shared filename helpers for MAT identity dataset image exports."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry

logger = logging.getLogger(__name__)

_DETECTION_IMAGE_RE = re.compile(
    r"^did(?P<detection_id>\d+)\.(?P<ext>png|jpe?g)$",
    re.IGNORECASE,
)
_INTERPOLATED_IMAGE_RE = re.compile(
    r"^interp_f(?P<frame_id>\d+)_traj(?P<trajectory_id>\d+)"
    r"_seg(?P<interp_from_start>\d+)-(?P<interp_from_end>\d+)"
    r"_p(?P<interp_index>\d+)of(?P<interp_total>\d+)\.(?P<ext>png|jpe?g)$",
    re.IGNORECASE,
)


def normalize_identity_image_extension(extension: str | None) -> str:
    """Normalize an export image extension to the canonical on-disk suffix."""
    value = str(extension or "png").strip().lower().lstrip(".")
    if value == "jpeg":
        return "jpg"
    if value in {"jpg", "png"}:
        return value
    return "png"


def build_detection_image_filename(
    detection_id: int,
    extension: str | None = "png",
) -> str:
    """Build the canonical flat filename for a detected crop image."""
    ext = normalize_identity_image_extension(extension)
    return f"did{int(detection_id)}.{ext}"


def build_interpolated_image_filename(
    frame_id: int,
    trajectory_id: int,
    interp_from: tuple[int, int],
    interp_index: int,
    interp_total: int,
    extension: str | None = "png",
) -> str:
    """Build the canonical flat filename for an interpolated crop image."""
    ext = normalize_identity_image_extension(extension)
    return (
        f"interp_f{int(frame_id):06d}_traj{int(trajectory_id):04d}"
        f"_seg{int(interp_from[0]):06d}-{int(interp_from[1]):06d}"
        f"_p{int(interp_index):03d}of{int(interp_total):03d}.{ext}"
    )


def synthetic_interpolated_det_id(
    frame_id: int,
    trajectory_id: int,
    interp_index: int,
) -> int:
    """Build a stable negative integer ID for interpolated rows."""
    return -(
        int(frame_id) * 1_000_000_000
        + int(trajectory_id) * 1_000
        + max(0, int(interp_index))
        + 1
    )


def parse_identity_image_filename(filename: str) -> dict[str, Any] | None:
    """Parse a flat MAT identity dataset filename into structured metadata."""
    name = str(filename or "").strip()
    match = _DETECTION_IMAGE_RE.match(name)
    if match:
        detection_id = int(match.group("detection_id"))
        return {
            "filename": name,
            "source_type": "detected",
            "interpolated": False,
            "detection_id": detection_id,
            "det_id": detection_id,
            "frame_idx": detection_id // 10000,
            "det_idx": detection_id % 10000,
        }

    match = _INTERPOLATED_IMAGE_RE.match(name)
    if match:
        frame_id = int(match.group("frame_id"))
        trajectory_id = int(match.group("trajectory_id"))
        interp_from_start = int(match.group("interp_from_start"))
        interp_from_end = int(match.group("interp_from_end"))
        interp_index = int(match.group("interp_index"))
        interp_total = int(match.group("interp_total"))
        return {
            "filename": name,
            "source_type": "interpolated",
            "interpolated": True,
            "detection_id": None,
            "det_id": synthetic_interpolated_det_id(
                frame_id,
                trajectory_id,
                interp_index,
            ),
            "frame_idx": frame_id,
            "det_idx": -max(1, interp_index),
            "trajectory_id": trajectory_id,
            "interp_from_start": interp_from_start,
            "interp_from_end": interp_from_end,
            "interp_index": interp_index,
            "interp_total": interp_total,
        }

    return None


def read_canonical_provenance(dataset_dir: Path) -> CanonicalGeometry | None:
    """Read the Layer 1 ``CanonicalGeometry`` a dataset was exported with.

    Reads ``parameters.canonical`` from ``<dataset_dir>/metadata.json`` (the
    same metadata.json the crop-dataset generator and the oriented-video
    exporter both write into -- no separate sidecar file). Returns ``None``
    -- never a default -- whenever the file, the ``parameters`` block, or the
    ``canonical`` sub-block is missing, so a dataset with unknown provenance
    (e.g. exported before this migration) is visibly unknown rather than
    silently assumed to match the current project geometry.
    """
    metadata_path = Path(dataset_dir) / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "Failed to parse %s while reading canonical provenance.",
            metadata_path,
            exc_info=True,
        )
        return None

    canonical = (data.get("parameters") or {}).get("canonical")
    if not canonical:
        return None
    try:
        return CanonicalGeometry.from_dict(canonical)
    except Exception:
        logger.warning(
            "Malformed parameters.canonical in %s while reading canonical provenance.",
            metadata_path,
            exc_info=True,
        )
        return None
