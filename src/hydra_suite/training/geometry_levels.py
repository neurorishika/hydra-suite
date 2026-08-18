"""Geometry-level model for DetectKit's polygon-first labels.

A label line stays ``class_id`` followed by a normalized point list. The
information content of a source is captured by a totally-ordered level:

    aabb  <  obb  <  polygon

Downward derivation (polygon -> minAreaRect -> obb -> aabb) is lossless to the
target; upward derivation needs new information and is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hydra_suite.utils.geometry_levels import (  # noqa: F401  (re-exported)
    GeometryLevel,
    classify_label_line,
)


@dataclass(frozen=True)
class SourceLevelScan:
    """Verdict of scanning a source's label directory for its geometry level."""

    resolved_level: GeometryLevel
    is_homogeneous: bool
    conflict_files: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    reason: str = ""


def _classify_file(path: Path) -> str:
    """Return the strongest evidence in a single label file.

    One of: "polygon", "four_point", "aabb", "empty", "invalid".
    A file mixing an aabb line with any oriented/contour (polygon or
    four-point) line is "invalid" (internally inconsistent); this check runs
    first so aabb evidence is never silently dropped. Otherwise any
    polygon-evidence line makes the file "polygon".
    """
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        kind = classify_label_line(len(line.split()))
        if kind == "invalid":
            return "invalid"
        seen.add(kind)
    if not seen:
        return "empty"
    if "aabb" in seen and ("polygon" in seen or "four_point" in seen):
        return "invalid"
    if "polygon" in seen:
        return "polygon"
    if "four_point" in seen:
        return "four_point"
    return "aabb"


def scan_source_levels(
    labels_dir: str | Path,
    intended_level: GeometryLevel = GeometryLevel.OBB,
    *,
    confirm_quads_are_polygons: bool = False,
) -> SourceLevelScan:
    """Scan a source's labels/ and resolve its single geometry level."""
    root = Path(labels_dir)
    poly_files: list[str] = []
    fourpt_files: list[str] = []
    aabb_files: list[str] = []
    invalid_files: list[str] = []

    for path in sorted(root.rglob("*.txt")):
        kind = _classify_file(path)
        if kind == "polygon":
            poly_files.append(path.name)
        elif kind == "four_point":
            fourpt_files.append(path.name)
        elif kind == "aabb":
            aabb_files.append(path.name)
        elif kind == "invalid":
            invalid_files.append(path.name)

    if invalid_files:
        return SourceLevelScan(
            resolved_level=intended_level,
            is_homogeneous=False,
            conflict_files=invalid_files,
            needs_confirmation=False,
            reason="Some label files contain malformed or internally mixed lines.",
        )

    has_poly = bool(poly_files)
    has_fourpt = bool(fourpt_files)
    has_aabb = bool(aabb_files)

    # aabb never coexists with obb/polygon evidence: you cannot mix axis-aligned
    # boxes with oriented/contour geometry in one homogeneous source.
    if has_aabb and (has_poly or has_fourpt):
        return SourceLevelScan(
            resolved_level=GeometryLevel.AABB,
            is_homogeneous=False,
            conflict_files=aabb_files,
            needs_confirmation=False,
            reason="Source mixes axis-aligned boxes with oriented/contour geometry.",
        )

    if has_poly and has_fourpt:
        if confirm_quads_are_polygons:
            return SourceLevelScan(
                resolved_level=GeometryLevel.POLYGON,
                is_homogeneous=True,
                reason="Quad files confirmed as genuine contours.",
            )
        return SourceLevelScan(
            resolved_level=GeometryLevel.POLYGON,
            is_homogeneous=False,
            conflict_files=fourpt_files,
            needs_confirmation=True,
            reason="Source mixes polygon files with four-point-only files.",
        )

    if has_poly:
        return SourceLevelScan(GeometryLevel.POLYGON, True)
    if has_fourpt:
        return SourceLevelScan(intended_level, True)
    if has_aabb:
        return SourceLevelScan(GeometryLevel.AABB, True)
    # No labels at all: treat as the intended level, homogeneous.
    return SourceLevelScan(
        intended_level, True, reason="No non-empty label files found."
    )
