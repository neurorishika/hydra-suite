"""YOLO label-file writing for active-learning datasets.

One label line is ``class_id`` followed by normalized coordinates. The encoding
depends on the level:

    aabb     -> 5 fields:  class cx cy w h        (YOLO detect)
    obb      -> 9 fields:  class x1 y1 ... x4 y4  (YOLO OBB)
    polygon  -> odd >= 7:  class x1 y1 ... xP yP  (YOLO segment)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from hydra_suite.utils.geometry_levels import GeometryLevel, classify_label_line

from .escalation import LabelRecord


def _normalized_points(points: np.ndarray, height: int, width: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)
    return pts


def _polygon_points(pts: np.ndarray) -> np.ndarray:
    """Point list for a POLYGON-level line, guaranteed to encode as one.

    `classify_label_line` reads any 9-field line as ``four_point`` (an OBB or
    quad), so a native contour with exactly 4 points would be written into a
    root stamped ``level=polygon`` and then read back as an OBB --
    `scan_source_levels` would disagree with that root's own `source.json`.
    bgsub happens to be safe (its contour filter drops anything with < 5
    points); YOLO's ``masks.xy`` carries no such guarantee, and neither does
    SAM2. Repeat the final vertex so the count moves off 4 without moving the
    geometry: a repeated vertex is a no-op for every polygon consumer.

    Fewer than 3 points is not a polygon at all and is refused rather than
    padded -- padding it would invent a shape the model never produced.
    """
    if pts.shape[0] < 3:
        raise ValueError(
            f"polygon-level record has only {pts.shape[0]} point(s); a polygon "
            "needs at least 3. Refusing to pad it into a shape the model did "
            "not produce."
        )
    if pts.shape[0] == 4:
        return np.vstack([pts, pts[-1:]])
    return pts


def _format_line(
    rec: LabelRecord, height: int, width: int, level: GeometryLevel
) -> str:
    pts = _normalized_points(rec.points, height, width)
    if level is GeometryLevel.AABB:
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        coords = [x1 + w * 0.5, y1 + h * 0.5, w, h]
    elif level is GeometryLevel.POLYGON:
        coords = list(_polygon_points(pts).reshape(-1))
    else:
        coords = list(pts.reshape(-1))
    body = " ".join(f"{float(v):.6f}" for v in coords)
    return f"{int(rec.class_id)} {body}\n"


def write_label_file(
    path: str | Path,
    records: Sequence[LabelRecord],
    frame_size: tuple[int, int],
    level: GeometryLevel,
) -> None:
    """Write one YOLO label file. `frame_size` is (height, width)."""
    height, width = int(frame_size[0]), int(frame_size[1])
    with Path(path).open("w") as fp:
        for rec in records:
            fp.write(_format_line(rec, height, width, level))


_LEVEL_BY_KIND = {
    "aabb": GeometryLevel.AABB,
    "four_point": GeometryLevel.OBB,
    "polygon": GeometryLevel.POLYGON,
}


def read_label_file(
    path: str | Path,
    frame_size: tuple[int, int],
) -> list[LabelRecord]:
    """Read one YOLO label file back into pixel-space LabelRecords.

    The inverse of `write_label_file`. `frame_size` is (height, width), the
    same convention, because the file stores normalised coordinates.

    Each line's level comes from its own field count via
    `classify_label_line`, not from a caller-supplied level. The `four_point`
    case (9 fields) is ambiguous in principle -- an OBB or a 4-point quad
    polygon -- but not here: `_polygon_points` repeats the final vertex
    precisely so a polygon-level file never contains a 4-point line. A
    9-field line is therefore always an OBB, including inside a source whose
    own `level` says polygon (an unpromoted leftover), which is exactly what
    a caller merging into that source needs to know.

    Confidence is not stored on disk. Records read back carry
    ``confidence=1.0`` ("asserted"); nothing downstream reads it, and
    `write_label_file` ignores it.

    Unparseable lines are skipped, matching `parse_obb_label`'s tolerance for
    files a user may have hand-edited. A missing file reads as empty -- a
    frame with no label file has no labels, which is not an error.
    """
    height, width = int(frame_size[0]), int(frame_size[1])
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []

    records: list[LabelRecord] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = classify_label_line(len(parts))
        level = _LEVEL_BY_KIND.get(kind)
        if level is None:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if level is GeometryLevel.AABB:
            cx, cy, w, h = coords
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0
            flat = [x1, y1, x2, y1, x2, y2, x1, y2]
        else:
            flat = coords

        pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)
        records.append(
            LabelRecord(
                class_id=class_id,
                confidence=1.0,
                points=pts,
                level=level,
            )
        )
    return records
