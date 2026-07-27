"""Sliced (tiled) training-data builder for DetectKit direct OBB models.

Tiles a merged OBB dataset so a direct model learns to detect at the SAME scale
SAHI feeds at inference. Tiles through ``utils.slice_geometry`` — the exact grid
the inference path uses (Approach B). See
docs/superpowers/specs/2026-07-27-detectkit-sahi-sliced-training-design.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry_levels import GeometryLevel


def measure_reference_body_px(labels, frame_wh) -> float:
    """Median OBB major axis (px) over a frame's normalized-point labels."""
    w, h = float(frame_wh[0]), float(frame_wh[1])
    majors: list[float] = []
    for _cls_id, pts_norm in labels:
        pts = np.asarray(pts_norm, dtype=np.float32).copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        if pts.shape[0] < 3:
            continue
        _c, (bw, bh), _a = cv2.minAreaRect(pts.astype(np.float32))
        majors.append(float(max(bw, bh)))
    if not majors:
        return 0.0
    return float(np.median(np.asarray(majors, dtype=np.float64)))


def project_to_level(poly_norm: np.ndarray, level: GeometryLevel) -> np.ndarray:
    """Re-derive a normalized (M,2) contour DOWN to ``level`` (contour space kept)."""
    poly = np.asarray(poly_norm, dtype=np.float32)
    if level == GeometryLevel.POLYGON:
        return poly
    if level == GeometryLevel.OBB:
        box = cv2.boxPoints(cv2.minAreaRect(poly))
        return np.asarray(box, dtype=np.float32)
    # AABB: axis-aligned envelope corners.
    x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def label_line_for_level(
    class_id: int, pts_norm: np.ndarray, level: GeometryLevel
) -> str:
    """Format one YOLO label line for ``level`` (coords clipped to [0,1], %.6f)."""
    pts = np.clip(np.asarray(pts_norm, dtype=np.float32), 0.0, 1.0)
    if level == GeometryLevel.AABB:
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        return f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    coords = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {coords}"
