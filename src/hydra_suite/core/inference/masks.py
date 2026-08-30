"""Binary-mask and polygon geometry shared by SAM2 and SAM3 escalation."""

from __future__ import annotations

import cv2
import numpy as np


def clip_mask_to_polygon(
    mask: np.ndarray, polygon_px: list[tuple[float, float]]
) -> np.ndarray:
    """Zero out every *mask* pixel outside *polygon_px*.

    SAM2 treats its box prompt as soft guidance, not a hard crop -- the
    predicted mask can bleed past the OBB it was prompted with. Rasterizing
    the OBB's own polygon (not just its axis-aligned bbox) keeps a rotated
    OBB's escalated mask bounded correctly too.
    """
    poly_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    pts = np.round(np.asarray(polygon_px, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(poly_mask, [pts.reshape(-1, 1, 2)], 1)
    return mask.astype(bool) & poly_mask.astype(bool)


def mask_to_contour(
    mask: np.ndarray,
    epsilon_frac: float = 0.01,
    min_points: int = 6,
    min_area: float = 4.0,
) -> np.ndarray | None:
    """Largest external contour of a binary mask as an (P, 2) float32 array.

    Simplified with approxPolyDP (epsilon = epsilon_frac * perimeter). Returns
    None when the mask is empty or the largest contour is degenerate. Single
    contour only (YOLO-seg has no holes).
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    if approx.shape[0] < min_points:
        approx = c.reshape(-1, 2).astype(np.float32)  # keep detail if oversimplified
    return approx


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Rasterized IoU of two (P, 2) pixel-space polygons.

    Rasterizes both onto a shared integer grid covering their combined
    bounding box and counts pixels. Chosen over an analytic clip because
    SAM3 contours are arbitrary NON-CONVEX polygons: the convex
    Sutherland-Hodgman clip in ``utils/rotated_iou.py`` is only valid for
    quads and silently returns wrong areas for these. The polygons came
    from rasterized masks in the first place, so nothing is lost.

    The grid is supersampled 4x before counting: ``cv2.fillPoly`` fills
    boundary pixels on *all* edges of an axis-aligned, grid-aligned
    rectangle (a 20-unit square rasterizes to 21x21, not 20x20), which
    over-estimates area/overlap by a full pixel of border on every edge.
    Supersampling shrinks that one-pixel bias to a quarter-pixel at the
    finer resolution, which cancels out in the final IoU ratio.

    Returns 0.0 if either polygon has fewer than 3 points.

    Disjoint axis-aligned bounding boxes short-circuit to 0.0 BEFORE any
    canvas is allocated. Without that, two 20 px polygons at opposite
    corners of a 4512^2 frame share a ~4512^2 combined bbox, which at 4x
    supersampling is two (17680)^2 uint8 canvases -- ~625 MB and ~65 ms to
    compute a guaranteed 0.0. ``merge_candidates`` runs this for every
    candidate x survivor pair, and calibration repeats that over the whole
    confidence x fraction grid, so the disjoint case is the common case.
    """
    supersample = 4
    pa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if pa.shape[0] < 3 or pb.shape[0] < 3:
        return 0.0

    # Disjoint-bbox early-out. Uses >= / <= because polygons that merely
    # touch along an edge have zero-area intersection anyway.
    if (
        pa[:, 0].min() >= pb[:, 0].max()
        or pb[:, 0].min() >= pa[:, 0].max()
        or pa[:, 1].min() >= pb[:, 1].max()
        or pb[:, 1].min() >= pa[:, 1].max()
    ):
        return 0.0

    x0 = int(np.floor(min(pa[:, 0].min(), pb[:, 0].min())))
    y0 = int(np.floor(min(pa[:, 1].min(), pb[:, 1].min())))
    x1 = int(np.ceil(max(pa[:, 0].max(), pb[:, 0].max())))
    y1 = int(np.ceil(max(pa[:, 1].max(), pb[:, 1].max())))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0.0

    canvas_a = np.zeros((h * supersample, w * supersample), dtype=np.uint8)
    canvas_b = np.zeros((h * supersample, w * supersample), dtype=np.uint8)
    for poly, canvas in ((pa, canvas_a), (pb, canvas_b)):
        pts = np.round(
            (poly - np.array([x0, y0], dtype=np.float64)) * supersample
        ).astype(np.int32)
        cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], 1)

    inter = int(np.count_nonzero(canvas_a & canvas_b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(canvas_a | canvas_b))
    return float(inter) / float(union)
