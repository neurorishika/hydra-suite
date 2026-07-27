"""Pure tile-planning geometry shared by inference, training, and preview.

Extracted from ``core/inference/stages/slicing.py`` so the training dataset
builder and DetectKit preview tile with the EXACT same grid the inference path
uses (Approach B). No ``core.inference`` / ``training`` / Qt imports — plain
geometry beside ``utils/rotated_iou.py`` and ``utils/obb_from_mask.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Hard ceiling on tiles per frame (see the original slicing.py note, finding I5):
# refuse pathological slice/overlap combos loudly instead of spinning for hours.
MAX_TILES_PER_FRAME = 4096


@dataclass
class SlicePlan:
    """A memoizable tiling of a fixed-size frame."""

    tiles: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) per tile
    full_frame: bool  # append one full-frame pass in addition to tiles
    slice_wh: tuple[int, int]  # (w, h) of each tile
    frame_wh: tuple[int, int]  # (w, h) of the source frame

    @property
    def jobs_per_frame(self) -> int:
        return len(self.tiles) + (1 if self.full_frame else 0)


def _axis_starts(total: int, size: int, step: int) -> list[int]:
    """Tile start offsets along one axis, last tile flush to the far edge."""
    if size >= total:
        return [0]
    starts = list(range(0, total - size + 1, step))
    last = total - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_geometry(frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w):
    """Return ``(xs, ys, slice_w, slice_h)`` with sizes clamped to the frame."""
    slice_w = min(slice_w, frame_w)
    slice_h = min(slice_h, frame_h)
    step_x = max(1, int(slice_w * (1.0 - overlap_w)))
    step_y = max(1, int(slice_h * (1.0 - overlap_h)))
    return (
        _axis_starts(frame_w, slice_w, step_x),
        _axis_starts(frame_h, slice_h, step_y),
        slice_w,
        slice_h,
    )


def get_slice_bboxes(frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w):
    """SAHI ``get_slice_bboxes``: fixed-size tiles, last tile flush to the edge.

    Pure geometry primitive: deliberately UNGUARDED (``plan_tiles`` owns the
    tile-count ceiling), so tests can exercise degenerate steps directly.
    """
    xs, ys, slice_w, slice_h = _axis_geometry(
        frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w
    )
    return [(x, y, x + slice_w, y + slice_h) for y in ys for x in xs]


def tiles_overlap(tiles):
    """True when ANY two planned tiles actually intersect. Analytic for a regular grid."""
    n = len(tiles)
    if n <= 1:
        return False
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    w = tiles[0][2] - tiles[0][0]
    h = tiles[0][3] - tiles[0][1]
    regular_grid = n == len(xs) * len(ys) and all(
        (t[2] - t[0]) == w and (t[3] - t[1]) == h for t in tiles
    )
    if not regular_grid:
        return _tiles_overlap_pairwise(tiles)
    if any(b - a < w for a, b in zip(xs, xs[1:])):
        return True
    return any(b - a < h for a, b in zip(ys, ys[1:]))


def _tiles_overlap_pairwise(tiles):
    """O(T^2) reference definition; only reached for a non-grid tile list."""
    for i in range(len(tiles)):
        ax0, ay0, ax1, ay1 = tiles[i]
        for j in range(i + 1, len(tiles)):
            bx0, by0, bx1, by1 = tiles[j]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                return True
    return False


def tile_size_for_mode(
    *,
    geometry_mode: str,
    imgsz: int,
    reference_body_px: float,
    object_tile_fraction: float,
    slice_width: int,
    slice_height: int,
) -> tuple[int, int]:
    """Return (w, h) tile size for the configured geometry mode.

    Reproduces ``stages/slicing.py:_tile_size`` exactly (semantics are frozen).
    """
    if geometry_mode == "custom":
        w = slice_width if slice_width > 0 else imgsz
        h = slice_height if slice_height > 0 else imgsz
        return int(w), int(h)
    if geometry_mode == "auto_object" and reference_body_px > 0:
        frac = max(0.01, min(0.9, object_tile_fraction))
        size = int(round(reference_body_px / frac))
        size = max(64, min(4096, size))
        return size, size
    # auto_model (and auto_object fallback when no ref object is known).
    return int(imgsz), int(imgsz)


def plan_tiles(
    frame_hw,
    slice_w: int,
    slice_h: int,
    overlap_w: float,
    overlap_h: float,
    *,
    full_frame: bool = False,
    roi_mask: "np.ndarray | None" = None,
) -> SlicePlan:
    """Compute the tile plan for one frame size.

    Raises ``ValueError`` when the geometry would produce more than
    ``MAX_TILES_PER_FRAME`` tiles. ROI gating drops tiles with no live mask
    pixel; a mask whose shape mismatches the frame is treated as ``None``.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    xs, ys, eff_w, eff_h = _axis_geometry(
        frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w
    )
    n_tiles = len(xs) * len(ys)
    if n_tiles > MAX_TILES_PER_FRAME:
        raise ValueError(
            f"Sliced tiling would produce {n_tiles} tiles per frame "
            f"({len(xs)}x{len(ys)}) for a {frame_w}x{frame_h} frame with "
            f"{eff_w}x{eff_h} tiles at overlap ({overlap_w}, {overlap_h}) -- "
            f"above the {MAX_TILES_PER_FRAME}-tile ceiling. Increase the slice "
            f"size or lower the overlap."
        )
    tiles = [(x, y, x + eff_w, y + eff_h) for y in ys for x in xs]
    if roi_mask is not None and roi_mask.shape[:2] != (frame_h, frame_w):
        logger.warning(
            "ROI mask shape %s does not match frame (%d, %d); skipping ROI tile "
            "gating for this frame to avoid mis-gating.",
            tuple(roi_mask.shape[:2]),
            frame_h,
            frame_w,
        )
        roi_mask = None
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        kept = []
        for x0, y0, x1, y1 in tiles:
            yy0, yy1 = max(0, y0), min(h, y1)
            xx0, xx1 = max(0, x0), min(w, x1)
            if yy1 > yy0 and xx1 > xx0 and roi_mask[yy0:yy1, xx0:xx1].any():
                kept.append((x0, y0, x1, y1))
        tiles = kept if kept else tiles
    return SlicePlan(
        tiles=tiles,
        full_frame=bool(full_frame),
        slice_wh=(eff_w, eff_h),
        frame_wh=(frame_w, frame_h),
    )


def polygon_area(poly: np.ndarray) -> float:
    """Absolute area of an (N,2) polygon via the shoelace formula."""
    p = np.asarray(poly, dtype=np.float64)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_against_edge(poly, inside_fn, intersect_fn):
    """One Sutherland-Hodgman pass against a single half-plane."""
    out = []
    n = len(poly)
    if n == 0:
        return out
    for i in range(n):
        cur = poly[i]
        prev = poly[i - 1]
        cur_in = inside_fn(cur)
        prev_in = inside_fn(prev)
        if cur_in:
            if not prev_in:
                out.append(intersect_fn(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect_fn(prev, cur))
    return out


def clip_polygon_to_tile(
    poly_px: np.ndarray, tile: tuple[int, int, int, int]
) -> "np.ndarray | None":
    """Sutherland-Hodgman clip of an (N,2) polygon against an axis-aligned tile rect.

    Returns the clipped (M,2) polygon in the same pixel space, or None when the
    intersection is empty or degenerate (< 3 vertices, ~zero area).
    """
    x0, y0, x1, y1 = (float(tile[0]), float(tile[1]), float(tile[2]), float(tile[3]))
    poly = [
        np.asarray(p, dtype=np.float64) for p in np.asarray(poly_px, dtype=np.float64)
    ]

    def _lerp(a, b, t):
        return a + (b - a) * t

    # Left x>=x0, Right x<=x1, Top y>=y0, Bottom y<=y1.
    edges = [
        (
            lambda p: p[0] >= x0,
            lambda a, b: _lerp(
                a, b, (x0 - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
            ),
        ),
        (
            lambda p: p[0] <= x1,
            lambda a, b: _lerp(
                a, b, (x1 - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
            ),
        ),
        (
            lambda p: p[1] >= y0,
            lambda a, b: _lerp(
                a, b, (y0 - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
            ),
        ),
        (
            lambda p: p[1] <= y1,
            lambda a, b: _lerp(
                a, b, (y1 - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
            ),
        ),
    ]
    for inside_fn, intersect_fn in edges:
        poly = _clip_against_edge(poly, inside_fn, intersect_fn)
        if not poly:
            return None
    if len(poly) < 3:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    if polygon_area(arr) <= 1e-6:
        return None
    return arr
