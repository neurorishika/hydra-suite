from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SliceConfig


@dataclass
class SlicePlan:
    """A memoizable tiling of a fixed-size frame."""

    tiles: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) per tile
    full_frame: bool  # append one full-frame pass in addition to tiles
    slice_wh: tuple[int, int]  # (w, h) of each tile
    frame_wh: tuple[int, int]  # (w, h) of the source frame


def get_slice_bboxes(
    frame_h: int,
    frame_w: int,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
) -> list[tuple[int, int, int, int]]:
    """SAHI ``get_slice_bboxes``: fixed-size tiles, last tile flush to the edge.

    Step = slice * (1 - overlap). The final tile in each axis is shifted back so
    its far edge sits exactly on the frame edge (never a shrunken runt tile), so
    two frames of the same size always tile identically.
    """
    slice_w = min(slice_w, frame_w)
    slice_h = min(slice_h, frame_h)
    step_x = max(1, int(slice_w * (1.0 - overlap_w)))
    step_y = max(1, int(slice_h * (1.0 - overlap_h)))

    def _starts(total: int, size: int, step: int) -> list[int]:
        if size >= total:
            return [0]
        starts = list(range(0, total - size + 1, step))
        last = total - size
        if starts[-1] != last:
            starts.append(last)
        return starts

    xs = _starts(frame_w, slice_w, step_x)
    ys = _starts(frame_h, slice_h, step_y)
    return [(x, y, x + slice_w, y + slice_h) for y in ys for x in xs]


def _tile_size(
    slice_cfg: SliceConfig, imgsz: int, ref_object_px: float
) -> tuple[int, int]:
    """Return (w, h) tile size for the configured geometry mode."""
    if slice_cfg.geometry_mode == "custom":
        w = slice_cfg.slice_width if slice_cfg.slice_width > 0 else imgsz
        h = slice_cfg.slice_height if slice_cfg.slice_height > 0 else imgsz
        return int(w), int(h)
    if slice_cfg.geometry_mode == "auto_object" and ref_object_px > 0:
        frac = max(0.01, min(0.9, slice_cfg.object_tile_fraction))
        size = int(round(ref_object_px / frac))
        size = max(64, min(4096, size))
        return size, size
    # auto_model (and auto_object fallback when no ref object is known).
    return int(imgsz), int(imgsz)


def plan_slices(
    frame_hw: tuple[int, int],
    slice_cfg: SliceConfig,
    imgsz: int,
    roi_mask: np.ndarray | None,
    ref_object_px: float = 0.0,
) -> SlicePlan:
    """Compute the tile plan for one frame size.

    ROI gating is a compute optimization only: the mask is re-applied per-detection
    downstream in the filtering stage, so dropping tiles only saves forward passes.
    If the mask would eliminate every tile (e.g. empty ROI), we fall back to the
    full grid to prevent silent detection failure, degrading gracefully to "no compute
    savings" while still producing correct ROI-filtered detections.

    Cheap; caller memoizes per video.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    slice_w, slice_h = _tile_size(slice_cfg, imgsz, ref_object_px)
    tiles = get_slice_bboxes(
        frame_h,
        frame_w,
        slice_h,
        slice_w,
        slice_cfg.overlap_height_ratio,
        slice_cfg.overlap_width_ratio,
    )
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        kept = []
        for x0, y0, x1, y1 in tiles:
            yy0, yy1 = max(0, y0), min(h, y1)
            xx0, xx1 = max(0, x0), min(w, x1)
            if yy1 > yy0 and xx1 > xx0 and roi_mask[yy0:yy1, xx0:xx1].any():
                kept.append((x0, y0, x1, y1))
        # Fallback to the full grid if ROI gating would drop every tile. This prevents
        # silent detection failure while still producing correct ROI-filtered detections
        # downstream (filtering stage reapplies the mask per-detection).
        tiles = kept if kept else tiles
    return SlicePlan(
        tiles=tiles,
        full_frame=bool(slice_cfg.perform_standard_pred),
        slice_wh=(slice_w, slice_h),
        frame_wh=(frame_w, frame_h),
    )
