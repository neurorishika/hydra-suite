from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import SliceConfig
from ..result import OBBResult


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


def _extract_tile(result: Any, model_task: str, config, tile_local_idx: int):
    """Run the correct per-task extractor on one tile's ultralytics result.

    Returns an OBBResult in the TILE's local coordinate space (frame_idx is a
    throwaway tile index; re-stamped after remap).
    """
    import math as _math

    from .obb import (
        _extract_obb_from_boxes,
        _extract_obb_from_masks,
        extract_obb_result,
    )

    if model_task == "detect":
        return _extract_obb_from_boxes(
            result, tile_local_idx, _math.radians(config.direct.fixed_angle_deg)
        )
    if model_task == "segment":
        return _extract_obb_from_masks(
            result,
            tile_local_idx,
            config.raw_detection_cap,
            num_angles=config.direct.seg_num_angles,
            crop_size=config.direct.seg_crop_size,
            pad_ratio=config.direct.seg_pad_ratio,
            mask_threshold=config.direct.seg_mask_threshold,
        )
    return extract_obb_result(result, tile_local_idx)


def _offset_result(res, x0: int, y0: int, frame_idx: int):
    """Return a copy of ``res`` with all coordinates shifted by (x0, y0)."""
    from .obb import _empty_obb_result

    if res.num_detections == 0:
        return _empty_obb_result(frame_idx)
    centroids = res.centroids.copy()
    centroids[:, 0] += x0
    centroids[:, 1] += y0
    corners = res.corners.copy()
    corners[..., 0] += x0
    corners[..., 1] += y0
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=res.angles,
        sizes=res.sizes,
        shapes=res.shapes,
        confidences=res.confidences,
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, res.num_detections),
        class_ids=res.class_ids_or_zeros,
    )


def run_direct_sliced(frames, model, config, runtime):
    """Sliced-inference wrapper around the direct predict+extract path.

    Same return contract as ``_run_direct``. This module-level function is
    dispatched from ``run_obb`` when ``config.obb.direct.slice.enabled`` is True.
    CPU/MPS/gpu_fast return ``OBBResult`` per frame; the native-cuda tensor path
    is handled in Task 7.
    """
    from .merge import band_membership, merge_obb_detections
    from .obb import _apply_raw_detection_cap, _resolve_imgsz, merge_obb_results

    slice_cfg = config.direct.slice
    model_task = config.direct.model_task
    imgsz = _resolve_imgsz(model)

    # Native-cuda tensor path: delegated to Task 7 helper.
    if getattr(runtime, "tensor_on_cuda", False):
        from .slicing_cuda import run_direct_sliced_cuda

        return run_direct_sliced_cuda(frames, model, config, runtime, imgsz)

    # Plan is identical for every frame in the window (same size). Memoize on
    # the first frame's shape.
    first = frames[0]
    frame_hw = (int(first.shape[0]), int(first.shape[1]))
    plan = plan_slices(
        frame_hw,
        slice_cfg,
        imgsz,
        None,
        ref_object_px=slice_cfg.reference_body_px,
    )

    # Build the flattened tile job list across all frames.
    jobs = []  # (frame_idx, x0, y0, x1, y1) ; x1==-1 marks a full-frame job
    for fi, frame in enumerate(frames):
        for x0, y0, x1, y1 in plan.tiles:
            jobs.append((fi, x0, y0, x1, y1))
        if plan.full_frame:
            jobs.append((fi, 0, 0, -1, -1))

    # Crop every tile (numpy views; contiguous copy for the predict call).
    tile_imgs = []
    for fi, x0, y0, x1, y1 in jobs:
        if x1 < 0:
            tile_imgs.append(frames[fi])
        else:
            tile_imgs.append(np.ascontiguousarray(frames[fi][y0:y1, x0:x1]))

    conf_floor = config.direct.confidence_floor
    results = model.predict(
        tile_imgs,
        conf=conf_floor,
        iou=1.0,
        classes=config.target_classes or None,
        verbose=False,
        device=runtime.device,
    )

    # Extract + offset-remap, grouped by source frame.
    per_frame: dict[int, list] = {fi: [] for fi in range(len(frames))}
    for job, res in zip(jobs, results):
        fi, x0, y0, x1, y1 = job
        local = _extract_tile(res, model_task, config, fi)
        per_frame[fi].append(_offset_result(local, max(0, x0), max(0, y0), fi))

    out = []
    for fi in range(len(frames)):
        concat = merge_obb_results(fi, per_frame[fi])
        if concat.num_detections <= 1:
            merged = concat
        else:
            bands = band_membership(concat.corners, plan.tiles)
            merged = merge_obb_detections(
                concat,
                policy=slice_cfg.merge_policy,
                metric=slice_cfg.merge_metric,
                threshold=slice_cfg.merge_threshold,
                backend="cv2",  # non-cuda paths always cv2
                overlap_bands=bands,
                runtime=runtime,
            )
        out.append(_apply_raw_detection_cap(merged, config.raw_detection_cap))
    return out
