from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import SliceConfig
from ..result import OBBResult

logger = logging.getLogger(__name__)

# Hard ceiling on tiles per frame. `SLICE_OVERLAP` clamps to 0.9 and the tile
# size to 8192, but slice=64 + overlap=0.9 (reachable via advanced_config.json)
# gives step 6 -> ~53k tiles on 1080p, i.e. 53k forward passes per frame. That
# is never a configuration anybody wants; refuse it loudly at plan time instead
# of spinning for hours with no log line (finding I5).
MAX_TILES_PER_FRAME = 4096

# Upper bound on the number of tile images handed to a single ``predict`` call.
# Mirrors ``runner._sliced_tile_batch``'s cap so a chunk can never exceed the
# dynamic-batch profile the TensorRT engine was exported with.
MAX_TILE_CHUNK = 128

# Emitted at most once per process: the ``gpu`` merge backend only exists on
# the native-CUDA (device-tensor) path (``slicing_cuda.py``); this host path
# (numpy detections, no on-device tensor for the gpu kernel to act on) always
# merges with cv2, which is also the correctness oracle. A user who sets
# ``merge_backend="gpu"`` on cpu/mps/gpu_fast gets cv2 silently otherwise.
_gpu_merge_backend_downgrade_logged = False


def _log_gpu_merge_backend_downgrade_once() -> None:
    global _gpu_merge_backend_downgrade_logged
    if _gpu_merge_backend_downgrade_logged:
        return
    logger.info(
        "SliceConfig.merge_backend='gpu' is only honored on the native-CUDA "
        "runtime tier; this run merges cross-tile detections with the cv2 "
        "backend instead (cv2 is the correctness oracle, so results are "
        "unaffected)."
    )
    _gpu_merge_backend_downgrade_logged = True


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


def _axis_geometry(
    frame_h: int,
    frame_w: int,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
) -> tuple[list[int], list[int], int, int]:
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

    Pure geometry primitive: deliberately UNGUARDED (``plan_slices`` owns the
    tile-count ceiling), so tests can exercise degenerate steps directly.
    """
    xs, ys, slice_w, slice_h = _axis_geometry(
        frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w
    )
    return [(x, y, x + slice_w, y + slice_h) for y in ys for x in xs]


def tiles_overlap(tiles: list[tuple[int, int, int, int]]) -> bool:
    """True when ANY two planned tiles actually intersect.

    Pure predicate over tile boxes -- no detection data, no device sync.

    Callers must use THIS, not ``SliceConfig.overlap_*_ratio``, to decide
    whether cross-tile dedup is needed: ``get_slice_bboxes`` flushes the last
    tile in each axis to the frame edge, so tiles genuinely overlap even at a
    configured ratio of 0.0 (a 300px frame with 256px tiles yields [0,256) and
    [44,300) -- 212px of real overlap).

    Computed ANALYTICALLY for the regular grid ``get_slice_bboxes`` emits
    (finding I5): all tiles share one size, so two distinct tiles intersect iff
    two distinct starts on some axis are closer than that axis's tile extent --
    which, since the closest pair of starts is always an adjacent pair, is a
    single scan over the sorted unique starts. The old pairwise scan was O(T^2):
    on a pathological 53k-tile plan that is ~2.8e9 pure-Python iterations before
    any forward pass, indistinguishable from a hang. An irregular tile list (not
    produced by ``get_slice_bboxes``) falls back to the O(T^2) definition.
    """
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


def _tiles_overlap_pairwise(tiles: list[tuple[int, int, int, int]]) -> bool:
    """O(T^2) reference definition; only reached for a non-grid tile list."""
    for i in range(len(tiles)):
        ax0, ay0, ax1, ay1 = tiles[i]
        for j in range(i + 1, len(tiles)):
            bx0, by0, bx1, by1 = tiles[j]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                return True
    return False


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

    COORDINATE-SPACE CONTRACT: ``roi_mask`` MUST be in the SAME pixel space as
    ``frame_hw`` (the frame the tiles are cut from). A mask whose ``shape[:2]``
    differs from ``frame_hw`` would gate the WRONG tiles (dropping real
    detections), so it is defensively treated as ``None`` (no gating, full grid)
    with a warning rather than silently mis-gating.

    Raises ``ValueError`` when the configured geometry would produce more than
    ``MAX_TILES_PER_FRAME`` tiles (finding I5).

    Cheap; caller memoizes per video.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    slice_w, slice_h = _tile_size(slice_cfg, imgsz, ref_object_px)
    xs, ys, eff_w, eff_h = _axis_geometry(
        frame_h,
        frame_w,
        slice_h,
        slice_w,
        slice_cfg.overlap_height_ratio,
        slice_cfg.overlap_width_ratio,
    )
    n_tiles = len(xs) * len(ys)
    if n_tiles > MAX_TILES_PER_FRAME:
        raise ValueError(
            f"Sliced inference would produce {n_tiles} tiles per frame "
            f"({len(xs)}x{len(ys)}) for a {frame_w}x{frame_h} frame with "
            f"{eff_w}x{eff_h} tiles at overlap "
            f"({slice_cfg.overlap_width_ratio}, {slice_cfg.overlap_height_ratio}) "
            f"-- above the {MAX_TILES_PER_FRAME}-tile ceiling, which would mean "
            f"{n_tiles} forward passes per frame. Increase the slice size or "
            f"lower SLICE_OVERLAP."
        )
    tiles = [(x, y, x + eff_w, y + eff_h) for y in ys for x in xs]
    if roi_mask is not None and roi_mask.shape[:2] != (frame_h, frame_w):
        # Wrong coordinate space: gating here would drop the wrong tiles. Degrade
        # to no gating (full grid) rather than mis-gate; downstream per-detection
        # ROI filtering still produces correct results, only compute is unsaved.
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
        # Fallback to the full grid if ROI gating would drop every tile. This prevents
        # silent detection failure while still producing correct ROI-filtered detections
        # downstream (filtering stage reapplies the mask per-detection).
        tiles = kept if kept else tiles
    return SlicePlan(
        tiles=tiles,
        full_frame=bool(slice_cfg.perform_standard_pred),
        slice_wh=(eff_w, eff_h),
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


def _build_tile_jobs(frames: list, plan: SlicePlan, device_tiles: bool):
    """Flatten (frame, tile) into parallel job/image lists.

    ``jobs`` are ``(frame_idx, x0, y0)`` provenance records; ``images`` are the
    cropped tiles in the SAME kind as the incoming frames -- zero-copy device
    views for CUDA-tensor frames, contiguous numpy copies otherwise (a numpy
    slice is a non-contiguous view, which ultralytics rejects).
    """
    jobs: list[tuple[int, int, int]] = []
    images: list[Any] = []
    for fi, frame in enumerate(frames):
        for x0, y0, x1, y1 in plan.tiles:
            jobs.append((fi, x0, y0))
            crop = frame[y0:y1, x0:x1]
            images.append(crop if device_tiles else np.ascontiguousarray(crop))
        if plan.full_frame:
            jobs.append((fi, 0, 0))
            images.append(frame)
    return jobs, images


def _predict_tiles(
    images: list,
    model,
    config,
    runtime,
    imgsz: int,
    *,
    letterbox: bool,
    chunk_size: int,
) -> list:
    """Run ``predict`` over the flattened tile list in bounded chunks.

    Chunking (finding I2): the flattened list is ``frames x tiles_per_frame``
    images. Issuing it as ONE predict call multiplies the peak activation memory
    the user configured via ``detection_batch_size`` by the tile count (25x for
    a 5x5 grid) with no cap. ``DirectExecutorAdapter`` self-chunks, but the
    plain torch CUDA/MPS path does not. ``chunk_size`` is bounded by
    tiles-per-frame (and ``MAX_TILE_CHUNK``), which is exactly the dynamic-batch
    profile ``runner._sliced_tile_batch`` exports TensorRT engines with.

    ``letterbox`` selects the preprocess hook: CUDA-tensor tiles fed to a plain
    ultralytics model must be GPU-letterboxed into one ``(B,3,imgsz,imgsz)``
    batch (a list of tensors is not a valid predict source) and the transform
    inverted on each result, exactly as ``obb._run_direct`` does.
    """
    from .obb import _gpu_letterbox_batch, _invert_letterbox_on_result

    conf_floor = config.direct.confidence_floor
    classes = config.target_classes or None
    results: list = []
    for start in range(0, len(images), chunk_size):
        part = images[start : start + chunk_size]
        if letterbox:
            batched, lb_params = _gpu_letterbox_batch(part, imgsz)
            chunk_results = model.predict(
                batched,
                conf=conf_floor,
                iou=1.0,
                classes=classes,
                verbose=False,
                device=runtime.device,
            )
            # Invert the letterbox so extract functions see tile-local
            # coordinates. When every tile is exactly imgsz x imgsz (the common
            # auto_model case) this is r=1, no pad -> a true no-op, skipped
            # entirely so no result tensor is touched. Real letterboxing only
            # kicks in for a custom tile size that differs from imgsz, or the
            # (rare) full-frame pass.
            for tile_img, res, (r, pad_left, pad_top) in zip(
                part, chunk_results, lb_params
            ):
                if r != 1.0 or pad_left != 0.0 or pad_top != 0.0:
                    _invert_letterbox_on_result(
                        res,
                        r,
                        pad_left,
                        pad_top,
                        orig_shape=(int(tile_img.shape[0]), int(tile_img.shape[1])),
                    )
        else:
            chunk_results = model.predict(
                part,
                conf=conf_floor,
                iou=1.0,
                classes=classes,
                verbose=False,
                device=runtime.device,
            )
        results.extend(chunk_results)
    return results


def _merge_frame_obb_results(parts, fi: int, plan: SlicePlan, config, runtime):
    """Concatenate one frame's per-tile ``OBBResult``s and dedup across tiles."""
    from .merge import band_membership, merge_obb_detections
    from .obb import _apply_raw_detection_cap, merge_obb_results

    concat = merge_obb_results(fi, parts)
    # Cap BEFORE the merge (finding I3): this bounds the O(n^2) cv2 hull/IoU
    # work to `cap` detections instead of `tiles x max_det`, and keeps this path
    # selecting the SAME detections as the device-tensor path (which caps inside
    # `materialize_tensors`). Cap again after merging so a nmm union that
    # reduces the count still yields cap-ordered ids.
    concat = _apply_raw_detection_cap(concat, config.raw_detection_cap)
    if concat.num_detections <= 1:
        return concat
    slice_cfg = config.direct.slice
    if slice_cfg.merge_backend == "gpu":
        _log_gpu_merge_backend_downgrade_once()
    bands = band_membership(concat.corners, plan.tiles)
    merged = merge_obb_detections(
        concat,
        policy=slice_cfg.merge_policy,
        metric=slice_cfg.merge_metric,
        threshold=slice_cfg.merge_threshold,
        # The gpu merge backend is reserved for the native-cuda (device tensor)
        # path; every host-side path uses the cv2 oracle (logged above once).
        backend="cv2",
        overlap_bands=bands,
        runtime=runtime,
    )
    return _apply_raw_detection_cap(merged, config.raw_detection_cap)


def run_direct_sliced(frames, model, config, runtime, roi_mask=None):
    """Sliced-inference wrapper around the direct predict+extract path.

    Same return contract as ``_run_direct``. This module-level function is
    dispatched from ``run_obb`` when ``config.obb.direct.slice.enabled`` is True.

    ``roi_mask`` (frame-space, same H x W as the frames) enables ROI tile gating
    in ``plan_slices``: tiles with no live ROI pixel are dropped, saving forward
    passes without changing the final ROI-filtered detection set. ``None``
    (the default) keeps the full tile grid.

    TWO ORTHOGONAL DISPATCH DECISIONS (finding C1 -- conflating them crashed
    both CUDA tiers):

    * **Tiling / preprocess** is decided by the FRAME KIND, exactly as
      ``obb._run_direct`` decides it: CUDA-tensor frames (NVDEC) fed to a plain
      ultralytics model need device-side tile views plus ``_gpu_letterbox_batch``;
      numpy frames -- and CUDA frames fed to a ``DirectExecutorAdapter``, which
      does its own letterbox -- take the plain frames-list path.
    * **Extraction** is decided by ``runtime.tensor_on_cuda``, which means
      "extraction yields device tensors", NOT "frames are CUDA tensors".

    The two are mutually exclusive in production (``tensor_on_cuda`` requires the
    torch backend = tier ``gpu``; NVDEC frames are confined to tier ``gpu_fast``
    -- see ``runtime._should_use_nvdec``), so neither may be inferred from the
    other. All four combinations are handled here:

    ==================  ================  =====================================
    frames              tensor_on_cuda    path
    ==================  ================  =====================================
    numpy               True              numpy tiles -> raw device tensors (gpu)
    numpy               False             numpy tiles -> OBBResult (cpu/mps/CoreML)
    cuda tensors        False             device tiles -> OBBResult (gpu_fast+TRT)
    cuda tensors        True              device tiles -> raw device tensors
    ==================  ================  =====================================
    """
    from .obb import DirectExecutorAdapter, _frames_are_cuda_tensors, _resolve_imgsz

    slice_cfg = config.direct.slice
    model_task = config.direct.model_task
    imgsz = _resolve_imgsz(model)

    device_frames = _frames_are_cuda_tensors(frames)
    # DirectExecutorAdapter accepts (and internally letterboxes) a raw list of
    # CUDA frames; pre-batching it double-preprocesses and corrupts the shape
    # fed to TensorRT. See the long comment at obb._run_direct's dispatch site.
    letterbox = device_frames and not isinstance(model, DirectExecutorAdapter)

    # Plan is identical for every frame in the window (same size). Memoize on
    # the first frame's shape.
    first = frames[0]
    frame_hw = (int(first.shape[0]), int(first.shape[1]))
    plan = plan_slices(
        frame_hw,
        slice_cfg,
        imgsz,
        roi_mask,
        ref_object_px=slice_cfg.reference_body_px,
    )

    jobs, images = _build_tile_jobs(frames, plan, device_frames)
    chunk_size = max(1, min(plan.jobs_per_frame, MAX_TILE_CHUNK))
    results = _predict_tiles(
        images,
        model,
        config,
        runtime,
        imgsz,
        letterbox=letterbox,
        chunk_size=chunk_size,
    )

    if runtime.tensor_on_cuda:
        from .slicing_cuda import assemble_raw_frames

        return assemble_raw_frames(jobs, results, len(frames), plan, config, runtime)

    per_frame: dict[int, list] = {fi: [] for fi in range(len(frames))}
    for (fi, x0, y0), res in zip(jobs, results):
        local = _extract_tile(res, model_task, config, fi)
        per_frame[fi].append(_offset_result(local, max(0, x0), max(0, y0), fi))
    return [
        _merge_frame_obb_results(per_frame[fi], fi, plan, config, runtime)
        for fi in range(len(frames))
    ]
