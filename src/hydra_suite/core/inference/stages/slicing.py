from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydra_suite.utils.slice_geometry import (  # noqa: F401 -- re-exported for callers/tests
    MAX_TILES_PER_FRAME,
    SlicePlan,
    get_slice_bboxes,
    plan_tiles,
    tile_size_for_mode,
    tiles_overlap,
)

from ..config import SliceConfig

logger = logging.getLogger(__name__)

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


def _tile_size(
    slice_cfg: SliceConfig, imgsz: int, ref_object_px: float
) -> tuple[int, int]:
    return tile_size_for_mode(
        geometry_mode=slice_cfg.geometry_mode,
        imgsz=imgsz,
        reference_body_px=ref_object_px,
        object_tile_fraction=slice_cfg.object_tile_fraction,
        slice_width=slice_cfg.slice_width,
        slice_height=slice_cfg.slice_height,
    )


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
    slice_w, slice_h = _tile_size(slice_cfg, imgsz, ref_object_px)
    return plan_tiles(
        frame_hw,
        slice_w,
        slice_h,
        slice_cfg.overlap_width_ratio,
        slice_cfg.overlap_height_ratio,
        full_frame=bool(slice_cfg.perform_standard_pred),
        roi_mask=roi_mask,
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


# The predict-tile routine formerly assembled here as `run_direct_sliced`
# (plan_slices -> _build_tile_jobs -> _predict_tiles -> extract/merge) now
# lives, verbatim, in `regions.Grid.plan` (tiling) + `regions.Grid.execute`
# (tile predict) -- `run_obb` (obb.py) drives the shared
# extract_with_transform/merge_per_frame tail for every RegionSource,
# including Grid. See regions.py and the retired function's history for the
# TWO ORTHOGONAL DISPATCH DECISIONS (finding C1) this used to document
# inline: tiling/preprocess is decided by the frame kind; extraction is
# decided by ``runtime.tensor_on_cuda``. Both decisions are unchanged, just
# relocated to `Grid.execute` / `extract_with_transform`.
