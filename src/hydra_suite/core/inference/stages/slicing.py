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
from ..result import OBBResult
from .obb import extract_with_transform, merge_per_frame
from .regions import Affine

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


def _offset_result(res, x0: int, y0: int, frame_idx: int):
    """Return a copy of ``res`` with all coordinates shifted by (x0, y0).

    Retained (Task 6 kept this despite retiring its ``run_direct_sliced``
    caller in favor of ``extract_with_transform``) because
    ``detectkit/gui/prediction_preview.py::predict_sliced_obb_result`` -- the
    executor-driven preview/AL sliced path, which has no ``RuntimeContext`` to
    hand ``extract_with_transform`` -- still imports and calls this directly.
    """
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
        per_frame[fi].append(
            extract_with_transform(
                res,
                fi,
                model_task,
                Affine(offset=(float(max(0, x0)), float(max(0, y0)))),
                config,
                runtime,
            )
        )
    return [
        merge_per_frame(per_frame[fi], "overlap_band_nms", plan, config, runtime)
        for fi in range(len(frames))
    ]
