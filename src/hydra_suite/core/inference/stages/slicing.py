from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator

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

# A tile batch is admitted against this fixed byte ceiling before any pixels
# are copied.  The model-input estimate uses float32 because that is the
# largest normal inference representation.  Callers may request a smaller
# limit, but not a larger one, until the sidecar-level resource policy can
# supply a stricter live budget.
MAX_TILE_BATCH_BYTES = 256 * 1024 * 1024
COMPACT_OUTPUT_BYTES_PER_DETECTION = 128
DENSE_MASK_BYTES_PER_PIXEL = 4


@dataclass(frozen=True)
class TileJob:
    """Lightweight provenance for one tile prediction."""

    frame_idx: int
    box: tuple[int, int, int, int] | None

    @property
    def offset(self) -> tuple[float, float]:
        if self.box is None:
            return (0.0, 0.0)
        return (float(self.box[0]), float(self.box[1]))


def estimated_prediction_job_bytes(
    *,
    imgsz: int,
    task: str,
    max_detections: int,
    source_bytes: int = 0,
) -> int:
    """Conservative live input/output bytes for one model prediction item."""
    side = max(1, int(imgsz))
    candidates = max(1, int(max_detections))
    model_input_bytes = side * side * 3 * 4
    compact_output = candidates * COMPACT_OUTPUT_BYTES_PER_DETECTION
    dense_output = (
        candidates * side * side * DENSE_MASK_BYTES_PER_PIXEL
        if task == "segment"
        else 0
    )
    return max(0, int(source_bytes)) + model_input_bytes + compact_output + dense_output


def _tile_job_estimated_bytes(
    plan: SlicePlan,
    imgsz: int,
    device_tiles: bool,
    task: str,
    max_detections: int,
) -> int:
    """Conservative peak bytes attributable to one live tile job."""

    tile_w, tile_h = plan.slice_wh
    frame_w, frame_h = plan.frame_wh
    source_pixels = max(
        tile_w * tile_h,
        frame_w * frame_h if plan.full_frame else 0,
    )
    # Numpy tiles need a contiguous uint8 copy. CUDA tile views do not, but
    # both paths can materialize a float32 model-input tensor.
    source_bytes = 0 if device_tiles else source_pixels * 3
    return estimated_prediction_job_bytes(
        imgsz=imgsz,
        task=task,
        max_detections=max_detections,
        source_bytes=source_bytes,
    )


def admitted_prediction_chunk_size(
    *,
    imgsz: int,
    task: str,
    max_detections: int,
    requested: int,
    source_bytes: int = 0,
    byte_budget: int = MAX_TILE_BATCH_BYTES,
    description: str = "prediction",
) -> int:
    """Admit a model batch against combined input and worst-case output bytes."""
    effective_budget = min(MAX_TILE_BATCH_BYTES, max(1, int(byte_budget)))
    per_job = estimated_prediction_job_bytes(
        imgsz=imgsz,
        task=task,
        max_detections=max_detections,
        source_bytes=source_bytes,
    )
    if per_job > effective_budget:
        raise ValueError(
            f"{description} is not resource-admissible: estimated peak={per_job} "
            f"bytes for one item exceeds the {effective_budget}-byte model batch budget"
        )
    return max(1, min(int(requested), MAX_TILE_CHUNK, effective_budget // per_job))


def admitted_tile_chunk_size(
    plan: SlicePlan,
    *,
    imgsz: int,
    device_tiles: bool,
    requested: int,
    byte_budget: int = MAX_TILE_BATCH_BYTES,
    task: str = "obb",
    max_detections: int = 20,
) -> int:
    """Return a finite tile chunk admitted by an explicit byte budget.

    Geometry is rejected before crop materialization when even one tile cannot
    fit. The diagnostic deliberately includes the geometry and estimated peak
    requested by the hardening plan.
    """

    effective_budget = min(MAX_TILE_BATCH_BYTES, max(1, int(byte_budget)))
    per_job = _tile_job_estimated_bytes(plan, imgsz, device_tiles, task, max_detections)
    if per_job > effective_budget:
        frame_w, frame_h = plan.frame_wh
        tile_w, tile_h = plan.slice_wh
        raise ValueError(
            "Sliced inference geometry is not resource-admissible: "
            f"frame={frame_w}x{frame_h}, tile={tile_w}x{tile_h}, "
            f"tiles={len(plan.tiles)}, estimated peak={per_job} bytes for one "
            f"tile exceeds the {effective_budget}-byte tile budget"
        )
    by_bytes = max(1, effective_budget // per_job)
    return max(1, min(int(requested), MAX_TILE_CHUNK, by_bytes))


def _iter_tile_jobs(frames: list, plan: SlicePlan) -> Iterator[TileJob]:
    """Yield tile provenance without retaining tile pixels."""

    for frame_idx in range(len(frames)):
        for box in plan.tiles:
            yield TileJob(frame_idx=frame_idx, box=box)
        if plan.full_frame:
            yield TileJob(frame_idx=frame_idx, box=None)


def iter_tile_job_chunks(
    frames: list,
    plan: SlicePlan,
    *,
    device_tiles: bool,
    chunk_size: int,
) -> Iterator[list[tuple[TileJob, Any]]]:
    """Materialize at most ``chunk_size`` tile images at a time."""

    jobs = iter(_iter_tile_jobs(frames, plan))
    while True:
        provenance = list(islice(jobs, max(1, int(chunk_size))))
        if not provenance:
            return
        chunk: list[tuple[TileJob, Any]] = []
        for job in provenance:
            frame = frames[job.frame_idx]
            if job.box is None:
                image = frame
            else:
                x0, y0, x1, y1 = job.box
                crop = frame[y0:y1, x0:x1]
                image = crop if device_tiles else np.ascontiguousarray(crop)
            chunk.append((job, image))
        yield chunk
        # On resume, discard the generator's own references before asking the
        # job iterator for another batch. A caller that intentionally retains
        # the yielded list still owns it; normal streaming callers do not pay
        # a hidden two-chunk lifetime tax.
        del chunk, provenance, frame, image, job


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
    for chunk in iter_tile_job_chunks(
        frames,
        plan,
        device_tiles=device_tiles,
        chunk_size=MAX_TILE_CHUNK,
    ):
        for job, image in chunk:
            x0, y0 = job.offset
            jobs.append((job.frame_idx, int(x0), int(y0)))
            images.append(image)
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
    from .obb import (
        _gpu_letterbox_batch,
        _invert_letterbox_on_result,
        effective_raw_detection_cap,
    )

    conf_floor = config.direct.confidence_floor
    classes = config.target_classes or None
    candidate_cap = effective_raw_detection_cap(config)
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
                max_det=candidate_cap,
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
                max_det=candidate_cap,
            )
        results.extend(chunk_results)
    return results


# The predict-tile routine formerly assembled here as `run_direct_sliced`
# (plan_slices -> _build_tile_jobs -> _predict_tiles -> extract/merge) now
# streams through `regions.Grid.iter_region_results`; `run_obb` (obb.py)
# immediately drives the shared extract_with_transform/merge_per_frame tail.
# See regions.py and the retired function's history for the
# TWO ORTHOGONAL DISPATCH DECISIONS (finding C1) this used to document
# inline: tiling/preprocess is decided by the frame kind; extraction is
# decided by ``runtime.tensor_on_cuda``. Both decisions are unchanged, just
# relocated to `Grid.iter_region_results` / `extract_with_transform`.
