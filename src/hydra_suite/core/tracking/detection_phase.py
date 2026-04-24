"""Batched YOLO detection phase for the tracking pipeline.

Runs YOLO detection on all frames (or a specified range) and caches
the results, reporting progress via callbacks.

When PyNvVideoCodec and cupy are installed and the runtime uses a CUDA-backed
direct OBB executor, the phase can decode frames directly to GPU memory via
NVDec hardware decode, eliminating all CPU↔GPU frame transfers during Phase 1.
"""

import logging
import time
from collections import deque

import cv2

from hydra_suite.utils.batch_optimizer import BatchOptimizer

logger = logging.getLogger(__name__)


def _init_batch_optimizer(params):
    """Create a BatchOptimizer from tracking parameters."""
    advanced_config = params.get("ADVANCED_CONFIG", {}).copy()
    advanced_config["enable_tensorrt"] = params.get("ENABLE_TENSORRT", False)
    advanced_config["tensorrt_max_batch_size"] = params.get(
        "TENSORRT_MAX_BATCH_SIZE", 16
    )
    advanced_config["tracking_realtime_mode"] = params.get(
        "TRACKING_REALTIME_MODE", False
    )
    advanced_config["tracking_workflow_mode"] = params.get(
        "TRACKING_WORKFLOW_MODE", "non_realtime"
    )
    return BatchOptimizer(advanced_config)


def _read_batch_frames(
    cap, batch_size, start_frame, end_frame, frame_idx, resize_factor, is_stop_requested
):
    """Read up to *batch_size* frames from *cap*, applying optional resize.

    Returns ``(batch_frames, frames_consumed)`` where *frames_consumed* is
    the number of frames successfully read (and added to the batch).
    """
    batch_frames = []
    consumed = 0
    for _ in range(batch_size):
        if is_stop_requested():
            break
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_index = start_frame + frame_idx + consumed
        if current_frame_index > end_frame:
            break
        if resize_factor < 1.0:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=resize_factor,
                fy=resize_factor,
                interpolation=cv2.INTER_AREA,
            )
        batch_frames.append(frame)
        consumed += 1
    return batch_frames, consumed


def _cache_batch_results(detection_cache, batch_results, batch_start_idx, start_frame):
    """Write a batch of detection results into the detection cache."""
    for local_idx, (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_heading_hints,
        raw_heading_confidences,
        raw_directed_mask,
        raw_canonical_affines,
    ) in enumerate(batch_results):
        actual_frame_idx = start_frame + batch_start_idx + local_idx
        detection_ids = [actual_frame_idx * 10000 + i for i in range(len(raw_meas))]
        detection_cache.add_frame(
            actual_frame_idx,
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            detection_ids,
            raw_heading_hints,
            raw_heading_confidences,
            raw_directed_mask,
            canonical_affines=raw_canonical_affines,
        )


def _compute_batch_stats(
    batch_times, batch_size, batch_frames, frame_idx, total_frames, detection_start_time
):
    """Compute FPS, ETA, and percentage progress after a batch completes."""
    elapsed = time.time() - detection_start_time
    if len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
        frames_per_batch = (
            batch_size if len(batch_frames) == batch_size else len(batch_frames)
        )
        current_fps = frames_per_batch / avg_batch_time if avg_batch_time > 0 else 0
    else:
        current_fps = 0

    eta = (total_frames - frame_idx) / current_fps if current_fps > 0 else 0
    percentage = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
    return current_fps, elapsed, eta, percentage


# ---------------------------------------------------------------------------
# NVDec GPU hardware-decode helpers
# ---------------------------------------------------------------------------


def _nvdec_frame_to_cuda_tensor(frame, cp):
    """Convert a PyNvVideoCodec DecodedFrame to a CUDA torch.Tensor (zero-copy).

    The frame must have been decoded with ``useDeviceMemory=True`` and
    ``outputColorType=RGB``.  The returned tensor shares decoder memory and
    *must* be cloned before the next ``get_batch_frames()`` call.
    """
    import torch

    planes = frame.cuda()
    if not planes:
        raise ValueError("NVDec frame has no CUDA planes")
    cai = planes[0].__cuda_array_interface__
    shape = cai["shape"]
    byte_size = shape[0] * shape[1] * shape[2]
    mem = cp.cuda.UnownedMemory(cai["data"][0], byte_size, frame)
    ptr = cp.cuda.MemoryPointer(mem, 0)
    strides = cai.get("strides") or None
    cp_arr = cp.ndarray(shape=shape, dtype=cp.uint8, memptr=ptr, strides=strides)
    return torch.as_tensor(cp_arr, device="cuda")


def _should_use_nvdec(params: dict, detector) -> bool:
    """Return True when NVDec GPU decode should be used for Phase-1 batched detection.

    Conditions (all must hold):
    - PyNvVideoCodec and cupy are importable
    - COMPUTE_RUNTIME is ``onnx_cuda`` or ``tensorrt``
    - The detector's direct OBB executor is active
    - RESIZE_FACTOR is 1.0 (sub-resolution resize is handled by the executor's
      GPU letterbox, not a separate cv2 step, so the overall frame scale must
      be 1.0 for coordinate bookkeeping to remain consistent)
    """
    if float(params.get("RESIZE_FACTOR", 1.0)) != 1.0:
        return False
    compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu")).strip().lower()
    if compute_runtime not in {"onnx_cuda", "tensorrt"}:
        return False
    if getattr(detector, "_direct_obb_executor", None) is None:
        return False
    try:
        import cupy  # noqa: F401
        import PyNvVideoCodec  # noqa: F401

        return True
    except ImportError:
        return False


def _try_open_nvdec(video_path: str, start_frame: int):
    """Try to open an NVDec hardware decoder for *video_path*.

    Returns ``(decoder, metadata, cp_module)`` on success, ``None`` on failure.
    """
    try:
        import cupy as cp
        import PyNvVideoCodec as nvc
    except ImportError:
        return None
    try:
        dec = nvc.CreateSimpleDecoder(
            encSource=str(video_path),
            gpuid=0,
            useDeviceMemory=True,
            outputColorType=nvc.OutputColorType.RGB,
        )
        meta = dec.get_stream_metadata()
        if start_frame > 0:
            dec.seek_to_index(int(start_frame))
        return dec, meta, cp
    except Exception as exc:
        logger.warning("NVDec: failed to open decoder for %s: %s", video_path, exc)
        return None


def _read_nvdec_batch(
    sdec, cp, batch_size, start_frame, end_frame, frame_idx, is_stop_requested
):
    """Read up to *batch_size* NVDec-decoded CUDA tensors.

    Each frame is cloned immediately after decode so the decoder buffer can
    safely be reused for the next frame.

    Returns ``(batch_tensors, frames_consumed)``.
    """
    batch_tensors = []
    consumed = 0
    for _ in range(batch_size):
        if is_stop_requested():
            break
        current_frame_index = start_frame + frame_idx + consumed
        if current_frame_index > end_frame:
            break
        batch = sdec.get_batch_frames(1)
        if not batch:
            break
        cuda_tensor = _nvdec_frame_to_cuda_tensor(batch[0], cp)
        # Clone immediately — NVDec decoder buffer is reused on next get_batch_frames().
        batch_tensors.append(cuda_tensor.clone())
        consumed += 1
    return batch_tensors, consumed


def run_batched_detection_phase(
    cap,
    detection_cache,
    detector,
    params,
    start_frame,
    end_frame,
    is_stop_requested,
    on_progress=None,
    on_stats=None,
    profiler=None,
    video_path: str = "",
):
    """Run batched YOLO detection on a frame range and cache results.

    Args:
        cap: OpenCV VideoCapture object.
        detection_cache: DetectionCache for writing.
        detector: YOLOOBBDetector instance.
        params: Configuration parameters.
        start_frame: Starting frame index (0-based).
        end_frame: Ending frame index (0-based).
        is_stop_requested: Callable returning True when stop is requested.
        on_progress: Optional callback ``(percentage: int, status: str) -> None``.
        on_stats: Optional callback ``(stats: dict) -> None``.
        profiler: Optional TrackingProfiler.
        video_path: Source video path string (required for NVDec GPU decode).

    Returns:
        int: Total frames processed.
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Batched YOLO Detection")
    logger.info("=" * 80)

    batch_optimizer = _init_batch_optimizer(params)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = end_frame - start_frame + 1

    logger.info(
        f"Processing frame range: {start_frame} to {end_frame} ({total_frames} frames)"
    )

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    effective_width = int(frame_width * resize_factor)
    effective_height = int(frame_height * resize_factor)

    model_name = params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt")
    batch_size = batch_optimizer.estimate_batch_size(
        effective_width, effective_height, model_name
    )

    logger.info(f"Video: {frame_width}x{frame_height}, {total_frames} frames")
    if resize_factor < 1.0:
        logger.info(
            f"Resize factor: {resize_factor} → Effective: {effective_width}x{effective_height}"
        )
    logger.info(f"Batch size: {batch_size}")

    # Try to open the NVDec hardware decoder as a zero-copy alternative to cv2.
    # Falls back to cv2 silently on any failure or when conditions are not met.
    use_nvdec = False
    nvdec_dec = None
    nvdec_cp = None
    if video_path and _should_use_nvdec(params, detector):
        _nvdec_result = _try_open_nvdec(video_path, start_frame)
        if _nvdec_result is not None:
            nvdec_dec, _nvdec_meta, nvdec_cp = _nvdec_result
            use_nvdec = True
            logger.info(
                "NVDec GPU hardware decode enabled for Phase-1 " "(%dx%d, ~%d frames)",
                _nvdec_meta.width,
                _nvdec_meta.height,
                _nvdec_meta.num_frames,
            )
        else:
            logger.info("NVDec decoder unavailable for this video; using cv2 decode.")

    detection_start_time = time.time()
    batch_times = deque(maxlen=30)

    frame_idx = 0
    batch_count = 0
    total_batches = (total_frames + batch_size - 1) // batch_size

    while not is_stop_requested():
        batch_start_time = time.time()
        batch_start_idx = frame_idx

        if profiler:
            profiler.tick("batched_frame_read")
        _t_decode_start = time.time()
        if use_nvdec:
            batch_frames, consumed = _read_nvdec_batch(
                nvdec_dec,
                nvdec_cp,
                batch_size,
                start_frame,
                end_frame,
                frame_idx,
                is_stop_requested,
            )
        else:
            batch_frames, consumed = _read_batch_frames(
                cap,
                batch_size,
                start_frame,
                end_frame,
                frame_idx,
                resize_factor,
                is_stop_requested,
            )
        frame_idx += consumed
        _t_decode = time.time() - _t_decode_start
        if profiler:
            profiler.tock("batched_frame_read")

        if not batch_frames or is_stop_requested():
            break

        batch_count += 1
        logger.info(
            f"Processing batch {batch_count}/{total_batches} ({len(batch_frames)} frames)"
        )

        def progress_cb(
            current,
            total,
            msg,
            _batch_start=batch_start_idx,
            _batch_num=batch_count,
            _total_batches=total_batches,
        ):
            if is_stop_requested() or total <= 0:
                return
            if current != total and current % 10 != 0:
                return
            batch_fraction = float(current) / float(total)
            overall_processed = _batch_start + current
            overall_pct = (
                int((overall_processed * 100) / total_frames) if total_frames > 0 else 0
            )
            if on_progress:
                on_progress(
                    overall_pct,
                    "Detecting objects: "
                    f"batch {_batch_num}/{_total_batches}, "
                    f"within-batch {int(batch_fraction * 100)}% "
                    f"({current}/{total})",
                )

        batch_results = detector.detect_objects_batched(
            batch_frames,
            batch_start_idx,
            progress_cb,
            return_raw=True,
            profiler=profiler,
        )

        _bt = getattr(detector, "_batch_timings", None)
        if _bt:
            logger.info(
                "  decode=%.3fs  obb=%.3fs  nms=%.3fs  ht=%.3fs(%d crops)",
                _t_decode,
                _bt.get("obb_s", 0.0),
                _bt.get("nms_s", 0.0),
                _bt.get("ht_s", 0.0),
                _bt.get("n_ht_crops", 0),
            )

        _cache_batch_results(
            detection_cache, batch_results, batch_start_idx, start_frame
        )

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        current_fps, elapsed, eta, percentage = _compute_batch_stats(
            batch_times,
            batch_size,
            batch_frames,
            frame_idx,
            total_frames,
            detection_start_time,
        )

        status_text = (
            f"Detecting objects: batch {batch_count}/{total_batches} ({percentage}%)"
        )
        if on_progress:
            on_progress(percentage, status_text)
        if on_stats:
            on_stats({"fps": current_fps, "elapsed": elapsed, "eta": eta})

    logger.info(
        f"Detection phase complete: {frame_idx} frames processed in {batch_count} batches"
    )
    return frame_idx
