"""Probe TrackerKit OBB detector runtime on real video frames.

This module benchmarks the same detector code used by TrackerKit against
sampled frames from a real video so runtime discrepancies can be broken down
into frame I/O, resize, raw detector call, detector filtering, and the
internal profiler phases such as ``yolo_obb_inference``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2

from hydra_suite.core.detectors import YOLOOBBDetector
from hydra_suite.core.tracking.profiler import TrackingProfiler
from hydra_suite.runtime.compute_runtime import CANONICAL_RUNTIMES, _normalize_runtime
from hydra_suite.utils.frame_prefetcher import FramePrefetcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to the source video.")
    parser.add_argument("--model", required=True, help="Path to the YOLO OBB model.")
    parser.add_argument(
        "--runtime",
        default="tensorrt",
        choices=sorted(CANONICAL_RUNTIMES),
        help="Canonical runtime to use for the OBB detector.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Detector batch size. Use 1 to match live realtime tracking.",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Resize factor applied before detection, matching TrackerKit.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First video frame index to sample.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride between sampled frames.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=8,
        help="Number of sampled frames to warm up before measuring.",
    )
    parser.add_argument(
        "--measure-frames",
        type=int,
        default=64,
        help="Number of sampled frames to measure.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=25,
        help="Maximum target count passed to the detector.",
    )
    parser.add_argument(
        "--tensorrt-max-batch-size",
        type=int,
        default=None,
        help="Static TensorRT engine batch size. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--tensorrt-build-batch-size",
        type=int,
        default=None,
        help="TensorRT build batch size. Defaults to --tensorrt-max-batch-size.",
    )
    parser.add_argument(
        "--tensorrt-workspace-gb",
        type=float,
        default=4.0,
        help="TensorRT builder workspace in GB.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional explicit YOLO image size override.",
    )
    parser.add_argument(
        "--headtail-model",
        default="",
        help="Optional head-tail model path to include inline classifier cost.",
    )
    parser.add_argument(
        "--headtail-runtime",
        default=None,
        choices=sorted(CANONICAL_RUNTIMES),
        help="Optional head-tail runtime override. Defaults to --runtime.",
    )
    parser.add_argument(
        "--headtail-batch-size",
        type=int,
        default=64,
        help="Head-tail crop batch size when --headtail-model is used.",
    )
    parser.add_argument(
        "--tracking-realtime-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TrackerKit realtime mode behavior for head-tail hints.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the final summary as JSON.",
    )
    parser.add_argument(
        "--read-mode",
        choices=["direct", "prefetch"],
        default="direct",
        help="Frame read strategy to benchmark.",
    )
    parser.add_argument(
        "--prefetch-buffer-size",
        type=int,
        default=2,
        help="Buffer size when --read-mode=prefetch.",
    )
    return parser


def build_detector_params(
    *,
    model_path: str,
    runtime: str,
    batch_size: int,
    max_targets: int,
    tensorrt_max_batch_size: int | None = None,
    tensorrt_build_batch_size: int | None = None,
    tensorrt_workspace_gb: float = 4.0,
    imgsz: int | None = None,
    headtail_model_path: str = "",
    headtail_runtime: str | None = None,
    headtail_batch_size: int = 64,
    tracking_realtime_mode: bool = True,
) -> dict[str, Any]:
    normalized_runtime = _normalize_runtime(runtime)
    resolved_batch_size = max(1, int(batch_size))
    trt_max_batch_size = max(
        1,
        int(
            tensorrt_max_batch_size
            if tensorrt_max_batch_size is not None
            else resolved_batch_size
        ),
    )
    trt_build_batch_size = max(
        1,
        int(
            tensorrt_build_batch_size
            if tensorrt_build_batch_size is not None
            else trt_max_batch_size
        ),
    )
    device = "cpu"
    enable_tensorrt = False
    enable_onnx_runtime = False
    if normalized_runtime == "mps":
        device = "mps"
    elif normalized_runtime in {"cuda", "rocm"}:
        device = "cuda:0"
    elif normalized_runtime == "tensorrt":
        device = "cuda:0"
        enable_tensorrt = True
    elif normalized_runtime == "onnx_coreml":
        device = "mps"
        enable_onnx_runtime = True
    elif normalized_runtime in {"onnx_cpu", "cpu"}:
        device = "cpu"
        enable_onnx_runtime = normalized_runtime.startswith("onnx_")
    elif normalized_runtime in {"onnx_cuda", "onnx_rocm"}:
        device = "cuda:0"
        enable_onnx_runtime = True

    params: dict[str, Any] = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_MODEL_PATH": str(model_path),
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": str(model_path),
        "YOLO_DETECT_MODEL_PATH": "",
        "YOLO_CROP_OBB_MODEL_PATH": str(model_path),
        "YOLO_HEADTAIL_MODEL_PATH": str(headtail_model_path or ""),
        "YOLO_DEVICE": device,
        "ENABLE_TENSORRT": enable_tensorrt,
        "ENABLE_ONNX_RUNTIME": enable_onnx_runtime,
        "ENABLE_YOLO_BATCHING": resolved_batch_size > 1,
        "YOLO_BATCH_SIZE_MODE": "manual",
        "YOLO_MANUAL_BATCH_SIZE": resolved_batch_size,
        "TENSORRT_MAX_BATCH_SIZE": trt_max_batch_size,
        "TENSORRT_BUILD_BATCH_SIZE": trt_build_batch_size,
        "TENSORRT_BUILD_WORKSPACE_GB": float(tensorrt_workspace_gb),
        "MAX_TARGETS": max(1, int(max_targets)),
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.7,
        "RAW_YOLO_CONFIDENCE_FLOOR": 1e-3,
        "USE_CUSTOM_OBB_IOU_FILTERING": True,
        "TRACKING_REALTIME_MODE": bool(tracking_realtime_mode),
        "TRACKING_WORKFLOW_MODE": (
            "realtime" if tracking_realtime_mode else "non_realtime"
        ),
        "HEADTAIL_BATCH_SIZE": max(1, int(headtail_batch_size)),
        "HEADTAIL_COMPUTE_RUNTIME": str(
            headtail_runtime or normalized_runtime or "cpu"
        ),
        "YOLO_HEADTAIL_CONF_THRESHOLD": 0.5,
        "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD": 0.25,
    }
    if imgsz not in (None, 0):
        params["YOLO_IMGSZ"] = int(imgsz)
    return params


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p95_ms": 0.0,
        }
    sorted_values = sorted(float(value) for value in values)
    p95_index = min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)
    return {
        "count": len(sorted_values),
        "mean_ms": float(statistics.fmean(sorted_values)),
        "median_ms": float(statistics.median(sorted_values)),
        "min_ms": float(sorted_values[0]),
        "max_ms": float(sorted_values[-1]),
        "p95_ms": float(sorted_values[p95_index]),
    }


def _read_next_sampled_frame(
    capture: cv2.VideoCapture,
    stride: int,
) -> tuple[bool, Any]:
    ok, frame = capture.read()
    if not ok:
        return False, None
    for _ in range(max(0, stride - 1)):
        if not capture.grab():
            break
    return True, frame


def _read_next_prefetched_frame(
    prefetcher: FramePrefetcher,
    stride: int,
) -> tuple[bool, Any]:
    ok, frame = prefetcher.read()
    if not ok:
        return False, None
    for _ in range(max(0, stride - 1)):
        skip_ok, _skip_frame = prefetcher.read()
        if not skip_ok:
            break
    return True, frame


def _resize_frame(frame, resize_factor: float):
    if resize_factor >= 0.999:
        return frame
    return cv2.resize(
        frame,
        (0, 0),
        fx=resize_factor,
        fy=resize_factor,
        interpolation=cv2.INTER_LINEAR,
    )


def _phase_delta(
    profiler: TrackingProfiler,
    before: dict[str, float],
    phase_name: str,
) -> float:
    after_value = float(getattr(profiler, "_phase_times", {}).get(phase_name, 0.0))
    return max(0.0, (after_value - float(before.get(phase_name, 0.0))) * 1000.0)


def _capture_phase_snapshot(profiler: TrackingProfiler) -> dict[str, float]:
    return dict(getattr(profiler, "_phase_times", {}))


def normalize_raw_detector_output(raw: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize single-frame and batched return_raw detector payloads.

    Single-frame ``detect_objects(..., return_raw=True)`` includes the
    intermediate ``yolo_results`` item while ``detect_objects_batched`` does not.
    This helper strips that field so downstream timing code can treat both
    variants uniformly.
    """
    if len(raw) == 10:
        (
            raw_meas,
            raw_sizes,
            raw_shapes,
            _yolo_results,
            raw_confidences,
            raw_obb_corners,
            raw_heading_hints,
            raw_heading_confidences,
            raw_directed_mask,
            raw_canonical_affines,
        ) = raw
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            raw_heading_hints,
            raw_heading_confidences,
            raw_directed_mask,
            raw_canonical_affines,
        )
    if len(raw) == 9:
        return raw
    raise ValueError(f"Unexpected raw detector payload length: {len(raw)}")


def benchmark_video(
    *,
    video_path: str,
    params: dict[str, Any],
    resize_factor: float,
    batch_size: int,
    start_frame: int,
    stride: int,
    warmup_frames: int,
    measure_frames: int,
    read_mode: str = "direct",
    prefetch_buffer_size: int = 2,
) -> dict[str, Any]:
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")

    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    detector = YOLOOBBDetector(params)
    profiler = TrackingProfiler(enabled=True)
    prefetcher = None

    if str(read_mode).strip().lower() == "prefetch":
        prefetcher = FramePrefetcher(
            capture,
            buffer_size=max(1, int(prefetch_buffer_size)),
        )
        prefetcher.start()

    read_ms: list[float] = []
    resize_ms: list[float] = []
    detector_ms: list[float] = []
    filter_ms: list[float] = []
    worker_detection_ms: list[float] = []
    yolo_phase_ms: list[float] = []
    yolo_model_execute_ms: list[float] = []
    yolo_extract_raw_ms: list[float] = []
    headtail_crop_ms: list[float] = []
    headtail_inference_ms: list[float] = []
    detections_per_frame: list[float] = []
    sample_frame_indices: list[int] = []

    total_needed = max(0, int(warmup_frames)) + max(0, int(measure_frames))
    processed_samples = 0
    measured_frames = 0
    current_frame_index = max(0, int(start_frame))
    batch_buffer: list[Any] = []
    batch_indices: list[int] = []

    try:
        while processed_samples < total_needed:
            read_started = time.perf_counter()
            if prefetcher is not None:
                ok, frame = _read_next_prefetched_frame(prefetcher, stride)
            else:
                ok, frame = _read_next_sampled_frame(capture, stride)
            read_elapsed_ms = (time.perf_counter() - read_started) * 1000.0
            if not ok:
                break

            resize_started = time.perf_counter()
            frame = _resize_frame(frame, resize_factor)
            resize_elapsed_ms = (time.perf_counter() - resize_started) * 1000.0

            batch_buffer.append(frame)
            batch_indices.append(current_frame_index)
            current_frame_index += max(1, int(stride))

            if len(batch_buffer) < max(1, int(batch_size)):
                continue

            is_measurement_batch = processed_samples >= int(warmup_frames)
            phase_before = _capture_phase_snapshot(profiler)
            detect_started = time.perf_counter()
            if len(batch_buffer) == 1:
                raw_output = detector.detect_objects(
                    batch_buffer[0],
                    batch_indices[0],
                    return_raw=True,
                    profiler=profiler,
                )
                batch_outputs = [raw_output]
            else:
                batch_outputs = detector.detect_objects_batched(
                    batch_buffer,
                    batch_indices[0],
                    return_raw=True,
                    profiler=profiler,
                )
            detect_elapsed_ms = (time.perf_counter() - detect_started) * 1000.0

            filter_started = time.perf_counter()
            batch_detection_counts: list[int] = []
            for raw in batch_outputs:
                raw = normalize_raw_detector_output(raw)
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_heading_hints,
                    raw_heading_confidences,
                    raw_directed_mask,
                    _raw_canonical_affines,
                ) = raw
                filtered = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=None,
                    detection_ids=None,
                    heading_hints=raw_heading_hints,
                    heading_confidences=raw_heading_confidences,
                    directed_mask=raw_directed_mask,
                )
                batch_detection_counts.append(len(filtered[0]))
            filter_elapsed_ms = (time.perf_counter() - filter_started) * 1000.0

            batch_yolo_phase_ms = _phase_delta(
                profiler, phase_before, "yolo_obb_inference"
            )
            batch_yolo_model_execute_ms = _phase_delta(
                profiler, phase_before, "yolo_obb_model_execute"
            )
            batch_yolo_extract_raw_ms = _phase_delta(
                profiler, phase_before, "yolo_obb_extract_raw"
            )
            batch_headtail_crop_ms = _phase_delta(
                profiler, phase_before, "headtail_crop"
            )
            batch_headtail_inference_ms = _phase_delta(
                profiler, phase_before, "headtail_inference"
            )

            processed_samples += len(batch_buffer)
            if is_measurement_batch:
                per_frame_detect_ms = detect_elapsed_ms / float(len(batch_buffer))
                per_frame_filter_ms = filter_elapsed_ms / float(len(batch_buffer))
                per_frame_yolo_ms = batch_yolo_phase_ms / float(len(batch_buffer))
                per_frame_yolo_model_ms = batch_yolo_model_execute_ms / float(
                    len(batch_buffer)
                )
                per_frame_yolo_extract_ms = batch_yolo_extract_raw_ms / float(
                    len(batch_buffer)
                )
                per_frame_headtail_crop_ms = batch_headtail_crop_ms / float(
                    len(batch_buffer)
                )
                per_frame_headtail_inference_ms = batch_headtail_inference_ms / float(
                    len(batch_buffer)
                )
                for index_within_batch, frame_index in enumerate(batch_indices):
                    if measured_frames >= int(measure_frames):
                        break
                    read_ms.append(read_elapsed_ms / float(len(batch_buffer)))
                    resize_ms.append(resize_elapsed_ms / float(len(batch_buffer)))
                    detector_ms.append(per_frame_detect_ms)
                    filter_ms.append(per_frame_filter_ms)
                    worker_detection_ms.append(
                        per_frame_detect_ms + per_frame_filter_ms
                    )
                    yolo_phase_ms.append(per_frame_yolo_ms)
                    yolo_model_execute_ms.append(per_frame_yolo_model_ms)
                    yolo_extract_raw_ms.append(per_frame_yolo_extract_ms)
                    headtail_crop_ms.append(per_frame_headtail_crop_ms)
                    headtail_inference_ms.append(per_frame_headtail_inference_ms)
                    detections_per_frame.append(
                        float(batch_detection_counts[index_within_batch])
                    )
                    sample_frame_indices.append(int(frame_index))
                    measured_frames += 1

            batch_buffer = []
            batch_indices = []

            if measured_frames >= int(measure_frames):
                break
    finally:
        if prefetcher is not None:
            prefetcher.stop()
        capture.release()
        closer = getattr(detector, "close", None)
        if callable(closer):
            closer()

    return {
        "video_path": str(video),
        "sampled_frame_indices": sample_frame_indices,
        "measured_frames": measured_frames,
        "batch_size": max(1, int(batch_size)),
        "resize_factor": float(resize_factor),
        "read_mode": str(read_mode).strip().lower(),
        "prefetch_buffer_size": max(1, int(prefetch_buffer_size)),
        "params": params,
        "metrics": {
            "frame_read": summarize_series(read_ms),
            "resize": summarize_series(resize_ms),
            "detector_call": summarize_series(detector_ms),
            "filter_raw_detections": summarize_series(filter_ms),
            "worker_style_detection_total": summarize_series(worker_detection_ms),
            "yolo_obb_inference_phase": summarize_series(yolo_phase_ms),
            "yolo_obb_model_execute_phase": summarize_series(yolo_model_execute_ms),
            "yolo_obb_extract_raw_phase": summarize_series(yolo_extract_raw_ms),
            "headtail_crop_phase": summarize_series(headtail_crop_ms),
            "headtail_inference_phase": summarize_series(headtail_inference_ms),
            "detections_per_frame": summarize_series(detections_per_frame),
        },
    }


def format_summary(summary: dict[str, Any]) -> str:
    metric_order = [
        "frame_read",
        "resize",
        "detector_call",
        "filter_raw_detections",
        "worker_style_detection_total",
        "yolo_obb_inference_phase",
        "yolo_obb_model_execute_phase",
        "yolo_obb_extract_raw_phase",
        "headtail_crop_phase",
        "headtail_inference_phase",
        "detections_per_frame",
    ]
    lines = [
        f"video: {summary['video_path']}",
        f"measured_frames: {summary['measured_frames']}",
        f"batch_size: {summary['batch_size']}",
        f"resize_factor: {summary['resize_factor']:.4f}",
        f"read_mode: {summary.get('read_mode', 'direct')}",
    ]
    metrics = summary.get("metrics", {})
    for metric_name in metric_order:
        metric = metrics.get(metric_name, {})
        if not metric:
            continue
        lines.append(
            "{name}: mean={mean:.2f} ms median={median:.2f} ms p95={p95:.2f} ms min={minv:.2f} ms max={maxv:.2f} ms n={count}".format(
                name=metric_name,
                mean=float(metric.get("mean_ms", 0.0)),
                median=float(metric.get("median_ms", 0.0)),
                p95=float(metric.get("p95_ms", 0.0)),
                minv=float(metric.get("min_ms", 0.0)),
                maxv=float(metric.get("max_ms", 0.0)),
                count=int(metric.get("count", 0)),
            )
        )
    return "\n".join(lines)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    params = build_detector_params(
        model_path=args.model,
        runtime=args.runtime,
        batch_size=args.batch_size,
        max_targets=args.max_targets,
        tensorrt_max_batch_size=args.tensorrt_max_batch_size,
        tensorrt_build_batch_size=args.tensorrt_build_batch_size,
        tensorrt_workspace_gb=args.tensorrt_workspace_gb,
        imgsz=args.imgsz,
        headtail_model_path=args.headtail_model,
        headtail_runtime=args.headtail_runtime,
        headtail_batch_size=args.headtail_batch_size,
        tracking_realtime_mode=args.tracking_realtime_mode,
    )
    summary = benchmark_video(
        video_path=args.video,
        params=params,
        resize_factor=args.resize_factor,
        batch_size=args.batch_size,
        start_frame=args.start_frame,
        stride=args.stride,
        warmup_frames=args.warmup_frames,
        measure_frames=args.measure_frames,
        read_mode=args.read_mode,
        prefetch_buffer_size=args.prefetch_buffer_size,
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run_from_args(args)
    print(format_summary(summary))
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
