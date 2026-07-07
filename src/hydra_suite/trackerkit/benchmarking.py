"""TrackerKit benchmarking utilities for runtime and batch recommendations."""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from hydra_suite.core.canonicalization.crop import compute_crop_dimensions
from hydra_suite.core.identity.properties.cache import _file_fingerprint, _hash_payload
from hydra_suite.paths import get_config_dir
from hydra_suite.runtime.compute_runtime import (
    _normalize_runtime,
    derive_pose_runtime_settings,
    runtime_label,
)
from hydra_suite.runtime.resolver import (
    available_tiers,
    detect_platform,
    resolve_compute_runtime,
    tier_label,
)
from hydra_suite.trackerkit.gui.model_utils import (
    resolve_model_path,
    resolve_pose_model_path,
)
from hydra_suite.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)

_CACHE_VERSION = "1.1"
_RECOMMENDATION_MARGIN = 0.05


def _emit_benchmark_message(
    message: str,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    logger.info("TrackerKit benchmark: %s", message)
    print(f"[TrackerKit Benchmark] {message}", flush=True)
    if callable(status_callback):
        status_callback(message)


class _PoseBenchmarkBackendCache:
    """Cache reusable pose backends during a single benchmark session."""

    def __init__(self) -> None:
        self._backends: dict[tuple[Any, ...], Any] = {}

    def get(self, key: tuple[Any, ...]) -> Any | None:
        return self._backends.get(key)

    def put(self, key: tuple[Any, ...], backend: Any) -> None:
        self._backends[key] = backend

    def close(self) -> None:
        for backend in self._backends.values():
            closer = getattr(backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.debug(
                        "Failed to close cached pose benchmark backend.",
                        exc_info=True,
                    )
        self._backends.clear()


def _can_reuse_pose_backend(backend_family: str, tier: str, platform: Any) -> bool:
    family = str(backend_family or "yolo").strip().lower()
    rt = str(tier or "cpu").strip().lower()
    if family == "yolo":
        # `tier` is always "cpu"/"gpu"/"gpu_fast" here, never the canonical
        # runtime string "tensorrt" -- YOLO pose never has an export artifact
        # (see stages/pose.py::load_pose_model), so gpu_fast is always just
        # the native backend on a different device. Always reusable.
        return True
    if family == "sleap":
        # cpu/gpu tiers always resolve to the native SLEAP service backend.
        # gpu_fast is the one case that can differ: on CUDA hardware it
        # resolves to an exported (tensorrt/onnx) backend (not reusable);
        # on Apple hardware SLEAP has no CoreML path, so gpu_fast still
        # resolves to the native/mps backend (reusable).
        return rt != "gpu_fast" or not platform.has_cuda
    return False


def _pose_backend_cache_key(
    backend_family: str,
    tier: str,
    model_path: str,
    crop_size: int,
    platform: Any,
    keypoint_names: list[str] | None = None,
    sleap_env: str = "sleap",
) -> tuple[Any, ...] | None:
    if not _can_reuse_pose_backend(backend_family, tier, platform):
        return None
    family = str(backend_family or "yolo").strip().lower()
    return (
        family,
        str(tier or "cpu").strip().lower(),
        str(model_path),
        int(crop_size),
        tuple(str(name) for name in (keypoint_names or [])),
        str(sleap_env or "sleap").strip().lower(),
    )


def _update_pose_backend_batch_size(backend: Any, batch_size: int) -> None:
    if hasattr(backend, "batch_size"):
        backend.batch_size = max(1, int(batch_size))
    if hasattr(backend, "sleap_batch"):
        backend.sleap_batch = max(1, int(batch_size))


@dataclass
class BenchmarkGeometry:
    """Effective benchmark geometry derived from the current TrackerKit session."""

    frame_width: int
    frame_height: int
    resize_factor: float
    effective_frame_width: int
    effective_frame_height: int
    reference_body_size: float
    reference_aspect_ratio: float
    padding_fraction: float
    canonical_crop_width: int
    canonical_crop_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result for a single target/runtime/batch benchmark configuration."""

    model_type: str
    model_path: str
    runtime: str
    runtime_label: str
    batch_size: int
    input_shape: tuple[int, int]
    warmup_iters: int
    bench_iters: int
    resolved_backend: str = (
        ""  # "torch"/"tensorrt"/"coreml" — what RuntimeResolver picked
    )
    individual_batch_size: int | None = None
    latencies_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    mean_per_frame_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_fps: float = 0.0
    ram_peak_mb: float | None = None
    ram_delta_mb: float | None = None
    accelerator_peak_mb: float | None = None
    accelerator_delta_mb: float | None = None
    success: bool = True
    error: str = ""

    def compute_stats(self) -> None:
        if not self.latencies_ms:
            return
        self.mean_ms = float(np.mean(self.latencies_ms))
        self.median_ms = float(np.median(self.latencies_ms))
        self.std_ms = float(np.std(self.latencies_ms))
        self.min_ms = float(min(self.latencies_ms))
        self.max_ms = float(max(self.latencies_ms))
        sorted_lat = sorted(self.latencies_ms)
        idx_95 = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
        idx_99 = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
        self.p95_ms = float(sorted_lat[idx_95])
        self.p99_ms = float(sorted_lat[idx_99])
        if self.batch_size > 0:
            self.mean_per_frame_ms = self.mean_ms / float(self.batch_size)
        if self.mean_ms > 0:
            self.throughput_fps = (self.batch_size / self.mean_ms) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        payload = dict(data)
        payload["input_shape"] = tuple(payload.get("input_shape", (0, 0)))
        payload["latencies_ms"] = list(payload.get("latencies_ms", []))
        return cls(**payload)


@dataclass
class BenchmarkRecommendation:
    """Recommended runtime/batch choice for one benchmark target."""

    target_key: str
    target_label: str
    runtime: str
    runtime_label: str
    batch_size: int
    mean_ms: float
    throughput_fps: float
    reason: str
    model_path: str
    individual_batch_size: int | None = None
    mean_per_frame_ms: float = 0.0
    ram_peak_mb: float | None = None
    accelerator_peak_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecommendation":
        return cls(**dict(data))


@dataclass
class BenchmarkTargetSpec:
    """Configurable benchmark target collected from the current GUI state."""

    key: str
    label: str
    pipeline: str
    model_path: str
    runtimes: list[str]
    batch_sizes: list[int]
    individual_batch_sizes: list[int] | None = None
    current_runtime: str = "cpu"
    current_batch_size: int = 1
    current_individual_batch_size: int | None = None
    backend_family: str = ""
    extra_model_paths: list[str] = field(default_factory=list)
    supports_batch_apply: bool = True
    benchmark_context: dict[str, Any] = field(default_factory=dict)

    def cache_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "pipeline": self.pipeline,
            "backend_family": self.backend_family,
            "model": _file_fingerprint(self.model_path),
            "extra_models": [
                _file_fingerprint(path) for path in self.extra_model_paths
            ],
            "benchmark_context": dict(self.benchmark_context),
        }


def build_benchmark_geometry_from_dimensions(
    frame_width: int,
    frame_height: int,
    resize_factor: float,
    reference_body_size: float,
    reference_aspect_ratio: float,
    padding_fraction: float,
) -> BenchmarkGeometry:
    """Build effective frame and crop geometry from raw dimensions and UI values."""
    safe_resize = max(0.1, float(resize_factor))
    effective_frame_width = max(8, int(round(float(frame_width) * safe_resize)))
    effective_frame_height = max(8, int(round(float(frame_height) * safe_resize)))
    canonical_long_edge = max(
        8,
        int(
            round(
                float(reference_body_size) * (1.0 + max(0.0, float(padding_fraction)))
            )
        ),
    )
    crop_width, crop_height = compute_crop_dimensions(
        canonical_long_edge,
        max(1.0, float(reference_aspect_ratio)),
    )
    return BenchmarkGeometry(
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        resize_factor=safe_resize,
        effective_frame_width=effective_frame_width,
        effective_frame_height=effective_frame_height,
        reference_body_size=float(reference_body_size),
        reference_aspect_ratio=float(reference_aspect_ratio),
        padding_fraction=float(padding_fraction),
        canonical_crop_width=int(crop_width),
        canonical_crop_height=int(crop_height),
    )


def read_video_frame_size(video_path: str) -> tuple[int, int]:
    """Read frame width/height for a video path."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        frame_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Could not read video geometry: {video_path}")
    return frame_width, frame_height


def derive_benchmark_geometry_from_video(
    video_path: str,
    resize_factor: float,
    reference_body_size: float,
    reference_aspect_ratio: float,
    padding_fraction: float,
) -> BenchmarkGeometry:
    """Build benchmark geometry from a loaded video path and current UI settings."""
    frame_width, frame_height = read_video_frame_size(video_path)
    return build_benchmark_geometry_from_dimensions(
        frame_width=frame_width,
        frame_height=frame_height,
        resize_factor=resize_factor,
        reference_body_size=reference_body_size,
        reference_aspect_ratio=reference_aspect_ratio,
        padding_fraction=padding_fraction,
    )


def _trackerkit_cache_dir() -> Path:
    path = get_config_dir() / "trackerkit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def benchmark_cache_path() -> Path:
    """Return the persistent TrackerKit benchmark cache path."""
    return _trackerkit_cache_dir() / "benchmark_recommendations.json"


def build_hardware_fingerprint() -> str:
    """Build a deterministic fingerprint for the current hardware/runtime environment."""
    info = get_device_info() or {}
    payload = {
        "cuda_available": bool(info.get("cuda_available")),
        "mps_available": bool(info.get("mps_available")),
        "torch_cuda_available": bool(info.get("torch_cuda_available")),
        "onnxruntime_providers": list(info.get("onnxruntime_providers", [])),
        "cuda_device_name": info.get("torch_cuda_device_name"),
        "backend": info.get("backend"),
        "torch_version": info.get("torch_version"),
        "onnxruntime_version": info.get("onnxruntime_version"),
        "tensorrt_version": info.get("tensorrt_version"),
    }
    return _hash_payload(payload)


def _cache_key(
    target: BenchmarkTargetSpec,
    geometry: BenchmarkGeometry,
    *,
    realtime_enabled: bool,
) -> str:
    payload = {
        "schema_version": _CACHE_VERSION,
        "hardware": build_hardware_fingerprint(),
        "target": target.cache_payload(),
        "geometry": geometry.to_dict(),
        "realtime_enabled": bool(realtime_enabled),
    }
    return _hash_payload(payload)


def _load_cache() -> dict[str, Any]:
    path = benchmark_cache_path()
    if not path.exists():
        return {"version": _CACHE_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load TrackerKit benchmark cache", exc_info=True)
        return {"version": _CACHE_VERSION, "entries": {}}
    if not isinstance(data, dict):
        return {"version": _CACHE_VERSION, "entries": {}}
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    return data


def _save_cache(cache: dict[str, Any]) -> None:
    path = benchmark_cache_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def choose_recommendation(
    target: BenchmarkTargetSpec,
    results: list[BenchmarkResult],
) -> BenchmarkRecommendation | None:
    """Choose a stable recommendation from successful benchmark results."""
    successful = [
        result
        for result in results
        if result.success and result.mean_ms > 0 and result.mean_per_frame_ms > 0
    ]
    if not successful:
        return None
    best_per_frame_ms = min(result.mean_per_frame_ms for result in successful)
    margin = best_per_frame_ms * (1.0 + _RECOMMENDATION_MARGIN)
    candidates = [result for result in successful if result.mean_per_frame_ms <= margin]
    candidates.sort(
        key=lambda result: (
            result.mean_per_frame_ms,
            -result.throughput_fps,
            result.batch_size,
            result.individual_batch_size or result.batch_size,
            result.mean_ms,
        )
    )
    selected = candidates[0]
    reason = (
        "Best per-frame latency within 5% of the optimum; higher throughput preferred, then smaller batch as a tie-breaker."
        if len(candidates) > 1
        else "Best observed per-frame latency for the current model and geometry."
    )
    return BenchmarkRecommendation(
        target_key=target.key,
        target_label=target.label,
        runtime=selected.runtime,
        runtime_label=selected.runtime_label,
        batch_size=int(selected.batch_size),
        individual_batch_size=(
            int(selected.individual_batch_size)
            if selected.individual_batch_size is not None
            else None
        ),
        mean_ms=float(selected.mean_ms),
        throughput_fps=float(selected.throughput_fps),
        reason=reason,
        model_path=str(target.model_path),
        mean_per_frame_ms=float(selected.mean_per_frame_ms),
        ram_peak_mb=selected.ram_peak_mb,
        accelerator_peak_mb=selected.accelerator_peak_mb,
    )


def _process_rss_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
    except Exception:
        return None
    if not output:
        return None
    return float(int(output)) / 1024.0


def _synchronize_runtime(runtime: str = "") -> None:
    """Synchronize whichever accelerator is genuinely present and busy.

    This intentionally does NOT branch on a runtime/tier string (e.g.
    "cuda", "gpu", "gpu_fast"). Those vocabularies drift independently of
    this module (tiers vs. canonical runtimes), and gating on them has
    previously caused this synchronization to be silently skipped when the
    caller passed a tier string this function didn't recognize. Instead we
    probe real hardware availability directly: if CUDA is available,
    synchronize CUDA; if MPS is available, synchronize MPS. Both checks are
    safe no-ops when the corresponding accelerator has no in-flight work,
    so there is no correctness cost to checking unconditionally.

    ``runtime`` is accepted only for logging/debugging context and is never
    used to decide which branch executes.
    """
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            synchronize = getattr(torch.mps, "synchronize", None)
            if callable(synchronize):
                synchronize()
    except Exception:
        logger.debug(
            "Runtime synchronization probe failed for %s.", runtime, exc_info=True
        )


def _sample_accelerator_memory_mb(runtime: str = "") -> float | None:
    """Sample accelerator memory usage from whichever hardware is present.

    Like ``_synchronize_runtime``, this checks real hardware availability
    (``torch.cuda.is_available()`` / ``torch.backends.mps.is_available()``)
    rather than branching on a runtime/tier string, so it stays correct
    regardless of what runtime-vocabulary changes happen elsewhere in the
    codebase. ``runtime`` is accepted only for logging/debugging context.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            allocated = float(torch.cuda.memory_allocated())
            reserved = float(torch.cuda.memory_reserved())
            return max(allocated, reserved) / (1024.0 * 1024.0)
        if torch.backends.mps.is_available():
            current = getattr(torch.mps, "current_allocated_memory", None)
            driver = getattr(torch.mps, "driver_allocated_memory", None)
            samples: list[float] = []
            if callable(current):
                samples.append(float(current()))
            if callable(driver):
                samples.append(float(driver()))
            if samples:
                return max(samples) / (1024.0 * 1024.0)
    except Exception:
        logger.debug("Accelerator memory probe failed for %s.", runtime, exc_info=True)
        return None
    return None


def _run_timed_benchmark_iterations(
    result: BenchmarkResult,
    iterations: int,
    run_once: Callable[[], None],
    iteration_callback: Callable[[int, int], None] | None = None,
) -> None:
    baseline_rss_mb = _process_rss_mb()
    baseline_accel_mb = _sample_accelerator_memory_mb(result.runtime)
    max_rss_mb = baseline_rss_mb
    max_accel_mb = baseline_accel_mb

    for iteration_index in range(iterations):
        if callable(iteration_callback):
            iteration_callback(iteration_index + 1, iterations)
        gc.disable()
        try:
            _synchronize_runtime(result.runtime)
            started = time.perf_counter()
            run_once()
            _synchronize_runtime(result.runtime)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        finally:
            gc.enable()
        result.latencies_ms.append(elapsed_ms)

        current_rss_mb = _process_rss_mb()
        if current_rss_mb is not None:
            max_rss_mb = (
                current_rss_mb
                if max_rss_mb is None
                else max(max_rss_mb, current_rss_mb)
            )

        current_accel_mb = _sample_accelerator_memory_mb(result.runtime)
        if current_accel_mb is not None:
            max_accel_mb = (
                current_accel_mb
                if max_accel_mb is None
                else max(max_accel_mb, current_accel_mb)
            )

    result.ram_peak_mb = max_rss_mb
    if baseline_rss_mb is not None and max_rss_mb is not None:
        result.ram_delta_mb = max(0.0, max_rss_mb - baseline_rss_mb)
    result.accelerator_peak_mb = max_accel_mb
    if baseline_accel_mb is not None and max_accel_mb is not None:
        result.accelerator_delta_mb = max(0.0, max_accel_mb - baseline_accel_mb)


def store_cached_results(
    target: BenchmarkTargetSpec,
    geometry: BenchmarkGeometry,
    results: list[BenchmarkResult],
    *,
    realtime_enabled: bool,
) -> BenchmarkRecommendation | None:
    """Persist benchmark results and their derived recommendation."""
    recommendation = choose_recommendation(target, results)
    cache = _load_cache()
    key = _cache_key(target, geometry, realtime_enabled=realtime_enabled)
    cache["entries"][key] = {
        "target_key": target.key,
        "target_label": target.label,
        "pipeline": target.pipeline,
        "results": [result.to_dict() for result in results],
        "recommendation": recommendation.to_dict() if recommendation else None,
        "updated_at": time.time(),
    }
    _save_cache(cache)
    return recommendation


def lookup_cached_recommendation(
    target: BenchmarkTargetSpec,
    geometry: BenchmarkGeometry,
    *,
    realtime_enabled: bool,
) -> BenchmarkRecommendation | None:
    """Return a cached recommendation matching the current target and geometry."""
    cache = _load_cache()
    key = _cache_key(target, geometry, realtime_enabled=realtime_enabled)
    entry = cache.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return None
    recommendation = entry.get("recommendation")
    if not isinstance(recommendation, dict):
        return None
    return BenchmarkRecommendation.from_dict(recommendation)


def _default_batch_sizes(current: int, maximum: int) -> list[int]:
    current = max(1, int(current or 1))
    maximum = max(1, int(maximum or current))
    candidates = [1, current]
    if current > 2:
        candidates.append(max(1, current // 2))
    candidates.append(min(maximum, max(2, current * 2)))
    ordered: list[int] = []
    for value in candidates:
        bounded = max(1, min(maximum, int(value)))
        if bounded not in ordered:
            ordered.append(bounded)
    return ordered


def _resolve_existing_model_path(model_path: object) -> str:
    resolved = str(resolve_model_path(model_path) or "").strip()
    if resolved and os.path.exists(resolved):
        return resolved
    return ""


def _resolve_existing_pose_model_path(model_path: object, backend: str) -> str:
    resolved = str(resolve_pose_model_path(model_path, backend=backend) or "").strip()
    if resolved and os.path.exists(resolved):
        return resolved
    return ""


def collect_active_targets(
    main_window: Any,
) -> tuple[list[BenchmarkTargetSpec], list[str]]:
    """Collect benchmarkable targets from the current TrackerKit GUI state."""
    targets: list[BenchmarkTargetSpec] = []
    notices: list[str] = []

    if not getattr(main_window, "_detection_panel", None) or not getattr(
        main_window, "_identity_panel", None
    ):
        return targets, notices

    _platform = detect_platform()
    current_tier = (
        str(main_window._setup_panel.combo_runtime_tier.currentData() or "cpu")
        if hasattr(main_window, "_setup_panel")
        and hasattr(main_window._setup_panel, "combo_runtime_tier")
        else "cpu"
    )
    detection_tiers = available_tiers(_platform)
    max_targets = max(1, int(main_window._setup_panel.spin_max_targets.value()))
    current_detection_runtime = current_tier
    current_detection_batch = int(
        main_window._detection_panel.spin_detection_batch_size.value()
    )
    seq_individual_widget = getattr(
        main_window._detection_panel,
        "spin_yolo_seq_individual_batch_size",
        None,
    )
    current_detection_individual_batch = max(
        1,
        (
            int(seq_individual_widget.value())
            if seq_individual_widget is not None
            else int(max_targets)
        ),
    )
    detection_mode_index = (
        main_window._detection_panel.combo_yolo_obb_mode.currentIndex()
    )
    if main_window._is_yolo_detection_mode():
        if detection_mode_index == 0:
            direct_model = _resolve_existing_model_path(
                main_window._get_selected_yolo_model_path() or ""
            )
            if direct_model:
                targets.append(
                    BenchmarkTargetSpec(
                        key="detection_direct",
                        label="Detection (Direct OBB)",
                        pipeline="obb",
                        model_path=direct_model,
                        runtimes=detection_tiers,
                        batch_sizes=_default_batch_sizes(current_detection_batch, 64),
                        current_runtime=current_detection_runtime,
                        current_batch_size=current_detection_batch,
                        benchmark_context={
                            "max_targets": int(max_targets),
                        },
                    )
                )
            elif str(main_window._get_selected_yolo_model_path() or "").strip():
                notices.append(
                    "Detection benchmark skipped because the selected direct OBB model could not be resolved."
                )
        else:
            detect_model = _resolve_existing_model_path(
                main_window._get_selected_yolo_detect_model_path() or ""
            )
            crop_model = _resolve_existing_model_path(
                main_window._get_selected_yolo_crop_obb_model_path() or ""
            )
            if detect_model and crop_model:
                targets.append(
                    BenchmarkTargetSpec(
                        key="detection_sequential",
                        label="Detection (Sequential)",
                        pipeline="sequential",
                        model_path=crop_model,
                        extra_model_paths=[detect_model],
                        runtimes=detection_tiers,
                        batch_sizes=_default_batch_sizes(current_detection_batch, 64),
                        individual_batch_sizes=_default_batch_sizes(
                            current_detection_individual_batch,
                            max(
                                64, current_detection_individual_batch, int(max_targets)
                            ),
                        ),
                        current_runtime=current_detection_runtime,
                        current_batch_size=current_detection_batch,
                        current_individual_batch_size=current_detection_individual_batch,
                        benchmark_context={
                            "max_targets": int(max_targets),
                            "yolo_seq_crop_pad_ratio": float(
                                main_window._detection_panel.spin_yolo_seq_crop_pad.value()
                            ),
                            "yolo_seq_min_crop_size_px": int(
                                main_window._detection_panel.spin_yolo_seq_min_crop_px.value()
                            ),
                            "yolo_seq_enforce_square_crop": bool(
                                main_window._detection_panel.chk_yolo_seq_square_crop.isChecked()
                            ),
                            "yolo_seq_stage2_imgsz": int(
                                main_window._detection_panel.spin_yolo_seq_stage2_imgsz.value()
                            ),
                            "yolo_seq_individual_batch_size": int(
                                current_detection_individual_batch
                            ),
                            "yolo_seq_stage2_pow2_pad": bool(
                                main_window._detection_panel.chk_yolo_seq_stage2_pow2_pad.isChecked()
                            ),
                            "yolo_seq_detect_conf_threshold": float(
                                main_window._detection_panel.spin_yolo_seq_detect_conf.value()
                            ),
                        },
                    )
                )
            elif (
                str(main_window._get_selected_yolo_detect_model_path() or "").strip()
                or str(
                    main_window._get_selected_yolo_crop_obb_model_path() or ""
                ).strip()
            ):
                notices.append(
                    "Sequential detection benchmark skipped because one or more selected YOLO models could not be resolved."
                )
    else:
        notices.append("Detection benchmarking currently supports YOLO OBB mode only.")

    headtail_enabled = True
    is_headtail_enabled = getattr(main_window, "_is_headtail_compute_enabled", None)
    if callable(is_headtail_enabled):
        headtail_enabled = bool(is_headtail_enabled())

    headtail_getter = getattr(
        main_window._identity_panel,
        "_get_selected_yolo_headtail_model_path",
        None,
    )
    raw_headtail_model = str(
        headtail_getter() if callable(headtail_getter) else ""
    ).strip()
    headtail_model = (
        _resolve_existing_model_path(raw_headtail_model) if headtail_enabled else ""
    )
    if headtail_model:
        headtail_tiers = available_tiers(_platform)
        headtail_batch_widget = getattr(
            main_window._identity_panel,
            "spin_headtail_batch",
            None,
        )
        current_headtail_batch = max(
            1,
            (
                int(headtail_batch_widget.value())
                if headtail_batch_widget is not None
                else min(max_targets, 4)
            ),
        )
        targets.append(
            BenchmarkTargetSpec(
                key="headtail",
                label="Head-tail Orientation",
                pipeline="headtail",
                model_path=headtail_model,
                runtimes=headtail_tiers,
                batch_sizes=_default_batch_sizes(
                    current_headtail_batch,
                    max(max_targets, current_headtail_batch * 2),
                ),
                current_runtime=current_tier,
                current_batch_size=current_headtail_batch,
            )
        )
    elif headtail_enabled and raw_headtail_model:
        notices.append(
            "Head-tail benchmark skipped because the selected orientation model could not be resolved."
        )

    if main_window._is_pose_inference_enabled():
        backend = main_window._current_pose_backend_key()
        if backend == "vitpose":
            notices.append("Pose benchmarking does not support ViTPose yet.")
        else:
            raw_pose_model_path = str(
                main_window._get_resolved_pose_model_dir(backend) or ""
            ).strip()
            pose_model_path = _resolve_existing_pose_model_path(
                raw_pose_model_path, backend
            )
            if pose_model_path:
                benchmark_context: dict[str, Any] = {}
                if backend == "sleap":
                    keypoint_names = list(
                        main_window._load_pose_skeleton_keypoint_names()
                    )
                    if not keypoint_names:
                        notices.append(
                            "SLEAP pose benchmarking requires a skeleton JSON with keypoint names."
                        )
                        pose_model_path = ""
                    else:
                        benchmark_context = {
                            "keypoint_names": keypoint_names,
                            "sleap_env": str(main_window._selected_pose_sleap_env()),
                        }
                if pose_model_path:
                    targets.append(
                        BenchmarkTargetSpec(
                            key=f"pose_{backend}",
                            label="Pose Extraction",
                            pipeline="pose",
                            model_path=pose_model_path,
                            runtimes=available_tiers(_platform),
                            batch_sizes=_default_batch_sizes(
                                int(
                                    main_window._identity_panel.spin_pose_batch.value()
                                ),
                                256,
                            ),
                            current_runtime=current_tier,
                            current_batch_size=int(
                                main_window._identity_panel.spin_pose_batch.value()
                            ),
                            backend_family=backend,
                            benchmark_context=benchmark_context,
                        )
                    )
            elif raw_pose_model_path:
                notices.append(
                    f"Pose benchmark skipped because the selected {backend.upper()} model could not be resolved."
                )

    cnn_runtime = current_tier
    for index, classifier in enumerate(
        main_window._identity_config().get("cnn_classifiers", []) or []
    ):
        raw_model_path = str(classifier.get("model_path") or "").strip()
        model_path = _resolve_existing_model_path(raw_model_path)
        if not model_path:
            if raw_model_path:
                notices.append(
                    f"CNN benchmark skipped for '{classifier.get('label') or Path(raw_model_path).stem}' because the model could not be resolved."
                )
            continue
        label = str(classifier.get("label") or Path(model_path).stem)
        batch_size = int(classifier.get("batch_size", 1) or 1)
        targets.append(
            BenchmarkTargetSpec(
                key=f"cnn_{index}",
                label=f"CNN: {label}",
                pipeline="classify",
                model_path=model_path,
                runtimes=available_tiers(_platform),
                batch_sizes=_default_batch_sizes(batch_size, 256),
                current_runtime=cnn_runtime,
                current_batch_size=batch_size,
            )
        )

    return targets, notices


def make_synthetic_frame(height: int = 640, width: int = 640) -> np.ndarray:
    """Generate a deterministic synthetic frame for detection benchmarks."""
    rng = np.random.default_rng(42)
    frame = rng.integers(100, 200, (height, width, 3), dtype=np.uint8)
    for _ in range(5):
        center_x = rng.integers(50, width - 50)
        center_y = rng.integers(50, height - 50)
        radius = rng.integers(10, 40)
        cv2.circle(frame, (int(center_x), int(center_y)), int(radius), (50, 50, 50), -1)
    return frame


def make_synthetic_crops(
    count: int = 16,
    height: int = 128,
    width: int = 128,
) -> list[np.ndarray]:
    """Generate synthetic crops for pose and classification benchmarks."""
    rng = np.random.default_rng(42)
    crops: list[np.ndarray] = []
    for _ in range(count):
        crop = rng.integers(80, 220, (height, width, 3), dtype=np.uint8)
        cv2.ellipse(
            crop,
            (width // 2, height // 2),
            (width // 4, height // 6),
            int(rng.integers(0, 180)),
            0,
            360,
            (40, 40, 40),
            -1,
        )
        crops.append(crop)
    return crops


def make_synthetic_obb_corners(
    count: int,
    frame_height: int,
    frame_width: int,
    crop_size: int = 160,
) -> list[np.ndarray]:
    """Generate simple OBB corners for head-tail benchmarks."""
    if count <= 0:
        return []
    corners_list: list[np.ndarray] = []
    step_x = max(1, frame_width // (count + 1))
    box_w = float(max(24, min(crop_size, frame_width // 4)))
    box_h = float(max(16, min(max(24, crop_size // 2), frame_height // 3)))
    for idx in range(count):
        center_x = float(min(frame_width - 1, max(0, step_x * (idx + 1))))
        center_y = float(frame_height * (0.35 + 0.3 * ((idx % 2) == 0)))
        angle = float((idx * 23) % 180)
        rect = ((center_x, center_y), (box_w, box_h), angle)
        box = cv2.boxPoints(rect).astype(np.float32)
        box[:, 0] = np.clip(box[:, 0], 0, frame_width - 1)
        box[:, 1] = np.clip(box[:, 1], 0, frame_height - 1)
        corners_list.append(box)
    return corners_list


def bench_obb(
    model_path: str,
    tier: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    frame_size: tuple[int, int],
    *,
    max_targets: int = 25,
) -> BenchmarkResult:
    """Benchmark direct OBB inference for a runtime tier via the production executor."""
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor
    from hydra_suite.runtime.resolver import RuntimeResolver

    platform = detect_platform()
    resolved = RuntimeResolver(tier, platform).resolve("obb")
    compute_runtime = resolve_compute_runtime(tier, platform, stage="obb")
    result = BenchmarkResult(
        model_type="obb",
        model_path=model_path,
        runtime=tier,
        runtime_label=tier_label(tier, platform),
        resolved_backend=resolved.backend,
        batch_size=batch_size,
        input_shape=frame_size,
        warmup_iters=warmup,
        bench_iters=iterations,
    )
    try:
        executor = load_obb_executor(
            model_path,
            compute_runtime,
            max_det=max(20, max_targets),
            batch_size=batch_size,
        )
        frames = [make_synthetic_frame(*frame_size) for _ in range(batch_size)]

        def _run_once() -> None:
            executor.predict(frames, conf=1e-3, iou=1.0, verbose=False)

        for _ in range(warmup):
            _run_once()
        _run_timed_benchmark_iterations(result, iterations, _run_once)
        result.compute_stats()
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Direct OBB benchmark failed [%s]: %s", tier, exc)
    return result


def bench_sequential(
    detect_model_path: str,
    crop_obb_model_path: str,
    tier: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    individual_batch_size: int,
    frame_size: tuple[int, int],
    crop_size: int,
    *,
    max_targets: int = 25,
) -> BenchmarkResult:
    """Benchmark sequential detect-plus-crop OBB inference via production executors."""
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor
    from hydra_suite.runtime.resolver import RuntimeResolver

    platform = detect_platform()
    resolved = RuntimeResolver(tier, platform).resolve("obb")
    compute_runtime = resolve_compute_runtime(tier, platform, stage="obb")
    result = BenchmarkResult(
        model_type="sequential",
        model_path=f"{detect_model_path} | {crop_obb_model_path}",
        runtime=tier,
        runtime_label=tier_label(tier, platform),
        resolved_backend=resolved.backend,
        batch_size=batch_size,
        individual_batch_size=int(individual_batch_size),
        input_shape=frame_size,
        warmup_iters=warmup,
        bench_iters=iterations,
    )
    try:
        detect_executor = load_obb_executor(
            detect_model_path,
            compute_runtime,
            task="detect",
            batch_size=batch_size,
            max_det=max(20, max_targets),
        )
        crop_executor = load_obb_executor(
            crop_obb_model_path,
            compute_runtime,
            task="obb",
            batch_size=individual_batch_size,
            imgsz_override=crop_size,
            max_det=max(20, max_targets),
        )
        frames = [make_synthetic_frame(*frame_size) for _ in range(batch_size)]
        crops = make_synthetic_crops(individual_batch_size, crop_size, crop_size)

        def _run_once() -> None:
            detect_executor.predict(frames, conf=0.25, iou=0.7, verbose=False)
            crop_executor.predict(list(crops), conf=1e-3, iou=1.0, verbose=False)

        for _ in range(warmup):
            _run_once()
        _run_timed_benchmark_iterations(result, iterations, _run_once)
        result.compute_stats()
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Sequential benchmark failed [%s]: %s", tier, exc)
    return result


def bench_headtail(
    model_path: str,
    runtime: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    frame_size: tuple[int, int],
    crop_size: int,
) -> BenchmarkResult:
    """Benchmark detector-side head-tail analysis."""
    result = BenchmarkResult(
        model_type="headtail",
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=batch_size,
        input_shape=(crop_size, crop_size),
        warmup_iters=warmup,
        bench_iters=iterations,
    )
    try:
        # NOTE(task-5): `_make_detector_runtime_stub` was deleted in Task 5 of the
        # runtime-tier-benchmarking-cleanup plan; Task 7 rewrites bench_headtail onto
        # the production head-tail classifier path and removes this reference. Until
        # Task 7 lands, this call intentionally raises NameError, which is caught below
        # and surfaced as a benchmark failure rather than crashing the caller.
        detector = _make_detector_runtime_stub(  # noqa: F821
            runtime, model_path, batch_size=batch_size
        )
        detector.params["YOLO_HEADTAIL_MODEL_PATH"] = model_path
        detector._load_headtail_model(model_path)
        analyzer = detector._headtail_analyzer
        if analyzer is None or not analyzer.is_available:
            raise RuntimeError("Failed to load head-tail model")
        frame = make_synthetic_frame(*frame_size)
        obb_corners = make_synthetic_obb_corners(
            batch_size,
            frame_size[0],
            frame_size[1],
            crop_size,
        )
        for _ in range(warmup):
            detector._compute_headtail_hints(frame, obb_corners)

        def _run_once() -> None:
            detector._compute_headtail_hints(frame, obb_corners)

        _run_timed_benchmark_iterations(result, iterations, _run_once)
        result.compute_stats()
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Head-tail benchmark failed [%s]: %s", runtime, exc)
    return result


def bench_pose(
    model_path: str,
    tier: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    crop_size: int,
    *,
    crop_hw: tuple[int, int] | None = None,
    backend_family: str = "yolo",
    keypoint_names: list[str] | None = None,
    sleap_env: str = "sleap",
    status_callback: Callable[[str], None] | None = None,
    pose_backend_cache: _PoseBenchmarkBackendCache | None = None,
) -> BenchmarkResult:
    """Benchmark pose inference for the selected backend."""
    crop_height, crop_width = (
        (int(crop_hw[0]), int(crop_hw[1]))
        if isinstance(crop_hw, tuple) and len(crop_hw) >= 2
        else (int(crop_size), int(crop_size))
    )
    platform = detect_platform()
    pipeline = (
        "sleap_pose" if str(backend_family).strip().lower() == "sleap" else "yolo_pose"
    )
    result = BenchmarkResult(
        model_type="pose",
        model_path=model_path,
        runtime=tier,
        runtime_label=tier_label(tier, platform),
        batch_size=batch_size,
        input_shape=(crop_height, crop_width),
        warmup_iters=warmup,
        bench_iters=iterations,
    )
    try:
        _emit_benchmark_message(
            "Pose benchmark setup started. Export, backend initialization, and service startup are not included in the timed metrics.",
            status_callback,
        )
        cache_key = _pose_backend_cache_key(
            backend_family,
            tier,
            model_path,
            crop_size,
            platform,
            keypoint_names=keypoint_names,
            sleap_env=sleap_env,
        )
        backend = (
            pose_backend_cache.get(cache_key)
            if (pose_backend_cache and cache_key)
            else None
        )
        reused_backend = backend is not None
        if str(backend_family).strip().lower() == "sleap":
            from hydra_suite.core.identity.pose.api import (
                create_pose_backend_from_config,
            )
            from hydra_suite.core.identity.pose.types import PoseRuntimeConfig

            def _sleap_artifact_available() -> bool:
                # Mirrors core/inference/stages/pose.py's SLEAP tier->flavor gate:
                # gpu_fast on CUDA uses the exported TensorRT/ONNX path; on Apple,
                # SLEAP has no CoreML path and always uses native (mps) service.
                return platform.has_cuda

            sleap_compute_runtime = resolve_compute_runtime(
                tier,
                platform,
                stage=pipeline,
                artifact_available=_sleap_artifact_available,
            )
            derived = derive_pose_runtime_settings(
                sleap_compute_runtime, backend_family="sleap"
            )
            runtime_flavor = (
                str(derived.get("pose_runtime_flavor", "cpu")).strip().lower()
            )
            sleap_device = str(derived.get("pose_sleap_device", "cpu")).strip() or "cpu"
            if reused_backend:
                _emit_benchmark_message(
                    f"Reusing SLEAP pose backend ({runtime_flavor} on {sleap_device}) for batch {batch_size}.",
                    status_callback,
                )
                _update_pose_backend_batch_size(backend, batch_size)
            else:
                _emit_benchmark_message(
                    f"Creating SLEAP pose backend ({runtime_flavor} on {sleap_device}, batch {batch_size}).",
                    status_callback,
                )
                backend = create_pose_backend_from_config(
                    PoseRuntimeConfig(
                        backend_family="sleap",
                        runtime_flavor=runtime_flavor,
                        device=sleap_device,
                        batch_size=batch_size,
                        model_path=model_path,
                        out_root=str(_trackerkit_cache_dir() / "benchmark_pose"),
                        sleap_env=str(sleap_env or "sleap").strip() or "sleap",
                        sleap_device=sleap_device,
                        sleap_batch=batch_size,
                        sleap_max_instances=1,
                        keypoint_names=list(keypoint_names or []),
                    )
                )
                if pose_backend_cache is not None and cache_key is not None:
                    pose_backend_cache.put(cache_key, backend)
            crops = make_synthetic_crops(batch_size, crop_height, crop_width)
            warmup_fn = getattr(backend, "warmup", None)
            if callable(warmup_fn) and not reused_backend:
                try:
                    _emit_benchmark_message(
                        "Starting and warming the SLEAP pose service.",
                        status_callback,
                    )
                    warmup_fn()
                except Exception:
                    logger.debug("SLEAP benchmark warmup failed.", exc_info=True)
            for warmup_index in range(warmup):
                _emit_benchmark_message(
                    f"Pose benchmark warmup {warmup_index + 1}/{warmup}.",
                    status_callback,
                )
                backend.predict_batch(crops)

            def _run_once() -> None:
                backend.predict_batch(crops)

            _run_timed_benchmark_iterations(
                result,
                iterations,
                _run_once,
                iteration_callback=lambda current, total: _emit_benchmark_message(
                    f"Pose benchmark timed iteration {current}/{total}.",
                    status_callback,
                ),
            )
            result.compute_stats()
            consume_profile = getattr(backend, "consume_last_profile", None)
            if callable(consume_profile):
                profile = dict(consume_profile() or {})
                if profile:
                    _emit_benchmark_message(
                        "Last pose batch profile: "
                        f"transport={1000.0 * float(profile.get('pose_transport_s', 0.0) or 0.0):.1f} ms, "
                        f"inference={1000.0 * float(profile.get('pose_inference_s', 0.0) or 0.0):.1f} ms, "
                        f"postprocess={1000.0 * float(profile.get('pose_postprocess_s', 0.0) or 0.0):.1f} ms.",
                        status_callback,
                    )
            _emit_benchmark_message(
                f"Pose benchmark complete: mean batch {result.mean_ms:.2f} ms, mean per crop {result.mean_per_frame_ms:.2f} ms.",
                status_callback,
            )
            if pose_backend_cache is None or cache_key is None:
                closer = getattr(backend, "close", None)
                if callable(closer):
                    closer()
            return result

        from hydra_suite.core.identity.pose.backends.yolo import YoloNativeBackend

        compute_runtime = resolve_compute_runtime(
            tier, platform, stage="yolo_pose", artifact_available=lambda: False
        )
        device = (
            "cuda:0"
            if compute_runtime == "cuda"
            else ("mps" if compute_runtime in ("mps", "coreml") else "cpu")
        )
        if reused_backend:
            _emit_benchmark_message(
                f"Reusing YOLO pose backend (native on {device}) for batch {batch_size}.",
                status_callback,
            )
            _update_pose_backend_batch_size(backend, batch_size)
        else:
            _emit_benchmark_message(
                f"Creating YOLO pose backend (native on {device}, batch {batch_size}).",
                status_callback,
            )
            backend = YoloNativeBackend(
                model_path=model_path,
                device=device,
                batch_size=batch_size,
            )
            if pose_backend_cache is not None and cache_key is not None:
                pose_backend_cache.put(cache_key, backend)
        crops = make_synthetic_crops(batch_size, crop_height, crop_width)
        warmup_fn = getattr(backend, "warmup", None)
        if callable(warmup_fn) and not reused_backend:
            try:
                _emit_benchmark_message(
                    "Running pose backend warmup.",
                    status_callback,
                )
                warmup_fn()
            except Exception:
                logger.debug("YOLO pose benchmark warmup failed.", exc_info=True)
        for warmup_index in range(warmup):
            _emit_benchmark_message(
                f"Pose benchmark warmup {warmup_index + 1}/{warmup}.",
                status_callback,
            )
            backend.predict_batch(crops)

        def _run_once() -> None:
            backend.predict_batch(crops)

        _run_timed_benchmark_iterations(
            result,
            iterations,
            _run_once,
            iteration_callback=lambda current, total: _emit_benchmark_message(
                f"Pose benchmark timed iteration {current}/{total}.",
                status_callback,
            ),
        )
        result.compute_stats()
        _emit_benchmark_message(
            f"Pose benchmark complete: mean batch {result.mean_ms:.2f} ms, mean per crop {result.mean_per_frame_ms:.2f} ms.",
            status_callback,
        )
        if pose_backend_cache is None or cache_key is None:
            backend.close()
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Pose benchmark failed [%s]: %s", tier, exc)
    return result


def bench_classify(
    model_path: str,
    runtime: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    crop_size: int,
) -> BenchmarkResult:
    """Benchmark CNN identity classification inference."""
    result = BenchmarkResult(
        model_type="classify",
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=batch_size,
        input_shape=(crop_size, crop_size),
        warmup_iters=warmup,
        bench_iters=iterations,
    )
    try:
        from hydra_suite.core.identity.classification.cnn import (
            CNNIdentityBackend,
            CNNIdentityConfig,
        )

        config = CNNIdentityConfig(
            model_path=model_path,
            confidence=0.5,
            batch_size=batch_size,
        )
        backend = CNNIdentityBackend(
            config=config,
            model_path=model_path,
            compute_runtime=runtime,
        )
        crops = make_synthetic_crops(batch_size, crop_size, crop_size)
        for _ in range(warmup):
            backend.predict_batch(crops)

        def _run_once() -> None:
            backend.predict_batch(crops)

        _run_timed_benchmark_iterations(result, iterations, _run_once)
        result.compute_stats()
        backend.close()
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Classification benchmark failed [%s]: %s", runtime, exc)
    return result


def run_target_benchmark(
    target: BenchmarkTargetSpec,
    geometry: BenchmarkGeometry,
    runtime: str,
    batch_size: int,
    individual_batch_size: int | None = None,
    status_callback: Callable[[str], None] | None = None,
    pose_backend_cache: _PoseBenchmarkBackendCache | None = None,
    *,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """Run one benchmark target for a single runtime and batch size."""
    normalized_runtime = _normalize_runtime(runtime)
    frame_size = (geometry.effective_frame_height, geometry.effective_frame_width)
    crop_size = int(max(geometry.canonical_crop_width, geometry.canonical_crop_height))
    crop_hw = (
        int(geometry.canonical_crop_height),
        int(geometry.canonical_crop_width),
    )
    if target.pipeline == "obb":
        return bench_obb(
            target.model_path,
            runtime,
            warmup,
            iterations,
            batch_size,
            frame_size,
            max_targets=int(target.benchmark_context.get("max_targets", 25)),
        )
    if target.pipeline == "sequential":
        detect_model_path = (
            target.extra_model_paths[0] if target.extra_model_paths else ""
        )
        sequential_stage2_imgsz = int(
            target.benchmark_context.get("yolo_seq_stage2_imgsz", crop_size)
            or crop_size
        )
        return bench_sequential(
            detect_model_path,
            target.model_path,
            runtime,
            warmup,
            iterations,
            batch_size,
            int(
                individual_batch_size
                if individual_batch_size is not None
                else target.benchmark_context.get(
                    "yolo_seq_individual_batch_size",
                    target.benchmark_context.get("max_targets", 1),
                )
            ),
            frame_size,
            sequential_stage2_imgsz,
            max_targets=int(target.benchmark_context.get("max_targets", 25)),
        )
    if target.pipeline == "headtail":
        return bench_headtail(
            target.model_path,
            normalized_runtime,
            warmup,
            iterations,
            batch_size,
            frame_size,
            crop_size,
        )
    if target.pipeline == "pose":
        return bench_pose(
            target.model_path,
            runtime,
            warmup,
            iterations,
            batch_size,
            crop_size,
            crop_hw=crop_hw,
            backend_family=target.backend_family or "yolo",
            keypoint_names=list(target.benchmark_context.get("keypoint_names", [])),
            sleap_env=str(target.benchmark_context.get("sleap_env", "sleap")),
            status_callback=status_callback,
            pose_backend_cache=pose_backend_cache,
        )
    if target.pipeline == "classify":
        return bench_classify(
            target.model_path,
            normalized_runtime,
            warmup,
            iterations,
            batch_size,
            crop_size,
        )
    raise ValueError(f"Unsupported benchmark pipeline: {target.pipeline}")
