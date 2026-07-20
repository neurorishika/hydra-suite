"""Shared tracking-cache identity and path resolution helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hydra_suite.trackerkit.gui.model_utils import resolve_model_path
from hydra_suite.utils.video_artifacts import (
    build_detection_cache_path,
    candidate_artifact_base_dirs,
    choose_writable_artifact_base_dir,
    find_existing_detection_cache_path,
)

logger = logging.getLogger(__name__)


def _normalize_float_for_cache(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return float(value)


@dataclass(frozen=True)
class TrackingCachePlan:
    """Resolved detection cache path and model IDs for a tracking run."""

    inference_model_id: str
    engine_model_id: str | None
    detection_cache_path: str


def resolve_detection_cache_runtime(params: dict) -> str:
    """Stable detection-cache runtime string derived from ``RUNTIME_TIER``.

    The detection-cache identity historically hashed the ``COMPUTE_RUNTIME``
    param, which the GUI set to ``resolve_compute_runtime(tier, platform, "obb")``.
    After the COMPUTE_RUNTIME param family was retired (Runtime Gen-2 FT1) this
    reproduces the SAME string directly from the live ``RUNTIME_TIER`` param so
    existing detection caches stay valid. Does not call ``resolve_compute_runtime``
    (deleted in a later slice).
    """
    from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform

    tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
    if tier not in {"cpu", "gpu", "gpu_fast"}:
        tier = "cpu"
    resolved = RuntimeResolver(tier, detect_platform()).resolve("obb")
    if resolved.backend == "tensorrt":
        return "tensorrt"
    if resolved.backend == "coreml":
        return "coreml"
    return resolved.device  # cpu / cuda / mps


def normalize_tracking_cache_value(value: object):
    """Convert values to deterministic, JSON-safe forms for cache hashing."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "type": "ndarray",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "digest": hashlib.md5(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _normalize_float_for_cache(float(value))
    if isinstance(value, float):
        return _normalize_float_for_cache(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): normalize_tracking_cache_value(inner)
            for key, inner in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_tracking_cache_value(inner) for inner in value]
    return value


def get_tracking_model_fingerprint(model_path: object) -> dict[str, Any]:
    """Return a stable fingerprint for a configured model path."""
    configured = str(model_path or "")
    resolved = str(resolve_model_path(configured))
    fingerprint = {"configured_path": configured, "resolved_path": resolved}
    if resolved and os.path.exists(resolved):
        try:
            stat = os.stat(resolved)
            fingerprint["size_bytes"] = stat.st_size
            fingerprint["mtime_ns"] = stat.st_mtime_ns
        except OSError:
            fingerprint["size_bytes"] = None
            fingerprint["mtime_ns"] = None
    else:
        fingerprint["size_bytes"] = None
        fingerprint["mtime_ns"] = None
    return fingerprint


def get_tracking_cache_model_ids(
    params: dict[str, Any], detection_method: str
) -> dict[str, str | None]:
    """Generate raw-detection and TensorRT-engine cache identity keys."""
    resize_factor = params.get("RESIZE_FACTOR", 1.0)
    resize_str = f"r{int(resize_factor * 100)}"
    _compute_runtime = resolve_detection_cache_runtime(params)

    def _extract(keys):
        return {
            key: normalize_tracking_cache_value(
                _compute_runtime if key == "COMPUTE_RUNTIME" else params.get(key)
            )
            for key in keys
        }

    def _build_id(prefix, cache_params, model_stem=""):
        digest = hashlib.md5(
            json.dumps(cache_params, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        if model_stem:
            return f"{prefix}_{model_stem}_{resize_str}_{digest}"
        return f"{prefix}_{resize_str}_{digest}"

    common_detection_keys = (
        "DETECTION_METHOD",
        "RESIZE_FACTOR",
        "MAX_TARGETS",
        "COMPUTE_RUNTIME",
    )

    if detection_method == "yolo_obb":
        yolo_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        direct_model = params.get(
            "YOLO_OBB_DIRECT_MODEL_PATH",
            params.get("YOLO_MODEL_PATH", "best.pt"),
        )
        crop_obb_model = params.get(
            "YOLO_CROP_OBB_MODEL_PATH", params.get("YOLO_MODEL_PATH", "best.pt")
        )
        active_obb_model = direct_model if yolo_mode == "direct" else crop_obb_model
        model_fingerprint = get_tracking_model_fingerprint(active_obb_model)
        model_name = os.path.basename(
            model_fingerprint["resolved_path"] or model_fingerprint["configured_path"]
        )
        model_stem = os.path.splitext(model_name)[0] or "model"
        safe_model_stem = "".join(
            char if char.isalnum() or char in ("_", "-") else "_" for char in model_stem
        )

        yolo_inference_keys = (
            "YOLO_TARGET_CLASSES",
            "YOLO_DEVICE",
            "ENABLE_TENSORRT",
            "TENSORRT_MAX_BATCH_SIZE",
            "YOLO_OBB_MODE",
            "YOLO_SEQ_CROP_PAD_RATIO",
            "YOLO_SEQ_MIN_CROP_SIZE_PX",
            "YOLO_SEQ_ENFORCE_SQUARE_CROP",
            "YOLO_SEQ_STAGE2_IMGSZ",
            "YOLO_SEQ_INDIVIDUAL_BATCH_SIZE",
            "YOLO_SEQ_STAGE2_POW2_PAD",
            "YOLO_HEADTAIL_CONF_THRESHOLD",
            "POSE_OVERRIDES_HEADTAIL",
        )
        cache_params = {
            "common": _extract(common_detection_keys),
            "yolo": _extract(yolo_inference_keys),
            "models": normalize_tracking_cache_value(
                {
                    "active_obb": model_fingerprint,
                    "direct_obb": get_tracking_model_fingerprint(direct_model),
                    "detect": get_tracking_model_fingerprint(
                        params.get("YOLO_DETECT_MODEL_PATH", "")
                    ),
                    "crop_obb": get_tracking_model_fingerprint(crop_obb_model),
                    "headtail": get_tracking_model_fingerprint(
                        params.get("YOLO_HEADTAIL_MODEL_PATH", "")
                    ),
                }
            ),
            "raw_detection_cache_version": 4,
        }
        classes = cache_params["yolo"].get("YOLO_TARGET_CLASSES")
        if classes is not None:
            if isinstance(classes, str):
                raw_classes = [
                    item.strip() for item in classes.split(",") if item.strip()
                ]
            elif isinstance(classes, (list, tuple)):
                raw_classes = list(classes)
            else:
                raw_classes = [classes]
            try:
                cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                    int(item) for item in raw_classes
                )
            except (TypeError, ValueError):
                cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                    str(item) for item in raw_classes
                )

        build_batch_size = params.get(
            "TENSORRT_BUILD_BATCH_SIZE",
            params.get("TENSORRT_MAX_BATCH_SIZE", 1),
        )
        try:
            build_batch_size = max(1, int(build_batch_size or 1))
        except (TypeError, ValueError):
            build_batch_size = max(
                1, int(params.get("TENSORRT_MAX_BATCH_SIZE", 1) or 1)
            )
        try:
            build_workspace_gb = float(params.get("TENSORRT_BUILD_WORKSPACE_GB", 4.0))
        except (TypeError, ValueError):
            build_workspace_gb = 4.0

        engine_cache_params = {
            "engine": {
                "runtime": "tensorrt",
                "device": normalize_tracking_cache_value(params.get("YOLO_DEVICE")),
                "build_batch_size": build_batch_size,
                "workspace_gb": round(max(0.5, build_workspace_gb), 3),
                "active_obb": model_fingerprint,
                "export_profile": "trt_fp16_static_v1",
            },
            "engine_cache_version": 1,
        }
        return {
            "inference": _build_id("yolo", cache_params, model_stem=safe_model_stem),
            "engine": _build_id(
                "yolo_engine", engine_cache_params, model_stem=safe_model_stem
            ),
        }

    bg_detection_keys = (
        "MAX_CONTOUR_MULTIPLIER",
        "ENABLE_SIZE_FILTERING",
        "MIN_OBJECT_SIZE",
        "MAX_OBJECT_SIZE",
        "ROI_MASK",
        "BACKGROUND_PRIME_FRAMES",
        "ENABLE_ADAPTIVE_BACKGROUND",
        "BACKGROUND_LEARNING_RATE",
        "ENABLE_GPU_BACKGROUND",
        "GPU_DEVICE_ID",
        "THRESHOLD_VALUE",
        "MORPH_KERNEL_SIZE",
        "ENABLE_ADDITIONAL_DILATION",
        "DILATION_ITERATIONS",
        "DILATION_KERNEL_SIZE",
        "BRIGHTNESS",
        "CONTRAST",
        "GAMMA",
        "DARK_ON_LIGHT_BACKGROUND",
        "ENABLE_LIGHTING_STABILIZATION",
        "LIGHTING_SMOOTH_FACTOR",
        "LIGHTING_MEDIAN_WINDOW",
        "ENABLE_CONSERVATIVE_SPLIT",
        "CONSERVATIVE_KERNEL_SIZE",
        "CONSERVATIVE_ERODE_ITER",
        "MIN_CONTOUR_AREA",
        "MIN_DETECTIONS_TO_START",
        "MIN_DETECTION_COUNTS",
    )
    cache_params = {
        "common": _extract(common_detection_keys),
        "background_subtraction": _extract(bg_detection_keys),
    }
    return {
        "inference": _build_id("bgsub", cache_params),
        "engine": None,
    }


def plan_tracking_cache(
    video_path: str,
    *,
    params: dict[str, Any],
    preferred_output_dir: str | None = None,
    use_cached_detections: bool,
) -> TrackingCachePlan:
    """Resolve the detection-cache path and model IDs for a run.

    When ``use_cached_detections`` is false, existing caches are ignored and a fresh
    canonical output path is returned, matching the GUI's checkbox semantics.
    """
    detection_method = params.get("DETECTION_METHOD", "background_subtraction")
    cache_ids = get_tracking_cache_model_ids(params, detection_method)
    model_id = cache_ids["inference"]
    csv_dir = str(preferred_output_dir or "").strip()
    artifact_base_dirs = candidate_artifact_base_dirs(
        video_path,
        preferred_base_dirs=[csv_dir],
    )
    artifact_base_dir = choose_writable_artifact_base_dir(
        video_path,
        preferred_base_dirs=[csv_dir],
    )
    if artifact_base_dir != Path(video_path).parent:
        logger.warning(
            "Video directory not writable; using artifact root: %s",
            artifact_base_dir,
        )

    detection_cache_path = None
    if use_cached_detections:
        existing_detection_cache = find_existing_detection_cache_path(
            video_path,
            model_id,
            artifact_base_dirs=artifact_base_dirs,
        )
        if existing_detection_cache is not None:
            detection_cache_path = str(existing_detection_cache)

    if detection_cache_path is None:
        detection_cache_path = str(
            build_detection_cache_path(
                video_path,
                model_id,
                artifact_base_dir=artifact_base_dir,
            )
        )

    return TrackingCachePlan(
        inference_model_id=model_id,
        engine_model_id=cache_ids.get("engine"),
        detection_cache_path=detection_cache_path,
    )
