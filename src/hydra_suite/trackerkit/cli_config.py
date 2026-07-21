"""Pure TrackerKit CLI config/session helpers without MainWindow state."""

from __future__ import annotations

import json
import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from hydra_suite.runtime.resolver import (
    ResolvedBackend,
    RuntimeResolver,
    detect_platform,
)
from hydra_suite.trackerkit.gui.model_utils import resolve_model_path

logger = logging.getLogger(__name__)


def legacy_detection_runtime_fields(runtime: ResolvedBackend) -> dict:
    """Map a resolved backend to legacy detection config fields.

    Takes a ``ResolvedBackend`` (Runtime Gen-2 vocabulary) — the sole input
    since the legacy ``compute_runtime`` string path was retired (FT7b). Both
    the live GUI path and the CLI path resolve a ``RUNTIME_TIER`` to a
    ``ResolvedBackend`` and pass it here.

    These fields no longer drive any live detector construction: the
    ``"yolo_obb"`` detection method runs entirely through
    ``InferenceRunner``/``load_obb_executor``, keyed off ``RUNTIME_TIER``,
    which never reads these fields back. They are kept only for (a) legacy
    config-file field backward-compatibility (display / round-tripping old
    preset files) and (b) contributing to the detection/engine
    cache-invalidation hash key (see
    ``trackerkit/gui/orchestrators/tracking.py``'s cache-id builder), so the
    derived values MUST stay stable to preserve existing tracking caches.

    ``yolo_device`` is the resolved device (``"cuda"`` -> ``"cuda:0"``),
    ``enable_tensorrt`` is ``backend == "tensorrt"``, ``enable_gpu_background``
    is ``device != "cpu"``, and ``enable_onnx_runtime`` is always ``False``
    (the resolver never emits an ONNX-Runtime backend). ``"coreml"`` (native
    Apple GPU-Fast) maps to the plain ``"mps"`` device with no ONNX flag set,
    distinct from the legacy ``"onnx_coreml"`` string.
    """
    device_map = {"cpu": "cpu", "cuda": "cuda:0", "mps": "mps"}
    yolo_device = device_map.get(runtime.device, "cpu")
    return {
        "yolo_device": yolo_device,
        "enable_tensorrt": runtime.backend == "tensorrt",
        "enable_onnx_runtime": False,
        "enable_gpu_background": runtime.device != "cpu",
    }


KALMAN_ANISOTROPY_RATIO_CONST = 50.0
POSE_REJECTION_THRESHOLD_CONST = 0.5
POSE_REJECTION_MIN_VISIBILITY_CONST = 0.5
DENSITY_GAUSSIAN_SIGMA_SCALE_CONST = 1.0
DENSITY_BINARIZE_THRESHOLD_CONST = 0.3
DENSITY_DOWNSAMPLE_FACTOR_CONST = 8
SOLVER_AUTOPICK_GREEDY_THRESHOLD = 50
MIN_DETECTIONS_TO_START_CONST = 1


@dataclass(frozen=True)
class TrackerCliVideoProbe:
    """Basic video metadata needed for current-video defaults."""

    fps: float = 30.0
    total_frames: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class TrackerCliSession:
    """Resolved non-GUI session state for a single video."""

    video_path: str
    config_path: str | None
    video_probe: TrackerCliVideoProbe
    config: dict[str, Any]
    raw_csv_path: str
    final_csv_path: str
    params: dict[str, Any]
    save_confidence_metrics: bool
    use_cached_detections: bool
    enable_backward_tracking: bool
    enable_postprocessing: bool
    interpolation_method: str
    interpolation_max_gap_seconds: float
    heading_flip_max_burst: int
    identity_method: str
    enable_pose_extractor: bool

    def supports_direct_run(self) -> bool:
        """Return whether the CLI can run this session without MainWindow."""
        return not self.enable_pose_extractor and self.identity_method in {
            "",
            "none",
            "none_disabled",
        }


def _cfg_get(cfg: Mapping[str, Any], new_key: str, *legacy_keys: str, default=None):
    if new_key in cfg:
        return cfg[new_key]
    for key in legacy_keys:
        if key in cfg:
            return cfg[key]
    return default


def _cfg_get_time(
    cfg: Mapping[str, Any], seconds_key: str, *frame_keys: str, default_seconds: float
) -> float:
    value = _cfg_get(cfg, seconds_key, default=None)
    if value is not None:
        return float(value)
    config_fps = float(_cfg_get(cfg, "fps", default=30.0) or 30.0)
    for frame_key in frame_keys:
        if frame_key in cfg:
            return float(cfg[frame_key]) / max(config_fps, 1e-6)
    return default_seconds


def _coerce_int_list(raw_value: Any) -> list[int] | None:
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif isinstance(raw_value, (list, tuple)):
        items = list(raw_value)
    else:
        items = [raw_value]
    try:
        return [int(item) for item in items]
    except (TypeError, ValueError):
        return None


def _autopick_greedy(n_targets: int) -> bool:
    return int(n_targets) >= SOLVER_AUTOPICK_GREEDY_THRESHOLD


def _default_advanced_config() -> dict[str, Any]:
    return {
        "roi_crop_warning_threshold": 0.6,
        "roi_crop_auto_suggest": True,
        "roi_crop_remind_every_session": False,
        "roi_crop_padding_fraction": 0.05,
        "video_crop_codec": "libx264",
        "video_crop_crf": 18,
        "video_crop_preset": "medium",
        "mps_memory_fraction": 0.3,
        "cuda_memory_fraction": 0.7,
        "tensorrt_build_workspace_gb": 4.0,
        "tensorrt_build_batch_size": None,
        "yolo_headtail_detect_conf_threshold": 0.25,
        "headtail_batch_size": 64,
        "realtime_visualization_emit_stride": 1,
        "visualization_emit_stride": 1,
        "dataset_yolo_confidence_threshold": 0.05,
        "dataset_yolo_iou_threshold": 0.5,
        "identity_swap_conf_margin": 0.2,
        "identity_rejoin_velocity_budget": 1.5,
        "identity_rejoin_dist_floor": None,
    }


def load_advanced_tracker_config() -> dict[str, Any]:
    """Load advanced TrackerKit config with the same defaults as the GUI path."""
    from hydra_suite.paths import get_advanced_config_path

    config = _default_advanced_config()
    config_path = Path(get_advanced_config_path())
    if not config_path.exists():
        return config
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        if isinstance(user_config, dict):
            config.update(user_config)
    except Exception:
        logger.warning("Failed to load advanced TrackerKit config", exc_info=True)
    return config


def load_tracker_cli_config(config_path: str | None) -> dict[str, Any]:
    """Load a saved tracking config JSON file."""
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Tracker config must be a JSON object: {config_path}")
    return payload


def probe_video(video_path: str) -> TrackerCliVideoProbe:
    """Read the minimum video metadata needed for headless defaults."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        return TrackerCliVideoProbe(
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )
    finally:
        cap.release()


def _default_output_paths(video_path: str) -> tuple[str, str]:
    video = Path(video_path)
    base = video.with_suffix("")
    return (
        str(base.parent / f"{base.name}_tracking.csv"),
        str(base.parent / f"{base.name}_tracking.mp4"),
    )


def _build_roi_mask(
    roi_shapes: list[dict[str, Any]] | None,
    *,
    width: int | None,
    height: int | None,
) -> np.ndarray | None:
    if not roi_shapes or not width or not height:
        return None
    combined_mask = np.zeros((height, width), np.uint8)
    for shape in roi_shapes:
        if shape.get("mode", "include") != "include":
            continue
        if shape.get("type") == "circle":
            center_x, center_y, radius = shape.get("params", [0, 0, 0])
            cv2.circle(
                combined_mask,
                (int(center_x), int(center_y)),
                int(radius),
                255,
                -1,
            )
        elif shape.get("type") == "polygon":
            points = np.array(shape.get("params", []), dtype=np.int32)
            if len(points) > 0:
                cv2.fillPoly(combined_mask, [points], 255)
    for shape in roi_shapes:
        if shape.get("mode", "include") != "exclude":
            continue
        if shape.get("type") == "circle":
            center_x, center_y, radius = shape.get("params", [0, 0, 0])
            cv2.circle(
                combined_mask,
                (int(center_x), int(center_y)),
                int(radius),
                0,
                -1,
            )
        elif shape.get("type") == "polygon":
            points = np.array(shape.get("params", []), dtype=np.int32)
            if len(points) > 0:
                cv2.fillPoly(combined_mask, [points], 0)
    return combined_mask


def build_tracking_parameters(
    cfg: Mapping[str, Any],
    *,
    video_probe: TrackerCliVideoProbe,
    advanced_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate saved TrackerKit JSON config into worker params."""
    advanced = dict(advanced_config or load_advanced_tracker_config())
    advanced["yolo_seq_individual_batch_size"] = int(
        _cfg_get(
            cfg,
            "yolo_seq_individual_batch_size",
            default=advanced.get("yolo_seq_individual_batch_size", 4),
        )
    )
    advanced["reference_aspect_ratio"] = float(
        _cfg_get(
            cfg,
            "reference_aspect_ratio",
            default=advanced.get("reference_aspect_ratio", 4.0),
        )
    )
    advanced["enable_aspect_ratio_filtering"] = bool(
        _cfg_get(
            cfg,
            "enable_aspect_ratio_filtering",
            default=advanced.get("enable_aspect_ratio_filtering", False),
        )
    )
    advanced["min_aspect_ratio_multiplier"] = float(
        _cfg_get(
            cfg,
            "min_aspect_ratio_multiplier",
            default=advanced.get("min_aspect_ratio_multiplier", 0.5),
        )
    )
    advanced["max_aspect_ratio_multiplier"] = float(
        _cfg_get(
            cfg,
            "max_aspect_ratio_multiplier",
            default=advanced.get("max_aspect_ratio_multiplier", 2.0),
        )
    )

    fps = float(
        _cfg_get(cfg, "fps", default=video_probe.fps) or video_probe.fps or 30.0
    )
    max_targets = int(_cfg_get(cfg, "max_targets", default=4))
    reference_body_size = float(_cfg_get(cfg, "reference_body_size", default=20.0))
    resize_factor = float(_cfg_get(cfg, "resize_factor", default=1.0))
    scaled_body_size = reference_body_size * resize_factor
    reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
    scaled_body_area = reference_body_area * (resize_factor**2)

    def _seconds_to_frames(seconds: float, min_frames: int = 1) -> int:
        return max(min_frames, round(seconds * max(fps, 1e-6)))

    min_object_size_pixels = int(
        float(_cfg_get(cfg, "min_object_size_multiplier", default=0.3))
        * scaled_body_area
    )
    max_object_size_pixels = int(
        float(_cfg_get(cfg, "max_object_size_multiplier", default=3.0))
        * scaled_body_area
    )
    max_distance_multiplier = float(
        _cfg_get(cfg, "max_assignment_distance_multiplier", default=2.5)
    )
    min_respawn_multiplier = float(
        _cfg_get(cfg, "min_respawn_distance_multiplier", default=1.0)
    )
    velocity_threshold_pixels_per_frame = (
        float(_cfg_get(cfg, "velocity_threshold", default=3.0))
        * scaled_body_size
        / max(fps, 1e-6)
    )
    max_velocity_break_pixels_per_frame = (
        float(_cfg_get(cfg, "max_velocity_break", default=50.0))
        * scaled_body_size
        / max(fps, 1e-6)
    )

    lost_threshold_frames = _seconds_to_frames(
        _cfg_get_time(
            cfg, "lost_threshold_seconds", "lost_threshold_frames", default_seconds=0.5
        )
    )
    kalman_maturity_age = _seconds_to_frames(
        _cfg_get_time(
            cfg,
            "kalman_maturity_age_seconds",
            "kalman_maturity_age",
            default_seconds=0.33,
        )
    )
    bg_prime_frames = _seconds_to_frames(
        _cfg_get_time(
            cfg,
            "background_prime_seconds",
            "background_prime_frames",
            "bg_prime_frames",
            default_seconds=0.33,
        ),
        min_frames=0,
    )
    min_detection_counts = _seconds_to_frames(
        _cfg_get_time(
            cfg, "min_detect_seconds", "min_detection_counts", default_seconds=0.33
        )
    )
    min_trajectory_length = _seconds_to_frames(
        _cfg_get_time(
            cfg,
            "min_trajectory_length_seconds",
            "min_trajectory_length",
            default_seconds=0.33,
        )
    )
    max_occlusion_gap = _seconds_to_frames(
        _cfg_get_time(
            cfg,
            "max_occlusion_gap_seconds",
            "max_occlusion_gap",
            default_seconds=1.0,
        ),
        min_frames=0,
    )
    velocity_zscore_window = _seconds_to_frames(
        _cfg_get_time(
            cfg,
            "velocity_zscore_window_seconds",
            "velocity_zscore_window",
            default_seconds=0.33,
        ),
        min_frames=5,
    )
    stitch_max_gap_frames = _seconds_to_frames(
        _cfg_get_time(cfg, "stitch_max_gap_seconds", default_seconds=0.1),
        min_frames=0,
    )

    # RUNTIME_TIER is the sole runtime knob (Runtime Gen-2 FT1). Prefer the
    # config's explicit tier; if a legacy config carries an explicit
    # compute_runtime, migrate it; otherwise default to the pipeline tier "gpu".
    from hydra_suite.core.inference.config import migrate_runtime_to_tier

    runtime_tier = str(_cfg_get(cfg, "runtime_tier", default="")).strip().lower()
    if runtime_tier not in {"cpu", "gpu", "gpu_fast"}:
        legacy_runtime = _cfg_get(cfg, "compute_runtime", default=None)
        if legacy_runtime:
            runtime_tier = migrate_runtime_to_tier({str(legacy_runtime)})
        else:
            runtime_tier = "gpu"
    # Legacy detection fields derive from the resolved backend for the tier
    # (Runtime Gen-2). The resolver is host-dependent (matching the live GUI
    # path), and the ResolvedBackend branch of ``legacy_detection_runtime_fields``
    # reproduces the historical cache-keyed values byte-for-byte, so existing
    # tracking caches stay valid. Detection resolves against the "obb" stage.
    resolved_backend = RuntimeResolver(runtime_tier, detect_platform()).resolve("obb")
    detection_runtime = legacy_detection_runtime_fields(resolved_backend)
    yolo_mode = str(_cfg_get(cfg, "yolo_obb_mode", default="direct")).strip().lower()
    yolo_direct_path = resolve_model_path(
        _cfg_get(cfg, "yolo_obb_direct_model_path", "yolo_model_path", default="")
    )
    yolo_detect_path = resolve_model_path(
        _cfg_get(cfg, "yolo_detect_model_path", default="")
    )
    yolo_crop_obb_path = resolve_model_path(
        _cfg_get(cfg, "yolo_crop_obb_model_path", "yolo_model_path", default="")
    )
    yolo_headtail_path = resolve_model_path(
        _cfg_get(cfg, "yolo_headtail_model_path", default="")
    )
    yolo_path = yolo_direct_path if yolo_mode == "direct" else yolo_crop_obb_path

    trt_build_batch_size_raw = advanced.get("tensorrt_build_batch_size")
    if trt_build_batch_size_raw in (None, "", 0, "0"):
        trt_build_batch_size = None
    else:
        try:
            trt_build_batch_size = max(1, int(trt_build_batch_size_raw))
        except (TypeError, ValueError):
            trt_build_batch_size = None

    kalman_longitudinal_noise = float(
        _cfg_get(cfg, "kalman_longitudinal_noise_multiplier", default=5.0)
    )
    kalman_lateral_noise = float(
        _cfg_get(
            cfg,
            "kalman_lateral_noise_multiplier",
            default=kalman_longitudinal_noise / KALMAN_ANISOTROPY_RATIO_CONST,
        )
    )
    start_frame_default = 0
    end_frame_default = (
        max(0, int(video_probe.total_frames) - 1)
        if video_probe.total_frames is not None
        else None
    )
    start_frame = int(_cfg_get(cfg, "start_frame", default=start_frame_default))
    end_frame = _cfg_get(cfg, "end_frame", default=end_frame_default)
    if end_frame is not None:
        end_frame = int(end_frame)

    rng = np.random.default_rng(42)
    colors = [
        tuple(color.tolist()) for color in rng.integers(0, 255, size=(max_targets, 3))
    ]
    roi_mask = _build_roi_mask(
        cfg.get("roi_shapes") or [],
        width=video_probe.width,
        height=video_probe.height,
    )
    enable_greedy = bool(
        _cfg_get(cfg, "enable_greedy_assignment", default=_autopick_greedy(max_targets))
    )
    enable_spatial = bool(
        _cfg_get(
            cfg, "enable_spatial_optimization", default=_autopick_greedy(max_targets)
        )
    )

    return {
        "ADVANCED_CONFIG": advanced,
        "DETECTION_METHOD": str(
            _cfg_get(cfg, "detection_method", default="background_subtraction")
        ),
        "FPS": fps,
        "START_FRAME": start_frame,
        "END_FRAME": end_frame,
        "YOLO_MODEL_PATH": yolo_path,
        "YOLO_OBB_MODE": yolo_mode,
        "YOLO_OBB_DIRECT_MODEL_PATH": yolo_direct_path,
        "YOLO_DETECT_MODEL_PATH": yolo_detect_path,
        "YOLO_CROP_OBB_MODEL_PATH": yolo_crop_obb_path,
        "YOLO_HEADTAIL_MODEL_PATH": yolo_headtail_path,
        "POSE_OVERRIDES_HEADTAIL": bool(
            _cfg_get(cfg, "pose_overrides_headtail", default=False)
        ),
        "YOLO_SEQ_CROP_PAD_RATIO": float(
            _cfg_get(cfg, "yolo_seq_crop_pad_ratio", default=0.15)
        ),
        "YOLO_SEQ_MIN_CROP_SIZE_PX": int(
            _cfg_get(cfg, "yolo_seq_min_crop_size_px", default=64)
        ),
        "YOLO_SEQ_ENFORCE_SQUARE_CROP": bool(
            _cfg_get(cfg, "yolo_seq_enforce_square_crop", default=True)
        ),
        "YOLO_SEQ_STAGE2_IMGSZ": int(
            _cfg_get(cfg, "yolo_seq_stage2_imgsz", default=160)
        ),
        "YOLO_SEQ_INDIVIDUAL_BATCH_SIZE": int(
            _cfg_get(cfg, "yolo_seq_individual_batch_size", default=4)
        ),
        "YOLO_SEQ_STAGE2_RUNTIME_BUILD_BATCH_SIZE": int(
            _cfg_get(cfg, "yolo_seq_individual_batch_size", default=4)
        ),
        "YOLO_BATCH_SIZE": int(_cfg_get(cfg, "detection_batch_size", default=1)),
        "YOLO_SEQ_STAGE2_POW2_PAD": bool(
            _cfg_get(cfg, "yolo_seq_stage2_pow2_pad", default=False)
        ),
        "YOLO_SEQ_DETECT_CONF_THRESHOLD": float(
            _cfg_get(cfg, "yolo_seq_detect_conf_threshold", default=0.25)
        ),
        "YOLO_HEADTAIL_CONF_THRESHOLD": float(
            _cfg_get(cfg, "yolo_headtail_conf_threshold", default=0.25)
        ),
        "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD": float(
            _cfg_get(
                cfg,
                "yolo_headtail_detect_conf_threshold",
                default=advanced.get("yolo_headtail_detect_conf_threshold", 0.25),
            )
        ),
        "HEADTAIL_BATCH_SIZE": int(
            _cfg_get(
                cfg,
                "headtail_batch_size",
                default=advanced.get("headtail_batch_size", 64),
            )
        ),
        "YOLO_CONFIDENCE_THRESHOLD": float(
            _cfg_get(cfg, "yolo_confidence_threshold", default=0.25)
        ),
        "YOLO_IOU_THRESHOLD": float(_cfg_get(cfg, "yolo_iou_threshold", default=0.45)),
        "USE_CUSTOM_OBB_IOU_FILTERING": bool(
            _cfg_get(cfg, "use_custom_obb_iou_filtering", default=True)
        ),
        "YOLO_TARGET_CLASSES": _coerce_int_list(
            _cfg_get(cfg, "yolo_target_classes", default=None)
        ),
        "RUNTIME_TIER": runtime_tier,
        "YOLO_DEVICE": detection_runtime["yolo_device"],
        "ENABLE_GPU_BACKGROUND": detection_runtime["enable_gpu_background"],
        "ENABLE_TENSORRT": detection_runtime["enable_tensorrt"],
        "ENABLE_ONNX_RUNTIME": detection_runtime["enable_onnx_runtime"],
        # Defaults to the detection batch the engine will actually be fed.
        # Previously defaulted to the legacy manual YOLO batch key (now removed).
        "TENSORRT_MAX_BATCH_SIZE": int(
            _cfg_get(
                cfg,
                "tensorrt_max_batch_size",
                default=max(
                    1, int(_cfg_get(cfg, "detection_batch_size", default=1) or 1)
                ),
            )
        ),
        "TENSORRT_BUILD_WORKSPACE_GB": float(
            advanced.get("tensorrt_build_workspace_gb", 4.0)
        ),
        "TENSORRT_BUILD_BATCH_SIZE": trt_build_batch_size,
        "MAX_TARGETS": max_targets,
        "THRESHOLD_VALUE": float(
            _cfg_get(cfg, "subtraction_threshold", "threshold_value", default=50.0)
        ),
        "MORPH_KERNEL_SIZE": int(_cfg_get(cfg, "morph_kernel_size", default=5)),
        "MIN_CONTOUR_AREA": float(_cfg_get(cfg, "min_contour_area", default=50.0)),
        "ENABLE_SIZE_FILTERING": bool(
            _cfg_get(cfg, "enable_size_filtering", default=False)
        ),
        "MIN_OBJECT_SIZE": min_object_size_pixels,
        "MAX_OBJECT_SIZE": max_object_size_pixels,
        "MAX_CONTOUR_MULTIPLIER": float(
            _cfg_get(cfg, "max_contour_multiplier", default=20.0)
        ),
        "MAX_DISTANCE_THRESHOLD": max_distance_multiplier * scaled_body_size,
        "MAX_DISTANCE_MULTIPLIER": max_distance_multiplier,
        "ENABLE_POSTPROCESSING": bool(
            _cfg_get(cfg, "enable_postprocessing", default=True)
        ),
        "MIN_TRAJECTORY_LENGTH": min_trajectory_length,
        "MAX_VELOCITY_BREAK": max_velocity_break_pixels_per_frame,
        "MAX_OCCLUSION_GAP": max_occlusion_gap,
        "ENABLE_TRACKLET_RELINKING": bool(
            _cfg_get(cfg, "enable_tracklet_relinking", default=False)
        ),
        "RELINK_POSE_MAX_DISTANCE": float(
            _cfg_get(cfg, "relink_pose_max_distance", default=0.45)
        ),
        "POSE_EXPORT_MIN_VALID_FRACTION": float(
            _cfg_get(cfg, "pose_export_min_valid_fraction", default=0.5)
        ),
        "POSE_EXPORT_MIN_VALID_KEYPOINTS": int(
            _cfg_get(cfg, "pose_export_min_valid_keypoints", default=3)
        ),
        "RELINK_MIN_POSE_QUALITY": float(
            _cfg_get(cfg, "relink_min_pose_quality", default=0.6)
        ),
        "POSE_POSTPROC_MAX_GAP": int(_cfg_get(cfg, "pose_postproc_max_gap", default=5)),
        "POSE_TEMPORAL_OUTLIER_ZSCORE": float(
            _cfg_get(cfg, "pose_temporal_outlier_zscore", default=3.0)
        ),
        "MAX_VELOCITY_ZSCORE": float(_cfg_get(cfg, "max_velocity_zscore", default=0.0)),
        "VELOCITY_ZSCORE_WINDOW": velocity_zscore_window,
        "CHANGEPOINT_PENALTY": float(_cfg_get(cfg, "changepoint_penalty", default=3.0)),
        "FRAGMENT_CNN_WEIGHT": float(
            _cfg_get(cfg, "fragment_cnn_weight", default=0.40)
        ),
        "FRAGMENT_TAG_WEIGHT": float(
            _cfg_get(cfg, "fragment_tag_weight", default=0.15)
        ),
        "ONLINE_PRIOR_WEIGHT": float(
            _cfg_get(cfg, "online_prior_weight", default=0.25)
        ),
        "FRAGMENT_LENGTH_WEIGHT": float(
            _cfg_get(cfg, "fragment_length_weight", default=0.60)
        ),
        "ASSIGNMENT_MARGIN_THRESHOLD": float(
            _cfg_get(cfg, "assignment_margin_threshold", default=0.10)
        ),
        "MIN_FRAGMENT_FRAMES": int(_cfg_get(cfg, "min_fragment_frames", default=5)),
        "PELT_MODEL": str(_cfg_get(cfg, "pelt_model", default="rbf")),
        "ENABLE_FRAGMENT_SCORING": bool(
            _cfg_get(cfg, "enable_fragment_scoring", default=True)
        ),
        "ENABLE_PELT_SPLITTING": bool(
            _cfg_get(cfg, "enable_pelt_splitting", default=False)
        ),
        "VELOCITY_ZSCORE_MIN_VELOCITY": (
            float(_cfg_get(cfg, "velocity_zscore_min_velocity", default=2.0))
            * scaled_body_size
            / max(fps, 1e-6)
        ),
        "MIN_RESPAWN_DISTANCE": min_respawn_multiplier * scaled_body_size,
        "MIN_DETECTION_COUNTS": min_detection_counts,
        "MIN_DETECTIONS_TO_START": MIN_DETECTIONS_TO_START_CONST,
        "TRAJECTORY_HISTORY_SECONDS": float(
            _cfg_get(cfg, "trajectory_history_seconds", default=2.0)
        ),
        "BACKGROUND_PRIME_FRAMES": bg_prime_frames,
        "BACKGROUND_CONVERGENCE_EPSILON": float(
            _cfg_get(cfg, "background_convergence_epsilon", default=1e-4)
        ),
        "BACKGROUND_CONVERGENCE_FRAMES": int(
            _cfg_get(cfg, "background_convergence_frames", default=30)
        ),
        "BACKGROUND_CONVERGENCE_PIXEL_DELTA": float(
            _cfg_get(cfg, "background_convergence_pixel_delta", default=5.0)
        ),
        "ENABLE_LIGHTING_STABILIZATION": bool(
            _cfg_get(cfg, "enable_lighting_stabilization", default=True)
        ),
        "ENABLE_ADAPTIVE_BACKGROUND": bool(
            _cfg_get(
                cfg, "enable_adaptive_background", "adaptive_background", default=True
            )
        ),
        "BACKGROUND_LEARNING_RATE": float(
            _cfg_get(cfg, "background_learning_rate", default=0.001)
        ),
        "LIGHTING_SMOOTH_FACTOR": float(
            _cfg_get(cfg, "lighting_smooth_factor", default=0.95)
        ),
        "LIGHTING_MEDIAN_WINDOW": int(
            _cfg_get(cfg, "lighting_median_window", default=5)
        ),
        "KALMAN_NOISE_COVARIANCE": float(
            _cfg_get(cfg, "kalman_process_noise", default=0.5)
        ),
        "KALMAN_MEASUREMENT_NOISE_COVARIANCE": float(
            _cfg_get(cfg, "kalman_measurement_noise", default=1.0)
        ),
        "KALMAN_DAMPING": float(_cfg_get(cfg, "kalman_velocity_damping", default=0.9)),
        "KALMAN_MATURITY_AGE": kalman_maturity_age,
        "KALMAN_INITIAL_VELOCITY_RETENTION": float(
            _cfg_get(cfg, "kalman_initial_velocity_retention", default=0.1)
        ),
        "KALMAN_MAX_VELOCITY_MULTIPLIER": float(
            _cfg_get(cfg, "kalman_max_velocity_multiplier", default=3.0)
        ),
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": kalman_longitudinal_noise,
        "KALMAN_LATERAL_NOISE_MULTIPLIER": kalman_lateral_noise,
        "KALMAN_ANISOTROPY_RATIO": max(
            1.0,
            kalman_longitudinal_noise / max(kalman_lateral_noise, 1e-6),
        ),
        "RESIZE_FACTOR": resize_factor,
        "ENABLE_CONSERVATIVE_SPLIT": bool(
            _cfg_get(cfg, "enable_conservative_split", default=True)
        ),
        "CONSERVATIVE_KERNEL_SIZE": int(
            _cfg_get(cfg, "conservative_kernel_size", default=3)
        ),
        "CONSERVATIVE_ERODE_ITER": int(
            _cfg_get(
                cfg,
                "conservative_erode_iterations",
                "conservative_erode_iter",
                default=1,
            )
        ),
        "ENABLE_ADDITIONAL_DILATION": bool(
            _cfg_get(cfg, "enable_additional_dilation", default=False)
        ),
        "DILATION_ITERATIONS": int(_cfg_get(cfg, "dilation_iterations", default=0)),
        "DILATION_KERNEL_SIZE": int(_cfg_get(cfg, "dilation_kernel_size", default=3)),
        "BRIGHTNESS": int(_cfg_get(cfg, "brightness", default=0)),
        "CONTRAST": float(_cfg_get(cfg, "contrast", default=1.0)),
        "GAMMA": float(_cfg_get(cfg, "gamma", default=1.0)),
        "DARK_ON_LIGHT_BACKGROUND": bool(
            _cfg_get(cfg, "dark_on_light_background", default=True)
        ),
        "VELOCITY_THRESHOLD": velocity_threshold_pixels_per_frame,
        "INSTANT_FLIP_ORIENTATION": bool(
            _cfg_get(cfg, "enable_instant_flip", default=False)
        ),
        "MAX_ORIENT_DELTA_STOPPED": float(
            _cfg_get(cfg, "max_orientation_delta_stopped", default=20.0)
        ),
        "DIRECTED_ORIENT_POSTHOC_CONSISTENCY": bool(
            str(yolo_headtail_path or "").strip()
        ),
        "LOST_THRESHOLD_FRAMES": lost_threshold_frames,
        "W_POSITION": float(_cfg_get(cfg, "weight_position", default=0.8)),
        "W_ORIENTATION": float(_cfg_get(cfg, "weight_orientation", default=0.3)),
        "W_AREA": float(_cfg_get(cfg, "weight_area", default=0.2)),
        "W_ASPECT": float(_cfg_get(cfg, "weight_aspect_ratio", default=0.1)),
        "W_POSE_DIRECTION": float(_cfg_get(cfg, "weight_pose_direction", default=0.5)),
        "W_POSE_LENGTH": float(_cfg_get(cfg, "weight_pose_length", default=0.0)),
        "POSE_VALID_ORIENTATION_SCALE": float(
            _cfg_get(cfg, "pose_valid_orientation_scale", default=0.15)
        ),
        "USE_MAHALANOBIS": bool(
            _cfg_get(cfg, "use_mahalanobis_distance", default=False)
        ),
        "ENABLE_GREEDY_ASSIGNMENT": enable_greedy,
        "ENABLE_SPATIAL_OPTIMIZATION": enable_spatial,
        "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER": float(
            _cfg_get(cfg, "association_stage1_motion_gate_multiplier", default=1.0)
        ),
        "ASSOCIATION_STAGE1_MAX_AREA_RATIO": float(
            _cfg_get(cfg, "association_stage1_max_area_ratio", default=2.0)
        ),
        "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF": float(
            _cfg_get(cfg, "association_stage1_max_aspect_diff", default=1.0)
        ),
        "ENABLE_POSE_REJECTION": bool(
            _cfg_get(cfg, "enable_pose_rejection", default=False)
        ),
        "POSE_REJECTION_THRESHOLD": float(
            _cfg_get(
                cfg, "pose_rejection_threshold", default=POSE_REJECTION_THRESHOLD_CONST
            )
        ),
        "POSE_REJECTION_MIN_VISIBILITY": float(
            _cfg_get(
                cfg,
                "pose_rejection_min_visibility",
                default=POSE_REJECTION_MIN_VISIBILITY_CONST,
            )
        ),
        "TRACK_FEATURE_EMA_ALPHA": float(
            _cfg_get(cfg, "track_feature_ema_alpha", default=0.25)
        ),
        "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD": float(
            _cfg_get(cfg, "association_high_confidence_threshold", default=0.8)
        ),
        "TRAJECTORY_COLORS": colors,
        "SHOW_FG": False,
        "SHOW_BG": False,
        "SHOW_CIRCLES": False,
        "SHOW_ORIENTATION": False,
        "SHOW_YOLO_OBB": False,
        "SHOW_TRAJECTORIES": False,
        "SHOW_LABELS": False,
        "SHOW_STATE": False,
        "SHOW_KALMAN_UNCERTAINTY": False,
        "VISUALIZATION_FREE_MODE": True,
        "TRACKING_REALTIME_MODE": False,
        "TRACKING_WORKFLOW_MODE": "non_realtime",
        "zoom_factor": 1.0,
        "ROI_MASK": roi_mask,
        "REFERENCE_BODY_SIZE": reference_body_size,
        "AGREEMENT_DISTANCE": float(
            _cfg_get(cfg, "merge_agreement_distance_multiplier", default=0.5)
        )
        * scaled_body_size,
        "MIN_OVERLAP_FRAMES": int(_cfg_get(cfg, "min_overlap_frames", default=5)),
        "STITCH_MAX_GAP_FRAMES": stitch_max_gap_frames,
        "STITCH_DENSITY_TIGHTEN_FACTOR": float(
            _cfg_get(cfg, "stitch_density_tighten_factor", default=0.5)
        ),
        "STITCH_SINGLE_OPTION_MARGIN": float(
            _cfg_get(cfg, "stitch_single_option_margin", default=0.5)
        ),
        "STITCH_HEADING_GATE_DEG": float(
            _cfg_get(cfg, "stitch_heading_gate_deg", default=60.0)
        ),
        "IDENTITY_DISAGREE_MIN_RUN": int(
            _cfg_get(cfg, "identity_disagree_min_run", default=5)
        ),
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": bool(
            _cfg_get(cfg, "identity_gates_trajectory_structure", default=True)
        ),
        "ENABLE_CONFIDENCE_DENSITY_MAP": bool(
            _cfg_get(cfg, "enable_confidence_density_map", default=True)
        ),
        "DENSITY_GAUSSIAN_SIGMA_SCALE": float(
            _cfg_get(
                cfg,
                "density_gaussian_sigma_scale",
                default=DENSITY_GAUSSIAN_SIGMA_SCALE_CONST,
            )
        ),
        "DENSITY_TEMPORAL_SIGMA": float(
            _cfg_get(cfg, "density_temporal_sigma", default=2.0)
        ),
        "DENSITY_BINARIZE_THRESHOLD": float(
            _cfg_get(
                cfg,
                "density_binarize_threshold",
                default=DENSITY_BINARIZE_THRESHOLD_CONST,
            )
        ),
        "DENSITY_CONSERVATIVE_FACTOR": float(
            _cfg_get(cfg, "density_conservative_factor", default=0.7)
        ),
        "DENSITY_MIN_FRAME_DURATION": int(
            _cfg_get(cfg, "density_min_frame_duration", default=3)
        ),
        "DENSITY_MIN_AREA_BODIES": float(
            _cfg_get(cfg, "density_min_area_bodies", default=0.25)
        ),
        "DENSITY_DOWNSAMPLE_FACTOR": int(
            _cfg_get(
                cfg,
                "density_downsample_factor",
                default=DENSITY_DOWNSAMPLE_FACTOR_CONST,
            )
        ),
        "EXPORT_CONFIDENCE_DENSITY_VIDEO": bool(
            _cfg_get(cfg, "export_confidence_density_video", default=False)
        ),
    }


def load_tracker_cli_session(
    video_path: str,
    *,
    config_path: str | None = None,
    config_data: Mapping[str, Any] | None = None,
    video_probe: TrackerCliVideoProbe | None = None,
    advanced_config: Mapping[str, Any] | None = None,
) -> TrackerCliSession:
    """Resolve a pure headless session for one video/config pair."""
    cfg = (
        deepcopy(dict(config_data))
        if config_data is not None
        else load_tracker_cli_config(config_path)
    )
    probe = video_probe or probe_video(video_path)
    raw_csv_path, _video_output_path = _default_output_paths(video_path)
    params = build_tracking_parameters(
        cfg,
        video_probe=probe,
        advanced_config=advanced_config,
    )
    final_csv_path = f"{os.path.splitext(raw_csv_path)[0]}_forward_processed.csv"
    return TrackerCliSession(
        video_path=video_path,
        config_path=config_path,
        video_probe=probe,
        config=cfg,
        raw_csv_path=raw_csv_path,
        final_csv_path=final_csv_path,
        params=params,
        save_confidence_metrics=bool(
            _cfg_get(cfg, "save_confidence_metrics", default=True)
        ),
        use_cached_detections=bool(
            _cfg_get(cfg, "use_cached_detections", default=True)
        ),
        enable_backward_tracking=bool(
            _cfg_get(cfg, "enable_backward_tracking", default=False)
        ),
        enable_postprocessing=bool(
            _cfg_get(cfg, "enable_postprocessing", default=True)
        ),
        interpolation_method=str(_cfg_get(cfg, "interpolation_method", default="None")),
        interpolation_max_gap_seconds=float(
            _cfg_get_time(
                cfg,
                "interpolation_max_gap_seconds",
                "interpolation_max_gap",
                default_seconds=0.33,
            )
        ),
        heading_flip_max_burst=int(_cfg_get(cfg, "heading_flip_max_burst", default=5)),
        identity_method=str(_cfg_get(cfg, "identity_method", default="none_disabled"))
        .strip()
        .lower(),
        enable_pose_extractor=bool(
            _cfg_get(cfg, "enable_pose_extractor", default=False)
        ),
    )
