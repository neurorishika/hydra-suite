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

from hydra_suite.core.inference.model_paths import (
    resolve_model_path,
    resolve_pose_model_path,
)
from hydra_suite.runtime.resolver import (
    ResolvedBackend,
    RuntimeResolver,
    detect_platform,
)

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
        """Every session runs the direct Qt-free path (Slice 4 CLI cutover).

        The hidden-MainWindow bridge was deleted once TrackingSessionCore
        reached parity, so pose/identity sessions no longer need Qt. Kept as a
        method (not deleted) so the CLI's call site stays stable.
        """
        return True


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


def _coerce_pose_keypoint_tokens(raw_value: Any) -> list:
    """Parse a pose keypoint group into name/index tokens.

    Mirrors the bridge's ``MainWindow._parse_pose_keypoint_tokens`` (a
    comma-separated-string-or-list parser that keeps numeric tokens as
    ``int`` and everything else as a stripped ``str``). The CLI has no list
    widgets to read a live selection from, so the config's stored
    ``pose_*_keypoints`` list (already the bridge's own parsed/selected
    output at save time) is re-parsed the same way to keep the token types
    identical.
    """
    if not raw_value:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        values = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    tokens: list = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            tokens.append(int(text))
        except ValueError:
            tokens.append(text)
    return tokens


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

    from hydra_suite.core.tracking.session_policy import build_trajectory_colors

    colors = build_trajectory_colors(max_targets)
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

    # Individual/identity pipeline gate: faithfully replicate the bridge's
    # derivation (gui/orchestrators/config.py:1797-1798, 2385-2386), which
    # delegates to MainWindow._is_individual_pipeline_enabled() ->
    # session_policy.is_individual_pipeline_enabled({"detection_method": ...}).
    # That predicate is purely a function of detection_method == "yolo_obb" --
    # not of any identity/individual-analysis checkbox -- so reuse it directly
    # from the same config-dict-shaped mapping the CLI already has.
    from hydra_suite.core.tracking.session_policy import is_individual_pipeline_enabled

    individual_pipeline_enabled = is_individual_pipeline_enabled(cfg)

    # CNN_CLASSIFIERS / USE_APRILTAGS: the bridge reads these from
    # MainWindow._identity_config() (detection_panel.py:_identity_config),
    # which is gated by the identity-panel "enabled" checkbox and persisted
    # verbatim into the saved config's "cnn_classifiers"/"use_apriltags"
    # fields at save time (gui/orchestrators/config.py:1803-1805). The CLI has
    # no live widgets, so the persisted config fields already encode that same
    # gated state -- read them directly. Each CNN classifier's "model_path" is
    # stored as an absolute path resolved from the user's models root at save
    # time (identity_panel.py CnnClassifierRow.to_config); re-resolve it here
    # via the same resolve_model_path() used for YOLO model paths above, so a
    # relative/portable path in a hand-authored or fixture config still
    # resolves to an existing file the way the bridge's absolute path would.
    use_apriltags = bool(_cfg_get(cfg, "use_apriltags", default=False))
    cnn_classifiers_raw = _cfg_get(cfg, "cnn_classifiers", default=[]) or []
    cnn_classifiers = []
    for entry in cnn_classifiers_raw:
        entry = dict(entry)
        entry["model_path"] = resolve_model_path(entry.get("model_path", ""))
        cnn_classifiers.append(entry)

    # Identity-in-tracking param block: faithfully replicate the bridge's
    # derivation (gui/orchestrators/config.py:2400-2436). This block feeds
    # the per-frame Hungarian assignment's Bayesian identity-cost term
    # (core/assigners/hungarian.py:239, _apply_bayesian_identity_cost) and
    # the identity-first slot rejoining (core/tracking/worker.py:2899-2910),
    # both of which the CLI previously left at their permissive defaults
    # (decoder off, hint scale 0) regardless of what the config asked for.
    enable_postprocessing_flag = bool(
        _cfg_get(cfg, "enable_postprocessing", default=True)
    )
    # bridge: config.py:1808 saves the raw checkbox; default mirrors the
    # widget's initial checked state (tracking_panel.py:537).
    enable_identity_in_tracking = bool(
        _cfg_get(cfg, "enable_identity_in_tracking", default=True)
    )
    # bridge: config.py:2401-2404 ANDs the online-decoder checkbox (default
    # unchecked, tracking_panel.py:568) with the master identity-in-tracking
    # gate above.
    enable_identity_online_decoder = enable_identity_in_tracking and bool(
        _cfg_get(cfg, "enable_identity_online_decoder", default=False)
    )
    # bridge: config.py:722-727 -- a saved config with no
    # "identity_postprocess_mode" key falls back to "Fragment Solver" (NOT
    # the raw widget default), then config.py:2405-2409 gates the *value*
    # emitted into params on the postprocessing master switch.
    saved_identity_postprocess_mode = _cfg_get(
        cfg, "identity_postprocess_mode", default=None
    )
    if saved_identity_postprocess_mode is None:
        saved_identity_postprocess_mode = "Fragment Solver"
    saved_identity_postprocess_mode = str(saved_identity_postprocess_mode)
    identity_postprocess_mode = (
        saved_identity_postprocess_mode if enable_postprocessing_flag else "None"
    )
    # bridge: config.py:2410-2414.
    enable_identity_fragment_solver = (
        enable_postprocessing_flag
        and saved_identity_postprocess_mode == "Fragment Solver"
    )
    # bridge: config.py:1799/2116-2118 -- IDENTITY_METHOD is the canonical
    # gated identity-method string persisted at save time by
    # MainWindow._selected_identity_method(); replicate TrackerCliSession's
    # own normalization of the same field (cli_config.py's
    # load_tracker_cli_session) rather than re-deriving the GUI enable-gate
    # logic (no live widgets exist headlessly).
    identity_method = (
        str(_cfg_get(cfg, "identity_method", default="none_disabled")).strip().lower()
    )

    # Pose block: faithfully replicate the bridge's derivation
    # (gui/orchestrators/config.py:2436-2460). ``ENABLE_POSE_EXTRACTOR`` uses
    # the SAME pure predicate the bridge calls
    # (MainWindow._is_pose_inference_enabled() ->
    # session_orch._is_pose_inference_enabled() ->
    # session_policy.is_pose_inference_enabled(build_config_dict())): gated on
    # the individual/YOLO-OBB pipeline being enabled, the
    # "enable_pose_extractor" flag, AND a non-empty "pose_model_dir" -- so a
    # non-pose config (detection_method != yolo_obb, or the checkbox off, or
    # no model configured) derives it falsy, exactly like the bridge. The CLI
    # config dict already carries the same field shape as
    # ConfigOrchestrator.build_config_dict() (both are the persisted
    # snake_case JSON schema), so the predicate can run directly against
    # ``cfg`` without reconstructing the bridge's widget-derived dict.
    from hydra_suite.core.tracking.session_policy import is_pose_inference_enabled

    pose_extractor_enabled = is_pose_inference_enabled(cfg)
    pose_model_type = (
        str(_cfg_get(cfg, "pose_model_type", default="yolo")).strip().lower()
    )
    if pose_model_type not in ("yolo", "sleap", "vitpose"):
        pose_model_type = "yolo"
    # bridge: config.py:2440-2449 resolves
    # ``self._mw._pose_model_path_for_backend(active_backend)`` -- which, on
    # config load (config.py:1161-1173), is the backend-specific
    # "pose_<backend>_model_dir" field, falling back to the legacy
    # "pose_model_dir" field only when the backend-specific field is empty.
    # Replicate the same backend-specific-with-legacy-fallback lookup here.
    pose_model_dir_raw = str(
        _cfg_get(cfg, f"pose_{pose_model_type}_model_dir", default="")
    ).strip()
    if not pose_model_dir_raw:
        pose_model_dir_raw = str(_cfg_get(cfg, "pose_model_dir", default="")).strip()
    pose_model_dir = resolve_pose_model_path(
        pose_model_dir_raw, backend=pose_model_type
    )
    # bridge: config.py:2458, MainWindow._selected_pose_sleap_env() falls
    # back to "sleap" when the config value is empty or a placeholder
    # "no sleap envs ..." combo-box entry.
    pose_sleap_env = str(_cfg_get(cfg, "pose_sleap_env", default="sleap")).strip()
    if not pose_sleap_env or pose_sleap_env.lower().startswith("no sleap envs"):
        pose_sleap_env = "sleap"
    # bridge: config.py:1216-1225 -- shared pose batch size loaded from
    # "pose_batch_size", falling back through the legacy "pose_yolo_batch" /
    # "pose_sleap_batch" fields, then the advanced-config default.
    pose_batch_size = int(
        _cfg_get(
            cfg,
            "pose_batch_size",
            default=_cfg_get(
                cfg,
                "pose_yolo_batch",
                default=_cfg_get(
                    cfg,
                    "pose_sleap_batch",
                    default=advanced.get("pose_batch_size", 4),
                ),
            ),
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
        # Pose block: see the derivation block above build_tracking_parameters'
        # return statement (mirrors gui/orchestrators/config.py:2436-2460).
        "ENABLE_POSE_EXTRACTOR": pose_extractor_enabled,
        "POSE_MODEL_TYPE": pose_model_type,
        "POSE_MODEL_DIR": pose_model_dir,
        "POSE_EXPORTED_MODEL_PATH": "",
        "POSE_MIN_KPT_CONF_VALID": float(
            _cfg_get(cfg, "pose_min_kpt_conf_valid", default=0.2)
        ),
        "POSE_SKELETON_FILE": str(_cfg_get(cfg, "pose_skeleton_file", default="")),
        "POSE_IGNORE_KEYPOINTS": _coerce_pose_keypoint_tokens(
            _cfg_get(cfg, "pose_ignore_keypoints", default=[])
        ),
        "POSE_DIRECTION_ANTERIOR_KEYPOINTS": _coerce_pose_keypoint_tokens(
            _cfg_get(cfg, "pose_direction_anterior_keypoints", default=[])
        ),
        "POSE_DIRECTION_POSTERIOR_KEYPOINTS": _coerce_pose_keypoint_tokens(
            _cfg_get(cfg, "pose_direction_posterior_keypoints", default=[])
        ),
        "POSE_YOLO_BATCH": pose_batch_size,
        "POSE_BATCH_SIZE": pose_batch_size,
        "POSE_SLEAP_ENV": pose_sleap_env,
        "POSE_SLEAP_BATCH": int(
            _cfg_get(cfg, "pose_sleap_batch", default=pose_batch_size)
        ),
        "POSE_SLEAP_MAX_INSTANCES": int(
            _cfg_get(cfg, "pose_sleap_max_instances", default=1)
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
        "ENABLE_POSTPROCESSING": enable_postprocessing_flag,
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
        "ENABLE_IDENTITY_ANALYSIS": individual_pipeline_enabled,
        "ENABLE_INDIVIDUAL_PIPELINE": individual_pipeline_enabled,
        "IDENTITY_METHOD": identity_method,
        "USE_APRILTAGS": use_apriltags,
        "CNN_CLASSIFIERS": cnn_classifiers,
        "CNN_CLASSIFIER_WINDOW": 10,
        "ENABLE_IDENTITY_IN_TRACKING": enable_identity_in_tracking,
        "ENABLE_IDENTITY_ONLINE_DECODER": enable_identity_online_decoder,
        "IDENTITY_POSTPROCESS_MODE": identity_postprocess_mode,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": enable_identity_fragment_solver,
        "ASSOCIATION_IDENTITY_HINT_SCALE": float(
            _cfg_get(cfg, "identity_weight", default=1.0)
        ),
        "IDENTITY_COMMIT_THRESHOLD": float(
            _cfg_get(cfg, "identity_commit_threshold", default=0.85)
        ),
        "IDENTITY_DISPLAY_THRESHOLD": float(
            _cfg_get(cfg, "identity_display_threshold", default=0.6)
        ),
        "IDENTITY_TRANSITION_EPSILON": float(
            _cfg_get(cfg, "identity_transition_epsilon", default=0.02)
        ),
        "IDENTITY_UNKNOWN_PRIOR": float(
            _cfg_get(cfg, "identity_unknown_prior", default=0.05)
        ),
        "IDENTITY_REJOIN_THRESHOLD": float(
            _cfg_get(cfg, "identity_rejoin_threshold", default=0.5)
        ),
        "IDENTITY_SWAP_ENABLED": bool(
            _cfg_get(cfg, "enable_identity_swap_correction", default=True)
        ),
        "IDENTITY_SWAP_MIN_FRAMES": int(
            _cfg_get(cfg, "identity_swap_min_frames", default=8)
        ),
        "IDENTITY_SWAP_CONF_MARGIN": float(
            advanced.get("identity_swap_conf_margin", 0.2)
        ),
        "IDENTITY_REJOIN_VELOCITY_BUDGET": float(
            advanced.get("identity_rejoin_velocity_budget", 1.5)
        ),
        "IDENTITY_REJOIN_DIST_FLOOR": advanced.get("identity_rejoin_dist_floor", None),
        "APRILTAG_FAMILY": str(_cfg_get(cfg, "apriltag_family", default="tag36h11")),
        "APRILTAG_DECIMATE": float(_cfg_get(cfg, "apriltag_decimate", default=1.0)),
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
