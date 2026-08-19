"""Shared Qt-free engine-parameter builder for TrackerKit.

This module holds the pure (no Qt, no MainWindow) derivation that turns a
saved tracking config dict into the flat ``dict`` of engine params consumed
by ``TrackingWorker``/``TrackingSessionCore``. It was extracted verbatim from
``cli_config.build_tracking_parameters`` (Task 1 of the shared
engine-param-builder program) so it can eventually be shared between the CLI
and the GUI bridge. This first extraction is a pure refactor: output must be
byte-identical to the pre-extraction CLI derivation.

Nothing here may import ``hydra_suite.trackerkit.gui`` or PySide6.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
from hydra_suite.trackerkit.config.identity_schema import IdentityConfig

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


@dataclass
class RuntimeContext:
    """Video/runtime facts the pure param builder needs but cannot derive.

    Both the CLI (from ``TrackerCliVideoProbe`` + its own output-path logic)
    and, eventually, the GUI bridge (from the live video/session state)
    construct one of these to drive ``build_engine_params``.
    """

    fps: float
    total_frames: int | None
    frame_width: int | None
    frame_height: int | None
    roi_mask: "np.ndarray | None" = None
    # START_FRAME / END_FRAME overrides. The GUI emits these straight off its
    # spin boxes (config.py) regardless of whether the controls are disabled
    # (during a tracking/backward pass), so it supplies them here to preserve
    # that exact value. The CLI leaves them ``None`` -> the builder falls back
    # to the config-derived range (unchanged CLI behaviour).
    start_frame: int | None = None
    end_frame: int | None = None
    dataset_output_dir: str | None = None
    final_media_video_output_dir: str | None = None
    individual_dataset_output_dir: str | None = None
    individual_dataset_name: str | None = None
    individual_dataset_run_id: str | None = None
    individual_properties_cache_path: str | None = None


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


def _dataset_class_names(cfg) -> list[str]:
    """Ordered class names (index = class id), falling back to the single name."""
    raw = str(_cfg_get(cfg, "dataset_class_names", default="") or "")
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if names:
        return names
    single = str(_cfg_get(cfg, "dataset_class_name", default="") or "").strip()
    return [single or "object"]


def _autopick_greedy(n_targets: int) -> bool:
    return int(n_targets) >= SOLVER_AUTOPICK_GREEDY_THRESHOLD


def _coerce_pose_keypoint_tokens(raw_value: Any) -> list:
    """Parse a pose keypoint group into plain-string tokens.

    Mirrors the bridge's ``MainWindow._selected_pose_group_keypoints``
    (``gui/main_window.py:1159-1167``), which returns
    ``[item.text().strip() for item in list_widget.selectedItems() if
    item.text().strip()]`` -- plain stripped strings, never int-coerced
    (even for numeric-looking keypoint names). The CLI has no list widgets
    to read a live selection from, so the config's stored
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
        tokens.append(text)
    return tokens


def build_roi_mask(
    roi_shapes: list[dict[str, Any]] | None,
    width: int | None,
    height: int | None,
) -> np.ndarray | None:
    """Rasterize saved ROI shapes into a mask, or ``None`` if there is none.

    Moved from ``cli_config._build_roi_mask`` (which took ``width``/``height``
    as keyword-only args); ``build_roi_mask`` takes them positionally per the
    shared-builder interface.
    """
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


def build_arena_labels(
    roi_shapes: list[dict[str, Any]] | None,
    width: int | None,
    height: int | None,
) -> tuple[np.ndarray | None, int]:
    """Rasterize ROI shapes into a uint16 arena-label image.

    Pixel value is ``arena_id + 1``; 0 means outside every arena. The set
    ``labels > 0`` is pixel-identical to ``build_roi_mask`` on the same shapes,
    so detection gating semantics are unchanged.

    Shapes without an ``arena_id`` key map to arena 0 -- a legacy project that
    drew several shapes as one region keeps single-arena behavior exactly.
    Sparse ids are densified to a contiguous 0..n-1 range.
    """
    if not roi_shapes or not width or not height:
        return None, 1

    # Mirror build_roi_mask EXACTLY: it partitions three ways, rendering a shape
    # only on `== "include"` / `== "exclude"` and silently dropping any other
    # mode string.  Selecting includes with `!= "exclude"` would render unknown
    # modes here that the ROI mask drops, breaking the union invariant.
    includes = [s for s in roi_shapes if s.get("mode", "include") == "include"]
    raw_ids = sorted({int(s.get("arena_id", 0)) for s in includes}) or [0]
    dense = {raw: i for i, raw in enumerate(raw_ids)}

    labels = np.zeros((height, width), np.uint16)
    for shape in includes:
        value = dense[int(shape.get("arena_id", 0))] + 1
        _fill_shape(labels, shape, value)
    for shape in roi_shapes:
        if shape.get("mode", "include") == "exclude":
            _fill_shape(labels, shape, 0)
    return labels, len(raw_ids)


def _fill_shape(canvas: np.ndarray, shape: dict[str, Any], value: int) -> None:
    """Rasterize one ROI shape onto *canvas* with the given fill value."""
    if shape.get("type") == "circle":
        center_x, center_y, radius = shape.get("params", [0, 0, 0])
        cv2.circle(canvas, (int(center_x), int(center_y)), int(radius), value, -1)
    elif shape.get("type") == "polygon":
        points = np.array(shape.get("params", []), dtype=np.int32)
        if len(points) > 0:
            cv2.fillPoly(canvas, [points], value)


def _seconds_to_frames(seconds: float, fps: float, min_frames: int = 1) -> int:
    return max(min_frames, round(seconds * max(fps, 1e-6)))


def _default_advanced_config_fallback() -> dict[str, Any]:
    """Load advanced config for a bare ``build_engine_params`` call.

    ``build_engine_params`` itself owns no advanced-config file I/O or
    defaults -- ``cli_config.load_advanced_tracker_config`` (and, later, the
    GUI-side loader) is the single source of truth. Deferred import avoids a
    module-load-time circular import with ``cli_config``, which imports
    ``build_engine_params`` itself.
    """
    from hydra_suite.trackerkit.cli_config import load_advanced_tracker_config

    return load_advanced_tracker_config()


def build_engine_params(
    config: Mapping[str, Any],
    *,
    runtime: RuntimeContext,
    advanced_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a saved TrackerKit JSON config into engine worker params.

    This is a mechanical, Qt-free transform of
    ``cli_config.build_tracking_parameters``'s former body: video-probe
    attribute reads become ``runtime.*`` reads, the inline ROI-mask build
    becomes ``runtime.roi_mask`` with a ``build_roi_mask`` fallback, and the
    (never-emitted-by-the-CLI-today) output-dir keys are sourced from
    ``runtime.*`` and simply omitted when the runtime field is ``None`` --
    reproducing today's CLI output exactly.
    """
    cfg = config
    advanced = dict(advanced_config or _default_advanced_config_fallback())
    advanced["yolo_seq_individual_batch_size"] = int(
        _cfg_get(
            cfg,
            "yolo_seq_individual_batch_size",
            default=advanced.get("yolo_seq_individual_batch_size", 4),
        )
    )
    # 2.0 matches _default_advanced_config, resources/configs/default.json and
    # the GUI spin. This literal read 4.0 -- a disagreeing fourth default that
    # applied whenever the advanced table lacked the key.
    advanced["reference_aspect_ratio"] = float(
        _cfg_get(
            cfg,
            "reference_aspect_ratio",
            default=advanced.get("reference_aspect_ratio", 2.0),
        )
    )
    # The operator's dial against clipped animals under global canonicalization.
    advanced["canonical_margin"] = float(
        _cfg_get(
            cfg,
            "canonical_margin",
            default=advanced.get("canonical_margin", 1.3),
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
    # Mirror the GUI's ADVANCED_CONFIG assembly (config.py:2085-2112): the
    # bridge overlays the live detection-batch + video-pose overlay widgets
    # onto ``advanced_config`` before handing it to the engine. The CLI never
    # did this, so ADVANCED_CONFIG carried only the raw disk keys. Source each
    # from the persisted config field (build_config_dict serializes them).
    advanced["detection_batch_size"] = int(
        _cfg_get(
            cfg,
            "detection_batch_size",
            default=advanced.get("detection_batch_size", 1),
        )
    )
    advanced["video_show_pose"] = bool(
        _cfg_get(cfg, "video_show_pose", default=advanced.get("video_show_pose", False))
    )
    advanced["video_pose_point_radius"] = int(
        _cfg_get(
            cfg,
            "video_pose_point_radius",
            default=advanced.get("video_pose_point_radius", 3),
        )
    )
    advanced["video_pose_point_thickness"] = int(
        _cfg_get(
            cfg,
            "video_pose_point_thickness",
            default=advanced.get("video_pose_point_thickness", -1),
        )
    )
    advanced["video_pose_line_thickness"] = int(
        _cfg_get(
            cfg,
            "video_pose_line_thickness",
            default=advanced.get("video_pose_line_thickness", 2),
        )
    )
    advanced["video_pose_color_mode"] = str(
        _cfg_get(
            cfg,
            "video_pose_color_mode",
            default=advanced.get("video_pose_color_mode", "track"),
        )
    )
    advanced["video_pose_color"] = [
        int(component)
        for component in _cfg_get(
            cfg,
            "video_pose_color",
            default=advanced.get("video_pose_color", [255, 255, 255]),
        )
    ]

    fps = float(_cfg_get(cfg, "fps", default=runtime.fps) or runtime.fps or 30.0)
    max_targets = int(_cfg_get(cfg, "max_targets", default=4))
    reference_body_size = float(_cfg_get(cfg, "reference_body_size", default=20.0))
    resize_factor = float(_cfg_get(cfg, "resize_factor", default=1.0))
    # RESIZE_FACTOR is a background-subtraction knob. The worker's frame
    # downscale is method-agnostic, but only bg-sub is coherent under it: the
    # bg-sub stage detects on the already-resized frame and keys its cache on
    # the factor, while YOLO's batch pass decodes at NATIVE resolution
    # (run_batch_pass -> make_frame_source never sees it) and the OBB cache key
    # omits it. Left unclamped, batch YOLO returns native-space detections that
    # rescale_coordinates then divides by the factor, and realtime YOLO -- which
    # DOES detect on the downscaled frame -- silently disagrees with batch YOLO
    # for one config. Downscaling before YOLO would save decode, not inference
    # (the model letterboxes to imgsz regardless), so scoping the knob beats
    # threading it through the OBB pipeline.
    _detection_method = str(
        _cfg_get(cfg, "detection_method", default="background_subtraction")
    )
    if _detection_method != "background_subtraction" and resize_factor != 1.0:
        logger.warning(
            "RESIZE_FACTOR=%.3f ignored for detection_method=%r -- Scale applies "
            "to background subtraction only; using 1.0.",
            resize_factor,
            _detection_method,
        )
        resize_factor = 1.0
    scaled_body_size = reference_body_size * resize_factor
    reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
    scaled_body_area = reference_body_area * (resize_factor**2)

    def _s2f(seconds: float, min_frames: int = 1) -> int:
        return _seconds_to_frames(seconds, fps, min_frames=min_frames)

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

    lost_threshold_frames = _s2f(
        _cfg_get_time(
            cfg, "lost_threshold_seconds", "lost_threshold_frames", default_seconds=0.5
        )
    )
    kalman_maturity_age = _s2f(
        _cfg_get_time(
            cfg,
            "kalman_maturity_age_seconds",
            "kalman_maturity_age",
            default_seconds=0.33,
        )
    )
    bg_prime_frames = _s2f(
        _cfg_get_time(
            cfg,
            "background_prime_seconds",
            "background_prime_frames",
            "bg_prime_frames",
            default_seconds=0.33,
        ),
        min_frames=0,
    )
    min_detection_counts = _s2f(
        _cfg_get_time(
            cfg, "min_detect_seconds", "min_detection_counts", default_seconds=0.33
        )
    )
    min_trajectory_length = _s2f(
        _cfg_get_time(
            cfg,
            "min_trajectory_length_seconds",
            "min_trajectory_length",
            default_seconds=0.33,
        )
    )
    max_occlusion_gap = _s2f(
        _cfg_get_time(
            cfg,
            "max_occlusion_gap_seconds",
            "max_occlusion_gap",
            default_seconds=1.0,
        ),
        min_frames=0,
    )
    velocity_zscore_window = _s2f(
        _cfg_get_time(
            cfg,
            "velocity_zscore_window_seconds",
            "velocity_zscore_window",
            default_seconds=0.33,
        ),
        min_frames=5,
    )
    stitch_max_gap_frames = _s2f(
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
    # The GUI resolves the head-tail model via
    # ``identity._get_selected_yolo_headtail_model_path()``
    # (identity_panel.py:1532-1536), which returns "" when the "Enable
    # Head-Tail Orientation" group (``g_headtail``) is unchecked and only
    # otherwise returns the configured path. build_config_dict persists that
    # checkbox as ``enable_headtail_orientation`` (config.py:1574); the loader
    # default when the key is absent is ``bool(configured path)``
    # (config.py:436-443). Reproduce that gate so a config with a configured
    # head-tail model but the group disabled emits "" (and leaves
    # DIRECTED_ORIENT_POSTHOC_CONSISTENCY off), exactly like the GUI.
    yolo_headtail_configured = _cfg_get(cfg, "yolo_headtail_model_path", default="")
    headtail_orientation_enabled = bool(
        _cfg_get(
            cfg,
            "enable_headtail_orientation",
            default=bool(str(yolo_headtail_configured or "").strip()),
        )
    )
    yolo_headtail_path = resolve_model_path(
        yolo_headtail_configured if headtail_orientation_enabled else ""
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
        max(0, int(runtime.total_frames) - 1)
        if runtime.total_frames is not None
        else None
    )
    if runtime.start_frame is not None:
        start_frame = int(runtime.start_frame)
    else:
        start_frame = int(_cfg_get(cfg, "start_frame", default=start_frame_default))
    if runtime.end_frame is not None:
        end_frame = int(runtime.end_frame)
    else:
        end_frame = _cfg_get(cfg, "end_frame", default=end_frame_default)
        if end_frame is not None:
            end_frame = int(end_frame)

    from hydra_suite.core.tracking.session_policy import build_trajectory_colors

    colors = build_trajectory_colors(max_targets)
    roi_mask = (
        runtime.roi_mask
        if runtime.roi_mask is not None
        else build_roi_mask(
            cfg.get("roi_shapes"),
            runtime.frame_width,
            runtime.frame_height,
        )
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

    # Legacy singular CNN classifier params (bridge: config.py:2408-2415 +
    # 2548-2549). The GUI emits a hard-coded empty label/path + fixed
    # confidence, then back-fills CNN_CLASSIFIER_MODEL_PATH from
    # COLOR_TAG_MODEL_PATH when empty, and reads the batch size off the first
    # configured CNN classifier (falling back to 64). Reproduce verbatim.
    color_tag_model_path = str(_cfg_get(cfg, "color_tag_model_path", default=""))
    cnn_classifier_model_path = color_tag_model_path
    cnn_classifier_batch_size = int(
        cnn_classifiers[0].get("batch_size", 64) if cnn_classifiers else 64
    )

    # Final-media / canonical-still export gates (bridge: config.py:2127-2129,
    # 2487-2490). ENABLE_INDIVIDUAL_DATASET/IMAGE_SAVE are hard-coded False in
    # the GUI params dict. EXPORT_FINAL_CANONICAL_IMAGES equals the GUI's
    # ``_is_individual_image_save_enabled()`` -- build_config_dict persists that
    # exact value into ``export_final_canonical_images`` (config.py:1876), so
    # read it back directly. FINAL_MEDIA_EXPORT_VIDEOS_ENABLED is the GUI's
    # ``_should_export_final_media_videos()`` -- the pure session_policy
    # predicate over the persisted config (config.py:989-991).
    from hydra_suite.core.tracking.session_policy import (
        should_export_final_media_videos,
    )

    export_final_canonical_images = bool(
        _cfg_get(cfg, "export_final_canonical_images", default=False)
    )
    final_media_export_videos_enabled = should_export_final_media_videos(cfg)

    # ``_cfg_get``'s ``default`` is keyword-only (after ``*legacy_keys``), but
    # ``IdentityConfig.from_engine_config`` calls ``cfg_get(cfg, key, default)``
    # positionally -- adapt so the positional default routes to the keyword.
    identity_cfg = IdentityConfig.from_engine_config(
        cfg,
        advanced,
        cfg_get=lambda _c, _k, _d=None: _cfg_get(_c, _k, default=_d),
    )

    # Debug/Release derivation (Task 2 of the debug-mode plan): an absent
    # ``debug_mode`` key means Debug/legacy behavior. When present, it drives
    # ENABLE_PROFILING/EXPORT_CONFIDENCE_DENSITY_VIDEO; when absent, those keep
    # their independently stored values (backward compat with pre-debug-mode
    # configs, incl. the equivalence-gate fixtures).
    _debug_present = "debug_mode" in cfg
    _debug_mode = bool(_cfg_get(cfg, "debug_mode", default=True))

    _retired_padding = _cfg_get(cfg, "individual_crop_padding", default=None)
    if _retired_padding is not None:
        logger.warning(
            "Config key 'individual_crop_padding' (=%s) is retired and ignored. "
            "Crop framing for every model- and dataset-facing crop now comes "
            "from ADVANCED_CONFIG.canonical_margin. Before this change, "
            "AprilTag crops used this same '%s' value as their padding; "
            "AprilTag crops now default to 'apriltag_crop_padding' = 0.0 "
            "(the detection's exact extent), so your tag decode may change. "
            "To preserve previous AprilTag crops, set 'apriltag_crop_padding' "
            "to %s. See docs/superpowers/specs/2026-08-18-crop-padding-"
            "retirement-design.md.",
            _retired_padding,
            _retired_padding,
            _retired_padding,
        )

    params: dict[str, Any] = {
        "ADVANCED_CONFIG": advanced,
        "DEBUG_MODE": _debug_mode,
        "DETECTION_METHOD": _detection_method,
        "FPS": fps,
        "START_FRAME": start_frame,
        "END_FRAME": end_frame,
        "YOLO_MODEL_PATH": yolo_path,
        "YOLO_OBB_MODE": yolo_mode,
        # OBB direct-task + fixed-angle knobs (bridge: config.py:2161-2173).
        "YOLO_OBB_DIRECT_TASK": str(
            _cfg_get(cfg, "yolo_obb_direct_task", default="obb")
        ),
        "YOLO_OBB_FIXED_ANGLE_DEG": float(
            _cfg_get(cfg, "yolo_fixed_angle_deg", default=0.0)
        ),
        # Segment-as-OBB rotated-rect kernel knobs (advanced-config only; read
        # only when YOLO_OBB_DIRECT_TASK == "segment"). Bridge reads them from
        # advanced_config with these defaults (config.py:2168-2173).
        "YOLO_OBB_SEG_NUM_ANGLES": advanced.get("obb_seg_num_angles", 24),
        "YOLO_OBB_SEG_CROP_SIZE": advanced.get("obb_seg_crop_size", 64),
        "YOLO_OBB_SEG_PAD_RATIO": advanced.get("obb_seg_pad_ratio", 0.15),
        "YOLO_OBB_SEG_MASK_THRESHOLD": advanced.get("obb_seg_mask_threshold", 0.5),
        # SAHI slicing knobs (bridge: config.py:2174-2191). SLICE_ENABLED /
        # SLICE_GEOMETRY_MODE are persisted config fields; the rest live in
        # advanced_config with these defaults.
        "SLICE_ENABLED": bool(_cfg_get(cfg, "slice_enabled", default=False)),
        "SLICE_GEOMETRY_MODE": str(
            _cfg_get(cfg, "slice_geometry_mode", default="auto_model")
        ),
        "SLICE_OVERLAP": advanced.get("slice_overlap", 0.2),
        "SLICE_HEIGHT": advanced.get("slice_height", 0),
        "SLICE_WIDTH": advanced.get("slice_width", 0),
        "SLICE_OBJECT_TILE_FRACTION": advanced.get("slice_object_tile_fraction", 0.15),
        "SLICE_TRAINED_BODY_PX": advanced.get("slice_trained_body_px", 0.0),
        "SLICE_MERGE_POLICY": advanced.get("slice_merge_policy", "greedy_nmm"),
        "SLICE_MERGE_METRIC": advanced.get("slice_merge_metric", "ios"),
        "SLICE_MERGE_THRESHOLD": advanced.get("slice_merge_threshold", 0.5),
        "SLICE_MERGE_BACKEND": advanced.get("slice_merge_backend", "cv2"),
        "SLICE_PERFORM_STANDARD_PRED": advanced.get(
            "slice_perform_standard_pred", False
        ),
        "YOLO_OBB_DIRECT_MODEL_PATH": yolo_direct_path,
        "YOLO_DETECT_MODEL_PATH": yolo_detect_path,
        "YOLO_CROP_OBB_MODEL_PATH": yolo_crop_obb_path,
        "YOLO_HEADTAIL_MODEL_PATH": yolo_headtail_path,
        "POSE_OVERRIDES_HEADTAIL": bool(
            _cfg_get(cfg, "pose_overrides_headtail", default=False)
        ),
        # Pose block: see the derivation block above build_engine_params'
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
        # bridge: config.py:2472-2475 -- POSE_YOLO_BATCH, POSE_BATCH_SIZE,
        # and POSE_SLEAP_BATCH all come from the single
        # spin_pose_batch.value() widget. Reuse pose_batch_size verbatim
        # rather than re-reading a distinct "pose_sleap_batch" config field.
        "POSE_SLEAP_BATCH": pose_batch_size,
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
        # Retained for CLI/GUI config back-compat only. The worker no longer
        # reads this param: the three confidence columns are always emitted
        # per row, and the header is always built to match.
        "SAVE_CONFIDENCE_METRICS": bool(
            _cfg_get(cfg, "save_confidence_metrics", default=True)
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
        "MAX_BRIDGE_GAP_FRAMES": int(
            _cfg_get(cfg, "max_bridge_gap_frames", default=30)
        ),
        "FRAGMENT_SPATIAL_VETO_THRESHOLD": float(
            _cfg_get(cfg, "fragment_spatial_veto_threshold", default=0.05)
        ),
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
        # NB: BACKGROUND_CONVERGENCE_EPSILON/FRAMES/PIXEL_DELTA are intentionally
        # NOT emitted -- the GUI reference (get_parameters_dict) never emits
        # them, so the engine falls back to its own defaults (1e-4 / 30 / 5.0,
        # core/background/model.py:579-581 and core/inference/config.py:285-291),
        # which equal the values the CLI used to emit -> behaviour-identical.
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
        # bridge: config.py:2308-2310 -- a directed heading source is active
        # when a head-tail model is selected OR pose extraction is enabled.
        "DIRECTED_ORIENT_POSTHOC_CONSISTENCY": bool(
            str(yolo_headtail_path or "").strip() or pose_extractor_enabled
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
        "IDENTITY_DISAGREE_MIN_RUN": identity_cfg.posthoc.disagree_min_run,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": (
            identity_cfg.posthoc.gates_trajectory_structure
        ),
        "ENABLE_IDENTITY_ANALYSIS": individual_pipeline_enabled,
        "ENABLE_INDIVIDUAL_PIPELINE": individual_pipeline_enabled,
        "IDENTITY_METHOD": identity_method,
        "USE_APRILTAGS": use_apriltags,
        "CNN_CLASSIFIERS": cnn_classifiers,
        "CNN_CLASSIFIER_WINDOW": 10,
        "ENABLE_IDENTITY_IN_TRACKING": identity_cfg.realtime.enabled,
        "ENABLE_IDENTITY_ONLINE_DECODER": identity_cfg.realtime.bayesian_cost_enabled,
        "IDENTITY_POSTPROCESS_MODE": identity_cfg.posthoc.postprocess_mode,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": (
            identity_cfg.posthoc.fragment_solver_enabled
        ),
        "IDENTITY_POSTHOC_ENABLED": identity_cfg.posthoc.enabled,
        "IDENTITY_ENABLE_SMOOTHING": identity_cfg.posthoc.smoothing_enabled,
        "ASSOCIATION_IDENTITY_HINT_SCALE": identity_cfg.realtime.association_weight,
        "IDENTITY_COMMIT_THRESHOLD": identity_cfg.realtime.commit_threshold,
        "IDENTITY_DISPLAY_THRESHOLD": identity_cfg.realtime.display_threshold,
        "IDENTITY_TRANSITION_EPSILON": identity_cfg.realtime.transition_epsilon,
        "IDENTITY_UNKNOWN_PRIOR": identity_cfg.realtime.unknown_prior,
        "IDENTITY_REJOIN_THRESHOLD": identity_cfg.realtime.rejoin_threshold,
        "IDENTITY_SWAP_ENABLED": identity_cfg.realtime.swap_enabled,
        "IDENTITY_SWAP_MIN_FRAMES": identity_cfg.realtime.slot_lock.swap_min_frames,
        "IDENTITY_SWAP_CONF_MARGIN": (identity_cfg.realtime.slot_lock.swap_conf_margin),
        "IDENTITY_REJOIN_VELOCITY_BUDGET": (
            identity_cfg.realtime.slot_lock.rejoin_velocity_budget
        ),
        "IDENTITY_REJOIN_DIST_FLOOR": identity_cfg.realtime.slot_lock.rejoin_dist_floor,
        "IDENTITY_CALIBRATION_REQUIRED": identity_cfg.calibration_required,
        "IDENTITY_CALIBRATION_OVERRIDE": bool(
            _cfg_get(cfg, "identity_calibration_override", default=False)
        ),
        "APRILTAG_FAMILY": str(_cfg_get(cfg, "apriltag_family", default="tag36h11")),
        "APRILTAG_DECIMATE": float(_cfg_get(cfg, "apriltag_decimate", default=1.0)),
        "APRILTAG_CROP_PADDING": float(
            _cfg_get(cfg, "apriltag_crop_padding", default=0.0)
        ),
        "COLOR_TAG_MODEL_PATH": str(_cfg_get(cfg, "color_tag_model_path", default="")),
        "COLOR_TAG_CONFIDENCE": float(
            _cfg_get(cfg, "color_tag_confidence", default=0.5)
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
        "EXPORT_CONFIDENCE_DENSITY_VIDEO": (
            _debug_mode
            if _debug_present
            else bool(_cfg_get(cfg, "export_confidence_density_video", default=False))
        ),
        # --- Dataset generation (bridge: config.py:2367-2399). Active-learning
        # export knobs; inert for tracking. DATASET_NAME/CONF_THRESHOLD are
        # hard-coded in the GUI; the YOLO conf/iou come from advanced_config;
        # the rest are persisted config fields.
        "ENABLE_DATASET_GENERATION": bool(
            _cfg_get(cfg, "enable_dataset_generation", default=False)
        ),
        "DATASET_NAME": "",
        "DATASET_CLASS_NAME": str(_cfg_get(cfg, "dataset_class_name", default="")),
        "DATASET_MAX_FRAMES": int(_cfg_get(cfg, "dataset_max_frames", default=50)),
        "DATASET_CONF_THRESHOLD": 0.5,
        "DATASET_MIN_SELECTION_SCORE": float(
            _cfg_get(cfg, "dataset_min_selection_score", default=0.0)
        ),
        "DATASET_AL_PRESET": str(
            _cfg_get(cfg, "dataset_al_preset", default="tracker_default")
        ),
        "DATASET_YOLO_CONFIDENCE_THRESHOLD": advanced.get(
            "dataset_yolo_confidence_threshold", 0.05
        ),
        "DATASET_YOLO_IOU_THRESHOLD": advanced.get("dataset_yolo_iou_threshold", 0.5),
        "DATASET_DIVERSITY_WINDOW": int(
            _cfg_get(cfg, "dataset_diversity_window", default=10)
        ),
        "DATASET_INCLUDE_CONTEXT": bool(
            _cfg_get(cfg, "dataset_include_context", default=False)
        ),
        "DATASET_PROBABILISTIC_SAMPLING": bool(
            _cfg_get(cfg, "dataset_probabilistic_sampling", default=True)
        ),
        "DATASET_EXPORT_LEVELS": list(
            _cfg_get(cfg, "dataset_export_levels", default=["polygon", "obb", "aabb"])
        ),
        "DATASET_DEDUP_METHOD": str(
            _cfg_get(cfg, "dataset_dedup_method", default="phash")
        ),
        "DATASET_DEDUP_THRESHOLD": int(
            _cfg_get(cfg, "dataset_dedup_threshold", default=8)
        ),
        "DATASET_CLASS_NAMES": _dataset_class_names(cfg),
        # Active-learning metric selectors (bridge: config.py:2394-2399).
        "METRIC_LOW_CONFIDENCE": bool(
            _cfg_get(cfg, "metric_low_confidence", default=True)
        ),
        "METRIC_COUNT_MISMATCH": bool(
            _cfg_get(cfg, "metric_count_mismatch", default=True)
        ),
        "METRIC_FRAGMENTED_DETECTIONS": bool(
            _cfg_get(cfg, "metric_fragmented_detections", default=True)
        ),
        "METRIC_CROWDING": bool(_cfg_get(cfg, "metric_crowding", default=True)),
        "METRIC_HIGH_ASSIGNMENT_COST": bool(
            _cfg_get(cfg, "metric_high_assignment_cost", default=True)
        ),
        "METRIC_TRACK_LOSS": bool(_cfg_get(cfg, "metric_track_loss", default=True)),
        "METRIC_HIGH_UNCERTAINTY": bool(
            _cfg_get(cfg, "metric_high_uncertainty", default=True)
        ),
        # Legacy singular CNN classifier params (bridge: config.py:2408-2415,
        # 2548-2549). See derivation block above the return statement.
        "CNN_CLASSIFIER_MODEL_PATH": cnn_classifier_model_path,
        "CNN_CLASSIFIER_CONFIDENCE": 0.5,
        "CNN_CLASSIFIER_LABEL": "",
        "CNN_CLASSIFIER_BATCH_SIZE": cnn_classifier_batch_size,
        # Final media / individual-crop export (bridge: config.py:2486-2533).
        # ENABLE_INDIVIDUAL_DATASET/IMAGE_SAVE are hard-coded False in the GUI
        # params dict; the export gates + individual-crop knobs come from the
        # persisted config fields (or the session_policy predicate).
        "ENABLE_INDIVIDUAL_DATASET": False,
        "ENABLE_INDIVIDUAL_IMAGE_SAVE": False,
        "EXPORT_FINAL_CANONICAL_IMAGES": export_final_canonical_images,
        "FINAL_MEDIA_EXPORT_VIDEOS_ENABLED": final_media_export_videos_enabled,
        "FINAL_MEDIA_EXPORT_FIX_DIRECTION_FLIPS": bool(
            _cfg_get(cfg, "final_media_export_fix_direction_flips", default=False)
        ),
        "FINAL_MEDIA_EXPORT_HEADING_FLIP_MAX_BURST": int(
            _cfg_get(cfg, "final_media_export_heading_flip_burst", default=5)
        ),
        "FINAL_MEDIA_EXPORT_ENABLE_AFFINE_STABILIZATION": bool(
            _cfg_get(
                cfg, "final_media_export_enable_affine_stabilization", default=False
            )
        ),
        "FINAL_MEDIA_EXPORT_STABILIZATION_WINDOW": int(
            _cfg_get(cfg, "final_media_export_stabilization_window", default=5)
        ),
        "INDIVIDUAL_OUTPUT_FORMAT": str(
            _cfg_get(cfg, "individual_output_format", default="png")
        ),
        "INDIVIDUAL_SAVE_INTERVAL": int(
            _cfg_get(cfg, "individual_save_interval", default=1)
        ),
        "INDIVIDUAL_INTERPOLATE_OCCLUSIONS": bool(
            _cfg_get(cfg, "individual_interpolate_occlusions", default=True)
        ),
        "INDIVIDUAL_BACKGROUND_COLOR": [
            int(component)
            for component in _cfg_get(
                cfg, "individual_background_color", default=[0, 0, 0]
            )
        ],
        "SUPPRESS_FOREIGN_OBB_REGIONS": bool(
            _cfg_get(cfg, "suppress_foreign_obb_regions", default=False)
        ),
        "SUPPRESS_FOREIGN_OBB_DATASET": bool(
            _cfg_get(cfg, "suppress_foreign_obb_individual_dataset", default=False)
        ),
        "SUPPRESS_FOREIGN_OBB_ORIENTED_VIDEO": bool(
            _cfg_get(cfg, "suppress_foreign_obb_oriented_videos", default=False)
        ),
        # Profiling toggle (bridge: config.py:2544).
        "ENABLE_PROFILING": (
            _debug_mode
            if _debug_present
            else bool(_cfg_get(cfg, "enable_profiling", default=False))
        ),
    }

    # Runtime-overlay output-dir / cache keys. These depend on the live
    # video/session (not the config), so they travel on the RuntimeContext.
    # The GUI supplies them (its ``get_parameters_dict`` has always emitted
    # them); the CLI leaves the fields ``None`` and does NOT emit these keys --
    # byte-identical to today's CLI, whose consumers read them via ``.get(...)``
    # (absent == None). Each is emitted only when the caller supplied it.
    _overlay_dirs = {
        "DATASET_OUTPUT_DIR": runtime.dataset_output_dir,
        "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR": runtime.final_media_video_output_dir,
        "INDIVIDUAL_DATASET_OUTPUT_DIR": runtime.individual_dataset_output_dir,
        "INDIVIDUAL_DATASET_NAME": runtime.individual_dataset_name,
        "INDIVIDUAL_PROPERTIES_CACHE_PATH": runtime.individual_properties_cache_path,
    }
    caller_supplied_output_context = any(
        value is not None for value in _overlay_dirs.values()
    )
    for key, value in _overlay_dirs.items():
        if value is not None:
            params[key] = value
    # INDIVIDUAL_DATASET_RUN_ID: the GUI ALWAYS emits this key (its value is
    # ``None`` until a run starts), so emit it whenever the caller supplied the
    # live output-context fields (i.e. the GUI). The CLI leaves them all
    # ``None`` and does not emit the key -- byte-identical to today's CLI; its
    # sole consumer (dataset generator ``.get(...)``) treats absent == None.
    if caller_supplied_output_context:
        params["INDIVIDUAL_DATASET_RUN_ID"] = runtime.individual_dataset_run_id

    return params
