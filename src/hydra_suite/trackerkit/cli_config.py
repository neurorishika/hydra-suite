"""Pure TrackerKit CLI config/session helpers without MainWindow state.

The engine-parameter derivation itself (``build_tracking_parameters``'s
former body) now lives in the Qt-free ``engine_params`` module as
``build_engine_params``/``RuntimeContext``; this module keeps the
session/IO-level helpers (video probing, config loading, output-path
defaults) and a thin shim that adapts them onto the shared builder. Moved
helpers are re-exported below (under their original private names) for any
other module that still imports them from here.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2

from .engine_params import _autopick_greedy  # noqa: F401  (back-compat re-export)
from .engine_params import _cfg_get  # noqa: F401  (back-compat re-export)
from .engine_params import _cfg_get_time  # noqa: F401  (back-compat re-export)
from .engine_params import _coerce_int_list  # noqa: F401  (back-compat re-export)
from .engine_params import (  # noqa: F401  (back-compat re-export)
    RuntimeContext,
    _coerce_pose_keypoint_tokens,
    build_engine_params,
)
from .engine_params import build_roi_mask as _build_roi_mask  # noqa: F401
from .engine_params import (  # noqa: F401  (back-compat re-export)
    legacy_detection_runtime_fields,
)

logger = logging.getLogger(__name__)


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


def build_tracking_parameters(
    cfg: Mapping[str, Any],
    *,
    video_probe: TrackerCliVideoProbe,
    advanced_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate saved TrackerKit JSON config into worker params.

    Thin shim: builds a ``RuntimeContext`` from ``video_probe`` (the CLI's
    own video-metadata probe) and delegates to the shared, Qt-free
    ``build_engine_params``. ``roi_mask`` is left ``None`` on the context so
    ``build_engine_params`` falls back to rasterizing ``cfg["roi_shapes"]``
    itself -- exactly what this function did inline before extraction. The
    output-dir ``RuntimeContext`` fields are left at their ``None`` defaults
    because the CLI does not derive/emit those keys today.
    """
    runtime = RuntimeContext(
        fps=video_probe.fps,
        total_frames=video_probe.total_frames,
        frame_width=video_probe.width,
        frame_height=video_probe.height,
    )
    return build_engine_params(cfg, runtime=runtime, advanced_config=advanced_config)


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
