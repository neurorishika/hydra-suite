"""Pure, Qt-free policy predicates over the tracking config dict.

Each function answers a runtime-behavior question the GUI previously answered by
reading widgets. The GUI methods now delegate here; the CLI calls them directly.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

import numpy as np


def _truthy(config: Mapping[str, Any], key: str, default: bool = False) -> bool:
    return bool(config.get(key, default))


def is_individual_pipeline_enabled(config: Mapping[str, Any]) -> bool:
    """Individual analysis runs only under YOLO-OBB detection."""
    return str(config.get("detection_method", "")).strip().lower() == "yolo_obb"


def is_pose_export_enabled(config: Mapping[str, Any]) -> bool:
    return is_individual_pipeline_enabled(config) and _truthy(
        config, "enable_pose_extractor"
    )


def is_pose_inference_enabled(config: Mapping[str, Any]) -> bool:
    if not is_pose_export_enabled(config):
        return False
    return bool(str(config.get("pose_model_dir", "") or "").strip())


def is_headtail_compute_enabled(config: Mapping[str, Any]) -> bool:
    if not (
        is_individual_pipeline_enabled(config)
        and _truthy(config, "enable_headtail_orientation")
    ):
        return False
    return bool(str(config.get("yolo_headtail_model_path", "") or "").strip())


def should_export_final_canonical_images(config: Mapping[str, Any]) -> bool:
    return _truthy(
        config, "enable_individual_dataset"
    ) and is_individual_pipeline_enabled(config)


def should_export_final_media_videos(config: Mapping[str, Any]) -> bool:
    return _truthy(
        config, "final_media_export_videos_enabled"
    ) and is_individual_pipeline_enabled(config)


def should_run_interpolated_postpass(config: Mapping[str, Any]) -> bool:
    if not _truthy(config, "individual_interpolate_occlusions", default=True):
        return False
    if not is_individual_pipeline_enabled(config):
        return False
    return (
        should_export_final_canonical_images(config)
        or is_pose_export_enabled(config)
        or should_export_final_media_videos(config)
    )


def workflow_mode_key(config: Mapping[str, Any]) -> str:
    return "realtime" if _truthy(config, "realtime_tracking_mode") else "non_realtime"


def build_trajectory_colors(n: int) -> List[Tuple[int, int, int]]:
    """Deterministic track overlay colors — the single shared implementation.

    Uses the legacy global-seed + randint form so GUI-rendered videos keep their
    existing color baselines; the CLI adopts these exact values. The global RNG
    state is saved before seeding and restored afterward, so this call does not
    leak a fixed seed into unrelated code that also uses ``np.random``.
    """
    state = np.random.get_state()
    try:
        np.random.seed(42)
        return [tuple(int(v) for v in c) for c in np.random.randint(0, 255, (n, 3))]
    finally:
        np.random.set_state(state)
