"""Source-aware inference configuration shared by autotuner cache consumers."""

from __future__ import annotations

from typing import Any

from hydra_suite.core.inference.config import (
    BgSubConfig,
    InferenceConfig,
    build_inference_config_from_params,
    migrate_runtime_to_tier,
)


def inference_config_for_optimizer_params(params: dict[str, Any]) -> InferenceConfig:
    """Build the same detection-source config production tracking uses.

    The general builder owns the YOLO-OBB mapping. Production tracking builds
    background subtraction separately because it has no OBB confidence/IoU
    gate; doing the same here prevents NaN bg-sub confidences from being sent
    through an OBB-only filter.
    """

    detection_method = str(
        params.get("DETECTION_METHOD", "background_subtraction")
    ).lower()
    if detection_method == "yolo_obb":
        return build_inference_config_from_params(params)

    compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
    raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
    runtime_tier = (
        raw_tier
        if raw_tier in {"cpu", "gpu", "gpu_fast"}
        else migrate_runtime_to_tier({compute_runtime})
    )
    return InferenceConfig(
        obb=None,
        bgsub=BgSubConfig.from_params(params),
        runtime_tier=runtime_tier,
        detection_batch_size=int(params.get("DETECTION_BATCH_SIZE", 1) or 1),
    )
