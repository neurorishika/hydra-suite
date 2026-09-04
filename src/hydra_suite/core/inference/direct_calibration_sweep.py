"""Build calibration configs from params and replay cached detections offline.

Two config types are in play and they are NOT interchangeable:
``merge_per_frame`` takes an ``OBBConfig`` (obb.py:1476) while
``filter_for_source`` takes the whole ``InferenceConfig`` (filtering.py:339).
This module always holds the ``InferenceConfig`` and passes ``.obb`` where an
``OBBConfig`` is wanted.

Production order is merge -> filter, and predict-time confidence is the fixed
``direct.confidence_floor`` (1e-3), independent of the filter threshold -- so
parts collected once are valid across the whole confidence sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydra_suite.core.inference.config import InferenceConfig, build_obb_only_config
from hydra_suite.core.inference.direct_calibration import CalibrationDetection
from hydra_suite.core.inference.stages.filtering import filter_for_source
from hydra_suite.core.inference.stages.obb import (
    _empty_obb_result,
    _RawOBBTensors,
    materialize_tensors,
    merge_per_frame,
)


@dataclass(frozen=True)
class MergeSettings:
    """Cross-tile merge knobs a profile may claim to set.

    ``merge_backend`` is deliberately absent: it is forced to cv2 on every host
    path (obb.py:1571-1581), so sweeping it would produce duplicate rows on
    MPS. The resolved backend is recorded with the measurement instead.
    """

    policy: str = "greedy_nmm"
    metric: str = "ios"
    threshold: float = 0.5


def build_calibration_config(
    model_path: str,
    *,
    slice_params: dict,
    max_targets: int,
    confidence: float,
    runtime_tier: str = "cpu",
    model_task: str = "obb",
) -> InferenceConfig:
    """One production InferenceConfig, built through the shared params mapping.

    Going through ``build_obb_only_config`` (config.py:1151) rather than
    hand-building dataclasses is what guarantees a measured operating point is
    expressible as TrackerKit settings: both sides read the same SLICE_* keys.
    """
    return build_obb_only_config(
        model_path,
        runtime_tier=runtime_tier,
        confidence_threshold=float(confidence),
        max_targets=int(max_targets),
        model_task=model_task,
        extra_params=dict(slice_params),
    )


def config_for_point(
    model_path: str,
    *,
    slice_params: dict,
    merge: MergeSettings,
    confidence: float,
    max_targets: int,
    runtime_tier: str = "cpu",
    model_task: str = "obb",
) -> InferenceConfig:
    """A config for one measured row: geometry + merge + confidence + cap."""
    params = dict(slice_params)
    params.update(
        {
            "SLICE_MERGE_POLICY": merge.policy,
            "SLICE_MERGE_METRIC": merge.metric,
            "SLICE_MERGE_THRESHOLD": float(merge.threshold),
        }
    )
    return build_calibration_config(
        model_path,
        slice_params=params,
        max_targets=max_targets,
        confidence=confidence,
        runtime_tier=runtime_tier,
        model_task=model_task,
    )


def rescore_parts(
    parts, source, inference_config: InferenceConfig, runtime, *, frame_idx: int
):
    """Merge one frame's cached parts, materialize, then filter -- production order."""
    if not parts:
        return _empty_obb_result(frame_idx)
    merged = merge_per_frame(
        parts,
        source.merge_policy,
        source.merge_plan(frame_idx),
        inference_config.obb,
        runtime,
    )
    if isinstance(merged, _RawOBBTensors):
        merged = materialize_tensors(merged, inference_config.obb.raw_detection_cap)
    filtered, _indices = filter_for_source(inference_config, merged, None)
    return filtered


def detections_from_result(result) -> list[CalibrationDetection]:
    """Frame-space calibration records for one post-merge, filtered result."""
    out: list[CalibrationDetection] = []
    polygons = getattr(result, "polygons", None)
    for index in range(int(result.num_detections)):
        polygon = None
        if polygons is not None and polygons[index] is not None:
            polygon = np.asarray(polygons[index], dtype=np.float32).reshape(-1, 2)
        if polygon is None or polygon.shape[0] < 3:
            polygon = np.asarray(result.corners[index], dtype=np.float32).reshape(-1, 2)
        out.append(
            CalibrationDetection(
                class_id=int(result.class_ids_or_zeros[index]),
                polygon_px=polygon,
                confidence=float(result.confidences[index]),
            )
        )
    return out
