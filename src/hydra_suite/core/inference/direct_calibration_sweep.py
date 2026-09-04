"""Build calibration configs from params and replay cached detections offline.

Two config types are in play and they are NOT interchangeable:
``merge_per_frame`` takes an ``OBBConfig`` (obb.py:1476) while
``filter_for_source`` takes the whole ``InferenceConfig`` (filtering.py:339).
This module always holds the ``InferenceConfig`` and passes ``.obb`` where an
``OBBConfig`` is wanted.
"""

from hydra_suite.core.inference.config import InferenceConfig, build_obb_only_config


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
