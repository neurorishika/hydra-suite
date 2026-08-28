"""Adapt DetectKit's resolved model info into an InferenceConfig for AL scoring.

`detectkit_resolve_inference_models` (detectkit/gui/project.py) classifies a
project's active model path into a `(kind, primary, secondary)` tuple:

- ``kind == "obb_direct"`` -- ``primary`` is a single OBB-direct checkpoint;
  ``secondary`` is None.
- ``kind == "sequential"`` -- ``primary`` is the stage-1 detect checkpoint;
  ``secondary`` is the stage-2 crop-OBB checkpoint.

This module maps that tuple onto ``build_obb_only_config``
(core/inference/config.py), the same helper TrackerKit's AL export path
already uses to build an OBB-only ``InferenceConfig``. Qt-free.

Note that ``build_obb_only_config``'s ``extra_params`` is merged into a raw
UPPER_SNAKE params dict consumed by ``build_inference_config_from_params`` --
it is NOT a kwargs passthrough to ``OBBSequentialConfig``. For
``mode="sequential"`` the relevant keys are ``YOLO_DETECT_MODEL_PATH``,
``YOLO_CROP_OBB_MODEL_PATH``, and ``YOLO_SEQ_CROP_PAD_RATIO``.
"""

from __future__ import annotations

from hydra_suite.core.inference.config import InferenceConfig, build_obb_only_config


def build_obb_config_for_al(
    kind: str,
    primary_model_path: str,
    secondary_model_path: str | None,
    *,
    crop_pad_ratio: float,
    confidence_threshold: float,
    iou_threshold: float,
    runtime_tier: str | None = None,
) -> InferenceConfig:
    """Build an OBB-only ``InferenceConfig`` for one AL scoring pass.

    ``kind``/``primary_model_path``/``secondary_model_path`` are exactly the
    tuple returned by ``detectkit_resolve_inference_models``. ``crop_pad_ratio``
    is only meaningful (and required) for ``kind == "sequential"``, matching
    the field DetectKit's own `_load_active_detector_fn` already threads into
    `predict_obb_for_frame_sequential`.

    Raises ``ValueError`` for any ``kind`` other than "obb_direct" or
    "sequential" (notably "unknown" -- callers must resolve to a supported
    kind before requesting an AL config), or if ``kind == "sequential"`` and
    ``secondary_model_path`` is missing.
    """
    if kind == "obb_direct":
        return build_obb_only_config(
            primary_model_path,
            runtime_tier=runtime_tier,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            mode="direct",
        )
    if kind == "sequential":
        if not secondary_model_path:
            raise ValueError(
                "AL sequential-mode config requires a secondary (crop-OBB) "
                "model path; got None."
            )
        return build_obb_only_config(
            primary_model_path,
            runtime_tier=runtime_tier,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            mode="sequential",
            extra_params={
                "YOLO_DETECT_MODEL_PATH": primary_model_path,
                "YOLO_CROP_OBB_MODEL_PATH": secondary_model_path,
                "YOLO_SEQ_CROP_PAD_RATIO": crop_pad_ratio,
            },
        )
    raise ValueError(f"Unsupported AL detector kind: {kind!r}")
