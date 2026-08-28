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
``YOLO_CROP_OBB_MODEL_PATH``, ``YOLO_SEQ_CROP_PAD_RATIO``, and (Task 10 fix
round) ``YOLO_SEQ_DETECT_CONF_THRESHOLD``.

Stage-1 confidence note: without ``YOLO_SEQ_DETECT_CONF_THRESHOLD`` explicitly
set, ``config.py``'s ``build_obb_only_config`` resolves
``OBBSequentialConfig.detect_confidence_threshold`` to its dataclass default
(``0.25``) regardless of the caller's requested ``confidence_threshold`` --
``extra_params`` is merged via ``dict.setdefault`` (a key this module doesn't
supply is simply never overridden). The retired pre-Task-9 AL detector
closure (``predict_obb_for_frame_sequential``) applied the caller's single
``conf`` to BOTH the stage-1 detect pass and the stage-2 OBB pass, so leaving
stage 1 pinned at 0.25 is a real behavior change from that: a caller
requesting a low ``confidence_threshold`` (e.g. 0.05, typical for AL scoring,
which wants to see everything the model finds) got only detections that also
cleared the unrelated 0.25 stage-1 gate -- 8 filtered detections instead of
the ~44 raw candidates the old per-frame closure would have proposed, on the
``ant_obb_sleap`` fixture at conf=0.05 (measured; see the Task 10 report).
``detect_confidence_threshold`` (below) restores the caller's control over
that gate.
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
    detect_confidence_threshold: float | None = None,
) -> InferenceConfig:
    """Build an OBB-only ``InferenceConfig`` for one AL scoring pass.

    ``kind``/``primary_model_path``/``secondary_model_path`` are exactly the
    tuple returned by ``detectkit_resolve_inference_models``. ``crop_pad_ratio``
    is only meaningful (and required) for ``kind == "sequential"``, matching
    the field DetectKit's own `_load_active_detector_fn` already threads into
    `predict_obb_for_frame_sequential`.

    ``detect_confidence_threshold`` overrides the sequential-mode stage-1
    (detect) confidence gate; it is ignored for ``kind == "obb_direct"``
    (single-stage, no separate detect pass). ``None`` (the default) leaves
    ``build_obb_only_config``'s own default (0.25) in place -- pass the
    caller's actual AL confidence threshold here to match the old per-frame
    detector closure's behavior of applying one ``conf`` to both stages (see
    the module docstring's "Stage-1 confidence note").

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
        extra_params = {
            "YOLO_DETECT_MODEL_PATH": primary_model_path,
            "YOLO_CROP_OBB_MODEL_PATH": secondary_model_path,
            "YOLO_SEQ_CROP_PAD_RATIO": crop_pad_ratio,
        }
        if detect_confidence_threshold is not None:
            extra_params["YOLO_SEQ_DETECT_CONF_THRESHOLD"] = detect_confidence_threshold
        return build_obb_only_config(
            primary_model_path,
            runtime_tier=runtime_tier,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            mode="sequential",
            extra_params=extra_params,
        )
    raise ValueError(f"Unsupported AL detector kind: {kind!r}")
