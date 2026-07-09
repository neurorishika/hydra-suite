"""
Shared pose inference runtime API for MAT + PoseKit.

Centralizes backend selection and runtime behavior while keeping
the calling surface small and stable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydra_suite.core.identity.pose.backends.sleap import (
    SleapExportedBackend,
    SleapServiceBackend,
    auto_export_sleap_model,
    looks_like_sleap_export_path,
)
from hydra_suite.core.identity.pose.backends.yolo import (
    YoloNativeBackend,
    auto_export_yolo_model,
)
from hydra_suite.core.identity.pose.types import PoseInferenceBackend, PoseRuntimeConfig
from hydra_suite.core.identity.pose.utils import (
    normalize_runtime_flavor,
    parse_runtime_request,
)

logger = logging.getLogger(__name__)


def create_pose_backend_from_config(config: PoseRuntimeConfig) -> PoseInferenceBackend:
    backend_family = str(config.backend_family or "yolo").strip().lower()
    requested_runtime = str(config.runtime_flavor or "auto").strip().lower()
    parsed_runtime, parsed_device = parse_runtime_request(requested_runtime)
    runtime_flavor = normalize_runtime_flavor(backend_family, requested_runtime)
    effective_device = (
        parsed_device
        if parsed_device
        else (
            str(config.sleap_device or "auto")
            if backend_family == "sleap"
            else str(config.device or "auto")
        )
    )
    if parsed_runtime == "auto":
        logger.info(
            "Pose runtime auto-selected for %s backend: %s",
            backend_family,
            runtime_flavor,
        )

    if backend_family == "yolo":
        model_candidate = str(config.model_path).strip()
        model_candidate_path = (
            Path(model_candidate).expanduser().resolve() if model_candidate else None
        )
        if model_candidate_path is not None and model_candidate_path.exists():
            if model_candidate_path.is_dir():
                raise RuntimeError(
                    "POSE_MODEL_TYPE is set to 'yolo' but POSE_MODEL_DIR points to a "
                    "directory. This looks like a SLEAP model directory. "
                    "Set POSE_MODEL_TYPE='sleap' for this model, or select a YOLO model "
                    "file (.pt/.onnx/.engine)."
                )
            valid_yolo_suffixes = {".pt", ".onnx", ".engine", ".trt"}
            if model_candidate_path.suffix.lower() not in valid_yolo_suffixes:
                raise RuntimeError(
                    "Unsupported YOLO model path for pose inference: "
                    f"{model_candidate_path}. Expected one of: "
                    ".pt, .onnx, .engine, .trt"
                )
        if runtime_flavor in ("onnx", "tensorrt"):
            try:
                model_candidate = auto_export_yolo_model(
                    config, runtime_flavor, runtime_device=effective_device
                )
            except Exception as exc:
                logger.warning(
                    "YOLO %s runtime initialization failed (%s). Falling back to native runtime.",
                    runtime_flavor,
                    exc,
                )
                runtime_flavor = "native"
                model_candidate = str(config.model_path).strip()
        if not model_candidate:
            raise RuntimeError("Pose model path is empty.")
        return YoloNativeBackend(
            model_path=model_candidate,
            device=effective_device,
            min_valid_conf=config.min_valid_conf,
            keypoint_names=config.keypoint_names,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            max_det=config.yolo_max_det,
            batch_size=config.yolo_batch,
        )

    if backend_family == "sleap":
        if not config.keypoint_names:
            raise RuntimeError(
                "SLEAP backend requires keypoint_names (from skeleton JSON or override)."
            )

        exported_candidate = ""
        if runtime_flavor in ("onnx", "tensorrt"):
            try:
                requested_export = str(config.exported_model_path or "").strip()
                if requested_export and looks_like_sleap_export_path(
                    requested_export,
                    runtime_flavor,
                ):
                    exported_candidate = str(
                        Path(requested_export).expanduser().resolve()
                    )
                else:
                    exported_candidate = auto_export_sleap_model(config, runtime_flavor)
                logger.info(
                    "SLEAP model exported for %s runtime: %s",
                    runtime_flavor,
                    exported_candidate,
                )
                return SleapExportedBackend(
                    exported_model_path=exported_candidate,
                    runtime_flavor=runtime_flavor,
                    runtime_request=requested_runtime,
                    device=effective_device,
                    keypoint_names=config.keypoint_names,
                    min_valid_conf=config.min_valid_conf,
                    batch_size=config.sleap_batch,
                    export_input_hw=config.sleap_export_input_hw,
                )
            except Exception as exc:
                logger.warning(
                    "SLEAP exported runtime (%s) initialization failed: %s. "
                    "Falling back to SLEAP service backend with native runtime.",
                    runtime_flavor,
                    exc,
                )
                runtime_flavor = "native"
                exported_candidate = ""

        service_backend = SleapServiceBackend(
            model_dir=config.model_path,
            out_root=config.out_root,
            keypoint_names=config.keypoint_names,
            min_valid_conf=config.min_valid_conf,
            sleap_env=config.sleap_env,
            sleap_device=effective_device,
            sleap_batch=config.sleap_batch,
            sleap_max_instances=max(1, int(config.sleap_max_instances)),
            skeleton_edges=config.skeleton_edges,
            runtime_flavor=runtime_flavor,
            exported_model_path=exported_candidate,
            export_input_hw=config.sleap_export_input_hw,
        )
        return service_backend

    raise RuntimeError(f"Unsupported pose backend family: {backend_family}")
