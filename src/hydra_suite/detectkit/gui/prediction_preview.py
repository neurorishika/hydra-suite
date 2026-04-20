"""Prediction preview helpers for DetectKit image overlays."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2

from hydra_suite.core.detectors.yolo_detector import YOLOOBBDetector
from hydra_suite.runtime.compute_runtime import (
    derive_detection_runtime_settings,
    infer_compute_runtime_from_legacy,
)

logger = logging.getLogger(__name__)


def _resolve_preview_runtime(device_preference: str) -> tuple[str, dict[str, object]]:
    preferred = str(device_preference or "auto").strip().lower()

    if preferred.startswith("cuda"):
        return "cuda", {
            "yolo_device": preferred if ":" in preferred else "cuda:0",
            "enable_tensorrt": False,
            "enable_onnx_runtime": False,
        }

    runtime = infer_compute_runtime_from_legacy(
        yolo_device=preferred,
        enable_tensorrt=False,
        pose_runtime_flavor="",
    )
    settings = derive_detection_runtime_settings(runtime)
    return runtime, settings


@lru_cache(maxsize=6)
def _get_preview_detector(
    model_path: str,
    device_preference: str,
) -> YOLOOBBDetector:
    runtime, settings = _resolve_preview_runtime(device_preference)
    params = {
        "YOLO_MODEL_PATH": model_path,
        "YOLO_OBB_DIRECT_MODEL_PATH": model_path,
        "YOLO_OBB_MODE": "direct",
        "YOLO_DEVICE": str(settings.get("yolo_device", "cpu")),
        "COMPUTE_RUNTIME": runtime,
        "ENABLE_TENSORRT": bool(settings.get("enable_tensorrt", False)),
        "ENABLE_ONNX_RUNTIME": bool(settings.get("enable_onnx_runtime", False)),
        "MAX_TARGETS": 64,
    }
    return YOLOOBBDetector(params)


def predict_preview_detections(
    image_path: str,
    model_path: str,
    *,
    device_preference: str = "auto",
    confidence_threshold: float = 0.5,
) -> list[dict[str, object]]:
    """Run one-image OBB preview inference and return canvas-ready detections."""
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read preview image: {image_path}")

    detector = _get_preview_detector(resolved_model_path, device_preference)
    raw_floor = max(1e-4, float(confidence_threshold))
    max_det = max(64, detector._raw_detection_cap())
    results = detector._predict_obb_results(
        frame,
        target_classes=None,
        raw_conf_floor=raw_floor,
        max_det=max_det,
    )
    if not results:
        return []

    result = results[0]
    if result is None or getattr(result, "obb", None) is None or len(result.obb) == 0:
        return []

    (
        _meas,
        _sizes,
        _shapes,
        confidences,
        corners,
        class_ids,
    ) = detector._extract_raw_detections(result.obb, return_class_ids=True)

    detections: list[dict[str, object]] = []
    for polygon, confidence, class_id in zip(corners, confidences, class_ids):
        detections.append(
            {
                "class_id": max(0, int(class_id)),
                "polygon_px": [
                    (float(point[0]), float(point[1])) for point in polygon[:4]
                ],
                "confidence": float(confidence),
            }
        )
    logger.debug(
        "DetectKit preview produced %d detections for %s",
        len(detections),
        image_path,
    )
    return detections
