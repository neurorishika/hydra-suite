"""PyTorch-only prediction helpers for DetectKit overlays."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def _resolve_torch_device(device_preference: str) -> str:
    """Map a high-level device preference to an Ultralytics-friendly device string."""
    pref = str(device_preference or "auto").strip().lower()

    if pref.startswith("cuda"):
        return pref if ":" in pref else "cuda:0"
    if pref == "mps":
        return "mps"
    if pref == "cpu":
        return "cpu"

    try:
        from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE
    except Exception:
        return "cpu"

    if TORCH_CUDA_AVAILABLE:
        return "cuda:0"
    if MPS_AVAILABLE:
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def _get_torch_model(model_path: str, device: str):
    """Load and cache an Ultralytics YOLO model on the requested torch device."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    try:
        model.to(device)
    except Exception:
        logger.warning(
            "Could not move YOLO model to device %s; falling back to default.",
            device,
            exc_info=True,
        )
    return model


def _detections_from_result(result) -> list[dict[str, object]]:
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return []

    xyxyxyxy = getattr(obb, "xyxyxyxy", None)
    confs = getattr(obb, "conf", None)
    cls = getattr(obb, "cls", None)
    if xyxyxyxy is None or confs is None or cls is None:
        return []

    polygons = xyxyxyxy.detach().cpu().numpy()
    confidences = confs.detach().cpu().numpy()
    class_ids = cls.detach().cpu().numpy()

    detections: list[dict[str, object]] = []
    for poly, confidence, class_id in zip(polygons, confidences, class_ids):
        detections.append(
            {
                "class_id": max(0, int(class_id)),
                "polygon_px": [
                    (float(point[0]), float(point[1])) for point in poly[:4]
                ],
                "confidence": float(confidence),
            }
        )
    return detections


def predict_preview_detections(
    image_path: str,
    model_path: str,
    *,
    device_preference: str = "auto",
    confidence_threshold: float = 0.5,
) -> list[dict[str, object]]:
    """Run one-image OBB preview inference (PyTorch only) and return canvas-ready detections."""
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read preview image: {image_path}")

    device = _resolve_torch_device(device_preference)
    model = _get_torch_model(resolved_model_path, device)
    raw_floor = max(1e-4, float(confidence_threshold))
    results = model.predict(
        source=frame,
        device=device,
        conf=raw_floor,
        verbose=False,
    )
    if not results:
        return []
    return _detections_from_result(results[0])


def predict_preview_detections_for_image(
    model,
    image_path: str,
    *,
    device: str,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    """Run inference using a pre-loaded model on a single image. For batch reuse."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    raw_floor = max(1e-4, float(confidence_threshold))
    results = model.predict(
        source=frame,
        device=device,
        conf=raw_floor,
        verbose=False,
    )
    if not results:
        return []
    return _detections_from_result(results[0])


def load_torch_model(model_path: str, device_preference: str = "auto"):
    """Public wrapper around the cached PyTorch YOLO model loader."""
    resolved = str(Path(model_path).expanduser().resolve())
    device = _resolve_torch_device(device_preference)
    return _get_torch_model(resolved, device), device
