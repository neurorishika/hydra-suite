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


@lru_cache(maxsize=8)
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


def predict_obb_for_frame(
    model,
    frame,
    *,
    device: str,
    conf: float,
    iou: float = 0.7,
) -> list[tuple[float, float, float, float, float, float]]:
    """Run OBB inference on a single in-memory BGR frame and return
    (cx, cy, w, h, theta_rad, confidence) tuples. For AL detector_fn use.
    """
    raw_floor = max(1e-4, float(conf))
    results = model.predict(
        source=frame,
        device=device,
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
    )
    if not results:
        return []
    obb = getattr(results[0], "obb", None)
    if obb is None or len(obb) == 0:
        return []

    xywhr = getattr(obb, "xywhr", None)
    confs = getattr(obb, "conf", None)
    if xywhr is None or confs is None:
        return []

    rows = xywhr.detach().cpu().numpy()
    confidences = confs.detach().cpu().numpy()
    out: list[tuple[float, float, float, float, float, float]] = []
    for row, c in zip(rows, confidences):
        cx, cy, w, h, theta = (float(v) for v in row[:5])
        out.append((cx, cy, w, h, theta, float(c)))
    return out


def predict_obb_for_frame_sequential(
    detect_model,
    obb_model,
    frame,
    *,
    detect_device: str,
    obb_device: str,
    conf: float,
    iou: float = 0.7,
    crop_pad_ratio: float = 0.15,
) -> list[tuple[float, float, float, float, float, float]]:
    """Two-stage OBB prediction. Stage 1: axis-aligned detection on the full
    frame. Stage 2: oriented-bbox prediction on each detected crop. Crops are
    expanded by `crop_pad_ratio` and clipped to frame bounds. Returns
    (cx, cy, w, h, theta_rad, confidence) tuples in original-frame coordinates.
    """
    import numpy as np

    if frame is None or frame.size == 0:
        return []
    h_img, w_img = frame.shape[:2]

    raw_floor = max(1e-4, float(conf))
    detect_results = detect_model.predict(
        source=frame,
        device=detect_device,
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
    )
    if not detect_results:
        return []
    boxes = getattr(detect_results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    detect_confs = boxes.conf.detach().cpu().numpy()

    out: list[tuple[float, float, float, float, float, float]] = []
    for box, det_conf in zip(xyxy, detect_confs):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * float(crop_pad_ratio)
        pad_y = bh * float(crop_pad_ratio)
        cx_pad_min = max(0, int(round(x1 - pad_x)))
        cy_pad_min = max(0, int(round(y1 - pad_y)))
        cx_pad_max = min(w_img, int(round(x2 + pad_x)))
        cy_pad_max = min(h_img, int(round(y2 + pad_y)))
        if cx_pad_max <= cx_pad_min or cy_pad_max <= cy_pad_min:
            continue
        crop = frame[cy_pad_min:cy_pad_max, cx_pad_min:cx_pad_max]
        if crop.size == 0:
            continue

        obb_results = obb_model.predict(
            source=crop,
            device=obb_device,
            conf=raw_floor,
            iou=float(iou),
            verbose=False,
        )
        if not obb_results:
            continue
        obb = getattr(obb_results[0], "obb", None)
        if obb is None or len(obb) == 0:
            continue
        xywhr = getattr(obb, "xywhr", None)
        confs = getattr(obb, "conf", None)
        if xywhr is None or confs is None:
            continue
        rows = xywhr.detach().cpu().numpy()
        obb_confs = confs.detach().cpu().numpy()
        if len(rows) == 0:
            continue
        # Pick the highest-confidence OBB inside the crop.
        best_idx = int(np.argmax(obb_confs))
        cx_local, cy_local, w_box, h_box, theta = (float(v) for v in rows[best_idx][:5])
        cx_global = cx_local + cx_pad_min
        cy_global = cy_local + cy_pad_min
        # Combined confidence: geometric mean of detect and OBB conf.
        combined_conf = float(np.sqrt(float(det_conf) * float(obb_confs[best_idx])))
        out.append((cx_global, cy_global, w_box, h_box, theta, combined_conf))
    return out


def predict_preview_detections_sequential(
    image_path: str,
    detect_model_path: str,
    obb_model_path: str,
    *,
    device_preference: str = "auto",
    confidence_threshold: float = 0.5,
    crop_pad_ratio: float = 0.15,
) -> list[dict[str, object]]:
    """Two-stage OBB preview inference: detect on full image, OBB on each crop.
    Returns canvas-ready detection dicts with `class_id`, `polygon_px`, `confidence`.
    """
    import numpy as np

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read preview image: {image_path}")

    detect_resolved = str(Path(detect_model_path).expanduser().resolve())
    obb_resolved = str(Path(obb_model_path).expanduser().resolve())
    device = _resolve_torch_device(device_preference)

    detect_model = _get_torch_model(detect_resolved, device)
    obb_model = _get_torch_model(obb_resolved, device)

    tuples = predict_obb_for_frame_sequential(
        detect_model,
        obb_model,
        frame,
        detect_device=device,
        obb_device=device,
        conf=confidence_threshold,
        iou=0.7,
        crop_pad_ratio=crop_pad_ratio,
    )
    out: list[dict[str, object]] = []
    for cx, cy, w, h, theta, conf in tuples:
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        local = np.array(
            [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
            dtype=np.float32,
        )
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        corners = local @ rot.T + np.array([cx, cy], dtype=np.float32)
        out.append(
            {
                "class_id": 0,
                "polygon_px": [(float(p[0]), float(p[1])) for p in corners],
                "confidence": float(conf),
            }
        )
    return out
