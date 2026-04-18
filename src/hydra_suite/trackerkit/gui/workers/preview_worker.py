"""PreviewDetectionWorker — non-blocking single-frame detection preview."""

import hashlib
import logging
import math
import os
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.utils.pose_visualization import is_renderable_pose_keypoint
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level background cache
# ---------------------------------------------------------------------------

_PREVIEW_BACKGROUND_CACHE_MAX_ENTRIES = 4
_PREVIEW_BACKGROUND_CACHE = OrderedDict()
_PREVIEW_BACKGROUND_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Local path helper (avoids circular import from main_window)
# ---------------------------------------------------------------------------


def _get_models_root_directory() -> str:
    """Return user-local models/ root and create it when missing."""
    from hydra_suite.paths import get_models_dir

    return str(get_models_dir())


def resolve_model_path(model_path: object) -> object:
    """
    Resolve a model path to an absolute path.

    If the path is relative, look for it in the models directory.
    If absolute and exists, return as-is.

    Args:
        raw_heading_confidences,
        model_path: Relative or absolute model path

    Returns:
        Absolute path to the model file, or original path if not found
    """
    if not model_path:
        return model_path

    path_str = str(model_path).strip()

    # If already absolute and exists, return it
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    models_root = _get_models_root_directory()
    candidate = os.path.join(models_root, path_str)
    if os.path.exists(candidate):
        return candidate

    # If relative path doesn't exist in models dir, try as-is
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # Return original if nothing works (will fail later with clear error)
    return model_path


# ---------------------------------------------------------------------------
# Region 1: preview background cache helpers
# ---------------------------------------------------------------------------


def _clear_preview_background_cache() -> None:
    """Clear preview-only cached background models."""
    with _PREVIEW_BACKGROUND_CACHE_LOCK:
        _PREVIEW_BACKGROUND_CACHE.clear()


def _hash_preview_roi_mask(roi_mask) -> str | None:
    """Build a stable hash for the preview ROI mask."""
    if roi_mask is None:
        return None

    mask = np.ascontiguousarray(roi_mask)
    digest = hashlib.sha1()
    digest.update(str(mask.shape).encode("ascii"))
    digest.update(str(mask.dtype).encode("ascii"))
    digest.update(memoryview(mask))
    return digest.hexdigest()


def _preview_background_cache_key(context: dict) -> tuple:
    """Return the cache key for preview background priming inputs."""
    return (
        "preview-background-v1",
        os.path.abspath(os.path.expanduser(str(context.get("video_path", "")))),
        int(context.get("bg_prime_frames", 30)),
        int(context.get("brightness", 0)),
        round(float(context.get("contrast", 1.0)), 6),
        round(float(context.get("gamma", 1.0)), 6),
        round(float(context.get("resize_factor", 1.0)), 6),
        _hash_preview_roi_mask(context.get("roi_mask")),
    )


def _preview_object_size_pixels(context: dict, key: str, default: float) -> int:
    """Convert a size-filter multiplier from the context to pixel area."""
    ref = float(context.get("reference_body_size", 20.0))
    rf = float(context.get("resize_factor", 1.0))
    body_area = math.pi * (ref / 2.0) ** 2 * rf**2
    return int(float(context.get(key, default)) * body_area)


def _build_preview_background_params(context: dict) -> dict:
    """Build preview background-subtraction parameters from the frozen context."""
    _fps = float(context.get("fps", 30.0))
    _bg_seconds = float(
        context.get("bg_prime_seconds", context.get("bg_prime_frames", 30) / _fps)
    )
    return {
        "BACKGROUND_PRIME_FRAMES": max(0, round(_bg_seconds * _fps)),
        "BRIGHTNESS": int(context.get("brightness", 0)),
        "CONTRAST": float(context.get("contrast", 1.0)),
        "GAMMA": float(context.get("gamma", 1.0)),
        "ROI_MASK": context.get("roi_mask"),
        "RESIZE_FACTOR": float(context.get("resize_factor", 1.0)),
        "DARK_ON_LIGHT_BACKGROUND": bool(context.get("dark_on_light", False)),
        "THRESHOLD_VALUE": int(context.get("threshold_value", 20)),
        "MORPH_KERNEL_SIZE": int(context.get("morph_kernel_size", 3)),
        "ENABLE_ADDITIONAL_DILATION": bool(
            context.get("enable_additional_dilation", False)
        ),
        "DILATION_KERNEL_SIZE": int(context.get("dilation_kernel_size", 3)),
        "DILATION_ITERATIONS": int(context.get("dilation_iterations", 1)),
        "ENABLE_CONSERVATIVE_SPLIT": bool(
            context.get("enable_conservative_split", True)
        ),
        "CONSERVATIVE_KERNEL_SIZE": int(context.get("conservative_kernel_size", 3)),
        "CONSERVATIVE_ERODE_ITER": int(context.get("conservative_erode_iterations", 1)),
        "MAX_TARGETS": int(context.get("max_targets", 5)),
        "MIN_CONTOUR_AREA": int(context.get("min_contour", 50)),
        "MAX_CONTOUR_MULTIPLIER": int(context.get("max_contour_multiplier", 20)),
        "REFERENCE_BODY_SIZE": float(context.get("reference_body_size", 20.0)),
        "MIN_OBJECT_SIZE": _preview_object_size_pixels(context, "min_object_size", 0.3),
        "MAX_OBJECT_SIZE": _preview_object_size_pixels(context, "max_object_size", 3.0),
    }


def _get_cached_preview_background_state(context: dict) -> dict | None:
    """Return a copy of cached preview background state if available."""
    cache_key = _preview_background_cache_key(context)
    with _PREVIEW_BACKGROUND_CACHE_LOCK:
        cached_state = _PREVIEW_BACKGROUND_CACHE.get(cache_key)
        if cached_state is None:
            return None
        _PREVIEW_BACKGROUND_CACHE.move_to_end(cache_key)
        return {
            "lightest_background": cached_state["lightest_background"].copy(),
            "adaptive_background": cached_state["adaptive_background"].copy(),
            "reference_intensity": cached_state["reference_intensity"],
        }


def _store_preview_background_state(context: dict, bg_model) -> None:
    """Store a copy of preview background state for reuse across previews."""
    if bg_model.lightest_background is None or bg_model.adaptive_background is None:
        return

    cache_key = _preview_background_cache_key(context)
    cache_entry = {
        "lightest_background": bg_model.lightest_background.copy(),
        "adaptive_background": bg_model.adaptive_background.copy(),
        "reference_intensity": bg_model.reference_intensity,
    }

    with _PREVIEW_BACKGROUND_CACHE_LOCK:
        _PREVIEW_BACKGROUND_CACHE[cache_key] = cache_entry
        _PREVIEW_BACKGROUND_CACHE.move_to_end(cache_key)
        while len(_PREVIEW_BACKGROUND_CACHE) > _PREVIEW_BACKGROUND_CACHE_MAX_ENTRIES:
            _PREVIEW_BACKGROUND_CACHE.popitem(last=False)


def _build_preview_background_model(context: dict):
    """Build or restore a preview-only primed background model."""
    from hydra_suite.core.background.model import BackgroundModel

    bg_params = _build_preview_background_params(context)
    bg_model = BackgroundModel(bg_params)

    cached_state = _get_cached_preview_background_state(context)
    if cached_state is not None:
        bg_model.lightest_background = cached_state["lightest_background"]
        bg_model.adaptive_background = cached_state["adaptive_background"]
        bg_model.reference_intensity = cached_state["reference_intensity"]
        logger.info("Reusing cached background model for test detection")
        return bg_model, bg_params

    logger.info("Building background model for test detection...")
    cap = cv2.VideoCapture(str(context.get("video_path", "")))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for background priming")

    try:
        bg_model.prime_background(cap)
    finally:
        cap.release()

    if bg_model.lightest_background is None:
        raise RuntimeError("Failed to build background model")

    _store_preview_background_state(context, bg_model)
    return bg_model, bg_params


# ---------------------------------------------------------------------------
# Region 2: preview rendering helpers + worker + job
# ---------------------------------------------------------------------------


def _normalize_preview_model_names(names) -> dict[int, str]:
    """Normalize Ultralytics model names into an int->label mapping."""
    if isinstance(names, dict):
        out = {}
        for key, value in names.items():
            try:
                out[int(key)] = str(value)
            except Exception:
                continue
        return out
    if isinstance(names, (list, tuple)):
        return {int(i): str(value) for i, value in enumerate(names)}
    return {}


def _preview_class_label(names: dict[int, str], class_id: object) -> str:
    """Return a readable class label for one prediction."""
    try:
        cls_idx = int(class_id)
    except Exception:
        return "cls ?"
    if cls_idx < 0:
        return "cls ?"
    return names.get(cls_idx, f"cls {cls_idx}")


def _preview_label_anchor(
    corners: np.ndarray, image_shape: tuple[int, ...]
) -> tuple[int, int]:
    """Place annotation text just outside an OBB when possible."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    img_h, img_w = image_shape[:2]
    x = int(np.max(pts[:, 0]) + 8)
    y = int(np.min(pts[:, 1]) + 14)
    if x > img_w - 170:
        x = max(4, int(np.min(pts[:, 0]) - 166))
    return max(4, x), int(np.clip(y, 14, max(14, img_h - 6)))


def _draw_preview_label_stack(
    image: np.ndarray,
    anchor_xy: tuple[int, int],
    lines: list[str],
    color: tuple[int, int, int],
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Draw a compact multi-line label block with a solid backing box."""
    text_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not text_lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    gap = 4
    sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines
    ]
    text_w = max((size[0] for size in sizes), default=0)
    text_h = sum(size[1] for size in sizes) + gap * max(0, len(sizes) - 1)
    pad = 4
    x, y = anchor_xy
    img_h, img_w = image.shape[:2]
    x = int(np.clip(x, 0, max(0, img_w - text_w - 2 * pad - 1)))
    top = int(np.clip(y - sizes[0][1], 0, max(0, img_h - text_h - 2 * pad - 1)))
    bottom = min(img_h - 1, top + text_h + 2 * pad)
    right = min(img_w - 1, x + text_w + 2 * pad)
    cv2.rectangle(image, (x, top), (right, bottom), (0, 0, 0), -1)

    cursor_y = top + pad + sizes[0][1]
    for idx, line in enumerate(text_lines):
        cv2.putText(
            image,
            line,
            (x + pad, cursor_y),
            font,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        if idx + 1 < len(text_lines):
            cursor_y += sizes[idx + 1][1] + gap


def _draw_preview_pose_points(
    image: np.ndarray,
    keypoints: object,
    min_valid_conf: float,
    color: tuple[int, int, int] = (255, 0, 255),
) -> None:
    """Render valid pose keypoints directly on the preview image."""
    if keypoints is None:
        return
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return
    for keypoint in arr:
        if not is_renderable_pose_keypoint(
            keypoint[0], keypoint[1], keypoint[2], min_valid_conf
        ):
            continue
        x = int(round(float(keypoint[0])))
        y = int(round(float(keypoint[1])))
        cv2.circle(image, (x, y), 3, color, -1, lineType=cv2.LINE_AA)


class PreviewDetectionWorker(BaseWorker):
    """Worker thread for non-blocking preview detection."""

    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, preview_frame_rgb, context, use_detection_filters) -> None:
        super().__init__()
        self.preview_frame_rgb = preview_frame_rgb
        self.context = context
        self.use_detection_filters = bool(use_detection_filters)

    def execute(self):
        try:
            result = _run_preview_detection_job(
                self.preview_frame_rgb,
                self.context,
                self.use_detection_filters,
            )
            self.finished_signal.emit(result)
        except Exception as exc:
            import traceback

            self.error_signal.emit(f"{exc}\n{traceback.format_exc()}")


def _preview_resize_frame(frame_bgr, test_frame, resize_f):
    if resize_f < 1.0:
        frame_bgr = cv2.resize(
            frame_bgr, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA
        )
        test_frame = cv2.resize(
            test_frame, (0, 0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_AREA
        )
    return frame_bgr, test_frame


def _preview_bg_size_thresholds(context, resize_f, use_detection_filters):
    reference_body_size = float(context.get("reference_body_size", 50.0))
    reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
    scaled_body_area = reference_body_area * (resize_f**2)
    apply_ar = bool(
        use_detection_filters and context.get("enable_aspect_ratio_filtering", False)
    )
    if use_detection_filters:
        min_size_px2 = float(context.get("min_object_size", 0.0)) * scaled_body_area
        max_size_px2 = float(context.get("max_object_size", 999.0)) * scaled_body_area
    else:
        min_size_px2 = 0.0
        max_size_px2 = float("inf")
    ref_ar = float(context.get("reference_aspect_ratio", 2.0))
    min_ar = ref_ar * float(context.get("min_aspect_ratio_multiplier", 0.5))
    max_ar = ref_ar * float(context.get("max_aspect_ratio_multiplier", 2.0))
    return min_size_px2, max_size_px2, apply_ar, min_ar, max_ar


def _preview_run_bg_subtraction(
    frame_bgr, test_frame, context, resize_f, use_detection_filters
):
    from hydra_suite.core.detectors import ObjectDetector
    from hydra_suite.utils.image_processing import apply_image_adjustments

    bg_model, bg_params = _build_preview_background_model(context)
    frame_to_process, test_frame = _preview_resize_frame(
        frame_bgr, test_frame, resize_f
    )

    gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
    gray = apply_image_adjustments(
        gray,
        bg_params["BRIGHTNESS"],
        bg_params["CONTRAST"],
        bg_params["GAMMA"],
        use_gpu=False,
    )

    roi_for_test = bg_params["ROI_MASK"]
    if roi_for_test is not None and resize_f < 1.0:
        roi_for_test = cv2.resize(
            roi_for_test,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Use update_and_get_background to match the production tracking
    # pipeline. tracking_stabilized=False returns the lightest
    # background, which is correct for a single preview frame.
    bg_u8 = bg_model.update_and_get_background(
        gray, roi_mask=None, tracking_stabilized=False
    )
    if bg_u8 is None:
        bg_u8 = cv2.convertScaleAbs(bg_model.lightest_background)
    fg_mask = bg_model.generate_foreground_mask(gray, bg_u8)

    # Apply ROI mask to foreground mask (not to gray) to match the
    # production tracking pipeline in worker.py.
    if roi_for_test is not None:
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_for_test)

    # Apply conservative split to separate merged blobs, matching the
    # production pipeline in worker.py.
    if bg_params.get("ENABLE_CONSERVATIVE_SPLIT", True):
        det = ObjectDetector(bg_params)
        fg_mask = det.apply_conservative_split(fg_mask, gray, bg_u8)

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour = float(context.get("min_contour", 50.0))
    min_size_px2, max_size_px2, apply_ar, min_ar, max_ar = _preview_bg_size_thresholds(
        context, resize_f, use_detection_filters
    )

    detections = []
    detected_dimensions = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_contour or len(c) < 5:
            continue
        (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
        major_axis = max(ax1, ax2)
        minor_axis = min(ax1, ax2)
        aspect_ratio = (
            float(major_axis) / float(minor_axis)
            if minor_axis and float(minor_axis) > 0.0
            else float("inf")
        )
        if use_detection_filters and not (min_size_px2 <= area <= max_size_px2):
            continue
        if apply_ar and not (min_ar <= aspect_ratio <= max_ar):
            continue
        detections.append(((cx, cy), (ax1, ax2), ang, area))
        detected_dimensions.append((major_axis, minor_axis))
        cv2.ellipse(
            test_frame, ((int(cx), int(cy)), (int(ax1), int(ax2)), ang), (0, 255, 0), 2
        )
        cv2.circle(test_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    small_fg = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
    test_frame[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
        small_fg, cv2.COLOR_GRAY2BGR
    )
    small_bg = cv2.resize(bg_u8, (0, 0), fx=0.3, fy=0.3)
    bg_bgr = cv2.cvtColor(small_bg, cv2.COLOR_GRAY2BGR)
    test_frame[0 : bg_bgr.shape[0], -bg_bgr.shape[1] :] = bg_bgr

    cv2.putText(
        test_frame,
        f"Detections: {len(detections)} (BG from {bg_params['BACKGROUND_PRIME_FRAMES']} frames)",
        (10, test_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return detections, detected_dimensions, test_frame


def _preview_build_yolo_params(context, resize_f, use_detection_filters):
    reference_body_size = float(context.get("reference_body_size", 50.0))
    reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
    scaled_body_area = reference_body_area * (resize_f**2)
    if use_detection_filters:
        min_size_px2 = int(
            float(context.get("min_object_size", 0.0)) * scaled_body_area
        )
        max_size_px2 = int(
            float(context.get("max_object_size", 999.0)) * scaled_body_area
        )
    else:
        min_size_px2 = 0
        max_size_px2 = float("inf")
    return {
        "YOLO_MODEL_PATH": resolve_model_path(context.get("yolo_model_path", "")),
        "YOLO_OBB_MODE": str(context.get("yolo_obb_mode", "direct")).strip().lower(),
        "ADVANCED_CONFIG": {
            "reference_aspect_ratio": float(context.get("reference_aspect_ratio", 2.0)),
            "enable_aspect_ratio_filtering": bool(
                use_detection_filters
                and context.get("enable_aspect_ratio_filtering", False)
            ),
            "min_aspect_ratio_multiplier": float(
                context.get("min_aspect_ratio_multiplier", 0.5)
            ),
            "max_aspect_ratio_multiplier": float(
                context.get("max_aspect_ratio_multiplier", 2.0)
            ),
        },
        "YOLO_OBB_DIRECT_MODEL_PATH": resolve_model_path(
            context.get(
                "yolo_obb_direct_model_path", context.get("yolo_model_path", "")
            )
        ),
        "YOLO_DETECT_MODEL_PATH": resolve_model_path(
            context.get("yolo_detect_model_path", "")
        ),
        "YOLO_CROP_OBB_MODEL_PATH": resolve_model_path(
            context.get("yolo_crop_obb_model_path", "")
        ),
        "YOLO_HEADTAIL_MODEL_PATH": resolve_model_path(
            context.get("yolo_headtail_model_path", "")
        ),
        "POSE_OVERRIDES_HEADTAIL": bool(context.get("pose_overrides_headtail", True)),
        "HEADTAIL_COMPUTE_RUNTIME": str(
            context.get("headtail_runtime", context.get("compute_runtime", "cpu"))
        ),
        "HEADTAIL_BATCH_SIZE": int(context.get("headtail_batch_size", 64)),
        "YOLO_SEQ_CROP_PAD_RATIO": float(context.get("yolo_seq_crop_pad_ratio", 0.15)),
        "YOLO_SEQ_MIN_CROP_SIZE_PX": int(context.get("yolo_seq_min_crop_size_px", 64)),
        "YOLO_SEQ_ENFORCE_SQUARE_CROP": bool(
            context.get("yolo_seq_enforce_square_crop", True)
        ),
        "YOLO_SEQ_STAGE2_IMGSZ": int(context.get("yolo_seq_stage2_imgsz", 160)),
        "YOLO_SEQ_INDIVIDUAL_BATCH_SIZE": int(
            context.get("yolo_seq_individual_batch_size", 16)
        ),
        "YOLO_SEQ_STAGE2_POW2_PAD": bool(
            context.get("yolo_seq_stage2_pow2_pad", False)
        ),
        "YOLO_SEQ_DETECT_CONF_THRESHOLD": float(
            context.get("yolo_seq_detect_conf_threshold", 0.25)
        ),
        "YOLO_HEADTAIL_CONF_THRESHOLD": float(
            context.get("yolo_headtail_conf_threshold", 0.50)
        ),
        "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD": float(
            context.get("yolo_headtail_detect_conf_threshold", 0.25)
        ),
        "YOLO_CONFIDENCE_THRESHOLD": float(context.get("yolo_confidence", 0.5)),
        "YOLO_IOU_THRESHOLD": float(context.get("yolo_iou", 0.45)),
        "USE_CUSTOM_OBB_IOU_FILTERING": True,
        "YOLO_TARGET_CLASSES": context.get("yolo_target_classes"),
        "YOLO_DEVICE": context.get("yolo_device"),
        "ENABLE_GPU_BACKGROUND": bool(context.get("enable_gpu_background", False)),
        "ENABLE_TENSORRT": bool(context.get("enable_tensorrt", False)),
        "ENABLE_ONNX_RUNTIME": bool(context.get("enable_onnx_runtime", False)),
        "TENSORRT_MAX_BATCH_SIZE": int(context.get("tensorrt_max_batch_size", 1)),
        "MAX_TARGETS": int(context.get("max_targets", 1)),
        "MAX_CONTOUR_MULTIPLIER": float(context.get("max_contour_multiplier", 3.0)),
        "ENABLE_SIZE_FILTERING": bool(use_detection_filters),
        "MIN_OBJECT_SIZE": min_size_px2,
        "MAX_OBJECT_SIZE": max_size_px2,
    }


def _preview_run_yolo_raw_detection(detector, frame_to_process, yolo_params):

    raw_conf_floor = max(
        1e-4, float(yolo_params.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3))
    )
    yolo_mode = str(yolo_params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if yolo_mode == "sequential":
        (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            raw_class_ids,
            stage1_result,
        ) = detector._run_sequential_raw_detection(
            frame_to_process,
            target_classes=yolo_params.get("YOLO_TARGET_CLASSES"),
            raw_conf_floor=raw_conf_floor,
            max_det=max(1, int(yolo_params.get("MAX_TARGETS", 1))) * 2,
            return_class_ids=True,
        )
    else:
        (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            raw_class_ids,
            stage1_result,
        ) = detector._run_direct_raw_detection(
            frame_to_process,
            target_classes=yolo_params.get("YOLO_TARGET_CLASSES"),
            raw_conf_floor=raw_conf_floor,
            max_det=max(1, int(yolo_params.get("MAX_TARGETS", 1))) * 2,
            return_class_ids=True,
        )
    raw_heading_hints = []
    raw_heading_confidences = []
    raw_directed_mask = []
    if raw_meas:
        candidate_indices = detector._select_headtail_candidate_indices(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
        )
        (
            raw_heading_hints,
            raw_heading_confidences,
            raw_directed_mask,
            _,
        ) = detector._compute_headtail_hints_for_indices(
            frame_to_process,
            raw_obb_corners,
            candidate_indices,
        )
    return (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
        raw_heading_hints,
        raw_heading_confidences,
        raw_directed_mask,
        stage1_result,
    )


def _preview_yolo_sequential_stage1_viz(
    test_frame, detector, stage1_result, filtered_obb_corners, detected_dimensions
):
    detect_color = (255, 200, 0)
    boxes = getattr(stage1_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detected_dimensions
    try:
        det_xyxy = np.ascontiguousarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
        det_conf = np.ascontiguousarray(boxes.conf.cpu().numpy(), dtype=np.float32)
        det_cls = np.ascontiguousarray(boxes.cls.cpu().numpy(), dtype=np.int32)
    except Exception:
        det_xyxy = np.empty((0, 4), dtype=np.float32)
        det_conf = np.empty((0,), dtype=np.float32)
        det_cls = np.empty((0,), dtype=np.int32)
    detect_names = _normalize_preview_model_names(
        getattr(stage1_result, "names", None)
        or getattr(getattr(detector.detect_model, "model", None), "names", None)
        or getattr(detector.detect_model, "names", None)
    )
    for di in range(len(det_xyxy)):
        x1, y1, x2, y2 = [int(v) for v in det_xyxy[di]]
        cv2.rectangle(test_frame, (x1, y1), (x2, y2), detect_color, 1)
        if di < len(det_conf):
            detect_label = _preview_class_label(
                detect_names, det_cls[di] if di < len(det_cls) else -1
            )
            _draw_preview_label_stack(
                test_frame,
                (min(test_frame.shape[1] - 140, x2 + 6), max(14, y1 + 12)),
                [f"det {detect_label} {float(det_conf[di]):.2f}"],
                detect_color,
            )
    # In sequential mode, stage-2 OBB can occasionally yield zero
    # usable detections in preview. Fall back to stage-1 detect box
    # dimensions so body-size auto-set remains functional.
    if len(filtered_obb_corners) == 0 and len(det_xyxy) > 0:
        for di in range(len(det_xyxy)):
            x1f, y1f, x2f, y2f = [float(v) for v in det_xyxy[di]]
            w_box = max(1.0, x2f - x1f)
            h_box = max(1.0, y2f - y1f)
            detected_dimensions.append((max(w_box, h_box), min(w_box, h_box)))
    return detected_dimensions


def _preview_compute_canonical_crops(filtered_corners, frame_to_process, context):
    from hydra_suite.core.canonicalization.crop import (
        compute_native_scale_affine,
        extract_canonical_crop,
    )

    crop_padding = float(context.get("individual_crop_padding", 0.1))
    bg_color = tuple(
        int(v) for v in context.get("individual_background_color", [0, 0, 0])
    )
    suppress_foreign = bool(context.get("suppress_foreign_obb_regions", True))
    ref_aspect_ratio = float(context.get("reference_aspect_ratio", 2.0))
    canonical_crops = [None] * len(filtered_corners)
    canonical_inverses = [None] * len(filtered_corners)
    for i, corners in enumerate(filtered_corners):
        try:
            M_align, canvas_w, canvas_h, _ = compute_native_scale_affine(
                corners, ref_aspect_ratio, crop_padding
            )
            foreign = (
                [other for ci, other in enumerate(filtered_corners) if ci != i]
                if suppress_foreign
                else None
            )
            canonical_crops[i] = extract_canonical_crop(
                frame_to_process,
                M_align,
                canvas_w,
                canvas_h,
                bg_color=bg_color,
                foreign_corners=foreign,
            )
            canonical_inverses[i] = cv2.invertAffineTransform(M_align).astype(
                np.float32
            )
        except Exception:
            canonical_crops[i] = None
            canonical_inverses[i] = None
    return canonical_crops, canonical_inverses, crop_padding, bg_color, suppress_foreign


def _preview_run_pose_overlay(
    filtered_corners,
    canonical_crops,
    canonical_inverses,
    context,
    label_stacks,
    pose_keypoints_by_det,
):
    if not filtered_corners or not bool(context.get("enable_pose_extractor", False)):
        return None
    try:
        from hydra_suite.core.canonicalization.crop import (
            invert_keypoints as _invert_kpts,
        )
        from hydra_suite.core.identity.pose.api import (
            build_runtime_config,
            create_pose_backend_from_config,
        )

        preview_pose_params = {
            "POSE_MODEL_TYPE": str(context.get("pose_model_type", "yolo")),
            "POSE_MODEL_DIR": str(context.get("pose_model_dir", "")),
            "POSE_RUNTIME_FLAVOR": str(context.get("pose_runtime_flavor", "auto")),
            "POSE_SKELETON_FILE": str(context.get("pose_skeleton_file", "")),
            "POSE_MIN_KPT_CONF_VALID": float(
                context.get("pose_min_kpt_conf_valid", 0.2)
            ),
            "POSE_YOLO_BATCH": int(context.get("pose_batch_size", 4)),
            "POSE_BATCH_SIZE": int(context.get("pose_batch_size", 4)),
            "POSE_SLEAP_BATCH": int(context.get("pose_batch_size", 4)),
            "POSE_SLEAP_ENV": str(context.get("pose_sleap_env", "sleap")),
            "POSE_SLEAP_DEVICE": str(context.get("pose_sleap_device", "auto")),
            "COMPUTE_RUNTIME": str(context.get("compute_runtime", "cpu")),
            "YOLO_DEVICE": str(context.get("yolo_device", "cpu")),
        }
        video_path = str(context.get("video_path", "") or "").strip()
        out_root = (
            str(Path(video_path).expanduser().resolve().parent)
            if video_path
            else os.getcwd()
        )
        pose_config = build_runtime_config(preview_pose_params, out_root=out_root)
        pose_backend = create_pose_backend_from_config(pose_config)
        valid_pose_entries = [
            (idx, crop)
            for idx, crop in enumerate(canonical_crops)
            if crop is not None and getattr(crop, "size", 0) > 0
        ]
        if valid_pose_entries:
            pose_results = pose_backend.predict_batch(
                [crop for _, crop in valid_pose_entries]
            )
            for pidx, (det_idx, _crop) in enumerate(valid_pose_entries):
                pose_out = pose_results[pidx] if pidx < len(pose_results) else None
                if pose_out is None:
                    continue
                pose_mean_conf = float(getattr(pose_out, "mean_conf", 0.0))
                pose_num_valid = int(getattr(pose_out, "num_valid", 0))
                pose_num_keypoints = int(getattr(pose_out, "num_keypoints", 0))
                label_stacks[det_idx].append(
                    f"pose {pose_mean_conf:.2f} {pose_num_valid}/{pose_num_keypoints}"
                )
                keypoints = getattr(pose_out, "keypoints", None)
                if (
                    keypoints is not None
                    and canonical_inverses[det_idx] is not None
                    and len(keypoints) > 0
                ):
                    pose_keypoints_by_det[det_idx] = _invert_kpts(
                        np.asarray(keypoints, dtype=np.float32),
                        canonical_inverses[det_idx],
                    ).astype(np.float32)
        return pose_backend
    except Exception as exc:
        logger.warning("Preview pose overlay disabled: %s", exc)
        return None


def _preview_run_cnn_overlay(filtered_corners, canonical_crops, context, label_stacks):
    cnn_cfgs = context.get("cnn_classifiers", []) or []
    if not filtered_corners or not cnn_cfgs:
        return []
    cnn_backends = []
    try:
        from hydra_suite.core.identity.classification.cnn import (
            CNNIdentityBackend,
            CNNIdentityConfig,
        )

        valid_cnn_entries = [
            (idx, crop)
            for idx, crop in enumerate(canonical_crops)
            if crop is not None and getattr(crop, "size", 0) > 0
        ]
        if valid_cnn_entries:
            cnn_crops = [crop for _, crop in valid_cnn_entries]
            for cnn_cfg in cnn_cfgs:
                model_path = str(cnn_cfg.get("model_path", ""))
                if not model_path or not os.path.exists(model_path):
                    continue
                label_name = str(cnn_cfg.get("label", "cnn"))
                backend = CNNIdentityBackend(
                    CNNIdentityConfig(
                        model_path=model_path,
                        confidence=float(cnn_cfg.get("confidence", 0.5)),
                        scoring_mode=str(cnn_cfg.get("scoring_mode", "atomic")),
                        batch_size=int(cnn_cfg.get("batch_size", 64)),
                    ),
                    model_path=model_path,
                    compute_runtime=str(
                        context.get(
                            "cnn_runtime",
                            context.get("compute_runtime", "cpu"),
                        )
                    ),
                )
                cnn_backends.append(backend)
                cnn_predictions = backend.predict_batch(cnn_crops)
                for pidx, (det_idx, _crop) in enumerate(valid_cnn_entries):
                    if pidx >= len(cnn_predictions):
                        continue
                    prediction = cnn_predictions[pidx]
                    pred_label = str(getattr(prediction, "class_name", None) or "?")
                    pred_conf = float(getattr(prediction, "confidence", 0.0))
                    label_stacks[det_idx].append(
                        f"{label_name}: {pred_label} {pred_conf:.2f}"
                    )
    except Exception as exc:
        logger.warning("Preview CNN overlay disabled: %s", exc)
    return cnn_backends


def _preview_run_apriltag_overlay(
    filtered_corners,
    frame_to_process,
    context,
    label_stacks,
    test_frame,
    crop_padding,
    suppress_foreign,
    bg_color,
):
    if not filtered_corners or not bool(context.get("use_apriltags", False)):
        return None
    apriltag_color = (0, 165, 255)
    try:
        from hydra_suite.core.identity.classification.apriltag import (
            AprilTagConfig,
            AprilTagDetector,
        )
        from hydra_suite.core.tracking.pose_pipeline import (
            extract_one_crop as _extract_aabb_crop,
        )

        apriltag_detector = AprilTagDetector(
            AprilTagConfig.from_params(
                {
                    "APRILTAG_FAMILY": context.get("apriltag_family", "tag36h11"),
                    "APRILTAG_DECIMATE": context.get("apriltag_decimate", 1.0),
                    "INDIVIDUAL_CROP_PADDING": crop_padding,
                }
            )
        )
        tag_crops = []
        tag_offsets = []
        tag_det_indices = []
        for det_idx, corners in enumerate(filtered_corners):
            aabb_result = _extract_aabb_crop(
                frame_to_process,
                corners,
                det_idx,
                crop_padding,
                filtered_corners,
                suppress_foreign,
                bg_color,
            )
            if aabb_result is None:
                continue
            crop, offset, mapped_idx = aabb_result
            tag_crops.append(crop)
            tag_offsets.append(offset)
            tag_det_indices.append(mapped_idx)
        if tag_crops:
            tag_obs = apriltag_detector.detect_in_crops(
                tag_crops, tag_offsets, det_indices=tag_det_indices
            )
            tags_by_det = defaultdict(list)
            for obs in tag_obs:
                tags_by_det[int(obs.det_index)].append(int(obs.tag_id))
                tag_corners = np.asarray(obs.corners, dtype=np.int32)
                cv2.polylines(
                    test_frame,
                    [tag_corners],
                    isClosed=True,
                    color=apriltag_color,
                    thickness=2,
                )
                _draw_preview_label_stack(
                    test_frame,
                    (
                        int(np.max(tag_corners[:, 0]) + 6),
                        int(np.min(tag_corners[:, 1]) + 14),
                    ),
                    [f"tag {int(obs.tag_id)}"],
                    apriltag_color,
                    font_scale=0.4,
                )
            for det_idx, tag_ids in tags_by_det.items():
                unique_ids = ",".join(str(tag_id) for tag_id in sorted(set(tag_ids)))
                label_stacks[det_idx].append(f"tag {unique_ids}")
        return apriltag_detector
    except Exception as exc:
        logger.warning("Preview AprilTag overlay disabled: %s", exc)
        return None


def _preview_draw_obb_annotations(
    test_frame,
    filtered_corners,
    detection_confidences,
    filtered_class_labels,
    label_stacks,
    label_anchors,
    pose_keypoints_by_det,
    filtered_headtail,
    context,
):
    obb_color = (0, 255, 255)
    headtail_color = (0, 255, 0)
    pose_color = (255, 0, 255)
    for i, corners in enumerate(filtered_corners):
        corners_int = corners.astype(np.int32)
        cv2.polylines(
            test_frame, [corners_int], isClosed=True, color=obb_color, thickness=2
        )
        cx = int(corners[:, 0].mean())
        cy = int(corners[:, 1].mean())
        cv2.circle(test_frame, (cx, cy), 4, obb_color, -1, lineType=cv2.LINE_AA)
        conf = (
            detection_confidences[i] if i < len(detection_confidences) else float("nan")
        )
        label_lines = []
        if not np.isnan(conf):
            label_lines.append(
                f"{filtered_class_labels[i] if i < len(filtered_class_labels) else 'cls ?'} {float(conf):.2f}"
            )
        else:
            label_lines.append(
                filtered_class_labels[i] if i < len(filtered_class_labels) else "cls ?"
            )
        label_lines.extend(label_stacks[i])
        _draw_preview_label_stack(test_frame, label_anchors[i], label_lines, obb_color)
        if i in pose_keypoints_by_det:
            _draw_preview_pose_points(
                test_frame,
                pose_keypoints_by_det[i],
                float(context.get("pose_min_kpt_conf_valid", 0.2)),
                color=pose_color,
            )
        if i < len(filtered_headtail):
            heading, ht_conf, directed = filtered_headtail[i]
            if int(directed) == 1 and np.isfinite(float(heading)):
                ex = int(cx + 34 * math.cos(float(heading)))
                ey = int(cy + 34 * math.sin(float(heading)))
                cv2.arrowedLine(
                    test_frame, (cx, cy), (ex, ey), headtail_color, 2, tipLength=0.3
                )
                _draw_preview_label_stack(
                    test_frame,
                    (min(test_frame.shape[1] - 90, ex + 6), max(14, ey + 12)),
                    [f"head {float(ht_conf):.2f}"],
                    headtail_color,
                    font_scale=0.4,
                )
            elif float(ht_conf) > 0.0:
                _draw_preview_label_stack(
                    test_frame,
                    (min(test_frame.shape[1] - 120, cx + 6), max(14, cy + 18)),
                    [f"head abstain {float(ht_conf):.2f}"],
                    headtail_color,
                    font_scale=0.4,
                )


def _preview_cleanup_backends(pose_backend, cnn_backends, apriltag_detector):
    try:
        if pose_backend is not None and hasattr(pose_backend, "close"):
            pose_backend.close()
    except Exception:
        pass
    for backend in cnn_backends:
        try:
            backend.close()
        except Exception:
            pass
    try:
        if apriltag_detector is not None:
            apriltag_detector.close()
    except Exception:
        pass


def _preview_draw_yolo_footer(
    test_frame, meas, yolo_params, context, filtered_headtail=None
):
    active_layers = []
    if str(context.get("yolo_headtail_model_path", "")).strip():
        active_layers.append("head-tail")
    if bool(context.get("enable_pose_extractor", False)):
        active_layers.append("pose")
    if bool(context.get("use_apriltags", False)):
        active_layers.append("apriltag")
    if context.get("cnn_classifiers"):
        active_layers.append("cnn")
    footer = f"Detections: {len(meas)} (IOU={yolo_params['YOLO_IOU_THRESHOLD']:.2f})"
    if active_layers:
        footer += f" | preview: {', '.join(active_layers)}"
    configured_headtail = str(
        context.get(
            "configured_headtail_model_path",
            context.get("yolo_headtail_model_path", ""),
        )
        or ""
    ).strip()
    headtail_enabled = bool(
        context.get(
            "headtail_enabled",
            bool(str(context.get("yolo_headtail_model_path", "")).strip()),
        )
    )
    if configured_headtail and not headtail_enabled:
        footer += " | head-tail disabled"
    elif filtered_headtail is not None and len(filtered_headtail) > 0:
        directed_count = sum(
            1
            for heading, _conf, directed in filtered_headtail
            if int(directed) == 1 and np.isfinite(float(heading))
        )
        footer += f" | head-tail {directed_count}/{len(filtered_headtail)} directed"
    cv2.putText(
        test_frame,
        footer,
        (10, test_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def _preview_run_yolo_branch(
    frame_bgr, test_frame, context, resize_f, use_detection_filters
):
    from hydra_suite.core.detectors import YOLOOBBDetector

    frame_to_process, test_frame = _preview_resize_frame(
        frame_bgr, test_frame, resize_f
    )
    yolo_params = _preview_build_yolo_params(context, resize_f, use_detection_filters)

    roi_for_yolo = context.get("roi_mask")
    if roi_for_yolo is not None and resize_f < 1.0:
        roi_for_yolo = cv2.resize(
            roi_for_yolo,
            (frame_to_process.shape[1], frame_to_process.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    logger.info(
        f"Running YOLO detection (conf={yolo_params['YOLO_CONFIDENCE_THRESHOLD']:.2f}, "
        f"iou={yolo_params['YOLO_IOU_THRESHOLD']:.2f})"
    )
    detector = YOLOOBBDetector(yolo_params)
    yolo_mode = str(yolo_params.get("YOLO_OBB_MODE", "direct")).strip().lower()

    (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
        raw_heading_hints,
        raw_heading_confidences,
        raw_directed_mask,
        stage1_result,
    ) = _preview_run_yolo_raw_detection(detector, frame_to_process, yolo_params)

    raw_ids = list(range(len(raw_meas)))
    (
        meas,
        _sizes,
        _shapes,
        detection_confidences,
        filtered_obb_corners,
        filtered_ids,
        filtered_heading_hints,
        filtered_heading_confidences,
        filtered_directed_mask,
    ) = detector.filter_raw_detections(
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        roi_mask=roi_for_yolo,
        detection_ids=raw_ids,
        heading_hints=raw_heading_hints,
        heading_confidences=raw_heading_confidences,
        directed_mask=raw_directed_mask,
    )

    stage2_names = _normalize_preview_model_names(
        getattr(detector.model, "names", None)
        or getattr(getattr(detector.model, "model", None), "names", None)
    )
    filtered_class_labels = []
    for det_id in filtered_ids:
        if 0 <= int(det_id) < len(raw_class_ids):
            filtered_class_labels.append(
                _preview_class_label(stage2_names, raw_class_ids[int(det_id)])
            )
        else:
            filtered_class_labels.append("cls ?")

    filtered_headtail = []
    for idx in range(len(filtered_obb_corners)):
        heading = (
            filtered_heading_hints[idx]
            if idx < len(filtered_heading_hints)
            else float("nan")
        )
        confidence = (
            filtered_heading_confidences[idx]
            if idx < len(filtered_heading_confidences)
            else 0.0
        )
        directed = (
            filtered_directed_mask[idx] if idx < len(filtered_directed_mask) else 0
        )
        filtered_headtail.append((heading, confidence, directed))

    detected_dimensions = []
    if yolo_mode == "sequential" and stage1_result is not None:
        detected_dimensions = _preview_yolo_sequential_stage1_viz(
            test_frame,
            detector,
            stage1_result,
            filtered_obb_corners,
            detected_dimensions,
        )

    filtered_corners = [np.asarray(c, dtype=np.float32) for c in filtered_obb_corners]
    label_stacks = [[] for _ in range(len(filtered_corners))]
    label_anchors = []
    for corners in filtered_corners:
        major_axis = float(np.linalg.norm(corners[1] - corners[0]))
        minor_axis = float(np.linalg.norm(corners[2] - corners[1]))
        if major_axis < minor_axis:
            major_axis, minor_axis = minor_axis, major_axis
        detected_dimensions.append((major_axis, minor_axis))
        label_anchors.append(_preview_label_anchor(corners, test_frame.shape))

    canonical_crops, canonical_inverses, crop_padding, bg_color, suppress_foreign = (
        _preview_compute_canonical_crops(filtered_corners, frame_to_process, context)
    )

    pose_keypoints_by_det = {}
    pose_backend = _preview_run_pose_overlay(
        filtered_corners,
        canonical_crops,
        canonical_inverses,
        context,
        label_stacks,
        pose_keypoints_by_det,
    )
    cnn_backends = _preview_run_cnn_overlay(
        filtered_corners, canonical_crops, context, label_stacks
    )
    apriltag_detector = _preview_run_apriltag_overlay(
        filtered_corners,
        frame_to_process,
        context,
        label_stacks,
        test_frame,
        crop_padding,
        suppress_foreign,
        bg_color,
    )

    _preview_draw_obb_annotations(
        test_frame,
        filtered_corners,
        detection_confidences,
        filtered_class_labels,
        label_stacks,
        label_anchors,
        pose_keypoints_by_det,
        filtered_headtail,
        context,
    )
    _preview_cleanup_backends(pose_backend, cnn_backends, apriltag_detector)
    _preview_draw_yolo_footer(
        test_frame,
        meas,
        yolo_params,
        context,
        filtered_headtail=filtered_headtail,
    )

    return detected_dimensions, test_frame


def _run_preview_detection_job(
    frame_rgb, context: dict, use_detection_filters: bool
) -> dict:
    """Run preview detection using a frozen parameter snapshot."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    detection_method = int(context.get("detection_method", 0))
    is_background_subtraction = detection_method == 0
    resize_f = float(context.get("resize_factor", 1.0))

    test_frame = frame_bgr.copy()

    if is_background_subtraction:
        _detections, detected_dimensions, test_frame = _preview_run_bg_subtraction(
            frame_bgr, test_frame, context, resize_f, use_detection_filters
        )
    else:
        detected_dimensions, test_frame = _preview_run_yolo_branch(
            frame_bgr, test_frame, context, resize_f, use_detection_filters
        )

    return {
        "test_frame_rgb": cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB),
        "resize_factor": resize_f,
        "detected_dimensions": detected_dimensions,
    }
