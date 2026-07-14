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
        "OBB_COMPUTE_RUNTIME": str(
            context.get("obb_compute_runtime", context.get("compute_runtime", "cpu"))
        ),
        "YOLO_MODEL_PATH": resolve_model_path(context.get("yolo_model_path", "")),
        "YOLO_OBB_MODE": str(context.get("yolo_obb_mode", "direct")).strip().lower(),
        "YOLO_OBB_DIRECT_TASK": str(context.get("yolo_obb_direct_task", "obb"))
        .strip()
        .lower(),
        "YOLO_OBB_FIXED_ANGLE_DEG": float(context.get("yolo_fixed_angle_deg", 0.0)),
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
        "ENABLE_GPU_BACKGROUND": bool(context.get("enable_gpu_background", False)),
        "MAX_TARGETS": int(context.get("max_targets", 1)),
        "MAX_CONTOUR_MULTIPLIER": float(context.get("max_contour_multiplier", 3.0)),
        "ENABLE_SIZE_FILTERING": bool(use_detection_filters),
        "MIN_OBJECT_SIZE": min_size_px2,
        "MAX_OBJECT_SIZE": max_size_px2,
    }


def _preview_runtime_context_for(compute_runtime: str):
    """Build a minimal ``RuntimeContext`` that maps back to ``compute_runtime``.

    Preview only needs ``RuntimeContext`` to satisfy ``load_headtail_model``'s
    and ``run_headtail``'s signatures (neither reads it for anything other than
    ``runtime_to_compute_runtime`` at load time) -- so we synthesize a context
    whose fields round-trip through ``runtime_to_compute_runtime`` back to the
    requested runtime string, rather than constructing a full ``InferenceConfig``.
    """
    from hydra_suite.core.inference.runtime import RuntimeContext

    rt = str(compute_runtime).strip().lower()
    if rt == "cuda":
        return RuntimeContext(
            cuda_mode=True,
            device="cuda:0",
            use_nvdec=False,
            default_runtime="cuda",
            tensor_on_cuda=True,
            coreml_mode=False,
        )
    if rt == "tensorrt":
        return RuntimeContext(
            cuda_mode=True,
            device="cuda:0",
            use_nvdec=False,
            default_runtime="cuda",
            tensor_on_cuda=False,
            coreml_mode=False,
        )
    if rt == "coreml":
        return RuntimeContext(
            cuda_mode=False,
            device="mps",
            use_nvdec=False,
            default_runtime="cpu",
            tensor_on_cuda=False,
            coreml_mode=True,
        )
    if rt == "mps":
        return RuntimeContext(
            cuda_mode=False,
            device="mps",
            use_nvdec=False,
            default_runtime="cpu",
            tensor_on_cuda=False,
            coreml_mode=False,
        )
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
        coreml_mode=False,
    )


def _preview_load_obb_executors(yolo_params, obb_compute_runtime, yolo_mode):
    """Load the production OBB executor(s) for the preview's direct/sequential mode."""
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    max_det = max(1, int(yolo_params.get("MAX_TARGETS", 1))) * 2
    if yolo_mode == "sequential":
        detect_model_path = yolo_params.get("YOLO_DETECT_MODEL_PATH")
        if not detect_model_path:
            raise ValueError(
                "Sequential YOLO OBB mode requires YOLO_DETECT_MODEL_PATH."
            )
        detect_executor = load_obb_executor(
            detect_model_path,
            obb_compute_runtime,
            task="detect",
            max_det=max_det,
        )
        crop_obb_model_path = yolo_params.get(
            "YOLO_CROP_OBB_MODEL_PATH"
        ) or yolo_params.get("YOLO_OBB_DIRECT_MODEL_PATH")
        stage2_imgsz = int(yolo_params.get("YOLO_SEQ_STAGE2_IMGSZ", 160))
        obb_executor = load_obb_executor(
            crop_obb_model_path,
            obb_compute_runtime,
            task="obb",
            max_det=max_det,
            imgsz_override=stage2_imgsz if stage2_imgsz > 0 else None,
        )
        return {"mode": "sequential", "detect": detect_executor, "obb": obb_executor}

    obb_model_path = yolo_params.get("YOLO_OBB_DIRECT_MODEL_PATH") or yolo_params.get(
        "YOLO_MODEL_PATH"
    )
    obb_executor = load_obb_executor(
        obb_model_path, obb_compute_runtime, task="obb", max_det=max_det
    )
    return {"mode": "direct", "obb": obb_executor}


def _preview_select_headtail_candidate_indices(
    params,
    raw_meas,
    raw_sizes,
    raw_shapes,
    raw_confidences,
    raw_obb_corners,
    *,
    roi_mask=None,
):
    """Cheap pre-filter selecting raw-detection indices worth sending through head-tail.

    Ported from ``YOLOOBBDetector._select_headtail_candidate_indices`` (which
    only ever depended on ``self.params``): confidence/size/AR/ROI gates only,
    intentionally skipping OBB NMS/IOU suppression so head-tail candidates are a
    superset of ``filter_raw_detections``'s final kept set.
    """
    from hydra_suite.core.detectors._utils import _advanced_config_value

    if not raw_meas:
        return []
    conf_threshold = float(params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
    detect_conf_threshold = float(
        params.get(
            "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD",
            params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
        )
    )
    meas_arr = np.ascontiguousarray(np.asarray(raw_meas, dtype=np.float32))
    sizes_arr = np.ascontiguousarray(np.asarray(raw_sizes, dtype=np.float32))
    shapes_arr = np.ascontiguousarray(np.asarray(raw_shapes, dtype=np.float32))
    conf_arr = np.ascontiguousarray(np.asarray(raw_confidences, dtype=np.float32))

    n = min(len(meas_arr), len(sizes_arr), len(shapes_arr), len(conf_arr))
    if raw_obb_corners:
        n = min(n, len(raw_obb_corners))
    if n <= 0:
        return []

    meas_arr = meas_arr[:n]
    sizes_arr = sizes_arr[:n]
    shapes_arr = shapes_arr[:n]
    conf_arr = conf_arr[:n]

    keep_mask = conf_arr >= conf_threshold

    if params.get("ENABLE_SIZE_FILTERING", False):
        min_size = float(params.get("MIN_OBJECT_SIZE", 0))
        max_size = float(params.get("MAX_OBJECT_SIZE", float("inf")))
        ellipse_area_arr = shapes_arr[:, 0] if shapes_arr.ndim == 2 else sizes_arr
        keep_mask &= (ellipse_area_arr >= min_size) & (ellipse_area_arr <= max_size)

    if _advanced_config_value(params, "enable_aspect_ratio_filtering", False):
        ref_ar = float(_advanced_config_value(params, "reference_aspect_ratio", 2.0))
        min_ar_mult = float(
            _advanced_config_value(params, "min_aspect_ratio_multiplier", 0.5)
        )
        max_ar_mult = float(
            _advanced_config_value(params, "max_aspect_ratio_multiplier", 2.0)
        )
        min_ar = ref_ar * min_ar_mult
        max_ar = ref_ar * max_ar_mult
        ar_arr = shapes_arr[:, 1] if shapes_arr.ndim == 2 else np.ones(len(sizes_arr))
        keep_mask &= (ar_arr >= min_ar) & (ar_arr <= max_ar)

    if roi_mask is not None and len(meas_arr) > 0:
        h, w = roi_mask.shape[:2]
        cx = meas_arr[:, 0].astype(np.int32)
        cy = meas_arr[:, 1].astype(np.int32)
        in_bounds = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        cx_safe = np.clip(cx, 0, max(0, w - 1))
        cy_safe = np.clip(cy, 0, max(0, h - 1))
        in_roi = roi_mask[cy_safe, cx_safe] > 0
        keep_mask &= in_bounds & in_roi

    if detect_conf_threshold > 0.0:
        keep_mask &= conf_arr >= detect_conf_threshold

    return [int(idx) for idx in np.flatnonzero(keep_mask)]


def _preview_load_headtail_model(yolo_params):
    """Load a real head-tail classifier for preview via the production stage loader.

    Returns ``None`` when no head-tail model is configured (or loading fails,
    logged as a warning so preview degrades to undirected detections rather
    than crashing), otherwise ``(model, config, runtime)``.
    """
    from hydra_suite.core.detectors._utils import _advanced_config_value

    headtail_path = str(yolo_params.get("YOLO_HEADTAIL_MODEL_PATH", "") or "").strip()
    if not headtail_path:
        return None

    from hydra_suite.core.inference.config import HeadTailConfig
    from hydra_suite.core.inference.stages.headtail import load_headtail_model

    ref_ar = float(_advanced_config_value(yolo_params, "reference_aspect_ratio", 2.0))
    margin = float(
        _advanced_config_value(yolo_params, "yolo_headtail_canonical_margin", 1.3)
    )
    ht_config = HeadTailConfig(
        model_path=headtail_path,
        confidence_threshold=float(
            yolo_params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.50)
        ),
        batch_size=max(1, int(yolo_params.get("HEADTAIL_BATCH_SIZE", 64))),
        canonical_aspect_ratio=ref_ar,
        canonical_margin=margin,
    )
    runtime = _preview_runtime_context_for(
        str(yolo_params.get("HEADTAIL_COMPUTE_RUNTIME", "cpu"))
    )
    try:
        model = load_headtail_model(ht_config, runtime)
    except Exception as exc:
        logger.warning("Preview head-tail model failed to load: %s", exc)
        return None
    return model, ht_config, runtime


def _preview_run_headtail(
    headtail_state,
    frame_to_process,
    raw_meas,
    raw_sizes,
    raw_shapes,
    raw_confidences,
    raw_obb_corners,
    yolo_params,
    roi_mask=None,
):
    """Run the real head-tail model on a cheap-prefiltered candidate subset.

    Preserves the legacy ordering: cheap candidate pre-filter -> head-tail
    model call on the raw candidate set -> caller's later ``filter_raw_detections``
    trims hints down to the surviving detections.
    """
    total = len(raw_meas)
    heading_hints = [float("nan")] * total
    heading_confidences = [0.0] * total
    directed_mask = [0] * total

    if headtail_state is None or not raw_meas:
        return heading_hints, heading_confidences, directed_mask

    candidate_indices = _preview_select_headtail_candidate_indices(
        yolo_params,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        roi_mask=roi_mask,
    )
    if not candidate_indices:
        return heading_hints, heading_confidences, directed_mask

    from hydra_suite.core.detectors._utils import _advanced_config_value
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.stages.headtail import run_headtail

    model, ht_config, runtime = headtail_state
    sel_centroids = np.asarray(
        [np.asarray(raw_meas[i], dtype=np.float32)[:2] for i in candidate_indices],
        dtype=np.float32,
    )
    sel_angles = np.asarray(
        [np.asarray(raw_meas[i], dtype=np.float32)[2] for i in candidate_indices],
        dtype=np.float32,
    )
    sel_sizes = np.asarray([raw_sizes[i] for i in candidate_indices], dtype=np.float32)
    sel_shapes = np.asarray(
        [raw_shapes[i] for i in candidate_indices], dtype=np.float32
    )
    sel_conf = np.asarray(
        [raw_confidences[i] for i in candidate_indices], dtype=np.float32
    )
    sel_corners = np.asarray(
        [raw_obb_corners[i] for i in candidate_indices], dtype=np.float32
    )
    subset = OBBResult(
        frame_idx=0,
        centroids=sel_centroids,
        angles=sel_angles,
        sizes=sel_sizes,
        shapes=sel_shapes,
        confidences=sel_conf,
        corners=sel_corners,
        detection_ids=OBBResult.make_detection_ids(0, len(candidate_indices)),
    )
    ref_ar = float(_advanced_config_value(yolo_params, "reference_aspect_ratio", 2.0))
    margin = float(
        _advanced_config_value(yolo_params, "yolo_headtail_canonical_margin", 1.3)
    )
    result = run_headtail(
        frame_to_process,
        subset,
        model,
        ht_config,
        runtime,
        aspect_ratio=ref_ar,
        margin=margin,
    )
    for slot, raw_idx in enumerate(candidate_indices):
        heading_hints[raw_idx] = float(result.heading_hints[slot])
        heading_confidences[raw_idx] = float(result.heading_confidences[slot])
        directed_mask[raw_idx] = int(result.directed_mask[slot])
    return heading_hints, heading_confidences, directed_mask


def _preview_run_direct_raw_detection(
    extractor, executor, frame_to_process, target_classes, raw_conf_floor, max_det
):
    """Direct-mode raw OBB extraction via a production ``load_obb_executor`` result."""
    results = executor.predict(
        [frame_to_process],
        conf=raw_conf_floor,
        iou=1.0,
        classes=target_classes,
        max_det=max_det,
        verbose=False,
    )
    if not results:
        return [], [], [], [], [], [], None
    result0 = results[0]
    if getattr(result0, "obb", None) is None or len(result0.obb) == 0:
        return [], [], [], [], [], [], result0
    (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
    ) = extractor._extract_raw_detections(result0.obb, return_class_ids=True)
    return (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
        result0,
    )


class _PreviewSeqCropSpec:
    """Minimal stand-in for ``OBBSequentialConfig`` fields ``_build_crops`` reads."""

    def __init__(self, crop_pad_ratio, min_crop_size_px, enforce_square_crop):
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_size_px = min_crop_size_px
        self.enforce_square_crop = enforce_square_crop


def _preview_accumulate_crop_detections(
    extractor, stage2_results, crop_offsets, crop_original_sizes, predict_imgsz
):
    """Merge per-crop stage-2 detections back into full-frame coordinates.

    Mirrors legacy ``YOLOOBBDetector._seq_accumulate_crop_detections``.
    """
    merged_meas, merged_sizes, merged_shapes = [], [], []
    merged_conf, merged_corners, merged_class_ids = [], [], []
    n = min(len(stage2_results), len(crop_offsets), len(crop_original_sizes))
    for i in range(n):
        result = stage2_results[i]
        x0, y0 = crop_offsets[i]
        if (
            result is None
            or getattr(result, "obb", None) is None
            or len(result.obb) == 0
        ):
            continue
        (
            crop_meas,
            crop_sizes,
            crop_shapes,
            crop_conf,
            crop_corners,
            crop_class_ids,
        ) = extractor._extract_raw_detections(result.obb, return_class_ids=True)
        if not crop_meas:
            continue
        if predict_imgsz:
            orig_w, orig_h = crop_original_sizes[i]
            sx = orig_w / float(predict_imgsz)
            sy = orig_h / float(predict_imgsz)
        else:
            sx, sy = 1.0, 1.0
        for j in range(len(crop_meas)):
            m = np.asarray(crop_meas[j], dtype=np.float32).copy()
            m[0] = m[0] * np.float32(sx) + np.float32(x0)
            m[1] = m[1] * np.float32(sy) + np.float32(y0)
            c = np.asarray(crop_corners[j], dtype=np.float32).copy()
            c[:, 0] = c[:, 0] * np.float32(sx) + np.float32(x0)
            c[:, 1] = c[:, 1] * np.float32(sy) + np.float32(y0)
            merged_meas.append(m)
            merged_sizes.append(float(crop_sizes[j]) * sx * sy)
            merged_shapes.append(tuple(crop_shapes[j]))
            merged_conf.append(float(crop_conf[j]))
            merged_corners.append(c)
            merged_class_ids.append(int(crop_class_ids[j]))
    return (
        merged_meas,
        merged_sizes,
        merged_shapes,
        merged_conf,
        merged_corners,
        merged_class_ids,
    )


def _preview_sort_merged_detections(merged, max_det):
    (
        merged_meas,
        merged_sizes,
        merged_shapes,
        merged_conf,
        merged_corners,
        merged_class_ids,
    ) = merged
    if not merged_meas:
        return [], [], [], [], [], []
    conf_arr = np.asarray(merged_conf, dtype=np.float32)
    order = np.argsort(conf_arr)[::-1]
    if len(order) > max_det:
        order = order[:max_det]
    return (
        [merged_meas[i] for i in order],
        [merged_sizes[i] for i in order],
        [merged_shapes[i] for i in order],
        [merged_conf[i] for i in order],
        [merged_corners[i] for i in order],
        [merged_class_ids[i] for i in order],
    )


def _preview_run_sequential_raw_detection(
    extractor,
    executors,
    frame_to_process,
    yolo_params,
    raw_conf_floor,
    target_classes,
    max_det,
):
    """Sequential-mode raw OBB extraction: stage-1 detect + stage-2 crop-OBB.

    Uses production ``load_obb_executor`` results plus the same crop-building
    helper ``core/inference/stages/obb.py`` uses (``_build_crops``/
    ``_resize_crops_for_stage2``), mirroring legacy
    ``YOLOOBBDetector._run_sequential_raw_detection``'s CPU-numpy path.
    """
    from hydra_suite.core.inference.stages.obb import (
        _build_crops,
        _resize_crops_for_stage2,
    )

    detect_executor = executors.get("detect")
    obb_executor = executors.get("obb")
    if detect_executor is None or obb_executor is None:
        return [], [], [], [], [], [], None

    seq_detect_conf = max(
        1e-4, float(yolo_params.get("YOLO_SEQ_DETECT_CONF_THRESHOLD", raw_conf_floor))
    )
    detect_target_classes = yolo_params.get(
        "YOLO_DETECT_TARGET_CLASSES", target_classes
    )
    detect_results = detect_executor.predict(
        [frame_to_process],
        conf=seq_detect_conf,
        iou=1.0,
        classes=detect_target_classes,
        max_det=max_det,
        verbose=False,
    )
    if not detect_results:
        return [], [], [], [], [], [], None
    det0 = detect_results[0]
    boxes = getattr(det0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], [], [], [], [], [], det0

    # NOTE: unlike legacy ``YOLOOBBDetector._run_sequential_raw_detection``
    # (which explicitly did ``np.argsort(det_conf)[::-1][:max_det]`` before
    # building crops), ``boxes`` here is used as-is, un-re-sorted. This is
    # intentional and verified, not an oversight: ``detect_executor``'s
    # ``.predict()`` for both backends is routed through the same shared
    # ultralytics NMS entry point, which already guarantees
    # confidence-descending, ``max_det``-capped output --
    #   * torch cpu/mps/cuda (``_load_torch_executor``) and CoreML (which
    #     also loads via ``_load_torch_model``) call
    #     ``ultralytics.utils.nms.non_max_suppression`` (see
    #     ``ultralytics/utils/nms.py``): NMS keep-indices are built from
    #     ``scores.argsort(descending=True)`` (``TorchNMS.nms``/
    #     ``fast_nms``) or ``torchvision.ops.nms`` (also score-descending
    #     per its docs), then capped via ``i = i[:max_det]`` (nms.py:157) --
    #     so the surviving order is already confidence-descending and capped.
    #   * the direct TensorRT/ONNX executor
    #     (``core/detectors/_direct_obb_runtime.py``, ``_postprocess``,
    #     ~line 266) calls that *exact same*
    #     ``ultralytics.utils.nms.non_max_suppression(..., max_det=max_det)``
    #     function -- byte-identical sort/cap guarantee, no separate
    #     TensorRT-side NMS implementation to diverge.
    # This matches current production ``_run_sequential``
    # (``core/inference/stages/obb.py``), which also builds crops directly
    # off stage-1 ``boxes`` with no explicit re-sort, for the same reason.
    seq_spec = _PreviewSeqCropSpec(
        crop_pad_ratio=float(yolo_params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15)),
        min_crop_size_px=float(yolo_params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64)),
        enforce_square_crop=bool(yolo_params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True)),
    )
    crops, crop_offsets = _build_crops(frame_to_process, boxes, seq_spec, None)
    if not crops:
        return [], [], [], [], [], [], det0

    crop_original_sizes = [(c.shape[1], c.shape[0]) for c in crops]
    stage2_imgsz = int(yolo_params.get("YOLO_SEQ_STAGE2_IMGSZ", 160))
    crops_for_stage2 = (
        _resize_crops_for_stage2(crops, stage2_imgsz) if stage2_imgsz > 0 else crops
    )
    predict_imgsz = stage2_imgsz if stage2_imgsz > 0 else None

    individual_batch_size = max(
        1, int(yolo_params.get("YOLO_SEQ_INDIVIDUAL_BATCH_SIZE", 16))
    )
    stage2_results = []
    for start in range(0, len(crops_for_stage2), individual_batch_size):
        chunk = crops_for_stage2[start : start + individual_batch_size]
        chunk_results = obb_executor.predict(
            chunk,
            conf=raw_conf_floor,
            iou=1.0,
            classes=target_classes,
            max_det=max_det,
            verbose=False,
            imgsz=predict_imgsz,
        )
        stage2_results.extend(list(chunk_results)[: len(chunk)])

    merged = _preview_accumulate_crop_detections(
        extractor, stage2_results, crop_offsets, crop_original_sizes, predict_imgsz
    )
    (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
    ) = _preview_sort_merged_detections(merged, max_det)
    return (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
        det0,
    )


def _preview_run_yolo_raw_detection(
    executors, frame_to_process, yolo_params, headtail_state=None, extractor=None
):
    """Run raw OBB detection (direct or sequential) via production OBB executors.

    Returns the same 10-tuple contract the legacy detector-backed function
    returned, so downstream filtering/annotation code needs no changes.

    ``extractor``: an optional pre-built ``DetectionFilter`` instance to reuse
    (the caller may already need one for ``filter_raw_detections``); a fresh
    one is constructed when omitted.
    """
    raw_conf_floor = max(
        1e-4, float(yolo_params.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3))
    )
    target_classes = yolo_params.get("YOLO_TARGET_CLASSES")
    max_det = max(1, int(yolo_params.get("MAX_TARGETS", 1))) * 2
    yolo_mode = str(yolo_params.get("YOLO_OBB_MODE", "direct")).strip().lower()

    if extractor is None:
        from hydra_suite.core.detectors import DetectionFilter

        extractor = DetectionFilter(yolo_params)

    if yolo_mode == "sequential":
        (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            raw_class_ids,
            stage1_result,
        ) = _preview_run_sequential_raw_detection(
            extractor,
            executors,
            frame_to_process,
            yolo_params,
            raw_conf_floor,
            target_classes,
            max_det,
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
        ) = _preview_run_direct_raw_detection(
            extractor,
            executors["obb"],
            frame_to_process,
            target_classes,
            raw_conf_floor,
            max_det,
        )

    if raw_meas:
        raw_heading_hints, raw_heading_confidences, raw_directed_mask = (
            _preview_run_headtail(
                headtail_state,
                frame_to_process,
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                yolo_params,
            )
        )
    else:
        raw_heading_hints, raw_heading_confidences, raw_directed_mask = [], [], []

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
    test_frame,
    detect_model_names,
    stage1_result,
    filtered_obb_corners,
    detected_dimensions,
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
    detect_names = detect_model_names or _normalize_preview_model_names(
        getattr(stage1_result, "names", None)
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
        from hydra_suite.core.identity.pose.utils import load_skeleton_from_json
        from hydra_suite.runtime.resolver import (
            detect_platform,
            resolve_compute_runtime,
        )

        backend_family = str(context.get("pose_model_type", "yolo")).strip().lower()
        model_path = str(context.get("pose_model_dir", ""))
        min_valid_conf = float(context.get("pose_min_kpt_conf_valid", 0.2))
        batch_size = int(context.get("pose_batch_size", 4))
        skeleton_file = str(context.get("pose_skeleton_file", ""))
        keypoint_names, skeleton_edges = load_skeleton_from_json(skeleton_file)

        # Mirror core/inference/stages/pose.py::load_pose_model's compute_runtime
        # derivation: prefer the tier-based resolver (same authority the real
        # tracking pipeline uses) when a runtime tier is available in the
        # preview context, falling back to the already-resolved runtime string
        # for older/headless callers that only supply "compute_runtime".
        stage = "yolo_pose" if backend_family == "yolo" else "sleap_pose"
        tier = str(context.get("runtime_tier", "")).strip().lower()
        if tier in ("cpu", "gpu", "gpu_fast"):
            compute_runtime = resolve_compute_runtime(
                tier, detect_platform(), stage=stage
            )
        else:
            compute_runtime = str(context.get("compute_runtime", "cpu"))

        if backend_family == "yolo":
            from hydra_suite.core.identity.pose.backends.yolo import YoloNativeBackend

            device = (
                "cuda:0"
                if compute_runtime in ("cuda", "onnx_cuda", "tensorrt")
                else ("mps" if compute_runtime in ("mps", "coreml") else "cpu")
            )
            pose_backend = YoloNativeBackend(
                model_path=model_path,
                device=device,
                min_valid_conf=min_valid_conf,
                keypoint_names=keypoint_names if keypoint_names else None,
                batch_size=batch_size,
            )
        else:
            from hydra_suite.core.identity.pose.api import (
                create_pose_backend_from_config,
            )
            from hydra_suite.core.identity.pose.types import PoseRuntimeConfig

            # Debug/A-B override kept for parity with load_pose_model: lets us
            # force a SLEAP runtime flavor independent of the resolved tier.
            _flavor_override = os.environ.get("HYDRA_SLEAP_FLAVOR", "").strip().lower()
            if _flavor_override:
                runtime_flavor = _flavor_override
                device = "cpu" if _flavor_override == "onnx_cpu" else "cuda"
            elif compute_runtime in ("cuda", "onnx_cuda"):
                runtime_flavor = "onnx_cuda"
                device = "cuda"
            elif compute_runtime in ("mps", "coreml"):
                runtime_flavor = "native"
                device = "mps"
            elif compute_runtime == "tensorrt":
                runtime_flavor = "tensorrt"
                device = "cuda"
            else:
                runtime_flavor = "onnx_cpu"
                device = "cpu"

            video_path = str(context.get("video_path", "") or "").strip()
            out_root = (
                str(Path(video_path).expanduser().resolve().parent)
                if video_path
                else os.getcwd()
            )
            pose_config = PoseRuntimeConfig(
                backend_family="sleap",
                runtime_flavor=runtime_flavor,
                device=device,
                batch_size=max(1, batch_size),
                model_path=model_path,
                out_root=out_root,
                min_valid_conf=min_valid_conf,
                sleap_env=str(context.get("pose_sleap_env", "sleap")),
                sleap_device=device,
                sleap_batch=max(1, batch_size),
                sleap_max_instances=int(context.get("pose_sleap_max_instances", 1)),
                keypoint_names=list(keypoint_names),
                skeleton_edges=skeleton_edges,
            )
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


def _preview_format_cnn_prediction(label_name: str, prediction) -> str:
    """Return a preview-friendly label for flat or multihead CNN predictions."""
    factor_names = tuple(getattr(prediction, "factor_names", ()) or ())
    class_names = tuple(getattr(prediction, "class_names", ()) or ())
    confidences = tuple(getattr(prediction, "confidences", ()) or ())

    if len(factor_names) <= 1:
        pred_label = str(getattr(prediction, "class_name", None) or "?")
        pred_conf = float(getattr(prediction, "confidence", 0.0))
        return f"{label_name}: {pred_label} {pred_conf:.2f}"

    parts = []
    for idx, factor_name in enumerate(factor_names):
        factor_label = (
            str(class_names[idx])
            if idx < len(class_names) and class_names[idx] is not None
            else "unknown"
        )
        factor_conf = float(confidences[idx]) if idx < len(confidences) else 0.0
        parts.append(f"{factor_name}={factor_label} {factor_conf:.2f}")
    return f"{label_name}: " + " | ".join(parts)


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
                model_path = str(resolve_model_path(cnn_cfg.get("model_path", "")))
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
                    label_stacks[det_idx].append(
                        _preview_format_cnn_prediction(label_name, prediction)
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
        from hydra_suite.core.tracking.pose.pose_pipeline import (
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
    from hydra_suite.core.detectors import DetectionFilter

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
    yolo_mode = str(yolo_params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    obb_compute_runtime = str(yolo_params.get("OBB_COMPUTE_RUNTIME", "cpu"))
    executors = _preview_load_obb_executors(yolo_params, obb_compute_runtime, yolo_mode)
    headtail_state = _preview_load_headtail_model(yolo_params)
    extractor = DetectionFilter(yolo_params)

    try:
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
        ) = _preview_run_yolo_raw_detection(
            executors,
            frame_to_process,
            yolo_params,
            headtail_state,
            extractor=extractor,
        )

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
        ) = extractor.filter_raw_detections(
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
            getattr(executors["obb"], "names", None)
            or getattr(getattr(executors["obb"], "model", None), "names", None)
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
            detect_names = _normalize_preview_model_names(
                getattr(executors.get("detect"), "names", None)
            )
            detected_dimensions = _preview_yolo_sequential_stage1_viz(
                test_frame,
                detect_names,
                stage1_result,
                filtered_obb_corners,
                detected_dimensions,
            )

        filtered_corners = [
            np.asarray(c, dtype=np.float32) for c in filtered_obb_corners
        ]
        label_stacks = [[] for _ in range(len(filtered_corners))]
        label_anchors = []
        for corners in filtered_corners:
            major_axis = float(np.linalg.norm(corners[1] - corners[0]))
            minor_axis = float(np.linalg.norm(corners[2] - corners[1]))
            if major_axis < minor_axis:
                major_axis, minor_axis = minor_axis, major_axis
            detected_dimensions.append((major_axis, minor_axis))
            label_anchors.append(_preview_label_anchor(corners, test_frame.shape))

        (
            canonical_crops,
            canonical_inverses,
            crop_padding,
            bg_color,
            suppress_foreign,
        ) = _preview_compute_canonical_crops(
            filtered_corners, frame_to_process, context
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
    finally:
        if headtail_state is not None:
            model, _ht_config, _runtime = headtail_state
            try:
                model.backend.close()
            except Exception:
                pass


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
