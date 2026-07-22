"""PreviewDetectionWorker — non-blocking single-frame detection preview."""

import hashlib
import logging
import math
import os
import threading
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.config import (
    BgSubConfig,
    InferenceConfig,
    build_inference_config_from_params,
)
from hydra_suite.core.inference.runner import InferenceRunner
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


def _preview_build_bgsub_params(context: dict, use_detection_filters: bool) -> dict:
    """Assemble an UPPER_SNAKE bg-sub param dict for ``BgSubConfig.from_params``.

    Reuses the existing preview bg-sub param mapping and layers on the two
    knobs the InferenceRunner bg-sub stage reads that the raw mapping omits:

    * ``ENABLE_SIZE_FILTERING`` — the ``BackgroundMeasurer`` only applies the
      MIN/MAX_OBJECT_SIZE window when this is set (measure.py:231). Toggling it
      off is how ``use_detection_filters=False`` yields the unfiltered set;
      MIN_CONTOUR_AREA still applies in both modes (measure.py:210), matching
      the old preview loop and production. Aspect-ratio filtering the old
      preview loop did is intentionally dropped — the production measurer has
      no such filter.
    * ``RUNTIME_TIER`` — the sole runtime knob; bg-sub only uses it to pick the
      grayscale/adjustment device, but the config carries it for parity.
    """
    params = _build_preview_background_params(context)
    params["ENABLE_SIZE_FILTERING"] = bool(use_detection_filters)
    _tier = str(context.get("runtime_tier", "") or "").strip().lower()
    params["RUNTIME_TIER"] = _tier if _tier in {"cpu", "gpu", "gpu_fast"} else "cpu"
    return params


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
    """Run bg-sub preview detection through the shared InferenceRunner stage.

    Behaviour matches PRODUCTION bg-sub (worker.py), not the old hand-rolled
    preview loop: the runner primes the background from the video on each call
    (there is no cross-preview background cache anymore), applies lighting
    stabilization, and filters via BackgroundMeasurer. See the plan's Slice 1
    acceptance note.
    """
    frame_to_process, test_frame = _preview_resize_frame(
        frame_bgr, test_frame, resize_f
    )

    params = _preview_build_bgsub_params(context, use_detection_filters)
    cfg = InferenceConfig(
        obb=None,
        bgsub=BgSubConfig.from_params(params),
        runtime_tier=params["RUNTIME_TIER"],
    )

    roi_for_bgsub = context.get("roi_mask")
    if roi_for_bgsub is not None and resize_f < 1.0:
        roi_for_bgsub = cv2.resize(
            roi_for_bgsub,
            (frame_to_process.shape[1], frame_to_process.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    logger.info("Running bg-sub preview via InferenceRunner.run_realtime")

    runner = InferenceRunner(
        cfg, cache_dir=None, video_path=str(context.get("video_path", "")) or None
    )
    try:
        fr = runner.run_realtime(frame_to_process, roi_mask=roi_for_bgsub)

        obb = getattr(fr, "obb", None)
        keep = list(getattr(fr, "filtered_indices", []) or [])
        if obb is None:
            keep = []

        detections = []
        detected_dimensions = []
        for i in keep:
            cx, cy = float(obb.centroids[i, 0]), float(obb.centroids[i, 1])
            corners = np.asarray(obb.corners[i], dtype=np.float32)
            major_axis = float(np.linalg.norm(corners[1] - corners[0]))
            minor_axis = float(np.linalg.norm(corners[2] - corners[1]))
            if major_axis < minor_axis:
                major_axis, minor_axis = minor_axis, major_axis
            ang = float(np.degrees(obb.angles[i]))
            area = float(obb.sizes[i])
            detections.append(((cx, cy), (major_axis, minor_axis), ang, area))
            detected_dimensions.append((major_axis, minor_axis))
            cv2.ellipse(
                test_frame,
                ((int(cx), int(cy)), (int(major_axis), int(minor_axis)), ang),
                (0, 255, 0),
                2,
            )
            cv2.circle(test_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # FG / BG thumbnails come straight from what the stage detected on.
        fg_mask = getattr(fr, "fg_mask", None)
        bg_u8 = getattr(fr, "bg_u8", None)
        if fg_mask is not None:
            small_fg = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
            test_frame[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
                small_fg, cv2.COLOR_GRAY2BGR
            )
        if bg_u8 is not None:
            small_bg = cv2.resize(bg_u8, (0, 0), fx=0.3, fy=0.3)
            bg_bgr = cv2.cvtColor(small_bg, cv2.COLOR_GRAY2BGR)
            test_frame[0 : bg_bgr.shape[0], -bg_bgr.shape[1] :] = bg_bgr

        prime_frames = params.get("BACKGROUND_PRIME_FRAMES", 0)
        cv2.putText(
            test_frame,
            f"Detections: {len(detections)} (BG from {prime_frames} frames)",
            (10, test_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return detections, detected_dimensions, test_frame
    finally:
        try:
            runner.close()
        except Exception:
            pass


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


def _preview_build_inference_params(context, resize_f, use_detection_filters):
    """Assemble an UPPERCASE params dict for ``build_inference_config_from_params``.

    Extends the OBB/head-tail params from :func:`_preview_build_yolo_params`
    with the CNN / pose / AprilTag / runtime keys the structured
    ``InferenceConfig`` builder reads, mapping them off the lowercase preview
    context. Model paths run through :func:`resolve_model_path` so relative
    preview paths resolve like the rest of the preview branch.
    """
    params = _preview_build_yolo_params(context, resize_f, use_detection_filters)

    # Runtime: RUNTIME_TIER is the sole runtime knob (drives backend/device
    # selection in the redesign). The COMPUTE_RUNTIME string family was retired
    # (Runtime Gen-2 FT1); build_inference_config_from_params reads RUNTIME_TIER.
    _tier = str(context.get("runtime_tier", "") or "").strip().lower()
    params["RUNTIME_TIER"] = _tier if _tier in {"cpu", "gpu", "gpu_fast"} else "cpu"

    # CNN classifiers (model paths resolved; existence-gated inside the builder).
    cnn_cfgs = []
    for cnn_cfg in context.get("cnn_classifiers", []) or []:
        cfg = dict(cnn_cfg)
        cfg["model_path"] = str(resolve_model_path(cfg.get("model_path", "")))
        cnn_cfgs.append(cfg)
    params["CNN_CLASSIFIERS"] = cnn_cfgs

    # Pose stage.
    params["ENABLE_POSE_EXTRACTOR"] = bool(context.get("enable_pose_extractor", False))
    params["POSE_MODEL_TYPE"] = str(context.get("pose_model_type", "yolo"))
    _pose_model = str(resolve_model_path(context.get("pose_model_dir", "")))
    params["POSE_MODEL_DIR"] = _pose_model
    params["POSE_SLEAP_MODEL_DIR"] = _pose_model
    params["POSE_YOLO_MODEL_DIR"] = _pose_model
    params["POSE_MODEL_PATH"] = _pose_model
    params["POSE_SKELETON_FILE"] = str(context.get("pose_skeleton_file", "") or "")
    params["POSE_BATCH_SIZE"] = int(context.get("pose_batch_size", 4))
    params["POSE_MIN_KPT_CONF_VALID"] = float(
        context.get("pose_min_kpt_conf_valid", 0.2)
    )
    params["POSE_IGNORE_KEYPOINTS"] = list(
        context.get("pose_ignore_keypoints", []) or []
    )
    params["POSE_DIRECTION_ANTERIOR_KEYPOINTS"] = list(
        context.get("pose_direction_anterior_keypoints", []) or []
    )
    params["POSE_DIRECTION_POSTERIOR_KEYPOINTS"] = list(
        context.get("pose_direction_posterior_keypoints", []) or []
    )

    # Shared crop geometry (pose + AprilTag).
    params["INDIVIDUAL_CROP_PADDING"] = float(
        context.get("individual_crop_padding", 0.1)
    )
    params["SUPPRESS_FOREIGN_OBB_REGIONS"] = bool(
        context.get("suppress_foreign_obb_regions", True)
    )

    # AprilTag stage.
    params["USE_APRILTAGS"] = bool(context.get("use_apriltags", False))
    params["APRILTAG_FAMILY"] = str(context.get("apriltag_family", "tag36h11"))
    params["APRILTAG_DECIMATE"] = float(context.get("apriltag_decimate", 1.0))

    return params


def _preview_aabb_crop_origin(obb, det_idx, padding, frame_shape):
    """Reconstruct the AABB-crop origin ``extract_aabb_crops`` used for ``det_idx``.

    ``run_apriltag`` returns tag corners in CROP-LOCAL coordinates and
    ``FrameResult`` does not carry the crop offsets, so preview re-derives them
    from the OBB corners with the exact geometry ``extract_aabb_crops`` used, to
    place tag outlines back in frame space.
    """
    if obb is None or det_idx < 0 or det_idx >= obb.num_detections:
        return 0, 0
    corners = np.asarray(obb.corners[det_idx], dtype=np.float32)
    x1 = float(corners[:, 0].min())
    y1 = float(corners[:, 1].min())
    x2 = float(corners[:, 0].max())
    y2 = float(corners[:, 1].max())
    bw, bh = x2 - x1, y2 - y1
    pad = padding * max(bw, bh)
    ox1 = max(0, int(x1 - pad))
    oy1 = max(0, int(y1 - pad))
    return ox1, oy1


def _preview_format_cnn_result(label_name: str, det_pred) -> str:
    """Format a preview label from a FrameResult ``CNNDetectionPrediction``.

    Reads the ``(factor_name, class_names, raw_probabilities)`` contract of the
    inference ``CNNResult``, taking the arg-max class per factor. Flat (K=1)
    models render ``"label: class conf"``; multi-head render
    ``"label: f=class c | ..."``.
    """
    factors = list(getattr(det_pred, "factors", []) or [])
    if not factors:
        return f"{label_name}: ?"

    def _top(factor):
        probs = np.asarray(getattr(factor, "raw_probabilities", []), dtype=np.float32)
        names = list(getattr(factor, "class_names", []) or [])
        if probs.size == 0 or not names:
            return "unknown", 0.0
        j = int(np.argmax(probs))
        name = (
            str(names[j]) if 0 <= j < len(names) and names[j] is not None else "unknown"
        )
        return name, float(probs[j])

    if len(factors) == 1:
        name, conf = _top(factors[0])
        return f"{label_name}: {name} {conf:.2f}"
    parts = []
    for factor in factors:
        name, conf = _top(factor)
        parts.append(f"{getattr(factor, 'factor_name', '?')}={name} {conf:.2f}")
    return f"{label_name}: " + " | ".join(parts)


def _preview_run_pose_overlay(
    pose_result, context, label_stacks, pose_keypoints_by_det
):
    """Populate pose label lines + frame-space keypoints from ``FrameResult.pose``.

    ``run_pose`` returns image-space keypoints (already inverted out of the
    canonical crop), so they are drawn directly. Per-detection stats
    (mean_conf, valid/total) are recomputed from the keypoint confidences.
    """
    if pose_result is None:
        return
    keypoints = getattr(pose_result, "keypoints", None)
    valid_mask = getattr(pose_result, "valid_mask", None)
    if keypoints is None:
        return
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[0] == 0:
        return
    min_valid_conf = float(context.get("pose_min_kpt_conf_valid", 0.2))
    for i in range(min(kp.shape[0], len(label_stacks))):
        det_kp = kp[i]
        num_keypoints = int(det_kp.shape[0])
        if num_keypoints == 0:
            continue
        confs = det_kp[:, 2]
        num_valid = int(np.count_nonzero(confs >= min_valid_conf))
        is_valid = (
            bool(valid_mask[i])
            if valid_mask is not None and i < len(valid_mask)
            else num_valid > 0
        )
        if not is_valid and num_valid == 0:
            continue
        mean_conf = float(np.mean(confs))
        label_stacks[i].append(f"pose {mean_conf:.2f} {num_valid}/{num_keypoints}")
        pose_keypoints_by_det[i] = det_kp


def _preview_run_cnn_overlay(cnn_results, label_stacks):
    """Populate CNN label lines from ``FrameResult.cnn`` (one CNNResult per phase)."""
    for cnn_result in cnn_results or []:
        label_name = str(getattr(cnn_result, "label", "cnn"))
        for det_pred in getattr(cnn_result, "predictions", []) or []:
            det_idx = int(getattr(det_pred, "det_index", -1))
            if 0 <= det_idx < len(label_stacks):
                label_stacks[det_idx].append(
                    _preview_format_cnn_result(label_name, det_pred)
                )


def _preview_run_apriltag_overlay(
    apriltag_result, obb, context, label_stacks, test_frame
):
    """Draw AprilTag outlines + labels from ``FrameResult.apriltag``.

    Corners come back crop-local from ``run_apriltag``; they are offset back
    into frame space via the reconstructed AABB-crop origin for the tag's
    detection.
    """
    if apriltag_result is None:
        return
    tag_ids = list(getattr(apriltag_result, "tag_ids", []) or [])
    if not tag_ids:
        return
    det_indices = list(getattr(apriltag_result, "det_indices", []) or [])
    corners = np.asarray(
        getattr(apriltag_result, "corners", np.zeros((0, 4, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    apriltag_color = (0, 165, 255)
    crop_padding = float(context.get("individual_crop_padding", 0.1))
    tags_by_det = defaultdict(list)
    for t, tag_id in enumerate(tag_ids):
        det_idx = int(det_indices[t]) if t < len(det_indices) else -1
        tags_by_det[det_idx].append(int(tag_id))
        if t >= len(corners):
            continue
        ox1, oy1 = _preview_aabb_crop_origin(
            obb, det_idx, crop_padding, test_frame.shape
        )
        frame_corners = (corners[t] + np.asarray([ox1, oy1], dtype=np.float32)).astype(
            np.int32
        )
        cv2.polylines(
            test_frame,
            [frame_corners],
            isClosed=True,
            color=apriltag_color,
            thickness=2,
        )
        _draw_preview_label_stack(
            test_frame,
            (
                int(np.max(frame_corners[:, 0]) + 6),
                int(np.min(frame_corners[:, 1]) + 14),
            ),
            [f"tag {int(tag_id)}"],
            apriltag_color,
            font_scale=0.4,
        )
    for det_idx, ids in tags_by_det.items():
        if 0 <= det_idx < len(label_stacks):
            unique_ids = ",".join(str(i) for i in sorted(set(ids)))
            label_stacks[det_idx].append(f"tag {unique_ids}")


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
    frame_to_process, test_frame = _preview_resize_frame(
        frame_bgr, test_frame, resize_f
    )
    params = _preview_build_inference_params(context, resize_f, use_detection_filters)

    roi_for_yolo = context.get("roi_mask")
    if roi_for_yolo is not None and resize_f < 1.0:
        roi_for_yolo = cv2.resize(
            roi_for_yolo,
            (frame_to_process.shape[1], frame_to_process.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    logger.info(
        "Running YOLO preview via InferenceRunner.run_realtime "
        "(conf=%.2f, iou=%.2f)",
        float(params.get("YOLO_CONFIDENCE_THRESHOLD", 0.5)),
        float(params.get("YOLO_IOU_THRESHOLD", 0.45)),
    )
    if str(params.get("YOLO_OBB_MODE", "direct")).strip().lower() == "sequential":
        logger.info(
            "Preview: sequential stage-1 detect-box overlay is no longer shown "
            "(run_realtime does not expose the internal stage-1 boxes)."
        )

    cfg = build_inference_config_from_params(params)
    runner = InferenceRunner(cfg)
    try:
        fr = runner.run_realtime(frame_to_process, roi_mask=roi_for_yolo)

        obb = getattr(fr, "obb", None)
        num_dets = obb.num_detections if obb is not None else 0

        filtered_corners = [
            np.asarray(obb.corners[i], dtype=np.float32) for i in range(num_dets)
        ]
        detection_confidences = [float(obb.confidences[i]) for i in range(num_dets)]
        names = runner.obb_class_names or {}
        class_ids = obb.class_ids_or_zeros if obb is not None else []
        filtered_class_labels = [
            _preview_class_label(names, int(class_ids[i])) for i in range(num_dets)
        ]

        ht = getattr(fr, "headtail", None)
        filtered_headtail = []
        for i in range(num_dets):
            if ht is not None:
                heading = float(ht.heading_hints[i])
                ht_conf = float(ht.heading_confidences[i])
                directed = int(ht.directed_mask[i])
            else:
                heading, ht_conf, directed = float("nan"), 0.0, 0
            filtered_headtail.append((heading, ht_conf, directed))

        detected_dimensions = []
        label_anchors = []
        label_stacks = [[] for _ in range(num_dets)]
        for corners in filtered_corners:
            major_axis = float(np.linalg.norm(corners[1] - corners[0]))
            minor_axis = float(np.linalg.norm(corners[2] - corners[1]))
            if major_axis < minor_axis:
                major_axis, minor_axis = minor_axis, major_axis
            detected_dimensions.append((major_axis, minor_axis))
            label_anchors.append(_preview_label_anchor(corners, test_frame.shape))

        pose_keypoints_by_det: dict[int, np.ndarray] = {}
        _preview_run_pose_overlay(
            getattr(fr, "pose", None), context, label_stacks, pose_keypoints_by_det
        )
        _preview_run_cnn_overlay(getattr(fr, "cnn", None) or [], label_stacks)
        _preview_run_apriltag_overlay(
            getattr(fr, "apriltag", None), obb, context, label_stacks, test_frame
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
        _preview_draw_yolo_footer(
            test_frame,
            filtered_corners,
            params,
            context,
            filtered_headtail=filtered_headtail,
        )
        return detected_dimensions, test_frame
    finally:
        try:
            runner.close()
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
