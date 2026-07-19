"""Prediction helpers for DetectKit overlays.

These wrap the inference pipeline's public API -- ``load_obb_executor`` plus the
public ``core/inference/stages/obb.py`` helpers (``extract_obb_result``,
``build_crops``, ``resize_crops_for_stage2``, ``merge_obb_results``) -- so the
preview path gains every production runtime (torch cpu/mps/cuda plus
ONNX/TensorRT/CoreML) and real per-detection ``class_id`` from the model's OBB
head, instead of hand-parsing raw ultralytics ``Results`` objects.

The mirror for this wiring is
``trackerkit/gui/dialogs/model_test_dialog.py`` (the Quick Test dialog): it uses
the exact same ``load_obb_executor(...) -> executor.predict(...) ->
extract_obb_result(...)`` flow.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

from hydra_suite.core.inference.runtime_artifacts import load_obb_executor
from hydra_suite.core.inference.stages.obb import (
    build_crops,
    extract_obb_result,
    merge_obb_results,
    resize_crops_for_stage2,
)

logger = logging.getLogger(__name__)

#: IOU threshold for the preview's NMS pass. Matches the legacy ultralytics
#: default that ``predict_obb_for_frame`` previously relied on.
_PREVIEW_IOU = 0.7

#: Max detections fed to the executor's NMS. Matches the legacy YOLO OBB path's
#: ``YOLO_MAX_TARGETS`` (100) -- the executor's own default (20) is too low for
#: busy multi-animal scenes and would silently hide real detections.
_PREVIEW_MAX_DET = 100


class _SeqCropSpec:
    """Minimal stand-in for the ``OBBSequentialConfig`` fields ``build_crops`` reads.

    Mirrors ``model_test_dialog._SeqCropSpec``. Preview inference has no full
    config object, so we supply the crop-geometry fields ``build_crops`` needs
    (production defaults: padded, enforced-square crops).
    """

    def __init__(
        self,
        crop_pad_ratio: float = 0.15,
        min_crop_size_px: float = 64.0,
        enforce_square_crop: bool = True,
    ) -> None:
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_size_px = min_crop_size_px
        self.enforce_square_crop = enforce_square_crop


def _resolve_torch_device(device_preference: str) -> str:
    """Map a high-level device preference to a torch-style device string."""
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


def _resolve_compute_runtime(device_preference: str) -> str:
    """Map a device preference onto a ``load_obb_executor`` compute-runtime string.

    DetectKit's device selector only exposes auto/cpu/mps/cuda, so this always
    resolves to one of the torch runtimes (``cpu``/``mps``/``cuda``). Those load
    a plain ultralytics model (no export), which is why a preview click can
    never trigger a slow TensorRT/CoreML build. Should a gpu_fast tier ever be
    surfaced here, ``auto_export=False`` (below) keeps the click cheap: a
    missing artifact surfaces as an error rather than a multi-minute export.
    """
    device = _resolve_torch_device(device_preference)
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


@lru_cache(maxsize=8)
def _get_torch_model(model_path: str, compute_runtime: str, task: str = "obb") -> Any:
    """Load and cache an OBB executor for a model path + compute runtime.

    Replaces the old raw-ultralytics ``YOLO(...)`` loader. Executors are
    reusable, so the ``lru_cache`` is preserved. ``auto_export=False`` keeps a
    preview interaction from kicking off a slow TensorRT/CoreML export -- a
    missing artifact is surfaced as an error instead (torch runtimes ignore the
    flag). ``task`` is forwarded so a sequential stage-1 detector is parsed as
    a plain detector under gpu_fast runtimes (ignored by the torch runtimes,
    which already know their own task).
    """
    return load_obb_executor(
        model_path,
        compute_runtime,
        auto_export=False,
        max_det=_PREVIEW_MAX_DET,
        task=task,
    )


def _dicts_from_obb_result(obb: Any) -> list[dict[str, object]]:
    """Build canvas-ready detection dicts from an ``OBBResult``.

    This is where DetectKit finally gets real per-detection class ids: they come
    straight from ``obb.class_ids_or_zeros`` (the model's OBB head), not a
    hardcoded ``0``.
    """
    detections: list[dict[str, object]] = []
    class_ids = obb.class_ids_or_zeros
    for i in range(obb.num_detections):
        detections.append(
            {
                "class_id": int(class_ids[i]),
                "polygon_px": [(float(x), float(y)) for (x, y) in obb.corners[i]],
                "confidence": float(obb.confidences[i]),
            }
        )
    return detections


def _tuples_from_obb_result(
    obb: Any,
) -> list[tuple[float, float, float, float, float, float]]:
    """Build ``(cx, cy, w, h, theta_rad, confidence)`` tuples from an ``OBBResult``.

    Used by the active-learning ``detector_fn`` path. ``OBBResult`` stores area
    and aspect ratio rather than raw width/height, so major/minor are
    reconstructed exactly: ``major = sqrt(size * aspect)``,
    ``minor = sqrt(size / aspect)``. Feeding ``(cx, cy, major, minor, angle)``
    to ``geometry.obb_corners_from_dims`` reproduces ``obb.corners`` byte-for-byte
    (both build corners in the major-axis frame; see ``_corners_from_xywhr``).
    """
    out: list[tuple[float, float, float, float, float, float]] = []
    for i in range(obb.num_detections):
        cx = float(obb.centroids[i][0])
        cy = float(obb.centroids[i][1])
        theta = float(obb.angles[i])
        size = float(obb.sizes[i])
        aspect = float(obb.shapes[i][1])
        if aspect > 0:
            major = math.sqrt(max(size * aspect, 0.0))
            minor = math.sqrt(max(size / aspect, 0.0))
        else:
            major = minor = math.sqrt(max(size, 0.0))
        out.append((cx, cy, major, minor, theta, float(obb.confidences[i])))
    return out


def _detections_from_result(result: Any) -> list[dict[str, object]]:
    """Extract canvas-ready detection dicts from a raw executor result."""
    obb = extract_obb_result(result, frame_idx=0)
    return _dicts_from_obb_result(obb)


def _predict_direct(
    executor: Any,
    frame,
    *,
    confidence_threshold: float,
    iou: float = _PREVIEW_IOU,
) -> Any:
    """Run direct-mode OBB inference on a single BGR frame; return the ``OBBResult``."""
    raw_floor = max(1e-4, float(confidence_threshold))
    results = executor.predict(
        [frame],
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
    )
    if not results:
        return None
    return extract_obb_result(results[0], frame_idx=0)


def _sequential_obb_result(
    detect_executor: Any,
    obb_executor: Any,
    frame,
    *,
    conf: float,
    iou: float,
    crop_pad_ratio: float,
) -> Any:
    """Two-stage OBB inference on one frame -> merged ``OBBResult``.

    Mirrors ``model_test_dialog._run_sequential``: stage-1 detect on the full
    frame, ``build_crops`` off the boxes, stage-2 OBB on each crop, then
    ``merge_obb_results``. Preview has no explicit stage-2 image size, so crops
    are fed to the executor at their native size (no ``resize_crops_for_stage2``
    rescale, scale factor 1.0).
    """
    raw_floor = max(1e-4, float(conf))

    detect_results = detect_executor.predict(
        [frame],
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
    )
    if not detect_results:
        return None
    boxes = getattr(detect_results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return merge_obb_results(0, [])

    crop_spec = _SeqCropSpec(crop_pad_ratio=float(crop_pad_ratio))
    crops, offsets = build_crops(frame, boxes, crop_spec, None)
    if not crops:
        return merge_obb_results(0, [])

    orig_sizes = [(c.shape[1], c.shape[0]) for c in crops]
    # Preview keeps crops at native size (stage2 size unknown here). Kept
    # explicit so the parallel to model_test_dialog is obvious.
    crops_for_stage2 = resize_crops_for_stage2(crops, 0)

    obb_results = obb_executor.predict(
        crops_for_stage2,
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
    )

    sub = []
    for i, r in enumerate(obb_results):
        _ = orig_sizes[i]  # native-size crops => scale (1.0, 1.0)
        sub.append(extract_obb_result(r, 0, offset=offsets[i], scale=(1.0, 1.0)))
    return merge_obb_results(0, sub)


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

    compute_runtime = _resolve_compute_runtime(device_preference)
    executor = _get_torch_model(resolved_model_path, compute_runtime)
    obb = _predict_direct(executor, frame, confidence_threshold=confidence_threshold)
    if obb is None:
        return []
    return _dicts_from_obb_result(obb)


def predict_preview_detections_for_image(
    model,
    image_path: str,
    *,
    device: str,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    """Run inference using a pre-loaded executor on a single image. For batch reuse.

    ``model`` is an executor handle (from :func:`load_torch_model`); ``device``
    is retained for call-site compatibility but no longer used (the executor is
    already bound to its compute runtime).
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    obb = _predict_direct(model, frame, confidence_threshold=confidence_threshold)
    if obb is None:
        return []
    return _dicts_from_obb_result(obb)


def load_torch_model(model_path: str, device_preference: str = "auto"):
    """Load an OBB executor and return ``(executor, compute_runtime)``.

    The second element is the ``load_obb_executor`` compute-runtime string (not
    a torch device string). Downstream preview helpers accept it as their
    ``device`` argument but no longer use it -- the executor is already bound to
    its runtime.
    """
    resolved = str(Path(model_path).expanduser().resolve())
    compute_runtime = _resolve_compute_runtime(device_preference)
    return _get_torch_model(resolved, compute_runtime), compute_runtime


def predict_obb_for_frame(
    model,
    frame,
    *,
    device: str,
    conf: float,
    iou: float = _PREVIEW_IOU,
) -> list[tuple[float, float, float, float, float, float]]:
    """Run OBB inference on a single in-memory BGR frame and return
    (cx, cy, w, h, theta_rad, confidence) tuples. For AL detector_fn use.

    ``model`` is an executor handle; ``device`` is retained for call-site
    compatibility but unused.
    """
    obb = _predict_direct(model, frame, confidence_threshold=conf, iou=iou)
    if obb is None:
        return []
    return _tuples_from_obb_result(obb)


def predict_obb_for_frame_sequential(
    detect_model,
    obb_model,
    frame,
    *,
    detect_device: str,
    obb_device: str,
    conf: float,
    iou: float = _PREVIEW_IOU,
    crop_pad_ratio: float = 0.15,
) -> list[tuple[float, float, float, float, float, float]]:
    """Two-stage OBB prediction via the public inference helpers.

    Stage 1: axis-aligned detection on the full frame. Stage 2: oriented-bbox
    prediction on each ``build_crops`` crop, merged with ``merge_obb_results``.
    Returns (cx, cy, w, h, theta_rad, confidence) tuples in original-frame
    coordinates. ``detect_model``/``obb_model`` are executor handles;
    ``detect_device``/``obb_device`` are retained for call-site compatibility
    but unused.
    """
    if frame is None or frame.size == 0:
        return []
    merged = _sequential_obb_result(
        detect_model,
        obb_model,
        frame,
        conf=conf,
        iou=iou,
        crop_pad_ratio=crop_pad_ratio,
    )
    if merged is None:
        return []
    return _tuples_from_obb_result(merged)


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

    Returns canvas-ready detection dicts with `class_id`, `polygon_px`,
    `confidence`. The `class_id` now comes from the merged ``OBBResult`` (the
    stage-2 OBB head), not a hardcoded ``0``.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read preview image: {image_path}")

    detect_resolved = str(Path(detect_model_path).expanduser().resolve())
    obb_resolved = str(Path(obb_model_path).expanduser().resolve())
    compute_runtime = _resolve_compute_runtime(device_preference)

    detect_executor = _get_torch_model(detect_resolved, compute_runtime, "detect")
    obb_executor = _get_torch_model(obb_resolved, compute_runtime)

    merged = _sequential_obb_result(
        detect_executor,
        obb_executor,
        frame,
        conf=confidence_threshold,
        iou=_PREVIEW_IOU,
        crop_pad_ratio=crop_pad_ratio,
    )
    if merged is None:
        return []
    return _dicts_from_obb_result(merged)
