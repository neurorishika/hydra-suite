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
from typing import Any, Callable

import cv2
import numpy as np

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.runtime_artifacts import load_obb_executor
from hydra_suite.core.inference.stages.obb import (
    _bound_compact_parts,
    _extract_obb_from_boxes,
    _extract_obb_from_masks,
    extract_obb_result,
    iter_crops,
    merge_obb_results,
    merge_per_frame,
)
from hydra_suite.core.inference.stages.slicing import (
    admitted_tile_chunk_size,
    iter_tile_job_chunks,
)
from hydra_suite.utils.slice_geometry import plan_tiles, tile_size_for_mode

logger = logging.getLogger(__name__)

#: IOU threshold for the preview's NMS pass. Matches the legacy ultralytics
#: default that ``predict_obb_for_frame`` previously relied on.
_PREVIEW_IOU = 0.7

#: Max detections fed to the executor's NMS. Matches the legacy YOLO OBB path's
#: ``YOLO_MAX_TARGETS`` (100) -- the executor's own default (20) is too low for
#: busy multi-animal scenes and would silently hide real detections.
_PREVIEW_MAX_DET = 100

# MPSGraph rejects tensors with more than INT_MAX elements.  A large image can
# yield hundreds of small SAHI tiles; passing every tile to a large YOLO model
# in one ``predict`` call creates enormous intermediate feature maps even
# though every individual tile is safe.  Four 640px tiles leave ample headroom
# for a YOLO-x segmentation backbone.  Other runtimes retain a modestly larger
# batch for throughput without ever constructing an unbounded tensor.
_MPS_SLICE_BATCH_SIZE = 4
_DEFAULT_SLICE_BATCH_SIZE = 16

#: A CPU, numpy-universe ``RuntimeContext`` for the preview's shared-seam merge.
#: The preview executor always returns CPU ultralytics ``Results`` (never
#: on-device CUDA tensors), so ``tensor_on_cuda`` is False and ``merge_per_frame``
#: runs the numpy universe; the forced cv2 merge backend never reads ``device``.
_PREVIEW_RUNTIME = RuntimeContext(
    cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False
)


def _preview_slice_merge_config(merge_threshold: float) -> OBBConfig:
    """Minimal ``OBBConfig`` carrying only what ``merge_per_frame`` reads for the
    preview's cross-tile OBB merge: the slice merge policy/metric/threshold/
    backend and ``raw_detection_cap``.

    Matches the preview's prior hand-rolled merge byte-for-byte -- ``greedy_nmm``
    / ``ios`` / cv2, with the same finite preview candidate ceiling used by
    the executor.
    """
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="",
            slice=SliceConfig(
                enabled=True,
                merge_policy="greedy_nmm",
                merge_metric="ios",
                merge_threshold=float(merge_threshold),
                merge_backend="cv2",
            ),
        ),
        max_detections=_PREVIEW_MAX_DET,
        raw_detection_cap=_PREVIEW_MAX_DET,
    )


def _slice_prediction_batch_size(executor: Any) -> int:
    """Return a bounded tile batch size for a preview executor.

    Ultralytics exposes the bound torch module as ``executor.model``.  Direct
    TensorRT/CoreML adapters do not, and safely use the general bound.  Do not
    derive a batch size from the full tile list: that was the source of the
    MPSGraph INT_MAX failure on high-resolution images.
    """
    try:
        model = getattr(executor, "model", None)
        parameter = next(model.parameters()) if model is not None else None
        if parameter is not None and str(parameter.device).startswith("mps"):
            return _MPS_SLICE_BATCH_SIZE
    except (AttributeError, StopIteration, TypeError):
        pass
    return _DEFAULT_SLICE_BATCH_SIZE


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


def preview_object_tile_fraction(target_sizes, object_tile_fraction, imgsz) -> float:
    """object_tile_fraction for auto_object preview: median(target_sizes)/imgsz.

    Uses the trained apparent scale (median of the training target sizes) so the
    preview tiles at a scale the model was actually trained at. Falls back to the
    stored object_tile_fraction when target_sizes is empty or imgsz <= 0. Clamped
    to the same [0.01, 0.9] range tile_size_for_mode uses.
    """
    sizes = [float(t) for t in (target_sizes or [])]
    if not sizes or int(imgsz) <= 0:
        return float(object_tile_fraction)

    frac = float(np.median(np.asarray(sizes, dtype=np.float64))) / float(imgsz)
    return max(0.01, min(0.9, frac))


def _resolve_torch_device(device_preference: str) -> str:
    """Map a high-level device preference to a torch-style device string."""
    from hydra_suite.core.inference.torch_device import resolve_torch_device

    return resolve_torch_device(device_preference)


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
        native_polygon = (
            obb.polygons[i]
            if obb.polygons is not None and i < len(obb.polygons)
            else obb.corners[i]
        )
        detections.append(
            {
                "class_id": int(class_ids[i]),
                "polygon_px": [
                    (float(x), float(y)) for (x, y) in np.asarray(native_polygon)
                ],
                "confidence": float(obb.confidences[i]),
            }
        )
    return detections


def dicts_from_obb_result(obb: Any) -> list[dict[str, object]]:
    """Public wrapper around :func:`_dicts_from_obb_result`.

    Lets callers outside this module (e.g. ``main_window``) convert an
    ``OBBResult`` -- such as the one returned by :func:`predict_sliced_obb_result`
    -- into canvas-ready detection dicts without reaching into a private name.
    """
    return _dicts_from_obb_result(obb)


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
    emit_native_geometry: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> Any:
    """Run direct-mode OBB inference on a single BGR frame; return the ``OBBResult``."""
    if should_stop is not None and should_stop():
        return None
    raw_floor = max(1e-4, float(confidence_threshold))
    results = executor.predict(
        [frame],
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
        max_det=_PREVIEW_MAX_DET,
    )
    if should_stop is not None and should_stop():
        return None
    if not results:
        return None
    return extract_obb_result(
        results[0], frame_idx=0, emit_native_geometry=emit_native_geometry
    )


def predict_sliced_obb_result(
    executor,
    frame,
    *,
    geometry_mode: str,
    imgsz: int,
    reference_body_px: float,
    object_tile_fraction: float,
    slice_width: int,
    slice_height: int,
    overlap: float,
    merge_threshold: float,
    confidence_threshold: float,
    iou: float = _PREVIEW_IOU,
    task: str = "obb",
    should_stop: Callable[[], bool] | None = None,
):
    """Executor-level sliced OBB inference on one BGR frame (preview/AL).

    Tiles via ``utils.slice_geometry`` (same grid inference uses), predicts each
    tile, offsets detections into frame space, and merges cross-tile duplicates
    with the shipped cv2 oracle. Returns a frame-space ``OBBResult`` or None.
    """
    if should_stop is not None and should_stop():
        return None

    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    tw, th = tile_size_for_mode(
        geometry_mode=geometry_mode,
        imgsz=imgsz,
        reference_body_px=reference_body_px,
        object_tile_fraction=object_tile_fraction,
        slice_width=slice_width,
        slice_height=slice_height,
    )
    plan = plan_tiles((fh, fw), tw, th, overlap, overlap)
    raw_floor = max(1e-4, float(confidence_threshold))
    if not plan.tiles:
        return _predict_direct(
            executor,
            frame,
            confidence_threshold=confidence_threshold,
            iou=iou,
            emit_native_geometry=task == "segment",
            should_stop=should_stop,
        )

    # Route extraction + cross-tile merge through the SAME production seam the
    # ``Grid`` region source uses.
    def _extract_tile(res, x0: int, y0: int):
        offset = (float(max(0, x0)), float(max(0, y0)))
        if task == "detect":
            return _extract_obb_from_boxes(
                res, frame_idx=0, fixed_angle_rad=0.0, offset=offset
            )
        if task == "segment":
            return _extract_obb_from_masks(
                res,
                frame_idx=0,
                raw_detection_cap=_PREVIEW_MAX_DET,
                offset=offset,
                emit_native_geometry=True,
            )
        return extract_obb_result(res, frame_idx=0, offset=offset)

    batch_size = admitted_tile_chunk_size(
        plan,
        imgsz=imgsz,
        device_tiles=False,
        requested=_slice_prediction_batch_size(executor),
        task=task,
        max_detections=_PREVIEW_MAX_DET,
    )
    parts = []
    chunk_iter = iter(
        iter_tile_job_chunks([frame], plan, device_tiles=False, chunk_size=batch_size)
    )
    while True:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break
        if should_stop is not None and should_stop():
            return None
        tiles_img = [image for _job, image in chunk]
        results = executor.predict(
            tiles_img,
            conf=raw_floor,
            iou=float(iou),
            verbose=False,
            max_det=_PREVIEW_MAX_DET,
        )
        for (job, _image), result in zip(chunk, results):
            x0, y0 = job.offset
            parts.append(_extract_tile(result, int(x0), int(y0)))
        parts = _bound_compact_parts(parts, 0, _PREVIEW_MAX_DET)
        if should_stop is not None and should_stop():
            return None
        # A ``for`` loop requests the next generator item before rebinding its
        # target, which would overlap the prior tile/result batch with the
        # next materialized chunk. Release all batch-local payloads first.
        chunk.clear()
        del job, _image, result, results, tiles_img, chunk

    # Route extraction + cross-tile merge through the SAME production seam the
    # ``Grid`` region source uses: ``extract_obb_result``'s native ``offset=``
    # (retiring the preview's own ``_offset_result``) and the shared
    # ``merge_per_frame`` overlap-band merge (retiring the preview's hand-rolled
    # ``band_membership`` + ``merge_obb_detections`` block). The preview keeps
    # its OWN ``executor.predict`` above -- it runs the executor's within-tile
    # NMS (``iou=_PREVIEW_IOU``) because, unlike the tracking pipeline, there is
    # no downstream filtering stage here to dedup within-tile duplicates.
    return merge_per_frame(
        parts,
        "overlap_band_nms",
        plan,
        _preview_slice_merge_config(merge_threshold),
        _PREVIEW_RUNTIME,
    )


def _sequential_obb_result(
    detect_executor: Any,
    obb_executor: Any,
    frame,
    *,
    conf: float,
    iou: float,
    crop_pad_ratio: float,
    should_stop: Callable[[], bool] | None = None,
) -> Any:
    """Two-stage OBB inference on one frame -> merged ``OBBResult``.

    Mirrors ``model_test_dialog._run_sequential``: stage-1 detect on the full
    frame, lazy crops off the boxes, bounded stage-2 OBB chunks, then
    ``merge_obb_results``. Preview has no explicit stage-2 image size, so crops
    are fed to the executor at their native size (no resize; scale factor 1.0).
    """
    if should_stop is not None and should_stop():
        return None

    raw_floor = max(1e-4, float(conf))

    detect_results = detect_executor.predict(
        [frame],
        conf=raw_floor,
        iou=float(iou),
        verbose=False,
        max_det=_PREVIEW_MAX_DET,
    )
    if should_stop is not None and should_stop():
        return None
    if not detect_results:
        return None
    boxes = getattr(detect_results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return merge_obb_results(0, [])

    crop_spec = _SeqCropSpec(crop_pad_ratio=float(crop_pad_ratio))
    sub = []
    batch_size = _slice_prediction_batch_size(obb_executor)
    crop_iter = iter_crops(
        frame,
        boxes,
        crop_spec,
        None,
        max_candidates=_PREVIEW_MAX_DET,
    )
    pending_crops = []
    pending_offsets = []
    for crop, offset in crop_iter:
        pending_crops.append(crop)
        pending_offsets.append(offset)
        if len(pending_crops) < batch_size:
            continue
        if should_stop is not None and should_stop():
            return None
        results = obb_executor.predict(
            pending_crops,
            conf=raw_floor,
            iou=float(iou),
            verbose=False,
            max_det=_PREVIEW_MAX_DET,
        )
        sub.extend(
            extract_obb_result(result, 0, offset=offset, scale=(1.0, 1.0))
            for result, offset in zip(results, pending_offsets)
        )
        pending_crops.clear()
        pending_offsets.clear()
        del results, crop, offset
    if pending_crops:
        if should_stop is not None and should_stop():
            return None
        results = obb_executor.predict(
            pending_crops,
            conf=raw_floor,
            iou=float(iou),
            verbose=False,
            max_det=_PREVIEW_MAX_DET,
        )
        sub.extend(
            extract_obb_result(result, 0, offset=offset, scale=(1.0, 1.0))
            for result, offset in zip(results, pending_offsets)
        )
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
    task: str = "obb",
    should_stop: Callable[[], bool] | None = None,
) -> list[dict[str, object]]:
    """Run inference using a pre-loaded executor on a single image. For batch reuse.

    ``model`` is an executor handle (from :func:`load_torch_model`); ``device``
    is retained for call-site compatibility but no longer used (the executor is
    already bound to its compute runtime).
    """
    if should_stop is not None and should_stop():
        return []
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    obb = _predict_direct(
        model,
        frame,
        confidence_threshold=confidence_threshold,
        emit_native_geometry=task == "segment",
        should_stop=should_stop,
    )
    if obb is None:
        return []
    return _dicts_from_obb_result(obb)


def load_torch_model(
    model_path: str,
    device_preference: str = "auto",
    task: str = "obb",
):
    """Load an OBB executor and return ``(executor, compute_runtime)``.

    The second element is the ``load_obb_executor`` compute-runtime string (not
    a torch device string). Downstream preview helpers accept it as their
    ``device`` argument but no longer use it -- the executor is already bound to
    its runtime.
    """
    resolved = str(Path(model_path).expanduser().resolve())
    compute_runtime = _resolve_compute_runtime(device_preference)
    return _get_torch_model(resolved, compute_runtime, task), compute_runtime


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


# `predict_obb_for_frame_export` and its exclusive helper
# `_tuples_with_polygons_from_obb_result` lived here. Their only caller was
# DetectKit's AL detector closure (`main_window._load_active_detector_fn`),
# retired when AL scoring moved onto `InferenceRunner`'s batched, cached
# detection pass; nothing else in the repo referenced either. Deleted rather
# than left as a never-exercised path. The native-geometry capability itself
# is unaffected -- it is a defaulted `_predict_direct(..., emit_native_geometry=)`
# keyword, and `OBBConfig.emit_native_geometry` remains the supported way to
# ask a real inference config for polygons.


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
    should_stop: Callable[[], bool] | None = None,
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
        should_stop=should_stop,
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
