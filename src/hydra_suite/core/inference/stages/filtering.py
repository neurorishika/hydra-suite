from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ..config import OBBConfig
from ..result import OBBResult
from ..runtime import RuntimeContext
from .obb import _RawOBBTensors

# Size gates compare against ELLIPSE area, not the OBB rectangle area. The
# MIN/MAX_OBJECT_SIZE thresholds are derived from a circular body area
# (pi*(body/2)**2 in cli_config), and the legacy detector filters on the inscribed
# ellipse area (shapes[:,0] = pi/4 * w * h) — see core/detectors/_obb_geometry.py.
# OBBResult.sizes is the rectangle area (w*h = major*minor), which is ~27% larger,
# so comparing it directly would reject the largest detections the legacy pipeline
# keeps. Multiply by pi/4 to convert rectangle area -> ellipse area for parity.
_ELLIPSE_AREA_FRACTION = np.pi / 4.0


def filter_raw(
    raw: OBBResult | _RawOBBTensors,
    config: OBBConfig,
    roi_mask: np.ndarray | None,
    roi_mask_cuda: torch.Tensor | None,
    runtime: RuntimeContext,
) -> OBBResult:
    """Dispatcher: CUDA path uses filter_from_tensors; CPU/MPS uses filter_detections."""
    if isinstance(raw, _RawOBBTensors):
        return filter_from_tensors(raw, config, roi_mask_cuda, runtime)
    return filter_detections(raw, config, roi_mask)


def filter_detections(
    raw: OBBResult,
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
) -> OBBResult:
    """CPU/MPS path: apply confidence, size, ROI, NMS, and max-count gates in NumPy.

    Per Correction 14: detection_ids of survivors are SUBSETS of raw.detection_ids;
    they are never regenerated — this preserves cache stability across threshold edits.
    """
    n = raw.num_detections
    if n == 0:
        return raw

    keep = np.ones(n, dtype=bool)
    keep &= raw.confidences >= config.confidence_threshold

    ellipse_area = raw.sizes * _ELLIPSE_AREA_FRACTION
    if config.min_object_size > 0:
        keep &= ellipse_area >= config.min_object_size
    if config.max_object_size < float("inf"):
        keep &= ellipse_area <= config.max_object_size

    if config.min_aspect_ratio > 0 or config.max_aspect_ratio < float("inf"):
        aspect = raw.shapes[:, 1]
        keep &= (aspect >= config.min_aspect_ratio) & (
            aspect <= config.max_aspect_ratio
        )

    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        for i in range(n):
            if not keep[i]:
                continue
            cx, cy = int(raw.centroids[i, 0]), int(raw.centroids[i, 1])
            if 0 <= cy < h and 0 <= cx < w:
                keep[i] = bool(roi_mask[cy, cx])
            else:
                keep[i] = False

    indices = np.where(keep)[0]
    if len(indices) == 0:
        return _select(raw, indices)

    if config.iou_threshold < 1.0 and len(indices) > 1:
        indices = _obb_nms(raw, indices, config.iou_threshold)

    if config.max_detections > 0 and len(indices) > config.max_detections:
        # H5 parity: legacy keeps the LARGEST detections (sort by size), not the
        # most confident — _obb_geometry:587-588.
        order = np.argsort(raw.sizes[indices])[::-1][: config.max_detections]
        indices = indices[order]

    return _select(raw, indices)


def filter_from_tensors(
    raw: _RawOBBTensors,
    config: OBBConfig,
    roi_mask_cuda: torch.Tensor | None,
    runtime: RuntimeContext,
) -> OBBResult:
    """CUDA path: gates run as tensor ops on device; only survivors are pulled to CPU for NMS.

    Per Correction 14: this path generates detection_ids fresh because _RawOBBTensors
    does not carry them — IDs follow `frame_idx * STRIDE + slot` over the post-filter
    survivors, which becomes the canonical primary key for downstream caches.
    """
    n = raw.xywhr.shape[0]
    if n == 0:
        return _empty_obb_result(raw.frame_idx)

    keep = raw.conf >= config.confidence_threshold

    w_t = raw.xywhr[:, 2]
    h_t = raw.xywhr[:, 3]
    # H6 parity: drop non-finite / non-positive-geometry detections (legacy
    # _obb_geometry:303-312) so NaN/Inf can't reach assignment/Kalman.
    keep = keep & torch.isfinite(raw.xywhr).all(dim=1) & torch.isfinite(raw.conf)
    keep = keep & (w_t > 0) & (h_t > 0)
    sizes_t = w_t * h_t
    ellipse_area_t = sizes_t * _ELLIPSE_AREA_FRACTION
    if config.min_object_size > 0:
        keep = keep & (ellipse_area_t >= config.min_object_size)
    if config.max_object_size < float("inf"):
        keep = keep & (ellipse_area_t <= config.max_object_size)

    if config.min_aspect_ratio > 0 or config.max_aspect_ratio < float("inf"):
        major_t = torch.maximum(w_t, h_t)
        minor_t = torch.minimum(w_t, h_t).clamp_min(1e-6)
        aspect_t = major_t / minor_t
        keep = (
            keep
            & (aspect_t >= config.min_aspect_ratio)
            & (aspect_t <= config.max_aspect_ratio)
        )

    if roi_mask_cuda is not None:
        mask_h, mask_w = roi_mask_cuda.shape[:2]
        cx = raw.xywhr[:, 0].long().clamp(0, mask_w - 1)
        cy = raw.xywhr[:, 1].long().clamp(0, mask_h - 1)
        keep = keep & roi_mask_cuda[cy, cx].bool()

    indices_t = keep.nonzero(as_tuple=True)[0]
    if indices_t.numel() == 0:
        return _empty_obb_result(raw.frame_idx)

    xywhr_np = raw.xywhr[indices_t].cpu().numpy()
    corners_np = raw.corners[indices_t].cpu().numpy()
    conf_np = raw.conf[indices_t].cpu().numpy()
    cls_np = (
        raw.cls[indices_t].cpu().numpy().astype(np.int64)
        if raw.cls is not None
        else np.zeros(int(indices_t.numel()), dtype=np.int64)
    )
    sizes_np = (xywhr_np[:, 2] * xywhr_np[:, 3]).astype(np.float32)
    safe_h = np.where(xywhr_np[:, 3] > 0, xywhr_np[:, 3], 1.0)
    aspect_np = np.where(xywhr_np[:, 3] > 0, xywhr_np[:, 2] / safe_h, 1.0).astype(
        np.float32
    )

    m = int(len(conf_np))
    subset = OBBResult(
        frame_idx=raw.frame_idx,
        centroids=xywhr_np[:, :2].astype(np.float32),
        angles=xywhr_np[:, 4].astype(np.float32),
        sizes=sizes_np,
        shapes=np.stack([sizes_np, aspect_np], axis=1),
        confidences=conf_np.astype(np.float32),
        corners=corners_np.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(raw.frame_idx, m),
        class_ids=cls_np,
    )

    local_idx = np.arange(m)

    if config.iou_threshold < 1.0 and m > 1:
        local_idx = _obb_nms(subset, local_idx, config.iou_threshold)

    if config.max_detections > 0 and len(local_idx) > config.max_detections:
        # H5 parity: keep the LARGEST detections (sort by size) — _obb_geometry:587-588.
        order = np.argsort(subset.sizes[local_idx])[::-1][: config.max_detections]
        local_idx = local_idx[order]

    return _select(subset, local_idx)


def _obb_nms(raw: OBBResult, indices: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over oriented bounding boxes.

    Mirrors legacy ``_obb_geometry._filter_overlapping_detections`` exactly: the
    IoU is computed on the OBB corner polygons via ``cv2.convexHull`` +
    ``cv2.intersectConvexConvex`` (NOT a reconstructed RotatedRect), so the
    suppression decisions near the IoU threshold match the legacy detector.
    """
    order = indices[np.argsort(raw.confidences[indices])[::-1]]
    hulls: dict[int, tuple[np.ndarray, float]] = {}
    # Axis-aligned bbox per detection for the cheap overlap pre-check (matches
    # legacy: boxes whose AABBs don't overlap have zero polygon IoU, so the
    # expensive convex-hull intersection is skipped).
    bbox_min = raw.corners.min(axis=1)
    bbox_max = raw.corners.max(axis=1)

    def hull(idx: int) -> tuple[np.ndarray, float]:
        cached = hulls.get(idx)
        if cached is None:
            p = cv2.convexHull(np.asarray(raw.corners[idx], dtype=np.float32)).reshape(
                -1, 2
            )
            cached = (p, float(abs(cv2.contourArea(p))))
            hulls[idx] = cached
        return cached

    keep: list[int] = []
    suppressed = np.zeros(len(raw.confidences), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        p1, area1 = hull(idx)
        if area1 <= 1e-9:
            continue
        cmin, cmax = bbox_min[idx], bbox_max[idx]
        for other in order:
            if suppressed[other] or other == idx:
                continue
            # AABB overlap pre-check (both width and height must overlap).
            if (
                cmin[0] >= bbox_max[other, 0]
                or cmax[0] <= bbox_min[other, 0]
                or cmin[1] >= bbox_max[other, 1]
                or cmax[1] <= bbox_min[other, 1]
            ):
                continue
            if _obb_iou_corners(p1, area1, *hull(other)) >= iou_threshold:
                suppressed[other] = True

    return np.array(keep, dtype=int)


def _obb_iou_corners(
    p1: np.ndarray, area1: float, p2: np.ndarray, area2: float
) -> float:
    """IoU of two convex corner polygons (matches legacy _compute_obb_iou_batch)."""
    if area1 <= 1e-9 or area2 <= 1e-9:
        return 0.0
    try:
        inter_area, _ = cv2.intersectConvexConvex(p1, p2)
        inter_area = float(max(0.0, inter_area))
    except Exception:
        inter_area = 0.0
    union = area1 + area2 - inter_area
    return float(inter_area / union) if union > 1e-9 else 0.0


def _select(raw: OBBResult, indices: np.ndarray) -> OBBResult:
    """Subset all OBBResult arrays by `indices` — preserves detection_ids."""
    if len(indices) == 0:
        return _empty_obb_result(raw.frame_idx)
    return OBBResult(
        frame_idx=raw.frame_idx,
        centroids=raw.centroids[indices],
        angles=raw.angles[indices],
        sizes=raw.sizes[indices],
        shapes=raw.shapes[indices],
        confidences=raw.confidences[indices],
        corners=raw.corners[indices],
        detection_ids=raw.detection_ids[indices],
        class_ids=raw.class_ids_or_zeros[indices],
    )


def filter_with_indices(
    raw: OBBResult,
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
) -> tuple[OBBResult, np.ndarray]:
    """Run the same gates as filter_detections and return (filtered, pre-filter indices).

    Returned indices index into `raw`. They are used as the primary key by downstream
    caches so that a threshold edit never invalidates HeadTail/CNN/Pose caches —
    only the OBB detection cache stores pre-filter results; downstream caches are
    keyed by these indices and re-aligned on load_frame.
    """
    n = raw.num_detections
    if n == 0:
        return raw, np.zeros(0, dtype=np.int32)

    keep = raw.confidences >= config.confidence_threshold
    ellipse_area = raw.sizes * _ELLIPSE_AREA_FRACTION
    if config.min_object_size > 0:
        keep = keep & (ellipse_area >= config.min_object_size)
    if config.max_object_size < float("inf"):
        keep = keep & (ellipse_area <= config.max_object_size)
    if config.min_aspect_ratio > 0 or config.max_aspect_ratio < float("inf"):
        aspect = raw.shapes[:, 1]
        keep = (
            keep
            & (aspect >= config.min_aspect_ratio)
            & (aspect <= config.max_aspect_ratio)
        )
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        cx = np.clip(raw.centroids[:, 0].astype(np.int32), 0, w - 1)
        cy = np.clip(raw.centroids[:, 1].astype(np.int32), 0, h - 1)
        keep = keep & roi_mask[cy, cx].astype(bool)

    indices = np.where(keep)[0]
    subset = _select(raw, indices)
    if config.iou_threshold < 1.0 and len(indices) > 1:
        keep_nms = _obb_nms(subset, np.arange(len(indices)), config.iou_threshold)
        indices = indices[keep_nms]
        subset = _select(raw, indices)
    if config.max_detections > 0 and len(indices) > config.max_detections:
        # H5 parity: keep the LARGEST detections (sort by size) — _obb_geometry:587-588.
        order = np.argsort(raw.sizes[indices])[::-1][: config.max_detections]
        indices = indices[order]
        subset = _select(raw, indices)
    return subset, indices.astype(np.int32)


def filter_for_source(
    config: Any,
    raw: OBBResult,
    roi_mask: np.ndarray | None = None,
) -> tuple[OBBResult, np.ndarray]:
    """Detection-source-aware dispatch in front of ``filter_with_indices``.

    OBB emits raw, un-gated detections, so the gates live here. bg-sub does not:
    ``BackgroundMeasurer.detect_objects`` already applies the contour-area, size,
    and MAX_TARGETS gates, and ``run_bgsub`` already intersects the ROI with the
    foreground mask — so by the time a bg-sub ``OBBResult`` reaches this layer
    there is nothing left to filter and the identity is correct. There is
    also no ``OBBConfig`` to gate with (``config.obb is None``), and bg-sub's
    confidences are NaN, so running the OBB gates would silently drop every
    detection on the confidence comparison.
    """
    if config.detection_source == "bgsub":
        return raw, np.arange(raw.num_detections, dtype=np.int32)
    return filter_with_indices(raw, config.obb, roi_mask)


def _empty_obb_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), dtype=np.float32),
        angles=np.zeros(0, dtype=np.float32),
        sizes=np.zeros(0, dtype=np.float32),
        shapes=np.zeros((0, 2), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, 0),
        class_ids=np.zeros(0, dtype=np.int64),
    )
