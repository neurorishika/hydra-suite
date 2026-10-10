from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import OBBConfig
from ..limits import MAX_DETECTIONS_PER_FRAME, require_target_count_within_limit
from ..result import OBBResult

# Size gates compare against ELLIPSE area, not the OBB rectangle area. The
# MIN/MAX_OBJECT_SIZE thresholds are derived from a circular body area
# (pi*(body/2)**2 in cli_config), and the legacy detector filters on the inscribed
# ellipse area (shapes[:,0] = pi/4 * w * h) — see core/detectors/_obb_geometry.py.
# OBBResult.sizes is the rectangle area (w*h = major*minor), which is ~27% larger,
# so comparing it directly would reject the largest detections the legacy pipeline
# keeps. Multiply by pi/4 to convert rectangle area -> ellipse area for parity.
_ELLIPSE_AREA_FRACTION = np.pi / 4.0


def _rank(confidences: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Order (into the given arrays) by confidence desc, then raw index asc.

    The single replay ranking. Detection caches are stored in exactly this
    order (``obb.rank_and_bound``), so a cache prefix IS a top-k under this
    ranking -- which makes the 2N window, NMS and the final cut agree for
    every N and keeps every replay set inside the per-animal superset.
    """
    return np.lexsort((np.asarray(positions), -np.asarray(confidences)))


def _final_cap(config: OBBConfig) -> int:
    """The replay-time final cap N. 0/unset means 'no N cut' (the limit)."""
    requested = int(getattr(config, "max_detections", 0) or 0)
    if requested <= 0:
        return MAX_DETECTIONS_PER_FRAME
    return require_target_count_within_limit(requested)


def _obb_nms(raw: OBBResult, indices: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over oriented bounding boxes.

    Mirrors legacy ``_obb_geometry._filter_overlapping_detections`` exactly: the
    IoU is computed on the OBB corner polygons via ``cv2.convexHull`` +
    ``cv2.intersectConvexConvex`` (NOT a reconstructed RotatedRect), so the
    suppression decisions near the IoU threshold match the legacy detector.
    """
    order = indices[_rank(raw.confidences[indices], indices)]
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
        polygons=(
            [raw.polygons[int(i)] for i in indices]
            if raw.polygons is not None
            else None
        ),
    )


def filter_with_indices(
    raw: OBBResult,
    config: OBBConfig,
    roi_mask: np.ndarray | None = None,
    *,
    apply_max_detections: bool = True,
) -> tuple[OBBResult, np.ndarray]:
    """Run the same gates as filter_detections and return (filtered, pre-filter indices).

    Returned indices index into `raw`. They are used as the primary key by downstream
    caches so that a threshold edit never invalidates HeadTail/CNN/Pose caches —
    only the OBB detection cache stores pre-filter results; downstream caches are
    keyed by these indices and re-aligned on load_frame.

    ``apply_max_detections=False`` is a diagnostic-only mode for measuring
    source candidates before the configured final tracking-target cap. The
    independent hard downstream crop ceiling is still retained.
    """
    n = raw.num_detections
    if n == 0:
        return raw, np.zeros(0, dtype=np.int32)

    final_cap = _final_cap(config) if apply_max_detections else MAX_DETECTIONS_PER_FRAME
    keep = raw.confidences >= config.confidence_threshold
    if apply_max_detections:
        # 2N replay window: the cache is confidence-ranked, so the first
        # min(2N, n) rows are exactly the legacy "raw cap" candidates.
        keep[min(n, 2 * final_cap) :] = False
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
    if len(indices) > final_cap:
        order = _rank(raw.confidences[indices], indices)[:final_cap]
        indices = indices[np.sort(order)]
        subset = _select(raw, indices)
    return subset, indices.astype(np.int32)


def filter_for_source(
    config: Any,
    raw: OBBResult,
    roi_mask: np.ndarray | None = None,
    *,
    apply_max_detections: bool = True,
) -> tuple[OBBResult, np.ndarray]:
    """Detection-source-aware dispatch in front of ``filter_with_indices``.

    OBB emits raw, un-gated detections, so the gates live here. bg-sub does not:
    ``BackgroundMeasurer.detect_objects`` applies the contour-area and size
    gates (the N-dependent MAX_TARGETS rules are applied here, at replay), and ``run_bgsub`` already intersects the ROI with the
    foreground mask — so by the time a bg-sub ``OBBResult`` reaches this layer
    there is nothing left to filter and the identity is correct. There is
    also no ``OBBConfig`` to gate with (``config.obb is None``), and bg-sub's
    confidences are NaN, so running the OBB gates would silently drop every
    detection on the confidence comparison.

    ``apply_max_detections=False`` is reserved for source-count diagnostics;
    normal inference and replay use the default final target cap.
    """
    if config.detection_source == "bgsub":
        bg = getattr(config, "bgsub", None)
        order = np.argsort(-np.asarray(raw.sizes, np.float64), kind="stable")
        cap = MAX_DETECTIONS_PER_FRAME
        if apply_max_detections and bg is not None:
            target = require_target_count_within_limit(max(1, int(bg.max_targets)))
            budget = target * int(bg.max_contour_multiplier)
            # Contour-budget noise guard. It now counts the STORED contours
            # (after the N-free area/size filters), not the raw findContours
            # count -- an accepted semantic change of the N-free cache.
            if raw.num_detections > budget:
                order = order[:0]
            cap = target
        # Top-`cap` by area; re-sorted ascending so raw-index keying stays
        # monotone for downstream caches.
        indices = np.ascontiguousarray(np.sort(order[:cap]), dtype=np.int32)
        if len(indices) == raw.num_detections:
            return raw, indices
        return _select(raw, indices), indices
    return filter_with_indices(
        raw,
        config.obb,
        roi_mask,
        apply_max_detections=apply_max_detections,
    )


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
