from __future__ import annotations

import cv2
import numpy as np
import torch

from ..config import OBBConfig
from ..result import OBBResult
from ..runtime import RuntimeContext
from .obb import _RawOBBTensors


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

    if config.min_object_size > 0:
        keep &= raw.sizes >= config.min_object_size
    if config.max_object_size < float("inf"):
        keep &= raw.sizes <= config.max_object_size

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
        order = np.argsort(raw.confidences[indices])[::-1][: config.max_detections]
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
    sizes_t = w_t * h_t
    if config.min_object_size > 0:
        keep = keep & (sizes_t >= config.min_object_size)
    if config.max_object_size < float("inf"):
        keep = keep & (sizes_t <= config.max_object_size)

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
    )

    local_idx = np.arange(m)

    if config.iou_threshold < 1.0 and m > 1:
        local_idx = _obb_nms(subset, local_idx, config.iou_threshold)

    if config.max_detections > 0 and len(local_idx) > config.max_detections:
        order = np.argsort(subset.confidences[local_idx])[::-1][: config.max_detections]
        local_idx = local_idx[order]

    return _select(subset, local_idx)


def _obb_nms(raw: OBBResult, indices: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over oriented bounding boxes via cv2.rotatedRectangleIntersection."""
    order = indices[np.argsort(raw.confidences[indices])[::-1]]
    keep: list[int] = []
    suppressed = np.zeros(len(raw.confidences), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        rect_a = _obb_to_cv2_rect(raw, idx)
        for other in order:
            if suppressed[other] or other == idx:
                continue
            if _rotated_iou(rect_a, _obb_to_cv2_rect(raw, other)) > iou_threshold:
                suppressed[other] = True

    return np.array(keep, dtype=int)


def _obb_to_cv2_rect(raw: OBBResult, idx: int) -> tuple:
    """Convert OBBResult entry to cv2 RotatedRect tuple: ((cx, cy), (w, h), angle_deg)."""
    cx, cy = float(raw.centroids[idx, 0]), float(raw.centroids[idx, 1])
    corners = raw.corners[idx]
    w = float(np.linalg.norm(corners[1] - corners[0]))
    h = float(np.linalg.norm(corners[3] - corners[0]))
    angle = float(np.degrees(raw.angles[idx]))
    return (cx, cy), (w, h), angle


def _rotated_iou(rect_a: tuple, rect_b: tuple) -> float:
    """IOU between two cv2 RotatedRect tuples."""
    try:
        ret, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if ret == cv2.INTERSECT_NONE or intersection is None:
            return 0.0
        inter_area = cv2.contourArea(intersection)
        _, (wa, ha), _ = rect_a
        _, (wb, hb), _ = rect_b
        union = wa * ha + wb * hb - inter_area
        return float(inter_area / union) if union > 0 else 0.0
    except Exception:
        return 0.0


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
    )
