"""OBB geometry, IOU computation, NMS filtering, and raw detection extraction.

Provided as a mixin class so that both ``YOLOOBBDetector`` and the lightweight
``DetectionFilter`` can share the same logic without method-borrowing tricks.
"""

import logging

import cv2
import numpy as np

from ._utils import _advanced_config_value, _normalize_detection_ids

logger = logging.getLogger(__name__)


class OBBGeometryMixin:
    """Mixin supplying OBB geometry helpers that only depend on ``self.params``."""

    # ------------------------------------------------------------------
    # Advanced config helper (thin wrapper around the module-level fn)
    # ------------------------------------------------------------------

    def _advanced_config_value(self, key: str, default=None):
        """Read a power-user override from ADVANCED_CONFIG when present."""
        return _advanced_config_value(self.params, key, default)

    # ------------------------------------------------------------------
    # OBB IOU
    # ------------------------------------------------------------------

    def _compute_obb_iou_batch(self, corners1, corners_list, indices):
        """
        Batch compute IOU between one OBB and multiple OBBs efficiently.

        Args:
            corners1: (4, 2) array for reference OBB
            corners_list: List of all corner arrays
            indices: List of indices to compute IOU for

        Returns:
            Array of IOU values
        """
        if len(indices) == 0:
            return np.array([])

        p1 = cv2.convexHull(np.asarray(corners1, dtype=np.float32)).reshape(-1, 2)
        area1 = float(abs(cv2.contourArea(p1)))
        if area1 <= 1e-9:
            return np.zeros(len(indices), dtype=np.float32)

        ious = np.zeros(len(indices), dtype=np.float32)
        for i, idx in enumerate(indices):
            p2 = cv2.convexHull(
                np.asarray(corners_list[idx], dtype=np.float32)
            ).reshape(-1, 2)
            area2 = float(abs(cv2.contourArea(p2)))
            if area2 <= 1e-9:
                continue
            try:
                inter_area, _ = cv2.intersectConvexConvex(p1, p2)
                inter_area = float(max(0.0, inter_area))
            except Exception:
                inter_area = 0.0
            union = area1 + area2 - inter_area
            if union > 1e-9:
                ious[i] = inter_area / union

        return ious

    # ------------------------------------------------------------------
    # NMS / overlap filtering
    # ------------------------------------------------------------------

    def _filter_overlapping_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb_corners_list,
        iou_threshold,
        detection_ids=None,
        heading_hints=None,
        heading_confidences=None,
        directed_mask=None,
    ):
        """
        Filter overlapping detections using spatially-optimized IOU-based NMS for OBB.
        Keeps highest confidence detections and removes overlapping ones.
        Optimized for high detection counts (25-200 animals).

        Args:
            meas: List of measurements [cx, cy, angle]
            sizes: List of detection areas
            shapes: List of (area, aspect_ratio) tuples
            confidences: List of confidence scores (0-1)
            obb_corners_list: List of corner arrays (4, 2) for each detection
            iou_threshold: IOU threshold for considering detections as overlapping

        Returns:
            Filtered versions of all input lists. If detection_ids is provided,
            it is filtered identically and returned as the final element.
        """
        if len(meas) <= 1:
            if detection_ids is None and heading_hints is None:
                return meas, sizes, shapes, confidences, obb_corners_list
            if detection_ids is not None and heading_hints is None:
                return meas, sizes, shapes, confidences, obb_corners_list, detection_ids
            if detection_ids is None:
                return (
                    meas,
                    sizes,
                    shapes,
                    confidences,
                    obb_corners_list,
                    heading_hints,
                    heading_confidences,
                    directed_mask,
                )
            return (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners_list,
                detection_ids,
                heading_hints,
                heading_confidences,
                directed_mask,
            )

        n_detections = len(meas)

        # Convert inputs to numpy arrays for vectorized operations
        confidences_arr = np.array(confidences)

        # Pre-compute axis-aligned bounding boxes (fully vectorized)
        corners_array = np.array(obb_corners_list)  # (n, 4, 2)
        bbox_mins = corners_array.min(axis=1)  # (n, 2)
        bbox_maxs = corners_array.max(axis=1)  # (n, 2)

        # Sort indices by confidence (highest first)
        sorted_indices = np.argsort(confidences_arr)[::-1]

        keep_mask = np.zeros(n_detections, dtype=bool)

        idx = 0
        while idx < len(sorted_indices):
            # Keep the detection with highest remaining confidence
            current_idx = sorted_indices[idx]
            keep_mask[current_idx] = True

            if idx == len(sorted_indices) - 1:
                break

            # Get current box bounding box
            curr_min = bbox_mins[current_idx]
            curr_max = bbox_maxs[current_idx]

            # Get remaining candidates
            remaining_indices = sorted_indices[idx + 1 :]
            rem_mins = bbox_mins[remaining_indices]
            rem_maxs = bbox_maxs[remaining_indices]

            # Vectorized axis-aligned bbox overlap check
            inter_mins = np.maximum(curr_min, rem_mins)
            inter_maxs = np.minimum(curr_max, rem_maxs)

            # Check if boxes overlap (width and height both positive)
            inter_wh = inter_maxs - inter_mins
            overlaps = (inter_wh[:, 0] > 0) & (inter_wh[:, 1] > 0)

            # For non-overlapping boxes, skip IOU calculation
            keep_remaining = ~overlaps

            # For overlapping boxes, compute IOU
            if overlaps.any():
                # Initial state: keep overlapping candidates unless precise OBB IOU says suppress.
                overlapping_local = np.where(overlaps)[0]
                keep_remaining[overlapping_local] = True

                # Run precise polygon IOU for all AABB-overlapping candidates.
                # This matches direct-mode manual OBB suppression behavior.
                precise_check_global = remaining_indices[overlapping_local]
                precise_ious = self._compute_obb_iou_batch(
                    obb_corners_list[current_idx],
                    obb_corners_list,
                    precise_check_global,
                )

                suppress = precise_ious >= iou_threshold
                keep_remaining[overlapping_local] = ~suppress

            # Update sorted indices to keep only non-suppressed detections
            sorted_indices = np.concatenate(
                [sorted_indices[: idx + 1], remaining_indices[keep_remaining]]
            )

            idx += 1

        # Use numpy indexing for final filtering (much faster than list comprehension)
        keep_indices = np.where(keep_mask)[0]

        # Convert back to lists with proper indexing
        meas = [meas[i] for i in keep_indices]
        sizes = [sizes[i] for i in keep_indices]
        shapes = [shapes[i] for i in keep_indices]
        confidences = [confidences[i] for i in keep_indices]
        obb_corners_list = [obb_corners_list[i] for i in keep_indices]
        if heading_hints is not None:
            heading_hints = [heading_hints[i] for i in keep_indices]
            if heading_confidences is None:
                heading_confidences = [0.0] * len(heading_hints)
            else:
                heading_confidences = [heading_confidences[i] for i in keep_indices]
            if directed_mask is None:
                directed_mask = [0] * len(heading_hints)
            else:
                directed_mask = [directed_mask[i] for i in keep_indices]
        if detection_ids is not None:
            detection_ids = [detection_ids[i] for i in keep_indices]
            if heading_hints is None:
                return meas, sizes, shapes, confidences, obb_corners_list, detection_ids
            return (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners_list,
                detection_ids,
                heading_hints,
                heading_confidences,
                directed_mask,
            )
        if heading_hints is not None:
            return (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners_list,
                heading_hints,
                directed_mask,
            )
        return meas, sizes, shapes, confidences, obb_corners_list

    # ------------------------------------------------------------------
    # Raw detection extraction
    # ------------------------------------------------------------------

    def _raw_detection_cap(self) -> int:
        """Cap raw detections to 2x MAX_TARGETS to bound cache size and filtering cost."""
        max_targets = max(1, int(self.params.get("MAX_TARGETS", 8)))
        return max_targets * 2

    def _extract_raw_detections(self, obb_data, return_class_ids: bool = False):
        """Extract raw OBB detections, sorted by confidence and capped by policy."""
        if obb_data is None or len(obb_data) == 0:
            if return_class_ids:
                return [], [], [], [], [], []
            return [], [], [], [], []

        xywhr = np.ascontiguousarray(obb_data.xywhr.cpu().numpy(), dtype=np.float32)
        conf_scores = np.ascontiguousarray(
            obb_data.conf.cpu().numpy(), dtype=np.float32
        )
        if return_class_ids:
            raw_cls = getattr(obb_data, "cls", None)
            if raw_cls is None:
                class_ids = np.full(len(conf_scores), -1, dtype=np.int32)
            else:
                class_ids = np.ascontiguousarray(
                    raw_cls.cpu().numpy(), dtype=np.int32
                ).reshape(-1)

        if xywhr.size == 0 or conf_scores.size == 0:
            if return_class_ids:
                return [], [], [], [], [], []
            return [], [], [], [], []

        cx = xywhr[:, 0]
        cy = xywhr[:, 1]
        w = xywhr[:, 2]
        h = xywhr[:, 3]
        angle_raw = xywhr[:, 4]
        # Runtime parity guard:
        # Some exported backends may report theta in degrees instead of radians.
        if np.nanmax(np.abs(angle_raw)) > (2.0 * np.pi + 1e-3):
            angle_rad = np.deg2rad(angle_raw)
        else:
            angle_rad = angle_raw

        angle_deg = np.rad2deg(angle_rad) % 180.0
        swap_mask = w < h
        major = np.where(swap_mask, h, w)
        minor = np.where(swap_mask, w, h)
        angle_deg = np.where(swap_mask, (angle_deg + 90.0) % 180.0, angle_deg)
        angle_rad_fixed = np.deg2rad(angle_deg).astype(np.float32)

        # Drop invalid detections early so NaN/Inf does not propagate into
        # assignment costs or Kalman updates.
        valid_mask = (
            np.isfinite(cx)
            & np.isfinite(cy)
            & np.isfinite(major)
            & np.isfinite(minor)
            & np.isfinite(angle_rad_fixed)
            & np.isfinite(conf_scores)
            & (major > 0)
            & (minor > 0)
        )
        if not np.all(valid_mask):
            dropped = int(len(valid_mask) - np.count_nonzero(valid_mask))
            if dropped > 0:
                logger.warning(
                    "Dropping %d invalid OBB detections with non-finite or non-positive geometry.",
                    dropped,
                )
            cx = cx[valid_mask]
            cy = cy[valid_mask]
            major = major[valid_mask]
            minor = minor[valid_mask]
            angle_rad_fixed = angle_rad_fixed[valid_mask]
            conf_scores = conf_scores[valid_mask]
            if return_class_ids:
                class_ids = class_ids[valid_mask]

        if cx.size == 0:
            if return_class_ids:
                return [], [], [], [], [], []
            return [], [], [], [], []

        sizes = (major * minor).astype(np.float32)
        ellipse_area = (np.pi * (major / 2.0) * (minor / 2.0)).astype(np.float32)
        aspect_ratio = np.divide(
            major,
            minor,
            out=np.zeros_like(major, dtype=np.float32),
            where=minor > 0,
        )

        meas_arr = np.column_stack((cx, cy, angle_rad_fixed)).astype(np.float32)
        shapes_arr = np.column_stack((ellipse_area, aspect_ratio)).astype(np.float32)

        # Build OBB corners from xywhr directly for stable geometry across runtimes
        # (ONNX/TensorRT can disagree on provided xyxyxyxy corner ordering/decoding).
        half_w = major / 2.0
        half_h = minor / 2.0
        x_offsets = np.stack((-half_w, half_w, half_w, -half_w), axis=1)
        y_offsets = np.stack((-half_h, -half_h, half_h, half_h), axis=1)
        cos_t = np.cos(angle_rad_fixed)
        sin_t = np.sin(angle_rad_fixed)
        x_coords = cx[:, None] + x_offsets * cos_t[:, None] - y_offsets * sin_t[:, None]
        y_coords = cy[:, None] + x_offsets * sin_t[:, None] + y_offsets * cos_t[:, None]
        corners = np.stack((x_coords, y_coords), axis=2).astype(np.float32, copy=False)

        cap = self._raw_detection_cap()
        order = np.argsort(conf_scores)[::-1]
        if len(order) > cap:
            order = order[:cap]

        meas_arr = np.ascontiguousarray(meas_arr[order], dtype=np.float32)
        sizes = np.ascontiguousarray(sizes[order], dtype=np.float32)
        shapes_arr = np.ascontiguousarray(shapes_arr[order], dtype=np.float32)
        conf_scores = np.ascontiguousarray(conf_scores[order], dtype=np.float32)
        corners = np.ascontiguousarray(corners[order], dtype=np.float32)
        if return_class_ids:
            class_ids = np.ascontiguousarray(class_ids[order], dtype=np.int32)

        meas = [meas_arr[i] for i in range(len(meas_arr))]
        sizes_list = sizes.tolist()
        shapes = [tuple(shapes_arr[i]) for i in range(len(shapes_arr))]
        confidences = conf_scores.tolist()
        obb_corners_list = [corners[i] for i in range(len(corners))]
        if return_class_ids:
            return (
                meas,
                sizes_list,
                shapes,
                confidences,
                obb_corners_list,
                class_ids.tolist(),
            )

        return meas, sizes_list, shapes, confidences, obb_corners_list

    # ------------------------------------------------------------------
    # Full raw-detection filter pipeline
    # ------------------------------------------------------------------

    def filter_raw_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb_corners_list,
        roi_mask=None,
        detection_ids=None,
        heading_hints=None,
        heading_confidences=None,
        directed_mask=None,
    ):
        """
        Apply vectorized confidence/size/ROI filtering, then custom OBB IOU suppression.
        This is shared by live detection and cached-raw detection paths.
        """
        if not meas:
            if heading_hints is None:
                return [], [], [], [], [], []
            return [], [], [], [], [], [], [], [], []

        conf_threshold = float(self.params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
        iou_threshold = float(self.params.get("YOLO_IOU_THRESHOLD", 0.7))
        max_targets = max(1, int(self.params.get("MAX_TARGETS", 8)))

        meas_arr = np.ascontiguousarray(np.asarray(meas, dtype=np.float32))
        sizes_arr = np.ascontiguousarray(np.asarray(sizes, dtype=np.float32))
        shapes_arr = np.ascontiguousarray(np.asarray(shapes, dtype=np.float32))
        conf_arr = np.ascontiguousarray(np.asarray(confidences, dtype=np.float32))

        if detection_ids is None:
            ids_arr = np.arange(len(meas_arr), dtype=np.int64)
        else:
            ids_arr = np.ascontiguousarray(
                np.asarray(_normalize_detection_ids(detection_ids), dtype=np.int64)
            )

        n = min(
            len(meas_arr), len(sizes_arr), len(shapes_arr), len(conf_arr), len(ids_arr)
        )
        if obb_corners_list:
            n = min(n, len(obb_corners_list))
        if heading_hints is not None:
            n = min(n, len(heading_hints))
            if heading_confidences is not None:
                n = min(n, len(heading_confidences))
            if directed_mask is not None:
                n = min(n, len(directed_mask))
        if n == 0:
            if heading_hints is None:
                return [], [], [], [], [], []
            return [], [], [], [], [], [], [], [], []

        meas_arr = meas_arr[:n]
        sizes_arr = sizes_arr[:n]
        shapes_arr = shapes_arr[:n]
        conf_arr = conf_arr[:n]
        ids_arr = ids_arr[:n]

        if obb_corners_list:
            obb_arr = np.ascontiguousarray(
                np.asarray(obb_corners_list, dtype=np.float32)
            )
            obb_arr = obb_arr[:n]
        else:
            obb_arr = np.empty((n, 4, 2), dtype=np.float32)

        if heading_hints is not None:
            heading_arr = np.ascontiguousarray(
                np.asarray(heading_hints, dtype=np.float32)
            )[:n]
            if heading_confidences is None:
                heading_conf_arr = np.zeros(n, dtype=np.float32)
            else:
                heading_conf_arr = np.ascontiguousarray(
                    np.asarray(heading_confidences, dtype=np.float32)
                )[:n]
            if directed_mask is None:
                directed_arr = np.zeros(n, dtype=np.uint8)
            else:
                directed_arr = np.ascontiguousarray(
                    np.asarray(directed_mask, dtype=np.uint8)
                )[:n]
        else:
            heading_arr = None
            heading_conf_arr = None
            directed_arr = None

        keep_mask = conf_arr >= conf_threshold

        if self.params.get("ENABLE_SIZE_FILTERING", False):
            min_size = float(self.params.get("MIN_OBJECT_SIZE", 0))
            max_size = float(self.params.get("MAX_OBJECT_SIZE", float("inf")))
            # Use ellipse area (shapes_arr[:, 0]) for size filtering to match
            # the GUI's circular-area formula.  Previously used sizes_arr
            # (OBB rect area = major*minor) which was ~27% larger.
            ellipse_area_arr = shapes_arr[:, 0] if shapes_arr.ndim == 2 else sizes_arr
            keep_mask &= (ellipse_area_arr >= min_size) & (ellipse_area_arr <= max_size)

        if self._advanced_config_value("enable_aspect_ratio_filtering", False):
            ref_ar = float(self._advanced_config_value("reference_aspect_ratio", 2.0))
            min_ar_mult = float(
                self._advanced_config_value("min_aspect_ratio_multiplier", 0.5)
            )
            max_ar_mult = float(
                self._advanced_config_value("max_aspect_ratio_multiplier", 2.0)
            )
            min_ar = ref_ar * min_ar_mult
            max_ar = ref_ar * max_ar_mult
            ar_arr = (
                shapes_arr[:, 1] if shapes_arr.ndim == 2 else np.ones(len(sizes_arr))
            )
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

        if not np.any(keep_mask):
            if heading_hints is None:
                return [], [], [], [], [], []
            return [], [], [], [], [], [], [], [], []

        meas_arr = meas_arr[keep_mask]
        sizes_arr = sizes_arr[keep_mask]
        shapes_arr = shapes_arr[keep_mask]
        conf_arr = conf_arr[keep_mask]
        ids_arr = ids_arr[keep_mask]
        obb_arr = obb_arr[keep_mask]
        if heading_arr is not None:
            heading_arr = heading_arr[keep_mask]
            heading_conf_arr = heading_conf_arr[keep_mask]
            directed_arr = directed_arr[keep_mask]

        meas_list = [meas_arr[i] for i in range(len(meas_arr))]
        sizes_list = sizes_arr.tolist()
        shapes_list = [tuple(shapes_arr[i]) for i in range(len(shapes_arr))]
        conf_list = conf_arr.tolist()
        ids_list = [int(v) for v in ids_arr.tolist()]
        obb_list = [obb_arr[i] for i in range(len(obb_arr))]
        heading_list = heading_arr.tolist() if heading_arr is not None else None
        heading_conf_list = (
            heading_conf_arr.tolist() if heading_conf_arr is not None else None
        )
        directed_list = directed_arr.tolist() if directed_arr is not None else None

        if len(meas_list) > 1:
            if heading_list is None:
                (
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    ids_list,
                ) = self._filter_overlapping_detections(
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    iou_threshold,
                    detection_ids=ids_list,
                )
            else:
                (
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    ids_list,
                    heading_list,
                    heading_conf_list,
                    directed_list,
                ) = self._filter_overlapping_detections(
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    iou_threshold,
                    detection_ids=ids_list,
                    heading_hints=heading_list,
                    heading_confidences=heading_conf_list,
                    directed_mask=directed_list,
                )

        if len(meas_list) > max_targets:
            idxs = np.argsort(sizes_list)[::-1][:max_targets]
            meas_list = [meas_list[i] for i in idxs]
            sizes_list = [sizes_list[i] for i in idxs]
            shapes_list = [shapes_list[i] for i in idxs]
            conf_list = [conf_list[i] for i in idxs]
            obb_list = [obb_list[i] for i in idxs]
            ids_list = [ids_list[i] for i in idxs]
            if heading_list is not None:
                heading_list = [heading_list[i] for i in idxs]
                heading_conf_list = [heading_conf_list[i] for i in idxs]
                directed_list = [directed_list[i] for i in idxs]

        if heading_list is None:
            return meas_list, sizes_list, shapes_list, conf_list, obb_list, ids_list
        return (
            meas_list,
            sizes_list,
            shapes_list,
            conf_list,
            obb_list,
            ids_list,
            heading_list,
            heading_conf_list,
            directed_list,
        )
