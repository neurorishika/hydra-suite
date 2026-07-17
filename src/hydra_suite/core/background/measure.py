"""Measure objects from a background-subtraction foreground mask."""

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Copied from core/detectors/_utils.py: core/background must not import from
# core/detectors (and detectors is being retired from the bg-sub path).
_CONSERVATIVE_SPLIT_MIN_ANIMALS = 1.6


class BackgroundMeasurer:
    """
    Detects objects in foreground masks and extracts measurements.
    """

    def __init__(self, params):
        self.params = params

    def _local_threshold_map(self, sub_diff):
        """Build a spatially varying threshold map for a suspicious contour.

        The boost should act locally, not as one contour-wide cutoff. We do
        that by pulling the threshold upward toward nearby strong foreground
        support, which trims weak bridges while keeping moderately weak body
        regions whose local support remains low.
        """
        diff_f = sub_diff.astype(np.float32)
        blur_kernel = int(self.params.get("CONSERVATIVE_KERNEL_SIZE", 3) or 1)
        blur_kernel = max(1, blur_kernel | 1)
        if blur_kernel > 1:
            diff_f = cv2.GaussianBlur(diff_f, (blur_kernel, blur_kernel), 0)

        expected_body_area = self._expected_body_area()
        support_kernel_size = max(3, 2 * blur_kernel + 1)
        if expected_body_area and expected_body_area > 0:
            body_radius = math.sqrt(expected_body_area / math.pi)
            support_kernel_size = max(support_kernel_size, int(round(body_radius)) | 1)
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (support_kernel_size, support_kernel_size)
        )
        local_support = cv2.dilate(diff_f, support_kernel)

        base_thresh = float(self.params.get("THRESHOLD_VALUE", 20) or 20)
        boost_steps = max(
            0.0, float(self.params.get("CONSERVATIVE_ERODE_ITER", 1) or 0.0)
        )
        boost_fraction = min(0.85, 0.25 * boost_steps)
        threshold_map = base_thresh + boost_fraction * np.maximum(
            local_support - base_thresh, 0.0
        )
        return diff_f, threshold_map

    def _local_rethreshold(self, sub_diff, sub_mask):
        """Re-threshold a diff sub-region using a local threshold map.

        Only tightens — never expands beyond the existing foreground mask.
        """
        diff_f, threshold_map = self._local_threshold_map(sub_diff)
        tighter = np.where(diff_f > threshold_map, 255, 0).astype(np.uint8)
        tighter = cv2.bitwise_and(sub_mask, tighter)

        min_area = max(1, int(self.params.get("MIN_CONTOUR_AREA", 1) or 1))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tighter)
        if num_labels <= 1:
            return tighter

        filtered = np.zeros_like(tighter)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == label] = 255
        return filtered

    def _expected_body_area(self):
        """Estimate one-animal area in resized pixels from reference size or filters.

        Prefers MIN_OBJECT_SIZE (user-calibrated minimum pixel area for a
        single animal) over the circular approximation from
        REFERENCE_BODY_SIZE, which dramatically overestimates area for
        elongated animals and prevents the split from triggering.
        """
        min_size = float(self.params.get("MIN_OBJECT_SIZE", 0.0) or 0.0)
        if min_size > 0:
            return min_size

        max_size = float(self.params.get("MAX_OBJECT_SIZE", 0.0) or 0.0)
        if max_size > 0 and np.isfinite(max_size):
            return 0.5 * max_size

        # Fallback: circular estimate from reference body size
        reference_body_size = float(self.params.get("REFERENCE_BODY_SIZE", 0.0) or 0.0)
        resize_factor = float(self.params.get("RESIZE_FACTOR", 1.0) or 1.0)
        if reference_body_size > 0:
            scaled_body_size = reference_body_size * resize_factor
            radius = scaled_body_size / 2.0
            return math.pi * radius * radius
        return None

    def _should_split_contour(self, contour_area):
        """Split only contours whose area suggests multiple nearby animals."""
        expected_body_area = self._expected_body_area()
        if not expected_body_area or expected_body_area <= 0:
            return False

        estimated_animals = contour_area / expected_body_area
        return estimated_animals >= _CONSERVATIVE_SPLIT_MIN_ANIMALS

    def apply_conservative_split(self, fg_mask, gray=None, background=None):
        """Split merged blobs by locally raising the threshold.

        If *gray* and *background* are provided the split uses a tighter
        threshold on the raw difference image inside suspicious regions,
        which preserves animal shape better than erosion.  Falls back to
        simple erosion when the raw images are unavailable (e.g. cached
        detection replays).
        """
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        suspicious = [
            cv2.boundingRect(c)
            for c in cnts
            if self._should_split_contour(cv2.contourArea(c))
        ]
        if not suspicious:
            return fg_mask

        p = self.params
        boost = p.get("CONSERVATIVE_ERODE_ITER", 1)

        if gray is not None and background is not None:
            # Compute raw difference for re-thresholding
            dark_on_light = p.get("DARK_ON_LIGHT_BACKGROUND", True)
            if dark_on_light:
                diff = cv2.subtract(background, gray)
            else:
                diff = cv2.subtract(gray, background)

            for bx, by, bw, bh in suspicious:
                sub_diff = diff[by : by + bh, bx : bx + bw]
                sub_mask = fg_mask[by : by + bh, bx : bx + bw]
                fg_mask[by : by + bh, bx : bx + bw] = self._local_rethreshold(
                    sub_diff, sub_mask
                )
        else:
            # Fallback: simple erosion when raw images are unavailable
            k = p.get("CONSERVATIVE_KERNEL_SIZE", 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            for bx, by, bw, bh in suspicious:
                sub = fg_mask[by : by + bh, bx : bx + bw]
                fg_mask[by : by + bh, bx : bx + bw] = cv2.erode(
                    sub, kernel, iterations=boost
                )

        return fg_mask

    def detect_objects(
        self, fg_mask: np.ndarray, frame_count: int
    ) -> tuple[list, list, list, list]:
        """Detect and measure objects from the final foreground mask.

        Returns:
            meas: list of np.array([cx, cy, angle_radians], float32)
            sizes: list of ellipse areas (px^2)
            shapes: list of (ellipse_area, aspect_ratio) tuples
            confidences: list of np.nan -- confidence is not feasible for
                background subtraction (quality is too context-specific).
        """
        p = self.params
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        N = p["MAX_TARGETS"]
        max_allowed_contours = N * p.get("MAX_CONTOUR_MULTIPLIER", 20)

        if len(cnts) > max_allowed_contours:
            logger.debug(
                f"Frame {frame_count}: Too many contours ({len(cnts)}), skipping."
            )
            return [], [], [], []

        meas, sizes, shapes, confidences = [], [], [], []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < p["MIN_CONTOUR_AREA"] or len(c) < 5:
                continue

            (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)

            if ax1 < ax2:
                ax1, ax2 = ax2, ax1
                ang = (ang + 90) % 180

            # Detection confidence not feasible for background subtraction
            # Quality is too context-specific (lighting, camera, species, etc.)
            confidence = np.nan

            # Use ellipse area for size filtering (consistent with YOLO OBB path
            # and the GUI's circular-area formula).
            ellipse_area = np.pi * (ax1 / 2.0) * (ax2 / 2.0)
            meas.append(np.array([cx, cy, np.deg2rad(ang)], np.float32))
            sizes.append(ellipse_area)
            shapes.append((ellipse_area, ax1 / ax2 if ax2 > 0 else 0))
            confidences.append(confidence)

        if meas and p.get("ENABLE_SIZE_FILTERING", False):
            min_size = p.get("MIN_OBJECT_SIZE", 0)
            max_size = p.get("MAX_OBJECT_SIZE", float("inf"))

            original_count = len(meas)
            filtered = [
                (m, s, sh, conf)
                for m, s, sh, conf in zip(meas, sizes, shapes, confidences)
                if min_size <= s <= max_size
            ]

            if filtered:
                meas, sizes, shapes, confidences = zip(*filtered)
                meas, sizes, shapes, confidences = (
                    list(meas),
                    list(sizes),
                    list(shapes),
                    list(confidences),
                )
            else:
                meas, sizes, shapes, confidences = [], [], [], []

            if len(meas) != original_count:
                logger.debug(
                    f"Size filtering: {original_count} -> {len(meas)} detections"
                )

        if len(meas) > N:
            idxs = np.argsort(sizes)[::-1][:N]
            meas = [meas[i] for i in idxs]
            sizes = [sizes[i] for i in idxs]
            shapes = [shapes[i] for i in idxs]
            confidences = [confidences[i] for i in idxs]

        return meas, sizes, shapes, confidences
