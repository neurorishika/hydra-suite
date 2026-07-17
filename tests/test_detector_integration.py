"""
Tests for BackgroundMeasurer class (background subtraction detection).

Tests cover:
- Basic object detection from foreground masks
- Conservative split for merged objects
- Size filtering
- Detection count limiting
- Measurement extraction (position, angle, size)
"""

import cv2
import numpy as np

from hydra_suite.core.background.measure import BackgroundMeasurer


def create_test_foreground_mask(num_objects=2, width=320, height=240, object_size=30):
    """Create a synthetic foreground mask with circular objects."""
    mask = np.zeros((height, width), dtype=np.uint8)

    if num_objects == 1:
        # Single object in center
        cv2.circle(mask, (width // 2, height // 2), object_size, 255, -1)
    elif num_objects == 2:
        # Two objects
        cv2.circle(mask, (width // 3, height // 2), object_size, 255, -1)
        cv2.circle(mask, (2 * width // 3, height // 2), object_size, 255, -1)
    elif num_objects >= 3:
        # Multiple objects in a grid
        positions = [
            (width // 4, height // 3),
            (3 * width // 4, height // 3),
            (width // 4, 2 * height // 3),
            (3 * width // 4, 2 * height // 3),
        ]
        for i, (x, y) in enumerate(positions[:num_objects]):
            cv2.circle(mask, (x, y), object_size, 255, -1)

    return mask


class TestBackgroundMeasurer:
    """Test suite for BackgroundMeasurer class."""

    def test_initialization(self):
        """Test detector initialization with basic parameters."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "REFERENCE_BODY_SIZE": 30.0,
            "RESIZE_FACTOR": 1.0,
        }

        detector = BackgroundMeasurer(params)

        assert detector.params == params

    def test_detect_single_object(self):
        """Test detection of a single object."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(num_objects=1, object_size=25)

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Should detect one object
        assert len(meas) == 1
        assert len(sizes) == 1
        assert len(shapes) == 1
        assert len(confidences) == 1

        # Check measurement format [cx, cy, angle]
        assert meas[0].shape == (3,)
        assert 0 <= meas[0][0] <= 320  # cx
        assert 0 <= meas[0][1] <= 240  # cy

        # Check that size is reasonable
        assert sizes[0] > 100  # Should be larger than min area

    def test_detect_multiple_objects(self):
        """Test detection of multiple objects."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(num_objects=3, object_size=25)

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Should detect three objects
        assert len(meas) == 3
        assert len(sizes) == 3
        assert len(shapes) == 3
        assert len(confidences) == 3

    def test_min_contour_area_filtering(self):
        """Test that small contours are filtered out."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 500,  # High threshold
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(
            num_objects=2, object_size=10
        )  # Small objects

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Small objects should be filtered
        assert len(meas) == 0

    def test_max_targets_limiting(self):
        """Test that detections are limited to MAX_TARGETS."""
        params = {
            "MAX_TARGETS": 2,  # Limit to 2
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(
            num_objects=4, object_size=25
        )  # Create 4 objects

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Should be limited to MAX_TARGETS
        assert len(meas) <= 2

    def test_size_filtering_enabled(self):
        """Test size-based filtering of detections."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 50,
            "ENABLE_SIZE_FILTERING": True,
            "MIN_OBJECT_SIZE": 1500,  # Minimum size threshold
            "MAX_OBJECT_SIZE": 3000,  # Maximum size threshold
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)

        # Create mask with objects of different sizes
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (80, 120), 15, 255, -1)  # Small (area ~700)
        cv2.circle(mask, (160, 120), 25, 255, -1)  # Medium (area ~2000)
        cv2.circle(mask, (240, 120), 40, 255, -1)  # Large (area ~5000)

        meas, sizes, shapes, confidences = detector.detect_objects(mask, frame_count=0)

        # Only medium object should pass filtering
        assert len(meas) == 1
        assert 1500 <= sizes[0] <= 3000

    def test_size_filtering_disabled(self):
        """Test that all objects are detected when size filtering is disabled."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "ENABLE_SIZE_FILTERING": False,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(num_objects=3, object_size=25)

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        assert len(meas) == 3

    def test_empty_mask(self):
        """Test detection on empty mask."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = np.zeros((240, 320), dtype=np.uint8)  # Empty mask

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        assert len(meas) == 0
        assert len(sizes) == 0
        assert len(shapes) == 0
        assert len(confidences) == 0

    def test_too_many_contours(self):
        """Test handling of excessive contours (noise)."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 1,  # Very low to allow noise
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)

        # Create very noisy mask with many small contours
        mask = np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255

        meas, sizes, shapes, confidences = detector.detect_objects(mask, frame_count=0)

        # Should handle gracefully (may return empty or limited results)
        assert isinstance(meas, list)
        assert isinstance(sizes, list)

    def test_shape_information(self):
        """Test that shape information (area, aspect ratio) is computed."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(num_objects=2, object_size=25)

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Check shapes format [(area, aspect_ratio), ...]
        for shape in shapes:
            assert len(shape) == 2
            area, aspect_ratio = shape
            assert area > 0
            assert aspect_ratio > 0

    def test_confidence_values(self):
        """Test that confidence values are returned (NaN for background subtraction)."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)
        fg_mask = create_test_foreground_mask(num_objects=2, object_size=25)

        meas, sizes, shapes, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Confidences should be NaN for background subtraction
        assert len(confidences) == len(meas)
        for conf in confidences:
            assert np.isnan(conf)

    def test_conservative_split_basic(self):
        """Test conservative split on a simple mask."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "MERGE_AREA_THRESHOLD": 1000,
        }

        detector = BackgroundMeasurer(params)

        # Create a large blob that might be merged objects
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(mask, (100, 80), (220, 160), 255, -1)  # Large rectangle

        result_mask = detector.apply_conservative_split(mask)

        # Should return a valid mask
        assert result_mask.shape == mask.shape
        assert result_mask.dtype == np.uint8

    def test_conservative_split_preserves_small_objects(self):
        """Test that conservative split doesn't over-erode small objects."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "REFERENCE_BODY_SIZE": 40.0,
            "RESIZE_FACTOR": 1.0,
        }

        detector = BackgroundMeasurer(params)

        # Create small objects that shouldn't be affected
        mask = create_test_foreground_mask(num_objects=2, object_size=20)
        original_white_pixels = np.sum(mask > 0)

        result_mask = detector.apply_conservative_split(mask)
        result_white_pixels = np.sum(result_mask > 0)

        # Should preserve most pixels for small objects below threshold
        assert result_white_pixels >= original_white_pixels * 0.5

    def test_conservative_split_skips_isolated_blobs_when_targets_missing(self):
        """Missing targets alone should not trigger erosion of single-animal blobs."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "REFERENCE_BODY_SIZE": 40.0,
            "RESIZE_FACTOR": 1.0,
        }

        detector = BackgroundMeasurer(params)

        mask = create_test_foreground_mask(num_objects=2, object_size=20)
        result_mask = detector.apply_conservative_split(mask.copy())

        assert np.array_equal(result_mask, mask)

    def test_conservative_split_uses_reference_body_size_for_clusters(self):
        """Large merged clusters should still be split based on body-size crowding."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "REFERENCE_BODY_SIZE": 30.0,
            "RESIZE_FACTOR": 1.0,
        }

        detector = BackgroundMeasurer(params)

        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (145, 120), 24, 255, -1)
        cv2.circle(mask, (175, 120), 24, 255, -1)
        original_white_pixels = np.sum(mask > 0)

        result_mask = detector.apply_conservative_split(mask.copy())
        result_white_pixels = np.sum(result_mask > 0)

        assert result_white_pixels < original_white_pixels

    def test_conservative_split_rethreshold_separates_weak_bridge(self):
        """Local re-thresholding should split blobs at the weak connection."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 1,  # no smoothing
            "CONSERVATIVE_ERODE_ITER": 2,  # boost = 2 → local_thresh = 20 * 1.5 = 30
            "THRESHOLD_VALUE": 20,
            "DARK_ON_LIGHT_BACKGROUND": True,
            "REFERENCE_BODY_SIZE": 30.0,
            "RESIZE_FACTOR": 1.0,
        }
        detector = BackgroundMeasurer(params)

        # Build a background (white) and a gray frame with two dark circles
        # connected by a faint bridge.
        background = np.full((240, 320), 200, dtype=np.uint8)
        gray = background.copy()
        # Two strong blobs (diff = 200-100 = 100, well above any threshold)
        cv2.circle(gray, (130, 120), 20, 100, -1)
        cv2.circle(gray, (180, 120), 20, 100, -1)
        # Faint bridge between them (diff = 200-175 = 25, above base 20, below boosted 30)
        cv2.rectangle(gray, (148, 115), (162, 125), 175, -1)

        # Build the fg_mask at base threshold (bridge is included)
        diff = cv2.subtract(background, gray)
        _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # Confirm bridge creates a single merged contour
        cnts_before, _ = cv2.findContours(
            fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert len(cnts_before) == 1, "bridge should merge the two blobs"

        result = detector.apply_conservative_split(fg_mask, gray, background)

        # After split the bridge should be removed → two contours
        cnts_after, _ = cv2.findContours(
            result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert (
            len(cnts_after) == 2
        ), f"expected 2 blobs after split, got {len(cnts_after)}"

    def test_conservative_split_boost_stays_local_within_cluster(self):
        """Weak local body regions should survive while the bridge is removed."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 2,
            "THRESHOLD_VALUE": 20,
            "DARK_ON_LIGHT_BACKGROUND": True,
            "REFERENCE_BODY_SIZE": 30.0,
            "RESIZE_FACTOR": 1.0,
        }
        detector = BackgroundMeasurer(params)

        background = np.full((240, 320), 200, dtype=np.uint8)
        gray = background.copy()
        cv2.circle(gray, (130, 120), 20, 172, -1)  # weaker animal, diff = 28
        cv2.circle(gray, (180, 120), 20, 100, -1)  # stronger animal, diff = 100
        cv2.rectangle(gray, (148, 115), (162, 125), 176, -1)  # weak bridge, diff = 24

        diff = cv2.subtract(background, gray)
        _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        cnts_before, _ = cv2.findContours(
            fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert len(cnts_before) == 1, "bridge should merge the two blobs"

        result = detector.apply_conservative_split(fg_mask, gray, background)

        cnts_after, _ = cv2.findContours(
            result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        assert (
            len(cnts_after) == 2
        ), f"expected 2 blobs after split, got {len(cnts_after)}"
        assert result[120, 130] == 255, "weak local body core should remain foreground"
        assert result[120, 155] == 0, "weak bridge should be removed"

    def test_measurement_coordinates(self):
        """Test that measurements contain valid coordinates."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)

        # Create object at known position
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (160, 120), 25, 255, -1)  # Center of image

        meas, sizes, shapes, confidences = detector.detect_objects(mask, frame_count=0)

        assert len(meas) == 1
        cx, cy, angle = meas[0]

        # Should be near center
        assert 140 <= cx <= 180
        assert 100 <= cy <= 140
        # Angle should be in radians
        assert -np.pi <= angle <= np.pi

    def test_ellipse_fitting(self):
        """Test that ellipse fitting works for elongated objects."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)

        # Create elongated ellipse
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.ellipse(mask, (160, 120), (40, 20), 0, 0, 360, 255, -1)

        meas, sizes, shapes, confidences = detector.detect_objects(mask, frame_count=0)

        assert len(meas) == 1
        assert len(shapes) == 1

        area, aspect_ratio = shapes[0]
        # Elongated ellipse should have aspect ratio > 1
        assert aspect_ratio > 1.5

    def test_different_frame_sizes(self):
        """Test detection on different frame sizes."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = BackgroundMeasurer(params)

        sizes = [(240, 320), (480, 640), (120, 160)]

        for height, width in sizes:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (width // 2, height // 2), 25, 255, -1)

            meas, det_sizes, shapes, confidences = detector.detect_objects(
                mask, frame_count=0
            )

            assert len(meas) == 1
            # Coordinates should be within frame
            assert 0 <= meas[0][0] <= width
            assert 0 <= meas[0][1] <= height
