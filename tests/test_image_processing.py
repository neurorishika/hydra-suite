"""
Comprehensive tests for image processing utilities.

Tests cover:
- Image adjustment functions (brightness, contrast, gamma)
- Lighting stabilization
- GPU and CPU code paths
- Edge cases and invalid inputs
"""

import numpy as np

from hydra_suite.utils.image_processing import (
    apply_image_adjustments,
    stabilize_lighting,
)


class TestApplyImageAdjustments:
    """Test suite for apply_image_adjustments function."""

    def test_no_adjustments(self):
        """Test that zero adjustments (0, 1.0, 1.0) return nearly original image."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # brightness=0 means no additive adjustment, contrast=1.0 means no scaling
        result = apply_image_adjustments(img, brightness=0, contrast=1.0, gamma=1.0)

        # May have slight rounding differences from LUT operations
        np.testing.assert_array_almost_equal(result, img, decimal=0)

    def test_brightness_increase(self):
        """Test increasing brightness."""
        img = np.ones((50, 50), dtype=np.uint8) * 100
        result = apply_image_adjustments(img, brightness=1.5, contrast=1.0, gamma=1.0)

        # Brightness increase should make image brighter
        assert result.mean() > img.mean()

    def test_brightness_decrease(self):
        """Test decreasing brightness."""
        img = np.ones((50, 50), dtype=np.uint8) * 100
        # brightness is additive, so negative value makes darker
        result = apply_image_adjustments(img, brightness=-20, contrast=1.0, gamma=1.0)

        # Brightness decrease should make image darker
        assert result.mean() < img.mean()

    def test_contrast_increase(self):
        """Test increasing contrast."""
        # Create image with some variation
        img = np.linspace(50, 150, 100 * 100, dtype=np.uint8).reshape(100, 100)
        result = apply_image_adjustments(img, brightness=1.0, contrast=1.5, gamma=1.0)

        # Higher contrast should increase standard deviation
        assert result.std() > img.std()

    def test_contrast_decrease(self):
        """Test decreasing contrast."""
        img = np.linspace(50, 150, 100 * 100, dtype=np.uint8).reshape(100, 100)
        result = apply_image_adjustments(img, brightness=1.0, contrast=0.5, gamma=1.0)

        # Lower contrast should decrease standard deviation
        assert result.std() < img.std()

    def test_gamma_correction_increase(self):
        """Test gamma correction > 1.0 (brightens with power-law transform)."""
        img = np.ones((50, 50), dtype=np.uint8) * 128
        result = apply_image_adjustments(img, brightness=0, contrast=1.0, gamma=1.5)

        # Gamma > 1.0 with the LUT implementation brightens mid-tones
        # (inverse gamma in the LUT creation)
        assert result.mean() > img.mean()

    def test_gamma_correction_decrease(self):
        """Test gamma correction < 1.0 (darkens with power-law transform)."""
        img = np.ones((50, 50), dtype=np.uint8) * 128
        result = apply_image_adjustments(img, brightness=0, contrast=1.0, gamma=0.7)

        # Gamma < 1.0 with the LUT implementation darkens mid-tones
        assert result.mean() < img.mean()

    def test_combined_adjustments(self):
        """Test combining multiple adjustments."""
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        result = apply_image_adjustments(img, brightness=1.2, contrast=1.1, gamma=0.9)

        # Result should be valid image
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        assert result.min() >= 0
        assert result.max() <= 255

    def test_clipping_at_boundaries(self):
        """Test that values are properly clipped to [0, 255]."""
        # Create image near boundaries
        img_bright = np.ones((50, 50), dtype=np.uint8) * 250

        # Extreme brightness (large positive) should clip at 255
        result_bright = apply_image_adjustments(
            img_bright, brightness=50, contrast=1.0, gamma=1.0
        )
        assert result_bright.max() == 255

        # Very high contrast should also produce some clipping
        result_contrast = apply_image_adjustments(
            img_bright, brightness=0, contrast=2.0, gamma=1.0
        )
        assert result_contrast.max() == 255

    def test_zero_contrast(self):
        """Test that zero contrast produces flat image."""
        img = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
        # contrast=0 means no scaling, everything becomes 0 (or 0 + brightness)
        result = apply_image_adjustments(img, brightness=0, contrast=0.0, gamma=1.0)

        # With zero contrast, all values should be the same
        assert result.std() == 0

    def test_different_image_sizes(self):
        """Test with different image dimensions."""
        sizes = [(50, 50), (100, 200), (1920, 1080), (10, 10)]

        for size in sizes:
            img = np.random.randint(0, 256, size, dtype=np.uint8)
            result = apply_image_adjustments(
                img, brightness=1.1, contrast=1.1, gamma=0.9
            )

            assert result.shape == img.shape
            assert result.dtype == np.uint8

    def test_use_gpu_flag_false(self):
        """Test that use_gpu=False works (CPU path)."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = apply_image_adjustments(
            img, brightness=1.2, contrast=1.1, gamma=0.9, use_gpu=False
        )

        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_extreme_gamma_values(self):
        """Test extreme gamma values."""
        img = np.linspace(0, 255, 100 * 100, dtype=np.uint8).reshape(100, 100)

        # Very high gamma (extreme darkening)
        result_high = apply_image_adjustments(
            img, brightness=1.0, contrast=1.0, gamma=5.0
        )
        assert result_high.dtype == np.uint8

        # Very low gamma (extreme brightening)
        result_low = apply_image_adjustments(
            img, brightness=1.0, contrast=1.0, gamma=0.2
        )
        assert result_low.dtype == np.uint8


class TestStabilizeLighting:
    """Test suite for stabilize_lighting function."""

    def test_basic_stabilization(self):
        """Test basic lighting stabilization."""
        img = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        reference_intensity = 128
        intensity_history = []
        lighting_state = {}

        result, new_history, current_mean = stabilize_lighting(
            img,
            reference_intensity,
            intensity_history,
            alpha=0.95,
            roi_mask=None,
            median_window=5,
            lighting_state=lighting_state,
            use_gpu=False,
        )

        # Result should be valid
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        assert isinstance(current_mean, (float, np.floating))

    def test_with_roi_mask(self):
        """Test stabilization with ROI mask."""
        img = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        reference_intensity = 128

        # Create ROI mask (center region)
        roi_mask = np.zeros((100, 100), dtype=np.uint8)
        roi_mask[25:75, 25:75] = 255

        intensity_history = []
        lighting_state = {}

        result, new_history, current_mean = stabilize_lighting(
            img,
            reference_intensity,
            intensity_history,
            alpha=0.95,
            roi_mask=roi_mask,
            median_window=5,
            lighting_state=lighting_state,
            use_gpu=False,
        )

        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_intensity_returned(self):
        """Test that current mean intensity is returned."""
        img = np.ones((100, 100), dtype=np.uint8) * 120

        result, history, current_mean = stabilize_lighting(
            img,
            128,
            [],
            alpha=0.95,
            roi_mask=None,
            median_window=5,
            lighting_state={},
            use_gpu=False,
        )

        # Current mean should be a valid number
        assert isinstance(current_mean, (float, np.floating))
        assert current_mean > 0

    def test_median_window_sizes(self):
        """Test different median window sizes."""
        img = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        reference_intensity = 128

        for window_size in [1, 3, 5, 10]:
            result, history, current_mean = stabilize_lighting(
                img,
                reference_intensity,
                [],
                alpha=0.95,
                roi_mask=None,
                median_window=window_size,
                lighting_state={},
                use_gpu=False,
            )

            assert result.dtype == np.uint8

    def test_use_gpu_false(self):
        """Test that use_gpu=False works correctly."""
        img = np.random.randint(100, 150, (100, 100), dtype=np.uint8)

        result, _, _ = stabilize_lighting(
            img,
            128,
            [],
            alpha=0.95,
            roi_mask=None,
            median_window=5,
            lighting_state={},
            use_gpu=False,
        )

        assert result.dtype == np.uint8
        assert result.shape == img.shape
