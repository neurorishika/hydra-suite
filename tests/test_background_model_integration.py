"""
Integration tests for BackgroundModel class.

Tests cover:
- Background model initialization
- GPU acceleration setup
- Background priming
- Foreground mask generation
- Adaptive background updates
- Different parameter configurations
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.core.background.model import BackgroundModel


def create_test_video_for_background(
    num_frames=50, width=320, height=240, moving_object=True
):
    """Create a test video with static background and optional moving object."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        # Static background (gray with some texture)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        # Add some texture
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if moving_object:
            # Add moving dark object
            x = int((i / num_frames) * width * 0.8 + width * 0.1)
            y = height // 2
            cv2.circle(frame, (x, y), 20, (50, 50, 50), -1)

        writer.write(frame)

    writer.release()
    return temp_path


class TestBackgroundModelInitialization:
    """Test BackgroundModel initialization and setup."""

    def test_basic_initialization(self):
        """Test basic initialization with default params."""
        params = {
            "BRIGHTNESS": 1.0,
            "CONTRAST": 1.0,
            "GAMMA": 1.0,
            "THRESHOLD_VALUE": 30,
            "MORPH_KERNEL_SIZE": 3,
            "BACKGROUND_PRIME_FRAMES": 10,
            "ADAPTIVE_LEARNING_RATE": 0.01,
            "ENABLE_GPU_BACKGROUND": False,
        }

        bg_model = BackgroundModel(params)

        assert bg_model.params == params
        assert bg_model.lightest_background is None
        assert bg_model.adaptive_background is None
        assert bg_model.reference_intensity is None
        assert bg_model.use_gpu is False

    def test_gpu_disabled_by_default(self):
        """Test that GPU is disabled when not explicitly enabled."""
        params = {
            "BRIGHTNESS": 1.0,
            "CONTRAST": 1.0,
            "GAMMA": 1.0,
            "THRESHOLD_VALUE": 30,
            "MORPH_KERNEL_SIZE": 3,
            "BACKGROUND_PRIME_FRAMES": 10,
            "ADAPTIVE_LEARNING_RATE": 0.01,
            "ENABLE_GPU_BACKGROUND": False,
        }

        bg_model = BackgroundModel(params)

        assert bg_model.use_gpu is False
        assert bg_model.gpu_type is None

    def test_different_parameter_values(self):
        """Test initialization with various parameter values."""
        params = {
            "BRIGHTNESS": 1.2,
            "CONTRAST": 1.1,
            "GAMMA": 0.9,
            "THRESHOLD_VALUE": 50,
            "MORPH_KERNEL_SIZE": 5,
            "BACKGROUND_PRIME_FRAMES": 20,
            "ADAPTIVE_LEARNING_RATE": 0.05,
            "ENABLE_GPU_BACKGROUND": False,
            "DARK_ON_LIGHT_BACKGROUND": True,
        }

        bg_model = BackgroundModel(params)

        assert bg_model.params["THRESHOLD_VALUE"] == 50
        assert bg_model.params["MORPH_KERNEL_SIZE"] == 5
        assert bg_model.params["ADAPTIVE_LEARNING_RATE"] == 0.05


class TestBackgroundPriming:
    """Test background priming functionality."""

    def test_prime_background_basic(self):
        """Test basic background priming."""
        video_path = create_test_video_for_background(30, moving_object=False)

        try:
            params = {
                "BRIGHTNESS": 1.0,
                "CONTRAST": 1.0,
                "GAMMA": 1.0,
                "THRESHOLD_VALUE": 30,
                "MORPH_KERNEL_SIZE": 3,
                "BACKGROUND_PRIME_FRAMES": 10,
                "ADAPTIVE_LEARNING_RATE": 0.01,
                "ENABLE_GPU_BACKGROUND": False,
                "RESIZE_FACTOR": 1.0,
            }

            bg_model = BackgroundModel(params)
            cap = cv2.VideoCapture(video_path)

            bg_model.prime_background(cap)

            # After priming, backgrounds should be initialized
            assert bg_model.lightest_background is not None
            assert bg_model.adaptive_background is not None
            assert bg_model.reference_intensity is not None

            # Background should be the right shape
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            assert bg_model.lightest_background.shape == gray.shape

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_prime_with_resize(self):
        """Test background priming with frame resizing."""
        video_path = create_test_video_for_background(
            30, width=640, height=480, moving_object=False
        )

        try:
            params = {
                "BRIGHTNESS": 1.0,
                "CONTRAST": 1.0,
                "GAMMA": 1.0,
                "THRESHOLD_VALUE": 30,
                "MORPH_KERNEL_SIZE": 3,
                "BACKGROUND_PRIME_FRAMES": 10,
                "ADAPTIVE_LEARNING_RATE": 0.01,
                "ENABLE_GPU_BACKGROUND": False,
                "RESIZE_FACTOR": 0.5,  # Resize to 50%
            }

            bg_model = BackgroundModel(params)
            cap = cv2.VideoCapture(video_path)

            bg_model.prime_background(cap)

            # Background should be resized
            expected_shape = (240, 320)  # 50% of (480, 640)
            assert bg_model.lightest_background.shape == expected_shape

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_prime_with_roi_mask(self):
        """Test background priming with ROI mask."""
        video_path = create_test_video_for_background(
            30, width=320, height=240, moving_object=False
        )

        try:
            # Create ROI mask (center region)
            roi_mask = np.zeros((240, 320), dtype=np.uint8)
            roi_mask[60:180, 80:240] = 255

            params = {
                "BRIGHTNESS": 1.0,
                "CONTRAST": 1.0,
                "GAMMA": 1.0,
                "THRESHOLD_VALUE": 30,
                "MORPH_KERNEL_SIZE": 3,
                "BACKGROUND_PRIME_FRAMES": 10,
                "ADAPTIVE_LEARNING_RATE": 0.01,
                "ENABLE_GPU_BACKGROUND": False,
                "RESIZE_FACTOR": 1.0,
                "ROI_MASK": roi_mask,
            }

            bg_model = BackgroundModel(params)
            cap = cv2.VideoCapture(video_path)

            bg_model.prime_background(cap)

            # Should still prime successfully with ROI
            assert bg_model.lightest_background is not None
            assert bg_model.reference_intensity is not None

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_prime_with_few_frames(self):
        """Test priming with fewer frames than requested."""
        video_path = create_test_video_for_background(
            5, moving_object=False
        )  # Only 5 frames

        try:
            params = {
                "BRIGHTNESS": 1.0,
                "CONTRAST": 1.0,
                "GAMMA": 1.0,
                "THRESHOLD_VALUE": 30,
                "MORPH_KERNEL_SIZE": 3,
                "BACKGROUND_PRIME_FRAMES": 20,  # Request 20 but only 5 available
                "ADAPTIVE_LEARNING_RATE": 0.01,
                "ENABLE_GPU_BACKGROUND": False,
                "RESIZE_FACTOR": 1.0,
            }

            bg_model = BackgroundModel(params)
            cap = cv2.VideoCapture(video_path)

            bg_model.prime_background(cap)

            # Should still work with fewer frames
            assert bg_model.lightest_background is not None

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestBackgroundUpdate:
    """Test adaptive background update functionality."""

    def test_update_and_get_background(self):
        """Test updating adaptive background."""
        video_path = create_test_video_for_background(30, moving_object=False)

        try:
            params = {
                "BRIGHTNESS": 1.0,
                "CONTRAST": 1.0,
                "GAMMA": 1.0,
                "THRESHOLD_VALUE": 30,
                "MORPH_KERNEL_SIZE": 3,
                "BACKGROUND_PRIME_FRAMES": 10,
                "ADAPTIVE_LEARNING_RATE": 0.01,
                "ENABLE_GPU_BACKGROUND": False,
                "RESIZE_FACTOR": 1.0,
            }

            bg_model = BackgroundModel(params)
            cap = cv2.VideoCapture(video_path)

            bg_model.prime_background(cap)

            # Read a frame and update background
            cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            bg = bg_model.update_and_get_background(gray, roi_mask=None)

            assert bg is not None
            assert bg.shape == gray.shape
            assert bg.dtype == np.uint8

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_update_before_priming_returns_none(self):
        """Test that update before priming returns None."""
        params = {
            "BRIGHTNESS": 1.0,
            "CONTRAST": 1.0,
            "GAMMA": 1.0,
            "THRESHOLD_VALUE": 30,
            "MORPH_KERNEL_SIZE": 3,
            "BACKGROUND_PRIME_FRAMES": 10,
            "ADAPTIVE_LEARNING_RATE": 0.01,
            "ENABLE_GPU_BACKGROUND": False,
        }

        bg_model = BackgroundModel(params)

        # Create a fake gray frame
        gray = np.ones((240, 320), dtype=np.uint8) * 128

        bg = bg_model.update_and_get_background(gray, roi_mask=None)

        # Should return None if not primed
        assert bg is None


class TestForegroundMaskGeneration:
    """Test foreground mask generation."""

    def test_generate_foreground_mask_basic(self):
        """Test basic foreground mask generation."""
        video_path = create_test_video_for_background(30, moving_object=True)

        try:
            params = {
                "BRIGHTNESS": 1.0,
                "CONTRAST": 1.0,
                "GAMMA": 1.0,
                "THRESHOLD_VALUE": 30,
                "MORPH_KERNEL_SIZE": 3,
                "BACKGROUND_PRIME_FRAMES": 10,
                "ADAPTIVE_LEARNING_RATE": 0.01,
                "ENABLE_GPU_BACKGROUND": False,
                "RESIZE_FACTOR": 1.0,
                "DARK_ON_LIGHT_BACKGROUND": True,
            }

            bg_model = BackgroundModel(params)
            cap = cv2.VideoCapture(video_path)

            bg_model.prime_background(cap)

            # Read a frame with moving object
            cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            bg = bg_model.update_and_get_background(gray, roi_mask=None)
            fg_mask = bg_model.generate_foreground_mask(gray, bg)

            # Check mask properties
            assert fg_mask is not None
            assert fg_mask.shape == gray.shape
            assert fg_mask.dtype == np.uint8
            assert fg_mask.max() <= 255
            assert fg_mask.min() >= 0

            # Should detect the moving object (non-zero pixels)
            assert np.sum(fg_mask > 0) > 0

            cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_dark_on_light_vs_light_on_dark(self):
        """Test different object/background contrast modes."""
        video_path = create_test_video_for_background(30, moving_object=True)

        try:
            for dark_on_light in [True, False]:
                params = {
                    "BRIGHTNESS": 1.0,
                    "CONTRAST": 1.0,
                    "GAMMA": 1.0,
                    "THRESHOLD_VALUE": 30,
                    "MORPH_KERNEL_SIZE": 3,
                    "BACKGROUND_PRIME_FRAMES": 10,
                    "ADAPTIVE_LEARNING_RATE": 0.01,
                    "ENABLE_GPU_BACKGROUND": False,
                    "RESIZE_FACTOR": 1.0,
                    "DARK_ON_LIGHT_BACKGROUND": dark_on_light,
                }

                bg_model = BackgroundModel(params)
                cap = cv2.VideoCapture(video_path)

                bg_model.prime_background(cap)

                cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                bg = bg_model.update_and_get_background(gray, roi_mask=None)
                fg_mask = bg_model.generate_foreground_mask(gray, bg)

                assert fg_mask is not None
                assert fg_mask.dtype == np.uint8

                cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_different_morph_kernel_sizes(self):
        """Test foreground mask with different morphological kernel sizes."""
        video_path = create_test_video_for_background(30, moving_object=True)

        try:
            for kernel_size in [3, 5, 7]:
                params = {
                    "BRIGHTNESS": 1.0,
                    "CONTRAST": 1.0,
                    "GAMMA": 1.0,
                    "THRESHOLD_VALUE": 30,
                    "MORPH_KERNEL_SIZE": kernel_size,
                    "BACKGROUND_PRIME_FRAMES": 10,
                    "ADAPTIVE_LEARNING_RATE": 0.01,
                    "ENABLE_GPU_BACKGROUND": False,
                    "RESIZE_FACTOR": 1.0,
                }

                bg_model = BackgroundModel(params)
                cap = cv2.VideoCapture(video_path)

                bg_model.prime_background(cap)

                cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                bg = bg_model.update_and_get_background(gray, roi_mask=None)
                fg_mask = bg_model.generate_foreground_mask(gray, bg)

                assert fg_mask is not None
                assert fg_mask.dtype == np.uint8

                cap.release()
        finally:
            Path(video_path).unlink(missing_ok=True)
