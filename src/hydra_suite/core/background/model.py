"""
Background modeling utilities for multi-object tracking.
Functionally identical to the original implementation's background logic.
"""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from hydra_suite.utils.gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    NUMBA_AVAILABLE,
    cp,
    cupy_ndimage,
    njit,
    prange,
    torch,
)
from hydra_suite.utils.image_processing import apply_image_adjustments

# Provide fallback decorators if Numba not available
if not NUMBA_AVAILABLE:

    def njit(*args, **kwargs):
        """CPU fallback for numba.njit when Numba is unavailable; returns the function unchanged."""

        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):
        """CPU fallback for numba.prange when Numba is unavailable; delegates to built-in range."""
        return range(*args, **kwargs)


logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True, parallel=True)
def _update_adaptive_background_numba(background, gray, learning_rate):
    """Numba-accelerated pixel-wise adaptive background update.

    Computes: background = (1 - lr) * background + lr * gray
    Using parallel loops for maximum performance on large arrays.
    """
    height, width = background.shape
    inv_lr = 1.0 - learning_rate

    for i in prange(height):
        for j in range(width):
            background[i, j] = inv_lr * background[i, j] + learning_rate * gray[i, j]

    return background


def _update_adaptive_background_gpu(background_gpu, gray_gpu, learning_rate):
    """GPU-accelerated background update using CuPy.

    Args:
        background_gpu: CuPy array on GPU
        gray_gpu: CuPy array on GPU
        learning_rate: float

    Returns:
        Updated background (CuPy array on GPU)
    """
    return (1.0 - learning_rate) * background_gpu + learning_rate * gray_gpu


def _generate_foreground_mask_gpu(gray_gpu, background_gpu, params):
    """GPU-accelerated foreground mask generation using CuPy.

    Args:
        gray_gpu: CuPy array on GPU
        background_gpu: CuPy array on GPU
        params: Parameter dictionary

    Returns:
        Foreground mask (CuPy array on GPU)
    """
    dark_on_light = params.get("DARK_ON_LIGHT_BACKGROUND", True)

    if dark_on_light:
        diff = cp.maximum(background_gpu - gray_gpu, 0)
    else:
        diff = cp.maximum(gray_gpu - background_gpu, 0)

    # Thresholding on GPU
    fg_mask = (diff > params["THRESHOLD_VALUE"]).astype(cp.uint8) * 255

    # Morphological operations on GPU using elliptical kernels to match CPU path
    ksz = params["MORPH_KERNEL_SIZE"]
    kernel = cp.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))

    # Open (erode then dilate) - removes noise
    fg_mask = cupy_ndimage.grey_erosion(fg_mask, footprint=kernel)
    fg_mask = cupy_ndimage.grey_dilation(fg_mask, footprint=kernel)

    # Close (dilate then erode) - fills gaps
    fg_mask = cupy_ndimage.grey_dilation(fg_mask, footprint=kernel)
    fg_mask = cupy_ndimage.grey_erosion(fg_mask, footprint=kernel)

    # Additional dilation if enabled
    if params.get("ENABLE_ADDITIONAL_DILATION", False):
        dil_ksz = params.get("DILATION_KERNEL_SIZE", 3)
        dil_iter = params.get("DILATION_ITERATIONS", 2)
        dil_kernel = cp.asarray(
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksz, dil_ksz))
        )
        for _ in range(dil_iter):
            fg_mask = cupy_ndimage.grey_dilation(fg_mask, footprint=dil_kernel)

    return fg_mask


def _update_adaptive_background_mps(background_torch, gray_torch, learning_rate):
    """MPS-accelerated background update using PyTorch (Apple Silicon).

    Args:
        background_torch: PyTorch tensor on MPS device
        gray_torch: PyTorch tensor on MPS device
        learning_rate: float

    Returns:
        Updated background (PyTorch tensor on MPS device)
    """
    return (1.0 - learning_rate) * background_torch + learning_rate * gray_torch


def _generate_foreground_mask_mps(gray_torch, background_torch, params, device):
    """MPS-accelerated foreground mask generation using PyTorch (Apple Silicon).

    Args:
        gray_torch: PyTorch tensor on MPS device
        background_torch: PyTorch tensor on MPS device
        params: Parameter dictionary
        device: torch.device for MPS

    Returns:
        Foreground mask (numpy array)
    """
    dark_on_light = params.get("DARK_ON_LIGHT_BACKGROUND", True)

    if dark_on_light:
        diff = torch.clamp(background_torch - gray_torch, min=0)
    else:
        diff = torch.clamp(gray_torch - background_torch, min=0)

    # Thresholding on GPU, then transfer to CPU for morphological ops.
    # Morphology uses OpenCV's elliptical kernels for consistency with the
    # CPU and CUDA paths.  The threshold step is the expensive part;
    # morphology on a small binary mask is fast on CPU.
    threshold = params["THRESHOLD_VALUE"]
    fg_mask = ((diff > threshold).to(torch.uint8) * 255).cpu().numpy()

    ksz = params["MORPH_KERNEL_SIZE"]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, ker)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, ker)

    if params.get("ENABLE_ADDITIONAL_DILATION", False):
        dil_ksz = params.get("DILATION_KERNEL_SIZE", 3)
        dil_iter = params.get("DILATION_ITERATIONS", 2)
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksz, dil_ksz))
        fg_mask = cv2.dilate(fg_mask, dil_kernel, iterations=dil_iter)

    return fg_mask


class BackgroundModel:
    """
    Manages background models for foreground detection in tracking.
    Supports GPU acceleration via:
      - CUDA (NVIDIA GPUs) using CuPy
      - MPS (Apple Silicon) using PyTorch
      - CPU fallback with Numba JIT optimization
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.lightest_background: Optional[np.ndarray] = None
        self.adaptive_background: Optional[np.ndarray] = None
        self.reference_intensity: Optional[float] = None

        # GPU acceleration setup
        self.use_gpu = False
        self.gpu_type = None  # 'cuda', 'mps', or None
        self.gpu_device = None
        self.torch_device = None
        self._setup_gpu_acceleration()

    def _setup_gpu_acceleration(self) -> None:
        """
        Initialize GPU acceleration if available.
        Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU fallback
        """
        if not self.params.get("ENABLE_GPU_BACKGROUND", False):
            logger.info("GPU background processing disabled in config")
            return

        device_id = self.params.get("GPU_DEVICE_ID", 0)

        # Try CUDA first (NVIDIA GPUs via CuPy)
        if CUDA_AVAILABLE:
            try:
                self.gpu_device = cp.cuda.Device(device_id)
                self.gpu_type = "cuda"
                self.use_gpu = True
                logger.info(
                    f"GPU background processing enabled: CUDA device {device_id}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA device {device_id}: {e}")

        # Try MPS (Apple Silicon via PyTorch)
        if MPS_AVAILABLE:
            try:
                self.torch_device = torch.device("mps")
                # Test MPS with a small operation
                _ = torch.zeros(1, device=self.torch_device)
                self.gpu_type = "mps"
                self.use_gpu = True
                logger.info("GPU background processing enabled: Apple MPS")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize MPS: {e}")

        # CPU fallback
        if self.params.get("ENABLE_GPU_BACKGROUND", False):
            logger.info(
                "GPU background processing requested but no GPU available. "
                "Using CPU with Numba JIT optimization."
            )

    @staticmethod
    def _prepare_roi_mask(
        cap: cv2.VideoCapture, ROI_mask, resize_f: float
    ) -> Optional[np.ndarray]:
        """Pre-resize ROI mask to match detection frame dimensions."""
        if ROI_mask is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, sample_frame = cap.read()
        if not ret:
            return None
        if resize_f < 1.0:
            sample_frame = cv2.resize(
                sample_frame,
                (0, 0),
                fx=resize_f,
                fy=resize_f,
                interpolation=cv2.INTER_AREA,
            )
        gray_sample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        if resize_f != 1.0:
            return cv2.resize(
                ROI_mask,
                (gray_sample.shape[1], gray_sample.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return ROI_mask

    @staticmethod
    def _iqr_mean_intensity(pixels: np.ndarray) -> Optional[float]:
        """Compute mean intensity within the inter-quartile range."""
        if len(pixels) <= 100:
            return None
        p25, p75 = np.percentile(pixels, [25, 75])
        mask = (pixels >= p25) & (pixels <= p75)
        if np.sum(mask) > 0:
            return float(np.mean(pixels[mask]))
        return None

    def _fallback_reference_intensity(
        self, bg_temp: np.ndarray, roi_resized: Optional[np.ndarray]
    ) -> float:
        """Compute reference intensity as fallback when no IQR samples exist."""
        if roi_resized is not None:
            roi_bg_pixels = bg_temp[roi_resized > 0]
            return (
                float(np.mean(roi_bg_pixels))
                if len(roi_bg_pixels) > 0
                else float(np.mean(bg_temp))
            )
        return float(np.mean(bg_temp))

    def prime_background(self, cap: cv2.VideoCapture) -> None:
        """
        Initialize background model using "lightest pixel" method with lighting reference.
        This is an exact port of the original `prime_lightest_background` method.
        """
        p = self.params
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or p["BACKGROUND_PRIME_FRAMES"] < 1:
            return

        count = min(p["BACKGROUND_PRIME_FRAMES"], total)
        br, ct, gm = p["BRIGHTNESS"], p["CONTRAST"], p["GAMMA"]
        ROI_mask = p.get("ROI_MASK", None)
        resize_f = p.get("RESIZE_FACTOR", 1.0)

        # Evenly-spaced rather than random: deterministic without needing a
        # seed (which makes the detection cache key honest -- see
        # core/inference/cache/keys.py) and strictly guarantees the temporal
        # coverage random sampling only achieves on average.
        idxs = [int(round(i)) for i in np.linspace(0, total - 1, count)]
        bg_temp = None
        intensity_samples = []

        roi_resized = self._prepare_roi_mask(cap, ROI_mask, resize_f)

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if resize_f < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_f,
                    fy=resize_f,
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(gray, br, ct, gm, self.use_gpu)

            pixels = (
                gray[roi_resized > 0] if roi_resized is not None else gray.flatten()
            )
            sample = self._iqr_mean_intensity(pixels)
            if sample is not None:
                intensity_samples.append(sample)

            if bg_temp is None:
                bg_temp = gray.astype(np.float32)
            else:
                bg_temp = np.maximum(bg_temp, gray.astype(np.float32))

        if bg_temp is not None:
            self.lightest_background = bg_temp
            self.adaptive_background = bg_temp.copy()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if intensity_samples:
                self.reference_intensity = np.median(intensity_samples)
                logger.info(
                    f"Reference intensity established: {self.reference_intensity:.1f}"
                )
            else:
                self.reference_intensity = self._fallback_reference_intensity(
                    bg_temp, roi_resized
                )
                logger.info(
                    f"Fallback reference intensity: {self.reference_intensity:.1f}"
                )

    def _adaptive_update_cpu(self, gray_f32: np.ndarray, learning_rate: float) -> None:
        """Update adaptive background using the best available CPU backend."""
        if NUMBA_AVAILABLE:
            self.adaptive_background = _update_adaptive_background_numba(
                self.adaptive_background, gray_f32, learning_rate
            )
        else:
            self.adaptive_background = (
                1 - learning_rate
            ) * self.adaptive_background + learning_rate * gray_f32

    def _adaptive_update_gpu(self, gray_f32: np.ndarray, learning_rate: float) -> None:
        """Attempt GPU-accelerated adaptive background update, fall back to CPU."""
        try:
            if self.gpu_type == "cuda":
                with self.gpu_device:
                    gray_gpu = cp.asarray(gray_f32)
                    bg_gpu = cp.asarray(self.adaptive_background)
                    bg_gpu = _update_adaptive_background_gpu(
                        bg_gpu, gray_gpu, learning_rate
                    )
                    self.adaptive_background = cp.asnumpy(bg_gpu)
            elif self.gpu_type == "mps":
                gray_torch = torch.from_numpy(gray_f32).to(self.torch_device)
                bg_torch = torch.from_numpy(self.adaptive_background).to(
                    self.torch_device
                )
                bg_torch = _update_adaptive_background_mps(
                    bg_torch, gray_torch, learning_rate
                )
                self.adaptive_background = bg_torch.cpu().numpy()
            else:
                self._adaptive_update_cpu(gray_f32, learning_rate)
        except Exception as e:
            if not hasattr(self, "_gpu_fallback_warned"):
                logger.warning(
                    f"{self.gpu_type} GPU operation failed, falling back to CPU: {e}"
                )
                self._gpu_fallback_warned = True
            self.use_gpu = False
            self._adaptive_update_cpu(gray_f32, learning_rate)

    def update_and_get_background(
        self,
        gray: np.ndarray,
        roi_mask: Optional[np.ndarray],
        tracking_stabilized: bool,
    ) -> Optional[np.ndarray]:
        """Update the background model and return the active subtraction background."""
        p = self.params
        if self.lightest_background is None:
            self.lightest_background = gray.astype(np.float32)
            self.adaptive_background = gray.astype(np.float32)
            return None  # Indicates first frame

        # Update full-frame background (ROI masking happens during detection)
        self.lightest_background = np.maximum(
            self.lightest_background, gray.astype(np.float32)
        )

        if (
            p.get("ENABLE_ADAPTIVE_BACKGROUND", True)
            and self.adaptive_background is not None
        ):
            learning_rate = p.get("BACKGROUND_LEARNING_RATE", 0.001)
            gray_f32 = gray.astype(np.float32)

            if self.use_gpu:
                self._adaptive_update_gpu(gray_f32, learning_rate)
            else:
                self._adaptive_update_cpu(gray_f32, learning_rate)

        # Only switch when adaptive updating is actually on. Otherwise
        # adaptive_background is frozen at its primed value and switching to
        # it would silently subtract against a stale snapshot.
        adaptive_enabled = p.get("ENABLE_ADAPTIVE_BACKGROUND", True)
        if tracking_stabilized and adaptive_enabled:
            return cv2.convertScaleAbs(self.adaptive_background)
        return cv2.convertScaleAbs(self.lightest_background)

    def _foreground_mask_gpu(
        self, gray: np.ndarray, background: np.ndarray
    ) -> Optional[np.ndarray]:
        """Attempt GPU-accelerated foreground mask; return None on failure."""
        p = self.params
        try:
            if self.gpu_type == "cuda":
                with self.gpu_device:
                    gray_gpu = cp.asarray(gray.astype(np.float32))
                    bg_gpu = cp.asarray(background.astype(np.float32))
                    fg_mask_gpu = _generate_foreground_mask_gpu(gray_gpu, bg_gpu, p)
                    return cp.asnumpy(fg_mask_gpu)
            elif self.gpu_type == "mps":
                gray_torch = torch.from_numpy(gray.astype(np.float32)).to(
                    self.torch_device
                )
                bg_torch = torch.from_numpy(background.astype(np.float32)).to(
                    self.torch_device
                )
                return _generate_foreground_mask_mps(
                    gray_torch, bg_torch, p, self.torch_device
                )
        except Exception as e:
            if not hasattr(self, "_gpu_fallback_warned"):
                logger.warning(
                    f"{self.gpu_type} GPU operation failed, falling back to CPU: {e}"
                )
                if self.gpu_type == "cuda":
                    logger.info(
                        "This may be due to CUDA/CuPy version incompatibility. "
                        "Consider updating CuPy: pip install --upgrade cupy-cuda12x"
                    )
                self._gpu_fallback_warned = True
            self.use_gpu = False
        return None

    def _foreground_mask_cpu(
        self, gray: np.ndarray, background: np.ndarray
    ) -> np.ndarray:
        """Generate foreground mask using CPU (OpenCV)."""
        p = self.params
        dark_on_light = p.get("DARK_ON_LIGHT_BACKGROUND", True)

        if dark_on_light:
            diff = cv2.subtract(background, gray)
        else:
            diff = cv2.subtract(gray, background)

        _, fg_mask = cv2.threshold(diff, p["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)

        ksz = p["MORPH_KERNEL_SIZE"]
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, ker)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, ker)

        if p.get("ENABLE_ADDITIONAL_DILATION", False):
            dil_ksz = p.get("DILATION_KERNEL_SIZE", 3)
            dil_iter = p.get("DILATION_ITERATIONS", 2)
            dil_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dil_ksz, dil_ksz)
            )
            fg_mask = cv2.dilate(fg_mask, dil_kernel, iterations=dil_iter)

        return fg_mask

    def generate_foreground_mask(
        self, gray: np.ndarray, background: np.ndarray
    ) -> np.ndarray:
        """Generates the foreground mask from the gray frame and background.

        Uses GPU acceleration if available for significant speedup on large frames.
        Supports both CUDA (NVIDIA) and MPS (Apple Silicon) GPUs.
        Falls back to CPU if GPU operations fail (e.g., CuPy compilation errors).
        """
        if self.use_gpu:
            result = self._foreground_mask_gpu(gray, background)
            if result is not None:
                return result

        return self._foreground_mask_cpu(gray, background)
