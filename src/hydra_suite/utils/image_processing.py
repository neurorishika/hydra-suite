"""
Utility functions for image processing in the HYDRA Suite.

Optimized with Numba JIT and GPU acceleration (CuPy/PyTorch) where available.
"""

from functools import lru_cache

import cv2
import numpy as np

from hydra_suite.utils.gpu_utils import CUDA_AVAILABLE, NUMBA_AVAILABLE, cp, njit


@lru_cache(maxsize=128)
def _create_gamma_lut(gamma):
    """
    Create cached gamma correction lookup table.

    Uses LRU cache to avoid recreating identical LUTs.
    Rounds gamma to 3 decimals for effective caching.
    """
    gamma_rounded = round(gamma, 3)
    lut = np.empty(256, dtype=np.uint8)
    inv_gamma = 1.0 / gamma_rounded
    for i in range(256):
        lut[i] = np.clip(((i / 255.0) ** inv_gamma) * 255.0, 0, 255)
    return lut


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _compute_percentiles_numba(data, percentiles):
        """Fast percentile computation using sorting (Numba-optimized)."""
        sorted_data = np.sort(data.ravel())
        n = len(sorted_data)
        results = np.empty(len(percentiles), dtype=np.float64)
        for i, p in enumerate(percentiles):
            idx = int((p / 100.0) * (n - 1))
            results[i] = sorted_data[idx]
        return results


def apply_image_adjustments(
    gray: object,
    brightness: object,
    contrast: object,
    gamma: object,
    use_gpu: object = False,
) -> object:
    """
    Apply brightness, contrast, and gamma corrections to grayscale image.

    Optimized with:
    - Cached gamma LUT generation (avoids Python loops)
    - CuPy GPU acceleration when available
    - Numba JIT for CPU path
    - Vectorized operations

    Args:
        gray (np.ndarray): Input grayscale image
        brightness (float): Brightness adjustment (-255 to +255)
        contrast (float): Contrast multiplier (0.0 to 3.0+)
        gamma (float): Gamma correction factor (0.1 to 3.0+)
        use_gpu (bool): Use GPU acceleration if available

    Returns:
        np.ndarray: Adjusted grayscale image

    Note:
        - Brightness: Additive adjustment (linear shift)
        - Contrast: Multiplicative adjustment (scaling)
        - Gamma: Power-law transformation for non-linear luminance correction
    """
    # GPU path (CuPy)
    if use_gpu and CUDA_AVAILABLE:
        gray_gpu = cp.asarray(gray)

        # Round (cp.rint) to match the CPU path's cv2.convertScaleAbs, which
        # rounds rather than truncates. Both round half-to-even (cvRound uses the
        # default FPU mode), so cp.rint reproduces it exactly — floor(x+0.5) would
        # differ at exact half-values. Without this the GPU and CPU paths differ
        # by up to 1 grey level, and since runtime_tier is not part of the bg-sub
        # detection cache key, a GPU-tier and a CPU-tier run could otherwise write
        # divergent detections under the same key.
        adj_gpu = cp.clip(cp.rint(gray_gpu * contrast + brightness), 0, 255).astype(
            cp.uint8
        )

        # Apply gamma correction if needed
        if abs(gamma - 1.0) > 1e-3:
            # Use cached LUT on GPU
            lut = _create_gamma_lut(gamma)
            lut_gpu = cp.asarray(lut)
            adj_gpu = lut_gpu[adj_gpu]

        return cp.asnumpy(adj_gpu)

    # CPU path - use OpenCV for brightness/contrast (well-optimized)
    adj = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

    # Apply gamma correction with cached LUT
    if abs(gamma - 1.0) > 1e-3:
        lut = _create_gamma_lut(gamma)
        adj = cv2.LUT(adj, lut)

    return adj


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _calculate_robust_mean_numba(data, p25, p75, p10, p90):
        """Numba-optimized robust mean calculation."""
        # Try tight percentile range first
        count_tight = 0
        sum_tight = 0.0
        for val in data:
            if p25 <= val <= p75:
                sum_tight += val
                count_tight += 1

        # Use tight range if enough pixels
        if count_tight > len(data) * 0.1:
            return sum_tight / count_tight

        # Fallback to wider range
        count_wide = 0
        sum_wide = 0.0
        for val in data:
            if p10 <= val <= p90:
                sum_wide += val
                count_wide += 1

        if count_wide > 0:
            return sum_wide / count_wide
        return np.mean(data)


def _extract_roi_pixels(frame, roi_mask, use_gpu):
    """Extract pixels of interest from frame, respecting ROI mask and GPU setting."""
    if roi_mask is not None:
        if use_gpu and CUDA_AVAILABLE:
            frame_gpu = cp.asarray(frame)
            roi_mask_gpu = cp.asarray(roi_mask)
            roi_pixels = frame_gpu[roi_mask_gpu > 0]
            if len(roi_pixels) < 100:
                frame_flat = frame_gpu.ravel()
            else:
                frame_flat = roi_pixels
            return cp.asnumpy(frame_flat)
        roi_pixels = frame[roi_mask > 0]
        if len(roi_pixels) < 100:
            return frame.ravel()
        return roi_pixels
    return frame.ravel()


def _compute_robust_mean(frame_flat):
    """Compute robust mean intensity using percentile-based trimming."""
    if NUMBA_AVAILABLE:
        percentiles = _compute_percentiles_numba(
            frame_flat, np.array([10.0, 25.0, 75.0, 90.0])
        )
        p10, p25, p75, p90 = percentiles
        return _calculate_robust_mean_numba(frame_flat, p25, p75, p10, p90)

    p10, p25, p75, p90 = np.percentile(frame_flat, [10, 25, 75, 90])
    mask = (frame_flat >= p25) & (frame_flat <= p75)
    if np.sum(mask) > frame_flat.size * 0.1:
        return np.mean(frame_flat[mask])
    mask = (frame_flat >= p10) & (frame_flat <= p90)
    return np.mean(frame_flat[mask]) if np.sum(mask) > 0 else np.mean(frame_flat)


def _smooth_intensity(current_intensity_history, median_window, alpha, lighting_state):
    """Apply median filtering and exponential smoothing to intensity history."""
    if len(current_intensity_history) >= median_window:
        recent_values = np.array(list(current_intensity_history)[-median_window:])
        median_intensity = np.median(recent_values)
        if "smoothed_value" in lighting_state:
            smoothed = (
                alpha * lighting_state["smoothed_value"]
                + (1 - alpha) * median_intensity
            )
        else:
            smoothed = median_intensity
    else:
        recent_values = np.array(list(current_intensity_history))
        smoothed = np.mean(recent_values)
    lighting_state["smoothed_value"] = smoothed
    return smoothed


def stabilize_lighting(
    frame: object,
    reference_intensity: object,
    current_intensity_history: object,
    alpha: object = 0.95,
    roi_mask: object = None,
    median_window: object = 5,
    lighting_state: object = None,
    use_gpu: object = False,
) -> object:
    """
    Stabilize lighting conditions by normalizing frame intensity to a reference level.

    Optimized with:
    - Numba JIT for percentile and robust mean calculations
    - CuPy GPU acceleration for array operations
    - Efficient vectorized operations
    - Reduced Python overhead

    This function compensates for gradual lighting changes by:
    1. Computing frame's global intensity statistics (within ROI if provided)
    2. Comparing to reference intensity established during background priming
    3. Applying smooth intensity correction to maintain consistent illumination
    4. Using rolling history with median filtering to suppress high-frequency noise

    Args:
        frame (np.ndarray): Input grayscale frame
        reference_intensity (float): Target intensity level from background priming
        current_intensity_history (deque): Rolling history of recent frame intensities
        alpha (float): Smoothing factor for intensity adaptation (0.9-0.99)
        roi_mask (np.ndarray, optional): Binary mask defining region of interest
        median_window (int): Window size for median filtering (3-15)
        lighting_state (dict, optional): Dictionary to store smoothing state
        use_gpu (bool): Use GPU acceleration if available

    Returns:
        tuple: (stabilized_frame, updated_intensity_history, current_mean_intensity)
    """
    if reference_intensity is None:
        return frame, current_intensity_history, np.mean(frame)

    if lighting_state is None:
        lighting_state = {}

    frame_flat = _extract_roi_pixels(frame, roi_mask, use_gpu)
    current_mean = _compute_robust_mean(frame_flat)
    current_intensity_history.append(current_mean)

    smoothed_intensity = _smooth_intensity(
        current_intensity_history, median_window, alpha, lighting_state
    )

    # Calculate and smooth correction factor
    if smoothed_intensity > 0:
        correction_factor = np.clip(reference_intensity / smoothed_intensity, 0.7, 1.4)

        if "last_correction" in lighting_state:
            correction_smooth_alpha = 0.8
            correction_factor = (
                correction_smooth_alpha * lighting_state["last_correction"]
                + (1 - correction_smooth_alpha) * correction_factor
            )
        lighting_state["last_correction"] = correction_factor
    else:
        correction_factor = 1.0

    # Apply lighting correction (GPU or CPU)
    if use_gpu and CUDA_AVAILABLE:
        frame_gpu = cp.asarray(frame)
        # Round (cp.rint) to match the CPU path's cv2.convertScaleAbs (see the
        # brightness/contrast path above for why tier-consistent rounding matters
        # to the bg-sub cache key).
        stabilized_gpu = cp.clip(cp.rint(frame_gpu * correction_factor), 0, 255).astype(
            cp.uint8
        )
        stabilized = cp.asnumpy(stabilized_gpu)
    else:
        stabilized = cv2.convertScaleAbs(frame, alpha=correction_factor, beta=0)

    return stabilized, current_intensity_history, current_mean


def compute_median_color_from_frame(frame: object) -> object:
    """
    Compute the median color (BGR) from a frame.

    Useful for setting background color to match the input video's color profile.

    Args:
        frame: Input frame (BGR, shape: H x W x 3)

    Returns:
        Tuple of (B, G, R) median values
    """
    # Reshape frame to list of pixels
    pixels = frame.reshape(-1, 3)
    # Compute median for each channel
    median_color = tuple(np.median(pixels, axis=0).astype(np.uint8))
    return median_color
