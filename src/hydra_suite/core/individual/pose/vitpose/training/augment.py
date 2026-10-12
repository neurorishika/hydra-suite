"""Training augmentation for ViTPose fine-tuning (numpy / OpenCV only).

Applied to the warped model input (BGR uint8), after the geometric warp, so
they never move keypoints.  The recipe follows mmpose's top-down ViTPose /
RTMPose pipelines: photometric distortion, occasional blur, and one random
occluding rectangle (CoarseDropout), which teaches the model to place a
keypoint hidden behind a neighbour or debris.

Every function takes an explicit ``np.random.Generator`` so data-loader
workers can be seeded independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentParams:
    rotation: float = 40.0
    rotation_prob: float = 0.6
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.0
    hue: float = 0.0
    blur_prob: float = 0.0
    noise_prob: float = 0.0
    occlusion_prob: float = 0.0


def sample_rotation(params: AugmentParams, rng: np.random.Generator) -> float:
    """Rotation in degrees: N(0, r/2) clipped to +/-r, on ``rotation_prob``."""
    if params.rotation <= 0 or rng.random() >= params.rotation_prob:
        return 0.0
    r = float(params.rotation)
    return float(np.clip(rng.normal(0.0, r / 2.0), -r, r))


def photometric(
    img_bgr: np.ndarray, params: AugmentParams, rng: np.random.Generator
) -> np.ndarray:
    """Brightness, contrast, saturation and hue jitter."""
    out = img_bgr.astype(np.float32)
    if params.brightness > 0:
        out *= rng.uniform(1.0 - params.brightness, 1.0 + params.brightness)
    if params.contrast > 0:
        mean = out.mean(axis=(0, 1), keepdims=True)
        factor = rng.uniform(1.0 - params.contrast, 1.0 + params.contrast)
        out = (out - mean) * factor + mean
    out = np.clip(out, 0, 255)
    if params.saturation > 0 or params.hue > 0:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        if params.hue > 0:
            # OpenCV hue is 0..179 (2 degrees per unit)
            shift = rng.uniform(-params.hue, params.hue) * 180.0
            hsv[..., 0] = np.mod(hsv[..., 0] + shift, 180.0)
        if params.saturation > 0:
            hsv[..., 1] *= rng.uniform(1.0 - params.saturation, 1.0 + params.saturation)
        hsv = np.clip(hsv, 0, [179, 255, 255])
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out.astype(np.uint8)


def blur(
    img_bgr: np.ndarray, params: AugmentParams, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian or median blur (kernel 3 or 5) on ``blur_prob`` of samples."""
    if params.blur_prob <= 0 or rng.random() >= params.blur_prob:
        return img_bgr
    k = int(rng.choice([3, 5]))
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img_bgr, (k, k), 0)
    return cv2.medianBlur(img_bgr, k)


def noise(
    img_bgr: np.ndarray, params: AugmentParams, rng: np.random.Generator
) -> np.ndarray:
    """Gaussian pixel noise (sigma 2-10 grey levels) on ``noise_prob``."""
    if params.noise_prob <= 0 or rng.random() >= params.noise_prob:
        return img_bgr
    sigma = rng.uniform(2.0, 10.0)
    noisy = img_bgr.astype(np.float32) + rng.normal(0.0, sigma, img_bgr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def occlude(
    img_bgr: np.ndarray, params: AugmentParams, rng: np.random.Generator
) -> np.ndarray:
    """One rectangle, 10-40% of each side, filled with random grey, on
    ``occlusion_prob`` of samples.  Targets are unchanged: the model learns
    where a hidden keypoint is from the rest of the animal."""
    if params.occlusion_prob <= 0 or rng.random() >= params.occlusion_prob:
        return img_bgr
    h, w = img_bgr.shape[:2]
    rh = max(1, int(round(h * rng.uniform(0.1, 0.4))))
    rw = max(1, int(round(w * rng.uniform(0.1, 0.4))))
    y = int(rng.integers(0, max(1, h - rh + 1)))
    x = int(rng.integers(0, max(1, w - rw + 1)))
    out = img_bgr.copy()
    out[y : y + rh, x : x + rw] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return out


def augment_pixels(
    img_bgr: np.ndarray, params: AugmentParams, rng: np.random.Generator
) -> np.ndarray:
    """All pixel-level augmentation, in the order mmpose applies it."""
    out = photometric(img_bgr, params, rng)
    out = blur(out, params, rng)
    out = noise(out, params, rng)
    return occlude(out, params, rng)
