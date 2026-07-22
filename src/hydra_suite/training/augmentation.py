"""Decode/resample-robustness augmentations for classifier training.

Colortag/identity classifiers read color as their signal and are trained on ONE
decode+resample convention (cv2 CPU decode, BT.601 limited range, cv2.warpAffine
crops). At inference the GPU-Fast/NVDEC path decodes YCbCr->RGB with a different
matrix/range and warps with torch grid_sample, leaving a small but systematic
shift that tips borderline predictions. Rather than chase bit-exact matching per
codec/GPU/driver, these augmentations train the model to be invariant to the
*space* of decode and resample transforms. See
docs/superpowers/specs/2026-07-22-classkit-decode-robust-augmentation-design.md.

Both functions operate on RGB uint8 (H, W, 3) arrays and take an explicit
``numpy.random.Generator`` so training augmentation is reproducible given a seed.
"""

from __future__ import annotations

import numpy as np

# (Kr, Kb) luma coefficients for the two common matrices.
_BT601 = (0.299, 0.114)
_BT709 = (0.2126, 0.0722)


def _rgb_to_ycbcr(rgb: np.ndarray, kr: float, kb: float, full: bool) -> np.ndarray:
    """RGB float [0,255] -> YCbCr float, given luma coeffs and range."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    kg = 1.0 - kr - kb
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb))
    cr = (r - y) / (2.0 * (1.0 - kr))
    if full:
        return np.stack([y, cb + 128.0, cr + 128.0], axis=-1)
    return np.stack(
        [
            16.0 + y * (219.0 / 255.0),
            128.0 + cb * (224.0 / 255.0),
            128.0 + cr * (224.0 / 255.0),
        ],
        axis=-1,
    )


def _ycbcr_to_rgb(ycc: np.ndarray, kr: float, kb: float, full: bool) -> np.ndarray:
    """YCbCr float -> RGB float [0,255], given luma coeffs and range."""
    Y, Cb, Cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    kg = 1.0 - kr - kb
    if full:
        y = Y
        cb = Cb - 128.0
        cr = Cr - 128.0
    else:
        y = (Y - 16.0) * (255.0 / 219.0)
        cb = (Cb - 128.0) * (255.0 / 224.0)
        cr = (Cr - 128.0) * (255.0 / 224.0)
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.stack([r, g, b], axis=-1)


def simulate_decode_color(
    rgb_u8: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Re-decode the crop through a randomly sampled YCbCr<->RGB convention.

    With probability ``strength``: recover YCbCr using the training origin
    (BT.601 limited), then re-decode with a randomly sampled matrix in
    {BT.601, BT.709} and range in {limited, full}, blended toward the original by
    a random severity, followed by mild brightness/contrast jitter. Reproduces the
    decode variation (matrix + range) the model meets at inference without NVDEC
    data. Identity RGB uint8 in / RGB uint8 out.
    """
    if strength <= 0.0 or rng.random() >= strength:
        return rgb_u8
    rgb = rgb_u8.astype(np.float32)
    ycc = _rgb_to_ycbcr(rgb, *_BT601, full=False)  # recover origin YCbCr
    kr, kb = _BT601 if rng.random() < 0.5 else _BT709
    full = rng.random() < 0.5
    shifted = _ycbcr_to_rgb(ycc, kr, kb, full)
    blend = float(rng.uniform(0.25, 1.0))
    out = rgb + (shifted - rgb) * blend
    # mild photometric jitter
    out *= float(rng.uniform(0.95, 1.05))  # brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * float(rng.uniform(0.95, 1.05)) + mean  # contrast
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def simulate_resample(
    rgb_u8: np.ndarray, prob: float, rng: np.random.Generator
) -> np.ndarray:
    """Re-warp the crop through an alternate resampler with probability ``prob``.

    Randomly chooses cv2.warpAffine (random INTER_LINEAR/CUBIC/AREA + sub-pixel
    translation) or torch grid_sample (bilinear, align_corners=False), simulating
    the cv2.warpAffine-vs-grid_sample geometry gap between training and the GPU
    inference crop path. RGB uint8 in / RGB uint8 out.
    """
    if prob <= 0.0 or rng.random() >= prob:
        return rgb_u8
    import cv2

    h, w = rgb_u8.shape[:2]
    dx = float(rng.uniform(-0.5, 0.5))
    dy = float(rng.uniform(-0.5, 0.5))
    if rng.random() < 0.5:
        interp = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(
            rgb_u8, M, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT
        )
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float()
    theta = torch.tensor(
        [[[1.0, 0.0, 2.0 * dx / w], [0.0, 1.0, 2.0 * dy / h]]], dtype=torch.float32
    )
    grid = F.affine_grid(theta, list(t.shape), align_corners=False)
    warped = F.grid_sample(t, grid, mode="bilinear", align_corners=False)
    return warped[0].permute(1, 2, 0).clamp(0.0, 255.0).round().byte().numpy()


def make_decode_resample_pil_transform(profile, rng: np.random.Generator):
    """Return a callable ``PIL.Image -> PIL.Image`` applying the two augmentations,
    or ``None`` when both are off. For the torchvision Compose pipeline: insert it
    BEFORE ToTensor()/Normalize (train transform only).
    """
    decode = float(getattr(profile, "decode_color_sim", 0.0) or 0.0)
    resample = float(getattr(profile, "resample_sim", 0.0) or 0.0)
    if decode <= 0.0 and resample <= 0.0:
        return None

    from PIL import Image

    def _tf(pil_img):
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        if decode > 0.0:
            arr = simulate_decode_color(arr, decode, rng)
        if resample > 0.0:
            arr = simulate_resample(arr, resample, rng)
        return Image.fromarray(arr)

    return _tf
