"""Layer 2: fit any image into a model's input tensor, identically everywhere.

This is the only step between a canonical crop and a model, so it pins every
property a second implementation could get wrong: dtype uint8 [0, 255], BGR
channel order, one resampler (INTER_AREA down, INTER_LINEAR up), and zero pad.

The pad value is deliberately NOT the foreign-mask background colour: masking
hides a neighbouring animal inside the crop, padding fills canvas outside the
source image.  Pose already pads zeros at both training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FitResult:
    model_wh: tuple[int, int]
    inner_wh: tuple[int, int]
    offset_xy: tuple[int, int]
    scale: float


def fit_to_model_input(
    source_wh: tuple[int, int],
    model_wh: tuple[int, int],
) -> FitResult:
    """Isotropic centred letterbox parameters. Pure arithmetic."""
    sw, sh = int(source_wh[0]), int(source_wh[1])
    mw, mh = int(model_wh[0]), int(model_wh[1])
    if sw <= 0 or sh <= 0 or mw <= 0 or mh <= 0:
        raise ValueError(f"non-positive dimensions: {source_wh} -> {model_wh}")

    scale = min(mw / sw, mh / sh)
    inner_w = max(1, min(mw, int(round(sw * scale))))
    inner_h = max(1, min(mh, int(round(sh * scale))))
    return FitResult(
        model_wh=(mw, mh),
        inner_wh=(inner_w, inner_h),
        offset_xy=((mw - inner_w) // 2, (mh - inner_h) // 2),
        scale=float(scale),
    )


def apply_fit(image: np.ndarray, fit: FitResult) -> np.ndarray:
    """Resize *image* by ``fit`` and paste it centred on a zero canvas."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"Layer 2 requires uint8 [0,255] input, got {arr.dtype}. "
            "Convert at the producer, not here."
        )
    if arr.ndim == 2:
        arr = arr[:, :, None]
    channels = arr.shape[2]

    interp = cv2.INTER_AREA if fit.scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, fit.inner_wh, interpolation=interp)
    if resized.ndim == 2:
        resized = resized[:, :, None]

    mw, mh = fit.model_wh
    if fit.inner_wh == (mw, mh):
        # Letterbox with zero padding on both axes: the paste below would cover
        # the whole canvas, so the canvas is exactly ``resized``. Skipping the
        # allocation + copy is byte-identical, not an approximation.
        return resized
    canvas = np.zeros((mh, mw, channels), dtype=np.uint8)
    ox, oy = fit.offset_xy
    canvas[oy : oy + fit.inner_wh[1], ox : ox + fit.inner_wh[0]] = resized
    return canvas


def fit_affine(fit: FitResult) -> np.ndarray:
    """2x3 affine mapping source pixels to model-input pixels."""
    return np.array(
        [[fit.scale, 0.0, fit.offset_xy[0]], [0.0, fit.scale, fit.offset_xy[1]]],
        dtype=np.float64,
    )
