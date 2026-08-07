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

import numpy as np
import torch


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
    """Resize *image* by ``fit`` and paste it centred on a zero canvas.

    Delegates to the torch seam (:func:`~hydra_suite.core.canonicalization.
    resample.letterbox_fit`): converts the HWC uint8 input to a CHW float
    tensor once, resamples with antialiased bilinear interpolation, then
    converts back to HWC uint8. Import of the seam is local to avoid a
    circular import (``resample`` imports ``fit_to_model_input`` from this
    module).
    """
    from hydra_suite.core.canonicalization.resample import letterbox_fit

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"Layer 2 requires uint8 [0,255] input, got {arr.dtype}. "
            "Convert at the producer, not here."
        )
    if arr.ndim == 2:
        arr = arr[:, :, None]
    channels = arr.shape[2]

    chw = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
    out_chw = letterbox_fit(chw, fit.model_wh)
    canvas = (
        out_chw.round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    if channels == 1 and canvas.ndim == 2:
        canvas = canvas[:, :, None]
    return canvas


def fit_affine(fit: FitResult) -> np.ndarray:
    """2x3 affine mapping source pixels to model-input pixels."""
    return np.array(
        [[fit.scale, 0.0, fit.offset_xy[0]], [0.0, fit.scale, fit.offset_xy[1]]],
        dtype=np.float64,
    )
