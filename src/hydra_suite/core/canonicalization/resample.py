"""The one torch resampler seam for the canonical-crop pipeline.

``canonical_warp``/``canonical_warp_batch`` replace ``cv2.warpAffine`` with
``F.grid_sample`` for frames already resident on a torch device (CPU/CUDA/MPS).
The normalised ``theta`` derivation is ported verbatim from
``core/canonicalization/crop.py``'s ``gpu_canonical_crop``/``gpu_canonical_crop_batch``
-- it was numerically verified to land features at the analytic canonical
location with zero sub-pixel shift, so it must not be re-derived here.

``letterbox_fit`` is the torch counterpart of ``core/canonicalization/fit.py``'s
``apply_fit`` (Layer 2): it reuses ``fit_to_model_input`` for the arithmetic and
resamples with antialiased bilinear interpolation instead of ``cv2.resize``.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hydra_suite.core.canonicalization.fit import fit_to_model_input
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry


def _theta_from_m_align(
    m_align: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    w_in: int,
    h_in: int,
) -> np.ndarray:
    """Verbatim port of the theta derivation in ``crop.gpu_canonical_crop``.

    Builds the normalised ``(2, 3)`` theta for ``F.affine_grid`` with
    ``align_corners=True`` from the forward ``m_align`` (frame -> canvas)
    affine, by inverting it to the canvas -> frame mapping grid_sample needs.
    """
    m_inv = cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))

    sw = float(canvas_w - 1)
    sh = float(canvas_h - 1)
    inv_win = 1.0 / max(float(w_in - 1), 1.0)
    inv_hin = 1.0 / max(float(h_in - 1), 1.0)

    t00 = m_inv[0, 0] * sw * inv_win
    t01 = m_inv[0, 1] * sh * inv_win
    t10 = m_inv[1, 0] * sw * inv_hin
    t11 = m_inv[1, 1] * sh * inv_hin

    theta = np.array(
        [
            [t00, t01, t00 + t01 + 2.0 * m_inv[0, 2] * inv_win - 1.0],
            [t10, t11, t10 + t11 + 2.0 * m_inv[1, 2] * inv_hin - 1.0],
        ],
        dtype=np.float32,
    )
    return theta


def canonical_warp(
    frame_chw: torch.Tensor,
    m_align: np.ndarray,
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    """Warp ``frame_chw`` into the canonical canvas via ``F.grid_sample``.

    ``m_align`` is the 2x3 forward affine from ``canonical_affine`` mapping
    frame pixel coords to canvas pixel coords. Returns
    ``(C, canvas_h, canvas_w)`` on the same device as ``frame_chw``.
    """
    canvas_w, canvas_h = geometry.canvas_w, geometry.canvas_h
    c, h_in, w_in = frame_chw.shape

    theta = _theta_from_m_align(m_align, canvas_w, canvas_h, w_in, h_in)
    theta_t = torch.as_tensor(
        theta, dtype=torch.float32, device=frame_chw.device
    ).unsqueeze(0)

    with torch.inference_mode():
        grid = F.affine_grid(theta_t, (1, c, canvas_h, canvas_w), align_corners=True)
        crop = F.grid_sample(
            frame_chw.unsqueeze(0).float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return crop.squeeze(0)


def canonical_warp_batch(
    frame_chw: torch.Tensor,
    m_aligns: List[np.ndarray],
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    """Batch version of :func:`canonical_warp` for N crops from one frame.

    Returns ``(N, C, canvas_h, canvas_w)`` on the same device as
    ``frame_chw``.
    """
    canvas_w, canvas_h = geometry.canvas_w, geometry.canvas_h
    c, h_in, w_in = frame_chw.shape
    n = len(m_aligns)

    if n == 0:
        return torch.zeros(
            0, c, canvas_h, canvas_w, dtype=frame_chw.dtype, device=frame_chw.device
        )
    if n == 1:
        return canonical_warp(frame_chw, m_aligns[0], geometry).unsqueeze(0)

    thetas_np = np.empty((n, 2, 3), dtype=np.float32)
    for i, m_align in enumerate(m_aligns):
        thetas_np[i] = _theta_from_m_align(m_align, canvas_w, canvas_h, w_in, h_in)

    thetas_t = torch.as_tensor(thetas_np, dtype=torch.float32, device=frame_chw.device)

    with torch.inference_mode():
        grid = F.affine_grid(thetas_t, (n, c, canvas_h, canvas_w), align_corners=True)
        frame_expanded = frame_chw.unsqueeze(0).expand(n, -1, -1, -1).float()
        crops = F.grid_sample(
            frame_expanded.contiguous(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return crops


def letterbox_fit(
    crop_chw: torch.Tensor,
    model_wh: tuple,
) -> torch.Tensor:
    """Antialiased bilinear letterbox-fit into a model input canvas.

    Torch counterpart of ``fit.apply_fit``: reuses ``fit_to_model_input`` for
    the isotropic centred-letterbox arithmetic, then resamples with
    antialiased bilinear interpolation and pastes onto a zero canvas.

    Accepts ``(C, H, W)`` (returns ``(C, model_h, model_w)``) or a batched
    ``(N, C, H, W)`` input (returns ``(N, C, model_h, model_w)``).
    """
    single = crop_chw.dim() == 3
    x = crop_chw.unsqueeze(0) if single else crop_chw
    n, c, sh, sw = x.shape
    fit = fit_to_model_input((sw, sh), model_wh)
    iw, ih = fit.inner_wh
    mw, mh = fit.model_wh
    with torch.inference_mode():
        resized = F.interpolate(
            x, size=(ih, iw), mode="bilinear", align_corners=False, antialias=True
        )
        if (ih, iw) == (mh, mw):
            out = resized
        else:
            out = x.new_zeros((n, c, mh, mw))
            ox, oy = fit.offset_xy
            out[:, :, oy : oy + ih, ox : ox + iw] = resized
    return out.squeeze(0) if single else out
