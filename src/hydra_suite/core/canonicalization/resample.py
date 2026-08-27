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
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span


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


def _theta_for_subregion(
    m_align: np.ndarray,
    x0: int,
    y0: int,
    canvas_wh: tuple,
    pad_wh: tuple,
) -> np.ndarray:
    """``_theta_from_m_align`` for a sub-region input.

    Input pixel ``(u, v)`` of the (padded) sub-region corresponds to frame
    pixel ``(u + x0, v + y0)``, so the canvas->frame map ``m_inv`` has its
    translation shifted by ``-(x0, y0)`` and is normalised by the padded
    sub-region size ``pad_wh`` instead of the full frame. Equals
    ``_theta_from_m_align`` when ``x0 == y0 == 0`` and ``pad_wh == (w_in, h_in)``.
    """
    canvas_w, canvas_h = int(canvas_wh[0]), int(canvas_wh[1])
    pad_w, pad_h = int(pad_wh[0]), int(pad_wh[1])
    m_inv = cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
    tx = m_inv[0, 2] - float(x0)
    ty = m_inv[1, 2] - float(y0)

    sw = float(canvas_w - 1)
    sh = float(canvas_h - 1)
    inv_win = 1.0 / max(float(pad_w - 1), 1.0)
    inv_hin = 1.0 / max(float(pad_h - 1), 1.0)

    t00 = m_inv[0, 0] * sw * inv_win
    t01 = m_inv[0, 1] * sh * inv_win
    t10 = m_inv[1, 0] * sw * inv_hin
    t11 = m_inv[1, 1] * sh * inv_hin
    return np.array(
        [
            [t00, t01, t00 + t01 + 2.0 * tx * inv_win - 1.0],
            [t10, t11, t10 + t11 + 2.0 * ty * inv_hin - 1.0],
        ],
        dtype=np.float32,
    )


def _canvas_footprint_aabb(
    m_align: np.ndarray,
    geometry: CanonicalGeometry,
    frame_hw: tuple,
) -> tuple:
    """Clamped, +1px-padded frame-space AABB of the canonical canvas footprint.

    Maps the four canvas corners back through ``m_inv`` (canvas -> frame) and
    bounds them. The +1px pad guarantees bilinear neighbours of any in-frame
    sampled coordinate are inside the crop; clamping to the frame makes
    out-of-frame samples fall outside the sub-region (grid_sample zeros),
    exactly matching the full-frame ``padding_mode="zeros"`` behaviour.
    """
    h_in, w_in = int(frame_hw[0]), int(frame_hw[1])
    cw, ch = geometry.canvas_w, geometry.canvas_h
    m_inv = cv2.invertAffineTransform(np.asarray(m_align, dtype=np.float64))
    xs = np.array([0.0, cw - 1.0, 0.0, cw - 1.0])
    ys = np.array([0.0, 0.0, ch - 1.0, ch - 1.0])
    fx = m_inv[0, 0] * xs + m_inv[0, 1] * ys + m_inv[0, 2]
    fy = m_inv[1, 0] * xs + m_inv[1, 1] * ys + m_inv[1, 2]
    pad = 1
    x0 = max(0, int(np.floor(fx.min())) - pad)
    y0 = max(0, int(np.floor(fy.min())) - pad)
    x1 = min(w_in, int(np.ceil(fx.max())) + pad)
    y1 = min(h_in, int(np.ceil(fy.max())) + pad)
    return x0, y0, x1, y1


def _frame_view(frame):
    """Squeeze a raw frame to a sliceable 3-D view and report its layout.

    Returns ``(view, h_in, w_in, layout)`` where ``layout`` is ``"hwc"`` or
    ``"chw"``. Mirrors the layout rules of the crop layer's
    ``_frame_to_chw_float``: numpy frames are always ``(H, W, 3)``; torch
    frames may be ``(3, H, W)``, ``(H, W, 3)`` (NVDEC), or carry a leading
    batch axis of 1.
    """
    if isinstance(frame, np.ndarray):
        return frame, int(frame.shape[0]), int(frame.shape[1]), "hwc"
    view = frame
    if view.ndim == 4:
        view = view.squeeze(0)
    if view.ndim == 3 and view.shape[-1] == 3 and view.shape[0] != 3:
        return view, int(view.shape[0]), int(view.shape[1]), "hwc"
    return view, int(view.shape[1]), int(view.shape[2]), "chw"


def _slice_frame_view(view, layout: str, x0: int, y0: int, x1: int, y1: int):
    """Spatial sub-region of a frame view, preserving its layout."""
    if layout == "hwc":
        return view[y0:y1, x0:x1]
    return view[:, y0:y1, x0:x1]


def canonical_warp_batch_from_frame(
    frame,
    m_aligns: List[np.ndarray],
    geometry: CanonicalGeometry,
    to_chw_float,
) -> torch.Tensor:
    """``canonical_warp_batch`` that never materialises the whole frame.

    Identical output to ``canonical_warp_batch(to_chw_float(frame), ...)``,
    but the per-detection AABB pre-crop happens on the RAW frame and only
    those sub-regions are handed to ``to_chw_float``. ``to_chw_float`` is an
    elementwise conversion (layout change + dtype cast + optional scale), and
    elementwise conversion commutes exactly with slicing, so the pasted values
    are bit-for-bit what the full-frame conversion would have produced -- while
    the conversion cost drops from O(frame area) to O(sum of crop footprints).

    This matters because the conversion is per *consumer*: head-tail, each CNN
    phase and pose each call a per-frame crop extractor, so a 4512x4512 frame
    was converted to a 244 MB float32 tensor three times per frame to sample a
    few dozen small crops.
    """
    view, h_in, w_in, layout = _frame_view(frame)
    canvas_w, canvas_h = geometry.canvas_w, geometry.canvas_h
    n = len(m_aligns)

    if n == 0:
        probe = to_chw_float(
            _slice_frame_view(view, layout, 0, 0, min(1, w_in), min(1, h_in))
        )
        return torch.zeros(
            0,
            probe.shape[0],
            canvas_h,
            canvas_w,
            dtype=probe.dtype,
            device=probe.device,
        )

    boxes = [_canvas_footprint_aabb(m, geometry, (h_in, w_in)) for m in m_aligns]
    subs: List[torch.Tensor | None] = []
    # Separated from WARP_BATCH because this cost is O(sum of crop footprints)
    # and the warp scales with n — the signature that identifies an O(frame
    # area) conversion regression.
    with span(N.FRAME_TO_CHW, units=len(boxes)):
        for x0, y0, x1, y1 in boxes:
            if x1 > x0 and y1 > y0:
                subs.append(
                    to_chw_float(_slice_frame_view(view, layout, x0, y0, x1, y1))
                )
            else:
                subs.append(None)

    ref = next((s for s in subs if s is not None), None)
    if ref is None:
        ref = to_chw_float(
            _slice_frame_view(view, layout, 0, 0, min(1, w_in), min(1, h_in))
        )
    c = int(ref.shape[0])
    pad_h = max(1, max((int(s.shape[1]) for s in subs if s is not None), default=1))
    pad_w = max(1, max((int(s.shape[2]) for s in subs if s is not None), default=1))

    batch = ref.new_zeros((n, c, pad_h, pad_w))
    thetas_np = np.empty((n, 2, 3), dtype=np.float32)
    for i, (x0, y0, _x1, _y1) in enumerate(boxes):
        sub = subs[i]
        if sub is not None:
            batch[i, :, : sub.shape[1], : sub.shape[2]] = sub
        thetas_np[i] = _theta_for_subregion(
            m_aligns[i], x0, y0, (canvas_w, canvas_h), (pad_w, pad_h)
        )

    thetas_t = torch.as_tensor(thetas_np, dtype=torch.float32, device=batch.device)
    with torch.inference_mode():
        grid = F.affine_grid(thetas_t, (n, c, canvas_h, canvas_w), align_corners=True)
        return F.grid_sample(
            batch.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def canonical_warp(
    frame_chw: torch.Tensor,
    m_align: np.ndarray,
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    """Single-crop canonical warp (see :func:`canonical_warp_batch`)."""
    return canonical_warp_batch(frame_chw, [m_align], geometry).squeeze(0)


def canonical_warp_batch(
    frame_chw: torch.Tensor,
    m_aligns: List[np.ndarray],
    geometry: CanonicalGeometry,
) -> torch.Tensor:
    """Batch canonical warp via per-detection AABB pre-crop + one grid_sample.

    Numerically equivalent to the previous full-frame ``expand(N).contiguous()``
    path within the float32 grid-normalization noise floor (~1e-4), NOT bitwise
    ``torch.equal`` -- see
    ``docs/superpowers/specs/2026-08-17-crop-warp-aabb-precrop-design.md``
    (Acceptance bar). Samples only each detection's canvas footprint (a small
    frame region) instead of replicating the whole frame N times.
    """
    canvas_w, canvas_h = geometry.canvas_w, geometry.canvas_h
    c, h_in, w_in = frame_chw.shape
    n = len(m_aligns)
    if n == 0:
        return torch.zeros(
            0, c, canvas_h, canvas_w, dtype=frame_chw.dtype, device=frame_chw.device
        )

    boxes = [_canvas_footprint_aabb(m, geometry, (h_in, w_in)) for m in m_aligns]
    sub_w = [max(0, x1 - x0) for (x0, _y0, x1, _y1) in boxes]
    sub_h = [max(0, y1 - y0) for (_x0, y0, _x1, y1) in boxes]
    pad_w = max(1, max(sub_w))
    pad_h = max(1, max(sub_h))

    batch = frame_chw.new_zeros((n, c, pad_h, pad_w))
    thetas_np = np.empty((n, 2, 3), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        sw_i, sh_i = sub_w[i], sub_h[i]
        if sw_i > 0 and sh_i > 0:
            batch[i, :, :sh_i, :sw_i] = frame_chw[:, y0:y1, x0:x1]
        thetas_np[i] = _theta_for_subregion(
            m_aligns[i], x0, y0, (canvas_w, canvas_h), (pad_w, pad_h)
        )

    thetas_t = torch.as_tensor(thetas_np, dtype=torch.float32, device=frame_chw.device)
    with torch.inference_mode():
        grid = F.affine_grid(thetas_t, (n, c, canvas_h, canvas_w), align_corners=True)
        crops = F.grid_sample(
            batch.float(),
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


def squash_fit(crop_chw: torch.Tensor, model_wh: tuple) -> torch.Tensor:
    """Anisotropic antialiased-bilinear resize straight to ``model_wh`` (no paste).

    Layer 2 for artifacts trained with torchvision ``Resize((sz, sz))`` on PIL
    images (every classifier published before 2026-08-05). PIL's Resize is
    antialiased; ``F.interpolate(antialias=True)`` is the closest torch match.
    """
    single = crop_chw.dim() == 3
    x = crop_chw.unsqueeze(0) if single else crop_chw
    mw, mh = int(model_wh[0]), int(model_wh[1])
    with torch.inference_mode():
        out = F.interpolate(
            x, size=(mh, mw), mode="bilinear", align_corners=False, antialias=True
        )
    return out.squeeze(0) if single else out


def fit_batch_for_model(
    crops_chw: torch.Tensor, model_wh: tuple, policy: str
) -> torch.Tensor:
    """Dispatch Layer 2 by ``policy`` (see ``ClassifierMetadata.fit_policy``)."""
    if policy == "letterbox":
        return letterbox_fit(crops_chw, model_wh)
    if policy == "squash":
        return squash_fit(crops_chw, model_wh)
    raise ValueError(f"unsupported fit_policy for tensor Layer 2: {policy!r}")
