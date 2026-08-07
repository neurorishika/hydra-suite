"""Heatmap decoding, UDP/DARK.

Two implementations:
  decode_udp_cv2   -- faithful port of upstream post_dark_udp. The ORACLE.
                      float64 numpy on CPU. Not the production path.
  decode_udp_torch -- device-resident, float32. Production (Task 8).

They are bound by a parity test. cv2 anchors us to mmpose; the parity test
anchors torch to cv2; Gate C validates the whole chain.

On the blur sigma: cv2.GaussianBlur(hm, (11, 11), 0) means "derive sigma from
kernel" -> 0.3*((11-1)*0.5 - 1) + 0.8 == 2.0, exactly the training sigma.
HuggingFace instead hardcodes sigma=0.8, which does not track kernel size. That
is an unflagged deviation and we deliberately do not follow it.

`get_max_preds` and `decode_udp_cv2` are transcribed from upstream
mmpose/core/evaluation/top_down_eval.py (ViTPose fork):
  - `_get_max_preds`  -> lines 63-95
  - `post_dark_udp`   -> lines 335-396

The one deliberate deviation from upstream: upstream mutates its input
in place (`cv2.GaussianBlur(..., heatmap)`, `np.clip(..., batch_heatmaps)`,
`np.log(..., batch_heatmaps)`). Our `decode_udp_cv2` copies first so callers
keep their original heatmaps. The math is otherwise identical.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_max_preds(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get keypoint predictions from score maps.

    Transcribed from upstream `_get_max_preds`
    (mmpose/core/evaluation/top_down_eval.py, lines 63-95).

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _post_dark_udp(
    coords: np.ndarray, batch_heatmaps: np.ndarray, kernel: int = 3
) -> np.ndarray:
    """DARK post-processing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Transcribed from upstream `post_dark_udp`
    (mmpose/core/evaluation/top_down_eval.py, lines 335-396).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge"
    ).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def decode_udp_cv2(
    heatmaps: np.ndarray, kernel: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """Decode keypoint coordinates from heatmaps via UDP/DARK refinement.

    Faithful port of upstream `post_dark_udp`, wired the way the upstream
    top-down caller wires `_get_max_preds` + `post_dark_udp`
    (mmpose/core/evaluation/top_down_eval.py, lines 567-570):
        preds, maxvals = _get_max_preds(heatmaps)
        preds = post_dark_udp(preds, heatmaps, kernel=kernel)

    Must NOT mutate `heatmaps` — upstream blurs/clips/logs in place, so we
    copy first (the one deliberate deviation from upstream; math unchanged).

    Shape: (N, K, H, W) -> coords (N, K, 2), maxvals (N, K, 1).
    """
    heatmaps = heatmaps.astype(np.float64).copy()
    coords, maxvals = get_max_preds(heatmaps)
    coords = _post_dark_udp(coords, heatmaps, kernel=kernel)
    return coords, maxvals


def flip_back(
    heatmaps: np.ndarray, flip_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Mirror heatmaps and swap left/right keypoint channels.

    With UDP, do NOT additionally apply the shift_heatmap column shift
    (`hm[:, :, :, 1:] = hm[:, :, :, :-1]`). That is the non-UDP correction;
    applying both double-corrects.
    """
    out = heatmaps[..., ::-1].copy()
    for a, b in flip_pairs:
        tmp = out[:, a].copy()
        out[:, a] = out[:, b]
        out[:, b] = tmp
    return out


def _gaussian_kernel1d(kernel: int, device, dtype) -> torch.Tensor:
    """Match cv2.GaussianBlur(..., 0): sigma derived from kernel size, and the
    same normalised kernel cv2 builds."""
    sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8
    x = torch.arange(kernel, device=device, dtype=dtype) - (kernel - 1) / 2
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def decode_udp_torch(
    heatmaps: torch.Tensor, kernel: int = 11
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-resident UDP/DARK decode: a faithful torch translation of
    upstream `_get_max_preds` + `post_dark_udp`.

    Verified against /tmp/vitpose-ref/top_down_eval.py lines 335-396 (Task 7
    Step 1). An earlier draft of this function was written from memory and
    was wrong in four independent ways (it normalised the blur by
    orig_max/blur_max -- which belongs to the NON-UDP `_gaussian_blur`, not to
    post_dark_udp; used a 2-step 0.25-scaled stencil instead of upstream's
    1-step; guarded the Hessian on |det| instead of adding eps to the
    diagonal; and padded 'reflect' instead of 'edge'). Any of those silently
    shifts sub-pixel coordinates.

    Upstream reference, verbatim:
        cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)   # in-place
        np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
        np.log(batch_heatmaps, batch_heatmaps)
        batch_heatmaps_pad = np.pad(..., ((0,0),(0,0),(1,1),(1,1)), mode='edge')
        index  = coords[...,0] + 1 + (coords[...,1] + 1) * (W + 2)
        dx  = 0.5 * (ix1 - ix1_)
        dy  = 0.5 * (iy1 - iy1_)
        dxx = ix1 - 2*i_ + ix1_
        dyy = iy1 - 2*i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()

    The 2x2 inverse is closed-form rather than torch.linalg.solve: MPS has
    linalg gaps, and a 2x2 inverse is exact anyway.
    """
    n, k, h, w = heatmaps.shape
    device, dtype = heatmaps.device, heatmaps.dtype

    # --- _get_max_preds: integer argmax, masked to -1 where maxval <= 0
    flat = heatmaps.reshape(n * k, -1)
    maxvals, idx = flat.max(dim=1, keepdim=True)
    px = (idx % w).to(dtype)
    py = torch.div(idx, w, rounding_mode="floor").to(dtype)
    coords = torch.cat([px, py], dim=1)
    coords = torch.where(maxvals > 0.0, coords, torch.full_like(coords, -1.0))

    # --- blur: separable, matching cv2.GaussianBlur(..., 0) with its default
    #     BORDER_REFLECT_101 (torch's "reflect" is the same convention).
    #     NOTE: no orig_max/blur_max renormalisation -- upstream post_dark_udp
    #     does not do it.
    k1 = _gaussian_kernel1d(kernel, device, dtype)
    pad = kernel // 2
    x = heatmaps.reshape(n * k, 1, h, w)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k1.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, k1.view(1, 1, -1, 1))
    blurred = x.clamp(0.001, 50.0).log()  # clip THEN log, as upstream

    # --- pad by 1 with 'edge' (== replicate) and index at coords+1
    # NOTE on the coords == -1 (maxval <= 0) masked path: bx/by below become 0,
    # so torch reads the padded top-left corner -- not numpy's flat-index
    # wraparound. This is inert: maxval <= 0 means the whole channel is <= 0
    # (the blur kernel is non-negative and normalized, so it can't create a
    # positive), and the clamp's strictly-positive lower bound (0.001, below)
    # then floors that whole channel to a uniform log(0.001), everywhere
    # including the padded border. All seven samples are equal, so
    # dx=dy=dxx=dyy=dxy=0, the Hessian is exactly eps*I, and the offset is 0 --
    # which pixel bx/by point at no longer matters. If the clamp's lower bound
    # is ever made <= 0, this argument breaks.
    hm = F.pad(blurred, (1, 1, 1, 1), mode="replicate").reshape(n * k, h + 2, w + 2)
    bx = coords[:, 0].long() + 1
    by = coords[:, 1].long() + 1
    b = torch.arange(n * k, device=device)

    i_ = hm[b, by, bx]
    ix1 = hm[b, by, bx + 1]
    ix1_ = hm[b, by, bx - 1]
    iy1 = hm[b, by + 1, bx]
    iy1_ = hm[b, by - 1, bx]
    ix1y1 = hm[b, by + 1, bx + 1]
    ix1_y1_ = hm[b, by - 1, bx - 1]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    dxx = ix1 - 2.0 * i_ + ix1_
    dyy = iy1 - 2.0 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)

    # --- hessian + eps*I, closed-form 2x2 inverse, coords -= inv @ derivative
    eps = torch.finfo(torch.float32).eps
    a = dxx + eps
    d = dyy + eps
    bb = dxy
    det = a * d - bb * bb
    ox = (d * dx - bb * dy) / det
    oy = (a * dy - bb * dx) / det

    refined = coords.clone()
    refined[:, 0] = coords[:, 0] - ox
    refined[:, 1] = coords[:, 1] - oy

    return refined.reshape(n, k, 2), maxvals.reshape(n, k, 1)
