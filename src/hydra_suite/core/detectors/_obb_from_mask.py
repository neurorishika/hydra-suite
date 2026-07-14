"""GPU-native, cv2-free rotated-rectangle extraction from segmentation masks.

Used to treat a YOLO instance-segmentation checkpoint as an OBB source: given
a batch of binary/soft masks and their (matching-space) bounding boxes, find
each detection's minimum-area rotated rectangle without ever calling
``cv2.findContours``/``cv2.minAreaRect`` or leaving the accelerator.

Method
------
1. Crop each detection's mask to an ISOTROPIC (square, non-rotated) tile via
   a single batched ``torchvision.ops.roi_align`` call -- isotropic scale is
   required so that an angle measured in the crop's local coordinate frame
   equals the angle in the source mask's coordinate frame (a non-square
   resample would shear angles).
2. Compute the mask-weighted centroid of the crop.
3. Project the crop's foreground pixels onto ``num_angles`` candidate axes in
   one batched matmul, take the masked (min, max) extent per axis to get a
   width/height/area per candidate angle, and pick the angle minimizing area
   (a coarse, batched analogue of the rotating-calipers objective that
   ``cv2.minAreaRect`` solves exactly for a polygon hull).
4. Refine the winning angle to sub-grid accuracy with a closed-form 3-point
   parabolic fit through the winning bin and its two neighbours (same idea as
   sub-pixel peak refinement in stereo/optical-flow disparity search), then
   recompute width/height once more at the refined angle.
5. Canonicalize w/h so that the wider dimension is always ``w`` and ``angle``
   always tracks the major axis (swap w/h and adjust angle by π/2 if needed).

Everything after step 1 operates on a small, fixed-size tensor
(``crop_size x crop_size`` per detection), so this stays cheap even for many
detections per frame and never performs a per-detection Python loop.
"""

from __future__ import annotations

import math

import torch
from torchvision.ops import roi_align


def _letterbox_gain_pad(
    mask_shape: tuple[int, int], orig_shape: tuple[int, int]
) -> tuple[float, float, float]:
    """Return ``(gain, pad_x, pad_y)`` mapping ``orig_shape`` -> ``mask_shape``.

    Mirrors ``ultralytics.utils.ops.scale_boxes``'s own formula exactly: a
    single uniform ``gain`` (``mask`` canvases are always square, matching a
    square YOLO letterboxed input) plus a symmetric pad per axis. Using a
    single scalar gain (rather than independent per-axis ratios) is what
    keeps rotation angles correct when the original frame is not square.

    To go orig -> mask space: ``x_mask = x_orig * gain + pad_x`` (same for y).
    To go mask -> orig space: ``x_orig = (x_mask - pad_x) / gain``.
    """
    mh, mw = mask_shape
    oh, ow = orig_shape
    gain = min(mh / oh, mw / ow)
    pad_x = (mw - round(ow * gain)) / 2.0
    pad_y = (mh - round(oh * gain)) / 2.0
    return float(gain), float(pad_x), float(pad_y)


def rotated_rect_from_masks(
    masks: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    num_angles: int = 24,
    crop_size: int = 64,
    pad_ratio: float = 0.15,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """Find each detection's minimum-area rotated rectangle from its mask.

    Parameters
    ----------
    masks:
        ``(N, H, W)`` tensor (bool/float/uint8), same coordinate space as
        ``boxes_xyxy``. May be on CPU or CUDA.
    boxes_xyxy:
        ``(N, 4)`` axis-aligned boxes in the SAME coordinate space as
        ``masks``, used only to size/center the isotropic crop.
    num_angles:
        Number of coarse candidate angles searched over ``[0, pi)`` before
        parabolic refinement.
    crop_size:
        Side length (pixels) of the isotropic square tile each detection's
        mask is resampled into.
    pad_ratio:
        Fractional padding added around the (square-ified) box before
        cropping, so the crop is not clipped exactly at the mask edge.

    Returns
    -------
    torch.Tensor
        ``(N, 5)``: ``(cx, cy, w, h, angle_rad)``, same coordinate space as
        the inputs, same device/dtype family (float32). A detection whose
        mask has no foreground pixels inside its crop yields an all-``NaN``
        row -- callers should drop these via the existing finite-value guard
        (``_valid_detection_mask`` in ``stages/obb.py`` already does this).
    """
    device = masks.device
    n = masks.shape[0]
    if n == 0:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)

    # --- 1. Isotropic square crop via a single batched roi_align call. ---
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    bw = (x2 - x1).clamp(min=1.0)
    bh = (y2 - y1).clamp(min=1.0)
    bcx = (x1 + x2) / 2.0
    bcy = (y1 + y2) / 2.0
    half = torch.maximum(bw, bh) / 2.0 * (1.0 + pad_ratio)
    side = 2.0 * half  # physical size (pixels) of each detection's crop
    sx1, sy1 = bcx - half, bcy - half
    sx2, sy2 = bcx + half, bcy + half
    # Use float32 for batch_idx (not masks.dtype): torch.arange(n, dtype=torch.bool)
    # would clip indices to {0, 1}, silently corrupting roi_align batch indices for N > 2.
    batch_idx = torch.arange(n, device=device, dtype=torch.float32)
    roi_boxes = torch.stack([batch_idx, sx1, sy1, sx2, sy2], dim=1)

    crops = roi_align(
        masks.unsqueeze(1).float(),
        roi_boxes,
        output_size=(crop_size, crop_size),
        aligned=True,
    ).squeeze(
        1
    )  # (N, crop_size, crop_size)
    weights = (crops > mask_threshold).float()  # (N, crop_size, crop_size)
    weights_flat = weights.reshape(n, -1)  # (N, P), P = crop_size**2

    # --- 2. Mask-weighted centroid, in LOCAL unit-square coordinates. ---
    lin = torch.linspace(-0.5, 0.5, crop_size, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)  # (2, P)

    total_weight = weights_flat.sum(dim=1).clamp(min=1e-6)  # (N,)
    centroid_local = (weights_flat @ grid.T) / total_weight[:, None]  # (N, 2)
    has_foreground = weights_flat.sum(dim=1) > 0  # (N,)

    coords = grid[None, :, :] - centroid_local[:, :, None]  # (N, 2, P)
    coords = coords.transpose(1, 2)  # (N, P, 2)

    # --- 3. Coarse batched angle search. ---
    angles = torch.linspace(
        0.0, math.pi, num_angles + 1, device=device, dtype=torch.float32
    )[:-1]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    # Rotation matrices for every candidate angle: (K, 2, 2). Uses R(-angle)
    # (u = dx*cos + dy*sin, v = -dx*sin + dy*cos) so that the reported angle
    # is the physical orientation of the "u" (width) axis in the input
    # coordinate frame, not its negation.
    rot = torch.stack(
        [torch.stack([cos_a, sin_a], dim=1), torch.stack([-sin_a, cos_a], dim=1)],
        dim=1,
    )
    # (N, P, 2) @ (K, 2, 2)^T broadcast -> (N, K, P, 2)
    proj = torch.einsum("npc,kdc->nkpd", coords, rot)
    u, v = proj[..., 0], proj[..., 1]  # each (N, K, P)

    fg = weights_flat[:, None, :] > 0  # (N, 1, P) broadcast over K
    pos_inf = torch.full_like(u, float("inf"))
    neg_inf = torch.full_like(u, float("-inf"))
    u_max = torch.where(fg, u, neg_inf).amax(dim=-1)
    u_min = torch.where(fg, u, pos_inf).amin(dim=-1)
    v_max = torch.where(fg, v, neg_inf).amax(dim=-1)
    v_min = torch.where(fg, v, pos_inf).amin(dim=-1)
    width_k = (u_max - u_min).clamp(min=0.0)  # (N, K)
    height_k = (v_max - v_min).clamp(min=0.0)
    area_k = width_k * height_k

    k_star = area_k.argmin(dim=1)  # (N,)

    # --- 4. Closed-form 3-point parabolic sub-grid refinement (circular). ---
    k_prev = (k_star - 1) % num_angles
    k_next = (k_star + 1) % num_angles
    area_prev = area_k.gather(1, k_prev[:, None]).squeeze(1)
    area_star = area_k.gather(1, k_star[:, None]).squeeze(1)
    area_next = area_k.gather(1, k_next[:, None]).squeeze(1)
    denom = area_next - 2.0 * area_star + area_prev
    safe = denom.abs() > 1e-6
    offset = torch.where(
        safe,
        -0.5 * (area_next - area_prev) / denom.clamp(min=1e-6),
        torch.zeros_like(denom),
    )
    offset = offset.clamp(-1.0, 1.0)
    angle_step = math.pi / num_angles
    theta_star = angles.gather(0, k_star)
    theta_refined = (theta_star + offset * angle_step) % math.pi

    # --- Recompute width/height once more at the refined per-detection angle. ---
    cos_r, sin_r = torch.cos(theta_refined), torch.sin(theta_refined)
    rot_r = torch.stack(
        [torch.stack([cos_r, sin_r], dim=1), torch.stack([-sin_r, cos_r], dim=1)], dim=1
    )  # (N, 2, 2)
    proj_r = torch.bmm(coords, rot_r.transpose(1, 2))  # (N, P, 2)
    u_r, v_r = proj_r[..., 0], proj_r[..., 1]
    fg2 = weights_flat > 0
    u_r_max = torch.where(fg2, u_r, torch.full_like(u_r, float("-inf"))).amax(dim=-1)
    u_r_min = torch.where(fg2, u_r, torch.full_like(u_r, float("inf"))).amin(dim=-1)
    v_r_max = torch.where(fg2, v_r, torch.full_like(v_r, float("-inf"))).amax(dim=-1)
    v_r_min = torch.where(fg2, v_r, torch.full_like(v_r, float("inf"))).amin(dim=-1)
    w_local = (u_r_max - u_r_min).clamp(min=0.0)
    h_local = (v_r_max - v_r_min).clamp(min=0.0)

    # --- Canonicalize: a rectangle probed at angle theta with (w, h) and at
    #     theta + pi/2 with (h, w) describe the identical physical rectangle
    #     (rotating the probing axes by exactly 90 degrees swaps u/v and
    #     always yields the same bounding area), so argmin over the coarse
    #     grid can land on either tied candidate. Canonicalize on the "w is
    #     the longer side" convention so the reported angle consistently
    #     tracks the major axis. ---
    swap = h_local > w_local
    w_local, h_local = (
        torch.where(swap, h_local, w_local),
        torch.where(swap, w_local, h_local),
    )
    theta_refined = torch.where(
        swap, (theta_refined + math.pi / 2.0) % math.pi, theta_refined
    )

    # --- Map centroid + size back from local unit-square units to the input
    #     masks'/boxes' physical coordinate space. ---
    cx = bcx + centroid_local[:, 0] * side
    cy = bcy + centroid_local[:, 1] * side
    w = w_local * side
    h = h_local * side

    result = torch.stack([cx, cy, w, h, theta_refined], dim=1)
    nan_row = torch.full((5,), float("nan"), device=device, dtype=torch.float32)
    result = torch.where(has_foreground[:, None], result, nan_row[None, :])
    return result
