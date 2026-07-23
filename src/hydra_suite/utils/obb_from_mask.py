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
5. Recenter: the extents above are measured about the mass centroid, so the
   emitted center is shifted to the MIDPOINT of the (min, max) extent in the
   refined rotated frame — the rectangle's true center, which for an
   asymmetric mask is not the mass centroid.
6. Canonicalize w/h so that the wider dimension is always ``w`` and ``angle``
   always tracks the major axis (swap w/h and adjust angle by π/2 if needed).

Everything after step 1 operates on a small, fixed-size tensor
(``crop_size x crop_size`` per detection), so this stays cheap even for many
detections per frame and never performs a per-detection Python loop.
"""

from __future__ import annotations

import logging
import math

import torch
from torchvision.ops import roi_align

_logger = logging.getLogger(__name__)

# Set once the CPU fallback below has fired, so the warning is emitted a
# single time per process rather than once per frame/detection batch.
_warned_mps_roi_align_fallback = False


def _is_mps_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` lives on an MPS device.

    Factored out (rather than inlined) so tests can monkeypatch this single
    predicate to deterministically exercise the MPS fallback branch below
    without requiring actual MPS hardware.
    """
    return tensor.device.type == "mps"


def _roi_align_with_mps_fallback(
    input_tensor: torch.Tensor,
    boxes: torch.Tensor,
    *,
    output_size: tuple[int, int],
    aligned: bool,
) -> torch.Tensor:
    """Call ``torchvision.ops.roi_align``, tolerating a missing MPS kernel.

    ``roi_align`` only gained a native MPS backend in torchvision 0.16.0 (see
    the ``mps`` extra's ``torchvision>=0.16.0`` pin in ``pyproject.toml``); on
    an older-but-otherwise-valid install it raises ``NotImplementedError`` for
    MPS tensors, and since this repo does not set
    ``PYTORCH_ENABLE_MPS_FALLBACK``, that would otherwise hard-crash the
    segment-as-OBB path. Detect exactly that case -- ``NotImplementedError``
    on an ``mps`` tensor -- and transparently retry on CPU, moving the result
    back to the original device before returning. Any other exception, or a
    ``NotImplementedError`` on a non-MPS device, propagates unchanged so real
    errors are never swallowed.
    """
    try:
        return roi_align(input_tensor, boxes, output_size=output_size, aligned=aligned)
    except NotImplementedError:
        if not _is_mps_tensor(input_tensor):
            raise
        global _warned_mps_roi_align_fallback
        if not _warned_mps_roi_align_fallback:
            _logger.warning(
                "torchvision.ops.roi_align has no MPS kernel on this install "
                "(requires torchvision>=0.16.0); falling back to a CPU "
                "roi_align for the segment-as-OBB path. This is slower but "
                "correct -- upgrade torchvision to remove this fallback."
            )
            _warned_mps_roi_align_fallback = True
        result = roi_align(
            input_tensor.cpu(),
            boxes.cpu(),
            output_size=output_size,
            aligned=aligned,
        )
        return result.to(input_tensor.device)


def letterbox_gain_pad(
    mask_shape: tuple[int, int], orig_shape: tuple[int, int]
) -> tuple[float, float, float]:
    """Return ``(gain, pad_x, pad_y)`` mapping ``orig_shape`` -> ``mask_shape``.

    Follows the same structure as ``ultralytics.utils.ops.scale_boxes``: a
    single uniform ``gain`` (``mask`` canvases are always square, matching a
    square YOLO letterboxed input) plus a symmetric pad per axis. Using a
    single scalar gain (rather than independent per-axis ratios) is what
    keeps rotation angles correct when the original frame is not square.
    (``scale_boxes`` additionally subtracts 0.1 before rounding its pad; the
    resulting sub-pixel difference is irrelevant here, where the pad only
    re-centers a crop window, so this uses the plain rounded half-pad.)

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
    foreground_only: bool = False,
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
    foreground_only:
        When ``True``, project only each detection's FOREGROUND pixels
        (ragged, padded to the batch-max foreground count ``M``) instead of
        all ``crop_size**2`` grid points, shrinking the projection tensor
        5-8x and running ~37% faster at large N. Bit-identical to the default
        (up to float summation-order noise on the centroid, ``atol`` ~1e-4).
        Its ONE cost is a single host sync to read ``M`` as a Python int, so
        it must only be enabled on paths that already materialize to CPU per
        frame -- NOT on the zero-CPU-sync native-CUDA raw path. Default
        ``False`` preserves today's exact full-pixel, sync-free behavior.

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

    crops = _roi_align_with_mps_fallback(
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
    # roi_align(aligned=True) samples the CENTER of each of the crop_size bins
    # spanning the ROI, i.e. bin i sits at (i + 0.5)/crop_size of the ROI side.
    # An endpoint-inclusive linspace(-0.5, 0.5, crop_size) would instead space
    # samples by 1/(crop_size - 1), stretching every measured extent by
    # crop_size/(crop_size - 1) (~1.6% at crop_size=64) and systematically
    # over-estimating w/h (and hence `size`, which feeds the size filters).
    lin = (
        torch.arange(crop_size, device=device, dtype=torch.float32) + 0.5
    ) / crop_size - 0.5
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)  # (2, P)
    grid_pts = grid.T  # (P, 2), one (x, y) per grid point

    fg_count = weights_flat.sum(dim=1)  # (N,), float count of foreground pixels
    has_foreground = fg_count > 0  # (N,)

    if foreground_only:
        # Gather ONLY the foreground pixel coordinates into a compact,
        # padded (N, M, 2) tensor (M = batch-max foreground count) plus a
        # (N, M) validity mask marking real vs padding slots. This replaces
        # the full (N, P, 2) grid so every downstream min/max/mean touches
        # ~1/(fg-fraction) fewer elements. Reading M is the ONE host sync
        # this branch performs (acceptable only on the CPU-materializing
        # path -- see the ``foreground_only`` docstring note).
        nz = weights_flat.nonzero(as_tuple=False)  # (T, 2): (row, col), row-sorted
        counts = fg_count.long()  # (N,)
        M = int(counts.max().item()) if nz.shape[0] > 0 else 0
        if M == 0:
            # Every mask is empty: no foreground pixel anywhere. Short-circuit
            # to all-NaN rows -- proceeding would reduce over a zero-width dim.
            return torch.full((n, 5), float("nan"), dtype=torch.float32, device=device)
        rows = nz[:, 0]
        cols = nz[:, 1]
        # Per-row slot index 0..count-1: since nonzero() is row-sorted, the
        # slot is the running position minus that row's exclusive-prefix start.
        offsets = torch.zeros(n, device=device, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)[:-1]
        slot = torch.arange(nz.shape[0], device=device) - offsets[rows]  # (T,)
        coords_pad = torch.zeros(n, M, 2, device=device, dtype=torch.float32)
        coords_pad[rows, slot] = grid_pts[cols]
        valid_pix = torch.zeros(n, M, device=device, dtype=torch.bool)
        valid_pix[rows, slot] = True
        # Centroid = mean of foreground coords. Padding slots are zero and
        # thus contribute nothing to the sum; divide by the true count.
        centroid_local = coords_pad.sum(dim=1) / fg_count.clamp(min=1e-6)[:, None]
        coords = coords_pad - centroid_local[:, None, :]  # (N, M, 2)
    else:
        total_weight = fg_count.clamp(min=1e-6)  # (N,)
        centroid_local = (weights_flat @ grid_pts) / total_weight[:, None]  # (N, 2)
        valid_pix = weights_flat > 0  # (N, P)
        coords = grid_pts[None, :, :] - centroid_local[:, None, :]  # (N, P, 2)

    # ``valid_pix`` is (N, L) with L in {P, M}; ``coords`` is (N, L, 2).
    bg_pix = ~valid_pix  # (N, L): True at background/padding slots

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
    # (N, L, 2) @ (K, 2, 2)^T broadcast -> (N, K, L, 2)
    proj = torch.einsum("nlc,kdc->nkld", coords, rot)
    u, v = proj[..., 0], proj[..., 1]  # each (N, K, L)

    # Background/padding pixels are pushed to +/-inf so they can never win a
    # min/max. masked_fill (rather than torch.where against two full_like
    # sentinel tensors) avoids materialising two extra (N, K, L) tensors --
    # ~300 MB of transient allocations at N=100, K=24, L=4096. Numerically
    # identical.
    bg = bg_pix[:, None, :]  # (N, 1, L), broadcasts over K
    u_max = u.masked_fill(bg, float("-inf")).amax(dim=-1)
    u_min = u.masked_fill(bg, float("inf")).amin(dim=-1)
    v_max = v.masked_fill(bg, float("-inf")).amax(dim=-1)
    v_min = v.masked_fill(bg, float("inf")).amin(dim=-1)
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
    proj_r = torch.bmm(coords, rot_r.transpose(1, 2))  # (N, L, 2)
    u_r, v_r = proj_r[..., 0], proj_r[..., 1]
    bg2 = bg_pix
    u_r_max = u_r.masked_fill(bg2, float("-inf")).amax(dim=-1)
    u_r_min = u_r.masked_fill(bg2, float("inf")).amin(dim=-1)
    v_r_max = v_r.masked_fill(bg2, float("-inf")).amax(dim=-1)
    v_r_min = v_r.masked_fill(bg2, float("inf")).amin(dim=-1)
    w_local = (u_r_max - u_r_min).clamp(min=0.0)
    h_local = (v_r_max - v_r_min).clamp(min=0.0)

    # --- Move the center from the mask's MASS CENTROID to the RECTANGLE's
    #     center: w/h above are the full extents measured ABOUT the centroid,
    #     so for any asymmetric mask (i.e. every real animal) a rectangle
    #     centered on the centroid does not bound the mask and its center is
    #     biased toward the heavier end. The rectangle's center is the midpoint
    #     of the (min, max) extent in the refined rotated frame; rotate that
    #     midpoint back into local coordinates (dx = u*cos - v*sin,
    #     dy = u*sin + v*cos -- the inverse of the R(-theta) probe above).
    #     Done BEFORE the w/h swap below, which does not move the center. ---
    mid_u = (u_r_max + u_r_min) / 2.0
    mid_v = (v_r_max + v_r_min) / 2.0
    center_local = centroid_local + torch.stack(
        [mid_u * cos_r - mid_v * sin_r, mid_u * sin_r + mid_v * cos_r], dim=1
    )  # (N, 2)

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
    cx = bcx + center_local[:, 0] * side
    cy = bcy + center_local[:, 1] * side
    w = w_local * side
    h = h_local * side

    result = torch.stack([cx, cy, w, h, theta_refined], dim=1)
    nan_row = torch.full((5,), float("nan"), device=device, dtype=torch.float32)
    result = torch.where(has_foreground[:, None], result, nan_row[None, :])
    return result
