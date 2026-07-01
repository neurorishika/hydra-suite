"""Canonical crop extraction for HYDRA tracking.

Provides a single affine-warped crop per detection where:
- The animal's major axis is horizontal.
- Head faces right (after head-tail orientation).
- Foreign OBB regions are suppressed.
- Canvas aspect ratio matches the species (adaptive dimensions).

All downstream consumers (head-tail classifier, pose estimator, CNN identity,
dataset export) are served from this one canonical crop.  An invertible
affine matrix ``M_canonical`` maps frame → canonical coordinates, and its
inverse maps predictions back to the original frame.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CanonicalCropResult:
    """Result of canonical crop extraction for one detection."""

    crop: np.ndarray  # (H, W, C) canonical image
    M_canonical: np.ndarray  # (2, 3) composite affine: frame → canonical
    M_inverse: np.ndarray  # (2, 3) pseudo-inverse: canonical → frame
    heading_rad: float  # directed heading (radians, 0 = right)
    directed: bool  # True if heading is reliable


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_crop_dimensions(
    long_edge: int,
    reference_aspect_ratio: float,
) -> Tuple[int, int]:
    """Derive (W, H) from long edge and species aspect ratio.

    Returns:
        (width, height) where width >= height.
    """
    long_edge = max(8, int(long_edge))
    ar = max(1.0, float(reference_aspect_ratio))
    short_edge = max(8, round(long_edge / ar))
    return long_edge, short_edge


def compute_native_crop_dimensions(
    corners: np.ndarray,
    reference_aspect_ratio: float,
    padding_fraction: float,
) -> Tuple[int, int]:
    """Derive canvas (W, H) from an OBB's native pixel extent.

    The long edge of the canvas matches the OBB's major axis (times padding)
    at native pixel scale — no downsampling.  The short edge is derived
    from ``reference_aspect_ratio`` so all crops share a consistent AR.

    Both dimensions are rounded to the nearest even integer (≥ 8).

    Args:
        corners: (4, 2) OBB corner array in frame coordinates.
        reference_aspect_ratio: Species AR (long / short), e.g. 2.0.
        padding_fraction: Fractional expansion (e.g. 0.1 = 10 %).

    Returns:
        (width, height) — width is the long (major-axis) dimension.
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))
    major = max(e01, e12)

    margin = 1.0 + max(0.0, float(padding_fraction))
    ar = max(1.0, float(reference_aspect_ratio))

    raw_w = major * margin

    # Round W to even first, then derive H from the rounded W so that
    # canvas_w / canvas_h stays close to the target AR.
    canvas_w = max(8, int(math.ceil(raw_w / 2.0) * 2))
    canvas_h = max(8, int(round(canvas_w / ar / 2.0) * 2))
    return canvas_w, canvas_h


def compute_native_scale_affine(
    corners: np.ndarray,
    reference_aspect_ratio: float,
    padding_fraction: float,
) -> Tuple[np.ndarray, int, int, float]:
    """Build a native-scale alignment affine for one OBB.

    Like :func:`compute_alignment_affine`, but the output canvas is sized
    to preserve the OBB's native pixel extent (no down- or up-sampling).
    The canvas aspect ratio is standardised to *reference_aspect_ratio*.

    Args:
        corners: (4, 2) OBB corner array in frame coordinates.
        reference_aspect_ratio: Species AR (long / short).
        padding_fraction: Fractional expansion (e.g. 0.1).

    Returns:
        (M_align, canvas_w, canvas_h, major_axis_theta)
    """
    canvas_w, canvas_h = compute_native_crop_dimensions(
        corners, reference_aspect_ratio, padding_fraction
    )
    M_align, theta = compute_alignment_affine(
        corners, canvas_w, canvas_h, padding_fraction
    )
    return M_align, canvas_w, canvas_h, theta


def compute_alignment_affine(
    corners: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    padding_fraction: float,
) -> Tuple[np.ndarray, float]:
    """Compute M_align from OBB corners.

    Builds a 2×3 affine matrix that maps the padded OBB from frame space
    into a rotation-normalised canvas of size ``(canvas_w, canvas_h)``
    with the major axis horizontal and the centroid centred.

    Args:
        corners: (4, 2) OBB corner array in frame coordinates.
        canvas_w: Output width in pixels.
        canvas_h: Output height in pixels.
        padding_fraction: Fractional expansion applied to the OBB.

    Returns:
        (M_align, major_axis_theta) — the 2×3 affine matrix and the
        OBB major-axis angle in radians.
    """
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))

    if e01 < 1e-3 or e12 < 1e-3:
        raise ValueError("Degenerate OBB (zero-length edge)")

    if e01 >= e12:
        major_vec = c[1] - c[0]
    else:
        major_vec = c[2] - c[1]

    cx = float(np.mean(c[:, 0]))
    cy = float(np.mean(c[:, 1]))
    angle = float(math.atan2(float(major_vec[1]), float(major_vec[0])))

    major = max(e01, e12)
    minor = min(e01, e12)

    margin = 1.0 + padding_fraction
    w_exp = float(major) * margin
    h_exp = float(minor) * margin
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    hw = w_exp * 0.5
    hh = h_exp * 0.5

    # Source triangle: top-left, top-right, bottom-left of the expanded OBB
    src_pts = np.array(
        [
            [cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a],
            [cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a],
            [cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[0, 0], [canvas_w, 0], [0, canvas_h]],
        dtype=np.float32,
    )

    M_align = cv2.getAffineTransform(src_pts, dst_pts)
    return M_align, angle


def extract_canonical_crop(
    frame: np.ndarray,
    M_align: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    foreign_corners: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Apply M_align to extract a rotation-normalised crop.

    Optionally masks foreign OBB regions in canonical space.
    """
    crop = cv2.warpAffine(
        frame,
        M_align,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if foreign_corners:
        _apply_foreign_mask_canonical(crop, M_align, foreign_corners, bg_color)

    return crop


def gpu_canonical_crop(
    frame_chw: "torch.Tensor",
    M_align: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> "torch.Tensor":
    """GPU-native affine warp replicating ``extract_canonical_crop``.

    Replaces ``cv2.warpAffine`` for frames already resident on a CUDA (or MPS)
    device.  ``M_align`` is the 2×3 forward affine produced by
    :func:`compute_alignment_affine` mapping frame pixel coords to canvas pixel
    coords.  The function inverts it on CPU (negligible), builds a normalised
    ``F.affine_grid`` theta, and uses ``F.grid_sample`` with bilinear
    interpolation and border replication — matching the cv2 default behaviour.

    Parameters
    ----------
    frame_chw:
        CUDA tensor ``(C, H, W)`` float32.  Channel order is preserved
        unchanged (caller is responsible for any BGR↔RGB flip).
    M_align:
        ``(2, 3)`` float64/float32 numpy array from ``compute_alignment_affine``.
    canvas_w, canvas_h:
        Output canvas dimensions in pixels.

    Returns
    -------
    torch.Tensor
        ``(C, canvas_h, canvas_w)`` float32 on the same device as
        ``frame_chw``.
    """
    import cv2 as _cv2
    import torch
    import torch.nn.functional as F

    C, H_in, W_in = frame_chw.shape

    # Invert M_align (forward src→dst) to get dst→src mapping required by
    # F.grid_sample (which samples source coords for each output pixel).
    M_inv = _cv2.invertAffineTransform(np.asarray(M_align, dtype=np.float64))

    # Build normalised theta (2×3) for F.affine_grid with align_corners=True.
    # Derivation: norm_src = theta @ [norm_dst_x, norm_dst_y, 1]^T
    # where norm = 2*pixel / (dim-1) - 1 (align_corners=True convention).
    #
    # theta[row, 0] = M_inv[row, 0] * (W_out-1) / (dim_in[row]-1)
    # theta[row, 1] = M_inv[row, 1] * (H_out-1) / (dim_in[row]-1)
    # theta[row, 2] = theta[row,0] + theta[row,1] + 2*M_inv[row,2]/(dim_in[row]-1) - 1
    sw = float(canvas_w - 1)
    sh = float(canvas_h - 1)
    inv_win = 1.0 / max(float(W_in - 1), 1.0)
    inv_hin = 1.0 / max(float(H_in - 1), 1.0)

    t00 = M_inv[0, 0] * sw * inv_win
    t01 = M_inv[0, 1] * sh * inv_win
    t10 = M_inv[1, 0] * sw * inv_hin
    t11 = M_inv[1, 1] * sh * inv_hin

    theta = np.array(
        [
            [t00, t01, t00 + t01 + 2.0 * M_inv[0, 2] * inv_win - 1.0],
            [t10, t11, t10 + t11 + 2.0 * M_inv[1, 2] * inv_hin - 1.0],
        ],
        dtype=np.float32,
    )

    theta_t = torch.as_tensor(
        theta, dtype=torch.float32, device=frame_chw.device
    ).unsqueeze(
        0
    )  # (1, 2, 3)

    with torch.inference_mode():
        grid = F.affine_grid(theta_t, (1, C, canvas_h, canvas_w), align_corners=True)
        crop = F.grid_sample(
            frame_chw.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return crop.squeeze(0)  # (C, canvas_h, canvas_w)


def gpu_canonical_crop_batch(
    frame_chw: "torch.Tensor",
    M_aligns: list,
    canvas_w: int,
    canvas_h: int,
) -> "torch.Tensor":
    """Batch version of :func:`gpu_canonical_crop` for N crops from *one* frame.

    Replaces N serial ``F.affine_grid`` + ``F.grid_sample`` calls with a single
    pair of batched calls.  This reduces GPU kernel launch overhead from O(N) to
    O(1) when extracting many detections from the same frame, which is the
    common case for dense multi-animal tracking (e.g. 50 animals × 8 frames).

    Parameters
    ----------
    frame_chw:
        CUDA tensor ``(C, H, W)`` float32 — shared source for all N crops.
    M_aligns:
        List of N ``(2, 3)`` numpy float64/float32 arrays from
        :func:`compute_alignment_affine`, one per detection.
    canvas_w, canvas_h:
        Output canvas dimensions (same for every crop).

    Returns
    -------
    torch.Tensor
        ``(N, C, canvas_h, canvas_w)`` float32 on the same device as
        ``frame_chw``.
    """
    import torch
    import torch.nn.functional as F

    N = len(M_aligns)
    if N == 0:
        C = frame_chw.shape[0]
        return torch.zeros(
            0,
            C,
            canvas_h,
            canvas_w,
            dtype=frame_chw.dtype,
            device=frame_chw.device,
        )
    if N == 1:
        return gpu_canonical_crop(frame_chw, M_aligns[0], canvas_w, canvas_h).unsqueeze(
            0
        )

    C, H_in, W_in = frame_chw.shape
    sw = float(canvas_w - 1)
    sh = float(canvas_h - 1)
    inv_win = 1.0 / max(float(W_in - 1), 1.0)
    inv_hin = 1.0 / max(float(H_in - 1), 1.0)

    # Build all theta matrices on CPU (negligible cost) then transfer once.
    thetas_np = np.empty((N, 2, 3), dtype=np.float32)
    for i, M_align in enumerate(M_aligns):
        M_inv = cv2.invertAffineTransform(np.asarray(M_align, dtype=np.float64))
        t00 = M_inv[0, 0] * sw * inv_win
        t01 = M_inv[0, 1] * sh * inv_win
        t10 = M_inv[1, 0] * sw * inv_hin
        t11 = M_inv[1, 1] * sh * inv_hin
        thetas_np[i, 0, 0] = t00
        thetas_np[i, 0, 1] = t01
        thetas_np[i, 0, 2] = t00 + t01 + 2.0 * M_inv[0, 2] * inv_win - 1.0
        thetas_np[i, 1, 0] = t10
        thetas_np[i, 1, 1] = t11
        thetas_np[i, 1, 2] = t10 + t11 + 2.0 * M_inv[1, 2] * inv_hin - 1.0

    thetas_t = torch.as_tensor(
        thetas_np, dtype=torch.float32, device=frame_chw.device
    )  # (N, 2, 3)

    with torch.inference_mode():
        # ONE affine_grid call + ONE grid_sample call for all N crops.
        grid = F.affine_grid(
            thetas_t, (N, C, canvas_h, canvas_w), align_corners=True
        )  # (N, canvas_h, canvas_w, 2)
        frame_expanded = frame_chw.unsqueeze(0).expand(N, -1, -1, -1)  # (N, C, H, W)
        crops = F.grid_sample(
            frame_expanded.contiguous(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (N, C, canvas_h, canvas_w)
        return crops


def apply_headtail_rotation(
    crop: np.ndarray,
    M_align: np.ndarray,
    direction: str,
    canvas_w: int,
    canvas_h: int,
    treat_updown_as_unknown: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rotate crop so head faces right based on head-tail classification.

    Args:
        crop: Rotation-normalised crop (major axis horizontal).
        M_align: The 2×3 alignment affine.
        direction: One of ``'left'``, ``'right'``, ``'up'``, ``'down'``,
            ``'unknown'``.
        canvas_w: Original canvas width.
        canvas_h: Original canvas height.
        treat_updown_as_unknown: If True, treat ``'up'``/``'down'`` as
            ``'unknown'`` (no rotation applied).

    Returns:
        (rotated_crop, M_canonical, M_inverse, orientation_offset_rad)
    """
    if treat_updown_as_unknown and direction in ("up", "down"):
        direction = "unknown"

    if direction == "left":
        # 180° rotation about canvas centre
        rotated = cv2.rotate(crop, cv2.ROTATE_180)
        offset_rad = math.pi
        out_w, out_h = canvas_w, canvas_h
    elif direction == "up":
        # 90° CW
        rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        offset_rad = -math.pi / 2.0
        out_w, out_h = canvas_h, canvas_w
    elif direction == "down":
        # 90° CCW
        rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        offset_rad = math.pi / 2.0
        out_w, out_h = canvas_h, canvas_w
    else:
        # 'right' or 'unknown' — no rotation needed
        rotated = crop
        offset_rad = 0.0
        out_w, out_h = canvas_w, canvas_h

    M_orient = _rotation_matrix(offset_rad, canvas_w, canvas_h, out_w, out_h)
    M_canonical = _compose_affine(M_orient, M_align)
    M_inverse = cv2.invertAffineTransform(M_canonical)

    return rotated, M_canonical, M_inverse, offset_rad


def invert_keypoints(
    keypoints: np.ndarray,
    M_inverse: np.ndarray,
) -> np.ndarray:
    """Map (K, 2) or (K, 3) keypoints from canonical to frame coordinates.

    Confidence values (column 2, if present) pass through unchanged.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] == 0:
        return kp

    has_conf = kp.shape[1] >= 3
    xy = kp[:, :2]

    M = np.asarray(M_inverse, dtype=np.float64)
    ones = np.ones((xy.shape[0], 1), dtype=np.float64)
    xy_h = np.hstack([xy, ones])  # (K, 3)
    mapped = (M @ xy_h.T).T  # (K, 2)

    if has_conf:
        result = np.empty_like(kp)
        result[:, :2] = mapped
        result[:, 2:] = kp[:, 2:]
        return result
    return mapped


def extract_and_classify_batch(
    frames: List[np.ndarray],
    per_frame_corners: List[List[np.ndarray]],
    canvas_w: int,
    canvas_h: int,
    padding_fraction: float = 0.1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    suppress_foreign: bool = True,
    per_frame_all_corners: Optional[List[List[np.ndarray]]] = None,
) -> List[List[Optional[CanonicalCropResult]]]:
    """Full canonical pipeline for a batch of frames (without head-tail).

    Extracts rotation-normalised canonical crops for every detection across
    all frames.  Head-tail classification is *not* run here — the caller
    is responsible for directing the crops afterwards via
    ``apply_headtail_rotation``.

    Args:
        frames: List of video frames (BGR).
        per_frame_corners: Per-frame list of OBB corner arrays.
        canvas_w: Canonical crop width.
        canvas_h: Canonical crop height.
        padding_fraction: OBB expansion factor.
        bg_color: Background fill colour.
        suppress_foreign: Whether to mask foreign OBB regions.
        per_frame_all_corners: Per-frame list of *all* OBB corners for
            foreign-OBB masking.  If None, ``per_frame_corners`` is used.

    Returns:
        Nested list ``[frame][detection]`` of ``CanonicalCropResult | None``.
    """
    results: List[List[Optional[CanonicalCropResult]]] = []

    for fi, frame in enumerate(frames):
        corners_list = per_frame_corners[fi]
        all_corners = (
            per_frame_all_corners[fi] if per_frame_all_corners else corners_list
        )
        frame_results: List[Optional[CanonicalCropResult]] = []

        for di, corners in enumerate(corners_list):
            try:
                M_align, axis_theta = compute_alignment_affine(
                    corners, canvas_w, canvas_h, padding_fraction
                )
            except ValueError:
                frame_results.append(None)
                continue

            # Foreign corners: everything except current detection
            foreign = None
            if suppress_foreign and len(all_corners) > 1:
                foreign = [all_corners[j] for j in range(len(all_corners)) if j != di]

            crop = extract_canonical_crop(
                frame, M_align, canvas_w, canvas_h, bg_color, foreign
            )

            M_inverse = cv2.invertAffineTransform(M_align)

            frame_results.append(
                CanonicalCropResult(
                    crop=crop,
                    M_canonical=M_align.astype(np.float32),
                    M_inverse=M_inverse.astype(np.float32),
                    heading_rad=float(axis_theta),
                    directed=False,
                )
            )

        results.append(frame_results)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_foreign_mask_canonical(
    crop: np.ndarray,
    M_align: np.ndarray,
    foreign_corners_list: List[np.ndarray],
    bg_color: Tuple[int, int, int],
) -> None:
    """Fill foreign OBB regions with background colour in canonical space.

    Transforms each foreign OBB's corners into canonical space via M_align,
    then fills the polygon with *bg_color*.  Modifies *crop* in-place.
    """
    M = np.asarray(M_align, dtype=np.float64)
    R = M[:, :2]  # (2, 2)
    t = M[:, 2:]  # (2, 1)

    for corners in foreign_corners_list:
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        canonical_pts = (R @ pts.T + t).T  # (N, 2)
        poly = canonical_pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(crop, [poly], bg_color)


def _rotation_matrix(
    angle_rad: float,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    """Build a 2×3 rotation matrix about the source canvas centre.

    For 90° rotations the destination canvas dimensions are swapped, so
    the translation component accounts for the canvas resize.
    """
    cx_src = src_w / 2.0
    cy_src = src_h / 2.0
    cx_dst = dst_w / 2.0
    cy_dst = dst_h / 2.0

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rotation about source centre, then translate so output centres match
    tx = cx_dst - cos_a * cx_src + sin_a * cy_src
    ty = cy_dst - sin_a * cx_src - cos_a * cy_src

    return np.array(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
        dtype=np.float64,
    )


def _compose_affine(M2: np.ndarray, M1: np.ndarray) -> np.ndarray:
    """Compose two 2×3 affine transforms: result = M2 ∘ M1.

    Promotes to 3×3, multiplies, then extracts the top 2 rows.
    """
    A = np.eye(3, dtype=np.float64)
    A[:2, :] = np.asarray(M2, dtype=np.float64)
    B = np.eye(3, dtype=np.float64)
    B[:2, :] = np.asarray(M1, dtype=np.float64)
    C = A @ B
    return C[:2, :].astype(np.float64)
