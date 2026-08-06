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

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)

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
# Layer 1 geometry resolution
# ---------------------------------------------------------------------------


def _resolve_canvas(
    canvas_w: Optional[int],
    canvas_h: Optional[int],
    geometry: Optional[CanonicalGeometry],
) -> Tuple[int, int]:
    """Reconcile the legacy ``(canvas_w, canvas_h)`` ints with a ``geometry``.

    Either ``(canvas_w, canvas_h)`` or ``geometry`` must be supplied. If both
    are supplied they must agree -- a caller that passes a geometry must not
    also be able to silently smuggle in mismatched dimensions.
    """
    if geometry is not None:
        gw, gh = geometry.canvas_w, geometry.canvas_h
        if canvas_w is not None and int(canvas_w) != gw:
            raise ValueError(
                f"canvas_w={canvas_w} disagrees with geometry.canvas_w={gw}"
            )
        if canvas_h is not None and int(canvas_h) != gh:
            raise ValueError(
                f"canvas_h={canvas_h} disagrees with geometry.canvas_h={gh}"
            )
        return gw, gh
    if canvas_w is None or canvas_h is None:
        raise ValueError(
            "extract_canonical_crop requires either geometry or both "
            "canvas_w and canvas_h"
        )
    return int(canvas_w), int(canvas_h)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def extract_canonical_crop(
    frame: np.ndarray,
    M_align: np.ndarray,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    foreign_corners: Optional[List[np.ndarray]] = None,
    own_corners: Optional[np.ndarray] = None,
    *,
    geometry: Optional[CanonicalGeometry] = None,
) -> np.ndarray:
    """Apply M_align to extract a rotation-normalised crop.

    Optionally masks foreign OBB regions in canonical space. *own_corners*
    (the current detection's own OBB, frame coordinates), when given,
    excludes any overlap with foreign OBBs so the current animal's own body
    is never masked out — see ``_apply_foreign_mask_canonical``.

    Either ``(canvas_w, canvas_h)`` or ``geometry`` (Layer 1's
    :class:`~hydra_suite.core.canonicalization.geometry.CanonicalGeometry`)
    must be given. If both are given they must agree.
    """
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
    crop = cv2.warpAffine(
        frame,
        M_align,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if foreign_corners:
        _apply_foreign_mask_canonical(
            crop, M_align, foreign_corners, bg_color, own_corners=own_corners
        )

    return crop


def gpu_canonical_crop(
    frame_chw: "torch.Tensor",
    M_align: np.ndarray,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    *,
    geometry: Optional[CanonicalGeometry] = None,
) -> "torch.Tensor":
    """GPU-native affine warp replicating ``extract_canonical_crop``.

    Replaces ``cv2.warpAffine`` for frames already resident on a CUDA (or MPS)
    device.  ``M_align`` is the 2×3 forward affine produced by
    :func:`~hydra_suite.core.canonicalization.geometry.canonical_affine`
    mapping frame pixel coords to canvas pixel
    coords.  The function inverts it on CPU (negligible), builds a normalised
    ``F.affine_grid`` theta, and uses ``F.grid_sample`` with bilinear
    interpolation and zero padding — matching ``extract_canonical_crop``'s
    ``cv2.BORDER_CONSTANT`` (value 0) so out-of-frame canvas pixels mean
    "no data" on both CPU and GPU.

    Parameters
    ----------
    frame_chw:
        CUDA tensor ``(C, H, W)`` float32.  Channel order is preserved
        unchanged (caller is responsible for any BGR↔RGB flip).
    M_align:
        ``(2, 3)`` float64/float32 numpy array from ``canonical_affine``.
    canvas_w, canvas_h:
        Output canvas dimensions in pixels. Either these or ``geometry`` must
        be given; if both are given they must agree.
    geometry:
        Layer 1 :class:`CanonicalGeometry`, alternative to ``canvas_w``/``canvas_h``.

    Returns
    -------
    torch.Tensor
        ``(C, canvas_h, canvas_w)`` float32 on the same device as
        ``frame_chw``.
    """
    import cv2 as _cv2
    import torch
    import torch.nn.functional as F

    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
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
            padding_mode="zeros",
            align_corners=True,
        )
        return crop.squeeze(0)  # (C, canvas_h, canvas_w)


def gpu_canonical_crop_batch(
    frame_chw: "torch.Tensor",
    M_aligns: list,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    *,
    geometry: Optional[CanonicalGeometry] = None,
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
        :func:`~hydra_suite.core.canonicalization.geometry.canonical_affine`, one per detection.
    canvas_w, canvas_h:
        Output canvas dimensions (same for every crop). Either these or
        ``geometry`` must be given; if both are given they must agree.
    geometry:
        Layer 1 :class:`CanonicalGeometry`, alternative to ``canvas_w``/``canvas_h``.

    Returns
    -------
    torch.Tensor
        ``(N, C, canvas_h, canvas_w)`` float32 on the same device as
        ``frame_chw``.
    """
    import torch
    import torch.nn.functional as F

    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
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
            padding_mode="zeros",
            align_corners=True,
        )  # (N, C, canvas_h, canvas_w)
        return crops


def apply_headtail_rotation(
    crop: np.ndarray,
    M_align: np.ndarray,
    direction: str,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    treat_updown_as_unknown: bool = True,
    *,
    geometry: Optional[CanonicalGeometry] = None,
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
        geometry: Layer 1 :class:`CanonicalGeometry`, alternative to
            ``canvas_w``/``canvas_h``. Either these or ``geometry`` must be
            given; if both are given they must agree.

    Returns:
        (rotated_crop, M_canonical, M_inverse, orientation_offset_rad)
    """
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
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
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    padding_fraction: Optional[float] = None,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    suppress_foreign: bool = True,
    per_frame_all_corners: Optional[List[List[np.ndarray]]] = None,
    *,
    geometry: Optional[CanonicalGeometry] = None,
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
        padding_fraction: OBB expansion factor. Ignored when ``geometry`` is
            given — the geometry's own ``margin`` is used instead so the
            transform stays rigid (Layer 1 contract).
        bg_color: Background fill colour.
        suppress_foreign: Whether to mask foreign OBB regions.
        per_frame_all_corners: Per-frame list of *all* OBB corners for
            foreign-OBB masking.  If None, ``per_frame_corners`` is used.
        geometry: Layer 1 :class:`CanonicalGeometry`, alternative to
            ``canvas_w``/``canvas_h``. Either these or ``geometry`` must be
            given; if both are given they must agree.

    Returns:
        Nested list ``[frame][detection]`` of ``CanonicalCropResult | None``.
    """
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
    # A geometry already carries the margin. Accepting a padding_fraction
    # alongside it would let a caller believe they had set a padding that the
    # geometry path silently ignores -- the same silent-mismatch class this
    # module exists to remove, so it is an error rather than a preference.
    if geometry is not None:
        implied = geometry.margin - 1.0
        if padding_fraction is not None and abs(padding_fraction - implied) > 1e-9:
            raise ValueError(
                f"padding_fraction={padding_fraction} disagrees with "
                f"geometry.margin={geometry.margin} (implies {implied}). "
                "Pass the geometry alone."
            )
        padding_fraction = implied
    elif padding_fraction is None:
        padding_fraction = 0.1

    results: List[List[Optional[CanonicalCropResult]]] = []

    for fi, frame in enumerate(frames):
        corners_list = per_frame_corners[fi]
        all_corners = (
            per_frame_all_corners[fi] if per_frame_all_corners else corners_list
        )
        frame_results: List[Optional[CanonicalCropResult]] = []

        # One code path: a caller that passed bare canvas dimensions gets a
        # geometry synthesised from them, rather than a second affine builder.
        effective_geometry = geometry or CanonicalGeometry(
            canvas_wh=(int(canvas_w), int(canvas_h)),
            margin=1.0 + float(padding_fraction),
            aspect_ratio=max(1.0, float(canvas_w) / max(1.0, float(canvas_h))),
        )

        for di, corners in enumerate(corners_list):
            try:
                M_align, axis_theta, _clipped = canonical_affine(
                    corners, effective_geometry
                )
            except ValueError:
                frame_results.append(None)
                continue

            # Foreign corners: everything except current detection
            foreign = None
            if suppress_foreign and len(all_corners) > 1:
                foreign = [all_corners[j] for j in range(len(all_corners)) if j != di]

            crop = extract_canonical_crop(
                frame,
                M_align,
                canvas_w,
                canvas_h,
                bg_color,
                foreign,
                own_corners=corners,
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
    own_corners: Optional[np.ndarray] = None,
) -> None:
    """Fill foreign OBB regions with background colour in canonical space.

    Transforms each foreign OBB's corners into canonical space via M_align,
    then fills the polygon with *bg_color*.  Modifies *crop* in-place.

    When two detections' OBBs overlap (adjacent/touching animals), a foreign
    OBB's polygon can spill into the current detection's own OBB region. If
    *own_corners* is given, that overlap is excluded from the mask so the
    current animal's own body is never blanked out — only the parts of the
    foreign region outside the current detection's own OBB are filled.
    """
    M = np.asarray(M_align, dtype=np.float64)
    R = M[:, :2]  # (2, 2)
    t = M[:, 2:]  # (2, 1)

    def _to_canonical(corners: np.ndarray) -> np.ndarray:
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        return (R @ pts.T + t).T

    own_poly = None
    if own_corners is not None:
        own_poly = _to_canonical(own_corners).astype(np.int32).reshape(-1, 1, 2)

    h, w = crop.shape[:2]
    for corners in foreign_corners_list:
        poly = _to_canonical(corners).astype(np.int32).reshape(-1, 1, 2)
        if own_poly is None:
            cv2.fillPoly(crop, [poly], bg_color)
            continue

        # Exclude any overlap with the current detection's own OBB: rasterise
        # both polygons and only fill pixels claimed by the foreign OBB but
        # not by the current detection's own OBB.
        foreign_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(foreign_mask, [poly], 1)
        own_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(own_mask, [own_poly], 1)
        mask = (foreign_mask & ~own_mask).astype(bool)
        if mask.any():
            crop[mask] = bg_color


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
