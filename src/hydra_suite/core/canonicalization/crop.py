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
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.canonicalization.resample import (
    canonical_warp,
    canonical_warp_batch,
    canonical_warp_batch_from_frame,
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


def _geometry_from_canvas(canvas_w: int, canvas_h: int) -> CanonicalGeometry:
    """Synthesise a :class:`CanonicalGeometry` from bare canvas dimensions.

    ``margin``/``aspect_ratio`` are irrelevant to the warp seam (only
    ``canvas_w``/``canvas_h`` are consumed), so this exists purely to give
    callers of the legacy ``(canvas_w, canvas_h)`` ints a single geometry
    object to hand to :func:`~hydra_suite.core.canonicalization.resample.
    canonical_warp`/``canonical_warp_batch``.
    """
    return CanonicalGeometry(
        canvas_wh=(int(canvas_w), int(canvas_h)),
        margin=1.0,
        aspect_ratio=max(1.0, float(canvas_w) / max(1.0, float(canvas_h))),
    )


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

    Delegates to the torch seam
    (:func:`~hydra_suite.core.canonicalization.resample.canonical_warp_batch_from_frame`),
    which slices this detection's canvas footprint out of the RAW frame and
    converts only that sub-region to a CHW float tensor, then warps via
    ``F.grid_sample`` and converts back to HWC uint8. Converting the whole
    frame here was O(frame area) PER CROP -- and every caller
    (``extract_canonical_crops_batch``, the head-tail analyzer, the individual
    dataset generator) loops this over a frame's detections, so a 4512x4512
    frame was turned into a 244 MB float32 tensor once per animal. Slicing
    before an elementwise conversion is bit-for-bit what slicing after it
    produced. This replaces the former OpenCV affine-warp kernel; the
    foreign-mask call sequence below (still ``cv2.fillPoly``-based) is
    unchanged.
    """
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
    effective_geometry = geometry or _geometry_from_canvas(canvas_w, canvas_h)

    def _to_chw_float(sub: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(sub)).permute(2, 0, 1).float()

    crop_chw = canonical_warp_batch_from_frame(
        np.asarray(frame), [M_align], effective_geometry, _to_chw_float
    ).squeeze(0)
    crop = (
        crop_chw.round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
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

    Thin wrapper over the shared torch seam
    (:func:`~hydra_suite.core.canonicalization.resample.canonical_warp`) for
    frames already resident on a CUDA (or MPS) device.  ``M_align`` is the
    2x3 forward affine produced by
    :func:`~hydra_suite.core.canonicalization.geometry.canonical_affine`
    mapping frame pixel coords to canvas pixel coords.

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
        ``frame_chw``.  Carries no autograd graph (the seam runs under
        ``torch.inference_mode()``).
    """
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
    effective_geometry = geometry or _geometry_from_canvas(canvas_w, canvas_h)
    return canonical_warp(frame_chw, M_align, effective_geometry)


def gpu_canonical_crop_batch(
    frame_chw: "torch.Tensor",
    M_aligns: list,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
    *,
    geometry: Optional[CanonicalGeometry] = None,
) -> "torch.Tensor":
    """Batch version of :func:`gpu_canonical_crop` for N crops from *one* frame.

    Thin wrapper over
    :func:`~hydra_suite.core.canonicalization.resample.canonical_warp_batch`,
    which replaces N serial ``F.affine_grid`` + ``F.grid_sample`` calls with a
    single pair of batched calls.  This reduces GPU kernel launch overhead
    from O(N) to O(1) when extracting many detections from the same frame,
    which is the common case for dense multi-animal tracking (e.g. 50
    animals x 8 frames).

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
        ``frame_chw``.  Carries no autograd graph (the seam runs under
        ``torch.inference_mode()``).
    """
    canvas_w, canvas_h = _resolve_canvas(canvas_w, canvas_h, geometry)
    effective_geometry = geometry or _geometry_from_canvas(canvas_w, canvas_h)
    return canonical_warp_batch(frame_chw, list(M_aligns), effective_geometry)


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

    results: List[List[Optional[CanonicalCropResult]]] = []

    for fi, frame in enumerate(frames):
        corners_list = per_frame_corners[fi]
        all_corners = (
            per_frame_all_corners[fi] if per_frame_all_corners else corners_list
        )
        frame_results: List[Optional[CanonicalCropResult]] = []

        # One code path: a caller that passed bare canvas dimensions gets a
        # geometry synthesised from them. There is no separate padding knob --
        # the canvas IS the framing (spec 2026-08-18).
        effective_geometry = geometry or CanonicalGeometry(
            canvas_wh=(int(canvas_w), int(canvas_h)),
            margin=1.0,
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
