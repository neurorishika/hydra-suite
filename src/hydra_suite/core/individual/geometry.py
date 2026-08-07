"""Shared geometry utilities for the identity module.

Includes angle normalization, OBB axis disambiguation, and directed
heading resolution helpers used by both tracking and identity pipelines.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Angle / theta helpers
# ---------------------------------------------------------------------------


def normalize_theta(theta: float) -> float:
    """Normalize radians to [0, 2*pi)."""
    try:
        value = float(theta)
    except Exception:
        value = 0.0
    return value % (2 * math.pi)


def circular_abs_diff_rad(a: float, b: float) -> float:
    """Absolute circular difference between two angles in radians."""
    d = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def collapse_obb_axis_theta(theta_axis: float, reference_theta) -> float:
    """Resolve 180-degree OBB axis ambiguity.

    Picks *theta_axis* or *theta_axis + pi* — whichever is angularly closer to
    *reference_theta*.  Returns ``normalize_theta(theta_axis)`` when
    *reference_theta* is None or non-finite.
    """
    theta0 = normalize_theta(theta_axis)
    theta1 = normalize_theta(theta0 + math.pi)
    if reference_theta is None:
        return theta0
    try:
        ref = float(reference_theta)
        if not math.isfinite(ref):
            return theta0
        ref = normalize_theta(ref)
    except Exception:
        return theta0
    d0 = circular_abs_diff_rad(theta0, ref)
    d1 = circular_abs_diff_rad(theta1, ref)
    return theta0 if d0 <= d1 else theta1


def select_directed_heading(
    pose_heading,
    pose_directed: bool,
    headtail_heading,
    headtail_directed: bool,
    pose_overrides_headtail: bool = True,
) -> Tuple[float, bool]:
    """Choose directed heading source (pose / head-tail) by precedence.

    Returns ``(heading_radians, is_directed)``.
    """
    try:
        pose_valid = bool(pose_directed) and math.isfinite(float(pose_heading))
    except Exception:
        pose_valid = False
    try:
        headtail_valid = bool(headtail_directed) and math.isfinite(
            float(headtail_heading)
        )
    except Exception:
        headtail_valid = False
    if pose_overrides_headtail:
        if pose_valid:
            return float(pose_heading), True
        if headtail_valid:
            return float(headtail_heading), True
        return float("nan"), False
    if headtail_valid:
        return float(headtail_heading), True
    if pose_valid:
        return float(pose_heading), True
    return float("nan"), False


def _safe_float_at(arr, idx: int) -> float:
    """Safely extract a float from an array-like at the given index."""
    try:
        if arr is not None and idx < len(arr):
            return float(arr[idx])
    except Exception:
        pass
    return math.nan


def _safe_bool_at(arr, idx: int) -> bool:
    """Safely extract a boolean from an array-like at the given index."""
    try:
        return bool(arr is not None and idx < len(arr) and arr[idx])
    except Exception:
        return False


def build_detection_direction_overrides(
    n_detections: int,
    pose_headings,
    pose_directed_mask,
    headtail_headings,
    headtail_directed_mask,
    pose_overrides_headtail: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-detection directed heading overrides and validity mask."""
    try:
        count = max(int(n_detections), 0)
    except Exception:
        count = 0

    detection_directed_heading = np.full(count, np.nan, dtype=np.float32)
    detection_directed_mask = np.zeros(count, dtype=np.uint8)

    for det_idx in range(count):
        selected_heading, is_directed = select_directed_heading(
            pose_heading=_safe_float_at(pose_headings, det_idx),
            pose_directed=_safe_bool_at(pose_directed_mask, det_idx),
            headtail_heading=_safe_float_at(headtail_headings, det_idx),
            headtail_directed=_safe_bool_at(headtail_directed_mask, det_idx),
            pose_overrides_headtail=pose_overrides_headtail,
        )
        if is_directed:
            detection_directed_heading[det_idx] = np.float32(selected_heading)
            detection_directed_mask[det_idx] = 1

    return detection_directed_heading, detection_directed_mask


def resolve_tracking_theta(
    track_idx: int,
    measured_theta: float,
    pose_directed: bool,
    orientation_last,
    fallback_theta=None,
) -> float:
    """Resolve directed vs axis-aligned orientation for one track.

    *orientation_last* is a list indexed by track index whose entries are
    the last committed theta (float) or None.
    """
    if pose_directed:
        return normalize_theta(measured_theta)
    reference_theta = None
    if orientation_last is not None:
        try:
            if 0 <= int(track_idx) < len(orientation_last):
                reference_theta = orientation_last[int(track_idx)]
        except Exception:
            pass
    if reference_theta is None and fallback_theta is not None:
        try:
            candidate = float(fallback_theta)
            if math.isfinite(candidate):
                reference_theta = candidate
        except Exception:
            pass
    return collapse_obb_axis_theta(measured_theta, reference_theta)


def resolve_detection_tracking_theta(
    track_idx: int,
    measured_theta: float,
    directed_heading,
    pose_directed: bool,
    orientation_last,
    fallback_theta=None,
) -> float:
    """Resolve tracking theta, preferring selected directed headings when valid."""
    tracking_theta = measured_theta
    if pose_directed:
        try:
            candidate = float(directed_heading)
            if math.isfinite(candidate):
                tracking_theta = candidate
        except Exception:
            pass
    return resolve_tracking_theta(
        track_idx,
        tracking_theta,
        pose_directed,
        orientation_last,
        fallback_theta=fallback_theta,
    )


# ---------------------------------------------------------------------------
# OBB / ellipse geometry
# ---------------------------------------------------------------------------


def resolve_directed_angle(
    theta: float,
    heading_hint: Optional[float] = None,
    heading_directed: bool = False,
    vx: Optional[float] = None,
    vy: Optional[float] = None,
) -> Tuple[float, bool, str]:
    """Resolve the best directed orientation angle for a crop.

    Priority:
    1. Head-tail model heading (heading_directed=True and finite heading_hint).
    2. Motion velocity (vx, vy non-negligible) — disambiguates theta ± π.
    3. OBB axis angle (undirected, 180° ambiguity retained).

    The returned angle points tail → head so that the affine-warp canonicalization
    places the head on the right side (+x) of the canonical crop.

    Returns:
        (angle_rad, is_directed, source_str)
    """
    # Priority 1: head-tail model
    if (
        heading_directed
        and heading_hint is not None
        and math.isfinite(float(heading_hint))
    ):
        return float(heading_hint) % (2.0 * math.pi), True, "head_tail_model"

    # Priority 2: motion velocity disambiguates OBB axis
    if vx is not None and vy is not None:
        fvx, fvy = float(vx), float(vy)
        if math.isfinite(fvx) and math.isfinite(fvy) and (fvx * fvx + fvy * fvy) > 1e-6:
            motion_angle = math.atan2(fvy, fvx) % (2.0 * math.pi)
            theta0 = float(theta) % (2.0 * math.pi)
            theta1 = (theta0 + math.pi) % (2.0 * math.pi)
            diff0 = abs(((motion_angle - theta0 + math.pi) % (2.0 * math.pi)) - math.pi)
            diff1 = abs(((motion_angle - theta1 + math.pi) % (2.0 * math.pi)) - math.pi)
            resolved = theta0 if diff0 <= diff1 else theta1
            return resolved, True, "motion_velocity"

    # Priority 3: undirected OBB axis (180° ambiguous)
    return float(theta) % (2.0 * math.pi), False, "tracking_theta"


def ellipse_to_obb_corners(
    cx: float, cy: float, major_axis: float, minor_axis: float, theta: float
) -> np.ndarray:
    """Convert ellipse parameters to OBB corner points.

    The OBB is the oriented bounding box that exactly fits the ellipse,
    which is a rotated rectangle with dimensions (major_axis x minor_axis).

    Args:
        cx, cy: Center coordinates of the ellipse
        major_axis: Full length of the major axis (not semi-axis)
        minor_axis: Full length of the minor axis (not semi-axis)
        theta: Rotation angle in radians (orientation of major axis)

    Returns:
        corners: numpy array of shape (4, 2) with corner coordinates
    """
    a = major_axis / 2.0
    b = minor_axis / 2.0
    local_corners = np.array([[a, b], [-a, b], [-a, -b], [a, -b]], dtype=np.float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return (local_corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(
        np.float32
    )


def ellipse_axes_from_area(area: float, aspect_ratio: float) -> Tuple[float, float]:
    """Compute ellipse major/minor axes from area and aspect ratio.

    Returns:
        (major_axis, minor_axis)
    """
    minor = math.sqrt(4.0 * float(area) / (math.pi * float(aspect_ratio)))
    major = float(aspect_ratio) * minor
    return major, minor
