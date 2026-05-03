"""
Utility functions for geometry operations in the HYDRA Suite.
"""

import math

import numpy as np


def fit_circle_to_points(points: object) -> object:
    """
    Fit a circle to a set of points using least squares optimization.

    Uses algebraic circle fitting method for robust estimation from 3+ points.

    Args:
        points (list): List of (x, y) coordinate tuples

    Returns:
        tuple: (center_x, center_y, radius) or None if fitting fails
    """
    if len(points) < 3:
        return None

    try:
        # Convert to numpy array
        points = np.array(points, dtype=np.float64)
        x, y = points[:, 0], points[:, 1]

        # Set up system of equations for circle fitting
        # Circle equation: (x-a)² + (y-b)² = r²
        # Expanded: x² + y² - 2ax - 2by + (a² + b² - r²) = 0
        # Linear form: x² + y² = 2ax + 2by - (a² + b² - r²)

        A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        b = x**2 + y**2

        # Solve using least squares
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        center_x, center_y, c = params

        # Calculate radius from fitted parameters
        radius = np.sqrt(center_x**2 + center_y**2 + c)

        # Validate the result
        if radius > 0 and not np.isnan(radius):
            return (float(center_x), float(center_y), float(radius))
        else:
            return None

    except (np.linalg.LinAlgError, ValueError):
        return None


def wrap_angle_degs(deg: float) -> float:
    """
    Normalize angle to [-180, 180] degree range.

    This function is crucial for orientation tracking to ensure smooth
    angle transitions and prevent discontinuities at the 0/360 boundary.

    Args:
        deg (float): Input angle in degrees

    Returns:
        float: Normalized angle in range [-180, 180] degrees
    """
    deg %= 360
    return deg - 360 if deg >= 180 else deg


def estimate_detection_crop_quality(shape, reference_body_size):
    """Estimate crop quality from detection geometry.

    Returns a float in [0, 1] measuring how well the detection's minor axis
    matches the reference body size.
    """
    try:
        area = float(shape[0])
        aspect = float(shape[1])
    except Exception:
        return 0.0
    if not np.isfinite(area) or area <= 0:
        return 0.0
    aspect = max(1e-3, float(abs(aspect)))
    minor = math.sqrt(max(1e-6, (4.0 * area) / (math.pi * max(aspect, 1e-3))))
    ref = max(1.0, float(reference_body_size) * 0.75)
    return float(np.clip(minor / ref, 0.0, 1.0))


# ---------------------------------------------------------------------------
# HYDRA suite crop contamination utilities
# ---------------------------------------------------------------------------


def apply_foreign_obb_mask(
    crop: np.ndarray,
    x_offset: int,
    y_offset: int,
    other_corners_list,
    background_color=128,
) -> np.ndarray:
    """Fill pixels in *crop* that belong to other animals' OBB regions.

    Shifts each foreign OBB from frame coordinates into crop-local coordinates
    and fills the polygon with *background_color* using ``cv2.fillPoly``.

    Args:
        crop: BGR (or grayscale) image crop extracted from the full frame.
        x_offset: Horizontal offset of the crop's top-left corner in frame coords.
        y_offset: Vertical offset of the crop's top-left corner in frame coords.
        other_corners_list: Sequence of (4, 2) float32 arrays of OBB corners in
            *frame* coordinates for every other detected animal.
        background_color: Fill value — either a scalar (0–255) applied to all
            channels, or a (B, G, R) tuple for colour crops.

    Returns:
        Modified copy of *crop* with foreign-animal regions filled.
    """
    if crop is None or not other_corners_list:
        return crop

    import cv2 as _cv2

    out = crop.copy()
    crop_h, crop_w = out.shape[:2]

    # Resolve fill colour — accept scalar int or (B, G, R) tuple
    if isinstance(background_color, (list, tuple)) and len(background_color) == 3:
        bgr = tuple(int(np.clip(c, 0, 255)) for c in background_color)
        fill_color = bgr if out.ndim == 3 else int(bgr[0])
    else:
        fill = int(np.clip(background_color, 0, 255))
        fill_color = (fill, fill, fill) if out.ndim == 3 else fill

    for corners in other_corners_list:
        try:
            arr = np.asarray(corners, dtype=np.float32)
            if arr.shape != (4, 2):
                continue
            local = arr.copy()
            local[:, 0] -= float(x_offset)
            local[:, 1] -= float(y_offset)
            local[:, 0] = np.clip(local[:, 0], 0, crop_w - 1)
            local[:, 1] = np.clip(local[:, 1], 0, crop_h - 1)
            poly = local.astype(np.int32)
            _cv2.fillPoly(out, [poly], fill_color)
        except Exception:
            continue

    return out


def _zero_keypoints_inside_poly(
    arr: np.ndarray,
    poly: np.ndarray,
) -> None:
    """Set confidence to zero for keypoints inside a polygon (in-place)."""
    import cv2 as _cv2

    for k in range(len(arr)):
        if arr[k, 2] <= 0.0:
            continue
        x, y = float(arr[k, 0]), float(arr[k, 1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        dist = _cv2.pointPolygonTest(poly, (x, y), measureDist=False)
        if dist >= 0.0:
            arr[k, 2] = 0.0


def filter_keypoints_by_foreign_obbs(
    keypoints,
    all_corners_list,
    target_idx: int,
) -> np.ndarray:
    """Zero confidence of keypoints that fall inside another animal's OBB.

    Operates on *global frame coordinates* (after crop back-projection).

    Args:
        keypoints: [K, 3] float32 array of (x, y, conf) in frame coordinates.
        all_corners_list: List of (4, 2) float32 OBB corner arrays for every
            detection in the frame (including the target).
        target_idx: Index into *all_corners_list* identifying the current
            animal — its own OBB is skipped.

    Returns:
        Modified copy of *keypoints* with contaminated entries having conf=0.
        X/Y coordinates are preserved.
    """
    if keypoints is None:
        return keypoints

    arr = np.asarray(keypoints, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return arr
    if not all_corners_list:
        return arr

    for j, corners in enumerate(all_corners_list):
        if j == target_idx:
            continue
        try:
            poly = np.asarray(corners, dtype=np.float32)
            if poly.shape != (4, 2):
                continue
            _zero_keypoints_inside_poly(arr, poly)
        except Exception:
            continue

    return arr


# ---------------------------------------------------------------------------
# OBB rotation helpers (shared across data and detectkit layers)
# ---------------------------------------------------------------------------


def clamp01(value: float) -> float:
    """Clamp a scalar to the inclusive range [0, 1]."""
    return float(max(0.0, min(1.0, value)))


def obb_corners_from_dims(cx, cy, w, h, theta) -> np.ndarray:
    """Return pixel-space oriented-box corners (4x2 array) for a detection.

    Corners are ordered top-left, top-right, bottom-right, bottom-left in the
    rotated frame.
    """
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    half_w = float(w) / 2.0
    half_h = float(h) / 2.0
    local = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    rotation = np.array(
        [[cos_t, -sin_t], [sin_t, cos_t]],
        dtype=np.float32,
    )
    return local @ rotation.T + np.array([float(cx), float(cy)], dtype=np.float32)


def polygon_overlap_ratio(corners_a, corners_b) -> float:
    """Intersection area divided by the smaller polygon area, clipped to [0, 1].

    Both inputs are interpreted as 4x2 (or Nx2) corner arrays in pixel space.
    Returns 0.0 when either polygon is degenerate or non-overlapping.
    """
    import cv2

    poly_a = np.asarray(corners_a, dtype=np.float32).reshape(-1, 1, 2)
    poly_b = np.asarray(corners_b, dtype=np.float32).reshape(-1, 1, 2)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    except cv2.error:
        return 0.0
    if inter <= 0.0:
        return 0.0
    return clamp01(float(inter) / max(min(area_a, area_b), 1e-6))
