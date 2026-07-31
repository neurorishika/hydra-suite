"""
Pure-function utilities for pose keypoint feature extraction.

Stateless functions for parsing keypoint groups, building detection-keypoint
maps from the pose cache, computing geometry (heading, body length, visibility),
normalizing keypoints, and loading pose context from parameters.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hydra_suite.core.identity.geometry import normalize_theta

logger = logging.getLogger(__name__)


def parse_pose_group_tokens(raw_spec) -> List:
    """Parse keypoint group spec from list/tuple/string into tokens.

    Returns a list of tokens that are either int (index) or str (name).
    """
    if raw_spec is None:
        return []
    if isinstance(raw_spec, str):
        raw_tokens = raw_spec.split(",")
    elif isinstance(raw_spec, (list, tuple)):
        raw_tokens = list(raw_spec)
    else:
        raw_tokens = [raw_spec]

    tokens = []
    for token in raw_tokens:
        t = str(token).strip()
        if not t:
            continue
        try:
            tokens.append(int(t))
        except Exception:
            tokens.append(t)
    return tokens


def resolve_pose_group_indices(raw_spec, keypoint_names: List[str]) -> List[int]:
    """Resolve keypoint group names/indices to a deduplicated index list.

    Tokens may be integer indices or string names (case-insensitive match).
    Indices outside the valid range or unknown names are silently skipped.
    """
    names = [str(v) for v in (keypoint_names or [])]
    tokens = parse_pose_group_tokens(raw_spec)
    if not tokens:
        return []

    lower_map = {name.lower(): idx for idx, name in enumerate(names)}
    indices = []
    seen = set()
    for tok in tokens:
        idx = None
        if isinstance(tok, int):
            if 0 <= tok < len(names):
                idx = int(tok)
        else:
            idx = lower_map.get(str(tok).strip().lower(), None)
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def _frame_dict_from_pose_cache(
    pose_cache,
    detection_cache,
    frame_idx: int,
) -> Optional[Dict]:
    """Adapter: return a frame dict shaped like the legacy IndividualPropertiesCache.

    Reads PoseResult from the new PoseCache and the matching detection_ids array
    from DetectionCache. Returns None if either cache has no entry for this frame.

    This adapter enables build_pose_detection_keypoint_map to work with both the
    legacy IndividualPropertiesCache (via get_frame()) and the new cache pair.
    """
    try:
        pose_result = pose_cache.read_frame(frame_idx)
        obb_result = detection_cache.read_frame(frame_idx)
    except Exception:
        return None
    if pose_result is None or obb_result is None:
        return None
    return {
        "detection_ids": obb_result.detection_ids,
        "pose_keypoints": pose_result.keypoints,  # (D, K, 3)
        "pose_valid_mask": pose_result.valid_mask,  # (D,)
    }


def build_pose_detection_keypoint_map(
    pose_props_cache, frame_idx: int, detection_cache=None
) -> Dict[int, Any]:
    """Return {detection_id: keypoints_array} for one frame from cache.

    Supports two call patterns:
    - Legacy: ``build_pose_detection_keypoint_map(legacy_cache, frame_idx)``
      where legacy_cache has a ``.get_frame()`` method.
    - New pipeline: ``build_pose_detection_keypoint_map(pose_cache, frame_idx,
      detection_cache=det_cache)`` where both are new-style cache handles.

    Returns an empty dict if the cache is None or the frame is not found.
    """
    if pose_props_cache is None:
        return {}

    # New pipeline path: pose_cache + detection_cache pair
    if detection_cache is not None:
        frame = _frame_dict_from_pose_cache(
            pose_props_cache, detection_cache, frame_idx
        )
        if frame is None:
            return {}
    else:
        # Legacy path: IndividualPropertiesCache with get_frame()
        try:
            frame = pose_props_cache.get_frame(int(frame_idx))
        except Exception:
            return {}

    ids = frame.get("detection_ids", [])
    keypoints = frame.get("pose_keypoints", [])
    n = min(len(ids), len(keypoints))
    out = {}
    for i in range(n):
        try:
            det_id = int(ids[i])
        except Exception:
            continue
        out[det_id] = keypoints[i]
    return out


def _weighted_centroid(
    arr: np.ndarray, indices: List, ignore_set: set, min_valid_conf: float
) -> Optional[Tuple[float, float]]:
    """Compute confidence-weighted centroid of keypoints at given indices."""
    pts = []
    weights = []
    for idx in indices:
        if idx in ignore_set or idx < 0 or idx >= len(arr):
            continue
        x, y, conf = arr[idx]
        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or not np.isfinite(conf)
            or float(conf) < float(min_valid_conf)
        ):
            continue
        pts.append((float(x), float(y)))
        weights.append(max(1e-6, float(conf)))
    if not pts:
        return None
    pts_arr = np.asarray(pts, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    cx = float(np.average(pts_arr[:, 0], weights=w_arr))
    cy = float(np.average(pts_arr[:, 1], weights=w_arr))
    return cx, cy


def _keypoint_visibility(
    arr: np.ndarray, ignore_set: set, min_valid_conf: float
) -> float:
    """Compute fraction of visible (non-ignored, above-confidence) keypoints."""
    valid_total = 0
    visible_total = 0
    for idx in range(len(arr)):
        if idx in ignore_set:
            continue
        valid_total += 1
        conf = arr[idx, 2]
        if np.isfinite(conf) and float(conf) >= float(min_valid_conf):
            visible_total += 1
    return float(visible_total) / float(valid_total) if valid_total > 0 else 0.0


def compute_pose_geometry_from_keypoints(
    keypoints,
    anterior_indices,
    posterior_indices,
    min_valid_conf,
    ignore_indices=None,
) -> Optional[Dict[str, Any]]:
    """Extract heading, body length, and visibility from pose keypoints.

    Returns a dict with keys ``"heading"`` (float or None), ``"body_length"``
    (float or None), and ``"visibility"`` (float in [0, 1]), or None if the
    input keypoints array is invalid.
    """
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return None

    ignore_set = {int(idx) for idx in (ignore_indices or [])}
    visibility = _keypoint_visibility(arr, ignore_set, min_valid_conf)

    ant = _weighted_centroid(arr, anterior_indices, ignore_set, min_valid_conf)
    post = _weighted_centroid(arr, posterior_indices, ignore_set, min_valid_conf)
    if ant is None or post is None:
        return {
            "heading": None,
            "body_length": None,
            "visibility": float(np.clip(visibility, 0.0, 1.0)),
        }

    dx = ant[0] - post[0]
    dy = ant[1] - post[1]
    if not np.isfinite(dx) or not np.isfinite(dy):
        heading = None
        body_length = None
    else:
        heading = normalize_theta(math.atan2(dy, dx))
        body_length = float(math.hypot(dx, dy))

    return {
        "heading": heading,
        "body_length": body_length,
        "visibility": float(np.clip(visibility, 0.0, 1.0)),
    }


def normalize_pose_keypoints(
    keypoints, min_valid_conf: float, ignore_indices=None
) -> Optional[np.ndarray]:
    """Center and scale pose keypoints for shape comparison.

    Returns an (N, 3) float32 array with nan-filled invalid entries, or None
    if there are no valid keypoints above *min_valid_conf*.
    """
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return None

    ignore_set = {int(idx) for idx in (ignore_indices or [])}
    valid = np.zeros(len(arr), dtype=bool)
    valid_points = []
    valid_weights = []
    for idx in range(len(arr)):
        if idx in ignore_set:
            continue
        x, y, conf = arr[idx]
        if (
            np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(conf)
            and float(conf) >= float(min_valid_conf)
        ):
            valid[idx] = True
            valid_points.append((float(x), float(y)))
            valid_weights.append(max(1e-6, float(conf)))

    if not valid_points:
        return None

    pts_arr = np.asarray(valid_points, dtype=np.float64)
    w_arr = np.asarray(valid_weights, dtype=np.float64)
    cx = float(np.average(pts_arr[:, 0], weights=w_arr))
    cy = float(np.average(pts_arr[:, 1], weights=w_arr))
    centered = pts_arr - np.array([[cx, cy]], dtype=np.float64)
    radii = np.sqrt(np.sum(centered**2, axis=1))
    scale = float(np.median(radii[radii > 1e-6])) if np.any(radii > 1e-6) else 1.0
    scale = max(scale, 1.0)

    out = np.full((len(arr), 3), np.nan, dtype=np.float32)
    out[:, 2] = 0.0
    valid_indices = np.where(valid)[0]
    for src_idx, kp_idx in enumerate(valid_indices):
        out[kp_idx, 0] = np.float32(centered[src_idx, 0] / scale)
        out[kp_idx, 1] = np.float32(centered[src_idx, 1] / scale)
        out[kp_idx, 2] = np.float32(arr[kp_idx, 2])
    return out


def count_valid_normalized_pose_keypoints(keypoints) -> int:
    """Count normalized pose keypoints with finite x/y coordinates."""
    if keypoints is None:
        return 0
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return 0
    valid_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    return int(np.sum(valid_mask))


def is_pose_heading_reliable(
    normalized_keypoints,
    visibility: float,
    min_visibility: float = 0.6,
    min_valid_keypoints: int = 3,
) -> bool:
    """Return True when pose heading is reliable enough to override orientation."""
    if not np.isfinite(visibility) or float(visibility) < float(min_visibility):
        return False
    return count_valid_normalized_pose_keypoints(normalized_keypoints) >= max(
        1, int(min_valid_keypoints)
    )


def load_pose_context_from_params(
    params: Dict[str, Any],
) -> Tuple[Any, List[int], List[int], List[int], bool]:
    """Open the pose properties cache and resolve keypoint group indices.

    Returns a 5-tuple:
        (pose_props_cache | None,
         anterior_indices,
         posterior_indices,
         ignore_indices,
         pose_direction_enabled)

    ``pose_direction_enabled`` is True only when both anterior and posterior
    index lists are non-empty.
    """
    pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
    cache_path = str(params.get("INDIVIDUAL_PROPERTIES_CACHE_PATH", "") or "").strip()
    if not pose_enabled or not cache_path or not os.path.exists(cache_path):
        return None, [], [], [], False

    from hydra_suite.core.identity.properties.cache import IndividualPropertiesCache

    pose_props_cache = IndividualPropertiesCache(cache_path, mode="r")
    if not pose_props_cache.is_compatible():
        logger.warning(
            "Pose cache incompatible, pose direction disabled: %s", cache_path
        )
        pose_props_cache.close()
        return None, [], [], [], False

    names_raw = pose_props_cache.metadata.get("pose_keypoint_names", [])
    keypoint_names = (
        [str(v) for v in names_raw] if isinstance(names_raw, (list, tuple)) else []
    )

    ignore_indices = resolve_pose_group_indices(
        params.get("POSE_IGNORE_KEYPOINTS", []), keypoint_names
    )
    anterior_indices = resolve_pose_group_indices(
        params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), keypoint_names
    )
    posterior_indices = resolve_pose_group_indices(
        params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), keypoint_names
    )

    pose_direction_enabled = bool(anterior_indices and posterior_indices)
    if not pose_direction_enabled:
        logger.info(
            "Pose direction disabled: define both anterior/posterior keypoint groups."
        )
    return (
        pose_props_cache,
        anterior_indices,
        posterior_indices,
        ignore_indices,
        pose_direction_enabled,
    )


def compute_detection_pose_features(
    detection_ids,
    pose_keypoint_map,
    anterior_indices,
    posterior_indices,
    ignore_indices,
    min_valid_conf,
    return_headings: bool = False,
) -> tuple:
    """Compute normalized pose keypoints and visibility for each detection.

    For each detection ID, look up pose keypoints in *pose_keypoint_map* and
    compute a normalized keypoint array plus a visibility score.

    Args:
        return_headings: when True, return a 3-tuple with an extra
            ``list[float | None]`` of per-detection pose headings.

    Returns (return_headings=False):
        detection_pose_keypoints : list[ndarray | None], length == len(detection_ids)
        detection_pose_visibility : float32 ndarray, length == len(detection_ids)

    Returns (return_headings=True):
        detection_pose_keypoints, detection_pose_visibility,
        detection_pose_headings : list[float | None]
    """
    n = len(detection_ids)
    detection_pose_keypoints = [None] * n
    detection_pose_visibility = np.zeros(n, dtype=np.float32)
    detection_pose_headings: List[Optional[float]] = [None] * n

    for det_idx in range(n):
        try:
            det_id = int(detection_ids[det_idx])
        except Exception:
            continue
        keypoints = pose_keypoint_map.get(det_id)
        features = compute_pose_geometry_from_keypoints(
            keypoints,
            anterior_indices,
            posterior_indices,
            min_valid_conf,
            ignore_indices,
        )
        if features is None:
            continue
        detection_pose_visibility[det_idx] = float(
            features.get("visibility", 0.0) or 0.0
        )
        detection_pose_keypoints[det_idx] = normalize_pose_keypoints(
            keypoints, min_valid_conf, ignore_indices=ignore_indices
        )
        detection_pose_headings[det_idx] = features.get("heading")
    if return_headings:
        return (
            detection_pose_keypoints,
            detection_pose_visibility,
            detection_pose_headings,
        )
    return detection_pose_keypoints, detection_pose_visibility
