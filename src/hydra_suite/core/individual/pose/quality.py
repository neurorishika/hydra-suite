"""
Pose quality assessment and post-processing for the HYDRA suite.

All functions are pure (stateless) and can be imported from both the GUI
and any script-based pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hydra_suite.core.individual.pose.features import (
    compute_pose_geometry_from_keypoints,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PoseQualityResult:
    """Result of a single-row pose quality assessment."""

    cleaned_keypoints: np.ndarray  # [K, 3] float32; bad kpts have conf=0, X/Y kept
    valid_mask: np.ndarray  # [K] bool
    quality_score: float  # [0, 1]
    quality_state: str  # "good" | "partial" | "bad" | "rejected"
    quality_flags: List[str]  # e.g. ["low_conf:3", "too_few_valid"]
    was_cleaned: bool


@dataclass
class BodyLengthPrior:
    """Statistics-based body-length calibration prior."""

    median_px: float
    mad_px: float  # median absolute deviation
    n_samples: int
    is_valid: bool  # True only when n_samples >= 20


@dataclass
class EdgeLengthPriors:
    """Per-skeleton-edge length calibration priors.

    ``priors`` maps a canonical edge key ``(min_idx, max_idx)`` to a dict with
    keys ``median_px``, ``mad_px``, and ``n_samples``.  ``is_valid`` is True
    when at least one edge has been calibrated with >= 20 samples.
    """

    priors: Dict[Tuple[int, int], Dict]
    is_valid: bool


# ---------------------------------------------------------------------------
# assess_pose_row
# ---------------------------------------------------------------------------


def _rejected_result(n_kpts: int, flags: List[str]) -> PoseQualityResult:
    kpts_out = (
        np.zeros((n_kpts, 3), dtype=np.float32)
        if n_kpts > 0
        else np.zeros((0, 3), dtype=np.float32)
    )
    mask = np.zeros(n_kpts, dtype=bool)
    return PoseQualityResult(
        cleaned_keypoints=kpts_out,
        valid_mask=mask,
        quality_score=0.0,
        quality_state="rejected",
        quality_flags=flags,
        was_cleaned=True,
    )


def _clean_keypoints(
    arr: np.ndarray,
    min_valid_conf: float,
    ignore_set: set,
) -> Tuple[np.ndarray, np.ndarray, bool, List[str]]:
    """Clean per-keypoint confidence and return (cleaned, valid_mask, was_cleaned, flags)."""
    K = len(arr)
    cleaned = arr.copy()
    valid_mask = np.zeros(K, dtype=bool)
    was_cleaned = False
    low_conf_count = 0
    invalid_coords_count = 0

    for i in range(K):
        if i in ignore_set:
            continue
        x, y, conf = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])
        coords_ok = np.isfinite(x) and np.isfinite(y)
        conf_ok = np.isfinite(conf) and conf >= float(min_valid_conf)

        if not conf_ok:
            cleaned[i, 2] = 0.0
            was_cleaned = True
            low_conf_count += 1
        elif not coords_ok:
            cleaned[i, 2] = 0.0
            was_cleaned = True
            invalid_coords_count += 1
        else:
            valid_mask[i] = True

    flags: List[str] = []
    if low_conf_count > 0:
        flags.append(f"low_conf:{low_conf_count}")
    if invalid_coords_count > 0:
        flags.append(f"invalid_coords:{invalid_coords_count}")

    return cleaned, valid_mask, was_cleaned, flags


def _check_body_length_outlier(
    arr: np.ndarray,
    body_length_prior: Optional[BodyLengthPrior],
    anterior_indices: Optional[List[int]],
    posterior_indices: Optional[List[int]],
    min_valid_conf: float,
    ignore_set: set,
    z_threshold: float,
) -> bool:
    """Return True if body length is a statistical outlier."""
    if (
        body_length_prior is None
        or not body_length_prior.is_valid
        or not anterior_indices
        or not posterior_indices
    ):
        return False
    geom = compute_pose_geometry_from_keypoints(
        arr,
        anterior_indices,
        posterior_indices,
        min_valid_conf,
        list(ignore_set) if ignore_set else None,
    )
    if geom is None or geom.get("body_length") is None:
        return False
    bl = float(geom["body_length"])
    denom = max(float(body_length_prior.mad_px), 1.0)
    z_score = abs(bl - float(body_length_prior.median_px)) / denom
    return z_score > float(z_threshold)


def _count_edge_outliers(
    cleaned: np.ndarray,
    valid_mask: np.ndarray,
    skeleton_edges: Optional[List[Tuple[int, int]]],
    edge_length_priors: Optional[EdgeLengthPriors],
    z_threshold: float,
) -> int:
    """Count skeleton edges whose length exceeds the calibrated prior."""
    if (
        not skeleton_edges
        or edge_length_priors is None
        or not edge_length_priors.is_valid
    ):
        return 0
    K = len(cleaned)
    n_outliers = 0
    for edge in skeleton_edges:
        try:
            ei, ej = int(edge[0]), int(edge[1])
        except Exception:
            continue
        if ei >= K or ej >= K:
            continue
        if not (valid_mask[ei] and valid_mask[ej]):
            continue
        key = (min(ei, ej), max(ei, ej))
        prior = edge_length_priors.priors.get(key)
        if prior is None or int(prior.get("n_samples", 0)) < 20:
            continue
        dx = float(cleaned[ei, 0]) - float(cleaned[ej, 0])
        dy = float(cleaned[ei, 1]) - float(cleaned[ej, 1])
        edge_len = math.sqrt(dx * dx + dy * dy)
        denom = max(float(prior.get("mad_px", 1.0)), 1.0)
        z = abs(edge_len - float(prior["median_px"])) / denom
        if z > float(z_threshold):
            n_outliers += 1
    return n_outliers


def _compute_quality(
    valid_fraction: float,
    cleaned: np.ndarray,
    valid_mask: np.ndarray,
    body_length_outlier: bool,
    n_edge_outliers: int,
) -> Tuple[float, str]:
    """Compute quality score and state from valid-fraction and outlier info."""
    valid_confs = [float(cleaned[i, 2]) for i in range(len(cleaned)) if valid_mask[i]]
    mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
    score = valid_fraction * mean_conf
    if body_length_outlier:
        score *= 0.7
    if n_edge_outliers > 0:
        score *= max(0.5, 1.0 - 0.15 * n_edge_outliers)
    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.2:
        state = "bad"
    elif score < 0.7:
        state = "partial"
    else:
        state = "good"
    return score, state


def assess_pose_row(
    keypoints: np.ndarray,
    min_valid_conf: float,
    min_valid_fraction: float,
    min_valid_keypoints: int,
    ignore_indices: Optional[List[int]] = None,
    body_length_prior: Optional[BodyLengthPrior] = None,
    anterior_indices: Optional[List[int]] = None,
    posterior_indices: Optional[List[int]] = None,
    body_length_z_threshold: float = 3.5,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    edge_length_priors: Optional[EdgeLengthPriors] = None,
    edge_length_z_threshold: float = 4.0,
) -> PoseQualityResult:
    """Assess pose quality for a single keypoint array.

    Cleans raw pose keypoints by zeroing confidence for bad keypoints while
    retaining X/Y coordinates as a best guess.  Never drops rows — rejected
    rows get all conf=0 plus quality flags.

    Args:
        keypoints: [K, 3] float32 array of (x, y, conf) per keypoint.
        min_valid_conf: Minimum confidence threshold to consider a keypoint valid.
        min_valid_fraction: Minimum fraction of (considered) keypoints that must
            be valid for the row to be accepted.
        min_valid_keypoints: Minimum absolute count of valid keypoints required.
        ignore_indices: Keypoint indices excluded from all quality computations.
        body_length_prior: Optional prior for body-length outlier detection.
        anterior_indices: Keypoint indices defining the anterior region.
        posterior_indices: Keypoint indices defining the posterior region.
        body_length_z_threshold: Z-score threshold above which body length is
            flagged as an outlier.
        skeleton_edges: List of (i, j) index pairs defining connected keypoints.
        edge_length_priors: Calibrated per-edge length priors for anatomy checks.
        edge_length_z_threshold: MAD-based z-score above which an edge length is
            flagged as an anatomical outlier.

    Returns:
        PoseQualityResult with cleaned keypoints and quality metadata.
    """
    ignore_set = {int(i) for i in (ignore_indices or [])}

    # 1. Validate input
    if keypoints is None:
        return _rejected_result(0, ["null_input"])
    try:
        arr = np.asarray(keypoints, dtype=np.float32)
    except Exception:
        return _rejected_result(0, ["invalid_input"])
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return _rejected_result(0, ["invalid_shape"])
    if not np.any(np.isfinite(arr)):
        return _rejected_result(len(arr), ["all_nan"])

    K = len(arr)

    # 2. Per-keypoint cleaning
    cleaned, valid_mask, was_cleaned, flags = _clean_keypoints(
        arr, min_valid_conf, ignore_set
    )

    # 3-4. Count valid keypoints (ignoring ignored indices in denominator)
    num_considered = sum(1 for i in range(K) if i not in ignore_set)
    num_valid = int(np.sum(valid_mask))
    valid_fraction = (
        float(num_valid) / float(num_considered) if num_considered > 0 else 0.0
    )

    # 5. Rejection check
    rejected = False
    if valid_fraction < float(min_valid_fraction) or num_valid < int(
        min_valid_keypoints
    ):
        rejected = True
        flags.append("too_few_valid")
        for i in range(K):
            if i not in ignore_set:
                cleaned[i, 2] = 0.0
        was_cleaned = True

    if rejected:
        return PoseQualityResult(
            cleaned_keypoints=cleaned,
            valid_mask=valid_mask,
            quality_score=0.0,
            quality_state="rejected",
            quality_flags=flags,
            was_cleaned=was_cleaned,
        )

    # 6. Body-length outlier check (on ORIGINAL keypoints before zeroing)
    body_length_outlier = _check_body_length_outlier(
        arr,
        body_length_prior,
        anterior_indices,
        posterior_indices,
        min_valid_conf,
        ignore_set,
        body_length_z_threshold,
    )
    if body_length_outlier:
        flags.append("body_length_outlier")

    # 6b. Per-edge skeleton length check
    n_edge_outliers = _count_edge_outliers(
        cleaned,
        valid_mask,
        skeleton_edges,
        edge_length_priors,
        edge_length_z_threshold,
    )
    if n_edge_outliers > 0:
        flags.append(f"edge_outlier:{n_edge_outliers}")

    # 7-8. Quality score and state
    quality_score, quality_state = _compute_quality(
        valid_fraction,
        cleaned,
        valid_mask,
        body_length_outlier,
        n_edge_outliers,
    )

    return PoseQualityResult(
        cleaned_keypoints=cleaned,
        valid_mask=valid_mask,
        quality_score=quality_score,
        quality_state=quality_state,
        quality_flags=flags,
        was_cleaned=was_cleaned,
    )


# ---------------------------------------------------------------------------
# calibrate_body_length_prior
# ---------------------------------------------------------------------------


def _filter_high_conf_rows(df: pd.DataFrame, high_conf_floor: float) -> pd.DataFrame:
    """Filter DataFrame to rows with PoseMeanConf >= threshold."""
    if "PoseMeanConf" not in df.columns:
        return df
    try:
        return df[df["PoseMeanConf"] >= float(high_conf_floor)]
    except Exception:
        return df


def _collect_body_lengths(
    high_conf_df: pd.DataFrame,
    pose_labels: List[str],
    anterior_indices: List[int],
    posterior_indices: List[int],
    min_valid_conf: float,
) -> List[float]:
    """Extract valid body lengths from high-confidence rows.

    Vectorized equivalent of the historical ``iterrows()`` loop that called
    ``_extract_keypoints_from_row`` + ``compute_pose_geometry_from_keypoints``
    per row. Produces the identical sample sequence (same rows, same order,
    same NaN/valid filtering) because ``calibrate_body_length_prior`` never
    passes ``ignore_indices`` -- the ignore set is always empty here, which
    is what makes the anterior/posterior weighted-centroid math reducible to
    a per-index accumulation across all rows at once.
    """
    if high_conf_df.empty:
        return []

    kpts, any_found = _stack_keypoints(high_conf_df, pose_labels)
    if not bool(any_found.any()):
        return []

    ax, ay, a_valid = _weighted_centroid_batch(kpts, anterior_indices, min_valid_conf)
    px, py, p_valid = _weighted_centroid_batch(kpts, posterior_indices, min_valid_conf)

    dx = ax - px
    dy = ay - py
    coords_finite = np.isfinite(dx) & np.isfinite(dy)

    # math.hypot(dx, dy) per row, matching np.hypot's elementwise semantics
    # exactly (both ultimately delegate to the platform libm hypot).
    body_length = np.hypot(dx, dy)

    mask = any_found & a_valid & p_valid & coords_finite & (body_length > 0.0)
    return [float(v) for v in body_length[mask]]


def _body_length_prior_from_samples(samples: List[float]) -> BodyLengthPrior:
    """Compute BodyLengthPrior from a list of body-length samples."""
    n = len(samples)
    if n == 0:
        return BodyLengthPrior(median_px=0.0, mad_px=0.0, n_samples=0, is_valid=False)
    arr = np.asarray(samples, dtype=np.float64)
    median_px = float(np.median(arr))
    mad_px = float(np.median(np.abs(arr - median_px)))
    return BodyLengthPrior(
        median_px=median_px,
        mad_px=mad_px,
        n_samples=n,
        is_valid=n >= 20,
    )


def calibrate_body_length_prior(
    df: pd.DataFrame,
    pose_labels: List[str],
    anterior_indices: List[int],
    posterior_indices: List[int],
    min_valid_conf: float,
    high_conf_floor: float = 0.7,
) -> BodyLengthPrior:
    """Estimate a body-length prior from high-confidence frames.

    Filters to rows where PoseMeanConf >= high_conf_floor and computes the
    median and MAD of the body length across those rows.

    Args:
        df: DataFrame with pose keypoint columns (PoseKpt_{label}_X/Y/Conf)
            and a PoseMeanConf summary column.
        pose_labels: Ordered list of keypoint label strings.
        anterior_indices: Indices of anterior keypoints (for body-length calc).
        posterior_indices: Indices of posterior keypoints.
        min_valid_conf: Minimum confidence for individual keypoints.
        high_conf_floor: Minimum PoseMeanConf to include a row in calibration.

    Returns:
        BodyLengthPrior; is_valid=True only when n_samples >= 20.
    """
    _invalid = BodyLengthPrior(median_px=0.0, mad_px=0.0, n_samples=0, is_valid=False)

    if df is None or df.empty or not anterior_indices or not posterior_indices:
        return _invalid
    if not pose_labels:
        return _invalid

    high_conf_df = _filter_high_conf_rows(df, high_conf_floor)
    if high_conf_df.empty:
        return _invalid

    body_lengths = _collect_body_lengths(
        high_conf_df,
        pose_labels,
        anterior_indices,
        posterior_indices,
        min_valid_conf,
    )
    return _body_length_prior_from_samples(body_lengths)


# ---------------------------------------------------------------------------
# calibrate_edge_length_priors
# ---------------------------------------------------------------------------


def _measure_edge_distance(
    kpts: np.ndarray,
    ei: int,
    ej: int,
    min_valid_conf: float,
) -> Optional[float]:
    """Measure pixel distance between two keypoints if both are valid."""
    xi, yi, ci = float(kpts[ei, 0]), float(kpts[ei, 1]), float(kpts[ei, 2])
    xj, yj, cj = float(kpts[ej, 0]), float(kpts[ej, 1]), float(kpts[ej, 2])
    if ci < float(min_valid_conf) or cj < float(min_valid_conf):
        return None
    if not (math.isfinite(xi) and math.isfinite(yi)):
        return None
    if not (math.isfinite(xj) and math.isfinite(yj)):
        return None
    dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    return dist if dist > 0.0 else None


def _accumulate_edge_samples(
    high_conf_df: pd.DataFrame,
    pose_labels: List[str],
    skeleton_edges: List[Tuple[int, int]],
    min_valid_conf: float,
) -> Dict[Tuple[int, int], List[float]]:
    """Accumulate per-edge distance samples from high-confidence rows.

    Vectorized equivalent of the historical ``iterrows()`` loop that called
    ``_extract_keypoints_from_row`` + ``_measure_edge_distance`` per row per
    edge. The outer loop over rows is replaced by numpy operations across
    all rows at once; the (small, fixed-size) loop over ``skeleton_edges``
    is retained so that per-edge sample lists are appended in the exact row
    order the ``iterrows()`` path produced -- this matters when multiple
    edges canonicalize to the same ``(min_idx, max_idx)`` key (e.g. a
    duplicate/reversed edge pair), since both edges' samples must interleave
    the same way per row within that shared list.
    """
    edge_samples: Dict[Tuple[int, int], List[float]] = {}
    K = len(pose_labels)
    if high_conf_df.empty:
        return edge_samples

    kpts, any_found = _stack_keypoints(high_conf_df, pose_labels)
    if K == 0 or not bool(any_found.any()):
        return edge_samples

    for edge in skeleton_edges:
        try:
            ei, ej = int(edge[0]), int(edge[1])
        except Exception:
            continue
        if ei >= K or ej >= K:
            continue

        xi = kpts[:, ei, 0].astype(np.float64)
        yi = kpts[:, ei, 1].astype(np.float64)
        ci = kpts[:, ei, 2].astype(np.float64)
        xj = kpts[:, ej, 0].astype(np.float64)
        yj = kpts[:, ej, 1].astype(np.float64)
        cj = kpts[:, ej, 2].astype(np.float64)

        # Mirrors _measure_edge_distance exactly: NaN confidence does NOT
        # fail the gate (NaN < threshold is False), only an explicit
        # below-threshold confidence does.
        conf_ok = ~(ci < float(min_valid_conf)) & ~(cj < float(min_valid_conf))
        coord_ok = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(xj) & np.isfinite(yj)

        dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        valid = any_found & conf_ok & coord_ok & (dist > 0.0)
        if not np.any(valid):
            continue

        key = (min(ei, ej), max(ei, ej))
        samples = [float(v) for v in dist[valid]]
        edge_samples.setdefault(key, []).extend(samples)
    return edge_samples


def _build_edge_priors(
    edge_samples: Dict[Tuple[int, int], List[float]],
) -> EdgeLengthPriors:
    """Build EdgeLengthPriors from accumulated per-edge distance samples."""
    priors: Dict[Tuple[int, int], Dict] = {}
    any_valid = False
    for key, samples in edge_samples.items():
        n = len(samples)
        arr = np.asarray(samples, dtype=np.float64)
        median_px = float(np.median(arr))
        mad_px = float(np.median(np.abs(arr - median_px)))
        priors[key] = {"median_px": median_px, "mad_px": mad_px, "n_samples": n}
        if n >= 20:
            any_valid = True
    return EdgeLengthPriors(priors=priors, is_valid=any_valid)


def calibrate_edge_length_priors(
    df: pd.DataFrame,
    pose_labels: List[str],
    skeleton_edges: List[Tuple[int, int]],
    min_valid_conf: float,
    high_conf_floor: float = 0.7,
) -> EdgeLengthPriors:
    """Estimate per-skeleton-edge length distributions from high-confidence frames.

    For each edge (i, j) defined in *skeleton_edges*, measures the pixel
    distance between the two keypoints in every high-confidence row and
    computes the median and MAD of those distances.

    Args:
        df: DataFrame with PoseKpt_{label}_X/Y/Conf columns and PoseMeanConf.
        pose_labels: Ordered list of keypoint label strings.
        skeleton_edges: List of (i, j) keypoint index pairs.
        min_valid_conf: Minimum confidence to treat a keypoint as valid.
        high_conf_floor: Minimum PoseMeanConf to include a row in calibration.

    Returns:
        EdgeLengthPriors; is_valid=True when at least one edge has >= 20 samples.
    """
    _invalid = EdgeLengthPriors(priors={}, is_valid=False)

    if df is None or df.empty or not pose_labels or not skeleton_edges:
        return _invalid

    high_conf_df = _filter_high_conf_rows(df, high_conf_floor)
    if high_conf_df.empty:
        return _invalid

    edge_samples = _accumulate_edge_samples(
        high_conf_df,
        pose_labels,
        skeleton_edges,
        min_valid_conf,
    )
    return _build_edge_priors(edge_samples)


# ---------------------------------------------------------------------------
# normalize_pose_keypoints_for_relink
# ---------------------------------------------------------------------------


def normalize_pose_keypoints_for_relink(
    window_df: pd.DataFrame,
    pose_labels: List[str],
    min_valid_conf: float,
) -> tuple:
    """Aggregate and normalize pose keypoints across a short window of rows.

    This is the canonical replacement for ``_normalize_pose_keypoints_window``
    in ``processing.py``.  Results are identical to that function.

    Args:
        window_df: DataFrame of up to ~3 rows for a single trajectory window.
        pose_labels: Ordered list of keypoint label strings.
        min_valid_conf: Minimum confidence to include a keypoint in aggregation.

    Returns:
        (normalized_array, visibility) where normalized_array is [K, 3] float32
        or None, and visibility is a float in [0, 1].
    """
    if window_df is None or window_df.empty or not pose_labels:
        return None, 0.0

    K = len(pose_labels)
    keypoints = np.full((K, 3), np.nan, dtype=np.float32)
    keypoints[:, 2] = 0.0
    valid_points = []
    valid_weights = []

    for idx, label in enumerate(pose_labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        if (
            x_col not in window_df.columns
            or y_col not in window_df.columns
            or c_col not in window_df.columns
        ):
            continue

        vals = window_df[[x_col, y_col, c_col]].copy()
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        vals = vals[vals[c_col] >= float(min_valid_conf)]
        if vals.empty:
            continue

        x = float(vals[x_col].median())
        y = float(vals[y_col].median())
        conf = float(vals[c_col].median())
        keypoints[idx] = np.array([x, y, conf], dtype=np.float32)
        valid_points.append((x, y))
        valid_weights.append(max(1e-6, conf))

    if not valid_points:
        return None, 0.0

    pts_arr = np.asarray(valid_points, dtype=np.float64)
    w_arr = np.asarray(valid_weights, dtype=np.float64)
    cx = float(np.average(pts_arr[:, 0], weights=w_arr))
    cy = float(np.average(pts_arr[:, 1], weights=w_arr))
    centered = pts_arr - np.array([[cx, cy]], dtype=np.float64)
    radii = np.sqrt(np.sum(centered**2, axis=1))
    scale = float(np.median(radii[radii > 1e-6])) if np.any(radii > 1e-6) else 1.0
    scale = max(scale, 1.0)

    normalized = np.full((K, 3), np.nan, dtype=np.float32)
    normalized[:, 2] = 0.0
    valid_counter = 0
    for idx in range(K):
        x, y, conf = keypoints[idx]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
            continue
        normalized[idx, 0] = np.float32((float(x) - cx) / scale)
        normalized[idx, 1] = np.float32((float(y) - cy) / scale)
        normalized[idx, 2] = np.float32(conf)
        valid_counter += 1

    visibility = float(valid_counter) / float(K) if K > 0 else 0.0
    return normalized, float(np.clip(visibility, 0.0, 1.0))


# ---------------------------------------------------------------------------
# apply_quality_to_dataframe
# ---------------------------------------------------------------------------


def apply_quality_to_dataframe(
    df: pd.DataFrame,
    pose_labels: List[str],
    params: dict,
    body_length_prior: Optional[BodyLengthPrior] = None,
    anterior_indices: Optional[List[int]] = None,
    posterior_indices: Optional[List[int]] = None,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    edge_length_priors: Optional[EdgeLengthPriors] = None,
) -> pd.DataFrame:
    """Apply pose quality assessment to all rows of a trajectory DataFrame.

    Modifies confidence columns in-place on a copy, adds quality metadata
    columns, and returns the modified copy.  Never raises exceptions for bad
    rows — they are marked as rejected.

    Args:
        df: Input DataFrame with PoseKpt_{label}_X/Y/Conf columns.
        pose_labels: Ordered list of keypoint label strings.
        params: Tracking parameter dict.  Relevant keys:
            POSE_MIN_KPT_CONF_VALID, POSE_EXPORT_MIN_VALID_FRACTION,
            POSE_EXPORT_MIN_VALID_KEYPOINTS, POSE_IGNORE_KEYPOINTS.
        body_length_prior: Optional prior for body-length outlier filtering.
        anterior_indices: Anterior keypoint indices for body-length calc.
        posterior_indices: Posterior keypoint indices for body-length calc.
        skeleton_edges: List of (i, j) index pairs for connected-keypoint
            anatomy checks.
        edge_length_priors: Calibrated per-edge length priors.

    Returns:
        Modified copy of df with added/updated quality columns.
    """
    if df is None or df.empty:
        return df

    # ------------------------------------------------------------------
    # Extract parameters
    # ------------------------------------------------------------------
    min_valid_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
    min_valid_fraction = float(params.get("POSE_EXPORT_MIN_VALID_FRACTION", 0.5))
    min_valid_keypoints = int(params.get("POSE_EXPORT_MIN_VALID_KEYPOINTS", 3))
    ignore_kpts_raw = params.get("POSE_IGNORE_KEYPOINTS", [])
    ignore_indices: List[int] = [int(i) for i in (ignore_kpts_raw or [])]

    # ------------------------------------------------------------------
    # Build column triplets
    # ------------------------------------------------------------------
    col_triplets = []
    for label in pose_labels:
        col_triplets.append(
            (f"PoseKpt_{label}_X", f"PoseKpt_{label}_Y", f"PoseKpt_{label}_Conf")
        )

    out = df.copy()

    # ------------------------------------------------------------------
    # Initialize new columns
    # ------------------------------------------------------------------
    out["PoseQualityScore"] = float("nan")
    out["PoseQualityState"] = ""
    out["PoseQualityFlags"] = ""
    out["PoseSource"] = ""
    out["PoseWasCleaned"] = 0

    # Determine which rows have any pose data
    all_conf_cols = [c for _, _, c in col_triplets if c in out.columns]

    for row_idx in out.index:
        row = out.loc[row_idx]

        # Check if this row has any pose data at all
        has_pose = False
        if all_conf_cols:
            try:
                has_pose = any(
                    not pd.isna(row[c]) for c in all_conf_cols if c in out.columns
                )
            except Exception:
                has_pose = False

        if not has_pose:
            out.at[row_idx, "PoseQualityState"] = "no_pose"
            out.at[row_idx, "PoseQualityScore"] = 0.0
            out.at[row_idx, "PoseSource"] = ""
            continue

        # Extract keypoints for this row
        kpts = _extract_keypoints_from_row(row, pose_labels)

        result = assess_pose_row(
            kpts,
            min_valid_conf=min_valid_conf,
            min_valid_fraction=min_valid_fraction,
            min_valid_keypoints=min_valid_keypoints,
            ignore_indices=ignore_indices if ignore_indices else None,
            body_length_prior=body_length_prior,
            anterior_indices=anterior_indices,
            posterior_indices=posterior_indices,
            skeleton_edges=skeleton_edges,
            edge_length_priors=edge_length_priors,
        )

        # Write cleaned confidences back
        for i, label in enumerate(pose_labels):
            c_col = f"PoseKpt_{label}_Conf"
            if c_col in out.columns and i < len(result.cleaned_keypoints):
                out.at[row_idx, c_col] = float(result.cleaned_keypoints[i, 2])

        out.at[row_idx, "PoseQualityScore"] = result.quality_score
        out.at[row_idx, "PoseQualityState"] = result.quality_state
        out.at[row_idx, "PoseQualityFlags"] = "|".join(result.quality_flags)
        out.at[row_idx, "PoseWasCleaned"] = int(result.was_cleaned)
        out.at[row_idx, "PoseSource"] = "cache"

    return out


# ---------------------------------------------------------------------------
# apply_temporal_pose_postprocessing
# ---------------------------------------------------------------------------


def _suppress_temporal_outliers(
    out: pd.DataFrame,
    pose_labels: List[str],
    rolling_window: int,
    z_score_threshold: float,
) -> None:
    """Zero confidence for keypoints that are rolling-z-score outliers (in-place)."""
    _VALID_STATES = {"good", "partial"}
    for label in pose_labels:
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"

        if not all(c in out.columns for c in (x_col, y_col, c_col)):
            continue

        valid_idx = _get_valid_label_indices(out, c_col, _VALID_STATES)
        if len(valid_idx) < 3:
            continue

        x_series = out.loc[valid_idx, x_col].astype(float)
        y_series = out.loc[valid_idx, y_col].astype(float)

        for series in (x_series, y_series):
            _flag_rolling_outliers(
                out,
                series,
                valid_idx,
                c_col,
                rolling_window,
                z_score_threshold,
            )


def _get_valid_label_indices(
    out: pd.DataFrame,
    c_col: str,
    valid_states: set,
) -> list:
    """Return row indices where quality state is acceptable and conf > 0."""
    if "PoseQualityState" in out.columns:
        quality_ok = out["PoseQualityState"].isin(valid_states)
    else:
        quality_ok = pd.Series([True] * len(out), index=out.index)
    conf_ok = out[c_col].apply(lambda v: pd.notna(v) and float(v) > 0.0)
    return out.index[quality_ok & conf_ok].tolist()


def _flag_rolling_outliers(
    out: pd.DataFrame,
    series: pd.Series,
    valid_idx: list,
    c_col: str,
    rolling_window: int,
    z_score_threshold: float,
) -> None:
    """Flag individual index positions as temporal outliers based on rolling z-score."""
    roll_mean = series.rolling(rolling_window, min_periods=3, center=True).mean()
    roll_std = series.rolling(rolling_window, min_periods=3, center=True).std()

    for idx_val in valid_idx:
        if idx_val not in roll_mean.index:
            continue
        mean_v = roll_mean.loc[idx_val]
        if pd.isna(mean_v):
            continue
        std_v = (
            float(roll_std.loc[idx_val]) if not pd.isna(roll_std.loc[idx_val]) else 0.0
        )
        z = abs(float(series.loc[idx_val]) - float(mean_v)) / max(std_v, 1e-6)
        if z > float(z_score_threshold):
            out.at[idx_val, c_col] = 0.0
            _add_flag(out, idx_val, "temporal_outlier")
            out.at[idx_val, "PoseWasCleaned"] = 1


def _interpolate_gaps(
    out: pd.DataFrame,
    pose_labels: List[str],
    max_gap: int,
) -> None:
    """Linearly interpolate keypoint X/Y across short gaps (in-place)."""
    for label in pose_labels:
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"

        if not all(c in out.columns for c in (x_col, y_col, c_col)):
            continue

        valid_mask = out[c_col].apply(lambda v: pd.notna(v) and float(v) > 0.0)
        valid_positions = out.index[valid_mask].tolist()
        if len(valid_positions) < 2:
            continue

        for seg_start_pos, seg_end_pos in zip(
            valid_positions[:-1], valid_positions[1:]
        ):
            _fill_single_gap(
                out, seg_start_pos, seg_end_pos, x_col, y_col, c_col, max_gap
            )


def _fill_single_gap(
    out: pd.DataFrame,
    seg_start_pos,
    seg_end_pos,
    x_col: str,
    y_col: str,
    c_col: str,
    max_gap: int,
) -> None:
    """Linearly interpolate X/Y for a single gap between two valid positions."""
    start_iloc = out.index.get_loc(seg_start_pos)
    end_iloc = out.index.get_loc(seg_end_pos)
    gap_length = end_iloc - start_iloc - 1

    if gap_length <= 0 or gap_length > max_gap:
        return

    x_start = float(out.at[seg_start_pos, x_col])
    x_end = float(out.at[seg_end_pos, x_col])
    y_start = float(out.at[seg_start_pos, y_col])
    y_end = float(out.at[seg_end_pos, y_col])

    gap_indices = out.index[start_iloc + 1 : end_iloc]
    for step, gap_idx in enumerate(gap_indices, start=1):
        t = float(step) / float(gap_length + 1)
        out.at[gap_idx, x_col] = x_start + t * (x_end - x_start)
        out.at[gap_idx, y_col] = y_start + t * (y_end - y_start)
        out.at[gap_idx, c_col] = 0.3  # low-trust interpolated conf
        out.at[gap_idx, "PoseSource"] = "cleaned"
        out.at[gap_idx, "PoseWasCleaned"] = 1


def apply_temporal_pose_postprocessing(
    trajectory_df: pd.DataFrame,
    pose_labels: List[str],
    max_gap: int,
    z_score_threshold: float,
    fill_interpolated: bool = True,
) -> pd.DataFrame:
    """Apply temporal outlier suppression and gap-filling to one trajectory.

    Input must be a single trajectory's rows, ideally sorted by FrameID.
    This function sorts by FrameID internally.

    Args:
        trajectory_df: DataFrame for one animal trajectory.
        pose_labels: Ordered list of keypoint label strings.
        max_gap: Maximum number of consecutive bad frames to gap-fill.
        z_score_threshold: Rolling z-score above which a keypoint position
            is flagged as a temporal outlier.
        fill_interpolated: When True, linearly interpolate keypoint X/Y across
            gaps of at most ``max_gap`` frames.

    Returns:
        Modified copy of trajectory_df.
    """
    if trajectory_df is None or trajectory_df.empty or not pose_labels:
        return trajectory_df

    out = trajectory_df.copy()

    if "FrameID" in out.columns:
        out = out.sort_values("FrameID").reset_index(drop=True)

    rolling_window = max(5, max_gap * 2)

    _suppress_temporal_outliers(out, pose_labels, rolling_window, z_score_threshold)

    if fill_interpolated:
        _interpolate_gaps(out, pose_labels, max_gap)

    _recompute_pose_summary(out, pose_labels)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float_array(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Elementwise emulate ``float(x)`` semantics for a 1-D array.

    Returns ``(values, ok)`` where ``values`` is float64 (NaN where a
    conversion failed or wasn't attempted) and ``ok`` is a bool mask that is
    True exactly where ``float(x)`` would succeed without raising.  Numeric
    dtypes (float/int/bool) always succeed, matching ``float(x)`` for a
    plain numpy scalar (including NaN, which converts to NaN without
    raising). Object-dtype columns fall back to an elementwise try/except to
    stay faithful to arbitrary cell contents (e.g. a stray non-numeric
    string), matching the original per-row ``float(x)`` conversion exactly.
    """
    if raw.dtype.kind in "fiub":
        vals = raw.astype(np.float64, copy=False)
        ok = np.ones(len(raw), dtype=bool)
        return vals, ok

    n = len(raw)
    vals = np.full(n, np.nan, dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    for idx in range(n):
        try:
            vals[idx] = float(raw[idx])
            ok[idx] = True
        except (ValueError, TypeError):
            ok[idx] = False
    return vals, ok


def _stack_keypoints(
    df: pd.DataFrame, pose_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of calling ``_extract_keypoints_from_row`` for
    every row of *df*.

    Returns ``(kpts, any_found)`` where ``kpts`` is ``(M, K, 3)`` float32
    (NaN x/y, conf=0.0 defaults, exactly as ``_extract_keypoints_from_row``
    initializes each row) and ``any_found[m]`` is True iff at least one
    keypoint was successfully extracted for row *m* -- mirroring that
    function returning ``None`` for a row when nothing was found.
    """
    M = len(df)
    K = len(pose_labels)
    kpts = np.full((M, K, 3), np.nan, dtype=np.float32)
    kpts[:, :, 2] = 0.0
    any_found = np.zeros(M, dtype=bool)

    for i, label in enumerate(pose_labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        if (
            x_col not in df.columns
            or y_col not in df.columns
            or c_col not in df.columns
        ):
            continue

        fx, ok_x = _to_float_array(df[x_col].to_numpy())
        fy, ok_y = _to_float_array(df[y_col].to_numpy())
        fc, ok_c = _to_float_array(df[c_col].to_numpy())
        ok = ok_x & ok_y & ok_c
        if not np.any(ok):
            continue

        kpts[ok, i, 0] = fx[ok].astype(np.float32)
        kpts[ok, i, 1] = fy[ok].astype(np.float32)
        kpts[ok, i, 2] = fc[ok].astype(np.float32)
        any_found |= ok

    return kpts, any_found


def _weighted_centroid_batch(
    kpts: np.ndarray,
    indices: Optional[List[int]],
    min_valid_conf: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``_weighted_centroid`` (features.py) across
    all rows of a stacked ``(M, K, 3)`` keypoints array, for an always-empty
    ignore set (the only case this calibration path needs).

    Returns ``(cx, cy, has_valid)``, each length ``M``. Accumulation is a
    left-to-right running sum over *indices* (matching the accumulation
    order ``_weighted_centroid``'s python loop would use) with invalid
    entries contributing an exact ``+0.0`` -- mathematically a no-op under
    IEEE-754 addition, so the running sum for each row is bit-identical to
    summing only the valid terms in the same order.
    """
    M, K, _ = kpts.shape
    sum_w = np.zeros(M, dtype=np.float64)
    sum_wx = np.zeros(M, dtype=np.float64)
    sum_wy = np.zeros(M, dtype=np.float64)
    has_valid = np.zeros(M, dtype=bool)

    for idx in indices or []:
        idx = int(idx)
        if idx < 0 or idx >= K:
            continue
        x = kpts[:, idx, 0].astype(np.float64)
        y = kpts[:, idx, 1].astype(np.float64)
        c = kpts[:, idx, 2].astype(np.float64)
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(c)
            & (c >= float(min_valid_conf))
        )
        w = np.where(valid, np.maximum(1e-6, c), 0.0)
        sum_w = sum_w + w
        sum_wx = sum_wx + np.where(valid, w * x, 0.0)
        sum_wy = sum_wy + np.where(valid, w * y, 0.0)
        has_valid |= valid

    with np.errstate(invalid="ignore", divide="ignore"):
        cx = sum_wx / sum_w
        cy = sum_wy / sum_w
    return cx, cy, has_valid


def _extract_keypoints_from_row(row, pose_labels: List[str]) -> Optional[np.ndarray]:
    """Extract a [K, 3] float32 keypoints array from a DataFrame row."""
    K = len(pose_labels)
    kpts = np.full((K, 3), np.nan, dtype=np.float32)
    kpts[:, 2] = 0.0

    any_found = False
    for i, label in enumerate(pose_labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        try:
            x = row[x_col]
            y = row[y_col]
            conf = row[c_col]
        except (KeyError, TypeError):
            continue
        try:
            fx, fy, fc = float(x), float(y), float(conf)
        except (ValueError, TypeError):
            continue
        kpts[i] = np.array([fx, fy, fc], dtype=np.float32)
        any_found = True

    return kpts if any_found else None


def _add_flag(df: pd.DataFrame, idx, flag: str) -> None:
    """Append a quality flag to PoseQualityFlags for the given row index."""
    if "PoseQualityFlags" not in df.columns:
        return
    current = (
        str(df.at[idx, "PoseQualityFlags"])
        if pd.notna(df.at[idx, "PoseQualityFlags"])
        else ""
    )
    if flag in current.split("|"):
        return
    new_val = f"{current}|{flag}" if current else flag
    df.at[idx, "PoseQualityFlags"] = new_val


def _collect_row_conf_stats(
    row,
    present_conf_cols: List[str],
) -> Tuple[List[float], int]:
    """Collect finite confidence values and count of positive-confidence keypoints."""
    confs: List[float] = []
    valid_count = 0
    for c in present_conf_cols:
        v = row[c]
        try:
            fv = float(v)
            if np.isfinite(fv):
                confs.append(fv)
                if fv > 0.0:
                    valid_count += 1
        except (ValueError, TypeError):
            pass
    return confs, valid_count


def _recompute_pose_summary(df: pd.DataFrame, pose_labels: List[str]) -> None:
    """Recompute PoseMeanConf and PoseValidFraction columns in-place."""
    if not pose_labels:
        return
    conf_cols = [f"PoseKpt_{label}_Conf" for label in pose_labels]
    present_conf_cols = [c for c in conf_cols if c in df.columns]
    if not present_conf_cols:
        return

    K = len(pose_labels)
    has_mean = "PoseMeanConf" in df.columns
    has_frac = "PoseValidFraction" in df.columns

    if has_mean or has_frac:
        for idx in df.index:
            confs, valid_count = _collect_row_conf_stats(df.loc[idx], present_conf_cols)
            if has_mean:
                df.at[idx, "PoseMeanConf"] = float(np.mean(confs)) if confs else 0.0
            if has_frac:
                df.at[idx, "PoseValidFraction"] = (
                    float(valid_count) / float(K) if K > 0 else 0.0
                )
