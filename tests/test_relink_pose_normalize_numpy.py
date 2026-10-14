"""normalize_pose_keypoints_for_relink: numpy path, identical results.

Regression: pose relinking spent ~82% of post-processing in this function on a
crowded full-arena clip (8.5k trajectories, two windows each, dozens of tiny
pandas operations per call). The numpy rewrite must give identical output.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.individual.pose.quality import normalize_pose_keypoints_for_relink

LABELS = ["a", "b", "c", "d", "e"]


def _reference(
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


def _window(rng, rows):
    data = {"FrameID": np.arange(rows)}
    for lbl in LABELS:
        x = rng.normal(500, 50, rows)
        y = rng.normal(500, 50, rows)
        c = rng.uniform(0, 1, rows)
        for arr in (x, y, c):
            mask = rng.uniform(size=rows)
            arr[mask < 0.15] = np.nan
            arr[(mask >= 0.15) & (mask < 0.2)] = np.inf
        data[f"PoseKpt_{lbl}_X"] = x
        data[f"PoseKpt_{lbl}_Y"] = y
        data[f"PoseKpt_{lbl}_Conf"] = c
    df = pd.DataFrame(data)
    if rng.uniform() < 0.2:  # a label missing entirely
        df = df.drop(columns=[f"PoseKpt_{LABELS[0]}_Conf"])
    return df


@pytest.mark.parametrize("seed", range(300))
def test_numpy_path_matches_reference(seed):
    rng = np.random.default_rng(seed)
    df = _window(rng, int(rng.integers(1, 4)))
    conf = float(rng.choice([0.0, 0.2, 0.5]))
    got = normalize_pose_keypoints_for_relink(df, LABELS, conf)
    want = _reference(df, LABELS, conf)
    if want[0] is None:
        assert got[0] is None and got[1] == want[1]
        return
    np.testing.assert_array_equal(got[0], want[0])
    assert got[1] == want[1]


def test_empty_inputs():
    assert normalize_pose_keypoints_for_relink(pd.DataFrame(), LABELS, 0.2) == (
        None,
        0.0,
    )
    assert normalize_pose_keypoints_for_relink(None, LABELS, 0.2) == (None, 0.0)
