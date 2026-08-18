"""Byte-identical characterization test for Task 8: vectorized temporal pose
post-processing.

This test freezes the *original* (pre-vectorization) implementations of
``_suppress_temporal_outliers``, ``_interpolate_gaps`` and
``_recompute_pose_summary`` (plus their original private helpers, copied
verbatim from ``quality.py`` before the vectorization edit) and compares
their output, byte-for-byte, against the live (post-vectorization) module
functions on a deliberately non-degenerate fixture that exercises every
branch called out in the task-8 brief:

- Temporal outliers on X only, on Y only, and on both (to prove X-before-Y
  ordering / flag dedup behaves identically).
- Rows straddling the z-score threshold (just above / just below).
- Partial windows at the start/end of the series (min_periods=3 boundary).
- Short gaps that get interpolated (t = step/(gap+1) fill) and gaps that are
  too long to fill (left as-is).
- The roll_std == 0 -> 1e-6 epsilon path (a run of identical values).
- Multiple flags co-occurring (existing PoseQualityFlags + temporal_outlier
  appended, in token order).
- The summary recompute, including a non-finite (NaN) confidence value that
  must be excluded from the PoseMeanConf average (exercises the exact-order
  compaction path, not a padded-with-zero approximation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hydra_suite.core.individual.pose import quality as quality_mod
from hydra_suite.core.individual.pose.quality import (
    _add_flag,
    apply_temporal_pose_postprocessing,
)

# ---------------------------------------------------------------------------
# Frozen reference implementation (verbatim copy of the pre-vectorization
# code in quality.py, as of the start of Task 8).
# ---------------------------------------------------------------------------


def _ref_get_valid_label_indices(out, c_col, valid_states):
    if "PoseQualityState" in out.columns:
        quality_ok = out["PoseQualityState"].isin(valid_states)
    else:
        quality_ok = pd.Series([True] * len(out), index=out.index)
    conf_ok = out[c_col].apply(lambda v: pd.notna(v) and float(v) > 0.0)
    return out.index[quality_ok & conf_ok].tolist()


def _ref_flag_rolling_outliers(
    out, series, valid_idx, c_col, rolling_window, z_score_threshold
):
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


def _ref_suppress_temporal_outliers(
    out, pose_labels, rolling_window, z_score_threshold
):
    _VALID_STATES = {"good", "partial"}
    for label in pose_labels:
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"

        if not all(c in out.columns for c in (x_col, y_col, c_col)):
            continue

        valid_idx = _ref_get_valid_label_indices(out, c_col, _VALID_STATES)
        if len(valid_idx) < 3:
            continue

        x_series = out.loc[valid_idx, x_col].astype(float)
        y_series = out.loc[valid_idx, y_col].astype(float)

        for series in (x_series, y_series):
            _ref_flag_rolling_outliers(
                out, series, valid_idx, c_col, rolling_window, z_score_threshold
            )


def _ref_fill_single_gap(out, seg_start_pos, seg_end_pos, x_col, y_col, c_col, max_gap):
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
        out.at[gap_idx, c_col] = 0.3
        out.at[gap_idx, "PoseSource"] = "cleaned"
        out.at[gap_idx, "PoseWasCleaned"] = 1


def _ref_interpolate_gaps(out, pose_labels, max_gap):
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
            _ref_fill_single_gap(
                out, seg_start_pos, seg_end_pos, x_col, y_col, c_col, max_gap
            )


def _ref_collect_row_conf_stats(row, present_conf_cols):
    confs = []
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


def _ref_recompute_pose_summary(df, pose_labels):
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
            confs, valid_count = _ref_collect_row_conf_stats(
                df.loc[idx], present_conf_cols
            )
            if has_mean:
                df.at[idx, "PoseMeanConf"] = float(np.mean(confs)) if confs else 0.0
            if has_frac:
                df.at[idx, "PoseValidFraction"] = (
                    float(valid_count) / float(K) if K > 0 else 0.0
                )


def _ref_apply_temporal_pose_postprocessing(
    trajectory_df, pose_labels, max_gap, z_score_threshold, fill_interpolated=True
):
    if trajectory_df is None or trajectory_df.empty or not pose_labels:
        return trajectory_df

    out = trajectory_df.copy()

    if "FrameID" in out.columns:
        out = out.sort_values("FrameID").reset_index(drop=True)

    rolling_window = max(5, max_gap * 2)

    _ref_suppress_temporal_outliers(out, pose_labels, rolling_window, z_score_threshold)

    if fill_interpolated:
        _ref_interpolate_gaps(out, pose_labels, max_gap)

    _ref_recompute_pose_summary(out, pose_labels)

    return out


# ---------------------------------------------------------------------------
# Fixture: a single, non-degenerate trajectory exercising every branch.
# ---------------------------------------------------------------------------

_LABELS = ["head", "tail"]


def _build_fixture() -> pd.DataFrame:
    """25-frame single-trajectory fixture with distinct values per row.

    Layout (0-indexed FrameID):
      head X: smooth trajectory with a deliberate spike at frame 10 (X
              outlier only) and another at frame 18 (X+Y outlier, to prove
              X-before-Y ordering matters for the shared conf column).
      head Y: spike at frame 14 (Y outlier only) and frame 18 (shared).
      tail X/Y: mostly flat run (std==0 -> 1e-6 epsilon path) with one
                short gap (frames 6-7, length 2, fillable) and one long gap
                (frames 20-23, length 4 > max_gap=3, NOT filled).
      Row 24: dropped confidence (conf=0) at the tail end -> tests a
              partial rolling window at the series boundary.
      Row 0/1: also boundary rows (min_periods=3 -> first two positions of
               any window get NaN roll stats unless enough neighbors).
    """
    n = 25
    rng = np.random.default_rng(12345)

    frame_id = np.arange(n)

    # --- head keypoint: smooth base + 2 distinct spikes -------------------
    head_x = 100.0 + 2.0 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 0.05, n)
    head_y = 200.0 + 1.5 * np.cos(np.linspace(0, 6, n)) + rng.normal(0, 0.05, n)
    head_x[10] += 120.0  # X-only outlier
    head_y[14] += 120.0  # Y-only outlier
    head_x[18] += 110.0  # X+Y outlier (shared row)
    head_y[18] += 110.0
    head_conf = np.full(n, 0.8)
    head_conf += rng.normal(0, 0.01, n)  # distinct values, still >0

    # --- tail keypoint: flat run (std==0 path) + gaps ----------------------
    tail_x = np.full(n, 300.0)
    tail_y = np.full(n, 400.0)
    tail_x[8:15] = 300.0  # identical run -> roll_std == 0 over this window
    tail_y[8:15] = 400.0
    tail_conf = np.full(n, 0.6)
    tail_conf += rng.normal(0, 0.01, n)

    # short fillable gap: frames 6,7 invalid (gap length 2 <= max_gap=3)
    tail_conf[6] = 0.0
    tail_conf[7] = 0.0
    # long unfillable gap: frames 20,21,22,23 invalid (length 4 > max_gap=3)
    tail_conf[20] = 0.0
    tail_conf[21] = 0.0
    tail_conf[22] = 0.0
    tail_conf[23] = 0.0
    # nudge the endpoints around the gaps to be distinct (non-degenerate)
    tail_x[5] = 301.3
    tail_x[8] = 302.7
    tail_y[5] = 401.1
    tail_y[8] = 402.9
    tail_x[19] = 305.5
    tail_x[24] = 309.9
    tail_y[19] = 405.5
    tail_y[24] = 409.9

    # tail: also drop confidence at the very last row (boundary window)
    quality_state = np.full(n, "good", dtype=object)
    quality_state[3] = "partial"
    quality_state[16] = "bad"  # excluded from valid_idx despite conf>0

    df = pd.DataFrame(
        {
            "FrameID": frame_id,
            "TrajectoryID": 1,
            "PoseKpt_head_X": head_x,
            "PoseKpt_head_Y": head_y,
            "PoseKpt_head_Conf": head_conf,
            "PoseKpt_tail_X": tail_x,
            "PoseKpt_tail_Y": tail_y,
            "PoseKpt_tail_Conf": tail_conf,
            "PoseQualityState": quality_state,
            "PoseQualityFlags": "",
            "PoseSource": "cache",
            "PoseWasCleaned": 0,
            "PoseMeanConf": 0.0,
            "PoseValidFraction": 0.0,
        }
    )

    # Pre-seed some existing flags on a couple of rows to prove flag-token
    # ordering (existing flag -> temporal_outlier appended after).
    df.loc[10, "PoseQualityFlags"] = "low_conf:1"
    df.loc[18, "PoseQualityFlags"] = "edge_outlier:2"

    # Introduce a NaN confidence value (non-finite path) for the summary
    # recompute exact-order-compaction test, on a row untouched by outlier
    # suppression (row 2).
    df.loc[2, "PoseKpt_head_Conf"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_df():
    return _build_fixture()


def test_characterization_matches_frozen_reference(fixture_df):
    """Sanity: the frozen reference reproduces itself deterministically and
    both branches (X-outlier, Y-outlier, shared-outlier, short-gap fill,
    long-gap skip, std==0 epsilon path) actually fire on this fixture."""
    ref = _ref_apply_temporal_pose_postprocessing(
        fixture_df, _LABELS, max_gap=3, z_score_threshold=1.5
    )

    # X-only outlier at frame 10 fired: suppression zeroes it (isolated bad
    # frame -> gap length 1 <= max_gap=3), so interpolation (which runs
    # AFTER suppression) then refills it at conf=0.3/"cleaned" -- proving
    # the suppression->interpolation ordering, not that it stays at 0.0.
    assert ref.loc[10, "PoseKpt_head_Conf"] == pytest.approx(0.3)
    assert ref.loc[10, "PoseSource"] == "cleaned"
    assert "temporal_outlier" in ref.loc[10, "PoseQualityFlags"].split("|")
    # Flag ordering preserved: existing flag first, then temporal_outlier.
    assert ref.loc[10, "PoseQualityFlags"] == "low_conf:1|temporal_outlier"

    # Y-only outlier at frame 14 fired (same cascade into interpolation).
    assert ref.loc[14, "PoseKpt_head_Conf"] == pytest.approx(0.3)
    assert ref.loc[14, "PoseSource"] == "cleaned"
    assert "temporal_outlier" in ref.loc[14, "PoseQualityFlags"].split("|")

    # Shared X+Y outlier at frame 18: flag appears exactly once (dedup).
    flags_18 = ref.loc[18, "PoseQualityFlags"].split("|")
    assert flags_18.count("temporal_outlier") == 1
    assert ref.loc[18, "PoseQualityFlags"] == "edge_outlier:2|temporal_outlier"

    # Row 16 (quality_state == "bad") never enters valid_idx -> untouched.
    assert ref.loc[16, "PoseQualityFlags"] == ""

    # Frame 8's X (302.7) is itself far enough from the flat tail run to be
    # flagged a temporal outlier by suppression -- which *widens* the gap
    # that interpolation later sees (5 valid, 6/7 originally invalid, 8 now
    # also invalid) to length 3, exactly == max_gap: a real ordering
    # cascade (suppression -> interpolation) AND a gap-length boundary case.
    assert "temporal_outlier" in ref.loc[8, "PoseQualityFlags"].split("|")
    assert ref.loc[6, "PoseKpt_tail_Conf"] == pytest.approx(0.3)
    assert ref.loc[7, "PoseKpt_tail_Conf"] == pytest.approx(0.3)
    assert ref.loc[8, "PoseKpt_tail_Conf"] == pytest.approx(0.3)
    assert ref.loc[6, "PoseSource"] == "cleaned"
    assert ref.loc[7, "PoseSource"] == "cleaned"
    assert ref.loc[8, "PoseSource"] == "cleaned"
    # Linear interpolation formula t = step/(gap+1), gap bordered by 5 and 9.
    x5, x9 = fixture_df.loc[5, "PoseKpt_tail_X"], fixture_df.loc[9, "PoseKpt_tail_X"]
    expected_x6 = x5 + (1 / 4) * (x9 - x5)
    expected_x7 = x5 + (2 / 4) * (x9 - x5)
    expected_x8 = x5 + (3 / 4) * (x9 - x5)
    assert ref.loc[6, "PoseKpt_tail_X"] == pytest.approx(expected_x6)
    assert ref.loc[7, "PoseKpt_tail_X"] == pytest.approx(expected_x7)
    assert ref.loc[8, "PoseKpt_tail_X"] == pytest.approx(expected_x8)

    # Long gap (frames 20-23) NOT filled: conf stays 0, source untouched.
    for f in (20, 21, 22, 23):
        assert ref.loc[f, "PoseKpt_tail_Conf"] == 0.0
        assert ref.loc[f, "PoseSource"] == "cache"

    # NaN confidence row (row 2) excluded from PoseMeanConf average.
    assert np.isfinite(ref.loc[2, "PoseMeanConf"])


def test_vectorized_matches_frozen_reference_exact(fixture_df):
    """The live (vectorized) implementation must match the frozen reference
    byte-for-byte on every affected column."""
    ref_input = fixture_df.copy(deep=True)
    live_input = fixture_df.copy(deep=True)

    ref = _ref_apply_temporal_pose_postprocessing(
        ref_input, _LABELS, max_gap=3, z_score_threshold=1.5
    )
    live = apply_temporal_pose_postprocessing(
        live_input, _LABELS, max_gap=3, z_score_threshold=1.5
    )

    assert_frame_equal(ref, live, check_exact=True, check_dtype=True)

    # Explicit ordering / token-order re-confirmation directly on live output.
    assert live.loc[10, "PoseQualityFlags"] == "low_conf:1|temporal_outlier"
    assert live.loc[18, "PoseQualityFlags"] == "edge_outlier:2|temporal_outlier"
    assert live.loc[18, "PoseQualityFlags"].split("|").count("temporal_outlier") == 1


def test_vectorized_matches_frozen_reference_no_interpolation(fixture_df):
    """Same check with fill_interpolated=False (suppression + summary only,
    exercising the suppression->recompute ordering without the interpolation
    step in between)."""
    ref_input = fixture_df.copy(deep=True)
    live_input = fixture_df.copy(deep=True)

    ref = _ref_apply_temporal_pose_postprocessing(
        ref_input, _LABELS, max_gap=3, z_score_threshold=1.5, fill_interpolated=False
    )
    live = apply_temporal_pose_postprocessing(
        live_input, _LABELS, max_gap=3, z_score_threshold=1.5, fill_interpolated=False
    )
    assert_frame_equal(ref, live, check_exact=True, check_dtype=True)


def test_vectorized_matches_frozen_reference_multi_trajectory():
    """Concatenate two distinct trajectories (different RNG offsets) run
    independently, matching the pose_merge.py groupby-driver usage pattern."""
    df1 = _build_fixture()
    df2 = _build_fixture()
    df2["TrajectoryID"] = 2
    # Perturb the second trajectory so it isn't a duplicate (non-degenerate).
    df2["PoseKpt_head_X"] = df2["PoseKpt_head_X"] + 7.3
    df2["PoseKpt_tail_Y"] = df2["PoseKpt_tail_Y"] - 4.1

    for traj_df in (df1, df2):
        ref_input = traj_df.copy(deep=True)
        live_input = traj_df.copy(deep=True)
        ref = _ref_apply_temporal_pose_postprocessing(
            ref_input, _LABELS, max_gap=3, z_score_threshold=1.5
        )
        live = apply_temporal_pose_postprocessing(
            live_input, _LABELS, max_gap=3, z_score_threshold=1.5
        )
        assert_frame_equal(ref, live, check_exact=True, check_dtype=True)


def test_vectorized_matches_frozen_reference_varied_thresholds():
    """Vary z_score_threshold and max_gap to sweep the boundary conditions
    (rows exactly at/near the z threshold, gap == max_gap exactly)."""
    base = _build_fixture()
    for z_thr in (1.5, 2.0, 3.0, 5.0):
        for max_gap in (1, 2, 3, 4):
            ref_input = base.copy(deep=True)
            live_input = base.copy(deep=True)
            ref = _ref_apply_temporal_pose_postprocessing(
                ref_input, _LABELS, max_gap=max_gap, z_score_threshold=z_thr
            )
            live = apply_temporal_pose_postprocessing(
                live_input, _LABELS, max_gap=max_gap, z_score_threshold=z_thr
            )
            assert_frame_equal(
                ref,
                live,
                check_exact=True,
                check_dtype=True,
                obj=f"z_thr={z_thr} max_gap={max_gap}",
            )


def test_recompute_pose_summary_matches_reference_directly():
    """Directly target ``_recompute_pose_summary`` (bypassing suppression /
    interpolation) with a fixture containing NaN, +inf and a non-numeric
    string in the confidence columns, matching the reference's finite-value
    compaction (which must NOT be approximated by zero-padding -- see task
    brief warning about float non-associativity)."""
    n = 12
    rng = np.random.default_rng(7)
    conf_a = 0.1 + 0.05 * np.arange(n) + rng.normal(0, 0.001, n)
    conf_b = 0.2 + 0.03 * np.arange(n) + rng.normal(0, 0.001, n)
    conf_c = 0.3 + 0.02 * np.arange(n) + rng.normal(0, 0.001, n)

    df = pd.DataFrame(
        {
            "PoseKpt_a_Conf": conf_a.astype(object),
            "PoseKpt_b_Conf": conf_b,
            "PoseKpt_c_Conf": conf_c,
            "PoseMeanConf": 0.0,
            "PoseValidFraction": 0.0,
        }
    )
    # Non-finite / bad values on a subset of rows (partial-row path).
    df.loc[2, "PoseKpt_a_Conf"] = np.nan
    df.loc[5, "PoseKpt_b_Conf"] = np.inf
    df.loc[7, "PoseKpt_c_Conf"] = -np.inf
    df.loc[9, "PoseKpt_a_Conf"] = 0.0  # exactly zero -> not counted as valid

    ref_df = df.copy(deep=True)
    live_df = df.copy(deep=True)

    labels = ["a", "b", "c"]
    _ref_recompute_pose_summary(ref_df, labels)
    quality_mod._recompute_pose_summary(live_df, labels)

    assert_frame_equal(ref_df, live_df, check_exact=True, check_dtype=True)
