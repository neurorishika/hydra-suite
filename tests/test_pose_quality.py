"""Unit tests for the pose_quality module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.individual.pose.quality import BodyLengthPrior, EdgeLengthPriors
from hydra_suite.core.individual.pose.quality import (
    _extract_keypoints_from_row as _ref_extract_keypoints_from_row,
)
from hydra_suite.core.individual.pose.quality import (
    apply_quality_to_dataframe,
    apply_temporal_pose_postprocessing,
    assess_pose_row,
    calibrate_body_length_prior,
    calibrate_edge_length_priors,
    normalize_pose_keypoints_for_relink,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABELS = ["head", "thorax", "abdomen", "tail"]
_ANTERIOR = [0]  # head
_POSTERIOR = [3]  # tail


def _good_kpts(n: int = 4, conf: float = 0.9) -> np.ndarray:
    """Create a simple valid [n, 3] keypoints array."""
    kpts = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        kpts[i] = [float(i * 10), 0.0, conf]
    return kpts


def _make_pose_df(n_rows: int, labels=None, conf: float = 0.9) -> pd.DataFrame:
    """Build a small DataFrame with pose columns for n_rows rows."""
    if labels is None:
        labels = _LABELS
    data: dict = {"FrameID": list(range(n_rows))}
    for i, label in enumerate(labels):
        data[f"PoseKpt_{label}_X"] = [float(i * 10)] * n_rows
        data[f"PoseKpt_{label}_Y"] = [0.0] * n_rows
        data[f"PoseKpt_{label}_Conf"] = [conf] * n_rows
    data["PoseMeanConf"] = [conf] * n_rows
    data["PoseValidFraction"] = [1.0] * n_rows
    data["PoseQualityState"] = ["good"] * n_rows
    data["PoseQualityFlags"] = [""] * n_rows
    data["PoseWasCleaned"] = [0] * n_rows
    data["PoseSource"] = ["cache"] * n_rows
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# assess_pose_row
# ---------------------------------------------------------------------------


class TestAssessPoseRowInputValidation:
    def test_none_input_returns_rejected(self):
        result = assess_pose_row(None, 0.2, 0.5, 2)
        assert result.quality_state == "rejected"
        assert result.quality_score == 0.0
        assert "null_input" in result.quality_flags

    def test_all_nan_input_returns_rejected(self):
        kpts = np.full((4, 3), np.nan, dtype=np.float32)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        assert result.quality_state == "rejected"
        assert "all_nan" in result.quality_flags

    def test_wrong_shape_1d_returns_rejected(self):
        kpts = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        assert result.quality_state == "rejected"
        assert "invalid_shape" in result.quality_flags

    def test_wrong_shape_wrong_cols_returns_rejected(self):
        kpts = np.ones((4, 2), dtype=np.float32)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        assert result.quality_state == "rejected"


class TestAssessPoseRowAllLowConf:
    def test_all_low_conf_returns_rejected(self):
        kpts = _good_kpts(4, conf=0.05)  # all below threshold 0.2
        result = assess_pose_row(
            kpts, min_valid_conf=0.2, min_valid_fraction=0.5, min_valid_keypoints=2
        )
        assert result.quality_state == "rejected"
        assert "too_few_valid" in result.quality_flags
        assert result.quality_score == 0.0

    def test_all_low_conf_zeroes_confs(self):
        kpts = _good_kpts(4, conf=0.05)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        # All confs in cleaned output should be 0
        assert np.all(result.cleaned_keypoints[:, 2] == 0.0)

    def test_all_low_conf_flags_low_conf(self):
        kpts = _good_kpts(4, conf=0.05)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        # Should flag the low-conf keypoints before or alongside rejection
        has_low_conf = any(f.startswith("low_conf") for f in result.quality_flags)
        assert has_low_conf


class TestAssessPoseRowHealthy:
    def test_healthy_row_is_good(self):
        kpts = _good_kpts(4, conf=0.9)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        assert result.quality_state == "good"
        assert result.quality_score > 0.7
        assert result.was_cleaned is False

    def test_healthy_row_valid_mask(self):
        kpts = _good_kpts(4, conf=0.9)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        assert np.all(result.valid_mask)

    def test_healthy_row_cleaned_keypoints_unchanged(self):
        kpts = _good_kpts(4, conf=0.9)
        result = assess_pose_row(kpts, 0.2, 0.5, 2)
        np.testing.assert_array_almost_equal(result.cleaned_keypoints, kpts, decimal=5)

    def test_partial_row_partial_state(self):
        kpts = np.array(
            [[0.0, 0.0, 0.9], [10.0, 0.0, 0.9], [20.0, 0.0, 0.05], [30.0, 0.0, 0.05]],
            dtype=np.float32,
        )
        result = assess_pose_row(kpts, 0.2, 0.3, 2)
        # 2 valid out of 4 = 0.5 fraction; quality_score = 0.5 * 0.9 = 0.45 -> partial
        assert result.quality_state == "partial"

    def test_was_cleaned_set_when_bad_kpts_zeroed(self):
        kpts = np.array(
            [[0.0, 0.0, 0.9], [10.0, 0.0, 0.05]],
            dtype=np.float32,
        )
        result = assess_pose_row(kpts, 0.2, 0.3, 1)
        assert result.was_cleaned is True
        assert result.cleaned_keypoints[1, 2] == 0.0

    def test_xy_preserved_for_low_conf_kpts(self):
        kpts = np.array([[5.0, 7.0, 0.05]], dtype=np.float32)
        result = assess_pose_row(kpts, 0.2, 0.0, 0)
        # X/Y should be kept even when conf is zeroed
        assert result.cleaned_keypoints[0, 0] == pytest.approx(5.0)
        assert result.cleaned_keypoints[0, 1] == pytest.approx(7.0)
        assert result.cleaned_keypoints[0, 2] == pytest.approx(0.0)


class TestAssessPoseRowTooFewValid:
    def test_too_few_valid_fraction_rejected(self):
        kpts = np.array(
            [[0.0, 0.0, 0.9], [10.0, 0.0, 0.05], [20.0, 0.0, 0.05], [30.0, 0.0, 0.05]],
            dtype=np.float32,
        )
        # 1/4 = 0.25 valid fraction, min_valid_fraction=0.5 => rejected
        result = assess_pose_row(kpts, 0.2, 0.5, 1)
        assert result.quality_state == "rejected"

    def test_too_few_absolute_rejected(self):
        kpts = np.array(
            [[0.0, 0.0, 0.9], [10.0, 0.0, 0.9]],
            dtype=np.float32,
        )
        # 2 valid but min_valid_keypoints=3 => rejected
        result = assess_pose_row(kpts, 0.2, 0.0, 3)
        assert result.quality_state == "rejected"
        assert "too_few_valid" in result.quality_flags

    def test_rejected_all_confs_zeroed(self):
        kpts = np.array(
            [[0.0, 0.0, 0.9], [10.0, 0.0, 0.05], [20.0, 0.0, 0.05], [30.0, 0.0, 0.05]],
            dtype=np.float32,
        )
        result = assess_pose_row(kpts, 0.2, 0.5, 1)
        assert np.all(result.cleaned_keypoints[:, 2] == 0.0)


class TestAssessPoseRowBodyLengthOutlier:
    def _make_prior(self, median=100.0, mad=5.0) -> BodyLengthPrior:
        return BodyLengthPrior(
            median_px=median, mad_px=mad, n_samples=50, is_valid=True
        )

    def test_normal_body_length_no_flag(self):
        # head at (0,0), tail at (100,0) => body length = 100, median=100, mad=5 => z=0
        kpts = np.array(
            [[0.0, 0.0, 0.9], [50.0, 0.0, 0.9], [80.0, 0.0, 0.9], [100.0, 0.0, 0.9]],
            dtype=np.float32,
        )
        prior = self._make_prior(median=100.0, mad=5.0)
        result = assess_pose_row(
            kpts,
            0.2,
            0.3,
            2,
            body_length_prior=prior,
            anterior_indices=[0],
            posterior_indices=[3],
        )
        assert "body_length_outlier" not in result.quality_flags

    def test_outlier_body_length_flagged(self):
        # head at (0,0), tail at (1000,0) => body length=1000, median=100, mad=5 => huge z
        kpts = np.array(
            [[0.0, 0.0, 0.9], [333.0, 0.0, 0.9], [666.0, 0.0, 0.9], [1000.0, 0.0, 0.9]],
            dtype=np.float32,
        )
        prior = self._make_prior(median=100.0, mad=5.0)
        result = assess_pose_row(
            kpts,
            0.2,
            0.3,
            2,
            body_length_prior=prior,
            anterior_indices=[0],
            posterior_indices=[3],
        )
        assert "body_length_outlier" in result.quality_flags

    def test_outlier_body_length_applies_score_penalty(self):
        kpts = np.array(
            [[0.0, 0.0, 0.9], [333.0, 0.0, 0.9], [666.0, 0.0, 0.9], [1000.0, 0.0, 0.9]],
            dtype=np.float32,
        )
        prior = self._make_prior(median=100.0, mad=5.0)
        result_with_prior = assess_pose_row(
            kpts,
            0.2,
            0.3,
            2,
            body_length_prior=prior,
            anterior_indices=[0],
            posterior_indices=[3],
        )
        result_no_prior = assess_pose_row(kpts, 0.2, 0.3, 2)
        # With outlier penalty, score should be lower
        assert result_with_prior.quality_score < result_no_prior.quality_score

    def test_invalid_prior_not_used(self):
        """An is_valid=False prior must be ignored."""
        kpts = _good_kpts(4, conf=0.9)
        invalid_prior = BodyLengthPrior(
            median_px=100.0, mad_px=5.0, n_samples=5, is_valid=False
        )
        result = assess_pose_row(
            kpts,
            0.2,
            0.3,
            2,
            body_length_prior=invalid_prior,
            anterior_indices=[0],
            posterior_indices=[3],
        )
        assert "body_length_outlier" not in result.quality_flags


# ---------------------------------------------------------------------------
# calibrate_body_length_prior
# ---------------------------------------------------------------------------


class TestCalibrateBodyLengthPrior:
    def test_empty_df_returns_invalid_prior(self):
        df = pd.DataFrame()
        prior = calibrate_body_length_prior(df, _LABELS, _ANTERIOR, _POSTERIOR, 0.2)
        assert prior.is_valid is False
        assert prior.n_samples == 0

    def test_missing_anterior_returns_invalid(self):
        df = _make_pose_df(30)
        prior = calibrate_body_length_prior(df, _LABELS, [], _POSTERIOR, 0.2)
        assert prior.is_valid is False

    def test_missing_posterior_returns_invalid(self):
        df = _make_pose_df(30)
        prior = calibrate_body_length_prior(df, _LABELS, _ANTERIOR, [], 0.2)
        assert prior.is_valid is False

    def test_insufficient_samples_not_valid(self):
        """Fewer than 20 samples → is_valid=False."""
        df = _make_pose_df(10, conf=0.9)  # only 10 rows
        prior = calibrate_body_length_prior(df, _LABELS, _ANTERIOR, _POSTERIOR, 0.2)
        assert prior.n_samples <= 10
        assert prior.is_valid is False

    def test_sufficient_samples_valid(self):
        """20+ samples with high conf → is_valid=True."""
        df = _make_pose_df(30, conf=0.9)
        prior = calibrate_body_length_prior(df, _LABELS, _ANTERIOR, _POSTERIOR, 0.2)
        assert prior.is_valid is True
        assert prior.n_samples >= 20

    def test_body_length_correct(self):
        """head at x=0, tail at x=30 => body_length=30."""
        labels = ["head", "tail"]
        n = 30
        df = pd.DataFrame(
            {
                "FrameID": list(range(n)),
                "PoseKpt_head_X": [0.0] * n,
                "PoseKpt_head_Y": [0.0] * n,
                "PoseKpt_head_Conf": [0.9] * n,
                "PoseKpt_tail_X": [30.0] * n,
                "PoseKpt_tail_Y": [0.0] * n,
                "PoseKpt_tail_Conf": [0.9] * n,
                "PoseMeanConf": [0.9] * n,
            }
        )
        prior = calibrate_body_length_prior(df, labels, [0], [1], 0.2)
        assert prior.is_valid is True
        assert prior.median_px == pytest.approx(30.0, abs=1.0)

    def test_high_conf_floor_filters_low_conf_rows(self):
        """Rows below high_conf_floor=0.7 are excluded from calibration."""
        labels = ["head", "tail"]
        n = 40
        # Half rows have high conf, half have low conf
        confs = [0.9 if i < 20 else 0.3 for i in range(n)]
        df = pd.DataFrame(
            {
                "FrameID": list(range(n)),
                "PoseKpt_head_X": [0.0] * n,
                "PoseKpt_head_Y": [0.0] * n,
                "PoseKpt_head_Conf": confs,
                "PoseKpt_tail_X": [30.0] * n,
                "PoseKpt_tail_Y": [0.0] * n,
                "PoseKpt_tail_Conf": confs,
                "PoseMeanConf": confs,
            }
        )
        prior = calibrate_body_length_prior(
            df, labels, [0], [1], 0.2, high_conf_floor=0.7
        )
        # Should only use the 20 high-conf rows
        assert prior.n_samples == 20
        assert prior.is_valid is True


# ---------------------------------------------------------------------------
# normalize_pose_keypoints_for_relink
# ---------------------------------------------------------------------------


class TestNormalizePoseKeypointsForRelink:
    def test_none_window_returns_none(self):
        result, vis = normalize_pose_keypoints_for_relink(None, _LABELS, 0.2)
        assert result is None
        assert vis == 0.0

    def test_empty_window_returns_none(self):
        result, vis = normalize_pose_keypoints_for_relink(pd.DataFrame(), _LABELS, 0.2)
        assert result is None
        assert vis == 0.0

    def test_no_labels_returns_none(self):
        df = _make_pose_df(1)
        result, vis = normalize_pose_keypoints_for_relink(df, [], 0.2)
        assert result is None
        assert vis == 0.0

    def test_single_row_returns_normalized(self):
        df = _make_pose_df(1, conf=0.9)
        result, vis = normalize_pose_keypoints_for_relink(df, _LABELS, 0.2)
        assert result is not None
        assert result.shape == (len(_LABELS), 3)
        assert vis > 0.0

    def test_three_row_window_returns_normalized(self):
        df = _make_pose_df(3, conf=0.9)
        result, vis = normalize_pose_keypoints_for_relink(df, _LABELS, 0.2)
        assert result is not None
        assert result.shape == (len(_LABELS), 3)
        assert vis == pytest.approx(1.0, abs=0.01)

    def test_all_low_conf_in_window_returns_none(self):
        df = _make_pose_df(2, conf=0.05)
        result, vis = normalize_pose_keypoints_for_relink(df, _LABELS, 0.2)
        assert result is None
        assert vis == 0.0

    def test_visibility_partial(self):
        """Only 2 of 4 labels above conf threshold => visibility = 0.5."""
        labels = ["a", "b", "c", "d"]
        df = pd.DataFrame(
            {
                "PoseKpt_a_X": [0.0],
                "PoseKpt_a_Y": [0.0],
                "PoseKpt_a_Conf": [0.9],
                "PoseKpt_b_X": [10.0],
                "PoseKpt_b_Y": [0.0],
                "PoseKpt_b_Conf": [0.9],
                "PoseKpt_c_X": [20.0],
                "PoseKpt_c_Y": [0.0],
                "PoseKpt_c_Conf": [0.05],
                "PoseKpt_d_X": [30.0],
                "PoseKpt_d_Y": [0.0],
                "PoseKpt_d_Conf": [0.05],
            }
        )
        result, vis = normalize_pose_keypoints_for_relink(df, labels, 0.2)
        assert result is not None
        assert vis == pytest.approx(0.5, abs=0.01)

    def test_output_dtype_float32(self):
        df = _make_pose_df(2, conf=0.9)
        result, _ = normalize_pose_keypoints_for_relink(df, _LABELS, 0.2)
        assert result is not None
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# apply_quality_to_dataframe
# ---------------------------------------------------------------------------


class TestApplyQualityToDataframe:
    _PARAMS = {
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "POSE_EXPORT_MIN_VALID_FRACTION": 0.5,
        "POSE_EXPORT_MIN_VALID_KEYPOINTS": 2,
        "POSE_IGNORE_KEYPOINTS": [],
    }

    def test_none_df_returns_none(self):
        result = apply_quality_to_dataframe(None, _LABELS, self._PARAMS)
        assert result is None

    def test_empty_df_returned_unchanged(self):
        df = pd.DataFrame()
        result = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        assert result.empty

    def test_adds_quality_columns(self):
        df = _make_pose_df(5, conf=0.9)
        # Remove quality columns to test they are added
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])
        result = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_good_rows_get_good_state(self):
        df = _make_pose_df(5, conf=0.9)
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])
        result = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        assert (result["PoseQualityState"] == "good").all()

    def test_rows_with_no_pose_get_no_pose_state(self):
        # Build df with no pose columns at all
        df = pd.DataFrame({"FrameID": [0, 1, 2]})
        result = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        assert (result["PoseQualityState"] == "no_pose").all()

    def test_low_conf_rows_rejected_and_confs_zeroed(self):
        df = _make_pose_df(3, conf=0.05)  # all below threshold
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])
        result = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        assert (result["PoseQualityState"] == "rejected").all()
        for label in _LABELS:
            c_col = f"PoseKpt_{label}_Conf"
            if c_col in result.columns:
                assert (result[c_col] == 0.0).all()

    def test_source_set_to_cache_for_pose_rows(self):
        df = _make_pose_df(3, conf=0.9)
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])
        result = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        assert (result["PoseSource"] == "cache").all()

    def test_with_body_length_prior_smoke(self):
        df = _make_pose_df(5, conf=0.9)
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])
        prior = BodyLengthPrior(median_px=30.0, mad_px=2.0, n_samples=50, is_valid=True)
        result = apply_quality_to_dataframe(
            df,
            _LABELS,
            self._PARAMS,
            body_length_prior=prior,
            anterior_indices=_ANTERIOR,
            posterior_indices=_POSTERIOR,
        )
        assert "PoseQualityState" in result.columns
        # Should not crash; states should be valid strings
        assert result["PoseQualityState"].notna().all()

    def test_does_not_modify_input_df(self):
        df = _make_pose_df(3, conf=0.9)
        original_vals = df["PoseKpt_head_Conf"].copy()
        for col in [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]:
            if col in df.columns:
                df = df.drop(columns=[col])
        _ = apply_quality_to_dataframe(df, _LABELS, self._PARAMS)
        # Input df should be unchanged
        pd.testing.assert_series_equal(df["PoseKpt_head_Conf"], original_vals)


# ---------------------------------------------------------------------------
# Frozen pre-vectorization reference of apply_quality_to_dataframe
#
# This is a literal copy of the row-loop implementation as it existed BEFORE
# Task 7's vectorization, built only from helpers that Task 7 does NOT touch
# (``_extract_keypoints_from_row`` and ``assess_pose_row``). It is the
# non-tautological oracle for the characterization test below: it must match
# the CURRENT (pre-vectorization) code exactly, and must continue to match
# the vectorized code after the change.
# ---------------------------------------------------------------------------


def _reference_apply_quality_to_dataframe(
    df,
    pose_labels,
    params,
    body_length_prior=None,
    anterior_indices=None,
    posterior_indices=None,
    skeleton_edges=None,
    edge_length_priors=None,
):
    if df is None or df.empty:
        return df

    min_valid_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
    min_valid_fraction = float(params.get("POSE_EXPORT_MIN_VALID_FRACTION", 0.5))
    min_valid_keypoints = int(params.get("POSE_EXPORT_MIN_VALID_KEYPOINTS", 3))
    ignore_kpts_raw = params.get("POSE_IGNORE_KEYPOINTS", [])
    ignore_indices = [int(i) for i in (ignore_kpts_raw or [])]

    col_triplets = []
    for label in pose_labels:
        col_triplets.append(
            (f"PoseKpt_{label}_X", f"PoseKpt_{label}_Y", f"PoseKpt_{label}_Conf")
        )

    out = df.copy()

    out["PoseQualityScore"] = float("nan")
    out["PoseQualityState"] = ""
    out["PoseQualityFlags"] = ""
    out["PoseSource"] = ""
    out["PoseWasCleaned"] = 0

    all_conf_cols = [c for _, _, c in col_triplets if c in out.columns]

    for row_idx in out.index:
        row = out.loc[row_idx]

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

        kpts = _ref_extract_keypoints_from_row(row, pose_labels)

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


_CHAR_LABELS = [f"p{i}" for i in range(10)]


def _char_row(overrides=None):
    """Baseline fully-valid 10-keypoint row: p0=anterior, p9=posterior,
    edge (p3,p4) distance ~10, body length (p0-p9) distance ~30."""
    vals = {
        0: (0.0, 0.0, 0.9),
        1: (1.0, 0.0, 0.9),
        2: (2.0, 0.0, 0.9),
        3: (0.0, 0.0, 0.9),
        4: (10.0, 0.0, 0.9),
        5: (5.0, 0.0, 0.9),
        6: (6.0, 0.0, 0.9),
        7: (7.0, 0.0, 0.9),
        8: (8.0, 0.0, 0.9),
        9: (30.0, 0.0, 0.9),
    }
    if overrides:
        vals.update(overrides)
    row = {}
    for i in range(10):
        x, y, c = vals[i]
        row[f"PoseKpt_p{i}_X"] = x
        row[f"PoseKpt_p{i}_Y"] = y
        row[f"PoseKpt_p{i}_Conf"] = c
    return row


def _build_characterization_df():
    rows = []
    row_names = []

    # 1. fully valid, clean, no flags
    rows.append(_char_row())
    row_names.append("good_all_valid_clean")

    # 2. low_conf:2
    rows.append(_char_row({1: (1.0, 0.0, 0.1), 2: (2.0, 0.0, 0.1)}))
    row_names.append("low_conf_2kpts")

    # 3. invalid_coords:1
    rows.append(_char_row({2: (float("nan"), 0.0, 0.9)}))
    row_names.append("invalid_coords_1kpt")

    # 4. rejected (too_few_valid), all non-ignored confs zeroed incl. previously-valid
    overrides = {i: (float(i), 0.0, 0.05) for i in range(1, 9)}
    rows.append(_char_row(overrides))
    row_names.append("rejected_too_few")

    # 5. boundary partial near 0.2 (score=0.204)
    ov5 = {}
    base = _char_row()
    for i in range(10):
        x = base[f"PoseKpt_p{i}_X"]
        y = base[f"PoseKpt_p{i}_Y"]
        if i in (0, 5, 6, 9):
            ov5[i] = (x, y, 0.51)
        else:
            ov5[i] = (x, y, 0.1)
    rows.append(_char_row(ov5))
    row_names.append("boundary_partial_0.2")

    # 6. boundary bad near 0.2 (score=0.196)
    ov6 = dict(ov5)
    for i in (0, 5, 6, 9):
        x, y, _ = ov6[i]
        ov6[i] = (x, y, 0.49)
    rows.append(_char_row(ov6))
    row_names.append("boundary_bad_0.2")

    # 7. boundary partial near 0.7 (score=0.688)
    ov7 = {}
    for i in range(10):
        x = base[f"PoseKpt_p{i}_X"]
        y = base[f"PoseKpt_p{i}_Y"]
        if i in (1, 2):
            ov7[i] = (x, y, 0.1)
        else:
            ov7[i] = (x, y, 0.86)
    rows.append(_char_row(ov7))
    row_names.append("boundary_partial_0.7")

    # 8. boundary good near 0.7 (score=0.704)
    ov8 = dict(ov7)
    for i in range(10):
        if i not in (1, 2):
            x, y, _ = ov8[i]
            ov8[i] = (x, y, 0.88)
    rows.append(_char_row(ov8))
    row_names.append("boundary_good_0.7")

    # 9. body_length_outlier flagged
    rows.append(_char_row({9: (1000.0, 0.0, 0.9)}))
    row_names.append("body_length_outlier_flagged")

    # 10. edge_outlier flagged
    rows.append(_char_row({4: (500.0, 0.0, 0.9)}))
    row_names.append("edge_outlier_flagged")

    # 11. multiple flags co-occurring (token order)
    rows.append(
        _char_row(
            {
                1: (1.0, 0.0, 0.1),  # low_conf
                2: (float("nan"), 0.0, 0.9),  # invalid_coords
                9: (1000.0, 0.0, 0.9),  # body_length_outlier
                4: (500.0, 0.0, 0.9),  # edge_outlier
            }
        )
    )
    row_names.append("multi_flag_order")

    # 12. no_pose: all conf NaN
    ov12 = {i: (float("nan"), float("nan"), float("nan")) for i in range(10)}
    rows.append(_char_row(ov12))
    row_names.append("no_pose")

    # 13. all_nan edge case: has_pose True (conf=inf, not NaN) but nothing finite
    ov13 = {i: (float("inf"), float("inf"), float("inf")) for i in range(10)}
    rows.append(_char_row(ov13))
    row_names.append("all_nan_edge_case")

    # 14. null_input edge case: conf non-NaN but X is a non-numeric string
    # for every keypoint -> extraction fails entirely despite has_pose True.
    ov14 = {}
    for i in range(10):
        ov14[i] = ("bad", 0.0, 0.9)
    row14 = {}
    for i in range(10):
        x, y, c = ov14[i]
        row14[f"PoseKpt_p{i}_X"] = x
        row14[f"PoseKpt_p{i}_Y"] = y
        row14[f"PoseKpt_p{i}_Conf"] = c
    rows.append(row14)
    row_names.append("null_input_edge_case")

    df = pd.DataFrame(rows)
    df.insert(0, "FrameID", list(range(len(rows))))
    df.insert(1, "RowName", row_names)
    return df


_CHAR_PARAMS = {
    "POSE_MIN_KPT_CONF_VALID": 0.2,
    "POSE_EXPORT_MIN_VALID_FRACTION": 0.3,
    "POSE_EXPORT_MIN_VALID_KEYPOINTS": 2,
    "POSE_IGNORE_KEYPOINTS": [],
}
_CHAR_ANTERIOR = [0]
_CHAR_POSTERIOR = [9]
_CHAR_SKELETON_EDGES = [(3, 4)]
_CHAR_BODY_LENGTH_PRIOR = BodyLengthPrior(
    median_px=30.0, mad_px=2.0, n_samples=50, is_valid=True
)
_CHAR_EDGE_PRIORS = EdgeLengthPriors(
    priors={(3, 4): {"median_px": 10.0, "mad_px": 1.0, "n_samples": 25}},
    is_valid=True,
)


class TestApplyQualityToDataframeCharacterization:
    """Non-tautological byte-identity oracle for Task 7's vectorization.

    Compares the LIVE ``apply_quality_to_dataframe`` against the FROZEN
    pre-vectorization reference implementation above, across a DataFrame
    exercising every branch: clean/low_conf/invalid_coords, valid-fraction
    rejection (incl. zeroing of previously-valid confs), score/state
    thresholds straddling 0.2 and 0.7, body-length outlier, edge outlier,
    multiple co-occurring flags (token order), no_pose, and the two
    keypoint-extraction edge cases (all_nan / null_input).
    """

    def _run_both(self):
        df = _build_characterization_df()
        reference = _reference_apply_quality_to_dataframe(
            df,
            _CHAR_LABELS,
            _CHAR_PARAMS,
            body_length_prior=_CHAR_BODY_LENGTH_PRIOR,
            anterior_indices=_CHAR_ANTERIOR,
            posterior_indices=_CHAR_POSTERIOR,
            skeleton_edges=_CHAR_SKELETON_EDGES,
            edge_length_priors=_CHAR_EDGE_PRIORS,
        )
        live = apply_quality_to_dataframe(
            df,
            _CHAR_LABELS,
            _CHAR_PARAMS,
            body_length_prior=_CHAR_BODY_LENGTH_PRIOR,
            anterior_indices=_CHAR_ANTERIOR,
            posterior_indices=_CHAR_POSTERIOR,
            skeleton_edges=_CHAR_SKELETON_EDGES,
            edge_length_priors=_CHAR_EDGE_PRIORS,
        )
        return df, reference, live

    def test_reference_matches_live_exact(self):
        df, reference, live = self._run_both()
        pd.testing.assert_frame_equal(
            reference, live, check_exact=True, check_dtype=True
        )

    def test_reference_matches_live_csv_text(self):
        """Catch write-boundary formatting divergence (float32 round-trip,
        flag-string order) that DataFrame equality alone might not surface."""
        df, reference, live = self._run_both()
        conf_cols = [f"PoseKpt_p{i}_Conf" for i in range(10)]
        meta_cols = [
            "PoseQualityScore",
            "PoseQualityState",
            "PoseQualityFlags",
            "PoseSource",
            "PoseWasCleaned",
        ]
        cols = conf_cols + meta_cols
        ref_csv = reference[cols].to_csv(index=False)
        live_csv = live[cols].to_csv(index=False)
        assert ref_csv == live_csv

    def test_expected_states_and_flags(self):
        df, reference, live = self._run_both()
        expected_state = {
            "good_all_valid_clean": "good",
            "low_conf_2kpts": "good",
            "invalid_coords_1kpt": "good",
            "rejected_too_few": "rejected",
            "boundary_partial_0.2": "partial",
            "boundary_bad_0.2": "bad",
            "boundary_partial_0.7": "partial",
            "boundary_good_0.7": "good",
            "body_length_outlier_flagged": "partial",
            "edge_outlier_flagged": "good",
            "multi_flag_order": "partial",
            "no_pose": "no_pose",
            "all_nan_edge_case": "rejected",
            "null_input_edge_case": "rejected",
        }
        expected_flags = {
            "good_all_valid_clean": "",
            "low_conf_2kpts": "low_conf:2",
            "invalid_coords_1kpt": "invalid_coords:1",
            "rejected_too_few": "low_conf:8|too_few_valid",
            "boundary_partial_0.2": "low_conf:6",
            "boundary_bad_0.2": "low_conf:6",
            "boundary_partial_0.7": "low_conf:2",
            "boundary_good_0.7": "low_conf:2",
            "body_length_outlier_flagged": "body_length_outlier",
            "edge_outlier_flagged": "edge_outlier:1",
            "multi_flag_order": "low_conf:1|invalid_coords:1|body_length_outlier|edge_outlier:1",
            "no_pose": "",
            "all_nan_edge_case": "all_nan",
            "null_input_edge_case": "null_input",
        }
        for name, exp_state in expected_state.items():
            i = df.index[df["RowName"] == name][0]
            assert (
                live.loc[i, "PoseQualityState"] == exp_state
            ), f"{name}: state {live.loc[i, 'PoseQualityState']!r} != {exp_state!r}"
            assert (
                live.loc[i, "PoseQualityFlags"] == expected_flags[name]
            ), f"{name}: flags {live.loc[i, 'PoseQualityFlags']!r} != {expected_flags[name]!r}"

    def test_null_input_leaves_conf_columns_untouched(self):
        df, reference, live = self._run_both()
        i = df.index[df["RowName"] == "null_input_edge_case"][0]
        for label in _CHAR_LABELS:
            assert live.loc[i, f"PoseKpt_{label}_Conf"] == 0.9
        assert live.loc[i, "PoseWasCleaned"] == 1

    def test_all_nan_zeroes_all_conf_columns(self):
        df, reference, live = self._run_both()
        i = df.index[df["RowName"] == "all_nan_edge_case"][0]
        for label in _CHAR_LABELS:
            assert live.loc[i, f"PoseKpt_{label}_Conf"] == 0.0

    def test_rejected_zeroes_previously_valid_confs(self):
        df, reference, live = self._run_both()
        i = df.index[df["RowName"] == "rejected_too_few"][0]
        for label in _CHAR_LABELS:
            assert live.loc[i, f"PoseKpt_{label}_Conf"] == 0.0

    def test_was_cleaned_false_when_only_geometry_flag(self):
        df, reference, live = self._run_both()
        i = df.index[df["RowName"] == "body_length_outlier_flagged"][0]
        assert live.loc[i, "PoseWasCleaned"] == 0
        i2 = df.index[df["RowName"] == "edge_outlier_flagged"][0]
        assert live.loc[i2, "PoseWasCleaned"] == 0


class TestApplyQualityToDataframeIgnoreIndices:
    """Ignored keypoints must be excluded from cleaning, counting, and
    the valid-fraction denominator, and must never be zeroed."""

    def test_ignored_keypoint_untouched_and_not_counted(self):
        labels = ["a", "b", "c", "d"]
        df = pd.DataFrame(
            {
                "PoseKpt_a_X": [0.0],
                "PoseKpt_a_Y": [0.0],
                "PoseKpt_a_Conf": [0.9],
                "PoseKpt_b_X": [1.0],
                "PoseKpt_b_Y": [0.0],
                "PoseKpt_b_Conf": [0.05],  # would be low_conf, but ignored
                "PoseKpt_c_X": [2.0],
                "PoseKpt_c_Y": [0.0],
                "PoseKpt_c_Conf": [0.9],
                "PoseKpt_d_X": [3.0],
                "PoseKpt_d_Y": [0.0],
                "PoseKpt_d_Conf": [0.9],
            }
        )
        params = {
            "POSE_MIN_KPT_CONF_VALID": 0.2,
            "POSE_EXPORT_MIN_VALID_FRACTION": 0.5,
            "POSE_EXPORT_MIN_VALID_KEYPOINTS": 2,
            "POSE_IGNORE_KEYPOINTS": [1],
        }
        reference = _reference_apply_quality_to_dataframe(df, labels, params)
        live = apply_quality_to_dataframe(df, labels, params)
        pd.testing.assert_frame_equal(
            reference, live, check_exact=True, check_dtype=True
        )
        assert live.loc[0, "PoseQualityState"] == "good"
        assert live.loc[0, "PoseQualityFlags"] == ""
        assert live.loc[0, "PoseKpt_b_Conf"] == pytest.approx(0.05)  # untouched
        assert live.loc[0, "PoseWasCleaned"] == 0


# ---------------------------------------------------------------------------
# apply_temporal_pose_postprocessing
# ---------------------------------------------------------------------------


class TestApplyTemporalPosePostprocessing:
    def test_empty_df_returned_unchanged(self):
        df = pd.DataFrame()
        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=3, z_score_threshold=3.0
        )
        assert result.empty

    def test_no_labels_returned_unchanged(self):
        df = _make_pose_df(5)
        result = apply_temporal_pose_postprocessing(
            df, [], max_gap=3, z_score_threshold=3.0
        )
        assert len(result) == len(df)

    def test_outlier_suppression(self):
        """A single spike in X should be detected and zeroed (or gap-filled at low trust).

        The pipeline first zeroes the outlier conf, then gap-fill may re-introduce
        the position at conf=0.3 if the gap is within max_gap.  Either way the
        PoseWasCleaned flag must be set and conf must not be the original 0.9.
        """
        n = 20
        df = _make_pose_df(n, conf=0.9)
        # Inject a spike in head_X at row 10
        spike_idx = df.index[10]
        df.at[spike_idx, "PoseKpt_head_X"] = 9999.0

        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=3, z_score_threshold=2.0
        )
        # PoseWasCleaned must be set — the row was touched by the pipeline
        assert result.at[spike_idx, "PoseWasCleaned"] == 1
        # conf must not be the original high value (either 0.0 or 0.3 after gap-fill)
        conf_after = result.at[spike_idx, "PoseKpt_head_Conf"]
        assert conf_after != pytest.approx(
            0.9
        ), f"Expected conf to be modified from 0.9, got {conf_after}"

    def test_outlier_flag_added(self):
        """Temporal outlier rows should have the flag appended."""
        n = 20
        df = _make_pose_df(n, conf=0.9)
        spike_idx = df.index[10]
        df.at[spike_idx, "PoseKpt_head_X"] = 9999.0

        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=3, z_score_threshold=2.0
        )
        flags = str(result.at[spike_idx, "PoseQualityFlags"])
        assert "temporal_outlier" in flags

    def test_gap_fill_within_max_gap(self):
        """A gap of 1 frame (within max_gap=3) should be interpolated."""
        n = 10
        df = _make_pose_df(n, conf=0.9)
        # Zero out the middle row's conf to simulate a gap
        gap_idx = df.index[5]
        df.at[gap_idx, "PoseKpt_head_Conf"] = 0.0
        df.at[gap_idx, "PoseQualityState"] = "bad"

        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=3, z_score_threshold=10.0, fill_interpolated=True
        )
        # The gap row should have been filled with conf=0.3
        assert result.at[gap_idx, "PoseKpt_head_Conf"] == pytest.approx(0.3)
        assert result.at[gap_idx, "PoseSource"] == "cleaned"
        assert result.at[gap_idx, "PoseWasCleaned"] == 1

    def test_gap_too_large_not_filled(self):
        """A gap larger than max_gap should NOT be filled."""
        n = 20
        df = _make_pose_df(n, conf=0.9)
        # Zero out rows 5..9 (5 consecutive rows) => gap_length=5 > max_gap=3
        for i in range(5, 10):
            df.at[df.index[i], "PoseKpt_head_Conf"] = 0.0
            df.at[df.index[i], "PoseQualityState"] = "bad"

        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=3, z_score_threshold=10.0, fill_interpolated=True
        )
        # All gap rows should still have conf=0 (not filled)
        for i in range(5, 10):
            assert result.at[result.index[i], "PoseKpt_head_Conf"] == pytest.approx(
                0.0
            ), f"Row {i} should not have been filled"

    def test_fill_interpolated_false_skips_gap_fill(self):
        """When fill_interpolated=False, gaps should not be filled."""
        n = 10
        df = _make_pose_df(n, conf=0.9)
        gap_idx = df.index[5]
        df.at[gap_idx, "PoseKpt_head_Conf"] = 0.0
        df.at[gap_idx, "PoseQualityState"] = "bad"

        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=3, z_score_threshold=10.0, fill_interpolated=False
        )
        # Should NOT be filled
        assert result.at[gap_idx, "PoseKpt_head_Conf"] == pytest.approx(0.0)

    def test_gap_fill_interpolates_xy(self):
        """Gap-filled X/Y should be linearly interpolated between neighbours."""
        n = 5
        # head at x=0 at frame 0, x=40 at frame 4; zero out frames 1,2,3
        labels = ["head"]
        data = {
            "FrameID": list(range(n)),
            "PoseKpt_head_X": [0.0, 10.0, 20.0, 30.0, 40.0],
            "PoseKpt_head_Y": [0.0] * n,
            "PoseKpt_head_Conf": [0.9, 0.0, 0.0, 0.0, 0.9],
            "PoseMeanConf": [0.9, 0.0, 0.0, 0.0, 0.9],
            "PoseValidFraction": [1.0, 0.0, 0.0, 0.0, 1.0],
            "PoseQualityState": ["good", "bad", "bad", "bad", "good"],
            "PoseQualityFlags": ["", "", "", "", ""],
            "PoseWasCleaned": [0, 0, 0, 0, 0],
            "PoseSource": ["cache"] * n,
        }
        df = pd.DataFrame(data)
        result = apply_temporal_pose_postprocessing(
            df, labels, max_gap=5, z_score_threshold=10.0, fill_interpolated=True
        )
        # Frame 2 (middle) should be interpolated to x=20
        assert result.iloc[2]["PoseKpt_head_X"] == pytest.approx(20.0, abs=1.0)

    def test_sort_by_frame_id(self):
        """Function should handle unsorted input."""
        df = _make_pose_df(5, conf=0.9)
        df = df.iloc[::-1].reset_index(drop=True)  # reverse order
        result = apply_temporal_pose_postprocessing(
            df, _LABELS, max_gap=2, z_score_threshold=5.0
        )
        assert len(result) == 5


# ---------------------------------------------------------------------------
# calibrate_edge_length_priors + EdgeLengthPriors in assess_pose_row
# ---------------------------------------------------------------------------


def _edge_df(n_rows: int = 30) -> pd.DataFrame:
    """Build a minimal DataFrame with two keypoints connected by an edge."""
    rows = []
    for _ in range(n_rows):
        rows.append(
            {
                "PoseKpt_head_X": 10.0,
                "PoseKpt_head_Y": 10.0,
                "PoseKpt_head_Conf": 0.9,
                "PoseKpt_tail_X": 10.0,
                "PoseKpt_tail_Y": 30.0,  # distance = 20 px
                "PoseKpt_tail_Conf": 0.9,
                "PoseMeanConf": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_calibrate_edge_length_priors_returns_invalid_for_empty_df():
    result = calibrate_edge_length_priors(
        pd.DataFrame(), ["head", "tail"], [(0, 1)], 0.2
    )
    assert not result.is_valid
    assert result.priors == {}


def test_calibrate_edge_length_priors_returns_invalid_for_no_edges():
    df = _edge_df()
    result = calibrate_edge_length_priors(df, ["head", "tail"], [], 0.2)
    assert not result.is_valid


def test_calibrate_edge_length_priors_correct_median():
    df = _edge_df(30)
    result = calibrate_edge_length_priors(df, ["head", "tail"], [(0, 1)], 0.2)
    assert result.is_valid
    key = (0, 1)
    assert key in result.priors
    assert result.priors[key]["median_px"] == pytest.approx(20.0, abs=0.5)
    assert result.priors[key]["n_samples"] == 30


def test_calibrate_edge_length_priors_invalid_below_20_samples():
    df = _edge_df(10)  # only 10 rows → not enough samples
    result = calibrate_edge_length_priors(df, ["head", "tail"], [(0, 1)], 0.2)
    assert not result.is_valid
    assert result.priors[(0, 1)]["n_samples"] == 10


def test_calibrate_edge_length_priors_canonical_key_ordering():
    """Edge (1, 0) and (0, 1) should produce the same canonical key."""
    df = _edge_df(30)
    r1 = calibrate_edge_length_priors(df, ["head", "tail"], [(0, 1)], 0.2)
    r2 = calibrate_edge_length_priors(df, ["head", "tail"], [(1, 0)], 0.2)
    assert r1.priors == r2.priors


def test_assess_pose_row_edge_outlier_flag():
    """A keypoint arrangement that breaks the skeleton edge prior should be flagged."""
    prior = EdgeLengthPriors(
        priors={(0, 1): {"median_px": 20.0, "mad_px": 1.0, "n_samples": 30}},
        is_valid=True,
    )
    # distance = 200 px; z = (200 - 20) / 1 = 180 >> threshold
    kpts = np.array([[10.0, 10.0, 0.9], [10.0, 210.0, 0.9]], dtype=np.float32)
    result = assess_pose_row(
        kpts,
        min_valid_conf=0.2,
        min_valid_fraction=0.5,
        min_valid_keypoints=1,
        skeleton_edges=[(0, 1)],
        edge_length_priors=prior,
        edge_length_z_threshold=4.0,
    )
    assert any("edge_outlier" in f for f in result.quality_flags)
    # Quality score penalised
    assert result.quality_score < 0.9 * 0.9  # lower than unflagged would be


def test_assess_pose_row_normal_edge_no_flag():
    """An edge within the expected range should not trigger the outlier flag."""
    prior = EdgeLengthPriors(
        priors={(0, 1): {"median_px": 20.0, "mad_px": 2.0, "n_samples": 30}},
        is_valid=True,
    )
    # distance = 21 px; z = (21 - 20) / 2 = 0.5 << threshold
    kpts = np.array([[10.0, 10.0, 0.9], [10.0, 31.0, 0.9]], dtype=np.float32)
    result = assess_pose_row(
        kpts,
        min_valid_conf=0.2,
        min_valid_fraction=0.5,
        min_valid_keypoints=1,
        skeleton_edges=[(0, 1)],
        edge_length_priors=prior,
        edge_length_z_threshold=4.0,
    )
    assert not any("edge_outlier" in f for f in result.quality_flags)


def test_assess_pose_row_edge_skipped_when_prior_invalid():
    """Edge priors that are not valid should not affect quality scoring."""
    prior = EdgeLengthPriors(priors={}, is_valid=False)
    kpts = np.array([[10.0, 10.0, 0.9], [10.0, 210.0, 0.9]], dtype=np.float32)
    result = assess_pose_row(
        kpts,
        min_valid_conf=0.2,
        min_valid_fraction=0.5,
        min_valid_keypoints=1,
        skeleton_edges=[(0, 1)],
        edge_length_priors=prior,
    )
    assert not any("edge_outlier" in f for f in result.quality_flags)


def test_assess_pose_row_edge_skipped_when_keypoint_invalid():
    """Edge check is skipped for keypoints that failed the confidence gate."""
    prior = EdgeLengthPriors(
        priors={(0, 1): {"median_px": 20.0, "mad_px": 1.0, "n_samples": 30}},
        is_valid=True,
    )
    # Tail keypoint has low confidence — edge check should be skipped
    kpts = np.array([[10.0, 10.0, 0.9], [10.0, 210.0, 0.05]], dtype=np.float32)
    result = assess_pose_row(
        kpts,
        min_valid_conf=0.2,
        min_valid_fraction=0.0,  # allow partial so row is not rejected first
        min_valid_keypoints=0,
        skeleton_edges=[(0, 1)],
        edge_length_priors=prior,
        edge_length_z_threshold=4.0,
    )
    assert not any("edge_outlier" in f for f in result.quality_flags)
