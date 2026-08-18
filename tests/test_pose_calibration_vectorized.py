"""Byte-identity characterization guard for Task 6 (vectorize pose calibration).

This test freezes a literal copy of the PRE-vectorization ``iterrows()`` based
implementations of ``_collect_body_lengths`` and ``_accumulate_edge_samples``
(as they existed in ``quality.py`` before this task) as a non-tautological
oracle. It builds a realistic, branch-covering high-confidence pose
DataFrame, captures the reference implementation's output, and — after
vectorization — asserts the real (module) implementations reproduce the
reference sample sequences (and downstream median/MAD priors) with EXACT
elementwise equality (including NaN handling and item order).

Branch coverage exercised by the fixture DataFrame:
  - Multiple animals/frames (>= 20 high-conf rows, for is_valid=True priors)
  - Rows below the high_conf_floor (must be filtered out by
    ``_filter_high_conf_rows`` before either target function runs)
  - Varied per-keypoint confidences straddling ``min_valid_conf`` (some
    keypoints valid, some not, within otherwise-valid rows)
  - NaN coordinates on some keypoints (invalidates that keypoint for both
    weighted-centroid and edge-distance purposes)
  - NaN confidence on an edge endpoint (edge-distance gate has NO explicit
    isfinite(conf) check, so NaN conf must NOT block the edge sample --
    "NaN < threshold" is False in Python/NumPy)
  - NaN confidence on a weighted-centroid keypoint (weighted centroid DOES
    require isfinite(conf), so this must exclude the keypoint)
  - A duplicate/reversed edge pair ((tail, head) vs (head, tail)) mapping to
    the same canonical key, to test multi-edge-same-key accumulation order
  - An edge referencing an out-of-range keypoint index (skipped entirely)
  - A malformed edge entry that raises on ``int()`` (skipped entirely)
  - A row with zero body length (ant == post) to test the ``bl > 0`` filter
  - A pose-label column present in the DataFrame but with a non-numeric
    (object-dtype) cell for one row, to exercise the ``float(x)`` raising a
    ValueError/TypeError fallback path in keypoint extraction
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.individual.pose.features import (
    compute_pose_geometry_from_keypoints,
)
from hydra_suite.core.individual.pose.quality import (
    _accumulate_edge_samples,
    _body_length_prior_from_samples,
    _build_edge_priors,
    _collect_body_lengths,
    _extract_keypoints_from_row,
    _filter_high_conf_rows,
    _measure_edge_distance,
)

# ---------------------------------------------------------------------------
# Frozen reference implementations (literal copy of pre-vectorization code)
# ---------------------------------------------------------------------------


def _reference_collect_body_lengths(
    high_conf_df: pd.DataFrame,
    pose_labels: List[str],
    anterior_indices: List[int],
    posterior_indices: List[int],
    min_valid_conf: float,
) -> List[float]:
    body_lengths: List[float] = []
    for _, row in high_conf_df.iterrows():
        kpts = _extract_keypoints_from_row(row, pose_labels)
        if kpts is None:
            continue
        geom = compute_pose_geometry_from_keypoints(
            kpts,
            anterior_indices,
            posterior_indices,
            min_valid_conf,
        )
        if geom is None:
            continue
        bl = geom.get("body_length")
        if bl is not None and float(bl) > 0.0:
            body_lengths.append(float(bl))
    return body_lengths


def _reference_accumulate_edge_samples(
    high_conf_df: pd.DataFrame,
    pose_labels: List[str],
    skeleton_edges: List[Tuple[int, int]],
    min_valid_conf: float,
) -> Dict[Tuple[int, int], List[float]]:
    edge_samples: Dict[Tuple[int, int], List[float]] = {}
    K = len(pose_labels)
    for _, row in high_conf_df.iterrows():
        kpts = _extract_keypoints_from_row(row, pose_labels)
        if kpts is None or len(kpts) < K:
            continue
        for edge in skeleton_edges:
            try:
                ei, ej = int(edge[0]), int(edge[1])
            except Exception:
                continue
            if ei >= K or ej >= K:
                continue
            dist = _measure_edge_distance(kpts, ei, ej, min_valid_conf)
            if dist is not None:
                key = (min(ei, ej), max(ei, ej))
                edge_samples.setdefault(key, []).append(dist)
    return edge_samples


# ---------------------------------------------------------------------------
# Fixture DataFrame construction
# ---------------------------------------------------------------------------

_LABELS = ["head", "thorax", "abdomen", "leg1", "tail"]
# indices:    0       1         2         3       4
_ANTERIOR = [0, 1]  # weighted centroid over 2 indices
_POSTERIOR = [4]
_MIN_VALID_CONF = 0.2
_HIGH_CONF_FLOOR = 0.7

_SKELETON_EDGES: List = [
    (0, 1),  # head-thorax
    (1, 2),  # thorax-abdomen
    (2, 4),  # abdomen-tail
    (4, 0),  # tail-head (reverse of a later dup below -> tests canonical key)
    (0, 4),  # head-tail (duplicate edge mapping to same key as (4, 0))
    (3, 10),  # leg1 - out of range -> skipped entirely
    ("bad", 1),  # malformed -> int() raises -> skipped entirely
]


def _build_fixture_df() -> pd.DataFrame:
    n_high = 26  # >= 20 for is_valid priors
    n_low = 5  # filtered out by high_conf_floor

    rows = []

    # -- Bulk of well-formed high-confidence rows with varied per-kpt conf --
    for i in range(n_high):
        base_x = 100.0 + i * 1.5
        base_y = 50.0 - i * 0.7
        row = {
            "FrameID": i,
            "PoseKpt_head_X": base_x,
            "PoseKpt_head_Y": base_y,
            "PoseKpt_head_Conf": 0.9 if i % 4 != 0 else 0.15,  # some below thresh
            "PoseKpt_thorax_X": base_x + 5.0,
            "PoseKpt_thorax_Y": base_y + 1.0,
            "PoseKpt_thorax_Conf": 0.85,
            "PoseKpt_abdomen_X": base_x + 10.0,
            "PoseKpt_abdomen_Y": base_y + 2.0,
            "PoseKpt_abdomen_Conf": 0.8,
            "PoseKpt_leg1_X": base_x + 3.0,
            "PoseKpt_leg1_Y": base_y - 4.0,
            "PoseKpt_leg1_Conf": 0.5,
            # Head-tail OFFSET (not just base position) varies per row so
            # the (0, 4) canonical edge key's samples are all distinct --
            # this is required to give an order-sensitive equality
            # assertion real power to distinguish row-major merge order
            # from edge-major merge order for the duplicate (4,0)/(0,4)
            # edge pair below. A constant offset would make every sample
            # for this key identical, making any ordering bug invisible.
            "PoseKpt_tail_X": base_x + 20.0 + i * 0.3,
            "PoseKpt_tail_Y": base_y + 4.0 - i * 0.2,
            "PoseKpt_tail_Conf": 0.88,
            "PoseMeanConf": 0.8,
        }
        rows.append(row)

    # -- Rows below high_conf_floor: must be filtered out entirely --
    for j in range(n_low):
        rows.append(
            {
                "FrameID": 1000 + j,
                "PoseKpt_head_X": 1.0,
                "PoseKpt_head_Y": 1.0,
                "PoseKpt_head_Conf": 0.9,
                "PoseKpt_thorax_X": 2.0,
                "PoseKpt_thorax_Y": 2.0,
                "PoseKpt_thorax_Conf": 0.9,
                "PoseKpt_abdomen_X": 3.0,
                "PoseKpt_abdomen_Y": 3.0,
                "PoseKpt_abdomen_Conf": 0.9,
                "PoseKpt_leg1_X": 4.0,
                "PoseKpt_leg1_Y": 4.0,
                "PoseKpt_leg1_Conf": 0.9,
                "PoseKpt_tail_X": 5.0,
                "PoseKpt_tail_Y": 5.0,
                "PoseKpt_tail_Conf": 0.9,
                "PoseMeanConf": 0.1,  # below floor
            }
        )

    # -- NaN coordinate on a non-anchor keypoint (leg1) --
    rows.append(
        {
            "FrameID": 2001,
            "PoseKpt_head_X": 200.0,
            "PoseKpt_head_Y": 60.0,
            "PoseKpt_head_Conf": 0.9,
            "PoseKpt_thorax_X": 205.0,
            "PoseKpt_thorax_Y": 61.0,
            "PoseKpt_thorax_Conf": 0.9,
            "PoseKpt_abdomen_X": 210.0,
            "PoseKpt_abdomen_Y": 62.0,
            "PoseKpt_abdomen_Conf": 0.9,
            "PoseKpt_leg1_X": np.nan,
            "PoseKpt_leg1_Y": np.nan,
            "PoseKpt_leg1_Conf": 0.9,
            "PoseKpt_tail_X": 220.0,
            "PoseKpt_tail_Y": 64.0,
            "PoseKpt_tail_Conf": 0.9,
            "PoseMeanConf": 0.9,
        }
    )

    # -- NaN confidence on tail (edge endpoint): must NOT block edge sample
    #    (edge gate has no isfinite(conf) check) but MUST block weighted
    #    centroid contribution from tail (posterior index) since
    #    _weighted_centroid explicitly requires isfinite(conf). --
    rows.append(
        {
            "FrameID": 2002,
            "PoseKpt_head_X": 300.0,
            "PoseKpt_head_Y": 70.0,
            "PoseKpt_head_Conf": 0.9,
            "PoseKpt_thorax_X": 305.0,
            "PoseKpt_thorax_Y": 71.0,
            "PoseKpt_thorax_Conf": 0.9,
            "PoseKpt_abdomen_X": 310.0,
            "PoseKpt_abdomen_Y": 72.0,
            "PoseKpt_abdomen_Conf": 0.9,
            "PoseKpt_leg1_X": 303.0,
            "PoseKpt_leg1_Y": 68.0,
            "PoseKpt_leg1_Conf": 0.9,
            "PoseKpt_tail_X": 320.0,
            "PoseKpt_tail_Y": 74.0,
            "PoseKpt_tail_Conf": np.nan,
            "PoseMeanConf": 0.9,
        }
    )

    # -- Zero body length row (head == thorax, so anterior centroid could
    #    coincide with... actually make ant == post exactly): --
    rows.append(
        {
            "FrameID": 2003,
            "PoseKpt_head_X": 50.0,
            "PoseKpt_head_Y": 50.0,
            "PoseKpt_head_Conf": 0.9,
            "PoseKpt_thorax_X": 50.0,
            "PoseKpt_thorax_Y": 50.0,
            "PoseKpt_thorax_Conf": 0.9,
            "PoseKpt_abdomen_X": 55.0,
            "PoseKpt_abdomen_Y": 55.0,
            "PoseKpt_abdomen_Conf": 0.9,
            "PoseKpt_leg1_X": 52.0,
            "PoseKpt_leg1_Y": 48.0,
            "PoseKpt_leg1_Conf": 0.9,
            "PoseKpt_tail_X": 50.0,
            "PoseKpt_tail_Y": 50.0,  # tail == head/thorax centroid -> bl == 0
            "PoseKpt_tail_Conf": 0.9,
            "PoseMeanConf": 0.9,
        }
    )

    df = pd.DataFrame(rows)

    # -- Object-dtype column with a non-numeric string cell for one row on
    #    the "leg1" X column, to exercise the float() conversion-failure
    #    fallback path in keypoint extraction. --
    df["PoseKpt_leg1_X"] = df["PoseKpt_leg1_X"].astype(object)
    bad_row_idx = df.index[df["FrameID"] == 5][0]  # a normal high-conf row
    df.loc[bad_row_idx, "PoseKpt_leg1_X"] = "not_a_number"

    return df


@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    return _build_fixture_df()


@pytest.fixture(scope="module")
def high_conf_df(fixture_df: pd.DataFrame) -> pd.DataFrame:
    return _filter_high_conf_rows(fixture_df, _HIGH_CONF_FLOOR)


# ---------------------------------------------------------------------------
# Sanity: confirm branch coverage in the fixture itself
# ---------------------------------------------------------------------------


def test_fixture_filters_low_conf_rows(fixture_df, high_conf_df):
    assert len(fixture_df) > len(high_conf_df)
    assert (high_conf_df["PoseMeanConf"] >= _HIGH_CONF_FLOOR).all()


def test_fixture_has_object_dtype_column_with_bad_value(fixture_df):
    assert fixture_df["PoseKpt_leg1_X"].dtype == object
    assert "not_a_number" in fixture_df["PoseKpt_leg1_X"].values


def test_fixture_has_nan_coords_and_nan_conf(fixture_df):
    assert (
        fixture_df["PoseKpt_leg1_X"]
        .apply(lambda v: isinstance(v, float) and math.isnan(v))
        .any()
    )
    assert fixture_df["PoseKpt_tail_Conf"].isna().any()


# ---------------------------------------------------------------------------
# Reference capture (pre-vectorization behavior) -- this is the oracle
# ---------------------------------------------------------------------------


def test_reference_body_lengths_reasonable(high_conf_df):
    """Sanity check the reference implementation itself produces plausible
    output before using it as the byte-identity oracle."""
    body_lengths = _reference_collect_body_lengths(
        high_conf_df, _LABELS, _ANTERIOR, _POSTERIOR, _MIN_VALID_CONF
    )
    # 26 bulk rows + NaN-leg1 row + zero-bl row (excluded, bl==0) + NaN-conf
    # row (tail conf NaN -> posterior centroid excluded -> geom body_length
    # None -> excluded). So expect 26 (bulk, all have valid head/thorax
    # antr conf and valid tail conf) + 1 (NaN leg1 doesn't affect ant/post).
    assert len(body_lengths) >= 20
    assert all(bl > 0.0 for bl in body_lengths)


def test_reference_edge_samples_canonical_key_merges_duplicates(high_conf_df):
    edge_samples = _reference_accumulate_edge_samples(
        high_conf_df, _LABELS, _SKELETON_EDGES, _MIN_VALID_CONF
    )
    # (4, 0) and (0, 4) both canonicalize to (0, 4)
    assert (0, 4) in edge_samples
    # out-of-range and malformed edges must not appear
    assert not any(k for k in edge_samples if 10 in k)
    # the duplicate (4, 0) / (0, 4) edge pair contributes two samples per
    # row (once per listed edge) into the shared canonical key
    assert len(edge_samples[(0, 4)]) >= len(edge_samples[(0, 1)])


# ---------------------------------------------------------------------------
# Byte-identity: real implementation vs frozen reference
# ---------------------------------------------------------------------------


def test_collect_body_lengths_matches_reference_exactly(high_conf_df):
    expected = _reference_collect_body_lengths(
        high_conf_df, _LABELS, _ANTERIOR, _POSTERIOR, _MIN_VALID_CONF
    )
    actual = _collect_body_lengths(
        high_conf_df, _LABELS, _ANTERIOR, _POSTERIOR, _MIN_VALID_CONF
    )
    assert len(actual) == len(expected)
    assert actual == expected  # exact float equality, exact order
    # Also verify as numpy arrays with strict equality (covers any subtle
    # float32/float64 promotion mismatch that Python list `==` could mask
    # for NaN, though none are expected in body-length samples here).
    assert np.array_equal(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        equal_nan=True,
    )


def test_accumulate_edge_samples_matches_reference_exactly(high_conf_df):
    expected = _reference_accumulate_edge_samples(
        high_conf_df, _LABELS, _SKELETON_EDGES, _MIN_VALID_CONF
    )
    actual = _accumulate_edge_samples(
        high_conf_df, _LABELS, _SKELETON_EDGES, _MIN_VALID_CONF
    )
    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        assert actual[key] == expected[key], f"mismatch for edge key {key}"
        assert np.array_equal(
            np.asarray(actual[key], dtype=np.float64),
            np.asarray(expected[key], dtype=np.float64),
            equal_nan=True,
        )


def test_downstream_body_length_prior_matches_reference(high_conf_df):
    expected_samples = _reference_collect_body_lengths(
        high_conf_df, _LABELS, _ANTERIOR, _POSTERIOR, _MIN_VALID_CONF
    )
    actual_samples = _collect_body_lengths(
        high_conf_df, _LABELS, _ANTERIOR, _POSTERIOR, _MIN_VALID_CONF
    )
    expected_prior = _body_length_prior_from_samples(expected_samples)
    actual_prior = _body_length_prior_from_samples(actual_samples)
    assert actual_prior.n_samples == expected_prior.n_samples
    assert actual_prior.is_valid == expected_prior.is_valid
    assert actual_prior.median_px == expected_prior.median_px
    assert actual_prior.mad_px == expected_prior.mad_px


def test_downstream_edge_priors_match_reference(high_conf_df):
    expected_samples = _reference_accumulate_edge_samples(
        high_conf_df, _LABELS, _SKELETON_EDGES, _MIN_VALID_CONF
    )
    actual_samples = _accumulate_edge_samples(
        high_conf_df, _LABELS, _SKELETON_EDGES, _MIN_VALID_CONF
    )
    expected_priors = _build_edge_priors(expected_samples)
    actual_priors = _build_edge_priors(actual_samples)
    assert actual_priors.is_valid == expected_priors.is_valid
    assert set(actual_priors.priors.keys()) == set(expected_priors.priors.keys())
    for key in expected_priors.priors:
        exp = expected_priors.priors[key]
        act = actual_priors.priors[key]
        assert act["n_samples"] == exp["n_samples"]
        assert act["median_px"] == exp["median_px"]
        assert act["mad_px"] == exp["mad_px"]
