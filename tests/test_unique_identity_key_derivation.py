"""Tests for the UniqueIdentityKey writer (Phase 6 addendum).

Covers the serializer (``format_identity_key``), the per-row derivation
(``derive_unique_identity_key_series``), and an integration check that
``apply_identity_postprocessing_to_df`` actually writes the column.
"""

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.postprocess_df import (
    apply_identity_postprocessing_to_df,
)
from hydra_suite.core.post.identity_postprocess import (
    derive_unique_identity_key_series,
    format_identity_key,
    identity_sources_conflict,
    parse_identity_key,
)


def test_format_identity_key_basic():
    assert format_identity_key({"cnn:uid": "alpha"}) == "cnn:uid=alpha"


def test_format_identity_key_round_trips_with_parse():
    sources = {"apriltag": "3", "cnn:uid": "alpha", "cnn:uid:color": "red"}
    key = format_identity_key(sources)
    assert parse_identity_key(key) == sources


def test_format_identity_key_empty_is_empty_string():
    assert format_identity_key({}) == ""
    assert format_identity_key({"apriltag": ""}) == ""


def test_derive_series_cnn_2part_head():
    df = pd.DataFrame({"CNN_uid_Class": ["alpha", "beta"]})
    result = derive_unique_identity_key_series(df)
    assert result.tolist() == ["cnn:uid=alpha", "cnn:uid=beta"]


def test_derive_series_apriltag_from_label():
    df = pd.DataFrame({"DetectedTagLabel": ["3"]})
    result = derive_unique_identity_key_series(df)
    assert result.tolist() == ["apriltag=3"]


def test_derive_series_apriltag_from_id_fallback():
    df = pd.DataFrame({"DetectedTagID": [3]})
    result = derive_unique_identity_key_series(df)
    assert result.tolist() == ["apriltag=3"]


def test_derive_series_cnn_factor_head_omits_empty_factor():
    df = pd.DataFrame(
        {
            "CNN_uid_color_Class": ["red"],
            "CNN_uid_shape_Class": [""],
        }
    )
    result = derive_unique_identity_key_series(df)
    assert result.tolist() == ["cnn:uid:color=red"]


def test_derive_series_conf_column_is_not_a_class_column():
    df = pd.DataFrame(
        {
            "CNN_uid_Class": ["alpha"],
            "CNN_uid_Conf": [0.9],
        }
    )
    result = derive_unique_identity_key_series(df)
    assert result.tolist() == ["cnn:uid=alpha"]


def test_derive_series_sorted_multi_source_join():
    df = pd.DataFrame(
        {
            "DetectedTagLabel": ["3"],
            "CNN_uid_Class": ["alpha"],
        }
    )
    result = derive_unique_identity_key_series(df)
    assert result.tolist() == ["apriltag=3|cnn:uid=alpha"]


def test_derive_series_empty_row_is_nan():
    df = pd.DataFrame({"CNN_uid_Class": [np.nan], "DetectedTagLabel": [np.nan]})
    result = derive_unique_identity_key_series(df)
    assert len(result) == 1
    assert pd.isna(result.iloc[0])


def test_derive_series_no_evidence_columns_at_all_is_nan():
    df = pd.DataFrame({"FrameID": [0, 1]})
    result = derive_unique_identity_key_series(df)
    assert all(pd.isna(v) for v in result)


def test_no_token_without_equals_sign():
    # Every token derived must contain "=" -- parse_identity_key drops
    # bare-label tokens, which would silently disable conflict gating.
    df = pd.DataFrame(
        {
            "DetectedTagLabel": ["3"],
            "CNN_uid_Class": ["alpha"],
            "CNN_uid_color_Class": ["red"],
        }
    )
    result = derive_unique_identity_key_series(df)
    for key in result.dropna():
        for token in key.split("|"):
            assert "=" in token


def test_apply_identity_postprocessing_writes_unique_identity_key():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 1],
            "FrameID": [0, 1, 0],
            "CNN_uid_Class": ["alpha", "alpha", "beta"],
            "CNN_uid_Conf": [0.9, 0.9, 0.8],
        }
    )
    params = {
        "IDENTITY_POSTHOC_ENABLED": False,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
    }
    result = apply_identity_postprocessing_to_df(df, params)
    assert C.UNIQUE_IDENTITY_KEY in result.columns
    assert result[C.UNIQUE_IDENTITY_KEY].notna().all()
    assert result.loc[0, C.UNIQUE_IDENTITY_KEY] == "cnn:uid=alpha"
    assert result.loc[2, C.UNIQUE_IDENTITY_KEY] == "cnn:uid=beta"


def _two_head_df():
    return pd.DataFrame(
        {
            "CNN_colortag_Class": ["red_blue", "red_blue"],
            "CNN_colortag_Conf": [0.8, 0.8],
            "CNN_behavior_Class": ["walking", "grooming"],
            "CNN_behavior_Conf": [0.98, 0.97],
        }
    )


def test_key_excludes_non_identity_heads():
    keys = derive_unique_identity_key_series(
        _two_head_df(), identity_heads=("colortag",)
    )
    assert parse_identity_key(keys.iloc[0]) == {"cnn:colortag": "red_blue"}
    assert parse_identity_key(keys.iloc[1]) == {"cnn:colortag": "red_blue"}


def test_behavior_change_is_not_an_identity_conflict():
    keys = derive_unique_identity_key_series(
        _two_head_df(), identity_heads=("colortag",)
    )
    lhs = parse_identity_key(keys.iloc[0])
    rhs = parse_identity_key(keys.iloc[1])
    assert not identity_sources_conflict(lhs, rhs)


def test_behavior_change_conflicts_under_legacy_unscoped_call():
    # Documents the bug this task fixes: unscoped, the behavior head makes two
    # fragments of the SAME animal look like an identity conflict.
    keys = derive_unique_identity_key_series(_two_head_df())
    lhs = parse_identity_key(keys.iloc[0])
    rhs = parse_identity_key(keys.iloc[1])
    assert identity_sources_conflict(lhs, rhs)


def test_empty_identity_heads_drops_all_cnn_sources():
    keys = derive_unique_identity_key_series(_two_head_df(), identity_heads=())
    assert keys.isna().all()


def test_all_classifier_labels_disambiguates_prefix_collision():
    df = pd.DataFrame(
        {
            "CNN_tag_Class": ["a", "a"],
            "CNN_tag_v2_Class": ["b", "c"],
        }
    )
    keys = derive_unique_identity_key_series(
        df, identity_heads=("tag",), all_classifier_labels=("tag", "tag_v2")
    )
    assert parse_identity_key(keys.iloc[0]) == {"cnn:tag": "a"}
    assert parse_identity_key(keys.iloc[1]) == {"cnn:tag": "a"}
