import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.postprocess_df import (
    apply_identity_postprocessing_to_df,
)


def test_empty_df_passthrough():
    empty = pd.DataFrame()
    assert apply_identity_postprocessing_to_df(empty, {}).empty


def test_annotates_summary_columns_when_solver_disabled():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            C.REALTIME_LABEL: ["antA", "antA"],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )
    assert C.EVIDENCE_SOURCES in out.columns
    assert C.EVIDENCE_CONFLICT_FLAG in out.columns
    assert out[C.EVIDENCE_CONFLICT_FLAG].tolist() == [0, 0]
    # Legacy pre-Phase-6 name must be gone.
    assert "IdentityConflictFlag" not in out.columns


def test_evidence_summary_uses_final_family_for_offline_source():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [0],
            C.FINAL_LABEL: ["antA"],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )
    assert out[C.EVIDENCE_SOURCES].tolist() == ["offline"]


def test_evidence_summary_top_label_and_confidence_from_cnn_columns():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            "CNN_colorlabel_Class": ["antA", "antB"],
            "CNN_colorlabel_Conf": [0.9, 0.4],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )
    assert C.EVIDENCE_TOPLABEL in out.columns
    assert C.EVIDENCE_CONFIDENCE in out.columns
    assert out[C.EVIDENCE_TOPLABEL].tolist() == ["antA", "antB"]
    assert out[C.EVIDENCE_CONFIDENCE].tolist() == [0.9, 0.4]
    assert out[C.EVIDENCE_TOPLABEL].dtype == object


def test_evidence_summary_no_evidence_columns_leaves_toplabel_nan():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [0],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )
    assert C.EVIDENCE_TOPLABEL in out.columns
    assert pd.isna(out[C.EVIDENCE_TOPLABEL].iloc[0])
    assert out[C.EVIDENCE_TOPLABEL].dtype == object


def test_realtime_to_final_mirror_is_non_destructive():
    """Phase 6 Task 5: with the fragment solver OFF and IdentityRealtimeLabel
    populated, apply_identity_postprocessing_to_df must mirror it into
    IdentityFinalLabel with IdentityFinalSource == "realtime", and must
    leave IdentityRealtimeLabel itself untouched (realtime is read-only)."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            C.REALTIME_LABEL: ["antA", "antA"],
            C.REALTIME_ID: [1.0, 1.0],
            C.REALTIME_CONFIDENCE: [0.8, 0.8],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )

    assert out[C.FINAL_LABEL].tolist() == ["antA", "antA"]
    assert out[C.FINAL_SOURCE].tolist() == ["realtime", "realtime"]
    assert out[C.FINAL_ID].tolist() == [1.0, 1.0]
    assert out[C.FINAL_CONFIDENCE].tolist() == [0.8, 0.8]
    # Realtime columns are read-only for this stage -- unchanged.
    assert out[C.REALTIME_LABEL].tolist() == ["antA", "antA"]
    assert out[C.REALTIME_ID].tolist() == [1.0, 1.0]
    assert out[C.REALTIME_CONFIDENCE].tolist() == [0.8, 0.8]


def test_realtime_to_final_mirror_does_not_overwrite_offline_rows():
    """Rows the offline solver already resolved (IdentityFinalSource ==
    "offline") must never be clobbered by the realtime mirror, even when a
    (possibly stale/disagreeing) realtime label is present on the same row."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [0],
            C.FINAL_LABEL: ["antB"],
            C.FINAL_SOURCE: [C.IdentityFinalSource.OFFLINE],
            C.REALTIME_LABEL: ["antA"],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )

    assert out[C.FINAL_LABEL].tolist() == ["antB"]
    assert out[C.FINAL_SOURCE].tolist() == ["offline"]


def test_tag_resolved_rows_get_final_source_tag():
    """Rows with no realtime evidence but a detected AprilTag are mirrored
    into Final with IdentityFinalSource == "tag"."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [0],
            "DetectedTagLabel": ["antC"],
            "DetectedTagConf": [0.99],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )

    assert out[C.FINAL_LABEL].tolist() == ["antC"]
    assert out[C.FINAL_SOURCE].tolist() == ["tag"]


def test_evidence_summary_does_not_write_realtime_or_final_columns():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0],
            "FrameID": [0],
            "CNN_colorlabel_Class": ["antA"],
            "CNN_colorlabel_Conf": [0.9],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )
    assert C.REALTIME_LABEL not in out.columns
    assert C.FINAL_LABEL not in out.columns


def _df_with_confident_behavior_head():
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            "CNN_colortag_Class": ["red_blue", "red_blue"],
            "CNN_colortag_Conf": [0.80, 0.80],
            "CNN_behavior_Class": ["walking", "walking"],
            "CNN_behavior_Conf": [0.98, 0.98],
        }
    )


_PARAMS = {
    "CNN_CLASSIFIERS": [
        {"label": "colortag", "unique_identifier": True},
        {"label": "behavior", "unique_identifier": False},
    ],
    "IDENTITY_POSTHOC_ENABLED": False,
    "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
}


def test_top_evidence_label_ignores_more_confident_non_identity_head():
    out = apply_identity_postprocessing_to_df(
        _df_with_confident_behavior_head(), _PARAMS
    )
    assert out[C.EVIDENCE_TOPLABEL].tolist() == ["red_blue", "red_blue"]
    assert out[C.EVIDENCE_CONFIDENCE].tolist() == [0.80, 0.80]


def test_non_identity_head_columns_are_still_exported():
    out = apply_identity_postprocessing_to_df(
        _df_with_confident_behavior_head(), _PARAMS
    )
    assert out["CNN_behavior_Class"].tolist() == ["walking", "walking"]


def test_absent_classifier_config_keeps_legacy_all_columns_behavior():
    # No CNN_CLASSIFIERS key at all -> legacy fallback, behavior head wins.
    out = apply_identity_postprocessing_to_df(
        _df_with_confident_behavior_head(),
        {"IDENTITY_POSTHOC_ENABLED": False, "ENABLE_IDENTITY_FRAGMENT_SOLVER": False},
    )
    assert out[C.EVIDENCE_TOPLABEL].tolist() == ["walking", "walking"]


def test_zero_identity_heads_leaves_evidence_toplabel_nan():
    # CNN_CLASSIFIERS present, but no entry has unique_identifier=True ->
    # heads = (), not HEADS_UNKNOWN -> scoped scan finds no identity columns.
    params = {
        "CNN_CLASSIFIERS": [
            {"label": "colortag", "unique_identifier": False},
            {"label": "behavior", "unique_identifier": False},
        ],
        "IDENTITY_POSTHOC_ENABLED": False,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
    }
    out = apply_identity_postprocessing_to_df(
        _df_with_confident_behavior_head(), params
    )
    assert out[C.EVIDENCE_TOPLABEL].isna().all()
