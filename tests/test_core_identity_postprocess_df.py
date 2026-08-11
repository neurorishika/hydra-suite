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
