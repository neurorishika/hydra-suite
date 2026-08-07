import pandas as pd

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
            "IdentityAssignedLabel": ["antA", "antA"],
        }
    )
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )
    assert "IdentityEvidenceSources" in out.columns
    assert "IdentityConflictFlag" in out.columns
    assert out["IdentityConflictFlag"].tolist() == [0, 0]
