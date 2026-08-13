"""Tests for identity-aware trajectory post-processing helpers.

NOTE: The former split/join/interpolation tests in this file exercised
``apply_identity_postprocessing(df, params)``, which was renamed and moved to
``apply_identity_postprocessing_to_df`` in
``hydra_suite/core/individual/postprocess_df.py`` — and whose split/join is now
gated behind ``ENABLE_IDENTITY_FRAGMENT_SOLVER`` (default off). Those stale
tests were removed (2026-08); the fragment-solver behaviour they covered is a
separate concern that needs fixtures written against the new, gated API. The
consensus-fill helper below still lives in ``core/post/identity_postprocess.py``
and is the sole remaining coverage for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/core/post/identity_postprocess.py",
    "identity_postprocess_under_test",
)

fill_identity_nans_with_consensus = mod.fill_identity_nans_with_consensus


def test_fill_identity_nans_with_consensus_handles_float_label_columns() -> None:
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 1],
            "FrameID": [0, 1, 2],
            "IdentityAssignedLabel": [np.nan, np.nan, np.nan],
            "IdentityAssignedID": [np.nan, np.nan, np.nan],
            "IdentityAssignedConfidence": [np.nan, np.nan, np.nan],
            "IdentitySlotLockLabel": [np.nan, np.nan, np.nan],
        }
    )

    out = fill_identity_nans_with_consensus(df)

    assert out["IdentityAssignedLabel"].tolist() == ["unknown", "unknown", "unknown"]
    assert out["IdentitySlotLockLabel"].tolist() == ["unknown", "unknown", "unknown"]
    assert out["IdentityAssignedID"].tolist() == [0.0, 0.0, 0.0]
    assert out["IdentityAssignedConfidence"].tolist() == [0.0, 0.0, 0.0]
