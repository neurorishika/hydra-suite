"""Characterization guard for Task 5 (vectorize identity-evidence `.apply` calls).

This test snapshots the exact pre-vectorization output of
``apply_identity_postprocessing_to_df``'s ``IdentityEvidence*`` summary
columns (``_row_sources``/``_row_conflict``/``_row_top_evidence`` in
``hydra_suite.core.individual.postprocess_df``) on two hand-built input
DataFrames that exercise every branch of those three helpers:

- apriltag-only evidence (``DetectedTagID``/``InterpTagID``)
- CNN-only evidence (``CNN_*_Class``/``CNN_*_Conf`` pairs)
- ``IdentityFinalLabel`` present with no ``IdentityFinalSource`` -> "offline"
  fallback
- ``IdentityFinalSource == "tag"`` -> apriltag added to sources (precedence)
- ``IdentityFinalSource == "realtime"`` -> the "pass" branch, with the
  separate ``IdentityRealtimeLabel`` check adding "realtime"
- conflict via "assigned label differs from an observed label"
- conflict via "more than one distinct observed label, no assigned label"
- no conflict (single observed label matching the assigned label)
- top-evidence argmax across two CNN classifier heads
- top-evidence tie-break (equal confidences -> first column in dict/column
  order wins, via strict ``>``)
- top-evidence NaN handling (label or confidence missing skips that head)
- top-evidence fallback to a detected AprilTag label/confidence when no CNN
  evidence is present
- top-evidence fallback to a detected AprilTag label with a NaN confidence
- the "no sources at all" -> ``np.nan`` branch (isolated in a second,
  minimal DataFrame with no ``TrajectoryID`` column, so the unrelated
  ``fill_identity_nans_with_consensus`` step -- which fills empty
  trajectories with ``"unknown"`` -- cannot mask it)

The expected values below were captured by running the CURRENT (pre-Task-5)
production code once and hardcoding the output. They are therefore a valid
non-tautological byte-identical oracle for the vectorized replacement: if
vectorization changes any value, dtype, or NaN handling, this test fails.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.postprocess_df import (
    apply_identity_postprocessing_to_df,
)

nan = np.nan


def _build_multi_branch_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TrajectoryID": [0] * 13 + [1],
            "FrameID": list(range(13)) + [0],
            "DetectedTagID": [
                1,
                nan,
                nan,
                nan,
                nan,
                2,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            "InterpTagID": [nan] * 14,
            "DetectedTagLabel": [
                nan,
                nan,
                nan,
                nan,
                nan,
                "antB",
                "antD",
                "antF",
                nan,
                nan,
                "antJ",
                "antK",
                nan,
                nan,
            ],
            "DetectedTagConf": [
                nan,
                nan,
                nan,
                nan,
                nan,
                0.2,
                0.1,
                0.6,
                nan,
                nan,
                0.3,
                nan,
                nan,
                nan,
            ],
            "CNN_colorA_Class": [
                nan,
                "antX",
                nan,
                nan,
                nan,
                "antC",
                nan,
                nan,
                "antG",
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            "CNN_colorA_Conf": [
                nan,
                0.7,
                nan,
                nan,
                nan,
                0.55,
                nan,
                nan,
                0.5,
                nan,
                0.9,
                nan,
                nan,
                nan,
            ],
            "CNN_colorB_Class": [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                "antE",
                nan,
                "antH",
                "antI",
                nan,
                nan,
                nan,
                nan,
            ],
            "CNN_colorB_Conf": [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                0.45,
                nan,
                0.5,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            C.FINAL_LABEL: [
                nan,
                nan,
                "antY",
                "antZ2",
                "antZ",
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            C.FINAL_SMOOTHED_LABEL: [nan] * 14,
            C.FINAL_SOURCE: [
                "",
                "",
                "",
                "tag",
                "realtime",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            C.REALTIME_LABEL: [
                nan,
                nan,
                nan,
                nan,
                "antZ",
                "antA",
                nan,
                "antF",
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
        }
    )


def test_identity_evidence_columns_characterization_multi_branch():
    """Snapshot of _row_sources/_row_conflict/_row_top_evidence on 14 rows.

    Covers: apriltag/cnn/offline/realtime/tag-precedence sources, both
    conflict branches (assigned-mismatch and multi-observed-no-assigned),
    no-conflict, CNN argmax + tie-break + NaN skipping, and AprilTag
    fallback (with and without a confidence value).
    """
    df = _build_multi_branch_df()
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )

    expected = pd.DataFrame(
        {
            C.EVIDENCE_SOURCES: [
                "apriltag,offline",
                "cnn,offline",
                "offline",
                "apriltag",
                "realtime",
                "apriltag,cnn",
                "apriltag,cnn",
                "realtime",
                "cnn,offline",
                "cnn,offline",
                "apriltag",
                "apriltag",
                "offline",
                "offline",
            ],
            C.EVIDENCE_CONFLICT_FLAG: [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            C.EVIDENCE_TOPLABEL: [
                nan,
                "antX",
                nan,
                nan,
                nan,
                "antC",
                "antE",
                "antF",
                "antG",
                nan,
                "antJ",
                "antK",
                nan,
                nan,
            ],
            C.EVIDENCE_CONFIDENCE: [
                nan,
                0.7,
                nan,
                nan,
                nan,
                0.55,
                0.45,
                0.6,
                0.5,
                nan,
                0.3,
                nan,
                nan,
                nan,
            ],
        },
        index=out.index,
    )
    expected[C.EVIDENCE_CONFLICT_FLAG] = expected[C.EVIDENCE_CONFLICT_FLAG].astype(
        "int64"
    )
    expected[C.EVIDENCE_TOPLABEL] = expected[C.EVIDENCE_TOPLABEL].astype("object")

    pdt.assert_frame_equal(
        out[list(expected.columns)],
        expected,
        check_exact=True,
        check_dtype=True,
    )


def test_identity_evidence_columns_characterization_no_sources_branch():
    """Isolates the "no evidence at all" -> np.nan branch of _row_sources.

    No ``TrajectoryID`` column is present, so
    ``fill_identity_nans_with_consensus`` (an unrelated pipeline step that
    fills empty trajectories with the literal ``"unknown"``) short-circuits
    and cannot mask the branch under test.
    """
    df = pd.DataFrame({"FrameID": [0]})
    out = apply_identity_postprocessing_to_df(
        df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False}
    )

    expected = pd.DataFrame(
        {
            C.EVIDENCE_SOURCES: pd.Series([np.nan], dtype="float64"),
            C.EVIDENCE_CONFLICT_FLAG: pd.Series([0], dtype="int64"),
            C.EVIDENCE_TOPLABEL: pd.Series([np.nan], dtype="object"),
            C.EVIDENCE_CONFIDENCE: pd.Series([np.nan], dtype="float64"),
        },
        index=out.index,
    )

    pdt.assert_frame_equal(
        out[list(expected.columns)],
        expected,
        check_exact=True,
        check_dtype=True,
    )
