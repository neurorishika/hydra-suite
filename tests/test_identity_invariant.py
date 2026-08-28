import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.identity_postprocess import (
    assert_one_identity_per_trajectory,
    collapse_to_majority_identity,
)


def _df():
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0, 1, 1],
            "FrameID": [1, 2, 3, 1, 2],
            C.FINAL_LABEL: ["a", "a", "b", "c", "c"],
            C.FINAL_ID: [1, 1, 2, 3, 3],
            C.FINAL_SOURCE: ["offline", "offline", "offline", "offline", "offline"],
            C.FINAL_CONFIDENCE: [0.9, 0.9, 0.2, 0.8, 0.8],
        }
    )


def test_offenders_are_reported():
    assert assert_one_identity_per_trajectory(_df()) == [0]


def test_collapse_uses_majority_and_min_confidence():
    out = collapse_to_majority_identity(_df(), [0])
    t0 = out[out.TrajectoryID == 0]
    assert t0[C.FINAL_LABEL].unique().tolist() == ["a"]
    assert t0[C.FINAL_ID].unique().tolist() == [1]
    assert (t0[C.FINAL_CONFIDENCE] == 0.2).all()
    assert assert_one_identity_per_trajectory(out) == []


def test_no_offenders_when_columns_missing():
    df = pd.DataFrame({"TrajectoryID": [0, 0], "FrameID": [1, 2]})
    assert assert_one_identity_per_trajectory(df) == []


def test_no_offenders_on_clean_frame():
    df = _df()
    df.loc[df.TrajectoryID == 0, C.FINAL_LABEL] = "a"
    df.loc[df.TrajectoryID == 0, C.FINAL_ID] = 1
    assert assert_one_identity_per_trajectory(df) == []


def test_collapse_is_noop_for_untouched_trajectories():
    out = collapse_to_majority_identity(_df(), [0])
    t1 = out[out.TrajectoryID == 1]
    assert t1[C.FINAL_LABEL].tolist() == ["c", "c"]
    assert t1[C.FINAL_CONFIDENCE].tolist() == [0.8, 0.8]


def _pure_source_provenance_df():
    """Same label everywhere, source varies only because one row was never
    resolved by realtime/tag/offline evidence and got consensus-filled
    (``fill_identity_nans_with_consensus``) with confidence 0.0 and source
    left at ``NONE``. This is NOT a genuine identity conflict (C1)."""
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0, 0],
            "FrameID": [1, 2, 3, 4],
            C.FINAL_LABEL: ["ant_a", "ant_a", "ant_a", "ant_a"],
            C.FINAL_ID: [1, 1, 1, 1],
            C.FINAL_SOURCE: [
                C.IdentityFinalSource.REALTIME,
                C.IdentityFinalSource.REALTIME,
                C.IdentityFinalSource.NONE,
                C.IdentityFinalSource.REALTIME,
            ],
            C.FINAL_CONFIDENCE: [0.9, 0.8, 0.0, 0.95],
        }
    )


def test_pure_source_provenance_variation_is_not_an_offender():
    """C1: a trajectory with identical labels, one un-evidenced (NONE-source,
    consensus-filled) row must NOT be flagged purely because its source
    differs from the rest -- nothing here actually conflicted."""
    df = _pure_source_provenance_df()
    assert assert_one_identity_per_trajectory(df) == []


def test_pure_source_provenance_variation_never_reaches_collapse():
    """C1's actual pipeline contract: since this trajectory is not an
    offender, the real caller (``rich_export.py``) never invokes
    ``collapse_to_majority_identity`` on it at all, so its real per-row
    confidences and source values ship untouched."""
    df = _pure_source_provenance_df()
    offenders = assert_one_identity_per_trajectory(df)
    assert offenders == []
    out = collapse_to_majority_identity(df, offenders)
    pd.testing.assert_series_equal(
        out[C.FINAL_CONFIDENCE], df[C.FINAL_CONFIDENCE], check_names=False
    )
    pd.testing.assert_series_equal(
        out[C.FINAL_SOURCE], df[C.FINAL_SOURCE], check_names=False
    )


def test_collapse_never_fabricates_source_on_none_rows_even_if_forced():
    """Defense in depth: even if a future caller invokes
    ``collapse_to_majority_identity`` directly on this trajectory (bypassing
    the offender check), it must never overwrite the NONE-source row's
    source with a fabricated value -- only rows that genuinely carried a
    source get rewritten."""
    df = _pure_source_provenance_df()
    out = collapse_to_majority_identity(df, [0])
    assert out.loc[2, C.FINAL_SOURCE] == C.IdentityFinalSource.NONE
    assert out.loc[[0, 1, 3], C.FINAL_SOURCE].eq(C.IdentityFinalSource.REALTIME).all()


def test_genuine_source_conflict_still_detected_and_collapsed():
    """A trajectory whose EVIDENCED rows genuinely disagree on source (and
    label) must still be caught and collapsed -- the C1 fix must not break
    real conflict detection."""
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0, 0],
            "FrameID": [1, 2, 3, 4],
            C.FINAL_LABEL: ["ant_a", "ant_a", "ant_b", "ant_a"],
            C.FINAL_ID: [1, 1, 2, 1],
            C.FINAL_SOURCE: [
                C.IdentityFinalSource.REALTIME,
                C.IdentityFinalSource.REALTIME,
                C.IdentityFinalSource.TAG,
                C.IdentityFinalSource.NONE,
            ],
            C.FINAL_CONFIDENCE: [0.9, 0.8, 0.6, 0.0],
        }
    )
    offenders = assert_one_identity_per_trajectory(df)
    assert offenders == [0]

    out = collapse_to_majority_identity(df, offenders)
    t0 = out[out.TrajectoryID == 0]
    assert t0[C.FINAL_LABEL].unique().tolist() == ["ant_a"]
    assert t0[C.FINAL_ID].unique().tolist() == [1]
    # The minority (conflicting) row is relabeled to the majority and its
    # real source overwritten to the majority row's source.
    assert out.loc[2, C.FINAL_SOURCE] == C.IdentityFinalSource.REALTIME
    # The never-evidenced (NONE-source) row keeps NONE -- never fabricated.
    assert out.loc[3, C.FINAL_SOURCE] == C.IdentityFinalSource.NONE
    # Confidence collapses to the min among EVIDENCED rows only (0.6, the
    # relabeled minority row's real confidence), not the NONE row's 0.0.
    assert (t0[C.FINAL_CONFIDENCE] == 0.6).all()
    assert assert_one_identity_per_trajectory(out) == []
