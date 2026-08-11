"""Tests for resolve_simultaneous_identity_conflicts in post/processing.py."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from tests.helpers.module_loader import load_src_module


def _scipy_stub() -> dict[str, object]:
    interp_ns = types.SimpleNamespace(
        CubicSpline=object,
        UnivariateSpline=object,
        interp1d=object,
    )
    scipy_ns = types.SimpleNamespace(interpolate=interp_ns)
    return {
        "scipy": scipy_ns,
        "scipy.interpolate": interp_ns,
    }


mod = load_src_module(
    "hydra_suite/core/post/processing.py",
    "processing_under_test",
    stubs=_scipy_stub(),
)

resolve_simultaneous_identity_conflicts = mod.resolve_simultaneous_identity_conflicts
_IDENTITY_LABEL_COL = mod._IDENTITY_LABEL_COL
_IDENTITY_ID_COL = mod._IDENTITY_ID_COL
_IDENTITY_CONF_COL = mod._IDENTITY_CONF_COL
_IDENTITY_CONFLICT_COL = mod._IDENTITY_CONFLICT_COL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_traj(
    frames: list[int],
    label: str | None = None,
    conf: float = 0.9,
    tag_votes: int = 0,
    source: str = "forward",
) -> pd.DataFrame:
    rows = []
    for f in frames:
        row = {
            "FrameID": f,
            "X": float(f),
            "Y": 0.0,
            _IDENTITY_LABEL_COL: label,
            _IDENTITY_CONF_COL: conf if label is not None else np.nan,
            _IDENTITY_ID_COL: 0 if label is not None else np.nan,
            "TagVotes": tag_votes,
            "_source": source,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _label(df: pd.DataFrame) -> str | float:
    vals = df[_IDENTITY_LABEL_COL].dropna()
    return vals.iloc[0] if not vals.empty else np.nan


def _conflict_flag(df: pd.DataFrame) -> bool:
    return bool(
        _IDENTITY_CONFLICT_COL in df.columns and df[_IDENTITY_CONFLICT_COL].any()
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_conflict_different_labels() -> None:
    """Two overlapping tracks with different labels are left unchanged."""
    a = _make_traj([1, 2, 3], label="ant_1")
    b = _make_traj([2, 3, 4], label="ant_2")
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_1"
    assert _label(result[1]) == "ant_2"
    assert not _conflict_flag(result[0])
    assert not _conflict_flag(result[1])


def test_no_conflict_no_frame_overlap() -> None:
    """Two tracks with the same label but non-overlapping frames are both kept."""
    a = _make_traj([1, 2, 3], label="ant_1")
    b = _make_traj([4, 5, 6], label="ant_1")
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_1"
    assert _label(result[1]) == "ant_1"
    assert not _conflict_flag(result[0])
    assert not _conflict_flag(result[1])


def test_higher_tag_votes_wins() -> None:
    """Track with more tag votes wins when both claim the same identity."""
    a = _make_traj([1, 2, 3], label="ant_1", tag_votes=10)
    b = _make_traj([2, 3, 4], label="ant_1", tag_votes=2)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_1"
    assert pd.isna(_label(result[1]))
    assert _conflict_flag(result[1])
    assert not _conflict_flag(result[0])


def test_higher_confidence_wins_when_no_tags() -> None:
    """When tag votes are equal (0), the track with higher conf wins."""
    a = _make_traj([1, 2, 3], label="ant_2", conf=0.95, tag_votes=0)
    b = _make_traj([2, 3, 4], label="ant_2", conf=0.60, tag_votes=0)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_2"
    assert pd.isna(_label(result[1]))
    assert _conflict_flag(result[1])


def test_longer_track_wins_when_scores_tied() -> None:
    """When conf and tags are equal, more frames wins."""
    a = _make_traj([1, 2, 3, 4, 5], label="ant_3", conf=0.8, tag_votes=0)
    b = _make_traj([3, 4], label="ant_3", conf=0.8, tag_votes=0)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_3"
    assert pd.isna(_label(result[1]))


def test_forward_source_breaks_tie() -> None:
    """When all numeric scores are equal, forward-pass track wins."""
    a = _make_traj([1, 2, 3], label="ant_4", conf=0.8, tag_votes=0, source="forward")
    b = _make_traj([1, 2, 3], label="ant_4", conf=0.8, tag_votes=0, source="backward")
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_4"
    assert pd.isna(_label(result[1]))


def test_loser_identity_columns_cleared() -> None:
    """Loser has label/id/conf stripped and IdentityFinalConflictResolved set."""
    a = _make_traj([1, 2, 3], label="ant_5", conf=0.9, tag_votes=5)
    b = _make_traj([2, 3, 4], label="ant_5", conf=0.5, tag_votes=0)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    loser = result[1]
    assert pd.isna(loser[_IDENTITY_LABEL_COL].iloc[0])
    assert pd.isna(loser[_IDENTITY_ID_COL].iloc[0])
    assert float(loser[_IDENTITY_CONF_COL].iloc[0]) == 0.0
    assert bool(loser[_IDENTITY_CONFLICT_COL].iloc[0])


def test_unlabeled_tracks_ignored() -> None:
    """Tracks without IdentityFinalLabel are untouched."""
    a = _make_traj([1, 2, 3], label=None)
    b = _make_traj([2, 3, 4], label="ant_6", conf=0.9)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert pd.isna(_label(result[0]))
    assert _label(result[1]) == "ant_6"


def test_three_way_conflict_two_losers() -> None:
    """When three tracks overlap with the same label, only the strongest survives."""
    a = _make_traj([1, 2, 3], label="ant_7", conf=0.9, tag_votes=10)
    b = _make_traj([2, 3, 4], label="ant_7", conf=0.7, tag_votes=3)
    c = _make_traj([1, 2, 4], label="ant_7", conf=0.5, tag_votes=1)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy(), c.copy()])
    assert _label(result[0]) == "ant_7"
    assert pd.isna(_label(result[1]))
    assert pd.isna(_label(result[2]))


def test_empty_list_returns_empty() -> None:
    result = resolve_simultaneous_identity_conflicts([])
    assert result == []


def test_single_track_unchanged() -> None:
    a = _make_traj([1, 2, 3], label="ant_8")
    result = resolve_simultaneous_identity_conflicts([a.copy()])
    assert _label(result[0]) == "ant_8"
    assert not _conflict_flag(result[0])


def test_non_overlapping_same_label_both_kept() -> None:
    """Sequential tracks with same label and no shared frame are both valid."""
    a = _make_traj(list(range(1, 50)), label="ant_9", conf=0.9)
    b = _make_traj(list(range(50, 100)), label="ant_9", conf=0.9)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy()])
    assert _label(result[0]) == "ant_9"
    assert _label(result[1]) == "ant_9"


def _make_traj_mixed_labels(
    frames: list[int],
    labels: list[str],
    conf: float = 0.8,
    tag_votes: int = 0,
    source: str = "forward",
) -> pd.DataFrame:
    """Build a trajectory whose per-row labels vary (used to exercise agreement)."""
    assert len(frames) == len(labels)
    rows = []
    for f, lbl in zip(frames, labels):
        rows.append(
            {
                "FrameID": f,
                "X": float(f),
                "Y": 0.0,
                _IDENTITY_LABEL_COL: lbl,
                _IDENTITY_CONF_COL: conf,
                _IDENTITY_ID_COL: 0,
                "TagVotes": tag_votes,
                "_source": source,
            }
        )
    return pd.DataFrame(rows)


def test_jittery_loses_to_consistent_at_same_length_and_conf() -> None:
    """Two tracks of equal length and mean confidence: the one whose per-row
    labels actually agree on the modal label wins. Agreement enters the score
    multiplicatively, so this case used to tie under the old lex-tuple."""
    consistent = _make_traj([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="ant_x", conf=0.7)
    jittery_labels = ["ant_x", "ant_y", "ant_x", "ant_y", "ant_x"] * 2
    jittery = _make_traj_mixed_labels(list(range(1, 11)), jittery_labels, conf=0.7)
    result = resolve_simultaneous_identity_conflicts(
        [consistent.copy(), jittery.copy()]
    )
    assert _label(result[0]) == "ant_x", "consistent track must keep its label"
    assert pd.isna(_label(result[1])), "jittery (low-agreement) track must lose"
    assert _conflict_flag(result[1])


def test_long_high_conf_wins_over_short_high_conf_at_zero_tags() -> None:
    """Length × confidence multiplicative weighting: a 50-frame track at conf
    0.8 beats a 5-frame track at conf 0.85 even though the short one has a
    higher mean confidence — the length factor dominates."""
    long_track = _make_traj(list(range(1, 51)), label="ant_z", conf=0.8)
    short_track = _make_traj(list(range(20, 25)), label="ant_z", conf=0.85)
    result = resolve_simultaneous_identity_conflicts(
        [long_track.copy(), short_track.copy()]
    )
    assert _label(result[0]) == "ant_z"
    assert pd.isna(_label(result[1]))


def test_near_tied_three_way_overlap_only_highest_survives() -> None:
    """Three mutually-overlapping same-label trajectories with near-tied
    (but distinct) scores: exactly the highest-scoring one keeps the label,
    the other two are cleared. Regression for the CRITICAL bug where a
    group-wide `solve_unique_assignment` over per-slot `score/sum(scores)`
    shares put EVERY slot below the solver's implicit "dummy beats me"
    line for n>=3 (average share 1/n < 0.5), stripping all three at once."""
    # Scores ~0.4 / ~0.35 / ~0.45 via conf alone (agreement=1, length_factor=1
    # for equal-length tracks, tag_votes=0 so no tag bonus).
    a = _make_traj([1, 2, 3], label="ant_tri", conf=0.45, tag_votes=0)
    b = _make_traj([1, 2, 3], label="ant_tri", conf=0.40, tag_votes=0)
    c = _make_traj([1, 2, 3], label="ant_tri", conf=0.35, tag_votes=0)
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy(), c.copy()])
    assert _label(result[0]) == "ant_tri", "highest-scoring claimant must survive"
    assert pd.isna(_label(result[1]))
    assert pd.isna(_label(result[2]))


def test_disjoint_ends_of_overlap_chain_both_keep_label() -> None:
    """A-B-C chain where A and C are temporally DISJOINT but both overlap B:
    A and C must BOTH keep the label (they never actually compete with each
    other); only B — which overlaps both higher scorers — is cleared.
    Regression for the CRITICAL bug where union-find grouped A-B-C into one
    transitive component and a whole-component Hunganrian solve wrongly
    stripped A and C too, despite them sharing zero frames."""
    a = _make_traj(list(range(1, 11)), label="ant_chain", conf=0.95, tag_votes=0)
    b = _make_traj(list(range(8, 21)), label="ant_chain", conf=0.5, tag_votes=0)
    c = _make_traj(list(range(18, 31)), label="ant_chain", conf=0.95, tag_votes=0)
    assert set(range(1, 11)).isdisjoint(set(range(18, 31))), "A and C must be disjoint"
    result = resolve_simultaneous_identity_conflicts([a.copy(), b.copy(), c.copy()])
    assert _label(result[0]) == "ant_chain", "A does not overlap C; must keep label"
    assert pd.isna(_label(result[1])), "B overlaps two higher scorers; must lose"
    assert _label(result[2]) == "ant_chain", "C does not overlap A; must keep label"


def test_two_independent_conflicts_resolve_independently() -> None:
    """Two unrelated same-label overlap conflicts in one call are resolved
    independently of each other."""
    a1 = _make_traj([1, 2, 3], label="ant_p", conf=0.9, tag_votes=0)
    a2 = _make_traj([2, 3, 4], label="ant_p", conf=0.4, tag_votes=0)
    b1 = _make_traj([100, 101, 102], label="ant_q", conf=0.3, tag_votes=0)
    b2 = _make_traj([101, 102, 103], label="ant_q", conf=0.9, tag_votes=0)
    result = resolve_simultaneous_identity_conflicts(
        [a1.copy(), a2.copy(), b1.copy(), b2.copy()]
    )
    assert _label(result[0]) == "ant_p"
    assert pd.isna(_label(result[1]))
    assert pd.isna(_label(result[2]))
    assert _label(result[3]) == "ant_q"


def test_strong_tag_evidence_overrides_long_low_margin_track() -> None:
    """A short track with strong AprilTag confirmation must beat a long track
    with weaker mean confidence and no tags — tag-vote bonus is additive and
    weighted heavily enough to dominate the length-driven unary term."""
    long_no_tags = _make_traj(
        list(range(1, 101)), label="ant_q", conf=0.45, tag_votes=0
    )
    short_with_tags = _make_traj(
        list(range(40, 50)), label="ant_q", conf=0.8, tag_votes=15
    )
    result = resolve_simultaneous_identity_conflicts(
        [long_no_tags.copy(), short_with_tags.copy()]
    )
    assert pd.isna(_label(result[0])), "long no-tag track must lose to tagged short one"
    assert _label(result[1]) == "ant_q"
