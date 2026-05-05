"""Tests for hydra_suite.refinekit.core.merge_candidates."""

import numpy as np
import pandas as pd

from hydra_suite.refinekit.core.merge_candidates import (
    SwapCandidate,
    build_candidates,
    build_candidates_for_target_count,
    build_swap_candidates,
    extract_segments,
    predict_position,
    update_after_merge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(tracks):
    """Build a simple trajectory DataFrame from a list of (track_id, frames) specs.

    Each spec is (track_id, frame_start, frame_end, x_start, y_start, dx, dy).
    """
    rows = []
    for tid, f_start, f_end, x0, y0, dx, dy in tracks:
        for f in range(f_start, f_end + 1):
            rows.append(
                {
                    "FrameID": f,
                    "TrajectoryID": tid,
                    "X": x0 + dx * (f - f_start),
                    "Y": y0 + dy * (f - f_start),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# predict_position
# ---------------------------------------------------------------------------


class TestPredictPosition:
    def test_zero_velocity(self):
        pos = predict_position(np.array([10.0, 20.0]), np.array([0.0, 0.0]), 5)
        np.testing.assert_array_almost_equal(pos, [10.0, 20.0])

    def test_constant_velocity_no_damping(self):
        pos = predict_position(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), 3, damping=1.0
        )
        np.testing.assert_array_almost_equal(pos, [3.0, 0.0])

    def test_damping_reduces_prediction(self):
        pos_damped = predict_position(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), 5, damping=0.9
        )
        pos_undamped = predict_position(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), 5, damping=1.0
        )
        # Damped should travel less distance
        assert pos_damped[0] < pos_undamped[0]

    def test_zero_frames(self):
        pos = predict_position(np.array([5.0, 5.0]), np.array([2.0, 3.0]), 0)
        np.testing.assert_array_almost_equal(pos, [5.0, 5.0])


# ---------------------------------------------------------------------------
# extract_segments
# ---------------------------------------------------------------------------


class TestExtractSegments:
    def test_basic_extraction(self):
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 1.0, 0.0),
                (2, 30, 50, 50.0, 10.0, -1.0, 0.5),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        assert len(segs) == 2
        ids = {s.track_id for s in segs}
        assert ids == {1, 2}

    def test_frame_birth_death(self):
        df = _make_df([(1, 10, 30, 0.0, 0.0, 0.0, 0.0)])
        segs = extract_segments(df, last_frame=50)
        assert len(segs) == 1
        s = segs[0]
        assert s.frame_birth == 10
        assert s.frame_death == 30

    def test_is_alive_at_end(self):
        df = _make_df([(1, 0, 100, 0.0, 0.0, 0.0, 0.0)])
        segs = extract_segments(df, last_frame=100)
        assert segs[0].is_alive_at_end is True

    def test_not_alive_at_end(self):
        df = _make_df([(1, 0, 50, 0.0, 0.0, 0.0, 0.0)])
        segs = extract_segments(df, last_frame=100)
        assert segs[0].is_alive_at_end is False

    def test_position_endpoints(self):
        df = _make_df([(1, 0, 10, 5.0, 5.0, 2.0, 3.0)])
        segs = extract_segments(df, last_frame=10)
        s = segs[0]
        np.testing.assert_array_almost_equal(s.pos_birth, [5.0, 5.0])
        np.testing.assert_array_almost_equal(s.pos_death, [25.0, 35.0])

    def test_empty_df(self):
        df = pd.DataFrame(columns=["FrameID", "TrajectoryID", "X", "Y"])
        segs = extract_segments(df, last_frame=100)
        assert segs == []


# ---------------------------------------------------------------------------
# build_candidates
# ---------------------------------------------------------------------------


class TestBuildCandidates:
    def test_adjacent_tracks_produce_candidate(self):
        """Track 1 dies at frame 20, track 2 is born at frame 25 nearby."""
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 0.5, 0.0),
                (2, 25, 50, 15.0, 10.0, 0.5, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        cands = build_candidates(segs)
        # Track 1 should have candidate(s) pointing to track 2
        assert 1 in cands
        target_ids = [c.target_id for c in cands[1]]
        assert 2 in target_ids

    def test_distant_tracks_no_candidate(self):
        """Tracks too far apart — no merge candidates."""
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 0.0, 0.0),
                (2, 25, 50, 500.0, 500.0, 0.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        cands = build_candidates(segs)
        # Track 1 should have no candidates (target is too far)
        if 1 in cands:
            target_ids = [c.target_id for c in cands[1]]
            assert 2 not in target_ids

    def test_large_gap_no_candidate(self):
        """Gap exceeds max_gap — no merge candidates."""
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 0.0, 0.0),
                (2, 100, 120, 12.0, 10.0, 0.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=120)
        cands = build_candidates(segs, max_gap=30)
        if 1 in cands:
            assert all(c.target_id != 2 for c in cands[1])

    def test_overlapping_tracks_never_produce_candidate(self):
        """Overlapping fragments should never be proposed for merge."""
        df = _make_df(
            [
                (1, 0, 25, 10.0, 10.0, 0.5, 0.0),
                (2, 20, 50, 15.0, 10.0, 0.5, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        cands = build_candidates(segs, max_overlap=10)
        if 1 in cands:
            assert all(c.target_id != 2 for c in cands[1])

    def test_auto_expand_merge_radius_for_required_reductions(self):
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 0.5, 0.0),
                (2, 25, 50, 260.0, 10.0, 0.5, 0.0),
                (3, 0, 50, 500.0, 300.0, 0.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        cands, tuning = build_candidates_for_target_count(segs, max_animals=2)

        assert tuning.max_dist > 200.0
        assert 1 in cands
        assert any(c.target_id == 2 for c in cands[1])

    def test_candidates_sorted_by_score(self):
        """Candidates for a source should be sorted by score descending."""
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 1.0, 0.0),
                (2, 22, 50, 20.0, 10.0, 1.0, 0.0),
                (3, 23, 50, 30.0, 12.0, 1.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        cands = build_candidates(segs)
        if 1 in cands and len(cands[1]) > 1:
            scores = [c.score for c in cands[1]]
            assert scores == sorted(scores, reverse=True)

    def test_alive_at_end_not_source(self):
        """Tracks alive at end of video should not appear as sources."""
        df = _make_df(
            [
                (1, 0, 100, 10.0, 10.0, 0.0, 0.0),
                (2, 80, 100, 12.0, 10.0, 0.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=100)
        cands = build_candidates(segs)
        # Track 1 is alive at end, so it should not be a source
        assert 1 not in cands


# ---------------------------------------------------------------------------
# update_after_merge
# ---------------------------------------------------------------------------


class TestUpdateAfterMerge:
    def test_merged_segment_survives(self):
        """After merging 1→2, a combined segment should exist."""
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 1.0, 0.0),
                (2, 25, 50, 22.0, 10.0, 1.0, 0.0),
                (3, 55, 80, 35.0, 10.0, 1.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=80)
        cands = build_candidates(segs)

        new_segs, new_cands = update_after_merge(segs, cands, 1, 2)

        # Target (2) should be gone; source (1) absorbs it
        ids = {s.track_id for s in new_segs}
        assert 2 not in ids
        assert 1 in ids
        # Merged segment should have source's birth frame
        seg1 = next(s for s in new_segs if s.track_id == 1)
        assert seg1.frame_birth == 0
        # Merged segment should have target's death frame
        assert seg1.frame_death == 50

    def test_stale_candidates_removed(self):
        """After merge, source should no longer appear in candidate graph."""
        df = _make_df(
            [
                (1, 0, 20, 10.0, 10.0, 1.0, 0.0),
                (2, 25, 50, 22.0, 10.0, 1.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=50)
        cands = build_candidates(segs)

        new_segs, new_cands = update_after_merge(segs, cands, 1, 2)
        assert 1 not in new_cands
        # Also, no candidate should reference source_id=1
        for sid, clist in new_cands.items():
            for c in clist:
                assert c.source_id != 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_track_no_candidates(self):
        df = _make_df([(1, 0, 50, 0.0, 0.0, 0.0, 0.0)])
        segs = extract_segments(df, last_frame=100)
        cands = build_candidates(segs)
        # Single dying track with no birth to match — empty candidates
        assert all(len(v) == 0 for v in cands.values()) if cands else True

    def test_nan_positions_handled(self):
        """Rows with NaN X/Y should be gracefully skipped."""
        df = _make_df([(1, 0, 10, 10.0, 10.0, 1.0, 0.0)])
        # Inject some NaN rows
        extra = pd.DataFrame(
            {
                "FrameID": [11, 12],
                "TrajectoryID": [1, 1],
                "X": [np.nan, np.nan],
                "Y": [np.nan, np.nan],
            }
        )
        df = pd.concat([df, extra], ignore_index=True)
        segs = extract_segments(df, last_frame=20)
        assert len(segs) == 1
        # Death frame should still be 10 (last non-NaN frame)
        assert segs[0].frame_death == 10

    def test_theta_column_used_for_heading(self):
        """If Theta column is present, heading should use it."""
        df = _make_df([(1, 0, 10, 10.0, 10.0, 1.0, 0.0)])
        df["Theta"] = 1.57  # ~pi/2
        segs = extract_segments(df, last_frame=10)
        # Heading should be close to 1.57 (from Theta column)
        assert abs(segs[0].heading_death - 1.57) < 0.1


# ---------------------------------------------------------------------------
# build_swap_candidates
# ---------------------------------------------------------------------------


class TestBuildSwapCandidates:
    def _make_swap_scenario(self):
        """Create a scenario where track 1 dies and track 2 is nearby and alive.

        Track 1: frames 0-50, moving right at y=100
        Track 2: frames 0-100, moving right at y=105 (close by)
        At frame 50, track 1 dies while track 2 passes very close.
        """
        return _make_df(
            [
                (1, 0, 50, 10.0, 100.0, 1.0, 0.0),  # dies at 50
                (2, 0, 100, 10.0, 105.0, 1.0, 0.0),  # alive throughout
            ]
        )

    def test_basic_swap_candidate_found(self):
        """A track alive and nearby when source dies should be a swap candidate."""
        df = self._make_swap_scenario()
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        assert 1 in cands
        target_ids = [c.target_id for c in cands[1]]
        assert 2 in target_ids

    def test_swap_candidate_has_swap_frame(self):
        df = self._make_swap_scenario()
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        sc = cands[1][0]
        assert isinstance(sc, SwapCandidate)
        # Swap frame should be near the death frame of track 1
        assert abs(sc.swap_frame - 50) <= 20

    def test_swap_candidate_not_before_source_death(self):
        df = self._make_swap_scenario()
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        sc = cands[1][0]

        assert sc.swap_frame >= 50

    def test_swap_score_positive(self):
        df = self._make_swap_scenario()
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        sc = cands[1][0]
        assert sc.score > 0

    def test_distant_track_no_swap(self):
        """Track too far away should not be a swap candidate."""
        df = _make_df(
            [
                (1, 0, 50, 10.0, 100.0, 1.0, 0.0),
                (2, 0, 100, 500.0, 500.0, 0.0, 0.0),  # far away
            ]
        )
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        if 1 in cands:
            assert all(c.target_id != 2 for c in cands[1])

    def test_recently_born_target_excluded(self):
        """A target born too recently should NOT be a swap candidate.

        Tracks born near the death time are merge candidates, not swaps.
        """
        df = _make_df(
            [
                (1, 0, 50, 10.0, 100.0, 1.0, 0.0),
                (2, 45, 100, 55.0, 102.0, 1.0, 0.0),  # born just 5 frames before
            ]
        )
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs, min_pre_swap=10)
        # Track 2 born at 45, only 5 frames before death at 50 — below min_pre_swap
        if 1 in cands:
            assert all(c.target_id != 2 for c in cands[1])

    def test_target_dead_before_source_excluded(self):
        """Target must be alive after source dies."""
        df = _make_df(
            [
                (1, 0, 50, 10.0, 100.0, 1.0, 0.0),
                (2, 0, 40, 10.0, 105.0, 1.0, 0.0),  # dies before source
            ]
        )
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        if 1 in cands:
            assert all(c.target_id != 2 for c in cands[1])

    def test_alive_at_end_not_source(self):
        """Tracks alive at end should not be swap sources."""
        df = _make_df(
            [
                (1, 0, 100, 10.0, 100.0, 1.0, 0.0),
                (2, 0, 100, 10.0, 105.0, 1.0, 0.0),
            ]
        )
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        assert 1 not in cands
        assert 2 not in cands

    def test_swap_candidates_sorted_by_score(self):
        """Multiple swap candidates should be sorted by score descending."""
        df = _make_df(
            [
                (1, 0, 50, 10.0, 100.0, 1.0, 0.0),
                (2, 0, 100, 10.0, 103.0, 1.0, 0.0),  # very close
                (3, 0, 100, 10.0, 115.0, 1.0, 0.0),  # a bit farther
            ]
        )
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        if 1 in cands and len(cands[1]) > 1:
            scores = [c.score for c in cands[1]]
            assert scores == sorted(scores, reverse=True)

    def test_min_distance_recorded(self):
        df = self._make_swap_scenario()
        segs = extract_segments(df, last_frame=100)
        cands = build_swap_candidates(df, segs)
        sc = cands[1][0]
        # The two tracks are 5px apart (y=100 vs y=105)
        assert sc.min_distance <= 10.0
