"""Tests for PoseKit's Smart Select cluster-coverage frame selection."""

from __future__ import annotations

import numpy as np


def test_select_frames_by_cluster_coverage_maximizes_new_cluster_coverage():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    # 4 individuals across 2 frames. Frame "f1" spans clusters {0, 1}
    # (maximal single-frame coverage); frame "f2" spans only cluster {0}.
    eligible_indices = [0, 1, 2, 3]
    cluster_id = np.array([0, 1, 0, 0])
    frame_key_of_index = {0: "f1", 1: "f1", 2: "f2", 3: "f2"}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=1,
    )

    assert selected == ["f1"]


def test_select_frames_by_cluster_coverage_continues_after_full_coverage():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    # 2 clusters total. Frame "f1" covers both; frame "f2" covers only
    # cluster 0 (fewer total distinct clusters) but is picked once
    # coverage is exhausted and budget remains.
    eligible_indices = [0, 1, 2]
    cluster_id = np.array([0, 1, 0])
    frame_key_of_index = {0: "f1", 1: "f1", 2: "f2"}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=2,
    )

    assert selected == ["f1", "f2"]


def test_select_frames_by_cluster_coverage_deterministic_tie_break():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    # Two frames with identical coverage profiles -- tie-break must be
    # deterministic (smallest frame key wins).
    eligible_indices = [0, 1]
    cluster_id = np.array([0, 1])
    frame_key_of_index = {0: ("src", 5), 1: ("src", 2)}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=1,
    )

    assert selected == [("src", 2)]


def test_select_frames_by_cluster_coverage_respects_budget_over_frame_count():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    eligible_indices = [0, 1, 2]
    cluster_id = np.array([0, 1, 2])
    frame_key_of_index = {0: "f1", 1: "f2", 2: "f3"}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=2,
    )

    assert len(selected) == 2


def test_select_frames_by_cluster_coverage_empty_input_returns_empty():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    selected = select_frames_by_cluster_coverage(
        cluster_id=np.array([]),
        eligible_indices=[],
        frame_key_of_index={},
        want_n_frames=5,
    )

    assert selected == []
