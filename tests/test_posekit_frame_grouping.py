"""Tests for PoseKit's frame-grouping helper."""

from __future__ import annotations


def test_group_indices_by_frame_groups_matching_identity_filenames():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    filenames = ["did10000.jpg", "did10001.jpg", "did20000.jpg", "plain.png"]
    source_ids = ["src_a", "src_a", "src_a", "src_a"]

    groups = group_indices_by_frame(filenames, source_ids)

    assert groups[("src_a", 1)] == [0, 1]
    assert groups[("src_a", 2)] == [2]
    # Non-matching filenames get their own singleton key.
    assert groups[("src_a", -4)] == [3]


def test_group_indices_by_frame_scopes_by_source():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    filenames = ["did10000.jpg", "did10000.jpg"]
    source_ids = ["src_a", "src_b"]

    groups = group_indices_by_frame(filenames, source_ids)

    # Same frame_idx (1) but different sources must not collide.
    assert groups[("src_a", 1)] == [0]
    assert groups[("src_b", 1)] == [1]


def test_group_indices_by_frame_singleton_keys_never_collide_with_real_frames():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    # frame_idx=0 is a real, valid frame index (did0.jpg -> detection_id=0 -> frame_idx=0).
    filenames = ["did0.jpg", "plain_0.png", "plain_1.png"]
    source_ids = ["src_a", "src_a", "src_a"]

    groups = group_indices_by_frame(filenames, source_ids)

    assert groups[("src_a", 0)] == [0]
    # Singleton keys use negative frame components, so they can never
    # collide with a real (always >= 0) frame_idx.
    assert groups[("src_a", -2)] == [1]
    assert groups[("src_a", -3)] == [2]


def test_group_indices_by_frame_empty_input_returns_empty_dict():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    assert group_indices_by_frame([], []) == {}
