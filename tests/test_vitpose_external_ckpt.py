"""Unit tests for the external-ViTPose-checkpoint probe tool."""

from __future__ import annotations

import pytest

from tools.vitpose.external_ckpt.skeleton import builtin_skeleton


def test_ant_skeleton_has_nine_named_keypoints():
    spec = builtin_skeleton("ant")
    assert spec.num_keypoints == 9
    assert spec.keypoint_names == [
        "A_R_T",
        "A_L_T",
        "A_R_M",
        "A_L_M",
        "Head_T",
        "Centroid",
        "Abd_T",
        "Abd_B",
        "Head_B",
    ]
    assert spec.skeleton_edges == [
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 8),
        (8, 5),
        (5, 6),
        (6, 7),
    ]


def test_fly_skeleton_has_twentynine_keypoints_and_legs():
    spec = builtin_skeleton("fly")
    assert spec.num_keypoints == 29
    assert spec.keypoint_names[:4] == [
        "headTop",
        "thoraxCenter",
        "abdomenTop",
        "abdomenCenter",
    ]
    assert "hindlegRight" in spec.keypoint_names
    assert len(spec.skeleton_edges) == 28


def test_skeleton_colors_are_bgr_reversed_from_config_rgb():
    # ant keypoint 0 (A_R_T) is RGB [148, 0, 211] in the mmpose config.
    spec = builtin_skeleton("ant")
    assert spec.keypoint_colors_bgr[0] == (211, 0, 148)
    assert len(spec.keypoint_colors_bgr) == spec.num_keypoints
    assert len(spec.edge_colors_bgr) == len(spec.skeleton_edges)


def test_edges_index_within_range():
    for species in ("ant", "fly"):
        spec = builtin_skeleton(species)
        for a, b in spec.skeleton_edges:
            assert 0 <= a < spec.num_keypoints
            assert 0 <= b < spec.num_keypoints


def test_unknown_species_rejected():
    with pytest.raises(ValueError, match="unknown species"):
        builtin_skeleton("beetle")
