"""Unit tests for the shared pose_features module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydra_suite.core.identity.geometry import (
    build_detection_direction_overrides,
    normalize_theta,
    resolve_detection_tracking_theta,
)
from hydra_suite.core.identity.pose.features import (
    build_pose_detection_keypoint_map,
    compute_detection_pose_features,
    compute_pose_geometry_from_keypoints,
    count_valid_normalized_pose_keypoints,
    is_pose_heading_reliable,
    normalize_pose_keypoints,
    parse_pose_group_tokens,
    resolve_pose_group_indices,
)
from hydra_suite.utils.geometry import (
    apply_foreign_obb_mask,
    filter_keypoints_by_foreign_obbs,
)

# ---------------------------------------------------------------------------
# normalize_theta
# ---------------------------------------------------------------------------


def test_normalize_theta_zero():
    assert abs(normalize_theta(0.0)) < 1e-6


def test_normalize_theta_full_rotation_wraps_to_zero():
    assert abs(normalize_theta(2 * math.pi)) < 1e-6


def test_normalize_theta_negative_pi_becomes_pi():
    assert abs(normalize_theta(-math.pi) - math.pi) < 1e-6


def test_normalize_theta_negative_two_pi():
    assert abs(normalize_theta(-2 * math.pi)) < 1e-6


def test_normalize_theta_three_halves_pi():
    expected = 3 * math.pi / 2
    assert abs(normalize_theta(-math.pi / 2) - expected) < 1e-6


def test_normalize_theta_invalid_input_returns_zero():
    assert abs(normalize_theta("bad")) < 1e-6


# ---------------------------------------------------------------------------
# parse_pose_group_tokens
# ---------------------------------------------------------------------------


def test_parse_pose_group_tokens_none():
    assert parse_pose_group_tokens(None) == []


def test_parse_pose_group_tokens_empty_string():
    assert parse_pose_group_tokens("") == []


def test_parse_pose_group_tokens_string_mixed():
    result = parse_pose_group_tokens("0, head, 2")
    assert result == [0, "head", 2]


def test_parse_pose_group_tokens_list():
    result = parse_pose_group_tokens([1, "tail", 3])
    assert result == [1, "tail", 3]


def test_parse_pose_group_tokens_single_int():
    assert parse_pose_group_tokens(5) == [5]


# ---------------------------------------------------------------------------
# resolve_pose_group_indices
# ---------------------------------------------------------------------------


def test_resolve_pose_group_indices_by_name():
    names = ["thorax", "head", "abdomen"]
    result = resolve_pose_group_indices(["head", "thorax"], names)
    assert set(result) == {0, 1}


def test_resolve_pose_group_indices_by_int():
    names = ["a", "b", "c"]
    assert resolve_pose_group_indices([0, 2], names) == [0, 2]


def test_resolve_pose_group_indices_deduplicates():
    names = ["a", "b"]
    assert resolve_pose_group_indices([0, 0, 1], names) == [0, 1]


def test_resolve_pose_group_indices_out_of_range_skipped():
    names = ["a", "b"]
    assert resolve_pose_group_indices([0, 5], names) == [0]


def test_resolve_pose_group_indices_unknown_name_skipped():
    names = ["a", "b"]
    assert resolve_pose_group_indices(["a", "unknown"], names) == [0]


def test_resolve_pose_group_indices_empty_spec():
    names = ["a", "b"]
    assert resolve_pose_group_indices(None, names) == []


def test_resolve_pose_group_indices_empty_names():
    assert resolve_pose_group_indices([0], []) == []


# ---------------------------------------------------------------------------
# build_pose_detection_keypoint_map
# ---------------------------------------------------------------------------


def test_build_pose_detection_keypoint_map_none_cache():
    assert build_pose_detection_keypoint_map(None, 0) == {}


class _FakePoseCache:
    """Minimal stub matching the IndividualPropertiesCache.get_frame interface."""

    def __init__(self, data):
        self._data = (
            data  # {frame_idx: {"detection_ids": [...], "pose_keypoints": [...]}}
        )

    def get_frame(self, frame_idx):
        return self._data.get(
            int(frame_idx), {"detection_ids": [], "pose_keypoints": []}
        )


def test_build_pose_detection_keypoint_map_basic():
    kpts = np.array([[1.0, 2.0, 0.9]], dtype=np.float32)
    cache = _FakePoseCache({7: {"detection_ids": [42], "pose_keypoints": [kpts]}})
    result = build_pose_detection_keypoint_map(cache, 7)
    assert 42 in result
    np.testing.assert_array_equal(result[42], kpts)


def test_build_pose_detection_keypoint_map_missing_frame():
    cache = _FakePoseCache({})
    assert build_pose_detection_keypoint_map(cache, 99) == {}


# ---------------------------------------------------------------------------
# compute_pose_geometry_from_keypoints
# ---------------------------------------------------------------------------


def test_compute_pose_geometry_from_keypoints_basic_heading():
    # anterior at (10, 0), posterior at (0, 0) => heading ~= 0 rad
    kpts = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    result = compute_pose_geometry_from_keypoints(kpts, [0], [1], min_valid_conf=0.1)
    assert result is not None
    assert result["heading"] is not None
    assert abs(result["heading"]) < 0.1 or abs(result["heading"] - 2 * math.pi) < 0.1


def test_compute_pose_geometry_from_keypoints_body_length():
    kpts = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    result = compute_pose_geometry_from_keypoints(kpts, [0], [1], min_valid_conf=0.1)
    assert result is not None
    assert result["body_length"] == pytest.approx(10.0, abs=0.1)


def test_compute_pose_geometry_from_keypoints_full_visibility():
    kpts = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    result = compute_pose_geometry_from_keypoints(kpts, [0], [1], min_valid_conf=0.1)
    assert result is not None
    assert result["visibility"] == pytest.approx(1.0, abs=0.01)


def test_compute_pose_geometry_from_keypoints_low_conf_heading_is_none():
    # Both keypoints below min_valid_conf
    kpts = np.array([[10.0, 0.0, 0.05], [0.0, 0.0, 0.05]], dtype=np.float32)
    result = compute_pose_geometry_from_keypoints(kpts, [0], [1], min_valid_conf=0.1)
    assert result is not None
    assert result["heading"] is None


def test_compute_pose_geometry_from_keypoints_none_input():
    assert compute_pose_geometry_from_keypoints(None, [0], [1], 0.2) is None


def test_compute_pose_geometry_from_keypoints_invalid_shape():
    kpts = np.array([1.0, 2.0], dtype=np.float32)  # 1-D, invalid
    assert compute_pose_geometry_from_keypoints(kpts, [0], [1], 0.2) is None


def test_compute_pose_geometry_from_keypoints_ignore_indices():
    # kpt[0] is anterior at (10, 0) but ignored; kpt[2] is at (8, 0) above conf
    kpts = np.array(
        [[10.0, 0.0, 0.9], [0.0, 0.0, 0.9], [8.0, 0.0, 0.9]],
        dtype=np.float32,
    )
    result = compute_pose_geometry_from_keypoints(
        kpts, [0, 2], [1], min_valid_conf=0.1, ignore_indices=[0]
    )
    assert result is not None
    # Only kpt[2] (8,0) is valid for anterior after ignore; heading ~= 0
    assert result["heading"] is not None


# ---------------------------------------------------------------------------
# normalize_pose_keypoints
# ---------------------------------------------------------------------------


def test_normalize_pose_keypoints_none_input():
    assert normalize_pose_keypoints(None, 0.2) is None


def test_normalize_pose_keypoints_invalid_shape():
    assert normalize_pose_keypoints(np.array([1.0, 2.0]), 0.2) is None


def test_normalize_pose_keypoints_all_low_conf():
    kpts = np.array([[2.0, 0.0, 0.05], [-2.0, 0.0, 0.05]], dtype=np.float32)
    assert normalize_pose_keypoints(kpts, min_valid_conf=0.1) is None


def test_normalize_pose_keypoints_centered():
    # Two equal-confidence points symmetric about origin after centering
    kpts = np.array([[2.0, 0.0, 0.9], [-2.0, 0.0, 0.9]], dtype=np.float32)
    out = normalize_pose_keypoints(kpts, min_valid_conf=0.1)
    assert out is not None
    # Centroid x should sum to zero after centering
    assert abs(float(out[0, 0]) + float(out[1, 0])) < 1e-5


def test_normalize_pose_keypoints_invalid_entries_become_nan():
    kpts = np.array([[2.0, 0.0, 0.9], [-2.0, 0.0, 0.05]], dtype=np.float32)
    out = normalize_pose_keypoints(kpts, min_valid_conf=0.1)
    assert out is not None
    # kpt[1] had low conf — its x/y should be nan
    assert np.isnan(out[1, 0])
    assert np.isnan(out[1, 1])


def test_normalize_pose_keypoints_conf_preserved():
    kpts = np.array([[2.0, 0.0, 0.85], [-2.0, 0.0, 0.75]], dtype=np.float32)
    out = normalize_pose_keypoints(kpts, min_valid_conf=0.1)
    assert out is not None
    assert out[0, 2] == pytest.approx(0.85, abs=1e-5)
    assert out[1, 2] == pytest.approx(0.75, abs=1e-5)


def test_count_valid_normalized_pose_keypoints_counts_finite_xy_only():
    kpts = np.array(
        [[0.0, 0.0, 1.0], [np.nan, np.nan, 0.0], [1.0, 1.0, 0.8]],
        dtype=np.float32,
    )
    assert count_valid_normalized_pose_keypoints(kpts) == 2


def test_is_pose_heading_reliable_requires_visibility_and_enough_keypoints():
    kpts = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.9]],
        dtype=np.float32,
    )
    assert is_pose_heading_reliable(kpts, visibility=0.75, min_visibility=0.6)
    assert not is_pose_heading_reliable(kpts, visibility=0.4, min_visibility=0.6)
    assert not is_pose_heading_reliable(
        kpts[:2], visibility=0.75, min_visibility=0.6, min_valid_keypoints=3
    )


# ---------------------------------------------------------------------------
# compute_detection_pose_features
# ---------------------------------------------------------------------------


def test_compute_detection_pose_features_no_match():
    kpt_map = {}
    kpts, vis = compute_detection_pose_features([12345], kpt_map, [0], [1], [], 0.2)
    assert kpts == [None]
    assert vis[0] == pytest.approx(0.0)


def test_compute_detection_pose_features_with_match():
    det_id = 42
    kpts_raw = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    kpt_map = {det_id: kpts_raw}
    out_kpts, out_vis = compute_detection_pose_features(
        [det_id], kpt_map, [0], [1], [], 0.2
    )
    assert out_kpts[0] is not None
    assert out_vis[0] > 0.0


def test_compute_detection_pose_features_partial_match():
    det_ids = [1, 2, 3]
    kpts_raw = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    kpt_map = {2: kpts_raw}  # only middle detection has pose
    out_kpts, out_vis = compute_detection_pose_features(
        det_ids, kpt_map, [0], [1], [], 0.2
    )
    assert out_kpts[0] is None
    assert out_kpts[1] is not None
    assert out_kpts[2] is None
    assert out_vis[0] == pytest.approx(0.0)
    assert out_vis[1] > 0.0
    assert out_vis[2] == pytest.approx(0.0)


def test_compute_detection_pose_features_empty():
    out_kpts, out_vis = compute_detection_pose_features([], {}, [0], [1], [], 0.2)
    assert out_kpts == []
    assert len(out_vis) == 0


# ---------------------------------------------------------------------------
# Directed heading merge / tracking resolution
# ---------------------------------------------------------------------------


def test_build_detection_direction_overrides_prefers_pose_by_default():
    headings, directed = build_detection_direction_overrides(
        2,
        pose_headings=[math.pi / 4, None],
        pose_directed_mask=[1, 0],
        headtail_headings=[math.pi, math.pi / 2],
        headtail_directed_mask=[1, 1],
    )

    assert directed.tolist() == [1, 1]
    assert headings[0] == pytest.approx(math.pi / 4, abs=1e-6)
    assert headings[1] == pytest.approx(math.pi / 2, abs=1e-6)


def test_build_detection_direction_overrides_can_prefer_headtail():
    headings, directed = build_detection_direction_overrides(
        1,
        pose_headings=[math.pi / 4],
        pose_directed_mask=[1],
        headtail_headings=[math.pi],
        headtail_directed_mask=[1],
        pose_overrides_headtail=False,
    )

    assert directed.tolist() == [1]
    assert headings[0] == pytest.approx(math.pi, abs=1e-6)


def test_resolve_detection_tracking_theta_uses_directed_heading_override():
    theta = resolve_detection_tracking_theta(
        0,
        measured_theta=0.0,
        directed_heading=math.pi / 2,
        pose_directed=True,
        orientation_last=[None],
    )

    assert theta == pytest.approx(math.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_foreign_obb_mask
# ---------------------------------------------------------------------------


def _square_corners(x0, y0, size):
    """Helper: axis-aligned square OBB corners as float32 (4, 2)."""
    return np.array(
        [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
        dtype=np.float32,
    )


def test_apply_foreign_obb_mask_no_list_returns_crop():
    crop = np.ones((50, 50, 3), dtype=np.uint8) * 200
    result = apply_foreign_obb_mask(crop, 0, 0, [], background_color=0)
    assert np.array_equal(result, crop)


def test_apply_foreign_obb_mask_none_crop_returns_none():
    result = apply_foreign_obb_mask(
        None, 0, 0, [_square_corners(0, 0, 10)], background_color=0
    )
    assert result is None


def test_apply_foreign_obb_mask_fills_foreign_region():
    """Pixels inside the foreign OBB should be set to background_color."""
    crop = np.ones((100, 100, 3), dtype=np.uint8) * 200
    # Foreign OBB occupies top-left 30x30 of the frame; crop starts at (0, 0)
    other = [_square_corners(0, 0, 30)]
    result = apply_foreign_obb_mask(crop, 0, 0, other, background_color=128)
    # Center of the foreign OBB should be filled
    assert result[15, 15, 0] == 128
    # Outside the OBB should be untouched
    assert result[50, 50, 0] == 200


def test_apply_foreign_obb_mask_applies_offset():
    """Crop offset shifts foreign OBB into local coordinates correctly."""
    crop = np.ones((50, 50, 3), dtype=np.uint8) * 200
    # Foreign animal at frame coords (80, 80)–(110, 110); crop starts at (80, 80)
    other = [_square_corners(80, 80, 30)]
    result = apply_foreign_obb_mask(crop, 80, 80, other, background_color=0)
    # Local (0,0) maps to frame (80,80) — should be filled
    assert result[5, 5, 0] == 0
    # Pixels that mapped outside the crop (clipped) should also be affected


def test_apply_foreign_obb_mask_returns_copy():
    """Original crop must not be mutated."""
    crop = np.ones((50, 50, 3), dtype=np.uint8) * 200
    other = [_square_corners(0, 0, 30)]
    result = apply_foreign_obb_mask(crop, 0, 0, other, background_color=0)
    assert result[10, 10, 0] == 0
    assert crop[10, 10, 0] == 200  # original unchanged


def test_apply_foreign_obb_mask_ignores_bad_shape():
    """A corners array with wrong shape should be silently skipped."""
    crop = np.ones((50, 50, 3), dtype=np.uint8) * 200
    bad = [np.array([[0, 0], [10, 10]], dtype=np.float32)]  # shape (2, 2) not (4, 2)
    result = apply_foreign_obb_mask(crop, 0, 0, bad, background_color=0)
    # Crop should be unchanged since bad polygon was skipped
    assert np.all(result == 200)


def test_apply_foreign_obb_mask_grayscale():
    """Grayscale (2-D) crops should be handled without error."""
    crop = np.ones((50, 50), dtype=np.uint8) * 200
    other = [_square_corners(0, 0, 20)]
    result = apply_foreign_obb_mask(crop, 0, 0, other, background_color=50)
    assert result[10, 10] == 50
    assert result[40, 40] == 200


def test_apply_foreign_obb_mask_bgr_tuple_color():
    """Passing a (B, G, R) tuple fills the region with the correct colour."""
    crop = np.ones((50, 50, 3), dtype=np.uint8) * 200
    other = [_square_corners(0, 0, 20)]
    result = apply_foreign_obb_mask(crop, 0, 0, other, background_color=(10, 20, 30))
    assert result[10, 10, 0] == 10  # B
    assert result[10, 10, 1] == 20  # G
    assert result[10, 10, 2] == 30  # R
    assert result[40, 40, 0] == 200  # outside OBB unchanged


# ---------------------------------------------------------------------------
# filter_keypoints_by_foreign_obbs
# ---------------------------------------------------------------------------


def _make_kpts(*xys_conf):
    """Build (K, 3) keypoints array from (x, y, conf) tuples."""
    return np.array(xys_conf, dtype=np.float32)


def _rect_obb(x0, y0, x1, y1):
    """Axis-aligned rectangle as (4, 2) float32 OBB corners."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def test_filter_keypoints_by_foreign_obbs_none_returns_none():
    result = filter_keypoints_by_foreign_obbs(None, [_rect_obb(0, 0, 50, 50)], 0)
    assert result is None


def test_filter_keypoints_by_foreign_obbs_empty_list():
    kpts = _make_kpts((10.0, 10.0, 0.9))
    result = filter_keypoints_by_foreign_obbs(kpts, [], 0)
    assert result[0, 2] == pytest.approx(0.9)


def test_filter_keypoints_by_foreign_obbs_zero_conf_kpt_skipped():
    """Keypoints already at zero confidence should not be evaluated."""
    kpts = _make_kpts((10.0, 10.0, 0.0))
    all_obbs = [_rect_obb(100, 100, 200, 200), _rect_obb(0, 0, 50, 50)]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    # conf was 0 going in; should remain 0, not touched
    assert result[0, 2] == pytest.approx(0.0)


def test_filter_keypoints_by_foreign_obbs_own_obb_skipped():
    """Keypoints inside the target animal's own OBB should NOT be zeroed."""
    # Target animal OBB covers (0–50, 0–50); target_idx=0
    kpts = _make_kpts((10.0, 10.0, 0.9))  # inside own OBB
    all_obbs = [_rect_obb(0, 0, 50, 50)]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    assert result[0, 2] == pytest.approx(0.9)


def test_filter_keypoints_by_foreign_obbs_foreign_obb_zeroes_conf():
    """A keypoint inside another animal's OBB should have its conf zeroed."""
    # Animal 0 at (0,0)–(50,50); animal 1 at (100,100)–(200,200)
    # Keypoint at (120, 120) is inside animal 1's OBB
    kpts = _make_kpts((120.0, 120.0, 0.8))
    all_obbs = [_rect_obb(0, 0, 50, 50), _rect_obb(100, 100, 200, 200)]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    assert result[0, 2] == pytest.approx(0.0)
    # X/Y preserved
    assert result[0, 0] == pytest.approx(120.0)
    assert result[0, 1] == pytest.approx(120.0)


def test_filter_keypoints_by_foreign_obbs_outside_foreign_obb_unchanged():
    """A keypoint clearly outside all foreign OBBs should retain its conf."""
    kpts = _make_kpts((10.0, 10.0, 0.85))  # inside own (target) OBB
    all_obbs = [_rect_obb(0, 0, 50, 50), _rect_obb(100, 100, 200, 200)]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    assert result[0, 2] == pytest.approx(0.85)


def test_filter_keypoints_by_foreign_obbs_multiple_keypoints_mixed():
    """Only keypoints inside foreign OBBs should be zeroed; others unchanged."""
    # Target at (0,0)–(50,50); foreign at (80,80)–(130,130)
    kpts = _make_kpts(
        (10.0, 10.0, 0.9),  # inside own OBB — keep
        (100.0, 100.0, 0.7),  # inside foreign OBB — zero
        (200.0, 200.0, 0.6),  # outside all — keep
    )
    all_obbs = [_rect_obb(0, 0, 50, 50), _rect_obb(80, 80, 130, 130)]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    assert result[0, 2] == pytest.approx(0.9)
    assert result[1, 2] == pytest.approx(0.0)
    assert result[2, 2] == pytest.approx(0.6)


def test_filter_keypoints_by_foreign_obbs_returns_copy():
    """Original keypoints array must not be mutated."""
    kpts = _make_kpts((100.0, 100.0, 0.8))
    all_obbs = [_rect_obb(0, 0, 50, 50), _rect_obb(80, 80, 130, 130)]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    assert result[0, 2] == pytest.approx(0.0)
    assert kpts[0, 2] == pytest.approx(0.8)  # original unchanged


def test_filter_keypoints_by_foreign_obbs_ignores_bad_shape():
    """OBB arrays with wrong shape should be silently skipped."""
    kpts = _make_kpts((10.0, 10.0, 0.9))
    bad_obb = np.array([[0, 0], [10, 10]], dtype=np.float32)  # shape (2, 2)
    all_obbs = [_rect_obb(0, 0, 50, 50), bad_obb]
    result = filter_keypoints_by_foreign_obbs(kpts, all_obbs, target_idx=0)
    # Bad OBB skipped, own OBB skipped (target_idx=0) — conf unchanged
    assert result[0, 2] == pytest.approx(0.9)
