"""Tests for pose/features.py adapter to new PoseCache + DetectionCache (Task 17b).

Verifies that build_pose_detection_keypoint_map works with both the legacy
IndividualPropertiesCache interface and the new cache pair.
"""

from unittest.mock import MagicMock

import numpy as np

from hydra_suite.core.identity.pose.features import build_pose_detection_keypoint_map


def _make_mock_pose_cache(frame_idx: int = 0, n_dets: int = 3, n_kpts: int = 5):
    """Mock PoseCache.read_frame() returning a PoseResult-like object."""
    pose_result = MagicMock()
    pose_result.keypoints = (
        np.random.default_rng(42)
        .uniform(0, 100, (n_dets, n_kpts, 3))
        .astype(np.float32)
    )
    pose_result.valid_mask = np.ones(n_dets, dtype=bool)
    cache = MagicMock()
    cache.read_frame.return_value = pose_result
    return cache


def _make_mock_detection_cache(frame_idx: int = 0, n_dets: int = 3):
    """Mock DetectionCache.read_frame() returning an OBBResult-like object."""
    from hydra_suite.core.inference.result import OBBResult

    obb = OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((n_dets, 2), dtype=np.float32),
        angles=np.zeros(n_dets, dtype=np.float32),
        sizes=np.ones(n_dets, dtype=np.float32) * 100,
        shapes=np.ones((n_dets, 2), dtype=np.float32),
        confidences=np.ones(n_dets, dtype=np.float32) * 0.9,
        corners=np.zeros((n_dets, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n_dets),
    )
    cache = MagicMock()
    cache.read_frame.return_value = obb
    return cache


def test_build_keypoint_map_with_new_cache_returns_dict():
    """New-cache path returns {detection_id: keypoints} dict."""
    pose_cache = _make_mock_pose_cache(n_dets=3)
    det_cache = _make_mock_detection_cache(n_dets=3)

    result = build_pose_detection_keypoint_map(
        pose_cache, frame_idx=0, detection_cache=det_cache
    )
    assert isinstance(result, dict)
    assert len(result) == 3


def test_build_keypoint_map_keys_are_detection_ids():
    """Keys must be the int64 detection IDs from OBBResult.make_detection_ids."""
    from hydra_suite.core.inference.result import OBBResult

    pose_cache = _make_mock_pose_cache(n_dets=2)
    det_cache = _make_mock_detection_cache(frame_idx=5, n_dets=2)

    result = build_pose_detection_keypoint_map(
        pose_cache, frame_idx=5, detection_cache=det_cache
    )
    expected_ids = OBBResult.make_detection_ids(5, 2)
    assert set(result.keys()) == set(expected_ids.tolist())


def test_build_keypoint_map_returns_empty_when_cache_none():
    """None cache must return empty dict (legacy + new API)."""
    assert build_pose_detection_keypoint_map(None, frame_idx=0) == {}
    assert (
        build_pose_detection_keypoint_map(None, frame_idx=0, detection_cache=None) == {}
    )


def test_build_keypoint_map_returns_empty_when_pose_read_fails():
    """If read_frame raises, return empty dict gracefully."""
    pose_cache = MagicMock()
    pose_cache.read_frame.side_effect = RuntimeError("cache error")
    det_cache = _make_mock_detection_cache(n_dets=2)

    result = build_pose_detection_keypoint_map(
        pose_cache, frame_idx=0, detection_cache=det_cache
    )
    assert result == {}


def test_build_keypoint_map_returns_empty_when_det_read_returns_none():
    """If detection_cache.read_frame returns None, return empty dict."""
    pose_cache = _make_mock_pose_cache(n_dets=2)
    det_cache = MagicMock()
    det_cache.read_frame.return_value = None

    result = build_pose_detection_keypoint_map(
        pose_cache, frame_idx=0, detection_cache=det_cache
    )
    assert result == {}


def test_legacy_path_still_works():
    """Legacy IndividualPropertiesCache.get_frame() path must still function."""
    from hydra_suite.core.inference.result import OBBResult

    detection_ids = OBBResult.make_detection_ids(0, 2)
    keypoints = np.random.default_rng(1).uniform(0, 100, (2, 5, 3)).astype(np.float32)

    legacy_cache = MagicMock()
    legacy_cache.get_frame.return_value = {
        "detection_ids": detection_ids,
        "pose_keypoints": keypoints,
    }

    result = build_pose_detection_keypoint_map(legacy_cache, frame_idx=0)
    assert len(result) == 2
    assert set(result.keys()) == set(detection_ids.tolist())
