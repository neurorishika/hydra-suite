"""Tests for foreign-OBB pose-keypoint suppression in scatter.

Unit test for suppress_foreign_keypoints:
  - keypoint inside target OBB: confidence kept
  - keypoint inside a foreign OBB: confidence zeroed

Scatter-level integration tests:
  - flag ON: keypoint inside neighbor's OBB is zeroed
  - flag OFF: keypoint inside neighbor's OBB is kept
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from hydra_suite.core.inference.stages.assemble import suppress_foreign_keypoints

if TYPE_CHECKING:
    from hydra_suite.core.inference.config import InferenceConfig
    from hydra_suite.core.inference.result import OBBResult, PoseResult

# ---------------------------------------------------------------------------
# Unit tests: suppress_foreign_keypoints
# ---------------------------------------------------------------------------


def test_keypoint_inside_foreign_obb_is_zeroed():
    # keypoint at (40,20) lands inside the foreign box [30..50]x[10..30]
    kpts = np.array(
        [[[15.0, 20.0, 0.9], [40.0, 20.0, 0.9]]], np.float32
    )  # (1 det, 2 kpts, 3)
    target_corners = np.array([[10, 10], [25, 10], [25, 30], [10, 30]], np.float32)
    foreign = [np.array([[30, 10], [50, 10], [50, 30], [30, 30]], np.float32)]
    out = suppress_foreign_keypoints(kpts[0], target_corners, foreign)
    assert out[0, 2] == 0.9  # inside target → kept
    assert out[1, 2] == 0.0  # inside foreign → zeroed


def test_keypoint_outside_all_obbs_is_kept():
    kpts = np.array([[5.0, 5.0, 0.8]], np.float32)  # outside both OBBs
    target_corners = np.array([[10, 10], [25, 10], [25, 30], [10, 30]], np.float32)
    foreign = [np.array([[30, 10], [50, 10], [50, 30], [30, 30]], np.float32)]
    out = suppress_foreign_keypoints(kpts, target_corners, foreign)
    assert out[0, 2] == pytest.approx(0.8)


def test_keypoint_with_zero_confidence_is_unchanged():
    # zero-confidence keypoints should not be processed (already suppressed)
    kpts = np.array([[40.0, 20.0, 0.0]], np.float32)  # inside foreign OBB but conf=0
    target_corners = np.array([[10, 10], [25, 10], [25, 30], [10, 30]], np.float32)
    foreign = [np.array([[30, 10], [50, 10], [50, 30], [30, 30]], np.float32)]
    out = suppress_foreign_keypoints(kpts, target_corners, foreign)
    assert out[0, 2] == 0.0  # stayed 0 (was already 0)


def test_no_foreign_obbs_returns_unchanged():
    kpts = np.array([[15.0, 20.0, 0.9]], np.float32)
    target_corners = np.array([[10, 10], [25, 10], [25, 30], [10, 30]], np.float32)
    out = suppress_foreign_keypoints(kpts, target_corners, [])
    assert out[0, 2] == pytest.approx(0.9)


def test_none_keypoints_returns_none():
    target_corners = np.array([[10, 10], [25, 10], [25, 30], [10, 30]], np.float32)
    out = suppress_foreign_keypoints(None, target_corners, [])
    assert out is None


# ---------------------------------------------------------------------------
# Scatter-level integration tests
# ---------------------------------------------------------------------------


def _make_obb_result(frame_idx: int, corners_list: list[np.ndarray]) -> "OBBResult":
    """Build an OBBResult from a list of (4,2) corner arrays."""
    from hydra_suite.core.inference.result import OBBResult

    n = len(corners_list)
    if n == 0:
        return OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), np.float32),
            angles=np.zeros(0, np.float32),
            sizes=np.zeros(0, np.float32),
            shapes=np.zeros((0, 2), np.float32),
            confidences=np.zeros(0, np.float32),
            corners=np.zeros((0, 4, 2), np.float32),
            detection_ids=np.zeros(0, np.int64),
        )
    corners = np.stack(corners_list, axis=0).astype(np.float32)  # (n, 4, 2)
    centroids = corners.mean(axis=1)  # (n, 2)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 100.0, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def _make_pose_result(keypoints_nd: np.ndarray) -> "PoseResult":
    """Build PoseResult from (D, K, 3) keypoints."""
    from hydra_suite.core.inference.result import PoseResult

    D = keypoints_nd.shape[0]
    return PoseResult(
        keypoints=keypoints_nd.astype(np.float32),
        valid_mask=np.ones(D, dtype=bool),
    )


def _make_inference_config(suppress: bool) -> "InferenceConfig":
    """Build minimal InferenceConfig with pose.suppress_foreign_regions set."""
    from hydra_suite.core.inference.config import (
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
        PoseConfig,
    )

    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/fake.pt"),
        ),
        pose=PoseConfig(suppress_foreign_regions=suppress),
    )


def test_scatter_foreign_suppression_flag_on_zeroes_overlapping_keypoint():
    """With suppress_foreign_regions=True, keypoint inside neighbor OBB is zeroed."""
    from hydra_suite.core.inference.stages.assemble import scatter

    # Detection 0: box [0,0]-[20,20]
    corners0 = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    # Detection 1: box [30,0]-[60,20]
    corners1 = np.array([[30, 0], [60, 0], [60, 20], [30, 20]], np.float32)
    obb = _make_obb_result(0, [corners0, corners1])

    # Det 0's keypoint at (40, 10) falls inside det 1's box — should be zeroed
    # Det 1's keypoint at (45, 10) is inside its own box — should be kept
    kpts = np.array(
        [
            [[40.0, 10.0, 0.9]],  # det 0: inside foreign (det 1's OBB)
            [[45.0, 10.0, 0.9]],  # det 1: inside own OBB
        ],
        dtype=np.float32,
    )  # (2, 1, 3)
    pose = _make_pose_result(kpts)
    config = _make_inference_config(suppress=True)

    results = scatter(
        {0: obb}, headtail=None, cnns=None, pose={0: pose}, apriltag=None, config=config
    )
    out_kpts = results[0].pose.keypoints

    assert out_kpts[0, 0, 2] == 0.0, "det 0 kpt inside foreign OBB should be zeroed"
    assert out_kpts[1, 0, 2] == pytest.approx(
        0.9
    ), "det 1 kpt inside own OBB should be kept"


def test_scatter_foreign_suppression_flag_off_keeps_overlapping_keypoint():
    """With suppress_foreign_regions=False, keypoint inside neighbor OBB is kept."""
    from hydra_suite.core.inference.stages.assemble import scatter

    corners0 = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    corners1 = np.array([[30, 0], [60, 0], [60, 20], [30, 20]], np.float32)
    obb = _make_obb_result(0, [corners0, corners1])

    kpts = np.array(
        [
            [[40.0, 10.0, 0.9]],  # det 0: inside foreign but flag OFF
            [[45.0, 10.0, 0.9]],
        ],
        dtype=np.float32,
    )
    pose = _make_pose_result(kpts)
    config = _make_inference_config(suppress=False)

    results = scatter(
        {0: obb}, headtail=None, cnns=None, pose={0: pose}, apriltag=None, config=config
    )
    out_kpts = results[0].pose.keypoints

    assert out_kpts[0, 0, 2] == pytest.approx(
        0.9
    ), "det 0 kpt should be kept when flag OFF"
