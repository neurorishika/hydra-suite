"""Tests for frame_result_bridge helpers.

Verifies that populate_live_cnn_store, populate_live_pose_store, and
populate_live_tag_store produce the same keys / shapes that the legacy
precompute callback path would have written into the live stores.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    OBBResult,
    PoseResult,
)
from hydra_suite.core.tracking.frame_result_bridge import (
    frame_result_to_meas,
    populate_live_cnn_store,
    populate_live_pose_store,
    populate_live_tag_store,
)
from hydra_suite.core.tracking.live_features import (
    LiveCNNIdentityStore,
    LivePosePropertiesStore,
    LiveTagObservationStore,
)

# ---- Helpers to build fake FrameResult components ----


def _make_obb(frame_idx: int, n: int = 2) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array([[10.0, 20.0], [30.0, 40.0]][:n], dtype=np.float32),
        angles=np.array([0.1, 0.2][:n], dtype=np.float32),
        sizes=np.array([100.0, 150.0][:n], dtype=np.float32),
        shapes=np.array([[80.0, 1.5], [120.0, 1.8]][:n], dtype=np.float32),
        confidences=np.array([0.9, 0.8][:n], dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _make_cnn_result(label: str, n: int = 2, n_classes: int = 3) -> CNNResult:
    probs = np.zeros(n_classes, dtype=np.float32)
    probs[1] = 0.9
    predictions = [
        CNNDetectionPrediction(
            det_index=i,
            factors=[
                CNNFactorPrediction(
                    factor_name="flat",
                    class_names=[f"class_{j}" for j in range(n_classes)],
                    raw_probabilities=probs.copy() if i == 0 else probs[::-1].copy(),
                )
            ],
        )
        for i in range(n)
    ]
    return CNNResult(label=label, predictions=predictions)


def _make_pose_result(n: int = 2, n_kpts: int = 4) -> PoseResult:
    kpts = np.random.rand(n, n_kpts, 3).astype(np.float32)
    kpts[:, :, 2] = 0.8  # all confident
    valid_mask = np.array([True] * n, dtype=bool)
    return PoseResult(keypoints=kpts, valid_mask=valid_mask)


def _make_apriltag_result(n_tags: int = 2) -> AprilTagResult:
    return AprilTagResult(
        tag_ids=list(range(n_tags)),
        det_indices=list(range(n_tags)),
        centers=np.array([[5.0, 6.0], [7.0, 8.0]][:n_tags], dtype=np.float32),
        corners=np.zeros((n_tags, 4, 2), dtype=np.float32),
    )


# ---- frame_result_to_meas ----


class TestFrameResultToMeas:
    def test_returns_list_of_arrays(self):
        centroids = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        headings = np.array([0.5, 1.2], dtype=np.float32)
        meas = frame_result_to_meas(centroids, headings)
        assert len(meas) == 2
        assert meas[0].shape == (3,)
        assert meas[1].shape == (3,)

    def test_values_match_centroid_and_heading(self):
        centroids = np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32)
        headings = np.array([0.77, 1.33], dtype=np.float32)
        meas = frame_result_to_meas(centroids, headings)
        np.testing.assert_allclose(meas[0], [11.0, 22.0, 0.77], rtol=1e-5)
        np.testing.assert_allclose(meas[1], [33.0, 44.0, 1.33], rtol=1e-5)

    def test_empty_returns_empty_list(self):
        centroids = np.zeros((0, 2), dtype=np.float32)
        headings = np.zeros(0, dtype=np.float32)
        meas = frame_result_to_meas(centroids, headings)
        assert meas == []

    def test_single_detection(self):
        centroids = np.array([[5.0, 10.0]], dtype=np.float32)
        headings = np.array([2.5], dtype=np.float32)
        meas = frame_result_to_meas(centroids, headings)
        assert len(meas) == 1
        assert float(meas[0][2]) == pytest.approx(2.5, rel=1e-5)


# ---- populate_live_cnn_store ----


class TestPopulateLiveCNNStore:
    def test_basic_two_detections(self):
        store = LiveCNNIdentityStore()
        cnn_result = _make_cnn_result("id_cnn", n=2)
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_cnn_store(
            store, [cnn_result], det_ids, frame_idx=5, phase_label="id_cnn"
        )

        preds = store.load(5)
        assert len(preds) == 2

    def test_class_names_are_argmax_of_probabilities(self):
        """The top class name matches the argmax of raw_probabilities."""
        store = LiveCNNIdentityStore()
        probs = np.array([0.1, 0.7, 0.2], dtype=np.float32)
        cnn_result = CNNResult(
            label="test",
            predictions=[
                CNNDetectionPrediction(
                    det_index=0,
                    factors=[
                        CNNFactorPrediction(
                            factor_name="flat",
                            class_names=["a", "b", "c"],
                            raw_probabilities=probs,
                        )
                    ],
                )
            ],
        )
        det_ids = np.array([0], dtype=np.int64)
        populate_live_cnn_store(
            store, [cnn_result], det_ids, frame_idx=0, phase_label="test"
        )

        preds = store.load(0)
        assert len(preds) == 1
        assert preds[0].class_names[0] == "b"
        assert preds[0].confidences[0] == pytest.approx(0.7, rel=1e-5)

    def test_no_matching_phase_stores_empty(self):
        store = LiveCNNIdentityStore()
        cnn_result = _make_cnn_result("phase_a")
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_cnn_store(
            store, [cnn_result], det_ids, frame_idx=3, phase_label="phase_b"
        )

        preds = store.load(3)
        assert preds == []

    def test_empty_frame_produces_empty_predictions(self):
        store = LiveCNNIdentityStore()
        cnn_result = CNNResult(label="id_cnn", predictions=[])
        det_ids = np.zeros(0, dtype=np.int64)

        populate_live_cnn_store(
            store, [cnn_result], det_ids, frame_idx=10, phase_label="id_cnn"
        )

        preds = store.load(10)
        assert preds == []

    def test_det_index_preserved(self):
        store = LiveCNNIdentityStore()
        probs = np.array([0.3, 0.7], dtype=np.float32)
        cnn_result = CNNResult(
            label="cls",
            predictions=[
                CNNDetectionPrediction(
                    det_index=7,
                    factors=[
                        CNNFactorPrediction(
                            factor_name="flat",
                            class_names=["x", "y"],
                            raw_probabilities=probs,
                        )
                    ],
                )
            ],
        )
        det_ids = np.array([7], dtype=np.int64)
        populate_live_cnn_store(
            store, [cnn_result], det_ids, frame_idx=0, phase_label="cls"
        )

        preds = store.load(0)
        assert len(preds) == 1
        assert preds[0].det_index == 7


# ---- populate_live_pose_store ----


class TestPopulateLivePoseStore:
    def test_basic_two_detections(self):
        store = LivePosePropertiesStore()
        pose = _make_pose_result(n=2, n_kpts=4)
        det_ids = np.array([100, 101], dtype=np.int64)

        populate_live_pose_store(store, pose, det_ids, frame_idx=3)

        frame_data = store.get_frame(3)
        assert len(frame_data["detection_ids"]) == 2
        assert frame_data["detection_ids"][0] == 100
        assert frame_data["detection_ids"][1] == 101

    def test_keypoints_shape_matches(self):
        store = LivePosePropertiesStore()
        pose = _make_pose_result(n=2, n_kpts=5)
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_pose_store(store, pose, det_ids, frame_idx=0)

        frame_data = store.get_frame(0)
        kpts = frame_data["pose_keypoints"]
        assert len(kpts) == 2
        assert kpts[0].shape == (5, 3)

    def test_invalid_detection_stored_as_none(self):
        store = LivePosePropertiesStore()
        kpts = np.ones((2, 3, 3), dtype=np.float32)
        valid = np.array([True, False], dtype=bool)
        pose = PoseResult(keypoints=kpts, valid_mask=valid)
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_pose_store(store, pose, det_ids, frame_idx=0)

        frame_data = store.get_frame(0)
        assert frame_data["pose_keypoints"][0] is not None
        assert frame_data["pose_keypoints"][1] is None

    def test_none_pose_stores_empty(self):
        store = LivePosePropertiesStore()
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_pose_store(store, None, det_ids, frame_idx=7)

        frame_data = store.get_frame(7)
        assert frame_data["detection_ids"] == []

    def test_empty_det_ids_stores_empty(self):
        store = LivePosePropertiesStore()
        pose = _make_pose_result(n=0)
        det_ids = np.zeros(0, dtype=np.int64)

        populate_live_pose_store(store, pose, det_ids, frame_idx=0)

        frame_data = store.get_frame(0)
        assert frame_data["detection_ids"] == []


# ---- populate_live_tag_store ----


class TestPopulateLiveTagStore:
    def test_basic_two_tags(self):
        store = LiveTagObservationStore()
        at = _make_apriltag_result(n_tags=2)
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_tag_store(store, at, det_ids, frame_idx=4)

        frame_data = store.get_frame(4)
        assert list(frame_data["tag_ids"]) == [0, 1]

    def test_det_indices_preserved(self):
        store = LiveTagObservationStore()
        at = AprilTagResult(
            tag_ids=[5, 9],
            det_indices=[2, 3],
            centers=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            corners=np.zeros((2, 4, 2), dtype=np.float32),
        )
        det_ids = np.array([2, 3], dtype=np.int64)

        populate_live_tag_store(store, at, det_ids, frame_idx=1)

        frame_data = store.get_frame(1)
        assert list(frame_data["det_indices"]) == [2, 3]

    def test_centers_match(self):
        store = LiveTagObservationStore()
        at = _make_apriltag_result(n_tags=1)
        det_ids = np.array([0], dtype=np.int64)

        populate_live_tag_store(store, at, det_ids, frame_idx=0)

        frame_data = store.get_frame(0)
        np.testing.assert_allclose(frame_data["centers_xy"][0], [5.0, 6.0], rtol=1e-5)

    def test_none_apriltag_stores_empty(self):
        store = LiveTagObservationStore()
        det_ids = np.array([0], dtype=np.int64)

        populate_live_tag_store(store, None, det_ids, frame_idx=2)

        frame_data = store.get_frame(2)
        assert len(frame_data["tag_ids"]) == 0

    def test_empty_tags_stores_empty(self):
        store = LiveTagObservationStore()
        at = AprilTagResult(
            tag_ids=[],
            det_indices=[],
            centers=np.zeros((0, 2), dtype=np.float32),
            corners=np.zeros((0, 4, 2), dtype=np.float32),
        )
        det_ids = np.array([0, 1], dtype=np.int64)

        populate_live_tag_store(store, at, det_ids, frame_idx=0)

        frame_data = store.get_frame(0)
        assert len(frame_data["tag_ids"]) == 0
