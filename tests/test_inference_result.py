import numpy as np
import pytest

from hydra_suite.core.inference.result import (
    DETECTION_ID_STRIDE,
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    FrameResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
    assemble_resolved_headings,
)


def _make_obb(n: int = 3, frame_idx: int = 0) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((n, 2)),
        angles=np.array([0.1, 0.2, 0.3][:n] + [0.0] * max(0, n - 3)),
        sizes=np.ones(n) * 500.0,
        shapes=np.ones((n, 2)),
        confidences=np.ones(n) * 0.9,
        corners=np.zeros((n, 4, 2)),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def test_obb_result_num_detections():
    obb = _make_obb(3)
    assert obb.num_detections == 3


def test_resolved_headings_fallback_to_obb():
    obb = _make_obb(2)
    headings = assemble_resolved_headings(
        obb, None, None, None, overrides_headtail=True
    )
    np.testing.assert_array_almost_equal(headings, obb.angles)


def test_resolved_headings_headtail_overrides_obb():
    obb = _make_obb(2)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5, float("nan")]),
        heading_confidences=np.array([0.9, 0.0]),
        directed_mask=np.array([1, 0], dtype=np.uint8),
        canonical_affines=np.zeros((2, 2, 3)),
    )
    headings = assemble_resolved_headings(obb, headtail, None, None)
    assert headings[0] == pytest.approx(1.5)
    assert headings[1] == pytest.approx(0.2)


def test_resolved_headings_pose_overrides_headtail():
    obb = _make_obb(2)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5, 2.0]),
        heading_confidences=np.array([0.9, 0.9]),
        directed_mask=np.array([1, 1], dtype=np.uint8),
        canonical_affines=np.zeros((2, 2, 3)),
    )
    pose_headings = np.array([0.3, float("nan")])
    pose_valid = np.array([True, False])
    headings = assemble_resolved_headings(
        obb, headtail, pose_headings, pose_valid, overrides_headtail=True
    )
    assert headings[0] == pytest.approx(0.3)
    assert headings[1] == pytest.approx(2.0)


def test_resolved_headings_pose_does_not_override_when_flag_false():
    obb = _make_obb(1)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5]),
        heading_confidences=np.array([0.9]),
        directed_mask=np.array([1], dtype=np.uint8),
        canonical_affines=np.zeros((1, 2, 3)),
    )
    pose_headings = np.array([0.3])
    pose_valid = np.array([True])
    headings = assemble_resolved_headings(
        obb, headtail, pose_headings, pose_valid, overrides_headtail=False
    )
    assert headings[0] == pytest.approx(1.5)


def test_cnn_result_multi_head_structure():
    pred = CNNDetectionPrediction(
        det_index=0,
        factors=[
            CNNFactorPrediction("color", ["red", "blue"], np.array([0.8, 0.2])),
            CNNFactorPrediction("size", ["small", "large"], np.array([0.3, 0.7])),
        ],
    )
    assert len(pred.factors) == 2
    assert pred.factors[0].factor_name == "color"
    np.testing.assert_array_almost_equal(pred.factors[1].raw_probabilities, [0.3, 0.7])


def test_apriltag_result_empty():
    result = AprilTagResult(
        tag_ids=[],
        det_indices=[],
        centers=np.zeros((0, 2)),
        corners=np.zeros((0, 4, 2)),
    )
    assert len(result.tag_ids) == 0


def test_detection_ids_are_unique_across_frames():
    """Detection IDs must be unique and follow the legacy stride convention."""
    ids_f0 = OBBResult.make_detection_ids(0, 5)
    ids_f1 = OBBResult.make_detection_ids(1, 5)
    ids_f100 = OBBResult.make_detection_ids(100, 3)
    assert set(ids_f0).isdisjoint(set(ids_f1))
    assert set(ids_f0).isdisjoint(set(ids_f100))
    assert ids_f0.dtype == np.int64
    # Stride preserved: id 0 of frame 1 = STRIDE
    assert ids_f1[0] == DETECTION_ID_STRIDE
    # Slot 2 of frame 100 = 100 * STRIDE + 2
    assert ids_f100[2] == 100 * DETECTION_ID_STRIDE + 2


def test_obb_result_carries_detection_ids():
    obb = _make_obb(4, frame_idx=7)
    assert obb.detection_ids.shape == (4,)
    assert obb.detection_ids[0] == 7 * DETECTION_ID_STRIDE
    assert obb.detection_ids[3] == 7 * DETECTION_ID_STRIDE + 3


def test_headtail_result_canonical_affines_optional():
    """canonical_affines must accept None for cache-loaded results."""
    ht = HeadTailResult(
        heading_hints=np.array([1.5]),
        heading_confidences=np.array([0.9]),
        directed_mask=np.array([1], dtype=np.uint8),
        canonical_affines=None,
    )
    assert ht.canonical_affines is None


def test_resolved_headings_works_with_none_canonical_affines():
    """assemble_resolved_headings must work when canonical_affines is None."""
    obb = _make_obb(2)
    headtail = HeadTailResult(
        heading_hints=np.array([1.5, float("nan")]),
        heading_confidences=np.array([0.9, 0.0]),
        directed_mask=np.array([1, 0], dtype=np.uint8),
        canonical_affines=None,
    )
    headings = assemble_resolved_headings(obb, headtail, None, None)
    assert headings[0] == pytest.approx(1.5)
    assert headings[1] == pytest.approx(0.2)


def test_frame_result_construction():
    """FrameResult holds typed inference outputs and resolved headings."""
    obb = _make_obb(2)
    fr = FrameResult(
        frame_idx=0,
        obb=obb,
        filtered_indices=[0, 1],
        headtail=None,
        cnn=[],
        pose=None,
        apriltag=None,
        resolved_headings=obb.angles.copy(),
    )
    assert fr.obb.num_detections == 2
    assert fr.filtered_indices == [0, 1]
    np.testing.assert_array_almost_equal(fr.resolved_headings, obb.angles)


def test_pose_result_construction():
    pose = PoseResult(
        keypoints=np.zeros((2, 5, 3)),
        valid_mask=np.array([True, False]),
    )
    assert pose.keypoints.shape == (2, 5, 3)
    assert pose.valid_mask[0] is np.True_ or pose.valid_mask[0] == True  # noqa: E712


def test_cnn_result_label_and_predictions():
    cnn = CNNResult(
        label="identity",
        predictions=[
            CNNDetectionPrediction(
                det_index=0,
                factors=[
                    CNNFactorPrediction(
                        "id", ["a", "b"], np.array([0.7, 0.3], dtype=np.float32)
                    )
                ],
            )
        ],
    )
    assert cnn.label == "identity"
    assert len(cnn.predictions) == 1
    assert cnn.predictions[0].det_index == 0
