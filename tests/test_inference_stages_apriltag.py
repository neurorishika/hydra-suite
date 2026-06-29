from unittest.mock import MagicMock

import numpy as np

from hydra_suite.core.inference.config import AprilTagConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.apriltag import (
    AprilTagModel,
    _preprocess_crop,
    run_apriltag,
)


def _obb(n: int) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2), dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def test_run_apriltag_disabled_returns_empty():
    config = AprilTagConfig(enabled=False)
    model = AprilTagModel(detector=None, config=config)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    result = run_apriltag(crops, _obb(1), model, config)
    assert len(result.tag_ids) == 0


def test_run_apriltag_empty_crops():
    config = AprilTagConfig(enabled=True)
    model = AprilTagModel(detector=MagicMock(), config=config)
    result = run_apriltag([], _obb(0), model, config)
    assert len(result.tag_ids) == 0
    assert result.centers.shape == (0, 2)


def test_preprocess_crop_returns_uint8():
    crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    config = AprilTagConfig(contrast_factor=1.5, unsharp_amount=1.5)
    result = _preprocess_crop(crop, config)
    assert result.dtype == np.uint8
    assert result.shape == crop.shape


def test_run_apriltag_no_detections():
    config = AprilTagConfig(enabled=True, tag_family="tag36h11")
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    model = AprilTagModel(detector=mock_detector, config=config)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    result = run_apriltag(crops, _obb(1), model, config)
    assert len(result.tag_ids) == 0


def test_run_apriltag_collects_detections():
    """Multiple tags from multiple crops are collected with correct det_indices."""
    config = AprilTagConfig(enabled=True, tag_family="tag36h11")

    def _det(tag_id, cx, cy):
        # Lab apriltag fork returns dict-style detections.
        return {
            "id": tag_id,
            "center": (cx, cy),
            "lb-rb-rt-lt": [
                (cx - 1, cy - 1),
                (cx + 1, cy - 1),
                (cx + 1, cy + 1),
                (cx - 1, cy + 1),
            ],
            "hamming": 0,
        }

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = [
        [_det(7, 10.0, 20.0)],  # crop 0 → 1 tag
        [],  # crop 1 → no tags
        [_det(3, 5.0, 5.0), _det(4, 30.0, 30.0)],  # crop 2 → 2 tags
    ]
    model = AprilTagModel(detector=mock_detector, config=config)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)] * 3
    result = run_apriltag(crops, _obb(3), model, config)
    assert result.tag_ids == [7, 3, 4]
    assert result.det_indices == [0, 2, 2]
    assert result.centers.shape == (3, 2)


def test_run_apriltag_max_tag_id_filter():
    """tags above max_tag_id are filtered out."""
    config = AprilTagConfig(enabled=True, max_tag_id=10)

    def _det(tag_id):
        # Lab apriltag fork returns dict-style detections.
        return {
            "id": tag_id,
            "center": (0.0, 0.0),
            "lb-rb-rt-lt": [(0, 0), (1, 0), (1, 1), (0, 1)],
            "hamming": 0,
        }

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [_det(5), _det(15), _det(8)]
    model = AprilTagModel(detector=mock_detector, config=config)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8)]
    result = run_apriltag(crops, _obb(1), model, config)
    assert result.tag_ids == [5, 8]
