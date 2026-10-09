import numpy as np
import pytest

from hydra_suite.core.inference.downstream_select import (
    DownstreamCacheError,
    apriltag_positions_to_raw,
    apriltag_raw_to_positions,
    cnn_positions_to_raw,
    cnn_raw_to_positions,
    concat_cnn,
    concat_headtail,
    positions_in,
    select_cnn,
    select_headtail,
    select_pose,
    split_rows,
)
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
)


def _ht(vals):
    v = np.asarray(vals, np.float32)
    return HeadTailResult(v, v / 10, (v > 0).astype(np.uint8), None)


def _cnn(n, label="id"):
    return CNNResult(label, [CNNDetectionPrediction(i, []) for i in range(n)])


def test_positions_in_and_missing():
    sup = np.array([2, 5, 9, 11])
    assert positions_in(sup, np.array([5, 11])).tolist() == [1, 3]
    with pytest.raises(DownstreamCacheError, match="7"):
        positions_in(sup, np.array([7]))


def test_select_headtail_and_pose():
    ht = select_headtail(_ht([1.0, 2.0, 3.0]), np.array([2, 0]))
    assert ht.heading_hints.tolist() == [3.0, 1.0]
    p = PoseResult(
        np.arange(6, dtype=np.float32).reshape(3, 1, 2), np.array([1, 0, 1], bool)
    )
    p.heading_overrides = np.array([0.1, 0.2, 0.3], np.float32)
    s = select_pose(p, np.array([1]))
    assert s.keypoints.tolist() == [[[2.0, 3.0]]]
    assert s.heading_overrides.tolist() == [pytest.approx(0.2)]


def test_cnn_raw_roundtrip():
    sup = np.array([4, 8, 15])
    raw = cnn_positions_to_raw(_cnn(3), sup)
    assert [p.det_index for p in raw.predictions] == [4, 8, 15]
    back = cnn_raw_to_positions(raw.predictions, np.array([15, 4]), "id")
    assert [p.det_index for p in back.predictions] == [0, 1]
    with pytest.raises(DownstreamCacheError):
        cnn_raw_to_positions(raw.predictions, np.array([16]), "id")


def test_select_cnn_renumbers():
    s = select_cnn(_cnn(4), np.array([3, 1]))
    assert [p.det_index for p in s.predictions] == [0, 1]


def test_apriltag_roundtrip():
    at = AprilTagResult(
        [7, 9], [0, 2], np.zeros((2, 2), np.float32), np.zeros((2, 4, 2), np.float32)
    )
    raw = apriltag_positions_to_raw(at, np.array([10, 20, 30]))
    assert raw.det_indices == [10, 30]
    back = apriltag_raw_to_positions(raw, np.array([30]))
    assert back.tag_ids == [9] and back.det_indices == [0]


def test_split_and_concat():
    n = 5
    obb = OBBResult(
        0,
        np.zeros((n, 2), np.float32),
        np.zeros(n, np.float32),
        np.ones(n, np.float32),
        np.ones((n, 2), np.float32),
        np.ones(n, np.float32),
        np.zeros((n, 4, 2), np.float32),
        OBBResult.make_detection_ids(0, n),
    )
    chunks = split_rows(obb, 2)
    assert [(o, c.num_detections) for o, c in chunks] == [(0, 2), (2, 2), (4, 1)]
    ht = concat_headtail([_ht([1, 2]), _ht([3, 4]), _ht([5])])
    assert ht.heading_hints.tolist() == [1, 2, 3, 4, 5]
    cnn = concat_cnn([(0, _cnn(2)), (2, _cnn(2)), (4, _cnn(1))])
    assert [p.det_index for p in cnn.predictions] == [0, 1, 2, 3, 4]
