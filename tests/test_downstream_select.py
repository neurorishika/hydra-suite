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


def test_select_cnn_missing_raises():
    with pytest.raises(DownstreamCacheError, match="5"):
        select_cnn(_cnn(2), np.array([5]))


def _line_obb(n):
    from hydra_suite.core.inference.result import OBBResult

    xs = np.arange(n, dtype=np.float32) * 10.0
    c = np.stack([xs, np.zeros(n, np.float32)], 1)
    corners = np.stack([c + d for d in ([-1, -1], [1, -1], [1, 1], [-1, 1])], 1)
    return OBBResult(
        0,
        c,
        np.zeros(n, np.float32),
        np.ones(n, np.float32),
        np.ones((n, 2), np.float32),
        np.ones(n, np.float32),
        corners.astype(np.float32),
        OBBResult.make_detection_ids(0, n),
    )


def test_run_superset_chunked_concats_and_keeps_phase_alignment():
    from hydra_suite.core.inference.downstream_select import (
        narrow_to_final,
        run_superset_chunked,
    )
    from hydra_suite.core.inference.result import CNNDetectionPrediction, CNNResult

    sup = _line_obb(5)
    seen = []

    def _run_chunk(chunk, foreign):
        seen.append((chunk.num_detections, foreign.self_rows.tolist()))
        assert len(foreign) == 5  # always the FULL superset
        xs = chunk.centroids[:, 0]
        cnn_b = CNNResult(
            "b",
            [
                CNNDetectionPrediction(det_index=i, factors=[float(x)])
                for i, x in enumerate(xs)
            ],
        )
        # phase "a" never produces a result -> None placeholder at index 0
        return _ht(xs), [None, cnn_b], None, None

    ht, cnn, pose, at = run_superset_chunked(sup, _run_chunk, 2, 2)
    assert seen == [(2, [0, 1]), (2, [2, 3]), (1, [4])]
    assert ht.heading_hints.tolist() == [0, 10, 20, 30, 40]
    assert cnn[0] is None
    assert [p.det_index for p in cnn[1].predictions] == [0, 1, 2, 3, 4]
    assert pose is None and at is None

    # superset raw [3, 5, 6, 8, 9]; final raw [5, 8] -> positions [1, 3]
    ht_f, cnn_f, pose_f, at_f = narrow_to_final(
        np.array([3, 5, 6, 8, 9]), np.array([5, 8]), ht, cnn, pose, at
    )
    assert ht_f.heading_hints.tolist() == [10, 30]
    assert len(cnn_f) == 1  # the None phase is dropped in memory
    assert [p.det_index for p in cnn_f[0].predictions] == [0, 1]
    assert [p.factors[0] for p in cnn_f[0].predictions] == [10.0, 30.0]
    assert pose_f is None and at_f is None
    with pytest.raises(DownstreamCacheError):
        narrow_to_final(np.array([3, 5]), np.array([4]), ht, cnn, pose, at)
