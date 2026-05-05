import numpy as np
import pytest

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.store import (
    AprilTagCacheHandle,
    CNNCacheHandle,
    DetectionCacheHandle,
    HeadTailCacheHandle,
    PoseCacheHandle,
)
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    OBBResult,
)


def _key(path="/m.pt") -> CacheKey:
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=path,
        model_mtime=0.0,
        config_hash="abc",
    )


def _obb(frame_idx: int, n: int = 2) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.ones((n, 2), dtype=np.float32) * frame_idx,
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _cnn_preds(n_dets: int = 2) -> list[CNNDetectionPrediction]:
    return [
        CNNDetectionPrediction(
            det_index=i,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["red", "blue"],
                    raw_probabilities=np.array([0.7, 0.3], dtype=np.float32),
                ),
                CNNFactorPrediction(
                    factor_name="size",
                    class_names=["small", "medium", "large"],
                    raw_probabilities=np.array([0.1, 0.6, 0.3], dtype=np.float32),
                ),
            ],
        )
        for i in range(n_dets)
    ]


# ---- DetectionCacheHandle ----


def test_detection_round_trip(tmp_path):
    path = tmp_path / "test.obb.npz"
    key = _key()
    handle = DetectionCacheHandle(path=path, key=key)
    assert not handle.is_valid()

    handle.write_frame(0, result=_obb(0, n=2))
    handle.write_frame(1, result=_obb(1, n=3))
    handle.write_frame(2, result=_obb(2, n=0))
    handle.close()

    handle2 = DetectionCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    r0 = handle2.read_frame(0)
    assert r0.num_detections == 2
    assert r0.centroids[0, 0] == pytest.approx(0.0)
    # Per Correction 14: detection_ids must round-trip through cache
    assert r0.detection_ids.shape == (2,)
    assert r0.detection_ids[0] == 0  # frame_idx=0, slot 0
    r1 = handle2.read_frame(1)
    assert r1.num_detections == 3
    assert r1.detection_ids[0] == 1 * 10000
    r2 = handle2.read_frame(2)
    assert r2.num_detections == 0
    assert r2.detection_ids.shape == (0,)


def test_detection_key_mismatch_returns_invalid(tmp_path):
    path = tmp_path / "test.obb.npz"
    handle = DetectionCacheHandle(path=path, key=_key("/a.pt"))
    handle.write_frame(0, result=_obb(0))
    handle.close()

    assert not DetectionCacheHandle(path=path, key=_key("/b.pt")).is_valid()


def test_detection_missing_file_is_invalid(tmp_path):
    assert not DetectionCacheHandle(path=tmp_path / "no.npz", key=_key()).is_valid()


def test_detection_schema_version_mismatch_is_invalid(tmp_path):
    """Per Correction 16: a cache written with an older schema_version must be invalid."""
    path = tmp_path / "test.obb.npz"
    legacy_key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION - 1,
        model_path="/m.pt",
        model_mtime=0.0,
        config_hash="abc",
    )
    handle = DetectionCacheHandle(path=path, key=legacy_key)
    handle.write_frame(0, result=_obb(0))
    handle.close()

    current_key = _key()
    assert not DetectionCacheHandle(path=path, key=current_key).is_valid()


# ---- HeadTailCacheHandle ----


def test_headtail_round_trip(tmp_path):
    path = tmp_path / "test.ht.npz"
    key = _key()
    handle = HeadTailCacheHandle(path=path, key=key)

    hints = np.array([0.0, 1.5], dtype=np.float32)
    confs = np.array([0.8, 0.9], dtype=np.float32)
    directed = np.array([1, 1], dtype=np.uint8)
    handle.write_frame(
        0,
        det_indices=np.array([0, 1]),
        heading_hints=hints,
        heading_confidences=confs,
        directed_mask=directed,
    )
    handle.close()

    handle2 = HeadTailCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    di, h, c, d = handle2.read_frame(0)
    assert di.tolist() == [0, 1]
    assert h.shape == (2,)
    assert h[1] == pytest.approx(1.5)
    assert d[0] == 1


def test_headtail_read_invalid_returns_none(tmp_path):
    assert (
        HeadTailCacheHandle(path=tmp_path / "no.npz", key=_key()).read_frame(0) is None
    )


# ---- CNNCacheHandle ----


def test_cnn_round_trip(tmp_path):
    path = tmp_path / "test.cnn.npz"
    key = _key()
    handle = CNNCacheHandle(path=path, key=key, label="id")

    handle.write_frame(0, predictions=_cnn_preds(n_dets=2))
    handle.close()

    handle2 = CNNCacheHandle(path=path, key=key, label="id")
    assert handle2.is_valid()
    loaded = handle2.read_frame(0)
    assert len(loaded) == 2
    det0 = loaded[0]
    assert det0.det_index == 0
    assert len(det0.factors) == 2
    assert det0.factors[0].factor_name == "color"
    assert det0.factors[0].raw_probabilities[0] == pytest.approx(0.7)
    assert det0.factors[1].factor_name == "size"
    assert det0.factors[1].raw_probabilities[1] == pytest.approx(0.6)


def test_cnn_empty_frame_round_trip(tmp_path):
    path = tmp_path / "test.cnn.npz"
    key = _key()
    handle = CNNCacheHandle(path=path, key=key, label="id")
    handle.write_frame(0, predictions=[])
    handle.close()

    handle2 = CNNCacheHandle(path=path, key=key, label="id")
    assert handle2.read_frame(0) == []


# ---- PoseCacheHandle ----


def test_pose_round_trip(tmp_path):
    path = tmp_path / "test.pose.npz"
    key = _key()
    handle = PoseCacheHandle(path=path, key=key)

    kp = np.random.rand(2, 17, 3).astype(np.float32)
    valid = np.array([True, False], dtype=bool)
    handle.write_frame(0, det_indices=np.array([0, 1]), keypoints=kp, valid_mask=valid)
    handle.close()

    handle2 = PoseCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    kp2, det_idx2, valid2 = handle2.read_frame(0)
    assert kp2.shape == (2, 17, 3)
    assert kp2[0, 0, 0] == pytest.approx(kp[0, 0, 0])
    assert bool(valid2[0]) is True
    assert bool(valid2[1]) is False


# ---- AprilTagCacheHandle ----


def test_apriltag_round_trip(tmp_path):
    path = tmp_path / "test.at.npz"
    key = _key()
    handle = AprilTagCacheHandle(path=path, key=key)

    result = AprilTagResult(
        tag_ids=np.array([3, 7], dtype=np.int32),
        det_indices=np.array([0, 1], dtype=np.int32),
        centers=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
    )
    handle.write_frame(0, result=result)
    handle.close()

    handle2 = AprilTagCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    r2 = handle2.read_frame(0)
    assert list(r2.tag_ids) == [3, 7]
    assert r2.centers[1, 0] == pytest.approx(30.0)


def test_apriltag_empty_frame(tmp_path):
    path = tmp_path / "test.at.npz"
    key = _key()
    handle = AprilTagCacheHandle(path=path, key=key)
    empty = AprilTagResult(
        tag_ids=np.array([], dtype=np.int32),
        det_indices=np.array([], dtype=np.int32),
        centers=np.zeros((0, 2), dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
    )
    handle.write_frame(0, result=empty)
    handle.close()

    r2 = AprilTagCacheHandle(path=path, key=key).read_frame(0)
    assert len(r2.tag_ids) == 0


# ---- independence ----


def test_detection_and_headtail_caches_are_independent(tmp_path):
    det_path = tmp_path / "test.obb.npz"
    ht_path = tmp_path / "test.ht.npz"

    det_handle = DetectionCacheHandle(path=det_path, key=_key("/obb.pt"))
    det_handle.write_frame(0, result=_obb(0))
    det_handle.close()

    ht_handle = HeadTailCacheHandle(path=ht_path, key=_key("/ht.pt"))
    assert not ht_handle.is_valid()
    assert DetectionCacheHandle(path=det_path, key=_key("/obb.pt")).is_valid()
