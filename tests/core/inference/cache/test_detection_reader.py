import numpy as np

from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.reader import open_detection_cache_reader
from hydra_suite.core.inference.cache.store import DetectionCacheHandle
from hydra_suite.core.inference.result import OBBResult


def _write_one_frame(path):
    key = CacheKey(schema_version=3, model_path="m", model_mtime=1.0, config_hash="h")
    h = DetectionCacheHandle(path=path, key=key)
    h.write_frame(
        0,
        result=OBBResult(
            frame_idx=0,
            centroids=np.array([[1.0, 2.0]], np.float32),
            angles=np.array([0.5], np.float32),
            sizes=np.array([10.0], np.float32),
            shapes=np.array([[100.0, 2.0]], np.float32),
            confidences=np.array([0.9], np.float32),
            corners=np.zeros((1, 4, 2), np.float32),
            detection_ids=np.array([7], np.int64),
        ),
    )
    h.close()


def test_reader_round_trips_a_frame(tmp_path):
    p = tmp_path / "detection.npz"
    _write_one_frame(p)
    reader = open_detection_cache_reader(p)
    res = reader.read_frame(0)
    assert res is not None
    assert res.detection_ids.tolist() == [7]
    assert reader.read_frame(999) is None  # unwritten frame -> None


def test_reader_missing_file_reads_none(tmp_path):
    reader = open_detection_cache_reader(tmp_path / "detection.npz")
    assert reader.read_frame(0) is None
