"""``_get_detection_size`` must read the modern ``DetectionCacheHandle``
(via ``open_detection_cache_reader``) instead of the legacy
``DetectionCache``, while preserving the exact return contract: (width,
height) for a present detection id, ``(None, None)`` on a miss (the caller
-- ``_process_occluded_run`` -- applies the ``REFERENCE_BODY_SIZE``
fallback on that ``None``)."""

import numpy as np
import pytest

from hydra_suite.core.inference.cache import open_detection_cache_reader
from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.post.interpolated_crops import _get_detection_size


def _write_single_frame_cache(path, detection_id=100000):
    key = CacheKey(schema_version=0, model_path="m", model_mtime=1.0, config_hash="h")
    writer = DetectionCacheHandle(path=path, key=key)
    # A 10x4 axis-aligned OBB: corners[1]-corners[0] has length 10 (width),
    # corners[2]-corners[1] has length 4 (height) -- matches
    # ``_size_from_obb_corners``'s edge-vector convention.
    corners = np.array(
        [[[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]], dtype=np.float32
    )
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[5.0, 2.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([40.0], dtype=np.float32),
        shapes=np.array([[40.0, 2.5]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=corners,
        detection_ids=np.array([detection_id], dtype=np.int64),
    )
    writer.write_frame(0, result=obb)
    writer.close()


def test_get_detection_size_returns_known_obb_dims(tmp_path):
    cache_path = tmp_path / "detection.npz"
    _write_single_frame_cache(cache_path, detection_id=100000)

    reader = open_detection_cache_reader(cache_path)

    w, h = _get_detection_size(reader, frame_id=0, detection_id=100000)

    assert w == pytest.approx(10.0)
    assert h == pytest.approx(4.0)


def test_get_detection_size_missing_id_returns_none_none(tmp_path):
    cache_path = tmp_path / "detection.npz"
    _write_single_frame_cache(cache_path, detection_id=100000)

    reader = open_detection_cache_reader(cache_path)

    # A detection id absent from the cached frame -- caller
    # (_process_occluded_run) is responsible for the REFERENCE_BODY_SIZE
    # fallback when this returns (None, None).
    w, h = _get_detection_size(reader, frame_id=0, detection_id=999999)

    assert w is None
    assert h is None


def test_get_detection_size_none_cache_returns_none_none():
    w, h = _get_detection_size(None, frame_id=0, detection_id=100000)
    assert w is None
    assert h is None


def test_get_detection_size_none_detection_id_returns_none_none(tmp_path):
    cache_path = tmp_path / "detection.npz"
    _write_single_frame_cache(cache_path, detection_id=100000)
    reader = open_detection_cache_reader(cache_path)

    w, h = _get_detection_size(reader, frame_id=0, detection_id=None)
    assert w is None
    assert h is None
