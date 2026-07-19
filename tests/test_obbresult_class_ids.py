"""Tests for OBBResult.class_ids: defaults, cache round-trip, backward
compatibility with pre-existing caches, and alignment through filtering.

See docs/superpowers/specs/... (task F): class_ids must travel EVERYWHERE
detection_ids already travels, using the exact same subsetting/copying/
serialization, so it stays aligned with the other per-detection arrays.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle, _npz_save
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import _select


def _key(path="/m.pt") -> CacheKey:
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=path,
        model_mtime=0.0,
        config_hash="abc",
    )


def _obb(frame_idx: int, n: int = 2, class_ids: np.ndarray | None = None) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.ones((n, 2), dtype=np.float32) * frame_idx,
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
        class_ids=class_ids,
    )


def test_class_ids_or_zeros_defaults_to_zeros_when_none():
    result = _obb(0, n=3, class_ids=None)
    assert result.class_ids is None
    zeros = result.class_ids_or_zeros
    assert zeros.dtype == np.int64
    assert np.array_equal(zeros, np.zeros(3, dtype=np.int64))


def test_class_ids_or_zeros_returns_actual_array_when_set():
    result = _obb(0, n=2, class_ids=np.array([3, 1], dtype=np.int64))
    assert np.array_equal(result.class_ids_or_zeros, np.array([3, 1]))


def test_detection_cache_round_trips_class_ids(tmp_path):
    path = tmp_path / "test.obb.npz"
    key = _key()
    handle = DetectionCacheHandle(path=path, key=key)

    handle.write_frame(
        0, result=_obb(0, n=2, class_ids=np.array([3, 1], dtype=np.int64))
    )
    handle.close()

    handle2 = DetectionCacheHandle(path=path, key=key)
    assert handle2.is_valid()
    r0 = handle2.read_frame(0)
    assert r0.class_ids is not None
    assert np.array_equal(r0.class_ids, np.array([3, 1]))


def test_detection_cache_reads_old_format_without_class_ids_key(tmp_path):
    """Simulate a cache written before class_ids existed: no such key in the
    npz. read_frame must not crash and must fall back to None/zeros."""
    path = tmp_path / "old.obb.npz"
    key = _key()

    n = 2
    frame_idx = 0
    _npz_save(
        path,
        key,
        frame_count=np.array([1]),
        frame_indices=np.array([frame_idx] * n, dtype=np.int32),
        written_frames=np.array([frame_idx], dtype=np.int32),
        centroids=np.ones((n, 2), dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
        # deliberately no "class_ids" key
    )

    handle = DetectionCacheHandle(path=path, key=key)
    assert handle.is_valid()
    result = handle.read_frame(frame_idx)
    assert result is not None
    assert result.class_ids is None
    assert np.array_equal(result.class_ids_or_zeros, np.zeros(n, dtype=np.int64))


def test_filtering_select_subsets_class_ids_aligned_with_other_arrays():
    raw = _obb(0, n=3, class_ids=np.array([5, 6, 7], dtype=np.int64))
    indices = np.array([0, 2])
    filtered = _select(raw, indices)
    assert np.array_equal(filtered.class_ids, np.array([5, 7]))
    # Alignment check: class_ids track detection_ids through the same subset.
    assert np.array_equal(filtered.detection_ids, raw.detection_ids[indices])
