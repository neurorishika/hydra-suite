"""A read-only detection-cache handle must never write to disk.

`open_detection_cache_reader` hands out a handle with a PLACEHOLDER key
(`v0||0.000000|`) because its consumers do not care which run produced the
geometry. But `DetectionCacheHandle.close()` flushes unconditionally: with an
empty `_buffer` it overwrites the target with a zero-detection cache stamped
with the handle's own key. Closing a reader therefore DESTROYS the real
`detection.npz` -- every detection is gone and the key can no longer match any
run config, so `caches_all_valid()` fails forever after and the pipeline
re-runs full inference on every subsequent run despite "reuse cache" being on.

`core/post/dataset_export.py` documents this hazard and defends against it by
never closing its handle, but `core/post/interpolated_crops._cleanup_backends`
does close it -- and that postpass runs on essentially every identity/head-tail
run. Making `close()` a no-op for read-only handles fixes it at the source
rather than relying on every caller remembering the rule.
"""

import numpy as np
import pytest

from hydra_suite.core.inference.cache import (
    open_cnn_cache_reader,
    open_detection_cache_reader,
)
from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle, _npz_save


def _real_key():
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path="/models/obb.pt",
        model_mtime=1234.5,
        config_hash="abc123",
    )


def _write_populated_cache(path):
    """A cache as a real forward pass leaves it: 3 frames, 3 detections."""
    _npz_save(
        path,
        _real_key(),
        frame_count=np.array([3]),
        frame_indices=np.array([0, 1, 2], np.int32),
        written_frames=np.array([0, 1, 2], np.int32),
        centroids=np.array([[1, 2], [3, 4], [5, 6]], np.float32),
        angles=np.zeros(3, np.float32),
        sizes=np.ones(3, np.float32),
        shapes=np.ones((3, 2), np.float32),
        confidences=np.full(3, 0.9, np.float32),
        corners=np.zeros((3, 4, 2), np.float32),
        detection_ids=np.array([10, 11, 12], np.int64),
        class_ids=np.zeros(3, np.int64),
    )


def test_closing_a_reader_does_not_destroy_the_cache(tmp_path):
    path = tmp_path / "detection.npz"
    _write_populated_cache(path)

    reader = open_detection_cache_reader(path)
    assert reader.read_frame(1) is not None
    reader.close()

    after = np.load(path)
    assert str(after["cache_key"][0]) == _real_key().as_string(), (
        "closing a read-only handle stamped the placeholder key over the real "
        "one -- caches_all_valid() can never match again"
    )
    assert after["centroids"].shape == (3, 2), "detections were wiped"
    assert list(after["written_frames"]) == [0, 1, 2], "frame coverage was wiped"


def test_closed_reader_cache_stays_reusable(tmp_path):
    """End state that matters: the cache still validates against its run key."""
    path = tmp_path / "detection.npz"
    _write_populated_cache(path)

    open_detection_cache_reader(path).close()

    writer_view = DetectionCacheHandle(path=path, key=_real_key())
    assert writer_view.is_valid(), (
        "after a reader was closed the cache no longer matches its own run "
        "config -> the next run re-runs full inference"
    )
    assert writer_view.covers_frame_range(0, 2)


def test_real_writer_still_flushes_an_empty_cache(tmp_path):
    """A genuine writer that produced nothing must still write its cache."""
    path = tmp_path / "detection.npz"
    handle = DetectionCacheHandle(path=path, key=_real_key())
    handle.close()

    assert path.exists()
    data = np.load(path)
    assert str(data["cache_key"][0]) == _real_key().as_string()
    assert int(data["chunked_format_version"][0]) == 2
    reader = DetectionCacheHandle(path=path, key=_real_key())
    assert reader.is_valid()
    assert reader.written_frames() == set()


def test_cnn_reader_is_read_only(tmp_path):
    reader = open_cnn_cache_reader(tmp_path / "cnn_identity.npz", "identity")
    with pytest.raises(RuntimeError, match="read-only"):
        reader.write_frame(0, predictions=[])
    reader.close()


@pytest.mark.parametrize("frames", [(0, 2), (0, 5)])
def test_reader_survives_the_interpolated_crops_cleanup_path(tmp_path, frames):
    """`_cleanup_backends` closes the handle via `_close_resource`."""
    from hydra_suite.core.post.interpolated_crops import _close_resource

    path = tmp_path / "detection.npz"
    _write_populated_cache(path)

    _close_resource(open_detection_cache_reader(path))

    data = np.load(path)
    assert str(data["cache_key"][0]) == _real_key().as_string()
    assert data["centroids"].shape == (3, 2)
