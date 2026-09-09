"""A zero-detection frame must not poison the pose cache's keypoint shape.

Producers hand back ``(0, 0, 3)`` for a frame with no detections. Before this
guard that latched ``_keypoint_shape = (0, 3)``, and every subsequent real
frame raised ``pose keypoint dimensions must remain constant``, killing the
whole InferenceRunner batch pass.

Surfaced 2026-09-06 by retuning the ant_cnn_identity fixture's
``reference_body_size`` (76.81 -> 49.83), which tightened MIN_OBJECT_SIZE
enough to make zero-detection frames reachable on that clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.store import PoseCacheHandle

KPTS = 13


@pytest.fixture
def key():
    return CacheKey(CACHE_SCHEMA_VERSION, "/model.pt", "config")


def _write(handle, frame_idx, n_det, n_kpt):
    handle.write_frame(
        frame_idx,
        det_indices=np.arange(n_det, dtype=np.int32),
        keypoints=np.zeros((n_det, n_kpt, 3), dtype=np.float32),
        valid_mask=np.ones(n_det, dtype=np.uint8),
    )


def test_empty_frame_first_then_real_frames(tmp_path, key):
    """The order that used to crash: an empty frame before any real one."""
    h = PoseCacheHandle(tmp_path / "pose.npz", key, chunk_size=8)
    _write(h, 0, 0, 0)  # zero detections -> (0, 0, 3)
    _write(h, 1, 3, KPTS)  # would raise before the fix
    _write(h, 2, 2, KPTS)
    h.close()

    r = PoseCacheHandle(tmp_path / "pose.npz", key)
    assert r.read_frame(1)[0].shape == (3, KPTS, 3)
    assert r.read_frame(2)[0].shape == (2, KPTS, 3)


def test_empty_frame_between_real_frames(tmp_path, key):
    h = PoseCacheHandle(tmp_path / "pose.npz", key, chunk_size=8)
    _write(h, 0, 2, KPTS)
    _write(h, 1, 0, 0)
    _write(h, 2, 4, KPTS)
    h.close()

    r = PoseCacheHandle(tmp_path / "pose.npz", key)
    assert r.read_frame(0)[0].shape == (2, KPTS, 3)
    assert r.read_frame(2)[0].shape == (4, KPTS, 3)


def test_a_real_shape_change_is_still_rejected(tmp_path, key):
    """The guard must not weaken the invariant it protects."""
    h = PoseCacheHandle(tmp_path / "pose.npz", key, chunk_size=8)
    _write(h, 0, 2, KPTS)
    with pytest.raises(ValueError, match="keypoint dimensions must remain constant"):
        _write(h, 1, 2, KPTS + 1)
