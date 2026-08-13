import numpy as np

from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.refinekit.gui.overlay_utils import (
    discover_detection_cache,
    load_frame_detections,
)


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


def test_discover_finds_modern_inference_cache(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    _write_one_frame(cache_dir / "detection.npz")
    assert discover_detection_cache(str(video)) == cache_dir / "detection.npz"


def test_discover_returns_none_when_absent(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    assert discover_detection_cache(str(video)) is None


def test_load_frame_detections_from_modern_cache(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    _write_one_frame(cache_dir / "detection.npz")
    fd = load_frame_detections(str(video))
    assert fd is not None
    got = fd.get(0)
    assert got is not None  # (meas, semi_axes, obb) tuple
    meas, semi_axes, obb = got
    assert len(meas) == 1
    assert meas.shape == (1, 3)
    assert semi_axes.shape == (1, 2)
