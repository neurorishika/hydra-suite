"""``detection_cache_dir_covers_range`` (the optimizer cache validity probe in
``orchestrators/config.py``) must recognize only the modern
``InferenceRunner`` cache-directory layout -- the legacy single-file
``DetectionCache(mode="r")`` branch has been removed. A non-directory /
non-existent path must return False gracefully, never raise."""

from pathlib import Path

import numpy as np

from hydra_suite.core.inference.cache.keys import video_signature
from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runner import _open_caches
from hydra_suite.trackerkit.gui.orchestrators.config import (
    detection_cache_dir_covers_range,
)


def _make_obb_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((1, 2), dtype=np.float32),
        angles=np.zeros((1,), dtype=np.float32),
        sizes=np.array([10.0], dtype=np.float32),
        shapes=np.array([[10.0, 4.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=np.zeros((1, 4, 2), dtype=np.float32),
        detection_ids=np.array([1], dtype=np.int64),
    )


def _write_modern_detection_cache_dir(cache_dir: Path, params: dict, frames: range):
    """Populate the atomic cache-set layout written by ``InferenceRunner``."""
    cfg = build_inference_config_from_params(params)
    caches = _open_caches(
        cfg,
        cache_dir,
        video_signature(""),
        params.get("ROI_MASK", None),
        write_mode="fresh",
    )
    handle = caches.detection
    assert handle is not None
    for frame_idx in frames:
        handle.write_frame(frame_idx, result=_make_obb_result(frame_idx))
    caches.close()


def test_modern_cache_dir_covering_range_is_valid(tmp_path):
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))

    active_before = (cache_dir / "cache_set.json").read_bytes()
    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is True
    )
    assert (cache_dir / "cache_set.json").read_bytes() == active_before


def test_modern_cache_file_path_covering_range_is_valid(tmp_path):
    """``current_detection_cache_path`` (and optimizer-reuse candidates) may be
    the ``detection.npz`` FILE itself, not just its containing directory --
    the probe must normalize to the containing dir before opening caches."""
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))
    cache_file = cache_dir / "detection.npz"

    assert (
        detection_cache_dir_covers_range(
            str(cache_file), "", params, start_frame=0, end_frame=4
        )
        is True
    )


def test_modern_cache_dir_missing_frames_is_not_valid(tmp_path):
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 3))

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=9
        )
        is False
    )


def test_modern_cache_with_corrupt_set_manifest_is_not_valid(tmp_path):
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))
    (cache_dir / "cache_set.json").write_text("{}", encoding="utf-8")

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is False
    )


def test_nonexistent_path_returns_false_without_raising(tmp_path):
    missing = tmp_path / "does_not_exist.npz"

    assert (
        detection_cache_dir_covers_range(
            str(missing), "", {}, start_frame=0, end_frame=4
        )
        is False
    )


def test_empty_path_returns_false(tmp_path):
    assert (
        detection_cache_dir_covers_range("", "", {}, start_frame=0, end_frame=4)
        is False
    )
