from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.result import OBBResult


def _cfg() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"),
        ),
        detection_batch_size=2,
    )


def _make_obb(n: int = 3, frame_idx: int = 0) -> OBBResult:
    rng = np.random.default_rng(0)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=rng.uniform(0, 640, (n, 2)).astype(np.float32),
        angles=rng.uniform(0, np.pi, n).astype(np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=rng.uniform(0, 640, (n, 4, 2)).astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def test_run_batch_pass_raises_without_cache_dir():
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=None)
    with pytest.raises(RuntimeError, match="cache_dir"):
        runner.run_batch_pass(Path("video.mp4"))


def test_run_batch_pass_raises_on_unreadable_video(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    with (
        patch("hydra_suite.core.inference.runner._load_all_models"),
        patch("hydra_suite.core.inference.runner._open_caches") as mock_open,
    ):
        mock_caches = MagicMock()
        mock_caches.all_handles.return_value = []
        mock_open.return_value = mock_caches
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
    with pytest.raises(IOError, match="Cannot open"):
        runner.run_batch_pass(tmp_path / "nonexistent.mp4")


def test_run_batch_pass_calls_progress_callback(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.read.side_effect = [
        (True, fake_frame),
        (True, fake_frame),
        (True, fake_frame),
        (True, fake_frame),
        (True, fake_frame),
        (False, None),
    ]
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 5.0

    progress_calls: list[tuple] = []

    with (
        patch("hydra_suite.core.inference.runner._load_all_models"),
        patch("hydra_suite.core.inference.runner._open_caches") as mock_open,
    ):
        mock_caches = MagicMock()
        mock_caches.all_handles.return_value = []
        mock_open.return_value = mock_caches
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        runner._run_batch = MagicMock()
        with patch("cv2.VideoCapture", return_value=mock_cap):
            runner.run_batch_pass(
                tmp_path / "video.mp4",
                progress_cb=lambda done, total: progress_calls.append((done, total)),
            )
    assert len(progress_calls) > 0
    assert progress_calls[-1][1] == 5


def test_load_frame_raises_without_cache_dir():
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=None)
    with pytest.raises(RuntimeError, match="cache_dir"):
        runner.load_frame(0)


def test_load_frame_raises_on_missing_frame(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    with (
        patch("hydra_suite.core.inference.runner._load_all_models"),
        patch("hydra_suite.core.inference.runner._open_caches") as mock_open,
    ):
        mock_caches = MagicMock()
        mock_caches.detection.read_frame.return_value = None
        mock_caches.all_handles.return_value = [mock_caches.detection]
        mock_open.return_value = mock_caches
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
    with pytest.raises(KeyError, match="0"):
        runner.load_frame(0)


def test_load_headtail_aligns_by_det_indices():
    from hydra_suite.core.inference.runner import _load_headtail_for_indices

    cached_det_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    heading_hints = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    heading_confs = np.array([0.8, 0.9, 0.7, 0.95], dtype=np.float32)
    directed = np.array([1, 0, 1, 0], dtype=np.uint8)

    mock_cache = MagicMock()
    mock_cache.read_frame.return_value = (
        cached_det_indices,
        heading_hints,
        heading_confs,
        directed,
    )

    filtered_obb = _make_obb(2, frame_idx=7)
    det_indices = np.array([1, 3], dtype=np.int32)

    result = _load_headtail_for_indices(mock_cache, 7, det_indices, filtered_obb)
    assert result is not None
    np.testing.assert_allclose(result.heading_hints, [2.0, 4.0])
    np.testing.assert_allclose(result.heading_confidences, [0.9, 0.95])
    np.testing.assert_array_equal(result.directed_mask, [0, 0])
