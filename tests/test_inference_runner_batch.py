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


def test_caches_invalidated_when_video_file_changes(tmp_path):
    """Regression: a detection cache must not be reused after the source video
    changes under the same name (e.g. a clip regenerated with more frames).
    Without the video-signature binding this returned a stale, truncated cache.
    """
    from hydra_suite.core.inference.runner import InferenceRunner, _open_caches

    cfg = _cfg()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 100)

    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner = InferenceRunner(cfg, cache_dir=tmp_path, video_path=str(video))

    # Write a detection cache bound to this exact video's signature.
    caches = _open_caches(cfg, tmp_path, runner._video_sig)
    caches.detection.write_frame(0, result=_make_obb(2, 0))
    caches.detection.close()
    assert runner.caches_all_valid() is True

    # Regenerate the video under the same name with different content/size.
    video.write_bytes(b"y" * 5000)
    with patch("hydra_suite.core.inference.runner._load_all_models"):
        runner2 = InferenceRunner(cfg, cache_dir=tmp_path, video_path=str(video))
    assert runner2.caches_all_valid() is False


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

        # run_batch_pass now drives the whole pass via Pipeline.run; stub it so
        # this test verifies the runner's read-loop wiring (frame source drained,
        # range_total, final progress callback) without real OBB/stage work.
        def fake_run(frame_source, frame_range, progress_cb=None, range_total=0):
            processed = sum(1 for _ in frame_source)
            if progress_cb:
                progress_cb(processed, range_total)

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch(
                "hydra_suite.core.inference.pipeline.Pipeline.run",
                side_effect=fake_run,
                autospec=False,
            ),
        ):
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


def test_run_batch_iterates_frames_and_writes_caches(tmp_path):
    """Integration test: _run_batch runs per-frame, writes detection cache, and
    threads detection_ids through to downstream cache writes."""
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.runner import InferenceRunner, _CacheSet

    cfg = _cfg()  # InferenceConfig with no headtail/cnn/pose/apriltag

    # Stub run_obb to return predictable OBBResults for each frame
    def fake_run_obb(frames, models, obb_config, runtime):
        return [_make_obb(n=2, frame_idx=i) for i in range(len(frames))]

    # Mock cache handles to record writes
    detection_cache = MagicMock()
    detection_cache.is_valid.return_value = False
    caches = _CacheSet(detection=detection_cache)

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        # OBB now runs inside the depth=1 Pipeline (_process_window), so patch the
        # symbol in the pipeline module's namespace, not the runner's.
        patch("hydra_suite.core.inference.pipeline.run_obb", side_effect=fake_run_obb),
    ):
        ml.return_value = MagicMock(
            obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
        )
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        # Exercise _run_batch directly (skip cv2.VideoCapture)
        fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
        runner._run_batch(fake_frames, [0, 1, 2], caches)

    # Detection cache write_frame called once per frame
    assert detection_cache.write_frame.call_count == 3
    # Each call passes an OBBResult with detection_ids
    for call_idx, call in enumerate(detection_cache.write_frame.call_args_list):
        kwargs = call[1] or {}
        result = kwargs.get("result")
        assert isinstance(result, OBBResult)
        assert result.frame_idx == call_idx
        assert result.detection_ids.shape == (2,)
        # IDs follow frame_idx * STRIDE + slot
        assert result.detection_ids[0] == call_idx * 10000
