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
            direct=OBBDirectConfig(model_path="/m.pt"),
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
        def fake_run(
            frame_source, frame_range, progress_cb=None, range_total=0, should_stop=None
        ):
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
    def fake_run_obb(frames, models, obb_config, runtime, roi_mask=None):
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


# ---------------------------------------------------------------------------
# Regression: the batch/cached YOLO-OBB path must ROI-filter detections.
#
# Two independent call sites re-derive the filtered detection set from the
# raw detection cache, and BOTH were missing `roi_mask` entirely:
#   - Pipeline._process_obb_results() (core/inference/pipeline.py) -- used
#     during the live batch pass (run_batch_pass).
#   - InferenceRunner.load_frame() (core/inference/runner.py) -- used by
#     worker.py's cached-replay branch (both forward cache-reuse AND the
#     backward pass) to read tracked positions back out.
# Confirmed against a real user's tracking output (33.6% of positions
# outside their configured ROI) and via tests/test_arena_tiling_oracle.py's
# own module docstring, which documented this exact gap.
# ---------------------------------------------------------------------------


def _obb_two_detections(frame_idx: int = 0) -> OBBResult:
    """One detection inside a small top-left ROI, one far outside it."""
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array([[5.0, 5.0], [500.0, 500.0]], dtype=np.float32),
        angles=np.zeros(2, dtype=np.float32),
        sizes=np.full(2, 100.0, dtype=np.float32),
        shapes=np.ones((2, 2), dtype=np.float32),
        confidences=np.full(2, 0.9, dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, 2),
    )


def _small_topleft_roi_mask() -> np.ndarray:
    mask = np.zeros((640, 640), dtype=np.uint8)
    mask[:20, :20] = 255
    return mask


def test_load_frame_applies_roi_mask_filter(tmp_path):
    """InferenceRunner.load_frame() must filter cached detections by the
    roi_mask configured at construction. Must FAIL pre-fix (both detections
    survive) and PASS post-fix (only the in-ROI one survives)."""
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    obb = _obb_two_detections()
    roi_mask = _small_topleft_roi_mask()

    with (
        patch("hydra_suite.core.inference.runner._load_all_models"),
        patch("hydra_suite.core.inference.runner._open_caches") as mock_open,
    ):
        mock_caches = MagicMock()
        mock_caches.detection.read_frame.return_value = obb
        mock_caches.cnn = []
        mock_caches.headtail = None
        mock_caches.pose = None
        mock_caches.apriltag = None
        mock_caches.all_handles.return_value = [mock_caches.detection]
        mock_open.return_value = mock_caches
        # video_path=None -> _frame_space_roi_mask can't probe frame geometry
        # and returns roi_mask unchanged (no resample needed for this test).
        runner = InferenceRunner(
            cfg, cache_dir=tmp_path, video_path=None, roi_mask=roi_mask
        )
        result = runner.load_frame(0)

    assert result.obb.num_detections == 1, (
        "load_frame() must drop the out-of-ROI detection; got "
        f"{result.obb.num_detections} surviving"
    )
    np.testing.assert_allclose(result.obb.centroids[0], [5.0, 5.0])


def test_process_obb_results_applies_roi_mask_filter(tmp_path):
    """Pipeline._process_obb_results() (the batch-pass consumer stage) must
    filter detections by self.stages.roi_mask. Must FAIL pre-fix (both
    detections survive) and PASS post-fix (only the in-ROI one survives)."""
    from hydra_suite.core.inference.pipeline import BatchWindow
    from hydra_suite.core.inference.runner import InferenceRunner, _CacheSet

    cfg = _cfg()
    roi_mask = _small_topleft_roi_mask()

    def fake_run_obb(frames, models, obb_config, runtime, roi_mask=None):
        return [_obb_two_detections(frame_idx=i) for i in range(len(frames))]

    detection_cache = MagicMock()
    detection_cache.is_valid.return_value = False
    caches = _CacheSet(detection=detection_cache)

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch("hydra_suite.core.inference.pipeline.run_obb", side_effect=fake_run_obb),
    ):
        ml.return_value = MagicMock(
            obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
        )
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        pipeline = runner._build_pipeline(caches, roi_mask=roi_mask)
        window = BatchWindow(
            frames=[np.zeros((480, 640, 3), dtype=np.uint8)], frame_indices=[0]
        )
        results = pipeline._process_window(window)

    assert len(results) == 1
    assert results[0].obb.num_detections == 1, (
        "_process_obb_results() must drop the out-of-ROI detection; got "
        f"{results[0].obb.num_detections} surviving"
    )
    np.testing.assert_allclose(results[0].obb.centroids[0], [5.0, 5.0])


# ---------------------------------------------------------------------------
# detect_batch_raw: extracted raw (pre-filter_for_source) path underlying
# detect_batch. Must return the unfiltered per-frame OBBResult, while
# detect_batch (now a thin wrapper) keeps returning the filtered results.
# ---------------------------------------------------------------------------


def test_detect_batch_raw_returns_unfiltered_results(tmp_path):
    """detect_batch_raw() returns the raw, unfiltered OBBResult per frame;
    detect_batch() must still apply filter_for_source on top of it, so raw
    results have >= detections than the corresponding filtered results."""
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    roi_mask = _small_topleft_roi_mask()

    def fake_run_obb(frames, models, obb_config, runtime, roi_mask=None):
        return [_obb_two_detections(frame_idx=i) for i in range(len(frames))]

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch("hydra_suite.core.inference.runner.run_obb", side_effect=fake_run_obb),
    ):
        ml.return_value = MagicMock(
            obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
        )
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]

        raw_results = runner.detect_batch_raw(frames, frame_indices=[0, 1])
        filtered_results = runner.detect_batch(
            frames, frame_indices=[0, 1], roi_mask=roi_mask
        )

    assert len(raw_results) == 2
    assert all(isinstance(r, OBBResult) for r in raw_results)
    # Raw results are unfiltered: both the in-ROI and out-of-ROI detection
    # survive, while the ROI-filtered detect_batch() call drops one.
    assert all(r.num_detections == 2 for r in raw_results)
    for raw, filtered in zip(raw_results, filtered_results):
        assert raw.num_detections >= filtered.num_detections
    assert filtered_results[0].num_detections == 1


# ---------------------------------------------------------------------------
# Follow-up fix (task-review finding on the roi_mask fix above):
# InferenceRunner._frame_space_roi_mask() is called once PER FRAME by
# load_frame() during cached-replay (both forward cache-reuse and the whole
# backward pass). It previously re-probed the video's frame geometry via a
# fresh cv2.VideoCapture open on every single call, even though the mask and
# the video's geometry never change within one pass. It is now memoized per
# (video_path, id(self._roi_mask)).
# ---------------------------------------------------------------------------


def test_frame_space_roi_mask_memoized_across_calls(tmp_path):
    """_frame_space_roi_mask() must probe the video's frame geometry only
    ONCE across repeated calls with the same video_path and roi_mask, not
    once per call."""
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    roi_mask = _small_topleft_roi_mask()

    with (
        patch("hydra_suite.core.inference.runner._load_all_models"),
        patch(
            "hydra_suite.core.inference.runner._probe_frame_hw",
            return_value=(640, 640),
        ) as mock_probe,
    ):
        runner = InferenceRunner(
            cfg, cache_dir=tmp_path, video_path="fake.mp4", roi_mask=roi_mask
        )
        first = runner._frame_space_roi_mask("fake.mp4")
        second = runner._frame_space_roi_mask("fake.mp4")

    assert mock_probe.call_count == 1, (
        "_frame_space_roi_mask() must memoize its result; the underlying "
        f"frame-geometry probe was called {mock_probe.call_count} times "
        "across 2 calls, expected 1"
    )
    np.testing.assert_array_equal(first, second)
    assert first is second


def test_frame_space_roi_mask_invalidates_on_roi_mask_reassignment(tmp_path):
    """Reassigning self._roi_mask to a DIFFERENT mask object (as
    run_batch_pass's optional roi_mask override does) must invalidate the
    memoization cache rather than silently returning the stale resample."""
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = _cfg()
    roi_mask_a = _small_topleft_roi_mask()
    roi_mask_b = _small_topleft_roi_mask()  # different object, same content

    with (
        patch("hydra_suite.core.inference.runner._load_all_models"),
        patch(
            "hydra_suite.core.inference.runner._probe_frame_hw",
            return_value=(640, 640),
        ) as mock_probe,
    ):
        runner = InferenceRunner(
            cfg, cache_dir=tmp_path, video_path="fake.mp4", roi_mask=roi_mask_a
        )
        first = runner._frame_space_roi_mask("fake.mp4")
        assert mock_probe.call_count == 1

        runner._roi_mask = roi_mask_b
        second = runner._frame_space_roi_mask("fake.mp4")

    assert mock_probe.call_count == 2, (
        "reassigning self._roi_mask to a different object must force a "
        f"fresh resample; probe was called {mock_probe.call_count} times, "
        "expected 2"
    )
    assert first is not second
