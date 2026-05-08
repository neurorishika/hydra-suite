"""Integration tests for worker.py USE_NEW_INFERENCE_PIPELINE flag (Task 17).

Tests verify:
- Flag exists and is True by default
- _run_with_new_pipeline calls run_batch_pass when caches are invalid
- _run_with_new_pipeline skips run_batch_pass when caches are valid
- _run_realtime_with_new_pipeline calls run_realtime per frame
- Backward mode refuses to run inference (Correction 24)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_frame_result():
    from hydra_suite.core.inference.result import FrameResult, OBBResult

    obb = OBBResult(
        frame_idx=0,
        centroids=np.zeros((2, 2), dtype=np.float32),
        angles=np.zeros(2, dtype=np.float32),
        sizes=np.ones(2, dtype=np.float32) * 100,
        shapes=np.ones((2, 2), dtype=np.float32),
        confidences=np.array([0.9, 0.8], dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, 2),
    )
    return FrameResult(
        frame_idx=0,
        obb=obb,
        filtered_indices=[0, 1],
        headtail=None,
        cnn=[],
        pose=None,
        apriltag=None,
        resolved_headings=np.zeros(2, dtype=np.float32),
    )


def test_worker_uses_new_pipeline_flag():
    """Verify USE_NEW_INFERENCE_PIPELINE constant exists and is True."""
    from hydra_suite.core.tracking import worker

    assert hasattr(worker, "USE_NEW_INFERENCE_PIPELINE")
    assert worker.USE_NEW_INFERENCE_PIPELINE is True


def test_run_with_new_pipeline_calls_batch_pass_when_caches_invalid(tmp_path):
    """When caches are invalid, run_batch_pass is called before the tracking loop."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    mock_runner = MagicMock()
    mock_runner.caches_all_valid.return_value = False
    mock_runner.load_frame.return_value = _make_frame_result()

    with (
        patch(
            "hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner
        ),
        patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls,
    ):
        mock_cfg_cls.from_json.return_value = MagicMock(realtime=False, cnn_phases=[])
        worker_obj = TrackingWorker.__new__(TrackingWorker)
        worker_obj._identity_builders = []
        worker_obj.backward_mode = False
        worker_obj.progress_signal = MagicMock()
        worker_obj._run_with_new_pipeline(
            video_path=tmp_path / "video.mp4",
            config_path=str(tmp_path / "cfg.json"),
            cache_dir=tmp_path,
            total_frames=2,
        )

    mock_runner.run_batch_pass.assert_called_once()


def test_run_with_new_pipeline_skips_batch_pass_when_caches_valid(tmp_path):
    """When caches are valid, run_batch_pass is NOT called."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    mock_runner = MagicMock()
    mock_runner.caches_all_valid.return_value = True
    mock_runner.load_frame.return_value = _make_frame_result()

    with (
        patch(
            "hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner
        ),
        patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls,
    ):
        mock_cfg_cls.from_json.return_value = MagicMock(realtime=False, cnn_phases=[])
        worker_obj = TrackingWorker.__new__(TrackingWorker)
        worker_obj._identity_builders = []
        worker_obj.backward_mode = False
        worker_obj.progress_signal = MagicMock()
        worker_obj._run_with_new_pipeline(
            video_path=tmp_path / "video.mp4",
            config_path=str(tmp_path / "cfg.json"),
            cache_dir=tmp_path,
            total_frames=2,
        )

    mock_runner.run_batch_pass.assert_not_called()


def test_run_realtime_calls_run_realtime_per_frame(tmp_path):
    """RT path calls run_realtime() once per frame."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
    mock_runner = MagicMock()
    mock_runner.run_realtime.return_value = _make_frame_result()

    worker_obj = TrackingWorker.__new__(TrackingWorker)
    worker_obj._identity_builders = []
    worker_obj._run_realtime_with_new_pipeline(frames, mock_runner)

    assert mock_runner.run_realtime.call_count == 3


def test_backward_mode_refuses_to_run_inference(tmp_path):
    """Backward pass must NOT call run_batch_pass — caches must already exist."""
    from hydra_suite.core.tracking.worker import TrackingWorker

    mock_runner = MagicMock()
    mock_runner.caches_all_valid.return_value = False  # caches missing

    with (
        patch(
            "hydra_suite.core.tracking.worker.InferenceRunner", return_value=mock_runner
        ),
        patch("hydra_suite.core.tracking.worker.InferenceConfig") as mock_cfg_cls,
    ):
        mock_cfg_cls.from_json.return_value = MagicMock(realtime=False, cnn_phases=[])
        worker_obj = TrackingWorker.__new__(TrackingWorker)
        worker_obj._identity_builders = []
        worker_obj.backward_mode = True
        worker_obj.progress_signal = MagicMock()
        with pytest.raises(RuntimeError, match="forward-pass caches"):
            worker_obj._run_with_new_pipeline(
                video_path=tmp_path / "video.mp4",
                config_path=str(tmp_path / "cfg.json"),
                cache_dir=tmp_path,
                total_frames=2,
            )

    mock_runner.run_batch_pass.assert_not_called()
