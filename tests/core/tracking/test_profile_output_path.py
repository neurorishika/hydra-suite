"""Tests for tracking-profile JSON path selection (Task 10).

The profile path must land under ``<stem>_logs/`` when no output video is
configured, instead of polluting the (retired) cache folder.
"""

from pathlib import Path

from hydra_suite.core.tracking.worker import TrackingEngineCore
from hydra_suite.utils.video_artifacts import build_video_log_dir


class _FakeWorker:
    """Lightweight stand-in exposing only the attributes the helper reads."""

    backward_mode = False

    def __init__(self, video_path, video_output_path=None):
        self.video_path = video_path
        self.video_output_path = video_output_path


def test_profile_path_no_output_video_forward(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    worker = _FakeWorker(str(video_path))
    worker.backward_mode = False

    result = TrackingEngineCore._resolve_profile_path(worker, "forward")

    expected = build_video_log_dir(str(video_path)) / "tracking_profile_forward.json"
    assert Path(result) == expected


def test_profile_path_no_output_video_backward(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    worker = _FakeWorker(str(video_path))
    worker.backward_mode = True

    result = TrackingEngineCore._resolve_profile_path(worker, "backward")

    expected = build_video_log_dir(str(video_path)) / "tracking_profile_backward.json"
    assert Path(result) == expected


def test_profile_path_with_output_video_unchanged(tmp_path):
    video_path = tmp_path / "clip.mp4"
    output_path = tmp_path / "out.mp4"
    video_path.write_bytes(b"")
    worker = _FakeWorker(str(video_path), video_output_path=str(output_path))

    result = TrackingEngineCore._resolve_profile_path(worker, "forward")

    expected_base = Path(str(output_path)).with_suffix("")
    expected = Path(f"{expected_base}_forward.profile.json")
    assert Path(result) == expected
