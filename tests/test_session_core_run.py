import pandas as pd
import pytest

from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.session import (
    SessionCallbacks,
    SessionResult,
    TrackingSessionCore,
)


def _write_raw_csv(path):
    pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0],
            "X": [10.0, 11.0, 12.0],
            "Y": [5.0, 5.0, 5.0],
            "Theta": [0.0, 0.0, 0.0],
            "FrameID": [0, 1, 2],
            "State": ["tracked"] * 3,
        }
    ).to_csv(path, index=False)


def _config():
    return {
        "enable_postprocessing": False,
        "interpolation_method": "none",
        "interpolation_max_gap_seconds": 1.0,
        "heading_flip_max_burst": 5,
        "enable_backward_tracking": False,
        "individual_interpolate_occlusions": False,
    }


def test_forward_only_writes_final_csv(tmp_path):
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))
    core = TrackingSessionCore(
        video_path=str(tmp_path / "clip.mp4"),
        config=_config(),
        params={"FPS": 30.0, "RESIZE_FACTOR": 1.0, "MIN_TRAJECTORY_LENGTH": 1},
        paths={
            "raw_csv_path": str(raw),
            "detection_cache_path": str(tmp_path / "d.npz"),
        },
    )
    result = core.run_post_tracking(pd.read_csv(str(raw)))
    assert isinstance(result, SessionResult)
    assert result.success is True
    assert result.final_csv_path is not None
    assert pd.read_csv(result.final_csv_path).shape[0] == 3
    assert result.media_paths == []
    assert result.dataset_result is None
    assert isinstance(result.summary_lines, list)
    assert any("Trajectories:" in ln for ln in result.summary_lines)


def test_should_stop_between_stages_yields_unsuccessful_result(tmp_path):
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))
    core = TrackingSessionCore(
        video_path=str(tmp_path / "clip.mp4"),
        config=_config(),
        params={"FPS": 30.0, "RESIZE_FACTOR": 1.0, "MIN_TRAJECTORY_LENGTH": 1},
        paths={
            "raw_csv_path": str(raw),
            "detection_cache_path": str(tmp_path / "d.npz"),
        },
        callbacks=SessionCallbacks(should_stop=lambda: True),
    )
    result = core.run_post_tracking(pd.read_csv(str(raw)))
    assert result.success is False


def test_merge_raises_when_video_unreadable(tmp_path, monkeypatch):
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))
    core = TrackingSessionCore(
        video_path=str(tmp_path / "does_not_exist.mp4"),
        config=_config(),
        params={"FPS": 30.0, "RESIZE_FACTOR": 1.0, "MIN_TRAJECTORY_LENGTH": 1},
        paths={
            "raw_csv_path": str(raw),
            "detection_cache_path": str(tmp_path / "d.npz"),
        },
    )

    class _StubCap:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.cv2.VideoCapture",
        lambda *_a, **_k: _StubCap(),
    )

    df = pd.read_csv(str(raw))
    with pytest.raises(TrackingSessionError):
        core._merge(df, df)
