"""Slice 4 CLI cutover: every session runs the direct (Qt-free) path."""

from __future__ import annotations

from hydra_suite.trackerkit.cli_config import TrackerCliSession, TrackerCliVideoProbe


def _make_session(**overrides) -> TrackerCliSession:
    base = dict(
        video_path="video.mp4",
        config_path=None,
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=10, width=64, height=64
        ),
        config={},
        raw_csv_path="video_tracking.csv",
        final_csv_path="video_tracking_forward_processed.csv",
        params={"FPS": 30.0},
        save_confidence_metrics=False,
        use_cached_detections=False,
        enable_backward_tracking=False,
        enable_postprocessing=True,
        interpolation_method="None",
        interpolation_max_gap_seconds=0.0,
        heading_flip_max_burst=5,
        identity_method="none_disabled",
        enable_pose_extractor=False,
    )
    base.update(overrides)
    return TrackerCliSession(**base)


def test_supports_direct_run_true_for_pose_sessions():
    session = _make_session(enable_pose_extractor=True)
    assert session.supports_direct_run() is True


def test_supports_direct_run_true_for_identity_sessions():
    session = _make_session(identity_method="apriltags")
    assert session.supports_direct_run() is True


def test_supports_direct_run_true_for_plain_session():
    assert _make_session().supports_direct_run() is True
