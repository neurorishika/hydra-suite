from __future__ import annotations

from hydra_suite.trackerkit.cli_config import TrackerCliSession, TrackerCliVideoProbe
from hydra_suite.trackerkit.headless_tracking import (
    ensure_headless_qt_application,
    run_headless_tracking_session,
)


def test_ensure_headless_qt_application_creates_core_app(monkeypatch):
    class FakeCoreApplication:
        _instance = None

        def __init__(self, args):
            self.args = list(args)
            self.application_name = None
            type(self)._instance = self

        @classmethod
        def instance(cls):
            return cls._instance

        def setApplicationName(self, name):
            self.application_name = name

    monkeypatch.setattr(
        "hydra_suite.trackerkit.headless_tracking.QCoreApplication",
        FakeCoreApplication,
    )

    app = ensure_headless_qt_application()

    assert isinstance(app, FakeCoreApplication)
    assert app.args == []
    assert app.application_name == "TrackerKit CLI"


def test_run_headless_tracking_session_ensures_qt_app_before_running(
    monkeypatch, tmp_path
):
    calls = {"ensure": 0}

    def _fake_ensure():
        calls["ensure"] += 1
        return object()

    monkeypatch.setattr(
        "hydra_suite.trackerkit.headless_tracking.ensure_headless_qt_application",
        _fake_ensure,
    )

    class _CachePlan:
        inference_model_id = "bgsub_test"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(
        "hydra_suite.trackerkit.headless_tracking.plan_tracking_cache",
        lambda *args, **kwargs: _CachePlan(),
    )

    def _fake_forward_only(session, *, params, detection_cache_path):
        assert params["INFERENCE_MODEL_ID"] == "bgsub_test"
        assert detection_cache_path == str(tmp_path / "cache.npz")
        return {"success": True, "lines": [session.video_path]}

    monkeypatch.setattr(
        "hydra_suite.trackerkit.headless_tracking._run_forward_only",
        _fake_forward_only,
    )

    session = TrackerCliSession(
        video_path=str(tmp_path / "video.mp4"),
        config_path=None,
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=10, width=64, height=64
        ),
        config={},
        raw_csv_path=str(tmp_path / "video_tracking.csv"),
        final_csv_path=str(tmp_path / "video_tracking_final.csv"),
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

    result = run_headless_tracking_session(session)

    assert calls["ensure"] == 1
    assert result["success"] is True
