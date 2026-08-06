"""SessionWorker wraps TrackingSessionCore.run_post_tracking as a BaseWorker."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.core.tracking import session as session_module
from hydra_suite.core.tracking.session import SessionResult
from hydra_suite.trackerkit.gui.workers.session_worker import SessionWorker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _result() -> SessionResult:
    return SessionResult(
        success=True,
        final_csv_path="/tmp/out_final.csv",
        rich_export_path=None,
        media_paths=[],
        dataset_result=None,
        summary_lines=["final_csv=out_final.csv"],
        error=None,
    )


def test_execute_runs_service_and_emits_result(qapp, monkeypatch):
    captured = {"init": None, "ran": False, "callbacks": None}

    def _fake_init(self, *, video_path, config, params, paths, callbacks=None):
        captured["init"] = dict(video_path=video_path, paths=paths)
        captured["callbacks"] = callbacks
        self.callbacks = callbacks

    def _fake_run(self, forward, backward=None):
        captured["ran"] = True
        # exercise the bridged callbacks so the test proves they are wired
        self.callbacks.progress(50, "half")
        self.callbacks.warning("W", "msg")
        return _result()

    monkeypatch.setattr(session_module.TrackingSessionCore, "__init__", _fake_init)
    monkeypatch.setattr(
        session_module.TrackingSessionCore, "run_post_tracking", _fake_run
    )

    worker = SessionWorker(
        video_path="v.mp4",
        config={},
        params={"FPS": 30.0},
        paths={"raw_csv_path": "/tmp/v_tracking.csv"},
    )

    progresses: list[tuple[int, str]] = []
    warnings: list[tuple[str, str]] = []
    results: list[object] = []
    worker.progress_signal.connect(lambda v, m: progresses.append((v, m)))
    worker.warning_signal.connect(lambda t, m: warnings.append((t, m)))
    worker.finished_signal.connect(lambda r: results.append(r))

    worker.execute()  # call directly (synchronous) — do NOT start() the QThread

    assert captured["ran"] is True
    assert captured["init"]["paths"]["raw_csv_path"] == "/tmp/v_tracking.csv"
    assert (
        results and isinstance(results[0], SessionResult) and results[0].success is True
    )
    assert (50, "half") in progresses
    assert ("W", "msg") in warnings


def test_stop_sets_should_stop(qapp):
    worker = SessionWorker(video_path="v.mp4", config={}, params={}, paths={})
    assert worker._should_stop() is False
    worker.stop()
    assert worker._should_stop() is True
