"""The GUI post-tracking flow builds a SessionWorker over run_post_tracking."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.core.tracking.session import SessionResult
from hydra_suite.trackerkit.gui.orchestrators import tracking as tracking_module
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _orchestrator(tmp_path):
    raw = str(tmp_path / "video_tracking.csv")
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            csv_line=SimpleNamespace(text=lambda: raw),
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
        ),
    )
    mw = SimpleNamespace(
        _stop_all_requested=False,
        _session_final_csv_path=None,
        _session_fps_list=[30.0],
        current_detection_cache_path=str(tmp_path / "cache.npz"),
        current_individual_properties_cache_path=None,
        current_detected_properties_cache_path=None,
        get_parameters_dict=lambda: {"FPS": 30.0},
    )
    orch = TrackingOrchestrator(main_window=mw, config=object(), panels=panels)
    orch._mw = mw
    return orch, mw, raw


def test_run_session_worker_builds_worker_over_service(qapp, tmp_path, monkeypatch):
    orch, mw, raw = _orchestrator(tmp_path)

    built = {"paths": None, "config": None, "params": None}

    class _Sig:
        def __init__(self):
            self._slots = []

        def connect(self, fn):
            self._slots.append(fn)

        def emit(self, *a):
            for fn in list(self._slots):
                fn(*a)

    class _Worker:
        def __init__(self, *, video_path, config, params, paths):
            built["paths"] = paths
            built["config"] = config
            built["params"] = params
            self.progress_signal = _Sig()
            self.finished_signal = _Sig()
            self.error_signal = _Sig()
            self.warning_signal = _Sig()

        def start(self):
            self.finished_signal.emit(
                SessionResult(True, str(0), None, [], None, ["ok"], None)
            )

    monkeypatch.setattr(tracking_module, "SessionWorker", _Worker)
    monkeypatch.setattr(
        orch,
        "_finalize_tracking_session_ui",
        lambda: mw.__setattr__("_finalized", True),
    )

    # config building: whatever the orchestrator calls to build config must be stubbed
    monkeypatch.setattr(
        orch, "_build_session_config", lambda: {"cfg": 1}, raising=False
    )

    orch._run_session_worker()

    assert built["paths"]["raw_csv_path"] == raw
    assert built["paths"]["detection_cache_path"] == mw.current_detection_cache_path
    assert built["params"] == {"FPS": 30.0}
    assert getattr(mw, "_finalized", False) is True


class _FakeControl:
    def setVisible(self, *_a, **_k):
        return None

    def setValue(self, *_a, **_k):
        return None

    def setText(self, *_a, **_k):
        return None

    def setChecked(self, *_a, **_k):
        return None

    def setEnabled(self, *_a, **_k):
        return None

    def blockSignals(self, *_a, **_k):
        return None


def _populate_stop_tracking_attrs(mw, *, session_worker) -> None:
    """Set every mw attribute stop_tracking() touches (mirrors the full
    fixture in test_trackerkit_tracking_orchestrator_dialogs.py)."""
    mw.session_worker = session_worker
    mw.csv_writer_thread = None
    mw._cache_builder_worker = None
    mw.merge_worker = None
    mw.postprocess_worker = None
    mw.dataset_worker = None
    mw.interp_worker = None
    mw.final_media_export_worker = None
    mw.preview_detection_worker = None
    mw.tracking_worker = None
    mw.progress_bar = _FakeControl()
    mw.progress_label = _FakeControl()
    mw._set_ui_controls_enabled = lambda _enabled: None
    mw.current_video_path = "video.mp4"
    mw._apply_ui_state = lambda _state: None
    mw.btn_preview = _FakeControl()
    mw.btn_start = _FakeControl()
    mw._individual_dataset_run_id = "run-id"
    mw.current_detection_cache_path = "cache.npz"
    mw.current_individual_properties_cache_path = "individual_props.npz"
    mw.current_detected_properties_cache_path = "detected_props.npz"
    mw.current_interpolated_roi_npz_path = None
    mw.current_interpolated_pose_csv_path = None
    mw.current_interpolated_pose_df = None
    mw.current_interpolated_tag_csv_path = None
    mw.current_interpolated_tag_df = None
    mw.current_interpolated_cnn_csv_paths = {}
    mw.current_interpolated_cnn_dfs = {}
    mw.current_interpolated_headtail_csv_path = None
    mw.current_interpolated_headtail_df = None
    mw.label_current_fps = _FakeControl()
    mw.label_elapsed_time = _FakeControl()
    mw.label_eta = _FakeControl()
    mw._tracking_frame_size = None
    mw._cleanup_session_logging = lambda: None


def test_stop_tracking_reaches_session_worker_should_stop(qapp, tmp_path, monkeypatch):
    """stop_tracking() must reach SessionWorker._should_stop() (Finding 1).

    Uses the real SessionWorker class (not a fake) so the assertion proves the
    stop mechanism (`stop()` / `requestInterruption()`, invoked by
    `_request_qthread_stop`) actually flips `_should_stop()` to True end to
    end -- not just that *some* callable was invoked.
    """
    from hydra_suite.trackerkit.gui.workers.session_worker import SessionWorker

    orch, mw, raw = _orchestrator(tmp_path)
    worker = SessionWorker(
        video_path="video.mp4",
        config={},
        params={"FPS": 30.0},
        paths={"raw_csv_path": raw},
    )
    assert worker._should_stop() is False
    # Simulate "still running" without actually starting the QThread, so
    # _request_qthread_stop's early "not running" guard doesn't skip the
    # stop mechanism under test.
    monkeypatch.setattr(worker, "isRunning", lambda: True)
    monkeypatch.setattr(worker, "wait", lambda *_a, **_k: True)

    _populate_stop_tracking_attrs(mw, session_worker=worker)

    orch.stop_tracking()

    assert worker._should_stop() is True


def test_stop_tracking_invokes_session_worker_stop_mechanism(tmp_path, monkeypatch):
    """stop_tracking() must include session_worker among the workers it stops."""
    orch, mw, raw = _orchestrator(tmp_path)

    stopped: list[str] = []
    cleaned: list[str] = []
    monkeypatch.setattr(
        orch,
        "_request_qthread_stop",
        lambda _worker, worker_name, **_kwargs: stopped.append(worker_name),
    )
    monkeypatch.setattr(orch, "_stop_csv_writer", lambda timeout_sec=2.0: None)
    monkeypatch.setattr(
        orch, "_cleanup_thread_reference", lambda attr_name: cleaned.append(attr_name)
    )

    _populate_stop_tracking_attrs(mw, session_worker=object())

    orch.stop_tracking()

    assert "SessionWorker" in stopped
    assert "session_worker" in cleaned


def test_on_session_finished_early_returns_when_stop_all_requested(qapp, tmp_path):
    """Finding 1: after Stop, a late finished_signal must not finalize/summarize."""
    orch, mw, raw = _orchestrator(tmp_path)
    mw._stop_all_requested = True
    mw._session_final_csv_path = "stale-before.csv"
    mw._session_summary_lines = ["stale summary"]
    finalize_calls = []
    orch._finalize_tracking_session_ui = lambda: finalize_calls.append(True)

    result = SessionResult(True, "new.csv", None, [], None, ["new summary"], None)
    orch._on_session_finished(result)

    assert finalize_calls == []
    assert mw._session_final_csv_path == "stale-before.csv"
    assert mw._session_summary_lines == ["stale summary"]


def test_on_session_error_early_returns_when_stop_all_requested(tmp_path):
    orch, mw, raw = _orchestrator(tmp_path)
    mw._stop_all_requested = True
    finalize_calls = []
    orch._finalize_tracking_session_ui = lambda: finalize_calls.append(True)

    # Must not raise (no QMessageBox parent plumbing needed) and must not finalize.
    orch._on_session_error("boom")

    assert finalize_calls == []


def test_on_session_progress_early_returns_when_stop_all_requested(tmp_path):
    orch, mw, raw = _orchestrator(tmp_path)
    mw._stop_all_requested = True

    class TrackedControl:
        def __init__(self):
            self.calls = []

        def setVisible(self, v):
            self.calls.append(("setVisible", v))

        def setValue(self, v):
            self.calls.append(("setValue", v))

        def setText(self, v):
            self.calls.append(("setText", v))

    mw.progress_bar = TrackedControl()
    mw.progress_label = TrackedControl()

    orch._on_session_progress(50, "halfway")

    assert mw.progress_bar.calls == []
    assert mw.progress_label.calls == []


def test_start_tracking_on_video_resets_session_summary_lines(tmp_path, monkeypatch):
    """Finding 2: a fresh run must not show the previous run's stale summary."""
    orch, mw, raw = _orchestrator(tmp_path)
    mw.tracking_worker = None
    mw._session_summary_lines = ["leftover from a previous failed/successful run"]
    mw.is_playing = False

    class _StopAfterReset(Exception):
        pass

    def _fake_setup_csv_writer(_backward_mode):
        # By the time this is called, the run-start reset block has already
        # executed, so raising here lets the test assert on the reset without
        # driving the rest of the (heavy) tracking-launch pipeline.
        raise _StopAfterReset

    monkeypatch.setattr(orch, "_setup_tracking_csv_writer", _fake_setup_csv_writer)

    with pytest.raises(_StopAfterReset):
        orch.start_tracking_on_video("video.mp4", backward_mode=False)

    assert mw._session_summary_lines == []


def test_start_tracking_on_video_backward_aborts_on_pending_stop(tmp_path, monkeypatch):
    """Part A fix: a Stop requested during the forward->backward handoff must
    abort the backward pass instead of silently resetting _stop_all_requested
    and proceeding (the confirmed race in start_tracking_on_video)."""
    orch, mw, raw = _orchestrator(tmp_path)
    mw.tracking_worker = None
    mw._stop_all_requested = True

    setup_csv_writer = Mock()
    monkeypatch.setattr(orch, "_setup_tracking_csv_writer", setup_csv_writer)

    orch.start_tracking_on_video("video.mp4", backward_mode=True)

    setup_csv_writer.assert_not_called()
    assert mw._stop_all_requested is True


def test_start_tracking_on_video_forward_resets_stop_all_requested(
    tmp_path, monkeypatch
):
    """Forward runs must still clear _stop_all_requested for a fresh start."""
    orch, mw, raw = _orchestrator(tmp_path)
    mw.tracking_worker = None
    mw._stop_all_requested = True
    mw.is_playing = False

    class _StopAfterReset(Exception):
        pass

    def _fake_setup_csv_writer(_backward_mode):
        # By the time this is called, the run-start reset block has already
        # executed, so raising here lets the test assert on the reset without
        # driving the rest of the (heavy) tracking-launch pipeline.
        raise _StopAfterReset

    monkeypatch.setattr(orch, "_setup_tracking_csv_writer", _fake_setup_csv_writer)

    with pytest.raises(_StopAfterReset):
        orch.start_tracking_on_video("video.mp4", backward_mode=False)

    assert mw._stop_all_requested is False
