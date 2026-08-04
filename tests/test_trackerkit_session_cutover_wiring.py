"""The GUI post-tracking flow builds a SessionWorker over run_post_tracking."""

from __future__ import annotations

import os
from types import SimpleNamespace

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
        current_detected_cnn_cache_paths={},
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
