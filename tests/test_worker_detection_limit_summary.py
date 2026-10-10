"""End-of-run detection-limit summary and user-facing limit errors."""

import logging
from types import SimpleNamespace

import pytest

from hydra_suite.core.canonicalization.geometry import ClippingStats
from hydra_suite.core.inference.limits import (
    DetectionLimitError,
    DetectionLimitStats,
    require_target_count_within_limit,
)


def test_worker_emits_detection_limit_summary():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    stats = DetectionLimitStats()
    stats.record(12, 3000)
    runner = SimpleNamespace(
        clipping_stats=ClippingStats(), detection_limit_stats=stats
    )
    seen = []
    worker = TrackingEngineCore.__new__(TrackingEngineCore)
    worker._on_warning = lambda title, msg: seen.append((title, msg))
    worker._report_inference_run_summaries([runner, None])
    assert seen and seen[0][0] == "Detection limit reached" and "1024" in seen[0][1]


def test_worker_silent_without_limit_hits():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    runner = SimpleNamespace(
        clipping_stats=ClippingStats(), detection_limit_stats=DetectionLimitStats()
    )
    seen = []
    worker = TrackingEngineCore.__new__(TrackingEngineCore)
    worker._on_warning = lambda title, msg: seen.append((title, msg))
    worker._report_inference_run_summaries([runner, None])
    assert seen == []


def test_record_wording_follows_criterion(caplog):
    stats = DetectionLimitStats()
    with caplog.at_level(logging.WARNING):
        stats.record(1, 2000)
        stats.record(2, 2000, criterion="area")
    assert "by confidence" in caplog.records[0].getMessage()
    assert "by area" in caplog.records[1].getMessage()


def test_setup_panel_tooltip_uses_limit_constant():
    import inspect

    from hydra_suite.trackerkit.gui.panels import setup_panel

    src = inspect.getsource(setup_panel)
    assert "(1-200)" not in src
    assert "1-{MAX_DETECTIONS_PER_FRAME}" in src


def test_cli_prints_one_line_limit_error(monkeypatch, capsys, caplog):
    from hydra_suite.trackerkit import app

    def _boom(*a, **k):
        require_target_count_within_limit(5000)

    monkeypatch.setattr(app, "check_dependencies", lambda: True)
    monkeypatch.setattr(app, "run_tracking_cli", _boom)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as ei:
            app.main(["track", "video.mp4"])
    assert ei.value.code == 1
    lines = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Error:")
    ]
    assert len(lines) == 1 and "1024" in lines[0]
    assert all(r.exc_info is None for r in caplog.records)


def test_gui_start_tracking_shows_limit_message(monkeypatch):
    from hydra_suite.trackerkit.gui.orchestrators import tracking as tmod

    shown = []
    monkeypatch.setattr(
        tmod.QMessageBox, "warning", lambda *a, **k: shown.append(a[1:])
    )

    def _raise():
        raise DetectionLimitError("N=5000 exceeds the hard limit of 1024")

    orch = tmod.TrackingOrchestrator.__new__(tmod.TrackingOrchestrator)
    orch._mw = SimpleNamespace(
        tracking_worker=None,
        _stop_all_requested=False,
        get_parameters_dict=_raise,
    )
    orch._panels = SimpleNamespace(
        setup=SimpleNamespace(csv_line=SimpleNamespace(text=lambda: "out.csv"))
    )
    assert orch.start_tracking_on_video("v.mp4") is None
    assert shown and shown[0][0] == "Detection limit exceeded"
    assert "1024" in shown[0][1]
