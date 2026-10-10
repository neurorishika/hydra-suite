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
    monkeypatch.setattr(app, "setup_logging", lambda **k: None)  # keep caplog
    monkeypatch.setattr(app, "run_tracking_cli", _boom)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as ei:
            app.main(["track", "video.mp4"])
    assert ei.value.code == 1
    lines = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Error:")
    ]
    assert len(lines) == 1 and "1024" in lines[0]
    assert caplog.records and all(r.exc_info is None for r in caplog.records)


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


# --- GUI guards for N x arenas > limit (offscreen) -------------------------


def _raiser():
    raise DetectionLimitError("N=2000 exceeds the hard limit of 1024")


@pytest.fixture
def shown(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    msgs = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: msgs.append(a[1:]))
    return msgs


def _orch(mw):
    from hydra_suite.trackerkit.gui.orchestrators import tracking as tmod

    orch = tmod.TrackingOrchestrator.__new__(tmod.TrackingOrchestrator)
    orch._mw = mw
    orch._panels = SimpleNamespace()
    return orch


def test_gui_preview_guard(shown):
    mw = SimpleNamespace(
        tracking_worker=None,
        _stop_all_requested=False,
        is_playing=False,
        get_parameters_dict=_raiser,
    )
    assert _orch(mw).start_preview_on_video("v.mp4") is None
    assert shown and shown[0][0] == "Detection limit exceeded"


def test_gui_calibration_guard(shown):
    orch = _orch(
        SimpleNamespace(current_video_path="v.mp4", get_parameters_dict=_raiser)
    )
    orch._calibration_is_active = lambda: False
    orch._calibration_dialog = None
    orch.open_calibration_dialog()
    assert shown and shown[0][0] == "Detection limit exceeded"


def test_parameter_changed_slot_shows_live_message(shown):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    status, label = [], SimpleNamespace(setText=lambda t: status.append(("lbl", t)))
    emitted = []
    fake = SimpleNamespace(
        get_parameters_dict=_raiser,
        statusBar=lambda: SimpleNamespace(
            showMessage=lambda m: status.append(("bar", m))
        ),
        _setup_panel=SimpleNamespace(lbl_animals_per_arena_total=label),
        parameters_changed=SimpleNamespace(emit=emitted.append),
    )
    MainWindow._on_parameter_changed(fake)  # must not raise
    assert not emitted and not shown  # no modal per keystroke
    assert any("1024" in t for _, t in status)


def test_bg_helper_entry_guard(shown, monkeypatch, tmp_path):
    from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    orch = ConfigOrchestrator.__new__(ConfigOrchestrator)
    orch._mw = SimpleNamespace()
    orch.get_parameters_dict = _raiser
    orch._panels = SimpleNamespace(
        setup=SimpleNamespace(file_line=SimpleNamespace(text=lambda: str(video)))
    )
    orch._open_bg_parameter_helper()
    assert shown and shown[0][0] == "Detection limit exceeded"


# --- I2: GUI config load never silently clamps N ------------------------------


def _load_core_tracking(max_targets, start_value=26):
    """Run the real ``_load_config_core_tracking`` against real setup spinbox
    semantics; every other widget it touches is a permissive stand-in."""
    from unittest.mock import MagicMock

    from PySide6.QtWidgets import QApplication, QSpinBox

    from hydra_suite.core.inference.limits import MAX_DETECTIONS_PER_FRAME
    from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator

    QApplication.instance() or QApplication([])
    spin = QSpinBox()
    spin.setRange(1, MAX_DETECTIONS_PER_FRAME)  # as setup_panel builds it
    spin.setValue(start_value)
    orch = ConfigOrchestrator.__new__(ConfigOrchestrator)
    orch._mw = MagicMock()
    orch._panels = MagicMock()
    orch._panels.setup.spin_max_targets = spin
    cfg = {"max_targets": max_targets}

    def get_cfg(*keys, default=None):
        for k in keys:
            if k in cfg:
                return cfg[k]
        return default

    try:
        orch._load_config_core_tracking(get_cfg, lambda *a, **k: 1.0)
    except Exception:  # noqa: BLE001 - later stand-in widgets are irrelevant
        pass
    return spin


def test_loading_config_above_limit_reports_and_does_not_clamp(shown):
    import inspect

    from hydra_suite.core.inference.limits import MAX_DETECTIONS_PER_FRAME
    from hydra_suite.trackerkit.gui.panels import setup_panel

    assert "spin_max_targets.setRange(1, MAX_DETECTIONS_PER_FRAME)" in (
        inspect.getsource(setup_panel)
    )
    spin = _load_core_tracking(2000)
    assert shown and shown[0][0] == "Detection limit exceeded"
    assert "2000" in shown[0][1] and "1024" in shown[0][1]
    assert spin.value() == 26, "N above the limit must not be clamped to it"
    assert spin.value() != MAX_DETECTIONS_PER_FRAME


def test_loading_config_within_limit_applies_silently(shown):
    spin = _load_core_tracking(300)
    assert spin.value() == 300
    assert not shown


# --- M1: calibrate / job print one line, no traceback ------------------------


@pytest.mark.parametrize(
    "argv, module, attr",
    [
        (["calibrate", "--video", "video.mp4"], "calibrate_cli", "run_calibrate_cli"),
        (["job", "run", "jobdir"], "job_cli", "run_job_cli"),
    ],
)
def test_calibrate_and_job_print_one_line_limit_error(
    monkeypatch, capsys, caplog, argv, module, attr
):
    import importlib

    from hydra_suite.trackerkit import app

    def _boom(*a, **k):
        require_target_count_within_limit(5000)

    monkeypatch.setattr(app, "check_dependencies", lambda: True)
    # basicConfig(force=True) would remove caplog's root handler.
    monkeypatch.setattr(app, "setup_logging", lambda **k: None)
    monkeypatch.setattr(
        importlib.import_module(f"hydra_suite.trackerkit.{module}"), attr, _boom
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as ei:
            app.main(argv)
    assert ei.value.code == 1
    lines = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Error:")
    ]
    assert len(lines) == 1 and "1024" in lines[0]
    assert caplog.records and all(r.exc_info is None for r in caplog.records)
