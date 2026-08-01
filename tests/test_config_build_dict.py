"""build_config_dict() returns the same dict save_config would write, without touching disk."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def qtbot_config_stub(monkeypatch, qapp):
    """A real, offscreen MainWindow's ConfigOrchestrator (no disk/network I/O).

    Mirrors the offscreen-MainWindow convention in
    tests/test_vitpose_trackerkit_persistence.py: construct a real MainWindow
    under QT_QPA_PLATFORM=offscreen with the advanced-config disk hooks
    stubbed out, so the widget tree is real and purity assertions on
    build_config_dict() are meaningful.
    """
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window._config_orch
    finally:
        window.close()


def test_build_config_dict_is_pure(monkeypatch, qtbot_config_stub):
    orch = qtbot_config_stub  # a constructed ConfigOrchestrator with real panels (offscreen)
    called = {"atomic_write": 0, "resolve_path": 0}
    monkeypatch.setattr(
        orch,
        "_atomic_json_write",
        lambda *a, **k: called.__setitem__("atomic_write", called["atomic_write"] + 1)
        or (True, None),
    )
    monkeypatch.setattr(
        orch,
        "_resolve_config_save_path",
        lambda *a, **k: called.__setitem__("resolve_path", called["resolve_path"] + 1)
        or None,
    )

    cfg = orch.build_config_dict()

    assert isinstance(cfg, dict)
    assert "detection_method" in cfg
    assert "enable_pose_extractor" in cfg
    assert called["atomic_write"] == 0
    assert called["resolve_path"] == 0
