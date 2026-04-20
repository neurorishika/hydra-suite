"""Tests for DetectKit HistoryDialog."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_proj(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    return DetectKitProject(project_dir=tmp_path, class_names=["ant"])


_FAKE_RUNS = [
    {
        "run_id": "run_001",
        "role": "obb_direct",
        "status": "completed",
        "started_at": "2026-04-01T10:00:00",
        "spec": {"base_model": "yolo26s-obb.pt", "hyperparams": {"epochs": 50}},
        "artifact_paths": ["/some/source_model.pt"],
        "project_model_path": "/some/model.pt",
        "published_model_path": "/some/model.pt",
    },
    {
        "run_id": "run_002",
        "role": "seq_detect",
        "status": "failed",
        "started_at": "2026-04-02T10:00:00",
        "spec": {"base_model": "yolo26s.pt", "hyperparams": {"epochs": 30}},
        "artifact_paths": [],
        "published_model_path": "",
    },
]


def test_history_dialog_imports(qapp):
    from hydra_suite.detectkit.gui.dialogs.history_dialog import (  # noqa: F401
        HistoryDialog,
        _load_runs,
    )


def test_history_dialog_is_base_dialog(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_history_dialog_has_close_accept_buttons(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_history_dialog_populates_table(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert dlg.table.rowCount() == 2


def test_history_dialog_uses_classkit_style_table(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert dlg.table.alternatingRowColors() is True
    assert "alternate-background-color: #2d2d30" in dlg.table.styleSheet()


def test_history_dialog_empty_when_no_runs(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: [])
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert dlg.table.rowCount() == 0


def test_history_dialog_load_for_inference_sets_active_model(
    qapp, tmp_path, monkeypatch
):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    proj = _make_proj(tmp_path)
    dlg = HistoryDialog(proj)
    dlg.table.selectRow(0)
    dlg._load_for_inference()
    assert proj.active_model_path == "/some/model.pt"


def test_history_dialog_has_detail_label(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "detail_label")
    assert "run_001" in dlg.detail_label.text()


def test_history_dialog_has_export_button(qapp, tmp_path, monkeypatch):
    import hydra_suite.detectkit.gui.dialogs.history_dialog as hd

    monkeypatch.setattr(hd, "_load_runs", lambda proj: _FAKE_RUNS)
    from hydra_suite.detectkit.gui.dialogs.history_dialog import HistoryDialog

    dlg = HistoryDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "_btn_export")
