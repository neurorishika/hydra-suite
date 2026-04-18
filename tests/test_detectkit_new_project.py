"""Tests for DetectKit new-project dialog and main-window flow."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from hydra_suite.detectkit.gui.dialogs import NewProjectDialog  # noqa: E402
from hydra_suite.detectkit.gui.main_window import MainWindow  # noqa: E402
from hydra_suite.detectkit.gui.project import (  # noqa: E402
    create_project,
    project_file_path,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def test_detectkit_new_project_dialog_defaults_to_detectkit_projects_root(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()

    assert dialog.location_edit.text() == str(tmp_path / "hydra-projects" / "DetectKit")
    assert (
        dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled() is False
    )

    dialog.name_edit.setText("ants")
    project_info = dialog.get_project_info()

    assert dialog._buttons.button(dialog._buttons.StandardButton.Ok).isEnabled() is True
    assert project_info == {
        "name": "ants",
        "path": str(tmp_path / "hydra-projects" / "DetectKit" / "ants"),
        "class_names": ["object"],
        "class_name": "object",
    }
    dialog.close()


def test_detectkit_new_project_dialog_returns_custom_class_names(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()
    dialog.name_edit.setText("colony_a")
    dialog.class_names_edit.setPlainText("ant\nbee")

    assert (
        dialog.get_project_path()
        == tmp_path / "hydra-projects" / "DetectKit" / "colony_a"
    )
    assert dialog.get_project_info()["class_names"] == ["ant", "bee"]
    assert dialog.get_project_info()["class_name"] == "ant"
    dialog.close()


def test_detectkit_main_window_new_project_creates_project_from_dialog(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_path = tmp_path / "DetectKit" / "experiment_a"

    class _FakeDialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, _parent=None):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_project_info(self):
            return {
                "name": "experiment_a",
                "path": str(project_path),
                "class_names": ["ant", "bee"],
                "class_name": "ant",
            }

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.NewProjectDialog",
        _FakeDialog,
    )

    window = MainWindow()
    loaded_projects = []
    monkeypatch.setattr(
        window, "_load_project", lambda proj: loaded_projects.append(proj)
    )

    window.new_project()

    assert project_path.exists()
    assert project_file_path(project_path).exists()
    assert loaded_projects
    assert loaded_projects[0].project_dir == project_path
    assert loaded_projects[0].class_name == "ant"
    assert loaded_projects[0].class_names == ["ant", "bee"]
    window.close()


def test_detectkit_main_window_new_project_opens_existing_project(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_path = tmp_path / "DetectKit" / "existing_project"
    create_project(project_path, "bee")

    class _FakeDialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, _parent=None):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_project_info(self):
            return {
                "name": "existing_project",
                "path": str(project_path),
                "class_names": ["ant", "bee"],
                "class_name": "ant",
            }

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.NewProjectDialog",
        _FakeDialog,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    window = MainWindow()
    loaded_projects = []
    monkeypatch.setattr(
        window, "_load_project", lambda proj: loaded_projects.append(proj)
    )

    window.new_project()

    assert loaded_projects
    assert loaded_projects[0].project_dir == project_path
    assert loaded_projects[0].class_name == "bee"
    window.close()


def test_detectkit_main_window_side_panels_keep_readable_minimum_widths(qapp):
    window = MainWindow()

    assert hasattr(window, "splitter")
    assert window._dataset_panel.minimumWidth() >= 360
    assert window._right_tabs.minimumWidth() >= 480
    assert window._canvas.minimumWidth() >= 480
    assert window.minimumWidth() >= 1300
    assert window.splitter.childrenCollapsible() is False
    assert window.splitter.isCollapsible(0) is False
    assert window.splitter.isCollapsible(1) is False
    assert window.splitter.isCollapsible(2) is False

    window.close()
