"""Tests for the ClassKit new-project dialog."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

from hydra_suite.classkit.config.presets import save_scheme_preset  # noqa: E402
from hydra_suite.classkit.config.schemas import Factor, LabelingScheme  # noqa: E402
from hydra_suite.classkit.gui.dialogs.new_project import NewProjectDialog  # noqa: E402


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


def test_classkit_new_project_dialog_defaults_to_classkit_projects_root(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()
    create_button = dialog._buttons.button(QDialogButtonBox.Ok)

    assert dialog.location_edit.text() == str(tmp_path / "hydra-projects" / "ClassKit")
    assert create_button.isEnabled() is False

    dialog.name_edit.setText("colony_labels")

    assert create_button.isEnabled() is True
    assert (
        dialog.get_project_path()
        == tmp_path / "hydra-projects" / "ClassKit" / "colony_labels"
    )
    assert dialog.get_project_info()["path"] == str(
        tmp_path / "hydra-projects" / "ClassKit" / "colony_labels"
    )


def test_classkit_new_project_dialog_returns_selected_preset_classes(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()
    dialog.name_edit.setText("ages")
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("age"))

    project_info = dialog.get_project_info()

    assert project_info["classes"] == ["young", "old"]
    assert project_info["scheme"] is not None


def test_classkit_new_project_dialog_lists_saved_custom_presets(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "config"))

    save_scheme_preset(
        "Colony Tags",
        LabelingScheme(
            name="colony_tags",
            factors=[
                Factor(name="tag_1", labels=["red", "blue"]),
                Factor(name="tag_2", labels=["left", "right"]),
            ],
            training_modes=["flat_tiny", "multihead_tiny"],
        ),
    )

    dialog = NewProjectDialog()
    dialog.name_edit.setText("colony")
    preset_index = dialog.preset_combo.findData("custom:colony_tags")

    assert preset_index >= 0

    dialog.preset_combo.setCurrentIndex(preset_index)
    project_info = dialog.get_project_info()

    # Multi-factor schemes auto-append "unknown" to each factor (LabelingScheme
    # __post_init__), so the flattened class list carries a per-factor "unknown".
    assert project_info["classes"] == [
        "red",
        "blue",
        "unknown",
        "left",
        "right",
        "unknown",
    ]
    assert project_info["scheme"] is not None
    assert project_info["scheme"].name == "colony_tags"
