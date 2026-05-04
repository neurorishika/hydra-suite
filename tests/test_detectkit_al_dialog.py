"""Smoke test for the DetectKit active-learning dialog."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.detectkit.gui.dialogs.active_learning import ActiveLearningDialog
from hydra_suite.detectkit.gui.models import DetectKitProject


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_dialog_constructs_with_project(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)
    assert dlg is not None
    presets = [dlg.preset_combo.itemText(i) for i in range(dlg.preset_combo.count())]
    assert "balanced" in presets
    assert "uncertainty_heavy" in presets
    assert "exploration_heavy" in presets
    dlg.close()


def test_dialog_disables_run_until_inputs_valid(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)
    assert not dlg.run_button.isEnabled()
    dlg.close()


def test_dialog_locks_inputs_while_running(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    project.active_model_path = str(tmp_path / "best.pt")
    dlg = ActiveLearningDialog(project=project)
    dlg.rb_project.setChecked(True)
    dlg._sync_run_enabled()

    assert dlg.run_button.isEnabled()

    dlg.set_running(True)

    assert not dlg.input_group.isEnabled()
    assert not dlg.acquisition_group.isEnabled()
    assert not dlg.run_button.isEnabled()
    assert "Inputs are locked" in dlg.status_label.text()

    dlg.set_running(False)

    assert dlg.input_group.isEnabled()
    assert dlg.acquisition_group.isEnabled()
    assert dlg.run_button.isEnabled()
    dlg.close()
