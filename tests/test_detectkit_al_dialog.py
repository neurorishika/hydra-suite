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


def test_dialog_gates_by_default_without_set_model_task(qapp, tmp_path):
    """Regression: the dialog must be self-consistent from construction, not
    only after a caller remembers to call `set_model_task`.

    `_build_levels_group` starts every checkbox checked and enabled; if that
    ungated state survived until `build_request()`, a caller that skips
    `set_model_task` would get `export_level="polygon"` with all three
    levels requested for whatever default project/model is loaded -- the
    exact bug this task closes, just relocated to depend on caller
    discipline.
    """
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)

    assert dlg.chk_level_polygon.isEnabled() is False
    request = dlg.build_request()

    assert request.export_level == "obb"
    assert request.export_levels == ["obb", "aabb"]
    assert request.native_level == "obb"
    dlg.close()


def test_dialog_sets_export_level_from_the_model_task(qapp, tmp_path):
    """Regression: ALRequest.export_level was never set, so it stayed 'obb'."""
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)

    dlg.set_model_task("segment")
    request = dlg.build_request()

    assert request.export_level == "polygon"
    assert "polygon" in request.export_levels
    dlg.close()


def test_dialog_refuses_polygon_for_an_obb_model(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)

    dlg.set_model_task("obb")

    assert dlg.chk_level_polygon.isEnabled() is False
    assert dlg.build_request().export_level == "obb"
    dlg.close()


def test_dialog_gates_all_but_aabb_for_a_detect_only_model(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)

    dlg.set_model_task("detect")

    assert dlg.chk_level_polygon.isEnabled() is False
    assert dlg.chk_level_obb.isEnabled() is False
    assert dlg.chk_level_aabb.isEnabled() is True
    request = dlg.build_request()
    assert request.export_level == "aabb"
    assert request.export_levels == ["aabb"]
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
