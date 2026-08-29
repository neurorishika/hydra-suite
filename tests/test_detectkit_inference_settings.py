"""Tests for DetectKit's runtime-only inference settings."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.models import (  # noqa: E402
    DetectKitProject,
    InferenceRunSettings,
    SliceTrainingSettings,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_runtime_inference_settings_snapshot_does_not_mutate_project(tmp_path):
    project = DetectKitProject(project_dir=tmp_path, device="mps")
    project.slice_settings = SliceTrainingSettings(
        enabled=True,
        target_sizes=[200.0, 300.0, 400.0],
    )

    runtime = InferenceRunSettings.from_project(project, confidence_threshold=0.72)
    runtime.slice_settings.target_sizes[:] = [160.0]
    runtime.slice_settings.overlap = 0.35

    assert project.slice_settings.target_sizes == [200.0, 300.0, 400.0]
    assert project.slice_settings.overlap == 0.2
    assert runtime.device == "mps"
    assert runtime.confidence_threshold == 0.72


def test_runtime_inference_settings_cache_key_changes_for_sahi_geometry():
    settings = InferenceRunSettings(
        device="mps",
        confidence_threshold=0.5,
        slice_settings=SliceTrainingSettings(enabled=True, target_sizes=[200.0]),
    )
    old_key = settings.cache_key()
    settings.slice_settings.target_sizes = [400.0]
    assert settings.cache_key() != old_key


def test_inference_settings_dialog_is_runtime_only(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )

    project = DetectKitProject(project_dir=tmp_path, device="auto")
    project.slice_settings = SliceTrainingSettings(enabled=False)
    defaults = InferenceRunSettings.from_project(project, confidence_threshold=0.5)
    dialog = InferenceSettingsDialog(defaults, defaults)
    dialog.chk_sliced.setChecked(True)
    dialog.combo_geometry.setCurrentText("auto_object")
    dialog.spin_target_size.setValue(200)
    result = dialog.settings()

    assert result.slice_settings.enabled is True
    assert result.slice_settings.target_sizes == [200.0]
    assert project.slice_settings.enabled is False


def test_tools_panel_exposes_inference_settings_button(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert panel._btn_inference_settings.text() == "Inference Settings…"
