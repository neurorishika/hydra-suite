"""Tests for DetectKit's runtime-only inference settings."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.models import (  # noqa: E402
    INFERENCE_CONFIDENCE_FLOOR,
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
    assert project.slice_settings.overlap == SliceTrainingSettings().overlap
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


def test_runtime_inference_settings_cache_key_excludes_display_confidence():
    settings = InferenceRunSettings(confidence_threshold=0.25)
    old_key = settings.cache_key()
    settings.confidence_threshold = 0.75
    assert settings.cache_key() == old_key


def test_live_confidence_filter_reuses_low_floor_candidates():
    from hydra_suite.detectkit.gui.main_window import _filter_detections_by_confidence

    cached = [
        {"confidence": INFERENCE_CONFIDENCE_FLOOR},
        {"confidence": 0.49},
        {"confidence": 0.91},
    ]
    assert len(_filter_detections_by_confidence(cached, 0.50)) == 1
    assert len(_filter_detections_by_confidence(cached, 0.10)) == 2


def test_inference_settings_dialog_is_runtime_only(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )

    project = DetectKitProject(project_dir=tmp_path, device="auto")
    project.slice_settings = SliceTrainingSettings(enabled=False)
    defaults = InferenceRunSettings.from_project(project, confidence_threshold=0.5)
    dialog = InferenceSettingsDialog(defaults, defaults)
    dialog.chk_sliced.setChecked(True)
    dialog.combo_geometry.setCurrentIndex(dialog.combo_geometry.findData("auto_object"))
    dialog.spin_object_fraction.setValue(0.3)
    result = dialog.settings()

    assert result.slice_settings.enabled is True
    # Fraction-only (F1): no 640-anchored pixel list is written.
    assert result.slice_settings.target_size_fractions == [0.3]
    assert result.slice_settings.object_tile_fraction == 0.3
    assert project.slice_settings.enabled is False


def test_tools_panel_exposes_inference_settings_button(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert panel._btn_inference_settings.text() == "Inference Settings…"


def test_inference_dialog_collapses_when_sahi_off_and_refits(qapp):
    """S6: SAHI off collapses the slice widget to its checkbox; the dialog
    re-fits shorter (schedule_fit) and grows back when SAHI is ticked."""
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )

    settings = InferenceRunSettings(slice_settings=SliceTrainingSettings(enabled=True))
    dlg = InferenceSettingsDialog(settings, settings, model_input_size=1024)

    def settle() -> None:
        for _ in range(5):
            qapp.processEvents()

    dlg.show()
    settle()
    on_height = dlg.height()
    before = dlg.settings()
    assert dlg.slice_widget.preview.isVisibleTo(dlg)
    dlg.chk_sliced.setChecked(False)
    settle()
    assert not dlg.slice_widget.preview.isVisibleTo(dlg)
    assert not dlg.combo_geometry.isVisibleTo(dlg)
    assert dlg.height() < on_height - 100
    dlg.chk_sliced.setChecked(True)
    settle()
    assert dlg.height() == on_height
    assert dlg.settings() == before
    # Opening Advanced grows the dialog so no row is squeezed.
    dlg.slice_widget.btn_slice_advanced.setChecked(True)
    settle()
    assert dlg.height() >= dlg.minimumSizeHint().height()
    assert dlg.height() > on_height
    dlg.close()


def test_inference_dialog_keeps_a_user_resized_height(qapp):
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )

    settings = InferenceRunSettings(slice_settings=SliceTrainingSettings(enabled=True))
    dlg = InferenceSettingsDialog(settings, settings, model_input_size=1024)
    dlg.show()
    for _ in range(5):
        qapp.processEvents()
    dlg.resize(dlg.width(), dlg.height() + 120)  # the user drags it taller
    user_height = dlg.height()
    dlg.chk_sliced.setChecked(False)
    dlg.chk_sliced.setChecked(True)
    for _ in range(5):
        qapp.processEvents()
    assert dlg.height() == user_height
    dlg.close()
