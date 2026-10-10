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


def _settle(qapp) -> None:
    for _ in range(5):
        qapp.processEvents()


def _inference_dialog(enabled: bool):
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )

    current = InferenceRunSettings(
        slice_settings=SliceTrainingSettings(enabled=enabled)
    )
    defaults = InferenceRunSettings(slice_settings=SliceTrainingSettings(enabled=True))
    return InferenceSettingsDialog(current, defaults, model_input_size=1024)


def _rows_do_not_overlap(widget) -> bool:
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox

    rects = []
    for kind in (QAbstractSpinBox, QComboBox):
        for child in widget.findChildren(kind):
            if child.isVisibleTo(widget):
                rects.append(
                    QRect(child.mapTo(widget, child.rect().topLeft()), child.size())
                )
    return all(not a.intersects(b) for i, a in enumerate(rects) for b in rects[i + 1 :])


def test_inference_dialog_collapse_still_works_after_advanced_was_opened(qapp):
    dlg = _inference_dialog(True)
    dlg.show()
    _settle(qapp)
    dlg.slice_widget.btn_slice_advanced.setChecked(True)
    _settle(qapp)
    advanced_height = dlg.height()
    width = dlg.width()
    dlg.chk_sliced.setChecked(False)
    _settle(qapp)
    assert dlg.height() < advanced_height - 100
    assert dlg.width() == width  # no width wobble on collapse
    dlg.chk_sliced.setChecked(True)
    _settle(qapp)
    assert dlg.height() == advanced_height
    dlg.close()


def test_inference_dialog_collapsed_has_no_dead_gap(qapp):
    dlg = _inference_dialog(False)
    dlg.show()
    _settle(qapp)
    layout = dlg.layout()
    margins = dlg.contentsMargins()
    needed = layout.totalHeightForWidth(dlg.width() - margins.left() - margins.right())
    assert dlg.height() <= needed + 2
    dlg.close()


def test_restore_defaults_from_sahi_off_refits_without_clipping(qapp):
    dlg = _inference_dialog(False)
    dlg.show()
    _settle(qapp)
    dlg.btn_restore_defaults.click()  # defaults turn SAHI on (signals blocked)
    _settle(qapp)
    assert dlg.chk_sliced.isChecked()
    assert dlg.height() >= dlg.minimumSizeHint().height()
    assert _rows_do_not_overlap(dlg.slice_widget)
    dlg.close()


def test_user_resize_is_honoured_across_toggles_and_advanced(qapp):
    dlg = _inference_dialog(True)
    dlg.show()
    _settle(qapp)
    dlg.resize(dlg.width() + 60, dlg.height() + 150)
    _settle(qapp)
    user = dlg.size()
    for toggle in (dlg.chk_sliced, dlg.chk_sliced):
        toggle.setChecked(not toggle.isChecked())
        _settle(qapp)
    assert dlg.size() == user
    dlg.close()
