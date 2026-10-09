"""SAM3 / SAM2 escalation dialogs host the shared SAHI widget (S4 Task 17)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource  # noqa: E402
from hydra_suite.widgets.slice_settings import SliceSettingsWidget  # noqa: E402

# SemanticEscalationDialog.parameters() keys on main before S4 (pasted).
SAM3_PARAMETER_KEYS = {
    "class_name",
    "device",
    "confidence",
    "max_instances",
    "overlap",
    "seam_margin_px",
    "merge_iou",
    "reference_body_px",
    "tile_fraction",
    "area_min_px2",
    "area_max_px2",
}


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # Repo convention (no pytest-qt): see tests/test_detectkit_review_bar.py.
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def available_checkpoint(monkeypatch):
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod

    available = SimpleNamespace(usable=True, checkpoint_missing=False, reason="")
    monkeypatch.setattr(mod, "probe_checkpoint", lambda *_a, **_k: available)
    return mod


def _sam3(body: float = 82.2, **kwargs):
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    return SemanticEscalationDialog([OBBSource(name="s", level="obb")], body, **kwargs)


def _sam2(**kwargs):
    from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import (
        EscalateSam2Dialog,
    )

    return EscalateSam2Dialog([OBBSource(name="s", level="obb")], **kwargs)


def test_sam3_dialog_contains_one_escalate_sam3_widget(available_checkpoint):
    dialog = _sam3()
    widgets = dialog.findChildren(SliceSettingsWidget)
    assert [w.role for w in widgets] == ["escalate_sam3"]
    w = widgets[0]
    assert dialog._reference_body is w.spin_slice_body
    assert dialog._tile_fraction is w.spin_slice_object_fraction
    assert dialog._overlap is w.spin_slice_overlap
    assert dialog._merge_iou is w.spin_slice_merge
    assert dialog._seam_margin is w.spin_slice_seam_margin
    assert dialog._tile_label is w.lbl_slice_tile_size


def test_sam3_parameters_emit_exactly_todays_keys(available_checkpoint):
    params = _sam3().parameters()
    assert set(params) == SAM3_PARAMETER_KEYS
    assert isinstance(params["seam_margin_px"], float)


def test_sam3_fresh_dialog_overlap_is_the_semantic_default(available_checkpoint):
    from hydra_suite.core.inference.semantic.tiling import DEFAULT_OVERLAP

    assert _sam3().parameters()["overlap"] == pytest.approx(DEFAULT_OVERLAP)


def test_sam3_fresh_dialog_overlap_meets_the_whole_animal_minimum(
    available_checkpoint,
):
    """The saved/default 0.5 is above max(scale)+margin: no nudge, no button."""
    widget = _sam3().findChildren(SliceSettingsWidget)[0]
    assert widget.lbl_slice_overlap_minimum.text().startswith("≥")
    assert widget.btn_slice_overlap_raise.isHidden()


def test_sam3_body_badge_is_dataset_when_seeded_from_the_labels(
    available_checkpoint,
):
    w = _sam3(
        82.2,
        body_px_origin="the median longest side of your existing labels",
        body_px_source="dataset",
    )
    widget = w.findChildren(SliceSettingsWidget)[0]
    assert widget.source_badge("reference_body_px") == "dataset"
    assert not w._reference_body.isEnabled()  # read-only until Override


def test_sam3_body_badge_is_stamped_after_prefill_from_sidecar(
    available_checkpoint, monkeypatch
):
    dialog = _sam3(0.0)
    widget = dialog.findChildren(SliceSettingsWidget)[0]
    assert widget.source_badge("reference_body_px") == "user"
    assert dialog._reference_body.isEnabled()  # unknown body stays typeable (I6)
    monkeypatch.setattr(
        available_checkpoint,
        "sidecar_for",
        lambda _key: {"reference_body_px": 64.0, "object_tile_fraction": 0.06},
    )
    dialog.prefill_from_sidecar("published-model")
    assert dialog.reference_body_px() == pytest.approx(64.0)
    assert widget.source_badge("reference_body_px") == "stamped"


def test_sam3_full_frame_round_trip_through_a_calibration_choice(
    available_checkpoint,
):
    dialog = _sam3()
    dialog.apply_calibration_choice(SimpleNamespace(tile_fraction=None, confidence=0.4))
    assert dialog.tile_fraction() is None
    assert dialog.parameters()["tile_fraction"] is None
    assert "full frame" in dialog._tile_label.text()
    dialog.apply_calibration_choice(SimpleNamespace(tile_fraction=0.08, confidence=0.4))
    assert dialog.tile_fraction() == pytest.approx(0.08)


def test_sam3_settings_persist_and_restore(available_checkpoint, tmp_path):
    project = DetectKitProject(project_dir=Path(tmp_path))
    first = _sam3(50.0, project=project)
    first._overlap.setValue(0.3)
    first._merge_iou.setValue(0.6)
    first._seam_margin.setValue(12)
    first._persist_settings()
    reopened = _sam3(50.0, project=project)
    params = reopened.parameters()
    assert params["overlap"] == pytest.approx(0.3)
    assert params["merge_iou"] == pytest.approx(0.6)
    assert params["seam_margin_px"] == pytest.approx(12.0)


def test_sam2_dialog_contains_one_escalate_sam2_widget():
    dialog = _sam2()
    widgets = dialog.findChildren(SliceSettingsWidget)
    assert [w.role for w in widgets] == ["escalate_sam2"]
    assert dialog._reference_body is widgets[0].spin_slice_body
    assert dialog._tile_fraction is widgets[0].spin_slice_object_fraction
    assert not widgets[0].spin_slice_overlap.isEnabled()


def test_sam2_tiling_parameters_keys_unchanged():
    from hydra_suite.core.inference.semantic.tiling import DEFAULT_OVERLAP

    params = _sam2(reference_body_px=40.0).tiling_parameters()
    assert set(params) == {"reference_body_px", "tile_fraction", "overlap"}
    assert params["overlap"] == DEFAULT_OVERLAP


def test_sam3_body_badge_is_project_for_the_sliced_training_reference(
    available_checkpoint,
):
    dialog = _sam3(
        48.0,
        body_px_origin="the project's sliced-training reference body size",
        body_px_source="project",
    )
    widget = dialog.findChildren(SliceSettingsWidget)[0]
    assert widget.source_badge("reference_body_px") == "project"
    assert not widget.chk_slice_body_override.isHidden()


def test_sam2_body_badge_carries_the_real_source():
    for source in ("project", "dataset"):
        dialog = _sam2(reference_body_px=40.0, body_px_source=source)
        widget = dialog.findChildren(SliceSettingsWidget)[0]
        assert widget.source_badge("reference_body_px") == source
    widget = _sam2(reference_body_px=0.0).findChildren(SliceSettingsWidget)[0]
    assert widget.source_badge("reference_body_px") == "user"


def test_reference_body_resolver_names_its_source(tmp_path):
    from hydra_suite.detectkit.gui.escalation_actions import resolve_reference_body
    from hydra_suite.detectkit.gui.models import SliceTrainingSettings

    project = DetectKitProject(project_dir=tmp_path)
    project.slice_settings = SliceTrainingSettings(reference_body_px=50.0)
    value, _note, source = resolve_reference_body(project)
    assert (value, source) == (50.0, "project")
    project.slice_settings = SliceTrainingSettings(reference_body_px=0.0)
    value, _note, source = resolve_reference_body(project)
    assert (value, source) == (0.0, "user")


def test_sam2_full_frame_round_trip_through_a_calibration_choice():
    dialog = _sam2(reference_body_px=40.0)
    point = SimpleNamespace(
        tile_fraction=None, tile_px=None, median_iou=0.5, seconds_per_frame=1.0
    )
    dialog.apply_calibration_choice(point)
    assert dialog.tiling_parameters()["tile_fraction"] is None
    assert dialog._tile_label.text() == "full frame — tiling off."
    point.tile_fraction, point.tile_px = 0.1, 400
    dialog.apply_calibration_choice(point)
    assert dialog._tile_label.text() == "400 px (40 px / 0.1)"


def _rows_do_not_collide(dialog) -> None:
    from PySide6.QtCore import QPoint, QRect

    dialog.show()
    for _ in range(4):
        QApplication.processEvents()
    label = dialog._tile_label
    assert label.height() >= label.heightForWidth(label.width())
    rows = {
        "fraction": dialog._tile_fraction,
        "body": dialog._reference_body,
        "label": label,
        "overlap": dialog._tiling.spin_slice_overlap,
    }
    rects = {
        name: QRect(widget.mapTo(dialog, QPoint(0, 0)), widget.size())
        for name, widget in rows.items()
    }
    names = list(rects)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            assert not rects[first].intersects(rects[second]), (first, second)
    dialog.hide()


@pytest.mark.parametrize("body", [0.0, 82.2])
def test_sam3_tiling_rows_never_overlap(available_checkpoint, body):
    dialog = _sam3(body, body_px_origin="")
    if body == 0.0:
        assert "enter a body size" in dialog._tile_label.text().lower()
        assert "small-object recall" in dialog._tile_label.toolTip()
    _rows_do_not_collide(dialog)


@pytest.mark.parametrize("body", [0.0, 40.0])
def test_sam2_tiling_rows_never_overlap(body):
    dialog = _sam2(reference_body_px=body)
    dialog._tile_fraction.setValue(0.1)
    _rows_do_not_collide(dialog)


def test_sam3_dialog_shrinks_back_after_advanced_collapses(available_checkpoint):
    dialog = _sam3()
    dialog.show()
    for _ in range(3):
        QApplication.processEvents()
    min_height, height = dialog.minimumHeight(), dialog.height()
    dialog._tiling.btn_slice_advanced.click()
    QApplication.processEvents()
    assert dialog.minimumHeight() >= min_height
    assert dialog.height() >= dialog.minimumHeight()
    dialog._tiling.btn_slice_advanced.click()
    QApplication.processEvents()
    assert dialog.minimumHeight() == min_height
    assert dialog.height() == height
    dialog.hide()


def test_fit_keeps_a_user_resized_dialog_size(available_checkpoint):
    dialog = _sam3()
    dialog.show()
    QApplication.processEvents()
    dialog.resize(dialog.width() + 120, dialog.height() + 80)
    user_size = dialog.size()
    dialog._tiling.btn_slice_advanced.click()
    dialog._tiling.btn_slice_advanced.click()
    QApplication.processEvents()
    assert dialog.size() == user_size
    dialog.hide()


def _growing_dialog():
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QWidget

    from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog

    dialog = DetectKitDialog("fit")
    content = QWidget()
    content.setMinimumSize(300, 120)
    dialog.add_content(content)
    dialog.resize(420, 260)
    dialog.fit_to_content(QSize(320, 200))
    return dialog, content


def test_fit_to_content_grows_then_shrinks_back():
    dialog, content = _growing_dialog()
    dialog.show()
    QApplication.processEvents()
    min_height, size = dialog.minimumHeight(), dialog.size()
    content.setMinimumHeight(700)
    dialog.fit_to_content()
    assert dialog.minimumHeight() > 700 and dialog.height() >= dialog.minimumHeight()
    content.setMinimumHeight(120)
    dialog.fit_to_content()
    assert dialog.minimumHeight() == min_height
    assert dialog.size() == size
    dialog.hide()


def test_fit_to_content_reruns_on_show():
    dialog, content = _growing_dialog()
    content.setMinimumHeight(700)  # grew while hidden, nobody re-fitted
    dialog.show()
    QApplication.processEvents()
    assert dialog.minimumHeight() > 700
    dialog.hide()


def test_user_resize_survives_repeated_toggles_and_reshow(available_checkpoint):
    dialog = _sam3()
    dialog.show()
    QApplication.processEvents()
    dialog.resize(dialog.width() + 150, dialog.height() + 90)
    QApplication.processEvents()
    user_size = dialog.size()
    for _ in range(3):
        dialog._tiling.btn_slice_advanced.click()  # the user path: re-fits
        dialog._tiling.btn_slice_advanced.click()
        QApplication.processEvents()
    assert dialog.size() == user_size
    dialog.hide()
    dialog.show()
    for _ in range(3):
        QApplication.processEvents()
    assert dialog.size() == user_size
    dialog.hide()


def test_pre_expanded_dialog_fits_after_show():
    dialog, content = _growing_dialog()
    content.setMinimumHeight(700)  # grew before the first show
    dialog.show()
    for _ in range(3):
        QApplication.processEvents()
    assert dialog.height() >= dialog.minimumSizeHint().height()
    dialog.hide()
