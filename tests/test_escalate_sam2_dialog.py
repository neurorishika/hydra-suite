import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication

from hydra_suite.core.inference.sam2.checkpoints import (
    DEFAULT_VARIANT,
    available_variants,
)
from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import EscalateSam2Dialog
from hydra_suite.detectkit.gui.models import OBBSource

_app = QApplication.instance() or QApplication([])


def test_dialog_lists_variants_and_eligible_sources():
    sources = [
        OBBSource(name="a", level="obb"),
        OBBSource(name="b_seg", level="polygon"),
    ]  # already polygon -> disabled
    dlg = EscalateSam2Dialog(sources)
    assert dlg.selected_variant() == DEFAULT_VARIANT
    assert set(dlg._variant_combo_items()) == set(available_variants())
    # 'a' selectable, 'b_seg' disabled
    assert "a" in dlg.selectable_source_names()
    assert "b_seg" not in dlg.selectable_source_names()
    assert dlg.selected_sources() == ["a"]


def test_dialog_uses_clickable_multi_selection_instead_of_checkboxes():
    sources = [
        OBBSource(name="a", level="obb"),
        OBBSource(name="b", level="aabb"),
    ]
    dlg = EscalateSam2Dialog(sources)

    assert dlg._list.selectionMode() == QAbstractItemView.SelectionMode.MultiSelection
    for row in range(dlg._list.count()):
        assert not (dlg._list.item(row).flags() & Qt.ItemFlag.ItemIsUserCheckable)

    dlg.show()
    _app.processEvents()
    second = dlg._list.item(1)
    QTest.mouseClick(
        dlg._list.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dlg._list.visualItemRect(second).center(),
    )

    assert dlg.selected_sources() == ["a"]
    dlg.close()


def test_dialog_keeps_duplicate_display_names_distinct_by_path():
    sources = [
        OBBSource(path="/one/duplicate", name="duplicate", level="obb"),
        OBBSource(path="/two/duplicate", name="duplicate", level="obb"),
    ]
    dlg = EscalateSam2Dialog(sources)
    dlg._list.item(1).setSelected(False)

    assert dlg.selected_sources() == ["duplicate"]
    assert dlg.selected_source_paths() == ["/one/duplicate"]


def test_sam2_dialog_tiling_defaults_to_full_frame():
    dlg = EscalateSam2Dialog([OBBSource(name="a", level="obb")])
    assert dlg.tiling_parameters()["tile_fraction"] is None
    assert dlg.selected_sources() == ["a"]  # every eligible source, as before


def test_sam2_dialog_restores_calibrated_fraction_and_split_lists(tmp_path):
    from types import SimpleNamespace

    from tests.test_calibration_frames import OBB, POLY, make_source

    poly = make_source(tmp_path, "poly", {"a": POLY})
    box = make_source(tmp_path, "box", {"a": OBB}, level="obb")
    point = dict(
        tile_fraction=0.1,
        tile_px=200,
        owned_tiles_per_frame=4.0,
        seconds_per_frame=1.0,
        median_iou=0.8,
        p10_iou=0.6,
        fallback_rate=0.0,
        seam_fallback_rate=0.0,
        n_instances=40,
    )
    project = SimpleNamespace(
        project_dir=str(tmp_path),
        geometry_calibration={
            DEFAULT_VARIANT: {"recommended_index": 0, "points": [point]}
        },
        geometry_escalation_settings={},
    )
    dlg = EscalateSam2Dialog([poly, box], project=project, reference_body_px=20.0)
    assert dlg.tiling_parameters()["tile_fraction"] == pytest.approx(0.1)
    assert dlg._selector.calibration_sources() == [poly]
    assert dlg.selected_sources() == ["box"]
    assert dlg._btn_calibrate.isEnabled()
    assert dlg._results.rowCount() == 1


def test_sam2_dialog_persists_split_selection_and_tiling(tmp_path):
    from types import SimpleNamespace

    from tests.test_calibration_frames import OBB, POLY, make_source

    poly = make_source(tmp_path, "poly", {"a": POLY})
    box = make_source(tmp_path, "box", {"a": OBB}, level="obb")
    project = SimpleNamespace(
        project_dir=str(tmp_path),
        geometry_calibration={},
        geometry_escalation_settings={},
    )
    saves = []
    dlg = EscalateSam2Dialog(
        [poly, box],
        project=project,
        reference_body_px=20.0,
        persist_callback=lambda: saves.append(1),
    )
    dlg._tile_fraction.setValue(0.2)
    dlg.accept()
    saved = project.geometry_escalation_settings
    assert saved["tile_fraction"] == pytest.approx(0.2)
    assert saved["escalation_source_names"] == ["box"] and saves
    again = EscalateSam2Dialog([poly, box], project=project)
    assert again.tiling_parameters()["tile_fraction"] == pytest.approx(0.2)
    assert again.selected_sources() == ["box"]


def test_sam2_dialog_without_polygon_sources_cannot_calibrate():
    dlg = EscalateSam2Dialog([OBBSource(name="a", level="obb", path="/nope")])
    assert not dlg._btn_calibrate.isEnabled()
    assert "polygon" in dlg._btn_calibrate.toolTip()


def test_dialog_tiling_parameters_feed_the_escalation_request():
    from types import SimpleNamespace

    from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest

    dlg = EscalateSam2Dialog([OBBSource(name="a", level="obb")], reference_body_px=79.0)
    dlg._tile_fraction.setValue(0.2)
    req = EscalationRequest(
        SimpleNamespace(), ["a"], DEFAULT_VARIANT, **dlg.tiling_parameters()
    )
    assert req.tiling.resolved_tile_px() == 395


def test_dialog_offers_device_choice_defaulting_to_auto():
    dlg = EscalateSam2Dialog([OBBSource(name="a", level="obb")])
    assert dlg.selected_device() == "auto"


def test_sam2_dialog_persists_the_device_with_the_tiling(tmp_path):
    from types import SimpleNamespace

    project = SimpleNamespace(
        project_dir=str(tmp_path),
        geometry_calibration={},
        geometry_escalation_settings={"variant": DEFAULT_VARIANT, "device": "cpu"},
    )
    dlg = EscalateSam2Dialog([OBBSource(name="a", level="obb")], project=project)
    assert dlg.selected_device() == "cpu"
    dlg.accept()
    assert project.geometry_escalation_settings["device"] == "cpu"
