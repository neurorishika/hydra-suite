"""The shared two-list source selector: calibrate on polygons, escalate any."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.widgets.calibration_source_selector import (  # noqa: E402
    CalibrationSourceSelector,
)
from tests.test_calibration_frames import OBB, POLY, make_source  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _sources(tmp_path):
    return [
        make_source(tmp_path, "poly", {"a": POLY}),
        make_source(tmp_path, "box", {"a": OBB}, level="obb"),
    ]


def test_calibrate_lists_only_polygon_sources_all_selected(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([poly, box])
    assert sel.calibration_list.count() == 1
    assert sel.calibration_sources() == [poly]
    assert sel.escalation_list.count() == 2 and sel.escalation_sources() == []


def test_restore_legacy_names_and_state_round_trip(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([poly, box])
    sel.restore(None, ["box"], None)
    assert sel.escalation_sources() == [box]
    st = sel.state()
    assert st["escalation_source_names"] == ["box"]
    sel2 = CalibrationSourceSelector([poly, box])
    sel2.restore(st["calibration_source_paths"], None, st["escalation_source_paths"])
    assert sel2.escalation_sources() == [box] and sel2.calibration_sources() == [poly]


def test_restore_can_deselect_every_calibration_source(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([poly, box])
    sel.restore([], None, None)
    assert sel.calibration_sources() == []


def test_ineligible_escalation_rows_are_disabled(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector(
        [poly, box],
        escalation_eligible=lambda s: s.name != "poly",
        ineligible_reason="Already polygon",
    )
    sel.restore(None, ["poly", "box"], None)
    assert sel.escalation_sources() == [box]
    sel.select_escalation_source("poly")
    assert sel.escalation_sources() == []


def test_empty_state_when_no_polygon_sources(tmp_path):
    _poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([box])
    assert not sel.has_calibration_sources()
    assert not sel.calibration_empty_label.isHidden()


def test_selection_changes_emit_signals(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([poly, box])
    seen = []
    sel.escalation_changed.connect(lambda: seen.append("e"))
    sel.calibration_changed.connect(lambda: seen.append("c"))
    sel.escalation_list.item(1).setSelected(True)
    sel.calibration_list.item(0).setSelected(False)
    assert "e" in seen and "c" in seen


def test_scale_warning_only_when_sizes_differ(tmp_path):
    from hydra_suite.detectkit.gui.widgets.calibration_source_selector import (
        scale_warning_text,
    )

    small = make_source(tmp_path, "small", {"a": POLY})  # ~25 px across
    big_obb = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"  # ~80 px across
    big = make_source(tmp_path, "big", {"a": big_obb}, level="obb")
    same = make_source(tmp_path, "same", {"a": POLY}, level="obb")
    cache: dict = {}
    assert "differ in size" in scale_warning_text(cache, [small], [big])
    assert scale_warning_text(cache, [small], [same]) == ""
    assert scale_warning_text(cache, [small], []) == ""
