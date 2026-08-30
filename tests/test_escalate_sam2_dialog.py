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
