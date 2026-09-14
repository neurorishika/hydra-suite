"""Regression tests for DetectKit dialog numeric-input wheel handling."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialogButtonBox,
    QDoubleSpinBox,
    QSpinBox,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wheel_event() -> QWheelEvent:
    return QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


@pytest.mark.parametrize("spin_box_type", [QSpinBox, QDoubleSpinBox])
def test_detectkit_dialog_wheel_does_not_change_numeric_values(qapp, spin_box_type):
    from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog

    dialog = DetectKitDialog(
        "Wheel guard", buttons=QDialogButtonBox.StandardButton.Close
    )
    spin_box = spin_box_type()
    spin_box.setRange(0, 100)
    spin_box.setValue(50)
    dialog.add_content(spin_box)

    QApplication.sendEvent(spin_box, _wheel_event())

    assert spin_box.value() == 50


def test_detectkit_dialog_wheel_guard_does_not_affect_other_widgets(qapp):
    from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog

    dialog = DetectKitDialog(
        "Wheel guard", buttons=QDialogButtonBox.StandardButton.Close
    )
    unrelated_spin_box = QSpinBox()
    unrelated_spin_box.setRange(0, 100)
    unrelated_spin_box.setValue(50)

    QApplication.sendEvent(unrelated_spin_box, _wheel_event())

    assert unrelated_spin_box.value() == 51
    dialog.deleteLater()
