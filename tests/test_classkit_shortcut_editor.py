from __future__ import annotations

import gc
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication
QKeySequence = QtGui.QKeySequence
Qt = QtCore.Qt

ShortcutEditorDialog = pytest.importorskip(
    "hydra_suite.classkit.gui.dialogs.shortcut_editor"
).ShortcutEditorDialog
HYDRA_DIALOGS = pytest.importorskip("hydra_suite.widgets.dialogs")


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def test_review_shortcut_defaults_use_plus_and_minus(qapp) -> None:
    dialog = ShortcutEditorDialog()
    shortcuts = dialog.get_shortcuts()

    assert shortcuts["Approve review label"] == QKeySequence(Qt.Key.Key_Plus).toString()
    assert shortcuts["Reject review label"] == QKeySequence(Qt.Key.Key_Minus).toString()


def test_shortcut_editor_excludes_mode_shortcuts(qapp) -> None:
    dialog = ShortcutEditorDialog()
    shortcuts = dialog.get_shortcuts()

    assert set(shortcuts) == {
        "Approve review label",
        "Reject review label",
        "Sample next candidates",
        "Previous unlabeled",
        "Next unlabeled",
        "Undo last label (Ctrl+Z)",
    }
    assert "Explore mode" not in shortcuts
    assert "Labeling mode" not in shortcuts
    assert "Review mode" not in shortcuts
    assert "Predictions mode" not in shortcuts


def test_shortcut_editor_uses_shared_dialog_theme(qapp) -> None:
    dialog = ShortcutEditorDialog()
    info_labels = [
        label
        for label in dialog.findChildren(QtWidgets.QLabel)
        if label.text().startswith("Click a row and press")
    ]

    assert dialog.styleSheet() == HYDRA_DIALOGS.HYDRA_DIALOG_STYLE
    assert info_labels
    assert HYDRA_DIALOGS.HYDRA_DIALOG_MUTED_TEXT_COLOR in info_labels[0].styleSheet()
