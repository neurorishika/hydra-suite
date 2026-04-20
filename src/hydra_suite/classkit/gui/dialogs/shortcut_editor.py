"""ShortcutEditorDialog — edit keyboard shortcuts for ClassKit global actions."""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.classkit.gui.dialogs._helpers import _KeyCapture
from hydra_suite.widgets.dialogs import (
    HYDRA_DIALOG_MUTED_TEXT_COLOR,
    HYDRA_DIALOG_STYLE,
)

_PLUS_KEY = QKeySequence(Qt.Key.Key_Plus).toString()
_MINUS_KEY = QKeySequence(Qt.Key.Key_Minus).toString()


class ShortcutEditorDialog(QDialog):
    """Edit keyboard shortcut assignments for non-label global actions."""

    DEFAULT_SHORTCUTS = [
        ("Approve review label", _PLUS_KEY),
        ("Reject review label", _MINUS_KEY),
        ("Sample next candidates", "Space"),
        ("Previous unlabeled", "Left"),
        ("Next unlabeled", "Right"),
        ("Undo last label (Ctrl+Z)", "Ctrl+Z"),
    ]

    def __init__(self, current: Optional[dict] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(500)
        self.setStyleSheet(HYDRA_DIALOG_STYLE)

        self._shortcuts: dict = {name: seq for name, seq in self.DEFAULT_SHORTCUTS}
        if current:
            self._shortcuts.update(current)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hdr = QLabel("<b>Global Labeling Shortcuts</b>")
        layout.addWidget(hdr)

        info = QLabel(
            "Click a row and press the desired key combination.  "
            "All key types are supported: letters, digits, "
            "<b>arrow keys</b> (\u2191 \u2193 \u2190 \u2192), <b>symbols</b> (+  \u2212  Space  Tab \u2026), "
            "and modifier combos (Ctrl+Z, Shift+Left, \u2026).  "
            "Label-specific shortcuts are defined in the <b>Class Scheme Editor</b>."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{HYDRA_DIALOG_MUTED_TEXT_COLOR}; font-size:11px;")
        layout.addWidget(info)

        form = QFormLayout()
        form.setSpacing(8)
        self._key_edits: dict = {}
        for name, default_key in self.DEFAULT_SHORTCUTS:
            current_key = self._shortcuts.get(name, default_key)
            edit = _KeyCapture()
            try:
                edit.setKeySequence(QKeySequence(current_key))
            except Exception:
                pass
            self._key_edits[name] = edit
            form.addRow(QLabel(name), edit)
        layout.addLayout(form)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setStyleSheet(
            "QPushButton { background-color:#3e3e42; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:6px 14px; }"
            "QPushButton:hover { background-color:#555558; }"
        )
        btn_reset.clicked.connect(self._reset_defaults)
        layout.addWidget(btn_reset)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reset_defaults(self):
        for name, default_key in self.DEFAULT_SHORTCUTS:
            edit = self._key_edits.get(name)
            if edit is not None:
                try:
                    edit.setKeySequence(QKeySequence(default_key))
                except Exception:
                    pass

    def get_shortcuts(self) -> dict:
        out = {}
        for name, edit in self._key_edits.items():
            ks = edit.keySequence()
            out[name] = ks.toString() if ks else self._shortcuts.get(name, "")
        return out
