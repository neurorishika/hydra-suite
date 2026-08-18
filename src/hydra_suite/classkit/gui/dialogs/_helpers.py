"""Private helper classes shared across ClassKit dialogs."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

_BTN_DEL = (
    "QPushButton { background-color:#4a1a1a; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#6b2424; }"
)


class _KeyCapture(QLineEdit):
    """Drop-in replacement for QKeySequenceEdit.

    ``QKeySequenceEdit`` registers with the macOS Text Services Manager (TSM)
    as a text-input client.  On macOS + Python 3.13 + Qt 6 this registration
    races with the TSM UI-server port and produces a SIGBUS crash.

    This widget avoids all native IME / TSM hooks by:
    • staying permanently read-only (no system text-input session),
    • disabling the Qt input-method bridge (``WA_InputMethodEnabled = False``),
    • capturing key presses directly in ``keyPressEvent``.

    Public interface is compatible with ``QKeySequenceEdit``:
    ``setKeySequence(QKeySequence)``, ``keySequence() -> QKeySequence``.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._seq = QKeySequence()
        self._capturing = False
        self.setReadOnly(True)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        self.setPlaceholderText("click → press key")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ── public interface MatchQKeySequenceEdit ────────────────────────────────

    def setKeySequence(self, seq):
        if not isinstance(seq, QKeySequence):
            seq = QKeySequence(seq)
        self._seq = seq
        self._refresh()

    def keySequence(self) -> QKeySequence:
        return self._seq

    # ── internal ──────────────────────────────────────────────────────────────

    def _refresh(self):
        self.setText(self._seq.toString() if self._seq else "")

    def mousePressEvent(self, event) -> None:
        self._capturing = True
        self.setText("⌨  press key…")
        self.setFocus()

    def keyPressEvent(self, event) -> None:
        if not self._capturing:
            super().keyPressEvent(event)
            return
        key = event.key()
        # Ignore bare modifier presses
        if key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
            Qt.Key.Key_unknown,
        ):
            return
        self._seq = QKeySequence(event.modifiers().value | int(key))
        self._capturing = False
        self._refresh()

    def focusOutEvent(self, event) -> None:
        if self._capturing:
            self._capturing = False
            self._refresh()
        super().focusOutEvent(event)


class _LabelRow(QWidget):
    """A single label row: name QLineEdit + QKeySequenceEdit for shortcut."""

    def __init__(
        self,
        label: str = "",
        shortcut: str = "",
        parent=None,
        origin: str | None = None,
    ) -> None:
        super().__init__(parent)
        # Label this row carried when the dialog opened (None for a freshly
        # added row). Editing the text is then a rename of *origin*, which lets
        # stored labels be migrated instead of pruned.
        self.origin = origin
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._name_edit = QLineEdit(label)
        self._name_edit.setPlaceholderText("label name")
        self._name_edit.setMinimumWidth(120)
        row.addWidget(self._name_edit, 2)

        key_lbl = QLabel("key:")
        key_lbl.setStyleSheet("color:#ffffff; font-size:11px;")
        key_lbl.setFixedWidth(28)
        row.addWidget(key_lbl)

        self._key_edit = _KeyCapture()
        if shortcut:
            try:
                self._key_edit.setKeySequence(QKeySequence(shortcut))
            except Exception:
                pass
        self._key_edit.setFixedWidth(110)
        self._key_edit.setToolTip(
            "Press any key: letters, digits, arrows, symbols (e.g. ↑ ↓ ← →, Space, +, -, …)"
        )
        row.addWidget(self._key_edit)

        self._btn_del = QPushButton("✕")
        self._btn_del.setFixedSize(26, 26)
        self._btn_del.setStyleSheet(_BTN_DEL)
        self._btn_del.setToolTip("Remove this label")
        row.addWidget(self._btn_del)

    def label(self) -> str:
        return self._name_edit.text().strip()

    def shortcut(self) -> str:
        ks = self._key_edit.keySequence()
        return ks.toString() if ks else ""

    def delete_button(self) -> QPushButton:
        return self._btn_del


class _SchemeWrapper:
    """Wraps a raw scheme dict to satisfy the ``scheme.to_dict()`` contract."""

    def __init__(self, d: dict) -> None:
        self._d = d

    def to_dict(self) -> dict:
        return self._d

    @property
    def factors(self):
        return self._d.get("factors", [])
