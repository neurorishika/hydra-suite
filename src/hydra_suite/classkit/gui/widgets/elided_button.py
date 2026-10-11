"""Push button that shrinks with its layout and elides long text."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QPushButton, QSizePolicy, QStyle, QStyleOptionButton


class ElidedButton(QPushButton):
    """A QPushButton whose text ends in "…" instead of widening the layout.

    The full text stays available through ``fullText()`` and the tooltip.
    """

    MIN_WIDTH = 40

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setFullText(text)

    def fullText(self) -> str:
        return self._full_text

    def setFullText(self, text: str) -> None:
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._update_elision()

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(self.MIN_WIDTH, hint.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_elision()

    def _text_width(self) -> int:
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        rect = self.style().subElementRect(QStyle.SE_PushButtonContents, opt, self)
        # Stylesheet padding is not reported by every style; keep a margin.
        return max(0, rect.width() - 4)

    def _update_elision(self) -> None:
        shown = self.fontMetrics().elidedText(
            self._full_text, Qt.ElideRight, self._text_width()
        )
        if shown != super().text():
            super().setText(shown)
