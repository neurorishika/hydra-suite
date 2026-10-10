"""A row of widgets that wraps onto more rows instead of widening its host."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QGridLayout, QLayout, QWidget


class ReflowRow(QWidget):
    """Lay ``widgets`` out in as many columns as fit the current width.

    All in one row when they fit side by side; otherwise fewer columns, down
    to one per row. The minimum width is the widest single widget, so a
    narrow side panel never scrolls sideways because of this row. Each
    widget keeps its full text. The column count depends only on the width,
    so re-laying out on resize is stable.
    """

    def __init__(self, widgets, parent=None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._widgets = list(widgets)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        # The minimum is ours (the widest widget), not the current column
        # count's: let the row be narrower, then reflow on resize.
        self._grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self._grid.setHorizontalSpacing(spacing)
        self._grid.setVerticalSpacing(spacing)
        self._columns = 0
        self._place(len(self._widgets))

    def _needed_width(self, columns: int) -> int:
        widths = [0] * columns
        for index, widget in enumerate(self._widgets):
            column = index % columns
            widths[column] = max(widths[column], widget.sizeHint().width())
        return sum(widths) + self._grid.horizontalSpacing() * (columns - 1)

    def columns_for(self, width: int) -> int:
        for columns in range(len(self._widgets), 1, -1):
            if self._needed_width(columns) <= width:
                return columns
        return 1

    def _place(self, columns: int) -> None:
        if columns == self._columns:
            return
        self._columns = columns
        for widget in self._widgets:
            self._grid.removeWidget(widget)
        for column in range(len(self._widgets)):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)
        for index, widget in enumerate(self._widgets):
            self._grid.addWidget(widget, index // columns, index % columns)
        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        return QSize(self._needed_width(1), hint.height())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        return QSize(self._needed_width(len(self._widgets)), hint.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._place(self.columns_for(self.width()))
