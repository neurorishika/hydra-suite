"""Table of SAM2 geometry-calibration points: one row per tile fraction."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem

_COLUMNS = (
    "Tile size",
    "Encodes/frame",
    "s/frame",
    "Median IoU",
    "p10 IoU",
    "Fell back",
    "Animals",
)
_RECOMMENDED = QColor(46, 125, 50, 90)


def _tile_text(point) -> str:
    if point.tile_fraction is None:
        return "full frame"
    return f"{point.tile_px} px (fraction {point.tile_fraction:g})"


class GeometryCalibrationResults(QTableWidget):
    """Measured points; selecting a row emits ``point_chosen(point)``."""

    point_chosen = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(_COLUMNS), parent)
        self.setHorizontalHeaderLabels(list(_COLUMNS))
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self._points: list = []
        self.itemSelectionChanged.connect(self._emit_choice)

    def points(self) -> list:
        return list(self._points)

    def set_points(self, points, recommended=None) -> None:
        self.blockSignals(True)
        self._points = list(points)
        self.setRowCount(len(self._points))
        for row, p in enumerate(self._points):
            cells = (
                _tile_text(p),
                f"{p.owned_tiles_per_frame:.1f}",
                f"{p.seconds_per_frame:.2f}",
                f"{p.median_iou:.3f}",
                f"{p.p10_iou:.3f}",
                f"{p.fallback_rate:.1%}",
                str(p.n_instances),
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if p is recommended:
                    item.setBackground(QBrush(_RECOMMENDED))
                    item.setToolTip("Recommended")
                self.setItem(row, col, item)
        self.resizeColumnsToContents()
        self.blockSignals(False)

    def _emit_choice(self) -> None:
        rows = {index.row() for index in self.selectedIndexes()}
        if len(rows) == 1:
            self.point_chosen.emit(self._points[rows.pop()])
