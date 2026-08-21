"""ArenaCanvas: the video preview widget that owns arena drawing.

Replaces the previous ``QLabel`` whose event handlers were monkeypatched
from ``MainWindow``. The decisive difference is WHERE the overlay is painted:
the old code painted into the frame's image pixels and scaled the result, so
a 2 px pen was 2 IMAGE pixels (apparent width scaled with zoom) and click
coordinates were only valid at 100% zoom -- which is why zoom had to be
force-disabled while drawing. Here the frame is painted scaled and the
overlay is painted afterwards in WIDGET coordinates, so pen widths are device
pixels and clicks map back through the inverse transform at any zoom.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from hydra_suite.trackerkit.arena_geometry import arena_at_point
from hydra_suite.trackerkit.gui.widgets.arena_style import CLICK_DRAG_THRESHOLD_PX


class ArenaCanvas(QWidget):
    """Frame display plus arena overlay, with an explicit image/viewport map."""

    point_added = Signal(float, float)
    point_removed = Signal()
    arena_clicked = Signal(int)
    pan_delta = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: QImage | None = None
        self._scaled: QPixmap | None = None
        self._zoom = 1.0
        self._shapes: list[dict[str, Any]] = []
        self._points: list[tuple[float, float]] = []
        self._current_arena: int | None = None
        self._drawing = False
        self._press_pos: QPointF | None = None
        self._panning = False
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # -- state ------------------------------------------------------------

    def set_frame(self, image: QImage | None) -> None:
        self._frame = image
        self._rescale()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.1, float(zoom))
        self._rescale()

    def set_shapes(self, shapes: list[dict[str, Any]] | None) -> None:
        self._shapes = list(shapes or [])
        self.update()

    def set_points(self, points: list[tuple[float, float]] | None) -> None:
        self._points = list(points or [])
        self.update()

    def set_current_arena(self, arena_id: int | None) -> None:
        self._current_arena = arena_id
        self.update()

    def set_drawing(self, drawing: bool) -> None:
        self._drawing = bool(drawing)
        self.setCursor(Qt.CrossCursor if drawing else Qt.OpenHandCursor)
        self.setContextMenuPolicy(
            Qt.PreventContextMenu if drawing else Qt.DefaultContextMenu
        )
        self.update()

    def _rescale(self) -> None:
        if self._frame is None:
            self._scaled = None
            return
        width = max(1, int(self._frame.width() * self._zoom))
        height = max(1, int(self._frame.height() * self._zoom))
        self._scaled = QPixmap.fromImage(
            self._frame.scaled(
                width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
        )
        self.setFixedSize(width, height)
        self.update()

    # -- transform --------------------------------------------------------

    def to_image(self, point: QPointF) -> tuple[float, float]:
        """Widget coordinates -> image coordinates."""
        return (point.x() / self._zoom, point.y() / self._zoom)

    def to_viewport(self, x: float, y: float) -> QPointF:
        """Image coordinates -> widget coordinates."""
        return QPointF(float(x) * self._zoom, float(y) * self._zoom)

    @staticmethod
    def _is_click(x0: float, y0: float, x1: float, y1: float) -> bool:
        """Whether a press/release pair was a click rather than a drag.

        One button must serve both marking and panning, so displacement
        decides. Strictly under the threshold, so a gesture exactly at the
        threshold is a drag.
        """
        return (abs(x1 - x0) < CLICK_DRAG_THRESHOLD_PX) and (
            abs(y1 - y0) < CLICK_DRAG_THRESHOLD_PX
        )

    # -- input ------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._press_pos = QPointF(event.position())
        if event.button() == Qt.MiddleButton:
            self._panning = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_pos is None:
            event.accept()
            return
        current = QPointF(event.position())
        if not self._is_click(
            self._press_pos.x(), self._press_pos.y(), current.x(), current.y()
        ):
            self._panning = True
            self.pan_delta.emit(
                int(current.x() - self._press_pos.x()),
                int(current.y() - self._press_pos.y()),
            )
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        press, self._press_pos = self._press_pos, None
        panning, self._panning = self._panning, False
        if press is None:
            event.accept()
            return
        release = QPointF(event.position())
        was_click = self._is_click(press.x(), press.y(), release.x(), release.y())

        if event.button() == Qt.RightButton and self._drawing:
            self.point_removed.emit()
        elif event.button() == Qt.LeftButton and was_click and not panning:
            image_x, image_y = self.to_image(release)
            if self._drawing:
                self.point_added.emit(image_x, image_y)
            else:
                arena_id = arena_at_point(self._shapes, image_x, image_y)
                if arena_id is not None:
                    self.arena_clicked.emit(arena_id)
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self._scaled is not None:
            painter.drawPixmap(0, 0, self._scaled)
        painter.end()
