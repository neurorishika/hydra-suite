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

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from hydra_suite.trackerkit.arena_geometry import arena_at_point, shape_centroid
from hydra_suite.trackerkit.gui.widgets.arena_style import (
    CLICK_DRAG_THRESHOLD_PX,
    TEXT_ALPHA,
    VEIL_ALPHA,
    frame_palette,
    glyph_size_px,
    line_width_px,
)


def paint_arena_number(
    painter: QPainter,
    text: str,
    center: QPointF,
    size_px: int,
    glyph_rgb: tuple[int, int, int],
    halo_rgb: tuple[int, int, int],
    alpha: float,
) -> None:
    """Draw a haloed arena number, composited ONCE at *alpha*.

    The glyph and its halo are rendered into an offscreen ARGB layer at full
    opacity and that layer is composited in one pass. Stroking and filling
    directly at partial alpha would composite twice where the halo underlies
    the glyph, letting the halo bleed through the glyph edge and making the
    number look doubled.
    """
    font = QFont()
    font.setPixelSize(int(size_px))
    font.setBold(True)

    path = QPainterPath()
    path.addText(0.0, 0.0, font, text)
    bounds = path.boundingRect()
    path.translate(-bounds.center().x(), -bounds.center().y())

    stroker = QPainterPathStroker()
    stroker.setWidth(max(2.0, size_px * 0.18))
    halo_path = stroker.createStroke(path)

    pad = int(max(4.0, size_px * 0.5))
    layer_rect = path.boundingRect().united(halo_path.boundingRect())
    layer = QImage(
        int(layer_rect.width()) + 2 * pad,
        int(layer_rect.height()) + 2 * pad,
        QImage.Format_ARGB32_Premultiplied,
    )
    layer.fill(Qt.transparent)

    layer_painter = QPainter(layer)
    layer_painter.setRenderHint(QPainter.Antialiasing, True)
    layer_painter.translate(
        pad - layer_rect.left(),
        pad - layer_rect.top(),
    )
    layer_painter.fillPath(halo_path, QBrush(QColor(*halo_rgb)))
    layer_painter.fillPath(path, QBrush(QColor(*glyph_rgb)))
    layer_painter.end()

    painter.save()
    painter.setOpacity(float(alpha))
    painter.drawImage(
        QPointF(
            center.x() - layer.width() / 2.0,
            center.y() - layer.height() / 2.0,
        ),
        layer,
    )
    painter.restore()


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
        self._luminance: float | None = None
        self._zoom = 1.0
        self._shapes: list[dict[str, Any]] = []
        self._points: list[tuple[float, float]] = []
        self._current_arena: int | None = None
        self._drawing = False
        self._press_pos: QPointF | None = None
        self._panning = False
        self._toast_text: str | None = None
        self._input_paused = False
        self._preview_shape: dict[str, Any] | None = None
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    # -- state ------------------------------------------------------------

    def set_frame(self, image: QImage | None) -> None:
        self._frame = image
        self._luminance = None
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

    def set_preview_shape(self, shape: dict[str, Any] | None) -> None:
        """The in-progress shape (fitted circle, or polygon-so-far), or None."""
        self._preview_shape = shape
        self.update()

    def is_input_paused(self) -> bool:
        """Whether a toast is showing and input should be ignored."""
        return self._input_paused

    def show_toast(self, text: str, duration_ms: int = 3000) -> None:
        """Show a small, self-dismissing overlay message without touching the
        displayed frame. Input is ignored for the duration so a stray click
        during the message can't be misinterpreted."""
        self._toast_text = text
        self._input_paused = True
        self.update()
        QTimer.singleShot(duration_ms, self._clear_toast)

    def _clear_toast(self) -> None:
        self._toast_text = None
        self._input_paused = False
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
        if self._input_paused:
            event.ignore()
            return
        self._press_pos = QPointF(event.position())
        if event.button() == Qt.MiddleButton:
            self._panning = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._input_paused:
            event.ignore()
            return
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
        if self._input_paused:
            event.ignore()
            return
        press, self._press_pos = self._press_pos, None
        panning, self._panning = self._panning, False
        if press is None:
            event.accept()
            return
        release = QPointF(event.position())
        was_click = self._is_click(press.x(), press.y(), release.x(), release.y())

        if (
            event.button() == Qt.RightButton
            and self._drawing
            and was_click
            and not panning
        ):
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
        self.render_overlay(painter)
        painter.end()

    # -- overlay ------------------------------------------------------------

    def mean_luminance(self) -> float:
        """Mean luminance of the base frame, 0.0-1.0. Cached per frame."""
        if self._frame is None:
            return 0.5
        if self._luminance is None:
            small = self._frame.scaled(
                64, 64, Qt.IgnoreAspectRatio, Qt.FastTransformation
            ).convertToFormat(QImage.Format_Grayscale8)
            buffer = np.frombuffer(
                small.constBits(), dtype=np.uint8, count=small.sizeInBytes()
            )
            self._luminance = float(buffer.mean()) / 255.0
        return self._luminance

    def _palette(self):
        return frame_palette(self.mean_luminance())

    def _line_width(self) -> int:
        """Device-pixel outline width -- independent of zoom by construction."""
        return line_width_px(min(self.parentWidth(), self.parentHeight()))

    def parentWidth(self) -> int:
        parent = self.parentWidget()
        return parent.width() if parent is not None else 800

    def parentHeight(self) -> int:
        parent = self.parentWidget()
        return parent.height() if parent is not None else 600

    def _outline_width_for(self, arena_id: int) -> int:
        base = self._line_width()
        return base * 2 if arena_id == self._current_arena else base

    def _shape_path(self, shape: dict[str, Any]) -> QPainterPath:
        """The shape as a viewport-space path."""
        path = QPainterPath()
        if shape.get("type") == "circle":
            cx, cy, radius = (float(v) for v in shape["params"])
            top_left = self.to_viewport(cx - radius, cy - radius)
            path.addEllipse(
                QRectF(
                    top_left.x(),
                    top_left.y(),
                    2.0 * radius * self._zoom,
                    2.0 * radius * self._zoom,
                )
            )
        else:
            polygon = QPolygonF([self.to_viewport(x, y) for x, y in shape["params"]])
            path.addPolygon(polygon)
            path.closeSubpath()
        return path

    def render_overlay(self, painter: QPainter) -> None:
        """Paint veil, outlines, in-progress points and arena numbers.

        Everything here is in WIDGET coordinates: pen widths and glyph sizes
        are device pixels, so apparent size does not change with zoom.
        """
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self._palette()

        include = [s for s in self._shapes if s.get("mode", "include") == "include"]
        exclude = [s for s in self._shapes if s.get("mode", "include") == "exclude"]

        # Veil: inside the include region, minus every exclude hole.
        if include:
            veil_path = QPainterPath()
            for shape in include:
                veil_path = veil_path.united(self._shape_path(shape))
            for shape in exclude:
                veil_path = veil_path.subtracted(self._shape_path(shape))
            painter.save()
            painter.setOpacity(VEIL_ALPHA)
            painter.fillPath(veil_path, QBrush(QColor(*palette.veil)))
            painter.restore()

        for shape in self._shapes:
            is_include = shape.get("mode", "include") == "include"
            colour = palette.line_include if is_include else palette.line_exclude
            width = self._outline_width_for(int(shape.get("arena_id", 0)))
            painter.setPen(QPen(QColor(*colour), width))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._shape_path(shape))

        # In-progress points and their preview outline.
        if self._points:
            preview = QColor(*palette.line_preview)
            painter.setPen(QPen(preview, self._line_width() * 2))
            for x, y in self._points:
                painter.drawPoint(self.to_viewport(x, y))

        if self._preview_shape is not None:
            painter.setPen(QPen(QColor(*palette.line_preview), self._line_width()))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._shape_path(self._preview_shape))

        for arena_id in sorted({int(s.get("arena_id", 0)) for s in include}):
            members = [s for s in include if int(s.get("arena_id", 0)) == arena_id]
            centroids = [shape_centroid(s) for s in members]
            center_x = sum(c[0] for c in centroids) / len(centroids)
            center_y = sum(c[1] for c in centroids) / len(centroids)
            box = self._shape_path(members[0]).boundingRect()
            paint_arena_number(
                painter,
                str(arena_id + 1),
                self.to_viewport(center_x, center_y),
                glyph_size_px(min(box.width(), box.height()) / 2.0),
                palette.glyph,
                palette.halo,
                TEXT_ALPHA,
            )

        if self._toast_text:
            painter.save()
            font = painter.font()
            font.setPixelSize(16)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(self._toast_text)
            pad_x, pad_y = 24, 14
            banner_w = text_rect.width() + 2 * pad_x
            banner_h = text_rect.height() + 2 * pad_y
            banner_x = (self.width() - banner_w) / 2.0
            banner_y = 24
            banner_rect = QRectF(banner_x, banner_y, banner_w, banner_h)
            painter.setOpacity(0.85)
            painter.setBrush(QBrush(QColor(20, 20, 20)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(banner_rect, 8, 8)
            painter.setOpacity(1.0)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(banner_rect, Qt.AlignCenter, self._toast_text)
            painter.restore()
