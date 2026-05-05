"""Per-animal timeline panel for RefineKit.

Draws one horizontal bar per trajectory with a label column on the left.
Clicking a bar emits ``split_requested(track_id, frame)``. Dragging a bar to
another lane requests a whole-trajectory reassignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QLabel, QMenu, QScrollArea, QVBoxLayout, QWidget

from hydra_suite.refinekit.gui.overlay_utils import TAB20_RGB

# Colour palette (RGB) — canonical source for RefineKit overlays.
_PALETTE_RGB = list(TAB20_RGB)

_LABEL_WIDTH = 60
_DEFAULT_ROW_HEIGHT = 22
_BAR_MARGIN = 2
_MAX_MANUAL_REGION = 300  # max frames selectable for manual review


# ---------------------------------------------------------------------------
# _TimelineCanvas
# ---------------------------------------------------------------------------


class _TimelineCanvas(QWidget):
    """Custom-painted widget showing one horizontal bar per track."""

    split_at = Signal(int, int)  # (track_id, frame)
    region_edit_requested = Signal(int, int)  # (frame_start, frame_end)
    track_move_requested = Signal(int, int)  # (source_track_id, target_track_id)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tracks: Dict[int, Tuple[int, int]] = {}
        self._track_order: List[int] = []
        self._frame_start: int = 0
        self._frame_end: int = 0
        self._row_height: int = _DEFAULT_ROW_HEIGHT
        self._highlight_range: Optional[Tuple[int, int]] = None

        # Right-click drag selection state
        self._sel_start_x: Optional[int] = None
        self._sel_end_x: Optional[int] = None
        self._is_right_dragging: bool = False

        # Left-drag track move state
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_track_id: Optional[int] = None
        self._drag_target_track: Optional[int] = None
        self._is_left_dragging: bool = False

        self.setMouseTracking(True)
        self.setMinimumHeight(50)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_tracks(
        self,
        tracks: Dict[int, Tuple[int, int]],
        frame_start: int,
        frame_end: int,
    ) -> None:
        """Populate the panel with track presence data and set the total frame count for coordinate mapping."""
        self._tracks = dict(tracks)
        self._track_order = sorted(tracks.keys())
        self._frame_start = frame_start
        self._frame_end = max(frame_start, frame_end)
        self._update_size()
        self.update()

    def _update_size(self) -> None:
        self.setMinimumHeight(max(len(self._track_order) * self._row_height + 4, 50))

    def set_highlight_range(self, frame_range: Optional[Tuple[int, int]]) -> None:
        """Update the highlighted frame range and repaint the widget."""
        self._highlight_range = frame_range
        self.update()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _bar_area_width(self) -> int:
        return max(self.width() - _LABEL_WIDTH, 1)

    def _frame_to_x(self, frame: int) -> int:
        span = max(self._frame_end - self._frame_start, 1)
        return _LABEL_WIDTH + int(
            (frame - self._frame_start) / span * self._bar_area_width()
        )

    def _x_to_frame(self, x: int) -> int:
        bar_w = self._bar_area_width()
        frac = max(0.0, min((x - _LABEL_WIDTH) / bar_w, 1.0))
        span = max(self._frame_end - self._frame_start, 1)
        return self._frame_start + int(frac * span)

    def _y_to_row(self, y: int) -> int:
        return y // self._row_height

    def _track_bar_rect(self, track_id: int) -> Optional[Tuple[int, int, int, int]]:
        if track_id not in self._tracks:
            return None
        row = self._track_order.index(track_id)
        fmin, fmax = self._tracks[track_id]
        x1 = self._frame_to_x(fmin)
        x2 = self._frame_to_x(fmax + 1)
        bar_height = max(self._row_height - 2 * _BAR_MARGIN, 6)
        bar_y = row * self._row_height + _BAR_MARGIN
        bar_width = max(x2 - x1, 3)
        return (x1, bar_y, bar_width, bar_height)

    def _hit_track_bar(self, pos: QPoint) -> Optional[int]:
        if pos.x() < _LABEL_WIDTH:
            return None
        row = self._y_to_row(pos.y())
        if not (0 <= row < len(self._track_order)):
            return None
        tid = self._track_order[row]
        rect = self._track_bar_rect(tid)
        if rect is None:
            return None
        x, y, width, height = rect
        if x <= pos.x() <= x + width and y <= pos.y() <= y + height:
            return tid
        return None

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        base = self.palette().color(self.backgroundRole())
        label_bg = base.darker(135)
        lane_bg = base.lighter(108) if base.lightness() < 128 else base.darker(108)
        lane_line = base.lighter(120) if base.lightness() < 128 else base.darker(120)
        text_color = self.palette().color(self.foregroundRole())

        painter.fillRect(self.rect(), base)
        painter.fillRect(0, 0, _LABEL_WIDTH, self.height(), label_bg)
        painter.fillRect(
            _LABEL_WIDTH,
            0,
            self.width() - _LABEL_WIDTH,
            self.height(),
            lane_bg,
        )
        painter.setPen(QPen(lane_line, 1))
        painter.drawLine(_LABEL_WIDTH, 0, _LABEL_WIDTH, self.height())

        if self._highlight_range is not None:
            h_start, h_end = self._highlight_range
            hx1 = self._frame_to_x(h_start)
            hx2 = self._frame_to_x(h_end + 1)
            painter.fillRect(
                hx1,
                0,
                max(hx2 - hx1, 1),
                self.height(),
                QColor(255, 215, 0, 45),
            )

        painter.setPen(QPen(text_color, 1))
        for row, tid in enumerate(self._track_order):
            top = row * self._row_height
            painter.drawLine(
                0,
                top + self._row_height - 1,
                self.width(),
                top + self._row_height - 1,
            )
            painter.drawText(
                6,
                top,
                _LABEL_WIDTH - 12,
                self._row_height,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"T{tid}",
            )

            fmin, fmax = self._tracks[tid]
            x1, bar_y, bar_width, bar_height = self._track_bar_rect(tid)
            color = QColor(*_PALETTE_RGB[tid % len(_PALETTE_RGB)])
            painter.fillRect(x1, bar_y, bar_width, bar_height, color)
            painter.setPen(QPen(color.lighter(140), 1))
            painter.drawRect(x1, bar_y, bar_width, bar_height)

        if self._is_left_dragging and self._drag_target_track is not None:
            row = self._track_order.index(self._drag_target_track)
            y = row * self._row_height + _BAR_MARGIN
            h = max(self._row_height - 2 * _BAR_MARGIN, 6)
            painter.fillRect(
                _LABEL_WIDTH,
                y,
                self.width() - _LABEL_WIDTH,
                h,
                QColor(0, 200, 0, 30),
            )
            painter.setPen(QPen(QColor(0, 200, 0, 140), 2, Qt.PenStyle.DashLine))
            painter.drawRect(
                _LABEL_WIDTH,
                y,
                self.width() - _LABEL_WIDTH - 1,
                h - 1,
            )

        if self._sel_start_x is not None and self._sel_end_x is not None:
            sx1 = min(self._sel_start_x, self._sel_end_x)
            sx2 = max(self._sel_start_x, self._sel_end_x)
            painter.fillRect(
                sx1,
                0,
                max(sx2 - sx1, 1),
                self.height(),
                QColor(64, 160, 255, 45),
            )
            painter.setPen(QPen(QColor(64, 160, 255, 150), 1, Qt.PenStyle.DashLine))
            painter.drawRect(sx1, 0, max(sx2 - sx1, 1), self.height() - 1)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit a split-at signal on left-click over a track bar, or begin a right-drag range selection."""
        pos = event.position().toPoint()

        if event.button() == Qt.MouseButton.LeftButton:
            tid = self._hit_track_bar(pos)
            if tid is not None:
                self._drag_start_pos = pos
                self._drag_track_id = tid
                self._drag_target_track = tid
                self._is_left_dragging = False
                self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            if pos.x() >= _LABEL_WIDTH:
                self._sel_start_x = pos.x()
                self._sel_end_x = pos.x()
                self._is_right_dragging = True
                self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start_pos is not None and self._drag_track_id is not None:
            delta = event.position().toPoint() - self._drag_start_pos
            if not self._is_left_dragging and delta.manhattanLength() > 8:
                self._is_left_dragging = True
            if self._is_left_dragging:
                row = self._y_to_row(event.position().toPoint().y())
                if 0 <= row < len(self._track_order):
                    self._drag_target_track = self._track_order[row]
                else:
                    self._drag_target_track = None
                self.update()

        if self._is_right_dragging:
            x = max(_LABEL_WIDTH, event.position().toPoint().x())
            self._sel_end_x = x
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._drag_track_id is not None
        ):
            source_track = self._drag_track_id
            target_track = self._drag_target_track
            click_track = self._hit_track_bar(event.position().toPoint())
            if self._is_left_dragging:
                if target_track is not None and target_track != source_track:
                    self.track_move_requested.emit(source_track, target_track)
            elif click_track == source_track:
                frame = self._x_to_frame(event.position().toPoint().x())
                self.split_at.emit(source_track, frame)

            self._drag_start_pos = None
            self._drag_track_id = None
            self._drag_target_track = None
            self._is_left_dragging = False
            self.update()

        if event.button() == Qt.MouseButton.RightButton and self._is_right_dragging:
            self._is_right_dragging = False
            sx1 = min(self._sel_start_x or 0, self._sel_end_x or 0)
            sx2 = max(self._sel_start_x or 0, self._sel_end_x or 0)
            f1 = self._x_to_frame(sx1)
            f2 = self._x_to_frame(sx2)
            self._sel_start_x = None
            self._sel_end_x = None
            self.update()
            if f2 - f1 >= 2:
                self._show_region_menu(event.position().toPoint(), f1, f2)
        super().mouseReleaseEvent(event)

    def _show_region_menu(self, pos: QPoint, f1: int, f2: int) -> None:
        span = f2 - f1
        label = f"Review region  [{f1}\u2013{f2}]  ({span} frames)"
        if span > _MAX_MANUAL_REGION:
            label += f"  \u26a0 will be capped to {_MAX_MANUAL_REGION}"
        menu = QMenu(self)
        act = QAction(label, self)
        act.triggered.connect(lambda: self.region_edit_requested.emit(f1, f2))
        menu.addAction(act)
        menu.exec(self.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Wheel — Ctrl+scroll scales row height
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 2 if delta > 0 else -2
            new_h = max(12, min(80, self._row_height + step))
            if new_h != self._row_height:
                self._row_height = new_h
                self._update_size()
                self.update()
            event.accept()
        else:
            super().wheelEvent(event)


# ---------------------------------------------------------------------------
# TimelinePanelWidget
# ---------------------------------------------------------------------------


class TimelinePanelWidget(QWidget):
    """Per-animal timeline bars in a scrollable container."""

    split_requested = Signal(int, int)  # (track_id, frame)
    region_edit_requested = Signal(int, int)  # (frame_start, frame_end)
    track_move_requested = Signal(int, int)  # (source_track_id, target_track_id)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._canvas = _TimelineCanvas()
        self._canvas.split_at.connect(self.split_requested)
        self._canvas.region_edit_requested.connect(self.region_edit_requested)
        self._canvas.track_move_requested.connect(self.track_move_requested)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll)

        hint = QLabel(
            "Drag a bar to move its trajectory to another lane when non-overlapping"
            "  \u00b7  Right-click drag to select a region for manual review"
            "  \u00b7  Ctrl+scroll to resize rows"
        )
        hint.setStyleSheet("color: #555555; font-size: 10px; padding: 1px 4px;")
        layout.addWidget(hint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_trajectories(self, df: pd.DataFrame) -> None:
        """Compute per-track frame ranges from a trajectory DataFrame."""
        tracks: Dict[int, Tuple[int, int]] = {}
        if df.empty:
            self._canvas.set_tracks({}, 0, 0)
            return

        frame_start = int(df["FrameID"].min())
        frame_end = int(df["FrameID"].max())

        for tid, grp in df.groupby("TrajectoryID"):
            fmin = int(grp["FrameID"].min())
            fmax = int(grp["FrameID"].max())
            tracks[int(tid)] = (fmin, fmax)

        self._canvas.set_tracks(tracks, frame_start, frame_end)

    def highlight_event(self, event) -> None:
        """Highlight a swap suspicion event's frame range."""
        if event is not None and hasattr(event, "frame_range"):
            self._canvas.set_highlight_range(event.frame_range)
        else:
            self._canvas.set_highlight_range(None)
