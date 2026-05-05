"""Interactive timeline editor widget for RefineKit.

Renders one horizontal lane per track.  Each contiguous fragment is drawn
as a coloured bar.  The user can:

* **Right-click** a bar → "Split here" (context menu).
* **Click** a bar → select it (highlighted border).
* **Drag** a selected bar to another lane → reassign (if no overlap).
* **Delete** key on a selected bar → mark it deleted.
* **Ctrl-Z** / **Cmd-Z** → undo.

The widget communicates with :class:`TrackEditorModel` for data and
emits ``model_changed`` whenever the fragment set is altered so the
parent dialog can repaint the video preview.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QLabel, QMenu, QScrollArea, QToolTip, QVBoxLayout, QWidget

from hydra_suite.refinekit.core.track_editor_model import (
    TrackEditorModel,
    TrackFragment,
)
from hydra_suite.refinekit.gui.overlay_utils import TAB20_RGB

# Colour palette (RGB) — canonical source for all RefineKit views.
# Import this from other modules to keep colours consistent.
PALETTE_RGB: List[Tuple[int, int, int]] = list(TAB20_RGB)
_DELETED_COLOR = QColor(80, 80, 80, 100)
_SELECTED_PEN = QPen(QColor(255, 255, 255), 2)
_TARGET_LANE_PEN = QPen(QColor(0, 200, 0, 140), 2, Qt.PenStyle.DashLine)

_LABEL_WIDTH = 64
_ROW_HEIGHT = 28
_BAR_MARGIN = 3
_MIN_BAR_PX = 4


# ---------------------------------------------------------------------------
# _TimelineEditorCanvas  (custom-painted)
# ---------------------------------------------------------------------------


class _TimelineEditorCanvas(QWidget):
    """Low-level custom-painted timeline with mouse interaction."""

    model_changed = Signal()  # fragment set was mutated
    frame_cursor_changed = Signal(int)  # user moved the cursor
    fragment_selected = Signal(object)  # TrackFragment or None
    reassign_failed = Signal()  # drag-to-reassign was rejected (overlap)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model: Optional[TrackEditorModel] = None
        self._selected_frag_id: Optional[int] = None
        self._cursor_frame: Optional[int] = None

        # drag state
        self._dragging = False
        self._drag_frag_id: Optional[int] = None
        self._drag_target_track: Optional[int] = None
        self._drag_start_pos: Optional[QPoint] = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(80)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_model(self, model: TrackEditorModel) -> None:
        self._model = model
        self._selected_frag_id = None
        self._update_height()
        self.update()

    def refresh(self) -> None:
        """Re-render after external model change (e.g. undo)."""
        self._update_height()
        self.update()

    def _update_height(self) -> None:
        if self._model:
            rows = len(self._model.visible_tracks)
            self.setMinimumHeight(max(rows * _ROW_HEIGHT + 8, 80))

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _bar_width(self) -> int:
        return max(self.width() - _LABEL_WIDTH, 1)

    def _frame_to_x(self, frame: int) -> int:
        if not self._model:
            return _LABEL_WIDTH
        fs, fe = self._model.frame_range
        span = max(fe - fs, 1)
        return _LABEL_WIDTH + int((frame - fs) / span * self._bar_width())

    def _x_to_frame(self, x: int) -> int:
        if not self._model:
            return 0
        fs, fe = self._model.frame_range
        span = max(fe - fs, 1)
        frac = max(0.0, min((x - _LABEL_WIDTH) / self._bar_width(), 1.0))
        return fs + int(frac * span)

    def _y_to_track(self, y: int) -> Optional[int]:
        if not self._model:
            return None
        row = y // _ROW_HEIGHT
        tracks = self._model.visible_tracks
        if 0 <= row < len(tracks):
            return tracks[row]
        return None

    def _track_to_row(self, track_id: int) -> int:
        if not self._model:
            return 0
        tracks = self._model.visible_tracks
        if track_id in tracks:
            return tracks.index(track_id)
        return 0

    def _frag_rect(self, frag: TrackFragment) -> QRect:
        if not self._model:
            return QRect()
        view_start, view_end = self._model.frame_range
        display_start = max(frag.frame_start, view_start)
        display_end = min(frag.frame_end, view_end)
        row = self._track_to_row(frag.track_id)
        x1 = self._frame_to_x(display_start)
        x2 = self._frame_to_x(display_end + 1)
        y = row * _ROW_HEIGHT + _BAR_MARGIN
        return QRect(x1, y, max(x2 - x1, _MIN_BAR_PX), _ROW_HEIGHT - 2 * _BAR_MARGIN)

    def _hit_fragment(self, pos: QPoint) -> Optional[TrackFragment]:
        """Return the topmost fragment under *pos*, or None."""
        if not self._model:
            return None
        for frag in reversed(self._model.fragments):
            if frag.deleted:
                continue
            if self._frag_rect(frag).contains(pos):
                return frag
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

        if not self._model:
            painter.end()
            return

        tracks = self._model.visible_tracks
        for row, track_id in enumerate(tracks):
            top = row * _ROW_HEIGHT
            painter.setPen(QPen(lane_line, 1))
            painter.drawLine(
                0, top + _ROW_HEIGHT - 1, self.width(), top + _ROW_HEIGHT - 1
            )
            painter.setPen(QPen(text_color, 1))
            painter.drawText(
                6,
                top,
                _LABEL_WIDTH - 12,
                _ROW_HEIGHT,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"T{track_id}",
            )

        for frag in self._model.fragments:
            if frag.track_id not in tracks:
                continue
            rect = self._frag_rect(frag)
            if frag.deleted:
                painter.fillRect(rect, _DELETED_COLOR)
                painter.setPen(
                    QPen(QColor(180, 180, 180, 140), 1, Qt.PenStyle.DashLine)
                )
                painter.drawRect(rect)
                continue

            color = QColor(*PALETTE_RGB[frag.track_id % len(PALETTE_RGB)])
            painter.fillRect(rect, color)
            border = (
                color.lighter(140) if color.lightness() < 160 else color.darker(140)
            )
            painter.setPen(QPen(border, 1))
            painter.drawRect(rect)

            if self._selected_frag_id == frag.frag_id:
                inset = rect.adjusted(1, 1, -1, -1)
                painter.setPen(_SELECTED_PEN)
                painter.drawRect(inset)

        if self._dragging and self._drag_target_track is not None:
            row = self._track_to_row(self._drag_target_track)
            target_rect = QRect(
                _LABEL_WIDTH,
                row * _ROW_HEIGHT + _BAR_MARGIN,
                self.width() - _LABEL_WIDTH,
                _ROW_HEIGHT - 2 * _BAR_MARGIN,
            )
            painter.fillRect(target_rect, QColor(0, 200, 0, 30))
            painter.setPen(_TARGET_LANE_PEN)
            painter.drawRect(target_rect.adjusted(0, 0, -1, -1))

        if self._cursor_frame is not None:
            x = self._frame_to_x(self._cursor_frame)
            painter.setPen(QPen(QColor(255, 170, 0), 2))
            painter.drawLine(x, 0, x, self.height())

        painter.end()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        pos = ev.position().toPoint()
        frag = self._hit_fragment(pos)

        if ev.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(pos, frag)
            return

        if ev.button() == Qt.MouseButton.LeftButton:
            if pos.x() >= _LABEL_WIDTH:
                self._cursor_frame = self._x_to_frame(pos.x())
                self.frame_cursor_changed.emit(self._cursor_frame)

            if frag:
                self._selected_frag_id = frag.frag_id
                self.fragment_selected.emit(frag)
                # Start potential drag
                self._drag_start_pos = pos
                self._drag_frag_id = frag.frag_id
            else:
                self._selected_frag_id = None
                self.fragment_selected.emit(None)
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        pos = ev.position().toPoint()

        # Start drag if moved far enough
        if (
            self._drag_start_pos is not None
            and not self._dragging
            and self._drag_frag_id is not None
        ):
            delta = pos - self._drag_start_pos
            if delta.manhattanLength() > 8:
                self._dragging = True

        if self._dragging:
            self._drag_target_track = self._y_to_track(pos.y())
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # noqa: N802
        if self._dragging and self._drag_frag_id is not None and self._model:
            target = self._drag_target_track
            if target is not None:
                if self._model.reassign(self._drag_frag_id, target):
                    self.model_changed.emit()
                else:
                    self.reassign_failed.emit()
        self._dragging = False
        self._drag_frag_id = None
        self._drag_target_track = None
        self._drag_start_pos = None
        self.update()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos: QPoint, frag: Optional[TrackFragment]) -> None:
        menu = QMenu(self)

        if frag and not frag.deleted:
            frame = self._x_to_frame(pos.x())

            if frag.frame_start < frame <= frag.frame_end:
                split_action = QAction(f"Split T{frag.track_id} at frame {frame}", self)
                split_action.triggered.connect(
                    lambda: self._do_split(frag.frag_id, frame)
                )
                menu.addAction(split_action)

            delete_action = QAction(
                f"Delete fragment T{frag.track_id} [{frag.frame_start}–{frag.frame_end}]",
                self,
            )
            delete_action.triggered.connect(lambda: self._do_delete(frag.frag_id))
            menu.addAction(delete_action)

        if frag and frag.deleted:
            # Allow un-delete via undo
            pass  # Undo is keyboard-only (Ctrl-Z)

        if not menu.isEmpty():
            menu.exec(self.mapToGlobal(pos))

    def _do_split(self, frag_id: int, frame: int) -> None:
        if self._model and self._model.split(frag_id, frame):
            self._update_height()
            self.model_changed.emit()
            self.update()

    def _do_delete(self, frag_id: int) -> None:
        if self._model and self._model.delete(frag_id):
            self.model_changed.emit()
            self.update()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, ev: QKeyEvent) -> None:  # noqa: N802
        if not self._model:
            return super().keyPressEvent(ev)

        # Ctrl-Z / Cmd-Z → undo
        if ev.key() == Qt.Key.Key_Z and (
            ev.modifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        ):
            if self._model.undo():
                self._update_height()
                self.model_changed.emit()
                self.update()
            return

        # Delete / Backspace → delete selected fragment
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_frag_id is not None:
                self._do_delete(self._selected_frag_id)
                self._selected_frag_id = None
            return

        super().keyPressEvent(ev)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_cursor_frame(self, frame: int) -> None:
        """Move the orange cursor line without emitting a signal."""
        self._cursor_frame = frame
        self.update()

    # ------------------------------------------------------------------
    # Tooltip
    # ------------------------------------------------------------------

    def event(self, ev) -> bool:  # noqa: N802
        from PySide6.QtCore import QEvent  # noqa: PLC0415

        if ev.type() == QEvent.Type.ToolTip:
            frag = self._hit_fragment(ev.pos())
            if frag is not None:
                length = frag.frame_end - frag.frame_start + 1
                QToolTip.showText(
                    ev.globalPos(),
                    f"T{frag.track_id}  \u2022  frames {frag.frame_start}\u2013{frag.frame_end}"
                    f"  ({length} frame{'s' if length != 1 else ''})",
                    self,
                )
            else:
                QToolTip.hideText()
            ev.accept()
            return True
        return super().event(ev)


# ---------------------------------------------------------------------------
# TimelineEditorWidget  (scrollable wrapper)
# ---------------------------------------------------------------------------


class TimelineEditorWidget(QWidget):
    """Scrollable wrapper around :class:`_TimelineEditorCanvas`.

    Signals
    -------
    model_changed:
        The fragment model was mutated (split / delete / reassign / undo).
    frame_cursor_changed:
        The user clicked a new frame position on the timeline.
    fragment_selected:
        A fragment was selected (or None for deselect).
    """

    model_changed = Signal()
    frame_cursor_changed = Signal(int)
    fragment_selected = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._canvas = _TimelineEditorCanvas()
        self._canvas.model_changed.connect(self.model_changed)
        self._canvas.frame_cursor_changed.connect(self.frame_cursor_changed)
        self._canvas.fragment_selected.connect(self.fragment_selected)
        self._canvas.reassign_failed.connect(self._on_reassign_failed)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll)

        # Transient error message shown when a drag-reassign is rejected
        self._msg_label = QLabel()
        self._msg_label.setStyleSheet(
            "color: #ff6b6b; font-size: 11px; padding: 2px 6px;"
        )
        self._msg_label.setVisible(False)
        layout.addWidget(self._msg_label)
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(lambda: self._msg_label.setVisible(False))

    def _on_reassign_failed(self) -> None:
        self._msg_label.setText("⚠  Cannot reassign: fragments overlap")
        self._msg_label.setVisible(True)
        self._msg_timer.start(2000)

    def set_model(self, model: TrackEditorModel) -> None:
        self._canvas.set_model(model)

    def refresh(self) -> None:
        self._canvas.refresh()

    def set_cursor_frame(self, frame: int) -> None:
        self._canvas.set_cursor_frame(frame)
