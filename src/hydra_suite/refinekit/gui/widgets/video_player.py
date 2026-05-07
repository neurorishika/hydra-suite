"""Video player widget for RefineKit.

Provides:
* InteractiveCanvas viewport – drag-pan, Ctrl+wheel zoom, pinch-zoom,
  double-click fit, zoom slider (matching MAT main window behaviour).
* Play / Pause with a smooth sequential-decode playback engine.
  A dedicated QThread reads frames in order; on pause or seek the thread
  is replaced so there is never a stale sequential position.
* Configurable debounced seek for scrubbing responsiveness.
* Size-normalised overlay: marker radius, line width and font scale all
  derive from min(frame_width, frame_height) so they look consistent
  across camera resolutions.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Set

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.refinekit.gui.overlay_utils import (
    build_detection_track_map,
    draw_detections,
    load_frame_detections,
    review_overlay_style_from_shape,
    tab20_bgr,
)
from hydra_suite.refinekit.gui.widgets.interactive_canvas import InteractiveCanvas

# ---------------------------------------------------------------------------
# Cache / prefetch constants
# ---------------------------------------------------------------------------

_MAX_CACHE = 300
_PREFETCH_AHEAD = 80
_PREFETCH_BEHIND = 20
_SEEK_DEBOUNCE_MS = 60
_PLAYBACK_FPS = 25


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def draw_overlay(
    img: np.ndarray,
    df_by_frame: dict,
    frame_idx: int,
    highlight_ids: Set[int],
    scale_factor: float = 1.0,
    frame_dets=None,
) -> None:
    """Draw trajectory circles and ID labels, normalised to image size.

    *scale_factor* is passed straight to :func:`_overlay_scale`.
    """
    rows = df_by_frame.get(frame_idx)
    if rows is None:
        return
    font_scale, radius, thickness, outline_th = review_overlay_style_from_shape(
        img.shape,
        scale_factor,
    )
    highlight_radius = int(radius * 1.5)

    if frame_dets is not None and "DetectionID" in rows.columns:
        det_colors = {}
        det_map = build_detection_track_map(rows, frame_idx, frame_idx)
        for det_idx, track_id in det_map.get(frame_idx, {}).items():
            det_colors[det_idx] = tab20_bgr(track_id)
        draw_detections(
            img,
            frame_dets,
            frame_idx,
            0,
            0,
            det_colors,
            thickness=thickness,
        )

    for _, row in rows.iterrows():
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            continue
        tid = int(row["TrajectoryID"])
        cx = int(round(row["X"]))
        cy = int(round(row["Y"]))
        colour = tab20_bgr(tid)
        cv2.circle(img, (cx, cy), radius, colour, outline_th)
        cv2.putText(
            img,
            str(tid),
            (cx + radius + 2, cy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            outline_th,
            cv2.LINE_AA,
        )
        if tid in highlight_ids:
            cv2.circle(img, (cx, cy), highlight_radius, colour, outline_th)


# ---------------------------------------------------------------------------
# Background prefetch thread
# ---------------------------------------------------------------------------


class _PrefetchThread(QThread):
    """Sequentially reads frames from *start* to *end* and emits each."""

    frame_decoded = Signal(int, object)

    def __init__(self, path: str, start: int, end: int, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._start = start
        self._end = end

    def run(self) -> None:
        """Read frames from ``start`` to ``end`` sequentially and emit each via ``frame_decoded``."""
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
        idx = self._start
        while idx <= self._end:
            if self.isInterruptionRequested():
                break
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_decoded.emit(idx, frame)
            idx += 1
        cap.release()


# ---------------------------------------------------------------------------
# Playback thread (sequential, no per-frame seek)
# ---------------------------------------------------------------------------


class _PlaybackThread(QThread):
    """Reads frames sequentially at playback speed and emits each."""

    frame_ready = Signal(int, object)
    finished_playback = Signal()

    def __init__(
        self, path: str, start: int, total: int, fps: float, parent=None
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._start = start
        self._total = total
        self._interval_ms = max(1, int(1000.0 / max(fps, 1.0)))

    def run(self) -> None:
        """Play frames sequentially at the target FPS, emitting each via ``frame_ready`` with a throttled sleep."""
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
        idx = self._start
        while idx < self._total:
            if self.isInterruptionRequested():
                break
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_ready.emit(idx, frame)
            idx += 1
            self.msleep(self._interval_ms)
        cap.release()
        if not self.isInterruptionRequested():
            self.finished_playback.emit()


# ---------------------------------------------------------------------------
# Public widget
# ---------------------------------------------------------------------------


class VideoPlayerWidget(QWidget):
    """Interactive video player with trajectory overlay and play/pause."""

    frame_changed = Signal(int)
    frame_axis_margins_changed = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._video_path: Optional[str] = None
        self._total_frames: int = 0
        self._current_frame: int = 0
        self._fps: float = _PLAYBACK_FPS

        self._df: Optional[pd.DataFrame] = None
        self._df_by_frame: dict = {}
        self._highlight_ids: Set[int] = set()
        self._marker_scale: float = 1.0
        self._frame_dets = None
        self._trajectory_frame_range: Optional[tuple[int, int]] = None
        self._display_frame_start: int = 0
        self._display_frame_end: int = 0

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        self._prefetch_thread: Optional[_PrefetchThread] = None
        self._prefetch_range: tuple = (-1, -1)

        self._playback_thread: Optional[_PlaybackThread] = None
        self._is_playing: bool = False

        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(_SEEK_DEBOUNCE_MS)
        self._seek_timer.timeout.connect(self._do_seek)
        self._pending_frame: int = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._canvas = InteractiveCanvas()
        root.addWidget(self._canvas, stretch=1)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 2, 4, 2)
        ctrl.setSpacing(6)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.setToolTip("Play / Pause  (Space)")
        self._btn_play.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._btn_play)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(150)
        self._canvas.add_toolbar_widget(self._frame_label)

        marker_spin = QSpinBox()
        marker_spin.setRange(25, 300)
        marker_spin.setValue(100)
        marker_spin.setSuffix("%")
        marker_spin.setFixedWidth(68)
        marker_spin.setToolTip("Overlay marker size (100% = default)")
        marker_spin.valueChanged.connect(self._on_marker_scale)
        self._canvas.add_toolbar_widget(marker_spin)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self._slider, stretch=1)

        root.addLayout(ctrl)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _on_marker_scale(self, value: int) -> None:
        self._marker_scale = value / 100.0
        self._refresh()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Toggle play/pause on Space and step one frame with Left/Right arrow keys."""
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
        elif event.key() == Qt.Key.Key_Right:
            self.seek_to(self._current_frame + 1)
        elif event.key() == Qt.Key.Key_Left:
            self.seek_to(self._current_frame - 1)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_video(self, path: str) -> None:
        """Open a video file, read its frame count and FPS, and reset the slider and display."""
        self._stop_playback()
        self._stop_prefetch()
        self._video_path = path
        self._cache.clear()
        self._frame_dets = load_frame_detections(path)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._total_frames = 0
            self._apply_frame_window()
            return
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps > 0 else _PLAYBACK_FPS
        cap.release()

        self._apply_frame_window()
        self.seek_to(self._display_frame_start)

    def load_trajectories(self, df: pd.DataFrame) -> None:
        """Attach a trajectory DataFrame and rebuild the per-frame index used for overlay rendering."""
        self._df = df
        self._df_by_frame = (
            {int(fid): grp for fid, grp in df.groupby("FrameID")}
            if df is not None
            else {}
        )
        if df is not None and not df.empty:
            self._trajectory_frame_range = (
                int(df["FrameID"].min()),
                int(df["FrameID"].max()),
            )
        else:
            self._trajectory_frame_range = None
        self._apply_frame_window()
        self._refresh()

    def seek_to(self, frame: int) -> None:
        """Jump to the given frame index, stopping playback if active, and refresh the display."""
        frame = max(self._display_frame_start, min(frame, self._display_frame_end))
        if self._is_playing:
            self._stop_playback()
        self._current_frame = frame
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._update_frame_label()
        self._refresh()

    def highlight_tracks(self, track_ids: List[int]) -> None:
        """Set the tracks to render with emphasis in the overlay and refresh the current frame."""
        self._highlight_ids = set(track_ids)
        self._refresh()

    # ------------------------------------------------------------------
    # Play / Pause
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._video_path is None or self._total_frames == 0:
            return
        self._stop_prefetch()
        start = self._current_frame
        if start >= self._display_frame_end:
            start = self._display_frame_start
        t = _PlaybackThread(
            self._video_path,
            start,
            self._display_frame_end + 1,
            self._fps,
            self,
        )
        t.frame_ready.connect(self._on_playback_frame)
        t.finished_playback.connect(self._on_playback_done)
        t.finished.connect(self._on_playback_thread_done)
        t.finished.connect(t.deleteLater)
        t.start()
        self._playback_thread = t
        self._is_playing = True
        self._btn_play.setText("⏸")

    def _stop_playback(self) -> None:
        t = self._playback_thread
        self._playback_thread = None
        self._is_playing = False
        self._btn_play.setText("▶")
        if t is None:
            return
        try:
            if t.isRunning():
                t.requestInterruption()
                t.wait(500)
        except RuntimeError:
            pass

    def _on_playback_frame(self, idx: int, frame: object) -> None:
        if idx not in self._cache:
            while len(self._cache) >= _MAX_CACHE:
                self._cache.popitem(last=False)
            self._cache[idx] = frame
        self._current_frame = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._update_frame_label()
        self._show_frame(frame, idx)  # type: ignore[arg-type]
        self.frame_changed.emit(idx)

    def _on_playback_done(self) -> None:
        self._stop_playback()

    def _on_playback_thread_done(self) -> None:
        try:
            if self.sender() is self._playback_thread:
                self._playback_thread = None
        except RuntimeError:
            self._playback_thread = None

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def _prefetch_thread_running(self) -> bool:
        if self._prefetch_thread is None:
            return False
        try:
            return self._prefetch_thread.isRunning()
        except RuntimeError:
            self._prefetch_thread = None
            return False

    def _stop_prefetch(self) -> None:
        if self._prefetch_thread is None:
            return
        try:
            if self._prefetch_thread.isRunning():
                self._prefetch_thread.requestInterruption()
        except RuntimeError:
            pass
        self._prefetch_thread = None

    def _prefetch_covers(self, frame: int) -> bool:
        s, e = self._prefetch_range
        return s <= frame <= e and self._prefetch_thread_running()

    def _start_prefetch(self, center: int) -> None:
        if self._video_path is None or self._is_playing:
            return
        start = max(self._display_frame_start, center - _PREFETCH_BEHIND)
        end = min(self._display_frame_end, center + _PREFETCH_AHEAD)
        if self._prefetch_thread_running():
            try:
                self._prefetch_thread.requestInterruption()
            except RuntimeError:
                self._prefetch_thread = None
        self._prefetch_range = (start, end)
        t = _PrefetchThread(self._video_path, start, end, self)
        t.frame_decoded.connect(self._on_prefetch_frame)
        t.finished.connect(self._on_prefetch_thread_done)
        t.finished.connect(t.deleteLater)
        t.start()
        self._prefetch_thread = t

    def _on_prefetch_frame(self, idx: int, frame: object) -> None:
        if idx not in self._cache:
            while len(self._cache) >= _MAX_CACHE:
                self._cache.popitem(last=False)
            self._cache[idx] = frame

    def _on_prefetch_thread_done(self) -> None:
        try:
            if self.sender() is self._prefetch_thread:
                self._prefetch_thread = None
        except RuntimeError:
            self._prefetch_thread = None

    # ------------------------------------------------------------------
    # Slider
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        if self._is_playing:
            self._stop_playback()
        self._pending_frame = value
        self._current_frame = value
        self._update_frame_label()
        self._seek_timer.start()
        if not self._prefetch_covers(value):
            self._start_prefetch(value)

    def _do_seek(self) -> None:
        self._current_frame = self._pending_frame
        self._refresh()
        self.frame_changed.emit(self._current_frame)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        if self._video_path is None:
            return None
        cap = cv2.VideoCapture(self._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        while len(self._cache) >= _MAX_CACHE:
            self._cache.popitem(last=False)
        self._cache[idx] = frame
        return frame

    def _refresh(self) -> None:
        self._update_frame_label()
        frame = self._read_frame(self._current_frame)
        if frame is None:
            return
        self._show_frame(frame, self._current_frame)

    def _show_frame(self, bgr: np.ndarray, idx: int) -> None:
        display = bgr.copy()
        if self._df_by_frame:
            draw_overlay(
                display,
                self._df_by_frame,
                idx,
                self._highlight_ids,
                self._marker_scale,
                self._frame_dets,
            )
        h, w = display.shape[:2]
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._canvas.set_pixmap(QPixmap.fromImage(qimg))

    def _apply_frame_window(self) -> None:
        max_end = max(self._total_frames - 1, 0)
        if self._trajectory_frame_range is None:
            start, end = 0, max_end
        else:
            start = max(0, min(self._trajectory_frame_range[0], max_end))
            end = max(start, min(self._trajectory_frame_range[1], max_end))
        self._display_frame_start = start
        self._display_frame_end = end
        self._slider.setRange(start, end)
        self._current_frame = max(start, min(self._current_frame, end))
        self._update_frame_label()
        self._emit_frame_axis_margins_changed()

    def _update_frame_label(self) -> None:
        self._frame_label.setText(
            f"{self._current_frame} [{self._display_frame_start}-{self._display_frame_end}]"
        )
        self._emit_frame_axis_margins_changed()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._emit_frame_axis_margins_changed()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._emit_frame_axis_margins_changed()

    def _slider_handle_center_x(self, value: int) -> int:
        option = QStyleOptionSlider()
        self._slider.initStyleOption(option)
        option.sliderPosition = value
        option.sliderValue = value
        handle_rect = self._slider.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self._slider,
        )
        return self._slider.x() + handle_rect.center().x()

    def frame_axis_margins(self) -> tuple[int, int]:
        if self._slider.width() <= 0:
            return (0, 0)
        left = self._slider_handle_center_x(self._slider.minimum())
        right = self._slider_handle_center_x(self._slider.maximum())
        return (left, right)

    def _emit_frame_axis_margins_changed(self) -> None:
        left, right = self.frame_axis_margins()
        self.frame_axis_margins_changed.emit(left, right)

    def frame_to_slider_x(self, frame: int) -> int:
        start = self._display_frame_start
        end = self._display_frame_end
        axis_start, axis_end = self.frame_axis_margins()
        axis_end = max(axis_end, axis_start + 1)
        if end <= start:
            return axis_start
        clamped = max(start, min(int(frame), end))
        span = end - start
        return int(
            round(axis_start + ((clamped - start) / span) * (axis_end - axis_start))
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop active playback and prefetch threads before the widget is destroyed."""
        self._stop_playback()
        self._stop_prefetch()
        super().closeEvent(event)
