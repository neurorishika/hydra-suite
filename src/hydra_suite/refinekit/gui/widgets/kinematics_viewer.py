"""Collapsed-by-default kinematics overlay viewer for the RefineKit main UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

SERIES_SPECS = (
    ("velocity", "Velocity", QColor("#4fc3f7"), True),
    ("acceleration", "Acceleration", QColor("#ffb74d"), True),
    ("angle", "Angle", QColor("#ce93d8"), True),
    ("angular_velocity", "Angular Velocity", QColor("#81c784"), True),
    ("angular_acceleration", "Angular Acceleration", QColor("#ef5350"), False),
    ("detection_confidence", "Detection Confidence", QColor("#fff176"), False),
    ("assignment_confidence", "Assignment Confidence", QColor("#f48fb1"), False),
)

DEFAULT_SERIES_ENABLED = {key: enabled for key, _label, _color, enabled in SERIES_SPECS}


@dataclass(frozen=True)
class TrackKinematics:
    """Precomputed normalized kinematics series for one trajectory."""

    frames: np.ndarray
    frame_start: int
    frame_end: int
    normalized_series: Dict[str, np.ndarray]
    finite_masks: Dict[str, np.ndarray]


def _resolve_column(columns: pd.Index, *candidates: str) -> Optional[str]:
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _differentiate(values: np.ndarray, frame_delta: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if values.size < 2:
        return result
    delta = np.diff(values, prepend=np.nan)
    valid = np.isfinite(delta) & np.isfinite(frame_delta) & (frame_delta > 0)
    result[valid] = delta[valid] / frame_delta[valid]
    return result


def _unwrap_angles(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    radians = values.astype(float).copy()
    if np.nanmax(np.abs(radians[finite])) > (2 * np.pi + 0.5):
        radians[finite] = np.deg2rad(radians[finite])
    result[finite] = np.unwrap(radians[finite])
    return result


def _normalize_series(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=np.float32)
    if not finite.any():
        return normalized, finite
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if abs(vmax - vmin) < 1e-9:
        normalized[finite] = 0.0
    else:
        normalized[finite] = (
            ((values[finite] - vmin) / (vmax - vmin)) * 2.0 - 1.0
        ).astype(np.float32)
    return normalized, finite


def build_kinematics_cache(
    df: Optional[pd.DataFrame],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[Dict[int, TrackKinematics], tuple[int, int]]:
    """Precompute normalized kinematics series for all tracks in *df*."""
    if df is None or df.empty:
        return {}, (0, 0)

    frame_range = (int(df["FrameID"].min()), int(df["FrameID"].max()))
    ordered = df.sort_values(["TrajectoryID", "FrameID"])
    theta_col = _resolve_column(ordered.columns, "Theta")
    detection_col = _resolve_column(
        ordered.columns,
        "DetectionConfidence",
        "detection_confidence",
        "DetectionScore",
    )
    assignment_col = _resolve_column(
        ordered.columns,
        "AssignmentConfidence",
        "assignment_confidence",
        "AssignmentScore",
    )

    grouped = list(ordered.groupby("TrajectoryID", sort=True))
    total = len(grouped)
    cache: Dict[int, TrackKinematics] = {}

    for index, (track_id, track_df) in enumerate(grouped, start=1):
        if should_cancel is not None and should_cancel():
            return {}, frame_range

        frames = track_df["FrameID"].to_numpy(dtype=np.float32)
        x = track_df.get("X", pd.Series(np.nan, index=track_df.index)).to_numpy(
            dtype=float
        )
        y = track_df.get("Y", pd.Series(np.nan, index=track_df.index)).to_numpy(
            dtype=float
        )
        frame_delta = np.diff(frames, prepend=np.nan)
        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)

        velocity = np.full(frames.shape, np.nan, dtype=float)
        valid_motion = (
            np.isfinite(dx)
            & np.isfinite(dy)
            & np.isfinite(frame_delta)
            & (frame_delta > 0)
        )
        velocity[valid_motion] = (
            np.hypot(dx[valid_motion], dy[valid_motion]) / frame_delta[valid_motion]
        )
        acceleration = _differentiate(velocity, frame_delta)

        if theta_col is not None and track_df[theta_col].notna().any():
            angle = _unwrap_angles(track_df[theta_col].to_numpy(dtype=float))
        else:
            angle = _unwrap_angles(np.arctan2(dy, dx))
        angular_velocity = _differentiate(angle, frame_delta)
        angular_acceleration = _differentiate(angular_velocity, frame_delta)

        detection_confidence = (
            track_df[detection_col].to_numpy(dtype=float)
            if detection_col is not None
            else np.full(frames.shape, np.nan, dtype=float)
        )
        assignment_confidence = (
            track_df[assignment_col].to_numpy(dtype=float)
            if assignment_col is not None
            else np.full(frames.shape, np.nan, dtype=float)
        )

        raw_series = {
            "velocity": velocity,
            "acceleration": acceleration,
            "angle": angle,
            "angular_velocity": angular_velocity,
            "angular_acceleration": angular_acceleration,
            "detection_confidence": detection_confidence,
            "assignment_confidence": assignment_confidence,
        }
        normalized_series = {}
        finite_masks = {}
        for key, values in raw_series.items():
            normalized, finite = _normalize_series(values)
            normalized_series[key] = normalized
            finite_masks[key] = finite

        cache[int(track_id)] = TrackKinematics(
            frames=track_df["FrameID"].to_numpy(dtype=np.int32),
            frame_start=int(track_df["FrameID"].min()),
            frame_end=int(track_df["FrameID"].max()),
            normalized_series=normalized_series,
            finite_masks=finite_masks,
        )

        if progress_callback is not None:
            progress_callback(index, total)

    return cache, frame_range


class KinematicsViewerWidget(QWidget):
    """Normalized multi-series kinematics viewer aligned to the review frame range."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame_range = (0, 0)
        self._current_frame = 0
        self._active_track_id: Optional[int] = None
        self._enabled_series = dict(DEFAULT_SERIES_ENABLED)
        self._track_cache: Dict[int, TrackKinematics] = {}
        self._loading = False
        self._loading_text = "Preparing kinematics…"
        self._plot_cache_key: Optional[tuple] = None
        self._plot_cache_pixmap: Optional[QPixmap] = None
        self._axis_left_x = 10
        self._axis_right_x = 10

        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    @property
    def active_track_id(self) -> Optional[int]:
        return self._active_track_id

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def enabled_series(self) -> Dict[str, bool]:
        return dict(self._enabled_series)

    @property
    def is_loading(self) -> bool:
        return self._loading

    def set_data(self, df: Optional[pd.DataFrame]) -> None:
        self._track_cache.clear()
        self._invalidate_plot_cache()
        self._track_cache.clear()
        if df is None or df.empty:
            self._frame_range = (0, 0)
            self._active_track_id = None
            self._current_frame = 0
            self._loading = False
            self.update()
            return
        frame_start = int(df["FrameID"].min())
        frame_end = int(df["FrameID"].max())
        self._frame_range = (frame_start, frame_end)
        self._current_frame = max(frame_start, min(self._current_frame, frame_end))
        valid_tracks = {
            int(track_id) for track_id in df["TrajectoryID"].dropna().tolist()
        }
        if self._active_track_id not in valid_tracks:
            self._active_track_id = None
        self.update()

    def set_loading(
        self, loading: bool, message: str = "Preparing kinematics…"
    ) -> None:
        if self._loading == loading and self._loading_text == message:
            return
        self._loading = loading
        self._loading_text = message
        self._invalidate_plot_cache()
        self.update()

    def set_precomputed_data(
        self,
        track_cache: Dict[int, TrackKinematics],
        frame_range: tuple[int, int],
    ) -> None:
        self._track_cache = dict(track_cache)
        self._frame_range = frame_range
        self._current_frame = max(
            frame_range[0], min(self._current_frame, frame_range[1])
        )
        if self._active_track_id not in self._track_cache:
            self._active_track_id = None
        self._loading = False
        self._invalidate_plot_cache()
        self.update()

    def set_active_track(self, track_id: Optional[int]) -> None:
        if track_id == self._active_track_id:
            return
        self._active_track_id = None if track_id is None else int(track_id)
        self._invalidate_plot_cache()
        self.update()

    def set_current_frame(self, frame: int) -> None:
        clamped = max(self._frame_range[0], min(int(frame), self._frame_range[1]))
        if clamped == self._current_frame:
            return
        self._current_frame = clamped
        self.update()

    def set_frame_axis_margins(self, left_margin: int, right_margin: int) -> None:
        left_margin = max(0, int(left_margin))
        right_margin = max(left_margin + 1, int(right_margin))
        if left_margin == self._axis_left_x and right_margin == self._axis_right_x:
            return
        self._axis_left_x = left_margin
        self._axis_right_x = right_margin
        self._invalidate_plot_cache()
        self.update()

    def set_enabled_series(self, enabled: Dict[str, bool]) -> None:
        updated = {
            key: bool(enabled.get(key, self._enabled_series.get(key, True)))
            for key, _label, _color, _default in SERIES_SPECS
        }
        if updated == self._enabled_series:
            return
        self._enabled_series = updated
        self._invalidate_plot_cache()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._invalidate_plot_cache()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        self._ensure_plot_cache()
        if self._plot_cache_pixmap is not None:
            painter.drawPixmap(0, 0, self._plot_cache_pixmap)

        track_series = (
            None
            if self._active_track_id is None
            else self._track_cache.get(self._active_track_id)
        )
        if track_series is not None:
            plot_rect = self._plot_rect()
            cursor_x = self._frame_to_x(self._current_frame)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawLine(cursor_x, plot_rect.top(), cursor_x, plot_rect.bottom())
        painter.end()

    def _invalidate_plot_cache(self) -> None:
        self._plot_cache_key = None
        self._plot_cache_pixmap = None

    def _ensure_plot_cache(self) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return

        track_series = (
            None
            if self._active_track_id is None
            else self._track_cache.get(self._active_track_id)
        )
        enabled_keys = []
        if track_series is not None:
            enabled_keys = [
                key
                for key, _label, _color, _default in SERIES_SPECS
                if self._enabled_series.get(key, False)
                and track_series.finite_masks.get(key, np.array([], dtype=bool)).any()
            ]

        cache_key = (
            self.width(),
            self.height(),
            self._active_track_id,
            tuple(enabled_keys),
            self._frame_range,
            self._loading,
            self._loading_text,
        )
        if self._plot_cache_key == cache_key and self._plot_cache_pixmap is not None:
            return

        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#1e1e1e"))
        painter = QPainter(pixmap)

        plot_rect = self._plot_rect()
        painter.fillRect(plot_rect, QColor("#252526"))
        painter.setPen(QPen(QColor("#3e3e42"), 1))
        painter.drawRect(plot_rect.adjusted(0, 0, -1, -1))

        painter.setPen(QColor("#9cdcfe"))
        title = (
            "Kinematics"
            if self._active_track_id is None
            else f"Kinematics  ·  T{self._active_track_id}"
        )
        painter.drawText(12, 18, title)

        if self._active_track_id is None:
            self._draw_center_message(
                painter,
                plot_rect,
                "Select a track in the bottom timeline to inspect normalized traces.",
            )
        elif track_series is None:
            message = (
                self._loading_text
                if self._loading
                else "No kinematic data for this track."
            )
            self._draw_center_message(painter, plot_rect, message)
        elif not enabled_keys:
            self._draw_center_message(
                painter, plot_rect, "Enable at least one trace to plot it here."
            )
        else:
            self._draw_static_plot(painter, plot_rect, track_series, enabled_keys)

        painter.end()
        self._plot_cache_key = cache_key
        self._plot_cache_pixmap = pixmap

    def _draw_center_message(self, painter: QPainter, plot_rect, message: str) -> None:
        painter.setPen(QColor("#777777"))
        painter.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, message)

    def _draw_static_plot(
        self,
        painter: QPainter,
        plot_rect,
        track_series: TrackKinematics,
        enabled_keys: list[str],
    ) -> None:
        highlight_left = self._frame_to_x(track_series.frame_start)
        highlight_right = self._frame_to_x(track_series.frame_end)
        painter.fillRect(
            highlight_left,
            plot_rect.top(),
            max(highlight_right - highlight_left, 1),
            plot_rect.height(),
            QColor(255, 255, 255, 12),
        )

        mid_y = plot_rect.center().y()
        painter.setPen(QPen(QColor("#3e3e42"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(plot_rect.left(), mid_y, plot_rect.right(), mid_y)

        legend_x = 12
        for key, label, color, _default in SERIES_SPECS:
            if key not in enabled_keys:
                continue
            painter.setPen(QPen(color, 2))
            painter.drawLine(legend_x, 24, legend_x + 12, 24)
            legend_x += 16
            painter.setPen(color)
            painter.drawText(legend_x, 28, label)
            legend_x += max(72, len(label) * 7)

        y_radius = max(plot_rect.height() / 2.0 - 8.0, 1.0)
        frames = track_series.frames.astype(np.float32, copy=False)
        for key, _label, color, _default in SERIES_SPECS:
            if key not in enabled_keys:
                continue
            normalized = track_series.normalized_series[key]
            finite = track_series.finite_masks[key]
            path = QPainterPath()
            drawing = False
            for frame, value, valid in zip(frames, normalized, finite):
                if not valid:
                    drawing = False
                    continue
                x = float(self._frame_to_x(int(frame)))
                y = mid_y - float(value) * y_radius
                if not drawing:
                    path.moveTo(x, y)
                    drawing = True
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(color, 1.6))
            painter.drawPath(path)

        painter.setPen(QColor("#777777"))
        painter.drawText(
            plot_rect.adjusted(0, 0, -4, -4),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
            "normalized overlays",
        )

    def _plot_rect(self):
        left = min(max(self._axis_left_x, 0), max(self.width() - 2, 0))
        right = min(max(self._axis_right_x, left + 1), max(self.width() - 1, left + 1))
        return self.rect().adjusted(left, 30, -(self.width() - right), -12)

    def _frame_to_x(self, frame: int) -> int:
        plot_rect = self._plot_rect()
        span_start, span_end = self._frame_range
        if span_end <= span_start:
            return plot_rect.left()
        clamped = max(span_start, min(int(frame), span_end))
        span = span_end - span_start
        return int(
            round(
                plot_rect.left() + ((clamped - span_start) / span) * plot_rect.width()
            )
        )
