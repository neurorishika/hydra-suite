"""Selectable two-objective views of measured direct-calibration evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.backend_bases import MouseEvent, PickEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.direct_calibration import DirectCalibrationPoint

PlotValue = tuple[int, float, float]


def frontier_rows(values: Sequence[PlotValue]) -> set[int]:
    """Return non-dominated row IDs; minimize x and maximize y, retaining ties.

    Sorting by cost, then descending quality makes a sweep sufficient even
    for a large calibration grid. Identical coordinates never dominate each
    other: dominance requires at least one strict improvement.
    """
    best = -math.inf
    best_x = None
    result = set()
    for row, x, y in sorted(values, key=lambda value: (value[1], -value[2])):
        if y > best or (y == best and x == best_x):
            result.add(row)
            best, best_x = y, x
    return result


def plot_values(
    points: Sequence[DirectCalibrationPoint], cost: str, quality: str
) -> list[PlotValue]:
    """Keep measured, finite observations with their original row identity."""
    values = []
    for row, point in enumerate(points):
        x = getattr(point.score if cost == "precision" else point, cost)
        y = getattr(point.score, quality)
        if point.failed_reason or point.score.frames <= 0 or x is None or y is None:
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if cost != "precision" and x <= 0:
            continue
        values.append((row, float(x), float(y)))
    return values


class _PlotToolbar(NavigationToolbar2QT):
    """Keep the navigation tools visible with readable text labels."""

    toolitems = tuple(
        (
            "Reset view" if name == "Home" else "Box zoom" if name == "Zoom" else name,
            tooltip,
            icon,
            callback,
        )
        for name, tooltip, icon, callback in NavigationToolbar2QT.toolitems
        if name in {"Home", "Back", "Forward", "Pan", "Zoom", "Save", None}
    )

    def __init__(self, canvas: FigureCanvasQTAgg, parent: QWidget) -> None:
        super().__init__(canvas, parent, coordinates=False)
        self.setPalette(parent.palette())
        self.setStyleSheet(
            "QToolBar { background: #1e1e1e; color: #e5e7eb; }"
            "QToolButton { color: #e5e7eb; }"
            "QToolButton:checked { background: #0e639c; }"
            "QToolButton:hover { background: #374151; }"
        )
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def _icon(self, name: str) -> QIcon:
        # QToolBar does not inherit the parent's palette by default. Set it
        # before Matplotlib renders each icon so dark-mode icons use light ink.
        self.setPalette(self.parentWidget().palette())
        return super()._icon(name)


class CalibrationTradeoffs(QWidget):
    """Explore measured tradeoffs without running inference or applying settings."""

    row_selected = Signal(int)
    VIEWS = (
        (
            "Recall vs measured time",
            "seconds_per_frame",
            "recall",
            "Seconds per full frame ↓",
        ),
        (
            "Recall vs tile cost",
            "tiles_per_frame",
            "recall",
            "Tiles per full frame (1 = unsliced) ↓",
        ),
        (
            "Precision vs recall",
            "precision",
            "recall",
            "Precision (fewer extra detections) ↑",
        ),
        ("F1 vs measured time", "seconds_per_frame", "f1", "Seconds per full frame ↓"),
        (
            "Match quality vs measured time",
            "seconds_per_frame",
            "mean_quality",
            "Seconds per full frame ↓",
        ),
    )

    def __init__(
        self, points: Sequence[DirectCalibrationPoint], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.points = points
        self.selected_row = -1
        self.recommended_row = -1
        self._last_pick: tuple[int, ...] = ()
        self.setObjectName("calibrationTradeoffs")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e5e7eb"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e5e7eb"))
        self.setPalette(palette)
        self.setStyleSheet(
            "QWidget#calibrationTradeoffs { background: #1e1e1e; }"
            "QLabel { color: #e5e7eb; background: transparent; }"
        )
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.view = QComboBox()
        self.view.addItems([item[0] for item in self.VIEWS])
        controls.addWidget(self.view, 1)
        self.frontier_only = QCheckBox("Pareto frontier only")
        self.frontier_only.setToolTip(
            "Hide dominated settings for the two displayed metrics. "
            "Frontier membership does not imply eligibility for recommendation."
        )
        controls.addWidget(self.frontier_only)
        layout.addLayout(controls)
        self.figure = Figure(figsize=(8, 5), layout="constrained", facecolor="#1e1e1e")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(580, 360)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toolbar = _PlotToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        hint = QLabel(
            "Scroll to zoom • Box zoom: drag a rectangle • Pan: drag the view • Click to inspect"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)
        self.hover_details = QLabel(
            "Hover over a point to see its settings and measurements."
        )
        self.hover_details.setTextFormat(Qt.TextFormat.PlainText)
        self.hover_details.setWordWrap(True)
        self.hover_details.setMinimumHeight(44)
        layout.addWidget(self.hover_details)
        nearby = QHBoxLayout()
        nearby.addWidget(QLabel("Points at click:"))
        self.nearby_points = QComboBox()
        self.nearby_points.setMinimumContentsLength(16)
        self.nearby_points.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.nearby_points.setToolTip(
            "Choose an exact measurement when several points overlap."
        )
        self.nearby_points.setEnabled(False)
        nearby.addWidget(self.nearby_points, 1)
        self.btn_recommended = QPushButton("Inspect recommended")
        self.btn_recommended.clicked.connect(self._select_recommended)
        nearby.addWidget(self.btn_recommended)
        layout.addLayout(nearby)
        self.count = QLabel()
        self.count.setWordWrap(True)
        layout.addWidget(self.count)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.view.currentIndexChanged.connect(self.redraw)
        self.frontier_only.toggled.connect(self.redraw)
        self.nearby_points.currentIndexChanged.connect(self._select_nearby)
        self.redraw()

    def set_selection(self, row: int) -> None:
        """Update the highlight in place, preserving zoom, pan and history."""
        self.selected_row = row
        values = [(x, y) for index, x, y in self._values if index == row]
        self._selected_artist.set_offsets(np.array(values).reshape(-1, 2))
        self.nearby_points.blockSignals(True)
        self.nearby_points.setCurrentIndex(self.nearby_points.findData(row))
        self.nearby_points.blockSignals(False)
        self.canvas.draw_idle()

    def _select_recommended(self) -> None:
        if 0 <= self.recommended_row < len(self.points):
            self.row_selected.emit(self.recommended_row)
            values = [
                (x, y) for row, x, y in self._values if row == self.recommended_row
            ]
            if values:
                x, y = values[0]
                ax = self._axes
                if not (
                    ax.get_xlim()[0] <= x <= ax.get_xlim()[1]
                    and ax.get_ylim()[0] <= y <= ax.get_ylim()[1]
                ):
                    self.toolbar.push_current()
                    for center, limits, setter in (
                        (x, ax.get_xlim(), ax.set_xlim),
                        (y, ax.get_ylim(), ax.set_ylim),
                    ):
                        half_span = (limits[1] - limits[0]) / 2
                        setter(center - half_span, center + half_span)
                    self.toolbar.push_current()
                    self.canvas.draw_idle()

    def _select_nearby(self, index: int) -> None:
        if index >= 0:
            self.row_selected.emit(self.nearby_points.itemData(index))

    def _point_text(self, row: int) -> str:
        point = self.points[row]
        return (
            f"#{row + 1} {point.label} · confidence {point.confidence:g} · "
            f"merge {point.merge_threshold:g} · {point.tiles_per_frame} tiles · "
            f"{point.seconds_per_frame:.3f} s/frame · recall {point.score.recall:.1%} · "
            f"precision {point.score.precision:.1%} · F1 {point.score.f1:.3f}"
        )

    def _on_pick(self, event: PickEvent) -> None:
        mouse_event = getattr(event, "mouseevent", None)
        if mouse_event is not None and (
            mouse_event.name != "button_press_event" or mouse_event.button != 1
        ):
            return
        if self.toolbar.mode or event.artist is not self._scatter:
            return
        indices = list(event.ind)
        if mouse_event is not None:
            indices.sort(
                key=lambda index: sum(
                    (a - b) ** 2
                    for a, b in zip(
                        self._axes.transData.transform(
                            self._display_values[int(index)][1:]
                        ),
                        (mouse_event.x, mouse_event.y),
                    )
                )
            )
        rows = tuple(self._display_values[int(index)][0] for index in indices)
        if not rows:
            return
        if rows == self._last_pick and self.selected_row in rows:
            row = rows[(rows.index(self.selected_row) + 1) % len(rows)]
        else:
            row = rows[0]
        self._last_pick = rows
        self.nearby_points.blockSignals(True)
        self.nearby_points.clear()
        for index in rows:
            self.nearby_points.addItem(self._point_text(index), index)
        self.nearby_points.setCurrentIndex(rows.index(row))
        self.nearby_points.blockSignals(False)
        self.nearby_points.setEnabled(True)
        self.row_selected.emit(row)

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes is not self._axes or event.xdata is None or event.ydata is None:
            return
        if not event.step:
            return
        self.toolbar.push_current()
        scale = 1.25 ** (-max(-4, min(4, event.step)))
        ax = self._axes
        for center, limits, setter in (
            (event.xdata, ax.get_xlim(), ax.set_xlim),
            (event.ydata, ax.get_ylim(), ax.set_ylim),
        ):
            setter(
                center + (limits[0] - center) * scale,
                center + (limits[1] - center) * scale,
            )
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def _on_hover(self, event: MouseEvent) -> None:
        if self.toolbar.mode or event.inaxes is not self._axes:
            return
        contains, hit = self._scatter.contains(event)
        if not contains or not len(hit["ind"]):
            return
        indices = hit["ind"]
        # The nearest screen-space point is most useful in a dense cloud.
        nearest = min(
            indices,
            key=lambda index: sum(
                (a - b) ** 2
                for a, b in zip(
                    self._axes.transData.transform(
                        self._display_values[int(index)][1:]
                    ),
                    (event.x, event.y),
                )
            ),
        )
        row = self._display_values[int(nearest)][0]
        suffix = (
            f" · {len(indices)} nearby points; click to choose"
            if len(indices) > 1
            else ""
        )
        self.hover_details.setText(self._point_text(row) + suffix)

    def redraw(self, *_unused: object) -> None:
        """Reset navigation when changing metrics or the visible point filter."""
        _, cost, quality, xlabel = self.VIEWS[self.view.currentIndex()]
        self._values = plot_values(self.points, cost, quality)
        frontier = frontier_rows(
            [(row, -x if cost == "precision" else x, y) for row, x, y in self._values]
        )
        self._display_values = [
            value
            for value in self._values
            if not self.frontier_only.isChecked() or value[0] in frontier
        ]
        self._last_pick = ()
        self.nearby_points.blockSignals(True)
        self.nearby_points.clear()
        self.nearby_points.blockSignals(False)
        self.nearby_points.setEnabled(False)
        self.btn_recommended.setEnabled(0 <= self.recommended_row < len(self.points))
        self.figure.clear()
        ax = self._axes = self.figure.add_subplot(facecolor="#252526")
        ax.set_xlabel(xlabel, color="#e5e7eb", fontsize=11)
        ax.set_ylabel(
            {
                "recall": "Recall (animals found) ↑",
                "f1": "F1 ↑",
                "mean_quality": "Match quality ↑",
            }[quality],
            color="#e5e7eb",
            fontsize=11,
        )
        ax.tick_params(colors="#e5e7eb", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#6b7280")
        ax.grid(alpha=0.22, color="#9ca3af")
        self._scatter = ax.scatter(
            [x for _, x, _ in self._display_values],
            [y for _, _, y in self._display_values],
            color="#9ca3af",
            alpha=0.55,
            s=28,
            picker=7,
            label="Measured settings",
        )
        for rows, color, marker, label in (
            (frontier, "#60a5fa", "o", "Pareto frontier"),
            ({self.recommended_row}, "#4ade80", "*", "Recommended"),
        ):
            values = [(x, y) for row, x, y in self._values if row in rows]
            if values:
                ax.scatter(
                    [x for x, _ in values],
                    [y for _, y in values],
                    s=110,
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    linewidths=2,
                    label=label,
                )
        self._selected_artist = ax.scatter(
            [],
            [],
            s=150,
            facecolors="none",
            edgecolors="#fb923c",
            marker="s",
            linewidths=2,
            label="Selected",
        )
        if self._values:
            # Restore all-measurement bounds even when only the frontier is
            # visible; toggling the filter must not distort the comparison.
            xs = [x for _, x, _ in self._values]
            ys = [y for _, _, y in self._values]
            dx = (max(xs) - min(xs)) * 0.06 or max(abs(xs[0]) * 0.06, 0.01)
            dy = (max(ys) - min(ys)) * 0.08 or 0.02
            ax.set_xlim(min(xs) - dx, max(xs) + dx)
            ax.set_ylim(min(ys) - dy, max(ys) + dy)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.14),
                ncol=4,
                fontsize=9,
                facecolor="#252526",
                edgecolor="#6b7280",
                labelcolor="#e5e7eb",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No measurements available for this view",
                transform=ax.transAxes,
                ha="center",
                color="#e5e7eb",
            )
        self.count.setText(
            f"{len(self._display_values)} shown / {len(self._values)} measured · "
            f"{len(frontier)} frontier · {len(self.points) - len(self._values)} unavailable (see table)"
        )
        self.hover_details.setText(
            "Hover to inspect. Timings apply to this calibration data and runtime."
        )
        self.set_selection(self.selected_row)
        self.toolbar.update()
        self.toolbar.push_current()
        self.canvas.draw_idle()
