"""Selectable two-objective views of measured direct-calibration evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence

from matplotlib.backend_bases import PickEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

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


class CalibrationTradeoffs(QWidget):
    """Plot selection emits the table row; it never modifies model settings."""

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
        self._last_pick = ()
        layout = QVBoxLayout(self)
        self.view = QComboBox()
        self.view.addItems([item[0] for item in self.VIEWS])
        layout.addWidget(self.view)
        hint = QLabel(
            "Click a point to inspect its results and preview. Blue rings mark the Pareto frontier: "
            "no other point improves one displayed metric without worsening the other. "
            "A frontier point may still fail the recommendation quality floors. "
            "Repeated clicks cycle overlapping points. Tiles indicate work, not measured speed."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #222; background: #fafafa;")
        layout.addWidget(hint)
        self.figure = Figure(figsize=(6, 2.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)
        layout.addWidget(self.canvas, 1)
        self.count = QLabel()
        self.count.setWordWrap(True)
        self.count.setStyleSheet("color: #222; background: #fafafa;")
        layout.addWidget(self.count)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.view.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def set_selection(self, row: int) -> None:
        """Highlight the table selection without emitting another selection."""
        self.selected_row = row
        self.redraw()

    def _on_pick(self, event: PickEvent) -> None:
        if event.artist is not self._scatter:
            return
        rows = tuple(self._values[int(index)][0] for index in event.ind)
        if not rows:
            return
        # Multiple settings can yield exactly the same metrics. Keep all of
        # them reachable rather than silently choosing the first forever.
        if rows == self._last_pick and self.selected_row in rows:
            row = rows[(rows.index(self.selected_row) + 1) % len(rows)]
        else:
            row = rows[0]
        self._last_pick = rows
        self.row_selected.emit(row)

    def redraw(self, *_unused: object) -> None:
        """Rebuild the active metric view while retaining original row IDs."""
        _, cost, quality, xlabel = self.VIEWS[self.view.currentIndex()]
        self._values = plot_values(self.points, cost, quality)
        frontier = frontier_rows(
            [(row, -x if cost == "precision" else x, y) for row, x, y in self._values]
        )
        self.figure.clear()
        ax = self.figure.add_subplot()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            {
                "recall": "Recall (animals found) ↑",
                "f1": "F1 ↑",
                "mean_quality": "Match quality ↑",
            }[quality]
        )
        ax.grid(alpha=0.2)
        self._scatter = ax.scatter(
            [x for _, x, _ in self._values],
            [y for _, _, y in self._values],
            color="#9ca3af",
            s=24,
            picker=6,
            label="Measured settings",
        )
        for rows, color, marker, label in (
            (frontier, "#2563eb", "o", "Pareto frontier"),
            ({self.recommended_row}, "#16a34a", "*", "Recommended"),
            ({self.selected_row}, "#ea580c", "s", "Selected"),
        ):
            values = [(x, y) for row, x, y in self._values if row in rows]
            if values:
                ax.scatter(
                    [x for x, _ in values],
                    [y for _, y in values],
                    s=85,
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    linewidths=1.8,
                    label=label,
                )
        if self._values:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No measurements available for this view",
                transform=ax.transAxes,
                ha="center",
            )
        self.count.setText(
            f"{len(self._values)} plotted settings · {len(frontier)} on this frontier · "
            f"{len(self.points) - len(self._values)} unavailable (see table). "
            "Timings apply to this calibration data and runtime."
        )
        self.canvas.draw_idle()
