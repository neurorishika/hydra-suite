"""Live loss/metrics plot widget for ultralytics training output.

Uses only Qt QPainter for rendering — no matplotlib dependency.
"""

from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

# Regex to match ultralytics epoch lines. The on-screen format from recent
# Ultralytics releases looks like
#
#   [role]   10/500    8.02G    1.171   0.7284   0.0164   0.09518   ...   100% ...
#
# i.e. an optional bracketed worker prefix, then ``epoch/total``, an optional
# ``<gpu_mem>G`` column, then the loss columns. OBB models add an extra
# ``angle_loss`` column which we ignore.
_EPOCH_RE = re.compile(
    r"^\s*"
    r"(?:\[[A-Za-z0-9_\-]+\]\s+)?"  # optional "[role]" prefix
    r"(\d+)/(\d+)"  # epoch / total
    r"\s+(?:[\d.]+\s*G\s+)?"  # optional GPU memory column
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)"  # box_loss, cls_loss, dfl_loss
)

_SERIES_KEYS = ("box_loss", "cls_loss", "dfl_loss")


def parse_ultralytics_log_line(line: str) -> Optional[dict]:
    """Parse a single ultralytics training log line.

    Returns a dict with epoch, total_epochs, box_loss, cls_loss, dfl_loss
    if the line matches the epoch pattern at the *final* (100%) per-epoch
    progress update; otherwise ``None``. Filtering on the 100% line keeps
    one point per epoch and avoids the noisy in-progress samples that
    Ultralytics emits as the bar fills.
    """
    if "100%" not in line:
        return None
    m = _EPOCH_RE.match(line)
    if m is None:
        return None
    return {
        "epoch": int(m.group(1)),
        "total_epochs": int(m.group(2)),
        "box_loss": float(m.group(3)),
        "cls_loss": float(m.group(4)),
        "dfl_loss": float(m.group(5)),
    }


class LossPlotWidget(QWidget):
    """A lightweight live-updating loss plot drawn with QPainter."""

    _MARGIN_LEFT = 55
    _MARGIN_RIGHT = 90
    _MARGIN_TOP = 20
    _MARGIN_BOTTOM = 30

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.series: dict[str, list[float]] = {k: [] for k in _SERIES_KEYS}
        self.colors: dict[str, QColor] = {
            "box_loss": QColor(66, 165, 245),  # blue
            "cls_loss": QColor(239, 83, 80),  # red
            "dfl_loss": QColor(102, 187, 106),  # green
        }
        self._last_epoch: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_metrics(self, metrics: dict) -> None:
        """Append parsed metric values and schedule a repaint.

        If *metrics* carries an ``epoch`` key that matches the previously
        appended sample, the new sample replaces the last one rather than
        being appended. This keeps a single point per epoch even when the
        same epoch's 100% line shows up more than once in the log.
        """
        epoch = metrics.get("epoch")
        replace = epoch is not None and epoch == self._last_epoch
        for key in _SERIES_KEYS:
            if key not in metrics:
                continue
            if replace and self.series[key]:
                self.series[key][-1] = metrics[key]
            else:
                self.series[key].append(metrics[key])
        if epoch is not None:
            self._last_epoch = int(epoch)
        self.update()

    def clear(self) -> None:
        """Remove all accumulated data."""
        for key in _SERIES_KEYS:
            self.series[key].clear()
        self._last_epoch = None
        self.update()

    def ingest_log_line(self, line: str) -> None:
        """Parse *line*; if it contains epoch metrics, add them."""
        metrics = parse_ultralytics_log_line(line)
        if metrics is not None:
            self.add_metrics(metrics)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Dark background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        ml = self._MARGIN_LEFT
        mr = self._MARGIN_RIGHT
        mt = self._MARGIN_TOP
        mb = self._MARGIN_BOTTOM
        plot_w = max(1, w - ml - mr)
        plot_h = max(1, h - mt - mb)

        # Determine Y range across all series
        all_vals: list[float] = []
        for vals in self.series.values():
            all_vals.extend(vals)

        if not all_vals:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for training data…")
            painter.end()
            return

        y_min = 0.0
        y_max = max(all_vals) * 1.1 or 1.0
        n_points = max(len(v) for v in self.series.values())

        # Draw axes
        axis_pen = QPen(QColor(100, 100, 100))
        painter.setPen(axis_pen)
        # X axis
        painter.drawLine(ml, h - mb, w - mr, h - mb)
        # Y axis
        painter.drawLine(ml, mt, ml, h - mb)

        # Y-axis tick labels
        label_font = QFont("monospace", 8)
        painter.setFont(label_font)
        painter.setPen(QColor(160, 160, 160))
        n_ticks = 5
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            y_val = y_min + (y_max - y_min) * (1 - frac)
            y_px = int(mt + frac * plot_h)
            painter.drawText(
                0, y_px - 6, ml - 5, 14, Qt.AlignRight | Qt.AlignVCenter, f"{y_val:.3f}"
            )
            painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.DotLine))
            painter.drawLine(ml, y_px, w - mr, y_px)
            painter.setPen(QColor(160, 160, 160))

        # X-axis labels (first and last epoch)
        if n_points > 0:
            painter.drawText(
                ml - 5, h - mb + 4, 40, 16, Qt.AlignLeft | Qt.AlignTop, "1"
            )
            painter.drawText(
                w - mr - 35,
                h - mb + 4,
                40,
                16,
                Qt.AlignRight | Qt.AlignTop,
                str(n_points),
            )

        # Draw series
        for key in _SERIES_KEYS:
            vals = self.series[key]
            if len(vals) < 2:
                continue
            pen = QPen(self.colors[key], 2)
            painter.setPen(pen)
            for i in range(1, len(vals)):
                x0 = ml + int((i - 1) / max(1, n_points - 1) * plot_w)
                x1 = ml + int(i / max(1, n_points - 1) * plot_w)
                y0 = mt + int((1 - (vals[i - 1] - y_min) / (y_max - y_min)) * plot_h)
                y1 = mt + int((1 - (vals[i] - y_min) / (y_max - y_min)) * plot_h)
                painter.drawLine(x0, y0, x1, y1)

        # Legend
        lx = w - mr + 10
        ly = mt + 5
        painter.setFont(label_font)
        for key in _SERIES_KEYS:
            painter.setPen(self.colors[key])
            painter.fillRect(lx, ly, 10, 10, self.colors[key])
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(lx + 14, ly, 70, 14, Qt.AlignLeft | Qt.AlignVCenter, key)
            ly += 18

        painter.end()
