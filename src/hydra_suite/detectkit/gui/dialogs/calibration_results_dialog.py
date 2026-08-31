"""The calibration frontier: every (tile fraction, confidence) measured.

Shows what was measured on the user's OWN frames and hardware. It is the
only place a run-time projection may come from -- the archived dev-machine
timings do not reconcile and are never quoted (see the design doc's Cost
section).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

COLUMNS = [
    ("tile", "Tiling"),
    ("confidence", "Confidence"),
    ("missed", "Missed /frame"),
    ("extra", "To delete /frame"),
    ("recall", "Recall"),
    ("matched", "Matched"),
    # How WELL the matches match, not just how many. A configuration can
    # post a high recall with masks covering whole regions or single legs;
    # this column is where that shows.
    ("quality", "Match quality"),
    ("seconds", "s/frame (measured)"),
    ("projected", "Projected run"),
]


def _humanise(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return (
            f"{hours} h {minutes:02d} m" if minutes >= 10 else f"{hours} h {minutes} m"
        )
    return f"{minutes} m" if minutes else f"{total} s"


def frontier_rows(points, recommended, project_frames: int) -> list[dict]:
    """Format the frontier: cheapest tiling first, confidence descending."""
    ordered = sorted(points, key=lambda p: (p.tiles_per_frame, -p.confidence))
    rows = []
    for p in ordered:
        tile = (
            "full frame"
            if p.tile_fraction is None
            else f"{p.tile_fraction:.2f} ({p.tiles_per_frame} tiles/frame)"
        )
        rows.append(
            {
                "tile": tile,
                "confidence": f"{p.confidence:.2f}",
                "missed": f"{p.missed_per_frame:.1f}",
                "extra": f"{p.extra_per_frame:.1f}",
                "recall": f"{p.recall:.1%}",
                "matched": str(p.n_matched),
                "quality": (
                    f"{p.mean_quality:.2f} "
                    f"(IoU {p.median_iou:.2f}, area {p.median_area_ratio:.2f})"
                ),
                "seconds": f"{p.seconds_per_frame:.1f}",
                "projected": _humanise(p.seconds_per_frame * max(project_frames, 0)),
                "recommended": recommended is not None and p is recommended,
                "point": p,
            }
        )
    return rows


class CalibrationResultsDialog(BaseDialog):
    """Pick an operating point off the measured frontier."""

    def __init__(
        self,
        points,
        recommended,
        reason: str,
        *,
        project_frames: int,
        partial: bool = False,
        parent=None,
    ) -> None:
        super().__init__("Calibration results", parent=parent)
        self.partial = bool(partial)
        self._rows = frontier_rows(points, recommended, project_frames)
        self._chosen = None

        container = QWidget()
        outer = QVBoxLayout(container)
        headline = QLabel(
            reason
            if recommended is None
            else (
                "Recommended: the cheapest tiling that still finds "
                f"{recommended.recall:.0%} of your labelled animals, then the "
                "highest confidence at that tiling. Chosen for recall, not F1 — "
                "a spurious polygon is one click, a missed animal must be found "
                "by eye. Pick any row to override."
            )
        )
        headline.setWordWrap(True)
        outer.addWidget(headline)

        # F6: a cancelled sweep is not a finished one. Fractions whose frames
        # were only part-inferred are dropped, so what survives is measured on
        # FEWER frames than you asked for -- comparable between rows, but a
        # thinner sample than the dialog otherwise implies.
        if self.partial:
            warning = QLabel(
                "\u26a0 PARTIAL — this calibration was cancelled before it "
                "finished. Only fully-inferred frames are counted, so these "
                "rows rest on fewer frames than you selected. Re-run to "
                "completion before trusting the recommendation."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b36b00; font-weight: bold;")
            outer.addWidget(warning)

        self._table = QTableWidget(len(self._rows), len(COLUMNS))
        self._table.setHorizontalHeaderLabels([label for _key, label in COLUMNS])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        for r, row in enumerate(self._rows):
            for c, (key, _label) in enumerate(COLUMNS):
                self._table.setItem(r, c, QTableWidgetItem(row[key]))
            if row["recommended"]:
                self._table.selectRow(r)
        outer.addWidget(self._table)

        note = QLabel(
            "Timings are measured on this machine and these frames. Tile "
            "fraction changes require re-running inference; confidence does "
            "not — a staged run can be re-thresholded from its candidate cache."
        )
        note.setWordWrap(True)
        outer.addWidget(note)
        self.add_content(container)

    def accept(self) -> None:
        rows = {i.row() for i in self._table.selectedIndexes()}
        if rows:
            self._chosen = self._rows[sorted(rows)[0]]["point"]
        super().accept()

    def chosen(self):
        return self._chosen
