"""The calibration frontier: every (tile fraction, confidence) measured.

Shows what was measured on the user's OWN frames and hardware. It is the
only place a run-time projection may come from -- the archived dev-machine
timings do not reconcile and are never quoted (see the design doc's Cost
section).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.semantic.tiling import merge_candidates
from hydra_suite.detectkit.gui.canvas import OBBCanvas
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
        preview_frames=None,
        merge_iou: float = 0.5,
        parent=None,
    ) -> None:
        super().__init__("Calibration results", parent=parent)
        self.partial = bool(partial)
        self._rows = frontier_rows(points, recommended, project_frames)
        self._chosen = None
        self._preview_frames = list(preview_frames or [])
        self._merge_iou = float(merge_iou)
        self._frame_index = 0
        self._loaded_image_path = None

        self.setMinimumSize(900, 650)
        self.resize(1280, 860)

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

        splitter = QSplitter(Qt.Orientation.Vertical)

        self._table = QTableWidget(len(self._rows), len(COLUMNS))
        self._table.setHorizontalHeaderLabels([label for _key, label in COLUMNS])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        for column, width in enumerate((180, 100, 125, 135, 90, 90, 155, 125)):
            self._table.setColumnWidth(column, width)
        selected_row = 0
        for r, row in enumerate(self._rows):
            for c, (key, _label) in enumerate(COLUMNS):
                self._table.setItem(r, c, QTableWidgetItem(row[key]))
            if row["recommended"]:
                selected_row = r
                self._table.selectRow(r)
        self._table.currentCellChanged.connect(self._on_row_changed)
        splitter.addWidget(self._table)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 8, 0, 0)
        preview_layout.setSpacing(8)

        preview_header = QHBoxLayout()
        self._previous_frame = QPushButton("◀ Previous frame")
        self._previous_frame.setObjectName("calibrationPreviousFrame")
        self._previous_frame.clicked.connect(lambda: self._step_frame(-1))
        preview_header.addWidget(self._previous_frame)
        self._frame_label = QLabel("")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_header.addWidget(self._frame_label, 1)
        self._next_frame = QPushButton("Next frame ▶")
        self._next_frame.setObjectName("calibrationNextFrame")
        self._next_frame.clicked.connect(lambda: self._step_frame(1))
        preview_header.addWidget(self._next_frame)
        self._show_gt = QCheckBox("Ground truth")
        self._show_gt.setChecked(True)
        self._show_gt.toggled.connect(self._refresh_visibility)
        preview_header.addWidget(self._show_gt)
        self._show_predictions = QCheckBox("Predictions")
        self._show_predictions.setChecked(True)
        self._show_predictions.toggled.connect(self._refresh_visibility)
        preview_header.addWidget(self._show_predictions)
        preview_layout.addLayout(preview_header)

        self._preview_status = QLabel("")
        self._preview_status.setWordWrap(True)
        preview_layout.addWidget(self._preview_status)
        self._canvas = OBBCanvas()
        self._canvas.setObjectName("calibrationOverlayCanvas")
        self._canvas.setMinimumHeight(260)
        preview_layout.addWidget(self._canvas, 1)
        self._preview_hint = QLabel(
            "Green solid fill = ground truth · Blue dashed fill = prediction. "
            "Ctrl+wheel or +/- zooms, drag pans, and double-click fits."
        )
        self._preview_hint.setWordWrap(True)
        preview_layout.addWidget(self._preview_hint)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([300, 480])
        outer.addWidget(splitter, 1)

        note = QLabel(
            "Timings are measured on this machine and these frames. Tile "
            "fraction changes require re-running inference; confidence does "
            "not — a staged run can be re-thresholded from its candidate cache."
        )
        note.setWordWrap(True)
        outer.addWidget(note)
        self.add_content(container)

        if self._rows:
            self._table.selectRow(selected_row)
            self._render_preview()
        else:
            self._render_preview()

    def _on_row_changed(self, row: int, _column: int, *_unused) -> None:
        if row < 0:
            return
        self._frame_index = min(
            self._frame_index, max(len(self._preview_frames) - 1, 0)
        )
        self._render_preview()

    def _step_frame(self, amount: int) -> None:
        if not self._preview_frames:
            return
        self._frame_index = (self._frame_index + amount) % len(self._preview_frames)
        self._render_preview()

    def _refresh_visibility(self) -> None:
        self._canvas.set_overlay_visibility(
            self._show_gt.isChecked(), self._show_predictions.isChecked()
        )

    def _render_preview(self) -> None:
        has_frames = bool(self._preview_frames)
        self._previous_frame.setEnabled(len(self._preview_frames) > 1)
        self._next_frame.setEnabled(len(self._preview_frames) > 1)
        self._show_gt.setEnabled(has_frames)
        self._show_predictions.setEnabled(has_frames)
        if not has_frames:
            self._canvas.clear_all()
            self._loaded_image_path = None
            self._frame_label.setText("No visual preview")
            self._preview_status.setText(
                "Visual evidence is not available for this calibration. Re-run "
                "calibration to generate explorable prediction overlays."
            )
            return

        row = self._table.currentRow()
        if row < 0 or row >= len(self._rows):
            row = 0
        point = self._rows[row]["point"]
        self._frame_index %= len(self._preview_frames)
        frame = self._preview_frames[self._frame_index]
        candidates = frame.candidates_by_fraction.get(point.tile_fraction)
        if candidates is None:
            self._canvas.clear_all()
            self._loaded_image_path = None
            self._frame_label.setText(
                f"Frame {self._frame_index + 1} of {len(self._preview_frames)} · "
                f"{frame.image_path.name}"
            )
            self._preview_status.setText(
                "This tiling was not measured on this frame, so no comparable "
                "overlay is available."
            )
            return
        if self._loaded_image_path != frame.image_path and not self._canvas.load_image(
            str(frame.image_path)
        ):
            self._canvas.clear_all()
            self._loaded_image_path = None
            self._frame_label.setText(frame.image_path.name)
            self._preview_status.setText(
                f"The calibration image could not be read: {frame.image_path}"
            )
            return
        self._loaded_image_path = frame.image_path

        merged = merge_candidates(
            candidates,
            confidence_threshold=point.confidence,
            iou_threshold=self._merge_iou,
        )
        gt_detections = [
            {
                "class_id": 0,
                "polygon_px": item.polygon_px.tolist(),
            }
            for item in frame.ground_truth
        ]
        prediction_detections = [
            {
                "class_id": 2,
                "polygon_px": instance.polygon_px.tolist(),
                "confidence": instance.confidence,
            }
            for instance in merged
        ]
        labels = {0: "Ground truth", 2: "Prediction"}
        self._canvas.set_gt_detections(gt_detections, labels, fill_alpha=65)
        self._canvas.set_pred_detections(prediction_detections, labels, fill_alpha=55)
        self._refresh_visibility()
        self._frame_label.setText(
            f"Frame {self._frame_index + 1} of {len(self._preview_frames)} · "
            f"{frame.image_path.name}"
        )
        tiling = (
            "full frame"
            if point.tile_fraction is None
            else f"tile fraction {point.tile_fraction:.2f} ({point.tile_px} px)"
        )
        self._preview_status.setText(
            f"Selected calibration: {tiling}, confidence {point.confidence:.2f}. "
            f"This frame has {len(gt_detections)} ground-truth and "
            f"{len(prediction_detections)} predicted segment(s)."
        )

    def accept(self) -> None:
        rows = {i.row() for i in self._table.selectedIndexes()}
        if rows:
            self._chosen = self._rows[sorted(rows)[0]]["point"]
        super().accept()

    def chosen(self):
        return self._chosen
