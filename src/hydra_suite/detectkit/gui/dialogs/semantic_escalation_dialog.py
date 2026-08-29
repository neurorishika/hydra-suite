"""SemanticEscalationDialog — prompt, tiling, calibration, one-tile preview."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.semantic.checkpoints import (
    CHECKPOINT_SIZE_GB,
    probe_availability,
)
from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    SEMANTIC_TILE_FRACTION_SEED,
    resolve_tile_px,
)
from hydra_suite.widgets.dialogs import BaseDialog


def _humanise_seconds(seconds: float) -> str:
    """A rough duration, from a MEASURED number of seconds."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


class SemanticEscalationDialog(BaseDialog):
    """Configure a SAM3 semantic escalation run.

    Calibration is offered whenever the selected sources hold a labelled
    frame at ANY geometry level -- choosing an operating point needs
    instance COUNTS, not masks, so OBB and AABB labels work as well as
    polygons.
    """

    def __init__(
        self,
        sources,
        reference_body_px: float,
        parent=None,
        body_px_origin: str = "",
    ) -> None:
        super().__init__(
            "Semantic escalation (SAM3)",
            parent=parent,
            buttons=(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            ),
        )
        self._sources = list(sources)
        self._body_px_origin = body_px_origin
        self.calibration_points: list = []
        self._preview_worker = None

        container = QWidget()
        outer = QVBoxLayout(container)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for src in self._sources:
            self._list.addItem(f"{src.name}  ({src.level})")
        outer.addWidget(QLabel("Sources to escalate:"))
        outer.addWidget(self._list)

        form = QFormLayout()
        self._variant = QComboBox()
        from hydra_suite.core.inference.semantic.checkpoints import available_variants

        self._variant.addItems(available_variants())
        form.addRow("Model:", self._variant)

        self._prompt = QLineEdit("ant")
        self._prompt.setToolTip(
            "A noun phrase. Wording matters far less than tile size — try "
            "variants in the preview if results look wrong."
        )
        form.addRow("Prompt:", self._prompt)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.01, 0.99)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(0.35)
        form.addRow("Confidence:", self._confidence)

        self._max_instances = QSpinBox()
        self._max_instances.setRange(0, 10000)
        self._max_instances.setSpecialValueText("unlimited")
        form.addRow("Max instances/tile:", self._max_instances)

        self._overlap = QDoubleSpinBox()
        self._overlap.setRange(0.0, 0.9)
        self._overlap.setSingleStep(0.1)
        self._overlap.setValue(DEFAULT_OVERLAP)
        form.addRow("Tile overlap:", self._overlap)

        self._seam_margin = QSpinBox()
        self._seam_margin.setRange(0, 64)
        self._seam_margin.setValue(int(DEFAULT_SEAM_MARGIN_PX))
        form.addRow("Seam margin (px):", self._seam_margin)

        self._merge_iou = QDoubleSpinBox()
        self._merge_iou.setRange(0.05, 0.95)
        self._merge_iou.setSingleStep(0.05)
        self._merge_iou.setValue(DEFAULT_MERGE_IOU)
        form.addRow("Cross-tile merge IoU:", self._merge_iou)

        # I6: link 3 of the reference_body_px resolution chain (project
        # setting -> median of the source's existing labels -> THE USER).
        # Without an editable control the chain dead-ends and an unresolved
        # value silently switches tiling off -- the measured-worst
        # configuration (F1 0.719 -> 0.075).
        self._reference_body = QDoubleSpinBox()
        self._reference_body.setRange(0.0, 4096.0)
        self._reference_body.setDecimals(1)
        self._reference_body.setSingleStep(5.0)
        self._reference_body.setSpecialValueText("unknown (tiling off)")
        self._reference_body.setValue(float(reference_body_px or 0.0))
        self._reference_body.setToolTip(
            "The typical longest side of one animal, in pixels. Tile size = "
            "this / tile fraction. With no value, tiling is off — which is "
            "the worst measured configuration for small animals."
        )
        self._reference_body.valueChanged.connect(self._refresh_tile_label)
        form.addRow("Reference body size (px):", self._reference_body)
        self._body_origin_label = QLabel(body_px_origin or "entered by you")
        self._body_origin_label.setWordWrap(True)
        form.addRow("  from:", self._body_origin_label)

        # The fraction is a CALIBRATED parameter. The seed is presented as a
        # guess, never as a tuned or recommended value -- it was back-derived
        # from one measured configuration on one dataset.
        self._tile_fraction = QDoubleSpinBox()
        self._tile_fraction.setRange(0.0, 0.90)
        self._tile_fraction.setSingleStep(0.01)
        self._tile_fraction.setDecimals(2)
        self._tile_fraction.setSpecialValueText("full frame (no tiling)")
        self._tile_fraction.setValue(SEMANTIC_TILE_FRACTION_SEED)
        self._tile_fraction.setToolTip(
            "Tile size = reference body size / this fraction. The default is a "
            "starting guess from one dataset, not a tuned value — calibrate "
            "against your own labelled frames to fit it."
        )
        self._tile_fraction.valueChanged.connect(self._refresh_tile_label)
        form.addRow("Tile fraction:", self._tile_fraction)

        self._tile_label = QLabel("")
        self._tile_label.setWordWrap(True)
        form.addRow("Tile size:", self._tile_label)
        self._refresh_tile_label()
        outer.addLayout(form)

        self._exhaustive = QCheckBox(
            "My labelled frames are exhaustively labelled (every animal is marked)"
        )
        self._exhaustive.setToolTip(
            "Calibration counts an unlabelled real animal as a false positive, "
            "which biases the recommended threshold upward."
        )
        outer.addWidget(self._exhaustive)

        self._btn_calibrate = QPushButton("Calibrate against labelled frames…")
        self._btn_calibrate.setEnabled(False)
        outer.addWidget(self._btn_calibrate)
        self._btn_calibrate.clicked.connect(self._run_calibration)
        self._refresh_calibration_enabled()

        self._btn_preview = QPushButton("Preview one tile")
        self._btn_preview.setToolTip(
            "Runs ONE tile, not the whole frame: a full-frame preview shows "
            "near-zero detections and teaches you the feature is broken. "
            "Reports the instance count and the measured time for that tile."
        )
        self._btn_preview.clicked.connect(self._run_preview)
        outer.addWidget(self._btn_preview)

        # C1: the 3.45 GB download is surfaced HERE, before any run starts.
        # The tools-panel button is enabled when only the checkpoint is
        # missing precisely so this dialog can be reached to offer it.
        self._checkpoint_note = QLabel("")
        self._checkpoint_note.setWordWrap(True)
        outer.addWidget(self._checkpoint_note)
        self._refresh_checkpoint_note()
        self._variant.currentTextChanged.connect(
            lambda _t: self._refresh_checkpoint_note()
        )

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self.add_content(container)

    # -- accessors used by the handler -------------------------------------

    def selected_sources(self) -> list:
        rows = [i.row() for i in self._list.selectedIndexes()]
        return [self._sources[r] for r in sorted(rows)]

    def selected_variant(self) -> str:
        return self._variant.currentText()

    def prompt(self) -> str:
        return self._prompt.text().strip()

    def reference_body_px(self) -> float:
        return float(self._reference_body.value())

    def parameters(self) -> dict:
        return {
            "confidence": float(self._confidence.value()),
            "max_instances": int(self._max_instances.value()),
            "overlap": float(self._overlap.value()),
            "seam_margin_px": float(self._seam_margin.value()),
            "merge_iou": float(self._merge_iou.value()),
            "reference_body_px": self.reference_body_px(),
            "tile_fraction": self.tile_fraction(),
        }

    def _refresh_tile_label(self) -> None:
        body_px = self.reference_body_px()
        tile_px = resolve_tile_px(body_px, self.tile_fraction())
        if tile_px:
            self._tile_label.setText(
                f"{tile_px} px (reference body size "
                f"{body_px:.0f} px / "
                f"{self.tile_fraction():.2f})"
            )
        elif self.tile_fraction() is None:
            self._tile_label.setText("full frame — tiling off by choice.")
        else:
            self._tile_label.setText(
                "full frame — no reference body size is known, so tiling is off. "
                "Enter one above (or set one in project settings) for much "
                "better small-object recall."
            )

    def _project_frame_count(self) -> int:
        """Images across the selected sources — the run-time projection base."""
        from hydra_suite.detectkit.gui.constants import IMG_EXTS

        total = 0
        for src in self.selected_sources() or self._sources:
            images = Path(src.path) / "images"
            if images.is_dir():
                total += sum(
                    1 for p in images.rglob("*") if p.suffix.lower() in IMG_EXTS
                )
        return total

    def tile_fraction(self) -> float | None:
        value = float(self._tile_fraction.value())
        return None if value <= 0.0 else value

    def apply_calibration_choice(self, point) -> None:
        """Write a chosen frontier point back into the dialog's controls."""
        self._confidence.setValue(float(point.confidence))
        self._tile_fraction.setValue(
            0.0 if point.tile_fraction is None else float(point.tile_fraction)
        )
        self._refresh_tile_label()

    def set_calibration_enabled(self, enabled: bool, reason: str = "") -> None:
        self._btn_calibrate.setEnabled(enabled)
        if not enabled and reason:
            self._btn_calibrate.setToolTip(reason)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def _refresh_calibration_enabled(self) -> None:
        from hydra_suite.detectkit.jobs.semantic_escalation import has_labelled_frames

        # Calibration works at ANY geometry level -- it needs instance COUNTS,
        # not masks -- so OBB and AABB sources qualify too. has_labelled_frames
        # is a label-FILE scan: the old check decoded every labelled image
        # on the GUI thread just to answer a yes/no.
        has_labels = any(has_labelled_frames(s) for s in self._sources)
        self.set_calibration_enabled(
            has_labels,
            "No labelled frames in these sources. Label a few (any geometry "
            "level) to calibrate the threshold to your data — or proceed and "
            "tune it by eye.",
        )

    def _run_calibration(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.core.inference.semantic.calibration import recommend
        from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (
            CalibrationResultsDialog,
        )
        from hydra_suite.detectkit.jobs.semantic_escalation import (
            CalibrationWorker,
            labelled_frames_for,
        )

        if not self._exhaustive.isChecked():
            QMessageBox.information(
                self,
                "Calibrate",
                "Confirm your labelled frames are exhaustively labelled first. "
                "An unlabelled real animal counts as a false positive and biases "
                "the recommended threshold upward.",
            )
            return
        if not self.confirm_checkpoint():
            return
        frames = [
            f
            for s in self.selected_sources() or self._sources
            for f in labelled_frames_for(s)
        ]
        if not frames:
            QMessageBox.information(self, "Calibrate", "No labelled frames found.")
            return

        progress = QProgressDialog("Calibrating…", "Cancel", 0, 100, self)
        progress.setMinimumDuration(0)
        worker = CalibrationWorker(
            frames, self.prompt(), self.selected_variant(), self.parameters()
        )
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)

        def _done(points) -> None:
            progress.close()
            self.calibration_points = points
            best, reason = recommend(points)
            results = CalibrationResultsDialog(
                points,
                best,
                reason,
                project_frames=self._project_frame_count(),
                # F6: a cancelled sweep must SAY it is partial.
                partial=worker.cancelled,
                parent=self,
            )
            results.exec()
            chosen = results.chosen()
            if chosen is None:
                self.set_status(reason or "Calibration finished; no point chosen.")
                return
            self.apply_calibration_choice(chosen)
            tile_desc = (
                "full frame"
                if chosen.tile_fraction is None
                else f"tile fraction {chosen.tile_fraction:.2f} "
                f"({chosen.tile_px} px, {chosen.tiles_per_frame} tiles/frame)"
            )
            self.set_status(
                f"Using {tile_desc} at confidence {chosen.confidence:.2f}: misses "
                f"{chosen.missed_per_frame:.1f} animal(s)/frame, leaves "
                f"{chosen.extra_per_frame:.1f} polygon(s)/frame to delete "
                f"(recall {chosen.recall:.1%}, {chosen.n_matched} matched, "
                f"{chosen.seconds_per_frame:.1f} s/frame measured here)."
            )

        worker.result_ready.connect(_done)
        worker.finished.connect(progress.close)
        self._calibration_worker = worker  # keep a reference alive
        worker.start()

    # -- checkpoint download, surfaced before anything runs -----------------

    def _refresh_checkpoint_note(self) -> None:
        avail = probe_availability(self.selected_variant())
        if avail.checkpoint_missing:
            self._checkpoint_note.setText(
                f"⚠ The {self.selected_variant()} checkpoint "
                f"(~{CHECKPOINT_SIZE_GB:.2f} GB) is not on this machine yet. It "
                "will be downloaded once, after you confirm, before the run "
                "starts."
            )
        elif not avail.usable:
            self._checkpoint_note.setText(f"⚠ {avail.reason}")
        else:
            self._checkpoint_note.setText("")

    def confirm_checkpoint(self) -> bool:
        """Ask before a 3.45 GB download. True = go ahead."""
        avail = probe_availability(self.selected_variant())
        if avail.usable:
            return True
        if not avail.checkpoint_missing:
            QMessageBox.warning(self, "Semantic escalation", avail.reason)
            return False
        reply = QMessageBox.question(
            self,
            "Download the SAM3 checkpoint?",
            f"The {self.selected_variant()} checkpoint (~{CHECKPOINT_SIZE_GB:.2f} "
            "GB) has not been downloaded yet.\n\nIt will be downloaded once "
            "and cached; the run cannot start without it. Download now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    # -- one-tile preview ---------------------------------------------------

    def _run_preview(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.detectkit.jobs.semantic_escalation import TilePreviewWorker

        if not self.prompt():
            QMessageBox.information(self, "Preview one tile", "Enter a prompt first.")
            return
        sources = self.selected_sources() or self._sources
        if not sources:
            QMessageBox.information(self, "Preview one tile", "Select a source.")
            return
        if not self.confirm_checkpoint():
            return

        frames = self._project_frame_count()
        progress = QProgressDialog("Running one tile…", None, 0, 0, self)
        progress.setMinimumDuration(0)
        self._btn_preview.setEnabled(False)

        def _done(res) -> None:
            progress.close()
            tile_desc = (
                "the full frame" if res.tile_px is None else f"a {res.tile_px} px tile"
            )
            # Every number below is MEASURED on the tile just run; nothing
            # here is a hardcoded timing figure.
            projection = ""
            if frames:
                total_s = res.seconds * res.tiles_per_frame * frames
                projection = (
                    f"\n\nAt that rate, {res.tiles_per_frame} tile(s)/frame x "
                    f"{frames} frame(s) projects to about "
                    f"{_humanise_seconds(total_s)} for the full run."
                )
            self.set_status(
                f"Preview: {res.instances} instance(s) in {tile_desc} of "
                f"{res.image_name}, in {res.seconds:.1f} s (measured)." + projection
            )
            QMessageBox.information(
                self,
                "Preview one tile",
                f"{res.instances} instance(s) found in {tile_desc} of "
                f"{res.image_name}.\n\nMeasured: {res.seconds:.1f} s for that "
                f"one tile." + projection,
            )

        def _failed(msg: str) -> None:
            progress.close()
            QMessageBox.warning(self, "Preview one tile", msg)

        worker = TilePreviewWorker(
            sources[0], self.prompt(), self.selected_variant(), self.parameters()
        )
        worker.result_ready.connect(_done)
        worker.error.connect(_failed)
        worker.finished.connect(lambda: self._btn_preview.setEnabled(True))
        worker.finished.connect(progress.close)
        self._preview_worker = worker  # keep a reference alive
        worker.start()

    def accept(self) -> None:  # noqa: D102
        if not self.prompt():
            QMessageBox.warning(self, "Semantic escalation", "Enter a prompt first.")
            return
        if not self.selected_sources():
            QMessageBox.warning(self, "Semantic escalation", "Select a source.")
            return
        if not self.confirm_checkpoint():
            return
        super().accept()
