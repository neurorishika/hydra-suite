"""SemanticEscalationDialog — prompt, tiling, calibration, visual test frame."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
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


def _saved_value(saved: dict, key: str, default, cast):
    """Read a hand-editable persisted setting without making the dialog fragile."""
    try:
        return cast(saved.get(key, default))
    except (TypeError, ValueError):
        return default


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
        project=None,
        persist_callback=None,
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
        self._project = project
        self._persist_callback = persist_callback
        saved = dict(getattr(project, "semantic_escalation_settings", {}) or {})
        # Restored, not rebuilt: refitting the band would need the labelled
        # frames re-read, and a reopened dialog must offer the same gate the
        # last calibration chose.
        self._area_band = (
            _saved_value(saved, "area_min_px2", 0.0, float),
            _saved_value(saved, "area_max_px2", 0.0, float),
        )
        self._saved_calibration = dict(
            getattr(project, "semantic_calibration", {}) or {}
        )
        self.calibration_points = self._restore_calibration_points(
            self._saved_calibration
        )
        self.calibration_preview_frames: list = []
        self._preview_worker = None

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)

        sources_group = QGroupBox("Sources to escalate")
        sources_layout = QVBoxLayout(sources_group)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        selected_names = set(saved.get("source_names") or [])
        for row, src in enumerate(self._sources):
            self._list.addItem(f"{src.name}  ({src.level})")
            if src.name in selected_names:
                self._list.item(row).setSelected(True)
        self._list.setMinimumSize(240, 180)
        sources_layout.addWidget(self._list)
        source_hint = QLabel("Click entries to toggle one or more sources.")
        source_hint.setWordWrap(True)
        sources_layout.addWidget(source_hint)
        top.addWidget(sources_group, 2)

        settings_group = QGroupBox("Run settings")
        form = QGridLayout(settings_group)
        self._settings_grid = form
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        def add_field(row: int, pair: int, label: str, widget: QWidget) -> None:
            column = pair * 2
            form.addWidget(QLabel(label), row, column)
            form.addWidget(widget, row, column + 1)

        self._variant = QComboBox()
        from hydra_suite.core.inference.semantic.checkpoints import available_variants

        self._variant.addItems(available_variants())
        saved_variant = str(saved.get("variant", ""))
        if self._variant.findText(saved_variant) >= 0:
            self._variant.setCurrentText(saved_variant)
        add_field(0, 0, "Model", self._variant)

        self._prompt = QLineEdit(str(saved.get("prompt", "ant") or "ant"))
        self._prompt.setToolTip(
            "A noun phrase. Wording matters far less than tile size — try "
            "variants in the preview if results look wrong."
        )
        add_field(0, 1, "Prompt", self._prompt)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.01, 0.99)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(_saved_value(saved, "confidence", 0.35, float))
        add_field(1, 0, "Confidence", self._confidence)

        self._max_instances = QSpinBox()
        self._max_instances.setRange(0, 10000)
        self._max_instances.setSpecialValueText("unlimited")
        self._max_instances.setValue(_saved_value(saved, "max_instances", 0, int))
        add_field(1, 1, "Max instances/tile", self._max_instances)

        self._overlap = QDoubleSpinBox()
        self._overlap.setRange(0.0, 0.9)
        self._overlap.setSingleStep(0.1)
        self._overlap.setValue(_saved_value(saved, "overlap", DEFAULT_OVERLAP, float))
        add_field(2, 0, "Tile overlap", self._overlap)

        self._seam_margin = QSpinBox()
        self._seam_margin.setRange(0, 64)
        self._seam_margin.setValue(
            _saved_value(saved, "seam_margin_px", int(DEFAULT_SEAM_MARGIN_PX), int)
        )
        add_field(2, 1, "Seam margin (px)", self._seam_margin)

        self._merge_iou = QDoubleSpinBox()
        self._merge_iou.setRange(0.05, 0.95)
        self._merge_iou.setSingleStep(0.05)
        self._merge_iou.setValue(
            _saved_value(saved, "merge_iou", DEFAULT_MERGE_IOU, float)
        )
        add_field(3, 0, "Merge IoU", self._merge_iou)

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
        self._reference_body.setValue(
            _saved_value(
                saved,
                "reference_body_px",
                float(reference_body_px or 0.0),
                float,
            )
        )
        self._reference_body.setToolTip(
            "The typical longest side of one animal, in pixels. Tile size = "
            "this / tile fraction. With no value, tiling is off — which is "
            "the worst measured configuration for small animals."
        )
        self._reference_body.valueChanged.connect(self._refresh_tile_label)
        add_field(3, 1, "Body size (px)", self._reference_body)
        origin_text = (
            "saved from the previous SAM3 dialog"
            if "reference_body_px" in saved
            else body_px_origin or "entered by you"
        )
        self._body_origin_label = QLabel(origin_text)
        self._body_origin_label.setWordWrap(True)
        self._body_origin_label.setToolTip(self._body_origin_label.text())

        # The fraction is a CALIBRATED parameter. The seed is presented as a
        # guess, never as a tuned or recommended value -- it was back-derived
        # from one measured configuration on one dataset.
        self._tile_fraction = QDoubleSpinBox()
        self._tile_fraction.setRange(0.0, 0.90)
        self._tile_fraction.setSingleStep(0.01)
        self._tile_fraction.setDecimals(2)
        self._tile_fraction.setSpecialValueText("full frame (no tiling)")
        self._tile_fraction.setValue(
            _saved_value(saved, "tile_fraction", SEMANTIC_TILE_FRACTION_SEED, float)
        )
        self._tile_fraction.setToolTip(
            "Tile size = reference body size / this fraction. The default is a "
            "starting guess from one dataset, not a tuned value — calibrate "
            "against your own labelled frames to fit it."
        )
        self._tile_fraction.valueChanged.connect(self._refresh_tile_label)
        add_field(4, 0, "Tile fraction", self._tile_fraction)

        self._tile_label = QLabel("")
        self._tile_label.setWordWrap(True)
        self._tile_label.setMinimumWidth(180)
        add_field(4, 1, "Resolved tile", self._tile_label)
        self._refresh_tile_label()

        origin = QLabel(f"Body-size source: {self._body_origin_label.text()}")
        origin.setWordWrap(True)
        origin.setToolTip(self._body_origin_label.toolTip())
        form.addWidget(origin, 5, 0, 1, 4)
        self._body_origin_display = origin
        top.addWidget(settings_group, 5)
        outer.addLayout(top, 1)

        self._exhaustive = QCheckBox(
            "Labelled frames are exhaustive (every animal is marked)"
        )
        self._exhaustive.setChecked(bool(saved.get("exhaustive", False)))
        self._exhaustive.setToolTip(
            "Calibration counts an unlabelled real animal as a false positive, "
            "which biases the recommended threshold upward."
        )
        outer.addWidget(self._exhaustive)

        self._btn_calibrate = QPushButton("Calibrate against labelled frames…")
        self._btn_calibrate.setEnabled(False)
        self._btn_calibrate.clicked.connect(self._run_calibration)
        self._refresh_calibration_enabled()

        self._btn_view_calibration = QPushButton("View saved calibration…")
        self._btn_view_calibration.clicked.connect(self._view_saved_calibration)

        self._btn_preview = QPushButton("Test random image…")
        self._btn_preview.setToolTip(
            "Chooses one random image from the selected sources, processes the "
            "complete image with the current tiling and confidence settings, "
            "then shows a zoomable prediction overlay and measured run-time "
            "estimate. No labels are written."
        )
        self._btn_preview.clicked.connect(self._run_preview)

        actions = QHBoxLayout()
        actions.addWidget(self._btn_calibrate, 2)
        actions.addWidget(self._btn_view_calibration, 1)
        actions.addWidget(self._btn_preview, 1)
        outer.addLayout(actions)

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

        self._refresh_saved_calibration_ui()

        self.add_content(container)
        self.setMinimumSize(720, 500)
        self.resize(820, 560)

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
            # The label-derived size gate from calibration. Not a control:
            # it is FITTED to the user's labels, so there is nothing to
            # type. 0/0 until a frontier point is chosen, which is exactly
            # the ungated behaviour of a run that skipped calibration.
            "area_min_px2": float(self._area_band[0]),
            "area_max_px2": float(self._area_band[1]),
        }

    @staticmethod
    def _restore_calibration_points(record: dict) -> list:
        from hydra_suite.core.inference.semantic.calibration import CalibrationPoint

        points = []
        for raw in record.get("points", []) if isinstance(record, dict) else []:
            try:
                points.append(CalibrationPoint(**dict(raw)))
            except (TypeError, ValueError):
                continue
        return points

    def _settings_payload(self) -> dict:
        return {
            "variant": self.selected_variant(),
            "prompt": self.prompt(),
            "source_names": [src.name for src in self.selected_sources()],
            "exhaustive": self._exhaustive.isChecked(),
            # Persist 0.0 rather than None so QDoubleSpinBox can restore the
            # explicit full-frame choice without special-case coercion.
            "tile_fraction": float(self._tile_fraction.value()),
            **{
                key: value
                for key, value in self.parameters().items()
                if key != "tile_fraction"
            },
        }

    def _persist_settings(self) -> None:
        if self._project is None:
            return
        self._project.semantic_escalation_settings = self._settings_payload()
        if self._persist_callback is not None:
            self._persist_callback()

    def _store_calibration(
        self, points, recommended, reason: str, preview_frames=None
    ) -> None:
        self.calibration_points = list(points)
        self.calibration_preview_frames = list(preview_frames or [])
        # The band belongs to the LABELS, not to the chosen operating point:
        # every point in one sweep shares it. Adopting it here (rather than
        # only in apply_calibration_choice) stops a recalibration that the
        # user closes without picking a point from leaving the PREVIOUS
        # run's band in place, gating new data by old animal sizes.
        for point in points:
            bounds = (
                float(getattr(point, "area_min_px2", 0.0) or 0.0),
                float(getattr(point, "area_max_px2", 0.0) or 0.0),
            )
            if bounds[1] > bounds[0] > 0.0:
                self._area_band = bounds
            break
        if self._project is None:
            self._refresh_saved_calibration_ui()
            return
        recommended_index = -1
        if recommended is not None:
            recommended_index = next(
                (i for i, point in enumerate(points) if point is recommended), -1
            )
        saved = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "variant": self.selected_variant(),
            "prompt": self.prompt(),
            "source_names": [src.name for src in self.selected_sources()],
            "parameters": self.parameters(),
            "reason": str(reason or ""),
            "recommended_index": recommended_index,
            "points": [asdict(point) for point in points],
        }
        if self.calibration_preview_frames:
            from hydra_suite.detectkit.gui.calibration_preview_store import (
                save_calibration_previews,
            )

            saved["preview_artifact"] = save_calibration_previews(
                Path(self._project.project_dir), self.calibration_preview_frames
            )
        self._saved_calibration = saved
        self._project.semantic_calibration = dict(self._saved_calibration)
        self._persist_settings()
        self._refresh_saved_calibration_ui()

    def _saved_recommendation(self):
        index = _saved_value(self._saved_calibration, "recommended_index", -1, int)
        if 0 <= index < len(self.calibration_points):
            return self.calibration_points[index]
        return None

    def _refresh_saved_calibration_ui(self) -> None:
        available = bool(self.calibration_points)
        self._btn_view_calibration.setEnabled(available)
        self._btn_calibrate.setText(
            "Recalibrate labelled frames…"
            if available
            else "Calibrate against labelled frames…"
        )
        if available and not self._status.text():
            created = str(self._saved_calibration.get("created_at", ""))[:10]
            when = f" from {created}" if created else ""
            self.set_status(
                f"Saved calibration{when}: {len(self.calibration_points)} "
                "measured operating point(s)."
            )

    def _show_calibration_results(
        self,
        points,
        recommended,
        reason: str,
        *,
        partial: bool,
        preview_frames=None,
        merge_iou: float | None = None,
    ) -> None:
        from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (
            CalibrationResultsDialog,
        )

        results = CalibrationResultsDialog(
            points,
            recommended,
            reason,
            project_frames=self._project_frame_count(),
            partial=partial,
            preview_frames=preview_frames,
            merge_iou=float(
                self.parameters()["merge_iou"] if merge_iou is None else merge_iou
            ),
            parent=self,
        )
        results.exec()
        chosen = results.chosen()
        if chosen is None:
            self.set_status(reason or "Calibration finished; no point chosen.")
            return
        self.apply_calibration_choice(chosen)
        self._persist_settings()
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

    def _view_saved_calibration(self) -> None:
        if not self.calibration_points:
            return
        if not self.calibration_preview_frames and self._project is not None:
            artifact = str(self._saved_calibration.get("preview_artifact", ""))
            if artifact:
                from hydra_suite.detectkit.gui.calibration_preview_store import (
                    load_calibration_previews,
                )

                self.calibration_preview_frames = load_calibration_previews(
                    Path(self._project.project_dir), artifact
                )
        self._show_calibration_results(
            self.calibration_points,
            self._saved_recommendation(),
            str(self._saved_calibration.get("reason", "")),
            partial=False,
            preview_frames=self.calibration_preview_frames,
            merge_iou=_saved_value(
                dict(self._saved_calibration.get("parameters", {}) or {}),
                "merge_iou",
                self.parameters()["merge_iou"],
                float,
            ),
        )

    def _refresh_tile_label(self) -> None:
        body_px = self.reference_body_px()
        tile_px = resolve_tile_px(body_px, self.tile_fraction())
        if tile_px:
            self._tile_label.setText(
                f"{tile_px} px\n{body_px:.0f} px / {self.tile_fraction():.2f}"
            )
            self._tile_label.setToolTip(
                f"Tile size {tile_px} px = {body_px:.0f} px reference body "
                f"size / {self.tile_fraction():.2f} tile fraction."
            )
        elif self.tile_fraction() is None:
            self._tile_label.setText("full frame — tiling off by choice.")
            self._tile_label.setToolTip("")
        else:
            self._tile_label.setText(
                "full frame — no reference body size is known, so tiling is off. "
                "Enter one above (or set one in project settings) for much "
                "better small-object recall."
            )
            self._tile_label.setToolTip(self._tile_label.text())

    def _project_frame_count(self) -> int:
        """Images across the selected sources — the run-time projection base."""
        return self._frame_count_for(self.selected_sources() or self._sources)

    @staticmethod
    def _frame_count_for(sources) -> int:
        from hydra_suite.detectkit.gui.constants import IMG_EXTS

        total = 0
        for src in sources:
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
        self._area_band = (
            float(getattr(point, "area_min_px2", 0.0) or 0.0),
            float(getattr(point, "area_max_px2", 0.0) or 0.0),
        )
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
        from hydra_suite.detectkit.jobs.semantic_escalation import CalibrationWorker

        if not self._exhaustive.isChecked():
            QMessageBox.information(
                self,
                "Calibrate",
                "Confirm your labelled frames are exhaustively labelled first. "
                "An unlabelled real animal counts as a false positive and biases "
                "the recommended threshold upward.",
            )
            return
        sources = self.selected_sources() or self._sources
        if not sources:
            QMessageBox.information(self, "Calibrate", "No sources selected.")
            return
        if self.calibration_points:
            created = str(self._saved_calibration.get("created_at", ""))[:10]
            suffix = f" from {created}" if created else ""
            reply = QMessageBox.question(
                self,
                "Replace saved calibration?",
                "A saved calibration"
                f"{suffix} already exists. Completing a new calibration will "
                "replace its measured frontier and recommendation. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if not self.confirm_checkpoint():
            return

        self._persist_settings()

        # F4: the progress dialog exists BEFORE any decoding starts. Reading
        # the labelled frames is itself a cv2.imread of every labelled image
        # of every selected source, so it belongs behind this, in the worker,
        # under Cancel -- not on the GUI thread with the window frozen.
        progress = QProgressDialog("Reading labelled frames…", "Cancel", 0, 100, self)
        progress.setWindowTitle("SAM3 calibration")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setModal(True)
        progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        progress.setMinimumWidth(420)
        worker = CalibrationWorker(
            sources, self.prompt(), self.selected_variant(), self.parameters()
        )
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)

        def _done(points) -> None:
            progress.close()
            if not points:
                self.set_status(
                    "Calibration produced nothing: no labelled frames were found "
                    "in the selected source(s), or it was cancelled before the "
                    "first frame finished."
                )
                return
            best, reason = recommend(points)
            # A cancelled/partial sweep is useful to inspect, but must not erase
            # the last complete calibration stored with the project.
            if not worker.cancelled:
                self._store_calibration(
                    points, best, reason, preview_frames=worker.preview_frames
                )
            self._show_calibration_results(
                points,
                best,
                reason,
                partial=worker.cancelled,
                preview_frames=worker.preview_frames,
            )

        worker.result_ready.connect(_done)
        worker.finished.connect(progress.close)
        self._calibration_worker = worker  # keep a reference alive
        progress.show()
        progress.raise_()
        progress.activateWindow()
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

    # -- random complete-frame preview -------------------------------------

    def _run_preview(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.detectkit.jobs.semantic_escalation import FramePreviewWorker

        if not self.prompt():
            QMessageBox.information(self, "Test random image", "Enter a prompt first.")
            return
        sources = self.selected_sources()
        if not sources:
            QMessageBox.information(self, "Test random image", "Select a source.")
            return
        if not self.confirm_checkpoint():
            return

        self._persist_settings()

        selected_frames = self._frame_count_for(sources)
        project_frames = self._frame_count_for(self._sources)
        progress = QProgressDialog("Choosing a random image…", "Cancel", 0, 100, self)
        progress.setWindowTitle("SAM3 complete-frame check")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setModal(True)
        self._btn_preview.setEnabled(False)

        def _done(res) -> None:
            progress.close()
            self.set_status(
                f"Tested complete image {res.image_path.name}: "
                f"{len(res.predictions)} prediction(s) in {res.seconds:.1f} s."
            )
            from hydra_suite.detectkit.gui.dialogs.semantic_frame_preview_dialog import (
                SemanticFramePreviewDialog,
            )

            preview = SemanticFramePreviewDialog(
                res,
                selected_frames=selected_frames,
                project_frames=project_frames,
                parent=self,
            )
            preview.exec()

        def _failed(msg: str) -> None:
            progress.close()
            if worker.cancelled:
                self.set_status("Random image check cancelled.")
                return
            QMessageBox.warning(self, "Test random image", msg)

        worker = FramePreviewWorker(
            sources, self.prompt(), self.selected_variant(), self.parameters()
        )
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)
        worker.result_ready.connect(_done)
        worker.error.connect(_failed)
        worker.finished.connect(lambda: self._btn_preview.setEnabled(True))
        worker.finished.connect(progress.close)
        self._preview_worker = worker  # keep a reference alive
        progress.show()
        progress.raise_()
        progress.activateWindow()
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
        self._persist_settings()
        super().accept()
