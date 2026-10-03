"""Dialog to pick sources, SAM2 variant and tiling for escalate-all.

Left: what to calibrate on (polygon ground truth) and what to escalate.
Right: the SAM2 version, the tile size (SAHI), and a calibration that fits
the tile size to the project's own polygon labels.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.sam2.checkpoints import (
    DEFAULT_VARIANT,
    available_variants,
)
from hydra_suite.core.inference.semantic.tiling import DEFAULT_OVERLAP, resolve_tile_px
from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog
from hydra_suite.detectkit.gui.dialogs.geometry_calibration_results import (
    GeometryCalibrationResults,
)
from hydra_suite.detectkit.gui.widgets.calibration_source_selector import (
    NO_POLYGON_SOURCES,
    CalibrationSourceSelector,
    scale_warning_text,
)
from hydra_suite.widgets.device_combo import DeviceCombo

TITLE = "Escalate to segment (SAM2)"
ALREADY_POLYGON = "This source already contains segmentation polygons."


def _restore_points(record: dict) -> list:
    from hydra_suite.core.inference.sam2.calibration import GeometryCalibrationPoint

    points = []
    for raw in record.get("points", []) if isinstance(record, dict) else []:
        try:
            points.append(GeometryCalibrationPoint(**dict(raw)))
        except (TypeError, ValueError):
            continue
    return points


class EscalateSam2Dialog(DetectKitDialog):
    """Pick which OBB/AABB sources to escalate to SAM2 segmentation, and how."""

    def __init__(
        self,
        sources,
        parent=None,
        *,
        project=None,
        reference_body_px: float = 0.0,
        persist_callback=None,
    ) -> None:
        super().__init__(TITLE, parent)
        self._sources = list(sources)
        self._project = project
        self._persist_callback = persist_callback
        self._calibration_worker = None
        self._scale_cache: dict = {}
        saved = dict(getattr(project, "geometry_escalation_settings", {}) or {})
        self._saved_settings = saved

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()

        self._selector = CalibrationSourceSelector(
            self._sources,
            escalation_eligible=lambda s: s.level != "polygon",
            ineligible_reason=ALREADY_POLYGON,
        )
        if saved:
            self._selector.restore(
                saved.get("calibration_source_paths"),
                saved.get("escalation_source_names"),
                saved.get("escalation_source_paths"),
            )
        else:
            # Today's default: every source that is not yet polygon.
            self._selector.select_all_eligible_escalation()
        self._list = self._selector.escalation_list
        top.addWidget(self._selector, 2)

        settings = QGroupBox("SAM2 settings")
        form = QGridLayout(settings)
        form.addWidget(QLabel("SAM2 version"), 0, 0)
        self._variant = QComboBox()
        for v in available_variants():
            self._variant.addItem(v)
        saved_variant = str(saved.get("variant", "") or DEFAULT_VARIANT)
        self._variant.setCurrentText(
            saved_variant
            if self._variant.findText(saved_variant) >= 0
            else DEFAULT_VARIANT
        )
        form.addWidget(self._variant, 0, 1)

        form.addWidget(QLabel("Run on"), 1, 0)
        self._device = DeviceCombo(str(saved.get("device", "auto") or "auto"))
        form.addWidget(self._device, 1, 1)

        form.addWidget(QLabel("Body size (px)"), 2, 0)
        self._reference_body = QDoubleSpinBox()
        self._reference_body.setRange(0.0, 4096.0)
        self._reference_body.setDecimals(1)
        self._reference_body.setSingleStep(5.0)
        self._reference_body.setSpecialValueText("unknown (tiling off)")
        self._reference_body.setValue(
            float(saved.get("reference_body_px", reference_body_px) or 0.0)
        )
        self._reference_body.setToolTip(
            "The typical longest side of one animal, in pixels. Tile size = "
            "this / tile fraction."
        )
        form.addWidget(self._reference_body, 2, 1)

        form.addWidget(QLabel("Tile fraction"), 3, 0)
        self._tile_fraction = QDoubleSpinBox()
        self._tile_fraction.setRange(0.0, 0.9)
        self._tile_fraction.setSingleStep(0.05)
        self._tile_fraction.setDecimals(3)
        self._tile_fraction.setSpecialValueText("full frame (no tiling)")
        self._tile_fraction.setToolTip(
            "SAM2 segments each box inside a tile of body size / this "
            "fraction, so small animals are not shrunk to a few pixels. "
            "Full frame is the uncalibrated default; calibrate to fit it."
        )
        form.addWidget(self._tile_fraction, 3, 1)

        form.addWidget(QLabel("Resolved tile"), 4, 0)
        self._tile_label = QLabel("")
        self._tile_label.setWordWrap(True)
        form.addWidget(self._tile_label, 4, 1)

        self._btn_calibrate = QPushButton("Calibrate against polygon frames…")
        self._btn_calibrate.clicked.connect(self._run_calibration)
        form.addWidget(self._btn_calibrate, 5, 0, 1, 2)

        self._results = GeometryCalibrationResults()
        self._results.setMinimumHeight(140)
        self._results.point_chosen.connect(self.apply_calibration_choice)
        form.addWidget(self._results, 6, 0, 1, 2)
        top.addWidget(settings, 3)
        outer.addLayout(top, 1)

        self._scale_note = QLabel("")
        self._scale_note.setWordWrap(True)
        outer.addWidget(self._scale_note)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._tile_fraction.valueChanged.connect(self._refresh_tile_label)
        self._reference_body.valueChanged.connect(self._refresh_tile_label)
        self._variant.currentTextChanged.connect(self._load_saved_calibration)
        self._selector.calibration_changed.connect(self._refresh_calibration_enabled)
        self._selector.calibration_changed.connect(self._refresh_scale_warning)
        self._selector.escalation_changed.connect(self._refresh_scale_warning)

        self._load_saved_calibration()
        self._refresh_tile_label()
        self._refresh_calibration_enabled()
        self._refresh_scale_warning()

        self.add_content(container)
        self.setMinimumSize(760, 480)

    # -- legacy accessors used by the handler and tests ---------------------

    def _variant_combo_items(self) -> list[str]:
        return [self._variant.itemText(i) for i in range(self._variant.count())]

    def selectable_source_names(self) -> list[str]:
        return [
            self._selector.source_for(item).name
            for item in self._selector.escalation_items()
            if item.flags() & Qt.ItemFlag.ItemIsEnabled
        ]

    def preselect_source(self, name: str) -> None:
        """Select only the named source (used when launched from a role block)."""
        self._selector.select_escalation_source(name)

    def selected_variant(self) -> str:
        return self._variant.currentText()

    def selected_device(self) -> str:
        return self._device.device()

    def selected_device(self) -> str:
        return self._device.device()

    def selected_sources(self) -> list[str]:
        return [s.name for s in self._selector.escalation_sources()]

    def selected_source_paths(self) -> list[str]:
        """Stable source identities corresponding to the selected rows."""
        return [
            str(getattr(s, "path", "") or s.name)
            for s in self._selector.escalation_sources()
        ]

    # -- tiling --------------------------------------------------------------

    def tile_fraction(self) -> float | None:
        value = float(self._tile_fraction.value())
        return None if value <= 0.0 else value

    def tiling_parameters(self) -> dict:
        return {
            "reference_body_px": float(self._reference_body.value()),
            "tile_fraction": self.tile_fraction(),
            "overlap": DEFAULT_OVERLAP,
        }

    def _refresh_tile_label(self) -> None:
        body_px = float(self._reference_body.value())
        tile_px = resolve_tile_px(body_px, self.tile_fraction())
        if tile_px:
            self._tile_label.setText(
                f"{tile_px} px ({body_px:.0f} px / {self.tile_fraction():g})"
            )
        elif self.tile_fraction() is None:
            self._tile_label.setText("full frame — tiling off.")
        else:
            self._tile_label.setText(
                "full frame — no body size is known, so tiling is off."
            )

    def _refresh_scale_warning(self) -> None:
        self._scale_note.setText(
            scale_warning_text(
                self._scale_cache,
                self._selector.calibration_sources(),
                self._selector.escalation_sources(),
            )
        )

    # -- calibration ---------------------------------------------------------

    def _refresh_calibration_enabled(self) -> None:
        if not self._selector.has_calibration_sources():
            self._btn_calibrate.setEnabled(False)
            self._btn_calibrate.setToolTip(NO_POLYGON_SOURCES)
            return
        enabled = bool(self._selector.calibration_sources())
        self._btn_calibrate.setEnabled(enabled)
        self._btn_calibrate.setToolTip(
            "" if enabled else 'Select at least one source under "Calibrate on".'
        )

    def _saved_record(self) -> dict:
        store = getattr(self._project, "geometry_calibration", None) or {}
        return dict(store.get(self.selected_variant(), {}) or {})

    def _load_saved_calibration(self) -> None:
        """Show THIS variant's calibration and set the tile fraction for it.

        A calibration belongs to one SAM2 variant, so switching variant must
        never carry another variant's fraction over. Precedence: the
        fraction the user last accepted for this variant, then this
        variant's calibrated choice, then full frame (the uncalibrated
        default).
        """
        from hydra_suite.core.inference.sam2.calibration import recommend_geometry

        record = self._saved_record()
        points = _restore_points(record)
        index = int(record.get("chosen_index", record.get("recommended_index", -1)))
        chosen = points[index] if 0 <= index < len(points) else None
        recommended, _reason = recommend_geometry(points) if points else (None, "")
        self._results.set_points(points, recommended)
        saved = self._saved_settings
        if saved.get("variant") == self.selected_variant() and "tile_fraction" in saved:
            self._tile_fraction.setValue(float(saved.get("tile_fraction") or 0.0))
            self._status.setText("")
        elif chosen is not None:
            self.apply_calibration_choice(chosen)
        else:
            self._tile_fraction.setValue(0.0)
            self._status.setText("")
        if points:
            created = str(record.get("created_at", ""))[:10]
            self._status.setText(
                f"Saved calibration{f' from {created}' if created else ''}: "
                f"{len(points)} measured tile size(s)."
            )
        self._refresh_tile_label()

    def apply_calibration_choice(self, point) -> None:
        self._tile_fraction.setValue(
            0.0 if point.tile_fraction is None else float(point.tile_fraction)
        )
        self._refresh_tile_label()
        self._status.setText(
            f"Using {'full frame' if point.tile_fraction is None else f'{point.tile_px} px tiles'}: "
            f"median IoU {point.median_iou:.3f} against your polygons, "
            f"{point.seconds_per_frame:.2f} s/frame measured here."
        )

    def _run_calibration(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.core.inference.sam2.calibration import recommend_geometry
        from hydra_suite.detectkit.jobs.sam2_escalation import Sam2CalibrationWorker

        sources = self._selector.calibration_sources()
        if not sources:
            self._status.setText('Select a source under "Calibrate on".')
            return
        params = self.tiling_parameters()
        if params["reference_body_px"] <= 0:
            self._status.setText(
                "Enter a body size first: tile sizes are measured relative to it."
            )
            return
        progress = QProgressDialog("Reading polygon frames…", "Cancel", 0, 100, self)
        progress.setWindowTitle("SAM2 calibration")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        worker = Sam2CalibrationWorker(
            sources,
            self.selected_variant(),
            reference_body_px=params["reference_body_px"],
            overlap=params["overlap"],
            device=self.selected_device(),
        )
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)

        def _done(points) -> None:
            # Read BEFORE close(): QProgressDialog.close() emits `canceled`,
            # wired to worker.cancel, which made every sweep look cancelled.
            cancelled = worker.cancelled
            progress.close()
            if not points:
                self._status.setText("Calibration cancelled; nothing was saved.")
                return
            best, reason = recommend_geometry(points)
            self._results.set_points(points, best)
            if best is not None:
                self.apply_calibration_choice(best)
            else:
                self._status.setText(reason)
            if not cancelled:
                self._store_calibration(points, best, reason, worker, sources)

        def _failed(message: str) -> None:
            progress.close()
            QMessageBox.warning(self, "SAM2 calibration", message)

        worker.result_ready.connect(_done)
        worker.error.connect(_failed)
        worker.finished.connect(progress.close)
        self._calibration_worker = worker
        progress.show()
        worker.start()

    def _store_calibration(self, points, best, reason, worker, sources) -> None:
        if self._project is None:
            return
        index = next((i for i, p in enumerate(points) if p is best), -1)
        store = dict(getattr(self._project, "geometry_calibration", {}) or {})
        store[self.selected_variant()] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "points": [asdict(p) for p in points],
            "recommended_index": index,
            "chosen_index": index,
            "reason": str(reason or ""),
            "calibration_source_paths": self._selector.state()[
                "calibration_source_paths"
            ],
            "reference_body_px": float(self._reference_body.value()),
            "sampled_frames": list(worker.sampled_frames),
        }
        self._project.geometry_calibration = store
        if self._persist_callback is not None:
            self._persist_callback()

    # -- accept --------------------------------------------------------------

    def _persist_settings(self) -> None:
        if self._project is None:
            return
        self._project.geometry_escalation_settings = {
            **self._selector.state(),
            "variant": self.selected_variant(),
            "device": self.selected_device(),
            "reference_body_px": float(self._reference_body.value()),
            "tile_fraction": float(self._tile_fraction.value()),
        }
        if self._persist_callback is not None:
            self._persist_callback()

    def accept(self) -> None:  # noqa: D102
        self._persist_settings()
        super().accept()
