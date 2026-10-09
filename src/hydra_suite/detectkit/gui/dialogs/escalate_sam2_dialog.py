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
from hydra_suite.core.inference.semantic.tiling import DEFAULT_OVERLAP
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
from hydra_suite.widgets.slice_settings import (
    SliceSettingsWidget,
    SliceWidgetCapabilities,
)

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


def _format_tile_label(
    tile_px: int | None, body_px: float, fraction: float | None
) -> tuple[str, str]:
    """This dialog's resolved-tile wording (the shared widget renders it)."""
    if tile_px:
        return f"{tile_px} px ({body_px:.0f} px / {fraction:g})", ""
    if fraction is None:
        return "full frame — tiling off.", ""
    return "full frame — no body size is known, so tiling is off.", ""


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
        # Per-variant (fraction, body) the user left in THIS dialog session,
        # so switching SAM2 version and back never loses a fresh calibration
        # or a row the user picked.
        self._session_tiling: dict[str, tuple[float, float]] = {}

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

        # Tiling rows come from the shared SAHI widget (S4); the old
        # attribute names alias its controls. SAM2 owner tiles overlap by the
        # fixed DEFAULT_OVERLAP, shown disabled.
        self._tiling = SliceSettingsWidget(
            role="escalate_sam2",
            title="Tiling (SAHI)",
            capabilities=SliceWidgetCapabilities(
                advanced_merge=False,
                tile_label_formatter=_format_tile_label,
                fixed_overlap=DEFAULT_OVERLAP,
            ),
        )
        self._reference_body = self._tiling.spin_slice_body
        self._tile_fraction = self._tiling.spin_slice_object_fraction
        self._tile_label = self._tiling.lbl_slice_tile_size
        self._tile_fraction.setToolTip(
            "SAM2 segments each box inside a tile of body size / this "
            "fraction, so small animals are not shrunk to a few pixels. "
            "Full frame is the uncalibrated default; calibrate to fit it."
        )
        body = max(0.0, float(saved.get("reference_body_px", reference_body_px) or 0.0))
        self._tiling.set_reference_body(
            body, "user" if "reference_body_px" in saved else "dataset"
        )
        form.addWidget(self._tiling, 2, 0, 1, 2)

        self._btn_calibrate = QPushButton("Calibrate against polygon frames…")
        self._btn_calibrate.clicked.connect(self._run_calibration)
        form.addWidget(self._btn_calibrate, 3, 0, 1, 2)

        self._results = GeometryCalibrationResults()
        self._results.setMinimumHeight(140)
        self._results.point_chosen.connect(self._on_point_chosen)
        form.addWidget(self._results, 4, 0, 1, 2)
        top.addWidget(settings, 3)
        outer.addLayout(top, 1)

        self._scale_note = QLabel("")
        self._scale_note.setWordWrap(True)
        outer.addWidget(self._scale_note)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._current_variant = self.selected_variant()
        self._variant.currentTextChanged.connect(self._on_variant_changed)
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
        self._tiling.refresh()

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

    def _on_variant_changed(self, variant: str) -> None:
        self._session_tiling[self._current_variant] = (
            float(self._tile_fraction.value()),
            float(self._reference_body.value()),
        )
        self._current_variant = variant
        self._load_saved_calibration()

    def _load_saved_calibration(self) -> None:
        """Show THIS variant's calibration and set the tiling for it.

        A calibration belongs to one SAM2 variant, so switching variant never
        carries another variant's fraction over. Precedence: what the user
        left for this variant earlier in this dialog session, then
        ``default_geometry_tiling`` -- the same answer the headless command
        uses (accepted settings, then the calibrated choice, then full frame).
        """
        from hydra_suite.core.inference.sam2.calibration import recommend_geometry
        from hydra_suite.detectkit.jobs.sam2_escalation import default_geometry_tiling

        record = self._saved_record()
        points = _restore_points(record)
        recommended, _reason = recommend_geometry(points) if points else (None, "")
        self._results.set_points(points, recommended)
        self._status.setText("")
        variant = self.selected_variant()
        if variant in self._session_tiling:
            fraction, body = self._session_tiling[variant]
            source = "user"
        else:
            default = default_geometry_tiling(self._project, variant)
            fraction = float(default["tile_fraction"] or 0.0)
            body = float(default["reference_body_px"])
            accepted = dict(
                getattr(self._project, "geometry_escalation_settings", {}) or {}
            )
            # Accepted settings are the user's; otherwise the calibration's.
            source = (
                "user"
                if accepted.get("variant") == variant and "tile_fraction" in accepted
                else "profile"
            )
            if fraction <= 0:
                body = 0.0  # keep the prefilled body size; tiling is off anyway
        if fraction > 0:
            self._tiling.set_reference_body(body, source)
        self._tile_fraction.setValue(fraction)
        if points:
            created = str(record.get("created_at", ""))[:10]
            self._status.setText(
                f"Saved calibration{f' from {created}' if created else ''}: "
                f"{len(points)} measured tile size(s)."
            )
        self._refresh_tile_label()

    def _on_point_chosen(self, point) -> None:
        """A row click: adopt it and remember it as this variant's choice."""
        self.apply_calibration_choice(point)
        if self._project is None:
            return
        store = dict(getattr(self._project, "geometry_calibration", {}) or {})
        record = dict(store.get(self.selected_variant(), {}) or {})
        points = self._results.points()
        index = next((i for i, p in enumerate(points) if p is point), -1)
        if not record or index < 0 or record.get("chosen_index") == index:
            return
        record["chosen_index"] = index
        store[self.selected_variant()] = record
        self._project.geometry_calibration = store
        if self._persist_callback is not None:
            self._persist_callback()

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
