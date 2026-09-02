"""DetectKit ToolsPanel — resizable right rail for workspace controls."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..models import DetectKitProject

from ..models import INFERENCE_CONFIDENCE_FLOOR

logger = logging.getLogger(__name__)

_PANEL_MIN_WIDTH = 300


class OverlaySettings(NamedTuple):
    """Overlay display settings passed from ToolsPanel to MainWindow."""

    show_gt: bool
    show_pred: bool
    show_derived_levels: bool
    show_escalation: bool
    confidence_threshold: float
    visible_class_ids: set
    active_model_path: str


class _CollapsibleSection(QWidget):
    """A toggle-button header with a collapsible content area."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toggle_btn = QPushButton(f"▶  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setProperty("detectkitVariant", "quiet")
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; font-weight: 600; padding: 6px 8px; }"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle_btn)

        self._content_area = QWidget()
        self._content_area.setVisible(False)
        self._content_layout = QVBoxLayout(self._content_area)
        self._content_layout.setContentsMargins(8, 4, 0, 4)
        layout.addWidget(self._content_area)

    def set_content(self, widget: QWidget) -> None:
        """Set the collapsible content widget."""
        # Clear old content
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._content_layout.addWidget(widget)

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._content_area.setVisible(checked)
        arrow = "▼" if checked else "▶"
        title = self._toggle_btn.text()[2:].strip()
        self._toggle_btn.setText(f"{arrow}  {title}")

    def toggle(self) -> None:
        """Programmatically toggle the section."""
        self._toggle_btn.setChecked(not self._expanded)
        self._on_toggle(not self._expanded)

    def is_expanded(self) -> bool:
        return self._expanded


class ToolsPanel(QWidget):
    """Resizable right panel with dataset, overlay, and analysis controls."""

    overlay_settings_changed = Signal()
    run_inference_requested = Signal()
    inference_settings_requested = Signal()
    escalate_geometry_requested = Signal()
    semantic_escalation_requested = Signal()
    mark_reviewed_requested = Signal()
    review_escalations_requested = Signal()
    stage_predictions_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proj = None
        self._class_checkboxes: list[QCheckBox] = []
        self._portability_status = "Unknown"
        self._linked_counts: dict[str, int] = {}
        self._active_model_path: str = ""
        self.setMinimumWidth(_PANEL_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setProperty("detectkitRole", "panelShell")
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        header = QLabel("Workspace Tools")
        header.setProperty("detectkitRole", "sectionTitle")
        outer.addWidget(header)

        intro = QLabel(
            "Track dataset readiness, inspect recent metrics, and control preview overlays from a single workspace rail."
        )
        intro.setWordWrap(True)
        intro.setProperty("detectkitRole", "sectionHint")
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_overview_group())
        layout.addWidget(self._build_overlay_group())
        layout.addWidget(self._build_escalation_group())
        layout.addWidget(self._build_analysis_section())
        layout.addStretch(1)

    def _build_escalation_group(self) -> QGroupBox:
        box = QGroupBox("Escalation")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        hint = QLabel(
            "Geometry escalation converts the boxes you already have into "
            "masks — it cannot add an animal that was never labelled. "
            "Semantic escalation finds instances from a prompt, including "
            "animals missing from the current labels, and needs no labels to "
            "start."
        )
        hint.setWordWrap(True)
        hint.setProperty("detectkitRole", "sectionHint")
        v.addWidget(hint)

        self._btn_escalate_sam2 = QPushButton("Geometry escalation (SAM2)…")
        self._btn_escalate_sam2.setToolTip("Convert existing boxes to masks with SAM2.")
        self._btn_escalate_sam2.clicked.connect(self.escalate_geometry_requested)
        try:
            from hydra_suite.core.inference.sam2.checkpoints import available_variants

            available_variants()
        except Exception:  # pragma: no cover - depends on optional SAM2 assets
            self._btn_escalate_sam2.setEnabled(False)
            self._btn_escalate_sam2.setToolTip(
                "SAM2 catalog unavailable — install the SAM2 checkpoints to enable "
                "escalation."
            )
        v.addWidget(self._btn_escalate_sam2)

        self._btn_semantic = QPushButton("Semantic escalation (SAM3)…")
        self._btn_semantic.setToolTip(
            "Find prompted instances and create masks with SAM3."
        )
        self._btn_semantic.clicked.connect(self.semantic_escalation_requested)
        # A REAL probe, not the SAM2 pattern: available_variants() above only
        # lists a static dict, so it would let a click start a silent 3.45 GB
        # download. probe_availability checks deps AND the on-disk checkpoint
        # without ever touching the network.
        try:
            from hydra_suite.core.inference.semantic.checkpoints import (
                probe_availability,
            )

            avail = probe_availability()
        except Exception as exc:  # pragma: no cover - optional SAM3 assets
            from hydra_suite.core.inference.semantic.checkpoints import Sam3Availability

            avail = Sam3Availability(False, str(exc))
        # `usable` alone would disable the button whenever the 3.45 GB
        # checkpoint is absent -- but the only offer to download it is inside
        # the dialog BEHIND this button, so the feature would be unreachable
        # on any machine without a pre-placed checkpoint. A missing checkpoint
        # therefore enables the button and warns; a missing dependency does not.
        if avail.checkpoint_missing:
            self._btn_semantic.setToolTip(
                avail.reason + " Click to configure the run; the dialog asks "
                "before downloading anything."
            )
        elif not avail.usable:
            self._btn_semantic.setEnabled(False)
            self._btn_semantic.setToolTip(avail.reason)
        v.addWidget(self._btn_semantic)

        self._btn_mark_reviewed = QPushButton("Mark reviewed…")
        self._btn_mark_reviewed.clicked.connect(self.mark_reviewed_requested)
        v.addWidget(self._btn_mark_reviewed)

        self._btn_review_escalations = QPushButton("Review escalations…")
        self._btn_review_escalations.clicked.connect(self.review_escalations_requested)
        v.addWidget(self._btn_review_escalations)

        return box

    def _build_overview_group(self) -> QGroupBox:
        box = QGroupBox("Dataset Overview")
        v = QVBoxLayout(box)
        v.setSpacing(4)

        self._overview_progress = QProgressBar()
        self._overview_progress.setRange(0, 100)
        self._overview_progress.setValue(0)
        self._overview_progress.setTextVisible(True)
        self._overview_progress.setFormat("0 / 0 labeled")
        v.addWidget(self._overview_progress)

        self._overview_portability = QLabel("Project portability: unknown")
        self._overview_portability.setWordWrap(True)
        self._overview_portability.setProperty("detectkitRole", "compactInfo")
        v.addWidget(self._overview_portability)

        self._overview_sources_layout = QVBoxLayout()
        self._overview_sources_layout.setSpacing(2)
        v.addLayout(self._overview_sources_layout)

        return box

    def _build_analysis_section(self) -> _CollapsibleSection:
        self._analysis_section = _CollapsibleSection("Analysis")
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(0, 0, 0, 0)

        self._metrics_view = QTextEdit()
        self._metrics_view.setReadOnly(True)
        self._metrics_view.setPlaceholderText(
            "Run inference to see prediction statistics."
        )
        self._metrics_view.setMinimumHeight(140)
        v.addWidget(self._metrics_view)

        self._analysis_section.set_content(content)
        self._analysis_section.toggle()
        return self._analysis_section

    def _build_overlay_group(self) -> QGroupBox:
        box = QGroupBox("Inference Overlay")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        self._chk_show_gt = QCheckBox("Show ground truth")
        self._chk_show_gt.setChecked(True)
        self._chk_show_gt.stateChanged.connect(self._emit_overlay_changed)
        v.addWidget(self._chk_show_gt)

        self._chk_show_pred = QCheckBox("Show predictions")
        self._chk_show_pred.setChecked(True)
        self._chk_show_pred.stateChanged.connect(self._emit_overlay_changed)
        v.addWidget(self._chk_show_pred)

        self._chk_show_derived_levels = QCheckBox("Show derived levels")
        self._chk_show_derived_levels.setChecked(True)
        self._chk_show_derived_levels.setToolTip(
            "Also show geometry levels derived from a source's native level "
            "(e.g. a polygon source's derived OBB and AABB outlines)."
        )
        self._chk_show_derived_levels.stateChanged.connect(self._emit_overlay_changed)
        v.addWidget(self._chk_show_derived_levels)

        self._chk_show_escalation = QCheckBox("Show staged escalation")
        self._chk_show_escalation.setChecked(True)
        self._chk_show_escalation.setToolTip(
            "Overlay the masks a pending SAM3/SAM2 escalation has staged for "
            "the current source, in their own colour, so they can be judged "
            "on the frame before being accepted."
        )
        self._chk_show_escalation.stateChanged.connect(self._emit_overlay_changed)
        v.addWidget(self._chk_show_escalation)

        overlay_hint = QLabel(
            "Choose a model and threshold, then run inference on the current image when you want to refresh predictions."
        )
        overlay_hint.setWordWrap(True)
        overlay_hint.setProperty("detectkitRole", "sectionHint")
        v.addWidget(overlay_hint)

        v.addWidget(QLabel("Model:"))
        model_frame = QFrame()
        model_frame.setFrameShape(QFrame.Shape.StyledPanel)
        model_frame.setFrameShadow(QFrame.Shadow.Sunken)
        model_frame_layout = QVBoxLayout(model_frame)
        model_frame_layout.setContentsMargins(4, 4, 4, 4)
        model_frame_layout.setSpacing(0)
        self._model_display = QLabel("(no model selected)")
        # Checkpoint filenames commonly contain long underscore-delimited run
        # ids. QLabel cannot wrap those tokens, so its minimum-width hint used
        # to widen the scroll content and visibly distort this fixed-width rail.
        # Keep the compact label constrained; the complete path is available
        # on hover.
        self._model_display.setWordWrap(False)
        self._model_display.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        model_frame_layout.addWidget(self._model_display)
        v.addWidget(model_frame)
        model_hint = QLabel(
            "Select a model from the History dialog (double-click a row)."
        )
        model_hint.setWordWrap(True)
        model_hint.setProperty("detectkitRole", "sectionHint")
        v.addWidget(model_hint)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self._conf_label = QLabel("0.50")
        conf_row.addWidget(self._conf_label)
        v.addLayout(conf_row)

        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(round(INFERENCE_CONFIDENCE_FLOOR * 100), 100)
        self._conf_slider.setValue(50)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        v.addWidget(self._conf_slider)

        self._class_filter_label = QLabel("Classes:")
        v.addWidget(self._class_filter_label)
        self._class_checkboxes_widget = QWidget()
        self._class_checkboxes_layout = QVBoxLayout(self._class_checkboxes_widget)
        self._class_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        self._class_checkboxes_layout.setSpacing(2)
        v.addWidget(self._class_checkboxes_widget)

        self._btn_inference_settings = QPushButton("Inference Settings…")
        self._btn_inference_settings.setProperty("detectkitVariant", "quiet")
        self._btn_inference_settings.clicked.connect(self.inference_settings_requested)
        v.addWidget(self._btn_inference_settings)

        self._btn_run_inference = QPushButton("Run Inference")
        self._btn_run_inference.clicked.connect(self.run_inference_requested)
        v.addWidget(self._btn_run_inference)

        self.btn_stage_predictions = QPushButton("Stage Predictions for Review")
        self.btn_stage_predictions.setToolTip(
            "Write the current predictions into a staged review, so they can be "
            "accepted or rejected frame by frame like an escalation result. "
            "Looking at predictions never creates reviewable state; this does."
        )
        self.btn_stage_predictions.clicked.connect(self.stage_predictions_requested)
        v.addWidget(self.btn_stage_predictions)

        return box

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, proj: "DetectKitProject") -> None:
        """Bind panel to a project and refresh all groups."""
        self._proj = proj
        # Project switches must replace, not inherit, session model state.
        # Suppress the overlay signal while MainWindow is still binding the
        # rest of the new project.
        self.set_active_model_path(proj.active_model_path, emit=False)
        self._rebuild_class_checkboxes(proj.class_names)
        self.refresh_overview()

    def refresh_overview(self) -> None:
        """Refresh the Dataset Overview group from the bound project."""
        if self._proj is None:
            return
        # Clear old source rows
        while self._overview_sources_layout.count():
            item = self._overview_sources_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sources = self._proj.sources or []

        if self._portability_status == "Portable":
            self._overview_portability.setText("Project portability: Portable")
        else:
            source_count = int(self._linked_counts.get("sources", 0) or 0)
            artifact_count = int(self._linked_counts.get("artifacts", 0) or 0)
            details: list[str] = []
            if source_count:
                details.append(f"{source_count} linked source(s)")
            if artifact_count:
                details.append(f"{artifact_count} linked artifact(s)")
            suffix = f" ({', '.join(details)})" if details else ""
            self._overview_portability.setText(f"Project portability: Linked{suffix}")

        self._overview_progress.setRange(0, max(1, len(sources)))
        self._overview_progress.setValue(len(sources))

        for src in sources:
            descriptor = src.name or src.path
            if getattr(src, "imported", False) and getattr(src, "source_kind", ""):
                descriptor = f"{descriptor} - imported {src.source_kind}"
            row_lbl = QLabel(descriptor)
            row_lbl.setWordWrap(True)
            row_lbl.setProperty("detectkitRole", "compactInfo")
            self._overview_sources_layout.addWidget(row_lbl)

        n_src = len(sources)
        self._overview_progress.setFormat(f"{n_src} source(s) connected")

    def set_portability_status(
        self,
        status: str,
        linked_counts: dict[str, int] | None = None,
    ) -> None:
        """Update the overview portability state for the bound project."""
        self._portability_status = str(status or "Unknown")
        self._linked_counts = dict(linked_counts or {})
        if self._proj is not None:
            self.refresh_overview()

    def refresh_model_selector(self, model_paths: list[str]) -> None:
        """Update internal model path list; auto-select first when none active."""
        if not self._active_model_path and model_paths:
            self.set_active_model_path(model_paths[0])

    def update_inference_stats(
        self,
        stats: dict,
        *,
        class_names: list[str] | None = None,
    ) -> None:
        """Display per-run inference statistics in the Analysis section."""
        if not stats:
            self._metrics_view.setPlainText(
                "Run inference to see prediction statistics."
            )
            return

        image_count = int(stats.get("image_count", 0))
        detection_count = int(stats.get("detection_count", 0))
        mean_confidence = float(stats.get("mean_confidence", 0.0))
        class_counts = stats.get("class_counts", {}) or {}
        per_image = stats.get("per_image", {}) or {}

        per_image_count = max(1, image_count)
        average_per_image = detection_count / per_image_count
        images_with_detections = sum(1 for dets in per_image.values() if dets)

        lines = [
            f"Images processed:    {image_count:,}",
            f"Images w/ detections: {images_with_detections:,}",
            f"Total detections:    {detection_count:,}",
            f"Detections / image:  {average_per_image:.2f}",
            f"Mean confidence:     {mean_confidence:.3f}",
        ]

        if class_counts:
            lines.append("")
            lines.append("By class:")
            ordered = sorted(class_counts.items(), key=lambda kv: int(kv[0]))
            for class_id, count in ordered:
                cid = int(class_id)
                name = (
                    class_names[cid]
                    if class_names is not None and 0 <= cid < len(class_names)
                    else f"class {cid}"
                )
                lines.append(f"  {cid}: {name} — {int(count):,}")

        self._metrics_view.setPlainText("\n".join(lines))

    def reset_inference_stats(self) -> None:
        """Clear the Analysis stats display."""
        self._metrics_view.clear()

    def get_overlay_settings(self) -> OverlaySettings:
        """Return the current overlay display settings."""
        show_gt = self._chk_show_gt.isChecked()
        show_pred = self._chk_show_pred.isChecked()
        confidence = self._conf_slider.value() / 100.0

        visible_ids: set[int] = set()
        for chk in self._class_checkboxes:
            if chk.isChecked():
                class_id = chk.property("class_id")
                if class_id is not None:
                    visible_ids.add(int(class_id))

        return OverlaySettings(
            show_gt=show_gt,
            show_pred=show_pred,
            show_derived_levels=self._chk_show_derived_levels.isChecked(),
            show_escalation=self._chk_show_escalation.isChecked(),
            confidence_threshold=confidence,
            visible_class_ids=visible_ids,
            active_model_path=self._active_model_path,
        )

    def set_confidence_threshold(self, value: float) -> None:
        """Synchronize the visible confidence control with runtime settings."""
        self._conf_slider.setValue(
            max(
                round(INFERENCE_CONFIDENCE_FLOOR * 100),
                min(100, round(float(value) * 100)),
            )
        )

    def set_active_model_path(
        self,
        primary: str,
        secondary: str | None = None,
        *,
        status: str | None = None,
        emit: bool = True,
    ) -> None:
        """Set the active model path and update the read-only display label."""
        self._active_model_path = str(primary or "").strip()
        if not self._active_model_path:
            self._model_display.setText("(no model selected)")
            self._model_display.setToolTip("")
        elif secondary:
            p_name = Path(primary).name
            s_name = Path(secondary).name
            self._model_display.setText(f"Detect: {p_name}\nOBB: {s_name}")
            self._model_display.setToolTip(
                f"Detect: {primary}\nCrop model: {secondary}"
            )
        else:
            self._model_display.setText(Path(primary).name)
            self._model_display.setToolTip(self._active_model_path)
        if status and self._active_model_path:
            self._model_display.setText(
                f"{self._model_display.text()}\n({str(status).strip()})"
            )
            self._model_display.setToolTip(
                f"{self._model_display.toolTip()}\n{str(status).strip()}"
            )
        if emit:
            self._emit_overlay_changed()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_class_checkboxes(self, class_names: list[str]) -> None:
        """Recreate per-class checkboxes for the class filter."""
        # Clear old
        while self._class_checkboxes_layout.count():
            item = self._class_checkboxes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._class_checkboxes.clear()

        for idx, name in enumerate(class_names):
            chk = QCheckBox(name)
            chk.setChecked(True)
            chk.setProperty("class_id", idx)
            chk.stateChanged.connect(self._emit_overlay_changed)
            self._class_checkboxes_layout.addWidget(chk)
            self._class_checkboxes.append(chk)

    def _on_conf_changed(self, value: int) -> None:
        self._conf_label.setText(f"{value / 100.0:.2f}")
        self._emit_overlay_changed()

    def _emit_overlay_changed(self, *_) -> None:
        self.overlay_settings_changed.emit()
