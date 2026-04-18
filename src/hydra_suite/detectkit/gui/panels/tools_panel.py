"""DetectKit ToolsPanel — fixed 280px right panel with 4 collapsible groups."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

logger = logging.getLogger(__name__)

_PANEL_WIDTH = 280


class OverlaySettings(NamedTuple):
    """Overlay display settings passed from ToolsPanel to MainWindow."""

    show_gt: bool
    show_pred: bool
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
    """Fixed-width right panel with Dataset Overview, Analysis, Overlay, Navigation."""

    overlay_settings_changed = Signal()
    prev_requested = Signal()
    next_requested = Signal()
    train_requested = Signal()
    evaluate_requested = Signal()
    history_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proj = None
        self._class_checkboxes: list[QCheckBox] = []
        self.setFixedWidth(_PANEL_WIDTH)
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
        layout.addWidget(self._build_analysis_section())
        layout.addWidget(self._build_overlay_group())
        layout.addWidget(self._build_navigation_group())
        layout.addStretch(1)

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
        self._metrics_view.setPlaceholderText("Run training to see model metrics.")
        self._metrics_view.setMaximumHeight(120)
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

        v.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._model_combo.currentIndexChanged.connect(self._emit_overlay_changed)
        v.addWidget(self._model_combo)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self._conf_label = QLabel("0.50")
        conf_row.addWidget(self._conf_label)
        v.addLayout(conf_row)

        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(0, 100)
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

        return box

    def _build_navigation_group(self) -> QGroupBox:
        box = QGroupBox("Navigation & Actions")
        v = QVBoxLayout(box)

        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_next = QPushButton("Next ▶")
        self._btn_prev.setProperty("detectkitVariant", "secondary")
        self._btn_next.setProperty("detectkitVariant", "secondary")
        self._btn_prev.clicked.connect(self.prev_requested)
        self._btn_next.clicked.connect(self.next_requested)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        v.addLayout(nav_row)

        self._counter_label = QLabel("0 / 0")
        self._counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._counter_label)

        self._btn_train = QPushButton("Train…")
        self._btn_evaluate = QPushButton("Evaluate…")
        self._btn_history = QPushButton("History…")
        self._btn_history.setProperty("detectkitVariant", "secondary")
        self._btn_train.clicked.connect(self.train_requested)
        self._btn_evaluate.clicked.connect(self.evaluate_requested)
        self._btn_history.clicked.connect(self.history_requested)
        v.addWidget(self._btn_train)
        v.addWidget(self._btn_evaluate)
        v.addWidget(self._btn_history)

        return box

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, proj: "DetectKitProject") -> None:
        """Bind panel to a project and refresh all groups."""
        self._proj = proj
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

    def set_image_counter(self, current: int, total: int) -> None:
        """Update the navigation counter label."""
        self._counter_label.setText(f"{current} / {total}")

    def refresh_model_selector(self, model_paths: list[str]) -> None:
        """Repopulate the model combo box."""
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for path in model_paths:
            self._model_combo.addItem(path)
        self._model_combo.blockSignals(False)

    def update_model_metrics(self, metrics: dict) -> None:
        """Display model metrics in the Analysis section."""
        if not metrics:
            self._metrics_view.setPlainText("No model metrics available yet.")
            return
        lines = [f"{k}: {v}" for k, v in metrics.items()]
        self._metrics_view.setPlainText("\n".join(lines))

    def get_overlay_settings(self) -> OverlaySettings:
        """Return the current overlay display settings."""
        show_gt = self._chk_show_gt.isChecked()
        show_pred = self._chk_show_pred.isChecked()
        confidence = self._conf_slider.value() / 100.0
        active_model = self._model_combo.currentText()

        visible_ids: set[int] = set()
        for chk in self._class_checkboxes:
            if chk.isChecked():
                class_id = chk.property("class_id")
                if class_id is not None:
                    visible_ids.add(int(class_id))

        return OverlaySettings(
            show_gt=show_gt,
            show_pred=show_pred,
            confidence_threshold=confidence,
            visible_class_ids=visible_ids,
            active_model_path=active_model,
        )

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
