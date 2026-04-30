"""Benchmark dialog for TrackerKit runtime and batch recommendations."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.benchmarking import (
    BenchmarkRecommendation,
    BenchmarkResult,
    BenchmarkTargetSpec,
    _PoseBenchmarkBackendCache,
    collect_active_targets,
    derive_benchmark_geometry_from_video,
    lookup_cached_recommendation,
    run_target_benchmark,
    runtime_label,
    store_cached_results,
)
from hydra_suite.widgets.dialogs import BaseDialog
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


def _stream_benchmark_status(message: str) -> None:
    logger.info("TrackerKit benchmark: %s", message)
    print(f"[TrackerKit Benchmark] {message}", flush=True)


class _MetricTile(QFrame):
    """Compact value tile for benchmark geometry inputs."""

    def __init__(self, title: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BenchmarkMetricTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("BenchmarkMetricTitle")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("BenchmarkMetricValue")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)


class _RuntimeSelectionWidget(QWidget):
    """Validated runtime selector that avoids free-text entry."""

    def __init__(self, runtimes: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checkboxes: dict[str, QCheckBox] = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        for index, runtime in enumerate(runtimes):
            checkbox = QCheckBox(runtime_label(runtime))
            checkbox.setChecked(True)
            checkbox.setToolTip(runtime)
            layout.addWidget(checkbox, index // 3, index % 3)
            self._checkboxes[runtime] = checkbox

    def selected_runtimes(self) -> list[str]:
        return [
            runtime
            for runtime, checkbox in self._checkboxes.items()
            if checkbox.isChecked()
        ]


class _BatchSizeSelectionWidget(QWidget):
    """Manage validated batch-size candidates one value at a time."""

    def __init__(self, batch_sizes: list[int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        initial_values = sorted({max(1, int(value)) for value in batch_sizes}) or [1]
        self._default_values = list(initial_values)
        self._values = list(initial_values)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._summary_label = QLabel()
        self._summary_label.setObjectName("BenchmarkInlineHint")
        self._summary_label.setWordWrap(True)
        root.addWidget(self._summary_label)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 1024)
        self._batch_spin.setValue(self._values[-1])
        self._batch_spin.setToolTip("Add one validated batch size at a time.")
        controls.addWidget(self._batch_spin)

        self._add_button = QPushButton("Add")
        self._add_button.clicked.connect(self._add_current_value)
        controls.addWidget(self._add_button)

        self._selected_combo = QComboBox()
        self._selected_combo.setToolTip("Current batch-size candidates.")
        self._selected_combo.setMinimumWidth(92)
        controls.addWidget(self._selected_combo, 1)

        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(self._remove_selected_value)
        controls.addWidget(self._remove_button)

        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset_defaults)
        controls.addWidget(self._reset_button)

        root.addLayout(controls)
        self._refresh_ui()

    def values(self) -> list[int]:
        return list(self._values)

    def _add_current_value(self) -> None:
        value = int(self._batch_spin.value())
        if value not in self._values:
            self._values.append(value)
            self._values.sort()
        self._refresh_ui(selected_value=value)

    def _remove_selected_value(self) -> None:
        if len(self._values) <= 1:
            return
        value = self._selected_combo.currentData()
        if value in self._values:
            self._values.remove(int(value))
        next_value = self._values[
            min(self._selected_combo.currentIndex(), len(self._values) - 1)
        ]
        self._refresh_ui(selected_value=next_value)

    def _reset_defaults(self) -> None:
        self._values = list(self._default_values)
        self._refresh_ui(selected_value=self._values[0])

    def _refresh_ui(self, selected_value: int | None = None) -> None:
        self._summary_label.setText(
            "Selected: " + ", ".join(str(value) for value in self._values)
        )
        self._selected_combo.blockSignals(True)
        self._selected_combo.clear()
        for value in self._values:
            self._selected_combo.addItem(str(value), value)
        self._selected_combo.blockSignals(False)
        if selected_value is not None:
            index = self._selected_combo.findData(selected_value)
            if index >= 0:
                self._selected_combo.setCurrentIndex(index)
        self._remove_button.setEnabled(len(self._values) > 1)
        self._reset_button.setEnabled(self._values != self._default_values)


class _BenchmarkWorker(BaseWorker):
    """Run TrackerKit model benchmarks off the UI thread."""

    completed: Signal = Signal(object)

    def __init__(
        self,
        targets: list[BenchmarkTargetSpec],
        geometry,
        *,
        warmup: int,
        iterations: int,
    ) -> None:
        super().__init__()
        self._targets = targets
        self._geometry = geometry
        self._warmup = int(warmup)
        self._iterations = int(iterations)
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def execute(self) -> None:
        total = sum(
            max(
                1,
                len(target.runtimes)
                * len(target.batch_sizes)
                * (
                    len(target.individual_batch_sizes)
                    if target.individual_batch_sizes
                    else 1
                ),
            )
            for target in self._targets
        )
        completed = 0
        results: dict[str, list[BenchmarkResult]] = {}
        pose_backend_cache = _PoseBenchmarkBackendCache()
        try:
            for target in self._targets:
                target_results: list[BenchmarkResult] = []
                for runtime in target.runtimes:
                    for batch_size in target.batch_sizes:
                        individual_batches = target.individual_batch_sizes or [None]
                        for individual_batch_size in individual_batches:
                            if self._cancel_requested:
                                self.completed.emit(
                                    {"cancelled": True, "results": results}
                                )
                                return
                            completed += 1
                            status = (
                                f"Benchmarking {target.label}: {runtime_label(runtime)} @ frame {batch_size} / crop {individual_batch_size}"
                                if individual_batch_size is not None
                                else f"Benchmarking {target.label}: {runtime_label(runtime)} @ batch {batch_size}"
                            )
                            _stream_benchmark_status(status)
                            self.status.emit(status)
                            self.progress.emit(int((completed / max(1, total)) * 100))
                            target_results.append(
                                run_target_benchmark(
                                    target,
                                    self._geometry,
                                    runtime,
                                    batch_size,
                                    individual_batch_size,
                                    status_callback=self.status.emit,
                                    pose_backend_cache=pose_backend_cache,
                                    warmup=self._warmup,
                                    iterations=self._iterations,
                                )
                            )
                results[target.key] = target_results
        finally:
            pose_backend_cache.close()
        self.completed.emit({"cancelled": False, "results": results})


class TrackerBenchmarkDialog(BaseDialog):
    """Dialog for benchmarking the current TrackerKit model configuration."""

    def __init__(self, main_window: Any, parent: QWidget | None = None) -> None:
        super().__init__(
            title="Benchmark Runtime Recommendations",
            parent=parent,
            buttons=QDialogButtonBox.NoButton,
            apply_dark_style=True,
        )
        self.setMinimumSize(980, 680)
        self._main_window = main_window
        self._worker: _BenchmarkWorker | None = None
        self._targets, self._collection_notes = collect_active_targets(main_window)
        self._geometry = derive_benchmark_geometry_from_video(
            main_window._setup_panel.file_line.text().strip(),
            resize_factor=float(main_window._setup_panel.spin_resize.value()),
            reference_body_size=float(
                main_window._detection_panel.spin_reference_body_size.value()
            ),
            reference_aspect_ratio=float(
                main_window._detection_panel.spin_reference_aspect_ratio.value()
            ),
            padding_fraction=float(
                main_window._identity_panel.spin_individual_padding.value()
            ),
        )
        self._row_widgets: dict[str, dict[str, Any]] = {}
        self._result_payload: dict[str, list[BenchmarkResult]] = {}
        self._recommendations: dict[str, BenchmarkRecommendation] = {}
        self._active_targets: list[BenchmarkTargetSpec] = []
        self._target_labels = {target.key: target.label for target in self._targets}
        self._target_specs = {target.key: target for target in self._targets}
        self._applied = False
        self._apply_custom_styles()
        self._build_ui()

    def recommendations(self) -> dict[str, BenchmarkRecommendation]:
        return dict(self._recommendations)

    def _load_cached_recommendations(self) -> dict[str, BenchmarkRecommendation]:
        recommendations: dict[str, BenchmarkRecommendation] = {}
        realtime_enabled = self._main_window._is_realtime_tracking_mode_enabled()
        for target in self._targets:
            recommendation = lookup_cached_recommendation(
                target,
                self._geometry,
                realtime_enabled=realtime_enabled,
            )
            if recommendation is not None:
                recommendations[target.key] = recommendation
        return recommendations

    def _apply_custom_styles(self) -> None:
        self.setStyleSheet(self.styleSheet() + """
QWidget#BenchmarkDialogRoot {
    background: transparent;
}
QFrame#BenchmarkMetricTile {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 10px;
}
QLabel#BenchmarkMetricTitle {
    color: #ffffff;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
QLabel#BenchmarkMetricValue {
    color: #ffffff;
    font-size: 16px;
    font-weight: 600;
}
QLabel#BenchmarkSectionHint {
    color: #ffffff;
    font-size: 11px;
}
QFrame#BenchmarkNoticeCard {
    background-color: #5a5032;
    border: 1px solid #8c7a45;
    border-radius: 8px;
}
QLabel#BenchmarkNoticeText {
    color: #ffffff;
    font-size: 11px;
}
QFrame#BenchmarkTargetCard {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 12px;
}
QFrame#BenchmarkTargetCard:hover {
    border-color: #0e639c;
}
QFrame#BenchmarkTargetCard[targetEnabled="false"] {
    border-color: #3e3e42;
    background-color: #202020;
}
QLabel#BenchmarkTargetTitle {
    color: #ffffff;
    font-size: 14px;
    font-weight: 600;
}
QLabel#BenchmarkTargetSubtitle {
    color: #ffffff;
    font-size: 11px;
}
QLabel#BenchmarkBadge {
    background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 11px;
    color: #ffffff;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}
QLabel#BenchmarkFieldLabel {
    color: #ffffff;
    font-size: 11px;
    font-weight: 600;
}
QLabel#BenchmarkInlineHint {
    color: #ffffff;
    font-size: 11px;
}
QFrame#BenchmarkStatusCard {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 10px;
}
QLabel#BenchmarkStatusText {
    color: #ffffff;
    font-size: 12px;
    font-weight: 500;
}
QTableWidget#BenchmarkResultsTable {
    background-color: #252526;
    gridline-color: #3e3e42;
    alternate-background-color: #2d2d30;
}
QTableWidget#BenchmarkResultsTable::item:alternate {
    background-color: #2d2d30;
}
QTableWidget#BenchmarkResultsTable::item {
    padding: 6px 8px;
}
            """)

    def _sync_target_card_visibility(
        self,
        selector: QCheckBox,
        details_widget: QWidget,
        hint_label: QLabel,
        card: QFrame,
    ) -> None:
        enabled = bool(selector.isChecked())
        details_widget.setVisible(enabled)
        hint_label.setVisible(enabled)
        card.setProperty("targetEnabled", enabled)
        self.style().unpolish(card)
        self.style().polish(card)
        card.update()

    def _build_notes_card(self, notes: list[str]) -> QFrame:
        frame = QFrame()
        frame.setObjectName("BenchmarkNoticeCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        text = QLabel("\n".join(f"- {note}" for note in notes))
        text.setObjectName("BenchmarkNoticeText")
        text.setWordWrap(True)
        layout.addWidget(text)
        return frame

    def _target_comparison_note(self, target: BenchmarkTargetSpec) -> str:
        assumed_targets = int(target.benchmark_context.get("max_targets", 0) or 0)
        if target.pipeline == "obb" and assumed_targets > 0:
            return f"Comparison note: benchmarks the single full-frame OBB model and assumes {assumed_targets} animals/frame from the Setup tab."
        if target.pipeline == "sequential" and assumed_targets > 0:
            return f"Comparison note: benchmarks stage-1 full-frame box detection plus stage-2 crop OBB per frame, assuming {assumed_targets} animals/frame from the Setup tab."
        if target.pipeline == "pose" and target.backend_family == "sleap":
            return "Comparison note: native CPU/MPS/CUDA rows benchmark the SLEAP service path, while ONNX/TensorRT rows benchmark exported direct runtimes. Both include crop transport, preprocess, inference, and postprocess during the timed phase."
        return ""

    def _format_runtime_display(
        self,
        target_key: str,
        result: BenchmarkResult,
    ) -> tuple[str, str | None]:
        target = self._target_specs.get(target_key)
        runtime_text = result.runtime_label
        runtime_tooltip: str | None = None
        if (
            target is None
            or target.pipeline != "pose"
            or target.backend_family != "sleap"
        ):
            return runtime_text, runtime_tooltip

        runtime_value = str(result.runtime or "").strip().lower()
        if runtime_value in {"cpu", "mps", "cuda"}:
            return (
                f"{runtime_text} [Native Service]",
                "SLEAP native runtime measured through the persistent SLEAP service backend, including crop transport and result extraction in the timed phase.",
            )
        if runtime_value.startswith("onnx_") or runtime_value == "tensorrt":
            return (
                f"{runtime_text} [Exported Direct]",
                "SLEAP exported runtime measured through the direct exported backend path, including crop preparation and postprocess in the timed phase.",
            )
        return runtime_text, runtime_tooltip

    def _build_target_card(
        self, target: BenchmarkTargetSpec
    ) -> tuple[QWidget, dict[str, Any]]:
        card = QFrame()
        card.setObjectName("BenchmarkTargetCard")
        card.setProperty("targetEnabled", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)

        selector = QCheckBox()
        selector.setChecked(True)
        header.addWidget(selector, 0, Qt.AlignTop)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)

        title_label = QLabel(target.label)
        title_label.setObjectName("BenchmarkTargetTitle")
        title_column.addWidget(title_label)

        model_name = os.path.basename(str(target.model_path).rstrip("/\\")) or str(
            target.model_path
        )
        subtitle = QLabel(f"Model: {model_name}")
        subtitle.setObjectName("BenchmarkTargetSubtitle")
        subtitle.setWordWrap(True)
        title_column.addWidget(subtitle)
        header.addLayout(title_column, 1)

        badge_column = QVBoxLayout()
        badge_column.setContentsMargins(0, 0, 0, 0)
        badge_column.setSpacing(6)

        current_text = f"Current: {runtime_label(target.current_runtime)}"
        if target.supports_batch_apply:
            if target.current_individual_batch_size is not None:
                current_text += f" | frame {target.current_batch_size} / crop {target.current_individual_batch_size}"
            else:
                current_text += f" | batch {target.current_batch_size}"
        current_badge = QLabel(current_text)
        current_badge.setObjectName("BenchmarkBadge")
        badge_column.addWidget(current_badge, 0, Qt.AlignRight)

        cached = lookup_cached_recommendation(
            target,
            self._geometry,
            realtime_enabled=self._main_window._is_realtime_tracking_mode_enabled(),
        )
        if cached is not None:
            if cached.individual_batch_size is not None:
                cached_text = f"Cached: {cached.runtime_label} @ frame {cached.batch_size} / crop {cached.individual_batch_size}"
            else:
                cached_text = (
                    f"Cached: {cached.runtime_label} @ batch {cached.batch_size}"
                )
            cached_badge = QLabel(cached_text)
            cached_badge.setObjectName("BenchmarkBadge")
            badge_column.addWidget(cached_badge, 0, Qt.AlignRight)
        else:
            badge_column.addStretch(1)
        header.addLayout(badge_column)
        card_layout.addLayout(header)

        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(14)

        runtime_box = QVBoxLayout()
        runtime_box.setContentsMargins(0, 0, 0, 0)
        runtime_box.setSpacing(6)
        runtime_label_widget = QLabel("Runtimes to test")
        runtime_label_widget.setObjectName("BenchmarkFieldLabel")
        runtime_box.addWidget(runtime_label_widget)
        runtime_selector = _RuntimeSelectionWidget(target.runtimes)
        runtime_box.addWidget(runtime_selector)
        controls.addLayout(runtime_box, 3)

        batch_box = QVBoxLayout()
        batch_box.setContentsMargins(0, 0, 0, 0)
        batch_box.setSpacing(6)
        batch_label_widget = QLabel(
            "Frame batch candidates"
            if target.individual_batch_sizes is not None
            else "Batch size candidates"
        )
        batch_label_widget.setObjectName("BenchmarkFieldLabel")
        batch_box.addWidget(batch_label_widget)
        batch_selector = _BatchSizeSelectionWidget(target.batch_sizes)
        batch_box.addWidget(batch_selector)
        controls.addLayout(batch_box, 4)

        individual_batch_selector = None
        if target.individual_batch_sizes is not None:
            individual_box = QVBoxLayout()
            individual_box.setContentsMargins(0, 0, 0, 0)
            individual_box.setSpacing(6)
            individual_label_widget = QLabel("Crop batch candidates")
            individual_label_widget.setObjectName("BenchmarkFieldLabel")
            individual_box.addWidget(individual_label_widget)
            individual_batch_selector = _BatchSizeSelectionWidget(
                target.individual_batch_sizes
            )
            individual_box.addWidget(individual_batch_selector)
            controls.addLayout(individual_box, 4)

        details_layout.addLayout(controls)

        comparison_note_text = self._target_comparison_note(target)
        comparison_note = QLabel(comparison_note_text)
        comparison_note.setObjectName("BenchmarkInlineHint")
        comparison_note.setWordWrap(True)
        comparison_note.setVisible(bool(comparison_note_text))
        details_layout.addWidget(comparison_note)

        hint = QLabel(
            "Recommendations will be written back into the existing runtime and batch controls after a successful run."
        )
        hint.setObjectName("BenchmarkInlineHint")
        hint.setWordWrap(True)
        details_layout.addWidget(hint)
        card_layout.addWidget(details_widget)

        selector.toggled.connect(
            lambda _checked: self._sync_target_card_visibility(
                selector,
                details_widget,
                hint,
                card,
            )
        )
        self._sync_target_card_visibility(selector, details_widget, hint, card)

        return card, {
            "target": target,
            "selector": selector,
            "runtimes": runtime_selector,
            "batches": batch_selector,
            "individual_batches": individual_batch_selector,
            "comparison_note": comparison_note,
        }

    def _build_ui(self) -> None:
        container = QWidget()
        container.setObjectName("BenchmarkDialogRoot")
        root = QVBoxLayout(container)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(14)

        summary_group = QGroupBox("Current Benchmark Inputs")
        summary_layout = QVBoxLayout(summary_group)
        summary_hint = QLabel(
            "Benchmarks use the currently loaded video geometry and your crop settings, so the recommendations stay specific to this session."
        )
        summary_hint.setObjectName("BenchmarkSectionHint")
        summary_hint.setWordWrap(True)
        summary_layout.addWidget(summary_hint)

        metrics = [
            (
                "Video Frame",
                f"{self._geometry.frame_width} × {self._geometry.frame_height}",
            ),
            (
                "Effective Detection",
                f"{self._geometry.effective_frame_width} × {self._geometry.effective_frame_height}",
            ),
            (
                "Canonical Crop",
                f"{self._geometry.canonical_crop_width} × {self._geometry.canonical_crop_height}",
            ),
            ("Resize Factor", f"{self._geometry.resize_factor:.2f}×"),
            ("Body Size", f"{self._geometry.reference_body_size:.1f} px"),
            (
                "Aspect / Padding",
                f"{self._geometry.reference_aspect_ratio:.2f} / {self._geometry.padding_fraction:.2f}",
            ),
        ]
        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(10)
        metrics_grid.setVerticalSpacing(10)
        for index, (title, value) in enumerate(metrics):
            metrics_grid.addWidget(_MetricTile(title, value), index // 3, index % 3)
        summary_layout.addLayout(metrics_grid)

        if self._collection_notes:
            summary_layout.addWidget(self._build_notes_card(self._collection_notes))
        root.addWidget(summary_group)

        target_group = QGroupBox("Targets")
        target_layout = QVBoxLayout(target_group)
        target_hint = QLabel(
            "Choose which pipelines to benchmark, then trim the runtime and batch-size candidates for each one."
        )
        target_hint.setObjectName("BenchmarkSectionHint")
        target_hint.setWordWrap(True)
        target_layout.addWidget(target_hint)

        target_scroll = QScrollArea()
        target_scroll.setWidgetResizable(True)
        target_scroll.setFrameShape(QScrollArea.NoFrame)
        target_widget = QWidget()
        target_cards = QVBoxLayout(target_widget)
        target_cards.setContentsMargins(0, 0, 0, 0)
        target_cards.setSpacing(10)
        for target in self._targets:
            card, row_info = self._build_target_card(target)
            target_cards.addWidget(card)
            self._row_widgets[target.key] = row_info
        target_cards.addStretch(1)
        target_scroll.setWidget(target_widget)
        target_layout.addWidget(target_scroll)
        root.addWidget(target_group, 1)

        status_group = QGroupBox("Run Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel(
            "Ready to benchmark the currently selected model targets."
        )
        self.status_label.setObjectName("BenchmarkStatusText")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        root.addWidget(status_group)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        results_hint = QLabel(
            "Timed metrics exclude model setup, artifact export, and warmup. Recommendations use per-frame latency first, then throughput, and memory columns reflect the timed benchmark phase. For SLEAP pose targets, native CPU/MPS/CUDA rows use the SLEAP service path while ONNX/TensorRT rows use exported direct runtimes."
        )
        results_hint.setObjectName("BenchmarkSectionHint")
        results_hint.setWordWrap(True)
        results_layout.addWidget(results_hint)
        self.results_table = QTableWidget(0, 10)
        self.results_table.setObjectName("BenchmarkResultsTable")
        self.results_table.setHorizontalHeaderLabels(
            [
                "Target",
                "Runtime",
                "Batch",
                "Batch (ms)",
                "Per Frame (ms)",
                "FPS",
                "RAM Peak (MB)",
                "Accel Peak (MB)",
                "Status",
                "Recommended",
            ]
        )
        header_tooltips = {
            0: "Benchmark target or pipeline being measured.",
            1: "Execution backend used for this run.",
            2: "Number of frames processed together. Sequential rows show frame and crop batches as F#/I#.",
            3: "Mean wall-clock time for one timed batch inference call. Setup, artifact export, and warmup are excluded.",
            4: "Mean runtime per frame or crop, computed as batch runtime divided by batch size.",
            5: "Effective throughput in frames per second during the timed benchmark phase.",
            6: "Peak process resident memory observed during timed iterations only.",
            7: "Peak accelerator memory observed during timed iterations when the backend exposes it.",
            8: "Run status or the captured failure message.",
            9: "Marks the configuration chosen as the default recommendation for this target.",
        }
        for column, tooltip in header_tooltips.items():
            item = self.results_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tooltip)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionMode(QTableWidget.NoSelection)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Stretch)
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.results_table, 1)
        root.addWidget(results_group, 1)

        button_row = QHBoxLayout()
        self.footer_hint = QLabel(
            "Run the benchmark first, then apply the generated recommendations back to TrackerKit."
        )
        self.footer_hint.setObjectName("BenchmarkSectionHint")
        self.footer_hint.setWordWrap(True)
        button_row.addWidget(self.footer_hint, 1)
        self.run_button = QPushButton("Run Benchmark")
        self.run_button.clicked.connect(self._start_benchmark)
        self.apply_button = QPushButton("Apply Recommendations")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply_and_close)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.close_button)
        root.addLayout(button_row)

        self.add_content(container)

    def _selected_targets(self) -> tuple[list[BenchmarkTargetSpec], list[str]]:
        selected: list[BenchmarkTargetSpec] = []
        errors: list[str] = []
        for row in self._row_widgets.values():
            if not row["selector"].isChecked():
                continue
            target = row["target"]
            runtimes = row["runtimes"].selected_runtimes()
            if not runtimes:
                errors.append(f"{target.label}: select at least one runtime.")
                continue
            selected.append(
                replace(
                    target,
                    runtimes=runtimes,
                    batch_sizes=row["batches"].values(),
                    individual_batch_sizes=(
                        row["individual_batches"].values()
                        if row.get("individual_batches") is not None
                        else None
                    ),
                )
            )
        return selected, errors

    def _start_benchmark(self) -> None:
        selected_targets, validation_errors = self._selected_targets()
        if validation_errors:
            QMessageBox.warning(
                self,
                "Invalid Target Configuration",
                "Resolve these target selections before benchmarking:\n\n"
                + "\n".join(validation_errors),
            )
            return
        if not selected_targets:
            QMessageBox.warning(
                self,
                "No Targets Selected",
                "Select at least one valid benchmark target before running the benchmark.",
            )
            return
        self._active_targets = list(selected_targets)
        self.results_table.setRowCount(0)
        self._result_payload = {}
        self._recommendations = {}
        self.apply_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status_label.setText("Starting benchmark…")
        self.progress_bar.setValue(0)
        self._worker = _BenchmarkWorker(
            selected_targets,
            self._geometry,
            warmup=2,
            iterations=5,
        )
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.status.connect(self.status_label.setText)
        self._worker.error.connect(self._on_error)
        self._worker.completed.connect(self._on_completed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_error(self, message: str) -> None:
        self.status_label.setText(message)
        QMessageBox.warning(self, "Benchmark Error", message)

    def _on_completed(self, payload: dict[str, Any]) -> None:
        if payload.get("cancelled"):
            self.status_label.setText("Benchmark cancelled.")
            return
        self._result_payload = dict(payload.get("results", {}))
        self._recommendations = self._load_cached_recommendations()
        for target in self._active_targets:
            results = list(self._result_payload.get(target.key, []))
            recommendation = store_cached_results(
                target,
                self._geometry,
                results,
                realtime_enabled=self._main_window._is_realtime_tracking_mode_enabled(),
            )
            if recommendation is not None:
                self._recommendations[target.key] = recommendation
            else:
                self._recommendations.pop(target.key, None)
        self._populate_results_table()
        self.apply_button.setEnabled(bool(self._recommendations))
        self.status_label.setText(
            "Benchmark complete. Review the results and apply the recommendations when ready."
        )

    def _on_finished(self) -> None:
        self.run_button.setEnabled(True)

    def _populate_results_table(self) -> None:
        rows: list[tuple[str, BenchmarkResult]] = []
        for target_key, results in self._result_payload.items():
            for result in results:
                rows.append((target_key, result))
        self.results_table.setRowCount(len(rows))
        for row_index, (target_key, result) in enumerate(rows):
            recommendation = self._recommendations.get(target_key)
            recommended = (
                recommendation is not None
                and recommendation.runtime == result.runtime
                and recommendation.batch_size == result.batch_size
                and recommendation.individual_batch_size == result.individual_batch_size
            )
            target_item = QTableWidgetItem(
                self._target_labels.get(target_key, target_key)
            )
            runtime_text, runtime_tooltip = self._format_runtime_display(
                target_key, result
            )
            runtime_item = QTableWidgetItem(runtime_text)
            if runtime_tooltip:
                runtime_item.setToolTip(runtime_tooltip)
            batch_text = (
                f"F{result.batch_size}/I{int(result.individual_batch_size)}"
                if result.individual_batch_size is not None
                else str(result.batch_size)
            )
            batch_item = QTableWidgetItem(batch_text)
            mean_item = QTableWidgetItem(
                f"{result.mean_ms:.2f}" if result.success else "—"
            )
            per_frame_item = QTableWidgetItem(
                f"{result.mean_per_frame_ms:.2f}" if result.success else "—"
            )
            fps_item = QTableWidgetItem(
                f"{result.throughput_fps:.2f}" if result.success else "—"
            )
            ram_item = QTableWidgetItem(
                f"{result.ram_peak_mb:.0f}"
                if result.success and result.ram_peak_mb is not None
                else "—"
            )
            accel_item = QTableWidgetItem(
                f"{result.accelerator_peak_mb:.0f}"
                if result.success and result.accelerator_peak_mb is not None
                else "—"
            )
            status_item = QTableWidgetItem(
                "OK" if result.success else result.error or "Failed"
            )
            recommended_item = QTableWidgetItem("Yes" if recommended else "")

            for item in (
                batch_item,
                mean_item,
                per_frame_item,
                fps_item,
                ram_item,
                accel_item,
                recommended_item,
            ):
                item.setTextAlignment(Qt.AlignCenter)
            runtime_item.setTextAlignment(Qt.AlignCenter)

            self.results_table.setItem(row_index, 0, target_item)
            self.results_table.setItem(
                row_index,
                1,
                runtime_item,
            )
            self.results_table.setItem(row_index, 2, batch_item)
            self.results_table.setItem(row_index, 3, mean_item)
            self.results_table.setItem(row_index, 4, per_frame_item)
            self.results_table.setItem(row_index, 5, fps_item)
            self.results_table.setItem(row_index, 6, ram_item)
            self.results_table.setItem(row_index, 7, accel_item)
            self.results_table.setItem(row_index, 8, status_item)
            self.results_table.setItem(row_index, 9, recommended_item)

    def _apply_and_close(self) -> None:
        if not self._recommendations:
            QMessageBox.information(
                self,
                "No Recommendations",
                "No successful benchmark recommendations are available to apply.",
            )
            return
        self._applied = True
        self.accept()

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        super().reject()
