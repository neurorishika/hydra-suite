"""EvaluationDialog — dataset analysis and validation metrics for DetectKit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.evaluation import (
    EvaluationCandidate,
    EvaluationResult,
    collect_evaluation_candidates,
)
from hydra_suite.detectkit.jobs.evaluation import EvaluationWorker
from hydra_suite.widgets.dialogs import BaseDialog

from ..evaluation import build_dataset_analysis_report, open_quick_test_dialog

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class EvaluationDialog(BaseDialog):
    """Compare trained models using their real held-out validation splits."""

    _RESULT_HEADERS = (
        "Evaluated",
        "Run",
        "Role",
        "Dataset",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "Inference",
        "Status",
    )

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Evaluate",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._candidates = collect_evaluation_candidates(project)
        self._worker: EvaluationWorker | None = None
        self._session_failures = 0
        self._worker_error = ""
        self.resize(980, 760)
        self._build_content()
        self._populate_candidate_table()
        self._populate_results_table()

    def _build_content(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_model_eval_group(), 1)
        layout.addWidget(self._build_dataset_analysis_group())
        self.add_content(container)

    def _build_dataset_analysis_group(self) -> QGroupBox:
        box = QGroupBox("Dataset Analysis")
        layout = QVBoxLayout(box)
        self.btn_analyze = QPushButton("Analyze Dataset")
        self.btn_analyze.clicked.connect(self._run_dataset_analysis)
        layout.addWidget(self.btn_analyze)
        self._analysis_view = QTextEdit()
        self._analysis_view.setReadOnly(True)
        self._analysis_view.setPlaceholderText(
            "Inspect source statistics and compatibility warnings."
        )
        self._analysis_view.setMinimumHeight(120)
        layout.addWidget(self._analysis_view)
        return box

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)

    def _build_model_eval_group(self) -> QGroupBox:
        box = QGroupBox("Validation-Set Model Evaluation")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)
        note = QLabel(
            "Select one or more completed training runs. DetectKit evaluates each "
            "checkpoint on that run's held-out val split and records task-level "
            "precision, recall, mAP50, and mAP50-95 for side-by-side comparison."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.candidate_table = QTableWidget(0, 5)
        self.candidate_table.setHorizontalHeaderLabels(
            ["Run", "Role", "Model", "Validation Dataset", "Availability"]
        )
        self._configure_table(self.candidate_table)
        self.candidate_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.candidate_table.setMinimumHeight(150)
        self.candidate_table.itemSelectionChanged.connect(
            self._update_evaluate_selected_enabled
        )
        layout.addWidget(self.candidate_table)

        button_row = QHBoxLayout()
        self.btn_evaluate_selected = QPushButton("Evaluate Selected")
        self.btn_evaluate_all = QPushButton("Evaluate All Available")
        self.btn_stop = QPushButton("Stop After Current Run")
        self.btn_stop.setEnabled(False)
        self.btn_quick_test = QPushButton("Quick Test…")
        self.btn_evaluate_selected.clicked.connect(self._evaluate_selected)
        self.btn_evaluate_all.clicked.connect(self._evaluate_all)
        self.btn_stop.clicked.connect(self._cancel_evaluation)
        self.btn_quick_test.clicked.connect(self._quick_test)
        button_row.addWidget(self.btn_evaluate_selected)
        button_row.addWidget(self.btn_evaluate_all)
        button_row.addWidget(self.btn_stop)
        button_row.addStretch()
        button_row.addWidget(self.btn_quick_test)
        layout.addLayout(button_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        layout.addWidget(self.progress)
        self.status_label = QLabel("Choose training runs to compare.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.results_table = QTableWidget(0, len(self._RESULT_HEADERS))
        self.results_table.setHorizontalHeaderLabels(list(self._RESULT_HEADERS))
        self._configure_table(self.results_table)
        self.results_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.results_table.setMinimumHeight(180)
        layout.addWidget(self.results_table)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Validation progress and errors appear here.")
        self.log_view.setMinimumHeight(90)
        layout.addWidget(self.log_view)
        return box

    def _populate_candidate_table(self) -> None:
        self.candidate_table.setRowCount(0)
        first_available_row = -1
        for row, candidate in enumerate(self._candidates):
            self.candidate_table.insertRow(row)
            values = (
                candidate.run_id,
                candidate.role,
                candidate.model_path.rsplit("/", 1)[-1] or "-",
                candidate.dataset_dir.rsplit("/", 1)[-1] or "-",
                "Ready" if candidate.available else candidate.unavailable_reason,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row)
                if not candidate.available:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self.candidate_table.setItem(row, column, item)
            if candidate.available and first_available_row < 0:
                first_available_row = row

        if first_available_row >= 0:
            self.candidate_table.selectRow(first_available_row)
        elif not self._candidates:
            self.status_label.setText(
                "No training runs are recorded yet. Train a model before evaluating."
            )
        else:
            self.status_label.setText(
                "No recorded run currently has both a model and derived validation set."
            )
        self._update_evaluate_selected_enabled()
        self.btn_evaluate_all.setEnabled(any(c.available for c in self._candidates))

    @staticmethod
    def _format_metric(value: object) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "-"

    def _append_result_record(self, record: dict) -> None:
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        error = str(record.get("error", "") or "")
        inference_ms = float(record.get("inference_ms", 0.0) or 0.0)
        values = (
            str(record.get("evaluated_at", "") or "")[:19],
            str(record.get("run_id", "") or ""),
            str(record.get("role", "") or ""),
            str(record.get("dataset_name", "") or "-"),
            self._format_metric(record.get("precision")),
            self._format_metric(record.get("recall")),
            self._format_metric(record.get("map50")),
            self._format_metric(record.get("map50_95")),
            f"{inference_ms:.1f} ms" if inference_ms > 0.0 else "-",
            error or "Completed",
        )
        for column, value in enumerate(values):
            self.results_table.setItem(row, column, QTableWidgetItem(value))
        self.results_table.scrollToBottom()

    def _populate_results_table(self) -> None:
        self.results_table.setRowCount(0)
        for record in reversed(list(self._project.evaluation_history or [])):
            self._append_result_record(record)

    def _selected_candidates(self) -> list[EvaluationCandidate]:
        rows = sorted(
            index.row()
            for index in self.candidate_table.selectionModel().selectedRows()
        )
        return [
            self._candidates[row]
            for row in rows
            if 0 <= row < len(self._candidates) and self._candidates[row].available
        ]

    def _update_evaluate_selected_enabled(self) -> None:
        running = self._worker is not None and self._worker.isRunning()
        self.btn_evaluate_selected.setEnabled(
            not running and bool(self._selected_candidates())
        )

    def _evaluate_selected(self) -> None:
        self._start_evaluation(self._selected_candidates())

    def _evaluate_all(self) -> None:
        self._start_evaluation([c for c in self._candidates if c.available])

    def _start_evaluation(self, candidates: list[EvaluationCandidate]) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Evaluation Running", "Please wait or stop.")
            return
        if not candidates:
            QMessageBox.information(
                self,
                "No Runs Selected",
                "Select at least one available training run to evaluate.",
            )
            return

        from ..project import detectkit_artifact_paths

        output_root = detectkit_artifact_paths(self._project.project_dir)["evaluation"]
        batch = 1 if self._project.auto_batch else max(1, int(self._project.batch))
        worker = EvaluationWorker(
            candidates,
            output_root=output_root,
            device=self._project.device or "auto",
            batch=batch,
        )
        worker.result_ready.connect(self._on_evaluation_result)
        worker.log_signal.connect(self.log_view.append)
        worker.status.connect(self.status_label.setText)
        worker.progress.connect(self.progress.setValue)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker

        self.progress.setValue(0)
        self.progress.setFormat("Evaluating… %p%")
        self._session_failures = 0
        self._worker_error = ""
        self.status_label.setText(
            f"Evaluating {len(candidates)} run(s) on held-out validation data."
        )
        self._set_running(True)
        worker.start()

    def _set_running(self, running: bool) -> None:
        self.candidate_table.setEnabled(not running)
        self.btn_evaluate_selected.setEnabled(
            not running and bool(self._selected_candidates())
        )
        self.btn_evaluate_all.setEnabled(
            not running and any(c.available for c in self._candidates)
        )
        self.btn_stop.setEnabled(running)
        self.btn_analyze.setEnabled(not running)

    def _on_evaluation_result(self, result: EvaluationResult) -> None:
        if not result.success:
            self._session_failures += 1
        record = result.to_project_record()
        history = list(self._project.evaluation_history or [])
        history.append(record)
        self._project.evaluation_history = history[-200:]
        self._append_result_record(record)
        try:
            from ..project import save_project

            save_project(self._project)
        except Exception as exc:  # noqa: BLE001 - result remains visible in memory
            logger.warning(
                "Could not persist DetectKit evaluation history", exc_info=True
            )
            self.log_view.append(f"WARNING: Could not save evaluation history: {exc}")

    def _on_worker_error(self, message: str) -> None:
        self._worker_error = str(message)
        self.log_view.append(f"Evaluation worker failed: {message}")
        self.status_label.setText("Evaluation failed. See the log for details.")

    def _on_worker_finished(self) -> None:
        worker = self._worker
        cancelled = bool(worker is not None and worker.is_cancelled())
        self._worker = None
        self._set_running(False)
        if cancelled:
            self.progress.setFormat("Cancelled")
            self.status_label.setText("Evaluation stopped between validation runs.")
        elif self._worker_error:
            self.progress.setFormat("Failed")
            self.status_label.setText("Evaluation failed. See the log for details.")
        elif self._session_failures:
            self.progress.setValue(100)
            self.progress.setFormat("Complete with failures")
            self.status_label.setText(
                f"Evaluation complete with {self._session_failures} failed run(s). "
                "Compare successful rows and review error statuses."
            )
        else:
            self.progress.setValue(100)
            self.progress.setFormat("Complete")
            self.status_label.setText("Evaluation complete. Compare the metrics above.")

    def _cancel_evaluation(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self.btn_stop.setEnabled(False)
        self.status_label.setText(
            "Stop requested. The current validation run will finish first."
        )

    def _run_dataset_analysis(self) -> None:
        report, warnings = build_dataset_analysis_report(self._project)
        self._analysis_view.setPlainText(report)
        if warnings:
            QMessageBox.warning(self, "Dataset Analysis Warnings", "\n".join(warnings))

    def _quick_test(self) -> None:
        open_quick_test_dialog(self._project, parent=self)

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Evaluation Running",
                "Stop evaluation before closing this dialog.",
            )
            return
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Evaluation Running",
                "Stop evaluation before closing this dialog.",
            )
            event.ignore()
            return
        super().closeEvent(event)
