"""HistoryDialog — browse training run history and select a model for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


def _load_runs(project: "DetectKitProject") -> list[dict]:
    """Load training run history for *project*. Monkeypatchable in tests."""
    runs = getattr(project, "training_history", None) or []
    return list(reversed(list(runs)))


def _entry_model_path(entry: dict) -> str:
    project_paths = entry.get("project_model_paths") or []
    if project_paths:
        for path in project_paths:
            if path:
                return str(path)
    project_path = str(entry.get("project_model_path", "") or "").strip()
    if project_path:
        return project_path
    published = str(entry.get("published_model_path", "") or "").strip()
    if published:
        return published
    artifact_paths = entry.get("artifact_paths") or []
    if artifact_paths:
        return str(artifact_paths[0])
    return ""


class HistoryDialog(BaseDialog):
    """Browse training run history; optionally load a model for inference."""

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Previously Trained Models",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._runs: list[dict] = []
        self.setMinimumSize(940, 620)
        self._build_content()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QLabel(
            "Browse, export, remove, or load project-trained DetectKit checkpoints for inference."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._qt_align = Qt.AlignVCenter | Qt.AlignLeft

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Run ID", "Started", "Role", "Base Model", "Status", "Epochs"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeToContents
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #252526; color: #ffffff; "
            "alternate-background-color: #2d2d30; gridline-color: #3a3a3a; border: none; }"
            "QTableWidget::item { padding: 4px 8px; }"
            "QTableWidget::item:selected { background-color: #094771; }"
            "QHeaderView::section { background-color: #2d2d2d; color: #ffffff; "
            "padding: 4px; border: none; border-right: 1px solid #3a3a3a; }"
        )
        self.table.doubleClicked.connect(self._load_for_inference)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setFixedHeight(120)
        self.detail_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail_label.setStyleSheet(
            "background: #252526; color: #ffffff; font-size: 11px; "
            "padding: 7px 10px; border-radius: 4px;"
        )
        layout.addWidget(self.detail_label)

        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("Use for Inference")
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._load_for_inference)
        self._btn_export = QPushButton("Export to Project Models")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_selected)
        self._btn_delete = QPushButton("Delete Run")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._delete_run)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_export)
        btn_row.addWidget(self._btn_delete)
        layout.addLayout(btn_row)

        container = QWidget()
        container.setLayout(layout)
        self.add_content(container)

    @staticmethod
    def _started_at(entry: dict) -> str:
        return str(entry.get("started_at", "") or "")[:19]

    @staticmethod
    def _base_model(entry: dict) -> str:
        spec = entry.get("spec") or {}
        base_model = str(spec.get("base_model", "") or "")
        return Path(base_model).name if base_model else "-"

    @staticmethod
    def _epochs(entry: dict) -> str:
        spec = entry.get("spec") or {}
        hyperparams = spec.get("hyperparams") or {}
        epochs = hyperparams.get("epochs")
        return str(epochs) if epochs not in (None, "") else "-"

    @staticmethod
    def _status_label(entry: dict) -> str:
        status = str(entry.get("status", "") or "unknown").strip()
        if entry.get("project_model_path") or entry.get("project_model_paths"):
            if status == "completed":
                return "completed/exported"
        return status or "unknown"

    @staticmethod
    def _artifact_names(entry: dict, key: str) -> str:
        names = [Path(path).name for path in entry.get(key) or [] if str(path).strip()]
        return ", ".join(names) if names else "-"

    def _refresh_table(self, select_row: int | None = None) -> None:
        self.table.setRowCount(0)
        for row, entry in enumerate(self._runs):
            self.table.insertRow(row)
            values = [
                str(entry.get("run_id", "?")),
                self._started_at(entry),
                str(entry.get("role", "")),
                self._base_model(entry),
                self._status_label(entry),
                self._epochs(entry),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(self._qt_align)
                self.table.setItem(row, column, item)

        if self._runs:
            target = (
                0
                if select_row is None
                else max(0, min(select_row, len(self._runs) - 1))
            )
            self.table.selectRow(target)
        else:
            self.detail_label.setText("No project training history entries remain.")
            self._btn_load.setEnabled(False)
            self._btn_export.setEnabled(False)
            self._btn_delete.setEnabled(False)

    def _selected_row(self) -> int:
        return int(self.table.currentRow())

    def _get_selected_entry(self) -> dict | None:
        row = self._selected_row()
        if 0 <= row < len(self._runs):
            return self._runs[row]
        return None

    def _set_detail_text(self, entry: dict | None) -> None:
        if entry is None:
            self.detail_label.setText("")
            return

        exported_model = _entry_model_path(entry)
        export_name = Path(exported_model).name if exported_model else "-"
        run_dir = Path(str(entry.get("project_run_dir", "") or "")).name or "-"
        metrics_files = self._artifact_names(entry, "project_metrics_paths")
        source_files = self._artifact_names(entry, "artifact_paths")
        log_name = Path(str(entry.get("project_log_path", "") or "")).name or "-"

        self.detail_label.setText(
            f"<b>{entry.get('run_id', '?')}</b> &nbsp;&bull;&nbsp; Role: <b>{entry.get('role', '-')}</b>"
            f" &nbsp;&bull;&nbsp; Status: <b>{self._status_label(entry)}</b><br>"
            f"<span style='color:#ffffff'>Base model:</span> {self._base_model(entry)}"
            f" &nbsp;&bull;&nbsp; <span style='color:#ffffff'>Started:</span> {self._started_at(entry) or '-'}<br>"
            f"<span style='color:#ffffff'>Exported model:</span> <span style='color:#9cdcfe'>{export_name}</span>"
            f" &nbsp;&bull;&nbsp; <span style='color:#ffffff'>Run folder:</span> {run_dir}<br>"
            f"<span style='color:#ffffff'>Source artifacts:</span> {source_files}<br>"
            f"<span style='color:#ffffff'>Project metrics:</span> {metrics_files}"
            f" &nbsp;&bull;&nbsp; <span style='color:#ffffff'>Log:</span> {log_name}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        self._runs = _load_runs(self._project)
        self._refresh_table(select_row=0)

    def _on_selection_changed(self) -> None:
        entry = self._get_selected_entry()
        has_entry = entry is not None
        self._btn_load.setEnabled(has_entry and bool(_entry_model_path(entry)))
        self._btn_export.setEnabled(has_entry and bool(entry.get("artifact_paths")))
        self._btn_delete.setEnabled(has_entry)
        self._set_detail_text(entry)

    def _load_for_inference(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        model_path = _entry_model_path(entry)
        if not model_path:
            QMessageBox.warning(
                self,
                "No Model",
                "No exported or recorded model artifact is available for this run.",
            )
            return
        self._project.active_model_path = model_path
        self.accept()

    def _export_selected(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return

        from ..project import export_training_history_entry

        run_id = str(entry.get("run_id", "")).strip()
        updated = export_training_history_entry(self._project, run_id)
        if updated is None or not _entry_model_path(updated):
            QMessageBox.warning(
                self,
                "Export Failed",
                "Could not export this run into the project's models folder.",
            )
            return

        selected_row = self._selected_row()
        self._refresh()
        for index, entry in enumerate(self._runs):
            if str(entry.get("run_id", "")).strip() == run_id:
                self.table.selectRow(index)
                break
        else:
            if self._runs:
                self.table.selectRow(max(0, min(selected_row, len(self._runs) - 1)))

        self.detail_label.setText(
            f"<span style='color:#4ec9b0'>Exported model to:</span> {Path(_entry_model_path(updated)).name}"
        )

        QMessageBox.information(
            self,
            "Export Complete",
            f"Model exported to:\n{_entry_model_path(updated)}",
        )

    def _delete_run(self) -> None:
        row = self._selected_row()
        entry = self._get_selected_entry()
        if entry is None:
            return
        run_id = entry.get("run_id", "?")
        ans = QMessageBox.question(
            self,
            "Delete Run",
            f"Delete run '{run_id}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            from ..project import delete_training_history_entry

            delete_training_history_entry(self._project, run_id)
        except Exception as exc:
            logger.warning("Could not delete run %s: %s", run_id, exc)
        self._refresh()
        if self._runs:
            self.table.selectRow(max(0, min(row - 1, len(self._runs) - 1)))
