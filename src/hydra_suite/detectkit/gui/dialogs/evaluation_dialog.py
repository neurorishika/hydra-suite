"""EvaluationDialog — dataset analysis and model evaluation for DetectKit."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

from ..evaluation import build_dataset_analysis_report, open_quick_test_dialog

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class EvaluationDialog(BaseDialog):
    """Dataset analysis and model evaluation."""

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Evaluate",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self.resize(600, 500)
        self._build_content()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setSpacing(10)
        v.setContentsMargins(0, 0, 0, 0)

        v.addWidget(self._build_dataset_analysis_group())
        v.addWidget(self._build_model_eval_group())

        self.add_content(container)

    def _build_dataset_analysis_group(self) -> QGroupBox:
        box = QGroupBox("Dataset Analysis")
        v = QVBoxLayout(box)

        self.btn_analyze = QPushButton("Analyze Dataset")
        self.btn_analyze.clicked.connect(self._run_dataset_analysis)
        v.addWidget(self.btn_analyze)

        self._analysis_view = QTextEdit()
        self._analysis_view.setReadOnly(True)
        self._analysis_view.setPlaceholderText(
            "Click 'Analyze Dataset' to inspect source statistics and compatibility warnings."
        )
        self._analysis_view.setMinimumHeight(160)
        v.addWidget(self._analysis_view)

        return box

    def _build_model_eval_group(self) -> QGroupBox:
        box = QGroupBox("Model Evaluation")
        v = QVBoxLayout(box)

        note = QLabel(
            "Uses the active model set via Run History or after a completed training run."
        )
        note.setWordWrap(True)
        v.addWidget(note)

        self.btn_quick_test = QPushButton("Quick Test\u2026")
        self.btn_quick_test.clicked.connect(self._quick_test)
        v.addWidget(self.btn_quick_test)

        return box

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _run_dataset_analysis(self) -> None:
        report, warnings = build_dataset_analysis_report(self._project)
        self._analysis_view.setPlainText(report)
        if warnings:
            QMessageBox.warning(self, "Dataset Analysis Warnings", "\n".join(warnings))

    def _quick_test(self) -> None:
        open_quick_test_dialog(self._project, parent=self)
