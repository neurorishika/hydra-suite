"""EvaluationDialog — dataset analysis and model evaluation for DetectKit."""

from __future__ import annotations

import logging
from pathlib import Path
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
        sources = self._project.sources
        if not sources:
            self._analysis_view.setPlainText("No dataset sources configured.")
            return

        try:
            from hydra_suite.training.dataset_inspector import (
                DatasetInspection,
                analyze_obb_sizes,
                format_size_analysis,
                inspect_obb_or_detect_dataset,
            )
        except ImportError:
            self._analysis_view.setPlainText(
                "Dataset inspector not available. Install training dependencies."
            )
            return

        merged = DatasetInspection(root_dir="(merged)")
        for src in sources:
            if not src.path:
                continue
            try:
                insp = inspect_obb_or_detect_dataset(src.path)
            except Exception as exc:
                logger.warning("Failed to inspect %s: %s", src.path, exc)
                continue
            for split_name, items in insp.splits.items():
                merged.splits.setdefault(split_name, []).extend(items)
            merged.class_names.update(insp.class_names)

        if not any(merged.splits.values()):
            self._analysis_view.setPlainText(
                "No valid dataset items found in the configured sources."
            )
            return

        stats = analyze_obb_sizes(
            merged,
            pad_ratio=self._project.crop_pad_ratio,
            min_crop_size_px=self._project.min_crop_size_px,
            enforce_square=self._project.enforce_square,
        )

        report_seq, warnings_seq = format_size_analysis(
            stats,
            training_imgsz=self._project.imgsz_seq_crop_obb,
            pipeline_mode="crop",
        )
        report_direct, warnings_direct = format_size_analysis(
            stats,
            training_imgsz=self._project.imgsz_obb_direct,
            pipeline_mode="full_image",
        )

        lines = [
            "=== Seq Crop OBB Pipeline ===",
            f"(imgsz = {self._project.imgsz_seq_crop_obb})",
            "",
            report_seq,
        ]
        if warnings_seq:
            lines += ["", "WARNINGS:"] + [f"  \u26a0 {w}" for w in warnings_seq]

        lines += [
            "",
            "=== OBB Direct Pipeline ===",
            f"(imgsz = {self._project.imgsz_obb_direct})",
            "",
            report_direct,
        ]
        if warnings_direct:
            lines += ["", "WARNINGS:"] + [f"  \u26a0 {w}" for w in warnings_direct]

        self._analysis_view.setPlainText("\n".join(lines))

        all_warnings = warnings_seq + warnings_direct
        if all_warnings:
            QMessageBox.warning(
                self, "Dataset Analysis Warnings", "\n".join(all_warnings)
            )

    def _quick_test(self) -> None:
        model_path = self._project.active_model_path
        if not model_path or not Path(model_path).exists():
            QMessageBox.information(
                self,
                "Quick Test",
                "No active model found.\n\n"
                "Run training first, or select a model from Run History.",
            )
            return

        sources = self._project.sources
        dataset_dir = sources[0].path if sources else ""

        try:
            from ..project import (
                detectkit_latest_model_path_for_role,
                detectkit_training_history_entry_for_model_path,
            )
        except ImportError:
            detectkit_latest_model_path_for_role = None
            detectkit_training_history_entry_for_model_path = None

        entry = (
            detectkit_training_history_entry_for_model_path(self._project, model_path)
            if detectkit_training_history_entry_for_model_path is not None
            else None
        )
        role = (
            str((entry or {}).get("role") or "obb_direct").strip().lower()
            or "obb_direct"
        )
        if role == "seq_detect":
            QMessageBox.information(
                self,
                "Quick Test",
                "The selected model is a sequence-detect stage-1 checkpoint. Quick Test supports OBB direct models and sequence crop OBB checkpoints with a paired detect model.",
            )
            return

        imgsz = {
            "obb_direct": int(self._project.imgsz_obb_direct),
            "seq_crop_obb": int(self._project.imgsz_seq_crop_obb),
        }.get(role, int(self._project.imgsz_obb_direct))
        detect_model_path = ""
        if role == "seq_crop_obb":
            detect_model_path = (
                detectkit_latest_model_path_for_role(self._project, "seq_detect")
                if detectkit_latest_model_path_for_role is not None
                else ""
            )
            if not detect_model_path:
                QMessageBox.information(
                    self,
                    "Quick Test",
                    "The selected sequence crop OBB model needs a paired sequence-detect checkpoint before quick testing.",
                )
                return

        try:
            from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import (
                ModelTestDialog,
            )
        except ImportError:
            QMessageBox.information(
                self, "Not Available", "Model test dialog is not available."
            )
            return

        dlg = ModelTestDialog(
            model_path=model_path,
            role=role,
            dataset_dir=dataset_dir,
            device=self._project.device or "cpu",
            imgsz=imgsz,
            crop_pad_ratio=self._project.crop_pad_ratio,
            min_crop_size_px=self._project.min_crop_size_px,
            enforce_square=self._project.enforce_square,
            detect_model_path=detect_model_path,
            parent=self,
        )
        dlg.open()
