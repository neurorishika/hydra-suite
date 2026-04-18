"""SourceManagerDialog — add/remove/scan dataset source directories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

from ..source_import import inspect_detectkit_source, materialize_detectkit_source
from .source_validation import confirm_detectkit_source_addition

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class SourceManagerDialog(BaseDialog):
    """Manage dataset source directories for a DetectKit project."""

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Manage Sources",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._build_content()
        self._refresh_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        v.addWidget(QLabel("Dataset source directories:"))

        self._source_list = QListWidget()
        self._source_list.setMinimumHeight(200)
        v.addWidget(self._source_list)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Source…")
        self.btn_add.clicked.connect(self._add_source)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        v.addLayout(btn_row)

        self.add_content(container)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        self._source_list.clear()
        for src in self._project.sources:
            display = src.name if src.name else (src.original_path or src.path)
            if src.imported and src.source_kind:
                display = f"{display} [{src.source_kind}]"
            self._source_list.addItem(display)

    def _add_source(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Source Directory", ""
        )
        if not directory:
            return
        selected_path = str(Path(directory).expanduser().resolve())
        from ..models import OBBSource

        # Avoid duplicates
        existing_paths = {
            candidate
            for src in self._project.sources
            for candidate in (src.path, src.original_path)
            if candidate
        }
        if selected_path in existing_paths:
            QMessageBox.information(self, "Add Source", "Source already added.")
            return

        try:
            inspection = inspect_detectkit_source(selected_path)
        except Exception as exc:
            QMessageBox.warning(self, "Add Source", str(exc))
            return

        if not confirm_detectkit_source_addition(self, selected_path, inspection):
            return

        try:
            materialized = materialize_detectkit_source(
                selected_path,
                self._project.project_dir,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Add Source", str(exc))
            return

        canonical_path = str(materialized.canonical_path)
        original_path = str(materialized.source_root)
        if canonical_path in existing_paths or original_path in existing_paths:
            QMessageBox.information(self, "Add Source", "Source already added.")
            return

        self._project.sources.append(
            OBBSource(
                path=canonical_path,
                name=materialized.display_name,
                original_path=original_path,
                source_kind=materialized.source_kind,
                imported=materialized.imported,
            )
        )
        self._refresh_list()

    def _remove_selected(self) -> None:
        row = self._source_list.currentRow()
        if row < 0 or row >= len(self._project.sources):
            return
        self._project.sources.pop(row)
        self._refresh_list()
