"""Standalone source manager for PoseKit projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

from ..models import DataSource, Project
from ..project import _resolve_images_dir
from .add_source import AddSourceDialog


@dataclass
class _ManagedSource:
    source_id: str
    dataset_root: Path
    images_dir: Path
    description: str
    existing: bool


class SourceManagerDialog(BaseDialog):
    """Review, add, and remove PoseKit dataset sources before applying changes."""

    def __init__(self, project: Project, parent=None) -> None:
        super().__init__("Source Manager", parent=parent)
        self.setMinimumSize(680, 460)

        self._project = project
        self._managed_sources: List[_ManagedSource] = [
            _ManagedSource(
                source_id=src.source_id,
                dataset_root=src.dataset_root,
                images_dir=src.images_dir,
                description=src.description,
                existing=True,
            )
            for src in project.sources
        ]
        self._original_ids = [src.source_id for src in project.sources]
        self._pending_counter = 0

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        intro = QLabel(
            "Manage the dataset folders attached to this PoseKit project. "
            "Use Add Source to reuse the existing FilterKit-assisted source picker."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self._list, 1)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Source…")
        self._btn_add.clicked.connect(self._add_source)
        btn_row.addWidget(self._btn_add)
        btn_row.addStretch(1)
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._btn_remove)
        layout.addLayout(btn_row)

        self._summary = QLabel("")
        self._summary.setStyleSheet("QLabel { color: #9f9f9f; font-size: 11px; }")
        layout.addWidget(self._summary)

        self.add_content(content)

        self._buttons.accepted.disconnect()
        self._buttons.accepted.connect(self.accept)
        apply_button = self._buttons.button(QDialogButtonBox.Ok)
        apply_button.setText("Apply")
        self._buttons.button(QDialogButtonBox.Cancel).setText("Cancel")

        self._list.itemSelectionChanged.connect(self._update_remove_state)
        self._refresh_list()

    def _make_stub_project(self) -> Project:
        stub = Project(
            images_dir=self._project.images_dir,
            out_root=self._project.out_root,
            labels_dir=self._project.labels_dir,
            project_path=self._project.project_path,
            class_names=list(self._project.class_names),
            keypoint_names=list(self._project.keypoint_names),
            skeleton_edges=list(self._project.skeleton_edges),
        )
        stub.sources = [
            DataSource(
                source_id=src.source_id,
                dataset_root=src.dataset_root,
                images_dir=src.images_dir,
                labels_dir=self._project.labels_dir,
                description=src.description,
            )
            for src in self._managed_sources
        ]
        return stub

    def _refresh_list(self) -> None:
        self._list.clear()
        if not self._managed_sources:
            placeholder = QListWidgetItem("(no sources added yet)")
            placeholder.setFlags(Qt.NoItemFlags)
            self._list.addItem(placeholder)
        else:
            for src in self._managed_sources:
                label = src.description or src.dataset_root.name or src.source_id
                tag = "" if src.existing else "[NEW] "
                item = QListWidgetItem(
                    f"{tag}{label}\n{src.dataset_root}\n{src.images_dir}"
                )
                item.setData(Qt.UserRole, src.source_id)
                if not src.existing:
                    item.setForeground(Qt.darkGreen)
                self._list.addItem(item)
        self._summary.setText(f"{len(self._managed_sources)} source(s)")
        self._update_remove_state()

    def _update_remove_state(self) -> None:
        self._btn_remove.setEnabled(bool(self._list.selectedItems()))

    def _add_source(self) -> None:
        dlg = AddSourceDialog(self._make_stub_project(), parent=self)
        if dlg.exec() != AddSourceDialog.Accepted or dlg.selected_dir is None:
            return

        dataset_root = dlg.selected_dir.expanduser().resolve()
        images_dir = _resolve_images_dir(dataset_root)
        self._pending_counter += 1
        self._managed_sources.append(
            _ManagedSource(
                source_id=f"pending_{self._pending_counter}",
                dataset_root=dataset_root,
                images_dir=images_dir,
                description=dlg.description or dataset_root.name,
                existing=False,
            )
        )
        self._refresh_list()

    def _remove_selected(self) -> None:
        selected_ids = {
            str(item.data(Qt.UserRole))
            for item in self._list.selectedItems()
            if item.data(Qt.UserRole)
        }
        if not selected_ids:
            return
        self._managed_sources = [
            src for src in self._managed_sources if src.source_id not in selected_ids
        ]
        self._refresh_list()

    @property
    def source_ids_to_remove(self) -> List[str]:
        current_ids = {src.source_id for src in self._managed_sources if src.existing}
        return [
            source_id
            for source_id in self._original_ids
            if source_id not in current_ids
        ]

    @property
    def sources_to_add(self) -> List[tuple[Path, str]]:
        return [
            (src.dataset_root, src.description)
            for src in self._managed_sources
            if not src.existing
        ]

    @property
    def has_changes(self) -> bool:
        return bool(self.source_ids_to_remove or self.sources_to_add)

    @property
    def preferred_source_id(self) -> str:
        if not self._managed_sources:
            return ""
        current_ids = {src.source_id for src in self._managed_sources if src.existing}
        if self._project.last_source_id and self._project.last_source_id in current_ids:
            return self._project.last_source_id
        return self._managed_sources[0].source_id
