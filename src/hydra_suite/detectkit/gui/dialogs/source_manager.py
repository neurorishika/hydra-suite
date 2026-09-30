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
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.gui.dialogs._base import DetectKitDialog

from ...jobs.sam2_escalation import remove_staged_escalation_dir
from ..models import OBBSource
from ..source_import import (
    IMPORT_MODE_LINKED,
    IMPORT_MODE_PORTABLE,
    compute_positional_class_remap,
    resolve_al_round_authoritative_level,
)
from ..source_workers import SourceImportWorker, SourceInspectionWorker
from .source_validation import (
    SOURCE_ADD_MODE_LINKED,
    SOURCE_ADD_MODE_PORTABLE,
    confirm_detectkit_source_addition,
)

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class SourceManagerDialog(DetectKitDialog):
    """Manage dataset source directories for a DetectKit project."""

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Manage Sources",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._inspection_worker: SourceInspectionWorker | None = None
        self._import_worker: SourceImportWorker | None = None
        self._pending_inspection: tuple[str, set[str]] | None = None
        self._pending_add: tuple[str, object, set[str]] | None = None
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

        self._import_status = QLabel("")
        self._import_status.hide()
        v.addWidget(self._import_status)
        self._import_progress = QProgressBar()
        self._import_progress.setRange(0, 100)
        self._import_progress.hide()
        v.addWidget(self._import_progress)

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
        if self._inspection_worker is not None or self._import_worker is not None:
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Select Source Directory", ""
        )
        if not directory:
            return
        selected_path = str(Path(directory).expanduser().resolve())
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

        self._pending_inspection = (selected_path, existing_paths)
        worker = SourceInspectionWorker(selected_path, parent=self)
        self._inspection_worker = worker
        self._set_busy("Inspecting source…", indeterminate=True)
        worker.status.connect(self._import_status.setText)
        worker.finished.connect(self._finish_inspection)
        worker.start()

    def _finish_inspection(self) -> None:
        worker = self._inspection_worker
        pending = self._pending_inspection
        self._inspection_worker = None
        self._pending_inspection = None
        self._clear_busy()
        if worker is None or pending is None:
            return
        worker.deleteLater()
        if worker.failure_exception is not None:
            QMessageBox.warning(self, "Add Source", str(worker.failure_exception))
            return
        selected_path, existing_paths = pending
        inspection, level_scan = worker.result

        self._confirm_source(selected_path, existing_paths, inspection, level_scan)

    def _confirm_source(self, selected_path, existing_paths, inspection, level_scan):
        # inspection.dataset_root may differ from selected_path when the user
        # picked an active-learning round container -- review the resolved
        # (authoritative) dataset root, not the container.
        selection = confirm_detectkit_source_addition(
            self, str(inspection.dataset_root), inspection, level_scan=level_scan
        )
        if selection in {None, False}:
            return
        selection_mode = getattr(
            selection,
            "mode",
            (
                SOURCE_ADD_MODE_LINKED
                if selection == SOURCE_ADD_MODE_LINKED
                else SOURCE_ADD_MODE_PORTABLE
            ),
        )
        import_mode = (
            IMPORT_MODE_LINKED
            if selection_mode == SOURCE_ADD_MODE_LINKED
            else IMPORT_MODE_PORTABLE
        )

        force_remap = False
        project_classes = list(self._project.class_names)
        source_classes = list(inspection.discovered_labels)
        if source_classes != project_classes:
            remap_preview = compute_positional_class_remap(
                source_classes, project_classes
            )
            mapping_lines: list[str] = []
            for source_idx, target_idx in sorted(remap_preview.items()):
                source_name = (
                    source_classes[source_idx]
                    if 0 <= source_idx < len(source_classes)
                    else f"class {source_idx}"
                )
                target_name = (
                    project_classes[target_idx]
                    if 0 <= target_idx < len(project_classes)
                    else f"class {target_idx}"
                )
                mapping_lines.append(
                    f"  source[{source_idx}] {source_name!r} → "
                    f"project[{target_idx}] {target_name!r}"
                )
            dropped = sorted(
                {
                    source_idx
                    for source_idx in range(len(source_classes))
                    if source_idx not in remap_preview
                }
            )
            preview_text = (
                "Source classes do not match the project class scheme.\n\n"
                f"Project classes: {project_classes}\n"
                f"Source classes:  {source_classes}\n\n"
                "Force the source labels to match the project classes by mapping "
                "by position?\n" + "\n".join(mapping_lines)
            )
            if dropped:
                dropped_names = ", ".join(
                    f"{i}:{source_classes[i]!r}"
                    for i in dropped
                    if 0 <= i < len(source_classes)
                )
                preview_text += (
                    "\n\nThese source classes will be dropped: " + dropped_names
                )
            answer = QMessageBox.question(
                self,
                "Class Mismatch",
                preview_text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            force_remap = True

        self._pending_add = (selected_path, selection, existing_paths)
        worker = SourceImportWorker(
            selected_path,
            self._project.project_dir,
            import_mode,
            source_classes,
            project_classes,
            force_remap,
            parent=self,
        )
        self._import_worker = worker
        self._set_busy("Preparing source import…")
        worker.status.connect(self._import_status.setText)
        worker.progress.connect(self._import_progress.setValue)
        worker.finished.connect(self._finish_import)
        worker.start()

    def _set_busy(self, status: str, *, indeterminate: bool = False) -> None:
        self.btn_add.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Close).setEnabled(False)
        self._import_status.setText(status)
        self._import_status.show()
        self._import_progress.setRange(0, 0 if indeterminate else 100)
        self._import_progress.setValue(0)
        self._import_progress.show()

    def _clear_busy(self) -> None:
        self.btn_add.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self._buttons.button(QDialogButtonBox.StandardButton.Close).setEnabled(True)
        self._import_status.hide()
        self._import_progress.hide()

    def _finish_import(self) -> None:
        worker = self._import_worker
        pending = self._pending_add
        self._import_worker = None
        self._pending_add = None
        self._clear_busy()
        if worker is None or pending is None:
            return
        worker.deleteLater()
        if worker.failure_exception is not None:
            QMessageBox.warning(self, "Add Source", str(worker.failure_exception))
            return

        materialized = worker.result
        selected_path, selection, existing_paths = pending

        canonical_path = str(materialized.canonical_path)
        original_path = str(materialized.source_root)
        if canonical_path in existing_paths or original_path in existing_paths:
            QMessageBox.information(self, "Add Source", "Source already added.")
            return

        # An AL round's manifest declares which level is authoritative --
        # trust that over the validation dialog's re-scanned `selection.level`
        # (label files in an AL export are 9-field quads for every level, so
        # a re-scan cannot tell OBB apart from an axis-aligned-quad AABB;
        # see resolve_al_round_authoritative_level's docstring).
        manifest_level = resolve_al_round_authoritative_level(selected_path)
        level = (
            manifest_level
            if manifest_level is not None
            else (
                selection.level
                if getattr(selection, "level", None)
                else materialized.level
            )
        )

        self._project.sources.append(
            OBBSource(
                path=canonical_path,
                name=Path(selected_path).name,
                original_path=original_path,
                source_kind=materialized.source_kind,
                imported=materialized.imported,
                level=level,
            )
        )
        self._refresh_list()

    def reject(self) -> None:
        if self._inspection_worker is None and self._import_worker is None:
            super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._inspection_worker is not None or self._import_worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def _remove_selected(self) -> None:
        if self._inspection_worker is not None or self._import_worker is not None:
            return
        row = self._source_list.currentRow()
        if row < 0 or row >= len(self._project.sources):
            return
        removed = self._project.sources.pop(row)
        pending = removed.staged_review
        if pending is not None and pending.staged_path:
            # Bounded delete: staged_path round-trips through the saved
            # project file, so it is untrusted input from disk.
            remove_staged_escalation_dir(pending.staged_path, self._project.project_dir)
        self._refresh_list()
