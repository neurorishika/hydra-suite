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

from ..source_import import (
    IMPORT_MODE_LINKED,
    IMPORT_MODE_PORTABLE,
    compute_positional_class_remap,
    inspect_al_round,
    inspect_detectkit_source,
    materialize_al_round,
    materialize_detectkit_source,
    remap_materialized_source_classes,
)
from .source_validation import (
    SOURCE_ADD_MODE_LINKED,
    SOURCE_ADD_MODE_PORTABLE,
    confirm_detectkit_source_addition,
)

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

        # An active-learning export round (manifest.json + one sibling dataset
        # root per geometry level, e.g. obb/ + aabb/) is reviewed/imported as a
        # whole: every sibling gets registered, linked back to the
        # authoritative root via derived_from, mirroring how
        # jobs/al_worker.py registers an internally generated round. This
        # keeps a manually-added external round compatible with the same
        # role-gating/SAM2-escalation machinery an internal round gets,
        # instead of silently dropping every non-authoritative level.
        al_roots = inspect_al_round(selected_path)
        if al_roots is not None:
            inspection = al_roots[0].inspection
        else:
            try:
                inspection = inspect_detectkit_source(selected_path)
            except Exception as exc:
                QMessageBox.warning(self, "Add Source", str(exc))
                return

        # inspection.dataset_root may differ from selected_path when the user
        # picked an active-learning round container -- review the resolved
        # (authoritative) dataset root, not the container.
        selection = confirm_detectkit_source_addition(
            self, str(inspection.dataset_root), inspection
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

        if al_roots is not None:
            self._add_al_round_sources(
                selected_path,
                existing_paths,
                import_mode,
                force_remap,
                source_classes,
                project_classes,
            )
            return

        try:
            materialized = materialize_detectkit_source(
                selected_path,
                self._project.project_dir,
                import_mode=import_mode,
                force_import=force_remap,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Add Source", str(exc))
            return

        if force_remap:
            try:
                remap = compute_positional_class_remap(source_classes, project_classes)
                remap_materialized_source_classes(
                    Path(materialized.canonical_path),
                    project_classes,
                    remap,
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
                level=(
                    selection.level
                    if getattr(selection, "level", None)
                    else materialized.level
                ),
            )
        )
        self._refresh_list()

    def _add_al_round_sources(
        self,
        round_path: str,
        existing_paths: set[str],
        import_mode: str,
        force_remap: bool,
        source_classes: list[str],
        project_classes: list[str],
    ) -> None:
        """Materialize + register every sibling root of an AL round export.

        One project source per geometry level, non-authoritative roots linked
        back to the authoritative one via ``derived_from`` -- matching
        ``jobs/al_worker.py``'s registration of an internally generated round.
        """
        from ..models import OBBSource

        try:
            materialized_roots = materialize_al_round(
                round_path,
                self._project.project_dir,
                import_mode=import_mode,
                force_import=force_remap,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Add Source", str(exc))
            return

        # Every sibling shares one class list (export_al_dataset writes the
        # same reconciled class_names to each root), so one remap covers the
        # whole round -- compute it once rather than per-entry, and re-check
        # the SAME already-registered path map for the whole round rather
        # than recomputing it as entries get appended one at a time.
        remap = (
            compute_positional_class_remap(source_classes, project_classes)
            if force_remap
            else None
        )
        name_by_existing_path: dict[str, str] = {}
        for src in self._project.sources:
            for candidate in (src.path, src.original_path):
                if candidate:
                    name_by_existing_path[candidate] = src.name

        round_name = Path(round_path).name
        authoritative_name: str | None = None
        added_any = False
        for entry in materialized_roots or []:
            materialized = entry.materialized
            canonical_path = str(materialized.canonical_path)
            original_path = str(materialized.source_root)

            if remap is not None:
                try:
                    remap_materialized_source_classes(
                        Path(materialized.canonical_path), project_classes, remap
                    )
                except Exception as exc:
                    QMessageBox.warning(self, "Add Source", str(exc))
                    # Stop importing further siblings, but keep -- and show
                    # -- whatever was already registered this call rather
                    # than returning with the sources list and the dialog's
                    # list widget silently out of sync.
                    break

            is_duplicate = (
                canonical_path in existing_paths or original_path in existing_paths
            )
            if entry.authoritative and authoritative_name is None:
                # Even if the authoritative root was already registered
                # (e.g. previously added on its own, before this round was
                # added as a whole), resolve derived_from to its ACTUAL
                # registered name -- not the name this call would have used
                # -- so a skipped duplicate never leaves a derived sibling's
                # derived_from silently None (which would make it look
                # authoritative instead of just unlinked).
                if is_duplicate:
                    authoritative_name = name_by_existing_path.get(
                        canonical_path
                    ) or name_by_existing_path.get(original_path)
                else:
                    authoritative_name = f"{round_name}_{entry.level}"

            if is_duplicate:
                continue

            name = f"{round_name}_{entry.level}"
            self._project.sources.append(
                OBBSource(
                    path=canonical_path,
                    name=name,
                    original_path=original_path,
                    source_kind=materialized.source_kind,
                    imported=materialized.imported,
                    # entry.level (the manifest's own declared level), not
                    # materialized.level: every AL-export root's labels are
                    # stored as 9-field quads regardless of level, and
                    # _detect_source_level hardcodes intended_level=OBB for
                    # that shape -- it can't tell an authoritative OBB root
                    # from an axis-aligned-quad AABB sibling apart by
                    # re-scanning. The manifest already recorded which is
                    # which at export time; trust it.
                    level=entry.level,
                    reviewed=entry.reviewed,
                    derived_from=None if entry.authoritative else authoritative_name,
                )
            )
            existing_paths.add(canonical_path)
            existing_paths.add(original_path)
            name_by_existing_path[canonical_path] = name
            name_by_existing_path[original_path] = name
            added_any = True

        if not added_any:
            QMessageBox.information(self, "Add Source", "Source already added.")
        self._refresh_list()

    def _remove_selected(self) -> None:
        row = self._source_list.currentRow()
        if row < 0 or row >= len(self._project.sources):
            return
        self._project.sources.pop(row)
        self._refresh_list()
