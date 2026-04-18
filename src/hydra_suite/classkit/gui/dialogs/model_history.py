"""ModelHistoryDialog — browse, rename, delete, and load trained models."""

import json
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ModelHistoryDialog(QDialog):
    """Browse, rename, delete, and load trained models from project history."""

    def __init__(
        self,
        model_entries: list,
        project_path=None,
        db_path: Optional[Path] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Previously Trained Models")
        self.setMinimumSize(940, 620)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        self._entries = list(model_entries)
        self._project_path = project_path
        self._db_path = Path(db_path) if db_path else None
        self._selected_entry_data = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QLabel(
            "Select, rename, delete, or export trained checkpoints before loading one for inference."
        )
        header.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(header)

        self._qt_align = Qt.AlignVCenter | Qt.AlignLeft
        self._table_item_type = QTableWidgetItem

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Timestamp", "Mode", "Classes", "Val Acc"]
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
            "gridline-color: #3a3a3a; border: none; }"
            "QTableWidget::item:selected { background-color: #094771; }"
            "QHeaderView::section { background-color: #2d2d2d; color: #ffffff; "
            "padding: 4px; border: none; border-right: 1px solid #3a3a3a; }"
        )
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setFixedHeight(104)
        self.detail_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail_label.setStyleSheet(
            "background: #252526; color: #ffffff; font-size: 11px; "
            "padding: 7px 10px; border-radius: 4px;"
        )
        layout.addWidget(self.detail_label)

        btn_row = QHBoxLayout()

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.setFixedHeight(36)
        self.rename_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #ffffff; border-radius: 4px; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        self.rename_btn.clicked.connect(self._rename_selected)
        btn_row.addWidget(self.rename_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setFixedHeight(36)
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: #553333; color: #ffcccc; border-radius: 4px; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #7a3f3f; color: white; }"
        )
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.delete_btn)

        self.export_btn = QPushButton("Export to models/")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #ffffff; border-radius: 4px; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #4a6b4a; color: white; }"
        )
        self.export_btn.setToolTip(
            "Copy this model's artifact files to the project models/ directory"
        )
        self.export_btn.clicked.connect(self._export_selected)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch(1)

        self.load_btn = QPushButton("Load and Run Inference")
        self.load_btn.setFixedHeight(36)
        self.load_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border-radius: 4px; "
            "padding: 0 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #666; }"
        )
        self.load_btn.clicked.connect(self._accept_selection)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #ffffff; border-radius: 4px; "
            "padding: 0 16px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        self._refresh_table(select_row=0)

    def _display_name(self, entry: dict) -> str:
        name = str(entry.get("display_name") or "").strip()
        if name:
            return name
        return str(entry.get("mode") or "model")

    def _refresh_table(self, select_row: int | None = None) -> None:
        self.table.setRowCount(0)
        for row, entry in enumerate(self._entries):
            self.table.insertRow(row)
            ts = str(entry.get("timestamp", ""))[:19]
            mode = str(entry.get("mode", ""))
            names = ", ".join(entry.get("class_names") or [])
            acc = entry.get("best_val_acc")
            acc_str = f"{acc:.3f}" if acc is not None else "\u2014"
            display_name = self._display_name(entry)

            values = [display_name, ts, mode, names, acc_str]
            for col, text in enumerate(values):
                item = self._table_item_type(text)
                item.setTextAlignment(self._qt_align)
                self.table.setItem(row, col, item)

        if self._entries:
            target = (
                0
                if select_row is None
                else max(0, min(select_row, len(self._entries) - 1))
            )
            self.table.selectRow(target)
        else:
            self.detail_label.setText("No model history entries remain.")
            self.rename_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.load_btn.setEnabled(False)

    def _selected_row(self) -> int:
        return int(self.table.currentRow())

    def _get_selected_entry(self) -> dict | None:
        row = self._selected_row()
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def _on_double_click(self, index):
        self._accept_selection()

    def _accept_selection(self):
        entry = self._get_selected_entry()
        if entry is not None:
            self._selected_entry_data = entry
            self.accept()

    def selected_entry(self):
        return self._selected_entry_data

    def _on_selection_changed(self):
        entry = self._get_selected_entry()
        has_entry = entry is not None
        self.rename_btn.setEnabled(has_entry)
        self.delete_btn.setEnabled(has_entry)
        self.export_btn.setEnabled(has_entry)
        self.load_btn.setEnabled(has_entry)

        if not has_entry:
            self.detail_label.setText("")
            return

        ts = str(entry.get("timestamp", ""))
        mode = str(entry.get("mode", ""))
        display_name = self._display_name(entry)
        names = entry.get("class_names") or []
        acc = entry.get("best_val_acc")
        acc_str = f"{acc:.4f}" if acc is not None else "\u2014"
        artifact_paths = entry.get("artifact_paths") or []
        filenames = [Path(p).name for p in artifact_paths]
        classes_html = ", ".join(names) if names else "\u2014"
        files_html = (
            "  ".join(f"<span style='color:#9cdcfe'>{f}</span>" for f in filenames)
            if filenames
            else "\u2014"
        )
        self.detail_label.setText(
            f"<b>{display_name}</b> &nbsp;&bull;&nbsp; Mode: <b>{mode}</b>"
            f" &nbsp;&bull;&nbsp; Val acc: <b>{acc_str}</b><br>"
            f"<span style='color:#ffffff'>Classes:</span> {classes_html}<br>"
            f"<span style='color:#ffffff'>Files:</span> {files_html}<br>"
            f"<span style='color:#ffffff;font-size:10px'>{ts}</span>"
        )

    def _rename_selected(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        if self._db_path is None:
            QMessageBox.warning(
                self, "Rename Failed", "No project database is available."
            )
            return

        current_name = self._display_name(entry)
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model Entry",
            "Display name:",
            text=current_name,
        )
        if not ok:
            return

        from hydra_suite.classkit.core.store.db import ClassKitDB

        updated = ClassKitDB(self._db_path).set_model_cache_display_name(
            int(entry.get("id", -1)),
            str(new_name).strip(),
        )
        if updated < 1:
            QMessageBox.warning(
                self,
                "Rename Failed",
                "Could not update this history entry in the database.",
            )
            return

        entry["display_name"] = str(new_name).strip()
        row = self._selected_row()
        self._refresh_table(select_row=row)

    def _confirm_delete_selected(self) -> Optional[bool]:
        """Return whether to delete artifact files, or None if cancelled."""
        prompt = QMessageBox(self)
        prompt.setIcon(QMessageBox.Warning)
        prompt.setWindowTitle("Delete Model Entry")
        prompt.setText("Delete this checkpoint entry from model history?")
        prompt.setInformativeText(
            "You can remove only the DB record, or delete both the record and artifact files."
        )
        remove_record_btn = prompt.addButton(
            "Remove History Entry", QMessageBox.AcceptRole
        )
        delete_files_btn = prompt.addButton(
            "Delete Entry + Files", QMessageBox.DestructiveRole
        )
        prompt.addButton(QMessageBox.Cancel)
        prompt.setDefaultButton(remove_record_btn)
        prompt.exec()

        clicked = prompt.clickedButton()
        if clicked == remove_record_btn:
            return False
        if clicked == delete_files_btn:
            return True
        return None

    @staticmethod
    def _delete_artifacts(entry: dict) -> tuple[int, list[str]]:
        """Delete artifact files recorded on an entry."""
        deleted_files = 0
        file_errors: list[str] = []
        for src in entry.get("artifact_paths") or []:
            src_path = Path(src)
            if not src_path.exists():
                continue
            try:
                src_path.unlink()
                deleted_files += 1
            except Exception as exc:
                file_errors.append(f"{src_path.name}: {exc}")
        return deleted_files, file_errors

    def _remove_selected_entry_from_db(self, entry: dict) -> bool:
        """Delete the selected model-history row from the database."""
        from hydra_suite.classkit.core.store.db import ClassKitDB

        deleted_rows = ClassKitDB(self._db_path).delete_model_cache_entry(
            int(entry.get("id", -1))
        )
        return deleted_rows >= 1

    def _remove_entry_from_view(self, entry: dict) -> None:
        """Remove an entry from the local list and refresh selection."""
        keep_id = int(entry.get("id", -1))
        self._entries = [e for e in self._entries if int(e.get("id", -1)) != keep_id]
        row = self._selected_row()
        self._refresh_table(select_row=max(0, row - 1))

    def _show_delete_result(
        self, delete_files: bool, deleted_files: int, file_errors: list[str]
    ) -> None:
        """Render the final delete status message."""
        if file_errors:
            QMessageBox.warning(
                self,
                "Entry Deleted With Warnings",
                f"Removed history entry. Deleted files: {deleted_files}\n\n"
                + "\n".join(file_errors),
            )
            return

        if delete_files:
            self.detail_label.setText(
                f"<span style='color:#4ec9b0'>Removed history entry and deleted {deleted_files} artifact file(s).</span>"
            )
            return

        self.detail_label.setText(
            "<span style='color:#4ec9b0'>Removed history entry from the database.</span>"
        )

    def _delete_selected(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        if self._db_path is None:
            QMessageBox.warning(
                self, "Delete Failed", "No project database is available."
            )
            return

        delete_files = self._confirm_delete_selected()
        if delete_files is None:
            return

        deleted_files, file_errors = (
            self._delete_artifacts(entry) if delete_files else (0, [])
        )
        if not self._remove_selected_entry_from_db(entry):
            QMessageBox.warning(
                self,
                "Delete Failed",
                "Could not remove this history entry from the database.",
            )
            return

        self._remove_entry_from_view(entry)
        self._show_delete_result(delete_files, deleted_files, file_errors)

    @staticmethod
    def _slug(text: object, max_len: int = 48) -> str:
        """Convert free text to a short filesystem-safe slug."""
        raw = str(text or "").strip().lower()
        out = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
        if not out:
            return ""
        while "__" in out:
            out = out.replace("__", "_")
        return out[:max_len]

    def _resolve_export_dir(self) -> Path:
        """Return and create the destination models directory."""
        if self._project_path:
            dest_dir = Path(str(self._project_path)) / "models"
        else:
            dest_dir = Path.cwd() / "models"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir

    def _build_export_stem(self, entry: dict) -> str:
        """Build the descriptive base filename used for exported artifacts."""
        mode_slug = self._slug(entry.get("mode", "model"), max_len=32) or "model"
        class_names = entry.get("class_names") or []
        class_slug = "-".join(
            part
            for part in (self._slug(name, max_len=16) for name in class_names)
            if part
        )
        if not class_slug:
            class_slug = "classes"
        if len(class_slug) > 48:
            class_slug = f"{max(1, len(class_names))}cls"
        ts_raw = str(entry.get("timestamp", "")).strip()
        ts_slug = "".join(ch for ch in ts_raw if ch.isdigit())[:14] or "unknown"
        return f"classkit_{mode_slug}_{class_slug}_{ts_slug}"

    @classmethod
    def _build_export_filename(
        cls, src_path: Path, base_stem: str, index: int, total_artifacts: int
    ) -> str:
        """Build the preferred destination filename for one artifact."""
        ext = src_path.suffix or ".bin"
        hint = cls._slug(src_path.stem, max_len=24)
        if hint in {"", "best", "last", "weights", "model", "checkpoint"}:
            hint = ""
        if total_artifacts == 1:
            return f"{base_stem}{ext}"
        if hint:
            return f"{base_stem}_{hint}{ext}"
        return f"{base_stem}_f{index + 1}{ext}"

    @staticmethod
    def _dedupe_export_path(dest_dir: Path, desired_name: str) -> Path:
        """Append a numeric suffix until the export path is unique."""
        dst = dest_dir / desired_name
        suffix_counter = 1
        while dst.exists():
            ext = dst.suffix
            dst = dest_dir / f"{dst.stem}_{suffix_counter}{ext}"
            suffix_counter += 1
        return dst

    @classmethod
    def _copy_artifacts_to_dir(
        cls, artifact_paths: list, dest_dir: Path, base_stem: str
    ) -> tuple[list[str], list[str]]:
        """Copy artifacts into the export directory and report successes/failures."""
        from hydra_suite.training.model_publish import _copy_classifier_sidecar

        copied: list[str] = []
        failed: list[str] = []
        total_artifacts = len(artifact_paths)

        for idx, src in enumerate(artifact_paths):
            src_path = Path(src)
            if not src_path.exists():
                failed.append(f"{src_path.name}: file not found")
                continue

            desired_name = cls._build_export_filename(
                src_path, base_stem, idx, total_artifacts
            )
            dst = cls._dedupe_export_path(dest_dir, desired_name)
            try:
                shutil.copy2(str(src_path), str(dst))
                copied.append(dst.name)
                if _copy_classifier_sidecar(src_path, dst):
                    copied.append(dst.with_suffix(".v2meta.json").name)
            except Exception as exc:
                failed.append(f"{src_path.name}: {exc}")

        return copied, failed

    def _show_export_result(
        self, dest_dir: Path, copied: list[str], failed: list[str]
    ) -> None:
        """Render the export result in the dialog and a message box."""
        if copied:
            self.detail_label.setText(
                f"<span style='color:#4ec9b0'>Exported \u2192 {dest_dir}:</span> "
                + "  ".join(f"<b>{f}</b>" for f in copied)
                + (
                    f"<br><span style='color:#f88'>Skipped: {'; '.join(failed)}</span>"
                    if failed
                    else ""
                )
            )
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(copied)} file(s) to:\n{dest_dir}"
                + ("\n\nSkipped:\n" + "\n".join(failed) if failed else ""),
            )
            return

        QMessageBox.warning(
            self,
            "Export Failed",
            "No model files could be exported.\n\n" + "\n".join(failed),
        )

    def _export_selected(self):
        """Copy selected artifacts to project ``models/`` using descriptive names."""
        entry = self._get_selected_entry()
        if entry is None:
            return
        artifact_paths = entry.get("artifact_paths") or []
        if not artifact_paths:
            QMessageBox.warning(
                self, "No Artifacts", "No model files are recorded for this entry."
            )
            return

        try:
            dest_dir = self._resolve_export_dir()
        except Exception as exc:
            project_models_dir = (
                Path(str(self._project_path)) / "models"
                if self._project_path
                else (Path.cwd() / "models")
            )
            QMessageBox.warning(
                self,
                "Export Failed",
                f"Could not create export directory:\n{project_models_dir}\n\n{exc}",
            )
            return
        base_stem = self._build_export_stem(entry)
        copied, failed = self._copy_artifacts_to_dir(
            artifact_paths, dest_dir, base_stem
        )
        copied_artifact_paths = [
            dest_dir / name for name in copied if not str(name).endswith(".v2meta.json")
        ]
        training_settings = ((entry.get("meta") or {}).get("training_settings")) or {}
        mode = str(entry.get("mode") or "")
        fallback_input_size: tuple[int, int] | None = None
        if "yolo" in mode:
            fallback_input_size = (640, 640)
        elif (
            "custom" in mode
            and training_settings.get("custom_backbone") != "tinyclassifier"
        ):
            size = int(training_settings.get("custom_input_size", 224) or 224)
            fallback_input_size = (size, size)
        fallback_monochrome = bool(training_settings.get("monochrome", False))

        if copied_artifact_paths and "yolo" in mode:
            try:
                from hydra_suite.training.model_publish import (
                    classifier_metadata_for_artifact,
                )

                for artifact_path in copied_artifact_paths:
                    if artifact_path.suffix.lower() != ".pt":
                        continue
                    sidecar_path = artifact_path.with_suffix(".v2meta.json")
                    if sidecar_path.exists():
                        continue
                    meta = classifier_metadata_for_artifact(
                        artifact_path,
                        fallback_input_size=fallback_input_size,
                        fallback_monochrome=fallback_monochrome,
                    )
                    sidecar_path.write_text(
                        json.dumps(meta, indent=2), encoding="utf-8"
                    )
                    copied.append(sidecar_path.name)
            except Exception as exc:
                failed.append(f"sidecar metadata: {exc}")

        if (
            copied_artifact_paths
            and len(copied_artifact_paths) > 1
            and str(entry.get("mode", "")).startswith("multihead")
        ):
            try:
                from hydra_suite.classkit.model_bundle import (
                    write_model_bundle_manifest,
                )
                from hydra_suite.training.model_publish import (
                    classifier_metadata_for_artifact,
                    write_classifier_multihead_manifest,
                )

                manifest_name = f"{base_stem}.bundle.json"
                write_model_bundle_manifest(
                    dest_dir / manifest_name,
                    mode=str(entry.get("mode") or ""),
                    artifact_paths=copied_artifact_paths,
                    class_names=list(entry.get("class_names") or []),
                )
                copied = copied + [manifest_name]

                factor_entries = []
                bundle_input_size: tuple[int, int] | None = None
                bundle_monochrome = fallback_monochrome
                for idx, artifact_path in enumerate(copied_artifact_paths):
                    meta = classifier_metadata_for_artifact(
                        artifact_path,
                        fallback_input_size=fallback_input_size,
                        fallback_monochrome=fallback_monochrome,
                    )
                    input_size_list = meta.get("input_size") or [224, 224]
                    if bundle_input_size is None:
                        bundle_input_size = (
                            int(input_size_list[0]),
                            int(input_size_list[1]),
                        )
                    bundle_monochrome = bool(meta.get("monochrome", bundle_monochrome))
                    factor_name = str(
                        (meta.get("factor_names") or [f"factor_{idx + 1}"])[0]
                    )
                    factor_entries.append(
                        {
                            "factor": factor_name,
                            "path": artifact_path,
                            "class_names": list(
                                (meta.get("class_names_per_factor") or [[]])[0]
                            ),
                        }
                    )
                trackerkit_manifest = f"{base_stem}.multihead.json"
                write_classifier_multihead_manifest(
                    dest_dir / trackerkit_manifest,
                    factor_entries=factor_entries,
                    input_size=bundle_input_size or fallback_input_size or (224, 224),
                    monochrome=bundle_monochrome,
                )
                copied = copied + [trackerkit_manifest]
            except Exception as exc:
                failed.append(f"bundle manifest: {exc}")
        self._show_export_result(dest_dir, copied, failed)
