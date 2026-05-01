"""SourceManagerDialog — view / add / remove image source folders."""

import sys
from pathlib import Path
from typing import List

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.classkit.gui.dialogs.source_validation import (
    count_classkit_images,
    inspect_classkit_source_dir,
)
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.dialogs import (
    HYDRA_DIALOG_MUTED_TEXT_COLOR,
    HYDRA_DIALOG_STYLE,
)

CLASSKIT_SIEVE_THRESHOLD = 5000

_FILTERKIT_TRANSACTION_FILE = ".filterkit_last_transaction.json"


def _get_transaction_mtime(dataset_root: Path) -> "float | None":
    """Return the mtime of FilterKit's transaction file, or None if absent."""
    p = dataset_root / _FILTERKIT_TRANSACTION_FILE
    try:
        return p.stat().st_mtime
    except OSError:
        return None


class SourceManagerDialog(QDialog):
    """Full source manager showing existing sources with add/remove capability."""

    def __init__(self, db_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Source Manager")
        self.setMinimumSize(640, 480)
        self.setStyleSheet(HYDRA_DIALOG_STYLE)

        self._db_path = db_path
        self._to_add: List[Path] = []
        self._to_remove: List[str] = []
        self._existing: List[dict] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel(
            "<b>Image Sources</b><br>"
            "Manage the folders that supply images to this project. "
            "ClassKit links accepted source folders by default — image files stay "
            "where they are and the database records their original paths. Use "
            "<b>Make Project Portable</b> on the toolbar to copy linked images into "
            "the project bundle. "
            "Adding a folder will ingest its images; removing one will "
            "delete those images from the database."
            "<br><br>"
            "<b>Supported dataset formats:</b>"
            "<ul style='margin-top:4px; margin-bottom:0;'>"
            "<li><b>Flat image folder</b> — images directly in the selected folder "
            "(or inside an <code>images/</code> subdirectory)</li>"
            "<li><b>COCO JSON</b> — dataset root containing <code>annotations.json</code> "
            "or an <code>annotations/</code> directory</li>"
            "<li><b>YOLO OBB / Detect</b> — dataset root with train/val/test splits "
            "and <code>.txt</code> label files</li>"
            "<li><b>Class-folder</b> — subdirectories named after class labels, "
            "optionally under train/val/test splits</li>"
            "</ul>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setMinimumHeight(200)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()

        btn_add = QPushButton("Add Folder\u2026")
        btn_add.setStyleSheet(
            "QPushButton { background-color:#1a4a1a; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:8px 16px; }"
            "QPushButton:hover { background-color:#235a23; }"
        )
        btn_add.clicked.connect(self._browse_add)
        btn_row.addWidget(btn_add)

        btn_row.addStretch(1)

        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.setStyleSheet(
            "QPushButton { background-color:#6b1c1c; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:8px 16px; }"
            "QPushButton:hover { background-color:#8b2424; }"
        )
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_remove.setEnabled(False)
        btn_row.addWidget(self._btn_remove)

        layout.addLayout(btn_row)

        self._list.itemSelectionChanged.connect(
            lambda: self._btn_remove.setEnabled(bool(self._list.selectedItems()))
        )

        self._summary = QLabel("")
        self._summary.setStyleSheet(
            f"color: {HYDRA_DIALOG_MUTED_TEXT_COLOR}; font-size: 11px;"
        )
        layout.addWidget(self._summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_existing_sources()

    def _load_existing_sources(self):
        from hydra_suite.classkit.core.store.db import ClassKitDB

        db = ClassKitDB(self._db_path)
        self._existing = db.get_source_folders()
        self._rebuild_list()

    def _rebuild_list(self):
        self._list.clear()
        total_images = 0

        for src in self._existing:
            folder = src["folder"]
            count = src["count"]
            is_pending_remove = folder in self._to_remove
            item = QListWidgetItem()
            display = f"{Path(folder).name}  \u2014  {count:,} images\n{folder}"
            if is_pending_remove:
                display = f"[REMOVING]  {display}"
                item.setForeground(Qt.GlobalColor.darkRed)
            item.setText(display)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            self._list.addItem(item)
            if not is_pending_remove:
                total_images += count

        for add_path in self._to_add:
            count = self._count_images(add_path)
            item = QListWidgetItem()
            item.setText(
                f"[NEW]  {add_path.name}  \u2014  {count:,} images\n{add_path}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(add_path))
            item.setForeground(Qt.GlobalColor.darkGreen)
            self._list.addItem(item)
            total_images += count

        n_sources = len(self._existing) - len(self._to_remove) + len(self._to_add)
        self._summary.setText(
            f"{n_sources} source(s)  \u00b7  ~{total_images:,} images total"
        )

    @staticmethod
    def _count_images(folder: Path) -> int:
        """Return the display count for a pending source folder."""
        return count_classkit_images(folder)

    def _browse_add(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing images", str(Path.home())
        )
        if not path:
            return

        d = Path(path).expanduser().resolve()
        try:
            resolved = self._resolve_selected_source(d)
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid Source Folder",
                str(exc),
            )
            return
        if resolved is None:
            return

        count = count_classkit_images(resolved)

        existing_folders = {s["folder"] for s in self._existing}
        pending_add_folders = {str(p) for p in self._to_add}
        if str(resolved) in existing_folders or str(resolved) in pending_add_folders:
            QMessageBox.warning(
                self, "Already Added", "That folder is already a source."
            )
            return

        if count > CLASSKIT_SIEVE_THRESHOLD:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Large Dataset Detected")
            msg.setText(
                f"This folder contains {count:,} images \u2014 that is a lot to label.\n\n"
                "FilterKit can reduce near-duplicates and create a smaller representative subset before labeling."
            )
            msg.setInformativeText(
                "Open this folder in FilterKit now, or add it as-is?"
            )
            btn_sieve = msg.addButton("Open in FilterKit", QMessageBox.AcceptRole)
            btn_add = msg.addButton("Add Anyway", QMessageBox.DestructiveRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_sieve:
                pre_mtime = _get_transaction_mtime(d)
                proc = QProcess(self)
                proc.finished.connect(
                    lambda _ec, _es, _d=d, _r=resolved, _m=pre_mtime: (
                        self._on_filterkit_closed(_d, _r, _m)
                        if self.isVisible()
                        else None
                    )
                )
                proc.errorOccurred.connect(
                    lambda _err: QMessageBox.warning(
                        self, "Launch Failed", "Could not launch FilterKit."
                    )
                )
                proc.start(sys.executable, ["-m", "hydra_suite.filterkit", str(d)])
                return
            if clicked != btn_add:
                return

        self._to_add.append(resolved)
        self._rebuild_list()

    def _on_filterkit_closed(self, d: Path, resolved: Path, pre_mtime) -> None:
        """Called when FilterKit exits. Auto-add folder if filtering was applied."""
        import json as _json

        post_mtime = _get_transaction_mtime(d)
        if post_mtime == pre_mtime:
            reply = QMessageBox.question(
                self,
                "FilterKit Closed",
                "FilterKit was closed without applying a filter.\n\n"
                "Add the original folder as-is?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        add_path = resolved
        tx_file = d / _FILTERKIT_TRANSACTION_FILE
        if tx_file.exists():
            try:
                tx = _json.loads(tx_file.read_text(encoding="utf-8"))
                output_path = tx.get("output_path")
                if output_path and Path(output_path).exists():
                    add_path = Path(output_path)
            except Exception:
                pass

        self._to_add.append(add_path)
        self._rebuild_list()

    def _resolve_selected_source(self, dataset_root: Path) -> Path | None:
        """Validate a selected source folder and return its dataset root."""
        inspect_classkit_source_dir(dataset_root)
        return dataset_root.resolve()

    def _confirm_standardization(self, inspection) -> bool:
        """Ask whether a flat image folder should be converted into dataset_root/images."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Standardize Source Folder")
        msg.setText(
            "This folder contains compatible images directly in the selected "
            "directory instead of inside images/."
        )
        msg.setInformativeText(
            f"ClassKit can create images/ and copy {inspection.images_count:,} "
            "image(s) into it before import.\n\n"
            "This may require additional disk space. Continue?"
        )
        btn_standardize = msg.addButton("Standardize and Add", QMessageBox.AcceptRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec()
        return msg.clickedButton() == btn_standardize

    def _remove_selected(self):
        for item in self._list.selectedItems():
            folder = item.data(Qt.ItemDataRole.UserRole)
            if not folder:
                continue
            folder_path = Path(folder).resolve()
            for i, p in enumerate(self._to_add):
                if p == folder_path or str(p) == folder:
                    self._to_add.pop(i)
                    break
            else:
                if folder not in self._to_remove:
                    self._to_remove.append(folder)
        self._rebuild_list()

    @property
    def folders_to_add(self) -> List[Path]:
        return list(self._to_add)

    @property
    def folders_to_remove(self) -> List[str]:
        return list(self._to_remove)

    @property
    def has_changes(self) -> bool:
        return bool(self._to_add) or bool(self._to_remove)
