"""AddSourceDialog — pick image source folders for a ClassKit project."""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QProcess
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
    CLASSKIT_IMAGES_SUBDIR,
    count_classkit_images,
    inspect_classkit_source_dir,
)
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.dialogs import HYDRA_DIALOG_STYLE

CLASSKIT_SIEVE_THRESHOLD = 5000

_FILTERKIT_TRANSACTION_FILE = ".filterkit_last_transaction.json"


def _get_transaction_mtime(dataset_root: Path) -> "float | None":
    """Return the mtime of FilterKit's transaction file, or None if absent."""
    p = dataset_root / _FILTERKIT_TRANSACTION_FILE
    try:
        return p.stat().st_mtime
    except OSError:
        return None


class AddSourceDialog(QDialog):
    """Pick one or more image source folders for a ClassKit project.

    Accepted roots can be flat image folders, folders with an ``images/``
    subdirectory, COCO / YOLO dataset roots, or train/val class-folder datasets.
    ClassKit links accepted source folders by default — image files stay where
    they are and the database records their original paths. Use
    "Make Project Portable" later to copy them into the project bundle.
    """

    def __init__(
        self, existing_sources: Optional[List[Path]] = None, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Image Sources")
        self.setMinimumWidth(580)
        self.setStyleSheet(HYDRA_DIALOG_STYLE)

        # Resolved dataset roots already added (for duplicate checks).
        self._existing: List[Path] = [
            p.expanduser().resolve() for p in (existing_sources or [])
        ]
        # (dataset_root, resolved_dataset_root, description)
        self._sources: List[Tuple[Path, Path, str]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info = QLabel(
            "Add one or more dataset root folders. Sources may be flat image folders, "
            f"folders with an <b>{CLASSKIT_IMAGES_SUBDIR}/</b> subdirectory, COCO / YOLO roots, "
            "or train/val class-folder datasets. ClassKit links these folders by default — "
            "images stay where they are. Use <b>Make Project Portable</b> later to copy "
            "them into the project bundle."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = QListWidget()
        self._list.setMinimumHeight(120)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Folder\u2026")
        btn_add.clicked.connect(self._browse)
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

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Done")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing images", str(Path.home())
        )
        if not path:
            return

        d = Path(path).expanduser().resolve()

        try:
            inspection = inspect_classkit_source_dir(d)
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

        # Duplicate check
        if resolved in self._existing or any(
            r == resolved for _, r, _ in self._sources
        ):
            QMessageBox.warning(
                self, "Already Added", "That folder has already been added."
            )
            return

        # FilterKit check
        if count > CLASSKIT_SIEVE_THRESHOLD:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Large Dataset Detected")
            msg.setText(
                f"This folder contains {count:,} images \u2014 that is a lot to label.\n\n"
                "FilterKit can reduce near-duplicates and create a smaller "
                "representative subset before labeling."
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

        self._sources.append((d, resolved, d.name))
        usage_text = (
            f"detected {inspection.source_kind} import \u2014 linked"
            if inspection.source_kind != "images"
            else "linked source"
        )
        item = QListWidgetItem(
            f"{d.name}  \u2014  {count:,} images  ({usage_text})\n{d}"
        )
        self._list.addItem(item)

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

        count = count_classkit_images(add_path)
        self._sources.append((d, add_path, d.name))
        self._list.addItem(
            QListWidgetItem(f"{d.name}  \u2014  {count:,} images  (linked source)\n{d}")
        )

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
            f"ClassKit can create {CLASSKIT_IMAGES_SUBDIR}/ and copy "
            f"{inspection.images_count:,} image(s) into it before import.\n\n"
            "This may require additional disk space. Continue?"
        )
        btn_standardize = msg.addButton("Standardize and Add", QMessageBox.AcceptRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec()
        return msg.clickedButton() == btn_standardize

    def _remove_selected(self):
        for item in self._list.selectedItems():
            row = self._list.row(item)
            self._list.takeItem(row)
            if row < len(self._sources):
                self._sources.pop(row)

    @property
    def sources(self) -> List[Tuple[Path, Path, str]]:
        """List of (dataset_root, resolved_dataset_root, description)."""
        return list(self._sources)
