"""DetectKit source-validation dialog shown before adding a dataset source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.training.geometry_levels import (
    GeometryLevel,
    SourceLevelScan,
    scan_source_levels,
)
from hydra_suite.widgets.dialogs import HYDRA_DIALOG_MUTED_TEXT_COLOR, BaseDialog

from ..source_import import DetectKitSourceInspection

SOURCE_ADD_MODE_PORTABLE = "portable"
SOURCE_ADD_MODE_LINKED = "linked"


@dataclass(slots=True, frozen=True)
class DetectKitSourceAdditionChoice:
    """Choice returned by the DetectKit source review dialog."""

    mode: str
    level: str = "obb"


def resolve_source_level_or_block(
    labels_dir,
    intended_level: GeometryLevel = GeometryLevel.OBB,
    *,
    confirm: bool = False,
) -> SourceLevelScan:
    """Resolve a source's level, honoring an explicit quads-are-contours override."""
    return scan_source_levels(
        labels_dir, intended_level=intended_level, confirm_quads_are_polygons=confirm
    )


def _intended_level_for_kind(source_kind: str) -> GeometryLevel:
    if source_kind == "yolo_detect":
        return GeometryLevel.AABB
    return GeometryLevel.OBB


def _describe_source_kind(source_kind: str) -> str:
    descriptions = {
        "detectkit": "DetectKit canonical source",
        "yolo_detect": "YOLO detect dataset",
        "yolo_obb": "YOLO OBB dataset",
        "coco": "COCO annotations dataset",
        "detectkit_al": "Active-learning export (authoritative level root)",
    }
    return descriptions.get(source_kind, source_kind)


def _describe_portable_action(inspection: DetectKitSourceInspection) -> str:
    if inspection.requires_import:
        return (
            "Import as portable copies this source into the DetectKit project and "
            "normalizes it to DetectKit's canonical images/, labels/, and "
            "classes.txt layout."
        )
    return (
        "Import as portable copies this source into the DetectKit project so the "
        "project remains self-contained."
    )


def _describe_linked_action(inspection: DetectKitSourceInspection) -> str:
    if inspection.requires_import:
        return (
            "Keep at source links the dataset in place and normalizes it there. "
            "This will modify labels at the source dataset and may create or "
            "update DetectKit canonical images/, labels/, and classes.txt files "
            "in that source folder."
        )
    return (
        "Keep at source links the existing dataset in place. DetectKit may update "
        "labels at the source dataset if class remapping is required."
    )


class DetectKitSourceValidationDialog(BaseDialog):
    """Review a selected DetectKit source before adding it to the project."""

    def __init__(
        self,
        source_root: str | Path,
        inspection: DetectKitSourceInspection,
        parent=None,
    ) -> None:
        super().__init__(
            "Review Source Import",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        self.resize(720, 360)
        self.setMinimumWidth(620)
        self._selection: DetectKitSourceAdditionChoice | None = None

        root = Path(source_root).expanduser().resolve()
        class_names = ", ".join(inspection.discovered_labels) or "No classes detected"

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        intro = QLabel(
            "Review the selected source before adding it to this DetectKit project."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._path_value = QLabel(str(root))
        self._path_value.setWordWrap(True)
        form.addRow("Selected folder:", self._path_value)

        self._kind_value = QLabel(_describe_source_kind(inspection.source_kind))
        self._kind_value.setWordWrap(True)
        form.addRow("Detected format:", self._kind_value)

        self._images_value = QLabel(f"{inspection.images_count:,}")
        form.addRow("Images:", self._images_value)

        self._annotations_value = QLabel(f"{inspection.annotation_count:,}")
        form.addRow("Annotations:", self._annotations_value)

        self._class_names_value = QLabel(class_names)
        self._class_names_value.setWordWrap(True)
        form.addRow("Classes:", self._class_names_value)

        self._portable_action_value = QLabel(_describe_portable_action(inspection))
        self._portable_action_value.setWordWrap(True)
        self._portable_action_value.setStyleSheet(
            f"color: {HYDRA_DIALOG_MUTED_TEXT_COLOR};"
        )
        form.addRow("Import as portable:", self._portable_action_value)

        self._linked_action_value = QLabel(_describe_linked_action(inspection))
        self._linked_action_value.setWordWrap(True)
        self._linked_action_value.setStyleSheet(
            f"color: {HYDRA_DIALOG_MUTED_TEXT_COLOR};"
        )
        form.addRow("Keep at source:", self._linked_action_value)

        scan = resolve_source_level_or_block(
            root / "labels",
            _intended_level_for_kind(inspection.source_kind),
        )
        self._level_scan = scan
        self._level_value = QLabel(scan.resolved_level.label)
        form.addRow("Geometry level:", self._level_value)

        layout.addLayout(form)

        if not scan.is_homogeneous:
            warn = QLabel(
                f"Mixed geometry: {scan.reason}\nConflicting files: "
                + ", ".join(scan.conflict_files[:8])
                + (" …" if len(scan.conflict_files) > 8 else "")
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #c0392b;")
            layout.addWidget(warn)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_area.setWidget(container)

        self.add_content(self._scroll_area)

        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.hide()
        cancel_button = self._buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button is not None:
            cancel_button.setText("Cancel")

        self._portable_button = self._buttons.addButton(
            "Import as Portable",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._linked_button = self._buttons.addButton(
            "Keep at Source",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self._portable_button.clicked.connect(
            lambda: self._accept_choice(SOURCE_ADD_MODE_PORTABLE)
        )
        self._linked_button.clicked.connect(
            lambda: self._accept_choice(SOURCE_ADD_MODE_LINKED)
        )
        self._portable_button.setDefault(True)

    def _accept_choice(self, mode: str) -> None:
        if not self._level_scan.is_homogeneous:
            if self._level_scan.needs_confirmation:
                from PySide6.QtWidgets import QMessageBox

                answer = QMessageBox.question(
                    self,
                    "Mixed Geometry",
                    "This source mixes polygon files with four-point-only files. "
                    "Confirm the four-point files are genuine contours (treat the "
                    "whole source as polygon)? Choose No to cancel and fix the source.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                self._level_scan = resolve_source_level_or_block(
                    Path(self._path_value.text()) / "labels",
                    GeometryLevel.OBB,
                    confirm=True,
                )
            else:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Mixed Geometry",
                    "This source cannot be added until its geometry is homogeneous:\n\n"
                    + self._level_scan.reason,
                )
                return
        self._selection = DetectKitSourceAdditionChoice(
            mode=mode, level=self._level_scan.resolved_level.label
        )
        self.accept()

    def selected_choice(self) -> DetectKitSourceAdditionChoice | None:
        return self._selection


def confirm_detectkit_source_addition(
    parent,
    source_root: str | Path,
    inspection: DetectKitSourceInspection,
) -> DetectKitSourceAdditionChoice | None:
    """Show the pre-import review dialog and return the selected add mode."""
    dialog = DetectKitSourceValidationDialog(source_root, inspection, parent=parent)
    if dialog.exec() != dialog.DialogCode.Accepted:
        return None
    return dialog.selected_choice()
