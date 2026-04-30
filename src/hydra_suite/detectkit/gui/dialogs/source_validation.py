"""DetectKit source-validation dialog shown before adding a dataset source."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import HYDRA_DIALOG_MUTED_TEXT_COLOR, BaseDialog

from ..source_import import DetectKitSourceInspection


def _describe_source_kind(source_kind: str) -> str:
    descriptions = {
        "detectkit": "DetectKit canonical source",
        "yolo_detect": "YOLO detect dataset",
        "yolo_obb": "YOLO OBB dataset",
        "coco": "COCO annotations dataset",
    }
    return descriptions.get(source_kind, source_kind)


def _describe_import_action(inspection: DetectKitSourceInspection) -> str:
    if inspection.requires_import:
        return (
            "This source will be copied into the DetectKit project and normalized "
            "to DetectKit's canonical images/, labels/, and classes.txt layout "
            "before use."
        )
    return "This source already matches DetectKit's canonical layout and will be added as-is."


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
        self.setMinimumWidth(620)

        root = Path(source_root).expanduser().resolve()
        action_text = _describe_import_action(inspection)
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

        self._action_value = QLabel(action_text)
        self._action_value.setWordWrap(True)
        self._action_value.setStyleSheet(f"color: {HYDRA_DIALOG_MUTED_TEXT_COLOR};")
        form.addRow("Action:", self._action_value)
        layout.addLayout(form)

        self.add_content(container)

        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText(
                "Import and Add" if inspection.requires_import else "Add Source"
            )


def confirm_detectkit_source_addition(
    parent,
    source_root: str | Path,
    inspection: DetectKitSourceInspection,
) -> bool:
    """Show the pre-import review dialog and return whether the user accepted it."""
    dialog = DetectKitSourceValidationDialog(source_root, inspection, parent=parent)
    return dialog.exec() == QDialog.DialogCode.Accepted
