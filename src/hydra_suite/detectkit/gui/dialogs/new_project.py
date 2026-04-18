"""DetectKit new-project dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.gui.models import DEFAULT_CLASS_NAME, normalize_class_names
from hydra_suite.detectkit.gui.project import default_project_parent_dir
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.dialogs import BaseDialog

_BTN_NEUTRAL = (
    "QPushButton { background-color:#3e3e42; color:#e0e0e0; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#555558; }"
)


class NewProjectDialog(BaseDialog):
    """Collect DetectKit project name, location, and initial class names."""

    def __init__(self, parent=None) -> None:
        super().__init__("Create New Project", parent=parent)
        self.setMinimumWidth(580)

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        header = QLabel("<h2 style='margin:0;'>Create New Project</h2>")
        intro = QLabel(
            "Choose where the project lives and define the object classes DetectKit should start with."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #ffffff;")
        layout.addWidget(header)
        layout.addWidget(intro)

        details_group = QGroupBox("Project Details")
        details_layout = QVBoxLayout(details_group)
        details_layout.setSpacing(10)
        details_form = QFormLayout()
        details_form.setSpacing(10)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my_detection_project")
        self.name_edit.textChanged.connect(self._update_project_preview)
        self.name_edit.textChanged.connect(self._validate)
        details_form.addRow("Project Name", self.name_edit)

        location_row = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.location_edit.setText(str(default_project_parent_dir()))
        self.location_edit.setPlaceholderText("Choose project location...")
        self.location_edit.textChanged.connect(self._update_project_preview)
        self.location_edit.textChanged.connect(self._validate)
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(_BTN_NEUTRAL)
        browse_btn.clicked.connect(self._browse_location)
        location_row.addWidget(self.location_edit, 1)
        location_row.addWidget(browse_btn)
        details_form.addRow("Location", location_row)
        details_layout.addLayout(details_form)

        preview_label = QLabel("Project Folder")
        preview_label.setStyleSheet("color: #cfcfcf;")
        details_layout.addWidget(preview_label)

        self.project_preview = QLabel()
        self.project_preview.setWordWrap(True)
        self.project_preview.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px; "
            "border-left: 3px solid #0e639c;"
        )
        details_layout.addWidget(self.project_preview)
        layout.addWidget(details_group)

        setup_group = QGroupBox("Detection Setup")
        setup_layout = QVBoxLayout(setup_group)
        self.class_names_edit = QPlainTextEdit()
        self.class_names_edit.setPlainText(DEFAULT_CLASS_NAME)
        self.class_names_edit.setPlaceholderText("ant\nbee")
        self.class_names_edit.setFixedHeight(84)
        self.class_names_edit.textChanged.connect(self._validate)
        classes_help = QLabel(
            "Add one class per line. The first class becomes the default active label for the project."
        )
        classes_help.setWordWrap(True)
        classes_help.setStyleSheet("color: #cfcfcf;")
        setup_layout.addWidget(classes_help)
        setup_layout.addWidget(self.class_names_edit)
        layout.addWidget(setup_group)

        help_label = QLabel(
            "DetectKit will create a project folder containing dataset sources, training settings, and evaluation history."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px; "
            "border-left: 3px solid #0e639c; color: #ffffff; line-height: 1.5;"
        )
        layout.addWidget(help_label)

        self.add_content(content)

        create_button = self._buttons.button(QDialogButtonBox.Ok)
        create_button.setText("Create Project")
        self._update_project_preview()
        self._validate()

    def _browse_location(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Location",
            self.location_edit.text() or str(default_project_parent_dir()),
        )
        if folder:
            self.location_edit.setText(folder)

    def _update_project_preview(self) -> None:
        project_path = self.get_project_path()
        if project_path is None:
            self.project_preview.setText(
                "Project folder: select a location and enter a project name."
            )
            return
        self.project_preview.setText(f"Project folder: {project_path}")

    def _validate(self) -> None:
        create_button = self._buttons.button(QDialogButtonBox.Ok)
        create_button.setEnabled(
            bool(self.name_edit.text().strip())
            and bool(self.location_edit.text().strip())
            and bool(self.class_names())
        )

    def class_names(self) -> list[str]:
        """Return normalized class names from the editor."""
        return normalize_class_names(self.class_names_edit.toPlainText().splitlines())

    def get_project_path(self) -> Path | None:
        """Return the full DetectKit project directory, or None if incomplete."""
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()
        if not name or not location:
            return None
        return Path(location).expanduser() / name

    def get_project_info(self) -> dict[str, object]:
        """Return the project details collected by the dialog."""
        project_path = self.get_project_path()
        class_names = self.class_names()
        return {
            "name": self.name_edit.text().strip(),
            "path": str(project_path) if project_path is not None else "",
            "class_names": class_names,
            "class_name": class_names[0],
        }
