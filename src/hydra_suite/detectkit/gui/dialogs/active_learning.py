"""Modal dialog for running an active-learning round in DetectKit."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.data.al.acquisition import PRESETS
from hydra_suite.detectkit.gui.models import DetectKitProject
from hydra_suite.widgets.dialogs import BaseDialog


class ActiveLearningDialog(BaseDialog):
    """Three-section AL dialog: Input, Acquisition, Execution."""

    def __init__(
        self,
        project: DetectKitProject,
        parent: QWidget | None = None,
    ):
        super().__init__(
            title="Active Learning",
            parent=parent,
            buttons=QDialogButtonBox.NoButton,
        )
        self._project = project
        self._run_handler: Callable[[], None] | None = None
        self._cancel_handler: Callable[[], None] | None = None
        self._build_ui()
        self._sync_run_enabled()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.add_content(self._build_input_section())
        self.add_content(self._build_acquisition_section())
        self.add_content(self._build_execution_section())

    def _build_input_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Input</b>"))

        form = QFormLayout()

        self.rb_video = QRadioButton("Video")
        self.rb_folder = QRadioButton("Image folder")
        self.rb_project = QRadioButton("Existing project source (unlabeled)")
        self.rb_video.setChecked(True)
        for rb in (self.rb_video, self.rb_folder, self.rb_project):
            rb.toggled.connect(self._sync_run_enabled)
        rb_row = QHBoxLayout()
        rb_row.addWidget(self.rb_video)
        rb_row.addWidget(self.rb_folder)
        rb_row.addWidget(self.rb_project)
        form.addRow("Source kind", _wrap(rb_row))

        self.input_path_edit = QLineEdit()
        self.input_path_edit.textChanged.connect(self._sync_run_enabled)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.input_path_edit)
        path_row.addWidget(browse_btn)
        form.addRow("Path", _wrap(path_row))

        layout.addLayout(form)
        return section

    def _build_acquisition_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Acquisition</b>"))

        self.preset_combo = QComboBox()
        for name in PRESETS:
            if name != "tracker_default":
                self.preset_combo.addItem(name)

        self.expected_count_spin = QSpinBox()
        self.expected_count_spin.setRange(0, 1000)
        self.expected_count_spin.setValue(0)

        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 1000)
        self.budget_spin.setValue(50)

        form = QFormLayout()
        form.addRow("Preset", self.preset_combo)
        form.addRow("Expected count per frame (0 = unknown)", self.expected_count_spin)
        form.addRow("Budget (top-K)", self.budget_spin)

        layout.addLayout(form)
        return section

    def _build_execution_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("<b>Execution</b>"))

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Idle.")

        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self._on_run)
        self.cancel_button.clicked.connect(self._on_cancel)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.cancel_button)

        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_row)
        return section

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        if self.rb_video.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select video",
                "",
                "Video files (*.mp4 *.mov *.avi)",
            )
        elif self.rb_folder.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select image folder")
        else:
            path = ""
        if path:
            self.input_path_edit.setText(path)

    def _sync_run_enabled(self, *_):
        path_ok = self.rb_project.isChecked() or bool(
            self.input_path_edit.text().strip()
        )
        model_ok = bool(self._project.active_model_path)
        self.run_button.setEnabled(path_ok and model_ok)
        if not model_ok:
            self.status_label.setText(
                "Set an active model in DetectKit before running AL."
            )
        elif not path_ok:
            self.status_label.setText("Pick an input source.")
        else:
            self.status_label.setText("Ready.")

    def _on_run(self) -> None:
        if self._run_handler is not None:
            self._run_handler()

    def _on_cancel(self) -> None:
        if self._cancel_handler is not None:
            self._cancel_handler()

    def set_run_handler(self, handler: Callable[[], None]) -> None:
        """Main window wires this to construct + start the AL worker."""
        self._run_handler = handler

    def set_cancel_handler(self, handler: Callable[[], None]) -> None:
        self._cancel_handler = handler


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
