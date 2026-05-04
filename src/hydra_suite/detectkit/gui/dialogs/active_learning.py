"""Modal dialog for running an active-learning round in DetectKit."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
    """Active-learning frame-selection dialog (Input / Acquisition / Run)."""

    def __init__(
        self,
        project: DetectKitProject,
        parent: QWidget | None = None,
    ):
        super().__init__(
            "Active Learning",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._run_handler: Callable[[], None] | None = None
        self._running = False
        self.resize(560, 540)
        self._build_content()
        self._sync_run_enabled()

    # ------------------------------------------------------------------
    def _build_content(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_acquisition_group())
        layout.addWidget(self._build_run_group())
        self.add_content(container)

    def _build_input_group(self) -> QGroupBox:
        self.input_group = QGroupBox("Input")
        form = QFormLayout(self.input_group)

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
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.input_path_edit)
        path_row.addWidget(self.browse_button)
        form.addRow("Path", _wrap(path_row))

        return self.input_group

    def _build_acquisition_group(self) -> QGroupBox:
        self.acquisition_group = QGroupBox("Acquisition")
        form = QFormLayout(self.acquisition_group)

        self.preset_combo = QComboBox()
        for name in PRESETS:
            if name != "tracker_default":
                self.preset_combo.addItem(name)
        form.addRow("Preset", self.preset_combo)

        self.expected_count_spin = QSpinBox()
        self.expected_count_spin.setRange(0, 1000)
        self.expected_count_spin.setValue(0)
        form.addRow("Expected count per frame (0 = unknown)", self.expected_count_spin)

        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 1000)
        self.budget_spin.setValue(50)
        form.addRow("Budget (top-K)", self.budget_spin)

        return self.acquisition_group

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run")
        v = QVBoxLayout(box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        self.status_label = QLabel("Idle.")
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_button)
        v.addLayout(btn_row)
        return box

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
        if self._running:
            self.run_button.setEnabled(False)
            self.status_label.setText("Active learning is running. Inputs are locked.")
            return
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

    def set_run_handler(self, handler: Callable[[], None]) -> None:
        """Main window wires this to construct + start the AL worker."""
        self._run_handler = handler

    def set_running(self, running: bool) -> None:
        """Lock editable controls while an AL round is active."""
        self._running = bool(running)
        self.input_group.setEnabled(not self._running)
        self.acquisition_group.setEnabled(not self._running)
        self._sync_run_enabled()


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w
