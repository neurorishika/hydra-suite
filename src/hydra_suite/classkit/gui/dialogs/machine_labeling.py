"""MachineLabelingDialog — unified launcher for machine-assisted labeling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.identity.classification.apriltag import AprilTagConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog
from hydra_suite.widgets.dialogs import BaseDialog

APRILTAG_FAMILIES = [
    "tag36h11",
    "tag25h9",
    "tag16h5",
    "tagCircle21h7",
    "tagCircle49h12",
    "tagCustom48h12",
    "tagStandard41h12",
    "tagStandard52h13",
]


class MachineLabelingDialog(BaseDialog):
    """Configure a machine-labeling run from a single dialog."""

    METHOD_MODEL = "model"
    METHOD_APRILTAG = "apriltag"
    MODEL_SOURCE_LOADED = "loaded"
    MODEL_SOURCE_HISTORY = "history"
    MODEL_SOURCE_OTHER_PROJECT = "other_project"
    MODEL_SOURCE_FILE = "file"

    def __init__(
        self,
        *,
        scope_options: list[tuple[str, list[int]]],
        predictions_available: bool,
        image_count: int,
        model_history_entries: Optional[list[dict]] = None,
        project_path=None,
        db_path=None,
        parent=None,
    ) -> None:
        super().__init__("Machine Labeling", parent=parent)
        self.setMinimumWidth(620)
        self._scope_options = scope_options
        self._predictions_available = bool(predictions_available)
        self._model_history_entries = list(model_history_entries or [])
        self._project_path = project_path
        self._db_path = db_path
        self._selected_model_entry: Optional[dict] = None
        self._selected_checkpoint_path: Optional[str] = None
        self._other_project_path: Optional[Path] = None
        self._other_project_db_path: Optional[Path] = None

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        intro = QLabel(
            "Choose a machine-labeling method, select the dataset scope, and apply labels as review candidates. "
            "Verified human labels are preserved by default."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#ffffff;")
        layout.addWidget(intro)

        launch_group = QGroupBox("Machine Labeling Run")
        launch_form = QFormLayout(launch_group)
        launch_form.setSpacing(10)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Trained model", self.METHOD_MODEL)
        self.method_combo.addItem("AprilTags", self.METHOD_APRILTAG)
        self.method_combo.currentIndexChanged.connect(self._sync_method_state)
        launch_form.addRow("Method", self.method_combo)

        self.scope_combo = QComboBox()
        for label, indices in self._scope_options:
            self.scope_combo.addItem(label, list(indices))
        launch_form.addRow("Scope", self.scope_combo)

        self.skip_verified_check = QCheckBox(
            "Skip existing verified labels when applying machine labels"
        )
        self.skip_verified_check.setChecked(True)
        launch_form.addRow("", self.skip_verified_check)

        self.method_summary = QLabel()
        self.method_summary.setWordWrap(True)
        self.method_summary.setStyleSheet(
            "padding: 8px; background:#252526; border-radius:6px; color:#ffffff;"
        )
        launch_form.addRow("", self.method_summary)

        layout.addWidget(launch_group)

        self.method_stack = QStackedWidget()
        self.method_stack.addWidget(self._build_model_page(image_count))
        self.method_stack.addWidget(self._build_apriltag_page())
        layout.addWidget(self.method_stack)

        self.add_content(content)

        if self._predictions_available:
            default_source = self.MODEL_SOURCE_LOADED
        elif self._model_history_entries:
            default_source = self.MODEL_SOURCE_HISTORY
        else:
            default_source = self.MODEL_SOURCE_FILE
        default_index = self.model_source_combo.findData(default_source)
        if default_index >= 0:
            self.model_source_combo.setCurrentIndex(default_index)
        self._sync_method_state()

    def _build_model_page(self, image_count: int) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        note = QLabel(
            "Use a trained model to write unverified review labels. You can reuse the current loaded predictions, "
            "pick a model from this project's history, browse another project's history, or choose a checkpoint file. "
            f"This can target a subset or all {image_count:,} images in the project."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#ffffff;")
        layout.addWidget(note)

        source_group = QGroupBox("Model Source")
        source_form = QFormLayout(source_group)
        source_form.setSpacing(8)

        self.model_source_combo = QComboBox()
        self.model_source_combo.addItem(
            "Current loaded predictions", self.MODEL_SOURCE_LOADED
        )
        self.model_source_combo.addItem(
            "Choose from this project's model history", self.MODEL_SOURCE_HISTORY
        )
        self.model_source_combo.addItem(
            "Choose from another project's model history",
            self.MODEL_SOURCE_OTHER_PROJECT,
        )
        self.model_source_combo.addItem(
            "Choose checkpoint file", self.MODEL_SOURCE_FILE
        )
        self.model_source_combo.currentIndexChanged.connect(self._sync_method_state)
        source_form.addRow("Source", self.model_source_combo)

        source_button_host = QWidget()
        source_button_row = QHBoxLayout(source_button_host)
        source_button_row.setContentsMargins(0, 0, 0, 0)
        source_button_row.setSpacing(8)

        self.model_source_pick_btn = QPushButton("Choose…")
        self.model_source_pick_btn.clicked.connect(self._pick_model_source)
        source_button_row.addWidget(self.model_source_pick_btn)

        self.model_source_clear_btn = QPushButton("Clear")
        self.model_source_clear_btn.clicked.connect(self._clear_model_source_selection)
        source_button_row.addWidget(self.model_source_clear_btn)
        source_button_row.addStretch(1)
        source_form.addRow("", source_button_host)

        self.model_source_detail = QLabel()
        self.model_source_detail.setWordWrap(True)
        self.model_source_detail.setStyleSheet(
            "padding: 6px 0; color:#ffffff; font-size:11px;"
        )
        source_form.addRow("", self.model_source_detail)

        layout.addWidget(source_group)

        self.model_warning = QLabel()
        self.model_warning.setWordWrap(True)
        self.model_warning.setStyleSheet(
            "padding: 8px; background:#2b1d1d; border-left:3px solid #d16969; border-radius:6px;"
        )
        layout.addWidget(self.model_warning)
        layout.addStretch(1)
        return page

    def _build_apriltag_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        warn = QLabel(
            "AprilTag labeling currently bootstraps an AprilTag scheme for the project before writing unverified machine labels. "
            "Only the selected scope is processed after the scheme reset."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "padding: 8px; background:#2d2a1f; border-left:3px solid #d7ba7d; border-radius:6px; color:#d7ba7d;"
        )
        layout.addWidget(warn)

        det_group = QGroupBox("AprilTag Detection Parameters")
        det_form = QFormLayout(det_group)
        det_form.setSpacing(8)

        self.family_combo = QComboBox()
        self.family_combo.addItems(APRILTAG_FAMILIES)
        self.family_combo.setCurrentText("tag36h11")
        det_form.addRow("Tag family", self.family_combo)

        self.max_tag_id_spin = QSpinBox()
        self.max_tag_id_spin.setRange(0, 999)
        self.max_tag_id_spin.setValue(9)
        det_form.addRow("Max tag ID", self.max_tag_id_spin)

        self.max_hamming_spin = QSpinBox()
        self.max_hamming_spin.setRange(0, 3)
        self.max_hamming_spin.setValue(1)
        det_form.addRow("Max hamming", self.max_hamming_spin)

        self.decimate_spin = QDoubleSpinBox()
        self.decimate_spin.setRange(1.0, 8.0)
        self.decimate_spin.setSingleStep(0.5)
        self.decimate_spin.setValue(2.0)
        det_form.addRow("Decimate", self.decimate_spin)

        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 5.0)
        self.blur_spin.setSingleStep(0.1)
        self.blur_spin.setValue(0.8)
        det_form.addRow("Blur", self.blur_spin)

        self.unsharp_amount_spin = QDoubleSpinBox()
        self.unsharp_amount_spin.setRange(0.0, 10.0)
        self.unsharp_amount_spin.setSingleStep(0.1)
        self.unsharp_amount_spin.setValue(1.0)
        det_form.addRow("Unsharp amount", self.unsharp_amount_spin)

        self.unsharp_sigma_spin = QDoubleSpinBox()
        self.unsharp_sigma_spin.setRange(0.1, 10.0)
        self.unsharp_sigma_spin.setSingleStep(0.1)
        self.unsharp_sigma_spin.setValue(1.0)
        det_form.addRow("Unsharp sigma", self.unsharp_sigma_spin)

        self.unsharp_kernel_spin = QSpinBox()
        self.unsharp_kernel_spin.setRange(1, 31)
        self.unsharp_kernel_spin.setSingleStep(2)
        self.unsharp_kernel_spin.setValue(5)
        det_form.addRow("Unsharp kernel size", self.unsharp_kernel_spin)

        layout.addWidget(det_group)

        label_group = QGroupBox("Labeling Parameters")
        label_form = QFormLayout(label_group)
        label_form.setSpacing(8)

        thresh_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(60)
        self.threshold_value = QLabel("0.60")
        self.threshold_slider.valueChanged.connect(
            lambda value: self.threshold_value.setText(f"{value / 100:.2f}")
        )
        thresh_row.addWidget(self.threshold_slider)
        thresh_row.addWidget(self.threshold_value)
        label_form.addRow("Confidence threshold", thresh_row)

        self.replace_scheme_check = QCheckBox(
            "Replace the project scheme with an AprilTag labeling scheme and clear existing labels first"
        )
        self.replace_scheme_check.setChecked(True)
        self.replace_scheme_check.setEnabled(False)
        label_form.addRow("", self.replace_scheme_check)

        layout.addWidget(label_group)
        return page

    @staticmethod
    def _model_entry_display_name(entry: Optional[dict]) -> str:
        if not entry:
            return ""
        name = str(entry.get("display_name") or "").strip()
        if name:
            return name
        paths = entry.get("artifact_paths") or []
        if paths:
            return Path(paths[0]).stem
        return str(entry.get("mode") or "model")

    def selected_model_source(self) -> str:
        return str(self.model_source_combo.currentData())

    def _clear_model_source_selection(self) -> None:
        source = self.selected_model_source()
        if source == self.MODEL_SOURCE_HISTORY:
            self._selected_model_entry = None
        elif source == self.MODEL_SOURCE_OTHER_PROJECT:
            self._selected_model_entry = None
            self._other_project_path = None
            self._other_project_db_path = None
        elif source == self.MODEL_SOURCE_FILE:
            self._selected_checkpoint_path = None
        self._sync_method_state()

    def _pick_model_source(self) -> None:
        source = self.selected_model_source()
        if source == self.MODEL_SOURCE_HISTORY:
            self._pick_model_history_entry()
        elif source == self.MODEL_SOURCE_OTHER_PROJECT:
            self._pick_other_project_history_entry()
        elif source == self.MODEL_SOURCE_FILE:
            self._pick_checkpoint_file()

    def _pick_model_history_entry(self) -> None:
        if not self._model_history_entries:
            QMessageBox.information(
                self,
                "No Project History",
                "No trained models are registered in this project's history yet. Choose a checkpoint file instead.",
            )
            return

        from .model_history import ModelHistoryDialog

        dlg = ModelHistoryDialog(
            self._model_history_entries,
            project_path=self._project_path,
            db_path=self._db_path,
            parent=self,
        )
        if dlg.exec() and dlg.selected_entry():
            self._selected_model_entry = dict(dlg.selected_entry())
            self._sync_method_state()

    def _pick_other_project_history_entry(self) -> None:
        from hydra_suite.classkit.core.store.db import ClassKitDB
        from hydra_suite.classkit.gui.project import (
            classkit_db_path,
            legacy_classkit_db_path,
            project_exists,
        )

        from .model_history import ModelHistoryDialog

        start_dir = ""
        if self._other_project_path is not None:
            start_dir = str(Path(self._other_project_path).parent)
        elif self._project_path:
            start_dir = str(Path(self._project_path).parent)

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Another ClassKit Project",
            start_dir,
        )
        if not chosen:
            return

        project_dir = Path(chosen).expanduser().resolve()
        if not project_exists(project_dir):
            QMessageBox.warning(
                self,
                "Not a ClassKit Project",
                f"The selected folder does not contain a ClassKit project bundle:\n{project_dir}",
            )
            return

        if self._project_path is not None:
            try:
                current_dir = Path(self._project_path).expanduser().resolve()
            except Exception:
                current_dir = None
            if current_dir is not None and current_dir == project_dir:
                QMessageBox.information(
                    self,
                    "Same Project Selected",
                    "The chosen folder is the current project. Use 'Choose from this project's model history' instead.",
                )
                return

        bundle_db = classkit_db_path(project_dir)
        legacy_db = legacy_classkit_db_path(project_dir)
        if bundle_db.exists():
            db_path = bundle_db
        elif legacy_db.exists():
            db_path = legacy_db
        else:
            QMessageBox.warning(
                self,
                "No Model Database",
                f"The selected project has no ClassKit database:\n{project_dir}",
            )
            return

        try:
            entries = ClassKitDB(db_path).list_model_caches()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Failed To Load History",
                f"Could not read model history from:\n{db_path}\n\n{exc}",
            )
            return

        if not entries:
            QMessageBox.information(
                self,
                "No Models In Project",
                f"The selected project has no trained models in its history:\n{project_dir}",
            )
            return

        dlg = ModelHistoryDialog(
            entries,
            project_path=project_dir,
            db_path=db_path,
            parent=self,
        )
        dlg.setWindowTitle(f"Models — {project_dir.name}")
        if dlg.exec() and dlg.selected_entry():
            self._selected_model_entry = dict(dlg.selected_entry())
            self._other_project_path = project_dir
            self._other_project_db_path = db_path
            self._sync_method_state()

    def _pick_checkpoint_file(self) -> None:
        start_dir = ""
        if self._project_path is not None:
            models_dir = Path(self._project_path) / "models"
            start_dir = str(models_dir if models_dir.exists() else self._project_path)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Classifier Checkpoint",
            start_dir,
            "PyTorch Checkpoint (*.pt *.pth)",
        )
        if not file_path:
            return
        self._selected_checkpoint_path = str(file_path)
        self._sync_method_state()

    def _sync_method_state(self) -> None:
        method = self.selected_method()
        is_model = method == self.METHOD_MODEL
        self.method_stack.setCurrentIndex(0 if is_model else 1)

        if is_model:
            source = self.selected_model_source()
            needs_picker = source in {
                self.MODEL_SOURCE_HISTORY,
                self.MODEL_SOURCE_OTHER_PROJECT,
                self.MODEL_SOURCE_FILE,
            }
            self.model_source_pick_btn.setVisible(needs_picker)
            self.model_source_clear_btn.setVisible(
                (
                    source == self.MODEL_SOURCE_HISTORY
                    and self._selected_model_entry is not None
                )
                or (
                    source == self.MODEL_SOURCE_OTHER_PROJECT
                    and self._selected_model_entry is not None
                )
                or (
                    source == self.MODEL_SOURCE_FILE
                    and bool(self._selected_checkpoint_path)
                )
            )

            ok_enabled = False
            if source == self.MODEL_SOURCE_LOADED:
                self.model_source_pick_btn.setText("Choose…")
                self.model_source_detail.setText(
                    "Use the predictions already loaded into ClassKit for this project."
                )
                if self._predictions_available:
                    self.model_warning.setText(
                        "Predictions are already loaded. Labels will be written as unverified machine labels, ready for review."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#1f2d22; border-left:3px solid #4ec943; border-radius:6px; color:#cfe9d1;"
                    )
                    ok_enabled = True
                else:
                    self.model_warning.setText(
                        "No predictions are currently loaded. Choose a model from project history or select a checkpoint file."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#2b1d1d; border-left:3px solid #d16969; border-radius:6px; color:#f0c0c0;"
                    )
            elif source == self.MODEL_SOURCE_HISTORY:
                self.model_source_pick_btn.setText("Choose Model…")
                if self._selected_model_entry is not None:
                    display_name = self._model_entry_display_name(
                        self._selected_model_entry
                    )
                    artifact_paths = (
                        self._selected_model_entry.get("artifact_paths") or []
                    )
                    path_name = Path(artifact_paths[0]).name if artifact_paths else ""
                    self.model_source_detail.setText(
                        f"Selected model history entry: <b>{display_name}</b>"
                        + (
                            f"<br><span style='color:#ffffff'>{path_name}</span>"
                            if path_name
                            else ""
                        )
                    )
                    self.model_warning.setText(
                        "ClassKit will load the selected project-history model, run inference on this project's images, and apply the resulting labels as review candidates."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#1f2d22; border-left:3px solid #4ec943; border-radius:6px; color:#cfe9d1;"
                    )
                    ok_enabled = True
                else:
                    self.model_source_detail.setText(
                        "Pick a previously trained model recorded in this project's history."
                    )
                    self.model_warning.setText(
                        "No project-history model is selected yet. Choose one to run machine labeling from that checkpoint."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#2d2a1f; border-left:3px solid #d7ba7d; border-radius:6px; color:#f0ddb0;"
                    )
            elif source == self.MODEL_SOURCE_OTHER_PROJECT:
                self.model_source_pick_btn.setText("Choose Project…")
                if (
                    self._selected_model_entry is not None
                    and self._other_project_path is not None
                ):
                    display_name = self._model_entry_display_name(
                        self._selected_model_entry
                    )
                    artifact_paths = (
                        self._selected_model_entry.get("artifact_paths") or []
                    )
                    path_name = Path(artifact_paths[0]).name if artifact_paths else ""
                    project_name = Path(self._other_project_path).name
                    self.model_source_detail.setText(
                        f"Selected model from <b>{project_name}</b>: <b>{display_name}</b>"
                        + (
                            f"<br><span style='color:#ffffff'>{path_name}</span>"
                            if path_name
                            else ""
                        )
                        + f"<br><span style='color:#9cdcfe'>{self._other_project_path}</span>"
                    )
                    self.model_warning.setText(
                        "ClassKit will load the selected model from the other project's history, run inference on this project's images, and apply the resulting labels as review candidates. The other project's files are read in place — nothing is copied."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#1f2d22; border-left:3px solid #4ec943; border-radius:6px; color:#cfe9d1;"
                    )
                    ok_enabled = True
                else:
                    self.model_source_detail.setText(
                        "Pick a different ClassKit project folder, then choose one of its trained models from history."
                    )
                    self.model_warning.setText(
                        "No external project model is selected yet. Choose a project folder to browse its model history."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#2d2a1f; border-left:3px solid #d7ba7d; border-radius:6px; color:#f0ddb0;"
                    )
            else:
                self.model_source_pick_btn.setText("Choose File…")
                if self._selected_checkpoint_path:
                    selected_path = Path(self._selected_checkpoint_path)
                    self.model_source_detail.setText(
                        f"Selected checkpoint file: <b>{selected_path.name}</b><br>"
                        f"<span style='color:#ffffff'>{selected_path}</span>"
                    )
                    self.model_warning.setText(
                        "ClassKit will load this checkpoint, run inference on the current project's images, and apply matching labels as unverified review candidates."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#1f2d22; border-left:3px solid #4ec943; border-radius:6px; color:#cfe9d1;"
                    )
                    ok_enabled = True
                else:
                    self.model_source_detail.setText(
                        "Choose a .pt or .pth checkpoint file, including a model trained in another project."
                    )
                    self.model_warning.setText(
                        "No checkpoint file is selected yet. Choose one to run machine labeling from that model."
                    )
                    self.model_warning.setStyleSheet(
                        "padding: 8px; background:#2d2a1f; border-left:3px solid #d7ba7d; border-radius:6px; color:#f0ddb0;"
                    )

            self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
                ok_enabled
            )
            self.method_summary.setText(
                "<b>Trained model</b><br>Applies the selected model's top class to the selected scope as unverified review labels."
            )
        else:
            self.model_source_pick_btn.setVisible(False)
            self.model_source_clear_btn.setVisible(False)
            self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            self.method_summary.setText(
                "<b>AprilTags</b><br>Runs AprilTag detection on the selected scope and writes the resulting labels as unverified machine labels."
            )

    def selected_method(self) -> str:
        return str(self.method_combo.currentData())

    def selected_scope(self) -> tuple[str, list[int]]:
        return self.scope_combo.currentText(), list(
            self.scope_combo.currentData() or []
        )

    def get_apriltag_config(self) -> AprilTagConfig:
        n = self.unsharp_kernel_spin.value()
        return AprilTagConfig(
            family=self.family_combo.currentText(),
            max_hamming=self.max_hamming_spin.value(),
            decimate=self.decimate_spin.value(),
            blur=self.blur_spin.value(),
            unsharp_amount=self.unsharp_amount_spin.value(),
            unsharp_sigma=self.unsharp_sigma_spin.value(),
            unsharp_kernel_size=(n, n),
            max_tag_id=self.max_tag_id_spin.value(),
        )

    def get_settings(self) -> dict:
        scope_label, scope_indices = self.selected_scope()
        payload = {
            "method": self.selected_method(),
            "scope_label": scope_label,
            "scope_indices": scope_indices,
            "skip_verified": self.skip_verified_check.isChecked(),
        }
        if payload["method"] == self.METHOD_MODEL:
            payload.update(
                {
                    "model_source": self.selected_model_source(),
                    "model_entry": self._selected_model_entry,
                    "checkpoint_path": self._selected_checkpoint_path,
                    "other_project_path": (
                        str(self._other_project_path)
                        if self._other_project_path is not None
                        else None
                    ),
                }
            )
        if payload["method"] == self.METHOD_APRILTAG:
            payload.update(
                {
                    "apriltag_config": self.get_apriltag_config(),
                    "apriltag_threshold": self.threshold_slider.value() / 100.0,
                    "replace_scheme": self.replace_scheme_check.isChecked(),
                }
            )
        return payload
