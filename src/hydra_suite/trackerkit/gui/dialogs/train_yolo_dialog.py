"""Role-aware MAT Training Center dialog for multi-model YOLO workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.gui.widgets.loss_plot_widget import LossPlotWidget
from hydra_suite.training import (
    AugmentationProfile,
    PublishPolicy,
    SourceDataset,
    SplitConfig,
    TrainingHyperParams,
    TrainingOrchestrator,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.validation import format_validation_report
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.utils.gpu_utils import get_device_info
from hydra_suite.widgets.dialogs import BaseDialog
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class RoleTrainingWorker(BaseWorker):
    """Run selected role trainings sequentially in a background thread."""

    log_signal = Signal(str)
    role_started = Signal(str)
    role_finished = Signal(str, bool, str)
    progress_signal = Signal(str, int, int)
    done_signal = Signal(list)

    def __init__(self, orchestrator, role_entries) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.role_entries = role_entries
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _should_cancel(self) -> bool:
        return bool(self._cancel)

    def execute(self):
        results = []
        parent_run = ""
        for entry in self.role_entries:
            if self._cancel:
                break
            role = entry["role"]
            spec = entry["spec"]
            publish_meta = entry["publish_meta"]
            self.role_started.emit(role.value)

            def _log(msg: str, _role=role):
                self.log_signal.emit(f"[{_role.value}] {msg}")

            def _prog(cur: int, total: int, _role=role):
                self.progress_signal.emit(_role.value, int(cur), int(total))

            try:
                result = self.orchestrator.run_role_training(
                    spec,
                    parent_run_id=parent_run,
                    publish_metadata=publish_meta,
                    log_cb=_log,
                    progress_cb=_prog,
                    should_cancel=self._should_cancel,
                )
            except Exception as exc:
                result = {
                    "run_id": "",
                    "success": False,
                    "error": str(exc),
                    "published_registry_key": "",
                    "published_model_path": "",
                }

            result["role"] = role.value
            results.append(result)
            ok = bool(result.get("success", False))
            msg = (
                f"run_id={result.get('run_id', '')}"
                if ok
                else (
                    result.get("error") or f"exit={result.get('exit_code', 'unknown')}"
                )
            )
            self.role_finished.emit(role.value, ok, msg)
            if result.get("run_id"):
                parent_run = str(result["run_id"])

        self.done_signal.emit(results)


class TrainYoloDialog(BaseDialog):
    """MAT role-aware training center replacing legacy OBB-only trainer."""

    def __init__(self, parent=None, class_name="object", conda_envs=None) -> None:
        super().__init__(
            title="Training Center (YOLO Multi-Role)",
            parent=parent,
            buttons=QDialogButtonBox.NoButton,
            apply_dark_style=True,
        )
        self.resize(1100, 820)

        self.class_name = str(class_name or "object")
        self.conda_envs = conda_envs or []
        self.worker = None
        self._last_training_results: list[dict] = []

        from hydra_suite.paths import get_training_workspace_dir

        self.workspace_default = get_training_workspace_dir("YOLO")
        self.orchestrator = TrainingOrchestrator(self.workspace_default)

        self.role_dataset_dirs: dict[str, str] = {}

        self._build_ui()

    def _build_ui(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(self._build_sources_group())
        layout.addWidget(self._build_roles_group())
        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_augmentation_group())
        layout.addWidget(self._build_run_group())

        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.add_content(container)

    def _build_sources_group(self):
        gb = QGroupBox("Step 1: Sources + Validation")
        v = QVBoxLayout(gb)

        self.table_sources = QTableWidget(0, 3)
        self.table_sources.setHorizontalHeaderLabels(["Type", "Path", "Status"])
        self.table_sources.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table_sources)

        row = QHBoxLayout()
        self.btn_add_obb = QPushButton("Add OBB Source Datasets…")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_clear = QPushButton("Clear All")
        self.btn_validate = QPushButton("Validate Sources")
        row.addWidget(self.btn_add_obb)
        row.addWidget(self.btn_remove)
        row.addWidget(self.btn_clear)
        row.addWidget(self.btn_validate)
        v.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_save_list = QPushButton("Save Dataset List…")
        self.btn_load_list = QPushButton("Load Dataset List…")
        self.btn_analyze = QPushButton("Analyze && Preview")
        self.btn_save_list.setToolTip(
            "Save the current source dataset paths to a JSON file for quick reuse."
        )
        self.btn_load_list.setToolTip(
            "Load a previously saved dataset list (appends to current sources)."
        )
        self.btn_analyze.setToolTip(
            "Analyze object sizes, preview crops at current settings, "
            "and check for training/inference compatibility issues."
        )
        row2.addWidget(self.btn_save_list)
        row2.addWidget(self.btn_load_list)
        row2.addWidget(self.btn_analyze)
        row2.addStretch()
        v.addLayout(row2)

        self.validation_view = QTextEdit()
        self.validation_view.setReadOnly(True)
        self.validation_view.setPlaceholderText("Validation report appears here.")
        v.addWidget(self.validation_view)

        # Crop preview area — shows sample crops at current settings.
        self.preview_container = QGroupBox("Crop Preview (seq_crop_obb settings)")
        self.preview_container.setVisible(False)
        preview_layout = QVBoxLayout(self.preview_container)
        self.preview_grid = QHBoxLayout()
        preview_layout.addLayout(self.preview_grid)
        self.preview_info = QLabel("")
        self.preview_info.setWordWrap(True)
        preview_layout.addWidget(self.preview_info)
        v.addWidget(self.preview_container)

        self.btn_add_obb.clicked.connect(self._add_obb_sources)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_sources)
        self.btn_validate.clicked.connect(self._validate_sources)
        self.btn_save_list.clicked.connect(self._save_dataset_list)
        self.btn_load_list.clicked.connect(self._load_dataset_list)
        self.btn_analyze.clicked.connect(self._analyze_and_preview)

        return gb

    def _build_roles_group(self):
        gb = QGroupBox("Step 2: Role Selection")
        g = QGridLayout(gb)

        self.chk_role_obb_direct = QCheckBox("obb_direct")
        self.chk_role_seq_detect = QCheckBox("seq_detect")
        self.chk_role_seq_crop_obb = QCheckBox("seq_crop_obb")

        self.chk_role_obb_direct.setChecked(True)
        self.chk_role_seq_detect.setChecked(True)
        self.chk_role_seq_crop_obb.setChecked(True)

        g.addWidget(self.chk_role_obb_direct, 0, 0)
        g.addWidget(self.chk_role_seq_detect, 0, 1)
        g.addWidget(self.chk_role_seq_crop_obb, 0, 2)

        info = QLabel(
            "<b>Classification training has moved to ClassKit.</b><br>"
            "Use <code>classkit-labeler</code> to label and train classifiers."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 8px; background: #1a2a1a; border-left: 3px solid #4caf50;"
            " color: #ffffff; border-radius: 3px;"
        )
        g.addWidget(info, 1, 0, 1, 3)

        note = QLabel("Sequential datasets are auto-derived from OBB sources.")
        note.setWordWrap(True)
        g.addWidget(note, 2, 0, 1, 3)
        return gb

    def _build_config_group(self):
        gb = QGroupBox("Step 3: Config")
        v = QVBoxLayout(gb)

        form = QFormLayout()
        self.line_workspace = QLineEdit(str(self.workspace_default))
        self.btn_workspace = QPushButton("Browse…")
        h_ws = QHBoxLayout()
        h_ws.addWidget(self.line_workspace, 1)
        h_ws.addWidget(self.btn_workspace)
        form.addRow("Workspace Root", h_ws)

        self.line_class = QLineEdit(self.class_name)
        form.addRow("Class Name", self.line_class)

        self.spin_train = QDoubleSpinBox()
        self.spin_train.setRange(0.05, 0.95)
        self.spin_train.setSingleStep(0.05)
        self.spin_train.setValue(0.8)
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.05, 0.95)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.2)
        h_split = QHBoxLayout()
        h_split.addWidget(QLabel("train"))
        h_split.addWidget(self.spin_train)
        h_split.addWidget(QLabel("val"))
        h_split.addWidget(self.spin_val)
        form.addRow("Split", h_split)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        form.addRow("Seed", self.spin_seed)

        self.chk_dedup = QCheckBox("Deduplicate source images by content hash")
        self.chk_dedup.setChecked(True)
        form.addRow("", self.chk_dedup)

        self.spin_crop_pad = QDoubleSpinBox()
        self.spin_crop_pad.setRange(0.0, 1.0)
        self.spin_crop_pad.setSingleStep(0.01)
        self.spin_crop_pad.setValue(0.15)
        self.spin_crop_min_px = QSpinBox()
        self.spin_crop_min_px.setRange(8, 2048)
        self.spin_crop_min_px.setValue(64)
        self.chk_crop_square = QCheckBox("Enforce square crop")
        self.chk_crop_square.setChecked(True)
        h_crop = QHBoxLayout()
        h_crop.addWidget(QLabel("pad"))
        h_crop.addWidget(self.spin_crop_pad)
        h_crop.addWidget(QLabel("min px"))
        h_crop.addWidget(self.spin_crop_min_px)
        h_crop.addWidget(self.chk_crop_square)
        form.addRow("Sequential crop derivation", h_crop)

        self.combo_device = QComboBox()
        self.combo_device.setEditable(True)
        self.combo_device.setToolTip(
            "Select compute device. For multi-GPU, type a comma-separated list "
            "like '0,1' in the editable combo box."
        )
        self.combo_device.addItems(self._build_device_options())
        form.addRow("Device", self.combo_device)

        v.addLayout(form)

        gb_train = QGroupBox("Training Hyperparameters")
        tr = QGridLayout(gb_train)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.chk_auto_batch = QCheckBox("Auto")
        self.chk_auto_batch.setToolTip(
            "Let Ultralytics auto-detect optimal batch size (batch=-1). "
            "Overrides manual batch setting."
        )
        self.chk_auto_batch.toggled.connect(
            lambda checked: self.spin_batch.setEnabled(not checked)
        )
        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(1e-5, 1.0)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 500)
        self.spin_patience.setValue(30)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        self.chk_cache = QCheckBox("Cache")
        tr.addWidget(QLabel("epochs"), 0, 0)
        tr.addWidget(self.spin_epochs, 0, 1)
        tr.addWidget(QLabel("batch"), 0, 2)
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(self.spin_batch)
        batch_layout.addWidget(self.chk_auto_batch)
        batch_widget = QWidget()
        batch_widget.setLayout(batch_layout)
        tr.addWidget(batch_widget, 0, 3)
        tr.addWidget(QLabel("lr0"), 0, 4)
        tr.addWidget(self.spin_lr0, 0, 5)
        tr.addWidget(QLabel("patience"), 1, 0)
        tr.addWidget(self.spin_patience, 1, 1)
        tr.addWidget(QLabel("workers"), 1, 2)
        tr.addWidget(self.spin_workers, 1, 3)
        tr.addWidget(self.chk_cache, 1, 4, 1, 2)

        # Per-role imgsz — critical for training/inference compatibility.
        # obb_direct and seq_detect train on full frames (640px default).
        # seq_crop_obb trains on small crops and MUST match the inference
        # stage-2 imgsz (default 160) to avoid distribution shift.
        self.spin_imgsz_obb_direct = QSpinBox()
        self.spin_imgsz_obb_direct.setRange(64, 2048)
        self.spin_imgsz_obb_direct.setValue(640)
        self.spin_imgsz_seq_detect = QSpinBox()
        self.spin_imgsz_seq_detect.setRange(64, 2048)
        self.spin_imgsz_seq_detect.setValue(640)
        self.spin_imgsz_seq_crop_obb = QSpinBox()
        self.spin_imgsz_seq_crop_obb.setRange(64, 2048)
        self.spin_imgsz_seq_crop_obb.setValue(160)
        self.spin_imgsz_seq_crop_obb.setToolTip(
            "Must match YOLO_SEQ_STAGE2_IMGSZ used during inference (default 160).\n"
            "Training at a different size causes distribution shift and poor detection."
        )
        tr.addWidget(QLabel("imgsz (obb_direct)"), 2, 0)
        tr.addWidget(self.spin_imgsz_obb_direct, 2, 1)
        tr.addWidget(QLabel("imgsz (seq_detect)"), 2, 2)
        tr.addWidget(self.spin_imgsz_seq_detect, 2, 3)
        tr.addWidget(QLabel("imgsz (seq_crop_obb)"), 2, 4)
        tr.addWidget(self.spin_imgsz_seq_crop_obb, 2, 5)

        v.addWidget(gb_train)

        gb_models = QGroupBox("Base Models")
        fm = QFormLayout(gb_models)
        self.combo_model_obb_direct = QComboBox()
        self.combo_model_obb_direct.setEditable(True)
        self.combo_model_obb_direct.addItems(
            [
                "yolo26n-obb.pt",
                "yolo26s-obb.pt",
                "yolo26m-obb.pt",
                "yolo26l-obb.pt",
                "yolo26x-obb.pt",
            ]
        )
        self.combo_model_obb_direct.setCurrentText("yolo26s-obb.pt")

        self.combo_model_seq_detect = QComboBox()
        self.combo_model_seq_detect.setEditable(True)
        self.combo_model_seq_detect.addItems(
            ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
        )
        self.combo_model_seq_detect.setCurrentText("yolo26s.pt")

        self.combo_model_seq_crop_obb = QComboBox()
        self.combo_model_seq_crop_obb.setEditable(True)
        self.combo_model_seq_crop_obb.addItems(
            ["yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt"]
        )
        self.combo_model_seq_crop_obb.setCurrentText("yolo26s-obb.pt")

        fm.addRow("obb_direct", self.combo_model_obb_direct)
        fm.addRow("seq_detect", self.combo_model_seq_detect)
        fm.addRow("seq_crop_obb", self.combo_model_seq_crop_obb)
        v.addWidget(gb_models)

        gb_publish = QGroupBox("Publish Metadata")
        fp = QFormLayout(gb_publish)
        self.line_species = QLineEdit(self.class_name)
        self.line_model_tag = QLineEdit("train")
        self.chk_auto_import = QCheckBox(
            "Auto-import successful models into repository"
        )
        self.chk_auto_import.setChecked(True)
        self.chk_auto_select = QCheckBox("Auto-select newly imported models in main UI")
        self.chk_auto_select.setChecked(False)
        fp.addRow("species", self.line_species)
        fp.addRow("model tag", self.line_model_tag)
        fp.addRow("", self.chk_auto_import)
        fp.addRow("", self.chk_auto_select)
        v.addWidget(gb_publish)

        self.btn_workspace.clicked.connect(self._choose_workspace)

        return gb

    def _build_augmentation_group(self):
        self.aug_group = QGroupBox("Augmentation")
        self.aug_group.setCheckable(True)
        self.aug_group.setChecked(True)
        v = QVBoxLayout(self.aug_group)

        note = QLabel(
            "These are passed directly to Ultralytics. Defaults match"
            " Ultralytics v8 defaults. Set fliplr=0 for asymmetric animals."
        )
        note.setWordWrap(True)
        v.addWidget(note)

        form = QFormLayout()

        def _spin(default: float, maximum: float = 1.0) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(0.0, maximum)
            sb.setDecimals(3)
            sb.setSingleStep(0.05)
            sb.setValue(default)
            return sb

        self.aug_fliplr = _spin(0.5)
        form.addRow("fliplr", self.aug_fliplr)

        self.aug_flipud = _spin(0.0)
        form.addRow("flipud", self.aug_flipud)

        self.aug_degrees = _spin(0.0, 360.0)
        form.addRow("degrees", self.aug_degrees)

        self.aug_mosaic = _spin(1.0)
        form.addRow("mosaic", self.aug_mosaic)

        self.aug_mixup = _spin(0.0)
        form.addRow("mixup", self.aug_mixup)

        self.aug_hsv_h = _spin(0.015)
        form.addRow("hsv_h", self.aug_hsv_h)

        self.aug_hsv_s = _spin(0.7)
        form.addRow("hsv_s", self.aug_hsv_s)

        self.aug_hsv_v = _spin(0.4)
        form.addRow("hsv_v", self.aug_hsv_v)

        v.addLayout(form)
        return self.aug_group

    def _build_run_group(self):
        gb = QGroupBox("Step 4: Build + Train + Monitor")
        v = QVBoxLayout(gb)

        row = QHBoxLayout()
        self.btn_build = QPushButton("Build Role Datasets")
        self.btn_train = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_resume = QPushButton("Resume Last Run")
        self.btn_resume.setEnabled(False)
        self.btn_resume.setToolTip(
            "Resume training from the last.pt checkpoint of the most recent run."
        )
        self.btn_detach = QPushButton("Start Detached")
        self.btn_detach.setToolTip(
            "Launch training as a background process. You can close this dialog "
            "and continue tracking. Check Run History for results."
        )
        self.btn_history = QPushButton("Run History...")
        self.btn_quick_test = QPushButton("Quick Test...")
        self.btn_quick_test.setEnabled(False)
        self.btn_quick_test.setToolTip(
            "Run the last trained model on sample images to visually verify detections."
        )
        row.addWidget(self.btn_build)
        row.addWidget(self.btn_train)
        row.addWidget(self.btn_detach)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_resume)
        row.addWidget(self.btn_history)
        row.addWidget(self.btn_quick_test)
        v.addLayout(row)

        cfg_row = QHBoxLayout()
        self.btn_save_config = QPushButton("Save Config...")
        self.btn_load_config = QPushButton("Load Config...")
        cfg_row.addWidget(self.btn_save_config)
        cfg_row.addWidget(self.btn_load_config)
        cfg_row.addStretch()
        v.addLayout(cfg_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v.addWidget(self.progress)

        self.loss_plot = LossPlotWidget()
        self.loss_plot.setMinimumHeight(180)
        v.addWidget(self.loss_plot)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        v.addWidget(self.log_view)

        self.btn_build.clicked.connect(self._build_role_datasets)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_detach.clicked.connect(self._start_detached)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_resume.clicked.connect(self._resume_training)
        self.btn_history.clicked.connect(self._show_history)
        self.btn_quick_test.clicked.connect(self._quick_test)
        self.btn_save_config.clicked.connect(self._save_training_config)
        self.btn_load_config.clicked.connect(self._load_training_config)

        return gb

    # ------------------------------------------------------------------
    # Save / Load training config
    # ------------------------------------------------------------------

    def _save_training_config(self):
        """Export all dialog settings to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Config",
            str(getattr(self, "workspace_default", Path.home())),
            "JSON Files (*.json)",
        )
        if not path:
            return

        # Gather roles
        roles = []
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            chk = getattr(self, f"chk_role_{role_key}", None)
            if chk and chk.isChecked():
                roles.append(role_key)

        # Gather sources from table
        sources = []
        for row in range(self.table_sources.rowCount()):
            st = self.table_sources.item(row, 0).data(Qt.UserRole)
            p = self.table_sources.item(row, 1).text().strip()
            sources.append({"source_type": st, "path": p})

        config = {
            "version": 1,
            "class_name": self.line_class.text().strip(),
            "roles": roles,
            "sources": sources,
            "hyperparams": {
                "epochs": self.spin_epochs.value(),
                "batch": self.spin_batch.value(),
                "lr0": self.spin_lr0.value(),
                "patience": self.spin_patience.value(),
                "workers": self.spin_workers.value(),
                "cache": self.chk_cache.isChecked(),
            },
            "imgsz": {
                "obb_direct": self.spin_imgsz_obb_direct.value(),
                "seq_detect": self.spin_imgsz_seq_detect.value(),
                "seq_crop_obb": self.spin_imgsz_seq_crop_obb.value(),
            },
            "split": {
                "train": self.spin_train.value(),
                "val": self.spin_val.value(),
            },
            "seed": self.spin_seed.value(),
            "dedup": self.chk_dedup.isChecked(),
            "crop_derivation": {
                "pad_ratio": self.spin_crop_pad.value(),
                "min_crop_size_px": self.spin_crop_min_px.value(),
                "enforce_square": self.chk_crop_square.isChecked(),
            },
            "base_models": {
                "obb_direct": self.combo_model_obb_direct.currentText(),
                "seq_detect": self.combo_model_seq_detect.currentText(),
                "seq_crop_obb": self.combo_model_seq_crop_obb.currentText(),
            },
            "augmentation": {
                "enabled": self.aug_group.isChecked(),
                "fliplr": self.spin_aug_fliplr.value(),
                "flipud": self.spin_aug_flipud.value(),
                "degrees": self.spin_aug_degrees.value(),
                "mosaic": self.spin_aug_mosaic.value(),
                "mixup": self.spin_aug_mixup.value(),
                "hsv_h": self.spin_aug_hsv_h.value(),
                "hsv_s": self.spin_aug_hsv_s.value(),
                "hsv_v": self.spin_aug_hsv_v.value(),
            },
            "device": self.combo_device.currentText(),
            "publish": {
                "species": self.line_species.text().strip(),
                "model_tag": self.line_model_tag.text().strip(),
                "auto_import": self.chk_auto_import.isChecked(),
                "auto_select": self.chk_auto_select.isChecked(),
            },
        }

        try:
            Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
            self._append_log(f"Config saved to {path}")
        except Exception as exc:
            self._append_log(f"Failed to save config: {exc}")

    def _load_training_config(self):
        """Restore dialog settings from a JSON config file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Training Config",
            str(getattr(self, "workspace_default", Path.home())),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._append_log(f"Failed to load config: {exc}")
            return

        if "class_name" in config:
            self.line_class.setText(config["class_name"])

        self._apply_roles_config(config.get("roles", []))
        self._apply_sources_config(config.get("sources", []))
        self._apply_hyperparams_config(config.get("hyperparams", {}))
        self._apply_imgsz_config(config.get("imgsz", {}))
        self._apply_split_config(config.get("split", {}))
        self._apply_seed_dedup_config(config)
        self._apply_crop_derivation_config(config.get("crop_derivation", {}))
        self._apply_base_models_config(config.get("base_models", {}))
        self._apply_augmentation_config(config.get("augmentation", {}))

        if "device" in config:
            self.combo_device.setCurrentText(config["device"])

        self._apply_publish_config(config.get("publish", {}))
        self._append_log(f"Config loaded from {path}")

    def _apply_roles_config(self, roles):
        """Apply role checkboxes from loaded config."""
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            chk = getattr(self, f"chk_role_{role_key}", None)
            if chk:
                chk.setChecked(role_key in roles)

    def _apply_sources_config(self, sources):
        """Apply source table rows from loaded config."""
        if sources:
            self.table_sources.setRowCount(0)
            for src in sources:
                self._add_source_row(
                    src.get("source_type", "obb"),
                    src.get("path", ""),
                )

    def _apply_hyperparams_config(self, hp):
        """Apply hyperparameter widgets from loaded config."""
        for key, widget_name in (
            ("epochs", "spin_epochs"),
            ("batch", "spin_batch"),
            ("lr0", "spin_lr0"),
            ("patience", "spin_patience"),
            ("workers", "spin_workers"),
        ):
            if key in hp:
                w = getattr(self, widget_name, None)
                if w:
                    w.setValue(hp[key])
        if "cache" in hp:
            self.chk_cache.setChecked(hp["cache"])

    def _apply_imgsz_config(self, isz):
        """Apply image-size spinners from loaded config."""
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            if role_key in isz:
                w = getattr(self, f"spin_imgsz_{role_key}", None)
                if w:
                    w.setValue(isz[role_key])

    def _apply_split_config(self, sp):
        """Apply train/val split values from loaded config."""
        if "train" in sp:
            self.spin_train.setValue(sp["train"])
        if "val" in sp:
            self.spin_val.setValue(sp["val"])

    def _apply_seed_dedup_config(self, config):
        """Apply seed and dedup settings from loaded config."""
        if "seed" in config:
            self.spin_seed.setValue(config["seed"])
        if "dedup" in config:
            self.chk_dedup.setChecked(config["dedup"])

    def _apply_crop_derivation_config(self, cd):
        """Apply crop derivation settings from loaded config."""
        if "pad_ratio" in cd:
            self.spin_crop_pad.setValue(cd["pad_ratio"])
        if "min_crop_size_px" in cd:
            self.spin_crop_min_px.setValue(cd["min_crop_size_px"])
        if "enforce_square" in cd:
            self.chk_crop_square.setChecked(cd["enforce_square"])

    def _apply_base_models_config(self, bm):
        """Apply base model combo boxes from loaded config."""
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            if role_key in bm:
                combo = getattr(self, f"combo_model_{role_key}", None)
                if combo:
                    combo.setCurrentText(bm[role_key])

    def _apply_augmentation_config(self, aug):
        """Apply augmentation settings from loaded config."""
        if "enabled" in aug:
            self.aug_group.setChecked(aug["enabled"])
        for key in (
            "fliplr",
            "flipud",
            "degrees",
            "mosaic",
            "mixup",
            "hsv_h",
            "hsv_s",
            "hsv_v",
        ):
            if key in aug:
                w = getattr(self, f"spin_aug_{key}", None)
                if w:
                    w.setValue(aug[key])

    def _apply_publish_config(self, pub):
        """Apply publish settings from loaded config."""
        if "species" in pub:
            self.line_species.setText(pub["species"])
        if "model_tag" in pub:
            self.line_model_tag.setText(pub["model_tag"])
        if "auto_import" in pub:
            self.chk_auto_import.setChecked(pub["auto_import"])
        if "auto_select" in pub:
            self.chk_auto_select.setChecked(pub["auto_select"])

    def _build_device_options(self):
        info = get_device_info()
        options = ["auto", "cpu"]
        if info.get("torch_cuda_available"):
            options.append("cuda")
            count = int(info.get("torch_cuda_device_count", 0) or 0)
            for i in range(count):
                options.append(f"cuda:{i}")
        if info.get("mps_available"):
            options.append("mps")
        return options

    def _append_log(self, text: str):
        self.log_view.append(str(text))
        if hasattr(self, "loss_plot"):
            self.loss_plot.ingest_log_line(str(text))

    def _choose_workspace(self):
        d = QFileDialog.getExistingDirectory(self, "Select Workspace Root")
        if d:
            self.line_workspace.setText(d)
            self.orchestrator = TrainingOrchestrator(d)

    def _get_multiple_dirs(self, title):
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec() != QFileDialog.Accepted:
            return []
        return dlg.selectedFiles()

    def _add_source_row(self, source_type: str, path: str):
        row = self.table_sources.rowCount()
        self.table_sources.insertRow(row)
        pretty = "OBB Source"
        self.table_sources.setItem(row, 0, QTableWidgetItem(pretty))
        self.table_sources.setItem(row, 1, QTableWidgetItem(path))
        self.table_sources.setItem(row, 2, QTableWidgetItem("Pending"))
        self.table_sources.item(row, 0).setData(Qt.UserRole, source_type)

    def _add_obb_sources(self):
        for d in self._get_multiple_dirs("Select OBB Source Datasets"):
            self._add_source_row("obb", d)

    def _remove_selected(self):
        rows = {idx.row() for idx in self.table_sources.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table_sources.removeRow(row)

    def _clear_sources(self):
        self.table_sources.setRowCount(0)

    def _save_dataset_list(self):
        if self.table_sources.rowCount() == 0:
            QMessageBox.warning(
                self, "No Sources", "Add at least one dataset before saving."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dataset List",
            str(self.workspace_default / "dataset_list.json"),
            "JSON Files (*.json)",
        )
        if not path:
            return
        entries = []
        for row in range(self.table_sources.rowCount()):
            st = self.table_sources.item(row, 0).data(Qt.UserRole)
            p = self.table_sources.item(row, 1).text().strip()
            entries.append({"source_type": st, "path": p})
        try:
            Path(path).write_text(json.dumps(entries, indent=2), encoding="utf-8")
            self._append_log(f"Saved dataset list ({len(entries)} entries) → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _load_dataset_list(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dataset List",
            str(self.workspace_default),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return
        if not isinstance(entries, list):
            QMessageBox.critical(
                self, "Load Failed", "Expected a JSON array of dataset entries."
            )
            return
        added = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            st = str(entry.get("source_type", "obb"))
            p = str(entry.get("path", "")).strip()
            if not p:
                continue
            self._add_source_row(st, p)
            added += 1
        self._append_log(f"Loaded {added} datasets from {path}")

    def _analyze_and_preview(self):
        """Analyze object/crop sizes and show sample crop previews."""
        obb_sources = self._collect_sources()
        if not obb_sources:
            QMessageBox.warning(
                self, "No Sources", "Add at least one OBB source dataset."
            )
            return

        from hydra_suite.training.dataset_inspector import (
            analyze_obb_sizes,
            format_size_analysis,
            inspect_obb_or_detect_dataset,
        )

        pad = self.spin_crop_pad.value()
        min_px = self.spin_crop_min_px.value()
        square = self.chk_crop_square.isChecked()
        crop_imgsz = self.spin_imgsz_seq_crop_obb.value()
        direct_imgsz = self.spin_imgsz_obb_direct.value()

        # Merge inspections from all sources.
        from hydra_suite.training.dataset_inspector import DatasetInspection

        merged = DatasetInspection(root_dir="(merged)")
        for src in obb_sources:
            try:
                insp = inspect_obb_or_detect_dataset(src.path)
            except Exception as exc:
                self._append_log(f"Skip {src.path}: {exc}")
                continue
            for split, items in insp.splits.items():
                merged.splits.setdefault(split, []).extend(items)

        if not any(merged.splits.values()):
            QMessageBox.warning(self, "No Data", "No valid images found in sources.")
            return

        try:
            stats = analyze_obb_sizes(
                merged,
                pad_ratio=pad,
                min_crop_size_px=min_px,
                enforce_square=square,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Failed", str(exc))
            return

        # Format reports for both crop OBB and direct roles.
        crop_report, crop_warnings = format_size_analysis(
            stats,
            training_imgsz=crop_imgsz,
            pipeline_mode="crop",
        )
        _, direct_warnings = format_size_analysis(
            stats,
            training_imgsz=direct_imgsz,
            pipeline_mode="full_image",
        )

        sections = []
        sections.append("=" * 50)
        sections.append("DATASET SIZE ANALYSIS")
        sections.append("=" * 50)
        sections.append("")
        sections.append(crop_report)

        if crop_warnings:
            sections.append("")
            sections.append("-" * 40)
            sections.append(f"seq_crop_obb (imgsz={crop_imgsz}):")
            for w in crop_warnings:
                sections.append(f"  {w}")

        if direct_warnings:
            sections.append("")
            sections.append("-" * 40)
            sections.append(f"obb_direct (imgsz={direct_imgsz}):")
            for w in direct_warnings:
                sections.append(f"  {w}")

        if not crop_warnings and not direct_warnings:
            sections.append("")
            sections.append("No compatibility issues detected.")

        self.validation_view.setPlainText("\n".join(sections))

        # Generate crop preview samples.
        self._show_crop_previews(merged, pad, min_px, square, crop_imgsz)

    def _show_crop_previews(self, inspection, pad, min_px, square, imgsz, n_samples=6):
        """Render sample crops in the preview grid."""
        import random

        import cv2
        import numpy as np

        # Clear previous previews.
        while self.preview_grid.count():
            item = self.preview_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        all_items = []
        for split_items in inspection.splits.values():
            all_items.extend(split_items)
        if not all_items:
            self.preview_container.setVisible(False)
            return

        rng = random.Random(42)
        rng.shuffle(all_items)

        previews = []
        for item in all_items:
            if len(previews) >= n_samples:
                break
            lbl_path = Path(item.label_path)
            img_path = Path(item.image_path)
            if not lbl_path.exists() or not img_path.exists():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            h, w = img.shape[:2]

            try:
                lines = lbl_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for ln in lines:
                if len(previews) >= n_samples:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 9:
                    continue
                try:
                    coords = np.asarray(
                        [float(v) for v in parts[1:]], dtype=np.float32
                    ).reshape(4, 2)
                except Exception:
                    continue

                px = coords[:, 0] * float(w)
                py = coords[:, 1] * float(h)
                x1, x2 = float(np.min(px)), float(np.max(px))
                y1, y2 = float(np.min(py)), float(np.max(py))
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

                crop_w = max(float(min_px), bw * (1.0 + 2.0 * max(0.0, pad)))
                crop_h = max(float(min_px), bh * (1.0 + 2.0 * max(0.0, pad)))
                if square:
                    side = max(crop_w, crop_h)
                    crop_w = side
                    crop_h = side

                xi1 = max(0, int(cx - crop_w * 0.5))
                yi1 = max(0, int(cy - crop_h * 0.5))
                xi2 = min(w, int(cx + crop_w * 0.5))
                yi2 = min(h, int(cy + crop_h * 0.5))
                if xi2 <= xi1 or yi2 <= yi1:
                    continue

                crop = img[yi1:yi2, xi1:xi2].copy()

                # Draw the OBB polygon on the crop.
                poly_crop = np.zeros((4, 2), dtype=np.int32)
                poly_crop[:, 0] = (px - xi1).astype(np.int32)
                poly_crop[:, 1] = (py - yi1).astype(np.int32)
                cv2.polylines(crop, [poly_crop], True, (0, 255, 0), 1)

                original_size = max(crop.shape[:2])

                # Resize to training imgsz.
                if imgsz > 0:
                    crop = cv2.resize(
                        crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR
                    )

                previews.append((crop, original_size))

        if not previews:
            self.preview_container.setVisible(False)
            return

        display_size = 140
        for crop_img, orig_sz in previews:
            rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            h_c, w_c = rgb.shape[:2]
            qimg = QImage(rgb.data, w_c, h_c, w_c * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                display_size, display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            lbl = QLabel()
            lbl.setPixmap(pixmap)
            scale = imgsz / max(1, orig_sz) if imgsz > 0 else 1.0
            if scale > 1.05:
                lbl.setToolTip(
                    f"Original {orig_sz}px → {imgsz}px (upscaled {scale:.1f}x)"
                )
            elif scale < 0.95:
                lbl.setToolTip(
                    f"Original {orig_sz}px → {imgsz}px (downscaled {1 / scale:.1f}x)"
                )
            else:
                lbl.setToolTip(f"Original {orig_sz}px ≈ {imgsz}px (good match)")
            lbl.setStyleSheet("border: 1px solid #555; padding: 2px;")
            self.preview_grid.addWidget(lbl)

        self.preview_info.setText(
            f"Showing {len(previews)} sample crops at imgsz={imgsz}. "
            f"Green polygon = OBB annotation. Hover for scale info."
        )
        self.preview_container.setVisible(True)

    def _collect_sources(self):
        obb_sources = []
        for row in range(self.table_sources.rowCount()):
            st = self.table_sources.item(row, 0).data(Qt.UserRole)
            p = self.table_sources.item(row, 1).text().strip()
            if st == "obb":
                obb_sources.append(
                    SourceDataset(path=p, source_type="yolo_obb", name=Path(p).name)
                )
        return obb_sources

    def _validate_sources(self):
        obb_sources = self._collect_sources()
        if not obb_sources:
            QMessageBox.warning(
                self, "No OBB Sources", "Add at least one OBB source dataset."
            )
            return False

        try:
            report = self.orchestrator.preflight_obb_sources(
                obb_sources, require_train_val=False
            )
        except Exception as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return False

        self.validation_view.setPlainText(format_validation_report(report))

        status = "Validated" if report.valid else "Validation Failed"
        for row in range(self.table_sources.rowCount()):
            self.table_sources.setItem(row, 2, QTableWidgetItem(status))

        if not report.valid:
            QMessageBox.warning(
                self,
                "Validation Failed",
                "Source validation failed. See report for actionable issues.",
            )
            return False
        return True

    def _selected_roles(self):
        roles = []
        if self.chk_role_obb_direct.isChecked():
            roles.append(TrainingRole.OBB_DIRECT)
        if self.chk_role_seq_detect.isChecked():
            roles.append(TrainingRole.SEQ_DETECT)
        if self.chk_role_seq_crop_obb.isChecked():
            roles.append(TrainingRole.SEQ_CROP_OBB)
        return roles

    def _build_role_datasets(self):
        if not self._validate_sources():
            return False

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return False

        obb_sources = self._collect_sources()
        try:
            split = SplitConfig(
                train=self.spin_train.value(), val=self.spin_val.value(), test=0.0
            )
            merged = self.orchestrator.build_merged_obb_dataset(
                obb_sources,
                class_name=self.line_class.text().strip() or "object",
                split_cfg=split,
                seed=self.spin_seed.value(),
                dedup=self.chk_dedup.isChecked(),
            )
            self.role_dataset_dirs = {}
            self._append_log(f"Merged dataset: {merged.dataset_dir}")

            for role in roles:
                build = self.orchestrator.build_role_dataset(
                    role,
                    merged.dataset_dir,
                    class_name=self.line_class.text().strip() or "object",
                    crop_pad_ratio=self.spin_crop_pad.value(),
                    min_crop_size_px=self.spin_crop_min_px.value(),
                    enforce_square=self.chk_crop_square.isChecked(),
                )
                self.role_dataset_dirs[role.value] = build.dataset_dir
                self._append_log(
                    f"Prepared [{role.value}] dataset: {build.dataset_dir}"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Build Failed", str(exc))
            return False

        QMessageBox.information(
            self, "Datasets Ready", "Role datasets built successfully."
        )
        return True

    def _base_model_for_role(self, role: TrainingRole) -> str:
        if role == TrainingRole.OBB_DIRECT:
            return self.combo_model_obb_direct.currentText().strip()
        if role == TrainingRole.SEQ_DETECT:
            return self.combo_model_seq_detect.currentText().strip()
        if role == TrainingRole.SEQ_CROP_OBB:
            return self.combo_model_seq_crop_obb.currentText().strip()
        return ""

    def _imgsz_for_role(self, role: TrainingRole) -> int:
        if role == TrainingRole.OBB_DIRECT:
            return self.spin_imgsz_obb_direct.value()
        if role == TrainingRole.SEQ_DETECT:
            return self.spin_imgsz_seq_detect.value()
        if role == TrainingRole.SEQ_CROP_OBB:
            return self.spin_imgsz_seq_crop_obb.value()
        return 640

    @staticmethod
    def _infer_size_token(model_path: str) -> str:
        name = Path(str(model_path or "")).name.lower()
        for token in (
            "26n",
            "26s",
            "26m",
            "26l",
            "26x",
            "11n",
            "11s",
            "11m",
            "11l",
            "11x",
        ):
            if token in name:
                return token
        return "unknown"

    def _publish_meta_for_role(self, role: TrainingRole, base_model: str) -> dict:
        species = self.line_species.text().strip() or "species"
        tag = self.line_model_tag.text().strip() or "train"
        role_suffix = role.value
        training_params = {
            "imgsz": self._imgsz_for_role(role),
        }
        if role == TrainingRole.SEQ_CROP_OBB:
            training_params["crop_pad_ratio"] = self.spin_crop_pad.value()
            training_params["min_crop_size_px"] = self.spin_crop_min_px.value()
            training_params["enforce_square"] = self.chk_crop_square.isChecked()
        return {
            "size": self._infer_size_token(base_model),
            "species": species,
            "model_info": f"{tag}_{role_suffix}",
            "training_params": training_params,
        }

    def _start_detached(self):
        """Launch training as a detached subprocess."""
        import subprocess

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return
        if not self.role_dataset_dirs:
            if not self._build_role_datasets():
                return

        launched = []
        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                continue
            base_model = self._base_model_for_role(role)
            if not base_model:
                continue

            batch_val = (
                -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
            )
            spec = TrainingRunSpec(
                role=role,
                source_datasets=self._collect_sources(),
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
            )
            from hydra_suite.training.runner import build_ultralytics_command

            run_dir = self.workspace_default / "runs" / ("detached_" + role.value)
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_ultralytics_command(spec, str(run_dir))

            log_file = run_dir / "detached_output.log"
            with open(log_file, "w") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            launched.append((role.value, proc.pid, str(log_file)))
            self._append_log(
                "Detached "
                + role.value
                + " training: PID="
                + str(proc.pid)
                + ", log="
                + str(log_file)
            )

        if launched:
            msg = "\n".join(
                "* " + role + ": PID " + str(pid) + "\n  Log: " + log
                for role, pid, log in launched
            )
            QMessageBox.information(
                self,
                "Detached Training Started",
                "Training launched in background:\n\n" + msg + "\n\n"
                "You can close this dialog. Check Run History for results.",
            )

    def _start_training(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return

        if not self.role_dataset_dirs:
            if not self._build_role_datasets():
                return

        if not self.chk_auto_import.isChecked() and self.chk_auto_select.isChecked():
            QMessageBox.warning(
                self,
                "Invalid Publish Settings",
                "Auto-select requires auto-import to be enabled.",
            )
            return

        source_obb = self._collect_sources()
        role_entries = []
        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                QMessageBox.warning(
                    self,
                    "Missing Dataset",
                    f"No dataset prepared for role: {role.value}",
                )
                return

            base_model = self._base_model_for_role(role)
            if not base_model:
                QMessageBox.warning(
                    self, "Base Model", f"Set base model for role: {role.value}"
                )
                return

            aug_args: dict[str, float] = {}
            if self.aug_group.isChecked():
                aug_args = {
                    "fliplr": self.aug_fliplr.value(),
                    "flipud": self.aug_flipud.value(),
                    "degrees": self.aug_degrees.value(),
                    "mosaic": self.aug_mosaic.value(),
                    "mixup": self.aug_mixup.value(),
                    "hsv_h": self.aug_hsv_h.value(),
                    "hsv_s": self.aug_hsv_s.value(),
                    "hsv_v": self.aug_hsv_v.value(),
                }

            batch_val = (
                -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
            )
            spec = TrainingRunSpec(
                role=role,
                source_datasets=source_obb,
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
                augmentation_profile=AugmentationProfile(
                    enabled=self.aug_group.isChecked(),
                    args=aug_args,
                ),
                publish_policy=PublishPolicy(
                    auto_import=self.chk_auto_import.isChecked(),
                    auto_select=self.chk_auto_select.isChecked(),
                ),
            )
            role_entries.append(
                {
                    "role": role,
                    "spec": spec,
                    "publish_meta": self._publish_meta_for_role(role, base_model),
                }
            )

        self.worker = RoleTrainingWorker(self.orchestrator, role_entries)
        self.worker.log_signal.connect(self._append_log)
        self.worker.role_started.connect(self._on_role_started)
        self.worker.role_finished.connect(self._on_role_finished)
        self.worker.progress_signal.connect(self._on_role_progress)
        self.worker.done_signal.connect(self._on_training_done)

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        if hasattr(self, "loss_plot"):
            self.loss_plot.clear()
        self.worker.start()

    def _stop_training(self):
        if self.worker is not None:
            self.worker.cancel()
            self._append_log("Cancellation requested…")

    def _on_role_started(self, role: str):
        self._append_log(f"=== START {role} ===")

    def _on_role_finished(self, role: str, ok: bool, message: str):
        self._append_log(f"=== {'OK' if ok else 'FAIL'} {role}: {message} ===")

    def _on_role_progress(self, role: str, cur: int, total: int):
        total = max(1, int(total))
        cur = max(0, min(total, int(cur)))
        pct = int((cur / total) * 100.0)
        self.progress.setValue(pct)
        self.progress.setFormat(f"{role}: {cur}/{total} ({pct}%)")

    def _on_training_done(self, results: list):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setFormat("Done")

        # Store results and check for resumable runs
        self._last_training_results = results
        for r in results:
            run_dir = ""
            artifact = r.get("artifact_path", "")
            if artifact:
                _wdir = Path(artifact).parent
                if _wdir.name == "weights":
                    run_dir = str(_wdir.parent)
            r["_run_dir"] = run_dir
        self.btn_resume.setEnabled(
            any(
                r.get("_run_dir")
                and Path(r["_run_dir"]).joinpath("weights", "last.pt").exists()
                for r in results
            )
        )

        succeeded = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        self.btn_quick_test.setEnabled(bool(succeeded))

        self._append_log(
            f"Session complete: {len(succeeded)} success, {len(failed)} failed"
        )

        if self.chk_auto_select.isChecked():
            self._try_auto_select_parent_models(results)

        if failed:
            QMessageBox.warning(
                self,
                "Training Completed with Failures",
                f"Succeeded: {len(succeeded)}\nFailed: {len(failed)}\nSee logs for details.",
            )
        else:
            QMessageBox.information(
                self,
                "Training Completed",
                f"All {len(succeeded)} selected roles completed successfully.",
            )

    def _show_history(self):
        """Open the training run history viewer dialog."""
        from hydra_suite.trackerkit.gui.dialogs.run_history_dialog import (
            RunHistoryDialog,
        )

        dlg = RunHistoryDialog(parent=self)
        dlg.exec()

    def _quick_test(self):
        """Open the Quick Test dialog for the last successful training result."""
        from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import ModelTestDialog

        results = getattr(self, "_last_training_results", [])
        succeeded = [r for r in results if r.get("success")]
        if not succeeded:
            QMessageBox.warning(
                self,
                "No Model Available",
                "No successfully trained model found. Run training first.",
            )
            return

        result = succeeded[0]
        model_path = result.get("published_model_path") or result.get(
            "artifact_path", ""
        )
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Model file not found: {model_path}",
            )
            return

        role = result.get("role", "obb_direct")
        dataset_dir = self.role_dataset_dirs.get(role, "")
        if not dataset_dir:
            QMessageBox.warning(
                self,
                "No Dataset",
                f"No dataset directory found for role '{role}'.",
            )
            return

        device = self.combo_device.currentText() or "cpu"

        # Pick the appropriate imgsz for the role
        imgsz_map = {
            "obb_direct": self.spin_imgsz_obb_direct.value(),
            "seq_detect": self.spin_imgsz_seq_detect.value(),
            "seq_crop_obb": self.spin_imgsz_seq_crop_obb.value(),
        }
        imgsz = imgsz_map.get(role, 640)

        dlg = ModelTestDialog(
            model_path=model_path,
            role=role,
            dataset_dir=dataset_dir,
            device=device,
            imgsz=imgsz,
            crop_pad_ratio=self.spin_crop_pad.value(),
            min_crop_size_px=self.spin_crop_min_px.value(),
            enforce_square=self.chk_crop_square.isChecked(),
            parent=self,
        )
        dlg.exec()

    def _resume_training(self):
        """Resume training from the last.pt checkpoint of the most recent run."""
        last_pt = None
        resume_result = None
        for r in reversed(self._last_training_results):
            run_dir = r.get("_run_dir", "")
            if run_dir:
                candidate = Path(run_dir) / "weights" / "last.pt"
                if candidate.exists():
                    last_pt = candidate
                    resume_result = r
                    break

        if last_pt is None:
            QMessageBox.warning(
                self,
                "No Checkpoint Found",
                "Could not find a last.pt checkpoint from the previous training run.",
            )
            return

        role_str = str(resume_result.get("role", ""))
        try:
            role = TrainingRole(role_str)
        except ValueError:
            QMessageBox.warning(
                self,
                "Resume Failed",
                f"Unknown training role: {role_str}",
            )
            return

        resume_batch_val = (
            -1 if self.chk_auto_batch.isChecked() else int(self.spin_batch.value())
        )
        spec = TrainingRunSpec(
            role=role,
            source_datasets=[],
            derived_dataset_dir=resume_result.get("_run_dir", ""),
            base_model=str(last_pt),
            hyperparams=TrainingHyperParams(
                epochs=int(self.spin_epochs.value()),
                imgsz=int(self.spin_imgsz.value()),
                batch=resume_batch_val,
                lr0=float(self.spin_lr0.value()),
                patience=int(self.spin_patience.value()),
                workers=int(self.spin_workers.value()),
            ),
            resume_from=str(last_pt),
        )

        publish_meta = {
            "class_name": self.class_name,
            "resumed_from": str(last_pt),
        }
        entry = {"role": role, "spec": spec, "publish_meta": publish_meta}

        self._append_log(f"Resuming training from {last_pt}")
        self.btn_train.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("Resuming...")

        self.worker = RoleTrainingWorker(self.orchestrator, [entry])
        self.worker.log_signal.connect(self._append_log)
        self.worker.role_started.connect(self._on_role_started)
        self.worker.role_finished.connect(self._on_role_finished)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.done_signal.connect(self._on_training_done)
        self.worker.start()

    def _try_auto_select_parent_models(self, results: list[dict]):
        parent = self.parent()
        if parent is None:
            return

        for r in results:
            if not r.get("success"):
                continue
            key = str(r.get("published_registry_key") or "").strip()
            if not key:
                continue
            role = str(r.get("role") or "")
            try:
                if role == TrainingRole.OBB_DIRECT.value:
                    if hasattr(parent, "_refresh_yolo_model_combo"):
                        parent._refresh_yolo_model_combo(preferred_model_path=key)
                    if hasattr(parent, "_set_yolo_model_selection"):
                        parent._set_yolo_model_selection(key)
                elif role == TrainingRole.SEQ_DETECT.value:
                    if hasattr(parent, "_refresh_yolo_detect_model_combo"):
                        parent._refresh_yolo_detect_model_combo(
                            preferred_model_path=key
                        )
                    if hasattr(parent, "_set_yolo_detect_model_selection"):
                        parent._set_yolo_detect_model_selection(key)
                elif role == TrainingRole.SEQ_CROP_OBB.value:
                    if hasattr(parent, "_refresh_yolo_crop_obb_model_combo"):
                        parent._refresh_yolo_crop_obb_model_combo(
                            preferred_model_path=key
                        )
                    if hasattr(parent, "_set_yolo_crop_obb_model_selection"):
                        parent._set_yolo_crop_obb_model_selection(key)
                    if hasattr(parent, "_apply_crop_obb_training_params"):
                        parent._apply_crop_obb_training_params()
            except Exception as exc:
                logger.warning("Auto-select model failed for role %s: %s", role, exc)
