#!/usr/bin/env python3
"""
Training dialogs and workers.
"""

import csv
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.paths import get_models_dir, get_training_runs_dir
from hydra_suite.utils.conda_utils import popen_conda, run_conda
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from ...core.extensions import (
    build_coco_keypoints_dataset,
    build_yolo_pose_dataset,
    list_labeled_indices,
)
from ...core.vitpose_checkpoints import CATALOG
from ...core.vitpose_training import (
    build_training_command,
    parse_progress_line,
    prepare_run,
)
from .evaluation import EvaluationDashboardDialog
from .utils import (
    _load_dialog_settings,
    _save_dialog_settings,
    get_available_devices,
    get_yolo_pose_base_models,
    list_images_in_dir,
    list_sleap_envs,
    load_yolo_dataset_items,
    make_loss_plot_image,
)

logger = logging.getLogger("pose_label.dialogs.training")


class TrainingWorker(QObject):
    """Worker for running YOLO pose training."""

    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        model_weights: str,
        dataset_yaml: Path,
        run_dir: Path,
        epochs: int,
        patience: int,
        batch: int,
        imgsz: int,
        device: str,
        augment: bool,
        auto_batch: bool,
        hsv_h: float,
        hsv_s: float,
        hsv_v: float,
        degrees: float,
        translate: float,
        scale: float,
    ):
        super().__init__()
        self.model_weights = model_weights
        self.dataset_yaml = Path(dataset_yaml)
        self.run_dir = Path(run_dir)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch = int(batch)
        self.imgsz = int(imgsz)
        self.device = device
        self.augment = bool(augment)
        self.auto_batch = bool(auto_batch)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self._cancel = False
        self._model = None
        self._proc = None

    def cancel(self):
        """Request early termination and send SIGTERM to any running training subprocess."""
        self._cancel = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self):
        """Build the YOLO pose dataset, launch the training subprocess, and stream log output."""
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Training run dir: {self.run_dir}")
            self.progress.emit(0, max(1, int(self.epochs)))

            def _run_cmd(cmd):
                return subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
                )

            yolo_bin = shutil.which("yolo")
            batch = int(self.batch)

            while True:
                self.log.emit(f"Starting training (subprocess)… batch={batch}")
                cli_args = [
                    "task=pose",
                    "mode=train",
                    f"model={self.model_weights}",
                    f"data={str(self.dataset_yaml)}",
                    f"epochs={self.epochs}",
                    f"patience={int(getattr(self, 'patience', 10))}",
                    f"batch={batch}",
                    f"imgsz={self.imgsz}",
                    f"project={str(self.run_dir.parent)}",
                    f"name={self.run_dir.name}",
                    "exist_ok=True",
                ]
                if self.device and self.device != "auto":
                    cli_args.append(f"device={self.device}")
                if self.augment:
                    cli_args += [
                        "augment=True",
                        f"hsv_h={self.hsv_h}",
                        f"hsv_s={self.hsv_s}",
                        f"hsv_v={self.hsv_v}",
                        f"degrees={self.degrees}",
                        f"translate={self.translate}",
                        f"scale={self.scale}",
                    ]

                if yolo_bin:
                    self._proc = _run_cmd([yolo_bin] + cli_args)
                else:
                    # Fallback to python -c
                    py_code = (
                        "from ultralytics import YOLO\n"
                        "model=YOLO(r'''{model}''')\n"
                        "model.train(\n"
                        "  data=r'''{data}''',\n"
                        "  epochs={epochs},\n"
                        "  patience={patience},\n"
                        "  batch={batch},\n"
                        "  imgsz={imgsz},\n"
                        "  project=r'''{project}''',\n"
                        "  name=r'''{name}''',\n"
                        "  exist_ok=True,\n"
                        "  device=r'''{device}''',\n"
                        "  augment={augment},\n"
                        "  hsv_h={hsv_h},\n"
                        "  hsv_s={hsv_s},\n"
                        "  hsv_v={hsv_v},\n"
                        "  degrees={degrees},\n"
                        "  translate={translate},\n"
                        "  scale={scale},\n"
                        ")\n"
                    ).format(
                        model=self.model_weights,
                        data=str(self.dataset_yaml),
                        epochs=self.epochs,
                        patience=self.patience,
                        batch=batch,
                        imgsz=self.imgsz,
                        project=str(self.run_dir.parent),
                        name=self.run_dir.name,
                        device=self.device if self.device else "auto",
                        augment=bool(self.augment),
                        hsv_h=self.hsv_h,
                        hsv_s=self.hsv_s,
                        hsv_v=self.hsv_v,
                        degrees=self.degrees,
                        translate=self.translate,
                        scale=self.scale,
                    )
                    self._proc = _run_cmd([sys.executable, "-c", py_code])

                assert self._proc.stdout is not None
                ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
                saw_oom = False
                for line in self._proc.stdout:
                    if self._cancel:
                        break
                    msg = ansi_re.sub("", line).rstrip()
                    self.log.emit(msg)
                    if (
                        "out of memory" in msg.lower()
                        or "cuda out of memory" in msg.lower()
                    ):
                        saw_oom = True
                    m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", msg)
                    if not m:
                        m = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s", msg)
                    if m:
                        try:
                            cur = int(m.group(1))
                            total = int(m.group(2))
                            if total > 0:
                                self.progress.emit(cur, total)
                        except Exception:
                            pass

                rc = self._proc.wait()
                if self._cancel:
                    self.failed.emit("Canceled.")
                    return
                if rc == 0:
                    break
                if saw_oom and self.auto_batch and batch > 1:
                    new_batch = max(1, batch // 2)
                    if new_batch == batch:
                        self.failed.emit("Training failed (OOM).")
                        return
                    self.log.emit(f"OOM detected. Retrying with batch={new_batch}")
                    batch = new_batch
                    continue
                self.failed.emit(f"Training failed (exit code {rc}).")
                return

            weights_dir = self.run_dir / "weights"
            best = weights_dir / "best.pt"
            last = weights_dir / "last.pt"
            weights_path = best if best.exists() else last

            self.log.emit("Training finished.")
            self.finished.emit(
                {
                    "run_dir": str(self.run_dir),
                    "weights": str(weights_path) if weights_path.exists() else "",
                }
            )
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            self._cleanup_gpu()

    def _cleanup_gpu(self):
        try:
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        except Exception:
            pass


class ViTPoseTrainingWorker(QObject):
    """Runs ViTPose fine-tuning as an in-env subprocess (mirrors TrainingWorker)."""

    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        image_paths,
        labels_dir,
        run_dir,
        cache_dir,
        class_names,
        keypoint_names,
        skeleton_edges,
        variant,
        init_checkpoint,
        num_keypoints,
        epochs,
        batch,
        device,
    ):
        super().__init__()
        self.image_paths = list(image_paths)
        self.labels_dir = Path(labels_dir)
        self.run_dir = Path(run_dir)
        self.cache_dir = Path(cache_dir)
        self.class_names = list(class_names)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges)
        self.variant = variant
        self.init_checkpoint = init_checkpoint
        self.num_keypoints = int(num_keypoints)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.device = device
        self._cancel = False
        self._proc = None

    def cancel(self):
        self._cancel = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self):
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit("Building COCO-keypoints dataset…")
            ds = build_coco_keypoints_dataset(
                image_paths=self.image_paths,
                labels_dir=self.labels_dir,
                output_dir=self.run_dir / "dataset",
                class_names=self.class_names,
                keypoint_names=self.keypoint_names,
                skeleton_edges=self.skeleton_edges,
            )
            params = dict(
                init_checkpoint=self.init_checkpoint,
                variant=self.variant,
                num_keypoints=self.num_keypoints,
                dataset_dir=str(ds["dataset_dir"]),
                device=self.device,
                epochs=self.epochs,
                batch_size=self.batch,
            )
            run_json = prepare_run(params, self.run_dir, self.cache_dir)
            cmd = build_training_command(run_json)
            self.log.emit(f"Launching: {' '.join(cmd)}")
            self.progress.emit(0, max(1, self.epochs))

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            for line in self._proc.stdout:
                if self._cancel:
                    break
                self.log.emit(line.rstrip())
                prog = parse_progress_line(line)
                if prog is not None:
                    self.progress.emit(prog["epoch"] + 1, self.epochs)
            code = self._proc.wait()
            if self._cancel:
                self.failed.emit("Training cancelled.")
            elif code != 0:
                self.failed.emit(f"Training subprocess exited with code {code}.")
            else:
                self.finished.emit(
                    {
                        "run_dir": str(self.run_dir),
                        "best": str(self.run_dir / "best.pt"),
                    }
                )
        except Exception as exc:  # surfaced to the dialog
            self.failed.emit(str(exc))


class SleapExportWorker(QObject):
    """Worker for exporting to SLEAP."""

    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        env_name: str,
        image_paths: List[Path],
        labels_dir: Path,
        out_root: Path,
        class_names: List[str],
        keypoint_names: List[str],
        skeleton_edges: List[Tuple[int, int]],
        slp_path: Path,
        include_aux: bool,
        embed: bool,
        aux_datasets: Optional[List[Tuple[List[Path], Path]]] = None,
        aux_items: Optional[List[Tuple[Path, Path]]] = None,
    ):
        super().__init__()
        self.env_name = env_name
        self.image_paths = list(image_paths)
        self.labels_dir = Path(labels_dir)
        self.out_root = Path(out_root)
        self.class_names = list(class_names)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges)
        self.slp_path = Path(slp_path)
        self.include_aux = bool(include_aux)
        self.embed = bool(embed)
        self.aux_datasets = aux_datasets or []
        self.aux_items = aux_items or []

    def run(self):
        """Export annotations to COCO format and convert them to a SLEAP .slp dataset via the conda env."""
        try:
            if not self.env_name:
                self.failed.emit("No SLEAP conda environment selected.")
                return

            export_dir = self.slp_path.parent / f"{self.slp_path.stem}_data"
            if export_dir.exists():
                shutil.rmtree(export_dir, ignore_errors=True)
            export_dir.mkdir(parents=True, exist_ok=True)

            extra_datasets = self.aux_datasets if self.include_aux else []
            extra_items = self.aux_items if self.include_aux else []

            self.log.emit("Building COCO export dataset...")
            info = build_coco_keypoints_dataset(
                self.image_paths,
                self.labels_dir,
                export_dir,
                self.class_names,
                self.keypoint_names,
                self.skeleton_edges,
                extra_datasets=extra_datasets,
                extra_items=extra_items,
            )
            self.log.emit(
                f"Export dataset ready: {info.get('labeled_count', 0)} frames."
            )

            coco_path = info["coco_path"]

            req = {
                "coco_path": str(coco_path),
                "out_slp": str(self.slp_path),
                "embed": "all" if self.embed else "",
            }
            req_path = (
                Path(tempfile.gettempdir())
                / f"sleap_export_{os.getpid()}_{uuid.uuid4().hex}.json"
            )
            req_path.write_text(json.dumps(req), encoding="utf-8")

            code = (
                "import json,sys\n"
                "from pathlib import Path\n"
                "import sleap_io as sio\n"
                "cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
                "coco_path=cfg['coco_path']\n"
                "labels=sio.load_file(coco_path, format='coco')\n"
                "embed=cfg.get('embed') or None\n"
                "if embed in ('all','True','true'):\n"
                "  embed='all'\n"
                "if hasattr(sio,'save_file'):\n"
                "  sio.save_file(labels, cfg['out_slp'], format='slp', embed=embed)\n"
                "else:\n"
                "  sio.save_slp(labels, cfg['out_slp'], embed=embed)\n"
                "print('OK')\n"
            )

            cmd = [
                "conda",
                "run",
                "-n",
                self.env_name,
                "python",
                "-c",
                code,
                str(req_path),
            ]
            proc = run_conda(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                err = (
                    proc.stderr.strip() or proc.stdout.strip() or "SLEAP export failed."
                )
                if "No module named" in err and "sleap_io" in err:
                    err = (
                        "sleap-io is not installed in the selected conda env. "
                        "Install it in that env (e.g., `conda install -c conda-forge sleap-io`)."
                    )
                self.failed.emit(err)
                return

            self.finished.emit(str(self.slp_path))
        except Exception as e:
            self.failed.emit(str(e))


class TrainingRunnerDialog(QDialog):
    """Dialog to configure and run training/export."""

    def __init__(self, parent, project, image_paths: List[Path]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Runner")
        self.setMinimumSize(QSize(760, 600))

        self.project = project
        self.image_paths = image_paths
        self._thread = None
        self._worker = None
        self._last_run_dir = None
        self._last_weights = None
        self._train_start_ts = None
        self._loss_source_path = None
        self._sleap_thread = None
        self._sleap_worker = None
        self._sleap_open_after_export = False
        self._sleap_env_pref = ""
        self._aux_datasets: List[Dict[str, object]] = []
        self._aux_items: List[Tuple[Path, Path]] = []
        self._last_aux_path = ""
        self._loss_timer = QTimer(self)
        self._loss_timer.setInterval(1000)
        self._loss_timer.timeout.connect(self._update_loss_plot)

        layout = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        # Backend
        self.backend_group = QGroupBox("Backend")
        backend_layout = QFormLayout(self.backend_group)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["YOLO Pose", "ViTPose", "SLEAP"])
        backend_layout.addRow("Backend", self.backend_combo)

        content_layout.addWidget(self.backend_group)

        # ViTPose config (variant + init checkpoint)
        self.vitpose_group = QGroupBox("ViTPose Config")
        vitpose_layout = QFormLayout(self.vitpose_group)

        self.vitpose_variant_combo = QComboBox()
        self.vitpose_variant_combo.addItems(["B", "L", "H"])
        self.vitpose_variant_combo.setCurrentText("B")
        vitpose_layout.addRow("Variant", self.vitpose_variant_combo)

        self.vitpose_checkpoint_combo = QComboBox()
        self.vitpose_checkpoint_combo.setEditable(True)
        self.vitpose_checkpoint_combo.addItems(list(CATALOG.keys()) + ["Browse…"])
        vitpose_layout.addRow("Init checkpoint", self.vitpose_checkpoint_combo)

        content_layout.addWidget(self.vitpose_group)

        # Config
        self.cfg_group = QGroupBox("Config")
        cfg_layout = QFormLayout(self.cfg_group)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(get_yolo_pose_base_models())
        self.model_combo.setCurrentText("yolo26n-pose.pt")
        self.btn_model_browse = QPushButton("Browse…")
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.btn_model_browse)
        cfg_layout.addRow("Base weights", model_row)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(16)
        cfg_layout.addRow("Batch size", self.batch_spin)

        self.cb_auto_batch = QCheckBox("Auto-reduce batch on OOM")
        self.cb_auto_batch.setChecked(True)
        cfg_layout.addRow("", self.cb_auto_batch)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)
        cfg_layout.addRow("Epochs", self.epochs_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(10)
        cfg_layout.addRow("Early stopping patience", self.patience_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        cfg_layout.addRow("Image size", self.imgsz_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        cfg_layout.addRow("Device", self.device_combo)

        # Dataset options (moved here, YOLO-only)
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.05, 0.95)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.8)
        cfg_layout.addRow("Train fraction", self.train_split_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        cfg_layout.addRow("Random seed", self.seed_spin)

        self.cb_ignore_occluded = QCheckBox("Ignore occluded keypoints in training")
        self.cb_ignore_occluded.setChecked(True)
        cfg_layout.addRow("", self.cb_ignore_occluded)

        self.lbl_labeled = QLabel("")
        self._refresh_labeled_count()
        cfg_layout.addRow("Labeled status", self.lbl_labeled)

        content_layout.addWidget(self.cfg_group)

        # Augmentations
        self.aug_group = QGroupBox("Augmentations")
        aug_layout = QVBoxLayout(self.aug_group)
        self.cb_augment = QCheckBox("Enable augmentations")
        self.cb_augment.setChecked(True)
        aug_layout.addWidget(self.cb_augment)

        self.aug_widgets = []
        grid = QGridLayout()
        aug_layout.addLayout(grid)
        row = 0
        col = 0

        def add_row(label: str, widget: QWidget, tip: str):
            nonlocal row, col
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            widget.setToolTip(tip)
            grid.addWidget(lbl, row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)
            if col == 0:
                col = 1
            else:
                col = 0
                row += 1

        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setSingleStep(0.005)
        self.hsv_h_spin.setValue(0.01)
        add_row(
            "hsv_h:",
            self.hsv_h_spin,
            "Hue jitter (fraction). Small values recommended.",
        )
        self.aug_widgets.append(self.hsv_h_spin)

        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setSingleStep(0.05)
        self.hsv_s_spin.setValue(0.2)
        add_row(
            "hsv_s:",
            self.hsv_s_spin,
            "Saturation jitter (fraction).",
        )
        self.aug_widgets.append(self.hsv_s_spin)

        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setSingleStep(0.05)
        self.hsv_v_spin.setValue(0.1)
        add_row(
            "hsv_v:",
            self.hsv_v_spin,
            "Value/brightness jitter (fraction).",
        )
        self.aug_widgets.append(self.hsv_v_spin)

        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setSingleStep(1.0)
        self.degrees_spin.setValue(5.0)
        add_row(
            "degrees:",
            self.degrees_spin,
            "Rotation range in degrees.",
        )
        self.aug_widgets.append(self.degrees_spin)

        self.translate_spin = QDoubleSpinBox()
        self.translate_spin.setRange(0.0, 1.0)
        self.translate_spin.setSingleStep(0.05)
        self.translate_spin.setValue(0.05)
        add_row(
            "translate:",
            self.translate_spin,
            "Translation range as fraction of image size.",
        )
        self.aug_widgets.append(self.translate_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(0.2)
        add_row(
            "scale:",
            self.scale_spin,
            "Scale variation (fraction).",
        )
        self.aug_widgets.append(self.scale_spin)

        content_layout.addWidget(self.aug_group)

        # Dataset
        self.data_group = QGroupBox("Dataset")
        data_layout = QFormLayout(self.data_group)

        self.aux_list = QListWidget()
        self.btn_add_aux = QPushButton("Add auxiliary project…")
        self.btn_remove_aux = QPushButton("Remove selected")
        aux_row = QHBoxLayout()
        aux_row.addWidget(self.btn_add_aux)
        aux_row.addWidget(self.btn_remove_aux)
        data_layout.addRow("Auxiliary datasets", self.aux_list)
        data_layout.addRow("", aux_row)

        content_layout.addWidget(self.data_group)

        # SLEAP Export
        self.sleap_group = QGroupBox("SLEAP Export")
        sleap_layout = QFormLayout(self.sleap_group)

        # Conda env
        env_row = QHBoxLayout()
        self.combo_sleap_env = QComboBox()
        self.combo_sleap_env.setToolTip(
            "Select a conda environment with SLEAP installed.\n"
            "Environment name must start with 'sleap'."
        )
        self.btn_sleap_refresh = QPushButton("↻")
        self.btn_sleap_refresh.setMaximumWidth(40)
        self.btn_sleap_refresh.setToolTip("Refresh conda environments list")
        env_row.addWidget(self.combo_sleap_env, 1)
        env_row.addWidget(self.btn_sleap_refresh)
        sleap_layout.addRow("Conda environment", env_row)

        self.lbl_sleap_env_status = QLabel("")
        self.lbl_sleap_env_status.setStyleSheet("QLabel { color: #b00; }")
        sleap_layout.addRow("", self.lbl_sleap_env_status)

        # Output path
        out_row = QHBoxLayout()
        self.sleap_out_edit = QLineEdit("")
        self.btn_sleap_browse = QPushButton("Browse…")
        out_row.addWidget(self.sleap_out_edit, 1)
        out_row.addWidget(self.btn_sleap_browse)
        sleap_layout.addRow("Output .slp file", out_row)

        # Options
        self.cb_sleap_include_aux = QCheckBox("Include auxiliary datasets")
        self.cb_sleap_include_aux.setChecked(True)
        sleap_layout.addRow("", self.cb_sleap_include_aux)

        self.cb_sleap_embed = QCheckBox("Embed frames in .slp")
        self.cb_sleap_embed.setChecked(False)
        sleap_layout.addRow("", self.cb_sleap_embed)

        self.lbl_sleap_format = QLabel(
            "Format: COCO keypoints (occluded treated as missing for SLEAP)"
        )
        self.lbl_sleap_format.setStyleSheet("QLabel { color: #555; }")
        sleap_layout.addRow("", self.lbl_sleap_format)

        # Actions
        sleap_btns = QHBoxLayout()
        self.btn_sleap_export = QPushButton("Export SLEAP Labels")
        self.btn_sleap_open = QPushButton("Open in SLEAP")
        sleap_btns.addWidget(self.btn_sleap_export)
        sleap_btns.addWidget(self.btn_sleap_open)
        sleap_layout.addRow("", sleap_btns)

        content_layout.addWidget(self.sleap_group)

        # Run info
        self.info_group = QGroupBox("Run")
        info_layout = QFormLayout(self.info_group)
        self.lbl_run_dir = QLabel("Run dir: (not started)")
        info_layout.addRow(self.lbl_run_dir)
        content_layout.addWidget(self.info_group)

        # Logs + progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        content_layout.addWidget(self.progress)

        self.lbl_loss_plot = QLabel()
        self.lbl_loss_plot.setMinimumHeight(220)
        self.lbl_loss_plot.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.lbl_loss_plot)

        # Loss components selector
        self.loss_components_box = QGroupBox("Loss components")
        self.loss_components_layout = QHBoxLayout(self.loss_components_box)
        self.loss_component_checks: Dict[str, QCheckBox] = {}
        content_layout.addWidget(self.loss_components_box)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        content_layout.addWidget(self.log_view, 1)

        # Buttons
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_open_eval = QPushButton("Open Evaluation")
        self.btn_open_eval.setEnabled(False)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_open_eval)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        content_layout.addLayout(btns)

        # Wiring
        self.btn_model_browse.clicked.connect(self._browse_model)
        self.btn_start.clicked.connect(self._start_training)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_close.clicked.connect(self.reject)
        self.btn_open_eval.clicked.connect(self._open_eval)
        self.btn_add_aux.clicked.connect(self._add_aux_project)
        self.btn_remove_aux.clicked.connect(self._remove_aux_project)
        self.cb_augment.toggled.connect(self._toggle_aug_widgets)
        self.btn_sleap_refresh.clicked.connect(self._refresh_sleap_envs)
        self.btn_sleap_browse.clicked.connect(self._browse_sleap_out)
        self.btn_sleap_export.clicked.connect(self._export_sleap_labels)
        self.btn_sleap_open.clicked.connect(self._open_sleap)
        self.backend_combo.currentIndexChanged.connect(self._update_backend_ui)
        self.vitpose_checkpoint_combo.activated.connect(
            self._on_vitpose_checkpoint_activated
        )

        self._toggle_aug_widgets(self.cb_augment.isChecked())
        self._apply_settings()
        self._refresh_sleap_envs()
        self._apply_latest_weights_default()
        if not self.sleap_out_edit.text().strip():
            self.sleap_out_edit.setText(str(self._default_sleap_out_path()))
        self._update_backend_ui()

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select weights", "", "*.pt")
        if path:
            self.model_combo.setCurrentText(path)

    def _toggle_aug_widgets(self, enabled: bool):
        for w in self.aug_widgets:
            w.setEnabled(enabled)

    def _add_aux_project(self):
        start_dir = self._last_aux_path or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pose_project.json or dataset.yaml",
            start_dir,
            "pose_project.json;dataset.yaml;*.yaml;*.yml",
        )
        if not path:
            return
        self._last_aux_path = str(Path(path).parent)
        path_obj = Path(path)
        if path_obj.name == "pose_project.json":
            try:
                data = json.loads(path_obj.read_text(encoding="utf-8"))
                images_dir = Path(data["images_dir"]).expanduser().resolve()
                labels_dir = Path(data["labels_dir"]).expanduser().resolve()
                class_names = data.get("class_names", [])
                keypoint_names = data.get("keypoint_names", [])
            except Exception as e:
                QMessageBox.warning(self, "Invalid project", str(e))
                return

            if (
                class_names != self.project.class_names
                or keypoint_names != self.project.keypoint_names
            ):
                QMessageBox.warning(
                    self,
                    "Mismatch",
                    "Aux project classes/keypoints do not match the current project.",
                )
                return

            image_paths = list_images_in_dir(images_dir)
            if not image_paths:
                QMessageBox.warning(self, "Empty", "No images found in aux project.")
                return

            item = QListWidgetItem(f"{images_dir}  (labels: {labels_dir})")
            self.aux_list.addItem(item)
            self._aux_datasets.append(
                {"images": image_paths, "labels_dir": labels_dir, "project_path": path}
            )
        else:
            try:
                items, info = load_yolo_dataset_items(path_obj)
            except Exception as e:
                QMessageBox.warning(self, "Invalid dataset", str(e))
                return

            names = info.get("names") or {}
            kpt_shape = info.get("kpt_shape")
            kpt_names = info.get("kpt_names")
            if list(names.values()) != self.project.class_names:
                QMessageBox.warning(
                    self,
                    "Mismatch",
                    "YOLO dataset class names do not match the current project.",
                )
                return
            if not kpt_shape or not kpt_names:
                QMessageBox.warning(
                    self,
                    "Missing keypoints",
                    "YOLO dataset is missing keypoint metadata.",
                )
                return
            if list(kpt_names.values())[0] != self.project.keypoint_names:
                QMessageBox.warning(
                    self,
                    "Mismatch",
                    "YOLO dataset keypoint names do not match the current project.",
                )
                return
            if not items:
                QMessageBox.warning(self, "Empty", "No labeled items found in dataset.")
                return

            item = QListWidgetItem(f"{path_obj}  (YOLO dataset)")
            self.aux_list.addItem(item)
            self._aux_items.extend(items)
            self._aux_datasets.append({"project_path": str(path_obj), "items": items})

        self._refresh_labeled_count()

    def _apply_settings(self):
        settings = _load_dialog_settings("training_runner")
        if not settings:
            return
        self.backend_combo.setCurrentText(
            settings.get("backend", self.backend_combo.currentText())
        )
        self.model_combo.setCurrentText(
            settings.get("model", self.model_combo.currentText())
        )
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.epochs_spin.setValue(int(settings.get("epochs", self.epochs_spin.value())))
        self.imgsz_spin.setValue(int(settings.get("imgsz", self.imgsz_spin.value())))
        self.patience_spin.setValue(
            int(settings.get("patience", self.patience_spin.value()))
        )
        self.device_combo.setCurrentText(
            settings.get("device", self.device_combo.currentText())
        )
        self.cb_augment.setChecked(
            bool(settings.get("augment", self.cb_augment.isChecked()))
        )
        self.hsv_h_spin.setValue(float(settings.get("hsv_h", self.hsv_h_spin.value())))
        self.hsv_s_spin.setValue(float(settings.get("hsv_s", self.hsv_s_spin.value())))
        self.hsv_v_spin.setValue(float(settings.get("hsv_v", self.hsv_v_spin.value())))
        self.degrees_spin.setValue(
            float(settings.get("degrees", self.degrees_spin.value()))
        )
        self.translate_spin.setValue(
            float(settings.get("translate", self.translate_spin.value()))
        )
        self.scale_spin.setValue(float(settings.get("scale", self.scale_spin.value())))
        self.train_split_spin.setValue(
            float(settings.get("train_split", self.train_split_spin.value()))
        )
        self.seed_spin.setValue(int(settings.get("seed", self.seed_spin.value())))
        self.cb_ignore_occluded.setChecked(
            bool(settings.get("ignore_occluded", self.cb_ignore_occluded.isChecked()))
        )
        self._last_aux_path = settings.get("last_aux_path", self._last_aux_path)
        self.cb_sleap_include_aux.setChecked(
            bool(
                settings.get("sleap_include_aux", self.cb_sleap_include_aux.isChecked())
            )
        )
        self.cb_sleap_embed.setChecked(
            bool(settings.get("sleap_embed", self.cb_sleap_embed.isChecked()))
        )
        self._sleap_env_pref = settings.get("sleap_env_name", self._sleap_env_pref)
        sleap_out = settings.get("sleap_out_path", "").strip()
        if sleap_out:
            self.sleap_out_edit.setText(sleap_out)
        elif not self.sleap_out_edit.text().strip():
            self.sleap_out_edit.setText(str(self._default_sleap_out_path()))

    def _apply_latest_weights_default(self):
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.model_combo.setCurrentText(str(self.project.latest_pose_weights))

    def _update_backend_ui(self):
        backend = self.backend_combo.currentText()
        is_yolo = backend == "YOLO Pose"
        is_vitpose = backend == "ViTPose"
        is_sleap = backend.startswith("SLEAP")

        self.cfg_group.setVisible(is_yolo)
        self.aug_group.setVisible(is_yolo)
        self.info_group.setVisible(is_yolo or is_vitpose)
        self.progress.setVisible(is_yolo or is_vitpose)
        self.lbl_loss_plot.setVisible(is_yolo)
        self.loss_components_box.setVisible(is_yolo)
        self.log_view.setVisible(is_yolo or is_vitpose)
        self.btn_start.setVisible(is_yolo or is_vitpose)
        self.btn_stop.setVisible(is_yolo or is_vitpose)
        self.btn_open_eval.setVisible(is_yolo)

        self.vitpose_group.setVisible(is_vitpose)
        self.sleap_group.setVisible(is_sleap)

    def _on_vitpose_checkpoint_activated(self, index: int):
        if self.vitpose_checkpoint_combo.itemText(index) == "Browse…":
            self._resolve_vitpose_checkpoint()

    def _resolve_vitpose_checkpoint(self) -> str:
        text = self.vitpose_checkpoint_combo.currentText().strip()
        if text == "Browse…":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select checkpoint", "", "*.pth *.pt"
            )
            if path:
                self.vitpose_checkpoint_combo.setCurrentText(path)
                return path
            return ""
        return text

    def _default_sleap_out_path(self) -> Path:
        base = self.project.out_root / "posekit" / "sleap"
        base.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        proj_name = self.project.project_path.stem or "pose_project"
        return base / f"{proj_name}_{ts}.slp"

    def _refresh_sleap_envs(self):
        self.combo_sleap_env.clear()
        self.combo_sleap_env.setEnabled(True)
        envs, err = list_sleap_envs()
        if not envs:
            self.combo_sleap_env.addItem("No sleap envs found")
            self.combo_sleap_env.setEnabled(False)
            self.btn_sleap_open.setEnabled(False)
            self.btn_sleap_export.setEnabled(False)
            self.lbl_sleap_env_status.setText(
                err or "No conda envs starting with 'sleap' found."
            )
        else:
            self.combo_sleap_env.addItems(envs)
            if self._sleap_env_pref and self._sleap_env_pref in envs:
                self.combo_sleap_env.setCurrentText(self._sleap_env_pref)
            self.lbl_sleap_env_status.setText("")
            self.btn_sleap_open.setEnabled(True)
            self.btn_sleap_export.setEnabled(True)

    def _browse_sleap_out(self):
        start = self.sleap_out_edit.text().strip() or str(
            self._default_sleap_out_path()
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Select SLEAP output (.slp)", start, "*.slp"
        )
        if path:
            if not path.lower().endswith(".slp"):
                path = f"{path}.slp"
            self.sleap_out_edit.setText(path)

    def _get_sleap_env(self) -> Optional[str]:
        env = self.combo_sleap_env.currentText().strip()
        if not env or env.lower().startswith("no "):
            return None
        return env

    def _sleap_out_path(self) -> Optional[Path]:
        txt = self.sleap_out_edit.text().strip()
        if not txt:
            txt = str(self._default_sleap_out_path())
            self.sleap_out_edit.setText(txt)
        if not txt.lower().endswith(".slp"):
            txt = f"{txt}.slp"
            self.sleap_out_edit.setText(txt)
        return Path(txt).expanduser().resolve()

    def _export_sleap_labels(self, open_after: bool = False):
        env = self._get_sleap_env()
        if not env:
            QMessageBox.warning(
                self, "No SLEAP env", "Select a conda env starting with 'sleap'."
            )
            return
        slp_path = self._sleap_out_path()
        if slp_path is None:
            return

        self._sleap_open_after_export = bool(open_after)
        self.btn_sleap_export.setEnabled(False)
        self.btn_sleap_open.setEnabled(False)
        self._append_log(f"[SLEAP] Exporting to {slp_path} ...")

        self._sleap_thread = QThread()
        self._sleap_worker = SleapExportWorker(
            env_name=env,
            image_paths=self.image_paths,
            labels_dir=self.project.labels_dir,
            out_root=self.project.out_root,
            class_names=self.project.class_names,
            keypoint_names=self.project.keypoint_names,
            skeleton_edges=self.project.skeleton_edges,
            slp_path=slp_path,
            include_aux=self.cb_sleap_include_aux.isChecked(),
            embed=self.cb_sleap_embed.isChecked(),
            aux_datasets=[
                (d["images"], d["labels_dir"])
                for d in self._aux_datasets
                if "images" in d and "labels_dir" in d
            ],
            aux_items=list(self._aux_items),
        )
        self._sleap_worker.moveToThread(self._sleap_thread)
        self._sleap_thread.started.connect(self._sleap_worker.run)
        self._sleap_worker.log.connect(self._append_log)
        self._sleap_worker.finished.connect(self._on_sleap_export_finished)
        self._sleap_worker.failed.connect(self._on_sleap_export_failed)
        self._sleap_worker.finished.connect(self._sleap_thread.quit)
        self._sleap_worker.failed.connect(self._sleap_thread.quit)
        self._sleap_thread.finished.connect(self._sleap_thread.deleteLater)
        self._sleap_thread.start()

    def _on_sleap_export_finished(self, slp_path: str):
        self.btn_sleap_export.setEnabled(True)
        self.btn_sleap_open.setEnabled(True)
        self._append_log(f"[SLEAP] Exported: {slp_path}")
        self._set_latest_sleap_dataset(Path(slp_path))
        if self._sleap_open_after_export:
            self._sleap_open_after_export = False
            self._launch_sleap(Path(slp_path))

    def _on_sleap_export_failed(self, msg: str):
        self.btn_sleap_export.setEnabled(True)
        self.btn_sleap_open.setEnabled(True)
        self._append_log(f"[SLEAP] Export failed: {msg}")
        QMessageBox.warning(self, "SLEAP export failed", msg)

    def _set_latest_sleap_dataset(self, slp_path: Path):
        try:
            self.project.latest_sleap_dataset = slp_path
            self.project.project_path.write_text(
                json.dumps(self.project.to_json(), indent=2), encoding="utf-8"
            )
        except Exception as e:
            self._append_log(f"[SLEAP] Warning: failed to save latest dataset: {e}")

    def _open_sleap(self):
        slp_path = self._sleap_out_path()
        if slp_path is None:
            return
        if not slp_path.exists():
            self._export_sleap_labels(open_after=True)
            return
        self._launch_sleap(slp_path)

    def _launch_sleap(self, slp_path: Path):
        env = self._get_sleap_env()
        if not env:
            QMessageBox.warning(
                self, "No SLEAP env", "Select a conda env starting with 'sleap'."
            )
            return

        def _supports_cmd(args: List[str]) -> bool:
            try:
                res = run_conda(
                    ["conda", "run", "-n", env] + args,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return res.returncode == 0
            except Exception:
                return False

        if _supports_cmd(["sleap", "label", "--help"]):
            cmd = ["conda", "run", "-n", env, "sleap", "label", str(slp_path)]
        else:
            cmd = ["conda", "run", "-n", env, "sleap-label", str(slp_path)]

        try:
            popen_conda(cmd)
            self._append_log(f"[SLEAP] Launching: {' '.join(cmd)}")
        except Exception as e:
            QMessageBox.warning(self, "SLEAP launch failed", str(e))

    def _save_settings(self):
        _save_dialog_settings(
            "training_runner",
            {
                "backend": self.backend_combo.currentText(),
                "model": self.model_combo.currentText().strip(),
                "batch": int(self.batch_spin.value()),
                "auto_batch": bool(self.cb_auto_batch.isChecked()),
                "epochs": int(self.epochs_spin.value()),
                "patience": int(self.patience_spin.value()),
                "imgsz": int(self.imgsz_spin.value()),
                "device": self.device_combo.currentText(),
                "augment": bool(self.cb_augment.isChecked()),
                "hsv_h": float(self.hsv_h_spin.value()),
                "hsv_s": float(self.hsv_s_spin.value()),
                "hsv_v": float(self.hsv_v_spin.value()),
                "degrees": float(self.degrees_spin.value()),
                "translate": float(self.translate_spin.value()),
                "scale": float(self.scale_spin.value()),
                "train_split": float(self.train_split_spin.value()),
                "seed": int(self.seed_spin.value()),
                "ignore_occluded": bool(self.cb_ignore_occluded.isChecked()),
                "last_aux_path": self._last_aux_path,
                "sleap_env_name": self.combo_sleap_env.currentText().strip(),
                "sleap_out_path": self.sleap_out_edit.text().strip(),
                "sleap_embed": bool(self.cb_sleap_embed.isChecked()),
                "sleap_include_aux": bool(self.cb_sleap_include_aux.isChecked()),
            },
        )

    def closeEvent(self, event) -> None:
        """Persist dialog settings and cancel any running training or export worker before closing."""
        self._save_settings()
        if self._thread:
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait()
        super().closeEvent(event)

    def _remove_aux_project(self):
        row = self.aux_list.currentRow()
        if row < 0:
            return
        self.aux_list.takeItem(row)
        try:
            removed = self._aux_datasets.pop(row)
            if "items" in removed:
                items = set(removed.get("items") or [])
                self._aux_items = [i for i in self._aux_items if i not in items]
        except Exception:
            pass
        self._refresh_labeled_count()

    def _refresh_labeled_count(self):
        labeled_count = len(
            list_labeled_indices(self.image_paths, self.project.labels_dir)
        )
        labeled_count += len(self._aux_items)
        for aux in self._aux_datasets:
            if "images" in aux and "labels_dir" in aux:
                labeled_count += len(
                    list_labeled_indices(aux["images"], aux["labels_dir"])
                )
        self.lbl_labeled.setText(f"Labeled frames: {labeled_count}")

    def _start_training(self):
        backend = self.backend_combo.currentText()
        if backend == "ViTPose":
            self._start_vitpose_training()
            return
        if backend != "YOLO Pose":
            QMessageBox.information(
                self,
                "Not Implemented",
                "Only YOLO Pose and ViTPose are wired up right now. Other backends are coming soon.",
            )
            return

        labeled_count = len(
            list_labeled_indices(self.image_paths, self.project.labels_dir)
        )
        for aux in self._aux_datasets:
            labeled_count += len(list_labeled_indices(aux["images"], aux["labels_dir"]))
        if labeled_count < 2:
            QMessageBox.warning(
                self,
                "Not enough labels",
                "Need at least 2 labeled frames to train.",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = self.project.out_root / "runs" / "yolo_pose" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        data_dir = run_dir / "data"
        try:
            dataset_info = build_yolo_pose_dataset(
                self.image_paths,
                self.project.labels_dir,
                data_dir,
                self.train_split_spin.value(),
                self.seed_spin.value(),
                self.project.class_names,
                self.project.keypoint_names,
                extra_datasets=[
                    (d["images"], d["labels_dir"])
                    for d in self._aux_datasets
                    if "images" in d and "labels_dir" in d
                ],
                extra_items=list(self._aux_items),
                ignore_occluded_train=self.cb_ignore_occluded.isChecked(),
                ignore_occluded_val=False,
            )
        except Exception as e:
            QMessageBox.critical(self, "Dataset error", str(e))
            return

        config = {
            "backend": "yolo_pose",
            "model_weights": self.model_combo.currentText().strip(),
            "epochs": int(self.epochs_spin.value()),
            "patience": int(self.patience_spin.value()),
            "batch": int(self.batch_spin.value()),
            "auto_batch": bool(self.cb_auto_batch.isChecked()),
            "imgsz": int(self.imgsz_spin.value()),
            "device": self.device_combo.currentText(),
            "augment": bool(self.cb_augment.isChecked()),
            "ignore_occluded_train": bool(self.cb_ignore_occluded.isChecked()),
            "ignore_occluded_val": False,
            "hsv_h": float(self.hsv_h_spin.value()),
            "hsv_s": float(self.hsv_s_spin.value()),
            "hsv_v": float(self.hsv_v_spin.value()),
            "degrees": float(self.degrees_spin.value()),
            "translate": float(self.translate_spin.value()),
            "scale": float(self.scale_spin.value()),
            "dataset": {
                "yaml": str(dataset_info["yaml_path"]),
                "train": str(dataset_info["train_list"]),
                "val": str(dataset_info["val_list"]),
                "train_count": dataset_info["train_count"],
                "val_count": dataset_info["val_count"],
                "manifest": str(dataset_info.get("manifest", "")),
            },
            "aux_datasets": [d["project_path"] for d in self._aux_datasets],
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        self._last_run_dir = run_dir
        self._train_start_ts = time.time()
        self._loss_source_path = None
        self.lbl_run_dir.setText(f"Run dir: {run_dir}")
        self.log_view.clear()
        self.progress.setValue(0)
        self.lbl_loss_plot.clear()
        self.lbl_loss_plot.setText("Waiting for loss...")
        self._update_loss_plot()
        self._loss_timer.start()

        self._thread = QThread()
        self._worker = TrainingWorker(
            model_weights=self.model_combo.currentText().strip(),
            dataset_yaml=Path(dataset_info["yaml_path"]),
            run_dir=run_dir,
            epochs=self.epochs_spin.value(),
            patience=self.patience_spin.value(),
            batch=self.batch_spin.value(),
            imgsz=self.imgsz_spin.value(),
            device=self.device_combo.currentText(),
            augment=self.cb_augment.isChecked(),
            auto_batch=self.cb_auto_batch.isChecked(),
            hsv_h=self.hsv_h_spin.value(),
            hsv_s=self.hsv_s_spin.value(),
            hsv_v=self.hsv_v_spin.value(),
            degrees=self.degrees_spin.value(),
            translate=self.translate_spin.value(),
            scale=self.scale_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open_eval.setEnabled(False)

    def _start_vitpose_training(self):
        labeled_count = len(
            list_labeled_indices(self.image_paths, self.project.labels_dir)
        )
        if labeled_count < 2:
            QMessageBox.warning(
                self,
                "Not enough labels",
                "Need at least 2 labeled frames to train.",
            )
            return

        checkpoint = self._resolve_vitpose_checkpoint()
        if not checkpoint:
            QMessageBox.warning(
                self, "No checkpoint", "Select or browse for an init checkpoint."
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = get_training_runs_dir() / "vitpose" / timestamp
        cache_dir = get_models_dir()

        self._last_run_dir = run_dir
        self._train_start_ts = time.time()
        self._loss_source_path = None
        self.lbl_run_dir.setText(f"Run dir: {run_dir}")
        self.log_view.clear()
        self.progress.setValue(0)

        self._thread = QThread()
        self._worker = ViTPoseTrainingWorker(
            image_paths=self.image_paths,
            labels_dir=self.project.labels_dir,
            run_dir=run_dir,
            cache_dir=cache_dir,
            class_names=self.project.class_names,
            keypoint_names=self.project.keypoint_names,
            skeleton_edges=self.project.skeleton_edges,
            variant=self.vitpose_variant_combo.currentText(),
            init_checkpoint=checkpoint,
            num_keypoints=len(self.project.keypoint_names),
            epochs=self.epochs_spin.value(),
            batch=self.batch_spin.value(),
            device=self.device_combo.currentText(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open_eval.setEnabled(False)

    def _stop_training(self):
        if self._worker:
            self._worker.cancel()
            self._append_log("Stop requested. Waiting for trainer to stop...")
        self.btn_stop.setEnabled(False)
        if self._loss_timer.isActive():
            self._loss_timer.stop()

    def _on_progress(self, epoch: int, epochs: int):
        if epochs > 0:
            pct = int((epoch / epochs) * 100)
            self.progress.setValue(min(100, max(0, pct)))

    def _on_finished(self, info: dict):
        self._append_log("Training complete.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        run_dir_info = info.get("run_dir")
        if run_dir_info:
            try:
                self._last_run_dir = Path(run_dir_info)
            except Exception:
                pass
        if self._loss_timer.isActive():
            self._loss_timer.stop()
        self._update_loss_plot()
        weights = info.get("weights") or ""
        self._last_weights = weights if weights else None
        if self._last_run_dir:
            run_dir = Path(self._last_run_dir)
            try:
                # Find newest results
                candidates = list(run_dir.parent.rglob("results.csv"))
                if self._train_start_ts:
                    candidates = [
                        p
                        for p in candidates
                        if p.exists()
                        and p.stat().st_mtime >= (self._train_start_ts - 2)
                    ]
                if candidates:
                    newest = max(candidates, key=lambda p: p.stat().st_mtime)
                    run_dir = newest.parent
                    self._last_run_dir = run_dir
                    self.lbl_run_dir.setText(f"Run dir: {run_dir}")
            except Exception:
                pass
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            if best.exists():
                self._last_weights = str(best)
            elif last.exists():
                self._last_weights = str(last)
        if weights:
            self._append_log(f"Weights: {weights}")
            self.btn_open_eval.setEnabled(True)
            self._set_latest_weights(weights)
        elif self._last_weights:
            self._append_log(f"Weights: {self._last_weights}")
            self.btn_open_eval.setEnabled(True)
            self._set_latest_weights(self._last_weights)
        elif (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.btn_open_eval.setEnabled(True)
        if self._last_run_dir:
            run_dir = Path(self._last_run_dir)
            if list(run_dir.glob("**/events.out.tfevents.*")):
                self._append_log(f"TensorBoard: tensorboard --logdir {run_dir}")
            if (run_dir / "wandb").exists():
                self._append_log(
                    "W&B run detected in wandb/ (check your W&B dashboard)."
                )

    def _set_latest_weights(self, weights: str):
        try:
            w = Path(weights).resolve()
            if not w.is_file() or w.suffix != ".pt":
                self._append_log(f"Warning: ignoring non-.pt weights: {w}")
                return
            self.project.latest_pose_weights = w
            self.project.project_path.write_text(
                json.dumps(self.project.to_json(), indent=2), encoding="utf-8"
            )
        except Exception as e:
            self._append_log(f"Warning: failed to save latest weights: {e}")

    def _on_failed(self, msg: str):
        self._append_log(msg)
        QMessageBox.critical(self, "Training failed", msg)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._loss_timer.isActive():
            self._loss_timer.stop()

    def _open_eval(self):
        weights = self._last_weights
        if not weights and hasattr(self.project, "latest_pose_weights"):
            if (
                self.project.latest_pose_weights
                and Path(self.project.latest_pose_weights).exists()
            ):
                weights = str(self.project.latest_pose_weights)
        if not weights:
            QMessageBox.information(
                self, "Missing weights", "No valid weights found to evaluate."
            )
            return
        dlg = EvaluationDashboardDialog(
            self,
            self.project,
            self.image_paths,
            weights_path=weights,
        )
        dlg.exec()

    def _update_loss_plot(self):
        if not self._last_run_dir:
            return
        run_dir = Path(self._last_run_dir)
        results_path = run_dir / "results.csv"
        # Ultralytics results.csv location can vary
        if not results_path.exists():
            candidates = list(run_dir.rglob("results.csv"))
            if not candidates:
                candidates = list(run_dir.parent.rglob("results.csv"))
            if self._train_start_ts:
                candidates = [
                    p
                    for p in candidates
                    if p.exists() and p.stat().st_mtime >= (self._train_start_ts - 2)
                ]
            if candidates:
                results_path = max(candidates, key=lambda p: p.stat().st_mtime)
        if not results_path.exists():
            return

        if results_path != self._loss_source_path:
            self._loss_source_path = results_path
            self.log_view.appendPlainText(f"[loss] using {results_path}")
        try:
            # Read full csv
            lines = results_path.read_text(encoding="utf-8").splitlines()
            rows = list(csv.reader(lines))
            if len(rows) < 2:
                return
            header = [h.strip() for h in rows[0]]

            # Identify columns
            train_cols = [
                (i, h.replace("train/", ""))
                for i, h in enumerate(header)
                if ("train/" in h or h.startswith("box_loss")) and "loss" in h
            ]
            # Some versions use train/box_loss, others box_loss.
            # We want columns ending in loss basically.

            val_cols = [
                (i, h.replace("val/", ""))
                for i, h in enumerate(header)
                if ("val/" in h) and "loss" in h
            ]

            # Only graph explicitly loss columns
            if not train_cols and not val_cols:
                return

            keys = sorted({name for _, name in train_cols + val_cols})

            # Rebuild component checks if keys changed
            if not self.loss_component_checks or set(
                self.loss_component_checks.keys()
            ) != set(keys):
                for cb in self.loss_component_checks.values():
                    cb.deleteLater()
                self.loss_component_checks = {}
                # Clear layout first? The widgets are deleted, but layout item remains?
                # Using deleteLater should work.
                for name in keys:
                    cb = QCheckBox(name)
                    cb.setChecked(True)
                    cb.toggled.connect(self._update_loss_plot)
                    self.loss_components_layout.addWidget(cb)
                    self.loss_component_checks[name] = cb

            train_vals = {k: [] for k in keys}
            val_vals = {k: [] for k in keys}

            for row in rows[1:]:
                if not row:
                    continue

                # Helper to parse float
                def _getf(idx, _row=row):
                    if idx >= len(_row):
                        return None
                    s = _row[idx].strip()
                    if not s:
                        return None
                    try:
                        return float(s)
                    except ValueError:
                        return None

                for i, name in train_cols:
                    if name in keys:
                        train_vals[name].append(_getf(i))
                for i, name in val_cols:
                    if name in keys:
                        val_vals[name].append(_getf(i))

            # Filter by checkbox
            selected = [
                k for k, cb in self.loss_component_checks.items() if cb.isChecked()
            ]
            train_vals = {k: train_vals[k] for k in selected}
            val_vals = {k: val_vals[k] for k in selected}

            img = make_loss_plot_image(
                train_vals,
                val_vals,
                width=self.lbl_loss_plot.width(),
                height=self.lbl_loss_plot.height(),
            )
            pix = QPixmap.fromImage(img)
            self.lbl_loss_plot.setPixmap(pix)

        except Exception:
            pass
            pass
