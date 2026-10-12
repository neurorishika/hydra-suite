#!/usr/bin/env python3
"""
Active Learning dialog and worker.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from .utils import (
    _load_dialog_settings,
    _save_dialog_settings,
    format_float,
    get_available_devices,
    list_sleap_envs,
    make_pose_infer,
)

logger = logging.getLogger("pose_label.dialogs.active_learning")


class ActiveLearningWorker(QObject):
    """Worker for running active learning sampling."""

    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        strategy: str,
        image_paths: List[Path],
        candidate_indices: List[int],
        num_kpts: int,
        keypoint_names: List[str],
        out_root: Path,
        weights_a: Optional[str],
        weights_b: Optional[str],
        device: str,
        imgsz: int,
        conf: float,
        batch: int,
        eval_csv: Optional[str],
        preds_cache_a: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
        preds_cache_b: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
        keypoint_index: int = 0,
        backend: str = "yolo",
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
    ):
        super().__init__()
        self.strategy = strategy
        self.image_paths = image_paths
        self.candidate_indices = candidate_indices
        self.num_kpts = int(num_kpts)
        self.keypoint_names = list(keypoint_names)
        self.out_root = Path(out_root)
        self.weights_a = weights_a
        self.weights_b = weights_b
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.batch = int(batch)
        self.eval_csv = eval_csv
        self.preds_cache_a = preds_cache_a
        self.preds_cache_b = preds_cache_b
        self.keypoint_index = int(keypoint_index)
        self.backend = (backend or "yolo").lower()
        self.sleap_env = sleap_env
        self.sleap_device = sleap_device
        self.sleap_batch = int(sleap_batch)
        self._cancel = False

    def cancel(self):
        """Request early termination of the active-learning scoring run."""
        self._cancel = True

    def _predict_keypoints(self, weights_path: str, paths: List[Path]):
        infer = make_pose_infer(self.out_root, self.keypoint_names)
        preds_list, err = infer.predict(
            Path(weights_path),
            paths,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            batch=self.batch,
            progress_cb=self.progress.emit,
            backend=self.backend,
            sleap_env=self.sleap_env,
            sleap_device=self.sleap_device,
            sleap_batch=self.sleap_batch,
            sleap_max_instances=1,
        )
        if preds_list is None:
            self.failed.emit(err or "Prediction failed.")
            return None

        # Convert to arrays
        preds = {}
        total = len(paths)
        for i, p in enumerate(paths):
            pred = preds_list.get(str(p)) or preds_list.get(str(p.resolve()))
            if not pred:
                preds[str(p)] = (None, None)
            else:
                if len(pred) != self.num_kpts:
                    self.failed.emit(
                        "Prediction keypoint count mismatch. "
                        f"Model has {len(pred)} keypoints, project expects {self.num_kpts}. "
                        "Please select a matching model."
                    )
                    return None
                pred_xy = np.array([[x, y] for x, y, _ in pred], dtype=np.float32)
                pred_conf = np.array([c for _, _, c in pred], dtype=np.float32)
                preds[str(p)] = (pred_xy, pred_conf)
            self.progress.emit(i + 1, total)
        return preds

    def _run_eval_error_strategy(self, paths: List[Path]) -> None:
        """Score candidate frames by evaluation CSV error (highest first)."""
        if not self.eval_csv or not Path(self.eval_csv).exists():
            self.failed.emit("Eval CSV not found.")
            return
        path_set = {str(p.resolve()) for p in paths}
        path_set.update({str(p) for p in paths})

        rows: list = []
        with Path(self.eval_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        scores: list = []
        for row in rows:
            raw = row.get("image_path", "")
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = (Path(self.eval_csv).parent / p).resolve()
            else:
                p = p.resolve()

            if str(p) in path_set:
                path_str = str(p)
            elif raw in path_set:
                path_str = raw
            else:
                continue

            score = float(row.get("mean_error_norm", 0.0) or 0.0)
            scores.append((path_str, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        self.finished.emit(scores)

    def _load_predictions(self, paths: List[Path]):
        """Load or compute predictions for model A (and B for disagreement).

        Returns ``(preds_a, preds_b)`` or ``None`` on failure (error already emitted).
        """
        preds_a = None
        if self.strategy in ("lowest_conf", "lowest_conf_kpt", "disagreement"):
            if not self.preds_cache_a:
                if not self.weights_a:
                    self.failed.emit("Weights not found.")
                    return None
                preds_a = self._predict_keypoints(self.weights_a, paths)
                if preds_a is None:
                    return None

        preds_b = None
        if self.strategy == "disagreement":
            if not self.preds_cache_b:
                if not self.weights_b:
                    self.failed.emit("Weights B not found.")
                    return None
                preds_b = self._predict_keypoints(self.weights_b, paths)
                if preds_b is None:
                    return None

        return preds_a, preds_b

    @staticmethod
    def _get_pred(target_path, cache, computed_dict):
        """Retrieve prediction arrays for a single path from cache or dict."""
        if computed_dict:
            return computed_dict.get(str(target_path))
        if cache:
            val = cache.get(str(target_path)) or cache.get(str(target_path.resolve()))
            if not val:
                return None
            xy = np.array([[x, y] for x, y, _ in val], dtype=np.float32)
            conf = np.array([c for _, _, c in val], dtype=np.float32)
            return (xy, conf)
        return None

    def _score_lowest_conf(self, p, preds_a):
        ret = self._get_pred(p, self.preds_cache_a, preds_a)
        if ret and ret[1] is not None:
            return float(np.mean(ret[1]))
        return 0.0

    def _score_lowest_conf_kpt(self, p, preds_a):
        ret = self._get_pred(p, self.preds_cache_a, preds_a)
        ki = self.keypoint_index
        if ret and ret[1] is not None and 0 <= ki < len(ret[1]):
            return float(ret[1][ki])
        return 0.0

    def _score_disagreement(self, p, preds_a, preds_b):
        ret_a = self._get_pred(p, self.preds_cache_a, preds_a)
        ret_b = self._get_pred(p, self.preds_cache_b, preds_b)
        if ret_a and ret_b and ret_a[0] is not None and ret_b[0] is not None:
            diff = ret_a[0] - ret_b[0]
            dists = np.linalg.norm(diff, axis=1)
            return float(np.mean(dists)) if dists.size else 0.0
        # High priority when one model fails to predict.
        return 1e9

    def _run_prediction_strategy(self, paths: List[Path]) -> None:
        """Score candidates using prediction-based strategies."""
        result = self._load_predictions(paths)
        if result is None:
            return
        preds_a, preds_b = result

        scores = []
        for p in paths:
            if self.strategy == "lowest_conf":
                score = self._score_lowest_conf(p, preds_a)
            elif self.strategy == "lowest_conf_kpt":
                score = self._score_lowest_conf_kpt(p, preds_a)
            elif self.strategy == "disagreement":
                score = self._score_disagreement(p, preds_a, preds_b)
            else:
                score = 0.0
            scores.append((str(p), score))

        if self.strategy == "disagreement":
            scores.sort(key=lambda x: x[1], reverse=True)
        else:
            scores.sort(key=lambda x: x[1])  # Low confidence first

        self.finished.emit(scores)

    def run(self):
        """Score candidate frames using the selected active-learning strategy and emit ranked results."""
        try:
            paths = [self.image_paths[i] for i in self.candidate_indices]
            if not paths:
                self.failed.emit("No candidate frames.")
                return

            if self.strategy == "eval_error":
                self._run_eval_error_strategy(paths)
            else:
                self._run_prediction_strategy(paths)

        except Exception as e:
            self.failed.emit(str(e))
            logger.error(f"Active learning failed: {e}", exc_info=True)


class ActiveLearningDialog(QDialog):
    """Dialog for active learning sampling."""

    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        is_labeled_fn,
        labeling_indices,
        add_frames_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Active Learning Sampler")
        self.setMinimumSize(QSize(860, 650))

        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn
        self.labeling_indices = set(labeling_indices)
        self.add_frames_callback = add_frames_callback
        self._lock_model = False
        self._thread = None
        self._worker = None
        self._path_to_index = {}
        for i, p in enumerate(image_paths):
            try:
                self._path_to_index[str(p)] = i
                self._path_to_index[str(p.resolve())] = i
            except Exception:
                self._path_to_index[str(p)] = i
        self.infer = make_pose_infer(self.project.out_root, self.project.keypoint_names)

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        # Strategy
        strat_group = QGroupBox("Strategy")
        strat_layout = QFormLayout(strat_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(
            [
                "Lowest keypoint confidence (mean)",
                "Lowest confidence for keypoint",
                "Largest model disagreement",
                "Largest train/val error",
            ]
        )
        strat_layout.addRow("Strategy", self.strategy_combo)
        content_layout.addWidget(strat_group)

        # Common config
        common_group = QGroupBox("Selection")
        common_layout = QFormLayout(common_group)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["YOLO", "SLEAP"])
        common_layout.addRow("Backend", self.backend_combo)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Unlabeled only", "All frames", "Labeling set"])
        common_layout.addRow("Scope", self.scope_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 5000)
        self.n_spin.setValue(50)
        common_layout.addRow("Suggested frame count", self.n_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        common_layout.addRow("Device", self.device_combo)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(0, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setSpecialValueText("Model's training size")
        self.imgsz_spin.setValue(0)
        self.imgsz_spin.setToolTip(
            "YOLO prediction size. The default uses the size the model was "
            "trained at; predicting at another size lowers accuracy."
        )
        common_layout.addRow("Image size", self.imgsz_spin)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        common_layout.addRow("Confidence threshold", self.conf_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        common_layout.addRow("Batch size", self.batch_spin)

        self.cb_use_cache = QCheckBox("Use cached predictions when available")
        self.cb_use_cache.setChecked(True)
        common_layout.addRow("", self.cb_use_cache)

        self.sleap_env_row = QWidget()
        sleap_env_layout = QHBoxLayout(self.sleap_env_row)
        sleap_env_layout.setContentsMargins(0, 0, 0, 0)
        self.sleap_env_combo = QComboBox()
        self.sleap_env_combo.setToolTip("Environment name must start with 'sleap'.")
        self.btn_sleap_refresh = QPushButton("↻")
        self.btn_sleap_refresh.setMaximumWidth(40)
        self.btn_sleap_refresh.setToolTip("Refresh conda environments list")
        sleap_env_layout.addWidget(self.sleap_env_combo, 1)
        sleap_env_layout.addWidget(self.btn_sleap_refresh)
        common_layout.addRow("SLEAP environment", self.sleap_env_row)
        self.lbl_sleap_env_status = QLabel("")
        self.lbl_sleap_env_status.setStyleSheet("QLabel { color: #b00; }")
        common_layout.addRow("", self.lbl_sleap_env_status)

        content_layout.addWidget(common_group)

        # Strategy-specific controls
        self.stack = QStackedWidget()

        # Lowest confidence
        w1 = QWidget()
        w1_layout = QFormLayout(w1)
        self.weights_a_edit = QLineEdit("")
        self.btn_weights_a = QPushButton("Browse…")
        row_a = QHBoxLayout()
        row_a.addWidget(self.weights_a_edit, 1)
        row_a.addWidget(self.btn_weights_a)
        self.weights_a_label = QLabel("Weights")
        w1_layout.addRow(self.weights_a_label, row_a)
        self.kpt_combo = QComboBox()
        self.kpt_combo.addItems(self.project.keypoint_names)
        w1_layout.addRow("Keypoint", self.kpt_combo)

        # Disagreement
        w2 = QWidget()
        w2_layout = QFormLayout(w2)
        self.weights_b1_edit = QLineEdit("")
        self.btn_weights_b1 = QPushButton("Browse…")
        row_b1 = QHBoxLayout()
        row_b1.addWidget(self.weights_b1_edit, 1)
        row_b1.addWidget(self.btn_weights_b1)
        self.weights_b1_label = QLabel("Weights A")
        w2_layout.addRow(self.weights_b1_label, row_b1)

        self.weights_b2_edit = QLineEdit("")
        self.btn_weights_b2 = QPushButton("Browse…")
        row_b2 = QHBoxLayout()
        row_b2.addWidget(self.weights_b2_edit, 1)
        row_b2.addWidget(self.btn_weights_b2)
        self.weights_b2_label = QLabel("Weights B")
        w2_layout.addRow(self.weights_b2_label, row_b2)

        # Eval error
        w3 = QWidget()
        w3_layout = QFormLayout(w3)
        self.eval_csv_edit = QLineEdit("")
        self.btn_eval_csv = QPushButton("Browse…")
        row_c = QHBoxLayout()
        row_c.addWidget(self.eval_csv_edit, 1)
        row_c.addWidget(self.btn_eval_csv)
        w3_layout.addRow("Evaluation CSV", row_c)

        self.stack.addWidget(w1)
        self.stack.addWidget(w2)
        self.stack.addWidget(w3)
        content_layout.addWidget(self.stack)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        content_layout.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        content_layout.addWidget(self.log_view, 1)

        # Results
        self.results_list = QListWidget()
        content_layout.addWidget(QLabel("Suggested frames"))
        content_layout.addWidget(self.results_list, 2)

        # Buttons
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Suggest Frames")
        self.btn_add = QPushButton("Add Selected to Labeling")
        self.btn_add.setEnabled(False)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_add)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        content_layout.addLayout(btns)

        # Wiring
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self.btn_weights_a.clicked.connect(
            lambda: self._browse_weights(self.weights_a_edit)
        )
        self.btn_weights_b1.clicked.connect(
            lambda: self._browse_weights(self.weights_b1_edit)
        )
        self.btn_weights_b2.clicked.connect(
            lambda: self._browse_weights(self.weights_b2_edit)
        )
        self.btn_eval_csv.clicked.connect(self._browse_eval_csv)
        self.btn_run.clicked.connect(self._run)
        self.btn_add.clicked.connect(self._add_selected)
        self.btn_close.clicked.connect(self.reject)
        self.btn_sleap_refresh.clicked.connect(self._refresh_sleap_envs)

        self._on_strategy_changed(0)
        self._apply_settings()
        self._apply_latest_weights_default()
        self._refresh_sleap_envs()
        self._apply_backend_ui()

    def lock_model_path(self: object, path: str) -> object:
        """Pre-fill and lock all model-path fields so the user cannot change them."""
        if path:
            self.weights_a_edit.setText(path)
            self.weights_b1_edit.setText(path)
            self.weights_b2_edit.setText(path)
        self._lock_model = True
        self.weights_a_edit.setReadOnly(True)
        self.weights_b1_edit.setReadOnly(True)
        self.weights_b2_edit.setReadOnly(True)
        self.btn_weights_a.setEnabled(False)
        self.btn_weights_b1.setEnabled(False)
        self.btn_weights_b2.setEnabled(False)
        self._apply_backend_ui()

    def _apply_settings(self):
        settings = _load_dialog_settings("active_learning")
        if not settings:
            return
        self.backend_combo.setCurrentText(
            settings.get("backend", self.backend_combo.currentText())
        )
        self.strategy_combo.setCurrentIndex(int(settings.get("strategy_index", 0)))
        self.scope_combo.setCurrentText(
            settings.get("scope", self.scope_combo.currentText())
        )
        self.n_spin.setValue(int(settings.get("n", self.n_spin.value())))
        self.device_combo.setCurrentText(
            settings.get("device", self.device_combo.currentText())
        )
        self.imgsz_spin.setValue(
            int(settings.get("pred_imgsz", self.imgsz_spin.value()))
        )
        self.conf_spin.setValue(float(settings.get("conf", self.conf_spin.value())))
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.weights_a_edit.setText(
            settings.get("weights_a", self.weights_a_edit.text())
        )
        self.weights_b1_edit.setText(
            settings.get("weights_b1", self.weights_b1_edit.text())
        )
        self.weights_b2_edit.setText(
            settings.get("weights_b2", self.weights_b2_edit.text())
        )
        self.eval_csv_edit.setText(settings.get("eval_csv", self.eval_csv_edit.text()))
        if "kpt_index" in settings:
            self.kpt_combo.setCurrentIndex(int(settings.get("kpt_index", 0)))
        if "sleap_env" in settings:
            self.sleap_env_combo.setCurrentText(
                settings.get("sleap_env", self.sleap_env_combo.currentText())
            )

    def _apply_latest_weights_default(self):
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            if self.backend_combo.currentText().strip().lower() != "sleap":
                latest = str(self.project.latest_pose_weights)
                if not self.weights_a_edit.text().strip():
                    self.weights_a_edit.setText(latest)
                if not self.weights_b1_edit.text().strip():
                    self.weights_b1_edit.setText(latest)

    def _save_settings(self):
        _save_dialog_settings(
            "active_learning",
            {
                "backend": self.backend_combo.currentText(),
                "strategy_index": int(self.strategy_combo.currentIndex()),
                "scope": self.scope_combo.currentText(),
                "n": int(self.n_spin.value()),
                "device": self.device_combo.currentText(),
                "pred_imgsz": int(self.imgsz_spin.value()),
                "conf": float(self.conf_spin.value()),
                "batch": int(self.batch_spin.value()),
                "weights_a": self.weights_a_edit.text().strip(),
                "weights_b1": self.weights_b1_edit.text().strip(),
                "weights_b2": self.weights_b2_edit.text().strip(),
                "eval_csv": self.eval_csv_edit.text().strip(),
                "kpt_index": int(self.kpt_combo.currentIndex()),
                "sleap_env": self.sleap_env_combo.currentText().strip(),
            },
        )

    def closeEvent(self, event) -> None:
        """Persist dialog settings to disk before the window closes."""
        self._save_settings()
        super().closeEvent(event)

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _on_backend_changed(self, _text: str):
        self._apply_backend_ui()
        if self.backend_combo.currentText().strip().lower() == "sleap":
            self._refresh_sleap_envs()

    def _apply_backend_ui(self):
        is_sleap = self.backend_combo.currentText().strip().lower() == "sleap"
        self.sleap_env_row.setVisible(is_sleap)
        self.lbl_sleap_env_status.setVisible(is_sleap)
        label = "Model dir:" if is_sleap else "Weights:"
        self.weights_a_label.setText(label)
        self.weights_b1_label.setText("Model dir A:" if is_sleap else "Weights A:")
        self.weights_b2_label.setText("Model dir B:" if is_sleap else "Weights B:")
        self.imgsz_spin.setEnabled(not is_sleap)
        self.conf_spin.setEnabled(not is_sleap)
        if self._lock_model:
            for w in (
                self.weights_a_label,
                self.weights_b1_label,
                self.weights_b2_label,
                self.weights_a_edit,
                self.weights_b1_edit,
                self.weights_b2_edit,
                self.btn_weights_a,
                self.btn_weights_b1,
                self.btn_weights_b2,
            ):
                w.setVisible(False)

    def _refresh_sleap_envs(self):
        current = self.sleap_env_combo.currentText().strip()
        self.sleap_env_combo.clear()
        self.sleap_env_combo.setEnabled(True)
        envs, err = list_sleap_envs()
        if not envs:
            self.sleap_env_combo.addItem("No sleap envs found")
            self.sleap_env_combo.setEnabled(False)
            self.lbl_sleap_env_status.setText(
                err or "No conda envs starting with 'sleap' found."
            )
        else:
            self.sleap_env_combo.addItems(envs)
            if current and current in envs:
                self.sleap_env_combo.setCurrentText(current)
            self.lbl_sleap_env_status.setText("")

    def _get_sleap_env(self) -> Optional[str]:
        env = self.sleap_env_combo.currentText().strip()
        if not env or env.lower().startswith("no sleap"):
            return None
        return env

    def _browse_weights(self, line_edit: QLineEdit):
        if self._lock_model:
            return
        if self.backend_combo.currentText().strip().lower() == "sleap":
            path = QFileDialog.getExistingDirectory(
                self, "Select SLEAP model directory"
            )
            if path:
                line_edit.setText(path)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select weights", "", "*.pt")
            if path:
                line_edit.setText(path)

    def _browse_eval_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select eval CSV", "", "*.csv")
        if path:
            self.eval_csv_edit.setText(path)

    def _on_strategy_changed(self, idx: int):
        if idx <= 1:
            self.stack.setCurrentIndex(0)
        elif idx == 2:
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(2)
        self.kpt_combo.setEnabled(idx == 1)

    def _collect_candidates(self) -> List[int]:
        scope = self.scope_combo.currentText()
        indices = []
        for i, p in enumerate(self.image_paths):
            if scope == "Unlabeled only" and self.is_labeled_fn(p):
                continue
            if scope == "Labeling set" and i not in self.labeling_indices:
                continue
            indices.append(i)
        return indices

    def _run(self):
        candidates = self._collect_candidates()
        if not candidates:
            QMessageBox.warning(self, "No candidates", "No frames match the scope.")
            return

        self.results_list.clear()
        self.log_view.clear()
        self.progress.setValue(0)

        strat_idx = self.strategy_combo.currentIndex()
        if strat_idx == 0:
            strategy = "lowest_conf"
            weights_a = self.weights_a_edit.text().strip()
            weights_b = None
            eval_csv = None
        elif strat_idx == 1:
            strategy = "lowest_conf_kpt"
            weights_a = self.weights_a_edit.text().strip()
            weights_b = None
            eval_csv = None
        elif strat_idx == 2:
            strategy = "disagreement"
            weights_a = self.weights_b1_edit.text().strip()
            weights_b = self.weights_b2_edit.text().strip()
            eval_csv = None
        else:
            strategy = "eval_error"
            weights_a = None
            weights_b = None
            eval_csv = self.eval_csv_edit.text().strip()

        backend = self.backend_combo.currentText().strip().lower()
        needs_infer = strategy in {"lowest_conf", "lowest_conf_kpt", "disagreement"}
        if needs_infer:
            if not weights_a:
                QMessageBox.warning(
                    self,
                    "Missing model",
                    (
                        "Please select SLEAP model directory."
                        if backend == "sleap"
                        else "Please select model weights."
                    ),
                )
                return
            if strategy == "disagreement" and not weights_b:
                QMessageBox.warning(
                    self,
                    "Missing model",
                    (
                        "Please select SLEAP model directory B."
                        if backend == "sleap"
                        else "Please select model weights B."
                    ),
                )
                return
        if backend == "sleap" and needs_infer:
            sleap_env = self._get_sleap_env()
            if not sleap_env:
                QMessageBox.warning(
                    self,
                    "No SLEAP env",
                    "Select a conda env starting with 'sleap' for SLEAP inference.",
                )
                return
            if weights_a and not Path(weights_a).is_dir():
                QMessageBox.warning(
                    self, "Missing model", "SLEAP model directory A not found."
                )
                return
            if weights_b and not Path(weights_b).is_dir():
                QMessageBox.warning(
                    self, "Missing model", "SLEAP model directory B not found."
                )
                return

        preds_cache_a = None
        preds_cache_b = None
        if self.cb_use_cache.isChecked():
            if weights_a:
                cache_a = self.infer.get_cache_for_paths(
                    Path(weights_a),
                    [self.image_paths[i] for i in candidates],
                    backend=backend,
                )
                if cache_a is not None:
                    preds_cache_a = cache_a
            if weights_b:
                cache_b = self.infer.get_cache_for_paths(
                    Path(weights_b),
                    [self.image_paths[i] for i in candidates],
                    backend=backend,
                )
                if cache_b is not None:
                    preds_cache_b = cache_b

        self._thread = QThread()
        self._worker = ActiveLearningWorker(
            strategy=strategy,
            image_paths=self.image_paths,
            candidate_indices=candidates,
            num_kpts=len(self.project.keypoint_names),
            keypoint_names=self.project.keypoint_names,
            out_root=self.project.out_root,
            weights_a=weights_a,
            weights_b=weights_b,
            device=self.device_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            conf=self.conf_spin.value(),
            batch=self.batch_spin.value(),
            eval_csv=eval_csv,
            preds_cache_a=preds_cache_a,
            preds_cache_b=preds_cache_b,
            keypoint_index=self.kpt_combo.currentIndex(),
            backend=backend,
            sleap_env=self._get_sleap_env() if backend == "sleap" else None,
            sleap_device=self.device_combo.currentText(),
            sleap_batch=self.batch_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            pct = int((done / total) * 100)
            self.progress.setValue(min(100, max(0, pct)))

    def _on_finished(self, scores: List[Tuple[str, float]]):
        n = min(self.n_spin.value(), len(scores))
        for path, score in scores[:n]:
            try:
                resolved = str(Path(path).resolve())
            except Exception:
                resolved = str(path)
            name = Path(path).name
            item = QListWidgetItem(f"{name} | score={format_float(score)}")
            item.setData(Qt.UserRole, resolved)
            self.results_list.addItem(item)
        self.btn_add.setEnabled(True if n > 0 else False)
        self._append_log("Suggestions ready.")

    def _on_failed(self, msg: str):
        self._append_log(msg)
        QMessageBox.critical(self, "Active learning failed", msg)

    def _add_selected(self):
        selected = self.results_list.selectedItems()
        if not selected:
            selected = [
                self.results_list.item(i) for i in range(self.results_list.count())
            ]
        if not selected:
            return
        indices = []
        for item in selected:
            path = item.data(Qt.UserRole)
            if path in self._path_to_index:
                indices.append(self._path_to_index[path])
                continue
            try:
                resolved = str(Path(path).resolve())
            except Exception:
                resolved = None
            if resolved and resolved in self._path_to_index:
                indices.append(self._path_to_index[resolved])
        if not indices:
            return
        if self.add_frames_callback:
            self.add_frames_callback(indices, "Active learning")
            self.accept()
            return
        if hasattr(self.parent(), "_add_indices_to_labeling"):
            self.parent()._add_indices_to_labeling(indices, "Active learning")
        self.accept()
        self.accept()
