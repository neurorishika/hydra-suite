"""PoseKit main window: coordinates canvas, panels, and annotation state."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.posekit.config.schemas import PoseKitConfig
from hydra_suite.utils.conda_utils import run_conda
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from .canvas import FrameListDelegate, PoseCanvas

# Local refactored modules
from .constants import DEFAULT_AUTOSAVE_DELAY_MS, DEFAULT_PROJECT_NAME
from .dialogs.project_wizard import ProjectWizard
from .dialogs.skeleton import SkeletonEditorDialog
from .dialogs.source_manager import SourceManagerDialog
from .io import load_yolo_pose_label, migrate_labels_keypoints, save_yolo_pose_label
from .models import DataSource, FrameAnn, Keypoint, Project, compute_bbox_from_kpts
from .panels import PoseSourceBrowserPanel
from .project import (
    build_image_list,
    create_standalone_project_via_wizard,
    find_project,
    load_project_with_repairs,
    open_project_from_path,
    remove_source_from_project,
    save_project_state,
)
from .runtimes import (
    CANONICAL_RUNTIMES,
    available_tiers,
    canonical_runtime_to_tier,
    derive_pose_runtime_settings,
    detect_platform,
    infer_compute_runtime_from_legacy,
    tier_label,
    tier_to_canonical_runtime,
)
from .utils import (
    enhance_for_pose,
    get_default_skeleton_dir,
    list_images,
    load_ui_settings,
    save_ui_settings,
)
from .workers import BulkPosePredictWorker, PosePredictWorker, SleapServiceWorker

logger = logging.getLogger("pose_label")

# External imports (formerly in try/except blocks in main.py)
try:
    from hydra_suite.integrations.sleap.service import PoseInferenceService
    from hydra_suite.posekit.core.extensions import (
        IncrementalEmbeddingCache,
        MetadataManager,
        build_yolo_pose_dataset,
        cluster_stratified_split,
    )
    from hydra_suite.posekit.gui.dialogs.active_learning import ActiveLearningDialog
    from hydra_suite.posekit.gui.dialogs.evaluation import EvaluationDashboardDialog
    from hydra_suite.posekit.gui.dialogs.exploration import SmartSelectDialog
    from hydra_suite.posekit.gui.dialogs.training import TrainingRunnerDialog
except ImportError:
    # Try alternate path if running standalone script?
    # Assuming standard package structure for now
    pass


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QMainWindow):
    """MainWindow API surface documentation."""

    def __init__(
        self,
        project: Project,
        image_paths: List[Path],
        *,
        show_welcome_when_empty: bool = True,
    ) -> None:
        super().__init__()
        self.config = PoseKitConfig()
        self.setWindowTitle("PoseKit")
        self.apply_stylesheet()
        self._show_welcome_when_empty = bool(show_welcome_when_empty)

        self.project = project
        self.image_paths = image_paths
        self.current_index = max(0, min(project.last_index, len(image_paths) - 1))
        self.current_kpt = 0
        self.mode = "frame"  # frame | keypoint
        self._img_bgr = None
        self._img_display = None
        self._img_wh = (1, 1)
        self._ann: Optional[FrameAnn] = None
        self._dirty = False
        self._undo_stack: List[List[Keypoint]] = []
        self._undo_max = 50
        self._frame_cache: Dict[int, FrameAnn] = {}
        # Multi-source routing: populated by _build_source_index()
        self._source_map: Dict[str, DataSource] = {}  # source_id -> DataSource
        self._image_source: Dict[str, str] = {}  # str(img_path) -> source_id
        self._suppress_list_rebuild = False
        self._pred_thread: Optional[QThread] = None
        self._pred_worker: Optional[PosePredictWorker] = None
        self._bulk_pred_thread: Optional[QThread] = None
        self._bulk_pred_worker: Optional[BulkPosePredictWorker] = None
        self._bulk_prediction_locked = False
        self._sleap_service_thread: Optional[QThread] = None
        self._sleap_service_worker: Optional[SleapServiceWorker] = None
        self.show_predictions = True
        self.show_pred_conf = False
        self._sleap_env_pref = ""
        self._pred_conf_cache_key: Optional[str] = None
        self._pred_conf_map: Dict[str, Optional[float]] = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map: Dict[str, Optional[int]] = {}
        self._list_items: Dict[int, QListWidgetItem] = {}
        self._cluster_ids_cache: Optional[List[int]] = None
        self._cluster_ids_mtime: Optional[float] = None
        self.infer = PoseInferenceService(
            self.project.out_root,
            self.project.keypoint_names,
            self.project.skeleton_edges,
        )

        # Track which frames are in the labeling set (empty by default)
        self.labeling_frames: set = set()
        if getattr(project, "labeling_frames", None):
            self.labeling_frames = {
                int(i)
                for i in project.labeling_frames
                if 0 <= int(i) < len(image_paths)
            }
        self.metadata_manager = MetadataManager(
            self.project.out_root / "posekit" / "metadata.json"
        )
        self.autosave_delay_ms = DEFAULT_AUTOSAVE_DELAY_MS
        self._rebuild_path_index()
        self._suppress_source_combo_events = False
        self.current_source_id = self._resolve_initial_source_id()

        splitter = QSplitter(Qt.Horizontal)
        self._content_stack = QStackedWidget(self)
        self.setCentralWidget(self._content_stack)
        self._content_stack.addWidget(self._make_welcome_page())  # index 0
        self._content_stack.addWidget(splitter)  # index 1

        left = PoseSourceBrowserPanel(self)
        self.left_panel = left
        self.labeling_list = left.labeling_list
        self.labeling_list.setDragDropMode(QListWidget.DragDrop)
        self.labeling_list.setDefaultDropAction(Qt.MoveAction)
        self.labeling_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.labeling_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.labeling_list.setTextElideMode(Qt.ElideRight)
        self.labeling_list.setItemDelegate(FrameListDelegate(self.labeling_list))
        self.frame_list = left.frame_list
        self.frame_list.setDragDropMode(QListWidget.DragDrop)
        self.frame_list.setDefaultDropAction(Qt.MoveAction)
        self.frame_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.frame_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_list.setTextElideMode(Qt.ElideRight)
        self.frame_list.setItemDelegate(FrameListDelegate(self.frame_list))
        self.search_edit = left.search_edit
        self.sort_combo = left.sort_combo
        self.source_combo = left.source_combo
        self.source_summary = left.source_summary
        self.btn_manage_sources_inline = left.btn_manage_sources
        self.btn_unlabeled_to_labeling = left.btn_unlabeled_to_labeling
        self.btn_unlabeled_to_all = left.btn_unlabeled_to_all
        self.btn_random_to_labeling = left.btn_random_to_labeling
        self.spin_random_count = left.spin_random_count
        self.btn_smart_select = left.btn_smart_select
        self.btn_delete_frames = left.btn_delete_frames

        # Canvas
        # Load UI settings - will be applied after widgets are created
        self.canvas = PoseCanvas(parent=self)
        self.canvas.set_callbacks(
            self.on_place_kpt, self.on_move_kpt, self.on_select_kpt
        )
        self.canvas.set_kpt_radius(self.project.kpt_radius)
        self.canvas.set_label_font_size(self.project.label_font_size)
        self.canvas.set_kpt_opacity(self.project.kpt_opacity)
        self.canvas.set_edge_opacity(self.project.edge_opacity)
        self.canvas.set_edge_width(self.project.edge_width)
        self.canvas_hint = QLabel(
            "Left click: place/move  •  Right click: toggle vis / place occluded  •  "
            "Wheel: zoom  •  A/D: prev/next  •  Q/E: prev/next keypoint  •  "
            "Space: advance  •  Ctrl+S: save"
        )
        self.canvas_hint.setWordWrap(True)
        self.canvas_hint.setAlignment(Qt.AlignCenter)
        self.canvas_hint.setStyleSheet(
            "QLabel { color: #6a6a6a; padding: 6px; font-size: 11px; font-style: italic; }"
        )

        self._setting_meta = False
        self.meta_tags_label = QLabel("Frame tags")
        self.meta_tags = {}
        tags_row = QHBoxLayout()
        for tag in [
            "occluded",
            "weird_posture",
            "motion_blur",
            "poor_lighting",
            "partial_view",
            "unclear",
        ]:
            cb = QCheckBox(tag)
            cb.toggled.connect(self._on_meta_changed)
            self.meta_tags[tag] = cb
            tags_row.addWidget(cb)
        tags_row.addStretch(1)

        self.meta_notes = QLineEdit()
        self.meta_notes.setPlaceholderText("Notes for this frame…")
        self.meta_notes.textEdited.connect(self._on_meta_changed)

        meta_box = QWidget()
        meta_layout = QVBoxLayout(meta_box)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(4)
        meta_layout.addWidget(self.meta_tags_label)
        meta_layout.addLayout(tags_row)
        meta_layout.addWidget(self.meta_notes)
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(4)
        canvas_layout.addWidget(self.canvas, 1)
        canvas_layout.addWidget(self.canvas_hint, 0)
        canvas_layout.addWidget(meta_box, 0)

        # Tools
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        # Annotation
        ann_group = QGroupBox("Labeling")
        ann_layout = QVBoxLayout(ann_group)
        ann_layout.addWidget(QLabel("Which class is being annotated?"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.project.class_names)
        ann_layout.addWidget(self.class_combo)
        ann_layout.addSpacing(4)
        ann_layout.addWidget(QLabel("Which keypoints are being edited?"))
        self.kpt_list = QListWidget()
        self._rebuild_kpt_list()
        ann_layout.addWidget(self.kpt_list, 1)
        ann_layout.addSpacing(4)
        ann_layout.addWidget(QLabel("How should annotation progression advance?"))
        self.rb_frame = QRadioButton("Frame-by-frame")
        self.rb_kpt = QRadioButton("Keypoint-by-keypoint")
        self.rb_frame.setChecked(True)
        ann_layout.addWidget(self.rb_frame)
        ann_layout.addWidget(self.rb_kpt)
        self.btn_clear_all_kpts = QPushButton("Clear Frame Keypoints")
        self.btn_clear_all_kpts.setToolTip(
            "Remove all keypoints from the current frame"
        )
        ann_layout.addWidget(self.btn_clear_all_kpts)
        right_layout.addWidget(ann_group)

        # Display
        disp_group = QGroupBox("View")
        disp_layout = QVBoxLayout(disp_group)
        self.cb_enhance = QCheckBox("Enhance contrast (CLAHE)")
        self.cb_enhance.setChecked(bool(self.project.enhance_enabled))
        self.btn_enhance_settings = QPushButton("Enhancement settings…")
        disp_layout.addWidget(self.cb_enhance)
        disp_layout.addWidget(self.btn_enhance_settings)
        self.cb_show_preds = QCheckBox("Show predictions")
        self.cb_show_preds.setChecked(True)
        disp_layout.addWidget(self.cb_show_preds)
        self.cb_show_pred_conf = QCheckBox("Show pred confidence")
        self.cb_show_pred_conf.setChecked(False)
        disp_layout.addWidget(self.cb_show_pred_conf)

        autosave_row = QHBoxLayout()
        autosave_row.addWidget(QLabel("How often should autosave run (sec)?"))
        self.sp_autosave_delay = QDoubleSpinBox()
        self.sp_autosave_delay.setRange(0.5, 30.0)
        self.sp_autosave_delay.setSingleStep(0.5)
        self.sp_autosave_delay.setValue(self.autosave_delay_ms / 1000.0)
        autosave_row.addWidget(self.sp_autosave_delay)
        disp_layout.addLayout(autosave_row)

        opacity_row1 = QHBoxLayout()
        opacity_row1.addWidget(QLabel("Keypoint opacity"))
        self.sp_kpt_opacity = QDoubleSpinBox()
        self.sp_kpt_opacity.setRange(0.0, 1.0)
        self.sp_kpt_opacity.setSingleStep(0.05)
        self.sp_kpt_opacity.setValue(self.project.kpt_opacity)
        opacity_row1.addWidget(self.sp_kpt_opacity)
        disp_layout.addLayout(opacity_row1)

        opacity_row2 = QHBoxLayout()
        opacity_row2.addWidget(QLabel("Edge opacity"))
        self.sp_edge_opacity = QDoubleSpinBox()
        self.sp_edge_opacity.setRange(0.0, 1.0)
        self.sp_edge_opacity.setSingleStep(0.05)
        self.sp_edge_opacity.setValue(self.project.edge_opacity)
        opacity_row2.addWidget(self.sp_edge_opacity)
        disp_layout.addLayout(opacity_row2)

        edge_width_row = QHBoxLayout()
        edge_width_row.addWidget(QLabel("Edge width"))
        self.sp_edge_width = QDoubleSpinBox()
        self.sp_edge_width.setRange(0.5, 10.0)
        self.sp_edge_width.setSingleStep(0.25)
        self.sp_edge_width.setValue(float(self.project.edge_width))
        edge_width_row.addWidget(self.sp_edge_width)
        disp_layout.addLayout(edge_width_row)

        size_row = QHBoxLayout()
        self.sp_kpt_size = QDoubleSpinBox()
        self.sp_kpt_size.setRange(0.5, 20.0)
        self.sp_kpt_size.setSingleStep(0.5)
        self.sp_kpt_size.setValue(float(self.project.kpt_radius))
        self.sp_label_size = QSpinBox()
        self.sp_label_size.setRange(4, 20)
        self.sp_label_size.setValue(int(self.project.label_font_size))
        size_row.addWidget(QLabel("Point size"))
        size_row.addWidget(self.sp_kpt_size)
        size_row.addSpacing(6)
        size_row.addWidget(QLabel("Text size"))
        size_row.addWidget(self.sp_label_size)
        disp_layout.addLayout(size_row)

        self.btn_fit_view = QPushButton("Fit to View (Ctrl+0)")
        disp_layout.addWidget(self.btn_fit_view)
        right_layout.addWidget(disp_group)

        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        row1 = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev (A)")
        self.btn_next = QPushButton("Next (D) ▶")
        row1.addWidget(self.btn_prev)
        row1.addWidget(self.btn_next)
        nav_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_save = QPushButton("Save (Ctrl+S)")
        self.btn_next_unl = QPushButton("Next Unlabeled (Ctrl+F)")
        row2.addWidget(self.btn_save)
        row2.addWidget(self.btn_next_unl)
        nav_layout.addLayout(row2)
        right_layout.addWidget(nav_group)

        # Model
        model_group = QGroupBox("Inference")
        model_layout = QVBoxLayout(model_group)

        backend_row = QHBoxLayout()
        backend_row.addWidget(
            QLabel("Which inference backend should generate predictions?")
        )
        self.combo_pred_backend = QComboBox()
        self.combo_pred_backend.addItems(["YOLO", "SLEAP"])
        backend_row.addWidget(self.combo_pred_backend, 1)
        model_layout.addLayout(backend_row)

        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel("Runtime"))
        self.combo_pred_runtime = QComboBox()
        self.combo_pred_runtime.setToolTip(
            "Inference runtime flavor.\n"
            "ONNX/TensorRT artifacts are auto-exported next to the selected model."
        )
        self._populate_pred_runtime_options("yolo")
        runtime_row.addWidget(self.combo_pred_runtime, 1)
        model_layout.addLayout(runtime_row)

        exported_row = QHBoxLayout()
        self.lbl_pred_exported = QLabel("Exported model")
        exported_row.addWidget(self.lbl_pred_exported)
        self.pred_exported_edit = QLineEdit("")
        self.pred_exported_edit.setPlaceholderText(
            "Optional exported model path (.onnx/.engine or exported SLEAP dir)"
        )
        self.btn_pred_exported = QPushButton("Browse…")
        exported_row.addWidget(self.pred_exported_edit, 1)
        exported_row.addWidget(self.btn_pred_exported)
        # Exported artifact paths are fully automatic; no user-facing selector row.
        self.lbl_pred_exported.setVisible(False)
        self.pred_exported_edit.setVisible(False)
        self.btn_pred_exported.setVisible(False)

        pred_conf_row = QHBoxLayout()
        pred_conf_row.addWidget(
            QLabel("What minimum prediction confidence should be shown/applied?")
        )
        self.sp_pred_conf = QDoubleSpinBox()
        self.sp_pred_conf.setRange(0.0, 1.0)
        self.sp_pred_conf.setSingleStep(0.05)
        self.sp_pred_conf.setValue(0.25)
        self.sp_pred_conf.setToolTip(
            "Minimum keypoint confidence to display/apply (YOLO + SLEAP)."
        )
        pred_conf_row.addWidget(self.sp_pred_conf)
        model_layout.addLayout(pred_conf_row)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch size"))
        self.spin_pred_batch = QSpinBox()
        self.spin_pred_batch.setRange(1, 256)
        self.spin_pred_batch.setValue(16)
        self.spin_pred_batch.setToolTip(
            "Shared batch size for dataset prediction across YOLO and SLEAP backends."
        )
        batch_row.addWidget(self.spin_pred_batch)
        model_layout.addLayout(batch_row)

        # YOLO settings
        self.yolo_pred_widget = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_pred_widget)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        yolo_layout.addWidget(QLabel("Pose weights (.pt)"))
        pred_weights_row = QHBoxLayout()
        self.pred_weights_edit = QLineEdit("")
        self.pred_weights_edit.setPlaceholderText("Select weights (.pt)")
        self.btn_pred_weights = QPushButton("Browse…")
        self.btn_pred_weights_latest = QPushButton("Use Latest")
        pred_weights_row.addWidget(self.pred_weights_edit, 1)
        pred_weights_row.addWidget(self.btn_pred_weights)
        pred_weights_row.addWidget(self.btn_pred_weights_latest)
        yolo_layout.addLayout(pred_weights_row)
        model_layout.addWidget(self.yolo_pred_widget)

        # SLEAP settings
        self.sleap_pred_widget = QWidget()
        sleap_layout = QFormLayout(self.sleap_pred_widget)
        sleap_layout.setContentsMargins(0, 0, 0, 0)
        env_row = QHBoxLayout()
        self.combo_sleap_env = QComboBox()
        self.combo_sleap_env.setToolTip("Environment name must start with 'sleap'.")
        self.btn_sleap_refresh = QPushButton("↻")
        self.btn_sleap_refresh.setMaximumWidth(40)
        self.btn_sleap_refresh.setToolTip("Refresh conda environments list")
        env_row.addWidget(self.combo_sleap_env, 1)
        env_row.addWidget(self.btn_sleap_refresh)
        sleap_layout.addRow("Conda environment", env_row)
        self.lbl_sleap_env_status = QLabel("")
        self.lbl_sleap_env_status.setStyleSheet(
            "QLabel { color: #f14c4c; font-size: 12px; }"
        )
        sleap_layout.addRow("", self.lbl_sleap_env_status)

        model_row = QHBoxLayout()
        self.sleap_model_edit = QLineEdit("")
        self.sleap_model_edit.setPlaceholderText("Select SLEAP model directory")
        self.btn_sleap_model = QPushButton("Browse…")
        self.btn_sleap_model_latest = QPushButton("Use Latest")
        model_row.addWidget(self.sleap_model_edit, 1)
        model_row.addWidget(self.btn_sleap_model)
        model_row.addWidget(self.btn_sleap_model_latest)
        sleap_layout.addRow("Model directory", model_row)

        sleap_btns = QHBoxLayout()
        self.btn_sleap_start = QPushButton("Start SLEAP Service")
        self.btn_sleap_stop = QPushButton("Stop SLEAP Service")
        sleap_btns.addWidget(self.btn_sleap_start)
        sleap_btns.addWidget(self.btn_sleap_stop)
        sleap_layout.addRow("", sleap_btns)

        model_layout.addWidget(self.sleap_pred_widget)

        self.btn_train = QPushButton("Train / Fine-tune…")
        self.btn_eval = QPushButton("Evaluate…")
        self.btn_active = QPushButton("Active Learning…")
        self.btn_predict = QPushButton("Predict Keypoints…")
        self.btn_predict_bulk = QPushButton("Predict Dataset…")
        self.btn_apply_preds = QPushButton("Apply Predictions")
        self.btn_clear_pred_cache = QPushButton("Clear Prediction Cache")
        model_btn_grid = QGridLayout()
        model_btn_grid.setHorizontalSpacing(6)
        model_btn_grid.setVerticalSpacing(6)
        model_btn_grid.addWidget(self.btn_train, 0, 0)
        model_btn_grid.addWidget(self.btn_eval, 0, 1)
        model_btn_grid.addWidget(self.btn_active, 1, 0)
        model_btn_grid.addWidget(self.btn_predict, 1, 1)
        model_btn_grid.addWidget(self.btn_predict_bulk, 2, 0)
        model_btn_grid.addWidget(self.btn_apply_preds, 2, 1)
        model_btn_grid.addWidget(self.btn_clear_pred_cache, 3, 0, 1, 2)
        model_layout.addLayout(model_btn_grid)
        right_layout.addWidget(model_group)

        # Project
        proj_group = QGroupBox("Project setup")
        proj_layout = QVBoxLayout(proj_group)
        self.btn_skel = QPushButton("Skeleton Editor (Ctrl+G)")
        self.btn_proj = QPushButton("Project Settings (Ctrl+P)")
        self.btn_export = QPushButton("Export dataset.yaml + splits…")
        proj_btn_grid = QGridLayout()
        proj_btn_grid.setHorizontalSpacing(6)
        proj_btn_grid.setVerticalSpacing(6)
        proj_btn_grid.addWidget(self.btn_skel, 0, 0)
        proj_btn_grid.addWidget(self.btn_proj, 0, 1)
        proj_btn_grid.addWidget(self.btn_export, 1, 0, 1, 2)
        proj_layout.addLayout(proj_btn_grid)
        right_layout.addWidget(proj_group)

        # Shortcuts
        self.controls_group = QGroupBox("Keyboard shortcuts")
        controls_layout = QVBoxLayout(self.controls_group)
        self.controls_label = QLabel(
            "Left click: place/move keypoint\n"
            "Right click: toggle vis (on keypoint)\n"
            "Wheel: zoom\n"
            "A/D: prev/next frame\n"
            "Q/E: prev/next keypoint\n"
            "Space: advance\n"
            "Ctrl+S: save\n"
            "Ctrl+F: next unlabeled"
        )
        self.controls_label.setWordWrap(True)
        controls_layout.addWidget(self.controls_label)
        right_layout.addWidget(self.controls_group)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        right_layout.addWidget(self.lbl_info)
        right_layout.addStretch(1)
        self._set_panel_controls_expanding(left)
        left.layout().activate()
        left_min_w = max(400, int(left.sizeHint().width()) + 24)
        left.setMinimumWidth(left_min_w)
        right.layout().activate()
        right_min_w = max(340, min(420, int(right.sizeHint().width()) + 16))
        right.setMinimumWidth(right_min_w)

        # Wrap left and right panels in ScrollAreas to allow scaling
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(left_min_w)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setMinimumWidth(right_min_w)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter.addWidget(left_scroll)
        splitter.addWidget(canvas_container)
        splitter.addWidget(right_scroll)

        # Give significantly more space to canvas (center is 10x larger than side panels)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 10)
        splitter.setStretchFactor(2, 1)

        splitter.setHandleWidth(6)

        # Set initial sizes: ~200px per side panel, rest to center
        # This will be adjusted when window is shown based on total width
        splitter.setSizes([left_min_w, 800, right_min_w])

        # Set minimum sizes to 0 to allow full collapsing/scaling
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, True)

        self.setStatusBar(QStatusBar())
        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setFixedWidth(160)
        self.status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)
        self.status_sleap = QLabel("SLEAP: off")
        self.status_sleap.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_sleap)
        self.status_autosave = QLabel("Saved")
        self.statusBar().addPermanentWidget(self.status_autosave)

        self._build_actions()

        # Periodic garbage collection to keep memory pressure down after heavy ops.
        self._gc_timer = QTimer(self)
        self._gc_timer.setInterval(60000)
        self._gc_timer.timeout.connect(lambda: gc.collect())
        self._gc_timer.start()

        # Timed autosave to avoid frequent blocking writes.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._perform_autosave)

        # Signals
        self.labeling_list.currentRowChanged.connect(self._on_labeling_frame_selected)
        self.frame_list.currentRowChanged.connect(self._on_all_frame_selected)
        self.labeling_list.model().rowsMoved.connect(self._on_labeling_list_changed)
        self.frame_list.model().rowsMoved.connect(self._on_all_list_changed)
        self.kpt_list.currentRowChanged.connect(self._on_kpt_selected)
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_next_unl.clicked.connect(self.next_unlabeled)
        self.btn_skel.clicked.connect(self.open_skeleton_editor)
        self.btn_proj.clicked.connect(self.open_project_settings)
        self.btn_export.clicked.connect(self.export_dataset_dialog)
        self.sp_autosave_delay.valueChanged.connect(self._update_autosave_delay)
        self.sp_pred_conf.valueChanged.connect(self._on_pred_conf_changed)
        self.btn_train.clicked.connect(self.open_training_runner)
        self.btn_eval.clicked.connect(self.open_evaluation_dashboard)
        self.btn_active.clicked.connect(self.open_active_learning)
        self.btn_predict.clicked.connect(self.predict_current_frame)
        self.btn_predict_bulk.clicked.connect(self.predict_dataset)
        self.cb_show_preds.toggled.connect(self._toggle_show_predictions)
        self.cb_show_pred_conf.toggled.connect(self._toggle_show_pred_conf)
        self.btn_apply_preds.clicked.connect(self.apply_predictions_current)
        self.btn_clear_pred_cache.clicked.connect(self._clear_prediction_cache)
        self.btn_pred_weights.clicked.connect(self._browse_pred_weights)
        self.btn_pred_weights_latest.clicked.connect(self._use_latest_pred_weights)
        self.btn_pred_exported.clicked.connect(self._browse_pred_exported_model)
        self.combo_pred_backend.currentTextChanged.connect(self._update_pred_backend_ui)
        self.combo_pred_runtime.currentTextChanged.connect(self._update_pred_backend_ui)
        self.btn_sleap_refresh.clicked.connect(self._refresh_sleap_envs)
        self.btn_sleap_model.clicked.connect(self._browse_sleap_model_dir)
        self.btn_sleap_model_latest.clicked.connect(self._use_latest_sleap_model)
        self.btn_sleap_start.clicked.connect(self._start_sleap_service)
        self.btn_sleap_stop.clicked.connect(self._stop_sleap_service)
        self.btn_unlabeled_to_labeling.clicked.connect(self._move_unlabeled_to_labeling)
        self.btn_unlabeled_to_all.clicked.connect(self._move_unlabeled_to_all)
        self.btn_random_to_labeling.clicked.connect(self._add_random_to_labeling)
        self.btn_smart_select.clicked.connect(self.open_smart_select)
        self.btn_delete_frames.clicked.connect(self._delete_selected_frames)
        self.btn_manage_sources_inline.clicked.connect(self.manage_sources)
        self.search_edit.textChanged.connect(self._populate_frames)
        self.sort_combo.currentTextChanged.connect(self._populate_frames)
        self.source_combo.currentIndexChanged.connect(self._on_source_selector_changed)
        self.rb_frame.toggled.connect(self._update_mode)
        self.btn_clear_all_kpts.clicked.connect(self.clear_all_keypoints)
        self.class_combo.currentIndexChanged.connect(self._mark_dirty)
        self.cb_enhance.toggled.connect(self._toggle_enhancement)
        self.btn_enhance_settings.clicked.connect(self._open_enhancement_settings)
        self.sp_kpt_size.valueChanged.connect(self._update_kpt_size)
        self.sp_label_size.valueChanged.connect(self._update_label_size)
        self.sp_kpt_opacity.valueChanged.connect(self._update_kpt_opacity)
        self.sp_edge_opacity.valueChanged.connect(self._update_edge_opacity)
        self.sp_edge_width.valueChanged.connect(self._update_edge_width)
        self.btn_fit_view.clicked.connect(self.fit_to_view)

        # Load UI settings now that all widgets are created
        self._refresh_sleap_envs()
        self._load_ui_settings()
        self._refresh_source_selector()
        self._update_pred_backend_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Populate list + load
        self._populate_frames()
        # Don't auto-load a frame on startup to avoid odd zoom; user clicks to load.
        self.frame_list.setCurrentRow(-1)
        self.labeling_list.setCurrentRow(-1)
        self.source_combo.setCurrentIndex(self.source_combo.currentIndex())
        self.lbl_info.setText("Select a frame to display.")
        self._show_canvas_logo_placeholder()
        self._set_workspace_visible(
            bool(self.image_paths) or not self._show_welcome_when_empty
        )

    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown when PoseKit starts without a loaded project."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("posekit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="posekit.svg",
            tagline="Pose Labeling Workspace",
            buttons=[
                ButtonDef(label="New Project\u2026", callback=self.new_project_wizard),
                ButtonDef(
                    label="Open Existing Project\u2026", callback=self.open_project
                ),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
            recent_display_name=self._recent_project_display_name,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    @staticmethod
    def _recent_project_display_name(path: str) -> str:
        """Return the label to show for a recent PoseKit project entry."""
        project_path = Path(str(path).strip()).expanduser()
        if (
            project_path.name == DEFAULT_PROJECT_NAME
            and project_path.parent.name == "state"
            and project_path.parent.parent.name
        ):
            return project_path.parent.parent.name
        if project_path.name == DEFAULT_PROJECT_NAME and project_path.parent.name:
            return project_path.parent.name
        return project_path.name or str(path)

    def _open_recent_project(self, path: str):
        """Open a project from the recent items list."""
        from pathlib import Path

        project_path = Path(path)
        if project_path.exists():
            proj = open_project_from_path(project_path)
            if proj is not None:
                self._switch_project_window(proj)
            else:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self, "Open Failed", f"Could not open project at:\n{path}"
                )
                if hasattr(self, "_recents_store"):
                    self._recents_store.remove(path)
                    if hasattr(self, "_welcome_page"):
                        self._welcome_page.refresh_recents()
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()

    def apply_stylesheet(self):
        """Apply the PoseKit dark theme to the entire window."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
            }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px 8px 8px 8px;
                font-weight: 600;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                background-color: #1e1e1e;
                color: #9cdcfe;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #252526;
                alternate-background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 3px;
                margin: 1px 0px;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background-color: #2a2d2e;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8f;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #777777;
            }
            QPushButton[flat="true"] {
                background-color: transparent;
                border: 1px solid #3e3e42;
                color: #cccccc;
            }
            QPushButton[flat="true"]:hover {
                background-color: #2a2d2e;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QComboBox:hover {
                border-color: #0e639c;
            }
            QComboBox:focus {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #252526;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
                selection-color: #ffffff;
                outline: none;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QLineEdit:hover {
                border-color: #0e639c;
            }
            QLineEdit:focus {
                border-color: #007acc;
            }
            QLineEdit::placeholder {
                color: #6a6a6a;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 4px 4px 8px;
                min-height: 22px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #0e639c;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #007acc;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                background-color: #4a4a4a;
                border: none;
                border-radius: 2px;
                width: 16px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #4a4a4a;
                border: none;
                border-radius: 2px;
                width: 16px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #0e639c;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border-color: #007acc;
            }
            QCheckBox::indicator:hover {
                border-color: #007acc;
            }
            QRadioButton {
                color: #cccccc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 7px;
                background-color: #3c3c3c;
            }
            QRadioButton::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QRadioButton::indicator:hover {
                border-color: #007acc;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QToolBar {
                background-color: #252526;
                border-bottom: 1px solid #3e3e42;
                spacing: 6px;
                padding: 4px 6px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                color: #cccccc;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #2a2d2e;
            }
            QToolButton:pressed {
                background-color: #094771;
            }
            QToolButton:checked {
                background-color: #094771;
                color: #4fc1ff;
            }
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
                border-top: 1px solid #0098ff;
                font-weight: 500;
                font-size: 12px;
            }
            QStatusBar QLabel {
                background-color: transparent;
                color: #ffffff;
                padding: 0px 4px;
            }
            QMenuBar {
                background-color: #252526;
                color: #cccccc;
                border-bottom: 1px solid #3e3e42;
                padding: 2px;
            }
            QMenuBar::item {
                padding: 5px 10px;
                background-color: transparent;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #2a2d2e;
            }
            QMenuBar::item:pressed {
                background-color: #094771;
            }
            QMenu {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #3e3e42;
                margin: 4px 8px;
            }
            QSplitter::handle {
                background-color: #3e3e42;
            }
            QSplitter::handle:hover {
                background-color: #007acc;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #252526;
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #007acc;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #252526;
                height: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a;
                border-radius: 5px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #007acc;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
                color: #cccccc;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 3px;
            }
            QPlainTextEdit, QTextEdit {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
            }
            QPlainTextEdit:focus, QTextEdit:focus {
                border-color: #007acc;
            }
            QTableWidget {
                background-color: #252526;
                gridline-color: #3e3e42;
                border: 1px solid #3e3e42;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
            QTableWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #cccccc;
                border: none;
                border-right: 1px solid #3e3e42;
                border-bottom: 1px solid #3e3e42;
                padding: 4px 8px;
                font-weight: 600;
            }
            QFrame[frameShape="4"], QFrame[frameShape="5"] {
                color: #3e3e42;
            }
        """)

    def _show_canvas_logo_placeholder(self):
        """Show PoseKit logo in the center canvas when no frame is active."""
        try:
            from PySide6.QtCore import QByteArray

            from hydra_suite.paths import get_brand_icon_bytes

            logo_data = get_brand_icon_bytes("posekit.svg")
            vw = max(1000, self.canvas.viewport().width())
            vh = max(700, self.canvas.viewport().height())
            canvas = QPixmap(vw, vh)
            canvas.fill(QColor(18, 18, 18))

            renderer = (
                QSvgRenderer(QByteArray(logo_data)) if logo_data else QSvgRenderer()
            )
            if renderer.isValid():
                view_box = renderer.viewBoxF()
                if view_box.isEmpty():
                    default_size = renderer.defaultSize()
                    view_box = QRectF(
                        0,
                        0,
                        max(1, default_size.width()),
                        max(1, default_size.height()),
                    )

                # Use "contain" scaling from the SVG viewBox to avoid stretch/crop artifacts.
                max_w = max(1, int(vw * 0.9))
                max_h = max(1, int(vh * 0.88))
                scale = min(max_w / view_box.width(), max_h / view_box.height())
                logo_w = max(1, int(view_box.width() * scale))
                logo_h = max(1, int(view_box.height() * scale))
                x = (vw - logo_w) // 2
                y = (vh - logo_h) // 2
                painter = QPainter(canvas)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                renderer.render(painter, QRectF(x, y, logo_w, logo_h))
                painter.end()

                self.canvas.pix_item.setPixmap(canvas)
                self.canvas.scene.setSceneRect(
                    QRectF(0, 0, canvas.width(), canvas.height())
                )
                self.canvas.resetTransform()
                self.canvas.fitInView(self.canvas.pix_item, Qt.KeepAspectRatio)
                # Fit again after layout settles to avoid tiny initial scale on startup.
                QTimer.singleShot(0, self.canvas.fit_to_view)
                return
        except Exception:
            pass

    def new_project_wizard(self) -> None:
        """Run the New Project wizard and switch to the created project."""
        proj = create_standalone_project_via_wizard(parent_widget=self)
        if proj is None:
            return
        self._switch_project_window(proj, open_source_manager_if_empty=True)

    def closeEvent(self: object, event: object) -> object:
        """Save UI settings when window closes."""
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass

        # Ensure background workers are fully stopped before Qt tears down objects.
        self._shutdown_worker_thread(
            thread_attr="_bulk_pred_thread",
            worker_attr="_bulk_pred_worker",
            cancel_method="cancel",
        )
        self._shutdown_worker_thread(
            thread_attr="_pred_thread",
            worker_attr="_pred_worker",
            cancel_method=None,
        )
        self._shutdown_worker_thread(
            thread_attr="_sleap_service_thread",
            worker_attr="_sleap_service_worker",
            cancel_method=None,
        )

        self._perform_autosave()
        self.save_project()
        self._save_ui_settings()
        try:
            PoseInferenceService.shutdown_sleap_service()
        except Exception:
            pass
        super().closeEvent(event)

    def _shutdown_worker_thread(
        self,
        thread_attr: str,
        worker_attr: Optional[str] = None,
        cancel_method: Optional[str] = "cancel",
        timeout_ms: int = 4000,
    ) -> None:
        thread = getattr(self, thread_attr, None)
        worker = getattr(self, worker_attr, None) if worker_attr else None

        if worker is not None and cancel_method and hasattr(worker, cancel_method):
            try:
                getattr(worker, cancel_method)()
            except Exception:
                pass

        if thread is None:
            if worker_attr:
                setattr(self, worker_attr, None)
            return

        try:
            if thread.isRunning():
                try:
                    thread.quit()
                except Exception:
                    pass
                try:
                    stopped = bool(thread.wait(int(max(100, timeout_ms))))
                except Exception:
                    stopped = False
                if not stopped:
                    try:
                        thread.terminate()
                    except Exception:
                        pass
                    try:
                        thread.wait(1000)
                    except Exception:
                        pass
        except RuntimeError:
            # Qt object may already be deleted.
            pass
        finally:
            setattr(self, thread_attr, None)
            if worker_attr:
                setattr(self, worker_attr, None)

    def eventFilter(self, obj, event) -> bool:
        """Suppress mouse-wheel events on spin-boxes to prevent accidental value changes."""
        if event is not None and event.type() == QEvent.Wheel:
            if isinstance(obj, QAbstractSpinBox):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self: object, event: object) -> object:
        """Handle arrow key nudging of keypoints with modifier keys."""
        # Only process arrow keys
        if event.key() not in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
            super().keyPressEvent(event)
            return

        # Get current keypoint value
        if self._ann is None or self.current_kpt >= len(self._ann.kpts):
            super().keyPressEvent(event)
            return

        x, y, v = (
            self._ann.kpts[self.current_kpt].x,
            self._ann.kpts[self.current_kpt].y,
            self._ann.kpts[self.current_kpt].v,
        )

        # Only nudge if keypoint is placed
        if v <= 0:
            super().keyPressEvent(event)
            return

        # In keypoint mode, only allow nudging if all keypoints are labeled
        if self.mode == "keypoint":
            all_labeled = all(kp.v > 0 for kp in self._ann.kpts)
            if not all_labeled:
                super().keyPressEvent(event)
                return

        # Calculate base nudge amount (0.5% of average image dimension)
        if self._img_bgr is not None:
            h, w = self._img_bgr.shape[:2]
            base_nudge = 0.005 * ((w + h) / 2)
        else:
            base_nudge = 2.0  # fallback

        # Apply modifiers
        modifiers = event.modifiers()
        if modifiers & Qt.ShiftModifier:
            nudge = base_nudge * 5.0  # 5x faster
        elif modifiers & Qt.ControlModifier:
            nudge = base_nudge * 0.2  # 0.2x slower (pixel-level precision)
        else:
            nudge = base_nudge  # normal speed

        # Apply nudge based on arrow key
        if event.key() == Qt.Key_Left:
            x -= nudge
        elif event.key() == Qt.Key_Right:
            x += nudge
        elif event.key() == Qt.Key_Up:
            y -= nudge
        elif event.key() == Qt.Key_Down:
            y += nudge

        # Bounds checking
        if self._img_bgr is not None:
            h, w = self._img_bgr.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

        # Update keypoint
        self._ann.kpts[self.current_kpt] = Keypoint(x, y, v)
        self._set_dirty()

        # Rebuild overlays
        self._rebuild_canvas()

        # Save/cache based on mode
        if self.mode == "frame":
            self.save_current(refresh_ui=False)
        else:  # keypoint mode
            self._cache_current_frame()

        event.accept()

    def _save_ui_settings(self):
        """Save UI settings to persistent storage."""
        settings = {
            "enhance_enabled": self.project.enhance_enabled,
            "kpt_radius": self.project.kpt_radius,
            "label_font_size": self.project.label_font_size,
            "kpt_opacity": self.project.kpt_opacity,
            "edge_opacity": self.project.edge_opacity,
            "edge_width": self.project.edge_width,
            "clahe_clip": self.project.clahe_clip,
            "clahe_grid": list(self.project.clahe_grid),
            "sharpen_amt": self.project.sharpen_amt,
            "blur_sigma": self.project.blur_sigma,
            "controls_open": bool(self.controls_group.isChecked()),
            "frame_search": self.search_edit.text().strip(),
            "autosave_delay_ms": int(self.autosave_delay_ms),
            "pred_conf": float(self.sp_pred_conf.value()),
            "pred_weights": self.pred_weights_edit.text().strip(),
            "pred_backend": self.combo_pred_backend.currentText().strip(),
            "runtime_tier": self._selected_tier(),
            "compute_runtime": self._selected_compute_runtime(),
            "pred_runtime": self._selected_compute_runtime(),
            "pred_exported_model": "",
            "pred_batch": int(self.spin_pred_batch.value()),
            "sleap_env": self.combo_sleap_env.currentText().strip(),
            "sleap_model_dir": self.sleap_model_edit.text().strip(),
            "show_predictions": bool(self.cb_show_preds.isChecked()),
            "show_pred_conf": bool(self.cb_show_pred_conf.isChecked()),
        }
        save_ui_settings(settings)

    def _apply_display_settings(self, settings):
        if "kpt_radius" in settings:
            self.project.kpt_radius = float(settings["kpt_radius"])
            self.sp_kpt_size.setValue(self.project.kpt_radius)
        if "label_font_size" in settings:
            self.project.label_font_size = int(settings["label_font_size"])
            self.sp_label_size.setValue(self.project.label_font_size)
        if "kpt_opacity" in settings:
            self.project.kpt_opacity = float(settings["kpt_opacity"])
            self.sp_kpt_opacity.setValue(self.project.kpt_opacity)
        if "edge_opacity" in settings:
            self.project.edge_opacity = float(settings["edge_opacity"])
            self.sp_edge_opacity.setValue(self.project.edge_opacity)
        if "edge_width" in settings:
            self.project.edge_width = float(settings["edge_width"])
            self.sp_edge_width.setValue(self.project.edge_width)
        if "controls_open" in settings:
            self.controls_group.setChecked(bool(settings["controls_open"]))
        if "frame_search" in settings:
            self.search_edit.setText(str(settings["frame_search"]))
        if "autosave_delay_ms" in settings:
            self.autosave_delay_ms = int(settings["autosave_delay_ms"])
            self.sp_autosave_delay.setValue(self.autosave_delay_ms / 1000.0)

    def _apply_pred_runtime_setting(self, settings):
        runtime_setting = str(
            settings.get(
                "runtime_tier",
                settings.get("compute_runtime", settings.get("pred_runtime", "")),
            )
        ).strip()
        if not runtime_setting:
            return
        # If a tier id is stored directly, use it; otherwise migrate a legacy runtime.
        platform = detect_platform()
        if runtime_setting in available_tiers(platform):
            preferred_tier = runtime_setting
        else:
            # Legacy path: canonicalise the runtime string then map to tier.
            canonical_runtime = runtime_setting
            if canonical_runtime not in CANONICAL_RUNTIMES:
                canonical_runtime = infer_compute_runtime_from_legacy(
                    yolo_device="auto",
                    enable_tensorrt=False,
                    pose_runtime_flavor=runtime_setting,
                )
            preferred_tier = canonical_runtime_to_tier(canonical_runtime)
        self._populate_pred_runtime_options(preferred_tier=preferred_tier)

    def _apply_pred_batch_setting(self, settings):
        if "pred_batch" in settings:
            self.spin_pred_batch.setValue(int(settings["pred_batch"]))
        elif "pred_yolo_batch" in settings:
            self.spin_pred_batch.setValue(int(settings["pred_yolo_batch"]))
        elif "sleap_batch_predict" in settings:
            self.spin_pred_batch.setValue(int(settings["sleap_batch_predict"]))
        elif "sleap_batch" in settings:
            # Backwards compatibility for older settings key.
            self.spin_pred_batch.setValue(int(settings["sleap_batch"]))

    def _apply_pred_settings(self, settings):
        if "pred_conf" in settings:
            self.sp_pred_conf.setValue(float(settings["pred_conf"]))
        if "pred_weights" in settings:
            self.pred_weights_edit.setText(str(settings["pred_weights"]))
        if "pred_backend" in settings:
            backend = str(settings["pred_backend"])
            if backend:
                self.combo_pred_backend.setCurrentText(backend)
        self._apply_pred_runtime_setting(settings)
        self.pred_exported_edit.setText("")
        self._apply_pred_batch_setting(settings)

    def _apply_sleap_settings(self, settings):
        if "sleap_env" in settings:
            self._sleap_env_pref = str(settings["sleap_env"]).strip()
            if self._sleap_env_pref:
                if self._sleap_env_pref in [
                    self.combo_sleap_env.itemText(i)
                    for i in range(self.combo_sleap_env.count())
                ]:
                    self.combo_sleap_env.setCurrentText(self._sleap_env_pref)
        if "sleap_model_dir" in settings:
            self.sleap_model_edit.setText(str(settings["sleap_model_dir"]))

    def _apply_prediction_visibility_settings(self, settings):
        if "show_predictions" in settings:
            self.cb_show_preds.setChecked(bool(settings["show_predictions"]))
            self.show_predictions = bool(self.cb_show_preds.isChecked())
        if "show_pred_conf" in settings:
            self.cb_show_pred_conf.setChecked(bool(settings["show_pred_conf"]))
            self.show_pred_conf = bool(self.cb_show_pred_conf.isChecked())
        if not self.pred_weights_edit.text().strip():
            if (
                self.project.latest_pose_weights
                and Path(self.project.latest_pose_weights).exists()
            ):
                self.pred_weights_edit.setText(str(self.project.latest_pose_weights))

    def _load_ui_settings(self):
        """Load UI settings from persistent storage and apply defaults."""
        settings = load_ui_settings()
        if settings:
            self._apply_display_settings(settings)
            self._apply_pred_settings(settings)
            self._apply_sleap_settings(settings)
            self._apply_prediction_visibility_settings(settings)

    def _set_workspace_visible(self, visible: bool) -> None:
        """Toggle between the welcome page and the working PoseKit shell."""
        self._content_stack.setCurrentIndex(1 if visible else 0)
        self.menuBar().setVisible(visible)
        for toolbar in self.findChildren(QToolBar):
            toolbar.setVisible(visible)

    # ----- menus / shortcuts -----
    def _build_actions(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("&File")
        m_nav = menubar.addMenu("&Navigate")
        m_tools = menubar.addMenu("&Tools")
        m_model = menubar.addMenu("&Model")

        act_save = QAction("Save", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self.save_all_labeling_frames)

        act_new_proj = QAction("New Project…", self)
        act_new_proj.setShortcut(QKeySequence("Ctrl+N"))
        act_new_proj.triggered.connect(self.new_project_wizard)

        act_open_proj = QAction("Open Project…", self)
        act_open_proj.setShortcut(QKeySequence.Open)
        act_open_proj.triggered.connect(self.open_project)

        act_manage_sources = QAction("Source Manager…", self)
        act_manage_sources.setShortcut(QKeySequence("Ctrl+I"))
        act_manage_sources.triggered.connect(self.manage_sources)

        act_export = QAction("Export dataset.yaml + splits…", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.export_dataset_dialog)

        act_proj = QAction("Project Settings…", self)
        act_proj.setShortcut(QKeySequence("Ctrl+P"))
        act_proj.triggered.connect(self.open_project_settings)

        act_prev = QAction("Prev Frame", self)
        act_prev.setShortcut(QKeySequence("A"))
        act_prev.triggered.connect(self.prev_frame)

        act_next = QAction("Next Frame", self)
        act_next.setShortcut(QKeySequence("D"))
        act_next.triggered.connect(self.next_frame)

        act_prev_k = QAction("Prev Keypoint", self)
        act_prev_k.setShortcut(QKeySequence("Q"))
        act_prev_k.triggered.connect(self.prev_keypoint)

        act_next_k = QAction("Next Keypoint", self)
        act_next_k.setShortcut(QKeySequence("E"))
        act_next_k.triggered.connect(self.next_keypoint)

        act_next_unl = QAction("Next Unlabeled", self)
        act_next_unl.setShortcut(QKeySequence("Ctrl+F"))
        act_next_unl.triggered.connect(self.next_unlabeled)

        act_skel = QAction("Skeleton Editor", self)
        act_skel.setShortcut(QKeySequence("Ctrl+G"))
        act_skel.triggered.connect(self.open_skeleton_editor)

        act_clear = QAction("Clear Current Keypoint", self)
        act_clear.setShortcut(QKeySequence(Qt.Key_Delete))
        act_clear.triggered.connect(self.clear_current_keypoint)

        act_clear_all = QAction("Clear All Keypoints", self)
        act_clear_all.setShortcut(QKeySequence("Ctrl+Shift+Delete"))
        act_clear_all.triggered.connect(self.clear_all_keypoints)

        act_undo = QAction("Undo", self)
        act_undo.setShortcut(QKeySequence.Undo)
        act_undo.triggered.connect(self.undo_last)

        act_fit = QAction("Fit to View", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self.fit_to_view)

        self.act_enhance = QAction("Enhance Contrast (CLAHE)", self)
        self.act_enhance.setCheckable(True)
        self.act_enhance.setChecked(bool(self.project.enhance_enabled))
        self.act_enhance.setShortcut(QKeySequence("Ctrl+H"))
        self.act_enhance.triggered.connect(
            lambda checked: self._toggle_enhancement(checked)
        )

        self.act_enhance_settings = QAction("Enhancement Settings…", self)
        self.act_enhance_settings.triggered.connect(self._open_enhancement_settings)

        self.act_show_pred_conf = QAction("Show Prediction Confidence", self)
        self.act_show_pred_conf.setCheckable(True)
        self.act_show_pred_conf.setChecked(bool(self.show_pred_conf))
        self.act_show_pred_conf.triggered.connect(self._toggle_show_pred_conf)

        act_train = QAction("Training Runner…", self)
        act_train.triggered.connect(self.open_training_runner)

        act_eval = QAction("Evaluation Dashboard…", self)
        act_eval.triggered.connect(self.open_evaluation_dashboard)

        act_active = QAction("Active Learning…", self)
        act_active.triggered.connect(self.open_active_learning)

        m_file.addAction(act_save)
        m_file.addSeparator()
        m_file.addAction(act_new_proj)
        m_file.addAction(act_open_proj)
        m_file.addAction(act_manage_sources)
        m_file.addSeparator()
        m_file.addAction(act_proj)
        m_file.addSeparator()
        m_file.addAction(act_export)

        m_nav.addAction(act_prev)
        m_nav.addAction(act_next)
        m_nav.addAction(act_prev_k)
        m_nav.addAction(act_next_k)
        m_nav.addSeparator()
        m_nav.addSeparator()
        m_nav.addAction(act_next_unl)

        m_tools.addAction(act_skel)
        m_tools.addAction(act_clear)
        m_tools.addAction(act_clear_all)
        m_tools.addSeparator()
        m_tools.addAction(act_undo)
        m_tools.addSeparator()
        m_tools.addAction(act_fit)
        m_tools.addSeparator()
        m_tools.addAction(self.act_enhance)
        m_tools.addAction(self.act_enhance_settings)
        m_tools.addSeparator()
        m_tools.addAction(self.act_show_pred_conf)

        m_model.addAction(act_train)
        m_model.addAction(act_eval)
        m_model.addAction(act_active)

        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        tb.addAction(act_new_proj)
        tb.addAction(act_open_proj)
        tb.addAction(act_save)
        tb.addAction(act_manage_sources)
        tb.addSeparator()
        tb.addAction(act_prev)
        tb.addAction(act_next)
        tb.addAction(act_next_unl)
        tb.addSeparator()
        tb.addAction(act_skel)
        tb.addAction(act_proj)
        tb.addAction(act_export)
        tb.addSeparator()
        tb.addAction(self.act_enhance)
        tb.addSeparator()
        tb.addAction(act_undo)

    # ----- file paths -----
    def _label_path_for(self, img_path: Path) -> Path:
        """Return the label .txt path for *img_path*, routed per-source when available."""
        src_id = self._image_source.get(str(img_path)) or self._image_source.get(
            str(img_path.resolve()) if img_path.exists() else str(img_path)
        )
        if src_id and src_id in self._source_map:
            labels_dir = self._source_map[src_id].labels_dir
        else:
            labels_dir = self.project.labels_dir
        return labels_dir / (img_path.stem + ".txt")

    def _is_labeled(self, img_path: Path) -> bool:
        lp = self._label_path_for(img_path)
        if not lp.exists():
            return False
        return bool(lp.read_text(encoding="utf-8").strip())

    def _set_panel_controls_expanding(self, panel: QWidget):
        for w in panel.findChildren(QWidget):
            if isinstance(w, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
                pol = w.sizePolicy()
                pol.setHorizontalPolicy(QSizePolicy.Expanding)
                w.setSizePolicy(pol)

    def _count_labeled_kpts(self, img_path: Path) -> int:
        try:
            loaded = load_yolo_pose_label(
                self._label_path_for(img_path), len(self.project.keypoint_names)
            )
        except Exception:
            return 0
        if loaded is None:
            return 0
        _cls, kpts_norm, _bbox = loaded
        return sum(1 for kp in kpts_norm if kp.v > 0)

    def save_project(self: object) -> object:
        """save_project method documentation."""
        if hasattr(self.project, "labeling_frames"):
            self.project.labeling_frames = sorted(
                {int(i) for i in self.labeling_frames}
            )
        self.project.last_index = self.current_index
        save_project_state(self.project)

    # ----- list / info -----
    def _populate_frames(self):
        self._suppress_list_rebuild = True
        self.labeling_list.blockSignals(True)
        self.frame_list.blockSignals(True)
        self.labeling_list.setUpdatesEnabled(False)
        self.frame_list.setUpdatesEnabled(False)
        self.labeling_list.clear()
        self.frame_list.clear()
        self._list_items = {}

        query = self.search_edit.text().strip().lower()
        indices = []
        for idx, img_path in enumerate(self.image_paths):
            if query and query not in img_path.name.lower():
                continue
            indices.append(idx)

        conf_map = self._get_pred_conf_for_indices(indices)
        kpt_map = self._get_pred_kpt_count_for_indices(indices)
        cluster_map = self._get_cluster_id_for_indices(indices)
        sort_mode = (
            self.sort_combo.currentText().strip()
            if hasattr(self, "sort_combo")
            else "Default"
        )
        reverse = sort_mode in {
            "Pred conf (high to low)",
            "Detected kpts (high to low)",
            "Cluster id (high to low)",
        }

        items = []
        for idx in indices:
            img_path = self.image_paths[idx]
            # Check if saved to disk
            is_saved = self._is_labeled(img_path)
            in_cache = idx in self._frame_cache

            # Determine marker: tick (saved), asterisk (modified but unsaved), or empty
            if in_cache and not is_saved:
                tick = "* "  # Modified but not saved
            elif is_saved:
                tick = "✓ "  # Saved to disk
            else:
                tick = "  "  # No changes

            # Count labeled keypoints - prefer cache over disk
            num_labeled = 0
            total_kpts = len(self.project.keypoint_names)
            if in_cache:
                # Use cache if available (most up-to-date)
                cached = self._frame_cache[idx]
                num_labeled = sum(1 for kp in cached.kpts if kp.v > 0)
            elif is_saved:
                num_labeled = self._count_labeled_kpts(img_path)

            # Determine color: Green=all labeled, Orange=some labeled, White=none
            if num_labeled == total_kpts:
                color = QColor(0, 200, 0)  # Green
            elif num_labeled > 0:
                color = QColor(255, 165, 0)  # Orange
            else:
                color = QColor(220, 220, 220)  # White

            item_text = f"{tick}{img_path.name}"
            # Show a short source badge when the project has multiple sources
            if len(self.project.sources) > 1:
                src_id = self._image_source.get(str(img_path), "")
                src = self._source_map.get(src_id)
                badge = (src.description or src.source_id) if src else src_id
                # Truncate long descriptions to keep list readable
                if len(badge) > 20:
                    badge = badge[:17] + "…"
                item_text = f"{tick}[{badge}] {img_path.name}"
            pred_conf = conf_map.get(idx)
            pred_kpt_count = kpt_map.get(idx)
            cluster_id = cluster_map.get(idx)
            items.append(
                {
                    "idx": idx,
                    "is_saved": is_saved,
                    "in_labeling": idx in self.labeling_frames,
                    "in_current_source": self._matches_current_source(idx),
                    "item_text": item_text,
                    "color": color,
                    "pred_conf": pred_conf,
                    "pred_kpt_count": pred_kpt_count,
                    "cluster_id": cluster_id,
                }
            )

        if sort_mode != "Default":
            if sort_mode.startswith("Pred conf"):
                items.sort(
                    key=lambda it: (
                        it["pred_conf"]
                        if it["pred_conf"] is not None
                        else (-1.0 if reverse else 1e9)
                    ),
                    reverse=reverse,
                )
            elif sort_mode.startswith("Detected kpts"):
                items.sort(
                    key=lambda it: (
                        it["pred_kpt_count"]
                        if it["pred_kpt_count"] is not None
                        else (-1 if reverse else 1_000_000)
                    ),
                    reverse=reverse,
                )
            elif sort_mode.startswith("Cluster id"):
                items.sort(
                    key=lambda it: (
                        it["cluster_id"]
                        if it["cluster_id"] is not None
                        else (-1 if reverse else 1_000_000)
                    ),
                    reverse=reverse,
                )

        for it in items:
            idx = it["idx"]
            # Labeled frames always go to labeling list
            if it["is_saved"] or it["in_labeling"]:
                if it["is_saved"]:
                    self.labeling_frames.add(idx)
                item = QListWidgetItem(it["item_text"])
                item.setData(Qt.UserRole, idx)
                item.setData(FrameListDelegate.CONF_ROLE, it["pred_conf"])
                item.setData(FrameListDelegate.KP_COUNT_ROLE, it["pred_kpt_count"])
                item.setData(FrameListDelegate.CLUSTER_ROLE, it["cluster_id"])
                item.setForeground(it["color"])
                self.labeling_list.addItem(item)
                self._list_items[idx] = item
            elif it["in_current_source"]:
                item = QListWidgetItem(it["item_text"])
                item.setData(Qt.UserRole, idx)
                item.setData(FrameListDelegate.CONF_ROLE, it["pred_conf"])
                item.setData(FrameListDelegate.KP_COUNT_ROLE, it["pred_kpt_count"])
                item.setData(FrameListDelegate.CLUSTER_ROLE, it["cluster_id"])
                item.setForeground(it["color"])
                self.frame_list.addItem(item)
                self._list_items[idx] = item

        self.labeling_list.blockSignals(False)
        self.frame_list.blockSignals(False)
        self.labeling_list.setUpdatesEnabled(True)
        self.frame_list.setUpdatesEnabled(True)
        self._suppress_list_rebuild = False

    @staticmethod
    def _apply_item_data(item, color, text, pred_conf, pred_kpt_count, cluster_id):
        """Apply visual properties and data roles to a list widget item."""
        item.setForeground(color)
        item.setText(text)
        item.setData(FrameListDelegate.CONF_ROLE, pred_conf)
        item.setData(FrameListDelegate.KP_COUNT_ROLE, pred_kpt_count)
        item.setData(FrameListDelegate.CLUSTER_ROLE, cluster_id)

    def _compute_frame_display_info(self, idx: int):
        """Compute the marker tick, color, and text for a frame list item.

        Returns (tick, color, item_text_suffix).
        """
        img_path = self.image_paths[idx]
        is_saved = self._is_labeled(img_path)
        in_cache = idx in self._frame_cache

        if in_cache and not is_saved:
            tick = "* "
        elif is_saved:
            tick = "\u2713 "
        else:
            tick = "  "

        num_labeled = 0
        total_kpts = len(self.project.keypoint_names)
        if in_cache:
            cached = self._frame_cache[idx]
            num_labeled = sum(1 for kp in cached.kpts if kp.v > 0)
        elif is_saved:
            num_labeled = self._count_labeled_kpts(img_path)

        if num_labeled == total_kpts:
            color = QColor(0, 200, 0)
        elif num_labeled > 0:
            color = QColor(255, 165, 0)
        else:
            color = QColor(220, 220, 220)

        item_text = f"{tick}{img_path.name}"
        if len(self.project.sources) > 1:
            src_id = self._image_source.get(str(img_path), "")
            src = self._source_map.get(src_id)
            badge = (src.description or src.source_id) if src else src_id
            if len(badge) > 20:
                badge = badge[:17] + "\u2026"
            item_text = f"{tick}[{badge}] {img_path.name}"

        return is_saved, color, item_text

    def _update_frame_item(
        self,
        idx: int,
        pred_conf: Optional[float] = None,
        pred_kpt_count: Optional[int] = None,
        cluster_id: Optional[int] = None,
        conf_only: bool = False,
    ):
        """Update a single frame item in the lists without rebuilding everything."""
        if conf_only:
            item = self._list_items.get(idx)
            if item is not None:
                item.setData(FrameListDelegate.CONF_ROLE, pred_conf)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, pred_kpt_count)
                if cluster_id is not None:
                    item.setData(FrameListDelegate.CLUSTER_ROLE, cluster_id)
            return

        is_saved, color, item_text = self._compute_frame_display_info(idx)

        if pred_conf is None:
            pred_conf = self._get_pred_conf_for_indices([idx]).get(idx)
        if pred_kpt_count is None:
            pred_kpt_count = self._get_pred_kpt_count_for_indices([idx]).get(idx)
        if cluster_id is None:
            cluster_id = self._get_cluster_id_for_indices([idx]).get(idx)

        # Try fast-path via _list_items cache.
        item = self._list_items.get(idx)
        if item is not None:
            needs_labeling = is_saved or (idx in self.labeling_frames)
            row_in_frame = self.frame_list.row(item)
            if needs_labeling and row_in_frame >= 0:
                self.frame_list.takeItem(row_in_frame)
                self.labeling_frames.add(idx)
                self._apply_item_data(
                    item, color, item_text, pred_conf, pred_kpt_count, cluster_id
                )
                self.labeling_list.addItem(item)
            else:
                self._apply_item_data(
                    item, color, item_text, pred_conf, pred_kpt_count, cluster_id
                )
            return

        # Fallback: scan both lists for the item by index.
        for list_widget in (self.labeling_list, self.frame_list):
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.data(Qt.UserRole) == idx:
                    self._apply_item_data(
                        item, color, item_text, pred_conf, pred_kpt_count, cluster_id
                    )
                    return

    def _schedule_frame_item_refresh(self, idx: int) -> None:
        """Defer frame-list mutations until after Qt finishes selection handling."""

        QTimer.singleShot(0, lambda frame_idx=idx: self._update_frame_item(frame_idx))

    def _clear_conf_display(self) -> None:
        if self._list_items:
            for item in self._list_items.values():
                item.setData(FrameListDelegate.CONF_ROLE, None)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, None)
            return
        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            if item is not None:
                item.setData(FrameListDelegate.CONF_ROLE, None)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, None)
        for i in range(self.frame_list.count()):
            item = self.frame_list.item(i)
            if item is not None:
                item.setData(FrameListDelegate.CONF_ROLE, None)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, None)

    def _rebuild_path_index(self):
        self._path_to_index: Dict[str, int] = {}
        for i, p in enumerate(self.image_paths):
            try:
                self._path_to_index[str(p)] = i
                self._path_to_index[str(p.resolve())] = i
            except Exception:
                self._path_to_index[str(p)] = i
        self._build_source_index()

    def _build_source_index(self):
        """Rebuild _source_map and _image_source from project.sources."""
        self._source_map = {s.source_id: s for s in self.project.sources}
        self._image_source = {}
        for src in self.project.sources:
            try:
                for p in list_images(src.images_dir):
                    self._image_source[str(p)] = src.source_id
                    try:
                        self._image_source[str(p.resolve())] = src.source_id
                    except Exception:
                        pass
            except Exception:
                pass

    def _source_id_for_path(self, img_path: Path) -> str:
        return self._image_source.get(str(img_path)) or self._image_source.get(
            str(img_path.resolve()) if img_path.exists() else str(img_path),
            "",
        )

    def _source_id_for_index(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.image_paths):
            return ""
        return self._source_id_for_path(self.image_paths[idx])

    def _resolve_initial_source_id(self, preferred_path: Optional[Path] = None) -> str:
        if not self.project.sources:
            return ""

        preferred_id = ""
        if preferred_path is not None:
            preferred_id = self._source_id_for_path(preferred_path)
        if preferred_id and preferred_id in self._source_map:
            return preferred_id
        if (
            self.project.last_source_id
            and self.project.last_source_id in self._source_map
        ):
            return self.project.last_source_id
        if self.image_paths:
            current_id = self._source_id_for_index(
                max(0, min(self.current_index, len(self.image_paths) - 1))
            )
            if current_id and current_id in self._source_map:
                return current_id
        return self.project.sources[0].source_id

    def _matches_current_source(self, idx: int) -> bool:
        if not self.project.sources or not self.current_source_id:
            return True
        return self._source_id_for_index(idx) == self.current_source_id

    def _first_index_for_current_source(self) -> Optional[int]:
        if not self.image_paths:
            return None
        for idx in range(len(self.image_paths)):
            if self._matches_current_source(idx):
                return idx
        return None

    def _refresh_source_selector(self) -> None:
        if not hasattr(self, "source_combo"):
            return

        self._suppress_source_combo_events = True
        self.source_combo.clear()

        if not self.project.sources:
            self.current_source_id = ""
            self.source_combo.addItem("No sources added", "")
            self.source_combo.setEnabled(False)
            if self.image_paths:
                self.source_summary.setText(
                    f"{len(self.image_paths):,} project frame(s)"
                )
            else:
                self.source_summary.setText("Use Source Manager to add folders.")
            self._suppress_source_combo_events = False
            return

        counts: Dict[str, int] = {}
        for img_path in self.image_paths:
            source_id = self._source_id_for_path(img_path)
            if source_id:
                counts[source_id] = counts.get(source_id, 0) + 1

        if self.current_source_id not in self._source_map:
            self.current_source_id = self._resolve_initial_source_id()

        for src in self.project.sources:
            label = src.description or src.dataset_root.name or src.source_id
            self.source_combo.addItem(
                f"{label} ({counts.get(src.source_id, 0):,})",
                src.source_id,
            )

        combo_index = self.source_combo.findData(self.current_source_id)
        if combo_index < 0:
            combo_index = 0
            self.current_source_id = str(self.source_combo.itemData(0) or "")
        self.source_combo.setCurrentIndex(combo_index)
        self.source_combo.setEnabled(True)
        self.project.last_source_id = self.current_source_id

        current_source = self._source_map.get(self.current_source_id)
        if current_source is None:
            self.source_summary.setText("Select a source to browse its frames.")
        else:
            self.source_summary.setText(
                f"{counts.get(current_source.source_id, 0):,} frame(s) from {current_source.images_dir}"
            )
        self._suppress_source_combo_events = False

    def _set_current_source_for_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self.image_paths):
            return False
        source_id = self._source_id_for_index(idx)
        if not source_id or source_id == self.current_source_id:
            return False
        self.current_source_id = source_id
        self.project.last_source_id = source_id
        if hasattr(self, "source_combo"):
            self._suppress_source_combo_events = True
            combo_index = self.source_combo.findData(source_id)
            if combo_index >= 0:
                self.source_combo.setCurrentIndex(combo_index)
            self._suppress_source_combo_events = False
            self._refresh_source_selector()
        return True

    def _on_source_selector_changed(self, _row: int) -> None:
        if self._suppress_source_combo_events:
            return
        selected_id = str(self.source_combo.currentData() or "")
        if selected_id == self.current_source_id:
            return

        self.current_source_id = selected_id
        self.project.last_source_id = selected_id
        self._refresh_source_selector()
        self._populate_frames()
        if self._select_frame_in_list(self.current_index, trigger_load=False):
            return
        first_idx = self._first_index_for_current_source()
        if first_idx is not None:
            self.load_frame(first_idx)
            self._select_frame_in_list(first_idx, trigger_load=False)

    def _refresh_after_source_change(self, preferred_source_id: str = "") -> None:
        current_path = None
        if 0 <= self.current_index < len(self.image_paths):
            current_path = self.image_paths[self.current_index]

        old_labeling_paths = {
            self.image_paths[idx]
            for idx in self.labeling_frames
            if 0 <= idx < len(self.image_paths)
        }

        self.image_paths = build_image_list(self.project)
        self._rebuild_path_index()
        self.labeling_frames = set()
        for idx, img_path in enumerate(self.image_paths):
            if img_path in old_labeling_paths or self._is_labeled(img_path):
                self.labeling_frames.add(idx)

        self._frame_cache.clear()
        self._undo_stack.clear()
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._cluster_ids_cache = None
        self._cluster_ids_mtime = None

        if preferred_source_id and preferred_source_id in self._source_map:
            self.current_source_id = preferred_source_id
            self.project.last_source_id = preferred_source_id
        else:
            self.current_source_id = self._resolve_initial_source_id(current_path)

        self._refresh_source_selector()

        if not self.image_paths:
            self.current_index = 0
            self._ann = None
            self._dirty = False
            self._populate_frames()
            self.labeling_list.clearSelection()
            self.frame_list.clearSelection()
            self.lbl_info.setText("Add a source to start labeling.")
            self.statusBar().showMessage("Project has no source images.", 3000)
            self._show_canvas_logo_placeholder()
            self.save_project()
            return

        if current_path and str(current_path) in self._path_to_index:
            self.current_index = self._path_to_index[str(current_path)]
        else:
            fallback = self._first_index_for_current_source()
            self.current_index = fallback if fallback is not None else 0

        self.load_frame(self.current_index)
        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)
        self.save_project()

    def _get_pred_conf_for_indices(
        self, indices: List[int]
    ) -> Dict[int, Optional[float]]:
        self._ensure_pred_conf_cache()
        out: Dict[int, Optional[float]] = {}
        for idx in indices:
            img_path = self.image_paths[idx]
            key = str(img_path)
            val = self._pred_conf_map.get(key)
            if val is None:
                try:
                    val = self._pred_conf_map.get(str(Path(img_path).resolve()))
                except Exception:
                    val = None
            out[idx] = val
        return out

    def _get_pred_kpt_count_for_indices(
        self, indices: List[int]
    ) -> Dict[int, Optional[int]]:
        self._ensure_pred_conf_cache()
        out: Dict[int, Optional[int]] = {}
        for idx in indices:
            img_path = self.image_paths[idx]
            key = str(img_path)
            val = self._pred_kpt_count_map.get(key)
            if val is None:
                try:
                    val = self._pred_kpt_count_map.get(str(Path(img_path).resolve()))
                except Exception:
                    val = None
            out[idx] = val
        return out

    def _ensure_cluster_ids_cache(self) -> None:
        csv_path = self.project.out_root / "posekit" / "clusters" / "clusters.csv"
        if not csv_path.exists():
            self._cluster_ids_cache = None
            self._cluster_ids_mtime = None
            return
        try:
            mtime = float(csv_path.stat().st_mtime)
        except Exception:
            mtime = None
        refreshed = not (
            self._cluster_ids_cache is not None
            and self._cluster_ids_mtime == mtime
            and len(self._cluster_ids_cache) == len(self.image_paths)
        )
        if not refreshed:
            return
        cluster_ids = self._load_cluster_ids_from_csv(csv_path)
        self._cluster_ids_cache = cluster_ids
        self._cluster_ids_mtime = mtime
        if not self._suppress_list_rebuild and self._list_items:
            self._refresh_cluster_roles_all()

    def _get_cluster_id_for_indices(
        self, indices: List[int]
    ) -> Dict[int, Optional[int]]:
        self._ensure_cluster_ids_cache()
        out: Dict[int, Optional[int]] = {}
        for idx in indices:
            cid = None
            if self._cluster_ids_cache and idx < len(self._cluster_ids_cache):
                cid_val = self._cluster_ids_cache[idx]
                cid = cid_val if cid_val is not None and cid_val >= 0 else None
            out[idx] = cid
        return out

    def _refresh_cluster_roles_all(self) -> None:
        if not self._cluster_ids_cache:
            return
        for idx, item in self._list_items.items():
            if idx < 0 or idx >= len(self.image_paths):
                continue
            cid_val = None
            if idx < len(self._cluster_ids_cache):
                raw = self._cluster_ids_cache[idx]
                cid_val = raw if raw is not None and raw >= 0 else None
            item.setData(FrameListDelegate.CLUSTER_ROLE, cid_val)

    def _pred_cache_key_for(self, model: Optional[Path], backend: str) -> Optional[str]:
        if not model:
            return None
        try:
            sig = self.infer._model_sig(model, backend)
        except Exception:
            sig = None
        conf_thr = self._pred_conf_default()
        runtime = self._pred_runtime_flavor()
        exported_model = self._get_pred_exported_model_silent()
        exported_sig = None
        if exported_model is not None:
            try:
                stat = exported_model.stat()
                token = f"{exported_model.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
            except OSError:
                token = str(exported_model)
            exported_sig = hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
        return (
            f"{backend}|{model}|{sig}|runtime={runtime}|"
            f"exported={exported_sig}|thr={conf_thr:.6f}"
        )

    def _current_pred_cache_key(self) -> Optional[str]:
        backend = self._pred_backend()
        model = self._pred_cache_model(self._get_pred_model_silent(), backend)
        cache_backend = self._pred_cache_backend(backend)
        return self._pred_cache_key_for(model, cache_backend)

    def _ensure_pred_conf_cache(self) -> None:
        key = self._current_pred_cache_key()
        if not key:
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            return
        if key == self._pred_conf_cache_key and self._pred_conf_complete:
            return
        model = self._get_pred_model_silent()
        if not model:
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            return
        backend = self._pred_backend()
        model = self._pred_cache_model(model, backend)
        cache_backend = self._pred_cache_backend(backend)
        if model is None:
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            return
        preds_cache = self.infer.load_cache(model, backend=cache_backend)
        self._pred_conf_cache_key = key
        conf_map, kpt_map = self._build_pred_stats_maps(preds_cache)
        self._pred_conf_map = conf_map
        self._pred_kpt_count_map = kpt_map
        self._pred_conf_complete = True

    def _build_pred_stats_maps(
        self, preds_cache: Dict[str, List[Tuple[float, float, float]]]
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[int]]]:
        conf_map: Dict[str, Optional[float]] = {}
        kpt_map: Dict[str, Optional[int]] = {}
        conf_thr = self._pred_conf_default()
        for path, pred_list in preds_cache.items():
            confs = []
            for p in pred_list:
                if len(p) < 3 or not np.isfinite(p[2]):
                    continue
                confs.append(float(np.clip(p[2], 0.0, 1.0)))
            val = float(np.mean(confs)) if confs else None
            if confs:
                kpt_count = int(sum(1 for c in confs if c > conf_thr))
            else:
                kpt_count = 0 if pred_list else None
            conf_map[path] = val
            kpt_map[path] = kpt_count
            try:
                resolved = str(Path(path).resolve())
                conf_map[resolved] = val
                kpt_map[resolved] = kpt_count
            except Exception:
                pass
        return conf_map, kpt_map

    def _update_pred_conf_map_from_preds(
        self,
        preds: Dict[str, List[Tuple[float, float, float]]],
        model: Optional[Path] = None,
        backend: Optional[str] = None,
    ) -> None:
        if not preds:
            return
        if backend is None:
            backend = self._pred_backend()
        if model is None:
            model = self._get_pred_model_silent()
        key = self._pred_cache_key_for(model, backend)
        if not key:
            return
        if key != self._pred_conf_cache_key:
            self._pred_conf_cache_key = key
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
        conf_thr = self._pred_conf_default()
        for path, pred_list in preds.items():
            confs = []
            for p in pred_list:
                if len(p) < 3 or not np.isfinite(p[2]):
                    continue
                confs.append(float(np.clip(p[2], 0.0, 1.0)))
            val = float(np.mean(confs)) if confs else None
            if confs:
                kpt_count = int(sum(1 for c in confs if c > conf_thr))
            else:
                kpt_count = 0 if pred_list else None
            self._pred_conf_map[path] = val
            self._pred_kpt_count_map[path] = kpt_count
            try:
                resolved = str(Path(path).resolve())
                self._pred_conf_map[resolved] = val
                self._pred_kpt_count_map[resolved] = kpt_count
            except Exception:
                pass

    def _rebuild_kpt_list(self):
        self.kpt_list.clear()
        for i, nm in enumerate(self.project.keypoint_names):
            self.kpt_list.addItem(f"{i}: {nm}")
        self.kpt_list.setCurrentRow(min(self.current_kpt, self.kpt_list.count() - 1))

    def _update_info(self):
        img_path = self.image_paths[self.current_index]
        labeled = self._is_labeled(img_path)
        done = sum(1 for kp in self._ann.kpts if kp.v > 0) if self._ann else 0
        total = len(self.project.keypoint_names)
        self.lbl_info.setText(
            f"{img_path.name}\n"
            f"Frame {self.current_index + 1}/{len(self.image_paths)} | "
            f"Keypoints {done}/{total} | "
            f"{'LABELED' if labeled else 'unlabeled'}"
        )
        self.statusBar().showMessage(f"{img_path.name} ({done}/{total} kpts)")

    # ----- load frame -----
    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def _kpts_to_save_space(
        self, kpts: "List[Keypoint]", img_path: "Path"
    ) -> "tuple[List[Keypoint], int, int]":
        """Return ``(kpts_px, save_w, save_h)`` in image pixel space."""
        return list(kpts), self._img_wh[0], self._img_wh[1]

    def _clone_ann(self, ann: FrameAnn) -> FrameAnn:
        return FrameAnn(
            cls=int(ann.cls),
            bbox_xyxy=ann.bbox_xyxy,
            kpts=[Keypoint(kp.x, kp.y, kp.v) for kp in ann.kpts],
        )

    def _cache_current_frame(self):
        if self._ann is None:
            return
        self._frame_cache[self.current_index] = self._clone_ann(self._ann)
        logger.debug(
            "Cached frame %d (kpts=%d)", self.current_index, len(self._ann.kpts)
        )
        # Update the frame item display to reflect cache state
        self._update_frame_item(self.current_index)

    def _prime_cache_for_labeling(self):
        if not self.labeling_frames:
            return
        logger.debug("Priming cache for %d labeling frames", len(self.labeling_frames))
        for idx in sorted(self.labeling_frames):
            if idx in self._frame_cache:
                continue
            try:
                ann = self._load_ann_from_disk(idx)
            except Exception:
                continue
            self._frame_cache[idx] = self._clone_ann(ann)
        logger.debug("Cache size after prime: %d", len(self._frame_cache))

    def _load_ann_from_disk(self, idx: int) -> FrameAnn:
        img_path = self.image_paths[idx]
        # Labels on disk are stored in image pixel coordinates.
        # Read raw dims directly so denormalization is correct.
        raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if raw is None:
            raw = self._read_image(img_path)
        orig_h, orig_w = raw.shape[:2]
        k = len(self.project.keypoint_names)
        kpts = [Keypoint(0.0, 0.0, 0) for _ in range(k)]
        bbox = None
        cls = int(self.class_combo.currentIndex())

        loaded = load_yolo_pose_label(self._label_path_for(img_path), k)
        if loaded is not None:
            cls_loaded, kpts_norm, bbox_cxcywh = loaded
            cls = int(cls_loaded)

            kpts = []
            for kp in kpts_norm:
                if kp.v == 0:
                    kpts.append(Keypoint(0.0, 0.0, 0))
                else:
                    x_px = kp.x * orig_w
                    y_px = kp.y * orig_h
                    kpts.append(Keypoint(float(x_px), float(y_px), int(kp.v)))

            if bbox_cxcywh is not None:
                cx, cy, bw, bh = bbox_cxcywh
                cx *= orig_w
                cy *= orig_h
                bw *= orig_w
                bh *= orig_h
                x1 = cx - bw / 2
                x2 = cx + bw / 2
                y1 = cy - bh / 2
                y2 = cy + bh / 2
                bbox = (x1, y1, x2, y2)

        return FrameAnn(cls=cls, bbox_xyxy=bbox, kpts=kpts)

    def load_frame(self: object, idx: int) -> object:
        """load_frame method documentation."""
        if not self.image_paths:
            return
        idx = max(0, min(idx, len(self.image_paths) - 1))

        logger.debug("Load frame requested: idx=%d", idx)

        # Cache current annotations before switching
        self._cache_current_frame()

        # Commit any pending changes synchronously before leaving the current frame.
        # load_frame resets _dirty = False unconditionally, which would cancel the
        # delayed autosave timer without actually writing the label to disk.  Saving
        # here (before current_index changes) ensures the label is on disk and the
        # frame can be moved to the labeling list.
        _prev_index = self.current_index
        _was_dirty = self._dirty
        if _was_dirty and self._ann is not None:
            self.save_current(refresh_ui=False)

        self.current_index = idx
        source_changed = False
        if hasattr(self, "_set_current_source_for_index"):
            source_changed = self._set_current_source_for_index(idx)
        img_path = self.image_paths[idx]
        self._img_bgr = self._read_image(img_path)
        self._img_display = None
        h, w = self._img_bgr.shape[:2]
        self._img_wh = (w, h)

        if idx in self._frame_cache:
            cached = self._frame_cache[idx]
            cls = int(cached.cls)
            kpts = [Keypoint(kp.x, kp.y, kp.v) for kp in cached.kpts]
            bbox = cached.bbox_xyxy
            logger.debug("Loaded frame %d from cache", idx)
        else:
            ann = self._load_ann_from_disk(idx)
            cls = int(ann.cls)
            kpts = ann.kpts
            bbox = ann.bbox_xyxy
            if self.mode == "keypoint":
                self._frame_cache[idx] = self._clone_ann(ann)
            logger.debug(
                "Loaded frame %d from disk (cached=%s)", idx, self.mode == "keypoint"
            )

        self.class_combo.blockSignals(True)
        self.class_combo.setCurrentIndex(max(0, min(cls, self.class_combo.count() - 1)))
        self.class_combo.blockSignals(False)

        self._ann = FrameAnn(cls=cls, bbox_xyxy=bbox, kpts=kpts)
        self._dirty = False

        # Clear undo stack when changing frames to prevent cross-frame undo
        self._undo_stack.clear()

        # In keypoint mode, auto-switch to first unlabeled keypoint
        if self.mode == "keypoint":
            first_unlabeled = None
            for i, kp in enumerate(self._ann.kpts):
                if kp.v == 0:
                    first_unlabeled = i
                    break

            if first_unlabeled is not None and first_unlabeled != self.current_kpt:
                # Notify user that we're switching keypoints
                old_kpt_name = self.project.keypoint_names[self.current_kpt]
                new_kpt_name = self.project.keypoint_names[first_unlabeled]
                self.statusBar().showMessage(
                    f"Switched from '{old_kpt_name}' to unlabeled '{new_kpt_name}'",
                    3000,
                )
                self.current_kpt = first_unlabeled
                self.kpt_list.setCurrentRow(first_unlabeled)

        self.canvas.set_current_keypoint(self.current_kpt)
        self._refresh_canvas_image()
        self._rebuild_canvas()
        self._update_info()
        self._load_metadata_ui()

        # If the previous frame was saved above, update its entry in the frame lists
        # after the selection event unwinds. Moving list items inline from a
        # currentRowChanged handler can crash PySide/Qt on macOS/Python 3.13.
        if _was_dirty:
            self._schedule_frame_item_refresh(_prev_index)

        if source_changed:
            self._populate_frames()
            self._select_frame_in_list(self.current_index, trigger_load=False)

        return

    # ----- events -----
    def _on_labeling_frame_selected(self, row: int):
        if row < 0:
            return
        # Get the actual index from the item
        item = self.labeling_list.item(row)
        if item:
            actual_idx = item.data(Qt.UserRole)
            if actual_idx is not None:
                logger.debug("Labeling frame selected: row=%s idx=%s", row, actual_idx)
                # Ensure frame is in labeling_frames set (for navigation to work)
                self.labeling_frames.add(actual_idx)
                self._maybe_autosave()
                self.load_frame(actual_idx)
                # Deselect all frames list
                self.frame_list.clearSelection()

    def _on_all_frame_selected(self, row: int):
        if row < 0:
            return
        # Get the actual index from the item
        item = self.frame_list.item(row)
        if item:
            actual_idx = item.data(Qt.UserRole)
            if actual_idx is not None:
                logger.debug("All frame selected: row=%s idx=%s", row, actual_idx)
                self._maybe_autosave()
                self.load_frame(actual_idx)
                # Deselect labeling list
                self.labeling_list.clearSelection()

    def _on_labeling_list_changed(self, parent, start, end, dest, row):
        """Update labeling_frames when items are moved."""
        if self._suppress_list_rebuild:
            logger.debug("Labeling list rowsMoved suppressed")
            return
        self._rebuild_labeling_set()

    def _on_all_list_changed(self, parent, start, end, dest, row):
        """Update labeling_frames when items are moved."""
        if self._suppress_list_rebuild:
            logger.debug("All list rowsMoved suppressed")
            return
        self._rebuild_labeling_set()

    def _rebuild_labeling_set(self):
        """Rebuild the labeling_frames set from current list contents."""
        logger.debug("Rebuild labeling set (before): %s", sorted(self.labeling_frames))
        self.labeling_frames.clear()
        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            idx = item.data(Qt.UserRole)
            if idx is not None:
                self.labeling_frames.add(idx)

        # Ensure labeled frames are always in labeling set (prevent drag-out)
        for idx, img_path in enumerate(self.image_paths):
            if self._is_labeled(img_path):
                self.labeling_frames.add(idx)

        # Repopulate to enforce labeled frames stay in labeling list
        self._populate_frames()
        logger.debug("Rebuild labeling set (after): %s", sorted(self.labeling_frames))

    def _move_unlabeled_to_labeling(self):
        """Move unlabeled frames from the current source into the labeling set."""
        for idx, img_path in enumerate(self.image_paths):
            if (
                self._matches_current_source(idx)
                and not self._is_labeled(img_path)
                and idx not in self.labeling_frames
            ):
                self.labeling_frames.add(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)

    def _move_unlabeled_to_all(self):
        """Move unlabeled frames from the current source back to the source browser."""
        unlabeled_to_remove = []
        for idx in list(self.labeling_frames):
            if self._matches_current_source(idx) and not self._is_labeled(
                self.image_paths[idx]
            ):
                unlabeled_to_remove.append(idx)
        for idx in unlabeled_to_remove:
            self.labeling_frames.remove(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)

    def _add_random_to_labeling(self):
        """Add random unlabeled frames from All Frames list to labeling set."""
        import random

        count = self.spin_random_count.value()

        # Get all unlabeled frames from the current source browser.
        candidates = []
        for idx, img_path in enumerate(self.image_paths):
            if (
                self._matches_current_source(idx)
                and not self._is_labeled(img_path)
                and idx not in self.labeling_frames
            ):
                candidates.append(idx)

        if not candidates:
            QMessageBox.information(
                self, "No frames", "No unlabeled frames available in All Frames list."
            )
            return

        # Randomly select up to 'count' frames
        to_add = random.sample(candidates, min(count, len(candidates)))
        for idx in to_add:
            self.labeling_frames.add(idx)

        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)
        QMessageBox.information(
            self, "Added frames", f"Added {len(to_add)} frames to labeling set."
        )

    def _on_kpt_selected(self, row: int):
        if row < 0:
            return
        self.current_kpt = row
        self.canvas.set_current_keypoint(row)

    def _update_mode(self):
        prev_mode = self.mode
        self.mode = "frame" if self.rb_frame.isChecked() else "keypoint"
        logger.debug("Mode update: %s -> %s", prev_mode, self.mode)
        if self.mode == "keypoint" and prev_mode != "keypoint":
            # Prime cache for keypoint-by-keypoint workflow
            logger.debug(
                "Priming cache for keypoint mode. labeling_frames=%d",
                len(self.labeling_frames),
            )
            self._cache_current_frame()
            self._prime_cache_for_labeling()

    def _toggle_enhancement(self, checked: bool):
        self.project.enhance_enabled = bool(checked)
        if self.cb_enhance.isChecked() != self.project.enhance_enabled:
            self.cb_enhance.setChecked(self.project.enhance_enabled)
        if self.act_enhance.isChecked() != self.project.enhance_enabled:
            self.act_enhance.setChecked(self.project.enhance_enabled)
        self._img_display = None
        self._refresh_canvas_image()
        self.save_project()

    def _update_kpt_size(self, value: float):
        self.project.kpt_radius = float(value)
        self.canvas.set_kpt_radius(self.project.kpt_radius)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_label_size(self, value: int):
        self.project.label_font_size = int(value)
        self.canvas.set_label_font_size(self.project.label_font_size)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_kpt_opacity(self, value: float):
        self.project.kpt_opacity = float(value)
        self.canvas.set_kpt_opacity(self.project.kpt_opacity)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_edge_opacity(self, value: float):
        self.project.edge_opacity = float(value)
        self.canvas.set_edge_opacity(self.project.edge_opacity)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_edge_width(self, value: float):
        self.project.edge_width = float(value)
        self.canvas.set_edge_width(self.project.edge_width)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def fit_to_view(self: object) -> object:
        """Fit image to view."""
        self.canvas.fit_to_view()

    def _get_display_image(self) -> Optional[np.ndarray]:
        if self._img_bgr is None:
            return None
        if self._img_display is not None:
            return self._img_display
        if not self.project.enhance_enabled:
            self._img_display = self._img_bgr
            return self._img_display

        try:
            self._img_display = enhance_for_pose(
                self._img_bgr,
                clahe_clip=self.project.clahe_clip,
                clahe_grid=self.project.clahe_grid,
                sharpen_amt=self.project.sharpen_amt,
                blur_sigma=self.project.blur_sigma,
            )
        except Exception:
            self._img_display = self._img_bgr
        return self._img_display

    def _refresh_canvas_image(self):
        img = self._get_display_image()
        if img is None:
            return
        self.canvas.set_image(img)

    def _open_enhancement_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Enhancement Settings")
        layout = QFormLayout(dlg)

        clip = QDoubleSpinBox()
        clip.setRange(0.1, 10.0)
        clip.setSingleStep(0.1)
        clip.setValue(float(self.project.clahe_clip))

        grid_x = QSpinBox()
        grid_x.setRange(2, 64)
        grid_x.setValue(int(self.project.clahe_grid[0]))

        grid_y = QSpinBox()
        grid_y.setRange(2, 64)
        grid_y.setValue(int(self.project.clahe_grid[1]))

        sharpen = QDoubleSpinBox()
        sharpen.setRange(0.0, 3.0)
        sharpen.setSingleStep(0.1)
        sharpen.setValue(float(self.project.sharpen_amt))

        blur = QDoubleSpinBox()
        blur.setRange(0.0, 5.0)
        blur.setSingleStep(0.1)
        blur.setValue(float(self.project.blur_sigma))

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("X"))
        grid_row.addWidget(grid_x)
        grid_row.addWidget(QLabel("Y"))
        grid_row.addWidget(grid_y)

        layout.addRow("CLAHE clip limit", clip)
        layout.addRow("CLAHE grid size", grid_row)
        layout.addRow("Sharpen amount", sharpen)
        layout.addRow("Blur sigma", blur)

        btns = QHBoxLayout()
        ok = QPushButton("Apply")
        cancel = QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addRow(btns)

        cancel.clicked.connect(dlg.reject)
        ok.clicked.connect(dlg.accept)

        if dlg.exec() != QDialog.Accepted:
            return

        self.project.clahe_clip = float(clip.value())
        self.project.clahe_grid = (int(grid_x.value()), int(grid_y.value()))
        self.project.sharpen_amt = float(sharpen.value())
        self.project.blur_sigma = float(blur.value())
        self._img_display = None
        self._refresh_canvas_image()
        self.save_project()

    def _set_autosave_status(self, text: str):
        if hasattr(self, "status_autosave") and self.status_autosave is not None:
            self.status_autosave.setText(text)

    def _set_saved_status(self):
        ts = datetime.now().strftime("%H:%M:%S")
        self._set_autosave_status(f"Saved {ts}")

    def _set_dirty(self):
        self._dirty = True
        self._set_autosave_status("Unsaved changes…")

    def _mark_dirty(self, *_):
        self._set_dirty()
        self._schedule_autosave()

    # ----- undo -----
    def _snapshot_kpts(self) -> Optional[List[Keypoint]]:
        if self._ann is None:
            return None
        return [Keypoint(kp.x, kp.y, kp.v) for kp in self._ann.kpts]

    def _push_undo(self):
        snap = self._snapshot_kpts()
        if snap is None:
            return
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._undo_max:
            self._undo_stack = self._undo_stack[-self._undo_max :]

    def undo_last(self: object) -> object:
        """undo_last method documentation."""
        if not self._undo_stack or self._ann is None:
            return
        self._ann.kpts = self._undo_stack.pop()
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()

    # ----- edits -----
    def on_place_kpt(self: object, kpt_idx: int, x: float, y: float, v: int) -> object:
        """on_place_kpt method documentation."""
        if self._ann is None:
            return
        logger.debug(
            "Place kpt: idx=%d v=%d at (%.1f, %.1f) mode=%s frame=%d",
            kpt_idx,
            v,
            x,
            y,
            self.mode,
            self.current_index,
        )
        w, h = self._img_wh
        kpt_idx = max(0, min(kpt_idx, len(self._ann.kpts) - 1))
        # In frame-by-frame mode, enforce sequential keypoint placement
        if self.mode == "frame" and kpt_idx > 0:
            # Check if all previous keypoints have been placed
            for i in range(kpt_idx):
                if self._ann.kpts[i].v == 0:
                    QMessageBox.warning(
                        self,
                        "Sequential Labeling",
                        f"Please label keypoint {i} ({self.project.keypoint_names[i]}) before keypoint {kpt_idx}.",
                    )
                    return

        self._push_undo()

        if v == 0:
            self._ann.kpts[kpt_idx] = Keypoint(0.0, 0.0, 0)
        else:
            self._ann.kpts[kpt_idx] = Keypoint(
                x=max(0.0, min(float(w - 1), x)),
                y=max(0.0, min(float(h - 1), y)),
                v=int(v),
            )
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()

        if self.mode == "keypoint":
            # In keypoint mode, keep in-memory cache only
            self._cache_current_frame()
            # Frame item is already updated by _cache_current_frame
            # Find next frame that needs this keypoint
            self._advance_keypoint_mode()
        else:
            # Save immediately after each placement in frame mode
            # Don't refresh UI since we're about to navigate away
            self.save_current(refresh_ui=False)

            # Check if all keypoints are now labeled
            all_labeled = all(kp.v > 0 for kp in self._ann.kpts)

            # In frame mode: if all labeled, advance to next frame
            # Otherwise, jump to first unlabeled keypoint
            if all_labeled:
                self.next_frame(prefer_missing=True)
            else:
                # Find first unlabeled keypoint
                for i, kp in enumerate(self._ann.kpts):
                    if kp.v == 0:
                        self.current_kpt = i
                        self.kpt_list.setCurrentRow(i)
                        self.canvas.set_current_keypoint(i)
                        # Force UI update to prevent race condition with next click
                        QApplication.processEvents()
                        break

    def _advance_keypoint_mode(self):
        """Find next frame/keypoint to label in keypoint-by-keypoint mode."""
        if not self.labeling_frames:
            return

        # Find next frame in labeling set that needs current keypoint
        next_frame_idx = self._find_next_frame_needing_keypoint(self.current_kpt)

        if next_frame_idx is not None:
            # Found a frame that needs this keypoint
            self._select_frame_in_list(next_frame_idx)
        else:
            # All frames have this keypoint - move to next keypoint
            kpt_name = self.project.keypoint_names[self.current_kpt]

            # Check if all frames have all keypoints
            if self._all_frames_fully_labeled():
                QMessageBox.information(
                    self,
                    "Labeling Complete",
                    "All keypoints have been labeled on all frames in the labeling set!",
                )
                return

            # Move to next keypoint
            next_kpt = (self.current_kpt + 1) % len(self.project.keypoint_names)
            self.current_kpt = next_kpt
            self.kpt_list.setCurrentRow(next_kpt)
            self.canvas.set_current_keypoint(next_kpt)
            # Force UI update to prevent race condition
            QApplication.processEvents()

            next_kpt_name = self.project.keypoint_names[next_kpt]
            QMessageBox.information(
                self,
                "Keypoint Complete",
                f"All frames have keypoint '{kpt_name}'.\nMoving to keypoint '{next_kpt_name}'.",
            )

            # Find first frame that needs the new keypoint
            next_frame_idx = self._find_next_frame_needing_keypoint(next_kpt)
            if next_frame_idx is not None:
                self._select_frame_in_list(next_frame_idx)

    def _find_next_frame_needing_keypoint(self, kpt_idx: int) -> Optional[int]:
        """Find next frame in labeling set that doesn't have the specified keypoint."""
        labeling_indices = sorted(self.labeling_frames)

        # Start from current frame and wrap around
        try:
            current_pos = labeling_indices.index(self.current_index)
            search_order = (
                labeling_indices[current_pos + 1 :]
                + labeling_indices[: current_pos + 1]
            )
        except ValueError:
            # Current frame not in labeling set, search all
            search_order = labeling_indices

        for idx in search_order:
            # Check cache first, then disk
            if idx in self._frame_cache:
                ann = self._frame_cache[idx]
                if ann.kpts[kpt_idx].v == 0:
                    return idx
            else:
                # Load from disk to check
                try:
                    ann = self._load_ann_from_disk(idx)
                    if ann.kpts[kpt_idx].v == 0:
                        return idx
                except Exception:
                    # If we can't load, assume it needs the keypoint
                    return idx

        return None

    def _all_frames_fully_labeled(self) -> bool:
        """Check if all frames in labeling set have all keypoints labeled."""
        for idx in self.labeling_frames:
            if idx in self._frame_cache:
                ann = self._frame_cache[idx]
            else:
                try:
                    ann = self._load_ann_from_disk(idx)
                except Exception:
                    return False

            # Check if any keypoint is missing
            if any(kp.v == 0 for kp in ann.kpts):
                return False

        return True

    def _frame_has_missing_labels(self, idx: int) -> bool:
        """Return True if any keypoint is missing for the given frame."""
        if idx in self._frame_cache:
            ann = self._frame_cache[idx]
        else:
            try:
                ann = self._load_ann_from_disk(idx)
            except Exception:
                # If we can't load, assume it needs labeling.
                return True
        return any(kp.v == 0 for kp in ann.kpts)

    def _find_next_frame_with_missing_labels(self) -> Optional[int]:
        """Find next frame in labeling set with any missing keypoints."""
        labeling_indices = sorted(self.labeling_frames)
        if not labeling_indices:
            return None

        try:
            current_pos = labeling_indices.index(self.current_index)
            search_order = (
                labeling_indices[current_pos + 1 :]
                + labeling_indices[: current_pos + 1]
            )
        except ValueError:
            search_order = labeling_indices

        for idx in search_order:
            if self._frame_has_missing_labels(idx):
                return idx
        return None

    def on_move_kpt(self: object, kpt_idx: int, x: float, y: float) -> object:
        """on_move_kpt method documentation."""
        if self._ann is None:
            return
        logger.debug(
            "Move kpt: idx=%d to (%.1f, %.1f) mode=%s frame=%d",
            kpt_idx,
            x,
            y,
            self.mode,
            self.current_index,
        )
        self._push_undo()
        w, h = self._img_wh
        kp = self._ann.kpts[kpt_idx]
        if kp.v == 0:
            kp.v = 2
        kp.x = max(0.0, min(float(w - 1), x))
        kp.y = max(0.0, min(float(h - 1), y))
        self._mark_dirty()
        self._rebuild_canvas()
        self._update_info()
        if self.mode == "keypoint":
            self._cache_current_frame()

    def on_select_kpt(self: object, idx: int) -> object:
        """Called when user clicks on an existing keypoint - make it current."""
        if idx >= 0 and idx < len(self.project.keypoint_names):
            self.current_kpt = idx
            self.kpt_list.setCurrentRow(idx)
            self.canvas.set_current_keypoint(idx)

    def clear_current_keypoint(self: object) -> object:
        """clear_current_keypoint method documentation."""
        self.on_place_kpt(self.current_kpt, 0.0, 0.0, 0)

    def clear_all_keypoints(self: object) -> object:
        """Clear all keypoints from the current frame."""
        if self._ann is None:
            return
        self._push_undo()
        for i in range(len(self._ann.kpts)):
            self._ann.kpts[i] = Keypoint(0.0, 0.0, 0)
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()
        if self.mode == "keypoint":
            self._cache_current_frame()
        else:
            self.save_current()

    # ----- navigation -----
    def _maybe_autosave(self):
        if self.mode == "keypoint":
            # In keypoint mode, keep in-memory cache only
            self._cache_current_frame()
            if self.project.autosave and self._dirty:
                self._schedule_autosave()
            return
        if self.project.autosave and self._dirty:
            self._schedule_autosave()

    def _schedule_autosave(self):
        if not self.project.autosave:
            return
        if self._dirty:
            self._autosave_timer.start(self.autosave_delay_ms)

    def _perform_autosave(self):
        if self.project.autosave and self._dirty:
            self._set_autosave_status("Autosaving…")
            self.save_current()

    def _update_autosave_delay(self, seconds: float):
        self.autosave_delay_ms = int(max(0.5, float(seconds)) * 1000)
        if self._autosave_timer.isActive():
            self._autosave_timer.start(self.autosave_delay_ms)

    def prev_frame(self: object) -> object:
        """prev_frame method documentation."""
        labeling_indices = sorted(self.labeling_frames)
        if not labeling_indices:
            return

        # Refresh frame lists to show updated labeling status
        prev_idx = self.current_index

        try:
            current_pos = labeling_indices.index(self.current_index)
            if current_pos > 0:
                prev_idx_target = labeling_indices[current_pos - 1]
                self._select_frame_in_list(prev_idx_target)
                # Reset to first keypoint in frame mode
                if self.mode == "frame":
                    self.current_kpt = 0
                    self.kpt_list.setCurrentRow(0)
                    self.canvas.set_current_keypoint(0)
        except ValueError:
            # Current not in labeling set, go to last
            if labeling_indices:
                self._select_frame_in_list(labeling_indices[-1])
                if self.mode == "frame":
                    self.current_kpt = 0
                    self.kpt_list.setCurrentRow(0)
                    self.canvas.set_current_keypoint(0)

        # Refresh lists after navigation to show updated status
        if prev_idx != self.current_index:
            # Block signals to prevent triggering frame load during list rebuild
            self.labeling_list.blockSignals(True)
            self.frame_list.blockSignals(True)
            self._populate_frames()
            self.labeling_list.blockSignals(False)
            self.frame_list.blockSignals(False)
            # Restore selection without triggering load (already loaded)
            for i in range(self.labeling_list.count()):
                item = self.labeling_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.labeling_list.setCurrentRow(i)
                    break

    def _toggle_kpt_visibility(self, kpt_idx: int):
        if self._ann is None:
            return
        if kpt_idx < 0 or kpt_idx >= len(self._ann.kpts):
            return
        kp = self._ann.kpts[kpt_idx]
        if kp.v <= 0:
            return
        self._push_undo()
        if kp.v == 2:
            kp.v = 1
        elif kp.v == 1:
            kp.v = 0
            kp.x = 0.0
            kp.y = 0.0
        else:
            kp.v = 2
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()
        if self.mode == "keypoint":
            self._cache_current_frame()
            for i in range(self.frame_list.count()):
                item = self.frame_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.frame_list.setCurrentRow(i)
                    break

    def next_frame(self: object, prefer_missing: bool = False) -> object:
        """next_frame method documentation."""
        labeling_indices = sorted(self.labeling_frames)
        logger.debug(
            "next_frame: current=%d labeling_frames=%s cache_size=%d",
            self.current_index,
            labeling_indices,
            len(self._frame_cache),
        )
        if not labeling_indices:
            logger.debug("next_frame: no labeling frames, returning")
            return

        # Refresh frame lists to show updated labeling status
        prev_idx = self.current_index

        if prefer_missing and self.mode == "frame":
            next_missing = self._find_next_frame_with_missing_labels()
            if next_missing is not None:
                self._select_frame_in_list(next_missing)
                self.current_kpt = 0
                self.kpt_list.setCurrentRow(0)
                self.canvas.set_current_keypoint(0)
            else:
                self.statusBar().showMessage(
                    "All frames in labeling set are fully labeled.", 2000
                )
                return
        else:
            try:
                current_pos = labeling_indices.index(self.current_index)
                if current_pos < len(labeling_indices) - 1:
                    next_idx = labeling_indices[current_pos + 1]
                    self._select_frame_in_list(next_idx)
                    # Reset to first keypoint in frame mode
                    if self.mode == "frame":
                        self.current_kpt = 0
                        self.kpt_list.setCurrentRow(0)
                        self.canvas.set_current_keypoint(0)
            except ValueError:
                # Current not in labeling set, go to first
                if labeling_indices:
                    self._select_frame_in_list(labeling_indices[0])
                    if self.mode == "frame":
                        self.current_kpt = 0
                        self.kpt_list.setCurrentRow(0)
                        self.canvas.set_current_keypoint(0)

        # Refresh lists after navigation to show updated status
        if prev_idx != self.current_index:
            # Block signals to prevent triggering frame load during list rebuild
            self.labeling_list.blockSignals(True)
            self.frame_list.blockSignals(True)
            self._populate_frames()
            self.labeling_list.blockSignals(False)
            self.frame_list.blockSignals(False)
            # Restore selection without triggering load (already loaded)
            for i in range(self.labeling_list.count()):
                item = self.labeling_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.labeling_list.setCurrentRow(i)
                    break
            for i in range(self.frame_list.count()):
                item = self.frame_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.frame_list.setCurrentRow(i)
                    break

    def _select_frame_in_list(self, idx: int, trigger_load: bool = True) -> bool:
        """Select a frame by its actual index in the appropriate list."""
        # Check labeling list first
        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            if item.data(Qt.UserRole) == idx:
                if trigger_load:
                    self.labeling_list.setCurrentRow(i)
                else:
                    self.labeling_list.blockSignals(True)
                    self.frame_list.blockSignals(True)
                    self.frame_list.clearSelection()
                    self.labeling_list.setCurrentRow(i)
                    self.labeling_list.blockSignals(False)
                    self.frame_list.blockSignals(False)
                return True
        # Check all frames list
        for i in range(self.frame_list.count()):
            item = self.frame_list.item(i)
            if item.data(Qt.UserRole) == idx:
                if trigger_load:
                    self.frame_list.setCurrentRow(i)
                else:
                    self.labeling_list.blockSignals(True)
                    self.frame_list.blockSignals(True)
                    self.labeling_list.clearSelection()
                    self.frame_list.setCurrentRow(i)
                    self.labeling_list.blockSignals(False)
                    self.frame_list.blockSignals(False)
                return True
        return False

    def prev_keypoint(self: object) -> object:
        """prev_keypoint method documentation."""
        if self.current_kpt > 0:
            self.current_kpt -= 1
            self.kpt_list.setCurrentRow(self.current_kpt)

    def next_keypoint(self: object) -> object:
        """next_keypoint method documentation."""
        if self.current_kpt < self.kpt_list.count() - 1:
            self.current_kpt += 1
            self.kpt_list.setCurrentRow(self.current_kpt)

    def next_unlabeled(self: object) -> object:
        """next_unlabeled method documentation."""
        labeling_indices = sorted(self.labeling_frames)
        if not labeling_indices:
            QMessageBox.information(
                self, "No labeling frames", "No frames in labeling set."
            )
            return

        # Find current position in labeling set
        try:
            current_pos = labeling_indices.index(self.current_index)
            search_start = current_pos + 1
        except ValueError:
            search_start = 0

        # Search forward in labeling frames
        for i in range(search_start, len(labeling_indices)):
            idx = labeling_indices[i]
            if not self._is_labeled(self.image_paths[idx]):
                self._maybe_autosave()
                self._select_frame_in_list(idx)
                return

        QMessageBox.information(
            self,
            "Done",
            "No unlabeled frames found in labeling set after current frame.",
        )

    # ----- save / export -----
    def save_current(self: object, refresh_ui: object = True) -> object:
        """save_current method documentation."""
        if self._ann is None:
            return
        # Keep cache in sync
        self._cache_current_frame()
        logger.debug(
            "Save current frame=%d refresh_ui=%s", self.current_index, refresh_ui
        )
        img_path = self.image_paths[self.current_index]
        label_path = self._label_path_for(img_path)

        cls = int(self.class_combo.currentIndex())
        self._ann.cls = cls

        # Convert keypoints to image pixel space before saving.
        kpts_save, w, h = self._kpts_to_save_space(self._ann.kpts, img_path)

        bbox = compute_bbox_from_kpts(kpts_save, self.project.bbox_pad_frac, w, h)

        save_yolo_pose_label(
            label_path=label_path,
            cls=cls,
            img_w=w,
            img_h=h,
            kpts_px=kpts_save,
            bbox_xyxy_px=bbox,
            pad_frac=self.project.bbox_pad_frac,
        )
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
        self._dirty = False

        # Only refresh UI if we're staying on the current frame
        if refresh_ui:
            self._populate_frames()
            self._select_frame_in_list(self.current_index)

        self.statusBar().showMessage(f"Saved: {label_path.name}", 2000)
        self._set_saved_status()
        self.save_project()

    def save_all_labeling_frames(self: object) -> object:
        """Save all labeling frames to disk using current in-memory state."""
        if not self.labeling_frames:
            return

        logger.debug("Save all labeling frames: count=%d", len(self.labeling_frames))

        # Cache current frame state before batch save
        self._cache_current_frame()

        current_idx = self.current_index
        saved_count = 0

        for idx in sorted(self.labeling_frames):
            ann = None
            if idx == current_idx and self._ann is not None:
                ann = self._clone_ann(self._ann)
            elif idx in self._frame_cache:
                ann = self._clone_ann(self._frame_cache[idx])
            else:
                ann = self._load_ann_from_disk(idx)

            logger.debug(
                "Saving frame %d (from %s)",
                idx,
                (
                    "current"
                    if idx == current_idx
                    else ("cache" if idx in self._frame_cache else "disk")
                ),
            )

            img_path = self.image_paths[idx]

            # Convert keypoints to image pixel space for saving.
            kpts_save, w, h = self._kpts_to_save_space(ann.kpts, img_path)

            bbox = compute_bbox_from_kpts(kpts_save, self.project.bbox_pad_frac, w, h)
            save_yolo_pose_label(
                label_path=self._label_path_for(img_path),
                cls=int(ann.cls),
                img_w=w,
                img_h=h,
                kpts_px=kpts_save,
                bbox_xyxy_px=bbox,
                pad_frac=self.project.bbox_pad_frac,
            )
            saved_count += 1

        self._dirty = False
        self._populate_frames()
        self._select_frame_in_list(current_idx)
        self.statusBar().showMessage(f"Saved {saved_count} labeling frames", 2000)
        self._set_saved_status()
        self.save_project()

    def open_skeleton_editor(self: object) -> object:
        """open_skeleton_editor method documentation."""
        old_kpts = list(self.project.keypoint_names)
        dlg = SkeletonEditorDialog(
            self.project.keypoint_names,
            self.project.skeleton_edges,
            self,
            default_dir=get_default_skeleton_dir(),
        )
        if dlg.exec() == QDialog.Accepted:
            names, edges = dlg.get_result()

            if names != old_kpts:
                resp = QMessageBox.question(
                    self,
                    "Update keypoints",
                    "Keypoints changed in Skeleton Editor.\n"
                    "Migrate existing label files to the new layout?",
                )
                if resp == QMessageBox.Yes:
                    modified, total = migrate_labels_keypoints(
                        self.project.labels_dir, old_kpts, names, mode="name"
                    )
                    QMessageBox.information(
                        self,
                        "Keypoint migration",
                        f"Migrated {modified} / {total} label files.",
                    )

            self.project.keypoint_names = names
            self._rebuild_kpt_list()

            if self._ann is not None:
                k = len(self.project.keypoint_names)
                if len(self._ann.kpts) < k:
                    self._ann.kpts.extend(
                        [Keypoint(0.0, 0.0, 0) for _ in range(k - len(self._ann.kpts))]
                    )
                else:
                    self._ann.kpts = self._ann.kpts[:k]

            k = len(self.project.keypoint_names)
            self.project.skeleton_edges = [
                (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
            ]
            self.infer = PoseInferenceService(
                self.project.out_root,
                self.project.keypoint_names,
                self.project.skeleton_edges,
            )
            if self._ann is not None:
                self._rebuild_canvas()
            self._populate_frames()
            self._update_info()
            self.save_project()

    def open_project_settings(self: object) -> object:
        """open_project_settings method documentation."""
        dataset_dir = self.project.images_dir.parent
        wiz = ProjectWizard(dataset_dir, existing=self.project, parent=self)
        if wiz.exec() != QDialog.Accepted:
            return

        old_classes = list(self.project.class_names)
        old_kpts = list(self.project.keypoint_names)

        new_root, new_labels = wiz.get_paths()
        new_classes = wiz.get_classes()
        new_kpts = wiz.get_keypoints()
        new_edges = wiz.get_edges()
        autosave, pad = wiz.get_options()
        do_mig, mig_mode = wiz.get_migration()

        # Apply paths
        self.project.out_root = new_root
        self.project.labels_dir = new_labels
        self.project.labels_dir.mkdir(parents=True, exist_ok=True)

        # Warning if classes changed
        if new_classes != old_classes:
            QMessageBox.information(
                self,
                "Note on classes",
                "Changing class ordering can change the meaning of existing labels.\n"
                "If you already labeled data, consider keeping class order stable.",
            )
        self.project.class_names = new_classes

        # Keypoints changed -> optional migration
        if new_kpts != old_kpts and do_mig:
            modified, total = migrate_labels_keypoints(
                self.project.labels_dir, old_kpts, new_kpts, mode=mig_mode
            )
            QMessageBox.information(
                self,
                "Keypoint migration",
                f"Migrated {modified} / {total} label files using mode='{mig_mode}'.",
            )

        self.project.keypoint_names = new_kpts
        # clamp edges to new range
        k = len(new_kpts)
        self.project.skeleton_edges = [
            (a, b) for (a, b) in new_edges if 0 <= a < k and 0 <= b < k and a != b
        ]
        self.infer = PoseInferenceService(
            self.project.out_root,
            self.project.keypoint_names,
            self.project.skeleton_edges,
        )
        self.project.autosave = autosave
        self.project.bbox_pad_frac = pad

        # Refresh UI
        self.class_combo.clear()
        self.class_combo.addItems(self.project.class_names)
        self._rebuild_kpt_list()

        # Ensure current annotation matches new K
        if self._ann is not None:
            if len(self._ann.kpts) < k:
                self._ann.kpts.extend(
                    [Keypoint(0.0, 0.0, 0) for _ in range(k - len(self._ann.kpts))]
                )
            else:
                self._ann.kpts = self._ann.kpts[:k]

        self._rebuild_canvas()
        self._populate_frames()
        self.save_project()
        self._update_info()

    def manage_sources(self) -> None:
        """Open PoseKit's standalone Source Manager and refresh the workspace."""
        dlg = SourceManagerDialog(self.project, parent=self)
        if dlg.exec() != SourceManagerDialog.Accepted or not dlg.has_changes:
            return

        if dlg.source_ids_to_remove:
            reply = QMessageBox.warning(
                self,
                "Remove sources?",
                "Removing sources deletes their project-local label folders from this PoseKit project. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        for source_id in dlg.source_ids_to_remove:
            remove_source_from_project(self.project, source_id)
        for dataset_dir, description in dlg.sources_to_add:
            from .project import add_source_to_project

            add_source_to_project(self.project, dataset_dir, description)

        preferred_source_id = dlg.preferred_source_id
        if dlg.sources_to_add and preferred_source_id.startswith("pending_"):
            preferred_source_id = ""

        self._refresh_after_source_change(preferred_source_id=preferred_source_id)

    def _prompt_adjust_sources_if_empty(self) -> None:
        """Open Source Manager automatically when a project has no source images yet."""
        if not self.image_paths:
            self.manage_sources()

    def _switch_project_window(
        self,
        proj: Project,
        *,
        open_source_manager_if_empty: bool = False,
    ):
        if hasattr(self, "_recents_store") and proj is not None:
            # Store the project file or directory path
            if hasattr(proj, "project_path") and proj.project_path:
                self._recents_store.add(str(proj.project_path))
        imgs = build_image_list(proj)
        try:
            self._perform_autosave()
            self.save_project()
        except Exception:
            pass
        new_win = MainWindow(proj, imgs, show_welcome_when_empty=False)
        new_win.resize(self.size())
        new_win.showMaximized()
        if open_source_manager_if_empty and not imgs:
            QTimer.singleShot(150, new_win._prompt_adjust_sources_if_empty)
        app = QApplication.instance()
        if app is not None:
            if not hasattr(app, "_posekit_windows"):
                app._posekit_windows = []
            app._posekit_windows.append(new_win)
        self.close()

    def open_project(self: object) -> object:
        """Open an existing PoseKit project.

        The user may select either:
          • A PoseKit project JSON file (standalone or legacy).
          • A dataset root folder that contains a ``posekit_project/`` sub-dir
            (legacy dataset-anchored layout).
        """
        from PySide6.QtWidgets import QFileDialog

        from hydra_suite.paths import get_projects_dir

        start = (
            str(self.project.images_dir.parent)
            if self.project.images_dir
            else str(get_projects_dir())
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PoseKit Project",
            start,
            "PoseKit Project (pose_project.json);;All Files (*)",
        )
        if path:
            # User selected a .json file directly
            proj = open_project_from_path(Path(path))
            if proj is not None:
                self._switch_project_window(proj)
            return

        # Fall back to folder-browse (legacy dataset-anchored layout)
        folder = QFileDialog.getExistingDirectory(
            self, "Or select a legacy dataset folder", start
        )
        if not folder:
            return
        dataset_dir = Path(folder).expanduser().resolve()
        found = find_project(dataset_dir)
        if found:
            proj = load_project_with_repairs(found, dataset_dir)
        else:
            QMessageBox.information(
                self,
                "No Project Found",
                f"No PoseKit project found in:\n{dataset_dir}\n\n"
                'Use "New Project" to create one, or select a pose_project.json file.',
            )
            return
        if proj is None:
            return
        self._switch_project_window(proj)

    def export_dataset_dialog(self: object) -> object:
        """export_dataset_dialog method documentation."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Export dataset.yaml + copied images/labels")
        layout = QFormLayout(dlg)

        out_root = QLineEdit(str(self.project.out_root))
        split = QDoubleSpinBox()
        split.setRange(0.05, 0.95)
        split.setSingleStep(0.05)
        split.setValue(0.8)

        seed = QSpinBox()
        seed.setRange(0, 999999)
        seed.setValue(0)

        split_method = QComboBox()
        split_method.addItems(["Random", "Cluster-stratified"])

        cluster_csv = QLineEdit("")
        cluster_csv.setPlaceholderText("Optional: clusters.csv (auto-detected)")
        btn_cluster = QPushButton("Choose…")

        def pick_dir() -> object:
            """pick_dir method documentation."""
            d = QFileDialog.getExistingDirectory(
                self, "Select output root", out_root.text()
            )
            if d:
                out_root.setText(d)

        def pick_cluster() -> object:
            """pick_cluster method documentation."""
            path, _ = QFileDialog.getOpenFileName(
                self, "Select cluster CSV", str(self.project.out_root), "CSV (*.csv)"
            )
            if path:
                cluster_csv.setText(path)

        btn_pick = QPushButton("Choose…")
        btn_pick.clicked.connect(pick_dir)
        btn_cluster.clicked.connect(pick_cluster)

        row = QHBoxLayout()
        row.addWidget(out_root, 1)
        row.addWidget(btn_pick)

        cl_row = QHBoxLayout()
        cl_row.addWidget(cluster_csv, 1)
        cl_row.addWidget(btn_cluster)

        layout.addRow("Output directory", row)
        layout.addRow("Split method", split_method)
        layout.addRow("Train fraction", split)
        layout.addRow("Random seed", seed)
        layout.addRow("Cluster CSV", cl_row)

        btns = QHBoxLayout()
        ok = QPushButton("Export")
        cancel = QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addRow(btns)

        cancel.clicked.connect(dlg.reject)
        ok.clicked.connect(dlg.accept)

        if dlg.exec() != QDialog.Accepted:
            return

        root = Path(out_root.text()).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        train_items = None
        val_items = None
        if split_method.currentText() == "Cluster-stratified":
            csv_path = (
                Path(cluster_csv.text().strip()) if cluster_csv.text().strip() else None
            )
            if csv_path is None or not csv_path.exists():
                default_csv = (
                    self.project.out_root / "posekit" / "clusters" / "clusters.csv"
                )
                csv_path = default_csv if default_csv.exists() else None
            if not csv_path:
                QMessageBox.warning(
                    self,
                    "Missing clusters",
                    "No cluster CSV found. Run Smart Select → Clustering first.",
                )
                return

            cluster_ids = self._load_cluster_ids_from_csv(csv_path)
            if not cluster_ids:
                QMessageBox.warning(
                    self,
                    "Missing clusters",
                    "Could not load cluster IDs from CSV.",
                )
                return

            labeled_indices = [
                i for i, p in enumerate(self.image_paths) if self._is_labeled(p)
            ]
            items = []
            item_cluster_ids = []
            for i in labeled_indices:
                img = self.image_paths[i]
                lbl = self._label_path_for(img)
                if lbl.exists():
                    items.append((img, lbl))
                    item_cluster_ids.append(cluster_ids[i])

            if len(items) < 2:
                QMessageBox.warning(
                    self, "Not enough labels", "Need at least 2 labeled frames."
                )
                return

            train_idx, val_idx, _ = cluster_stratified_split(
                [p for p, _ in items],
                item_cluster_ids,
                train_frac=float(split.value()),
                val_frac=1.0 - float(split.value()),
                test_frac=0.0,
                min_per_cluster=1,
                seed=int(seed.value()),
            )
            train_items = [items[i] for i in train_idx]
            val_items = [items[i] for i in val_idx]

        try:
            info = build_yolo_pose_dataset(
                self.image_paths,
                self.project.labels_dir,
                root,
                float(split.value()),
                int(seed.value()),
                self.project.class_names,
                self.project.keypoint_names,
                # Pass each extra source so build_yolo_pose_dataset collects all labels
                extra_datasets=[
                    ([p for p in list_images(s.images_dir)], s.labels_dir)
                    for s in self.project.sources
                    if s.labels_dir != self.project.labels_dir
                ]
                or None,
                train_items=train_items,
                val_items=val_items,
                ignore_occluded_train=False,
                ignore_occluded_val=False,
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        QMessageBox.information(
            self,
            "Exported",
            "Wrote:\n"
            f"- {info['yaml_path']}\n"
            f"- images/train + labels/train\n"
            f"- images/val + labels/val\n"
            f"- {info.get('manifest', '')}",
        )

    def open_smart_select(self: object) -> object:
        """open_smart_select method documentation."""
        dlg = SmartSelectDialog(self, self.project, self.image_paths, self._is_labeled)
        if dlg.exec() != QDialog.Accepted or not getattr(dlg, "_did_add", False):
            return

        # If they added nothing, ignore
        picked = getattr(dlg, "selected_indices", None)
        if not picked:
            return

        self._add_indices_to_labeling(picked, "Smart Select")

    def _add_indices_to_labeling(self, indices: List[int], title: str):
        if not indices:
            return
        for idx in indices:
            self.labeling_frames.add(int(idx))
        self._populate_frames()
        self._select_frame_in_list(self.current_index)
        QMessageBox.information(
            self, title, f"Added {len(indices)} frames to labeling set."
        )

    def _collect_selected_indices(self) -> List[int]:
        idxs = set()
        for item in self.labeling_list.selectedItems():
            try:
                idx = int(item.data(Qt.UserRole))
                idxs.add(idx)
            except Exception:
                continue
        for item in self.frame_list.selectedItems():
            try:
                idx = int(item.data(Qt.UserRole))
                idxs.add(idx)
            except Exception:
                continue
        return sorted(idxs)

    def _delete_selected_frames(self):
        indices = self._collect_selected_indices()
        if not indices:
            QMessageBox.information(self, "Delete", "No frames selected.")
            return

        names = [self.image_paths[i].name for i in indices if i < len(self.image_paths)]
        preview = "\n".join(names[:10])
        if len(names) > 10:
            preview += f"\n… (+{len(names) - 10} more)"

        resp = QMessageBox.warning(
            self,
            "Delete selected frames?",
            "This will permanently delete the selected images and any corresponding labels.\n\n"
            f"Count: {len(indices)}\n\n{preview}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        # Keep track of current frame path for remapping
        cur_path = None
        if 0 <= self.current_index < len(self.image_paths):
            cur_path = self.image_paths[self.current_index]

        deleted_paths: List[Path] = []

        # Delete files
        for idx in indices:
            if idx < 0 or idx >= len(self.image_paths):
                continue
            img_path = self.image_paths[idx]
            deleted_paths.append(img_path)
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception:
                pass
            try:
                lbl_path = self._label_path_for(img_path)
                if lbl_path.exists():
                    lbl_path.unlink()
            except Exception:
                pass

        self._cleanup_deleted_cache(deleted_paths)

        # Rebuild image list and labeling set by path
        old_labeling_paths = {
            self.image_paths[i]
            for i in self.labeling_frames
            if 0 <= i < len(self.image_paths)
        }
        self.image_paths = build_image_list(self.project)
        self._rebuild_path_index()
        self.labeling_frames = set()
        for i, p in enumerate(self.image_paths):
            if p in old_labeling_paths or self._is_labeled(p):
                self.labeling_frames.add(i)

        # Reset caches and current state
        self._frame_cache.clear()
        self._ann = None
        self._dirty = False
        self._undo_stack.clear()
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._cluster_ids_cache = None
        self._cluster_ids_mtime = None

        # Remap current index
        self.current_index = 0
        if cur_path:
            try:
                self.current_index = self.image_paths.index(cur_path)
            except ValueError:
                self.current_index = 0

        self._populate_frames()
        if self.image_paths:
            self.load_frame(self.current_index)
        else:
            self._show_canvas_logo_placeholder()
            self.lbl_info.setText("No images found.")

    @staticmethod
    def _build_deleted_keys(deleted_paths: List[Path]) -> set:
        """Build a set of string keys (original + resolved) for deleted paths."""
        deleted_keys: set = set()
        for p in deleted_paths:
            try:
                deleted_keys.add(str(p))
                deleted_keys.add(str(Path(p).resolve()))
            except Exception:
                deleted_keys.add(str(p))
        return deleted_keys

    @staticmethod
    def _key_in_deleted(key: str, deleted_keys: set) -> bool:
        """Check whether *key* (a path string) matches any deleted path."""
        try:
            return key in deleted_keys or str(Path(key).resolve()) in deleted_keys
        except Exception:
            return key in deleted_keys

    def _cleanup_metadata(self, deleted_keys: set) -> None:
        """Remove metadata entries whose paths are in *deleted_keys*."""
        try:
            keys = list(self.metadata_manager.metadata.keys())
            for k in keys:
                if self._key_in_deleted(k, deleted_keys):
                    self.metadata_manager.metadata.pop(k, None)
            self.metadata_manager.save()
        except Exception:
            pass

    def _cleanup_prediction_cache(self, deleted_keys: set) -> None:
        """Purge deleted paths from JSON prediction caches."""
        pred_dir = self.project.out_root / "posekit" / "predictions"
        if not pred_dir.exists():
            return
        for cache_path in pred_dir.glob("*.json"):
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                preds = data.get("preds", {})
                if not preds:
                    continue
                changed = False
                for k in list(preds.keys()):
                    if k in deleted_keys:
                        preds.pop(k, None)
                        changed = True
                if changed:
                    data["preds"] = preds
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
            except Exception:
                continue

    def _cleanup_embeddings_cache(self, deleted_keys: set) -> None:
        """Remove deleted paths from incremental embedding caches."""
        try:
            emb_root = self.project.out_root / "posekit" / "embeddings"
            if not emb_root.exists():
                return
            for child in emb_root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    cache = IncrementalEmbeddingCache(
                        self.project.out_root / "posekit", child.name
                    )
                    cache.remove_paths(deleted_keys)
                except Exception:
                    continue
        except Exception:
            pass

    def _cleanup_clusters_csv(self, deleted_keys: set) -> None:
        """Remove deleted paths from the clusters CSV cache."""
        try:
            cluster_csv = (
                self.project.out_root / "posekit" / "clusters" / "clusters.csv"
            )
            if not cluster_csv.exists():
                return
            rows = []
            changed = False
            with cluster_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("image") or row.get("image_path") or row.get("path")
                    if not img:
                        continue
                    if self._key_in_deleted(img, deleted_keys):
                        changed = True
                        continue
                    rows.append({"image": img, "cluster_id": row.get("cluster_id")})
            if changed:
                cluster_csv.parent.mkdir(parents=True, exist_ok=True)
                with cluster_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["image", "cluster_id"])
                    for row in rows:
                        writer.writerow([row["image"], row["cluster_id"]])
        except Exception:
            pass

    def _cleanup_deleted_cache(self, deleted_paths: List[Path]) -> None:
        if not deleted_paths:
            return
        deleted_keys = self._build_deleted_keys(deleted_paths)
        self._cleanup_metadata(deleted_keys)
        self._cleanup_prediction_cache(deleted_keys)
        self._cleanup_embeddings_cache(deleted_keys)
        self._cleanup_clusters_csv(deleted_keys)

    def _preds_to_keypoints(
        self, preds: List[Tuple[float, float, float]], conf_thr: float = 0.25
    ) -> List[Keypoint]:
        kpts = []
        for x, y, c in preds:
            if not np.isfinite(c):
                c = 0.0
            if not np.isfinite(x) or not np.isfinite(y):
                x, y, c = 0.0, 0.0, 0.0
            if c > conf_thr and c > 0.0:
                kpts.append(Keypoint(float(x), float(y), 2))
            else:
                kpts.append(Keypoint(0.0, 0.0, 0))
        return kpts

    def _pred_conf_default(self) -> float:
        try:
            return float(self.sp_pred_conf.value())
        except Exception:
            return 0.25

    def _on_pred_conf_changed(self, _value: float):
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._populate_frames()
        self._rebuild_canvas()

    def _pred_backend(self) -> str:
        try:
            txt = self.combo_pred_backend.currentText().strip().lower()
            return "sleap" if txt.startswith("sleap") else "yolo"
        except Exception:
            return "yolo"

    def _selected_tier(self) -> str:
        """Return the currently selected runtime tier id (e.g. ``'cpu'``, ``'gpu'``)."""
        if hasattr(self, "combo_pred_runtime"):
            data = self.combo_pred_runtime.currentData()
            if data:
                return str(data).strip().lower()
        return "cpu"

    def _selected_compute_runtime(self, stage: Optional[str] = None) -> str:
        """Return the canonical compute runtime for the currently selected tier.

        ``stage`` selects the resolver's capability profile ("yolo_pose" /
        "sleap_pose"); defaults to the currently selected prediction backend
        so YOLO and SLEAP resolve against their own profile, mirroring
        ``core/inference/stages/pose.py``.
        """
        if hasattr(self, "combo_pred_runtime"):
            data = self.combo_pred_runtime.currentData()
            tier = str(data).strip().lower() if data else ""
            if not tier:
                # Fall back to text if itemData is absent (legacy combo entries).
                txt = self.combo_pred_runtime.currentText().strip().lower()
                # If the text is already a canonical runtime, return it directly.
                if txt in CANONICAL_RUNTIMES:
                    return txt
                tier = txt
            # Map tier → canonical runtime using detected platform.
            platform = detect_platform()
            resolved_stage = stage or (
                "sleap_pose" if self._pred_backend() == "sleap" else "yolo_pose"
            )
            return tier_to_canonical_runtime(tier, platform, stage=resolved_stage)
        return "cpu"

    def _populate_pred_runtime_options(
        self, backend: str = "", preferred_tier: Optional[str] = None
    ):
        """Populate the runtime combo with platform-level tiers (cpu / gpu / gpu_fast).

        The *backend* argument is kept for call-site compatibility but is ignored —
        tiers are platform-level, not per-pipeline.  *preferred_tier* is the tier
        id (``"cpu"``, ``"gpu"``, ``"gpu_fast"``) to pre-select; if ``None`` the
        current selection is preserved.
        """
        if not hasattr(self, "combo_pred_runtime"):
            return
        combo = self.combo_pred_runtime
        platform = detect_platform()
        tiers = available_tiers(platform)
        # Determine which tier to pre-select.
        current_data = combo.currentData()
        current_tier = str(current_data).strip().lower() if current_data else ""
        if preferred_tier and preferred_tier in tiers:
            selected_tier = preferred_tier
        elif current_tier in tiers:
            selected_tier = current_tier
        else:
            selected_tier = tiers[0]
        combo.blockSignals(True)
        combo.clear()
        for t in tiers:
            combo.addItem(tier_label(t, platform), t)
        idx = combo.findData(selected_tier)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _set_sleap_status(self, text: str, visible: Optional[bool] = None):
        if not hasattr(self, "status_sleap"):
            return
        if visible is not None:
            self.status_sleap.setVisible(bool(visible))
        self.status_sleap.setText(text)

    def _set_busy_progress(self, msg: Optional[str] = None):
        self.status_progress.setRange(0, 0)
        self.status_progress.setVisible(True)
        if msg:
            self.statusBar().showMessage(msg)

    def _clear_progress(self):
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setVisible(False)

    def _set_bulk_prediction_locked(self, locked: bool) -> None:
        """Block interactive UI controls while dataset prediction is running."""
        self._bulk_prediction_locked = bool(locked)
        enabled = not self._bulk_prediction_locked
        widget_names = [
            "list_widget",
            "labeling_list",
            "search_edit",
            "combo_pred_backend",
            "combo_pred_runtime",
            "pred_exported_edit",
            "btn_pred_exported",
            "sp_pred_conf",
            "spin_pred_batch",
            "pred_weights_edit",
            "btn_pred_weights",
            "btn_pred_weights_latest",
            "combo_sleap_env",
            "btn_sleap_refresh",
            "sleap_model_edit",
            "btn_sleap_model",
            "btn_sleap_model_latest",
            "btn_sleap_start",
            "btn_sleap_stop",
            "btn_predict",
            "btn_predict_bulk",
            "btn_apply_preds",
            "btn_clear_pred_cache",
        ]
        for name in widget_names:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(enabled)
        if hasattr(self, "menuBar") and self.menuBar() is not None:
            self.menuBar().setEnabled(enabled)
        if self._bulk_prediction_locked:
            self.statusBar().showMessage(
                "Predicting dataset: controls are locked until completion."
            )
        else:
            self._update_pred_backend_ui()

    def _update_pred_backend_ui(self):
        if self._bulk_prediction_locked:
            return
        backend = self._pred_backend()
        is_sleap = backend == "sleap"
        if hasattr(self, "yolo_pred_widget"):
            self.yolo_pred_widget.setVisible(not is_sleap)
        if hasattr(self, "sleap_pred_widget"):
            self.sleap_pred_widget.setVisible(is_sleap)
        if hasattr(self, "lbl_pred_exported"):
            self.lbl_pred_exported.setVisible(False)
        if hasattr(self, "pred_exported_edit") and hasattr(self, "btn_pred_exported"):
            self.pred_exported_edit.setVisible(False)
            self.btn_pred_exported.setVisible(False)
        if is_sleap:
            running = PoseInferenceService.sleap_service_running()
            self._set_sleap_status(
                "SLEAP: running" if running else "SLEAP: idle", visible=True
            )
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(not running)
                self.btn_sleap_stop.setEnabled(running)
        else:
            self._set_sleap_status("SLEAP: off", visible=False)
        if not is_sleap:
            if PoseInferenceService.sleap_service_running():
                self._stop_sleap_service()
            else:
                try:
                    PoseInferenceService.shutdown_sleap_service()
                except Exception:
                    pass

    def _refresh_sleap_envs(self):
        if not hasattr(self, "combo_sleap_env"):
            return
        self.combo_sleap_env.clear()
        self.combo_sleap_env.setEnabled(True)
        envs: List[str] = []
        try:
            res = run_conda(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    name = parts[0]
                    if name.lower().startswith("sleap"):
                        envs.append(name)
            else:
                self.lbl_sleap_env_status.setText("Unable to list conda environments.")
        except FileNotFoundError:
            self.lbl_sleap_env_status.setText("Conda not found on PATH.")
            envs = []
        except Exception as e:
            self.lbl_sleap_env_status.setText(f"Env scan failed: {e}")
            envs = []

        if not envs:
            self.combo_sleap_env.addItem("No sleap envs found")
            self.combo_sleap_env.setEnabled(False)
            if not self.lbl_sleap_env_status.text():
                self.lbl_sleap_env_status.setText(
                    "No conda envs starting with 'sleap' found."
                )
        else:
            self.combo_sleap_env.addItems(envs)
            if self._sleap_env_pref and self._sleap_env_pref in envs:
                self.combo_sleap_env.setCurrentText(self._sleap_env_pref)
            self.lbl_sleap_env_status.setText("")

    def _browse_sleap_model_dir(self):
        start = self.sleap_model_edit.text().strip() or str(self.project.out_root)
        path = QFileDialog.getExistingDirectory(
            self, "Select SLEAP model directory", start
        )
        if path:
            self._validate_sleap_model_dir(Path(path), notify=True)

    def _is_sleap_model_dir(self, path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        return (path / "training_config.yaml").exists() or (
            path / "training_config.json"
        ).exists()

    def _find_latest_sleap_model_dir_from(self, base: Path) -> Optional[Path]:
        models_dir = base / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            return None
        best_ts = None
        best_path = None
        for child in models_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            prefix = name.split(".", 1)[0]
            try:
                ts = datetime.strptime(prefix, "%y%m%d_%H%M%S")
            except Exception:
                continue
            if not self._is_sleap_model_dir(child):
                continue
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_path = child
        return best_path

    def _resolve_sleap_model_dir(self, path: Path) -> Optional[Path]:
        try:
            path = path.expanduser().resolve()
        except Exception:
            pass
        if path.is_file():
            path = path.parent
        if self._is_sleap_model_dir(path):
            return path
        found = self._find_latest_sleap_model_dir_from(path)
        if found:
            return found
        try:
            parent = path.parent
        except Exception:
            parent = None
        if parent:
            found = self._find_latest_sleap_model_dir_from(parent)
            if found:
                return found
        return None

    def _validate_sleap_model_dir(
        self, path: Path, notify: bool = True
    ) -> Optional[Path]:
        resolved = self._resolve_sleap_model_dir(path)
        if not resolved:
            if notify:
                QMessageBox.warning(
                    self,
                    "Invalid SLEAP model dir",
                    "Selected folder does not contain training_config.yaml/json.\n"
                    "Choose a model folder under models/<YYMMDD_HHMMSS>.*",
                )
            return None
        if notify and resolved != path:
            QMessageBox.information(
                self,
                "SLEAP model folder resolved",
                f"'{path}' is not a model folder.\nUsing:\n{resolved}",
            )
        self.sleap_model_edit.setText(str(resolved))
        return resolved

    def _find_latest_sleap_model_dir(self, slp_path: Path) -> Optional[Path]:
        try:
            base = slp_path.parent
        except Exception:
            return None
        return self._find_latest_sleap_model_dir_from(base)

    def _use_latest_sleap_model(self):
        slp = getattr(self.project, "latest_sleap_dataset", None)
        if not slp:
            QMessageBox.information(
                self,
                "No SLEAP dataset",
                "No latest SLEAP dataset recorded. Export a SLEAP dataset first.",
            )
            return
        slp_path = Path(slp)
        if not slp_path.exists():
            QMessageBox.warning(
                self,
                "Missing dataset",
                f"Latest SLEAP dataset not found:\n{slp_path}",
            )
            return
        model_dir = self._find_latest_sleap_model_dir(slp_path)
        if not model_dir:
            QMessageBox.warning(
                self,
                "No models found",
                f"No models folder with timestamped runs found next to:\n{slp_path}",
            )
            return
        self._validate_sleap_model_dir(model_dir, notify=False)

    def _get_sleap_env(self) -> Optional[str]:
        env = self.combo_sleap_env.currentText().strip()
        if not env or env.lower().startswith("no sleap"):
            return None
        return env

    def _get_sleap_model_or_prompt(self) -> Optional[Path]:
        txt = self.sleap_model_edit.text().strip()
        if txt:
            resolved = self._validate_sleap_model_dir(Path(txt), notify=True)
            if resolved:
                return resolved
            return None
        self._browse_sleap_model_dir()
        txt = self.sleap_model_edit.text().strip()
        if not txt:
            return None
        resolved = self._validate_sleap_model_dir(Path(txt), notify=True)
        if resolved:
            return resolved
        return None

    def _get_sleap_model_silent(self) -> Optional[Path]:
        txt = self.sleap_model_edit.text().strip()
        if not txt:
            return None
        return self._validate_sleap_model_dir(Path(txt), notify=False)

    def _start_sleap_service(self):
        env = self._get_sleap_env()
        if not env:
            QMessageBox.warning(
                self,
                "No SLEAP env",
                "Select a conda env starting with 'sleap' for SLEAP inference.",
            )
            return
        if PoseInferenceService.sleap_service_running():
            self._set_sleap_status("SLEAP: running", visible=True)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)
            return
        if self._sleap_service_thread is not None:
            try:
                if self._sleap_service_thread.isRunning():
                    return
            except RuntimeError:
                # Qt object can already be deleted; drop stale reference.
                self._sleap_service_thread = None
        self._set_sleap_status("SLEAP: starting", visible=True)
        self._set_busy_progress("Starting SLEAP service…")
        self.btn_sleap_start.setEnabled(False)
        self.btn_sleap_stop.setEnabled(False)
        self._sleap_service_thread = QThread()
        self._sleap_service_worker = SleapServiceWorker(env, self.project.out_root)
        self._sleap_service_worker.moveToThread(self._sleap_service_thread)
        self._sleap_service_thread.started.connect(self._sleap_service_worker.run)
        self._sleap_service_worker.finished.connect(self._on_sleap_service_started)
        self._sleap_service_worker.finished.connect(self._sleap_service_thread.quit)
        self._sleap_service_thread.finished.connect(
            self._on_sleap_service_thread_finished
        )
        self._sleap_service_thread.finished.connect(
            self._sleap_service_thread.deleteLater
        )
        self._sleap_service_thread.start()

    def _on_sleap_service_thread_finished(self):
        self._sleap_service_thread = None
        self._sleap_service_worker = None

    def _stop_sleap_service(self):
        if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
            self.btn_sleap_start.setEnabled(False)
            self.btn_sleap_stop.setEnabled(False)
        self._set_sleap_status("SLEAP: stopping", visible=True)
        self.statusBar().showMessage("Stopping SLEAP service…", 2000)
        self._clear_progress()

        try:
            PoseInferenceService.shutdown_sleap_service()
        except Exception as e:
            self._set_sleap_status("SLEAP: error", visible=True)
            self.statusBar().showMessage(f"SLEAP stop failed: {e}", 4000)
        if self._sleap_service_thread is not None:
            try:
                if self._sleap_service_thread.isRunning():
                    self._sleap_service_thread.quit()
            except RuntimeError:
                # Already deleted by Qt lifecycle.
                self._sleap_service_thread = None

        running = PoseInferenceService.sleap_service_running()
        if running:
            self._set_sleap_status("SLEAP: running", visible=True)
            self.statusBar().showMessage("SLEAP service still running.", 4000)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)
            return

        self._set_sleap_status("SLEAP: idle", visible=True)
        self.statusBar().showMessage("SLEAP service stopped.", 3000)
        if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
            self.btn_sleap_start.setEnabled(True)
            self.btn_sleap_stop.setEnabled(False)

    def _on_sleap_service_started(self, ok: bool, err: str, log_path: str):
        self._clear_progress()
        self.btn_sleap_start.setEnabled(True)
        self.btn_sleap_stop.setEnabled(True)
        if not ok:
            self._set_sleap_status("SLEAP: error", visible=True)
            QMessageBox.warning(self, "SLEAP service failed", err)
            return
        self._set_sleap_status("SLEAP: running", visible=True)
        self.btn_sleap_start.setEnabled(False)
        self.btn_sleap_stop.setEnabled(True)
        if log_path:
            self.statusBar().showMessage(
                f"SLEAP service running. Log: {log_path}", 4000
            )

    def _get_pred_model_or_prompt(self) -> Optional[Path]:
        backend = self._pred_backend()
        if backend == "sleap":
            return self._get_sleap_model_or_prompt()
        return self._get_pred_weights_or_prompt()

    def _get_pred_model_silent(self) -> Optional[Path]:
        backend = self._pred_backend()
        if backend == "sleap":
            return self._get_sleap_model_silent()
        return self._get_pred_weights_silent()

    def _pred_runtime_flavor(self) -> str:
        backend = self._pred_backend()
        compute_runtime = self._selected_compute_runtime()
        # "coreml" (native Apple GPU-Fast) has no representation in
        # derive_pose_runtime_settings — that helper's `_normalize_runtime`
        # collapses "coreml" to the legacy "onnx_coreml" canonical runtime and
        # returns the ONNX-CoreML-EP flavor ("onnx_mps"), silently reintroducing
        # the exact bug this module exists to fix (see
        # docs/superpowers/specs/2026-04-04-codebase-simplification-design.md
        # Task 6). "coreml" uniquely identifies the native-backend case and
        # needs no further translation for either backend family, so short-
        # circuit here rather than routing it through that shared, cross-app
        # helper (out of scope to modify — used by trackerkit too).
        if compute_runtime == "coreml":
            return "coreml"
        derived = derive_pose_runtime_settings(compute_runtime, backend_family=backend)
        return str(derived.get("pose_runtime_flavor", "cpu")).strip().lower()

    def _browse_pred_exported_model(self):
        backend = self._pred_backend()
        runtime = self._pred_runtime_flavor()
        start = self.pred_exported_edit.text().strip() or str(self.project.out_root)
        if backend == "sleap":
            path = QFileDialog.getExistingDirectory(
                self, "Select exported SLEAP model directory", start
            )
            if path:
                self.pred_exported_edit.setText(path)
            return

        if runtime.startswith("onnx"):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select exported YOLO ONNX model",
                start,
                "ONNX models (*.onnx);;All files (*)",
            )
        elif runtime.startswith("tensorrt"):
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select exported YOLO TensorRT engine",
                start,
                "TensorRT engines (*.engine);;All files (*)",
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select exported YOLO model",
                start,
                "Exported models (*.onnx *.engine);;All files (*)",
            )
        if file_path:
            self.pred_exported_edit.setText(file_path)

    def _get_pred_exported_model_silent(self) -> Optional[Path]:
        txt = self.pred_exported_edit.text().strip()
        if not txt:
            return None
        p = Path(txt).expanduser().resolve()
        if p.exists():
            return p
        return None

    def _pred_cache_model(
        self,
        model: Optional[Path],
        backend: str,
        runtime_flavor: Optional[str] = None,
        exported_model: Optional[Path] = None,
    ) -> Optional[Path]:
        if model is None:
            return None
        return model

    def _pred_cache_backend(
        self, backend: str, runtime_flavor: Optional[str] = None
    ) -> str:
        backend_norm = str(backend or "yolo").strip().lower()
        if backend_norm != "sleap":
            return backend_norm
        flavor = (
            str(runtime_flavor or self._pred_runtime_flavor() or "native")
            .strip()
            .lower()
        )
        try:
            batch = int(max(1, int(self.spin_pred_batch.value())))
        except Exception:
            batch = 1
        return f"sleap:{flavor}:b{batch}"

    def _browse_pred_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select pose weights", "", "*.pt")
        if path:
            self.pred_weights_edit.setText(path)

    def _use_latest_pred_weights(self):
        if (
            self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.pred_weights_edit.setText(str(self.project.latest_pose_weights))

    def _get_pred_weights_or_prompt(self) -> Optional[Path]:
        txt = self.pred_weights_edit.text().strip()
        if txt:
            p = Path(txt).expanduser().resolve()
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
            QMessageBox.warning(
                self, "Invalid weights", "Prediction weights not found."
            )
            return None
        weights = self._get_latest_weights_or_prompt()
        if weights:
            self.pred_weights_edit.setText(str(weights))
        return weights

    def _get_pred_weights_silent(self) -> Optional[Path]:
        txt = self.pred_weights_edit.text().strip()
        if txt:
            p = Path(txt).expanduser().resolve()
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
            return None
        if self.project.latest_pose_weights:
            p = Path(self.project.latest_pose_weights)
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
        return None

    def _get_pred_overlay_for_current(
        self,
    ) -> Tuple[Optional[List[Keypoint]], Optional[List[float]]]:
        if not self.show_predictions:
            return None, None
        backend = self._pred_backend()
        cache_backend = self._pred_cache_backend(backend)
        model = self._pred_cache_model(self._get_pred_model_silent(), backend)
        if not model:
            return None, None
        preds = self._get_pred_for_frame(
            model, self.image_paths[self.current_index], backend=cache_backend
        )
        if not preds:
            return None, None
        conf_thr = self._pred_conf_default()
        pred_kpts = self._preds_to_keypoints(preds, conf_thr=conf_thr)
        pred_confs = []
        for p in preds:
            if len(p) >= 3 and np.isfinite(p[2]):
                pred_confs.append(float(p[2]))
            else:
                pred_confs.append(0.0)
        return pred_kpts, pred_confs

    def _rebuild_canvas(self):
        if self._ann is None:
            return
        pred_kpts, pred_confs = self._get_pred_overlay_for_current()
        self.canvas.rebuild_overlays(
            self._ann.kpts,
            self.project.keypoint_names,
            self.project.skeleton_edges,
            pred_kpts=pred_kpts,
            pred_confs=pred_confs,
            show_pred_conf=self.show_pred_conf,
        )

    def _toggle_show_predictions(self, checked: bool):
        self.show_predictions = bool(checked)
        self._rebuild_canvas()

    def _toggle_show_pred_conf(self, checked: bool):
        checked = bool(checked)
        self.show_pred_conf = checked
        if (
            hasattr(self, "cb_show_pred_conf")
            and self.cb_show_pred_conf.isChecked() != checked
        ):
            self.cb_show_pred_conf.blockSignals(True)
            self.cb_show_pred_conf.setChecked(checked)
            self.cb_show_pred_conf.blockSignals(False)
        if (
            hasattr(self, "act_show_pred_conf")
            and self.act_show_pred_conf.isChecked() != checked
        ):
            self.act_show_pred_conf.blockSignals(True)
            self.act_show_pred_conf.setChecked(checked)
            self.act_show_pred_conf.blockSignals(False)
        self._rebuild_canvas()

    def _get_pred_for_frame(
        self, model: Path, img_path: Path, backend: str = "yolo"
    ) -> Optional[List[Tuple[float, float, float]]]:
        return self.infer.get_cached_pred(model, img_path, backend=backend)

    def _current_frame_has_labels(self) -> bool:
        if self._ann is not None and any(kp.v > 0 for kp in self._ann.kpts):
            return True
        img_path = self.image_paths[self.current_index]
        if self._is_labeled(img_path):
            try:
                ann = self._load_ann_from_disk(self.current_index)
                return any(kp.v > 0 for kp in ann.kpts)
            except Exception:
                return True
        return False

    def _get_latest_weights_or_prompt(self) -> Optional[Path]:
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
        ):
            p = Path(self.project.latest_pose_weights)
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p

        path, _ = QFileDialog.getOpenFileName(self, "Select pose weights", "", "*.pt")
        if not path:
            return None
        weights = Path(path).resolve()
        self.project.latest_pose_weights = weights
        self.save_project()
        return weights

    def predict_current_frame(self: object) -> object:
        """predict_current_frame method documentation."""
        if not self.image_paths or self._ann is None:
            QMessageBox.information(self, "No frame", "Select a frame first.")
            return

        backend = self._pred_backend()
        model = self._get_pred_model_or_prompt()
        if not model:
            return
        sleap_env = None
        if backend == "sleap":
            sleap_env = self._get_sleap_env()
            if not sleap_env:
                QMessageBox.warning(
                    self,
                    "No SLEAP env",
                    "Select a conda env starting with 'sleap' for SLEAP inference.",
                )
                return
            if not PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: starting", visible=True)
                self._set_busy_progress("Starting SLEAP service…")
            else:
                self._set_sleap_status("SLEAP: running", visible=True)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)
            self._last_pred_conf = None
        else:
            self._last_pred_conf = self._pred_conf_default()
        runtime_flavor = self._pred_runtime_flavor()
        cache_backend = self._pred_cache_backend(backend, runtime_flavor=runtime_flavor)
        cache_model = self._pred_cache_model(
            model,
            backend,
            runtime_flavor=runtime_flavor,
            exported_model=None,
        )
        self._last_pred_model = cache_model
        self._last_pred_backend = backend
        self._last_pred_cache_backend = cache_backend

        cached = self._get_pred_for_frame(
            cache_model, self.image_paths[self.current_index], backend=cache_backend
        )
        if cached is not None:
            self._on_pred_finished(cached)
            return

        if backend != "sleap":
            self.statusBar().showMessage(f"Predicting keypoints… ({model.name})")
        self.btn_predict.setEnabled(False)
        if backend == "sleap":
            if PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: running", visible=True)

        self._pred_thread = QThread()
        self._pred_worker = PosePredictWorker(
            model_path=model,
            image_path=self.image_paths[self.current_index],
            out_root=self.project.out_root,
            keypoint_names=self.project.keypoint_names,
            skeleton_edges=self.project.skeleton_edges,
            backend=backend,
            runtime_flavor=runtime_flavor,
            exported_model_path=None,
            device="auto",
            imgsz=640,
            conf=self._last_pred_conf or 0.0,
            yolo_batch=int(self.spin_pred_batch.value()),
            sleap_env=sleap_env,
            sleap_device="auto",
            sleap_batch=int(self.spin_pred_batch.value()),
            sleap_max_instances=1,
            cache_backend=cache_backend,
        )
        self._pred_worker.moveToThread(self._pred_thread)
        self._pred_thread.started.connect(self._pred_worker.run)
        self._pred_worker.resolved_exported_model_signal.connect(
            self._on_pred_exported_model_resolved
        )
        self._pred_worker.finished.connect(self._on_pred_finished)
        self._pred_worker.failed.connect(self._on_pred_failed)
        self._pred_worker.finished.connect(self._pred_thread.quit)
        self._pred_worker.failed.connect(self._pred_thread.quit)
        self._pred_thread.finished.connect(self._on_pred_thread_finished)
        self._pred_thread.finished.connect(self._pred_thread.deleteLater)
        self._pred_thread.start()

    def _on_pred_finished(self, preds: List[Tuple[float, float, float]]):
        self.btn_predict.setEnabled(True)
        if not preds:
            backend = getattr(self, "_last_pred_backend", "yolo")
            conf = getattr(self, "_last_pred_conf", None)
            if backend == "sleap":
                msg = "No predictions found."
            elif conf is not None:
                msg = f"No predictions above conf={conf:.2f}. Try a lower conf."
            else:
                msg = "No predictions above confidence threshold. Try a lower conf."
            self.statusBar().showMessage(msg, 3000)
            if backend == "sleap":
                self._set_sleap_status("SLEAP: idle", visible=True)
                self._clear_progress()
            return
        self.statusBar().showMessage("Prediction loaded. Adjust and save.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
            self._clear_progress()
        if self._ann is None:
            return
        if self.mode != "frame":
            self.rb_frame.setChecked(True)
        model = getattr(self, "_last_pred_model", None)
        backend = getattr(
            self,
            "_last_pred_cache_backend",
            getattr(self, "_last_pred_backend", "yolo"),
        )
        if model:
            img_path = str(self.image_paths[self.current_index])
            self.infer.merge_cache(model, {img_path: preds}, backend=backend)
            self._update_pred_conf_map_from_preds(
                {img_path: preds}, model=model, backend=backend
            )
            conf_val = self._pred_conf_map.get(img_path)
            kpt_val = self._pred_kpt_count_map.get(img_path)
            self._update_frame_item(
                self.current_index,
                pred_conf=conf_val,
                pred_kpt_count=kpt_val,
                conf_only=True,
            )
        self._rebuild_canvas()

    def _on_pred_failed(self, msg: str):
        self.btn_predict.setEnabled(True)
        self.statusBar().showMessage("Prediction failed.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
            self._clear_progress()
        QMessageBox.warning(self, "Prediction failed", msg)

    def _on_pred_thread_finished(self):
        """Clean up single-frame prediction thread after it exits."""
        self._pred_thread = None
        self._pred_worker = None

    def _on_pred_exported_model_resolved(self, path: str):
        resolved = str(path or "").strip()
        if not resolved:
            return
        if hasattr(self, "pred_exported_edit"):
            current = self.pred_exported_edit.text().strip()
            if current == resolved:
                return
            self.pred_exported_edit.setText(resolved)
        self.statusBar().showMessage(
            f"Using exported runtime model: {Path(resolved).name}", 4000
        )
        self._save_ui_settings()

    def _clear_prediction_cache(self):
        backend = self._pred_backend()
        cache_backend = self._pred_cache_backend(backend)
        model = self._pred_cache_model(self._get_pred_model_silent(), backend)
        if model:
            removed = self.infer.clear_cache(model, backend=cache_backend)
            self.statusBar().showMessage(
                f"Cleared prediction cache ({removed} file(s)).", 3000
            )
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            self._clear_conf_display()
            self._rebuild_canvas()
            return

        resp = QMessageBox.question(
            self,
            "Clear prediction cache",
            "No prediction model selected. Clear all cached predictions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return
        removed = self.infer.clear_cache(None)
        self.statusBar().showMessage(
            f"Cleared all prediction caches ({removed} file(s)).", 3000
        )
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._clear_conf_display()
        self._rebuild_canvas()

    def _adopt_prediction_kpt(self, kpt_idx: int, x: float, y: float):
        backend = self._pred_backend()
        cache_backend = self._pred_cache_backend(backend)
        model = self._pred_cache_model(self._get_pred_model_or_prompt(), backend)
        preds = None
        if model:
            preds = self.infer.get_cached_pred(
                model, self.image_paths[self.current_index], backend=cache_backend
            )

        if not self._current_frame_has_labels():
            if not preds:
                return
            # Update dragged keypoint in predictions before adopting all.
            if kpt_idx < len(preds):
                conf = preds[kpt_idx][2] if len(preds[kpt_idx]) >= 3 else 1.0
                preds[kpt_idx] = (float(x), float(y), float(conf))
            self._push_undo()
            conf_thr = self._pred_conf_default()
            self._ann.kpts = self._preds_to_keypoints(preds, conf_thr=conf_thr)
            self._set_dirty()
            self._rebuild_canvas()
            self._update_info()
            self.labeling_frames.add(self.current_index)
            self._populate_frames()
            self._select_frame_in_list(self.current_index)
            if model:
                self.infer.merge_cache(
                    model,
                    {str(self.image_paths[self.current_index]): preds},
                    backend=cache_backend,
                )
            return

        # Prompt only when user tries to drag a prediction and GT exists
        res = QMessageBox.question(
            self,
            "Replace keypoint?",
            "This frame already has labels. Replace this keypoint with the prediction you adjusted?",
        )
        if res != QMessageBox.Yes:
            self._rebuild_canvas()
            return

        self.on_place_kpt(kpt_idx, x, y, 2)
        self.labeling_frames.add(self.current_index)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)

        # Update prediction cache with adjusted point (preserve existing conf if present)
        if model and preds and kpt_idx < len(preds):
            conf = preds[kpt_idx][2] if len(preds[kpt_idx]) >= 3 else 1.0
            preds[kpt_idx] = (float(x), float(y), float(conf))
            self.infer.merge_cache(
                model,
                {str(self.image_paths[self.current_index]): preds},
                backend=cache_backend,
            )

    def apply_predictions_current(self: object) -> object:
        """apply_predictions_current method documentation."""
        if self._ann is None:
            QMessageBox.information(self, "No frame", "Select a frame first.")
            return
        backend = self._pred_backend()
        cache_backend = self._pred_cache_backend(backend)
        model = self._pred_cache_model(self._get_pred_model_or_prompt(), backend)
        if not model:
            return
        preds = self.infer.get_cached_pred(
            model, self.image_paths[self.current_index], backend=cache_backend
        )
        if not preds:
            QMessageBox.information(
                self,
                "No predictions",
                "No cached predictions for this frame. Run Predict Keypoints first.",
            )
            return
        if self._current_frame_has_labels():
            res = QMessageBox.question(
                self,
                "Replace labels?",
                "This frame already has labels. Replace all keypoints with predictions?",
            )
            if res != QMessageBox.Yes:
                return
        self._push_undo()
        conf_thr = self._pred_conf_default()
        self._ann.kpts = self._preds_to_keypoints(preds, conf_thr=conf_thr)
        self._mark_dirty()
        self._rebuild_canvas()
        self._update_info()
        self.labeling_frames.add(self.current_index)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)

    def predict_dataset(self: object) -> object:
        """predict_dataset method documentation."""
        if self._bulk_prediction_locked:
            return
        if not self.image_paths:
            QMessageBox.information(self, "No frames", "No images loaded.")
            return
        backend = self._pred_backend()
        model = self._get_pred_model_or_prompt()
        if not model:
            return
        sleap_env = None
        if backend == "sleap":
            sleap_env = self._get_sleap_env()
            if not sleap_env:
                QMessageBox.warning(
                    self,
                    "No SLEAP env",
                    "Select a conda env starting with 'sleap' for SLEAP inference.",
                )
                return
            if not PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: starting", visible=True)
                self._set_busy_progress("Starting SLEAP service…")
            else:
                self._set_sleap_status("SLEAP: running", visible=True)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)

        items = ["All frames", "Labeling set", "Unlabeled only"]
        scope, ok = QInputDialog.getItem(
            self, "Predict dataset", "Scope:", items, 0, False
        )
        if not ok:
            return
        if scope == "Labeling set":
            indices = sorted(self.labeling_frames)
        elif scope == "Unlabeled only":
            indices = [
                i for i, p in enumerate(self.image_paths) if not self._is_labeled(p)
            ]
        else:
            indices = list(range(len(self.image_paths)))

        paths = [self.image_paths[i] for i in indices]
        if not paths:
            QMessageBox.information(self, "No frames", "No frames in selected scope.")
            return

        if backend != "sleap":
            self.statusBar().showMessage(f"Predicting dataset… ({model.name})")
        self.btn_predict_bulk.setEnabled(False)
        self._set_bulk_prediction_locked(True)
        if backend == "sleap":
            if PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: running", visible=True)

        runtime_flavor = self._pred_runtime_flavor()
        cache_backend = self._pred_cache_backend(backend, runtime_flavor=runtime_flavor)
        cache_model = self._pred_cache_model(
            model,
            backend,
            runtime_flavor=runtime_flavor,
            exported_model=None,
        )
        self._last_pred_model = cache_model
        self._last_pred_backend = backend
        self._last_pred_cache_backend = cache_backend
        pred_batch = int(self.spin_pred_batch.value())
        try:
            self._bulk_pred_thread = QThread()
            self._bulk_pred_worker = BulkPosePredictWorker(
                model_path=model,
                image_paths=paths,
                out_root=self.project.out_root,
                keypoint_names=self.project.keypoint_names,
                skeleton_edges=self.project.skeleton_edges,
                backend=backend,
                runtime_flavor=runtime_flavor,
                exported_model_path=None,
                device="auto",
                imgsz=640,
                conf=self._pred_conf_default() if backend == "yolo" else 0.0,
                batch=pred_batch,
                sleap_env=sleap_env,
                sleap_device="auto",
                sleap_batch=int(pred_batch),
                sleap_max_instances=1,
                cache_backend=cache_backend,
            )
            self._bulk_pred_worker.moveToThread(self._bulk_pred_thread)
            self._bulk_pred_thread.started.connect(self._bulk_pred_worker.run)
            self._bulk_pred_worker.resolved_exported_model_signal.connect(
                self._on_pred_exported_model_resolved
            )
            self._bulk_pred_worker.progress.connect(self._on_bulk_pred_progress)
            self._bulk_pred_worker.finished.connect(self._on_bulk_pred_finished)
            self._bulk_pred_worker.failed.connect(self._on_bulk_pred_failed)
            self._bulk_pred_worker.finished.connect(self._bulk_pred_thread.quit)
            self._bulk_pred_worker.failed.connect(self._bulk_pred_thread.quit)
            self._bulk_pred_thread.finished.connect(self._bulk_pred_thread.deleteLater)
            self._bulk_pred_thread.finished.connect(self._on_bulk_pred_thread_finished)
            self._bulk_pred_thread.start()
        except Exception as e:
            self._set_bulk_prediction_locked(False)
            self.btn_predict_bulk.setEnabled(True)
            QMessageBox.warning(self, "Prediction failed", str(e))

    def _on_bulk_pred_progress(self, done: int, total: int):
        if total > 0:
            if self.status_progress.maximum() == 0:
                self.status_progress.setRange(0, 100)
            pct = int((done / total) * 100)
            self.status_progress.setVisible(True)
            self.status_progress.setValue(min(100, max(0, pct)))
            self.statusBar().showMessage(f"Predicting dataset… {done}/{total}")
            if getattr(self, "_last_pred_backend", "yolo") == "sleap":
                self._set_sleap_status("SLEAP: running", visible=True)

    def _on_bulk_pred_finished(
        self, preds: Dict[str, List[Tuple[float, float, float]]]
    ):
        self.btn_predict_bulk.setEnabled(True)
        self._set_bulk_prediction_locked(False)
        self._bulk_pred_worker = None
        # NOTE: Do NOT set self._bulk_pred_thread = None here!
        # Thread cleanup happens in _on_bulk_pred_thread_finished after thread.quit() completes
        self._clear_progress()
        self.statusBar().showMessage("Predictions cached.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
        model = getattr(self, "_last_pred_model", None)
        backend = getattr(
            self,
            "_last_pred_cache_backend",
            getattr(self, "_last_pred_backend", "yolo"),
        )
        if model:
            self.infer.merge_cache(model, preds, backend=backend)
            self._update_pred_conf_map_from_preds(preds, model=model, backend=backend)
            for path in preds.keys():
                idx = self._path_to_index.get(path)
                if idx is None:
                    try:
                        idx = self._path_to_index.get(str(Path(path).resolve()))
                    except Exception:
                        idx = None
                if idx is None:
                    continue
                conf_val = self._pred_conf_map.get(path)
                if conf_val is None:
                    try:
                        conf_val = self._pred_conf_map.get(str(Path(path).resolve()))
                    except Exception:
                        conf_val = None
                kpt_val = self._pred_kpt_count_map.get(path)
                if kpt_val is None:
                    try:
                        kpt_val = self._pred_kpt_count_map.get(
                            str(Path(path).resolve())
                        )
                    except Exception:
                        kpt_val = None
                self._update_frame_item(
                    idx, pred_conf=conf_val, pred_kpt_count=kpt_val, conf_only=True
                )
        self._rebuild_canvas()

    def _on_bulk_pred_failed(self, msg: str):
        self.btn_predict_bulk.setEnabled(True)
        self._set_bulk_prediction_locked(False)
        self._bulk_pred_worker = None
        # NOTE: Do NOT set self._bulk_pred_thread = None here!
        # Thread cleanup happens in _on_bulk_pred_thread_finished after thread.quit() completes
        self._clear_progress()
        self.statusBar().showMessage("Prediction failed.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
        QMessageBox.warning(self, "Prediction failed", msg)

    def _on_bulk_pred_thread_finished(self):
        """Called when the bulk prediction thread has fully finished execution."""
        self._bulk_pred_thread = None

    def open_training_runner(self: object) -> object:
        """open_training_runner method documentation."""
        dlg = TrainingRunnerDialog(self, self.project, self.image_paths)
        try:
            backend = self._pred_backend()
            if backend == "sleap":
                dlg.backend_combo.setCurrentText("SLEAP")
            else:
                dlg.backend_combo.setCurrentText("YOLO Pose")
            if hasattr(dlg, "_update_backend_ui"):
                dlg._update_backend_ui()
        except Exception:
            pass
        dlg.exec()

    def open_evaluation_dashboard(self: object) -> object:
        """open_evaluation_dashboard method documentation."""
        dlg = EvaluationDashboardDialog(
            self,
            self.project,
            self.image_paths,
            weights_path=(
                str(self.project.latest_pose_weights)
                if self.project.latest_pose_weights
                else None
            ),
            add_frames_callback=lambda idxs, reason="Evaluation": self._add_indices_to_labeling(
                idxs, reason
            ),
        )
        try:
            backend = self._pred_backend()
            dlg.backend_combo.setCurrentText("SLEAP" if backend == "sleap" else "YOLO")
            if backend == "sleap":
                model_path = self.sleap_model_edit.text().strip()
            else:
                model_path = self.pred_weights_edit.text().strip()
            if model_path:
                dlg.lock_model_path(model_path)
            if hasattr(dlg, "_apply_backend_ui"):
                dlg._apply_backend_ui()
        except Exception:
            pass
        dlg.exec()

    def open_active_learning(self: object) -> object:
        """open_active_learning method documentation."""
        dlg = ActiveLearningDialog(
            self,
            self.project,
            self.image_paths,
            self._is_labeled,
            set(self.labeling_frames),
            add_frames_callback=lambda idxs, reason="Active learning": self._add_indices_to_labeling(
                idxs, reason
            ),
        )
        try:
            backend = self._pred_backend()
            dlg.backend_combo.setCurrentText("SLEAP" if backend == "sleap" else "YOLO")
            if backend == "sleap":
                model_path = self.sleap_model_edit.text().strip()
            else:
                model_path = self.pred_weights_edit.text().strip()
            if model_path:
                dlg.lock_model_path(model_path)
            if hasattr(dlg, "_apply_backend_ui"):
                dlg._apply_backend_ui()
        except Exception:
            pass
        dlg.exec()

    # Backwards/alternate name used in older menu wiring.
    def _load_metadata_ui(self):
        if not self.image_paths:
            return
        self._setting_meta = True
        img_path = str(self.image_paths[self.current_index])
        meta = self.metadata_manager.get_metadata(img_path)
        for tag, cb in self.meta_tags.items():
            cb.setChecked(tag in meta.tags)
        self.meta_notes.setText(meta.notes or "")
        self._setting_meta = False

    def _on_meta_changed(self, *_args):
        if self._setting_meta or not self.image_paths:
            return
        img_path = str(self.image_paths[self.current_index])
        meta = self.metadata_manager.get_metadata(img_path)
        meta.tags = {t for t, cb in self.meta_tags.items() if cb.isChecked()}
        meta.notes = self.meta_notes.text().strip()
        self.metadata_manager.save()

    def _load_cluster_ids_from_csv(self, csv_path: Path) -> Optional[List[int]]:
        if not csv_path.exists():
            return None
        mapping = {}
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("image") or row.get("image_path") or row.get("path")
                    cid = row.get("cluster_id") or row.get("cluster")
                    if img is None or cid is None:
                        continue
                    try:
                        mapping[str(Path(img).resolve())] = int(float(cid))
                    except Exception:
                        continue
        except Exception:
            return None

        if not mapping:
            return None

        cluster_ids: List[int] = []
        for p in self.image_paths:
            key = str(p.resolve())
            if key in mapping:
                cluster_ids.append(mapping[key])
            else:
                cluster_ids.append(-1)
        return cluster_ids
