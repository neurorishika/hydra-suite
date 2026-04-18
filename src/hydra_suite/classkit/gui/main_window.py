"""
ClassKit Main Window - Polished and feature-complete UI
"""

import hashlib
import json
import re
import time
from html import escape
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QThreadPool, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from .project import (
    classkit_config_path,
    classkit_export_dir,
    classkit_model_dir,
    classkit_scheme_path,
    ensure_classkit_project_layout,
    prepare_project_directory,
    project_exists,
)
from .widgets.color_utils import best_text_color, build_category_color_map, to_hex


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ClassKit")
        self.resize(1600, 1000)

        # Project state
        self.project_path = None
        self.db_path = None
        self.embeddings = None
        self._current_embedding_cache_id = None
        self._current_cluster_cache_id = None
        self.umap_coords = None
        self.cluster_assignments = None
        self.image_paths = []
        self.image_labels = []
        self.image_confidences = []
        self._image_review_status = {}
        self._image_metadata_by_path = {}
        self._review_candidate_indices = []
        self.classes = ["class_1", "class_2"]
        self.selected_point_index = None
        self.candidate_indices = []
        self.round_labeled_indices = []
        self._labeling_flow_mode = "batch"
        self._labeling_navigation_scope = "pool"
        self._infinite_label_distance_cache = None
        self._infinite_label_cluster_counts = None
        self._infinite_label_owner_cache = None
        self.explorer_mode = "explore"
        self.hover_locked = False
        self._label_shortcuts = []
        self.label_history = []
        self.last_assigned_stack = []
        self.last_umap_params = {"n_neighbors": 15, "min_dist": 0.1}
        self.last_preview_index = None
        self._history_icon_cache = {}
        self._history_thumb_load_budget = 2
        self._command_busy = False
        self._command_block_until = 0.0
        self._command_squelch_pending = False
        self._active_jobs = 0  # reference count: number of running background workers
        self._pending_label_updates = {}
        self._autosave_interval_ms = 5000
        self._autosave_last_save_time = None
        self._autosave_heartbeat_on = False
        self._trained_classifier = None
        self._last_training_settings = None
        self._yolo_model_path = None  # Path to loaded YOLO classification model
        self._model_probs = None  # (N, C) per-image class probabilities
        self._model_class_names = None  # class names from (YOLO/tiny) model
        self.umap_model_coords = None  # UMAP computed in model logits space
        self.pca_model_coords = None  # PCA computed in model logits/probability space
        self._show_model_umap = False  # Explorer toggle: embedding vs model UMAP
        self._show_model_pca = False  # Explorer toggle: embedding vs model PCA
        self._al_candidates = None  # np.ndarray of selected AL batch indices
        self._active_model_mode = None  # "yolo", "yolo_multihead", "tiny", "custom_cnn", "custom_cnn_multihead", or None
        self._heldout_validation_summary = None  # training-time held-out metric
        self._active_evaluation_paths = None
        self._active_evaluation_split_name = ""
        self._current_knn_neighbors = []
        self._stepper = None
        self._custom_shortcuts: dict = {}  # action_name → key sequence string
        self._outline_threshold = 0.60
        self._marker_size_multiplier = 1.0

        # Display enhancement settings (sync with PoseKit)
        self.clahe_clip = 2.0
        self.clahe_grid = (8, 8)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(False)
        self._autosave_timer.setInterval(self._autosave_interval_ms)
        self._autosave_timer.timeout.connect(self._flush_pending_label_updates)

        self._autosave_heartbeat_timer = QTimer(self)
        self._autosave_heartbeat_timer.setSingleShot(False)
        self._autosave_heartbeat_timer.setInterval(700)
        self._autosave_heartbeat_timer.timeout.connect(self._tick_autosave_heartbeat)

        self._history_refresh_timer = QTimer(self)
        self._history_refresh_timer.setSingleShot(True)
        self._history_refresh_timer.setInterval(40)
        self._history_refresh_timer.timeout.connect(self.refresh_label_history_strip)

        self._plot_refresh_pending = False
        self._plot_refresh_force_fit = False
        self._plot_refresh_timer = QTimer(self)
        self._plot_refresh_timer.setSingleShot(True)
        self._plot_refresh_timer.setInterval(24)
        self._plot_refresh_timer.timeout.connect(self._flush_explorer_update)

        self._pending_preview_index = None
        self._pending_preview_source = "hover"
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.setInterval(20)
        self._preview_refresh_timer.timeout.connect(self._flush_preview_update)

        self._context_refresh_pending = False
        self._context_refresh_timer = QTimer(self)
        self._context_refresh_timer.setSingleShot(True)
        self._context_refresh_timer.setInterval(60)
        self._context_refresh_timer.timeout.connect(self._flush_context_update)

        # Threadpool
        self.threadpool = QThreadPool.globalInstance()

        # Apply dark theme stylesheet
        self.apply_stylesheet()

        # Setup UI
        self.setup_menus()
        self.setup_toolbar()
        self.setup_central_widget()
        self.setup_statusbar()

        # Hide the menu bar and toolbar while on the welcome page.
        self.menuBar().hide()
        self.toolbar.hide()

    def apply_stylesheet(self):
        """Apply modern dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
            }
            QListWidget {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
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
                color: #888888;
            }
            QLabel {
                color: #cccccc;
            }
            QToolBar {
                background-color: #252526;
                border-bottom: 1px solid #3e3e42;
                spacing: 8px;
                padding: 6px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                color: #cccccc;
            }
            QToolButton:hover {
                background-color: #2a2d2e;
            }
            QToolButton:pressed {
                background-color: #094771;
            }
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
                border-top: 1px solid #0098ff;
                font-weight: 500;
            }
            QMenuBar {
                background-color: #252526;
                color: #cccccc;
                border-bottom: 1px solid #3e3e42;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #2a2d2e;
            }
            QMenu {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QSplitter::handle {
                background-color: #3e3e42;
            }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 3px;
            }
        """)

    def setup_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_project = QAction("&New Project...", self)
        new_project.setShortcut("Ctrl+N")
        new_project.triggered.connect(self.new_project)
        file_menu.addAction(new_project)

        open_project = QAction("&Open Project...", self)
        open_project.setShortcut("Ctrl+O")
        open_project.triggered.connect(self.open_project)
        file_menu.addAction(open_project)

        save_project = QAction("&Save Project", self)
        save_project.setShortcut("Ctrl+S")
        save_project.triggered.connect(self.save_project)
        file_menu.addAction(save_project)

        file_menu.addSeparator()

        source_mgr_action = QAction("&Source Manager...", self)
        source_mgr_action.setShortcut("Ctrl+I")
        source_mgr_action.triggered.connect(self.manage_sources)
        file_menu.addAction(source_mgr_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Dataset...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_dataset)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Compute menu
        compute_menu = menubar.addMenu("&Compute")

        embed_action = QAction("Compute &Embeddings...", self)
        embed_action.triggered.connect(self.compute_embeddings)
        compute_menu.addAction(embed_action)

        cluster_action = QAction("&Cluster Data...", self)
        cluster_action.triggered.connect(self.cluster_data)
        compute_menu.addAction(cluster_action)

        umap_action = QAction("Compute &UMAP...", self)
        umap_action.triggered.connect(self.compute_umap)
        compute_menu.addAction(umap_action)

        compute_menu.addSeparator()

        self._machine_label_action = QAction("&Machine Labeling…", self)
        self._machine_label_action.setEnabled(False)
        self._machine_label_action.triggered.connect(self.open_machine_labeling_dialog)
        compute_menu.addAction(self._machine_label_action)

        train_action = QAction("&Train Classifier...", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self.train_classifier)
        compute_menu.addAction(train_action)

        load_checkpoint_action = QAction("&Load Classifier Checkpoint...", self)
        load_checkpoint_action.triggered.connect(self.load_classifier_checkpoint)
        compute_menu.addAction(load_checkpoint_action)

        model_history_action = QAction("&Previously Trained Models...", self)
        model_history_action.setShortcut("Ctrl+Shift+H")
        model_history_action.setStatusTip(
            "Browse and load past trained models registered in this project"
        )
        model_history_action.triggered.connect(self._open_model_history)
        compute_menu.addAction(model_history_action)

        build_al_action = QAction("Build Active &Learning Batch...", self)
        build_al_action.setShortcut("Ctrl+Shift+B")
        build_al_action.triggered.connect(self._build_al_batch)
        compute_menu.addAction(build_al_action)

        review_menu = menubar.addMenu("&Review")

        self._enter_review_mode_action = QAction("Enter &Review Mode", self)
        self._enter_review_mode_action.triggered.connect(self.enter_review_mode)
        self._enter_review_mode_action.setEnabled(False)
        review_menu.addAction(self._enter_review_mode_action)

        review_menu.addSeparator()

        self._approve_selected_review_action = QAction("Approve &Selected", self)
        self._approve_selected_review_action.triggered.connect(
            self.approve_selected_review_label
        )
        self._approve_selected_review_action.setEnabled(False)
        review_menu.addAction(self._approve_selected_review_action)

        self._approve_all_review_action = QAction("Approve &All Machine Labels", self)
        self._approve_all_review_action.triggered.connect(
            self.approve_all_machine_labels
        )
        self._approve_all_review_action.setEnabled(False)
        review_menu.addAction(self._approve_all_review_action)

        clear_unverified_action = QAction("Clear All &Unverified", self)
        clear_unverified_action.triggered.connect(
            self.clear_all_unverified_machine_labels
        )
        review_menu.addAction(clear_unverified_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_view)
        view_menu.addAction(refresh_action)

        view_menu.addSeparator()

        self.act_enhance = QAction("&Enhance Contrast (CLAHE)", self)
        self.act_enhance.setShortcut("Ctrl+Shift+E")
        self.act_enhance.setCheckable(True)
        self.act_enhance.setChecked(False)
        self.act_enhance.setStatusTip(
            "Toggle CLAHE contrast enhancement for better visibility"
        )
        self.act_enhance.toggled.connect(self.on_enhance_toggled)
        view_menu.addAction(self.act_enhance)

        enhance_settings = QAction("Contrast &Settings...", self)
        enhance_settings.triggered.connect(self.open_contrast_settings)
        view_menu.addAction(enhance_settings)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About ClassKit", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.toolbar = toolbar  # keep reference for busy-blocking
        self.addToolBar(toolbar)

        # Project section
        new_btn = QAction("New", self)
        new_btn.setStatusTip("Create a new ClassKit project")
        new_btn.triggered.connect(self.new_project)
        toolbar.addAction(new_btn)

        open_btn = QAction("Open", self)
        open_btn.setStatusTip("Open an existing project")
        open_btn.triggered.connect(self.open_project)
        toolbar.addAction(open_btn)

        save_btn = QAction("Save", self)
        save_btn.setStatusTip("Save project labels")
        save_btn.triggered.connect(self.save_project)
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        # Data section
        source_mgr_btn = QAction("Source Manager", self)
        source_mgr_btn.setStatusTip("View, add, or remove image source folders")
        source_mgr_btn.triggered.connect(self.manage_sources)
        toolbar.addAction(source_mgr_btn)

        toolbar.addSeparator()

        # Compute section
        embed_btn = QAction("Embed", self)
        embed_btn.setStatusTip("Compute embeddings")
        embed_btn.triggered.connect(self.compute_embeddings)
        toolbar.addAction(embed_btn)

        cluster_btn = QAction("Cluster", self)
        cluster_btn.setStatusTip("Cluster embeddings")
        cluster_btn.triggered.connect(self.cluster_data)
        toolbar.addAction(cluster_btn)

        toolbar.addSeparator()

        self._machine_label_toolbar_action = QAction("Machine Label", self)
        self._machine_label_toolbar_action.setStatusTip(
            "Configure AprilTag or model-based machine labeling"
        )
        self._machine_label_toolbar_action.triggered.connect(
            self.open_machine_labeling_dialog
        )
        self._machine_label_toolbar_action.setEnabled(False)
        toolbar.addAction(self._machine_label_toolbar_action)

        toolbar.addSeparator()

        # Training section
        train_btn = QAction("Train", self)
        train_btn.setStatusTip("Train classifier")
        train_btn.triggered.connect(self.train_classifier)
        toolbar.addAction(train_btn)

        history_btn = QAction("Previously Trained Models", self)
        history_btn.setStatusTip("Browse and load past trained models")
        history_btn.triggered.connect(self._open_model_history)
        toolbar.addAction(history_btn)

        toolbar.addSeparator()

        # Export section
        export_btn = QAction("Export", self)
        export_btn.setStatusTip("Export labeled dataset")
        export_btn.triggered.connect(self.export_dataset)
        toolbar.addAction(export_btn)

    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown before any project is opened."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("classkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="classkit.svg",
            tagline="Active Learning Dataset Builder",
            buttons=[
                ButtonDef(label="New Project\u2026", callback=self.new_project),
                ButtonDef(label="Open Project\u2026", callback=self.open_project),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    def _open_recent_project(self, path: str):
        """Open a project from the recent items list."""
        from pathlib import Path

        self._flush_pending_label_updates(force=True)
        project_path = Path(path)
        if project_path.exists():
            self.project_path = project_path
            self.db_path = prepare_project_directory(project_path)
            if self.db_path.exists():
                self.load_project_data()
                self.update_context_panel()
                self.status.showMessage(f"Opened project: {self.project_path.name}")
            else:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Invalid Project",
                    "This directory does not contain a valid ClassKit project.\n\n"
                    "Expected file: classkit.db",
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

    def setup_central_widget(self):
        """Setup main UI layout for UMAP exploration and fast labeling."""
        self.splitter = QSplitter(Qt.Horizontal)
        # Stacked widget: page 0 = welcome logo, page 1 = working UI
        self._stacked = QStackedWidget()
        self._stacked.addWidget(self._make_welcome_page())  # index 0
        self._stacked.addWidget(self.splitter)  # index 1
        self.setCentralWidget(self._stacked)

        from .widgets.explorer import ExplorerView
        from .widgets.image_viewer import ImageCanvas

        # 1) Left: project metadata + settings + label controls
        self.context_panel = QScrollArea()
        self.context_panel.setWidgetResizable(True)
        self.context_panel.setFixedWidth(380)
        self.context_panel.setStyleSheet(
            "QScrollArea { border: none; background: #1e1e1e; }"
        )

        self.context_content = QWidget()
        self.context_layout = QVBoxLayout(self.context_content)
        self.context_layout.setContentsMargins(16, 16, 16, 16)
        self.context_layout.setSpacing(14)

        # ── Group 1: Project Info ─────────────────────────────────────
        group_info = QGroupBox("Project Info")
        layout_info = QVBoxLayout(group_info)
        layout_info.setContentsMargins(12, 10, 12, 12)
        layout_info.setSpacing(8)
        group_info.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )

        self.context_info = QLabel("No project loaded.")
        self.context_info.setWordWrap(True)
        self.context_info.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.context_info.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        self.context_info.setStyleSheet(
            "color: #cccccc; font-size: 11px; line-height: 1.3;"
        )
        layout_info.addWidget(self.context_info)

        autosave_row = QHBoxLayout()
        autosave_row.setContentsMargins(0, 2, 0, 0)
        autosave_row.setSpacing(8)
        autosave_label = QLabel("Autosave")
        autosave_label.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        autosave_row.addWidget(autosave_label)
        autosave_row.addStretch(1)
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 300)
        self.autosave_spin.setValue(max(1, self._autosave_interval_ms // 1000))
        self.autosave_spin.setSuffix(" s")
        self.autosave_spin.setFixedWidth(82)
        self.autosave_spin.valueChanged.connect(self.on_autosave_interval_changed)
        autosave_row.addWidget(self.autosave_spin)
        layout_info.addLayout(autosave_row)

        marker_size_row = QHBoxLayout()
        marker_size_row.addWidget(QLabel("Marker size:"))
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(0.5, 3.0)
        self.marker_size_spin.setSingleStep(0.1)
        self.marker_size_spin.setDecimals(1)
        self.marker_size_spin.setValue(self._marker_size_multiplier)
        self.marker_size_spin.setToolTip(
            "Scale explorer marker sizes for denser or sparser projections."
        )
        self.marker_size_spin.valueChanged.connect(self.on_marker_size_changed)
        marker_size_row.addWidget(self.marker_size_spin)
        layout_info.addLayout(marker_size_row)

        self.context_layout.addWidget(group_info)

        # ── Group 2: Selection & Neighbors ────────────────────────────
        group_selection = QGroupBox("Selection & Neighbors")
        layout_selection = QVBoxLayout(group_selection)

        self.selection_info = QLabel("No point selected.")
        self.selection_info.setWordWrap(True)
        self.selection_info.setStyleSheet("color: #aaaaaa; padding: 4px;")
        layout_selection.addWidget(self.selection_info)

        self.knn_info = QTextEdit()
        self.knn_info.setReadOnly(True)
        self.knn_info.setMaximumHeight(100)
        self.knn_info.setStyleSheet(
            "background: #111; border: 1px solid #333; border-radius: 4px;"
        )
        layout_selection.addWidget(self.knn_info)

        knn_actions = QHBoxLayout()
        self.knn_jump_btn = QPushButton("Jump")
        self.knn_jump_btn.clicked.connect(self.jump_to_nearest_neighbor)
        self.knn_jump_btn.setEnabled(False)
        knn_actions.addWidget(self.knn_jump_btn)

        self.knn_bulk_btn = QPushButton("Bulk Label 5")
        self.knn_bulk_btn.clicked.connect(self.bulk_label_nearest_neighbors)
        self.knn_bulk_btn.setEnabled(False)
        knn_actions.addWidget(self.knn_bulk_btn)
        layout_selection.addLayout(knn_actions)

        self.context_layout.addWidget(group_selection)

        # ── Group 3: Labeling ─────────────────────────────────────────
        group_labeling = QGroupBox("Labeling")
        layout_labeling = QVBoxLayout(group_labeling)

        # Search / Filter classes
        self.class_search = QLineEdit()
        self.class_search.setPlaceholderText("Filter classes...")
        self.class_search.textChanged.connect(self.filter_label_buttons)
        layout_labeling.addWidget(self.class_search)

        self.label_buttons_container = QWidget()
        self.label_buttons_layout = QGridLayout(self.label_buttons_container)
        self.label_buttons_layout.setSpacing(6)
        layout_labeling.addWidget(self.label_buttons_container)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.clicked.connect(self.on_prev_image)
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.on_next_image)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        layout_labeling.addLayout(nav_row)

        self.context_layout.addWidget(group_labeling)

        group_review = QGroupBox("Review Queue")
        layout_review = QVBoxLayout(group_review)

        self.review_info = QLabel("No machine review items yet.")
        self.review_info.setWordWrap(True)
        self.review_info.setStyleSheet(
            "color: #cccccc; font-size: 12px; line-height: 1.45;"
        )
        layout_review.addWidget(self.review_info)

        self.review_hint = QLabel(
            "Use Machine Label to generate candidates, then switch to Review mode to approve or reject with the keyboard."
        )
        self.review_hint.setWordWrap(True)
        self.review_hint.setStyleSheet("color:#9a9a9a; font-size:11px;")
        layout_review.addWidget(self.review_hint)

        review_actions = QHBoxLayout()
        review_actions.setSpacing(8)

        self.review_selected_btn = QPushButton("Approve")
        self.review_selected_btn.clicked.connect(self.approve_selected_review_label)
        self.review_selected_btn.setEnabled(False)
        review_actions.addWidget(self.review_selected_btn)

        self.review_reject_btn = QPushButton("Reject")
        self.review_reject_btn.clicked.connect(self.reject_selected_review_label)
        self.review_reject_btn.setEnabled(False)
        review_actions.addWidget(self.review_reject_btn)

        self.review_clear_unverified_btn = QPushButton("Clear All Unverified")
        self.review_clear_unverified_btn.clicked.connect(
            self.clear_all_unverified_machine_labels
        )
        self.review_clear_unverified_btn.setEnabled(False)
        review_actions.addWidget(self.review_clear_unverified_btn)

        layout_review.addLayout(review_actions)

        self.context_layout.addWidget(group_review)

        # ── Actions ───────────────────────────────────────────────────
        edit_row = QHBoxLayout()
        btn_edit_classes = QPushButton("Edit Scheme")
        btn_edit_classes.clicked.connect(self.open_class_editor)
        btn_edit_classes.setStyleSheet("background: #1a3a5e; padding: 5px;")
        edit_row.addWidget(btn_edit_classes)

        btn_edit_shortcuts = QPushButton("Shortcuts")
        btn_edit_shortcuts.clicked.connect(self.open_shortcut_editor)
        btn_edit_shortcuts.setStyleSheet("background: #3a3a1a; padding: 5px;")
        edit_row.addWidget(btn_edit_shortcuts)
        self.context_layout.addLayout(edit_row)

        self.shortcut_help = QLabel()
        self.shortcut_help.setWordWrap(True)
        self.shortcut_help.setStyleSheet("font-size: 11px; color: #888; padding: 4px;")
        self.context_layout.addWidget(self.shortcut_help)
        self._refresh_shortcut_help()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.context_layout.addWidget(self.progress_bar)

        self.context_layout.addStretch()
        self.context_panel.setWidget(self.context_content)

        # 2) Center: main UMAP explorer + Tabs for Metrics/Batch
        self.center_tabs = QWidget()
        center_tab_layout = QVBoxLayout(self.center_tabs)
        center_tab_layout.setContentsMargins(0, 0, 0, 0)

        from PySide6.QtWidgets import QTabWidget

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3e3e42; top: -1px; background: #1e1e1e; }
            QTabBar::tab { background: #252526; color: #888888; padding: 10px 20px; border: 1px solid #3e3e42; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; border-bottom: 1px solid #1e1e1e; }
            QTabBar::tab:hover { background: #2d2d2d; }
        """)

        # Tab 1: Explorer
        self.explorer_page = QWidget()
        explorer_layout = QVBoxLayout(self.explorer_page)
        explorer_layout.setContentsMargins(8, 8, 8, 8)
        explorer_layout.setSpacing(8)

        self.explorer = ExplorerView()
        self.explorer.set_uncertainty_outline_threshold(self._outline_threshold)
        self.explorer.set_marker_size_multiplier(self._marker_size_multiplier)
        self.explorer.point_clicked.connect(self.on_explorer_point_clicked)
        self.explorer.point_hovered.connect(self.on_explorer_point_hovered)
        self.explorer.empty_hovered.connect(self.on_explorer_empty_hover)
        self.explorer.empty_double_clicked.connect(
            self.on_explorer_background_double_click
        )

        # Top controls row: mode + projection space
        top_controls_row = QHBoxLayout()
        top_controls_row.setSpacing(8)
        self.mode_toggle_lbl = QLabel("Mode:")
        top_controls_row.addWidget(self.mode_toggle_lbl)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Explore (clusters)", "explore")
        self.view_mode_combo.addItem("Labeling (labels)", "labeling")
        self.view_mode_combo.addItem("Review (machine labels)", "review")
        self.view_mode_combo.addItem("Predictions", "predictions")
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        top_controls_row.addWidget(self.view_mode_combo)

        top_controls_row.addSpacing(12)
        self.outline_threshold_label = QLabel("Outline confidence <")
        top_controls_row.addWidget(self.outline_threshold_label)
        self.outline_threshold_spin = QDoubleSpinBox()
        self.outline_threshold_spin.setRange(0.0, 1.0)
        self.outline_threshold_spin.setSingleStep(0.05)
        self.outline_threshold_spin.setDecimals(2)
        self.outline_threshold_spin.setValue(self._outline_threshold)
        self.outline_threshold_spin.setFixedWidth(72)
        self.outline_threshold_spin.setToolTip(
            "Confidence threshold for white uncertainty outlines in Predictions mode.\n"
            "Set to 0 to disable uncertainty outlines."
        )
        self.outline_threshold_spin.valueChanged.connect(
            self.on_outline_threshold_changed
        )
        top_controls_row.addWidget(self.outline_threshold_spin)
        self.outline_threshold_label.setVisible(False)
        self.outline_threshold_spin.setVisible(False)

        top_controls_row.addStretch(1)
        top_controls_row.addWidget(QLabel("<b>Projection:</b>"))
        self.btn_umap_embedding = QPushButton("Embeddings")
        self.btn_umap_embedding.setCheckable(True)
        self.btn_umap_embedding.setChecked(True)
        self.btn_umap_embedding.setFixedWidth(110)
        self.btn_umap_embedding.clicked.connect(
            lambda: self._switch_projection_space("embedding")
        )
        top_controls_row.addWidget(self.btn_umap_embedding)
        self.btn_umap_model = QPushButton("Model UMAP")
        self.btn_umap_model.setCheckable(True)
        self.btn_umap_model.setChecked(False)
        self.btn_umap_model.setEnabled(False)
        self.btn_umap_model.setVisible(False)
        self.btn_umap_model.setFixedWidth(110)
        self.btn_umap_model.setToolTip(
            "Switch explorer to UMAP of trained model predictions (computed automatically after checkpoint load)"
        )
        self.btn_umap_model.clicked.connect(
            lambda: self._switch_projection_space("model_umap")
        )
        top_controls_row.addWidget(self.btn_umap_model)

        self.btn_pca_model = QPushButton("Model PCA")
        self.btn_pca_model.setCheckable(True)
        self.btn_pca_model.setChecked(False)
        self.btn_pca_model.setEnabled(False)
        self.btn_pca_model.setVisible(False)
        self.btn_pca_model.setFixedWidth(110)
        self.btn_pca_model.setToolTip(
            "Switch explorer to PCA of model predictions (computed on demand)"
        )
        self.btn_pca_model.clicked.connect(
            lambda: self._switch_projection_space("model_pca")
        )
        top_controls_row.addWidget(self.btn_pca_model)

        explorer_layout.addLayout(top_controls_row)
        explorer_layout.addWidget(self.explorer, 1)

        # Labeling controls
        self.labeling_options_group = QGroupBox("Labeling Options")
        self.labeling_options_group.setVisible(False)
        labeling_options_layout = QVBoxLayout(self.labeling_options_group)
        labeling_options_layout.setContentsMargins(10, 10, 10, 10)
        labeling_options_layout.setSpacing(8)

        self.labeling_options_hint = QLabel(
            "Sample and manage the current labeling candidate set."
        )
        self.labeling_options_hint.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        labeling_options_layout.addWidget(self.labeling_options_hint)

        self.label_controls_row = QHBoxLayout()
        self.label_controls_row.setSpacing(12)

        self.label_controls_row.addWidget(QLabel("<b>Set:</b>"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(6)
        self.sample_spin.setToolTip("Unlabeled images to sample from each cluster")
        self.label_controls_row.addWidget(self.sample_spin)

        self.btn_sample_next = QPushButton("Sample Next")
        self.btn_sample_next.setStyleSheet(
            "background: #0e639c; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_sample_next.clicked.connect(self.on_sample_next_triggered)
        self.label_controls_row.addWidget(self.btn_sample_next)

        self.btn_clear_candidates = QPushButton("Clear Candidates")
        self.btn_clear_candidates.setStyleSheet(
            "background: #3e3e42; padding: 6px 12px;"
        )
        self.btn_clear_candidates.clicked.connect(self.on_clear_candidates_triggered)
        self.label_controls_row.addWidget(self.btn_clear_candidates)

        self.label_controls_row.addStretch(1)
        labeling_options_layout.addLayout(self.label_controls_row)

        explorer_layout.addWidget(self.labeling_options_group)

        # ── Inline Active Learning panel ────────────────────────────────

        al_group = QGroupBox("Active Learning")
        al_group.setStyleSheet("""
            QGroupBox { color: #777; border: 1px solid #3e3e42; border-radius: 4px;
                        margin-top: 8px; font-size: 11px; padding-top: 2px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        al_group_layout = QVBoxLayout(al_group)
        al_group_layout.setContentsMargins(8, 4, 8, 6)
        al_group_layout.setSpacing(4)

        al_ctrl_row = QHBoxLayout()
        al_ctrl_row.setSpacing(8)

        self.al_status_label = QLabel(
            "Model: none  |  Labeled: 0 / 0  |  Predictions: none"
        )
        self.al_status_label.setStyleSheet("color: #777; font-size: 11px;")
        al_ctrl_row.addWidget(self.al_status_label, 1)

        al_ctrl_row.addWidget(QLabel("<b>Batch:</b>"))
        self.al_batch_spin = QSpinBox()
        self.al_batch_spin.setRange(5, 500)
        self.al_batch_spin.setValue(50)
        self.al_batch_spin.setFixedWidth(68)
        self.al_batch_spin.setStyleSheet("background: #252526;")
        al_ctrl_row.addWidget(self.al_batch_spin)

        self.al_build_btn = QPushButton("⚙  Build Batch")
        self.al_build_btn.setStyleSheet(
            "background: #0e639c; font-weight: bold; padding: 4px 12px;"
        )
        self.al_build_btn.setToolTip(
            "Select the highest-value unlabeled images for labeling —\n"
            "40% uncertain · 35% diverse · 15% representative · 10% audit"
        )
        self.al_build_btn.clicked.connect(self._build_al_batch)
        al_ctrl_row.addWidget(self.al_build_btn)

        self.al_candidates_badge = QLabel("")
        self.al_candidates_badge.setStyleSheet(
            "color: #ccc; font-size: 11px; font-weight: bold;"
        )
        al_ctrl_row.addWidget(self.al_candidates_badge)

        self.al_start_btn = QPushButton("▶  Label")
        self.al_start_btn.setStyleSheet(
            "background: #28a745; color: white; font-weight: bold; padding: 4px 12px;"
        )
        self.al_start_btn.setEnabled(False)
        self.al_start_btn.setToolTip(
            "Switch to Labeling mode with AL candidates as the active set"
        )
        self.al_start_btn.clicked.connect(self._start_labeling_al_batch)
        al_ctrl_row.addWidget(self.al_start_btn)

        self.al_highlight_btn = QPushButton("◆ Highlight")
        self.al_highlight_btn.setStyleSheet("background: #3e3e42; padding: 4px 10px;")
        self.al_highlight_btn.setEnabled(False)
        self.al_highlight_btn.setToolTip(
            "Show AL candidates highlighted on the UMAP without entering Labeling mode"
        )
        self.al_highlight_btn.clicked.connect(self._highlight_al_batch_on_map)
        al_ctrl_row.addWidget(self.al_highlight_btn)

        al_group_layout.addLayout(al_ctrl_row)

        # Candidate list — hidden until a batch is built
        self.al_candidate_list = QListWidget()
        self.al_candidate_list.setFixedHeight(100)
        self.al_candidate_list.setStyleSheet(
            "background: #1a1a1a; color: #ccc; font-size: 11px; font-family: monospace;"
        )
        self.al_candidate_list.setAlternatingRowColors(True)
        self.al_candidate_list.itemDoubleClicked.connect(self._al_candidate_goto)
        self.al_candidate_list.hide()
        al_group_layout.addWidget(self.al_candidate_list)

        explorer_layout.addWidget(al_group)
        # ──────────────────────────────────────────────────────────────────

        self.history_title = QLabel("<b>Recent Labels (click to undo + relabel)</b>")
        explorer_layout.addWidget(self.history_title)

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFixedHeight(140)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.history_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_scroll_content = QWidget()
        self.history_scroll_layout = QHBoxLayout(self.history_scroll_content)
        self.history_scroll_layout.setContentsMargins(8, 8, 8, 8)
        self.history_scroll_layout.setSpacing(8)
        self._history_slots = []
        for _ in range(24):
            card = QFrame()
            card.setFixedSize(110, 110)
            card.setToolTip("")
            card.setStyleSheet(
                "padding: 4px; border: 1px solid #3e3e42; border-radius: 6px;"
            )
            card.setProperty("history_index", None)
            card.installEventFilter(self)

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(4, 4, 4, 4)
            card_layout.setSpacing(4)

            thumb = QLabel()
            thumb.setFixedSize(72, 72)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setAttribute(Qt.WA_TransparentForMouseEvents, True)

            caption = QLabel("")
            caption.setAlignment(Qt.AlignCenter)
            caption.setWordWrap(True)
            caption.setAttribute(Qt.WA_TransparentForMouseEvents, True)

            card_layout.addWidget(thumb, 0, Qt.AlignCenter)
            card_layout.addWidget(caption, 0, Qt.AlignCenter)
            card.hide()

            self._history_slots.append((card, thumb, caption))
            self.history_scroll_layout.addWidget(card)

        self.history_scroll_layout.addStretch(1)
        self.history_scroll.setWidget(self.history_scroll_content)
        explorer_layout.addWidget(self.history_scroll)

        self.tabs.addTab(self.explorer_page, "Explorer")

        # Tab 2: Metrics
        self.metrics_page = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_page)
        metrics_layout.setContentsMargins(8, 8, 8, 8)
        metrics_layout.setSpacing(6)

        self.metrics_validation_label = QLabel(
            "Held-out validation: unavailable for the current prediction set."
        )
        self.metrics_validation_label.setWordWrap(True)
        self.metrics_validation_label.setStyleSheet(
            "background:#1a1a1a; color:#d8c27a; border-radius:4px; padding:8px;"
        )
        metrics_layout.addWidget(self.metrics_validation_label)

        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setMaximumHeight(180)
        self.metrics_view.setPlaceholderText(
            "Train a model to see evaluation metrics here."
        )
        self.metrics_view.setStyleSheet(
            "font-family: 'SF Mono', 'Roboto Mono', monospace; font-size: 12px; background: #111;"
        )
        metrics_layout.addWidget(self.metrics_view)

        # Matplotlib figure area (confusion matrix + per-class bars)
        self.metrics_figure_label = QLabel("(Train a model to see visualizations)")
        self.metrics_figure_label.setAlignment(Qt.AlignCenter)
        self.metrics_figure_label.setStyleSheet(
            "background: #111; color: #555; border-radius: 4px; padding: 20px;"
        )
        metrics_figure_scroll = QScrollArea()
        metrics_figure_scroll.setWidgetResizable(True)
        metrics_figure_scroll.setWidget(self.metrics_figure_label)
        metrics_layout.addWidget(metrics_figure_scroll, 1)

        self.tabs.addTab(self.metrics_page, "Metrics")

        center_tab_layout.addWidget(self.tabs)

        # 3) Right: large hover/selection image preview
        self.preview_panel = QWidget()
        self.preview_panel.setFixedWidth(420)
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(10)

        preview_title = QLabel(
            "<b style='font-size:16px; color:#ffffff;'>Image Preview</b>"
        )
        preview_layout.addWidget(preview_title)

        # Enhance toggle

        self.preview_canvas = ImageCanvas()
        self.preview_canvas.setMinimumHeight(320)
        preview_layout.addWidget(self.preview_canvas, 1)

        enhance_row = QHBoxLayout()
        enhance_row.setSpacing(8)

        self.cb_enhance = QCheckBox("Enhance contrast (CLAHE)")
        self.cb_enhance.setStyleSheet("color: #aaa; font-size: 11px;")
        self.cb_enhance.toggled.connect(self.on_enhance_toggled)
        enhance_row.addWidget(self.cb_enhance)

        self.btn_contrast_settings = QPushButton("CLAHE Settings…")
        self.btn_contrast_settings.setStyleSheet(
            "background: #3e3e42; color: #d4d4d4; padding: 4px 10px;"
        )
        self.btn_contrast_settings.clicked.connect(self.open_contrast_settings)
        enhance_row.addWidget(self.btn_contrast_settings)
        enhance_row.addStretch(1)
        preview_layout.addLayout(enhance_row)

        self.preview_info = QLabel(
            "<div style='line-height:1.55; color:#aaaaaa;'>"
            "Hover a point to preview the source image.<br>"
            "Select a point, then label using 1-9 or the left-side controls."
            "</div>"
        )
        self.preview_info.setWordWrap(True)
        self.preview_info.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px;"
        )
        preview_layout.addWidget(self.preview_info)

        self.splitter.addWidget(self.context_panel)
        self.splitter.addWidget(self.center_tabs)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.setSizes([380, 920, 420])

        self.review_all_btn = QPushButton("Approve All Machine")
        self.review_all_btn.clicked.connect(self.approve_all_machine_labels)
        self.review_all_btn.hide()
        self.review_apply_model_btn = QPushButton("Apply Model Predictions")
        self.review_apply_model_btn.clicked.connect(self.open_machine_labeling_dialog)
        self.review_apply_model_btn.hide()

        self.setup_label_shortcuts()
        self.rebuild_label_buttons()

    def setup_statusbar(self):
        """Setup status bar."""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.labeling_progress_label = QLabel()
        self.labeling_progress_label.setVisible(False)
        self.labeling_progress_label.setStyleSheet(
            "color: #7ec8e3; font-size: 12px; padding: 0 8px;"
        )
        self.status.addPermanentWidget(self.labeling_progress_label)
        self.autosave_indicator = QLabel("Autosave ○ no project")
        self.status.addPermanentWidget(self.autosave_indicator)
        self.status.showMessage("Ready • No project loaded")
        self._autosave_heartbeat_timer.start()

    # ================== Project Management ==================

    def new_project(self):
        """Create a new ClassKit project."""
        from .dialogs import NewProjectDialog

        try:
            self._flush_pending_label_updates(force=True)
            dialog = NewProjectDialog(self)
            if dialog.exec():
                project_info = dialog.get_project_info()
                self.project_path = Path(project_info["path"])
                self.db_path = ensure_classkit_project_layout(self.project_path)

                # Create project directory
                self.project_path.mkdir(parents=True, exist_ok=True)

                # Initialize database
                from ..core.store.db import ClassKitDB

                ClassKitDB(self.db_path)

                # Save project config
                config = {
                    "name": project_info["name"],
                    "version": "1.0",
                    "classes": project_info.get("classes", []),
                    "autosave_interval_ms": self._autosave_interval_ms,
                    "marker_size_multiplier": self._marker_size_multiplier,
                }
                with open(self._project_config_path(), "w") as f:
                    json.dump(config, f, indent=2)

                # Save labeling scheme if one was selected
                scheme = project_info.get("scheme")
                if scheme is not None:
                    with open(self._project_scheme_path(), "w") as f:
                        json.dump(scheme.to_dict(), f, indent=2)

                self.classes = project_info.get("classes", []) or ["class_1", "class_2"]
                self.rebuild_label_buttons()
                self.setup_label_shortcuts()
                self._refresh_shortcut_help()

                self.update_context_panel()
                self.status.showMessage(f"Created project: {project_info['name']}")

                # If no scheme was set, offer to open the editor now
                if project_info.get("scheme") is None:
                    reply = QMessageBox.question(
                        self,
                        "Project Created",
                        f"Project created:\n{self.project_path}\n\n"
                        "No labeling scheme was defined yet.\n"
                        "Would you like to open the Class Scheme Editor now?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
                        self.open_class_editor()
                else:
                    QMessageBox.information(
                        self,
                        "Project Created",
                        f"Successfully created project:\n\n{self.project_path}",
                    )

                # Always prompt to add sources after a brand new project
                QTimer.singleShot(100, self._prompt_adjust_sources_if_empty)
                if hasattr(self, "_recents_store"):
                    self._recents_store.add(str(self.project_path))
                if hasattr(self, "_stacked"):
                    self._stacked.setCurrentIndex(1)  # reveal working UI
                    self.menuBar().show()
                    self.toolbar.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\\n{e}")

    def open_project(self):
        """Open an existing project."""
        self._flush_pending_label_updates(force=True)
        from hydra_suite.paths import get_projects_dir

        project_dir = QFileDialog.getExistingDirectory(
            self,
            "Open ClassKit Project",
            str(get_projects_dir()),
            QFileDialog.ShowDirsOnly,
        )

        if project_dir:
            self.project_path = Path(project_dir)
            if not project_exists(self.project_path):
                QMessageBox.warning(
                    self,
                    "Invalid Project",
                    "This directory does not contain a valid ClassKit project.",
                )
                return

            self.db_path = prepare_project_directory(self.project_path)

            if not self.db_path.exists():
                QMessageBox.warning(
                    self,
                    "Invalid Project",
                    "This directory does not contain a valid ClassKit project.",
                )
                return

            self.load_project_data()
            self.update_context_panel()
            self.status.showMessage(f"Opened project: {self.project_path.name}")
            QTimer.singleShot(200, self._guide_to_next_step)

    def save_project(self):
        """Force immediate flush of all pending label updates."""
        if not self.db_path:
            self.status.showMessage("No project loaded; nothing to save")
            return

        self.status.showMessage("Saving project...")
        self._flush_pending_label_updates(force=True)
        self.status.showMessage("Project saved", 2000)

    def _load_project_config(self) -> dict:
        """Load project.json if present, otherwise return an empty config."""
        project_config_path = self._project_config_path()
        if not project_config_path.exists():
            return {}
        with open(project_config_path, "r") as f:
            return json.load(f)

    def _project_config_path(self) -> Path:
        """Return the canonical config path for the current project."""
        return classkit_config_path(self.project_path)

    def _project_scheme_path(self) -> Path:
        """Return the canonical scheme path for the current project."""
        return classkit_scheme_path(self.project_path)

    def _apply_project_config(self, config: dict) -> None:
        """Apply persisted project settings to the current UI state."""
        self.classes = config.get("classes", []) or ["class_1", "class_2"]
        stored_training_settings = config.get("last_training_settings")
        if isinstance(stored_training_settings, dict) and stored_training_settings:
            self._last_training_settings = dict(stored_training_settings)

        autosave_interval = int(
            config.get("autosave_interval_ms", self._autosave_interval_ms)
        )
        self._autosave_interval_ms = max(1000, min(300000, autosave_interval))
        self._autosave_timer.setInterval(self._autosave_interval_ms)
        if hasattr(self, "autosave_spin"):
            self.autosave_spin.blockSignals(True)
            self.autosave_spin.setValue(self._autosave_interval_ms // 1000)
            self.autosave_spin.blockSignals(False)

        self._marker_size_multiplier = float(
            config.get("marker_size_multiplier", self._marker_size_multiplier)
        )
        self._marker_size_multiplier = max(0.5, min(3.0, self._marker_size_multiplier))
        if hasattr(self, "marker_size_spin"):
            self.marker_size_spin.blockSignals(True)
            self.marker_size_spin.setValue(self._marker_size_multiplier)
            self.marker_size_spin.blockSignals(False)
        if hasattr(self, "explorer") and self.explorer is not None:
            self.explorer.set_marker_size_multiplier(self._marker_size_multiplier)

        saved_shortcuts = config.get("custom_shortcuts", {})
        if isinstance(saved_shortcuts, dict):
            self._custom_shortcuts = saved_shortcuts

        self.clahe_clip = float(config.get("clahe_clip", 2.0))
        self.clahe_grid = tuple(config.get("clahe_grid", [8, 8]))
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.set_clahe_params(self.clahe_clip, self.clahe_grid)
            self.preview_canvas.use_clahe = bool(config.get("enhance_enabled", False))
        if hasattr(self, "cb_enhance"):
            self.cb_enhance.blockSignals(True)
            self.cb_enhance.setChecked(bool(config.get("enhance_enabled", False)))
            self.cb_enhance.blockSignals(False)
        if hasattr(self, "act_enhance"):
            self.act_enhance.blockSignals(True)
            self.act_enhance.setChecked(bool(config.get("enhance_enabled", False)))
            self.act_enhance.blockSignals(False)

    def _finalize_project_load(self, db) -> None:
        """Refresh derived UI state after database-backed project data loads."""
        self.label_history = []
        self.last_assigned_stack = []
        self.selected_point_index = None
        self.hover_locked = False
        self.candidate_indices = []
        self.round_labeled_indices = []
        self.rebuild_label_buttons()
        self.setup_label_shortcuts()
        self._refresh_shortcut_help()
        self.request_refresh_label_history_strip()
        self._pending_label_updates = {}
        self._autosave_last_save_time = None
        self._update_autosave_heartbeat_text()
        self.try_autoload_cached_artifacts(db)
        self.update_explorer_plot()

        if hasattr(self, "_machine_label_action"):
            self._machine_label_action.setEnabled(self.project_path is not None)
        if hasattr(self, "_machine_label_toolbar_action"):
            self._machine_label_toolbar_action.setEnabled(self.project_path is not None)
        if hasattr(self, "_recents_store"):
            self._recents_store.add(str(self.project_path))
        if hasattr(self, "_stacked"):
            self._stacked.setCurrentIndex(1)
            self.menuBar().show()
            self.toolbar.show()

    def load_project_data(self):
        """Load project data from database."""
        if not self.db_path:
            return

        try:
            from pathlib import Path

            from ..core.store.db import ClassKitDB

            db = ClassKitDB(self.db_path)

            # Migrate any non-resolved paths from older ingests (no-op if already resolved)
            db.migrate_paths_to_resolved()

            # Load image paths
            path_strings = db.get_all_image_paths()
            self.image_paths = [Path(p) for p in path_strings]
            self.image_confidences = [None] * len(self.image_paths)
            self._reload_label_state_from_db(db)

            config = self._load_project_config()
            if config:
                self._apply_project_config(config)
            else:
                self.classes = ["class_1", "class_2"]

            self._invalidate_image_set_dependent_state()

            self._finalize_project_load(db)

            self.status.showMessage(
                f"Loaded {len(self.image_paths):,} images from database"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Load Error", f"Failed to load project data:\\n{e}"
            )

    def update_context_panel(self):
        """Update context panel with project info."""
        if not self.project_path:
            self.context_info.setText(
                "<div style='line-height:1.35;'>"
                "<div style='color:#f2f2f2; font-size:13px; font-weight:600; margin-bottom:4px;'>No project loaded</div>"
                "<div style='color:#8d8d8d;'>File -> New Project or File -> Open Project</div>"
                "</div>"
            )
            if hasattr(self, "review_info"):
                self.review_info.setText("No machine review items yet.")
            self._update_labeling_progress_indicator()
            return

        labeled_count = sum(1 for label in self.image_labels if label)
        total_count = len(self.image_paths)
        unlabeled_count = max(0, total_count - labeled_count)
        n_clusters = (
            len(set(self.cluster_assignments))
            if self.cluster_assignments is not None
            else 0
        )

        classes_preview = ", ".join(escape(name) for name in self.classes[:4])
        if len(self.classes) > 4:
            classes_preview += ", ..."

        candidate_text = (
            f"{len(self.candidate_indices):,} points"
            if self.candidate_indices
            else "none"
        )
        db_state = (
            "Connected" if self.db_path and self.db_path.exists() else "Not ready"
        )
        embedding_state = "Ready" if self.embeddings is not None else "Pending"
        cluster_state = f"{n_clusters:,}" if n_clusters else "Pending"
        umap_state = "Ready" if self.umap_coords is not None else "Pending"

        info_rows = [
            ("DB", db_state),
            ("Images", f"{total_count:,}"),
            ("Labeled", f"{labeled_count:,}"),
            ("Open", f"{unlabeled_count:,}"),
            (
                "Classes",
                (
                    f"{len(self.classes):,} ({classes_preview})"
                    if classes_preview
                    else f"{len(self.classes):,}"
                ),
            ),
            ("Embeddings", embedding_state),
            ("Clusters", cluster_state),
            ("UMAP", umap_state),
            ("Mode", self._mode_display_name(self.explorer_mode)),
            ("Candidates", candidate_text),
        ]

        info_html = "<div style='line-height:1.3;'>"
        info_html += f"<div style='color:#ffffff; font-size:14px; font-weight:600;'>{escape(self.project_path.name)}</div>"
        info_html += f"<div style='color:#7f7f7f; font-size:10px; margin:2px 0 8px 0;'>{escape(str(self.project_path))}</div>"
        info_html += (
            "<table cellspacing='0' cellpadding='0' "
            "style='width:100%; color:#cfcfcf; font-size:11px;'>"
        )
        for label, value in info_rows:
            info_html += (
                "<tr>"
                f"<td style='color:#8d8d8d; padding:2px 10px 2px 0; vertical-align:top; white-space:nowrap;'>{label}</td>"
                f"<td style='color:#f1f1f1; padding:2px 0; vertical-align:top;'>{value}</td>"
                "</tr>"
            )
        info_html += "</table>"

        # ── Next Step guidance ──────────────────────────────────────────
        next_step = ""
        if total_count == 0:
            next_step = "File -> Source Manager"
        elif self.embeddings is None:
            next_step = "Actions -> Compute Embeddings"
        elif self.cluster_assignments is None:
            next_step = "Actions -> Cluster Embeddings"
        elif self.umap_coords is None:
            next_step = "Actions -> Compute UMAP"
        elif len(self.candidate_indices) == 0:
            next_step = "Labeling -> Sample next candidates"
        else:
            next_step = "Labeling -> Enter Labeling Mode (L)"

        info_html += (
            "<div style='margin-top:8px; padding-top:8px; border-top:1px solid #2f2f2f;'>"
            "<span style='color:#5ea6da; font-size:10px; font-weight:600; text-transform:uppercase;'>Next</span>"
            f"<div style='color:#f3f3f3; font-size:11px; margin-top:2px;'>{next_step}</div>"
        )
        info_html += "</div>"

        self.context_info.setText(info_html)
        self._update_review_panel()
        self._update_labeling_progress_indicator()

    def _reload_label_state_from_db(self, db=None) -> None:
        """Refresh labels and review metadata from the project DB."""
        if not self.db_path:
            self.image_labels = []
            self._image_review_status = {}
            self._image_metadata_by_path = {}
            self._review_candidate_indices = []
            self._invalidate_infinite_labeling_cache()
            return

        if db is None:
            from ..core.store.db import ClassKitDB

            db = ClassKitDB(self.db_path)

        self.image_labels = db.get_all_labels()
        self._image_review_status = db.get_label_review_status_by_path()
        self._image_metadata_by_path = db.get_image_metadata_by_path()
        self._refresh_review_candidate_indices()
        self._invalidate_infinite_labeling_cache()
        if len(self.image_confidences) != len(self.image_paths):
            self.image_confidences = [None] * len(self.image_paths)

    def _review_status_for_path(self, path: Path | str) -> dict:
        return dict(self._image_review_status.get(str(path), {}))

    def _review_status_for_index(self, index: int) -> dict:
        if index is None or index < 0 or index >= len(self.image_paths):
            return {}
        return self._review_status_for_path(self.image_paths[index])

    @staticmethod
    def _resolved_path_key(path: Path | str) -> str:
        return str(Path(path).resolve())

    def _derive_training_group_key(self, path: Path | str) -> str:
        resolved_path = self._resolved_path_key(path)
        metadata = self._image_metadata_by_path.get(resolved_path, {})

        original_path_text = metadata.get("original_image_path")
        try:
            original_path = (
                Path(str(original_path_text)).expanduser().resolve()
                if original_path_text
                else Path(resolved_path)
            )
        except Exception:
            original_path = Path(resolved_path)

        source_root = None
        source_root_text = metadata.get("source_root")
        if source_root_text:
            try:
                source_root = Path(str(source_root_text)).expanduser().resolve()
            except Exception:
                source_root = None

        try:
            relative_path = original_path.relative_to(source_root)
        except Exception:
            relative_path = original_path

        parent_parts = [str(part).lower() for part in relative_path.parts[:-1]]
        if parent_parts and parent_parts[0] in {"train", "val", "test"}:
            parent_parts = parent_parts[1:]

        class_folder = str(metadata.get("class_folder") or "").strip().lower()
        if class_folder and parent_parts and parent_parts[-1] == class_folder:
            parent_parts = parent_parts[:-1]

        stem = original_path.stem.lower()
        stem_group = re.sub(
            r"(?i)(?:[_-]?(?:frame|img|image))?[_-]?\d{2,}$",
            "",
            stem,
        ).strip("_-")
        if not stem_group:
            stem_group = (
                stem or hashlib.sha1(resolved_path.encode("utf-8")).hexdigest()[:12]
            )

        source_bucket = str(source_root or original_path.parent)
        parent_bucket = "/".join(parent_parts) if parent_parts else "."
        return f"{source_bucket}::{parent_bucket}::{stem_group}"

    def _training_group_keys_for_paths(self, paths: list[Path]) -> list[str]:
        return [self._derive_training_group_key(path) for path in paths]

    def _set_active_evaluation_selection(
        self,
        paths: list[str] | None,
        split_name: str = "",
    ) -> None:
        normalized = [self._resolved_path_key(path) for path in (paths or []) if path]
        self._active_evaluation_paths = set(normalized) if normalized else None
        self._active_evaluation_split_name = str(split_name or "").strip()

    def _set_active_evaluation_from_meta(self, meta) -> None:
        evaluation = meta.get("evaluation") if isinstance(meta, dict) else None
        if not isinstance(evaluation, dict):
            self._set_active_evaluation_selection(None)
            return
        self._set_active_evaluation_selection(
            evaluation.get("paths") or [],
            str(evaluation.get("split_name") or "").strip(),
        )

    def _cached_model_evaluation_info(self, path: Path):
        if not self.db_path:
            return None
        try:
            from ..core.store.db import ClassKitDB as _CKDb

            resolved = self._resolved_path_key(path)
            for entry in _CKDb(self.db_path).list_model_caches():
                artifact_paths = {
                    self._resolved_path_key(artifact)
                    for artifact in entry.get("artifact_paths", [])
                }
                if resolved in artifact_paths:
                    meta = entry.get("meta")
                    return meta if isinstance(meta, dict) else None
        except Exception:
            return None
        return None

    def _cached_model_cache_entry(self, path: Path):
        """Return the cached model entry containing *path* as an artifact."""
        if not self.db_path:
            return None
        try:
            from ..core.store.db import ClassKitDB as _CKDb

            resolved = self._resolved_path_key(path)
            for entry in _CKDb(self.db_path).list_model_caches():
                artifact_paths = {
                    self._resolved_path_key(artifact)
                    for artifact in entry.get("artifact_paths", [])
                }
                if resolved in artifact_paths:
                    return entry
        except Exception:
            return None
        return None

    @staticmethod
    def _review_source_label(raw_source: object) -> str:
        mapping = {
            None: "human",
            "human": "human",
            "auto": "machine",
            "auto_apriltag": "AprilTag",
            "auto_model": "model",
        }
        source = str(raw_source) if raw_source is not None else None
        return mapping.get(source, source.replace("_", " ") if source else "human")

    def _pending_machine_review_count(self) -> int:
        return sum(
            1
            for record in self._image_review_status.values()
            if record.get("label") and not record.get("verified")
        )

    def _mark_local_review_state(
        self,
        path: Path | str,
        *,
        label: str | None,
        label_source: str | None,
        verified: bool,
        confidence: float | None = None,
        auto_label_metadata: dict | None = None,
        verified_at: str | None = None,
    ) -> None:
        self._image_review_status[str(path)] = {
            "label": label,
            "confidence": confidence,
            "label_source": label_source,
            "verified": bool(verified),
            "verified_at": verified_at,
            "auto_label_metadata": auto_label_metadata or {},
        }
        self._refresh_review_candidate_indices()

    def _refresh_review_candidate_indices(self) -> None:
        self._review_candidate_indices = [
            index
            for index, path in enumerate(self.image_paths)
            if self._image_review_status.get(str(path), {}).get("label")
            and not self._image_review_status.get(str(path), {}).get("verified")
        ]

    def _update_review_panel(self) -> None:
        pending = self._pending_machine_review_count()
        verified_human = sum(
            1
            for record in self._image_review_status.values()
            if record.get("label") and record.get("verified")
        )
        selected_status = self._review_status_for_index(self.selected_point_index)
        selected_label = selected_status.get("label")
        selected_source = self._review_source_label(selected_status.get("label_source"))
        selected_confidence = selected_status.get("confidence")
        selected_state = (
            "verified" if selected_status.get("verified") else "needs review"
        )
        selected_color = "#4ec943" if selected_status.get("verified") else "#ffb454"

        if selected_label:
            selected_html = (
                f"<br><br><b>Selected:</b> {selected_label} "
                f"<span style='color:{selected_color};'>({selected_state})</span><br>"
                f"<b>Source:</b> {selected_source}"
            )
            if selected_confidence is not None:
                selected_html += (
                    f"<br><b>Confidence:</b> {float(selected_confidence):.3f}"
                )
        else:
            selected_html = "<br><br><b>Selected:</b> unlabeled"

        self.review_info.setText(
            "<div style='line-height:1.55;'>"
            f"<b>Pending machine review:</b> <span style='color:#ffb454;'>{pending:,}</span><br>"
            f"<b>Verified labels:</b> <span style='color:#4ec943;'>{verified_human:,}</span><br>"
            f"<b>Model predictions:</b> {'ready' if self._model_probs is not None else 'not loaded'}"
            f"{selected_html}"
            "</div>"
        )

        selected_can_approve = bool(selected_label) and not selected_status.get(
            "verified"
        )
        has_pending = pending > 0
        has_predictions = self._model_probs is not None and bool(
            self._model_class_names
        )

        if hasattr(self, "review_selected_btn"):
            self.review_selected_btn.setEnabled(selected_can_approve)
        if hasattr(self, "review_reject_btn"):
            self.review_reject_btn.setEnabled(selected_can_approve)
        if hasattr(self, "review_clear_unverified_btn"):
            self.review_clear_unverified_btn.setEnabled(has_pending)
        if hasattr(self, "review_all_btn"):
            self.review_all_btn.setEnabled(has_pending)
        if hasattr(self, "review_apply_model_btn"):
            self.review_apply_model_btn.setEnabled(has_predictions)
        if hasattr(self, "_approve_selected_review_action"):
            self._approve_selected_review_action.setEnabled(selected_can_approve)
        if hasattr(self, "_approve_all_review_action"):
            self._approve_all_review_action.setEnabled(has_pending)
        if hasattr(self, "_enter_review_mode_action"):
            self._enter_review_mode_action.setEnabled(has_pending)
        if hasattr(self, "_machine_label_action"):
            self._machine_label_action.setEnabled(self.project_path is not None)
        if hasattr(self, "_machine_label_toolbar_action"):
            self._machine_label_toolbar_action.setEnabled(self.project_path is not None)

    def _update_labeling_progress_indicator(self):
        """Update the status bar progress label and context panel progress bar."""
        in_labeling = self.explorer_mode == "labeling" and bool(
            self.candidate_indices or self.round_labeled_indices
        )
        if not in_labeling:
            self.labeling_progress_label.setVisible(False)
            self.progress_bar.setVisible(False)
            return

        # Batch progress: how many candidates in current set have been labeled
        candidate_set = set(self.candidate_indices) | set(self.round_labeled_indices)
        batch_total = len(candidate_set)
        labels = self.image_labels or []
        batch_labeled = sum(
            1 for i in self.round_labeled_indices if i < len(labels) and labels[i]
        )

        # Overall progress across entire dataset
        total_count = len(self.image_paths)
        total_labeled = sum(1 for lbl in labels if lbl)

        # Status bar label
        remaining = batch_total - batch_labeled
        self.labeling_progress_label.setText(
            f"Batch: {batch_labeled}/{batch_total}  ·  "
            f"Total: {total_labeled:,}/{total_count:,}  ·  "
            f"{remaining} remaining"
        )
        self.labeling_progress_label.setVisible(True)

        # Context panel progress bar
        self.progress_bar.setMaximum(batch_total)
        self.progress_bar.setValue(batch_labeled)
        pct = int(100 * batch_labeled / batch_total) if batch_total else 0
        self.progress_bar.setFormat(f"Batch: {batch_labeled}/{batch_total} ({pct}%)")
        self.progress_bar.setVisible(True)

    def _has_active_labeling_batch(self) -> bool:
        """Return True when labeling interactions should stay scoped to a batch."""
        return bool(self.candidate_indices or self.round_labeled_indices)

    def _set_view_mode_combo_value(self, mode: str) -> None:
        """Synchronize the mode combo without re-triggering change handlers."""
        if not hasattr(self, "view_mode_combo"):
            return
        idx = self.view_mode_combo.findData(mode)
        if idx < 0 or self.view_mode_combo.currentIndex() == idx:
            return
        self.view_mode_combo.blockSignals(True)
        self.view_mode_combo.setCurrentIndex(idx)
        self.view_mode_combo.blockSignals(False)

    def _invalidate_infinite_labeling_cache(self) -> None:
        """Drop cached diversity state used by infinite labeling."""
        self._infinite_label_distance_cache = None
        self._infinite_label_cluster_counts = None
        self._infinite_label_owner_cache = None

    def _labeled_index_signature(self, labeled_indices: list[int] | None = None) -> str:
        """Return a stable signature for the currently labeled dataset indices."""
        indices = (
            labeled_indices
            if labeled_indices is not None
            else self._labeled_image_indices()
        )
        if not indices:
            return "empty"
        payload = np.asarray(indices, dtype=np.int32).tobytes()
        return hashlib.sha1(payload).hexdigest()

    def _persist_infinite_labeling_cache(self) -> None:
        """Write the current infinite-labeling cache state to the project DB."""
        if (
            not self.db_path
            or self._infinite_label_distance_cache is None
            or self.embeddings is None
            or self._current_embedding_cache_id is None
        ):
            return
        try:
            from ..core.store.db import ClassKitDB

            labeled_indices = self._labeled_image_indices()
            ClassKitDB(self.db_path).save_infinite_label_cache(
                np.asarray(self._infinite_label_distance_cache, dtype=np.float32),
                (
                    np.asarray(self._infinite_label_cluster_counts, dtype=np.int32)
                    if self._infinite_label_cluster_counts is not None
                    else None
                ),
                (
                    np.asarray(self._infinite_label_owner_cache, dtype=np.int32)
                    if self._infinite_label_owner_cache is not None
                    else None
                ),
                meta={
                    "embedding_cache_id": int(self._current_embedding_cache_id),
                    "cluster_cache_id": (
                        int(self._current_cluster_cache_id)
                        if self._current_cluster_cache_id is not None
                        else None
                    ),
                    "label_signature": self._labeled_index_signature(labeled_indices),
                    "labeled_count": len(labeled_indices),
                },
            )
        except Exception:
            pass

    def _restore_persisted_infinite_labeling_cache(self) -> bool:
        """Load a previously saved infinite-labeling cache if it matches current state."""
        if (
            not self.db_path
            or self.embeddings is None
            or self._current_embedding_cache_id is None
            or len(self.image_paths) == 0
        ):
            return False
        try:
            from ..core.store.db import ClassKitDB

            cached = ClassKitDB(self.db_path).get_most_recent_infinite_label_cache(
                embedding_cache_id=self._current_embedding_cache_id,
            )
        except Exception:
            return False

        if not cached:
            return False

        if cached.get("label_signature") != self._labeled_index_signature():
            return False

        cached_cluster_id = cached.get("cluster_cache_id")
        if cached_cluster_id is not None or self._current_cluster_cache_id is not None:
            if cached_cluster_id != self._current_cluster_cache_id:
                return False

        distance_cache = np.asarray(cached.get("distance_cache"), dtype=np.float32)
        if len(distance_cache) != len(self.image_paths):
            return False

        owner_cache = cached.get("owner_cache")
        if owner_cache is not None:
            owner_cache = np.asarray(owner_cache, dtype=np.int32)
            if len(owner_cache) != len(self.image_paths):
                return False
            self._infinite_label_owner_cache = owner_cache
        else:
            self._infinite_label_owner_cache = None

        cluster_counts = cached.get("cluster_counts")
        if self.cluster_assignments is None:
            if cluster_counts is not None:
                return False
            self._infinite_label_cluster_counts = None
        else:
            cluster_ids = np.asarray(self.cluster_assignments, dtype=int)
            expected_size = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
            if cluster_counts is None:
                return False
            cluster_counts = np.asarray(cluster_counts, dtype=np.int32)
            if len(cluster_counts) != expected_size:
                return False
            self._infinite_label_cluster_counts = cluster_counts

        self._infinite_label_distance_cache = distance_cache
        return True

    def _ensure_infinite_labeling_cache(self) -> np.ndarray | None:
        """Build the infinite-labeling distance and cluster caches on demand."""
        num_images = len(self.image_paths)
        if num_images == 0:
            self._invalidate_infinite_labeling_cache()
            return None

        if self.embeddings is None:
            self._invalidate_infinite_labeling_cache()
            return None

        embeddings = np.asarray(self.embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != num_images:
            self._invalidate_infinite_labeling_cache()
            return None

        if (
            self._infinite_label_distance_cache is not None
            and len(self._infinite_label_distance_cache) == num_images
        ):
            return self._infinite_label_distance_cache

        if self._restore_persisted_infinite_labeling_cache():
            return self._infinite_label_distance_cache

        labeled_indices = self._labeled_image_indices()
        if labeled_indices:
            reference_indices = self._diversity_reference_indices(labeled_indices)
            result = self._min_distance_and_owner_to_reference_scores(
                list(range(num_images)), reference_indices
            )
            if result is None:
                self._invalidate_infinite_labeling_cache()
                return None
            cache, owners = result
            self._infinite_label_distance_cache = cache.astype(np.float32, copy=False)
            self._infinite_label_owner_cache = owners.astype(np.int32, copy=False)
        else:
            self._infinite_label_distance_cache = np.full(
                num_images, np.inf, dtype=np.float32
            )
            self._infinite_label_owner_cache = np.full(num_images, -1, dtype=np.int32)

        assignments = self.cluster_assignments
        if (
            assignments is not None
            and len(assignments) == num_images
            and labeled_indices
        ):
            cluster_ids = np.asarray(assignments, dtype=int)
            labeled_clusters = cluster_ids[labeled_indices]
            counts = np.bincount(
                labeled_clusters,
                minlength=int(cluster_ids.max()) + 1,
            )
            self._infinite_label_cluster_counts = counts.astype(np.int32, copy=False)
        elif assignments is not None and len(assignments) == num_images:
            cluster_ids = np.asarray(assignments, dtype=int)
            self._infinite_label_cluster_counts = np.zeros(
                int(cluster_ids.max()) + 1,
                dtype=np.int32,
            )
        else:
            self._infinite_label_cluster_counts = None

        self._persist_infinite_labeling_cache()

        return self._infinite_label_distance_cache

    def _update_infinite_labeling_cache_for_new_label(
        self, index: int, *, allow_build: bool = True
    ) -> bool:
        """Incrementally update infinite-labeling caches after one new label."""
        cache = self._infinite_label_distance_cache
        if cache is None or len(cache) != len(self.image_paths):
            if not self._restore_persisted_infinite_labeling_cache():
                if not allow_build:
                    return False
                cache = self._ensure_infinite_labeling_cache()
            else:
                cache = self._infinite_label_distance_cache
        if cache is None or self.embeddings is None:
            return False

        embeddings = np.asarray(self.embeddings)
        if embeddings.ndim != 2 or index < 0 or index >= embeddings.shape[0]:
            self._invalidate_infinite_labeling_cache()
            return False

        ref = embeddings[index].astype(np.float32, copy=False)
        deltas = embeddings.astype(np.float32, copy=False) - ref
        distances = np.einsum("ij,ij->i", deltas, deltas, optimize=True).astype(
            np.float32,
            copy=False,
        )
        self._infinite_label_distance_cache = np.minimum(cache, distances)
        self._infinite_label_distance_cache[index] = 0.0
        if self._infinite_label_owner_cache is None or len(
            self._infinite_label_owner_cache
        ) != len(self.image_paths):
            self._infinite_label_owner_cache = np.full(
                len(self.image_paths), -1, dtype=np.int32
            )
        improved = distances < cache
        self._infinite_label_owner_cache[improved] = int(index)
        self._infinite_label_owner_cache[index] = int(index)

        assignments = self.cluster_assignments
        counts = self._infinite_label_cluster_counts
        if (
            counts is not None
            and assignments is not None
            and 0 <= index < len(assignments)
        ):
            cluster_id = int(assignments[index])
            if 0 <= cluster_id < len(counts):
                counts[cluster_id] += 1

        self._persist_infinite_labeling_cache()
        return True

    def _update_infinite_labeling_cache_for_removed_label(
        self, index: int, *, allow_build: bool = True
    ) -> bool:
        """Update infinite-labeling caches after removing one existing label."""
        if index < 0 or index >= len(self.image_paths):
            return False

        labeled_indices = self._labeled_image_indices()
        if index not in labeled_indices:
            return False

        cache = self._infinite_label_distance_cache
        owners = self._infinite_label_owner_cache
        if (
            cache is None
            or owners is None
            or len(cache) != len(self.image_paths)
            or len(owners) != len(self.image_paths)
        ):
            if not self._restore_persisted_infinite_labeling_cache():
                if not allow_build:
                    return False
                cache = self._ensure_infinite_labeling_cache()
                owners = self._infinite_label_owner_cache
            else:
                cache = self._infinite_label_distance_cache
                owners = self._infinite_label_owner_cache

        if cache is None or owners is None:
            return False

        if len(labeled_indices) > 256:
            self._invalidate_infinite_labeling_cache()
            return False

        remaining_reference_indices = [i for i in labeled_indices if i != index]
        updated_cache = np.asarray(cache, dtype=np.float32).copy()
        updated_owners = np.asarray(owners, dtype=np.int32).copy()

        if remaining_reference_indices:
            affected_indices = [
                int(candidate_index)
                for candidate_index, owner in enumerate(updated_owners)
                if int(owner) == int(index) or candidate_index == index
            ]
            if affected_indices:
                result = self._min_distance_and_owner_to_reference_scores(
                    affected_indices,
                    remaining_reference_indices,
                )
                if result is None:
                    self._invalidate_infinite_labeling_cache()
                    return False
                rescored_cache, rescored_owners = result
                updated_cache[affected_indices] = rescored_cache
                updated_owners[affected_indices] = rescored_owners
        else:
            updated_cache = np.full(len(self.image_paths), np.inf, dtype=np.float32)
            updated_owners = np.full(len(self.image_paths), -1, dtype=np.int32)

        assignments = self.cluster_assignments
        counts = self._infinite_label_cluster_counts
        if (
            counts is not None
            and assignments is not None
            and 0 <= index < len(assignments)
        ):
            cluster_id = int(assignments[index])
            if 0 <= cluster_id < len(counts) and counts[cluster_id] > 0:
                counts[cluster_id] -= 1

        self._infinite_label_distance_cache = updated_cache
        self._infinite_label_owner_cache = updated_owners
        return True

    def _clear_explorer_selection_lock(self, clear_preview: bool = False) -> None:
        """Reset selection state so explorer preview returns to hover-driven behavior."""
        self.selected_point_index = None
        self.hover_locked = False
        self.request_update_explorer_selection(None)
        if clear_preview:
            self.clear_preview_display()
        else:
            self.update_knn_panel(None)

    def _ask_yes_no(self, title: str, text: str) -> bool:
        """Reusable yes/no prompt helper."""
        reply = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def _tick_autosave_heartbeat(self):
        """Animate autosave heartbeat indicator in the status bar."""
        self._autosave_heartbeat_on = not self._autosave_heartbeat_on
        self._update_autosave_heartbeat_text()

    def _clear_candidate_state(self, persist: bool = False) -> None:
        """Reset candidate and current labeling-batch state."""
        self.candidate_indices = []
        self.round_labeled_indices = []
        self._labeling_flow_mode = "batch"
        self._labeling_navigation_scope = "pool"

        if self.explorer_mode == "labeling":
            self.set_explorer_mode("explore")

        if not persist or not self.db_path:
            return

        try:
            from ..core.store.db import ClassKitDB

            ClassKitDB(self.db_path).save_candidate_cache([])
        except Exception:
            pass

    def _reset_labeling_hover_session(self, persist: bool = False) -> None:
        """Clear active labeling pool state while staying in labeling mode."""
        self.candidate_indices = []
        self.round_labeled_indices = []
        self._labeling_flow_mode = "batch"
        self._labeling_navigation_scope = "pool"

        if not persist or not self.db_path:
            return

        try:
            from ..core.store.db import ClassKitDB

            ClassKitDB(self.db_path).save_candidate_cache([])
        except Exception:
            pass

    def _clear_embedding_projection_view(self) -> None:
        """Clear the visible explorer plot when embedding-space data is stale."""
        if not hasattr(self, "explorer"):
            return
        if self._show_model_umap or self._show_model_pca:
            return
        self.explorer.clear_data()

    def _invalidate_embedding_downstream_state(
        self, persist_candidates: bool = True
    ) -> None:
        """Invalidate analysis artifacts derived from embeddings."""
        self.cluster_assignments = None
        self._current_cluster_cache_id = None
        self.umap_coords = None
        self._clear_candidate_state(persist=persist_candidates)
        self._clear_embedding_projection_view()

    def _invalidate_image_set_dependent_state(self) -> None:
        """Invalidate in-memory state tied to the current project image set."""
        self.embeddings = None
        self._current_embedding_cache_id = None
        self._invalidate_infinite_labeling_cache()
        self._invalidate_embedding_downstream_state(persist_candidates=True)
        self._model_probs = None
        self._model_class_names = None
        self.umap_model_coords = None
        self.pca_model_coords = None
        self._al_candidates = None
        self.image_confidences = [None] * len(self.image_paths)
        self._show_model_umap = False
        self._show_model_pca = False
        if self.explorer_mode == "predictions":
            self.set_explorer_mode("explore")
        self._clear_metrics_display()
        if hasattr(self, "btn_umap_embedding"):
            self.btn_umap_embedding.setChecked(True)
        if hasattr(self, "btn_umap_model"):
            self.btn_umap_model.setChecked(False)
            self._set_model_projection_buttons_enabled(False)
        if hasattr(self, "btn_pca_model"):
            self.btn_pca_model.setChecked(False)

    def _update_autosave_heartbeat_text(self):
        """Render autosave status in bottom-right status bar indicator."""
        indicator = "●" if self._autosave_heartbeat_on else "○"
        pending = len(self._pending_label_updates)
        period_sec = max(1, self._autosave_interval_ms // 1000)
        if not self.project_path:
            text = f"Autosave {indicator} no project"
        elif pending > 0:
            text = f"Autosave {indicator} {period_sec}s pending {pending}"
        elif self._autosave_last_save_time is not None:
            last_saved = time.strftime(
                "%H:%M:%S", time.localtime(self._autosave_last_save_time)
            )
            text = f"Autosave {indicator} {period_sec}s saved {last_saved}"
        else:
            text = f"Autosave {indicator} {period_sec}s idle"
        self.autosave_indicator.setText(text)

    def _save_project_runtime_settings(self):
        """Persist runtime settings such as autosave interval into project config."""
        if not self.project_path:
            return
        project_config_path = self._project_config_path()
        config = {}
        if project_config_path.exists():
            with open(project_config_path, "r") as f:
                config = json.load(f)
        config["autosave_interval_ms"] = int(self._autosave_interval_ms)
        config["marker_size_multiplier"] = float(self._marker_size_multiplier)
        project_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(project_config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _save_project_classes(self, classes: list[str]) -> None:
        """Persist the current project class list into project config."""
        if not self.project_path:
            return
        project_config_path = self._project_config_path()
        config = {}
        if project_config_path.exists():
            with open(project_config_path, "r") as f:
                config = json.load(f)
        config["classes"] = list(classes)
        project_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(project_config_path, "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def _build_imported_scheme_dict(labels: list[str]) -> dict:
        """Return a single-factor scheme dictionary for imported class labels."""
        return {
            "name": "imported_labels",
            "description": "Single-factor labeling scheme created from imported source labels",
            "factors": [
                {
                    "name": "class",
                    "labels": list(labels),
                    "shortcut_keys": [""] * len(labels),
                }
            ],
            "training_modes": ["flat_custom", "flat_yolo"],
        }

    def _replace_project_schema_from_labels(self, labels: list[str]) -> None:
        """Replace the project classes and scheme with imported flat labels."""
        if not labels:
            return

        normalized_labels = [
            str(label).strip() for label in labels if str(label).strip()
        ]
        scheme_dict = self._build_imported_scheme_dict(normalized_labels)

        if self.project_path:
            config = self._load_project_config()
            config["classes"] = normalized_labels
            config_path = self._project_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as handle:
                json.dump(config, handle, indent=2)

            scheme_path = self._project_scheme_path()
            scheme_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scheme_path, "w") as handle:
                json.dump(scheme_dict, handle, indent=2)

        self.classes = normalized_labels
        self.rebuild_label_buttons()
        self.setup_label_shortcuts()
        self._refresh_shortcut_help()
        self.update_context_panel()

    def _project_image_count(self) -> int:
        """Return the number of images currently registered in the project DB."""
        if not self.db_path:
            return len(self.image_paths or [])
        from ..core.store.db import ClassKitDB

        return ClassKitDB(self.db_path).count_images()

    def _prompt_schema_mismatch_resolution(
        self,
        source_root: Path,
        imported_labels: list[str],
        *,
        can_rewrite: bool,
    ) -> str:
        """Ask how to handle imported labels that do not fit the current schema."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Source Labels Do Not Match Project Schema")
        imported_preview = ", ".join(imported_labels[:8]) or "(none)"
        current_preview = ", ".join((self.classes or [])[:8]) or "(none)"
        if len(imported_labels) > 8:
            imported_preview += ", ..."
        if len(self.classes or []) > 8:
            current_preview += ", ..."
        msg.setText(
            f"Source '{source_root.name}' contains labels that do not match the current project schema."
        )
        msg.setInformativeText(
            "Imported labels: "
            f"{imported_preview}\n"
            "Project schema: "
            f"{current_preview}\n\n"
            + (
                "This is the first source in the project, so you can replace the project schema with these imported labels, import images only, or cancel."
                if can_rewrite
                else "You can import the images without labels, or cancel this source import."
            )
        )

        rewrite_button = None
        if can_rewrite:
            rewrite_button = msg.addButton(
                "Rewrite Schema and Import Labels",
                QMessageBox.AcceptRole,
            )
        import_images_only_button = msg.addButton(
            "Import Images Only",
            QMessageBox.ActionRole,
        )
        cancel_button = msg.addButton(QMessageBox.Cancel)
        msg.exec()

        clicked = msg.clickedButton()
        if rewrite_button is not None and clicked == rewrite_button:
            return "rewrite"
        if clicked == import_images_only_button:
            return "images_only"
        if clicked == cancel_button:
            return "cancel"
        return "cancel"

    def _resolve_ingest_request(self, source_root: Path) -> dict | None:
        """Return the next ingest request, prompting when source labels mismatch."""
        from ..core.data.source_import import build_source_import_plan

        plan = build_source_import_plan(source_root)
        imported_labels = [
            str(label).strip() for label in plan.discovered_labels if str(label).strip()
        ]
        project_labels = [
            str(label).strip() for label in (self.classes or []) if str(label).strip()
        ]
        placeholder_labels = ["class_1", "class_2"]
        project_label_set = set(project_labels)
        imported_label_set = set(imported_labels)
        mismatch = bool(imported_labels) and (
            not project_labels
            or project_labels == placeholder_labels
            or not imported_label_set.issubset(project_label_set)
        )

        if not mismatch:
            return {"source_root": source_root, "import_labels": True}

        can_rewrite = self._project_image_count() == 0
        resolution = self._prompt_schema_mismatch_resolution(
            source_root,
            imported_labels,
            can_rewrite=can_rewrite,
        )
        if resolution == "rewrite":
            self._replace_project_schema_from_labels(imported_labels)
            return {"source_root": source_root, "import_labels": True}
        if resolution == "images_only":
            return {"source_root": source_root, "import_labels": False}
        return None

    def _save_last_training_settings(self) -> None:
        """Persist the most recent training dialog settings into the project config."""
        if not self.project_path or not isinstance(self._last_training_settings, dict):
            return
        project_config_path = self._project_config_path()
        config = {}
        if project_config_path.exists():
            with open(project_config_path, "r") as f:
                config = json.load(f)
        config["last_training_settings"] = dict(self._last_training_settings)
        project_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(project_config_path, "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def _nearest_computer_friendly_size(value: object) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 224
        rounded = int(round(numeric / 32.0) * 32)
        return max(32, min(512, rounded))

    def _estimate_average_image_dimensions(
        self, max_samples: int = 128
    ) -> tuple[float, float] | None:
        """Estimate average image dimensions from a bounded sample of project images."""
        if not self.image_paths:
            return None

        sample_paths = list(self.image_paths)
        if len(sample_paths) > max_samples:
            indices = np.linspace(0, len(sample_paths) - 1, num=max_samples, dtype=int)
            sample_paths = [sample_paths[int(idx)] for idx in indices]

        widths = []
        heights = []
        try:
            from PIL import Image
        except Exception:
            return None

        for path in sample_paths:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except Exception:
                continue
            if width > 0 and height > 0:
                widths.append(float(width))
                heights.append(float(height))

        if not widths or not heights:
            return None
        return float(np.mean(widths)), float(np.mean(heights))

    def _default_training_settings_from_project(self) -> dict:
        """Build training-dialog defaults from the current project's image sizes."""
        average_dims = self._estimate_average_image_dimensions()
        if average_dims is None:
            return {}

        avg_width, avg_height = average_dims
        return {
            "tiny_width": self._nearest_computer_friendly_size(avg_width),
            "tiny_height": self._nearest_computer_friendly_size(avg_height),
            "custom_input_size": self._nearest_computer_friendly_size(
                (avg_width + avg_height) / 2.0
            ),
        }

    def _get_recent_project_training_settings(self) -> dict:
        """Return the most recent persisted training settings for this project."""
        if (
            isinstance(self._last_training_settings, dict)
            and self._last_training_settings
        ):
            return dict(self._last_training_settings)

        if self.db_path:
            try:
                from ..core.store.db import ClassKitDB

                recent_model = ClassKitDB(self.db_path).get_most_recent_model_cache()
                meta = (
                    recent_model.get("meta") if isinstance(recent_model, dict) else None
                )
                training_settings = (
                    meta.get("training_settings") if isinstance(meta, dict) else None
                )
                if isinstance(training_settings, dict) and training_settings:
                    self._last_training_settings = dict(training_settings)
                    return dict(self._last_training_settings)
            except Exception:
                pass

        return self._default_training_settings_from_project()

    def _list_recent_trainable_model_paths(self) -> list[str]:
        """Return project-history model artifacts usable as training warm starts."""
        if not self.db_path:
            return []

        try:
            from ..core.store.db import ClassKitDB

            ordered: list[str] = []
            seen = set()
            for entry in ClassKitDB(self.db_path).list_model_caches():
                for raw_path in entry.get("artifact_paths") or []:
                    try:
                        artifact = Path(str(raw_path)).expanduser().resolve()
                    except Exception:
                        continue
                    if not artifact.exists() or artifact.suffix.lower() not in {
                        ".pt",
                        ".pth",
                    }:
                        continue
                    artifact_text = str(artifact)
                    if artifact_text not in seen:
                        seen.add(artifact_text)
                        ordered.append(artifact_text)
            return ordered
        except Exception:
            return []

    def _save_preview_enhancement_settings(self) -> None:
        """Persist preview enhancement settings into the project config."""
        if not self.project_path:
            return
        project_config_path = self._project_config_path()
        config = {}
        if project_config_path.exists():
            with open(project_config_path, "r") as f:
                config = json.load(f)
        config["enhance_enabled"] = bool(
            getattr(self.preview_canvas, "use_clahe", False)
            if hasattr(self, "preview_canvas")
            else False
        )
        config["clahe_clip"] = float(self.clahe_clip)
        config["clahe_grid"] = list(self.clahe_grid)
        project_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(project_config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _reset_analysis_view(self) -> None:
        """Return the explorer to embedding-space Explore mode for analysis runs."""
        self.set_explorer_mode("explore")
        self._show_model_umap = False
        self._show_model_pca = False
        if hasattr(self, "btn_umap_embedding"):
            self.btn_umap_embedding.setChecked(True)
        if hasattr(self, "btn_umap_model"):
            self.btn_umap_model.setChecked(False)
        if hasattr(self, "btn_pca_model"):
            self.btn_pca_model.setChecked(False)

    def on_autosave_interval_changed(self, seconds: int):
        """Handle autosave interval changes from the UI."""
        self._autosave_interval_ms = max(1000, int(seconds) * 1000)
        self._autosave_timer.setInterval(self._autosave_interval_ms)
        self._save_project_runtime_settings()
        self._update_autosave_heartbeat_text()
        self.status.showMessage(f"Autosave interval set to {seconds}s")

    def on_marker_size_changed(self, value: float) -> None:
        """Apply a global marker-size multiplier to the explorer."""
        self._marker_size_multiplier = max(0.5, min(3.0, float(value)))
        if hasattr(self, "explorer") and self.explorer is not None:
            self.explorer.set_marker_size_multiplier(self._marker_size_multiplier)
            self.request_update_explorer_plot()
        self._save_project_runtime_settings()
        self.status.showMessage(
            f"Marker size set to {self._marker_size_multiplier:.1f}x"
        )

    def _flush_pending_label_updates(self, force: bool = False):
        """Persist buffered label updates to DB on autosave cadence or forced flush."""
        if not self._pending_label_updates:
            self._update_autosave_heartbeat_text()
            return
        if not self.db_path:
            if force:
                self._pending_label_updates = {}
            self._update_autosave_heartbeat_text()
            return

        try:
            from ..core.store.db import ClassKitDB

            db = ClassKitDB(self.db_path)
            # Create a local copy to avoid mutation during iteration
            updates = dict(self._pending_label_updates)
            count = len(updates)

            updated_count = db.update_labels_batch(updates)

            if updated_count < count:
                import logging

                logging.getLogger(__name__).warning(
                    f"Autosave: only {updated_count} of {count} labels were updated in DB. "
                    "This usually means some image paths in memory don't match the DB exactly."
                )

            # Only clear the ones we just successfully updated
            for path in updates:
                self._pending_label_updates.pop(path, None)

            self._autosave_last_save_time = time.time()
            if self._autosave_timer.isActive():
                self._autosave_timer.stop()

            self.status.showMessage(f"Autosaved {updated_count} labels", 2000)
        except Exception as exc:
            self.status.showMessage(f"Autosave failed: {exc}")
            import logging

            logging.getLogger(__name__).error(f"Autosave failure: {exc}", exc_info=True)

        self._update_autosave_heartbeat_text()

    def _autoload_cached_analysis(self, db):
        """Restore the latest valid cached embeddings, clusters, UMAP, and candidates."""
        restored = []
        cached_embeddings = db.get_most_recent_embeddings()
        if cached_embeddings is not None and self.embeddings is None:
            embeddings, metadata = cached_embeddings
            self.embeddings = embeddings
            self._current_embedding_cache_id = metadata.get("id")
            restored.append("embeddings")

        cached_cluster = db.get_most_recent_cluster_cache(
            self._current_embedding_cache_id
        )
        if cached_cluster is not None and self.cluster_assignments is None:
            self.cluster_assignments = cached_cluster["assignments"]
            self._current_cluster_cache_id = cached_cluster.get("id")
            restored.append("clusters")

        cached_umap = db.get_most_recent_umap_cache(
            embedding_cache_id=self._current_embedding_cache_id,
        )
        if cached_umap is not None and self.umap_coords is None:
            self.umap_coords = cached_umap["coords"]
            self.last_umap_params = {
                "n_neighbors": cached_umap.get("n_neighbors", 15),
                "min_dist": cached_umap.get("min_dist", 0.1),
            }
            restored.append("UMAP")

        cached_candidates = db.get_most_recent_candidate_cache()
        if cached_candidates is not None and not self.candidate_indices:
            self._restore_cached_candidate_batch(
                cached_candidates.get("candidate_indices", [])
            )
            self.set_explorer_mode("labeling")
            restored.append("candidates")

        if restored:
            self.status.showMessage(f"Restored cached {', '.join(restored)}", 5000)

    def _restore_cached_candidate_batch(self, candidate_indices) -> None:
        """Restore a cached candidate set while honoring labels already stored in the DB."""
        labels = self.image_labels or []
        restored_candidates = []
        restored_labeled = []
        seen = set()

        for raw_index in candidate_indices or []:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(self.image_paths) or index in seen:
                continue
            seen.add(index)
            restored_candidates.append(index)
            if index < len(labels) and labels[index]:
                restored_labeled.append(index)

        self.candidate_indices = restored_candidates
        self.round_labeled_indices = restored_labeled

    def _apply_cached_predictions(self, cached_preds, db):
        """Apply cached prediction data and restore model-space projections."""
        summary = None
        get_recent_model = getattr(db, "get_most_recent_model_cache", None)
        if callable(get_recent_model):
            try:
                recent_model = get_recent_model()
            except Exception:
                recent_model = None
            if isinstance(recent_model, dict):
                summary = self._validation_summary_from_value(
                    recent_model.get("best_val_acc"),
                    prefix="Saved held-out validation accuracy",
                )
                self._set_active_evaluation_from_meta(recent_model.get("meta"))
        if cached_preds.get("evaluation"):
            self._set_active_evaluation_from_meta(cached_preds)
        self._set_heldout_validation_summary(summary)
        self._model_probs = cached_preds["probs"]
        self._model_class_names = cached_preds["class_names"]
        self._active_model_mode = cached_preds.get("active_model_mode", "yolo")
        self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
        self._set_model_projection_buttons_enabled(True)
        self._update_al_status()
        cached_mumap = db.get_most_recent_umap_cache(kind="model")
        if cached_mumap:
            self.umap_model_coords = cached_mumap["coords"]
        cached_mpca = db.get_most_recent_umap_cache(kind="model_pca")
        if cached_mpca:
            self.pca_model_coords = cached_mpca["coords"]
        self._activate_predictions_view_if_available()
        self._evaluate_model_on_labeled(activate_metrics_tab=False)

    def _activate_predictions_view_if_available(self) -> None:
        """Show prediction coloring after startup restore when no labeling batch is active."""
        if self._model_probs is None or self.candidate_indices:
            return
        if self.explorer_mode != "predictions":
            self.set_explorer_mode("predictions")
        else:
            self.update_explorer_plot(force_fit=True)
            self.update_context_panel()

    @staticmethod
    def _predictions_cover_latest_model(cached_preds, recent_model) -> bool:
        """Return True when a cached prediction set is fresh enough for the latest model."""
        if not cached_preds:
            return False
        if not recent_model:
            return True

        cached_model_id = cached_preds.get("model_cache_id")
        recent_model_id = recent_model.get("id")
        if cached_model_id is not None and recent_model_id is not None:
            try:
                return int(cached_model_id) == int(recent_model_id)
            except (TypeError, ValueError):
                return False

        cached_ts = str(cached_preds.get("timestamp") or "")
        recent_ts = str(recent_model.get("timestamp") or "")
        if not recent_ts:
            return True
        if not cached_ts:
            return False
        return cached_ts >= recent_ts

    def _activate_saved_labels_view_if_available(self) -> None:
        """Show saved label coloring after project load when no batch or predictions override it."""
        if self._model_probs is not None or self.candidate_indices:
            return
        if not any(label for label in self.image_labels or []):
            return
        if self.explorer_mode != "labeling":
            self.set_explorer_mode("labeling")
        else:
            self.update_explorer_plot(force_fit=True)
            self.update_context_panel()

    def _autoload_yolo_classifier(self, db):
        """Restore recent YOLO predictions or rerun the published classifier."""
        if not self.project_path:
            return False
        yolo_ckpt = classkit_model_dir(self.project_path) / "yolo_classifier_latest.pt"
        if not yolo_ckpt.exists():
            return False
        if self._yolo_model_path is not None or self._model_probs is not None:
            return False

        cached_preds = db.get_most_recent_prediction_cache()
        if cached_preds:
            self._apply_cached_predictions(cached_preds, db)
            self.status.showMessage("Restored cached predictions", 5000)
            return False

        self._yolo_model_path = yolo_ckpt
        self._active_model_mode = "yolo"
        self.status.showMessage(
            f"Restoring published YOLO classifier: {yolo_ckpt.name}", 5000
        )
        QTimer.singleShot(
            200,
            lambda p=yolo_ckpt: self._run_yolo_inference(
                p,
                force_monochrome=self._resolve_yolo_force_monochrome(p),
                on_success=lambda _result: self._activate_predictions_view_if_available(),
            ),
        )
        return True  # signal caller to skip DB model check

    def _autoload_model_from_db(self):
        """Restore the most recent model predictions or trained model from the project DB."""
        if not self.db_path or self._model_probs is not None:
            return
        try:
            from ..core.store.db import ClassKitDB as _CKDb

            _db2 = _CKDb(self.db_path)
            _recent = _db2.get_most_recent_model_cache()
            cached_preds = _db2.get_most_recent_prediction_cache()
            if self._predictions_cover_latest_model(cached_preds, _recent):
                self._apply_cached_predictions(cached_preds, _db2)
                self.status.showMessage("Restored cached predictions", 5000)
            else:
                if _recent:
                    self.status.showMessage("Restoring most recent trained model", 5000)
                    QTimer.singleShot(
                        200,
                        lambda e=_recent: self._load_model_from_cache_entry(
                            e,
                            on_success=lambda: self._activate_predictions_view_if_available(),
                        ),
                    )
        except Exception:
            pass

    def try_autoload_cached_artifacts(self, db=None):
        """Offer to autoload latest embeddings, cluster assignments, and UMAP."""
        if not self.db_path:
            return

        if db is None:
            from ..core.store.db import ClassKitDB

            db = ClassKitDB(self.db_path)

        try:
            self._autoload_cached_analysis(db)

            if self._autoload_yolo_classifier(db):
                return  # skip DB check after filesystem hit

            self._autoload_model_from_db()
            self._activate_saved_labels_view_if_available()

            self.update_explorer_plot(force_fit=True)
            self.update_context_panel()
        except Exception:
            pass

    def _clear_label_shortcuts(self) -> None:
        """Remove existing shortcut objects before rebuilding bindings."""
        for shortcut in self._label_shortcuts:
            shortcut.setParent(None)
        self._label_shortcuts = []

    def _load_scheme_shortcuts(self) -> dict[str, str]:
        """Load flat label shortcuts from the project scheme when available."""
        scheme_shortcuts: dict[str, str] = {}
        if not self.project_path:
            return scheme_shortcuts
        try:
            scheme_path = self._project_scheme_path()
            if scheme_path.exists():
                with open(scheme_path) as _f:
                    scheme_dict = json.load(_f)
                factors = scheme_dict.get("factors", [])
                if factors:
                    factor = factors[0]
                    for label, key in zip(
                        factor.get("labels", []), factor.get("shortcut_keys", [])
                    ):
                        if key:
                            scheme_shortcuts[label] = key
        except Exception:
            return {}
        return scheme_shortcuts

    def _register_label_shortcut(self, key: str | QKeySequence, callback) -> None:
        """Create one QShortcut and track it for later teardown."""
        sequence = key if isinstance(key, QKeySequence) else QKeySequence(key)
        shortcut = QShortcut(sequence, self)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(callback)
        self._label_shortcuts.append(shortcut)

    def _install_label_assignment_shortcuts(
        self, scheme_shortcuts: dict[str, str]
    ) -> None:
        """Install class-label shortcuts when not using the multi-factor stepper."""
        if self._stepper is not None:
            return

        for i, class_name in enumerate(self.classes[:9], start=1):
            if class_name not in scheme_shortcuts:
                self._register_label_shortcut(
                    str(i), lambda c=class_name: self.assign_label_to_selected(c)
                )

        for label, key in scheme_shortcuts.items():
            self._register_label_shortcut(
                key, lambda c=label: self.assign_label_to_selected(c)
            )

        self._register_label_shortcut(
            "0", lambda: self.assign_label_to_selected("unknown")
        )

    def _install_global_navigation_shortcuts(
        self, active: dict, defaults: dict
    ) -> None:
        """Install mode and navigation shortcuts shared across labeling modes."""

        def _key(action: str) -> QKeySequence:
            return QKeySequence(active.get(action, defaults.get(action, "")))

        shortcut_actions = [
            ("Explore mode", lambda: self.set_explorer_mode("explore")),
            ("Labeling mode", lambda: self.set_explorer_mode("labeling")),
            ("Review mode", self.enter_review_mode),
            ("Predictions mode", lambda: self.set_explorer_mode("predictions")),
            ("Approve review label", self.approve_selected_review_label),
            ("Reject review label", self.reject_selected_review_label),
            ("Sample next candidates", self.on_sample_next_triggered),
            ("Previous unlabeled", self.on_prev_image),
            ("Next unlabeled", self.on_next_image),
            ("Undo last label (Ctrl+Z)", self.undo_last_assignment),
        ]
        for action_name, callback in shortcut_actions:
            self._register_label_shortcut(_key(action_name), callback)

    def setup_label_shortcuts(self):
        """Create keyboard shortcuts for labeling and mode switching."""
        self._clear_label_shortcuts()

        from .dialogs import ShortcutEditorDialog

        defaults = dict(ShortcutEditorDialog.DEFAULT_SHORTCUTS)
        active = {**defaults, **self._custom_shortcuts}
        scheme_shortcuts = self._load_scheme_shortcuts()
        self._install_label_assignment_shortcuts(scheme_shortcuts)
        self._install_global_navigation_shortcuts(active, defaults)

    def _begin_command(self) -> bool:
        """Return False when command input should be ignored during active/cooldown windows."""
        now = time.monotonic()
        if self._command_busy:
            self._command_squelch_pending = True
            return False
        if now < self._command_block_until:
            self._command_squelch_pending = True
            return False
        self._command_busy = True
        self._set_command_controls_enabled(False)
        return True

    def _end_command(self, cooldown_seconds: float = 0.10):
        """Release command processing and apply short cooldown to drop queued key events."""
        self._command_busy = False
        now = time.monotonic()
        hold = cooldown_seconds
        if self._command_squelch_pending:
            hold = max(hold, 0.35)
            self._command_squelch_pending = False

        self._command_block_until = max(self._command_block_until, now + hold)
        QTimer.singleShot(int(hold * 1000), self._restore_command_controls)

    def _restore_command_controls(self):
        """Re-enable interactive command controls after cooldown expires."""
        if self._command_busy:
            return
        now = time.monotonic()
        if now < self._command_block_until:
            remaining_ms = max(10, int((self._command_block_until - now) * 1000))
            QTimer.singleShot(remaining_ms, self._restore_command_controls)
            return
        self._set_command_controls_enabled(True)

    # ── Background-job UI locking ─────────────────────────────────────────

    def _set_ui_busy(self, busy: bool) -> None:
        """Disable/re-enable toolbar and menu bar while a background job is running."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            toolbar.setEnabled(not busy)
        menubar = self.menuBar()
        if menubar is not None:
            menubar.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _job_start(self) -> None:
        """Increment active-job counter and lock the UI on the first job."""
        self._active_jobs += 1
        if self._active_jobs == 1:
            self._set_ui_busy(True)

    def _job_done(self) -> None:
        """Decrement active-job counter and unlock the UI when all jobs finish."""
        self._active_jobs = max(0, self._active_jobs - 1)
        if self._active_jobs == 0:
            self._set_ui_busy(False)

    def _threadpool_start(self, worker) -> None:
        """Register a job, connect the cleanup hook, and submit to the thread pool."""
        self._job_start()
        worker.signals.finished.connect(self._job_done)
        self.threadpool.start(worker)

    def _set_command_controls_enabled(self, enabled: bool):
        """Enable/disable controls and shortcuts that can generate rapid command bursts."""
        for attr in ("sample_btn", "prev_btn", "next_btn"):
            button = getattr(self, attr, None)
            if button is not None:
                button.setEnabled(enabled)

        labels_enabled = enabled and self.explorer_mode == "labeling"
        for i in range(self.label_buttons_layout.count()):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setEnabled(labels_enabled)

        for shortcut in self._label_shortcuts:
            shortcut.setEnabled(enabled)

    def request_update_explorer_plot(self, force_fit: bool = False):
        """Coalesce expensive UMAP redraws into a single latest update."""
        self._plot_refresh_pending = True
        self._plot_refresh_force_fit = self._plot_refresh_force_fit or force_fit
        if not self._plot_refresh_timer.isActive():
            self._plot_refresh_timer.start()

    def _flush_explorer_update(self):
        """Apply the latest queued explorer redraw request."""
        if not self._plot_refresh_pending:
            return
        force_fit = self._plot_refresh_force_fit
        self._plot_refresh_pending = False
        self._plot_refresh_force_fit = False
        self.update_explorer_plot(force_fit=force_fit)

    def request_preview_for_index(self, index: int, source: str = "hover"):
        """Coalesce preview updates so rapid input keeps only the latest target."""
        self._pending_preview_index = index
        self._pending_preview_source = source
        if not self._preview_refresh_timer.isActive():
            self._preview_refresh_timer.start()

    def request_update_context_panel(self):
        """Coalesce context panel updates to avoid frequent expensive HTML rebuilds."""
        self._context_refresh_pending = True
        if not self._context_refresh_timer.isActive():
            self._context_refresh_timer.start()

    def _flush_context_update(self):
        """Apply latest queued context panel update request."""
        if not self._context_refresh_pending:
            return
        self._context_refresh_pending = False
        self.update_context_panel()

    def _flush_preview_update(self):
        """Apply the latest queued preview update request."""
        index = self._pending_preview_index
        source = self._pending_preview_source
        self._pending_preview_index = None
        if index is None:
            return
        self.load_preview_for_index(index, source=source)

    def request_update_explorer_selection(self, selected_index: int | None):
        """Fast-path selection-only update; falls back to full redraw when needed."""
        if not hasattr(self, "explorer"):
            return
        if self.explorer.set_selected_index(selected_index):
            return
        self.request_update_explorer_plot()

    def refresh_label_history_strip(self):
        """Refresh bottom strip of recently labeled items for quick undo/relabel."""
        recent = list(reversed(self.label_history[-24:]))
        allow_thumbnail_load = (not self._command_busy) and (
            time.monotonic() >= self._command_block_until
        )
        loads_remaining = self._history_thumb_load_budget
        pending_uncached = False

        for slot_idx, slot in enumerate(self._history_slots):
            card, thumb, caption = slot
            if slot_idx >= len(recent):
                card.setProperty("history_index", None)
                card.setToolTip("")
                thumb.clear()
                caption.clear()
                card.hide()
                continue

            entry = recent[slot_idx]
            index = entry["index"]
            label = entry["label"]
            image_path = (
                self.image_paths[index] if index < len(self.image_paths) else None
            )

            card.setProperty("history_index", int(index))
            card.setToolTip(f"Click to undo label and relabel: {label}")
            caption.setText(f"{label}\n#{index}")
            thumb.clear()

            if image_path and image_path.exists():
                icon_key = str(image_path)
                pixmap = self._history_icon_cache.get(icon_key)
                if pixmap is None and allow_thumbnail_load and loads_remaining > 0:
                    pixmap = QPixmap(str(image_path))
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(
                            72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                        self._history_icon_cache[icon_key] = pixmap
                    loads_remaining -= 1
                elif pixmap is None:
                    pending_uncached = True
                if pixmap is not None and not pixmap.isNull():
                    thumb.setPixmap(pixmap)

            card.show()

        if pending_uncached and not self._history_refresh_timer.isActive():
            self._history_refresh_timer.start(80)

    def request_refresh_label_history_strip(self):
        """Coalesce strip rebuilds to avoid repeated widget churn during rapid labeling."""
        self._history_refresh_timer.start()

    def keyPressEvent(self, event) -> None:
        """Route digit keys to stepper when in multi-factor labeling mode."""
        if self._stepper is not None:
            key_text = event.text()
            if key_text and self._stepper.handle_key(key_text):
                event.accept()
                return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        """Handle clicks on history cards without QPushButton construction."""
        if event.type() == QEvent.MouseButtonRelease:
            index = watched.property("history_index") if watched is not None else None
            if index is not None:
                self.undo_label_from_history(int(index))
                return True
        return super().eventFilter(watched, event)

    def _remove_history_for_index(self, index: int):
        """Remove history entries for a point index."""
        self.label_history = [
            entry for entry in self.label_history if entry.get("index") != index
        ]

    def undo_label_from_history(self, index: int):
        """Undo a specific label from history and move selection back for relabel."""
        if not self._begin_command():
            return
        try:
            if index < 0 or index >= len(self.image_paths):
                return
            previous = (
                self.image_labels[index] if index < len(self.image_labels) else None
            )
            if not previous:
                return

            self._set_label_for_index(index, None)
            self.selected_point_index = index
            if index not in self.candidate_indices:
                self.candidate_indices.insert(0, index)
            self.round_labeled_indices = [
                i for i in self.round_labeled_indices if i != index
            ]
            self._remove_history_for_index(index)
            self.request_refresh_label_history_strip()
            self.hover_locked = True
            self.request_preview_for_index(index, source="undo")
            self.request_update_explorer_plot()
            self.request_update_context_panel()
            self.status.showMessage(
                f"Reverted label on point {index}; ready to relabel"
            )
        finally:
            self._end_command(0.12)

    def undo_last_assignment(self):
        """Undo the most recent assignment (Ctrl+Z)."""
        if not self._begin_command():
            return
        try:
            if not self.last_assigned_stack:
                self.status.showMessage("No recent assignment to undo")
                return

            last = self.last_assigned_stack.pop()
            index = last["index"]
            previous_label = last.get("previous_label")
            self._set_label_for_index(index, previous_label)

            if previous_label:
                if index not in self.round_labeled_indices:
                    self.round_labeled_indices.append(index)
            else:
                self.round_labeled_indices = [
                    i for i in self.round_labeled_indices if i != index
                ]
                if index not in self.candidate_indices:
                    self.candidate_indices.insert(0, index)

            if previous_label is None:
                self._remove_history_for_index(index)

            self.request_refresh_label_history_strip()

            self.selected_point_index = index
            self.hover_locked = True
            self.request_preview_for_index(index, source="undo")
            self.request_update_explorer_plot()
            self.request_update_context_panel()
            self.status.showMessage(f"Undid assignment for point {index}")
        finally:
            self._end_command(0.12)

    def _on_stepper_label_committed(self, composite_label: str) -> None:
        """Called when the stepper completes all factors for the current image."""
        if self.selected_point_index is None:
            return
        index = self.selected_point_index
        self._set_label_for_index(index, composite_label)
        self.round_labeled_indices = [
            i for i in self.round_labeled_indices if i != index
        ]
        self.round_labeled_indices.append(index)
        self._remove_history_for_index(index)
        self.label_history.append({"index": index, "label": composite_label})
        self.request_refresh_label_history_strip()
        self.on_label_assigned(composite_label)
        self.request_update_explorer_plot()
        self.request_update_context_panel()
        # Flush immediately after each composite label is finished to ensure data safety
        self._flush_pending_label_updates()
        self.on_next_image()

    def _set_label_for_index(self, index: int, label):
        """Set or clear label for an index and persist to DB."""
        if index < 0 or index >= len(self.image_paths):
            return

        if self.db_path:
            # Key must match the file_path string stored in DB exactly
            self._pending_label_updates[str(self.image_paths[index])] = label
            if not self._autosave_timer.isActive():
                self._autosave_timer.start()
            self._update_autosave_heartbeat_text()

        if not self.image_labels or len(self.image_labels) != len(self.image_paths):
            self.image_labels = [None] * len(self.image_paths)
        previous_label = self.image_labels[index]

        removed_cache_updated = False
        if previous_label and not label:
            removed_cache_updated = (
                self._update_infinite_labeling_cache_for_removed_label(
                    index,
                    allow_build=self._labeling_flow_mode == "infinite",
                )
            )

        self.image_labels[index] = label

        if bool(label) and not previous_label:
            self._update_infinite_labeling_cache_for_new_label(
                index,
                allow_build=self._labeling_flow_mode == "infinite",
            )
        elif previous_label and not label and removed_cache_updated:
            self._persist_infinite_labeling_cache()
        elif bool(label) != bool(previous_label):
            self._invalidate_infinite_labeling_cache()

    def on_explorer_background_double_click(self):
        """Return to hover mode when empty region is double-clicked."""
        if self.explorer_mode == "labeling":
            self._reset_labeling_hover_session(persist=True)
        self._clear_explorer_selection_lock(clear_preview=False)
        self.request_update_explorer_plot()
        self.request_update_context_panel()
        self.selection_info.setText(
            "<div style='line-height:1.5;'>"
            "<b>Selected Point:</b> none<br>"
            "<b>Hovered Point:</b> none<br>"
            "<b>Current Label:</b> unlabeled<br>"
            "Selection cleared. Hover candidates to preview; click one to select for labeling."
            "</div>"
        )
        if self.explorer_mode == "labeling":
            self.status.showMessage("Hover mode restored — active labeling set cleared")
        else:
            self.status.showMessage("Hover mode restored")

    def _get_navigation_pool(self):
        """Return indices eligible for next/prev navigation in labeling context."""
        if self.explorer_mode == "review" and self._review_candidate_indices:
            return self._review_candidate_indices
        if (
            self.explorer_mode == "labeling"
            and self._labeling_navigation_scope == "database"
        ):
            return []
        if self.candidate_indices:
            return self.candidate_indices
        return list(range(len(self.image_paths)))

    def _clear_label_buttons(self) -> None:
        """Remove all existing widgets from the label button layout."""
        for i in reversed(range(self.label_buttons_layout.count())):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def _load_label_scheme(self):
        """Load the project labeling scheme and first-factor shortcuts if available."""
        scheme = None
        scheme_shortcuts = {}
        if not self.project_path:
            return scheme, scheme_shortcuts
        try:
            from ..config.schemas import LabelingScheme

            scheme_path = self._project_scheme_path()
            if scheme_path.exists():
                with open(scheme_path) as _f:
                    scheme = LabelingScheme.from_dict(json.load(_f))
                factors = scheme.factors
                if factors:
                    first_factor = factors[0]
                    for label, key in zip(
                        first_factor.labels, first_factor.shortcut_keys
                    ):
                        if key:
                            scheme_shortcuts[label] = key
        except Exception:
            scheme = None
        return scheme, scheme_shortcuts

    def _build_stepper_label_buttons(self, scheme) -> None:
        """Build multi-factor stepper UI for composite label schemes."""
        from .widgets.factor_stepper import _build_qt_widget

        FactorStepperWidget = _build_qt_widget(scheme)
        stepper = FactorStepperWidget(scheme, parent=self.label_buttons_container)
        stepper.label_committed.connect(self._on_stepper_label_committed)
        stepper.skipped.connect(self.on_next_image)
        self.label_buttons_layout.addWidget(stepper, 0, 0, 1, 2)
        self._stepper = stepper

    def _resolve_label_button_shortcut(
        self, class_name: str, index: int, scheme_shortcuts: dict[str, str]
    ) -> str | None:
        """Return the display shortcut string for one flat class button."""
        shortcut = scheme_shortcuts.get(class_name)
        if not shortcut and index < 9:
            return str(index + 1)
        return shortcut

    @staticmethod
    def _resolve_class_button_color(class_color_map, class_name: str):
        """Resolve a stable button background color for a class name."""
        bg = class_color_map.get(class_name)
        if bg is None:
            bg = class_color_map.get(str(class_name))
        if bg is None:
            bg = class_color_map.get("unknown")
        return bg

    def _schema_category_order(
        self, extra_categories: list[str] | None = None
    ) -> list[str]:
        """Return the canonical schema-first category order used for ClassKit colors."""
        ordered: list[str] = []
        seen: set[str] = set()
        for raw_value in [*(self.classes or []), "unknown", *(extra_categories or [])]:
            category = str(raw_value).strip()
            if not category or category in seen:
                continue
            seen.add(category)
            ordered.append(category)
        return ordered

    def _schema_category_color_map(self, extra_categories: list[str] | None = None):
        """Return the shared category color map for buttons, explorer points, and tags."""
        order = self._schema_category_order(extra_categories=extra_categories)
        return build_category_color_map(order, category_order=order)

    def _label_tag_html(self, label: str | None, *, class_color_map) -> str:
        """Render a label as a colored chip using the shared schema color map."""
        if not label:
            return "unlabeled"
        bg = self._resolve_class_button_color(class_color_map, str(label))
        fg = best_text_color(bg)
        return (
            "<span style='display:inline-block; padding:1px 7px; border-radius:9px; "
            f"background-color:{to_hex(bg)}; color:{to_hex(fg)}; border:1px solid #2f2f2f;'>"
            f"{escape(str(label))}</span>"
        )

    def _build_flat_label_buttons(self, scheme_shortcuts: dict[str, str]) -> None:
        """Build one button per class plus the unknown class button."""
        self._stepper = None
        class_color_map = self._schema_category_color_map()

        for i, class_name in enumerate(self.classes):
            shortcut = self._resolve_label_button_shortcut(
                class_name, i, scheme_shortcuts
            )
            btn_text = f"[{shortcut}] {class_name}" if shortcut else class_name
            button = QPushButton(btn_text)
            bg = self._resolve_class_button_color(class_color_map, class_name)
            fg = best_text_color(bg)
            button.setStyleSheet(
                "text-align: left; padding: 4px; "
                f"background-color: {to_hex(bg)}; color: {to_hex(fg)};"
                "border: 1px solid #2f2f2f;"
            )
            button.clicked.connect(
                lambda checked=False, c=class_name: self.assign_label_to_selected(c)
            )
            button.setProperty("class_name", class_name)
            self.label_buttons_layout.addWidget(button, i // 2, i % 2)

        unknown_btn = QPushButton("[0] unknown")
        unknown_bg = class_color_map.get("unknown")
        unknown_fg = best_text_color(unknown_bg)
        unknown_btn.setStyleSheet(
            "text-align: left; padding: 4px; "
            f"background-color: {to_hex(unknown_bg)}; color: {to_hex(unknown_fg)};"
            "border: 1px solid #2f2f2f;"
        )
        unknown_btn.clicked.connect(
            lambda checked=False: self.assign_label_to_selected("unknown")
        )
        unknown_btn.setProperty("class_name", "unknown")
        total_btns = len(self.classes)
        self.label_buttons_layout.addWidget(
            unknown_btn, total_btns // 2, total_btns % 2
        )

        if hasattr(self, "class_search"):
            self.filter_label_buttons(self.class_search.text())

    def rebuild_label_buttons(self):
        """Rebuild class buttons shown in the left settings panel."""
        self._clear_label_buttons()
        scheme, scheme_shortcuts = self._load_label_scheme()

        if scheme is not None and len(scheme.factors) > 1:
            try:
                self._build_stepper_label_buttons(scheme)
                return
            except Exception:
                self._stepper = None

        self._build_flat_label_buttons(scheme_shortcuts)

    def filter_label_buttons(self, text: str):
        """Filter visible label buttons based on search text."""
        text = text.lower()
        for i in range(self.label_buttons_layout.count()):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget and hasattr(widget, "property"):
                class_name = widget.property("class_name")
                if class_name:
                    widget.setVisible(text in class_name.lower())

    @staticmethod
    def _mode_display_name(mode: str) -> str:
        return {
            "explore": "Explore",
            "labeling": "Labeling",
            "review": "Review",
            "predictions": "Predictions",
        }.get(str(mode), str(mode))

    def _prediction_labels_for_plot(self) -> list:
        """Return per-image predicted class labels for prediction-colored explorer mode."""
        if self._model_probs is None:
            return [None] * len(self.image_paths)

        probs = np.asarray(self._model_probs)
        if probs.ndim != 2:
            return [None] * len(self.image_paths)

        num_images = len(self.image_paths)
        out = [None] * num_images
        eval_n = min(num_images, probs.shape[0])

        names = list(self._model_class_names or [])
        for i in range(eval_n):
            pred_idx = int(np.argmax(probs[i]))
            if 0 <= pred_idx < len(names):
                out[i] = names[pred_idx]
            else:
                out[i] = f"pred_{pred_idx}"
        return out

    @staticmethod
    def _format_prediction_confidence(value: float | None) -> str:
        """Format prediction confidence for compact panel and tooltip display."""
        if value is None:
            return "n/a"
        try:
            numeric = float(value)
        except Exception:
            return "n/a"
        if 0.0 <= numeric <= 1.0:
            return f"{numeric:.1%}"
        return f"{numeric:.3f}"

    def _prediction_summary_for_index(
        self, index: int, *, top_k: int = 3
    ) -> dict | None:
        """Return predicted class details for one point in the current model output."""
        if index is None or index < 0 or self._model_probs is None:
            return None

        probs = np.asarray(self._model_probs)
        if probs.ndim != 2 or index >= probs.shape[0]:
            return None

        row = np.asarray(probs[index], dtype=float).reshape(-1)
        if row.size == 0:
            return None

        finite_row = np.nan_to_num(row, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
        pred_idx = int(np.argmax(finite_row))
        class_names = list(self._model_class_names or [])

        def _label_for(class_index: int) -> str:
            if 0 <= class_index < len(class_names):
                return str(class_names[class_index])
            return f"pred_{class_index}"

        top_count = max(1, min(int(top_k), finite_row.size))
        top_indices = np.argsort(finite_row)[::-1][:top_count]
        top_predictions = [
            {
                "index": int(class_index),
                "label": _label_for(int(class_index)),
                "confidence": float(row[int(class_index)]),
            }
            for class_index in top_indices
        ]

        return {
            "predicted_index": pred_idx,
            "predicted_label": _label_for(pred_idx),
            "confidence": float(row[pred_idx]),
            "top_predictions": top_predictions,
        }

    def _prediction_tooltips_for_plot(self, *, top_k: int = 3) -> list[str | None]:
        """Return per-point tooltip strings used in prediction mode."""
        tooltips: list[str | None] = [None] * len(self.image_paths)
        for index in range(len(tooltips)):
            summary = self._prediction_summary_for_index(index, top_k=top_k)
            if summary is None:
                continue
            lines = [
                f"Prediction: {summary['predicted_label']} ({self._format_prediction_confidence(summary['confidence'])})"
            ]
            if summary["top_predictions"]:
                lines.append("Top predictions:")
                for item in summary["top_predictions"]:
                    lines.append(
                        f"  {item['label']}: {self._format_prediction_confidence(item['confidence'])}"
                    )
            tooltips[index] = "\n".join(lines)
        return tooltips

    def _prediction_details_html(self, index: int, *, top_k: int = 3) -> str:
        """Return formatted prediction details for preview-side panels."""
        summary = self._prediction_summary_for_index(index, top_k=top_k)
        if summary is None:
            return "<b>Prediction:</b> unavailable"

        class_color_map = self._schema_category_color_map(
            extra_categories=list(self._model_class_names or [])
        )

        top_html = "<br>".join(
            f"&nbsp;&nbsp;{rank}. {self._label_tag_html(item['label'], class_color_map=class_color_map)} "
            f"({self._format_prediction_confidence(item['confidence'])})"
            for rank, item in enumerate(summary["top_predictions"], start=1)
        )
        if top_html:
            top_html = (
                f"<br><b>Top-{len(summary['top_predictions'])}:</b><br>{top_html}"
            )
        return (
            f"<b>Prediction:</b> {self._label_tag_html(summary['predicted_label'], class_color_map=class_color_map)} "
            f"({self._format_prediction_confidence(summary['confidence'])})"
            f"{top_html}"
        )

    def set_explorer_mode(self, mode: str):
        """Set explorer mode to cluster, labeling, or prediction coloring."""
        if mode not in {"explore", "labeling", "review", "predictions"}:
            return

        previous_mode = self.explorer_mode

        if mode == "predictions" and self._model_probs is None:
            self._set_view_mode_combo_value(previous_mode)
            QMessageBox.information(
                self,
                "Predictions Unavailable",
                "Load a checkpoint first. Predictions and model-space UMAP are computed automatically on load.",
            )
            return

        if mode == "review" and not self._review_candidate_indices:
            self._set_view_mode_combo_value(previous_mode)
            QMessageBox.information(
                self,
                "Review Queue Empty",
                "There are no unverified machine labels to review yet.",
            )
            return

        self.explorer_mode = mode
        if mode != "labeling":
            self._labeling_flow_mode = "batch"
            self._labeling_navigation_scope = "pool"
        if previous_mode in {"labeling", "review"} and mode not in {
            "labeling",
            "review",
        }:
            self._clear_explorer_selection_lock(clear_preview=True)

        # Label assignment is only allowed in labeling mode.
        labels_enabled = mode == "labeling"
        outlines_enabled = mode == "predictions"
        for i in range(self.label_buttons_layout.count()):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setEnabled(labels_enabled)

        if hasattr(self, "labeling_options_group"):
            self.labeling_options_group.setVisible(labels_enabled)

        if hasattr(self, "outline_threshold_label"):
            self.outline_threshold_label.setVisible(outlines_enabled)
        if hasattr(self, "outline_threshold_spin"):
            self.outline_threshold_spin.setVisible(outlines_enabled)

        if (
            mode == "review"
            and self.selected_point_index not in self._review_candidate_indices
        ):
            self.selected_point_index = (
                self._review_candidate_indices[0]
                if self._review_candidate_indices
                else self.selected_point_index
            )
            if self.selected_point_index is not None:
                self.request_preview_for_index(
                    self.selected_point_index, source="selection"
                )

        self._set_view_mode_combo_value(mode)

        self.request_update_explorer_plot()
        self.update_knn_panel(self.selected_point_index)
        self.update_context_panel()
        self.status.showMessage(f"Mode: {self._mode_display_name(mode)}")

    def on_view_mode_changed(self, index):
        """Handle mode selection from combo box."""
        mode = self.view_mode_combo.itemData(index)
        self.set_explorer_mode(mode)

    def on_outline_threshold_changed(self, value: float) -> None:
        """Apply uncertainty outline threshold used by Explorer point styling."""
        self._outline_threshold = float(value)
        if hasattr(self, "explorer") and self.explorer is not None:
            self.explorer.set_uncertainty_outline_threshold(self._outline_threshold)
            self.request_update_explorer_plot()
        self.status.showMessage(
            f"Uncertainty outline threshold: {self._outline_threshold:.2f}"
        )

    def on_sample_next_triggered(self):
        """Action for 'Sample Next' button."""
        if self.explorer_mode == "review":
            if not self._review_candidate_indices:
                self.status.showMessage("No machine-review candidates available")
                return
            if self.selected_point_index in self._review_candidate_indices:
                next_index = self._advance_pool_index(self._review_candidate_indices, 1)
            else:
                next_index = self._review_candidate_indices[0]
            if next_index is not None:
                self.selected_point_index = int(next_index)
                self.hover_locked = True
                self.request_preview_for_index(
                    self.selected_point_index, source="selection"
                )
                self.request_update_explorer_selection(self.selected_point_index)
                self.status.showMessage(
                    f"Review candidate {self.selected_point_index + 1}/{len(self._review_candidate_indices)}"
                )
            return
        self.sample_candidates_for_labeling()

    def on_clear_candidates_triggered(self):
        """Clear current candidate set and return to explore mode."""
        self.candidate_indices = []
        self.round_labeled_indices = []
        self._labeling_flow_mode = "batch"

        if self.db_path:
            try:
                from ..core.store.db import ClassKitDB

                db = ClassKitDB(self.db_path)
                db.save_candidate_cache([])  # Save empty list to effectively clear
            except Exception:
                pass

        self.set_explorer_mode("explore")
        self.status.showMessage("Candidates cleared")

    def _ensure_candidate_sampling_ready(self) -> bool:
        """Return True only when the project has the data needed for candidate sampling."""
        if self.umap_coords is None or self.cluster_assignments is None:
            QMessageBox.information(
                self,
                "Need UMAP + Clusters",
                "Compute clustering and UMAP first, then sample candidate sets.",
            )
            return False
        if not self.image_labels or len(self.image_labels) != len(self.image_paths):
            self.image_labels = [None] * len(self.image_paths)
        return True

    @staticmethod
    def _sample_cluster_candidates(assignments, labels, per_cluster: int) -> list[int]:
        """Pick up to `per_cluster` unlabeled samples from each cluster."""
        sampled: list[int] = []
        for cluster_id in sorted(set(assignments)):
            indices = [
                idx
                for idx, value in enumerate(assignments)
                if value == cluster_id and (idx < len(labels) and not labels[idx])
            ]
            sampled.extend(indices[:per_cluster])
        return sampled

    def _merge_labeled_candidates(self, sampled: list[int], labels) -> list[int]:
        """Keep already-labeled items from the current batch visible after resampling."""
        seen = set(sampled)
        for index in self.candidate_indices:
            if index not in seen and index < len(labels) and labels[index]:
                sampled.append(index)
                seen.add(index)
        return sampled

    def _persist_candidate_indices(self) -> None:
        """Store the current candidate set in the project DB when available."""
        if not self.db_path:
            return
        try:
            from ..core.store.db import ClassKitDB

            db = ClassKitDB(self.db_path)
            db.save_candidate_cache(self.candidate_indices)
        except Exception:
            pass

    def _labeled_image_indices(self) -> list[int]:
        """Return dataset indices that already have an assigned label."""
        labels = self.image_labels or []
        return [
            index
            for index in range(len(self.image_paths))
            if index < len(labels) and bool(labels[index])
        ]

    def _unlabeled_image_indices(self) -> list[int]:
        """Return dataset indices that still need a label."""
        labels = self.image_labels or []
        return [
            index
            for index in range(len(self.image_paths))
            if index >= len(labels) or not labels[index]
        ]

    def _diversity_reference_indices(
        self, labeled_indices: list[int], max_refs: int = 256
    ) -> list[int]:
        """Reduce large labeled sets to a representative reference subset."""
        if len(labeled_indices) <= max_refs:
            return list(labeled_indices)
        try:
            embeddings = np.asarray(self.embeddings)
            from ..core.al.density import select_diverse_samples

            local = select_diverse_samples(
                embeddings[labeled_indices], max_refs, seed=0
            )
            return [int(labeled_indices[int(idx)]) for idx in local]
        except Exception:
            step = max(1, len(labeled_indices) // max_refs)
            return [int(idx) for idx in labeled_indices[::step][:max_refs]]

    def _min_distance_to_reference_scores(
        self, candidate_indices: list[int], reference_indices: list[int]
    ) -> np.ndarray | None:
        """Score each candidate by its minimum embedding distance to references."""
        if not candidate_indices or not reference_indices:
            return None
        if self.embeddings is None:
            return None

        embeddings = np.asarray(self.embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.image_paths):
            return None

        candidate_embs = embeddings[candidate_indices].astype(np.float32, copy=False)
        reference_embs = embeddings[reference_indices].astype(np.float32, copy=False)
        scores = np.full(len(candidate_indices), np.inf, dtype=np.float32)

        candidate_chunk = 256
        reference_chunk = 128
        for cand_start in range(0, len(candidate_embs), candidate_chunk):
            cand_stop = cand_start + candidate_chunk
            cand_chunk = candidate_embs[cand_start:cand_stop]
            chunk_scores = np.full(len(cand_chunk), np.inf, dtype=np.float32)
            for ref_start in range(0, len(reference_embs), reference_chunk):
                ref_stop = ref_start + reference_chunk
                ref_chunk = reference_embs[ref_start:ref_stop]
                deltas = cand_chunk[:, None, :] - ref_chunk[None, :, :]
                distances = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
                chunk_scores = np.minimum(chunk_scores, distances.min(axis=1))
            scores[cand_start:cand_stop] = chunk_scores

        return scores

    def _min_distance_and_owner_to_reference_scores(
        self, candidate_indices: list[int], reference_indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return each candidate's min distance and nearest labeled reference index."""
        if not candidate_indices or not reference_indices:
            return None
        if self.embeddings is None:
            return None

        embeddings = np.asarray(self.embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.image_paths):
            return None

        candidate_embs = embeddings[candidate_indices].astype(np.float32, copy=False)
        reference_embs = embeddings[reference_indices].astype(np.float32, copy=False)
        reference_ids = np.asarray(reference_indices, dtype=np.int32)
        scores = np.full(len(candidate_indices), np.inf, dtype=np.float32)
        owners = np.full(len(candidate_indices), -1, dtype=np.int32)

        candidate_chunk = 256
        reference_chunk = 128
        for cand_start in range(0, len(candidate_embs), candidate_chunk):
            cand_stop = cand_start + candidate_chunk
            cand_chunk = candidate_embs[cand_start:cand_stop]
            chunk_scores = np.full(len(cand_chunk), np.inf, dtype=np.float32)
            chunk_owners = np.full(len(cand_chunk), -1, dtype=np.int32)
            for ref_start in range(0, len(reference_embs), reference_chunk):
                ref_stop = ref_start + reference_chunk
                ref_chunk = reference_embs[ref_start:ref_stop]
                deltas = cand_chunk[:, None, :] - ref_chunk[None, :, :]
                distances = np.einsum("ijk,ijk->ij", deltas, deltas, optimize=True)
                local_best = distances.argmin(axis=1)
                local_scores = distances[np.arange(len(cand_chunk)), local_best]
                improved = local_scores < chunk_scores
                if np.any(improved):
                    chunk_scores[improved] = local_scores[improved]
                    chunk_owners[improved] = reference_ids[
                        ref_start + local_best[improved]
                    ]
            scores[cand_start:cand_stop] = chunk_scores
            owners[cand_start:cand_stop] = chunk_owners

        return scores, owners

    def _select_infinite_labeling_candidate(self) -> int | None:
        """Pick the next unlabeled point that most expands dataset coverage."""
        unlabeled_indices = self._unlabeled_image_indices()
        if not unlabeled_indices:
            return None
        if len(unlabeled_indices) == 1:
            return int(unlabeled_indices[0])

        labeled_indices = self._labeled_image_indices()
        assignments = self.cluster_assignments
        candidate_pool = list(unlabeled_indices)
        distance_cache = (
            self._ensure_infinite_labeling_cache() if labeled_indices else None
        )

        if assignments is not None and len(assignments) == len(self.image_paths):
            if labeled_indices:
                cluster_counts = self._infinite_label_cluster_counts
                min_count = min(
                    (
                        int(cluster_counts[int(assignments[index])])
                        if cluster_counts is not None
                        and 0 <= int(assignments[index]) < len(cluster_counts)
                        else 0
                    )
                    for index in unlabeled_indices
                )
                candidate_pool = [
                    int(index)
                    for index in unlabeled_indices
                    if (
                        int(cluster_counts[int(assignments[index])])
                        if cluster_counts is not None
                        and 0 <= int(assignments[index]) < len(cluster_counts)
                        else 0
                    )
                    == min_count
                ]
            else:
                sampled = self._sample_cluster_candidates(
                    list(assignments),
                    self.image_labels or [None] * len(self.image_paths),
                    1,
                )
                if sampled:
                    return int(sampled[0])

        if distance_cache is not None and len(distance_cache) == len(self.image_paths):
            distance_scores = distance_cache[candidate_pool]
            best_local = int(np.argmax(distance_scores))
            return int(candidate_pool[best_local])

        return int(candidate_pool[0]) if candidate_pool else None

    def _activate_infinite_labeling_candidate(self, index: int) -> None:
        """Make one distinct unlabeled point the active singleton candidate set."""
        self._labeling_flow_mode = "infinite"
        self._labeling_navigation_scope = "pool"
        self.candidate_indices = [int(index)]
        self.round_labeled_indices = []
        self._persist_candidate_indices()

    def _start_infinite_labeling_mode(self) -> bool:
        """Enter a rolling singleton labeling flow driven by diversity."""
        next_index = self._select_infinite_labeling_candidate()
        if next_index is None:
            self._labeling_flow_mode = "batch"
            QMessageBox.information(
                self,
                "Infinite Labeling Complete",
                "There are no unlabeled points left to surface.",
            )
            return False

        self._activate_infinite_labeling_candidate(next_index)
        self.set_explorer_mode("labeling")
        self.selected_point_index = int(next_index)
        self._apply_navigation_selection("infinite")
        self.request_update_explorer_plot()
        self.request_update_context_panel()
        self.status.showMessage(
            "Infinite labeling mode active — showing the next most distinct unlabeled point"
        )
        return True

    def _handle_exhausted_labeling_pool(self) -> tuple[int | None, str | None]:
        """Resolve what should happen when the current labeling pool runs out."""
        if self._labeling_flow_mode == "infinite":
            next_index = self._select_infinite_labeling_candidate()
            if next_index is None:
                self._labeling_flow_mode = "batch"
                QMessageBox.information(
                    self,
                    "Infinite Labeling Complete",
                    "There are no unlabeled points left to surface.",
                )
                return None, None
            self._activate_infinite_labeling_candidate(next_index)
            return int(next_index), None
        return None, self._prompt_after_label_set_complete()

    def _apply_sampled_candidates(self, assignments) -> None:
        """Update UI state after sampling a new labeling candidate set."""
        if self.candidate_indices:
            self._labeling_flow_mode = "batch"
            self._labeling_navigation_scope = "pool"
            self.set_explorer_mode("labeling")
            self.selected_point_index = None
            self.hover_locked = False
            self.selection_info.setText(
                "<div style='line-height:1.5;'>"
                "<b>Selected Point:</b> none<br>"
                "<b>Hovered Point:</b> none<br>"
                "<b>Current Label:</b> unlabeled<br>"
                "Hover over candidate points to preview, then click to select one for labeling."
                "</div>"
            )
            self.status.showMessage(
                f"Sampled {len(self.candidate_indices):,} unlabeled candidates across {len(set(assignments))} clusters"
            )
            return
        self.status.showMessage("No unlabeled points left in current clusters")

    def sample_candidates_for_labeling(self):
        """Sample diverse unlabeled points from each cluster for fast labeling."""
        if not self._begin_command():
            return
        try:
            if not self._ensure_candidate_sampling_ready():
                return

            per_cluster = self.sample_spin.value()
            assignments = list(self.cluster_assignments)
            labels = self.image_labels

            sampled = self._sample_cluster_candidates(assignments, labels, per_cluster)
            self.candidate_indices = self._merge_labeled_candidates(sampled, labels)
            self._persist_candidate_indices()
            self._apply_sampled_candidates(assignments)

            self.request_update_explorer_plot()
            self.request_update_context_panel()
        finally:
            self._end_command(0.15)

    def load_preview_for_index(self, index: int, source: str = "hover"):
        """Load preview image and metadata for a UMAP point index."""
        if index is None or index < 0 or index >= len(self.image_paths):
            return

        if source == "hover" and self.last_preview_index == index:
            return

        image_path = self.image_paths[index]

        self.preview_canvas.set_image(str(image_path))

        self.last_preview_index = index
        current_label = (
            self.image_labels[index] if index < len(self.image_labels) else None
        )
        cluster_id = None
        if self.cluster_assignments is not None and index < len(
            self.cluster_assignments
        ):
            cluster_id = self.cluster_assignments[index]
        prediction_html = self._prediction_details_html(index, top_k=3)
        class_color_map = self._schema_category_color_map(
            extra_categories=list(self._model_class_names or [])
        )
        current_label_html = self._label_tag_html(
            current_label if current_label else None,
            class_color_map=class_color_map,
        )

        self.preview_info.setText(
            "<div style='line-height:1.55;'>"
            f"<b>Point:</b> {index}<br>"
            f"<b>Cluster:</b> {cluster_id if cluster_id is not None else 'n/a'}<br>"
            f"<b>Label:</b> {current_label_html}<br>"
            f"{prediction_html}<br>"
            f"<b>Source:</b> {source}<br>"
            f"<span style='color:#9e9e9e; font-size:11px;'>{escape(image_path.name)}</span>"
            "</div>"
        )

        self.selection_info.setText(
            "<div style='line-height:1.5;'>"
            f"<b>Selected Point:</b> {self.selected_point_index if self.selected_point_index is not None else 'none'}<br>"
            f"<b>Hovered Point:</b> {index}<br>"
            f"<b>Current Label:</b> {current_label_html}<br>"
            f"{prediction_html}<br>"
            "Assign using number keys 1-9 or class buttons."
            "</div>"
        )

        knn_anchor = self.selected_point_index
        if knn_anchor is None and source in {"click", "next", "prev", "undo", "label"}:
            knn_anchor = index
        self.update_knn_panel(knn_anchor)

    def clear_preview_display(self) -> None:
        """Clear the preview panel when no point is actively hovered or selected."""
        self._pending_preview_index = None
        self._pending_preview_source = "hover"
        if self._preview_refresh_timer.isActive():
            self._preview_refresh_timer.stop()
        self.last_preview_index = None
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.clear_image()
        self.preview_info.setText(
            "<div style='line-height:1.55; color:#aaaaaa;'>"
            "Hover a point to preview the source image.<br>"
            "Select a point, then label using 1-9 or the left-side controls."
            "</div>"
        )
        self.selection_info.setText(
            "<div style='line-height:1.5;'>"
            "<b>Selected Point:</b> none<br>"
            "<b>Hovered Point:</b> none<br>"
            "<b>Current Label:</b> unlabeled<br>"
            f"{self._selection_idle_hint()}"
            "</div>"
        )
        self._update_review_panel()
        self.update_knn_panel(None)

    def _selection_idle_hint(self) -> str:
        """Return the context-sensitive instruction shown when no point is selected."""
        if self.explorer_mode in {"labeling", "review"}:
            return "Hover candidates to preview; click one to select for labeling."
        return "Hover a point to preview; selection lock is disabled in this mode."

    def _compute_knn_neighbors(self, anchor_index: int, k: int = 8):
        """Compute nearest neighbors from embedding space for a selected point."""
        if (
            self.embeddings is None
            or anchor_index is None
            or anchor_index < 0
            or anchor_index >= len(self.image_paths)
        ):
            return []

        query = self.embeddings[anchor_index]
        deltas = self.embeddings - query
        distances = np.sum(deltas * deltas, axis=1)
        ranking = np.argsort(distances)

        neighbors = []
        for candidate in ranking:
            candidate = int(candidate)
            if candidate == anchor_index:
                continue
            label = (
                self.image_labels[candidate]
                if candidate < len(self.image_labels)
                else None
            )
            neighbors.append(
                {
                    "index": candidate,
                    "distance": float(distances[candidate]),
                    "label": label,
                }
            )
            if len(neighbors) >= k:
                break
        return neighbors

    def update_knn_panel(self, anchor_index: int | None):
        """Render nearest-neighbor context for the current selected point."""
        if (
            anchor_index is None
            or self.explorer_mode != "labeling"
            or self.embeddings is None
        ):
            self._current_knn_neighbors = []
            self.knn_info.setHtml(
                "<span style='color:#9f9f9f;'>Select a point in labeling mode to view nearest neighbors.</span>"
            )
            self.knn_jump_btn.setEnabled(False)
            self.knn_bulk_btn.setEnabled(False)
            return

        neighbors = self._compute_knn_neighbors(anchor_index, k=8)
        self._current_knn_neighbors = neighbors

        if not neighbors:
            self.knn_info.setHtml(
                "<span style='color:#9f9f9f;'>No neighbors available.</span>"
            )
            self.knn_jump_btn.setEnabled(False)
            self.knn_bulk_btn.setEnabled(False)
            return

        lines = [
            f"<b>Anchor:</b> #{anchor_index}<br>",
            "<b>Nearest points:</b><br>",
        ]
        unlabeled_neighbors = 0
        for rank, item in enumerate(neighbors, start=1):
            label = item["label"] if item["label"] else "unlabeled"
            if not item["label"]:
                unlabeled_neighbors += 1
            lines.append(
                f"{rank}. #{item['index']} · <i>{label}</i> · d={item['distance']:.4f}<br>"
            )

        self.knn_info.setHtml("".join(lines))
        self.knn_jump_btn.setEnabled(True)

        anchor_label = (
            self.image_labels[anchor_index]
            if anchor_index < len(self.image_labels)
            else None
        )
        self.knn_bulk_btn.setEnabled(bool(anchor_label) and unlabeled_neighbors > 0)

    def jump_to_nearest_neighbor(self):
        """Jump preview/selection to the nearest neighbor of current anchor."""
        if not self._current_knn_neighbors:
            self.status.showMessage("No neighbors available for jump")
            return

        neighbor_index = int(self._current_knn_neighbors[0]["index"])
        self.selected_point_index = neighbor_index
        self.hover_locked = True
        self.request_preview_for_index(neighbor_index, source="knn-jump")
        self.request_update_explorer_selection(neighbor_index)
        self.status.showMessage(f"Jumped to nearest neighbor #{neighbor_index}")

    def bulk_label_nearest_neighbors(self):
        """Apply anchor label to nearby unlabeled neighbors after confirmation."""
        if not self._begin_command():
            return

        try:
            anchor_index = self.selected_point_index
            if anchor_index is None or anchor_index >= len(self.image_labels):
                self.status.showMessage("Select a point first")
                return

            anchor_label = self.image_labels[anchor_index]
            if not anchor_label:
                self.status.showMessage(
                    "Anchor point must be labeled before bulk assist"
                )
                return

            unlabeled_neighbors = [
                item for item in self._current_knn_neighbors if not item.get("label")
            ]
            if not unlabeled_neighbors:
                self.status.showMessage("No unlabeled neighbors available")
                return

            to_apply = unlabeled_neighbors[:5]
            if not self._ask_yes_no(
                "Confirm Bulk Label Assist",
                f"Apply label '{anchor_label}' to {len(to_apply)} nearest unlabeled neighbors?\n\n"
                "You can undo these assignments with Ctrl+Z.",
            ):
                return

            for item in to_apply:
                idx = int(item["index"])
                prev = self.image_labels[idx] if idx < len(self.image_labels) else None
                self.last_assigned_stack.append(
                    {
                        "index": idx,
                        "previous_label": prev,
                        "new_label": anchor_label,
                    }
                )
                self._set_label_for_index(idx, anchor_label)
                self._remove_history_for_index(idx)
                self.label_history.append({"index": idx, "label": anchor_label})
                self.candidate_indices = [i for i in self.candidate_indices if i != idx]

            self.request_refresh_label_history_strip()
            self.request_update_explorer_plot()
            self.request_update_context_panel()
            self.update_knn_panel(anchor_index)
            self.status.showMessage(
                f"Applied '{anchor_label}' to {len(to_apply)} nearest unlabeled neighbors"
            )
        finally:
            self._end_command(0.15)

    def assign_label_to_selected(self, label: str):
        """Assign label to selected point and persist in DB."""
        if not self._begin_command():
            return
        if self.explorer_mode != "labeling":
            self.status.showMessage(
                "Label assignment is disabled outside Labeling mode"
            )
            self._end_command(0.08)
            return

        next_action = None
        try:
            index = self._selected_index_for_assignment()
            if index is None:
                return

            self._apply_selected_label(index, label)
            self.on_label_assigned(label)
            next_unlabeled = self._next_unlabeled_candidate_index(index)
            if next_unlabeled is None:
                next_unlabeled, next_action = self._handle_exhausted_labeling_pool()
            if next_unlabeled is not None:
                self.selected_point_index = next_unlabeled
                self.request_preview_for_index(
                    self.selected_point_index, source="next-unlabeled"
                )
                self.hover_locked = True
            else:
                self.selected_point_index = None
                self.hover_locked = False
                self.request_update_explorer_selection(None)
                self.request_preview_for_index(index, source="label")

            self.request_update_explorer_plot()
            self.request_update_context_panel()
        finally:
            self._end_command(0.12)

        if next_action == "sample":
            QTimer.singleShot(0, self.sample_candidates_for_labeling)
        elif next_action == "al":
            QTimer.singleShot(0, self._build_al_batch)
        elif next_action == "infinite":
            QTimer.singleShot(0, self._start_infinite_labeling_mode)

    def _selected_index_for_assignment(self) -> int | None:
        """Validate the current selection before applying a label."""
        if self.selected_point_index is None:
            self.status.showMessage("Select a point first before assigning a label")
            return None
        index = self.selected_point_index
        if index < 0 or index >= len(self.image_paths):
            return None
        return index

    def _apply_selected_label(self, index: int, label: str) -> None:
        """Persist one explicit label assignment and update local history/state."""
        previous_label = (
            self.image_labels[index] if index < len(self.image_labels) else None
        )
        self._labeling_navigation_scope = "pool"
        self.last_assigned_stack.append(
            {
                "index": index,
                "previous_label": previous_label,
                "new_label": label,
            }
        )
        self._set_label_for_index(index, label)
        self.round_labeled_indices = [
            i for i in self.round_labeled_indices if i != index
        ]
        self.round_labeled_indices.append(index)
        self._remove_history_for_index(index)
        self.label_history.append({"index": index, "label": label})
        self.request_refresh_label_history_strip()

    def _next_unlabeled_candidate_index(self, current_index: int) -> int | None:
        """Remove the labeled item from the candidate pool and return the next unlabeled one."""
        self.candidate_indices = [
            i for i in self.candidate_indices if i != current_index
        ]
        labels = self.image_labels or []
        for index in self.candidate_indices:
            if index >= len(labels) or not labels[index]:
                return index
        return None

    def _prompt_after_label_set_complete(self) -> str | None:
        """Prompt next step when all points in current sampled set are labeled."""
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Labeling Set Complete")
        message.setText(
            "There are no unlabeled points left in the current labeling set."
        )
        message.setInformativeText(
            "Choose what to do next: sample another labeling set, or enter infinite labeling mode to keep surfacing distinct unlabeled points across clusters and embedding space until the dataset is exhausted."
        )
        sample_btn = message.addButton("Sample Another Set", QMessageBox.AcceptRole)
        infinite_btn = message.addButton(
            "Start Infinite Labeling", QMessageBox.ActionRole
        )
        message.addButton(QMessageBox.Close)
        message.setDefaultButton(sample_btn)
        message.exec()

        clicked = message.clickedButton()
        if clicked == sample_btn:
            return "sample"
        if clicked == infinite_btn:
            return "infinite"
        return None

    def update_explorer_plot(self, force_fit: bool = False):
        """Refresh explorer points with current mode, labels, and candidate emphasis."""
        if not hasattr(self, "explorer"):
            return

        # Choose which UMAP coordinates to display
        if self._show_model_umap and self.umap_model_coords is not None:
            coords = self.umap_model_coords
        elif self._show_model_pca and self.pca_model_coords is not None:
            coords = self.pca_model_coords
        else:
            coords = self.umap_coords
        if coords is None:
            return

        category_order = None
        category_colors = None
        point_tooltips = None

        if self.explorer_mode == "explore":
            color_values = self.cluster_assignments
            candidate_indices = []
        elif self.explorer_mode == "labeling":
            seeded_labels = list(self.image_labels or [])
            seeded_labels.extend(list(self.classes or []))
            seeded_labels.append("unknown")
            color_values = seeded_labels
            category_colors = self._schema_category_color_map()
            candidate_indices = self.candidate_indices
        elif self.explorer_mode == "review":
            review_labels = [
                self._review_status_for_index(i).get("label")
                for i in range(len(self.image_paths))
            ]
            review_labels.extend(list(self.classes or []))
            review_labels.append("unknown")
            color_values = review_labels
            category_colors = self._schema_category_color_map()
            candidate_indices = self._review_candidate_indices
        else:
            color_values = self._prediction_labels_for_plot()
            category_order = self._schema_category_order(
                extra_categories=list(self._model_class_names or [])
            )
            category_colors = self._schema_category_color_map(
                extra_categories=list(self._model_class_names or [])
            )
            point_tooltips = self._prediction_tooltips_for_plot(top_k=3)
            candidate_indices = []

        if not force_fit and self.explorer.update_state(
            labels=color_values,
            confidences=self.image_confidences,
            candidate_indices=candidate_indices,
            round_labeled_indices=self.round_labeled_indices,
            selected_index=self.selected_point_index,
            labeling_mode=(self.explorer_mode in {"labeling", "review"}),
            prediction_mode=(self.explorer_mode == "predictions"),
            category_order=category_order,
            category_colors=category_colors,
            point_tooltips=point_tooltips,
        ):
            return

        self.explorer.set_data(
            coords,
            color_values,
            confidences=self.image_confidences,
            candidate_indices=candidate_indices,
            round_labeled_indices=self.round_labeled_indices,
            selected_index=self.selected_point_index,
            labeling_mode=(self.explorer_mode in {"labeling", "review"}),
            prediction_mode=(self.explorer_mode == "predictions"),
            category_order=category_order,
            category_colors=category_colors,
            point_tooltips=point_tooltips,
            preserve_view=(not force_fit),
        )

    # ================== Data Operations ==================

    def manage_sources(self):
        """Open the Source Manager to view, add, or remove image source folders."""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first.\n\n"
                "Use File → New Project to get started.",
            )
            return

        from .dialogs import SourceManagerDialog

        dlg = SourceManagerDialog(db_path=self.db_path, parent=self)
        if dlg.exec() != SourceManagerDialog.Accepted or not dlg.has_changes:
            return

        folders_to_remove = dlg.folders_to_remove
        folders_to_add = dlg.folders_to_add

        # ── removals ────────────────────────────────────────────────
        if folders_to_remove:
            from ..core.store.db import ClassKitDB

            db = ClassKitDB(self.db_path)
            total_removed = 0
            for folder in folders_to_remove:
                total_removed += db.remove_images_by_folder_exact(folder)
            if total_removed:
                self.status.showMessage(
                    f"Removed {total_removed:,} images from {len(folders_to_remove)} source(s)"
                )

        # ── additions ───────────────────────────────────────────────
        if folders_to_add:
            self._ingest_queue = list(folders_to_add)
            self._ingest_batch_total = len(self._ingest_queue)
            self._run_next_ingest()
        elif folders_to_remove:
            # Removals only — reload and offer to redo pipeline
            self._flush_pending_label_updates(force=True)
            self.load_project_data()
            self.update_context_panel()
            # Caches are now stale (image set changed) — offer auto-redo.
            self._invalidate_image_set_dependent_state()
            QTimer.singleShot(200, self._auto_pipeline_after_source_change)

    # keep legacy name as alias for internal callers
    ingest_images = manage_sources

    def _run_next_ingest(self):
        """Start the next item in the ingest queue."""
        if not self._ingest_queue:
            return
        folder = self._ingest_queue[0]
        request = self._resolve_ingest_request(folder)
        if request is None:
            self._ingest_queue = []
            self.progress_bar.setVisible(False)
            self.status.showMessage("Source import cancelled")
            return

        from ..jobs.task_workers import IngestWorker

        worker = IngestWorker(
            request["source_root"],
            self.db_path,
            project_classes=self.classes,
            import_labels=bool(request.get("import_labels", True)),
        )
        worker.signals.started.connect(
            lambda f=folder: self.status.showMessage(f"[Step 1/5] Ingesting {f.name}…")
        )
        worker.signals.progress.connect(self.on_ingest_progress)
        worker.signals.success.connect(self._on_ingest_batch_item_done)
        worker.signals.error.connect(self.on_ingest_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _on_ingest_batch_item_done(self, result):
        """Called when one folder in the ingest batch finishes."""
        self._ingest_queue.pop(0)
        if self._ingest_queue:
            # More folders to go
            self._run_next_ingest()
        else:
            # All done — flush any in-flight labels before reloading so they are preserved
            self._flush_pending_label_updates(force=True)
            num_images = result.get("num_images", 0)
            total = (
                num_images  # accumulative count not tracked per-folder; just show last
            )
            QMessageBox.information(
                self,
                "Ingestion Complete",
                f"Ingested images from {self._ingest_batch_total} folder(s).\n"
                f"Last batch: {total:,} images added.",
            )
            self.status.showMessage(
                f"Ingestion complete ({self._ingest_batch_total} folder(s))"
            )
            self.load_project_data()
            self.update_context_panel()
            # Caches are now stale — invalidate and offer to redo.
            self._invalidate_image_set_dependent_state()
            QTimer.singleShot(200, self._auto_pipeline_after_source_change)

    # ── auto-pipeline after source changes ─────────────────────────────

    def _auto_pipeline_after_source_change(self):
        """After sources changed: offer to re-run the full embed → cluster → UMAP
        pipeline using existing settings (without showing configuration dialogs)."""
        if not self.image_paths:
            return

        # Retrieve the most recent embedding settings from the DB so we can
        # re-run without prompting for configuration.
        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)

        # Look for last embedding metadata to recover settings
        last_embed_meta = self._get_last_embedding_settings(db)
        last_cluster_meta = self._get_last_cluster_settings(db)

        if last_embed_meta is None:
            # No previous pipeline — fall back to the interactive flow
            self._auto_pipeline_check()
            return

        model_name = last_embed_meta.get("model_name", "dinov2_vitb14")
        device = last_embed_meta.get("device", "cpu")
        batch_size = last_embed_meta.get("batch_size", 32)

        n_clusters = 500
        cluster_method = "minibatch"
        if last_cluster_meta:
            n_clusters = last_cluster_meta.get("n_clusters", 500)
            cluster_method = last_cluster_meta.get("method", "minibatch")

        n_neighbors = self.last_umap_params.get("n_neighbors", 15)
        min_dist = self.last_umap_params.get("min_dist", 0.1)

        summary = (
            f"Sources have changed. Re-run the pipeline with previous settings?\n\n"
            f"  Embedding: {model_name} on {device} (batch {batch_size})\n"
            f"  Clustering: {cluster_method}, k={n_clusters}\n"
            f"  UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}\n\n"
            f"Images: {len(self.image_paths):,}"
        )

        reply = QMessageBox.question(
            self,
            "Re-run Pipeline?",
            summary,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Chain: embed → cluster → UMAP using saved settings (no dialogs)
        self._auto_rerun_embed(
            model_name,
            device,
            batch_size,
            n_clusters,
            cluster_method,
            n_neighbors,
            min_dist,
        )

    def _get_last_embedding_settings(self, db):
        """Return the metadata dict from the most recent embedding run, or None."""
        with __import__("sqlite3").connect(db.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT model_name, device, batch_size, meta_json
                FROM embeddings
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = c.fetchone()
        if not row:
            return None
        model_name, device, batch_size, meta_json = row
        result = {"model_name": model_name, "device": device, "batch_size": batch_size}
        if meta_json:
            import json as _json

            try:
                meta = _json.loads(meta_json)
                if isinstance(meta, dict):
                    result.update(meta)
            except Exception:
                pass
        return result

    def _get_last_cluster_settings(self, db):
        """Return the metadata from the most recent cluster run, or None."""
        with __import__("sqlite3").connect(db.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT n_clusters, method
                FROM cluster_cache
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = c.fetchone()
        if not row:
            return None
        return {"n_clusters": row[0], "method": row[1]}

    def _auto_rerun_embed(
        self,
        model_name,
        device,
        batch_size,
        n_clusters,
        cluster_method,
        n_neighbors,
        min_dist,
    ):
        """Start embeddings computation with given settings, then chain cluster + UMAP."""
        from ..jobs.task_workers import EmbeddingWorker

        self._reset_analysis_view()

        worker = EmbeddingWorker(
            self.image_paths,
            model_name,
            device,
            batch_size,
            db_path=self.db_path,
            force_recompute=True,
        )
        worker.signals.started.connect(
            lambda: self.status.showMessage("Re-computing embeddings…")
        )
        worker.signals.progress.connect(self.on_embedding_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_embed_success(result):
            self.on_embedding_success(result)
            # Chain to clustering
            QTimer.singleShot(
                200,
                lambda: self._auto_rerun_cluster(
                    n_clusters,
                    cluster_method,
                    n_neighbors,
                    min_dist,
                ),
            )

        worker.signals.success.connect(_on_embed_success)
        worker.signals.error.connect(self.on_embedding_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _auto_rerun_cluster(self, n_clusters, method, n_neighbors, min_dist):
        """Run clustering with given settings, then chain UMAP."""
        if self.embeddings is None:
            return

        from ..jobs.task_workers import ClusteringWorker

        self._reset_analysis_view()

        worker = ClusteringWorker(self.embeddings, n_clusters, method)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Re-clustering data…")
        )
        worker.signals.progress.connect(self.on_clustering_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_cluster_success(result):
            self.on_clustering_success(result)
            # Chain to UMAP
            QTimer.singleShot(
                200,
                lambda: self._auto_rerun_umap(n_neighbors, min_dist),
            )

        worker.signals.success.connect(_on_cluster_success)
        worker.signals.error.connect(self.on_clustering_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _auto_rerun_umap(self, n_neighbors, min_dist):
        """Run UMAP with given settings."""
        if self.embeddings is None:
            return

        from ..jobs.task_workers import UMAPWorker

        self._reset_analysis_view()
        self.last_umap_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}
        worker = UMAPWorker(self.embeddings, n_neighbors, min_dist)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Re-computing UMAP…")
        )
        worker.signals.progress.connect(self.on_umap_progress)
        worker.signals.success.connect(self.on_umap_success)
        worker.signals.error.connect(self.on_umap_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    # ── guided-setup helpers ───────────────────────────────────────────────

    def _prompt_adjust_sources_if_empty(self):
        """After new-project creation: automatically open Source Manager if no images yet."""
        if not self.image_paths:
            self.manage_sources()

    def _guide_to_next_step(self):
        """After opening an existing project, guide the user to the next incomplete step."""
        if not self.project_path:
            return

        if not self.image_paths:
            reply = QMessageBox.question(
                self,
                "No Images Found",
                f"Project '{self.project_path.name}' has no images ingested yet.\n\n"
                "Would you like to add image sources now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.ingest_images()
            return

        if self.embeddings is None:
            reply = QMessageBox.question(
                self,
                "Embeddings Not Computed",
                f"{len(self.image_paths):,} images are loaded but embeddings have "
                "not been computed yet.\n\nCompute embeddings now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._compute_embeddings_then(callback=self._auto_cluster_check)
            return

        if self.cluster_assignments is None:
            reply = QMessageBox.question(
                self,
                "Clustering Not Done",
                "Embeddings are ready but clustering has not been run.\n\n"
                "Run clustering now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._cluster_data_then(callback=self._auto_umap_check)
            return

        if self.umap_coords is None:
            reply = QMessageBox.question(
                self,
                "UMAP Not Computed",
                "Clustering is done but UMAP has not been computed yet.\n\n"
                "Compute UMAP now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.compute_umap()

    def _auto_pipeline_check(self):
        """After ingestion: offer to run embed → cluster → UMAP if not yet done."""
        if not self.image_paths:
            return

        # Step 1: embeddings
        if self.embeddings is None:
            reply = QMessageBox.question(
                self,
                "Compute Embeddings?",
                f"{len(self.image_paths):,} images have been ingested.\n\n"
                "Embeddings are required before clustering and UMAP visualisation.\n"
                "Compute them now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._compute_embeddings_then(callback=self._auto_cluster_check)
        else:
            self._auto_cluster_check()

    def _auto_cluster_check(self):
        """After embeddings: offer to run cluster → UMAP."""
        if self.embeddings is None:
            return
        if self.cluster_assignments is None:
            reply = QMessageBox.question(
                self,
                "Cluster Data?",
                "Embeddings are ready.\n\n"
                "Clustering groups images into visual clusters to guide labeling.\n"
                "Run clustering now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._cluster_data_then(callback=self._auto_umap_check)
        else:
            self._auto_umap_check()

    def _auto_umap_check(self):
        """After clustering: offer to compute UMAP."""
        if self.cluster_assignments is None:
            return
        if self.umap_coords is None:
            reply = QMessageBox.question(
                self,
                "Compute UMAP?",
                "Clustering complete.\n\n"
                "Computing a 2-D UMAP projection allows you to visually explore and "
                "label your dataset.  Run UMAP now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.compute_umap()

    def _compute_embeddings_then(self, callback=None):
        """Like compute_embeddings() but calls *callback* on success."""
        if not self.image_paths:
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached = db.get_most_recent_embeddings()
        if cached is not None:
            embeddings, metadata = cached
            if self._ask_yes_no(
                "Use Cached Embeddings",
                "Cached embeddings found. Load them instead of recomputing?\n\n"
                f"Model: {metadata.get('model_name', 'unknown')}\n"
                f"Timestamp: {metadata.get('timestamp', 'unknown')}",
            ):
                self._reset_analysis_view()
                self.on_embedding_success(
                    {
                        "embeddings": embeddings,
                        "cached": True,
                        "metadata": metadata,
                    }
                )
                if callback:
                    QTimer.singleShot(200, callback)
                return

        from .dialogs import EmbeddingDialog

        dialog = EmbeddingDialog(self)
        if not dialog.exec():
            return
        model_name, device, batch_size, force_recompute = dialog.get_settings()
        self._reset_analysis_view()

        from ..jobs.task_workers import EmbeddingWorker

        worker = EmbeddingWorker(
            self.image_paths,
            model_name,
            device,
            batch_size,
            db_path=self.db_path,
            force_recompute=force_recompute,
        )
        worker.signals.started.connect(
            lambda: self.status.showMessage("Computing embeddings…")
        )
        worker.signals.progress.connect(self.on_embedding_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_success(result):
            self.on_embedding_success(result)
            if callback:
                QTimer.singleShot(200, callback)

        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(self.on_embedding_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _cluster_data_then(self, callback=None):
        """Like cluster_data() but calls *callback* on success."""
        if self.embeddings is None:
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached_cluster = db.get_most_recent_cluster_cache(
            self._current_embedding_cache_id
        )
        if cached_cluster is not None and self._ask_yes_no(
            "Use Cached Clusters",
            "Cached cluster assignments found. Load them?\n\n"
            f"Method: {cached_cluster.get('method', 'unknown')}\n"
            f"Timestamp: {cached_cluster.get('timestamp', 'unknown')}",
        ):
            self._reset_analysis_view()
            self.cluster_assignments = cached_cluster["assignments"]
            self._clear_candidate_state(persist=True)
            self.status.showMessage("Loaded cached cluster assignments")
            self.update_explorer_plot()
            self.update_context_panel()
            if callback:
                QTimer.singleShot(200, callback)
            return

        from .dialogs import ClusterDialog

        dialog = ClusterDialog(self)
        if not dialog.exec():
            return
        n_clusters, method = dialog.get_settings()
        self._reset_analysis_view()

        from ..jobs.task_workers import ClusteringWorker

        worker = ClusteringWorker(self.embeddings, n_clusters, method)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Clustering data…")
        )
        worker.signals.progress.connect(self.on_clustering_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_success(result):
            self.on_clustering_success(result)
            if callback:
                QTimer.singleShot(200, callback)

        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(self.on_clustering_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    # ── class editor ─────────────────────────────────────────────────────

    def open_class_editor(self):
        """Open the dedicated class / labeling-scheme editor dialog."""
        from .dialogs import ClassEditorDialog

        scheme_dict = None
        if self.project_path:
            scheme_path = self._project_scheme_path()
            if scheme_path.exists():
                try:
                    import json as _json

                    with open(scheme_path) as _f:
                        scheme_dict = _json.load(_f)
                except Exception:
                    pass

        dlg = ClassEditorDialog(
            classes=self.classes,
            scheme_dict=scheme_dict,
            parent=self,
        )
        if dlg.exec() != ClassEditorDialog.Accepted:
            return

        new_scheme = dlg.get_scheme_dict()
        flat_classes = dlg.flat_classes

        # Persist to project
        if self.project_path:
            import json as _json

            config_path = self._project_config_path()
            config = {}
            if config_path.exists():
                with open(config_path) as _f:
                    config = _json.load(_f)
            config["classes"] = flat_classes
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as _f:
                _json.dump(config, _f, indent=2)

            if new_scheme:
                scheme_path = self._project_scheme_path()
                scheme_path.parent.mkdir(parents=True, exist_ok=True)
                with open(scheme_path, "w") as _f:
                    _json.dump(new_scheme, _f, indent=2)

        self.classes = flat_classes
        self.rebuild_label_buttons()
        self.setup_label_shortcuts()
        self.update_context_panel()
        self.status.showMessage(f"Classes updated ({len(self.classes)})")

    # ── shortcut editor ──────────────────────────────────────────────────

    def open_shortcut_editor(self):
        """Open the keyboard-shortcut editor dialog."""
        from .dialogs import ShortcutEditorDialog

        dlg = ShortcutEditorDialog(current=self._custom_shortcuts, parent=self)
        if dlg.exec() != ShortcutEditorDialog.Accepted:
            return

        self._custom_shortcuts = dlg.get_shortcuts()
        self.setup_label_shortcuts()
        self._refresh_shortcut_help()

        if self.project_path:
            import json as _json

            config_path = self._project_config_path()
            config = {}
            if config_path.exists():
                with open(config_path) as _f:
                    config = _json.load(_f)
            config["custom_shortcuts"] = self._custom_shortcuts
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as _f:
                _json.dump(config, _f, indent=2)

        self.status.showMessage("Keyboard shortcuts updated")

    def open_contrast_settings(self):
        """Open the CLAHE contrast enhancement settings dialog."""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
        )

        class ContrastSettingsDialog(QDialog):
            def __init__(self, clip, grid, parent=None) -> None:
                super().__init__(parent)
                self.setWindowTitle("Contrast Enhancement Settings")
                layout = QVBoxLayout(self)
                layout.setSpacing(16)

                form = QFormLayout()
                self.clip_spin = QDoubleSpinBox()
                self.clip_spin.setRange(0.1, 40.0)
                self.clip_spin.setSingleStep(0.1)
                self.clip_spin.setValue(clip)
                form.addRow("<b>CLAHE Clip Limit:</b>", self.clip_spin)

                grid_row = QHBoxLayout()
                self.grid_x = QSpinBox()
                self.grid_x.setRange(1, 64)
                self.grid_x.setValue(grid[0])
                self.grid_y = QSpinBox()
                self.grid_y.setRange(1, 64)
                self.grid_y.setValue(grid[1])
                grid_row.addWidget(self.grid_x)
                grid_row.addWidget(QLabel("x"))
                grid_row.addWidget(self.grid_y)
                form.addRow("<b>CLAHE Grid Size:</b>", grid_row)

                layout.addLayout(form)

                info = QLabel(
                    "<span style='color: #888;'>"
                    "CLAHE (Contrast Limited Adaptive Histogram Equalization) "
                    "improves visibility in dark or low-contrast images."
                    "</span>"
                )
                info.setWordWrap(True)
                layout.addWidget(info)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
                )
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addWidget(buttons)

        dlg = ContrastSettingsDialog(self.clahe_clip, self.clahe_grid, self)
        if dlg.exec() == QDialog.Accepted:
            self.clahe_clip = dlg.clip_spin.value()
            self.clahe_grid = (dlg.grid_x.value(), dlg.grid_y.value())

            # Update canvas
            if hasattr(self, "preview_canvas"):
                self.preview_canvas.set_clahe_params(self.clahe_clip, self.clahe_grid)

            self._save_preview_enhancement_settings()

            self.status.showMessage("Contrast enhancement settings updated")

            # Refresh current preview if visible
            if self.last_preview_index is not None:
                self.load_preview_for_index(self.last_preview_index, source="enhance")

    def _refresh_shortcut_help(self):
        """Rebuild the shortcut reminder label from current bindings."""
        from .dialogs import ShortcutEditorDialog

        defaults = dict(ShortcutEditorDialog.DEFAULT_SHORTCUTS)
        active = {**defaults, **self._custom_shortcuts}

        label_instr = "1–9 (fallback)"
        if self._stepper is not None:
            label_instr = "stepper keys"
        else:
            # Check if we have scheme shortcuts
            try:
                scheme_path = self._project_scheme_path()
                if scheme_path.exists():
                    with open(scheme_path) as _f:
                        scheme_dict = json.load(_f)
                        factors = scheme_dict.get("factors", [])
                        if factors:
                            f = factors[0]
                            keys = [k for k in f.get("shortcut_keys", []) if k]
                            if keys:
                                label_instr = f"defined keys: {', '.join(keys[:5])}..."
            except Exception:
                pass

        lines = [
            "<div style='line-height:1.5; color:#bcbcbc;'>",
            "<b>Controls:</b><br>",
            f"• <b>{active.get('Explore mode', 'E')}</b> / <b>{active.get('Labeling mode', 'L')}</b> / <b>{active.get('Review mode', 'V')}</b> / <b>{active.get('Predictions mode', 'P')}</b>: set mode<br>",
            f"• <b>{label_instr}</b>: assign class<br>",
            "• <b>0</b>: mark as unknown<br>",
            f"• <b>{active.get('Approve review label', '+')}</b> / <b>{active.get('Reject review label', '-')}</b>: approve or reject selected machine label<br>",
            f"• <b>{active.get('Sample next candidates', 'Space')}</b>: sample candidates<br>",
            f"• <b>{active.get('Previous unlabeled', 'Left')}</b> / <b>{active.get('Next unlabeled', 'Right')}</b>: navigate<br>",
            f"• <b>{active.get('Undo last label (Ctrl+Z)', 'Ctrl+Z')}</b>: undo<br>",
            "</div>",
        ]
        if hasattr(self, "shortcut_help"):
            self.shortcut_help.setText("".join(lines))

    # ── startup overlay ──────────────────────────────────────────────────

    def compute_embeddings(self):
        """Compute embeddings for all images."""
        if not self.project_path:
            QMessageBox.warning(
                self, "No Project", "Please create or open a project first."
            )
            return

        # Check if we have images
        if not self.image_paths:
            QMessageBox.warning(
                self,
                "No Images",
                "Please ingest images first.\\n\\nUse File → Ingest Images.",
            )
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached = db.get_most_recent_embeddings()
        if cached is not None:
            embeddings, metadata = cached
            if self._ask_yes_no(
                "Use Cached Embeddings",
                "Most recent cached embeddings are available. Load them instead of recomputing?\n\n"
                f"Model: {metadata.get('model_name', 'unknown')}\n"
                f"Timestamp: {metadata.get('timestamp', 'unknown')}",
            ):
                self._reset_analysis_view()
                self.on_embedding_success(
                    {
                        "embeddings": embeddings,
                        "cached": True,
                        "metadata": metadata,
                    }
                )
                return

        from .dialogs import EmbeddingDialog

        dialog = EmbeddingDialog(self)
        if dialog.exec():
            model_name, device, batch_size, force_recompute = dialog.get_settings()
            self._reset_analysis_view()

            from ..jobs.task_workers import EmbeddingWorker

            worker = EmbeddingWorker(
                self.image_paths,
                model_name,
                device,
                batch_size,
                db_path=self.db_path,
                force_recompute=force_recompute,
            )
            worker.signals.started.connect(
                lambda: self.status.showMessage("Computing embeddings...")
            )
            worker.signals.progress.connect(self.on_embedding_progress)
            worker.signals.success.connect(self.on_embedding_success)
            worker.signals.error.connect(self.on_embedding_error)
            worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self._threadpool_start(worker)

    def cluster_data(self):
        """Cluster embeddings."""
        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Please compute embeddings first.\\n\\nUse Compute → Compute Embeddings.",
            )
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached_cluster = db.get_most_recent_cluster_cache(
            self._current_embedding_cache_id
        )
        if cached_cluster is not None and self._ask_yes_no(
            "Use Cached Clusters",
            "Most recent cluster assignments are available. Load them instead of reclustering?\n\n"
            f"Method: {cached_cluster.get('method', 'unknown')}\n"
            f"Timestamp: {cached_cluster.get('timestamp', 'unknown')}",
        ):
            self._reset_analysis_view()
            self.cluster_assignments = cached_cluster["assignments"]
            self._clear_candidate_state(persist=True)
            self.status.showMessage("Loaded cached cluster assignments")
            self.update_explorer_plot()
            self.update_context_panel()
            return

        from .dialogs import ClusterDialog

        dialog = ClusterDialog(self)
        if dialog.exec():
            n_clusters, method = dialog.get_settings()
            self._reset_analysis_view()

            from ..jobs.task_workers import ClusteringWorker

            worker = ClusteringWorker(self.embeddings, n_clusters, method)
            worker.signals.started.connect(
                lambda: self.status.showMessage("Clustering data...")
            )
            worker.signals.progress.connect(self.on_clustering_progress)
            worker.signals.success.connect(self.on_clustering_success)
            worker.signals.error.connect(self.on_clustering_error)
            worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self._threadpool_start(worker)

    def compute_umap(self):
        """Compute UMAP projection."""
        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Please compute embeddings first.\\n\\nUse Compute → Compute Embeddings.",
            )
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached_umap = db.get_most_recent_umap_cache(
            embedding_cache_id=self._current_embedding_cache_id,
        )
        if cached_umap is not None and self._ask_yes_no(
            "Use Cached UMAP",
            "Most recent UMAP projection is available. Load it instead of recomputing?\n\n"
            f"Timestamp: {cached_umap.get('timestamp', 'unknown')}\n"
            f"n_neighbors: {cached_umap.get('n_neighbors', 'unknown')}\n"
            f"min_dist: {cached_umap.get('min_dist', 'unknown')}",
        ):
            self._reset_analysis_view()
            self.umap_coords = cached_umap["coords"]
            self.last_umap_params = {
                "n_neighbors": cached_umap.get("n_neighbors", 15),
                "min_dist": cached_umap.get("min_dist", 0.1),
            }
            self.status.showMessage("Loaded cached UMAP projection")
            self.update_explorer_plot(force_fit=True)
            self.update_context_panel()
            return

        from ..jobs.task_workers import UMAPWorker

        self._reset_analysis_view()

        worker = UMAPWorker(self.embeddings)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Computing UMAP...")
        )
        worker.signals.progress.connect(self.on_umap_progress)
        worker.signals.success.connect(self.on_umap_success)
        worker.signals.error.connect(self.on_umap_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _resolve_training_scheme(self):
        """Load labeling scheme from project directory, or return None."""
        if not self.project_path:
            return None
        try:
            from ..config.schemas import LabelingScheme

            scheme_path = self._project_scheme_path()
            if scheme_path.exists():
                with open(scheme_path) as _f:
                    return LabelingScheme.from_dict(json.load(_f))
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to load scheme.json; proceeding without scheme",
                exc_info=True,
            )
        return None

    def _make_training_spec(self, settings, role, mode, is_yolo, dataset_dir):
        """Build a TrainingRunSpec from dialog settings."""
        import dataclasses

        from ...training.contracts import (
            AugmentationProfile,
            CustomCNNParams,
            TinyHeadTailParams,
            TrainingHyperParams,
            TrainingRunSpec,
        )

        aug = AugmentationProfile(
            enabled=True,
            flipud=settings.get("flipud", 0.0),
            fliplr=settings.get("fliplr", 0.0),
            hue=settings.get("hue", 0.0),
            saturation=settings.get("saturation", 0.0),
            brightness=settings.get("brightness", 0.0),
            contrast=settings.get("contrast", 0.0),
            monochrome=bool(settings.get("monochrome", False)),
            args={
                key: value
                for key, value in {
                    "flipud": settings.get("flipud", 0.0),
                    "fliplr": settings.get("fliplr", 0.0),
                    "hsv_h": settings.get("hue", 0.0),
                    "hsv_s": settings.get("saturation", 0.0),
                    "hsv_v": settings.get("brightness", 0.0),
                }.items()
                if float(value) > 0.0
            },
            label_expansion=settings.get("label_expansion") or {},
        )

        spec = TrainingRunSpec(
            role=role,
            source_datasets=[],
            derived_dataset_dir=str(dataset_dir),
            base_model=(
                settings.get("initial_model_path") or settings.get("base_model", "")
                if is_yolo
                else ""
            ),
            hyperparams=TrainingHyperParams(
                epochs=settings.get("epochs", 50),
                batch=settings.get("batch", 32),
                lr0=settings.get("lr", 0.001),
                patience=settings.get("patience", 10),
            ),
            tiny_params=TinyHeadTailParams(
                epochs=settings.get("epochs", 50),
                batch=settings.get("batch", 32),
                lr=settings.get("lr", 0.001),
                patience=settings.get("patience", 10),
                tiny_preset=settings.get("tiny_preset", "medium"),
                hidden_layers=settings.get("tiny_layers", 1),
                hidden_dim=settings.get("tiny_dim", 96),
                dropout=settings.get("tiny_dropout", 0.1),
                input_width=settings.get("tiny_width", 128),
                input_height=settings.get("tiny_height", 64),
                class_rebalance_mode=settings.get("tiny_rebalance_mode", "none"),
                class_rebalance_power=settings.get("tiny_rebalance_power", 1.0),
                label_smoothing=settings.get("tiny_label_smoothing", 0.0),
            ),
            device=settings.get("device", "cpu"),
            training_space="original",
            resume_from=(
                settings.get("initial_model_path", "")
                if mode in ("flat_custom", "multihead_custom")
                else ""
            ),
            augmentation_profile=aug,
        )
        if mode in ("flat_custom", "multihead_custom"):
            spec = dataclasses.replace(
                spec,
                custom_params=CustomCNNParams(
                    backbone=settings.get("custom_backbone", "tinyclassifier"),
                    fine_tune_method=settings.get(
                        "custom_fine_tune_method", "head_only"
                    ),
                    trainable_layers=settings.get("custom_trainable_layers", 0),
                    backbone_lr_scale=settings.get("custom_backbone_lr_scale", 0.1),
                    layerwise_lr_decay=settings.get("custom_layerwise_lr_decay", 0.75),
                    gradual_unfreeze_interval=settings.get(
                        "custom_gradual_unfreeze_interval", 5
                    ),
                    input_size=settings.get("custom_input_size", 224),
                    epochs=settings.get("epochs", 50),
                    batch=settings.get("batch", 32),
                    lr=settings.get("lr", 1e-3),
                    patience=settings.get("patience", 10),
                    weight_decay=1e-2,
                    tiny_preset=settings.get("tiny_preset", "medium"),
                    hidden_layers=settings.get("tiny_layers", 1),
                    hidden_dim=settings.get("tiny_dim", 96),
                    dropout=settings.get("tiny_dropout", 0.1),
                    input_width=settings.get("tiny_width", 128),
                    input_height=settings.get("tiny_height", 64),
                    label_smoothing=settings.get("tiny_label_smoothing", 0.0),
                    class_rebalance_mode=settings.get("tiny_rebalance_mode", "none"),
                    class_rebalance_power=settings.get("tiny_rebalance_power", 1.0),
                ),
            )
        return spec

    def _save_training_results_to_db(
        self,
        results,
        mode,
        labels_str,
        *,
        evaluation_paths=None,
        evaluation_split_name: str = "",
    ):
        """Persist training results to the project model cache DB."""
        if not self.db_path or not results:
            return
        try:
            from ..core.store.db import ClassKitDB as _CKDb

            _db = _CKDb(self.db_path)
            artifact_paths = [
                r.get("artifact_path", "")
                for r in results
                if r.get("artifact_path") and Path(r["artifact_path"]).exists()
            ]
            if not artifact_paths:
                return
            all_classes = sorted(set(labels_str))
            acc_values = []
            for r in results:
                value = r.get("best_val_acc")
                if value is None:
                    continue
                try:
                    acc_values.append(float(value))
                except Exception:
                    continue
            best_acc = max(acc_values) if acc_values else None
            _db.save_model_cache(
                mode=mode,
                artifact_paths=artifact_paths,
                class_names=all_classes,
                best_val_acc=best_acc,
                num_classes=len(all_classes),
                meta={
                    "training_settings": dict(self._last_training_settings or {}),
                    "evaluation": {
                        "split_name": str(evaluation_split_name or ""),
                        "paths": [
                            self._resolved_path_key(path)
                            for path in (evaluation_paths or [])
                            if path
                        ],
                    },
                },
            )
        except Exception:
            pass  # non-fatal

    def _run_post_training_inference(
        self, results, is_yolo, multi_head, labels_str, dialog, on_done
    ):
        """Dispatch inference after training completes."""
        if not results:
            return
        artifact = results[0].get("artifact_path", "")
        if not artifact or not Path(artifact).exists():
            return

        if is_yolo:
            if multi_head:
                all_artifacts = [
                    Path(r["artifact_path"])
                    for r in results
                    if r.get("artifact_path") and Path(r["artifact_path"]).exists()
                ]
                self._active_model_mode = "yolo_multihead"
                dialog.append_log(
                    f"Running multi-head YOLO inference ({len(all_artifacts)} models)..."
                )
                self._run_multihead_yolo_inference(
                    all_artifacts,
                    on_success=on_done,
                    force_monochrome=bool(
                        (self._last_training_settings or {}).get("monochrome", False)
                    ),
                )
            else:
                self._yolo_model_path = Path(artifact)
                self._active_model_mode = "yolo"
                dialog.append_log(f"Running YOLO inference: {Path(artifact).name}...")
                self._run_yolo_inference(
                    Path(artifact),
                    on_success=on_done,
                    force_monochrome=bool(
                        (self._last_training_settings or {}).get("monochrome", False)
                    ),
                )
        else:
            # Custom CNN or Tiny CNN .pth -- dispatch based on arch field
            import torch as _torch

            if multi_head:
                all_artifacts = [
                    Path(r["artifact_path"])
                    for r in results
                    if r.get("artifact_path") and Path(r["artifact_path"]).exists()
                ]
                self._active_model_mode = "custom_cnn_multihead"
                dialog.append_log(
                    f"Running multi-head Custom CNN inference ({len(all_artifacts)} models)..."
                )
                self._run_multihead_custom_inference(
                    all_artifacts,
                    on_success=on_done,
                )
            else:
                _ckpt = _torch.load(
                    str(artifact), map_location="cpu", weights_only=False
                )
                _arch = (
                    _ckpt.get("arch", "tinyclassifier")
                    if isinstance(_ckpt, dict)
                    else "tinyclassifier"
                )
                _class_names = _ckpt.get("class_names") or sorted(set(labels_str))
                dialog.append_log(
                    (
                        f"Running Custom CNN inference ({_arch}): {Path(artifact).name}..."
                        if _arch != "tinyclassifier"
                        else f"Running tiny CNN inference: {Path(artifact).name}..."
                    )
                )
                self._active_model_mode = (
                    "tiny" if _arch == "tinyclassifier" else "custom_cnn"
                )
                if _arch != "tinyclassifier":
                    _sz = _ckpt.get("input_size", (224, 224))
                    _sz = _sz[0] if isinstance(_sz, (list, tuple)) else int(_sz)
                    self._run_torchvision_inference(
                        Path(artifact),
                        class_names=_class_names,
                        input_size=_sz,
                        on_success=on_done,
                        force_monochrome=(
                            bool(_ckpt.get("monochrome", False))
                            if isinstance(_ckpt, dict)
                            else False
                        ),
                    )
                else:
                    self._run_tiny_inference(
                        Path(artifact),
                        class_names=_class_names,
                        on_success=on_done,
                    )

    def _publish_training_results(self, dialog, scheme, scheme_name):
        """Publish trained model artifacts to the models directory."""
        results = getattr(dialog, "_train_results", None) or []
        settings = dialog.get_settings()
        mode = settings.get("mode") or "flat_custom"
        is_yolo = "yolo" in mode
        multi_head = mode.startswith("multihead")
        role_map = {
            "flat_tiny": "classify_flat_tiny",
            "flat_yolo": "classify_flat_yolo",
            "multihead_tiny": "classify_multihead_tiny",
            "multihead_yolo": "classify_multihead_yolo",
            "flat_custom": "classify_flat_custom",
            "multihead_custom": "classify_multihead_custom",
        }
        from ...training.contracts import TrainingRole
        from ...training.model_publish import (
            classifier_metadata_for_artifact,
            publish_trained_model,
            write_classifier_multihead_manifest,
        )

        fallback_input_size: tuple[int, int] | None = None
        if "yolo" in mode:
            fallback_input_size = (640, 640)
        elif settings.get("custom_backbone") != "tinyclassifier" and "custom" in mode:
            size = int(settings.get("custom_input_size", 224) or 224)
            fallback_input_size = (size, size)
        fallback_monochrome = bool(settings.get("monochrome", False))

        role_val = role_map.get(mode, "classify_flat_custom")
        role = TrainingRole(role_val)
        published_paths: list[Path] = []
        published_factor_entries: list[dict[str, object]] = []
        bundle_input_size: tuple[int, int] | None = None
        bundle_monochrome = fallback_monochrome
        for fi, result in enumerate(results):
            artifact = result.get("artifact_path", "")
            if not artifact:
                continue
            try:
                classifier_v2_meta = classifier_metadata_for_artifact(
                    artifact,
                    fallback_input_size=fallback_input_size,
                    fallback_monochrome=fallback_monochrome,
                )
                _published_key, published_path = publish_trained_model(
                    role=role,
                    artifact_path=artifact,
                    size="tiny" if "tiny" in mode else "n",
                    species=(
                        self.project_path.name if self.project_path else "species"
                    ),
                    model_info=mode,
                    trained_from_run_id="",
                    dataset_fingerprint="",
                    base_model=settings.get("base_model", "") if is_yolo else "",
                    scheme_name=scheme_name,
                    factor_index=fi if multi_head else None,
                    factor_name=(
                        scheme.factors[fi].name if (multi_head and scheme) else None
                    ),
                    classifier_v2_meta=classifier_v2_meta,
                )
                published_paths.append(Path(published_path))
                input_size_list = classifier_v2_meta.get("input_size") or [224, 224]
                if bundle_input_size is None:
                    bundle_input_size = (
                        int(input_size_list[0]),
                        int(input_size_list[1]),
                    )
                bundle_monochrome = bool(
                    classifier_v2_meta.get("monochrome", bundle_monochrome)
                )
                factor_name = (
                    scheme.factors[fi].name
                    if (multi_head and scheme and fi < len(scheme.factors))
                    else str(
                        (
                            classifier_v2_meta.get("factor_names")
                            or [f"factor_{fi + 1}"]
                        )[0]
                    )
                )
                published_factor_entries.append(
                    {
                        "factor": factor_name,
                        "path": Path(published_path),
                        "class_names": list(
                            (classifier_v2_meta.get("class_names_per_factor") or [[]])[
                                0
                            ]
                        ),
                    }
                )
                from pathlib import Path as _Path

                dialog.append_log(f"Published: {_Path(artifact).name}")

                # Copy single-head YOLO model to project models/ as the "official" latest
                if is_yolo and not multi_head and fi == 0 and self.project_path:
                    try:
                        import shutil

                        model_dir = classkit_model_dir(self.project_path)
                        dest = model_dir / "yolo_classifier_latest.pt"
                        shutil.copy2(artifact, dest)
                        self._yolo_model_path = dest
                        dialog.append_log(f"Saved to project models: {dest.name}")
                    except Exception as copy_exc:
                        dialog.append_log(f"Model copy warning: {copy_exc}")

            except Exception as exc:
                dialog.append_log(f"Publish error: {exc}")

        if multi_head and len(published_paths) > 1:
            try:
                from hydra_suite.classkit.model_bundle import (
                    write_model_bundle_manifest,
                )

                manifest_path = published_paths[0].with_suffix(".bundle.json")
                write_model_bundle_manifest(
                    manifest_path,
                    mode=str(mode),
                    artifact_paths=published_paths,
                    class_names=list(self.classes or []),
                )
                dialog.append_log(f"Published bundle manifest: {manifest_path.name}")

                trackerkit_manifest = published_paths[0].with_suffix(".multihead.json")
                write_classifier_multihead_manifest(
                    trackerkit_manifest,
                    factor_entries=published_factor_entries,
                    input_size=bundle_input_size or fallback_input_size or (224, 224),
                    monochrome=bundle_monochrome,
                )
                dialog.append_log(
                    f"Published TrackerKit manifest: {trackerkit_manifest.name}"
                )
            except Exception as exc:
                dialog.append_log(f"Publish manifest warning: {exc}")

    def _validate_training_pairs(self):
        """Return labeled training pairs when training preconditions are satisfied."""
        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Compute embeddings before training.",
            )
            return None

        labeled_pairs = [
            (p, l) for p, l in zip(self.image_paths, self.image_labels) if l
        ]
        if len(labeled_pairs) < 4:
            QMessageBox.warning(
                self,
                "Not Enough Labels",
                "Need at least 4 labeled images.",
            )
            return None
        return labeled_pairs

    @staticmethod
    def _training_role_for_mode(mode):
        """Resolve the training role enum for the selected training mode."""
        from ...training.contracts import TrainingRole

        role_map = {
            "flat_tiny": TrainingRole.CLASSIFY_FLAT_TINY,
            "flat_yolo": TrainingRole.CLASSIFY_FLAT_YOLO,
            "multihead_tiny": TrainingRole.CLASSIFY_MULTIHEAD_TINY,
            "multihead_yolo": TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
            "flat_custom": TrainingRole.CLASSIFY_FLAT_CUSTOM,
            "multihead_custom": TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
        }
        return role_map.get(mode, TrainingRole.CLASSIFY_FLAT_CUSTOM)

    @staticmethod
    def _prepare_training_labels(labeled_pairs):
        """Convert labeled string labels into image paths and integer class ids."""
        from pathlib import Path

        images = [Path(path) for path, _ in labeled_pairs]
        labels_str = [label for _, label in labeled_pairs]
        unique = sorted(set(labels_str))
        label_map_int = {label: i for i, label in enumerate(unique)}
        int_labels = [label_map_int[label] for label in labels_str]
        class_names_int = {i: label for label, i in label_map_int.items()}
        return images, labels_str, int_labels, class_names_int

    def _build_training_context(self, dialog, labeled_pairs, settings=None):
        """Build the immutable context used across export, train, and inference."""
        from pathlib import Path

        from ..core.export.splits import (
            build_label_expansion_records,
            build_label_expansion_split_key,
            build_training_dataset_splits,
        )

        settings = dict(settings or dialog.get_settings())
        self._last_training_settings = dict(settings)
        self._save_last_training_settings()
        mode = settings.get("mode") or "flat_custom"
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        project_path = Path(self.project_path) if self.project_path else Path.cwd()
        run_dir = project_path / ".classkit_runs" / f"{mode}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        images, labels_str, int_labels, class_names_int = self._prepare_training_labels(
            labeled_pairs
        )
        group_keys = self._training_group_keys_for_paths(images)
        label_expansion = settings.get("label_expansion") or {}
        planned_records = build_label_expansion_records(
            labels_str,
            label_expansion=label_expansion,
            groups=(group_keys or None),
            known_labels=list(class_names_int.values()),
        )
        planned_labels = [str(record["label"]) for record in planned_records]
        planned_groups = (
            [record.get("group") for record in planned_records] if group_keys else None
        )
        planned_splits, used_group_fallback = build_training_dataset_splits(
            planned_labels,
            strategy=settings.get("split_strategy", "stratified"),
            val_fraction=float(settings.get("val_fraction", 0.2)),
            test_fraction=float(settings.get("test_fraction", 0.0)),
            groups=planned_groups,
        )
        source_splits: list[str] = []
        source_split_by_path: dict[str, str] = {}
        expanded_split_by_key: dict[str, str] = {}
        for record, split in zip(planned_records, planned_splits):
            source_path = images[int(record["source_index"])]
            if bool(record.get("is_expanded")):
                expanded_key = build_label_expansion_split_key(
                    source_path,
                    str(record.get("axis") or ""),
                    str(record["label"]),
                )
                expanded_split_by_key[expanded_key] = str(split)
                continue
            source_splits.append(str(split))
            source_split_by_path[self._resolved_path_key(source_path)] = str(split)

        evaluation_split_name = ""
        if "test" in source_splits:
            evaluation_split_name = "test"
        elif "val" in source_splits:
            evaluation_split_name = "val"
        evaluation_paths = [
            self._resolved_path_key(path)
            for path, split in zip(images, source_splits)
            if split == evaluation_split_name
        ]
        return {
            "settings": settings,
            "mode": mode,
            "is_yolo": "yolo" in mode,
            "multi_head": mode.startswith("multihead"),
            "role": self._training_role_for_mode(mode),
            "run_dir": run_dir,
            "images": images,
            "labels_str": labels_str,
            "int_labels": int_labels,
            "class_names_int": class_names_int,
            "group_keys": group_keys,
            "source_splits": source_splits,
            "source_split_by_path": source_split_by_path,
            "expanded_split_by_key": expanded_split_by_key,
            "used_group_fallback": bool(used_group_fallback),
            "evaluation_split_name": evaluation_split_name,
            "evaluation_paths": evaluation_paths,
        }

    def _validate_training_start_model(self, settings: dict) -> bool:
        """Validate any optional warm-start checkpoint before export/training begins."""
        mode = str(settings.get("mode") or "").strip()
        path_text = str(settings.get("initial_model_path") or "").strip()
        if not path_text:
            return True

        model_path = Path(path_text).expanduser()
        if not model_path.exists():
            QMessageBox.warning(
                self,
                "Missing Starting Model",
                f"The selected starting model was not found:\n{model_path}",
            )
            return False

        expected_suffix = ".pt" if "yolo" in mode else ".pth"
        if model_path.suffix.lower() != expected_suffix:
            QMessageBox.warning(
                self,
                "Unsupported Starting Model",
                "The selected starting model does not match the current training mode.\n\n"
                f"Expected a {expected_suffix} artifact for mode '{mode}', got '{model_path.suffix or 'no extension'}'.",
            )
            return False
        return True

    def _build_training_specs(self, context, scheme):
        """Build one or more training specs for the chosen training mode."""
        settings = context["settings"]
        role = context["role"]
        mode = context["mode"]
        is_yolo = context["is_yolo"]
        run_dir = context["run_dir"]

        if context["multi_head"] and scheme is not None:
            return [
                self._make_training_spec(
                    settings,
                    role,
                    mode,
                    is_yolo,
                    run_dir / f"export_f{fi}",
                )
                for fi in range(len(scheme.factors))
            ]
        return [
            self._make_training_spec(
                settings,
                role,
                mode,
                is_yolo,
                run_dir / "export",
            )
        ]

    def _on_training_progress(self, dialog, pct: int, msg: str) -> None:
        """Update the training dialog progress UI from worker callbacks."""
        if pct >= 0:
            dialog.progress_bar.setValue(pct)
        if msg:
            dialog.append_log(msg)

    def _on_training_error(self, dialog, err: str) -> None:
        """Restore dialog controls after a training/export failure."""
        dialog.append_log(f"ERROR: {err}")
        dialog.start_btn.setEnabled(True)
        dialog.cancel_btn.setEnabled(False)

    def _after_training_inference(self, dialog, _result=None) -> None:
        """Refresh metrics and model-space plots after post-training inference."""
        self._evaluate_model_on_labeled()
        self.tabs.setCurrentWidget(self.metrics_page)
        dialog.progress_bar.setValue(100)
        dialog.append_log("Auto-computing model-space UMAP...")
        QTimer.singleShot(
            100,
            lambda: self._replot_umap_model_space(
                auto_switch=True,
                suppress_errors=True,
                log_callback=dialog.append_log,
            ),
        )
        dialog.append_log("Training Complete. Metrics tab updated.")

    def _on_training_success(self, dialog, context, results: list) -> None:
        """Handle successful training completion before post-training inference."""
        self._set_heldout_validation_summary(
            self._validation_summary_from_results(results)
        )
        self._set_active_evaluation_selection(
            context.get("evaluation_paths"),
            context.get("evaluation_split_name", ""),
        )
        dialog._train_results = results
        dialog.publish_btn.setEnabled(True)
        dialog.append_log("Training finished. Refreshing predictions and metrics...")
        dialog.start_btn.setEnabled(True)
        dialog.cancel_btn.setEnabled(False)

        self._save_training_results_to_db(
            results,
            context["mode"],
            context["labels_str"],
            evaluation_paths=context.get("evaluation_paths"),
            evaluation_split_name=context.get("evaluation_split_name", ""),
        )
        self._run_post_training_inference(
            results,
            context["is_yolo"],
            context["multi_head"],
            context["labels_str"],
            dialog,
            lambda result: self._after_training_inference(dialog, result),
        )

    def _on_training_export_success(self, dialog, context, scheme, _result) -> None:
        """Start the actual training worker after dataset export finishes."""
        try:
            from ..jobs.task_workers import ClassKitTrainingWorker

            dialog.append_log("Dataset export finished. Building training specs...")
            specs = self._build_training_specs(context, scheme)
            dialog.append_log(f"Starting training worker ({len(specs)} spec(s))...")
            train_worker = ClassKitTrainingWorker(
                role=context["role"],
                specs=specs,
                run_dir=str(context["run_dir"]),
                multi_head=context["multi_head"],
            )
            dialog._worker = train_worker
            train_worker.signals.progress.connect(
                lambda pct, msg: self._on_training_progress(dialog, pct, msg)
            )
            train_worker.signals.success.connect(
                lambda results: self._on_training_success(dialog, context, results)
            )
            train_worker.signals.error.connect(
                lambda err: self._on_training_error(dialog, err)
            )
            self._threadpool_start(train_worker)
        except Exception as exc:
            self._on_training_error(dialog, f"Failed to start training: {exc}")

    @staticmethod
    def _decode_factor_labels(scheme, labels_str, factor_index: int) -> list[str]:
        """Decode one factor's labels from the composite label strings."""
        return [scheme.decode_label(label)[factor_index] for label in labels_str]

    def _create_multihead_export_worker(self, context, scheme):
        """Create a background worker that exports one dataset per scheme factor."""
        from pathlib import Path as _Path

        from ..jobs.task_workers import ExportWorker

        outer_self = self
        exp_label_expansion = context["settings"].get("label_expansion") or {}

        class MultiHeadExportWorker(ExportWorker):
            def __init__(self, *args, **kwargs) -> None:
                self.scheme = kwargs.pop("scheme")
                self.labels_str = kwargs.pop("labels_str")
                super().__init__(*args, **kwargs)

            @Slot()
            def run(self):
                try:
                    self.signals.started.emit()
                    for fi, _factor in enumerate(self.scheme.factors):
                        self.signals.progress.emit(0, f"Exporting factor {fi}...")
                        factor_labels = outer_self._decode_factor_labels(
                            self.scheme, self.labels_str, fi
                        )
                        unique = sorted(set(factor_labels))
                        label_map = {label: i for i, label in enumerate(unique)}
                        factor_int = [label_map[label] for label in factor_labels]
                        factor_names = {i: label for label, i in label_map.items()}

                        factor_dir = _Path(self.output_path) / f"export_f{fi}"
                        sub_worker = ExportWorker(
                            image_paths=self.image_paths,
                            labels=factor_int,
                            output_path=factor_dir,
                            format="ultralytics",
                            class_names=factor_names,
                            split_strategy=self.split_strategy,
                            val_fraction=self.val_fraction,
                            test_fraction=self.test_fraction,
                            label_expansion=exp_label_expansion,
                            preset_splits_by_path=context["source_split_by_path"],
                            preset_expanded_splits_by_key=context.get(
                                "expanded_split_by_key"
                            ),
                        )
                        sub_worker.run()

                    self.signals.success.emit({})
                except Exception as exc:
                    self.signals.error.emit(str(exc))
                finally:
                    self.signals.finished.emit()

        return MultiHeadExportWorker(
            image_paths=context["images"],
            labels=[0] * len(context["images"]),
            output_path=context["run_dir"],
            format="ultralytics",
            split_strategy=context["settings"].get("split_strategy", "stratified"),
            val_fraction=context["settings"].get("val_fraction", 0.2),
            test_fraction=context["settings"].get("test_fraction", 0.0),
            force_monochrome=bool(context["settings"].get("monochrome", False)),
            preset_splits_by_path=context["source_split_by_path"],
            preset_expanded_splits_by_key=context.get("expanded_split_by_key"),
            scheme=scheme,
            labels_str=context["labels_str"],
        )

    def _create_training_export_worker(self, context, scheme):
        """Create the export worker used before training begins."""
        from ..jobs.task_workers import ExportWorker

        if context["multi_head"] and scheme is not None:
            return self._create_multihead_export_worker(context, scheme)
        return ExportWorker(
            image_paths=context["images"],
            labels=context["int_labels"],
            output_path=context["run_dir"] / "export",
            format="ultralytics",
            class_names=context["class_names_int"],
            split_strategy=context["settings"].get("split_strategy", "stratified"),
            val_fraction=context["settings"].get("val_fraction", 0.2),
            test_fraction=context["settings"].get("test_fraction", 0.0),
            force_monochrome=bool(context["settings"].get("monochrome", False)),
            label_expansion=context["settings"].get("label_expansion") or {},
            preset_splits_by_path=context["source_split_by_path"],
            preset_expanded_splits_by_key=context.get("expanded_split_by_key"),
        )

    def _start_training_from_dialog(self, dialog, labeled_pairs, scheme) -> None:
        """Start dataset export for the training dialog's current settings."""
        settings = dialog.get_settings()
        if not self._validate_training_start_model(settings):
            return

        context = self._build_training_context(dialog, labeled_pairs, settings=settings)
        worker = self._create_training_export_worker(context, scheme)
        dialog._worker = worker
        worker.signals.progress.connect(
            lambda pct, msg: self._on_training_progress(dialog, pct, msg)
        )
        worker.signals.success.connect(
            lambda result: self._on_training_export_success(
                dialog, context, scheme, result
            )
        )
        worker.signals.error.connect(lambda err: self._on_training_error(dialog, err))

        dialog.start_btn.setEnabled(False)
        dialog.cancel_btn.setEnabled(True)

        # Log a full settings snapshot for debugging
        _s = settings
        _mode = _s.get("mode", "?")
        _backbone = _s.get("backbone", _s.get("base_model", "?"))
        _device = _s.get("device", "?")
        _epochs = _s.get("epochs", "?")
        _lr = _s.get("lr", "?")
        _batch = _s.get("batch_size", _s.get("batch", "?"))
        _patience = _s.get("patience", "?")
        _split = _s.get("split_strategy", "?")
        _val_frac = _s.get("val_fraction", "?")
        _test_frac = _s.get("test_fraction", "?")
        dialog.append_log(
            f"Training config: mode={_mode}, backbone={_backbone}, "
            f"device={_device}, epochs={_epochs}, lr={_lr}, batch={_batch}, "
            f"patience={_patience}"
        )
        dialog.append_log(
            f"Split: strategy={_split}, val_fraction={_val_frac}, "
            f"test_fraction={_test_frac}"
        )
        if _s.get("initial_model_path"):
            dialog.append_log(f"Warm-start from: {_s['initial_model_path']}")

        dialog.append_log("Starting dataset export...")
        if context.get("used_group_fallback"):
            dialog.append_log(
                "Grouped source holdout was too coarse for the requested split; using item-level train/val/test fallback for this training export."
            )
        dialog.append_log(dialog.current_data_summary_text())
        self._threadpool_start(worker)

    def train_classifier(self):
        """Open ClassKitTrainingDialog, export dataset, run training, offer publish."""
        self._flush_pending_label_updates(force=True)

        labeled_pairs = self._validate_training_pairs()
        if not labeled_pairs:
            return

        scheme = self._resolve_training_scheme()

        from .dialogs import ClassKitTrainingDialog

        project_class_choices = sorted(
            {str(lbl).strip() for _, lbl in labeled_pairs if str(lbl).strip()}
        )

        dialog = ClassKitTrainingDialog(
            scheme=scheme,
            n_labeled=len(labeled_pairs),
            class_choices=project_class_choices,
            labeled_label_names=[label for _, label in labeled_pairs],
            initial_settings=self._get_recent_project_training_settings(),
            recent_model_paths=self._list_recent_trainable_model_paths(),
            average_image_size=self._estimate_average_image_dimensions(),
            image_paths=[path for path, _ in labeled_pairs],
            group_keys=self._training_group_keys_for_paths(
                [path for path, _ in labeled_pairs]
            ),
            parent=self,
        )

        scheme_name = scheme.name if scheme else "classkit"

        dialog.start_btn.clicked.connect(
            lambda: self._start_training_from_dialog(dialog, labeled_pairs, scheme)
        )
        dialog.publish_btn.clicked.connect(
            lambda: self._publish_training_results(dialog, scheme, scheme_name)
        )
        dialog.exec()

    def export_dataset(self):
        """Export labeled dataset."""
        self._flush_pending_label_updates(force=True)

        if not self.project_path or not self.image_paths:
            QMessageBox.warning(
                self,
                "No Project Data",
                "Open a project with ingested images before exporting.",
            )
            return

        from .dialogs import ExportDialog

        default_output = str(classkit_export_dir(self.project_path))
        dialog = ExportDialog(default_output=default_output, parent=self)
        if not dialog.exec():
            return

        settings = dialog.get_settings()

        export_class_order = list(self.classes)
        for raw_label in self.image_labels:
            label_name = str(raw_label).strip() if raw_label is not None else ""
            if not label_name or label_name in export_class_order:
                continue
            export_class_order.append(label_name)

        label_to_index = {name: idx for idx, name in enumerate(export_class_order)}
        class_names = {idx: name for idx, name in enumerate(export_class_order)}
        include_unlabeled = settings.get("include_unlabeled", False)
        unlabeled_index = len(export_class_order)
        if include_unlabeled:
            class_names[unlabeled_index] = "unlabeled"

        export_paths = []
        export_labels = []
        for path, label_name in zip(self.image_paths, self.image_labels):
            if label_name in label_to_index:
                export_paths.append(path)
                export_labels.append(label_to_index[label_name])
            elif include_unlabeled:
                export_paths.append(path)
                export_labels.append(unlabeled_index)

        if not export_paths:
            QMessageBox.warning(
                self,
                "No Exportable Data",
                "No labeled images available to export with the selected options.",
            )
            return

        from ..jobs.task_workers import ExportWorker

        worker = ExportWorker(
            image_paths=export_paths,
            labels=export_labels,
            output_path=Path(settings["output_dir"]),
            format=settings["format"],
            class_names=class_names,
            val_fraction=float(settings.get("val_fraction", 0.2)),
            test_fraction=float(settings.get("test_fraction", 0.0)),
            copy_files=bool(settings.get("copy_files", True)),
        )

        worker.signals.started.connect(
            lambda: self.status.showMessage("Starting dataset export...")
        )
        worker.signals.progress.connect(self.on_export_progress)
        worker.signals.success.connect(self.on_export_success)
        worker.signals.error.connect(self.on_export_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def load_classifier_checkpoint(self):
        """Load a trained classifier checkpoint (embedding head or YOLO model)."""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Open a ClassKit project first.",
            )
            return

        model_dir = classkit_model_dir(self.project_path)
        latest_checkpoint = model_dir / "classifier_latest.pt"
        yolo_checkpoint = model_dir / "yolo_classifier_latest.pt"
        selected_path = None

        if yolo_checkpoint.exists() and self._ask_yes_no(
            "Load YOLO Classifier",
            f"Load latest saved YOLO classifier?\n\n{yolo_checkpoint}",
        ):
            selected_path = yolo_checkpoint
        elif latest_checkpoint.exists() and self._ask_yes_no(
            "Load Latest Checkpoint",
            f"Load the latest saved classifier checkpoint?\n\n{latest_checkpoint}",
        ):
            selected_path = latest_checkpoint
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Classifier Checkpoint",
                str(model_dir if model_dir.exists() else self.project_path),
                "PyTorch Checkpoint (*.pt *.pth)",
            )
            if not file_path:
                return
            selected_path = Path(file_path)

        self._load_checkpoint_from_path(selected_path)

    def _load_embedding_head_checkpoint(
        self,
        path: Path,
        ckpt,
        *,
        on_success=None,
        show_message_box: bool = True,
    ) -> None:
        """Load a classic embedding-head checkpoint and recompute predictions."""
        self._set_heldout_validation_summary(None)
        from ..core.train.trainer import EmbeddingHeadTrainer

        input_dim = (
            int(self.embeddings.shape[1]) if self.embeddings is not None else 768
        )
        trainer = EmbeddingHeadTrainer(
            model_type=ckpt.get("model_type", "linear"),
            input_dim=input_dim,
            num_classes=max(2, len(self.classes)),
            device=(self._last_training_settings or {}).get("device", "cpu"),
        )
        trainer.load(path)
        self._trained_classifier = trainer
        self.status.showMessage(f"Loaded embedding head: {path.name}")

        if self.embeddings is not None:
            probs = trainer.predict_proba(self.embeddings, calibrated=True)
            self._model_probs = probs
            self._model_class_names = list(self.classes)
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(probs.max(axis=1).astype(float))
            self.update_explorer_plot()
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self._evaluate_model_on_labeled()
            QTimer.singleShot(100, self._replot_umap_model_space)
            if on_success is not None:
                on_success()

        if show_message_box:
            QMessageBox.information(
                self, "Checkpoint Loaded", f"Loaded embedding head: {path.name}"
            )

    def _cached_model_class_names(self, path: Path):
        """Look up class names stored alongside a cached model artifact in the DB."""
        if not self.db_path:
            return None
        try:
            from ..core.store.db import ClassKitDB as _CKDb

            for entry in _CKDb(self.db_path).list_model_caches():
                if str(path) in entry.get("artifact_paths", []):
                    return entry.get("class_names")
        except Exception:
            return None
        return None

    def _cached_model_training_settings(self, path: Path):
        """Look up training settings stored alongside a cached model artifact."""
        if not self.db_path:
            return None
        try:
            from ..core.store.db import ClassKitDB as _CKDb

            for entry in _CKDb(self.db_path).list_model_caches():
                if str(path) not in entry.get("artifact_paths", []):
                    continue
                meta = entry.get("meta")
                training_settings = (
                    meta.get("training_settings") if isinstance(meta, dict) else None
                )
                if isinstance(training_settings, dict):
                    return training_settings
        except Exception:
            return None
        return None

    def _resolve_yolo_force_monochrome(self, path: Path) -> bool:
        """Resolve monochrome inference for YOLO models from cache or project state."""
        training_settings = self._cached_model_training_settings(path)
        if isinstance(training_settings, dict) and "monochrome" in training_settings:
            return bool(training_settings.get("monochrome", False))
        return bool((self._last_training_settings or {}).get("monochrome", False))

    def _resolve_classifier_compute_runtime(self, path: Path) -> str:
        """Resolve classifier inference runtime from cached training settings when available."""
        training_settings = self._cached_model_training_settings(path)
        if isinstance(training_settings, dict):
            runtime = training_settings.get("compute_runtime")
            if runtime:
                return str(runtime)
        return str((self._last_training_settings or {}).get("compute_runtime", "cpu"))

    def _load_custom_cnn_checkpoint(
        self,
        path: Path,
        ckpt,
        *,
        on_success=None,
        show_message_box: bool = True,
    ) -> None:
        """Load a torchvision custom CNN checkpoint and run inference."""
        self._set_active_evaluation_from_meta(self._cached_model_evaluation_info(path))
        self._set_heldout_validation_summary(
            self._validation_summary_from_value(
                ckpt.get("best_val_acc") if isinstance(ckpt, dict) else None,
                prefix="Checkpoint held-out validation accuracy",
            )
        )
        ckpt_names = ckpt.get("class_names")
        input_size = ckpt.get("input_size", (224, 224))
        size = (
            input_size[0] if isinstance(input_size, (list, tuple)) else int(input_size)
        )
        arch = (
            ckpt.get("arch", "tinyclassifier")
            if isinstance(ckpt, dict)
            else "tinyclassifier"
        )
        resolved = ckpt_names or list(self.classes)
        self._active_model_mode = "custom_cnn"
        self.status.showMessage(f"Loading Custom CNN ({arch}): {path.name}...")

        def _after_load(_result):
            self._evaluate_model_on_labeled()
            QTimer.singleShot(100, self._replot_umap_model_space)
            if show_message_box:
                QMessageBox.information(
                    self,
                    "Custom CNN Loaded",
                    f"Loaded: {path.name}\n"
                    f"Inference on {len(self.image_paths):,} images complete.\n"
                    "Metrics tab updated. Model UMAP computing...",
                )
            if on_success is not None:
                on_success()

        self._run_torchvision_inference(
            path,
            class_names=resolved,
            input_size=size,
            on_success=_after_load,
            force_monochrome=(
                bool(ckpt.get("monochrome", False)) if isinstance(ckpt, dict) else False
            ),
        )

    def _load_tiny_cnn_checkpoint(
        self,
        path: Path,
        ckpt,
        *,
        on_success=None,
        show_message_box: bool = True,
    ) -> None:
        """Load a tiny CNN checkpoint and run inference."""
        self._set_active_evaluation_from_meta(self._cached_model_evaluation_info(path))
        self._set_heldout_validation_summary(
            self._validation_summary_from_value(
                ckpt.get("best_val_acc") if isinstance(ckpt, dict) else None,
                prefix="Checkpoint held-out validation accuracy",
            )
        )
        ckpt_names = ckpt.get("class_names")
        resolved = (
            ckpt_names or self._cached_model_class_names(path) or list(self.classes)
        )
        self._active_model_mode = "tiny"
        self.status.showMessage(f"Loading tiny CNN: {path.name}...")

        def _after_load(_result):
            self._evaluate_model_on_labeled()
            QTimer.singleShot(100, self._replot_umap_model_space)
            if show_message_box:
                QMessageBox.information(
                    self,
                    "Tiny CNN Loaded",
                    f"Loaded: {path.name}\n"
                    f"Inference on {len(self.image_paths):,} images complete.\n"
                    "Metrics tab updated. Model UMAP computing...",
                )
            if on_success is not None:
                on_success()

        self._run_tiny_inference(
            path,
            class_names=resolved,
            on_success=_after_load,
            force_monochrome=(
                bool(ckpt.get("monochrome", False)) if isinstance(ckpt, dict) else False
            ),
        )

    def _load_yolo_checkpoint(
        self,
        path: Path,
        *,
        on_success=None,
        show_message_box: bool = True,
    ) -> None:
        """Load a YOLO classifier checkpoint and run inference."""
        self._set_active_evaluation_from_meta(self._cached_model_evaluation_info(path))
        self._set_heldout_validation_summary(None)
        self._yolo_model_path = path
        self.status.showMessage(f"Loading YOLO model: {path.name}...")

        def _after_load(_result):
            self._evaluate_model_on_labeled()
            QTimer.singleShot(100, self._replot_umap_model_space)
            if show_message_box:
                QMessageBox.information(
                    self,
                    "YOLO Model Loaded",
                    f"Loaded: {path.name}\n"
                    f"Inference on {len(self.image_paths):,} images complete.\n"
                    "Metrics tab updated. Model UMAP is being computed automatically.",
                )
            if on_success is not None:
                on_success()

        self._run_yolo_inference(
            path,
            on_success=_after_load,
            force_monochrome=self._resolve_yolo_force_monochrome(path),
        )

    def _load_checkpoint_from_path(
        self,
        path: Path,
        *,
        on_success=None,
        show_message_box: bool = True,
    ):
        """Load a checkpoint, auto-detecting YOLO vs embedding-head vs tiny CNN format."""
        try:
            cache_entry = self._cached_model_cache_entry(path)
            cache_paths = (cache_entry or {}).get("artifact_paths") or []
            if (
                str((cache_entry or {}).get("mode", "")).startswith("multihead")
                and len(cache_paths) > 1
            ):

                def _after_cache_load():
                    if show_message_box:
                        QMessageBox.information(
                            self,
                            "Checkpoint Loaded",
                            f"Loaded multi-head model set with {len(cache_paths)} artifacts.\n"
                            f"Inference on {len(self.image_paths):,} images complete.",
                        )
                    if on_success is not None:
                        on_success()

                self._load_model_from_cache_entry(
                    cache_entry, on_success=_after_cache_load
                )
                return

            from ..model_bundle import discover_multihead_model_bundle

            external_bundle = discover_multihead_model_bundle(path)
            if external_bundle is not None:

                def _after_external_load():
                    if show_message_box:
                        QMessageBox.information(
                            self,
                            "Checkpoint Loaded",
                            f"Loaded external {external_bundle.get('mode', 'multi-head')} bundle with "
                            f"{len(external_bundle.get('artifact_paths') or [])} artifacts.\n"
                            f"Inference on {len(self.image_paths):,} images complete.",
                        )
                    if on_success is not None:
                        on_success()

                self._load_model_from_cache_entry(
                    external_bundle,
                    on_success=_after_external_load,
                )
                return

            import torch

            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            is_tiny_cnn = path.suffix.lower() == ".pth" or (
                isinstance(ckpt, dict) and "model_state_dict" in ckpt
            )

            if isinstance(ckpt, dict) and "model_state" in ckpt:
                self._load_embedding_head_checkpoint(
                    path,
                    ckpt,
                    on_success=on_success,
                    show_message_box=show_message_box,
                )
            elif is_tiny_cnn:
                arch = (
                    ckpt.get("arch", "tinyclassifier")
                    if isinstance(ckpt, dict)
                    else "tinyclassifier"
                )
                if arch != "tinyclassifier":
                    self._load_custom_cnn_checkpoint(
                        path,
                        ckpt,
                        on_success=on_success,
                        show_message_box=show_message_box,
                    )
                else:
                    self._load_tiny_cnn_checkpoint(
                        path,
                        ckpt,
                        on_success=on_success,
                        show_message_box=show_message_box,
                    )
            else:
                self._load_yolo_checkpoint(
                    path,
                    on_success=on_success,
                    show_message_box=show_message_box,
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Checkpoint Load Failed",
                f"Failed to load checkpoint:\n\n{exc}",
            )

    def on_export_progress(self, percentage, message):
        """Update progress for export workflow."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(message)

    def on_export_success(self, result):
        """Handle successful export operation."""
        output_path = result.get("output_path")
        num_exported = int(result.get("num_exported", 0))
        num_classes = int(result.get("num_classes", 0))
        export_format = result.get("format", "unknown")

        self.status.showMessage(
            f"Export complete: {num_exported:,} images ({num_classes} classes)"
        )
        QMessageBox.information(
            self,
            "Export Complete",
            f"Export format: {export_format}\n"
            f"Images exported: {num_exported:,}\n"
            f"Classes: {num_classes}\n\n"
            f"Output:\n{output_path}",
        )

    def on_export_error(self, error_msg):
        """Handle export failure."""
        QMessageBox.critical(
            self,
            "Export Failed",
            f"Dataset export failed:\n\n{error_msg}",
        )
        self.status.showMessage("Export failed")

    # ================== UI Callbacks ==================

    def on_explorer_point_clicked(self, index):
        """Handle point click in explorer."""
        if self.explorer_mode not in {"labeling", "review"}:
            self.status.showMessage("Selection is disabled outside Labeling mode")
            return
        self.selected_point_index = index
        if self.explorer_mode == "labeling":
            self._labeling_navigation_scope = "database"
        self.hover_locked = True
        self.request_preview_for_index(index, source="click")
        self.request_update_explorer_selection(index)
        self.status.showMessage(
            f"Selected point {index} for {'review' if self.explorer_mode == 'review' else 'labeling'}"
        )

    def on_explorer_point_hovered(self, index):
        """Handle point hover in explorer."""
        if (
            self.explorer_mode == "labeling"
            and self.hover_locked
            and self.selected_point_index is not None
            and index != self.selected_point_index
        ):
            return
        if self._has_active_labeling_batch():
            self.request_preview_for_index(index, source="hover")
            return
        if (
            self.hover_locked
            and self.selected_point_index is not None
            and index != self.selected_point_index
        ):
            return
        self.request_preview_for_index(index, source="hover")

    def on_explorer_empty_hover(self) -> None:
        """Clear the preview whenever the cursor is not over an explorer point."""
        if (
            (
                self.explorer_mode == "labeling"
                and self.hover_locked
                and self.selected_point_index is not None
            )
            or self._has_active_labeling_batch()
            or self.explorer_mode == "review"
        ) and self.selected_point_index is not None:
            self.request_preview_for_index(
                self.selected_point_index, source="selection"
            )
            return
        self.clear_preview_display()

    def on_label_assigned(self, label):
        """Handle label assignment."""
        idx = (
            self.selected_point_index
            if self.selected_point_index is not None
            else "n/a"
        )
        self.status.showMessage(f"Assigned label '{label}' to point {idx}")

    def _advance_pool_index(self, pool, step: int) -> int | None:
        """Advance within the current navigation pool with wraparound."""
        if not pool:
            return None
        try:
            current_pos = pool.index(self.selected_point_index)
            return pool[(current_pos + step) % len(pool)]
        except ValueError:
            return pool[0]

    def _advance_unlabeled_pool(self, pool, step: int):
        """Advance to the next unlabeled candidate within the active pool."""
        labels = self.image_labels or []
        unlabeled = [i for i in pool if i >= len(labels) or not labels[i]]
        if not unlabeled:
            next_index, next_action = self._handle_exhausted_labeling_pool()
            if next_index is not None:
                return int(next_index), None
            self.selected_point_index = None
            self.hover_locked = False
            self.request_update_explorer_selection(None)
            return None, next_action

        unlabeled_set = set(unlabeled)
        try:
            current_pos = pool.index(self.selected_point_index)
        except ValueError:
            current_pos = -1 if step > 0 else 0

        for offset in range(1, len(pool) + 1):
            idx = pool[(current_pos + step * offset) % len(pool)]
            if idx in unlabeled_set:
                return idx, None
        return None, self._prompt_after_label_set_complete()

    def _fallback_navigation_index(self, step: int) -> int | None:
        """Advance linearly when no special candidate pool is active."""
        if not self.image_paths:
            return None
        current = (
            self.selected_point_index if self.selected_point_index is not None else 0
        )
        if step > 0:
            return min(len(self.image_paths) - 1, current + 1)
        return max(0, current - 1)

    def _apply_navigation_selection(self, source: str) -> None:
        """Refresh preview and explorer selection after moving to a new point."""
        if self.selected_point_index is None:
            return
        self.hover_locked = True
        self.request_preview_for_index(self.selected_point_index, source=source)
        self.request_update_explorer_selection(self.selected_point_index)

    def _navigate_selected_image(self, step: int, source: str):
        """Shared next/previous navigation logic for both labeling and explore modes."""
        if self.selected_point_index is None:
            self.status.showMessage(
                "No selected point; arrow keys are inactive until you click a point"
            )
            return None

        next_action = None
        pool = self._get_navigation_pool()
        if pool:
            if self.explorer_mode == "labeling":
                self.selected_point_index, next_action = self._advance_unlabeled_pool(
                    pool, step
                )
            else:
                self.selected_point_index = self._advance_pool_index(pool, step)
        else:
            self.selected_point_index = self._fallback_navigation_index(step)

        self._apply_navigation_selection(source)
        return next_action

    def on_next_image(self):
        """Navigate to next candidate or point."""
        if not self._begin_command():
            return
        try:
            next_action = self._navigate_selected_image(1, "next")
        finally:
            self._end_command(0.08)

        if next_action == "sample":
            QTimer.singleShot(0, self.sample_candidates_for_labeling)
        elif next_action == "al":
            QTimer.singleShot(0, self._build_al_batch)
        elif next_action == "infinite":
            QTimer.singleShot(0, self._start_infinite_labeling_mode)

    def on_prev_image(self):
        """Navigate to previous candidate or point."""
        if not self._begin_command():
            return
        try:
            next_action = self._navigate_selected_image(-1, "prev")
        finally:
            self._end_command(0.08)

        if next_action == "sample":
            QTimer.singleShot(0, self.sample_candidates_for_labeling)
        elif next_action == "al":
            QTimer.singleShot(0, self._build_al_batch)
        elif next_action == "infinite":
            QTimer.singleShot(0, self._start_infinite_labeling_mode)

    def refresh_view(self):
        """Refresh current view."""
        self.update_explorer_plot()
        self.update_context_panel()
        self.status.showMessage("Refreshed view")

    def on_enhance_toggled(self, checked: bool):
        """Handle CLAHE enhancement toggle."""
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.use_clahe = checked
            self._save_preview_enhancement_settings()
            # Refresh status
            status = "enabled" if checked else "disabled"
            self.status.showMessage(f"Contrast enhancement {status}", 2000)

            # Sync menu action
            if hasattr(self, "act_enhance") and self.act_enhance.isChecked() != checked:
                self.act_enhance.blockSignals(True)
                self.act_enhance.setChecked(checked)
                self.act_enhance.blockSignals(False)

            # Sync checkbox
            if hasattr(self, "cb_enhance") and self.cb_enhance.isChecked() != checked:
                self.cb_enhance.blockSignals(True)
                self.cb_enhance.setChecked(checked)
                self.cb_enhance.blockSignals(False)

            # Re-display current preview with the new setting
            if self.last_preview_index is not None:
                self.load_preview_for_index(self.last_preview_index, source="enhance")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ClassKit",
            "<h2>ClassKit</h2>"
            + "<p>Active Learning Dataset Builder</p>"
            + "<p>Version 1.0</p>"
            + "<p>Sister framework to PoseKit for image classification "
            + "with smart active learning workflows.</p>",
        )

    # ================== Worker Callbacks ==================

    def on_ingest_progress(self, percentage, message):
        """Update progress for ingestion."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 1/5] Ingesting: {message}")

    def on_ingest_error(self, error_msg):
        """Handle ingestion error."""
        QMessageBox.critical(
            self, "Ingestion Error", f"Failed to ingest images:\n\n{error_msg}"
        )
        self.status.showMessage("Ingestion failed")

    def on_embedding_progress(self, percentage, message):
        """Update progress for embeddings."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 2/5] Embedding: {message}")

    def on_embedding_success(self, result):
        """Handle successful embedding computation."""
        self._reset_analysis_view()
        self.embeddings = result["embeddings"]
        metadata = result.get("metadata") or {}
        self._current_embedding_cache_id = metadata.get("id")
        self._invalidate_embedding_downstream_state(persist_candidates=True)
        cached = result.get("cached", False)

        if cached:
            self.status.showMessage(
                f"Loaded {self.embeddings.shape[0]:,} cached embeddings"
            )
        else:
            self.status.showMessage(f"Computed {self.embeddings.shape[0]:,} embeddings")

        self.update_context_panel()
        self._auto_start_umap_if_ready()

    def on_embedding_error(self, error_msg):
        """Handle embedding error."""
        QMessageBox.critical(
            self, "Embedding Error", f"Failed to compute embeddings:\n\n{error_msg}"
        )
        self.status.showMessage("Embedding failed")

    def on_clustering_progress(self, percentage, message):
        """Update progress for clustering."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 3/5] Clustering: {message}")

    def on_clustering_success(self, result):
        """Handle successful clustering."""
        self._reset_analysis_view()
        self.cluster_assignments = result["assignments"]
        self._clear_candidate_state(persist=True)
        n_clusters = len(set(self.cluster_assignments))

        if self.db_path:
            try:
                from ..core.store.db import ClassKitDB

                db = ClassKitDB(self.db_path)
                self._current_cluster_cache_id = db.save_cluster_cache(
                    self.cluster_assignments,
                    result.get("centers"),
                    n_clusters,
                    result.get("method", "unknown"),
                    meta={"embedding_cache_id": self._current_embedding_cache_id},
                )
            except Exception:
                pass

        self._invalidate_infinite_labeling_cache()

        self.status.showMessage(f"Clustered into {n_clusters} groups")
        self.update_explorer_plot()
        self.update_context_panel()
        self._auto_start_umap_if_ready()

    def on_clustering_error(self, error_msg):
        """Handle clustering error."""
        QMessageBox.critical(
            self, "Clustering Error", f"Failed to cluster:\n\n{error_msg}"
        )
        self.status.showMessage("Clustering failed")

    def on_umap_progress(self, percentage, message):
        """Update progress for UMAP."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 4/5] UMAP: {message}")

    def on_umap_success(self, result):
        """Handle successful UMAP computation."""
        self._reset_analysis_view()
        self.umap_coords = result["coords"]

        if self.db_path:
            try:
                from ..core.store.db import ClassKitDB

                db = ClassKitDB(self.db_path)
                db.save_umap_cache(
                    self.umap_coords,
                    self.last_umap_params.get("n_neighbors", 15),
                    self.last_umap_params.get("min_dist", 0.1),
                    meta={"embedding_cache_id": self._current_embedding_cache_id},
                )
            except Exception:
                pass

        self.status.showMessage("UMAP projection complete")
        self.update_explorer_plot(force_fit=True)
        self.update_context_panel()

    def on_umap_error(self, error_msg):
        """Handle UMAP error."""
        QMessageBox.critical(
            self, "UMAP Error", f"Failed to compute UMAP:\n\n{error_msg}"
        )
        self.status.showMessage("UMAP failed")
        self.update_context_panel()

    def _auto_start_umap_if_ready(self):
        """Automatically trigger UMAP if embeddings and clustering are complete."""
        if (
            self.embeddings is not None
            and self.cluster_assignments is not None
            and self.umap_coords is None
        ):
            self.status.showMessage("Auto-triggering UMAP projection...")
            self.compute_umap()

    def closeEvent(self, event) -> None:
        """Ensure pending label updates are flushed before close."""
        self._flush_pending_label_updates(force=True)
        super().closeEvent(event)

    # ================== Model Inference & Evaluation ==================

    def _persist_prediction_cache(self, probs, class_names: list, mode: str) -> None:
        """Save inference probs + class names to the project DB prediction cache."""
        if not self.db_path or probs is None:
            return
        try:
            from ..core.store.db import ClassKitDB

            ClassKitDB(self.db_path).save_prediction_cache(
                probs=probs,
                class_names=class_names or [],
                active_model_mode=mode,
                meta={
                    "evaluation": {
                        "split_name": self._active_evaluation_split_name,
                        "paths": sorted(self._active_evaluation_paths or []),
                    }
                },
            )
        except Exception:
            pass  # non-fatal — just means cache won't be available next session

    def _run_yolo_inference(
        self,
        model_path: Path,
        on_success=None,
        force_monochrome: bool = False,
    ):
        """Run YOLO inference on all images in background and update confidences."""
        if not self.image_paths:
            return
        compute_runtime = (self._last_training_settings or {}).get(
            "compute_runtime", "cpu"
        )

        from ..jobs.task_workers import YoloInferenceWorker

        worker = YoloInferenceWorker(
            model_path,
            self.image_paths,
            compute_runtime=compute_runtime,
            batch_size=64,
            force_monochrome=force_monochrome,
        )

        def _inference_success(result):
            self._model_probs = result["probs"]
            self._model_class_names = result["class_names"]
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Inference done: {len(self.image_paths):,} images, "
                f"{len(self._model_class_names)} classes"
            )
            self._persist_prediction_cache(
                self._model_probs, self._model_class_names, "yolo"
            )
            if on_success:
                on_success(result)

        worker.signals.success.connect(_inference_success)
        worker.signals.error.connect(
            lambda e: self.status.showMessage(f"Inference error: {e}")
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[Inference] {m}") if m else None
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)

    def _run_tiny_inference(
        self,
        model_path: Path,
        class_names: list,
        on_success=None,
        force_monochrome: bool = False,
    ):
        """Run TinyCNN inference on all images in background and update confidences."""
        if not self.image_paths:
            return
        compute_runtime = self._resolve_classifier_compute_runtime(model_path)

        from ..jobs.task_workers import TinyCNNInferenceWorker

        worker = TinyCNNInferenceWorker(
            model_path,
            self.image_paths,
            class_names,
            compute_runtime=compute_runtime,
            batch_size=64,
            force_monochrome=force_monochrome,
        )

        def _tiny_success(result):
            self._model_probs = result["probs"]
            self._model_class_names = result["class_names"]
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Tiny CNN done: {len(self.image_paths):,} images, "
                f"{len(self._model_class_names)} classes"
            )
            self._persist_prediction_cache(
                self._model_probs, self._model_class_names, "tiny"
            )
            if on_success:
                on_success(result)

        worker.signals.success.connect(_tiny_success)
        worker.signals.error.connect(
            lambda e: self.status.showMessage(f"Tiny CNN error: {e}")
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[Tiny CNN] {m}") if m else None
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)

    def _run_torchvision_inference(
        self,
        model_path: Path,
        class_names: list,
        input_size: int = 224,
        on_success=None,
        force_monochrome: bool = False,
    ):
        """Launch TorchvisionInferenceWorker and wire signals to the standard post-inference path."""
        if not self.image_paths:
            return

        from ..jobs.task_workers import TorchvisionInferenceWorker

        rt = self._resolve_classifier_compute_runtime(model_path)
        worker = TorchvisionInferenceWorker(
            model_path=model_path,
            image_paths=self.image_paths,
            class_names=class_names,
            input_size=input_size,
            compute_runtime=rt,
            force_monochrome=force_monochrome,
        )

        def _torchvision_success(result):
            self._model_probs = result["probs"]
            self._model_class_names = result["class_names"]
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Custom CNN done: {len(self.image_paths):,} images, "
                f"{len(self._model_class_names)} classes"
            )
            self._active_model_mode = "custom_cnn"
            self._persist_prediction_cache(
                self._model_probs, self._model_class_names, "custom_cnn"
            )
            if on_success:
                on_success(result)

        worker.signals.success.connect(_torchvision_success)
        worker.signals.error.connect(
            lambda e: self.status.showMessage(f"Custom CNN error: {e}")
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[Custom CNN] {m}") if m else None
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)

    def _run_multihead_custom_inference(self, model_paths: list, on_success=None):
        """Run multi-head TinyClassifier/Custom CNN inference and concatenate outputs."""
        import numpy as np
        import torch

        from ..jobs.task_workers import (
            TinyCNNInferenceWorker,
            TorchvisionInferenceWorker,
        )

        if not model_paths or not self.image_paths:
            return

        valid_paths = [Path(path) for path in model_paths if Path(path).exists()]
        if not valid_paths:
            return

        collected: list[tuple[np.ndarray, list[str]]] = []
        total_heads = len(valid_paths)

        def _finish() -> None:
            all_probs = np.concatenate([probs for probs, _ in collected], axis=1)
            all_names = [name for _, names in collected for name in names]
            merged = {"probs": all_probs, "class_names": all_names}
            self._model_probs = all_probs
            self._model_class_names = all_names
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(all_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Multi-head Custom CNN inference done: {len(all_names)} combined classes"
            )
            self._active_model_mode = "custom_cnn_multihead"
            self._persist_prediction_cache(
                all_probs,
                all_names,
                "custom_cnn_multihead",
            )
            if on_success:
                on_success(merged)

        def _run_head(index: int) -> None:
            if index >= total_heads:
                _finish()
                return

            model_path = valid_paths[index]
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            arch = (
                ckpt.get("arch", "tinyclassifier")
                if isinstance(ckpt, dict)
                else "tinyclassifier"
            )
            class_names = (
                list(ckpt.get("class_names") or list(self.classes))
                if isinstance(ckpt, dict)
                else list(self.classes)
            )
            force_monochrome = (
                bool(ckpt.get("monochrome", False)) if isinstance(ckpt, dict) else False
            )
            compute_runtime = self._resolve_classifier_compute_runtime(model_path)

            if arch == "tinyclassifier":
                worker = TinyCNNInferenceWorker(
                    model_path,
                    self.image_paths,
                    class_names,
                    compute_runtime=compute_runtime,
                    batch_size=64,
                    force_monochrome=force_monochrome,
                )
            else:
                raw_size = (
                    ckpt.get("input_size", (224, 224))
                    if isinstance(ckpt, dict)
                    else (224, 224)
                )
                input_size = (
                    raw_size[0]
                    if isinstance(raw_size, (list, tuple))
                    else int(raw_size)
                )
                worker = TorchvisionInferenceWorker(
                    model_path=model_path,
                    image_paths=self.image_paths,
                    class_names=class_names,
                    input_size=input_size,
                    compute_runtime=compute_runtime,
                    batch_size=64,
                    force_monochrome=force_monochrome,
                )

            def _head_success(result, next_index=index + 1):
                collected.append((result["probs"], result["class_names"]))
                _run_head(next_index)

            worker.signals.success.connect(_head_success)
            worker.signals.error.connect(
                lambda error: self.status.showMessage(
                    f"Multi-head Custom CNN error: {error}"
                )
            )
            worker.signals.progress.connect(
                lambda _progress, message, head=index + 1: (
                    self.status.showMessage(
                        f"[Custom CNN {head}/{total_heads}] {message}"
                    )
                    if message
                    else None
                )
            )
            if index == 0:
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
            worker.signals.finished.connect(
                lambda finished_index=index: (
                    self.progress_bar.setVisible(False)
                    if finished_index + 1 >= total_heads
                    else None
                )
            )
            self._threadpool_start(worker)

        _run_head(0)

    def _run_multihead_yolo_inference(
        self,
        model_paths: list,
        on_success=None,
        force_monochrome: bool = False,
    ):
        """Run YOLO inference on each factor model, concatenate log-prob columns."""
        import numpy as np

        if not model_paths or not self.image_paths:
            return

        collected: list = []  # list of (probs ndarray, class_names list) per head
        remaining: list = [len(model_paths)]

        def _head_done(result):
            collected.append((result["probs"], result["class_names"]))
            remaining[0] -= 1
            if remaining[0] == 0:
                # Merge all heads column-wise
                all_probs = np.concatenate([p for p, _ in collected], axis=1)
                all_names = [n for _, names in collected for n in names]
                merged = {"probs": all_probs, "class_names": all_names}
                self._model_probs = all_probs
                self._model_class_names = all_names
                self.umap_model_coords = None
                self.pca_model_coords = None
                self._show_model_umap = False
                self._show_model_pca = False
                self.btn_umap_embedding.setChecked(True)
                self.btn_umap_model.setChecked(False)
                if hasattr(self, "btn_pca_model"):
                    self.btn_pca_model.setChecked(False)
                self.image_confidences = list(all_probs.max(axis=1).astype(float))
                self._set_model_projection_buttons_enabled(True)
                self._update_al_status()
                self.update_explorer_plot()
                self.status.showMessage(
                    f"Multi-head inference done: {len(all_names)} combined classes"
                )
                self._persist_prediction_cache(all_probs, all_names, "yolo_multihead")
                if on_success:
                    on_success(merged)

        for mp in model_paths:
            self._run_yolo_inference(
                mp,
                on_success=_head_done,
                force_monochrome=(
                    force_monochrome or self._resolve_yolo_force_monochrome(mp)
                ),
            )

    def _open_model_history(self):
        """Open the Model History dialog and load a selected past model."""
        if not self.db_path:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        from ..core.store.db import ClassKitDB
        from .dialogs import ModelHistoryDialog

        entries = ClassKitDB(self.db_path).list_model_caches()
        if not entries:
            QMessageBox.information(
                self, "No History", "No trained models found in project history yet."
            )
            return
        dlg = ModelHistoryDialog(
            entries,
            project_path=self.project_path,
            db_path=self.db_path,
            parent=self,
        )
        if dlg.exec() and dlg.selected_entry():
            self._load_model_from_cache_entry(dlg.selected_entry())

    def _load_model_from_cache_entry(self, entry: dict, on_success=None):
        """Load a model from a DB cache entry and run inference + UMAP."""
        import torch

        self._set_active_evaluation_from_meta(entry.get("meta"))
        self._set_heldout_validation_summary(
            self._validation_summary_from_value(
                entry.get("best_val_acc") if isinstance(entry, dict) else None,
                prefix="Saved held-out validation accuracy",
            )
        )
        mode = entry.get("mode", "")
        paths = entry.get("artifact_paths") or []
        if not paths or not Path(paths[0]).exists():
            QMessageBox.warning(
                self,
                "Missing Artifact",
                f"Model file not found:\n{paths[0] if paths else 'unknown'}",
            )
            return
        class_names = entry.get("class_names") or list(self.classes)

        def _after(r):
            self._evaluate_model_on_labeled()
            QTimer.singleShot(100, self._replot_umap_model_space)
            if on_success is not None:
                on_success()

        if "yolo" in mode:
            if mode.startswith("multihead"):
                all_paths = [Path(p) for p in paths if Path(p).exists()]
                self._active_model_mode = "yolo_multihead"
                self._run_multihead_yolo_inference(all_paths, on_success=_after)
            else:
                self._yolo_model_path = Path(paths[0])
                self._active_model_mode = "yolo"
                self._run_yolo_inference(Path(paths[0]), on_success=_after)
        else:
            if mode.startswith("multihead"):
                all_paths = [Path(p) for p in paths if Path(p).exists()]
                self._active_model_mode = "custom_cnn_multihead"
                self._run_multihead_custom_inference(all_paths, on_success=_after)
                return

            ckpt = torch.load(str(paths[0]), map_location="cpu", weights_only=False)
            arch = (
                ckpt.get("arch", "tinyclassifier")
                if isinstance(ckpt, dict)
                else "tinyclassifier"
            )
            resolved_names = (
                list(ckpt.get("class_names") or class_names)
                if isinstance(ckpt, dict)
                else class_names
            )
            force_monochrome = (
                bool(ckpt.get("monochrome", False)) if isinstance(ckpt, dict) else False
            )
            if arch == "tinyclassifier":
                self._active_model_mode = "tiny"
                self._run_tiny_inference(
                    Path(paths[0]),
                    resolved_names,
                    on_success=_after,
                    force_monochrome=force_monochrome,
                )
            else:
                raw_size = (
                    ckpt.get("input_size", (224, 224))
                    if isinstance(ckpt, dict)
                    else (224, 224)
                )
                size = (
                    raw_size[0]
                    if isinstance(raw_size, (list, tuple))
                    else int(raw_size)
                )
                self._active_model_mode = "custom_cnn"
                self._run_torchvision_inference(
                    Path(paths[0]),
                    class_names=resolved_names,
                    input_size=size,
                    on_success=_after,
                    force_monochrome=force_monochrome,
                )

    def _labeled_eval_arrays(self):
        """Return labeled indices, ground-truth ids, and evaluation-scope text."""
        import numpy as np

        labeled_indices = [i for i, lbl in enumerate(self.image_labels) if lbl]
        scope_text = "All labeled project images"
        if self._active_evaluation_paths:
            heldout_indices = [
                index
                for index, (path, lbl) in enumerate(
                    zip(self.image_paths, self.image_labels)
                )
                if lbl
                and self._resolved_path_key(path) in self._active_evaluation_paths
            ]
            split_label = (self._active_evaluation_split_name or "held-out").strip()
            if heldout_indices:
                labeled_indices = heldout_indices
                scope_text = f"{split_label.title()} split"
            else:
                scope_text = "All labeled project images (held-out subset unavailable)"
        if len(labeled_indices) < 2:
            return None, None, scope_text

        eval_class_names = self._evaluation_class_names()
        class_to_id = {c: i for i, c in enumerate(eval_class_names)}
        y_true = np.array(
            [class_to_id.get(self.image_labels[i], -1) for i in labeled_indices]
        )
        valid = y_true >= 0
        if valid.sum() < 2:
            return None, None, scope_text
        return np.array(labeled_indices)[valid], y_true[valid], scope_text

    def _evaluation_class_names(self) -> list[str]:
        """Return the effective class order for metrics and probability alignment."""

        ordered: list[str] = []

        def _add(value) -> None:
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)

        for value in self.classes:
            _add(value)
        for value in self._model_class_names or []:
            _add(value)
        for value in self.image_labels:
            if value:
                _add(value)

        return ordered

    def _aligned_eval_probs(self, idx_arr, probs_rows, class_names=None):
        """Align model probability columns to the project's class order."""
        import numpy as np

        eval_class_names = list(class_names or self._evaluation_class_names())

        if self._model_class_names:
            name_to_col = {
                str(name): i for i, name in enumerate(self._model_class_names)
            }
            probs_subset = np.zeros(
                (len(idx_arr), len(eval_class_names)), dtype=probs_rows.dtype
            )
            for target_col, class_name in enumerate(eval_class_names):
                source_col = name_to_col.get(str(class_name))
                if (
                    source_col is not None
                    and 0 <= int(source_col) < probs_rows.shape[1]
                ):
                    probs_subset[:, target_col] = probs_rows[:, int(source_col)]
            return probs_subset

        usable_cols = min(probs_rows.shape[1], len(eval_class_names))
        probs_subset = np.zeros(
            (len(idx_arr), len(eval_class_names)), dtype=probs_rows.dtype
        )
        if usable_cols > 0:
            probs_subset[:, :usable_cols] = probs_rows[:, :usable_cols]
        return probs_subset

    @staticmethod
    def _validation_summary_from_results(results: list[dict] | None):
        """Build a held-out validation summary from training results."""
        values = []
        for index, result in enumerate(results or []):
            raw = result.get("best_val_acc") if isinstance(result, dict) else None
            if raw is None:
                continue
            try:
                values.append((index, float(raw)))
            except Exception:
                continue
        if not values:
            return None
        if len(values) == 1:
            value = values[0][1]
            return {
                "text": f"Held-out validation accuracy (best epoch): {value:.3f}",
                "short_text": f"Held-out val acc: {value:.3f}",
            }

        mean_value = sum(value for _index, value in values) / len(values)
        per_factor = ", ".join(f"f{index}={value:.3f}" for index, value in values)
        return {
            "text": (
                "Held-out validation accuracy by factor (best epoch per head): "
                f"{per_factor}  |  mean={mean_value:.3f}"
            ),
            "short_text": f"Held-out val acc mean: {mean_value:.3f}",
        }

    @staticmethod
    def _validation_summary_from_value(value, *, prefix: str):
        """Build a held-out validation summary from one scalar value."""
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        return {
            "text": f"{prefix}: {numeric:.3f}",
            "short_text": f"Held-out val acc: {numeric:.3f}",
        }

    def _set_heldout_validation_summary(self, summary) -> None:
        """Update held-out validation state and the metrics banner."""
        self._heldout_validation_summary = (
            summary if isinstance(summary, dict) else None
        )
        if not hasattr(self, "metrics_validation_label"):
            return
        if self._heldout_validation_summary:
            self.metrics_validation_label.setText(
                self._heldout_validation_summary["text"]
            )
        else:
            self.metrics_validation_label.setText(
                "Held-out validation: unavailable for the current prediction set."
            )

    def _heldout_validation_short_text(self) -> str:
        """Return a short held-out validation text snippet for figure captions."""
        if not isinstance(self._heldout_validation_summary, dict):
            return ""
        return str(self._heldout_validation_summary.get("short_text") or "").strip()

    def _evaluate_model_on_labeled(self, activate_metrics_tab: bool = True):
        """Compute metrics from _model_probs on all labeled images and update Metrics tab."""
        if self._model_probs is None:
            return
        try:
            import numpy as np

            from ..core.train.metrics import compute_metrics

            idx_arr, y_true, scope_text = self._labeled_eval_arrays()
            if idx_arr is None or y_true is None:
                return

            eval_class_names = self._evaluation_class_names()
            probs_rows = np.asarray(self._model_probs[idx_arr])
            probs_subset = self._aligned_eval_probs(
                idx_arr,
                probs_rows,
                class_names=eval_class_names,
            )
            y_pred = probs_subset.argmax(axis=1)
            metrics = compute_metrics(
                y_pred,
                y_true,
                class_names=eval_class_names,
            )
            self._update_metrics_display(
                metrics,
                evaluation_scope=scope_text,
                activate_metrics_tab=activate_metrics_tab,
            )
        except Exception as e:
            self.metrics_view.setPlainText(f"Evaluation error: {e}")

    def _update_metrics_display(
        self,
        metrics,
        *,
        evaluation_scope: str = "All labeled project images",
        activate_metrics_tab: bool = True,
    ):
        """Update Metrics tab: text report + matplotlib confusion matrix / per-class bars."""
        from ..core.train.metrics import format_metrics_report

        report = (
            f"Evaluation subset: {evaluation_scope}\n\n{format_metrics_report(metrics)}"
        )
        heldout_text = (
            self._heldout_validation_summary.get("text")
            if isinstance(self._heldout_validation_summary, dict)
            else ""
        )
        if heldout_text:
            report = f"{heldout_text}\n\n{report}"
        self.metrics_view.setPlainText(report)
        if activate_metrics_tab:
            self.tabs.setCurrentWidget(self.metrics_page)

        try:
            import io

            import matplotlib
            import numpy as np

            matplotlib.use("Agg")
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from PySide6.QtGui import QImage, QPixmap

            n_classes = len(metrics.per_class)
            fig = Figure(figsize=(12, 4.5), facecolor="#1e1e1e")
            names = [c.class_name for c in metrics.per_class]

            # ── Left: confusion matrix ──
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_facecolor("#252526")
            cm = metrics.confusion_matrix.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
            img = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax1.set_title(
                "Confusion Matrix (row-normalised)", color="#ddd", fontsize=10
            )
            ax1.set_xlabel("Predicted", color="#aaa", fontsize=9)
            ax1.set_ylabel("True", color="#aaa", fontsize=9)
            ax1.set_xticks(range(n_classes))
            ax1.set_yticks(range(n_classes))
            ax1.set_xticklabels(
                names, rotation=45, ha="right", color="#ccc", fontsize=8
            )
            ax1.set_yticklabels(names, color="#ccc", fontsize=8)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax1.text(
                        j,
                        i,
                        f"{cm[i, j]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if cm_norm[i, j] > 0.5 else "#999",
                    )
            ax1.tick_params(colors="#aaa")
            cbar = fig.colorbar(img, ax=ax1)
            cbar.ax.tick_params(colors="#aaa")

            # ── Right: per-class P/R/F1 bars ──
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_facecolor("#252526")
            x = np.arange(n_classes)
            w = 0.25
            ax2.bar(
                x - w,
                [c.precision for c in metrics.per_class],
                w,
                label="Precision",
                color="#4e9de0",
            )
            ax2.bar(
                x,
                [c.recall for c in metrics.per_class],
                w,
                label="Recall",
                color="#5dbea3",
            )
            ax2.bar(
                x + w, [c.f1 for c in metrics.per_class], w, label="F1", color="#e07b4e"
            )
            ax2.axhline(
                metrics.accuracy,
                color="#ffff66",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
            )
            ax2.set_title(
                "Per-class Precision / Recall / F1", color="#ddd", fontsize=10
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                names, rotation=45, ha="right", color="#ccc", fontsize=8
            )
            ax2.set_ylim(0, 1.08)
            ax2.tick_params(colors="#aaa")
            ax2.spines[:].set_color("#555")
            legend = ax2.legend(
                loc="upper right", fontsize=8, facecolor="#252526", labelcolor="#ddd"
            )
            legend.get_frame().set_edgecolor("#555")

            fig.text(
                0.5,
                0.01,
                f"Eval subset: {evaluation_scope}  |  Accuracy: {metrics.accuracy:.3f}  |  Macro F1: {metrics.macro_f1:.3f}  |  "
                f"Weighted F1: {metrics.weighted_f1:.3f}  |  n={metrics.num_samples}"
                + (
                    f"  |  {self._heldout_validation_short_text()}"
                    if self._heldout_validation_short_text()
                    else ""
                ),
                ha="center",
                color="#aaa",
                fontsize=9,
            )
            fig.tight_layout(rect=[0, 0.05, 1, 1])

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1e1e1e"
            )
            buf.seek(0)
            qimg = QImage.fromData(buf.read())
            pixmap = QPixmap.fromImage(qimg)
            self.metrics_figure_label.setPixmap(pixmap)
            self.metrics_figure_label.setFixedSize(pixmap.size())
        except Exception as fig_exc:
            self.metrics_figure_label.setText(f"Figure error: {fig_exc}")

    def _clear_metrics_display(self) -> None:
        """Reset the Metrics tab to its empty project state."""
        self._set_heldout_validation_summary(None)
        if hasattr(self, "metrics_view"):
            self.metrics_view.clear()
        if hasattr(self, "metrics_figure_label"):
            self.metrics_figure_label.clear()
            self.metrics_figure_label.setPixmap(QPixmap())
            self.metrics_figure_label.setFixedSize(self.metrics_figure_label.sizeHint())
            self.metrics_figure_label.setText("(Train a model to see visualizations)")

    # ================== Model-Space Projections ==================

    def _set_model_projection_buttons_enabled(self, enabled: bool) -> None:
        """Enable/disable model-space projection toggles together."""
        self.btn_umap_model.setEnabled(enabled)
        self.btn_umap_model.setVisible(enabled)
        if hasattr(self, "btn_pca_model"):
            self.btn_pca_model.setEnabled(enabled)
            self.btn_pca_model.setVisible(enabled)

    def _switch_projection_space(self, target: str) -> None:
        """Switch explorer between embedding UMAP, model UMAP, and model PCA."""
        target = str(target or "embedding")

        if target == "model_umap" and self.umap_model_coords is None:
            if self._model_probs is not None:
                self._replot_umap_model_space(auto_switch=True)
                return
            QMessageBox.information(
                self,
                "No Model UMAP",
                "Load a model checkpoint first. Model-space UMAP is computed automatically.",
            )
            target = "embedding"

        if target == "model_pca" and self.pca_model_coords is None:
            if self._model_probs is not None:
                self._replot_pca_model_space(auto_switch=True)
                return
            QMessageBox.information(
                self,
                "No Model PCA",
                "Load a model checkpoint first, then compute model-space PCA.",
            )
            target = "embedding"

        self._show_model_umap = target == "model_umap"
        self._show_model_pca = target == "model_pca"

        self.btn_umap_embedding.setChecked(target == "embedding")
        self.btn_umap_model.setChecked(target == "model_umap")
        if hasattr(self, "btn_pca_model"):
            self.btn_pca_model.setChecked(target == "model_pca")

        self.update_explorer_plot(force_fit=True)

        label = {
            "embedding": "Embeddings (UMAP)",
            "model_umap": "Model Logits (UMAP)",
            "model_pca": "Model Logits (PCA)",
        }.get(target, target)
        self.status.showMessage(f"Explorer space → {label}")

    def _validate_model_umap_input(self) -> str | None:
        """Return a human-readable validation error for model-space UMAP input."""
        if self._model_probs is None:
            return (
                "Load a model first (Load Ckpt or train one). "
                "Predictions are computed automatically during model load."
            )

        probs = np.asarray(self._model_probs)
        if probs.ndim != 2:
            return "Expected a 2D model prediction matrix for model-space UMAP."

        num_samples, num_classes = probs.shape
        if num_samples < 3:
            return "Need at least 3 images with model predictions to compute model-space UMAP."
        if num_classes < 2:
            return "Need at least 2 prediction columns to compute model-space UMAP."
        if not np.isfinite(probs).all():
            return "Model predictions contain NaN or infinite values, so model-space UMAP was skipped."
        return None

    def _replot_umap_model_space(
        self,
        auto_switch: bool = True,
        suppress_errors: bool = False,
        log_callback=None,
    ):
        """Compute UMAP from current model probabilities and switch explorer to it."""
        validation_error = self._validate_model_umap_input()
        if validation_error:
            self.status.showMessage(validation_error)
            if log_callback is not None:
                log_callback(f"Model-space UMAP skipped: {validation_error}")
            if not suppress_errors:
                QMessageBox.warning(self, "No Model UMAP", validation_error)
            return

        from ..jobs.task_workers import LogitsUMAPWorker

        worker = LogitsUMAPWorker(
            self._model_probs,
            n_neighbors=self.last_umap_params.get("n_neighbors", 15),
            min_dist=self.last_umap_params.get("min_dist", 0.1),
        )

        def _on_model_umap_success(result):
            self.umap_model_coords = result["coords"]
            self._set_model_projection_buttons_enabled(True)
            if auto_switch:
                self._switch_projection_space("model_umap")
            self.status.showMessage(
                "Model-space UMAP ready — Explorer switched to model view"
            )
            # Persist so it's skipped on next project open
            if self.db_path:
                try:
                    from ..core.store.db import ClassKitDB

                    ClassKitDB(self.db_path).save_umap_cache(
                        result["coords"],
                        n_neighbors=self.last_umap_params.get("n_neighbors", 15),
                        min_dist=self.last_umap_params.get("min_dist", 0.1),
                        kind="model",
                    )
                except Exception:
                    pass

        def _on_model_umap_error(error_message: str) -> None:
            message = f"Model-space UMAP failed: {error_message}"
            self.status.showMessage(message)
            if log_callback is not None:
                log_callback(message)
            if not suppress_errors:
                QMessageBox.critical(
                    self, "UMAP Error", f"Model UMAP failed:\n{error_message}"
                )

        worker.signals.success.connect(_on_model_umap_success)
        worker.signals.error.connect(_on_model_umap_error)
        worker.signals.progress.connect(
            lambda p, m: (
                self.progress_bar.setValue(p),
                self.status.showMessage(f"[Model UMAP] {m}") if m else None,
            )
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)
        self.status.showMessage("Computing model-space UMAP…")
        self.tabs.setCurrentIndex(0)

    def _replot_pca_model_space(self, auto_switch: bool = True):
        """Compute PCA from current model probabilities and optionally switch to it."""
        if self._model_probs is None:
            QMessageBox.warning(
                self,
                "No Model Predictions",
                "Load a model first (Load Ckpt or train one).",
            )
            return

        from ..jobs.task_workers import LogitsPCAWorker

        worker = LogitsPCAWorker(self._model_probs)

        def _on_model_pca_success(result):
            self.pca_model_coords = result["coords"]
            self._set_model_projection_buttons_enabled(True)
            if auto_switch:
                self._switch_projection_space("model_pca")
            self.status.showMessage("Model-space PCA ready")

            if self.db_path:
                try:
                    from ..core.store.db import ClassKitDB

                    ClassKitDB(self.db_path).save_umap_cache(
                        result["coords"],
                        n_neighbors=2,
                        min_dist=0.0,
                        kind="model_pca",
                    )
                except Exception:
                    pass

        worker.signals.success.connect(_on_model_pca_success)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(self, "PCA Error", f"Model PCA failed:\n{e}")
        )
        worker.signals.progress.connect(
            lambda p, m: (
                self.progress_bar.setValue(p),
                self.status.showMessage(f"[Model PCA] {m}") if m else None,
            )
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)
        self.status.showMessage("Computing model-space PCA…")
        self.tabs.setCurrentIndex(0)

    # ================== Active Learning Batch Builder ==================

    def _update_al_status(self):
        """Refresh the Active Learning status label."""
        if not hasattr(self, "al_status_label"):
            return
        labeled = sum(1 for lbl in self.image_labels if lbl)
        total = len(self.image_paths)
        if self._active_model_mode == "yolo" and self._yolo_model_path:
            model_name = Path(self._yolo_model_path).name
        elif self._active_model_mode == "yolo_multihead":
            model_name = "multi-head YOLO"
        elif self._active_model_mode == "tiny":
            model_name = "tiny CNN"
        elif self._active_model_mode == "custom_cnn":
            model_name = "custom CNN"
        elif self._active_model_mode == "custom_cnn_multihead":
            model_name = "multi-head custom CNN"
        elif self._trained_classifier is not None:
            model_name = "embedding head"
        else:
            model_name = "none"
        preds_ready = self._model_probs is not None
        pct = f"{labeled / total:.0%}" if total else "—"
        _preds_span = (
            "<span style='color:#4ec943'>ready</span>"
            if preds_ready
            else "<span style='color:#888'>none</span>"
        )
        self.al_status_label.setText(
            f"Model: <b>{model_name}</b>  \u00b7  "
            f"Labeled: {labeled:,} / {total:,} ({pct})  \u00b7  "
            f"Predictions: {_preds_span}"
        )
        self.al_status_label.setTextFormat(Qt.RichText)

    def _build_al_batch(self):
        """Launch ALBatchWorker to select the next high-value labeling batch."""
        if self._model_probs is None:
            QMessageBox.warning(
                self,
                "No Predictions",
                "Load a trained model first.\n\n"
                "Use 'Load Ckpt' or finish a training run, then predictions will be "
                "computed automatically.",
            )
            return
        if self.embeddings is None:
            QMessageBox.warning(self, "No Embeddings", "Compute embeddings first.")
            return

        batch_size = (
            self.al_batch_spin.value() if hasattr(self, "al_batch_spin") else 50
        )
        labeled_mask = np.array([bool(lbl) for lbl in self.image_labels])

        from ..jobs.task_workers import ALBatchWorker

        worker = ALBatchWorker(
            embeddings=self.embeddings,
            probs=self._model_probs,
            labeled_mask=labeled_mask,
            cluster_assignments=self.cluster_assignments,
            batch_size=batch_size,
        )
        worker.signals.success.connect(self._on_al_batch_success)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(self, "AL Batch Error", e)
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[AL] {m}") if m else None
        )
        self.al_candidates_badge.setText("  building…")
        self.al_candidate_list.clear()
        self.al_candidate_list.show()
        self.al_start_btn.setEnabled(False)
        self.al_highlight_btn.setEnabled(False)
        self._threadpool_start(worker)

    def _on_al_batch_success(self, result):
        """Populate the Batch Builder candidate list."""
        self._al_candidates = result["selected_indices"]
        breakdown = result["breakdown"]

        # Build per-index reason map
        reason_map = {}
        for reason, indices in breakdown.items():
            for idx in indices:
                reason_map[int(idx)] = reason

        self.al_candidate_list.clear()
        for idx in self._al_candidates:
            idx = int(idx)
            reason = reason_map.get(idx, "selected")
            path_name = (
                self.image_paths[idx].name
                if idx < len(self.image_paths)
                else f"img_{idx}"
            )
            conf = (
                float(self._model_probs[idx].max())
                if self._model_probs is not None
                else 0.0
            )
            pred_col = (
                int(self._model_probs[idx].argmax())
                if self._model_probs is not None
                else 0
            )
            pred_class = (
                self._model_class_names[pred_col]
                if self._model_class_names and pred_col < len(self._model_class_names)
                else f"cls_{pred_col}"
            )
            item_text = (
                f"#{idx:5d}  {path_name:<30s}  conf={conf:.3f}  "
                f"pred={pred_class:<12s}  [{reason}]"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, idx)
            if conf < 0.6:
                from PySide6.QtGui import QColor

                item.setForeground(QColor("#ff9944"))
            self.al_candidate_list.addItem(item)

        n = len(self._al_candidates)
        self.al_candidates_badge.setText(f"  {n} selected")
        self.al_candidate_list.show()
        self.al_start_btn.setEnabled(n > 0)
        self.al_highlight_btn.setEnabled(n > 0)
        self.status.showMessage(
            f"AL batch ready: {n} candidates — click ▶ Label to start"
        )

    def _start_labeling_al_batch(self):
        """Set AL candidates as the active labeling set and enter labeling mode."""
        if self._al_candidates is None or len(self._al_candidates) == 0:
            return
        self._labeling_flow_mode = "batch"
        self._labeling_navigation_scope = "pool"
        new_candidates = [int(i) for i in self._al_candidates]
        # Preserve already-labeled images from the current candidate set so
        # they remain visible after switching to the AL batch.
        labels = self.image_labels or []
        seen = set(new_candidates)
        for i in self.candidate_indices:
            if i not in seen and i < len(labels) and labels[i]:
                new_candidates.append(i)
                seen.add(i)
        self.candidate_indices = new_candidates
        self.set_explorer_mode("labeling")
        self.tabs.setCurrentIndex(0)
        self.update_explorer_plot(force_fit=True)
        self.status.showMessage(
            f"Labeling {len(self.candidate_indices)} AL candidates — "
            "use label buttons or 1-9 keys"
        )

    def _highlight_al_batch_on_map(self):
        """Show AL candidates as the candidate set on the Explorer without entering labeling mode."""
        if self._al_candidates is None or len(self._al_candidates) == 0:
            return
        self.candidate_indices = [int(i) for i in self._al_candidates]
        self.tabs.setCurrentIndex(0)
        self.update_explorer_plot(force_fit=True)
        self.status.showMessage(
            f"Highlighted {len(self.candidate_indices)} AL candidates on map"
        )

    def _al_candidate_goto(self, item):
        """Jump explorer to the selected candidate when double-clicked in Batch Builder."""
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= int(idx) < len(self.image_paths):
            self.selected_point_index = int(idx)
            self.tabs.setCurrentIndex(0)
            self.update_explorer_plot()
            self.load_preview_for_index(int(idx))
            self.load_preview_for_index(int(idx))

    def enter_review_mode(self) -> None:
        """Switch explorer into machine-label review mode."""
        self.set_explorer_mode("review")

    def _machine_label_scope_choices(self) -> list[tuple[str, list[int]]]:
        """Return supported scope options for machine labeling."""
        scopes = self._prediction_scope_choices()
        if self._review_candidate_indices:
            scopes.insert(
                0,
                (
                    f"Pending review only ({len(self._review_candidate_indices)})",
                    [int(i) for i in self._review_candidate_indices],
                ),
            )
        return scopes

    def open_machine_labeling_dialog(self) -> None:
        """Launch the unified machine-labeling dialog and dispatch the selected method."""
        if self.project_path is None or not self.image_paths:
            QMessageBox.information(
                self,
                "No Project Data",
                "Open a project with images before running machine labeling.",
            )
            return

        model_history_entries = []
        if self.db_path:
            from ..core.store.db import ClassKitDB

            model_history_entries = ClassKitDB(self.db_path).list_model_caches()

        from ..gui.dialogs import MachineLabelingDialog

        dlg = MachineLabelingDialog(
            scope_options=self._machine_label_scope_choices(),
            predictions_available=(
                self._model_probs is not None and bool(self._model_class_names)
            ),
            image_count=len(self.image_paths),
            model_history_entries=model_history_entries,
            project_path=self.project_path,
            db_path=self.db_path,
            parent=self,
        )
        if dlg.exec() != MachineLabelingDialog.DialogCode.Accepted:
            return

        settings = dlg.get_settings()
        if settings["method"] == MachineLabelingDialog.METHOD_MODEL:
            self._run_machine_labeling_model_source(settings)
            return

        self._run_apriltag_autolabel_for_scope(
            indices=settings["scope_indices"],
            scope_label=settings["scope_label"],
            config=settings["apriltag_config"],
            threshold=float(settings["apriltag_threshold"]),
            replace_scheme=bool(settings.get("replace_scheme", True)),
            skip_verified=bool(settings.get("skip_verified", True)),
        )

    def _run_machine_labeling_model_source(self, settings: dict) -> None:
        """Load the selected model source and apply its predictions as review labels."""
        from ..gui.dialogs.machine_labeling import MachineLabelingDialog

        indices = [int(i) for i in settings.get("scope_indices") or []]
        scope_label = str(
            settings.get("scope_label") or f"Custom scope ({len(indices)})"
        )
        skip_verified = bool(settings.get("skip_verified", True))
        model_source = str(
            settings.get("model_source") or MachineLabelingDialog.MODEL_SOURCE_LOADED
        )

        if model_source == MachineLabelingDialog.MODEL_SOURCE_LOADED:
            self.apply_model_predictions_as_review_labels(
                indices=indices,
                scope_label=scope_label,
                skip_verified=skip_verified,
                model_provider="loaded_model",
            )
            return

        if model_source == MachineLabelingDialog.MODEL_SOURCE_HISTORY:
            entry = settings.get("model_entry") or {}
            artifact_paths = entry.get("artifact_paths") or []
            if not artifact_paths:
                QMessageBox.warning(
                    self,
                    "No Model Selected",
                    "Choose a model from this project's history before applying machine labels.",
                )
                return

            metadata = {
                "model_name": str(
                    entry.get("display_name")
                    or Path(str(artifact_paths[0])).stem
                    or entry.get("mode")
                    or "history_model"
                ),
                "model_path": str(artifact_paths[0]),
            }
            self._load_model_from_cache_entry(
                entry,
                on_success=lambda: self.apply_model_predictions_as_review_labels(
                    indices=indices,
                    scope_label=scope_label,
                    skip_verified=skip_verified,
                    model_provider="project_history_model",
                    model_metadata=metadata,
                ),
            )
            return

        checkpoint_path = settings.get("checkpoint_path")
        if not checkpoint_path:
            QMessageBox.warning(
                self,
                "No Checkpoint Selected",
                "Choose a checkpoint file before applying machine labels.",
            )
            return

        checkpoint = Path(str(checkpoint_path))
        self._load_checkpoint_from_path(
            checkpoint,
            on_success=lambda: self.apply_model_predictions_as_review_labels(
                indices=indices,
                scope_label=scope_label,
                skip_verified=skip_verified,
                model_provider="external_checkpoint",
                model_metadata={
                    "model_name": checkpoint.stem,
                    "model_path": str(checkpoint),
                },
            ),
            show_message_box=False,
        )

    def _run_apriltag_autolabel_for_scope(
        self,
        *,
        indices: list[int],
        scope_label: str,
        config,
        threshold: float,
        replace_scheme: bool,
        skip_verified: bool,
    ) -> None:
        """Run AprilTag machine labeling for a specific scope from the unified launcher."""
        from ..config.presets import apriltag_preset
        from ..core.store.db import ClassKitDB
        from ..jobs.task_workers import AprilTagAutoLabelWorker

        if self.project_path is None or not indices:
            QMessageBox.information(
                self,
                "No Images In Scope",
                "The selected machine-labeling scope does not contain any images.",
            )
            return

        db = ClassKitDB(self.db_path)
        all_paths = [Path(p) for p in db.get_all_image_paths()]

        if replace_scheme:
            scheme_path = self._project_scheme_path()
            if scheme_path.exists():
                reply = QMessageBox.question(
                    self,
                    "Replace Existing Scheme?",
                    "A labeling scheme already exists.\n\n"
                    "Replacing it will ERASE all existing labels.\n\n"
                    "Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            scheme = apriltag_preset(config.family, config.max_tag_id)
            scheme_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scheme_path, "w") as f:
                json.dump(scheme.to_dict(), f, indent=2)

            db.clear_all_labels()
            self.classes = scheme.factors[0].labels
            self.rebuild_label_buttons()
            target_paths = [
                all_paths[index]
                for index in indices
                if 0 <= int(index) < len(all_paths)
            ]
        else:
            target_paths = []
            for index in indices:
                if index < 0 or index >= len(all_paths):
                    continue
                path = all_paths[index]
                if skip_verified and self._review_status_for_index(index).get(
                    "verified"
                ):
                    continue
                target_paths.append(path)

        if not target_paths:
            QMessageBox.information(
                self,
                "No Images To Process",
                "No images in the selected scope were eligible for AprilTag machine labeling.",
            )
            return

        worker = AprilTagAutoLabelWorker(
            image_paths=target_paths,
            config=config,
            threshold=threshold,
            db=db,
        )

        def _on_progress(pct: int, msg: str) -> None:
            self.statusBar().showMessage(f"AprilTag auto-label: {msg} ({pct}%)")

        def _on_success(result: dict) -> None:
            n_tag = result.get("n_labeled", 0)
            n_no = result.get("n_no_tag", 0)
            n_skip = result.get("n_skipped", 0)
            self.statusBar().showMessage(
                f"Auto-label complete: {n_tag} tagged, {n_no} no_tag, {n_skip} uncertain"
            )
            self._reload_label_state_from_db(db)
            self.update_explorer_plot()
            self.update_context_panel()
            self._update_labeling_progress_indicator()
            self.enter_review_mode()

        def _on_error(msg: str) -> None:
            self.statusBar().showMessage(f"Auto-label error: {msg}")

        worker.signals.progress.connect(_on_progress)
        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(_on_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status.showMessage(
            f"Running AprilTag machine labeling on {len(target_paths):,} images from {scope_label}"
        )
        self._threadpool_start(worker)

    def _run_apriltag_autolabel(self) -> None:
        """Open the AprilTag auto-label dialog and start the background worker."""
        from ..config.presets import apriltag_preset
        from ..core.store.db import ClassKitDB
        from ..gui.dialogs import AprilTagAutoLabelDialog
        from ..jobs.task_workers import AprilTagAutoLabelWorker

        if self.project_path is None:
            return

        # Collect all image paths (dialog needs them for preview)
        db = ClassKitDB(self.db_path)
        all_paths = [Path(p) for p in db.get_all_image_paths()]

        dlg = AprilTagAutoLabelDialog(image_paths=all_paths, parent=self)
        if dlg.exec() != AprilTagAutoLabelDialog.DialogCode.Accepted:
            return

        config = dlg.get_config()
        threshold = dlg.get_threshold()
        max_tag_id = config.max_tag_id

        # Warn if existing scheme will be replaced
        scheme_path = self._project_scheme_path()
        if scheme_path.exists():
            reply = QMessageBox.question(
                self,
                "Replace Existing Scheme?",
                "A labeling scheme already exists.\n\n"
                "Replacing it will ERASE all existing labels.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # 1. Write the new scheme to disk
        scheme = apriltag_preset(config.family, max_tag_id)
        scheme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scheme_path, "w") as f:
            json.dump(scheme.to_dict(), f, indent=2)

        # 2. Clear all existing labels (prevents stale labels under new scheme)
        db.clear_all_labels()

        # 3. Update self.classes and rebuild label buttons
        self.classes = scheme.factors[0].labels
        self.rebuild_label_buttons()

        # 4. Collect unlabeled image paths
        labels = db.get_all_labels()
        unlabeled = [p for p, lbl in zip(all_paths, labels) if lbl is None]

        if not unlabeled:
            QMessageBox.information(
                self, "Auto-label", "No unlabeled images to process."
            )
            return

        # 5. Start worker
        worker = AprilTagAutoLabelWorker(
            image_paths=unlabeled,
            config=config,
            threshold=threshold,
            db=db,
        )

        def _on_progress(pct: int, msg: str) -> None:
            self.statusBar().showMessage(f"AprilTag auto-label: {msg} ({pct}%)")

        def _on_success(result: dict) -> None:
            n_tag = result.get("n_labeled", 0)
            n_no = result.get("n_no_tag", 0)
            n_skip = result.get("n_skipped", 0)
            self.statusBar().showMessage(
                f"Auto-label complete: {n_tag} tagged, {n_no} no_tag, {n_skip} uncertain"
            )
            self._reload_label_state_from_db(db)
            self.update_explorer_plot()
            self.update_context_panel()
            self._update_labeling_progress_indicator()

        def _on_error(msg: str) -> None:
            self.statusBar().showMessage(f"Auto-label error: {msg}")

        worker.signals.progress.connect(_on_progress)
        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(_on_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _prediction_scope_choices(self) -> list[tuple[str, list[int]]]:
        """Return available scopes for applying model predictions."""
        scopes: list[tuple[str, list[int]]] = []

        if (
            self.selected_point_index is not None
            and 0 <= self.selected_point_index < len(self.image_paths)
        ):
            scopes.append(("Selected point", [int(self.selected_point_index)]))

        if self.candidate_indices:
            scopes.append(
                (
                    f"Current candidate set ({len(self.candidate_indices)})",
                    [int(i) for i in self.candidate_indices],
                )
            )

        if self._al_candidates is not None and len(self._al_candidates):
            scopes.append(
                (
                    f"Active-learning batch ({len(self._al_candidates)})",
                    [int(i) for i in self._al_candidates],
                )
            )

        unlabeled = [
            index for index, label in enumerate(self.image_labels or []) if not label
        ]
        if unlabeled:
            scopes.append((f"Unlabeled only ({len(unlabeled)})", unlabeled))

        scopes.append(
            (
                f"Whole dataset ({len(self.image_paths)})",
                list(range(len(self.image_paths))),
            )
        )
        return scopes

    def approve_selected_review_label(self) -> None:
        """Mark the currently selected machine label as verified."""
        if self.selected_point_index is None:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a labeled point before approving it.",
            )
            return

        status = self._review_status_for_index(self.selected_point_index)
        if not status.get("label"):
            QMessageBox.information(
                self,
                "No Label",
                "The selected point does not have a label to approve.",
            )
            return
        if status.get("verified"):
            QMessageBox.information(
                self,
                "Already Verified",
                "The selected label is already verified.",
            )
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        path = self.image_paths[self.selected_point_index]
        updated = db.mark_labels_verified([path])
        if updated:
            refreshed = db.get_label_review_status_by_path().get(str(path), status)
            self._mark_local_review_state(
                path,
                label=refreshed.get("label"),
                label_source=refreshed.get("label_source"),
                verified=bool(refreshed.get("verified")),
                confidence=refreshed.get("confidence"),
                auto_label_metadata=refreshed.get("auto_label_metadata"),
                verified_at=refreshed.get("verified_at"),
            )
            self.status.showMessage(f"Approved label for {Path(path).name}", 4000)
            self.update_context_panel()
            self.load_preview_for_index(self.selected_point_index, source="selection")

    def reject_selected_review_label(self) -> None:
        """Clear the currently selected machine label without touching verified labels."""
        if self.selected_point_index is None:
            QMessageBox.information(
                self,
                "No Selection",
                "Select a labeled point before rejecting it.",
            )
            return

        status = self._review_status_for_index(self.selected_point_index)
        if not status.get("label"):
            QMessageBox.information(
                self,
                "No Label",
                "The selected point does not have a label to reject.",
            )
            return
        if status.get("verified"):
            QMessageBox.information(
                self,
                "Already Verified",
                "Verified labels are not cleared by the review reject action.",
            )
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        path = self.image_paths[self.selected_point_index]
        db.update_labels_batch({str(path): None})
        self._reload_label_state_from_db(db)
        self.update_explorer_plot()
        self.update_context_panel()

        if self.explorer_mode == "review":
            if self._review_candidate_indices:
                if self.selected_point_index not in self._review_candidate_indices:
                    self.selected_point_index = self._review_candidate_indices[0]
                self.load_preview_for_index(
                    self.selected_point_index, source="selection"
                )
            else:
                self.selected_point_index = None
                self.set_explorer_mode("explore")
                self.clear_preview_display()
        elif self.selected_point_index is not None:
            self.load_preview_for_index(self.selected_point_index, source="selection")

        self.status.showMessage(f"Rejected machine label for {Path(path).name}", 4000)

    def approve_all_machine_labels(self) -> None:
        """Bulk-approve every unverified machine-generated label in the project."""
        pending_paths = [
            path
            for path, record in self._image_review_status.items()
            if record.get("label") and not record.get("verified")
        ]
        if not pending_paths:
            QMessageBox.information(
                self,
                "Nothing To Approve",
                "There are no pending machine labels to approve.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Approve All Machine Labels?",
            f"Approve {len(pending_paths):,} unverified machine labels for training/export?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        updated = db.mark_labels_verified(pending_paths)
        if updated:
            self._reload_label_state_from_db(db)
            self.status.showMessage(f"Approved {updated:,} machine labels", 5000)
            self.update_context_panel()
            if self.selected_point_index is not None:
                self.load_preview_for_index(
                    self.selected_point_index, source="selection"
                )

    def clear_all_unverified_machine_labels(self) -> None:
        """Clear every unverified machine-generated label in the project."""
        pending_paths = [
            path
            for path, record in self._image_review_status.items()
            if record.get("label") and not record.get("verified")
        ]
        if not pending_paths:
            QMessageBox.information(
                self,
                "Nothing To Clear",
                "There are no unverified machine labels to clear.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Clear All Unverified Machine Labels?",
            f"Remove {len(pending_paths):,} pending machine labels from the project?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        db.update_labels_batch({str(path): None for path in pending_paths})
        self._reload_label_state_from_db(db)
        self.update_explorer_plot()
        self.update_context_panel()

        if self.explorer_mode == "review" and not self._review_candidate_indices:
            self.selected_point_index = None
            self.set_explorer_mode("explore")
            self.clear_preview_display()
        elif self.selected_point_index is not None:
            self.load_preview_for_index(self.selected_point_index, source="selection")

        self.status.showMessage(
            f"Cleared {len(pending_paths):,} unverified machine labels", 5000
        )

    def apply_model_predictions_as_review_labels(
        self,
        *,
        indices: list[int] | None = None,
        scope_label: str | None = None,
        skip_verified: bool = True,
        model_provider: str = "project_model",
        model_metadata: dict | None = None,
    ) -> None:
        """Apply current model predictions as unverified review labels."""
        if self._model_probs is None or not self._model_class_names:
            QMessageBox.warning(
                self,
                "No Predictions",
                "Load a trained model first so ClassKit has predictions to apply.",
            )
            return
        if not self.db_path or not self.image_paths:
            QMessageBox.warning(
                self,
                "No Project Data",
                "Open a project with images before applying model predictions.",
            )
            return

        if indices is None:
            scope_choices = self._prediction_scope_choices()
            scope_labels = [label for label, _indices in scope_choices]
            scope_label, accepted = QInputDialog.getItem(
                self,
                "Apply Model Predictions",
                "Scope:",
                scope_labels,
                0,
                False,
            )
            if not accepted or not scope_label:
                return

            indices = []
            for label, raw_indices in scope_choices:
                if label == scope_label:
                    indices = raw_indices
                    break
        else:
            indices = [int(i) for i in indices]
            scope_label = scope_label or f"Custom scope ({len(indices)})"

        if not indices:
            QMessageBox.information(
                self,
                "No Images In Scope",
                "The selected scope does not contain any images.",
            )
            return

        from ..core.store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        updates = {}
        metadata_by_path = {}
        skipped_verified = 0
        skipped_unknown = 0

        for index in indices:
            if index < 0 or index >= len(self.image_paths):
                continue
            path = self.image_paths[index]
            status = self._review_status_for_index(index)
            if skip_verified and status.get("verified"):
                skipped_verified += 1
                continue

            pred_idx = int(np.argmax(self._model_probs[index]))
            if pred_idx >= len(self._model_class_names):
                skipped_unknown += 1
                continue
            predicted_label = str(self._model_class_names[pred_idx])
            if predicted_label not in self.classes:
                skipped_unknown += 1
                continue

            confidence = float(np.max(self._model_probs[index]))
            updates[str(path)] = (predicted_label, confidence)
            prediction_metadata = {
                "provider": model_provider,
                "active_model_mode": self._active_model_mode,
                "predicted_index": pred_idx,
                "scope": scope_label,
            }
            if model_metadata:
                prediction_metadata.update(model_metadata)
            metadata_by_path[str(path)] = prediction_metadata

        if not updates:
            QMessageBox.information(
                self,
                "No Predictions Applied",
                "No unverified targets were eligible for prediction application in the selected scope.",
            )
            return

        db.update_labels_with_confidence_batch(
            updates,
            label_source="auto_model",
            verified=False,
            metadata_by_path=metadata_by_path,
        )
        self._reload_label_state_from_db(db)
        self.update_explorer_plot()
        self.update_context_panel()
        if self.selected_point_index is not None:
            self.load_preview_for_index(self.selected_point_index, source="selection")
        self.enter_review_mode()

        self.status.showMessage(
            f"Applied {len(updates):,} model predictions as review labels"
            + (
                f" ({skipped_verified} verified skipped, {skipped_unknown} unknown skipped)"
                if skipped_verified or skipped_unknown
                else ""
            ),
            6000,
        )
