#!/usr/bin/env python3
"""
Main application window for the HYDRA.

Refactored for improved UX with Tabbed interface and logical grouping.
"""

import json
import logging
import math
import os
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.identity.dataset.oriented_video import (
    resolve_individual_dataset_dir,
)
from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.utils.geometry import wrap_angle_degs

from . import model_utils as _model_utils
from .model_utils import (
    _sanitize_model_token,
    get_yolo_model_metadata,
    get_yolo_model_repository_directory,
    make_pose_model_path_relative,
    resolve_pose_model_path,
)
from .widgets.collapsible import CollapsibleGroupBox
from .widgets.help_label import CompactHelpLabel
from .workers.preview_worker import (  # noqa: F401 (re-export for tests)
    _build_preview_background_model,
    _clear_preview_background_cache,
)

try:
    from hydra_suite.posekit.gui.dialogs.utils import get_available_devices
except ImportError:

    def get_available_devices():
        """Return a fallback list of compute device names when PoseKit is unavailable."""
        return ["auto", "cpu", "cuda", "mps"]


# Configuration file for saving/loading tracking parameters
CONFIG_FILENAME = "tracking_config.json"  # Fallback for manual load/save


# Preserve the legacy helper import surface while panels/orchestrators are mid-migration.
get_models_root_directory = _model_utils.get_models_root_directory
get_models_directory = _model_utils.get_models_directory
get_pose_models_directory = _model_utils.get_pose_models_directory
resolve_model_path = _model_utils.resolve_model_path
make_model_path_relative = _model_utils.make_model_path_relative
get_yolo_model_registry_path = _model_utils.get_yolo_model_registry_path
load_yolo_model_registry = _model_utils.load_yolo_model_registry
save_yolo_model_registry = _model_utils.save_yolo_model_registry
register_yolo_model = _model_utils.register_yolo_model


def get_video_config_path(video_path: object) -> object:
    """Get the config file path for a given video file."""
    if not video_path:
        return None
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_config.json")


logger = logging.getLogger(__name__)

_HEADTAIL_USAGE_ROLE_ALIASES = {"headtail", "head_tail"}


class MainWindow(QMainWindow):
    """
    Main application window providing GUI interface for HYDRA configuration, model management, and execution.
    """

    parameters_changed = Signal(dict)

    def __init__(self) -> None:
        """Initialize the main application window and UI components."""
        super().__init__()
        self.config = TrackerConfig()
        self.setWindowTitle("HYDRA")
        self.resize(1360, 850)

        # Apply consistent VSCode dark theme (matches PoseKit and ClassKit)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
            }

            /* Tabs */
            QTabWidget::pane { border: 1px solid #3e3e42; top: -1px; background-color: #1e1e1e; }
            QTabBar::tab {
                background: #2d2d30; color: #cccccc; padding: 6px 10px; margin-right: 2px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                border: 1px solid #3e3e42; border-bottom: none;
            }
            QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; font-weight: 600; border-bottom: 2px solid #007acc; }
            QTabBar::tab:hover:!selected { background: #37373d; }

            /* Group boxes */
            QGroupBox {
                font-weight: 600; border: 1px solid #3e3e42; border-radius: 6px;
                margin-top: 8px; padding: 6px; background-color: #252526;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                left: 8px; padding: 1px 6px;
                background-color: #1e1e1e; color: #9cdcfe; border-radius: 3px;
            }

            /* Buttons */
            QPushButton {
                background-color: #0e639c; border: none; color: #ffffff;
                padding: 5px 12px; border-radius: 4px; min-height: 22px; font-weight: 500;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5a8f; }
            QPushButton:checked { background-color: #094771; border: 1px solid #007acc; }
            QPushButton:disabled { background-color: #3e3e42; color: #777777; border: none; }

            /* Action / Stop buttons */
            QPushButton#ActionBtn {
                background-color: #0e639c; font-weight: bold; font-size: 11px;
            }
            QPushButton#ActionBtn:hover { background-color: #1177bb; }
            QPushButton#StopBtn { background-color: #d9534f; font-weight: bold; }
            QPushButton#StopBtn:hover { background-color: #c9302c; }
            QPushButton#SecondaryBtn {
                background-color: #3a3a3a; color: #d6d6d6;
                font-size: 10px; font-weight: 500; min-height: 20px;
                padding: 4px 10px;
            }
            QPushButton#SecondaryBtn:hover { background-color: #4a4a4a; }

            /* Inputs */
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #3c3c3c; border: 1px solid #3e3e42; border-radius: 4px;
                padding: 4px 8px; color: #e0e0e0;
                selection-background-color: #094771; min-width: 100px; min-height: 22px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover, QComboBox:hover {
                border-color: #0e639c;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {
                border-color: #007acc;
            }

            /* ComboBox dropdown */
            QComboBox::drop-down {
                subcontrol-origin: padding; subcontrol-position: top right;
                width: 20px; border-left: 1px solid #3e3e42;
                background-color: #4a4a4a;
                border-top-right-radius: 4px; border-bottom-right-radius: 4px;
            }
            QComboBox::drop-down:hover { background-color: #5a5a5a; }
            QComboBox QAbstractItemView {
                background-color: #252526; border: 1px solid #3e3e42;
                selection-background-color: #094771; selection-color: #ffffff;
                color: #e0e0e0; outline: none;
            }
            QComboBox QAbstractItemView::item { padding: 6px 10px; min-height: 22px; }
            QComboBox QAbstractItemView::item:hover { background-color: #2a2d2e; }
            QComboBox QAbstractItemView::item:selected { background-color: #094771; color: #ffffff; }

            /* SpinBox arrows */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border; subcontrol-position: top right;
                width: 18px; border-left: 1px solid #3e3e42;
                background-color: #4a4a4a; border-top-right-radius: 4px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover { background-color: #0e639c; }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom right;
                width: 18px; border-left: 1px solid #3e3e42;
                background-color: #4a4a4a; border-bottom-right-radius: 4px;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #0e639c; }

            /* Checkboxes and Radio buttons */
            QCheckBox, QRadioButton { color: #cccccc; spacing: 8px; }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 14px; height: 14px; border: 1px solid #3e3e42;
                border-radius: 3px; background-color: #3c3c3c;
            }
            QRadioButton::indicator { border-radius: 7px; }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #0e639c; border-color: #007acc;
            }
            QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #007acc; }

            /* Labels */
            QLabel { color: #cccccc; background-color: transparent; }

            /* Toolbar */
            QToolBar {
                background-color: #252526; border-bottom: 1px solid #3e3e42;
                spacing: 6px; padding: 4px 6px;
            }
            QToolButton {
                background-color: transparent; border: none; border-radius: 4px;
                padding: 6px 10px; color: #cccccc;
            }
            QToolButton:hover { background-color: #2a2d2e; }
            QToolButton:pressed, QToolButton:checked { background-color: #094771; color: #4fc1ff; }

            /* Status bar */
            QStatusBar {
                background-color: #007acc; color: #ffffff;
                border-top: 1px solid #0098ff; font-weight: 500; font-size: 11px;
            }
            QStatusBar QLabel { background-color: transparent; color: #ffffff; padding: 0px 4px; }

            /* Menu */
            QMenuBar {
                background-color: #252526; color: #cccccc;
                border-bottom: 1px solid #3e3e42; padding: 2px;
            }
            QMenuBar::item { padding: 5px 10px; background-color: transparent; border-radius: 3px; }
            QMenuBar::item:selected { background-color: #2a2d2e; }
            QMenu {
                background-color: #252526; color: #cccccc;
                border: 1px solid #3e3e42; border-radius: 4px; padding: 4px;
            }
            QMenu::item { padding: 6px 20px 6px 12px; border-radius: 3px; }
            QMenu::item:selected { background-color: #094771; color: #ffffff; }
            QMenu::separator { height: 1px; background-color: #3e3e42; margin: 4px 8px; }

            /* Splitter */
            QSplitter::handle { background-color: #3e3e42; }
            QSplitter::handle:hover { background-color: #007acc; }

            /* Scrollbars */
            QScrollBar:vertical { background-color: #252526; width: 10px; border-radius: 5px; margin: 0px; }
            QScrollBar::handle:vertical { background-color: #5a5a5a; border-radius: 5px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background-color: #007acc; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar:horizontal { background-color: #252526; height: 10px; border-radius: 5px; margin: 0px; }
            QScrollBar::handle:horizontal { background-color: #5a5a5a; border-radius: 5px; min-width: 24px; }
            QScrollBar::handle:horizontal:hover { background-color: #007acc; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

            /* Progress bar */
            QProgressBar {
                border: 1px solid #3e3e42; border-radius: 4px;
                text-align: center; background-color: #252526; color: #cccccc; font-size: 11px;
            }
            QProgressBar::chunk { background-color: #0e639c; border-radius: 3px; }

            /* Lists and Tables */
            QListWidget, QTableWidget {
                background-color: #252526; alternate-background-color: #2d2d30; border: 1px solid #3e3e42; border-radius: 4px; outline: none;
            }
            QListWidget::item, QTableWidget::item { padding: 4px 8px; }
            QListWidget::item:selected, QTableWidget::item:selected { background-color: #094771; color: #ffffff; }
            QListWidget::item:hover:!selected, QTableWidget::item:hover:!selected { background-color: #2a2d2e; }
            QHeaderView::section {
                background-color: #2d2d30; color: #cccccc;
                border: none; border-right: 1px solid #3e3e42; border-bottom: 1px solid #3e3e42;
                padding: 4px 8px; font-weight: 600;
            }

            /* Text edits */
            QPlainTextEdit, QTextEdit {
                background-color: #252526; color: #e0e0e0;
                border: 1px solid #3e3e42; border-radius: 4px; padding: 4px;
            }
            QPlainTextEdit:focus, QTextEdit:focus { border-color: #007acc; }

            /* Sliders */
            QSlider::groove:horizontal {
                border: 1px solid #3e3e42; height: 4px;
                background-color: #3c3c3c; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #007acc; border: none;
                width: 14px; height: 14px; border-radius: 7px; margin: -5px 0;
            }
            QSlider::handle:horizontal:hover { background-color: #1177bb; }
            QSlider::sub-page:horizontal { background-color: #0e639c; border-radius: 2px; }
            """)

        # === STATE VARIABLES ===
        self.roi_base_frame = None
        self.roi_points = []
        self.roi_mask = None
        self.roi_selection_active = False
        self.roi_fitted_circle = None
        # Enhanced ROI support: multiple shapes with inclusion/exclusion
        self.roi_shapes = (
            []
        )  # List of dicts: {'type': 'circle'/'polygon', 'params': ..., 'mode': 'include'/'exclude'}
        self.roi_current_mode = "circle"  # 'circle' or 'polygon'
        self.roi_current_zone_type = "include"  # 'include' or 'exclude'

        self.tracking_worker = None
        self.merge_worker = None
        self.csv_writer_thread = None
        self.dataset_worker = None
        self.interp_worker = None
        self.final_media_export_worker = None
        self.preview_detection_worker = None
        self.temporary_files = []  # Track temporary files for cleanup
        self.session_log_handler = None  # Track current session log file handler
        self._individual_dataset_run_id = None
        self.current_detection_cache_path = None
        self.current_individual_properties_cache_path = None
        self.current_detected_properties_cache_path = None
        self.current_detected_cnn_cache_paths = {}
        self.current_interpolated_roi_npz_path = None
        self.current_interpolated_pose_csv_path = None
        self.current_interpolated_pose_df = None
        self.current_interpolated_tag_csv_path = None
        self.current_interpolated_tag_df = None
        self.current_interpolated_cnn_csv_paths = {}
        self.current_interpolated_cnn_dfs = {}
        self.current_interpolated_headtail_csv_path = None
        self.current_interpolated_headtail_df = None
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False
        self._pending_finish_after_track_videos = False

        # Preview frame for live image adjustments
        self.preview_frame_original = None  # Original frame without adjustments
        self.detection_test_result = None  # Store detection test result
        self.current_video_path = None
        self.detected_sizes = None  # Store detected object sizes for statistics
        self.batch_videos = []  # List of video paths for batch mode
        self.current_batch_index = -1

        # ROI display caching (for performance)
        self._roi_masked_cache = {}  # Cache: {(frame_id, roi_hash): masked_image}
        # Interactive pan/zoom state
        self._is_panning = False
        self._pan_start_pos = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0
        self._pan_start_pos = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0

        # Track first frame for auto-fit during tracking
        self._tracking_first_frame = True
        self._tracking_frame_size = None  # (width, height) of resized tracking frames

        # UI interaction state
        self._video_interactions_enabled = True
        self._saved_widget_enabled_states = {}
        self._pending_finish_after_interp = False
        self._stop_all_requested = False

        # Per-session summary state (reset at the start of each forward tracking run)
        self._session_result_dataset = None
        self._dataset_was_started = False
        self._show_summary_on_dataset_done = False
        self._session_wall_start = None
        self._session_final_csv_path = None
        self._session_fps_list = []
        self._session_frames_processed = 0
        self._headless_tracking_mode = False
        self._headless_tracking_callback = None
        self._headless_session_error = None
        self._ui_settings = self._load_ui_settings()
        self._ui_state_save_timer = QTimer(self)
        self._ui_state_save_timer.setSingleShot(True)
        self._ui_state_save_timer.setInterval(250)
        self._ui_state_save_timer.timeout.connect(self._save_ui_settings)
        self._collapsible_state_widgets = {}

        # Advanced configuration (for power users)
        self.advanced_config = self._load_advanced_config()
        self._benchmark_recommendations = {}

        # Video player state
        self.video_cap = None  # cv2.VideoCapture for video playback
        self.video_total_frames = 0
        self.video_current_frame_idx = 0
        self.last_read_frame_idx = (
            -1
        )  # Track last frame read for sequential optimization
        self.is_playing = False
        self.playback_timer = None  # QTimer for playback

        # === UI CONSTRUCTION ===
        self.init_ui()

        # === POST-INIT ===
        # Disable wheel events on all spinboxes to prevent accidental value changes
        self._disable_spinbox_wheel_events()

        # Config is now loaded automatically when a video is selected
        # instead of at startup
        self._connect_parameter_signals()

        # Cache preview-related controls for UI state transitions
        self._preview_controls = self._collect_preview_controls()

        # Restore persisted layout preferences after widgets are fully built.
        QTimer.singleShot(0, self._restore_ui_state)

        # Welcome page is visible at startup — workspace UI state is applied
        # when the user loads a video and _show_workspace() is called.

    def _get_ui_settings_path(self) -> Path:
        """Return the HYDRA UI settings path used for persistent layout preferences."""
        config_dir = Path.home() / ".hydra-suite"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "ui_settings.json"

    def _load_ui_settings(self) -> dict:
        """Load persistent HYDRA UI settings."""
        if hasattr(self, "_session_orch"):
            return self._session_orch._load_ui_settings()
        # Fallback during __init__ before orchestrators exist
        path = self._get_ui_settings_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _queue_ui_state_save(self) -> None:
        """Debounce HYDRA UI settings writes while the user resizes or switches tabs."""
        if hasattr(self, "_session_orch"):
            self._session_orch._queue_ui_state_save()
            return
        if hasattr(self, "_ui_state_save_timer"):
            self._ui_state_save_timer.start()

    def _remember_collapsible_state(
        self, key: str, collapsible: CollapsibleGroupBox
    ) -> None:
        """Restore and track expanded state for a collapsible section."""
        if hasattr(self, "_session_orch"):
            self._session_orch._remember_collapsible_state(key, collapsible)
            return
        # Fallback during early init before orchestrators exist
        self._collapsible_state_widgets[key] = collapsible
        saved = self._ui_settings.get("collapsed_sections", {}).get(key)
        if isinstance(saved, bool):
            collapsible.setExpanded(saved)
        collapsible.toggled.connect(
            lambda _expanded, _key=key: self._queue_ui_state_save()
        )

    def _restore_ui_state(self) -> None:
        self._session_orch._restore_ui_state()

    def _save_ui_settings(self) -> None:
        self._session_orch._save_ui_settings()

    def _set_compact_scroll_layout(self, layout: QLayout) -> None:
        """Prevent scroll-area content layouts from stretching sparse sections vertically."""
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

    def _set_compact_section_widget(self, widget: QWidget) -> None:
        """Make top-level section widgets hug their content vertically."""
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def _make_setup_divider(self) -> QFrame:
        """Create a subtle divider line used inside the Get Started tab."""
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        divider.setStyleSheet("color: #343434; background-color: #343434;")
        divider.setFixedHeight(1)
        return divider

    def init_ui(self: object) -> object:
        """Build the structured UI using Splitter and Tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.main_stack = QStackedWidget()
        root_layout.addWidget(self.main_stack)

        self._welcome_page_index = self.main_stack.addWidget(self._make_welcome_page())

        workspace_page = QWidget()
        self._workspace_page_index = self.main_stack.addWidget(workspace_page)
        self.main_stack.setCurrentIndex(self._welcome_page_index)

        # Main Layout is a horizontal splitter (Video Left | Controls Right)
        main_layout = QHBoxLayout(workspace_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(6)

        # --- LEFT PANEL: Video & ROI ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Video Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("background-color: #121212; border: none;")
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("color: #6a6a6a; font-size: 16px;")
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.video_label.setMinimumSize(320, 240)
        self.scroll.setWidget(self.video_label)
        self._show_video_logo_placeholder()

        # Enable mouse tracking and events for interactive pan/zoom
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._handle_video_mouse_press
        self.video_label.mouseMoveEvent = self._handle_video_mouse_move
        self.video_label.mouseReleaseEvent = self._handle_video_mouse_release
        self.video_label.mouseDoubleClickEvent = self._handle_video_double_click
        self.video_label.wheelEvent = self._handle_video_wheel

        # Enable pinch-to-zoom gesture
        self.video_label.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.video_label.grabGesture(Qt.PinchGesture)
        self.video_label.event = self._handle_video_event

        # ROI Toolbar (Contextual to video)
        roi_frame = QFrame()
        roi_frame.setStyleSheet("background-color: #252526; border-radius: 6px;")
        roi_main_layout = QVBoxLayout(roi_frame)
        roi_main_layout.setContentsMargins(8, 4, 8, 4)
        roi_main_layout.setSpacing(4)

        # Top row: mode selection and controls
        roi_layout = QHBoxLayout()
        roi_label = QLabel("ROI controls")
        roi_label.setStyleSheet("font-weight: bold; color: #cccccc;")

        # Mode selector
        self.combo_roi_mode = QComboBox()
        self.combo_roi_mode.addItems(["Circle", "Polygon"])
        self.combo_roi_mode.setToolTip("Select shape type for ROI")
        self.combo_roi_mode.currentIndexChanged.connect(self._on_roi_mode_changed)

        # Zone type selector (Include/Exclude)
        self.combo_roi_zone = QComboBox()
        self.combo_roi_zone.addItems(["Include Zone", "Exclude Zone"])
        self.combo_roi_zone.setToolTip(
            "Include: Area where tracking is active\n"
            "Exclude: Area to remove from tracking (applied after inclusions)"
        )
        self.combo_roi_zone.currentIndexChanged.connect(self._on_roi_zone_changed)

        self.btn_start_roi = QPushButton("Add Shape")
        self.btn_start_roi.clicked.connect(self.start_roi_selection)
        self.btn_start_roi.setShortcut("Ctrl+R")
        self.btn_start_roi.setToolTip("Start adding ROI shape (Ctrl+R)")

        self.btn_finish_roi = QPushButton("Confirm Shape")
        self.btn_finish_roi.clicked.connect(self.finish_roi_selection)
        self.btn_finish_roi.setEnabled(False)
        self.btn_finish_roi.setShortcut("Ctrl+F")
        self.btn_finish_roi.setToolTip("Finish current shape (Ctrl+F)")

        self.btn_undo_roi = QPushButton("Undo Last")
        self.btn_undo_roi.clicked.connect(self.undo_last_roi_shape)
        self.btn_undo_roi.setEnabled(False)
        self.btn_undo_roi.setToolTip("Remove last added shape")

        self.btn_clear_roi = QPushButton("Clear All")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_clear_roi.setShortcut("Ctrl+C")
        self.btn_clear_roi.setToolTip("Clear all ROI shapes (Ctrl+C)")

        self.btn_crop_video = QPushButton("Crop Video to ROI")
        self.btn_crop_video.clicked.connect(self.crop_video_to_roi)
        self.btn_crop_video.setEnabled(False)
        self.btn_crop_video.setToolTip(
            "Generate a cropped video containing only the ROI area\n"
            "This can significantly improve tracking performance"
        )
        self.btn_crop_video.setStyleSheet(
            "QPushButton { background-color: #2d7a3e; }"
            "QPushButton:hover { background-color: #3a9150; }"
            "QPushButton:disabled { background-color: #3e3e42; color: #777777; }"
        )

        self.roi_status_label = QLabel("No ROI")
        self.roi_status_label.setStyleSheet("color: #6a6a6a; margin-left: 10px;")

        roi_layout.addWidget(roi_label)
        roi_layout.addWidget(self.combo_roi_mode)
        roi_layout.addWidget(self.combo_roi_zone)
        roi_layout.addWidget(self.btn_start_roi)
        roi_layout.addWidget(self.btn_finish_roi)
        roi_layout.addWidget(self.btn_undo_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        roi_layout.addWidget(self.btn_crop_video)
        roi_layout.addStretch()

        roi_main_layout.addLayout(roi_layout)

        # Second row: status and optimization info
        roi_status_layout = QHBoxLayout()
        roi_status_layout.addWidget(self.roi_status_label)
        self.roi_optimization_label = QLabel("")
        self.roi_optimization_label.setStyleSheet(
            "color: #f0ad4e; margin-left: 10px; font-weight: bold;"
        )
        roi_status_layout.addWidget(self.roi_optimization_label)
        roi_main_layout.addLayout(roi_status_layout)

        # Instructions (Hidden unless active)
        self.roi_instructions = QLabel("")
        self.roi_instructions.setWordWrap(True)
        self.roi_instructions.setStyleSheet(
            "color: #4fc1ff; font-size: 11px; font-weight: bold; "
            "padding: 6px; background-color: #0d3354; border-radius: 4px;"
        )
        roi_main_layout.addWidget(self.roi_instructions)

        left_layout.addWidget(self.scroll, stretch=1)

        # Interactive instructions
        self.interaction_help = QLabel(
            "Double-click: Fit to screen  •  Drag: Pan  •  Ctrl+Scroll/Pinch: Zoom"
        )
        self.interaction_help.setAlignment(Qt.AlignCenter)
        self.interaction_help.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic; "
            "padding: 4px; background-color: #1e1e1e; border-radius: 3px;"
        )
        left_layout.addWidget(self.interaction_help)

        left_layout.addWidget(roi_frame)

        # Zoom control under video
        zoom_frame = QFrame()
        zoom_frame.setStyleSheet("background-color: #252526; border-radius: 6px;")
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(8, 4, 8, 4)

        zoom_label = QLabel("Zoom")
        zoom_label.setStyleSheet("font-weight: bold; color: #cccccc;")

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(10, 500)  # 0.1x to 5.0x, scaled by 100
        self.slider_zoom.setValue(100)  # 1.0x
        self.slider_zoom.setTickPosition(QSlider.TicksBelow)
        self.slider_zoom.setTickInterval(50)
        self.slider_zoom.valueChanged.connect(
            lambda v: (
                self._detection_panel._on_zoom_changed(v)
                if hasattr(self, "_detection_panel")
                else None
            )
        )

        self.label_zoom_val = QLabel("1.00x")
        self.label_zoom_val.setStyleSheet(
            "color: #4fc1ff; font-weight: bold; min-width: 50px;"
        )

        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.slider_zoom, stretch=1)
        zoom_layout.addWidget(self.label_zoom_val)

        left_layout.addWidget(zoom_frame)

        # Preview detection button (uses current player frame)
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #252526; border-radius: 6px;")
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 4, 8, 4)

        self.btn_test_detection = QPushButton("Test Detection on Preview")
        self.btn_test_detection.clicked.connect(
            lambda: (
                self._detection_panel._test_detection_on_preview()
                if hasattr(self, "_detection_panel")
                else None
            )
        )
        self.btn_test_detection.setEnabled(False)
        self.btn_test_detection.setStyleSheet(
            "background-color: #0e639c; color: white; font-weight: bold;"
        )
        preview_layout.addWidget(self.btn_test_detection)

        left_layout.addWidget(preview_frame)

        # --- RIGHT PANEL: Configuration Tabs & Actions ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.setUsesScrollButtons(
            True
        )  # Enable scroll buttons when tabs don't fit
        self.tabs.setElideMode(Qt.ElideNone)  # Don't truncate tab text

        # Tab 1: Setup (Files & Performance)
        from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel

        self._setup_panel = SetupPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._setup_panel, "Get Started")

        # Tab 2: Detection (Image, Method, Params)
        from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel

        self._detection_panel = DetectionPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._detection_panel, "Find Animals")

        # Tab 3: Individual Analysis (Identity)
        from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel

        self._identity_panel = IdentityPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._identity_panel, "Analyze Individuals")

        # Tab 4: Tracking (Kalman, Logic, Lifecycle)
        from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel

        self._tracking_panel = TrackingPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._tracking_panel, "Track Movement")

        # Tab 5: Data (Post-proc)
        from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel

        self._postprocess_panel = PostProcessPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._postprocess_panel, "Clean Results")
        # Cross-panel wire: pose-toggle in IdentityPanel must re-run cleaning visibility
        self._identity_panel.g_pose_runtime.toggled.connect(
            self._postprocess_panel._on_cleaning_toggled
        )

        # Tab 6: Dataset Generation (Active Learning)
        from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

        self._dataset_panel = DatasetPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._dataset_panel, "Build Dataset")

        control_panels = (
            self._setup_panel,
            self._detection_panel,
            self._identity_panel,
            self._tracking_panel,
            self._postprocess_panel,
            self._dataset_panel,
        )
        right_panel_min_width = max(
            560,
            self.tabs.minimumSizeHint().width() + 24,
            max(panel.minimumSizeHint().width() for panel in control_panels) + 24,
        )
        right_panel.setMinimumWidth(right_panel_min_width)

        right_layout.addWidget(self.tabs, stretch=1)

        # Persistent Action Panel (Bottom Right)
        action_frame = QFrame()
        action_frame.setStyleSheet(
            "background-color: #1e1e1e; border-top: 1px solid #3e3e42; border-radius: 0px;"
        )
        action_layout = QVBoxLayout(action_frame)
        action_layout.setContentsMargins(6, 6, 6, 6)
        action_layout.setSpacing(4)

        # Progress Info
        prog_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar)

        # Real-time stats display
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(2, 2, 2, 2)

        self.label_current_fps = QLabel("FPS: --")
        self.label_current_fps.setStyleSheet(
            "color: #4fc1ff; font-weight: bold; font-size: 11px;"
        )
        self.label_current_fps.setVisible(False)
        stats_layout.addWidget(self.label_current_fps)

        self.label_elapsed_time = QLabel("Elapsed: --")
        self.label_elapsed_time.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        self.label_elapsed_time.setVisible(False)
        stats_layout.addWidget(self.label_elapsed_time)

        self.label_eta = QLabel("ETA: --")
        self.label_eta.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        self.label_eta.setVisible(False)
        stats_layout.addWidget(self.label_eta)

        stats_layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_preview = QPushButton("Preview Mode")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(lambda ch: self.toggle_preview(ch))
        self.btn_preview.setMinimumHeight(34)

        self.btn_start = QPushButton("Start Full Tracking")
        self.btn_start.setObjectName("ActionBtn")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(lambda ch: self.toggle_tracking(ch))
        self.btn_start.setMinimumHeight(34)

        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_start)

        action_layout.addLayout(prog_layout)
        action_layout.addLayout(stats_layout)
        action_layout.addLayout(btn_layout)

        right_layout.addWidget(action_frame)

        # Add panels to splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)

        # Set initial splitter ratio (60% Video, 40% Controls) and minimum sizes
        total_width = 1360  # Default window width
        self.splitter.setSizes([int(total_width * 0.63), int(total_width * 0.37)])
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.splitterMoved.connect(lambda *_args: self._queue_ui_state_save())
        self.tabs.currentChanged.connect(lambda _index: self._queue_ui_state_save())

        main_layout.addWidget(self.splitter)

        # =====================================================================
        # INITIALIZE ORCHESTRATORS
        # =====================================================================
        from hydra_suite.trackerkit.gui.orchestrators.tracking import (
            TrackingOrchestrator,
        )

        self._tracking_orch = TrackingOrchestrator(
            main_window=self,
            config=self.config,
            panels=self._panels_bundle(),
        )
        from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator

        self._config_orch = ConfigOrchestrator(
            main_window=self,
            config=self.config,
            panels=self._panels_bundle(),
        )
        from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator

        self._session_orch = SessionOrchestrator(
            main_window=self,
            config=self.config,
            panels=self._panels_bundle(),
        )

        # =====================================================================
        # POST-ORCHESTRATOR BOOTSTRAP
        # =====================================================================
        # Setup panel bootstrap
        self._populate_preset_combo()
        self._on_runtime_context_changed()

        # Detection panel bootstrap
        self._detection_panel._refresh_yolo_model_combo()
        self._detection_panel._refresh_yolo_detect_model_combo()
        self._detection_panel._refresh_yolo_crop_obb_model_combo()

        # Identity panel bootstrap
        self._identity_panel._refresh_cnn_identity_model_combo()
        self._identity_panel._refresh_yolo_headtail_model_combo()
        self._update_background_color_button()
        self._refresh_pose_model_combo()
        self._identity_panel._refresh_pose_sleap_envs()
        self._identity_panel._refresh_pose_direction_keypoint_lists()
        self._sync_pose_backend_ui()
        self._sync_individual_analysis_mode_ui()

        # Load default preset (custom if available, otherwise default.json)
        self._load_default_preset_on_startup()

    def _panels_bundle(self):
        """Return a namespace of all panels for orchestrator access."""
        import types

        ns = types.SimpleNamespace()
        ns.setup = self._setup_panel
        ns.detection = self._detection_panel
        ns.tracking = self._tracking_panel
        ns.postprocess = self._postprocess_panel
        ns.dataset = self._dataset_panel
        ns.identity = self._identity_panel
        return ns

    def _make_welcome_page(self):
        """Create startup splash page with primary HYDRA session actions."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("tracker")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="trackerkit.svg",
            tagline="Track  |  Analyze  |  Refine",
            buttons=[
                ButtonDef(label="Load Video\u2026", callback=self.select_file),
                ButtonDef(
                    label="Load Video List\u2026", callback=self._import_batch_list
                ),
                ButtonDef(label="Load Config\u2026", callback=self.load_config),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Videos",
            recents_store=store,
            on_recent_clicked=self._open_recent_video,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    def _open_recent_video(self, path: str):
        """Open a video file from the recent items list."""
        from pathlib import Path

        video_path = Path(path)
        if video_path.exists():
            self._setup_video_file(str(video_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "File Not Found", f"Video not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()

    def _show_workspace(self):
        """Switch to the main HYDRA workspace view."""
        if hasattr(self, "main_stack") and hasattr(self, "_workspace_page_index"):
            self.main_stack.setCurrentIndex(self._workspace_page_index)

    # =========================================================================
    # TAB UI BUILDERS
    # =========================================================================

    def _find_or_plan_optimizer_cache_path(
        self, video_path: str, params: dict, start_frame: int, end_frame: int
    ) -> tuple:
        """Return (cache_path, already_valid) for the optimizer cache."""
        return self._config_orch._find_or_plan_optimizer_cache_path(
            video_path, params, start_frame, end_frame
        )

    def _build_optimizer_detection_cache(
        self, video_path: str, cache_path: str, params: dict
    ):
        """Spin up a DetectionCacheBuilderWorker and show progress in the main window."""
        self._config_orch._build_optimizer_detection_cache(
            video_path, cache_path, params
        )

    def _on_optimizer_cache_built(self, ok: bool, cache_path: str):
        """Called when DetectionCacheBuilderWorker finishes."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if getattr(self, "_stop_all_requested", False):
            return

        if not ok:
            QMessageBox.critical(
                self,
                "Detection Failed",
                "Could not build the detection cache. Check that the YOLO model "
                "path is correct and that the video is readable.",
            )
            return
        # Store built cache as the current cache so _open_parameter_helper finds it
        self.current_detection_cache_path = cache_path
        # Re-open the helper now that the cache is ready
        self._open_parameter_helper()

    def _on_preview_cache_built(self, ok: bool, cache_path: str):
        """Called when DetectionCacheBuilderWorker finishes building the preview cache."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if getattr(self, "_stop_all_requested", False):
            return

        if not ok:
            QMessageBox.critical(
                self,
                "Detection Failed",
                "Could not build the detection cache. Check that the YOLO model "
                "path is correct and that the video is readable.",
            )
            return
        self.current_detection_cache_path = cache_path
        video_path = getattr(
            self,
            "_pending_preview_video_path",
            self._setup_panel.file_line.text().strip(),
        )
        self.start_preview_on_video(video_path)

    def _apply_optimized_params(self, new_params):
        """Apply optimized parameter values from the helper dialog to UI widgets."""
        self._config_orch._apply_optimized_params(new_params)

    def _open_parameter_helper(self):
        """Open the tracking parameter selection helper dialog."""
        self._config_orch._open_parameter_helper()

    def _open_bg_parameter_helper(self):
        """Open the BG-subtraction parameter auto-tuner dialog."""
        self._config_orch._open_bg_parameter_helper()

    def _open_benchmark_dialog(self):
        """Open the runtime and batch benchmarking dialog."""
        self._config_orch._open_benchmark_dialog()

    def _refresh_benchmark_recommendations(self) -> None:
        """Refresh cached benchmark recommendations for the current UI state."""
        self._config_orch._refresh_benchmark_recommendations()

    def _current_detection_benchmark_recommendation(self):
        """Return the active detection recommendation for the current OBB mode."""
        key = "detection_sequential"
        if (
            hasattr(self, "_detection_panel")
            and self._detection_panel.combo_yolo_obb_mode.currentIndex() == 0
        ):
            key = "detection_direct"
        return self._benchmark_recommendations.get(key)

    def _current_headtail_benchmark_recommendation(self):
        """Return the cached head-tail recommendation, if available."""
        return self._benchmark_recommendations.get("headtail")

    def _current_pose_benchmark_recommendation(self):
        """Return the cached pose recommendation for the active backend."""
        return self._benchmark_recommendations.get(
            f"pose_{self._current_pose_backend_key()}"
        )

    def _current_cnn_benchmark_recommendations(self):
        """Return cached CNN recommendations keyed by classifier index."""
        return {
            key: value
            for key, value in self._benchmark_recommendations.items()
            if str(key).startswith("cnn_")
        }

    def _current_cnn_runtime_recommendation(self):
        """Return a shared CNN runtime recommendation only when all rows agree."""
        recommendations = list(self._current_cnn_benchmark_recommendations().values())
        if not recommendations:
            return None
        runtimes = {recommendation.runtime for recommendation in recommendations}
        if len(runtimes) != 1:
            return None
        return recommendations[0]

    def _ensure_pose_model_path_store(self):
        if not hasattr(self, "_pose_model_path_by_backend"):
            self._pose_model_path_by_backend = {"yolo": "", "sleap": "", "vitpose": ""}

    def _current_pose_backend_key(self):
        if not hasattr(self, "_identity_panel"):
            return "yolo"
        backend = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        if backend == "sleap":
            return "sleap"
        if backend == "vitpose":
            return "vitpose"
        return "yolo"

    def _pose_model_path_for_backend(self, backend=None):
        self._ensure_pose_model_path_store()
        key = (backend or self._current_pose_backend_key()).strip().lower()
        if key not in ("sleap", "vitpose"):
            key = "yolo"
        return str(self._pose_model_path_by_backend.get(key, "")).strip()

    def _get_resolved_pose_model_dir(self, backend: str) -> object:
        """Resolve pose model path for context dicts — callable by DetectionPanel."""
        return resolve_pose_model_path(
            self._pose_model_path_for_backend(backend),
            backend=backend,
        )

    def _set_pose_model_path_for_backend(self, path, backend=None, update_combo=False):
        self._ensure_pose_model_path_store()
        key = (backend or self._current_pose_backend_key()).strip().lower()
        if key not in ("sleap", "vitpose"):
            key = "yolo"
        value = str(path or "").strip()
        if value:
            resolved = str(resolve_pose_model_path(value, backend=key)).strip()
            if resolved and os.path.exists(resolved):
                value = str(make_pose_model_path_relative(os.path.abspath(resolved)))
        self._pose_model_path_by_backend[key] = value
        if update_combo:
            self._refresh_pose_model_combo(preferred_model_path=value)

    def _handle_add_new_pose_model(self):
        """Browse for a pose model, import it if outside repo, refresh combo, and select it."""
        self._config_orch._handle_add_new_pose_model()

    def _import_pose_model_to_repository(self, source_path, backend="yolo"):
        """Copy a selected pose model into models/pose/{YOLO|SLEAP|ViTPose} and return relative path."""
        return self._config_orch._import_pose_model_to_repository(
            source_path, backend=backend
        )

    def _select_pose_skeleton_file(self):
        """Select pose skeleton JSON file."""
        start = self._identity_panel.line_pose_skeleton_file.text().strip() or str(
            Path.home()
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pose Skeleton JSON",
            start,
            "JSON Files (*.json);;All Files (*)",
        )
        if selected:
            self._identity_panel.line_pose_skeleton_file.setText(selected)

    def _load_pose_skeleton_keypoint_names(self):
        """Load keypoint names from selected skeleton JSON."""
        skeleton_file = self._identity_panel.line_pose_skeleton_file.text().strip()
        if not skeleton_file:
            return []
        try:
            path = Path(skeleton_file).expanduser().resolve()
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("keypoint_names", data.get("keypoints", []))
            names = [str(v).strip() for v in raw if str(v).strip()]
            return names
        except Exception:
            return []

    def _selected_pose_group_keypoints(self, list_widget):
        """Return selected keypoint names for a pose direction list widget."""
        if list_widget is None:
            return []
        return [
            item.text().strip()
            for item in list_widget.selectedItems()
            if item.text().strip()
        ]

    def _pose_group_list_widget(self, attr_name):
        """Resolve a pose-group list widget from compatibility attrs or the identity panel."""
        widget = getattr(self, attr_name, None)
        if widget is not None:
            return widget
        panel = getattr(self, "_identity_panel", None)
        if panel is None:
            return None
        return getattr(panel, attr_name, None)

    def _set_pose_group_selection(self, list_widget, values):
        """Select keypoints in list widget from config-provided values."""
        if list_widget is None:
            return
        tokens = self._parse_pose_keypoint_tokens(values)
        if not tokens:
            return
        items_by_name = {}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            items_by_name[item.text().strip().lower()] = item
        for tok in tokens:
            if isinstance(tok, int):
                if 0 <= tok < list_widget.count():
                    list_widget.item(tok).setSelected(True)
                continue
            item = items_by_name.get(str(tok).strip().lower())
            if item is not None:
                item.setSelected(True)

    def _apply_pose_keypoint_selection_set(self, list_widget, selected_names):
        """Set list selection to exact keypoint-name set."""
        if list_widget is None:
            return
        target = {str(v).strip().lower() for v in selected_names if str(v).strip()}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_name = item.text().strip().lower()
            item.setSelected(item_name in target)

    def _set_pose_keypoints_enabled(self, list_widget, disabled_names):
        """Disable keypoints in a list by name while preserving others."""
        if list_widget is None:
            return
        disabled = {str(v).strip().lower() for v in disabled_names if str(v).strip()}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_name = item.text().strip().lower()
            flags = item.flags()
            if item_name in disabled:
                item.setSelected(False)
                item.setFlags(flags & ~Qt.ItemIsEnabled)
            else:
                item.setFlags(flags | Qt.ItemIsEnabled)

    def _apply_pose_keypoint_selection_constraints(self, changed_group="ignore"):
        """Enforce exclusivity across ignore/anterior/posterior keypoint selections."""
        ignore_list = self._pose_group_list_widget("list_pose_ignore_keypoints")
        ant_list = self._pose_group_list_widget("list_pose_direction_anterior")
        post_list = self._pose_group_list_widget("list_pose_direction_posterior")
        if ignore_list is None or ant_list is None or post_list is None:
            return

        ignore = {
            k.lower()
            for k in self._selected_pose_group_keypoints(ignore_list)
            if k.strip()
        }
        anterior = {
            k.lower()
            for k in self._selected_pose_group_keypoints(ant_list)
            if k.strip()
        }
        posterior = {
            k.lower()
            for k in self._selected_pose_group_keypoints(post_list)
            if k.strip()
        }

        # Ignore selection always wins over directional groups.
        anterior -= ignore
        posterior -= ignore

        # Anterior/posterior remain mutually exclusive. The changed group wins.
        if changed_group == "posterior":
            anterior -= posterior
        else:
            posterior -= anterior

        ant_list.blockSignals(True)
        post_list.blockSignals(True)
        self._apply_pose_keypoint_selection_set(ant_list, anterior)
        self._apply_pose_keypoint_selection_set(post_list, posterior)
        self._set_pose_keypoints_enabled(ant_list, ignore)
        self._set_pose_keypoints_enabled(post_list, ignore)
        ant_list.blockSignals(False)
        post_list.blockSignals(False)

    def _on_pose_keypoint_group_changed(self, changed_group):
        """Handle keypoint group updates and keep list constraints synchronized."""
        self._apply_pose_keypoint_selection_constraints(changed_group)

    def _parse_pose_keypoint_tokens(self, raw):
        """Parse comma-separated keypoint names and/or indices."""
        if not raw:
            return []
        if isinstance(raw, (list, tuple, set)):
            tokens = []
            for value in raw:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                try:
                    tokens.append(int(text))
                except ValueError:
                    tokens.append(text)
            return tokens
        out = []
        for token in str(raw).split(","):
            t = token.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except ValueError:
                out.append(t)
        return out

    def _parse_pose_ignore_keypoints(self):
        """Parse keypoints to ignore from list selection."""
        return self._selected_pose_group_keypoints(
            self._pose_group_list_widget("list_pose_ignore_keypoints")
        )

    def _parse_pose_direction_anterior_keypoints(self):
        """Parse anterior keypoint group from list selection."""
        return self._selected_pose_group_keypoints(
            self._pose_group_list_widget("list_pose_direction_anterior")
        )

    def _parse_pose_direction_posterior_keypoints(self):
        """Parse posterior keypoint group from list selection."""
        return self._selected_pose_group_keypoints(
            self._pose_group_list_widget("list_pose_direction_posterior")
        )

    def _selected_pose_sleap_env(self):
        """Return valid selected SLEAP env name or default."""
        if not hasattr(self, "_identity_panel"):
            return "sleap"
        txt = self._identity_panel.combo_pose_sleap_env.currentText().strip()
        if not txt or txt.lower().startswith("no sleap envs"):
            return "sleap"
        return txt

    def _on_runtime_tier_changed(self, _index: int = 0) -> None:
        """Store the selected tier and refresh dependent controls."""
        if hasattr(self, "_session_orch"):
            self._session_orch._on_runtime_tier_changed()

    def _selected_runtime_tier(self) -> str:
        """Return the currently selected runtime tier id.

        Prefers the GUI selector; falls back to the loaded config's
        ``runtime_tier`` (headless CLI runs have no populated combo) before the
        ``"gpu"`` default.
        """
        if hasattr(self, "_setup_panel") and hasattr(
            self._setup_panel, "combo_runtime_tier"
        ):
            data = self._setup_panel.combo_runtime_tier.currentData()
            if data:
                return str(data)
        cfg = getattr(self, "config", None)
        cfg_tier = getattr(cfg, "runtime_tier", None) if cfg is not None else None
        if cfg_tier:
            return str(cfg_tier)
        return "gpu"

    def _selected_compute_runtime(self) -> str:
        """Derive a concrete compute_runtime string from the selected tier."""
        if hasattr(self, "_session_orch"):
            return self._session_orch._selected_compute_runtime()
        return "cpu"

    def _selected_headtail_runtime(self) -> str:
        """Return the runtime key for head-tail (same as detection tier runtime)."""
        return self._selected_compute_runtime()

    def _selected_cnn_runtime(self) -> str:
        """Return the runtime key for CNN identity (same as detection tier runtime)."""
        return self._selected_compute_runtime()

    def _runtime_requires_fixed_yolo_batch(self, runtime=None) -> bool:
        """Return True when runtime mandates a fixed YOLO batch size."""
        if hasattr(self, "_session_orch"):
            return self._session_orch._runtime_requires_fixed_yolo_batch(runtime)
        return False

    @staticmethod
    def _preview_safe_runtime(runtime: str) -> str:
        """Downgrade ONNX/TensorRT runtimes to their native equivalents."""
        rt = str(runtime or "cpu").strip().lower()
        if rt == "onnx_cpu":
            return "cpu"
        if rt == "onnx_coreml":
            return "mps"
        if rt in ("onnx_cuda", "tensorrt"):
            return "cuda"
        return rt

    def _on_runtime_context_changed(self, *_args):
        """Update runtime combo and sync dependent controls."""
        if hasattr(self, "_session_orch"):
            self._session_orch._on_runtime_context_changed(*_args)

    def _selected_pose_runtime_flavor(self) -> str:
        """Return the currently selected pose runtime flavor key."""
        if hasattr(self, "_session_orch"):
            return self._session_orch._selected_pose_runtime_flavor()
        return "cpu"

    def _set_form_row_visible(self, form_layout, field_widget, visible: bool):
        """Show/hide a QFormLayout row by field widget."""
        if hasattr(self, "_session_orch"):
            self._session_orch._set_form_row_visible(form_layout, field_widget, visible)
            return
        if form_layout is None or field_widget is None:
            return
        label = form_layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(bool(visible))
        field_widget.setVisible(bool(visible))

    def _sync_pose_backend_ui(self):
        """Show/hide backend-specific pose controls."""
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_pose_backend_ui()

    def _on_confidence_density_map_toggled(self, state):
        """Compatibility wrapper for tracking density toggle handling."""
        if hasattr(self, "_tracking_panel"):
            self._tracking_panel._on_confidence_density_map_toggled(state)

    def _is_pose_inference_enabled(self) -> bool:
        """Return whether pose inference is actively enabled for the run."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_pose_inference_enabled()

    def _sync_video_pose_overlay_controls(self, *_args):
        """Gate pose video overlay controls based on pose inference enable state."""
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_video_pose_overlay_controls(*_args)

    def _is_yolo_detection_mode(self) -> bool:
        """Return True when current detection mode is YOLO OBB."""
        if not hasattr(self, "_detection_panel"):
            return False
        return self._detection_panel._is_yolo_detection_mode()

    def _is_individual_pipeline_enabled(self) -> bool:
        """Return effective runtime state for individual analysis pipeline."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_individual_pipeline_enabled()

    def _is_identity_analysis_enabled(self) -> bool:
        """Return effective runtime state for identity classification."""
        if not hasattr(self, "_detection_panel"):
            return False
        return self._detection_panel._is_identity_analysis_enabled()

    def _selected_identity_method(self) -> str:
        """Return canonical identity-method key for runtime/config usage."""
        if not hasattr(self, "_detection_panel"):
            return "none_disabled"
        return self._detection_panel._selected_identity_method()

    def _identity_config(self) -> dict:
        """Return use_apriltags + cnn_classifiers config dict."""
        if not hasattr(self, "_detection_panel"):
            return {"use_apriltags": False, "cnn_classifiers": []}
        return self._detection_panel._identity_config()

    def _is_individual_image_save_enabled(self) -> bool:
        """Return effective runtime state for final canonical still export."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_individual_image_save_enabled()

    def _is_realtime_tracking_mode_enabled(self) -> bool:
        """Return True when the setup tab requests realtime workflow."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_realtime_tracking_mode_enabled()

    def _should_export_final_media_videos(self) -> bool:
        """Return True when final per-track videos should be exported."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._should_export_final_media_videos()

    def _should_run_interpolated_postpass(self) -> bool:
        """Return True when interpolated post-pass should run."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._should_run_interpolated_postpass()

    def _sync_individual_analysis_mode_ui(self):
        """Enforce YOLO-only pipeline and run/save dependency in UI."""
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_individual_analysis_mode_ui()

    def _select_individual_background_color(self):
        """Open color picker for individual dataset background color."""
        self._session_orch._select_individual_background_color()

    def _update_background_color_button(self):
        """Update the color button display and label."""
        if hasattr(self, "_session_orch"):
            self._session_orch._update_background_color_button()

    def _compute_median_background_color(self):
        """Compute median color from current preview frame or load from video."""
        self._session_orch._compute_median_background_color()

    def closeEvent(self, event) -> None:
        """Stop active workers cleanly, then persist UI layout state on close."""
        if self._has_active_tracking_workers():
            reply = QMessageBox.question(
                self,
                "Tracking In Progress",
                "Tracking is currently running. Stop tracking and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self._tracking_orch.stop_tracking()
            except Exception:
                logger.exception("Error during stop_tracking on close.")
            finally:
                QApplication.restoreOverrideCursor()

        self._save_ui_settings()
        super().closeEvent(event)

    def _has_active_tracking_workers(self) -> bool:
        """Return True if any tracking-related worker thread is still running."""
        worker_attrs = (
            "tracking_worker",
            "_cache_builder_worker",
            "merge_worker",
            "postprocess_worker",
            "dataset_worker",
            "interp_worker",
            "final_media_export_worker",
            "preview_detection_worker",
        )
        for attr in worker_attrs:
            worker = getattr(self, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    return True
            except Exception:
                continue
        writer = getattr(self, "csv_writer_thread", None)
        if writer is not None:
            try:
                if writer.is_alive():
                    return True
            except Exception:
                pass
        return False

    def select_file(self: object) -> object:
        """Select video file via file dialog."""
        from hydra_suite.paths import get_projects_dir

        start_fp = (
            self._setup_panel.file_line.text().strip()
            if hasattr(self, "_setup_panel")
            else ""
        )
        start_dir = os.path.dirname(start_fp) if start_fp else str(get_projects_dir())
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Video", start_dir, "Video Files (*.mp4 *.avi *.mov)"
        )
        if fp:
            # If batch mode is checked, update the keystone
            if self._setup_panel.g_batch.isChecked():
                if not self.batch_videos:
                    self.batch_videos = [fp]
                else:
                    if fp in self.batch_videos:
                        self.batch_videos.remove(fp)
                    self.batch_videos.insert(0, fp)
                self._sync_batch_list_ui()

            self._setup_video_file(fp)

    def _setup_video_file(self, fp, skip_config_load=False, _probe=None):
        """
        Setup a video file for tracking.

        Args:
            fp: Path to the video file
            skip_config_load: If True, skip auto-loading config (used when loading config itself)
            _probe: Optional pre-loaded video probe from SessionOrchestrator._probe_video_io
        """
        self._config_orch._setup_video_file(
            fp, skip_config_load=skip_config_load, _probe=_probe
        )

    def _on_batch_mode_toggled(self, checked):
        """Handle showing/hiding batch controls and syncing keystone video."""
        self._setup_panel.lbl_batch_warning.setVisible(checked)
        self._setup_panel.container_batch.setVisible(checked)
        if hasattr(self, "_postprocess_panel"):
            self._postprocess_panel.set_batch_mode_active(checked)
        if checked:
            self._sync_keystone_to_batch()
        else:
            # If turning off batch mode, keep the current video but reset batch tracking state
            self.current_batch_index = -1

    def _sync_keystone_to_batch(self):
        """Ensure the currently loaded video is the FIRST item in the batch list."""
        video_path = self._setup_panel.file_line.text().strip()
        if not video_path:
            return

        if not self.batch_videos:
            self.batch_videos = [video_path]
        else:
            # If current video is already in list, move it to top
            if video_path in self.batch_videos:
                self.batch_videos.remove(video_path)
            self.batch_videos.insert(0, video_path)

        self._sync_batch_list_ui()

    def _sync_batch_list_ui(self):
        """Refresh the batch list widget with markers for the keystone."""
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_batch_list_ui()

    def _on_batch_video_double_clicked(self, *args):
        """Switch to a video from the batch list: auto-saves current config then loads the new video's config."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.warning(
                self,
                "Tracking Active",
                "Cannot switch videos while tracking is running.",
            )
            return

        row = self._setup_panel.list_batch_videos.currentRow()
        if not (0 <= row < len(self.batch_videos)):
            return
        fp = self.batch_videos[row]
        if fp == self.current_video_path:
            return
        previous_video_path = self.current_video_path

        # Auto-save the current video's config before switching
        if self.current_video_path:
            current_name = os.path.basename(self.current_video_path)
            saved_ok = self.save_config(prompt_if_exists=False)
            if saved_ok:
                self.statusBar().showMessage(
                    f"✓ Auto-saved config for {current_name}", 4000
                )
            else:
                if (
                    QMessageBox.warning(
                        self,
                        "Config Save Failed",
                        f"Could not auto-save config for {current_name}.\n\nSwitch anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    != QMessageBox.Yes
                ):
                    return

        # Stop any active playback before entering the nested event loop so that
        # the QTimer.singleShot chain cannot fire against a VideoCapture that is
        # being torn down or replaced.
        if self.is_playing:
            self._stop_playback()

        # Disable the batch list to prevent re-entrant double-clicks while loading.
        self._setup_panel.list_batch_videos.setEnabled(False)

        # Load the new video's metadata in a background thread while showing a
        # progress dialog (keeps the UI painting without blocking).
        from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator
        from hydra_suite.widgets.busy import (
            BusyTaskError,
            run_blocking_with_busy_dialog,
        )

        try:
            probe = run_blocking_with_busy_dialog(
                self,
                lambda ss, sp: SessionOrchestrator._probe_video_io(fp, ss, sp),
                title="Loading Video",
                message=f"Opening {os.path.basename(fp)}\u2026",
            )
        except BusyTaskError as exc:
            QMessageBox.warning(self, "Video Load Failed", str(exc))
            self._setup_panel.list_batch_videos.setEnabled(True)
            return
        finally:
            self._setup_panel.list_batch_videos.setEnabled(True)

        self._setup_video_file(fp, skip_config_load=False, _probe=probe)
        self._session_orch._refresh_batch_list_current_video(previous_video_path, fp)

    def _remove_from_batch(self):
        """Remove selected additional video from the batch list."""
        row = self._setup_panel.list_batch_videos.currentRow()
        if row == 0:
            QMessageBox.information(
                self,
                "Cannot Remove",
                "The Keystone video cannot be removed from the batch.",
            )
            return
        if row > 0:
            self.batch_videos.pop(row)
            self._sync_batch_list_ui()

    def _export_batch_list(self):
        """Save the current batch video list to a plain-text file (one path per line, keystone first)."""
        if not self.batch_videos:
            QMessageBox.information(
                self,
                "Nothing to Export",
                "The batch list is empty. Add videos first.",
            )
            return

        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export Batch Video List",
            "",
            "Batch List (*.txt);;All Files (*)",
        )
        if not fp:
            return

        try:
            with open(fp, "w", encoding="utf-8") as fh:
                for path in self.batch_videos:
                    fh.write(path + "\n")
            QMessageBox.information(
                self,
                "Exported",
                f"Batch list saved to:\n{fp}\n\n"
                f"{len(self.batch_videos)} video(s) listed (first = keystone).",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _import_batch_list(self):
        """Load a batch video list from a plain-text file and set up keystone + additional videos."""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Import Batch Video List",
            "",
            "Batch List (*.txt);;All Files (*)",
        )
        if not fp:
            return

        try:
            with open(fp, "r", encoding="utf-8") as fh:
                lines = [ln.rstrip("\n").strip() for ln in fh if ln.strip()]
        except OSError as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
            return

        if not lines:
            QMessageBox.warning(
                self, "Empty File", "The selected file contains no video paths."
            )
            return

        missing = [p for p in lines if not os.path.isfile(p)]
        valid = [p for p in lines if os.path.isfile(p)]

        if missing:
            missing_summary = "\n".join(f"  • {p}" for p in missing[:20])
            if len(missing) > 20:
                missing_summary += f"\n  … and {len(missing) - 20} more"

            if not valid:
                QMessageBox.critical(
                    self,
                    "Import Failed – All Paths Missing",
                    f"None of the {len(missing)} path(s) in the file could be found:\n\n{missing_summary}",
                )
                return

            if lines[0] not in valid:
                QMessageBox.critical(
                    self,
                    "Import Failed – Keystone Missing",
                    f"The keystone video (first line) does not exist:\n\n  {lines[0]}\n\n"
                    "Cannot import without a valid keystone.",
                )
                return

            reply = QMessageBox.question(
                self,
                "Missing Videos",
                f"{len(missing)} path(s) could not be found and will be skipped:\n\n"
                f"{missing_summary}\n\nProceed with {len(valid)} valid video(s)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if not valid:
            QMessageBox.warning(
                self, "Nothing Imported", "No valid video paths were found."
            )
            return

        keystone = valid[0]

        # Activate batch mode if not already on, then load the keystone
        self._setup_panel.g_batch.setChecked(True)

        # Replace the batch list with the imported paths (keystone first)
        self.batch_videos = valid

        # Trigger full keystone setup (loads config, autofills outputs, etc.)
        self._setup_video_file(keystone)

        # Sync the list widget now that keystone is loaded
        self._sync_batch_list_ui()

        QMessageBox.information(
            self,
            "Imported",
            f"Loaded {len(valid)} video(s).\n\nKeystone: {keystone}",
        )

    def select_csv(self: object) -> object:
        """select_csv method documentation."""
        fp, _ = QFileDialog.getSaveFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if fp:
            self._setup_panel.csv_line.setText(fp)

    # =========================================================================
    # VIDEO PLAYER FUNCTIONS
    # =========================================================================

    def _init_video_player(self, video_path, _probe=None):
        """Initialize video player with the loaded video."""
        self._session_orch._init_video_player(video_path, _probe=_probe)

    def _display_current_frame(self):
        """Display the current frame in the video label."""
        self._session_orch._display_current_frame()

    def _on_timeline_changed(self, value):
        """Handle timeline slider change."""
        self._session_orch._on_timeline_changed(value)

    def _on_timeline_pressed(self):
        """Pause playback before interactive timeline scrubbing."""
        self._session_orch._on_timeline_pressed()

    def _on_timeline_moved(self, value):
        """Update the timeline label while the user scrubs."""
        self._session_orch._on_timeline_moved(value)

    def _goto_first_frame(self):
        """Go to the first frame."""
        self._session_orch._goto_first_frame()

    def _goto_prev_frame(self):
        """Go to the previous frame."""
        self._session_orch._goto_prev_frame()

    def _goto_next_frame(self):
        """Go to the next frame."""
        self._session_orch._goto_next_frame()

    def _goto_last_frame(self):
        """Go to the last frame."""
        self._session_orch._goto_last_frame()

    def _goto_random_frame(self):
        """Jump to a random frame."""
        self._session_orch._goto_random_frame()

    def _toggle_playback(self):
        """Toggle play/pause."""
        self._session_orch._toggle_playback()

    def _start_playback(self):
        """Start video playback."""
        self._session_orch._start_playback()

    def _stop_playback(self):
        """Stop video playback."""
        self._session_orch._stop_playback()

    def _playback_step(self):
        """Advance one frame during playback."""
        self._session_orch._playback_step()

    def _on_frame_range_changed(self):
        """Handle frame range spinbox changes."""
        self._session_orch._on_frame_range_changed(changed_widget=self.sender())

    def _on_trail_history_changed(self):
        """Sync trail-history special values with trajectory overlay visibility."""
        self._session_orch._on_trail_history_changed()

    def _commit_pending_setup_edits(self):
        """Commit typed setup values before saving or launching tracking."""
        self._session_orch._commit_pending_setup_edits()

    def _update_range_info(self):
        """Update the frame range info label."""
        if hasattr(self, "_session_orch"):
            self._session_orch._update_range_info()

    def _set_start_to_current(self):
        """Set start frame to current frame."""
        self._session_orch._set_start_to_current()

    def _set_end_to_current(self):
        """Set end frame to current frame."""
        self._session_orch._set_end_to_current()

    def _reset_frame_range(self):
        """Reset frame range to full video."""
        self._session_orch._reset_frame_range()

    def _update_fps_info(self):
        """Update the FPS info label with time per frame."""
        self._session_orch._update_fps_info()

    def _auto_detect_fps(self, video_path):
        """Auto-detect FPS from video metadata and return the value."""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    logger.info(f"Detected FPS from video: {fps:.2f}")
                    return fps
                else:
                    logger.warning("Could not detect FPS from video metadata")
                    return None
            else:
                logger.warning("Could not open video for FPS detection")
                return None
        except Exception as e:
            logger.error(f"Error detecting FPS: {e}")

    def _auto_set_body_size_from_detection(self):
        """Auto-set reference body size from detected geometric mean."""
        if self.detected_sizes is None:
            return

        recommended_size = self.detected_sizes["recommended_body_size"]
        stats = self.detected_sizes["stats"]
        self._detection_panel.spin_reference_body_size.setValue(recommended_size)

        # Show confirmation with aspect ratio info
        QMessageBox.information(
            self,
            "Body Size Updated",
            f"Reference body size set to {recommended_size:.1f} px\n"
            f"(geometric mean of {self.detected_sizes['count']} detections)\n\n"
            f"Detected dimensions:\n"
            f"  • Major axis: {stats['major']['median']:.1f} px\n"
            f"  • Minor axis: {stats['minor']['median']:.1f} px\n"
            f"  • Aspect ratio: {stats['aspect_ratio']['median']:.2f}\n\n"
            f"All distance/size parameters will now scale relative to this value.",
        )

    def _auto_set_aspect_ratio_from_detection(self):
        """Auto-set reference aspect ratio from detected median."""
        if self.detected_sizes is None:
            return
        recommended_ar = self.detected_sizes["recommended_aspect_ratio"]
        self._detection_panel.spin_reference_aspect_ratio.setValue(recommended_ar)
        QMessageBox.information(
            self,
            "Aspect Ratio Updated",
            f"Reference aspect ratio set to {recommended_ar:.2f}\n"
            f"(median of {self.detected_sizes['count']} detections)\n\n"
            f"Head-tail crop dimensions will adapt to this ratio.\n"
            f"Aspect ratio filtering (if enabled) will use this as the centre.",
        )

    def _validate_yolo_model_requirements(self, params: dict, mode_label: str) -> bool:
        """Validate YOLO mode-specific model requirements before starting runs."""
        return self._tracking_orch._validate_yolo_model_requirements(params, mode_label)

    def _set_preview_test_running(self, running: bool):
        """Lock/unlock UI while async preview detection is running."""
        self._session_orch._set_preview_test_running(running)

    def _on_roi_mode_changed(self, index):
        """Handle ROI mode selection change."""
        self._session_orch._on_roi_mode_changed(index)

    def _on_roi_zone_changed(self, index):
        """Handle ROI zone type selection change."""
        self._session_orch._on_roi_zone_changed(index)

    def _handle_video_mouse_press(self, evt):
        """Handle mouse press on video - either ROI selection or pan/zoom."""
        self._session_orch._handle_video_mouse_press(evt)

    def _handle_video_mouse_move(self, evt):
        """Handle mouse move - update pan if active."""
        self._session_orch._handle_video_mouse_move(evt)

    def _handle_video_mouse_release(self, evt):
        """Handle mouse release - end pan."""
        self._session_orch._handle_video_mouse_release(evt)

    def _handle_video_double_click(self, evt):
        """Handle double-click on video to fit to screen."""
        self._session_orch._handle_video_double_click(evt)

    def _handle_video_wheel(self, evt):
        """Handle mouse wheel - zoom in/out."""
        if hasattr(self, "_session_orch"):
            self._session_orch._handle_video_wheel(evt)

    def _handle_video_event(self, evt):
        """Handle video events including pinch gestures."""
        if hasattr(self, "_session_orch"):
            return self._session_orch._handle_video_event(evt)
        return False

    def _handle_gesture_event(self, evt):
        """Handle pinch-to-zoom gesture."""
        if hasattr(self, "_session_orch"):
            return self._session_orch._handle_gesture_event(evt)
        return False

    def _display_roi_with_zoom(self):
        """Display the ROI base frame with mask and current zoom applied."""
        self._session_orch._display_roi_with_zoom()

    def _fit_image_to_screen(self):
        """Fit the image to the available screen space."""
        self._session_orch._fit_image_to_screen()

    def record_roi_click(self, evt):
        """Record an ROI click from the video label."""
        self._session_orch.record_roi_click(evt)

    def update_roi_preview(self):
        """Render current ROI shapes + in-progress points onto the video label."""
        self._session_orch.update_roi_preview()

    def start_roi_selection(self):
        """Start an ROI shape selection session."""
        self._session_orch.start_roi_selection()

    def finish_roi_selection(self):
        """Finalize the current ROI shape and add it to the shape list."""
        self._session_orch.finish_roi_selection()

    def _generate_combined_roi_mask(self, height, width):
        """Generate a combined mask from all ROI shapes with inclusion/exclusion support."""
        self._session_orch._generate_combined_roi_mask(height, width)

    def undo_last_roi_shape(self):
        """Remove the last added ROI shape."""
        self._session_orch.undo_last_roi_shape()

    def clear_roi(self):
        """Clear all ROI shapes and reset state."""
        self._session_orch.clear_roi()

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        self._session_orch.keyPressEvent(event)

    def _sanitize_model_token(self, text):
        """Sanitize model metadata token for safe filename use."""
        return _sanitize_model_token(text)

    def _format_yolo_model_label(self, model_path):
        """Build combo-box label for a model path, including metadata if available."""
        return self._config_orch._format_yolo_model_label(model_path)

    @staticmethod
    def _yolo_model_matches_filter(metadata, task_family=None, usage_role=None):
        from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator

        return ConfigOrchestrator._yolo_model_matches_filter(
            metadata, task_family=task_family, usage_role=usage_role
        )

    def _populate_yolo_model_combo(
        self,
        combo,
        preferred_model_path=None,
        default_path="",
        include_none=False,
        task_family=None,
        usage_role=None,
        repository_dir=None,
        recursive=False,
    ):
        """Populate a YOLO-model combo with optional metadata role filtering."""
        self._config_orch._populate_yolo_model_combo(
            combo,
            preferred_model_path=preferred_model_path,
            default_path=default_path,
            include_none=include_none,
            task_family=task_family,
            usage_role=usage_role,
            repository_dir=repository_dir,
            recursive=recursive,
        )

    def _set_model_selection_for_selector(self, combo, model_path, default_path=""):
        self._config_orch._set_model_selection_for_selector(
            combo, model_path, default_path
        )

    def _get_selected_model_path_from_selector(self, combo, default_path=""):
        return self._config_orch._get_selected_model_path_from_selector(
            combo, default_path
        )

    def _import_yolo_model_to_repository(
        self, source_path, task_family=None, usage_role=None, repository_dir=None
    ):
        """Import a YOLO model file into the repository with metadata."""
        return self._config_orch._import_yolo_model_to_repository(
            source_path,
            task_family=task_family,
            usage_role=usage_role,
            repository_dir=repository_dir,
        )

    def _populate_pose_model_combo(self, combo, backend, preferred_model_path=None):
        """Populate the pose model combo for the given backend."""
        self._config_orch._populate_pose_model_combo(
            combo, backend, preferred_model_path=preferred_model_path
        )

    def _refresh_pose_model_combo(self, preferred_model_path=None):
        """Refresh the pose model combo for the current backend."""
        self._config_orch._refresh_pose_model_combo(
            preferred_model_path=preferred_model_path
        )

    def on_pose_model_changed(self, index: int) -> None:
        """Handle selection change in the pose model combo."""
        if not hasattr(self, "_identity_panel"):
            return
        selected_data = self._identity_panel.combo_pose_model.itemData(
            index, Qt.UserRole
        )
        if selected_data == "__add_new__":
            self._handle_add_new_pose_model()
            return
        path = selected_data if selected_data and selected_data != "__none__" else ""
        self._set_pose_model_path_for_backend(
            path, backend=self._current_pose_backend_key(), update_combo=False
        )

    def _get_selected_yolo_model_path(self) -> object:
        """Return currently selected direct OBB model path."""
        if not hasattr(self, "_detection_panel"):
            return ""
        return self._get_selected_model_path_from_selector(
            self._detection_panel.combo_yolo_model,
            default_path="",
        )

    def _get_selected_yolo_detect_model_path(self) -> object:
        return self._get_selected_model_path_from_selector(
            self._detection_panel.combo_yolo_detect_model,
            default_path="",
        )

    def _get_selected_yolo_crop_obb_model_path(self) -> object:
        return self._get_selected_model_path_from_selector(
            self._detection_panel.combo_yolo_crop_obb_model,
            default_path="",
        )

    def _set_yolo_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._detection_panel.combo_yolo_model,
            model_path,
        )

    def _set_yolo_detect_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._detection_panel.combo_yolo_detect_model,
            model_path,
        )

    def _set_yolo_crop_obb_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._detection_panel.combo_yolo_crop_obb_model,
            model_path,
        )

    def _set_yolo_headtail_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._identity_panel.combo_yolo_headtail_model,
            model_path,
        )

    def _update_obb_mode_warning(self) -> None:
        """Show a performance hint when device/mode is a suboptimal combination."""
        if hasattr(self, "_session_orch"):
            self._session_orch._update_obb_mode_warning()

    def _apply_crop_obb_training_params(self):
        """Backward-compatible alias: apply seq_crop_obb training metadata."""
        self._auto_apply_yolo_training_params("seq_crop_obb")

    def _auto_apply_yolo_training_params(self, role: str) -> None:
        """Read model registry metadata for *role* and populate inference UI.

        DetectKit publishes ``training_params`` on every successful run so
        TrackerKit can recover the values it would otherwise need the user
        to retype. This applies them on selection (including programmatic
        selection right after import).
        """
        role = (role or "").strip().lower()
        if role == "seq_crop_obb":
            model_path = self._get_selected_yolo_crop_obb_model_path()
        elif role == "seq_detect":
            model_path = self._get_selected_yolo_detect_model_path()
        elif role == "obb_direct":
            model_path = self._get_selected_yolo_model_path()
        else:
            return
        if not model_path:
            return

        # Switch OBB mode to match the role of the model the user picked.
        panel = self._detection_panel
        try:
            if role in ("seq_detect", "seq_crop_obb"):
                if panel.combo_yolo_obb_mode.currentIndex() != 1:
                    panel.combo_yolo_obb_mode.setCurrentIndex(1)
            elif role == "obb_direct":
                if panel.combo_yolo_obb_mode.currentIndex() != 0:
                    panel.combo_yolo_obb_mode.setCurrentIndex(0)
        except Exception:
            pass

        meta = get_yolo_model_metadata(model_path) or {}
        tp = meta.get("training_params")
        if not isinstance(tp, dict):
            return

        applied: list[str] = []

        if role == "seq_crop_obb":
            if "imgsz" in tp:
                val = int(tp["imgsz"])
                panel.spin_yolo_seq_stage2_imgsz.setValue(val)
                applied.append(f"stage2_imgsz={val}")
            if "crop_pad_ratio" in tp:
                val = float(tp["crop_pad_ratio"])
                panel.spin_yolo_seq_crop_pad.setValue(val)
                applied.append(f"crop_pad={val}")
            if "min_crop_size_px" in tp:
                val = int(tp["min_crop_size_px"])
                panel.spin_yolo_seq_min_crop_px.setValue(val)
                applied.append(f"min_crop={val}")
            if "enforce_square" in tp:
                val = bool(tp["enforce_square"])
                panel.chk_yolo_seq_square_crop.setChecked(val)
                applied.append(f"square={val}")
        elif role == "seq_detect":
            # Stage-1 imgsz is auto-recovered by Ultralytics from the .pt;
            # apply matching crop derivation params so stage-1 and stage-2
            # share the same crop policy if the detect model recorded one.
            if "crop_pad_ratio" in tp:
                val = float(tp["crop_pad_ratio"])
                panel.spin_yolo_seq_crop_pad.setValue(val)
                applied.append(f"crop_pad={val}")
            if "min_crop_size_px" in tp:
                val = int(tp["min_crop_size_px"])
                panel.spin_yolo_seq_min_crop_px.setValue(val)
                applied.append(f"min_crop={val}")
            if "enforce_square" in tp:
                val = bool(tp["enforce_square"])
                panel.chk_yolo_seq_square_crop.setChecked(val)
                applied.append(f"square={val}")
        elif role == "obb_direct":
            # Direct OBB has no per-field UI for imgsz; metadata is logged
            # so users see what the model expects. The Ultralytics runtime
            # picks up the trained imgsz from the .pt automatically.
            if "imgsz" in tp:
                applied.append(f"trained_imgsz={int(tp['imgsz'])}")

        if applied:
            logger.info(
                "Auto-configured %s params from model metadata: %s",
                role,
                ", ".join(applied),
            )
            if hasattr(self, "statusBar"):
                try:
                    self.statusBar().showMessage(
                        f"Applied {role} model defaults: {', '.join(applied)}",
                        4000,
                    )
                except Exception:
                    pass

    def on_yolo_headtail_model_changed(self: object, index: object) -> object:
        """Handle head/tail classification model combo-box changes.

        Handles three cases:
        1. ``__add_new__`` sentinel — opens the add-model file dialog.
        2. Discovered (unannotated) classkit entry — opens the annotation dialog.
        3. Normal registered entry — syncs analysis mode UI.
        """
        combo = self._identity_panel.combo_yolo_headtail_model
        rel_path = combo.itemData(index, Qt.UserRole)
        if rel_path == "__add_new__":
            self._handle_add_new_headtail_model()
            return
        if rel_path and self._headtail_is_discovered_entry(rel_path):
            self._annotate_discovered_headtail_model(rel_path)
            return
        if hasattr(self, "_detection_panel"):
            self._detection_panel._on_yolo_mode_changed(index)
        self._sync_individual_analysis_mode_ui()

    def _headtail_is_discovered_entry(self, rel_path: str) -> bool:
        """Return True if rel_path is not yet annotated as head_tail in the registry."""
        from hydra_suite.training.model_publish import iter_registry_entries

        normalized_path = str(rel_path or "").strip()
        if not normalized_path or normalized_path in {"__add_new__", "__none__"}:
            return False

        return all(
            not (
                key == normalized_path
                and str(meta.get("usage_role") or "").strip().lower()
                in _HEADTAIL_USAGE_ROLE_ALIASES
            )
            for key, meta in iter_registry_entries()
        )

    def _handle_add_new_headtail_model(self) -> None:
        """Import a head-tail classifier through the dedicated validation dialog."""
        from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

        from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
            HeadTailImportDialog,
            describe_headtail_candidate,
            import_headtail_candidate,
        )

        combo = self._identity_panel.combo_yolo_headtail_model
        prev_data = combo.currentData(Qt.UserRole)
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Head-Tail Classifier",
            get_yolo_model_repository_directory(
                task_family="classify", usage_role="headtail"
            ),
            "Head-Tail Classifier Files (*.pth *.pt *.multihead.json);;All Files (*)",
        )
        if not src_path:
            idx = combo.findData(prev_data)
            combo.setCurrentIndex(max(idx, 0))
            return

        try:
            summary = describe_headtail_candidate(src_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Cannot read head-tail model:\n{exc}"
            )
            idx = combo.findData(prev_data)
            combo.setCurrentIndex(max(idx, 0))
            return

        dlg = HeadTailImportDialog(summary, parent=self)
        if dlg.exec() != QDialog.Accepted:
            idx = combo.findData(prev_data)
            combo.setCurrentIndex(max(idx, 0))
            self._identity_panel._refresh_yolo_headtail_model_combo()
            return

        try:
            rel_path = import_headtail_candidate(
                model_path=src_path,
                species=dlg.species(),
                description=dlg.description(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Could not import model:\n{exc}"
            )
            idx = combo.findData(prev_data)
            combo.setCurrentIndex(max(idx, 0))
            return

        self._identity_panel._refresh_yolo_headtail_model_combo()
        idx = combo.findData(rel_path)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self._sync_individual_analysis_mode_ui()

    def _annotate_discovered_headtail_model(self, rel_path: str) -> None:
        """Open the head-tail import dialog for a discovered classkit-published model."""
        from PySide6.QtWidgets import QDialog, QMessageBox

        from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
            HeadTailImportDialog,
            describe_headtail_candidate,
            import_headtail_candidate,
        )

        abs_path = os.path.join(get_models_root_directory(), rel_path)
        try:
            summary = describe_headtail_candidate(abs_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Cannot read head-tail model: {exc}"
            )
            return

        dlg = HeadTailImportDialog(summary, parent=self)
        if dlg.exec() != QDialog.Accepted:
            self._identity_panel._refresh_yolo_headtail_model_combo()
            return
        rel_path = import_headtail_candidate(
            model_path=abs_path,
            species=dlg.species(),
            description=dlg.description(),
        )
        self._identity_panel._refresh_yolo_headtail_model_combo()
        combo = self._identity_panel.combo_yolo_headtail_model
        idx = combo.findData(rel_path)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self._sync_individual_analysis_mode_ui()

    def _handle_add_new_yolo_model(
        self,
        combo,
        refresh_callback,
        selection_callback,
        task_family,
        usage_role,
        dialog_title,
        repository_dir=None,
    ):
        """Browse for a model, import it, refresh the combo, and select it."""
        self._config_orch._handle_add_new_yolo_model(
            combo,
            refresh_callback,
            selection_callback,
            task_family,
            usage_role,
            dialog_title,
            repository_dir=repository_dir,
        )

    def _handle_remove_selected_yolo_model(
        self,
        combo,
        refresh_callback,
        selection_callback,
        *,
        model_kind: str = "model",
    ) -> None:
        """Remove the selected repository-backed model from a TrackerKit selector."""
        self._config_orch._handle_remove_selected_yolo_model(
            combo,
            refresh_callback,
            selection_callback,
            model_kind=model_kind,
        )

    def _handle_remove_selected_pose_model(self) -> None:
        """Remove the selected pose model from the local repository."""
        self._config_orch._handle_remove_selected_pose_model()

    def _confirm_and_remove_repository_model(
        self, model_path: object, *, model_kind: str = "model"
    ) -> bool:
        """Confirm and remove a specific repository-backed model path."""
        return self._config_orch._confirm_and_remove_repository_model(
            model_path,
            model_kind=model_kind,
        )

    def toggle_preview(self, checked):
        """Toggle preview mode on/off."""
        self._session_orch.toggle_preview(checked)

    def toggle_tracking(self, checked):
        """Toggle full tracking on/off."""
        self._session_orch.toggle_tracking(checked)

    def toggle_debug_logging(self, checked):
        """Toggle debug logging level."""
        if checked:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug logging enabled")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging disabled")

    def _on_visualization_mode_changed(self, state):
        """Handle visualization-free mode toggle."""
        self._session_orch._on_visualization_mode_changed(state)

    def start_full(self: object) -> object:
        """start_full method documentation."""
        self._tracking_orch.start_full()

    def _request_qthread_stop(
        self,
        worker,
        worker_name: str,
        *,
        timeout_ms: int = 1500,
        force_terminate: bool = True,
    ) -> None:
        """Stop a QThread cooperatively, then force terminate if needed."""
        self._tracking_orch._request_qthread_stop(
            worker, worker_name, timeout_ms=timeout_ms, force_terminate=force_terminate
        )

    def _stop_csv_writer(self, timeout_sec: float = 2.0) -> None:
        """Stop background CSV writer thread safely without indefinite blocking."""
        self._tracking_orch._stop_csv_writer(timeout_sec)

    def _cleanup_thread_reference(self, attr_name: str) -> None:
        """Delete finished QThread references safely."""
        self._tracking_orch._cleanup_thread_reference(attr_name)

    def stop_tracking(self: object) -> object:
        """stop_tracking method documentation."""
        self._tracking_orch.stop_tracking()

    def _set_ui_controls_enabled(self, enabled: bool):
        self._session_orch._set_ui_controls_enabled(enabled)

    def _collect_preview_controls(self):
        return self._session_orch._collect_preview_controls()

    def _set_interactive_widgets_enabled(
        self,
        enabled: bool,
        allowlist=None,
        blocklist=None,
        remember_state: bool = True,
    ):
        self._session_orch._set_interactive_widgets_enabled(
            enabled,
            allowlist=allowlist,
            blocklist=blocklist,
            remember_state=remember_state,
        )

    def _set_video_interaction_enabled(self, enabled: bool):
        self._session_orch._set_video_interaction_enabled(enabled)

    def _prepare_tracking_display(self):
        self._session_orch._prepare_tracking_display()

    def _show_video_logo_placeholder(self):
        """Show HYDRA logo in the video panel when no video is loaded."""
        if hasattr(self, "_session_orch"):
            self._session_orch._show_video_logo_placeholder()
            return
        # Fallback during early init before orchestrators exist
        self._set_video_message("HYDRA\n\nLoad a video to begin...")

    def _set_video_pixmap(self, pixmap: QPixmap):
        """Display a pixmap and resize the video label to match it."""
        self.video_label.setText("")
        self.video_label.setPixmap(pixmap)
        if pixmap is not None and not pixmap.isNull():
            self.video_label.resize(pixmap.width(), pixmap.height())

    def _set_video_message(
        self, text: str, minimum_width: int = 320, minimum_height: int = 240
    ):
        """Display centered text in the video area with a stable minimum size."""
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(text)
        width_hint = self.video_label.sizeHint().width()
        height_hint = self.video_label.sizeHint().height()
        self.video_label.resize(
            max(minimum_width, width_hint),
            max(minimum_height, height_hint),
        )

    def _is_visualization_enabled(self) -> bool:
        # Preview should always render frames regardless of visualization-free toggle
        return (
            not self._setup_panel.chk_visualization_free.isChecked()
            or self.btn_preview.isChecked()
        )

    def _sync_contextual_controls(self):
        self._session_orch._sync_contextual_controls()

    def _apply_ui_state(self, state: str):
        self._session_orch._apply_ui_state(state)

    def _draw_roi_overlay(self, qimage):
        """Draw ROI shapes overlay on a QImage."""
        return self._session_orch._draw_roi_overlay(qimage)

    def _apply_roi_mask_to_image(self, qimage):
        """Apply ROI visualization - draw boundary overlay for all detection methods."""
        return self._session_orch._apply_roi_mask_to_image(qimage)

    @Slot(int, str)
    def on_progress_update(
        self: object, percentage: object, status_text: object
    ) -> object:
        """on_progress_update method documentation."""
        self._tracking_orch.on_progress_update(percentage, status_text)

    def on_pose_exported_model_resolved(self, artifact_path: str) -> None:
        """Update pose exported-model UI/config when runtime resolves an artifact path."""
        self._tracking_orch.on_pose_exported_model_resolved(artifact_path)

    def on_tracking_warning(self: object, title: object, message: object) -> object:
        """Display tracking warnings in the UI."""
        self._tracking_orch.on_tracking_warning(title, message)

    def show_gpu_info(self: object) -> object:
        """Display GPU and acceleration information dialog."""
        self._tracking_orch.show_gpu_info()

    def _clear_detection_caches(self) -> None:
        """Delete cached detection artifacts for the active video."""
        self._tracking_orch.clear_detection_caches()

    def on_stats_update(self: object, stats: object) -> object:
        """Update real-time tracking statistics."""
        self._tracking_orch.on_stats_update(stats)

    def on_new_frame(self: object, rgb: object) -> object:
        """on_new_frame method documentation."""
        self._tracking_orch.on_new_frame(rgb)

    def _scale_trajectories_to_original_space(self, trajectories_df, resize_factor):
        """Scale trajectory coordinates from resized space back to original video space."""
        return self._tracking_orch._scale_trajectories_to_original_space(
            trajectories_df, resize_factor
        )

    def save_trajectories_to_csv(
        self: object, trajectories: object, output_path: object
    ) -> object:
        """Save processed trajectories to CSV.

        Args:
            trajectories: Either list of tuples (old format) or pandas DataFrame (new format with confidence)
            output_path: Path to save CSV file
        """
        self._tracking_orch.save_trajectories_to_csv(trajectories, output_path)

    def merge_and_save_trajectories(self: object) -> object:
        """merge_and_save_trajectories method documentation."""
        self._tracking_orch.merge_and_save_trajectories()

    def on_merge_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during merge."""
        self._tracking_orch.on_merge_progress(value, message)

    def _store_interpolated_pose_result(self, pose_csv_path, pose_rows):
        """Store interpolated pose results from CSV path or in-memory rows."""
        self._tracking_orch._store_interpolated_pose_result(pose_csv_path, pose_rows)

    def _store_interpolated_tag_result(self, tag_csv_path, tag_rows):
        """Store interpolated AprilTag results from CSV path or in-memory rows."""
        self._tracking_orch._store_interpolated_tag_result(tag_csv_path, tag_rows)

    def _store_interpolated_cnn_result(self, cnn_csv_paths, cnn_rows):
        """Store interpolated CNN identity results from CSV paths or in-memory rows."""
        self._tracking_orch._store_interpolated_cnn_result(cnn_csv_paths, cnn_rows)

    def _store_interpolated_headtail_result(self, headtail_csv_path, headtail_rows):
        """Store interpolated head-tail results from CSV path or in-memory rows."""
        self._tracking_orch._store_interpolated_headtail_result(
            headtail_csv_path, headtail_rows
        )

    def _on_interpolated_crops_finished(self, result):
        self._tracking_orch._on_interpolated_crops_finished(result)

    def _resolve_source_video_fps(self) -> float:
        """Return the source video FPS, falling back to the UI value."""
        fps = 0.0
        video_path = (
            self._setup_panel.file_line.text().strip()
            if hasattr(self, "_setup_panel")
            else ""
        )
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            try:
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            finally:
                cap.release()
        if fps <= 0.0 and hasattr(self, "_setup_panel"):
            fps = float(self._setup_panel.spin_fps.value() or 0.0)
        return max(1.0, fps or 1.0)

    def _resolve_current_individual_dataset_dir(self):
        """Resolve the active per-session individual dataset directory."""
        params = self.get_parameters_dict()
        dataset_dir = resolve_individual_dataset_dir(
            params.get("INDIVIDUAL_DATASET_OUTPUT_DIR"),
            params.get("INDIVIDUAL_DATASET_NAME"),
            self._individual_dataset_run_id,
        )
        if dataset_dir is None:
            return None
        return Path(dataset_dir).expanduser()

    def _resolve_current_final_media_video_dir(self):
        """Resolve the active per-session final-media video output directory."""
        from hydra_suite.core.identity.dataset.oriented_video import (
            resolve_oriented_track_video_dir,
        )

        params = self.get_parameters_dict()
        output_dir = resolve_oriented_track_video_dir(
            params.get("FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR")
            or params.get("ORIENTED_TRACK_VIDEO_OUTPUT_DIR"),
            self._individual_dataset_run_id,
        )
        if output_dir is None:
            return None
        return Path(output_dir).expanduser()

    def _generate_final_media_export(self, final_csv_path):
        """Export orientation-fixed videos for final trajectories."""
        return self._tracking_orch._generate_final_media_export(final_csv_path)

    def _start_pending_final_media_export(self, final_csv_path) -> bool:
        """Start optional oriented track video export and hold the finish pipeline."""
        return self._tracking_orch._start_pending_final_media_export(final_csv_path)

    def _on_final_media_export_worker_thread_finished(self):
        """Release completed oriented track video worker safely."""
        self._tracking_orch._on_final_media_export_worker_thread_finished()

    def _on_final_media_export_finished(self, result):
        """Handle completion of oriented track video export."""
        self._tracking_orch._on_final_media_export_finished(result)

    def _on_final_media_export_error(self, error_message):
        """Handle oriented track video export errors without aborting the session."""
        self._tracking_orch._on_final_media_export_error(error_message)

    def on_merge_error(self: object, error_message: object) -> object:
        """Handle merge errors."""
        self._tracking_orch.on_merge_error(error_message)

    def on_merge_finished(self: object, resolved_trajectories: object) -> object:
        """Handle completion of trajectory merging."""
        self._tracking_orch.on_merge_finished(resolved_trajectories)

    def _generate_video_from_trajectories(
        self, trajectories_df, csv_path=None, finalize_on_complete=True
    ):
        """
        Generate annotated video from post-processed trajectories.

        Args:
            trajectories_df: DataFrame with merged/interpolated trajectories
            csv_path: Path to the CSV file (optional, for logging)
            finalize_on_complete: If True, continue full finish pipeline after render.
        """
        self._tracking_orch._generate_video_from_trajectories(
            trajectories_df, csv_path, finalize_on_complete
        )

    def on_tracking_finished(
        self: object, finished_normally: object, fps_list: object, full_traj: object
    ) -> object:
        """on_tracking_finished method documentation."""
        self._tracking_orch.on_tracking_finished(finished_normally, fps_list, full_traj)

    def _is_pose_export_enabled(self) -> bool:
        """Return True when pose extraction export should be produced."""
        if not hasattr(self, "_detection_panel"):
            return False
        return self._detection_panel._is_yolo_detection_mode() and bool(
            hasattr(self, "_identity_panel")
            and self._identity_panel.chk_enable_pose_extractor.isChecked()
        )

    def _build_pose_augmented_dataframe(self, final_csv_path):
        """Load final CSV and merge available cached/interpolated pose columns."""
        return self._tracking_orch._build_pose_augmented_dataframe(final_csv_path)

    def _export_pose_augmented_csv(self, final_csv_path):
        """Write a pose-augmented trajectories CSV next to the final CSV."""
        self._tracking_orch._export_pose_augmented_csv(final_csv_path)

    def _relink_final_pose_augmented_csv(self, final_csv_path):
        """Rewrite final CSV IDs after pose-aware relinking and regenerate the rich export CSV."""
        self._tracking_orch._relink_final_pose_augmented_csv(final_csv_path)

    def _load_video_trajectories(self, final_csv_path):
        """Load best available trajectories for video generation (prefers pose-augmented CSV)."""
        return self._tracking_orch._load_video_trajectories(final_csv_path)

    def _run_pending_video_generation_or_finalize(self):
        """Run video generation if queued; otherwise finalize UI/session cleanup."""
        self._tracking_orch._run_pending_video_generation_or_finalize()

    def _finish_tracking_session(self, final_csv_path=None):
        """Complete tracking session cleanup and UI updates."""
        self._tracking_orch._finish_tracking_session(final_csv_path=final_csv_path)

    def _finalize_tracking_session_ui(self):
        """Finalize session cleanup and return UI to idle state."""
        self._tracking_orch._finalize_tracking_session_ui()

    def _generate_interpolated_individual_crops(self, csv_path):
        """Post-pass interpolation for occluded segments in individual dataset."""
        self._tracking_orch._generate_interpolated_individual_crops(csv_path)

    def _interp_angle(self, theta_start, theta_end, t):
        deg0 = math.degrees(theta_start)
        deg1 = math.degrees(theta_end)
        # Treat OBB angles as 180-degree periodic; pick shortest path.
        candidates = (deg1, deg1 + 180.0, deg1 - 180.0)
        best_delta = None
        for cand in candidates:
            delta = wrap_angle_degs(cand - deg0)
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
        return math.radians(deg0 + (best_delta or 0.0) * t)

    def start_backward_tracking(self: object) -> object:
        """start_backward_tracking method documentation."""
        self._tracking_orch.start_backward_tracking()

    def start_tracking(
        self: object, preview_mode: bool, backward_mode: bool = False
    ) -> object:
        """start_tracking method documentation."""
        self._tracking_orch.start_tracking(
            preview_mode=preview_mode, backward_mode=backward_mode
        )

    def start_preview_on_video(self: object, video_path: object) -> object:
        """start_preview_on_video method documentation."""
        self._tracking_orch.start_preview_on_video(video_path)

    def start_tracking_on_video(
        self: object, video_path: object, backward_mode: object = False
    ) -> object:
        """start_tracking_on_video method documentation."""
        self._tracking_orch.start_tracking_on_video(
            video_path, backward_mode=backward_mode
        )

    def get_parameters_dict(self: object) -> object:
        """get_parameters_dict method documentation."""
        return self._config_orch.get_parameters_dict()

    def load_config(self: object) -> object:
        """Manually load config from file dialog."""
        self._config_orch.load_config()

    def _load_config_from_file(self, config_path, preset_mode=False):
        """Internal method to load config from a specific file path.

        This method supports both new standardized key names and legacy key names
        for backward compatibility with older config files.

        Args:
            config_path: Path to the config/preset file
            preset_mode: If True, skip loading video paths and ROI data (for organism presets)
        """
        self._config_orch._load_config_from_file(config_path, preset_mode=preset_mode)

    def _atomic_json_write(self, cfg, path):
        """Write a JSON config atomically. Returns (success, error_message)."""
        return self._config_orch._atomic_json_write(cfg, path)

    def _resolve_config_save_path(self, prompt_if_exists):
        """Determine the config file save path, prompting the user if needed."""
        return self._config_orch._resolve_config_save_path(prompt_if_exists)

    def save_config(
        self: object,
        preset_mode: object = False,
        preset_path: object = None,
        preset_name: object = None,
        preset_description: object = None,
        prompt_if_exists: bool = True,
    ) -> object:
        """Save current configuration to JSON file.

        Args:
            preset_mode: If True, skip video paths, device settings, and ROI data (for organism presets)
            preset_path: If provided, save directly to this path without prompting
            preset_name: Name for the preset (only used in preset_mode)
            preset_description: Description for the preset (only used in preset_mode)
            prompt_if_exists: If False, overwrite default config path without interactive replace dialog.

        Returns:
            bool: True if config was saved successfully, False if cancelled or failed
        """
        return self._config_orch.save_config(
            preset_mode=preset_mode,
            preset_path=preset_path,
            preset_name=preset_name,
            preset_description=preset_description,
            prompt_if_exists=prompt_if_exists,
        )

    def _setup_session_logging(self, video_path, backward_mode=False):
        self._session_orch._setup_session_logging(
            video_path, backward_mode=backward_mode
        )

    def _cleanup_session_logging(self):
        self._session_orch._cleanup_session_logging()

    def _generate_training_dataset(self, override_csv_path=None):
        """Generate training dataset from tracking results for active learning."""
        self._tracking_orch._generate_training_dataset(
            override_csv_path=override_csv_path
        )

    def on_dataset_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during dataset generation."""
        self._tracking_orch.on_dataset_progress(value, message)

    def on_dataset_finished(
        self: object, dataset_dir: object, num_frames: object
    ) -> object:
        """Handle dataset generation completion."""
        self._tracking_orch.on_dataset_finished(dataset_dir, num_frames)

    def on_dataset_error(self: object, error_message: object) -> object:
        """Handle dataset generation errors."""
        self._tracking_orch.on_dataset_error(error_message)

    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        self._tracking_orch._show_session_summary()

    def _open_refinekit(self):
        """Launch RefineKit as a subprocess pointing at the current video."""
        import subprocess
        import sys

        video_path = self.current_video_path
        if not video_path:
            QMessageBox.warning(
                self,
                "No Video",
                "No video is currently loaded. Please load a video first.",
            )
            return
        cmd = [sys.executable, "-m", "hydra_suite.refinekit.app", str(video_path)]
        logger.info("Launching RefineKit: %s", " ".join(cmd))
        subprocess.Popen(cmd)

    def _on_dataset_worker_thread_finished(self):
        """Release completed dataset worker safely."""
        self._tracking_orch._on_dataset_worker_thread_finished()

    def _is_worker_running(self, worker):
        return self._session_orch._is_worker_running(worker)

    def _has_active_progress_task(self) -> bool:
        return self._session_orch._has_active_progress_task()

    def _refresh_progress_visibility(self):
        self._session_orch._refresh_progress_visibility()

    def _cleanup_temporary_files(self):
        self._session_orch._cleanup_temporary_files()

    def _disable_spinbox_wheel_events(self):
        self._session_orch._disable_spinbox_wheel_events()

    def _connect_parameter_signals(self):
        self._session_orch._connect_parameter_signals()

    @Slot()
    def _on_parameter_changed(self):
        params = self.get_parameters_dict()
        self.parameters_changed.emit(params)

    def _create_help_label(self, text, attach_to_title=True):
        """Create a styled help label for section guidance."""
        label = CompactHelpLabel(text, attach_to_title=attach_to_title)
        return label

    def _invalidate_roi_cache(self):
        """Invalidate ROI display cache when ROI changes."""
        self._config_orch._invalidate_roi_cache()

    def _get_presets_dir(self):
        """Get the presets directory path."""
        from hydra_suite.paths import get_presets_dir

        return str(get_presets_dir())

    def _populate_preset_combo(self):
        """Populate the preset combo box by auto-scanning configs folder."""
        self._config_orch._populate_preset_combo()

    def _load_selected_preset(self):
        """Load the currently selected preset."""
        self._config_orch._load_selected_preset()

    def _save_custom_preset(self):
        """Save current settings as custom preset with user-defined name and description."""
        self._config_orch._save_custom_preset()

    def _load_default_preset_on_startup(self):
        """Load default preset on application startup."""
        self._config_orch._load_default_preset_on_startup()

    def _load_advanced_config(self):
        """Load advanced configuration for power users."""
        if hasattr(self, "_config_orch"):
            return self._config_orch._load_advanced_config()
        # Fallback during __init__ before orchestrators exist
        from hydra_suite.paths import get_advanced_config_path

        config_path = str(get_advanced_config_path())

        default_config = {
            "roi_crop_warning_threshold": 0.6,  # Warn if ROI is <60% of frame
            "roi_crop_auto_suggest": True,  # Auto-suggest cropping
            "roi_crop_remind_every_session": False,  # Remind every time or once
            "roi_crop_padding_fraction": 0.05,  # Padding as fraction of min(width, height) - typically 5%
            "video_crop_codec": "libx264",  # Codec for cropped videos (libx264 for quality)
            "video_crop_crf": 18,  # CRF quality (lower = better, 18 = visually lossless)
            "video_crop_preset": "medium",  # ffmpeg preset (ultrafast, fast, medium, slow, veryslow)
            # YOLO Batching - Memory Fractions (device-specific optimization)
            "mps_memory_fraction": 0.3,  # Conservative 30% of unified memory for MPS (Apple Silicon)
            "cuda_memory_fraction": 0.7,  # 70% of VRAM for CUDA (NVIDIA GPUs)
            "tensorrt_build_workspace_gb": 4.0,  # TensorRT builder workspace limit in GB
            "tensorrt_build_batch_size": None,  # Optional fixed TensorRT build batch override
            # Dataset Generation - YOLO Detection Parameters (separate from tracking)
            "dataset_yolo_confidence_threshold": 0.05,  # Very low - detect all animals including uncertain ones for annotation
            "dataset_yolo_iou_threshold": 0.5,  # Moderate - remove obvious duplicates but keep borderline cases for manual review
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded advanced config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load advanced config: {e}")
        else:
            # Auto-create advanced_config.json with defaults on first run
            # This makes the file discoverable for power users who want to customize
            try:
                config_dir = os.path.dirname(config_path)
                os.makedirs(config_dir, exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default advanced config at {config_path}")
            except Exception as e:
                logger.warning(f"Could not create advanced config file: {e}")

        return default_config

    def _save_advanced_config(self):
        """Save advanced configuration."""
        self._config_orch._save_advanced_config()

    def _calculate_roi_bounding_box(self, padding=None):
        """Calculate the bounding box of the current ROI mask with optional padding.

        Args:
            padding: Fraction of min(width, height) to add as padding (e.g., 0.05 = 5%).
                    If None, uses value from advanced config.

        Returns:
            Tuple (x, y, w, h) or None if no ROI
        """
        return self._config_orch._calculate_roi_bounding_box(padding=padding)

    def _estimate_roi_efficiency(self):
        """Estimate the efficiency gain from cropping to ROI.

        Returns:
            tuple: (roi_coverage_percent, potential_speedup_factor) or (None, None)
        """
        return self._config_orch._estimate_roi_efficiency()

    def _update_roi_optimization_info(self):
        """Update the ROI optimization label with efficiency information."""
        self._config_orch._update_roi_optimization_info()

    def crop_video_to_roi(self: object) -> object:
        """Crop the video to the ROI bounding box and save as new file."""
        self._config_orch.crop_video_to_roi()

    def _poll_crop_stderr_progress(self, process):
        """Read ffmpeg stderr progress (delegated to config orchestrator)."""
        self._config_orch._poll_crop_stderr_progress(process)

    def _load_cropped_video(self, output_path):
        """Set up the UI to use the newly cropped video."""
        self._config_orch._load_cropped_video(output_path)

    def _handle_crop_success(self, output_path, orig_w, orig_h, crop_w, crop_h):
        """Handle a successful crop completion."""
        self._config_orch._handle_crop_success(
            output_path, orig_w, orig_h, crop_w, crop_h
        )

    def _handle_crop_failure(self, return_code):
        """Handle a failed crop completion."""
        self._config_orch._handle_crop_failure(return_code)

    def _check_crop_completion(self):
        """Check if background crop process has completed."""
        self._config_orch._check_crop_completion()
