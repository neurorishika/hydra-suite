"""SetupPanel — preset selection, video files, display, and ROI configuration."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class SetupPanel(QWidget):
    """Preset picker, video/batch file selection, ROI, and display options."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._build_ui()

    def _build_ui(self) -> None:
        """Tab 1: Setup - Files, Video, Display & Debug."""
        layout = self._layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(8)
        self._main_window._set_compact_scroll_layout(form)

        # ============================================================
        # Preset Selector
        # ============================================================
        g_presets = QGroupBox("Presets")
        self._main_window._set_compact_section_widget(g_presets)
        vl_presets = QVBoxLayout(g_presets)
        vl_presets.setSpacing(6)
        vl_presets.addWidget(
            self._main_window._create_help_label(
                "Load optimized default values for different model organisms. Video-specific configs override presets."
            )
        )

        preset_layout = QHBoxLayout()
        preset_label = QLabel("Organism preset")
        preset_label.setStyleSheet("font-weight: bold;")

        self.combo_presets = QComboBox()
        self.combo_presets.setToolTip(
            "Select preset optimized for your organism.\n"
            "Custom: Your personal saved defaults (if exists)"
        )
        # NOTE: _populate_preset_combo() is called after panel construction in main_window.py

        self.btn_load_preset = QPushButton("Load Preset")
        self.btn_load_preset.clicked.connect(self._main_window._load_selected_preset)
        self.btn_load_preset.setToolTip("Apply selected preset to all parameters")

        self.btn_save_custom = QPushButton("Save as Custom")
        self.btn_save_custom.clicked.connect(self._main_window._save_custom_preset)
        self.btn_save_custom.setToolTip("Save current settings as your custom defaults")

        self.preset_status_label = QLabel("")
        self.preset_status_label.setStyleSheet(
            "color: #6a6a6a; font-style: italic; font-size: 10px;"
        )
        self.preset_status_label.setVisible(False)

        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.combo_presets, stretch=1)
        preset_layout.addWidget(self.btn_load_preset)
        preset_layout.addWidget(self.btn_save_custom)

        vl_presets.addLayout(preset_layout)
        vl_presets.addWidget(self.preset_status_label)

        # Description display
        self.preset_description_label = QLabel("")
        self.preset_description_label.setWordWrap(True)
        self.preset_description_label.setStyleSheet(
            "color: #9a9a9a; font-style: italic; font-size: 10px; padding: 5px; "
            "background-color: #252526; border-radius: 3px;"
        )
        self.preset_description_label.setVisible(False)
        vl_presets.addWidget(self.preset_description_label)

        # Connect combo box to show description
        self.combo_presets.currentIndexChanged.connect(
            self._on_preset_selection_changed
        )

        form.addWidget(g_presets)

        # ============================================================
        # Video Setup (File Management + Frame Rate)
        # ============================================================
        g_files = QGroupBox("Video")
        self._main_window._set_compact_section_widget(g_files)
        vl_files = QVBoxLayout(g_files)
        vl_files.setSpacing(6)
        vl_files.addWidget(
            self._main_window._create_help_label(
                "Select your input video and output locations. Configuration is auto-saved per video - "
                "next time you load the same video, your settings will be restored automatically."
            )
        )
        fl = QFormLayout(None)
        fl.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fl.setHorizontalSpacing(10)
        fl.setVerticalSpacing(8)

        self.btn_file = QPushButton("Browse...")
        self.btn_file.clicked.connect(self._main_window.select_file)
        self.btn_file.setObjectName("SecondaryBtn")
        self.btn_file.setFixedHeight(30)
        self.file_line = QLineEdit()
        self.file_line.setPlaceholderText("path/to/video.mp4")
        self.file_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.file_line.setFixedHeight(30)
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        file_row.addWidget(self.file_line, 1)
        file_row.addWidget(self.btn_file)
        fl.addRow("Input video", file_row)

        # Acquisition controls on one compact row
        acquisition_row = QHBoxLayout()
        acquisition_row.setSpacing(8)
        fps_caption = QLabel("FPS")
        fps_caption.setStyleSheet("font-size: 10px; font-weight: 600; color: #bdbdbd;")
        acquisition_row.addWidget(fps_caption)

        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 240.0)
        self.spin_fps.setSingleStep(1.0)
        self.spin_fps.setValue(30.0)
        self.spin_fps.setDecimals(2)
        self.spin_fps.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_fps.setFixedHeight(30)
        self.spin_fps.setMinimumWidth(92)
        self.spin_fps.setToolTip(
            "Acquisition frame rate (frames per second) at which the video was recorded.\n"
            "NOTE: This may differ from the video file's playback framerate.\n"
            "Use 'Detect from Video' to read from file metadata as a starting point.\n"
            "Time-dependent parameters (velocity, durations) scale with this.\n"
            "Affects: motion prediction, track lifecycle, velocity thresholds."
        )
        self.spin_fps.valueChanged.connect(self._main_window._update_fps_info)
        acquisition_row.addWidget(self.spin_fps)

        self.btn_detect_fps = QPushButton("Detect")
        self.btn_detect_fps.clicked.connect(self._detect_fps_from_current_video)
        self.btn_detect_fps.setEnabled(False)
        self.btn_detect_fps.setObjectName("SecondaryBtn")
        self.btn_detect_fps.setFixedHeight(30)
        self.btn_detect_fps.setToolTip(
            "Auto-detect frame rate from video metadata (may differ from actual acquisition rate)"
        )
        acquisition_row.addWidget(self.btn_detect_fps)

        animal_caption = QLabel("Animals")
        animal_caption.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd; margin-left: 6px;"
        )
        acquisition_row.addWidget(animal_caption)

        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 200)
        self.spin_max_targets.setValue(4)
        self.spin_max_targets.setFixedHeight(30)
        self.spin_max_targets.setMinimumWidth(84)
        self.spin_max_targets.setToolTip(
            "Maximum number of animals to track simultaneously (1-200).\n"
            "Set this to the expected number of animals in your video.\n"
            "Higher values use more memory and may slow down processing."
        )
        self.spin_max_targets.valueChanged.connect(self._sync_batch_policy_controls)
        acquisition_row.addWidget(self.spin_max_targets)
        acquisition_row.addStretch(1)
        fl.addRow("Acquisition", acquisition_row)

        # FPS info label
        self.label_fps_info = QLabel()
        self.label_fps_info.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic;"
        )
        fl.addRow("", self.label_fps_info)

        vl_files.addLayout(fl)

        # Batch Mode Section
        self.g_batch = QGroupBox("Batch")
        self.g_batch.setCheckable(True)
        self.g_batch.setChecked(False)
        self.g_batch.toggled.connect(self._main_window._on_batch_mode_toggled)
        self._main_window._set_compact_section_widget(self.g_batch)
        vl_batch = QVBoxLayout(self.g_batch)
        vl_batch.setSpacing(6)

        # Warning label (visible only while batch mode is active)
        self.lbl_batch_warning = QLabel(
            "⚠️ All videos in this batch will use the parameters currently selected for the Keystone video."
        )
        self.lbl_batch_warning.setWordWrap(True)
        self.lbl_batch_warning.setStyleSheet(
            "color: #f39c12; font-weight: bold; font-size: 11px; margin-bottom: 5px;"
        )
        self.lbl_batch_warning.setVisible(False)
        vl_batch.addWidget(self.lbl_batch_warning)

        # Container for the rest of the batch UI
        self.container_batch = QWidget()
        v_container = QVBoxLayout(self.container_batch)
        v_container.setContentsMargins(0, 0, 0, 0)

        batch_help = self._main_window._create_help_label(
            "The 'Keystone' video (top of list) defines the tracking parameters. "
            "Additional videos will use their own saved config if available, "
            "otherwise the Keystone settings are applied."
        )
        v_container.addWidget(batch_help)

        self.list_batch_videos = QListWidget()
        self.list_batch_videos.setMaximumHeight(120)
        self.list_batch_videos.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_batch_videos.setToolTip(
            "Double-click a video to switch to it.\n"
            "The current video's config will be auto-saved first."
        )
        self.list_batch_videos.itemDoubleClicked.connect(
            self._main_window._on_batch_video_double_clicked
        )
        v_container.addWidget(self.list_batch_videos)

        batch_btns = QHBoxLayout()
        self.btn_add_batch = QPushButton("Add Videos...")
        self.btn_add_batch.clicked.connect(self._add_videos_to_batch)
        self.btn_add_batch.setObjectName("SecondaryBtn")
        batch_btns.addWidget(self.btn_add_batch)

        self.btn_remove_batch = QPushButton("Remove Selected")
        self.btn_remove_batch.clicked.connect(self._main_window._remove_from_batch)
        self.btn_remove_batch.setObjectName("SecondaryBtn")
        batch_btns.addWidget(self.btn_remove_batch)

        self.btn_clear_batch = QPushButton("Clear Additional")
        self.btn_clear_batch.clicked.connect(self._clear_batch)
        self.btn_clear_batch.setObjectName("SecondaryBtn")
        batch_btns.addWidget(self.btn_clear_batch)
        v_container.addLayout(batch_btns)

        batch_io_btns = QHBoxLayout()
        self.btn_export_batch = QPushButton("Export List...")
        self.btn_export_batch.setToolTip(
            "Save the current batch video list to a text file.\n"
            "The first line will be the Keystone video path."
        )
        self.btn_export_batch.clicked.connect(self._main_window._export_batch_list)
        self.btn_export_batch.setObjectName("SecondaryBtn")
        batch_io_btns.addWidget(self.btn_export_batch)

        self.btn_import_batch = QPushButton("Import List...")
        self.btn_import_batch.setToolTip(
            "Load a batch video list from a text file.\n"
            "The first line must be the Keystone video.\n"
            "Missing files will be reported before proceeding."
        )
        self.btn_import_batch.clicked.connect(self._main_window._import_batch_list)
        self.btn_import_batch.setObjectName("SecondaryBtn")
        batch_io_btns.addWidget(self.btn_import_batch)
        v_container.addLayout(batch_io_btns)

        self.chk_batch_keystone_override = QCheckBox(
            "Override all videos with Keystone settings"
        )
        self.chk_batch_keystone_override.setToolTip(
            "When checked, all videos in the batch will use the Keystone's\n"
            "parameters, ignoring any per-video saved configs.\n"
            "When unchecked (default), a video's own saved config is used\n"
            "if one exists next to the video file."
        )
        self.chk_batch_keystone_override.setChecked(False)
        v_container.addWidget(self.chk_batch_keystone_override)

        vl_batch.addWidget(self.container_batch)
        self.container_batch.setVisible(False)  # Default hidden

        vl_files.addWidget(self.g_batch)
        form.addWidget(g_files)

        # ============================================================
        # Video Player & Frame Range
        # ============================================================
        self.g_video_player = QGroupBox("Preview")
        self._main_window._set_compact_section_widget(self.g_video_player)
        vl_player = QVBoxLayout(self.g_video_player)
        vl_player.setSpacing(6)
        vl_player.addWidget(
            self._main_window._create_help_label(
                "Preview video and select frame range for tracking. Use the slider to seek through the video."
            )
        )

        # Video info label
        self.lbl_video_info = QLabel("No video loaded")
        self.lbl_video_info.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic; padding: 5px;"
        )
        vl_player.addWidget(self.lbl_video_info)

        # Timeline slider
        timeline_layout = QVBoxLayout()
        self.lbl_current_frame = QLabel("Frame: -")
        self.lbl_current_frame.setStyleSheet("font-size: 10px; color: #9a9a9a;")
        timeline_layout.addWidget(self.lbl_current_frame)

        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setMinimum(0)
        self.slider_timeline.setMaximum(0)
        self.slider_timeline.setValue(0)
        self.slider_timeline.setTracking(not sys.platform.startswith("linux"))
        self.slider_timeline.setEnabled(False)
        self.slider_timeline.setToolTip("Seek through video frames")
        self.slider_timeline.valueChanged.connect(
            self._main_window._on_timeline_changed
        )
        self.slider_timeline.sliderPressed.connect(
            self._main_window._on_timeline_pressed
        )
        self.slider_timeline.sliderMoved.connect(self._main_window._on_timeline_moved)
        timeline_layout.addWidget(self.slider_timeline)
        vl_player.addLayout(timeline_layout)

        # Playback controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)

        self.btn_first_frame = QPushButton("⏮")
        self.btn_first_frame.setEnabled(False)
        self.btn_first_frame.clicked.connect(self._main_window._goto_first_frame)
        self.btn_first_frame.setToolTip("Go to first frame")
        self.btn_first_frame.setObjectName("SecondaryBtn")
        self.btn_first_frame.setFixedWidth(44)
        controls_layout.addWidget(self.btn_first_frame)

        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setEnabled(False)
        self.btn_prev_frame.clicked.connect(self._main_window._goto_prev_frame)
        self.btn_prev_frame.setToolTip("Previous frame")
        self.btn_prev_frame.setObjectName("SecondaryBtn")
        self.btn_prev_frame.setFixedWidth(44)
        controls_layout.addWidget(self.btn_prev_frame)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._main_window._toggle_playback)
        self.btn_play_pause.setToolTip("Play/pause video")
        controls_layout.addWidget(self.btn_play_pause)

        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setEnabled(False)
        self.btn_next_frame.clicked.connect(self._main_window._goto_next_frame)
        self.btn_next_frame.setToolTip("Next frame")
        self.btn_next_frame.setObjectName("SecondaryBtn")
        self.btn_next_frame.setFixedWidth(44)
        controls_layout.addWidget(self.btn_next_frame)

        self.btn_last_frame = QPushButton("⏭")
        self.btn_last_frame.setEnabled(False)
        self.btn_last_frame.clicked.connect(self._main_window._goto_last_frame)
        self.btn_last_frame.setToolTip("Go to last frame")
        self.btn_last_frame.setObjectName("SecondaryBtn")
        self.btn_last_frame.setFixedWidth(44)
        controls_layout.addWidget(self.btn_last_frame)

        controls_layout.addSpacing(4)

        self.btn_random_seek = QPushButton("🎲 Random")
        self.btn_random_seek.setEnabled(False)
        self.btn_random_seek.clicked.connect(self._main_window._goto_random_frame)
        self.btn_random_seek.setToolTip("Jump to a random frame")
        self.btn_random_seek.setObjectName("SecondaryBtn")
        controls_layout.addWidget(self.btn_random_seek)

        controls_layout.addSpacing(8)

        # Playback speed control
        speed_label = QLabel("Speed")
        speed_label.setStyleSheet("color: #8a8a8a;")
        controls_layout.addWidget(speed_label)
        self.combo_playback_speed = QComboBox()
        self.combo_playback_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.combo_playback_speed.setCurrentText("1x")
        self.combo_playback_speed.setEnabled(False)
        self.combo_playback_speed.setToolTip("Playback speed")
        self.combo_playback_speed.setMaximumWidth(84)
        controls_layout.addWidget(self.combo_playback_speed)
        controls_layout.addStretch(1)

        vl_player.addLayout(controls_layout)

        vl_player.addWidget(self._main_window._make_setup_divider())

        # Frame range selection
        range_label = QLabel("Frame range")
        range_label.setStyleSheet("font-weight: 600; color: #d0d0d0;")
        vl_player.addWidget(range_label)

        # Compact single row: Start [spinbox] [↕] · End [spinbox] [↕] [Reset]
        _range_row = QHBoxLayout()
        _range_row.setSpacing(6)
        _range_row.addWidget(QLabel("Start:"))
        self.spin_start_frame = QSpinBox()
        self.spin_start_frame.setMinimum(0)
        self.spin_start_frame.setMaximum(0)
        self.spin_start_frame.setValue(0)
        self.spin_start_frame.setKeyboardTracking(False)
        self.spin_start_frame.setEnabled(False)
        self.spin_start_frame.setToolTip("First frame to track (0-based index)")
        self.spin_start_frame.valueChanged.connect(
            self._main_window._on_frame_range_changed
        )
        _range_row.addWidget(self.spin_start_frame, 1)
        self.btn_set_start_current = QPushButton("↕")
        self.btn_set_start_current.setEnabled(False)
        self.btn_set_start_current.setMaximumWidth(30)
        self.btn_set_start_current.clicked.connect(
            self._main_window._set_start_to_current
        )
        self.btn_set_start_current.setToolTip("Set start frame to current frame")
        _range_row.addWidget(self.btn_set_start_current)
        _range_row.addSpacing(10)
        _range_row.addWidget(QLabel("End:"))
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setMinimum(0)
        self.spin_end_frame.setMaximum(0)
        self.spin_end_frame.setValue(0)
        self.spin_end_frame.setKeyboardTracking(False)
        self.spin_end_frame.setEnabled(False)
        self.spin_end_frame.setToolTip("Last frame to track (0-based index, inclusive)")
        self.spin_end_frame.valueChanged.connect(
            self._main_window._on_frame_range_changed
        )
        _range_row.addWidget(self.spin_end_frame, 1)
        self.btn_set_end_current = QPushButton("↕")
        self.btn_set_end_current.setEnabled(False)
        self.btn_set_end_current.setMaximumWidth(30)
        self.btn_set_end_current.clicked.connect(self._main_window._set_end_to_current)
        self.btn_set_end_current.setToolTip("Set end frame to current frame")
        _range_row.addWidget(self.btn_set_end_current)
        _range_row.addSpacing(10)
        self.btn_reset_range = QPushButton("Reset")
        self.btn_reset_range.setEnabled(False)
        self.btn_reset_range.clicked.connect(self._main_window._reset_frame_range)
        self.btn_reset_range.setToolTip("Reset to track entire video")
        self.btn_reset_range.setObjectName("SecondaryBtn")
        _range_row.addWidget(self.btn_reset_range)
        vl_player.addLayout(_range_row)

        # Range info
        self.lbl_range_info = QLabel()
        self.lbl_range_info.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic; padding: 5px;"
        )
        vl_player.addWidget(self.lbl_range_info)
        form.addWidget(self.g_video_player)

        # Initially hide video player (shown when video is loaded)
        self.g_video_player.setVisible(False)

        # ============================================================
        # Output Files
        # ============================================================
        g_output = QGroupBox("Outputs")
        self._main_window._set_compact_section_widget(g_output)
        vl_output = QVBoxLayout(g_output)
        vl_output.setSpacing(6)
        fl_output = QFormLayout(None)
        fl_output.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fl_output.setHorizontalSpacing(10)
        fl_output.setVerticalSpacing(8)

        self.btn_csv = QPushButton("Browse...")
        self.btn_csv.clicked.connect(self._main_window.select_csv)
        self.btn_csv.setObjectName("SecondaryBtn")
        self.btn_csv.setFixedHeight(30)
        self.csv_line = QLineEdit()
        self.csv_line.setPlaceholderText("path/to/output.csv")
        self.csv_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.csv_line.setFixedHeight(30)
        csv_row = QHBoxLayout()
        csv_row.setSpacing(8)
        csv_row.addWidget(self.csv_line, 1)
        csv_row.addWidget(self.btn_csv)
        fl_output.addRow("Tracking CSV", csv_row)

        # Config Management
        config_layout = QHBoxLayout()
        config_layout.setSpacing(6)
        self.btn_load_config = QPushButton("Load Config...")
        self.btn_load_config.clicked.connect(self._main_window.load_config)
        self.btn_load_config.setToolTip("Manually load configuration from a JSON file")
        self.btn_load_config.setObjectName("SecondaryBtn")
        self.btn_load_config.setFixedHeight(28)
        config_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Config...")
        self.btn_save_config.clicked.connect(self._main_window.save_config)
        self.btn_save_config.setToolTip("Save current settings to a JSON file")
        self.btn_save_config.setObjectName("SecondaryBtn")
        self.btn_save_config.setFixedHeight(28)
        config_layout.addWidget(self.btn_save_config)

        self.btn_show_gpu_info = QPushButton("GPU Info")
        self.btn_show_gpu_info.clicked.connect(self._main_window.show_gpu_info)
        self.btn_show_gpu_info.setToolTip(
            "Show available GPU and acceleration information"
        )
        self.btn_show_gpu_info.setObjectName("SecondaryBtn")
        self.btn_show_gpu_info.setFixedHeight(28)
        config_layout.addWidget(self.btn_show_gpu_info)
        config_layout.addStretch(1)

        fl_output.addRow("Config tools", config_layout)

        # Config status label
        self.config_status_label = QLabel("No config loaded (using defaults)")
        self.config_status_label.setStyleSheet(
            "color: #6a6a6a; font-style: italic; font-size: 10px;"
        )
        fl_output.addRow("", self.config_status_label)
        vl_output.addLayout(fl_output)

        form.addWidget(g_output)

        # ============================================================
        # System Performance
        # ============================================================
        g_sys = QGroupBox("Performance")
        self._main_window._set_compact_section_widget(g_sys)
        vl_sys = QVBoxLayout(g_sys)
        vl_sys.setSpacing(6)
        vl_sys.addWidget(
            self._main_window._create_help_label(
                "Resize factor reduces computational cost by downscaling frames. "
                "Lower values speed up processing but reduce spatial accuracy."
            )
        )
        self.form_performance = None
        self.performance_control_grid = QGridLayout()
        self.performance_control_grid.setContentsMargins(0, 0, 0, 0)
        self.performance_control_grid.setHorizontalSpacing(10)
        self.performance_control_grid.setVerticalSpacing(10)
        self._performance_base_control_cards = []
        self._performance_control_cards = []
        self._performance_optional_control_cards = {}

        self.spin_resize = QDoubleSpinBox()
        self.spin_resize.setRange(0.1, 1.0)
        self.spin_resize.setSingleStep(0.1)
        self.spin_resize.setValue(1.0)
        self.spin_resize.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_resize.setFixedHeight(30)
        self.spin_resize.setToolTip(
            "Downscale video frames for faster processing.\n"
            "1.0 = full resolution, 0.5 = half resolution (4× faster).\n"
            "All body-size-based parameters auto-scale with this value."
        )
        from hydra_suite.runtime.resolver import (
            available_tiers,
            detect_platform,
            tier_label,
        )

        self.combo_runtime_tier = QComboBox()
        self.combo_runtime_tier.setFixedHeight(30)
        self.combo_runtime_tier.setToolTip(
            "Compute tier for the whole pipeline.\n"
            "CPU · GPU (exact) · GPU-Fast (max speed, some accuracy loss)."
        )
        _platform = detect_platform()
        for _tier in available_tiers(_platform):
            self.combo_runtime_tier.addItem(tier_label(_tier, _platform), _tier)
        self.combo_runtime_tier.currentIndexChanged.connect(
            self._main_window._on_runtime_tier_changed
        )
        # §5.4 fallback indicator: shown under the tier selector when GPU-Fast is
        # active, noting that stages without a fast artifact run on the native GPU.
        self.lbl_runtime_fallback = QLabel("")
        self.lbl_runtime_fallback.setWordWrap(True)
        self.lbl_runtime_fallback.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_runtime_fallback.setVisible(False)

        scale_card = self._create_performance_control_card("Scale", self.spin_resize)
        self._performance_base_control_cards.append(scale_card)
        self._performance_control_cards.append(scale_card)
        # Stack the tier combo and the fallback hint in one container so the hint
        # renders inside the "Compute tier" card.
        _tier_container = QWidget()
        _tier_layout = QVBoxLayout(_tier_container)
        _tier_layout.setContentsMargins(0, 0, 0, 0)
        _tier_layout.setSpacing(2)
        _tier_layout.addWidget(self.combo_runtime_tier)
        _tier_layout.addWidget(self.lbl_runtime_fallback)
        tier_card = self._create_performance_control_card(
            "Compute tier",
            _tier_container,
        )
        self._performance_base_control_cards.append(tier_card)
        self._performance_control_cards.append(tier_card)

        # Pose runtime is fully derived from Compute tier (spec §2/§5.2) — no
        # UI control for it at all, not even read-only. See RuntimeResolver.

        self.check_save_confidence = QCheckBox("Save metrics")
        self.check_save_confidence.setChecked(True)
        self.check_save_confidence.setToolTip(
            "Save detection, assignment, and position uncertainty metrics to CSV.\n"
            "Useful for post-hoc quality control but adds ~10-20% processing time.\n"
            "Disable for maximum tracking speed."
        )

        # Use Cached Detections
        self.chk_use_cached_detections = QCheckBox("Reuse cache")
        self.chk_use_cached_detections.setChecked(True)
        self.chk_use_cached_detections.setToolTip(
            "Automatically reuse detections from previous runs if available.\n"
            "Cache is model-specific: only reused if detection method/model hasn't changed.\n"
            "Massive speedup for re-processing with different tracking parameters.\n"
            "Disable to force fresh detection on every run."
        )

        self.chk_realtime_mode = QCheckBox("Realtime workflow")
        self.chk_realtime_mode.setChecked(False)
        self.chk_realtime_mode.setToolTip(
            "Run the forward pass as a true streaming workflow.\n"
            "\n"
            "When enabled, tracking favors per-frame processing over cache-first\n"
            "non-realtime execution. Final canonical images and oriented videos are\n"
            "still exported only after backward tracking and cleanup complete."
        )
        self.chk_realtime_mode.stateChanged.connect(self._sync_realtime_cache_controls)

        # Visualization-Free Mode
        self.chk_visualization_free = QCheckBox("Headless preview")
        self.chk_visualization_free.setChecked(False)
        self.chk_visualization_free.setToolTip(
            "Skip all frame visualization and rendering.\n"
            "Significantly faster processing (2-4× speedup).\n"
            "Real-time FPS/ETA stats still shown in UI.\n"
            "Recommended for large batch processing."
        )
        self.chk_visualization_free.stateChanged.connect(
            self._main_window._on_visualization_mode_changed
        )

        for perf_checkbox in (
            self.check_save_confidence,
            self.chk_use_cached_detections,
            self.chk_realtime_mode,
            self.chk_visualization_free,
        ):
            perf_checkbox.setStyleSheet("font-size: 10px; spacing: 6px;")
            perf_checkbox.setMinimumHeight(26)
            perf_checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        perf_toggle_grid = QGridLayout()
        perf_toggle_grid.setHorizontalSpacing(12)
        perf_toggle_grid.setVerticalSpacing(6)
        perf_toggle_grid.setContentsMargins(0, 0, 0, 0)
        perf_toggle_grid.addWidget(self.check_save_confidence, 0, 0)
        perf_toggle_grid.addWidget(self.chk_use_cached_detections, 0, 1)
        perf_toggle_grid.addWidget(self.chk_realtime_mode, 0, 2)
        perf_toggle_grid.addWidget(self.chk_visualization_free, 0, 3)
        perf_toggle_grid.setColumnStretch(0, 1)
        perf_toggle_grid.setColumnStretch(1, 1)
        perf_toggle_grid.setColumnStretch(2, 1)
        perf_toggle_grid.setColumnStretch(3, 1)
        self._reflow_performance_controls()
        vl_sys.addLayout(self.performance_control_grid)
        vl_sys.addLayout(perf_toggle_grid)
        self._sync_realtime_cache_controls()

        self.btn_clear_detection_caches = QPushButton("Clear All Caches")
        self.btn_clear_detection_caches.setFixedHeight(28)
        self.btn_clear_detection_caches.setToolTip(
            "Delete TrackerKit cache files for the current video.\n"
            "This includes detection, pose, AprilTag, classifier, and related cache artifacts."
        )
        self.btn_clear_detection_caches.clicked.connect(
            self._main_window._clear_detection_caches
        )
        self.btn_benchmark_models = QPushButton("Benchmark Models...")
        self.btn_benchmark_models.setFixedHeight(28)
        self.btn_benchmark_models.setToolTip(
            "Benchmark the currently selected TrackerKit model pipelines across runtimes\n"
            "and batch sizes using the loaded video geometry and body-size settings."
        )
        self.btn_benchmark_models.clicked.connect(
            self._main_window._open_benchmark_dialog
        )
        _perf_actions_row = QHBoxLayout()
        _perf_actions_row.setContentsMargins(0, 0, 0, 0)
        _perf_actions_row.addStretch(1)
        _perf_actions_row.addWidget(self.btn_benchmark_models)
        _perf_actions_row.addWidget(self.btn_clear_detection_caches)
        vl_sys.addLayout(_perf_actions_row)

        form.addWidget(g_sys)

        # ============================================================
        # Display Settings (moved from Visuals tab)
        # ============================================================
        self.g_display = QGroupBox("Preview Overlays")
        self._main_window._set_compact_section_widget(self.g_display)
        vl_display = QVBoxLayout(self.g_display)
        vl_display.addWidget(
            self._main_window._create_help_label(
                "Configure visual overlays shown during tracking. These settings affect "
                "both the live preview and exported video output."
            )
        )

        # Common overlays (2 per row)
        self.chk_show_circles = QCheckBox("Show Track Markers (Circles)")
        self.chk_show_circles.setChecked(True)
        self.chk_show_circles.setToolTip("Draw circles around tracked animals.")

        self.chk_show_orientation = QCheckBox("Show Orientation Lines")
        self.chk_show_orientation.setChecked(True)
        self.chk_show_orientation.setToolTip("Draw lines showing heading direction.")

        self.chk_show_trajectories = QCheckBox("Show Trajectory Trails")
        self.chk_show_trajectories.setChecked(True)
        self.chk_show_trajectories.setToolTip(
            "Draw recent path history for each track."
        )

        self.chk_show_labels = QCheckBox("Show ID Labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.setToolTip("Display unique track IDs on each animal.")

        self.chk_show_state = QCheckBox("Show State Text")
        self.chk_show_state.setChecked(True)
        self.chk_show_state.setToolTip(
            "Display tracking state (ACTIVE, PREDICTED, etc.)."
        )

        self.chk_show_kalman_uncertainty = QCheckBox("Show prediction uncertainty")
        self.chk_show_kalman_uncertainty.setChecked(False)
        self.chk_show_kalman_uncertainty.setToolTip(
            "Draw ellipses showing Kalman filter position uncertainty.\n"
            "Larger ellipse = more uncertainty in predicted position.\n"
            "Useful for debugging tracking quality and filter convergence."
        )

        _disp_r1 = QHBoxLayout()
        _disp_r1.addWidget(self.chk_show_circles)
        _disp_r1.addWidget(self.chk_show_orientation)
        _disp_r2 = QHBoxLayout()
        _disp_r2.addWidget(self.chk_show_trajectories)
        _disp_r2.addWidget(self.chk_show_labels)
        _disp_r3 = QHBoxLayout()
        _disp_r3.addWidget(self.chk_show_state)
        _disp_r3.addWidget(self.chk_show_kalman_uncertainty)
        vl_display.addLayout(_disp_r1)
        vl_display.addLayout(_disp_r2)
        vl_display.addLayout(_disp_r3)

        # Trail length
        f_trail = QFormLayout(None)
        self.spin_traj_hist = QSpinBox()
        self.spin_traj_hist.setRange(-1, 999999)
        self.spin_traj_hist.setValue(5)
        self.spin_traj_hist.setKeyboardTracking(False)
        self.spin_traj_hist.setToolTip(
            "Length of trajectory trails to display in frames.\n"
            "Use -1 for the complete video history.\n"
            "Use 0 to disable trajectory trails.\n"
            "Positive values are capped to the loaded video length."
        )
        self.spin_traj_hist.valueChanged.connect(
            self._main_window._on_trail_history_changed
        )
        f_trail.addRow("Trail history", self.spin_traj_hist)
        vl_display.addLayout(f_trail)

        form.addWidget(self.g_display)

        # ============================================================
        # Advanced / Debug (moved from Visuals tab)
        # ============================================================
        g_debug = QGroupBox("Debug")
        self._main_window._set_compact_section_widget(g_debug)
        v_dbg = QVBoxLayout(g_debug)
        v_dbg.setSpacing(6)
        v_dbg.addWidget(
            self._main_window._create_help_label(
                "Enable verbose logging to see detailed tracking decisions. Useful for troubleshooting "
                "but generates large log files. Disable for production runs."
            )
        )
        self.chk_debug_logging = QCheckBox("Enable detailed debug logging")
        self.chk_debug_logging.stateChanged.connect(
            self._main_window.toggle_debug_logging
        )
        self.chk_enable_profiling = QCheckBox("Enable performance profiling")
        self.chk_enable_profiling.setToolTip(
            "Collect detailed timing for every tracking pipeline step "
            "(init, detection, precompute, tracking loop, post-processing). "
            "Exports a JSON profile next to outputs. Disabled by default for zero overhead."
        )
        debug_toggle_grid = QGridLayout()
        debug_toggle_grid.setContentsMargins(0, 0, 0, 0)
        debug_toggle_grid.setHorizontalSpacing(12)
        debug_toggle_grid.setVerticalSpacing(6)
        debug_toggle_grid.addWidget(self.chk_debug_logging, 0, 0)
        debug_toggle_grid.addWidget(self.chk_enable_profiling, 0, 1)
        debug_toggle_grid.setColumnStretch(0, 1)
        debug_toggle_grid.setColumnStretch(1, 1)
        v_dbg.addLayout(debug_toggle_grid)
        form.addWidget(g_debug)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _sync_realtime_cache_controls(self, *_args) -> None:
        """Disable cache reuse whenever realtime workflow is enabled."""
        realtime_enabled = bool(self.chk_realtime_mode.isChecked())
        if realtime_enabled and self.chk_use_cached_detections.isChecked():
            self.chk_use_cached_detections.setChecked(False)
        self.chk_use_cached_detections.setEnabled(not realtime_enabled)
        self._sync_batch_policy_controls()
        # NOTE: the compute-tier selector is populated at panel construction;
        # _on_runtime_context_changed is called afterward in main_window.py

    def _create_performance_control_card(self, title: str, widget: QWidget) -> QFrame:
        """Build a compact labeled card for one performance control."""
        card = QFrame()
        card.setObjectName("PerformanceControlCard")
        card.setStyleSheet(
            "QFrame#PerformanceControlCard {"
            "background-color: #252526;"
            "border: 1px solid #3e3e42;"
            "border-radius: 8px;"
            "}"
            "QLabel#PerformanceControlLabel {"
            "color: #bdbdbd;"
            "font-size: 10px;"
            "font-weight: 600;"
            "}"
        )
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        label = QLabel(title)
        label.setObjectName("PerformanceControlLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return card

    def _register_optional_performance_control(
        self,
        widget: QWidget,
        title: str,
    ) -> None:
        """Track an optional runtime control that should repack when toggled."""
        card = self._create_performance_control_card(title, widget)
        card.setVisible(False)
        self._performance_control_cards.append(card)
        self._performance_optional_control_cards[widget] = card

    def _set_performance_control_visible(
        self,
        widget: QWidget | None,
        visible: bool,
    ) -> bool:
        """Show or hide a tracked performance control and repack the grid."""
        card = self._performance_optional_control_cards.get(widget)
        if card is None or widget is None:
            return False
        widget.setVisible(bool(visible))
        card.setVisible(bool(visible))
        self._reflow_performance_controls()
        return True

    def _reflow_performance_controls(self) -> None:
        """Lay out visible performance controls into a compact two-column grid."""
        while self.performance_control_grid.count():
            self.performance_control_grid.takeAt(0)

        visible_cards = list(self._performance_base_control_cards)
        visible_cards.extend(
            card
            for card in self._performance_control_cards
            if card not in self._performance_base_control_cards and not card.isHidden()
        )
        column_count = 2
        for index, card in enumerate(visible_cards):
            row = index // column_count
            column = index % column_count
            self.performance_control_grid.addWidget(card, row, column)
        for column in range(column_count):
            self.performance_control_grid.setColumnStretch(column, 1)

    def _sync_batch_policy_controls(self, *_args) -> None:
        """Notify dependent panels when realtime or animal-count policy changes."""
        if hasattr(self._main_window, "_detection_panel"):
            self._main_window._detection_panel._sync_live_detection_batch_controls()
        if hasattr(self._main_window, "_identity_panel"):
            self._main_window._identity_panel._sync_realtime_individual_batch_ui()

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    # ------------------------------------------------------------------
    # Handler methods
    # ------------------------------------------------------------------

    def _on_preset_selection_changed(self, index):
        """Update description label when preset selection changes."""
        filepath = self.combo_presets.currentData()
        if not filepath or not os.path.exists(filepath):
            self.preset_description_label.setVisible(False)
            return

        try:
            with open(filepath, "r") as f:
                cfg = json.load(f)

            description = cfg.get("description", "")
            if description:
                self.preset_description_label.setText(f"📋 {description}")
                self.preset_description_label.setVisible(True)
            else:
                self.preset_description_label.setVisible(False)
        except (OSError, json.JSONDecodeError):
            self.preset_description_label.setVisible(False)

    def _add_videos_to_batch(self):
        """Add additional videos to the batch list."""
        from hydra_suite.paths import get_projects_dir

        start_dir = (
            os.path.dirname(self._main_window.batch_videos[0])
            if self._main_window.batch_videos
            else str(get_projects_dir())
        )
        fps, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Additional Videos",
            start_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if fps:
            for fp in fps:
                if fp not in self._main_window.batch_videos:
                    self._main_window.batch_videos.append(fp)
            self._main_window._sync_batch_list_ui()

    def _clear_batch(self):
        """Clear all additional videos, keeping only the keystone."""
        if len(self._main_window.batch_videos) > 1:
            self._main_window.batch_videos = [self._main_window.batch_videos[0]]
            self._main_window._sync_batch_list_ui()

    def _detect_fps_from_current_video(self):
        """Detect and set FPS from the currently loaded video."""
        if not self._main_window.current_video_path:
            QMessageBox.warning(
                self, "No Video Loaded", "Please load a video file first."
            )
            return

        detected_fps = self._main_window._auto_detect_fps(
            self._main_window.current_video_path
        )
        if detected_fps is not None:
            self.spin_fps.setValue(detected_fps)
            QMessageBox.information(
                self,
                "FPS Detected",
                f"Frame rate detected: {detected_fps:.2f} FPS\n\n"
                f"Time per frame: {1000.0 / detected_fps:.2f} ms",
            )
