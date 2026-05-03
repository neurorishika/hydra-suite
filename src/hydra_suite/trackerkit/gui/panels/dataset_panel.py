"""DatasetPanel — active learning dataset generation controls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class DatasetPanel(QWidget):
    """Active learning dataset generation: frame selection, export, and controls."""

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
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout."""
        layout = self._layout
        layout.setContentsMargins(0, 0, 0, 0)

        # Add scroll area
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
        # Active Learning Dataset Section
        # ============================================================
        self.g_active_learning = QGroupBox(
            "Do you want to generate a detection dataset?"
        )
        self._main_window._set_compact_section_widget(self.g_active_learning)
        vl_active = QVBoxLayout(self.g_active_learning)
        vl_active.addWidget(
            self._main_window._create_help_label(
                "Automatically identify challenging frames during tracking and export them for annotation.\n\n"
                "Workflow: Run tracking → Review/correct in DetectKit → Train improved YOLO model"
            )
        )

        # Enable checkbox
        self.chk_enable_dataset_gen = QCheckBox(
            "Enable Dataset Generation for Active Learning"
        )
        self.chk_enable_dataset_gen.setChecked(False)
        self.chk_enable_dataset_gen.toggled.connect(self._on_dataset_generation_toggled)
        vl_active.addWidget(self.chk_enable_dataset_gen)

        # Content container for all configuration options
        self.active_learning_content = QWidget()
        vl_content = QVBoxLayout(self.active_learning_content)

        # Dataset configuration
        self.g_dataset_config = QGroupBox("How should the dataset be configured?")
        self._main_window._set_compact_section_widget(self.g_dataset_config)
        f_config = QFormLayout(self.g_dataset_config)
        f_config.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Class name
        self.line_dataset_class_name = QLineEdit()
        self.line_dataset_class_name.setPlaceholderText("e.g., ant")
        self.line_dataset_class_name.setText("object")
        self.line_dataset_class_name.setToolTip(
            "Name of the object class being tracked.\n"
            "This will be used in the classes.txt file for YOLO training.\n"
            "Examples: ant, bee, mouse, fish, etc."
        )
        f_config.addRow("Class label", self.line_dataset_class_name)

        vl_content.addWidget(self.g_dataset_config)

        # Frame selection parameters
        self.g_frame_selection = QGroupBox("How should frames be selected?")
        self._main_window._set_compact_section_widget(self.g_frame_selection)
        f_selection = QFormLayout(self.g_frame_selection)
        f_selection.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Number of frames to export
        self.spin_dataset_max_frames = QSpinBox()
        self.spin_dataset_max_frames.setRange(10, 1000)
        self.spin_dataset_max_frames.setValue(100)
        self.spin_dataset_max_frames.setToolTip(
            "Maximum number of frames to export (10-1000).\n"
            "Higher values provide more training data but increase annotation time.\n"
            "Recommended: 50-200 frames for initial improvement."
        )
        f_selection.addRow("Maximum frames to export", self.spin_dataset_max_frames)

        self.spin_dataset_min_selection_score = QDoubleSpinBox()
        self.spin_dataset_min_selection_score.setRange(0.0, 1.0)
        self.spin_dataset_min_selection_score.setSingleStep(0.05)
        self.spin_dataset_min_selection_score.setDecimals(2)
        self.spin_dataset_min_selection_score.setValue(0.0)
        self.spin_dataset_min_selection_score.setToolTip(
            "Min selection score (0.0-1.0).\n\n"
            "Frames with a normalized acquisition score below this value are\n"
            "discarded before top-K selection. 0.0 = no filter (default).\n\n"
            "Replaces the legacy 'Quality threshold' control: under the new\n"
            "normalized scoring, an unbounded threshold no longer makes sense."
        )
        f_selection.addRow("Min selection score", self.spin_dataset_min_selection_score)

        self.combo_dataset_preset = QComboBox()
        for preset_name in (
            "tracker_default",
            "balanced",
            "uncertainty_heavy",
            "exploration_heavy",
        ):
            self.combo_dataset_preset.addItem(preset_name)
        self.combo_dataset_preset.setToolTip(
            "Acquisition weight preset. Default 'tracker_default' includes tracker-side\n"
            "signals (assignment cost, track loss). Others apply to detector-only paths."
        )
        f_selection.addRow("Acquisition preset", self.combo_dataset_preset)

        # Add help label explaining advanced options
        advanced_help = self._main_window._create_help_label(
            "Note: YOLO detection sensitivity for export (confidence=0.05, IOU=0.5) can be "
            "customized in advanced_config.json. These are separate from tracking parameters and "
            "optimized for annotation (detect everything, manual review corrects errors).",
            attach_to_title=False,
        )
        f_selection.addRow(advanced_help)

        # Visual diversity window
        self.spin_dataset_diversity_window = QSpinBox()
        self.spin_dataset_diversity_window.setRange(10, 500)
        self.spin_dataset_diversity_window.setValue(30)
        self.spin_dataset_diversity_window.setToolTip(
            "Minimum frame separation for visual diversity (10-500 frames).\n"
            "Prevents selecting too many consecutive similar frames.\n"
            "Higher = more spread out frames, more visual variety.\n"
            "Recommended: 20-50 frames (depends on video frame rate)."
        )
        f_selection.addRow(
            "Diversity window (frames)",
            self.spin_dataset_diversity_window,
        )

        # Include context frames
        self.chk_dataset_include_context = QCheckBox(
            "Include neighboring frames (+/-1)"
        )
        self.chk_dataset_include_context.setChecked(True)
        self.chk_dataset_include_context.setToolTip(
            "Export the frame before and after each selected frame.\n"
            "Provides temporal context which can improve annotation quality.\n"
            "Increases dataset size by 3x."
        )
        self.chk_dataset_probabilistic = QCheckBox("Probabilistic Sampling")
        self.chk_dataset_probabilistic.setChecked(True)
        self.chk_dataset_probabilistic.setToolTip(
            "Use rank-based probabilistic sampling instead of greedy selection.\n"
            "Probabilistic: Higher quality scores = higher probability (more variety).\n"
            "Greedy: Always select absolute worst frames first (may be too extreme).\n"
            "Recommended: Enabled for better training data diversity."
        )
        _sel_chk_row = QHBoxLayout()
        _sel_chk_row.addWidget(self.chk_dataset_include_context)
        _sel_chk_row.addWidget(self.chk_dataset_probabilistic)
        f_selection.addRow(_sel_chk_row)

        vl_content.addWidget(self.g_frame_selection)

        # Quality metrics
        self.g_quality_metrics = QGroupBox("Which quality checks should be applied?")
        self._main_window._set_compact_section_widget(self.g_quality_metrics)
        v_metrics = QVBoxLayout(self.g_quality_metrics)

        self.chk_metric_low_confidence = QCheckBox("Flag low detection confidence")
        self.chk_metric_low_confidence.setChecked(True)
        self.chk_metric_low_confidence.setToolTip(
            "Flag frames where YOLO confidence is below threshold."
        )
        self.chk_metric_count_mismatch = QCheckBox("Flag detection count mismatch")
        self.chk_metric_count_mismatch.setChecked(True)
        self.chk_metric_count_mismatch.setToolTip(
            "Flag frames where detected count doesn't match expected number of animals."
        )
        self.chk_metric_fragmented_detections = QCheckBox(
            "Flag suspicious split or duplicate detections"
        )
        self.chk_metric_fragmented_detections.setChecked(True)
        self.chk_metric_fragmented_detections.setToolTip(
            "Flag frames where detections overlap or cluster tightly enough to suggest one animal "
            "was split into multiple detections."
        )
        self.chk_metric_high_assignment_cost = QCheckBox(
            "Flag uncertain track assignment"
        )
        self.chk_metric_high_assignment_cost.setChecked(True)
        self.chk_metric_high_assignment_cost.setToolTip(
            "Flag frames where tracker struggles to match detections to tracks."
        )
        self.chk_metric_track_loss = QCheckBox("Flag frequent track loss")
        self.chk_metric_track_loss.setChecked(True)
        self.chk_metric_track_loss.setToolTip(
            "Flag frames where tracks are frequently lost."
        )
        self.chk_metric_high_uncertainty = QCheckBox("Flag high position uncertainty")
        self.chk_metric_high_uncertainty.setChecked(False)
        self.chk_metric_high_uncertainty.setToolTip(
            "Flag frames where Kalman filter is very uncertain about positions."
        )
        _m_row1 = QHBoxLayout()
        _m_row1.addWidget(self.chk_metric_low_confidence)
        _m_row1.addWidget(self.chk_metric_count_mismatch)
        _m_row2 = QHBoxLayout()
        _m_row2.addWidget(self.chk_metric_fragmented_detections)
        _m_row2.addWidget(self.chk_metric_high_assignment_cost)
        _m_row3 = QHBoxLayout()
        _m_row3.addWidget(self.chk_metric_track_loss)
        v_metrics.addLayout(_m_row1)
        v_metrics.addLayout(_m_row2)
        v_metrics.addLayout(_m_row3)
        v_metrics.addWidget(self.chk_metric_high_uncertainty)

        vl_content.addWidget(self.g_quality_metrics)

        # Add content to main group box
        vl_active.addWidget(self.active_learning_content)

        # Add main group box to form
        form.addWidget(self.g_active_learning)

        # Initially hide content (checkbox starts unchecked)
        self.active_learning_content.setVisible(False)

        # ============================================================
        # Final canonical image export section
        # ============================================================
        self.g_individual_dataset = QGroupBox(
            "Should final canonical crop images be exported after cleanup?"
        )
        self._main_window._set_compact_section_widget(self.g_individual_dataset)
        vl_ind_dataset = QVBoxLayout(self.g_individual_dataset)
        vl_ind_dataset.addWidget(
            self._main_window._create_help_label(
                "Export final canonical still images only after backward tracking and cleanup finish.\n\n"
                "• Uses the final cleaned track orientation instead of transient forward-pass orientation\n"
                "• Includes both detected frames and interpolated frames from the final trajectory set\n"
                "• Intended for downstream labeling/training workflows that need stable head-tail direction\n"
                "• Saved under individual_crops/<run_id>/images\n\n"
                "Note: Available only in YOLO OBB mode."
            )
        )

        self.chk_enable_individual_dataset = QCheckBox(
            "Export final canonical crop images after cleanup"
        )
        self.chk_enable_individual_dataset.toggled.connect(
            self._on_individual_dataset_toggled
        )
        vl_ind_dataset.addWidget(self.chk_enable_individual_dataset)

        # Output Configuration
        self.ind_output_group = QGroupBox(
            "How should final canonical images be written?"
        )
        ind_output_layout = QFormLayout(self.ind_output_group)
        ind_output_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Output format
        self.combo_individual_format = QComboBox()
        self.combo_individual_format.addItems(["PNG", "JPEG"])
        self.combo_individual_format.setCurrentText("PNG")
        self.combo_individual_format.setToolTip(
            "PNG: Lossless, larger files\nJPEG: Smaller files, slight quality loss"
        )
        # Save interval
        self.spin_individual_interval = QSpinBox()
        self.spin_individual_interval.setRange(1, 100)
        self.spin_individual_interval.setValue(1)
        self.spin_individual_interval.setSingleStep(1)
        self.spin_individual_interval.setToolTip(
            "Export canonical images every N frames during the final media pass.\n"
            "1 = every frame, 10 = every 10th frame, etc."
        )
        _ind_fmt_row = QHBoxLayout()
        _ind_fmt_row.addWidget(QLabel("Format"))
        _ind_fmt_row.addWidget(self.combo_individual_format)
        _ind_fmt_row.addWidget(QLabel("Save every N frames"))
        _ind_fmt_row.addWidget(self.spin_individual_interval)
        ind_output_layout.addRow(_ind_fmt_row)

        vl_ind_dataset.addWidget(
            self._main_window._create_help_label(
                "Padding, background, interpolation, and head-tail settings are configured in:\n"
                "Analyze Individuals -> Individual Analysis Pipeline Settings"
            )
        )

        vl_ind_dataset.addWidget(self.ind_output_group)

        self.chk_suppress_foreign_obb_individual_dataset = QCheckBox(
            "Suppress foreign animal regions in saved crop images"
        )
        self.chk_suppress_foreign_obb_individual_dataset.setChecked(False)
        self.chk_suppress_foreign_obb_individual_dataset.setToolTip(
            "Fill overlapping animals' OBB areas with the background color before\n"
            "writing final canonical crop images to disk.\n"
            "\n"
            "Prevents other animals from appearing inside saved crops used for\n"
            "downstream labeling or training. Only applies to YOLO OBB detections\n"
            "(no effect in background-subtraction mode)."
        )
        vl_ind_dataset.addWidget(self.chk_suppress_foreign_obb_individual_dataset)

        # Info label about filtering
        self.lbl_individual_info = self._main_window._create_help_label(
            "Final canonical images reuse detections already filtered by ROI and size settings.\n"
            "No forward-pass media export is performed.",
            attach_to_title=False,
        )
        vl_ind_dataset.addWidget(self.lbl_individual_info)

        form.addWidget(self.g_individual_dataset)

        # ============================================================
        # Oriented Video Export Section
        # ============================================================
        self.g_oriented_videos = QGroupBox(
            "Should oriented videos be exported after cleanup?"
        )
        self._main_window._set_compact_section_widget(self.g_oriented_videos)
        vl_oriented = QVBoxLayout(self.g_oriented_videos)
        vl_oriented.addWidget(
            self._main_window._create_help_label(
                "Export one orientation-fixed video per final cleaned trajectory.\n\n"
                "• Runs after final cleanup completes\n"
                "• Uses the detection cache plus interpolated ROI geometry\n"
                "• Can run without saving individual crop images\n"
                "• Saved beside active_learning/ and individual_crops/ under oriented_videos/<run_id>\n\n"
                "Requires head-tail orientation to be configured in Analyze Individuals."
            )
        )

        self.chk_generate_individual_track_videos = QCheckBox(
            "Generate orientation-fixed videos for final tracks after cleanup"
        )
        self.chk_generate_individual_track_videos.setChecked(False)
        self.chk_generate_individual_track_videos.setToolTip(
            "After final cleaning completes, export one orientation-fixed video per\n"
            "final TrajectoryID by streaming the source video and using the detection\n"
            "cache plus interpolated ROI cache. Independent from saved crop files."
        )
        self.chk_generate_individual_track_videos.toggled.connect(
            self._on_oriented_video_toggled
        )
        vl_oriented.addWidget(self.chk_generate_individual_track_videos)

        self.oriented_video_options = QGroupBox("Oriented Video Post-Processing")
        oriented_options_layout = QFormLayout(self.oriented_video_options)
        oriented_options_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.chk_fix_oriented_video_direction_flips = QCheckBox(
            "Fix short head-tail direction flip bursts"
        )
        self.chk_fix_oriented_video_direction_flips.setChecked(False)
        self.chk_fix_oriented_video_direction_flips.setToolTip(
            "Correct isolated ~180-degree direction bursts after tracking cleanup\n"
            "before rendering oriented videos. Uses the same bounded flip logic\n"
            "as trajectory post-processing, but only affects video export."
        )
        oriented_options_layout.addRow(
            "Direction fixing",
            self.chk_fix_oriented_video_direction_flips,
        )

        self.spin_oriented_video_heading_flip_burst = QSpinBox()
        self.spin_oriented_video_heading_flip_burst.setRange(1, 50)
        self.spin_oriented_video_heading_flip_burst.setValue(5)
        self.spin_oriented_video_heading_flip_burst.setToolTip(
            "Maximum length of an isolated direction-flip burst to correct.\n"
            "Longer runs are preserved as real orientation changes."
        )
        oriented_options_layout.addRow(
            "Max flip burst (frames)",
            self.spin_oriented_video_heading_flip_burst,
        )

        self.chk_enable_oriented_video_affine_stabilization = QCheckBox(
            "Apply temporal affine stabilization"
        )
        self.chk_enable_oriented_video_affine_stabilization.setChecked(False)
        self.chk_enable_oriented_video_affine_stabilization.setToolTip(
            "Temporally smooth crop center, size, and orientation after cleanup\n"
            "to reduce frame-to-frame jitter in exported oriented videos."
        )
        self.chk_enable_oriented_video_affine_stabilization.toggled.connect(
            self._sync_oriented_video_postprocess_controls
        )
        oriented_options_layout.addRow(
            "Affine stabilization",
            self.chk_enable_oriented_video_affine_stabilization,
        )

        self.spin_oriented_video_stabilization_window = QSpinBox()
        self.spin_oriented_video_stabilization_window.setRange(1, 31)
        self.spin_oriented_video_stabilization_window.setSingleStep(2)
        self.spin_oriented_video_stabilization_window.setValue(5)
        self.spin_oriented_video_stabilization_window.setToolTip(
            "Centered temporal smoothing window used for affine stabilization.\n"
            "Odd values work best; even values are rounded up internally."
        )
        oriented_options_layout.addRow(
            "Stabilization window (frames)",
            self.spin_oriented_video_stabilization_window,
        )

        vl_oriented.addWidget(self.oriented_video_options)

        self.chk_suppress_foreign_obb_oriented_videos = QCheckBox(
            "Suppress foreign animal regions in oriented videos"
        )
        self.chk_suppress_foreign_obb_oriented_videos.setChecked(False)
        self.chk_suppress_foreign_obb_oriented_videos.setToolTip(
            "Fill overlapping animals' OBB areas with the background color before\n"
            "rendering oriented-track video frames.\n"
            "\n"
            "Prevents other animals from appearing inside oriented-video exports,\n"
            "which can confuse review and visualization.\n"
            "Only applies to YOLO OBB detections (no effect in background-subtraction mode)."
        )
        vl_oriented.addWidget(self.chk_suppress_foreign_obb_oriented_videos)

        self.lbl_oriented_video_info = self._main_window._create_help_label(
            "Oriented videos reuse detections already filtered by ROI and size settings.\n"
            "No separate crop-dataset save is required.",
            attach_to_title=False,
        )
        vl_oriented.addWidget(self.lbl_oriented_video_info)

        form.addWidget(self.g_oriented_videos)

        # ============================================================
        # Next-step guidance
        # ============================================================
        self.g_downstream_tools = QGroupBox("What should you use next?")
        self._main_window._set_compact_section_widget(self.g_downstream_tools)
        vl_downstream = QVBoxLayout(self.g_downstream_tools)
        vl_downstream.addWidget(
            self._main_window._create_help_label(
                "TrackerKit no longer launches annotation tools from this tab.\n\n"
                "Use DetectKit from the HYDRA Suite launcher to review and correct detection datasets.\n"
                "Use PoseKit from the HYDRA Suite launcher to label pose datasets generated from individual crops."
            )
        )
        guidance = QLabel(
            "Detection review: open DetectKit from HYDRA Suite\n"
            "Pose labeling: open PoseKit from HYDRA Suite"
        )
        guidance.setWordWrap(True)
        guidance.setStyleSheet("color: #b8b8b8; font-size: 11px;")
        vl_downstream.addWidget(guidance)

        form.addWidget(self.g_downstream_tools)

        # Initially hide individual dataset widgets (checkbox starts unchecked)
        self.g_individual_dataset.setVisible(False)
        self.g_oriented_videos.setVisible(False)
        self.ind_output_group.setVisible(False)
        self.chk_suppress_foreign_obb_individual_dataset.setVisible(False)
        self.lbl_individual_info.setVisible(False)
        self.lbl_oriented_video_info.setVisible(False)
        self.oriented_video_options.setVisible(False)
        self._sync_oriented_video_postprocess_controls()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    # =========================================================================
    # Handler methods (moved from MainWindow)
    # =========================================================================

    def _on_dataset_generation_toggled(self, enabled):
        """Enable/disable dataset generation controls."""
        # Hide/show entire content container
        self.active_learning_content.setVisible(enabled)

    def _on_individual_dataset_toggled(self, enabled):
        """Enable/disable individual dataset generation controls."""
        self._main_window._sync_individual_analysis_mode_ui()

    def _on_oriented_video_toggled(self, enabled):
        """Show or hide oriented-video post-processing controls."""
        self.oriented_video_options.setVisible(bool(enabled))
        self._sync_oriented_video_postprocess_controls()

    def _sync_oriented_video_postprocess_controls(self):
        """Enable dependent oriented-video controls only when their toggles are active."""
        enabled = bool(self.chk_generate_individual_track_videos.isChecked())
        self.oriented_video_options.setEnabled(enabled)
        self.chk_suppress_foreign_obb_oriented_videos.setEnabled(enabled)
        self.spin_oriented_video_heading_flip_burst.setEnabled(
            enabled and self.chk_fix_oriented_video_direction_flips.isChecked()
        )
        self.spin_oriented_video_stabilization_window.setEnabled(
            enabled and self.chk_enable_oriented_video_affine_stabilization.isChecked()
        )
