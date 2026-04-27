"""PostProcessPanel — trajectory cleaning, relinking, and interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
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
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class PostProcessPanel(QWidget):
    """Trajectory post-processing: cleaning, velocity breaks, and interpolation."""

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)
        self._main_window._set_compact_scroll_layout(vbox)

        # Post-Processing
        g_pp = QGroupBox("How should tracks be cleaned after tracking?")
        self._main_window._set_compact_section_widget(g_pp)
        vl_pp = QVBoxLayout(g_pp)
        vl_pp.addWidget(
            self._main_window._create_help_label(
                "Clean trajectories after tracking by removing outliers and splitting at identity swaps. "
                "Velocity/distance breaks detect unrealistic jumps that indicate ID switching."
            )
        )
        self.enable_postprocessing = QCheckBox("Auto-clean trajectories")
        self.enable_postprocessing.setChecked(True)
        self.enable_postprocessing.setToolTip(
            "Automatically clean trajectories by removing outliers and fragments.\n"
            "Uses velocity and distance thresholds to detect anomalies.\n"
            "Recommended: Enable for cleaner data output."
        )
        self.enable_postprocessing.stateChanged.connect(self._on_cleaning_toggled)
        vl_pp.addWidget(self.enable_postprocessing)

        self.cleaning_sections_widget = QWidget()
        cleaning_sections_layout = QVBoxLayout(self.cleaning_sections_widget)
        cleaning_sections_layout.setContentsMargins(0, 0, 0, 0)
        cleaning_sections_layout.setSpacing(8)

        self.g_cleaning_filters, f_cleaning_filters = self._create_subsection_form(
            "Trajectory Filters",
            "Core cleanup thresholds that remove very short fragments and split long occlusions.",
        )

        self.g_motion_breaks, f_motion_breaks = self._create_subsection_form(
            "Motion Breaks",
            "Detect abrupt jumps that usually indicate identity swaps or implausible motion.",
        )

        self.g_relinking, f_relinking = self._create_subsection_form(
            "Fragment Relinking",
            "Reconnect fragments conservatively after interpolation when motion and pose remain consistent.",
        )

        self.g_pose_quality, f_pose_quality = self._create_subsection_form(
            "Pose Quality",
            "Only used when pose extraction is enabled. These gates suppress weak pose rows and temporal outliers.",
        )

        self.g_interpolation_merge, f_interpolation_merge = (
            self._create_subsection_form(
                "Interpolation And Merge",
                "Fill short gaps and control how forward/backward trajectories are merged into final clean tracks.",
            )
        )

        self.spin_min_trajectory_length = QDoubleSpinBox()
        self.spin_min_trajectory_length.setRange(0.01, 60.0)
        self.spin_min_trajectory_length.setSingleStep(0.1)
        self.spin_min_trajectory_length.setDecimals(2)
        self.spin_min_trajectory_length.setValue(0.33)
        self.spin_min_trajectory_length.setToolTip(
            "Remove trajectories shorter than this (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Filters out brief false detections and transient tracks.\n"
            "Recommended: 0.15-1.0 s depending on video length."
        )
        self.lbl_min_trajectory_length = QLabel("Minimum trajectory length (seconds)")
        self.spin_max_occlusion_gap = QDoubleSpinBox()
        self.spin_max_occlusion_gap.setRange(0.0, 10.0)
        self.spin_max_occlusion_gap.setSingleStep(0.1)
        self.spin_max_occlusion_gap.setDecimals(2)
        self.spin_max_occlusion_gap.setValue(1.0)
        self.spin_max_occlusion_gap.setToolTip(
            "Maximum occlusion duration before splitting trajectory (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Prevents unreliable interpolation across long gaps.\n"
            "Set to 0 to disable occlusion-based splitting.\n"
            "Recommended: 0.5-2.0 s for typical tracking scenarios."
        )
        self.lbl_max_occlusion_gap = QLabel("Maximum occlusion gap (seconds)")
        self.cleaning_duration_row_widget = self._build_field_grid(
            [
                (self.lbl_min_trajectory_length, self.spin_min_trajectory_length),
                (self.lbl_max_occlusion_gap, self.spin_max_occlusion_gap),
            ],
            columns=2,
        )
        f_cleaning_filters.addRow(self.cleaning_duration_row_widget)

        self.spin_max_velocity_break = QDoubleSpinBox()
        self.spin_max_velocity_break.setRange(1.0, 500.0)
        self.spin_max_velocity_break.setSingleStep(5.0)
        self.spin_max_velocity_break.setDecimals(1)
        self.spin_max_velocity_break.setValue(50.0)
        self.spin_max_velocity_break.setToolTip(
            "Maximum velocity (body-sizes/second) before breaking trajectory.\n"
            "Splits tracks at unrealistic speed jumps (likely identity swaps).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 30-100 body-sizes/s for typical animal motion."
        )
        self.lbl_max_velocity_break = QLabel(
            "Break trajectory above speed (body lengths/sec)"
        )

        self.chk_enable_tracklet_relinking = QCheckBox(
            "Relink fragments after pose interpolation"
        )
        self.chk_enable_tracklet_relinking.setChecked(False)
        self.chk_enable_tracklet_relinking.setToolTip(
            "\u26a0 USE WITH CAUTION \u2014 disabled by default.\n"
            "\n"
            "Reconnect short trajectory fragments after pose/interpolation completes.\n"
            "In dense multi-animal scenes this can cause identity swaps by incorrectly\n"
            "\n"
            "Bidirectional tracking (forward + backward pass) already handles most\n"
            "occlusion recovery. Enable relinking only if you see fragmented trajectories\n"
            "that bidirectional tracking could not repair, and verify results carefully."
        )
        f_relinking.addRow(self.chk_enable_tracklet_relinking)

        self.spin_relink_pose_max_distance = QDoubleSpinBox()
        self.spin_relink_pose_max_distance.setRange(0.05, 5.0)
        self.spin_relink_pose_max_distance.setSingleStep(0.05)
        self.spin_relink_pose_max_distance.setDecimals(2)
        self.spin_relink_pose_max_distance.setValue(0.45)
        self.spin_relink_pose_max_distance.setToolTip(
            "Maximum normalized same-keypoint pose distance allowed when relinking fragments.\n"
            "Lower values are stricter and reduce risky merges."
        )
        self.lbl_relink_pose_max_distance = QLabel("Max normalized pose distance")

        self.spin_pose_export_min_valid_fraction = QDoubleSpinBox()
        self.spin_pose_export_min_valid_fraction.setRange(0.0, 1.0)
        self.spin_pose_export_min_valid_fraction.setSingleStep(0.05)
        self.spin_pose_export_min_valid_fraction.setDecimals(2)
        self.spin_pose_export_min_valid_fraction.setValue(0.5)
        self.spin_pose_export_min_valid_fraction.setToolTip(
            "Minimum fraction of valid keypoints required for a pose row to be accepted.\n"
            "Rows below this threshold have all confidence values zeroed but are kept\n"
            "in the output with quality flags. Only active when pose extraction is enabled.\n"
            "Range: 0.0\u20131.0. Recommended: 0.3\u20130.6."
        )
        self.lbl_pose_export_min_valid_fraction = QLabel("Min valid keypoint fraction")

        self.spin_pose_export_min_valid_keypoints = QSpinBox()
        self.spin_pose_export_min_valid_keypoints.setRange(1, 50)
        self.spin_pose_export_min_valid_keypoints.setValue(3)
        self.spin_pose_export_min_valid_keypoints.setToolTip(
            "Minimum absolute count of valid keypoints required for a pose row to be accepted.\n"
            "Both this threshold and the fraction threshold must be satisfied."
        )
        self.lbl_pose_export_min_valid_keypoints = QLabel("Min valid keypoints (count)")
        self.pose_validity_row_widget = self._build_field_grid(
            [
                (
                    self.lbl_pose_export_min_valid_fraction,
                    self.spin_pose_export_min_valid_fraction,
                ),
                (
                    self.lbl_pose_export_min_valid_keypoints,
                    self.spin_pose_export_min_valid_keypoints,
                ),
            ],
            columns=2,
        )
        f_pose_quality.addRow(self.pose_validity_row_widget)

        self.spin_relink_min_pose_quality = QDoubleSpinBox()
        self.spin_relink_min_pose_quality.setRange(0.0, 1.0)
        self.spin_relink_min_pose_quality.setSingleStep(0.05)
        self.spin_relink_min_pose_quality.setDecimals(2)
        self.spin_relink_min_pose_quality.setValue(0.6)
        self.spin_relink_min_pose_quality.setToolTip(
            "Minimum pose quality score required at both endpoints of a candidate\n"
            "fragment pair before pose distance contributes to relinking score.\n"
            "Below this threshold, relinking uses motion-only scoring.\n"
            "Higher values (0.6+) are strongly recommended \u2014 motion-only relinking\n"
            "in dense scenes is likely to merge fragments from different animals."
        )
        self.lbl_relink_min_pose_quality = QLabel("Min pose quality for relinking")
        self.relink_quality_row_widget = self._build_field_grid(
            [
                (
                    self.lbl_relink_pose_max_distance,
                    self.spin_relink_pose_max_distance,
                ),
                (
                    self.lbl_relink_min_pose_quality,
                    self.spin_relink_min_pose_quality,
                ),
            ],
            columns=2,
        )
        f_relinking.addRow(self.relink_quality_row_widget)

        self.spin_pose_postproc_max_gap = QSpinBox()
        self.spin_pose_postproc_max_gap.setRange(0, 50)
        self.spin_pose_postproc_max_gap.setValue(5)
        self.spin_pose_postproc_max_gap.setToolTip(
            "Maximum consecutive missing frames to gap-fill via linear interpolation\n"
            "during pose temporal post-processing. Set to 0 to disable gap-filling."
        )
        self.lbl_pose_postproc_max_gap = QLabel("Pose postproc max gap (frames)")

        self.spin_pose_temporal_outlier_zscore = QDoubleSpinBox()
        self.spin_pose_temporal_outlier_zscore.setRange(0.0, 20.0)
        self.spin_pose_temporal_outlier_zscore.setSingleStep(0.5)
        self.spin_pose_temporal_outlier_zscore.setDecimals(1)
        self.spin_pose_temporal_outlier_zscore.setValue(3.0)
        self.spin_pose_temporal_outlier_zscore.setToolTip(
            "Rolling z-score threshold for temporal keypoint outlier suppression.\n"
            "Keypoint positions deviating beyond this threshold from their local\n"
            "rolling mean are zeroed. Set to 0.0 to disable. Recommended: 2.5\u20135.0."
        )
        self.lbl_pose_temporal_outlier_zscore = QLabel("Pose temporal outlier z-score")
        self.pose_temporal_row_widget = self._build_field_grid(
            [
                (
                    self.lbl_pose_postproc_max_gap,
                    self.spin_pose_postproc_max_gap,
                ),
                (
                    self.lbl_pose_temporal_outlier_zscore,
                    self.spin_pose_temporal_outlier_zscore,
                ),
            ],
            columns=2,
        )
        f_pose_quality.addRow(self.pose_temporal_row_widget)

        # Z-score based velocity breaking
        self.spin_max_velocity_zscore = QDoubleSpinBox()
        self.spin_max_velocity_zscore.setRange(0.0, 10.0)
        self.spin_max_velocity_zscore.setSingleStep(0.5)
        self.spin_max_velocity_zscore.setDecimals(1)
        self.spin_max_velocity_zscore.setValue(0.0)  # 0 = disabled
        self.spin_max_velocity_zscore.setToolTip(
            "Z-score threshold for velocity-based trajectory breaking (0 = disabled).\n"
            "Detects sudden, statistically anomalous velocity changes that often\n"
            "indicate identity swaps or tracking errors.\n\n"
            "Safeguards prevent false breaks when animals transition from rest to movement:\n"
            "\u2022 Only triggers on substantial velocities (>2 px/frame)\n"
            "\u2022 Uses regularized statistics to handle low-variability periods\n"
            "\u2022 Filters out stationary noise from baseline calculations\n\n"
            "Recommended: 3.0-5.0 for sensitive detection, 0 to disable."
        )
        self.lbl_max_velocity_zscore = QLabel("Velocity z-score threshold")

        self.spin_velocity_zscore_window = QDoubleSpinBox()
        self.spin_velocity_zscore_window.setRange(0.1, 5.0)
        self.spin_velocity_zscore_window.setSingleStep(0.1)
        self.spin_velocity_zscore_window.setDecimals(2)
        self.spin_velocity_zscore_window.setValue(0.33)
        self.spin_velocity_zscore_window.setToolTip(
            "Time window for z-score velocity calculation (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Larger windows = more stable statistics but less responsive.\n"
            "Smaller windows = more sensitive but may be noisy.\n"
            "Recommended: 0.3-0.7 s."
        )
        self.lbl_velocity_zscore_window = QLabel("Z-score window (seconds)")

        self.spin_velocity_zscore_min_vel = QDoubleSpinBox()
        self.spin_velocity_zscore_min_vel.setRange(0.1, 50.0)
        self.spin_velocity_zscore_min_vel.setSingleStep(0.5)
        self.spin_velocity_zscore_min_vel.setDecimals(1)
        self.spin_velocity_zscore_min_vel.setValue(2.0)
        self.spin_velocity_zscore_min_vel.setToolTip(
            "Minimum velocity for z-score breaking (body-sizes/second).\n"
            "Prevents false breaks when animal starts moving from stationary state.\n"
            "Z-score analysis only triggers when velocity exceeds this threshold.\n"
            "Automatically scaled by body size and frame rate.\n"
            "Recommended: 1.0-3.0 body-sizes/s depending on animal locomotion speed."
        )
        self.lbl_velocity_zscore_min_vel = QLabel(
            "Minimum speed for z-score check (body lengths/sec)"
        )
        self.motion_break_threshold_row_widget = self._build_field_grid(
            [
                (self.lbl_max_velocity_break, self.spin_max_velocity_break),
                (self.lbl_max_velocity_zscore, self.spin_max_velocity_zscore),
            ],
            columns=2,
        )
        f_motion_breaks.addRow(self.motion_break_threshold_row_widget)
        self.motion_break_support_row_widget = self._build_field_grid(
            [
                (self.lbl_velocity_zscore_window, self.spin_velocity_zscore_window),
                (
                    self.lbl_velocity_zscore_min_vel,
                    self.spin_velocity_zscore_min_vel,
                ),
            ],
            columns=2,
        )
        f_motion_breaks.addRow(self.motion_break_support_row_widget)

        # Interpolation settings
        self.combo_interpolation_method = QComboBox()
        self.combo_interpolation_method.addItems(["None", "Linear", "Cubic", "Spline"])
        self.combo_interpolation_method.setCurrentText("None")
        self.combo_interpolation_method.setToolTip(
            "Interpolation method for filling gaps in trajectories:\n"
            "\u2022 None: No interpolation (keep NaN values)\n"
            "\u2022 Linear: Simple linear interpolation\n"
            "\u2022 Cubic: Smooth cubic spline interpolation\n"
            "\u2022 Spline: Smoothing spline with automatic smoothing\n"
            "Applied to X, Y positions and heading (circular interpolation)."
        )
        self.lbl_interpolation_method = QLabel(
            "Which interpolation method should be used?"
        )

        self.spin_interpolation_max_gap = QDoubleSpinBox()
        self.spin_interpolation_max_gap.setRange(0.01, 10.0)
        self.spin_interpolation_max_gap.setSingleStep(0.1)
        self.spin_interpolation_max_gap.setDecimals(2)
        self.spin_interpolation_max_gap.setValue(0.33)
        self.spin_interpolation_max_gap.setToolTip(
            "Maximum gap duration to interpolate (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Gaps larger than this will remain as NaN.\n"
            "Prevents interpolation across large occlusions.\n"
            "Recommended: 0.15-0.50 s."
        )
        self.lbl_interpolation_max_gap = QLabel("Maximum interpolation gap (seconds)")
        self.interpolation_row_widget = self._build_field_grid(
            [
                (self.lbl_interpolation_method, self.combo_interpolation_method),
                (self.lbl_interpolation_max_gap, self.spin_interpolation_max_gap),
            ],
            columns=2,
        )
        f_interpolation_merge.addRow(self.interpolation_row_widget)

        self.spin_heading_flip_max_burst = QSpinBox()
        self.spin_heading_flip_max_burst.setRange(1, 50)
        self.spin_heading_flip_max_burst.setValue(5)
        self.spin_heading_flip_max_burst.setToolTip(
            "Maximum length (frames) of an isolated heading-flip burst that\n"
            "post-processing will correct. Contiguous runs of ~180\u00b0 flips\n"
            "shorter than this are assumed to be classifier errors and are\n"
            "reverted. Longer runs are kept as genuine orientation changes.\n"
            "Increase if brief real flips are being suppressed; decrease if\n"
            "extended flip artefacts survive. Recommended: 3\u201310."
        )
        self.lbl_heading_flip_max_burst = QLabel(
            "Max heading-flip burst to correct (frames)"
        )
        # Trajectory Merging Settings (Conservative Strategy)
        self.spin_merge_overlap_multiplier = QDoubleSpinBox()
        self.spin_merge_overlap_multiplier.setRange(0.1, 10.0)
        self.spin_merge_overlap_multiplier.setSingleStep(0.1)
        self.spin_merge_overlap_multiplier.setDecimals(2)
        self.spin_merge_overlap_multiplier.setValue(0.5)
        self.spin_merge_overlap_multiplier.setToolTip(
            "Agreement distance for merging forward/backward trajectories (\u00d7body size).\n"
            "Frames where both trajectories are within this distance are considered 'agreeing'.\n"
            "Disagreeing frames cause trajectory splits for conservative identity handling.\n"
            "Recommended: 0.3-0.7\u00d7 body size."
        )
        self.lbl_merge_overlap_multiplier = QLabel(
            "Merge agreement distance (body lengths)"
        )
        # Heading-flip burst row — hidden when post-hoc global consistency is active.
        self.heading_flip_burst_row_widget = self._build_field_grid(
            [(self.lbl_heading_flip_max_burst, self.spin_heading_flip_max_burst)],
            columns=1,
        )
        f_interpolation_merge.addRow(self.heading_flip_burst_row_widget)
        # Post-hoc note shown in place of the burst row.
        self.lbl_heading_flip_posthoc_note = self._main_window._create_help_label(
            "Global heading consistency (minimum-flips DP) is applied per track — "
            "the burst-flip threshold is not used."
        )
        self.lbl_heading_flip_posthoc_note.setVisible(False)
        f_interpolation_merge.addRow(self.lbl_heading_flip_posthoc_note)
        self.merge_threshold_row_widget = self._build_field_grid(
            [
                (
                    self.lbl_merge_overlap_multiplier,
                    self.spin_merge_overlap_multiplier,
                ),
            ],
            columns=1,
        )
        f_interpolation_merge.addRow(self.merge_threshold_row_widget)

        self.spin_min_overlap_frames = QSpinBox()
        self.spin_min_overlap_frames.setRange(1, 100)
        self.spin_min_overlap_frames.setValue(5)
        self.spin_min_overlap_frames.setToolTip(
            "Minimum agreeing frames required to consider trajectories as merge candidates.\n"
            "Forward/backward trajectory pairs need at least this many frames within\n"
            "the agreement distance to be merged. Higher = more conservative.\n"
            "Recommended: 5-15 frames."
        )
        self.lbl_min_overlap_frames = QLabel("Minimum overlap frames")
        self.min_overlap_row_widget = self._build_field_grid(
            [(self.lbl_min_overlap_frames, self.spin_min_overlap_frames)],
            columns=1,
        )
        f_interpolation_merge.addRow(self.min_overlap_row_widget)

        self.spin_identity_disagree_min_run = QSpinBox()
        self.spin_identity_disagree_min_run.setRange(0, 100)
        self.spin_identity_disagree_min_run.setValue(5)
        self.spin_identity_disagree_min_run.setToolTip(
            "Minimum consecutive frames of committed identity disagreement required\n"
            "to split a spatially-agreeing forward/backward trajectory pair.\n"
            "Forward and backward must both have committed (stable) identity labels\n"
            "that conflict for at least this many consecutive frames before a split\n"
            "is triggered. Set to 0 to split on every single disagreeing frame\n"
            "(more aggressive). Higher values suppress noise from brief identity\n"
            "uncertainty at segment boundaries. Recommended: 3-10 frames."
        )
        self.lbl_identity_disagree_min_run = QLabel("Identity split min run (frames)")
        self.identity_disagree_row_widget = self._build_field_grid(
            [(self.lbl_identity_disagree_min_run, self.spin_identity_disagree_min_run)],
            columns=1,
        )
        f_interpolation_merge.addRow(self.identity_disagree_row_widget)

        # Cleanup option
        self.chk_cleanup_temp_files = QCheckBox("Auto-cleanup temporary files")
        self.chk_cleanup_temp_files.setChecked(True)
        self.chk_cleanup_temp_files.setToolTip(
            "Automatically delete temporary files after successful tracking:\n"
            "\u2022 Intermediate CSV files (*_forward.csv, *_backward.csv)\n"
            "\u2022 Pose inference cache (posekit/ directory)\n"
            "Keeps only final merged/processed output files."
        )
        f_cleaning_filters.addRow("", self.chk_cleanup_temp_files)

        # --- Identity Post-Processing ---
        self.g_offline_identity = QGroupBox("Identity Post-Processing")
        self._main_window._set_compact_section_widget(self.g_offline_identity)
        vl_offline_identity = QVBoxLayout(self.g_offline_identity)
        vl_offline_identity.addWidget(
            self._main_window._create_help_label(
                "Choose how identity labels are assigned after tracking. "
                "Only takes effect when identity analysis is enabled.\n"
                "• None — no identity post-processing.\n"
                "• Slot Fill — vacancy-aware slot filling: resolves wholly-unknown trajectories "
                "using CNN evidence and spatial continuity.\n"
                "• Offline Decoder — Slot Fill first, then Bayesian HMM smoother + MILP global "
                "assignment for swap detection and final verification."
            )
        )
        self.cmb_identity_postprocess_mode = QComboBox()
        self.cmb_identity_postprocess_mode.addItems(
            ["None", "Slot Fill", "Offline Decoder"]
        )
        self.cmb_identity_postprocess_mode.setCurrentText("Slot Fill")
        self.cmb_identity_postprocess_mode.setToolTip(
            "None: skip identity post-processing entirely.\n"
            "Slot Fill: resolve wholly-unknown trajectories using CNN evidence and "
            "spatial continuity.\n"
            "Offline Decoder: run Slot Fill first, then the Bayesian HMM smoother "
            "and MILP global fragment assignment. Requires identity analysis to be configured."
        )
        self.cmb_identity_postprocess_mode.currentTextChanged.connect(
            self._on_offline_identity_toggled
        )
        vl_offline_identity.addWidget(self.cmb_identity_postprocess_mode)

        self.slot_fill_content = QWidget()
        slot_fill_layout = QFormLayout(self.slot_fill_content)
        slot_fill_layout.setContentsMargins(0, 4, 0, 0)
        slot_fill_layout.setSpacing(4)

        self.spin_slot_fill_min_score = QDoubleSpinBox()
        self.spin_slot_fill_min_score.setRange(0.0, 1.0)
        self.spin_slot_fill_min_score.setSingleStep(0.05)
        self.spin_slot_fill_min_score.setDecimals(2)
        self.spin_slot_fill_min_score.setValue(0.25)
        self.spin_slot_fill_min_score.setToolTip(
            "Minimum combined score (0–1) required to commit a slot-fill assignment.\n"
            "Higher values → fewer but more confident assignments.\n"
            "Recommended: 0.20–0.40."
        )
        slot_fill_layout.addRow("Slot-fill min score", self.spin_slot_fill_min_score)
        self.slot_fill_content.setVisible(True)
        vl_offline_identity.addWidget(self.slot_fill_content)

        self.offline_identity_content = QWidget()
        offline_identity_layout = QVBoxLayout(self.offline_identity_content)
        offline_identity_layout.setContentsMargins(0, 0, 0, 0)
        offline_identity_layout.setSpacing(8)

        self.g_offline_split, f_offline_split = self._create_subsection_form(
            "Trajectory Splitting",
            "Split trajectories where the smoothed identity posterior shows a sustained regime change.",
        )

        self.chk_identity_offline_split_trajectories = QCheckBox(
            "Allow HMM to split trajectories on sustained identity switches"
        )
        self.chk_identity_offline_split_trajectories.setChecked(
            bool(
                self._main_window.advanced_config.get(
                    "identity_offline_split_trajectories", False
                )
            )
        )
        self.chk_identity_offline_split_trajectories.setToolTip(
            "When disabled, the offline decoder smooths and assigns identities "
            "without rewriting TrajectoryID values."
        )
        f_offline_split.addRow("", self.chk_identity_offline_split_trajectories)

        self.spin_identity_offline_split_min_conf = QDoubleSpinBox()
        self.spin_identity_offline_split_min_conf.setRange(0.0, 1.0)
        self.spin_identity_offline_split_min_conf.setSingleStep(0.01)
        self.spin_identity_offline_split_min_conf.setDecimals(2)
        self.spin_identity_offline_split_min_conf.setValue(
            float(
                self._main_window.advanced_config.get(
                    "identity_offline_split_min_conf", 0.75
                )
            )
        )
        self.spin_identity_offline_split_min_conf.setToolTip(
            "Minimum smoothed posterior confidence required before offline splitting "
            "considers a label regime stable."
        )

        self.spin_identity_offline_split_min_margin = QDoubleSpinBox()
        self.spin_identity_offline_split_min_margin.setRange(0.0, 1.0)
        self.spin_identity_offline_split_min_margin.setSingleStep(0.01)
        self.spin_identity_offline_split_min_margin.setDecimals(2)
        self.spin_identity_offline_split_min_margin.setValue(
            float(
                self._main_window.advanced_config.get(
                    "identity_offline_split_min_margin", 0.2
                )
            )
        )
        self.spin_identity_offline_split_min_margin.setToolTip(
            "Required posterior margin between the top two labels before offline "
            "splitting treats a regime change as trustworthy."
        )

        self.split_conf_row = self._build_field_grid(
            [
                (QLabel("Min confidence"), self.spin_identity_offline_split_min_conf),
                (QLabel("Min margin"), self.spin_identity_offline_split_min_margin),
            ],
            columns=2,
        )
        f_offline_split.addRow(self.split_conf_row)

        self.spin_identity_offline_split_min_frames = QSpinBox()
        self.spin_identity_offline_split_min_frames.setRange(1, 1000)
        self.spin_identity_offline_split_min_frames.setValue(
            int(
                self._main_window.advanced_config.get(
                    "identity_offline_split_min_frames", 3
                )
            )
        )
        self.spin_identity_offline_split_min_frames.setToolTip(
            "Minimum consecutive frames needed on each side of an offline identity switch "
            "before the trajectory is split."
        )

        self.spin_identity_offline_split_max_bridge_frames = QSpinBox()
        self.spin_identity_offline_split_max_bridge_frames.setRange(0, 1000)
        self.spin_identity_offline_split_max_bridge_frames.setValue(
            int(
                self._main_window.advanced_config.get(
                    "identity_offline_split_max_bridge_frames", 6
                )
            )
        )
        self.spin_identity_offline_split_max_bridge_frames.setToolTip(
            "Maximum ambiguous bridge length tolerated between two strong offline identity "
            "regimes before splitting is suppressed."
        )

        self.split_span_row = self._build_field_grid(
            [
                (
                    QLabel("Min frames per side"),
                    self.spin_identity_offline_split_min_frames,
                ),
                (
                    QLabel("Max bridge frames"),
                    self.spin_identity_offline_split_max_bridge_frames,
                ),
            ],
            columns=2,
        )
        f_offline_split.addRow(self.split_span_row)

        self.g_offline_solver, f_offline_solver = self._create_subsection_form(
            "Global Solver",
            "MILP solver settings for the global fragment-to-identity assignment.",
        )

        self.spin_identity_offline_ilp_time_limit = QDoubleSpinBox()
        self.spin_identity_offline_ilp_time_limit.setRange(1.0, 3600.0)
        self.spin_identity_offline_ilp_time_limit.setSingleStep(1.0)
        self.spin_identity_offline_ilp_time_limit.setDecimals(1)
        self.spin_identity_offline_ilp_time_limit.setValue(
            float(
                self._main_window.advanced_config.get(
                    "identity_offline_ilp_time_limit", 30.0
                )
            )
        )
        self.spin_identity_offline_ilp_time_limit.setToolTip(
            "Wall-clock time limit in seconds for the offline global MILP fragment assignment solver."
        )

        self.spin_identity_offline_ilp_rel_gap = QDoubleSpinBox()
        self.spin_identity_offline_ilp_rel_gap.setRange(0.0, 1.0)
        self.spin_identity_offline_ilp_rel_gap.setSingleStep(0.0001)
        self.spin_identity_offline_ilp_rel_gap.setDecimals(6)
        self.spin_identity_offline_ilp_rel_gap.setValue(
            float(
                self._main_window.advanced_config.get(
                    "identity_offline_ilp_rel_gap", 1e-6
                )
            )
        )
        self.spin_identity_offline_ilp_rel_gap.setToolTip(
            "Relative optimality gap tolerated by the offline MILP solver before it stops early."
        )

        self.solver_row = self._build_field_grid(
            [
                (QLabel("Time limit (s)"), self.spin_identity_offline_ilp_time_limit),
                (QLabel("Rel. gap"), self.spin_identity_offline_ilp_rel_gap),
            ],
            columns=2,
        )
        f_offline_solver.addRow(self.solver_row)

        self.g_offline_respawn_prior, f_offline_respawn_prior = (
            self._create_subsection_form(
                "Respawn Prior",
                "Carry a slot's identity posterior forward when it respawns, decaying over time.",
            )
        )

        self.spin_identity_respawn_prior_strength = QDoubleSpinBox()
        self.spin_identity_respawn_prior_strength.setRange(0.0, 1.0)
        self.spin_identity_respawn_prior_strength.setSingleStep(0.01)
        self.spin_identity_respawn_prior_strength.setDecimals(2)
        self.spin_identity_respawn_prior_strength.setValue(
            float(
                self._main_window.advanced_config.get(
                    "identity_respawn_prior_strength", 0.75
                )
            )
        )
        self.spin_identity_respawn_prior_strength.setToolTip(
            "How strongly a recently lost slot's identity posterior seeds the next respawn of that slot."
        )

        self.spin_identity_respawn_prior_decay = QDoubleSpinBox()
        self.spin_identity_respawn_prior_decay.setRange(0.0, 1.0)
        self.spin_identity_respawn_prior_decay.setSingleStep(0.001)
        self.spin_identity_respawn_prior_decay.setDecimals(3)
        self.spin_identity_respawn_prior_decay.setValue(
            float(
                self._main_window.advanced_config.get(
                    "identity_respawn_prior_decay", 0.97
                )
            )
        )
        self.spin_identity_respawn_prior_decay.setToolTip(
            "Per-frame decay applied to the carried respawn prior as the slot stays absent."
        )

        self.spin_identity_respawn_prior_max_gap = QSpinBox()
        self.spin_identity_respawn_prior_max_gap.setRange(0, 10000)
        self.spin_identity_respawn_prior_max_gap.setValue(
            int(
                self._main_window.advanced_config.get(
                    "identity_respawn_prior_max_gap", 120
                )
            )
        )
        self.spin_identity_respawn_prior_max_gap.setToolTip(
            "Maximum gap in frames where a lost slot can still reuse its carried identity prior on respawn."
        )

        self.respawn_prior_row = self._build_field_grid(
            [
                (QLabel("Strength"), self.spin_identity_respawn_prior_strength),
                (QLabel("Decay"), self.spin_identity_respawn_prior_decay),
                (QLabel("Max gap (frames)"), self.spin_identity_respawn_prior_max_gap),
            ],
            columns=3,
        )
        f_offline_respawn_prior.addRow(self.respawn_prior_row)

        offline_identity_layout.addWidget(self.g_offline_split)
        offline_identity_layout.addWidget(self.g_offline_solver)
        offline_identity_layout.addWidget(self.g_offline_respawn_prior)
        vl_offline_identity.addWidget(self.offline_identity_content)
        self.offline_identity_content.setVisible(False)

        cleaning_sections_layout.addWidget(self.g_cleaning_filters)
        cleaning_sections_layout.addWidget(self.g_motion_breaks)
        cleaning_sections_layout.addWidget(self.g_relinking)
        cleaning_sections_layout.addWidget(self.g_pose_quality)
        cleaning_sections_layout.addWidget(self.g_interpolation_merge)
        cleaning_sections_layout.addWidget(self.g_offline_identity)
        vl_pp.addWidget(self.cleaning_sections_widget)
        vbox.addWidget(g_pp)

        # Video Export (from post-processed trajectories)
        g_video = QGroupBox("What export video should be created?")
        self._main_window._set_compact_section_widget(g_video)
        vl_video = QVBoxLayout(g_video)
        vl_video.addWidget(
            self._main_window._create_help_label(
                "Generate annotated video from final post-processed trajectories. "
                "Video is created AFTER merging and interpolation, showing clean tracks with stable IDs. "
                "This is recommended over real-time video output during tracking."
            )
        )

        self.check_video_output = QCheckBox("Export trajectory video")
        self.check_video_output.setChecked(False)
        self.check_video_output.toggled.connect(self._on_video_output_toggled)
        self.check_video_output.setToolTip(
            "Generate annotated video showing post-processed trajectories.\n"
            "Video is created from merged/interpolated tracks, not raw tracking.\n"
            "Shows clean, stable trajectories with final IDs.\n"
            "Recommended for publication and visualization."
        )
        vl_video.addWidget(self.check_video_output)

        self.video_export_content = QWidget()
        video_export_content_layout = QVBoxLayout(self.video_export_content)
        video_export_content_layout.setContentsMargins(0, 0, 0, 0)
        video_export_content_layout.setSpacing(8)

        self.g_video_destination, f_video_destination = self._create_subsection_form(
            "Output Destination",
            "Choose where the post-processed annotated video should be written.",
        )

        self.g_video_track_overlay, f_video_track_overlay = (
            self._create_subsection_form(
                "Track Overlay",
                "Control the visual style of track labels, markers, arrows, and trails in the exported video.",
            )
        )

        self.g_video_pose_overlay, f_video_pose_overlay = self._create_subsection_form(
            "Pose Overlay",
            "Optional pose overlay controls for keypoints and skeleton rendering in the exported video.",
        )

        self.btn_video_out = QPushButton("Select Video Output...")
        self.btn_video_out.clicked.connect(self.select_video_output)
        self.btn_video_out.setEnabled(False)
        self.video_out_line = QLineEdit()
        self.video_out_line.setPlaceholderText("Path for annotated video output")
        self.video_out_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_out_line.setEnabled(False)
        self.lbl_video_path = QLabel("Annotated video file")
        self.video_destination_row_widget = QWidget()
        video_destination_row_layout = QHBoxLayout(self.video_destination_row_widget)
        video_destination_row_layout.setContentsMargins(0, 0, 0, 0)
        video_destination_row_layout.setSpacing(6)
        video_destination_row_layout.addWidget(self.video_out_line, 1)
        video_destination_row_layout.addWidget(self.btn_video_out, 0)
        f_video_destination.addRow(
            self.lbl_video_path, self.video_destination_row_widget
        )

        self.check_show_labels = QCheckBox("Show Track IDs")
        self.check_show_labels.setChecked(True)
        self.check_show_labels.setToolTip(
            "Display trajectory ID labels next to each tracked animal."
        )

        self.check_show_orientation = QCheckBox("Show Orientation Arrows")
        self.check_show_orientation.setChecked(True)
        self.check_show_orientation.setToolTip(
            "Display arrows indicating heading direction."
        )

        self.check_show_trails = QCheckBox("Show Trajectory Trails")
        self.check_show_trails.setChecked(False)
        self.check_show_trails.setToolTip(
            "Display past trajectory path as a fading trail."
        )
        self.video_track_toggles_widget = self._build_checkbox_grid(
            [
                self.check_show_labels,
                self.check_show_orientation,
                self.check_show_trails,
            ],
            columns=2,
        )
        f_video_track_overlay.addRow(self.video_track_toggles_widget)

        self.spin_trail_duration = QDoubleSpinBox()
        self.spin_trail_duration.setRange(-1, 86400)
        self.spin_trail_duration.setSingleStep(1)
        self.spin_trail_duration.setDecimals(1)
        self.spin_trail_duration.setValue(1.0)
        self.spin_trail_duration.setToolTip(
            "Duration of trail history in seconds (0.1-10.0).\n"
            "Longer trails show more movement history.\n"
            "Automatically converted to frames using video FPS."
        )
        self.lbl_trail_duration = QLabel("Trail duration (seconds)")

        self.spin_marker_size = QDoubleSpinBox()
        self.spin_marker_size.setRange(0.1, 300.0)
        self.spin_marker_size.setSingleStep(0.1)
        self.spin_marker_size.setDecimals(1)
        self.spin_marker_size.setValue(0.3)
        self.spin_marker_size.setToolTip(
            "Size of position marker (0.1-5.0 \u00d7 body size).\n"
            "Scaled by reference body size for consistency."
        )
        self.lbl_marker_size = QLabel("Marker size (body lengths)")
        self.video_track_scale_row_widget = self._build_field_grid(
            [
                (self.lbl_trail_duration, self.spin_trail_duration),
                (self.lbl_marker_size, self.spin_marker_size),
            ],
            columns=2,
        )
        f_video_track_overlay.addRow(self.video_track_scale_row_widget)

        self.spin_text_scale = QDoubleSpinBox()
        self.spin_text_scale.setRange(0.3, 3.0)
        self.spin_text_scale.setSingleStep(0.1)
        self.spin_text_scale.setDecimals(1)
        self.spin_text_scale.setValue(0.5)
        self.spin_text_scale.setToolTip(
            "Scale factor for ID labels (0.3-3.0).\n" "Larger values = bigger text."
        )
        self.lbl_text_scale = QLabel("Text scale")

        self.spin_arrow_length = QDoubleSpinBox()
        self.spin_arrow_length.setRange(0.5, 10.0)
        self.spin_arrow_length.setSingleStep(0.5)
        self.spin_arrow_length.setDecimals(1)
        self.spin_arrow_length.setValue(0.7)
        self.spin_arrow_length.setToolTip(
            "Length of orientation arrow (0.5-10.0 \u00d7 body size).\n"
            "Scaled by reference body size."
        )
        self.lbl_arrow_length = QLabel("Arrow length (body lengths)")
        self.video_track_label_row_widget = self._build_field_grid(
            [
                (self.lbl_text_scale, self.spin_text_scale),
                (self.lbl_arrow_length, self.spin_arrow_length),
            ],
            columns=2,
        )
        f_video_track_overlay.addRow(self.video_track_label_row_widget)

        self.check_video_show_pose = QCheckBox("Show Pose Keypoints/Skeleton")
        self.check_video_show_pose.setChecked(
            bool(self._main_window.advanced_config.get("video_show_pose", True))
        )
        self.check_video_show_pose.setToolTip(
            "Overlay pose keypoints/skeleton in exported video.\n"
            "Requires pose inference to be enabled in Analyze Individuals."
        )
        self.check_video_show_pose.toggled.connect(
            self._main_window._sync_video_pose_overlay_controls
        )
        f_video_pose_overlay.addRow("", self.check_video_show_pose)

        self.combo_video_pose_color_mode = QComboBox()
        self.combo_video_pose_color_mode.addItems(["Track Color", "Fixed Color"])
        color_mode = str(
            self._main_window.advanced_config.get("video_pose_color_mode", "track")
        ).strip()
        self.combo_video_pose_color_mode.setCurrentIndex(
            0 if color_mode == "track" else 1
        )
        self.combo_video_pose_color_mode.setToolTip(
            "Pose color source for video overlay."
        )
        self.combo_video_pose_color_mode.currentIndexChanged.connect(
            self._main_window._sync_video_pose_overlay_controls
        )
        self.lbl_video_pose_color_mode = QLabel("Pose color mode")
        f_video_pose_overlay.addRow(
            self.lbl_video_pose_color_mode, self.combo_video_pose_color_mode
        )

        pose_color_row = QHBoxLayout()
        self.btn_video_pose_color = QPushButton()
        self.btn_video_pose_color.setMaximumWidth(60)
        self.btn_video_pose_color.setMinimumHeight(28)
        self.btn_video_pose_color.clicked.connect(self._select_video_pose_color)
        self.lbl_video_pose_color = QLabel("")
        pose_color_row.addWidget(self.btn_video_pose_color)
        pose_color_row.addWidget(self.lbl_video_pose_color)
        pose_color_cfg = self._main_window.advanced_config.get(
            "video_pose_color", [255, 255, 255]
        )
        if isinstance(pose_color_cfg, (list, tuple)) and len(pose_color_cfg) == 3:
            self._video_pose_color = tuple(
                int(max(0, min(255, float(v)))) for v in pose_color_cfg
            )
        else:
            self._video_pose_color = (255, 255, 255)
        # Update button appearance inline (panel not yet assigned on main_window at this point)
        b, g, r = self._video_pose_color
        self.btn_video_pose_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self.lbl_video_pose_color.setText(f"{self._video_pose_color}")
        self.lbl_video_pose_color_label = QLabel("Fixed pose color (BGR)")
        f_video_pose_overlay.addRow(self.lbl_video_pose_color_label, pose_color_row)

        self.spin_video_pose_point_radius = QSpinBox()
        self.spin_video_pose_point_radius.setRange(1, 20)
        self.spin_video_pose_point_radius.setValue(
            int(self._main_window.advanced_config.get("video_pose_point_radius", 3))
        )
        self.spin_video_pose_point_radius.setToolTip(
            "Radius of rendered pose keypoints in pixels."
        )
        self.lbl_video_pose_point_radius = QLabel("Pose keypoint radius (px)")
        f_video_pose_overlay.addRow(
            self.lbl_video_pose_point_radius, self.spin_video_pose_point_radius
        )

        self.spin_video_pose_point_thickness = QSpinBox()
        self.spin_video_pose_point_thickness.setRange(-1, 10)
        self.spin_video_pose_point_thickness.setValue(
            int(self._main_window.advanced_config.get("video_pose_point_thickness", -1))
        )
        self.spin_video_pose_point_thickness.setToolTip(
            "Keypoint circle thickness (-1 fills circles)."
        )
        self.lbl_video_pose_point_thickness = QLabel("Pose keypoint thickness")
        f_video_pose_overlay.addRow(
            self.lbl_video_pose_point_thickness, self.spin_video_pose_point_thickness
        )

        self.spin_video_pose_line_thickness = QSpinBox()
        self.spin_video_pose_line_thickness.setRange(1, 12)
        self.spin_video_pose_line_thickness.setValue(
            int(self._main_window.advanced_config.get("video_pose_line_thickness", 2))
        )
        self.spin_video_pose_line_thickness.setToolTip(
            "Skeleton edge line thickness in pixels."
        )
        self.lbl_video_pose_line_thickness = QLabel("Pose skeleton thickness (px)")
        f_video_pose_overlay.addRow(
            self.lbl_video_pose_line_thickness, self.spin_video_pose_line_thickness
        )

        self.lbl_video_pose_disabled_hint = QLabel(
            "Enable Pose Extraction in Analyze Individuals to edit pose overlay settings."
        )
        self.lbl_video_pose_disabled_hint.setWordWrap(True)
        self.lbl_video_pose_disabled_hint.setStyleSheet(
            "color: #a9b7c6; font-size: 11px;"
        )
        f_video_pose_overlay.addRow("", self.lbl_video_pose_disabled_hint)

        video_export_content_layout.addWidget(self.g_video_destination)
        video_export_content_layout.addWidget(self.g_video_track_overlay)
        video_export_content_layout.addWidget(self.g_video_pose_overlay)
        vl_video.addWidget(self.video_export_content)
        vbox.addWidget(g_video)

        self.video_export_content.setVisible(False)
        self.lbl_video_pose_color_mode.setVisible(False)
        self.combo_video_pose_color_mode.setVisible(False)
        self.lbl_video_pose_color_label.setVisible(False)
        self.btn_video_pose_color.setVisible(False)
        self.lbl_video_pose_color.setVisible(False)
        self.lbl_video_pose_point_radius.setVisible(False)
        self.spin_video_pose_point_radius.setVisible(False)
        self.lbl_video_pose_point_thickness.setVisible(False)
        self.spin_video_pose_point_thickness.setVisible(False)
        self.lbl_video_pose_line_thickness.setVisible(False)
        self.spin_video_pose_line_thickness.setVisible(False)
        self.lbl_video_pose_disabled_hint.setVisible(False)

        # RefineKit completion prompt
        self.g_refinekit = QGroupBox("Interactive Proofreading")
        self._main_window._set_compact_section_widget(self.g_refinekit)
        vl_refinekit = QVBoxLayout(self.g_refinekit)
        vl_refinekit.addWidget(
            self._main_window._create_help_label(
                "Prompt to open completed tracking results in RefineKit for "
                "interactive identity proofreading and swap correction. "
                "This is available only for single-video tracking runs."
            )
        )
        self.chk_prompt_open_refinekit = QCheckBox(
            "Prompt to open RefineKit when tracking completes"
        )
        self.chk_prompt_open_refinekit.setChecked(False)
        self.chk_prompt_open_refinekit.setToolTip(
            "Ask whether to open the current results in RefineKit after a "
            "single-video tracking run finishes."
        )
        vl_refinekit.addWidget(self.chk_prompt_open_refinekit)
        vbox.addWidget(self.g_refinekit)

        self._set_cleaning_section_state(self.enable_postprocessing.isChecked())
        self._set_video_output_section_state(self.check_video_output.isChecked())

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    def set_batch_mode_active(self, active: bool) -> None:
        """Hide RefineKit prompting during batch runs and clear any pending prompt."""
        if active:
            self.chk_prompt_open_refinekit.setChecked(False)
        self.g_refinekit.setVisible(not active)
        self.g_refinekit.setEnabled(not active)

    def _create_subsection_form(
        self,
        title: str,
        help_text: str | None = None,
    ) -> tuple[QGroupBox, QFormLayout]:
        """Create a compact subsection with an optional help label and a form layout."""
        group = QGroupBox(title)
        self._main_window._set_compact_section_widget(group)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        if help_text:
            layout.addWidget(
                self._main_window._create_help_label(
                    help_text,
                    attach_to_title=True,
                )
            )
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout.addLayout(form)
        return group, form

    @staticmethod
    def _build_field_grid(
        fields: list[tuple[QLabel, QWidget]],
        columns: int = 2,
    ) -> QWidget:
        """Arrange labeled fields as compact vertical cells in a grid."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for index, (label, field_widget) in enumerate(fields):
            row = index // columns
            column = index % columns
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            label.setStyleSheet("color: #cccccc;")
            label.setWordWrap(True)
            cell_layout.addWidget(label)
            cell_layout.addWidget(field_widget)
            grid.addWidget(cell, row, column)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        return widget

    @staticmethod
    def _build_checkbox_grid(checkboxes: list[QCheckBox], columns: int = 2) -> QWidget:
        """Arrange related checkboxes in a compact multi-column grid."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        for index, checkbox in enumerate(checkboxes):
            row = index // columns
            column = index % columns
            grid.addWidget(checkbox, row, column)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        return widget

    def _set_cleaning_section_state(self, enabled: bool) -> None:
        """Show or hide post-processing subsections based on the cleaning toggle."""
        pose_enabled = bool(enabled and self._main_window._is_pose_export_enabled())
        self.cleaning_sections_widget.setVisible(enabled)
        self.cleaning_sections_widget.setEnabled(enabled)
        self.g_cleaning_filters.setVisible(enabled)
        self.g_cleaning_filters.setEnabled(enabled)
        self.g_motion_breaks.setVisible(enabled)
        self.g_motion_breaks.setEnabled(enabled)
        self.g_relinking.setVisible(enabled)
        self.g_relinking.setEnabled(enabled)
        self.g_interpolation_merge.setVisible(enabled)
        self.g_interpolation_merge.setEnabled(enabled)
        self.g_pose_quality.setVisible(pose_enabled)
        self.g_pose_quality.setEnabled(pose_enabled)
        self.g_offline_identity.setVisible(enabled)
        self.g_offline_identity.setEnabled(enabled)
        if not enabled:
            self.offline_identity_content.setVisible(False)

    def _set_video_output_section_state(self, checked: bool) -> None:
        """Show or hide the grouped video-export controls."""
        self.video_export_content.setVisible(checked)
        self.video_export_content.setEnabled(checked)
        self.g_video_destination.setVisible(checked)
        self.g_video_track_overlay.setVisible(checked)
        self.g_video_pose_overlay.setVisible(checked)
        self.btn_video_out.setEnabled(checked)
        self.video_out_line.setEnabled(checked)

    # =========================================================================
    # HANDLER METHODS (moved from MainWindow)
    # =========================================================================

    def _on_cleaning_toggled(self, state):
        """Enable/disable trajectory cleaning controls based on checkbox."""
        enabled = self.enable_postprocessing.isChecked()
        self._set_cleaning_section_state(enabled)

    def _on_offline_identity_toggled(self, _=None):
        """Show or hide slot-fill and offline decoder sub-sections."""
        mode = self.cmb_identity_postprocess_mode.currentText()
        self.slot_fill_content.setVisible(mode in ("Slot Fill", "Offline Decoder"))
        self.offline_identity_content.setVisible(mode == "Offline Decoder")

    def sync_heading_flip_posthoc_ui(self, posthoc_active: bool) -> None:
        """Toggle the heading-flip burst control vs. the post-hoc note.

        Args:
            posthoc_active: True when head-tail or pose model is configured so
                global DP consistency is applied instead of burst correction.
        """
        self.heading_flip_burst_row_widget.setVisible(not posthoc_active)
        self.heading_flip_burst_row_widget.setEnabled(not posthoc_active)
        self.lbl_heading_flip_posthoc_note.setVisible(posthoc_active)

    def _on_video_output_toggled(self, checked):
        """Enable/disable video output controls."""
        self._set_video_output_section_state(checked)
        self._main_window._sync_video_pose_overlay_controls()

    def select_video_output(self) -> None:
        """select_video_output method documentation."""
        fp, _ = QFileDialog.getSaveFileName(
            self, "Select Video Output", "", "Video Files (*.mp4 *.avi)"
        )
        if fp:
            self.video_out_line.setText(fp)

    def _select_video_pose_color(self):
        """Open color picker for fixed pose overlay color (BGR)."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        b, g, r = self._video_pose_color
        initial_color = QColor(r, g, b)
        color = QColorDialog.getColor(initial_color, self, "Choose Pose Overlay Color")
        if color.isValid():
            self._video_pose_color = (
                color.blue(),
                color.green(),
                color.red(),
            )
            self._update_video_pose_color_button()

    def _update_video_pose_color_button(self):
        """Update fixed pose-color preview button and text label."""
        b, g, r = self._video_pose_color
        self.btn_video_pose_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self.lbl_video_pose_color.setText(f"{self._video_pose_color}")
