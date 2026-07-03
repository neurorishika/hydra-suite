"""IdentityPanel — identity classification, pose analysis, and keypoint config."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.utils.batch_policy import (
    clamp_realtime_individual_batch_size,
    should_warn_for_padding_waste,
)

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class IdentityPanel(QWidget):
    """CNN/appearance identity assignment and pose backend configuration."""

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
        """Populate the panel layout."""
        layout = self._layout
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(8)
        self._main_window._set_compact_scroll_layout(form)

        self.lbl_individual_yolo_only_notice = self._main_window._create_help_label(
            "Individual analysis requires YOLO OBB mode.\n"
            "Switch detection method to YOLO OBB to enable this pipeline.",
            attach_to_title=False,
        )
        self.lbl_individual_yolo_only_notice.setVisible(False)
        form.addWidget(self.lbl_individual_yolo_only_notice)

        # Identity Classification Section
        self.g_identity = QGroupBox("Enable Identity Classification")
        self.g_identity.setCheckable(True)
        self.g_identity.setChecked(False)
        self.g_identity.toggled.connect(self._on_identity_analysis_toggled)
        self._main_window._set_compact_section_widget(self.g_identity)
        vl_identity = QVBoxLayout(self.g_identity)
        self.identity_content = QWidget()
        self.identity_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        identity_content_layout = QVBoxLayout(self.identity_content)
        identity_content_layout.setContentsMargins(0, 0, 0, 0)
        identity_content_layout.setSpacing(8)
        self.lbl_identity_help = self._main_window._create_help_label(
            "Classify individual identity during tracking. Extracts crops around each detection "
            "and processes them with the selected method."
        )
        identity_content_layout.addWidget(self.lbl_identity_help)

        # Legacy registry banner — shown when pre-v2 entries are detected.
        self.lbl_legacy_banner = QLabel()
        self.lbl_legacy_banner.setWordWrap(True)
        self.lbl_legacy_banner.setStyleSheet(
            "color: #f48771; font-size: 11px; padding: 2px 0;"
        )
        self.lbl_legacy_banner.setVisible(False)
        identity_content_layout.addWidget(self.lbl_legacy_banner)

        # Hidden legacy widgets (referenced by model-import dialog)
        self.line_color_tag_model = QLineEdit()
        self.line_color_tag_model.setPlaceholderText("path/to/color_tag_model.pt")
        self.line_color_tag_model.setVisible(False)
        self.spin_color_tag_conf = QDoubleSpinBox()
        self.spin_color_tag_conf.setRange(0.01, 1.0)
        self.spin_color_tag_conf.setValue(0.5)
        self.spin_color_tag_conf.setSingleStep(0.05)
        self.spin_color_tag_conf.setToolTip(
            "Minimum confidence for color tag detection"
        )
        self.spin_color_tag_conf.setVisible(False)

        # --- AprilTags group ---
        self.g_apriltags = QGroupBox("AprilTags")
        self.g_apriltags.setCheckable(True)
        self.g_apriltags.setChecked(False)
        self.g_apriltags.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vl_apriltags_outer = QVBoxLayout(self.g_apriltags)
        vl_apriltags_outer.setContentsMargins(0, 0, 0, 0)
        self.apriltag_settings_widget = QWidget()
        self.apriltag_settings_widget.setVisible(False)
        fl_apriltags = QFormLayout(self.apriltag_settings_widget)
        fl_apriltags.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.combo_apriltag_family = QComboBox()
        self.combo_apriltag_family.addItems(self._get_apriltag_families())
        self.combo_apriltag_family.setToolTip(
            "AprilTag family to detect.\n"
            "The list is populated from your installed apriltag library."
        )
        self.spin_apriltag_decimate = QDoubleSpinBox()
        self.spin_apriltag_decimate.setRange(1.0, 4.0)
        self.spin_apriltag_decimate.setValue(1.0)
        self.spin_apriltag_decimate.setSingleStep(0.5)
        self.spin_apriltag_decimate.setToolTip(
            "Decimation factor for faster detection (higher = faster but less accurate)"
        )
        self.apriltag_row_widget = self._build_inline_fields_row(
            [
                ("Family", self.combo_apriltag_family, 1),
                ("Downsampling", self.spin_apriltag_decimate, 0),
            ]
        )
        fl_apriltags.addRow("AprilTag settings", self.apriltag_row_widget)
        vl_apriltags_outer.addWidget(self.apriltag_settings_widget)
        self.g_apriltags.toggled.connect(self.apriltag_settings_widget.setVisible)
        identity_content_layout.addWidget(self.g_apriltags)

        # --- CNN Classifiers group ---
        self.g_cnn_classifiers = QGroupBox("CNN Classifiers")
        self._main_window._set_compact_section_widget(self.g_cnn_classifiers)
        vl_cnn = QVBoxLayout(self.g_cnn_classifiers)
        vl_cnn.setSpacing(4)
        self.btn_add_cnn_classifier = QPushButton("\uff0b Add CNN Classifier")
        self.btn_add_cnn_classifier.clicked.connect(self._add_cnn_classifier_row)
        vl_cnn.addWidget(self.btn_add_cnn_classifier)
        self.lbl_individual_batch_notice = QLabel("")
        self.lbl_individual_batch_notice.setWordWrap(True)
        self.lbl_individual_batch_notice.setStyleSheet(
            "color: #d7ba7d; font-size: 11px; padding-top: 2px;"
        )
        self.lbl_individual_batch_notice.setVisible(False)
        vl_cnn.addWidget(self.lbl_individual_batch_notice)
        self.cnn_scroll_area = QScrollArea()
        self.cnn_scroll_area.setWidgetResizable(True)
        self.cnn_scroll_area.setFrameShape(QFrame.NoFrame)
        self.cnn_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.cnn_scroll_area.setMinimumHeight(0)
        self.cnn_scroll_area.setMaximumHeight(200)
        self.cnn_scroll_area.setVisible(False)
        cnn_scroll_widget = QWidget()
        cnn_scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.cnn_rows_layout = QVBoxLayout(cnn_scroll_widget)
        self.cnn_rows_layout.setSpacing(6)
        self.cnn_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.cnn_scroll_area.setWidget(cnn_scroll_widget)
        vl_cnn.addWidget(self.cnn_scroll_area)
        identity_content_layout.addWidget(self.g_cnn_classifiers)

        # Hidden sentinel combo for model import dialog (reused by CNNClassifierRow)
        self.combo_cnn_identity_model = QComboBox()
        self.combo_cnn_identity_model.setVisible(False)
        # Note: combo will be populated post-construction via _refresh_cnn_identity_model_combo
        # Hidden verification labels (may be queried by model import dialog)
        self.lbl_cnn_arch = QLabel("—")
        self.lbl_cnn_num_classes = QLabel("—")
        self.lbl_cnn_class_names = QLabel("—")
        self.lbl_cnn_input_size = QLabel("—")
        self.lbl_cnn_label = QLabel("—")
        self.lbl_cnn_recommended_confidence = QLabel("—")
        self.spin_cnn_confidence = QDoubleSpinBox()
        self.spin_cnn_confidence.setRange(0.0, 1.0)
        self.spin_cnn_confidence.setSingleStep(0.05)
        self.spin_cnn_confidence.setValue(0.5)
        self.spin_cnn_window = QSpinBox()
        self.spin_cnn_window.setRange(1, 100)
        self.spin_cnn_window.setValue(10)
        vl_identity.addWidget(self.identity_content)

        self.g_individual_pipeline_common = QGroupBox(
            "Individual Analysis Pipeline Settings"
        )
        self._main_window._set_compact_section_widget(self.g_individual_pipeline_common)
        fl_common = QFormLayout(self.g_individual_pipeline_common)

        self.chk_individual_interpolate = QCheckBox(
            "Interpolate Occluded Frames After Tracking"
        )
        self.chk_individual_interpolate.setChecked(True)
        self.chk_individual_interpolate.setToolTip(
            "After tracking completes, fill occluded gaps by interpolating center/size/angle\n"
            "and generate additional masked crops. Interpolated crops are prefixed with 'interp_'."
        )
        fl_common.addRow("Interpolate occluded frames", self.chk_individual_interpolate)

        self.spin_individual_padding = QDoubleSpinBox()
        self.spin_individual_padding.setRange(0.0, 5.0)
        self.spin_individual_padding.setValue(0.1)
        self.spin_individual_padding.setSingleStep(0.05)
        self.spin_individual_padding.setDecimals(2)
        self.spin_individual_padding.setToolTip(
            "Padding around OBB bounding box as fraction of size.\n"
            "0.1 = 10% padding on each side.\n"
            "Applies to all precompute phases: pose, AprilTag, and CNN identity."
        )
        fl_common.addRow(
            "Crop padding fraction (all phases)", self.spin_individual_padding
        )

        bg_color_layout = QHBoxLayout()
        self.btn_background_color = QPushButton()
        self.btn_background_color.setMaximumWidth(60)
        self.btn_background_color.setMinimumHeight(30)
        self.btn_background_color.setToolTip("Click to choose background color")
        self.btn_background_color.clicked.connect(
            self._main_window._select_individual_background_color
        )
        self._background_color = (0, 0, 0)  # BGR
        bg_color_layout.addWidget(self.btn_background_color)

        self.btn_median_color = QPushButton("Use Median from Frame")
        self.btn_median_color.setToolTip(
            "Compute median color from the preview frame and use as background"
        )
        self.btn_median_color.clicked.connect(
            self._main_window._compute_median_background_color
        )
        bg_color_layout.addWidget(self.btn_median_color)

        self.lbl_background_color = QLabel("(0, 0, 0)")
        self.lbl_background_color.setToolTip("Current background color in BGR format")
        bg_color_layout.addWidget(self.lbl_background_color)
        # Note: background color button will be updated post-construction
        fl_common.addRow("Crop background color", bg_color_layout)

        self.chk_suppress_foreign_obb = QCheckBox("Suppress foreign animal regions")
        self.chk_suppress_foreign_obb.setChecked(True)
        self.chk_suppress_foreign_obb.setToolTip(
            "Fill overlapping animals' OBB areas with the background color before\n"
            "pose inference, and zero out any keypoints that fall inside another\n"
            "animal's bounding box after inference.\n"
            "\n"
            "Prevents pose contamination when animals are close or overlapping.\n"
            "Disable only if you observe incorrect masking of valid keypoints."
        )
        fl_common.addRow(
            "Pose contamination suppression", self.chk_suppress_foreign_obb
        )

        form.addWidget(self.g_individual_pipeline_common)

        self.g_headtail = QGroupBox("Enable Head-Tail Orientation")
        self.g_headtail.setCheckable(True)
        self.g_headtail.setChecked(False)
        self.g_headtail.toggled.connect(self._on_headtail_analysis_toggled)
        self._main_window._set_compact_section_widget(self.g_headtail)
        vl_headtail = QVBoxLayout(self.g_headtail)
        self.headtail_content = QWidget()
        self.headtail_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        headtail_content_layout = QVBoxLayout(self.headtail_content)
        headtail_content_layout.setContentsMargins(0, 0, 0, 0)
        headtail_content_layout.setSpacing(8)
        self.lbl_headtail_help = self._main_window._create_help_label(
            "Optional orientation classifier that resolves head vs. tail along the OBB major axis."
        )
        headtail_content_layout.addWidget(self.lbl_headtail_help)
        fl_headtail = QFormLayout()
        fl_headtail.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.combo_yolo_headtail_model = QComboBox()
        # Note: combo_yolo_headtail_model will be populated post-construction
        self.combo_yolo_headtail_model.activated.connect(
            self._main_window.on_yolo_headtail_model_changed
        )
        self.combo_yolo_headtail_model.currentIndexChanged.connect(
            lambda _index: self._sync_headtail_model_remove_button()
        )
        self.combo_yolo_headtail_model.setFixedHeight(30)
        self.combo_yolo_headtail_model.setToolTip(
            "Optional classifier to resolve head vs. tail orientation along the OBB major axis.\n"
            "Runs during tracking and post-hoc individual analysis."
        )
        self.btn_remove_yolo_headtail_model = self._create_model_remove_button(
            "Remove the selected head-tail model from the local repository."
        )
        self.btn_remove_yolo_headtail_model.clicked.connect(
            lambda: self._main_window._handle_remove_selected_yolo_model(
                combo=self.combo_yolo_headtail_model,
                refresh_callback=self._refresh_yolo_headtail_model_combo,
                selection_callback=self._main_window._set_yolo_headtail_model_selection,
                model_kind="head-tail model",
            )
        )

        self.headtail_model_row_widget = QWidget()
        _headtail_row = QHBoxLayout(self.headtail_model_row_widget)
        _headtail_row.setContentsMargins(0, 0, 0, 0)
        _headtail_row.setSpacing(4)
        _headtail_row.addWidget(self.combo_yolo_headtail_model, 1)
        _headtail_row.addWidget(self.btn_remove_yolo_headtail_model, 0)
        fl_headtail.addRow("Head-tail model", self.headtail_model_row_widget)

        self.spin_yolo_headtail_conf = QDoubleSpinBox()
        self.spin_yolo_headtail_conf.setRange(0.0, 1.0)
        self.spin_yolo_headtail_conf.setSingleStep(0.01)
        self.spin_yolo_headtail_conf.setValue(0.50)
        self.spin_yolo_headtail_conf.setFixedHeight(30)
        self.spin_yolo_headtail_conf.setToolTip(
            "Minimum classifier confidence for a head-tail assignment to be accepted (0–1).\n"
            "Lower = more assignments accepted; higher = fewer but more reliable."
        )
        fl_headtail.addRow(
            "Head-tail min classifier confidence", self.spin_yolo_headtail_conf
        )

        self.spin_yolo_headtail_detect_conf = QDoubleSpinBox()
        self.spin_yolo_headtail_detect_conf.setRange(0.0, 1.0)
        self.spin_yolo_headtail_detect_conf.setSingleStep(0.01)
        self.spin_yolo_headtail_detect_conf.setValue(0.25)
        self.spin_yolo_headtail_detect_conf.setFixedHeight(30)
        self.spin_yolo_headtail_detect_conf.setToolTip(
            "Minimum detector confidence required before a detection is passed to head-tail inference (0–1).\n"
            "Detections below this threshold are not re-run and remain unknown."
        )
        fl_headtail.addRow(
            "Head-tail min detection confidence",
            self.spin_yolo_headtail_detect_conf,
        )

        self.spin_headtail_batch = QSpinBox()
        self.spin_headtail_batch.setRange(1, 256)
        self.spin_headtail_batch.setValue(
            int(self._main_window.advanced_config.get("headtail_batch_size", 64))
        )
        self.spin_headtail_batch.setFixedHeight(30)
        self.spin_headtail_batch.setToolTip(
            "Maximum number of canonical crops per head-tail inference call.\n"
            "Non-realtime tracking batches crops across detections and frames, then chunks them by this size."
        )
        fl_headtail.addRow("Batch size", self.spin_headtail_batch)
        self.spin_headtail_batch.valueChanged.connect(
            self._sync_realtime_individual_batch_ui
        )

        self.chk_pose_overrides_headtail = QCheckBox(
            "Pose orientation overrides head-tail"
        )
        self.chk_pose_overrides_headtail.setChecked(True)
        self.chk_pose_overrides_headtail.setToolTip(
            "When enabled, valid pose heading takes precedence over head-tail heading."
        )
        fl_headtail.addRow("", self.chk_pose_overrides_headtail)

        headtail_content_layout.addLayout(fl_headtail)
        vl_headtail.addWidget(self.headtail_content)

        form.addWidget(self.g_headtail)
        form.addWidget(self.g_identity)

        self.g_pose_runtime = QGroupBox("Enable Pose Extraction")
        self.g_pose_runtime.setCheckable(True)
        self.g_pose_runtime.setChecked(False)
        self.g_pose_runtime.toggled.connect(self._on_pose_analysis_toggled)
        self.g_pose_runtime.toggled.connect(
            self._main_window._sync_video_pose_overlay_controls
        )
        self.g_pose_runtime.toggled.connect(
            self._main_window._on_runtime_context_changed
        )
        # Cross-panel connection wired in MainWindow.init_ui() after both panels exist
        self.chk_enable_pose_extractor = self.g_pose_runtime
        self._main_window._set_compact_section_widget(self.g_pose_runtime)
        vl_pose = QVBoxLayout(self.g_pose_runtime)
        self.pose_runtime_content = QWidget()
        self.pose_runtime_content.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Maximum
        )
        pose_runtime_content_layout = QVBoxLayout(self.pose_runtime_content)
        pose_runtime_content_layout.setContentsMargins(0, 0, 0, 0)
        pose_runtime_content_layout.setSpacing(8)
        self.lbl_pose_runtime_help = self._main_window._create_help_label(
            "Minimum runtime pose settings used by the individual-analysis pipeline."
        )
        pose_runtime_content_layout.addWidget(self.lbl_pose_runtime_help)
        fl_pose = QFormLayout(None)
        fl_pose.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form_pose_runtime = fl_pose

        self.combo_pose_model_type = QComboBox()
        self.combo_pose_model_type.addItems(["YOLO", "SLEAP", "ViTPose"])
        self.combo_pose_model_type.setToolTip("Pose backend for individual analysis.")
        self.combo_pose_model_type.currentIndexChanged.connect(
            self._main_window._sync_pose_backend_ui
        )

        self.combo_pose_model = QComboBox()
        self.combo_pose_model.setToolTip(
            "Pose model for individual analysis.\n"
            "Choose an imported model or select '＋ Add New Model…' to browse and import."
        )
        self.combo_pose_model.currentIndexChanged.connect(
            self._main_window.on_pose_model_changed
        )
        self.combo_pose_model.currentIndexChanged.connect(
            lambda _index: self._sync_pose_model_remove_button()
        )
        # Note: combo_pose_model will be populated post-construction
        self.btn_remove_pose_model = self._create_model_remove_button(
            "Remove the selected pose model from the local repository."
        )
        self.btn_remove_pose_model.clicked.connect(
            self._main_window._handle_remove_selected_pose_model
        )
        self.pose_model_row_widget = QWidget()
        pose_model_row_layout = QHBoxLayout(self.pose_model_row_widget)
        pose_model_row_layout.setContentsMargins(0, 0, 0, 0)
        pose_model_row_layout.setSpacing(4)
        pose_model_row_layout.addWidget(self.combo_pose_model, 1)
        pose_model_row_layout.addWidget(self.btn_remove_pose_model, 0)
        self.pose_model_inline_row_widget = self._build_inline_fields_row(
            [
                ("Type", self.combo_pose_model_type, 0),
                ("Model", self.pose_model_row_widget, 1),
            ]
        )
        fl_pose.addRow("Pose model", self.pose_model_inline_row_widget)

        self.spin_pose_min_kpt_conf_valid = QDoubleSpinBox()
        self.spin_pose_min_kpt_conf_valid.setRange(0.0, 1.0)
        self.spin_pose_min_kpt_conf_valid.setSingleStep(0.05)
        self.spin_pose_min_kpt_conf_valid.setDecimals(2)
        self.spin_pose_min_kpt_conf_valid.setValue(0.2)
        self.spin_pose_min_kpt_conf_valid.setToolTip(
            "Minimum per-keypoint confidence to consider a keypoint valid."
        )

        self.spin_pose_batch = QSpinBox()
        self.spin_pose_batch.setRange(1, 256)
        self.spin_pose_batch.setValue(
            int(self._main_window.advanced_config.get("pose_batch_size", 4))
        )
        self.spin_pose_batch.setToolTip(
            "Shared batch size for pose inference across YOLO and SLEAP backends."
        )
        self.pose_runtime_thresholds_row_widget = self._build_inline_fields_row(
            [
                ("Min keypoint confidence", self.spin_pose_min_kpt_conf_valid, 0),
                ("Pose batch size", self.spin_pose_batch, 0),
            ]
        )
        fl_pose.addRow("Pose inference", self.pose_runtime_thresholds_row_widget)
        self.spin_pose_batch.valueChanged.connect(
            self._sync_realtime_individual_batch_ui
        )

        h_pose_skeleton = QHBoxLayout()
        self.line_pose_skeleton_file = QLineEdit()
        self.line_pose_skeleton_file.setPlaceholderText(
            "Select skeleton JSON (keypoint_names + skeleton_edges)..."
        )
        default_skeleton = str(
            self._main_window.advanced_config.get("pose_skeleton_file", "")
        ).strip()
        if not default_skeleton:
            from hydra_suite.paths import get_skeleton_dir

            candidate = get_skeleton_dir() / "ooceraea_biroi.json"
            if candidate.exists():
                default_skeleton = str(candidate)
        if default_skeleton:
            self.line_pose_skeleton_file.setText(default_skeleton)
        self.btn_browse_pose_skeleton_file = QPushButton("Browse...")
        self.btn_browse_pose_skeleton_file.clicked.connect(
            self._main_window._select_pose_skeleton_file
        )
        self.line_pose_skeleton_file.textChanged.connect(
            self._refresh_pose_direction_keypoint_lists
        )
        h_pose_skeleton.addWidget(self.line_pose_skeleton_file)
        h_pose_skeleton.addWidget(self.btn_browse_pose_skeleton_file)
        fl_pose.addRow("Skeleton file", h_pose_skeleton)

        self.list_pose_ignore_keypoints = QListWidget()
        self.list_pose_ignore_keypoints.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_ignore_keypoints.setMinimumHeight(96)
        self.list_pose_ignore_keypoints.setMaximumHeight(120)
        self.list_pose_ignore_keypoints.setToolTip(
            "Select keypoints to ignore in pose export and orientation logic."
        )

        self.list_pose_direction_anterior = QListWidget()
        self.list_pose_direction_anterior.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_direction_anterior.setMinimumHeight(96)
        self.list_pose_direction_anterior.setMaximumHeight(120)
        self.list_pose_direction_anterior.setToolTip(
            "Select anterior keypoints from skeleton keypoint list."
        )

        self.list_pose_direction_posterior = QListWidget()
        self.list_pose_direction_posterior.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_direction_posterior.setMinimumHeight(96)
        self.list_pose_direction_posterior.setMaximumHeight(120)
        self.list_pose_direction_posterior.setToolTip(
            "Select posterior keypoints from skeleton keypoint list."
        )

        pose_kpt_groups_widget = QWidget()
        pose_kpt_groups_layout = QHBoxLayout(pose_kpt_groups_widget)
        pose_kpt_groups_layout.setContentsMargins(0, 0, 0, 0)
        pose_kpt_groups_layout.setSpacing(8)

        ignore_col = QVBoxLayout()
        ignore_label = QLabel("Ignore")
        ignore_label.setStyleSheet("font-weight: bold;")
        ignore_col.addWidget(ignore_label)
        ignore_col.addWidget(self.list_pose_ignore_keypoints)
        pose_kpt_groups_layout.addLayout(ignore_col, 1)

        anterior_col = QVBoxLayout()
        anterior_label = QLabel("Anterior")
        anterior_label.setStyleSheet("font-weight: bold;")
        anterior_col.addWidget(anterior_label)
        anterior_col.addWidget(self.list_pose_direction_anterior)
        pose_kpt_groups_layout.addLayout(anterior_col, 1)

        posterior_col = QVBoxLayout()
        posterior_label = QLabel("Posterior")
        posterior_label.setStyleSheet("font-weight: bold;")
        posterior_col.addWidget(posterior_label)
        posterior_col.addWidget(self.list_pose_direction_posterior)
        pose_kpt_groups_layout.addLayout(posterior_col, 1)

        fl_pose.addRow("Keypoint groups", pose_kpt_groups_widget)
        self.list_pose_ignore_keypoints.itemSelectionChanged.connect(
            lambda: self._main_window._on_pose_keypoint_group_changed("ignore")
        )
        self.list_pose_direction_anterior.itemSelectionChanged.connect(
            lambda: self._main_window._on_pose_keypoint_group_changed("anterior")
        )
        self.list_pose_direction_posterior.itemSelectionChanged.connect(
            lambda: self._main_window._on_pose_keypoint_group_changed("posterior")
        )

        h_sleap_env = QHBoxLayout()
        self.combo_pose_sleap_env = QComboBox()
        self.combo_pose_sleap_env.setToolTip(
            "Conda environment name must start with 'sleap'."
        )
        h_sleap_env.addWidget(self.combo_pose_sleap_env, 1)
        self.btn_refresh_pose_sleap_envs = QPushButton("Refresh")
        self.btn_refresh_pose_sleap_envs.setToolTip("Refresh SLEAP conda envs list")
        self.btn_refresh_pose_sleap_envs.clicked.connect(self._refresh_pose_sleap_envs)
        h_sleap_env.addWidget(self.btn_refresh_pose_sleap_envs)
        self.pose_sleap_env_row_widget = QWidget()
        self.pose_sleap_env_row_widget.setLayout(h_sleap_env)
        fl_pose.addRow("SLEAP env", self.pose_sleap_env_row_widget)

        pose_runtime_content_layout.addLayout(fl_pose)
        vl_pose.addWidget(self.pose_runtime_content)
        form.addWidget(self.g_pose_runtime)

        # Note: pose_sleap_envs and pose_runtime_flavor will be populated post-construction
        self._sync_headtail_model_remove_button()
        self._sync_pose_model_remove_button()

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Initially disable all controls (sync will show/hide based on state)
        self.g_identity.setVisible(False)
        self.g_identity.setEnabled(False)
        self.g_individual_pipeline_common.setVisible(False)
        self.g_individual_pipeline_common.setEnabled(False)
        self.g_headtail.setVisible(False)
        self.g_headtail.setEnabled(False)
        self.g_pose_runtime.setVisible(False)
        self.g_pose_runtime.setEnabled(False)
        # Note: _refresh_pose_direction_keypoint_lists and _sync_pose_backend_ui
        # are called post-construction from main_window

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    # ------------------------------------------------------------------
    # Handler methods (migrated from MainWindow)
    # ------------------------------------------------------------------

    def _on_individual_analysis_toggled(self, state):
        """Enable/disable individual analysis controls."""
        self._main_window._sync_individual_analysis_mode_ui()

    def _on_identity_method_changed(self, index):
        """Update identity configuration stack when method changes."""
        self._sync_identity_method_ui()

    def _on_identity_analysis_toggled(self, state):
        """Enable/disable identity-method controls inside individual analysis."""
        self._sync_identity_method_ui()
        self._main_window._sync_individual_analysis_mode_ui()

    def _on_pose_analysis_toggled(self, state):
        """Enable/disable pose-extraction controls inside individual analysis."""
        self._sync_pose_analysis_ui()
        self._main_window._sync_individual_analysis_mode_ui()

    def _on_headtail_analysis_toggled(self, state):
        """Enable/disable head-tail controls inside individual analysis."""
        self._sync_headtail_analysis_ui()
        self._main_window._sync_individual_analysis_mode_ui()

    @staticmethod
    def _build_inline_fields_row(fields):
        """Build a compact row containing multiple labeled controls."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for label_text, field_widget, stretch in fields:
            label = QLabel(label_text)
            label.setStyleSheet("color: #cccccc;")
            layout.addWidget(label)
            layout.addWidget(field_widget, stretch)
        return widget

    @staticmethod
    def _get_apriltag_families() -> list:
        """Return list of tag families supported by installed apriltag library."""
        _FALLBACK = [
            "tag36h11",
            "tag25h9",
            "tag16h5",
            "tagCircle21h7",
            "tagCircle49h12",
            "tagStandard41h12",
            "tagStandard52h13",
            "tagCustom48h12",
        ]
        try:
            import apriltag as _at  # type: ignore[import-untyped]
        except ImportError:
            return _FALLBACK
        try:
            _at.apriltag("__probe__")
        except Exception as exc:
            import re

            families = re.findall(r"^\s+(\S+)$", str(exc), re.MULTILINE)
            if families:
                return sorted(families)
        return _FALLBACK

    def _select_color_tag_model(self):
        """Browse for color tag YOLO model."""
        from hydra_suite.trackerkit.gui.main_window import (
            get_models_directory,
            resolve_model_path,
        )

        start_dir = get_models_directory()
        if self.line_color_tag_model.text():
            current_path = resolve_model_path(self.line_color_tag_model.text())
            if os.path.exists(current_path):
                start_dir = os.path.dirname(current_path)
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Color Tag YOLO Model", start_dir, "YOLO Models (*.pt *.onnx)"
        )
        if filepath:
            models_dir = get_models_directory()
            try:
                rel_path = os.path.relpath(filepath, models_dir)
                is_in_archive = not rel_path.startswith("..")
            except (ValueError, TypeError):
                is_in_archive = False
            if not is_in_archive:
                reply = QMessageBox.question(
                    self,
                    "Copy Model to Archive?",
                    f"The selected model is outside the local model archive.\n\n"
                    f"Would you like to copy it to the archive at:\n{models_dir}\n\n"
                    f"This makes presets portable across devices.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    filename = os.path.basename(filepath)
                    dest_path = os.path.join(models_dir, filename)
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(
                                models_dir, f"{base}_{counter}{ext}"
                            )
                            counter += 1
                    try:
                        shutil.copy2(filepath, dest_path)
                        filepath = dest_path
                        logger.info(f"Copied color tag model to archive: {dest_path}")
                        QMessageBox.information(
                            self,
                            "Model Copied",
                            f"Model copied to archive as:\n{os.path.basename(dest_path)}",
                        )
                    except Exception as e:
                        logger.error(f"Failed to copy model: {e}")
                        QMessageBox.warning(
                            self,
                            "Copy Failed",
                            f"Could not copy model to archive:\n{e}\n\nUsing original path.",
                        )
            self.line_color_tag_model.setText(filepath)

    class CNNClassifierRow(QWidget):
        """Self-contained widget for one CNN classifier configuration row."""

        remove_requested = Signal(object)

        def __init__(self, main_window, parent=None) -> None:
            super().__init__(parent)
            self._main_window = main_window
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(4, 4, 4, 4)
            outer.setSpacing(4)
            header_row = QHBoxLayout()
            self.combo_model = QComboBox()
            self.combo_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.combo_model.activated.connect(self._on_model_selected)
            header_row.addWidget(self.combo_model)
            self.btn_remove = QPushButton("\u2715")
            self.btn_remove.setMaximumWidth(28)
            self.btn_remove.setToolTip("Remove this CNN classifier")
            self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
            header_row.addWidget(self.btn_remove)
            self.btn_remove_model = QPushButton("-")
            self.btn_remove_model.setObjectName("SecondaryBtn")
            self.btn_remove_model.setFixedSize(28, 28)
            self.btn_remove_model.setToolTip(
                "Remove the selected classification model from the local repository."
            )
            self.btn_remove_model.clicked.connect(self._handle_remove_selected_model)
            header_row.addWidget(self.btn_remove_model)
            outer.addLayout(header_row)
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            self.lbl_arch = QLabel("\u2014")
            self.lbl_num_classes = QLabel("\u2014")
            self.lbl_class_names = QLabel("\u2014")
            self.lbl_input_size = QLabel("\u2014")
            self.lbl_label = QLabel("\u2014")
            self.lbl_recommended_confidence = QLabel("\u2014")
            form.addRow("Architecture", self.lbl_arch)
            form.addRow("Num classes", self.lbl_num_classes)
            form.addRow("Class names", self.lbl_class_names)
            form.addRow("Input size", self.lbl_input_size)
            form.addRow("Classification label", self.lbl_label)
            form.addRow("Recommended threshold", self.lbl_recommended_confidence)
            self.chk_unique_identifier = QCheckBox(
                "Treat this classifier as a unique identifier"
            )
            self.chk_unique_identifier.setChecked(False)
            self.chk_unique_identifier.setToolTip(
                "When ON, this classifier's outputs feed the identity pipeline\n"
                "(catalog, online decoder, assignment cost, posterior cache,\n"
                "fragment solver) and drive IdentityAssignedLabel.\n"
                "When OFF, the classifier still runs and its predictions are\n"
                "appended to the *_with_individual.csv as CNN_<label>_Class /\n"
                "CNN_<label>_Prob columns, but it has no effect on identity\n"
                "assignment or tracking.\n"
                "Enable this only for classifiers whose classes uniquely identify individuals."
            )
            form.addRow("Unique identifier", self.chk_unique_identifier)
            self.spin_confidence = QDoubleSpinBox()
            self.spin_confidence.setRange(0.0, 1.0)
            self.spin_confidence.setSingleStep(0.05)
            self.spin_confidence.setValue(0.5)
            form.addRow("Confidence threshold", self.spin_confidence)
            self.spin_window = QSpinBox()
            self.spin_window.setRange(1, 100)
            self.spin_window.setValue(10)
            form.addRow("History window", self.spin_window)
            self.spin_batch = QSpinBox()
            self.spin_batch.setRange(1, 256)
            self.spin_batch.setValue(64)
            self.spin_batch.setToolTip(
                "Classifier batch size. In realtime, this is capped to the current animal count."
            )
            form.addRow("Batch size", self.spin_batch)
            outer.addLayout(form)
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            outer.addWidget(line)
            self.combo_model.currentIndexChanged.connect(
                lambda _index: self._sync_model_ui()
            )
            self._populate_model_combo()

        def _populate_model_combo(self):
            """Populate combo from the v2 classifier registry and discovered artifacts."""
            from hydra_suite.training.model_publish import (
                enumerate_classifier_artifacts,
            )

            panel = self._main_window._identity_panel
            registry = panel._cnn_registry_by_path()
            discovered = list(enumerate_classifier_artifacts(roles=("cnn_identity",)))
            self.combo_model.blockSignals(True)
            current = self.combo_model.currentData()
            self.combo_model.clear()
            self.combo_model.addItem("\u2014 select model \u2014", "")
            for rel_path, meta in registry.items():
                self.combo_model.addItem(
                    panel._format_cnn_identity_display(meta), rel_path
                )
            seen = set(registry)
            for entry in discovered:
                if entry["path"] in seen:
                    continue
                self.combo_model.addItem(f"[classkit] {entry['path']}", entry["path"])
            self.combo_model.addItem("\uff0b Add New Model\u2026", "__add_new__")
            idx = self.combo_model.findData(current)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)
            self.combo_model.blockSignals(False)
            self._sync_model_ui()

        def _on_model_selected(self, index: int):
            rel_path = self.combo_model.itemData(index)
            if rel_path == "__add_new__":
                self._main_window._identity_panel._handle_add_new_cnn_identity_model()
                self._populate_model_combo()
                return
            if (
                rel_path
                and not self._main_window._identity_panel._registry_has_cnn_entry(
                    rel_path
                )
            ):
                self._main_window._identity_panel._annotate_discovered_cnn_model(
                    rel_path
                )
                self._populate_model_combo()
                return
            self._sync_model_ui()

        def _handle_remove_selected_model(self) -> None:
            """Remove the currently selected CNN identity model from the repository."""
            rel_path = self.combo_model.currentData()
            if not self._has_selected_model(rel_path):
                return
            if not self._main_window._confirm_and_remove_repository_model(
                rel_path,
                model_kind="classification model",
            ):
                return
            self._main_window._identity_panel._refresh_cnn_classifier_model_rows()

        def _update_verification_labels(self, rel_path: str):
            meta = self._main_window._identity_panel._cnn_registry_entry(rel_path)
            self.lbl_arch.setText(str(meta.get("arch", "\u2014")))
            self.lbl_num_classes.setText(
                str(self._main_window._identity_panel._classifier_num_classes(meta))
            )
            self.lbl_class_names.setText(
                self._main_window._identity_panel._classifier_class_preview(meta)
            )
            self.lbl_input_size.setText(str(meta.get("input_size", "\u2014")))
            self.lbl_label.setText(str(meta.get("classification_label", "\u2014")))
            recommended = self._main_window._identity_panel._classifier_recommended_confidence_threshold(
                meta
            )
            self.lbl_recommended_confidence.setText(
                f"{recommended:.0%}" if recommended is not None else "\u2014"
            )
            if recommended is not None:
                self.spin_confidence.setValue(recommended)

        def to_config(self):
            """Return config dict or None if no model selected."""
            from hydra_suite.trackerkit.gui.main_window import get_models_root_directory

            rel_path = self.combo_model.currentData()
            if not rel_path or rel_path == "__add_new__":
                return None
            meta = self._main_window._identity_panel._cnn_registry_entry(rel_path)
            models_root = get_models_root_directory()
            abs_path = os.path.join(models_root, rel_path)
            label = str(meta.get("classification_label", "") or "cnn_identity")
            cnpf = meta.get("class_names_per_factor") or []
            all_labels: list[str] = []
            for factor_labels in cnpf:
                for lbl in factor_labels:
                    if lbl and lbl not in all_labels:
                        all_labels.append(str(lbl))
            return {
                "model_path": abs_path,
                "label": label,
                "labels": all_labels,
                "class_names_per_factor": [[str(l) for l in fl] for fl in cnpf if fl],
                "confidence": self.spin_confidence.value(),
                "window": self.spin_window.value(),
                "batch_size": self.spin_batch.value(),
                "unique_identifier": self.chk_unique_identifier.isChecked(),
                "scoring_mode": str(meta.get("scoring_mode", "atomic")),
                "rel_path": rel_path,
            }

        def load_from_config(self, cfg: dict):
            """Populate from a config dict entry."""
            from hydra_suite.trackerkit.gui.main_window import get_models_root_directory

            rel_path = cfg.get("rel_path", "")
            if not rel_path:
                abs_path = cfg.get("model_path", "")
                models_root = get_models_root_directory()
                try:
                    for (
                        rp,
                        meta,
                    ) in (
                        self._main_window._identity_panel._cnn_registry_by_path().items()
                    ):
                        if os.path.normpath(
                            os.path.join(models_root, rp)
                        ) == os.path.normpath(abs_path):
                            rel_path = rp
                            break
                except Exception:
                    pass
            if rel_path:
                self._populate_model_combo()
                idx = self.combo_model.findData(rel_path)
                if idx >= 0:
                    self.combo_model.setCurrentIndex(idx)
                    self._update_verification_labels(rel_path)
            if "confidence" in cfg:
                self.spin_confidence.setValue(float(cfg["confidence"]))
            if "window" in cfg:
                self.spin_window.setValue(int(cfg["window"]))
            if "batch_size" in cfg:
                self.spin_batch.setValue(int(cfg["batch_size"]))
            self.chk_unique_identifier.setChecked(
                bool(cfg.get("unique_identifier", False))
            )

        def set_realtime_batch_cap(self, max_animals: int, realtime_enabled: bool):
            """Apply realtime batch caps for this classifier row."""
            self.spin_batch.setMaximum(max_animals if realtime_enabled else 256)
            clamped_batch = clamp_realtime_individual_batch_size(
                self.spin_batch.value(),
                max_animals=max_animals,
                realtime_enabled=realtime_enabled,
            )
            if clamped_batch != self.spin_batch.value():
                self.spin_batch.setValue(clamped_batch)

        @staticmethod
        def _has_selected_model(rel_path: object) -> bool:
            """Return True when the current row points to a removable model."""
            return bool(rel_path and rel_path not in ("__add_new__", "__none__"))

        def _sync_model_ui(self) -> None:
            """Keep verification labels and the remove button in sync with selection."""
            rel_path = self.combo_model.currentData()
            self.btn_remove_model.setEnabled(self._has_selected_model(rel_path))
            self._update_verification_labels(
                rel_path if self._has_selected_model(rel_path) else ""
            )

    def _add_cnn_classifier_row(self) -> "IdentityPanel.CNNClassifierRow":
        """Add a new CNN classifier row and return it."""
        row = self.CNNClassifierRow(self._main_window, self)
        row.remove_requested.connect(self._remove_cnn_classifier_row)
        self.cnn_rows_layout.addWidget(row)
        self.cnn_scroll_area.setVisible(True)
        self._sync_realtime_individual_batch_ui()
        self._main_window._sync_individual_analysis_mode_ui()
        return row

    def _remove_cnn_classifier_row(self, row: "IdentityPanel.CNNClassifierRow"):
        """Remove a CNN classifier row."""
        self.cnn_rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self.cnn_scroll_area.setVisible(bool(self._cnn_classifier_rows()))
        self._sync_realtime_individual_batch_ui()
        self._main_window._sync_individual_analysis_mode_ui()

    def _clear_cnn_classifier_rows(self) -> None:
        """Remove all configured CNN classifier rows before loading new config."""
        for row in list(self._cnn_classifier_rows()):
            self._remove_cnn_classifier_row(row)

    def _cnn_classifier_rows(self) -> list:
        """Return list of all CNNClassifierRow instances."""
        rows = []
        for i in range(self.cnn_rows_layout.count()):
            item = self.cnn_rows_layout.itemAt(i)
            if item and isinstance(item.widget(), self.CNNClassifierRow):
                rows.append(item.widget())
        return rows

    def _refresh_cnn_classifier_model_rows(self) -> None:
        """Refresh the shared CNN model store and all visible classifier rows."""
        self._refresh_cnn_identity_model_combo()
        for row in self._cnn_classifier_rows():
            row._populate_model_combo()
        self._sync_realtime_individual_batch_ui()
        self._main_window._sync_individual_analysis_mode_ui()

    def _sync_realtime_individual_batch_ui(self) -> None:
        """Reflect realtime per-animal batch policy in pose/head-tail/CNN controls."""
        max_animals = 1
        realtime_enabled = False
        if hasattr(self._main_window, "_setup_panel"):
            max_animals = max(
                1, self._main_window._setup_panel.spin_max_targets.value()
            )
            realtime_enabled = bool(
                self._main_window._setup_panel.chk_realtime_mode.isChecked()
            )

        self.spin_pose_batch.setMaximum(max_animals if realtime_enabled else 256)
        clamped_pose_batch = clamp_realtime_individual_batch_size(
            self.spin_pose_batch.value(),
            max_animals=max_animals,
            realtime_enabled=realtime_enabled,
        )
        if clamped_pose_batch != self.spin_pose_batch.value():
            self.spin_pose_batch.setValue(clamped_pose_batch)

        self.spin_headtail_batch.setMaximum(max_animals if realtime_enabled else 256)
        clamped_headtail_batch = clamp_realtime_individual_batch_size(
            self.spin_headtail_batch.value(),
            max_animals=max_animals,
            realtime_enabled=realtime_enabled,
        )
        if clamped_headtail_batch != self.spin_headtail_batch.value():
            self.spin_headtail_batch.setValue(clamped_headtail_batch)

        effective_batches = [
            self.spin_pose_batch.value(),
            self.spin_headtail_batch.value(),
        ]
        for row in self._cnn_classifier_rows():
            row.set_realtime_batch_cap(max_animals, realtime_enabled)
            effective_batches.append(row.spin_batch.value())

        messages: list[str] = []
        if realtime_enabled:
            warn_waste = any(
                should_warn_for_padding_waste(batch_size, upper_bound=max_animals)
                for batch_size in effective_batches
            )
            message = (
                f"Realtime individual inference is capped to {max_animals} animal(s) per frame. "
                "Pose, head-tail, and CNN batch controls are clamped to avoid oversized per-frame batches."
            )
            if warn_waste:
                message += " Reduce larger saved values when switching back to non-realtime if you want to avoid partially empty exported batches."
            messages.append(message)

        headtail_recommendation = (
            self._main_window._current_headtail_benchmark_recommendation()
        )
        if headtail_recommendation is not None:
            messages.append(
                "Benchmark recommendation: "
                f"head-tail batch {headtail_recommendation.batch_size} on "
                f"{headtail_recommendation.runtime_label}."
            )

        pose_recommendation = self._main_window._current_pose_benchmark_recommendation()
        if pose_recommendation is not None:
            messages.append(
                f"Benchmark recommendation: pose batch {pose_recommendation.batch_size} on {pose_recommendation.runtime_label}."
            )

        cnn_recommendations = self._main_window._current_cnn_benchmark_recommendations()
        if cnn_recommendations:
            runtime_recommendation = (
                self._main_window._current_cnn_runtime_recommendation()
            )
            summaries = []
            for key in sorted(cnn_recommendations):
                recommendation = cnn_recommendations[key]
                label = Path(recommendation.model_path).stem
                summaries.append(f"{label}: {recommendation.batch_size}")
            prefix = "Benchmark recommendations"
            if runtime_recommendation is not None:
                prefix = (
                    f"Benchmark recommendation: CNN runtime {runtime_recommendation.runtime_label}; "
                    "batch sizes"
                )
            messages.append(prefix + " " + ", ".join(summaries) + ".")

        if not messages:
            self.lbl_individual_batch_notice.clear()
            self.lbl_individual_batch_notice.setVisible(False)
            return

        self.lbl_individual_batch_notice.setText("\n".join(messages))
        self.lbl_individual_batch_notice.setVisible(True)

    def _refresh_cnn_identity_model_combo(self) -> None:
        """Populate the CNN identity combo from the registry and ClassKit publish roots."""
        from hydra_suite.training.model_publish import enumerate_classifier_artifacts

        registry_by_path = self._cnn_registry_by_path()
        discovered = list(enumerate_classifier_artifacts(roles=("cnn_identity",)))

        self.combo_cnn_identity_model.blockSignals(True)
        current_path = self.combo_cnn_identity_model.currentData()
        self.combo_cnn_identity_model.clear()
        self.combo_cnn_identity_model.addItem("— select model —", "")

        # First the registered (annotated) entries
        for rel_path, meta in registry_by_path.items():
            self.combo_cnn_identity_model.addItem(
                self._format_cnn_identity_display(meta), rel_path
            )

        # Then discovered-but-not-yet-annotated artifacts
        seen = set(registry_by_path)
        for entry in discovered:
            if entry["path"] in seen:
                continue
            display = f"[classkit] {entry['path']}"
            self.combo_cnn_identity_model.addItem(display, entry["path"])

        self.combo_cnn_identity_model.addItem(
            "\uff0b Add New Model\u2026", "__add_new__"
        )

        idx = self.combo_cnn_identity_model.findData(current_path)
        if idx >= 0:
            self.combo_cnn_identity_model.setCurrentIndex(idx)
        self.combo_cnn_identity_model.blockSignals(False)
        self._update_legacy_registry_banner()

    def _update_legacy_registry_banner(self) -> None:
        from hydra_suite.training.model_publish import count_legacy_registry_entries

        n = count_legacy_registry_entries()
        if n == 0:
            self.lbl_legacy_banner.setVisible(False)
            return
        self.lbl_legacy_banner.setText(
            f"\u26a0 {n} model registry entry/entries use an unsupported legacy "
            f"format. Re-import them from the Models folder to use them."
        )
        self.lbl_legacy_banner.setVisible(True)

    def _on_cnn_identity_model_selected(self, index: int) -> None:
        """Handle combo activation — sentinel triggers import dialog."""
        rel_path = self.combo_cnn_identity_model.itemData(index)
        if rel_path == "__add_new__":
            self._handle_add_new_cnn_identity_model()
            return
        if rel_path and not self._registry_has_cnn_entry(rel_path):
            self._annotate_discovered_cnn_model(rel_path)
            return
        self._update_cnn_identity_verification_panel(rel_path)

    def _registry_has_cnn_entry(self, rel_path: str) -> bool:
        return rel_path in self._cnn_registry_by_path()

    def _annotate_discovered_cnn_model(self, rel_path: str) -> None:
        """Open the import dialog for a discovered classkit-published model."""
        from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
            CNNIdentityImportDialog,
            describe_cnn_identity_candidate,
            import_cnn_identity_candidate,
        )
        from hydra_suite.trackerkit.gui.main_window import get_models_root_directory

        abs_path = os.path.join(get_models_root_directory(), rel_path)
        try:
            summary = describe_cnn_identity_candidate(abs_path)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", f"Cannot read model: {exc}")
            return

        dlg = CNNIdentityImportDialog(summary, parent=self)
        if dlg.exec() != QDialog.Accepted:
            self._refresh_cnn_identity_model_combo()
            return
        rel_path = import_cnn_identity_candidate(
            model_path=abs_path,
            species=dlg.species(),
            classification_label=dlg.classification_label(),
            scoring_mode=dlg.scoring_mode(),
        )
        self._refresh_cnn_identity_model_combo()
        idx = self.combo_cnn_identity_model.findData(rel_path)
        if idx >= 0:
            self.combo_cnn_identity_model.setCurrentIndex(idx)
            self._update_cnn_identity_verification_panel(rel_path)

    def _update_cnn_identity_verification_panel(self, rel_path: str) -> None:
        """Populate the read-only verification labels from the registry entry."""
        meta = self._cnn_registry_entry(rel_path)
        self.lbl_cnn_arch.setText(str(meta.get("arch", "\u2014")))
        self.lbl_cnn_num_classes.setText(str(self._classifier_num_classes(meta)))
        self.lbl_cnn_class_names.setText(self._classifier_class_preview(meta))
        raw_size = meta.get("input_size", "\u2014")
        self.lbl_cnn_input_size.setText(str(raw_size))
        self.lbl_cnn_label.setText(str(meta.get("classification_label", "\u2014")))
        recommended = self._classifier_recommended_confidence_threshold(meta)
        self.lbl_cnn_recommended_confidence.setText(
            f"{recommended:.0%}" if recommended is not None else "\u2014"
        )
        if recommended is not None:
            self.spin_cnn_confidence.setValue(recommended)

    def _handle_add_new_cnn_identity_model(self) -> None:
        """Import a ClassKit-trained classifier for CNN identity."""
        from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
            CNNIdentityImportDialog,
            describe_cnn_identity_candidate,
            import_cnn_identity_candidate,
        )
        from hydra_suite.trackerkit.gui.main_window import get_models_root_directory

        prev_data = self.combo_cnn_identity_model.currentData()
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import ClassKit Model for CNN Identity",
            os.path.join(get_models_root_directory(), "classification", "identity"),
            "ClassKit Model Files (*.pth *.pt *.multihead.json);;All Files (*)",
        )
        if not src_path:
            idx = self.combo_cnn_identity_model.findData(prev_data)
            self.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return
        try:
            summary = describe_cnn_identity_candidate(src_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not read checkpoint metadata:\n{exc}",
            )
            idx = self.combo_cnn_identity_model.findData(prev_data)
            self.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return
        dlg = CNNIdentityImportDialog(summary, parent=self)
        if dlg.exec() != QDialog.Accepted:
            idx = self.combo_cnn_identity_model.findData(prev_data)
            self.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return
        try:
            rel_path = import_cnn_identity_candidate(
                model_path=src_path,
                species=dlg.species(),
                classification_label=dlg.classification_label(),
                scoring_mode=dlg.scoring_mode(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Could not import model:\n{exc}"
            )
            idx = self.combo_cnn_identity_model.findData(prev_data)
            self.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return
        self._refresh_cnn_identity_model_combo()
        idx = self.combo_cnn_identity_model.findData(rel_path)
        if idx >= 0:
            self.combo_cnn_identity_model.setCurrentIndex(idx)
            self._update_cnn_identity_verification_panel(rel_path)

    def _cnn_registry_by_path(self) -> dict[str, dict]:
        from hydra_suite.training.model_publish import iter_registry_entries

        return {
            key: meta
            for key, meta in iter_registry_entries()
            if meta.get("usage_role") == "cnn_identity"
        }

    def _cnn_registry_entry(self, rel_path: str) -> dict:
        if not rel_path:
            return {}
        return dict(self._cnn_registry_by_path().get(rel_path, {}))

    @staticmethod
    def _classifier_recommended_confidence_threshold(meta: dict) -> float | None:
        if not meta:
            return None
        raw_value = meta.get("recommended_confidence_threshold")
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, numeric))

    @staticmethod
    def _classifier_num_classes(meta: dict) -> object:
        if not meta:
            return "—"
        if meta.get("num_classes") is not None:
            return meta.get("num_classes")
        class_names = meta.get("class_names") or []
        if class_names:
            return len(class_names)
        return (
            sum(len(names) for names in (meta.get("class_names_per_factor") or []))
            or "—"
        )

    @staticmethod
    def _classifier_class_preview(meta: dict) -> str:
        if not meta:
            return "—"
        class_names = meta.get("class_names") or []
        if class_names:
            preview = ", ".join(class_names[:12])
            if len(class_names) > 12:
                preview += f", … ({len(class_names)} total)"
            return preview or "—"
        factors = meta.get("factor_names") or []
        per_factor = meta.get("class_names_per_factor") or []
        if len(factors) != len(per_factor) or not factors:
            return "—"
        parts = []
        for factor, names in zip(factors, per_factor):
            sample = ", ".join(list(names)[:4])
            if len(names) > 4:
                sample += f", … ({len(names)} total)"
            parts.append(f"{factor}: {sample}")
        return " | ".join(parts) or "—"

    def _format_cnn_identity_display(self, meta: dict) -> str:
        arch = str(meta.get("arch", "?"))
        factors = meta.get("factor_names") or ["flat"]
        species = str(meta.get("species", ""))
        label = str(meta.get("classification_label", ""))
        num_classes = self._classifier_num_classes(meta)
        tag = " [multi-head]" if len(factors) > 1 else ""
        display = f"{arch}{tag} | {num_classes} cls"
        if species:
            display += f" | {species}"
        if label:
            display += f" | {label}"
        return display

    def _sync_identity_method_ui(self):
        """Show the active identity configuration only when enabled."""
        identity_enabled = self._main_window._is_identity_analysis_enabled()
        self.identity_content.setVisible(identity_enabled)
        self.identity_content.setEnabled(identity_enabled)

    def _sync_pose_analysis_ui(self):
        """Show pose controls only when pose extraction is enabled."""
        pose_enabled = bool(
            self._main_window._is_individual_pipeline_enabled()
            and self.chk_enable_pose_extractor.isChecked()
        )
        self.pose_runtime_content.setVisible(pose_enabled)
        self.pose_runtime_content.setEnabled(pose_enabled)

    def _sync_headtail_analysis_ui(self):
        """Show head-tail controls only when the section is enabled."""
        headtail_enabled = bool(self.g_headtail.isChecked())
        configured_model = bool(self._get_configured_yolo_headtail_model_path().strip())
        self.headtail_content.setVisible(headtail_enabled)
        self.headtail_content.setEnabled(headtail_enabled)
        self.spin_yolo_headtail_conf.setEnabled(headtail_enabled and configured_model)
        self.spin_yolo_headtail_detect_conf.setEnabled(
            headtail_enabled and configured_model
        )
        self.chk_pose_overrides_headtail.setEnabled(headtail_enabled)

    def _refresh_pose_direction_keypoint_lists(self):
        """Populate ignore/anterior/posterior keypoint pickers from skeleton file."""
        prev_ignore = self._main_window._selected_pose_group_keypoints(
            self.list_pose_ignore_keypoints
        )
        prev_anterior = self._main_window._selected_pose_group_keypoints(
            self.list_pose_direction_anterior
        )
        prev_posterior = self._main_window._selected_pose_group_keypoints(
            self.list_pose_direction_posterior
        )
        names = self._main_window._load_pose_skeleton_keypoint_names()
        self.list_pose_ignore_keypoints.blockSignals(True)
        self.list_pose_direction_anterior.blockSignals(True)
        self.list_pose_direction_posterior.blockSignals(True)
        self.list_pose_ignore_keypoints.clear()
        self.list_pose_direction_anterior.clear()
        self.list_pose_direction_posterior.clear()
        self.list_pose_ignore_keypoints.addItems(names)
        self.list_pose_direction_anterior.addItems(names)
        self.list_pose_direction_posterior.addItems(names)
        self._main_window._set_pose_group_selection(
            self.list_pose_ignore_keypoints, prev_ignore
        )
        self._main_window._set_pose_group_selection(
            self.list_pose_direction_anterior, prev_anterior
        )
        self._main_window._set_pose_group_selection(
            self.list_pose_direction_posterior, prev_posterior
        )
        enabled = len(names) > 0
        self.list_pose_ignore_keypoints.setEnabled(enabled)
        self.list_pose_direction_anterior.setEnabled(enabled)
        self.list_pose_direction_posterior.setEnabled(enabled)
        self.list_pose_ignore_keypoints.blockSignals(False)
        self.list_pose_direction_anterior.blockSignals(False)
        self.list_pose_direction_posterior.blockSignals(False)
        self._main_window._apply_pose_keypoint_selection_constraints("ignore")

    def _refresh_pose_sleap_envs(self):
        """Refresh conda environments starting with 'sleap'."""
        self.combo_pose_sleap_env.clear()
        self.combo_pose_sleap_env.setEnabled(True)
        envs = []
        preferred = str(
            self._main_window.advanced_config.get("pose_sleap_env", "sleap")
        ).strip()
        try:
            import subprocess

            res = subprocess.run(
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
                    name = parts[0].strip()
                    if name.lower().startswith("sleap"):
                        envs.append(name)
        except Exception:
            envs = []
        if not envs:
            self.combo_pose_sleap_env.addItem("No sleap envs found")
            self.combo_pose_sleap_env.setEnabled(False)
            return
        self.combo_pose_sleap_env.addItems(envs)
        if preferred and preferred in envs:
            self.combo_pose_sleap_env.setCurrentText(preferred)

    def _refresh_yolo_headtail_model_combo(self, preferred_model_path: object = None):
        """Refresh the head-tail model combo across all classifier families."""
        from hydra_suite.trackerkit.gui.main_window import (
            get_yolo_model_repository_directory,
        )

        requested_path = (
            preferred_model_path or self.combo_yolo_headtail_model.currentData()
        )
        repo_dir = get_yolo_model_repository_directory(
            task_family="classify", usage_role="headtail"
        )
        os.makedirs(repo_dir, exist_ok=True)
        self._main_window._populate_yolo_model_combo(
            self.combo_yolo_headtail_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=True,
            task_family="classify",
            usage_role="headtail",
            repository_dir=repo_dir,
            recursive=True,
        )

        # Append [classkit] discovered head-tail artifacts not already in the combo.
        try:
            from hydra_suite.trackerkit.gui.dialogs.headtail_import_dialog import (
                describe_headtail_candidate,
            )
            from hydra_suite.training.model_publish import (
                enumerate_classifier_artifacts,
                iter_registry_entries,
            )
        except ImportError:
            self._sync_headtail_model_remove_button()
            self._update_legacy_registry_banner()
            return

        registry_paths = {
            key
            for key, meta in iter_registry_entries()
            if str(meta.get("usage_role") or "").strip().lower()
            in {"headtail", "head_tail"}
        }
        existing_data = set()
        for i in range(self.combo_yolo_headtail_model.count()):
            existing_data.add(self.combo_yolo_headtail_model.itemData(i))

        for rel_path in sorted(registry_paths):
            if rel_path in existing_data:
                continue
            self.combo_yolo_headtail_model.addItem(
                self._main_window._format_yolo_model_label(rel_path),
                rel_path,
            )
            existing_data.add(rel_path)

        for entry in enumerate_classifier_artifacts(roles=("head_tail",)):
            if entry["path"] in existing_data:
                continue
            try:
                summary = describe_headtail_candidate(entry["abs_path"])
            except Exception:
                continue
            if not summary.get("valid"):
                continue
            display = f"[classkit] {entry['path']}"
            self.combo_yolo_headtail_model.addItem(display, entry["path"])
            existing_data.add(entry["path"])

        if requested_path:
            selected_index = self.combo_yolo_headtail_model.findData(requested_path)
            if selected_index >= 0:
                self.combo_yolo_headtail_model.setCurrentIndex(selected_index)

        self._sync_headtail_model_remove_button()
        self._update_legacy_registry_banner()

    def _get_selected_yolo_headtail_model_path(self) -> object:
        """Return the effective head-tail model path for runtime use."""
        if not getattr(self, "g_headtail", None) or not self.g_headtail.isChecked():
            return ""
        return self._get_configured_yolo_headtail_model_path()

    def _get_configured_yolo_headtail_model_path(self) -> object:
        """Return the configured head-tail model path regardless of enable state."""
        return self._main_window._get_selected_model_path_from_selector(
            self.combo_yolo_headtail_model,
            default_path="",
        )

    @staticmethod
    def _create_model_remove_button(tooltip: str) -> QPushButton:
        """Create a compact remove button for model-selector rows."""
        button = QPushButton("-")
        button.setObjectName("SecondaryBtn")
        button.setFixedSize(28, 30)
        button.setToolTip(tooltip)
        return button

    @staticmethod
    def _combo_has_selected_model(combo: QComboBox) -> bool:
        """Return True when the combo currently points to a removable model."""
        selected_data = combo.currentData()
        return bool(selected_data and selected_data not in ("__add_new__", "__none__"))

    def _sync_headtail_model_remove_button(self) -> None:
        """Enable head-tail model removal only for a real current selection."""
        self.btn_remove_yolo_headtail_model.setEnabled(
            self._combo_has_selected_model(self.combo_yolo_headtail_model)
        )

    def _sync_pose_model_remove_button(self) -> None:
        """Enable pose-model removal only for a real current selection."""
        self.btn_remove_pose_model.setEnabled(
            self._combo_has_selected_model(self.combo_pose_model)
        )
