"""ConfigOrchestrator — config load/save, presets, ROI, video setup."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from hydra_suite.runtime.compute_runtime import (
    derive_detection_runtime_settings,
    derive_pose_runtime_settings,
    infer_compute_runtime_from_legacy,
)
from hydra_suite.trackerkit.benchmarking import (
    collect_active_targets,
    derive_benchmark_geometry_from_video,
    lookup_cached_recommendation,
)
from hydra_suite.trackerkit.gui.model_utils import (
    _normalize_usage_role,
    _sanitize_model_token,
    get_pose_models_directory,
    get_yolo_model_metadata,
    get_yolo_model_repository_directory,
    make_model_path_relative,
    make_pose_model_path_relative,
    register_yolo_model,
    remove_model_from_repository,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "tracking_config.json"


def _get_video_config_path(video_path: str):
    """Get the config file path for a given video file."""
    if not video_path:
        return None
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_config.json")


class ConfigOrchestrator:
    """Manages configuration load/save, presets, ROI, and video file setup."""

    def __init__(self, main_window, config, panels) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels

    # =========================================================================
    # CONFIG LOAD / SAVE
    # =========================================================================

    def load_config(self: object) -> object:
        """Manually load config from file dialog."""
        config_path, _ = QFileDialog.getOpenFileName(
            self._mw, "Load Configuration", "", "JSON Files (*.json)"
        )
        if config_path:
            self._load_config_from_file(config_path)
            self._mw._show_workspace()
            self._panels.setup.config_status_label.setText(
                f"✓ Loaded: {os.path.basename(config_path)}"
            )
            self._panels.setup.config_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Configuration loaded from {config_path}")

    @staticmethod
    def _cfg_get(cfg, new_key, *legacy_keys, default=None):
        """Get config value with fallback to legacy keys."""
        if new_key in cfg:
            return cfg[new_key]
        for key in legacy_keys:
            if key in cfg:
                return cfg[key]
        return default

    @staticmethod
    def _cfg_get_time(cfg, seconds_key, *frame_keys, default_seconds):
        """Load a time parameter, converting legacy frame-based values to seconds."""
        val = ConfigOrchestrator._cfg_get(cfg, seconds_key, default=None)
        if val is not None:
            return float(val)
        config_fps = float(ConfigOrchestrator._cfg_get(cfg, "fps", default=30.0))
        for fk in frame_keys:
            if fk in cfg:
                return float(cfg[fk]) / config_fps
        return default_seconds

    def _load_config_from_file(self, config_path, preset_mode=False):
        """Internal method to load config from a specific file path.

        This method supports both new standardized key names and legacy key names
        for backward compatibility with older config files.

        Args:
            config_path: Path to the config/preset file
            preset_mode: If True, skip loading video paths and ROI data (for organism presets)
        """
        if not os.path.isfile(config_path):
            return
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)

            def get_cfg(*a, **kw):
                """Look up a config key in the current file's dict, with fallback default."""
                return self._cfg_get(cfg, *a, **kw)

            def get_cfg_time(*a, **kw):
                """Look up a time-valued config key, converting legacy frame-based values to seconds."""
                return self._cfg_get_time(cfg, *a, **kw)

            self._load_config_file_paths(cfg, get_cfg, preset_mode)
            self._load_config_reference_params(cfg, get_cfg, preset_mode)
            self._load_config_system_performance(get_cfg)
            self._load_config_detection(get_cfg, get_cfg_time)
            self._load_config_yolo(cfg, get_cfg, preset_mode)
            self._load_config_core_tracking(get_cfg, get_cfg_time)
            self._load_config_orientation_and_lifecycle(get_cfg, get_cfg_time)
            self._load_config_postprocessing(get_cfg, get_cfg_time)
            self._load_config_visualization(get_cfg)
            self._load_config_dataset(get_cfg)
            self._load_config_individual_analysis(cfg, get_cfg)
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")

    def _load_config_file_paths(self, cfg, get_cfg, preset_mode):
        if preset_mode:
            return
        if not self._panels.setup.file_line.text().strip():
            video_path = get_cfg("file_path", default="")
            if video_path:
                self._setup_video_file(video_path, skip_config_load=True)
        if not self._panels.setup.csv_line.text().strip():
            self._panels.setup.csv_line.setText(get_cfg("csv_path", default=""))
        self._panels.postprocess.check_video_output.setChecked(
            get_cfg("video_output_enabled", default=False)
        )
        saved_video_path = get_cfg("video_output_path", default="")
        if (
            saved_video_path
            and not self._panels.postprocess.video_out_line.text().strip()
        ):
            self._panels.postprocess.video_out_line.setText(saved_video_path)

    def _load_config_reference_params(self, cfg, get_cfg, preset_mode):
        if preset_mode:
            return
        saved_fps = get_cfg("fps", default=None)
        if saved_fps is not None:
            self._panels.setup.spin_fps.setValue(saved_fps)
        saved_body_size = get_cfg("reference_body_size", default=None)
        if saved_body_size is not None:
            self._panels.detection.spin_reference_body_size.setValue(saved_body_size)
        saved_start_frame = get_cfg("start_frame", default=None)
        if (
            saved_start_frame is not None
            and self._panels.setup.spin_start_frame.isEnabled()
        ):
            self._panels.setup.spin_start_frame.setValue(saved_start_frame)
        saved_end_frame = get_cfg("end_frame", default=None)
        if (
            saved_end_frame is not None
            and self._panels.setup.spin_end_frame.isEnabled()
        ):
            self._panels.setup.spin_end_frame.setValue(saved_end_frame)

    def _load_config_system_performance(self, get_cfg):
        self._panels.setup.spin_resize.setValue(get_cfg("resize_factor", default=1.0))
        self._panels.setup.check_save_confidence.setChecked(
            get_cfg("save_confidence_metrics", default=True)
        )
        self._panels.setup.chk_use_cached_detections.setChecked(
            get_cfg("use_cached_detections", default=True)
        )
        self._panels.setup.chk_visualization_free.setChecked(
            get_cfg("visualization_free_mode", default=False)
        )

    def _load_config_detection(self, get_cfg, get_cfg_time):
        det_method = get_cfg("detection_method", default="background_subtraction")
        self._panels.detection.combo_detection_method.setCurrentIndex(
            0 if det_method == "background_subtraction" else 1
        )
        self._panels.detection.chk_size_filtering.setChecked(
            get_cfg("enable_size_filtering", default=False)
        )
        self._panels.detection.spin_min_object_size.setValue(
            get_cfg("min_object_size_multiplier", default=0.3)
        )
        self._panels.detection.spin_max_object_size.setValue(
            get_cfg("max_object_size_multiplier", default=3.0)
        )
        self._panels.detection.slider_brightness.setValue(
            int(get_cfg("brightness", default=0.0))
        )
        self._panels.detection.slider_contrast.setValue(
            int(get_cfg("contrast", default=1.0) * 100)
        )
        self._panels.detection.slider_gamma.setValue(
            int(get_cfg("gamma", default=1.0) * 100)
        )
        self._panels.detection.chk_dark_on_light.setChecked(
            get_cfg("dark_on_light_background", default=True)
        )
        self._panels.detection.spin_bg_prime.setValue(
            get_cfg_time(
                "background_prime_seconds",
                "background_prime_frames",
                "bg_prime_frames",
                default_seconds=0.33,
            )
        )
        self._panels.detection.chk_adaptive_bg.setChecked(
            get_cfg("enable_adaptive_background", "adaptive_background", default=True)
        )
        self._panels.detection.spin_bg_learning.setValue(
            get_cfg("background_learning_rate", default=0.001)
        )
        self._panels.detection.spin_threshold.setValue(
            get_cfg("subtraction_threshold", "threshold_value", default=50)
        )
        self._panels.detection.chk_lighting_stab.setChecked(
            get_cfg(
                "enable_lighting_stabilization",
                "lighting_stabilization",
                default=True,
            )
        )
        self._panels.detection.spin_lighting_smooth.setValue(
            get_cfg("lighting_smooth_factor", default=0.95)
        )
        self._panels.detection.spin_lighting_median.setValue(
            get_cfg("lighting_median_window", default=5)
        )
        self._panels.detection.spin_morph_size.setValue(
            get_cfg("morph_kernel_size", default=5)
        )
        self._panels.detection.spin_min_contour.setValue(
            get_cfg("min_contour_area", default=50)
        )
        self._panels.detection.spin_max_contour_multiplier.setValue(
            get_cfg("max_contour_multiplier", default=20)
        )
        self._panels.detection.chk_conservative_split.setChecked(
            get_cfg("enable_conservative_split", default=True)
        )
        self._panels.detection.spin_conservative_kernel.setValue(
            get_cfg("conservative_kernel_size", default=3)
        )
        self._panels.detection.spin_conservative_erode.setValue(
            get_cfg(
                "conservative_erode_iterations",
                "conservative_erode_iter",
                default=1,
            )
        )
        self._panels.detection.chk_additional_dilation.setChecked(
            get_cfg("enable_additional_dilation", default=False)
        )
        self._panels.detection.spin_dilation_kernel_size.setValue(
            get_cfg("dilation_kernel_size", default=3)
        )
        self._panels.detection.spin_dilation_iterations.setValue(
            get_cfg("dilation_iterations", default=2)
        )

    def _load_config_yolo(self, cfg, get_cfg, preset_mode):
        yolo_mode = str(get_cfg("yolo_obb_mode", default="direct")).strip().lower()
        if yolo_mode not in {"direct", "sequential"}:
            yolo_mode = "direct"
        self._panels.detection.combo_yolo_obb_mode.setCurrentIndex(
            1 if yolo_mode == "sequential" else 0
        )

        yolo_direct_model = get_cfg(
            "yolo_obb_direct_model_path",
            "yolo_model_path",
            default="",
        )
        yolo_detect_model = get_cfg("yolo_detect_model_path", default="")
        yolo_crop_obb_model = get_cfg(
            "yolo_crop_obb_model_path",
            default=yolo_direct_model,
        )
        yolo_headtail_model = get_cfg("yolo_headtail_model_path", default="")

        from hydra_suite.trackerkit.gui.main_window import resolve_model_path

        resolved_yolo_direct = resolve_model_path(yolo_direct_model)
        resolved_yolo_detect = resolve_model_path(yolo_detect_model)
        resolved_yolo_crop_obb = resolve_model_path(yolo_crop_obb_model)
        resolved_yolo_headtail = resolve_model_path(yolo_headtail_model)

        if preset_mode:
            for model_key, model_cfg, model_resolved in (
                ("Direct OBB model", yolo_direct_model, resolved_yolo_direct),
                (
                    "Sequential detect model",
                    yolo_detect_model,
                    resolved_yolo_detect,
                ),
                (
                    "Sequential crop OBB model",
                    yolo_crop_obb_model,
                    resolved_yolo_crop_obb,
                ),
                ("Head-tail model", yolo_headtail_model, resolved_yolo_headtail),
            ):
                if not model_cfg:
                    continue
                if os.path.isabs(str(model_cfg)):
                    continue
                if os.path.exists(model_resolved):
                    continue
                logger.warning(
                    "Preset references non-existent %s: %s (expected: %s)",
                    model_key,
                    model_cfg,
                    model_resolved,
                )

        self._panels.detection._refresh_yolo_model_combo(
            preferred_model_path=yolo_direct_model
        )
        self._mw._set_yolo_model_selection(resolved_yolo_direct)
        self._panels.detection._refresh_yolo_detect_model_combo(
            preferred_model_path=yolo_detect_model
        )
        self._mw._set_yolo_detect_model_selection(resolved_yolo_detect)
        self._panels.detection._refresh_yolo_crop_obb_model_combo(
            preferred_model_path=yolo_crop_obb_model
        )
        self._mw._set_yolo_crop_obb_model_selection(resolved_yolo_crop_obb)
        self._panels.identity._refresh_yolo_headtail_model_combo(
            preferred_model_path=yolo_headtail_model
        )
        self._mw._set_yolo_headtail_model_selection(resolved_yolo_headtail)
        self._panels.identity.g_headtail.setChecked(
            bool(
                get_cfg(
                    "enable_headtail_orientation",
                    default=bool(str(yolo_headtail_model).strip()),
                )
            )
        )
        self._panels.identity.chk_pose_overrides_headtail.setChecked(
            bool(get_cfg("pose_overrides_headtail", default=True))
        )
        self._panels.detection.spin_yolo_seq_crop_pad.setValue(
            float(get_cfg("yolo_seq_crop_pad_ratio", default=0.15))
        )
        self._panels.detection.spin_yolo_seq_min_crop_px.setValue(
            int(get_cfg("yolo_seq_min_crop_size_px", default=64))
        )
        self._panels.detection.chk_yolo_seq_square_crop.setChecked(
            bool(get_cfg("yolo_seq_enforce_square_crop", default=True))
        )
        self._panels.detection.spin_yolo_seq_stage2_imgsz.setValue(
            int(get_cfg("yolo_seq_stage2_imgsz", default=160))
        )
        self._panels.detection.spin_yolo_seq_individual_batch_size.setValue(
            int(get_cfg("yolo_seq_individual_batch_size", default=16))
        )
        self._panels.detection.chk_yolo_seq_stage2_pow2_pad.setChecked(
            bool(get_cfg("yolo_seq_stage2_pow2_pad", default=False))
        )
        self._panels.detection.spin_yolo_seq_detect_conf.setValue(
            float(get_cfg("yolo_seq_detect_conf_threshold", default=0.25))
        )
        self._panels.identity.spin_yolo_headtail_conf.setValue(
            float(get_cfg("yolo_headtail_conf_threshold", default=0.50))
        )
        self._panels.identity.spin_yolo_headtail_detect_conf.setValue(
            float(get_cfg("yolo_headtail_detect_conf_threshold", default=0.25))
        )
        self._panels.identity.spin_headtail_batch.setValue(
            int(
                get_cfg(
                    "headtail_batch_size",
                    default=int(
                        self._mw.advanced_config.get("headtail_batch_size", 64)
                    ),
                )
            )
        )
        self._panels.detection.spin_reference_aspect_ratio.setValue(
            float(get_cfg("reference_aspect_ratio", default=2.0))
        )
        self._panels.detection.chk_enable_aspect_ratio_filtering.setChecked(
            bool(get_cfg("enable_aspect_ratio_filtering", default=False))
        )
        self._panels.detection.spin_min_ar_multiplier.setValue(
            float(get_cfg("min_aspect_ratio_multiplier", default=0.5))
        )
        self._panels.detection.spin_max_ar_multiplier.setValue(
            float(get_cfg("max_aspect_ratio_multiplier", default=2.0))
        )
        self._panels.detection._on_yolo_mode_changed(
            self._panels.detection.combo_yolo_obb_mode.currentIndex()
        )

        self._panels.detection.spin_yolo_confidence.setValue(
            get_cfg("yolo_confidence_threshold", default=0.25)
        )
        self._panels.detection.spin_yolo_iou.setValue(
            get_cfg("yolo_iou_threshold", default=0.7)
        )
        self._panels.detection.chk_use_custom_obb_iou.setChecked(True)
        yolo_cls = get_cfg("yolo_target_classes", default=None)
        if yolo_cls:
            self._panels.detection.line_yolo_classes.setText(
                ",".join(map(str, yolo_cls))
            )
        else:
            self._panels.detection.line_yolo_classes.clear()

        compute_runtime_cfg = (
            str(
                get_cfg(
                    "compute_runtime",
                    default=infer_compute_runtime_from_legacy(
                        yolo_device=str(get_cfg("yolo_device", default="auto")),
                        enable_tensorrt=bool(get_cfg("enable_tensorrt", default=False)),
                        pose_runtime_flavor=str(
                            get_cfg("pose_runtime_flavor", default="auto")
                        ),
                    ),
                )
            )
            .strip()
            .lower()
        )
        self._mw._populate_compute_runtime_options(preferred=compute_runtime_cfg)
        self._mw._on_runtime_context_changed()
        self._mw._populate_headtail_runtime_options(
            preferred=str(get_cfg("headtail_runtime", default=compute_runtime_cfg))
            .strip()
            .lower()
        )
        self._mw._populate_cnn_runtime_options(
            preferred=str(get_cfg("cnn_runtime", default=compute_runtime_cfg))
            .strip()
            .lower()
        )

        # TensorRT batch size is still configurable (runtime-derived usage).
        self._panels.detection.spin_tensorrt_batch.setValue(
            get_cfg("tensorrt_max_batch_size", default=16)
        )
        self._panels.detection.spin_tensorrt_batch.setEnabled(
            bool(
                derive_detection_runtime_settings(self._mw._selected_compute_runtime())[
                    "enable_tensorrt"
                ]
            )
        )
        self._panels.detection.lbl_tensorrt_batch.setEnabled(
            bool(
                derive_detection_runtime_settings(self._mw._selected_compute_runtime())[
                    "enable_tensorrt"
                ]
            )
        )

        # YOLO Batching settings
        self._panels.detection.chk_enable_yolo_batching.setChecked(
            get_cfg("enable_yolo_batching", default=True)
        )
        batch_mode = get_cfg("yolo_batch_size_mode", default="auto")
        self._panels.detection.combo_yolo_batch_mode.setCurrentIndex(
            0 if batch_mode == "auto" else 1
        )
        self._panels.detection.spin_yolo_batch_size.setValue(
            get_cfg("yolo_manual_batch_size", default=16)
        )
        # Re-apply runtime-derived constraints (e.g., TensorRT => manual batch mode).
        self._mw._on_runtime_context_changed()

    def _load_config_core_tracking(self, get_cfg, get_cfg_time):
        self._panels.setup.spin_max_targets.setValue(get_cfg("max_targets", default=4))
        self._panels.tracking.spin_max_dist.setValue(
            get_cfg(
                "max_assignment_distance_multiplier",
                "max_dist_multiplier",
                default=1.5,
            )
        )
        self._panels.tracking.chk_enable_backward.setChecked(
            get_cfg("enable_backward_tracking", default=True)
        )
        self._panels.tracking.spin_kalman_noise.setValue(
            get_cfg("kalman_process_noise", "kalman_noise", default=0.03)
        )
        self._panels.tracking.spin_kalman_meas.setValue(
            get_cfg("kalman_measurement_noise", "kalman_meas_noise", default=0.1)
        )
        self._panels.tracking.spin_kalman_damping.setValue(
            get_cfg("kalman_velocity_damping", "kalman_damping", default=0.95)
        )
        self._panels.tracking.spin_kalman_maturity_age.setValue(
            get_cfg_time(
                "kalman_maturity_age_seconds",
                "kalman_maturity_age",
                default_seconds=0.17,
            )
        )
        self._panels.tracking.spin_kalman_initial_velocity_retention.setValue(
            get_cfg("kalman_initial_velocity_retention", default=0.2)
        )
        self._panels.tracking.spin_kalman_max_velocity.setValue(
            get_cfg("kalman_max_velocity_multiplier", default=2.0)
        )
        self._panels.tracking.spin_kalman_longitudinal_noise.setValue(
            get_cfg("kalman_longitudinal_noise_multiplier", default=5.0)
        )
        self._panels.tracking.spin_kalman_lateral_noise.setValue(
            get_cfg("kalman_lateral_noise_multiplier", default=0.1)
        )
        self._panels.tracking.spin_Wp.setValue(
            get_cfg("weight_position", "W_POSITION", default=1.0)
        )
        self._panels.tracking.spin_Wo.setValue(
            get_cfg("weight_orientation", "W_ORIENTATION", default=1.0)
        )
        self._panels.tracking.spin_Wa.setValue(
            get_cfg("weight_area", "W_AREA", default=0.001)
        )
        self._panels.tracking.spin_Wasp.setValue(
            get_cfg("weight_aspect_ratio", "W_ASPECT", default=0.1)
        )
        self._panels.tracking.chk_use_mahal.setChecked(
            get_cfg("use_mahalanobis_distance", "USE_MAHALANOBIS", default=True)
        )
        self._panels.tracking.combo_assignment_method.setCurrentIndex(
            1 if get_cfg("enable_greedy_assignment", default=False) else 0
        )
        self._panels.tracking.chk_spatial_optimization.setChecked(
            get_cfg("enable_spatial_optimization", default=False)
        )
        self._panels.tracking.spin_assoc_gate_multiplier.setValue(
            get_cfg(
                "association_stage1_motion_gate_multiplier",
                "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER",
                default=1.4,
            )
        )
        self._panels.tracking.spin_assoc_max_area_ratio.setValue(
            get_cfg(
                "association_stage1_max_area_ratio",
                "ASSOCIATION_STAGE1_MAX_AREA_RATIO",
                default=2.5,
            )
        )
        self._panels.tracking.spin_assoc_max_aspect_diff.setValue(
            get_cfg(
                "association_stage1_max_aspect_diff",
                "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF",
                default=0.8,
            )
        )
        self._panels.tracking.chk_enable_pose_rejection.setChecked(
            get_cfg(
                "enable_pose_rejection",
                "ENABLE_POSE_REJECTION",
                default=True,
            )
        )
        self._panels.tracking.spin_pose_rejection_threshold.setValue(
            get_cfg(
                "pose_rejection_threshold",
                "POSE_REJECTION_THRESHOLD",
                default=0.5,
            )
        )
        self._panels.tracking.spin_pose_rejection_min_visibility.setValue(
            get_cfg(
                "pose_rejection_min_visibility",
                "POSE_REJECTION_MIN_VISIBILITY",
                default=0.5,
            )
        )
        self._panels.tracking.spin_track_feature_ema_alpha.setValue(
            get_cfg(
                "track_feature_ema_alpha",
                "TRACK_FEATURE_EMA_ALPHA",
                default=0.85,
            )
        )
        self._panels.tracking.spin_assoc_high_conf_threshold.setValue(
            get_cfg(
                "association_high_confidence_threshold",
                "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD",
                default=0.7,
            )
        )

    def _load_config_orientation_and_lifecycle(self, get_cfg, get_cfg_time):
        self._panels.tracking.spin_velocity.setValue(
            get_cfg("velocity_threshold", default=5.0)
        )
        self._panels.tracking.chk_instant_flip.setChecked(
            get_cfg("enable_instant_flip", "instant_flip", default=True)
        )
        self._panels.tracking.spin_max_orient.setValue(
            get_cfg(
                "max_orientation_delta_stopped",
                "max_orient_delta_stopped",
                default=30.0,
            )
        )
        self._panels.tracking.chk_directed_orient_smoothing.setChecked(
            bool(get_cfg("directed_orient_smoothing", default=True))
        )
        self._panels.tracking.spin_directed_orient_flip_conf.setValue(
            float(get_cfg("directed_orient_flip_confidence", default=0.7))
        )
        self._panels.tracking.spin_directed_orient_flip_persist.setValue(
            int(get_cfg("directed_orient_flip_persistence", default=3))
        )
        self._panels.tracking.spin_lost_thresh.setValue(
            get_cfg_time(
                "lost_threshold_seconds",
                "lost_frames_threshold",
                "lost_threshold_frames",
                default_seconds=0.33,
            )
        )
        self._panels.tracking.spin_min_respawn_distance.setValue(
            get_cfg("min_respawn_distance_multiplier", default=2.5)
        )
        self._panels.tracking.spin_min_detections_to_start.setValue(
            get_cfg_time(
                "min_detections_to_start_seconds",
                "min_detections_to_start",
                default_seconds=0.03,
            )
        )
        self._panels.tracking.spin_min_detect.setValue(
            get_cfg_time(
                "min_detect_seconds",
                "min_detect_frames",
                "min_detect_counts",
                default_seconds=0.33,
            )
        )
        self._panels.tracking.spin_min_track.setValue(
            get_cfg_time(
                "min_track_seconds",
                "min_track_frames",
                "min_track_counts",
                default_seconds=0.33,
            )
        )

    def _load_config_postprocessing(self, get_cfg, get_cfg_time):
        self._panels.postprocess.enable_postprocessing.setChecked(
            get_cfg("enable_postprocessing", default=True)
        )
        _saved_mode = get_cfg("identity_postprocess_mode", default=None)
        if _saved_mode is None:
            # backward compat: migrate from the old boolean
            _saved_mode = (
                "Offline Decoder"
                if bool(get_cfg("enable_offline_identity_decoder", default=False))
                else "Heuristic"
            )
        self._panels.postprocess.cmb_identity_postprocess_mode.setCurrentText(
            str(_saved_mode)
        )
        self._panels.postprocess.chk_prompt_open_refinekit.setChecked(
            bool(get_cfg("prompt_open_refinekit_on_tracking_complete", default=False))
        )
        self._panels.postprocess.set_batch_mode_active(
            self._panels.setup.g_batch.isChecked()
        )
        self._panels.postprocess.spin_min_trajectory_length.setValue(
            get_cfg_time(
                "min_trajectory_length_seconds",
                "min_trajectory_length",
                default_seconds=0.33,
            )
        )
        self._panels.postprocess.spin_max_velocity_break.setValue(
            get_cfg("max_velocity_break", default=50.0)
        )
        self._panels.postprocess.spin_max_occlusion_gap.setValue(
            get_cfg_time(
                "max_occlusion_gap_seconds",
                "max_occlusion_gap",
                default_seconds=1.0,
            )
        )
        self._panels.postprocess.chk_enable_tracklet_relinking.setChecked(
            get_cfg("enable_tracklet_relinking", default=False)
        )
        self._panels.postprocess.spin_relink_pose_max_distance.setValue(
            get_cfg("relink_pose_max_distance", default=0.45)
        )
        self._panels.postprocess.spin_pose_export_min_valid_fraction.setValue(
            get_cfg("pose_export_min_valid_fraction", default=0.5)
        )
        self._panels.postprocess.spin_pose_export_min_valid_keypoints.setValue(
            get_cfg("pose_export_min_valid_keypoints", default=3)
        )
        self._panels.postprocess.spin_relink_min_pose_quality.setValue(
            get_cfg("relink_min_pose_quality", default=0.6)
        )
        self._panels.postprocess.spin_pose_postproc_max_gap.setValue(
            get_cfg("pose_postproc_max_gap", default=5)
        )
        self._panels.postprocess.spin_pose_temporal_outlier_zscore.setValue(
            get_cfg("pose_temporal_outlier_zscore", default=3.0)
        )
        self._panels.tracking.chk_enable_confidence_density_map.setChecked(
            get_cfg("enable_confidence_density_map", default=True)
        )
        self._panels.tracking.spin_density_gaussian_sigma_scale.setValue(
            get_cfg("density_gaussian_sigma_scale", default=1.0)
        )
        self._panels.tracking.spin_density_temporal_sigma.setValue(
            get_cfg("density_temporal_sigma", default=2.0)
        )
        self._panels.tracking.spin_density_binarize_threshold.setValue(
            get_cfg("density_binarize_threshold", default=0.3)
        )
        self._panels.tracking.spin_density_conservative_factor.setValue(
            get_cfg("density_conservative_factor", default=0.7)
        )
        self._panels.tracking.spin_density_min_duration.setValue(
            int(get_cfg("density_min_frame_duration", default=3))
        )
        self._panels.tracking.spin_density_min_area_bodies.setValue(
            float(get_cfg("density_min_area_bodies", default=0.25))
        )
        self._panels.tracking.spin_density_downsample_factor.setValue(
            int(get_cfg("density_downsample_factor", default=8))
        )
        self._panels.tracking.chk_export_confidence_density_video.setChecked(
            get_cfg("export_confidence_density_video", default=False)
        )
        self._mw._on_confidence_density_map_toggled(
            self._panels.tracking.chk_enable_confidence_density_map.checkState()
        )
        self._panels.postprocess.spin_max_velocity_zscore.setValue(
            get_cfg("max_velocity_zscore", default=0.0)
        )
        self._panels.postprocess.spin_velocity_zscore_window.setValue(
            get_cfg_time(
                "velocity_zscore_window_seconds",
                "velocity_zscore_window",
                default_seconds=0.33,
            )
        )
        self._panels.postprocess.spin_velocity_zscore_min_vel.setValue(
            get_cfg("velocity_zscore_min_velocity", default=2.0)
        )
        interp_method = get_cfg("interpolation_method", default="None")
        idx = self._panels.postprocess.combo_interpolation_method.findText(
            interp_method, Qt.MatchFixedString
        )
        if idx >= 0:
            self._panels.postprocess.combo_interpolation_method.setCurrentIndex(idx)
        self._panels.postprocess.spin_interpolation_max_gap.setValue(
            get_cfg_time(
                "interpolation_max_gap_seconds",
                "interpolation_max_gap",
                default_seconds=0.33,
            )
        )
        self._panels.postprocess.spin_identity_interpolation_max_gap.setValue(
            get_cfg_time(
                "identity_interpolation_max_gap_seconds",
                "identity_interpolation_max_gap",
                default_seconds=0.33,
            )
        )
        self._panels.postprocess.spin_heading_flip_max_burst.setValue(
            int(get_cfg("heading_flip_max_burst", default=5))
        )
        self._panels.postprocess.chk_cleanup_temp_files.setChecked(
            get_cfg("cleanup_temp_files", default=True)
        )
        self._panels.postprocess.spin_merge_overlap_multiplier.setValue(
            get_cfg("merge_agreement_distance_multiplier", default=0.5)
        )
        self._panels.postprocess.spin_min_overlap_frames.setValue(
            get_cfg("min_overlap_frames", default=5)
        )
        self._panels.postprocess.spin_identity_disagree_min_run.setValue(
            get_cfg("identity_disagree_min_run", default=5)
        )

    def _load_config_visualization(self, get_cfg):
        self._panels.postprocess.check_show_labels.setChecked(
            get_cfg("video_show_labels", default=True)
        )
        self._panels.postprocess.check_show_orientation.setChecked(
            get_cfg("video_show_orientation", default=True)
        )
        self._panels.postprocess.check_show_trails.setChecked(
            get_cfg("video_show_trails", default=False)
        )
        self._panels.postprocess.spin_trail_duration.setValue(
            get_cfg("video_trail_duration", default=1.0)
        )
        self._panels.postprocess.spin_marker_size.setValue(
            get_cfg("video_marker_size", default=0.3)
        )
        self._panels.postprocess.spin_text_scale.setValue(
            get_cfg("video_text_scale", default=0.5)
        )
        self._panels.postprocess.spin_arrow_length.setValue(
            get_cfg("video_arrow_length", default=0.7)
        )
        self._panels.postprocess.check_video_show_pose.setChecked(
            get_cfg(
                "video_show_pose",
                default=bool(self._mw.advanced_config.get("video_show_pose", True)),
            )
        )
        pose_color_mode = str(
            get_cfg(
                "video_pose_color_mode",
                default=self._mw.advanced_config.get("video_pose_color_mode", "track"),
            )
        ).strip()
        self._panels.postprocess.combo_video_pose_color_mode.setCurrentIndex(
            0 if pose_color_mode == "track" else 1
        )
        self._panels.postprocess.spin_video_pose_point_radius.setValue(
            int(
                get_cfg(
                    "video_pose_point_radius",
                    default=self._mw.advanced_config.get("video_pose_point_radius", 3),
                )
            )
        )
        self._panels.postprocess.spin_video_pose_point_thickness.setValue(
            int(
                get_cfg(
                    "video_pose_point_thickness",
                    default=self._mw.advanced_config.get(
                        "video_pose_point_thickness", -1
                    ),
                )
            )
        )
        self._panels.postprocess.spin_video_pose_line_thickness.setValue(
            int(
                get_cfg(
                    "video_pose_line_thickness",
                    default=self._mw.advanced_config.get(
                        "video_pose_line_thickness", 2
                    ),
                )
            )
        )
        pose_color = get_cfg(
            "video_pose_color",
            default=self._mw.advanced_config.get("video_pose_color", [255, 255, 255]),
        )
        if isinstance(pose_color, (list, tuple)) and len(pose_color) == 3:
            self._panels.postprocess._video_pose_color = tuple(
                int(max(0, min(255, float(v)))) for v in pose_color
            )
            self._panels.postprocess._update_video_pose_color_button()
        self._mw._sync_video_pose_overlay_controls()
        self._panels.setup.chk_show_circles.setChecked(
            get_cfg("show_track_markers", "show_circles", default=True)
        )
        self._panels.setup.chk_show_orientation.setChecked(
            get_cfg("show_orientation_lines", "show_orientation", default=True)
        )
        self._panels.setup.chk_show_trajectories.setChecked(
            get_cfg("show_trajectory_trails", "show_trajectories", default=True)
        )
        self._panels.setup.chk_show_labels.setChecked(
            get_cfg("show_id_labels", "show_labels", default=True)
        )
        self._panels.setup.chk_show_state.setChecked(
            get_cfg("show_state_text", "show_state", default=True)
        )
        self._panels.setup.chk_show_kalman_uncertainty.setChecked(
            get_cfg("show_kalman_uncertainty", default=False)
        )
        self._panels.detection.chk_show_fg.setChecked(
            get_cfg("show_foreground_mask", "show_fg", default=True)
        )
        self._panels.detection.chk_show_bg.setChecked(
            get_cfg("show_background_model", "show_bg", default=True)
        )
        self._panels.detection.chk_show_yolo_obb.setChecked(
            get_cfg("show_yolo_obb", default=False)
        )
        self._panels.setup.spin_traj_hist.setValue(
            get_cfg("trajectory_history_seconds", "traj_history", default=5)
        )
        self._panels.setup.chk_debug_logging.setChecked(
            get_cfg("debug_logging", default=False)
        )
        self._panels.setup.chk_enable_profiling.setChecked(
            get_cfg("enable_profiling", default=False)
        )
        self._mw.slider_zoom.setValue(int(get_cfg("zoom_factor", default=1.0) * 100))

    def _load_config_dataset(self, get_cfg):
        self._panels.dataset.chk_enable_dataset_gen.setChecked(
            get_cfg("enable_dataset_generation", default=False)
        )
        self._panels.dataset.line_dataset_class_name.setText(
            get_cfg("dataset_class_name", default="object")
        )
        self._panels.dataset.spin_dataset_max_frames.setValue(
            get_cfg("dataset_max_frames", default=100)
        )
        self._panels.dataset.spin_dataset_conf_threshold.setValue(
            get_cfg(
                "dataset_confidence_threshold",
                "dataset_conf_threshold",
                default=0.5,
            )
        )
        # Note: dataset YOLO conf/IOU now in advanced_config.json, not per-video config
        self._panels.dataset.spin_dataset_diversity_window.setValue(
            get_cfg("dataset_diversity_window", default=30)
        )
        self._panels.dataset.chk_dataset_include_context.setChecked(
            get_cfg("dataset_include_context", default=True)
        )
        self._panels.dataset.chk_dataset_probabilistic.setChecked(
            get_cfg("dataset_probabilistic_sampling", default=True)
        )
        self._panels.dataset.chk_metric_low_confidence.setChecked(
            get_cfg("metric_low_confidence", default=True)
        )
        self._panels.dataset.chk_metric_count_mismatch.setChecked(
            get_cfg("metric_count_mismatch", default=True)
        )
        self._panels.dataset.chk_metric_fragmented_detections.setChecked(
            get_cfg("metric_fragmented_detections", default=True)
        )
        self._panels.dataset.chk_metric_high_assignment_cost.setChecked(
            get_cfg("metric_high_assignment_cost", default=True)
        )
        self._panels.dataset.chk_metric_track_loss.setChecked(
            get_cfg("metric_track_loss", default=True)
        )
        self._panels.dataset.chk_metric_high_uncertainty.setChecked(
            get_cfg("metric_high_uncertainty", default=False)
        )

    def _load_config_individual_analysis(self, cfg, get_cfg):
        old_method = str(get_cfg("identity_method", default="none_disabled")).lower()
        # Backward compat: rename color_tags_yolo -> cnn_classifier on load
        if old_method == "color_tags_yolo":
            old_method = "cnn_classifier"
        self._panels.identity.g_identity.setChecked(old_method != "none_disabled")

        # --- New format or migrate from old format ---
        _new_cnn_classifiers = get_cfg("cnn_classifiers", default=None)
        if _new_cnn_classifiers is not None:
            # New format: load use_apriltags + cnn_classifiers list
            self._panels.identity.g_apriltags.setChecked(
                bool(get_cfg("use_apriltags", default=False))
            )
            for entry in _new_cnn_classifiers or []:
                row = self._panels.identity._add_cnn_classifier_row()
                row.load_from_config(entry)
        else:
            # Old single-method config: migrate
            if old_method == "apriltags":
                self._panels.identity.g_apriltags.setChecked(True)
            elif old_method in ("cnn_classifier",):
                cnn_model_rel = get_cfg("cnn_classifier_model_path", default="")
                if cnn_model_rel:
                    row = self._panels.identity._add_cnn_classifier_row()
                    row.load_from_config(
                        {
                            "rel_path": cnn_model_rel,
                            "label": get_cfg(
                                "cnn_classifier_label", default="identity"
                            ),
                            "confidence": float(
                                get_cfg("cnn_classifier_confidence", default=0.5)
                            ),
                            "window": int(get_cfg("cnn_classifier_window", default=10)),
                            "unique_identifier": False,
                        }
                    )

        # Identity decoder settings (tracking panel)
        self._panels.tracking.chk_enable_identity_online_decoder.setChecked(
            bool(get_cfg("enable_identity_online_decoder", default=False))
        )
        self._panels.tracking.spin_identity_weight.setValue(
            float(get_cfg("identity_weight", default=1.0))
        )
        self._panels.tracking.spin_identity_commit_threshold.setValue(
            float(get_cfg("identity_commit_threshold", default=0.85))
        )
        self._panels.tracking.spin_identity_display_threshold.setValue(
            float(get_cfg("identity_display_threshold", default=0.6))
        )
        self._panels.tracking.spin_identity_transition_epsilon.setValue(
            float(get_cfg("identity_transition_epsilon", default=0.02))
        )
        self._panels.tracking.spin_identity_unknown_prior.setValue(
            float(get_cfg("identity_unknown_prior", default=0.05))
        )
        self._panels.tracking.spin_identity_rejoin_threshold.setValue(
            float(get_cfg("identity_rejoin_threshold", default=0.5))
        )
        self._panels.postprocess.chk_identity_offline_split_trajectories.setChecked(
            bool(
                get_cfg(
                    "identity_offline_split_trajectories",
                    default=self._mw.advanced_config.get(
                        "identity_offline_split_trajectories", False
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_offline_split_min_conf.setValue(
            float(
                get_cfg(
                    "identity_offline_split_min_conf",
                    default=self._mw.advanced_config.get(
                        "identity_offline_split_min_conf", 0.75
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_offline_split_min_margin.setValue(
            float(
                get_cfg(
                    "identity_offline_split_min_margin",
                    default=self._mw.advanced_config.get(
                        "identity_offline_split_min_margin", 0.2
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_offline_split_min_frames.setValue(
            int(
                get_cfg(
                    "identity_offline_split_min_frames",
                    default=self._mw.advanced_config.get(
                        "identity_offline_split_min_frames", 3
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_offline_split_max_bridge_frames.setValue(
            int(
                get_cfg(
                    "identity_offline_split_max_bridge_frames",
                    default=self._mw.advanced_config.get(
                        "identity_offline_split_max_bridge_frames", 6
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_offline_ilp_time_limit.setValue(
            float(
                get_cfg(
                    "identity_offline_ilp_time_limit",
                    default=self._mw.advanced_config.get(
                        "identity_offline_ilp_time_limit", 30.0
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_offline_ilp_rel_gap.setValue(
            float(
                get_cfg(
                    "identity_offline_ilp_rel_gap",
                    default=self._mw.advanced_config.get(
                        "identity_offline_ilp_rel_gap", 1e-6
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_respawn_prior_strength.setValue(
            float(
                get_cfg(
                    "identity_respawn_prior_strength",
                    default=self._mw.advanced_config.get(
                        "identity_respawn_prior_strength", 0.75
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_respawn_prior_decay.setValue(
            float(
                get_cfg(
                    "identity_respawn_prior_decay",
                    default=self._mw.advanced_config.get(
                        "identity_respawn_prior_decay", 0.97
                    ),
                )
            )
        )
        self._panels.postprocess.spin_identity_respawn_prior_max_gap.setValue(
            int(
                get_cfg(
                    "identity_respawn_prior_max_gap",
                    default=self._mw.advanced_config.get(
                        "identity_respawn_prior_max_gap", 120
                    ),
                )
            )
        )
        apriltag_family = get_cfg("apriltag_family", default="tag36h11")
        idx = self._panels.identity.combo_apriltag_family.findText(apriltag_family)
        self._panels.identity.combo_apriltag_family.setCurrentIndex(max(0, idx))
        self._panels.identity.spin_apriltag_decimate.setValue(
            float(get_cfg("apriltag_decimate", default=1.0))
        )

        # Warn users who had a non-default cnn_classifier_crop_padding in their config
        _legacy_crop_padding = get_cfg("cnn_classifier_crop_padding", default=None)
        if _legacy_crop_padding is not None and float(_legacy_crop_padding) != 0.1:
            logger.warning(
                "Config key 'cnn_classifier_crop_padding' (value=%.2f) is no longer used. "
                "All precompute phases now use 'individual_crop_padding'. "
                "Update your crop padding setting in the Individual Analysis panel.",
                float(_legacy_crop_padding),
            )

        self._panels.identity.chk_enable_pose_extractor.setChecked(
            get_cfg("enable_pose_extractor", default=False)
        )
        pose_backend = str(get_cfg("pose_model_type", default="yolo")).strip().upper()
        pose_backend_idx = self._panels.identity.combo_pose_model_type.findText(
            pose_backend
        )
        if pose_backend_idx >= 0:
            self._panels.identity.combo_pose_model_type.setCurrentIndex(
                pose_backend_idx
            )
        yolo_pose_model = str(get_cfg("pose_yolo_model_dir", default="")).strip()
        sleap_pose_model = str(get_cfg("pose_sleap_model_dir", default="")).strip()
        legacy_pose_model = str(get_cfg("pose_model_dir", default="")).strip()
        if not yolo_pose_model and pose_backend.lower() == "yolo":
            yolo_pose_model = legacy_pose_model
        if not sleap_pose_model and pose_backend.lower() == "sleap":
            sleap_pose_model = legacy_pose_model
        self._mw._set_pose_model_path_for_backend(yolo_pose_model, backend="yolo")
        self._mw._set_pose_model_path_for_backend(sleap_pose_model, backend="sleap")
        active_backend = (
            self._panels.identity.combo_pose_model_type.currentText().strip().lower()
        )
        self._mw._refresh_pose_model_combo(
            preferred_model_path=self._mw._pose_model_path_for_backend(active_backend)
        )
        pose_runtime_flavor = (
            str(
                get_cfg(
                    "pose_runtime_flavor",
                    default=derive_pose_runtime_settings(
                        self._mw._selected_compute_runtime(),
                        backend_family=self._panels.identity.combo_pose_model_type.currentText()
                        .strip()
                        .lower(),
                    )["pose_runtime_flavor"],
                )
            )
            .strip()
            .lower()
        )
        self._mw._populate_pose_runtime_flavor_options(
            backend=self._panels.identity.combo_pose_model_type.currentText()
            .strip()
            .lower(),
            preferred=pose_runtime_flavor,
        )
        self._panels.identity.spin_pose_min_kpt_conf_valid.setValue(
            get_cfg("pose_min_kpt_conf_valid", default=0.2)
        )
        self._panels.identity.line_pose_skeleton_file.setText(
            get_cfg("pose_skeleton_file", default="")
        )
        self._panels.identity._refresh_pose_direction_keypoint_lists()
        pose_group_lists = (
            self._panels.identity.list_pose_ignore_keypoints,
            self._panels.identity.list_pose_direction_anterior,
            self._panels.identity.list_pose_direction_posterior,
        )
        for list_widget in pose_group_lists:
            list_widget.blockSignals(True)
            list_widget.clearSelection()
        ignore_kpts = get_cfg("pose_ignore_keypoints", default=[])
        self._mw._set_pose_group_selection(
            self._panels.identity.list_pose_ignore_keypoints, ignore_kpts
        )
        ant_kpts = get_cfg("pose_direction_anterior_keypoints", default=[])
        self._mw._set_pose_group_selection(
            self._panels.identity.list_pose_direction_anterior, ant_kpts
        )
        post_kpts = get_cfg("pose_direction_posterior_keypoints", default=[])
        self._mw._set_pose_group_selection(
            self._panels.identity.list_pose_direction_posterior, post_kpts
        )
        for list_widget in pose_group_lists:
            list_widget.blockSignals(False)
        self._mw._apply_pose_keypoint_selection_constraints("ignore")
        self._mw.advanced_config["pose_sleap_env"] = str(
            get_cfg("pose_sleap_env", default="sleap")
        )
        self._panels.identity._refresh_pose_sleap_envs()
        shared_pose_batch = int(
            get_cfg(
                "pose_batch_size",
                default=get_cfg(
                    "pose_yolo_batch",
                    default=get_cfg("pose_sleap_batch", default=4),
                ),
            )
        )
        self._panels.identity.spin_pose_batch.setValue(shared_pose_batch)
        self._panels.identity._sync_realtime_individual_batch_ui()

        self._panels.dataset.chk_suppress_foreign_obb_individual_dataset.setChecked(
            get_cfg(
                "suppress_foreign_obb_individual_dataset",
                "suppress_foreign_obb_dataset",
                default=False,
            )
        )
        self._panels.dataset.chk_suppress_foreign_obb_oriented_videos.setChecked(
            get_cfg(
                "suppress_foreign_obb_oriented_videos",
                "suppress_foreign_obb_dataset",
                default=False,
            )
        )
        realtime_mode = bool(
            get_cfg(
                "realtime_tracking_mode",
                default=(
                    str(get_cfg("tracking_workflow_mode", default="non_realtime"))
                    .strip()
                    .lower()
                    == "realtime"
                ),
            )
        )
        self._panels.setup.chk_realtime_mode.setChecked(realtime_mode)
        self._panels.dataset.chk_enable_individual_dataset.setChecked(
            get_cfg(
                "export_final_canonical_images",
                "enable_individual_image_save",
                "enable_individual_dataset",
                default=False,
            )
        )
        self._panels.dataset.chk_generate_individual_track_videos.setChecked(
            get_cfg(
                "final_media_export_videos_enabled",
                "generate_oriented_track_videos",
                default=False,
            )
        )
        self._panels.dataset.chk_fix_oriented_video_direction_flips.setChecked(
            get_cfg(
                "final_media_export_fix_direction_flips",
                "fix_oriented_video_direction_flips",
                default=False,
            )
        )
        self._panels.dataset.spin_oriented_video_heading_flip_burst.setValue(
            get_cfg(
                "final_media_export_heading_flip_burst",
                "oriented_video_heading_flip_burst",
                default=5,
            )
        )
        self._panels.dataset.chk_enable_oriented_video_affine_stabilization.setChecked(
            get_cfg(
                "final_media_export_enable_affine_stabilization",
                "enable_oriented_video_affine_stabilization",
                default=False,
            )
        )
        self._panels.dataset.spin_oriented_video_stabilization_window.setValue(
            get_cfg(
                "final_media_export_stabilization_window",
                "oriented_video_stabilization_window",
                default=5,
            )
        )
        format_text = get_cfg("individual_output_format", default="png").upper()
        format_idx = self._panels.dataset.combo_individual_format.findText(format_text)
        if format_idx >= 0:
            self._panels.dataset.combo_individual_format.setCurrentIndex(format_idx)
        self._panels.dataset.spin_individual_interval.setValue(
            get_cfg("individual_save_interval", default=1)
        )
        self._panels.identity.chk_individual_interpolate.setChecked(
            get_cfg("individual_interpolate_occlusions", default=True)
        )
        self._panels.identity.spin_individual_padding.setValue(
            get_cfg("individual_crop_padding", default=0.1)
        )
        # Load background color
        bg_color = get_cfg("individual_background_color", default=[0, 0, 0])
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            self._panels.identity._background_color = tuple(bg_color)
        self._mw._update_background_color_button()
        self._panels.identity.chk_suppress_foreign_obb.setChecked(
            get_cfg("suppress_foreign_obb_regions", default=True)
        )
        self._mw._sync_individual_analysis_mode_ui()

        # === ROI ===
        self._mw.roi_shapes = cfg.get("roi_shapes", [])
        if self._mw.roi_shapes:
            # Regenerate the combined mask from loaded shapes
            # Need to get video frame dimensions first
            video_path = cfg.get("file_path", "")
            if video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        fh, fw = frame.shape[:2]
                        self._mw._generate_combined_roi_mask(fh, fw)
                        num_shapes = len(self._mw.roi_shapes)
                        shape_summary = ", ".join(
                            [s["type"] for s in self._mw.roi_shapes]
                        )
                        self._mw.roi_status_label.setText(
                            f"Loaded ROI: {num_shapes} shape(s) ({shape_summary})"
                        )
                        self._mw.btn_undo_roi.setEnabled(True)
                        logger.info(f"Loaded {num_shapes} ROI shapes from config")
                    cap.release()
            else:
                # Video not available yet, just store shapes for later
                logger.info(
                    f"Loaded {len(self._mw.roi_shapes)} ROI shapes (mask will be generated when video loads)"
                )

    def _atomic_json_write(self, cfg, path):
        """Write a JSON config atomically. Returns (success, error_message)."""
        import tempfile as _tempfile

        tmp_path = None
        try:
            with _tempfile.NamedTemporaryFile(
                mode="w",
                dir=os.path.dirname(path),
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(cfg, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, path)
            return True, ""
        except Exception as e:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False, str(e)

    def _resolve_config_save_path(self, prompt_if_exists):
        """Determine the config file save path, prompting the user if needed."""
        video_path = self._panels.setup.file_line.text()
        default_path = (
            _get_video_config_path(video_path) if video_path else CONFIG_FILENAME
        )

        if default_path and os.path.exists(default_path) and prompt_if_exists:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Configuration File Exists")
            msg.setText(
                f"A configuration file already exists:\n{os.path.basename(default_path)}"
            )
            msg.setInformativeText(
                "Do you want to replace it or save to a different location?"
            )
            replace_btn = msg.addButton("Replace Existing", QMessageBox.AcceptRole)
            save_as_btn = msg.addButton("Save As...", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Cancel)
            msg.setDefaultButton(replace_btn)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == replace_btn:
                return default_path
            if clicked == save_as_btn:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration As", default_path, "JSON Files (*.json)"
                )
                return path or None
            return None

        if default_path:
            return default_path
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", CONFIG_FILENAME, "JSON Files (*.json)"
        )
        return path or None

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
        from hydra_suite.trackerkit.gui.main_window import (
            get_yolo_model_metadata,
            make_model_path_relative,
            make_pose_model_path_relative,
        )

        self._mw._commit_pending_setup_edits()

        yolo_mode = (
            "sequential"
            if self._panels.detection.combo_yolo_obb_mode.currentIndex() == 1
            else "direct"
        )
        yolo_direct_path = self._mw._get_selected_yolo_model_path()
        yolo_detect_path = self._mw._get_selected_yolo_detect_model_path()
        yolo_crop_obb_path = self._mw._get_selected_yolo_crop_obb_model_path()
        yolo_headtail_path = (
            self._panels.identity._get_configured_yolo_headtail_model_path()
        )
        yolo_path = yolo_direct_path if yolo_mode == "direct" else yolo_crop_obb_path
        yolo_cls = (
            [
                int(x.strip())
                for x in self._panels.detection.line_yolo_classes.text().split(",")
            ]
            if self._panels.detection.line_yolo_classes.text().strip()
            else None
        )

        cfg = {}

        # === PRESET METADATA ===
        # Add name and description when saving as preset
        if preset_mode:
            cfg.update(
                {
                    "preset_name": preset_name or "Custom",
                    "description": preset_description or "User-defined custom preset",
                }
            )

        # === FILE MANAGEMENT ===
        # Skip file paths when saving as preset
        if not preset_mode:
            cfg.update(
                {
                    "file_path": self._panels.setup.file_line.text(),
                    "csv_path": self._panels.setup.csv_line.text(),
                    "video_output_enabled": self._panels.postprocess.check_video_output.isChecked(),
                    "video_output_path": self._panels.postprocess.video_out_line.text(),
                    # Video-specific reference parameters
                    "fps": self._panels.setup.spin_fps.value(),
                    "reference_body_size": self._panels.detection.spin_reference_body_size.value(),
                    # Frame range
                    "start_frame": (
                        self._panels.setup.spin_start_frame.value()
                        if self._panels.setup.spin_start_frame.isEnabled()
                        else 0
                    ),
                    "end_frame": (
                        self._panels.setup.spin_end_frame.value()
                        if self._panels.setup.spin_end_frame.isEnabled()
                        else None
                    ),
                }
            )

        compute_runtime = self._mw._selected_compute_runtime()
        pose_runtime_derived = derive_pose_runtime_settings(
            compute_runtime,
            backend_family=self._panels.identity.combo_pose_model_type.currentText()
            .strip()
            .lower(),
        )

        cfg.update(
            {
                # === SYSTEM PERFORMANCE ===
                "resize_factor": self._panels.setup.spin_resize.value(),
                "save_confidence_metrics": self._panels.setup.check_save_confidence.isChecked(),
                "use_cached_detections": self._panels.setup.chk_use_cached_detections.isChecked(),
                "visualization_free_mode": self._panels.setup.chk_visualization_free.isChecked(),
                "prompt_open_refinekit_on_tracking_complete": self._panels.postprocess.chk_prompt_open_refinekit.isChecked(),
                # === DETECTION STRATEGY ===
                "detection_method": (
                    "background_subtraction"
                    if self._panels.detection.combo_detection_method.currentIndex() == 0
                    else "yolo_obb"
                ),
                # === SIZE FILTERING ===
                "enable_size_filtering": self._panels.detection.chk_size_filtering.isChecked(),
                "min_object_size_multiplier": self._panels.detection.spin_min_object_size.value(),
                "max_object_size_multiplier": self._panels.detection.spin_max_object_size.value(),
                # === IMAGE ENHANCEMENT ===
                "brightness": self._panels.detection.slider_brightness.value(),
                "contrast": self._panels.detection.slider_contrast.value() / 100.0,
                "gamma": self._panels.detection.slider_gamma.value() / 100.0,
                "dark_on_light_background": self._panels.detection.chk_dark_on_light.isChecked(),
                # === BACKGROUND SUBTRACTION ===
                "background_prime_seconds": self._panels.detection.spin_bg_prime.value(),
                "enable_adaptive_background": self._panels.detection.chk_adaptive_bg.isChecked(),
                "background_learning_rate": self._panels.detection.spin_bg_learning.value(),
                "subtraction_threshold": self._panels.detection.spin_threshold.value(),
                # === LIGHTING STABILIZATION ===
                "enable_lighting_stabilization": self._panels.detection.chk_lighting_stab.isChecked(),
                "lighting_smooth_factor": self._panels.detection.spin_lighting_smooth.value(),
                "lighting_median_window": self._panels.detection.spin_lighting_median.value(),
                # === MORPHOLOGY & NOISE ===
                "morph_kernel_size": self._panels.detection.spin_morph_size.value(),
                "min_contour_area": self._panels.detection.spin_min_contour.value(),
                "max_contour_multiplier": self._panels.detection.spin_max_contour_multiplier.value(),
                # === ADVANCED SEPARATION ===
                "enable_conservative_split": self._panels.detection.chk_conservative_split.isChecked(),
                "conservative_kernel_size": self._panels.detection.spin_conservative_kernel.value(),
                "conservative_erode_iterations": self._panels.detection.spin_conservative_erode.value(),
                "enable_additional_dilation": self._panels.detection.chk_additional_dilation.isChecked(),
                "dilation_kernel_size": self._panels.detection.spin_dilation_kernel_size.value(),
                "dilation_iterations": self._panels.detection.spin_dilation_iterations.value(),
                # === YOLO CONFIGURATION ===
                # Store relative path if model is in archive, otherwise absolute
                "yolo_model_path": make_model_path_relative(yolo_path),
                "yolo_obb_mode": yolo_mode,
                "yolo_obb_direct_model_path": make_model_path_relative(
                    yolo_direct_path
                ),
                "yolo_detect_model_path": make_model_path_relative(yolo_detect_path),
                "yolo_crop_obb_model_path": make_model_path_relative(
                    yolo_crop_obb_path
                ),
                "yolo_headtail_model_path": make_model_path_relative(
                    yolo_headtail_path
                ),
                "enable_headtail_orientation": self._panels.identity.g_headtail.isChecked(),
                "pose_overrides_headtail": self._panels.identity.chk_pose_overrides_headtail.isChecked(),
                "yolo_seq_crop_pad_ratio": self._panels.detection.spin_yolo_seq_crop_pad.value(),
                "yolo_seq_min_crop_size_px": self._panels.detection.spin_yolo_seq_min_crop_px.value(),
                "yolo_seq_enforce_square_crop": self._panels.detection.chk_yolo_seq_square_crop.isChecked(),
                "yolo_seq_stage2_imgsz": self._panels.detection.spin_yolo_seq_stage2_imgsz.value(),
                "yolo_seq_individual_batch_size": self._panels.detection.spin_yolo_seq_individual_batch_size.value(),
                "yolo_seq_stage2_pow2_pad": self._panels.detection.chk_yolo_seq_stage2_pow2_pad.isChecked(),
                "yolo_seq_detect_conf_threshold": self._panels.detection.spin_yolo_seq_detect_conf.value(),
                "yolo_headtail_conf_threshold": self._panels.identity.spin_yolo_headtail_conf.value(),
                "yolo_headtail_detect_conf_threshold": self._panels.identity.spin_yolo_headtail_detect_conf.value(),
                "headtail_batch_size": self._panels.identity.spin_headtail_batch.value(),
                "headtail_runtime": self._mw._selected_headtail_runtime(),
                "reference_aspect_ratio": self._panels.detection.spin_reference_aspect_ratio.value(),
                "enable_aspect_ratio_filtering": self._panels.detection.chk_enable_aspect_ratio_filtering.isChecked(),
                "min_aspect_ratio_multiplier": self._panels.detection.spin_min_ar_multiplier.value(),
                "max_aspect_ratio_multiplier": self._panels.detection.spin_max_ar_multiplier.value(),
                "yolo_confidence_threshold": self._panels.detection.spin_yolo_confidence.value(),
                "yolo_iou_threshold": self._panels.detection.spin_yolo_iou.value(),
                "use_custom_obb_iou_filtering": self._panels.detection.chk_use_custom_obb_iou.isChecked(),
                "yolo_target_classes": yolo_cls,
            }
        )
        yolo_meta = get_yolo_model_metadata(yolo_path) or {}
        if yolo_meta:
            cfg["yolo_model_size"] = yolo_meta.get("size", "")
            cfg["yolo_model_species"] = yolo_meta.get("species", "")
            cfg["yolo_model_info"] = yolo_meta.get("model_info", "")
            cfg["yolo_model_added_at"] = yolo_meta.get("added_at", "")

        role_models = {
            "yolo_obb_direct": yolo_direct_path,
            "yolo_seq_detect": yolo_detect_path,
            "yolo_seq_crop_obb": yolo_crop_obb_path,
            "yolo_headtail": yolo_headtail_path,
        }
        for role_key, model_path in role_models.items():
            if not model_path:
                continue
            role_meta = get_yolo_model_metadata(model_path) or {}
            if not role_meta:
                continue
            cfg[f"{role_key}_model_size"] = role_meta.get("size", "")
            cfg[f"{role_key}_model_species"] = role_meta.get("species", "")
            cfg[f"{role_key}_model_info"] = role_meta.get("model_info", "")
            cfg[f"{role_key}_model_added_at"] = role_meta.get("added_at", "")
            cfg[f"{role_key}_task_family"] = role_meta.get("task_family", "")
            cfg[f"{role_key}_usage_role"] = role_meta.get("usage_role", "")

        # === COMPUTE RUNTIME ===
        runtime_detection = derive_detection_runtime_settings(compute_runtime)
        cfg["compute_runtime"] = compute_runtime
        # Keep legacy fields writable for backward compatibility.
        if not preset_mode:
            cfg["yolo_device"] = runtime_detection["yolo_device"]

        cfg.update(
            {
                # TensorRT
                "enable_tensorrt": runtime_detection["enable_tensorrt"],
                "tensorrt_max_batch_size": (
                    self._panels.detection.spin_yolo_batch_size.value()
                    if self._mw._runtime_requires_fixed_yolo_batch(compute_runtime)
                    else self._panels.detection.spin_tensorrt_batch.value()
                ),
                # YOLO Batching
                "enable_yolo_batching": self._panels.detection.chk_enable_yolo_batching.isChecked(),
                "yolo_batch_size_mode": (
                    "auto"
                    if self._panels.detection.combo_yolo_batch_mode.currentIndex() == 0
                    else "manual"
                ),
                "yolo_manual_batch_size": self._panels.detection.spin_yolo_batch_size.value(),
                # === CORE TRACKING ===
                "max_targets": self._panels.setup.spin_max_targets.value(),
                "max_assignment_distance_multiplier": self._panels.tracking.spin_max_dist.value(),
                "enable_backward_tracking": self._panels.tracking.chk_enable_backward.isChecked(),
                # === KALMAN FILTER ===
                "kalman_process_noise": self._panels.tracking.spin_kalman_noise.value(),
                "kalman_measurement_noise": self._panels.tracking.spin_kalman_meas.value(),
                "kalman_velocity_damping": self._panels.tracking.spin_kalman_damping.value(),
                "kalman_maturity_age_seconds": self._panels.tracking.spin_kalman_maturity_age.value(),
                "kalman_initial_velocity_retention": self._panels.tracking.spin_kalman_initial_velocity_retention.value(),
                "kalman_max_velocity_multiplier": self._panels.tracking.spin_kalman_max_velocity.value(),
                "kalman_longitudinal_noise_multiplier": self._panels.tracking.spin_kalman_longitudinal_noise.value(),
                "kalman_lateral_noise_multiplier": self._panels.tracking.spin_kalman_lateral_noise.value(),
                # === COST FUNCTION WEIGHTS ===
                "weight_position": self._panels.tracking.spin_Wp.value(),
                "weight_orientation": self._panels.tracking.spin_Wo.value(),
                "weight_area": self._panels.tracking.spin_Wa.value(),
                "weight_aspect_ratio": self._panels.tracking.spin_Wasp.value(),
                "weight_pose_direction": 0.5,
                "weight_pose_length": 0.0,
                "pose_valid_orientation_scale": 0.15,
                "use_mahalanobis_distance": self._panels.tracking.chk_use_mahal.isChecked(),
                # === ASSIGNMENT ALGORITHM ===
                "enable_greedy_assignment": self._panels.tracking.combo_assignment_method.currentIndex()
                == 1,
                "enable_spatial_optimization": self._panels.tracking.chk_spatial_optimization.isChecked(),
                "association_stage1_motion_gate_multiplier": self._panels.tracking.spin_assoc_gate_multiplier.value(),
                "association_stage1_max_area_ratio": self._panels.tracking.spin_assoc_max_area_ratio.value(),
                "association_stage1_max_aspect_diff": self._panels.tracking.spin_assoc_max_aspect_diff.value(),
                "enable_pose_rejection": self._panels.tracking.chk_enable_pose_rejection.isChecked(),
                "pose_rejection_threshold": self._panels.tracking.spin_pose_rejection_threshold.value(),
                "pose_rejection_min_visibility": self._panels.tracking.spin_pose_rejection_min_visibility.value(),
                "track_feature_ema_alpha": self._panels.tracking.spin_track_feature_ema_alpha.value(),
                "association_high_confidence_threshold": self._panels.tracking.spin_assoc_high_conf_threshold.value(),
                # === ORIENTATION & MOTION ===
                "velocity_threshold": self._panels.tracking.spin_velocity.value(),
                "enable_instant_flip": self._panels.tracking.chk_instant_flip.isChecked(),
                "max_orientation_delta_stopped": self._panels.tracking.spin_max_orient.value(),
                "directed_orient_smoothing": self._panels.tracking.chk_directed_orient_smoothing.isChecked(),
                "directed_orient_flip_confidence": self._panels.tracking.spin_directed_orient_flip_conf.value(),
                "directed_orient_flip_persistence": self._panels.tracking.spin_directed_orient_flip_persist.value(),
                # === TRACK LIFECYCLE ===
                "lost_threshold_seconds": self._panels.tracking.spin_lost_thresh.value(),
                "min_respawn_distance_multiplier": self._panels.tracking.spin_min_respawn_distance.value(),
                "min_detections_to_start_seconds": self._panels.tracking.spin_min_detections_to_start.value(),
                "min_detect_seconds": self._panels.tracking.spin_min_detect.value(),
                "min_track_seconds": self._panels.tracking.spin_min_track.value(),
                # === POST-PROCESSING ===
                "enable_postprocessing": self._panels.postprocess.enable_postprocessing.isChecked(),
                "identity_postprocess_mode": self._panels.postprocess.cmb_identity_postprocess_mode.currentText(),
                "min_trajectory_length_seconds": self._panels.postprocess.spin_min_trajectory_length.value(),
                "max_velocity_break": self._panels.postprocess.spin_max_velocity_break.value(),
                "max_occlusion_gap_seconds": self._panels.postprocess.spin_max_occlusion_gap.value(),
                "enable_tracklet_relinking": self._panels.postprocess.chk_enable_tracklet_relinking.isChecked(),
                "relink_pose_max_distance": self._panels.postprocess.spin_relink_pose_max_distance.value(),
                "pose_export_min_valid_fraction": self._panels.postprocess.spin_pose_export_min_valid_fraction.value(),
                "pose_export_min_valid_keypoints": self._panels.postprocess.spin_pose_export_min_valid_keypoints.value(),
                "relink_min_pose_quality": self._panels.postprocess.spin_relink_min_pose_quality.value(),
                "pose_postproc_max_gap": self._panels.postprocess.spin_pose_postproc_max_gap.value(),
                "pose_temporal_outlier_zscore": self._panels.postprocess.spin_pose_temporal_outlier_zscore.value(),
                "enable_confidence_density_map": self._panels.tracking.chk_enable_confidence_density_map.isChecked(),
                "density_gaussian_sigma_scale": self._panels.tracking.spin_density_gaussian_sigma_scale.value(),
                "density_temporal_sigma": self._panels.tracking.spin_density_temporal_sigma.value(),
                "density_binarize_threshold": self._panels.tracking.spin_density_binarize_threshold.value(),
                "density_conservative_factor": self._panels.tracking.spin_density_conservative_factor.value(),
                "density_min_frame_duration": self._panels.tracking.spin_density_min_duration.value(),
                "density_min_area_bodies": self._panels.tracking.spin_density_min_area_bodies.value(),
                "density_downsample_factor": self._panels.tracking.spin_density_downsample_factor.value(),
                "export_confidence_density_video": self._panels.tracking.chk_export_confidence_density_video.isChecked(),
                "max_velocity_zscore": self._panels.postprocess.spin_max_velocity_zscore.value(),
                "velocity_zscore_window_seconds": self._panels.postprocess.spin_velocity_zscore_window.value(),
                "velocity_zscore_min_velocity": self._panels.postprocess.spin_velocity_zscore_min_vel.value(),
                "interpolation_method": self._panels.postprocess.combo_interpolation_method.currentText(),
                "interpolation_max_gap_seconds": self._panels.postprocess.spin_interpolation_max_gap.value(),
                "identity_interpolation_max_gap_seconds": self._panels.postprocess.spin_identity_interpolation_max_gap.value(),
                "heading_flip_max_burst": self._panels.postprocess.spin_heading_flip_max_burst.value(),
                "cleanup_temp_files": self._panels.postprocess.chk_cleanup_temp_files.isChecked(),
                # === TRAJECTORY MERGING (Conservative Strategy) ===
                # Agreement distance and min overlap frames for conservative merging
                "merge_agreement_distance_multiplier": self._panels.postprocess.spin_merge_overlap_multiplier.value(),
                "min_overlap_frames": self._panels.postprocess.spin_min_overlap_frames.value(),
                "identity_disagree_min_run": self._panels.postprocess.spin_identity_disagree_min_run.value(),
                # === VIDEO VISUALIZATION ===
                "video_show_labels": self._panels.postprocess.check_show_labels.isChecked(),
                "video_show_orientation": self._panels.postprocess.check_show_orientation.isChecked(),
                "video_show_trails": self._panels.postprocess.check_show_trails.isChecked(),
                "video_trail_duration": self._panels.postprocess.spin_trail_duration.value(),
                "video_marker_size": self._panels.postprocess.spin_marker_size.value(),
                "video_text_scale": self._panels.postprocess.spin_text_scale.value(),
                "video_arrow_length": self._panels.postprocess.spin_arrow_length.value(),
                "video_show_pose": self._panels.postprocess.check_video_show_pose.isChecked(),
                "video_pose_color_mode": (
                    "track"
                    if self._panels.postprocess.combo_video_pose_color_mode.currentIndex()
                    == 0
                    else "fixed"
                ),
                "video_pose_color": [
                    int(self._panels.postprocess._video_pose_color[0]),
                    int(self._panels.postprocess._video_pose_color[1]),
                    int(self._panels.postprocess._video_pose_color[2]),
                ],
                "video_pose_point_radius": self._panels.postprocess.spin_video_pose_point_radius.value(),
                "video_pose_point_thickness": self._panels.postprocess.spin_video_pose_point_thickness.value(),
                "video_pose_line_thickness": self._panels.postprocess.spin_video_pose_line_thickness.value(),
                # === VISUALIZATION OVERLAYS ===
                "show_track_markers": self._panels.setup.chk_show_circles.isChecked(),
                "show_orientation_lines": self._panels.setup.chk_show_orientation.isChecked(),
                "show_trajectory_trails": self._panels.setup.chk_show_trajectories.isChecked(),
                "show_id_labels": self._panels.setup.chk_show_labels.isChecked(),
                "show_state_text": self._panels.setup.chk_show_state.isChecked(),
                "show_kalman_uncertainty": self._panels.setup.chk_show_kalman_uncertainty.isChecked(),
                "show_foreground_mask": self._panels.detection.chk_show_fg.isChecked(),
                "show_background_model": self._panels.detection.chk_show_bg.isChecked(),
                "show_yolo_obb": self._panels.detection.chk_show_yolo_obb.isChecked(),
                "trajectory_history_seconds": self._panels.setup.spin_traj_hist.value(),
                "debug_logging": self._panels.setup.chk_debug_logging.isChecked(),
                "enable_profiling": self._panels.setup.chk_enable_profiling.isChecked(),
                "zoom_factor": self._mw.slider_zoom.value() / 100.0,
            }
        )

        # === ROI ===
        # Skip ROI when saving as preset
        if not preset_mode:
            cfg["roi_shapes"] = self._mw.roi_shapes

        cfg.update(
            {
                # === DATASET GENERATION ===
                "enable_dataset_generation": self._panels.dataset.chk_enable_dataset_gen.isChecked(),
                "dataset_class_name": self._panels.dataset.line_dataset_class_name.text(),
                "dataset_max_frames": self._panels.dataset.spin_dataset_max_frames.value(),
            }
        )

        cfg.update(
            {
                "dataset_confidence_threshold": self._panels.dataset.spin_dataset_conf_threshold.value(),
                # Note: dataset YOLO conf/IOU now in advanced_config.json, not per-video config
                "dataset_diversity_window": self._panels.dataset.spin_dataset_diversity_window.value(),
                "dataset_include_context": self._panels.dataset.chk_dataset_include_context.isChecked(),
                "dataset_probabilistic_sampling": self._panels.dataset.chk_dataset_probabilistic.isChecked(),
                "metric_low_confidence": self._panels.dataset.chk_metric_low_confidence.isChecked(),
                "metric_count_mismatch": self._panels.dataset.chk_metric_count_mismatch.isChecked(),
                "metric_fragmented_detections": self._panels.dataset.chk_metric_fragmented_detections.isChecked(),
                "metric_high_assignment_cost": self._panels.dataset.chk_metric_high_assignment_cost.isChecked(),
                "metric_track_loss": self._panels.dataset.chk_metric_track_loss.isChecked(),
                "metric_high_uncertainty": self._panels.dataset.chk_metric_high_uncertainty.isChecked(),
                # === INDIVIDUAL ANALYSIS ===
                "enable_identity_analysis": self._mw._is_individual_pipeline_enabled(),
                "enable_individual_pipeline": self._mw._is_individual_pipeline_enabled(),
                "identity_method": self._mw._selected_identity_method(),
                "use_apriltags": self._mw._identity_config().get(
                    "use_apriltags", False
                ),
                "cnn_classifiers": self._mw._identity_config().get(
                    "cnn_classifiers", []
                ),
                # Legacy CNN Classifier settings (for backward compat on load)
                "cnn_classifier_confidence": self._panels.identity.spin_cnn_confidence.value(),
                "enable_identity_online_decoder": self._panels.tracking.chk_enable_identity_online_decoder.isChecked(),
                "identity_weight": self._panels.tracking.spin_identity_weight.value(),
                "identity_commit_threshold": self._panels.tracking.spin_identity_commit_threshold.value(),
                "identity_display_threshold": self._panels.tracking.spin_identity_display_threshold.value(),
                "identity_transition_epsilon": self._panels.tracking.spin_identity_transition_epsilon.value(),
                "identity_unknown_prior": self._panels.tracking.spin_identity_unknown_prior.value(),
                "identity_rejoin_threshold": self._panels.tracking.spin_identity_rejoin_threshold.value(),
                "identity_offline_split_trajectories": self._panels.postprocess.chk_identity_offline_split_trajectories.isChecked(),
                "identity_offline_split_min_conf": self._panels.postprocess.spin_identity_offline_split_min_conf.value(),
                "identity_offline_split_min_margin": self._panels.postprocess.spin_identity_offline_split_min_margin.value(),
                "identity_offline_split_min_frames": self._panels.postprocess.spin_identity_offline_split_min_frames.value(),
                "identity_offline_split_max_bridge_frames": self._panels.postprocess.spin_identity_offline_split_max_bridge_frames.value(),
                "identity_offline_ilp_time_limit": self._panels.postprocess.spin_identity_offline_ilp_time_limit.value(),
                "identity_offline_ilp_rel_gap": self._panels.postprocess.spin_identity_offline_ilp_rel_gap.value(),
                "identity_respawn_prior_strength": self._panels.postprocess.spin_identity_respawn_prior_strength.value(),
                "identity_respawn_prior_decay": self._panels.postprocess.spin_identity_respawn_prior_decay.value(),
                "identity_respawn_prior_max_gap": self._panels.postprocess.spin_identity_respawn_prior_max_gap.value(),
                "cnn_classifier_window": self._panels.identity.spin_cnn_window.value(),
                "cnn_runtime": self._mw._selected_cnn_runtime(),
            }
        )

        cfg.update(
            {
                "apriltag_family": self._panels.identity.combo_apriltag_family.currentText(),
                "apriltag_decimate": self._panels.identity.spin_apriltag_decimate.value(),
                "enable_pose_extractor": self._panels.identity.chk_enable_pose_extractor.isChecked(),
                "pose_model_type": self._panels.identity.combo_pose_model_type.currentText()
                .strip()
                .lower(),
                "pose_model_dir": make_pose_model_path_relative(
                    self._mw._pose_model_path_for_backend(
                        self._panels.identity.combo_pose_model_type.currentText()
                        .strip()
                        .lower()
                    )
                ),
                "pose_yolo_model_dir": make_pose_model_path_relative(
                    self._mw._pose_model_path_for_backend("yolo")
                ),
                "pose_sleap_model_dir": make_pose_model_path_relative(
                    self._mw._pose_model_path_for_backend("sleap")
                ),
                "pose_runtime_flavor": self._mw._selected_pose_runtime_flavor(),
                "pose_exported_model_path": "",
                "pose_min_kpt_conf_valid": self._panels.identity.spin_pose_min_kpt_conf_valid.value(),
                "pose_skeleton_file": self._panels.identity.line_pose_skeleton_file.text().strip(),
                "pose_ignore_keypoints": self._mw._parse_pose_ignore_keypoints(),
                "pose_direction_anterior_keypoints": self._mw._parse_pose_direction_anterior_keypoints(),
                "pose_direction_posterior_keypoints": self._mw._parse_pose_direction_posterior_keypoints(),
                "pose_batch_size": self._panels.identity.spin_pose_batch.value(),
                "pose_yolo_batch": self._panels.identity.spin_pose_batch.value(),
                "pose_sleap_env": self._mw._selected_pose_sleap_env(),
                "pose_sleap_device": pose_runtime_derived["pose_sleap_device"],
                "pose_sleap_batch": self._panels.identity.spin_pose_batch.value(),
                "pose_sleap_max_instances": 1,
                "tracking_workflow_mode": self._mw._session_orch._workflow_mode_key(),
                "realtime_tracking_mode": self._mw._is_realtime_tracking_mode_enabled(),
                # === FINAL MEDIA EXPORT ===
                "export_final_canonical_images": self._mw._is_individual_image_save_enabled(),
                "enable_individual_dataset": self._mw._is_individual_image_save_enabled(),
                "enable_individual_image_save": self._mw._is_individual_image_save_enabled(),
                "final_media_export_videos_enabled": bool(
                    self._panels.dataset.chk_generate_individual_track_videos.isChecked()
                ),
                "final_media_export_fix_direction_flips": bool(
                    self._panels.dataset.chk_fix_oriented_video_direction_flips.isChecked()
                ),
                "final_media_export_heading_flip_burst": self._panels.dataset.spin_oriented_video_heading_flip_burst.value(),
                "final_media_export_enable_affine_stabilization": bool(
                    self._panels.dataset.chk_enable_oriented_video_affine_stabilization.isChecked()
                ),
                "final_media_export_stabilization_window": self._panels.dataset.spin_oriented_video_stabilization_window.value(),
                "individual_output_format": self._panels.dataset.combo_individual_format.currentText().lower(),
            }
        )

        cfg.update(
            {
                "individual_save_interval": self._panels.dataset.spin_individual_interval.value(),
                "individual_interpolate_occlusions": self._panels.identity.chk_individual_interpolate.isChecked(),
                "individual_crop_padding": self._panels.identity.spin_individual_padding.value(),
                "individual_background_color": [
                    int(c) for c in self._panels.identity._background_color
                ],  # Ensure JSON serializable
                "suppress_foreign_obb_regions": self._panels.identity.chk_suppress_foreign_obb.isChecked(),
                "suppress_foreign_obb_individual_dataset": self._panels.dataset.chk_suppress_foreign_obb_individual_dataset.isChecked(),
                "suppress_foreign_obb_oriented_videos": self._panels.dataset.chk_suppress_foreign_obb_oriented_videos.isChecked(),
            }
        )

        # If preset mode with path provided, save directly
        if preset_mode and preset_path:
            os.makedirs(os.path.dirname(preset_path), exist_ok=True)
            ok, err = self._atomic_json_write(cfg, preset_path)
            if ok:
                logger.info(f"Saved preset to {preset_path}")
                return True
            logger.error(f"Failed to save preset: {err}")
            QMessageBox.critical(
                self._mw, "Save Error", f"Failed to save preset:\n{err}"
            )
            return False

        config_path = self._resolve_config_save_path(prompt_if_exists)
        if not config_path:
            return False

        ok, err = self._atomic_json_write(cfg, config_path)
        if ok:
            logger.info(f"Configuration saved to {config_path} (including ROI shapes)")
            return True
        logger.warning(f"Failed to save configuration: {err}")
        return False

    # =========================================================================
    # PARAMETERS DICT
    # =========================================================================

    def get_parameters_dict(self: object) -> object:
        """get_parameters_dict method documentation."""
        self._mw._commit_pending_setup_edits()

        N = self._panels.setup.spin_max_targets.value()
        np.random.seed(42)
        colors = [tuple(c.tolist()) for c in np.random.randint(0, 255, (N, 3))]

        det_method = (
            "background_subtraction"
            if self._panels.detection.combo_detection_method.currentIndex() == 0
            else "yolo_obb"
        )

        yolo_mode = (
            "sequential"
            if self._panels.detection.combo_yolo_obb_mode.currentIndex() == 1
            else "direct"
        )
        from hydra_suite.trackerkit.gui.main_window import (
            resolve_model_path,
            resolve_pose_model_path,
        )

        yolo_direct_path = resolve_model_path(
            self._mw._get_selected_yolo_model_path() or ""
        )
        yolo_detect_path = resolve_model_path(
            self._mw._get_selected_yolo_detect_model_path() or ""
        )
        yolo_crop_obb_path = resolve_model_path(
            self._mw._get_selected_yolo_crop_obb_model_path() or ""
        )
        yolo_headtail_path = resolve_model_path(
            self._panels.identity._get_selected_yolo_headtail_model_path() or ""
        )
        yolo_path = yolo_direct_path if yolo_mode == "direct" else yolo_crop_obb_path

        yolo_cls = None
        if self._panels.detection.line_yolo_classes.text().strip():
            try:
                yolo_cls = [
                    int(x.strip())
                    for x in self._panels.detection.line_yolo_classes.text().split(",")
                ]
            except ValueError:
                pass

        # Calculate actual pixel values from body-size multipliers
        reference_body_size = self._panels.detection.spin_reference_body_size.value()
        resize_factor = self._panels.setup.spin_resize.value()
        scaled_body_size = reference_body_size * resize_factor

        # Area is π * (diameter/2)^2
        import math

        reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
        scaled_body_area = reference_body_area * (resize_factor**2)

        # Convert multipliers to actual pixels
        min_object_size_pixels = int(
            self._panels.detection.spin_min_object_size.value() * scaled_body_area
        )
        max_object_size_pixels = int(
            self._panels.detection.spin_max_object_size.value() * scaled_body_area
        )
        max_distance_pixels = (
            self._panels.tracking.spin_max_dist.value() * scaled_body_size
        )
        min_respawn_distance_pixels = (
            self._panels.tracking.spin_min_respawn_distance.value() * scaled_body_size
        )

        # Convert time-based velocities to frame-based for tracking
        fps = self._panels.setup.spin_fps.value()
        velocity_threshold_pixels_per_frame = (
            self._panels.tracking.spin_velocity.value() * scaled_body_size / fps
        )
        max_velocity_break_pixels_per_frame = (
            self._panels.postprocess.spin_max_velocity_break.value()
            * scaled_body_size
            / fps
        )

        # Convert time-based durations (seconds) to frame counts
        def _seconds_to_frames(seconds: float, min_frames: int = 1) -> int:
            """Convert a duration in seconds to an integer frame count."""
            return max(min_frames, round(seconds * fps))

        lost_threshold_frames = _seconds_to_frames(
            self._panels.tracking.spin_lost_thresh.value()
        )
        kalman_maturity_age = _seconds_to_frames(
            self._panels.tracking.spin_kalman_maturity_age.value()
        )
        bg_prime_frames = _seconds_to_frames(
            self._panels.detection.spin_bg_prime.value(), min_frames=0
        )
        min_detections_to_start = _seconds_to_frames(
            self._panels.tracking.spin_min_detections_to_start.value()
        )
        min_detection_counts = _seconds_to_frames(
            self._panels.tracking.spin_min_detect.value()
        )
        min_tracking_counts = _seconds_to_frames(
            self._panels.tracking.spin_min_track.value()
        )
        min_trajectory_length = _seconds_to_frames(
            self._panels.postprocess.spin_min_trajectory_length.value()
        )
        max_occlusion_gap = _seconds_to_frames(
            self._panels.postprocess.spin_max_occlusion_gap.value(), min_frames=0
        )
        velocity_zscore_window = _seconds_to_frames(
            self._panels.postprocess.spin_velocity_zscore_window.value(), min_frames=5
        )
        identity_interpolation_max_gap = _seconds_to_frames(
            self._panels.postprocess.spin_identity_interpolation_max_gap.value(),
            min_frames=0,
        )
        # YOLO Batching settings from UI (overrides advanced_config defaults)
        advanced_config = self._mw.advanced_config.copy()
        advanced_config["enable_yolo_batching"] = (
            self._panels.detection.chk_enable_yolo_batching.isChecked()
        )
        advanced_config["yolo_batch_size_mode"] = (
            "auto"
            if self._panels.detection.combo_yolo_batch_mode.currentIndex() == 0
            else "manual"
        )
        advanced_config["yolo_manual_batch_size"] = (
            self._panels.detection.spin_yolo_batch_size.value()
        )
        advanced_config["yolo_seq_individual_batch_size"] = (
            self._panels.detection.spin_yolo_seq_individual_batch_size.value()
        )
        advanced_config["video_show_pose"] = (
            self._panels.postprocess.check_video_show_pose.isChecked()
        )
        advanced_config["video_pose_point_radius"] = (
            self._panels.postprocess.spin_video_pose_point_radius.value()
        )
        advanced_config["video_pose_point_thickness"] = (
            self._panels.postprocess.spin_video_pose_point_thickness.value()
        )
        advanced_config["video_pose_line_thickness"] = (
            self._panels.postprocess.spin_video_pose_line_thickness.value()
        )
        advanced_config["video_pose_color_mode"] = (
            "track"
            if self._panels.postprocess.combo_video_pose_color_mode.currentIndex() == 0
            else "fixed"
        )
        advanced_config["video_pose_color"] = [
            int(self._panels.postprocess._video_pose_color[0]),
            int(self._panels.postprocess._video_pose_color[1]),
            int(self._panels.postprocess._video_pose_color[2]),
        ]
        # Canonical crop / aspect ratio params (from UI widgets)
        advanced_config["reference_aspect_ratio"] = (
            self._panels.detection.spin_reference_aspect_ratio.value()
        )
        advanced_config["enable_aspect_ratio_filtering"] = (
            self._panels.detection.chk_enable_aspect_ratio_filtering.isChecked()
        )
        advanced_config["min_aspect_ratio_multiplier"] = (
            self._panels.detection.spin_min_ar_multiplier.value()
        )
        advanced_config["max_aspect_ratio_multiplier"] = (
            self._panels.detection.spin_max_ar_multiplier.value()
        )
        advanced_config["identity_offline_split_trajectories"] = (
            self._panels.postprocess.chk_identity_offline_split_trajectories.isChecked()
        )
        advanced_config["identity_offline_split_min_conf"] = (
            self._panels.postprocess.spin_identity_offline_split_min_conf.value()
        )
        advanced_config["identity_offline_split_min_margin"] = (
            self._panels.postprocess.spin_identity_offline_split_min_margin.value()
        )
        advanced_config["identity_offline_split_min_frames"] = (
            self._panels.postprocess.spin_identity_offline_split_min_frames.value()
        )
        advanced_config["identity_offline_split_max_bridge_frames"] = (
            self._panels.postprocess.spin_identity_offline_split_max_bridge_frames.value()
        )
        advanced_config["identity_offline_ilp_time_limit"] = (
            self._panels.postprocess.spin_identity_offline_ilp_time_limit.value()
        )
        advanced_config["identity_offline_ilp_rel_gap"] = (
            self._panels.postprocess.spin_identity_offline_ilp_rel_gap.value()
        )
        advanced_config["identity_respawn_prior_strength"] = (
            self._panels.postprocess.spin_identity_respawn_prior_strength.value()
        )
        advanced_config["identity_respawn_prior_decay"] = (
            self._panels.postprocess.spin_identity_respawn_prior_decay.value()
        )
        advanced_config["identity_respawn_prior_max_gap"] = (
            self._panels.postprocess.spin_identity_respawn_prior_max_gap.value()
        )

        individual_pipeline_enabled = self._mw._is_individual_pipeline_enabled()
        final_canonical_image_export_enabled = (
            self._mw._is_individual_image_save_enabled()
        )
        pose_extractor_enabled = self._mw._is_pose_inference_enabled()
        identity_cfg = self._mw._identity_config()
        identity_method = (
            self._mw._selected_identity_method()
        )  # kept for backward compat
        compute_runtime = self._mw._selected_compute_runtime()
        headtail_runtime = self._mw._selected_headtail_runtime()
        cnn_runtime = self._mw._selected_cnn_runtime()
        runtime_detection = derive_detection_runtime_settings(compute_runtime)
        trt_batch_size = (
            self._panels.detection.spin_yolo_batch_size.value()
            if self._mw._runtime_requires_fixed_yolo_batch(compute_runtime)
            else self._panels.detection.spin_tensorrt_batch.value()
        )
        trt_build_batch_size_raw = advanced_config.get(
            "tensorrt_build_batch_size", None
        )
        if trt_build_batch_size_raw in (None, "", 0, "0"):
            trt_build_batch_size = None
        else:
            try:
                trt_build_batch_size = max(1, int(trt_build_batch_size_raw))
            except (TypeError, ValueError):
                trt_build_batch_size = None
        pose_backend_family = (
            self._panels.identity.combo_pose_model_type.currentText().strip().lower()
        )
        selected_pose_runtime = self._mw._selected_pose_runtime_flavor()
        pose_runtime_canonical = {
            "onnx_mps": "onnx_coreml",
            "onnx_coreml": "onnx_coreml",
            "onnx_cpu": "onnx_cpu",
            "onnx_cuda": "onnx_cuda",
            "tensorrt_cuda": "tensorrt",
        }.get(selected_pose_runtime, selected_pose_runtime)
        runtime_pose = derive_pose_runtime_settings(
            pose_runtime_canonical, backend_family=pose_backend_family
        )

        p = {
            "ADVANCED_CONFIG": advanced_config,  # Include advanced config for batch optimization
            "DETECTION_METHOD": det_method,
            "FPS": fps,  # Acquisition frame rate
            # Keep selected frame range stable even when controls are disabled
            # during tracking/backward pass.
            "START_FRAME": self._panels.setup.spin_start_frame.value(),
            "END_FRAME": self._panels.setup.spin_end_frame.value(),
            "YOLO_MODEL_PATH": yolo_path,
            "YOLO_OBB_MODE": yolo_mode,
            "YOLO_OBB_DIRECT_MODEL_PATH": yolo_direct_path,
            "YOLO_DETECT_MODEL_PATH": yolo_detect_path,
            "YOLO_CROP_OBB_MODEL_PATH": yolo_crop_obb_path,
            "YOLO_HEADTAIL_MODEL_PATH": yolo_headtail_path,
            "POSE_OVERRIDES_HEADTAIL": self._panels.identity.chk_pose_overrides_headtail.isChecked(),
            "YOLO_SEQ_CROP_PAD_RATIO": self._panels.detection.spin_yolo_seq_crop_pad.value(),
            "YOLO_SEQ_MIN_CROP_SIZE_PX": self._panels.detection.spin_yolo_seq_min_crop_px.value(),
            "YOLO_SEQ_ENFORCE_SQUARE_CROP": self._panels.detection.chk_yolo_seq_square_crop.isChecked(),
            "YOLO_SEQ_STAGE2_IMGSZ": self._panels.detection.spin_yolo_seq_stage2_imgsz.value(),
            "YOLO_SEQ_INDIVIDUAL_BATCH_SIZE": self._panels.detection.spin_yolo_seq_individual_batch_size.value(),
            "YOLO_SEQ_STAGE2_RUNTIME_BUILD_BATCH_SIZE": self._panels.detection.spin_yolo_seq_individual_batch_size.value(),
            "YOLO_SEQ_STAGE2_POW2_PAD": self._panels.detection.chk_yolo_seq_stage2_pow2_pad.isChecked(),
            "YOLO_SEQ_DETECT_CONF_THRESHOLD": self._panels.detection.spin_yolo_seq_detect_conf.value(),
            "YOLO_HEADTAIL_CONF_THRESHOLD": self._panels.identity.spin_yolo_headtail_conf.value(),
            "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD": self._panels.identity.spin_yolo_headtail_detect_conf.value(),
            "HEADTAIL_BATCH_SIZE": self._panels.identity.spin_headtail_batch.value(),
            "HEADTAIL_COMPUTE_RUNTIME": headtail_runtime,
            "YOLO_CONFIDENCE_THRESHOLD": self._panels.detection.spin_yolo_confidence.value(),
            "YOLO_IOU_THRESHOLD": self._panels.detection.spin_yolo_iou.value(),
            "USE_CUSTOM_OBB_IOU_FILTERING": True,
            "YOLO_TARGET_CLASSES": yolo_cls,
            "COMPUTE_RUNTIME": compute_runtime,
            "CNN_COMPUTE_RUNTIME": cnn_runtime,
            "YOLO_DEVICE": runtime_detection["yolo_device"],
            "ENABLE_GPU_BACKGROUND": runtime_detection["enable_gpu_background"],
            "ENABLE_TENSORRT": runtime_detection["enable_tensorrt"],
            "ENABLE_ONNX_RUNTIME": runtime_detection["enable_onnx_runtime"],
            "TENSORRT_MAX_BATCH_SIZE": trt_batch_size,
            "TENSORRT_BUILD_WORKSPACE_GB": float(
                advanced_config.get("tensorrt_build_workspace_gb", 4.0)
            ),
            "TENSORRT_BUILD_BATCH_SIZE": trt_build_batch_size,
            "MAX_TARGETS": N,
            "THRESHOLD_VALUE": self._panels.detection.spin_threshold.value(),
            "MORPH_KERNEL_SIZE": self._panels.detection.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self._panels.detection.spin_min_contour.value(),
            "ENABLE_SIZE_FILTERING": self._panels.detection.chk_size_filtering.isChecked(),
            "MIN_OBJECT_SIZE": min_object_size_pixels,
            "MAX_OBJECT_SIZE": max_object_size_pixels,
            "MAX_CONTOUR_MULTIPLIER": self._panels.detection.spin_max_contour_multiplier.value(),
            "MAX_DISTANCE_THRESHOLD": max_distance_pixels,
            "MAX_DISTANCE_MULTIPLIER": self._panels.tracking.spin_max_dist.value(),
            "ENABLE_POSTPROCESSING": self._panels.postprocess.enable_postprocessing.isChecked(),
            "MIN_TRAJECTORY_LENGTH": min_trajectory_length,
            "MAX_VELOCITY_BREAK": max_velocity_break_pixels_per_frame,
            "MAX_OCCLUSION_GAP": max_occlusion_gap,
            "ENABLE_TRACKLET_RELINKING": self._panels.postprocess.chk_enable_tracklet_relinking.isChecked(),
            "RELINK_POSE_MAX_DISTANCE": self._panels.postprocess.spin_relink_pose_max_distance.value(),
            "POSE_EXPORT_MIN_VALID_FRACTION": self._panels.postprocess.spin_pose_export_min_valid_fraction.value(),
            "POSE_EXPORT_MIN_VALID_KEYPOINTS": self._panels.postprocess.spin_pose_export_min_valid_keypoints.value(),
            "RELINK_MIN_POSE_QUALITY": self._panels.postprocess.spin_relink_min_pose_quality.value(),
            "POSE_POSTPROC_MAX_GAP": self._panels.postprocess.spin_pose_postproc_max_gap.value(),
            "POSE_TEMPORAL_OUTLIER_ZSCORE": self._panels.postprocess.spin_pose_temporal_outlier_zscore.value(),
            "MAX_VELOCITY_ZSCORE": self._panels.postprocess.spin_max_velocity_zscore.value(),
            "VELOCITY_ZSCORE_WINDOW": velocity_zscore_window,
            "IDENTITY_INTERPOLATION_MAX_GAP": identity_interpolation_max_gap,
            "VELOCITY_ZSCORE_MIN_VELOCITY": self._panels.postprocess.spin_velocity_zscore_min_vel.value()
            * scaled_body_size
            / fps,
            "MIN_RESPAWN_DISTANCE": min_respawn_distance_pixels,
            "MIN_DETECTION_COUNTS": min_detection_counts,
            "MIN_DETECTIONS_TO_START": min_detections_to_start,
            "MIN_TRACKING_COUNTS": min_tracking_counts,
            "TRAJECTORY_HISTORY_SECONDS": self._panels.setup.spin_traj_hist.value(),
            "BACKGROUND_PRIME_FRAMES": bg_prime_frames,
            "ENABLE_LIGHTING_STABILIZATION": self._panels.detection.chk_lighting_stab.isChecked(),
            "ENABLE_ADAPTIVE_BACKGROUND": self._panels.detection.chk_adaptive_bg.isChecked(),
            "BACKGROUND_LEARNING_RATE": self._panels.detection.spin_bg_learning.value(),
            "LIGHTING_SMOOTH_FACTOR": self._panels.detection.spin_lighting_smooth.value(),
            "LIGHTING_MEDIAN_WINDOW": self._panels.detection.spin_lighting_median.value(),
            "KALMAN_NOISE_COVARIANCE": self._panels.tracking.spin_kalman_noise.value(),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": self._panels.tracking.spin_kalman_meas.value(),
            "KALMAN_DAMPING": self._panels.tracking.spin_kalman_damping.value(),
            "KALMAN_MATURITY_AGE": kalman_maturity_age,
            "KALMAN_INITIAL_VELOCITY_RETENTION": self._panels.tracking.spin_kalman_initial_velocity_retention.value(),
            "KALMAN_MAX_VELOCITY_MULTIPLIER": self._panels.tracking.spin_kalman_max_velocity.value(),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": self._panels.tracking.spin_kalman_longitudinal_noise.value(),
            "KALMAN_LATERAL_NOISE_MULTIPLIER": self._panels.tracking.spin_kalman_lateral_noise.value(),
            # Derived anisotropy ratio for the autotune domain banner.
            # Lateral = Longitudinal / ratio, so ratio = long / lat (clamped ≥ 1).
            "KALMAN_ANISOTROPY_RATIO": max(
                1.0,
                self._panels.tracking.spin_kalman_longitudinal_noise.value()
                / max(self._panels.tracking.spin_kalman_lateral_noise.value(), 1e-6),
            ),
            "RESIZE_FACTOR": self._panels.setup.spin_resize.value(),
            "ENABLE_CONSERVATIVE_SPLIT": self._panels.detection.chk_conservative_split.isChecked(),
            "CONSERVATIVE_KERNEL_SIZE": self._panels.detection.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self._panels.detection.spin_conservative_erode.value(),
            "ENABLE_ADDITIONAL_DILATION": self._panels.detection.chk_additional_dilation.isChecked(),
            "DILATION_ITERATIONS": self._panels.detection.spin_dilation_iterations.value(),
            "DILATION_KERNEL_SIZE": self._panels.detection.spin_dilation_kernel_size.value(),
            "BRIGHTNESS": self._panels.detection.slider_brightness.value(),
            "CONTRAST": self._panels.detection.slider_contrast.value() / 100.0,
            "GAMMA": self._panels.detection.slider_gamma.value() / 100.0,
            "DARK_ON_LIGHT_BACKGROUND": self._panels.detection.chk_dark_on_light.isChecked(),
            "VELOCITY_THRESHOLD": velocity_threshold_pixels_per_frame,
            "INSTANT_FLIP_ORIENTATION": self._panels.tracking.chk_instant_flip.isChecked(),
            "MAX_ORIENT_DELTA_STOPPED": self._panels.tracking.spin_max_orient.value(),
            "DIRECTED_ORIENT_SMOOTHING": self._panels.tracking.chk_directed_orient_smoothing.isChecked(),
            "DIRECTED_ORIENT_FLIP_CONFIDENCE": self._panels.tracking.spin_directed_orient_flip_conf.value(),
            "DIRECTED_ORIENT_FLIP_PERSISTENCE": self._panels.tracking.spin_directed_orient_flip_persist.value(),
            # Post-hoc mode: when a directed heading source (head-tail or pose
            # model) is active, skip the online flip-hysteresis check and instead
            # resolve all heading ambiguities globally in post-processing.
            "DIRECTED_ORIENT_POSTHOC_CONSISTENCY": bool(
                str(yolo_headtail_path or "").strip() or pose_extractor_enabled
            ),
            "LOST_THRESHOLD_FRAMES": lost_threshold_frames,
            "W_POSITION": self._panels.tracking.spin_Wp.value(),
            "W_ORIENTATION": self._panels.tracking.spin_Wo.value(),
            "W_AREA": self._panels.tracking.spin_Wa.value(),
            "W_ASPECT": self._panels.tracking.spin_Wasp.value(),
            "W_POSE_DIRECTION": 0.5,
            "W_POSE_LENGTH": 0.0,
            "POSE_VALID_ORIENTATION_SCALE": 0.15,
            "USE_MAHALANOBIS": self._panels.tracking.chk_use_mahal.isChecked(),
            "ENABLE_GREEDY_ASSIGNMENT": self._panels.tracking.combo_assignment_method.currentIndex()
            == 1,
            "ENABLE_SPATIAL_OPTIMIZATION": self._panels.tracking.chk_spatial_optimization.isChecked(),
            "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER": self._panels.tracking.spin_assoc_gate_multiplier.value(),
            "ASSOCIATION_STAGE1_MAX_AREA_RATIO": self._panels.tracking.spin_assoc_max_area_ratio.value(),
            "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF": self._panels.tracking.spin_assoc_max_aspect_diff.value(),
            "ENABLE_POSE_REJECTION": self._panels.tracking.chk_enable_pose_rejection.isChecked(),
            "POSE_REJECTION_THRESHOLD": self._panels.tracking.spin_pose_rejection_threshold.value(),
            "POSE_REJECTION_MIN_VISIBILITY": self._panels.tracking.spin_pose_rejection_min_visibility.value(),
            "TRACK_FEATURE_EMA_ALPHA": self._panels.tracking.spin_track_feature_ema_alpha.value(),
            "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD": self._panels.tracking.spin_assoc_high_conf_threshold.value(),
            "TRAJECTORY_COLORS": colors,
            "SHOW_FG": self._panels.detection.chk_show_fg.isChecked(),
            "SHOW_BG": self._panels.detection.chk_show_bg.isChecked(),
            "SHOW_CIRCLES": self._panels.setup.chk_show_circles.isChecked(),
            "SHOW_ORIENTATION": self._panels.setup.chk_show_orientation.isChecked(),
            "SHOW_YOLO_OBB": self._panels.detection.chk_show_yolo_obb.isChecked(),
            "SHOW_TRAJECTORIES": self._panels.setup.chk_show_trajectories.isChecked(),
            "SHOW_LABELS": self._panels.setup.chk_show_labels.isChecked(),
            "SHOW_STATE": self._panels.setup.chk_show_state.isChecked(),
            "SHOW_KALMAN_UNCERTAINTY": self._panels.setup.chk_show_kalman_uncertainty.isChecked(),
            "VISUALIZATION_FREE_MODE": self._panels.setup.chk_visualization_free.isChecked(),
            "TRACKING_REALTIME_MODE": self._mw._is_realtime_tracking_mode_enabled(),
            "TRACKING_WORKFLOW_MODE": self._mw._session_orch._workflow_mode_key(),
            "zoom_factor": self._mw.slider_zoom.value() / 100.0,
            "ROI_MASK": self._mw.roi_mask,
            "REFERENCE_BODY_SIZE": reference_body_size,
            # Conservative trajectory merging parameters (in resized coordinate space)
            # These are used in resolve_trajectories() for bidirectional merging
            # AGREEMENT_DISTANCE: max distance for frames to be considered "agreeing"
            # MIN_OVERLAP_FRAMES: minimum agreeing frames to consider merge candidates
            "AGREEMENT_DISTANCE": self._panels.postprocess.spin_merge_overlap_multiplier.value()
            * scaled_body_size,
            "MIN_OVERLAP_FRAMES": self._panels.postprocess.spin_min_overlap_frames.value(),
            "IDENTITY_DISAGREE_MIN_RUN": self._panels.postprocess.spin_identity_disagree_min_run.value(),
            # Dataset generation parameters
            "ENABLE_DATASET_GENERATION": self._panels.dataset.chk_enable_dataset_gen.isChecked(),
            "DATASET_NAME": "",
            "DATASET_CLASS_NAME": self._panels.dataset.line_dataset_class_name.text(),
            "DATASET_OUTPUT_DIR": (
                os.path.join(
                    os.path.dirname(self._mw.current_video_path),
                    f"{os.path.splitext(os.path.basename(self._mw.current_video_path))[0]}_datasets",
                    "active_learning",
                )
                if self._mw.current_video_path
                else ""
            ),
            "DATASET_MAX_FRAMES": self._panels.dataset.spin_dataset_max_frames.value(),
            "DATASET_CONF_THRESHOLD": self._panels.dataset.spin_dataset_conf_threshold.value(),
            # Dataset-specific YOLO parameters from advanced config (for export, not tracking)
            "DATASET_YOLO_CONFIDENCE_THRESHOLD": self._mw.advanced_config.get(
                "dataset_yolo_confidence_threshold", 0.05
            ),
            "DATASET_YOLO_IOU_THRESHOLD": self._mw.advanced_config.get(
                "dataset_yolo_iou_threshold", 0.5
            ),
            "DATASET_DIVERSITY_WINDOW": self._panels.dataset.spin_dataset_diversity_window.value(),
            "DATASET_INCLUDE_CONTEXT": self._panels.dataset.chk_dataset_include_context.isChecked(),
            "DATASET_PROBABILISTIC_SAMPLING": self._panels.dataset.chk_dataset_probabilistic.isChecked(),
            "METRIC_LOW_CONFIDENCE": self._panels.dataset.chk_metric_low_confidence.isChecked(),
            "METRIC_COUNT_MISMATCH": self._panels.dataset.chk_metric_count_mismatch.isChecked(),
            "METRIC_FRAGMENTED_DETECTIONS": self._panels.dataset.chk_metric_fragmented_detections.isChecked(),
            "METRIC_HIGH_ASSIGNMENT_COST": self._panels.dataset.chk_metric_high_assignment_cost.isChecked(),
            "METRIC_TRACK_LOSS": self._panels.dataset.chk_metric_track_loss.isChecked(),
            "METRIC_HIGH_UNCERTAINTY": self._panels.dataset.chk_metric_high_uncertainty.isChecked(),
            # Individual analysis parameters
            "ENABLE_IDENTITY_ANALYSIS": individual_pipeline_enabled,
            "ENABLE_INDIVIDUAL_PIPELINE": individual_pipeline_enabled,
            "IDENTITY_METHOD": identity_method,
            "USE_APRILTAGS": identity_cfg.get("use_apriltags", False),
            "CNN_CLASSIFIERS": identity_cfg.get("cnn_classifiers", []),
            "COLOR_TAG_MODEL_PATH": self._panels.identity.line_color_tag_model.text(),
            "COLOR_TAG_CONFIDENCE": self._panels.identity.spin_color_tag_conf.value(),
            "CNN_CLASSIFIER_MODEL_PATH": "",
            "CNN_CLASSIFIER_CONFIDENCE": 0.5,
            "CNN_CLASSIFIER_LABEL": "",
            "CNN_CLASSIFIER_BATCH_SIZE": int(
                identity_cfg.get("cnn_classifiers", [{}])[0].get("batch_size", 64)
                if identity_cfg.get("cnn_classifiers")
                else 64
            ),
            "ENABLE_IDENTITY_ONLINE_DECODER": self._panels.tracking.chk_enable_identity_online_decoder.isChecked(),
            "IDENTITY_POSTPROCESS_MODE": (
                self._panels.postprocess.cmb_identity_postprocess_mode.currentText()
                if self._panels.postprocess.enable_postprocessing.isChecked()
                else "None"
            ),
            "ENABLE_IDENTITY_OFFLINE_DECODER": (
                self._panels.postprocess.enable_postprocessing.isChecked()
                and self._panels.postprocess.cmb_identity_postprocess_mode.currentText()
                == "Offline Decoder"
            ),
            "ASSOCIATION_IDENTITY_HINT_SCALE": self._panels.tracking.spin_identity_weight.value(),
            "IDENTITY_COMMIT_THRESHOLD": self._panels.tracking.spin_identity_commit_threshold.value(),
            "IDENTITY_DISPLAY_THRESHOLD": self._panels.tracking.spin_identity_display_threshold.value(),
            "IDENTITY_TRANSITION_EPSILON": self._panels.tracking.spin_identity_transition_epsilon.value(),
            "IDENTITY_UNKNOWN_PRIOR": self._panels.tracking.spin_identity_unknown_prior.value(),
            "IDENTITY_REJOIN_THRESHOLD": self._panels.tracking.spin_identity_rejoin_threshold.value(),
            "CNN_CLASSIFIER_WINDOW": 10,
            "APRILTAG_FAMILY": self._panels.identity.combo_apriltag_family.currentText(),
            "APRILTAG_DECIMATE": self._panels.identity.spin_apriltag_decimate.value(),
            "IDENTITY_OFFLINE_SPLIT_TRAJECTORIES": bool(
                advanced_config.get("identity_offline_split_trajectories", False)
            ),
            "IDENTITY_OFFLINE_SPLIT_MIN_CONF": float(
                advanced_config.get("identity_offline_split_min_conf", 0.75)
            ),
            "IDENTITY_OFFLINE_SPLIT_MIN_MARGIN": float(
                advanced_config.get("identity_offline_split_min_margin", 0.2)
            ),
            "IDENTITY_OFFLINE_SPLIT_MIN_FRAMES": int(
                advanced_config.get("identity_offline_split_min_frames", 3)
            ),
            "IDENTITY_OFFLINE_SPLIT_MAX_BRIDGE_FRAMES": int(
                advanced_config.get("identity_offline_split_max_bridge_frames", 6)
            ),
            "IDENTITY_OFFLINE_ILP_TIME_LIMIT": float(
                advanced_config.get("identity_offline_ilp_time_limit", 30.0)
            ),
            "IDENTITY_OFFLINE_ILP_REL_GAP": float(
                advanced_config.get("identity_offline_ilp_rel_gap", 1e-6)
            ),
            "IDENTITY_RESPAWN_PRIOR_STRENGTH": float(
                advanced_config.get("identity_respawn_prior_strength", 0.75)
            ),
            "IDENTITY_RESPAWN_PRIOR_DECAY": float(
                advanced_config.get("identity_respawn_prior_decay", 0.97)
            ),
            "IDENTITY_RESPAWN_PRIOR_MAX_GAP": int(
                advanced_config.get("identity_respawn_prior_max_gap", 120)
            ),
            "ENABLE_POSE_EXTRACTOR": pose_extractor_enabled,
            "POSE_MODEL_TYPE": self._panels.identity.combo_pose_model_type.currentText()
            .strip()
            .lower(),
            "POSE_MODEL_DIR": resolve_pose_model_path(
                self._mw._pose_model_path_for_backend(
                    self._panels.identity.combo_pose_model_type.currentText()
                    .strip()
                    .lower()
                ),
                backend=self._panels.identity.combo_pose_model_type.currentText()
                .strip()
                .lower(),
            ),
            "POSE_RUNTIME_FLAVOR": self._mw._selected_pose_runtime_flavor(),
            "POSE_EXPORTED_MODEL_PATH": "",
            "POSE_MIN_KPT_CONF_VALID": self._panels.identity.spin_pose_min_kpt_conf_valid.value(),
            "POSE_SKELETON_FILE": self._panels.identity.line_pose_skeleton_file.text().strip(),
            "POSE_IGNORE_KEYPOINTS": self._mw._parse_pose_ignore_keypoints(),
            "POSE_DIRECTION_ANTERIOR_KEYPOINTS": self._mw._parse_pose_direction_anterior_keypoints(),
            "POSE_DIRECTION_POSTERIOR_KEYPOINTS": self._mw._parse_pose_direction_posterior_keypoints(),
            "POSE_YOLO_BATCH": self._panels.identity.spin_pose_batch.value(),
            "POSE_BATCH_SIZE": self._panels.identity.spin_pose_batch.value(),
            "POSE_SLEAP_ENV": self._mw._selected_pose_sleap_env(),
            "POSE_SLEAP_DEVICE": runtime_pose["pose_sleap_device"],
            "POSE_SLEAP_BATCH": self._panels.identity.spin_pose_batch.value(),
            "POSE_SLEAP_MAX_INSTANCES": 1,
            "INDIVIDUAL_PROPERTIES_CACHE_PATH": str(
                self._mw.current_individual_properties_cache_path or ""
            ).strip(),
            # Final media export parameters
            "ENABLE_INDIVIDUAL_DATASET": False,
            "ENABLE_INDIVIDUAL_IMAGE_SAVE": False,
            "EXPORT_FINAL_CANONICAL_IMAGES": final_canonical_image_export_enabled,
            "FINAL_MEDIA_EXPORT_VIDEOS_ENABLED": self._mw._should_export_final_media_videos(),
            "FINAL_MEDIA_EXPORT_FIX_DIRECTION_FLIPS": bool(
                self._panels.dataset.chk_fix_oriented_video_direction_flips.isChecked()
            ),
            "FINAL_MEDIA_EXPORT_HEADING_FLIP_MAX_BURST": self._panels.dataset.spin_oriented_video_heading_flip_burst.value(),
            "FINAL_MEDIA_EXPORT_ENABLE_AFFINE_STABILIZATION": bool(
                self._panels.dataset.chk_enable_oriented_video_affine_stabilization.isChecked()
            ),
            "FINAL_MEDIA_EXPORT_STABILIZATION_WINDOW": self._panels.dataset.spin_oriented_video_stabilization_window.value(),
            "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR": (
                os.path.join(
                    os.path.dirname(self._mw.current_video_path),
                    f"{os.path.splitext(os.path.basename(self._mw.current_video_path))[0]}_datasets",
                    "oriented_videos",
                )
                if self._mw.current_video_path
                else ""
            ),
            "INDIVIDUAL_DATASET_NAME": (
                ""
                if str(
                    self._panels.identity._get_selected_yolo_headtail_model_path() or ""
                ).strip()
                else "unoriented"
            ),
            "INDIVIDUAL_DATASET_OUTPUT_DIR": (
                os.path.join(
                    os.path.dirname(self._mw.current_video_path),
                    f"{os.path.splitext(os.path.basename(self._mw.current_video_path))[0]}_datasets",
                    "individual_crops",
                )
                if self._mw.current_video_path
                else ""
            ),
            "INDIVIDUAL_OUTPUT_FORMAT": self._panels.dataset.combo_individual_format.currentText().lower(),
            "INDIVIDUAL_SAVE_INTERVAL": self._panels.dataset.spin_individual_interval.value(),
            "INDIVIDUAL_INTERPOLATE_OCCLUSIONS": self._panels.identity.chk_individual_interpolate.isChecked(),
            "INDIVIDUAL_CROP_PADDING": self._panels.identity.spin_individual_padding.value(),
            "INDIVIDUAL_BACKGROUND_COLOR": [
                int(c) for c in self._panels.identity._background_color
            ],  # Ensure JSON serializable
            "SUPPRESS_FOREIGN_OBB_REGIONS": self._panels.identity.chk_suppress_foreign_obb.isChecked(),
            "SUPPRESS_FOREIGN_OBB_DATASET": self._panels.dataset.chk_suppress_foreign_obb_individual_dataset.isChecked(),
            "SUPPRESS_FOREIGN_OBB_ORIENTED_VIDEO": self._panels.dataset.chk_suppress_foreign_obb_oriented_videos.isChecked(),
            "INDIVIDUAL_DATASET_RUN_ID": self._mw._individual_dataset_run_id,
            "ENABLE_CONFIDENCE_DENSITY_MAP": self._panels.tracking.chk_enable_confidence_density_map.isChecked(),
            "DENSITY_GAUSSIAN_SIGMA_SCALE": self._panels.tracking.spin_density_gaussian_sigma_scale.value(),
            "DENSITY_TEMPORAL_SIGMA": self._panels.tracking.spin_density_temporal_sigma.value(),
            "DENSITY_BINARIZE_THRESHOLD": self._panels.tracking.spin_density_binarize_threshold.value(),
            "DENSITY_CONSERVATIVE_FACTOR": self._panels.tracking.spin_density_conservative_factor.value(),
            "DENSITY_MIN_FRAME_DURATION": self._panels.tracking.spin_density_min_duration.value(),
            "DENSITY_MIN_AREA_BODIES": self._panels.tracking.spin_density_min_area_bodies.value(),
            "DENSITY_DOWNSAMPLE_FACTOR": self._panels.tracking.spin_density_downsample_factor.value(),
            "EXPORT_CONFIDENCE_DENSITY_VIDEO": self._panels.tracking.chk_export_confidence_density_video.isChecked(),
            "ENABLE_PROFILING": self._panels.setup.chk_enable_profiling.isChecked(),
        }

        # Backward compat: map old color_tag keys to new cnn_classifier keys
        if not p.get("CNN_CLASSIFIER_MODEL_PATH"):
            p["CNN_CLASSIFIER_MODEL_PATH"] = p.get("COLOR_TAG_MODEL_PATH", "")

        return p

    # =========================================================================
    # PRESET MANAGEMENT
    # =========================================================================

    def _populate_preset_combo(self):
        """Populate the preset combo box by auto-scanning configs folder."""
        presets_dir = self._mw._get_presets_dir()

        if not os.path.exists(presets_dir):
            return

        presets = []
        custom_preset = None

        # Scan all JSON files in configs directory
        for filename in sorted(os.listdir(presets_dir)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(presets_dir, filename)

            try:
                with open(filepath, "r") as f:
                    cfg = json.load(f)

                # Get preset name from file, fallback to filename
                preset_name = cfg.get(
                    "preset_name",
                    filename.replace(".json", "").replace("_", " ").title(),
                )

                # Custom preset gets special treatment (star marker, shown first)
                if filename == "custom.json":
                    custom_preset = (f"{preset_name} ★", filepath)
                else:
                    presets.append((preset_name, filepath))
            except Exception as e:
                logger.warning(f"Failed to load preset {filename}: {e}")
                continue

        # Populate combo box (custom first, then others alphabetically)
        self._panels.setup.combo_presets.clear()
        if custom_preset:
            self._panels.setup.combo_presets.addItem(custom_preset[0], custom_preset[1])
        for name, filepath in presets:
            self._panels.setup.combo_presets.addItem(name, filepath)

    def _load_selected_preset(self):
        """Load the currently selected preset."""
        filepath = self._panels.setup.combo_presets.currentData()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(
                self, "Preset Not Found", f"Preset file not found: {filepath}"
            )
            return

        # Confirm if current settings differ significantly
        reply = QMessageBox.question(
            self,
            "Load Preset",
            f"Load preset: {self._panels.setup.combo_presets.currentText()}?\n\n"
            "This will replace your current parameter values.\n"
            "(Video-specific configs will still override presets when loading videos)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            # Use existing config loader in preset mode
            self._load_config_from_file(filepath, preset_mode=True)

            # Update status
            preset_name = self._panels.setup.combo_presets.currentText()
            self._panels.setup.preset_status_label.setText(f"✓ Loaded: {preset_name}")
            self._panels.setup.preset_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            self._panels.setup.preset_status_label.setVisible(True)
            logger.info(f"Loaded preset: {preset_name} from {filepath}")

    def _save_custom_preset(self):
        """Save current settings as custom preset with user-defined name and description."""
        from PySide6.QtWidgets import QTextEdit

        # Create dialog for preset metadata
        dialog = QDialog(self._mw)
        dialog.setWindowTitle("Save Preset")
        dialog.setModal(True)
        dialog_layout = QVBoxLayout(dialog)

        # Name input
        name_label = QLabel("Preset name (e.g., danio rerio / zebrafish)")
        name_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        name_input = QLineEdit()
        name_input.setPlaceholderText("Scientific Name (Common Name)")
        name_input.setText("Custom")

        # Description input
        desc_label = QLabel("Description (optional)")
        desc_label.setStyleSheet("color: #e0e0e0; font-weight: bold; margin-top: 10px;")
        desc_input = QTextEdit()
        desc_input.setPlaceholderText(
            "Describe the optimizations or use case for this preset..."
        )
        desc_input.setMaximumHeight(80)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        dialog_layout.addWidget(name_label)
        dialog_layout.addWidget(name_input)
        dialog_layout.addWidget(desc_label)
        dialog_layout.addWidget(desc_input)
        dialog_layout.addWidget(button_box)

        if dialog.exec() != QDialog.Accepted:
            return

        preset_name = name_input.text().strip() or "Custom"
        description = desc_input.toPlainText().strip() or "User-defined custom preset"

        # Ask user for filename
        presets_dir = self._mw._get_presets_dir()
        os.makedirs(presets_dir, exist_ok=True)

        # Generate suggested filename from preset name
        suggested_filename = preset_name.lower()
        suggested_filename = suggested_filename.split("(")[
            0
        ].strip()  # Remove common name part
        suggested_filename = suggested_filename.replace(" ", "_").replace(".", "")
        suggested_filename = "".join(
            c for c in suggested_filename if c.isalnum() or c == "_"
        )
        suggested_filename = suggested_filename or "custom"
        suggested_path = os.path.join(presets_dir, f"{suggested_filename}.json")

        preset_path, _ = QFileDialog.getSaveFileName(
            self._mw, "Save Preset As", suggested_path, "JSON Files (*.json)"
        )

        if not preset_path:
            return

        custom_path = preset_path

        # Use existing config saver in preset mode with custom metadata
        success = self.save_config(
            preset_mode=True,
            preset_path=custom_path,
            preset_name=preset_name,
            preset_description=description,
        )

        if success:
            # Refresh combo box to show new preset
            self._populate_preset_combo()
            # Select the newly saved preset
            for i in range(self._panels.setup.combo_presets.count()):
                item_path = self._panels.setup.combo_presets.itemData(i)
                if item_path == custom_path:
                    self._panels.setup.combo_presets.setCurrentIndex(i)
                    break

            self._panels.setup.preset_status_label.setText(f"✓ Saved: {preset_name}")
            self._panels.setup.preset_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            self._panels.setup.preset_status_label.setVisible(True)

            filename = os.path.basename(custom_path)
            is_custom = filename == "custom.json"

            QMessageBox.information(
                self,
                "Preset Saved",
                f"Your settings have been saved as:\n{preset_name}\n\n"
                f"Location: {custom_path}\n\n"
                + (
                    "This preset will be loaded automatically on startup and will appear\n"
                    "at the top of the preset selector with a ★ indicator."
                    if is_custom
                    else "This preset is now available in the preset selector."
                ),
            )

    def _load_default_preset_on_startup(self):
        """Load default preset on application startup."""
        presets_dir = self._mw._get_presets_dir()

        # Try custom preset first
        custom_path = os.path.join(presets_dir, "custom.json")
        if os.path.exists(custom_path):
            logger.info("Loading custom preset on startup")
            self._load_config_from_file(custom_path, preset_mode=True)
            return

        # Fall back to default preset
        default_path = os.path.join(presets_dir, "default.json")
        if os.path.exists(default_path):
            logger.info("Loading default preset on startup")
            self._load_config_from_file(default_path, preset_mode=True)
            return

        logger.info("No preset found, using hardcoded defaults")

    # =========================================================================
    # ROI OPTIMIZATION AND VIDEO CROPPING
    # =========================================================================

    # =========================================================================
    # ADVANCED CONFIG
    # =========================================================================

    def _load_advanced_config(self):
        """Load advanced configuration for power users."""
        # Store config in the package directory (where this file is located)
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
            "yolo_headtail_detect_conf_threshold": 0.25,  # Minimum detection confidence before head-tail inference runs; lower-confidence detections remain unknown
            "headtail_batch_size": 64,  # Canonical crop batch size for head-tail classifier inference
            "realtime_visualization_emit_stride": 1,  # Emit GUI overlays every Nth frame during realtime tracking while preserving full-speed tracking/video output
            "visualization_emit_stride": 1,  # Optional GUI overlay decimation for non-realtime runs
            # Identity decoding - offline splitter / global solver / respawn priors
            "identity_offline_split_trajectories": False,
            "identity_offline_split_min_conf": 0.75,
            "identity_offline_split_min_margin": 0.2,
            "identity_offline_split_min_frames": 3,
            "identity_offline_split_max_bridge_frames": 6,
            "identity_offline_ilp_time_limit": 30.0,
            "identity_offline_ilp_rel_gap": 1e-6,
            "identity_respawn_prior_strength": 0.75,
            "identity_respawn_prior_decay": 0.97,
            "identity_respawn_prior_max_gap": 120,
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
        from hydra_suite.paths import get_advanced_config_path

        config_path = str(get_advanced_config_path())
        try:
            import tempfile

            # Write to temp file first, then rename (atomic on most filesystems)
            config_dir = os.path.dirname(config_path)
            os.makedirs(config_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=config_dir, delete=False, suffix=".tmp"
            ) as tmp:
                json.dump(self._mw.advanced_config, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, config_path)  # Atomic rename
            logger.info(f"Saved advanced config to {config_path}")
        except Exception as e:
            logger.error(f"Could not save advanced config: {e}")

    # =========================================================================
    # VIDEO SETUP
    # =========================================================================

    def _setup_video_file(self, fp, skip_config_load=False):
        """
        Setup a video file for tracking.

        Args:
            fp: Path to the video file
            skip_config_load: If True, skip auto-loading config (used when loading config itself)
        """
        self._panels.setup.file_line.setText(fp)
        self._mw.current_video_path = fp

        # Reset caches for the new video
        self._mw.current_detection_cache_path = None
        self._mw.current_individual_properties_cache_path = None

        if self._mw.roi_selection_active:
            self._mw.clear_roi()

        # Auto-generate output paths based on video name
        video_dir = os.path.dirname(fp)
        video_name = os.path.splitext(os.path.basename(fp))[0]

        # Auto-populate CSV output
        csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
        self._panels.setup.csv_line.setText(csv_path)

        # Auto-populate video output and enable it
        video_out_path = os.path.join(video_dir, f"{video_name}_tracking.mp4")
        self._panels.postprocess.video_out_line.setText(video_out_path)
        self._panels.postprocess.check_video_output.setChecked(True)

        # Enable preview detection button
        self._mw.btn_test_detection.setEnabled(True)
        self._panels.setup.btn_detect_fps.setEnabled(True)

        # Initialize video player
        self._mw._init_video_player(fp)

        # Update window title
        self._mw.setWindowTitle(f"HYDRA - {os.path.basename(fp)}")

        # Update Start/End frame spins
        self._panels.setup.spin_start_frame.setValue(0)
        self._panels.setup.spin_end_frame.setValue(self._mw.video_total_frames - 1)

        # Auto-load config if it exists for this video (unless explicitly skipped)
        if not skip_config_load:
            config_path = _get_video_config_path(fp)
            if config_path and os.path.isfile(config_path):
                self._load_config_from_file(config_path)
                self._panels.setup.config_status_label.setText(
                    f"✓ Loaded: {os.path.basename(config_path)}"
                )
        else:
            self._panels.setup.config_status_label.setText(
                "ℹ️ Using current UI parameters (Keystone)"
            )
            self._panels.setup.config_status_label.setStyleSheet(
                "color: #f39c12; font-style: italic; font-size: 10px;"
            )

        # Enable controls
        self._mw._apply_ui_state("idle")
        if hasattr(self._mw, "_recents_store"):
            self._mw._recents_store.add(fp)
        self._mw._show_workspace()

    # =========================================================================
    # ROI UTILITIES
    # =========================================================================

    def _invalidate_roi_cache(self):
        """Invalidate ROI display cache when ROI changes."""
        self._mw._roi_masked_cache.clear()

    # =========================================================================
    # PRESET MANAGEMENT
    # =========================================================================

    def _calculate_roi_bounding_box(self, padding=None):
        """Calculate the bounding box of the current ROI mask with optional padding.

        Args:
            padding: Fraction of min(width, height) to add as padding (e.g., 0.05 = 5%).
                    If None, uses value from advanced config.

        Returns:
            Tuple (x, y, w, h) or None if no ROI
        """
        if self._mw.roi_mask is None:
            return None

        # Find all non-zero points in the mask
        points = cv2.findNonZero(self._mw.roi_mask)
        if points is None or len(points) == 0:
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)

        # Apply padding if requested (helps with detection by providing context)
        if padding is None:
            padding = self._mw.advanced_config.get("roi_crop_padding_fraction", 0.05)

        if padding > 0:
            # Get frame dimensions
            frame_h, frame_w = self._mw.roi_mask.shape[:2]

            # Calculate padding in pixels based on smaller dimension
            min_dim = min(frame_w, frame_h)
            padding_pixels = int(min_dim * padding)

            # Add padding while staying within frame bounds
            x = max(0, x - padding_pixels)
            y = max(0, y - padding_pixels)
            w = min(frame_w - x, w + 2 * padding_pixels)
            h = min(frame_h - y, h + 2 * padding_pixels)

        return (x, y, w, h)

    def _estimate_roi_efficiency(self):
        """Estimate the efficiency gain from cropping to ROI.

        Returns:
            tuple: (roi_coverage_percent, potential_speedup_factor) or (None, None)
        """
        if self._mw.roi_mask is None or self._mw.preview_frame_original is None:
            return (None, None)

        bbox = self._calculate_roi_bounding_box()
        if bbox is None:
            return (None, None)

        x, y, w, h = bbox
        frame_h, frame_w = self._mw.roi_mask.shape

        frame_area = frame_w * frame_h
        roi_area = w * h

        roi_coverage = roi_area / frame_area
        # Speedup is roughly inverse of area ratio (simplification, but good estimate)
        potential_speedup = 1.0 / roi_coverage if roi_coverage > 0 else 1.0

        return (roi_coverage * 100, potential_speedup)

    def _update_roi_optimization_info(self):
        """Update the ROI optimization label with efficiency information."""
        coverage, speedup = self._estimate_roi_efficiency()

        if coverage is None:
            if hasattr(self, "roi_optimization_label"):
                self._mw.roi_optimization_label.setText("")
            return

        threshold = (
            self._mw.advanced_config.get("roi_crop_warning_threshold", 0.6) * 100
        )

        if coverage < threshold and hasattr(self, "roi_optimization_label"):
            self._mw.roi_optimization_label.setText(
                f"⚡ ROI is {coverage:.1f}% of frame - up to {speedup:.1f}x faster if cropped!"
            )
        elif hasattr(self, "roi_optimization_label"):
            self._mw.roi_optimization_label.setText("")

    def crop_video_to_roi(self: object) -> object:
        """Crop the video to the ROI bounding box and save as new file."""
        if self._mw.roi_mask is None:
            QMessageBox.warning(
                self._mw, "No ROI", "Please define an ROI before cropping."
            )
            return

        video_path = self._panels.setup.file_line.text()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self._mw, "No Video", "Please load a video first.")
            return

        # Get padding fraction from advanced config (default 5% of min dimension)
        padding_fraction = self._mw.advanced_config.get(
            "roi_crop_padding_fraction", 0.05
        )

        bbox = self._calculate_roi_bounding_box(padding=padding_fraction)
        if bbox is None:
            QMessageBox.warning(
                self, "Invalid ROI", "Could not calculate ROI bounding box."
            )
            return

        x, y, w, h = bbox
        padding_percent = padding_fraction * 100
        logger.info(
            f"ROI crop with {padding_percent:.1f}% padding: x={x}, y={y}, w={w}, h={h}"
        )

        # Suggest output filename
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        suggested_name = f"{video_name}_cropped_roi.mp4"
        suggested_path = os.path.join(video_dir, suggested_name)

        # Ask user for output location
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cropped Video",
            suggested_path,
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )

        if not output_path:
            return  # User cancelled

        try:
            # Use ffmpeg for high-quality cropping that preserves video properties
            # This is much faster and maintains quality better than re-encoding with OpenCV
            import subprocess

            # Get video properties for the success message
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open source video")

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Build ffmpeg command for high-quality cropping
            # Get settings from advanced config
            codec = self._mw.advanced_config.get("video_crop_codec", "libx264")
            crf = str(self._mw.advanced_config.get("video_crop_crf", 18))
            preset = self._mw.advanced_config.get("video_crop_preset", "medium")

            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-filter:v",
                f"crop={w}:{h}:{x}:{y}",  # Crop to ROI bounding box
                "-c:v",
                codec,  # Video codec from config
                "-crf",
                crf,  # Quality (lower = better, 18 is visually lossless)
                "-preset",
                preset,  # Encoding speed preset
                "-c:a",
                "copy",  # Copy audio without re-encoding
                "-movflags",
                "+faststart",  # Enable streaming/fast preview
                "-y",  # Overwrite output file
                output_path,
            ]

            # Log the ffmpeg command for debugging
            logger.info(
                f"Starting video crop: {frame_w}x{frame_h} -> {w}x{h} (padding: {padding_percent:.1f}%)"
            )
            logger.info(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")

            # Show non-blocking message
            QMessageBox.information(
                self,
                "Cropping Video",
                f"Video cropping has started in the background.\n\n"
                f"Original: {frame_w}x{frame_h}\n"
                f"Cropped: {w}x{h}\n"
                f"Padding: {padding_percent:.1f}% of frame (improves detection)\n\n"
                f"Note: Padding provides visual context for better YOLO confidence.\n"
                f"Adjust 'roi_crop_padding_fraction' in advanced_config.json if needed.\n\n"
                f"This may take a few minutes. The application will remain responsive.\n"
                f"You'll be notified when cropping is complete.",
            )

            # Run ffmpeg in background with progress logging

            # Run ffmpeg in background with progress logging
            # Capture stderr for progress tracking but don't block UI
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                start_new_session=True,  # Detach from parent process
            )

            # Get total frames for progress tracking
            cap_temp = cv2.VideoCapture(video_path)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_temp.release()

            # Store process info for potential future use
            self._mw._crop_process = {
                "process": process,
                "output_path": output_path,
                "original_size": (frame_w, frame_h),
                "cropped_size": (w, h),
                "total_frames": total_frames,
                "last_logged_progress": 0,
            }

            # Set up a timer to check when process completes
            from PySide6.QtCore import QTimer

            self._mw._crop_check_timer = QTimer()
            self._mw._crop_check_timer.timeout.connect(self._check_crop_completion)
            self._mw._crop_check_timer.start(2000)  # Check every 2 seconds

            # Disable UI controls while cropping is in progress
            self._mw._set_ui_controls_enabled(False)
            # Also disable crop button specifically
            if hasattr(self, "btn_crop_video"):
                self._mw.btn_crop_video.setText("Cropping...")
                self._mw.btn_crop_video.setEnabled(False)

            logger.info(f"Started background video crop: {video_path} -> {output_path}")

        except Exception as e:
            # Re-enable UI if crop failed to start
            self._mw._set_ui_controls_enabled(True)
            if hasattr(self, "btn_crop_video"):
                self._mw.btn_crop_video.setText("Crop Video to ROI")
                self._mw.btn_crop_video.setEnabled(True)

            QMessageBox.critical(
                self,
                "Crop Failed",
                f"Failed to start video crop:\n{str(e)}",
            )
            logger.error(f"Video crop failed: {e}")
            import traceback

            traceback.print_exc()

    def _poll_crop_stderr_progress(self, process):
        """Read ffmpeg stderr in non-blocking mode and log progress."""
        if not process.stderr:
            return
        try:
            import fcntl
            import os as os_module

            fd = process.stderr.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os_module.O_NONBLOCK)
            try:
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if "frame=" in line:
                        try:
                            frame_str = line.split("frame=")[1].split()[0]
                            current_frame = int(frame_str)
                            total_frames = self._mw._crop_process.get("total_frames", 0)
                            if total_frames > 0:
                                progress_pct = int((current_frame / total_frames) * 100)
                                last_logged = self._mw._crop_process.get(
                                    "last_logged_progress", 0
                                )
                                if progress_pct >= last_logged + 10:
                                    logger.info(
                                        f"Video crop progress: {progress_pct}% ({current_frame}/{total_frames} frames)"
                                    )
                                    self._mw._crop_process["last_logged_progress"] = (
                                        progress_pct
                                    )
                        except (ValueError, IndexError):
                            pass
            except (IOError, OSError):
                pass
        except Exception:
            pass

    def _load_cropped_video(self, output_path):
        """Set up the UI to use the newly cropped video."""
        self._panels.setup.file_line.setText(output_path)
        self._mw.current_video_path = output_path
        self._mw.clear_roi()

        video_dir = os.path.dirname(output_path)
        video_name = os.path.splitext(os.path.basename(output_path))[0]

        csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
        self._panels.setup.csv_line.setText(csv_path)

        video_out_path = os.path.join(video_dir, f"{video_name}_tracking.mp4")
        self._panels.postprocess.video_out_line.setText(video_out_path)
        self._panels.postprocess.check_video_output.setChecked(True)

        self._mw.btn_test_detection.setEnabled(True)
        self._panels.setup.btn_detect_fps.setEnabled(True)
        self._mw.btn_crop_video.setEnabled(False)
        if hasattr(self._mw, "roi_optimization_label"):
            self._mw.roi_optimization_label.setText("")

        config_path = _get_video_config_path(output_path)
        if config_path and os.path.isfile(config_path):
            self._load_config_from_file(config_path)
            self._panels.setup.config_status_label.setText(
                f"\u2713 Loaded: {os.path.basename(config_path)}"
            )
            self._panels.setup.config_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Cropped video loaded: {output_path} (auto-loaded config)")
        else:
            self._panels.setup.config_status_label.setText(
                "No config found (using current settings)"
            )
            self._panels.setup.config_status_label.setStyleSheet(
                "color: #f39c12; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Cropped video loaded: {output_path} (no config found)")

    def _handle_crop_success(self, output_path, orig_w, orig_h, crop_w, crop_h):
        """Handle a successful crop completion."""
        reply = QMessageBox.question(
            self._mw,
            "Crop Complete",
            f"Video successfully cropped to ROI!\n\n"
            f"Original: {orig_w}x{orig_h}\n"
            f"Cropped: {crop_w}x{crop_h}\n"
            f"Saved to: {os.path.basename(output_path)}\n\n"
            f"Would you like to load the cropped video now?\n"
            f"(Note: ROI will be cleared since the video is already cropped)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self._load_cropped_video(output_path)

        self._mw._set_ui_controls_enabled(True)
        if hasattr(self._mw, "btn_crop_video"):
            self._mw.btn_crop_video.setText("Crop Video to ROI")
        logger.info(f"Successfully cropped video to {output_path}")

    def _handle_crop_failure(self, return_code):
        """Handle a failed crop completion."""
        self._mw._set_ui_controls_enabled(True)
        if hasattr(self._mw, "btn_crop_video"):
            self._mw.btn_crop_video.setText("Crop Video to ROI")
            self._mw.btn_crop_video.setEnabled(True)
        logger.error(f"Video crop failed with return code {return_code}")
        QMessageBox.critical(
            self._mw,
            "Crop Failed",
            f"Video cropping failed (return code: {return_code})\n\n"
            f"Check that ffmpeg is installed and the video is valid.",
        )

    def _check_crop_completion(self):
        """Check if background crop process has completed."""
        if not hasattr(self._mw, "_crop_process"):
            if hasattr(self._mw, "_crop_check_timer"):
                self._mw._crop_check_timer.stop()
            return

        process = self._mw._crop_process["process"]
        self._poll_crop_stderr_progress(process)

        return_code = process.poll()
        if return_code is not None:
            self._mw._crop_check_timer.stop()
            output_path = self._mw._crop_process["output_path"]
            orig_w, orig_h = self._mw._crop_process["original_size"]
            crop_w, crop_h = self._mw._crop_process["cropped_size"]

            if return_code == 0 and os.path.exists(output_path):
                self._handle_crop_success(output_path, orig_w, orig_h, crop_w, crop_h)
            else:
                self._handle_crop_failure(return_code)

            del self._mw._crop_process

    # =========================================================================
    # OPTIMIZER / PARAMETER HELPER
    # =========================================================================

    def _find_or_plan_optimizer_cache_path(
        self, video_path: str, params: dict, start_frame: int, end_frame: int
    ) -> tuple:
        """
        Return (cache_path, already_valid) where:
          - cache_path  : best path to use (existing covering cache, or new build target)
          - already_valid: True when that path exists AND covers start_frame..end_frame

        Search order:
          1. current_detection_cache_path (set by the last tracking run in this session)
             — only if it was produced by the *same* detection method.
          2. Any compatible detection cache in the video's dedicated cache directory
             whose filename matches the current detection method, with legacy flat
             files still considered as a fallback.
          3. A freshly-computed optimizer-specific path in the dedicated cache directory
             (name encodes the detection method so different methods never collide).
        """
        import re

        from hydra_suite.data.detection_cache import DetectionCache
        from hydra_suite.utils.video_artifacts import (
            build_optimizer_detection_cache_path,
            candidate_artifact_base_dirs,
            choose_writable_artifact_base_dir,
            iter_detection_cache_candidates,
        )

        detection_method = params.get("DETECTION_METHOD", "background_subtraction")
        method_tag = "yolo" if detection_method == "yolo_obb" else "bgsub"

        def _cache_matches_method(path_str: str) -> bool:
            """Return True when *path_str*'s filename was produced by *method_tag*."""
            if not path_str:
                return False
            basename = os.path.basename(path_str)
            # Production tracking caches: ..._detection_cache_{method_tag}_...
            if "_detection_cache_" in basename:
                after = basename.split("_detection_cache_", 1)[1]
                return after.startswith(f"{method_tag}_") or after.startswith(
                    f"{method_tag}."
                )
            # Optimizer caches (new naming): ..._opt_cache.npz with method_tag
            if "_opt_cache" in basename:
                before_opt = basename.split("_opt_cache")[0]
                # Accept if the method tag appears as a delimited segment
                segments = before_opt.split("_")
                return method_tag in segments
            return False

        def _is_valid(path: str) -> bool:
            """Return True if *path* is a compatible cache covering the frame range."""
            if not path or not os.path.exists(path):
                return False
            try:
                dc = DetectionCache(path, mode="r")
                ok = dc.is_compatible() and dc.covers_frame_range(
                    start_frame, end_frame
                )
                dc.close()
                return ok
            except Exception:
                return False

        # 1. Production cache from current session — only if method-compatible
        if _cache_matches_method(self._mw.current_detection_cache_path) and _is_valid(
            self._mw.current_detection_cache_path
        ):
            return self._mw.current_detection_cache_path, True

        csv_dir = (
            os.path.dirname(self._panels.setup.csv_line.text())
            if self._panels.setup.csv_line.text()
            else ""
        )
        artifact_base_dirs = candidate_artifact_base_dirs(
            video_path,
            preferred_base_dirs=[csv_dir],
        )

        # 2. Scan known cache directories — accept only method-compatible caches.
        for candidate in iter_detection_cache_candidates(
            video_path,
            artifact_base_dirs=artifact_base_dirs,
        ):
            candidate_str = str(candidate)
            if _cache_matches_method(candidate_str) and _is_valid(candidate_str):
                return candidate_str, True

        # 3. Fallback: compute a write-target path for a new detection-only build.
        #    Include the detection method so different methods never share a cache.
        if detection_method == "yolo_obb":
            model_raw = os.path.splitext(
                os.path.basename(params.get("YOLO_MODEL_PATH", "model"))
            )[0]
            model = re.sub(r"[^A-Za-z0-9_-]", "_", model_raw)
            opt_model_name = f"yolo_{model}"
        else:
            opt_model_name = "bgsub"
        resize = int(params.get("RESIZE_FACTOR", 1.0) * 100)
        artifact_base_dir = choose_writable_artifact_base_dir(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        new_path = build_optimizer_detection_cache_path(
            video_path,
            opt_model_name,
            resize,
            artifact_base_dir=artifact_base_dir,
        )
        return str(new_path), False

    def _build_optimizer_detection_cache(
        self, video_path: str, cache_path: str, params: dict
    ):
        """Spin up a DetectionCacheBuilderWorker and show progress in the main window."""
        from hydra_suite.core.tracking.optimizer_workers import (
            DetectionCacheBuilderWorker,
        )

        self._mw._cache_builder_worker = DetectionCacheBuilderWorker(
            video_path,
            cache_path,
            params,
            self._panels.setup.spin_start_frame.value(),
            self._panels.setup.spin_end_frame.value(),
        )
        self._mw._cache_builder_worker.progress_signal.connect(self.on_progress_update)
        self._mw._cache_builder_worker.finished_signal.connect(
            self._on_optimizer_cache_built
        )
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Building detection cache for optimizer...")
        self._mw._cache_builder_worker.start()

    def _apply_optimized_params(self, new_params):
        """Apply optimized parameter values from the helper dialog to UI widgets."""
        _direct_mappings = [
            ("YOLO_CONFIDENCE_THRESHOLD", self._panels.detection.spin_yolo_confidence),
            ("YOLO_IOU_THRESHOLD", self._panels.detection.spin_yolo_iou),
            ("MAX_DISTANCE_MULTIPLIER", self._panels.tracking.spin_max_dist),
            ("KALMAN_NOISE_COVARIANCE", self._panels.tracking.spin_kalman_noise),
            (
                "KALMAN_MEASUREMENT_NOISE_COVARIANCE",
                self._panels.tracking.spin_kalman_meas,
            ),
            ("W_POSITION", self._panels.tracking.spin_Wp),
            ("W_ORIENTATION", self._panels.tracking.spin_Wo),
            ("W_AREA", self._panels.tracking.spin_Wa),
            ("W_ASPECT", self._panels.tracking.spin_Wasp),
            ("KALMAN_DAMPING", self._panels.tracking.spin_kalman_damping),
            (
                "KALMAN_MAX_VELOCITY_MULTIPLIER",
                self._panels.tracking.spin_kalman_max_velocity,
            ),
            (
                "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER",
                self._panels.tracking.spin_kalman_longitudinal_noise,
            ),
        ]
        for key, widget in _direct_mappings:
            if key in new_params:
                widget.setValue(new_params[key])

        # Frame-count-to-seconds conversions
        _opt_fps = self._panels.setup.spin_fps.value()
        if "KALMAN_MATURITY_AGE" in new_params:
            self._panels.tracking.spin_kalman_maturity_age.setValue(
                new_params["KALMAN_MATURITY_AGE"] / _opt_fps
            )
        if "LOST_THRESHOLD_FRAMES" in new_params:
            self._panels.tracking.spin_lost_thresh.setValue(
                new_params["LOST_THRESHOLD_FRAMES"] / _opt_fps
            )

    def _open_parameter_helper(self):
        """Open the tracking parameter selection helper dialog."""
        from hydra_suite.trackerkit.gui.dialogs.parameter_helper import (
            ParameterHelperDialog,
        )

        video_path = self._panels.setup.file_line.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self._mw, "No Video", "Please load a video first.")
            return

        start_frame = self._panels.setup.spin_start_frame.value()
        end_frame = self._panels.setup.spin_end_frame.value()

        if (end_frame - start_frame) > 1000:
            QMessageBox.warning(
                self._mw,
                "Range Too Large",
                "The selected range is very large. For faster optimization, "
                "please select a smaller slice (e.g., 100-500 frames) using "
                "the 'Start Frame' and 'End Frame' boxes.",
            )
            return

        params = self.get_parameters_dict()

        cache_path, already_valid = self._find_or_plan_optimizer_cache_path(
            video_path, params, start_frame, end_frame
        )

        if not already_valid:
            res = QMessageBox.question(
                self._mw,
                "Detection Required",
                "No detection cache covering frames "
                f"{start_frame}\u2013{end_frame} was found.\n\n"
                "Run a quick detection-only scan now?\n"
                "(No config save, no pose inference, no CSV output \u2014 "
                "detections only.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                self._build_optimizer_detection_cache(video_path, cache_path, params)
            return  # dialog opens via _on_optimizer_cache_built when ready

        dialog = ParameterHelperDialog(
            video_path, cache_path, start_frame, end_frame, params, self._mw
        )

        if dialog.exec() == QDialog.Accepted:
            new_params = dialog.get_selected_params()
            if new_params:
                self._apply_optimized_params(new_params)
                QMessageBox.information(
                    self._mw,
                    "Parameters Applied",
                    "The optimized parameters have been applied to the UI.",
                )

    # ── BG-subtraction auto-tuner ─────────────────────────────────────────

    def _open_bg_parameter_helper(self):
        """Open the BG-subtraction parameter auto-tuner dialog."""
        from hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper import (
            BgParameterHelperDialog,
        )

        video_path = self._panels.setup.file_line.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self._mw, "No Video", "Please load a video first.")
            return

        params = self.get_parameters_dict()

        dialog = BgParameterHelperDialog(video_path, params, self._mw)
        if dialog.exec() == QDialog.Accepted:
            new_p = dialog.get_selected_params()
            if not new_p:
                return
            import math

            fps = max(float(self._panels.setup.spin_fps.value()), 1e-6)
            reference_body_size = float(
                self._panels.detection.spin_reference_body_size.value()
            )
            resize_factor = float(self._panels.setup.spin_resize.value())
            scaled_body_area = (
                math.pi * (reference_body_size / 2.0) ** 2 * (resize_factor**2)
            )

            # Apply optimised values back to the UI widgets
            if "BRIGHTNESS" in new_p:
                self._panels.detection.slider_brightness.setValue(
                    int(new_p["BRIGHTNESS"])
                )
            if "CONTRAST" in new_p:
                self._panels.detection.slider_contrast.setValue(
                    int(round(float(new_p["CONTRAST"]) * 100.0))
                )
            if "GAMMA" in new_p:
                self._panels.detection.slider_gamma.setValue(
                    int(round(float(new_p["GAMMA"]) * 100.0))
                )
            if "DARK_ON_LIGHT_BACKGROUND" in new_p:
                self._panels.detection.chk_dark_on_light.setChecked(
                    bool(new_p["DARK_ON_LIGHT_BACKGROUND"])
                )
            if "BACKGROUND_PRIME_FRAMES" in new_p:
                self._panels.detection.spin_bg_prime.setValue(
                    float(new_p["BACKGROUND_PRIME_FRAMES"]) / fps
                )
            if "ENABLE_ADAPTIVE_BACKGROUND" in new_p:
                self._panels.detection.chk_adaptive_bg.setChecked(
                    bool(new_p["ENABLE_ADAPTIVE_BACKGROUND"])
                )
            if "BACKGROUND_LEARNING_RATE" in new_p:
                self._panels.detection.spin_bg_learning.setValue(
                    float(new_p["BACKGROUND_LEARNING_RATE"])
                )
            if "ENABLE_LIGHTING_STABILIZATION" in new_p:
                self._panels.detection.chk_lighting_stab.setChecked(
                    bool(new_p["ENABLE_LIGHTING_STABILIZATION"])
                )
            if "LIGHTING_SMOOTH_FACTOR" in new_p:
                self._panels.detection.spin_lighting_smooth.setValue(
                    float(new_p["LIGHTING_SMOOTH_FACTOR"])
                )
            if "LIGHTING_MEDIAN_WINDOW" in new_p:
                self._panels.detection.spin_lighting_median.setValue(
                    int(new_p["LIGHTING_MEDIAN_WINDOW"])
                )
            if "THRESHOLD_VALUE" in new_p:
                self._panels.detection.spin_threshold.setValue(new_p["THRESHOLD_VALUE"])
            if "MORPH_KERNEL_SIZE" in new_p:
                self._panels.detection.spin_morph_size.setValue(
                    new_p["MORPH_KERNEL_SIZE"]
                )
            if "MIN_CONTOUR_AREA" in new_p:
                self._panels.detection.spin_min_contour.setValue(
                    new_p["MIN_CONTOUR_AREA"]
                )
            if "MAX_CONTOUR_MULTIPLIER" in new_p:
                self._panels.detection.spin_max_contour_multiplier.setValue(
                    int(new_p["MAX_CONTOUR_MULTIPLIER"])
                )
            if "ENABLE_SIZE_FILTERING" in new_p:
                self._panels.detection.chk_size_filtering.setChecked(
                    bool(new_p["ENABLE_SIZE_FILTERING"])
                )
            if "MIN_OBJECT_SIZE" in new_p and scaled_body_area > 0:
                self._panels.detection.spin_min_object_size.setValue(
                    float(new_p["MIN_OBJECT_SIZE"]) / scaled_body_area
                )
            if "MAX_OBJECT_SIZE" in new_p and scaled_body_area > 0:
                self._panels.detection.spin_max_object_size.setValue(
                    float(new_p["MAX_OBJECT_SIZE"]) / scaled_body_area
                )
            if "ENABLE_ADDITIONAL_DILATION" in new_p:
                self._panels.detection.chk_additional_dilation.setChecked(
                    new_p["ENABLE_ADDITIONAL_DILATION"]
                )
            if "DILATION_KERNEL_SIZE" in new_p:
                self._panels.detection.spin_dilation_kernel_size.setValue(
                    new_p["DILATION_KERNEL_SIZE"]
                )
            if "DILATION_ITERATIONS" in new_p:
                self._panels.detection.spin_dilation_iterations.setValue(
                    new_p["DILATION_ITERATIONS"]
                )
            if "ENABLE_CONSERVATIVE_SPLIT" in new_p:
                self._panels.detection.chk_conservative_split.setChecked(
                    new_p["ENABLE_CONSERVATIVE_SPLIT"]
                )
            if "CONSERVATIVE_KERNEL_SIZE" in new_p:
                self._panels.detection.spin_conservative_kernel.setValue(
                    new_p["CONSERVATIVE_KERNEL_SIZE"]
                )
            if "CONSERVATIVE_ERODE_ITER" in new_p:
                self._panels.detection.spin_conservative_erode.setValue(
                    new_p["CONSERVATIVE_ERODE_ITER"]
                )
            QMessageBox.information(
                self,
                "Parameters Applied",
                "Detection parameters have been applied to the UI.\n"
                "Use 'Preview Detection' to verify the results.",
            )

    def _open_benchmark_dialog(self):
        """Open the TrackerKit benchmark dialog for current model selections."""
        from hydra_suite.trackerkit.gui.dialogs.benchmark_dialog import (
            TrackerBenchmarkDialog,
        )

        video_path = self._panels.setup.file_line.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self._mw, "No Video", "Please load a video first.")
            return

        targets, notes = collect_active_targets(self._mw)
        if not targets:
            message = "No benchmarkable model targets are currently enabled."
            if notes:
                message += "\n\n" + "\n".join(notes)
            QMessageBox.information(self._mw, "No Targets", message)
            return

        dialog = TrackerBenchmarkDialog(self._mw, self._mw)
        if dialog.exec() == QDialog.Accepted:
            notes = self._apply_benchmark_recommendations(dialog.recommendations())
            self._refresh_benchmark_recommendations()
            self._mw._on_runtime_context_changed()
            message = "Benchmark recommendations applied to the current UI."
            if notes:
                message += "\n\n" + "\n".join(f"• {note}" for note in notes)
            QMessageBox.information(self._mw, "Recommendations Applied", message)

    def _refresh_benchmark_recommendations(self) -> None:
        """Refresh cached benchmark recommendations for the current UI state."""
        video_path = self._panels.setup.file_line.text().strip()
        if not video_path or not os.path.exists(video_path):
            self._mw._benchmark_recommendations = {}
            return
        try:
            geometry = derive_benchmark_geometry_from_video(
                video_path,
                resize_factor=float(self._panels.setup.spin_resize.value()),
                reference_body_size=float(
                    self._panels.detection.spin_reference_body_size.value()
                ),
                reference_aspect_ratio=float(
                    self._panels.detection.spin_reference_aspect_ratio.value()
                ),
                padding_fraction=float(
                    self._panels.identity.spin_individual_padding.value()
                ),
            )
        except Exception:
            logger.debug(
                "Could not derive benchmark geometry for the current video.",
                exc_info=True,
            )
            self._mw._benchmark_recommendations = {}
            return

        recommendations = {}
        targets, _notes = collect_active_targets(self._mw)
        for target in targets:
            recommendation = lookup_cached_recommendation(
                target,
                geometry,
                realtime_enabled=self._mw._is_realtime_tracking_mode_enabled(),
            )
            if recommendation is not None:
                recommendations[target.key] = recommendation
        self._mw._benchmark_recommendations = recommendations

    def _apply_benchmark_recommendations(
        self, recommendations: dict[str, object]
    ) -> list[str]:
        """Apply cached benchmark recommendations back into the current UI."""
        notes: list[str] = []

        def _set_combo_data(combo: QComboBox, value: object) -> bool:
            index = combo.findData(value)
            if index < 0:
                return False
            combo.setCurrentIndex(index)
            return True

        detection = recommendations.get(
            "detection_direct"
            if self._panels.detection.combo_yolo_obb_mode.currentIndex() == 0
            else "detection_sequential"
        )
        if detection is not None:
            _set_combo_data(self._panels.setup.combo_compute_runtime, detection.runtime)
            self._panels.detection.chk_enable_yolo_batching.setChecked(True)
            self._panels.detection.combo_yolo_batch_mode.setCurrentIndex(1)
            self._panels.detection.spin_yolo_batch_size.setValue(
                int(detection.batch_size)
            )
            individual_batch_size = getattr(detection, "individual_batch_size", None)
            if individual_batch_size is not None and hasattr(
                self._panels.detection, "spin_yolo_seq_individual_batch_size"
            ):
                self._panels.detection.spin_yolo_seq_individual_batch_size.setValue(
                    int(individual_batch_size)
                )

        headtail = recommendations.get("headtail")
        if headtail is not None:
            _set_combo_data(self._panels.setup.combo_headtail_runtime, headtail.runtime)
            if hasattr(self._panels, "identity") and hasattr(
                self._panels.identity, "combo_headtail_runtime"
            ):
                _set_combo_data(
                    self._panels.identity.combo_headtail_runtime, headtail.runtime
                )
            if hasattr(self._panels.identity, "spin_headtail_batch"):
                self._panels.identity.spin_headtail_batch.setValue(
                    int(headtail.batch_size)
                )

        pose = recommendations.get(f"pose_{self._mw._current_pose_backend_key()}")
        if pose is not None:
            pose_flavor = derive_pose_runtime_settings(
                pose.runtime,
                backend_family=self._mw._current_pose_backend_key(),
            ).get("pose_runtime_flavor", "cpu")
            _set_combo_data(self._panels.setup.combo_pose_runtime_flavor, pose_flavor)
            self._panels.identity.spin_pose_batch.setValue(int(pose.batch_size))

        cnn_recommendations = [
            recommendation
            for key, recommendation in recommendations.items()
            if str(key).startswith("cnn_")
        ]
        if cnn_recommendations:
            shared_runtimes = {
                recommendation.runtime for recommendation in cnn_recommendations
            }
            if len(shared_runtimes) == 1:
                _set_combo_data(
                    self._panels.setup.combo_cnn_runtime,
                    cnn_recommendations[0].runtime,
                )
            else:
                notes.append(
                    "CNN runtime recommendations differed across classifiers, so the shared CNN runtime selector was left unchanged."
                )
            rows = self._panels.identity._cnn_classifier_rows()
            for index, row in enumerate(rows):
                recommendation = recommendations.get(f"cnn_{index}")
                if recommendation is not None:
                    row.spin_batch.setValue(int(recommendation.batch_size))

        self._panels.detection._sync_batch_policy_controls()
        self._panels.identity._sync_realtime_individual_batch_ui()
        return notes

    # =========================================================================
    # COMPUTE RUNTIME (DELEGATE)
    # =========================================================================

    def _populate_compute_runtime_options(self, preferred=None):
        """Populate the compute runtime combo box with valid options for the current UI state."""
        if not hasattr(self._mw, "_setup_panel"):
            return
        combo = self._mw._setup_panel.combo_compute_runtime
        selected = (
            str(preferred or self._mw._selected_compute_runtime() or "cpu")
            .strip()
            .lower()
        )
        options = self._mw._compute_runtime_options_for_current_ui()
        values = [value for _label, value in options]
        if selected not in values:
            selected = values[0] if values else "cpu"
        combo.blockSignals(True)
        combo.clear()
        for label, value in options:
            combo.addItem(label, value)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================

    def _format_yolo_model_label(self, model_path):
        """Build combo-box label for a model path, including metadata if available."""
        rel_path = make_model_path_relative(model_path)
        filename = os.path.basename(rel_path)
        metadata = get_yolo_model_metadata(rel_path) or {}

        size = metadata.get("size") or metadata.get("model_size")
        species = metadata.get("species")
        model_info = metadata.get("model_info")
        task_family = str(metadata.get("task_family", "")).strip().lower()
        usage_role = str(metadata.get("usage_role", "")).strip().lower()
        model_id = None
        if species and model_info:
            model_id = f"{species}_{model_info}"
        elif species:
            model_id = species
        suffix_parts = []
        if size:
            suffix_parts.append(str(size))
        if model_id:
            suffix_parts.append(str(model_id))
        if usage_role:
            suffix_parts.append(usage_role)
        elif task_family:
            suffix_parts.append(task_family)
        if suffix_parts:
            return f"{filename} ({', '.join(suffix_parts)})"
        return filename

    @staticmethod
    def _yolo_model_matches_filter(metadata, task_family=None, usage_role=None):
        if not isinstance(metadata, dict):
            return True
        meta_task = str(metadata.get("task_family", "")).strip().lower()
        meta_role = _normalize_usage_role(metadata.get("usage_role", ""))
        usage_role = _normalize_usage_role(usage_role)
        if not meta_task and not meta_role:
            return True
        if task_family and meta_task and meta_task != task_family:
            return False
        if usage_role and meta_role and meta_role != usage_role:
            return False
        return True

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
        selected_path = preferred_model_path
        if selected_path is None:
            selected_data = combo.currentData(Qt.UserRole)
            if selected_data and selected_data not in ("__add_new__", "__none__"):
                selected_path = str(selected_data)

        entries = {}
        models_dir = str(
            repository_dir
            or get_yolo_model_repository_directory(
                task_family=task_family, usage_role=usage_role
            )
        )
        try:
            if recursive:
                local_model_paths = []
                for dirpath, _dirnames, filenames in os.walk(models_dir):
                    for fn in sorted(filenames):
                        if os.path.splitext(fn)[1].lower() in (".pt", ".pth"):
                            local_model_paths.append(os.path.join(dirpath, fn))
            else:
                local_model_paths = sorted(
                    os.path.join(models_dir, f)
                    for f in os.listdir(models_dir)
                    if os.path.splitext(f)[1].lower() in (".pt", ".pth")
                )
        except Exception as e:
            logger.warning(f"Failed to list YOLO model directory '{models_dir}': {e}")
            local_model_paths = []

        for model_abs in local_model_paths:
            rel_path = make_model_path_relative(model_abs)
            metadata = get_yolo_model_metadata(rel_path) or {}
            if not self._yolo_model_matches_filter(
                metadata, task_family=task_family, usage_role=usage_role
            ):
                continue
            entries[rel_path] = self._format_yolo_model_label(rel_path)

        combo.blockSignals(True)
        combo.clear()
        for model_path, label in entries.items():
            combo.addItem(label, model_path)
        if include_none:
            combo.insertItem(0, "— None —", "__none__")
        combo.addItem("＋ Add New Model…", "__add_new__")
        combo.blockSignals(False)

        self._set_model_selection_for_selector(
            combo, selected_path, default_path=default_path
        )

    def _set_model_selection_for_selector(self, combo, model_path, default_path=""):
        target_path = make_model_path_relative(model_path or "")
        if not target_path:
            target_path = str(default_path or "")
        for i in range(combo.count()):
            item_data = combo.itemData(i, Qt.UserRole)
            if item_data == target_path:
                combo.setCurrentIndex(i)
                return
        none_idx = combo.findData("__none__", Qt.UserRole)
        if none_idx >= 0:
            combo.setCurrentIndex(none_idx)
        else:
            combo.setCurrentIndex(0)

    def _get_selected_model_path_from_selector(self, combo, default_path=""):
        selected_data = combo.currentData(Qt.UserRole)
        if selected_data and selected_data not in ("__add_new__", "__none__"):
            return str(selected_data)
        return str(default_path or "")

    @staticmethod
    def _selector_has_removable_model(combo) -> bool:
        """Return True when the selector currently points at a stored model."""
        selected_data = combo.currentData(Qt.UserRole)
        return bool(selected_data and selected_data not in ("__add_new__", "__none__"))

    def _confirm_and_remove_repository_model(
        self, model_path: object, *, model_kind: str = "model"
    ) -> bool:
        """Confirm and remove a specific stored model from the local repository."""
        stored_path = str(model_path or "").strip()
        if not stored_path:
            return False

        display_name = os.path.basename(stored_path.rstrip("/\\")) or stored_path
        reply = QMessageBox.question(
            self._mw,
            f"Remove {model_kind}",
            f"Remove this {model_kind} from the local repository?\n\n"
            f"{display_name}\n\n"
            "This deletes the stored model and removes any registry entry. "
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False

        try:
            removed = remove_model_from_repository(stored_path)
        except Exception as exc:
            logger.error("Failed to remove %s '%s': %s", model_kind, stored_path, exc)
            QMessageBox.warning(
                self._mw,
                f"Remove {model_kind}",
                f"Could not remove the selected {model_kind}:\n{exc}",
            )
            return False

        if not removed:
            QMessageBox.warning(
                self._mw,
                f"Remove {model_kind}",
                f"The selected {model_kind} could not be removed.",
            )
            return False

        logger.info("Removed %s from repository: %s", model_kind, stored_path)
        return True

    def _import_yolo_model_to_repository(
        self,
        source_path,
        task_family=None,
        usage_role=None,
        repository_dir=None,
    ):
        """Import a YOLO model file into the repository with metadata."""
        src = str(source_path or "")
        if not src or not os.path.exists(src):
            return None

        src_abs = os.path.abspath(src)
        models_dir = str(
            repository_dir
            or get_yolo_model_repository_directory(
                task_family=task_family, usage_role=usage_role
            )
        )
        try:
            rel_existing = os.path.relpath(src_abs, models_dir)
            if not rel_existing.startswith(".."):
                return make_model_path_relative(src_abs)
        except Exception:
            pass

        now_preview = datetime.now()
        dlg = QDialog(self._mw)
        dlg.setWindowTitle("Model Metadata")
        dlg_layout = QVBoxLayout(dlg)
        dlg_form = QFormLayout()

        size_combo = QComboBox(dlg)
        size_combo.addItems(["26n", "26s", "26m", "26l", "26x", "custom", "unknown"])
        size_combo.setCurrentText("26s")
        dlg_form.addRow("YOLO model size:", size_combo)

        stem_tokens = [t for t in Path(src).stem.replace("-", "_").split("_") if t]
        default_species = (
            _sanitize_model_token(stem_tokens[0]) if stem_tokens else "species"
        )
        default_info = (
            _sanitize_model_token("_".join(stem_tokens[1:]))
            if len(stem_tokens) > 1
            else "model"
        )

        species_line = QLineEdit(default_species, dlg)
        species_line.setPlaceholderText("species")
        dlg_form.addRow("Model species:", species_line)

        info_line = QLineEdit(default_info, dlg)
        info_line.setPlaceholderText("model-info")
        dlg_form.addRow("Model info:", info_line)

        ts_label = QLabel(now_preview.isoformat(timespec="seconds"), dlg)
        ts_label.setToolTip("Timestamp applied when model is added to repository")
        dlg_form.addRow("Added timestamp:", ts_label)

        dlg_layout.addLayout(dlg_form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        model_size = size_combo.currentText().strip() or "unknown"
        model_species = _sanitize_model_token(species_line.text())
        model_info = _sanitize_model_token(info_line.text())
        if not model_species or not model_info:
            QMessageBox.warning(
                self._mw,
                "Invalid Metadata",
                "Species and model info must both be provided.",
            )
            return None

        now = datetime.now()
        timestamp_token = now.strftime("%Y%m%d-%H%M%S")
        added_at = now.isoformat(timespec="seconds")
        ext = os.path.splitext(src)[1].lower() or ".pt"

        model_slug = f"{model_species}_{model_info}"
        base_name = f"{timestamp_token}_{model_size}_{model_slug}"
        dest_path = os.path.join(models_dir, f"{base_name}{ext}")
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(models_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        try:
            shutil.copy2(src, dest_path)
        except Exception as e:
            logger.error(f"Failed to copy model to repository: {e}")
            QMessageBox.warning(
                self._mw,
                "Import Failed",
                f"Could not import model into repository:\n{e}",
            )
            return None

        rel_path = make_model_path_relative(dest_path)
        metadata = {
            "size": model_size,
            "species": model_species,
            "model_info": model_info,
            "added_at": added_at,
            "source_path": src,
            "stored_filename": os.path.basename(dest_path),
        }
        if task_family:
            metadata["task_family"] = str(task_family).strip().lower()
        if usage_role:
            metadata["usage_role"] = str(usage_role).strip().lower()
        register_yolo_model(rel_path, metadata)
        logger.info(f"Imported model to repository: {dest_path}")
        return rel_path

    def _populate_pose_model_combo(self, combo, backend, preferred_model_path=None):
        """Populate the pose model combo for the given backend."""
        selected_path = preferred_model_path
        if selected_path is None:
            selected_data = combo.currentData(Qt.UserRole)
            if selected_data and selected_data not in ("__add_new__", "__none__"):
                selected_path = str(selected_data)

        backend_key = (
            "sleap"
            if backend == "sleap"
            else ("vitpose" if backend == "vitpose" else "yolo")
        )
        repo_dir = get_pose_models_directory(backend_key)
        os.makedirs(repo_dir, exist_ok=True)

        entries = {}
        try:
            if backend_key == "sleap":
                for name in sorted(os.listdir(repo_dir)):
                    full = os.path.join(repo_dir, name)
                    if os.path.isdir(full):
                        rel = make_pose_model_path_relative(full)
                        entries[rel] = name
            else:
                for fn in sorted(os.listdir(repo_dir)):
                    if os.path.splitext(fn)[1].lower() in (".pt", ".pth"):
                        full = os.path.join(repo_dir, fn)
                        rel = make_pose_model_path_relative(full)
                        entries[rel] = self._format_yolo_model_label(rel)
        except Exception as e:
            logger.warning(f"Failed to list pose model directory '{repo_dir}': {e}")

        combo.blockSignals(True)
        combo.clear()
        for path, label in entries.items():
            combo.addItem(label, path)
        combo.insertItem(0, "— None —", "__none__")
        combo.addItem("＋ Add New Model…", "__add_new__")
        combo.blockSignals(False)

        self._set_model_selection_for_selector(combo, selected_path, default_path="")

    def _refresh_pose_model_combo(self, preferred_model_path=None):
        """Refresh the pose model combo for the current backend."""
        if not hasattr(self._mw, "_identity_panel"):
            return
        backend = self._mw._current_pose_backend_key()
        self._populate_pose_model_combo(
            self._mw._identity_panel.combo_pose_model,
            backend=backend,
            preferred_model_path=preferred_model_path,
        )
        self._mw._identity_panel._sync_pose_model_remove_button()

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
        prev_data = None
        for i in range(combo.count()):
            d = combo.itemData(i, Qt.UserRole)
            if d not in ("__add_new__", "__none__", None):
                prev_data = d
                break

        start_dir = str(
            repository_dir
            or get_yolo_model_repository_directory(
                task_family=task_family, usage_role=usage_role
            )
        )
        from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog

        fp, _ = QFileDialog.getOpenFileName(
            self._mw,
            dialog_title,
            start_dir,
            "PyTorch Model Files (*.pt *.pth);;All Files (*)",
        )
        if not fp:
            combo.blockSignals(True)
            self._set_model_selection_for_selector(combo, prev_data)
            combo.blockSignals(False)
            return

        selected_abs = os.path.abspath(fp)
        try:
            rel_existing = os.path.relpath(selected_abs, start_dir)
            is_in_repo = not rel_existing.startswith("..")
        except (ValueError, TypeError):
            is_in_repo = False

        if is_in_repo:
            final_path = make_model_path_relative(selected_abs)
        else:
            final_path = self._import_yolo_model_to_repository(
                selected_abs,
                task_family=task_family,
                usage_role=usage_role,
                repository_dir=start_dir,
            )
            if not final_path:
                combo.blockSignals(True)
                self._set_model_selection_for_selector(combo, prev_data)
                combo.blockSignals(False)
                return
            QMessageBox.information(
                self._mw,
                "Model Added",
                f"Model added to repository:\n{os.path.basename(final_path)}",
            )

        refresh_callback(preferred_model_path=final_path)
        selection_callback(final_path)

    def _handle_remove_selected_yolo_model(
        self,
        combo,
        refresh_callback,
        selection_callback,
        *,
        model_kind: str = "model",
    ) -> None:
        """Remove the currently selected repository-backed YOLO/classification model."""
        if not self._selector_has_removable_model(combo):
            return

        selected_path = str(combo.currentData(Qt.UserRole) or "").strip()
        if not self._confirm_and_remove_repository_model(
            selected_path,
            model_kind=model_kind,
        ):
            return

        refresh_callback(preferred_model_path="")
        selection_callback("")

    def _handle_add_new_pose_model(self):
        """Browse for a pose model, import it if outside repo, refresh combo, and select it."""
        from hydra_suite.trackerkit.gui.model_utils import (
            get_pose_models_directory,
            make_pose_model_path_relative,
        )
        from hydra_suite.utils.file_dialogs import HydraFileDialog as _QFileDialog

        combo = getattr(self._mw, "combo_pose_model", None)
        prev_data = None
        if combo is not None:
            for i in range(combo.count()):
                d = combo.itemData(i, Qt.UserRole)
                if d and d not in ("__add_new__", "__none__"):
                    prev_data = d
                    break

        def _restore():
            if combo is not None:
                combo.blockSignals(True)
                self._set_model_selection_for_selector(combo, prev_data)
                combo.blockSignals(False)

        from hydra_suite.trackerkit.gui.model_utils import resolve_pose_model_path

        backend = (
            self._mw._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        backend_key = (
            "sleap"
            if backend == "sleap"
            else ("vitpose" if backend == "vitpose" else "yolo")
        )
        current = self._mw._pose_model_path_for_backend(backend)
        if current:
            resolved_current = str(resolve_pose_model_path(current, backend=backend))
            from pathlib import Path as _Path

            start = (
                resolved_current
                if os.path.isdir(resolved_current)
                else (os.path.dirname(resolved_current) or str(_Path.home()))
            )
        else:
            start = get_pose_models_directory(backend_key)

        if backend == "sleap":
            selected = _QFileDialog.getExistingDirectory(
                self._mw, "Select SLEAP Model Directory", start
            )
            if not selected:
                _restore()
                return
            selected_abs = os.path.abspath(selected)
            pose_root = get_pose_models_directory(backend_key)
            try:
                rel_path = os.path.relpath(selected_abs, pose_root)
                is_in_repo = not rel_path.startswith("..")
            except (ValueError, TypeError):
                is_in_repo = False
            if is_in_repo:
                final_path = make_pose_model_path_relative(selected_abs)
            else:
                final_path = self._import_pose_model_to_repository(
                    selected_abs, backend=backend_key
                )
                if not final_path:
                    _restore()
                    return
                QMessageBox.information(
                    self._mw,
                    "Model Added",
                    f"SLEAP model added to repository:\n{final_path}",
                )
            self._mw._set_pose_model_path_for_backend(
                final_path, backend=backend, update_combo=True
            )
            return

        selected, _ = _QFileDialog.getOpenFileName(
            self._mw,
            "Select Pose Weights",
            start,
            "PyTorch Weights (*.pt *.pth);;All Files (*)",
        )
        if not selected:
            _restore()
            return
        selected_abs = os.path.abspath(selected)
        pose_root = get_pose_models_directory(backend_key)
        try:
            rel_path = os.path.relpath(selected_abs, pose_root)
            is_in_repo = not rel_path.startswith("..")
        except (ValueError, TypeError):
            is_in_repo = False
        if is_in_repo:
            final_path = make_pose_model_path_relative(selected_abs)
        else:
            final_path = self._import_pose_model_to_repository(
                selected_abs, backend=backend_key
            )
            if not final_path:
                _restore()
                return
            QMessageBox.information(
                self._mw,
                "Model Added",
                f"Pose model added to repository:\n{final_path}",
            )
        self._mw._set_pose_model_path_for_backend(
            final_path, backend=backend, update_combo=True
        )

    def _handle_remove_selected_pose_model(self) -> None:
        """Remove the currently selected pose model or SLEAP model directory."""
        if not hasattr(self._mw, "_identity_panel"):
            return

        combo = self._mw._identity_panel.combo_pose_model
        if not self._selector_has_removable_model(combo):
            return

        backend = self._mw._current_pose_backend_key()
        model_kind = "SLEAP model" if backend == "sleap" else "pose model"
        selected_path = str(combo.currentData(Qt.UserRole) or "").strip()
        if not self._confirm_and_remove_repository_model(
            selected_path,
            model_kind=model_kind,
        ):
            return

        self._mw._set_pose_model_path_for_backend(
            "", backend=backend, update_combo=True
        )

    def _import_pose_model_to_repository(self, source_path, backend="yolo"):
        """Copy a selected pose model into models/pose/{YOLO|SLEAP|ViTPose} and return relative path."""
        import shutil as _shutil
        from pathlib import Path as _Path

        from hydra_suite.trackerkit.gui.model_utils import (
            get_pose_models_directory,
            make_pose_model_path_relative,
        )

        src = str(source_path or "").strip()
        if not src or not os.path.exists(src):
            return None

        bk = str(backend).strip().lower()
        backend_key = (
            "sleap" if bk == "sleap" else ("vitpose" if bk == "vitpose" else "yolo")
        )
        dest_dir = get_pose_models_directory(backend_key)

        try:
            src_path = _Path(src).expanduser().resolve()
        except Exception:
            src_path = _Path(src)

        try:
            rel_existing = os.path.relpath(
                str(src_path), str(_Path(dest_dir).resolve())
            )
            if not rel_existing.startswith(".."):
                return make_pose_model_path_relative(str(src_path))
        except Exception:
            pass

        now_preview = datetime.now()
        dlg = QDialog(self._mw)
        dlg.setWindowTitle("Pose Model Metadata")
        dlg_layout = QVBoxLayout(dlg)
        dlg_form = QFormLayout()

        stem_tokens = [t for t in src_path.stem.replace("-", "_").split("_") if t]
        default_species = (
            _sanitize_model_token(stem_tokens[0]) if stem_tokens else "species"
        )
        default_info = (
            _sanitize_model_token("_".join(stem_tokens[1:]))
            if len(stem_tokens) > 1
            else "model"
        )

        size_combo = None
        type_line = None
        if backend_key == "yolo":
            size_combo = QComboBox(dlg)
            size_combo.addItems(
                ["26n", "26s", "26m", "26l", "26x", "custom", "unknown"]
            )
            size_combo.setCurrentText("26s")
            dlg_form.addRow("YOLO model size:", size_combo)
        else:
            default_type = _sanitize_model_token(src_path.name) or "sleap_model"
            type_line = QLineEdit(default_type, dlg)
            type_line.setPlaceholderText("model-type")
            dlg_form.addRow("Model type:", type_line)

        species_line = QLineEdit(default_species, dlg)
        species_line.setPlaceholderText("species")
        dlg_form.addRow("Model species:", species_line)

        info_line = QLineEdit(default_info, dlg)
        info_line.setPlaceholderText("model-info")
        dlg_form.addRow("Model info:", info_line)

        ts_label = QLabel(now_preview.isoformat(timespec="seconds"), dlg)
        ts_label.setToolTip("Timestamp applied when model is added to repository")
        dlg_form.addRow("Added timestamp:", ts_label)

        dlg_layout.addLayout(dlg_form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        model_species = _sanitize_model_token(species_line.text())
        model_info = _sanitize_model_token(info_line.text())
        if not model_species or not model_info:
            QMessageBox.warning(
                self._mw,
                "Invalid Metadata",
                "Species and model info must both be provided.",
            )
            return None

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        if backend_key == "sleap":
            model_type = _sanitize_model_token(type_line.text()) if type_line else ""
            if not model_type:
                QMessageBox.warning(
                    self._mw,
                    "Invalid Metadata",
                    "SLEAP model type must be provided.",
                )
                return None
            target_name = f"{timestamp}_{model_type}_{model_species}_{model_info}"
            dest_path = _Path(dest_dir) / target_name
            counter = 1
            while dest_path.exists():
                dest_path = _Path(dest_dir) / f"{target_name}_{counter}"
                counter += 1
            try:
                _shutil.copytree(src_path, dest_path)
            except Exception as exc:
                logger.error("Failed to copy SLEAP model directory: %s", exc)
                QMessageBox.warning(
                    self._mw,
                    "Import Failed",
                    f"Could not import SLEAP model directory:\n{exc}",
                )
                return None
            return make_pose_model_path_relative(str(dest_path))

        model_size = size_combo.currentText().strip() if size_combo else "unknown"
        model_size = _sanitize_model_token(model_size) or "unknown"
        ext = src_path.suffix or ".pt"
        target_name = f"{timestamp}_{model_size}_{model_species}_{model_info}{ext}"
        dest_path = _Path(dest_dir) / target_name
        counter = 1
        while dest_path.exists():
            dest_path = (
                _Path(dest_dir)
                / f"{timestamp}_{model_size}_{model_species}_{model_info}_{counter}{ext}"
            )
            counter += 1
        try:
            _shutil.copy2(src_path, dest_path)
        except Exception as exc:
            logger.error("Failed to copy pose model: %s", exc)
            QMessageBox.warning(
                self._mw,
                "Import Failed",
                f"Could not import pose model:\n{exc}",
            )
            return None
        return make_pose_model_path_relative(str(dest_path))
