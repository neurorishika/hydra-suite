"""SessionOrchestrator — logging, progress, UI state machine."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from hydra_suite.runtime.resolver import detect_platform
from hydra_suite.utils.geometry import fit_circle_to_points

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)

HEADTAIL_RUNTIME_TOOLTIP = (
    "Head-tail runtime for oriented crop classification.\n"
    "Visible only when head-tail analysis is enabled.\n"
    "Exported ONNX/TensorRT runtimes are shown when available.\n"
    "When a .pth classifier requests an exported accelerator runtime but the matching ONNX provider is unavailable, HYDRA falls back to the native device runtime for that platform."
)

CNN_RUNTIME_TOOLTIP = (
    "CNN identity runtime for per-animal classifiers.\n"
    "Visible only when at least one CNN classifier is configured."
)

POSE_RUNTIME_TOOLTIP = (
    "Pose runtime for the pose extraction pipeline.\n"
    "Visible only when pose extraction is enabled."
)


class SessionOrchestrator:
    """Manages session logging, progress display, and UI state transitions."""

    def __init__(
        self, main_window: "MainWindow", config: "TrackerConfig", panels
    ) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels
        # Active arena new ROI shapes join. One arena is often several shapes
        # (an include circle plus an exclude hole punched in it), so a shape
        # only starts a new arena when the user explicitly presses "New
        # Arena" (start_new_arena()) -- shape count is never arena count.
        self.current_arena_id = 0

    def start_new_arena(self) -> int:
        """Begin a new arena; subsequent shapes join it until called again.

        Delegates to ``engine_params.next_free_arena_id`` -- the single
        shared rule for "what id does the next arena get" (every shape
        carries an ``arena_id``, excludes included, so the next free id is
        computed over ALL shapes, not just includes). The bulk grid
        generator (``MainWindow._on_generate_grid_clicked``) uses the same
        helper so the two never disagree and a generated grid can't collide
        with a hand-drawn arena's id.
        """
        from hydra_suite.trackerkit.engine_params import next_free_arena_id

        self.current_arena_id = next_free_arena_id(self._mw.roi_shapes)
        return self.current_arena_id

    # =========================================================================
    # UI STATE MACHINE
    # =========================================================================

    def _set_ui_controls_enabled(self, enabled: bool):
        if enabled:
            if self._mw.current_video_path:
                self._apply_ui_state("idle")
            else:
                self._apply_ui_state("no_video")
            return

        # Disabled state - choose mode based on tracking/preview status
        if self._mw.tracking_worker and self._mw.tracking_worker.isRunning():
            if self._mw.btn_preview.isChecked():
                self._apply_ui_state("preview")
            else:
                self._apply_ui_state("tracking")
        else:
            self._apply_ui_state("locked")

    def _collect_preview_controls(self):
        return [
            self._mw.btn_test_detection,
            self._panels.setup.slider_timeline,
            self._panels.setup.btn_first_frame,
            self._panels.setup.btn_prev_frame,
            self._panels.setup.btn_play_pause,
            self._panels.setup.btn_next_frame,
            self._panels.setup.btn_last_frame,
            self._panels.setup.btn_random_seek,
            self._panels.setup.combo_playback_speed,
            self._panels.setup.spin_start_frame,
            self._panels.setup.spin_end_frame,
            self._panels.setup.btn_set_start_current,
            self._panels.setup.btn_set_end_current,
            self._panels.setup.btn_reset_range,
        ]

    def _set_interactive_widgets_enabled(
        self,
        enabled: bool,
        allowlist=None,
        blocklist=None,
        remember_state: bool = True,
    ):
        from PySide6.QtWidgets import (
            QAbstractButton,
            QComboBox,
            QDoubleSpinBox,
            QLineEdit,
            QSlider,
            QSpinBox,
        )

        allow = set(allowlist or [])
        block = set(blocklist or [])
        interactive_types = (
            QAbstractButton,
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QSlider,
        )
        # Exclude welcome-page widgets — they are managed by WelcomePage itself
        welcome = getattr(self._mw, "_welcome_page", None)
        widgets = []
        for widget_type in interactive_types:
            for w in self._mw.findChildren(widget_type):
                if welcome is not None and welcome.isAncestorOf(w):
                    continue
                widgets.append(w)

        if enabled and remember_state and self._mw._saved_widget_enabled_states:
            for widget in widgets:
                if widget in block:
                    widget.setEnabled(False)
                elif widget in allow:
                    widget.setEnabled(True)
                elif widget in self._mw._saved_widget_enabled_states:
                    widget.setEnabled(self._mw._saved_widget_enabled_states[widget])
            self._mw._saved_widget_enabled_states = {}
            return

        if not enabled and remember_state:
            for widget in widgets:
                if widget in block or widget in allow:
                    continue
                self._mw._saved_widget_enabled_states[widget] = widget.isEnabled()

        for widget in widgets:
            if widget in block:
                widget.setEnabled(False)
            elif widget in allow:
                widget.setEnabled(True)
            else:
                widget.setEnabled(enabled)

    def _set_video_interaction_enabled(self, enabled: bool):
        self._mw._video_interactions_enabled = enabled
        self._mw.slider_zoom.setEnabled(enabled)
        # Keep the viewport enabled so placeholder/logo rendering is not dimmed
        # by disabled-widget styling (notably on macOS).
        self._mw.scroll.setEnabled(True)
        if not enabled:
            self._mw.video_label.unsetCursor()

    def _sync_contextual_controls(self):
        # ArenaPanel owns its enabled/disabled state, but the interactive-widget
        # enable sweep in _apply_ui_state can still blanket re-enable its
        # buttons on some UI-state transitions (it walks every QAbstractButton).
        # Re-run refresh() here so the panel's own lock/disable rules always win.
        self._panels.arena.refresh()

    def _apply_ui_state(self, state: str):
        if state == "no_video":
            self._set_interactive_widgets_enabled(
                False,
                allowlist=[
                    self._panels.setup.btn_file,
                    self._panels.setup.btn_load_config,
                ],
                remember_state=False,
            )
            self._mw.btn_start.setEnabled(False)
            self._mw.btn_preview.setEnabled(False)
            if hasattr(self._mw, "_tracking_panel"):
                self._mw._tracking_panel.btn_param_helper.setEnabled(False)
            self._set_video_interaction_enabled(False)
            self._panels.setup.g_video_player.setVisible(False)
            self._show_video_logo_placeholder()
            return

        if state == "idle":
            self._set_interactive_widgets_enabled(True)
            self._mw.btn_start.setEnabled(True)
            self._mw.btn_preview.setEnabled(True)
            if hasattr(self._mw, "_tracking_panel"):
                self._mw._tracking_panel.btn_param_helper.setEnabled(True)
            self._set_video_interaction_enabled(True)
            self._sync_contextual_controls()
            return

        if state == "tracking":
            allow = [self._mw.btn_start]
            if self._mw._is_visualization_enabled():
                allow.append(self._mw.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self._mw.btn_start.setEnabled(True)
            self._set_video_interaction_enabled(self._mw._is_visualization_enabled())
            return

        if state == "preview":
            allow = [self._mw.btn_preview] + list(self._mw._preview_controls)
            if self._mw._is_visualization_enabled():
                allow.append(self._mw.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self._mw.btn_preview.setEnabled(True)
            self._set_video_interaction_enabled(self._mw._is_visualization_enabled())
            return

        # Locked (non-tracking) state: disable all interactive widgets
        if state == "locked":
            self._set_interactive_widgets_enabled(False)
            self._set_video_interaction_enabled(False)
            return

    def _prepare_tracking_display(self):
        """Clear any stale frame before tracking starts."""
        # Clear stale detection-test result so zoom events don't re-render old frames
        self._mw.detection_test_result = None
        self._mw._last_tracking_frame_rgb = None
        if self._mw._is_visualization_enabled():
            self._mw._set_video_message("")
        else:
            self._mw._set_video_message(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below.",
                color="#9a9a9a",
                font_size=14,
            )

    def _show_video_logo_placeholder(self):
        """Show HYDRA logo in the video panel when no video is loaded."""
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer

        try:
            from PySide6.QtCore import QByteArray

            from hydra_suite.paths import get_brand_icon_bytes

            logo_data = get_brand_icon_bytes("trackerkit.svg")
            vw = max(640, self._mw.scroll.viewport().width())
            vh = max(420, self._mw.scroll.viewport().height())
            canvas = QPixmap(vw, vh)
            canvas.fill(QColor(0, 0, 0, 0))

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

                # Preserve source aspect ratio and size it prominently.
                max_w = max(1, int(vw * 0.9))
                max_h = max(1, int(vh * 0.8))
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
                self._mw._set_video_pixmap(canvas, already_scaled=True)
                return
        except Exception:
            pass
            self._mw._set_video_message("HYDRA\n\nLoad a video to begin...")

    # =========================================================================
    # PROGRESS VISIBILITY
    # =========================================================================

    def _is_worker_running(self, worker):
        """Safely check whether a worker thread-like object is running."""
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except Exception:
            return False

    def _has_active_progress_task(self) -> bool:
        """Return True if any async task that owns progress UI is still active."""
        return any(
            [
                self._is_worker_running(self._mw.tracking_worker),
                self._is_worker_running(getattr(self._mw, "merge_worker", None)),
                self._is_worker_running(self._mw.dataset_worker),
                self._is_worker_running(self._mw.interp_worker),
                self._is_worker_running(self._mw.final_media_export_worker),
            ]
        )

    def _refresh_progress_visibility(self):
        """Keep progress UI visible while any async tracking task is still running."""
        has_active_task = self._has_active_progress_task()
        self._mw.progress_bar.setVisible(has_active_task)
        self._mw.progress_label.setVisible(has_active_task)

    # =========================================================================
    # SESSION LOGGING
    # =========================================================================

    def _setup_session_logging(self, video_path, backward_mode=False):
        """Set up comprehensive logging for the entire tracking session."""
        from datetime import datetime
        from pathlib import Path

        from hydra_suite.utils.video_artifacts import (
            build_tracking_session_log_path,
            choose_writable_artifact_base_dir,
        )

        # Close existing session log if any
        self._cleanup_session_logging()

        # Only set up logging if not already set up
        if self._mw.session_log_handler is not None:
            logger.info("=" * 80)
            logger.info("Session log already active, continuing...")
            logger.info("=" * 80)
            return

        # Create a session log in the video's dedicated log directory.
        video_path = Path(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = (
            os.path.dirname(self._panels.setup.csv_line.text())
            if self._panels.setup.csv_line.text()
            else ""
        )
        artifact_base_dir = choose_writable_artifact_base_dir(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        log_path = build_tracking_session_log_path(
            video_path,
            timestamp,
            artifact_base_dir=artifact_base_dir,
            create_dir=True,
        )

        # Create file handler for session
        self._mw.session_log_handler = logging.FileHandler(log_path, mode="w")
        self._mw.session_log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._mw.session_log_handler.setFormatter(formatter)

        # Add to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self._mw.session_log_handler)

        logger.info("=" * 80)
        logger.info("TRACKING SESSION STARTED")
        logger.info(f"Session log: {log_path}")
        logger.info(f"Video: {video_path}")
        logger.info("=" * 80)

    def _cleanup_session_logging(self):
        """Remove session log handler from root logger."""
        if self._mw.session_log_handler:
            logger.info("=" * 80)
            logger.info("Tracking session completed")
            logger.info("=" * 80)

            root_logger = logging.getLogger()
            root_logger.removeHandler(self._mw.session_log_handler)
            self._mw.session_log_handler.close()
            self._mw.session_log_handler = None

    # =========================================================================
    # TEMPORARY FILES
    # =========================================================================

    def _cleanup_temporary_files(self):
        """Remove temporary files if cleanup is enabled.

        User mode (debug off) always cleans up -- User-mode runs are only
        supposed to leave the clean tracks.csv + annotated video behind.
        Debug mode retains all intermediate files.
        (Defensive ``getattr``: kept in case this runs before ``self._mw.config``
        is fully populated; defaults to debug-on, i.e. retain files.)
        """
        _debug = bool(getattr(self._mw.config, "debug_mode", True))
        if _debug:
            logger.info("Debug Mode enabled, keeping intermediate files.")
            return

        if not self._mw.temporary_files:
            logger.info("No temporary files to clean up.")
            return

        cleaned = []
        failed = []
        for temp_file in self._mw.temporary_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    cleaned.append(os.path.basename(temp_file))
                    logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    failed.append(os.path.basename(temp_file))
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")

        # Clear the list after cleanup attempt
        self._mw.temporary_files.clear()

        # Also clean up posekit directories if they exist
        params = self._mw.get_parameters_dict()
        output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
        if output_dir and os.path.exists(output_dir):
            posekit_dir = os.path.join(output_dir, "posekit")
            if os.path.exists(posekit_dir) and os.path.isdir(posekit_dir):
                try:
                    import shutil

                    shutil.rmtree(posekit_dir)
                    logger.info(f"Removed posekit directory: {posekit_dir}")
                    cleaned.append("posekit/")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove posekit directory {posekit_dir}: {e}"
                    )
                    failed.append("posekit/")

        if cleaned:
            logger.info(
                f"Cleaned up {len(cleaned)} temporary file(s): {', '.join(cleaned)}"
            )
        if failed:
            logger.warning(
                f"Failed to clean {len(failed)} file(s): {', '.join(failed)}"
            )

    # =========================================================================
    # WIDGET SETUP HELPERS
    # =========================================================================

    def _disable_spinbox_wheel_events(self):
        """Disable wheel events on all spinboxes to prevent accidental value changes."""
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

        # Find all QSpinBox and QDoubleSpinBox widgets
        spinboxes = self._mw.findChildren(QSpinBox) + self._mw.findChildren(
            QDoubleSpinBox
        )
        for spinbox in spinboxes:
            spinbox.wheelEvent = lambda event: None

    def _connect_parameter_signals(self):
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QSpinBox

        widgets_to_connect = (
            self._mw.findChildren(QSpinBox)
            + self._mw.findChildren(QDoubleSpinBox)
            + self._mw.findChildren(QCheckBox)
        )
        for widget in widgets_to_connect:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._mw._on_parameter_changed)
            elif hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._mw._on_parameter_changed)

    # =========================================================================
    # UI SETTINGS PERSISTENCE
    # =========================================================================

    def _load_ui_settings(self) -> dict:
        """Load persistent HYDRA UI settings."""
        import json

        path = self._mw._get_ui_settings_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _queue_ui_state_save(self) -> None:
        """Debounce HYDRA UI settings writes while the user resizes or switches tabs."""
        if hasattr(self._mw, "_ui_state_save_timer"):
            self._mw._ui_state_save_timer.start()

    def _remember_collapsible_state(self, key: str, collapsible) -> None:
        """Restore and track expanded state for a collapsible section."""
        self._mw._collapsible_state_widgets[key] = collapsible
        saved = self._mw._ui_settings.get("collapsed_sections", {}).get(key)
        if isinstance(saved, bool):
            collapsible.setExpanded(saved)
        collapsible.toggled.connect(
            lambda _expanded, _key=key: self._queue_ui_state_save()
        )

    def _restore_ui_state(self) -> None:
        """Apply persisted HYDRA UI layout preferences after construction."""
        settings = self._mw._ui_settings or {}

        detection_index = settings.get("detection_method_index")
        if isinstance(detection_index, int) and hasattr(self._mw, "_detection_panel"):
            self._mw._detection_panel.combo_detection_method.setCurrentIndex(
                max(
                    0,
                    min(
                        detection_index,
                        self._mw._detection_panel.combo_detection_method.count() - 1,
                    ),
                )
            )

        tab_index = settings.get("active_tab_index")
        if isinstance(tab_index, int) and hasattr(self._mw, "tabs"):
            tab_index = max(0, min(tab_index, self._mw.tabs.count() - 1))
            if self._mw.tabs.isTabEnabled(tab_index):
                self._mw.tabs.setCurrentIndex(tab_index)

        splitter_sizes = settings.get("splitter_sizes")
        if (
            isinstance(splitter_sizes, list)
            and len(splitter_sizes) == 2
            and all(isinstance(size, int) and size > 0 for size in splitter_sizes)
            and hasattr(self._mw, "splitter")
        ):
            self._mw.splitter.setSizes(splitter_sizes)

    def _save_ui_settings(self) -> None:
        """Persist HYDRA UI layout preferences without touching tracking configs."""
        import json

        if not hasattr(self._mw, "tabs") or not hasattr(self._mw, "splitter"):
            return

        collapsed_sections = {
            key: widget.isExpanded()
            for key, widget in self._mw._collapsible_state_widgets.items()
        }
        settings = {
            "active_tab_index": int(self._mw.tabs.currentIndex()),
            "splitter_sizes": [int(size) for size in self._mw.splitter.sizes()],
            "detection_method_index": (
                int(self._mw._detection_panel.combo_detection_method.currentIndex())
                if hasattr(self._mw, "_detection_panel")
                else 0
            ),
            "collapsed_sections": collapsed_sections,
        }

        path = self._mw._get_ui_settings_path()
        try:
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            self._mw._ui_settings = settings
        except Exception:
            logger.debug("Failed to save HYDRA UI settings", exc_info=True)

    # =========================================================================
    # CONTEXTUAL/SYNC UI HELPERS
    # =========================================================================

    def _sync_batch_list_ui(self):
        """Refresh the batch list widget with markers for the keystone."""
        from PySide6.QtWidgets import QListWidgetItem

        list_widget = self._panels.setup.list_batch_videos
        current_fp = (
            os.path.normpath(self._panels.setup.file_line.text().strip())
            if self._panels.setup.file_line.text().strip()
            else ""
        )

        selected_row = -1
        blocker = QSignalBlocker(list_widget)
        list_widget.setUpdatesEnabled(False)
        try:
            list_widget.clear()
            for i, fp in enumerate(self._mw.batch_videos):
                norm_fp = os.path.normpath(fp)
                item_text = f"⭐ KEYSTONE: {fp}" if i == 0 else fp

                if norm_fp == current_fp:
                    item_text = f"▶ CURRENT: {item_text}"
                    selected_row = i

                item = QListWidgetItem(item_text)
                item.setToolTip(fp)
                item.setData(Qt.UserRole, fp)

                if norm_fp == current_fp:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                list_widget.addItem(item)

            if 0 <= selected_row < list_widget.count():
                list_widget.setCurrentRow(selected_row)
            else:
                list_widget.clearSelection()
        finally:
            del blocker
            list_widget.setUpdatesEnabled(True)
            list_widget.viewport().update()

    def _refresh_batch_list_current_video(
        self, previous_fp: str | None, current_fp: str
    ):
        """Update only the affected batch-list rows when the current video changes."""
        list_widget = self._panels.setup.list_batch_videos
        current_norm = os.path.normpath(current_fp) if current_fp else ""
        previous_norm = os.path.normpath(previous_fp) if previous_fp else ""
        target_rows = set()

        for row, fp in enumerate(self._mw.batch_videos):
            norm_fp = os.path.normpath(fp)
            if norm_fp in (current_norm, previous_norm):
                target_rows.add(row)

        if not target_rows:
            return

        blocker = QSignalBlocker(list_widget)
        list_widget.setUpdatesEnabled(False)
        try:
            for row in sorted(target_rows):
                item = list_widget.item(row)
                if item is None:
                    continue
                fp = self._mw.batch_videos[row]
                item_text = f"⭐ KEYSTONE: {fp}" if row == 0 else fp
                is_current = os.path.normpath(fp) == current_norm
                if is_current:
                    item_text = f"▶ CURRENT: {item_text}"
                item.setText(item_text)
                item.setToolTip(fp)
                item.setData(Qt.UserRole, fp)
                font = item.font()
                font.setBold(is_current)
                item.setFont(font)

            current_row = next(
                (
                    row
                    for row, fp in enumerate(self._mw.batch_videos)
                    if os.path.normpath(fp) == current_norm
                ),
                -1,
            )
            if 0 <= current_row < list_widget.count():
                list_widget.setCurrentRow(current_row)
            else:
                list_widget.clearSelection()
        finally:
            del blocker
            list_widget.setUpdatesEnabled(True)
            list_widget.viewport().update()

    def _release_video_player_resources(self):
        """Drop resources tied to the previously loaded preview video."""
        if self._mw.video_cap is not None:
            self._mw.video_cap.release()
            self._mw.video_cap = None
        if self._mw.playback_timer:
            self._mw.playback_timer.stop()
            self._mw.playback_timer = None
        self._mw.is_playing = False
        self._mw.last_read_frame_idx = -1
        self._mw.preview_frame_original = None
        self._mw.detection_test_result = None
        self._mw._last_tracking_frame_rgb = None
        self._mw._roi_masked_cache.clear()

    def _sync_video_pose_overlay_controls(self, *_args):
        """Gate pose video overlay controls based on pose inference enable state."""
        panel = getattr(self._mw, "_postprocess_panel", None)
        has_controls = (
            panel is not None
            and hasattr(panel, "check_video_show_pose")
            and hasattr(panel, "combo_video_pose_color_mode")
        )
        if not has_controls:
            return

        video_visible = bool(
            hasattr(self._mw, "_postprocess_panel")
            and self._mw._postprocess_panel.check_video_output.isChecked()
        )
        pose_enabled = self._is_pose_inference_enabled()
        enabled = bool(video_visible and pose_enabled)

        # Hide the whole subsection, not just its contents. With pose off it
        # collapsed to a title, a dead checkbox and a line of explanation --
        # a section that looked like a setting but could not be set.
        self._mw._postprocess_panel.g_video_pose_overlay.setVisible(enabled)
        self._mw._postprocess_panel.check_video_show_pose.setEnabled(enabled)
        show_pose = bool(
            enabled and self._mw._postprocess_panel.check_video_show_pose.isChecked()
        )
        fixed_color_mode = (
            self._mw._postprocess_panel.combo_video_pose_color_mode.currentIndex() == 1
        )

        # Show detailed controls only when pose overlay is on.
        self._mw._postprocess_panel.lbl_video_pose_color_mode.setVisible(show_pose)
        self._mw._postprocess_panel.combo_video_pose_color_mode.setVisible(show_pose)
        self._mw._postprocess_panel.lbl_video_pose_point_radius.setVisible(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_radius.setVisible(show_pose)
        self._mw._postprocess_panel.lbl_video_pose_point_thickness.setVisible(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_thickness.setVisible(
            show_pose
        )
        self._mw._postprocess_panel.lbl_video_pose_line_thickness.setVisible(show_pose)
        self._mw._postprocess_panel.spin_video_pose_line_thickness.setVisible(show_pose)

        show_fixed_color = bool(show_pose and fixed_color_mode)
        self._mw._postprocess_panel.lbl_video_pose_color_label.setVisible(
            show_fixed_color
        )
        self._mw._postprocess_panel.btn_video_pose_color.setVisible(show_fixed_color)
        self._mw._postprocess_panel.lbl_video_pose_color.setVisible(show_fixed_color)

        self._mw._postprocess_panel.combo_video_pose_color_mode.setEnabled(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_radius.setEnabled(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_thickness.setEnabled(
            show_pose
        )
        self._mw._postprocess_panel.spin_video_pose_line_thickness.setEnabled(show_pose)
        self._mw._postprocess_panel.btn_video_pose_color.setEnabled(show_fixed_color)

        # The "enable pose extraction" branch is gone with the section: nobody
        # can read a hint inside a hidden group. That pointer now lives in the
        # video section's help, which stays reachable.
        self._mw._postprocess_panel.lbl_video_pose_disabled_hint.setVisible(enabled)
        self._mw._postprocess_panel.lbl_video_pose_disabled_hint.setText(
            "Pose overlay will use keypoints from pose-augmented tracking output."
        )

    def _sync_pose_backend_ui(self):
        """Show/hide backend-specific pose controls."""
        if not hasattr(self._mw, "_identity_panel"):
            return
        backend = (
            self._mw._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        is_sleap = backend == "sleap"
        if hasattr(self._mw, "_identity_panel") and hasattr(
            self._mw._identity_panel, "pose_sleap_env_row_widget"
        ):
            self._mw._set_form_row_visible(
                self._mw._identity_panel.form_pose_runtime,
                self._mw._identity_panel.pose_sleap_env_row_widget,
                is_sleap,
            )
        # Refresh pose model combo to show models for the selected backend.
        self._mw._refresh_pose_model_combo(
            preferred_model_path=self._mw._pose_model_path_for_backend(backend)
        )
        self._mw._on_runtime_context_changed()

    def _update_obb_mode_warning(self) -> None:
        """Show a performance hint when device/mode is a suboptimal combination."""
        if not hasattr(self._mw, "_detection_panel"):
            return
        resolved = (
            self._mw._resolved_obb_backend()
            if hasattr(self._mw, "_setup_panel")
            else None
        )
        sequential = (
            hasattr(self._mw, "_detection_panel")
            and self._mw._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
        )
        # "coreml" is Apple GPU-Fast's concrete backend, and is just as
        # Apple-Silicon-bound as native "mps" for this warning.
        is_apple_silicon = resolved is not None and (
            resolved.device == "mps" or resolved.backend == "coreml"
        )
        is_cuda = resolved is not None and resolved.device == "cuda"
        if is_apple_silicon and sequential:
            msg = (
                "⚠ Sequential mode is significantly slower on Apple Silicon (MPS). "
                "Direct mode is recommended for MPS — it runs ~4× faster."
            )
        elif is_cuda and not sequential:
            msg = (
                "⚠ Sequential mode is typically faster on CUDA GPUs. "
                "Consider switching to Sequential for better throughput."
            )
        else:
            msg = ""
        self._mw._detection_panel.lbl_obb_mode_warning.setText(msg)
        self._mw._detection_panel.lbl_obb_mode_warning.setVisible(bool(msg))

    def _update_range_info(self):
        """Update the frame range info label."""
        start = self._panels.setup.spin_start_frame.value()
        end = self._panels.setup.spin_end_frame.value()
        num_frames = end - start + 1

        fps = self._panels.setup.spin_fps.value()
        duration_sec = num_frames / fps if fps > 0 else 0

        self._panels.setup.lbl_range_info.setText(
            f"Tracking {num_frames} frames ({duration_sec:.2f} seconds)"
        )

    def _commit_pending_setup_edits(self):
        """Commit any typed spinbox text before reading setup values."""
        changed_widget = None
        for spinbox in (
            self._panels.setup.spin_start_frame,
            self._panels.setup.spin_end_frame,
        ):
            if spinbox.lineEdit().text() != str(spinbox.value()):
                changed_widget = spinbox
            spinbox.interpretText()
        if self._panels.setup.spin_traj_hist.lineEdit().text() != str(
            self._panels.setup.spin_traj_hist.value()
        ):
            self._panels.setup.spin_traj_hist.interpretText()
        self._normalize_frame_range(changed_widget=changed_widget)
        self._sync_trail_history_bounds()

    def _normalize_frame_range(self, changed_widget=None):
        """Clamp frame range bounds while preserving the field the user just edited."""
        start_spin = self._panels.setup.spin_start_frame
        end_spin = self._panels.setup.spin_end_frame
        max_frame = max(0, min(start_spin.maximum(), end_spin.maximum()))

        start = max(0, min(start_spin.value(), max_frame))
        end = max(0, min(end_spin.value(), max_frame))

        if start > end:
            if changed_widget is end_spin:
                start = end
            else:
                end = start

        if start_spin.value() != start:
            start_spin.blockSignals(True)
            start_spin.setValue(start)
            start_spin.blockSignals(False)
        if end_spin.value() != end:
            end_spin.blockSignals(True)
            end_spin.setValue(end)
            end_spin.blockSignals(False)

    def _sync_trail_history_bounds(self):
        """Cap trail history to the loaded video's frame count."""
        spinbox = self._panels.setup.spin_traj_hist
        previous_value = spinbox.value()
        max_history = (
            max(0, int(self._mw.video_total_frames))
            if getattr(self._mw, "video_total_frames", 0) > 0
            else max(int(spinbox.maximum()), 60)
        )
        spinbox.blockSignals(True)
        spinbox.setRange(-1, max_history)
        if previous_value > max_history:
            spinbox.setValue(max_history)
        spinbox.blockSignals(False)
        if spinbox.value() != previous_value:
            self._on_trail_history_changed()

    def _on_trail_history_changed(self):
        """Apply special trail-history values to the trajectory overlay toggle."""
        trail_history = self._panels.setup.spin_traj_hist.value()
        if trail_history == 0:
            self._panels.setup.chk_show_trajectories.setChecked(False)
        elif not self._panels.setup.chk_show_trajectories.isChecked():
            self._panels.setup.chk_show_trajectories.setChecked(True)

    # =========================================================================
    # PIPELINE STATE QUERIES
    # =========================================================================

    def _is_pose_inference_enabled(self) -> bool:
        """Return whether pose inference is actively enabled for the run."""
        from hydra_suite.core.tracking import session_policy

        return session_policy.is_pose_inference_enabled(
            self._mw._config_orch.build_config_dict()
        )

    def _is_headtail_compute_enabled(self) -> bool:
        """Return whether head-tail analysis is actively configured for the run."""
        from hydra_suite.core.tracking import session_policy

        return session_policy.is_headtail_compute_enabled(
            self._mw._config_orch.build_config_dict()
        )

    def _is_individual_pipeline_enabled(self) -> bool:
        """Return effective runtime state for individual analysis pipeline.

        NOTE: intentionally does NOT delegate through
        ``ConfigOrchestrator.build_config_dict()`` — that method itself calls
        this predicate (via ``self._mw._is_individual_pipeline_enabled()``) to
        populate ``enable_identity_analysis``/``enable_individual_pipeline``,
        so doing so would recurse infinitely. Build the minimal input the pure
        predicate needs directly from the detection widget instead.
        """
        from hydra_suite.core.tracking import session_policy

        return session_policy.is_individual_pipeline_enabled(
            {"detection_method": self._detection_method_value()}
        )

    def _detection_method_value(self) -> str:
        """Return the config-dict-shaped detection_method string from the widget."""
        if not hasattr(self._mw, "_detection_panel"):
            return "background_subtraction"
        return (
            "background_subtraction"
            if self._mw._detection_panel.combo_detection_method.currentIndex() == 0
            else "yolo_obb"
        )

    def _is_realtime_tracking_mode_enabled(self) -> bool:
        """Return True when the setup tab requests streaming realtime workflow."""
        if not hasattr(self._mw, "_setup_panel"):
            return False
        return bool(self._mw._setup_panel.chk_realtime_mode.isChecked())

    def _workflow_mode_key(self) -> str:
        """Return the normalized workflow mode key for runtime parameters.

        NOTE: intentionally does NOT delegate through
        ``ConfigOrchestrator.build_config_dict()`` — that method itself calls
        this predicate (via ``self._mw._session_orch._workflow_mode_key()``)
        to populate ``tracking_workflow_mode``, so doing so would recurse
        infinitely. Build the minimal input directly from the widget instead.
        """
        from hydra_suite.core.tracking import session_policy

        return session_policy.workflow_mode_key(
            {"realtime_tracking_mode": self._is_realtime_tracking_mode_enabled()}
        )

    def _should_export_final_canonical_images(self) -> bool:
        """Return effective runtime state for final canonical still export.

        NOTE: intentionally does NOT delegate through
        ``ConfigOrchestrator.build_config_dict()`` — that method itself calls
        this predicate (via ``self._mw._is_individual_image_save_enabled()``,
        an alias defined below) to populate
        ``export_final_canonical_images``/``enable_individual_dataset``, so
        doing so would recurse infinitely. Build the minimal input directly
        from the relevant widgets instead.
        """
        if not hasattr(self._mw, "_dataset_panel"):
            return False
        from hydra_suite.core.tracking import session_policy

        return session_policy.should_export_final_canonical_images(
            {
                "enable_individual_dataset": self._mw._dataset_panel.chk_enable_individual_dataset.isChecked(),
                "detection_method": self._detection_method_value(),
            }
        )

    def _is_individual_image_save_enabled(self) -> bool:
        """Backward-compatible alias for final canonical still export state."""
        return self._should_export_final_canonical_images()

    def _should_export_final_media_videos(self) -> bool:
        """Return True when final per-track videos should be exported."""
        from hydra_suite.core.tracking import session_policy

        return session_policy.should_export_final_media_videos(
            self._mw._config_orch.build_config_dict()
        )

    def _should_run_interpolated_postpass(self) -> bool:
        """
        Return True when interpolated post-pass should run.

        We run this pass when interpolation is enabled and either:
        - individual crop saving is enabled, or
        - pose export is enabled (to fill occluded-frame pose rows in final CSV), or
        - final media video export is enabled (to cache interpolated ROI geometry).
        """
        from hydra_suite.core.tracking import session_policy

        return session_policy.should_run_interpolated_postpass(
            self._mw._config_orch.build_config_dict()
        )

    # =========================================================================
    # RUNTIME / COMPUTE OPTIONS
    # =========================================================================

    def _current_runtime_tier(self) -> str:
        """Return the currently selected RuntimeTier id ("cpu"/"gpu"/"gpu_fast")."""
        if not hasattr(self._mw, "_setup_panel"):
            return "gpu"
        if not hasattr(self._mw._setup_panel, "combo_runtime_tier"):
            return "gpu"
        data = self._mw._setup_panel.combo_runtime_tier.currentData()
        return str(data).strip() if data else "gpu"

    def _resolved_obb_backend(self):
        """Resolve the OBB-stage backend for the selected tier and host platform."""
        from hydra_suite.runtime.resolver import RuntimeResolver

        return RuntimeResolver(self._current_runtime_tier(), detect_platform()).resolve(
            "obb"
        )

    def _has_cnn_identity_enabled(self) -> bool:
        """Return True when CNN identity analysis is configured and enabled."""
        if not (
            self._is_individual_pipeline_enabled()
            and self._mw._is_identity_analysis_enabled()
        ):
            return False
        return bool(self._mw._identity_config().get("cnn_classifiers", []))

    def _runtime_requires_fixed_yolo_batch(self, resolved=None) -> bool:
        """Return True when runtime mandates a fixed YOLO batch size."""
        resolved = resolved if resolved is not None else self._resolved_obb_backend()
        if resolved.backend == "tensorrt":
            return True
        return self._gpu_fast_obb_is_coreml_only()

    def _gpu_fast_obb_is_coreml_only(self) -> bool:
        """Return True when gpu_fast OBB detection will run on CoreML.

        ``_resolved_obb_backend()`` reports the "coreml" backend directly for
        gpu_fast on Apple Silicon, and the OBB stage
        internally upgrades to a CoreML direct executor whenever the exported
        ``.mlpackage`` artifact is available (see
        ``core/inference/runtime.py:resolved_backend_for``). CoreML's
        OBB export cannot use a dynamic batch axis (Spec 1 Phase A/B,
        2026-07-04: ultralytics' CoreML export hard-crashes at compile time
        for OBB models when both the batch and spatial dims are dynamic
        together), so OBB detection under this path is permanently batch=1,
        even though CoreML classification (identity/head-tail/CNN) batches
        normally. This is a platform limitation, not a config choice.
        """
        if self._current_runtime_tier() != "gpu_fast":
            return False
        platform = detect_platform()
        return bool(platform.has_mps and not platform.has_cuda)

    def _on_runtime_tier_changed(self) -> None:
        """Handle tier combo change: store tier to config and refresh dependent controls."""
        if hasattr(self._mw, "_setup_panel") and hasattr(
            self._mw._setup_panel, "combo_runtime_tier"
        ):
            tier = self._mw._setup_panel.combo_runtime_tier.currentData()
            if tier and hasattr(self._mw, "config"):
                self._mw.config.runtime_tier = str(tier)
        # _on_runtime_context_changed() refreshes the fallback hint, so no direct
        # _update_runtime_fallback_hint() call is needed here (avoids double-run).
        self._on_runtime_context_changed()

    def _update_runtime_fallback_hint(self) -> None:
        """Populate the GPU-Fast fallback hint (spec §5.4) under the tier selector.

        Informational only: at configuration time we cannot know which stages
        have a fast (TensorRT/CoreML) artifact, so we state the best-effort
        contract when GPU-Fast is selected and clear the hint otherwise.
        """
        panel = getattr(self._mw, "_setup_panel", None)
        lbl = getattr(panel, "lbl_runtime_fallback", None)
        if lbl is None:
            return
        tier = None
        if panel is not None and hasattr(panel, "combo_runtime_tier"):
            tier = panel.combo_runtime_tier.currentData()
        if str(tier) == "gpu_fast":
            from hydra_suite.runtime.resolver import detect_platform

            platform = detect_platform()
            fast = "TensorRT" if platform.has_cuda else "CoreML"
            lbl.setText(
                f"GPU-Fast: uses {fast} where a fast artifact exists, "
                "else the native GPU per stage."
            )
            lbl.setVisible(True)
        else:
            lbl.setText("")
            lbl.setVisible(False)

    def _on_runtime_context_changed(self, *_args):
        """Sync dependent controls when the runtime tier or context changes."""
        self._update_runtime_fallback_hint()
        self._mw._update_obb_mode_warning()
        if hasattr(self._mw, "_detection_panel"):
            self._mw._detection_panel._sync_live_detection_batch_controls()
        if hasattr(self._mw, "_identity_panel"):
            self._mw._identity_panel._sync_realtime_individual_batch_ui()
        if hasattr(self._mw, "_dataset_panel"):
            self._mw._dataset_panel.refresh_export_levels()

    def _set_form_row_visible(self, form_layout, field_widget, visible: bool):
        """Show/hide a QFormLayout row by field widget."""
        setup_panel = getattr(self._mw, "_setup_panel", None)
        if setup_panel is not None:
            perf_handler = getattr(
                setup_panel, "_set_performance_control_visible", None
            )
            if callable(perf_handler) and perf_handler(field_widget, visible):
                return
        if form_layout is None or field_widget is None:
            return
        label = form_layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(bool(visible))
        field_widget.setVisible(bool(visible))

    # =========================================================================
    # INDIVIDUAL ANALYSIS UI
    # =========================================================================

    def _sync_individual_analysis_mode_ui(self):
        """Enforce YOLO-only pipeline and run/save dependency in UI."""
        has_save_toggle = hasattr(self._mw, "_dataset_panel")
        is_yolo = self._mw._is_yolo_detection_mode()

        if hasattr(self._mw, "tabs") and hasattr(self._mw, "_identity_panel"):
            tab_index = self._mw.tabs.indexOf(self._mw._identity_panel)
            if tab_index >= 0:
                if (
                    not is_yolo
                    and self._mw.tabs.currentWidget() is self._mw._identity_panel
                ):
                    fallback_index = self._mw.tabs.indexOf(
                        getattr(self._mw, "_detection_panel", self._mw._setup_panel)
                    )
                    if fallback_index >= 0:
                        self._mw.tabs.setCurrentIndex(fallback_index)
                if hasattr(self._mw.tabs, "setTabVisible"):
                    self._mw.tabs.setTabVisible(tab_index, is_yolo)
                elif hasattr(self._mw.tabs, "tabBar") and hasattr(
                    self._mw.tabs.tabBar(), "setTabVisible"
                ):
                    self._mw.tabs.tabBar().setTabVisible(tab_index, is_yolo)
                self._mw.tabs.setTabEnabled(tab_index, is_yolo)

        pipeline_enabled = self._is_individual_pipeline_enabled()

        if hasattr(self._mw, "_identity_panel"):
            self._mw._identity_panel.lbl_individual_yolo_only_notice.setVisible(
                not is_yolo
            )
            self._mw._identity_panel.g_headtail.setVisible(pipeline_enabled)
            self._mw._identity_panel.g_headtail.setEnabled(pipeline_enabled)
            self._mw._identity_panel.g_identity.setVisible(pipeline_enabled)
            self._mw._identity_panel.g_identity.setEnabled(pipeline_enabled)
            self._mw._identity_panel.g_pose_runtime.setVisible(pipeline_enabled)
            self._mw._identity_panel.g_pose_runtime.setEnabled(pipeline_enabled)
            self._mw._identity_panel.g_individual_pipeline_common.setVisible(
                pipeline_enabled
            )
            self._mw._identity_panel.g_individual_pipeline_common.setEnabled(
                pipeline_enabled
            )
            self._mw._identity_panel._sync_headtail_analysis_ui()
            self._mw._identity_panel._sync_identity_method_ui()
            self._mw._identity_panel._sync_pose_analysis_ui()

        # Hide tracking-side identity decoder when classification is OFF.
        if hasattr(self._mw, "_tracking_panel"):
            # The master toggle alone is not enough: identity with no AprilTags
            # and no CNN classifier produces no evidence, so every downstream
            # identity control would be inert.
            identity_active = pipeline_enabled and self._mw._has_identity_source()
            self._mw._tracking_panel.set_identity_section_visible(identity_active)
            if hasattr(self._mw, "_postprocess_panel"):
                self._mw._postprocess_panel.set_identity_section_visible(
                    identity_active
                )
        if hasattr(self._mw, "_dataset_panel"):
            self._mw._dataset_panel.g_individual_dataset.setVisible(pipeline_enabled)
            self._mw._dataset_panel.g_individual_dataset.setEnabled(pipeline_enabled)
            self._mw._dataset_panel.g_oriented_videos.setVisible(pipeline_enabled)
            self._mw._dataset_panel.g_oriented_videos.setEnabled(pipeline_enabled)
        self._sync_pose_backend_ui()

        if has_save_toggle:
            self._mw._dataset_panel.chk_enable_individual_dataset.setEnabled(
                pipeline_enabled
            )

        save_enabled = self._should_export_final_canonical_images()
        if hasattr(self._mw, "_dataset_panel"):
            self._mw._dataset_panel.ind_output_group.setVisible(save_enabled)
            self._mw._dataset_panel.ind_output_group.setEnabled(save_enabled)
            self._mw._dataset_panel.chk_suppress_foreign_obb_individual_dataset.setVisible(
                save_enabled
            )
            self._mw._dataset_panel.chk_suppress_foreign_obb_individual_dataset.setEnabled(
                save_enabled
            )
            has_headtail = bool(
                str(
                    self._mw._identity_panel._get_selected_yolo_headtail_model_path()
                    or ""
                ).strip()
            )
            oriented_enabled = pipeline_enabled and has_headtail
            self._mw._dataset_panel.chk_generate_individual_track_videos.setEnabled(
                oriented_enabled
            )
            self._mw._dataset_panel.chk_suppress_foreign_obb_oriented_videos.setEnabled(
                oriented_enabled
                and self._mw._dataset_panel.chk_generate_individual_track_videos.isChecked()
            )
            if not oriented_enabled:
                self._mw._dataset_panel.chk_generate_individual_track_videos.setChecked(
                    False
                )
                self._mw._dataset_panel.chk_generate_individual_track_videos.setToolTip(
                    "Requires head-tail orientation to be enabled with a configured model."
                )
            else:
                self._mw._dataset_panel.chk_generate_individual_track_videos.setToolTip(
                    "After final cleaning completes, export one orientation-fixed video per\n"
                    "final TrajectoryID by streaming the source video and using the detection\n"
                    "cache plus interpolated ROI cache. Independent from saved crop files."
                )
        self._sync_video_pose_overlay_controls()
        self._mw._on_runtime_context_changed()
        self._sync_directed_orient_posthoc_ui()

    def _sync_directed_orient_posthoc_ui(self) -> None:
        """Update tracking and postprocess panels when head-tail/pose model changes.

        Hides the online flip-hysteresis controls and shows a note when a
        directed heading source (head-tail or pose model) is active, since the
        pipeline then uses global post-hoc heading consistency instead.
        """
        posthoc_active = bool(self._is_headtail_compute_enabled()) or bool(
            self._is_pose_inference_enabled()
        )
        if hasattr(self._mw, "_tracking_panel") and hasattr(
            self._mw._tracking_panel, "sync_directed_orient_posthoc_ui"
        ):
            self._mw._tracking_panel.sync_directed_orient_posthoc_ui(posthoc_active)
        if hasattr(self._mw, "_postprocess_panel") and hasattr(
            self._mw._postprocess_panel, "sync_heading_flip_posthoc_ui"
        ):
            self._mw._postprocess_panel.sync_heading_flip_posthoc_ui(posthoc_active)

    def _select_individual_background_color(self):
        """Open color picker for individual dataset background color."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        b, g, r = self._mw._identity_panel._background_color
        initial_color = QColor(r, g, b)
        color = QColorDialog.getColor(
            initial_color, self._mw, "Choose Background Color"
        )
        if color.isValid():
            self._mw._identity_panel._background_color = (
                color.blue(),
                color.green(),
                color.red(),
            )
            self._mw._update_background_color_button()

    def _update_background_color_button(self):
        """Update the color button display and label."""
        b, g, r = self._mw._identity_panel._background_color
        self._mw._identity_panel.btn_background_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self._mw._identity_panel.lbl_background_color.setText(
            f"{self._mw._identity_panel._background_color}"
        )

    def _compute_median_background_color(self):
        """Compute median color from current preview frame or load from video."""
        frame = None
        if (
            hasattr(self._mw, "preview_frame_original")
            and self._mw.preview_frame_original is not None
        ):
            frame = cv2.cvtColor(self._mw.preview_frame_original, cv2.COLOR_RGB2BGR)
        elif self._mw.current_video_path:
            cap = cv2.VideoCapture(self._mw.current_video_path)
            if cap.isOpened():
                ret, frame_bgr = cap.read()
                cap.release()
                if ret:
                    frame = frame_bgr

        if frame is None:
            QMessageBox.warning(
                self._mw,
                "No Frame",
                "Please load a video first to compute median color.",
            )
            return

        try:
            from hydra_suite.utils.image_processing import (
                compute_median_color_from_frame,
            )

            median_color = compute_median_color_from_frame(frame)
            self._mw._identity_panel._background_color = tuple(
                int(c) for c in median_color
            )
            self._mw._update_background_color_button()
            QMessageBox.information(
                self._mw,
                "Median Color Computed",
                f"Background color set to median:\nBGR: {median_color}",
            )
        except Exception as e:
            logger.error(f"Failed to compute median color: {e}")
            QMessageBox.warning(
                self._mw, "Error", f"Failed to compute median color:\n{e}"
            )

    # =========================================================================
    # VIDEO PLAYER
    # =========================================================================

    @staticmethod
    def _probe_video_io(video_path, set_status, _set_progress):
        """Open video file and decode the first frame — pure I/O, thread-safe.

        The VideoCapture is opened and closed entirely within this method so
        that no cv2 object crosses a thread boundary (GStreamer/FFMPEG backends
        on Linux are not safe to hand off between threads even sequentially).

        Returns a dict with keys: total_frames, fps, width, height,
        first_frame_rgb (ndarray or None).  Raises RuntimeError if the file
        cannot be opened.
        """
        set_status(f"Opening {os.path.basename(video_path)}\u2026")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        set_status("Decoding first frame\u2026")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame_bgr = cap.read()
        first_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if ok else None
        cap.release()  # Close here — do NOT pass cap across thread boundary
        return {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "first_frame_rgb": first_frame_rgb,
        }

    def _init_video_player(self, video_path, _probe=None):
        """Initialize video player with the loaded video.

        Parameters
        ----------
        video_path:
            Path to the video file.
        _probe:
            Optional pre-loaded result from :meth:`_probe_video_io`.  When
            provided the expensive ``cv2.VideoCapture`` open and first-frame
            decode are skipped because they were already done in a background
            thread via ``run_blocking_with_busy_dialog``.
        """
        self._release_video_player_resources()

        if _probe is not None:
            # Re-open cap on the main thread (fast: just open, no slow FRAME_COUNT
            # query needed since we already have the metadata from the probe thread).
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return
            total_frames = _probe["total_frames"]
            fps = _probe["fps"]
            width = _probe["width"]
            height = _probe["height"]
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._mw.video_cap = cap
        self._mw.video_total_frames = total_frames
        self._mw.video_width = width
        self._mw.video_height = height

        self._panels.setup.lbl_video_info.setText(
            f"Video: {total_frames} frames, {width}x{height}, {fps:.2f} FPS"
        )
        self._panels.setup.slider_timeline.setMaximum(total_frames - 1)
        self._panels.setup.slider_timeline.setEnabled(True)
        self._panels.setup.btn_first_frame.setEnabled(True)
        self._panels.setup.btn_prev_frame.setEnabled(True)
        self._panels.setup.btn_play_pause.setEnabled(True)
        self._panels.setup.btn_next_frame.setEnabled(True)
        self._panels.setup.btn_last_frame.setEnabled(True)
        self._panels.setup.btn_random_seek.setEnabled(True)
        self._panels.setup.combo_playback_speed.setEnabled(True)
        self._panels.setup.spin_start_frame.setMaximum(total_frames - 1)
        self._panels.setup.spin_start_frame.setEnabled(True)
        self._panels.setup.spin_end_frame.setMaximum(total_frames - 1)
        self._panels.setup.spin_end_frame.setValue(total_frames - 1)
        self._panels.setup.spin_end_frame.setEnabled(True)
        self._sync_trail_history_bounds()
        self._panels.setup.btn_set_start_current.setEnabled(True)
        self._panels.setup.btn_set_end_current.setEnabled(True)
        self._panels.setup.btn_reset_range.setEnabled(True)
        self._panels.setup.g_video_player.setVisible(True)

        self._mw.video_current_frame_idx = 0
        if _probe is not None and _probe.get("first_frame_rgb") is not None:
            # Apply the already-decoded frame directly — avoids a second disk read
            self._mw.preview_frame_original = _probe["first_frame_rgb"]
            self._mw.last_read_frame_idx = 0
            self._mw.detection_test_result = None
            if hasattr(self._mw, "_detection_panel"):
                self._mw._detection_panel._update_preview_display()
            self._set_current_frame_label(0)
            self._panels.setup.slider_timeline.blockSignals(True)
            self._panels.setup.slider_timeline.setValue(0)
            self._panels.setup.slider_timeline.blockSignals(False)
        else:
            self._mw._display_current_frame()
        QTimer.singleShot(0, self._mw._fit_image_to_screen)
        self._update_range_info()
        logger.info(f"Video player initialized: {total_frames} frames")

    def _set_current_frame_label(self, frame_idx: int, *, scrubbing: bool = False):
        """Refresh the preview frame label without forcing a video seek."""
        total_frames = max(self._mw.video_total_frames - 1, 0)
        suffix = " (release to seek)" if scrubbing else ""
        self._panels.setup.lbl_current_frame.setText(
            f"Frame: {frame_idx}/{total_frames}{suffix}"
        )

    def _seek_preview_frame(self, frame_idx: int):
        """Route programmatic seeks through the slider without double-rendering."""
        if self._mw.video_total_frames <= 0:
            return
        bounded_idx = max(0, min(frame_idx, self._mw.video_total_frames - 1))
        self._mw.video_current_frame_idx = bounded_idx
        slider = self._panels.setup.slider_timeline
        if slider.value() != bounded_idx:
            slider.setValue(bounded_idx)
            return
        self._mw._display_current_frame()

    def _display_current_frame(self):
        """Display the current frame in the video label."""
        if self._mw.video_cap is None:
            return
        if self._mw.last_read_frame_idx != self._mw.video_current_frame_idx - 1:
            self._mw.video_cap.set(
                cv2.CAP_PROP_POS_FRAMES, self._mw.video_current_frame_idx
            )
        ret, frame = self._mw.video_cap.read()
        if not ret:
            return
        self._mw.last_read_frame_idx = self._mw.video_current_frame_idx
        self._mw.preview_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._mw.detection_test_result = None
        if hasattr(self._mw, "_detection_panel"):
            self._mw._detection_panel._update_preview_display()
        self._set_current_frame_label(self._mw.video_current_frame_idx)
        self._panels.setup.slider_timeline.blockSignals(True)
        self._panels.setup.slider_timeline.setValue(self._mw.video_current_frame_idx)
        self._panels.setup.slider_timeline.blockSignals(False)

    def _on_timeline_changed(self, value):
        """Handle timeline slider change."""
        if (
            self._mw.is_playing
            and not self._panels.setup.slider_timeline.signalsBlocked()
        ):
            self._mw._stop_playback()
        self._mw.video_current_frame_idx = value
        self._mw._display_current_frame()

    def _on_timeline_pressed(self):
        """Pause playback before interactive timeline scrubbing begins."""
        if self._mw.is_playing:
            self._mw._stop_playback()

    def _on_timeline_moved(self, value):
        """Update the frame counter while the user drags the timeline handle."""
        if self._panels.setup.slider_timeline.hasTracking():
            return
        self._set_current_frame_label(value, scrubbing=True)

    def _goto_first_frame(self):
        """Go to the first frame."""
        if self._mw.is_playing:
            self._mw._stop_playback()
        self._seek_preview_frame(0)

    def _goto_prev_frame(self):
        """Go to the previous frame."""
        if self._mw.is_playing:
            self._mw._stop_playback()
        if self._mw.video_current_frame_idx > 0:
            self._seek_preview_frame(self._mw.video_current_frame_idx - 1)

    def _goto_next_frame(self):
        """Go to the next frame."""
        if self._mw.is_playing:
            self._mw._stop_playback()
        if self._mw.video_current_frame_idx < self._mw.video_total_frames - 1:
            self._seek_preview_frame(self._mw.video_current_frame_idx + 1)

    def _goto_last_frame(self):
        """Go to the last frame."""
        if self._mw.is_playing:
            self._mw._stop_playback()
        self._seek_preview_frame(self._mw.video_total_frames - 1)

    def _goto_random_frame(self):
        """Jump to a random frame."""
        import numpy as np

        if self._mw.is_playing:
            self._mw._stop_playback()
        if self._mw.video_total_frames <= 0:
            return
        self._seek_preview_frame(np.random.randint(0, self._mw.video_total_frames))

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self._mw.is_playing:
            self._mw._stop_playback()
        else:
            self._mw._start_playback()

    def _start_playback(self):
        """Start video playback."""
        from PySide6.QtCore import QTimer

        if self._mw.video_cap is None or self._mw.is_playing:
            return
        self._mw.is_playing = True
        self._panels.setup.btn_play_pause.setText("\u23f8 Pause")
        speed_text = self._panels.setup.combo_playback_speed.currentText()
        speed = float(speed_text.replace("x", ""))
        fps = self._mw.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        interval_ms = max(1, int((1000.0 / fps) / speed))
        if self._mw.playback_timer is None:
            self._mw.playback_timer = QTimer(self._mw)
        self._mw.playback_timer.singleShot(interval_ms, self._mw._playback_step)

    def _stop_playback(self):
        """Stop video playback."""
        if not self._mw.is_playing:
            return
        self._mw.is_playing = False
        self._panels.setup.btn_play_pause.setText("\u25b6 Play")
        if self._mw.playback_timer and self._mw.playback_timer.isActive():
            self._mw.playback_timer.stop()

    def _playback_step(self):
        """Advance one frame during playback."""
        if self._mw.playback_timer and self._mw.playback_timer.isActive():
            self._mw.playback_timer.stop()
        if not self._mw.is_playing:
            return
        if self._mw.video_current_frame_idx < self._mw.video_total_frames - 1:
            self._mw.video_current_frame_idx += 1
            self._mw._display_current_frame()
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()
            if self._mw.is_playing and self._mw.playback_timer:
                speed_text = self._panels.setup.combo_playback_speed.currentText()
                speed = float(speed_text.replace("x", ""))
                fps = self._mw.video_cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                interval_ms = max(1, int((1000.0 / fps) / speed))
                self._mw.playback_timer.singleShot(interval_ms, self._mw._playback_step)
        else:
            self._mw._stop_playback()

    def _on_frame_range_changed(self, changed_widget=None):
        """Handle frame range spinbox changes."""
        self._normalize_frame_range(changed_widget=changed_widget)
        self._update_range_info()

    def _set_start_to_current(self):
        """Set start frame to current frame."""
        self._panels.setup.spin_start_frame.setValue(self._mw.video_current_frame_idx)

    def _set_end_to_current(self):
        """Set end frame to current frame."""
        self._panels.setup.spin_end_frame.setValue(self._mw.video_current_frame_idx)

    def _reset_frame_range(self):
        """Reset frame range to full video."""
        self._panels.setup.spin_start_frame.setValue(0)
        self._panels.setup.spin_end_frame.setValue(self._mw.video_total_frames - 1)

    def _update_fps_info(self):
        """Update the FPS info label with time per frame."""
        fps = self._panels.setup.spin_fps.value()
        time_per_frame = 1000.0 / fps
        self._panels.setup.label_fps_info.setText(
            f"= {time_per_frame:.2f} ms per frame"
        )

    # =========================================================================
    # ROI SELECTION AND VIDEO INTERACTION
    # =========================================================================

    def _set_preview_test_running(self, running: bool):
        """Lock/unlock UI while async preview detection is running."""
        if running:
            # Disable without saving state — restore re-applies the idle state
            # instead of trusting a snapshot, so finished_signal handlers (e.g.
            # _update_detection_stats) can safely toggle widgets mid-flight.
            self._mw._set_interactive_widgets_enabled(False, remember_state=False)
            self._mw._set_video_interaction_enabled(False)
            self._mw.btn_test_detection.setText("Testing Detection...")
            self._mw.btn_test_detection.setEnabled(False)
            self._mw.progress_label.setText("Testing detection on preview...")
            self._mw.progress_label.setVisible(True)
            self._mw.progress_bar.setRange(0, 0)
            self._mw.progress_bar.setVisible(True)
            return

        # Test detection only runs from idle, so re-apply idle to fully restore.
        self._apply_ui_state("idle")
        self._mw._sync_individual_analysis_mode_ui()
        self._mw._sync_pose_backend_ui()
        if hasattr(self._mw, "_detection_panel"):
            self._mw._detection_panel._sync_model_selector_buttons()
        if hasattr(self._mw, "_identity_panel"):
            self._mw._identity_panel._sync_headtail_model_remove_button()
            self._mw._identity_panel._sync_pose_model_remove_button()
            for row in self._mw._identity_panel._cnn_classifier_rows():
                row._sync_model_ui()
        self._mw.btn_test_detection.setText("Test Detection on Preview")
        self._mw.btn_test_detection.setEnabled(
            self._mw.preview_frame_original is not None
        )
        # Result-driven enables: idle blanket-enables everything, but the
        # auto-set buttons should reflect whether detections were found.
        detected = getattr(self._mw, "detected_sizes", None)
        has_detections = bool(detected) and int(detected.get("count", 0)) > 0
        if hasattr(self._mw, "_detection_panel"):
            self._mw._detection_panel.btn_auto_set_body_size.setEnabled(has_detections)
            self._mw._detection_panel.btn_auto_set_aspect_ratio.setEnabled(
                has_detections
            )
        self._mw.progress_bar.setRange(0, 100)
        self._mw._refresh_progress_visibility()

    def _handle_video_double_click(self, evt):
        """Handle double-click: finish an in-progress polygon, else fit to screen."""
        if not self._mw._video_interactions_enabled:
            evt.ignore()
            return
        if evt.button() != Qt.LeftButton:
            return
        if self._mw.roi_selection_active and self._mw.roi_current_mode == "polygon":
            self.finish_roi_selection()
        else:
            QTimer.singleShot(0, self._mw._fit_image_to_screen)

    def _handle_video_wheel(self, evt):
        """Handle mouse wheel - zoom in/out, anchored to the cursor position."""
        if not self._mw._video_interactions_enabled:
            evt.ignore()
            return
        if self._mw.video_label.is_input_paused():
            evt.ignore()
            return
        if evt.modifiers() == Qt.ControlModifier:
            delta = evt.angleDelta().y()
            current_zoom = self._mw.slider_zoom.value()
            zoom_change = max(-15, min(15, round(delta / 24)))
            new_zoom = max(10, min(400, current_zoom + zoom_change))

            if new_zoom == current_zoom:
                evt.accept()
                return

            # --- Cursor-anchored zoom (parameter-helper pattern) ---
            # Map cursor from video_label coordinates to viewport coordinates.
            viewport = self._mw.scroll.viewport()
            label_pos = evt.position().toPoint()
            viewport_pos = viewport.mapFromGlobal(
                self._mw.video_label.mapToGlobal(label_pos)
            )
            self._capture_video_zoom_anchor(viewport_pos)

            # Apply zoom — _on_zoom_changed sets the new pixmap synchronously.
            self._mw.slider_zoom.setValue(new_zoom)

            self._restore_video_zoom_anchor()
            evt.accept()
        else:
            evt.ignore()

    def _capture_video_zoom_anchor(self, viewport_pos=None):
        """Store cursor-relative anchor before a zoom change."""
        viewport = self._mw.scroll.viewport()
        if viewport_pos is None:
            viewport_pos = viewport.rect().center()
        vp_x = max(0, min(viewport.width(), viewport_pos.x()))
        vp_y = max(0, min(viewport.height(), viewport_pos.y()))
        hbar = self._mw.scroll.horizontalScrollBar()
        vbar = self._mw.scroll.verticalScrollBar()
        label_w = max(self._mw.video_label.width(), 1)
        label_h = max(self._mw.video_label.height(), 1)
        # offset from scroll-area origin to label origin (when label is centered)
        offset_x = max((viewport.width() - label_w) // 2, 0)
        offset_y = max((viewport.height() - label_h) // 2, 0)
        local_x = hbar.value() + vp_x - offset_x
        local_y = vbar.value() + vp_y - offset_y
        local_x = max(0, min(label_w, local_x))
        local_y = max(0, min(label_h, local_y))
        self._mw._video_zoom_anchor = (
            vp_x,
            vp_y,
            local_x / label_w,
            local_y / label_h,
        )

    def _restore_video_zoom_anchor(self):
        """Restore scrollbars so the anchored image point stays under the cursor."""
        anchor = getattr(self._mw, "_video_zoom_anchor", None)
        if anchor is None:
            return
        self._mw._video_zoom_anchor = None
        vp_x, vp_y, rel_x, rel_y = anchor
        viewport = self._mw.scroll.viewport()
        label_w = max(self._mw.video_label.width(), 1)
        label_h = max(self._mw.video_label.height(), 1)
        offset_x = max((viewport.width() - label_w) // 2, 0)
        offset_y = max((viewport.height() - label_h) // 2, 0)
        target_x = int(round(rel_x * label_w - vp_x + offset_x))
        target_y = int(round(rel_y * label_h - vp_y + offset_y))
        hbar = self._mw.scroll.horizontalScrollBar()
        vbar = self._mw.scroll.verticalScrollBar()
        hbar.setValue(max(0, min(hbar.maximum(), target_x)))
        vbar.setValue(max(0, min(vbar.maximum(), target_y)))

    def _handle_video_event(self, evt):
        """Handle video events including pinch gestures."""
        from PySide6.QtWidgets import QWidget

        if evt.type() == QEvent.Gesture:
            if not self._mw._video_interactions_enabled:
                evt.ignore()
                return False
            return self._mw._handle_gesture_event(evt)
        return QWidget.event(self._mw.video_label, evt)

    def _handle_gesture_event(self, evt):
        """Handle pinch-to-zoom gesture."""
        if not self._mw._video_interactions_enabled:
            return False
        if self._mw.roi_selection_active:
            return False
        gesture = evt.gesture(Qt.PinchGesture)
        if gesture:
            if gesture.state() == Qt.GestureUpdated:
                scale_factor = gesture.scaleFactor()
                current_zoom = self._mw.slider_zoom.value()
                zoom_delta = int((scale_factor - 1.0) * 50)
                new_zoom = max(10, min(400, current_zoom + zoom_delta))
                self._mw.slider_zoom.setValue(new_zoom)
            return True
        return False

    def _display_roi_with_zoom(self):
        """Apply the current zoom to the canvas; the overlay follows."""
        self._mw.video_label.set_zoom(max(self._mw.slider_zoom.value() / 100.0, 0.1))

    def _fit_image_to_screen(self):
        """Fit the image to the available screen space."""
        tracking_active = (
            self._mw.tracking_worker is not None
            and self._mw.tracking_worker.isRunning()
        )
        if tracking_active and self._mw._tracking_frame_size is not None:
            effective_width, effective_height = self._mw._tracking_frame_size
        elif self._mw.detection_test_result is not None:
            if self._mw.preview_frame_original is not None:
                h, w = self._mw.preview_frame_original.shape[:2]
                resize_factor = self._panels.setup.spin_resize.value()
                effective_width = int(w * resize_factor)
                effective_height = int(h * resize_factor)
            else:
                return
        elif self._mw.preview_frame_original is not None:
            h, w = self._mw.preview_frame_original.shape[:2]
            effective_width = w
            effective_height = h
        elif self._mw.roi_base_frame is not None:
            effective_width = self._mw.roi_base_frame.width()
            effective_height = self._mw.roi_base_frame.height()
        else:
            return

        viewport_width = self._mw.scroll.viewport().width()
        viewport_height = self._mw.scroll.viewport().height()
        zoom_w = viewport_width / effective_width
        zoom_h = viewport_height / effective_height
        zoom_fit = min(zoom_w, zoom_h) * 0.95
        zoom_fit = max(0.1, min(5.0, zoom_fit))
        self._mw.slider_zoom.setValue(int(zoom_fit * 100))
        self._mw.scroll.horizontalScrollBar().setValue(0)
        self._mw.scroll.verticalScrollBar().setValue(0)

    def add_roi_point(self, image_x: float, image_y: float) -> None:
        """Append an ROI point. Coordinates are already IMAGE coordinates.

        The canvas converts through its inverse transform, so this is valid at
        any zoom -- previously the raw label position was stored directly,
        which only agreed with image space at 100% zoom.
        """
        if not self._mw.roi_selection_active:
            return
        self._mw.roi_points.append((image_x, image_y))
        self.update_roi_preview()

    def remove_last_roi_point(self) -> None:
        """Drop the most recent in-progress point."""
        if not self._mw.roi_selection_active or not self._mw.roi_points:
            return
        removed = self._mw.roi_points.pop()
        logger.info(f"Undid last ROI point: ({removed[0]:.1f}, {removed[1]:.1f})")
        self.update_roi_preview()

    def set_current_arena(self, arena_id: int) -> None:
        """Make *arena_id* the arena new shapes join."""
        if self._mw.roi_selection_active:
            return
        self.current_arena_id = int(arena_id)
        self._panels.arena.set_current_arena(arena_id)
        self._mw.video_label.set_current_arena(arena_id)

    def update_roi_preview(self):
        """Push current shapes and in-progress points to the canvas.

        No rasterization here: the canvas paints the overlay in viewport space
        on its next paintEvent. The old implementation deep-copied and
        repainted the whole QImage on every click -- about 61 MB per click on
        a 4512x4512 frame.
        """
        canvas = self._mw.video_label
        canvas.set_shapes(self._mw.roi_shapes)
        canvas.set_points(self._mw.roi_points)
        canvas.set_current_arena(self.current_arena_id)
        canvas.set_drawing(self._mw.roi_selection_active)

        valid = False
        preview_shape = None
        if self._mw.roi_current_mode == "circle" and len(self._mw.roi_points) >= 3:
            circle_fit = fit_circle_to_points(self._mw.roi_points)
            if circle_fit:
                self._mw.roi_fitted_circle = circle_fit
                valid = True
                cx, cy, radius = circle_fit
                preview_shape = {"type": "circle", "params": (cx, cy, radius)}
        elif self._mw.roi_current_mode == "polygon" and len(self._mw.roi_points) >= 3:
            valid = True
            preview_shape = {"type": "polygon", "params": list(self._mw.roi_points)}
        canvas.set_preview_shape(preview_shape)
        self._panels.arena.set_shape_valid(valid)

    def _ensure_roi_base_frame(self) -> bool:
        """Load the first video frame into roi_base_frame if not already loaded.

        Returns False (and shows a warning dialog) if there is no video or the
        frame can't be read. Also syncs the arena panel's known frame size,
        since both the manual-drawing and grid-generation paths need it.
        """
        if not self._panels.setup.file_line.text():
            QMessageBox.warning(
                self._mw, "No Video", "Please select a video file first."
            )
            self._panels.arena.set_drawing_active(False)
            return False
        if self._mw.roi_base_frame is None:
            cap = cv2.VideoCapture(self._panels.setup.file_line.text())
            if not cap.isOpened():
                QMessageBox.warning(self._mw, "Error", "Cannot open video file.")
                self._panels.arena.set_drawing_active(False)
                return False
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.warning(self._mw, "Error", "Cannot read video frame.")
                self._panels.arena.set_drawing_active(False)
                return False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._mw.roi_base_frame = qt_image
        self._panels.arena.set_frame_size(
            self._mw.roi_base_frame.width(), self._mw.roi_base_frame.height()
        )
        return True

    def start_roi_selection(self):
        """Start an ROI shape selection session."""
        if not self._ensure_roi_base_frame():
            return

        # The canvas overlay is painted on top of whatever frame it already
        # holds; push the ROI base frame explicitly so drawing starts against
        # it rather than whatever was last shown (e.g. a stale preview frame).
        self._mw._set_video_pixmap(QPixmap.fromImage(self._mw.roi_base_frame))
        self._mw.video_label.set_zoom(max(self._mw.slider_zoom.value() / 100.0, 0.1))

        self._mw.roi_points = []
        self._mw.roi_fitted_circle = None
        self._mw.roi_selection_active = True
        self._mw.video_label.setCursor(Qt.CrossCursor)

        zone_type = (
            "INCLUSION" if self._mw.roi_current_zone_type == "include" else "EXCLUSION"
        )
        if self._mw.roi_current_mode == "circle":
            self._mw.roi_status_label.setText(
                f"Click points on {zone_type.lower()} circle boundary"
            )
        else:
            self._mw.roi_status_label.setText(
                f"Click {zone_type.lower()} polygon vertices"
            )
        self.update_roi_preview()

    def finish_roi_selection(self):
        """Finalize the current ROI shape and add it to the shape list."""
        if not self._mw.roi_base_frame:
            return
        fh, fw = self._mw.roi_base_frame.height(), self._mw.roi_base_frame.width()

        if self._mw.roi_current_mode == "circle":
            if not self._mw.roi_fitted_circle:
                QMessageBox.warning(
                    self._mw, "No ROI", "No valid circle fit available."
                )
                return
            cx, cy, radius = self._mw.roi_fitted_circle
            self._mw.roi_shapes.append(
                {
                    "type": "circle",
                    "params": (cx, cy, radius),
                    "mode": self._mw.roi_current_zone_type,
                    "arena_id": self.current_arena_id,
                }
            )
            zone_type = (
                "inclusion"
                if self._mw.roi_current_zone_type == "include"
                else "exclusion"
            )
            logger.info(
                f"Added circle {zone_type} zone: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}"
            )
        elif self._mw.roi_current_mode == "polygon":
            if len(self._mw.roi_points) < 3:
                QMessageBox.warning(
                    self._mw, "No ROI", "Need at least 3 points for polygon."
                )
                return
            self._mw.roi_shapes.append(
                {
                    "type": "polygon",
                    "params": list(self._mw.roi_points),
                    "mode": self._mw.roi_current_zone_type,
                    "arena_id": self.current_arena_id,
                }
            )
            zone_type = (
                "inclusion"
                if self._mw.roi_current_zone_type == "include"
                else "exclusion"
            )
            logger.info(
                f"Added polygon {zone_type} zone with {len(self._mw.roi_points)} vertices"
            )

        self._mw._generate_combined_roi_mask(fh, fw)
        self._mw.roi_points = []
        self._mw.roi_fitted_circle = None
        self._mw.roi_selection_active = False
        self._panels.arena.set_shapes(self._mw.roi_shapes)
        self._panels.arena.set_drawing_active(False)
        self._panels.arena.mark_hand_drawn()
        self.update_roi_preview()

        if hasattr(Qt, "OpenHandCursor"):
            self._mw.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self._mw.video_label.unsetCursor()

        include_count = sum(
            1 for s in self._mw.roi_shapes if s.get("mode", "include") == "include"
        )
        exclude_count = sum(
            1 for s in self._mw.roi_shapes if s.get("mode", "include") == "exclude"
        )
        # Display-only count over ALL shapes (include + exclude). This
        # intentionally differs from the engine's authoritative count,
        # `engine_params.n_arenas_from_shapes` (include-shapes only) -- if the
        # highest arena id appears only on an exclude shape, this label shows
        # one more arena than the engine will actually use. Not unified with
        # the engine count because that would change engine behavior; this is
        # a status label, not a config value.
        arena_count = len({int(s.get("arena_id", 0)) for s in self._mw.roi_shapes})
        arena_note = (
            f", arena {self.current_arena_id + 1} ({arena_count} arena(s) total)"
            if arena_count > 1
            else ""
        )
        self._mw.roi_status_label.setText(
            f"Active ROI: {include_count} inclusion, {exclude_count} exclusion zone(s)"
            f"{arena_note}"
        )
        self._mw._update_roi_optimization_info()
        self._mw._update_animals_per_arena_total_label()

        if self._mw.roi_base_frame:
            QTimer.singleShot(50, self._mw._display_roi_with_zoom)

    def _generate_combined_roi_mask(self, height, width):
        """Generate a combined mask from all ROI shapes with inclusion/exclusion support."""
        if not self._mw.roi_shapes:
            self._mw.roi_mask = None
            return
        combined_mask = np.zeros((height, width), np.uint8)
        for shape in self._mw.roi_shapes:
            if shape.get("mode", "include") == "include":
                if shape["type"] == "circle":
                    cx, cy, radius = shape["params"]
                    cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 255, -1)
                elif shape["type"] == "polygon":
                    pts = np.array(shape["params"], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [pts], 255)
        for shape in self._mw.roi_shapes:
            if shape.get("mode", "include") == "exclude":
                if shape["type"] == "circle":
                    cx, cy, radius = shape["params"]
                    cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 0, -1)
                elif shape["type"] == "polygon":
                    pts = np.array(shape["params"], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [pts], 0)
        self._mw.roi_mask = combined_mask
        logger.info(
            f"Generated combined ROI mask from {len(self._mw.roi_shapes)} shape(s)"
        )
        self._mw._invalidate_roi_cache()

    def undo_last_roi_shape(self):
        """Remove the most recently added zone in the CURRENT arena only.

        Scoped so Undo can never silently remove a different arena's shape
        just because it happened to be the list's last entry overall.
        """
        current_id = self.current_arena_id
        removed = None
        for index in range(len(self._mw.roi_shapes) - 1, -1, -1):
            if int(self._mw.roi_shapes[index].get("arena_id", 0)) == current_id:
                removed = self._mw.roi_shapes.pop(index)
                break
        if removed is None:
            return
        logger.info(f"Removed last ROI shape: {removed['type']}")
        if self._mw.roi_base_frame:
            fh, fw = self._mw.roi_base_frame.height(), self._mw.roi_base_frame.width()
            self._mw._generate_combined_roi_mask(fh, fw)
        else:
            self._mw.roi_mask = None
        self._panels.arena.set_shapes(self._mw.roi_shapes)
        self._panels.arena.mark_hand_drawn()
        if self._mw.roi_shapes:
            num_shapes = len(self._mw.roi_shapes)
            shape_summary = ", ".join([s["type"] for s in self._mw.roi_shapes])
            self._mw.roi_status_label.setText(
                f"Active ROI: {num_shapes} shape(s) ({shape_summary})"
            )
        else:
            self._mw.roi_status_label.setText("No ROI")
        self._mw._update_animals_per_arena_total_label()
        # The base frame is already on the canvas (pushed once in
        # start_roi_selection); this just refreshes the shape/point overlay.
        self.update_roi_preview()

    def clear_roi(self):
        """Clear all ROI shapes and reset state."""
        self._mw.roi_mask = None
        self._mw.roi_points = []
        self._mw.roi_fitted_circle = None
        self._mw.roi_shapes = []
        self._mw.roi_selection_active = False
        self._mw.roi_base_frame = None
        self.current_arena_id = 0
        self._panels.arena.set_drawing_active(False)
        self._panels.arena.set_shapes([])
        self._mw.roi_status_label.setText("No ROI")
        self._mw.video_label.show_toast("ROI Cleared")
        self.update_roi_preview()
        if hasattr(Qt, "OpenHandCursor"):
            self._mw.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self._mw.video_label.unsetCursor()
        self._mw._update_animals_per_arena_total_label()
        logger.info("All ROI shapes cleared")

    def cancel_roi_shape(self) -> None:
        """Cancel the in-progress shape only; already-committed arenas are untouched."""
        self._panels.arena.set_drawing_active(False)
        if not self._mw.roi_selection_active:
            return
        self._mw.roi_points = []
        self._mw.roi_fitted_circle = None
        self._mw.roi_selection_active = False
        if hasattr(Qt, "OpenHandCursor"):
            self._mw.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self._mw.video_label.unsetCursor()
        self.update_roi_preview()

    def keyPressEvent(self, event) -> None:
        """Handle key press events - cancel the in-progress shape on Escape."""
        if event.key() == Qt.Key_Escape and self._mw.roi_selection_active:
            self.cancel_roi_shape()
        else:
            from PySide6.QtWidgets import QMainWindow

            QMainWindow.keyPressEvent(self._mw, event)

    # =========================================================================
    # TOGGLE / VISUALIZATION MODE
    # =========================================================================

    def toggle_preview(self, checked):
        """Toggle preview mode on/off."""
        if checked:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Preview Mode")
            msg.setText(
                "Preview mode will run forward tracking only without saving configuration."
            )
            msg.setInformativeText(
                "Preview features:\n"
                "\u2022 Forward pass only (no backward tracking)\n"
                "\u2022 Configuration is NOT saved\n"
                "\u2022 No CSV output\n\n"
                "Use 'Run Full Tracking' to save results and config."
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Ok)
            if msg.exec() == QMessageBox.Ok:
                self._mw.start_tracking(preview_mode=True)
                self._mw.btn_preview.setText("Stop Preview")
                self._mw.btn_start.setEnabled(False)
            else:
                self._mw.btn_preview.setChecked(False)
        else:
            self._mw.stop_tracking()
            self._mw.btn_preview.setText("Preview Mode")
            self._mw.btn_start.setEnabled(True)

    def toggle_tracking(self, checked):
        """Toggle full tracking on/off."""
        if checked:
            if self._mw.btn_preview.isChecked():
                self._mw.btn_preview.setChecked(False)
                self._mw.btn_preview.setText("Preview Mode")
                self._mw.stop_tracking()
            self._mw.btn_start.setText("Stop Tracking")
            self._mw.btn_preview.setEnabled(False)
            self._mw.start_full()
            if not (self._mw.tracking_worker and self._mw.tracking_worker.isRunning()):
                self._mw.btn_start.blockSignals(True)
                self._mw.btn_start.setChecked(False)
                self._mw.btn_start.blockSignals(False)
                self._mw.btn_start.setText("Start Full Tracking")
                self._mw.btn_preview.setEnabled(True)
        else:
            self._mw.stop_tracking()

    def _on_visualization_mode_changed(self, state):
        """Handle visualization-free mode toggle."""
        is_viz_free = self._panels.setup.is_visualization_free()
        is_preview_active = self._mw.btn_preview.isChecked()
        is_tracking_active = (
            self._mw.tracking_worker and self._mw.tracking_worker.isRunning()
        )

        if is_tracking_active and is_viz_free and not is_preview_active:
            self._mw._stored_preview_text = (
                self._mw._video_placeholder_text
                if self._mw._video_is_placeholder_text
                else None
            )
            self._mw._set_video_message(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below.",
                color="#9a9a9a",
                font_size=14,
            )
            logger.info("Visualization-Free Mode enabled - Maximum speed processing")
        elif is_tracking_active and not is_viz_free:
            if (
                hasattr(self._mw, "_stored_preview_text")
                and self._mw._stored_preview_text
            ):
                self._mw._set_video_message(self._mw._stored_preview_text)
            elif not self._mw._video_is_placeholder_text:
                pass
            else:
                self._mw._show_video_logo_placeholder()
