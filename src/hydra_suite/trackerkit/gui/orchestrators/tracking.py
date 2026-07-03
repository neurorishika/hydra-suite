"""TrackingOrchestrator — run→merge→export→finalize lifecycle."""

from __future__ import annotations

import csv
import gc
import glob as _glob
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox

from hydra_suite.core.identity.properties.export import (
    DETECTED_HEADING_COLUMNS,
    build_pose_keypoint_labels,
)
from hydra_suite.runtime.compute_runtime import (
    derive_detection_runtime_settings,
    derive_pose_runtime_settings,
)
from hydra_suite.trackerkit.gui.orchestrators.config import _get_video_config_path
from hydra_suite.trackerkit.session_plan import resolve_video_plan
from hydra_suite.trackerkit.tracking_cache import plan_tracking_cache
from hydra_suite.utils.pose_visualization import (
    is_renderable_pose_keypoint,
    normalize_pose_render_min_conf,
)
from hydra_suite.utils.video_artifacts import candidate_artifact_base_dirs
from hydra_suite.utils.video_encoder import VideoEncoder

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)

RICH_EXPORT_SUFFIX = "_with_individual"
LEGACY_RICH_EXPORT_SUFFIX = "_with_pose"

# Preview Mode runs the full detection/tracking pipeline live (no cache-only
# fast path), so an unbounded frame range makes "Preview" as slow as a full
# run. Cap it to a fixed wall-clock duration of source video.
PREVIEW_MAX_DURATION_SECONDS = 300


def compute_capped_preview_range(
    start_frame: int,
    end_frame: int,
    fps: float,
    max_duration_seconds: int = PREVIEW_MAX_DURATION_SECONDS,
) -> tuple[int, bool]:
    """Return (clamped_end_frame, was_clamped) for a preview frame range.

    Clamps ``end_frame`` so the selected range covers at most
    ``max_duration_seconds`` of video at ``fps``, measured from ``start_frame``.
    """
    max_frames = max(1, int(round(fps * max_duration_seconds)))
    selected_frames = end_frame - start_frame + 1
    if selected_frames <= max_frames:
        return end_frame, False
    return start_frame + max_frames - 1, True


class TrackingOrchestrator:
    """Owns the tracking lifecycle: start, stop, merge, export, finalize."""

    def __init__(
        self, main_window: "MainWindow", config: "TrackerConfig", panels
    ) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels

    @staticmethod
    def _rich_export_path(final_csv_path: str, *, legacy: bool = False) -> str:
        """Return the rich-export CSV path next to *final_csv_path*."""
        base, ext = os.path.splitext(final_csv_path)
        suffix = LEGACY_RICH_EXPORT_SUFFIX if legacy else RICH_EXPORT_SUFFIX
        return f"{base}{suffix}{ext or '.csv'}"

    def _write_rich_export_csv(
        self, rich_df: pd.DataFrame, final_csv_path: str
    ) -> str | None:
        """Write the canonical rich export and remove any stale legacy alias."""
        rich_path = self._rich_export_path(final_csv_path)
        legacy_path = self._rich_export_path(final_csv_path, legacy=True)
        try:
            cleaned_df = self._drop_empty_rich_export_columns(rich_df)
            cleaned_df.to_csv(rich_path, index=False)
            if legacy_path != rich_path and os.path.exists(legacy_path):
                os.remove(legacy_path)
        except Exception:
            logger.exception("Failed to save rich export CSV to: %s", rich_path)
            return None

        logger.info("Rich trajectories saved to: %s", rich_path)
        if legacy_path != rich_path:
            logger.info("Legacy rich-export alias removed: %s", legacy_path)
        return rich_path

    @staticmethod
    def _drop_empty_rich_export_columns(rich_df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that carry no information in the current export."""
        keep_columns: list[str] = []
        for column in rich_df.columns:
            series = rich_df[column]
            if series.isna().all():
                continue
            non_null = series.dropna()
            if not non_null.empty and non_null.astype(str).str.strip().eq("").all():
                continue
            keep_columns.append(column)
        return rich_df.loc[:, keep_columns].copy()

    def _remove_legacy_rich_exports(self, final_csv_path: str) -> None:
        """Remove any stale rich-export CSV variants next to *final_csv_path*."""
        for legacy in (False, True):
            candidate = self._rich_export_path(final_csv_path, legacy=legacy)
            if not os.path.exists(candidate):
                continue
            try:
                os.remove(candidate)
            except Exception:
                logger.warning("Failed to remove stale rich-export CSV: %s", candidate)

    def start_full(self):
        """start_full method documentation."""
        if self._mw.btn_preview.isChecked():
            self._mw.btn_preview.setChecked(False)
            self._mw.btn_preview.setText("Preview Mode")
            self.stop_tracking()

        # Set up comprehensive session logging once for entire tracking session
        video_path = self._panels.setup.file_line.text()
        if video_path:
            self._mw._setup_session_logging(video_path, backward_mode=False)
            from datetime import datetime

            self._mw._individual_dataset_run_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            self._mw.current_detection_cache_path = None
            self._mw.current_individual_properties_cache_path = None
            self._mw.current_detected_properties_cache_path = None
            self._mw.current_detected_cnn_cache_paths = {}
            self._mw.current_interpolated_roi_npz_path = None
            self._mw.current_interpolated_pose_csv_path = None
            self._mw.current_interpolated_pose_df = None
            self._mw.current_interpolated_tag_csv_path = None
            self._mw.current_interpolated_tag_df = None
            self._mw.current_interpolated_cnn_csv_paths = {}
            self._mw.current_interpolated_cnn_dfs = {}
            self._mw.current_interpolated_headtail_csv_path = None
            self._mw.current_interpolated_headtail_df = None
            self._mw._pending_pose_export_csv_path = None
            self._mw._pending_video_csv_path = None
            self._mw._pending_video_generation = False
            self._mw._pending_finish_after_track_videos = False

        self.start_tracking(preview_mode=False)

    def _request_qthread_stop(
        self,
        worker,
        worker_name: str,
        *,
        timeout_ms: int = 1500,
        force_terminate: bool = True,
    ) -> None:
        """Stop a QThread cooperatively, then force terminate if needed."""
        if worker is None:
            return
        try:
            if not worker.isRunning():
                return
        except Exception:
            return

        try:
            if hasattr(worker, "stop"):
                worker.stop()
        except Exception:
            logger.debug("Failed to call stop() on %s", worker_name, exc_info=True)

        try:
            worker.requestInterruption()
        except Exception:
            pass

        stopped = False
        try:
            stopped = bool(worker.wait(int(timeout_ms)))
        except Exception:
            stopped = False

        if stopped:
            logger.info("%s stopped.", worker_name)
            return

        if not force_terminate:
            logger.warning(
                "%s did not stop within %d ms (cooperative stop only).",
                worker_name,
                int(timeout_ms),
            )
            return

        logger.warning(
            "%s did not stop cooperatively; forcing terminate().", worker_name
        )
        try:
            worker.terminate()
        except Exception:
            logger.debug("terminate() failed for %s", worker_name, exc_info=True)
        try:
            worker.wait(max(500, int(timeout_ms)))
        except Exception:
            pass

    def _stop_csv_writer(self, timeout_sec: float = 2.0) -> None:
        """Stop background CSV writer thread safely without indefinite blocking."""
        writer = self._mw.csv_writer_thread
        if writer is None:
            return
        try:
            writer.stop()
        except Exception:
            logger.debug("Failed to request CSV writer stop.", exc_info=True)
        try:
            if writer.is_alive():
                writer.join(timeout=timeout_sec)
                if writer.is_alive():
                    logger.warning("CSV writer did not stop within %.1fs.", timeout_sec)
        except Exception:
            logger.debug("Failed to join CSV writer thread.", exc_info=True)
        finally:
            self._mw.csv_writer_thread = None

    def _cleanup_thread_reference(self, attr_name: str) -> None:
        """Delete finished QThread references safely."""
        worker = getattr(self._mw, attr_name, None)
        if worker is None:
            return
        try:
            running = bool(worker.isRunning())
        except Exception:
            running = False
        if not running:
            try:
                worker.deleteLater()
            except Exception:
                pass
            setattr(self._mw, attr_name, None)

    def stop_tracking(self):
        """stop_tracking method documentation."""
        self._mw._stop_all_requested = True
        self._mw._pending_finish_after_interp = False
        self._mw._pending_finish_after_track_videos = False
        self._mw._pending_pose_export_csv_path = None
        self._mw._pending_video_csv_path = None
        self._mw._pending_video_generation = False

        # Stop all active workers and subprocess-like threads.
        self._request_qthread_stop(
            getattr(self._mw, "_cache_builder_worker", None),
            "DetectionCacheBuilderWorker",
        )
        self._request_qthread_stop(
            getattr(self._mw, "merge_worker", None), "MergeWorker", timeout_ms=1200
        )
        self._request_qthread_stop(
            getattr(self._mw, "postprocess_worker", None),
            "PostProcessWorker",
            timeout_ms=1200,
        )
        self._request_qthread_stop(self._mw.dataset_worker, "DatasetGenerationWorker")
        self._request_qthread_stop(self._mw.interp_worker, "InterpolatedCropsWorker")
        self._request_qthread_stop(
            self._mw.final_media_export_worker, "FinalMediaExportWorker"
        )
        self._request_qthread_stop(
            getattr(self._mw, "preview_detection_worker", None),
            "PreviewDetectionWorker",
            timeout_ms=1200,
        )
        self._request_qthread_stop(self._mw.tracking_worker, "TrackingWorker")
        self._stop_csv_writer()

        self._cleanup_thread_reference("_cache_builder_worker")
        self._cleanup_thread_reference("merge_worker")
        self._cleanup_thread_reference("postprocess_worker")
        self._cleanup_thread_reference("dataset_worker")
        self._cleanup_thread_reference("interp_worker")
        self._cleanup_thread_reference("final_media_export_worker")
        self._cleanup_thread_reference("preview_detection_worker")
        self._cleanup_thread_reference("tracking_worker")

        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Ready")
        self._mw._set_ui_controls_enabled(True)
        # Ensure UI state is restored after stopping
        if self._mw.current_video_path:
            self._mw._apply_ui_state("idle")
        else:
            self._mw._apply_ui_state("no_video")
        self._mw.btn_preview.setChecked(False)
        self._mw.btn_preview.setText("Preview Mode")
        self._mw.btn_start.blockSignals(True)
        self._mw.btn_start.setChecked(False)
        self._mw.btn_start.blockSignals(False)
        self._mw.btn_start.setText("Start Full Tracking")
        self._mw.btn_start.setEnabled(True)
        self._mw.btn_preview.setEnabled(True)
        self._mw._individual_dataset_run_id = None
        self._mw.current_detection_cache_path = None
        self._mw.current_individual_properties_cache_path = None
        self._mw.current_detected_properties_cache_path = None
        self._mw.current_detected_cnn_cache_paths = {}
        self._mw.current_interpolated_roi_npz_path = None
        self._mw.current_interpolated_pose_csv_path = None
        self._mw.current_interpolated_pose_df = None
        self._mw.current_interpolated_tag_csv_path = None
        self._mw.current_interpolated_tag_df = None
        self._mw.current_interpolated_cnn_csv_paths = {}
        self._mw.current_interpolated_cnn_dfs = {}
        self._mw.current_interpolated_headtail_csv_path = None
        self._mw.current_interpolated_headtail_df = None

        # Hide stats labels when tracking stops
        self._mw.label_current_fps.setVisible(False)
        self._mw.label_elapsed_time.setVisible(False)
        self._mw.label_eta.setVisible(False)

        # Reset tracking frame size
        self._mw._tracking_frame_size = None
        self._mw._cleanup_session_logging()

    def on_progress_update(self: object, percentage, status_text):
        """on_progress_update method documentation."""
        if self._mw._stop_all_requested:
            return
        self._mw.progress_bar.setValue(percentage)
        self._mw.progress_label.setText(status_text)

    def on_pose_exported_model_resolved(self, artifact_path: str) -> None:
        """Update pose exported-model UI/config when runtime resolves an artifact path."""
        if self._mw._stop_all_requested:
            return
        path = str(artifact_path or "").strip()
        if not path:
            return
        logger.info("Pose runtime resolved exported model artifact: %s", path)
        try:
            # Persist run metadata immediately.
            self._mw.save_config(prompt_if_exists=False)
        except Exception:
            logger.debug(
                "Failed to persist resolved pose runtime artifact metadata.",
                exc_info=True,
            )

    def on_tracking_warning(self, title, message):
        """Display tracking warnings in the UI."""
        if self._mw._stop_all_requested:
            return
        QMessageBox.information(self._mw, title, message)

    def show_gpu_info(self):
        """Display GPU and acceleration information dialog."""
        from hydra_suite.utils.gpu_utils import get_device_info

        info = get_device_info()

        # Build formatted message
        lines = ["<b>GPU & Acceleration Status</b><br>"]

        # CUDA
        cuda_status = "✓ Available" if info["cuda_available"] else "✗ Not Available"
        lines.append(f"<br><b>NVIDIA CUDA:</b> {cuda_status}")
        if info["cuda_available"] and info.get("cuda_device_count", 0) > 0:
            lines.append(f"&nbsp;&nbsp;• Devices: {info['cuda_device_count']}")
            if "cupy_version" in info:
                lines.append(f"&nbsp;&nbsp;• CuPy: {info['cupy_version']}")

        # TensorRT
        tensorrt_status = (
            "✓ Available"
            if info.get("tensorrt_available", False)
            else "✗ Not Available"
        )
        lines.append(f"<br><b>NVIDIA TensorRT:</b> {tensorrt_status}")
        if info.get("tensorrt_available", False):
            lines.append("&nbsp;&nbsp;• 2-5× faster YOLO inference")

        # MPS (Apple Silicon)
        mps_status = "✓ Available" if info["mps_available"] else "✗ Not Available"
        lines.append(f"<br><b>Apple MPS:</b> {mps_status}")
        if info.get("torch_available", False) and "torch_version" in info:
            lines.append(f"&nbsp;&nbsp;• PyTorch: {info['torch_version']}")

        # CPU Acceleration
        numba_status = "✓ Available" if info["numba_available"] else "✗ Not Available"
        lines.append(f"<br><b>CPU JIT (Numba):</b> {numba_status}")
        if info["numba_available"] and "numba_version" in info:
            lines.append(f"&nbsp;&nbsp;• Version: {info['numba_version']}")

        # Overall status
        lines.append("<br><b>Overall Status:</b>")
        if info["cuda_available"]:
            lines.append("&nbsp;&nbsp;• Using NVIDIA GPU acceleration")
        elif info["mps_available"]:
            lines.append("&nbsp;&nbsp;• Using Apple Silicon GPU acceleration")
        elif info["numba_available"]:
            lines.append("&nbsp;&nbsp;• Using CPU JIT compilation")
        else:
            lines.append("&nbsp;&nbsp;• Using NumPy (no acceleration)")

        message = "<br>".join(lines)

        # Create message box with rich text
        msg_box = QMessageBox(self._mw)
        msg_box.setWindowTitle("GPU & Acceleration Info")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec()

    @staticmethod
    def _iter_cache_artifact_paths(video_path: str, artifact_base_dirs) -> list[Path]:
        """Return current-video cache files for the given video."""
        stem = Path(video_path).stem.strip() or "video"
        patterns = (f"{stem}*_cache*.npz",)
        found: dict[str, Path] = {}

        for base_dir in artifact_base_dirs:
            base_path = Path(base_dir).expanduser()
            search_dirs = [base_path / f"{stem}_caches", base_path]
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for pattern in patterns:
                    for cache_path in search_dir.glob(pattern):
                        try:
                            key = str(cache_path.resolve())
                        except OSError:
                            key = str(cache_path)
                        found[key] = cache_path

        return sorted(found.values(), key=lambda path: path.name)

    @staticmethod
    def _iter_inference_cache_dirs(video_path: str, artifact_base_dirs) -> list[Path]:
        """Return InferenceRunner per-video cache directories for the given video.

        The InferenceRunner (yolo_obb path) stores its caches in a hidden
        ``.inference_cache_<stem>/`` directory next to the video (see
        ``TrackingWorker._resolve_cache_dir``). These hold ``detection.npz``,
        ``headtail.npz``, ``cnn_*.npz``, ``pose.npz``, ``apriltag.npz``. The
        file-glob in ``_iter_cache_artifact_paths`` never matches them, so they
        must be discovered and removed explicitly.
        """
        stem = Path(video_path).stem.strip() or "video"
        found: dict[str, Path] = {}
        for base_dir in artifact_base_dirs:
            cache_dir = Path(base_dir).expanduser() / f".inference_cache_{stem}"
            if cache_dir.is_dir():
                try:
                    key = str(cache_dir.resolve())
                except OSError:
                    key = str(cache_dir)
                found[key] = cache_dir
        return sorted(found.values(), key=lambda path: str(path))

    def clear_detection_caches(self) -> None:
        """Delete all current-video cache files for the active video."""
        if self._mw._has_active_progress_task():
            QMessageBox.warning(
                self._mw,
                "Tracking Busy",
                "Stop active tracking or cache-building tasks before clearing caches.",
            )
            return

        video_path = str(self._panels.setup.file_line.text() or "").strip()
        if not video_path:
            QMessageBox.information(
                self._mw,
                "No Video Loaded",
                "Load a video before clearing caches.",
            )
            return

        csv_dir = (
            os.path.dirname(self._panels.setup.csv_line.text())
            if hasattr(self._panels.setup, "csv_line")
            and self._panels.setup.csv_line.text()
            else ""
        )
        artifact_base_dirs = candidate_artifact_base_dirs(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        cache_paths = self._iter_cache_artifact_paths(video_path, artifact_base_dirs)
        inference_cache_dirs = self._iter_inference_cache_dirs(
            video_path, artifact_base_dirs
        )

        current_cache_path = str(
            getattr(self._mw, "current_detection_cache_path", "") or ""
        ).strip()
        current_props_cache_path = str(
            getattr(self._mw, "current_individual_properties_cache_path", "") or ""
        ).strip()
        if current_cache_path:
            current_cache = Path(current_cache_path).expanduser()
            if current_cache.exists() and current_cache not in cache_paths:
                cache_paths.append(current_cache)
        if current_props_cache_path:
            current_props_cache = Path(current_props_cache_path).expanduser()
            if current_props_cache.exists() and current_props_cache not in cache_paths:
                cache_paths.append(current_props_cache)

        if not cache_paths and not inference_cache_dirs:
            QMessageBox.information(
                self._mw,
                "No Caches Found",
                "No cache files were found for the current video.",
            )
            if (
                current_cache_path
                and not Path(current_cache_path).expanduser().exists()
            ):
                self._mw.current_detection_cache_path = None
            if (
                current_props_cache_path
                and not Path(current_props_cache_path).expanduser().exists()
            ):
                self._mw.current_individual_properties_cache_path = None
            return

        reply = QMessageBox.question(
            self._mw,
            "Clear All Caches",
            "Delete cache files for this video?\n\n"
            "This removes reusable detection, pose, AprilTag, classifier, and related cache artifacts and forces fresh cache generation on the next run.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        deleted = 0
        failed: list[str] = []
        removed_current_cache = False
        removed_current_props_cache = False
        for cache_path in cache_paths:
            try:
                cache_path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
            except Exception:
                failed.append(str(cache_path))
                continue

            if (
                current_cache_path
                and cache_path == Path(current_cache_path).expanduser()
            ):
                removed_current_cache = True
            if (
                current_props_cache_path
                and cache_path == Path(current_props_cache_path).expanduser()
            ):
                removed_current_props_cache = True

            try:
                cache_path.with_suffix(".autotune_state.json").unlink(missing_ok=True)
            except Exception:
                logger.debug(
                    "Failed to delete cache sidecar for %s",
                    cache_path,
                    exc_info=True,
                )
            try:
                cache_path.with_name(
                    cache_path.stem + "_confidence_regions.json"
                ).unlink(missing_ok=True)
            except Exception:
                logger.debug(
                    "Failed to delete confidence-region sidecar for %s",
                    cache_path,
                    exc_info=True,
                )

        deleted_dirs = 0
        for cache_dir in inference_cache_dirs:
            try:
                shutil.rmtree(cache_dir)
                deleted_dirs += 1
            except FileNotFoundError:
                pass
            except Exception:
                failed.append(str(cache_dir))

        if removed_current_cache or (
            current_cache_path and not Path(current_cache_path).expanduser().exists()
        ):
            self._mw.current_detection_cache_path = None
        if removed_current_props_cache or (
            current_props_cache_path
            and not Path(current_props_cache_path).expanduser().exists()
        ):
            self._mw.current_individual_properties_cache_path = None

        logger.info(
            "Cleared %d cache file(s) and %d inference cache dir(s) for %s%s",
            deleted,
            deleted_dirs,
            video_path,
            f"; failed={len(failed)}" if failed else "",
        )

        if failed:
            QMessageBox.warning(
                self._mw,
                "Cache Cleanup Incomplete",
                f"Deleted {deleted} cache file(s) and {deleted_dirs} inference "
                f"cache folder(s), but {len(failed)} item(s) could not be removed.",
            )
            return

        QMessageBox.information(
            self._mw,
            "Caches Cleared",
            f"Deleted {deleted} cache file(s) and {deleted_dirs} inference "
            f"cache folder(s) for the current video.",
        )

    def on_stats_update(self, stats):
        """Update real-time tracking statistics."""
        if self._mw._stop_all_requested:
            return
        phase = str(stats.get("phase", "tracking"))
        is_precompute = phase == "individual_precompute"

        # Update FPS
        if "fps" in stats:
            if is_precompute:
                self._mw.label_current_fps.setText(
                    f"Precompute Rate: {stats['fps']:.1f}/s"
                )
            else:
                self._mw.label_current_fps.setText(f"FPS: {stats['fps']:.1f}")
            self._mw.label_current_fps.setVisible(True)

        # Update elapsed time
        if "elapsed" in stats:
            elapsed_sec = stats["elapsed"]
            hours = int(elapsed_sec // 3600)
            minutes = int((elapsed_sec % 3600) // 60)
            seconds = int(elapsed_sec % 60)
            if hours > 0:
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                elapsed_str = f"{minutes:02d}:{seconds:02d}"
            if is_precompute:
                self._mw.label_elapsed_time.setText(
                    f"Precompute Elapsed: {elapsed_str}"
                )
            else:
                self._mw.label_elapsed_time.setText(f"Elapsed: {elapsed_str}")
            self._mw.label_elapsed_time.setVisible(True)

        # Update ETA
        if "eta" in stats:
            eta_sec = stats["eta"]
            if eta_sec > 0:
                hours = int(eta_sec // 3600)
                minutes = int((eta_sec % 3600) // 60)
                seconds = int(eta_sec % 60)
                if hours > 0:
                    eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    eta_str = f"{minutes:02d}:{seconds:02d}"
                if is_precompute:
                    self._mw.label_eta.setText(f"Precompute ETA: {eta_str}")
                else:
                    self._mw.label_eta.setText(f"ETA: {eta_str}")
            else:
                if is_precompute:
                    self._mw.label_eta.setText("Precompute ETA: calculating...")
                else:
                    self._mw.label_eta.setText("ETA: calculating...")
            self._mw.label_eta.setVisible(True)

    def on_new_frame(self, rgb):
        """on_new_frame method documentation."""
        z = max(self._mw.slider_zoom.value() / 100.0, 0.1)
        h, w, _ = rgb.shape

        # Store tracking frame size for fit-to-screen calculation
        self._mw._tracking_frame_size = (w, h)

        # Cache last frame so zoom changes can re-render from the current frame
        self._mw._last_tracking_frame_rgb = rgb

        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

        # ROI masking is now done in tracking worker - no need to duplicate here
        scaled = qimg.scaled(
            int(w * z), int(h * z), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._mw._set_video_pixmap(QPixmap.fromImage(scaled))

        # Auto-fit to screen on first frame of tracking
        if self._mw._tracking_first_frame:
            self._mw._tracking_first_frame = False
            # Use QTimer to ensure frame is displayed first
            from PySide6.QtCore import QTimer

            QTimer.singleShot(50, self._mw._fit_image_to_screen)

    def _scale_trajectories_to_original_space(self, trajectories_df, resize_factor):
        """Scale trajectory coordinates from resized space back to original video space."""
        if trajectories_df is None or trajectories_df.empty:
            return trajectories_df

        if resize_factor == 1.0:
            return trajectories_df  # No scaling needed

        # Scale factor to go from resized -> original is 1/resize_factor
        scale_factor = 1.0 / resize_factor

        logger.info(
            f"Scaling trajectories to original video space (resize_factor={resize_factor:.3f}, scale_factor={scale_factor:.3f})"
        )

        result_df = trajectories_df.copy()

        # Scale X, Y coordinates
        result_df["X"] = result_df["X"] * scale_factor
        result_df["Y"] = result_df["Y"] * scale_factor

        # Theta doesn't need scaling (it's an angle)
        # FrameID doesn't need scaling

        logger.info(
            f"Scaled {len(result_df)} trajectory points to original video coordinates"
        )
        return result_df

    def save_trajectories_to_csv(self: object, trajectories, output_path):
        """Save processed trajectories to CSV.

        Args:
            trajectories: Either list of tuples (old format) or pandas DataFrame (new format with confidence)
            output_path: Path to save CSV file
        """
        if trajectories is None:
            logger.warning("No post-processed trajectories to save (None).")
            return False

        # Check if input is a DataFrame (new format with confidence)
        if isinstance(trajectories, pd.DataFrame):
            if trajectories.empty:
                logger.warning(
                    "No post-processed trajectories to save (empty DataFrame)."
                )
                return False
            try:
                # DataFrame already has all columns including confidence metrics
                # Convert X and Y to integers where possible (non-NaN values)
                df_to_save = trajectories.copy()
                for col in ["X", "Y", "FrameID"]:
                    if col in df_to_save.columns:
                        # Convert to float first to handle any issues, then to Int64 (nullable integer)
                        df_to_save[col] = pd.to_numeric(
                            df_to_save[col], errors="coerce"
                        )
                        # Use Int64 dtype which supports NaN values
                        df_to_save[col] = df_to_save[col].round().astype("Int64")

                # Drop unwanted columns from raw tracking data
                unwanted_cols = ["TrackID", "Index"]
                df_to_save = df_to_save.drop(
                    columns=[col for col in unwanted_cols if col in df_to_save.columns],
                    errors="ignore",
                )

                # Reorder columns to put basic trajectory info first
                base_cols = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
                other_cols = [col for col in df_to_save.columns if col not in base_cols]
                ordered_cols = base_cols + other_cols
                df_to_save[ordered_cols].to_csv(output_path, index=False)
                logger.info(
                    f"Successfully saved {df_to_save['TrajectoryID'].nunique()} post-processed trajectories "
                    f"({len(df_to_save)} rows) with {len(ordered_cols)} columns to {output_path}"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to save processed trajectories to {output_path}: {e}"
                )
                return False

        # Old format (list of tuples) - for backward compatibility
        if not trajectories:
            logger.warning("No post-processed trajectories to save.")
            return False
        header = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for trajectory_id, segment in enumerate(trajectories):
                    for x, y, theta, frame_id in segment:
                        # Handle NaN values - write as empty string or keep as float
                        x_val = int(x) if not np.isnan(x) else ""
                        y_val = int(y) if not np.isnan(y) else ""
                        frame_val = int(frame_id) if not np.isnan(frame_id) else ""
                        writer.writerow([trajectory_id, x_val, y_val, theta, frame_val])
            logger.info(
                f"Successfully saved {len(trajectories)} post-processed trajectories to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save processed trajectories to {output_path}: {e}")
            return False

    def merge_and_save_trajectories(self):
        """merge_and_save_trajectories method documentation."""
        if self._mw._stop_all_requested:
            return
        logger.info("=" * 80)
        logger.info("Starting trajectory merging process...")
        logger.info("=" * 80)

        forward_trajs = getattr(self._mw, "forward_processed_trajs", None)
        backward_trajs = getattr(self._mw, "backward_processed_trajs", None)

        # Check if trajectories exist and are not empty (handle both DataFrame and list)
        forward_empty = (
            forward_trajs is None
            or (isinstance(forward_trajs, pd.DataFrame) and forward_trajs.empty)
            or (isinstance(forward_trajs, list) and len(forward_trajs) == 0)
        )
        backward_empty = (
            backward_trajs is None
            or (isinstance(backward_trajs, pd.DataFrame) and backward_trajs.empty)
            or (isinstance(backward_trajs, list) and len(backward_trajs) == 0)
        )

        if forward_empty or backward_empty:
            QMessageBox.warning(
                self._mw,
                "No Trajectories",
                "No forward or backward trajectories available to merge.",
            )
            return

        video_fp = self._panels.setup.file_line.text()
        if not video_fp:
            return
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        current_params = self._mw.get_parameters_dict()
        resize_factor = self._panels.setup.spin_resize.value()
        interp_method = (
            self._panels.postprocess.combo_interpolation_method.currentText().lower()
        )
        max_gap = max(
            1,
            round(
                self._panels.postprocess.spin_interpolation_max_gap.value()
                * self._panels.setup.spin_fps.value()
            ),
        )
        heading_flip_max_burst = (
            self._panels.postprocess.spin_heading_flip_max_burst.value()
        )

        # Show progress bar
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Merging trajectories...")

        # Create and start merge worker thread
        # Discover tag observation cache for AprilTag identity resolution
        _tag_cache_path = None
        if (
            bool(current_params.get("USE_APRILTAGS", False))
            or str(current_params.get("IDENTITY_METHOD", "")).lower() == "apriltags"
        ):
            _det_cache = getattr(self._mw, "current_detection_cache_path", None)
            if _det_cache and os.path.exists(str(_det_cache)):
                _pattern = str(_det_cache).replace(".npz", "") + "_tags_*.npz"
                _candidates = sorted(_glob.glob(_pattern))
                if _candidates:
                    _tag_cache_path = _candidates[-1]

        # Determine profiling settings for MergeWorker
        _enable_profiling = current_params.get("ENABLE_PROFILING", False)
        _merge_profile_path = None
        if _enable_profiling:
            _det_cache = getattr(self._mw, "current_detection_cache_path", None)
            if _det_cache:
                _merge_profile_path = str(
                    Path(_det_cache).parent / "merge_profile.json"
                )
            elif video_fp:
                _merge_profile_path = str(Path(video_fp).parent / "merge_profile.json")

        from hydra_suite.trackerkit.gui.workers.merge_worker import MergeWorker

        self._mw.merge_worker = MergeWorker(
            forward_trajs,
            backward_trajs,
            total_frames,
            current_params,
            resize_factor,
            interp_method,
            max_gap,
            tag_cache_path=_tag_cache_path,
            heading_flip_max_burst=heading_flip_max_burst,
            directed_heading_posthoc=bool(
                current_params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)
            ),
            enable_profiling=_enable_profiling,
            profile_export_path=_merge_profile_path,
        )
        self._mw.merge_worker.progress_signal.connect(self.on_merge_progress)
        self._mw.merge_worker.finished_signal.connect(self.on_merge_finished)
        self._mw.merge_worker.error_signal.connect(self.on_merge_error)
        self._mw.merge_worker.start()

    def on_merge_progress(self, value, message):
        """Update progress bar during merge."""
        if self._mw._stop_all_requested:
            return
        sender = None
        if (
            sender is not None
            and self._mw.merge_worker is not None
            and sender is not self._mw.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._mw.progress_bar.setValue(value)
        self._mw.progress_label.setText(message)

    def _store_interpolated_pose_result(self, pose_csv_path, pose_rows):
        """Store interpolated pose results from CSV path or in-memory rows."""
        if pose_csv_path:
            self._mw.current_interpolated_pose_csv_path = pose_csv_path
            self._mw.current_interpolated_pose_df = None
            logger.info(f"Interpolated pose CSV saved: {pose_csv_path}")
        elif pose_rows:
            try:
                self._mw.current_interpolated_pose_df = pd.DataFrame(pose_rows)
                self._mw.current_interpolated_pose_csv_path = None
                logger.info(
                    "Interpolated pose rows kept in-memory: %d",
                    len(self._mw.current_interpolated_pose_df),
                )
            except Exception:
                self._mw.current_interpolated_pose_df = None

    def _store_interpolated_tag_result(self, tag_csv_path, tag_rows):
        """Store interpolated AprilTag results from CSV path or in-memory rows."""
        if tag_csv_path:
            self._mw.current_interpolated_tag_csv_path = tag_csv_path
            self._mw.current_interpolated_tag_df = None
            logger.info(f"Interpolated tag CSV saved: {tag_csv_path}")
        elif tag_rows:
            try:
                self._mw.current_interpolated_tag_df = pd.DataFrame(tag_rows)
                self._mw.current_interpolated_tag_csv_path = None
            except Exception:
                self._mw.current_interpolated_tag_df = None

    def _store_interpolated_cnn_result(self, cnn_csv_paths, cnn_rows):
        """Store interpolated CNN identity results from CSV paths or in-memory rows."""
        if cnn_csv_paths:
            self._mw.current_interpolated_cnn_csv_paths = cnn_csv_paths
            self._mw.current_interpolated_cnn_dfs = {}
            logger.info(f"Interpolated CNN CSVs: {cnn_csv_paths}")
        elif cnn_rows:
            try:
                self._mw.current_interpolated_cnn_dfs = {
                    label: pd.DataFrame(rows)
                    for label, rows in cnn_rows.items()
                    if rows
                }
                self._mw.current_interpolated_cnn_csv_paths = {}
            except Exception:
                self._mw.current_interpolated_cnn_dfs = {}

    def _store_interpolated_headtail_result(self, headtail_csv_path, headtail_rows):
        """Store interpolated head-tail results from CSV path or in-memory rows."""
        if headtail_csv_path:
            self._mw.current_interpolated_headtail_csv_path = headtail_csv_path
            self._mw.current_interpolated_headtail_df = None
            logger.info(f"Interpolated head-tail CSV saved: {headtail_csv_path}")
        elif headtail_rows:
            try:
                self._mw.current_interpolated_headtail_df = pd.DataFrame(headtail_rows)
                self._mw.current_interpolated_headtail_csv_path = None
            except Exception:
                self._mw.current_interpolated_headtail_df = None

    def _log_interpolated_postpass_summary(self, result, saved, gaps):
        """Log what the interpolated post-pass actually produced."""
        try:
            occluded_rows = int(result.get("occluded_rows", 0))
            interp_runs = int(result.get("interp_runs", 0))
            eligible_frames = int(result.get("eligible_frames", 0))
            eligible_rows = int(result.get("eligible_rows", 0))
            roi_rows_cached = int(result.get("roi_rows_cached", 0))
            pose_rows = int(result.get("pose_rows_produced", 0))
            tag_rows = int(result.get("tag_rows_produced", 0))
            cnn_rows = int(result.get("cnn_rows_produced", 0))
            headtail_rows = int(result.get("headtail_rows_produced", 0))
            no_work_reason = str(result.get("no_work_reason", "") or "").strip()
        except Exception:
            return

        if no_work_reason == "no_occluded_rows":
            logger.info(
                "Interpolated post-pass found no occluded rows; no interpolated analyses were needed."
            )
            return
        if no_work_reason == "no_eligible_gaps":
            logger.info(
                "Interpolated post-pass found %d occluded rows but no eligible bounded gaps; no interpolated analyses were produced.",
                occluded_rows,
            )
            return

        logger.info(
            "Interpolated post-pass summary: occluded rows=%d, bounded runs=%d, eligible rows=%d across %d frames, cached ROI rows=%d, pose rows=%d, tag rows=%d, CNN rows=%d, head-tail rows=%d, saved images=%d (gaps=%d)",
            occluded_rows,
            interp_runs,
            eligible_rows,
            eligible_frames,
            roi_rows_cached,
            pose_rows,
            tag_rows,
            cnn_rows,
            headtail_rows,
            saved,
            gaps,
        )

    @staticmethod
    def _count_augmented_pose_rows(with_pose_df):
        pose_cols = [col for col in with_pose_df.columns if str(col).startswith("Pose")]
        if not pose_cols:
            return 0, 0
        pose_present = with_pose_df[pose_cols].notna().any(axis=1)
        detection_present = pd.to_numeric(
            with_pose_df.get("DetectionID"), errors="coerce"
        ).notna()
        detection_rows = int((detection_present & pose_present).sum())
        interpolated_rows = int(((~detection_present) & pose_present).sum())
        return detection_rows, interpolated_rows

    @staticmethod
    def _count_interpolated_cnn_rows(with_pose_df):
        labels = []
        for col in with_pose_df.columns:
            match = re.match(r"^CNN_(.+)_Class$", str(col))
            if match:
                labels.append(match.group(1))
        labels = sorted(set(labels))
        parts = []
        for label in labels:
            class_col = f"CNN_{label}_Class"
            conf_col = f"CNN_{label}_Conf"
            present = pd.Series(False, index=with_pose_df.index)
            if class_col in with_pose_df.columns:
                present = present | with_pose_df[class_col].fillna("").astype(str).ne(
                    ""
                )
            if conf_col in with_pose_df.columns:
                present = present | with_pose_df[conf_col].notna()
            count = int(present.sum())
            if count > 0:
                parts.append(f"{label}={count}")
        return ", ".join(parts) if parts else "none"

    def _log_rich_export_summary(self, df: pd.DataFrame) -> None:
        """Log a structured per-source fill-rate summary for the rich export CSV."""
        total = len(df)
        if total == 0:
            logger.info("Rich export summary: 0 rows — nothing to summarize.")
            return

        def fill(col: str) -> int:
            return int(df[col].notna().sum()) if col in df.columns else 0

        def fill_any(cols: list) -> int:
            present = [c for c in cols if c in df.columns]
            if not present:
                return 0
            return int(df[present].notna().any(axis=1).sum())

        def pct(n: int) -> str:
            return f"{100.0 * n / total:.1f}%" if total > 0 else "—"

        lines = [f"Rich export summary — {total:,} rows"]

        # --- pose (detection-keyed vs interpolated) ---
        pose_cols = [c for c in df.columns if str(c).startswith("Pose")]
        kpt_x_cols = [
            c
            for c in df.columns
            if str(c).startswith("PoseKpt_") and str(c).endswith("_X")
        ]
        if pose_cols:
            det_present = pd.to_numeric(df.get("DetectionID"), errors="coerce").notna()
            pose_any = df[pose_cols].notna().any(axis=1)
            det_pose = int((det_present & pose_any).sum())
            interp_pose = int((~det_present & pose_any).sum())
            lines.append(
                f"  Pose (detection-keyed)   : {det_pose:>6,} / {total:,}  ({pct(det_pose)})"
            )
            if interp_pose:
                lines.append(
                    f"  Pose (interpolated)      : {interp_pose:>6,} / {total:,}  ({pct(interp_pose)})"
                )

        # --- detected heading ---
        heading_cols = [c for c in DETECTED_HEADING_COLUMNS if c in df.columns]
        if heading_cols:
            h_fill = fill_any(heading_cols)
            lines.append(
                f"  Detected heading         : {h_fill:>6,} / {total:,}  ({pct(h_fill)})"
            )

        # --- detected CNN per label ---
        cnn_class_cols = [c for c in df.columns if re.match(r"^CNN_.+_Class$", str(c))]
        cnn_labels = sorted(
            {re.match(r"^CNN_(.+)_Class$", str(c)).group(1) for c in cnn_class_cols}
        )
        for lbl in cnn_labels:
            n = fill_any([f"CNN_{lbl}_Class", f"CNN_{lbl}_Conf"])
            lines.append(
                f"  CNN [{lbl}]               : {n:>6,} / {total:,}  ({pct(n)})"
            )

        # --- interpolated AprilTag ---
        if "InterpTagID" in df.columns:
            n = fill("InterpTagID")
            if n:
                lines.append(
                    f"  AprilTag (interpolated)  : {n:>6,} / {total:,}  ({pct(n)})"
                )

        # --- interpolated head-tail ---
        if "InterpHeadingRad" in df.columns:
            n = fill("InterpHeadingRad")
            if n:
                lines.append(
                    f"  Head-tail (interpolated) : {n:>6,} / {total:,}  ({pct(n)})"
                )

        # --- per-keypoint fill rates (grouped 4 per line) ---
        if kpt_x_cols:
            _kpt_re = re.compile(r"^PoseKpt_(.+)_X$")
            kpt_entries = []
            for col in kpt_x_cols:
                m = _kpt_re.match(str(col))
                if m:
                    kpt_entries.append(f"{m.group(1)}: {pct(fill(col))}")
            if kpt_entries:
                lines.append("  Per-keypoint fill:")
                for i in range(0, len(kpt_entries), 4):
                    lines.append(
                        "    " + "   ".join(f"{s:<22}" for s in kpt_entries[i : i + 4])
                    )

        # --- trajectory count ---
        if "TrajectoryID" in df.columns:
            n_tracks = int(df["TrajectoryID"].nunique())
            lines.append(f"  Unique trajectories      : {n_tracks:,}")

        logger.info("\n".join(lines))

    def _on_interpolated_crops_finished(self, result):
        sender = None
        if (
            sender is not None
            and self._mw.interp_worker is not None
            and sender is not self._mw.interp_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._mw._stop_all_requested:
            self._cleanup_thread_reference("interp_worker")
            self._mw._refresh_progress_visibility()
            return

        saved = 0
        gaps = 0
        try:
            saved = int(result.get("saved", 0))
            gaps = int(result.get("gaps", 0))
        except Exception:
            pass

        self._mw._refresh_progress_visibility()
        logger.info(f"Interpolated individual crops saved: {saved} (gaps: {gaps})")
        self._log_interpolated_postpass_summary(result, saved, gaps)

        mapping_path = result.get("mapping_path")
        if mapping_path:
            logger.info(f"Interpolated mapping saved: {mapping_path}")

        roi_csv_path = result.get("roi_csv_path")
        if roi_csv_path:
            logger.info(f"Interpolated ROIs CSV saved: {roi_csv_path}")

        roi_npz_path = result.get("roi_npz_path")
        if roi_npz_path:
            self._mw.current_interpolated_roi_npz_path = roi_npz_path
            logger.info(f"Interpolated ROIs cache saved: {roi_npz_path}")

        self._store_interpolated_pose_result(
            result.get("pose_csv_path"), result.get("pose_rows")
        )
        self._store_interpolated_tag_result(
            result.get("tag_csv_path"), result.get("tag_rows")
        )
        self._store_interpolated_cnn_result(
            result.get("cnn_csv_paths"), result.get("cnn_rows")
        )
        self._store_interpolated_headtail_result(
            result.get("headtail_csv_path"), result.get("headtail_rows")
        )

        self._cleanup_thread_reference("interp_worker")
        self._mw._refresh_progress_visibility()

        if self._mw._pending_pose_export_csv_path:
            self._relink_and_export_rich_csv(self._mw._pending_pose_export_csv_path)

        if self._mw._pending_finish_after_interp:
            self._mw._pending_finish_after_interp = False
            if self._start_pending_final_media_export(self._mw._session_final_csv_path):
                return
            self._run_pending_video_generation_or_finalize()

    def _generate_final_media_export(self, final_csv_path):
        """Export final canonical stills and/or orientation-fixed videos."""
        from hydra_suite.trackerkit.gui.workers.video_worker import (
            FinalMediaExportWorker,
        )

        try:
            if self._mw._stop_all_requested:
                return False
            export_images = self._mw._is_individual_image_save_enabled()
            export_videos = self._mw._should_export_final_media_videos()
            if not export_images and not export_videos:
                return False
            if not final_csv_path or not os.path.exists(final_csv_path):
                return False
            if not self._mw.current_detection_cache_path or not os.path.exists(
                self._mw.current_detection_cache_path
            ):
                logger.warning(
                    "Skipping final canonical media export: no compatible detection cache is available."
                )
                return False

            image_root = (
                self._mw._resolve_current_individual_dataset_dir()
                if export_images
                else None
            )
            video_root = (
                self._mw._resolve_current_final_media_video_dir()
                if export_videos
                else None
            )
            if export_images and image_root is None:
                logger.warning(
                    "Skipping final canonical image export: no image output directory found."
                )
                export_images = False
            if export_videos and video_root is None:
                logger.warning(
                    "Skipping final media video export: no video output directory found."
                )
                export_videos = False
            if not export_images and not export_videos:
                return False

            export_root = video_root or image_root
            image_output_dir = (
                str((image_root / "images").expanduser()) if image_root else None
            )
            progress_text = "Generating final canonical media..."
            if export_images and not export_videos:
                progress_text = "Generating final canonical images..."
            elif export_videos and not export_images:
                progress_text = "Generating final media videos..."

            self._mw.progress_bar.setVisible(True)
            self._mw.progress_label.setVisible(True)
            self._mw.progress_bar.setValue(0)
            self._mw.progress_label.setText(progress_text)

            if (
                self._mw.final_media_export_worker is not None
                and self._mw.final_media_export_worker.isRunning()
            ):
                logger.warning(
                    "Final media video export already running; skipping duplicate request."
                )
                return True
            if (
                self._mw.final_media_export_worker is not None
                and not self._mw.final_media_export_worker.isRunning()
            ):
                self._mw.final_media_export_worker.deleteLater()
                self._mw.final_media_export_worker = None

            padding_fraction = (
                float(self._panels.identity.spin_individual_padding.value())
                if hasattr(self._mw, "_identity_panel")
                else 0.1
            )
            self._mw.final_media_export_worker = FinalMediaExportWorker(
                final_csv_path,
                str(export_root),
                image_output_dir,
                self._panels.setup.file_line.text().strip(),
                self._mw.current_detection_cache_path,
                self._mw.current_interpolated_roi_npz_path,
                self._mw._resolve_source_video_fps(),
                max(0.0, padding_fraction),
                tuple(int(c) for c in self._panels.identity._background_color),
                bool(
                    self._panels.dataset.chk_suppress_foreign_obb_oriented_videos.isChecked()
                    if export_videos
                    else self._panels.dataset.chk_suppress_foreign_obb_individual_dataset.isChecked()
                ),
                bool(
                    self._panels.dataset.chk_suppress_foreign_obb_individual_dataset.isChecked()
                ),
                bool(
                    self._panels.dataset.chk_suppress_foreign_obb_oriented_videos.isChecked()
                ),
                export_images,
                self._panels.dataset.spin_individual_interval.value(),
                self._panels.dataset.combo_individual_format.currentText(),
                export_videos,
                bool(
                    self._panels.dataset.chk_fix_oriented_video_direction_flips.isChecked()
                ),
                self._panels.dataset.spin_oriented_video_heading_flip_burst.value(),
                bool(
                    self._panels.dataset.chk_enable_oriented_video_affine_stabilization.isChecked()
                ),
                self._panels.dataset.spin_oriented_video_stabilization_window.value(),
                output_subdir="",
            )
            self._mw.final_media_export_worker.progress_signal.connect(
                self._mw.on_progress_update
            )
            self._mw.final_media_export_worker.finished_signal.connect(
                self._mw._on_final_media_export_finished
            )
            self._mw.final_media_export_worker.error_signal.connect(
                self._mw._on_final_media_export_error
            )
            self._mw.final_media_export_worker.finished.connect(
                self._mw._on_final_media_export_worker_thread_finished
            )
            self._mw.final_media_export_worker.start()
            return True
        except Exception as e:
            logger.warning(f"Final canonical media export failed to start: {e}")
            return False

    def _start_pending_final_media_export(self, final_csv_path) -> bool:
        """Start optional oriented track video export and hold the finish pipeline."""
        started = self._generate_final_media_export(final_csv_path)
        if started:
            self._mw._pending_finish_after_track_videos = True
        return started

    def _on_final_media_export_worker_thread_finished(self):
        """Release completed oriented track video worker safely."""
        sender = None
        if (
            sender is not None
            and self._mw.final_media_export_worker is not None
            and sender is not self._mw.final_media_export_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("final_media_export_worker")
        self._mw._refresh_progress_visibility()

    def _on_final_media_export_finished(self, result):
        """Handle completion of final canonical media export."""
        sender = None
        if (
            sender is not None
            and self._mw.final_media_export_worker is not None
            and sender is not self._mw.final_media_export_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._mw._stop_all_requested:
            self._cleanup_thread_reference("final_media_export_worker")
            self._mw._refresh_progress_visibility()
            return

        try:
            exported_videos = int(result.get("exported_videos", 0))
            exported_frames = int(result.get("exported_frames", 0))
            exported_tracks = int(result.get("exported_tracks", 0))
            exported_images = int(result.get("exported_images", 0))
            missing_rows = int(result.get("missing_rows", 0))
            missing_detected_rows = int(result.get("missing_detected_rows", 0))
            missing_interpolated_rows = int(result.get("missing_interpolated_rows", 0))
            invalid_geometry_rows = int(result.get("invalid_geometry_rows", 0))
            output_dir = str(result.get("output_dir", "")).strip()
            image_output_dir = str(result.get("image_output_dir", "")).strip()
        except Exception:
            exported_videos = 0
            exported_frames = 0
            exported_tracks = 0
            exported_images = 0
            missing_rows = 0
            missing_detected_rows = 0
            missing_interpolated_rows = 0
            invalid_geometry_rows = 0
            output_dir = ""
            image_output_dir = ""

        if output_dir:
            logger.info(
                "Final media export wrote videos to %s (%d/%d tracks, %d frames, missing rows=%d)",
                output_dir,
                exported_videos,
                exported_tracks,
                exported_frames,
                missing_rows,
            )
        if image_output_dir:
            logger.info(
                "Final media export wrote %d canonical images to %s",
                exported_images,
                image_output_dir,
            )
        if not output_dir and not image_output_dir:
            logger.info(
                "Final media export complete (%d videos, %d images across %d tracks, %d frames, missing rows=%d)",
                exported_videos,
                exported_images,
                exported_tracks,
                exported_frames,
                missing_rows,
            )
        if missing_rows:
            logger.info(
                "Final media export missing geometry breakdown: detected-cache=%d, interpolated-roi=%d, invalid-task=%d",
                missing_detected_rows,
                missing_interpolated_rows,
                invalid_geometry_rows,
            )

        self._cleanup_thread_reference("final_media_export_worker")
        self._mw._refresh_progress_visibility()

        if self._mw._pending_finish_after_track_videos:
            self._mw._pending_finish_after_track_videos = False
            self._run_pending_video_generation_or_finalize()

    def _on_final_media_export_error(self, error_message):
        """Handle final canonical media export errors without aborting the session."""
        sender = None
        if (
            sender is not None
            and self._mw.final_media_export_worker is not None
            and sender is not self._mw.final_media_export_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        logger.warning("Final canonical media export failed: %s", error_message)
        self._cleanup_thread_reference("final_media_export_worker")
        self._mw._refresh_progress_visibility()
        if self._mw._pending_finish_after_track_videos:
            self._mw._pending_finish_after_track_videos = False
            self._run_pending_video_generation_or_finalize()

    def on_merge_error(self, error_message):
        """Handle merge errors."""
        sender = None
        if (
            sender is not None
            and self._mw.merge_worker is not None
            and sender is not self._mw.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("merge_worker")
        if self._mw._stop_all_requested:
            self._mw._refresh_progress_visibility()
            return
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        QMessageBox.critical(
            self._mw,
            "Merge Error",
            f"Error during trajectory merging:\n{error_message}",
        )
        logger.error(f"Trajectory merge error: {error_message}")

    def on_merge_finished(self, resolved_trajectories):
        """Handle completion of trajectory merging."""
        sender = None
        if (
            sender is not None
            and self._mw.merge_worker is not None
            and sender is not self._mw.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("merge_worker")
        if self._mw._stop_all_requested:
            self._mw._refresh_progress_visibility()
            return
        self._mw.progress_label.setText("Saving merged trajectories...")

        raw_csv_path = self._panels.setup.csv_line.text()
        merged_csv_path = None
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            merged_csv_path = f"{base}_final.csv"
            if self.save_trajectories_to_csv(resolved_trajectories, merged_csv_path):
                # Track initial tracking CSV as temporary (only if cleanup enabled)
                if (
                    self._panels.postprocess.chk_cleanup_temp_files.isChecked()
                    and raw_csv_path not in self._mw.temporary_files
                ):
                    self._mw.temporary_files.append(raw_csv_path)
                logger.info(f"✓ Merged trajectory data saved to: {merged_csv_path}")

        # Complete session pipeline. Video generation is deferred to the very end
        # after pose export and interpolated individual analysis complete.
        self._finish_tracking_session(final_csv_path=merged_csv_path)

    def _get_video_draw_params(self, params, fps, trajectories_df):
        """Return drawing parameters derived from params, panel settings, and body size."""
        colors = params.get("TRAJECTORY_COLORS", [])
        reference_body_size = params.get("REFERENCE_BODY_SIZE", 30.0)
        show_labels = self._panels.postprocess.check_show_labels.isChecked()
        show_orientation = self._panels.postprocess.check_show_orientation.isChecked()
        show_trails = self._panels.postprocess.check_show_trails.isChecked()
        trail_duration_sec = self._panels.postprocess.spin_trail_duration.value()
        trail_duration_frames = int(trail_duration_sec * fps)
        marker_size = self._panels.postprocess.spin_marker_size.value()
        text_scale = self._panels.postprocess.spin_text_scale.value()
        arrow_length = self._panels.postprocess.spin_arrow_length.value()
        advanced_config = params.get("ADVANCED_CONFIG", {})
        marker_radius = int(marker_size * reference_body_size)
        arrow_len = int(arrow_length * reference_body_size)
        text_size = 0.5 * text_scale
        marker_thickness = max(2, int(0.15 * reference_body_size))
        pose_point_radius = int(
            max(
                1,
                advanced_config.get(
                    "video_pose_point_radius", max(2, marker_radius // 3)
                ),
            )
        )
        pose_point_thickness = int(
            advanced_config.get("video_pose_point_thickness", -1)
        )
        pose_line_thickness = int(
            max(1, advanced_config.get("video_pose_line_thickness", 2))
        )
        pose_color_mode = (
            str(advanced_config.get("video_pose_color_mode", "track")).strip().lower()
        )
        pose_fixed_color_raw = advanced_config.get("video_pose_color", [255, 255, 255])
        if (
            isinstance(pose_fixed_color_raw, (list, tuple))
            and len(pose_fixed_color_raw) == 3
        ):
            try:
                pose_fixed_color = tuple(
                    int(max(0, min(255, float(v)))) for v in pose_fixed_color_raw
                )
            except Exception:
                pose_fixed_color = (255, 255, 255)
        else:
            pose_fixed_color = (255, 255, 255)
        pose_min_conf = normalize_pose_render_min_conf(
            params.get("POSE_MIN_KPT_CONF_VALID", 0.2)
        )
        return dict(
            colors=colors,
            show_labels=show_labels,
            show_orientation=show_orientation,
            show_trails=show_trails,
            trail_duration_frames=trail_duration_frames,
            marker_radius=marker_radius,
            arrow_len=arrow_len,
            text_size=text_size,
            text_scale=text_scale,
            marker_thickness=marker_thickness,
            pose_point_radius=pose_point_radius,
            pose_point_thickness=pose_point_thickness,
            pose_line_thickness=pose_line_thickness,
            pose_color_mode=pose_color_mode,
            pose_fixed_color=pose_fixed_color,
            pose_min_conf=pose_min_conf,
            advanced_config=advanced_config,
        )

    def _get_pose_column_info(self, params, advanced_config, trajectories_df):
        """Return (pose_edges, pose_column_triplets, show_pose) for video rendering."""
        pose_edges = []
        pose_column_triplets = []
        show_pose = bool(advanced_config.get("video_show_pose", True))
        pose_col_pattern = re.compile(r"^PoseKpt_(.+)_(X|Y|Conf)$")
        pose_labels_available = {}
        for col in trajectories_df.columns:
            m = pose_col_pattern.match(str(col))
            if m is None:
                continue
            label = m.group(1)
            axis = m.group(2)
            pose_labels_available.setdefault(label, set()).add(axis)
        if not pose_labels_available:
            show_pose = False
        if show_pose:
            skeleton_names = []
            skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
            if skeleton_file and os.path.exists(skeleton_file):
                try:
                    with open(skeleton_file, "r", encoding="utf-8") as f:
                        skeleton_data = json.load(f)
                    names_raw = skeleton_data.get(
                        "keypoint_names", skeleton_data.get("keypoints", [])
                    )
                    skeleton_names = [str(n) for n in names_raw]
                    raw_edges = skeleton_data.get(
                        "skeleton_edges", skeleton_data.get("edges", [])
                    )
                    for edge in raw_edges:
                        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                            try:
                                pose_edges.append((int(edge[0]), int(edge[1])))
                            except Exception:
                                continue
                except Exception:
                    pose_edges = []
            ordered_labels = build_pose_keypoint_labels(
                skeleton_names, len(skeleton_names)
            )
            extras = sorted(
                [
                    lbl
                    for lbl in pose_labels_available.keys()
                    if lbl not in ordered_labels
                ]
            )
            ordered_labels.extend(extras)
            for label in ordered_labels:
                axes = pose_labels_available.get(label, set())
                if {"X", "Y", "Conf"}.issubset(axes):
                    pose_column_triplets.append(
                        (
                            f"PoseKpt_{label}_X",
                            f"PoseKpt_{label}_Y",
                            f"PoseKpt_{label}_Conf",
                        )
                    )
            if not pose_column_triplets:
                show_pose = False
        return pose_edges, pose_column_triplets, show_pose

    def _preextract_traj_arrays(
        self, trajectories_df, show_pose, pose_column_triplets, show_trails
    ):
        """Pre-extract trajectory arrays and index structures for O(1)/O(log N) lookups."""
        _frame_ids = trajectories_df["FrameID"].to_numpy(dtype=np.int32)
        _track_ids = trajectories_df["TrajectoryID"].to_numpy(dtype=np.int32)
        _xs = trajectories_df["X"].to_numpy(dtype=np.float64)
        _ys = trajectories_df["Y"].to_numpy(dtype=np.float64)
        _label_texts = self._build_video_track_label_array(trajectories_df)
        _thetas = (
            trajectories_df["Theta"].to_numpy(dtype=np.float64)
            if "Theta" in trajectories_df.columns
            else np.full(len(trajectories_df), np.nan)
        )
        _pose_kpts = None
        if show_pose and pose_column_triplets:
            _K = len(pose_column_triplets)
            _N = len(trajectories_df)
            _pose_kpts = np.full((_K, _N, 3), np.nan, dtype=np.float32)
            for _k, (_x_col, _y_col, _c_col) in enumerate(pose_column_triplets):
                if _x_col in trajectories_df.columns:
                    _pose_kpts[_k, :, 0] = trajectories_df[_x_col].to_numpy(
                        dtype=np.float32
                    )
                if _y_col in trajectories_df.columns:
                    _pose_kpts[_k, :, 1] = trajectories_df[_y_col].to_numpy(
                        dtype=np.float32
                    )
                if _c_col in trajectories_df.columns:
                    _pose_kpts[_k, :, 2] = trajectories_df[_c_col].to_numpy(
                        dtype=np.float32
                    )
        traj_indices_by_frame: dict = {}
        for _i in range(len(_frame_ids)):
            _fid = int(_frame_ids[_i])
            if _fid not in traj_indices_by_frame:
                traj_indices_by_frame[_fid] = []
            traj_indices_by_frame[_fid].append(_i)
        _track_sorted_row_indices: dict = {}
        _track_sorted_frame_vals: dict = {}
        if show_trails:
            _tmp_track: dict = {}
            for _i in range(len(_track_ids)):
                _tid = int(_track_ids[_i])
                if _tid not in _tmp_track:
                    _tmp_track[_tid] = []
                _tmp_track[_tid].append(_i)
            for _tid, _idxs in _tmp_track.items():
                _idx_arr = np.asarray(_idxs, dtype=np.int32)
                _order = np.argsort(_frame_ids[_idx_arr])
                _track_sorted_row_indices[_tid] = _idx_arr[_order]
                _track_sorted_frame_vals[_tid] = _frame_ids[_idx_arr[_order]]
        return (
            _frame_ids,
            _track_ids,
            _xs,
            _ys,
            _label_texts,
            _thetas,
            _pose_kpts,
            traj_indices_by_frame,
            _track_sorted_row_indices,
            _track_sorted_frame_vals,
        )

    def _format_video_track_label(self, track_id, unique_identity_key=None) -> str:
        """Return the overlay label for one rendered track row."""
        token = (
            str(unique_identity_key).strip() if unique_identity_key is not None else ""
        )
        if token and token.lower() != "nan":
            try:
                from hydra_suite.core.post.identity_postprocess import (
                    parse_identity_key,
                )

                parsed = parse_identity_key(token)
            except Exception:
                parsed = {}
            if parsed:
                compact_parts = []
                cnn_parts_by_label: dict[str, list[str]] = {}
                for source in sorted(parsed):
                    value = str(parsed[source]).strip()
                    if not value:
                        continue
                    if source == "apriltag":
                        compact_parts.append(f"Tag {value}")
                        continue
                    if source.startswith("cnn:"):
                        parts = source.split(":")
                        label = parts[1] if len(parts) >= 2 else source
                        compact_value = value
                        if len(parts) >= 3:
                            compact_value = value
                        elif "+" in value:
                            pieces = []
                            for item in value.split("+"):
                                item = str(item).strip()
                                if not item:
                                    continue
                                if ":" in item:
                                    item = str(item.split(":", 1)[1]).strip()
                                if item:
                                    pieces.append(item)
                            if pieces:
                                compact_value = " / ".join(pieces)
                        if compact_value:
                            cnn_parts_by_label.setdefault(label, []).append(
                                compact_value
                            )
                        continue
                    compact_parts.append(f"{source}={value}")
                for label in sorted(cnn_parts_by_label):
                    values = [value for value in cnn_parts_by_label[label] if value]
                    if not values:
                        continue
                    compact_parts.append(
                        values[0] if len(values) == 1 else " / ".join(values)
                    )
                if compact_parts:
                    return " | ".join(compact_parts)
            return token
        return f"ID{track_id}"

    def _build_video_track_label_array(self, trajectories_df):
        """Precompute one overlay label per row using stable identity when available.

        Checks identity columns in priority order (same as color key resolution):
        UniqueIdentityKey → IdentityAssignedLabel → IdentityOfflineLabel →
        IdentitySmoothedLabel.  Falls back to ``"ID{TrajectoryID}"`` when none
        are available for a row.
        """
        if trajectories_df is None or len(trajectories_df) == 0:
            return np.asarray([], dtype=object)

        identity_columns = [
            "UniqueIdentityKey",
            "IdentityAssignedLabel",
            "IdentityOfflineLabel",
            "IdentitySmoothedLabel",
        ]
        track_ids = trajectories_df["TrajectoryID"].tolist()
        labels = []
        for row_index, track_id in enumerate(track_ids):
            chosen_token = None
            for column in identity_columns:
                if column not in trajectories_df.columns:
                    continue
                token = self._normalize_video_identity_color_key(
                    trajectories_df.iloc[row_index][column]
                )
                if token:
                    chosen_token = token
                    break
            labels.append(self._format_video_track_label(track_id, chosen_token))
        return np.asarray(labels, dtype=object)

    def _normalize_video_identity_color_key(self, value):
        """Return a stable identity color key token or an empty string.

        Treats values that are missing, NaN, the bare word ``"unknown"``, or a
        source-keyed identity string whose every value is empty/``"unknown"``
        as having no identity — so the caller falls back to TrajectoryID for
        both labels and colors.
        """
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        token = str(value).strip()
        if not token or token.lower() == "nan":
            return ""
        if token.lower() == "unknown":
            return ""
        try:
            from hydra_suite.core.post.identity_postprocess import parse_identity_key

            parsed = parse_identity_key(token)
        except Exception:
            parsed = {}
        if parsed:
            informative_values = [
                str(v).strip()
                for v in parsed.values()
                if str(v).strip() and str(v).strip().lower() != "unknown"
            ]
            if not informative_values:
                return ""
        return token

    def _build_video_track_color_key_array(self, trajectories_df):
        """Precompute one color key per row, preferring identity evidence over TrajectoryID."""
        if trajectories_df is None or len(trajectories_df) == 0:
            return np.asarray([], dtype=object)

        identity_columns = [
            "UniqueIdentityKey",
            "IdentityAssignedLabel",
            "IdentityOfflineLabel",
            "IdentitySmoothedLabel",
        ]
        track_ids = trajectories_df["TrajectoryID"].tolist()
        color_keys = []
        for row_index, track_id in enumerate(track_ids):
            chosen_key = ""
            for column in identity_columns:
                if column not in trajectories_df.columns:
                    continue
                token = self._normalize_video_identity_color_key(
                    trajectories_df.iloc[row_index][column]
                )
                if token:
                    chosen_key = f"identity:{token}"
                    break
            if not chosen_key:
                chosen_key = f"trajectory:{int(track_id)}"
            color_keys.append(chosen_key)
        return np.asarray(color_keys, dtype=object)

    def _build_precomputed_color_palette(self, colors, _track_ids, color_keys):
        """Build per-row colors, reusing one color for rows with the same identity key."""
        _category20_colors = [
            (127, 127, 31),
            (188, 189, 34),
            (140, 86, 75),
            (255, 127, 14),
            (214, 39, 40),
            (255, 152, 150),
            (197, 176, 213),
            (148, 103, 189),
            (196, 156, 148),
            (227, 119, 194),
            (199, 199, 199),
            (140, 140, 140),
            (23, 190, 207),
            (158, 218, 229),
            (57, 59, 121),
            (82, 84, 163),
            (107, 110, 207),
            (156, 158, 222),
            (99, 121, 57),
            (140, 162, 82),
        ]
        _n_cat = len(_category20_colors)

        def _fallback_color(_track_id):
            _tid = int(_track_id)
            return (
                tuple(colors[_tid])
                if colors and _tid < len(colors)
                else _category20_colors[_tid % _n_cat]
            )

        _identity_palette = {}
        _next_identity_color_idx = 0
        _row_colors = []
        for _tid, _key in zip(_track_ids.tolist(), color_keys.tolist()):
            _key_token = str(_key)
            if _key_token.startswith("identity:"):
                if _key_token not in _identity_palette:
                    _identity_palette[_key_token] = (
                        tuple(colors[_next_identity_color_idx])
                        if colors and _next_identity_color_idx < len(colors)
                        else _category20_colors[_next_identity_color_idx % _n_cat]
                    )
                    _next_identity_color_idx += 1
                _row_colors.append(_identity_palette[_key_token])
                continue
            _row_colors.append(_fallback_color(_tid))
        return _row_colors

    def _draw_trail_for_track(
        self,
        frame,
        track_id,
        frame_idx,
        color,
        _xs,
        _ys,
        _track_sorted_frame_vals,
        _track_sorted_row_indices,
        trail_duration_frames,
        marker_thickness,
    ):
        """Draw the fading trail for a single track on the given frame."""
        if track_id not in _track_sorted_frame_vals:
            return
        _sfv = _track_sorted_frame_vals[track_id]
        _sri = _track_sorted_row_indices[track_id]
        _lo = int(np.searchsorted(_sfv, frame_idx - trail_duration_frames, side="left"))
        _hi = int(np.searchsorted(_sfv, frame_idx, side="left"))
        if _hi - _lo < 2:
            return
        _trail_xs = _xs[_sri[_lo:_hi]]
        _trail_ys = _ys[_sri[_lo:_hi]]
        _trail_fs = _sfv[_lo:_hi]
        _trail_lw = max(1, marker_thickness // 2)
        for _seg in range(_hi - _lo - 1):
            _px1, _py1 = _trail_xs[_seg], _trail_ys[_seg]
            _px2, _py2 = _trail_xs[_seg + 1], _trail_ys[_seg + 1]
            if np.isnan(_px1) or np.isnan(_py1) or np.isnan(_px2) or np.isnan(_py2):
                continue
            _age = frame_idx - int(_trail_fs[_seg])
            _alpha = 1.0 - (_age / trail_duration_frames)
            cv2.line(
                frame,
                (int(_px1), int(_py1)),
                (int(_px2), int(_py2)),
                (
                    int(color[0] * _alpha),
                    int(color[1] * _alpha),
                    int(color[2] * _alpha),
                ),
                _trail_lw,
            )

    def _draw_single_track_on_frame(
        self,
        frame,
        row_i,
        track_id,
        cx,
        cy,
        color,
        draw_p,
        _thetas,
        _pose_kpts,
        _label_texts,
        pose_edges,
    ):
        """Draw circle, label, orientation arrow, and pose for a single track."""
        marker_radius = draw_p["marker_radius"]
        marker_thickness = draw_p["marker_thickness"]
        cv2.circle(frame, (cx, cy), marker_radius, color, marker_thickness)
        if draw_p["show_labels"]:
            label_offset = int(marker_radius + 5)
            cv2.putText(
                frame,
                str(_label_texts[row_i]),
                (cx + label_offset, cy - label_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                draw_p["text_size"],
                color,
                max(1, int(draw_p["text_scale"] * 2)),
            )
        if draw_p["show_orientation"]:
            _theta = _thetas[row_i]
            if not np.isnan(_theta):
                cv2.arrowedLine(
                    frame,
                    (cx, cy),
                    (
                        int(cx + draw_p["arrow_len"] * np.cos(_theta)),
                        int(cy + draw_p["arrow_len"] * np.sin(_theta)),
                    ),
                    color,
                    marker_thickness,
                    tipLength=0.3,
                )
        if _pose_kpts is not None:
            kpts_arr = _pose_kpts[:, row_i, :]
            if np.any(np.isfinite(kpts_arr[:, 2])):
                pose_color = (
                    color
                    if draw_p["pose_color_mode"] == "track"
                    else draw_p["pose_fixed_color"]
                )
                if pose_edges:
                    for e0, e1 in pose_edges:
                        if (
                            e0 < 0
                            or e1 < 0
                            or e0 >= len(kpts_arr)
                            or e1 >= len(kpts_arr)
                        ):
                            continue
                        if not is_renderable_pose_keypoint(
                            kpts_arr[e0, 0],
                            kpts_arr[e0, 1],
                            kpts_arr[e0, 2],
                            draw_p["pose_min_conf"],
                        ) or not is_renderable_pose_keypoint(
                            kpts_arr[e1, 0],
                            kpts_arr[e1, 1],
                            kpts_arr[e1, 2],
                            draw_p["pose_min_conf"],
                        ):
                            continue
                        cv2.line(
                            frame,
                            (
                                int(round(float(kpts_arr[e0, 0]))),
                                int(round(float(kpts_arr[e0, 1]))),
                            ),
                            (
                                int(round(float(kpts_arr[e1, 0]))),
                                int(round(float(kpts_arr[e1, 1]))),
                            ),
                            pose_color,
                            draw_p["pose_line_thickness"],
                        )
                for kpt in kpts_arr:
                    if not is_renderable_pose_keypoint(
                        kpt[0], kpt[1], kpt[2], draw_p["pose_min_conf"]
                    ):
                        continue
                    cv2.circle(
                        frame,
                        (int(round(float(kpt[0]))), int(round(float(kpt[1])))),
                        draw_p["pose_point_radius"],
                        pose_color,
                        draw_p["pose_point_thickness"],
                    )

    def _render_annotated_video_frames(
        self,
        cap,
        out,
        start_frame,
        total_frames,
        draw_p,
        pose_edges,
        show_pose,
        arrays,
    ):
        """Write annotated frames from cap into out for the tracked frame range."""
        import queue as _queue
        import threading as _threading

        (
            _frame_ids,
            _track_ids,
            _xs,
            _ys,
            _label_texts,
            _thetas,
            _pose_kpts,
            traj_indices_by_frame,
            _track_sorted_row_indices,
            _track_sorted_frame_vals,
            _row_colors,
        ) = arrays
        _write_q: _queue.Queue = _queue.Queue(maxsize=4)

        def _writer_thread():
            while True:
                _item = _write_q.get()
                if _item is None:
                    break
                out.write(_item)

        _writer = _threading.Thread(target=_writer_thread, daemon=True)
        _writer.start()

        for rel_idx in range(total_frames):
            frame_idx = start_frame + rel_idx
            ret, frame = cap.read()
            if not ret:
                break

            frame_row_indices = traj_indices_by_frame.get(frame_idx, [])

            if draw_p["show_trails"]:
                for row_i in frame_row_indices:
                    track_id = int(_track_ids[row_i])
                    color = tuple(_row_colors[row_i])
                    self._draw_trail_for_track(
                        frame,
                        track_id,
                        frame_idx,
                        color,
                        _xs,
                        _ys,
                        _track_sorted_frame_vals,
                        _track_sorted_row_indices,
                        draw_p["trail_duration_frames"],
                        draw_p["marker_thickness"],
                    )

            for row_i in frame_row_indices:
                track_id = int(_track_ids[row_i])
                cx_f, cy_f = _xs[row_i], _ys[row_i]
                if np.isnan(cx_f) or np.isnan(cy_f):
                    continue
                cx, cy = int(cx_f), int(cy_f)
                color = tuple(_row_colors[row_i])
                self._draw_single_track_on_frame(
                    frame,
                    row_i,
                    track_id,
                    cx,
                    cy,
                    color,
                    draw_p,
                    _thetas,
                    _pose_kpts if show_pose else None,
                    _label_texts,
                    pose_edges,
                )

            _write_q.put(frame)

            if rel_idx % 30 == 0:
                progress = int(((rel_idx + 1) / total_frames) * 100)
                self._mw.progress_bar.setValue(progress)
                QApplication.processEvents()

        _write_q.put(None)
        _writer.join()

    def _open_video_cap_and_writer(self, video_path, output_path):
        """Open video capture and writer; return (cap, out, fps, total_video_frames) or None on error."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            out = VideoEncoder(
                output_path, fps=fps, width=frame_width, height=frame_height
            )
        except Exception:
            logger.error(f"Failed to create output video: {output_path}")
            cap.release()
            return None
        logger.info(f"Writing video: {frame_width}x{frame_height} @ {fps} FPS")
        return cap, out, fps, total_video_frames

    def _compute_video_frame_range(self, params, total_video_frames):
        """Return (start_frame, end_frame, total_frames) clamped to video bounds."""
        start_frame = int(params.get("START_FRAME", 0) or 0)
        end_frame = params.get("END_FRAME", None)
        if end_frame is None:
            end_frame = total_video_frames - 1 if total_video_frames > 0 else 0
        end_frame = int(end_frame)
        if total_video_frames > 0:
            start_frame = max(0, min(start_frame, total_video_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_video_frames - 1))
        total_frames = max(0, end_frame - start_frame + 1)
        logger.info(
            f"Exporting tracked frame range: {start_frame}-{end_frame} ({total_frames} frames)"
        )
        return start_frame, end_frame, total_frames

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
        logger.info("=" * 80)
        logger.info("Generating video from post-processed trajectories...")
        logger.info("=" * 80)

        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Generating video...")
        QApplication.processEvents()

        video_path = self._panels.setup.file_line.text()
        output_path = self._panels.postprocess.video_out_line.text()

        def _complete():
            if finalize_on_complete:
                self._finish_tracking_session(final_csv_path=csv_path)
            else:
                self._finalize_tracking_session_ui()

        if not video_path or not output_path:
            logger.error("Video input or output path not specified")
            _complete()
            return

        result = self._open_video_cap_and_writer(video_path, output_path)
        if result is None:
            _complete()
            return
        cap, out, fps, total_video_frames = result

        params = self._mw.get_parameters_dict()
        start_frame, end_frame, total_frames = self._compute_video_frame_range(
            params, total_video_frames
        )

        if total_frames <= 0:
            logger.error("Invalid frame range for video generation.")
            cap.release()
            out.release()
            _complete()
            return

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        draw_p = self._get_video_draw_params(params, fps, trajectories_df)
        pose_edges, pose_column_triplets, show_pose = self._get_pose_column_info(
            params, draw_p["advanced_config"], trajectories_df
        )
        (
            _frame_ids,
            _track_ids,
            _xs,
            _ys,
            _label_texts,
            _thetas,
            _pose_kpts,
            traj_indices_by_frame,
            _track_sorted_row_indices,
            _track_sorted_frame_vals,
        ) = self._preextract_traj_arrays(
            trajectories_df, show_pose, pose_column_triplets, draw_p["show_trails"]
        )
        _color_keys = self._build_video_track_color_key_array(trajectories_df)
        _row_colors = self._build_precomputed_color_palette(
            draw_p["colors"], _track_ids, _color_keys
        )

        arrays = (
            _frame_ids,
            _track_ids,
            _xs,
            _ys,
            _label_texts,
            _thetas,
            _pose_kpts,
            traj_indices_by_frame,
            _track_sorted_row_indices,
            _track_sorted_frame_vals,
            _row_colors,
        )
        self._render_annotated_video_frames(
            cap, out, start_frame, total_frames, draw_p, pose_edges, show_pose, arrays
        )

        cap.release()
        out.release()

        logger.info(f"✓ Video saved to: {output_path}")
        logger.info("=" * 80)

        _complete()

    def _handle_preview_mode_finished(self, finished_normally):
        """Reset UI and return True; caller should gc.collect() and return."""
        self._mw.btn_preview.setChecked(False)
        self._mw.btn_preview.setText("Preview Mode")
        self._mw.label_current_fps.setVisible(False)
        self._mw.label_elapsed_time.setVisible(False)
        self._mw.label_eta.setVisible(False)
        self._mw._set_ui_controls_enabled(True)
        self._mw.btn_start.blockSignals(True)
        self._mw.btn_start.setChecked(False)
        self._mw.btn_start.blockSignals(False)
        self._mw.btn_start.setText("Start Full Tracking")
        self._mw._apply_ui_state("idle" if self._mw.current_video_path else "no_video")
        if finished_normally:
            logger.info("Preview completed.")
        else:
            QMessageBox.warning(
                self._mw,
                "Preview Interrupted",
                "Preview was stopped or encountered an error.",
            )

    def _handle_forward_tracking_done(
        self, processed_trajectories, is_backward_enabled, fps_list
    ):
        """Save forward results, start backward pass or finalize forward-only session."""
        from hydra_suite.core.post.processing import interpolate_trajectories

        raw_csv_path = self._panels.setup.csv_line.text()
        processed_csv_path = None
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            forward_csv = f"{base}_forward{ext}"
            if (
                self._panels.postprocess.chk_cleanup_temp_files.isChecked()
                and forward_csv not in self._mw.temporary_files
            ):
                self._mw.temporary_files.append(forward_csv)

            processed_csv_path = f"{base}_forward_processed{ext}"
            if (
                is_backward_enabled
                and self._panels.postprocess.chk_cleanup_temp_files.isChecked()
                and processed_csv_path not in self._mw.temporary_files
            ):
                self._mw.temporary_files.append(processed_csv_path)

            self.save_trajectories_to_csv(processed_trajectories, processed_csv_path)

        if is_backward_enabled:
            self._mw.forward_processed_trajs = processed_trajectories
            if (
                isinstance(processed_trajectories, pd.DataFrame)
                and not processed_trajectories.empty
            ):
                logger.info(
                    f"Forward trajectories stored for merge: "
                    f"X range [{processed_trajectories['X'].min():.1f}, {processed_trajectories['X'].max():.1f}], "
                    f"Y range [{processed_trajectories['Y'].min():.1f}, {processed_trajectories['Y'].max():.1f}]"
                )
            self.start_backward_tracking()
        else:
            interp_method = (
                self._panels.postprocess.combo_interpolation_method.currentText().lower()
            )
            if interp_method != "none":
                max_gap = max(
                    1,
                    round(
                        self._panels.postprocess.spin_interpolation_max_gap.value()
                        * self._panels.setup.spin_fps.value()
                    ),
                )
                heading_flip_max_burst = (
                    self._panels.postprocess.spin_heading_flip_max_burst.value()
                )
                current_params = self._mw.get_parameters_dict()
                processed_trajectories = interpolate_trajectories(
                    processed_trajectories,
                    method=interp_method,
                    max_gap=max_gap,
                    heading_flip_max_burst=heading_flip_max_burst,
                    directed_heading_posthoc=bool(
                        current_params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)
                    ),
                )

            resize_factor = self._panels.setup.spin_resize.value()
            processed_trajectories = self._scale_trajectories_to_original_space(
                processed_trajectories, resize_factor
            )

            final_csv_path = None
            if raw_csv_path:
                base, ext = os.path.splitext(raw_csv_path)
                final_csv_path = f"{base}_forward_processed{ext}"
                self.save_trajectories_to_csv(processed_trajectories, final_csv_path)
                if (
                    self._panels.postprocess.chk_cleanup_temp_files.isChecked()
                    and raw_csv_path not in self._mw.temporary_files
                ):
                    self._mw.temporary_files.append(raw_csv_path)

            self._finish_tracking_session(final_csv_path=final_csv_path)

    def _handle_backward_tracking_done(self, processed_trajectories):
        """Save backward results and trigger merge or finalize session."""
        raw_csv_path = self._panels.setup.csv_line.text()
        processed_csv_path = None
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            backward_csv = f"{base}_backward{ext}"
            if (
                self._panels.postprocess.chk_cleanup_temp_files.isChecked()
                and backward_csv not in self._mw.temporary_files
            ):
                self._mw.temporary_files.append(backward_csv)

            processed_csv_path = f"{base}_backward_processed{ext}"
            if (
                self._panels.postprocess.chk_cleanup_temp_files.isChecked()
                and processed_csv_path not in self._mw.temporary_files
            ):
                self._mw.temporary_files.append(processed_csv_path)
            self.save_trajectories_to_csv(processed_trajectories, processed_csv_path)

        self._mw.backward_processed_trajs = processed_trajectories
        if (
            isinstance(processed_trajectories, pd.DataFrame)
            and not processed_trajectories.empty
        ):
            logger.info(
                f"Backward trajectories stored for merge: "
                f"X range [{processed_trajectories['X'].min():.1f}, {processed_trajectories['X'].max():.1f}], "
                f"Y range [{processed_trajectories['Y'].min():.1f}, {processed_trajectories['Y'].max():.1f}]"
            )

        has_forward = self._mw.forward_processed_trajs is not None and (
            isinstance(self._mw.forward_processed_trajs, pd.DataFrame)
            and not self._mw.forward_processed_trajs.empty
            or isinstance(self._mw.forward_processed_trajs, list)
            and len(self._mw.forward_processed_trajs) > 0
        )
        has_backward = self._mw.backward_processed_trajs is not None and (
            isinstance(self._mw.backward_processed_trajs, pd.DataFrame)
            and not self._mw.backward_processed_trajs.empty
            or isinstance(self._mw.backward_processed_trajs, list)
            and len(self._mw.backward_processed_trajs) > 0
        )

        if has_forward and has_backward:
            self.merge_and_save_trajectories()
        else:
            self._finish_tracking_session(final_csv_path=processed_csv_path)

    def _collect_worker_props_path(self):
        """Read export-relevant cache paths from tracking_worker and store them."""
        worker_props_path = ""
        worker_detected_props_path = ""
        worker_detected_cnn_paths = {}
        if self._mw.tracking_worker is not None:
            worker_props_path = str(
                getattr(
                    self._mw.tracking_worker, "individual_properties_cache_path", ""
                )
                or ""
            ).strip()
            worker_detected_props_path = str(
                getattr(self._mw.tracking_worker, "detected_properties_cache_path", "")
                or ""
            ).strip()
            worker_detected_cnn_paths = {
                str(label): str(path).strip()
                for label, path in (
                    getattr(self._mw.tracking_worker, "detected_cnn_cache_paths", {})
                    or {}
                ).items()
                if str(path).strip()
            }
        if worker_props_path:
            self._mw.current_individual_properties_cache_path = worker_props_path
            logger.info(
                "Using individual properties cache for export: %s",
                worker_props_path,
            )
        if worker_detected_props_path:
            self._mw.current_detected_properties_cache_path = worker_detected_props_path
            logger.info(
                "Using detected properties cache for export: %s",
                worker_detected_props_path,
            )
        if worker_detected_cnn_paths:
            self._mw.current_detected_cnn_cache_paths = worker_detected_cnn_paths
            logger.info(
                "Using detected CNN caches for export: %s",
                worker_detected_cnn_paths,
            )

    def _accumulate_session_fps(self, fps_list, is_backward_mode):
        """Update session-level fps and frames-processed stats."""
        if isinstance(fps_list, (list, tuple)) and fps_list:
            self._mw._session_fps_list = list(self._mw._session_fps_list) + [
                f for f in fps_list if f and f > 0
            ]
        if not is_backward_mode:
            self._mw._session_frames_processed = (
                len(fps_list) if isinstance(fps_list, (list, tuple)) else 0
            )

    def _handle_tracking_failed(self):
        """Show error dialog and finalize session when tracking did not finish normally."""
        logger.error("Tracking did not finish normally.")
        if getattr(self._mw, "_headless_tracking_mode", False):
            self._mw._headless_session_error = (
                "An error occurred during tracking. Check logs for details."
            )
        else:
            QMessageBox.warning(
                self._mw,
                "Tracking Failed",
                "An error occurred during tracking. Check logs for details.",
            )
        if self._panels.setup.g_batch.isChecked():
            self._mw.current_batch_index = -1
            logger.info("Batch mode aborted due to error.")
        self._finish_tracking_session(final_csv_path=None)

    def on_tracking_finished(self: object, finished_normally, fps_list, full_traj):
        """on_tracking_finished method documentation."""
        sender = None
        if (
            sender is not None
            and self._mw.tracking_worker is not None
            and sender is not self._mw.tracking_worker
        ):
            logger.debug(
                "Ignoring stale tracking finished signal from previous worker."
            )
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)

        self._stop_csv_writer()

        if self._mw._stop_all_requested:
            logger.info("Tracking stop requested; skipping post-processing pipeline.")
            self._cleanup_thread_reference("tracking_worker")
            self._mw._refresh_progress_visibility()
            gc.collect()
            return

        if self._mw.btn_preview.isChecked():
            self._handle_preview_mode_finished(finished_normally)
            gc.collect()
            return

        self._collect_worker_props_path()

        if not finished_normally:
            self._handle_tracking_failed()
            return

        logger.info("Tracking completed successfully.")
        is_backward_mode = (
            hasattr(self._mw.tracking_worker, "backward_mode")
            and self._mw.tracking_worker.backward_mode
        )
        self._accumulate_session_fps(fps_list, is_backward_mode)
        is_backward_enabled = self._panels.tracking.chk_enable_backward.isChecked()

        self._mw._postprocess_is_backward_mode = is_backward_mode
        self._mw._postprocess_is_backward_enabled = is_backward_enabled
        self._mw._postprocess_fps_list = fps_list

        self._start_postprocess_worker(is_backward_mode, is_backward_enabled)

    def _start_postprocess_worker(self, is_backward_mode, is_backward_enabled):
        """Launch PostProcessWorker to clean raw trajectory CSV in the background."""
        from hydra_suite.trackerkit.gui.workers.postprocess_worker import (
            PostProcessWorker,
        )

        params = self._mw.get_parameters_dict()
        raw_csv_path = self._panels.setup.csv_line.text()

        if is_backward_mode:
            base, ext = os.path.splitext(raw_csv_path)
            csv_to_process = f"{base}_backward{ext}"
        elif is_backward_enabled:
            base, ext = os.path.splitext(raw_csv_path)
            csv_to_process = f"{base}_forward{ext}"
        else:
            csv_to_process = raw_csv_path

        clean = self._panels.postprocess.enable_postprocessing.isChecked()

        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Post-processing trajectories...")

        self._mw.postprocess_worker = PostProcessWorker(
            csv_to_process, params, clean=clean
        )
        self._mw.postprocess_worker.progress_signal.connect(
            self.on_postprocess_progress
        )
        self._mw.postprocess_worker.finished_signal.connect(
            self.on_postprocess_finished
        )
        self._mw.postprocess_worker.error_signal.connect(self.on_postprocess_error)
        self._mw.postprocess_worker.start()

    def on_postprocess_progress(self, value, message):
        """Update progress bar during post-processing."""
        if self._mw._stop_all_requested:
            return
        self._mw.progress_bar.setValue(value)
        self._mw.progress_label.setText(message)

    def on_postprocess_finished(self, processed_trajectories):
        """Route processed trajectories to the appropriate pipeline stage."""
        self._cleanup_thread_reference("postprocess_worker")
        if self._mw._stop_all_requested:
            self._mw._refresh_progress_visibility()
            return
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)

        is_backward_mode = getattr(self._mw, "_postprocess_is_backward_mode", False)
        is_backward_enabled = getattr(
            self._mw, "_postprocess_is_backward_enabled", False
        )
        fps_list = getattr(self._mw, "_postprocess_fps_list", [])

        if not is_backward_mode:
            self._handle_forward_tracking_done(
                processed_trajectories, is_backward_enabled, fps_list
            )
        else:
            self._handle_backward_tracking_done(processed_trajectories)

    def on_postprocess_error(self, error_message):
        """Handle post-processing errors."""
        self._cleanup_thread_reference("postprocess_worker")
        if self._mw._stop_all_requested:
            self._mw._refresh_progress_visibility()
            return
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        if getattr(self._mw, "_headless_tracking_mode", False):
            self._mw._headless_session_error = (
                f"Error during trajectory post-processing: {error_message}"
            )
            self._finalize_tracking_session_ui()
            return
        QMessageBox.critical(
            self._mw,
            "Post-Processing Error",
            f"Error during trajectory post-processing:\n{error_message}",
        )
        logger.error(f"Trajectory post-processing error: {error_message}")

    def _check_pose_export_sources(self):
        """Return (has_other_analyses, cache_path, cache_available, interp_pose_path,
        interp_available, interp_pose_df_mem, interp_mem_available)."""
        _detected_props_path = str(
            getattr(self._mw, "current_detected_properties_cache_path", None) or ""
        ).strip()
        _has_detected_props = bool(
            _detected_props_path and os.path.exists(_detected_props_path)
        )
        _detected_cnn_paths = (
            getattr(self._mw, "current_detected_cnn_cache_paths", {}) or {}
        )
        _has_detected_cnn = any(
            str(path).strip() and os.path.exists(str(path).strip())
            for path in _detected_cnn_paths.values()
        )
        _has_interp_tag = bool(
            (getattr(self._mw, "current_interpolated_tag_csv_path", None))
            or (
                isinstance(
                    getattr(self._mw, "current_interpolated_tag_df", None),
                    pd.DataFrame,
                )
            )
        )
        _has_interp_cnn = bool(
            getattr(self._mw, "current_interpolated_cnn_csv_paths", None)
            or getattr(self._mw, "current_interpolated_cnn_dfs", None)
        )
        _has_interp_ht = bool(
            (getattr(self._mw, "current_interpolated_headtail_csv_path", None))
            or (
                isinstance(
                    getattr(self._mw, "current_interpolated_headtail_df", None),
                    pd.DataFrame,
                )
            )
        )
        _has_other_analyses = (
            _has_detected_props
            or _has_detected_cnn
            or _has_interp_tag
            or _has_interp_cnn
            or _has_interp_ht
        )
        cache_path = str(
            self._mw.current_individual_properties_cache_path or ""
        ).strip()
        cache_available = bool(cache_path and os.path.exists(cache_path))
        interp_pose_path = str(
            self._mw.current_interpolated_pose_csv_path or ""
        ).strip()
        interp_available = bool(interp_pose_path and os.path.exists(interp_pose_path))
        interp_pose_df_mem = getattr(self._mw, "current_interpolated_pose_df", None)
        interp_mem_available = (
            isinstance(interp_pose_df_mem, pd.DataFrame)
            and not interp_pose_df_mem.empty
        )
        return (
            _has_other_analyses,
            cache_path,
            cache_available,
            interp_pose_path,
            interp_available,
            interp_pose_df_mem,
            interp_mem_available,
        )

    def _merge_pose_sources_into_df(
        self,
        trajectories_df,
        cache_path,
        cache_available,
        interp_pose_path,
        interp_available,
        interp_pose_df_mem,
        interp_mem_available,
    ):
        """Merge pose cache, interpolated pose, AprilTag, CNN, and head-tail into trajectories_df."""
        from hydra_suite.core.identity.properties.export import (
            augment_trajectories_with_detected_cnn_cache,
            augment_trajectories_with_detected_properties_cache,
            augment_trajectories_with_pose_cache,
            merge_interpolated_pose_df,
        )

        with_pose_df = trajectories_df
        _detected_props_path = str(
            getattr(self._mw, "current_detected_properties_cache_path", None) or ""
        ).strip()
        if _detected_props_path and os.path.exists(_detected_props_path):
            with_pose_df = augment_trajectories_with_detected_properties_cache(
                with_pose_df,
                _detected_props_path,
            )

        _detected_cnn_paths = (
            getattr(self._mw, "current_detected_cnn_cache_paths", {}) or {}
        )
        for _cnn_label, _cnn_path in _detected_cnn_paths.items():
            _cnn_path = str(_cnn_path or "").strip()
            if not _cnn_path or not os.path.exists(_cnn_path):
                continue
            with_pose_df = augment_trajectories_with_detected_cnn_cache(
                with_pose_df,
                _cnn_path,
                label=str(_cnn_label),
            )

        _tag_cache_path = self._resolve_current_tag_cache_path()
        if _tag_cache_path and os.path.exists(_tag_cache_path):
            try:
                from hydra_suite.core.identity.properties.export import (
                    augment_trajectories_with_detected_apriltag_cache,
                )

                _tag_labels = [
                    str(_lbl)
                    for _lbl in (
                        self._mw.get_parameters_dict().get("TAG_IDENTITY_LABELS", [])
                        or []
                    )
                ]
                with_pose_df = augment_trajectories_with_detected_apriltag_cache(
                    with_pose_df,
                    _tag_cache_path,
                    tag_labels=_tag_labels,
                )
            except Exception:
                logger.debug(
                    "Detection-level AprilTag augmentation skipped.", exc_info=True
                )

        if cache_available:
            min_valid_conf = float(
                self._panels.identity.spin_pose_min_kpt_conf_valid.value()
            )
            _resize_factor = float(
                self._mw.get_parameters_dict().get("RESIZE_FACTOR", 1.0)
            )
            _coord_scale = (
                1.0 / _resize_factor
                if _resize_factor and _resize_factor != 1.0
                else 1.0
            )
            with_pose_df = augment_trajectories_with_pose_cache(
                with_pose_df,
                cache_path,
                ignore_keypoints=self._mw._parse_pose_ignore_keypoints(),
                min_valid_conf=min_valid_conf,
                coordinate_scale=_coord_scale,
            )
        if interp_available:
            interp_pose_df = pd.read_csv(interp_pose_path)
            with_pose_df = merge_interpolated_pose_df(with_pose_df, interp_pose_df)
        elif interp_mem_available:
            with_pose_df = merge_interpolated_pose_df(with_pose_df, interp_pose_df_mem)

        _interp_tag_path = str(
            getattr(self._mw, "current_interpolated_tag_csv_path", None) or ""
        ).strip()
        _interp_tag_df = getattr(self._mw, "current_interpolated_tag_df", None)
        try:
            from hydra_suite.core.identity.properties.export import (
                merge_interpolated_apriltag_df,
            )

            if _interp_tag_path and os.path.exists(_interp_tag_path):
                _tag_df = pd.read_csv(_interp_tag_path)
                with_pose_df = merge_interpolated_apriltag_df(with_pose_df, _tag_df)
            elif isinstance(_interp_tag_df, pd.DataFrame) and not _interp_tag_df.empty:
                with_pose_df = merge_interpolated_apriltag_df(
                    with_pose_df, _interp_tag_df
                )
        except Exception:
            logger.debug("Interpolated AprilTag merge skipped.", exc_info=True)

        _interp_cnn_paths = (
            getattr(self._mw, "current_interpolated_cnn_csv_paths", {}) or {}
        )
        _interp_cnn_dfs = getattr(self._mw, "current_interpolated_cnn_dfs", {}) or {}
        try:
            from hydra_suite.core.identity.properties.export import (
                merge_interpolated_cnn_df,
            )

            _all_cnn_labels = set(_interp_cnn_paths.keys()) | set(
                _interp_cnn_dfs.keys()
            )
            for _cnn_label in _all_cnn_labels:
                _cnn_path = str(_interp_cnn_paths.get(_cnn_label, "")).strip()
                if _cnn_path and os.path.exists(_cnn_path):
                    _cnn_df = pd.read_csv(_cnn_path)
                    with_pose_df = merge_interpolated_cnn_df(
                        with_pose_df, _cnn_df, label=_cnn_label
                    )
                elif _cnn_label in _interp_cnn_dfs:
                    _cnn_df = _interp_cnn_dfs[_cnn_label]
                    if isinstance(_cnn_df, pd.DataFrame) and not _cnn_df.empty:
                        with_pose_df = merge_interpolated_cnn_df(
                            with_pose_df, _cnn_df, label=_cnn_label
                        )
        except Exception:
            logger.debug("Interpolated CNN merge skipped.", exc_info=True)

        _interp_ht_path = str(
            getattr(self._mw, "current_interpolated_headtail_csv_path", None) or ""
        ).strip()
        _interp_ht_df = getattr(self._mw, "current_interpolated_headtail_df", None)
        try:
            from hydra_suite.core.identity.properties.export import (
                merge_interpolated_headtail_df,
            )

            if _interp_ht_path and os.path.exists(_interp_ht_path):
                _ht_df = pd.read_csv(_interp_ht_path)
                with_pose_df = merge_interpolated_headtail_df(with_pose_df, _ht_df)
            elif isinstance(_interp_ht_df, pd.DataFrame) and not _interp_ht_df.empty:
                with_pose_df = merge_interpolated_headtail_df(
                    with_pose_df, _interp_ht_df
                )
        except Exception:
            logger.debug("Interpolated head-tail merge skipped.", exc_info=True)

        return with_pose_df

    def _apply_pose_quality_postprocessing(self, with_pose_df, pose_labels, params):
        """Apply quality gating and temporal post-processing to pose-augmented dataframe."""
        from hydra_suite.core.identity.pose.features import resolve_pose_group_indices
        from hydra_suite.core.identity.pose.quality import (
            apply_quality_to_dataframe,
            apply_temporal_pose_postprocessing,
            calibrate_body_length_prior,
            calibrate_edge_length_priors,
        )

        kpt_names = []
        try:
            from hydra_suite.core.identity.properties.cache import (
                IndividualPropertiesCache,
            )

            _cache_path = str(
                self._mw.current_individual_properties_cache_path or ""
            ).strip()
            if _cache_path and os.path.exists(_cache_path):
                _cache = IndividualPropertiesCache(_cache_path, mode="r")
                try:
                    kpt_names = [
                        str(v)
                        for v in (_cache.metadata.get("pose_keypoint_names", []) or [])
                    ]
                finally:
                    _cache.close()
        except Exception:
            pass
        anterior_indices = resolve_pose_group_indices(
            params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), kpt_names
        )
        posterior_indices = resolve_pose_group_indices(
            params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), kpt_names
        )

        skeleton_edges = []
        try:
            _skel_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
            if _skel_file and os.path.exists(_skel_file):
                with open(_skel_file, "r", encoding="utf-8") as _sf:
                    _skel_data = json.load(_sf)
                for _edge in _skel_data.get(
                    "skeleton_edges", _skel_data.get("edges", [])
                ):
                    if isinstance(_edge, (list, tuple)) and len(_edge) >= 2:
                        try:
                            skeleton_edges.append((int(_edge[0]), int(_edge[1])))
                        except Exception:
                            pass
        except Exception:
            logger.exception(
                "Failed to load skeleton edges for anatomy check; skipping."
            )
            skeleton_edges = []

        body_length_prior = None
        if anterior_indices and posterior_indices:
            try:
                body_length_prior = calibrate_body_length_prior(
                    with_pose_df,
                    pose_labels,
                    anterior_indices,
                    posterior_indices,
                    min_valid_conf=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
                )
                if body_length_prior.is_valid:
                    logger.info(
                        "Body-length prior calibrated: median=%.1f px, MAD=%.1f px, n=%d",
                        body_length_prior.median_px,
                        body_length_prior.mad_px,
                        body_length_prior.n_samples,
                    )
            except Exception:
                logger.exception(
                    "Body-length prior calibration failed; skipping anatomy check."
                )
                body_length_prior = None

        edge_length_priors = None
        if skeleton_edges:
            try:
                edge_length_priors = calibrate_edge_length_priors(
                    with_pose_df,
                    pose_labels,
                    skeleton_edges,
                    min_valid_conf=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
                )
                if edge_length_priors.is_valid:
                    logger.info(
                        "Edge-length priors calibrated for %d edges.",
                        len(edge_length_priors.priors),
                    )
            except Exception:
                logger.exception(
                    "Edge-length prior calibration failed; skipping skeleton check."
                )
                edge_length_priors = None

        try:
            with_pose_df = apply_quality_to_dataframe(
                with_pose_df,
                pose_labels,
                params,
                body_length_prior=body_length_prior,
                anterior_indices=anterior_indices if anterior_indices else None,
                posterior_indices=posterior_indices if posterior_indices else None,
                skeleton_edges=skeleton_edges if skeleton_edges else None,
                edge_length_priors=edge_length_priors,
            )
        except Exception:
            logger.exception("Pose quality gating failed; using unfiltered pose.")

        max_gap = int(params.get("POSE_POSTPROC_MAX_GAP", 5))
        z_threshold = float(params.get("POSE_TEMPORAL_OUTLIER_ZSCORE", 3.0))
        if z_threshold > 0.0 and "TrajectoryID" in with_pose_df.columns:
            try:
                parts = []
                for _, traj_group in with_pose_df.groupby("TrajectoryID", sort=False):
                    parts.append(
                        apply_temporal_pose_postprocessing(
                            traj_group,
                            pose_labels,
                            max_gap=max_gap,
                            z_score_threshold=z_threshold,
                        )
                    )
                if parts:
                    with_pose_df = (
                        pd.concat(parts, ignore_index=True)
                        .sort_values(["TrajectoryID", "FrameID"], kind="stable")
                        .reset_index(drop=True)
                    )
            except Exception:
                logger.exception(
                    "Pose temporal post-processing failed; using unfiltered pose."
                )
        return with_pose_df

    def _resolve_current_tag_cache_path(self) -> str:
        """Return the best available detected AprilTag cache path for this session."""
        current_params = self._mw.get_parameters_dict()
        if not bool(current_params.get("USE_APRILTAGS", False)):
            return ""
        detection_cache_path = getattr(self._mw, "current_detection_cache_path", None)
        if not detection_cache_path or not os.path.exists(str(detection_cache_path)):
            return ""
        pattern = str(detection_cache_path).replace(".npz", "") + "_tags_*.npz"
        candidates = sorted(_glob.glob(pattern))
        return str(candidates[-1]) if candidates else ""

    def _apply_identity_postprocessing_to_df(
        self, with_pose_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run identity-aware split/join processing on the augmented dataframe."""
        if with_pose_df is None or with_pose_df.empty:
            return with_pose_df

        params = self._mw.get_parameters_dict()

        def _annotate_identity_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            cnn_class_columns = [
                col
                for col in out.columns
                if str(col).startswith("CNN_") and str(col).endswith("_Class")
            ]

            def _row_sources(row: pd.Series) -> object:
                sources = []
                if pd.notna(row.get("DetectedTagID")) or pd.notna(
                    row.get("InterpTagID")
                ):
                    sources.append("apriltag")
                if any(pd.notna(row.get(col)) for col in cnn_class_columns):
                    sources.append("cnn")
                if pd.notna(row.get("IdentityOfflineLabel")) or pd.notna(
                    row.get("IdentitySmoothedLabel")
                ):
                    sources.append("offline")
                if pd.notna(row.get("IdentityAssignedLabel")) and not sources:
                    sources.append("online")
                if not sources:
                    return np.nan
                return ",".join(sorted(set(sources)))

            def _row_conflict(row: pd.Series) -> int:
                assigned = row.get("IdentityAssignedLabel")
                observed = set()
                detected_tag_label = row.get("DetectedTagLabel")
                if pd.notna(detected_tag_label):
                    observed.add(str(detected_tag_label))
                for col in cnn_class_columns:
                    value = row.get(col)
                    if pd.notna(value):
                        observed.add(str(value))
                if pd.notna(assigned):
                    assigned_label = str(assigned)
                    if any(label != assigned_label for label in observed):
                        return 1
                return 1 if len(observed) > 1 else 0

            out["IdentityEvidenceSources"] = out.apply(_row_sources, axis=1)
            out["IdentityConflictFlag"] = out.apply(_row_conflict, axis=1).astype(int)
            return out

        try:
            # Build catalog the same way as the online decoder: CNN composite
            # class labels (cartesian product for multi-factor models) followed
            # by tag labels that match CNN classes.  Using the CNN *phase name*
            # (e.g. "test") instead of class names was the previous bug — phase
            # names are model identifiers, not individual animal identities.
            import itertools as _itertools

            from hydra_suite.core.identity.catalog import IdentityCatalog
            from hydra_suite.core.identity.fragment_solver import run_fragment_solver
            from hydra_suite.core.post.identity_postprocess import (
                fill_identity_nans_with_consensus,
                sort_trajectories_by_identity,
            )

            _raw_labels: list[str] = []
            for _cnn_cfg in params.get("CNN_CLASSIFIERS", []) or []:
                if not bool(_cnn_cfg.get("unique_identifier", False)):
                    continue
                _cnpf = list(_cnn_cfg.get("class_names_per_factor") or [])
                _non_empty = [fl for fl in _cnpf if fl]
                if len(_non_empty) > 1:
                    for _combo in _itertools.product(*_non_empty):
                        _c = "_".join(str(x) for x in _combo if x)
                        if _c and _c not in _raw_labels:
                            _raw_labels.append(_c)
                elif len(_non_empty) == 1:
                    for _l in _non_empty[0]:
                        if _l and str(_l) not in _raw_labels:
                            _raw_labels.append(str(_l))
                else:
                    for _l in _cnn_cfg.get("labels", []) or []:
                        if _l and str(_l) not in _raw_labels:
                            _raw_labels.append(str(_l))

            _cnn_label_set = set(_raw_labels)
            for _lbl in params.get("TAG_IDENTITY_LABELS", []) or []:
                _s = str(_lbl).strip()
                if not _s:
                    continue
                # When CNN classes are known, only accept tag labels that
                # match them — prevents garbage composites from entering.
                if _cnn_label_set and _s not in _cnn_label_set:
                    continue
                if _s not in _raw_labels:
                    _raw_labels.append(_s)

            # Tag-only config (no CNN): accept all tag labels.
            if not _raw_labels:
                for _lbl in params.get("TAG_IDENTITY_LABELS", []) or []:
                    _s = str(_lbl).strip()
                    if _s and _s not in _raw_labels:
                        _raw_labels.append(_s)

            if params.get("ENABLE_IDENTITY_FRAGMENT_SOLVER", False) and _raw_labels:
                try:
                    catalog = IdentityCatalog.from_labels(_raw_labels)
                    with_pose_df = run_fragment_solver(with_pose_df, catalog, params)
                    with_pose_df = _annotate_identity_summary_columns(with_pose_df)
                    logger.info("Fragment solver complete.")
                except Exception:
                    logger.exception("Fragment solver failed; results unchanged.")

            with_pose_df = fill_identity_nans_with_consensus(with_pose_df)
            with_pose_df = sort_trajectories_by_identity(with_pose_df)
        except Exception:
            logger.exception(
                "Identity-aware post-processing failed; using unmodified rich dataframe."
            )
        return _annotate_identity_summary_columns(with_pose_df)

    def _build_rich_export_dataframe(self, final_csv_path):
        """Load final CSV and merge all available analysis sources into a rich export dataframe."""
        if not final_csv_path or not os.path.exists(final_csv_path):
            return None

        (
            _has_other_analyses,
            cache_path,
            cache_available,
            interp_pose_path,
            interp_available,
            interp_pose_df_mem,
            interp_mem_available,
        ) = self._check_pose_export_sources()

        if (
            not cache_available
            and not interp_available
            and not interp_mem_available
            and not _has_other_analyses
        ):
            logger.warning(
                "Rich export skipped: no analysis sources found (pose_cache=%s, interp=%s, in_memory=%s).",
                cache_path or "<empty>",
                interp_pose_path or "<empty>",
                bool(interp_mem_available),
            )
            return None

        try:
            trajectories_df = pd.read_csv(final_csv_path)
        except Exception:
            logger.exception(
                "Rich export skipped: failed to load trajectories CSV: %s",
                final_csv_path,
            )
            return None

        try:
            with_pose_df = self._merge_pose_sources_into_df(
                trajectories_df,
                cache_path,
                cache_available,
                interp_pose_path,
                interp_available,
                interp_pose_df_mem,
                interp_mem_available,
            )
        except Exception:
            logger.exception(
                "Rich export skipped: failed while merging sources (pose_cache=%s, interp=%s)",
                cache_path or "<empty>",
                interp_pose_path or "<empty>",
            )
            return None

        if with_pose_df is None or with_pose_df.empty:
            logger.warning(
                "Rich export skipped: merged dataframe is empty for %s",
                final_csv_path,
            )
            return None

        _kpt_re = re.compile(r"^PoseKpt_(.+)_X$")
        pose_labels = [
            m.group(1) for col in with_pose_df.columns if (m := _kpt_re.match(str(col)))
        ]

        params = self._mw.get_parameters_dict()

        if pose_labels:
            with_pose_df = self._apply_pose_quality_postprocessing(
                with_pose_df, pose_labels, params
            )

        with_pose_df = self._apply_identity_postprocessing_to_df(with_pose_df)

        self._log_rich_export_summary(with_pose_df)

        return with_pose_df

    def _export_rich_csv(self, final_csv_path):
        """Write the rich individual-analysis CSV next to the final CSV."""
        with_pose_df = self._build_rich_export_dataframe(final_csv_path)
        if with_pose_df is None or with_pose_df.empty:
            return None

        return self._write_rich_export_csv(with_pose_df, final_csv_path)

    def _relink_and_export_rich_csv(self, final_csv_path):
        """Rewrite final CSV IDs after pose-aware relinking and regenerate the rich export CSV."""
        if not final_csv_path or not os.path.exists(final_csv_path):
            return None

        with_pose_df = self._build_rich_export_dataframe(final_csv_path)
        params = self._mw.get_parameters_dict()

        try:
            base_df = pd.read_csv(final_csv_path)
        except Exception:
            logger.exception(
                "Relinking skipped: failed to reload final CSV: %s", final_csv_path
            )
            return self._export_rich_csv(final_csv_path)

        relink_input_df = (
            with_pose_df
            if with_pose_df is not None and not with_pose_df.empty
            else base_df
        )
        from hydra_suite.core.post.processing import relink_trajectories_with_pose

        relinked_with_pose = relink_trajectories_with_pose(relink_input_df, params)
        if relinked_with_pose is None or relinked_with_pose.empty:
            relinked_with_pose = relink_input_df

        common_cols = [
            col for col in base_df.columns if col in relinked_with_pose.columns
        ]
        relinked_base = relinked_with_pose.loc[:, common_cols].copy()
        relinked_base = relinked_base.sort_values(
            ["TrajectoryID", "FrameID"], kind="stable"
        ).reset_index(drop=True)
        relinked_with_pose = relinked_with_pose.sort_values(
            ["TrajectoryID", "FrameID"], kind="stable"
        ).reset_index(drop=True)

        try:
            relinked_base.to_csv(final_csv_path, index=False)
        except Exception:
            logger.exception("Failed to rewrite relinked final CSV: %s", final_csv_path)
            return None

        if with_pose_df is not None and not with_pose_df.empty:
            rich_path = self._write_rich_export_csv(relinked_with_pose, final_csv_path)
            if not rich_path:
                return None
        else:
            self._remove_legacy_rich_exports(final_csv_path)

        logger.info(
            "Relinked final CSV rewritten: %s (%d trajectories)",
            final_csv_path,
            (
                int(relinked_base["TrajectoryID"].nunique())
                if "TrajectoryID" in relinked_base.columns
                else 0
            ),
        )
        if with_pose_df is not None and not with_pose_df.empty:
            rich_path = self._rich_export_path(final_csv_path)
            logger.info("Relinked rich-export CSV saved: %s", rich_path)
            return rich_path
        return final_csv_path

    def _load_video_trajectories(self, final_csv_path):
        """Load best available trajectories for video generation (prefers rich export CSV)."""
        if not final_csv_path:
            return None, None
        candidates = [
            self._rich_export_path(final_csv_path),
            self._rich_export_path(final_csv_path, legacy=True),
            final_csv_path,
        ]
        candidate = next((path for path in candidates if os.path.exists(path)), None)
        if not candidate:
            return None, None
        try:
            return pd.read_csv(candidate), candidate
        except Exception:
            logger.exception("Failed to load video trajectories from: %s", candidate)
            return None, None

    def _run_pending_video_generation_or_finalize(self):
        """Run video generation if queued; otherwise finalize UI/session cleanup."""
        if self._mw._stop_all_requested:
            self._finalize_tracking_session_ui()
            return
        csv_path = self._mw._pending_video_csv_path
        should_render_video = bool(self._mw._pending_video_generation and csv_path)
        self._mw._pending_video_generation = False
        self._mw._pending_video_csv_path = None
        self._mw._pending_pose_export_csv_path = None

        if should_render_video:
            trajectories_df, loaded_path = self._load_video_trajectories(csv_path)
            if trajectories_df is None or trajectories_df.empty:
                logger.warning(
                    "Skipping final video generation: no trajectories loaded from %s",
                    csv_path,
                )
                self._finalize_tracking_session_ui()
                return
            logger.info(
                "Final video rendering uses trajectories from: %s",
                loaded_path or csv_path,
            )
            self._generate_video_from_trajectories(
                trajectories_df,
                csv_path=csv_path,
                finalize_on_complete=False,
            )
            return

        self._finalize_tracking_session_ui()

    def _finish_tracking_session(self, final_csv_path=None):
        """Complete tracking session cleanup and UI updates."""
        if self._mw._stop_all_requested:
            self._finalize_tracking_session_ui()
            return
        self._mw._session_final_csv_path = final_csv_path
        # Hide progress elements
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)

        if final_csv_path:
            self._mw._pending_pose_export_csv_path = final_csv_path
            self._export_rich_csv(final_csv_path)

        self._mw._pending_video_csv_path = final_csv_path
        self._mw._pending_video_generation = bool(
            final_csv_path
            and self._panels.postprocess.check_video_output.isChecked()
            and self._panels.postprocess.video_out_line.text().strip()
        )

        # Generate dataset if enabled (BEFORE cleanup so files are still available)
        if self._panels.dataset.chk_enable_dataset_gen.isChecked():
            self._generate_training_dataset(override_csv_path=final_csv_path)
            self._mw._dataset_was_started = True

        # Interpolate occlusions for individual analysis (post-pass).
        # This also powers pose enrichment on occluded frames in final CSV.
        if self._mw._should_run_interpolated_postpass():
            started = self._generate_interpolated_individual_crops(final_csv_path)
            if started:
                # Hold final UI/session completion until interpolation finishes.
                self._mw._pending_finish_after_interp = True
                return

        if final_csv_path:
            self._relink_and_export_rich_csv(final_csv_path)

        if self._start_pending_final_media_export(final_csv_path):
            return

        self._run_pending_video_generation_or_finalize()

    def _finalize_tracking_session_ui(self):
        """Finalize session cleanup and return UI to idle state."""
        self._mw._pending_pose_export_csv_path = None
        self._mw._pending_video_csv_path = None
        self._mw._pending_video_generation = False
        self._mw._pending_finish_after_track_videos = False
        self._mw.current_interpolated_pose_df = None
        self._mw.current_interpolated_roi_npz_path = None
        # Force-clear progress UI at terminal session state.
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Ready")
        # Clean up session logging
        self._mw._cleanup_session_logging()
        self._mw._cleanup_temporary_files()

        # Hide stats labels
        self._mw.label_current_fps.setVisible(False)
        self._mw.label_elapsed_time.setVisible(False)
        self._mw.label_eta.setVisible(False)

        # Determine if we are continuing a batch
        is_batch_continuing = (
            self._panels.setup.g_batch.isChecked()
            and self._mw.current_batch_index >= 0
            and (self._mw.current_batch_index + 1) < len(self._mw.batch_videos)
        )

        if not is_batch_continuing:
            self._mw._set_ui_controls_enabled(True)
            self._mw.btn_start.blockSignals(True)
            self._mw.btn_start.setChecked(False)
            self._mw.btn_start.blockSignals(False)
            self._mw.btn_start.setText("Start Full Tracking")
            self._mw._apply_ui_state(
                "idle" if self._mw.current_video_path else "no_video"
            )
            logger.info("✓ Tracking session complete.")

            # Show end-of-session summary. If the dataset worker is still running,
            # defer the summary until it finishes so we can include its result.
            if getattr(
                self._mw, "_dataset_was_started", False
            ) and self._mw._is_worker_running(self._mw.dataset_worker):
                self._mw._show_summary_on_dataset_done = True
            else:
                self._show_session_summary()
        else:
            logger.info("✓ Video complete. Continuing batch...")
            # Disable deferred summary for intermediate batch items so it doesn't block
            self._mw._show_summary_on_dataset_done = False

        # --- Batch Mode Continuation ---
        if self._panels.setup.g_batch.isChecked() and self._mw.current_batch_index >= 0:
            self._mw.current_batch_index += 1
            if self._mw.current_batch_index < len(self._mw.batch_videos):
                # Load next video
                fp = self._mw.batch_videos[self._mw.current_batch_index]
                self._panels.setup.list_batch_videos.setCurrentRow(
                    self._mw.current_batch_index
                )

                # If the video has its own config, load it.  Otherwise restore the
                # keystone config so that videos without per-video configs always
                # use the keystone parameters, not leftover params from a previous
                # video that did have its own config.
                plan = resolve_video_plan(
                    fp,
                    keystone_config_path=_get_video_config_path(
                        self._mw.batch_videos[0]
                    ),
                    keystone_override=self._panels.setup.chk_batch_keystone_override.isChecked(),
                )
                if plan.use_keystone_baseline and plan.config_path:
                    self._mw._config_orch._load_config_from_file(plan.config_path)
                self._mw._setup_video_file(
                    fp,
                    skip_config_load=plan.use_keystone_baseline
                    or not plan.has_own_config,
                )

                # Small delay to ensure UI updates before starting next
                logger.info(
                    f"Batch Mode: Starting next video ({self._mw.current_batch_index + 1}/{len(self._mw.batch_videos)})"
                )
                QTimer.singleShot(1000, lambda: self.start_tracking(preview_mode=False))
            else:
                # Batch complete
                self._mw.current_batch_index = -1
                QMessageBox.information(
                    self._mw,
                    "Batch Complete",
                    f"Finished processing {len(self._mw.batch_videos)} videos.",
                )
        else:
            # Ensure reset if batch mode is disabled mid-run or not used
            self._mw.current_batch_index = -1

    def _generate_interpolated_individual_crops(self, csv_path):
        """Post-pass interpolation for occluded segments in individual dataset."""
        try:
            from hydra_suite.trackerkit.gui.workers.crops_worker import (
                InterpolatedCropsWorker,
            )

            if self._mw._stop_all_requested:
                return False
            if not self._panels.identity.chk_individual_interpolate.isChecked():
                return False

            target_csv = None
            if csv_path and os.path.exists(csv_path):
                target_csv = csv_path
            elif self._panels.setup.csv_line.text() and os.path.exists(
                self._panels.setup.csv_line.text()
            ):
                target_csv = self._panels.setup.csv_line.text()
            if not target_csv or not os.path.exists(target_csv):
                return False

            video_path = self._panels.setup.file_line.text()
            if not video_path or not os.path.exists(video_path):
                return False

            params = self._mw.get_parameters_dict()
            save_interpolated_outputs = bool(
                params.get("ENABLE_INDIVIDUAL_IMAGE_SAVE", False)
            )
            generate_oriented_videos = bool(
                params.get("FINAL_MEDIA_EXPORT_VIDEOS_ENABLED", False)
                or params.get("GENERATE_ORIENTED_TRACK_VIDEOS", False)
            )
            output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
            if not save_interpolated_outputs and generate_oriented_videos:
                output_dir = str(
                    params.get("FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR", "")
                    or params.get("ORIENTED_TRACK_VIDEO_OUTPUT_DIR", output_dir)
                ).strip()
            if not output_dir:
                # Keep interpolated analysis available even when image-save toggle is off.
                csv_dir = os.path.dirname(target_csv) if target_csv else ""
                fallback_output = (
                    os.path.join(csv_dir, "training_data") if csv_dir else ""
                )
                if fallback_output:
                    try:
                        os.makedirs(fallback_output, exist_ok=True)
                    except Exception:
                        pass
                    params["INDIVIDUAL_DATASET_OUTPUT_DIR"] = fallback_output
                    logger.info(
                        "Interpolated analysis output dir not set; using fallback: %s",
                        fallback_output,
                    )

            self._mw.progress_bar.setVisible(True)
            self._mw.progress_label.setVisible(True)
            self._mw.progress_bar.setValue(0)
            self._mw.progress_label.setText("Interpolating occluded crops...")

            if (
                self._mw.interp_worker is not None
                and self._mw.interp_worker.isRunning()
            ):
                logger.warning(
                    "Interpolated crop generation already in progress; skipping duplicate request."
                )
                return True
            if (
                self._mw.interp_worker is not None
                and not self._mw.interp_worker.isRunning()
            ):
                self._mw.interp_worker.deleteLater()
                self._mw.interp_worker = None

            self._mw.current_interpolated_roi_npz_path = None
            self._mw.current_interpolated_pose_csv_path = None
            self._mw.current_interpolated_pose_df = None
            self._mw.current_interpolated_tag_csv_path = None
            self._mw.current_interpolated_tag_df = None
            self._mw.current_interpolated_cnn_csv_paths = {}
            self._mw.current_interpolated_cnn_dfs = {}
            self._mw.current_interpolated_headtail_csv_path = None
            self._mw.current_interpolated_headtail_df = None

            _interp_profiling = bool(params.get("ENABLE_PROFILING", False))
            _interp_profile_path = None
            if _interp_profiling:
                if self._mw.current_detection_cache_path:
                    _interp_profile_path = str(
                        Path(self._mw.current_detection_cache_path).parent
                        / "interp_profile.json"
                    )
                elif video_path:
                    _interp_profile_path = str(
                        Path(video_path).parent / "interp_profile.json"
                    )

            self._mw.interp_worker = InterpolatedCropsWorker(
                target_csv,
                video_path,
                self._mw.current_detection_cache_path,
                params,
                enable_profiling=_interp_profiling,
                profile_export_path=_interp_profile_path,
            )
            self._mw.interp_worker.progress_signal.connect(self.on_progress_update)
            self._mw.interp_worker.finished_signal.connect(
                self._on_interpolated_crops_finished
            )
            self._mw.interp_worker.start()
            return True
        except Exception as e:
            logger.warning(f"Interpolated individual crops failed: {e}")
            return False

    def start_backward_tracking(self):
        """start_backward_tracking method documentation."""
        if self._mw._stop_all_requested:
            return
        logger.info("=" * 80)
        logger.info("Starting backward tracking pass (using cached detections)...")
        logger.info("=" * 80)

        video_fp = self._panels.setup.file_line.text()
        if not video_fp:
            return

        # Use original video (no reversal needed with detection caching)
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText(
            "Starting backward tracking (using cached detections)..."
        )
        QApplication.processEvents()

        # Start backward tracking directly on original video with cached detections
        self.start_tracking_on_video(video_fp, backward_mode=True)

    def start_tracking(self: object, preview_mode: bool, backward_mode: bool = False):
        """start_tracking method documentation."""
        if not preview_mode:
            # If batch mode group is checked, initialize batch processing
            if self._panels.setup.g_batch.isChecked():
                if self._mw.current_batch_index < 0:
                    res = QMessageBox.question(
                        self._mw,
                        "Start Batch Process",
                        f"This will process {len(self._mw.batch_videos)} videos sequentially using the CURRENT parameters.\n\n"
                        "Each video will have its own CSV and configuration file saved in its source directory.\n\n"
                        "Continue?",
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    if res == QMessageBox.No:
                        return

                    # Start at the first video (Keystone)
                    self._mw.current_batch_index = 0
                    self._mw._sync_keystone_to_batch()
                    fp = self._mw.batch_videos[0]
                    self._panels.setup.list_batch_videos.setCurrentRow(0)

                    # Ensure the keystone video is loaded WITHOUT overwriting current UI params
                    if self._mw.current_video_path != fp:
                        self._mw._setup_video_file(fp, skip_config_load=True)

            # Save config for the CURRENTLY LOADED video (this persists the keystone's params to the current video)
            # In batch mode, we automatically overwrite to avoid halting the automated process.
            if not self._mw.save_config(
                prompt_if_exists=not self._panels.setup.g_batch.isChecked()
            ):
                # User cancelled config save, abort tracking
                self._mw.current_batch_index = -1  # Reset batch if cancelled
                return

        video_fp = self._panels.setup.file_line.text()
        if not video_fp:
            QMessageBox.warning(
                self._mw, "No video", "Please select a video file first."
            )
            return
        if preview_mode:
            self.start_preview_on_video(video_fp)
        else:
            self.start_tracking_on_video(video_fp, backward_mode=False)

    def start_preview_on_video(self, video_path):
        """start_preview_on_video method documentation."""
        from hydra_suite.core.tracking import TrackingWorker

        if self._mw.tracking_worker and self._mw.tracking_worker.isRunning():
            return
        self._mw._stop_all_requested = False
        self._mw._pending_finish_after_interp = False

        # Stop video playback if active
        if self._mw.is_playing:
            self._mw._stop_playback()

        # Reset first frame flag for auto-fit
        self._mw._tracking_first_frame = True
        self._mw.csv_writer_thread = None

        params = self._mw.get_parameters_dict()
        if not self._validate_yolo_model_requirements(
            params, mode_label="tracking preview"
        ):
            return

        preview_fps = self._mw._resolve_source_video_fps()
        preview_start_frame = int(params.get("START_FRAME", 0))
        preview_end_frame = int(params.get("END_FRAME", preview_start_frame))
        clamped_end_frame, was_clamped = compute_capped_preview_range(
            preview_start_frame, preview_end_frame, preview_fps
        )
        if was_clamped:
            minutes = PREVIEW_MAX_DURATION_SECONDS // 60
            QMessageBox.warning(
                self._mw,
                "Preview Range Capped",
                f"The selected range ({preview_end_frame - preview_start_frame + 1} "
                f"frames) exceeds the {minutes}-minute preview limit.\n\n"
                f"Preview will run frames {preview_start_frame}-{clamped_end_frame} "
                "only. Use 'Start Full Tracking' to process the entire selected range.",
            )
            params["END_FRAME"] = clamped_end_frame

        # Preview should always render frames regardless of visualization-free toggle
        params["VISUALIZATION_FREE_MODE"] = False
        # Preview must not use ONNX/TensorRT — downgrade to the native device runtime.
        safe_rt = self._mw._preview_safe_runtime(params.get("COMPUTE_RUNTIME", "cpu"))
        if safe_rt != params.get("COMPUTE_RUNTIME"):
            safe_det = derive_detection_runtime_settings(safe_rt)
            params["COMPUTE_RUNTIME"] = safe_rt
            params["YOLO_DEVICE"] = safe_det["yolo_device"]
            params["ENABLE_GPU_BACKGROUND"] = safe_det["enable_gpu_background"]
            params["ENABLE_TENSORRT"] = safe_det["enable_tensorrt"]
            params["ENABLE_ONNX_RUNTIME"] = safe_det["enable_onnx_runtime"]
            safe_pose = derive_pose_runtime_settings(
                safe_rt, backend_family=params.get("POSE_MODEL_TYPE", "yolo")
            )
            params["POSE_RUNTIME_FLAVOR"] = safe_pose["pose_runtime_flavor"]
        params["HEADTAIL_COMPUTE_RUNTIME"] = self._mw._preview_safe_runtime(
            params.get("HEADTAIL_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
        )
        params["CNN_COMPUTE_RUNTIME"] = self._mw._preview_safe_runtime(
            params.get("CNN_COMPUTE_RUNTIME", params.get("COMPUTE_RUNTIME", "cpu"))
        )

        # Preview mode runs forward detection live, but reuses a valid,
        # range-covering YOLO-OBB InferenceRunner cache when one already
        # exists for the current model/config/video (see worker.py:1030-1054).
        # Background-subtraction has no forward-mode cache-read path, so this
        # flag is a no-op for it (see Task 3 for why bgsub must not *write*
        # into the shared cache during preview).
        self._mw.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=None,
            video_output_path=None,
            backward_mode=False,
            detection_cache_path=None,
            preview_mode=True,
            use_cached_detections=True,
        )
        self._mw.tracking_worker.set_parameters(params)
        self._mw.tracking_worker.frame_signal.connect(self.on_new_frame)
        self._mw.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self._mw.tracking_worker.progress_signal.connect(self.on_progress_update)
        self._mw.tracking_worker.stats_signal.connect(self.on_stats_update)
        self._mw.tracking_worker.warning_signal.connect(self.on_tracking_warning)
        self._mw.tracking_worker.pose_exported_model_resolved_signal.connect(
            self.on_pose_exported_model_resolved
        )

        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText("Preview Mode Active")

        self._mw._prepare_tracking_display()
        self._mw._apply_ui_state("preview")
        self._mw.tracking_worker.start()

    @staticmethod
    def _normalize_for_hash(value: object):
        """Convert values to deterministic, JSON-safe forms for hashing."""
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            return {
                "type": "ndarray",
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "digest": hashlib.md5(arr.tobytes()).hexdigest(),
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            if np.isnan(value):
                return "NaN"
            if np.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(k): TrackingOrchestrator._normalize_for_hash(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [TrackingOrchestrator._normalize_for_hash(v) for v in value]
        return value

    @staticmethod
    def _get_model_fingerprint(model_path: object):
        """Return size/mtime fingerprint dict for a model file."""
        from hydra_suite.trackerkit.gui.main_window import (
            resolve_model_path as _resolve_model_path,
        )

        configured = str(model_path or "")
        resolved = str(_resolve_model_path(configured))
        fingerprint = {"configured_path": configured, "resolved_path": resolved}
        if resolved and os.path.exists(resolved):
            try:
                stat = os.stat(resolved)
                fingerprint["size_bytes"] = stat.st_size
                fingerprint["mtime_ns"] = stat.st_mtime_ns
            except OSError:
                fingerprint["size_bytes"] = None
                fingerprint["mtime_ns"] = None
        else:
            fingerprint["size_bytes"] = None
            fingerprint["mtime_ns"] = None
        return fingerprint

    def _get_cache_model_ids(self, params, detection_method):
        """Generate raw-detection and TensorRT-engine cache identity keys."""
        resize_factor = params.get("RESIZE_FACTOR", 1.0)
        resize_str = f"r{int(resize_factor * 100)}"

        def _extract(keys):
            return {k: self._normalize_for_hash(params.get(k)) for k in keys}

        def _build_id(prefix, cache_params, model_stem=""):
            digest = hashlib.md5(
                json.dumps(cache_params, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            if model_stem:
                return f"{prefix}_{model_stem}_{resize_str}_{digest}"
            return f"{prefix}_{resize_str}_{digest}"

        common_detection_keys = (
            "DETECTION_METHOD",
            "RESIZE_FACTOR",
            "MAX_TARGETS",
            "COMPUTE_RUNTIME",
        )

        if detection_method == "yolo_obb":
            return self._get_yolo_obb_cache_ids(
                params, common_detection_keys, _extract, _build_id
            )

        bg_detection_keys = (
            "MAX_CONTOUR_MULTIPLIER",
            "ENABLE_SIZE_FILTERING",
            "MIN_OBJECT_SIZE",
            "MAX_OBJECT_SIZE",
            "ROI_MASK",
            "BACKGROUND_PRIME_FRAMES",
            "ENABLE_ADAPTIVE_BACKGROUND",
            "BACKGROUND_LEARNING_RATE",
            "ENABLE_GPU_BACKGROUND",
            "GPU_DEVICE_ID",
            "THRESHOLD_VALUE",
            "MORPH_KERNEL_SIZE",
            "ENABLE_ADDITIONAL_DILATION",
            "DILATION_ITERATIONS",
            "DILATION_KERNEL_SIZE",
            "BRIGHTNESS",
            "CONTRAST",
            "GAMMA",
            "DARK_ON_LIGHT_BACKGROUND",
            "ENABLE_LIGHTING_STABILIZATION",
            "LIGHTING_SMOOTH_FACTOR",
            "LIGHTING_MEDIAN_WINDOW",
            "ENABLE_CONSERVATIVE_SPLIT",
            "CONSERVATIVE_KERNEL_SIZE",
            "CONSERVATIVE_ERODE_ITER",
            "MIN_CONTOUR_AREA",
            "MIN_DETECTIONS_TO_START",
            "MIN_DETECTION_COUNTS",
        )
        cache_params = {
            "common": _extract(common_detection_keys),
            "background_subtraction": _extract(bg_detection_keys),
        }
        return {
            "inference": _build_id("bgsub", cache_params),
            "engine": None,
        }

    def _get_yolo_obb_cache_ids(
        self, params, common_detection_keys, _extract, _build_id
    ):
        """Build YOLO-OBB inference and engine cache IDs."""
        yolo_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        direct_model = params.get(
            "YOLO_OBB_DIRECT_MODEL_PATH",
            params.get("YOLO_MODEL_PATH", "best.pt"),
        )
        crop_obb_model = params.get(
            "YOLO_CROP_OBB_MODEL_PATH", params.get("YOLO_MODEL_PATH", "best.pt")
        )
        active_obb_model = direct_model if yolo_mode == "direct" else crop_obb_model
        model_fingerprint = self._get_model_fingerprint(active_obb_model)
        model_name = os.path.basename(
            model_fingerprint["resolved_path"] or model_fingerprint["configured_path"]
        )
        model_stem = os.path.splitext(model_name)[0] or "model"
        safe_model_stem = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in model_stem
        )

        yolo_inference_keys = (
            "YOLO_TARGET_CLASSES",
            "YOLO_DEVICE",
            "ENABLE_TENSORRT",
            "TENSORRT_MAX_BATCH_SIZE",
            "YOLO_OBB_MODE",
            "YOLO_SEQ_CROP_PAD_RATIO",
            "YOLO_SEQ_MIN_CROP_SIZE_PX",
            "YOLO_SEQ_ENFORCE_SQUARE_CROP",
            "YOLO_SEQ_STAGE2_IMGSZ",
            "YOLO_SEQ_INDIVIDUAL_BATCH_SIZE",
            "YOLO_SEQ_STAGE2_POW2_PAD",
            "YOLO_HEADTAIL_CONF_THRESHOLD",
            "POSE_OVERRIDES_HEADTAIL",
        )
        cache_params = {
            "common": _extract(common_detection_keys),
            "yolo": _extract(yolo_inference_keys),
            "models": self._normalize_for_hash(
                {
                    "active_obb": model_fingerprint,
                    "direct_obb": self._get_model_fingerprint(direct_model),
                    "detect": self._get_model_fingerprint(
                        params.get("YOLO_DETECT_MODEL_PATH", "")
                    ),
                    "crop_obb": self._get_model_fingerprint(crop_obb_model),
                    "headtail": self._get_model_fingerprint(
                        params.get("YOLO_HEADTAIL_MODEL_PATH", "")
                    ),
                }
            ),
            "raw_detection_cache_version": 4,
        }
        classes = cache_params["yolo"].get("YOLO_TARGET_CLASSES")
        if classes is not None:
            if isinstance(classes, str):
                raw_classes = [c.strip() for c in classes.split(",") if c.strip()]
            elif isinstance(classes, (list, tuple)):
                raw_classes = list(classes)
            else:
                raw_classes = [classes]
            try:
                cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                    int(c) for c in raw_classes
                )
            except (TypeError, ValueError):
                cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                    str(c) for c in raw_classes
                )

        build_batch_size = params.get(
            "TENSORRT_BUILD_BATCH_SIZE",
            params.get("TENSORRT_MAX_BATCH_SIZE", 1),
        )
        try:
            build_batch_size = max(1, int(build_batch_size or 1))
        except (TypeError, ValueError):
            build_batch_size = max(
                1, int(params.get("TENSORRT_MAX_BATCH_SIZE", 1) or 1)
            )
        try:
            build_workspace_gb = float(params.get("TENSORRT_BUILD_WORKSPACE_GB", 4.0))
        except (TypeError, ValueError):
            build_workspace_gb = 4.0

        engine_cache_params = {
            "engine": {
                "runtime": "tensorrt",
                "device": self._normalize_for_hash(params.get("YOLO_DEVICE")),
                "build_batch_size": build_batch_size,
                "workspace_gb": round(max(0.5, build_workspace_gb), 3),
                "active_obb": model_fingerprint,
                "export_profile": "trt_fp16_static_v1",
            },
            "engine_cache_version": 1,
        }

        return {
            "inference": _build_id("yolo", cache_params, model_stem=safe_model_stem),
            "engine": _build_id(
                "yolo_engine", engine_cache_params, model_stem=safe_model_stem
            ),
        }

    def _setup_tracking_csv_writer(self, backward_mode):
        """Create and start the CSV writer thread for tracking output."""
        self._mw.csv_writer_thread = None
        if not self._panels.setup.csv_line.text():
            return
        save_confidence = self._panels.setup.check_save_confidence.isChecked()
        if save_confidence:
            hdr = [
                "TrackID",
                "TrajectoryID",
                "Index",
                "X",
                "Y",
                "Theta",
                "FrameID",
                "State",
                "DetectionConfidence",
                "AssignmentConfidence",
                "PositionUncertainty",
                "DetectionID",
                "IdentityAssignedID",
                "IdentityAssignedLabel",
                "IdentityAssignedConfidence",
                "IdentityPosteriorMargin",
                "IdentityEntropy",
                "IdentityCommitted",
                "IdentityEvidenceSources",
                "IdentityConflictFlag",
                "IdentitySlotLockLabel",
            ]
        else:
            hdr = [
                "TrackID",
                "TrajectoryID",
                "Index",
                "X",
                "Y",
                "Theta",
                "FrameID",
                "State",
                "DetectionID",
                "IdentityAssignedID",
                "IdentityAssignedLabel",
                "IdentityAssignedConfidence",
                "IdentityPosteriorMargin",
                "IdentityEntropy",
                "IdentityCommitted",
                "IdentityEvidenceSources",
                "IdentityConflictFlag",
                "IdentitySlotLockLabel",
            ]
        if self._mw._selected_identity_method() == "apriltags":
            hdr.extend(
                [
                    "DetectedTagID",
                    "DetectedTagLabel",
                    "DetectedTagConf",
                    "DetectedTagHamming",
                ]
            )
        csv_path = self._panels.setup.csv_line.text()
        base, ext = os.path.splitext(csv_path)
        if backward_mode:
            csv_path = f"{base}_backward{ext}"
        elif self._panels.tracking.chk_enable_backward.isChecked():
            csv_path = f"{base}_forward{ext}"
        from hydra_suite.data.csv_writer import CSVWriterThread

        self._mw.csv_writer_thread = CSVWriterThread(csv_path, header=hdr)
        self._mw.csv_writer_thread.start()

    def start_tracking_on_video(self: object, video_path, backward_mode=False):
        """start_tracking_on_video method documentation."""
        if self._mw.tracking_worker and self._mw.tracking_worker.isRunning():
            return
        if not self._panels.setup.csv_line.text().strip():
            QMessageBox.warning(
                self._mw,
                "No Output CSV",
                "Please set an output CSV path before starting tracking.\n\n"
                "A default path is set automatically when you load a video.",
            )
            return
        self._mw._stop_all_requested = False
        self._mw._pending_finish_after_interp = False
        if not backward_mode:
            self._mw._session_result_dataset = None
            self._mw._dataset_was_started = False
            self._mw._show_summary_on_dataset_done = False
            self._mw._session_wall_start = time.time()
            self._mw._session_final_csv_path = None
            self._mw._session_fps_list = []
            self._mw._session_frames_processed = 0

        if self._mw.is_playing:
            self._mw._stop_playback()

        self._mw._tracking_first_frame = True

        self._setup_tracking_csv_writer(backward_mode)

        # Video output is no longer generated during tracking
        # Instead, it's generated from post-processed trajectories after merging
        # This ensures the video shows clean, merged trajectories with stable IDs
        video_output_path = None

        # Generate detection cache path based on video and detection method
        # Cache is always created for forward tracking to allow reuse on reruns
        detection_cache_path = None
        params = self._mw.get_parameters_dict()
        logger.info(
            f"Launching {'backward' if backward_mode else 'forward'} tracking for frame range "
            f"{params.get('START_FRAME')}..{params.get('END_FRAME')}"
        )
        use_cached_detections = self._panels.setup.chk_use_cached_detections.isChecked()
        if not self._validate_yolo_model_requirements(params, mode_label="tracking"):
            return

        csv_dir = (
            os.path.dirname(self._panels.setup.csv_line.text())
            if self._panels.setup.csv_line.text()
            else ""
        )
        cache_plan = plan_tracking_cache(
            video_path,
            params=params,
            preferred_output_dir=csv_dir,
            use_cached_detections=use_cached_detections,
        )
        params["INFERENCE_MODEL_ID"] = cache_plan.inference_model_id
        if cache_plan.engine_model_id:
            params["ENGINE_MODEL_ID"] = cache_plan.engine_model_id
        detection_cache_path = cache_plan.detection_cache_path

        # Do NOT delete old detection caches; keep all for reuse
        self._mw.current_detection_cache_path = detection_cache_path

        from hydra_suite.core.tracking import TrackingWorker

        self._mw.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=self._mw.csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
            detection_cache_path=detection_cache_path,
            preview_mode=False,  # Full tracking mode - batching enabled if applicable
            use_cached_detections=use_cached_detections,
        )
        self._mw.tracking_worker.set_parameters(params)
        self._mw.parameters_changed.connect(self._mw.tracking_worker.update_parameters)
        self._mw.tracking_worker.frame_signal.connect(self.on_new_frame)
        self._mw.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self._mw.tracking_worker.progress_signal.connect(self.on_progress_update)
        self._mw.tracking_worker.stats_signal.connect(self.on_stats_update)
        self._mw.tracking_worker.warning_signal.connect(self.on_tracking_warning)
        self._mw.tracking_worker.pose_exported_model_resolved_signal.connect(
            self.on_pose_exported_model_resolved
        )

        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_label.setText(
            "Backward Tracking..." if backward_mode else "Forward Tracking..."
        )

        self._mw._prepare_tracking_display()
        self._mw._apply_ui_state("tracking")
        self._mw.tracking_worker.start()

    def _generate_training_dataset(self, override_csv_path=None):
        """Generate training dataset from tracking results for active learning."""
        try:
            from hydra_suite.trackerkit.gui.workers.dataset_worker import (
                DatasetGenerationWorker,
            )

            if self._mw._stop_all_requested:
                return
            logger.info("Starting training dataset generation...")

            # Prevent launching overlapping dataset threads; this can lead to
            # QThread destruction while still running if references are replaced.
            if (
                self._mw.dataset_worker is not None
                and self._mw.dataset_worker.isRunning()
            ):
                logger.warning(
                    "Dataset generation already in progress; skipping duplicate request."
                )
                return
            if (
                self._mw.dataset_worker is not None
                and not self._mw.dataset_worker.isRunning()
            ):
                self._mw.dataset_worker.deleteLater()
                self._mw.dataset_worker = None

            video_path = self._panels.setup.file_line.text()
            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(
                    self._mw,
                    "Dataset Generation Error",
                    "Source video file not found.",
                )
                return

            # Validate parameters
            # Auto-compute output directory
            output_dir = os.path.join(
                os.path.dirname(video_path),
                f"{os.path.splitext(os.path.basename(video_path))[0]}_datasets",
                "active_learning",
            )

            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(
                        self._mw,
                        "Dataset Generation Error",
                        f"Could not create output directory: {output_dir}\nError: {e}",
                    )
                    return

            # Use override path if provided (e.g. valid processed CSV), otherwise fallback to UI field
            csv_path = (
                override_csv_path
                if override_csv_path
                else self._panels.setup.csv_line.text()
            )

            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(
                    self._mw,
                    "Dataset Generation Error",
                    "Tracking CSV file not found.",
                )
                return

            # Get parameters
            params = self._mw.get_parameters_dict()
            max_frames = self._panels.dataset.spin_dataset_max_frames.value()
            diversity_window = (
                self._panels.dataset.spin_dataset_diversity_window.value()
            )
            include_context = (
                self._panels.dataset.chk_dataset_include_context.isChecked()
            )
            probabilistic = self._panels.dataset.chk_dataset_probabilistic.isChecked()

            # Get class name
            class_name = self._panels.dataset.line_dataset_class_name.text().strip()
            if not class_name:
                class_name = "object"

            # Show progress bar
            self._mw.progress_bar.setVisible(True)
            self._mw.progress_label.setVisible(True)
            self._mw.progress_bar.setValue(0)
            self._mw.progress_label.setText("Preparing dataset generation...")

            # Create and start dataset generation worker thread
            self._mw.dataset_worker = DatasetGenerationWorker(
                video_path=video_path,
                csv_path=csv_path,
                detection_cache_path=self._mw.current_detection_cache_path,
                output_dir=output_dir,
                dataset_name="",
                class_name=class_name,
                params=params,
                max_frames=max_frames,
                diversity_window=diversity_window,
                include_context=include_context,
                probabilistic=probabilistic,
            )
            self._mw.dataset_worker.progress_signal.connect(self.on_dataset_progress)
            self._mw.dataset_worker.finished_signal.connect(self.on_dataset_finished)
            self._mw.dataset_worker.error_signal.connect(self.on_dataset_error)
            self._mw.dataset_worker.finished.connect(
                self._on_dataset_worker_thread_finished
            )
            self._mw.dataset_worker.start()

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}", exc_info=True)
            QMessageBox.critical(
                self._mw,
                "Dataset Generation Error",
                f"Failed to generate dataset:\n{str(e)}",
            )

    def on_dataset_progress(self, value, message):
        """Update progress bar during dataset generation."""
        sender = None
        if (
            sender is not None
            and self._mw.dataset_worker is not None
            and sender is not self._mw.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._mw._stop_all_requested:
            return
        self._mw.progress_bar.setValue(value)
        self._mw.progress_label.setText(message)

    def on_dataset_finished(self: object, dataset_dir, num_frames):
        """Handle dataset generation completion."""
        sender = None
        if (
            sender is not None
            and self._mw.dataset_worker is not None
            and sender is not self._mw.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._mw._stop_all_requested:
            self._cleanup_thread_reference("dataset_worker")
            self._mw._refresh_progress_visibility()
            return
        self._mw._refresh_progress_visibility()

        logger.info(f"Dataset generation complete: {dataset_dir}")
        logger.info(f"Frames exported: {num_frames}")
        logger.info(
            "Use DetectKit from HYDRA Suite to review/correct detection datasets"
        )

        # Store result; popup is deferred to end-of-session summary.
        self._mw._session_result_dataset = {
            "success": True,
            "num_frames": num_frames,
            "dir": dataset_dir,
        }
        if getattr(self._mw, "_show_summary_on_dataset_done", False):
            self._mw._show_summary_on_dataset_done = False
            self._show_session_summary()

    def on_dataset_error(self, error_message):
        """Handle dataset generation errors."""
        sender = None
        if (
            sender is not None
            and self._mw.dataset_worker is not None
            and sender is not self._mw.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._mw._stop_all_requested:
            self._cleanup_thread_reference("dataset_worker")
            self._mw._refresh_progress_visibility()
            return
        self._mw._refresh_progress_visibility()

        logger.error(f"Dataset generation error: {error_message}")

        # Store result; popup is deferred to end-of-session summary.
        self._mw._session_result_dataset = {
            "success": False,
            "error": error_message,
        }
        if getattr(self._mw, "_show_summary_on_dataset_done", False):
            self._mw._show_summary_on_dataset_done = False
            self._show_session_summary()

    def _build_session_summary_lines(self) -> list[str]:
        """Build end-of-session summary lines for GUI and CLI consumers."""
        lines = []

        # --- Timing ---
        if self._mw._session_wall_start is not None:
            elapsed = time.time() - self._mw._session_wall_start
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            lines.append(f"Duration: {elapsed_str}")

        # --- Frames / FPS ---
        frames = self._mw._session_frames_processed
        if frames > 0:
            lines.append(f"Frames processed: {frames}")
        fps_vals = [f for f in self._mw._session_fps_list if f and f > 0]
        if fps_vals:
            avg_fps = sum(fps_vals) / len(fps_vals)
            lines.append(f"Average FPS: {avg_fps:.1f}")

        # --- Video / CSV ---
        video_path = self._panels.setup.file_line.text()
        if video_path:
            lines.append(f"Video: {os.path.basename(video_path)}")
        csv_path = (
            self._mw._session_final_csv_path or self._panels.setup.csv_line.text()
        )
        if csv_path:
            lines.append(f"Output CSV: {os.path.basename(csv_path)}")

        # --- Trajectory / track count ---
        if csv_path and os.path.exists(csv_path):
            try:
                _df = pd.read_csv(csv_path, usecols=["TrajectoryID"])
                n_trajs = int(_df["TrajectoryID"].nunique())
                lines.append(f"Trajectories: {n_trajs}")
            except Exception:
                pass

        # --- Pipelines run ---
        pipelines = []
        if self._panels.postprocess.enable_postprocessing.isChecked():
            pipelines.append("Post-processing")
        if self._panels.tracking.chk_enable_backward.isChecked():
            pipelines.append("Backward tracking")
        if self._mw._is_individual_pipeline_enabled():
            pipelines.append("Individual analysis")
            if self._panels.identity.chk_enable_pose_extractor.isChecked():
                pipelines.append("Pose extraction")
        if pipelines:
            lines.append("Pipelines: " + ", ".join(pipelines))

        # --- Separator before optional sub-results ---
        lines.append("")

        # --- Dataset generation result ---
        result = getattr(self._mw, "_session_result_dataset", None)
        if result is not None:
            if result.get("success"):
                lines.append(
                    f"\u2713 Dataset generated: {result['num_frames']} frame(s)"
                    f"\n  Location: {result['dir']}"
                )
            else:
                lines.append(
                    f"\u2717 Dataset generation failed: {result.get('error', 'unknown error')}"
                )

        return lines

    def _clear_session_summary_state(self) -> None:
        """Reset per-session summary state after reporting the session."""

        self._mw._session_result_dataset = None
        self._mw._dataset_was_started = False
        self._mw._show_summary_on_dataset_done = False

    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        lines = self._build_session_summary_lines()
        error_message = str(
            getattr(self._mw, "_headless_session_error", "") or ""
        ).strip()
        if error_message:
            lines.extend(["", f"Error: {error_message}"])

        # Clean up state
        self._clear_session_summary_state()

        if getattr(self._mw, "_headless_tracking_mode", False):
            callback = getattr(self._mw, "_headless_tracking_callback", None)
            if callable(callback):
                callback(
                    {
                        "success": not bool(error_message),
                        "lines": lines,
                        "error": error_message or None,
                        "video_path": self._panels.setup.file_line.text() or None,
                        "csv_path": self._mw._session_final_csv_path
                        or self._panels.setup.csv_line.text()
                        or None,
                    }
                )
            self._mw._headless_session_error = None
            return

        QMessageBox.information(self._mw, "Tracking Complete", "\n".join(lines))

        # Offer to open RefineKit for interactive proofreading after single-video runs.
        should_prompt_refinekit = (
            bool(self._mw.current_video_path)
            and not self._panels.setup.g_batch.isChecked()
            and self._panels.postprocess.chk_prompt_open_refinekit.isChecked()
        )
        if should_prompt_refinekit:
            reply = QMessageBox.question(
                self._mw,
                "Open RefineKit?",
                "Tracking complete. Open in RefineKit for "
                "interactive identity proofreading?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._mw._open_refinekit()

    def _on_dataset_worker_thread_finished(self):
        """Release completed dataset worker safely."""
        sender = None
        if (
            sender is not None
            and self._mw.dataset_worker is not None
            and sender is not self._mw.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("dataset_worker")
        self._mw._refresh_progress_visibility()

    def _validate_yolo_model_requirements(self, params: dict, mode_label: str) -> bool:
        """Validate YOLO mode-specific model requirements before starting runs."""
        if str(params.get("DETECTION_METHOD", "")) != "yolo_obb":
            return True
        yolo_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        if yolo_mode != "sequential":
            return True
        detect_model = str(params.get("YOLO_DETECT_MODEL_PATH", "")).strip()
        crop_obb_model = str(params.get("YOLO_CROP_OBB_MODEL_PATH", "")).strip()
        if detect_model and crop_obb_model:
            return True
        QMessageBox.warning(
            self._mw,
            "Missing Sequential Models",
            (
                f"Sequential YOLO OBB mode in {mode_label} requires both a detect model "
                "and a crop OBB model."
            ),
        )
        return False

    def _get_detection_size(self, detection_cache, frame_id, detection_id, params):
        """Get physical size (w, h) of a detection from cache."""
        import math as _math

        import numpy as _np
        import pandas as _pd

        if detection_cache is None or detection_id is None or _pd.isna(detection_id):
            return None, None
        try:
            _, _, shapes, _, obb_corners, detection_ids, *_ = detection_cache.get_frame(
                int(frame_id)
            )
        except Exception:
            return None, None

        idx = None
        try:
            for i, did in enumerate(detection_ids):
                if int(did) == int(detection_id):
                    idx = i
                    break
        except Exception:
            idx = None

        if idx is None:
            return None, None

        if obb_corners and idx < len(obb_corners):
            c = _np.asarray(obb_corners[idx], dtype=_np.float32)
            if c.shape[0] >= 4:
                w = float(_np.linalg.norm(c[1] - c[0]))
                h = float(_np.linalg.norm(c[2] - c[1]))
                if w < h:
                    w, h = h, w
                return w, h

        if shapes and idx < len(shapes):
            area, aspect_ratio = shapes[idx][0], shapes[idx][1]
            if aspect_ratio > 0 and area > 0:
                ax2 = _math.sqrt(4 * area / (_math.pi * aspect_ratio))
                ax1 = aspect_ratio * ax2
                return ax1, ax2

        return None, None
