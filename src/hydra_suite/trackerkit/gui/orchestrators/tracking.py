"""TrackingOrchestrator — run→merge→export→finalize lifecycle."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox

from hydra_suite.trackerkit.cli_config import legacy_detection_runtime_fields
from hydra_suite.trackerkit.gui.orchestrators.config import _get_video_config_path
from hydra_suite.trackerkit.gui.workers.session_worker import SessionWorker
from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header
from hydra_suite.trackerkit.session_plan import resolve_video_plan
from hydra_suite.trackerkit.tracking_cache import (
    plan_tracking_cache,
    resolve_detection_cache_runtime,
)
from hydra_suite.utils.video_artifacts import (
    build_inference_cache_dir,
    candidate_artifact_base_dirs,
)

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)

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


# A fan-out's cooperative stop is bounded by _stop_children: sigint_grace +
# term_grace GLOBALLY, then up to 5 s waiting on each child and 5 s joining each
# log reader. A fixed 25 s ceiling is therefore blown by three stubborn children.
_FANOUT_STOP_PER_JOB_S = 10.0
_FANOUT_STOP_CEILING_S = 120.0


def fanout_stop_timeout_ms(n_jobs: int, options) -> int:
    """Join budget for stopping a fan-out, scaled by the number of jobs.

    Capped so a wedged worker can never hang the GUI indefinitely; the caller
    must handle the "still running when the budget expired" case rather than
    proceeding into teardown, which would orphan the children.
    """
    sigint = float(getattr(options, "sigint_grace_s", 10.0) or 0.0)
    term = float(getattr(options, "term_grace_s", 5.0) or 0.0)
    budget = sigint + term + _FANOUT_STOP_PER_JOB_S * max(1, int(n_jobs))
    return int(min(budget, _FANOUT_STOP_CEILING_S) * 1000)


class TrackingOrchestrator:
    """Owns the tracking lifecycle: start, stop, merge, export, finalize."""

    def __init__(
        self, main_window: "MainWindow", config: "TrackerConfig", panels
    ) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels
        self._calibration_dialog = None

    def _tracking_is_active(self) -> bool:
        """True while a track (single or fanned-out batch) is running."""
        worker = getattr(self._mw, "tracking_worker", None)
        fanout_worker = getattr(self._mw, "batch_fanout_worker", None)
        return bool(worker and worker.isRunning()) or bool(
            fanout_worker is not None and fanout_worker.isRunning()
        )

    def _calibration_is_active(self) -> bool:
        """True while a calibration search thread is still executing."""
        dialog = self._calibration_dialog
        if dialog is None:
            return False
        try:
            return bool(dialog.is_calibration_running())
        except RuntimeError:
            # Underlying C++ dialog already deleted -> nothing is running.
            self._calibration_dialog = None
            return False

    def open_calibration_dialog(self) -> None:
        """Open the one-click Calibrate dialog.

        Calibration and tracking are mutually exclusive: on MPS/CPU the
        autotune contention probe hardcodes ``contention_detected=False``
        (``core/inference/autotune/device.py``), so nothing below the GUI
        can detect an overlapping tracking run, and a calibration measured
        against one would silently produce a confidently wrong profile.
        Refuse to open while a track is running; the dialog itself is
        application-modal, so it symmetrically blocks starting a track
        while calibration is in progress.
        """
        if self._tracking_is_active():
            QMessageBox.warning(
                self._mw,
                "Tracking in progress",
                "Calibration cannot run while a tracking run is active. "
                "Stop tracking first, then calibrate.",
            )
            return

        video_path = str(self._mw.current_video_path or "").strip()
        if not video_path:
            QMessageBox.warning(
                self._mw, "No video loaded", "Load a video before calibrating."
            )
            return

        from hydra_suite.core.inference.config import build_inference_config_from_params
        from hydra_suite.trackerkit.calibrate_cli import derive_context_inputs
        from hydra_suite.trackerkit.gui.dialogs.calibration import (
            CalibrationDialog,
            describe_calibration_outcome,
        )

        params = self._mw.get_parameters_dict()
        inference_config = build_inference_config_from_params(params)

        class _FrameCountProbe:
            total_frames = int(getattr(self._mw, "video_total_frames", 0) or 0)

        inputs = derive_context_inputs(
            video_path,
            params,
            _FrameCountProbe(),
            use_cached_detections=bool(
                self._panels.setup.chk_use_cached_detections.isChecked()
            ),
        )

        dialog = CalibrationDialog(
            params=params,
            config=inference_config,
            video_path=video_path,
            frame_width=int(getattr(self._mw, "video_width", None) or 1),
            frame_height=int(getattr(self._mw, "video_height", None) or 1),
            start_frame=inputs.start_frame,
            end_frame=inputs.end_frame,
            realtime=inputs.realtime,
            cache_dir=inputs.cache_dir,
            use_cached_detections=inputs.use_cached_detections,
            default_budget_seconds=float(
                self._config.inference_autotune_budget_seconds
            ),
            parent=self._mw,
        )
        # Kept past exec() on purpose: a cancelled worker may outlive the
        # dialog (reject() waits only 5s), and _calibration_is_active must
        # still see it so start_full keeps refusing.
        self._calibration_dialog = dialog
        self._mw.btn_start.setEnabled(False)
        try:
            dialog.exec()
        finally:
            self._mw.btn_start.setEnabled(True)

        result = dialog.result_payload
        if result is not None:
            self._panels.setup.set_inference_autotune_status(
                describe_calibration_outcome(result)
            )

    def start_full(self):
        """start_full method documentation."""
        # Calibration and tracking are mutually exclusive. The MPS/CPU
        # contention probe hardcodes contention_detected=False, so an
        # overlapping track would silently corrupt the measurement.
        if self._calibration_is_active():
            QMessageBox.warning(
                self._mw,
                "Calibration in progress",
                "Tracking cannot start while an inference calibration is "
                "running. Wait for it to finish, then start tracking.",
            )
            self._mw.btn_start.setChecked(False)
            return

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
            running = worker.isRunning()
        except Exception:
            logger.debug(
                "requesting stop of %s (isRunning=<error checking>)", worker_name
            )
            return
        logger.debug("requesting stop of %s (isRunning=%s)", worker_name, running)
        if not running:
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
        session_worker = getattr(self._mw, "session_worker", None)
        try:
            session_worker_running = (
                bool(session_worker.isRunning())
                if session_worker is not None
                else False
            )
        except Exception:
            session_worker_running = False
        logger.debug(
            "stop_tracking() invoked: _stop_all_requested set True; "
            "session_worker running=%s",
            session_worker_running,
        )
        self._mw._stop_all_requested = True
        self._mw._pending_pose_export_csv_path = None
        self._mw._pending_video_csv_path = None
        self._mw._pending_video_generation = False

        # The fan-out's children are processes, not threads: cancelling walks
        # them through SIGINT -> SIGTERM -> SIGKILL with grace periods, so it
        # gets a long cooperative timeout and is never force-terminated (that
        # would abandon live children). The worker reference itself is dropped
        # by the QThread.finished handler, not here.
        fanout_worker = getattr(self._mw, "batch_fanout_worker", None)
        if fanout_worker is not None and fanout_worker.isRunning():
            fanout_worker.cancel()
            self._request_qthread_stop(
                fanout_worker,
                "BatchFanoutWorker",
                timeout_ms=fanout_stop_timeout_ms(
                    len(getattr(fanout_worker, "specs", ()) or ()),
                    getattr(fanout_worker, "options", None),
                ),
                force_terminate=False,
            )

        # Stop all active workers and subprocess-like threads.
        self._request_qthread_stop(
            getattr(self._mw, "_cache_builder_worker", None),
            "DetectionCacheBuildWorker",
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
        self._request_qthread_stop(
            self._mw.tracking_worker,
            "TrackingWorker",
            timeout_ms=10000,
            force_terminate=False,
        )
        self._request_qthread_stop(
            getattr(self._mw, "session_worker", None),
            "SessionWorker",
            # SessionWorker runs Python/pandas work. Killing its QThread
            # asynchronously can leave native resources in an invalid state;
            # its merge path polls should_stop cooperatively instead.
            timeout_ms=3000,
            force_terminate=False,
        )
        self._stop_csv_writer()

        self._cleanup_thread_reference("_cache_builder_worker")
        self._cleanup_thread_reference("merge_worker")
        self._cleanup_thread_reference("postprocess_worker")
        self._cleanup_thread_reference("dataset_worker")
        self._cleanup_thread_reference("interp_worker")
        self._cleanup_thread_reference("final_media_export_worker")
        self._cleanup_thread_reference("preview_detection_worker")
        self._cleanup_thread_reference("tracking_worker")
        self._cleanup_thread_reference("session_worker")

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
        found: dict[str, Path] = {}
        for base_dir in artifact_base_dirs:
            cache_dir = build_inference_cache_dir(
                video_path, artifact_base_dir=base_dir
            )
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
        tuning = stats.get("inference_autotune")
        if isinstance(tuning, dict):
            status = str(tuning.get("status", "unknown")).replace("_", " ")
            reason = str(tuning.get("reason", "")).strip()
            effective = tuning.get("effective", {})
            values = ""
            if isinstance(effective, dict):
                compact = ", ".join(
                    f"{name}={value}"
                    for name, value in effective.items()
                    if value not in (None, {}, [], ())
                )
                values = f" Effective: {compact}." if compact else ""
            profile = str(tuning.get("profile_id") or "").strip()
            profile_text = f" Profile {profile}." if profile else ""
            self._panels.setup.set_inference_autotune_status(
                f"{status.capitalize()} — {reason}.{profile_text}{values}"
            )
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
        self._mw.video_label.set_shapes([])
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
        self._mw._set_video_pixmap(QPixmap.fromImage(scaled), already_scaled=True)

        # Auto-fit to screen on first frame of tracking
        if self._mw._tracking_first_frame:
            self._mw._tracking_first_frame = False
            # Use QTimer to ensure frame is displayed first
            from PySide6.QtCore import QTimer

            QTimer.singleShot(50, self._mw._fit_image_to_screen)

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

    def _collect_worker_props_path(self):
        """Read export-relevant cache paths from tracking_worker and store them."""
        worker_props_path = ""
        worker_detected_props_path = ""
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
        QMessageBox.warning(
            self._mw,
            "Tracking Failed",
            "An error occurred during tracking. Check logs for details.",
        )
        if self._panels.setup.g_batch.isChecked():
            self._mw.current_batch_index = -1
            logger.info("Batch mode aborted due to error.")
        self._finalize_tracking_session_ui()

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

        if is_backward_mode:
            self._run_session_worker()
        elif is_backward_enabled:
            self.start_backward_tracking()
        else:
            self._run_session_worker()

    def _run_session_worker(self) -> None:
        """Drive all post-tracking through the Qt-free core service.

        Both tracking passes have already written the raw CSV(s) to disk; the
        service re-reads them and does postprocess/merge/rich-export/interp/
        media/dataset — replacing the old _finish_tracking_session chain.
        """
        raw_csv_path = self._panels.setup.csv_line.text()
        video_path = self._panels.setup.file_line.text()
        paths = {
            "raw_csv_path": raw_csv_path,
            "detection_cache_path": getattr(
                self._mw, "current_detection_cache_path", None
            ),
            "individual_properties_cache_path": getattr(
                self._mw, "current_individual_properties_cache_path", None
            ),
            "detected_properties_cache_path": getattr(
                self._mw, "current_detected_properties_cache_path", None
            ),
        }
        worker = SessionWorker(
            video_path=video_path,
            config=self._build_session_config(),
            params=self._mw.get_parameters_dict(),
            paths=paths,
        )
        self._mw.session_worker = worker
        worker.progress_signal.connect(self._on_session_progress)
        worker.warning_signal.connect(self.on_tracking_warning)
        worker.error_signal.connect(self._on_session_error)
        worker.finished_signal.connect(self._on_session_finished)
        worker.start()

    def _build_session_config(self):
        """Return the GUI's canonical config dict for TrackingSessionCore."""
        return self._mw._config_orch.build_config_dict()

    def _on_session_progress(self, value: int, message: str) -> None:
        if self._mw._stop_all_requested:
            return
        # Mirror the progress-bar update the deleted post-workers did.
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_bar.setValue(int(value))
        self._mw.progress_label.setVisible(True)
        self._mw.progress_label.setText(str(message))

    def _on_session_error(self, message: str) -> None:
        if self._mw._stop_all_requested:
            return
        QMessageBox.critical(
            self._mw,
            "Post-Processing Error",
            f"Error during trajectory post-processing:\n{message}",
        )
        logger.error("Session post-processing error: %s", message)
        self._finalize_tracking_session_ui()

    def _on_session_finished(self, result) -> None:
        if self._mw._stop_all_requested:
            return
        if not getattr(result, "success", False):
            QMessageBox.critical(
                self._mw,
                "Post-Processing Error",
                f"Error during trajectory post-processing:\n{result.error or 'unknown error'}",
            )
            self._finalize_tracking_session_ui()
            return
        self._mw._session_final_csv_path = result.final_csv_path
        self._mw._session_summary_lines = list(result.summary_lines or [])
        self._finalize_tracking_session_ui()

    def on_postprocess_error(self, error_message):
        """Handle post-processing errors."""
        self._cleanup_thread_reference("postprocess_worker")
        if self._mw._stop_all_requested:
            self._mw._refresh_progress_visibility()
            return
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        QMessageBox.critical(
            self._mw,
            "Post-Processing Error",
            f"Error during trajectory post-processing:\n{error_message}",
        )
        logger.error(f"Trajectory post-processing error: {error_message}")

    def _finalize_tracking_session_ui(self):
        """Finalize session cleanup and return UI to idle state."""
        self._mw._pending_pose_export_csv_path = None
        self._mw._pending_video_csv_path = None
        self._mw._pending_video_generation = False
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

            # Show end-of-session summary now. (Deferring the summary until an
            # in-flight dataset worker finished was handled by on_dataset_finished,
            # which was retired with the deleted GUI post-tracking chain.)
            self._show_session_summary()
        else:
            logger.info("✓ Video complete. Continuing batch...")

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
                # NOTE: _setup_video_file unconditionally clears the ROI/arena
                # state near its top (clear_roi()), so a keystone-config load
                # must happen AFTER it, not before -- loading first would just
                # be wiped out again by the clear inside _setup_video_file.
                # (Confirmed via a live repro; see Fix Wave 16 and Fix Wave 17
                # reports for the root cause and verification.)
                self._mw._setup_video_file(
                    fp,
                    skip_config_load=plan.use_keystone_baseline
                    or not plan.has_own_config,
                )
                if plan.use_keystone_baseline and plan.config_path:
                    self._mw._config_orch._load_config_from_file(plan.config_path)

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
                    parallel = self._batch_parallel_requested()
                    mode_word = (
                        "in parallel across GPUs" if parallel else "sequentially"
                    )
                    # The parallel path never writes a per-video config file:
                    # each child is handed the keystone's parameters (or the
                    # video's own saved config) directly, so promising a
                    # configuration file per video would be a lie.
                    detail = (
                        "Each video runs as its own child process using the CURRENT "
                        "parameters (the keystone's config) or its own saved config; "
                        "a CSV is written next to each video."
                        if parallel
                        else "Each video will have its own CSV and configuration file "
                        "saved in its source directory."
                    )
                    res = QMessageBox.question(
                        self._mw,
                        "Start Batch Process",
                        f"This will process {len(self._mw.batch_videos)} videos {mode_word} using the CURRENT parameters.\n\n"
                        f"{detail}\n\n"
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

            if self._batch_parallel_requested():
                # Parallel fan-out replaces the sequential re-entrant batch loop
                # entirely: children are headless CLI processes. current_batch_index
                # goes back to -1 so _finalize_tracking_session_ui's continuation
                # branch can never fire for this run.
                self._mw.current_batch_index = -1
                self.start_batch_fanout()
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

    # ------------------------------------------------------------------
    # Parallel batch fan-out (one child process per GPU)
    # ------------------------------------------------------------------

    def _batch_parallel_requested(self) -> bool:
        """True when Batch AND its Parallel checkbox are both on."""
        setup = self._panels.setup
        if not setup.g_batch.isChecked():
            return False
        checkbox = getattr(setup, "chk_batch_parallel", None)
        return checkbox is not None and bool(checkbox.isChecked())

    def _resolve_fanout_options(self):
        """Build ``FanoutOptions`` from the mirrored Batch › Parallel state.

        Returns ``None`` when the user's GPU selector cannot be resolved (the
        warning has already been shown).
        """
        from hydra_suite.runtime.cuda_devices import (
            list_cuda_devices,
            parse_gpu_selectors,
        )
        from hydra_suite.trackerkit.batch_fanout import (
            FanoutOptions,
            decide_gpu_slots,
            host_has_cuda,
        )

        # The GPU line edit only mirrors into config on editingFinished, so a
        # value typed and left focused would otherwise be silently ignored.
        try:
            self._mw._on_batch_parallel_changed()
        except Exception:
            logger.debug("Could not refresh batch-parallel config.", exc_info=True)
        cfg = self._mw.config

        # One nvidia-smi shell-out, not two: this runs on the GUI thread.
        # The decision itself is the CLI's, so the two paths cannot drift: an
        # empty device list used to be swallowed here and the batch ran with N
        # UNPINNED children, all of which pin cuda:0.
        try:
            devices = decide_gpu_slots(
                parse_gpu_selectors(cfg.batch_parallel_gpus or "auto"),
                list_cuda_devices(),
                host_has_cuda(),
            )
        except ValueError as exc:
            QMessageBox.warning(self._mw, "GPU selection", str(exc))
            return None

        requested = int(cfg.batch_parallel_jobs or 0)
        if devices and requested <= 0:
            # "Unspecified": the scheduler gives one slot per selected GPU.
            jobs = None
        else:
            jobs = max(1, requested or 1)
            if devices:
                jobs = min(jobs, len(devices))
        return FanoutOptions(gpus=devices, jobs=jobs, log_level="INFO")

    def start_batch_fanout(self) -> bool:
        """Run the whole batch as headless child processes, one per slot."""
        from hydra_suite.trackerkit.batch_plan import BatchPlanError, plan_batch_jobs
        from hydra_suite.trackerkit.gui.dialogs.batch_fanout_dialog import (
            BatchFanoutDialog,
        )
        from hydra_suite.trackerkit.gui.workers.batch_fanout_worker import (
            BatchFanoutWorker,
        )

        active = getattr(self._mw, "batch_fanout_worker", None)
        if active is not None and active.isRunning():
            logger.warning(
                "Parallel batch fan-out is already running; ignoring the start request."
            )
            return False

        setup = self._panels.setup
        videos = list(self._mw.batch_videos)
        # The CLI refuses a missing video up front (cli.run_tracking_cli); without
        # the same check here a stale batch entry fails INSIDE a child, halting
        # the batch with the cause buried in a per-job log.
        missing = [video for video in videos if not Path(video).is_file()]
        if missing:
            QMessageBox.warning(
                self._mw,
                "Batch cannot start",
                "These batch videos no longer exist:\n\n" + "\n".join(missing),
            )
            return False
        try:
            # The keystone's sidecar was just written by save_config(), so the
            # planner sees exactly what `trackerkit track --video-list` would.
            specs = plan_batch_jobs(
                videos, keystone_override=setup.chk_batch_keystone_override.isChecked()
            )
        except BatchPlanError as exc:
            QMessageBox.warning(self._mw, "Batch cannot start", str(exc))
            return False

        options = self._resolve_fanout_options()
        if options is None:
            return False
        # Mirror run_batch_fanout's own slot arithmetic so the status line
        # cannot claim more slots than the scheduler will actually open.
        if options.gpus:
            requested = len(options.gpus) if options.jobs is None else int(options.jobs)
            slots = max(1, min(requested, len(options.gpus)))
        else:
            slots = max(1, int(options.jobs or 1))

        # Never leave a previous run's table parented to the window forever.
        previous = getattr(self._mw, "batch_fanout_dialog", None)
        if previous is not None:
            try:
                previous.close()
                previous.deleteLater()
            except Exception:
                logger.debug(
                    "Could not release previous fan-out dialog.", exc_info=True
                )
            self._mw.batch_fanout_dialog = None

        worker = BatchFanoutWorker(specs, options, parent=self._mw)
        dialog = BatchFanoutDialog(specs, parent=self._mw)
        worker.job_started.connect(dialog.on_job_started)
        worker.job_progress.connect(dialog.on_job_progress)
        worker.job_finished.connect(dialog.on_job_finished)
        worker.fanout_finished.connect(dialog.on_fanout_finished)
        worker.fanout_finished.connect(self._on_batch_fanout_finished)
        worker.error.connect(dialog.on_fanout_error)
        worker.error.connect(self._on_batch_fanout_error)
        # QThread.finished fires AFTER run() returns, which fanout_finished
        # (emitted from inside execute()) does not: only here is isRunning()
        # guaranteed false, so only here can the reference actually be dropped.
        worker.finished.connect(self._on_batch_fanout_thread_finished)
        dialog.cancel_requested.connect(worker.cancel)
        self._mw.batch_fanout_worker = worker
        self._mw.batch_fanout_dialog = dialog
        # A new batch gets a fresh close budget: the "close again to force-kill"
        # escalation must not carry over from a previous, unrelated fan-out.
        self._mw._fanout_close_warned = False

        self._mw._stop_all_requested = False
        self._mw.btn_start.setText("Stop Tracking")
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_label.setVisible(True)
        # Busy indicator; the per-job bars live in the dialog.
        self._mw.progress_bar.setRange(0, 0)
        self._mw.progress_label.setText(
            f"Parallel batch: {len(specs)} videos on {slots} slot(s)…"
        )
        self._mw._apply_ui_state("tracking")
        dialog.show()
        worker.start()
        logger.info(
            "Parallel batch fan-out started: %d videos, %d slot(s)", len(specs), slots
        )
        return True

    def _on_batch_fanout_error(self, message: str) -> None:
        logger.error("Batch fan-out worker error: %s", message)
        self._restore_after_fanout()
        QMessageBox.critical(self._mw, "Parallel batch failed", message)

    def _on_batch_fanout_finished(self, result) -> None:
        ok = sum(1 for job in result.jobs if job.success)
        logger.info(
            "Parallel batch finished: %d/%d succeeded (cancelled=%s)",
            ok,
            len(result.jobs),
            result.cancelled,
        )
        self._restore_after_fanout()
        if not result.cancelled:
            QMessageBox.information(
                self._mw,
                "Batch Complete",
                f"{ok}/{len(result.jobs)} videos succeeded.\n"
                "Per-video logs are in each video's <stem>_logs/ folder.",
            )

    def _on_batch_fanout_thread_finished(self) -> None:
        self._cleanup_thread_reference("batch_fanout_worker")

    def _restore_after_fanout(self) -> None:
        """Return the main window to idle. Idempotent: stop_tracking may race it."""
        self._mw.progress_bar.setRange(0, 100)
        self._mw.progress_bar.setValue(0)
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        self._mw.progress_label.setText("Ready")
        self._mw._cleanup_session_logging()
        self._mw._set_ui_controls_enabled(True)
        self._mw.btn_start.blockSignals(True)
        self._mw.btn_start.setChecked(False)
        self._mw.btn_start.blockSignals(False)
        self._mw.btn_start.setText("Start Full Tracking")
        self._mw._apply_ui_state("idle" if self._mw.current_video_path else "no_video")
        self._mw.current_batch_index = -1

    def start_preview_on_video(self, video_path):
        """start_preview_on_video method documentation."""
        from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker

        if self._mw.tracking_worker and self._mw.tracking_worker.isRunning():
            return
        self._mw._stop_all_requested = False

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
        if not self._validate_identity_requirements("tracking preview"):
            return
        if not self._validate_arena_overlaps("tracking preview"):
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
        # Preview must not build exported accelerator engines (ONNX/TensorRT/
        # CoreML). Backend selection is driven by RUNTIME_TIER (already carried
        # in params); the retired COMPUTE_RUNTIME string family no longer needs
        # sanitizing here (Runtime Gen-2 FT1). We still downgrade the auxiliary
        # detection fields (owned by later slices) to their native-device
        # equivalents, deriving the pre-downgrade runtime from the selected tier
        # instead of the removed COMPUTE_RUNTIME param. The pose runtime is fully
        # tier-derived downstream (Runtime Gen-2 FT2), so no pose-flavor param is
        # threaded here.
        resolved_obb = self._mw._resolved_obb_backend()
        if resolved_obb.backend in ("tensorrt", "coreml"):
            import dataclasses

            safe = dataclasses.replace(resolved_obb, backend="torch")
            safe_det = legacy_detection_runtime_fields(safe)
            params["YOLO_DEVICE"] = safe_det["yolo_device"]
            params["ENABLE_GPU_BACKGROUND"] = safe_det["enable_gpu_background"]
            params["ENABLE_TENSORRT"] = safe_det["enable_tensorrt"]
            params["ENABLE_ONNX_RUNTIME"] = safe_det["enable_onnx_runtime"]

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
        _compute_runtime = resolve_detection_cache_runtime(params)

        def _extract(keys):
            return {
                k: self._normalize_for_hash(
                    _compute_runtime if k == "COMPUTE_RUNTIME" else params.get(k)
                )
                for k in keys
            }

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
        hdr = build_tracking_csv_header(
            identity_method=self._mw._selected_identity_method(),
            n_arenas=int(self._mw.get_parameters_dict().get("N_ARENAS", 1)),
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
        # A stop requested during the forward->backward handoff must abort the
        # backward pass (the flag is only reset for a fresh forward run, below).
        if backward_mode and self._mw._stop_all_requested:
            return
        if not self._panels.setup.csv_line.text().strip():
            QMessageBox.warning(
                self._mw,
                "No Output CSV",
                "Please set an output CSV path before starting tracking.\n\n"
                "A default path is set automatically when you load a video.",
            )
            return
        if not backward_mode:
            self._mw._stop_all_requested = False
            self._mw._session_result_dataset = None
            self._mw._dataset_was_started = False
            self._mw._session_wall_start = time.time()
            self._mw._session_final_csv_path = None
            self._mw._session_fps_list = []
            self._mw._session_frames_processed = 0
            self._mw._session_summary_lines = []

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
        if not self._validate_identity_requirements("tracking"):
            return
        if not self._validate_arena_overlaps("tracking"):
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

        from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker

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

    def _clear_session_summary_state(self) -> None:
        """Reset per-session summary state after reporting the session."""

        self._mw._session_result_dataset = None
        self._mw._dataset_was_started = False

    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        lines = getattr(self._mw, "_session_summary_lines", [])

        # Clean up state
        self._clear_session_summary_state()

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

    def _validate_identity_requirements(self, mode_label: str) -> bool:
        """Refuse to start when identity is on but no evidence source exists.

        ``_selected_identity_method()`` silently collapses to ``none_disabled``
        in that state, so the run would complete having done no identity work
        while the UI still claimed identity classification was enabled.
        """
        if not self._mw._is_identity_analysis_enabled():
            return True
        if self._mw._has_identity_source():
            return True
        QMessageBox.warning(
            self._mw,
            "No Identity Source Configured",
            (
                f"Identity classification is enabled, but {mode_label} has no "
                "identity source to use.\n\n"
                "Enable AprilTags, or add a CNN classifier and select a model "
                "for it, in Analyze Individuals — or turn off Enable Identity "
                "Classification."
            ),
        )
        return False

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

    def _validate_arena_overlaps(self, mode_label: str) -> bool:
        """Refuse to start while any two arenas share a pixel, or any single
        arena's shapes form disconnected regions.

        Overlapping arenas are not merely a UI blemish:
        ``engine_params.build_arena_labels`` resolves them by last-writer-wins
        in shape draw order, silently, so an animal in the shared region is
        assigned to whichever arena happened to rasterize last. Disconnected
        regions are refused for the same reason Fix Wave 18 introduced the
        check: two separate physical regions sharing one arena_id makes
        cross-region tracks possible, which is never the intended semantics.
        """
        allowed, reason = self._mw.arena_panel.can_track()
        if allowed:
            return True
        title = (
            f"{mode_label}: Overlapping Arenas"
            if self._mw.arena_panel.blocking_pairs()
            else f"{mode_label}: Disconnected Arena Regions"
        )
        QMessageBox.warning(self._mw, title, reason)
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
