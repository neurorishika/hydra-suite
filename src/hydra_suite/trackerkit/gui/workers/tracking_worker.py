"""Qt wrapper for the Qt-free TrackingEngineCore (core/tracking/worker.py).

Owns the QThread + the 6 Signals + the update_parameters Slot, and forwards the
engine's callbacks to those signals. Keeps the exact interface the tracking
orchestrator and the headless CLI depend on."""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot

from hydra_suite.core.tracking.worker import TrackingEngineCore

logger = logging.getLogger(__name__)


class TrackingWorker(QThread):
    frame_signal = Signal(np.ndarray)
    finished_signal = Signal(bool, list, list)
    progress_signal = Signal(int, str)
    stats_signal = Signal(dict)
    warning_signal = Signal(str, str)
    pose_exported_model_resolved_signal = Signal(str)

    def __init__(
        self,
        video_path,
        csv_writer_thread=None,
        video_output_path=None,
        backward_mode=False,
        detection_cache_path=None,
        preview_mode=False,
        use_cached_detections=False,
        parent=None,
    ):
        super().__init__(parent)
        self._core = TrackingEngineCore(
            video_path,
            csv_writer_thread=csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
            detection_cache_path=detection_cache_path,
            preview_mode=preview_mode,
            use_cached_detections=use_cached_detections,
            on_frame=lambda rgb: self.frame_signal.emit(rgb),
            on_finished=lambda ok, fps, traj: self.finished_signal.emit(ok, fps, traj),
            on_progress=lambda pct, msg: self.progress_signal.emit(int(pct), msg),
            on_stats=lambda stats: self.stats_signal.emit(stats),
            on_warning=lambda title, msg: self.warning_signal.emit(title, msg),
            on_pose_model_resolved=lambda p: self.pose_exported_model_resolved_signal.emit(
                p
            ),
        )

    # --- delegation: keep the exact public surface consumers use ---
    def set_parameters(self, p: dict) -> None:
        self._core.set_parameters(p)

    @Slot(dict)
    def update_parameters(self, new_params: dict) -> None:
        self._core.update_parameters(new_params)

    def get_current_params(self) -> dict:
        return self._core.get_current_params()

    def stop(self) -> None:
        self._core.stop()

    @property
    def _stop_requested(self) -> bool:  # some call sites / tests read this
        return self._core._stop_requested

    # --- straggler proxies: fields the tracking orchestrator reads directly
    # off the worker instance via getattr/hasattr (see gui/orchestrators/tracking.py) ---
    @property
    def backward_mode(self) -> bool:
        return self._core.backward_mode

    @property
    def individual_properties_cache_path(self):
        return self._core.individual_properties_cache_path

    @property
    def detected_properties_cache_path(self):
        return self._core.detected_properties_cache_path

    @property
    def detected_cnn_cache_paths(self):
        return self._core.detected_cnn_cache_paths

    def run(self) -> None:
        """QThread entry point. PySide6 silently swallows exceptions that escape
        a QThread.run() override, which would leave finished_signal unemitted and
        hang callers blocked on it (headless CLI's QEventLoop). Guard exactly as
        the old TrackingWorker.run() did."""
        try:
            self._core.run_tracking()
        except Exception:
            logger.exception(
                "Unhandled exception in TrackingWorker.run(); emitting "
                "finished_signal(False, ...) so callers waiting on it "
                "(e.g. the headless CLI's QEventLoop) don't hang forever."
            )
            self.finished_signal.emit(False, [], [])
