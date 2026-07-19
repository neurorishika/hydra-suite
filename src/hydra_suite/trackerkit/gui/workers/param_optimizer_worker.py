"""Qt wrappers for the pure tracking parameter-optimizer helpers (which live
Qt-free in ``core/tracking/optimization/optimizer_workers``).

DetectionCacheBuildWorker — builds an InferenceRunner detection cache for a
    frame range via ``InferenceRunner.run_batch_pass``.
TrackingPreviewWorker — emits preview frames using cached detections by
    delegating to ``run_tracking_preview`` and translating its frame/stop
    callbacks into Qt signals.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PySide6.QtCore import QThread, Signal

from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.core.inference.runner import InferenceRunner
from hydra_suite.core.tracking.optimization.optimizer_workers import (
    run_tracking_preview,
)

logger = logging.getLogger(__name__)


class DetectionCacheBuildWorker(QThread):
    """Phase-1-only worker: runs InferenceRunner.run_batch_pass over a frame
    range to populate an InferenceRunner detection cache for the Bayesian
    optimizer. No Kalman/CSV/pose stages.
    """

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)  # (success, cache_dir)

    def __init__(
        self,
        video_path: str,
        cache_dir: str,
        params: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_dir = cache_dir
        self.params = params.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            cfg = build_inference_config_from_params(self.params)
            runner = InferenceRunner(
                cfg, cache_dir=Path(self.cache_dir), video_path=self.video_path
            )
        except Exception as e:
            logger.error("DetectionCacheBuild: could not build runner: %s", e)
            self.finished_signal.emit(False, "")
            return
        try:

            def _progress_cb(processed, range_total):
                pct = int(processed * 100 / range_total) if range_total else 0
                self.progress_signal.emit(pct, f"Building detection cache: {pct}%")

            runner.run_batch_pass(
                Path(self.video_path),
                progress_cb=_progress_cb,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                should_stop=lambda: self._stop_requested,
            )
            if self._stop_requested:
                self.progress_signal.emit(0, "Cancelled.")
                self.finished_signal.emit(False, "")
                return
            logger.info("DetectionCacheBuild: cache saved to %s", self.cache_dir)
            self.finished_signal.emit(True, str(self.cache_dir))
        except Exception:
            logger.exception("DetectionCacheBuild error")
            self.finished_signal.emit(False, "")
        finally:
            runner.close()


class TrackingPreviewWorker(QThread):
    """
    Emits visualization frames for previewing optimization results.
    """

    frame_signal = Signal(np.ndarray)
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _emit_frame(self, rgb: np.ndarray) -> None:
        self.frame_signal.emit(rgb)
        self.msleep(20)

    def run(self):
        try:
            run_tracking_preview(
                self.video_path,
                self.detection_cache_path,
                self.start_frame,
                self.end_frame,
                self.params,
                frame_cb=self._emit_frame,
                stop_check=lambda: self._stop_requested,
            )
        finally:
            self.finished_signal.emit()
