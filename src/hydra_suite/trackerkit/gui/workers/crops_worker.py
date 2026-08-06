"""InterpolatedCropsWorker — per-animal interpolated crop export worker."""

import logging

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class InterpolatedCropsWorker(BaseWorker):
    """Worker thread for interpolating occluded crops without blocking the UI."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(dict)

    def __init__(
        self,
        csv_path,
        video_path,
        detection_cache_path,
        params,
        enable_profiling=False,
        profile_export_path=None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.params = params
        self.enable_profiling = enable_profiling
        self.profile_export_path = profile_export_path
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self):
        """Generate interpolated crops (delegates to core/post/interpolated_crops)."""
        from hydra_suite.core.post.interpolated_crops import run_interpolated_crops

        payload = run_interpolated_crops(
            self.csv_path,
            self.video_path,
            self.detection_cache_path,
            self.params,
            enable_profiling=self.enable_profiling,
            profile_export_path=self.profile_export_path,
            progress=lambda v, m: self.progress_signal.emit(v, m),
            should_stop=self._should_stop,
        )
        if not self._should_stop():
            self.finished_signal.emit(payload)
