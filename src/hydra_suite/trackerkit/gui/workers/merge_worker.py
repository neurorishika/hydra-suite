"""MergeWorker — trajectory merge and CSV export background worker."""

import logging

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class MergeWorker(BaseWorker):
    """Worker thread for merging trajectories without blocking the UI."""

    progress_signal = Signal(int, str)  # progress value, status message
    finished_signal = Signal(object)  # merged trajectories
    error_signal = Signal(str)  # error message

    def __init__(
        self,
        forward_trajs,
        backward_trajs,
        total_frames,
        params,
        resize_factor,
        interp_method,
        max_gap,
        tag_cache_path=None,
        heading_flip_max_burst=5,
        directed_heading_posthoc=False,
        enable_profiling=False,
        profile_export_path=None,
    ):
        super().__init__()
        self.forward_trajs = forward_trajs
        self.backward_trajs = backward_trajs
        self.total_frames = total_frames
        self.params = params
        self.resize_factor = resize_factor
        self.interp_method = interp_method
        self.max_gap = max_gap
        self.tag_cache_path = tag_cache_path
        self.heading_flip_max_burst = heading_flip_max_burst
        self.directed_heading_posthoc = directed_heading_posthoc
        self.enable_profiling = enable_profiling
        self.profile_export_path = profile_export_path
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self):
        """Merge forward and backward trajectories (delegates to core/post/merge)."""
        from hydra_suite.core.post.merge import merge_trajectories

        try:
            merged = merge_trajectories(
                self.forward_trajs,
                self.backward_trajs,
                total_frames=self.total_frames,
                params=self.params,
                resize_factor=self.resize_factor,
                interp_method=self.interp_method,
                max_gap=self.max_gap,
                tag_cache_path=self.tag_cache_path,
                heading_flip_max_burst=self.heading_flip_max_burst,
                directed_heading_posthoc=self.directed_heading_posthoc,
                enable_profiling=self.enable_profiling,
                profile_export_path=self.profile_export_path,
                progress=lambda v, m: self.progress_signal.emit(v, m),
                should_stop=self._should_stop,
            )
            if merged is not None and not self._should_stop():
                self.finished_signal.emit(merged)
        except Exception as e:
            logger.exception("Error during trajectory merging")
            self.error_signal.emit(str(e))


# Compatibility re-exports: crops_worker.py imports these from this module.
from hydra_suite.core.post import merge as _merge_core

_write_csv_artifact = _merge_core.write_csv_artifact
_write_roi_npz = _merge_core.write_roi_npz
