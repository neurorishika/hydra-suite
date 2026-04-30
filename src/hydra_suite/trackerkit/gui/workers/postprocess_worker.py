"""PostProcessWorker — initial trajectory clean-up background worker."""

import logging
import os

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class PostProcessWorker(BaseWorker):
    """Worker thread for post-processing raw tracking CSVs without blocking the UI."""

    progress_signal = Signal(int, str)  # progress value, status message
    finished_signal = Signal(object)  # processed trajectories DataFrame
    error_signal = Signal(str)  # error message

    def __init__(self, csv_to_process, params, clean=True):
        super().__init__()
        self.csv_to_process = csv_to_process
        self.params = params
        self.clean = clean

    def execute(self):
        """Run post-processing on the raw trajectory CSV."""
        from hydra_suite.core.post.processing import process_trajectories_from_csv

        if not self.csv_to_process or not os.path.exists(self.csv_to_process):
            raise FileNotFoundError(f"Tracking CSV not found: {self.csv_to_process!r}")

        self.progress_signal.emit(5, "Starting post-processing...")

        effective_params = self.params
        if not self.clean:
            effective_params = dict(self.params)
            effective_params["MIN_TRAJECTORY_LENGTH"] = 1
            effective_params["MAX_VELOCITY_BREAK"] = float("inf")
            effective_params["MAX_OCCLUSION_GAP"] = 0
            effective_params["MAX_VELOCITY_ZSCORE"] = 0.0

        self.progress_signal.emit(20, "Processing trajectories from CSV...")
        processed_trajectories, stats = process_trajectories_from_csv(
            self.csv_to_process, effective_params
        )
        label = "full clean" if self.clean else "collapse only"
        logger.info(f"Post-processing stats ({label}): {stats}")

        self.progress_signal.emit(100, "Post-processing complete!")
        self.finished_signal.emit(processed_trajectories)
