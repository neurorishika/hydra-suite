"""DatasetGenerationWorker — active-learning dataset export worker."""

import logging

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class DatasetGenerationWorker(BaseWorker):
    """Worker thread for generating training datasets without blocking the UI."""

    progress_signal = Signal(int, str)  # progress value, status message
    finished_signal = Signal(str, int)  # dataset_dir, num_frames
    error_signal = Signal(str)  # error message

    def __init__(
        self,
        video_path,
        csv_path,
        detection_cache_path,
        output_dir,
        dataset_name,
        class_name,
        params,
        max_frames,
        diversity_window,
        include_context,
        probabilistic,
    ):
        super().__init__()
        self.video_path = video_path
        self.csv_path = csv_path
        self.detection_cache_path = detection_cache_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.class_name = class_name
        self.params = params
        self.max_frames = max_frames
        self.diversity_window = diversity_window
        self.include_context = include_context
        self.probabilistic = probabilistic
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self):
        """Generate training datasets from detections and annotations."""
        from hydra_suite.core.post import dataset_export

        result = dataset_export.generate_active_learning_dataset(
            video_path=self.video_path,
            csv_path=self.csv_path,
            detection_cache_path=self.detection_cache_path,
            output_dir=self.output_dir,
            dataset_name=self.dataset_name,
            class_name=self.class_name,
            params=self.params,
            max_frames=self.max_frames,
            diversity_window=self.diversity_window,
            include_context=self.include_context,
            probabilistic=self.probabilistic,
            progress=self.progress_signal.emit,
            should_stop=self._should_stop,
        )
        if self._should_stop() or result.get("cancelled"):
            return
        if result.get("success"):
            self.finished_signal.emit(result["dir"], result["num_frames"])
        else:
            self.error_signal.emit(result.get("error", "Dataset generation failed."))
