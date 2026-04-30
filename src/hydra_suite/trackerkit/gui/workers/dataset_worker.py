"""DatasetGenerationWorker — active-learning dataset export worker."""

import logging
import os

import pandas as pd
from PySide6.QtCore import Signal

from hydra_suite.data.detection_cache import DetectionCache
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
        detection_cache = None
        try:
            from hydra_suite.data.dataset_generation import (
                FrameQualityScorer,
                export_dataset,
            )

            if self._should_stop():
                return
            self.progress_signal.emit(5, "Initializing dataset generation...")

            # Load tracking CSV to compute quality scores
            self.progress_signal.emit(10, "Loading tracking data...")
            df = pd.read_csv(self.csv_path)

            # Initialize quality scorer
            self.progress_signal.emit(15, "Initializing quality scorer...")
            scorer = FrameQualityScorer(self.params)
            if self.detection_cache_path and os.path.exists(self.detection_cache_path):
                try:
                    detection_cache = DetectionCache(
                        self.detection_cache_path, mode="r"
                    )
                    if not detection_cache.is_compatible():
                        detection_cache.close()
                        detection_cache = None
                except Exception:
                    detection_cache = None

            # Score each frame
            self.progress_signal.emit(20, "Scoring frames...")
            unique_frames = df["FrameID"].unique()
            total_unique = len(unique_frames)

            for idx, frame_id in enumerate(unique_frames):
                if self._should_stop():
                    return
                if idx % 100 == 0:  # Update progress every 100 frames
                    progress = 20 + int((idx / total_unique) * 30)
                    self.progress_signal.emit(
                        progress, f"Scoring frames ({idx}/{total_unique})..."
                    )

                frame_data = df[df["FrameID"] == frame_id]
                raw_meas = []
                raw_shapes = []
                raw_confidences = []
                raw_obb_corners = []
                used_detection_cache = False
                if detection_cache is not None:
                    try:
                        (
                            raw_meas,
                            _,
                            raw_shapes,
                            raw_confidences,
                            raw_obb_corners,
                            _,
                            *_,
                        ) = detection_cache.get_frame(int(frame_id))
                        used_detection_cache = True
                    except Exception:
                        raw_meas = []
                        raw_shapes = []
                        raw_confidences = []
                        raw_obb_corners = []

                detection_count = (
                    len(raw_meas) if used_detection_cache else len(frame_data)
                )

                # Detection data
                detection_data = {
                    "confidences": (
                        raw_confidences
                        if raw_confidences
                        else (
                            frame_data["DetectionConfidence"].tolist()
                            if "DetectionConfidence" in frame_data.columns
                            else []
                        )
                    ),
                    "count": detection_count,
                    "measurements": raw_meas,
                    "shapes": raw_shapes,
                    "obb_corners": raw_obb_corners,
                }

                # Tracking data
                tracking_data = {
                    "lost_tracks": int((frame_data["State"] == "lost").sum()),
                    "assignment_confidences": (
                        frame_data["AssignmentConfidence"].tolist()
                        if "AssignmentConfidence" in frame_data.columns
                        else []
                    ),
                    "uncertainties": (
                        frame_data["PositionUncertainty"].tolist()
                        if "PositionUncertainty" in frame_data.columns
                        else []
                    ),
                }

                scorer.score_frame(frame_id, detection_data, tracking_data)

            if self._should_stop():
                return
            # Select worst frames with diversity
            self.progress_signal.emit(50, "Selecting challenging frames...")
            selected_frames = scorer.get_worst_frames(
                self.max_frames, self.diversity_window, probabilistic=self.probabilistic
            )

            if not selected_frames:
                self.error_signal.emit("No frames met the quality criteria for export.")
                return

            # Export dataset
            self.progress_signal.emit(60, f"Exporting {len(selected_frames)} frames...")
            if self._should_stop():
                return
            dataset_dir = export_dataset(
                video_path=self.video_path,
                csv_path=self.csv_path,
                frame_ids=selected_frames,
                output_dir=self.output_dir,
                dataset_name=self.dataset_name,
                class_name=self.class_name,
                params=self.params,
                include_context=self.include_context,
            )

            if not self._should_stop():
                self.progress_signal.emit(100, "Dataset generation complete!")
                self.finished_signal.emit(dataset_dir, len(selected_frames))

        except Exception as e:
            logger.exception("Error during dataset generation")
            self.error_signal.emit(str(e))
        finally:
            if detection_cache is not None:
                try:
                    detection_cache.close()
                except Exception:
                    pass
