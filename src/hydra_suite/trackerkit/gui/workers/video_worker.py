"""FinalMediaExportWorker — final canonical still/video export worker."""

from PySide6.QtCore import Signal

from hydra_suite.core.individual.dataset.oriented_video import (
    OrientedTrackVideoExporter,
)
from hydra_suite.widgets.workers import BaseWorker


class FinalMediaExportWorker(BaseWorker):
    """Worker thread for exporting final canonical stills and videos."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(
        self,
        final_csv_path,
        dataset_dir,
        image_output_dir,
        video_path,
        detection_cache_path,
        interpolated_roi_npz_path,
        fps,
        padding_fraction,
        background_color,
        suppress_foreign_obb,
        suppress_foreign_obb_images=None,
        suppress_foreign_obb_videos=None,
        export_images=False,
        image_interval=1,
        image_format="png",
        export_videos=True,
        fix_direction_flips=False,
        heading_flip_max_burst=5,
        enable_affine_stabilization=False,
        stabilization_window=5,
        output_subdir="oriented_videos",
        geometry=None,
    ):
        super().__init__()
        self.final_csv_path = final_csv_path
        self.dataset_dir = dataset_dir
        self.image_output_dir = image_output_dir
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.interpolated_roi_npz_path = interpolated_roi_npz_path
        self.fps = fps
        self.padding_fraction = padding_fraction
        self.background_color = background_color
        self.suppress_foreign_obb = suppress_foreign_obb
        self.suppress_foreign_obb_images = suppress_foreign_obb_images
        self.suppress_foreign_obb_videos = suppress_foreign_obb_videos
        self.export_images = export_images
        self.image_interval = image_interval
        self.image_format = image_format
        self.export_videos = export_videos
        self.fix_direction_flips = fix_direction_flips
        self.heading_flip_max_burst = heading_flip_max_burst
        self.enable_affine_stabilization = enable_affine_stabilization
        self.stabilization_window = stabilization_window
        self.output_subdir = output_subdir
        self.geometry = geometry
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self):
        """Track and orient detected animals in video frames."""
        try:
            exporter = OrientedTrackVideoExporter(
                self.dataset_dir,
                self.final_csv_path,
                video_path=self.video_path,
                detection_cache_path=self.detection_cache_path,
                interpolated_roi_npz_path=self.interpolated_roi_npz_path,
                fps=self.fps,
                padding_fraction=self.padding_fraction,
                background_color=self.background_color,
                suppress_foreign_obb=self.suppress_foreign_obb,
                suppress_foreign_obb_images=self.suppress_foreign_obb_images,
                suppress_foreign_obb_videos=self.suppress_foreign_obb_videos,
                export_images=self.export_images,
                image_output_dir=self.image_output_dir,
                image_interval=self.image_interval,
                image_format=self.image_format,
                export_videos=self.export_videos,
                fix_direction_flips=self.fix_direction_flips,
                heading_flip_max_burst=self.heading_flip_max_burst,
                enable_affine_stabilization=self.enable_affine_stabilization,
                stabilization_window=self.stabilization_window,
                output_subdir=self.output_subdir,
                geometry=self.geometry,
            )
            result = exporter.export(
                progress_callback=self.progress_signal.emit,
                should_stop=self._should_stop,
            )
            if not self._should_stop():
                self.finished_signal.emit(result.to_dict())
        except Exception as exc:
            if not self._should_stop():
                self.error_signal.emit(str(exc))


OrientedTrackVideoWorker = FinalMediaExportWorker
