"""Smoke tests: all trackerkit workers importable from workers/ subpackage."""


def test_tracking_worker_stop_stops_active_prefetcher():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    class FakePrefetcher:
        def __init__(self) -> None:
            self.stop_called = False

        def stop(self) -> None:
            self.stop_called = True

    worker = TrackingEngineCore("video.mp4")
    worker.frame_prefetcher = FakePrefetcher()

    worker.stop()

    assert worker._stop_requested is True
    assert worker.frame_prefetcher.stop_called is True


def test_merge_worker_importable():
    from hydra_suite.trackerkit.gui.workers.merge_worker import MergeWorker

    assert MergeWorker is not None


def test_crops_worker_importable():
    from hydra_suite.trackerkit.gui.workers.crops_worker import InterpolatedCropsWorker

    assert InterpolatedCropsWorker is not None


def test_video_worker_importable():
    from hydra_suite.trackerkit.gui.workers.video_worker import OrientedTrackVideoWorker

    assert OrientedTrackVideoWorker is not None


def test_dataset_worker_importable():
    from hydra_suite.trackerkit.gui.workers.dataset_worker import (
        DatasetGenerationWorker,
    )

    assert DatasetGenerationWorker is not None


def test_preview_worker_importable():
    from hydra_suite.trackerkit.gui.workers.preview_worker import PreviewDetectionWorker

    assert PreviewDetectionWorker is not None
