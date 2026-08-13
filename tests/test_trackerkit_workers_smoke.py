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
