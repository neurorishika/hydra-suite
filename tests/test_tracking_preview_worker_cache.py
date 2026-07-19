"""TrackingPreviewWorker must read the new-format InferenceRunner detection
cache (via _open_caches / DetectionCacheHandle.read_frame), matching how the
optimizer (c4e1958) was migrated -- not the legacy DetectionCache API.
"""

import hydra_suite.core.tracking.optimization.optimizer_workers as ow


class _FakeDetectionHandle:
    """Stands in for DetectionCacheHandle: read-only, must never be closed."""

    def __init__(self):
        self.closed = False
        self.read_frames = []

    def is_valid(self):
        return True

    def read_frame(self, f_idx):
        self.read_frames.append(f_idx)
        # Legacy 12-tuple shape (accepted by _preview_filter_cached_detections).
        return (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            None,
            None,
        )

    def close(self):
        # Must never be called on a read-only preview handle.
        self.closed = True
        raise AssertionError(
            "TrackingPreviewWorker must not call close() on the detection cache handle"
        )


class _FakeCaches:
    def __init__(self, detection_handle):
        self.detection = detection_handle


def test_preview_worker_opens_new_format_cache_and_does_not_close_it(
    monkeypatch, tmp_path
):
    fake_handle = _FakeDetectionHandle()
    calls = {}

    def _fake_open_caches(cfg, cache_dir, video_sig):
        calls["cache_dir"] = cache_dir
        calls["video_sig"] = video_sig
        return _FakeCaches(fake_handle)

    def _fake_video_signature(video_path):
        return ("sig", video_path)

    def _boom(*a, **k):
        raise AssertionError(
            "TrackingPreviewWorker must not construct the legacy DetectionCache"
        )

    monkeypatch.setattr(ow, "_open_caches", _fake_open_caches, raising=False)
    monkeypatch.setattr(ow, "video_signature", _fake_video_signature, raising=False)
    monkeypatch.setattr(
        ow, "build_inference_config_from_params", lambda p: object(), raising=False
    )
    monkeypatch.setattr(ow, "DetectionCache", _boom, raising=False)

    # Stub cv2.VideoCapture so the cache-open/validation path is exercised in
    # full (isOpened() -> True), but the per-frame loop ends immediately
    # (read() -> no frame) so the test doesn't need real video/pose data.
    class _FakeCap:
        def isOpened(self):
            return True

        def set(self, *a, **k):
            pass

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(ow.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap())

    worker = ow.TrackingPreviewWorker(
        video_path="v.mp4",
        detection_cache_path=str(tmp_path),
        start_frame=0,
        end_frame=0,
        params={"MAX_TARGETS": 1},
    )

    emitted = {"finished": False}
    worker.finished_signal = type(
        "S", (), {"emit": lambda self: emitted.__setitem__("finished", True)}
    )()

    worker.run()

    assert calls.get("cache_dir") is not None
    assert emitted["finished"] is True
    assert fake_handle.closed is False


def test_preview_worker_no_longer_imports_legacy_DetectionCache_directly():
    # Guard: the module must not reference DetectionCache in TrackingPreviewWorker.run
    import inspect

    src = inspect.getsource(ow.TrackingPreviewWorker.run)
    assert (
        "DetectionCache(" not in src
    ), "TrackingPreviewWorker.run must not construct the legacy DetectionCache"
