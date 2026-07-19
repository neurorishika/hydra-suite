"""Qt-free tests for ``run_tracking_preview``, the pure module-level function
extracted from ``TrackingPreviewWorker.run`` (core/qt split, part B1).

Only the core module is imported here (no PySide6 widgets under test) so
these tests exercise the tracking-preview loop independent of the QThread
wrapper.
"""

import inspect

import hydra_suite.core.tracking.optimization.optimizer_workers as ow


def test_core_tracking_optimization_imports_no_qt():
    """Guard scoped to optimizer_workers.py specifically (Part B3): it is the
    module this task made Qt-free, and its pure helpers/`run_tracking_preview`
    must stay that way. `optimizer.py` in this same package still defines a
    `TrackingOptimizer(QThread)` and is a known, separately-tracked offender
    (out of scope here) -- do not widen this guard back to the whole package
    until that file is migrated too.
    """
    import ast
    from pathlib import Path

    import hydra_suite.core.tracking.optimization.optimizer_workers as opt_workers_mod

    py = Path(opt_workers_mod.__file__)
    offenders = []
    for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
        mod = (
            node.module
            if isinstance(node, ast.ImportFrom)
            else (
                ",".join(a.name for a in node.names)
                if isinstance(node, ast.Import)
                else None
            )
        )
        if mod and ("PySide6" in mod or "QtCore" in mod):
            offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, "optimizer_workers.py must not import Qt: " + "; ".join(
        offenders
    )


def test_run_tracking_preview_exists_and_is_callable():
    assert hasattr(ow, "run_tracking_preview")
    assert callable(ow.run_tracking_preview)
    sig = inspect.signature(ow.run_tracking_preview)
    params = list(sig.parameters)
    assert params[:5] == [
        "video_path",
        "detection_cache_path",
        "start_frame",
        "end_frame",
        "params",
    ]
    assert "frame_cb" in sig.parameters
    assert "stop_check" in sig.parameters


class _FakeDetectionHandle:
    """Stands in for DetectionCacheHandle: read-only, must never be closed."""

    def __init__(self, empty_obb=True):
        self.closed = False
        self.read_frames = []
        self._empty_obb = empty_obb

    def is_valid(self):
        return True

    def read_frame(self, f_idx):
        import numpy as np

        from hydra_suite.core.inference.result import OBBResult

        self.read_frames.append(f_idx)
        # Empty-but-well-formed OBBResult: no detections this frame, so the
        # preview loop takes its "no measurements" branch without needing a
        # real assigner/pose pipeline.
        return OBBResult(
            frame_idx=f_idx,
            centroids=np.zeros((0, 2), dtype=np.float32),
            angles=np.zeros((0,), dtype=np.float32),
            sizes=np.zeros((0,), dtype=np.float32),
            shapes=np.zeros((0, 2), dtype=np.float32),
            confidences=np.zeros((0,), dtype=np.float32),
            corners=np.zeros((0, 4, 2), dtype=np.float32),
            detection_ids=np.zeros((0,), dtype=np.int64),
        )

    def close(self):
        self.closed = True
        raise AssertionError(
            "run_tracking_preview must not call close() on the detection cache handle"
        )


class _FakeCaches:
    def __init__(self, detection_handle):
        self.detection = detection_handle


class _FakeCap:
    """Fake cv2.VideoCapture that yields a fixed number of blank frames."""

    def __init__(self, n_frames=5):
        self._remaining = n_frames

    def isOpened(self):
        return True

    def set(self, *a, **k):
        pass

    def read(self):
        import numpy as np

        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        pass


def _patch_cache_open(monkeypatch, fake_handle):
    monkeypatch.setattr(
        ow,
        "_open_caches",
        lambda cfg, cache_dir, video_sig: _FakeCaches(fake_handle),
        raising=False,
    )
    monkeypatch.setattr(
        ow, "video_signature", lambda video_path: ("sig", video_path), raising=False
    )
    monkeypatch.setattr(
        ow, "build_inference_config_from_params", lambda p: object(), raising=False
    )


def test_run_tracking_preview_honors_immediate_stop_check(monkeypatch, tmp_path):
    """stop_check() -> True immediately must return without crashing and
    without reading any cached frames."""
    fake_handle = _FakeDetectionHandle()
    _patch_cache_open(monkeypatch, fake_handle)
    monkeypatch.setattr(ow.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(n_frames=5))

    emitted = []
    ow.run_tracking_preview(
        video_path="v.mp4",
        detection_cache_path=str(tmp_path),
        start_frame=0,
        end_frame=4,
        params={"MAX_TARGETS": 1},
        frame_cb=emitted.append,
        stop_check=lambda: True,
    )

    assert emitted == []
    assert fake_handle.read_frames == []
    assert fake_handle.closed is False


def test_run_tracking_preview_stops_after_n_frames(monkeypatch, tmp_path):
    """stop_check should be honored mid-loop: once it flips to True after N
    emitted frames, the loop must stop emitting further frames."""
    fake_handle = _FakeDetectionHandle()
    _patch_cache_open(monkeypatch, fake_handle)
    monkeypatch.setattr(ow.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(n_frames=10))

    emitted = []
    stop_after = 3

    def _stop_check():
        return len(emitted) >= stop_after

    ow.run_tracking_preview(
        video_path="v.mp4",
        detection_cache_path=str(tmp_path),
        start_frame=0,
        end_frame=9,
        params={"MAX_TARGETS": 1},
        frame_cb=emitted.append,
        stop_check=_stop_check,
    )

    assert len(emitted) == stop_after
    assert fake_handle.closed is False


def test_run_tracking_preview_handles_missing_cache_gracefully(monkeypatch, tmp_path):
    """If the cache is None (open failed), the function must log and return
    rather than raising."""
    monkeypatch.setattr(
        ow,
        "_open_caches",
        lambda cfg, cache_dir, video_sig: _FakeCaches(None),
        raising=False,
    )
    monkeypatch.setattr(
        ow, "video_signature", lambda video_path: ("sig", video_path), raising=False
    )
    monkeypatch.setattr(
        ow, "build_inference_config_from_params", lambda p: object(), raising=False
    )
    monkeypatch.setattr(ow.cv2, "VideoCapture", lambda *_a, **_k: _FakeCap(n_frames=5))

    emitted = []
    # Must not raise.
    ow.run_tracking_preview(
        video_path="v.mp4",
        detection_cache_path=str(tmp_path),
        start_frame=0,
        end_frame=4,
        params={"MAX_TARGETS": 1},
        frame_cb=emitted.append,
        stop_check=None,
    )
    assert emitted == []
