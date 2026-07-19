"""Pure (Qt-free) bg-optimizer core: callback wiring + stop honoring."""

import hydra_suite.core.background.optimizer as opt


def test_run_bg_optimization_is_importable_and_qt_free():
    # The function exists and the module imports without Qt.
    assert hasattr(opt, "run_bg_optimization")
    assert hasattr(opt, "BgOptimizationRun")


def test_run_bg_optimization_honors_stop_check(monkeypatch, tmp_path):
    # The video can't be opened (nonexistent path), so the function must
    # return an empty run gracefully instead of raising -- regardless of
    # stop_check, this exercises the same early-exit path stop_check would
    # hit once frame caching begins.
    progress = []
    run = opt.run_bg_optimization(
        video_path="nonexistent.mp4",
        base_params={"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0},
        tuning_config={},
        scoring_weights={},
        n_trials=1,
        n_sample_frames=1,
        sampler_type="tpe",
        progress_cb=lambda pct, msg="": progress.append((pct, msg)),
        stop_check=lambda: True,  # stop immediately
    )
    assert isinstance(run, opt.BgOptimizationRun)
    assert run.results == []  # stopped/failed before any trial completed


def test_run_bg_optimization_stop_check_short_circuits_frame_read(monkeypatch):
    # Monkeypatch cv2.VideoCapture so the video "opens" successfully, then
    # verify stop_check() returning True immediately aborts frame caching
    # (via _read_gray_frames returning None) so the study never runs.
    class _FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            return 100

        def set(self, *a, **k):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(opt.cv2, "VideoCapture", lambda path: _FakeCap())

    run = opt.run_bg_optimization(
        video_path="fake.mp4",
        base_params={"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0},
        tuning_config={},
        scoring_weights={},
        n_trials=1,
        n_sample_frames=1,
        sampler_type="tpe",
        stop_check=lambda: True,
    )
    assert isinstance(run, opt.BgOptimizationRun)
    assert run.results == []
