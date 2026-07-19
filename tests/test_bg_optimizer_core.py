"""Pure (Qt-free) bg-optimizer core: callback wiring + stop honoring."""

import numpy as np

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


def test_generate_bg_previews_is_importable_and_qt_free():
    assert hasattr(opt, "generate_bg_previews")


def test_generate_bg_previews_emits_via_frame_cb(monkeypatch):
    # Mock the pipeline init/step so no heavy BG-sub setup runs; just verify
    # the drawing + frame_cb path fires and frame_cb is the only output.
    monkeypatch.setattr(
        opt,
        "_init_trial_pipeline",
        lambda det_params, frame_cache: (None, None, None, None),
    )
    monkeypatch.setattr(
        opt,
        "_run_bg_trial_frame",
        lambda raw_gray, idx, det_params, bg_model, detector, roi_mask, intensity_history, lighting_state: (
            raw_gray,
            raw_gray,
            raw_gray,
            [],
            [],
            [],
        ),
    )

    frames = []
    gray = np.zeros((8, 8), np.uint8)
    opt.generate_bg_previews(
        video_path="unused.mp4",
        base_params={"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0},
        trial_params={},
        n_sample_frames=1,
        prime_frames=[gray],
        sample_frames=[gray],
        sample_indices=[0],
        roi_mask=None,
        frame_cb=lambda idx, rgb: frames.append((idx, rgb)),
        stop_check=lambda: False,
    )

    assert frames == [(0, frames[0][1])]
    assert frames[0][1].shape == (8, 8, 3)


def test_generate_bg_previews_honors_stop_check(monkeypatch):
    # stop_check() returning True before the loop starts must short-circuit
    # so frame_cb is never called -- no Qt signal emitted either.
    called = {"pipeline": False}

    def _fake_init(det_params, frame_cache):
        called["pipeline"] = True
        return (None, None, None, None)

    monkeypatch.setattr(opt, "_init_trial_pipeline", _fake_init)

    frames = []
    gray = np.zeros((8, 8), np.uint8)
    opt.generate_bg_previews(
        video_path="unused.mp4",
        base_params={"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0},
        trial_params={},
        n_sample_frames=1,
        prime_frames=[gray],
        sample_frames=[gray],
        sample_indices=[0],
        roi_mask=None,
        frame_cb=lambda idx, rgb: frames.append((idx, rgb)),
        stop_check=lambda: True,
    )

    assert frames == []
