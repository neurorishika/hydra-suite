"""Depth=1 Pipeline orchestration tests.

The windowing logic (fixed frame-index windows of size W) is a pure function of
frame index and must be testable without real models. ``Pipeline.for_test`` +
``run_frames`` expose the same ``_iter_windows`` / ``_process_window`` machinery
the production ``run()`` drives, with a fake per-window stage callable.
"""

from hydra_suite.core.inference.pipeline import BatchWindow, Pipeline


def test_windows_are_frame_indexed_not_arrival_indexed():
    seen = []

    def fake_stage(window):
        seen.append([f.index for f in window.frames])
        return [object() for _ in window.frames]

    pipe = Pipeline.for_test(window_size=2, depth=1, stage=fake_stage)
    results = pipe.run_frames(range(5))
    assert seen == [[0, 1], [2, 3], [4]]
    assert len(results) == 5


def test_single_window_when_range_smaller_than_window():
    seen = []

    def fake_stage(window):
        seen.append([f.index for f in window.frames])
        return list(window.frames)

    pipe = Pipeline.for_test(window_size=8, depth=1, stage=fake_stage)
    results = pipe.run_frames(range(3))
    assert seen == [[0, 1, 2]]
    assert len(results) == 3


def test_exact_multiple_window_boundaries():
    seen = []

    def fake_stage(window):
        seen.append([f.index for f in window.frames])
        return list(window.frames)

    pipe = Pipeline.for_test(window_size=2, depth=1, stage=fake_stage)
    pipe.run_frames(range(4))
    assert seen == [[0, 1], [2, 3]]


def test_window_size_one_yields_singleton_windows():
    seen = []

    def fake_stage(window):
        seen.append([f.index for f in window.frames])
        return list(window.frames)

    pipe = Pipeline.for_test(window_size=1, depth=1, stage=fake_stage)
    pipe.run_frames(range(3))
    assert seen == [[0], [1], [2]]


def test_batch_window_holds_frames_and_indices():
    window = BatchWindow(frames=[object(), object()], frame_indices=[3, 4])
    assert window.frame_indices == [3, 4]
    assert len(window.frames) == 2
