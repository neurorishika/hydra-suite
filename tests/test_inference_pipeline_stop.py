from __future__ import annotations

from hydra_suite.core.inference.pipeline import Pipeline


def _fake_pipeline(window_size: int, depth: int) -> Pipeline:
    pipe = Pipeline.for_test(
        window_size=window_size, depth=depth, stage=lambda w: list(w.frames)
    )
    pipe._run_obb_for_window = lambda window: []
    pipe._process_obb_results = lambda window, raw_list: list(window.frames)
    return pipe


def test_run_sync_stops_early_when_should_stop_returns_true():
    pipe = _fake_pipeline(window_size=2, depth=1)
    call_count = {"n": 0}

    def should_stop():
        call_count["n"] += 1
        # Allow the first window (frames 0,1) through, then stop before window 2.
        return call_count["n"] > 1

    frame_source = [(i, None) for i in range(10)]
    result = pipe.run(frame_source, range(10), range_total=10, should_stop=should_stop)

    assert result.frames_processed == 2


def test_run_sync_processes_everything_when_should_stop_is_none():
    pipe = _fake_pipeline(window_size=3, depth=1)
    frame_source = [(i, None) for i in range(7)]
    result = pipe.run(frame_source, range(7), range_total=7)
    assert result.frames_processed == 7


def test_run_double_buffer_stops_early_when_should_stop_returns_true():
    pipe = _fake_pipeline(window_size=2, depth=2)
    call_count = {"n": 0}

    def should_stop():
        call_count["n"] += 1
        return call_count["n"] > 1

    frame_source = [(i, None) for i in range(20)]
    result = pipe.run(frame_source, range(20), range_total=20, should_stop=should_stop)

    assert result.frames_processed < 20


def test_run_double_buffer_processes_everything_when_should_stop_is_none():
    pipe = _fake_pipeline(window_size=2, depth=2)
    frame_source = [(i, None) for i in range(9)]
    result = pipe.run(frame_source, range(9), range_total=9)
    assert result.frames_processed == 9
