from __future__ import annotations

from types import SimpleNamespace

import pytest

from hydra_suite.core.inference import pipeline as pipeline_module
from hydra_suite.core.inference.cancellation import InferenceCancelled
from hydra_suite.core.inference.pipeline import BatchWindow, Pipeline


def _fake_pipeline(window_size: int, depth: int) -> Pipeline:
    pipe = Pipeline.for_test(
        window_size=window_size, depth=depth, stage=lambda w: list(w.frames)
    )
    pipe._run_detection_for_window = lambda window: []
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
    assert result.cancelled is True


def test_run_sync_processes_everything_when_should_stop_is_none():
    pipe = _fake_pipeline(window_size=3, depth=1)
    frame_source = [(i, None) for i in range(7)]
    result = pipe.run(frame_source, range(7), range_total=7)
    assert result.frames_processed == 7
    assert result.frame_results == []


def test_result_collection_is_explicit_and_stream_consumer_is_incremental():
    pipe = _fake_pipeline(window_size=2, depth=1)
    streamed = []
    frame_source = [(i, None) for i in range(5)]

    result = pipe.run(
        frame_source,
        range(5),
        collect_results=True,
        result_consumer=streamed.append,
    )

    assert result.frame_results == [None] * 5
    assert streamed == [None] * 5


def test_default_result_retention_is_constant_over_long_stream():
    frame_count = 10_000
    pipe = _fake_pipeline(window_size=37, depth=1)

    result = pipe.run(
        ((frame_idx, None) for frame_idx in range(frame_count)),
        range(frame_count),
        range_total=frame_count,
    )

    assert result.frames_processed == frame_count
    assert result.frame_results == []


def test_run_double_buffer_stops_early_when_should_stop_returns_true():
    pipe = _fake_pipeline(window_size=2, depth=2)
    call_count = {"n": 0}

    def should_stop():
        call_count["n"] += 1
        return call_count["n"] > 1

    frame_source = [(i, None) for i in range(20)]
    result = pipe.run(frame_source, range(20), range_total=20, should_stop=should_stop)

    assert result.frames_processed < 20
    assert result.cancelled is True


def test_final_only_window_cancel_does_not_count_or_report_complete_progress():
    pipe = _fake_pipeline(window_size=4, depth=1)
    progress: list[tuple[int, int]] = []

    def cancel_during_detection(window):
        raise InferenceCancelled("cancelled inside admitted tile chunk")

    pipe._run_detection_for_window = cancel_during_detection
    result = pipe.run(
        [(0, object()), (1, object())],
        range(2),
        range_total=2,
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    assert result.cancelled is True
    assert result.frames_processed == 0
    assert progress == [(0, 2)]


def test_frame_window_is_refused_before_pipeline_prefetch_exceeds_byte_budget():
    from hydra_suite.core.inference.pipeline import MAX_PIPELINE_BUFFER_BYTES

    pipe = _fake_pipeline(window_size=2, depth=2)
    frame_bytes = MAX_PIPELINE_BUFFER_BYTES // 2

    class _HugeLogicalFrame:
        nbytes = frame_bytes

    with pytest.raises(ValueError, match="pipeline frame buffer"):
        pipe.run(
            [(0, _HugeLogicalFrame())],
            range(1),
            range_total=1,
        )


def test_run_double_buffer_processes_everything_when_should_stop_is_none():
    pipe = _fake_pipeline(window_size=2, depth=2)
    frame_source = [(i, None) for i in range(9)]
    result = pipe.run(frame_source, range(9), range_total=9)
    assert result.frames_processed == 9


def test_detection_forwards_active_cancellation_to_bounded_region_source(monkeypatch):
    def callback():
        return False

    seen = {}

    def fake_run_obb(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(pipeline_module, "run_obb", fake_run_obb)
    pipe = Pipeline.__new__(Pipeline)
    pipe.stages = SimpleNamespace(
        config=SimpleNamespace(detection_source="obb", obb=object()),
        obb_models=object(),
        roi_mask=None,
    )
    pipe.runtime = SimpleNamespace(handoff=lambda value: value)
    pipe._active_should_stop = callback

    assert pipe._run_detection_for_window(BatchWindow([object()], [0])) == []
    assert seen["should_stop"] is callback
