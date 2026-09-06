"""Allocation and CUDA-stream lifetime guards for direct SAHI executors."""

from __future__ import annotations

import sys
from collections import OrderedDict
from types import SimpleNamespace

from hydra_suite.core.inference.direct_executors import (
    DirectTensorRTOBBExecutor,
    _BaseDirectOBBExecutor,
)


class _Tensor:
    def __init__(self, shape, **_kwargs):
        self.shape = tuple(shape)


class _Event:
    def __init__(self, done=True):
        self.recorded = None
        self.done = done
        self.synchronized = False

    def record(self, stream):
        self.recorded = stream

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True


def _fake_torch(created):
    def empty(shape, **kwargs):
        tensor = _Tensor(shape, **kwargs)
        created.append(tensor)
        return tensor

    return SimpleNamespace(
        Tensor=_Tensor,
        empty=empty,
        uint8="uint8",
        float32="float32",
        cuda=SimpleNamespace(Event=_Event, current_stream=lambda: "default"),
    )


def test_input_slots_reuse_matching_batch_and_bound_cache(monkeypatch):
    created = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(created))
    executor = object.__new__(_BaseDirectOBBExecutor)
    executor.imgsz = 32
    executor._input_buffers = OrderedDict()
    executor._host_staging_events = {}
    executor._wait_input_reusable = lambda: None

    first = executor._input_buffer(2)
    assert executor._input_buffer(2) is first
    for batch in (1, 3, 4, 5, 6):
        executor._input_buffer(batch)

    # Four reusable shapes are retained; unusual later dynamic batches do not
    # turn a long run into a cache that grows with every observed shape.
    assert len(executor._input_buffers) == 4
    assert len(created) == 12  # two tensors per distinct requested shape


def test_input_lru_replaces_autotune_probe_with_selected_production_batch(monkeypatch):
    created = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(created))
    executor = object.__new__(_BaseDirectOBBExecutor)
    executor.imgsz = 32
    executor._input_buffers = OrderedDict()
    executor._host_staging_events = {}
    executor._wait_input_reusable = lambda: None

    for batch in (1, 2, 4, 8):
        pinned, _ = executor._input_buffer(batch)
        executor._host_staging_events[id(pinned)] = _Event()

    selected = executor._input_buffer(16)
    selected_again = executor._input_buffer(16)

    assert selected_again is selected
    assert 1 not in executor._input_buffers
    assert tuple(executor._input_buffers) == (2, 4, 8, 16)
    assert len(executor._input_buffers) == 4
    # The evicted probe's event is removed with its pinned tensor, preventing
    # the event registry from outgrowing the bounded buffer cache.
    assert len(executor._host_staging_events) <= 3


def test_dynamic_output_shape_reuses_slot_and_is_bounded(monkeypatch):
    created = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(created))
    executor = object.__new__(DirectTensorRTOBBExecutor)
    executor._output_buffers = OrderedDict()
    executor._active_output_event = None

    first, event = executor._output_buffer((2, 9, 10), "cuda:0")
    same, same_event = executor._output_buffer((2, 9, 10), "cuda:0")
    assert first is same
    assert event is same_event is None
    for size in range(1, 7):
        executor._output_buffer((size, 9, 10), "cuda:0")
    assert len(executor._output_buffers) == 4


def test_output_release_records_default_stream_before_reuse(monkeypatch):
    created = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(created))
    executor = object.__new__(DirectTensorRTOBBExecutor)
    output = _Tensor((1, 9, 10))
    executor._output_buffers = OrderedDict({(1, 9, 10): (output, None)})

    executor._release_output(output)

    _, event = executor._output_buffers[(1, 9, 10)]
    assert isinstance(event, _Event)
    assert event.recorded == "default"


def test_pinned_host_staging_waits_for_h2d_not_tensorrt_stream():
    executor = object.__new__(_BaseDirectOBBExecutor)
    pinned = object()
    pending_h2d = _Event(done=False)
    executor._host_staging_events = {id(pinned): pending_h2d}

    executor._wait_host_staging_reusable(pinned)

    assert pending_h2d.synchronized
    # This is intentionally independent of _wait_input_reusable(), which only
    # protects TensorRT's later read of the CUDA input buffer.


def test_output_release_runs_when_postprocess_raises():
    executor = object.__new__(_BaseDirectOBBExecutor)
    executor._preprocess = lambda frames: "input"
    executor._run_inference = lambda image: "raw-output"
    executor._release_output_calls = []
    executor._release_output = executor._release_output_calls.append

    def fail(*args, **kwargs):
        raise RuntimeError("postprocess failed")

    executor._postprocess = fail

    import pytest

    with pytest.raises(RuntimeError, match="postprocess failed"):
        executor._predict_chunk(
            [object()],
            cuda_input=False,
            conf_thres=0.1,
            classes=None,
            max_det=1,
        )
    assert executor._release_output_calls == ["raw-output"]
