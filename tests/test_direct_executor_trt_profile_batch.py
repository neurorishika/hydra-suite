"""Dynamic TRT engines: batches beyond the profile are split, rejections are loud.

Regression: a sliced (SAHI) run sized its engine profile from a load-time tile
estimate (batch 2) but submitted 3-tile chunks at run time. TensorRT rejected
every ``setInputShape``, which it reports only by returning ``False``, so the
run "completed" with 3 trajectories instead of ~25.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.inference.direct_executors import (
    _BaseDirectOBBExecutor,
    _set_trt_input_shape,
    _trt_profile_max_batch,
)


class _Engine:
    def __init__(self, max_batch=None, raises=False):
        self.max_batch = max_batch
        self.raises = raises

    def get_tensor_profile_shape(self, name, profile):
        if self.raises:
            raise RuntimeError("no profile")
        return (1, 3, 32, 32), (1, 3, 1024, 1024), (self.max_batch, 3, 1024, 1024)


def test_profile_max_batch_reads_profile_zero_maximum():
    assert _trt_profile_max_batch(_Engine(max_batch=8), "images") == 8


def test_profile_max_batch_is_none_without_a_usable_profile():
    assert _trt_profile_max_batch(_Engine(raises=True), "images") is None
    assert _trt_profile_max_batch(_Engine(max_batch=-1), "images") is None


def _executor(profile_max, static=False, model_bs=1):
    executor = object.__new__(_BaseDirectOBBExecutor)
    executor._static_batch = static
    executor._model_batch_size = model_bs
    executor._profile_max_batch = profile_max
    calls = []

    def _predict_chunk(frames, **_kwargs):
        calls.append(len(frames))
        return [f"r{len(calls)}-{i}" for i in range(len(frames))]

    executor._predict_chunk = _predict_chunk
    return executor, calls


def _frames(n):
    return [np.zeros((4, 4, 3), np.uint8) for _ in range(n)]


def test_dynamic_engine_splits_batches_beyond_profile_maximum():
    executor, calls = _executor(profile_max=3)
    results = executor.predict(_frames(8), conf_thres=0.1, classes=None, max_det=10)
    assert calls == [3, 3, 2]
    assert len(results) == 8


def test_dynamic_engine_within_profile_is_one_call():
    executor, calls = _executor(profile_max=8)
    executor.predict(_frames(8), conf_thres=0.1, classes=None, max_det=10)
    assert calls == [8]


def test_unknown_profile_keeps_single_call():
    executor, calls = _executor(profile_max=None)
    executor.predict(_frames(5), conf_thres=0.1, classes=None, max_det=10)
    assert calls == [5]


def test_static_engine_chunking_is_unchanged():
    # A static batch-2 engine still chunks AND pads to exactly 2.
    executor, calls = _executor(profile_max=None, static=True, model_bs=2)
    results = executor.predict(_frames(3), conf_thres=0.1, classes=None, max_det=10)
    assert calls == [2, 2]
    assert len(results) == 3


class _Context:
    def __init__(self, accepts):
        self.accepts = accepts
        self.shapes = []

    def set_input_shape(self, name, shape):
        self.shapes.append(shape)
        return self.accepts


def test_rejected_input_shape_raises():
    with pytest.raises(
        RuntimeError, match=r"rejected input shape \(3, 3, 1024, 1024\)"
    ):
        _set_trt_input_shape(_Context(False), "images", (3, 3, 1024, 1024), 2)


def test_accepted_input_shape_binds():
    context = _Context(True)
    _set_trt_input_shape(context, "images", (2, 3, 1024, 1024), 2)
    assert context.shapes == [(2, 3, 1024, 1024)]


def test_pose_runner_rejected_input_shape_raises():
    from hydra_suite.core.individual.pose.runtime.tensorrt_engine import (
        TensorRTEngineRunner,
    )

    runner = object.__new__(TensorRTEngineRunner)
    runner._context = _Context(False)
    runner.input_name = "input"
    with pytest.raises(RuntimeError, match=r"rejected input shape \(9, 3, 64, 64\)"):
        runner.run(np.zeros((9, 3, 64, 64), np.float32))
