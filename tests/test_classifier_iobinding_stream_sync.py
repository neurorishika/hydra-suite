"""ONNX Runtime must not read a CUDA batch before torch has finished writing it.

Regression: on the gpu_fast tier with NVDEC frames, classifier crops are
preprocessed by torch kernels queued on torch's stream, then handed to ONNX
Runtime by raw pointer (IOBinding). ORT runs on its own stream, so on a busy
GPU it read half-written or stale inputs: ~13% wrong identity labels and ~9%
reversed head-tail calls, different on every run. ``CUDA_LAUNCH_BLOCKING=1``
made it disappear.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from hydra_suite.core.individual.classification.backend import ClassifierBackend


class _Stream:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append("torch_stream_synchronize")


class _Batch:
    shape = (2, 3, 4, 4)
    device = SimpleNamespace(index=0)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 1234


class _Binding:
    def bind_input(self, **_kwargs):
        pass

    def bind_output(self, *_args, **_kwargs):
        pass

    def copy_outputs_to_cpu(self):
        return [np.zeros((2, 5), np.float32)]


class _Session:
    def __init__(self, log):
        self.log = log

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def io_binding(self):
        return _Binding()

    def run_with_iobinding(self, _binding):
        self.log.append("ort_run")


def test_iobinding_waits_for_torch_stream_before_ort_reads(monkeypatch):
    log: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=lambda device=None: _Stream(log))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    backend = object.__new__(ClassifierBackend)
    backend._model = _Session(log)

    backend._forward_onnx_iobinding(_Batch())

    assert log == ["torch_stream_synchronize", "ort_run"]
