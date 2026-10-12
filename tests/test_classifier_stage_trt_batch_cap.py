"""Head-tail / CNN stages size the TensorRT profile to their own batch size.

The classifier backend's default TRT optimisation profile spans batch 1..512.
For an EfficientNet head-tail model at 512 the builder needs more than its
1 GiB workspace and fails outright, even though the stage never sends more
than ``config.batch_size`` crops per call (both stages chunk by it).
"""

from __future__ import annotations

import pytest

from hydra_suite.core.inference.config import CNNConfig, HeadTailConfig
from hydra_suite.runtime.resolver import ResolvedBackend


class _Captured(Exception):
    pass


def _capture_backend(monkeypatch):
    import hydra_suite.core.individual.classification.backend as backend_module

    seen: dict[str, object] = {}

    class _FakeBackend:
        def __init__(self, model_path, resolved, trt_profile_max_batch=None):
            seen["trt_profile_max_batch"] = trt_profile_max_batch
            raise _Captured

    monkeypatch.setattr(backend_module, "ClassifierBackend", _FakeBackend)
    return seen


class _Runtime:
    resolved = ResolvedBackend("tensorrt", "cuda", False)
    cuda_mode = False


def test_cnn_stage_caps_trt_profile_at_its_batch_size(monkeypatch):
    from hydra_suite.core.inference.stages.cnn import load_cnn_model

    seen = _capture_backend(monkeypatch)
    with pytest.raises(_Captured):
        load_cnn_model(
            CNNConfig(label="id", model_path="m.pth", batch_size=30), _Runtime()
        )
    assert seen["trt_profile_max_batch"] == 30


def test_headtail_stage_caps_trt_profile_at_its_batch_size(monkeypatch):
    from hydra_suite.core.inference.stages.headtail import load_headtail_model

    seen = _capture_backend(monkeypatch)
    with pytest.raises(_Captured):
        load_headtail_model(
            HeadTailConfig(model_path="m.pth", batch_size=30), _Runtime()
        )
    assert seen["trt_profile_max_batch"] == 30
