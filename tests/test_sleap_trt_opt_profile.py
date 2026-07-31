"""Regression guard: the native SLEAP TensorRT build must define an
optimization profile for the dynamic batch dim.

Bug: `_build_trt_engine_from_onnx` called `build_serialized_network` without
adding an optimization profile. The SLEAP UNet ONNX has a dynamic leading
(batch) dim, so TensorRT aborted with "Network has dynamic or shape inputs,
but no optimization profile has been defined" — the build always failed and
every session fell back to the slow ORT-TRT-EP path (~8-16s init). Fix: pin
min/opt/max shapes for the dynamic batch dim before building.
"""

import sys
import types

from hydra_suite.core.identity.pose.backends.sleap import (
    _TRT_PROFILE_MAX_BATCH,
    _build_trt_engine_from_onnx,
)


class _FakeInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeNetwork:
    def __init__(self, inputs):
        self._inputs = inputs

    @property
    def num_inputs(self):
        return len(self._inputs)

    def get_input(self, idx):
        return self._inputs[idx]


class _FakeProfile:
    def __init__(self):
        self.set_calls = []

    def set_shape(self, name, mn, opt, mx):
        self.set_calls.append((name, mn, opt, mx))


class _FakeConfig:
    def __init__(self):
        self.added_profiles = []

    def set_memory_pool_limit(self, *_):
        pass

    def add_optimization_profile(self, profile):
        self.added_profiles.append(profile)


class _FakeBuilder:
    def __init__(self, network, config, profile):
        self._network = network
        self._config = config
        self._profile = profile
        self.built = False

    def create_network(self, _flags):
        return self._network

    def create_builder_config(self):
        return self._config

    def create_optimization_profile(self):
        return self._profile

    def build_serialized_network(self, _network, _config):
        self.built = True
        return b"ENGINE_BYTES"


class _FakeParser:
    def __init__(self, *_):
        self.num_errors = 0

    def parse(self, _bytes):
        return True

    def get_error(self, _i):  # pragma: no cover - not hit on success
        return types.SimpleNamespace(desc=lambda: "")


def _install_fake_trt(monkeypatch, network, config, profile, holder):
    trt = types.ModuleType("tensorrt")

    class _Logger:
        WARNING = 0

        def __init__(self, *_):
            pass

    def _builder(_logger):
        b = _FakeBuilder(network, config, profile)
        holder["builder"] = b
        return b

    trt.Logger = _Logger
    trt.Builder = _builder
    trt.OnnxParser = _FakeParser
    trt.NetworkDefinitionCreationFlag = types.SimpleNamespace(EXPLICIT_BATCH=0)
    trt.MemoryPoolType = types.SimpleNamespace(WORKSPACE=0)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)


def test_dynamic_batch_input_gets_optimization_profile(tmp_path, monkeypatch):
    network = _FakeNetwork([_FakeInput("images", [-1, 1, 384, 384])])
    config = _FakeConfig()
    profile = _FakeProfile()
    holder = {}
    _install_fake_trt(monkeypatch, network, config, profile, holder)

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    engine = tmp_path / "model.trt"

    ok = _build_trt_engine_from_onnx(onnx, engine)

    assert ok is True
    assert engine.read_bytes() == b"ENGINE_BYTES"
    assert holder["builder"].built is True
    # Profile added to config and shaped with min/opt/max on the batch dim.
    assert config.added_profiles == [profile]
    assert profile.set_calls == [
        (
            "images",
            (1, 1, 384, 384),
            (64, 1, 384, 384),
            (_TRT_PROFILE_MAX_BATCH, 1, 384, 384),
        )
    ]


def test_dynamic_non_batch_dim_bails_to_ort_ep(tmp_path, monkeypatch):
    # A dynamic H/W we cannot infer -> return False (caller uses ORT-EP), and we
    # must NOT attempt to build a wrong engine.
    network = _FakeNetwork([_FakeInput("images", [-1, 1, -1, 384])])
    config = _FakeConfig()
    profile = _FakeProfile()
    holder = {}
    _install_fake_trt(monkeypatch, network, config, profile, holder)

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    engine = tmp_path / "model.trt"

    ok = _build_trt_engine_from_onnx(onnx, engine)

    assert ok is False
    assert not engine.exists()
    assert holder["builder"].built is False


def test_missing_tensorrt_returns_false(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    assert _build_trt_engine_from_onnx(onnx, tmp_path / "m.trt") is False
