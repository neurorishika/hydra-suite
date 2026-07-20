"""Pose runtime golden rule: all pose goes through load_pose_model, and the
cpu tier for SLEAP routes to the native (service-backend) torch-CPU path
instead of the exported ONNX-CPU path. Mirrors the cuda/gpu_fast tests in
test_inference_stages_pose.py.
"""

from unittest.mock import MagicMock

from hydra_suite.core.inference.config import PoseConfig, PoseSLEAPConfig
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import load_pose_model


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )


def _cuda_gpu_rt():
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=True,
        tensor_on_cuda=True,
        requested_gpu=True,
    )


def test_load_pose_model_sleap_cpu_tier_uses_native_torch_cpu(monkeypatch):
    """cpu tier must run SLEAP's native (non-exported) model on torch-CPU via
    the service backend -- NOT the exported onnx_cpu path -- so pose is
    consistent across all tiers (golden rule)."""
    import hydra_suite.core.identity.pose.api as api_mod

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    config = PoseConfig(
        backend="sleap",
        sleap=PoseSLEAPConfig(model_path="/fake/sleap_model_dir"),
    )
    load_pose_model(config, _cpu_rt())

    assert captured["runtime_flavor"] == "native"
    assert captured["device"] == "cpu"
    assert captured["runtime_flavor"] != "onnx_cpu"


def test_load_pose_model_sleap_cuda_tier_still_native_cuda(monkeypatch):
    """Guard: the cuda/gpu tier must remain unchanged (native torch CUDA)."""
    import hydra_suite.core.identity.pose.api as api_mod

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    config = PoseConfig(
        backend="sleap",
        sleap=PoseSLEAPConfig(model_path="/fake/sleap_model_dir"),
    )
    load_pose_model(config, _cuda_gpu_rt())

    assert captured["runtime_flavor"] == "native"
    assert captured["device"] == "cuda"


def test_load_pose_backend_returns_backend_not_wrapper(monkeypatch):
    """api.load_pose_backend must return PoseModel.backend, not the wrapper."""
    import hydra_suite.core.inference.api as inference_api
    import hydra_suite.core.inference.stages.pose as pose_stage

    sentinel_backend = object()
    captured = {}

    class _FakePoseModel:
        def __init__(self, backend):
            self.backend = backend

    def _fake_load_pose_model(pose_cfg, runtime, **kwargs):
        captured["pose_cfg"] = pose_cfg
        return _FakePoseModel(sentinel_backend)

    monkeypatch.setattr(pose_stage, "load_pose_model", _fake_load_pose_model)

    result = inference_api.load_pose_backend(
        backend_family="sleap",
        model_path="m",
        compute_runtime="cpu",
    )

    assert result is sentinel_backend
    assert captured["pose_cfg"].backend == "sleap"
    assert captured["pose_cfg"].sleap is not None
    assert captured["pose_cfg"].sleap.model_path == "m"
    assert captured["pose_cfg"].yolo is None


def test_load_pose_backend_end_to_end_cpu_routes_native_cpu(monkeypatch):
    """REAL end-to-end routing: api.load_pose_backend(compute_runtime="cpu")
    must actually resolve to the native/cpu SLEAP path.

    Unlike test_load_pose_backend_returns_backend_not_wrapper (which
    monkeypatches stages.pose.load_pose_model itself and therefore cannot
    catch a broken tier translation), this exercises the full
    shim -> runtime_tier -> RuntimeContext -> load_pose_model -> factory
    path, patching only the innermost SLEAP backend factory.

    torch.backends.mps.is_available is forced False so the cpu-tier request
    resolves device="cpu" deterministically on this (Apple Silicon) test
    host, where MPS would otherwise be opportunistically selected even for
    the cpu tier (see RuntimeContext.from_config / _cpu_or_mps_device).
    """
    import hydra_suite.core.identity.pose.api as api_mod
    import hydra_suite.core.inference.api as inference_api

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    inference_api.load_pose_backend(
        backend_family="sleap", model_path="m", compute_runtime="cpu"
    )

    assert captured["runtime_flavor"] == "native"
    assert captured["device"] == "cpu"


def test_load_pose_backend_end_to_end_cuda_routes_native_cuda(monkeypatch):
    """REAL end-to-end routing for compute_runtime="cuda" -> native/cuda.

    detect_platform and torch.cuda.is_available are patched so the "gpu"
    tier resolves to a CUDA device deterministically on this no-CUDA test
    host -- this exercises the same tier-resolution code path a real CUDA
    box would take, it just supplies the platform facts a CUDA box would
    have. Only the innermost SLEAP backend factory is faked.
    """
    import torch

    import hydra_suite.core.identity.pose.api as api_mod
    import hydra_suite.core.inference.api as inference_api
    import hydra_suite.runtime.resolver as resolver_mod

    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    inference_api.load_pose_backend(
        backend_family="sleap", model_path="m", compute_runtime="cuda"
    )

    assert captured["runtime_flavor"] == "native"
    assert captured["device"] == "cuda"


def test_load_pose_backend_end_to_end_tensorrt_routes_tensorrt_cuda(monkeypatch):
    """REAL end-to-end routing for compute_runtime="tensorrt" -> tensorrt/cuda.

    Same platform-faking rationale as the cuda test above: this host has no
    CUDA, so detect_platform / torch.cuda.is_available are patched to
    supply CUDA platform facts, letting the real gpu_fast tier-resolution
    code run its CUDA+TensorRT branch deterministically.
    """
    import torch

    import hydra_suite.core.identity.pose.api as api_mod
    import hydra_suite.core.inference.api as inference_api
    import hydra_suite.runtime.resolver as resolver_mod

    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    captured = {}

    def _fake_create_pose_backend_from_config(config):
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.device
        return MagicMock()

    monkeypatch.setattr(
        api_mod,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    inference_api.load_pose_backend(
        backend_family="sleap", model_path="m", compute_runtime="tensorrt"
    )

    assert captured["runtime_flavor"] == "tensorrt"
    assert captured["device"] == "cuda"


def test_load_pose_backend_yolo_dispatch(monkeypatch):
    """backend_family="yolo" must build a PoseConfig(backend="yolo", yolo=...)."""
    import hydra_suite.core.inference.api as inference_api
    import hydra_suite.core.inference.stages.pose as pose_stage

    sentinel_backend = object()
    captured = {}

    class _FakePoseModel:
        def __init__(self, backend):
            self.backend = backend

    def _fake_load_pose_model(pose_cfg, runtime, **kwargs):
        captured["pose_cfg"] = pose_cfg
        return _FakePoseModel(sentinel_backend)

    monkeypatch.setattr(pose_stage, "load_pose_model", _fake_load_pose_model)

    result = inference_api.load_pose_backend(
        backend_family="yolo",
        model_path="y.pt",
        compute_runtime="cpu",
    )

    assert result is sentinel_backend
    assert captured["pose_cfg"].backend == "yolo"
    assert captured["pose_cfg"].yolo is not None
    assert captured["pose_cfg"].yolo.model_path == "y.pt"
    assert captured["pose_cfg"].sleap is None
