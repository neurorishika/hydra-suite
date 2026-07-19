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
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _cuda_gpu_rt():
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=True,
        default_runtime="cuda",
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

    def _fake_load_pose_model(pose_cfg, runtime):
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


def test_load_pose_backend_yolo_dispatch(monkeypatch):
    """backend_family="yolo" must build a PoseConfig(backend="yolo", yolo=...)."""
    import hydra_suite.core.inference.api as inference_api
    import hydra_suite.core.inference.stages.pose as pose_stage

    sentinel_backend = object()
    captured = {}

    class _FakePoseModel:
        def __init__(self, backend):
            self.backend = backend

    def _fake_load_pose_model(pose_cfg, runtime):
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
