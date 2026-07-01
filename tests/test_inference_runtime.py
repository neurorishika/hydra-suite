import dataclasses
from unittest.mock import patch

import pytest

from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.runtime.resolver import PlatformInfo


def _cpu_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        ),
        runtime_tier="cpu",
    )


def _mps_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        ),
        runtime_tier="gpu",  # gpu tier on MPS-only host → device="mps", cuda_mode=False
    )


def _cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        ),
        runtime_tier="gpu",  # gpu tier on CUDA host → cuda_mode=True, tensor_on_cuda=True
    )


def _gpu_fast_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        ),
        runtime_tier="gpu_fast",  # TensorRT: cuda_mode=True, tensor_on_cuda=False
    )


_no_gpu = PlatformInfo(has_cuda=False, has_mps=False)
_mps_only = PlatformInfo(has_cuda=False, has_mps=True)
_cuda_host = PlatformInfo(has_cuda=True, has_mps=False)


def test_cpu_config_produces_cpu_mode():
    # cpu tier: never CUDA regardless of platform
    with patch("hydra_suite.runtime.resolver.detect_platform", return_value=_cuda_host):
        ctx = RuntimeContext.from_config(_cpu_config())
    assert ctx.cuda_mode is False
    # device is "mps" on Apple Silicon, "cpu" elsewhere — both are non-CUDA
    assert ctx.device in ("mps", "cpu")
    assert ctx.use_nvdec is False
    assert ctx.default_runtime == "cpu"


def test_mps_config_produces_cpu_mode():
    # gpu tier on MPS-only host: cuda_mode=False, device=mps
    with patch("hydra_suite.runtime.resolver.detect_platform", return_value=_mps_only):
        ctx = RuntimeContext.from_config(_mps_config())
    assert ctx.cuda_mode is False
    assert ctx.device in ("mps", "cpu")


def test_cuda_config_produces_cuda_mode():
    # gpu tier on CUDA host: cuda_mode=True
    with (
        patch("hydra_suite.runtime.resolver.detect_platform", return_value=_cuda_host),
        patch(
            "hydra_suite.core.inference.runtime._cuda_device_available",
            return_value="cuda:0",
        ),
        patch("hydra_suite.core.inference.runtime._nvdec_available", return_value=True),
    ):
        ctx = RuntimeContext.from_config(_cuda_config())
    assert ctx.cuda_mode is True
    assert ctx.device == "cuda:0"
    assert ctx.use_nvdec is True
    assert ctx.default_runtime == "cuda"


def test_cuda_without_nvdec():
    with (
        patch("hydra_suite.runtime.resolver.detect_platform", return_value=_cuda_host),
        patch(
            "hydra_suite.core.inference.runtime._cuda_device_available",
            return_value="cuda:0",
        ),
        patch(
            "hydra_suite.core.inference.runtime._nvdec_available", return_value=False
        ),
    ):
        ctx = RuntimeContext.from_config(_cuda_config())
    assert ctx.cuda_mode is True
    assert ctx.use_nvdec is False


def test_frozen_dataclass():
    ctx = RuntimeContext(
        cuda_mode=False, device="cpu", use_nvdec=False, default_runtime="cpu"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.cuda_mode = True  # type: ignore


def test_tensor_on_cuda_true_only_for_native_cuda():
    # tensor_on_cuda True: gpu tier (native torch) on CUDA host
    with (
        patch("hydra_suite.runtime.resolver.detect_platform", return_value=_cuda_host),
        patch(
            "hydra_suite.core.inference.runtime._cuda_device_available",
            return_value="cuda:0",
        ),
        patch(
            "hydra_suite.core.inference.runtime._nvdec_available", return_value=False
        ),
    ):
        ctx = RuntimeContext.from_config(_cuda_config())
    assert ctx.tensor_on_cuda is True

    # tensor_on_cuda False: gpu_fast (TensorRT) on CUDA host returns CPU numpy
    with (
        patch("hydra_suite.runtime.resolver.detect_platform", return_value=_cuda_host),
        patch(
            "hydra_suite.core.inference.runtime._cuda_device_available",
            return_value="cuda:0",
        ),
        patch(
            "hydra_suite.core.inference.runtime._nvdec_available", return_value=False
        ),
    ):
        ctx2 = RuntimeContext.from_config(_gpu_fast_config())
    assert ctx2.cuda_mode is True  # CUDA group
    assert ctx2.tensor_on_cuda is False  # TensorRT returns CPU numpy
