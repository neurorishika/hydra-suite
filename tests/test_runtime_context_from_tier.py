"""Tests: RuntimeContext.from_config derives cuda_mode/tensor_on_cuda from runtime_tier."""

from unittest.mock import patch

from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.runtime.resolver import PlatformInfo


def _cfg(tier: str) -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        runtime_tier=tier,
    )


_no_gpu = PlatformInfo(has_cuda=False, has_mps=False)
_cuda_host = PlatformInfo(has_cuda=True, has_mps=False)
_mps_host = PlatformInfo(has_cuda=False, has_mps=True)


def test_cpu_tier_context_is_not_cuda():
    with patch("hydra_suite.runtime.resolver.detect_platform", return_value=_cuda_host):
        ctx = RuntimeContext.from_config(_cfg("cpu"))
    assert ctx.cuda_mode is False
    assert ctx.tensor_on_cuda is False


def test_cpu_tier_no_gpu_platform():
    with patch("hydra_suite.runtime.resolver.detect_platform", return_value=_no_gpu):
        ctx = RuntimeContext.from_config(_cfg("cpu"))
    assert ctx.cuda_mode is False
    assert ctx.tensor_on_cuda is False
    # device is "mps" on Apple Silicon, "cpu" elsewhere — both are non-CUDA
    assert ctx.device in ("mps", "cpu")


def test_gpu_tier_on_cuda_host_is_cuda():
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
        ctx = RuntimeContext.from_config(_cfg("gpu"))
    assert ctx.cuda_mode is True
    assert ctx.tensor_on_cuda is True  # native GPU tier keeps tensors on CUDA


def test_gpu_fast_tier_on_cuda_host_is_cuda_but_no_tensor():
    # gpu_fast resolves to TensorRT — cuda_mode but tensor_on_cuda=False
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
        ctx = RuntimeContext.from_config(_cfg("gpu_fast"))
    assert ctx.cuda_mode is True
    assert ctx.tensor_on_cuda is False


def test_gpu_tier_on_mps_host_is_not_cuda():
    with patch("hydra_suite.runtime.resolver.detect_platform", return_value=_mps_host):
        ctx = RuntimeContext.from_config(_cfg("gpu"))
    assert ctx.cuda_mode is False
    assert ctx.tensor_on_cuda is False
    assert ctx.device in ("mps", "cpu")


def test_gpu_tier_on_no_gpu_host_falls_back_to_cpu():
    with patch("hydra_suite.runtime.resolver.detect_platform", return_value=_no_gpu):
        ctx = RuntimeContext.from_config(_cfg("gpu"))
    assert ctx.cuda_mode is False
    # device is "mps" on Apple Silicon, "cpu" elsewhere
    assert ctx.device in ("mps", "cpu")
