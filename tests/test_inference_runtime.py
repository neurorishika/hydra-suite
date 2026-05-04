import dataclasses
from unittest.mock import patch

import pytest

from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.runtime import RuntimeContext


def _cpu_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"),
        )
    )


def _mps_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="mps"),
        )
    )


def _cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"),
        )
    )


def test_cpu_config_produces_cpu_mode():
    ctx = RuntimeContext.from_config(_cpu_config())
    assert ctx.cuda_mode is False
    assert ctx.device == "cpu"
    assert ctx.use_nvdec is False
    assert ctx.default_runtime == "cpu"


def test_mps_config_produces_cpu_mode():
    # MPS is CPU-group — cuda_mode should be False.
    # On Apple Silicon hosts the device may resolve to "mps", elsewhere "cpu".
    ctx = RuntimeContext.from_config(_mps_config())
    assert ctx.cuda_mode is False
    assert ctx.device in ("mps", "cpu")


def test_cuda_config_produces_cuda_mode():
    with (
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
    # tensor_on_cuda True: pure cuda runtime only
    with (
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

    # tensor_on_cuda False: onnx_cuda uses GPU but returns CPU numpy
    onnx_cuda_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.onnx", compute_runtime="onnx_cuda"),
        )
    )
    with (
        patch(
            "hydra_suite.core.inference.runtime._cuda_device_available",
            return_value="cuda:0",
        ),
        patch(
            "hydra_suite.core.inference.runtime._nvdec_available", return_value=False
        ),
    ):
        ctx2 = RuntimeContext.from_config(onnx_cuda_cfg)
    assert ctx2.cuda_mode is True  # CUDA group
    assert ctx2.tensor_on_cuda is False  # but outputs are CPU numpy

    # tensor_on_cuda False: tensorrt also returns CPU numpy
    trt_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.engine", compute_runtime="tensorrt"),
        )
    )
    with (
        patch(
            "hydra_suite.core.inference.runtime._cuda_device_available",
            return_value="cuda:0",
        ),
        patch(
            "hydra_suite.core.inference.runtime._nvdec_available", return_value=False
        ),
    ):
        ctx3 = RuntimeContext.from_config(trt_cfg)
    assert ctx3.cuda_mode is True
    assert ctx3.tensor_on_cuda is False
