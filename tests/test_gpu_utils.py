from __future__ import annotations

import types

from tests.helpers.module_loader import load_src_module


def _load_mod():
    return load_src_module(
        "hydra_suite/utils/gpu_utils.py",
        "gpu_utils_under_test",
    )


def test_get_pose_runtime_options_on_macos_includes_mps_coreml_and_onnx_cpu(
    monkeypatch,
) -> None:
    mod = _load_mod()
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", True)
    monkeypatch.setattr(mod, "SLEAP_RUNTIME_ONNX_AVAILABLE", False)
    monkeypatch.setattr(mod, "TENSORRT_AVAILABLE", False)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", False)

    options = mod.get_pose_runtime_options("yolo")

    assert options == [
        ("Auto", "auto"),
        ("MPS", "mps"),
        ("CPU", "cpu"),
        ("ONNX (CoreML)", "onnx_coreml"),
        ("ONNX (CPU)", "onnx_cpu"),
    ]


def test_get_pose_runtime_options_on_linux_includes_cuda_onnx_and_tensorrt(
    monkeypatch,
) -> None:
    mod = _load_mod()
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TENSORRT_AVAILABLE", True)

    options = mod.get_pose_runtime_options("yolo")

    assert options == [
        ("Auto", "auto"),
        ("CPU", "cpu"),
        ("CUDA", "cuda"),
        ("ONNX (CPU)", "onnx_cpu"),
        ("ONNX (CUDA)", "onnx_cuda"),
        ("TensorRT (CUDA)", "tensorrt_cuda"),
    ]


def test_get_optimal_device_respects_preference_order(monkeypatch) -> None:
    mod = _load_mod()
    fake_cuda_device = object()
    fake_cp = types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=lambda _idx: fake_cuda_device)
    )
    fake_torch = types.SimpleNamespace(device=lambda name: ("torch-device", name))

    monkeypatch.setattr(mod, "cp", fake_cp)
    monkeypatch.setattr(mod, "torch", fake_torch)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)

    assert mod.get_optimal_device(enable_gpu=True, prefer_cuda=True) == (
        "cuda",
        fake_cuda_device,
    )
    assert mod.get_optimal_device(enable_gpu=True, prefer_cuda=False) == (
        "mps",
        ("torch-device", "mps"),
    )
    assert mod.get_optimal_device(enable_gpu=False, prefer_cuda=True) == ("cpu", None)


def test_get_device_info_collects_versions_and_device_details(monkeypatch) -> None:
    mod = _load_mod()
    fake_cp = types.SimpleNamespace(
        __version__="13.0",
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 2),
            Device=lambda _idx: types.SimpleNamespace(compute_capability="8.9"),
        ),
    )
    fake_torch = types.SimpleNamespace(
        __version__="2.7",
        cuda=types.SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda _idx: "Fake GPU",
        ),
        version=types.SimpleNamespace(hip="6.0"),
    )
    fake_ort = types.SimpleNamespace(__version__="1.22")

    monkeypatch.setattr(mod, "cp", fake_cp)
    monkeypatch.setattr(mod, "torch", fake_torch)
    monkeypatch.setattr(mod, "ort", fake_ort)
    monkeypatch.setattr(mod, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "MPS_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_PROVIDERS", ["CPUExecutionProvider"])
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", False)
    monkeypatch.setattr(mod, "TENSORRT_AVAILABLE", False)
    monkeypatch.setattr(mod, "NUMBA_AVAILABLE", False)
    monkeypatch.setattr(mod, "SLEAP_NN_EXPORT_AVAILABLE", False)
    monkeypatch.setattr(mod, "SLEAP_RUNTIME_ONNX_AVAILABLE", False)
    monkeypatch.setattr(mod, "SLEAP_RUNTIME_TENSORRT_AVAILABLE", False)
    monkeypatch.setattr(mod, "GPU_AVAILABLE", True)
    monkeypatch.setattr(mod, "ANY_ACCELERATION", True)

    info = mod.get_device_info()

    assert info["cupy_version"] == "13.0"
    assert info["torch_version"] == "2.7"
    assert info["onnxruntime_version"] == "1.22"
    assert info["onnxruntime_coreml_available"] is False
    assert info["cuda_device_count"] == 2
    assert info["cuda_device_name"] == "8.9"
    assert info["torch_cuda_device_name"] == "Fake GPU"
    assert info["backend"] == "CUDA (NVIDIA GPU)"
