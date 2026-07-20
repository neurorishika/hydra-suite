from __future__ import annotations

import importlib


def _load_mod():
    return importlib.import_module("hydra_suite.runtime.compute_runtime")


def test_allowed_runtimes_intersection_includes_explicit_onnx_variants(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", True)

    allowed = mod.allowed_runtimes_for_pipelines(["yolo_obb_detection", "yolo_pose"])
    assert "onnx_cpu" in allowed
    assert "onnx_cuda" in allowed
    assert "cpu" in allowed
    assert "cuda" in allowed
    assert "tensorrt" in allowed


def test_allowed_runtimes_for_sleap_pose_tracks_runtime_availability(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "TENSORRT_AVAILABLE", False)
    monkeypatch.setattr(mod, "SLEAP_RUNTIME_TENSORRT_AVAILABLE", False)

    allowed = mod.allowed_runtimes_for_pipelines(["sleap_pose"])
    assert allowed == ["cpu", "mps"]


def test_allowed_runtimes_for_sleap_pose_exposes_onnx_cpu_when_available(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", False)

    allowed = mod.allowed_runtimes_for_pipelines(["sleap_pose"])
    assert "onnx_cpu" in allowed
    assert "onnx_cuda" not in allowed


def test_allowed_runtimes_for_sleap_pose_exposes_onnx_coreml_when_available(
    monkeypatch,
):
    mod = _load_mod()
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", False)

    allowed = mod.allowed_runtimes_for_pipelines(["sleap_pose"])
    assert "onnx_coreml" in allowed


def test_allowed_runtimes_for_sleap_pose_exposes_tensorrt_when_available(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "MPS_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", False)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(mod, "TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(mod, "SLEAP_RUNTIME_TENSORRT_AVAILABLE", True)

    allowed = mod.allowed_runtimes_for_pipelines(["sleap_pose"])
    assert "tensorrt" in allowed


def test_allowed_runtimes_includes_onnx_coreml_on_mps(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CPU_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "CUDA_AVAILABLE", False)
    monkeypatch.setattr(mod, "TORCH_CUDA_AVAILABLE", False)

    allowed = mod.allowed_runtimes_for_pipelines(["yolo_obb_detection", "yolo_pose"])
    assert "onnx_coreml" in allowed


def test_normalize_runtime_maps_coreml_aliases_to_canonical():
    mod = _load_mod()
    assert mod._normalize_runtime("onnx_core_ml") == "onnx_coreml"
    assert mod._normalize_runtime("onnx_apple") == "onnx_coreml"
    assert mod._normalize_runtime("onnx_metal") == "onnx_coreml"


def test_derive_onnx_execution_providers_for_coreml(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod, "MPS_AVAILABLE", True)
    monkeypatch.setattr(mod, "ONNXRUNTIME_COREML_AVAILABLE", True)
    providers = mod.derive_onnx_execution_providers("onnx_coreml")
    assert providers[0][0] == "CoreMLExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"
