from hydra_suite.runtime.compute_runtime import (
    derive_onnx_execution_providers,
    execution_providers_for,
)
from hydra_suite.runtime.resolver import ResolvedBackend


def _names(providers):
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def test_tensorrt_matches_string_api():
    rb = ResolvedBackend("tensorrt", "cuda", False)
    assert _names(execution_providers_for(rb)) == _names(
        derive_onnx_execution_providers("tensorrt")
    )


def test_cpu_backend_is_cpu_only():
    rb = ResolvedBackend("torch", "cpu", False)
    assert _names(execution_providers_for(rb)) == ["CPUExecutionProvider"]


def test_coreml_matches_string_api():
    rb = ResolvedBackend("coreml", "mps", False)
    assert _names(execution_providers_for(rb)) == _names(
        derive_onnx_execution_providers("coreml")
    )
