from hydra_suite.runtime.compute_runtime import execution_providers_for
from hydra_suite.runtime.resolver import ResolvedBackend
from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, ONNXRUNTIME_COREML_AVAILABLE


def _names(providers):
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def test_tensorrt_backend_lists_trt_cuda_cpu():
    rb = ResolvedBackend("tensorrt", "cuda", False)
    assert _names(execution_providers_for(rb)) == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_cpu_backend_is_cpu_only():
    rb = ResolvedBackend("torch", "cpu", False)
    assert _names(execution_providers_for(rb)) == ["CPUExecutionProvider"]


def test_native_gpu_torch_backends_are_cpu_only_for_onnx():
    # torch backends never run ONNX, so no accel EP is emitted — CPU only.
    for device in ("cuda", "mps"):
        rb = ResolvedBackend("torch", device, False)
        assert _names(execution_providers_for(rb)) == ["CPUExecutionProvider"]


def test_coreml_backend_uses_coreml_ep_when_available():
    rb = ResolvedBackend("coreml", "mps", False)
    names = _names(execution_providers_for(rb))
    # CPU fallback is always last.
    assert names[-1] == "CPUExecutionProvider"
    if ONNXRUNTIME_COREML_AVAILABLE and MPS_AVAILABLE:
        assert names == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        assert names == ["CPUExecutionProvider"]
