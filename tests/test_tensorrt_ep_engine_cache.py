"""The ORT TensorRT-EP provider must carry a persistent engine-cache path.

Without it, ONNX Runtime rebuilds the TensorRT plan on every session (~8-16s
stall per model load) — the SLEAP pose backend hit this on every run. The
TensorrtExecutionProvider must be emitted as a (name, options) tuple whose
options enable a stable engine + timing cache, and must NOT enable fp16 (to
keep precision ~identical to native CUDA).
"""

from hydra_suite.runtime.compute_runtime import execution_providers_for
from hydra_suite.runtime.resolver import ResolvedBackend

_TENSORRT = ResolvedBackend("tensorrt", "cuda", False)


def _find_trt(providers):
    for p in providers:
        if isinstance(p, tuple) and p and p[0] == "TensorrtExecutionProvider":
            return p
    return None


def test_tensorrt_provider_has_persistent_engine_cache():
    providers = execution_providers_for(_TENSORRT)
    trt = _find_trt(providers)
    assert trt is not None, f"no TensorrtExecutionProvider tuple in {providers}"
    _name, opts = trt
    assert opts.get("trt_engine_cache_enable") is True
    assert opts.get("trt_engine_cache_path")  # non-empty path
    assert opts.get("trt_timing_cache_enable") is True
    assert opts.get("trt_timing_cache_path")
    # Precision must stay fp32 — never silently enable fp16 for SLEAP/OBB.
    assert "trt_fp16_enable" not in opts or opts["trt_fp16_enable"] is False


def test_tensorrt_still_lists_cuda_and_cpu_fallback():
    names = [
        p[0] if isinstance(p, tuple) else p for p in execution_providers_for(_TENSORRT)
    ]
    assert names[0] == "TensorrtExecutionProvider"
    assert "CUDAExecutionProvider" in names
    assert names[-1] == "CPUExecutionProvider"


def test_non_tensorrt_backend_unaffected():
    # A native CUDA (torch) backend must not gain a TRT provider — it is
    # CPU-only for ONNX purposes (the resolver never emits an ONNX-CUDA EP).
    names = [
        p[0] if isinstance(p, tuple) else p
        for p in execution_providers_for(ResolvedBackend("torch", "cuda", False))
    ]
    assert "TensorrtExecutionProvider" not in names
