"""ONNX Runtime execution-provider plumbing.

Builds the ordered ONNX Runtime provider list for a resolved Gen-2 backend
(``ResolvedBackend`` from ``runtime.resolver``), plus the CoreML and TensorRT
provider-option payloads and small provider dedup helpers. This is the only
ONNX-specific glue left after the legacy canonical-runtime surface was retired.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, ONNXRUNTIME_COREML_AVAILABLE

COREML_PROVIDER_OPTIONS = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
}


def _provider_name(provider: object) -> str:
    if isinstance(provider, tuple) and provider:
        return str(provider[0])
    return str(provider)


def _append_provider(providers: list[object], provider: object) -> None:
    name = _provider_name(provider)
    if any(_provider_name(existing) == name for existing in providers):
        return
    providers.append(provider)


def has_tensorrt_provider(providers) -> bool:
    """True when the TensorRT execution provider is in *providers*.

    Session creation with the TensorRT EP triggers an engine build into the
    shared per-machine cache, so callers use this to decide whether that
    session needs an ``artifact_build_lock``.
    """
    return any(
        _provider_name(provider) == "TensorrtExecutionProvider"
        for provider in providers
    )


def _trt_engine_cache_dir() -> Path:
    """The shared per-machine ORT TensorRT engine/timing cache directory.

    Imported lazily so tests can monkeypatch ``hydra_suite.paths.get_data_dir``.
    Raises whatever ``get_data_dir``/``mkdir`` raises; callers decide.
    """
    from hydra_suite.paths import get_data_dir

    cache_dir = Path(get_data_dir()) / "trt_engine_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def tensorrt_ep_lock_target(model_path) -> Path:
    """Lock target for a TRT-EP session build of *model_path*.

    The resource an ORT TensorRT-EP session build actually contends on is the
    shared engine cache directory, NOT the model file -- and the model may sit
    in a read-only directory (a mounted model share) where no ``.lock`` can be
    created. Keying the lock by a hash of the resolved model path inside the
    engine cache dir puts the lock next to the thing it protects, on a
    directory that is writable by construction. Falls back to the model-side
    path only when the data dir cannot be resolved at all.
    """
    model = Path(str(model_path)).expanduser()
    try:
        model = model.resolve()
    except OSError:  # pragma: no cover - resolve() is near-total on POSIX
        pass
    digest = hashlib.sha256(str(model).encode("utf-8")).hexdigest()[:24]
    try:
        return _trt_engine_cache_dir() / f"{digest}.trt_ep"
    except Exception:  # pragma: no cover - path resolution should not fail
        return Path(str(model_path)).with_suffix(".trt_ep")


def _tensorrt_ep_cache_options() -> dict:
    """Provider options that make the ORT TensorRT-EP plan persist across runs.

    Without a cache path ORT rebuilds the TensorRT engine on every session — a
    ~8-16 s stall each time the SLEAP (or any) ONNX model is loaded via the
    TensorRT-EP fallback. Pointing the engine + timing cache at a stable
    per-machine directory pays that build cost once. Engines are keyed by model
    + shape profile, so distinct models never collide. fp16 is deliberately NOT
    enabled here — keypoint/OBB precision must stay ~identical to native CUDA.
    """
    try:
        cache = str(_trt_engine_cache_dir())
    except Exception:  # pragma: no cover - path resolution should not fail
        return {}
    return {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache,
    }


def execution_providers_for(
    resolved, include_cpu_fallback: bool = True
) -> List[object]:
    """ONNX EP list keyed off a ResolvedBackend (the Gen-2 vocabulary).

    torch backends never run ONNX -> CPU only; tensorrt -> TRT-EP (with a
    persistent engine cache) + CUDA-EP; coreml -> CoreML-EP when ONNX Runtime's
    CoreML provider is present on an MPS host. A CPU provider is appended as a
    fallback when *include_cpu_fallback* is set or nothing else was added.
    """
    providers: list[object] = []
    if resolved.backend == "tensorrt":
        _append_provider(
            providers,
            ("TensorrtExecutionProvider", _tensorrt_ep_cache_options()),
        )
        _append_provider(providers, "CUDAExecutionProvider")
    elif (
        resolved.backend == "coreml" and ONNXRUNTIME_COREML_AVAILABLE and MPS_AVAILABLE
    ):
        _append_provider(
            providers,
            ("CoreMLExecutionProvider", dict(COREML_PROVIDER_OPTIONS)),
        )

    if include_cpu_fallback or not providers:
        _append_provider(providers, "CPUExecutionProvider")
    return providers
