"""Canonical compute runtime capability and translation helpers.

This module defines one user-facing runtime enum and translates it into
backend-specific settings for detection and pose inference.
"""

from __future__ import annotations

from typing import Iterable, List

from hydra_suite.utils.gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    ONNXRUNTIME_COREML_AVAILABLE,
    ONNXRUNTIME_CPU_AVAILABLE,
    ONNXRUNTIME_CUDA_AVAILABLE,
    SLEAP_RUNTIME_TENSORRT_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

# onnx_* entries kept for legacy-config migration only — not user-selectable.
CANONICAL_RUNTIMES: List[str] = [
    "cpu",
    "mps",
    "cuda",
    "onnx_coreml",
    "onnx_cpu",
    "onnx_cuda",
    "tensorrt",
]

COREML_PROVIDER_OPTIONS = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
}


def _best_explicit_onnx_runtime() -> str:  # legacy-config migration only
    if ONNXRUNTIME_CUDA_AVAILABLE and _cuda_like_available():
        return "onnx_cuda"
    if ONNXRUNTIME_COREML_AVAILABLE and MPS_AVAILABLE:
        return "onnx_coreml"
    if ONNXRUNTIME_CPU_AVAILABLE or ONNXRUNTIME_AVAILABLE:
        return "onnx_cpu"
    return "onnx_cpu"


def _normalize_runtime(
    runtime: str,
) -> str:  # onnx_* aliases kept for legacy-config migration
    rt = str(runtime or "cpu").strip().lower().replace("-", "_")
    if rt in {"", "auto"}:
        # Canonical runtime set intentionally excludes auto.
        # Default to CPU for deterministic fallback.
        return "cpu"
    if rt == "onnxruntime":
        return _best_explicit_onnx_runtime()
    if rt in {"trt", "tensor_rt"}:
        return "tensorrt"
    if rt == "onnx":
        return _best_explicit_onnx_runtime()
    if rt == "onnx_gpu":
        return "onnx_cuda"
    if rt in {"onnx_cpu", "onnx_cuda", "onnx_coreml", "onnx_mps"}:
        return "onnx_coreml" if rt == "onnx_mps" else rt
    if rt == "coreml":
        return "onnx_coreml"
    if rt in {"onnx_core_ml", "onnx_coreml", "onnx_apple", "onnx_metal"}:
        return "onnx_coreml"
    if rt.startswith("tensorrt"):
        return "tensorrt"
    if rt.startswith("cuda"):
        return "cuda"
    return rt if rt in CANONICAL_RUNTIMES else "cpu"


def runtime_label(runtime: str) -> str:
    """Return a human-readable display name for a canonical runtime identifier."""
    rt = _normalize_runtime(runtime)
    return {
        "cpu": "CPU",
        "mps": "MPS",
        "cuda": "CUDA",
        "onnx_coreml": "ONNX (CoreML)",
        "onnx_cpu": "ONNX (CPU)",
        "onnx_cuda": "ONNX (CUDA)",
        "tensorrt": "TensorRT",
    }[rt]


def _cuda_like_available() -> bool:
    return bool(CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE)


def _onnx_available(rt: str) -> bool:
    """Check local ONNX runtime availability for the given canonical ONNX runtime."""
    if rt == "onnx_coreml":
        return bool(ONNXRUNTIME_COREML_AVAILABLE and MPS_AVAILABLE)
    if rt == "onnx_cpu":
        return bool(ONNXRUNTIME_CPU_AVAILABLE or ONNXRUNTIME_AVAILABLE)
    if rt == "onnx_cuda":
        return bool(ONNXRUNTIME_CUDA_AVAILABLE and _cuda_like_available())
    return False


def _tensorrt_available() -> bool:
    return bool(TENSORRT_AVAILABLE and _cuda_like_available())


def _sleap_onnx_available(rt: str) -> bool:
    """Return whether exported SLEAP ONNX inference is runnable for a runtime.

    Export still depends on a SLEAP conda env, but once exported the inference path
    runs directly inside HYDRA via canonical ONNX Runtime providers.
    """
    return _onnx_available(rt)


def _provider_name(provider: object) -> str:
    if isinstance(provider, tuple) and provider:
        return str(provider[0])
    return str(provider)


def _append_provider(providers: list[object], provider: object) -> None:
    name = _provider_name(provider)
    if any(_provider_name(existing) == name for existing in providers):
        return
    providers.append(provider)


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
        from hydra_suite.paths import get_data_dir

        cache_dir = get_data_dir() / "trt_engine_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = str(cache_dir)
    except Exception:  # pragma: no cover - path resolution should not fail
        return {}
    return {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache,
    }


def derive_onnx_execution_providers(
    compute_runtime: str,
    include_cpu_fallback: bool = True,
) -> List[object]:
    """Return an ordered ONNX Runtime provider list for a canonical runtime."""
    rt = _normalize_runtime(compute_runtime)
    providers: list[object] = []

    if rt == "tensorrt":
        _append_provider(
            providers,
            ("TensorrtExecutionProvider", _tensorrt_ep_cache_options()),
        )
        _append_provider(providers, "CUDAExecutionProvider")
    elif rt == "onnx_cuda":
        _append_provider(providers, "CUDAExecutionProvider")
    elif rt == "onnx_coreml" and ONNXRUNTIME_COREML_AVAILABLE and MPS_AVAILABLE:
        _append_provider(
            providers,
            ("CoreMLExecutionProvider", dict(COREML_PROVIDER_OPTIONS)),
        )

    if include_cpu_fallback or not providers:
        _append_provider(providers, "CPUExecutionProvider")
    return providers


def _pipeline_supports_runtime(pipeline: str, runtime: str) -> bool:
    p = str(pipeline or "").strip().lower()
    rt = _normalize_runtime(runtime)

    # Baseline runtime support independent of pipeline.
    if rt == "cpu":
        return True
    if rt == "mps":
        return bool(MPS_AVAILABLE)
    if rt == "cuda":
        return bool(_cuda_like_available())

    # head_tail shares the identical capability table as cnn_identity.
    if p == "head_tail":
        return _pipeline_supports_runtime("cnn_identity", runtime)

    # bg-sub is elementwise CPU/CuPy/torch work with no exported-model story:
    # it supports cpu/mps/cuda (handled by the baseline checks above) and
    # explicitly does NOT support any ONNX/TensorRT flavor, regardless of
    # host availability.
    if p == "bgsub":
        if rt in {"onnx_coreml", "onnx_cpu", "onnx_cuda", "tensorrt"}:
            return False
        return True

    # SLEAP has its own ONNX/TRT availability logic.
    if p == "sleap_pose":
        if rt in {"onnx_coreml", "onnx_cpu", "onnx_cuda"}:
            return _sleap_onnx_available(rt)
        if rt == "tensorrt":
            return bool(
                (SLEAP_RUNTIME_TENSORRT_AVAILABLE or TENSORRT_AVAILABLE)
                and _cuda_like_available()
            )
        return True

    # All other pipelines use the standard ONNX/TRT availability checks.
    if rt in {"onnx_coreml", "onnx_cpu", "onnx_cuda"}:
        return _onnx_available(rt)
    if rt == "tensorrt":
        return _tensorrt_available()
    return True


def supported_runtimes_for_pipeline(pipeline: str) -> List[str]:
    """Return canonical runtimes supported for a single pipeline."""
    return [rt for rt in CANONICAL_RUNTIMES if _pipeline_supports_runtime(pipeline, rt)]


def allowed_runtimes_for_pipelines(pipelines: Iterable[str]) -> List[str]:
    """Return canonical runtimes allowed for all provided pipelines.

    If no pipelines are provided, returns host-capable runtimes from canonical set.
    """
    pls = [str(p).strip().lower() for p in pipelines if str(p).strip()]
    if not pls:
        return [
            rt for rt in CANONICAL_RUNTIMES if _pipeline_supports_runtime("generic", rt)
        ]

    allowed = []
    for rt in CANONICAL_RUNTIMES:
        if all(_pipeline_supports_runtime(p, rt) for p in pls):
            allowed.append(rt)
    return allowed


def _best_auto_runtime() -> str:
    """Pick the best available canonical runtime for auto-detection."""
    if _tensorrt_available():
        return "tensorrt"
    if MPS_AVAILABLE:
        return "mps"
    if _cuda_like_available():
        return "cuda"
    if ONNXRUNTIME_CUDA_AVAILABLE and _cuda_like_available():
        return "onnx_cuda"
    if ONNXRUNTIME_COREML_AVAILABLE and MPS_AVAILABLE:
        return "onnx_coreml"
    if ONNXRUNTIME_CPU_AVAILABLE or ONNXRUNTIME_AVAILABLE:
        return "onnx_cpu"
    return "cpu"


def _runtime_from_pose_flavor(pose_runtime_flavor: str) -> str | None:
    """Map legacy pose_runtime_flavor to canonical runtime, or None."""
    pr = str(pose_runtime_flavor or "").strip().lower()
    if pr.startswith("tensorrt"):
        return "tensorrt"
    if pr.startswith("onnx_mps") or pr.startswith("onnx_coreml"):
        return "onnx_coreml"
    if pr.startswith("onnx_cuda"):
        return "onnx_cuda"
    if pr.startswith("onnx"):
        return "onnx_cpu"
    return None


_DEVICE_MAP = {
    "mps": "mps",
    "cpu": "cpu",
}
_CUDA_DEVICES = {"cuda", "cuda:0", "gpu"}


def infer_compute_runtime_from_legacy(
    yolo_device: str,
    enable_tensorrt: bool,
    pose_runtime_flavor: str,
) -> str:
    """Infer canonical runtime from legacy config fields."""
    if bool(enable_tensorrt):
        return "tensorrt"

    from_pose = _runtime_from_pose_flavor(pose_runtime_flavor)
    if from_pose is not None:
        return from_pose

    dev = str(yolo_device or "auto").strip().lower()
    if dev in _DEVICE_MAP:
        return _DEVICE_MAP[dev]
    if dev in _CUDA_DEVICES:
        return "cuda"

    return _best_auto_runtime()


def derive_pose_runtime_settings(compute_runtime: str, backend_family: str) -> dict:
    """Map canonical runtime to pose runtime legacy settings consumed by runtime_api."""
    rt = _normalize_runtime(compute_runtime)

    if rt == "cpu":
        return {"pose_runtime_flavor": "cpu", "pose_sleap_device": "cpu"}
    if rt == "mps":
        return {"pose_runtime_flavor": "mps", "pose_sleap_device": "mps"}
    if rt == "cuda":
        return {"pose_runtime_flavor": "cuda", "pose_sleap_device": "cuda:0"}
    if rt == "tensorrt":
        return {
            "pose_runtime_flavor": "tensorrt_cuda",
            "pose_sleap_device": "cuda:0",
        }

    if rt == "onnx_coreml":
        return {"pose_runtime_flavor": "onnx_mps", "pose_sleap_device": "mps"}
    if rt == "onnx_cpu":
        return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
    if rt == "onnx_cuda":
        return {"pose_runtime_flavor": rt, "pose_sleap_device": "cuda:0"}

    # Legacy alias fallback (e.g. compute_runtime="onnx").
    resolved = _best_explicit_onnx_runtime()
    if resolved == "onnx_coreml":
        return {"pose_runtime_flavor": "onnx_mps", "pose_sleap_device": "mps"}
    if resolved == "onnx_cpu":
        return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
    return {"pose_runtime_flavor": resolved, "pose_sleap_device": "cuda:0"}
