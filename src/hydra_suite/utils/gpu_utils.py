"""
GPU utilities and device detection for the HYDRA Suite.

This module provides centralized GPU availability detection and utilities
that can be used throughout the codebase. Supports:
  - CUDA (NVIDIA GPUs via CuPy)
  - MPS (Apple Silicon via PyTorch)
  - Automatic fallback to CPU

Import this module to check GPU availability:
    from hydra_suite.utils.gpu_utils import CUDA_AVAILABLE, MPS_AVAILABLE, GPU_AVAILABLE
"""

import logging
import sys
from importlib.util import find_spec

logger = logging.getLogger(__name__)

# CuPy for CUDA GPU acceleration (NVIDIA GPUs)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    CUPY_AVAILABLE = True
    # Test if CUDA is actually available (not just installed)
    try:
        device = cp.cuda.Device(0)
        CUDA_AVAILABLE = True
    except (cp.cuda.runtime.CUDARuntimeError, Exception):
        CUDA_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False
    CUDA_AVAILABLE = False
    cp = None
    cupy_ndimage = None

# PyTorch for MPS GPU acceleration (Apple Silicon) and CUDA
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()

except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False
    torch = None
    F = None

# Numba for CPU JIT acceleration
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = None
    prange = None

# TensorRT for NVIDIA GPU inference optimization
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = CUDA_AVAILABLE  # TensorRT requires CUDA
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

# ONNX Runtime availability and provider capabilities.
try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
    try:
        ONNXRUNTIME_PROVIDERS = list(ort.get_available_providers() or [])
    except Exception:
        ONNXRUNTIME_PROVIDERS = []
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ONNXRUNTIME_PROVIDERS = []
    ort = None

ONNXRUNTIME_CPU_AVAILABLE = ONNXRUNTIME_AVAILABLE and (
    not ONNXRUNTIME_PROVIDERS or "CPUExecutionProvider" in ONNXRUNTIME_PROVIDERS
)
ONNXRUNTIME_CUDA_AVAILABLE = ONNXRUNTIME_AVAILABLE and any(
    p in ONNXRUNTIME_PROVIDERS
    for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
)
ONNXRUNTIME_COREML_AVAILABLE = ONNXRUNTIME_AVAILABLE and (
    "CoreMLExecutionProvider" in ONNXRUNTIME_PROVIDERS
)

# CoreML Tools for Apple Silicon native model conversion and inference
try:
    import coremltools as _ct  # noqa: F401

    COREMLTOOLS_AVAILABLE = True
except Exception:
    COREMLTOOLS_AVAILABLE = False


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except Exception:
        return False


# SLEAP exported-runtime capabilities.
SLEAP_NN_EXPORT_AVAILABLE = bool(
    _module_exists("sleap_nn.export.predictors")
    and _module_exists("sleap_nn.export.metadata")
)
SLEAP_RUNTIME_ONNX_AVAILABLE = SLEAP_NN_EXPORT_AVAILABLE and ONNXRUNTIME_AVAILABLE
SLEAP_RUNTIME_TENSORRT_AVAILABLE = SLEAP_NN_EXPORT_AVAILABLE and TENSORRT_AVAILABLE

# Summary flags
GPU_AVAILABLE = CUDA_AVAILABLE or MPS_AVAILABLE
ANY_ACCELERATION = GPU_AVAILABLE or NUMBA_AVAILABLE


def _safe_assign(info: dict, key: str, getter) -> None:
    """Safely assign a lazily-evaluated value into *info*."""
    try:
        info[key] = getter()
    except Exception:
        pass


def _collect_version_info(info: dict) -> None:
    """Populate library version strings into *info* dict."""
    version_sources = [
        (CUPY_AVAILABLE, "cupy_version", lambda: cp.__version__),
        (TORCH_AVAILABLE, "torch_version", lambda: torch.__version__),
        (
            TENSORRT_AVAILABLE and trt is not None,
            "tensorrt_version",
            lambda: trt.__version__,
        ),
        (
            ONNXRUNTIME_AVAILABLE and ort is not None,
            "onnxruntime_version",
            lambda: ort.__version__,
        ),
    ]

    for is_available, key, getter in version_sources:
        if is_available:
            _safe_assign(info, key, getter)

    if NUMBA_AVAILABLE:
        try:
            import numba

            _safe_assign(info, "numba_version", lambda: numba.__version__)
        except Exception:
            pass


def _collect_device_details(info: dict) -> None:
    """Populate hardware-specific device details into *info* dict."""
    if CUDA_AVAILABLE:
        _safe_assign(
            info, "cuda_device_count", lambda: cp.cuda.runtime.getDeviceCount()
        )
        _safe_assign(
            info, "cuda_device_name", lambda: cp.cuda.Device(0).compute_capability
        )

    if MPS_AVAILABLE:
        info["mps_device"] = "Apple Silicon (Metal)"

    if TORCH_CUDA_AVAILABLE:
        _safe_assign(info, "torch_cuda_device_count", lambda: torch.cuda.device_count())
        _safe_assign(
            info, "torch_cuda_device_name", lambda: torch.cuda.get_device_name(0)
        )
        info["backend"] = "CUDA (NVIDIA GPU)"


def get_device_info() -> object:
    """
    Get information about available compute devices.

    Returns:
        dict: Device availability information
    """
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "mps_available": MPS_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "torch_cuda_available": TORCH_CUDA_AVAILABLE,
        "tensorrt_available": TENSORRT_AVAILABLE,
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "onnxruntime_providers": list(ONNXRUNTIME_PROVIDERS),
        "onnxruntime_cpu_available": ONNXRUNTIME_CPU_AVAILABLE,
        "onnxruntime_cuda_available": ONNXRUNTIME_CUDA_AVAILABLE,
        "onnxruntime_coreml_available": ONNXRUNTIME_COREML_AVAILABLE,
        "sleap_nn_export_available": SLEAP_NN_EXPORT_AVAILABLE,
        "sleap_runtime_onnx_available": SLEAP_RUNTIME_ONNX_AVAILABLE,
        "sleap_runtime_tensorrt_available": SLEAP_RUNTIME_TENSORRT_AVAILABLE,
        "numba_available": NUMBA_AVAILABLE,
        "gpu_available": GPU_AVAILABLE,
        "any_acceleration": ANY_ACCELERATION,
    }

    _collect_version_info(info)
    _collect_device_details(info)

    return info


def _log_status(label: str, available: bool, details: list[str] | None = None) -> None:
    """Log a status line with optional indented detail lines."""
    prefix = "✓" if available else "✗"
    availability = "Available" if available else "Not available"
    logger.info(f"{prefix} {label}: {availability}")
    for detail in details or []:
        logger.info(f"  {detail}")


def _cuda_log_details(info: dict) -> list[str]:
    details = []
    if "cuda_device_count" in info:
        details.append(f"Devices: {info['cuda_device_count']}")
    return details


def _torch_cuda_log_details(info: dict) -> list[str]:
    details = []
    if "torch_cuda_device_name" in info:
        details.append(f"Device: {info['torch_cuda_device_name']}")
    details.append("Backend: CUDA (NVIDIA GPU)")
    return details


def _tensorrt_log_details() -> list[str]:
    if trt is None:
        return []
    return [f"Version: {trt.__version__}"]


def _onnxruntime_log_details(info: dict) -> list[str]:
    providers = ", ".join(info.get("onnxruntime_providers", [])) or "unknown"
    return [f"Providers: {providers}"]


def _log_acceleration_summary(info: dict) -> None:
    if info["gpu_available"]:
        logger.info("GPU acceleration: ENABLED")
        return
    if info["numba_available"]:
        logger.info("CPU acceleration: Numba JIT")
        return
    logger.info("CPU acceleration: NumPy only (slow)")


def log_device_info() -> object:
    """Log available compute devices to help with debugging."""
    info = get_device_info()

    logger.info("=" * 60)
    logger.info("Available Compute Devices:")
    logger.info("-" * 60)

    _log_status("CUDA (CuPy)", info["cuda_available"], _cuda_log_details(info))
    _log_status("MPS (Apple Silicon)", info["mps_available"])
    _log_status(
        "CUDA (PyTorch)",
        info["torch_cuda_available"],
        _torch_cuda_log_details(info) if info["torch_cuda_available"] else None,
    )
    _log_status(
        "TensorRT (NVIDIA)",
        info["tensorrt_available"],
        _tensorrt_log_details() if info["tensorrt_available"] else None,
    )
    _log_status(
        "ONNX Runtime",
        info["onnxruntime_available"],
        _onnxruntime_log_details(info) if info["onnxruntime_available"] else None,
    )
    _log_status("SLEAP export predictors", info["sleap_nn_export_available"])
    _log_status("Numba JIT", info["numba_available"])

    logger.info("-" * 60)
    _log_acceleration_summary(info)
    logger.info("=" * 60)
    return info


def _append_runtime_option(
    options: list[tuple[str, str]], label: str, value: str
) -> None:
    options.append((label, value))


def _add_macos_pose_runtime_options(
    options: list[tuple[str, str]],
    supports_onnx: bool,
    supports_coreml: bool,
) -> None:
    if MPS_AVAILABLE:
        _append_runtime_option(options, "MPS", "mps")
    _append_runtime_option(options, "CPU", "cpu")
    if supports_onnx:
        if supports_coreml:
            _append_runtime_option(options, "ONNX (CoreML)", "onnx_coreml")
        _append_runtime_option(options, "ONNX (CPU)", "onnx_cpu")


def _add_linux_pose_runtime_options(
    options: list[tuple[str, str]],
    supports_onnx: bool,
    supports_tensorrt: bool,
    cuda_like: bool,
) -> None:
    _append_runtime_option(options, "CPU", "cpu")

    if TORCH_CUDA_AVAILABLE:
        _append_runtime_option(options, "CUDA", "cuda")

    if supports_onnx:
        _append_runtime_option(options, "ONNX (CPU)", "onnx_cpu")
        if ONNXRUNTIME_CUDA_AVAILABLE and TORCH_CUDA_AVAILABLE:
            _append_runtime_option(options, "ONNX (CUDA)", "onnx_cuda")

    if supports_tensorrt and cuda_like:
        _append_runtime_option(options, "TensorRT (CUDA)", "tensorrt_cuda")


def _dedupe_runtime_options(options: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    deduped = []
    for label, value in options:
        if value in seen:
            continue
        seen.add(value)
        deduped.append((label, value))
    return deduped


def get_optimal_device(enable_gpu: object = True, prefer_cuda: object = True) -> object:
    """
    Get the optimal compute device based on availability.

    Args:
        enable_gpu: Whether to use GPU if available
        prefer_cuda: Prefer CUDA over MPS if both available

    Returns:
        tuple: (device_type, device_object)
            device_type: 'cuda', 'mps', or 'cpu'
            device_object: GPU device object or None for CPU
    """
    if not enable_gpu:
        return "cpu", None

    # Prefer CUDA if requested and available
    if prefer_cuda and CUDA_AVAILABLE:
        return "cuda", cp.cuda.Device(0)

    # Then try MPS
    if MPS_AVAILABLE:
        return "mps", torch.device("mps")

    # Then CUDA if not already tried
    if not prefer_cuda and CUDA_AVAILABLE:
        return "cuda", cp.cuda.Device(0)

    # Fallback to CPU
    return "cpu", None


def get_pose_runtime_options(backend_family: str = "yolo"):
    """
    Return runtime options for pose inference as list[(label, value)].

    Values are normalized ids consumed by runtime_api, e.g.:
    - auto
    - cpu / mps / cuda
    - onnx_coreml / onnx_cpu / onnx_cuda
    - tensorrt_cuda
    """
    backend = str(backend_family or "yolo").strip().lower()
    is_mac = sys.platform == "darwin"
    options = [("Auto", "auto")]

    supports_onnx = (
        ONNXRUNTIME_AVAILABLE if backend != "sleap" else SLEAP_RUNTIME_ONNX_AVAILABLE
    )
    supports_coreml = bool(
        backend != "sleap" and MPS_AVAILABLE and ONNXRUNTIME_COREML_AVAILABLE
    )
    supports_tensorrt = (
        TENSORRT_AVAILABLE if backend != "sleap" else SLEAP_RUNTIME_TENSORRT_AVAILABLE
    )
    cuda_like = CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE

    if is_mac:
        _add_macos_pose_runtime_options(options, supports_onnx, supports_coreml)
    else:
        _add_linux_pose_runtime_options(
            options,
            supports_onnx,
            supports_tensorrt,
            cuda_like,
        )

    return _dedupe_runtime_options(options)


# Export all availability flags and utilities
__all__ = [
    # Flags
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "TORCH_CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "TENSORRT_AVAILABLE",
    "ONNXRUNTIME_AVAILABLE",
    "ONNXRUNTIME_PROVIDERS",
    "ONNXRUNTIME_CPU_AVAILABLE",
    "ONNXRUNTIME_CUDA_AVAILABLE",
    "ONNXRUNTIME_COREML_AVAILABLE",
    "SLEAP_NN_EXPORT_AVAILABLE",
    "SLEAP_RUNTIME_ONNX_AVAILABLE",
    "SLEAP_RUNTIME_TENSORRT_AVAILABLE",
    "NUMBA_AVAILABLE",
    "GPU_AVAILABLE",
    "ANY_ACCELERATION",
    # Modules (may be None)
    "cp",
    "cupy_ndimage",
    "torch",
    "F",
    "njit",
    "prange",
    "trt",
    "ort",
    # Functions
    "get_device_info",
    "log_device_info",
    "get_optimal_device",
    "get_pose_runtime_options",
]
