"""Compute-runtime helpers and fallback stubs for PoseKit inference settings."""

try:
    from hydra_suite.runtime.compute_runtime import (
        CANONICAL_RUNTIMES,
        allowed_runtimes_for_pipelines,
        available_tiers,
        derive_pose_runtime_settings,
        infer_compute_runtime_from_legacy,
        runtime_label,
        tier_label,
    )
    from hydra_suite.runtime.resolver import PlatformInfo, detect_platform
except Exception:
    CANONICAL_RUNTIMES = [
        "cpu",
        "mps",
        "cuda",
        "onnx_coreml",
        "onnx_cpu",
        "onnx_cuda",
        "tensorrt",
    ]

    def runtime_label(runtime: str) -> str:
        """Return a human-readable uppercase label for a canonical runtime identifier."""
        return str(runtime or "cpu").strip().upper()

    def allowed_runtimes_for_pipelines(_pipelines):
        """Return the set of runtimes supported by all given pipeline keys (fallback: cpu only)."""
        return ["cpu"]

    def infer_compute_runtime_from_legacy(
        yolo_device, enable_tensorrt, pose_runtime_flavor
    ):
        """Derive the canonical compute-runtime string from legacy per-field settings."""
        if enable_tensorrt:
            return "tensorrt"
        flavor = str(pose_runtime_flavor or "").lower()
        if flavor.startswith("onnx_cuda"):
            return "onnx_cuda"
        if flavor.startswith("onnx_mps") or flavor.startswith("onnx_coreml"):
            return "onnx_coreml"
        if flavor.startswith("onnx"):
            return "onnx_cpu"
        return "cpu"

    def derive_pose_runtime_settings(compute_runtime: str, backend_family: str):
        """Translate a canonical compute-runtime key into backend-specific runtime settings dict."""
        rt = str(compute_runtime or "cpu").strip().lower()
        if rt == "onnx_coreml":
            return {"pose_runtime_flavor": "onnx_mps", "pose_sleap_device": "mps"}
        if rt == "onnx_cpu":
            return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
        if rt in {"onnx_cuda"}:
            return {"pose_runtime_flavor": rt, "pose_sleap_device": "cuda:0"}
        if rt == "tensorrt":
            return {
                "pose_runtime_flavor": "tensorrt_cuda",
                "pose_sleap_device": "cuda:0",
            }
        return {"pose_runtime_flavor": rt or "cpu", "pose_sleap_device": rt or "cpu"}

    class PlatformInfo:  # type: ignore[no-redef]
        """Fallback stub when runtime resolver is unavailable."""

        def __init__(self, has_cuda: bool = False, has_mps: bool = False):
            self.has_cuda = has_cuda
            self.has_mps = has_mps

    def detect_platform():  # type: ignore[misc]
        """Fallback: return CPU-only platform."""
        return PlatformInfo(has_cuda=False, has_mps=False)

    def available_tiers(_platform) -> list:  # type: ignore[misc]
        """Fallback: CPU-only tier list."""
        return ["cpu"]

    def tier_label(tier: str, _platform) -> str:  # type: ignore[misc]
        """Fallback: uppercase tier label."""
        return str(tier or "cpu").upper()


# ---------------------------------------------------------------------------
# Tier → canonical-runtime mapping
# ---------------------------------------------------------------------------

_TIER_TO_RUNTIME_CUDA = {
    "cpu": "cpu",
    "gpu": "cuda",
    "gpu_fast": "tensorrt",
}
_TIER_TO_RUNTIME_MPS = {
    "cpu": "cpu",
    "gpu": "mps",
    "gpu_fast": "onnx_coreml",
}
_TIER_TO_RUNTIME_CPU_ONLY = {
    "cpu": "cpu",
    "gpu": "cpu",
    "gpu_fast": "cpu",
}


def tier_to_canonical_runtime(tier: str, platform) -> str:
    """Map a runtime tier string to the best canonical runtime for *platform*.

    - ``"cpu"``      → ``"cpu"``
    - ``"gpu"``      → ``"cuda"`` (CUDA) / ``"mps"`` (Apple Silicon) / ``"cpu"``
    - ``"gpu_fast"`` → ``"tensorrt"`` (CUDA) / ``"onnx_coreml"`` (MPS) / ``"cpu"``
    """
    if getattr(platform, "has_cuda", False):
        table = _TIER_TO_RUNTIME_CUDA
    elif getattr(platform, "has_mps", False):
        table = _TIER_TO_RUNTIME_MPS
    else:
        table = _TIER_TO_RUNTIME_CPU_ONLY
    return table.get(str(tier or "cpu"), "cpu")


def canonical_runtime_to_tier(runtime: str) -> str:
    """Map a canonical runtime string back to the coarsest matching tier.

    Used when restoring a persisted canonical runtime into the tier combo.
    """
    rt = str(runtime or "cpu").strip().lower()
    if rt in {"tensorrt", "onnx_coreml", "onnx_cpu", "onnx_cuda"}:
        return "gpu_fast"
    if rt in {"cuda", "mps"}:
        return "gpu"
    return "cpu"
