"""Single authority mapping a runtime tier + platform + stage to a concrete backend.

Replaces per-stage compute_runtime selection and the ONNX/TensorRT capability
tables. Pure and deterministic: no torch import, no I/O — availability is passed
in as a callable so callers own artifact discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

RuntimeTier = Literal["cpu", "gpu", "gpu_fast"]

STAGES = ("obb", "head_tail", "cnn", "yolo_pose", "sleap_pose", "vitpose_pose", "bgsub")


@dataclass(frozen=True)
class PlatformInfo:
    has_cuda: bool
    has_mps: bool


@dataclass(frozen=True)
class ResolvedBackend:
    backend: Literal["torch", "tensorrt", "coreml"]
    device: Literal["cpu", "cuda", "mps"]
    used_fallback: bool


class RuntimeResolver:
    def __init__(self, tier: RuntimeTier, platform: PlatformInfo) -> None:
        self.tier = tier
        self.platform = platform

    def _native_gpu(self) -> tuple[str, str]:
        """Return (backend, device) for the native-GPU tier, or CPU degrade."""
        if self.platform.has_cuda:
            return ("torch", "cuda")
        if self.platform.has_mps:
            return ("torch", "mps")
        return ("torch", "cpu")

    def resolve(
        self,
        stage: str,
        artifact_available: Callable[[], bool] = lambda: True,
    ) -> ResolvedBackend:
        if self.tier == "cpu":
            return ResolvedBackend("torch", "cpu", False)

        if self.tier == "gpu":
            backend, device = self._native_gpu()
            return ResolvedBackend(backend, device, used_fallback=(device == "cpu"))

        if stage == "bgsub":
            # bg-sub has no TensorRT/CoreML implementation -- it is elementwise
            # work, not a network. gpu_fast resolves to the same backend as
            # gpu, always flagged as a fallback (the "fast" tier bought nothing).
            # The "cpu" and "gpu" tiers already returned above, so self.tier
            # can only be "gpu_fast" here -- no need to check it again.
            backend, device = self._native_gpu()
            return ResolvedBackend(backend, device, used_fallback=True)

        # gpu_fast
        if self.platform.has_cuda:
            if artifact_available():
                return ResolvedBackend("tensorrt", "cuda", False)
            return ResolvedBackend("torch", "cuda", used_fallback=True)
        if self.platform.has_mps:
            if artifact_available():
                return ResolvedBackend("coreml", "mps", False)
            return ResolvedBackend("torch", "mps", used_fallback=True)
        return ResolvedBackend("torch", "cpu", used_fallback=True)


def available_tiers(platform: PlatformInfo) -> list:
    """Return the ordered list of runtime tiers available on *platform*.

    Returns ``["cpu"]`` when no GPU accelerator is present, or
    ``["cpu", "gpu", "gpu_fast"]`` when CUDA or MPS is available.
    """
    if not (platform.has_cuda or platform.has_mps):
        return ["cpu"]
    return ["cpu", "gpu", "gpu_fast"]


def tier_label(tier: str, platform: PlatformInfo) -> str:
    """Return a human-readable label for *tier* on *platform*.

    - ``"cpu"``      -> ``"CPU"``
    - ``"gpu"``      -> ``"GPU (CUDA)"`` / ``"GPU (Metal)"``
    - ``"gpu_fast"`` -> ``"GPU-Fast (TensorRT)"`` / ``"GPU-Fast (CoreML)"``
    """
    if tier == "cpu":
        return "CPU"
    accel = "CUDA" if platform.has_cuda else ("Metal" if platform.has_mps else "CPU")
    fast = (
        "TensorRT" if platform.has_cuda else ("CoreML" if platform.has_mps else "CPU")
    )
    return {"gpu": f"GPU ({accel})", "gpu_fast": f"GPU-Fast ({fast})"}[tier]


def detect_platform() -> PlatformInfo:
    """Detect host acceleration via the existing gpu_utils availability flags."""
    from hydra_suite.utils.gpu_utils import CUDA_AVAILABLE, MPS_AVAILABLE

    return PlatformInfo(has_cuda=bool(CUDA_AVAILABLE), has_mps=bool(MPS_AVAILABLE))
