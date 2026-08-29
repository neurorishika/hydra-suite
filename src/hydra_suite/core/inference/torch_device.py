"""Single cuda -> mps -> cpu torch device picker for inference backends."""

from __future__ import annotations

from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE


def resolve_torch_device() -> str:
    """Best available torch device: cuda, else mps, else cpu."""
    if TORCH_CUDA_AVAILABLE:
        return "cuda"
    if MPS_AVAILABLE:
        return "mps"
    return "cpu"
