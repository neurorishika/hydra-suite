"""Availability-aware torch-device selection for inference backends."""

from __future__ import annotations

from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE


def resolve_torch_device(preference: str | None = None) -> str:
    """Resolve a requested torch device against the current environment.

    Explicit CPU remains opt-in.  CUDA and MPS preferences are honored only
    while that accelerator is available; a project opened on another machine
    therefore falls through to the best available torch device instead of
    trying to construct an unavailable backend.
    """
    requested = str(preference or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda") and TORCH_CUDA_AVAILABLE:
        return "cuda"
    if requested == "mps" and MPS_AVAILABLE:
        return "mps"

    if TORCH_CUDA_AVAILABLE:
        return "cuda"
    if MPS_AVAILABLE:
        return "mps"
    return "cpu"
