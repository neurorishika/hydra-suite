"""Compute-runtime helpers for PoseKit inference settings.

Thin re-export over ``hydra_suite.runtime.resolver`` — PoseKit has no
independent capability table; it resolves tiers exactly like the main
tracking pipeline (native TensorRT/CoreML on gpu_fast, never ONNX).
"""

from hydra_suite.runtime.resolver import (
    PlatformInfo,
    available_tiers,
    detect_platform,
    tier_label,
)

__all__ = [
    "PlatformInfo",
    "available_tiers",
    "detect_platform",
    "tier_label",
]
