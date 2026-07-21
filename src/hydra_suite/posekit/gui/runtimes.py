"""Compute-runtime helpers for PoseKit inference settings.

Thin wrapper over ``hydra_suite.runtime.resolver`` — PoseKit has no
independent capability table; it resolves tiers exactly like the main
tracking pipeline (native TensorRT/CoreML on gpu_fast, never ONNX).
"""

from hydra_suite.runtime.resolver import (
    PlatformInfo,
    available_tiers,
    detect_platform,
    resolve_compute_runtime,
    tier_label,
)

__all__ = [
    "PlatformInfo",
    "available_tiers",
    "canonical_runtime_to_tier",
    "detect_platform",
    "tier_label",
    "tier_to_canonical_runtime",
]


def tier_to_canonical_runtime(
    tier: str, platform: PlatformInfo, stage: str = "yolo_pose"
) -> str:
    """Map a runtime tier to the compute-runtime string PoseKit's backends expect.

    ``stage`` selects the capability profile ("yolo_pose" default; pass
    "sleap_pose" for SLEAP models, matching core/inference/stages/pose.py's
    per-backend resolution).
    """
    return resolve_compute_runtime(tier, platform, stage=stage)


def canonical_runtime_to_tier(runtime: str) -> str:
    """Map a legacy saved canonical-runtime string back to the coarsest matching tier.

    Migration-only: used when loading an old saved config that predates the
    tier system. Never used to drive a live inference decision.
    """
    rt = str(runtime or "cpu").strip().lower()
    if rt in {"tensorrt", "coreml", "onnx_coreml", "onnx_cpu", "onnx_cuda"}:
        return "gpu_fast"
    if rt in {"cuda", "mps"}:
        return "gpu"
    return "cpu"
