"""Shared compute runtime selection/resolution utilities."""

from .compute_runtime import (
    CANONICAL_RUNTIMES,
    allowed_runtimes_for_pipelines,
    derive_onnx_execution_providers,
    runtime_label,
    supported_runtimes_for_pipeline,
)

__all__ = [
    "CANONICAL_RUNTIMES",
    "runtime_label",
    "supported_runtimes_for_pipeline",
    "allowed_runtimes_for_pipelines",
    "derive_onnx_execution_providers",
]
