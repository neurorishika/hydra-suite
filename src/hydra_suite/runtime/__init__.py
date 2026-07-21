"""Shared compute runtime selection/resolution utilities."""

from .onnx_providers import execution_providers_for

__all__ = [
    "execution_providers_for",
]
