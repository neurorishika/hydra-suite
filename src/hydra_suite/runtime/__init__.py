"""Shared compute runtime selection/resolution utilities.

Keep this package initializer light.  Resource-limited child processes enter
through :mod:`hydra_suite.runtime.child_bootstrap` specifically so limits can
be installed before accelerator libraries are imported.
"""

from typing import Any


def execution_providers_for(*args: Any, **kwargs: Any):
    """Lazily resolve ONNX providers without importing torch at package import."""
    from .onnx_providers import execution_providers_for as _implementation

    return _implementation(*args, **kwargs)


__all__ = [
    "execution_providers_for",
]
