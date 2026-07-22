"""Accelerated-runner selection for pose backends (ViTPose).

Composes the moved OnnxSessionRunner / TensorRTEngineRunner into a simple
fallback ladder. Unlike SLEAP's instance-coupled ladder, this is a pure
function: a native .engine/.trt deserializes to a TensorRTEngineRunner; on
failure (or an .onnx input) it falls back to an ONNX Runtime session using the
TensorRT execution provider.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydra_suite.runtime.resolver import ResolvedBackend

from .onnx_session import OnnxSessionRunner
from .tensorrt_engine import TensorRTEngineRunner

logger = logging.getLogger(__name__)


def _sibling_onnx(model_path: Path):
    if model_path.suffix.lower() == ".onnx":
        return model_path
    siblings = sorted(model_path.parent.rglob("*.onnx"))
    return siblings[0] if siblings else None


def build_accelerated_runner(model_path: Path, resolved: ResolvedBackend):
    """Return the best available runner for *model_path* under *resolved*."""
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    if suffix in (".engine", ".trt"):
        try:
            return TensorRTEngineRunner(model_path)
        except Exception as exc:  # incl. ImportError when tensorrt absent
            logger.warning(
                "TensorRT engine %s failed to load (%s); falling back to ONNX "
                "Runtime TensorRT-EP.",
                model_path,
                exc,
            )
            onnx = _sibling_onnx(model_path)
            if onnx is None:
                raise
            return OnnxSessionRunner(onnx, resolved)
    if suffix == ".onnx":
        return OnnxSessionRunner(model_path, resolved)
    raise ValueError(f"build_accelerated_runner: unsupported artifact {model_path!r}")
