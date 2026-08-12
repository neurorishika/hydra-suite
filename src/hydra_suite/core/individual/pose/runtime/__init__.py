"""Shared pose runtime: reusable ONNX / TensorRT runners and engine builder.

Extracted from ``backends/sleap.py`` so pose backends (SLEAP, ViTPose) can
share the same ONNX Runtime session wrapper, native TensorRT engine runner,
and ONNX-to-TensorRT engine builder.
"""

from __future__ import annotations

from .onnx_session import OnnxSessionRunner
from .tensorrt_engine import TensorRTEngineRunner, build_trt_engine_from_onnx

__all__ = [
    "OnnxSessionRunner",
    "TensorRTEngineRunner",
    "build_trt_engine_from_onnx",
]
