"""ONNX Runtime session wrapper for exported pose artifacts.

Moved verbatim from ``backends/sleap.py`` (``_DirectOnnxSession`` and the
``_detect_onnx_*`` input-spec helpers) so pose backends can share a single
ONNX Runtime session runner. Behavior is unchanged; only the class name was
renamed ``_DirectOnnxSession`` -> ``OnnxSessionRunner``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hydra_suite.runtime.onnx_providers import execution_providers_for
from hydra_suite.runtime.resolver import ResolvedBackend

logger = logging.getLogger(__name__)


def _detect_onnx_input_spec(
    session: Any,
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    if session is None or not hasattr(session, "get_inputs"):
        return None, None
    try:
        inputs = session.get_inputs()
    except Exception:
        return None, None
    if not inputs:
        return None, None

    try:
        shape = list(getattr(inputs[0], "shape", []) or [])
    except Exception:
        shape = []
    dims: List[int] = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(-1)

    input_hw = None
    input_channels = None
    if len(dims) >= 4:
        if dims[-1] in (1, 3):
            input_channels = int(dims[-1])
            if dims[-3] > 0 and dims[-2] > 0:
                input_hw = (int(dims[-3]), int(dims[-2]))
        elif dims[1] in (1, 3):
            input_channels = int(dims[1])
            if dims[-2] > 0 and dims[-1] > 0:
                input_hw = (int(dims[-2]), int(dims[-1]))
    return input_hw, input_channels


def _detect_onnx_input_format(session: Any) -> Optional[Dict[str, Any]]:
    if session is None or not hasattr(session, "get_inputs"):
        return None
    try:
        inputs = session.get_inputs()
        if not inputs:
            return None
        inp = inputs[0]
        raw_type = str(getattr(inp, "type", "")).lower()
        shape = list(getattr(inp, "shape", []) or [])
    except Exception:
        return None

    dims: List[int] = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(-1)

    layout = "nhwc"
    if len(dims) >= 4:
        if dims[1] in (1, 3):
            layout = "nchw"
        elif dims[-1] in (1, 3):
            layout = "nhwc"
    return {"layout": layout, "is_float": "float" in raw_type}


def _detect_onnx_min_batch(session: Any) -> Optional[int]:
    if session is None or not hasattr(session, "get_inputs"):
        return None
    try:
        inputs = session.get_inputs()
        if not inputs:
            return None
        shape = getattr(inputs[0], "shape", [])
        if not shape:
            return None
        batch = int(shape[0])
        return batch if batch > 0 else None
    except Exception:
        return None


class OnnxSessionRunner:
    def __init__(
        self,
        model_path,
        resolved: ResolvedBackend,
    ) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path),
            providers=execution_providers_for(resolved),
        )
        self.input_name = self._session.get_inputs()[0].name
        self.output_names = [output.name for output in self._session.get_outputs()]
        self.input_hw, self.input_channels = _detect_onnx_input_spec(self._session)
        self.input_format = _detect_onnx_input_format(self._session)
        self.model_min_batch = _detect_onnx_min_batch(self._session)

    def run(self, batch: np.ndarray) -> Any:
        return self._session.run(None, {self.input_name: batch})

    def close(self) -> None:
        self._session = None
