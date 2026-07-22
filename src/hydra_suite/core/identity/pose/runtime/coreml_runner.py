"""Native CoreML .mlpackage runner for pose backends.

The leaf export pins CoreML input to a static batch of 1 (pos_embed has no
interpolation path), so this runner loops per sample. fp32 throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


class CoreMLRunner:
    def __init__(self, model_path: Path) -> None:
        import coremltools as ct  # lazy

        self._model = ct.models.MLModel(str(model_path))
        spec = self._model.get_spec()
        self._input_name = spec.description.input[0].name
        self._output_name = spec.description.output[0].name

    def run(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        batch = np.asarray(batch, dtype=np.float32)
        outs = []
        for i in range(batch.shape[0]):
            sample = batch[i : i + 1]  # (1,3,256,192)
            pred = self._model.predict({self._input_name: sample})
            outs.append(np.asarray(pred[self._output_name], dtype=np.float32))
        return {self._output_name: np.concatenate(outs, axis=0)}
