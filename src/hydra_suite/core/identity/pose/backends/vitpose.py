"""ViTPose pose backend. Thin driver over the Spec-1 leaf, mirroring yolo.py.

Native path only in Phase 1; ONNX/TensorRT/CoreML runners are wired in Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..types import PoseResult
from ..utils import summarize_keypoints
from ..vitpose.adapter import load_finetuned_checkpoint
from ..vitpose.config import IMAGE_SIZE_WH
from ..vitpose.infer import decode_and_project, preprocess_crop


def _resolve_device(device: str) -> str:
    if device not in ("auto", ""):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ViTPoseBackend:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        runtime_flavor: str = "native",
        min_valid_conf: float = 0.2,
        keypoint_names: Optional[Sequence[str]] = None,
        batch_size: int = 4,
    ) -> None:
        self._device = _resolve_device(device)
        self._runtime_flavor = runtime_flavor
        self._min_valid_conf = float(min_valid_conf)
        self._batch_size = int(batch_size)
        model, meta = load_finetuned_checkpoint(Path(model_path))
        self._meta = meta
        self._num_keypoints = meta.num_keypoints
        self.output_keypoint_names: List[str] = list(
            keypoint_names or [f"kp{i}" for i in range(meta.num_keypoints)]
        )
        if len(self.output_keypoint_names) != meta.num_keypoints:
            raise ValueError(
                f"keypoint_names has {len(self.output_keypoint_names)} entries "
                f"but checkpoint has {meta.num_keypoints} keypoints"
            )
        self._model = model.to(self._device).eval()

    @property
    def preferred_input_size(self) -> int:
        return IMAGE_SIZE_WH[1]  # 256 (H); the long side

    def warmup(self) -> None:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        try:
            self.predict_batch([dummy])
        except Exception:  # warmup must never raise
            pass

    def _forward_torch(self, batch_chw: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(batch_chw).to(self._device)
        with torch.no_grad():
            return self._model(t)  # (B, K, 64, 48) on device

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []
        results: List[PoseResult] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = crops[start : start + self._batch_size]
            chws, centers, scales = [], [], []
            for crop in chunk:
                chw, c, s = preprocess_crop(np.asarray(crop))
                chws.append(chw)
                centers.append(c)
                scales.append(s)
            batch = np.stack(chws, axis=0).astype(np.float32)
            heatmaps = self._forward_torch(batch)
            coords, maxvals = decode_and_project(
                heatmaps, np.stack(centers), np.stack(scales)
            )
            for i in range(len(chunk)):
                kpts = np.concatenate([coords[i], maxvals[i]], axis=1)  # (K,3)
                results.append(summarize_keypoints(kpts, self._min_valid_conf))
        return results

    def close(self) -> None:
        self._model = None
