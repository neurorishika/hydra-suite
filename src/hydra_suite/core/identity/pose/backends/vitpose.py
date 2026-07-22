"""ViTPose pose backend. Thin driver over the Spec-1 leaf, mirroring yolo.py.

Native path only in Phase 1; ONNX/TensorRT/CoreML runners are wired in Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..artifacts import (
    artifact_meta_matches,
    path_fingerprint_token,
    write_artifact_meta,
)
from ..types import PoseResult
from ..utils import summarize_keypoints
from ..vitpose.adapter import load_finetuned_checkpoint
from ..vitpose.config import IMAGE_SIZE_WH
from ..vitpose.export import build_tensorrt_engine, export_coreml, export_onnx
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
        exported_model_path: str = "",
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

        self._runner = None
        if runtime_flavor == "coreml" and exported_model_path:
            from ..runtime.coreml_runner import CoreMLRunner

            self._runner = CoreMLRunner(Path(exported_model_path))
        elif runtime_flavor == "tensorrt" and exported_model_path:
            from hydra_suite.runtime.resolver import ResolvedBackend

            from ..runtime.accelerated import build_accelerated_runner

            self._runner = build_accelerated_runner(
                Path(exported_model_path), ResolvedBackend("tensorrt", "cuda", False)
            )

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

    def _forward(self, batch_chw: np.ndarray) -> torch.Tensor:
        if self._runner is not None:
            out = self._runner.run(batch_chw.astype(np.float32))
            if isinstance(out, dict):
                arr = next(iter(out.values()))
            elif isinstance(out, (list, tuple)):
                arr = out[0]
            else:
                arr = out
            return torch.from_numpy(np.asarray(arr, dtype=np.float32))
        return self._forward_torch(batch_chw)

    def predict_batch_cuda(self, crops):
        # Convert any device tensors back to uint8 HWC numpy and reuse the
        # correct numpy path. This is the shippable, correct implementation
        # for every runner. Zero-copy TRT is a documented perf follow-up.
        np_crops = [
            (
                (c.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                if hasattr(c, "permute")
                else np.asarray(c)
            )
            for c in crops
        ]
        return self.predict_batch(np_crops)

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
            heatmaps = self._forward(batch)
            coords, maxvals = decode_and_project(
                heatmaps, np.stack(centers), np.stack(scales)
            )
            for i in range(len(chunk)):
                kpts = np.concatenate([coords[i], maxvals[i]], axis=1)  # (K,3)
                results.append(summarize_keypoints(kpts, self._min_valid_conf))
        return results

    def close(self) -> None:
        self._model = None


_VITPOSE_RECIPE_TAG = "vitpose-v1"


def _vitpose_artifact_signature(model_path: str, flavor: str) -> str:
    return f"{_VITPOSE_RECIPE_TAG}|{flavor}|opset17|fp32|{path_fingerprint_token(model_path)}"


def _artifact_path_for(model_path: Path, flavor: str) -> Path:
    if flavor == "tensorrt":
        return model_path.with_suffix(".engine")
    if flavor == "coreml":
        return model_path.with_suffix(".mlpackage")
    raise ValueError(f"no ViTPose artifact for flavor {flavor!r}")


def auto_export_vitpose_model(
    config, runtime_flavor: str, runtime_device: Optional[str] = None
) -> str:
    """Lazily export + cache a ViTPose artifact next to its checkpoint.

    Mirrors auto_export_yolo_model / auto_export_sleap_model: co-located
    artifact, signature-gated .runtime_meta.json sidecar, recipe-version tag.
    """
    model_path = Path(str(config.model_path))
    artifact = _artifact_path_for(model_path, runtime_flavor)
    signature = _vitpose_artifact_signature(str(model_path), runtime_flavor)
    if artifact.exists() and artifact_meta_matches(artifact, signature):
        return str(artifact.resolve())

    model, _meta = load_finetuned_checkpoint(model_path)
    model.eval()
    if runtime_flavor == "coreml":
        export_coreml(model, artifact)
    elif runtime_flavor == "tensorrt":
        onnx_path = model_path.with_suffix(".onnx")
        export_onnx(model, onnx_path)
        build_tensorrt_engine(onnx_path, artifact, fp16=False)
    else:
        raise ValueError(f"auto_export_vitpose_model: bad flavor {runtime_flavor!r}")
    write_artifact_meta(artifact, signature)
    return str(artifact.resolve())
