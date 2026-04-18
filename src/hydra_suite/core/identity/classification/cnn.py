"""CNN identity backend for MAT: config, predictions, cache, and inference backend.

Pure Python — no Qt dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CNNIdentityConfig:
    """Configuration for CNN Classifier identity method."""

    model_path: str = ""
    confidence: float = 0.5
    label: str = ""
    batch_size: int = 64
    match_bonus: float = 0.5
    mismatch_penalty: float = 1.0
    window: int = 10
    scoring_mode: str = "atomic"


@dataclass(frozen=True)
class ClassPrediction:
    """Single detection's classifier output.

    For flat models ``factor_names`` has length 1 and the ``class_name`` /
    ``confidence`` properties give the scalar view. For multi-head models
    each tuple index is a distinct factor.
    """

    det_index: int
    factor_names: tuple[str, ...]
    class_names: tuple[str | None, ...]
    confidences: tuple[float, ...]

    @property
    def is_unknown(self) -> tuple[bool, ...]:
        return tuple(name == "unknown" for name in self.class_names)

    @property
    def class_name(self) -> str | None:
        if len(self.factor_names) != 1:
            raise ValueError(
                "ClassPrediction.class_name is only defined for flat (K=1) "
                "predictions; use class_names tuple for multi-factor"
            )
        return self.class_names[0]

    @property
    def confidence(self) -> float:
        if len(self.factor_names) != 1:
            raise ValueError(
                "ClassPrediction.confidence is only defined for flat (K=1) "
                "predictions; use confidences tuple for multi-factor"
            )
        return self.confidences[0]


# ---------------------------------------------------------------------------
# CNNIdentityCache
# ---------------------------------------------------------------------------

_SENTINEL_NONE = "__NONE__"  # stored in npz when class_name is None


class CNNIdentityCache:
    """Persistent .npz cache of per-frame CNN identity predictions.

    Data is accumulated in memory via ``save()`` and written to disk in a
    single compressed write via ``flush()``.  Call ``load()`` during the
    tracking loop to retrieve per-frame predictions.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)
        self._data: dict[str, Any] = {}
        if self._path.exists():
            raw = np.load(str(self._path), allow_pickle=True)
            self._data = dict(raw)

    def exists(self) -> bool:
        """Return True if the cache file exists on disk."""
        return self._path.exists()

    def save(self, frame_idx: int, predictions: list[ClassPrediction]) -> None:
        """Update in-memory cache for *frame_idx*. Call flush() when done."""
        if not predictions:
            self._data[f"f{frame_idx}_det"] = np.array([], dtype=np.int32)
            self._data[f"f{frame_idx}_cls"] = np.array([], dtype=object)
            self._data[f"f{frame_idx}_conf"] = np.array([], dtype=np.float32)
        else:
            det_arr = np.array([p.det_index for p in predictions], dtype=np.int32)
            cls_arr = np.array(
                [
                    (
                        p.class_names[0]
                        if p.class_names[0] is not None
                        else _SENTINEL_NONE
                    )
                    for p in predictions
                ],
                dtype=object,
            )
            conf_arr = np.array(
                [p.confidences[0] for p in predictions], dtype=np.float32
            )
            self._data[f"f{frame_idx}_det"] = det_arr
            self._data[f"f{frame_idx}_cls"] = cls_arr
            self._data[f"f{frame_idx}_conf"] = conf_arr

    def flush(self) -> None:
        """Write all in-memory predictions to disk."""
        if not self._data:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self._path), **self._data)

    def load(self, frame_idx: int) -> list[ClassPrediction]:
        """Return saved predictions for *frame_idx*, or [] if not found."""
        key_det = f"f{frame_idx}_det"
        if key_det not in self._data:
            return []
        det_arr = self._data[key_det]
        cls_arr = self._data[f"f{frame_idx}_cls"]
        conf_arr = self._data[f"f{frame_idx}_conf"]
        results = []
        for i in range(len(det_arr)):
            raw_cls = str(cls_arr[i])
            class_name = None if raw_cls == _SENTINEL_NONE else raw_cls
            results.append(
                ClassPrediction(
                    det_index=int(det_arr[i]),
                    factor_names=("flat",),
                    class_names=(class_name,),
                    confidences=(float(conf_arr[i]),),
                )
            )
        return results

    def get_cached_frames(self) -> list[int]:
        """Return sorted frame indices present in the cache."""
        frames = []
        for key in self._data:
            if not key.startswith("f") or not key.endswith("_det"):
                continue
            try:
                frames.append(int(str(key)[1:-4]))
            except ValueError:
                continue
        return sorted(set(frames))


# ---------------------------------------------------------------------------
# CNNIdentityBackend
# ---------------------------------------------------------------------------


class CNNIdentityBackend:
    """Wraps model loading and batch inference for CNN identity classification.

    Supports:
    - .pth checkpoints (TinyClassifier or torchvision — detected via 'arch' field)
    - YOLO .pt checkpoints (ultralytics)
    Runtime selection via compute_runtime (cpu/mps/cuda). ONNX artifacts are
    derived lazily from .pth and cached alongside the source file.
    """

    def __init__(
        self,
        config: CNNIdentityConfig,
        model_path: str | None = None,
        compute_runtime: str = "cpu",
    ) -> None:
        self._config = config
        self._model_path = str(model_path or config.model_path or "")
        self._compute_runtime = str(compute_runtime or "cpu")
        self._model = None
        self._class_names: list[str] = []
        self._input_size: tuple[int, int] = (224, 224)
        self._arch: str = "tinyclassifier"
        self._is_yolo: bool = self._model_path.endswith(".pt")
        self._loaded: bool = False
        self._infer_fn = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        use_onnx = (
            self._compute_runtime.startswith("onnx_")
            or self._compute_runtime == "tensorrt"
        )
        device = self._torch_device(self._compute_runtime)

        if self._is_yolo:
            self._load_yolo(use_onnx, device)
        else:
            self._load_pth(use_onnx, device)
        self._loaded = True

    def _torch_device(self, rt: str) -> str:
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt in ("mps", "onnx_coreml"):
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"
        return "cpu"

    def _load_pth(self, use_onnx: bool, device: str) -> None:
        import torch

        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
        self._class_names = ckpt.get("class_names", [])
        raw_size = ckpt.get("input_size", (224, 224))
        self._input_size = (
            tuple(raw_size)
            if isinstance(raw_size, (list, tuple))
            else (raw_size, raw_size)
        )
        self._arch = ckpt.get("arch", "tinyclassifier")

        if use_onnx:
            onnx_path = self._derive_onnx(ckpt, device)
            import onnxruntime as ort

            from hydra_suite.runtime.compute_runtime import (
                derive_onnx_execution_providers,
            )

            providers = derive_onnx_execution_providers(self._compute_runtime)
            self._model = ort.InferenceSession(onnx_path, providers=providers)
            self._infer_fn = self._infer_onnx
        else:
            if self._arch == "tinyclassifier":
                from hydra_suite.training.tiny_model import load_tiny_classifier

                self._model, _ = load_tiny_classifier(self._model_path, device=device)
            else:
                # Requires Spec A (ClassKit Extended Training) to be implemented first
                from hydra_suite.training.torchvision_model import (
                    load_torchvision_classifier,
                )

                self._model, _ = load_torchvision_classifier(
                    self._model_path, device=device
                )
            self._infer_fn = lambda batch_np, dev=device: self._infer_torch(
                batch_np, dev
            )

    def _load_yolo(self, use_onnx: bool, device: str) -> None:
        from ultralytics import YOLO

        yolo = YOLO(self._model_path)
        names = yolo.names
        self._class_names = [names[i] for i in sorted(names.keys())]
        self._input_size = (224, 224)
        self._arch = "yolo"
        if use_onnx:
            onnx_path = str(Path(self._model_path).with_suffix(".onnx"))
            if not os.path.exists(onnx_path):
                yolo.export(format="onnx", imgsz=224)
            import onnxruntime as ort

            from hydra_suite.runtime.compute_runtime import (
                derive_onnx_execution_providers,
            )

            self._model = ort.InferenceSession(
                onnx_path,
                providers=derive_onnx_execution_providers(self._compute_runtime),
            )
            self._infer_fn = self._infer_onnx
        else:
            self._model = yolo
            self._infer_fn = self._infer_yolo

    def _derive_onnx(self, ckpt: dict, device: str) -> str:
        """Lazy-derive ONNX from .pth. Returns path to .onnx file."""
        onnx_path = str(Path(self._model_path).with_suffix(".onnx"))
        if os.path.exists(onnx_path):
            return onnx_path
        if self._arch == "tinyclassifier":
            import torch

            from hydra_suite.training.tiny_model import load_tiny_classifier

            model, _ = load_tiny_classifier(self._model_path, device="cpu")
            h, w = self._input_size
            dummy = torch.zeros(1, 3, h, w)
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                opset_version=17,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        else:
            from hydra_suite.training.torchvision_model import (
                export_torchvision_to_onnx,
                load_torchvision_classifier,
            )

            model, loaded_ckpt = load_torchvision_classifier(
                self._model_path, device="cpu"
            )
            export_torchvision_to_onnx(model, loaded_ckpt, onnx_path)
        return onnx_path

    def _infer_torch(self, batch_np: np.ndarray, device: str) -> np.ndarray:
        import torch

        t = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            logits = self._model(t).cpu().numpy()
        return logits

    def _infer_onnx(self, batch_np: np.ndarray) -> np.ndarray:
        input_name = self._model.get_inputs()[0].name
        return self._model.run(None, {input_name: batch_np.astype(np.float32)})[0]

    def _infer_yolo(self, crops: list[np.ndarray]) -> np.ndarray:
        # YOLO classify expects list of numpy arrays in HWC uint8 format
        results = self._model(crops, verbose=False)
        probs = np.array([r.probs.data.cpu().numpy() for r in results])
        return np.log(np.clip(probs, 1e-9, 1.0))

    def _preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """Resize and normalize crops to the model's expected input format.

        Uses vectorised OpenCV operations instead of per-crop PIL conversion.
        """
        import cv2

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
        h, w = self._input_size
        n = len(crops)

        # Pre-allocate batch (N, H, W, 3) float32
        batch_hwc = np.empty((n, h, w, 3), dtype=np.float32)

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                batch_hwc[i] = 0.0
                continue
            resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            # BGR→RGB and uint8→float32 in one step
            batch_hwc[i] = resized[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)

        # Vectorised normalise: (N, H, W, 3)
        batch_hwc = (batch_hwc - mean) / std

        # Transpose to (N, 3, H, W) for PyTorch / ONNX
        return batch_hwc.transpose(0, 3, 1, 2).astype(np.float32)

    def predict_batch(self, crops: list[np.ndarray]) -> list[ClassPrediction]:
        """Run inference on *crops*. Returns one ClassPrediction per crop."""
        if not crops:
            return []
        self._ensure_loaded()
        # YOLO native inference does its own preprocessing — pass raw crops
        if self._is_yolo and self._compute_runtime not in (
            "onnx_coreml",
            "onnx_cpu",
            "onnx_cuda",
            "onnx_rocm",
            "tensorrt",
        ):
            logits = self._infer_fn(crops)
        else:
            batch_np = self._preprocess(crops)
            logits = self._infer_fn(batch_np)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        results = []
        for i, prob in enumerate(probs):
            best_idx = int(np.argmax(prob))
            best_conf = float(prob[best_idx])
            if best_conf >= self._config.confidence and self._class_names:
                class_name = self._class_names[best_idx]
            else:
                class_name = None
            results.append(
                ClassPrediction(
                    det_index=i,
                    factor_names=("flat",),
                    class_names=(class_name,),
                    confidences=(best_conf,),
                )
            )
        return results

    def close(self) -> None:
        """Release the loaded model and inference function, marking the backend as unloaded."""
        self._model = None
        self._infer_fn = None
        self._loaded = False


# ---------------------------------------------------------------------------
# TrackCNNHistory
# ---------------------------------------------------------------------------


class TrackCNNHistory:
    """Sliding-window per-track history of multi-factor classifier predictions.

    Per-factor majority vote excludes ``None`` and ``"unknown"`` observations.
    Ties return ``None`` for that factor.
    """

    def __init__(self, *, window: int, factor_names: tuple[str, ...]) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not factor_names:
            raise ValueError("factor_names must be non-empty")
        self._window = int(window)
        self._factor_names = tuple(str(n) for n in factor_names)
        # per-track deque of (class_names_tuple, confidences_tuple)
        from collections import deque

        self._deque_cls = deque
        self._history: dict[
            int, "deque[tuple[tuple[str | None, ...], tuple[float, ...]]]"
        ] = {}

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self._factor_names

    def record(
        self,
        *,
        track_id: int,
        class_names: tuple[str | None, ...],
        confidences: tuple[float, ...],
    ) -> None:
        if len(class_names) != len(self._factor_names):
            raise ValueError(
                f"class_names length {len(class_names)} does not match "
                f"factor_names length {len(self._factor_names)}"
            )
        if len(confidences) != len(self._factor_names):
            raise ValueError(
                f"confidences length {len(confidences)} does not match "
                f"factor_names length {len(self._factor_names)}"
            )
        buf = self._history.get(track_id)
        if buf is None:
            buf = self._deque_cls(maxlen=self._window)
            self._history[track_id] = buf
        buf.append((tuple(class_names), tuple(confidences)))

    def majority_class(self, track_id: int) -> tuple[str | None, ...]:
        buf = self._history.get(track_id)
        if not buf:
            return tuple(None for _ in self._factor_names)
        result: list[str | None] = []
        for k in range(len(self._factor_names)):
            counts: dict[str, int] = {}
            for names_tuple, _confs in buf:
                name = names_tuple[k]
                if name is None or name == "unknown":
                    continue
                counts[name] = counts.get(name, 0) + 1
            if not counts:
                result.append(None)
                continue
            max_count = max(counts.values())
            winners = [name for name, n in counts.items() if n == max_count]
            result.append(winners[0] if len(winners) == 1 else None)
        return tuple(result)

    def clear_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def build_track_identity_list(self) -> dict[int, tuple[str | None, ...]]:
        return {tid: self.majority_class(tid) for tid in self._history}


# ---------------------------------------------------------------------------
# Hungarian cost helper
# ---------------------------------------------------------------------------


def apply_cnn_identity_cost(
    cost: float,
    det_class: str | None,
    track_identity: str | None,
    match_bonus: float,
    mismatch_penalty: float,
) -> float:
    """Apply CNN identity match bonus / mismatch penalty to a cost value.

    Returns cost unchanged when either side is uncertain (None).
    """
    if det_class is None or track_identity is None:
        return cost
    if det_class == track_identity:
        return cost - match_bonus
    return cost + mismatch_penalty


def cost_atomic(
    track: tuple[str | None, ...],
    det: tuple[str | None, ...],
    *,
    match_bonus: float,
    mismatch_penalty: float,
) -> float:
    """Atomic tuple compare: any ``None`` or ``"unknown"`` in either side -> no signal."""
    for x in (*track, *det):
        if x is None or x == "unknown":
            return 0.0
    return -float(match_bonus) if track == det else +float(mismatch_penalty)


def cost_per_head_average(
    track: tuple[str | None, ...],
    det: tuple[str | None, ...],
    *,
    match_bonus: float,
    mismatch_penalty: float,
    K: int,
) -> float:
    """Per-head average cost. Divisor is always K (not the number of comparable heads)."""
    if K <= 0:
        return 0.0
    contributions = 0.0
    for k in range(K):
        tk = track[k] if k < len(track) else None
        dk = det[k] if k < len(det) else None
        if tk is None or tk == "unknown":
            continue
        if dk is None or dk == "unknown":
            continue
        contributions += -float(match_bonus) if tk == dk else +float(mismatch_penalty)
    return contributions / float(K)
