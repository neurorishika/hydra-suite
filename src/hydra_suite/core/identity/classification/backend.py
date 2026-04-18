"""Shared classifier backend: checkpoint parsing, arch dispatch, runtime selection,
ONNX lazy-derivation, preprocessing, and batched forward pass.

Phase 1 supports tiny flat and YOLO flat artifacts. Torchvision, multi-head,
and ONNX lazy-derivation land in later phases.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hydra_suite.core.identity.classification.errors import (
    ClassifierConfigError,
    ClassifierError,
    ClassifierFormatError,
    ClassifierRuntimeError,
)

__all__ = ["ClassifierMetadata", "ClassifierBackend"]

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ClassifierMetadata:
    """Declarative description of a classifier artifact.

    All fields are populated eagerly from the checkpoint or manifest without
    loading weights, so import dialogs can preview the factor structure
    cheaply.

    Attributes:
        arch: Backbone identifier. One of ``"tinyclassifier"``, a torchvision
            backbone name (``"resnet50"``, ``"convnext_tiny"``, ...), ``"yolo"``
            (flat single ultralytics model), or ``"yolo_multihead"`` (bundle).
        input_size: Canonical ``(H, W)`` shape expected by the model.
        is_multihead: True when ``factor_names`` has length > 1.
        factor_names: Per-factor names. ``["flat"]`` when flat.
        class_names_per_factor: Per-factor class lists; length matches
            ``factor_names``.
        monochrome: Whether the model was trained with monochrome augmentation,
            affecting preprocessing normalization.
        source_path: Absolute path the artifact was loaded from (for display
            and ONNX-derivation peer location).
    """

    arch: str
    input_size: tuple[int, int]
    is_multihead: bool
    factor_names: list[str]
    class_names_per_factor: list[list[str]]
    monochrome: bool
    source_path: str


def _normalize_input_size(raw: Any) -> tuple[int, int]:
    """Canonicalize on-disk ``input_size`` to an in-memory ``(H, W)`` tuple.

    Accepts ``(H, W)`` tuples or lists from v2 checkpoints. Raises on malformed
    values — legacy ``[W, H]`` is rejected because it cannot be reliably
    distinguished from ``[H, W]`` for non-square inputs.
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            h, w = int(raw[0]), int(raw[1])
        except (TypeError, ValueError) as exc:
            raise ClassifierFormatError(f"invalid input_size {raw!r}") from exc
        if h <= 0 or w <= 0:
            raise ClassifierFormatError(f"invalid input_size {raw!r}")
        return (h, w)
    if isinstance(raw, int) and raw > 0:
        return (raw, raw)
    raise ClassifierFormatError(f"invalid input_size {raw!r}")


def _require_v2(ckpt: dict[str, Any], source_path: str) -> None:
    """Enforce schema_version == 2 on classifier artifacts."""
    sv = ckpt.get("schema_version")
    if sv != 2:
        raise ClassifierFormatError(
            f"classifier artifact {source_path!r} has schema_version={sv!r}; "
            "please re-export from ClassKit (expected schema_version=2)"
        )


def _parse_factor_structure(
    ckpt: dict[str, Any], source_path: str
) -> tuple[list[str], list[list[str]], bool]:
    """Extract factor_names and class_names_per_factor from a v2 checkpoint.

    Enforces the v2 rule that multi-head checkpoints use only
    ``class_names_per_factor`` (no flat ``class_names``).
    """
    factor_names = ckpt.get("factor_names")
    cnpf_raw = ckpt.get("class_names_per_factor")
    if not isinstance(factor_names, list) or not factor_names:
        raise ClassifierFormatError(f"{source_path!r}: missing or empty factor_names")
    if not isinstance(cnpf_raw, list) or not cnpf_raw:
        raise ClassifierFormatError(
            f"{source_path!r}: missing or empty class_names_per_factor"
        )
    if len(factor_names) != len(cnpf_raw):
        raise ClassifierFormatError(
            f"{source_path!r}: factor_names (len {len(factor_names)}) "
            f"does not match class_names_per_factor (len {len(cnpf_raw)})"
        )
    if len(set(factor_names)) != len(factor_names):
        raise ClassifierFormatError(f"{source_path!r}: factor_names must be unique")
    class_names_per_factor = [[str(n) for n in inner] for inner in cnpf_raw]
    is_multihead = len(factor_names) > 1
    if is_multihead and "class_names" in ckpt:
        raise ClassifierFormatError(
            f"{source_path!r}: multi-head checkpoint must not include a "
            "top-level class_names field"
        )
    return [str(n) for n in factor_names], class_names_per_factor, is_multihead


class _TinyLoader:
    """Loader for TinyClassifier v2 .pth checkpoints (flat and multi-head)."""

    @staticmethod
    def parse_metadata(path: str) -> ClassifierMetadata:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ClassifierFormatError(f"{path!r}: expected checkpoint dict")
        _require_v2(ckpt, path)
        factor_names, cnpf, is_multi = _parse_factor_structure(ckpt, path)
        return ClassifierMetadata(
            arch=str(ckpt.get("arch", "tinyclassifier")),
            input_size=_normalize_input_size(ckpt.get("input_size")),
            is_multihead=is_multi,
            factor_names=factor_names,
            class_names_per_factor=cnpf,
            monochrome=bool(ckpt.get("monochrome", False)),
            source_path=path,
        )

    @staticmethod
    def load(path: str, device: str):
        from hydra_suite.training.tiny_model import load_tiny_classifier

        model, _ = load_tiny_classifier(path, device=device)
        return model


class _YoloFlatLoader:
    """Loader for flat ultralytics YOLO-classify .pt checkpoints."""

    @staticmethod
    def parse_metadata(path: str) -> ClassifierMetadata:
        from ultralytics import YOLO

        yolo = YOLO(path)
        names = getattr(yolo, "names", None) or {}
        class_names = [str(names[i]) for i in sorted(names.keys())]
        return ClassifierMetadata(
            arch="yolo",
            input_size=(224, 224),
            is_multihead=False,
            factor_names=["flat"],
            class_names_per_factor=[class_names],
            monochrome=False,
            source_path=path,
        )

    @staticmethod
    def load(path: str, device: str):
        from ultralytics import YOLO

        return YOLO(path)


class _TorchvisionLoader:
    """Loader for torchvision v2 .pth checkpoints (flat and multi-head)."""

    @staticmethod
    def parse_metadata(path: str) -> ClassifierMetadata:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ClassifierFormatError(f"{path!r}: expected checkpoint dict")
        _require_v2(ckpt, path)
        factor_names, cnpf, is_multi = _parse_factor_structure(ckpt, path)
        return ClassifierMetadata(
            arch=str(ckpt["arch"]),
            input_size=_normalize_input_size(ckpt.get("input_size")),
            is_multihead=is_multi,
            factor_names=factor_names,
            class_names_per_factor=cnpf,
            monochrome=bool(ckpt.get("monochrome", False)),
            source_path=path,
        )

    @staticmethod
    def load(path: str, device: str):
        from hydra_suite.training.torchvision_model import load_torchvision_classifier

        model, _ = load_torchvision_classifier(path, device=device)
        return model


def _peek_ckpt_arch(path: str) -> str:
    """Read the ``arch`` field from a .pth checkpoint without instantiating weights."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "arch" not in ckpt:
        raise ClassifierFormatError(
            f"{path!r}: cannot determine arch; expected v2 checkpoint dict"
        )
    return str(ckpt["arch"])


def _select_loader(path: str):
    """Pick a loader from the artifact's suffix and, for .pth, the arch field."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pth":
        arch = _peek_ckpt_arch(path)
        if arch == "tinyclassifier":
            return _TinyLoader
        return _TorchvisionLoader
    if suffix == ".pt":
        return _YoloFlatLoader
    raise ClassifierFormatError(
        f"{path!r}: unsupported classifier artifact suffix {suffix!r}"
    )


def _torch_device(compute_runtime: str) -> str:
    rt = compute_runtime
    if rt in ("cuda", "onnx_cuda", "tensorrt"):
        return "cuda"
    if rt in ("mps", "onnx_coreml"):
        return "mps"
    if rt in ("rocm", "onnx_rocm"):
        return "cuda"
    return "cpu"


class ClassifierBackend:
    """Wraps model loading, preprocessing, and inference for a classifier
    artifact. Consumers apply their own semantics on the returned per-factor
    probabilities.
    """

    def __init__(self, model_path: str, compute_runtime: str = "cpu") -> None:
        if not model_path:
            raise ClassifierConfigError("model_path must be non-empty")
        path = str(model_path)
        if not os.path.exists(path):
            raise ClassifierFormatError(f"{path!r}: file does not exist")
        self._model_path = path
        self._compute_runtime = str(compute_runtime or "cpu")
        self._loader = _select_loader(path)
        self._metadata = self._loader.parse_metadata(path)
        self._model = None
        self._loaded = False

    @property
    def metadata(self) -> ClassifierMetadata:
        return self._metadata

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        device = _torch_device(self._compute_runtime)
        try:
            self._model = self._loader.load(self._model_path, device)
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierRuntimeError(
                f"failed to load classifier {self._model_path!r}: {exc}"
            ) from exc
        self._loaded = True

    def _normalization_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self._metadata.monochrome:
            mean = np.full(3, float(_IMAGENET_MEAN.mean()), dtype=np.float32)
            std = np.full(3, float(_IMAGENET_STD.mean()), dtype=np.float32)
            return mean.reshape(1, 1, 1, 3), std.reshape(1, 1, 1, 3)
        return (
            _IMAGENET_MEAN.reshape(1, 1, 1, 3),
            _IMAGENET_STD.reshape(1, 1, 1, 3),
        )

    def _preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """Resize, color-convert, and normalize crops into a (N, 3, H, W) batch."""
        import cv2

        h, w = self._metadata.input_size
        n = len(crops)
        batch = np.empty((n, h, w, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                batch[i] = 0.0
                continue
            resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            batch[i] = resized[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        mean, std = self._normalization_stats()
        batch = (batch - mean) / std
        return batch.transpose(0, 3, 1, 2).astype(np.float32)

    def _forward_torch(self, batch_np: np.ndarray) -> np.ndarray:
        import torch

        device = _torch_device(self._compute_runtime)
        t = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            logits = self._model(t).detach().cpu().numpy()
        return logits

    def _forward_yolo(self, crops: list[np.ndarray]) -> np.ndarray:
        results = self._model(crops, verbose=False)
        probs = np.array([r.probs.data.cpu().numpy() for r in results])
        return np.log(np.clip(probs, 1e-9, 1.0))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _cardinalities(self) -> list[int]:
        return [len(names) for names in self._metadata.class_names_per_factor]

    def predict_batch(self, crops: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Run inference on ``crops``. Returns ``[N_crops][K_factors]`` probability vectors."""
        if not crops:
            return []
        self._ensure_loaded()
        try:
            if self._metadata.arch == "yolo":
                logits = self._forward_yolo(crops)
            else:
                batch_np = self._preprocess(crops)
                logits = self._forward_torch(batch_np)
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierRuntimeError(
                f"inference failure for {self._model_path!r}: {exc}"
            ) from exc

        # Split concatenated logits into per-factor slices, then softmax each.
        cardinalities = self._cardinalities()
        expected_total = sum(cardinalities)
        if logits.shape[-1] != expected_total:
            raise ClassifierRuntimeError(
                f"{self._model_path!r}: model output width {logits.shape[-1]} "
                f"does not match expected total {expected_total} across factors "
                f"{self._metadata.factor_names}"
            )
        results: list[list[np.ndarray]] = []
        for row in logits:
            per_factor: list[np.ndarray] = []
            offset = 0
            for width in cardinalities:
                factor_logits = row[offset : offset + width]
                per_factor.append(self._softmax(factor_logits))
                offset += width
            results.append(per_factor)
        return results

    def close(self) -> None:
        """Release model weights; subsequent predict_batch will reload."""
        self._model = None
        self._loaded = False
