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
_GENERIC_MULTIHEAD_BUNDLE_KIND = "classifier_multihead_bundle"


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
    def _load_sidecar(path: str) -> dict[str, Any] | None:
        sidecar_path = Path(path).with_suffix(".v2meta.json")
        if not sidecar_path.exists():
            return None
        try:
            data = __import__("json").loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ClassifierFormatError(
                f"{path!r}: cannot read YOLO sidecar {sidecar_path.name}: {exc}"
            ) from exc
        if not isinstance(data, dict) or data.get("schema_version") != 2:
            raise ClassifierFormatError(
                f"{path!r}: YOLO sidecar must be a schema_version=2 dict"
            )
        return data

    @staticmethod
    def parse_metadata(path: str) -> ClassifierMetadata:
        sidecar = _YoloFlatLoader._load_sidecar(path) or {}
        factor_names = sidecar.get("factor_names") or ["flat"]
        class_names_per_factor = sidecar.get("class_names_per_factor")
        if not class_names_per_factor:
            from ultralytics import YOLO

            yolo = YOLO(path)
            names = getattr(yolo, "names", None) or {}
            class_names = [str(names[i]) for i in sorted(names.keys())]
            class_names_per_factor = [class_names]
        if len(factor_names) != 1 or len(class_names_per_factor) != 1:
            raise ClassifierFormatError(
                f"{path!r}: flat YOLO sidecar must describe exactly one factor; "
                f"use a .multihead.json bundle for multi-factor exports"
            )
        return ClassifierMetadata(
            arch=str(sidecar.get("arch") or "yolo"),
            input_size=_normalize_input_size(sidecar.get("input_size", [224, 224])),
            is_multihead=False,
            factor_names=[str(factor_names[0])],
            class_names_per_factor=[[str(name) for name in class_names_per_factor[0]]],
            monochrome=bool(sidecar.get("monochrome", False)),
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


class _ClassifierMultiheadBundleLoader:
    """Loader for a .multihead.json manifest describing N flat factor models."""

    @staticmethod
    def parse_metadata(path: str) -> "ClassifierMetadata":
        import json

        manifest_path = Path(path)
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ClassifierFormatError(
                f"{path!r}: cannot read multi-head manifest: {exc}"
            ) from exc
        if data.get("schema_version") != 2:
            raise ClassifierFormatError(f"{path!r}: manifest schema_version must be 2")
        manifest_kind = str(data.get("kind") or "")
        if manifest_kind not in (
            "yolo_multihead_bundle",
            _GENERIC_MULTIHEAD_BUNDLE_KIND,
        ):
            raise ClassifierFormatError(
                f"{path!r}: unexpected manifest kind {data.get('kind')!r}"
            )
        factor_names = data.get("factor_names")
        factor_models = data.get("factor_models")
        if not isinstance(factor_names, list) or not factor_names:
            raise ClassifierFormatError(f"{path!r}: missing factor_names")
        if not isinstance(factor_models, list) or len(factor_models) != len(
            factor_names
        ):
            raise ClassifierFormatError(
                f"{path!r}: factor_models length does not match factor_names"
            )
        class_names_per_factor: list[list[str]] = []
        for expected_name, entry in zip(factor_names, factor_models):
            if not isinstance(entry, dict):
                raise ClassifierFormatError(f"{path!r}: factor entry must be dict")
            if entry.get("factor") != expected_name:
                raise ClassifierFormatError(
                    f"{path!r}: factor order mismatch "
                    f"(expected {expected_name!r}, got {entry.get('factor')!r})"
                )
            names = entry.get("class_names")
            if not isinstance(names, list) or not names:
                raise ClassifierFormatError(
                    f"{path!r}: factor {expected_name!r} missing class_names"
                )
            class_names_per_factor.append([str(n) for n in names])

        return ClassifierMetadata(
            arch=(
                "yolo_multihead"
                if manifest_kind == "yolo_multihead_bundle"
                else "classifier_multihead"
            ),
            input_size=_normalize_input_size(data.get("input_size")),
            is_multihead=True,
            factor_names=[str(n) for n in factor_names],
            class_names_per_factor=class_names_per_factor,
            monochrome=bool(data.get("monochrome", False)),
            source_path=path,
        )

    @staticmethod
    def load(path: str, device: str):
        """Load each referenced factor model as a flat ClassifierBackend."""
        import json

        from hydra_suite.core.identity.classification.errors import (
            ClassifierFormatError,
        )

        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent
        models = []
        for entry in data["factor_models"]:
            factor_path = (base / entry["path"]).resolve()
            factor_backend = ClassifierBackend(str(factor_path), compute_runtime=device)
            if factor_backend.metadata.is_multihead:
                factor_backend.close()
                raise ClassifierFormatError(
                    f"{path!r}: factor model {factor_path.name!r} must be flat, "
                    "not multi-head"
                )
            models.append(factor_backend)
        return models


def _peek_ckpt_arch(path: str) -> str:
    """Read the ``arch`` field from a .pth checkpoint without instantiating weights."""
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ClassifierFormatError(
            f"{path!r}: cannot read classifier checkpoint"
        ) from exc
    if not isinstance(ckpt, dict) or "arch" not in ckpt:
        raise ClassifierFormatError(
            f"{path!r}: cannot determine arch; expected v2 checkpoint dict"
        )
    return str(ckpt["arch"])


def _select_loader(path: str):
    """Pick a loader from the artifact's suffix and, for .pth, the arch field."""
    p = Path(path)
    name_lower = p.name.lower()
    if name_lower.endswith(".multihead.json"):
        return _ClassifierMultiheadBundleLoader
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

    def _uses_onnx(self) -> bool:
        rt = self._compute_runtime
        return rt.startswith("onnx_") or rt == "tensorrt"

    def _uses_factor_backends(self) -> bool:
        return self._metadata.arch in ("yolo_multihead", "classifier_multihead")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            if self._uses_onnx() and not self._uses_factor_backends():
                self._load_onnx()
            else:
                loader_target = (
                    self._compute_runtime
                    if self._uses_factor_backends()
                    else _torch_device(self._compute_runtime)
                )
                self._model = self._loader.load(self._model_path, loader_target)
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierRuntimeError(
                f"failed to load classifier {self._model_path!r}: {exc}"
            ) from exc
        self._loaded = True

    def _derive_onnx_peer(self) -> Path:
        """Return (creating if necessary) the ONNX peer path for this artifact."""
        src = Path(self._model_path)
        peer = src.with_suffix(".onnx")
        if peer.exists():
            return peer
        h, w = self._metadata.input_size
        if self._metadata.arch == "yolo":
            from ultralytics import YOLO

            YOLO(str(src)).export(format="onnx", imgsz=max(h, w))
            return peer
        if self._metadata.arch == "tinyclassifier":
            from hydra_suite.training.tiny_model import (
                export_tiny_to_onnx,
                load_tiny_classifier,
            )

            model, _ = load_tiny_classifier(str(src), device="cpu")
            export_tiny_to_onnx(model, {"input_size": [h, w]}, str(peer))
            return peer
        # torchvision arch
        from hydra_suite.training.torchvision_model import (
            export_torchvision_to_onnx,
            load_torchvision_classifier,
        )

        model, ckpt = load_torchvision_classifier(str(src), device="cpu")
        export_torchvision_to_onnx(model, ckpt, str(peer))
        return peer

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        from hydra_suite.runtime.compute_runtime import derive_onnx_execution_providers

        peer = self._derive_onnx_peer()
        providers = derive_onnx_execution_providers(self._compute_runtime)
        self._model = ort.InferenceSession(str(peer), providers=providers)

    def _uses_imagenet_normalization(self) -> bool:
        """Return True when the checkpoint expects ImageNet mean/std normalization."""
        return self._metadata.arch != "tinyclassifier"

    def _normalization_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._uses_imagenet_normalization():
            mean = np.zeros(3, dtype=np.float32)
            std = np.ones(3, dtype=np.float32)
            return mean.reshape(1, 1, 1, 3), std.reshape(1, 1, 1, 3)
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
            rgb = resized[:, :, ::-1]
            if self._metadata.monochrome:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            batch[i] = rgb.astype(np.float32) * (1.0 / 255.0)
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

    def _forward_yolo_multi(self, crops: list[np.ndarray]) -> np.ndarray:
        """Run each flat factor backend and return per-factor stacked log-probs.

        Returns an ``(N, sum(C_k))`` array shaped like a flat multi-head logit
        tensor so the existing ``predict_batch`` split logic applies.
        """
        per_factor_logits: list[np.ndarray] = []
        for factor_backend in self._model:
            factor_probs = factor_backend.predict_batch(crops)
            probs = np.array(
                [per_crop[0] for per_crop in factor_probs], dtype=np.float32
            )
            per_factor_logits.append(np.log(np.clip(probs, 1e-9, 1.0)))
        return np.concatenate(per_factor_logits, axis=-1)

    def _forward_onnx(self, batch_np: np.ndarray) -> np.ndarray:
        input_name = self._model.get_inputs()[0].name
        return self._model.run(None, {input_name: batch_np.astype(np.float32)})[0]

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
            if self._uses_onnx() and not self._uses_factor_backends():
                batch_np = self._preprocess(crops)
                logits = self._forward_onnx(batch_np)
            elif self._metadata.arch == "yolo":
                logits = self._forward_yolo(crops)
            elif self._uses_factor_backends():
                logits = self._forward_yolo_multi(crops)
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
        if isinstance(self._model, list):
            for factor_backend in self._model:
                close = getattr(factor_backend, "close", None)
                if callable(close):
                    close()
        self._model = None
        self._loaded = False
