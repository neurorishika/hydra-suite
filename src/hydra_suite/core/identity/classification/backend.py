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
        recommended_confidence_threshold: Optional abstention threshold carried
            with the artifact. Consumers may use this as their default
            confidence cutoff for collapsing uncertain predictions to unknown.
        source_path: Absolute path the artifact was loaded from (for display
            and ONNX-derivation peer location).
    """

    arch: str
    input_size: tuple[int, int]
    is_multihead: bool
    factor_names: list[str]
    class_names_per_factor: list[list[str]]
    monochrome: bool
    recommended_confidence_threshold: float | None
    source_path: str


def _normalize_recommended_confidence_threshold(raw: Any) -> float | None:
    """Canonicalize an artifact-level abstention threshold."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, value))


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


def _dedupe_factor_names(factor_names: list[str]) -> list[str]:
    """Return a stable list of unique factor names.

    Multihead bundle manifests in the wild may still contain duplicate factor
    names such as ["flat", "flat"]. Suffix duplicates deterministically so the
    runtime can preserve each head instead of collapsing them into one column.
    """
    seen: dict[str, int] = {}
    unique: list[str] = []
    for idx, raw_name in enumerate(factor_names):
        base = str(raw_name).strip() or f"factor_{idx}"
        count = seen.get(base, 0)
        unique_name = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        unique.append(unique_name)
    return unique


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
            recommended_confidence_threshold=_normalize_recommended_confidence_threshold(
                ckpt.get("recommended_confidence_threshold")
            ),
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
            recommended_confidence_threshold=_normalize_recommended_confidence_threshold(
                sidecar.get("recommended_confidence_threshold")
            ),
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
            recommended_confidence_threshold=_normalize_recommended_confidence_threshold(
                ckpt.get("recommended_confidence_threshold")
            ),
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

        normalized_factor_names = _dedupe_factor_names(
            [str(name) for name in factor_names]
        )
        if normalized_factor_names != [str(name) for name in factor_names]:
            logger.warning(
                "Multihead manifest %s contains duplicate factor names %s; using %s instead.",
                path,
                factor_names,
                normalized_factor_names,
            )

        return ClassifierMetadata(
            arch=(
                "yolo_multihead"
                if manifest_kind == "yolo_multihead_bundle"
                else "classifier_multihead"
            ),
            input_size=_normalize_input_size(data.get("input_size")),
            is_multihead=True,
            factor_names=normalized_factor_names,
            class_names_per_factor=class_names_per_factor,
            monochrome=bool(data.get("monochrome", False)),
            recommended_confidence_threshold=_normalize_recommended_confidence_threshold(
                data.get("recommended_confidence_threshold")
            ),
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
        return "cuda"  # kept for legacy configs; ROCm is no longer supported
    return "cpu"


def _provider_key(provider: object) -> str:
    if isinstance(provider, tuple) and provider:
        return str(provider[0])
    return str(provider)


def _requested_onnx_accelerator_providers(compute_runtime: str) -> list[str]:
    from hydra_suite.runtime.compute_runtime import derive_onnx_execution_providers

    providers = derive_onnx_execution_providers(
        compute_runtime,
        include_cpu_fallback=False,
    )
    return [_provider_key(provider) for provider in providers]


def _available_onnx_provider_names() -> set[str]:
    try:
        import onnxruntime as ort

        return {str(name) for name in (ort.get_available_providers() or [])}
    except Exception:
        return set()


def _looks_like_cuda_alloc_failure(exc: BaseException) -> bool:
    """Return True when *exc* looks like a CUDA/cuBLAS memory-allocation error.

    ORT raises a plain ``Exception`` whose message contains the CUDA error
    string; there is no ORT-specific exception subclass to reliably catch.
    """
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cublas_status_alloc_failed",
            "cuda_error_out_of_memory",
            "alloc_failed",
            "cublascreate",
        )
    )


def _native_accelerator_available(compute_runtime: str) -> bool:
    device = _torch_device(compute_runtime)
    if device == "cuda":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False
    if device == "mps":
        try:
            import torch

            return bool(torch.backends.mps.is_available())
        except Exception:
            return False
    return False


class ClassifierBackend:
    """Wraps model loading, preprocessing, and inference for a classifier
    artifact. Consumers apply their own semantics on the returned per-factor
    probabilities.
    """

    # Maximum batch size encoded into the TRT optimization profile at engine-
    # compile time.  Inference requests that exceed this limit will be rejected
    # by the TRT execution context.  Keep this constant in sync with the
    # ``max_batch`` value used in :meth:`_build_trt_providers_with_profiles`.
    _TRT_PROFILE_MAX_BATCH: int = 512

    def __init__(
        self,
        model_path: str,
        compute_runtime: str = "cpu",
        trt_profile_max_batch: int | None = None,
    ) -> None:
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
        self._active_execution_backend = "unloaded"
        self._trt_profile_max_batch = (
            None
            if trt_profile_max_batch is None
            else max(1, int(trt_profile_max_batch))
        )

    @property
    def metadata(self) -> ClassifierMetadata:
        return self._metadata

    def _uses_onnx(self) -> bool:
        rt = self._compute_runtime
        return rt.startswith("onnx_") or rt == "tensorrt"

    def _uses_coreml(self) -> bool:
        return self._compute_runtime == "coreml"

    def _uses_factor_backends(self) -> bool:
        return self._metadata.arch in ("yolo_multihead", "classifier_multihead")

    def _should_fallback_to_native_runtime(self) -> bool:
        if not self._uses_onnx() or self._uses_factor_backends():
            return False
        if Path(self._model_path).suffix.lower() != ".pth":
            return False

        requested = _requested_onnx_accelerator_providers(self._compute_runtime)
        if not requested:
            return False

        available = _available_onnx_provider_names()
        if any(provider in available for provider in requested):
            return False

        return _native_accelerator_available(self._compute_runtime)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            if self._uses_coreml() and not self._uses_factor_backends():
                self._load_coreml()
            elif self._uses_onnx() and not self._uses_factor_backends():
                if self._should_fallback_to_native_runtime():
                    native_device = _torch_device(self._compute_runtime)
                    logger.warning(
                        "ClassifierBackend: requested ONNX runtime %s for %s but matching ONNX providers are unavailable; falling back to native %s execution",
                        self._compute_runtime,
                        self._model_path,
                        native_device,
                    )
                    self._model = self._loader.load(self._model_path, native_device)
                    self._active_execution_backend = "native"
                    if native_device.startswith("cuda"):
                        self._warmup_native_cuda_model()
                else:
                    self._load_onnx()
                    self._active_execution_backend = "onnx"
            else:
                loader_target = (
                    self._compute_runtime
                    if self._uses_factor_backends()
                    else _torch_device(self._compute_runtime)
                )
                self._model = self._loader.load(self._model_path, loader_target)
                self._active_execution_backend = "native"
                # Warm up native CUDA models immediately to trigger PyTorch/cuDNN
                # kernel JIT compilation now rather than stalling Phase-1 batch 1
                # for ~45 s on Ada/Hopper GPUs (e.g. EfficientNet-B0 head-tail).
                if not self._uses_factor_backends():
                    device_str = _torch_device(self._compute_runtime)
                    if device_str.startswith("cuda"):
                        self._warmup_native_cuda_model()
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierRuntimeError(
                f"failed to load classifier {self._model_path!r}: {exc}"
            ) from exc
        self._loaded = True

    def _warmup_native_cuda_model(self) -> None:
        """Run a single dummy forward pass to trigger PyTorch/cuDNN kernel JIT.

        On Ada Lovelace and Hopper GPUs the first forward pass through a native
        PyTorch model on CUDA triggers cuDNN algorithm selection and CUDA kernel
        compilation, stalling for ~40-50 s for models like EfficientNet-B0.
        Running this method during ``_ensure_loaded()`` moves that one-time cost
        to model setup time (before Phase 1 begins) rather than stalling batch 1.
        """
        if self._model is None:
            return
        try:
            import torch

            h, w = self._metadata.input_size
            device_str = _torch_device(self._compute_runtime)
            dummy = torch.zeros(1, 3, h, w, dtype=torch.float32, device=device_str)
            with torch.no_grad():
                self._model(dummy)
            torch.cuda.synchronize()
        except Exception:
            pass

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

        is_trt = any(
            "tensorrt" in (p if isinstance(p, str) else p[0]).lower() for p in providers
        )
        if is_trt:
            # Inject TRT optimization profiles so the engine covers batch sizes
            # 1..512 in a single compilation.  Without profiles ORT/TRT compiles
            # a separate engine for each newly observed batch size, causing a
            # ~44 s stall the first time a large crop batch hits batch 1.
            # Engine caching means only the very first run ever pays this cost.
            try:
                providers = self._build_trt_providers_with_profiles(peer, providers)
            except Exception:
                pass  # silently fall back to providers without profile options

        is_gpu_requested = any(
            any(
                kw in (p if isinstance(p, str) else p[0]).lower()
                for kw in ("cuda", "tensorrt")
            )
            for p in providers
        )
        try:
            self._model = ort.InferenceSession(str(peer), providers=providers)
        except Exception as gpu_exc:
            if is_gpu_requested and _looks_like_cuda_alloc_failure(gpu_exc):
                logger.warning(
                    "ClassifierBackend: CUDA/cuBLAS initialization failed for %s (%s); "
                    "falling back to CPU execution",
                    self._model_path,
                    gpu_exc,
                )
                self._model = ort.InferenceSession(
                    str(peer), providers=["CPUExecutionProvider"]
                )
                is_gpu_requested = False
            else:
                raise
        # Warm up the ORT session to trigger CUDA/TRT kernel JIT now (during
        # model loading) rather than stalling on the first real inference batch.
        # Two dummy runs cover both JIT passes on Ada/Hopper GPUs.
        # This applies to both TRT EP and CUDA EP sessions.
        is_cuda = is_gpu_requested and any(
            "cuda" in (p if isinstance(p, str) else p[0]).lower() for p in providers
        )
        if is_trt or is_cuda:
            self._warmup_ort_session_trt()

    def _build_trt_providers_with_profiles(
        self,
        peer_path: "Path",
        providers: list,
    ) -> list:
        """Replace the bare TRT EP string with a (name, options) tuple.

        Sets TRT optimization profiles covering batch sizes 1..max_batch
        (where max_batch is user-configurable but capped at 512) so TRT builds
        a single compiled engine for the target crop-batch range, and enables
        disk-based engine caching so subsequent process restarts avoid
        recompilation entirely.
        """
        import onnxruntime as ort

        from hydra_suite.paths import get_data_dir

        # Probe input metadata with a cheap CPU-only session — no GPU work.
        probe = ort.InferenceSession(str(peer_path), providers=["CPUExecutionProvider"])
        inp = probe.get_inputs()[0]
        inp_name = inp.name
        h, w = self._metadata.input_size
        # Always 3 channels in the ONNX exports produced by this codebase.
        c = 3
        max_batch = min(
            ClassifierBackend._TRT_PROFILE_MAX_BATCH,
            max(
                1,
                int(
                    self._trt_profile_max_batch
                    or ClassifierBackend._TRT_PROFILE_MAX_BATCH
                ),
            ),
        )

        cache_dir = get_data_dir() / "trt_engine_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ORT >= 1.14 uses plural key names (trt_profile_*_shapes) with
        # colon-separated format "input_name:DxCxHxW".
        opt_batch = min(max_batch, 64)
        trt_opts = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_profile_min_shapes": f"{inp_name}:1x{c}x{h}x{w}",
            "trt_profile_opt_shapes": f"{inp_name}:{opt_batch}x{c}x{h}x{w}",
            "trt_profile_max_shapes": f"{inp_name}:{max_batch}x{c}x{h}x{w}",
            "trt_max_workspace_size": 1 << 30,  # 1 GiB
            # FP16 inference: ~2x throughput on Ampere/Ada/Hopper with no label
            # accuracy loss for classification (argmax is preserved).
            "trt_fp16_enable": True,
            # Timing cache: TRT reuses kernel-selection decisions from previous
            # builds so re-exports and profile changes cost far less time.
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(cache_dir),
            # Builder optimisation level 4 maximises throughput; 3 is the ORT
            # default.  Only paid at first-build time, then cached.
            "trt_builder_optimization_level": 4,
        }

        updated: list = []
        for p in providers:
            p_name = p if isinstance(p, str) else p[0]
            if "tensorrt" in p_name.lower():
                updated.append((p_name, trt_opts))
            else:
                updated.append(p)
        return updated

    def _warmup_ort_session_trt(self) -> None:
        """Run two dummy inferences to finish Ada/Hopper TRT JIT compilation.

        The TensorRT "compiler backend" on Ada Lovelace and Hopper GPUs defers
        CUDA kernel JIT to the first N ``session.run()`` calls (typically two).
        Calling this method during session creation avoids ~43-second stalls at
        the start of Phase-1 batch inference.
        """
        if self._model is None:
            return
        try:
            import numpy as np

            inputs = self._model.get_inputs()
            if not inputs:
                return
            inp = inputs[0]
            dtype_map = {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(int64)": np.int64,
                "tensor(uint8)": np.uint8,
            }
            np_dtype = dtype_map.get(inp.type, np.float32)
            shape = [1 if (not isinstance(d, int) or d <= 0) else d for d in inp.shape]
            dummy = np.zeros(shape, dtype=np_dtype)
            self._model.run(None, {inp.name: dummy})  # JIT pass 1
            self._model.run(None, {inp.name: dummy})  # JIT pass 2
        except Exception:
            pass

    def _derive_coreml_peer(self) -> Path:
        """Return (creating if necessary) the CoreML .mlpackage peer for this artifact."""
        src = Path(self._model_path)
        peer = src.with_suffix(".mlpackage")
        if peer.exists():
            return peer
        h, w = self._metadata.input_size
        if self._uses_factor_backends():
            # Per-factor export: each factor backend derives its own peer.
            # This method should only be called for flat (non-bundle) artifacts.
            raise ClassifierRuntimeError(
                f"{self._model_path!r}: cannot derive a single CoreML peer for a "
                "multihead bundle — export each factor model individually"
            )
        if self._metadata.arch == "yolo":
            from ultralytics import YOLO

            YOLO(str(src)).export(format="coreml", imgsz=max(h, w))
            return peer
        if self._metadata.arch == "tinyclassifier":
            from hydra_suite.training.tiny_model import (
                export_tiny_to_coreml,
                load_tiny_classifier,
            )

            model, ckpt = load_tiny_classifier(str(src), device="cpu")
            export_tiny_to_coreml(model, ckpt, str(peer))
            return peer
        # torchvision / timm arch
        from hydra_suite.training.torchvision_model import (
            export_torchvision_to_coreml,
            load_torchvision_classifier,
        )

        model, ckpt = load_torchvision_classifier(str(src), device="cpu")
        export_torchvision_to_coreml(model, ckpt, str(peer))
        return peer

    def _load_coreml(self) -> None:
        """Load the .mlpackage via coremltools and cache the output feature name."""
        import coremltools

        peer = self._derive_coreml_peer()
        self._model = coremltools.models.MLModel(str(peer))
        # Resolve and cache the output feature name at load time so _forward_coreml
        # does not need to inspect the spec on every call.
        output_descs = self._model.output_description._fd_spec
        if output_descs:
            self._coreml_output_name: str | None = output_descs[0].name
        else:
            self._coreml_output_name = None
        self._active_execution_backend = "coreml"

    def _forward_coreml(self, batch_np: np.ndarray) -> np.ndarray:
        """Run a preprocessed (N, 3, H, W) float32 batch through the CoreML model.

        The output feature name assigned by coremltools varies by model graph
        (e.g. ``'var_23'``). We therefore index the prediction dict by position
        — taking the first value — rather than by a hardcoded name.

        The model was traced with an NCHW ``ct.TensorType(name="input", ...)``
        input, so we feed the preprocessed batch as-is in NCHW under the "input"
        key — no layout transpose is needed.
        """
        results: list[np.ndarray] = []
        for i in range(batch_np.shape[0]):
            single = batch_np[i : i + 1]  # (1, 3, H, W)
            pred = self._model.predict({"input": single})
            if (
                self._coreml_output_name is not None
                and self._coreml_output_name in pred
            ):
                logits = pred[self._coreml_output_name]
            else:
                # Fallback: take first value by position regardless of key name.
                logits = next(iter(pred.values()))
            results.append(np.asarray(logits, dtype=np.float32).reshape(-1))
        return np.stack(results, axis=0)

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
        t = torch.from_numpy(batch_np)
        if device.startswith("cuda") and torch.cuda.is_available():
            # Pinned staging enables async DMA to the CUDA device.
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        with torch.inference_mode():
            logits = self._model(t).float().cpu().numpy()
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

    def _forward_onnx_iobinding(self, batch_cuda) -> np.ndarray:
        """ONNX forward pass feeding the GPU batch directly via ORT IOBinding.

        Avoids the GPU\u2192CPU PCIe transfer that
        ``batch_cuda.contiguous().cpu().numpy()`` would incur.  Both
        ``CUDAExecutionProvider`` and ``TensorrtExecutionProvider`` support
        IOBinding; the output (K-class probabilities) is tiny and pulled to CPU
        after the forward pass.

        Falls back to the explicit transfer path on any error so that the
        caller never sees an unexpected exception from this optimisation.
        """
        try:
            inp = self._model.get_inputs()[0]
            # Ensure the tensor is contiguous before handing its data pointer
            # to ORT (e.g. the channel-flip in _preprocess_cuda can leave the
            # batch non-contiguous).
            batch_c = (
                batch_cuda if batch_cuda.is_contiguous() else batch_cuda.contiguous()
            )
            device_id = batch_c.device.index if batch_c.device.index is not None else 0
            binding = self._model.io_binding()
            binding.bind_input(
                name=inp.name,
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(batch_c.shape),
                buffer_ptr=batch_c.data_ptr(),
            )
            for out_meta in self._model.get_outputs():
                binding.bind_output(out_meta.name, device_type="cpu")
            self._model.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()[0]
        except Exception:
            batch_np = batch_cuda.contiguous().cpu().numpy()
            return self._forward_onnx(batch_np)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _cardinalities(self) -> list[int]:
        return [len(names) for names in self._metadata.class_names_per_factor]

    def _preprocess_cuda(
        self,
        crops_chw: list,  # list of (C, h, w) float32 CUDA tensors
        input_is_bgr: bool = True,
    ):
        """Resize, colour-convert, and normalise crops into an (N, 3, H, W) CUDA batch.

        GPU-native counterpart of :meth:`_preprocess`.  Each crop is an
        ``(C, h_crop, w_crop)`` float32 CUDA tensor in [0, 255] range.
        Returns a normalised float32 CUDA tensor suitable for
        :meth:`_forward_torch_cuda`.

        Parameters
        ----------
        crops_chw:
            List of ``(C, h, w)`` float32 CUDA tensors, one per detection.
        input_is_bgr:
            When True (default, standard cv2 BGR path), flip channels 0↔2
            to convert BGR→RGB before normalization.  Set False when crops
            come from an RGB source (e.g. NVDec).
        """
        import torch
        import torch.nn.functional as F

        h, w = self._metadata.input_size
        device = crops_chw[0].device

        # Fix degenerate crops and expand single-channel inputs first,
        # then stack into one tensor and do a SINGLE batch interpolate if needed.
        # This replaces the previous per-crop F.interpolate loop (O(N) kernel
        # launches) with at most one F.interpolate call on the full batch.
        fixed: list = []
        for crop in crops_chw:
            if crop is None or crop.numel() == 0:
                fixed.append(torch.zeros(3, h, w, dtype=torch.float32, device=device))
                continue
            t = crop
            if t.dim() == 2:
                # Single-channel → replicate to 3 channels
                t = t.unsqueeze(0).expand(3, -1, -1).contiguous()
            elif t.shape[0] == 1:
                t = t.expand(3, -1, -1).contiguous()
            fixed.append(t)

        batch = torch.stack(fixed, dim=0)  # (N, C, H_crop, W_crop) float32

        # Batch resize in a single F.interpolate call if needed.
        if batch.shape[-2] != h or batch.shape[-1] != w:
            batch = F.interpolate(
                batch,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )  # (N, C, H, W)

        # Channel flip BGR→RGB when the source is cv2-style BGR
        if input_is_bgr and batch.shape[1] == 3:
            batch = batch[:, [2, 1, 0], :, :]

        # Scale [0, 255] → [0, 1]
        batch = batch * (1.0 / 255.0)

        # Monochrome conversion (replicate luminance across 3 channels)
        if self._metadata.monochrome and batch.shape[1] == 3:
            gray = (
                0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3]
            )
            batch = gray.expand(-1, 3, -1, -1)

        # ImageNet normalisation
        if self._uses_imagenet_normalization():
            if self._metadata.monochrome:
                mean_v = float(_IMAGENET_MEAN.mean())
                std_v = float(_IMAGENET_STD.mean())
                mean_t = torch.tensor(
                    [mean_v, mean_v, mean_v], dtype=torch.float32, device=device
                ).view(1, 3, 1, 1)
                std_t = torch.tensor(
                    [std_v, std_v, std_v], dtype=torch.float32, device=device
                ).view(1, 3, 1, 1)
            else:
                mean_t = torch.as_tensor(
                    _IMAGENET_MEAN, dtype=torch.float32, device=device
                ).view(1, 3, 1, 1)
                std_t = torch.as_tensor(
                    _IMAGENET_STD, dtype=torch.float32, device=device
                ).view(1, 3, 1, 1)
            batch = (batch - mean_t) / std_t

        return batch  # (N, 3, H, W) float32 CUDA

    def _forward_torch_cuda(self, batch_cuda):
        """Run the torch model on a device-resident batch; returns logits on device.

        No host<->device transfers occur. The returned tensor stays on the same
        device as ``batch_cuda``.
        """
        import torch

        with torch.inference_mode():
            return self._model(batch_cuda).detach()

    def predict_batch_cuda(
        self,
        crops_chw: list,
        input_is_bgr: bool = True,
    ) -> "list[list[np.ndarray]]":
        """GPU-native :meth:`predict_batch` accepting CUDA crop tensors.

        Preprocessing and forward pass run entirely on-device.  Only the final
        per-crop probability vectors are moved to CPU.

        Parameters
        ----------
        crops_chw:
            List of ``(C, h, w)`` float32 CUDA tensors (e.g. from
            :func:`~hydra_suite.core.canonicalization.crop.gpu_canonical_crop`).
        input_is_bgr:
            Passed to :meth:`_preprocess_cuda` for channel-order handling.

        Returns
        -------
        list[list[np.ndarray]]
            Same ``[N_crops][K_factors]`` probability-vector structure as
            :meth:`predict_batch`.
        """
        if not crops_chw:
            return []
        self._ensure_loaded()

        if self._uses_factor_backends():
            # Factor-backend models require individual HWC numpy crops.
            numpy_crops = [
                c.permute(1, 2, 0).cpu().numpy() if hasattr(c, "cpu") else c
                for c in crops_chw
            ]
            return self.predict_batch(numpy_crops)

        try:
            # Preprocess on GPU in all cases — this is cheaper than 400 individual
            # GPU→CPU transfers and CPU cv2 resizes regardless of the forward backend.
            batch_cuda = self._preprocess_cuda(crops_chw, input_is_bgr=input_is_bgr)

            if self._active_execution_backend == "native":
                logits_cuda = self._forward_torch_cuda(batch_cuda)
                logits = logits_cuda.cpu().numpy()
            elif self._active_execution_backend == "onnx":
                # ONNX CUDA/TRT EP: feed from GPU memory via IOBinding to avoid
                # the PCIe round-trip of an explicit GPU\u2192CPU transfer.
                logits = self._forward_onnx_iobinding(batch_cuda)
            else:
                # Unknown backend — fall back to CPU path.
                numpy_crops = [
                    c.permute(1, 2, 0).cpu().numpy() if hasattr(c, "cpu") else c
                    for c in crops_chw
                ]
                return self.predict_batch(numpy_crops)
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierRuntimeError(
                f"CUDA inference failure for {self._model_path!r}: {exc}"
            ) from exc

        cardinalities = self._cardinalities()
        expected_total = sum(cardinalities)
        if logits.shape[-1] != expected_total:
            raise ClassifierRuntimeError(
                f"{self._model_path!r}: model output width {logits.shape[-1]} "
                f"does not match expected total {expected_total}"
            )
        results: list[list[np.ndarray]] = []
        for row in logits:
            per_factor: list[np.ndarray] = []
            offset = 0
            for width in cardinalities:
                per_factor.append(self._softmax(row[offset : offset + width]))
                offset += width
            results.append(per_factor)
        return results

    def _ensure_loaded_best_effort(self) -> None:
        """Load with best-effort TRT→native fallback for gpu_fast / coreml tiers.

        Calls ``_ensure_loaded()``; if the ONNX/TRT or CoreML path raises for any
        reason, logs a WARNING and reloads natively on the same device.  Never
        falls back to CPU — if the native device is unavailable the exception
        propagates.
        """
        if self._loaded:
            return
        if not self._uses_onnx() and not self._uses_coreml():
            self._ensure_loaded()
            return
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            native_device = _torch_device(self._compute_runtime)
            if not native_device or native_device == "cpu":
                # coreml only makes sense on Apple Silicon; fall back to mps
                native_device = "mps"
            logger.warning(
                "Accelerated classifier backend failed (%s); falling back to native %s",
                exc,
                native_device,
            )
            self._model = self._loader.load(self._model_path, native_device)
            self._active_execution_backend = "native"
            self._loaded = True

    def predict_batch(self, crops: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Run inference on ``crops``. Returns ``[N_crops][K_factors]`` probability vectors."""
        if not crops:
            return []
        self._ensure_loaded_best_effort()
        try:
            if self._active_execution_backend == "coreml":
                batch_np = self._preprocess(crops)
                logits = self._forward_coreml(batch_np)
            elif self._active_execution_backend == "onnx":
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
        self._active_execution_backend = "unloaded"
