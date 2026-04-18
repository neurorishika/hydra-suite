"""Standalone head-tail direction analyzer.

Wraps ``ClassifierBackend`` with head-tail-specific validation: flat model
only, labels must be a subset of {up, down, left, right, unknown} after alias
normalization.  Provides ``analyze_crops`` for frame-based inference (consumed
by the YOLO detector and the crops worker) and ``predict_labels`` /
``analyze_detections`` for crop-list-based callers.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Canonical class sets (shared with engine.py)
_HEADTAIL_DIRECTIONAL_CLASS_SET = frozenset({"left", "right"})

HEADTAIL_CANONICAL_LABELS: frozenset[str] = frozenset(
    {"up", "down", "left", "right", "unknown"}
)

_LABEL_ALIASES: Dict[str, str] = {
    "u": "up",
    "up": "up",
    "n": "up",
    "north": "up",
    "d": "down",
    "down": "down",
    "s": "down",
    "south": "down",
    "l": "left",
    "left": "left",
    "w": "left",
    "west": "left",
    "head_left": "left",
    "r": "right",
    "right": "right",
    "e": "right",
    "east": "right",
    "head_right": "right",
    "head_up": "up",
    "head_down": "down",
    "?": "unknown",
    "unknown": "unknown",
    "none": "unknown",
    "na": "unknown",
    "head_unknown": "unknown",
}


def normalize_headtail_label(raw: str) -> str:
    """Return the canonical head-tail label for ``raw``.

    Accepts case-insensitive aliases listed in ``_LABEL_ALIASES``. Raises
    ``ValueError`` for unrecognised tokens.
    """
    key = str(raw).strip().lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    raise ValueError(f"unknown head-tail label {raw!r}")


def validate_headtail_labels(labels: list[str]) -> list[str]:
    """Validate a checkpoint's class labels against the head-tail contract.

    Each label must normalize to a member of ``HEADTAIL_CANONICAL_LABELS``.
    Returns the list of normalized labels in input order. Raises
    ``HeadTailFormatError`` listing offending entries when the check fails.
    """
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError

    normalized: list[str] = []
    offending: list[str] = []
    for raw in labels:
        try:
            normalized.append(normalize_headtail_label(raw))
        except ValueError:
            offending.append(str(raw))
    if offending or not normalized:
        raise HeadTailFormatError(
            "head-tail model labels must be a non-empty subset of "
            f"{{up, down, left, right, unknown}}; offending labels: {offending}"
        )
    return normalized


class HeadTailAnalyzer:
    """Classifier-agnostic head-tail direction analyzer.

    Primary path: wraps ``ClassifierBackend`` for v2 classifier artifacts
    (TinyClassifier, TorchvisionClassifier, YOLO flat).  Enforces:
    - flat model (not multi-head) — raises ``HeadTailFormatError``
    - labels subset of {up, down, left, right, unknown} after alias normalization

    Legacy path: accepts pre-loaded model components via ``from_components``
    (consumed by ``YOLOOBBDetector._load_headtail_model`` YOLO fallback and
    existing tests that monkey-patch internal state).

    Consumers use ``analyze_crops(frames, per_frame_obb_corners)`` for the
    full frame-based pipeline.  New callers may use ``predict_labels(crops)``
    or ``analyze_detections(crops, obb_major_axes)`` for simpler crop lists.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        conf_threshold: float = 0.5,
        reference_aspect_ratio: float = 2.0,
        canonical_margin: float = 1.3,
        predict_device: Optional[str] = None,
        *,
        compute_runtime: Optional[str] = None,
    ) -> None:
        """Construct a HeadTailAnalyzer from a classifier artifact path.

        Accepts both the legacy ``device=`` parameter (maps to torch device)
        and the new ``compute_runtime=`` parameter (ClassifierBackend runtime).
        When ``compute_runtime`` is provided it takes precedence.

        Raises:
            HeadTailFormatError: model is multi-head or labels are not a subset
                of the canonical head-tail set.
        """
        self._conf_threshold = float(conf_threshold)
        self._ref_ar = max(1.0, reference_aspect_ratio)
        self._padding_fraction = max(0.0, canonical_margin - 1.0)
        self._canonical_margin = float(canonical_margin)
        self._predict_device = predict_device

        # Resolve device string from compute_runtime or legacy device arg
        if compute_runtime is not None:
            self._compute_runtime = str(compute_runtime)
            self._device = _runtime_to_device(self._compute_runtime)
        else:
            self._device = str(device) if device else "cpu"
            self._compute_runtime = self._device

        # These fields exist for backward compat with legacy path (from_components)
        self._backend: str = "none"
        self._model = None
        self._class_names: Optional[List[str]] = None
        self._input_size: Optional[Tuple[int, int]] = None
        self._backend_obj = None  # ClassifierBackend instance (v2 path)
        self._canonical_labels: tuple = ()

        if model_path:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Class methods / static factory
    # ------------------------------------------------------------------

    @classmethod
    def from_components(
        cls,
        model,
        backend: str,
        class_names: Optional[List[str]],
        input_size: Optional[Tuple[int, int]],
        device: str = "cpu",
        conf_threshold: float = 0.5,
        reference_aspect_ratio: float = 2.0,
        canonical_margin: float = 1.3,
        predict_device: Optional[str] = None,
    ) -> "HeadTailAnalyzer":
        """Create from a pre-loaded model, skipping file-based loading.

        Used by ``YOLOOBBDetector._load_headtail_model`` for YOLO classify
        models that go through the engine's own runtime loader.
        """
        obj = cls.__new__(cls)
        obj._device = str(device) if device else "cpu"
        obj._compute_runtime = obj._device
        obj._conf_threshold = float(conf_threshold)
        obj._ref_ar = max(1.0, reference_aspect_ratio)
        obj._padding_fraction = max(0.0, canonical_margin - 1.0)
        obj._canonical_margin = float(canonical_margin)
        obj._predict_device = predict_device
        obj._model = model
        obj._backend = str(backend)
        obj._class_names = class_names
        obj._input_size = input_size
        obj._backend_obj = None
        obj._canonical_labels = tuple(class_names) if class_names else ()
        return obj

    # ------------------------------------------------------------------
    # v2 class methods
    # ------------------------------------------------------------------

    @classmethod
    def valid_output_labels(cls) -> frozenset:
        """Return the frozenset of allowed canonical head-tail labels."""
        return HEADTAIL_CANONICAL_LABELS

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def canonical_labels(self) -> tuple:
        """Normalized labels in checkpoint order."""
        return self._canonical_labels

    @property
    def is_available(self) -> bool:
        """True if a model is loaded and the backend is not 'none'."""
        backend_obj = getattr(self, "_backend_obj", None)
        model = getattr(self, "_model", None)
        backend = getattr(self, "_backend", "none")
        return (model is not None or backend_obj is not None) and backend != "none"

    @property
    def backend(self) -> str:
        """Name of the active inference backend ('tiny', 'classkit_tiny', 'backend_v2', 'yolo', or 'none')."""
        return self._backend

    @property
    def class_names(self) -> Optional[List[str]]:
        """Class names reported by the loaded model, or None if unavailable."""
        return self._class_names

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        """Expected (height, width) crop input size for the loaded model, or None."""
        return self._input_size

    @property
    def model(self):
        """The underlying loaded model object (legacy path), or None for v2 path."""
        return self._model

    def is_loaded(self) -> bool:
        """True when the analyzer has a model ready for inference."""
        return self.is_available

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path_str: str) -> None:
        """Load classifier artifact, preferring v2 ClassifierBackend path."""
        from hydra_suite.core.identity.classification.backend import ClassifierBackend
        from hydra_suite.core.identity.classification.errors import (
            ClassifierFormatError,
            HeadTailFormatError,
        )

        path = str(model_path_str)
        path_obj = Path(path).expanduser().resolve()

        if not path_obj.exists():
            logger.warning("HeadTailAnalyzer: path does not exist: %s", path)
            return

        # Attempt v2 ClassifierBackend path first.
        # For .pth/.pt files, we peek at schema_version to avoid feeding
        # legacy raw state_dicts into ClassifierBackend (which requires v2).
        skip_v2 = False
        if path_obj.suffix.lower() in {".pth", ".pt"}:
            skip_v2 = self._is_legacy_checkpoint(path)

        if not skip_v2:
            try:
                backend_obj = ClassifierBackend(
                    path, compute_runtime=self._compute_runtime
                )
                meta = backend_obj.metadata
                if meta.is_multihead:
                    raise HeadTailFormatError(
                        f"head-tail requires a flat classifier, got multi-head with "
                        f"factors={meta.factor_names!r}"
                    )
                raw_labels = meta.class_names_per_factor[0]
                # raises HeadTailFormatError if labels not in canonical set
                normalized = validate_headtail_labels(raw_labels)
                self._backend_obj = backend_obj
                self._backend = "backend_v2"
                self._class_names = normalized
                self._input_size = meta.input_size
                self._canonical_labels = tuple(normalized)
                logger.info(
                    "HeadTailAnalyzer: loaded v2 classifier (%s) from %s",
                    meta.arch,
                    path_obj.name,
                )
                return
            except HeadTailFormatError:
                # Propagate head-tail validation errors — these are hard failures.
                raise
            except ClassifierFormatError:
                # Not a v2 artifact — fall through to legacy tiny loader.
                pass
            except Exception as exc:
                logger.debug("HeadTailAnalyzer: v2 path failed for %s: %s", path, exc)

        # Legacy tiny / classkit_tiny path for old single-output .pth files
        tiny_result = self._try_load_tiny(path)
        if tiny_result is not None:
            model, class_names, input_size = tiny_result
            self._input_size = input_size
            if class_names is not None:
                self._class_names = self._validate_class_names(class_names)
                self._backend = "classkit_tiny"
                self._canonical_labels = tuple(self._class_names)
            else:
                self._backend = "tiny"
                self._canonical_labels = ()
            self._model = model
            logger.info(
                "HeadTailAnalyzer: loaded legacy %s model from %s",
                self._backend,
                path_obj.name,
            )
            return

        # Try YOLO classify model via ultralytics
        try:
            from ultralytics import YOLO

            model = YOLO(path, task="classify")
            model_names = getattr(model, "names", None)
            if model_names is None:
                model_names = getattr(getattr(model, "model", None), "names", None)
            self._class_names = self._validate_class_names(model_names)
            self._backend = "yolo"
            self._model = model
            self._canonical_labels = (
                tuple(self._class_names) if self._class_names else ()
            )
            logger.info(
                "HeadTailAnalyzer: loaded YOLO classify model from %s",
                path_obj.name,
            )
        except Exception as exc:
            logger.warning("HeadTailAnalyzer: failed to load model: %s", exc)

    @staticmethod
    def _is_legacy_checkpoint(path: str) -> bool:
        """Return True if path looks like a legacy (pre-v2) checkpoint.

        We peek at the file without loading weights to avoid the overhead of
        loading a full checkpoint only to reject it.
        """
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not isinstance(ckpt, dict):
            return True  # raw state_dict or unknown format
        return ckpt.get("schema_version") != 2

    def _try_load_tiny(self, model_path_str: str):
        import torch

        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists():
            return None
        if model_path.suffix.lower() not in {".pth", ".pt"}:
            return None

        try:
            checkpoint = torch.load(
                str(model_path), map_location="cpu", weights_only=False
            )
        except Exception:
            return None

        state_dict = None
        input_size = (128, 64)
        class_names = None

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint.get("model_state_dict")
            maybe_size = checkpoint.get("input_size")
            if isinstance(maybe_size, (list, tuple)) and len(maybe_size) == 2:
                input_size = (int(maybe_size[0]), int(maybe_size[1]))
            raw_names = checkpoint.get("class_names")
            if isinstance(raw_names, (list, tuple)) and raw_names:
                class_names = [str(n) for n in raw_names]
        elif isinstance(checkpoint, (dict, OrderedDict)):
            state_dict = checkpoint
        else:
            return None

        if not isinstance(state_dict, (dict, OrderedDict)):
            return None
        keys = list(state_dict.keys())
        if not keys or not any(str(k).startswith("features.") for k in keys):
            return None

        linear_keys = sorted(
            [k for k in keys if k.startswith("classifier.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        if not linear_keys:
            return None
        n_out = int(state_dict[linear_keys[-1]].shape[0])

        if n_out == 1:
            model = self._build_tiny_classifier(input_size=input_size)
            model.load_state_dict(state_dict, strict=True)
        else:
            try:
                from hydra_suite.training.tiny_model import rebuild_from_checkpoint

                model = rebuild_from_checkpoint({"model_state_dict": state_dict})
            except Exception as exc:
                logger.warning("Failed to load ClassKit tiny head-tail: %s", exc)
                return None

        import torch

        device = torch.device(self._device)
        model.to(device)
        model.eval()
        return model, class_names, input_size

    @staticmethod
    def _build_tiny_classifier(input_size=(128, 64)):
        import torch.nn as nn

        class _TinyHeadClassifier(nn.Module):
            def __init__(self, input_size=(128, 64)):
                super().__init__()
                self.input_size = tuple(input_size)
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                """Run the feature extractor then the classifier head; returns a (B,1) logit tensor."""
                x = self.features(x)
                return self.classifier(x)

        return _TinyHeadClassifier(input_size=input_size)

    # ------------------------------------------------------------------
    # v2 API: predict_labels and analyze_detections
    # ------------------------------------------------------------------

    def predict_labels(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Return ``(canonical_label, confidence)`` per crop.

        Labels below ``conf_threshold`` are collapsed to ``"unknown"`` with
        their original confidence.  Works on both the v2 ClassifierBackend
        path and the legacy paths.
        """
        if not crops:
            return []

        if self._backend_obj is not None:
            # v2 ClassifierBackend path
            raw = self._backend_obj.predict_batch(crops)
            out: List[Tuple[str, float]] = []
            for per_factor in raw:
                probs = per_factor[0]
                best_idx = int(np.argmax(probs))
                best_conf = float(probs[best_idx])
                if self._canonical_labels and best_idx < len(self._canonical_labels):
                    label = self._canonical_labels[best_idx]
                else:
                    label = "unknown"
                if best_conf < self._conf_threshold:
                    label = "unknown"
                out.append((label, best_conf))
            return out

        # Legacy path: delegate to _predict and map results
        cls_results = self._predict(crops)
        if cls_results is None or len(cls_results) == 0:
            return [("unknown", 0.0)] * len(crops)

        out = []
        if self._backend == "tiny":
            probs = np.asarray(cls_results, dtype=np.float32).reshape(-1)
            for j in range(min(len(crops), len(probs))):
                p_right = float(probs[j])
                conf = max(p_right, 1.0 - p_right)
                label = "right" if p_right >= 0.5 else "left"
                if conf < self._conf_threshold:
                    label = "unknown"
                out.append((label, conf))
        elif self._backend == "classkit_tiny":
            for j in range(min(len(crops), len(cls_results))):
                try:
                    direction, conf = cls_results[j]
                except Exception:
                    out.append(("unknown", 0.0))
                    continue
                label = (
                    direction if direction in HEADTAIL_CANONICAL_LABELS else "unknown"
                )
                if float(conf) < self._conf_threshold:
                    label = "unknown"
                out.append((label, float(conf)))
        else:
            # YOLO or unknown
            for j in range(min(len(crops), len(cls_results))):
                try:
                    result = cls_results[j]
                    if result is None:
                        out.append(("unknown", 0.0))
                        continue
                    probs_obj = getattr(result, "probs", None)
                    if probs_obj is None:
                        out.append(("unknown", 0.0))
                        continue
                    top1 = int(getattr(probs_obj, "top1", -1))
                    top1_conf = float(getattr(probs_obj, "top1conf", 0.0))
                    label_str = self._label_from_top1(top1)
                    if top1 < 0 or top1_conf < self._conf_threshold:
                        out.append(("unknown", top1_conf))
                    else:
                        canonical = _LABEL_ALIASES.get(label_str, "unknown")
                        out.append((canonical, top1_conf))
                except Exception:
                    out.append(("unknown", 0.0))
        while len(out) < len(crops):
            out.append(("unknown", 0.0))
        return out

    def analyze_detections(
        self,
        crops: List[np.ndarray],
        obb_major_axes: List[float],
    ) -> List[Tuple[float, float, bool]]:
        """Return ``(heading_radians, confidence, directed_flag)`` per crop."""
        labels = self.predict_labels(crops)
        results: List[Tuple[float, float, bool]] = []
        for (label, conf), axis in zip(labels, obb_major_axes):
            if label == "unknown":
                results.append((float(axis), conf, False))
                continue
            if label in ("up", "left"):
                heading = axis
            else:  # down, right
                heading = axis + np.pi
            results.append((float(heading), conf, True))
        return results

    # ------------------------------------------------------------------
    # Frame-based API: analyze_crops (consumed by yolo_detector, crops_worker)
    # ------------------------------------------------------------------

    def analyze_crops(
        self,
        frames: List[np.ndarray],
        per_frame_obb_corners: List[List[np.ndarray]],
        profiler=None,
    ) -> List[List[Tuple[float, float, int]]]:
        """Run head-tail analysis on multiple frames.

        Args:
            frames: BGR video frames.
            per_frame_obb_corners: For each frame, a list of (4,2) OBB corners.

        Returns:
            Per-frame list of ``(heading_radians, confidence, directed_flag)``
            tuples.  ``heading_radians`` is ``nan`` when direction is ambiguous.
            ``directed_flag`` is 1 when classifier was confident, 0 otherwise.
        """
        if not self.is_available:
            return [
                [(float("nan"), 0.0, 0)] * len(corners)
                for corners in per_frame_obb_corners
            ]

        # Phase 1: collect canonical crops across all frames
        if profiler is not None:
            profiler.phase_start("headtail_crop")
        all_crops, all_meta = self._collect_canonical_crops(
            frames, per_frame_obb_corners
        )
        if profiler is not None:
            profiler.phase_end("headtail_crop", work_units=len(all_crops))

        # Pre-allocate results
        results: List[List[Tuple[float, float, int]]] = [
            [(float("nan"), 0.0, 0)] * len(c) for c in per_frame_obb_corners
        ]

        if not all_crops:
            return results

        # Phase 2: single GPU inference pass
        if profiler is not None:
            profiler.phase_start("headtail_inference")
        cls_results = self._predict(all_crops)
        if profiler is not None:
            profiler.phase_end("headtail_inference", work_units=len(all_crops))
        if cls_results is None or len(cls_results) == 0:
            return results

        # Phase 3: scatter results by backend type
        self._scatter_results(cls_results, all_meta, results)
        return results

    def _collect_canonical_crops(
        self,
        frames: List[np.ndarray],
        per_frame_obb_corners: List[List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, float, np.ndarray]]]:
        """Extract canonical crops and metadata from all frames."""
        all_crops: List[np.ndarray] = []
        all_meta: List[Tuple[int, int, float, np.ndarray]] = []
        for fi, (frame, corners_list) in enumerate(zip(frames, per_frame_obb_corners)):
            for di, corners in enumerate(corners_list):
                result = self._canonicalize_obb(frame, corners)
                if result is None:
                    continue
                crop, axis_theta, M_align = result
                all_crops.append(crop)
                all_meta.append((fi, di, float(axis_theta), M_align))
        return all_crops, all_meta

    def _scatter_results(
        self,
        cls_results,
        all_meta: List[Tuple[int, int, float, np.ndarray]],
        results: List[List[Tuple[float, float, int]]],
    ) -> None:
        """Scatter classification results back into the per-frame results grid."""
        if self._backend == "backend_v2":
            self._scatter_backend_v2(cls_results, all_meta, results)
        elif self._backend == "tiny":
            self._scatter_tiny(cls_results, all_meta, results)
        elif self._backend == "classkit_tiny":
            self._scatter_classkit_tiny(cls_results, all_meta, results)
        else:
            self._scatter_yolo(cls_results, all_meta, results)

    def _scatter_backend_v2(self, cls_results, all_meta, results) -> None:
        """Scatter v2 ClassifierBackend results (list of per_factor prob arrays)."""
        TWO_PI = 2.0 * np.pi
        for j in range(min(len(all_meta), len(cls_results))):
            fi, di, axis_theta, _ = all_meta[j]
            try:
                per_factor = cls_results[j]
                probs = per_factor[0]
                best_idx = int(np.argmax(probs))
                best_conf = float(probs[best_idx])
                if self._canonical_labels and best_idx < len(self._canonical_labels):
                    label = self._canonical_labels[best_idx]
                else:
                    label = "unknown"
            except Exception:
                continue
            if best_conf < self._conf_threshold or label == "unknown":
                results[fi][di] = (float("nan"), best_conf, 0)
                continue
            if label in ("up", "left"):
                theta = axis_theta
            else:  # down, right
                theta = axis_theta + np.pi
            results[fi][di] = (float(theta % TWO_PI), best_conf, 1)

    def _scatter_tiny(self, cls_results, all_meta, results) -> None:
        TWO_PI = 2.0 * np.pi
        probs = np.asarray(cls_results, dtype=np.float32).reshape(-1)
        for j in range(min(len(all_meta), len(probs))):
            fi, di, axis_theta, _ = all_meta[j]
            p_right = float(probs[j])
            conf = max(p_right, 1.0 - p_right)
            if conf < self._conf_threshold:
                results[fi][di] = (float("nan"), float(conf), 0)
                continue
            theta = axis_theta if p_right >= 0.5 else (axis_theta + np.pi)
            results[fi][di] = (float(theta % TWO_PI), float(conf), 1)

    def _scatter_classkit_tiny(self, cls_results, all_meta, results) -> None:
        TWO_PI = 2.0 * np.pi
        for j in range(min(len(all_meta), len(cls_results))):
            fi, di, axis_theta, _ = all_meta[j]
            try:
                direction, conf = cls_results[j]
            except Exception:
                continue
            if direction not in {"left", "right"} or float(conf) < self._conf_threshold:
                results[fi][di] = (float("nan"), float(conf), 0)
                continue
            theta = axis_theta if direction == "right" else (axis_theta + np.pi)
            results[fi][di] = (float(theta % TWO_PI), float(conf), 1)

    def _scatter_yolo(self, cls_results, all_meta, results) -> None:
        TWO_PI = 2.0 * np.pi
        for j in range(min(len(all_meta), len(cls_results))):
            fi, di, axis_theta, _ = all_meta[j]
            try:
                result = cls_results[j]
                if result is None:
                    continue
                probs_obj = getattr(result, "probs", None)
                if probs_obj is None:
                    continue
                top1 = int(getattr(probs_obj, "top1", -1))
                top1_conf = float(getattr(probs_obj, "top1conf", 0.0))
                if top1 < 0 or top1_conf < self._conf_threshold:
                    results[fi][di] = (float("nan"), top1_conf, 0)
                    continue
                label = self._label_from_top1(top1)
                direction = self._class_to_direction(label, cls_idx=top1)
                if direction is None:
                    results[fi][di] = (float("nan"), top1_conf, 0)
                    continue
                theta = axis_theta if direction == "right" else (axis_theta + np.pi)
                results[fi][di] = (float(theta % TWO_PI), top1_conf, 1)
            except Exception:
                continue

    def close(self) -> None:
        """Release the loaded model and reset the backend to 'none'."""
        self._model = None
        self._backend = "none"
        if self._backend_obj is not None:
            self._backend_obj.close()
            self._backend_obj = None

    # ------------------------------------------------------------------
    # Inference (routes to v2 backend or legacy model)
    # ------------------------------------------------------------------

    def _predict(self, source_crops: List[np.ndarray]):
        """Run inference on source crops. Routes by backend type."""
        if self._backend_obj is not None:
            # v2 path: use ClassifierBackend.predict_batch
            return self._backend_obj.predict_batch(source_crops)

        if self._model is None or not source_crops:
            return []

        if self._backend == "tiny":
            import torch

            batch = self._crops_to_tensor(source_crops, self._input_size)
            device = torch.device(self._device)
            batch = batch.to(device)
            with torch.inference_mode():
                logits = self._model(batch)
                probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            return probs

        if self._backend == "classkit_tiny":
            import torch
            import torch.nn.functional as F

            batch = self._crops_to_tensor(source_crops, self._input_size)
            device = torch.device(self._device)
            batch = batch.to(device)
            with torch.inference_mode():
                logits = self._model(batch)
                softmax = F.softmax(logits, dim=1)
                top1_conf, top1_idx = softmax.max(dim=1)
                top1_conf = top1_conf.detach().cpu().numpy()
                top1_idx = top1_idx.detach().cpu().numpy()

            classified = []
            for cls_idx, conf in zip(top1_idx, top1_conf):
                label = self._label_from_top1(int(cls_idx))
                direction = self._class_to_direction(label, cls_idx=int(cls_idx))
                classified.append((direction, float(conf)))
            return classified

        # YOLO backend
        try:
            kwargs = dict(source=source_crops, conf=0.0, verbose=False)
            if self._predict_device is not None:
                kwargs["device"] = self._predict_device
            return self._model.predict(**kwargs)
        except Exception:
            outputs = []
            for crop in source_crops:
                try:
                    kw = dict(source=crop, conf=0.0, verbose=False)
                    if self._predict_device is not None:
                        kw["device"] = self._predict_device
                    one = self._model.predict(**kw)
                    outputs.append(one[0] if one else None)
                except Exception:
                    outputs.append(None)
            return outputs

    @staticmethod
    def _crops_to_tensor(source_crops, target_hw=None):
        import torch

        tensors = []
        for crop in source_crops:
            c = np.asarray(crop)
            if c.ndim == 2:
                c = np.stack([c, c, c], axis=-1)
            if c.ndim == 3 and c.shape[2] == 3:
                c = c[:, :, ::-1].copy()  # BGR -> RGB
            if target_hw is not None:
                w, h = int(target_hw[0]), int(target_hw[1])
                if c.shape[1] != w or c.shape[0] != h:
                    c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Canonical crop extraction
    # ------------------------------------------------------------------

    def _canonicalize_obb(self, frame, corners):
        from hydra_suite.core.canonicalization.crop import (
            compute_alignment_affine,
            extract_canonical_crop,
        )

        if self._input_size is not None and len(self._input_size) == 2:
            out_w, out_h = max(8, int(self._input_size[0])), max(
                8, int(self._input_size[1])
            )
        else:
            out_w = 128
            out_h = max(8, int(round(128 / self._ref_ar)))
            out_h = out_h + (out_h % 2)

        try:
            M_align, axis_theta = compute_alignment_affine(
                corners, out_w, out_h, self._padding_fraction
            )
        except ValueError:
            return None

        crop = extract_canonical_crop(frame, M_align, out_w, out_h)
        if crop is None or crop.size == 0:
            return None
        return crop, axis_theta, M_align

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _label_from_top1(self, cls_idx: int) -> str:
        names = self._class_names
        if names is None:
            return ""
        if isinstance(names, dict):
            return str(names.get(int(cls_idx), "")).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= int(cls_idx) < len(names):
            return str(names[int(cls_idx)]).strip().lower()
        return ""

    def _class_to_direction(self, label: str, cls_idx=None) -> Optional[str]:
        text = _LABEL_ALIASES.get(
            str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
        )
        if text == "left":
            return "left"
        if text == "right":
            return "right"
        if text in {"up", "down", "unknown"}:
            return None
        # Fallback for unnamed binary classifiers
        names = self._class_names
        if names is not None:
            ordered = (
                [str(v) for v in names]
                if isinstance(names, (list, tuple))
                else [str(v) for _, v in sorted(names.items())]
            )
            if len(ordered) == 2 and cls_idx is not None:
                return "right" if int(cls_idx) == 1 else "left"
        return None

    @staticmethod
    def _validate_class_names(
        class_names, *, strict: bool = False, source: str = "model"
    ) -> List[str]:
        if class_names is None:
            if strict:
                raise ValueError(f"{source} is missing class names.")
            return []
        if isinstance(class_names, dict):
            try:
                ordered = [
                    str(v)
                    for _, v in sorted(class_names.items(), key=lambda kv: int(kv[0]))
                ]
            except Exception:
                ordered = [str(v) for v in class_names.values()]
        elif isinstance(class_names, (list, tuple)):
            ordered = [str(n) for n in class_names]
        else:
            if strict:
                raise ValueError(
                    f"Unexpected class_names type in {source}: {type(class_names)}"
                )
            return []

        normalized = []
        for raw in ordered:
            token = _LABEL_ALIASES.get(
                raw.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if token is None:
                if strict:
                    raise ValueError(
                        f"Unsupported head-tail class label {raw!r} in {source}. "
                        "Expected exactly left/right or up/down/left/right/unknown."
                    )
                return ordered  # can't normalize; return raw
            normalized.append(token)

        if strict:
            normalized_set = frozenset(normalized)
            if len(normalized_set) != len(normalized):
                raise ValueError(
                    f"Duplicate or aliased head-tail labels in {source}: {ordered}."
                )
            if normalized_set not in (
                _HEADTAIL_DIRECTIONAL_CLASS_SET,
                HEADTAIL_CANONICAL_LABELS,
            ):
                raise ValueError(
                    f"Unsupported head-tail class schema in {source}: {ordered}. "
                    "Expected exactly left/right or up/down/left/right/unknown."
                )

        return normalized


# ---------------------------------------------------------------------------
# Module-level helper: map compute_runtime to torch device string
# ---------------------------------------------------------------------------


def _runtime_to_device(compute_runtime: str) -> str:
    """Map a canonical compute_runtime string to a torch device string."""
    rt = str(compute_runtime or "cpu")
    if rt in ("cuda", "onnx_cuda", "tensorrt"):
        return "cuda"
    if rt in ("mps", "onnx_coreml"):
        return "mps"
    if rt in ("rocm", "onnx_rocm"):
        return "cuda"
    return "cpu"
