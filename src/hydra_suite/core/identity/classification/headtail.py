"""Standalone head-tail direction analyzer.

Wraps ``ClassifierBackend`` with head-tail-specific validation: flat model
only, labels must be a subset of {up, down, left, right, unknown} after alias
normalization.  Provides ``analyze_crops`` for frame-based inference (consumed
by the YOLO detector and the crops worker) and ``predict_labels`` /
``analyze_detections`` for crop-list-based callers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

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

_HEADTAIL_HEADING_OFFSETS: Dict[str, float] = {
    "right": 0.0,
    "left": np.pi,
    "up": -np.pi / 2.0,
    "down": np.pi / 2.0,
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
    if len(set(normalized)) != len(normalized):
        raise HeadTailFormatError(
            "head-tail model labels must be unique after alias normalization"
        )
    return normalized


def heading_for_direction(axis_theta: float, direction: str) -> Optional[float]:
    """Map a canonical head-tail label to a directed global heading.

    ``axis_theta`` is the global direction of the canonical crop's positive
    x-axis, which corresponds to the crop's right-hand side. The label offsets
    therefore match ``apply_headtail_rotation`` semantics: ``right`` is no
    rotation, ``left`` is 180 degrees, ``up`` is 90 degrees clockwise, and
    ``down`` is 90 degrees counter-clockwise in image coordinates.
    """
    canonical = _LABEL_ALIASES.get(str(direction or "").strip().lower())
    if canonical not in _HEADTAIL_HEADING_OFFSETS:
        return None
    return float(
        (float(axis_theta) + _HEADTAIL_HEADING_OFFSETS[canonical]) % (2.0 * np.pi)
    )


class HeadTailAnalyzer:
    """Classifier-agnostic head-tail direction analyzer.

    Wraps ``ClassifierBackend`` for v2 classifier artifacts and enforces:
    - flat model (not multi-head) — raises ``HeadTailFormatError``
    - labels subset of {up, down, left, right, unknown} after alias normalization

    Consumers use ``analyze_crops(frames, per_frame_obb_corners)`` for the
    full frame-based pipeline.  New callers may use ``predict_labels(crops)``
    or ``analyze_detections(crops, obb_major_axes)`` for simpler crop lists.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        conf_threshold: float = 0.5,
        batch_size: int = 64,
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
        self._batch_size = max(1, int(batch_size))
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

        self._backend: str = "none"
        self._model = None
        self._class_names: Optional[List[str]] = None
        self._input_size: Optional[Tuple[int, int]] = None
        self._backend_obj = None  # ClassifierBackend instance (v2 path)
        self._canonical_labels: tuple = ()

        if model_path:
            self._load_model(model_path)

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
        """Name of the active inference backend ('backend_v2' or 'none')."""
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
        """Retained for compatibility; v2-backed analyzers expose no raw model."""
        return self._model

    def is_loaded(self) -> bool:
        """True when the analyzer has a model ready for inference."""
        return self.is_available

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path_str: str) -> None:
        """Load a v2 classifier artifact through the shared backend."""
        from hydra_suite.core.identity.classification.backend import ClassifierBackend
        from hydra_suite.core.identity.classification.errors import (
            ClassifierFormatError,
            HeadTailFormatError,
        )

        path = str(model_path_str)
        path_obj = Path(path).expanduser().resolve()

        if not path_obj.exists():
            raise ClassifierFormatError(
                f"HeadTailAnalyzer: path does not exist: {path!r}"
            )

        try:
            backend_obj = ClassifierBackend(
                path,
                compute_runtime=self._compute_runtime,
                trt_profile_max_batch=self._batch_size,
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
            # Eagerly trigger ONNX session creation and TRT JIT warmup so that
            # Ada/Hopper JIT compilation happens now (during model setup) rather
            # than stalling on the first Phase-1 batch inference call.
            backend_obj._ensure_loaded()
            self._backend = "backend_v2"
            self._class_names = normalized
            self._input_size = meta.input_size
            self._canonical_labels = tuple(normalized)
            logger.info(
                "HeadTailAnalyzer: loaded classifier (%s) from %s",
                meta.arch,
                path_obj.name,
            )
            return
        except Exception:
            if "backend_obj" in locals() and backend_obj is not None:
                backend_obj.close()
            raise

    # ------------------------------------------------------------------
    # v2 API: predict_labels and analyze_detections
    # ------------------------------------------------------------------

    def predict_labels(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Return ``(canonical_label, confidence)`` per crop.

        Labels below ``conf_threshold`` are collapsed to ``"unknown"`` with
        their original confidence.
        """
        if not crops:
            return []

        cls_results = self._predict(crops)
        if cls_results is None or len(cls_results) == 0:
            return [("unknown", 0.0)] * len(crops)

        out: List[Tuple[str, float]] = []
        for per_factor in cls_results[: len(crops)]:
            try:
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
            heading = heading_for_direction(axis, label)
            if heading is None:
                results.append((float(axis), conf, False))
                continue
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

        # For CUDA-class runtimes, auto-route through the GPU-native path so
        # that crop extraction (batched GPU warp) and inference (GPU forward
        # pass) both stay on-device, avoiding N sequential cv2 warp+resize
        # calls on the CPU.  Any failure falls through to the CPU path below.
        if self._compute_runtime in ("cuda", "onnx_cuda", "tensorrt"):
            try:
                import torch

                if torch.cuda.is_available():
                    frames_cuda = [
                        torch.from_numpy(np.ascontiguousarray(f)).to(
                            "cuda", non_blocking=True
                        )
                        for f in frames
                    ]
                    return self.analyze_crops_cuda(
                        frames_cuda,
                        per_frame_obb_corners,
                        profiler=profiler,
                        input_is_bgr=True,
                    )
            except Exception:
                logger.debug(
                    "HeadTailAnalyzer.analyze_crops: GPU auto-route failed, "
                    "falling back to CPU path",
                    exc_info=True,
                )

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
        self._scatter_backend_v2(cls_results, all_meta, results)

    # ------------------------------------------------------------------
    # GPU-native API: analyze_crops_cuda
    # ------------------------------------------------------------------

    def analyze_crops_cuda(
        self,
        frames_hwc: list,
        per_frame_obb_corners: List[List[np.ndarray]],
        profiler=None,
        *,
        input_is_bgr: bool = False,
    ) -> List[List[Tuple[float, float, int]]]:
        """GPU-native head-tail analysis for CUDA-resident frames.

        Mirrors :meth:`analyze_crops` but accepts CUDA tensors instead of
        numpy arrays.  The affine warp, resize, and classifier forward pass
        all stay on-device; only the final (tiny) probability vectors are
        moved to CPU.

        Parameters
        ----------
        frames_hwc:
            List of ``(H, W, C)`` CUDA tensors (uint8 or float32).  Typically
            produced by NVDec (RGB, uint8) or the sequential GPU pipeline.
        per_frame_obb_corners:
            Same format as :meth:`analyze_crops`.
        input_is_bgr:
            Set True when frames use BGR channel ordering (cv2 convention).
            Default False assumes RGB (NVDec output).
        """
        if not self.is_available:
            return [
                [(float("nan"), 0.0, 0)] * len(corners)
                for corners in per_frame_obb_corners
            ]

        if profiler is not None:
            profiler.phase_start("headtail_crop")
        all_crops_cuda, all_meta = self._collect_canonical_crops_cuda(
            frames_hwc, per_frame_obb_corners
        )
        if profiler is not None:
            profiler.phase_end("headtail_crop", work_units=len(all_crops_cuda))

        results: List[List[Tuple[float, float, int]]] = [
            [(float("nan"), 0.0, 0)] * len(c) for c in per_frame_obb_corners
        ]

        if not all_crops_cuda:
            return results

        if profiler is not None:
            profiler.phase_start("headtail_inference")
        cls_results = self._predict_cuda(all_crops_cuda, input_is_bgr=input_is_bgr)
        if profiler is not None:
            profiler.phase_end("headtail_inference", work_units=len(all_crops_cuda))

        if cls_results is None or len(cls_results) == 0:
            return results

        self._scatter_results(cls_results, all_meta, results)
        return results

    def _collect_canonical_crops_cuda(
        self,
        frames_hwc: list,
        per_frame_obb_corners: List[List[np.ndarray]],
    ) -> Tuple[list, List[Tuple[int, int, float, np.ndarray]]]:
        """Extract canonical crops from CUDA frames and return metadata.

        Uses :func:`~hydra_suite.core.canonicalization.crop.gpu_canonical_crop_batch`
        to warp all detections from each frame in a single batched GPU call,
        reducing kernel launch overhead from O(total_crops) to O(n_frames).
        """
        import torch

        from hydra_suite.core.canonicalization.crop import (
            compute_alignment_affine,
            gpu_canonical_crop_batch,
        )

        if self._input_size is not None and len(self._input_size) == 2:
            out_w = max(8, int(self._input_size[0]))
            out_h = max(8, int(self._input_size[1]))
        else:
            out_w = 128
            out_h = max(8, int(round(128 / self._ref_ar)))
            out_h = out_h + (out_h % 2)

        all_crops: list = []
        all_meta: List[Tuple[int, int, float, np.ndarray]] = []

        for fi, (frame, corners_list) in enumerate(
            zip(frames_hwc, per_frame_obb_corners)
        ):
            if not corners_list:
                continue

            # Prepare frame tensor: HWC uint8/float32 → CHW float32
            t = frame
            if not isinstance(t, torch.Tensor):
                continue
            if t.dtype == torch.uint8:
                t = t.to(dtype=torch.float32)
            if t.dim() == 3 and t.shape[2] <= 4:  # HWC → CHW
                t = t.permute(2, 0, 1).contiguous()

            # Compute all alignment affines for this frame on CPU (fast).
            M_aligns_frame: list = []
            meta_frame: list = []
            for di, corners in enumerate(corners_list):
                try:
                    M_align, axis_theta = compute_alignment_affine(
                        corners, out_w, out_h, self._padding_fraction
                    )
                except ValueError:
                    continue
                M_aligns_frame.append(M_align)
                meta_frame.append((fi, di, float(axis_theta), M_align))

            if not M_aligns_frame:
                continue

            # ONE batched warp for all detections in this frame.
            try:
                crops_batch = gpu_canonical_crop_batch(t, M_aligns_frame, out_w, out_h)
            except Exception:
                continue

            for i, crop in enumerate(crops_batch.unbind(0)):
                if crop.numel() > 0:
                    all_crops.append(crop)
                    all_meta.append(meta_frame[i])

        return all_crops, all_meta

    def _canonicalize_obb_cuda(self, frame_hwc, corners):
        """Extract a GPU-resident canonical crop using ``gpu_canonical_crop``.

        Parameters
        ----------
        frame_hwc:
            CUDA tensor ``(H, W, C)`` uint8 or float32.
        corners:
            ``(4, 2)`` numpy array of OBB corner pixel coords.

        Returns
        -------
        tuple[Tensor, float, np.ndarray] | None
            ``(crop_chw, axis_theta, M_align)`` or None on failure.
        """
        import torch

        from hydra_suite.core.canonicalization.crop import (
            compute_alignment_affine,
            gpu_canonical_crop,
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

        # Convert (H, W, C) → (C, H, W) float32 for gpu_canonical_crop
        t = frame_hwc
        if not isinstance(t, torch.Tensor):
            return None
        if t.dtype == torch.uint8:
            t = t.to(dtype=torch.float32)
        if t.dim() == 3 and t.shape[2] <= 4:
            # HWC → CHW
            t = t.permute(2, 0, 1).contiguous()
        # t is now (C, H, W) float32 CUDA

        try:
            crop_cuda = gpu_canonical_crop(t, M_align, out_w, out_h)
        except Exception:
            return None

        if crop_cuda is None or crop_cuda.numel() == 0:
            return None
        return crop_cuda, axis_theta, M_align

    def _effective_infer_batch_size(self) -> int:
        """Return the runtime-safe per-call batch size for inference.

        The UI-configured ``batch_size`` is honored on all runtimes.  TensorRT
        has a hard profile upper bound, so the user value is clamped to that
        maximum to prevent runtime shape errors.
        """
        chunk = max(1, int(self._batch_size))
        if self._compute_runtime == "tensorrt":
            from hydra_suite.core.identity.classification.backend import (
                ClassifierBackend,
            )

            chunk = min(chunk, ClassifierBackend._TRT_PROFILE_MAX_BATCH)
        return chunk

    def _predict_cuda(self, crops_cuda: list, input_is_bgr: bool = False):
        """Run inference on CUDA crop tensors via :meth:`.predict_batch_cuda`.

        Crops are processed in fixed-size chunks controlled by ``_batch_size``
        so users can choose a stable throughput/memory operating point across
        runs.  TensorRT requests are capped by its compiled profile limit.
        """
        if self._backend_obj is None or not crops_cuda:
            return []
        chunk = self._effective_infer_batch_size()
        if len(crops_cuda) <= chunk:
            return self._backend_obj.predict_batch_cuda(
                crops_cuda, input_is_bgr=input_is_bgr
            )
        out = []
        for start in range(0, len(crops_cuda), chunk):
            out.extend(
                self._backend_obj.predict_batch_cuda(
                    crops_cuda[start : start + chunk],
                    input_is_bgr=input_is_bgr,
                )
            )
        return out

    def _scatter_backend_v2(self, cls_results, all_meta, results) -> None:
        """Scatter v2 ClassifierBackend results (list of per_factor prob arrays)."""
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
            theta = heading_for_direction(axis_theta, label)
            if theta is None:
                results[fi][di] = (float("nan"), best_conf, 0)
                continue
            results[fi][di] = (float(theta), best_conf, 1)

    def close(self) -> None:
        """Release the loaded model and reset the backend to 'none'."""
        self._model = None
        self._backend = "none"
        if self._backend_obj is not None:
            self._backend_obj.close()
            self._backend_obj = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, source_crops: List[np.ndarray]):
        """Run inference on source crops through the shared classifier backend."""
        if self._backend_obj is None or not source_crops:
            return []
        # GPU runtimes use the user-configured fixed chunk size, clamped for
        # TensorRT profile safety.
        compute_rt = getattr(self, "_compute_runtime", "cpu")
        if compute_rt in ("cuda", "onnx_cuda", "tensorrt"):
            chunk = self._effective_infer_batch_size()
            if len(source_crops) <= chunk:
                return self._backend_obj.predict_batch(source_crops)
            out = []
            for start in range(0, len(source_crops), chunk):
                out.extend(
                    self._backend_obj.predict_batch(source_crops[start : start + chunk])
                )
            return out
        # CPU / MPS path: chunk by _batch_size to keep peak memory bounded.
        if self._batch_size >= len(source_crops):
            return self._backend_obj.predict_batch(source_crops)

        out = []
        for start in range(0, len(source_crops), self._batch_size):
            out.extend(
                self._backend_obj.predict_batch(
                    source_crops[start : start + self._batch_size]
                )
            )
        return out

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
                        "Expected a non-empty subset of "
                        "up/down/left/right/unknown."
                    )
                return ordered  # can't normalize; return raw
            normalized.append(token)

        if strict:
            normalized_set = frozenset(normalized)
            if not normalized_set:
                raise ValueError(
                    f"Unsupported head-tail class schema in {source}: {ordered}. "
                    "Expected a non-empty subset of up/down/left/right/unknown."
                )
            if len(normalized_set) != len(normalized):
                raise ValueError(
                    f"Duplicate or aliased head-tail labels in {source}: {ordered}."
                )
            if not normalized_set.issubset(HEADTAIL_CANONICAL_LABELS):
                raise ValueError(
                    f"Unsupported head-tail class schema in {source}: {ordered}. "
                    "Expected a non-empty subset of up/down/left/right/unknown."
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
        return "cuda"  # kept for legacy configs; ROCm is no longer supported
    return "cpu"
