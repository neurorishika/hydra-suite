from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import HeadTailConfig
from ..result import HeadTailResult, OBBResult
from ..runtime import RuntimeContext, resolved_backend_for

logger = logging.getLogger(__name__)

_DIRECTION_OFFSET: dict[str, float] = {
    "right": 0.0,
    "left": math.pi,
    "up": -math.pi / 2,
    "down": math.pi / 2,
}


@dataclass
class HeadTailModel:
    backend: Any  # ClassifierBackend instance
    input_size: tuple[int, int]  # (H, W) expected by the model
    class_names: list[str]

    def close(self) -> None:
        pass


def load_headtail_model(
    config: HeadTailConfig, runtime: RuntimeContext
) -> HeadTailModel:
    """Load a head-tail classifier, enforcing the legacy head-tail contract.

    Mirrors ``HeadTailAnalyzer._load_model`` (H7 parity):
    - Rejects multi-head artifacts with ``HeadTailFormatError`` (a multi-head
      model would otherwise silently load and use only factor 0).
    - Normalizes the checkpoint's class labels through the canonical alias map
      (``head_left``, ``north``, ``n``, … → ``left``/``up``/…) via
      ``validate_headtail_labels``, so non-canonical-but-known labels still map
      to a heading offset instead of silently becoming undirected.
    """
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.core.identity.classification.errors import HeadTailFormatError
    from hydra_suite.core.identity.classification.headtail import (
        validate_headtail_labels,
    )

    # The RuntimeContext carries the single resolved backend/device (from
    # runtime_tier); per-stage compute_runtime fields no longer exist.
    # Use resolved_backend_for so a hand-built context (resolved=None) degrades
    # gracefully instead of raising AttributeError, matching obb/pose stages.
    resolved = resolved_backend_for(runtime)
    if resolved.backend in ("tensorrt", "coreml"):
        logger.warning(
            "HeadTail stage: gpu_fast (%s) requested — "
            "best-effort native fallback applies if the accelerated artifact "
            "is unavailable.",
            resolved.backend,
        )
    backend = ClassifierBackend(config.model_path, resolved)
    meta = backend.metadata
    if meta.is_multihead:
        backend.close()
        raise HeadTailFormatError(
            "head-tail requires a flat classifier, got multi-head with "
            f"factors={meta.factor_names!r}"
        )
    # Raises HeadTailFormatError if labels are not a subset of the canonical set.
    normalized = validate_headtail_labels(list(meta.class_names_per_factor[0]))
    if (
        getattr(runtime, "tensor_on_cuda", False)
        and not backend.supports_cuda_forward()
    ):
        backend.close()
        raise RuntimeError(
            f"Head-tail classifier {config.model_path!r} lacks a CUDA-native "
            "forward, but the gpu tier with NVDEC requires it (no silent CPU "
            "fallback). Use a native-torch / ONNX classifier, or run on the cpu tier."
        )
    input_size = (meta.input_size[0], meta.input_size[1])
    return HeadTailModel(
        backend=backend,
        input_size=input_size,
        class_names=normalized,
    )


def run_headtail(
    frame: "np.ndarray | torch.Tensor",
    obb_result: OBBResult,
    model: HeadTailModel,
    config: HeadTailConfig,
    runtime: RuntimeContext,
    aspect_ratio: float = 2.0,
    margin: float = 1.3,
) -> HeadTailResult:
    """Classify head-tail orientation per detection. No I/O, no mode branching.

    Crops are warped directly from the frame to the classifier input size
    (extract_classifier_crops), bit-identical to the legacy head-tail path, so
    direction decisions match legacy at the classifier boundary.

    Per Correction 15: canonical_affines is None — affines belong to the crops
    stage, not headtail. Downstream consumers must check for None and recompute
    if needed.
    """
    n = obb_result.num_detections

    if n == 0:
        return HeadTailResult(
            heading_hints=np.full(n, float("nan"), dtype=np.float32),
            heading_confidences=np.zeros(n, dtype=np.float32),
            directed_mask=np.zeros(n, dtype=np.uint8),
            canonical_affines=None,
        )

    from .crops import extract_classifier_crops

    np_crops = extract_classifier_crops(
        frame, obb_result, model.input_size, aspect_ratio, margin
    )

    all_probs = model.backend.predict_batch(np_crops)

    # axis_theta for the head-tail offset map is derived empirically from the
    # OBB corners with atan2(c[1]-c[0]). Reasoning: the new pipeline's crop
    # builder rotates by `obb.angles` (folded to [0, pi)), but YOLO's native
    # xyxyxyxy corners are in the *un-folded* axis ordering, so the canonical
    # crop the classifier sees is rotated 180 degrees relative to the legacy
    # detector's crop for ~94 percent of detections. Using atan2 of YOLO's
    # native corners as axis_theta cancels that flip and recovers legacy parity
    # for those detections. Falls back to obb.angles if corners are degenerate.
    signed_axes = _signed_major_axis_from_corners(obb_result.corners)

    return _assemble_headtail_result(all_probs, obb_result, model, config, signed_axes)


def _assemble_headtail_result(
    all_probs: list,
    obb_result: OBBResult,
    model: "HeadTailModel",
    config: HeadTailConfig,
    signed_axes: "np.ndarray | None",
) -> HeadTailResult:
    """Assemble HeadTailResult from raw backend predictions.

    Shared by run_headtail and run_headtail_batch so the per-detection logic
    is never duplicated.

    Mirrors legacy's ``_select_headtail_candidate_indices`` confidence gate
    (``YOLO_HEADTAIL_DETECT_CONF_THRESHOLD``): detections whose OBB confidence
    falls below ``config.candidate_confidence_threshold`` are left undirected
    even if the classifier would have returned a confident prediction for
    them. ``candidate_confidence_threshold=None`` (the default) classifies
    every detection, matching prior behavior.
    """
    n = obb_result.num_detections
    hints = np.full(n, float("nan"), dtype=np.float32)
    confs = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.uint8)
    candidate_threshold = config.candidate_confidence_threshold

    for i, probs_per_factor in enumerate(all_probs):
        if i >= n:
            break
        if (
            candidate_threshold is not None
            and float(obb_result.confidences[i]) < candidate_threshold
        ):
            continue
        factor_probs = probs_per_factor[0]
        winning_idx = int(np.argmax(factor_probs))
        winning_conf = float(factor_probs[winning_idx])
        if winning_conf < config.confidence_threshold:
            continue
        label = model.class_names[winning_idx]
        offset = _label_to_heading_offset(label)
        if offset is None:
            continue
        axis_theta = float(obb_result.angles[i])
        if signed_axes is not None and math.isfinite(float(signed_axes[i])):
            axis_theta = float(signed_axes[i])
        hints[i] = (axis_theta + offset) % (2.0 * math.pi)
        confs[i] = winning_conf
        mask[i] = 1

    return HeadTailResult(
        heading_hints=hints,
        heading_confidences=confs,
        directed_mask=mask,
        canonical_affines=None,
    )


def _label_to_heading_offset(label: str) -> float | None:
    """Map a direction label to its angle offset relative to the OBB major axis.

    Applies canonical alias normalization (``head_left``/``north``/``n``/… →
    ``left``/``up``/…) before lookup so non-canonical-but-known labels still
    resolve to an offset (H7 parity). Returns ``None`` for ``unknown`` or any
    unrecognized token, leaving the detection undirected.
    """
    from hydra_suite.core.identity.classification.headtail import (
        normalize_headtail_label,
    )

    try:
        canonical = normalize_headtail_label(label)
    except ValueError:
        return None
    return _DIRECTION_OFFSET.get(canonical)


def _signed_major_axis_from_corners(corners: np.ndarray) -> np.ndarray | None:
    """Return per-detection major-axis angle from OBB corners, in [-pi, pi].

    Mirrors ``compute_alignment_affine`` in core.canonicalization.crop: picks
    the longer of the first two edges (c[1]-c[0] vs c[2]-c[1]) and returns
    ``atan2(major_vec_y, major_vec_x)``. NaN for degenerate boxes.
    """
    if corners is None or corners.size == 0 or corners.ndim != 3:
        return None
    n = corners.shape[0]
    out = np.full(n, float("nan"), dtype=np.float32)
    for i in range(n):
        c = corners[i].reshape(4, 2)
        e01 = float(np.linalg.norm(c[1] - c[0]))
        e12 = float(np.linalg.norm(c[2] - c[1]))
        if e01 < 1e-3 or e12 < 1e-3:
            continue
        major_vec = c[1] - c[0] if e01 >= e12 else c[2] - c[1]
        out[i] = float(math.atan2(float(major_vec[1]), float(major_vec[0])))
    return out


def run_headtail_batch(
    frames: "list",
    obb_results: "list[OBBResult]",
    model: HeadTailModel,
    config: HeadTailConfig,
    runtime: RuntimeContext,
    aspect_ratio: float = 2.0,
    margin: float = 1.3,
) -> "dict[int, HeadTailResult]":
    """Run head-tail classification over a window; return one HeadTailResult per frame.

    Builds classifier crops internally via extract_classifier_crops_batch (single
    warpAffine to model.input_size, BGR uint8 — bit-identical to the per-frame
    run_headtail path). Runs the backend ONCE over all crops (cross-frame perf win),
    then splits per frame via batch.select_frame. Assembly delegates to
    _assemble_headtail_result (DRY with run_headtail).
    """
    from .crops import frames_on_cuda

    if frames_on_cuda(runtime, frames):
        # Pure-GPU path (NVDEC): warp + forward on-device, no frame D->H copy.
        # floor-quantize to [0,255] 8-bit to match the cv2/uint8 reference regime
        # (grid_sample != cv2 -> identity-agreement gate, not byte-identity).
        from .crops import extract_classifier_crops_batch_gpu

        batch = extract_classifier_crops_batch_gpu(
            frames, obb_results, model.input_size, aspect_ratio, margin, runtime.device
        )
        n_total = batch.crops.shape[0]
        if n_total:
            # NVDEC frames (the only CUDA frame source) are RGB -> input_is_bgr=False
            # so the model sees RGB, matching the CPU path's BGR->RGB flip.
            cuda_crops = [
                (batch.crops[i] * 255.0).floor().clamp(0, 255) for i in range(n_total)
            ]
            all_probs = model.backend.predict_batch_cuda(cuda_crops, input_is_bgr=False)
        else:
            all_probs = []
    else:
        from .crops import extract_classifier_crops_batch

        batch = extract_classifier_crops_batch(
            frames, obb_results, model.input_size, aspect_ratio, margin
        )
        # batch.crops is NCHW float [0,1]; convert back to HWC uint8 for
        # predict_batch. Single batched host transfer + vectorized uint8
        # quantization -- byte-identical to the former per-crop `.cpu().numpy()`
        # loop (same values), one D->H copy instead of N.
        n_total = batch.crops.shape[0]
        if n_total:
            hwc_all = np.ascontiguousarray(
                batch.crops.permute(0, 2, 3, 1).cpu().numpy()
            )
            stacked = (hwc_all * 255.0).clip(0, 255).astype(np.uint8)
            np_crops: list[np.ndarray] = list(stacked)
            all_probs = model.backend.predict_batch(np_crops)
        else:
            all_probs = []

    results: dict[int, HeadTailResult] = {}
    prob_offset = 0
    for frame_idx in sorted(batch.obb_by_frame):
        obb = batch.obb_by_frame[frame_idx]
        rows = batch.select_frame(frame_idx)
        n = len(rows)
        if n == 0:
            results[frame_idx] = HeadTailResult(
                heading_hints=np.full(
                    obb.num_detections, float("nan"), dtype=np.float32
                ),
                heading_confidences=np.zeros(obb.num_detections, dtype=np.float32),
                directed_mask=np.zeros(obb.num_detections, dtype=np.uint8),
                canonical_affines=None,
            )
            continue
        frame_probs = all_probs[prob_offset : prob_offset + n]
        prob_offset += n
        signed_axes = _signed_major_axis_from_corners(obb.corners)
        results[frame_idx] = _assemble_headtail_result(
            frame_probs, obb, model, config, signed_axes
        )
    return results
