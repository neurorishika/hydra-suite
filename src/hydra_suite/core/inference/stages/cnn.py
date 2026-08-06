from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry

from ..config import CNNConfig
from ..result import CNNDetectionPrediction, CNNFactorPrediction, CNNResult, OBBResult
from ..runtime import RuntimeContext, resolved_backend_for

logger = logging.getLogger(__name__)

# Fallback canonical geometry for callers that omit ``geometry`` (matches the
# project-wide default `InferenceConfig.canonical` built from
# REFERENCE_BODY_SIZE=20/aspect=2.0/margin=1.3 defaults).
_DEFAULT_CANONICAL_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)


@dataclass
class CNNModel:
    backend: Any  # ClassifierBackend
    input_size: tuple[int, int]  # (H, W)
    factor_names: list[str]  # one per factor (len=1 flat; len=K multi-head)
    factor_class_names: list[list[str]]  # class names per factor

    def close(self) -> None:
        pass


def load_cnn_model(config: CNNConfig, runtime: RuntimeContext) -> CNNModel:
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    # The RuntimeContext carries the single resolved backend/device (from
    # runtime_tier); per-stage compute_runtime fields no longer exist.
    # Use resolved_backend_for so a hand-built context (resolved=None) degrades
    # gracefully instead of raising AttributeError, matching obb/pose stages.
    resolved = resolved_backend_for(runtime)
    if resolved.backend in ("tensorrt", "coreml"):
        logger.warning(
            "CNN stage: gpu_fast (%s) requested — "
            "best-effort native fallback applies if the accelerated artifact "
            "is unavailable.",
            resolved.backend,
        )
    backend = ClassifierBackend(config.model_path, resolved)
    if getattr(runtime, "cuda_mode", False) and not backend.supports_cuda_forward():
        backend.close()
        raise RuntimeError(
            f"CNN classifier {config.model_path!r} lacks a CUDA-native forward, "
            "but a GPU tier (gpu/gpu_fast) on CUDA routes NVDEC/GPU crops through "
            "predict_batch_cuda (no silent CPU fallback). Use a native-torch / "
            "ONNX classifier, or run on the cpu tier."
        )
    meta = backend.metadata
    in_h, in_w = meta.input_size  # ClassifierMetadata documents (H, W)
    return CNNModel(
        backend=backend,
        input_size=(in_h, in_w),
        factor_names=list(meta.factor_names),
        factor_class_names=[list(cn) for cn in meta.class_names_per_factor],
    )


def run_cnn(
    frame: "np.ndarray | torch.Tensor",
    obb_result: OBBResult,
    model: CNNModel,
    config: CNNConfig,
    runtime: RuntimeContext,
    geometry: CanonicalGeometry = _DEFAULT_CANONICAL_GEOMETRY,
) -> CNNResult:
    """Run CNN identity classifier; returns raw pre-calibration probabilities.

    Crops are warped onto the shared canonical canvas (extract_classifier_crops,
    Layer 1), then fit to the model's exact input size via Layer 2
    (``fit_to_model_input`` / ``apply_fit``) -- the classifier crop is no longer
    warped straight to the model input in one step, so every consumer shares
    the same rigid Layer 1 transform.

    Per Correction 16 / spec audit: temperature and scoring_mode are applied
    at tracking time inside IdentityEvidenceBuilder, NOT here. Cache writes
    receive raw probabilities; calibration changes never invalidate the cache.
    """
    if obb_result.num_detections == 0:
        return CNNResult(label=config.label, predictions=[])

    from .crops import extract_classifier_crops

    canon_crops = extract_classifier_crops(frame, obb_result, geometry)
    in_h, in_w = model.input_size  # ClassifierMetadata documents (H, W)
    fit = fit_to_model_input(geometry.canvas_wh, (in_w, in_h))
    np_crops = [apply_fit(c, fit) for c in canon_crops]

    all_probs = model.backend.predict_batch(np_crops)

    return _assemble_cnn_result(all_probs, model, config)


def _assemble_cnn_result(
    all_probs: list,
    model: "CNNModel",
    config: CNNConfig,
    det_index_offset: int = 0,
) -> CNNResult:
    """Assemble CNNResult from raw backend predictions.

    Shared by run_cnn and run_cnn_batch to keep per-detection logic DRY.
    det_index_offset allows batch path to assign correct global detection indices.
    """
    predictions: list[CNNDetectionPrediction] = []
    for det_idx, probs_per_factor in enumerate(all_probs):
        factors = [
            CNNFactorPrediction(
                factor_name=model.factor_names[k],
                class_names=model.factor_class_names[k],
                raw_probabilities=np.array(probs_per_factor[k], dtype=np.float32),
            )
            for k in range(len(probs_per_factor))
        ]
        predictions.append(
            CNNDetectionPrediction(
                det_index=det_index_offset + det_idx, factors=factors
            )
        )
    return CNNResult(label=config.label, predictions=predictions)


def run_cnn_batch(
    frames: "list",
    obb_results: "list[OBBResult]",
    model: "CNNModel",
    config: CNNConfig,
    runtime: RuntimeContext,
    geometry: CanonicalGeometry = _DEFAULT_CANONICAL_GEOMETRY,
) -> "dict[int, CNNResult]":
    """Run CNN classifier over a window; return one CNNResult per frame.

    Builds classifier crops internally via extract_classifier_crops_batch (the
    shared canonical canvas, BGR uint8 -- bit-identical to the per-frame run_cnn
    path), then fits each to the model input via Layer 2 exactly like run_cnn.
    Runs the backend ONCE over all crops (cross-frame perf win), then splits
    per frame via batch.select_frame. Assembly delegates to _assemble_cnn_result
    (DRY with run_cnn).

    Both branches derive their fit from the SAME ``fit_to_model_input(geometry
    .canvas_wh, (in_w, in_h))`` call, computed once here. The GPU (NVDEC
    on-device) branch applies that fit on-device via ``apply_fit_gpu``
    (``F.interpolate`` + zero-canvas paste) instead of the CPU's ``cv2``-based
    ``apply_fit`` -- an anisotropic stretch here would silently feed the model
    a DIFFERENT geometry than the CPU path (a model trained on letterboxed
    crops fed non-letterboxed ones on CUDA), so both paths must letterbox
    identically; only the resample kernel differs (grid_sample/interpolate
    != cv2, so the acceptance gate is identity agreement, not byte-identity).
    """
    from .crops import frames_on_cuda

    in_h, in_w = model.input_size  # ClassifierMetadata documents (H, W)
    fit = fit_to_model_input(geometry.canvas_wh, (in_w, in_h))

    if frames_on_cuda(runtime, frames):
        # Pure-GPU path (NVDEC): warp crops on-device and forward on-device, no
        # frame device->host copy. predict_batch_cuda expects [0,255] CHW cuda
        # tensors; floor-quantize to 8 bits so the input stays in the same regime
        # as the cv2/uint8 reference (grid_sample != cv2, so the acceptance gate
        # is identity agreement, not byte-identity -- see the design spec).
        from .crops import apply_fit_gpu, extract_classifier_crops_batch_gpu

        batch = extract_classifier_crops_batch_gpu(
            frames, obb_results, geometry, runtime.device
        )
        n_total = batch.crops.shape[0]
        if n_total:
            # NVDEC frames (the only source of CUDA frames) are RGB, so
            # input_is_bgr=False: the model sees RGB, matching the CPU path where
            # _preprocess flips its BGR crop to RGB.
            fitted = apply_fit_gpu(batch.crops, fit)
            cuda_crops = [
                (fitted[i] * 255.0).floor().clamp(0, 255) for i in range(n_total)
            ]
            all_probs = model.backend.predict_batch_cuda(cuda_crops, input_is_bgr=False)
        else:
            all_probs = []
    else:
        from .crops import apply_fit_batch, extract_classifier_crops_batch_np

        # HWC uint8 BGR crops straight from the warp -- no float32 tensor round
        # trip (it was exactly value-preserving, so removing it is
        # byte-identical; see NumpyCropBatch).
        batch = extract_classifier_crops_batch_np(frames, obb_results, geometry)
        if batch.crops:
            np_crops: list[np.ndarray] = apply_fit_batch(batch.crops, fit)
            all_probs = model.backend.predict_batch(np_crops)
        else:
            all_probs = []

    results: dict[int, CNNResult] = {}
    prob_offset = 0
    for frame_idx in sorted(batch.obb_by_frame):
        rows = batch.select_frame(frame_idx)
        n = len(rows)
        frame_probs = all_probs[prob_offset : prob_offset + n]
        prob_offset += n
        results[frame_idx] = _assemble_cnn_result(frame_probs, model, config)
    return results
