from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import CNNConfig
from ..result import CNNDetectionPrediction, CNNFactorPrediction, CNNResult, OBBResult
from ..runtime import RuntimeContext, resolved_backend_for

logger = logging.getLogger(__name__)


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
    if (
        getattr(runtime, "tensor_on_cuda", False)
        and not backend.supports_cuda_forward()
    ):
        backend.close()
        raise RuntimeError(
            f"CNN classifier {config.model_path!r} lacks a CUDA-native forward, "
            "but the gpu tier with NVDEC requires it (no silent CPU fallback). "
            "Use a native-torch / ONNX classifier, or run on the cpu tier."
        )
    meta = backend.metadata
    return CNNModel(
        backend=backend,
        input_size=(meta.input_size[0], meta.input_size[1]),
        factor_names=list(meta.factor_names),
        factor_class_names=[list(cn) for cn in meta.class_names_per_factor],
    )


def run_cnn(
    frame: "np.ndarray | torch.Tensor",
    obb_result: OBBResult,
    model: CNNModel,
    config: CNNConfig,
    runtime: RuntimeContext,
    aspect_ratio: float = 2.0,
    margin: float = 1.3,
) -> CNNResult:
    """Run CNN identity classifier; returns raw pre-calibration probabilities.

    Crops are warped directly from the frame to the model input size
    (extract_classifier_crops), bit-identical to the legacy CNN crop path.

    Per Correction 16 / spec audit: temperature and scoring_mode are applied
    at tracking time inside IdentityEvidenceBuilder, NOT here. Cache writes
    receive raw probabilities; calibration changes never invalidate the cache.
    """
    if obb_result.num_detections == 0:
        return CNNResult(label=config.label, predictions=[])

    from .crops import extract_classifier_crops

    np_crops = extract_classifier_crops(
        frame, obb_result, model.input_size, aspect_ratio, margin
    )

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
    aspect_ratio: float = 2.0,
    margin: float = 1.3,
) -> "dict[int, CNNResult]":
    """Run CNN classifier over a window; return one CNNResult per frame.

    Builds classifier crops internally via extract_classifier_crops_batch (single
    warpAffine to model.input_size, BGR uint8 — bit-identical to the per-frame
    run_cnn path). Runs the backend ONCE over all crops (cross-frame perf win),
    then splits per frame via batch.select_frame. Assembly delegates to
    _assemble_cnn_result (DRY with run_cnn).
    """
    from .crops import frames_on_cuda

    if frames_on_cuda(runtime, frames):
        # Pure-GPU path (NVDEC): warp crops on-device and forward on-device, no
        # frame device->host copy. predict_batch_cuda expects [0,255] CHW cuda
        # tensors; floor-quantize to 8 bits so the input stays in the same regime
        # as the cv2/uint8 reference (grid_sample != cv2, so the acceptance gate
        # is identity agreement, not byte-identity -- see the design spec).
        from .crops import extract_classifier_crops_batch_gpu

        batch = extract_classifier_crops_batch_gpu(
            frames, obb_results, model.input_size, aspect_ratio, margin, runtime.device
        )
        n_total = batch.crops.shape[0]
        if n_total:
            # NVDEC frames (the only source of CUDA frames) are RGB, so
            # input_is_bgr=False: the model sees RGB, matching the CPU path where
            # _preprocess flips its BGR crop to RGB.
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
        n_total = batch.crops.shape[0]
        if n_total:
            # Single batched host transfer + vectorized uint8 quantization. This
            # is byte-identical to the former per-crop `.cpu().numpy()` loop (same
            # values) but performs ONE device->host copy instead of N, cutting the
            # per-crop sync overhead on dense frames.
            hwc_all = np.ascontiguousarray(
                batch.crops.permute(0, 2, 3, 1).cpu().numpy()
            )
            stacked = (hwc_all * 255.0).clip(0, 255).astype(np.uint8)
            np_crops: list[np.ndarray] = list(stacked)
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
