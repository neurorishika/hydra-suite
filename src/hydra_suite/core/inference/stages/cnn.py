from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hydra_suite.core.canonicalization.fit import fit_crops_for_model
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

from ..config import CNNConfig
from ..result import CNNDetectionPrediction, CNNFactorPrediction, CNNResult, OBBResult
from ..runtime import RuntimeContext, resolved_backend_for
from ._resource_close import close_backend_resource

logger = logging.getLogger(__name__)


@dataclass
class CNNModel:
    backend: Any  # ClassifierBackend
    input_size: tuple[int, int]  # (H, W)
    factor_names: list[str]  # one per factor (len=1 flat; len=K multi-head)
    factor_class_names: list[list[str]]  # class names per factor

    def close(self) -> None:
        # Reach past this wrapper into the real backend to actually release
        # model weights (each phase owns its own ClassifierBackend instance).
        close_backend_resource(self.backend)


def load_cnn_model(config: CNNConfig, runtime: RuntimeContext) -> CNNModel:
    from hydra_suite.core.individual.classification.backend import ClassifierBackend

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
    geometry: CanonicalGeometry,
) -> CNNResult:
    """Run CNN identity classifier; returns raw pre-calibration probabilities.

    Crops are warped onto the shared canonical canvas (extract_classifier_crops,
    Layer 1), then fit to the model's exact input size via Layer 2
    (``fit_crops_for_model``, dispatched on the artifact's ``fit_policy`` --
    letterbox/squash/native) -- the classifier crop is no longer warped
    straight to the model input in one step, so every consumer shares the
    same rigid Layer 1 transform.

    Per Correction 16 / spec audit: temperature and scoring_mode are applied
    at tracking time inside IdentityEvidenceBuilder, NOT here. Cache writes
    receive raw probabilities; calibration changes never invalidate the cache.
    """
    if obb_result.num_detections == 0:
        return CNNResult(label=config.label, predictions=[])

    from .crops import extract_classifier_crops

    canon_crops = extract_classifier_crops(frame, obb_result, geometry)
    np_crops = fit_crops_for_model(
        canon_crops, model.input_size, model.backend.metadata.fit_policy
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
    geometry: CanonicalGeometry,
    headtail_by_frame: "dict[int, Any] | None" = None,
) -> "dict[int, CNNResult]":
    """Run CNN classifier over a window; return one CNNResult per frame.

    ``headtail_by_frame`` (optional, ``{frame_idx: HeadTailResult}``): when
    given, the identity catalog's ordered classes (R8: ``pink_yellow`` !=
    ``yellow_pink``) get a head-first crop wherever the head/tail stage is
    confident (directed) for a detection -- threaded into both the CPU crop
    path (``extract_classifier_crops_batch_np``) and the CUDA/NVDEC crop path
    (``extract_canonical_crops_batch``) so identity agreement holds
    regardless of which branch a window takes. Omitted, both branches are
    exactly byte-identical to before.

    Builds classifier crops internally via extract_classifier_crops_batch_np (the
    shared canonical canvas, BGR uint8 -- bit-identical to the per-frame run_cnn
    path), then fits each to the model input via Layer 2 exactly like run_cnn.
    Runs the backend ONCE over all crops (cross-frame perf win), then splits
    per frame via batch.select_frame. Assembly delegates to _assemble_cnn_result
    (DRY with run_cnn).

    Both branches dispatch Layer 2 on the SAME ``model.backend.metadata
    .fit_policy`` (letterbox/squash/native). The GPU (NVDEC on-device) branch
    applies that policy on-device via the shared torch seam's
    ``resample.fit_batch_for_model`` (``F.interpolate``, letterbox pastes onto
    a zero canvas / squash stretches straight to the model canvas) instead of
    the CPU's ``fit_crops_for_model`` -- both paths must apply the SAME policy
    or a model would silently see a DIFFERENT geometry on CUDA than on CPU;
    only the resample kernel differs (grid_sample/interpolate != cv2, so the
    acceptance gate is identity agreement, not byte-identity).
    """
    from .crops import frames_on_cuda

    in_h, in_w = model.input_size  # ClassifierMetadata documents (H, W)
    policy = model.backend.metadata.fit_policy

    if frames_on_cuda(runtime, frames):
        # Pure-GPU path (NVDEC): warp crops on-device and forward on-device, no
        # frame device->host copy. predict_batch_cuda expects [0,255] CHW cuda
        # tensors; floor-quantize to 8 bits so the input stays in the same regime
        # as the cv2/uint8 reference (grid_sample != cv2, so the acceptance gate
        # is identity agreement, not byte-identity -- see the design spec).
        from hydra_suite.core.canonicalization.resample import fit_batch_for_model

        from .crops import extract_canonical_crops_batch

        with span(N.CROP_EXTRACT):
            batch = extract_canonical_crops_batch(
                frames,
                obb_results,
                geometry,
                runtime,
                headtail_by_frame=headtail_by_frame,
            )
        n_total = batch.crops.shape[0]
        if n_total:
            # NVDEC frames (the only source of CUDA frames) are RGB, so
            # input_is_bgr=False: the model sees RGB, matching the CPU path where
            # _preprocess flips its BGR crop to RGB.
            with span(N.APPLY_FIT, units=n_total):
                fitted = fit_batch_for_model(batch.crops, (in_w, in_h), policy)
                cuda_crops = [
                    (fitted[i] * 255.0).floor().clamp(0, 255) for i in range(n_total)
                ]
            with span(N.BACKEND_FORWARD, units=n_total, gpu=True):
                all_probs = model.backend.predict_batch_cuda(
                    cuda_crops, input_is_bgr=False
                )
        else:
            all_probs = []
    else:
        from .crops import apply_fit_batch_for_model, extract_classifier_crops_batch_np

        # HWC uint8 BGR crops straight from the warp -- no float32 tensor round
        # trip (it was exactly value-preserving, so removing it is
        # byte-identical; see NumpyCropBatch).
        with span(N.CROP_EXTRACT):
            batch = extract_classifier_crops_batch_np(
                frames, obb_results, geometry, headtail_by_frame=headtail_by_frame
            )
        if batch.crops:
            np_crops: list[np.ndarray] = apply_fit_batch_for_model(
                batch.crops, model.input_size, policy
            )
            with span(N.BACKEND_FORWARD, units=len(np_crops), gpu=True):
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
