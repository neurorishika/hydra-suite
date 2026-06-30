from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import CNNConfig
from ..result import (
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    OBBResult,
)
from ..runtime import RuntimeContext


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

    backend = ClassifierBackend(config.model_path, config.compute_runtime)
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
    from .crops import extract_classifier_crops_batch

    batch = extract_classifier_crops_batch(
        frames, obb_results, model.input_size, aspect_ratio, margin
    )

    n_total = batch.crops.shape[0]
    np_crops: list[np.ndarray] = []
    for i in range(n_total):
        hwc = batch.crops[i].permute(1, 2, 0).cpu().numpy()
        np_crops.append((hwc * 255.0).clip(0, 255).astype(np.uint8))

    all_probs = model.backend.predict_batch(np_crops) if np_crops else []

    results: dict[int, CNNResult] = {}
    prob_offset = 0
    for frame_idx in sorted(batch.obb_by_frame):
        rows = batch.select_frame(frame_idx)
        n = len(rows)
        frame_probs = all_probs[prob_offset : prob_offset + n]
        prob_offset += n
        results[frame_idx] = _assemble_cnn_result(frame_probs, model, config)
    return results
