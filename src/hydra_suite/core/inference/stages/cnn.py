from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CNNConfig
from ..result import CNNDetectionPrediction, CNNFactorPrediction, CNNResult, OBBResult
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
    factor_names = [f"factor_{i}" for i in range(len(meta.class_names))]
    return CNNModel(
        backend=backend,
        input_size=(meta.input_size[0], meta.input_size[1]),
        factor_names=factor_names,
        factor_class_names=[list(cn) for cn in meta.class_names],
    )


def run_cnn(
    crops: torch.Tensor,
    obb_result: OBBResult,
    model: CNNModel,
    config: CNNConfig,
    runtime: RuntimeContext,
) -> CNNResult:
    """Run CNN identity classifier; returns raw pre-calibration probabilities.

    Per Correction 16 / spec audit: temperature and scoring_mode are applied
    at tracking time inside IdentityEvidenceBuilder, NOT here. Cache writes
    receive raw probabilities; calibration changes never invalidate the cache.
    """
    if crops.shape[0] == 0 or obb_result.num_detections == 0:
        return CNNResult(label=config.label, predictions=[])

    resized = _resize(crops, model.input_size)
    np_crops = [
        resized[i].permute(1, 2, 0).cpu().numpy() for i in range(resized.shape[0])
    ]

    all_probs = model.backend.predict_batch(np_crops)

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
        predictions.append(CNNDetectionPrediction(det_index=det_idx, factors=factors))

    return CNNResult(label=config.label, predictions=predictions)


def _resize(crops: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    th, tw = target
    if crops.shape[2] == th and crops.shape[3] == tw:
        return crops
    return F.interpolate(crops, size=(th, tw), mode="bilinear", align_corners=False)
