"""Fit temperature-scaling calibration from a validation loader (Training layer).

Imports the calibration math from Core (allowed: Training -> Core). Produces one
temperature per factor plus ECE before/after and a model-weight signature, for
persistence into the model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hydra_suite.core.individual.identity.calibration import (
    expected_calibration_error,
    fit_temperature,
    model_weight_signature,
)


@dataclass
class CalibrationResult:
    temperatures: list[float]
    signature: str
    ece_before: list[float]
    ece_after: list[float]


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def _collect(model, val_loader, device, split_logits, num_factors):
    model.eval()
    per_factor_logits: list[list[np.ndarray]] = [[] for _ in range(num_factors)]
    per_factor_labels: list[list[np.ndarray]] = [[] for _ in range(num_factors)]
    for batch in val_loader:
        xs, ys = batch[0].to(device), batch[1]
        out = model(xs)
        parts = split_logits(out) if split_logits is not None else [out]
        for k in range(num_factors):
            per_factor_logits[k].append(parts[k].detach().cpu().numpy())
            yk = ys if ys.ndim == 1 else ys[:, k]
            per_factor_labels[k].append(yk.detach().cpu().numpy())
    return per_factor_logits, per_factor_labels


def fit_calibration_from_val(
    model, val_loader, device: str, *, split_logits=None, num_factors: int = 1
) -> CalibrationResult:
    logits_by_f, labels_by_f = _collect(
        model, val_loader, device, split_logits, num_factors
    )
    temps, ece_b, ece_a = [], [], []
    for k in range(num_factors):
        logits = np.concatenate(logits_by_f[k], axis=0)
        labels = np.concatenate(labels_by_f[k], axis=0)
        ece_b.append(expected_calibration_error(_softmax_np(logits), labels))
        t = fit_temperature(logits, labels)
        temps.append(t)
        ece_a.append(expected_calibration_error(_softmax_np(logits / t), labels))
    sig = model_weight_signature(model.state_dict())
    return CalibrationResult(
        temperatures=temps, signature=sig, ece_before=ece_b, ece_after=ece_a
    )
