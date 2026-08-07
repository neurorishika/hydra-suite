"""Identity posterior calibration.

Identity Phase 0: wraps temperature-scaling and Platt-scaling calibration for
CNN classifier outputs to produce calibrated log-posterior probabilities.

Calibration models are identified by a content-based ``calibration_signature``
so that evidence cached from different calibration runs can be distinguished
reliably when replaying or comparing runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CalibrationModel:
    """Temperature-scaling calibration for a single-head classifier.

    ``temperature > 1`` softens the posterior (entropy increases).
    ``temperature < 1`` sharpens it (entropy decreases).
    ``temperature == 1.0`` is a no-op identity calibration.

    Parameters
    ----------
    temperature:
        Scaling factor applied to raw logits before log-softmax.
    label_map:
        Optional list of label strings aligned with the output logits.
        Used to map model output indices to catalog label names when building
        ``IdentityEvidence`` objects.
    """

    temperature: float = 1.0
    label_map: Optional[list[str]] = None

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling; return calibrated log-softmax probabilities.

        Parameters
        ----------
        logits:
            Shape ``(..., C)`` raw model logits (pre-softmax).

        Returns
        -------
        log_probs:
            Shape ``(..., C)`` calibrated log-softmax probabilities.
        """
        scaled = logits / max(self.temperature, 1e-6)
        # Numerically stable log-softmax
        max_vals = scaled.max(axis=-1, keepdims=True)
        shifted = scaled - max_vals
        log_sum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-300)
        return shifted - log_sum

    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate from raw softmax probabilities.

        Converts probabilities to log-space, applies temperature scaling, and
        renormalises.

        Parameters
        ----------
        probs:
            Shape ``(..., C)`` softmax probabilities in ``[0, 1]``.

        Returns
        -------
        log_probs:
            Shape ``(..., C)`` calibrated log-softmax probabilities.
        """
        logits = np.log(np.clip(probs, 1e-300, None))
        return self.calibrate(logits)

    @property
    def signature(self) -> str:
        """Content-addressable hex identifier for this calibration model.

        Two calibration models with identical ``temperature`` and ``label_map``
        will produce the same signature.
        """
        content = json.dumps(
            {"temperature": self.temperature, "label_map": self.label_map},
            sort_keys=True,
        )
        return hashlib.sha1(content.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"CalibrationModel(temperature={self.temperature}, "
            f"n_labels={len(self.label_map) if self.label_map else None})"
        )


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, max_iter: int = 50
) -> float:
    """Fit a single temperature by NLL minimization (Guo et al. 2017), clamped [0.1, 10.0]."""
    import torch
    import torch.nn.functional as F

    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if z.ndim != 2 or z.shape[0] == 0:
        return 1.0
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / t.clamp_min(1e-3), y)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(np.clip(t.item(), 0.1, 10.0))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width-bin ECE over predicted-class confidence."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def model_weight_signature(state_dict) -> str:
    """Deterministic sha1 over sorted (name, contiguous tensor bytes)."""
    h = hashlib.sha1()
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        h.update(name.encode("utf-8"))
        arr = t.detach().cpu().contiguous().numpy()
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
