from collections import OrderedDict

import numpy as np
import torch

from hydra_suite.core.individual.identity.calibration import (
    expected_calibration_error,
    fit_temperature,
    model_weight_signature,
)


def _overconfident_logits(n=2000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    logits = rng.normal(0, 1, size=(n, k))
    # make the true class win but inflate magnitude → overconfident
    logits[np.arange(n), labels] += 2.5
    logits *= 3.0
    return logits.astype(np.float64), labels.astype(np.int64)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_fit_temperature_reduces_ece_on_overconfident_set():
    logits, labels = _overconfident_logits()
    ece_before = expected_calibration_error(_softmax(logits), labels)
    t = fit_temperature(logits, labels)
    assert t > 1.0  # overconfident ⇒ temperature > 1 softens
    ece_after = expected_calibration_error(_softmax(logits / t), labels)
    assert ece_after < ece_before


def test_fit_temperature_clamped_range():
    logits, labels = _overconfident_logits()
    t = fit_temperature(logits, labels)
    assert 0.1 <= t <= 10.0


def test_ece_zero_for_perfectly_calibrated():
    # one-hot-ish perfectly-confident-and-correct ⇒ ECE ~ 0
    n, k = 500, 3
    labels = np.tile(np.arange(k), n // k + 1)[:n]
    probs = np.full((n, k), 1e-6)
    probs[np.arange(n), labels] = 1.0
    probs /= probs.sum(1, keepdims=True)
    assert expected_calibration_error(probs, labels) < 1e-3


def test_weight_signature_deterministic_and_sensitive():
    sd1 = OrderedDict({"w": torch.ones(3, 3), "b": torch.zeros(3)})
    sd2 = OrderedDict({"w": torch.ones(3, 3), "b": torch.zeros(3)})
    sd3 = OrderedDict({"w": torch.ones(3, 3), "b": torch.ones(3)})
    assert model_weight_signature(sd1) == model_weight_signature(sd2)
    assert model_weight_signature(sd1) != model_weight_signature(sd3)
    assert (
        isinstance(model_weight_signature(sd1), str)
        and len(model_weight_signature(sd1)) >= 8
    )
