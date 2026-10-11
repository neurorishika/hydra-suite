"""Error-driven and rarity-driven candidate selection for ClassKit active learning.

Two signals the uncertainty / diversity / representative slots do not capture:

* **Where the model is wrong.**  Human-verified labels that disagree with the
  model's prediction are *known* failures.  Unlabeled images that look like those
  failures are the most valuable ones to label next, whether or not the model
  happens to be uncertain about them (a confidently-wrong model is not uncertain).
  :class:`ErrorNeighborSelector` ranks unlabeled images by similarity to each
  error and takes them round-robin across errors, worst (most confidently wrong)
  first, so every failure is represented rather than one dominating the batch.

* **Which clusters are rare.**  Picking per cluster, or by dense/low-coverage
  clusters, keeps feeding big clusters while small ones stay unlabeled.
  :class:`RareClusterSelector` samples images with weight
  ``cluster_size ** -alpha * (1 - label_coverage)`` without replacement, so a
  small cluster is far likelier than a big one, and stops being favoured once it
  is labeled.  ``alpha = 0`` is plain proportional sampling, ``alpha = 1``
  gives every cluster the same expected share of the slot.

The generic selectors live in :mod:`hydra_suite.core.selection.selectors`
(shared with PoseKit); this module keeps the classification-specific error
definition (a trusted label that disagrees with the prediction) and re-exports
the rest so existing imports keep working.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Re-exported: these moved to the shared core.
from hydra_suite.core.selection.selectors import (  # noqa: F401
    ErrorNeighborResult,
    ErrorNeighborSelector,
    ErrorSet,
    RareClusterSelector,
    l2_normalize,
)

_l2_normalize = l2_normalize  # old private name

# Predictions that mean "the model abstained", not "the model said X".
_ABSTAIN = {"", "unknown", "n/a", "none"}


def _norm(label: object) -> str:
    return str(label if label is not None else "").strip()


def find_prediction_errors(
    image_labels: Sequence[Optional[str]],
    predicted_labels: Sequence[Optional[str]],
    prediction_confidence: Optional[Sequence[float]] = None,
    trusted_mask: Optional[np.ndarray] = None,
) -> ErrorSet:
    """Return the trusted-labeled samples the model gets wrong, worst first.

    A sample is an error when it carries a trusted label, the model made a real
    prediction (not an abstention such as ``"unknown"``), and the two differ.
    Severity is the model's confidence in the wrong answer, so a confidently
    wrong prediction outranks a hesitant one.  ``trusted_mask`` should exclude
    unverified machine labels (they would count the model's own guesses as
    ground truth).
    """
    n = len(image_labels)
    if len(predicted_labels) != n:
        raise ValueError("image_labels and predicted_labels must have equal length")
    conf = (
        np.ones(n, dtype=np.float64)
        if prediction_confidence is None
        else np.nan_to_num(np.asarray(prediction_confidence, dtype=np.float64), nan=0.0)
    )
    trusted = (
        np.ones(n, dtype=bool) if trusted_mask is None else np.asarray(trusted_mask)
    )
    idx: List[int] = []
    for i in range(n):
        truth = _norm(image_labels[i])
        pred = _norm(predicted_labels[i])
        if not trusted[i] or not truth or pred.lower() in _ABSTAIN:
            continue
        if truth != pred:
            idx.append(i)
    if not idx:
        return ErrorSet(np.array([], dtype=int), np.array([]), [], [])
    arr = np.asarray(idx, dtype=int)
    sev = conf[arr]
    order = np.argsort(-sev, kind="stable")
    arr, sev = arr[order], sev[order]
    return ErrorSet(
        arr,
        sev,
        [_norm(image_labels[i]) for i in arr],
        [_norm(predicted_labels[i]) for i in arr],
    )
