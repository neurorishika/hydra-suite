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

Everything here is pure numpy (no Qt, no torch) so it is usable from scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Predictions that mean "the model abstained", not "the model said X".
_ABSTAIN = {"", "unknown", "n/a", "none"}

# Cap on the (errors x candidates) similarity block held in memory at once.
_MAX_BLOCK_ELEMS = 40_000_000


def _norm(label: object) -> str:
    return str(label if label is not None else "").strip()


@dataclass
class ErrorSet:
    """Known model failures on trusted (human-verified) labels."""

    indices: np.ndarray  # (E,) sample indices, most severe first
    severity: np.ndarray  # (E,) model confidence in its wrong answer
    true_labels: List[str]
    pred_labels: List[str]

    def __len__(self) -> int:
        return int(len(self.indices))


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


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


@dataclass
class ErrorNeighborResult:
    """Selected indices plus, per index, the error it was picked for."""

    indices: np.ndarray
    source_error: Dict[int, int] = field(default_factory=dict)


class ErrorNeighborSelector:
    """Pick unlabeled images most similar to known model errors.

    Similarity is cosine similarity of the embeddings.  Candidates whose
    *predicted* label equals the error's wrong prediction get ``same_pred_boost``
    added to their similarity: an image that looks like a known failure **and**
    receives the same wrong answer is the model's most probable next mistake.

    Selection is round-robin over errors ordered by severity, each taking its
    best not-yet-claimed neighbour per round, so errors are represented evenly
    and the most confidently wrong ones get their pick first.
    """

    def __init__(self, same_pred_boost: float = 0.05) -> None:
        self.same_pred_boost = float(same_pred_boost)

    def select(
        self,
        embeddings: np.ndarray,
        errors: ErrorSet,
        available_mask: np.ndarray,
        n_samples: int,
        predicted_labels: Optional[Sequence[Optional[str]]] = None,
    ) -> ErrorNeighborResult:
        empty = ErrorNeighborResult(np.array([], dtype=int))
        cand = np.where(np.asarray(available_mask))[0]
        if n_samples <= 0 or len(errors) == 0 or len(cand) == 0:
            return empty
        n_samples = min(int(n_samples), len(cand))

        # More errors than slots: keep the most severe ones (each then gets one).
        n_err = min(len(errors), n_samples)
        err_idx = errors.indices[:n_err]
        err_pred = errors.pred_labels[:n_err]

        emb = _l2_normalize(embeddings)
        cand_emb = emb[cand]
        cand_pred = (
            None
            if predicted_labels is None
            else np.array([_norm(predicted_labels[i]) for i in cand], dtype=object)
        )

        rounds = int(np.ceil(n_samples / n_err))
        depth = min(len(cand), rounds * 4 + 8)  # headroom for claimed neighbours

        # Per-error ranked candidate lists, computed in memory-bounded blocks.
        block = max(1, _MAX_BLOCK_ELEMS // max(1, len(cand)))
        ranked: List[np.ndarray] = []
        for start in range(0, n_err, block):
            sims = emb[err_idx[start : start + block]] @ cand_emb.T
            if cand_pred is not None and self.same_pred_boost:
                for row, pred in enumerate(err_pred[start : start + block]):
                    sims[row, cand_pred == pred] += self.same_pred_boost
            part = np.argpartition(-sims, depth - 1, axis=1)[:, :depth]
            for row in range(sims.shape[0]):
                top = part[row]
                ranked.append(top[np.argsort(-sims[row, top], kind="stable")])

        taken = np.zeros(len(cand), dtype=bool)
        selected: List[int] = []
        source: Dict[int, int] = {}
        for r in range(depth):
            progressed = False
            for e in range(n_err):
                if len(selected) >= n_samples:
                    break
                for pos in ranked[e]:
                    if not taken[pos]:
                        taken[pos] = True
                        gi = int(cand[pos])
                        selected.append(gi)
                        source[gi] = int(err_idx[e])
                        progressed = True
                        break
            if len(selected) >= n_samples or not progressed:
                break
        return ErrorNeighborResult(np.asarray(selected, dtype=int), source)


class RareClusterSelector:
    """Sample unlabeled images with a bias toward small, under-labeled clusters.

    Weight of an image ``i`` in cluster ``c``::

        w_i = size_c ** -alpha * max(1 - coverage_c, coverage_floor)

    and ``n`` images are drawn **without replacement** with probability
    proportional to ``w`` (Gumbel top-k).  Noise points (cluster id ``< 0``)
    are rare by definition, so they take the weight of the smallest real
    cluster.
    """

    def __init__(self, alpha: float = 0.75, coverage_floor: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.coverage_floor = float(coverage_floor)

    def weights(
        self,
        cluster_assignments: np.ndarray,
        label_coverage: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Per-sample sampling weights (unnormalised)."""
        ids = np.asarray(cluster_assignments)
        real = ids[ids >= 0]
        w = np.ones(len(ids), dtype=np.float64)
        if real.size == 0:
            return w
        uniq, counts = np.unique(real, return_counts=True)
        size = dict(zip(uniq.tolist(), counts.tolist()))
        min_size = float(counts.min())
        coverage = label_coverage or {}
        for cid in np.unique(ids):
            mask = ids == cid
            if cid < 0:
                n = min_size
                cov = 0.0
            else:
                n = float(size[int(cid)])
                cov = float(coverage.get(int(cid), 0.0))
            w[mask] = (n**-self.alpha) * max(1.0 - cov, self.coverage_floor)
        return w

    def select(
        self,
        cluster_assignments: np.ndarray,
        available_mask: np.ndarray,
        n_samples: int,
        label_coverage: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        cand = np.where(np.asarray(available_mask))[0]
        k = min(int(n_samples), len(cand))
        if k <= 0:
            return np.array([], dtype=int)
        rng = rng or np.random.default_rng()
        w = self.weights(cluster_assignments, label_coverage)[cand]
        keys = np.log(np.maximum(w, 1e-300)) + rng.gumbel(size=len(cand))
        top = np.argpartition(-keys, k - 1)[:k]
        return cand[top[np.argsort(-keys[top])]]
