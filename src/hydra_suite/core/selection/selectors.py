"""Candidate selectors shared by the labeling kits' batch builders.

Each selector answers one question about an unlabeled pool:

* :class:`UncertaintySelector` — where is the model least sure?
* :class:`ErrorNeighborSelector` — what looks like a known model failure?
  Human-verified samples the model gets wrong are *known* failures; unlabeled
  samples that look like them are the most valuable ones to label next,
  whether or not the model happens to be uncertain about them (a
  confidently-wrong model is not uncertain).  Neighbours are taken round-robin
  across errors, worst first, so every failure is represented.
* :class:`RareClusterSelector` — which clusters are small or under-labeled?
  Images are sampled with weight ``cluster_size ** -alpha * (1 - coverage)``
  without replacement, so a small cluster is far likelier than a big one and
  stops being favoured once it is labeled.  ``alpha = 0`` is plain
  proportional sampling, ``alpha = 1`` gives every cluster the same expected
  share.
* :class:`RepresentativeSelector` — which dense clusters are under-labeled?
* :class:`AuditSelector` — random (and optionally targeted) spot checks.

All randomness comes from an explicit ``np.random.Generator`` so a batch is
reproducible for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Cap on the (errors x candidates) similarity block held in memory at once.
_MAX_BLOCK_ELEMS = 40_000_000


def _norm(label: object) -> str:
    return str(label if label is not None else "").strip()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def top_scores(
    scores: np.ndarray,
    n_samples: int,
    mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Indices of the ``n_samples`` highest finite scores inside ``mask``.

    NaN / -inf scores (no signal) are never selected.  Ties keep index order,
    or are broken at random with ``rng`` (so equal scores don't all come from
    the start of the dataset, e.g. consecutive video frames).
    """
    s = np.asarray(scores, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool) & np.isfinite(s)
    k = min(int(n_samples), int(valid.sum()))
    if k <= 0:
        return np.array([], dtype=int)
    cand = np.where(valid)[0]
    if rng is None:
        order = np.argsort(-s[cand], kind="stable")[:k]
    else:
        order = np.lexsort((rng.random(len(cand)), -s[cand]))[:k]
    return cand[order]


class UncertaintySelector:
    """Select samples with highest uncertainty."""

    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        """Entropy of each row of ``(N, C)`` probabilities."""
        probs = np.clip(probs, 1e-10, 1.0)
        return -(probs * np.log(probs)).sum(axis=1)

    @staticmethod
    def margin(probs: np.ndarray) -> np.ndarray:
        """Top-1 minus top-2 probability (smaller = more uncertain)."""
        sorted_probs = np.sort(probs, axis=1)
        return sorted_probs[:, -1] - sorted_probs[:, -2]

    def select(
        self,
        probs: np.ndarray,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        method: str = "entropy",
    ) -> np.ndarray:
        """Select the most uncertain rows of class probabilities."""
        if method == "entropy":
            uncertainty = self.entropy(probs)
        elif method == "margin":
            uncertainty = -self.margin(probs)  # Negative so higher is more uncertain
        else:
            raise ValueError(f"Unknown method: {method}")

        uncertainty = np.asarray(uncertainty, dtype=np.float64)
        uncertainty[~np.asarray(unlabeled_mask, dtype=bool)] = -np.inf

        # Never hand back masked (already-labeled) rows when asked for more
        # samples than remain.
        n_samples = min(int(n_samples), int(np.count_nonzero(unlabeled_mask)))
        if n_samples <= 0:
            return np.array([], dtype=int)
        return np.argsort(uncertainty)[-n_samples:][::-1]

    def select_scores(
        self, scores: np.ndarray, n_samples: int, available_mask: np.ndarray
    ) -> np.ndarray:
        """Select by a precomputed per-sample uncertainty (higher = less sure).

        Samples without a score (NaN) are skipped rather than ranked.
        """
        return top_scores(scores, n_samples, available_mask)


class RepresentativeSelector:
    """Select samples from dense, under-labeled clusters."""

    def select(
        self,
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        label_coverage: Dict[int, float],
        cluster_densities: np.ndarray,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Random members of the densest, least-labeled clusters.

        Args:
            embeddings: (N, D)
            cluster_assignments: (N,) cluster IDs
            label_coverage: cluster_id -> fraction labeled
            cluster_densities: (K,) density per cluster (lower = denser)
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask
            rng: random generator for the within-cluster draw
        """
        rng = _rng(rng)
        ids = np.asarray(cluster_assignments)
        avail = np.asarray(unlabeled_mask, dtype=bool)
        n_clusters = len(cluster_densities)
        cluster_scores = np.zeros(n_clusters)

        for cluster_id in range(n_clusters):
            coverage = label_coverage.get(cluster_id, 0.0)
            density = cluster_densities[cluster_id]
            # Higher score = denser (smaller kNN distance) + less labeled
            if density > 0:
                cluster_scores[cluster_id] = (1.0 - coverage) / (density + 1e-6)

        selected: List[int] = []
        for cluster_id in np.argsort(-cluster_scores, kind="stable"):
            if len(selected) >= n_samples:
                break
            indices = np.where((ids == cluster_id) & avail)[0]
            if len(indices) == 0:
                continue
            take = min(len(indices), n_samples - len(selected))
            selected.extend(int(i) for i in rng.choice(indices, take, replace=False))

        return np.asarray(selected[:n_samples], dtype=int)


class AuditSelector:
    """Select audit samples for quality assurance."""

    def select(
        self,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
        cluster_disagreements: Optional[Dict[int, float]] = None,
        audit_fraction_random: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Random audits plus, when disagreements are known, targeted ones.

        Without ``cluster_disagreements`` every audit is random, so the whole
        budget is used.
        """
        rng = _rng(rng)
        avail = np.asarray(unlabeled_mask, dtype=bool)
        targeted = cluster_assignments is not None and bool(cluster_disagreements)
        n_random = int(n_samples * audit_fraction_random) if targeted else n_samples
        n_targeted = n_samples - n_random

        selected: List[int] = []
        unlabeled_indices = np.where(avail)[0]
        if len(unlabeled_indices) > 0 and n_random > 0:
            picked = rng.choice(
                unlabeled_indices, min(n_random, len(unlabeled_indices)), replace=False
            )
            selected.extend(int(i) for i in picked)

        if targeted and n_targeted > 0:
            taken = set(selected)
            ids = np.asarray(cluster_assignments)
            ranked = sorted(
                cluster_disagreements.items(), key=lambda x: x[1], reverse=True
            )
            for cluster_id, _ in ranked:
                if len(selected) >= n_samples:
                    break
                indices = [
                    int(i)
                    for i in np.where((ids == cluster_id) & avail)[0]
                    if int(i) not in taken
                ]
                if not indices:
                    continue
                take = min(len(indices), n_samples - len(selected))
                picked = rng.choice(indices, take, replace=False)
                selected.extend(int(i) for i in picked)
                taken.update(int(i) for i in picked)

        return np.asarray(selected[:n_samples], dtype=int)


@dataclass
class ErrorSet:
    """Known model failures on trusted (human-verified) labels."""

    indices: np.ndarray  # (E,) sample indices, most severe first
    severity: np.ndarray  # (E,) how bad each failure is (higher = worse)
    true_labels: List[str] = field(default_factory=list)
    pred_labels: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(len(self.indices))


def errors_from_scores(
    error_scores: np.ndarray, threshold: float = 0.0, max_errors: Optional[int] = None
) -> ErrorSet:
    """Build an :class:`ErrorSet` from a per-sample error magnitude.

    Samples whose (finite) score exceeds ``threshold`` are errors, worst first.
    Used where an error is a distance (e.g. keypoint error) rather than a
    label mismatch.
    """
    s = np.asarray(error_scores, dtype=np.float64)
    idx = np.where(np.isfinite(s) & (s > float(threshold)))[0]
    if idx.size == 0:
        return ErrorSet(np.array([], dtype=int), np.array([]))
    order = np.argsort(-s[idx], kind="stable")
    idx = idx[order]
    if max_errors is not None:
        idx = idx[: int(max_errors)]
    return ErrorSet(idx.astype(int), s[idx])


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
    and the most severe ones get their pick first.
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
        err_pred = list(errors.pred_labels[:n_err])
        use_boost = (
            predicted_labels is not None
            and self.same_pred_boost
            and len(err_pred) == n_err
        )

        emb = l2_normalize(embeddings)
        cand_emb = emb[cand]
        cand_pred = (
            np.array([_norm(predicted_labels[i]) for i in cand], dtype=object)
            if use_boost
            else None
        )

        rounds = int(np.ceil(n_samples / n_err))
        depth = min(len(cand), rounds * 4 + 8)  # headroom for claimed neighbours

        # Per-error ranked candidate lists, computed in memory-bounded blocks.
        block = max(1, _MAX_BLOCK_ELEMS // max(1, len(cand)))
        ranked: List[np.ndarray] = []
        for start in range(0, n_err, block):
            sims = emb[err_idx[start : start + block]] @ cand_emb.T
            if cand_pred is not None:
                for row, pred in enumerate(err_pred[start : start + block]):
                    sims[row, cand_pred == pred] += self.same_pred_boost
            part = np.argpartition(-sims, depth - 1, axis=1)[:, :depth]
            for row in range(sims.shape[0]):
                top = part[row]
                ranked.append(top[np.argsort(-sims[row, top], kind="stable")])

        taken = np.zeros(len(cand), dtype=bool)
        selected: List[int] = []
        source: Dict[int, int] = {}
        for _ in range(depth):
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
        rng = _rng(rng)
        w = self.weights(cluster_assignments, label_coverage)[cand]
        keys = np.log(np.maximum(w, 1e-300)) + rng.gumbel(size=len(cand))
        top = np.argpartition(-keys, k - 1)[:k]
        return cand[top[np.argsort(-keys[top])]]
