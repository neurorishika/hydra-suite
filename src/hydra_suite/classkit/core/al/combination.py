"""Combinatorial class balancing for multi-factor labeling schemes.

A scheme with factors A and B has ``|A| x |B|`` possible composite labels, and
a labeled set is rarely uniform over them: some combinations are plentiful,
some rare, some never seen.  A flat classifier cannot even *output* a
combination it never saw, so balancing over its predicted classes cannot find
them.  Instead this works on **per-factor marginals**: an image whose factor-A
marginal peaks at ``a`` and whose factor-B marginal peaks at ``b`` is a likely
``a_b`` even if ``a_b`` has no labeled example (the factors are treated as
independent when forming the joint).

Selection is greedy with diminishing returns: each pick adds its expected
combination mass to the running counts before the next pick, so a batch spreads
over several under-represented combinations rather than piling onto one.

Pure numpy (no Qt / torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_UNKNOWN = "unknown"
_SEP = "_"
_MAX_JOINT_ELEMS = 60_000_000  # cap on (candidates x combinations) floats


def count_labeled_combinations(
    image_labels: Sequence[Optional[str]],
    factor_labels: Sequence[Sequence[str]],
    trusted_mask: Optional[np.ndarray] = None,
) -> Dict[Tuple[str, ...], int]:
    """Count trusted labels per factor-value tuple (undecodable labels ignored)."""
    n_factors = len(factor_labels)
    counts: Dict[Tuple[str, ...], int] = {}
    for i, label in enumerate(image_labels):
        if not label or (trusted_mask is not None and not trusted_mask[i]):
            continue
        parts = tuple(str(label).split(_SEP)) if n_factors > 1 else (str(label),)
        if len(parts) != n_factors:
            continue
        if any(p not in factor_labels[f] for f, p in enumerate(parts)):
            continue
        counts[parts] = counts.get(parts, 0) + 1
    return counts


@dataclass
class CombinationResult:
    indices: np.ndarray
    target_combo: Dict[int, str] = field(default_factory=dict)  # idx -> combo name
    counts: Dict[str, int] = field(default_factory=dict)  # labeled count per combo
    missing: List[str] = field(default_factory=list)  # combos with 0 labels


class CombinationBalanceSelector:
    """Pick images likely to belong to under-represented factor combinations.

    Weight of a combination ``c`` is ``(count_c + 1) ** -power``; an image's
    score is ``sum_c P(c | image) * weight_c`` with ``P`` the product of its
    per-factor marginals.  Combinations containing ``unknown`` are not targets.
    """

    def __init__(self, power: float = 1.0, include_unknown: bool = False) -> None:
        self.power = float(power)
        self.include_unknown = bool(include_unknown)

    def _combos(self, factor_labels: Sequence[Sequence[str]]):
        combos = list(product(*[list(f) for f in factor_labels]))
        if not self.include_unknown:
            combos = [c for c in combos if _UNKNOWN not in c]
        return combos

    def select(
        self,
        factor_probs: Sequence[np.ndarray],
        factor_labels: Sequence[Sequence[str]],
        image_labels: Sequence[Optional[str]],
        available_mask: np.ndarray,
        n_samples: int,
        trusted_mask: Optional[np.ndarray] = None,
    ) -> CombinationResult:
        empty = CombinationResult(np.array([], dtype=int))
        cand = np.where(np.asarray(available_mask))[0]
        k = min(int(n_samples), len(cand))
        if k <= 0 or not factor_labels or len(factor_probs) != len(factor_labels):
            return empty

        combos = self._combos(factor_labels)
        if not combos:
            return empty
        counts = count_labeled_combinations(image_labels, factor_labels, trusted_mask)
        base = np.array([counts.get(c, 0) for c in combos], dtype=np.float64)

        # Candidate restriction keeps the (candidates x combos) block bounded.
        max_cand = max(1, _MAX_JOINT_ELEMS // len(combos))
        if len(cand) > max_cand:
            rng = np.random.default_rng(0)
            cand = np.sort(rng.choice(cand, size=max_cand, replace=False))

        joint = self._joint(
            [np.asarray(p, dtype=np.float32)[cand] for p in factor_probs],
            factor_labels,
            combos,
        )
        if not joint.any():
            return empty

        expected = base.copy()
        taken = np.zeros(len(cand), dtype=bool)
        picked: List[int] = []
        target: Dict[int, str] = {}
        for _ in range(k):
            w = (expected + 1.0) ** (-self.power)
            scores = joint @ w.astype(np.float32)
            scores[taken] = -np.inf
            j = int(np.argmax(scores))
            if not np.isfinite(scores[j]):
                break
            taken[j] = True
            gi = int(cand[j])
            picked.append(gi)
            target[gi] = _SEP.join(combos[int(np.argmax(joint[j] * w))])
            expected += joint[j]  # expected contribution of this pick

        names = [_SEP.join(c) for c in combos]
        return CombinationResult(
            np.asarray(picked, dtype=int),
            target,
            {n: int(b) for n, b in zip(names, base)},
            [n for n, b in zip(names, base) if b == 0],
        )

    @staticmethod
    def _joint(
        probs: List[np.ndarray],
        factor_labels: Sequence[Sequence[str]],
        combos: List[Tuple[str, ...]],
    ) -> np.ndarray:
        """(n, len(combos)) product-of-marginals probability of each combo."""
        index = [{lab: i for i, lab in enumerate(f)} for f in factor_labels]
        n = probs[0].shape[0]
        out = np.ones((n, len(combos)), dtype=np.float32)
        for f, p in enumerate(probs):
            cols = np.array([index[f][c[f]] for c in combos], dtype=int)
            out *= p[:, cols]
        return out


def factor_marginals_from_composite(
    probs: np.ndarray,
    class_names: Sequence[str],
    factor_labels: Sequence[Sequence[str]],
) -> List[np.ndarray]:
    """Per-factor marginals from a flat model over composite class names.

    ``P(factor_f = a) = sum of probs over classes whose f-th part is a``.
    Classes that cannot be split into the scheme's factors are ignored.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n_factors = len(factor_labels)
    out = [np.zeros((probs.shape[0], len(f)), dtype=np.float64) for f in factor_labels]
    index = [{lab: i for i, lab in enumerate(f)} for f in factor_labels]
    for col, name in enumerate(class_names):
        if col >= probs.shape[1]:
            break
        parts = str(name).split(_SEP) if n_factors > 1 else [str(name)]
        if len(parts) != n_factors:
            continue
        if any(p not in index[f] for f, p in enumerate(parts)):
            continue
        for f, p in enumerate(parts):
            out[f][:, index[f][p]] += probs[:, col]
    return [_row_normalize(o) for o in out]


def factor_marginals_from_heads(
    probs: np.ndarray,
    heads: Sequence[dict],
    factor_labels: Sequence[Sequence[str]],
) -> Optional[List[np.ndarray]]:
    """Per-factor marginals from multi-head output (``start``/``end`` slices)."""
    if len(heads) != len(factor_labels):
        return None
    probs = np.asarray(probs, dtype=np.float64)
    out: List[np.ndarray] = []
    for head, labels in zip(heads, factor_labels):
        block = probs[:, int(head["start"]) : int(head["end"])]
        names = list(head.get("class_names") or [])
        m = np.zeros((probs.shape[0], len(labels)), dtype=np.float64)
        pos = {lab: i for i, lab in enumerate(labels)}
        for c, name in enumerate(names):
            if name in pos and c < block.shape[1]:
                m[:, pos[name]] += block[:, c]
        out.append(_row_normalize(m))
    return out


def _row_normalize(m: np.ndarray) -> np.ndarray:
    s = m.sum(axis=1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)
