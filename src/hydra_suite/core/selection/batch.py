"""Slot bookkeeping for mixed active-learning batches.

A batch is split into *slots* (error neighbours, uncertainty, rare clusters,
diversity, …), each with a share of the batch.  Slots run in order against a
shrinking pool, so no sample is picked twice; a slot with no signal leaves its
budget for a later top-up so the requested batch size is still honoured.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from .selectors import l2_normalize


def slot_budgets(
    batch_size: int, fractions: Mapping[str, float], remainder: str
) -> Dict[str, int]:
    """Number of samples per slot.

    Negative fractions count as 0; an over-subscribed recipe (fractions
    summing past 1) is scaled down evenly.  Whatever the recipe leaves goes to
    the ``remainder`` slot.  Counts use largest-remainder rounding, so they
    always sum to ``batch_size`` and small batches keep the recipe's shape
    instead of pushing every rounding loss into the remainder slot.
    """
    batch_size = max(0, int(batch_size))
    fr = {k: max(0.0, float(v)) for k, v in fractions.items()}
    total = sum(fr.values())
    if total > 1.0:
        fr = {k: v / total for k, v in fr.items()}
        total = 1.0
    shares = dict(fr)
    shares[remainder] = shares.get(remainder, 0.0) + max(0.0, 1.0 - total)
    exact = {k: batch_size * v for k, v in shares.items()}
    budgets = {k: int(v) for k, v in exact.items()}
    left = batch_size - sum(budgets.values())
    # Hand the leftovers to the largest fractional parts (ties: recipe order).
    order = sorted(
        shares, key=lambda k: (-(exact[k] - budgets[k]), list(shares).index(k))
    )
    for k in order[: max(0, left)]:
        budgets[k] += 1
    return budgets


class SlotBatch:
    """Claim-once bookkeeping for a batch built slot by slot.

    ``available`` shrinks as slots claim samples; ``chosen`` keeps pick order
    and ``breakdown`` maps each reason to its picks.

    With ``dedup_threshold`` (cosine similarity, needs ``embeddings``), claiming
    a sample also removes every still-available sample at least that similar to
    it, so near-duplicate images never land in the same batch.
    """

    def __init__(
        self,
        available_mask: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        dedup_threshold: Optional[float] = None,
    ) -> None:
        self.available = np.asarray(available_mask, dtype=bool).copy()
        self.chosen: List[int] = []
        self.breakdown: Dict[str, List[int]] = {}
        self.reason_of: Dict[int, str] = {}
        self.suppressed = 0
        self._dedup = None
        if dedup_threshold is not None and embeddings is not None:
            self._dedup = (l2_normalize(embeddings), float(dedup_threshold))

    @property
    def remaining(self) -> int:
        """Samples still available to claim."""
        return int(self.available.sum())

    def __len__(self) -> int:
        return len(self.chosen)

    def claim(self, reason: str, picked: Iterable[int]) -> List[int]:
        """Claim ``picked`` (in order) for ``reason``; returns what was taken.

        Samples already claimed or no longer available are skipped.
        """
        taken: List[int] = []
        for i in np.asarray(list(picked), dtype=int).reshape(-1):
            i = int(i)
            if i < 0 or i >= len(self.available) or not self.available[i]:
                continue
            self.available[i] = False
            self.chosen.append(i)
            self.reason_of[i] = reason
            self.breakdown.setdefault(reason, []).append(i)
            taken.append(i)
            if self._dedup is not None:
                emb, thr = self._dedup
                near = self.available & ((emb @ emb[i]) >= thr)
                n_near = int(near.sum())
                if n_near:
                    self.available[near] = False
                    self.suppressed += n_near
        return taken
