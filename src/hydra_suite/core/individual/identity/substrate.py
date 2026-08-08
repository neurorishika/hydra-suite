"""Partial-injective assignment substrate for identity resolution.

`solve_unique_assignment` is lifted bit-for-bit from the online decoder's
`_solve_visible_assignment` + `_hungarian_assignment` + `_greedy_assignment`
(`core/individual/identity/online.py`), parameterized by explicit inputs
instead of instance state so it can be shared by both the online and any
future offline/batch consumers.

This module is Core: it must only depend on numpy/scipy/stdlib.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def solve_unique_assignment(
    posterior_probs: Sequence[np.ndarray],
    num_known: int,
    display_threshold: float,
    *,
    use_scipy: bool = True,
) -> list[Optional[int]]:
    """Solve the partial injective assignment of slots to known identities.

    Args:
        posterior_probs: per-slot full catalog posterior vectors, each
            ``[unknown, k1, k2, ...]`` (already normalized), indexed by
            slot position. Length ``N`` (number of visible slots).
        num_known: number of known identities ``K`` in the catalog.
        display_threshold: minimum posterior probability required for a
            slot to be assigned a known identity (post-uniqueness gate).
        use_scipy: if True, use the Hungarian algorithm (``scipy``) with
            dummy unassigned columns so slots can remain unassigned; falls
            back to greedy argmax if scipy is unavailable. If False, uses
            greedy directly.

    Returns:
        A list aligned to ``posterior_probs`` of per-slot 1-based known
        catalog indices, or ``None`` where the slot is unassigned / below
        threshold.
    """
    if not posterior_probs:
        return []
    if use_scipy:
        try:
            from scipy.optimize import linear_sum_assignment

            return _hungarian_assignment(
                posterior_probs, num_known, display_threshold, linear_sum_assignment
            )
        except ImportError:
            return _greedy_assignment(posterior_probs, display_threshold)
    return _greedy_assignment(posterior_probs, display_threshold)


def _hungarian_assignment(
    posterior_probs: Sequence[np.ndarray],
    num_known: int,
    display_threshold: float,
    linear_sum_assignment,
) -> list[Optional[int]]:
    """Hungarian-based uniqueness-constrained assignment."""
    N = len(posterior_probs)
    K = num_known

    # N x (K + N) cost matrix: K identity columns + N dummy columns
    cost = np.zeros((N, K + N), dtype=np.float64)
    for i in range(N):
        probs = posterior_probs[i]
        # Known identity columns: cost = -log(prob)
        for j in range(K):
            cost[i, j] = -np.log(max(probs[j + 1], 1e-300))  # j+1 skips unknown
        # Dummy columns: cost = -log(unknown_prob)
        unknown_cost = -np.log(max(probs[0], 1e-300))
        for j in range(N):
            cost[i, K + j] = unknown_cost

    rows, cols = linear_sum_assignment(cost)
    result: list[Optional[int]] = [None] * N
    for r, c in zip(rows, cols):
        if c < K:
            probs = posterior_probs[r]
            label_idx = c + 1  # skip unknown at 0
            if float(probs[label_idx]) >= display_threshold:
                result[r] = label_idx
            else:
                result[r] = None
        else:
            result[r] = None  # unassigned
    return result


def _greedy_assignment(
    posterior_probs: Sequence[np.ndarray],
    display_threshold: float,
) -> list[Optional[int]]:
    """Greedy argmax fallback (no uniqueness guarantee among low-confidence cases)."""
    result: list[Optional[int]] = []
    used: set[int] = set()
    for probs in posterior_probs:
        known_probs = probs[1:]
        best_k = int(np.argmax(known_probs))  # 0-based over knowns
        best_idx = best_k + 1  # catalog index
        best_conf = float(probs[best_idx])
        if best_conf >= display_threshold and best_idx not in used:
            result.append(best_idx)
            used.add(best_idx)
        else:
            result.append(None)
    return result
