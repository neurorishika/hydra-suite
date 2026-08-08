"""Partial-injective assignment substrate for identity resolution.

`solve_unique_assignment` is lifted bit-for-bit from the online decoder's
`_solve_visible_assignment` + `_hungarian_assignment` + `_greedy_assignment`
(`core/individual/identity/online.py`), parameterized by explicit inputs
instead of instance state so it can be shared by both the online and any
future offline/batch consumers.

`fuse_log_evidence` is lifted from the online decoder's `_fuse_evidence`
core (`core/individual/identity/online.py`), plus a no-op-by-default
robustness cap/floor.

`map_cnn_to_catalog` / `map_tag_to_catalog` are lifted from
`EvidenceBuilder._factor_log_prob` + `_build_log_probs_from_posteriors`
(`core/individual/identity/evidence_builder.py`) and
`IdentityCatalog.apriltag_log_prior` (`core/individual/identity/catalog.py`)
respectively, so the one factor->catalog / tag->catalog mapping is shared by
Layer 2 (the evidence stage, via `EvidenceBuilder`) and any future offline
consumer (Phase 5).

This module is Core: it must only depend on numpy/scipy/stdlib (the
`IdentityCatalog` reference in `map_cnn_to_catalog`'s flat-catalog branch is
only used for type-checking and its already-Core `index_of`/`contains` API,
so no cycle is introduced).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog


def fuse_log_evidence(
    log_posterior: np.ndarray,
    evidence_log_probs: np.ndarray,
    *,
    per_frame_cap: float = float("inf"),
    prob_floor: float = 0.0,
) -> np.ndarray:
    """One log-space Bayesian update: fuse one evidence vector into a belief.

    Lifted from the online decoder's `_fuse_evidence` core: for a single
    evidence item, ``log_posterior = log_posterior + evidence_log_probs``,
    then renormalized via ``np.logaddexp.reduce`` to prevent float
    underflow. Returns the *new* normalized log-posterior; does not mutate
    either input array.

    With default arguments (``per_frame_cap=inf``, ``prob_floor=0.0``) this
    is exactly ``renorm(log_posterior + evidence_log_probs)`` — bit-for-bit
    identical to online's `_fuse_evidence`. The caller (online) retains its
    own size-mismatch skip+warning guard and `hit_count` bookkeeping; this
    function is pure and raises on mismatch instead of skipping.

    Args:
        log_posterior: current log-space posterior over the catalog
            (``[unknown, k1, k2, ...]``), already normalized.
        evidence_log_probs: one evidence item's log-probabilities over the
            same catalog, same shape as `log_posterior`.
        per_frame_cap: robustness cap — bounds the log-shift this single
            evidence item can contribute per catalog entry to
            ``[-per_frame_cap, +per_frame_cap]`` before it is added. A
            finite cap prevents one frame's evidence from dominating the
            belief. ``inf`` (default) is an exact no-op: the clip bounds
            are unreachable, so ``evidence_log_probs`` passes through
            unchanged.
        prob_floor: robustness floor — after renormalization, no catalog
            entry's probability is allowed to fall below `prob_floor`
            (water-filled: floored entries are pinned at `prob_floor` and
            the remaining mass is redistributed among the rest, iterated
            to convergence), so no single frame can drive the belief to
            full certainty. ``0.0`` (default) is an exact no-op: the
            branch is skipped entirely (avoiding any exp/log round-trip).

    Returns:
        The new normalized log-posterior (same shape as `log_posterior`).

    Raises:
        ValueError: if `log_posterior` and `evidence_log_probs` shapes
            differ.
    """
    if log_posterior.shape != evidence_log_probs.shape:
        raise ValueError(
            f"log_posterior shape {log_posterior.shape} != "
            f"evidence_log_probs shape {evidence_log_probs.shape}"
        )

    if np.isfinite(per_frame_cap):
        contribution = np.clip(evidence_log_probs, -per_frame_cap, per_frame_cap)
    else:
        contribution = evidence_log_probs

    fused = log_posterior + contribution
    fused = fused - np.logaddexp.reduce(fused)

    if prob_floor > 0.0:
        probs = _apply_prob_floor(np.exp(fused), prob_floor)
        fused = np.log(probs)

    return fused


def _apply_prob_floor(probs: np.ndarray, prob_floor: float) -> np.ndarray:
    """Water-fill `probs` so every entry is >= `prob_floor`, then renormalize.

    A single floor-then-renormalize pass can push a floored entry back
    below `prob_floor` (dividing by a sum > 1 shrinks everything,
    including the just-floored entries). This iterates: entries below the
    floor are pinned to `prob_floor`, and the remaining probability mass is
    redistributed proportionally among the still-free entries, repeating
    until no free entry is below the floor.
    """
    probs = probs.astype(np.float64, copy=True)
    n = probs.size
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n):
        below = (~fixed) & (probs < prob_floor)
        if not below.any():
            break
        probs[below] = prob_floor
        fixed |= below
        remaining = 1.0 - probs[fixed].sum()
        free = ~fixed
        if not free.any():
            break
        free_sum = probs[free].sum()
        if free_sum > 0.0:
            probs[free] *= remaining / free_sum
        else:
            probs[free] = remaining / free.sum()
    return probs / probs.sum()


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


def _factor_log_prob(
    factor_index: int,
    factor_probs: np.ndarray,
    *,
    class_labels_per_factor: list[list[str]],
    factor_class_to_catalog: dict[tuple[int, str], list[int]],
    is_composite: bool,
    catalog_size: int,
    catalog: Optional["IdentityCatalog"],
) -> tuple[np.ndarray, np.ndarray]:
    """Map one factor's posterior to the catalog label space.

    Lifted verbatim from ``EvidenceBuilder._factor_log_prob``. For composite
    catalogs each factor's probabilities are distributed to all composite
    entries that contain that factor's class (via `factor_class_to_catalog`),
    so the sum over factors (in log space) gives the joint probability. For
    flat catalogs the original direct lookup (`catalog.index_of`) is used.
    """
    C = catalog_size
    label_map = []
    if 0 <= factor_index < len(class_labels_per_factor):
        label_map = list(class_labels_per_factor[factor_index] or [])

    floor = 1e-6
    probs = np.full(C, floor, dtype=np.float64)
    observed = np.zeros(C, dtype=bool)
    observed[0] = True

    factor_arr = np.asarray(factor_probs, dtype=np.float64)

    if is_composite:
        for class_idx, cls in enumerate(label_map):
            if class_idx >= len(factor_arr):
                break
            if not cls:
                continue
            prob = max(float(factor_arr[class_idx]), floor)
            for cat_idx in factor_class_to_catalog.get((factor_index, cls), []):
                probs[cat_idx] = prob
                observed[cat_idx] = True
    else:
        for class_idx, label in enumerate(label_map):
            if class_idx >= len(factor_arr):
                break
            if not label:
                continue
            try:
                catalog_idx = catalog.index_of(str(label))
            except KeyError:
                continue
            probs[catalog_idx] = max(float(factor_arr[class_idx]), floor)
            observed[catalog_idx] = True

    probs /= probs.sum()
    return np.log(np.clip(probs, 1e-300, None)), observed


def map_cnn_to_catalog(
    per_factor_probs: Optional[Sequence[np.ndarray]],
    *,
    class_labels_per_factor: list[list[str]],
    factor_class_to_catalog: dict[tuple[int, str], list[int]],
    is_composite: bool,
    catalog_size: int,
    catalog: Optional["IdentityCatalog"] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """CNN per-factor posteriors -> calibrated catalog log-probs.

    Bit-for-bit reproduction of ``EvidenceBuilder._factor_log_prob`` +
    ``EvidenceBuilder._build_log_probs_from_posteriors``: product over
    factors in log-space (``combined += factor_log`` per factor), then
    renormalized via ``np.logaddexp.reduce``.

    ``per_factor_probs`` must already be calibrated (temperature-scaled)
    probability vectors -- calibration is the caller's responsibility (e.g.
    ``EvidenceBuilder._calibrate_posterior``), not this function's; this
    keeps the substrate a pure map/fuse/solve layer with no calibration
    state of its own.

    ``catalog`` is only required for the flat (non-composite) branch, which
    resolves each factor class label directly via ``catalog.index_of``; it
    is unused (and may be omitted) when ``is_composite`` is True.

    Args:
        per_factor_probs: one already-calibrated probability vector per
            class-label factor, aligned to `class_labels_per_factor`'s
            non-empty entries. ``None`` or empty falls back to a uniform
            log-prior over the catalog with no observed mask.
        class_labels_per_factor: list of class label lists, one per raw
            posterior factor (may contain empty ``[]`` gap entries).
        factor_class_to_catalog: ``(factor_index, class_name) ->
            [catalog_indices]`` lookup for the composite branch (index
            compacted to the non-empty factors, gap-skipping; see
            `EvidenceBuilder.__init__` for construction semantics).
        is_composite: whether more than one non-empty class-label factor is
            in play (selects the composite-distribution vs. flat-lookup
            branch).
        catalog_size: size of the target catalog (including the leading
            "unknown" entry).
        catalog: the target `IdentityCatalog`, required by the flat branch's
            `index_of` lookup.

    Returns:
        ``(log_probs, observed_mask)`` -- `log_probs` has shape
        ``(catalog_size,)``; `observed_mask` is `None` only in the
        no-posteriors fallback case, else a boolean array of the same shape.
    """
    C = catalog_size
    if not per_factor_probs:
        return np.full(C, -np.log(C), dtype=np.float64), None

    combined = np.zeros(C, dtype=np.float64)
    observed_mask = np.zeros(C, dtype=bool)
    for factor_index, factor_probs in enumerate(per_factor_probs):
        factor_log, factor_observed = _factor_log_prob(
            factor_index,
            factor_probs,
            class_labels_per_factor=class_labels_per_factor,
            factor_class_to_catalog=factor_class_to_catalog,
            is_composite=is_composite,
            catalog_size=C,
            catalog=catalog,
        )
        combined += factor_log
        observed_mask |= factor_observed

    combined -= np.logaddexp.reduce(combined)
    return combined, observed_mask


def map_tag_to_catalog(
    catalog: "IdentityCatalog",
    tag_id: int,
    tag_to_label: dict[int, str],
    floor: float = 1e-4,
) -> np.ndarray:
    """AprilTag observation -> catalog log-prior.

    Delegates to `IdentityCatalog.apriltag_log_prior` -- the one tag->catalog
    mapping, exposed here so Layer 2 (evidence stage) and future offline
    consumers go through the same substrate entry point as
    `map_cnn_to_catalog`.
    """
    return catalog.apriltag_log_prior(tag_id, tag_to_label, floor=floor)
