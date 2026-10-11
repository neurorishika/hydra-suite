"""
Density and diversity utilities for embedding-based selection.

Used for representativeness and cluster-based selection.  kNN queries use
hnswlib when it is installed and a blockwise numpy fallback otherwise (the
fallback never materialises an ``N x N x D`` array).
"""

from __future__ import annotations

import importlib.util
from typing import Optional, Tuple, Union

import numpy as np

# Cap on the elements of one (queries x points) distance block.
_MAX_BLOCK_ELEMS = 20_000_000


def _pairwise_l2(queries: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Euclidean distances ``(len(queries), len(points))`` via dot products."""
    q = np.asarray(queries, dtype=np.float32)
    p = np.asarray(points, dtype=np.float32)
    sq = (q * q).sum(axis=1)[:, None] + (p * p).sum(axis=1)[None, :] - 2.0 * (q @ p.T)
    return np.sqrt(np.maximum(sq, 0.0))


def compute_knn_density(
    embeddings: np.ndarray,
    k: int = 10,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute density proxy as mean distance to k nearest neighbors.

    Lower distance = higher density (more "typical" samples).
    Higher distance = lower density (more "unusual" samples).

    Args:
        embeddings: (N, D) embeddings
        k: Number of neighbors
        return_indices: If True, also return neighbor indices

    Returns:
        densities: (N,) mean kNN distances
        indices (optional): (N, k) neighbor indices
    """
    N, D = embeddings.shape
    emb = np.ascontiguousarray(embeddings.astype(np.float32))
    k_search = min(k + 1, N)  # +1 to exclude self

    if importlib.util.find_spec("hnswlib") is not None:
        import hnswlib

        idx = hnswlib.Index(space="l2", dim=D)
        idx.init_index(max_elements=N, ef_construction=100, M=16)
        idx.add_items(emb)
        idx.set_ef(max(k_search, 50))
        indices, distances = idx.knn_query(emb, k=k_search)
        # hnswlib returns (labels, squared distances); exclude self (col 0)
        distances = np.sqrt(np.maximum(distances[:, 1:], 0.0))
        indices = indices[:, 1:]
    else:
        block = max(1, _MAX_BLOCK_ELEMS // max(1, N))
        idx_parts, dist_parts = [], []
        for start in range(0, N, block):
            d = _pairwise_l2(emb[start : start + block], emb)
            rows = np.arange(d.shape[0])
            d[rows, start + rows] = np.inf  # exclude self
            kk = max(0, k_search - 1)
            if kk == 0:
                idx_parts.append(np.zeros((d.shape[0], 0), dtype=int))
                dist_parts.append(np.zeros((d.shape[0], 0), dtype=np.float32))
                continue
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            pd = np.take_along_axis(d, part, axis=1)
            order = np.argsort(pd, axis=1)
            idx_parts.append(np.take_along_axis(part, order, axis=1))
            dist_parts.append(np.take_along_axis(pd, order, axis=1))
        indices = np.vstack(idx_parts)
        distances = np.vstack(dist_parts)

    densities = (
        distances.mean(axis=1) if distances.shape[1] else np.zeros(N, dtype=np.float32)
    )

    if return_indices:
        return densities, indices
    return densities


def compute_cluster_densities(
    embeddings: np.ndarray, cluster_assignments: np.ndarray, k: int = 10
) -> np.ndarray:
    """
    Compute per-cluster average density.

    Args:
        embeddings: (N, D)
        cluster_assignments: (N,) cluster ids; ``< 0`` (noise) is ignored
        k: kNN k

    Returns:
        cluster_densities: (K,) mean density per cluster
    """
    ids = np.asarray(cluster_assignments)
    n_clusters = int(ids.max()) + 1 if ids.size else 0
    cluster_densities = np.zeros(max(n_clusters, 0))
    if n_clusters <= 0 or len(embeddings) < 2:
        return cluster_densities
    point_densities = compute_knn_density(embeddings, k=k)

    for cluster_id in range(n_clusters):
        mask = ids == cluster_id
        if mask.sum() > 0:
            cluster_densities[cluster_id] = point_densities[mask].mean()

    return cluster_densities


def select_diverse_samples(
    embeddings: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Greedy farthest-first selection (k-center greedy).

    Args:
        embeddings: (N, D) candidate embeddings
        n_samples: Number of samples to select
        seed: Random seed for the first sample (ignored when ``rng`` is given)
        rng: Generator for the first sample
        reference: (M, D) points already covered (e.g. labeled samples).  When
            given, the first pick is the candidate farthest from them, so the
            selection fills the gaps the reference leaves instead of starting
            at random.

    Returns:
        selected_indices: (n_samples,) indices into ``embeddings``, in pick order
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    N = len(emb)
    n_samples = int(n_samples)
    if n_samples <= 0 or N == 0:
        return np.array([], dtype=int)
    has_ref = reference is not None and len(reference) > 0
    if n_samples >= N and not has_ref:
        return np.arange(N)
    n_samples = min(n_samples, N)

    min_dists = np.full(N, np.inf, dtype=np.float32)
    if has_ref:
        ref = np.asarray(reference, dtype=np.float32)
        block = max(1, _MAX_BLOCK_ELEMS // max(1, N))
        for start in range(0, len(ref), block):
            d = _pairwise_l2(ref[start : start + block], emb).min(axis=0)
            min_dists = np.minimum(min_dists, d)
        first = int(np.argmax(min_dists))
    else:
        if rng is None:
            rng = np.random.default_rng(seed)
        first = int(rng.integers(N))

    selected = [first]
    taken = np.zeros(N, dtype=bool)
    taken[first] = True
    min_dists = np.minimum(min_dists, _pairwise_l2(emb[first : first + 1], emb)[0])

    while len(selected) < n_samples:
        scores = np.where(taken, -np.inf, min_dists)
        next_idx = int(np.argmax(scores))
        selected.append(next_idx)
        taken[next_idx] = True
        min_dists = np.minimum(
            min_dists, _pairwise_l2(emb[next_idx : next_idx + 1], emb)[0]
        )

    return np.asarray(selected, dtype=int)


def select_coverage_samples(
    candidates: np.ndarray,
    n_samples: int,
    points: np.ndarray,
    reference: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_points: int = 3000,
    max_elems: int = 40_000_000,
) -> np.ndarray:
    """Greedy coverage (facility location): fill the biggest gaps in *mass*.

    Each pick is the candidate that most reduces the summed distance from
    ``points`` (the data to cover, e.g. every unlabeled sample) to their
    nearest covered sample (``reference`` plus earlier picks).  Unlike
    farthest-first, an isolated outlier only covers itself, so the batch goes
    to where many uncovered samples are, and each pick moves on to the next
    gap.  ``points`` are subsampled to ``max_points`` (fewer for large
    candidate pools, so ``candidates x points`` stays under ``max_elems``).

    Returns indices into ``candidates`` in pick order.
    """
    cand = np.asarray(candidates, dtype=np.float32)
    n_c = len(cand)
    n_samples = min(int(n_samples), n_c)
    if n_samples <= 0:
        return np.array([], dtype=int)
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        pts = cand
    rng = rng if rng is not None else np.random.default_rng(0)
    m = int(min(len(pts), max_points, max(256, max_elems // max(1, n_c))))
    if m < len(pts):
        pts = pts[np.sort(rng.choice(len(pts), m, replace=False))]

    dist = np.empty((n_c, len(pts)), dtype=np.float32)
    block = max(1, max_elems // max(1, len(pts)))
    for start in range(0, n_c, block):
        dist[start : start + block] = _pairwise_l2(cand[start : start + block], pts)

    if reference is not None and len(reference) > 0:
        ref = np.asarray(reference, dtype=np.float32)
        cur = np.full(len(pts), np.inf, dtype=np.float32)
        rblock = max(1, max_elems // max(1, len(pts)))
        for start in range(0, len(ref), rblock):
            cur = np.minimum(cur, _pairwise_l2(ref[start : start + rblock], pts).min(0))
    else:
        # Nothing covered yet: every point is as far as the farthest pair.
        cur = np.full(len(pts), float(dist.max()) if dist.size else 0.0, np.float32)

    taken = np.zeros(n_c, dtype=bool)
    picks = []
    for _ in range(n_samples):
        gain = np.maximum(cur[None, :] - dist, 0.0).sum(axis=1)
        gain[taken] = -1.0
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            # Every sampled point is already covered exactly: no pick helps,
            # so take the first remaining candidate to keep the slot filled.
            best = int(np.flatnonzero(~taken)[0])
        picks.append(best)
        taken[best] = True
        cur = np.minimum(cur, dist[best])
    return np.asarray(picks, dtype=int)
