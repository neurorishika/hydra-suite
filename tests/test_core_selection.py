"""Tests for the shared embedding-selection core (hydra_suite.core.selection)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from hydra_suite.core.selection import density as density_mod
from hydra_suite.core.selection.batch import SlotBatch, slot_budgets
from hydra_suite.core.selection.clustering import (
    SKLearnClusterer,
    auto_cluster_count,
    label_coverage,
)
from hydra_suite.core.selection.density import (
    compute_cluster_densities,
    compute_knn_density,
    select_diverse_samples,
)
from hydra_suite.core.selection.selectors import (
    AuditSelector,
    RepresentativeSelector,
    errors_from_scores,
    top_scores,
)


def _blobs(sizes, dim=8, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(sizes), dim)) * 3
    emb, ids = [], []
    for c, n in enumerate(sizes):
        emb.append(centers[c] + rng.normal(scale=spread, size=(n, dim)))
        ids += [c] * n
    return np.vstack(emb).astype(np.float32), np.asarray(ids)


def test_core_selection_imports_without_torch_or_qt():
    import subprocess

    code = (
        "import sys; import hydra_suite.core.selection.batch, "
        "hydra_suite.core.selection.clustering, hydra_suite.core.selection.density;"
        "print('torch' in sys.modules, 'PySide6' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False False"


def test_slot_budgets_sum_to_batch_and_scale_oversubscription():
    b = slot_budgets(50, {"a": 0.2, "b": 0.3}, remainder="r")
    assert b == {"a": 10, "b": 15, "r": 25}
    b = slot_budgets(10, {"a": 1.0, "b": 1.0, "c": -1}, remainder="r")
    assert sum(b.values()) == 10 and b["c"] == 0 and b["a"] == b["b"] == 5


def test_slot_batch_claims_once_in_order():
    sb = SlotBatch(np.array([True, True, True, False]))
    assert sb.claim("x", [2, 2, 3, 0]) == [2, 0]
    assert sb.claim("y", [0, 1]) == [1]
    assert sb.chosen == [2, 0, 1]
    assert sb.breakdown == {"x": [2, 0], "y": [1]}
    assert sb.remaining == 0


def test_slot_batch_suppresses_near_duplicates():
    emb = np.array([[1, 0], [0.999, 0.01], [0, 1]], dtype=np.float32)
    sb = SlotBatch(np.ones(3, bool), emb, dedup_threshold=0.95)
    sb.claim("x", [0, 1, 2])
    assert sb.chosen == [0, 2] and sb.suppressed == 1


def test_kmeans_is_deterministic_and_finds_blobs():
    emb, ids = _blobs([40, 30, 20])
    a = SKLearnClusterer(n_clusters=3, method="kmeans").fit(emb)
    b = SKLearnClusterer(n_clusters=3, method="kmeans").fit(emb)
    assert np.array_equal(a, b)
    # Each true blob maps to exactly one cluster.
    for c in range(3):
        assert len(np.unique(a[ids == c])) == 1


def test_clusterer_rejects_unknown_method_but_accepts_legacy_values():
    with pytest.raises(ValueError):
        SKLearnClusterer(method="ward")
    # Older ClassKit projects stored the cluster dialog's bool as '0'/'1'.
    for legacy in (True, False, None, "0", "1", "True", "false"):
        assert SKLearnClusterer(method=legacy).method == "kmeans"
    assert SKLearnClusterer(method=" MiniBatch ").method == "minibatch"


def test_label_coverage_and_auto_count():
    ids = np.array([0, 0, 1, 1, 1, -1])
    labeled = np.array([1, 0, 1, 1, 0, 1], bool)
    cov = label_coverage(ids, labeled)
    assert cov == {0: 0.5, 1: pytest.approx(2 / 3)}
    assert auto_cluster_count(8633) == 93
    assert auto_cluster_count(5) == 5 and auto_cluster_count(0) == 8


def test_knn_density_numpy_fallback_matches_hnswlib(monkeypatch):
    emb, _ = _blobs([60, 40], dim=6)
    pytest.importorskip("hnswlib")
    fast = compute_knn_density(emb, k=5)
    monkeypatch.setattr(density_mod.importlib.util, "find_spec", lambda name: None)
    slow = compute_knn_density(emb, k=5)
    np.testing.assert_allclose(fast, slow, rtol=1e-3, atol=1e-4)


def test_cluster_densities_ignore_noise():
    emb, ids = _blobs([10, 10])
    ids = ids.copy()
    ids[0] = -1
    d = compute_cluster_densities(emb, ids)
    assert d.shape == (2,) and (d > 0).all()


def test_diverse_samples_seeded_and_fills_gaps_from_reference():
    emb, ids = _blobs([30, 30, 30])
    a = select_diverse_samples(emb, 5, seed=3)
    b = select_diverse_samples(emb, 5, rng=np.random.default_rng(3))
    assert np.array_equal(a, b) and len(set(a.tolist())) == 5
    # Blobs 0 and 1 are covered by the reference: the first pick is in blob 2.
    ref = np.vstack([emb[ids == 0][:3], emb[ids == 1][:3]])
    picks = select_diverse_samples(emb, 3, reference=ref)
    assert ids[picks[0]] == 2


def test_top_scores_skips_nan_and_masked():
    s = np.array([0.9, np.nan, 0.5, 0.7])
    mask = np.array([False, True, True, True])
    assert top_scores(s, 5, mask).tolist() == [3, 2]


def test_errors_from_scores_threshold_and_order():
    errs = errors_from_scores(np.array([0.01, np.nan, 0.3, 0.1]), threshold=0.05)
    assert errs.indices.tolist() == [2, 3]
    assert errs.severity.tolist() == [0.3, 0.1]


def test_audit_uses_full_budget_without_disagreements():
    avail = np.ones(100, bool)
    picks = AuditSelector().select(10, avail, rng=np.random.default_rng(0))
    assert len(set(picks.tolist())) == 10


def test_representative_is_seeded():
    emb, ids = _blobs([20, 20])
    dens = compute_cluster_densities(emb, ids)
    avail = np.ones(len(ids), bool)
    a = RepresentativeSelector().select(
        emb, ids, {}, dens, 5, avail, rng=np.random.default_rng(1)
    )
    b = RepresentativeSelector().select(
        emb, ids, {}, dens, 5, avail, rng=np.random.default_rng(1)
    )
    assert np.array_equal(a, b)


def test_coverage_prefers_dense_uncovered_mass_over_outliers():
    from hydra_suite.core.selection.density import select_coverage_samples

    emb, ids = _blobs([40, 40, 1], spread=0.05)
    # Blob 0 is labeled; blob 1 is a dense uncovered region; blob 2 is a
    # single outlier (k-center would grab it first).
    ref = emb[ids == 0][:5]
    cand_idx = np.where(ids != 0)[0]
    picks = select_coverage_samples(
        emb[cand_idx], 2, points=emb[ids != 0], reference=ref
    )
    assert ids[cand_idx[picks[0]]] == 1
    assert len(set(picks.tolist())) == 2
    far = select_diverse_samples(emb[cand_idx], 1, reference=ref)
    assert ids[cand_idx[far[0]]] in (1, 2)  # k-center is outlier-prone


def test_coverage_handles_no_reference_and_tiny_pools():
    from hydra_suite.core.selection.density import select_coverage_samples

    emb, _ = _blobs([10, 10])
    assert select_coverage_samples(emb, 0, points=emb).size == 0
    picks = select_coverage_samples(emb, 30, points=emb)
    assert sorted(picks.tolist()) == list(range(20))
    one = select_coverage_samples(emb[:1], 3, points=emb[:0])
    assert one.tolist() == [0]


def test_slot_budgets_small_batches_keep_the_recipe():
    fr = {"e": 0.2, "u": 0.25, "r": 0.1, "d": 0.3, "t": 0.1}
    for n in range(0, 60):
        b = slot_budgets(n, fr, remainder="audit")
        assert sum(b.values()) == n
        assert b["audit"] <= max(1, round(n * 0.05) + 1)
    assert slot_budgets(3, fr, remainder="audit")["audit"] == 0


def test_hdbscan_falls_back_on_tiny_sets():
    pytest.importorskip("sklearn.cluster", minversion=None)
    emb, _ = _blobs([2, 1])
    ids = SKLearnClusterer(n_clusters=5, method="hdbscan").fit(emb)
    assert len(ids) == 3


def test_knn_density_is_deterministic_above_exact_limit(monkeypatch):
    pytest.importorskip("hnswlib")
    monkeypatch.setattr(density_mod, "EXACT_KNN_MAX", 100)
    emb, _ = _blobs([300, 300], dim=16)
    emb = np.vstack([emb, emb[:50]])  # exact duplicates
    a = compute_knn_density(emb, k=5)
    b = compute_knn_density(emb, k=5)
    assert np.array_equal(a, b)
