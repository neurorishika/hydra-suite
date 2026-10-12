from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.classkit.core.al.acquisition import BatchAcquisition, BatchConfig
from hydra_suite.classkit.core.al.error_mining import (
    ErrorNeighborSelector,
    RareClusterSelector,
    find_prediction_errors,
)


def _blobs(sizes, dim=8, spread=0.05, seed=0):
    """Well-separated clusters; returns (embeddings, cluster_ids)."""
    rng = np.random.default_rng(seed)
    embs, ids = [], []
    for c, n in enumerate(sizes):
        centre = np.zeros(dim)
        centre[c % dim] = 1.0
        embs.append(centre + spread * rng.standard_normal((n, dim)))
        ids += [c] * n
    return np.vstack(embs).astype(np.float32), np.asarray(ids)


# -- find_prediction_errors --------------------------------------------------


def test_errors_need_trusted_label_real_prediction_and_disagreement():
    labels = ["a", "a", "b", None, "a", "b"]
    preds = ["a", "b", "a", "a", "unknown", "a"]
    conf = [0.9, 0.7, 0.95, 0.9, 0.99, 0.2]
    trusted = np.array([True, True, True, True, True, False])
    e = find_prediction_errors(labels, preds, conf, trusted)
    assert e.indices.tolist() == [2, 1]  # most confidently wrong first
    assert e.true_labels == ["b", "a"] and e.pred_labels == ["a", "b"]


def test_no_errors_is_empty():
    e = find_prediction_errors(["a", "b"], ["a", "b"])
    assert len(e) == 0


# -- ErrorNeighborSelector -----------------------------------------------------


def test_error_neighbors_come_from_the_errors_cluster():
    emb, ids = _blobs([40, 40, 40])
    n = len(ids)
    labels = [None] * n
    preds = ["x"] * n
    # one labeled sample in cluster 1 that the model gets wrong
    err = int(np.where(ids == 1)[0][0])
    labels[err], preds[err] = "y", "x"
    errors = find_prediction_errors(labels, preds, np.ones(n))
    avail = np.array([lbl is None for lbl in labels])
    res = ErrorNeighborSelector().select(emb, errors, avail, 10, preds)
    assert len(res.indices) == 10
    assert set(ids[res.indices]) == {1}
    assert err not in res.indices
    assert set(res.source_error.values()) == {err}


def test_round_robin_gives_every_error_its_share():
    emb, ids = _blobs([40, 40, 40])
    n = len(ids)
    labels, preds = [None] * n, ["x"] * n
    errs = [int(np.where(ids == c)[0][0]) for c in (0, 1, 2)]
    for e in errs:
        labels[e] = "y"
    errors = find_prediction_errors(labels, preds, np.ones(n))
    avail = np.array([lbl is None for lbl in labels])
    res = ErrorNeighborSelector().select(emb, errors, avail, 9, preds)
    counts = np.bincount(ids[res.indices], minlength=3)
    assert counts.tolist() == [3, 3, 3]


def test_most_severe_error_picks_first_when_slots_are_scarce():
    emb, ids = _blobs([40, 40, 40])
    n = len(ids)
    labels, preds = [None] * n, ["x"] * n
    e0, e1 = int(np.where(ids == 0)[0][0]), int(np.where(ids == 1)[0][0])
    labels[e0] = labels[e1] = "y"
    conf = np.full(n, 0.5)
    conf[e1] = 0.99  # confidently wrong
    errors = find_prediction_errors(labels, preds, conf)
    avail = np.array([lbl is None for lbl in labels])
    res = ErrorNeighborSelector().select(emb, errors, avail, 1, preds)
    assert ids[res.indices[0]] == 1


def test_same_wrong_prediction_is_preferred_among_equally_similar():
    emb = np.zeros((6, 4), dtype=np.float32)
    emb[:, 0] = 1.0  # identical embeddings: only the boost can separate them
    labels = ["y", None, None, None, None, None]
    preds = ["x", "x", "z", "z", "x", "z"]
    errors = find_prediction_errors(labels, preds, np.ones(6))
    avail = np.array([False, True, True, True, True, True])
    res = ErrorNeighborSelector(same_pred_boost=0.1).select(
        emb, errors, avail, 2, preds
    )
    assert set(res.indices.tolist()) == {1, 4}


def test_no_errors_or_no_candidates_returns_empty():
    emb, ids = _blobs([5, 5])
    empty = find_prediction_errors(["a"] * 10, ["a"] * 10)
    assert (
        ErrorNeighborSelector().select(emb, empty, np.ones(10, bool), 3).indices.size
        == 0
    )


# -- RareClusterSelector -------------------------------------------------------


def _share_in_small(alpha, picks=60, trials=40):
    ids = np.array([0] * 900 + [1] * 30)  # cluster 1 is 3% of the data
    avail = np.ones(len(ids), bool)
    sel = RareClusterSelector(alpha=alpha)
    rng = np.random.default_rng(0)
    frac = []
    for _ in range(trials):
        idx = sel.select(ids, avail, picks, rng=rng)
        frac.append(np.mean(ids[idx] == 1))
    return float(np.mean(frac))


def test_alpha_controls_how_strongly_small_clusters_are_favoured():
    prop, mid, equal = (_share_in_small(a) for a in (0.0, 0.75, 1.0))
    assert prop < 0.06  # ~3%: proportional
    assert equal > 0.30  # ~ capped by the 30 images in the small cluster
    assert prop < mid < equal


def test_selected_are_unique_and_only_from_available():
    ids = np.array([0] * 50 + [1] * 10)
    avail = np.ones(60, bool)
    avail[:20] = False
    idx = RareClusterSelector().select(ids, avail, 30, rng=np.random.default_rng(1))
    assert len(set(idx.tolist())) == len(idx) == 30
    assert avail[idx].all()


def test_coverage_deprioritises_already_labeled_clusters():
    ids = np.array([0] * 40 + [1] * 40)
    avail = np.ones(80, bool)
    sel = RareClusterSelector(alpha=0.0)
    rng = np.random.default_rng(2)
    picks = np.concatenate(
        [sel.select(ids, avail, 20, {0: 0.9, 1: 0.0}, rng=rng) for _ in range(30)]
    )
    assert np.mean(ids[picks] == 1) > 0.8


def test_noise_points_are_treated_as_rare():
    ids = np.array([0] * 500 + [1] * 30 + [-1] * 20)
    w = RareClusterSelector(alpha=1.0).weights(ids)
    assert w[-1] > w[0]  # noise outweighs the big cluster ...
    assert w[-1] == pytest.approx(w[520])  # ... and matches the smallest real one


# -- BatchAcquisition ------------------------------------------------------------


def _problem(seed=0):
    emb, ids = _blobs([300, 200, 20], seed=seed)
    n = len(ids)
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(3), size=n)
    labels = [None] * n
    preds = [str(int(p)) for p in probs.argmax(1)]
    # verified labels; the model is wrong on a few in the rare cluster 2
    for i in rng.choice(np.where(ids < 2)[0], 30, replace=False):
        labels[i], preds[i] = preds[i], preds[i]
    wrong = list(np.where(ids == 2)[0][:3])
    for i in wrong:
        labels[i], preds[i] = "truth", "oops"
    unlabeled = np.array([lbl is None for lbl in labels])
    return emb, ids, probs, labels, preds, unlabeled, wrong


def test_batch_is_full_unique_unlabeled_and_has_all_slots():
    emb, ids, probs, labels, preds, unl, _ = _problem()
    acq = BatchAcquisition(BatchConfig(batch_size=60), seed=0)
    sel, bd = acq.select_batch(
        emb,
        probs,
        unl,
        cluster_assignments=ids,
        image_labels=labels,
        predicted_labels=preds,
        prediction_confidence=np.ones(len(ids)),
    )
    assert len(sel) == 60 and len(set(sel.tolist())) == 60
    assert unl[sel].all()
    for slot in ("error_neighbors", "uncertainty", "rare_cluster", "diversity"):
        assert bd.get(slot), slot
    flat = [i for v in bd.values() for i in v]
    assert len(flat) == len(set(flat))  # slots are mutually exclusive


def test_error_slot_draws_from_the_failing_region():
    emb, ids, probs, labels, preds, unl, wrong = _problem()
    cfg = BatchConfig(
        batch_size=40,
        error_fraction=0.3,
        rare_cluster_fraction=0.0,
        uncertainty_fraction=0.2,
        diversity_fraction=0.2,
        representative_fraction=0.0,
    )
    acq = BatchAcquisition(cfg, seed=0)
    _, bd = acq.select_batch(
        emb,
        probs,
        unl,
        cluster_assignments=ids,
        image_labels=labels,
        predicted_labels=preds,
        prediction_confidence=np.ones(len(ids)),
    )
    # the 3 errors live in the 20-image rare cluster, so its neighbours dominate
    assert np.mean(ids[bd["error_neighbors"]] == 2) > 0.9
    assert acq.last_info["n_errors"] == 3
    assert set(acq.last_info["error_sources"].values()) <= set(int(w) for w in wrong)


def test_without_errors_budget_goes_to_uncertainty_and_batch_stays_full():
    emb, ids, probs, labels, preds, unl, _ = _problem()
    preds_ok = [lbl if lbl is not None else p for lbl, p in zip(labels, preds)]
    acq = BatchAcquisition(BatchConfig(batch_size=50), seed=0)
    sel, bd = acq.select_batch(
        emb,
        probs,
        unl,
        cluster_assignments=ids,
        image_labels=labels,
        predicted_labels=preds_ok,
    )
    assert "error_neighbors" not in bd
    assert len(sel) == 50


def test_old_call_signature_still_works_and_honours_batch_size():
    emb, ids, probs, _, _, unl, _ = _problem()
    sel, bd = BatchAcquisition(BatchConfig(batch_size=30), seed=0).select_batch(
        emb, probs, unl, cluster_assignments=ids
    )
    assert len(sel) == 30 and unl[sel].all()


def test_oversubscribed_fractions_are_scaled_not_overflowed():
    emb, ids, probs, labels, preds, unl, _ = _problem()
    cfg = BatchConfig(batch_size=40, error_fraction=0.6, rare_cluster_fraction=0.5)
    sel, _ = BatchAcquisition(cfg, seed=0).select_batch(
        emb,
        probs,
        unl,
        cluster_assignments=ids,
        image_labels=labels,
        predicted_labels=preds,
    )
    assert len(sel) <= 40


def test_rare_cluster_is_sampled_far_above_its_size_share():
    emb, ids, probs, labels, preds, unl, _ = _problem()
    cfg = BatchConfig(
        batch_size=40,
        error_fraction=0.0,
        rare_cluster_fraction=0.5,
        rare_alpha=1.0,
        uncertainty_fraction=0.0,
        diversity_fraction=0.0,
        representative_fraction=0.0,
    )
    _, bd = BatchAcquisition(cfg, seed=0).select_batch(
        emb, probs, unl, cluster_assignments=ids
    )
    share = np.mean(ids[bd["rare_cluster"]] == 2)
    assert share > 0.25  # cluster 2 is only ~4% of the data
