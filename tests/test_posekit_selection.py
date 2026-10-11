"""Tests for PoseKit's Smart Select batch builder (posekit.core.selection)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydra_suite.posekit.core.selection import (
    REASON_AUDIT,
    REASON_DIVERSITY,
    REASON_ERROR,
    REASON_RARE,
    REASON_UNCERTAINTY,
    PoseBatchConfig,
    build_units,
    pose_error_norm,
    pose_uncertainty,
    select_pose_batch,
    unit_any,
    unit_embeddings,
    unit_max,
)


def _blobs(sizes, dim=16, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(sizes), dim)) * 3
    emb, ids = [], []
    for c, n in enumerate(sizes):
        emb.append(centers[c] + rng.normal(scale=spread, size=(n, dim)))
        ids += [c] * n
    return np.vstack(emb).astype(np.float32), np.asarray(ids)


# ── pose signals ─────────────────────────────────────────────────────────────


def test_pose_error_matches_evaluation_metric():
    # 100x100 image, bbox 0.5x0.5 -> scale 50 px.
    gt = [(0.5, 0.5, 2), (0.2, 0.2, 2), (0.9, 0.9, 0)]
    preds = [(55.0, 50.0, 0.9), (20.0, 20.0, 0.8), (0.0, 0.0, 0.1)]
    err = pose_error_norm(gt, preds, 100, 100, bbox=(0.5, 0.5, 0.5, 0.5))
    assert err == pytest.approx((5 / 50 + 0) / 2)


def test_pose_error_counts_missed_keypoints_and_handles_empty():
    gt = [(0.5, 0.5, 2), (0.2, 0.2, 2)]
    preds = [(50.0, 50.0, 0.9), (20.0, 20.0, 0.0)]
    assert pose_error_norm(gt, preds, 100, 100) == pytest.approx(0.5)
    assert pose_error_norm(gt, None, 100, 100) == pytest.approx(1.0)
    assert math.isnan(pose_error_norm([(0, 0, 0)], preds, 100, 100))
    assert math.isnan(pose_error_norm(gt, preds, 0, 100))


def test_pose_uncertainty():
    assert pose_uncertainty([(0, 0, 0.9), (0, 0, 0.5)]) == pytest.approx(0.3)
    assert pose_uncertainty([(0, 0, float("nan"))]) == pytest.approx(1.0)
    assert math.isnan(pose_uncertainty(None))


# ── units ────────────────────────────────────────────────────────────────────


def test_units_group_frames_and_aggregate():
    units = build_units(["b", "a", "b", "c"], 4)
    assert units == [[0, 2], [1], [3]]
    assert build_units(None, 3) == [[0], [1], [2]]
    emb = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    ue = unit_embeddings(emb, units)
    assert np.allclose(np.linalg.norm(ue, axis=1), 1)
    assert unit_any(np.array([0, 0, 1, 0], bool), units).tolist() == [
        True,
        False,
        False,
    ]
    m = unit_max(np.array([0.1, np.nan, 0.4, np.nan]), units)
    assert m[0] == pytest.approx(0.4) and math.isnan(m[1])


# ── batch ────────────────────────────────────────────────────────────────────


def _problem():
    emb, ids = _blobs([200, 150, 12])
    labeled = np.zeros(len(ids), bool)
    labeled[:20] = True  # blob 0 is partly labeled
    cand = ~labeled
    return emb, ids, labeled, cand


def test_batch_size_order_and_no_duplicates():
    emb, ids, labeled, cand = _problem()
    res = select_pose_batch(emb, cand, labeled, ids, PoseBatchConfig(batch_size=40))
    assert len(res.indices) == 40 == len(set(res.indices))
    assert all(cand[i] for i in res.indices)
    assert set(res.reasons) == set(res.indices)


def test_batch_is_deterministic_for_a_seed():
    emb, ids, labeled, cand = _problem()
    a = select_pose_batch(emb, cand, labeled, ids, PoseBatchConfig(batch_size=30))
    b = select_pose_batch(emb, cand, labeled, ids, PoseBatchConfig(batch_size=30))
    assert a.indices == b.indices
    c = select_pose_batch(
        emb, cand, labeled, ids, PoseBatchConfig(batch_size=30, seed=1)
    )
    assert a.indices != c.indices


def test_without_model_signals_budget_goes_to_coverage():
    emb, ids, labeled, cand = _problem()
    res = select_pose_batch(emb, cand, labeled, ids, PoseBatchConfig(batch_size=20))
    assert REASON_ERROR not in res.breakdown
    assert REASON_UNCERTAINTY not in res.breakdown
    assert res.info["topped_up"] > 0
    # The unlabeled blobs (1 and the rare blob 2) are both covered.
    picked_clusters = set(ids[res.indices].tolist())
    assert {1, 2} <= picked_clusters


def test_error_neighbours_target_the_failing_blob():
    emb, ids, labeled, cand = _problem()
    labeled[200:205] = True  # a few labels in blob 1 ...
    cand = ~labeled
    errors = np.full(len(ids), np.nan)
    errors[:20] = 0.01  # blob 0: model fine
    errors[200:205] = 0.4  # ... where the model fails
    cfg = PoseBatchConfig(
        batch_size=20,
        error_fraction=0.5,
        uncertainty_fraction=0,
        rare_fraction=0,
        diversity_fraction=0,
        representative_fraction=0,
    )
    res = select_pose_batch(emb, cand, labeled, ids, cfg, error_scores=errors)
    near = res.breakdown[REASON_ERROR]
    assert len(near) == 10 and set(ids[near].tolist()) == {1}
    assert res.info["n_errors"] == 5
    assert set(res.info["error_sources"].values()) <= set(range(200, 205))


def test_uncertainty_slot_takes_least_confident_and_skips_unknown():
    emb, ids, labeled, cand = _problem()
    unc = np.full(len(ids), np.nan)
    unc[300:310] = np.linspace(0.1, 0.9, 10)
    cfg = PoseBatchConfig(
        batch_size=10,
        error_fraction=0,
        uncertainty_fraction=0.5,
        rare_fraction=0,
        diversity_fraction=0,
        representative_fraction=0,
    )
    res = select_pose_batch(emb, cand, labeled, ids, cfg, uncertainty=unc)
    assert res.breakdown[REASON_UNCERTAINTY] == [309, 308, 307, 306, 305]
    assert res.info["uncertainty"][309] == pytest.approx(0.9)


def test_rare_cluster_slot_prefers_small_cluster():
    emb, ids, labeled, cand = _problem()
    cfg = PoseBatchConfig(
        batch_size=20,
        error_fraction=0,
        uncertainty_fraction=0,
        rare_fraction=0.5,
        rare_alpha=1.0,
        diversity_fraction=0,
        representative_fraction=0,
    )
    res = select_pose_batch(emb, cand, labeled, ids, cfg)
    rare = res.breakdown[REASON_RARE]
    # alpha=1 gives each cluster the same expected share, so the 12-image
    # cluster is heavily over-represented relative to its size.
    assert np.mean(ids[rare] == 2) > 12 / 362 * 3


def test_dedup_and_small_pools():
    emb, ids, labeled, cand = _problem()
    res = select_pose_batch(
        emb, cand, labeled, ids, PoseBatchConfig(batch_size=30, dedup_threshold=0.9)
    )
    sims = emb[res.indices] / np.linalg.norm(emb[res.indices], axis=1, keepdims=True)
    gram = sims @ sims.T
    np.fill_diagonal(gram, 0)
    assert gram.max() < 0.9
    tiny = np.zeros(len(ids), bool)
    tiny[[5, 300]] = True
    res = select_pose_batch(emb, tiny, labeled, ids, PoseBatchConfig(batch_size=10))
    assert sorted(res.indices) == [5, 300]
    none = select_pose_batch(
        emb, np.zeros(len(ids), bool), labeled, ids, PoseBatchConfig()
    )
    assert none.indices == []


def test_mismatched_lengths_raise():
    emb, ids, labeled, cand = _problem()
    with pytest.raises(ValueError):
        select_pose_batch(emb, cand[:-1], labeled, ids)


def test_audit_remainder_fills_batch():
    emb, ids, labeled, cand = _problem()
    cfg = PoseBatchConfig(
        batch_size=10,
        error_fraction=0,
        uncertainty_fraction=0,
        rare_fraction=0,
        diversity_fraction=0,
        representative_fraction=0,
    )
    res = select_pose_batch(emb, cand, labeled, ids, cfg)
    assert res.breakdown == {REASON_AUDIT: res.indices} and len(res.indices) == 10
    assert REASON_DIVERSITY not in res.breakdown


def test_error_neighbours_never_pick_labeled_frames():
    emb, ids, labeled, cand = _problem()
    labeled[200:210] = True
    cand = np.ones(len(ids), bool)  # "All frames": labeled are candidates too
    errors = np.full(len(ids), np.nan)
    errors[200:210] = 0.5
    cfg = PoseBatchConfig(
        batch_size=20,
        error_fraction=0.5,
        uncertainty_fraction=0,
        rare_fraction=0,
        diversity_fraction=0,
        representative_fraction=0,
    )
    res = select_pose_batch(emb, cand, labeled, ids, cfg, error_scores=errors)
    near = res.breakdown[REASON_ERROR]
    assert near and not any(labeled[i] for i in near)
    assert all(res.info["error_sources"][p] != p for p in near)


def test_dedup_top_up_fills_the_batch():
    rng = np.random.default_rng(0)
    groups = rng.normal(size=(30, 16)).astype(np.float32)
    emb = np.repeat(groups, 4, axis=0)  # 30 groups of 4 identical images
    ids = np.repeat(np.arange(30) % 6, 4)
    labeled = np.zeros(len(emb), bool)
    labeled[::4] = True
    res = select_pose_batch(
        emb,
        ~labeled,
        labeled,
        ids,
        PoseBatchConfig(batch_size=20, dedup_threshold=0.99),
    )
    assert len(res.indices) == 20
    assert len({i // 4 for i in res.indices}) == 20


def test_fully_confident_predictions_are_not_model_unsure():
    emb, ids, labeled, cand = _problem()
    unc = np.zeros(len(ids))  # model confident everywhere
    cfg = PoseBatchConfig(batch_size=10)
    res = select_pose_batch(emb, cand, labeled, ids, cfg, uncertainty=unc)
    assert REASON_UNCERTAINTY not in res.breakdown
    assert len(res.indices) == 10


def test_uncertainty_ties_are_not_index_order():
    emb, ids, labeled, cand = _problem()
    unc = np.full(len(ids), 0.5)
    cfg = PoseBatchConfig(
        batch_size=10,
        error_fraction=0,
        uncertainty_fraction=1.0,
        rare_fraction=0,
        diversity_fraction=0,
        representative_fraction=0,
    )
    res = select_pose_batch(emb, cand, labeled, ids, cfg, uncertainty=unc)
    picks = res.breakdown[REASON_UNCERTAINTY]
    assert len(picks) == 10 and picks != list(range(20, 30))


def test_pose_error_is_not_capped():
    gt = [(0.5, 0.5, 2)]
    preds = [(500.0, 50.0, 0.9)]  # 450 px off on a 100 px image
    assert pose_error_norm(gt, preds, 100, 100) == pytest.approx(4.5)
