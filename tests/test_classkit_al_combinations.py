import numpy as np

from hydra_suite.classkit.core.al.acquisition import BatchAcquisition, BatchConfig
from hydra_suite.classkit.core.al.combination import (
    CombinationBalanceSelector,
    count_labeled_combinations,
    factor_marginals_from_composite,
    factor_marginals_from_heads,
)

LABELS = [["a", "b", "unknown"], ["x", "y", "unknown"]]


def _onehot(idx, n):
    out = np.zeros((len(idx), n))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def test_composite_marginals_find_combos_the_model_never_outputs():
    # Model only knows a_x, a_y, b_x (never b_y).
    names = ["a_x", "a_y", "b_x"]
    probs = np.array([[0.5, 0.0, 0.5]])  # half a_x, half b_x
    fa, fb = factor_marginals_from_composite(probs, names, LABELS)
    assert np.allclose(fa[0], [0.5, 0.5, 0.0]) and np.allclose(fb[0], [1, 0, 0])


def test_head_marginals_slice_by_head():
    probs = np.array([[0.9, 0.1, 0.0, 0.2, 0.8, 0.0]])
    heads = [
        {"start": 0, "end": 3, "class_names": ["a", "b", "unknown"]},
        {"start": 3, "end": 6, "class_names": ["x", "y", "unknown"]},
    ]
    fa, fb = factor_marginals_from_heads(probs, heads, LABELS)
    assert fa[0, 0] > 0.8 and fb[0, 1] > 0.7


def test_selector_targets_missing_combination_and_spreads():
    # 40 images look like b_y (never labeled), 40 look like a_x (heavily labeled).
    n = 80
    fa = _onehot(np.array([1] * 40 + [0] * 40), 3)
    fb = _onehot(np.array([1] * 40 + [0] * 40), 3)
    labels = [None] * n
    for i in range(40, 70):
        labels[i] = "a_x"
    avail = np.array([l is None for l in labels])
    res = CombinationBalanceSelector().select([fa, fb], LABELS, labels, avail, 10)
    assert len(res.indices) == 10
    assert all(i < 40 for i in res.indices)  # the b_y-looking images
    assert "b_y" in res.missing and res.counts["a_x"] == 30
    assert set(res.target_combo.values()) == {"b_y"}


def test_diminishing_returns_spread_over_missing_combos():
    # Two missing combos each with plenty of candidates; picks must cover both.
    fa = _onehot(np.array([0] * 30 + [1] * 30), 3)
    fb = _onehot(np.array([0] * 30 + [1] * 30), 3)  # a_x then b_y
    labels = [None] * 60
    res = CombinationBalanceSelector().select(
        [fa, fb], LABELS, labels, np.ones(60, bool), 10
    )
    got = {res.target_combo[int(i)] for i in res.indices}
    assert got == {"a_x", "b_y"}


def test_unknown_parts_are_not_targets_and_untrusted_labels_ignored():
    labels = ["a_x", "a_x"]
    counts = count_labeled_combinations(labels, LABELS, np.array([True, False]))
    assert counts == {("a", "x"): 1}
    fa = _onehot(np.array([2, 0]), 3)  # image 0 looks like unknown_*
    fb = _onehot(np.array([0, 0]), 3)
    res = CombinationBalanceSelector().select(
        [fa, fb], LABELS, [None, None], np.ones(2, bool), 1
    )
    assert list(res.indices) == [1]


def test_batch_acquisition_uses_combination_slot():
    rng = np.random.default_rng(0)
    n = 120
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    probs = np.full((n, 3), 1 / 3)
    fa = _onehot(np.array([1] * 60 + [0] * 60), 3)
    fb = _onehot(np.array([1] * 60 + [0] * 60), 3)
    labels = [None] * n
    for i in range(60, 90):
        labels[i] = "a_x"
    avail = np.array([l is None for l in labels])
    cfg = BatchConfig(
        batch_size=20,
        combination_fraction=0.5,
        error_fraction=0.0,
        rare_cluster_fraction=0.0,
    )
    sel, breakdown = BatchAcquisition(cfg, seed=0).select_batch(
        emb,
        probs,
        avail,
        image_labels=labels,
        factor_probs=[fa, fb],
        factor_labels=LABELS,
    )
    assert len(set(sel.tolist())) == len(sel) == 20
    assert len(breakdown["combination"]) == 10
    assert all(i < 60 for i in breakdown["combination"])
