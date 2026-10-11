import numpy as np
import pytest

from hydra_suite.classkit.core.al.grid_balance import GridBalanceSession

LABELS = [["a", "b", "unknown"], ["x", "y", "unknown"]]


def _marginals(n, rng):
    fa = rng.dirichlet(np.ones(3), size=n)
    fb = rng.dirichlet(np.ones(3), size=n)
    return [fa, fb]


def _session(n=200, labels=None, **kw):
    rng = np.random.default_rng(1)
    probs = _marginals(n, rng)
    if labels is None:
        labels = [None] * n
        labels[:6] = ["a_x"] * 6  # a_x is the most common combination
        labels[6:8] = ["b_x"] * 2
        labels[8] = "a_y"
    return GridBalanceSession(probs, LABELS, labels, **kw), probs, labels


def test_rarest_first_target_is_most_common_and_unknown_excluded():
    s, _, _ = _session()
    assert s.target == 6
    assert s.combos[0] == "b_y"  # 0 labels
    assert s.combos[-1] == "b_x"  # 2 labels, a_y has 1
    assert "a_x" not in s.combos  # already at target
    assert not any("unknown" in c for c in s.combos)


def test_include_unknown_and_target_cap():
    s, _, _ = _session(include_unknown=True, target_cap=3)
    assert s.target == 3
    assert "unknown_unknown" in s.combos
    assert "b_x" in s.combos and s.counts["b_x"] == 2


def test_grid_only_shows_unlabeled_unique_images():
    s, _, labels = _session(grid_size=16)
    assert len(s.grid) == 16 and None not in s.grid
    assert len(set(s.grid)) == 16
    assert not any(labels[i] for i in s.grid)
    assert sorted(set(s.sources)) == ["explore", "high", "mid"]


def test_high_tiles_are_the_most_likely():
    s, _, _ = _session(grid_size=9, fractions=(1, 0, 0))
    p = s.probability(s.current)
    unl = [i for i in range(s.n) if not s.labels[i]]
    best = sorted(unl, key=lambda i: -p[i])[:9]
    assert sorted(s.grid) == sorted(best)


def test_click_labels_refills_and_never_reshows():
    s, _, _ = _session(grid_size=9)
    combo = s.current
    first = list(s.grid)
    res = s.click(4)
    assert res.index == first[4] and res.label == combo
    assert s.labels[first[4]] == combo and s.counts[combo] == 1
    assert s.grid[4] not in first
    seen = set(first)
    for _ in range(5):
        s.refresh()
        assert seen.isdisjoint(s.grid)  # unclicked tiles never come back
        seen.update(s.grid)


def test_unclicked_tiles_get_no_label():
    s, _, labels = _session(grid_size=9)
    shown = list(s.grid)
    s.refresh()
    s.next_combo()
    assert all(s.labels[i] is None for i in shown)


def test_completion_flag_and_next_combo_skips_completed():
    s, _, _ = _session(grid_size=9, target_cap=2)
    combo = s.current
    assert not s.click(0).completed
    assert s.click(1).completed
    assert s.next_combo()
    assert s.current != combo


def test_undo_restores_tile_and_label_even_across_combos():
    s, _, _ = _session(grid_size=9)
    c0 = s.current
    idx = s.grid[2]
    s.click(2)
    replacement = s.grid[2]
    s.undo()
    assert s.labels[idx] is None and s.counts[c0] == 0
    assert s.grid[2] == idx
    # The displaced replacement returns to the pool.
    assert s.remaining() >= 1 and replacement not in s.grid

    idx = s.grid[0]
    s.click(0)
    s.next_combo()
    assert s.current != c0
    hit = s.undo()
    assert hit.index == idx and s.current == c0 and s.grid[0] == idx
    assert s.undo() is None


def test_rare_combo_never_predicted_still_reachable():
    # Model is confident about factor A=b and factor B=y separately but never
    # assigns mass to b_y jointly in composite space; marginals still find it.
    n = 50
    fa = np.tile([0.1, 0.8, 0.1], (n, 1))
    fb = np.tile([0.1, 0.8, 0.1], (n, 1))
    fa[7] = [0.0, 1.0, 0.0]
    fb[7] = [0.0, 1.0, 0.0]
    s = GridBalanceSession([fa, fb], LABELS, [None] * n, grid_size=4)
    s.pos = s.combos.index("b_y")
    s._enter(s.pos)
    high = [i for i, src in zip(s.grid, s.sources) if src == "high"]
    assert 7 in high


def test_exhaustion_leaves_empty_tiles():
    s, _, _ = _session(n=12, labels=[None] * 12, grid_size=9)
    s.refresh()
    assert s.grid.count(None) == 9 - 3
    s.refresh()
    assert all(i is None for i in s.grid) and s.remaining() == 0


def test_finished_when_all_combos_visited():
    s, _, _ = _session(grid_size=4)
    while s.next_combo():
        pass
    assert s.finished and s.grid == [] and s.click(0) is None


def test_untrusted_labels_do_not_count():
    n = 30
    rng = np.random.default_rng(0)
    labels = ["a_x"] * 5 + [None] * (n - 5)
    trusted = np.zeros(n, dtype=bool)
    s = GridBalanceSession(
        _marginals(n, rng), LABELS, labels, trusted, grid_size=4, target_cap=4
    )
    assert s.counts["a_x"] == 0 and "a_x" in s.combos
    assert s.examples("a_x") == []


def test_flat_scheme():
    n = 40
    rng = np.random.default_rng(2)
    probs = [rng.dirichlet(np.ones(3), size=n)]
    labels = ["cat"] * 4 + [None] * (n - 4)
    s = GridBalanceSession(probs, [["cat", "dog", "bird"]], labels, grid_size=4)
    assert s.target == 4 and set(s.combos) == {"dog", "bird"}


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        GridBalanceSession([np.ones((3, 3))], LABELS, [None] * 3)


def test_mid_band_stays_plausible_on_large_pools():
    # 100k images, ~1% true members with high P; the rest near zero.
    n = 100_000
    rng = np.random.default_rng(4)
    members = rng.random(n) < 0.01
    pa = np.where(members, 0.9, 0.02)
    fa = np.stack([pa, 1 - pa, np.zeros(n)], axis=1)
    fb = np.stack([np.ones(n), np.zeros(n), np.zeros(n)], axis=1)
    s = GridBalanceSession([fa, fb], LABELS, [None] * n, grid_size=16)
    s.pos = s.combos.index("a_x")
    s._enter(s.pos)
    p = s.probability("a_x")
    mids = []
    for _ in range(40):
        s.refresh()
        mids += [i for i, src in zip(s.grid, s.sources) if src == "mid"]
    assert np.median(p[mids]) > 0.5


def test_undo_into_other_combo_releases_unanswered_tiles():
    s, _, _ = _session(grid_size=9)
    s.click(0)
    s.next_combo()
    other, unanswered = s.current, [i for i in s.grid if i is not None]
    s.undo()
    s.pos = s.combos.index(other)
    s._enter(s.pos)
    assert set(unanswered) & set(s.grid)


def test_one_tile_grid_shows_top_candidate():
    s, _, _ = _session(grid_size=1)
    p = s.probability(s.current)
    unl = [i for i in range(s.n) if not s.labels[i]]
    assert s.grid == [max(unl, key=lambda i: p[i])]


def test_zero_images_and_colliding_names():
    s = GridBalanceSession([np.zeros((0, 2))], [["a", "b"]], [])
    assert s.remaining() == 0
    with pytest.raises(ValueError):
        GridBalanceSession(
            [np.ones((2, 2)), np.ones((2, 2))],
            [["a_b", "a"], ["c", "b_c"]],
            [None, None],
        )


def test_joint_probs_keep_factor_correlation():
    n = 20
    fa = np.tile([0.5, 0.5, 0.0], (n, 1))
    fb = np.tile([0.5, 0.5, 0.0], (n, 1))
    joint = {"a_y": np.zeros(n), "a_x": np.full(n, 0.5)}
    joint["a_y"][3] = 0.9
    s = GridBalanceSession([fa, fb], LABELS, [None] * n, joint_probs=joint)
    assert s.probability("a_y")[3] == 0.9 and s.probability("a_y")[0] == 0.0
    assert s.probability("b_y")[0] == 0.25  # falls back to marginals


def test_replace_swaps_without_labeling():
    s, _, _ = _session(grid_size=9)
    idx = s.grid[3]
    new = s.replace(3)
    assert new != idx and s.labels[idx] is None and idx not in s.grid
