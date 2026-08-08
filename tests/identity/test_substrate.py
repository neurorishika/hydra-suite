"""Tests for the identity substrate's partial-injective Hungarian assignment.

`solve_unique_assignment` is lifted verbatim (bit-for-bit) from the online
decoder's `_hungarian_assignment` + `_greedy_assignment` (see
`core/individual/identity/online.py:497-569`). These tests pin the exact
assignment semantics: disjoint high-confidence identities each get their own
slot, uniqueness forces conflicting slots apart, sub-threshold posteriors
resolve to `None`, and the greedy fallback agrees with the Hungarian solver
on non-degenerate inputs.
"""

import numpy as np

from hydra_suite.core.individual.identity.substrate import solve_unique_assignment


def _p(*ps):
    """Build a normalized catalog posterior vector [unknown, k1, k2, ...]."""
    a = np.array(ps, dtype=np.float64)
    return a / a.sum()


def test_disjoint_identities_assigned():
    # 2 slots, 2 known; slot0 -> k1, slot1 -> k2 (disjoint high-confidence).
    post = [_p(0.01, 0.98, 0.01), _p(0.01, 0.01, 0.98)]
    out = solve_unique_assignment(post, num_known=2, display_threshold=0.6)
    assert out == [1, 2]


def test_uniqueness_forces_one_off():
    # Both slots favor k1; uniqueness constraint forces only one to get it.
    post = [_p(0.01, 0.98, 0.01), _p(0.02, 0.97, 0.01)]
    out = solve_unique_assignment(post, num_known=2, display_threshold=0.6)
    assert 1 in out
    assert out.count(1) == 1


def test_below_display_threshold_is_none():
    post = [_p(0.5, 0.5, 0.0)]
    assert solve_unique_assignment(post, num_known=2, display_threshold=0.6) == [None]


def test_greedy_matches_scipy():
    post = [_p(0.01, 0.9, 0.09), _p(0.01, 0.2, 0.79)]
    scipy_out = solve_unique_assignment(post, 2, 0.6, use_scipy=True)
    greedy_out = solve_unique_assignment(post, 2, 0.6, use_scipy=False)
    assert scipy_out == greedy_out


def test_empty_slots_returns_empty_list():
    assert solve_unique_assignment([], num_known=2, display_threshold=0.6) == []


def test_greedy_direct_path():
    # use_scipy=False should route straight through the greedy assignment,
    # bypassing scipy entirely.
    post = [_p(0.01, 0.98, 0.01), _p(0.01, 0.01, 0.98)]
    out = solve_unique_assignment(
        post, num_known=2, display_threshold=0.6, use_scipy=False
    )
    assert out == [1, 2]
