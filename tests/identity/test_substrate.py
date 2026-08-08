"""Tests for the identity substrate's partial-injective Hungarian assignment
and log-space Bayesian evidence fusion.

`solve_unique_assignment` is lifted verbatim (bit-for-bit) from the online
decoder's `_hungarian_assignment` + `_greedy_assignment` (see
`core/individual/identity/online.py:497-569`). These tests pin the exact
assignment semantics: disjoint high-confidence identities each get their own
slot, uniqueness forces conflicting slots apart, sub-threshold posteriors
resolve to `None`, and the greedy fallback agrees with the Hungarian solver
on non-degenerate inputs.

`fuse_log_evidence` is lifted from the online decoder's `_fuse_evidence`
core (`core/individual/identity/online.py:333-351`): `log_posterior +
evidence_log_probs`, renormalized via `np.logaddexp.reduce`. It adds a
no-op-by-default robustness cap/floor on top.
"""

import numpy as np
import pytest

from hydra_suite.core.individual.identity.substrate import (
    fuse_log_evidence,
    solve_unique_assignment,
)


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


def _log_posterior(*ps):
    """Build a log-space normalized posterior vector [unknown, k1, k2, ...]."""
    a = np.array(ps, dtype=np.float64)
    return np.log(a / a.sum())


def test_fuse_default_matches_online_core_exactly():
    # Default cap/floor must be bit-for-bit identical to online's
    # `_fuse_evidence` core: lp + ev, renormalized via logaddexp.reduce.
    lp = _log_posterior(0.3, 0.4, 0.3)
    ev = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    expected = lp + ev
    expected = expected - np.logaddexp.reduce(expected)

    out = fuse_log_evidence(lp, ev)
    assert np.array_equal(out, expected)


def test_fuse_does_not_mutate_inputs():
    lp = _log_posterior(0.3, 0.4, 0.3)
    lp_copy = lp.copy()
    ev = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    ev_copy = ev.copy()

    fuse_log_evidence(lp, ev)

    assert np.array_equal(lp, lp_copy)
    assert np.array_equal(ev, ev_copy)


def test_fuse_per_frame_cap_bounds_a_huge_spike():
    # A single frame's evidence should not be able to fully dominate when
    # per_frame_cap is finite: the post-fusion probability ratio between
    # any two entries is bounded by exp(2 * per_frame_cap).
    lp = _log_posterior(1.0, 1.0, 1.0)  # uniform prior
    ev = np.array([0.0, 0.0, 1e6], dtype=np.float64)  # huge spike on entry 2
    cap = 1.0

    out = fuse_log_evidence(lp, ev, per_frame_cap=cap)
    probs = np.exp(out)

    assert np.isfinite(probs).all()
    ratio = probs.max() / probs.min()
    assert ratio <= np.exp(2 * cap) + 1e-6

    # Uncapped fusion, by contrast, collapses almost entirely onto entry 2.
    out_uncapped = fuse_log_evidence(lp, ev)
    probs_uncapped = np.exp(out_uncapped)
    assert probs_uncapped[2] > 0.999


def test_fuse_prob_floor_keeps_every_entry_above_floor():
    lp = _log_posterior(1.0, 1.0, 1.0)
    # Strongly favor entry 0, driving the others toward ~0 probability.
    ev = np.array([50.0, -50.0, -50.0], dtype=np.float64)
    floor = 0.01

    out = fuse_log_evidence(lp, ev, prob_floor=floor)
    probs = np.exp(out)

    assert np.all(probs >= floor - 1e-12)
    assert np.isclose(probs.sum(), 1.0)

    # Without a floor, at least one entry collapses below it.
    out_nofloor = fuse_log_evidence(lp, ev)
    probs_nofloor = np.exp(out_nofloor)
    assert probs_nofloor.min() < floor


def test_fuse_cap_inf_and_floor_zero_are_exact_noops():
    lp = _log_posterior(0.2, 0.5, 0.3)
    ev = np.array([0.3, -0.1, 0.4], dtype=np.float64)

    out_default = fuse_log_evidence(lp, ev)
    out_explicit = fuse_log_evidence(lp, ev, per_frame_cap=float("inf"), prob_floor=0.0)

    assert np.array_equal(out_default, out_explicit)


def test_fuse_size_mismatch_raises_value_error():
    lp = _log_posterior(0.3, 0.4, 0.3)
    ev = np.array([0.1, -0.2], dtype=np.float64)  # wrong size

    with pytest.raises(ValueError):
        fuse_log_evidence(lp, ev)
