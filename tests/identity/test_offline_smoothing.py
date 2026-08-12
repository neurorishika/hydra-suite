"""Identity Phase 5 Task 2: forward-backward (two-filter) trajectory smoothing.

``smooth_trajectory_posteriors`` combines a trajectory's per-frame calibrated
evidence (as sourced by ``load_trajectory_evidence``, Task 1) with both its
left context (forward/causal pass) and right context (backward/anti-causal
pass), propagated through the same sticky-Markov transition the online
decoder uses (mirroring ``online.py``'s ``_build_log_transition`` /
``_predict_belief``), so a confident late burst of evidence can correct
ambiguous early frames -- something the online decoder's forward-only belief
can never do.

These tests pin the two load-bearing correctness properties documented on
`smooth_trajectory_posteriors`:
  * **correction**: a late-confident burst raises early-frame confidence in
    the correct identity, beyond what the raw (unsmoothed) evidence alone
    shows.
  * **no double-counting**: a single-frame trajectory's smoothed posterior
    is exactly its own evidence (the two-filter combination must not fuse
    that frame's evidence twice), and a two-frame stable trajectory's
    smoothed posterior matches a hand-computed value, not something sharper.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.smoothing import (
    smooth_trajectory_posteriors,
    smoothed_label_and_conf,
)


@pytest.fixture
def catalog() -> IdentityCatalog:
    return IdentityCatalog.from_labels(["ant_a", "ant_b"])


def _uniform_log_probs(catalog: IdentityCatalog) -> np.ndarray:
    v = np.zeros(catalog.size, dtype=np.float64)
    return v - np.logaddexp.reduce(v)


def _confident_log_probs(
    catalog: IdentityCatalog, label: str, strength: float = 6.0
) -> np.ndarray:
    v = np.zeros(catalog.size, dtype=np.float64)
    v[catalog.index_of(label)] = strength
    return v - np.logaddexp.reduce(v)


# ---------------------------------------------------------------------------
# Empty / length / order
# ---------------------------------------------------------------------------


def test_empty_trajectory_returns_empty():
    assert smooth_trajectory_posteriors([], transition_epsilon=0.05) == []


def test_length_and_order_preserved(catalog):
    seq = [
        _uniform_log_probs(catalog),
        _confident_log_probs(catalog, "ant_a"),
        _confident_log_probs(catalog, "ant_b"),
    ]
    smoothed = smooth_trajectory_posteriors(seq, transition_epsilon=0.05)
    assert len(smoothed) == len(seq)
    for lp in smoothed:
        assert lp.shape == (catalog.size,)
        # each frame's output is a normalized log-posterior
        assert np.isclose(np.logaddexp.reduce(lp), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# No double-counting
# ---------------------------------------------------------------------------


def test_single_frame_smoothed_equals_its_own_evidence(catalog):
    """No left/right context to fuse -> smoothed must equal the raw evidence
    exactly, not that evidence fused with itself a second time."""
    evidence = _confident_log_probs(catalog, "ant_b")
    smoothed = smooth_trajectory_posteriors([evidence], transition_epsilon=0.05)
    assert len(smoothed) == 1
    assert np.allclose(smoothed[0], evidence, atol=1e-12)


def test_two_frame_stable_matches_hand_computed_value(catalog):
    """Two identical confident frames: smoothed posteriors must match the
    forward-only chained-fuse value exactly (both frames see the same
    aggregate context by symmetry), not something sharper than that -- i.e.
    the two-filter combination must not double-count either frame's own
    evidence."""
    from hydra_suite.core.individual.identity.smoothing import (
        _build_log_transition,
        _normalize_log_probs,
        _predict,
    )
    from hydra_suite.core.individual.identity.substrate import fuse_log_evidence

    evidence = _confident_log_probs(catalog, "ant_b")
    seq = [evidence, evidence]

    log_transition = _build_log_transition(catalog.size, 0.05)
    predicted = _predict(evidence, log_transition)
    hand_computed = _normalize_log_probs(fuse_log_evidence(predicted, evidence))

    smoothed = smooth_trajectory_posteriors(seq, transition_epsilon=0.05)
    assert np.allclose(smoothed[0], hand_computed, atol=1e-9)
    assert np.allclose(smoothed[1], hand_computed, atol=1e-9)

    # Not sharper than the hand-computed two-frame chained-fuse value: the
    # ant_b probability must match (within float tolerance), never exceed it.
    b_idx = catalog.index_of("ant_b")
    smoothed_prob = float(np.exp(smoothed[0][b_idx]))
    hand_prob = float(np.exp(hand_computed[b_idx]))
    assert smoothed_prob <= hand_prob + 1e-9


# ---------------------------------------------------------------------------
# Correction: late-confident burst fixes early ambiguity
# ---------------------------------------------------------------------------


def test_late_confident_burst_corrects_early_ambiguous_frames(catalog):
    """Early frames are near-uniform (ambiguous); late frames are
    confidently ant_b. The SMOOTHED early-frame posteriors must favor ant_b
    (argmax == ant_b) with a higher ant_b-probability than the raw evidence
    shows -- something a forward-only (causal) chain cannot do, since a
    forward-only pass at an early frame has no knowledge of the future."""
    b_idx = catalog.index_of("ant_b")

    seq = [
        _uniform_log_probs(catalog),
        _uniform_log_probs(catalog),
        _uniform_log_probs(catalog),
        _confident_log_probs(catalog, "ant_b"),
        _confident_log_probs(catalog, "ant_b"),
        _confident_log_probs(catalog, "ant_b"),
    ]

    smoothed = smooth_trajectory_posteriors(seq, transition_epsilon=0.05)

    raw_first_probs = np.exp(seq[0] - np.logaddexp.reduce(seq[0]))
    smoothed_first_probs = np.exp(smoothed[0])

    # Forward-only (causal) chain: what the online decoder's belief would
    # look like at frame 0 -- exactly the raw evidence, since there is no
    # prior belief yet.
    forward_only_first = raw_first_probs

    assert np.argmax(smoothed_first_probs[1:]) == b_idx - 1  # known-only argmax
    assert smoothed_first_probs[b_idx] > raw_first_probs[b_idx]
    assert smoothed_first_probs[b_idx] > forward_only_first[b_idx]

    # A substantial correction, not a rounding-noise nudge.
    assert smoothed_first_probs[b_idx] > 0.5


# ---------------------------------------------------------------------------
# smoothed_label_and_conf
# ---------------------------------------------------------------------------


def test_smoothed_label_and_conf_empty_returns_empty(catalog):
    assert smoothed_label_and_conf([], catalog, display_threshold=0.5) == []


def test_smoothed_label_and_conf_above_threshold_returns_known_label(catalog):
    smoothed = [_confident_log_probs(catalog, "ant_b")]
    result = smoothed_label_and_conf(smoothed, catalog, display_threshold=0.5)
    assert len(result) == 1
    label, conf = result[0]
    assert label == "ant_b"
    assert conf > 0.5


def test_smoothed_label_and_conf_below_threshold_returns_empty_label(catalog):
    smoothed = [_uniform_log_probs(catalog)]
    result = smoothed_label_and_conf(smoothed, catalog, display_threshold=0.5)
    assert len(result) == 1
    label, conf = result[0]
    assert label == ""
    # confidence is zeroed below the display threshold (online-decoder convention)
    assert conf == 0.0


def test_smoothed_label_and_conf_preserves_length_and_order(catalog):
    smoothed = [
        _confident_log_probs(catalog, "ant_a"),
        _uniform_log_probs(catalog),
        _confident_log_probs(catalog, "ant_b"),
    ]
    result = smoothed_label_and_conf(smoothed, catalog, display_threshold=0.5)
    assert [label for label, _ in result] == ["ant_a", "", "ant_b"]
