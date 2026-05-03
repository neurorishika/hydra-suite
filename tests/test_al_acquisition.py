"""Tests for hydra_suite.data.al.acquisition."""

from __future__ import annotations

import numpy as np

from hydra_suite.data.al.acquisition import PRESETS, select
from hydra_suite.data.al.signals import ALSignals


def _signal(frame_id: int, **kwargs) -> ALSignals:
    return ALSignals(frame_id=frame_id, **kwargs)


def test_presets_are_normalized():
    for name, w in PRESETS.items():
        total = (
            w.uncertainty
            + w.nms_instability
            + w.count
            + w.crowd
            + w.edge
            + w.assignment
            + w.track_loss
            + w.position_uncertainty
        )
        assert abs(total - 1.0) < 1e-6, f"preset {name} weights sum to {total}"


def test_select_picks_highest_score():
    signals = [
        _signal(
            0, mean_confidence=0.95, margin=0.4, count_deviation=0.0, crowd_score=0.0
        ),
        _signal(
            100, mean_confidence=0.4, margin=0.0, count_deviation=0.5, crowd_score=0.7
        ),
        _signal(
            200, mean_confidence=0.85, margin=0.3, count_deviation=0.0, crowd_score=0.2
        ),
    ]
    picks = select(
        signals,
        weights=PRESETS["balanced"],
        k=1,
        diversity_window=0,
        probabilistic=False,
    )
    assert picks == [100]


def test_select_diversity_window_blocks_neighbors():
    signals = [_signal(i, mean_confidence=0.9 - 0.01 * i) for i in range(30)]
    picks = select(
        signals,
        weights=PRESETS["balanced"],
        k=3,
        diversity_window=10,
        probabilistic=False,
    )
    assert len(picks) == 3
    diffs = [abs(a - b) for a in picks for b in picks if a != b]
    assert min(diffs) > 10


def test_select_returns_at_most_k():
    signals = [_signal(i) for i in range(5)]
    picks = select(
        signals,
        weights=PRESETS["balanced"],
        k=20,
        diversity_window=0,
        probabilistic=False,
    )
    assert len(picks) <= 5


def test_select_probabilistic_deterministic_with_seed():
    signals = [_signal(i, mean_confidence=0.5 + 0.01 * i) for i in range(20)]
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = select(
        signals,
        weights=PRESETS["balanced"],
        k=5,
        diversity_window=2,
        probabilistic=True,
        rng=rng_a,
    )
    b = select(
        signals,
        weights=PRESETS["balanced"],
        k=5,
        diversity_window=2,
        probabilistic=True,
        rng=rng_b,
    )
    assert a == b


def test_select_min_score_filters_out_low_scoring_frames():
    signals = [_signal(i, mean_confidence=0.99 - 0.001 * i) for i in range(10)]
    picks = select(
        signals,
        weights=PRESETS["balanced"],
        k=10,
        diversity_window=0,
        probabilistic=False,
        min_score=0.5,
    )
    assert picks == []
