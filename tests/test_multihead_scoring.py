"""Table-driven tests for multi-head scoring modes."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "track, det, expected",
    [
        (("B", "G"), ("B", "B"), +1.0),
        (("B", "G"), ("B", "G"), -0.5),
        (("B", None), ("B", "G"), 0.0),
        (("B", "unknown"), ("B", "G"), 0.0),
        ((None, None), ("B", "G"), 0.0),
    ],
)
def test_cost_atomic(track, det, expected):
    from hydra_suite.core.identity.classification.cnn import cost_atomic

    got = cost_atomic(track, det, match_bonus=0.5, mismatch_penalty=1.0)
    assert got == expected


@pytest.mark.parametrize(
    "track, det, expected",
    [
        (("B", "G"), ("B", "B"), (-0.5 + 1.0) / 2),  # 1 match + 1 mismatch, K=2
        (("B", "G"), ("B", "G"), -0.5),  # avg of two -bonus = -bonus
        (("B", "unknown"), ("B", "G"), -0.5 / 2),  # 1 match + 1 skip, K=2
        (("B", None), ("B", "G"), -0.5 / 2),  # None behaves like unknown
        ((None, "G"), ("B", "G"), -0.5 / 2),  # None in track, G matches
        (("unknown", "unknown"), ("B", "G"), 0.0),  # all skipped
    ],
)
def test_cost_per_head_average(track, det, expected):
    from hydra_suite.core.identity.classification.cnn import cost_per_head_average

    got = cost_per_head_average(
        track, det, match_bonus=0.5, mismatch_penalty=1.0, K=len(track)
    )
    assert abs(got - expected) < 1e-9
