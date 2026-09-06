from __future__ import annotations

import pytest

from hydra_suite.core.tracking.optimization.optimizer import (
    _PROPOSAL_CYCLE_TERM_WEIGHT,
    _bounded_cycle_proposal_penalty,
    _compute_composite_score,
)


def _score(*, coverage_sum: float, detection_count_sum: float, n_targets: int = 2):
    return _compute_composite_score(
        10,
        n_targets,
        coverage_sum,
        0.0,
        0.0,
        10,
        0,
        detection_count_sum,
        [10] * n_targets,
        [1.0] * 10,
        [0.0] * 10,
        0.0,
        10,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )[0]


def test_composite_score_cannot_improve_when_coverage_worsens():
    assert _score(coverage_sum=5.0, detection_count_sum=20.0) > _score(
        coverage_sum=10.0, detection_count_sum=20.0
    )


def test_false_positive_penalty_is_fractional_not_divided_by_target_count_twice():
    # 50% excess detections reaches the full excess-detection component for
    # both small and large groups.
    small = _score(coverage_sum=10.0, detection_count_sum=30.0, n_targets=2)
    large = _score(coverage_sum=10.0, detection_count_sum=150.0, n_targets=10)
    assert small == large


def test_proposal_cycle_term_is_monotonic_and_bounded() -> None:
    penalties = [
        _bounded_cycle_proposal_penalty(loss) for loss in (0.0, 0.1, 1.0, 10.0)
    ]

    assert penalties[0] == 0.0
    assert penalties == sorted(penalties)
    assert penalties[-1] == pytest.approx(_PROPOSAL_CYCLE_TERM_WEIGHT * 10.0 / 11.0)
    assert all(0.0 <= penalty < _PROPOSAL_CYCLE_TERM_WEIGHT for penalty in penalties)
