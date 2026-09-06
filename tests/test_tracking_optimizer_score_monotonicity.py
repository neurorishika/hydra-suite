from __future__ import annotations

from hydra_suite.core.tracking.optimization.optimizer import _compute_composite_score


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
