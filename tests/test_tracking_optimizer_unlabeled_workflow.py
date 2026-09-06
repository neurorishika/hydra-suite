from __future__ import annotations

import numpy as np

from hydra_suite.core.tracking.optimization import optimizer as optimizer_module
from hydra_suite.core.tracking.optimization.optimizer import (
    OptimizationResult,
    TrackingOptimizerCore,
)
from hydra_suite.core.tracking.optimization.production_replay import (
    ProductionReplayResult,
)


def _optimizer() -> TrackingOptimizerCore:
    return TrackingOptimizerCore(
        "clip.mp4",
        "cache-dir",
        0,
        39,
        {
            "MAX_TARGETS": 1,
            "REFERENCE_BODY_SIZE": 1.0,
            "RESIZE_FACTOR": 1.0,
            "W_POSITION": 1.0,
        },
        {"W_POSITION": True},
        n_trials=1,
    )


def _result(params, *, baseline=False):
    return OptimizationResult(
        params,
        score=1.0 if baseline else 0.5,
        trial_number=-1 if baseline else 0,
        sub_scores={},
        candidate_id="baseline" if baseline else "trial-0",
        is_baseline=baseline,
    )


def _install_fake_replay(monkeypatch, baseline_pair, candidate_pair):
    class FakeReplayEvaluator:
        def __init__(self, _video, _cache, start, end, **_kwargs):
            self.start = start
            self.end = end

        def run(self, params, *, reverse=False):
            pair = candidate_pair if params["W_POSITION"] == 2.0 else baseline_pair
            positions = pair[1 if reverse else 0]
            return ProductionReplayResult(
                True,
                np.arange(self.start, self.end + 1),
                positions.copy(),
            )

    monkeypatch.setattr(
        optimizer_module, "ProductionReplayEvaluator", FakeReplayEvaluator
    )


def test_production_validation_recommends_only_dominating_heldout_candidate(
    monkeypatch,
):
    time = np.arange(9, dtype=np.float32)
    smooth = np.stack([time, np.zeros_like(time)], axis=1)[:, None, :]
    jittery = smooth.copy()
    jittery[1::2, 0, 1] = 0.5
    baseline_backward = jittery.copy()
    baseline_backward[:, 0, 0] += 1.0
    _install_fake_replay(
        monkeypatch,
        (jittery, baseline_backward),
        (smooth, smooth),
    )
    core = _optimizer()
    baseline = _result({}, baseline=True)
    candidate = _result({"W_POSITION": 2.0})

    core._production_validate_shortlist([baseline, candidate], (31, 39))

    assert candidate.recommended is True
    assert baseline.recommended is False
    assert candidate.pareto_rank == 1
    assert candidate.validation_metrics["cycle_loss"]["mean"] == 0.0


def test_production_validation_keeps_baseline_when_candidate_loses_coverage(
    monkeypatch,
):
    time = np.arange(9, dtype=np.float32)
    smooth = np.stack([time, np.zeros_like(time)], axis=1)[:, None, :]
    baseline_backward = smooth.copy()
    baseline_backward[:, 0, 0] += 0.5
    gappy = smooth.copy()
    gappy[4] = np.nan
    _install_fake_replay(
        monkeypatch,
        (smooth, baseline_backward),
        (gappy, gappy),
    )
    core = _optimizer()
    baseline = _result({}, baseline=True)
    candidate = _result({"W_POSITION": 2.0})

    core._production_validate_shortlist([baseline, candidate], (31, 39))

    assert baseline.recommended is True
    assert candidate.recommended is False
    assert "retained" in baseline.recommendation_reason


def test_out_of_range_current_value_is_not_clamped_into_seed():
    core = _optimizer()
    core.base_params["W_POSITION"] = 99.0

    assert "W_POSITION" not in core._build_seed_trial()


def test_short_ranges_explicitly_keep_baseline_without_heldout_claim():
    core = _optimizer()
    core.start_frame = 0
    core.end_frame = 6
    baseline = _result({}, baseline=True)

    core._production_validate_shortlist([baseline], None)

    assert baseline.recommended is True
    assert "too short" in baseline.recommendation_reason


def test_cancelled_validation_keeps_baseline_without_false_short_range_claim():
    core = _optimizer()
    core.request_stop()
    baseline = _result({}, baseline=True)

    core._production_validate_shortlist([baseline], (30, 39))

    assert baseline.recommended is True
    assert "cancelled" in baseline.recommendation_reason


def test_validation_coverage_is_conservative_across_both_directions():
    positions = np.zeros((6, 1, 2), dtype=np.float32)
    backward = positions.copy()
    backward[1] = np.nan

    evaluations = TrackingOptimizerCore._validation_evaluations(
        "candidate", positions, backward, 1.0
    )

    assert any(item.metrics["coverage_loss"] > 0 for item in evaluations)


def test_lifecycle_search_neighborhood_scales_with_video_fps():
    low_fps = _optimizer()
    high_fps = _optimizer()
    low_fps.base_params["LOST_THRESHOLD_FRAMES"] = 15  # 0.5 s at 30 FPS
    high_fps.base_params["LOST_THRESHOLD_FRAMES"] = 30  # 0.5 s at 60 FPS
    low_fps.base_params["FPS"] = 30.0
    high_fps.base_params["FPS"] = 60.0

    _, low_min, low_max = low_fps._search_range("LOST_THRESHOLD_FRAMES")
    _, high_min, high_max = high_fps._search_range("LOST_THRESHOLD_FRAMES")

    assert abs(low_min / 30.0 - high_min / 60.0) < 0.02
    assert low_max / 30.0 == high_max / 60.0


def test_lifecycle_search_never_exceeds_applyable_seconds_range():
    core = _optimizer()
    core.base_params.update({"FPS": 240.0, "LOST_THRESHOLD_FRAMES": 24000})

    _, _low, high = core._search_range("LOST_THRESHOLD_FRAMES")

    assert high == 40 * 240
