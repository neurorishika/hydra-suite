from __future__ import annotations

from pathlib import Path

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
    time = np.arange(16, dtype=np.float32)
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

    core._production_validate_shortlist([baseline, candidate], (24, 39))

    assert candidate.recommended is True
    assert baseline.recommended is False
    assert candidate.pareto_rank == 1
    assert candidate.validation_metrics["cycle_loss"]["mean"] == 0.0


def test_production_validation_keeps_baseline_when_candidate_loses_coverage(
    monkeypatch,
):
    time = np.arange(16, dtype=np.float32)
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

    core._production_validate_shortlist([baseline, candidate], (24, 39))

    assert baseline.recommended is True
    assert candidate.recommended is False
    assert "retained" in baseline.recommendation_reason


def test_out_of_range_current_value_is_not_clamped_into_seed():
    core = _optimizer()
    core.base_params["W_POSITION"] = 99.0

    assert "W_POSITION" not in core._build_seed_trial()


def test_seed_and_optuna_candidates_are_quantized_before_evaluation() -> None:
    core = _optimizer()
    core.base_params["W_POSITION"] = 1.2345
    core.tuning_config = {
        "W_POSITION": True,
        "KALMAN_DAMPING": True,
        "MAX_DISTANCE_MULTIPLIER": True,
    }

    class _Trial:
        def suggest_float(self, name, _low, _high, *, log=False):
            assert log is False
            return {
                "W_POSITION": 1.2345,
                "KALMAN_DAMPING": 0.91234,
                "MAX_DISTANCE_MULTIPLIER": 1.2345,
            }[name]

    assert core._build_seed_trial()["W_POSITION"] == 1.23
    assert core._suggest_trial_params(_Trial(), scaled_body_size=10.0) == {
        "W_POSITION": 1.23,
        "KALMAN_DAMPING": 0.912,
        "MAX_DISTANCE_MULTIPLIER": 1.23,
        "MAX_DISTANCE_THRESHOLD": 12.3,
    }


def test_optimizer_normalizes_detection_cache_member_path(tmp_path) -> None:
    cache_member = tmp_path / "cache" / "detection.npz"
    core = TrackingOptimizerCore(
        "clip.mp4",
        str(cache_member),
        0,
        1,
        {"MAX_TARGETS": 1},
        {},
    )

    assert Path(core.detection_cache_path) == cache_member.parent


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


def test_validation_regions_reuse_one_full_window_slot_alignment() -> None:
    time = np.arange(12, dtype=np.float32)
    forward = np.stack(
        [
            np.column_stack([time, np.zeros_like(time)]),
            np.column_stack([100.0 + time, np.zeros_like(time)]),
        ],
        axis=1,
    )
    backward = forward.copy()
    backward[3:6, [0, 1]] = backward[3:6, [1, 0]]

    evaluations = TrackingOptimizerCore._validation_evaluations(
        "candidate", forward, backward, 1.0
    )

    # Segment-local Hungarian alignment makes the swapped middle block look
    # perfect.  A single full-window alignment retains the identity error.
    assert evaluations[1].metrics["cycle_loss"] > 5.0


def test_validation_regions_include_output_safety_signals() -> None:
    positions = np.zeros((12, 2, 2), dtype=np.float32)
    positions[:, 1] = positions[:, 0]  # persistent collision / duplicate output
    positions[6:, 1] = np.nan  # a poorly covered second slot
    counts = np.full(12, 4, dtype=np.int64)  # twice the requested targets

    evaluations = TrackingOptimizerCore._validation_evaluations(
        "candidate",
        positions,
        positions.copy(),
        1.0,
        detection_counts=counts,
    )

    assert any(item.metrics["collision_loss"] > 0 for item in evaluations)
    assert any(item.metrics["detection_excess_loss"] > 0 for item in evaluations)
    assert any(
        item.metrics["worst_track_coverage_loss"] > item.metrics["coverage_loss"]
        for item in evaluations
    )


def test_validation_retains_baseline_when_temporal_support_is_inadequate():
    core = _optimizer()
    baseline = _result({}, baseline=True)

    core._production_validate_shortlist([baseline], (30, 39))

    assert baseline.recommended is True
    assert "temporal support" in baseline.recommendation_reason


def test_validation_support_scales_with_a_tuned_temporal_horizon():
    core = _optimizer()
    core.base_params["LOST_THRESHOLD_FRAMES"] = 5
    core.tuning_config["LOST_THRESHOLD_FRAMES"] = True

    reason = core._validation_support_reason((0, 39))

    assert reason is not None
    assert "20-frame horizon" in reason


def test_plateau_convergence_still_runs_heldout_validation(monkeypatch) -> None:
    time = np.arange(16, dtype=np.float32)
    smooth = np.stack([time, np.zeros_like(time)], axis=1)[:, None, :]
    baseline_backward = smooth.copy()
    baseline_backward[:, 0, 0] += 1.0
    _install_fake_replay(monkeypatch, (smooth, baseline_backward), (smooth, smooth))
    core = _optimizer()
    core._search_converged = True
    baseline = _result({}, baseline=True)
    candidate = _result({"W_POSITION": 2.0})

    core._production_validate_shortlist([baseline, candidate], (24, 39))

    assert candidate.recommended is True


def test_plateau_stop_marks_convergence_without_requesting_cancellation(
    monkeypatch,
) -> None:
    class _Trial:
        def __init__(self, number: int) -> None:
            self.number = number

    class _Study:
        def __init__(self) -> None:
            self.stopped = False

        def enqueue_trial(self, _params) -> None:
            pass

        def stop(self) -> None:
            self.stopped = True

        def optimize(self, objective, *, n_trials: int) -> None:
            for number in range(n_trials):
                objective(_Trial(number))
                if self.stopped:
                    break

    core = _optimizer()
    core.n_trials = 16
    core.on_plateau = "stop"
    core.tuning_config = {}
    monkeypatch.setattr(core, "_open_and_validate_cache", lambda: True)
    monkeypatch.setattr(core, "_preload_pose_data", lambda: None)
    monkeypatch.setattr(core, "_build_sampler", lambda _n_active: object())
    monkeypatch.setattr(core, "_search_and_validation_bounds", lambda: ((0, 1), None))
    monkeypatch.setattr(core, "_proposal_score", lambda *_args, **_kwargs: (1.0, {}))
    validated = []
    monkeypatch.setattr(
        core,
        "_production_validate_shortlist",
        lambda results, bounds: validated.append((results, bounds)),
    )
    monkeypatch.setattr(
        optimizer_module.optuna, "create_study", lambda **_kwargs: _Study()
    )

    core.optimize()

    assert core._search_converged is True
    assert core._stop_requested is False
    assert validated


def test_shortlist_includes_normalized_metric_extremes_and_diverse_candidates() -> None:
    core = _optimizer()
    candidates = []
    for index in range(6):
        result = _result({"W_POSITION": float(index + 1)})
        result.candidate_id = f"trial-{index}"
        result.score = float(index)
        result.sub_scores = {
            "cycle_loss": float(10 - index),
            "coverage": float(index),
            "assignment": float(index),
            "fragmentation": float(index),
            "occlusion": float(index),
            "velocity": float(index),
            "crowding": float(index),
        }
        candidates.append(result)

    shortlist = core._select_validation_shortlist(candidates, limit=3)

    # trial-0 has the least raw scalar score, whereas trial-5 is the only
    # cycle-consistency extreme.  Normalized multi-metric selection keeps both.
    assert {item.candidate_id for item in shortlist} >= {"trial-0", "trial-5"}


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
