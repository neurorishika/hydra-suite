from __future__ import annotations

from pathlib import Path

import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_for_source
from hydra_suite.core.tracking.optimization import optimizer as optimizer_module
from hydra_suite.core.tracking.optimization.detection_config import (
    inference_config_for_optimizer_params,
)
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


def test_validation_regions_reject_single_observation_cycle_evidence() -> None:
    """A globally valid slot mapping must not make one-frame blocks perfect."""

    forward = np.full((12, 1, 2), np.nan, dtype=np.float32)
    forward[[0, 3, 6, 9], 0] = np.array([1.0, 0.0], dtype=np.float32)
    backward = forward.copy()

    evaluations = TrackingOptimizerCore._validation_evaluations(
        "candidate", forward, backward, 1.0
    )

    # The pair has four shared observations across the full held-out window,
    # but only one in each region.  Reporting zero loss here would turn
    # accidental one-frame agreement into apparently decisive paired evidence.
    assert [item.metrics["cycle_loss"] for item in evaluations] == [10.0] * 4
    assert [item.metrics["cycle_observation_coverage"] for item in evaluations] == [
        0.0
    ] * 4


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


def test_validation_retains_baseline_when_long_tail_has_no_motion_triplets(
    monkeypatch,
) -> None:
    """Frame count alone is not temporal evidence for a sparse held-out tail."""

    sparse = np.full((16, 1, 2), np.nan, dtype=np.float32)
    sparse[[0, 4, 8, 12], 0] = np.array([1.0, 0.0], dtype=np.float32)
    time = np.arange(16, dtype=np.float32)
    smooth = np.stack([time, np.zeros_like(time)], axis=1)[:, None, :]
    _install_fake_replay(monkeypatch, (sparse, sparse.copy()), (smooth, smooth))
    core = _optimizer()
    baseline = _result({}, baseline=True)
    candidate = _result({"W_POSITION": 2.0})

    # Sixteen frames meet the current 4 x 3 frame horizon gate, yet the
    # baseline has no three-consecutive-observation motion evidence in any
    # paired region.  It must be retained rather than letting neutral roughness
    # values support a promotion.
    core._production_validate_shortlist([baseline, candidate], (24, 39))

    assert baseline.recommended is True
    assert candidate.recommended is False
    assert "motion-triplet" in baseline.recommendation_reason


def test_validation_motion_support_scales_with_tuned_temporal_horizon() -> None:
    core = _optimizer()
    core.base_params["LOST_THRESHOLD_FRAMES"] = 5
    core.tuning_config["LOST_THRESHOLD_FRAMES"] = True
    sparse_runs = np.full((80, 1, 2), np.nan, dtype=np.float32)
    for start in range(0, 80, 20):
        sparse_runs[start : start + 3, 0, 0] = np.arange(3, dtype=np.float32)

    evaluations = core._validation_evaluations(
        "candidate", sparse_runs, sparse_runs.copy(), 1.0
    )

    # Each 20-frame region contains one observed triplet, but a parameter with
    # a 20-frame effect horizon needs more than that isolated three-frame run.
    reason = core._validation_temporal_evidence_reason(
        sparse_runs, sparse_runs.copy(), 1.0, evaluations
    )

    assert reason is not None
    assert "18 observed motion-triplet" in reason


def test_detection_excess_uses_native_frame_roi_mask(monkeypatch) -> None:
    """The source-count safeguard must match cached replay's ROI geometry."""

    core = _optimizer()
    core.cache = object()
    params = dict(core.base_params)
    params["ROI_MASK"] = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    received_masks: list[np.ndarray] = []

    def _filtered(_detector, _cache, _frame, roi_mask, *, apply_max_detections=True):
        assert apply_max_detections is False
        received_masks.append(roi_mask.copy())
        return [object()], [], [], [], [], []

    monkeypatch.setattr(optimizer_module, "_filter_cached_detections", _filtered)
    monkeypatch.setattr(
        "hydra_suite.core.inference.runner._probe_frame_hw", lambda _path: (4, 4)
    )

    counts = core._validation_detection_counts(params, 0, 1)

    assert counts.tolist() == [1, 1]
    expected = np.array(
        [[255, 255, 0, 0], [255, 255, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    assert len(received_masks) == 2
    np.testing.assert_array_equal(received_masks[0], expected)
    np.testing.assert_array_equal(received_masks[1], expected)


def _uncapped_detection_obb(frame_idx: int = 0) -> OBBResult:
    """Four separated source detections that all survive non-count gates."""

    centroids = np.array(
        [[10.0, 10.0], [40.0, 10.0], [70.0, 10.0], [100.0, 10.0]], np.float32
    )
    corners = np.array(
        [
            [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]],
            [[35.0, 5.0], [45.0, 5.0], [45.0, 15.0], [35.0, 15.0]],
            [[65.0, 5.0], [75.0, 5.0], [75.0, 15.0], [65.0, 15.0]],
            [[95.0, 5.0], [105.0, 5.0], [105.0, 15.0], [95.0, 15.0]],
        ],
        np.float32,
    )
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=np.zeros(4, dtype=np.float32),
        sizes=np.full(4, 100.0, dtype=np.float32),
        shapes=np.ones((4, 2), dtype=np.float32),
        confidences=np.full(4, 0.9, dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, 4),
    )


def test_detection_excess_counts_pre_cap_detections_via_production_filter_path() -> (
    None
):
    """Source excess must see candidates that the tracking cap intentionally hides."""

    class _Cache:
        def read_frame(self, frame_idx):
            return _uncapped_detection_obb(frame_idx)

    core = _optimizer()
    core.cache = _Cache()
    params = {
        **core.base_params,
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_CONFIDENCE_THRESHOLD": 0.0,
        "YOLO_IOU_THRESHOLD": 1.0,
        # OBB extraction's production raw cap is 2 * MAX_TARGETS. Four raw
        # detections are therefore feasible here, while the final replay cap
        # intentionally exposes only the two requested tracking slots.
        "MAX_TARGETS": 2,
    }
    config = inference_config_for_optimizer_params(params)
    raw = _uncapped_detection_obb()

    # The production replay stays capped at the requested two targets.
    capped, _ = filter_for_source(config, raw)
    assert capped.num_detections == 2

    # The validation safeguard traverses that same source-aware filter path,
    # but observes the candidates immediately before its final target cap.
    counts = core._validation_detection_counts(params, 0, 3)
    assert counts is not None
    assert counts.tolist() == [4, 4, 4, 4]
    evaluations = core._validation_evaluations(
        "candidate",
        np.zeros((4, 2, 2), dtype=np.float32),
        np.zeros((4, 2, 2), dtype=np.float32),
        1.0,
        detection_counts=counts,
    )
    assert all(item.metrics["detection_excess_loss"] > 0 for item in evaluations)


def test_validation_rejects_two_shared_cycle_observations_per_region() -> None:
    """Two coincident frames must not become zero-standard-error evidence."""

    forward = np.full((20, 1, 2), np.nan, dtype=np.float32)
    backward = forward.copy()
    for start in range(0, 20, 5):
        forward[start : start + 3, 0] = np.column_stack(
            (np.arange(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        )
        backward[start + 1 : start + 4, 0] = np.column_stack(
            (np.arange(1, 4, dtype=np.float32), np.zeros(3, dtype=np.float32))
        )

    core = _optimizer()
    evaluations = core._validation_evaluations("candidate", forward, backward, 1.0)
    reason = core._validation_temporal_evidence_reason(
        forward, backward, 1.0, evaluations
    )

    # Both directions have a real motion triplet and every region has exactly
    # two shared cycle points. The reporting path itself must now turn that
    # insufficient overlap into non-evidence rather than a perfect estimate.
    assert reason is not None
    assert "shared forward/backward cycle observations" in reason


def test_validation_cycle_support_scales_with_track_slots() -> None:
    """A full-coverage three-point singleton cannot support four track slots."""

    positions = np.full((20, 4, 2), np.nan, dtype=np.float32)
    for start in range(0, 20, 5):
        positions[start : start + 3, 0] = np.column_stack(
            (np.arange(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        )

    core = _optimizer()
    evaluations = core._validation_evaluations(
        "candidate", positions, positions.copy(), 1.0
    )
    reason = core._validation_temporal_evidence_reason(
        positions, positions.copy(), 1.0, evaluations
    )

    assert reason is not None
    assert "requires at least 6 shared forward/backward cycle observations" in reason


def test_lifecycle_maturity_gate_rejects_many_short_observation_fragments() -> None:
    """Summing short fragments across tracks/regions cannot exercise either age."""

    core = _optimizer()
    core.base_params["KALMAN_MATURITY_AGE"] = 4
    core.tuning_config["KALMAN_MATURITY_AGE"] = True
    positions = np.full((80, 3, 2), np.nan, dtype=np.float32)
    for track in range(3):
        for start in range(0, 80, 4):
            positions[start : start + 3, track] = np.column_stack(
                (
                    track * 100.0 + np.arange(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                )
            )

    evaluations = core._validation_evaluations(
        "candidate", positions, positions.copy(), 1.0
    )
    reason = core._validation_temporal_evidence_reason(
        positions,
        positions.copy(),
        1.0,
        evaluations,
        candidate_params={"KALMAN_MATURITY_AGE": 8},
    )

    assert reason is not None
    assert "KALMAN_MATURITY_AGE" in reason


def test_lifecycle_maturity_gate_accepts_baseline_crossing_for_raised_candidate() -> (
    None
):
    """A run that reaches baseline age distinguishes it from a higher candidate age."""

    core = _optimizer()
    core.base_params["KALMAN_MATURITY_AGE"] = 4
    core.tuning_config["KALMAN_MATURITY_AGE"] = True
    positions = np.full((80, 2, 2), np.nan, dtype=np.float32)
    for track in range(2):
        for start in range(0, 80, 5):
            positions[start : start + 4, track] = np.column_stack(
                (
                    track * 100.0 + np.arange(4, dtype=np.float32),
                    np.zeros(4, dtype=np.float32),
                )
            )

    evaluations = core._validation_evaluations(
        "candidate", positions, positions.copy(), 1.0
    )

    assert (
        core._validation_temporal_evidence_reason(
            positions,
            positions.copy(),
            1.0,
            evaluations,
            candidate_params={"KALMAN_MATURITY_AGE": 8},
        )
        is None
    )


def test_lifecycle_lost_gate_requires_a_bracketed_threshold_length_gap() -> None:
    """A bracketed gap below both thresholds leaves a raised value unexercised."""

    core = _optimizer()
    core.base_params["LOST_THRESHOLD_FRAMES"] = 4
    core.tuning_config["LOST_THRESHOLD_FRAMES"] = True
    time = np.arange(80, dtype=np.float32)
    positions = np.stack(
        [
            np.column_stack((time, np.zeros_like(time))),
            np.column_stack((100.0 + time, np.zeros_like(time))),
        ],
        axis=1,
    )
    positions[30:33, 0] = np.nan  # only three missing frames, bracketed by observations

    evaluations = core._validation_evaluations(
        "candidate", positions, positions.copy(), 1.0
    )
    reason = core._validation_temporal_evidence_reason(
        positions,
        positions.copy(),
        1.0,
        evaluations,
        candidate_params={"LOST_THRESHOLD_FRAMES": 8},
    )
    assert reason is not None
    assert "LOST_THRESHOLD_FRAMES" in reason

    # Four missing frames cross the baseline loss transition but not the
    # raised candidate's threshold, so the held-out run distinguishes them.
    positions[30:34, 0] = np.nan
    evaluations = core._validation_evaluations(
        "candidate", positions, positions.copy(), 1.0
    )
    assert (
        core._validation_temporal_evidence_reason(
            positions,
            positions.copy(),
            1.0,
            evaluations,
            candidate_params={"LOST_THRESHOLD_FRAMES": 8},
        )
        is None
    )


def test_production_validation_does_not_promote_unexercised_lifecycle_candidate(
    monkeypatch,
) -> None:
    """An otherwise cleaner candidate cannot bypass its maturity evidence gate."""

    time = np.arange(80, dtype=np.float32)
    smooth = np.stack(
        [
            np.column_stack((track * 100.0 + time, np.zeros_like(time)))
            for track in range(3)
        ],
        axis=1,
    )
    baseline_backward = smooth.copy()
    baseline_backward[:, :, 0] += 1.0
    fragments = np.full((80, 3, 2), np.nan, dtype=np.float32)
    for track in range(3):
        for start in range(0, 80, 4):
            fragments[start : start + 3, track] = np.column_stack(
                (
                    track * 100.0 + np.arange(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                )
            )
    _install_fake_replay(
        monkeypatch,
        (smooth, baseline_backward),
        (fragments, fragments.copy()),
    )
    core = _optimizer()
    core.base_params["MAX_TARGETS"] = 3
    core.base_params["KALMAN_MATURITY_AGE"] = 4
    core.tuning_config["KALMAN_MATURITY_AGE"] = True
    baseline = _result({}, baseline=True)
    candidate = _result({"W_POSITION": 2.0, "KALMAN_MATURITY_AGE": 8})

    core._production_validate_shortlist([baseline, candidate], (0, 79))

    assert baseline.recommended is True
    assert candidate.recommended is False
    assert "KALMAN_MATURITY_AGE" in candidate.recommendation_reason


def test_production_validation_does_not_promote_unexercised_lost_threshold(
    monkeypatch,
) -> None:
    """A smooth held-out replay alone cannot validate an unseen loss transition."""

    time = np.arange(40, dtype=np.float32)
    smooth = np.stack([time, np.zeros_like(time)], axis=1)[:, None, :]
    baseline_backward = smooth.copy()
    baseline_backward[:, 0, 0] += 1.0
    _install_fake_replay(
        monkeypatch,
        (smooth, baseline_backward),
        (smooth, smooth.copy()),
    )
    core = _optimizer()
    core.base_params["LOST_THRESHOLD_FRAMES"] = 1
    core.tuning_config["LOST_THRESHOLD_FRAMES"] = True
    baseline = _result({}, baseline=True)
    candidate = _result({"W_POSITION": 2.0, "LOST_THRESHOLD_FRAMES": 4})

    core._production_validate_shortlist([baseline, candidate], (0, 39))

    assert baseline.recommended is True
    assert candidate.recommended is False
    assert "LOST_THRESHOLD_FRAMES" in candidate.recommendation_reason


def test_optimizer_caches_native_frame_roi_normalization(monkeypatch) -> None:
    """Hundreds of proposal replays must not repeatedly reopen the video."""

    core = _optimizer()
    raw_mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    native_mask = np.ones((4, 4), dtype=np.uint8)
    calls: list[tuple[np.ndarray, str]] = []

    def _normalize(mask, video_path):
        calls.append((mask, video_path))
        return native_mask

    monkeypatch.setattr(optimizer_module, "frame_space_roi_mask", _normalize)

    assert core._frame_space_roi_mask_for_params({"ROI_MASK": raw_mask}) is native_mask
    assert core._frame_space_roi_mask_for_params({"ROI_MASK": raw_mask}) is native_mask
    assert len(calls) == 1
    assert calls[0][0] is raw_mask
    assert calls[0][1] == "clip.mp4"


def test_proposal_loop_uses_cached_native_frame_roi_mask(monkeypatch) -> None:
    """Cheap forward/backward proposals receive the same ROI as replay."""

    core = _optimizer()
    core.cache = object()
    core._pose_run_context = (None, [], [], [], False)
    core._pose_frame_cache = {}
    native_mask = np.ones((4, 4), dtype=np.uint8)
    received_masks: list[np.ndarray] = []

    monkeypatch.setattr(
        core, "_frame_space_roi_mask_for_params", lambda _params: native_mask
    )

    def _filtered(_detector, _cache, _frame, roi_mask):
        received_masks.append(roi_mask)
        return [], [], [], [], [], []

    monkeypatch.setattr(optimizer_module, "_filter_cached_detections", _filtered)
    core._run_tracking_loop(
        {**core.base_params, "ROI_MASK": np.zeros((2, 2), dtype=np.uint8)},
        start_frame=0,
        end_frame=1,
    )

    assert received_masks == [native_mask, native_mask]


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
