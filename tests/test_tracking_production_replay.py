from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runner import _open_caches, video_signature
from hydra_suite.core.tracking.optimization.detection_config import (
    inference_config_for_optimizer_params,
)
from hydra_suite.core.tracking.optimization.production_replay import (
    ProductionReplayEvaluator,
    cache_directory,
    trajectories_to_positions,
)
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def test_cache_directory_accepts_directory_or_detection_member(tmp_path):
    assert cache_directory(tmp_path / "cache") == tmp_path / "cache"
    assert cache_directory(tmp_path / "cache" / "detection.npz") == tmp_path / "cache"


def test_trajectories_to_positions_preserves_only_observed_measurements():
    frames, positions = trajectories_to_positions(
        [
            [(10.0, 20.0, 0.0, 4), (12.0, 22.0, 0.0, 6)],
            [(30.0, 40.0, 0.0, 5)],
        ],
        start_frame=4,
        end_frame=6,
        n_tracks=2,
    )

    np.testing.assert_array_equal(frames, [4, 5, 6])
    np.testing.assert_allclose(positions[0, 0], [10.0, 20.0])
    assert np.isnan(positions[1, 0]).all()
    np.testing.assert_allclose(positions[1, 1], [30.0, 40.0])
    np.testing.assert_allclose(positions[2, 0], [12.0, 22.0])


def test_production_replay_is_read_only_and_uses_requested_direction(tmp_path):
    instances = []

    class FakeEngine:
        def __init__(self, _video_path, **kwargs):
            self.kwargs = kwargs
            self.params = None
            instances.append(self)

        def set_parameters(self, params):
            self.params = params

        def run_tracking(self):
            trajectories = [[(1.0, 2.0, 0.0, 7), (3.0, 4.0, 0.0, 8)]]
            self.kwargs["on_finished"](True, [], trajectories)

        def stop(self):
            raise AssertionError("unexpected stop")

    evaluator = ProductionReplayEvaluator(
        "clip.mp4",
        str(tmp_path / "cache" / "detection.npz"),
        7,
        8,
        engine_factory=FakeEngine,
    )
    result = evaluator.run({"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0}, reverse=True)

    assert result.success
    np.testing.assert_allclose(result.positions[:, 0], [[1.0, 2.0], [3.0, 4.0]])
    engine = instances[0]
    assert engine.kwargs["backward_mode"] is True
    assert engine.kwargs["preview_mode"] is False
    assert engine.kwargs["cache_read_only_replay"] is True
    assert engine.kwargs["inference_cache_dir"] == tmp_path / "cache"
    assert engine.params["START_FRAME"] == 7
    assert engine.params["END_FRAME"] == 8
    assert engine.params["VISUALIZATION_FREE_MODE"] is True


def test_production_replay_prerolls_without_scoring_preroll_observations(tmp_path):
    instances = []

    class FakeEngine:
        def __init__(self, _video_path, **kwargs):
            self.kwargs = kwargs
            self.params = None
            # Mirrors the engine's append-only replay sink: the first two
            # points warm state, while only the tail belongs in the score.
            self.replay_observations = [
                [
                    (1.0, 2.0, 0.0, 3),
                    (2.0, 2.0, 0.0, 4),
                    (3.0, 2.0, 0.0, 5),
                    (4.0, 2.0, 0.0, 6),
                ]
            ]
            instances.append(self)

        def set_parameters(self, params):
            self.params = params

        def run_tracking(self):
            # The legacy callback contains only the final trajectory epoch.
            self.kwargs["on_finished"](True, [], [[(4.0, 2.0, 0.0, 6)]])

    evaluator = ProductionReplayEvaluator(
        "clip.mp4",
        str(tmp_path / "cache" / "detection.npz"),
        5,
        6,
        pre_roll_start=3,
        engine_factory=FakeEngine,
    )
    result = evaluator.run(
        {"MAX_TARGETS": 1, "START_FRAME": 0, "END_FRAME": 9}, reverse=False
    )

    assert result.success
    np.testing.assert_array_equal(result.frame_indices, [5, 6])
    np.testing.assert_allclose(result.positions[:, 0], [[3.0, 2.0], [4.0, 2.0]])
    engine = instances[0]
    assert engine.params["START_FRAME"] == 3
    assert engine.params["END_FRAME"] == 6
    assert engine.kwargs["inference_cache_provenance_params"]["START_FRAME"] == 0
    assert engine.kwargs["inference_cache_provenance_params"]["END_FRAME"] == 9


def test_production_replay_forwards_per_frame_cancellation_token(tmp_path):
    captured = {}

    class FakeEngine:
        def __init__(self, _video_path, **kwargs):
            captured.update(kwargs)

        def set_parameters(self, _params):
            pass

        def run_tracking(self):
            captured["on_finished"](True, [], [])

    def should_stop():
        return True

    evaluator = ProductionReplayEvaluator(
        "clip.mp4",
        str(tmp_path / "cache"),
        0,
        1,
        engine_factory=FakeEngine,
        should_stop=should_stop,
    )

    evaluator.run({"MAX_TARGETS": 1})

    assert captured["should_stop"] is should_stop


def test_production_replay_reports_density_budget_refusal_as_evidence_failure(
    tmp_path,
):
    """A replay-density admission refusal rejects one candidate without a crash."""

    from hydra_suite.core.tracking.confidence.confidence_density import (
        DensityReplayBudgetExceeded,
        admit_density_map_working_set,
    )

    estimate = admit_density_map_working_set(2, 16, 16, downsample_factor=1)

    class _BudgetRefusingEngine:
        def __init__(self, _video_path, **_kwargs):
            pass

        def set_parameters(self, _params):
            pass

        def run_tracking(self):
            raise DensityReplayBudgetExceeded(estimate, estimate.peak_bytes - 1)

    result = ProductionReplayEvaluator(
        "clip.mp4",
        str(tmp_path / "cache"),
        0,
        1,
        engine_factory=_BudgetRefusingEngine,
    ).run({"MAX_TARGETS": 1})

    assert not result.success
    assert result.error is not None
    assert "AUTOTUNE_DENSITY_MAX_BYTES" in result.error


def test_production_replay_propagates_density_cancellation(tmp_path):
    """A density stop is cancellation, not a nonfatal rejected candidate."""

    from hydra_suite.core.tracking.confidence.confidence_density import (
        ConfidenceDensityCancelled,
    )

    class _CancelledDensityEngine:
        def __init__(self, _video_path, **_kwargs):
            pass

        def set_parameters(self, _params):
            pass

        def run_tracking(self):
            raise ConfidenceDensityCancelled("density replay was cancelled")

    evaluator = ProductionReplayEvaluator(
        "clip.mp4",
        str(tmp_path / "cache"),
        0,
        1,
        engine_factory=_CancelledDensityEngine,
    )
    with pytest.raises(ConfidenceDensityCancelled):
        evaluator.run({"MAX_TARGETS": 1})


def test_production_replay_rejects_missing_target_count(tmp_path):
    evaluator = ProductionReplayEvaluator(
        "clip.mp4", str(tmp_path / "cache"), 0, 1, engine_factory=lambda *_a, **_k: None
    )
    with pytest.raises(ValueError, match="MAX_TARGETS"):
        evaluator.run({})


def test_tracking_engine_uses_explicit_replay_cache_directory(tmp_path):
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    core = TrackingEngineCore(
        "clip.mp4",
        inference_cache_dir=tmp_path / "optimizer-cache",
        cache_read_only_replay=True,
    )

    assert core._resolve_cache_dir() == tmp_path / "optimizer-cache"
    assert core.cache_read_only_replay is True
    assert core._resolve_profile_path("forward") is None


def test_real_production_replay_preserves_cache_and_retains_observations_across_respawn(
    tmp_path,
):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32)
    )
    for _ in range(4):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    params = build_engine_params(
        {
            "frame_width": 32,
            "frame_height": 32,
            "detection_method": "background_subtraction",
            "animals_per_arena": 1,
            "reference_body_size": 4.0,
        },
        runtime=RuntimeContext(
            fps=30.0, total_frames=4, frame_width=32, frame_height=32
        ),
    )
    params.update(
        {
            "MIN_DETECTION_COUNTS": 1,
            "MIN_DETECTIONS_TO_START": 1,
            "LOST_THRESHOLD_FRAMES": 1,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_POSE_EXTRACTOR": False,
        }
    )

    cache_dir = tmp_path / "optimizer-cache"
    caches = _open_caches(
        inference_config_for_optimizer_params(params),
        cache_dir,
        video_signature(str(video_path)),
        write_mode="fresh",
    )
    assert caches.detection is not None
    for frame_idx in range(4):
        # Frame 0 bootstraps the lost slot, frame 1 emits the first epoch,
        # frame 2 marks it lost, and frame 3 takes the Phase-3 respawn path.
        # The respawn clears trajectories_full, so only replay_observations can
        # retain both real measurements for replay scoring.
        if frame_idx == 2:
            centroids = np.zeros((0, 2), dtype=np.float32)
            angles = np.zeros(0, dtype=np.float32)
            sizes = np.zeros(0, dtype=np.float32)
            shapes = np.zeros((0, 2), dtype=np.float32)
            confidences = np.zeros(0, dtype=np.float32)
            corners = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            center_x = 8.0 + frame_idx
            centroids = np.array([[center_x, 8.0]], dtype=np.float32)
            angles = np.zeros(1, dtype=np.float32)
            sizes = np.array([16.0], dtype=np.float32)
            shapes = np.array([[16.0, 1.0]], dtype=np.float32)
            confidences = np.array([np.nan], dtype=np.float32)
            corners = np.array(
                [
                    [
                        [center_x - 2, 6.0],
                        [center_x + 2, 6.0],
                        [center_x + 2, 10.0],
                        [center_x - 2, 10.0],
                    ]
                ],
                dtype=np.float32,
            )
        caches.detection.write_frame(
            frame_idx,
            result=OBBResult(
                frame_idx=frame_idx,
                centroids=centroids,
                angles=angles,
                sizes=sizes,
                shapes=shapes,
                confidences=confidences,
                corners=corners,
                detection_ids=OBBResult.make_detection_ids(frame_idx, len(centroids)),
            ),
        )
    caches.close()
    before = {
        path.relative_to(cache_dir): (path.stat().st_mtime_ns, path.read_bytes())
        for path in cache_dir.rglob("*")
        if path.is_file()
    }

    from hydra_suite.core.tracking.worker import TrackingEngineCore

    replay_instances = []

    class _CapturingReplayCore(TrackingEngineCore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            replay_instances.append(self)

    evaluator = ProductionReplayEvaluator(
        str(video_path), str(cache_dir), 0, 3, engine_factory=_CapturingReplayCore
    )
    forward = evaluator.run(params, reverse=False)
    backward = evaluator.run(params, reverse=True)

    # Regression: the cache was produced from the full 0..3 bg-sub range, but
    # held-out validation scores only its tail. The loop gets an unscored
    # pre-roll while the cache key retains full-run provenance.
    heldout = ProductionReplayEvaluator(
        str(video_path), str(cache_dir), 2, 3, pre_roll_start=0
    )
    heldout_forward = heldout.run(params, reverse=False)

    assert forward.success and backward.success
    assert heldout_forward.success
    np.testing.assert_array_equal(heldout_forward.frame_indices, [2, 3])
    # The production loop bootstraps a lost slot from its first detection and
    # starts exporting matched observations on the following frame in each
    # direction. The adapter must preserve that behavior rather than filling
    # the bootstrap frame from hidden Kalman state.
    np.testing.assert_allclose(forward.positions[[1, 3], 0], [[9.0, 8.0], [11.0, 8.0]])
    assert np.isnan(forward.positions[2, 0]).all()
    assert np.isfinite(forward.positions[:, 0]).all(axis=1).sum() == 2
    assert [point[3] for point in replay_instances[0].replay_observations[0]] == [1, 3]
    assert [point[3] for point in replay_instances[0].trajectories_full[0]] == [3]
    assert np.isfinite(backward.positions[:, 0]).all(axis=1).sum() == 2

    # `_cached_detection_iterator` also polls cancellation in normal replay,
    # so deliberately remove that iterator-level check.  This pins the
    # per-frame `run_tracking` loop's own poll: after frame 0, cancellation
    # must stop the core before it consumes frame 1.
    instances = []

    class _NonPollingReplayCore(TrackingEngineCore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def _cached_detection_iterator(
            self, total_frames, start_frame=0, end_frame=None, backward=False
        ):
            for relative_idx in range(total_frames):
                yield None, relative_idx + 1

    def stop_after_first_core_frame():
        return bool(instances and instances[0].frame_count >= 1)

    cancelled = ProductionReplayEvaluator(
        str(video_path),
        str(cache_dir),
        0,
        3,
        engine_factory=_NonPollingReplayCore,
        should_stop=stop_after_first_core_frame,
    ).run(params, reverse=False)

    assert not cancelled.success
    assert instances[0].frame_count == 1
    after = {
        path.relative_to(cache_dir): (path.stat().st_mtime_ns, path.read_bytes())
        for path in cache_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (tmp_path / "clip_logs").exists()
