from __future__ import annotations

import importlib
import types

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub


class _StubDetectionFilter:
    def __init__(self, params):
        self.params = params

    def filter_raw_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb,
        roi_mask=None,
        detection_ids=None,
        heading_hints=None,
        heading_confidences=None,
        directed_mask=None,
    ):
        return (
            meas,
            sizes,
            shapes,
            confidences,
            obb,
            detection_ids or [],
            heading_hints or [],
            heading_confidences or [],
            directed_mask or [],
        )


class _StubKalmanFilterManager:
    last_instance = None

    def __init__(self, n_targets, params):
        self.X = np.zeros((n_targets, 5), dtype=np.float32)
        self.P = np.tile(np.eye(5, dtype=np.float32)[None, :, :], (n_targets, 1, 1))
        self.corrected_measurements = []
        self.predict_count = 0
        _StubKalmanFilterManager.last_instance = self

    def predict(self):
        self.predict_count += 1
        return self.X

    def correct(self, track_idx, measurement):
        measurement = np.asarray(measurement, dtype=np.float32)
        self.corrected_measurements.append((int(track_idx), measurement.copy()))
        self.X[track_idx, :3] = measurement

    def initialize_filter(self, track_idx, state):
        self.X[track_idx] = np.asarray(state, dtype=np.float32)

    def get_mahalanobis_matrices(self):
        return np.tile(np.eye(3, dtype=np.float32)[None, :, :], (len(self.X), 1, 1))

    def get_position_uncertainties(self):
        return np.ones(len(self.X), dtype=np.float32)


class _StubTrackAssigner:
    last_association_data = None
    last_meas_ori_directed = None
    last_track_arena = "unset"
    last_meas_arena = "unset"

    def __init__(self, params):
        self.params = params
        self.track_arena = None

    def set_track_arena(self, track_arena) -> None:
        # Mirrors the real TrackAssigner: the optimizer MUST install the
        # per-slot arena mapping, or it tunes against an ungated simulation.
        self.track_arena = track_arena
        _StubTrackAssigner.last_track_arena = track_arena

    def compute_cost_matrix(
        self,
        N,
        meas,
        preds,
        shapes,
        kf_manager,
        last_shape_info,
        meas_ori_directed=None,
        association_data=None,
        meas_arena=None,
    ):
        _StubTrackAssigner.last_association_data = association_data
        _StubTrackAssigner.last_meas_arena = meas_arena
        _StubTrackAssigner.last_meas_ori_directed = np.asarray(
            meas_ori_directed, dtype=np.uint8
        )
        return np.zeros((N, len(meas)), dtype=np.float32), {}

    def assign_tracks(self, cost, N, M, meas, *args, **kwargs):
        _StubTrackAssigner.last_assign_meas_arena = kwargs.get("meas_arena", "missing")
        return [0], [0], [], []


class _FakeCache:
    def read_frame(self, frame_idx):
        from hydra_suite.core.inference.result import OBBResult

        return OBBResult(
            frame_idx=0,
            centroids=np.array([[10.0, 20.0]], dtype=np.float32),
            angles=np.array([0.0], dtype=np.float32),
            sizes=np.array([50.0], dtype=np.float32),
            shapes=np.array([[50.0, 1.0]], dtype=np.float32),
            confidences=np.array([0.95], dtype=np.float32),
            corners=np.array([[[0, 0], [4, 0], [4, 2], [0, 2]]], dtype=np.float32),
            detection_ids=np.array([101], dtype=np.int64),
        )


def _load_optimizer_module():
    detection_config = importlib.import_module(
        "hydra_suite.core.tracking.optimization.detection_config"
    )
    parameter_contract = importlib.import_module(
        "hydra_suite.core.tracking.optimization.parameter_contract"
    )
    production_replay = importlib.import_module(
        "hydra_suite.core.tracking.optimization.production_replay"
    )
    unlabeled_scoring = importlib.import_module(
        "hydra_suite.core.tracking.optimization.unlabeled_scoring"
    )
    hydra_suite_pkg = types.ModuleType("hydra_suite")
    hydra_suite_pkg.__path__ = []
    core_pkg = types.ModuleType("hydra_suite.core")
    core_pkg.__path__ = []
    core_tracking = types.ModuleType("hydra_suite.core.tracking")
    core_tracking.__path__ = []
    optimization_pkg = types.ModuleType("hydra_suite.core.tracking.optimization")
    optimization_pkg.__path__ = []
    data_pkg = types.ModuleType("hydra_suite.data")
    data_pkg.__path__ = []

    qtcore = types.ModuleType("PySide6.QtCore")

    class Signal:
        def __init__(self, *args, **kwargs):
            self.emissions = []

        def emit(self, *args, **kwargs):
            self.emissions.append((args, kwargs))

    class QThread:
        def __init__(self, parent=None):
            self.parent = parent

    qtcore.Signal = Signal
    qtcore.QThread = QThread

    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore

    optuna = types.ModuleType("optuna")
    optuna.logging = types.SimpleNamespace(WARNING=0, set_verbosity=lambda *_args: None)
    optuna.samplers = types.SimpleNamespace(
        QMCSampler=object,
        RandomSampler=object,
        GPSampler=object,
        TPESampler=object,
    )
    optuna.create_study = lambda **_kwargs: None

    assigner = types.ModuleType("hydra_suite.core.assigners.hungarian")
    assigner.TrackAssigner = _StubTrackAssigner

    core_detectors = types.ModuleType("hydra_suite.core.detectors")
    core_detectors.__path__ = []
    core_detectors.DetectionFilter = _StubDetectionFilter

    kalman = types.ModuleType("hydra_suite.core.filters.kalman")
    kalman.KalmanFilterManager = _StubKalmanFilterManager

    # Identity sub-package stubs (optimizer imports from canonical locations).
    # These are pure-Python modules; load the real implementations.
    core_identity = types.ModuleType("hydra_suite.core.individual")
    core_identity.__path__ = []

    identity_geometry = load_src_module(
        "hydra_suite/core/individual/geometry.py",
        "identity_geometry_for_optimizer_test",
    )

    pose_pkg = types.ModuleType("hydra_suite.core.individual.pose")
    pose_pkg.__path__ = []
    pose_features = load_src_module(
        "hydra_suite/core/individual/pose/features.py",
        "pose_features_for_optimizer_test",
    )

    stubs = {
        "cv2": make_cv2_stub(),
        "optuna": optuna,
        "PySide6": pyside,
        "PySide6.QtCore": qtcore,
        "hydra_suite": hydra_suite_pkg,
        "hydra_suite.core": core_pkg,
        "hydra_suite.core.tracking": core_tracking,
        "hydra_suite.core.tracking.optimization": optimization_pkg,
        "hydra_suite.core.tracking.optimization.detection_config": detection_config,
        "hydra_suite.core.tracking.optimization.parameter_contract": parameter_contract,
        "hydra_suite.core.tracking.optimization.production_replay": production_replay,
        "hydra_suite.core.tracking.optimization.unlabeled_scoring": unlabeled_scoring,
        "hydra_suite.data": data_pkg,
        "hydra_suite.core.assigners.hungarian": assigner,
        "hydra_suite.core.detectors": core_detectors,
        "hydra_suite.core.filters.kalman": kalman,
        "hydra_suite.core.individual": core_identity,
        "hydra_suite.core.individual.geometry": identity_geometry,
        "hydra_suite.core.individual.pose": pose_pkg,
        "hydra_suite.core.individual.pose.features": pose_features,
    }

    return load_src_module(
        "hydra_suite/core/tracking/optimization/optimizer.py",
        "tracking_optimizer_under_test",
        stubs=stubs,
    )


def test_optimizer_replay_uses_directed_heading_for_assignment_and_kf() -> None:
    mod = _load_optimizer_module()
    optimizer = mod.TrackingOptimizerCore(
        video_path="dummy.mp4",
        detection_cache_path="dummy.npz",
        start_frame=0,
        end_frame=0,
        base_params={},
        tuning_config={},
    )
    optimizer.cache = _FakeCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}
    optimizer._stop_requested = False

    # Detection caches are now bare OBBResults (no per-detection heading hints);
    # the directed heading comes from the pose pipeline. Inject a strong,
    # reliable pose heading of 1.25 so the directed-heading override fires.
    def _strong_pose_features(*_args, **_kwargs):
        return (
            [
                np.asarray(
                    [[0.0, 0.0, 0.99], [1.0, 0.0, 0.99], [2.0, 0.0, 0.99]],
                    dtype=np.float32,
                )
            ],
            [0.99],
            [np.float32(1.25)],
        )

    original_compute = mod._compute_pose_features_for_frame
    mod._compute_pose_features_for_frame = _strong_pose_features
    try:
        params = {
            "MAX_TARGETS": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "RESIZE_FACTOR": 1.0,
            "LOST_THRESHOLD_FRAMES": 5,
            "POSE_OVERRIDES_HEADTAIL": True,
            "POSE_DIRECTION_MIN_VALID_KEYPOINTS": 1,
            "POSE_DIRECTION_MIN_VISIBILITY": 0.0,
        }

        score, _, _ = optimizer._run_tracking_loop(params)
    finally:
        mod._compute_pose_features_for_frame = original_compute

    assert np.isfinite(score)
    np.testing.assert_allclose(
        _StubTrackAssigner.last_association_data["detection_pose_heading"],
        np.array([1.25], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(_StubTrackAssigner.last_meas_ori_directed, [1])
    _, corrected = _StubKalmanFilterManager.last_instance.corrected_measurements[0]
    assert corrected[2] == np.float32(1.25)


def _install_cache_admission_probe(
    mod,
    *,
    all_valid: bool = True,
    covers_range: bool = True,
    missing_frames: list[int] | None = None,
):
    """Install the shared replay-admission result and return its call log."""

    calls: list[tuple[object, ...]] = []

    def _admit(cache_path, video_path, params, start_frame, end_frame):
        calls.append((cache_path, video_path, params, start_frame, end_frame))
        return types.SimpleNamespace(
            cache_set_valid=all_valid,
            detection_range_covered=covers_range,
            missing_frames=tuple(missing_frames or []),
        )

    mod.inspect_replay_cache_admission = _admit
    return calls


def _record_direct_cache_open(calls: list[bool]):
    """Return an opener that proves rejected admission did not start replay."""

    def _open(*_args, **_kwargs):
        calls.append(True)

    return _open


def test_optimizer_opens_full_replay_cache_read_only() -> None:
    mod = _load_optimizer_module()
    cache_calls = []

    class _ReadableCache:
        def is_valid(self):
            return True

        def covers_frame_range(self, start_frame, end_frame):
            return (start_frame, end_frame) == (2, 4)

    def _open_caches(*args, **kwargs):
        cache_calls.append((args, kwargs))
        return types.SimpleNamespace(
            detection=_ReadableCache(), set_manifest_valid=True
        )

    mod._open_caches = _open_caches
    mod.inference_config_for_optimizer_params = lambda _params: object()
    mod.video_signature = lambda _path: "video-signature"
    admission_calls = _install_cache_admission_probe(mod)
    optimizer = mod.TrackingOptimizerCore(
        video_path="dummy.mp4",
        detection_cache_path="cache-dir",
        start_frame=2,
        end_frame=4,
        base_params={},
        tuning_config={},
    )

    assert optimizer._open_and_validate_cache()
    assert cache_calls[0][1]["read_only"] is True
    assert len(admission_calls) == 1
    cache_path, video_path, admission_params, start_frame, end_frame = admission_calls[
        0
    ]
    assert str(cache_path) == "cache-dir"
    assert video_path == "dummy.mp4"
    assert admission_params == {}
    assert (start_frame, end_frame) == (2, 4)


def test_optimizer_rejects_missing_downstream_replay_evidence_before_search() -> None:
    """A valid detection member cannot admit a replay missing a pose/CNN/etc. cache."""
    mod = _load_optimizer_module()
    direct_open_calls = []
    errors = []
    mod._open_caches = _record_direct_cache_open(direct_open_calls)
    mod.inference_config_for_optimizer_params = lambda _params: object()
    admission_calls = _install_cache_admission_probe(mod, all_valid=False)
    optimizer = mod.TrackingOptimizerCore(
        "dummy.mp4", "cache-dir", 2, 4, {}, {}, error_cb=errors.append
    )

    assert not optimizer._open_and_validate_cache()
    assert direct_open_calls == []
    assert optimizer.cache is None
    assert "Every enabled inference stage" in errors[0]
    assert len(admission_calls) == 1


def test_optimizer_rejects_corrupt_downstream_replay_evidence_before_search() -> None:
    mod = _load_optimizer_module()
    direct_open_calls = []
    errors = []
    mod._open_caches = _record_direct_cache_open(direct_open_calls)
    mod.inference_config_for_optimizer_params = lambda _params: object()
    admission_calls = _install_cache_admission_probe(mod, all_valid=False)
    optimizer = mod.TrackingOptimizerCore(
        "dummy.mp4", "cache-dir", 2, 4, {}, {}, error_cb=errors.append
    )

    assert not optimizer._open_and_validate_cache()
    assert direct_open_calls == []
    assert optimizer.cache is None
    assert "Replay evidence cache is incomplete or incompatible" in errors[0]
    assert len(admission_calls) == 1


def test_optimizer_rejects_mixed_downstream_replay_coverage_before_search() -> None:
    mod = _load_optimizer_module()
    direct_open_calls = []
    errors = []
    mod._open_caches = _record_direct_cache_open(direct_open_calls)
    mod.inference_config_for_optimizer_params = lambda _params: object()
    admission_calls = _install_cache_admission_probe(mod, all_valid=False)
    optimizer = mod.TrackingOptimizerCore(
        "dummy.mp4", "cache-dir", 2, 4, {}, {}, error_cb=errors.append
    )

    assert not optimizer._open_and_validate_cache()
    assert direct_open_calls == []
    assert optimizer.cache is None
    assert "Replay evidence cache is incomplete or incompatible" in errors[0]
    assert len(admission_calls) == 1


def test_optimizer_rejects_requested_range_after_full_cache_admission() -> None:
    mod = _load_optimizer_module()
    direct_open_calls = []
    errors = []
    mod._open_caches = _record_direct_cache_open(direct_open_calls)
    mod.inference_config_for_optimizer_params = lambda _params: object()
    admission_calls = _install_cache_admission_probe(
        mod, covers_range=False, missing_frames=[3]
    )
    optimizer = mod.TrackingOptimizerCore(
        "dummy.mp4", "cache-dir", 2, 4, {}, {}, error_cb=errors.append
    )

    assert not optimizer._open_and_validate_cache()
    assert direct_open_calls == []
    assert "Missing: [3]" in errors[0]
    assert len(admission_calls) == 1


def test_optimizer_replay_skips_directed_pose_heading_when_pose_is_weak() -> None:
    mod = _load_optimizer_module()

    # Inherits _FakeCache.read_frame (bare OBBResult); the "weak pose" signal
    # comes from the _compute_pose_features_for_frame override below.
    class _WeakPoseOnlyCache(_FakeCache):
        pass

    optimizer = mod.TrackingOptimizerCore(
        video_path="dummy.mp4",
        detection_cache_path="dummy.npz",
        start_frame=0,
        end_frame=0,
        base_params={},
        tuning_config={},
    )
    optimizer.cache = _WeakPoseOnlyCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}
    optimizer._stop_requested = False

    original_compute = mod._compute_pose_features_for_frame

    def _weak_pose_features(*args, **kwargs):
        _kpts, _vis, _heading = original_compute(*args, **kwargs)
        return (
            [np.asarray([[0.0, 0.0, 0.95], [1.0, 0.0, 0.95]], dtype=np.float32)],
            [0.9],
            _heading,
        )

    mod._compute_pose_features_for_frame = _weak_pose_features
    try:
        params = {
            "MAX_TARGETS": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "RESIZE_FACTOR": 1.0,
            "LOST_THRESHOLD_FRAMES": 5,
            "POSE_OVERRIDES_HEADTAIL": True,
        }

        score, _, _ = optimizer._run_tracking_loop(params)
    finally:
        mod._compute_pose_features_for_frame = original_compute

    assert np.isfinite(score)
    np.testing.assert_array_equal(_StubTrackAssigner.last_meas_ori_directed, [0])
    _, corrected = _StubKalmanFilterManager.last_instance.corrected_measurements[0]
    assert corrected[2] == np.float32(0.0)


class _ParamsFilter:
    """Minimal det_filter stand-in exposing only `.params` (Correction 21)."""

    def __init__(self, params):
        self.params = params


class _OBBFrameCache:
    def __init__(self, obb_result):
        self._obb_result = obb_result

    def read_frame(self, frame_idx):
        return self._obb_result


class _LegacyTupleCache:
    """Cache stub that still returns the retired legacy 12-tuple frame shape."""

    def read_frame(self, frame_idx):
        return (
            [np.array([10.0, 20.0, 0.0], dtype=np.float32)],
            [50.0],
            [(50.0, 1.0)],
            [0.95],
            [np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=np.float32)],
            [101],
            [0.0],
            [0.0],
            [0],
            None,
            None,
            None,
        )


def _make_obb_result():
    from hydra_suite.core.inference.result import OBBResult

    return OBBResult(
        frame_idx=0,
        centroids=np.array([[10.0, 20.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([50.0], dtype=np.float32),
        shapes=np.array([[50.0, 1.0]], dtype=np.float32),
        confidences=np.array([0.95], dtype=np.float32),
        corners=np.array([[[0, 0], [4, 0], [4, 2], [0, 2]]], dtype=np.float32),
        detection_ids=np.array([101], dtype=np.int64),
    )


def test_filter_cached_detections_returns_for_obb_result_frame() -> None:
    from hydra_suite.core.tracking.optimization.optimizer import (
        _filter_cached_detections,
    )

    det_filter = _ParamsFilter({"DETECTION_CONFIDENCE": 0.0})
    cache = _OBBFrameCache(_make_obb_result())

    meas, shapes, confs, detection_ids, headtail_hints, headtail_directed = (
        _filter_cached_detections(det_filter, cache, 0, roi_mask=None)
    )

    assert len(meas) == 1
    assert len(shapes) == 1
    assert len(confs) == 1
    assert detection_ids == [101]
    assert headtail_hints == []
    assert headtail_directed == []


def test_filter_cached_detections_raises_for_legacy_tuple_frame() -> None:
    from hydra_suite.core.tracking.optimization.optimizer import (
        _filter_cached_detections,
    )

    det_filter = _ParamsFilter({"DETECTION_CONFIDENCE": 0.0})
    cache = _LegacyTupleCache()

    try:
        _filter_cached_detections(det_filter, cache, 0, roi_mask=None)
    except TypeError as exc:
        assert "OBBResult" in str(exc)
    else:
        raise AssertionError("expected TypeError for legacy tuple cache frame")


def test_preview_filter_cached_detections_returns_for_obb_result_frame() -> None:
    from hydra_suite.core.tracking.optimization.optimizer_workers import (
        _preview_filter_cached_detections,
    )

    det_filter = _ParamsFilter({"DETECTION_CONFIDENCE": 0.0})
    cache = _OBBFrameCache(_make_obb_result())

    meas, shapes, confs, detection_ids, headtail_hints, headtail_directed = (
        _preview_filter_cached_detections(det_filter, cache, 0, roi_mask=None)
    )

    assert len(meas) == 1
    assert len(shapes) == 1
    assert len(confs) == 1
    assert detection_ids == [101]
    assert headtail_hints == []
    assert headtail_directed == []


def test_optimizer_replay_advances_kalman_across_empty_frames() -> None:
    mod = _load_optimizer_module()

    class SequenceCache:
        def read_frame(self, frame_idx):
            if frame_idx != 1:
                return _make_obb_result()
            obb = _make_obb_result()
            keep = np.zeros(0, dtype=np.int64)
            return type(obb)(
                frame_idx=frame_idx,
                centroids=obb.centroids[keep],
                angles=obb.angles[keep],
                sizes=obb.sizes[keep],
                shapes=obb.shapes[keep],
                confidences=obb.confidences[keep],
                corners=obb.corners[keep],
                detection_ids=obb.detection_ids[keep],
            )

    optimizer = mod.TrackingOptimizerCore("dummy.mp4", "cache-dir", 0, 2, {}, {})
    optimizer.cache = SequenceCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}
    optimizer._run_tracking_loop(
        {
            "MAX_TARGETS": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "RESIZE_FACTOR": 1.0,
            "LOST_THRESHOLD_FRAMES": 5,
        }
    )

    assert _StubKalmanFilterManager.last_instance.predict_count == 3


def test_optimizer_cycle_positions_are_observations_not_hidden_kalman_posteriors() -> (
    None
):
    mod = _load_optimizer_module()

    class OffsetPosteriorKalman(_StubKalmanFilterManager):
        def correct(self, track_idx, measurement):
            super().correct(track_idx, measurement)
            self.X[track_idx, :2] += 100.0

    mod.KalmanFilterManager = OffsetPosteriorKalman
    optimizer = mod.TrackingOptimizerCore("dummy.mp4", "cache-dir", 0, 1, {}, {})
    optimizer.cache = _FakeCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}

    _, _, positions = optimizer._run_tracking_loop(
        {
            "MAX_TARGETS": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "RESIZE_FACTOR": 1.0,
            "LOST_THRESHOLD_FRAMES": 5,
        }
    )

    # This fixture's probe assigner reports the initial detection as a direct
    # match; both frames must still expose the measurement, never the +100 KF
    # posterior injected above.
    np.testing.assert_allclose(positions[0][0], [10.0, 20.0])
    np.testing.assert_allclose(positions[1][0], [10.0, 20.0])
    np.testing.assert_allclose(
        OffsetPosteriorKalman.last_instance.X[0, :2], [110.0, 120.0]
    )


def test_preview_filter_cached_detections_raises_for_legacy_tuple_frame() -> None:
    from hydra_suite.core.tracking.optimization.optimizer_workers import (
        _preview_filter_cached_detections,
    )

    det_filter = _ParamsFilter({"DETECTION_CONFIDENCE": 0.0})
    cache = _LegacyTupleCache()

    try:
        _preview_filter_cached_detections(det_filter, cache, 0, roi_mask=None)
    except TypeError as exc:
        assert "OBBResult" in str(exc)
    else:
        raise AssertionError("expected TypeError for legacy tuple cache frame")


def _cached_filter_helpers():
    from hydra_suite.core.tracking.optimization.optimizer import (
        _filter_cached_detections,
    )
    from hydra_suite.core.tracking.optimization.optimizer_workers import (
        _preview_filter_cached_detections,
    )

    return _filter_cached_detections, _preview_filter_cached_detections


def _assert_cached_filters_match_production(params, raw, roi_mask=None):
    from hydra_suite.core.inference.stages.filtering import filter_for_source
    from hydra_suite.core.tracking.optimization.optimizer import (
        inference_config_for_optimizer_params,
    )

    config = inference_config_for_optimizer_params(params)
    expected, _ = filter_for_source(config, raw, roi_mask)
    for cached_filter in _cached_filter_helpers():
        result = cached_filter(
            _ParamsFilter(params), _OBBFrameCache(raw), raw.frame_idx, roi_mask
        )
        meas, shapes, confidences, detection_ids, *_ = result
        np.testing.assert_allclose(
            np.asarray(meas, dtype=np.float32),
            np.column_stack((expected.centroids, expected.angles)),
        )
        np.testing.assert_allclose(np.asarray(shapes), expected.shapes)
        np.testing.assert_allclose(np.asarray(confidences), expected.confidences)
        assert detection_ids == expected.detection_ids.tolist()
    return expected


def _make_filtering_obb(centroids, confidences, sizes, corners):
    from hydra_suite.core.inference.result import OBBResult

    count = len(centroids)
    return OBBResult(
        frame_idx=7,
        centroids=np.asarray(centroids, dtype=np.float32),
        angles=np.zeros(count, dtype=np.float32),
        sizes=np.asarray(sizes, dtype=np.float32),
        shapes=np.ones((count, 2), dtype=np.float32),
        confidences=np.asarray(confidences, dtype=np.float32),
        corners=np.asarray(corners, dtype=np.float32),
        detection_ids=np.arange(700, 700 + count, dtype=np.int64),
    )


def test_cached_filters_match_production_confidence_and_iou_gates() -> None:
    raw = _make_filtering_obb(
        [[20, 20], [100, 100], [102, 100]],
        [0.2, 0.8, 0.9],
        [100, 100, 100],
        [
            [[15, 15], [25, 15], [25, 25], [15, 25]],
            [[90, 90], [110, 90], [110, 110], [90, 110]],
            [[92, 90], [112, 90], [112, 110], [92, 110]],
        ],
    )
    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        "YOLO_IOU_THRESHOLD": 0.7,
        "MAX_TARGETS": 4,
    }

    expected = _assert_cached_filters_match_production(params, raw)
    assert expected.detection_ids.tolist() == [702]


def test_cached_filters_match_production_roi_and_max_count_gates() -> None:
    raw = _make_filtering_obb(
        [[10, 10], [30, 30], [50, 50], [90, 90]],
        # Distinct confidences: the final MAX_TARGETS cut is confidence-ordered,
        # so tied confidences would make the survivor set a tie-break artifact
        # rather than a property worth asserting. Sizes are deliberately
        # ANTI-correlated with confidence so a size-ordered cap and a
        # confidence-ordered cap cannot agree by accident.
        [0.50, 0.90, 0.80, 0.99],
        [100, 30, 20, 10],
        [
            [[5, 5], [15, 5], [15, 15], [5, 15]],
            [[25, 25], [35, 25], [35, 35], [25, 35]],
            [[45, 45], [55, 45], [55, 55], [45, 55]],
            [[85, 85], [95, 85], [95, 95], [85, 95]],
        ],
    )
    roi_mask = np.zeros((80, 80), dtype=np.uint8)
    roi_mask[:60, :60] = 1
    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_CONFIDENCE_THRESHOLD": 0.0,
        "YOLO_IOU_THRESHOLD": 1.0,
        "MAX_TARGETS": 2,
    }

    expected = _assert_cached_filters_match_production(params, raw, roi_mask)
    # id 703 is the most confident (0.99) but sits outside the ROI, so the ROI
    # gate drops it before the cap. Of the survivors (700=0.50, 701=0.90,
    # 702=0.80) the cap keeps the two most confident: 701 and 702.
    assert sorted(expected.detection_ids.tolist()) == [701, 702]


def test_cached_filters_preserve_bgsub_nan_confidence_detections() -> None:
    raw = _make_filtering_obb(
        [[10, 10], [30, 30]],
        [float("nan"), float("nan")],
        [10, 30],
        [
            [[5, 5], [15, 5], [15, 15], [5, 15]],
            [[25, 25], [35, 25], [35, 35], [25, 35]],
        ],
    )
    params = {
        "DETECTION_METHOD": "background_subtraction",
        "YOLO_CONFIDENCE_THRESHOLD": 0.99,
        "MAX_TARGETS": 1,
    }

    expected = _assert_cached_filters_match_production(params, raw)
    assert expected.detection_ids.tolist() == [700, 701]
