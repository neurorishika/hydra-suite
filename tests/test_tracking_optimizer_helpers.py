from __future__ import annotations

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
        _StubKalmanFilterManager.last_instance = self

    def predict(self):
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

    def __init__(self, params):
        self.params = params

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
    ):
        _StubTrackAssigner.last_association_data = association_data
        _StubTrackAssigner.last_meas_ori_directed = np.asarray(
            meas_ori_directed, dtype=np.uint8
        )
        return np.zeros((N, len(meas)), dtype=np.float32), {}

    def assign_tracks(self, cost, N, M, meas, *args, **kwargs):
        return [0], [0], [], []


class _FakeCache:
    def read_frame(self, frame_idx):
        return (
            [np.array([10.0, 20.0, 0.0], dtype=np.float32)],
            [50.0],
            [(50.0, 1.0)],
            [0.95],
            [np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=np.float32)],
            [101],
            [1.25],
            [0.88],
            [1],
            None,
            None,
            None,
        )


def _load_optimizer_module():
    hydra_suite_pkg = types.ModuleType("hydra_suite")
    hydra_suite_pkg.__path__ = []
    core_pkg = types.ModuleType("hydra_suite.core")
    core_pkg.__path__ = []
    core_tracking = types.ModuleType("hydra_suite.core.tracking")
    core_tracking.__path__ = []
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

    detection_cache = types.ModuleType("hydra_suite.data.detection_cache")
    detection_cache.DetectionCache = object

    # Identity sub-package stubs (optimizer imports from canonical locations).
    # These are pure-Python modules; load the real implementations.
    core_identity = types.ModuleType("hydra_suite.core.identity")
    core_identity.__path__ = []

    identity_geometry = load_src_module(
        "hydra_suite/core/identity/geometry.py",
        "identity_geometry_for_optimizer_test",
    )

    pose_pkg = types.ModuleType("hydra_suite.core.identity.pose")
    pose_pkg.__path__ = []
    pose_features = load_src_module(
        "hydra_suite/core/identity/pose/features.py",
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
        "hydra_suite.data": data_pkg,
        "hydra_suite.core.assigners.hungarian": assigner,
        "hydra_suite.core.detectors": core_detectors,
        "hydra_suite.core.filters.kalman": kalman,
        "hydra_suite.data.detection_cache": detection_cache,
        "hydra_suite.core.identity": core_identity,
        "hydra_suite.core.identity.geometry": identity_geometry,
        "hydra_suite.core.identity.pose": pose_pkg,
        "hydra_suite.core.identity.pose.features": pose_features,
    }

    return load_src_module(
        "hydra_suite/core/tracking/optimization/optimizer.py",
        "tracking_optimizer_under_test",
        stubs=stubs,
    )


def test_optimizer_replay_uses_directed_heading_for_assignment_and_kf() -> None:
    mod = _load_optimizer_module()
    optimizer = mod.TrackingOptimizer(
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

    params = {
        "MAX_TARGETS": 1,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "LOST_THRESHOLD_FRAMES": 5,
        "POSE_OVERRIDES_HEADTAIL": True,
    }

    score, _, _ = optimizer._run_tracking_loop(params)

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


def test_optimizer_replay_skips_directed_pose_heading_when_pose_is_weak() -> None:
    mod = _load_optimizer_module()

    class _WeakPoseOnlyCache(_FakeCache):
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

    optimizer = mod.TrackingOptimizer(
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
