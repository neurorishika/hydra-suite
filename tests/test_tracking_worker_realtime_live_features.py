from __future__ import annotations

import cv2
import numpy as np
import pytest

import hydra_suite.core.tracking.worker as worker_mod


class _StopAtAssociation(RuntimeError):
    pass


class _FakeProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def phase_start(self, *_args, **_kwargs):
        return None

    def phase_end(self, *_args, **_kwargs):
        return None

    def tick(self, *_args, **_kwargs):
        return None

    def tock(self, *_args, **_kwargs):
        return None

    def add_sample(self, *_args, **_kwargs):
        return None

    def add_phase_time(self, *_args, **_kwargs):
        return None

    def end_frame(self, *_args, **_kwargs):
        return None

    def log_periodic(self, *_args, **_kwargs):
        return None

    def log_final_summary(self, *_args, **_kwargs):
        return None

    def export_summary(self, *_args, **_kwargs):
        return None

    def set_config(self, **_kwargs):
        return None

    def discard_frame_state(self):
        return None

    def reset_interval(self):
        return None

    def notify_frame_index(self, *_args, **_kwargs):
        return None


class _FakeVideoCapture:
    def __init__(self, *_args, **_kwargs):
        self._frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
        self._idx = 0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame.copy()

    def get(self, prop_id):
        if prop_id == worker_mod.cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        if prop_id == worker_mod.cv2.CAP_PROP_FPS:
            return 30.0
        if prop_id == worker_mod.cv2.CAP_PROP_FRAME_WIDTH:
            return self._frames[0].shape[1]
        if prop_id == worker_mod.cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frames[0].shape[0]
        if prop_id == worker_mod.cv2.CAP_PROP_POS_FRAMES:
            return self._idx
        return 0

    def set(self, prop_id, value):
        if prop_id == worker_mod.cv2.CAP_PROP_POS_FRAMES:
            self._idx = int(value)
        return True

    def release(self):
        self._opened = False


class _FakeDetectionCache:
    def __init__(self, *_args, **_kwargs):
        self._cached_frames = set()
        self._frames = {}

    def is_compatible(self):
        return True

    def close(self):
        return None

    def save(self):
        return None

    def get_frame_range(self):
        return 0, 0

    def covers_frame_range(self, *_args, **_kwargs):
        return False

    def get_total_frames(self):
        return len(self._frames)

    def get_missing_frames(self, *_args, **_kwargs):
        return []

    def add_frame(
        self,
        frame_idx,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_detection_ids,
        raw_heading_hints,
        raw_heading_confidences,
        raw_directed_mask,
        canonical_affines=None,
    ):
        self._cached_frames.add(int(frame_idx))
        self._frames[int(frame_idx)] = (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            raw_detection_ids,
            raw_heading_hints,
            raw_heading_confidences,
            raw_directed_mask,
            canonical_affines,
            None,
            None,
        )

    def get_frame(self, frame_idx):
        return self._frames[int(frame_idx)]


class _FakeDetector:
    def detect_objects(self, _frame, _frame_count, return_raw=True, profiler=None):
        raw_meas = [np.array([4.0, 4.0, 0.0], dtype=np.float32)]
        raw_sizes = [np.array([2.0, 1.0], dtype=np.float32)]
        raw_shapes = [np.array([2.0, 1.0], dtype=np.float32)]
        raw_confidences = [0.95]
        raw_obb_corners = [
            np.array(
                [[3.0, 3.5], [5.0, 3.5], [5.0, 4.5], [3.0, 4.5]],
                dtype=np.float32,
            )
        ]
        raw_heading_hints = [0.0]
        raw_heading_confidences = [0.75]
        raw_directed_mask = [0]
        raw_canonical_affines = None
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            None,
            raw_confidences,
            raw_obb_corners,
            raw_heading_hints,
            raw_heading_confidences,
            raw_directed_mask,
            raw_canonical_affines,
        )

    def filter_raw_detections(
        self,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        roi_mask=None,
        detection_ids=None,
        heading_hints=None,
        heading_confidences=None,
        directed_mask=None,
    ):
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            detection_ids or [0],
            heading_hints or [0.0],
            heading_confidences or [0.0],
            directed_mask or [0],
        )


class _FakeKalmanFilterManager:
    def __init__(self, n_targets: int, _params):
        self.X = np.zeros((n_targets, 5), dtype=np.float32)

    def get_predictions(self):
        return np.zeros((len(self.X), 3), dtype=np.float32)


class _UnusedAssigner:
    def __init__(self, params, worker=None):
        self.params = params


def _write_test_video(path, colors, size=(32, 24), fps=5.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    assert writer.isOpened()
    width, height = size
    for color in colors:
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_tracking_worker_realtime_ignores_existing_detection_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _StopOnWrite(RuntimeError):
        pass

    cache_modes: list[str] = []

    class _CacheProbe:
        def __init__(self, _path, mode="r", start_frame=None, end_frame=None):
            cache_modes.append(str(mode))
            if mode == "r":
                raise AssertionError(
                    "existing detection cache should not be opened when reuse is disabled"
                )
            if mode == "w":
                raise _StopOnWrite()

    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"cache")

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(
        worker_mod,
        "create_detector",
        lambda *_args, **_kwargs: _FakeDetector(),
    )
    monkeypatch.setattr(worker_mod, "DetectionCache", _CacheProbe)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _UnusedAssigner)

    worker = worker_mod.TrackingWorker(
        str(tmp_path / "video.mp4"),
        detection_cache_path=str(cache_path),
        use_cached_detections=True,
    )
    worker.set_parameters(
        {
            "MAX_TARGETS": 1,
            "START_FRAME": 0,
            "END_FRAME": 0,
            "RESIZE_FACTOR": 1.0,
            "DETECTION_METHOD": "yolo_obb",
            "TRACKING_REALTIME_MODE": True,
            "TRACKING_WORKFLOW_MODE": "realtime",
            "ENABLE_POSE_EXTRACTOR": True,
            "USE_APRILTAGS": False,
            "CNN_CLASSIFIERS": [],
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
        }
    )

    with pytest.raises(_StopOnWrite):
        worker._run_impl()

    assert cache_modes == ["w"]


def test_tracking_worker_forward_yolo_without_detection_cache_still_initializes_detector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _StopAtDetection(RuntimeError):
        pass

    detector_calls: list[dict[str, object]] = []

    class _DetectorProbe:
        def detect_objects(self, *_args, **_kwargs):
            raise _StopAtDetection()

    def _create_detector(*_args, **_kwargs):
        detector_calls.append({"created": True})
        return _DetectorProbe()

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "create_detector", _create_detector)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _UnusedAssigner)

    worker = worker_mod.TrackingWorker(str(tmp_path / "video.mp4"))
    worker.set_parameters(
        {
            "MAX_TARGETS": 1,
            "START_FRAME": 0,
            "END_FRAME": 0,
            "RESIZE_FACTOR": 1.0,
            "DETECTION_METHOD": "yolo_obb",
            "TRACKING_REALTIME_MODE": False,
            "TRACKING_WORKFLOW_MODE": "non_realtime",
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
        }
    )

    with pytest.raises(_StopAtDetection):
        worker._run_impl()

    assert detector_calls == [{"created": True}]


def test_tracking_worker_backward_cached_yolo_skips_runtime_detector_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}

    class _CapturingAssigner:
        def __init__(self, params, worker=None):
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
            captured["meas"] = np.asarray(meas, dtype=np.float32)
            raise _StopAtAssociation()

    class _BackwardCacheProbe(_FakeDetectionCache):
        def __init__(self, _path, mode="r", start_frame=None, end_frame=None):
            super().__init__()
            assert mode == "r"
            self._cached_frames = {0}
            self._frames[0] = (
                [np.array([4.0, 4.0, 0.0], dtype=np.float32)],
                [2.0],
                [np.array([2.0, 1.0], dtype=np.float32)],
                [0.95],
                [
                    np.array(
                        [[3.0, 3.5], [5.0, 3.5], [5.0, 4.5], [3.0, 4.5]],
                        dtype=np.float32,
                    )
                ],
                [0],
                [0.0],
                [0.75],
                [0],
                None,
                None,
                None,
            )

        def covers_frame_range(self, *_args, **_kwargs):
            return True

        def get_total_frames(self):
            return 1

    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"cache")

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(
        worker_mod,
        "create_detector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("backward cached run should not create a runtime detector")
        ),
    )
    monkeypatch.setattr(worker_mod, "DetectionCache", _BackwardCacheProbe)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _CapturingAssigner)

    worker = worker_mod.TrackingWorker(
        str(tmp_path / "video.mp4"),
        backward_mode=True,
        detection_cache_path=str(cache_path),
    )
    worker.set_parameters(
        {
            "MAX_TARGETS": 1,
            "START_FRAME": 0,
            "END_FRAME": 0,
            "RESIZE_FACTOR": 1.0,
            "DETECTION_METHOD": "yolo_obb",
            "TRACKING_REALTIME_MODE": False,
            "TRACKING_WORKFLOW_MODE": "non_realtime",
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
        }
    )

    with pytest.raises(_StopAtAssociation):
        worker._run_impl()

    assert captured["meas"].shape == (1, 3)


def test_tracking_worker_realtime_yolo_obb_handles_zero_detection_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Realtime yolo_obb frames with zero detections must not crash (regression).

    InferenceRunner.run_realtime() can return a FrameResult whose OBB has
    zero detections (nothing found in frame). The realtime dispatch branch
    must still initialize `meas` and friends to empty, mirroring the cached
    branches, instead of leaving them unbound.
    """
    from hydra_suite.core.inference.result import FrameResult, OBBResult

    class _FakeInferenceRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_realtime(self, frame, frame_idx=0, roi_mask=None, roi_mask_cuda=None):
            empty_obb = OBBResult(
                frame_idx=frame_idx,
                centroids=np.zeros((0, 2), dtype=np.float32),
                angles=np.zeros((0,), dtype=np.float32),
                sizes=np.zeros((0,), dtype=np.float32),
                shapes=np.zeros((0, 2), dtype=np.float32),
                confidences=np.zeros((0,), dtype=np.float32),
                corners=np.zeros((0, 4, 2), dtype=np.float32),
                detection_ids=np.zeros((0,), dtype=np.int64),
            )
            return FrameResult(
                frame_idx=frame_idx,
                obb=empty_obb,
                filtered_indices=[],
                headtail=None,
                cnn=[],
                pose=None,
                apriltag=None,
                resolved_headings=np.zeros((0,), dtype=np.float32),
            )

        def caches_all_valid(self):
            return False

        def detection_cache_covers_range(self, *_args, **_kwargs):
            return False

        def close(self):
            return None

    class _UnusedAssignerProbe:
        def __init__(self, params, worker=None):
            self.params = params

        def compute_cost_matrix(self, *_args, **_kwargs):
            raise AssertionError(
                "compute_cost_matrix should not be called with zero detections"
            )

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _FakeInferenceRunner)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _UnusedAssignerProbe)

    results = {}

    def _capture_finished(success, trajectories, metrics):
        results["success"] = success

    worker = worker_mod.TrackingWorker(str(tmp_path / "video.mp4"))
    worker.finished_signal.connect(_capture_finished)
    worker.set_parameters(
        {
            "MAX_TARGETS": 1,
            "START_FRAME": 0,
            "END_FRAME": 0,
            "RESIZE_FACTOR": 1.0,
            "DETECTION_METHOD": "yolo_obb",
            "TRACKING_REALTIME_MODE": True,
            "TRACKING_WORKFLOW_MODE": "realtime",
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_POSE_EXTRACTOR": False,
            "USE_APRILTAGS": False,
            "CNN_CLASSIFIERS": [],
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "VISUALIZATION_FREE_MODE": True,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
        }
    )

    worker._run_impl()

    assert results.get("success") is True
