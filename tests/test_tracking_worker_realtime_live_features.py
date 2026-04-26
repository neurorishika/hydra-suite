from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

import hydra_suite.core.identity.classification.cnn as cnn_mod
import hydra_suite.core.tracking.precompute as precompute_mod
import hydra_suite.core.tracking.worker as worker_mod
from hydra_suite.core.identity.classification.cnn import (
    ClassPrediction,
    CNNIdentityCache,
    TrackCNNHistory,
)
from hydra_suite.core.identity.dataset.oriented_video import OrientedTrackVideoExporter
from hydra_suite.core.identity.pose.features import build_pose_detection_keypoint_map
from hydra_suite.core.identity.properties.cache import IndividualPropertiesCache
from hydra_suite.core.tracking.cnn_features import (
    cnn_build_association_entries,
    cnn_update_track_history,
)
from hydra_suite.core.tracking.tag_features import build_tag_detection_map
from hydra_suite.data.detection_cache import DetectionCache
from hydra_suite.data.tag_observation_cache import TagObservationCache
from hydra_suite.trackerkit.gui.workers.crops_worker import InterpolatedCropsWorker


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


class _FakeTrackTagHistory:
    def __init__(self, n_tracks: int, window: int = 30):
        self.n_tracks = n_tracks
        self.window = window

    def build_track_tag_id_list(self, n_tracks: int):
        return [worker_mod.NO_TAG] * n_tracks

    def resize(self, _n_tracks: int):
        return None

    def record(self, *_args, **_kwargs):
        return None

    def clear_track(self, *_args, **_kwargs):
        return None


class _FakeTrackCNNHistory:
    def __init__(self, *, window: int = 10, factor_names: tuple = ("flat",)):
        self.window = window
        self.factor_names = factor_names

    def build_track_identity_list(self):
        return {}

    def record(self, *_args, **_kwargs):
        return None

    def clear_track(self, *_args, **_kwargs):
        return None


class _FakeKalmanFilterManager:
    def __init__(self, n_targets: int, _params):
        self.X = np.zeros((n_targets, 5), dtype=np.float32)

    def get_predictions(self):
        return np.zeros((len(self.X), 3), dtype=np.float32)


class _UnusedAssigner:
    def __init__(self, params, worker=None):
        self.params = params


class _FakePhase:
    def __init__(self, name: str, cache_path: str = "", finalize_metadata=None):
        self.name = name
        self._cache_path = cache_path
        self._finalize_metadata = finalize_metadata or {}
        self._callback = None

    def has_cache_hit(self) -> bool:
        return False

    def set_frame_result_callback(self, callback):
        self._callback = callback


class _FakeUnifiedPrecompute:
    def __init__(self, phases, _crop_config):
        self.phases = phases
        self.process_calls = 0

    def process_live_frame(
        self,
        frame_idx,
        frame,
        detector,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confs,
        raw_obb,
        raw_ids,
        raw_headings,
        raw_heading_confidences,
        raw_directed,
        raw_canonical_affines,
        roi_mask,
        profiler=None,
    ):
        self.process_calls += 1
        for phase in self.phases:
            if phase._callback is None:
                continue
            if phase.name == "pose":
                phase._callback(
                    frame_idx,
                    list(raw_ids),
                    [
                        np.array(
                            [[10.0, 0.0, 0.95], [0.0, 0.0, 0.95]],
                            dtype=np.float32,
                        )
                    ],
                )
            elif phase.name == "apriltag":
                phase._callback(
                    frame_idx,
                    [42],
                    [(4.0, 4.0)],
                    [np.zeros((4, 2), dtype=np.float32)],
                    [0],
                    [0],
                )
            else:
                phase._callback(
                    frame_idx,
                    [
                        ClassPrediction(
                            det_index=0,
                            factor_names=("flat",),
                            class_names=("alpha",),
                            confidences=(0.99,),
                        )
                    ],
                )

    def sync_live_frame(self):
        return None

    def finalize_live(self, warning_cb=None):
        return {"pose": "pose-cache.npz", "apriltag": "tags.npz"}


class _ArtifactWritingUnifiedPrecompute:
    instances = []

    def __init__(self, phases, _crop_config):
        self.phases = phases
        self.process_calls = 0
        self.run_called = False
        self.pose_frames = {}
        self.tag_frames = {}
        self.cnn_frames = {}
        type(self).instances.append(self)

    def run(self, *args, **kwargs):
        self.run_called = True
        raise AssertionError("offline precompute should not run in realtime mode")

    def process_live_frame(
        self,
        frame_idx,
        frame,
        detector,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confs,
        raw_obb,
        raw_ids,
        raw_headings,
        raw_heading_confidences,
        raw_directed,
        raw_canonical_affines,
        roi_mask,
        profiler=None,
    ):
        self.process_calls += 1
        det_ids = list(raw_ids)
        pose_keypoints = [
            np.array(
                [[10.0, 0.0, 0.95], [0.0, 0.0, 0.95]],
                dtype=np.float32,
            )
        ]
        tag_payload = {
            "tag_ids": [42],
            "centers_xy": [(4.0, 4.0)],
            "corners": [
                np.array(
                    [[3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0]], dtype=np.float32
                )
            ],
            "det_indices": [0],
            "hammings": [0],
        }
        cnn_preds = [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("alpha",),
                confidences=(0.99,),
            )
        ]
        self.pose_frames[int(frame_idx)] = (det_ids, pose_keypoints)
        self.tag_frames[int(frame_idx)] = tag_payload
        self.cnn_frames[int(frame_idx)] = cnn_preds

        for phase in self.phases:
            callback = getattr(phase, "_callback", None)
            if callback is None:
                continue
            if phase.name == "pose":
                callback(frame_idx, det_ids, pose_keypoints)
            elif phase.name == "apriltag":
                callback(
                    frame_idx,
                    tag_payload["tag_ids"],
                    tag_payload["centers_xy"],
                    tag_payload["corners"],
                    tag_payload["det_indices"],
                    tag_payload["hammings"],
                )
            else:
                callback(frame_idx, cnn_preds)

    def sync_live_frame(self):
        return None

    def finalize_live(self, warning_cb=None):
        results = {}
        for phase in self.phases:
            cache_path = str(getattr(phase, "_cache_path", "") or "")
            if phase.name == "pose":
                cache = IndividualPropertiesCache(cache_path, mode="w")
                for frame_idx, (det_ids, keypoints) in sorted(self.pose_frames.items()):
                    cache.add_frame(frame_idx, det_ids, pose_keypoints=keypoints)
                cache.save(metadata=getattr(phase, "_finalize_metadata", {}))
                cache.close()
                results["pose"] = cache_path
            elif phase.name == "apriltag":
                tag_cache = TagObservationCache(
                    cache_path, mode="w", start_frame=0, end_frame=0
                )
                for frame_idx, payload in sorted(self.tag_frames.items()):
                    tag_cache.add_frame(
                        frame_idx,
                        payload["tag_ids"],
                        payload["centers_xy"],
                        payload["corners"],
                        payload["det_indices"],
                        payload["hammings"],
                    )
                tag_cache.save(metadata={"start_frame": 0, "end_frame": 0})
                tag_cache.close()
                results["apriltag"] = cache_path
            else:
                cnn_cache = CNNIdentityCache(cache_path)
                for frame_idx, preds in sorted(self.cnn_frames.items()):
                    cnn_cache.save(frame_idx, preds)
                cnn_cache.flush()
                results[phase.name] = cache_path
        return results


class _RealtimeArtifactPhase:
    def __init__(self, name: str, cache_path: str, finalize_metadata=None):
        self.name = name
        self._cache_path = cache_path
        self._finalize_metadata = finalize_metadata or {}
        self._callback = None

    def has_cache_hit(self) -> bool:
        return False

    def set_frame_result_callback(self, callback):
        self._callback = callback


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


def test_tracking_worker_realtime_streams_live_pose_tag_and_cnn_into_assignment(
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
            captured["association_data"] = association_data
            raise _StopAtAssociation()

    phases = [
        _FakePhase(
            "pose",
            cache_path=str(tmp_path / "pose_cache.npz"),
            finalize_metadata={"pose_keypoint_names": ["head", "tail"]},
        ),
        _FakePhase("apriltag", cache_path=str(tmp_path / "tag_cache.npz")),
        _FakePhase("cnn_identity", cache_path=str(tmp_path / "cnn_cache.npz")),
    ]

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(
        worker_mod, "create_detector", lambda *_args, **_kwargs: _FakeDetector()
    )
    monkeypatch.setattr(worker_mod, "DetectionCache", _FakeDetectionCache)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _CapturingAssigner)
    monkeypatch.setattr(worker_mod, "TrackTagHistory", _FakeTrackTagHistory)
    monkeypatch.setattr(cnn_mod, "TrackCNNHistory", _FakeTrackCNNHistory)
    monkeypatch.setattr(worker_mod, "UnifiedPrecompute", _FakeUnifiedPrecompute)
    monkeypatch.setattr(
        worker_mod,
        "CropConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        worker_mod.TrackingWorker,
        "_build_precompute_phases",
        lambda self, params, detection_method, detection_cache, start_frame, end_frame: phases,
    )
    monkeypatch.setattr(
        worker_mod,
        "_pf_build_direction_overrides",
        lambda n_det, pose_heading, pose_directed_mask, headtail_heading, headtail_directed_mask, pose_overrides_headtail=True: (
            np.asarray(pose_heading, dtype=np.float32),
            np.asarray(pose_directed_mask, dtype=np.uint8),
        ),
    )

    worker = worker_mod.TrackingWorker(
        str(tmp_path / "video.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
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
            "USE_APRILTAGS": True,
            "CNN_CLASSIFIERS": [{"label": "cnn_identity", "window": 5}],
            "POSE_DIRECTION_ANTERIOR_KEYPOINTS": ["head"],
            "POSE_DIRECTION_POSTERIOR_KEYPOINTS": ["tail"],
            "POSE_IGNORE_KEYPOINTS": [],
            "POSE_MIN_KPT_CONF_VALID": 0.2,
            "POSE_OVERRIDES_HEADTAIL": True,
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "SUPPRESS_FOREIGN_OBB_REGIONS": True,
            "INDIVIDUAL_CROP_PADDING": 0.1,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
            "INFERENCE_MODEL_ID": "test-model",
        }
    )

    with pytest.raises(_StopAtAssociation):
        worker.run()

    association_data = captured["association_data"]
    assert association_data["detection_tag_ids"] == [42]
    assert association_data["cnn_phases"][0]["label"] == "cnn_identity"
    assert association_data["cnn_phases"][0]["detection_classes"] == ["alpha"]
    assert association_data["detection_pose_keypoints"][0] is not None
    assert association_data["detection_pose_visibility"][0] > 0.0
    assert np.isfinite(association_data["detection_pose_heading"][0])


def test_tracking_worker_realtime_does_not_mark_pose_directed_when_pose_is_weak(
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
            captured["association_data"] = association_data
            captured["meas_ori_directed"] = np.asarray(
                meas_ori_directed, dtype=np.uint8
            )
            raise _StopAtAssociation()

    phases = [
        _FakePhase(
            "pose",
            cache_path=str(tmp_path / "pose_cache.npz"),
            finalize_metadata={"pose_keypoint_names": ["head", "tail"]},
        )
    ]

    def _weak_pose_live_frame(self, frame_idx, *args, **kwargs):
        self.process_calls += 1
        for phase in self.phases:
            if phase._callback is None:
                continue
            if phase.name == "pose":
                phase._callback(
                    frame_idx,
                    [0],
                    [np.array([[10.0, 0.0, 0.95], [0.0, 0.0, 0.95]], dtype=np.float32)],
                )

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(
        worker_mod, "create_detector", lambda *_args, **_kwargs: _FakeDetector()
    )
    monkeypatch.setattr(worker_mod, "DetectionCache", _FakeDetectionCache)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _CapturingAssigner)
    monkeypatch.setattr(worker_mod, "TrackTagHistory", _FakeTrackTagHistory)
    monkeypatch.setattr(cnn_mod, "TrackCNNHistory", _FakeTrackCNNHistory)
    monkeypatch.setattr(worker_mod, "UnifiedPrecompute", _FakeUnifiedPrecompute)
    monkeypatch.setattr(
        (
            worker_mod._FakeUnifiedPrecompute
            if hasattr(worker_mod, "_FakeUnifiedPrecompute")
            else worker_mod.UnifiedPrecompute
        ),
        "process_live_frame",
        _weak_pose_live_frame,
        raising=False,
    )
    monkeypatch.setattr(
        worker_mod,
        "CropConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        worker_mod.TrackingWorker,
        "_build_precompute_phases",
        lambda self, params, detection_method, detection_cache, start_frame, end_frame: phases,
    )
    monkeypatch.setattr(
        worker_mod,
        "_pf_build_direction_overrides",
        lambda n_det, pose_heading, pose_directed_mask, headtail_heading, headtail_directed_mask, pose_overrides_headtail=True: (
            np.asarray(pose_heading, dtype=np.float32),
            np.asarray(pose_directed_mask, dtype=np.uint8),
        ),
    )

    worker = worker_mod.TrackingWorker(
        str(tmp_path / "video.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
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
            "POSE_DIRECTION_ANTERIOR_KEYPOINTS": ["head"],
            "POSE_DIRECTION_POSTERIOR_KEYPOINTS": ["tail"],
            "POSE_IGNORE_KEYPOINTS": [],
            "POSE_MIN_KPT_CONF_VALID": 0.2,
            "POSE_OVERRIDES_HEADTAIL": True,
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "SUPPRESS_FOREIGN_OBB_REGIONS": True,
            "INDIVIDUAL_CROP_PADDING": 0.1,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
            "INFERENCE_MODEL_ID": "test-model",
        }
    )

    with pytest.raises(_StopAtAssociation):
        worker.run()

    assert np.isfinite(captured["association_data"]["detection_pose_heading"][0])
    np.testing.assert_array_equal(captured["meas_ori_directed"], [0])


def test_tracking_worker_realtime_keeps_cnn_hints_out_of_assignment_when_online_decoder_is_enabled(
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
            captured["association_data"] = association_data
            raise _StopAtAssociation()

    phases = [
        _FakePhase(
            "pose",
            cache_path=str(tmp_path / "pose_cache.npz"),
            finalize_metadata={"pose_keypoint_names": ["head", "tail"]},
        ),
        _FakePhase("apriltag", cache_path=str(tmp_path / "tag_cache.npz")),
        _FakePhase("cnn_identity", cache_path=str(tmp_path / "cnn_cache.npz")),
    ]

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(
        worker_mod, "create_detector", lambda *_args, **_kwargs: _FakeDetector()
    )
    monkeypatch.setattr(worker_mod, "DetectionCache", _FakeDetectionCache)
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _CapturingAssigner)
    monkeypatch.setattr(worker_mod, "TrackTagHistory", _FakeTrackTagHistory)
    monkeypatch.setattr(cnn_mod, "TrackCNNHistory", _FakeTrackCNNHistory)
    monkeypatch.setattr(worker_mod, "UnifiedPrecompute", _FakeUnifiedPrecompute)
    monkeypatch.setattr(
        worker_mod,
        "CropConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        worker_mod.TrackingWorker,
        "_build_precompute_phases",
        lambda self, params, detection_method, detection_cache, start_frame, end_frame: phases,
    )
    monkeypatch.setattr(
        worker_mod,
        "_pf_build_direction_overrides",
        lambda n_det, pose_heading, pose_directed_mask, headtail_heading, headtail_directed_mask, pose_overrides_headtail=True: (
            np.asarray(pose_heading, dtype=np.float32),
            np.asarray(pose_directed_mask, dtype=np.uint8),
        ),
    )

    worker = worker_mod.TrackingWorker(
        str(tmp_path / "video.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
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
            "USE_APRILTAGS": True,
            "ENABLE_INDIVIDUAL_PIPELINE": True,
            "ENABLE_IDENTITY_ONLINE_DECODER": True,
            "ENABLE_LEGACY_CNN_ASSOCIATION": False,
            "CNN_CLASSIFIERS": [
                {"label": "cnn_identity", "window": 5, "labels": ["alpha", "beta"]}
            ],
            "POSE_DIRECTION_ANTERIOR_KEYPOINTS": ["head"],
            "POSE_DIRECTION_POSTERIOR_KEYPOINTS": ["tail"],
            "POSE_IGNORE_KEYPOINTS": [],
            "POSE_MIN_KPT_CONF_VALID": 0.2,
            "POSE_OVERRIDES_HEADTAIL": True,
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "SUPPRESS_FOREIGN_OBB_REGIONS": True,
            "INDIVIDUAL_CROP_PADDING": 0.1,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
            "INFERENCE_MODEL_ID": "test-model",
        }
    )

    with pytest.raises(_StopAtAssociation):
        worker.run()

    association_data = captured["association_data"]
    assert association_data["detection_tag_ids"] == [42]
    assert "cnn_phases" not in association_data


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
        worker.run()

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
        worker.run()

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
    monkeypatch.setattr(worker_mod, "TrackTagHistory", _FakeTrackTagHistory)

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
        worker.run()

    assert captured["meas"].shape == (1, 3)


def test_tracking_worker_realtime_ignores_existing_analysis_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pose_cache_path = tmp_path / "pose_cache.npz"
    tag_cache_path = tmp_path / "tag_cache.npz"
    cnn_cache_path = tmp_path / "cnn_cache.npz"
    model_path = tmp_path / "cnn_model.fake"

    pose_cache = IndividualPropertiesCache(str(pose_cache_path), mode="w")
    pose_cache.add_frame(
        0,
        [0],
        pose_keypoints=[np.array([[1.0, 1.0, 0.8]], dtype=np.float32)],
    )
    pose_cache.save(metadata={"pose_keypoint_names": ["old_head"]})
    pose_cache.close()

    tag_cache = TagObservationCache(
        str(tag_cache_path), mode="w", start_frame=0, end_frame=0
    )
    tag_cache.add_frame(
        0,
        tag_ids=[7],
        centers_xy=[(1.0, 1.0)],
        corners=[np.zeros((4, 2), dtype=np.float32)],
        det_indices=[0],
        hammings=[0],
    )
    tag_cache.save(metadata={"start_frame": 0, "end_frame": 0})
    tag_cache.close()

    cnn_cache = CNNIdentityCache(str(cnn_cache_path))
    cnn_cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("stale",),
                confidences=(0.5,),
            )
        ],
    )
    cnn_cache.flush()

    model_path.write_bytes(b"fake-model")

    class _FakePoseBackend:
        output_keypoint_names = ["head", "tail"]
        preferred_input_size = 0

        def warmup(self):
            return None

        def close(self):
            return None

    class _FakeAprilTagDetector:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            return None

    class _FakeCNNBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            return None

    monkeypatch.setattr(
        "hydra_suite.core.identity.pose.api.build_runtime_config",
        lambda params, out_root: {"out_root": out_root},
    )
    monkeypatch.setattr(
        "hydra_suite.core.identity.pose.api.create_pose_backend_from_config",
        lambda _cfg: _FakePoseBackend(),
    )
    monkeypatch.setattr(precompute_mod, "AprilTagDetector", _FakeAprilTagDetector)
    monkeypatch.setattr(precompute_mod, "CNNIdentityBackend", _FakeCNNBackend)

    worker = worker_mod.TrackingWorker(
        str(tmp_path / "video.mp4"),
        detection_cache_path=str(tmp_path / "detections.npz"),
    )
    monkeypatch.setattr(
        worker,
        "_build_individual_properties_cache_path",
        lambda _properties_id, _start_frame, _end_frame: pose_cache_path,
    )
    monkeypatch.setattr(
        worker,
        "_build_tag_cache_path",
        lambda _apriltag_id, _start_frame, _end_frame: str(tag_cache_path),
    )
    monkeypatch.setattr(
        worker,
        "_build_cnn_identity_cache_path",
        lambda _label, _classify_id, _start_frame, _end_frame: str(cnn_cache_path),
    )

    params = {
        "TRACKING_REALTIME_MODE": True,
        "TRACKING_WORKFLOW_MODE": "realtime",
        "ENABLE_POSE_EXTRACTOR": True,
        "USE_APRILTAGS": True,
        "CNN_CLASSIFIERS": [{"label": "cnn_identity", "model_path": str(model_path)}],
        "COMPUTE_RUNTIME": "cpu",
        "INFERENCE_MODEL_ID": "test-model",
    }

    phases = worker._build_precompute_phases(
        params,
        "yolo_obb",
        object(),
        0,
        0,
    )
    try:
        phase_by_name = {phase.name: phase for phase in phases}
        assert set(phase_by_name) == {"pose", "apriltag", "cnn_identity"}
        assert phase_by_name["pose"].has_cache_hit() is False
        assert phase_by_name["apriltag"].has_cache_hit() is False
        assert phase_by_name["cnn_identity"].has_cache_hit() is False
    finally:
        for phase in phases:
            phase.close()


def test_realtime_forward_finalizes_artifacts_and_downstream_consumers_reuse_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    video_path = tmp_path / "source.mp4"
    detection_cache_path = tmp_path / "detections.npz"
    pose_cache_path = tmp_path / "pose_cache.npz"
    tag_cache_path = tmp_path / "tag_cache.npz"
    cnn_cache_path = tmp_path / "cnn_cache.npz"
    model_path = tmp_path / "cnn_model.fake"
    final_csv_path = tmp_path / "tracks_final.csv"
    dataset_dir = tmp_path / "exports"

    _write_test_video(video_path, [(0, 0, 255)])
    model_path.write_bytes(b"fake-model")

    phases = [
        _RealtimeArtifactPhase(
            "pose",
            str(pose_cache_path),
            finalize_metadata={"pose_keypoint_names": ["head", "tail"]},
        ),
        _RealtimeArtifactPhase("apriltag", str(tag_cache_path)),
        _RealtimeArtifactPhase("cnn_identity", str(cnn_cache_path)),
    ]

    def _build_phases(
        self, params, detection_method, detection_cache, start_frame, end_frame
    ):
        if self.backward_mode:
            return []
        return phases

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(
        worker_mod, "create_detector", lambda *_args, **_kwargs: _FakeDetector()
    )
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _UnusedAssigner)
    monkeypatch.setattr(
        worker_mod, "UnifiedPrecompute", _ArtifactWritingUnifiedPrecompute
    )
    monkeypatch.setattr(
        worker_mod.TrackingWorker, "_build_precompute_phases", _build_phases
    )
    monkeypatch.setattr(
        worker_mod.TrackingWorker,
        "_build_cnn_identity_cache_path",
        lambda self, label, classify_id, start_frame, end_frame: str(cnn_cache_path),
    )

    base_params = {
        "MAX_TARGETS": 1,
        "START_FRAME": 0,
        "END_FRAME": 0,
        "RESIZE_FACTOR": 1.0,
        "DETECTION_METHOD": "yolo_obb",
        "TRACKING_REALTIME_MODE": True,
        "TRACKING_WORKFLOW_MODE": "realtime",
        "ENABLE_POSE_EXTRACTOR": True,
        "USE_APRILTAGS": True,
        "CNN_CLASSIFIERS": [
            {"label": "cnn_identity", "model_path": str(model_path), "window": 5}
        ],
        "POSE_DIRECTION_ANTERIOR_KEYPOINTS": ["head"],
        "POSE_DIRECTION_POSTERIOR_KEYPOINTS": ["tail"],
        "POSE_IGNORE_KEYPOINTS": [],
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "POSE_OVERRIDES_HEADTAIL": True,
        "MIN_DETECTIONS_TO_START": 2,
        "MIN_DETECTION_COUNTS": 2,
        "LOST_THRESHOLD_FRAMES": 1,
        "REFERENCE_BODY_SIZE": 20.0,
        "MAX_DISTANCE_THRESHOLD": 1000.0,
        "ENABLE_CONFIDENCE_DENSITY_MAP": False,
        "ENABLE_FRAME_PREFETCH": False,
        "VISUALIZATION_FREE_MODE": True,
        "SUPPRESS_FOREIGN_OBB_REGIONS": True,
        "INDIVIDUAL_CROP_PADDING": 0.1,
        "ADVANCED_CONFIG": {},
        "COMPUTE_RUNTIME": "cpu",
        "INFERENCE_MODEL_ID": "test-model",
    }

    forward_worker = worker_mod.TrackingWorker(
        str(video_path),
        detection_cache_path=str(detection_cache_path),
    )
    forward_worker.set_parameters(dict(base_params))
    forward_worker.run()

    assert _ArtifactWritingUnifiedPrecompute.instances
    assert _ArtifactWritingUnifiedPrecompute.instances[-1].process_calls == 1
    assert _ArtifactWritingUnifiedPrecompute.instances[-1].run_called is False
    assert detection_cache_path.exists()
    assert pose_cache_path.exists()
    assert tag_cache_path.exists()
    assert cnn_cache_path.exists()

    detection_cache = DetectionCache(str(detection_cache_path), mode="r")
    try:
        frame = detection_cache.get_frame(0)
        assert frame[5] == [0]
        assert detection_cache.covers_frame_range(0, 0) is True

        pose_cache = IndividualPropertiesCache(str(pose_cache_path), mode="r")
        try:
            assert pose_cache.is_compatible() is True
            pose_map = build_pose_detection_keypoint_map(pose_cache, 0)
            assert 0 in pose_map
        finally:
            pose_cache.close()

        tag_cache = TagObservationCache(str(tag_cache_path), mode="r")
        try:
            assert tag_cache.is_compatible() is True
            assert build_tag_detection_map(tag_cache, 0) == {0: 42}
        finally:
            tag_cache.close()

        cnn_cache = CNNIdentityCache(str(cnn_cache_path))
        det_classes, track_identities, frame_preds = cnn_build_association_entries(
            cnn_cache,
            _FakeTrackCNNHistory(window=10, factor_names=("flat",)),
            0,
            1,
            1,
        )
        assert det_classes == ["alpha"]
        assert track_identities == [None]
        assert len(frame_preds) == 1

        backward_worker = worker_mod.TrackingWorker(
            str(video_path),
            backward_mode=True,
            detection_cache_path=str(detection_cache_path),
        )
        backward_params = dict(base_params)
        backward_params["INDIVIDUAL_PROPERTIES_CACHE_PATH"] = str(pose_cache_path)
        backward_worker.set_parameters(backward_params)
        backward_worker.run()
        assert backward_worker.frame_count == 1

        width, height = InterpolatedCropsWorker._get_detection_size(
            detection_cache,
            0,
            0,
        )
        assert width is not None and height is not None
    finally:
        detection_cache.close()

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 0,
                "Theta": 0.0,
                "State": "active",
            }
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=detection_cache_path,
        fps=5.0,
        padding_fraction=0.0,
        export_videos=True,
        export_images=False,
        output_subdir="oriented_videos",
    )
    result = exporter.export()
    assert result.exported_videos == 1
    assert result.exported_frames == 1
    assert (dataset_dir / "oriented_videos" / "trajectory_0001.mp4").exists()


def test_non_realtime_forward_defaults_to_streaming_individual_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    video_path = tmp_path / "source.mp4"
    detection_cache_path = tmp_path / "detections.npz"
    pose_cache_path = tmp_path / "pose_cache.npz"
    tag_cache_path = tmp_path / "tag_cache.npz"
    cnn_cache_path = tmp_path / "cnn_cache.npz"
    model_path = tmp_path / "cnn_model.fake"

    _ArtifactWritingUnifiedPrecompute.instances = []
    _write_test_video(video_path, [(0, 0, 255)])
    model_path.write_bytes(b"fake-model")

    phases = [
        _RealtimeArtifactPhase(
            "pose",
            str(pose_cache_path),
            finalize_metadata={"pose_keypoint_names": ["head", "tail"]},
        ),
        _RealtimeArtifactPhase("apriltag", str(tag_cache_path)),
        _RealtimeArtifactPhase("cnn_identity", str(cnn_cache_path)),
    ]

    def _build_phases(
        self, params, detection_method, detection_cache, start_frame, end_frame
    ):
        if self.backward_mode:
            return []
        return phases

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(
        worker_mod, "create_detector", lambda *_args, **_kwargs: _FakeDetector()
    )
    monkeypatch.setattr(worker_mod, "KalmanFilterManager", _FakeKalmanFilterManager)
    monkeypatch.setattr(worker_mod, "TrackAssigner", _UnusedAssigner)
    monkeypatch.setattr(
        worker_mod, "UnifiedPrecompute", _ArtifactWritingUnifiedPrecompute
    )
    monkeypatch.setattr(
        worker_mod.TrackingWorker, "_build_precompute_phases", _build_phases
    )

    worker = worker_mod.TrackingWorker(
        str(video_path),
        detection_cache_path=str(detection_cache_path),
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
            "ENABLE_POSE_EXTRACTOR": True,
            "USE_APRILTAGS": True,
            "CNN_CLASSIFIERS": [
                {"label": "cnn_identity", "model_path": str(model_path), "window": 5}
            ],
            "MIN_DETECTIONS_TO_START": 2,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "REFERENCE_BODY_SIZE": 20.0,
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "VISUALIZATION_FREE_MODE": True,
            "SUPPRESS_FOREIGN_OBB_REGIONS": True,
            "INDIVIDUAL_CROP_PADDING": 0.1,
            "ADVANCED_CONFIG": {},
            "COMPUTE_RUNTIME": "cpu",
        }
    )

    worker.run()

    assert _ArtifactWritingUnifiedPrecompute.instances
    assert _ArtifactWritingUnifiedPrecompute.instances[-1].process_calls == 1
    assert _ArtifactWritingUnifiedPrecompute.instances[-1].run_called is False
    assert detection_cache_path.exists()
    assert pose_cache_path.exists()
    assert tag_cache_path.exists()
    assert cnn_cache_path.exists()


def test_cnn_build_association_entries_handles_multihead_cache_for_flat_history(
    tmp_path,
):
    cache_path = tmp_path / "cnn_multi_assoc.npz"
    cache = CNNIdentityCache(cache_path, factor_names=("color", "shape"))
    cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("color", "shape"),
                class_names=("red", "circle"),
                confidences=(0.9, 0.8),
            )
        ],
    )
    cache.flush()

    det_classes, track_identities, frame_preds = cnn_build_association_entries(
        CNNIdentityCache(str(cache_path)),
        TrackCNNHistory(window=10, factor_names=("flat",)),
        0,
        1,
        1,
    )

    assert det_classes == ["red"]
    assert track_identities == [None]
    assert len(frame_preds) == 1


def test_cnn_build_association_entries_supports_detection_only_cache_reads(tmp_path):
    cache_path = tmp_path / "cnn_detection_only_assoc.npz"
    cache = CNNIdentityCache(cache_path)
    cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("alpha",),
                confidences=(0.91,),
            )
        ],
    )
    cache.flush()

    det_classes, track_identities, frame_preds = cnn_build_association_entries(
        CNNIdentityCache(str(cache_path)),
        None,
        0,
        1,
        2,
    )

    assert det_classes == ["alpha"]
    assert track_identities == [None, None]
    assert len(frame_preds) == 1


def test_cnn_update_track_history_records_all_multihead_factors():
    history = TrackCNNHistory(window=5, factor_names=("color", "shape"))
    frame_preds = [
        ClassPrediction(
            det_index=0,
            factor_names=("color", "shape"),
            class_names=("red", "circle"),
            confidences=(0.93, 0.81),
        )
    ]

    cnn_update_track_history(
        history,
        frame_preds,
        frame_idx=0,
        N=1,
        rows=[0],
        cols=[0],
    )

    assert history.build_track_identity_list() == {0: ("red", "circle")}
