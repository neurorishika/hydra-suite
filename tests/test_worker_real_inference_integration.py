"""Real end-to-end integration tests for worker.py InferenceRunner pipeline.

These tests exercise the actual run() tracking loop with InferenceRunner mocked
at the module boundary.  They verify the real code paths (Sites B, A, D, E, F)
rather than now-deleted stub methods.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest


def _make_obb(frame_idx: int, n: int = 2):
    from hydra_suite.core.inference.result import OBBResult

    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array([[10.0, 20.0], [30.0, 40.0]][:n], dtype=np.float32),
        angles=np.array([0.1, 0.2][:n], dtype=np.float32),
        sizes=np.array([100.0, 150.0][:n], dtype=np.float32),
        shapes=np.array([[80.0, 1.5], [120.0, 1.8]][:n], dtype=np.float32),
        confidences=np.array([0.9, 0.8][:n], dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _make_frame_result(frame_idx: int = 0, n: int = 2):
    from hydra_suite.core.inference.result import FrameResult

    obb = _make_obb(frame_idx, n)
    return FrameResult(
        frame_idx=frame_idx,
        obb=obb,
        filtered_indices=list(range(n)),
        headtail=None,
        cnn=[],
        pose=None,
        apriltag=None,
        resolved_headings=np.array([0.1, 0.2][:n], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Worker module-level import checks
# ---------------------------------------------------------------------------


def test_worker_module_imports_inference_runner():
    """InferenceRunner is importable at module level (no lazy-import guard)."""
    from hydra_suite.core.tracking import worker

    assert hasattr(worker, "InferenceRunner")
    assert hasattr(worker, "InferenceConfig")
    # Stub flag must be gone
    assert not hasattr(worker, "USE_NEW_INFERENCE_PIPELINE")


def test_frame_result_bridge_imported_at_module_level():
    """frame_result_bridge helpers are imported into worker at module level."""
    from hydra_suite.core.tracking import worker

    assert hasattr(worker, "frame_result_to_meas")
    assert hasattr(worker, "populate_live_cnn_store")
    assert hasattr(worker, "populate_live_pose_store")
    assert hasattr(worker, "populate_live_tag_store")
    assert hasattr(worker, "build_density_cache_dict")


# ---------------------------------------------------------------------------
# build_inference_config_from_params
# ---------------------------------------------------------------------------


def test_build_inference_config_returns_inference_config(tmp_path):
    """build_inference_config_from_params returns a valid InferenceConfig."""
    from hydra_suite.core.inference.config import (
        InferenceConfig,
        build_inference_config_from_params,
    )

    # Minimal params
    params = {
        "YOLO_OBB_DIRECT_MODEL_PATH": str(tmp_path / "model.pt"),
        "YOLO_CONFIDENCE_THRESHOLD": 0.5,
        "COMPUTE_RUNTIME": "cpu",
        "YOLO_OBB_MODE": "direct",
    }
    cfg = build_inference_config_from_params(params)
    assert isinstance(cfg, InferenceConfig)
    assert cfg.obb is not None
    assert cfg.obb.confidence_threshold == pytest.approx(0.5)


def test_build_inference_config_sets_runtime_tier(tmp_path):
    """RUNTIME_TIER propagates into cfg.runtime_tier (the sole runtime knob)."""
    from hydra_suite.core.inference.config import build_inference_config_from_params

    params = {
        "YOLO_OBB_DIRECT_MODEL_PATH": str(tmp_path / "model.pt"),
        "RUNTIME_TIER": "gpu",
        "YOLO_OBB_MODE": "direct",
    }
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.direct is not None
    assert cfg.runtime_tier == "gpu"


def test_build_inference_config_defaults_missing_tier_to_cpu(tmp_path):
    """Runtime Gen-2: an absent RUNTIME_TIER defaults to 'cpu' (no legacy migration)."""
    from hydra_suite.core.inference.config import build_inference_config_from_params

    params = {
        "YOLO_OBB_DIRECT_MODEL_PATH": str(tmp_path / "model.pt"),
        "YOLO_OBB_MODE": "direct",
    }
    cfg = build_inference_config_from_params(params)
    assert cfg.runtime_tier == "cpu"


# ---------------------------------------------------------------------------
# frame_result_to_meas integration
# ---------------------------------------------------------------------------


def test_frame_result_to_meas_shapes_and_values():
    """frame_result_to_meas produces correct [cx, cy, theta] arrays."""
    from hydra_suite.core.tracking.ingest.frame_result_bridge import (
        frame_result_to_meas,
    )

    obb = _make_obb(frame_idx=0, n=2)
    headings = np.array([1.0, 2.0], dtype=np.float32)
    meas = frame_result_to_meas(obb.centroids, headings)

    assert len(meas) == 2
    np.testing.assert_allclose(meas[0], [10.0, 20.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(meas[1], [30.0, 40.0, 2.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Site E note: the per-frame run_realtime dispatch is exercised end-to-end by
# test_tracking_worker_realtime_live_features.py
# (test_tracking_worker_realtime_yolo_obb_handles_zero_detection_frame), which
# drives the real run_tracking() loop with a faked InferenceRunner.run_realtime.
# The former test_site_e_* tests here only asserted that a MagicMock recorded
# its own calls (they exercised no worker code) and were removed. The cached
# load_frame / batch-pass dispatch decision is covered by the real-drive tests
# in the "Site A dispatch" section below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Site F: live store population from FrameResult
# ---------------------------------------------------------------------------


def test_site_f_populate_live_cnn_store_from_frame_result():
    """populate_live_cnn_store writes predictions for each frame."""
    from hydra_suite.core.inference.result import (
        CNNDetectionPrediction,
        CNNFactorPrediction,
        CNNResult,
    )
    from hydra_suite.core.tracking.features.live_features import LiveCNNIdentityStore
    from hydra_suite.core.tracking.ingest.frame_result_bridge import (
        populate_live_cnn_store,
    )

    store = LiveCNNIdentityStore()
    cnn_result = CNNResult(
        label="id_cnn",
        predictions=[
            CNNDetectionPrediction(
                det_index=0,
                factors=[
                    CNNFactorPrediction(
                        factor_name="flat",
                        class_names=["ant1", "ant2"],
                        raw_probabilities=np.array([0.3, 0.7], dtype=np.float32),
                    )
                ],
            )
        ],
    )
    det_ids = np.array([1000001], dtype=np.int64)
    populate_live_cnn_store(
        store, [cnn_result], det_ids, frame_idx=3, phase_label="id_cnn"
    )

    preds = store.load(3)
    assert len(preds) == 1
    assert preds[0].class_names[0] == "ant2"
    assert preds[0].confidences[0] == pytest.approx(0.7, rel=1e-5)


def test_site_f_populate_live_pose_store_from_frame_result():
    """populate_live_pose_store stores keypoints per detection ID."""
    from hydra_suite.core.inference.result import PoseResult
    from hydra_suite.core.tracking.features.live_features import LivePosePropertiesStore
    from hydra_suite.core.tracking.ingest.frame_result_bridge import (
        populate_live_pose_store,
    )

    store = LivePosePropertiesStore()
    kpts = np.ones((2, 4, 3), dtype=np.float32)
    valid = np.array([True, True], dtype=bool)
    pose = PoseResult(keypoints=kpts, valid_mask=valid)
    det_ids = np.array([1000000, 1000001], dtype=np.int64)

    populate_live_pose_store(store, pose, det_ids, frame_idx=10)

    frame_data = store.get_frame(10)
    assert list(frame_data["detection_ids"]) == [1000000, 1000001]
    assert frame_data["pose_keypoints"][0].shape == (4, 3)


def test_site_f_populate_live_tag_store_from_frame_result():
    """populate_live_tag_store stores AprilTag data per frame."""
    from hydra_suite.core.inference.result import AprilTagResult
    from hydra_suite.core.tracking.features.live_features import LiveTagObservationStore
    from hydra_suite.core.tracking.ingest.frame_result_bridge import (
        populate_live_tag_store,
    )

    store = LiveTagObservationStore()
    at = AprilTagResult(
        tag_ids=[3, 7],
        det_indices=[0, 1],
        centers=np.array([[5.0, 6.0], [15.0, 16.0]], dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
    )
    det_ids = np.array([1000000, 1000001], dtype=np.int64)

    populate_live_tag_store(store, at, det_ids, frame_idx=2)

    frame_data = store.get_frame(2)
    assert list(frame_data["tag_ids"]) == [3, 7]
    np.testing.assert_allclose(frame_data["centers_xy"][0], [5.0, 6.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# Site A dispatch: caches_all_valid() decides run_batch_pass vs cached replay.
#
# These drive the REAL run_tracking() dispatch in worker.py — a faked
# InferenceRunner at the module boundary, a real synthetic video, and a
# sentinel raised from the probed method to escape before the heavy
# Kalman/assignment/CSV machinery. That keeps each test focused on the branch
# decision under test while still executing the real control flow (unlike the
# former versions, which reimplemented the if/else inline and asserted against
# their own copy).
# ---------------------------------------------------------------------------


class _StopAfterDispatch(RuntimeError):
    """Sentinel raised from a probed runner method to end run_tracking() early."""


class _FakeProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    @contextmanager
    def armed(self):
        yield self

    def __getattr__(self, _name):  # any phase_*/tick/tock/etc. is a no-op
        return lambda *a, **k: None


class _FakeVideoCapture:
    """Minimal cv2.VideoCapture stand-in yielding a couple of 8x8 frames."""

    def __init__(self, *_args, **_kwargs):
        self._frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        self._idx = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame.copy()

    def get(self, prop_id):
        import hydra_suite.core.tracking.worker as _wm

        if prop_id == _wm.cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        if prop_id == _wm.cv2.CAP_PROP_FPS:
            return 30.0
        if prop_id == _wm.cv2.CAP_PROP_FRAME_WIDTH:
            return self._frames[0].shape[1]
        if prop_id == _wm.cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frames[0].shape[0]
        if prop_id == _wm.cv2.CAP_PROP_POS_FRAMES:
            return self._idx
        return 0

    def set(self, prop_id, value):
        import hydra_suite.core.tracking.worker as _wm

        if prop_id == _wm.cv2.CAP_PROP_POS_FRAMES:
            self._idx = int(value)
        return True

    def release(self):
        self._opened = False


def _dispatch_params(**overrides):
    p = {
        "MAX_TARGETS": 1,
        "START_FRAME": 0,
        "END_FRAME": 1,
        "RESIZE_FACTOR": 1.0,
        "DETECTION_METHOD": "yolo_obb",
        "TRACKING_WORKFLOW_MODE": "non_realtime",
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
    p.update(overrides)
    return p


def _run_with_faked_runner(monkeypatch, tmp_path, runner_cls, **ctor_kwargs):
    """Drive real run_tracking() with a faked InferenceRunner; return finished-success.

    Returns (success, raised) where `raised` is the _StopAfterDispatch sentinel if
    the probed method fired one (used to prove the branch was taken), else None.
    """
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", runner_cls)

    captured = {}

    def _on_finished(success, _fps, _traj):
        captured["success"] = success

    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"), on_finished=_on_finished, **ctor_kwargs
    )
    worker.set_parameters(_dispatch_params())

    raised = None
    try:
        worker.run_tracking()
    except _StopAfterDispatch as exc:
        raised = exc
    return captured.get("success"), raised


def test_backward_mode_refuses_without_valid_caches(monkeypatch, tmp_path):
    """Real run_tracking(): backward mode with invalid caches must abort (finished=False)
    and must NOT run a fresh inference batch pass."""
    calls = {"batch": 0, "caches_checked": 0}

    class _InvalidCacheRunner:
        def __init__(self, *_a, **_k):
            pass

        def caches_all_valid(self):
            calls["caches_checked"] += 1
            return False

        def detection_cache_covers_range(self, *_a, **_k):
            return False

        def run_batch_pass(self, *_a, **_k):
            calls["batch"] += 1  # must never happen in backward mode

        def close(self):
            pass

    success, _ = _run_with_faked_runner(
        monkeypatch,
        tmp_path,
        _InvalidCacheRunner,
        backward_mode=True,
        detection_cache_path=str(tmp_path / "forward_cache"),
    )

    assert calls["caches_checked"] >= 1, "backward guard must query caches_all_valid()"
    assert success is False, "invalid backward caches must abort the run"
    assert (
        calls["batch"] == 0
    ), "backward mode must never re-run inference (run_batch_pass)"


def test_forward_invalid_caches_triggers_batch_pass(monkeypatch, tmp_path):
    """Real run_tracking(): forward, non-realtime, caches invalid → run_batch_pass IS called.

    run_batch_pass records the call and then raises the sentinel to short-circuit the
    rest of the pipeline; worker.py catches it in its own batch-pass try/except (real
    code), so we assert on the recorded call, not on propagation.
    """
    calls = {"batch": 0}

    class _NeedsBatchRunner:
        def __init__(self, *_a, **_k):
            pass

        def caches_all_valid(self):
            return False

        def detection_cache_covers_range(self, *_a, **_k):
            return False

        def run_batch_pass(self, *_a, **_k):
            calls["batch"] += 1
            raise _StopAfterDispatch("run_batch_pass reached")

        def close(self):
            pass

    _run_with_faked_runner(
        monkeypatch, tmp_path, _NeedsBatchRunner, use_cached_detections=False
    )

    assert (
        calls["batch"] == 1
    ), "forward run with invalid caches must call run_batch_pass"


def test_forward_valid_caches_skips_batch_pass(monkeypatch, tmp_path):
    """Real run_tracking(): forward, cache reuse enabled, caches valid & covering →
    run_batch_pass is SKIPPED and the cached replay path (load_frame) is used instead.
    """

    class _CachedRunner:
        def __init__(self, *_a, **_k):
            pass

        def caches_all_valid(self):
            return True

        def detection_cache_covers_range(self, *_a, **_k):
            return True

        def detection_cache_missing_frames(self, *_a, **_k):
            return []

        def run_batch_pass(self, *_a, **_k):
            raise AssertionError("run_batch_pass must NOT run when caches are valid")

        def load_frame(self, frame_idx, *_a, **_k):
            # Cached replay reached — escape before the heavy tracking loop.
            raise _StopAfterDispatch(f"load_frame({frame_idx}) reached")

        def close(self):
            pass

    success, raised = _run_with_faked_runner(
        monkeypatch, tmp_path, _CachedRunner, use_cached_detections=True
    )

    assert (
        raised is not None
    ), "cached forward run must reach the load_frame replay path"


# ---------------------------------------------------------------------------
# Regression: realtime yolo_obb detection must thread ROI_mask_current into
# InferenceRunner.run_realtime(), mirroring the background_subtraction
# branch's own `roi_mask=ROI_mask_current` call. Before the fix, the yolo_obb
# realtime branch called `inference_runner.run_realtime(frame,
# actual_frame_index)` with no roi_mask at all, so `filter_for_source()`
# silently skipped ROI filtering for every YOLO-OBB detection (confirmed
# against a real user's tracking output: 33.6% of tracked positions fell
# outside their configured ROI).
# ---------------------------------------------------------------------------


def test_realtime_yolo_obb_threads_roi_mask_into_run_realtime(monkeypatch, tmp_path):
    """Real run_tracking(): realtime yolo_obb dispatch must call
    inference_runner.run_realtime(frame, frame_idx, roi_mask=ROI_mask_current).

    This must FAIL against the pre-fix code (missing `roi_mask` kwarg, which
    defaults to `None` and disables ROI filtering) and PASS after the fix.
    """
    import hydra_suite.core.tracking.worker as worker_mod

    calls: dict = {}

    class _RoiProbeRunner:
        def __init__(self, *_a, **_k):
            pass

        def caches_all_valid(self):
            return False

        def detection_cache_covers_range(self, *_a, **_k):
            return False

        def run_realtime(self, frame, frame_idx=0, roi_mask=None, roi_mask_cuda=None):
            calls["called"] = calls.get("called", 0) + 1
            calls["roi_mask"] = roi_mask
            calls["frame_idx"] = frame_idx
            raise _StopAfterDispatch("run_realtime reached")

        def close(self):
            pass

    # A non-trivial ROI mask (left half in-ROI, right half excluded) matching
    # the 8x8 fake-video-capture frame size used by _FakeVideoCapture, so
    # `_resolve_resized_roi_mask` returns it unchanged (no resample needed).
    roi_mask = np.zeros((8, 8), dtype=np.uint8)
    roi_mask[:, :4] = 255

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _RoiProbeRunner)

    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda *_a, **_k: None,
        use_cached_detections=False,
    )
    worker.set_parameters(
        _dispatch_params(
            TRACKING_REALTIME_MODE=True,
            TRACKING_WORKFLOW_MODE="realtime",
            ROI_MASK=roi_mask,
        )
    )

    try:
        worker.run_tracking()
    except _StopAfterDispatch:
        pass

    assert calls.get("called") == 1, "realtime yolo_obb dispatch must call run_realtime"
    assert (
        calls.get("roi_mask") is not None
    ), "run_realtime must be called with a non-None roi_mask when ROI_MASK is configured"
    np.testing.assert_array_equal(calls["roi_mask"], roi_mask)
