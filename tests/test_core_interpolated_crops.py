from collections import defaultdict

import pytest

from hydra_suite.core.post.interpolated_crops import run_interpolated_crops


def test_missing_csv_returns_empty_payload(tmp_path):
    result = run_interpolated_crops(
        str(tmp_path / "nope.csv"),
        str(tmp_path / "nope.mp4"),
        str(tmp_path / "nope.npz"),
        {},
    )
    # _validate_and_setup returns None on a missing CSV; the pipeline yields the
    # documented "nothing produced" payload rather than raising.
    assert isinstance(result, dict)
    assert result.get("saved", 0) == 0


def test_should_stop_before_setup_returns_empty_payload(tmp_path):
    result = run_interpolated_crops(
        str(tmp_path / "any.csv"),
        str(tmp_path / "any.mp4"),
        str(tmp_path / "any.npz"),
        {},
        should_stop=lambda: True,
    )
    assert result.get("saved", 0) == 0


def test_init_interpolation_backends_returns_config_and_runtime():
    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.post.interpolated_crops import _init_interpolation_backends

    params = {
        "ENABLE_POSE_EXTRACTOR": False,
        "USE_APRILTAGS": False,
        "CNN_CLASSIFIERS": [],
        "YOLO_HEADTAIL_MODEL_PATH": "",
        "RUNTIME_TIER": "cpu",
    }
    geometry = canonical_geometry_from_params(params)
    result = _init_interpolation_backends(params, "/tmp", geometry)
    cfg, runtime, pose_model, apriltag_model, cnn_models, cnn_labels, headtail_model = (
        result
    )
    assert cfg.pose is None
    assert pose_model is None
    assert apriltag_model is None
    assert cnn_models == []
    assert cnn_labels == []
    assert headtail_model is None
    assert runtime.device in {"cpu", "mps", "cuda:0"}


def test_process_occluded_run_uses_csv_value_when_present():
    """A row with non-NaN X/Y/Theta already in the CSV (mechanism-1 fill)
    must be used directly, not re-derived by linear interpolation."""
    import pandas as pd

    from hydra_suite.core.post.interpolated_crops import _process_occluded_run

    group = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "X": [0.0, 999.0, 20.0],  # frame 1 already filled by mechanism (1)
            "Y": [0.0, 888.0, 20.0],
            "Theta": [0.0, 1.23, 0.0],
            "State": ["tracked", "occluded", "tracked"],
            "DetectionID": [1, None, 2],
        }
    )
    frame_tasks = defaultdict(list)
    params = {"REFERENCE_BODY_SIZE": 20.0}
    result = _process_occluded_run(
        params,
        None,
        group,
        traj_id=5,
        last_valid_idx=0,
        i=1,
        j=2,
        detection_cache=None,
        position_scale=1.0,
        size_scale=1.0,
        frame_tasks=frame_tasks,
        interp_runs=0,
        interp_gaps=0,
    )
    assert result is not None
    task = frame_tasks[1][0]
    assert task["cx"] == pytest.approx(999.0)
    assert task["cy"] == pytest.approx(888.0)
    assert task["theta"] == pytest.approx(1.23)


def test_process_occluded_run_falls_back_when_csv_value_is_nan():
    import pandas as pd

    from hydra_suite.core.post.interpolated_crops import _process_occluded_run

    group = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "X": [0.0, float("nan"), 20.0],
            "Y": [0.0, float("nan"), 20.0],
            "Theta": [0.0, float("nan"), 0.0],
            "State": ["tracked", "occluded", "tracked"],
            "DetectionID": [1, None, 2],
        }
    )
    frame_tasks = defaultdict(list)
    params = {"REFERENCE_BODY_SIZE": 20.0}
    result = _process_occluded_run(
        params,
        None,
        group,
        traj_id=5,
        last_valid_idx=0,
        i=1,
        j=2,
        detection_cache=None,
        position_scale=1.0,
        size_scale=1.0,
        frame_tasks=frame_tasks,
        interp_runs=0,
        interp_gaps=0,
    )
    assert result is not None
    task = frame_tasks[1][0]
    # falls back to the existing linear midpoint: t=0.5 between (0,0) and (20,20)
    assert task["cx"] == pytest.approx(10.0)
    assert task["cy"] == pytest.approx(10.0)


def test_flush_pose_cnn_window_stamps_pose_source_interp():
    from types import SimpleNamespace

    import numpy as np

    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.inference.config import PoseConfig
    from hydra_suite.core.inference.result import PoseResult
    from hydra_suite.core.inference.stages.pose import PoseModel
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result

    class _FakeBackend:
        def predict_batch(self, crops):
            # Non-zero confidence (> PoseConfig.min_keypoint_confidence=0.2):
            # `run_pose_batch` -> `_assemble_pose_result` recomputes its OWN
            # valid_mask from the keypoint confidence column against
            # config.min_keypoint_confidence/min_valid_keypoints -- it does
            # NOT read this raw backend PoseResult's `valid_mask` at all.
            # All-zero keypoints (conf=0.0) would fail that recomputed check
            # and be treated as a fabricated/invalid result (see the C1
            # regression test below), so this fixture uses a real, confident
            # keypoint to represent a genuine backend hit.
            return [
                PoseResult(
                    keypoints=np.array([[[5.0, 5.0, 0.9]]], dtype=np.float32),
                    valid_mask=np.array([True]),
                )
                for _ in crops
            ]

        # run_pose_batch checks hasattr(backend, "predict_batch_cuda") to
        # decide the CUDA branch; omit it so the CPU branch is taken.

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)
    pose_model = PoseModel(
        backend=_FakeBackend(), n_keypoints=1, keypoint_names=["head"]
    )
    cfg = SimpleNamespace(pose=PoseConfig(), cnn_phases=[])

    task = {
        "frame_id": 1,
        "cx": 32.0,
        "cy": 32.0,
        "w": 20.0,
        "h": 8.0,
        "theta": 0.0,
        "traj_id": 5,
        "interp_index": 1,
        "interp_from": (0, 2),
        "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    interp_pose_rows = []
    ic._flush_pose_cnn_window(
        pending_frames=[frame],
        pending_obbs=[obb],
        pending_tasks_by_frame=[[task]],
        pose_model=pose_model,
        cnn_models=[],
        cnn_labels=[],
        cfg=cfg,
        runtime=None,
        geometry=geometry,
        interp_pose_rows=interp_pose_rows,
        interp_cnn_rows={},
        profiler=None,
    )
    assert len(interp_pose_rows) == 1
    assert interp_pose_rows[0]["PoseSource"] == "interp"
    assert interp_pose_rows[0]["trajectory_id"] == 5


def test_flush_pose_cnn_window_does_not_fabricate_zero_keypoints_for_invalid_pose():
    """Regression (C1): `_assemble_pose_result` (stages/pose.py) pre-allocates
    a zero-filled (n, K, 3) keypoints array and simply `continue`s past a
    detection the backend found nothing for -- so `pose_result.keypoints[i]`
    is NEVER None, even when the backend missed. Only `valid_mask[i]`
    distinguishes a real result from that fabrication. This test feeds a
    fake pose backend that returns `PoseResult(valid_mask=[False])` for one
    detection (backend miss / below min_valid_keypoints) and asserts NO
    `PoseKpt_*` keys appear in that row, and `PoseSource` is not stamped
    `"interp"` for it -- matching the old `_flush_pose_batch`'s `keypoints is
    not None and len(keypoints) > 0` semantics (no valid pose -> no
    PoseKpt_* columns, no false provenance stamp)."""
    from types import SimpleNamespace

    import numpy as np

    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.inference.config import PoseConfig
    from hydra_suite.core.inference.result import PoseResult
    from hydra_suite.core.inference.stages.pose import PoseModel
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result

    class _FakeBackend:
        def predict_batch(self, crops):
            # Backend "found nothing" for this detection: `_assemble_pose_result`
            # leaves the pre-allocated zero row untouched and valid_mask False.
            return [
                PoseResult(
                    keypoints=np.zeros((1, 1, 3), dtype=np.float32),
                    valid_mask=np.array([False]),
                )
                for _ in crops
            ]

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)
    pose_model = PoseModel(
        backend=_FakeBackend(), n_keypoints=1, keypoint_names=["head"]
    )
    cfg = SimpleNamespace(pose=PoseConfig(), cnn_phases=[])

    task = {
        "frame_id": 1,
        "cx": 32.0,
        "cy": 32.0,
        "w": 20.0,
        "h": 8.0,
        "theta": 0.0,
        "traj_id": 5,
        "interp_index": 1,
        "interp_from": (0, 2),
        "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    interp_pose_rows = []
    ic._flush_pose_cnn_window(
        pending_frames=[frame],
        pending_obbs=[obb],
        pending_tasks_by_frame=[[task]],
        pose_model=pose_model,
        cnn_models=[],
        cnn_labels=[],
        cfg=cfg,
        runtime=None,
        geometry=geometry,
        interp_pose_rows=interp_pose_rows,
        interp_cnn_rows={},
        profiler=None,
    )
    assert len(interp_pose_rows) == 1
    row = interp_pose_rows[0]
    assert row["PoseSource"] != "interp"
    assert not any(k.startswith("PoseKpt_") for k in row)
    assert row["PoseNumKeypoints"] == 0


def test_flush_pose_cnn_window_stamps_cnn_source_interp_with_argmax_class():
    """flatten_cnn_prediction_row expects pre-computed argmax class name +
    confidence per factor, NOT raw probability vectors -- confirmed by
    reading its body (export.py:141-161): it indexes class_names[idx] and
    confidences[idx] directly with no argmax of its own. This test feeds a
    fake CNN backend whose raw probabilities peak at index 1 ("b") and
    asserts the flattened row carries that argmax class + its probability,
    proving the adapter in ``_flush_pose_cnn_window`` does the argmax
    itself before calling ``flatten_cnn_prediction_row``.
    """
    from types import SimpleNamespace

    import numpy as np

    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.inference.config import CNNConfig, PoseConfig
    from hydra_suite.core.inference.stages.cnn import CNNModel
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result

    class _FakeCNNBackend:
        def predict_batch(self, crops):
            # one factor ("flat"), two classes; class index 1 ("b") wins.
            return [[[0.1, 0.9]] for _ in crops]

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)
    cnn_model = CNNModel(
        backend=_FakeCNNBackend(),
        input_size=(32, 32),
        factor_names=["flat"],
        factor_class_names=[["a", "b"]],
    )
    cnn_cfg = CNNConfig(label="identity", model_path="unused.onnx")
    cfg = SimpleNamespace(pose=PoseConfig(), cnn_phases=[cnn_cfg])

    task = {
        "frame_id": 1,
        "cx": 32.0,
        "cy": 32.0,
        "w": 20.0,
        "h": 8.0,
        "theta": 0.0,
        "traj_id": 5,
        "interp_index": 1,
        "interp_from": (0, 2),
        "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    interp_cnn_rows = {}
    ic._flush_pose_cnn_window(
        pending_frames=[frame],
        pending_obbs=[obb],
        pending_tasks_by_frame=[[task]],
        pose_model=None,
        cnn_models=[cnn_model],
        cnn_labels=["identity"],
        cfg=cfg,
        runtime=None,
        geometry=geometry,
        interp_pose_rows=[],
        interp_cnn_rows=interp_cnn_rows,
        profiler=None,
    )
    rows = interp_cnn_rows["identity"]
    assert len(rows) == 1
    row = rows[0]
    assert row["trajectory_id"] == 5
    assert row["CNN_identity_Source"] == "interp"
    assert row["CNN_identity_Class"] == "b"
    assert row["CNN_identity_Conf"] == pytest.approx(0.9)


def test_detect_apriltags_in_frame_writes_tag_source_via_run_apriltag(monkeypatch):
    import numpy as np

    from hydra_suite.core.inference.config import AprilTagConfig
    from hydra_suite.core.inference.result import AprilTagResult
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result

    task = {
        "frame_id": 1,
        "cx": 32.0,
        "cy": 32.0,
        "w": 20.0,
        "h": 8.0,
        "theta": 0.0,
        "traj_id": 5,
        "interp_index": 1,
        "interp_from": (0, 2),
        "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def _fake_run_apriltag(cpu_crops, obb_result, model, config):
        return AprilTagResult(
            tag_ids=[7],
            det_indices=[0],
            centers=np.array([[32.0, 32.0]], dtype=np.float32),
            corners=np.zeros((1, 4, 2), dtype=np.float32),
        )

    monkeypatch.setattr(ic, "run_apriltag", _fake_run_apriltag, raising=False)

    interp_tag_rows = []
    ic._detect_apriltags_in_frame(
        apriltag_model=object(),
        cfg=AprilTagConfig(enabled=True),
        frame=frame,
        obb=obb,
        tasks=[task],
        interp_tag_rows=interp_tag_rows,
    )
    assert interp_tag_rows == [{"frame_id": 1, "trajectory_id": 5, "tag_id": 7}]


def test_write_interpolation_artifacts_pose_csv_includes_pose_source(tmp_path):
    """Regression: pose_fieldnames must include "PoseSource" (Finding 4).

    ``_flush_pose_cnn_window`` stamps every pose row with
    ``"PoseSource": "interp"``, but ``write_csv_artifact`` uses
    ``csv.DictWriter(..., fieldnames=...)`` with the default
    ``extrasaction="raise"``, wrapped in a bare ``except Exception:
    return None`` (``core/post/merge.py``). Without "PoseSource" in
    ``pose_fieldnames``, a real pose row (which always carries that key)
    would silently fail to write -- no error, just a missing
    interpolated_pose.csv. Also checks the trimmed tag fieldnames
    (interpolated_tags.csv must be exactly frame_id/trajectory_id/tag_id,
    no center_x/center_y/hamming).
    """
    from types import SimpleNamespace

    from hydra_suite.core.post.interpolated_crops import _write_interpolation_artifacts

    gen = SimpleNamespace(run_dir=tmp_path, crops_dir=None)

    pose_row = {
        "frame_id": 1,
        "trajectory_id": 5,
        "filename": "x.png",
        "PoseSource": "interp",
        "PoseKpt_head_X": 1.0,
        "PoseKpt_head_Y": 2.0,
        "PoseKpt_head_Conf": 0.9,
    }
    tag_row = {"frame_id": 1, "trajectory_id": 5, "tag_id": 7}

    result = _write_interpolation_artifacts(
        gen,
        save_interpolated_outputs=True,
        cache_interpolated_artifacts=False,
        interp_rows=[],
        roi_rows=[],
        roi_corners=[],
        interp_pose_rows=[pose_row],
        interp_tag_rows=[tag_row],
        interp_cnn_rows={},
        interp_headtail_rows=[],
        pose_kpt_labels=["head"],
    )

    assert result["pose_csv_path"] is not None
    pose_header = result["pose_csv_path"].read_text().splitlines()[0]
    assert "PoseSource" in pose_header.split(",")

    assert result["tag_csv_path"] is not None
    tag_header = result["tag_csv_path"].read_text().splitlines()[0]
    assert tag_header == "frame_id,trajectory_id,tag_id"


def test_run_frame_tasks_loop_flushes_mid_loop_and_at_end(monkeypatch):
    """Windowing test for _run_frame_tasks_loop (Finding 5).

    5 needed frames, window_batch_size=3: frames 1-3 trigger a mid-loop
    flush (pending count hits the threshold at idx=3), leaving frame 4's
    task alone in the pending buffer (count=1, under threshold). The
    prefetcher then fails to read frame 5 (``ret=False``), which hits the
    loop's ``continue`` BEFORE the old flush-trigger check
    (``len(pending) >= window_batch_size or idx == total_frames``) is ever
    evaluated for idx=5 -- so pre-Finding-1-fix, frame 4's still-pending
    task is silently dropped with no error (the exact bug: the flush
    trigger only fired from *inside* a loop iteration, and this iteration's
    body exits early via `continue`). The Finding-1 fix's unconditional
    flush after the loop must catch it. Asserting frame 4's row (the LAST
    frame whose task actually reaches the pending buffer) is present is
    the point of this test, not just an early frame's.
    """
    import types

    import numpy as np

    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.inference.config import PoseConfig
    from hydra_suite.core.inference.result import PoseResult
    from hydra_suite.core.inference.stages.pose import PoseModel
    from hydra_suite.core.post import interpolated_crops as ic

    class _FakeBackend:
        def predict_batch(self, crops):
            return [
                PoseResult(
                    keypoints=np.zeros((1, 1, 3), dtype=np.float32),
                    valid_mask=np.array([True]),
                )
                for _ in crops
            ]

    params = {"RUNTIME_TIER": "cpu", "INTERP_POSE_INFERENCE_BATCH_SIZE": 3}
    geometry = canonical_geometry_from_params(params)
    pose_model = PoseModel(
        backend=_FakeBackend(), n_keypoints=1, keypoint_names=["head"]
    )
    cfg = types.SimpleNamespace(pose=PoseConfig(), cnn_phases=[], apriltag=None)

    def _task(frame_id, traj_id):
        return {
            "frame_id": frame_id,
            "cx": 32.0,
            "cy": 32.0,
            "w": 20.0,
            "h": 8.0,
            "theta": 0.0,
            "traj_id": traj_id,
            "interp_index": 1,
            "interp_from": (0, 2),
            "interp_total": 1,
        }

    frame_tasks = {
        1: [_task(1, 101)],
        2: [_task(2, 102)],
        3: [_task(3, 103)],
        4: [_task(4, 104)],  # left pending (count=1) after the idx=3 flush
        5: [_task(5, 105)],  # its own read fails, never reaches pending
    }
    frame_by_idx = {f: np.zeros((64, 64, 3), dtype=np.uint8) for f in frame_tasks}
    # Frame 5's read reports failure (ret=False) -- triggers the loop's
    # `continue` before the old in-loop flush-trigger check is reached.
    read_plan = [
        (1, True, frame_by_idx[1]),
        (2, True, frame_by_idx[2]),
        (3, True, frame_by_idx[3]),
        (4, True, frame_by_idx[4]),
        (5, False, None),
    ]

    class _FakePrefetcher:
        def __init__(self, plan):
            self._items = iter(plan)

        def start(self):
            pass

        def stop(self):
            pass

        def read(self):
            try:
                return next(self._items)
            except StopIteration:
                return None

    monkeypatch.setattr(
        ic, "_build_prefetcher", lambda cap, nf, tf: _FakePrefetcher(read_plan)
    )

    interp_pose_rows: list = []
    result = ic._run_frame_tasks_loop(
        params,
        None,
        None,
        frame_tasks,
        None,
        None,
        False,
        geometry,
        None,
        cfg,
        None,
        pose_model,
        [],
        [],
        None,
        None,
        0,
        [],
        [],
        [],
        interp_pose_rows,
        [],
        {},
        [],
        None,
    )

    assert result is not None
    traj_ids = {row["trajectory_id"] for row in interp_pose_rows}
    # Frames 1-3 came from the mid-loop flush; frame 4 (pending, never
    # flushed in-loop because frame 5's `continue` skips the old trigger
    # check) must survive via the trailing/final flush.
    assert traj_ids == {101, 102, 103, 104}
    assert 104 in traj_ids


def test_flush_headtail_window_writes_heading_rows(monkeypatch):
    import types

    import numpy as np

    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.inference.config import HeadTailConfig
    from hydra_suite.core.inference.stages.headtail import HeadTailModel
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result

    class _FakeBackend:
        def predict_batch(self, crops):
            return [[np.array([0.9, 0.1])] for _ in crops]

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)
    headtail_model = HeadTailModel(
        backend=_FakeBackend(), input_size=(32, 32), class_names=["right", "left"]
    )
    task = {
        "frame_id": 1,
        "cx": 32.0,
        "cy": 32.0,
        "w": 20.0,
        "h": 8.0,
        "theta": 0.0,
        "traj_id": 5,
        "interp_index": 1,
        "interp_from": (0, 2),
        "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    interp_headtail_rows = []
    headtail_cfg = HeadTailConfig(
        model_path="unused-in-test",
        candidate_confidence_threshold=0.0,
        confidence_threshold=0.0,
    )
    cfg = types.SimpleNamespace(headtail=headtail_cfg)
    ic._flush_headtail_window(
        pending_frames=[frame],
        pending_obbs=[obb],
        pending_tasks_by_frame=[[task]],
        headtail_model=headtail_model,
        cfg=cfg,
        runtime=None,
        geometry=geometry,
        interp_headtail_rows=interp_headtail_rows,
    )
    assert len(interp_headtail_rows) == 1
    assert interp_headtail_rows[0]["trajectory_id"] == 5


def test_cleanup_backends_closes_underlying_model_backend_not_wrapper():
    """Regression (I2): PoseModel/CNNModel/HeadTailModel/AprilTagModel's own
    `.close()` are all no-ops (shared infra also used by
    Pipeline/InferenceRunner for real detections, out of scope to change
    here) -- for the SLEAP service backend in particular, only
    `model.backend.close()` reaches `shutdown_sleap_service()` and actually
    terminates the subprocess. `_cleanup_backends` must reach into
    `model.backend` (or `.detector` for AprilTag) and close/release THAT
    object, not just call the wrapper's no-op `.close()`."""
    from hydra_suite.core.post.interpolated_crops import _cleanup_backends

    class _FakeUnderlyingBackend:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeWrapperModel:
        """Mimics PoseModel/CNNModel/HeadTailModel: a `.backend` attribute
        plus a no-op `.close()` (the wrapper's own close does nothing)."""

        def __init__(self, backend):
            self.backend = backend

        def close(self):
            pass  # the real wrapper no-op this bug is about

    class _FakeAprilTagModel:
        """AprilTagModel's field is named `.detector`, not `.backend`."""

        def __init__(self, detector):
            self.detector = detector

        def close(self):
            pass

    pose_backend = _FakeUnderlyingBackend()
    cnn_backend = _FakeUnderlyingBackend()
    headtail_backend = _FakeUnderlyingBackend()
    apriltag_detector = _FakeUnderlyingBackend()

    pose_model = _FakeWrapperModel(pose_backend)
    cnn_model = _FakeWrapperModel(cnn_backend)
    headtail_model = _FakeWrapperModel(headtail_backend)
    apriltag_model = _FakeAprilTagModel(apriltag_detector)

    _cleanup_backends(
        cap=None,
        detection_cache=None,
        pose_model=pose_model,
        apriltag_model=apriltag_model,
        cnn_models=[cnn_model],
        headtail_model=headtail_model,
    )

    assert pose_backend.closed is True
    assert cnn_backend.closed is True
    assert headtail_backend.closed is True
    assert apriltag_detector.closed is True


def test_init_interpolation_backends_degrades_on_config_build_failure(
    monkeypatch, caplog
):
    """Regression (I3): a `build_inference_config_from_params` failure (e.g.
    PoseModelUnresolvedError) used to propagate unhandled up to
    `run_interpolated_crops`'s outer blanket `except Exception:`, which had
    NO logging -- the entire post-pass silently vanished. Monkeypatch the
    config builder to raise and confirm `_init_interpolation_backends`
    degrades to the "no models available" shape instead of propagating, and
    logs the failure."""
    import logging

    from hydra_suite.core.canonicalization.geometry import (
        canonical_geometry_from_params,
    )
    from hydra_suite.core.inference import config as inference_config
    from hydra_suite.core.post.interpolated_crops import _init_interpolation_backends

    def _boom(params):
        raise RuntimeError("simulated config-build failure")

    monkeypatch.setattr(inference_config, "build_inference_config_from_params", _boom)

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)

    with caplog.at_level(logging.ERROR):
        result = _init_interpolation_backends(params, "/tmp", geometry)

    (
        cfg,
        runtime,
        pose_model,
        apriltag_model,
        cnn_models,
        cnn_labels,
        headtail_model,
    ) = result

    assert pose_model is None
    assert apriltag_model is None
    assert cnn_models == []
    assert cnn_labels == []
    assert headtail_model is None
    assert cfg.apriltag.enabled is False
    assert any(
        "failed to build the inference config" in rec.message for rec in caplog.records
    )


def test_run_interpolated_crops_logs_on_unhandled_exception(monkeypatch, caplog):
    """Regression (I3): the outer blanket `except Exception:` in
    `run_interpolated_crops` must log before returning the empty payload, so
    ANY future silent-failure class is at least visible in logs."""
    import logging

    from hydra_suite.core.post import interpolated_crops as ic

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated unhandled failure")

    monkeypatch.setattr(ic, "_validate_and_setup", _boom)

    with caplog.at_level(logging.ERROR):
        result = ic.run_interpolated_crops("csv", "video", "cache", {})

    assert result == {"saved": 0, "gaps": 0}
    assert any("Interpolated post-pass failed" in rec.message for rec in caplog.records)
