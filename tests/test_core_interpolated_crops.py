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
            return [
                PoseResult(
                    keypoints=np.zeros((1, 1, 3), dtype=np.float32),
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
