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
