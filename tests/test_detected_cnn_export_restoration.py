"""Detection-keyed CNN export: the columns the Gen-2 migration silently dropped.

`CNN_<label>[_<factor>]_Class`/`_Conf` used to reach the rich export from a V3
``CNNIdentityCache``. That cache's writer disappeared during the inference
migration, and its reader was ``os.path.exists``-guarded, so the columns simply
stopped appearing -- starving everything derived from them (the identity
evidence summary, ``UniqueIdentityKey``, and the non-identifying-class report).
These tests pin the restored path against the cache the CNN stage really writes.
"""

import json

import numpy as np
import pandas as pd

from hydra_suite.core.individual.properties.export import (
    augment_trajectories_with_detected_cnn_cache,
    build_detected_cnn_lookup_dataframe_from_cache,
)
from hydra_suite.core.inference.result import DETECTION_ID_STRIDE


def _write_cache(path, *, frames, dets, probs, factor_names, class_names):
    counts = np.array([len(c) for c in class_names], dtype=np.int32)
    np.savez(
        path,
        frame_indices=np.array(frames, dtype=np.int32),
        det_indices=np.array(dets, dtype=np.int32),
        factor_names_json=np.array([json.dumps(factor_names)]),
        class_names_json=np.array([json.dumps(class_names)]),
        class_counts=counts,
        probabilities=np.array(probs, dtype=np.float32),
    )
    return str(path)


def _two_factor_cache(tmp_path):
    # 2 detections x 2 factors x 3 classes; padding column is NaN.
    probs = [
        [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]],
        [[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]],
    ]
    return _write_cache(
        tmp_path / "cnn_colortag.npz",
        frames=[0, 0],
        dets=[0, 1],
        probs=probs,
        factor_names=["flat", "flat_1"],
        class_names=[["red", "green", "blue"], ["red", "green", "blue"]],
    )


def test_lookup_takes_per_factor_argmax_and_confidence(tmp_path):
    df = build_detected_cnn_lookup_dataframe_from_cache(
        _two_factor_cache(tmp_path), "colortag"
    )
    assert df["CNN_colortag_flat_Class"].tolist() == ["green", "red"]
    assert df["CNN_colortag_flat_1_Class"].tolist() == ["red", "blue"]
    np.testing.assert_allclose(
        df["CNN_colortag_flat_Conf"].tolist(), [0.7, 0.8], rtol=1e-6
    )


def test_detection_id_uses_the_global_stride_encoding(tmp_path):
    path = _write_cache(
        tmp_path / "cnn_x.npz",
        frames=[3, 7],
        dets=[2, 5],
        probs=[[[1.0, 0.0]], [[0.0, 1.0]]],
        factor_names=["flat"],
        class_names=[["a", "b"]],
    )
    df = build_detected_cnn_lookup_dataframe_from_cache(path, "x")
    assert df["_cnn_detection_id"].tolist() == [
        3 * DETECTION_ID_STRIDE + 2,
        7 * DETECTION_ID_STRIDE + 5,
    ]


def test_single_factor_model_keeps_the_flat_column_names(tmp_path):
    path = _write_cache(
        tmp_path / "cnn_beh.npz",
        frames=[0],
        dets=[0],
        probs=[[[0.2, 0.8]]],
        factor_names=["flat"],
        class_names=[["walk", "rest"]],
    )
    df = build_detected_cnn_lookup_dataframe_from_cache(path, "beh")
    assert "CNN_beh_Class" in df.columns
    assert df["CNN_beh_Class"].tolist() == ["rest"]


def test_all_nan_factor_row_stays_unlabelled(tmp_path):
    """Ragged padding must not argmax to class 0 and invent a prediction."""
    path = _write_cache(
        tmp_path / "cnn_p.npz",
        frames=[0, 0],
        dets=[0, 1],
        probs=[[[0.9, 0.1]], [[np.nan, np.nan]]],
        factor_names=["flat"],
        class_names=[["a", "b"]],
    )
    df = build_detected_cnn_lookup_dataframe_from_cache(path, "p")
    assert df["CNN_p_Class"].tolist()[0] == "a"
    assert pd.isna(df["CNN_p_Class"].tolist()[1])
    assert pd.isna(df["CNN_p_Conf"].tolist()[1])


def test_merge_joins_onto_trajectory_rows_by_detection(tmp_path):
    traj = pd.DataFrame(
        {
            "FrameID": [0, 0, 1],
            "DetectionID": [0, 1, np.nan],  # third row is interpolated: no detection
            "TrajectoryID": [0, 1, 0],
        }
    )
    out = augment_trajectories_with_detected_cnn_cache(
        traj, _two_factor_cache(tmp_path), "colortag"
    )
    assert out["CNN_colortag_flat_Class"].tolist()[:2] == ["green", "red"]
    assert pd.isna(out["CNN_colortag_flat_Class"].tolist()[2])
    assert len(out) == len(traj)


def test_missing_cache_is_a_no_op_not_a_crash(tmp_path):
    traj = pd.DataFrame({"FrameID": [0], "DetectionID": [0], "TrajectoryID": [0]})
    out = augment_trajectories_with_detected_cnn_cache(
        traj, str(tmp_path / "absent.npz"), "colortag"
    )
    assert len(out) == 1
