import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import _ensure_final_columns
from hydra_suite.core.post.identity_postprocess import normalize_final_source_series
from hydra_suite.core.post.trajectory_writer import write_final_trajectories


def test_none_is_explicit_token():
    assert C.IdentityFinalSource.NONE == "none"


def test_normalize_final_source_series_maps_blank_and_nan_to_none():
    s = pd.Series([np.nan, "", "  ", "offline", " tag "])
    out = normalize_final_source_series(s)
    assert out.tolist() == ["none", "none", "none", "offline", "tag"]


def test_ensure_final_columns_creates_conflict_flag_false_only_when_absent():
    df = pd.DataFrame({"TrajectoryID": [0, 0], "FrameID": [1, 2]})
    out = _ensure_final_columns(df)
    assert out[C.FINAL_CONFLICT_RESOLVED].tolist() == [False, False]
    assert out[C.FINAL_SOURCE].tolist() == ["none", "none"]
    # existing merge-time True must survive
    df2 = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [1, 2],
            C.FINAL_CONFLICT_RESOLVED: [True, np.nan],
        }
    )
    out2 = _ensure_final_columns(df2)
    assert bool(out2[C.FINAL_CONFLICT_RESOLVED].iloc[0]) is True
    assert pd.isna(
        out2[C.FINAL_CONFLICT_RESOLVED].iloc[1]
    )  # untouched; the writer fills NaN -> False


def test_written_csv_has_no_blank_source_and_boolean_conflict(tmp_path):
    final_csv = tmp_path / "clip_final.csv"
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 1],
            "FrameID": [1, 2, 1],
            "X": [1.0, 2.0, 3.0],
            "Y": [1.0, 2.0, 3.0],
            "Theta": [0.0, 0.0, 0.0],
            "State": ["active"] * 3,
            "DetectionID": [1, 2, 3],
            C.FINAL_LABEL: ["ant_a", "ant_a", "unknown"],
            C.FINAL_ID: [1, 1, 0],
            C.FINAL_CONFIDENCE: [0.9, 0.9, 0.0],
            C.FINAL_SOURCE: ["offline", "offline", np.nan],
            C.FINAL_CONFLICT_RESOLVED: [True, np.nan, np.nan],
        }
    )
    out = write_final_trajectories(df, str(final_csv), debug_mode=True, fps=10.0)
    written = pd.read_csv(out)
    assert written[C.FINAL_SOURCE].tolist() == ["offline", "offline", "none"]
    assert written[C.FINAL_CONFLICT_RESOLVED].tolist() == [True, False, False]
