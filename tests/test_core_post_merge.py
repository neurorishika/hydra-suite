import pandas as pd

from hydra_suite.core.post.merge import (
    convert_resolved_to_dataframe,
    merge_trajectories,
    rescale_coordinates,
    write_csv_artifact,
)


def _traj(tid, xs):
    return pd.DataFrame(
        {
            "TrajectoryID": tid,
            "X": xs,
            "Y": [10.0] * len(xs),
            "Theta": [0.0] * len(xs),
            "FrameID": list(range(len(xs))),
        }
    )


def test_convert_resolved_reassigns_trajectory_ids():
    out = convert_resolved_to_dataframe([_traj(99, [1.0, 2.0]), _traj(99, [3.0, 4.0])])
    assert isinstance(out, pd.DataFrame)
    assert sorted(out["TrajectoryID"].unique().tolist()) == [0, 1]


def test_rescale_coordinates_divides_by_resize_factor():
    df = _traj(0, [10.0, 20.0])
    out = rescale_coordinates(df, resize_factor=0.5)
    assert out["X"].tolist() == [20.0, 40.0]


def test_merge_reports_progress_and_returns_dataframe():
    seen = []
    merged = merge_trajectories(
        _traj(0, [1.0, 2.0, 3.0]),
        _traj(0, [1.0, 2.0, 3.0]),
        total_frames=3,
        params={"MIN_TRAJECTORY_LENGTH": 1},
        resize_factor=1.0,
        interp_method="none",
        max_gap=1,
        progress=lambda v, m: seen.append((v, m)),
    )
    assert isinstance(merged, pd.DataFrame)
    assert (100, "Merge complete!") in seen


def test_merge_honours_should_stop_before_completion():
    merged = merge_trajectories(
        _traj(0, [1.0, 2.0]),
        _traj(0, [1.0, 2.0]),
        total_frames=2,
        params={},
        resize_factor=1.0,
        interp_method="none",
        max_gap=1,
        should_stop=lambda: True,
    )
    assert merged is None


def test_write_csv_artifact_roundtrip(tmp_path):
    p = tmp_path / "a.csv"
    out = write_csv_artifact(str(p), ["k"], [{"k": 1}, {"k": 2}])
    assert out == str(p)
    assert p.read_text().splitlines()[0] == "k"
