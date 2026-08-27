"""relink_and_export_rich_csv must relink BEFORE resolving identity, resolve
exactly once, densify chains, and write a frame with one identity per track."""

import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post import rich_export


def test_relink_then_resolve_order_and_single_solve(tmp_path, monkeypatch):
    final = tmp_path / "clip_final.csv"
    base = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 1, 1],
            "FrameID": [1, 2, 5, 6],
            "X": [0.0, 1.0, 4.0, 5.0],
            "Y": [0.0, 0.0, 0.0, 0.0],
            "Theta": 0.0,
            "State": "active",
            "DetectionID": [1, 2, 3, 4],
        }
    )
    base.to_csv(final, index=False)
    calls = []

    def fake_build(
        final_csv_path,
        state,
        *,
        params,
        min_valid_conf,
        ignore_keypoints,
        identity_evidence_cache_path=None,
        resolve=True,
    ):
        calls.append(("build", resolve))
        return pd.read_csv(final_csv_path)

    def fake_relink(df, params):
        calls.append(("relink", df["TrajectoryID"].nunique()))
        out = df.copy()
        out["TrajectoryID"] = 0
        return out

    def fake_resolve(df, params, identity_evidence_cache_path=None):
        calls.append(("resolve", df["FrameID"].tolist()))
        out = df.copy()
        out[C.FINAL_LABEL] = "a"
        out[C.FINAL_ID] = 1
        out[C.FINAL_SOURCE] = "offline"
        out[C.FINAL_CONFIDENCE] = 0.9
        return out

    monkeypatch.setattr(rich_export, "build_rich_export_dataframe", fake_build)
    import hydra_suite.core.post.processing as P

    monkeypatch.setattr(P, "relink_trajectories_with_pose_by_arena", fake_relink)
    import hydra_suite.core.individual.postprocess_df as PD

    monkeypatch.setattr(PD, "resolve_identity", fake_resolve)

    params = {"FINAL_INTERPOLATION_MAX_GAP": 11, "ENABLE_TRACKLET_RELINKING": True}
    out = rich_export.relink_and_export_rich_csv(
        str(final),
        state=None,
        params=params,
        min_valid_conf=0.2,
        ignore_keypoints=None,
        debug_mode=True,
        fps=10.0,
    )
    assert calls[0] == ("build", False)
    assert calls[1][0] == "relink"
    assert calls[2][0] == "resolve" and calls[2][1] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]  # densified before resolve
    assert sum(1 for c in calls if c[0] == "resolve") == 1
    written = pd.read_csv(out)
    assert written["FrameID"].tolist() == [1, 2, 3, 4, 5, 6]
    assert written["X"].isna().sum() == 0
    assert written[C.FINAL_LABEL].nunique() == 1
