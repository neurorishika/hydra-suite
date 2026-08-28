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


def test_relink_then_resolve_collapses_disagreeing_identities(
    tmp_path, monkeypatch, caplog
):
    """Coverage for Finding 3: when the (stubbed) solver assigns genuinely
    different labels to different frames of the SAME post-relink
    TrajectoryID, relink_and_export_rich_csv must collapse the trajectory to
    its majority identity and log an ERROR naming the offending id."""
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
        return pd.read_csv(final_csv_path)

    def fake_relink(df, params):
        # Merge both original trajectories into a single TrajectoryID=0, same
        # as the order test above.
        out = df.copy()
        out["TrajectoryID"] = 0
        return out

    def fake_resolve_disagreeing(df, params, identity_evidence_cache_path=None):
        # Different label per frame within the single post-relink/densify
        # trajectory -- exercises the collapse branch that a same-label
        # stub never reaches.
        out = df.copy()
        labels = {f: ("a" if f <= 3 else "b") for f in out["FrameID"]}
        out[C.FINAL_LABEL] = out["FrameID"].map(labels)
        out[C.FINAL_ID] = out["FrameID"].map(
            {f: 1 if lab == "a" else 2 for f, lab in labels.items()}
        )
        out[C.FINAL_SOURCE] = "offline"
        out[C.FINAL_CONFIDENCE] = 0.9
        return out

    monkeypatch.setattr(rich_export, "build_rich_export_dataframe", fake_build)
    import hydra_suite.core.post.processing as P

    monkeypatch.setattr(P, "relink_trajectories_with_pose_by_arena", fake_relink)
    import hydra_suite.core.individual.postprocess_df as PD

    monkeypatch.setattr(PD, "resolve_identity", fake_resolve_disagreeing)

    params = {"FINAL_INTERPOLATION_MAX_GAP": 11, "ENABLE_TRACKLET_RELINKING": True}
    with caplog.at_level("ERROR", logger="hydra_suite.core.post.rich_export"):
        out = rich_export.relink_and_export_rich_csv(
            str(final),
            state=None,
            params=params,
            min_valid_conf=0.2,
            ignore_keypoints=None,
            debug_mode=True,
            fps=10.0,
        )

    written = pd.read_csv(out)
    assert written[C.FINAL_LABEL].nunique() == 1  # collapsed to majority
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("more than one" in r.getMessage() for r in error_records)
    assert any("0" in r.getMessage() for r in error_records)  # names TrajectoryID 0


def test_export_rich_csv_collapses_disagreeing_identities(
    tmp_path, monkeypatch, caplog
):
    """Same Finding 3 coverage, but for export_rich_csv's own
    assert_one_identity_per_trajectory / collapse_to_majority_identity call
    (the non-relink path)."""
    final = tmp_path / "clip_final.csv"
    base = pd.DataFrame(
        {
            "TrajectoryID": [2, 2, 2, 2],
            "FrameID": [1, 2, 3, 4],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 0.0, 0.0],
            "Theta": 0.0,
            "State": "active",
            "DetectionID": [1, 2, 3, 4],
        }
    )
    base.to_csv(final, index=False)

    def fake_build_resolved(
        final_csv_path,
        state,
        *,
        params,
        min_valid_conf,
        ignore_keypoints,
        identity_evidence_cache_path=None,
        resolve=True,
    ):
        out = pd.read_csv(final_csv_path)
        # Same TrajectoryID, disagreeing resolved identity across frames.
        out[C.FINAL_LABEL] = ["a", "a", "b", "b"]
        out[C.FINAL_ID] = [1, 1, 2, 2]
        out[C.FINAL_SOURCE] = "offline"
        out[C.FINAL_CONFIDENCE] = 0.9
        return out

    monkeypatch.setattr(rich_export, "build_rich_export_dataframe", fake_build_resolved)

    params = {"FINAL_INTERPOLATION_MAX_GAP": 11}
    with caplog.at_level("ERROR", logger="hydra_suite.core.post.rich_export"):
        out = rich_export.export_rich_csv(
            str(final),
            state=None,
            params=params,
            min_valid_conf=0.2,
            ignore_keypoints=None,
            debug_mode=True,
            fps=10.0,
        )

    written = pd.read_csv(out)
    assert written[C.FINAL_LABEL].nunique() == 1  # collapsed to majority
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("more than one" in r.getMessage() for r in error_records)
    assert any("2" in r.getMessage() for r in error_records)  # names TrajectoryID 2


def test_relink_honors_interpolation_method_none(tmp_path, monkeypatch):
    """I2: relink_and_export_rich_csv must NOT silently override an explicit
    interpolation_method="none" ("do not fabricate positions") with
    interpolate_trajectories's own "linear" default. Frame gaps still get
    densified (Task 5), but interior X/Y/Theta gaps must be left NaN, not
    linearly filled -- while trim_positionless_ends still runs."""
    final = tmp_path / "clip_final.csv"
    # TrajectoryID 0: frames 1,2 then a gap, then frame 5 -- densify inserts
    # frames 3,4 as position-less "occluded" rows; with method="none" they
    # must stay NaN rather than being linearly interpolated.
    base = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0],
            "FrameID": [1, 2, 5],
            "X": [0.0, 1.0, 4.0],
            "Y": [0.0, 0.0, 0.0],
            "Theta": 0.0,
            "State": "active",
            "DetectionID": [1, 2, 3],
        }
    )
    base.to_csv(final, index=False)

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
        return pd.read_csv(final_csv_path)

    def fake_relink(df, params):
        return df.copy()

    def fake_resolve(df, params, identity_evidence_cache_path=None):
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

    params = {
        "FINAL_INTERPOLATION_MAX_GAP": 11,
        "FINAL_INTERPOLATION_METHOD": "none",
    }
    out = rich_export.relink_and_export_rich_csv(
        str(final),
        state=None,
        params=params,
        min_valid_conf=0.2,
        ignore_keypoints=None,
        debug_mode=True,
        fps=10.0,
    )
    written = pd.read_csv(out)
    # Frame gaps still get densified (Task 5's design).
    assert written["FrameID"].tolist() == [1, 2, 3, 4, 5]
    # Interior gap rows (frames 3, 4) are NOT linearly filled.
    gap_rows = written[written["FrameID"].isin([3, 4])]
    assert gap_rows["X"].isna().all()
    assert gap_rows["Y"].isna().all()
    # No leading/trailing position-less rows here to trim, but the real
    # rows keep their genuine (unfilled) positions.
    assert written.loc[written["FrameID"] == 1, "X"].iloc[0] == 0.0
    assert written.loc[written["FrameID"] == 5, "X"].iloc[0] == 4.0


def test_relink_default_interpolation_method_is_linear_fill(tmp_path, monkeypatch):
    """Without FINAL_INTERPOLATION_METHOD in params (e.g. an older caller),
    relink_and_export_rich_csv preserves its prior always-linear-fill
    behavior byte-for-byte."""
    final = tmp_path / "clip_final.csv"
    base = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0],
            "FrameID": [1, 2, 5],
            "X": [0.0, 1.0, 4.0],
            "Y": [0.0, 0.0, 0.0],
            "Theta": 0.0,
            "State": "active",
            "DetectionID": [1, 2, 3],
        }
    )
    base.to_csv(final, index=False)

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
        return pd.read_csv(final_csv_path)

    def fake_relink(df, params):
        return df.copy()

    def fake_resolve(df, params, identity_evidence_cache_path=None):
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

    params = {"FINAL_INTERPOLATION_MAX_GAP": 11}
    out = rich_export.relink_and_export_rich_csv(
        str(final),
        state=None,
        params=params,
        min_valid_conf=0.2,
        ignore_keypoints=None,
        debug_mode=True,
        fps=10.0,
    )
    written = pd.read_csv(out)
    assert written["X"].isna().sum() == 0
    gap_rows = written[written["FrameID"].isin([3, 4])]
    assert not gap_rows["X"].isna().any()
