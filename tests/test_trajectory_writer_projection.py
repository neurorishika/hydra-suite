import math

import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.trajectory_writer import project_user_tracks


def _base_df():
    return pd.DataFrame(
        {
            "TrackID": [0, 0],
            "Index": [0, 1],
            "TrajectoryID": [3, 3],
            "FrameID": [0, 10],
            "X": [1.4, 2.6],
            "Y": [5.0, 6.0],
            "Theta": [0.0, -math.pi / 2],  # 0 rad, -90 deg
            "State": ["active", "occluded"],
            "DetectionConfidence": [0.9, 0.8],
            "AssignmentConfidence": [0.5, 0.4],
            "PositionUncertainty": [1.1, 1.2],
        }
    )


def test_core_columns_and_conversions():
    out = project_user_tracks(_base_df(), fps=10.0)
    assert list(out.columns) == [
        "id",
        "frame",
        "time_s",
        "x",
        "y",
        "heading_deg",
        "state",
        "detection_confidence",
    ]
    assert out["id"].tolist() == [3, 3]
    assert out["time_s"].tolist() == [0.0, 1.0]  # frame/fps
    # -pi/2 rad -> 270 deg (normalized to [0,360))
    assert out["heading_deg"].round(3).tolist() == [0.0, 270.0]
    # tracer-only confidences dropped
    assert "AssignmentConfidence" not in out.columns
    assert "PositionUncertainty" not in out.columns


def test_identity_columns_appear_only_when_final_present():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", ""]
    df[C.FINAL_SMOOTHED_LABEL] = ["antA", "antB"]
    df[C.FINAL_CONFIDENCE] = [0.7, 0.6]
    df[C.FINAL_SOURCE] = ["realtime", "offline"]
    out = project_user_tracks(df, fps=10.0)
    assert out["identity"].tolist() == [
        "antA",
        "antB",
    ]  # empty Final falls back to Smoothed
    assert out["identity_confidence"].tolist() == [0.7, 0.6]
    assert out["identity_source"].tolist() == ["realtime", "offline"]


def test_pose_triples_appear_only_when_pose_present():
    df = _base_df()
    df["PoseKpt_head_X"] = [1.0, 2.0]
    df["PoseKpt_head_Y"] = [3.0, 4.0]
    df["PoseKpt_head_Conf"] = [0.9, 0.8]
    out = project_user_tracks(df, fps=10.0)
    assert out["head_x"].tolist() == [1.0, 2.0]
    assert out["head_y"].tolist() == [3.0, 4.0]
    assert out["head_conf"].tolist() == [0.9, 0.8]


def test_no_fps_yields_nan_time():
    out = project_user_tracks(_base_df(), fps=None)
    assert out["time_s"].isna().all()


def test_identity_columns_absent_when_identity_did_not_run():
    """Even with C.FINAL_LABEL present, identity_ran=False suppresses the block.

    Rich-export's identity postprocessing unconditionally resolves
    C.FINAL_LABEL to a placeholder ("unknown") whenever the identity
    pipeline is enabled, regardless of whether a real method executed --
    column presence alone is not a reliable "identity actually ran" signal.
    """
    df = _base_df()
    df[C.FINAL_LABEL] = ["unknown", "unknown"]
    df[C.FINAL_CONFIDENCE] = [0.0, 0.0]
    df[C.FINAL_SOURCE] = ["", ""]
    out = project_user_tracks(df, fps=10.0, identity_ran=False)
    assert "identity" not in out.columns
    assert "identity_confidence" not in out.columns
    assert "identity_source" not in out.columns


def test_identity_columns_present_when_identity_ran_true():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", "antB"]
    df[C.FINAL_CONFIDENCE] = [0.7, 0.6]
    df[C.FINAL_SOURCE] = ["realtime", "offline"]
    out = project_user_tracks(df, fps=10.0, identity_ran=True)
    assert out["identity"].tolist() == ["antA", "antB"]
    assert out["identity_confidence"].tolist() == [0.7, 0.6]
    assert out["identity_source"].tolist() == ["realtime", "offline"]


def test_identity_id_travels_with_the_label():
    """The clean CSV must carry the resolved *slot*, not only the label.

    A non-identifying label (an untagged animal) is deliberately shared by
    several tracks; only ``IdentityFinalID == 0`` distinguishes "this label
    names no individual" from a resolved identity. Without the slot in this
    file, grouping by ``identity`` merges every untagged animal into one.
    """
    df = _base_df()
    df[C.FINAL_LABEL] = ["notag_notag", "antA"]
    df[C.FINAL_ID] = [0, 4]
    df[C.FINAL_CONFIDENCE] = [0.3, 0.9]
    df[C.FINAL_SOURCE] = ["nonidentifying", "offline"]
    out = project_user_tracks(df, fps=10.0)
    assert list(out.columns)[8:12] == [
        "identity",
        "identity_id",
        "identity_confidence",
        "identity_source",
    ]
    assert out["identity_id"].tolist() == [0, 4]
    assert str(out["identity_id"].dtype) == "Int64"


def test_identity_id_absent_when_identity_did_not_run():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", "antA"]
    df[C.FINAL_ID] = [1, 1]
    out = project_user_tracks(df, fps=10.0, identity_ran=False)
    assert "identity_id" not in out.columns


def test_identity_id_omitted_when_the_column_was_never_written():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", "antA"]
    out = project_user_tracks(df, fps=10.0)
    assert "identity" in out.columns
    assert "identity_id" not in out.columns


def _classifiers():
    return [
        {"label": "colortag", "unique_identifier": True},
        {"label": "behavior", "unique_identifier": False},
    ]


def _df_with_classifiers():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", "antA"]
    df[C.FINAL_ID] = [1, 1]
    df["CNN_colortag_Class"] = ["red", "red"]
    df["CNN_colortag_Conf"] = [0.9, 0.8]
    df["CNN_behavior_Class"] = ["walk", "groom"]
    df["CNN_behavior_Conf"] = [0.7, 0.6]
    return df


def test_non_identity_classifier_reaches_the_user_export():
    """A behavior classifier is output, not identity -- it must not be discarded.

    Without this the classifier runs, costs inference time, and never reaches
    a User-mode user in any form.
    """
    out = project_user_tracks(
        _df_with_classifiers(), fps=10.0, cnn_classifiers=_classifiers()
    )
    assert out["behavior_class"].tolist() == ["walk", "groom"]
    assert out["behavior_conf"].tolist() == [0.7, 0.6]


def test_identity_head_columns_stay_out_of_the_user_export():
    """The identity head's channel is `identity`; its per-frame calls are evidence."""
    out = project_user_tracks(
        _df_with_classifiers(), fps=10.0, cnn_classifiers=_classifiers()
    )
    assert "colortag_class" not in out.columns
    assert "colortag_conf" not in out.columns
    assert out["identity"].tolist() == ["antA", "antA"]


def test_multi_factor_non_identity_classifier_keeps_one_column_per_factor():
    df = _base_df()
    df["CNN_state_front_Class"] = ["a", "b"]
    df["CNN_state_front_Conf"] = [0.5, 0.6]
    df["CNN_state_back_Class"] = ["c", "d"]
    df["CNN_state_back_Conf"] = [0.7, 0.8]
    out = project_user_tracks(
        df, fps=10.0, cnn_classifiers=[{"label": "state", "unique_identifier": False}]
    )
    assert out["state_front_class"].tolist() == ["a", "b"]
    assert out["state_back_conf"].tolist() == [0.7, 0.8]


def test_no_classifier_config_leaves_the_schema_untouched():
    out = project_user_tracks(_df_with_classifiers(), fps=10.0)
    assert not [c for c in out.columns if c.endswith(("_class", "_conf"))]


def test_arena_id_travels_into_the_clean_export():
    """Multi-arena runs must carry the arena into the User-mode CSV.

    Trajectory ids are globally unique but arena-blind, so without this
    column a 24-arena plate exports as one undifferentiated pool and no
    per-arena analysis is possible from this file.
    """
    df = _base_df()
    df["arena_id"] = [7, 7]
    out = project_user_tracks(df, fps=10.0)
    assert list(out.columns)[:3] == ["id", "arena_id", "frame"]
    assert out["arena_id"].tolist() == [7, 7]
    assert str(out["arena_id"].dtype) == "Int64"


def test_arena_id_absent_for_single_arena_runs():
    out = project_user_tracks(_base_df(), fps=10.0)
    assert "arena_id" not in out.columns


def test_smoothed_label_carries_the_smoothed_confidence():
    """A row whose label falls back to Smoothed reports the Smoothed score."""
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", ""]
    df[C.FINAL_SMOOTHED_LABEL] = ["antA", "antB"]
    df[C.FINAL_CONFIDENCE] = [0.7, 0.0]
    df[C.FINAL_SMOOTHED_CONFIDENCE] = [0.1, 0.6]
    df[C.FINAL_SOURCE] = ["realtime", "offline"]
    out = project_user_tracks(df, fps=10.0)
    assert out["identity"].tolist() == ["antA", "antB"]
    # row 0 kept its Final label -> Final confidence; row 1 fell back.
    assert out["identity_confidence"].tolist() == [0.7, 0.6]


def test_empty_final_and_empty_smoothed_keeps_final_confidence():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", ""]
    df[C.FINAL_SMOOTHED_LABEL] = ["antA", "  "]
    df[C.FINAL_CONFIDENCE] = [0.7, 0.0]
    df[C.FINAL_SMOOTHED_CONFIDENCE] = [0.1, 0.9]
    out = project_user_tracks(df, fps=10.0)
    assert out["identity_confidence"].tolist() == [0.7, 0.0]


def test_heading_is_directed_travels_with_heading():
    """`heading_deg` mixes true headings and body axes; the flag separates them."""
    df = _base_df()
    df["HeadingIsDirected"] = [True, False]
    out = project_user_tracks(df, fps=10.0)
    cols = list(out.columns)
    assert cols[cols.index("heading_deg") + 1] == "heading_is_directed"
    assert out["heading_is_directed"].tolist() == [True, False]
    assert str(out["heading_is_directed"].dtype) == "boolean"


def test_heading_is_directed_absent_without_head_tail():
    out = project_user_tracks(_base_df(), fps=10.0)
    assert "heading_is_directed" not in out.columns


def test_heading_is_directed_keeps_na_for_detectionless_rows():
    """No detection = no evidence; that must not read as "not directed"."""
    df = _base_df()
    df["HeadingIsDirected"] = [True, float("nan")]
    out = project_user_tracks(df, fps=10.0)
    assert out["heading_is_directed"].tolist()[0] is True
    assert pd.isna(out["heading_is_directed"].tolist()[1])


def test_heading_is_directed_survives_a_csv_round_trip():
    """Strings must map by value: any non-empty string is truthy to bool()."""
    df = _base_df()
    df["HeadingIsDirected"] = ["True", "False"]
    out = project_user_tracks(df, fps=10.0)
    assert out["heading_is_directed"].tolist() == [True, False]
