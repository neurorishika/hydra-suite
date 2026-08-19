from hydra_suite.core.individual.identity.resolve import (
    excluded_display_labels,
    resolve_catalog_spec,
)


def _tags(non_identifying=()):
    return {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
        "factor_names": ["front", "back"],
        "non_identifying_classes": list(non_identifying),
    }


def test_no_marks_is_a_no_op():
    assert resolve_catalog_spec([_tags()], []).labels == (
        "red_blue",
        "red_notag",
        "notag_blue",
        "notag_notag",
    )


def test_bare_class_mark_excludes_every_containing_composite():
    spec = resolve_catalog_spec([_tags(["notag"])], [])
    assert spec.labels == ("red_blue",)


def test_axis_scoped_mark_excludes_only_that_axis():
    spec = resolve_catalog_spec([_tags(["front:notag"])], [])
    assert spec.labels == ("red_blue", "red_notag")


def test_whole_composite_mark_excludes_exactly_that_label():
    spec = resolve_catalog_spec([_tags(["notag_notag"])], [])
    assert spec.labels == ("red_blue", "red_notag", "notag_blue")


def test_excluded_display_labels_reports_what_was_dropped():
    assert excluded_display_labels([_tags(["notag_notag"])]) == frozenset(
        {"notag_notag"}
    )
    assert excluded_display_labels([_tags(["notag"])]) == frozenset(
        {"red_notag", "notag_blue", "notag_notag"}
    )


def test_all_excluded_yields_empty_spec_without_raising(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        spec = resolve_catalog_spec([_tags(["red", "notag", "blue"])], [])
    assert spec.entries == ()
    assert any("every identity" in r.getMessage().lower() for r in caplog.records)


def test_mark_that_matches_nothing_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        spec = resolve_catalog_spec([_tags(["notag_typo"])], [])
    # No exclusion happened -- the typo'd mark matched no combination.
    assert spec.labels == ("red_blue", "red_notag", "notag_blue", "notag_notag")
    assert any(
        "matched nothing" in r.getMessage().lower() and "notag_typo" in r.getMessage()
        for r in caplog.records
    )


def test_bare_string_non_identifying_classes_is_treated_as_one_mark():
    # A common config typo: a bare string instead of a one-element list.
    # Without a guard this iterates the string's characters ('n','o',...).
    cfg = _tags(["notag"])
    cfg["non_identifying_classes"] = "notag"
    spec = resolve_catalog_spec([cfg], [])
    assert spec.labels == ("red_blue",)


def test_excluding_a_middle_combination_preserves_survivor_order():
    cfg = {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["a", "b", "c"], ["z"]],
        "factor_names": ["front", "back"],
        "non_identifying_classes": ["b_z"],
    }
    spec = resolve_catalog_spec([cfg], [])
    # "b_z" is the middle entry of the a_z/b_z/c_z product; the survivors on
    # either side must keep their original relative order.
    assert spec.labels == ("a_z", "c_z")


import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.postprocess_df import (
    apply_identity_postprocessing_to_df,
)

_PARAMS = {
    "CNN_CLASSIFIERS": [
        {
            "label": "colortag",
            "unique_identifier": True,
            "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
            "factor_names": ["front", "back"],
            "non_identifying_classes": ["notag_notag"],
        }
    ],
    "IDENTITY_POSTHOC_ENABLED": False,
    "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
}


def _three_untagged_tracks():
    rows = []
    for traj in (0, 1, 2):
        for frame in (0, 1):
            rows.append(
                {
                    "TrajectoryID": traj,
                    "FrameID": frame,
                    "CNN_colortag_front_Class": "notag",
                    "CNN_colortag_front_Conf": 0.9,
                    "CNN_colortag_back_Class": "notag",
                    "CNN_colortag_back_Conf": 0.7,
                }
            )
    return pd.DataFrame(rows)


def test_untagged_tracks_are_labelled_not_unknown():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert set(out[C.FINAL_LABEL]) == {"notag_notag"}
    assert set(out[C.FINAL_SOURCE]) == {"nonidentifying"}


def test_untagged_tracks_keep_the_unknown_slot_id():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert set(out[C.FINAL_ID]) == {0}


def test_untagged_tracks_are_never_merged():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert out["TrajectoryID"].nunique() == 3


def test_confidence_is_the_weakest_axis():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert np.allclose(out[C.FINAL_CONFIDENCE], 0.7)


def test_a_real_identity_is_not_overwritten():
    df = _three_untagged_tracks()
    df.loc[df["TrajectoryID"] == 0, "CNN_colortag_front_Class"] = "red"
    df.loc[df["TrajectoryID"] == 0, "CNN_colortag_back_Class"] = "blue"
    df[C.FINAL_LABEL] = ["red_blue"] * 2 + [np.nan] * 4
    df[C.FINAL_SOURCE] = ["offline"] * 2 + [""] * 4
    out = apply_identity_postprocessing_to_df(df, _PARAMS)
    # NOTE: TrajectoryIDs are renumbered downstream by
    # sort_trajectories_by_identity (alphabetical by consensus label, a
    # preexisting and documented behavior out of this task's scope), so the
    # already-resolved track is identified by its untouched FINAL_SOURCE
    # rather than by its original TrajectoryID value.
    real = out[out[C.FINAL_SOURCE] == "offline"]
    assert set(real[C.FINAL_LABEL]) == {"red_blue"}
    assert len(real) == 2


def test_feature_off_stamp_is_a_no_op():
    """No declared non_identifying_classes -> _stamp_non_identifying_labels
    must not touch the dataframe at all (identity via `is`, not just equal
    values) -- the byte-identical equivalence gate for fixtures with no
    marks declared depends on this."""
    from hydra_suite.core.individual.postprocess_df import _stamp_non_identifying_labels

    params = {
        "CNN_CLASSIFIERS": [
            {
                "label": "colortag",
                "unique_identifier": True,
                "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
                "factor_names": ["front", "back"],
                # no non_identifying_classes declared
            }
        ],
    }
    df = _three_untagged_tracks()
    out = _stamp_non_identifying_labels(df, params)
    assert out is df


def test_no_matching_track_leaves_no_final_family():
    """Declared marks with no track that ever matches them must be a
    complete no-op: no ``IdentityFinal*`` family invented from nothing."""
    from hydra_suite.core.individual.postprocess_df import _stamp_non_identifying_labels

    df = _three_untagged_tracks()
    df["CNN_colortag_front_Class"] = "red"
    df["CNN_colortag_back_Class"] = "blue"
    out = _stamp_non_identifying_labels(df, _PARAMS)
    assert out is df
    assert not any(str(c).startswith("IdentityFinal") for c in out.columns)


def test_capitalized_factor_names_still_resolve_axis_columns():
    """The axis-column resolver must delegate to the same sanitization the
    writer uses (``build_cnn_output_columns``), not re-derive it -- a
    capitalized/punctuated ``factor_names`` entry must still match the
    columns the writer actually produced."""
    params = {
        "CNN_CLASSIFIERS": [
            {
                "label": "colortag",
                "unique_identifier": True,
                "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
                "factor_names": ["Front Tag", "Back-Tag"],
                "non_identifying_classes": ["notag_notag"],
            }
        ],
        "IDENTITY_POSTHOC_ENABLED": False,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
    }
    rows = []
    for frame in (0, 1):
        rows.append(
            {
                "TrajectoryID": 0,
                "FrameID": frame,
                "CNN_colortag_front_tag_Class": "notag",
                "CNN_colortag_front_tag_Conf": 0.9,
                "CNN_colortag_back_tag_Class": "notag",
                "CNN_colortag_back_tag_Conf": 0.7,
            }
        )
    df = pd.DataFrame(rows)
    out = apply_identity_postprocessing_to_df(df, params)
    assert set(out[C.FINAL_LABEL]) == {"notag_notag"}
    assert set(out[C.FINAL_SOURCE]) == {"nonidentifying"}


def test_axis_scoped_mark_stamps_only_the_marked_axis_value():
    """A factor-scoped mark ('back:notag') excludes only composites
    carrying that axis value, not every composite touching that model --
    proving the stamp doesn't collapse to whole-model exclusion."""
    params = {
        "CNN_CLASSIFIERS": [
            {
                "label": "colortag",
                "unique_identifier": True,
                "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
                "factor_names": ["front", "back"],
                "non_identifying_classes": ["back:notag"],
            }
        ],
        "IDENTITY_POSTHOC_ENABLED": False,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
    }
    rows = []
    for frame in (0, 1):
        rows.append(
            {
                "TrajectoryID": 0,
                "FrameID": frame,
                "CNN_colortag_front_Class": "red",
                "CNN_colortag_front_Conf": 0.9,
                "CNN_colortag_back_Class": "notag",
                "CNN_colortag_back_Conf": 0.6,
            }
        )
    df = pd.DataFrame(rows)
    out = apply_identity_postprocessing_to_df(df, params)
    assert set(out[C.FINAL_LABEL]) == {"red_notag"}
    assert set(out[C.FINAL_SOURCE]) == {"nonidentifying"}
    assert set(out[C.FINAL_ID]) == {0}


from hydra_suite.core.post.identity_postprocess import (
    derive_unique_identity_key_series,
    identity_sources_conflict,
    parse_identity_key,
)


def test_notag_is_not_evidence_of_agreement():
    df = pd.DataFrame(
        {
            "CNN_colortag_front_Class": ["notag", "notag"],
            "CNN_colortag_front_Conf": [0.9, 0.9],
            "CNN_colortag_back_Class": ["notag", "notag"],
            "CNN_colortag_back_Conf": [0.9, 0.9],
        }
    )
    keys = derive_unique_identity_key_series(
        df, identity_heads=("colortag",), non_identifying_values=("notag",)
    )
    # No evidence at all -> NaN, so two untagged fragments neither agree nor
    # conflict; the spatial gates alone decide whether they relink.
    assert keys.isna().all()


def test_real_class_survives_alongside_a_notag_axis():
    df = pd.DataFrame(
        {
            "CNN_colortag_front_Class": ["red"],
            "CNN_colortag_front_Conf": [0.9],
            "CNN_colortag_back_Class": ["notag"],
            "CNN_colortag_back_Conf": [0.9],
        }
    )
    keys = derive_unique_identity_key_series(
        df, identity_heads=("colortag",), non_identifying_values=("notag",)
    )
    assert parse_identity_key(keys.iloc[0]) == {"cnn:colortag:front": "red"}


def test_two_untagged_fragments_do_not_conflict():
    lhs = {"cnn:colortag:front": "notag"}
    rhs = {"cnn:colortag:front": "notag"}
    # Sanity: with the values present they'd count as agreement; the fix is
    # that they never reach the comparison at all.
    assert not identity_sources_conflict(lhs, rhs)
