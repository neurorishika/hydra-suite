from hydra_suite.core.individual.identity.heads import (
    HEADS_UNKNOWN,
    identity_class_columns,
    identity_head_labels,
    resolve_identity_heads,
)


def test_only_unique_identifier_entries_are_identity_heads():
    cfgs = [
        {"label": "colortag", "unique_identifier": True},
        {"label": "behavior", "unique_identifier": False},
        {"label": "caste"},  # missing key == not an identity head
    ]
    assert identity_head_labels(cfgs) == ("colortag",)


def test_identity_class_columns_matches_flat_and_multifactor():
    columns = [
        "CNN_colortag_Class",
        "CNN_colortag_thorax_Class",
        "CNN_colortag_thorax_Conf",
        "CNN_behavior_Class",
        "TrajectoryID",
    ]
    got = identity_class_columns(columns, ("colortag",))
    assert got == ["CNN_colortag_Class", "CNN_colortag_thorax_Class"]


def test_identity_class_columns_handles_underscore_in_head_label():
    # "^CNN_(.+)_Class$" cannot tell "colour_tag" (flat) from "colour"+"tag"
    # (factor). Matching against known head labels can.
    columns = ["CNN_colour_tag_Class", "CNN_colour_tag_left_Class"]
    got = identity_class_columns(columns, ("colour_tag",))
    assert got == ["CNN_colour_tag_Class", "CNN_colour_tag_left_Class"]


def test_no_identity_heads_yields_no_columns():
    cfgs = [{"label": "behavior", "unique_identifier": False}]
    assert identity_head_labels(cfgs) == ()
    assert identity_class_columns(["CNN_behavior_Class"], ()) == []


def test_resolve_identity_heads_distinguishes_absent_from_empty():
    # Absent key -> legacy fallback sentinel; present-but-none -> empty tuple.
    assert resolve_identity_heads({}) is HEADS_UNKNOWN
    assert resolve_identity_heads({"CNN_CLASSIFIERS": []}) == ()
    assert resolve_identity_heads(
        {"CNN_CLASSIFIERS": [{"label": "x", "unique_identifier": True}]}
    ) == ("x",)
