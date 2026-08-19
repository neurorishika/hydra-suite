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
