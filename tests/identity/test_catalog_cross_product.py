import logging

from hydra_suite.core.individual.identity.resolve import (
    identity_axes,
    resolve_catalog_spec,
)


def _thorax():
    return {
        "label": "thorax",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"]],
        "factor_names": ["dot"],
    }


def _abdomen():
    return {
        "label": "abdomen",
        "unique_identifier": True,
        "class_names_per_factor": [["square", "circle"]],
        "factor_names": ["shape"],
    }


def _behavior():
    return {
        "label": "behavior",
        "unique_identifier": False,
        "class_names_per_factor": [["walking", "grooming"]],
        "factor_names": ["state"],
    }


def test_axes_span_all_identity_models_in_config_order():
    axes = identity_axes([_thorax(), _abdomen(), _behavior()])
    assert [(a.model_label, a.factor_name, a.classes) for a in axes] == [
        ("thorax", "dot", ("red", "blue")),
        ("abdomen", "shape", ("square", "circle")),
    ]


def test_two_identity_models_cross_product_not_union():
    spec = resolve_catalog_spec([_thorax(), _abdomen()], [])
    assert spec.labels == ("red_square", "red_circle", "blue_square", "blue_circle")


def test_cross_product_entries_carry_qualified_factor_provenance():
    spec = resolve_catalog_spec([_thorax(), _abdomen()], [])
    assert spec.entries[0].factors == (
        ("thorax:dot", "red"),
        ("abdomen:shape", "square"),
    )


def test_non_identity_model_contributes_no_axis():
    spec = resolve_catalog_spec([_thorax(), _behavior()], [])
    assert spec.labels == ("red", "blue")


def test_single_multifactor_model_is_unchanged():
    # The pre-existing within-model product must be preserved exactly.
    cfg = {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"], ["big", "small"]],
        "factor_names": ["hue", "size"],
    }
    spec = resolve_catalog_spec([cfg], [])
    assert spec.labels == ("red_big", "red_small", "blue_big", "blue_small")


def test_missing_factor_names_fall_back_to_positional():
    cfg = {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"]],
    }
    spec = resolve_catalog_spec([cfg], [])
    assert spec.entries[0].factors == (("colortag:factor0", "red"),)


def _big_model(i):
    return {
        "label": f"m{i}",
        "unique_identifier": True,
        "class_names_per_factor": [[f"c{j}" for j in range(8)]],
        "factor_names": [f"f{i}"],
    }


def test_large_catalog_warns_and_names_axes(caplog):
    with caplog.at_level(logging.WARNING):
        spec = resolve_catalog_spec([_big_model(i) for i in range(4)], [])
    assert len(spec.entries) == 8**4
    assert any("m0:f0" in r.getMessage() for r in caplog.records)


def test_small_catalog_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_catalog_spec([_thorax(), _abdomen()], [])
    assert not caplog.records
