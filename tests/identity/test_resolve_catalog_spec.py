import itertools
import json

from hydra_suite.core.individual.identity.resolve import (
    _read_factors_from_model_file,
    resolve_catalog_spec,
)


def _legacy_labels(cnn_classifiers, tag_labels):
    """Cross-product oracle over ``class_names_per_factor`` axes.

    Ported from ``worker.py:1844-1905`` and updated for the cross-product
    catalog resolver (Task 4): the flat ``labels`` fallback field is no
    longer read here -- ``identity_axes`` only derives axes from
    ``class_names_per_factor`` (or the model-file fallback), so a classifier
    exposing only a bare ``labels`` list now contributes no axis at all.
    """
    known: list[str] = []
    for cfg in cnn_classifiers:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        cnpf = cfg.get("class_names_per_factor") or []
        non_empty = [fl for fl in cnpf if fl]
        if non_empty:
            for combo in itertools.product(*non_empty):
                comp = "_".join(str(c) for c in combo if c)
                if comp and comp not in known:
                    known.append(comp)
    cnn_derived = set(known)
    for lbl in tag_labels:
        s = str(lbl).strip() if lbl else ""
        if not s:
            continue
        if cnn_derived and s not in cnn_derived:
            continue
        if s not in known:
            known.append(s)
    return known


CASES = [
    # multi-factor composite
    (
        [
            {
                "unique_identifier": True,
                "class_names_per_factor": [["red", "blue"], ["big", "small"]],
            }
        ],
        [],
    ),
    # single factor
    ([{"unique_identifier": True, "class_names_per_factor": [["a", "b", "c"]]}], []),
    # flat labels field is not a supported axis source: contributes nothing
    ([{"unique_identifier": True, "labels": ["x", "y"]}], []),
    # non-unique classifier ignored
    ([{"unique_identifier": False, "class_names_per_factor": [["p", "q"]]}], []),
    # tag labels, filtered to CNN-derived set
    (
        [{"unique_identifier": True, "class_names_per_factor": [["red", "blue"]]}],
        ["red", "phaseA", "blue"],
    ),
    # tag-only (no CNN): all tags accepted
    ([], ["ant1", "ant2", "", "ant1"]),
]


def test_labels_match_legacy_oracle():
    for cnn, tags in CASES:
        spec = resolve_catalog_spec(cnn, tags)
        assert list(spec.labels) == _legacy_labels(cnn, tags), (cnn, tags)


def test_structured_factors_captured_for_composite():
    spec = resolve_catalog_spec(
        [
            {
                "unique_identifier": True,
                "class_names_per_factor": [["red", "blue"], ["big", "small"]],
            }
        ],
        [],
    )
    first = next(e for e in spec.entries if e.display_label == "red_big")
    assert first.factors == (("cnn:factor0", "red"), ("cnn:factor1", "big"))
    assert first.source == "cnn"


def test_tag_entry_has_empty_factors():
    spec = resolve_catalog_spec([], ["ant1"])
    assert spec.entries[0].factors == ()
    assert spec.entries[0].source == "tag"


# --- Model-file JSON fallback (_read_factors_from_model_file) ---
#
# resolve_catalog_spec only consults the model file when a classifier's
# class_names_per_factor is absent/empty and model_path exists on disk.


def test_model_file_fallback_used_when_class_names_per_factor_absent(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({"class_names_per_factor": [["red", "blue"], ["big", "small"]]})
    )
    spec = resolve_catalog_spec(
        [
            {
                "unique_identifier": True,
                "model_path": str(model_path),
            }
        ],
        [],
    )
    assert list(spec.labels) == ["red_big", "red_small", "blue_big", "blue_small"]
    first = next(e for e in spec.entries if e.display_label == "red_big")
    assert first.factors == (("cnn:factor0", "red"), ("cnn:factor1", "big"))
    assert first.source == "cnn"


def test_model_file_fallback_flat_class_names_tier2(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"class_names": ["x", "y", "z"]}))
    spec = resolve_catalog_spec(
        [{"unique_identifier": True, "model_path": str(model_path)}],
        [],
    )
    assert list(spec.labels) == ["x", "y", "z"]
    for entry in spec.entries:
        assert entry.factors == (("cnn:factor0", entry.display_label),)
        assert entry.source == "cnn"


def test_model_file_fallback_factor_models_tier3(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "factor_models": [
                    {"class_names": ["red", "blue"]},
                    {"class_names": ["big", "small"]},
                ]
            }
        )
    )
    spec = resolve_catalog_spec(
        [{"unique_identifier": True, "model_path": str(model_path)}],
        [],
    )
    assert list(spec.labels) == ["red_big", "red_small", "blue_big", "blue_small"]


def test_model_file_fallback_priority_class_names_per_factor_wins(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "class_names_per_factor": [["red", "blue"]],
                "class_names": ["x", "y", "z"],
            }
        )
    )
    factors = _read_factors_from_model_file(str(model_path))
    assert factors == [["red", "blue"]]

    spec = resolve_catalog_spec(
        [{"unique_identifier": True, "model_path": str(model_path)}],
        [],
    )
    assert list(spec.labels) == ["red", "blue"]


def test_model_file_fallback_missing_file_swallowed(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    assert _read_factors_from_model_file(str(missing_path)) == []

    spec = resolve_catalog_spec(
        [{"unique_identifier": True, "model_path": str(missing_path)}],
        [],
    )
    assert spec.entries == ()


def test_model_file_fallback_corrupt_json_swallowed(tmp_path):
    model_path = tmp_path / "corrupt.json"
    model_path.write_text("{not valid json,,,")
    assert _read_factors_from_model_file(str(model_path)) == []

    spec = resolve_catalog_spec(
        [{"unique_identifier": True, "model_path": str(model_path)}],
        [],
    )
    assert spec.entries == ()
