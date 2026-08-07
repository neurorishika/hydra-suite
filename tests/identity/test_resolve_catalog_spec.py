import itertools

from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec


def _legacy_labels(cnn_classifiers, tag_labels):
    """Verbatim port of worker.py:1844-1905 — the oracle we must match."""
    known: list[str] = []
    for cfg in cnn_classifiers:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        cnpf = cfg.get("class_names_per_factor") or []
        non_empty = [fl for fl in cnpf if fl]
        if len(non_empty) > 1:
            for combo in itertools.product(*non_empty):
                comp = "_".join(str(c) for c in combo if c)
                if comp and comp not in known:
                    known.append(comp)
        else:
            flat: list[str] = []
            for fl in non_empty:
                flat.extend([str(x) for x in fl if x])
            if not flat:
                flat = [str(x) for x in (cfg.get("labels", []) or []) if x]
            for lbl in flat:
                if lbl and lbl not in known:
                    known.append(lbl)
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
    # flat labels fallback
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
    assert first.factors == (("factor0", "red"), ("factor1", "big"))
    assert first.source == "cnn"


def test_tag_entry_has_empty_factors():
    spec = resolve_catalog_spec([], ["ant1"])
    assert spec.entries[0].factors == ()
    assert spec.entries[0].source == "tag"
