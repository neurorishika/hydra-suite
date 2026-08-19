import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.phase_remap import (
    build_phase_label_map,
    remap_phase_log_probs,
)
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec

THORAX = {
    "label": "thorax",
    "unique_identifier": True,
    "class_names_per_factor": [["red", "blue"]],
    "factor_names": ["dot"],
}
ABDOMEN = {
    "label": "abdomen",
    "unique_identifier": True,
    "class_names_per_factor": [["square", "circle"]],
    "factor_names": ["shape"],
}


def _legacy_remap(log_probs, source_labels, catalog):
    """The pre-change exact-match implementation, kept as the equality oracle."""
    arr = np.asarray(log_probs, dtype=np.float64)
    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)
    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)
    for src_idx, label in enumerate(source_labels):
        if not catalog.contains(label):
            continue
        remapped[catalog.index_of(label)] += float(probs[src_idx])
    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))


def test_single_model_remap_matches_legacy_exactly():
    spec = resolve_catalog_spec([THORAX], [])
    catalog = IdentityCatalog.from_spec(spec)
    phase_labels = ("unknown", "red", "blue")
    log_probs = np.log(np.array([0.1, 0.7, 0.2]))
    pmap = build_phase_label_map(spec, catalog, "thorax")
    got = remap_phase_log_probs(log_probs, phase_labels, catalog, pmap)
    np.testing.assert_array_equal(got, _legacy_remap(log_probs, phase_labels, catalog))


def test_two_model_remap_is_not_degenerate():
    # The failure mode this test exists for: exact-match remapping drops every
    # label and leaves a flat/unknown posterior, so identity silently dies.
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    pmap = build_phase_label_map(spec, catalog, "thorax")
    log_probs = np.log(np.array([0.05, 0.9, 0.05]))
    got = remap_phase_log_probs(log_probs, ("unknown", "red", "blue"), catalog, pmap)
    probs = np.exp(got)
    red_idxs = [catalog.index_of("red_square"), catalog.index_of("red_circle")]
    blue_idxs = [catalog.index_of("blue_square"), catalog.index_of("blue_circle")]
    assert probs[red_idxs].sum() > probs[blue_idxs].sum()
    assert probs[red_idxs].sum() > probs[catalog.unknown_index]


def test_two_models_fuse_to_the_correct_composite():
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    thorax_lp = remap_phase_log_probs(
        np.log(np.array([0.05, 0.9, 0.05])),
        ("unknown", "red", "blue"),
        catalog,
        build_phase_label_map(spec, catalog, "thorax"),
    )
    abdomen_lp = remap_phase_log_probs(
        np.log(np.array([0.05, 0.05, 0.9])),
        ("unknown", "square", "circle"),
        catalog,
        build_phase_label_map(spec, catalog, "abdomen"),
    )
    fused = thorax_lp + abdomen_lp
    assert catalog.label_of(int(np.argmax(fused))) == "red_circle"


def test_phase_label_map_covers_every_phase_label():
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    pmap = build_phase_label_map(spec, catalog, "thorax")
    assert sorted(pmap) == ["blue", "red"]
    assert len(pmap["red"]) == 2


def test_unmapped_phase_label_within_otherwise_working_map_is_dropped():
    """A single phase label with no catalog match (e.g. a classifier
    reporting an out-of-vocabulary class) is silently skipped -- distinct
    from the map-key-mismatch cases below, whose failure mode is the WHOLE
    map being empty. Here the map is healthy; only one label has zero
    targets, and the other labels' evidence must still come through intact."""
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    pmap = build_phase_label_map(spec, catalog, "thorax")
    assert "green" not in pmap

    log_probs = np.log(np.array([0.05, 0.05, 0.85, 0.05]))
    got = remap_phase_log_probs(
        log_probs, ("unknown", "blue", "red", "green"), catalog, pmap
    )
    probs = np.exp(got)
    red_idxs = [catalog.index_of("red_square"), catalog.index_of("red_circle")]
    # "green"'s mass is dropped, not crashed on and not smeared uniformly --
    # red (the actual majority label) still dominates the posterior.
    assert probs[red_idxs].sum() > 0.5


def _cnn_label_map_key(cfg: dict) -> str:
    """Matches worker.py's `_cnn_phase_states` label expression (~line 1572,
    also the `_phase_label_maps` build-loop's `_map_key`, ~line 1936): the
    `source_name` this map is looked up by at the evidence-consumption call
    site."""
    return str(cfg.get("label", "cnn_identity"))


def _cnn_axis_model_label(cfg: dict) -> str:
    """Matches resolve.py's `identity_axes()` model_label normalization
    (also the `_phase_label_maps` build-loop's `_axis_model_label`, ~line
    1938): the axis-prefix this classifier's catalog entries were actually
    built with."""
    return str(cfg.get("label", "") or "").strip() or "cnn"


def test_whitespace_padded_label_is_not_floored():
    """Task 6 fix round 1, Important-1: a label the worker's phase-state
    loop and the map-key loop each normalize DIFFERENTLY (one strips, one
    doesn't) must still route to a non-empty map when the two normalizations
    disagree in their raw form -- guarding the fix, not just the bug."""
    thorax_padded = dict(THORAX, label=" thorax ")
    spec = resolve_catalog_spec([thorax_padded, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)

    map_key = _cnn_label_map_key(thorax_padded)
    axis_label = _cnn_axis_model_label(thorax_padded)
    assert map_key == " thorax "
    assert axis_label == "thorax"
    assert map_key != axis_label  # the whitespace IS the mismatch trap

    phase_label_maps = {
        map_key: build_phase_label_map(spec, catalog, axis_label),
    }
    log_probs = np.log(np.array([0.05, 0.9, 0.05]))
    got = remap_phase_log_probs(
        log_probs, ("unknown", "red", "blue"), catalog, phase_label_maps[map_key]
    )
    probs = np.exp(got)
    red_idxs = [catalog.index_of("red_square"), catalog.index_of("red_circle")]
    assert probs[red_idxs].sum() > 0.5  # not floored to near-zero


def test_missing_label_falls_back_to_cnn_default_and_is_not_floored():
    """Task 6 fix round 1, Important-1: a classifier config with no 'label'
    key at all must still route to a non-empty map -- both normalizations
    fall back to their own default ('cnn_identity' vs 'cnn'), and the fix
    must keep them paired up even in the all-defaults case."""
    thorax_nolabel = {k: v for k, v in THORAX.items() if k != "label"}
    spec = resolve_catalog_spec([thorax_nolabel, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)

    map_key = _cnn_label_map_key(thorax_nolabel)
    axis_label = _cnn_axis_model_label(thorax_nolabel)
    assert map_key == "cnn_identity"
    assert axis_label == "cnn"

    pmap = build_phase_label_map(spec, catalog, axis_label)
    assert pmap  # must not be empty

    log_probs = np.log(np.array([0.05, 0.9, 0.05]))
    got = remap_phase_log_probs(log_probs, ("unknown", "red", "blue"), catalog, pmap)
    probs = np.exp(got)
    red_idxs = [catalog.index_of("red_square"), catalog.index_of("red_circle")]
    assert probs[red_idxs].sum() > 0.5


# --- build-time diagnostics (final-fix wave, Important 6) -----------------

import logging

from hydra_suite.core.individual.identity.phase_remap import build_phase_label_maps


def test_empty_map_warns(caplog):
    """The pre-existing diagnostic: a classifier that reaches zero catalog
    entries at all."""
    spec = resolve_catalog_spec([THORAX], [])
    catalog = IdentityCatalog.from_spec(spec)
    ghost = dict(ABDOMEN, label="ghost")
    with caplog.at_level(logging.WARNING):
        maps = build_phase_label_maps(spec, catalog, [THORAX, ghost])
    assert maps["ghost"] == {}
    assert any(
        "maps to ZERO entries" in r.getMessage() and "ghost" in r.getMessage()
        for r in caplog.records
    )


def test_non_empty_map_with_zero_label_overlap_warns(caplog):
    """The gap an empty-map check cannot see: the map is non-empty, but none
    of its keys is one of this classifier's own phase labels, so every
    lookup misses and the evidence floors just as completely."""
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    # A classifier whose evidence is written against a phase basis that does
    # not equal its axis join (a stale/mismatched class vocabulary).
    stale = dict(THORAX, class_names_per_factor=[["crimson", "azure"]])
    with caplog.at_level(logging.WARNING):
        maps = build_phase_label_maps(spec, catalog, [stale, ABDOMEN])
    assert maps["thorax"]  # non-empty -- the empty-map check would not fire
    msgs = [r.getMessage() for r in caplog.records]
    assert any("share NOTHING" in m and "thorax" in m for m in msgs)
    assert not any("maps to ZERO entries" in m for m in msgs)


def test_labels_colliding_after_strip_warn(caplog):
    """Two identity models whose labels differ only in whitespace collapse
    to one axis prefix; the resulting map's keys are joins across both
    models' axes, so no phase label ever hits."""
    padded = dict(ABDOMEN, label=" thorax ")
    spec = resolve_catalog_spec([THORAX, padded], [])
    catalog = IdentityCatalog.from_spec(spec)
    with caplog.at_level(logging.WARNING):
        build_phase_label_maps(spec, catalog, [THORAX, padded])
    assert any(
        "normalize to the axis prefix" in r.getMessage() and "thorax" in r.getMessage()
        for r in caplog.records
    )


def test_healthy_two_model_config_warns_about_nothing(caplog):
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    with caplog.at_level(logging.WARNING):
        build_phase_label_maps(spec, catalog, [THORAX, ABDOMEN])
    assert [r.getMessage() for r in caplog.records] == []
