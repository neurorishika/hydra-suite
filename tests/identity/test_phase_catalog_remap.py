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
