"""The offline twin of the phase->global catalog remap (final-fix wave, Critical 1).

``smoothing.load_trajectory_evidence`` -- the fragment solver's evidence
source -- used to hold a second, *exact-label-match* copy of the tracking
worker's remap. With a cross-product catalog (two identity models) every
phase label misses that exact match, all entries floor to ``1e-300``, and
renormalization then puts probability **1.0 on ``unknown``**: not merely
lost evidence but fabricated certainty fed into the solver.

These tests pin both halves of the fix: the single-model path stays
bit-identical to the historical implementation, and the two-model path
survives with a specific, non-uniform, correct distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.offline import run_fragment_solver
from hydra_suite.core.individual.identity.phase_remap import build_phase_label_maps
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
from hydra_suite.core.individual.identity.smoothing import load_trajectory_evidence

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


def _write_cache(
    tmp_path, catalog_labels, evidences_by_frame, catalog_labels_by_source=None
):
    path = tmp_path / "evidence_cache.npz"
    cache = IdentityEvidenceCache(
        path,
        catalog_labels=catalog_labels,
        mode="w",
        catalog_labels_by_source=catalog_labels_by_source,
    )
    for frame_idx, evidences in evidences_by_frame.items():
        cache.save_frame(frame_idx, evidences)
    cache.flush()
    return IdentityEvidenceCache(path, mode="r")


def _legacy_remap(log_probs, source_labels, catalog):
    """The pre-fix exact-label-match implementation, kept as the oracle for
    the single-identity-model path (which must stay bit-identical)."""
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


def test_single_identity_model_offline_remap_matches_legacy_exactly(tmp_path):
    """Feature-off invariant: one identity model, phase basis == global
    catalog, so the offline evidence loader must be bit-identical to the
    exact-match implementation it replaced."""
    spec = resolve_catalog_spec([THORAX], [])
    catalog = IdentityCatalog.from_spec(spec)
    phase_labels = ("unknown", "red", "blue")
    log_probs = np.log(np.array([0.1, 0.7, 0.2]))

    cache = _write_cache(
        tmp_path,
        catalog_labels=catalog.labels,
        evidences_by_frame={0: [IdentityEvidence.from_cnn(0, 5, "thorax", log_probs)]},
        catalog_labels_by_source={"thorax": phase_labels},
    )
    df = pd.DataFrame({"TrajectoryID": [1], "FrameID": [0], "DetectionID": [5]})

    maps = build_phase_label_maps(spec, catalog, [THORAX])
    got = load_trajectory_evidence(df, cache, catalog, maps)[1][0][1]

    expected_evidence = _legacy_remap(log_probs, phase_labels, catalog)
    # load_trajectory_evidence fuses the single source against a flat prior.
    from hydra_suite.core.individual.identity.substrate import fuse_log_evidence

    expected = fuse_log_evidence(catalog.uniform_log_prior(), expected_evidence)
    np.testing.assert_array_equal(got, expected)


def test_two_identity_models_offline_evidence_survives(tmp_path):
    """The bug: with a cross-product catalog the exact-match remap floored
    every entry and renormalized to probability 1.0 on ``unknown``.

    The assertion is a specific, non-uniform distribution -- the two
    ``red_*`` entries must carry the phase's ``red`` mass and dominate both
    ``unknown`` and the ``blue_*`` entries -- so a floored/uniform/
    all-on-unknown posterior all fail.
    """
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    assert catalog.labels == (
        "unknown",
        "red_square",
        "red_circle",
        "blue_square",
        "blue_circle",
    )

    phase_labels = ("unknown", "red", "blue")
    log_probs = np.log(np.array([0.05, 0.90, 0.05]))
    cache = _write_cache(
        tmp_path,
        catalog_labels=catalog.labels,
        evidences_by_frame={0: [IdentityEvidence.from_cnn(0, 5, "thorax", log_probs)]},
        catalog_labels_by_source={"thorax": phase_labels},
    )
    df = pd.DataFrame({"TrajectoryID": [1], "FrameID": [0], "DetectionID": [5]})

    maps = build_phase_label_maps(spec, catalog, [THORAX, ABDOMEN])
    got = load_trajectory_evidence(df, cache, catalog, maps)[1][0][1]
    probs = np.exp(got - np.logaddexp.reduce(got))

    red = probs[[catalog.index_of("red_square"), catalog.index_of("red_circle")]]
    blue = probs[[catalog.index_of("blue_square"), catalog.index_of("blue_circle")]]

    assert probs[catalog.unknown_index] < 0.2, "evidence floored onto unknown"
    assert red.sum() > 0.7
    assert red.sum() > 6 * blue.sum()
    # The phase says nothing about the abdomen axis, so its mass splits
    # evenly between the two red composites -- not concentrated on one.
    np.testing.assert_allclose(red[0], red[1], rtol=1e-9)


def test_fragment_solver_two_models_resolves_a_real_identity(tmp_path):
    """End-to-end through ``run_fragment_solver``: two identity models, one
    trajectory with consistent thorax=red + abdomen=circle evidence must
    resolve to ``red_circle`` -- with the pre-fix remap every phase floored
    and the solver saw certainty on ``unknown`` instead."""
    from hydra_suite.core.individual.identity import columns as C

    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)

    thorax_lp = np.log(np.array([0.02, 0.96, 0.02]))
    abdomen_lp = np.log(np.array([0.02, 0.02, 0.96]))
    evidences_by_frame = {
        f: [
            IdentityEvidence.from_cnn(f, 5, "thorax", thorax_lp),
            IdentityEvidence.from_cnn(f, 5, "abdomen", abdomen_lp),
        ]
        for f in range(4)
    }
    cache = _write_cache(
        tmp_path,
        catalog_labels=catalog.labels,
        evidences_by_frame=evidences_by_frame,
        catalog_labels_by_source={
            "thorax": ("unknown", "red", "blue"),
            "abdomen": ("unknown", "square", "circle"),
        },
    )
    df = pd.DataFrame(
        {
            "TrajectoryID": [1] * 4,
            "FrameID": list(range(4)),
            "DetectionID": [5] * 4,
            "CentroidX": [1.0, 1.0, 1.0, 1.0],
            "CentroidY": [1.0, 1.0, 1.0, 1.0],
        }
    )

    out = run_fragment_solver(
        df,
        catalog,
        {"CNN_CLASSIFIERS": [THORAX, ABDOMEN]},
        cache=cache,
        catalog_spec=spec,
    )
    assert set(out[C.FINAL_LABEL]) == {"red_circle"}
