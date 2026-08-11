"""Parity tests: ``EvidenceBuilder`` must reproduce the retired
``IdentityEvidenceEmitter`` output.

Identity Phase 3 / Task 2: the emitter was refactored to delegate its
structured-mapping + calibration + evidence-construction logic to the shared,
Qt-free ``EvidenceBuilder``. These tests prove the two paths agree exactly
(``np.array_equal`` on ``log_probs``) for identical inputs, since the whole
phase's later "byte-identical" tracking-output claim depends on this parity
holding.

Identity Phase 7 / Task 4: ``IdentityEvidenceEmitter`` itself has since been
deleted (evidence is now produced in-process by ``IdentityEvidenceStage``).
These tests now compare ``EvidenceBuilder`` output against a COMMITTED
golden snapshot of the emitter's output
(``tests/data/identity_evidence_goldens/builder_parity_*.npz``), frozen
while the emitter still existed. See
``tests/data/identity_evidence_goldens/generate_goldens.py`` for the
(historical, non-runnable) generation script.

Note: parity is asserted against the emitter's ``_build_log_probs_from_posteriors``
/ ``build_frame_evidences`` posteriors path (feeding raw per-factor softmax via
``posteriors=``). The emitter's top-1-prediction fallback
(``_build_log_probs_from_prediction``, used only when a source provides no
posteriors) is a degraded-mode path that is intentionally NOT lifted into the
builder, so it is out of scope here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hydra_suite.core.individual.identity.calibration import CalibrationModel
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence_builder import EvidenceBuilder

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "identity_evidence_goldens"


def _load_golden(name: str):
    data = np.load(GOLDEN_DIR / f"builder_parity_{name}.npz", allow_pickle=False)
    return data


def _build_builder(
    labels: list[list[str]],
    catalog_known_labels: list[str],
    calibration: CalibrationModel | None,
    source_name: str = "cnn0",
) -> EvidenceBuilder:
    catalog = IdentityCatalog.from_labels(catalog_known_labels)
    return EvidenceBuilder(
        catalog,
        source_name,
        labels,
        calibration=calibration,
        calibration_signature="calsig" if calibration is not None else "",
        runtime_signature="cpu",
    )


def _assert_matches_golden(ev_b, golden) -> None:
    assert len(ev_b) == len(golden["log_probs"]) == len(golden["detection_id"])
    for i, b in enumerate(ev_b):
        assert np.array_equal(b.log_probs, golden["log_probs"][i])
        assert b.source_name == str(golden["source_names"][i])
        assert np.array_equal(b.observed_mask, golden["observed_mask"][i])
        assert b.detection_id == int(golden["detection_id"][i])


def test_builder_matches_emitter_single_factor():
    labels = [["white", "black", "brown"]]
    probs = [
        [np.array([0.7, 0.2, 0.1])],
        [np.array([0.1, 0.1, 0.8])],
    ]
    det_ids = [10, 11]
    builder = _build_builder(labels, ["white", "black", "brown"], None)
    golden = _load_golden("single_factor")

    ev_b = builder.build_frame_evidences(5, det_ids, probs)
    _assert_matches_golden(ev_b, golden)


def test_builder_matches_emitter_multifactor_with_underscore():
    # "dark_red" contains "_" -> a naive split("_") on the composite catalog
    # label would corrupt this; the structured (factor_index, class) mapping
    # must be exercised instead.
    labels = [["dark_red", "blue"], ["big", "small"]]
    catalog_known_labels = [
        "dark_red_big",
        "dark_red_small",
        "blue_big",
        "blue_small",
    ]
    probs = [
        [np.array([0.7, 0.3]), np.array([0.6, 0.4])],
        [np.array([0.2, 0.8]), np.array([0.9, 0.1])],
    ]
    det_ids = [20, 21]
    builder = _build_builder(labels, catalog_known_labels, None)
    golden = _load_golden("multifactor_with_underscore")

    ev_b = builder.build_frame_evidences(7, det_ids, probs)
    _assert_matches_golden(ev_b, golden)


def test_builder_matches_emitter_with_calibration_temperature():
    labels = [["dark_red", "blue"], ["big", "small"]]
    catalog_known_labels = [
        "dark_red_big",
        "dark_red_small",
        "blue_big",
        "blue_small",
    ]
    probs = [
        [np.array([0.55, 0.45]), np.array([0.51, 0.49])],
        [np.array([0.9, 0.1]), np.array([0.3, 0.7])],
    ]
    det_ids = [30, 31]
    calibration = CalibrationModel(temperature=2.5)
    builder = _build_builder(labels, catalog_known_labels, calibration)
    golden = _load_golden("with_calibration_temperature")

    ev_b = builder.build_frame_evidences(9, det_ids, probs)
    assert len(ev_b) == len(golden["log_probs"])
    for i, b in enumerate(ev_b):
        assert np.array_equal(b.log_probs, golden["log_probs"][i])
        assert (
            b.calibration_signature
            == str(golden["calibration_signatures"][i])
            == "calsig"
        )
        assert np.array_equal(b.observed_mask, golden["observed_mask"][i])
        assert b.detection_id == int(golden["detection_id"][i])


def test_builder_matches_emitter_with_gapped_empty_factor():
    """Fix round 1: `class_labels_per_factor` with an empty factor list
    sandwiched between two non-empty ones. The composite index space is
    always compacted (non-empty factors only, gap-skipping), matching the
    original emitter's `_factor_class_to_catalog` construction exactly --
    the middle empty factor participates in neither the cartesian product
    nor the map keys.
    """
    labels = [["a", "b"], [], ["c", "d"]]
    catalog_known_labels = ["a_c", "a_d", "b_c", "b_d"]
    # Aligned to the two NON-EMPTY factors (per the builder's documented
    # contract): posteriors never carry an entry for a classless gap factor.
    probs = [
        [np.array([0.6, 0.4]), np.array([0.3, 0.7])],
        [np.array([0.2, 0.8]), np.array([0.9, 0.1])],
    ]
    det_ids = [40, 41]
    builder = _build_builder(labels, catalog_known_labels, None)
    golden = _load_golden("with_gapped_empty_factor")

    ev_b = builder.build_frame_evidences(11, det_ids, probs)
    _assert_matches_golden(ev_b, golden)


def test_builder_matches_emitter_with_colliding_composite_labels():
    """Fix round 1: two distinct factor-value combos that join (via "_") to
    the SAME display string -- factor0=["a", "a_b"] x factor1=["b_c", "c"]
    gives combo ("a", "b_c") -> "a_b_c" AND combo ("a_b", "c") -> "a_b_c".
    The original emitter registered only the FIRST occurrence
    (product-traversal order) of a novel joined label; later combos that
    collide with an already-seen label are dropped entirely. The catalog
    passed in here reflects that same first-occurrence-wins dedup.
    """
    labels = [["a", "a_b"], ["b_c", "c"]]
    # product order: (a,b_c)->"a_b_c" [kept], (a,c)->"a_c" [kept],
    # (a_b,b_c)->"a_b_b_c" [kept], (a_b,c)->"a_b_c" [collision, DROPPED].
    catalog_known_labels = ["a_b_c", "a_c", "a_b_b_c"]
    probs = [
        [np.array([0.6, 0.4]), np.array([0.7, 0.3])],
        [np.array([0.3, 0.7]), np.array([0.2, 0.8])],
    ]
    det_ids = [50, 51]
    builder = _build_builder(labels, catalog_known_labels, None)
    golden = _load_golden("with_colliding_composite_labels")

    ev_b = builder.build_frame_evidences(13, det_ids, probs)
    _assert_matches_golden(ev_b, golden)
