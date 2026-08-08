"""Parity tests: EvidenceBuilder must reproduce IdentityEvidenceEmitter output.

Identity Phase 3 / Task 2: the emitter is refactored to delegate its
structured-mapping + calibration + evidence-construction logic to the shared,
Qt-free ``EvidenceBuilder``. These tests prove the two paths agree exactly
(``np.array_equal`` on ``log_probs``) for identical inputs, since the whole
phase's later "byte-identical" tracking-output claim depends on this parity
holding.

Note: parity is asserted against the emitter's ``_build_log_probs_from_posteriors``
/ ``build_frame_evidences`` posteriors path (feeding raw per-factor softmax via
``posteriors=``). The emitter's top-1-prediction fallback
(``_build_log_probs_from_prediction``, used only when a source provides no
posteriors) is a degraded-mode path that is intentionally NOT lifted into the
builder, so it is out of scope here.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.individual.classification.cnn import ClassPrediction
from hydra_suite.core.individual.identity.calibration import CalibrationModel
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence_builder import EvidenceBuilder
from hydra_suite.core.tracking.identity.evidence_emitter import IdentityEvidenceEmitter


def _dummy_predictions(det_ids: list[int], n_factors: int) -> list[ClassPrediction]:
    """Build minimal ClassPrediction stand-ins for the emitter's ``predictions``
    argument. Their content is irrelevant to the posteriors path (only
    ``det_index`` is read, to resolve the stable detection id); the emitter
    only falls back to reading ``class_names``/``confidences`` when
    ``posteriors`` is None, which never happens in these tests.
    """
    preds = []
    for slot, _det_id in enumerate(det_ids):
        preds.append(
            ClassPrediction(
                det_index=slot,
                factor_names=tuple(f"factor_{i}" for i in range(n_factors)),
                class_names=tuple(None for _ in range(n_factors)),
                confidences=tuple(0.0 for _ in range(n_factors)),
            )
        )
    return preds


def _build_pair(
    labels: list[list[str]],
    catalog_known_labels: list[str],
    calibration: CalibrationModel | None,
    tmp_path,
    source_name: str = "cnn0",
):
    catalog = IdentityCatalog.from_labels(catalog_known_labels)
    builder = EvidenceBuilder(
        catalog,
        source_name,
        labels,
        calibration=calibration,
        calibration_signature="calsig" if calibration is not None else "",
        runtime_signature="cpu",
    )
    emitter = IdentityEvidenceEmitter(
        cache_path=str(tmp_path / "e.npz"),
        source_name=source_name,
        class_labels_per_factor=labels,
        runtime_signature="cpu",
        calibration_signature="calsig" if calibration is not None else "",
        calibration=calibration,
    )
    # Emitter's own composite catalog construction must match the caller's
    # IdentityCatalog exactly for this to be a meaningful parity check.
    assert emitter.catalog_labels == catalog.labels
    return builder, emitter


def test_builder_matches_emitter_single_factor(tmp_path):
    labels = [["white", "black", "brown"]]
    probs = [
        [np.array([0.7, 0.2, 0.1])],
        [np.array([0.1, 0.1, 0.8])],
    ]
    det_ids = [10, 11]
    builder, emitter = _build_pair(labels, ["white", "black", "brown"], None, tmp_path)

    ev_b = builder.build_frame_evidences(5, det_ids, probs)
    ev_e = emitter.build_frame_evidences(
        5,
        _dummy_predictions(det_ids, 1),
        posteriors=probs,
        detection_ids=det_ids,
    )

    assert len(ev_b) == len(ev_e) == len(det_ids)
    for b, e in zip(ev_b, ev_e):
        assert np.array_equal(b.log_probs, e.log_probs)
        assert b.source_name == e.source_name == "cnn0"
        assert np.array_equal(b.observed_mask, e.observed_mask)
        assert b.detection_id == e.detection_id


def test_builder_matches_emitter_multifactor_with_underscore(tmp_path):
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
    builder, emitter = _build_pair(labels, catalog_known_labels, None, tmp_path)

    ev_b = builder.build_frame_evidences(7, det_ids, probs)
    ev_e = emitter.build_frame_evidences(
        7,
        _dummy_predictions(det_ids, 2),
        posteriors=probs,
        detection_ids=det_ids,
    )

    assert len(ev_b) == len(ev_e) == len(det_ids)
    for b, e in zip(ev_b, ev_e):
        assert np.array_equal(b.log_probs, e.log_probs)
        assert b.source_name == e.source_name
        assert np.array_equal(b.observed_mask, e.observed_mask)
        assert b.detection_id == e.detection_id


def test_builder_matches_emitter_with_calibration_temperature(tmp_path):
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
    builder, emitter = _build_pair(labels, catalog_known_labels, calibration, tmp_path)

    ev_b = builder.build_frame_evidences(9, det_ids, probs)
    ev_e = emitter.build_frame_evidences(
        9,
        _dummy_predictions(det_ids, 2),
        posteriors=probs,
        detection_ids=det_ids,
    )

    assert len(ev_b) == len(ev_e) == len(det_ids)
    for b, e in zip(ev_b, ev_e):
        assert np.array_equal(b.log_probs, e.log_probs)
        assert b.calibration_signature == e.calibration_signature == "calsig"
        assert np.array_equal(b.observed_mask, e.observed_mask)
        assert b.detection_id == e.detection_id


def test_builder_matches_emitter_with_gapped_empty_factor(tmp_path):
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
    builder, emitter = _build_pair(labels, catalog_known_labels, None, tmp_path)

    ev_b = builder.build_frame_evidences(11, det_ids, probs)
    ev_e = emitter.build_frame_evidences(
        11,
        _dummy_predictions(det_ids, 2),
        posteriors=probs,
        detection_ids=det_ids,
    )

    assert len(ev_b) == len(ev_e) == len(det_ids)
    for b, e in zip(ev_b, ev_e):
        assert np.array_equal(b.log_probs, e.log_probs)
        assert np.array_equal(b.observed_mask, e.observed_mask)
        assert b.detection_id == e.detection_id


def test_builder_matches_emitter_with_colliding_composite_labels(tmp_path):
    """Fix round 1: two distinct factor-value combos that join (via "_") to
    the SAME display string -- factor0=["a", "a_b"] x factor1=["b_c", "c"]
    gives combo ("a", "b_c") -> "a_b_c" AND combo ("a_b", "c") -> "a_b_c".
    The original emitter registers only the FIRST occurrence
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
    builder, emitter = _build_pair(labels, catalog_known_labels, None, tmp_path)

    ev_b = builder.build_frame_evidences(13, det_ids, probs)
    ev_e = emitter.build_frame_evidences(
        13,
        _dummy_predictions(det_ids, 2),
        posteriors=probs,
        detection_ids=det_ids,
    )

    assert len(ev_b) == len(ev_e) == len(det_ids)
    for b, e in zip(ev_b, ev_e):
        assert np.array_equal(b.log_probs, e.log_probs)
        assert np.array_equal(b.observed_mask, e.observed_mask)
        assert b.detection_id == e.detection_id
