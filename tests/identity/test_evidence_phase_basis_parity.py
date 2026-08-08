"""Characterization test: OLD emitter path vs. NEW stage+worker-remap path
must produce bit-equal GLOBAL evidence.

Identity Phase 3 final-fix wave, Issue #1 (CRITICAL). The old tracking-time
``IdentityEvidenceEmitter`` built CNN evidence on its OWN per-phase cartesian
catalog (only that phase's factor products), then the tracking worker's
``_remap_source_log_probs_to_catalog`` (``core/tracking/worker.py``) mapped
that phase-basis evidence onto the decoder's GLOBAL catalog, flooring any
phase-unreachable entry to ``1e-300`` before renormalizing.

Task 4's new inference-time path (``_build_identity_evidence_stage`` +
``IdentityEvidenceStage``) originally built every CNN phase's
``EvidenceBuilder`` directly against the GLOBAL catalog, so phase-unreachable
entries got the builder's own internal floor (``1e-6``, see
``EvidenceBuilder._factor_log_prob``) instead of the remap's ``1e-300``, and
normalization happened once over the global catalog instead of twice
(once within the phase, once in the remap). For any config where a CNN
phase's own reachable-label set is a PROPER SUBSET of the global catalog --
CNN+AprilTag configs where the CNN doesn't define the whole identity domain,
or multi-CNN-phase configs where each phase only covers its own labels --
the resulting global ``log_probs`` were NOT bit-equal between the two paths.

This test drives both producers on identical synthetic raw per-factor
softmax and asserts bit-equal (``np.array_equal``) global ``log_probs`` for:
  1. A CNN+AprilTag catalog (CNN is an auxiliary, non-identity-providing
     phase; AprilTag alone defines the identity domain -- so the CNN's own
     label set is entirely disjoint from the global catalog).
  2. A two-CNN-phase catalog (each phase's own label set is a proper subset
     of the union global catalog).

``_remap_source_log_probs_to_catalog`` is a nested closure inside
``TrackingWorker`` (``core/tracking/worker.py``, not a module-level
function), so it cannot be imported directly; ``_remap_verbatim`` below is a
byte-for-byte copy (verified against the live source at test-writing time)
kept in sync by this test's own assertions -- any future edit to the real
closure that changes its semantics should be mirrored here deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.individual.classification.cnn import ClassPrediction
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
from hydra_suite.core.inference.identity_evidence_config import (
    IdentityEvidenceCNNPhaseConfig,
    IdentityEvidenceRunConfig,
)
from hydra_suite.core.inference.result import (
    CNNDetectionPrediction,
    CNNFactorPrediction,
)
from hydra_suite.core.inference.runner import _build_identity_evidence_stage
from hydra_suite.core.tracking.identity.evidence_emitter import IdentityEvidenceEmitter


def _remap_verbatim(
    log_probs: np.ndarray,
    source_labels,
    identity_catalog: IdentityCatalog,
) -> np.ndarray:
    """Verbatim copy of `TrackingWorker`'s `_remap_source_log_probs_to_catalog`
    nested closure (`core/tracking/worker.py`, ~line 1932), parameterized on
    `identity_catalog` (the closure captures `_identity_catalog` from its
    enclosing scope)."""
    if identity_catalog is None:
        return np.asarray(log_probs, dtype=np.float64)
    arr = np.asarray(log_probs, dtype=np.float64)
    if source_labels is None:
        if len(arr) == identity_catalog.size:
            out = arr.copy()
            out -= np.logaddexp.reduce(out)
            return out
        return identity_catalog.known_uniform_log_prior()

    labels = tuple(str(label) for label in source_labels)
    if len(labels) != len(arr):
        return identity_catalog.known_uniform_log_prior()

    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)
    remapped = np.full(identity_catalog.size, 1e-300, dtype=np.float64)
    for src_idx, label in enumerate(labels):
        if not identity_catalog.contains(label):
            continue
        remapped[identity_catalog.index_of(label)] += float(probs[src_idx])
    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))


def _dummy_predictions(det_ids: list[int], n_factors: int) -> list[ClassPrediction]:
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


def _old_global_log_probs(
    identity_catalog: IdentityCatalog,
    source_name: str,
    class_labels_per_factor: list[list[str]],
    raw_probs: list[np.ndarray],
    tmp_path,
) -> np.ndarray:
    """Build one detection's evidence via the OLD emitter path, remapped to
    the global catalog -- the pre-Task-5 tracking-time behavior."""
    emitter = IdentityEvidenceEmitter(
        cache_path=tmp_path / f"{source_name}_old.npz",
        source_name=source_name,
        class_labels_per_factor=class_labels_per_factor,
    )
    evidences = emitter.build_frame_evidences(
        frame_idx=0,
        predictions=_dummy_predictions([0], len(class_labels_per_factor)),
        posteriors=[raw_probs],
    )
    assert len(evidences) == 1
    return _remap_verbatim(
        evidences[0].log_probs, emitter.catalog_labels, identity_catalog
    )


def _new_global_log_probs(
    identity_catalog: IdentityCatalog,
    run_config: IdentityEvidenceRunConfig,
    source_name: str,
    class_labels_per_factor: list[list[str]],
    raw_probs: list[np.ndarray],
) -> np.ndarray:
    """Build one detection's evidence via the NEW inference-time
    IdentityEvidenceStage path, remapped to the global catalog exactly as the
    tracking worker does when consuming the sidecar."""
    catalog, stage = _build_identity_evidence_stage(run_config)
    assert catalog.labels == identity_catalog.labels

    cnn_reads = {
        source_name: [
            CNNDetectionPrediction(
                det_index=0,
                factors=[
                    CNNFactorPrediction(
                        factor_name=f"factor_{i}",
                        class_names=list(factor_labels),
                        raw_probabilities=np.asarray(raw_probs[i], dtype=np.float32),
                    )
                    for i, factor_labels in enumerate(class_labels_per_factor)
                ],
            )
        ]
    }
    evidences = stage.evidences_for_frame(0, [0], cnn_reads, None)
    matching = [e for e in evidences if e.source_name == source_name]
    assert len(matching) == 1

    source_labels = stage.catalog_labels_by_source[source_name]
    return _remap_verbatim(matching[0].log_probs, source_labels, identity_catalog)


def test_cnn_apriltag_catalog_phase_basis_parity(tmp_path):
    """CNN is an auxiliary (non-identity-providing) phase whose own label set
    is entirely disjoint from the global (AprilTag-only) catalog."""
    cnn_classifiers = [
        {
            "label": "cnn_color",
            "unique_identifier": False,
            "class_names_per_factor": [["white", "black", "brown"]],
        }
    ]
    tag_identity_labels = ["antA", "antB"]

    catalog_spec = resolve_catalog_spec(cnn_classifiers, tag_identity_labels)
    assert catalog_spec.entries  # sanity: tags define the domain
    identity_catalog = IdentityCatalog.from_spec(catalog_spec)
    # Sanity: the CNN's own labels are disjoint from the global catalog --
    # this is the exact condition that triggers the divergence.
    assert not ({"white", "black", "brown"} & set(identity_catalog.labels))

    run_config = IdentityEvidenceRunConfig(
        catalog_spec=catalog_spec,
        cnn_phases=(
            IdentityEvidenceCNNPhaseConfig(
                label="cnn_color",
                class_names_per_factor=[["white", "black", "brown"]],
            ),
        ),
        tag_to_label={0: "antA", 1: "antB"},
    )

    raw_probs = [np.array([0.7, 0.2, 0.1], dtype=np.float32)]

    old = _old_global_log_probs(
        identity_catalog,
        "cnn_color",
        [["white", "black", "brown"]],
        raw_probs,
        tmp_path,
    )
    new = _new_global_log_probs(
        identity_catalog,
        run_config,
        "cnn_color",
        [["white", "black", "brown"]],
        raw_probs,
    )

    assert np.array_equal(
        old, new
    ), f"CNN+AprilTag phase-basis divergence: old={old!r} new={new!r}"


def test_two_cnn_phase_catalog_phase_basis_parity(tmp_path):
    """Two CNN phases, each contributing its own disjoint subset of the
    union global catalog."""
    cnn_classifiers = [
        {
            "label": "cnn_p",
            "unique_identifier": True,
            "class_names_per_factor": [["p1", "p2"]],
        },
        {
            "label": "cnn_q",
            "unique_identifier": True,
            "class_names_per_factor": [["q1", "q2", "q3"]],
        },
    ]
    catalog_spec = resolve_catalog_spec(cnn_classifiers, [])
    identity_catalog = IdentityCatalog.from_spec(catalog_spec)
    assert set(identity_catalog.labels) == {"unknown", "p1", "p2", "q1", "q2", "q3"}

    run_config = IdentityEvidenceRunConfig(
        catalog_spec=catalog_spec,
        cnn_phases=(
            IdentityEvidenceCNNPhaseConfig(
                label="cnn_p", class_names_per_factor=[["p1", "p2"]]
            ),
            IdentityEvidenceCNNPhaseConfig(
                label="cnn_q", class_names_per_factor=[["q1", "q2", "q3"]]
            ),
        ),
        tag_to_label={},
    )

    raw_probs_p = [np.array([0.6, 0.4], dtype=np.float32)]
    raw_probs_q = [np.array([0.5, 0.3, 0.2], dtype=np.float32)]

    old_p = _old_global_log_probs(
        identity_catalog, "cnn_p", [["p1", "p2"]], raw_probs_p, tmp_path
    )
    new_p = _new_global_log_probs(
        identity_catalog, run_config, "cnn_p", [["p1", "p2"]], raw_probs_p
    )
    assert np.array_equal(
        old_p, new_p
    ), f"phase cnn_p diverged: old={old_p!r} new={new_p!r}"

    old_q = _old_global_log_probs(
        identity_catalog, "cnn_q", [["q1", "q2", "q3"]], raw_probs_q, tmp_path
    )
    new_q = _new_global_log_probs(
        identity_catalog, run_config, "cnn_q", [["q1", "q2", "q3"]], raw_probs_q
    )
    assert np.array_equal(
        old_q, new_q
    ), f"phase cnn_q diverged: old={old_q!r} new={new_q!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
