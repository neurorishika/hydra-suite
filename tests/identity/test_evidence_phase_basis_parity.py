"""Characterization test: retired emitter+remap path vs. NEW stage+worker-remap
path must produce bit-equal GLOBAL evidence.

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

This test drives the NEW producer on identical synthetic raw per-factor
softmax and asserts bit-equal (``np.array_equal``) global ``log_probs``
against a COMMITTED golden snapshot of the OLD emitter+remap path's output
for:
  1. A CNN+AprilTag catalog (CNN is an auxiliary, non-identity-providing
     phase; AprilTag alone defines the identity domain -- so the CNN's own
     label set is entirely disjoint from the global catalog).
  2. A two-CNN-phase catalog (each phase's own label set maps onto a slice
     of the CROSS-PRODUCT global catalog -- see below).

Identity Phase 7 / Task 4: ``IdentityEvidenceEmitter`` has since been
deleted. The "old" side of the comparison is now a committed golden
(``tests/data/identity_evidence_goldens/phase_basis_parity_*.npz``), frozen
while the emitter still existed via
``tests/data/identity_evidence_goldens/generate_goldens.py`` (historical,
non-runnable generation script).

Slice 2 (2026-08-18, phase->global remap generalization): the global catalog
changed from a UNION across identity CNN phases to a true CROSS-PRODUCT
(``core/individual/identity/resolve.py``, ``identity_axes`` /
``resolve_catalog_spec``). Case 2's committed golden froze the OLD union
catalog (labels ``p1, p2, q1, q2, q3``) and is no longer a valid oracle for
the cross-product catalog (labels ``p1_q1 .. p2_q3``) -- it encodes exactly
the bug this slice fixes (exact-label matching drops every phase label
against a cross-product catalog, flooring all identity evidence). Case 2 is
therefore no longer golden-npz-driven: ``test_two_cnn_phase_catalog_phase_basis_parity``
now pins the cross-product remap's output against ``_independent_composite_remap``,
an oracle written independently of ``phase_remap.py`` directly against
``catalog_spec.entries`` -- not a call into the code under test -- so it still
guards the evidence path rather than checking the implementation against
itself. Case 1 (CNN+AprilTag) is unaffected by the cross-product change (the
CNN there is non-identity-providing, so it contributes no axis and the remap
falls back to the old direct-lookup path) and keeps its original npz golden.

``_remap_source_log_probs_to_catalog`` is a nested closure inside
``TrackingWorker`` (``core/tracking/worker.py``); as of Slice 2 its body
delegates to the module-level, directly-importable
``core.individual.identity.phase_remap.{build_phase_label_map,
remap_phase_log_probs}``. ``_remap_verbatim`` below now calls those same
functions (previously it was a byte-for-byte copy of the closure's inline
body, back when the logic lived only inside the closure) -- it still exists
as this test's single seam onto "however the worker maps phase evidence onto
the global catalog", so a future change to the worker's call pattern (not
just the underlying math) is still forced through this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.phase_remap import (
    build_phase_label_map,
    remap_phase_log_probs,
)
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec
from hydra_suite.core.inference.identity_evidence_config import (
    IdentityEvidenceCNNPhaseConfig,
    IdentityEvidenceRunConfig,
)
from hydra_suite.core.inference.result import (
    CNNDetectionPrediction,
    CNNFactorPrediction,
)
from hydra_suite.core.inference.runner import _build_identity_evidence_stage

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "identity_evidence_goldens"


def _remap_verbatim(
    log_probs: np.ndarray,
    source_labels,
    identity_catalog: IdentityCatalog,
    catalog_spec: IdentityCatalogSpec,
    model_label: str,
) -> np.ndarray:
    """The same call the `TrackingWorker`'s `_remap_source_log_probs_to_catalog`
    nested closure (`core/tracking/worker.py`, ~line 1927) makes as of Slice 2:
    build that phase's label map once, then delegate to
    `phase_remap.remap_phase_log_probs`. Kept as a named seam in this test file
    (rather than calling the two functions inline at each call site) so a
    future change to the worker's call pattern is still forced through here."""
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

    phase_label_map = build_phase_label_map(catalog_spec, identity_catalog, model_label)
    return remap_phase_log_probs(arr, labels, identity_catalog, phase_label_map)


def _independent_composite_remap(
    log_probs: np.ndarray,
    local_labels: Sequence[str],
    model_label: str,
    catalog_spec: IdentityCatalogSpec,
    catalog: IdentityCatalog,
) -> np.ndarray:
    """Oracle for the two-CNN-phase case, written independently of
    `phase_remap.py` (no call into the code under test): directly walk
    `catalog_spec.entries`, pick out each entry's class on `model_label`'s
    own axes, and if it joins (with "_") to a phase-local label, assign that
    entry the phase's full probability mass -- then renormalize once over
    the whole catalog. This is the "assign then renormalize" semantics
    documented on `phase_remap.remap_phase_log_probs`, reimplemented from
    scratch so a bug in the production grouping (wrong index, double-count,
    missing renormalize, wrong floor) would show up as a mismatch here."""
    arr = np.asarray(log_probs, dtype=np.float64)
    local_probs = np.exp(arr - arr.max())
    local_probs /= local_probs.sum()
    local_prob_by_label = dict(zip(local_labels, local_probs))

    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)
    remapped[catalog.unknown_index] += local_prob_by_label.get("unknown", 0.0)

    prefix = f"{model_label}:"
    for entry in catalog_spec.entries:
        own_classes = [cls for axis, cls in entry.factors if axis.startswith(prefix)]
        if not own_classes:
            continue
        phase_label = "_".join(own_classes)
        prob = local_prob_by_label.get(phase_label)
        if prob is None:
            continue
        remapped[catalog.index_of(entry.display_label)] += prob

    remapped /= remapped.sum()
    return np.log(np.clip(remapped, 1e-300, None))


def _new_global_log_probs(
    identity_catalog: IdentityCatalog,
    run_config: IdentityEvidenceRunConfig,
    source_name: str,
    class_labels_per_factor: list[list[str]],
    raw_probs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build one detection's evidence via the NEW inference-time
    IdentityEvidenceStage path, remapped to the global catalog exactly as the
    tracking worker does when consuming the sidecar. Returns
    (remapped_log_probs, phase_local_log_probs, phase_local_labels) -- the
    latter two let callers build an independent oracle over the same
    phase-local evidence the production remap consumed."""
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
    remapped = _remap_verbatim(
        matching[0].log_probs,
        source_labels,
        identity_catalog,
        run_config.catalog_spec,
        source_name,
    )
    return (
        remapped,
        np.asarray(matching[0].log_probs, dtype=np.float64),
        tuple(str(lbl) for lbl in source_labels),
    )


def test_cnn_apriltag_catalog_phase_basis_parity():
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

    golden = np.load(GOLDEN_DIR / "phase_basis_parity_cnn_apriltag.npz")
    assert tuple(golden["catalog_labels"]) == identity_catalog.labels
    old = golden["old_log_probs"]
    new, _phase_lp, _phase_labels = _new_global_log_probs(
        identity_catalog,
        run_config,
        "cnn_color",
        [["white", "black", "brown"]],
        raw_probs,
    )

    assert np.array_equal(
        old, new
    ), f"CNN+AprilTag phase-basis divergence: old={old!r} new={new!r}"


def test_two_cnn_phase_catalog_phase_basis_parity():
    """Two CNN phases whose axes CROSS-PRODUCT into the global catalog
    (Slice 2: ``resolve_catalog_spec`` builds the true cross-product, not a
    union, of identity-providing CNN phases' factor axes -- p1/p2 x
    q1/q2/q3 -- so each phase's own label set now maps onto a SLICE of
    composite global entries, not a same-named subset).

    This case no longer has a valid ``old emitter`` npz golden: that golden
    froze the pre-Slice-2 UNION catalog (``p1, p2, q1, q2, q3``), which is
    exactly the catalog shape the cross-product change retired. Regenerating
    it from the deleted emitter is impossible, and reusing it would silently
    re-assert the union-catalog bug this slice fixes. Instead this pins the
    production remap's output against ``_independent_composite_remap``, an
    oracle implemented from scratch (not a call into ``phase_remap.py``)
    directly off ``catalog_spec.entries`` -- see that function's docstring.
    """
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
    assert identity_catalog.labels == (
        "unknown",
        "p1_q1",
        "p1_q2",
        "p1_q3",
        "p2_q1",
        "p2_q2",
        "p2_q3",
    )

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

    new_p, phase_lp_p, phase_labels_p = _new_global_log_probs(
        identity_catalog, run_config, "cnn_p", [["p1", "p2"]], raw_probs_p
    )
    expected_p = _independent_composite_remap(
        phase_lp_p, phase_labels_p, "cnn_p", catalog_spec, identity_catalog
    )
    assert np.array_equal(
        expected_p, new_p
    ), f"phase cnn_p diverged from oracle: expected={expected_p!r} new={new_p!r}"
    # Non-degeneracy: evidence for p1 must actually favor the p1_* slice, not
    # be flattened to (near-)uniform by a silent all-labels-dropped failure.
    probs_p = np.exp(new_p)
    p1_idxs = [identity_catalog.index_of(lbl) for lbl in ("p1_q1", "p1_q2", "p1_q3")]
    p2_idxs = [identity_catalog.index_of(lbl) for lbl in ("p2_q1", "p2_q2", "p2_q3")]
    assert probs_p[p1_idxs].sum() > probs_p[p2_idxs].sum()

    new_q, phase_lp_q, phase_labels_q = _new_global_log_probs(
        identity_catalog, run_config, "cnn_q", [["q1", "q2", "q3"]], raw_probs_q
    )
    expected_q = _independent_composite_remap(
        phase_lp_q, phase_labels_q, "cnn_q", catalog_spec, identity_catalog
    )
    assert np.array_equal(
        expected_q, new_q
    ), f"phase cnn_q diverged from oracle: expected={expected_q!r} new={new_q!r}"
    probs_q = np.exp(new_q)
    q1_idxs = [identity_catalog.index_of(lbl) for lbl in ("p1_q1", "p2_q1")]
    q3_idxs = [identity_catalog.index_of(lbl) for lbl in ("p1_q3", "p2_q3")]
    assert probs_q[q1_idxs].sum() > probs_q[q3_idxs].sum()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
