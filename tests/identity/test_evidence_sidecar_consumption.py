"""Regression test: the tracking worker must CONSUME the on-disk "batch"
identity-evidence sidecar during non-realtime (batch/cached) tracking.

Identity Phase 3 regression (2026-08-08): the equivalence gate caught
``ant_cnn_identity`` producing ZERO identity even though the batch sidecar
(written by ``InferenceRunner._write_identity_evidence_batch`` during
``run_batch_pass``) contains the correct, byte-identical-to-the-old-emitter
evidence. Root cause: for a NON-realtime tracking pass, ``InferenceRunner``
pre-runs the whole video via ``run_batch_pass`` (never calling
``run_realtime``), so ``inference_runner.identity_evidence_cache`` (the
REALTIME in-memory cache, only populated by
``InferenceRunner._write_identity_evidence_realtime`` inside
``run_realtime``) stays permanently empty/``None``. But the worker's
``_cnn_phase_states`` construction (``core/tracking/worker.py``, ~line 1600)
unconditionally attaches ``"evidence_cache": None`` whenever a
``LiveCNNIdentityStore`` exists for the phase -- which happens for BOTH
realtime AND non-realtime passes whenever ``InferenceRunner``-driven
precompute is active (i.e. essentially always on the current pipeline). So
the tracking loop's per-frame evidence-consumption block (worker.py
~3063-3115) always falls back to the empty realtime
``inference_runner.identity_evidence_cache`` instead of ever reading the
correctly-populated on-disk batch sidecar -> zero online-decoder evidence ->
zero ``IdentityAssignedLabel``.

The fix factors the phase-state's cache selection into
``evidence_cache_for_cnn_phase_state`` (``core/tracking/ingest/
frame_result_bridge.py``): non-realtime passes get the batch disk cache;
realtime passes get ``None`` (so the existing fallback to
``inference_runner.identity_evidence_cache`` -- which IS populated live
during ``run_realtime`` -- kicks in, unchanged).

This test drives the REAL consumption sequence the tracking loop runs:
  1. Build a real v2 batch sidecar via ``IdentityEvidenceStage`` +
     ``IdentityEvidenceCache`` (mode="w") -- the same producer as
     ``write_identity_evidence_sidecar``.
  2. Re-open it read-only, exactly as the worker does at pass start
     (``core/tracking/worker.py`` ~1543-1585).
  3. Select the phase-state's evidence cache via
     ``evidence_cache_for_cnn_phase_state`` for a NON-realtime pass.
  4. Run the worker's exact per-frame filter+remap sequence (``source_name
     == label`` filter, ``catalog_labels_for_source``,
     ``_remap_source_log_probs_to_catalog`` -- verbatim copy, matching the
     convention in ``test_evidence_phase_basis_parity.py``) and assert the
     result is NON-EMPTY and matches a direct stage call remapped to global.

Before the fix (i.e. with ``evidence_cache_for_cnn_phase_state`` hardcoded to
always return ``None``, mirroring the pre-fix worker.py literal), this test
fails: zero evidence is produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
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
from hydra_suite.core.tracking.ingest.frame_result_bridge import (
    evidence_cache_for_cnn_phase_state,
)


def _remap_verbatim(
    log_probs: np.ndarray,
    source_labels,
    identity_catalog: IdentityCatalog,
) -> np.ndarray:
    """Legacy-semantics exact-match remap oracle -- no longer a verbatim copy
    of the real `_remap_source_log_probs_to_catalog` closure, which as of
    Task 6 (Slice 2) delegates to `core.individual.identity.phase_remap` to
    handle cross-product catalogs. This copy stays valid here because this
    test only ever drives it with a SINGLE identity classifier, for which
    the new closure is required to be bit-identical to this exact-match
    implementation (see `test_phase_catalog_remap.py::
    test_single_model_remap_matches_legacy_exactly`). It is intentionally
    NOT kept in sync with the real closure's cross-product handling; for
    that, see `test_evidence_phase_basis_parity.py`."""
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


def _worker_consume_frame(
    frame_idx: int,
    det_id: int,
    label: str,
    evidence_cache,
    identity_catalog: IdentityCatalog,
):
    """The tracking worker's exact per-detection evidence-consumption
    sequence for one CNN identity-providing phase (core/tracking/worker.py
    ~3063-3115): filter this frame's cached evidence by source_name == label,
    look up this detection's log_probs, remap using this source's own phase
    catalog labels. Returns the remapped global log_probs, or None if no
    evidence was found (the empty-consumption bug symptom)."""
    if evidence_cache is None:
        return None
    live_evidence_map = {
        int(ev.detection_id): ev
        for ev in evidence_cache.load_frame(frame_idx)
        if ev.source_name == label
    }
    cached_ev = live_evidence_map.get(det_id)
    if cached_ev is None:
        return None
    source_labels = evidence_cache.catalog_labels_for_source(label)
    return _remap_verbatim(cached_ev.log_probs, source_labels, identity_catalog)


def test_batch_sidecar_evidence_consumed_in_nonrealtime_pass(tmp_path):
    """The bug repro + the fix's assertion: a non-realtime tracking pass must
    read the on-disk batch sidecar, not the (permanently-empty-in-batch-mode)
    realtime in-memory cache."""
    cnn_classifiers = [
        {
            "label": "colortag",
            "unique_identifier": True,
            "class_names_per_factor": [["ant1", "ant2", "ant3"]],
        }
    ]
    catalog_spec = resolve_catalog_spec(cnn_classifiers, [])
    identity_catalog = IdentityCatalog.from_spec(catalog_spec)

    run_config = IdentityEvidenceRunConfig(
        catalog_spec=catalog_spec,
        cnn_phases=(
            IdentityEvidenceCNNPhaseConfig(
                label="colortag", class_names_per_factor=[["ant1", "ant2", "ant3"]]
            ),
        ),
        tag_to_label={},
    )
    catalog, stage = _build_identity_evidence_stage(run_config)
    assert catalog.labels == identity_catalog.labels

    # --- Step 1: write a real v2 batch sidecar (mirrors write_identity_evidence_sidecar). ---
    sidecar_path = tmp_path / "detection_identity_evidence_batch_test.npz"
    writer_cache = IdentityEvidenceCache(
        sidecar_path,
        catalog_labels=identity_catalog.labels,
        mode="w",
        catalog_labels_by_source=stage.catalog_labels_by_source,
    )
    raw_probs = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    cnn_reads = {
        "colortag": [
            CNNDetectionPrediction(
                det_index=0,
                factors=[
                    CNNFactorPrediction(
                        factor_name="factor_0",
                        class_names=["ant1", "ant2", "ant3"],
                        raw_probabilities=raw_probs,
                    )
                ],
            )
        ]
    }
    evidences = stage.evidences_for_frame(0, [42], cnn_reads, None)
    assert evidences  # sanity: the stage itself produces evidence
    writer_cache.save_frame(0, evidences)
    writer_cache.flush()

    # --- Step 2: re-open read-only, exactly as the worker does at pass start. ---
    batch_evidence_cache = IdentityEvidenceCache(sidecar_path, mode="r")

    # --- Step 3: phase-state cache selection for a NON-realtime pass. ---
    effective_realtime_tracking_mode = False
    phase_evidence_cache = evidence_cache_for_cnn_phase_state(
        effective_realtime_tracking_mode, batch_evidence_cache
    )
    assert phase_evidence_cache is batch_evidence_cache, (
        "Non-realtime pass must select the on-disk batch sidecar, not the "
        "(empty-in-batch-mode) realtime cache."
    )

    # --- Step 4: run the worker's exact per-frame consumption sequence. ---
    mapped = _worker_consume_frame(
        0, 42, "colortag", phase_evidence_cache, identity_catalog
    )
    assert mapped is not None, (
        "Worker produced ZERO identity evidence from a non-empty batch "
        "sidecar -- this is the reported regression."
    )

    # Cross-check against a direct stage call remapped the same way.
    direct_evidences = stage.evidences_for_frame(0, [42], cnn_reads, None)
    matching = [e for e in direct_evidences if e.source_name == "colortag"]
    assert len(matching) == 1
    expected = _remap_verbatim(
        matching[0].log_probs,
        stage.catalog_labels_by_source["colortag"],
        identity_catalog,
    )
    assert np.array_equal(mapped, expected)


def test_evidence_cache_for_cnn_phase_state_realtime_uses_live_fallback():
    """A TRUE realtime pass must NOT be pinned to a (possibly stale/absent)
    batch cache -- it should get None, so the worker's existing fallback to
    inference_runner.identity_evidence_cache (populated live by
    run_realtime) is what actually feeds the decoder."""
    sentinel_batch_cache = object()
    assert evidence_cache_for_cnn_phase_state(True, sentinel_batch_cache) is None
    assert (
        evidence_cache_for_cnn_phase_state(False, sentinel_batch_cache)
        is sentinel_batch_cache
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
