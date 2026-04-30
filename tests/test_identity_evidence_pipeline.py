from __future__ import annotations

import numpy as np

from hydra_suite.core.identity.cache import IdentityEvidenceCache
from hydra_suite.core.identity.classification.cnn import ClassPrediction
from hydra_suite.core.tracking.evidence_emitter import IdentityEvidenceEmitter


def _log_probs(*values: float) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    probs /= probs.sum()
    return np.log(np.clip(probs, 1e-300, None))


def test_identity_evidence_emitter_uses_factor_posteriors(tmp_path) -> None:
    cache_path = tmp_path / "evidence.npz"
    emitter = IdentityEvidenceEmitter(
        cache_path=cache_path,
        source_name="cnn_identity",
        class_labels_per_factor=[["mouse1", "mouse2"], ["red", "blue"]],
        runtime_signature="cpu",
    )

    preds = [
        ClassPrediction(
            det_index=420000,
            factor_names=("identity", "coat"),
            class_names=("mouse1", "red"),
            confidences=(0.91, 0.82),
        )
    ]
    posteriors = [[np.array([0.91, 0.09]), np.array([0.82, 0.18])]]

    emitter.emit_frame(42, preds, posteriors=posteriors)
    emitter.flush()

    cache = IdentityEvidenceCache(cache_path, mode="r")
    try:
        assert cache.catalog_labels == ("unknown", "mouse1", "mouse2", "red", "blue")
        frame = cache.load_frame(42)
        assert len(frame) == 1
        evidence = frame[0]
        probs = np.exp(evidence.log_probs)
        probs /= probs.sum()
        assert evidence.detection_id == 420000
        assert probs[1] > probs[2]
        assert probs[3] > probs[4]
        assert evidence.observed_mask is not None
        assert bool(evidence.observed_mask[1]) is True
        assert bool(evidence.observed_mask[3]) is True
    finally:
        cache.close()


def test_identity_evidence_emitter_maps_slot_indices_to_stable_detection_ids(
    tmp_path,
) -> None:
    cache_path = tmp_path / "evidence_ids.npz"
    emitter = IdentityEvidenceEmitter(
        cache_path=cache_path,
        source_name="cnn_identity",
        class_labels_per_factor=[["mouse1", "mouse2"]],
        runtime_signature="cpu",
    )

    preds = [
        ClassPrediction(
            det_index=0,
            factor_names=("identity",),
            class_names=("mouse2",),
            confidences=(0.88,),
        )
    ]

    emitter.emit_frame(42, preds, detection_ids=[420123])
    emitter.flush()

    cache = IdentityEvidenceCache(cache_path, mode="r")
    try:
        frame = cache.load_frame(42)
        assert len(frame) == 1
        assert frame[0].detection_id == 420123
    finally:
        cache.close()
