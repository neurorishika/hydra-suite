from __future__ import annotations

import numpy as np

from hydra_suite.core.identity.catalog import IdentityCatalog
from hydra_suite.core.identity.evidence import IdentityEvidence
from hydra_suite.core.identity.online import OnlineIdentityDecoder


def _log_probs(*values: float) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    probs /= probs.sum()
    return np.log(np.clip(probs, 1e-300, None))


def test_online_decoder_marks_uniqueness_conflict_and_sources() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.99,
            "IDENTITY_COMMIT_MIN_HITS": 99,
        },
    )

    assignments = decoder.update_frame(
        10,
        [0, 1],
        {
            0: [IdentityEvidence.from_cnn(10, 100, "cnn_primary", _log_probs(0.01, 0.95, 0.04))],
            1: [IdentityEvidence.from_cnn(10, 101, "cnn_primary", _log_probs(0.01, 0.93, 0.06))],
        },
    )

    labels = {assignment.slot_index: assignment.label for assignment in assignments}
    assert labels[0] == "mouse1"
    assert labels[1] is None

    belief0 = decoder.get_belief(0)
    belief1 = decoder.get_belief(1)
    assert belief0 is not None
    assert belief1 is not None
    assert belief0.last_evidence_sources == ("cnn_primary",)
    assert belief0.last_conflict_flag is False
    assert belief1.last_evidence_sources == ("cnn_primary",)
    assert belief1.last_conflict_flag is True


def test_online_decoder_revises_committed_identity_after_override() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_SLOT_LOCK_MIN_FRAMES": 1,
            "IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN": 0.2,
        },
    )

    first = decoder.update_frame(
        1,
        [0],
        {
            0: [
                IdentityEvidence.from_cnn(
                    1,
                    100,
                    "cnn_primary",
                    _log_probs(0.01, 0.98, 0.01),
                )
            ]
        },
    )
    assert first[0].label == "mouse1"

    second = decoder.update_frame(
        2,
        [0],
        {
            0: [
                IdentityEvidence.from_cnn(
                    2,
                    101,
                    "cnn_primary",
                    _log_probs(0.01, 0.01, 0.98),
                )
            ]
        },
    )

    assert second[0].label == "mouse2"
    belief = decoder.get_belief(0)
    assert belief is not None
    assert belief.committed_label == "mouse2"
    assert belief.slot_lock_label == "mouse2"


def test_online_decoder_hungarian_respects_display_threshold() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.999,
            "IDENTITY_COMMIT_THRESHOLD": 0.99,
            "IDENTITY_COMMIT_MIN_HITS": 99,
        },
    )

    assignments = decoder.update_frame(
        5,
        [0],
        {
            0: [
                IdentityEvidence.from_cnn(
                    5,
                    500,
                    "cnn_primary",
                    _log_probs(0.01, 0.98, 0.01),
                )
            ]
        },
    )

    assert assignments[0].label is None


def test_online_decoder_new_belief_prior_avoids_committed_labels() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.95,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
        },
    )

    decoder.update_frame(
        1,
        [0],
        {
            0: [
                IdentityEvidence.from_cnn(
                    1,
                    100,
                    "cnn_primary",
                    _log_probs(0.01, 0.98, 0.01),
                )
            ]
        },
    )

    decoder.update_frame(
        2,
        [0, 1],
        {
            0: [
                IdentityEvidence.from_cnn(
                    2,
                    101,
                    "cnn_primary",
                    _log_probs(0.01, 0.98, 0.01),
                )
            ]
        },
    )

    belief = decoder.get_belief(1)
    assert belief is not None
    probs = decoder._posterior_probs(belief)

    assert probs[catalog.index_of("mouse2")] > probs[catalog.index_of("mouse1")]


def test_online_decoder_respawn_carries_recent_prior() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_RESPAWN_PRIOR_STRENGTH": 0.9,
            "IDENTITY_RESPAWN_PRIOR_DECAY": 0.99,
            "IDENTITY_RESPAWN_PRIOR_MAX_GAP": 20,
        },
    )

    decoder.update_frame(
        1,
        [0],
        {
            0: [
                IdentityEvidence.from_cnn(
                    1,
                    100,
                    "cnn_primary",
                    _log_probs(0.01, 0.98, 0.01),
                )
            ]
        },
    )
    decoder.clear_slot(0, reason="respawn", respawn_frame_idx=4)

    decoder.update_frame(4, [0], {})

    belief = decoder.get_belief(0)
    assert belief is not None
    probs = decoder._posterior_probs(belief)
    assert probs[catalog.index_of("mouse1")] > probs[catalog.index_of("mouse2")]