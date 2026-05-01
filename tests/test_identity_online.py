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
            0: [
                IdentityEvidence.from_cnn(
                    10, 100, "cnn_primary", _log_probs(0.01, 0.95, 0.04)
                )
            ],
            1: [
                IdentityEvidence.from_cnn(
                    10, 101, "cnn_primary", _log_probs(0.01, 0.93, 0.06)
                )
            ],
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


def test_online_decoder_respawn_prior_max_gap_applies_in_backward() -> None:
    """In backward processing frame_idx decreases; gap calc must use absolute
    difference so max_gap correctly discards far-back stale priors."""
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_RESPAWN_PRIOR_STRENGTH": 0.9,
            "IDENTITY_RESPAWN_PRIOR_DECAY": 0.99,
            "IDENTITY_RESPAWN_PRIOR_MAX_GAP": 3,
        },
    )

    # Establish a strong belief in mouse1 at frame 100.
    decoder.update_frame(
        100,
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
    # Backward respawn with |100-90|-1 = 9 > max_gap=3 -> prior must be discarded.
    decoder.clear_slot(0, reason="respawn", respawn_frame_idx=90)
    decoder.update_frame(90, [0], {})

    belief = decoder.get_belief(0)
    assert belief is not None
    probs = decoder._posterior_probs(belief)
    # Without carried prior, mouse1 and mouse2 should be near-equal.
    assert (
        abs(probs[catalog.index_of("mouse1")] - probs[catalog.index_of("mouse2")])
        < 0.05
    ), probs


# ---------------------------------------------------------------------------
# Live identity-swap correction (Fix A)
# ---------------------------------------------------------------------------


def _swap_decoder() -> OnlineIdentityDecoder:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    return OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.7,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_SLOT_LOCK_MIN_FRAMES": 1,
            "IDENTITY_SLOT_LOCK_STRENGTH": 0.9,
            "IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN": 0.5,
            "IDENTITY_SWAP_MIN_FRAMES": 3,
            "IDENTITY_SWAP_CONF_MARGIN": 0.2,
            "IDENTITY_SWAP_ENABLED": True,
        },
    )


def _commit_initial_identities(decoder: OnlineIdentityDecoder) -> None:
    """Frame 1: commit slot 0 to mouse1, slot 1 to mouse2."""
    decoder.update_frame(
        1,
        [0, 1],
        {
            0: [IdentityEvidence.from_cnn(1, 100, "cnn", _log_probs(0.02, 0.96, 0.02))],
            1: [IdentityEvidence.from_cnn(1, 101, "cnn", _log_probs(0.02, 0.02, 0.96))],
        },
    )


def test_swap_correction_swaps_committed_labels_after_sustained_mutual_mismatch() -> (
    None
):
    decoder = _swap_decoder()
    _commit_initial_identities(decoder)
    assert decoder.get_belief(0).committed_label == "mouse1"
    assert decoder.get_belief(1).committed_label == "mouse2"

    # Sustained mutual mismatch: it takes ~1 frame for evidence to overcome the
    # initial commitment posterior, then 3 frames at the swap threshold to
    # accumulate.  Use stronger evidence so the threshold is reached cleanly.
    for f in range(2, 7):
        decoder.update_frame(
            f,
            [0, 1],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.001, 0.005, 0.994)
                    )
                ],
                1: [
                    IdentityEvidence.from_cnn(
                        f, 101, "cnn", _log_probs(0.001, 0.994, 0.005)
                    )
                ],
            },
        )

    assert decoder.get_belief(0).committed_label == "mouse2"
    assert decoder.get_belief(1).committed_label == "mouse1"


def test_swap_correction_does_not_fire_on_single_frame_flicker() -> None:
    decoder = _swap_decoder()
    _commit_initial_identities(decoder)

    # Single frame of mutual mismatch — should not trigger swap (need 3 sustained)
    decoder.update_frame(
        2,
        [0, 1],
        {
            0: [IdentityEvidence.from_cnn(2, 100, "cnn", _log_probs(0.02, 0.05, 0.93))],
            1: [IdentityEvidence.from_cnn(2, 101, "cnn", _log_probs(0.02, 0.93, 0.05))],
        },
    )

    assert decoder.get_belief(0).committed_label == "mouse1"
    assert decoder.get_belief(1).committed_label == "mouse2"


def test_swap_correction_does_not_fire_on_one_way_disagreement() -> None:
    decoder = _swap_decoder()
    _commit_initial_identities(decoder)

    # Slot 0's evidence flips toward mouse2, but slot 1 still strongly mouse2.
    # Without mutual mismatch, no swap.
    for f in range(2, 6):
        decoder.update_frame(
            f,
            [0, 1],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.02, 0.05, 0.93)
                    )
                ],
                1: [
                    IdentityEvidence.from_cnn(
                        f, 101, "cnn", _log_probs(0.02, 0.02, 0.96)
                    )
                ],
            },
        )

    # Slot 1's commitment should be unchanged; slot 0 may revise via existing
    # override-margin path, but the swap counter must NOT have fired a swap.
    assert decoder.get_belief(1).committed_label == "mouse2"


def test_swap_correction_counter_resets_on_agreement_frame() -> None:
    decoder = _swap_decoder()
    _commit_initial_identities(decoder)

    # 2 frames of mutual mismatch (counter -> 2; below threshold 3)
    for f in (2, 3):
        decoder.update_frame(
            f,
            [0, 1],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.02, 0.05, 0.93)
                    )
                ],
                1: [
                    IdentityEvidence.from_cnn(
                        f, 101, "cnn", _log_probs(0.02, 0.93, 0.05)
                    )
                ],
            },
        )

    # Frame 4: agreement frame — counter should reset
    decoder.update_frame(
        4,
        [0, 1],
        {
            0: [IdentityEvidence.from_cnn(4, 100, "cnn", _log_probs(0.02, 0.96, 0.02))],
            1: [IdentityEvidence.from_cnn(4, 101, "cnn", _log_probs(0.02, 0.02, 0.96))],
        },
    )

    # Frame 5+6: only 2 more mismatch frames; insufficient to trigger
    for f in (5, 6):
        decoder.update_frame(
            f,
            [0, 1],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.02, 0.05, 0.93)
                    )
                ],
                1: [
                    IdentityEvidence.from_cnn(
                        f, 101, "cnn", _log_probs(0.02, 0.93, 0.05)
                    )
                ],
            },
        )

    # Without reset, 4 cumulative mismatch frames would have triggered (>= 3).
    # With reset, this run only has 2 consecutive — no swap.
    assert decoder.get_belief(0).committed_label == "mouse1"
    assert decoder.get_belief(1).committed_label == "mouse2"


def test_swap_correction_disabled_when_flag_false() -> None:
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.7,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_SLOT_LOCK_MIN_FRAMES": 1,
            "IDENTITY_SWAP_MIN_FRAMES": 3,
            "IDENTITY_SWAP_CONF_MARGIN": 0.2,
            "IDENTITY_SWAP_ENABLED": False,
        },
    )
    _commit_initial_identities(decoder)
    for f in range(2, 8):
        decoder.update_frame(
            f,
            [0, 1],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.02, 0.05, 0.93)
                    )
                ],
                1: [
                    IdentityEvidence.from_cnn(
                        f, 101, "cnn", _log_probs(0.02, 0.93, 0.05)
                    )
                ],
            },
        )

    # When swap correction is disabled, the existing per-slot revision logic may
    # still revise individually, but the deliberate atomic swap path is off.
    # We only assert the flag was honoured: no swap-detection state was kept.
    assert decoder._swap_evidence == {}


# ---------------------------------------------------------------------------
# Minimum track age gate (false-fragment suppression)
# ---------------------------------------------------------------------------


def test_min_track_age_blocks_label_on_young_slot() -> None:
    """A slot below the min age must not display a label even with confident
    classifier evidence — this is the gate that suppresses short false-detection
    fragments from stealing identities frame-by-frame.
    """
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_MIN_TRACK_AGE": 5,
        },
    )

    # Frame 1: only 1 frame old — under gate
    assignments = decoder.update_frame(
        1,
        [0],
        {0: [IdentityEvidence.from_cnn(1, 100, "cnn", _log_probs(0.01, 0.98, 0.01))]},
    )
    assert assignments[0].label is None


def test_min_track_age_releases_label_after_threshold() -> None:
    """Once a slot has persisted ``IDENTITY_MIN_TRACK_AGE`` frames, its
    accumulated belief is published normally."""
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_MIN_TRACK_AGE": 3,
        },
    )

    last_label = "sentinel"
    for f in range(1, 5):
        assignments = decoder.update_frame(
            f,
            [0],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.01, 0.98, 0.01)
                    )
                ]
            },
        )
        last_label = assignments[0].label

    # By frame 4 the slot has frame_count >= 3 and the belief is mouse1.
    assert last_label == "mouse1"


def test_min_track_age_does_not_block_aged_slot_when_young_slot_appears() -> None:
    """Aged slot must keep its label when a young slot with conflicting evidence
    appears alongside it — the young slot must not steal the label via Hungarian
    uniqueness."""
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_MIN_TRACK_AGE": 3,
        },
    )

    # Age slot 0 over several frames so it is past the gate and committed.
    for f in range(1, 5):
        decoder.update_frame(
            f,
            [0],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.01, 0.98, 0.01)
                    )
                ]
            },
        )

    # Frame 5: slot 1 spawns (young) with confident evidence for the SAME label
    # that slot 0 already holds.  The young slot must show None; slot 0 keeps
    # its label.
    assignments = decoder.update_frame(
        5,
        [0, 1],
        {
            0: [IdentityEvidence.from_cnn(5, 100, "cnn", _log_probs(0.01, 0.98, 0.01))],
            1: [IdentityEvidence.from_cnn(5, 200, "cnn", _log_probs(0.01, 0.98, 0.01))],
        },
    )
    labels = {a.slot_index: a.label for a in assignments}
    assert labels[0] == "mouse1"
    assert labels[1] is None


def test_min_track_age_resets_after_clear_slot() -> None:
    """When a slot is cleared (respawn), its frame_count is implicitly reset
    via belief destruction.  The new belief must wait the min age before
    displaying again."""
    catalog = IdentityCatalog.from_labels(["mouse1", "mouse2"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_COMMIT_THRESHOLD": 0.6,
            "IDENTITY_COMMIT_MIN_HITS": 1,
            "IDENTITY_MIN_TRACK_AGE": 3,
            # Disable carried-prior so respawn doesn't auto-commit instantly
            "IDENTITY_RESPAWN_PRIOR_STRENGTH": 0.0,
        },
    )

    # Age + commit slot 0
    for f in range(1, 6):
        decoder.update_frame(
            f,
            [0],
            {
                0: [
                    IdentityEvidence.from_cnn(
                        f, 100, "cnn", _log_probs(0.01, 0.98, 0.01)
                    )
                ]
            },
        )
    pre = decoder.update_frame(
        6,
        [0],
        {0: [IdentityEvidence.from_cnn(6, 100, "cnn", _log_probs(0.01, 0.98, 0.01))]},
    )
    assert pre[0].label == "mouse1"

    decoder.clear_slot(0, reason="respawn", respawn_frame_idx=7)

    # Frame 7: new belief, frame_count == 1 — gated
    fresh = decoder.update_frame(
        7,
        [0],
        {0: [IdentityEvidence.from_cnn(7, 200, "cnn", _log_probs(0.01, 0.98, 0.01))]},
    )
    assert fresh[0].label is None
