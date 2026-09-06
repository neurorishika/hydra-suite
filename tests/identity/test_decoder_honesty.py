"""The online decoder must report honest numbers (F5, F3, F4 of the identity audit).

Three defects, each independently reproduced against the real decoder before
the fix (see docs/superpowers/specs/notes/2026-09-06-identity-retention-plan-
adversarial-audit.md):

F5  the emitted confidence was ``p[assigned_label]`` while the emitted label and
    catalog index were the *committed* ones -- a row whose number described a
    different identity than its own label.
F3  the slot lock added ``log(strength) < 0`` to the locked entry, strictly
    *lowering* the identity it was meant to protect, and mutated the stored
    posterior so the penalty compounded every frame.
F4  the commit-override margin was vacuous: incumbent and challenger
    confidences come from the same normalised simplex, so a challenger that
    clears ``commit_threshold`` automatically clears the margin whenever
    ``2 * commit_threshold - 1 >= margin``.
"""

from __future__ import annotations

import logging

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.online import OnlineIdentityDecoder


def _log_probs(*values: float) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    probs /= probs.sum()
    return np.log(np.clip(probs, 1e-300, None))


def _feed(decoder, frame, slot, vec, det=100):
    return decoder.update_frame(
        frame, [slot], {slot: [IdentityEvidence.from_cnn(frame, det, "cnn", vec)]}
    )


# --------------------------------------------------------------------------
# F5 -- the reported confidence must belong to the reported label
# --------------------------------------------------------------------------


def test_committed_row_reports_confidence_of_its_own_label() -> None:
    """A committed slot outvoted by fresh evidence must not borrow the
    challenger's confidence for the incumbent's label."""
    catalog = IdentityCatalog.from_labels(["A", "B"])
    decoder = OnlineIdentityDecoder(
        catalog,
        {
            "IDENTITY_COMMIT_THRESHOLD": 0.85,
            "IDENTITY_COMMIT_MIN_HITS": 3,
            "IDENTITY_DISPLAY_THRESHOLD": 0.4,
            "IDENTITY_SLOT_LOCK_MIN_FRAMES": 10_000,  # keep the lock out of it
            "IDENTITY_SWAP_ENABLED": False,
        },
    )

    # Commit slot 0 to A.
    for f in range(6):
        _feed(decoder, f, 0, _log_probs(0.001, 0.9, 0.099))
    belief = decoder._beliefs[0]
    assert belief.committed and belief.committed_label == "A"

    # Now push B hard enough to win the assignment but not (yet) the
    # commitment. The invariant is checked on EVERY frame -- asserting only
    # after the loop lands on a frame where committed == assigned, where the
    # bug cannot show.
    saw_divergent_frame = False
    for f in range(6, 30):
        out = _feed(decoder, f, 0, _log_probs(0.001, 0.15, 0.849))[0]
        probs = np.exp(belief.log_posterior - belief.log_posterior.max())
        probs /= probs.sum()
        reported_idx = int(out.catalog_index)
        assert out.label == catalog.label_of(reported_idx), (
            f"frame {f}: label {out.label!r} disagrees with catalog_index "
            f"{reported_idx}"
        )
        assert abs(out.confidence - float(probs[reported_idx])) < 1e-12, (
            f"frame {f}: row says {out.label!r} with confidence "
            f"{out.confidence:.6f}, but p({out.label}) = "
            f"{probs[reported_idx]:.6f}"
        )
        # The interesting frames: a committed incumbent outvoted by the raw
        # posterior. That is where the old code paired A's name with B's number.
        if (
            out.label == "A"
            and probs[catalog.index_of("B")] > probs[catalog.index_of("A")]
        ):
            saw_divergent_frame = True

    assert saw_divergent_frame, (
        "scenario never produced a committed-but-outvoted frame -- the test "
        "would pass without ever exercising the defect"
    )


def test_uncommitted_row_confidence_matches_label() -> None:
    """The honest-confidence invariant also holds on the uncommitted path."""
    catalog = IdentityCatalog.from_labels(["A", "B"])
    decoder = OnlineIdentityDecoder(
        catalog, {"IDENTITY_COMMIT_MIN_HITS": 10_000, "IDENTITY_DISPLAY_THRESHOLD": 0.4}
    )
    out = _feed(decoder, 0, 0, _log_probs(0.01, 0.8, 0.19))[0]
    probs = np.exp(
        decoder._beliefs[0].log_posterior - decoder._beliefs[0].log_posterior.max()
    )
    probs /= probs.sum()
    assert abs(out.confidence - float(probs[int(out.catalog_index)])) < 1e-12


# --------------------------------------------------------------------------
# F3 -- the slot lock must protect, not penalise, and must not compound
# --------------------------------------------------------------------------


def test_slot_lock_raises_the_locked_label() -> None:
    catalog = IdentityCatalog.from_labels(["A", "B"])
    decoder = OnlineIdentityDecoder(catalog, {"IDENTITY_SLOT_LOCK_STRENGTH": 0.9})
    from hydra_suite.core.individual.identity.online import TrackIdentityBelief

    belief = TrackIdentityBelief(
        slot_index=0, log_posterior=_log_probs(0.05, 0.9, 0.05)
    )
    belief.slot_lock_label = "A"
    belief.slot_lock_strength = 0.9

    before = decoder._posterior_probs(belief)
    after = decoder._slot_lock_biased_probs(belief)

    a = catalog.index_of("A")
    # Pin the closed form, not just the direction: a no-op bias (log_bias = 0)
    # passes a bare `after > before` on a 1-ULP renormalisation artefact.
    odds = before[a] / (1.0 - before[a]) * (1.0 + 0.9)
    assert abs(after[a] - odds / (1.0 + odds)) < 1e-12, (
        f"slot lock magnitude is not (1+strength)x odds: "
        f"{before[a]:.6f} -> {after[a]:.6f}"
    )
    assert after[a] > before[a] + 1e-3
    # and it must not have touched the stored belief
    assert np.allclose(decoder._posterior_probs(belief), before)


def test_slot_lock_bias_does_not_compound_across_frames() -> None:
    """The lock is a per-frame reporting bias, not a persistent mutation.

    Two slots identical except for the lock, fed the same uninformative
    evidence, must decay at the same rate: the lock must not erode (or inflate)
    the stored posterior frame after frame.
    """
    catalog = IdentityCatalog.from_labels(["A", "B"])
    params = {
        "IDENTITY_COMMIT_THRESHOLD": 0.85,
        "IDENTITY_COMMIT_MIN_HITS": 3,
        "IDENTITY_SLOT_LOCK_MIN_FRAMES": 3,
        "IDENTITY_SWAP_ENABLED": False,
    }
    locked = OnlineIdentityDecoder(catalog, params)
    for f in range(40):
        _feed(locked, f, 0, _log_probs(0.001, 0.9, 0.099))
    assert locked._beliefs[0].slot_lock_label == "A", "lock never engaged"

    unlocked = OnlineIdentityDecoder(
        catalog, {**params, "IDENTITY_SLOT_LOCK_MIN_FRAMES": 10_000}
    )
    for f in range(40):
        _feed(unlocked, f, 0, _log_probs(0.001, 0.9, 0.099))
    assert unlocked._beliefs[0].slot_lock_label is None

    def p_a(dec):
        p = np.exp(dec._beliefs[0].log_posterior - dec._beliefs[0].log_posterior.max())
        return float((p / p.sum())[catalog.index_of("A")])

    flat = _log_probs(1 / 3, 1 / 3, 1 / 3)
    for f in range(40, 70):
        locked.update_frame(f, [0], {0: [IdentityEvidence.from_cnn(f, 1, "cnn", flat)]})
        unlocked.update_frame(
            f, [0], {0: [IdentityEvidence.from_cnn(f, 1, "cnn", flat)]}
        )

    # Equality, not `>=`: the lock must not touch the stored belief in EITHER
    # direction. `>=` alone still accepts a bias that compounds upward.
    assert np.allclose(
        locked._beliefs[0].log_posterior, unlocked._beliefs[0].log_posterior
    ), (
        f"the lock mutated the stored belief: locked p(A)={p_a(locked):.6f} "
        f"vs unlocked p(A)={p_a(unlocked):.6f}"
    )


# --------------------------------------------------------------------------
# F4 -- the override margin must be able to block something
# --------------------------------------------------------------------------


def test_vacuous_override_margin_is_announced(caplog) -> None:
    """The shipped defaults make the commit-revision gate arithmetically dead.

    Both confidences are entries of the same normalised posterior, so a
    challenger clearing `commit_threshold` forces the incumbent below
    `1 - commit_threshold`; the margin can only bind above
    `2 * commit_threshold - 1`. At 0.5 vs 2*0.85-1 = 0.70 it never blocks.
    That is not fixed here -- retuning it needs a retention oracle -- but it
    must not be silent.
    """
    from hydra_suite.core.individual.identity import online as online_mod

    online_mod._VACUOUS_MARGIN_WARNED.clear()
    catalog = IdentityCatalog.from_labels(["A", "B"])
    with caplog.at_level(logging.WARNING):
        OnlineIdentityDecoder(
            catalog,
            {
                "IDENTITY_COMMIT_THRESHOLD": 0.85,
                "IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN": 0.5,
            },
        )
    assert "can never block a revision" in caplog.text
    assert "0.700" in caplog.text


def test_non_vacuous_override_margin_is_silent() -> None:
    catalog = IdentityCatalog.from_labels(["A", "B"])
    import logging as _logging

    logger = _logging.getLogger("hydra_suite.core.individual.identity.online")
    records: list[_logging.LogRecord] = []

    class _Grab(_logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _Grab()
    logger.addHandler(h)
    try:
        OnlineIdentityDecoder(
            catalog,
            {
                "IDENTITY_COMMIT_THRESHOLD": 0.85,
                "IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN": 0.75,
            },
        )
    finally:
        logger.removeHandler(h)
    assert not [r for r in records if "can never block" in r.getMessage()]


def test_emitted_slot_lock_defaults_match_the_decoder_defaults() -> None:
    """Emitting the three slot-lock knobs must be behaviour-neutral: the
    schema defaults have to equal the values online.py used to hardcode."""
    from hydra_suite.trackerkit.config.identity_schema import SlotLockConfig

    catalog = IdentityCatalog.from_labels(["A", "B"])
    decoder = OnlineIdentityDecoder(catalog, {})  # all-hardcoded path
    cfg = SlotLockConfig()
    assert decoder._slot_lock_min_frames == cfg.min_frames
    assert decoder._slot_lock_strength == cfg.strength
    assert decoder._slot_lock_override_margin == cfg.override_margin


def test_slot_lock_changes_an_assignment_when_it_can_bite() -> None:
    """A no-op bias passed every other test in this file, so pin the one
    behaviour the lock exists for: a locked slot keeps a label it would
    otherwise lose to the display gate."""
    from hydra_suite.core.individual.identity.online import TrackIdentityBelief

    catalog = IdentityCatalog.from_labels(["X", "Y"])
    decoder = OnlineIdentityDecoder(catalog, {"IDENTITY_DISPLAY_THRESHOLD": 0.6})

    # Raw p(X) = 0.543 is under the 0.6 display gate; boosted by 1.9x odds it
    # clears it and the locked slot is assigned X.
    locked = TrackIdentityBelief(
        slot_index=0, log_posterior=_log_probs(0.007, 0.543, 0.45)
    )
    locked.slot_lock_label = "X"
    locked.slot_lock_strength = 0.9
    decoder._beliefs[0] = locked

    raw = decoder._posterior_probs(locked)
    biased = decoder._slot_lock_biased_probs(locked)
    x = catalog.index_of("X")
    assert raw[x] < 0.6 <= biased[x], (
        f"scenario does not straddle the display gate: raw={raw[x]:.3f} "
        f"biased={biased[x]:.3f}"
    )
    assert decoder._solve_visible_assignment([0])[0] == "X"

    # The same belief without the lock falls below the gate and goes
    # unassigned. That difference is exactly what the lock buys.
    decoder._beliefs[1] = TrackIdentityBelief(
        slot_index=1, log_posterior=_log_probs(0.007, 0.543, 0.45)
    )
    assert decoder._solve_visible_assignment([1])[1] is None


def test_non_vacuous_override_margin_actually_blocks_a_revision() -> None:
    """`..._is_silent` only proves silence. At commit 0.85 a margin of 0.75
    needs incumbent <= 0.10 and challenger >= 0.85 -- reachable, so pin that
    a non-vacuous margin really can block."""
    catalog = IdentityCatalog.from_labels(["A", "B"])
    params = {
        "IDENTITY_COMMIT_THRESHOLD": 0.85,
        "IDENTITY_COMMIT_MIN_HITS": 3,
        "IDENTITY_SLOT_LOCK_MIN_FRAMES": 10_000,
        "IDENTITY_SWAP_ENABLED": False,
    }
    strict = OnlineIdentityDecoder(
        catalog, {**params, "IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN": 0.95}
    )
    vacuous = OnlineIdentityDecoder(
        catalog, {**params, "IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN": 0.5}
    )
    for dec in (strict, vacuous):
        for f in range(6):
            _feed(dec, f, 0, _log_probs(0.001, 0.9, 0.099))
        assert dec._beliefs[0].committed_label == "A"

    def frames_to_revise(dec):
        for f in range(6, 60):
            _feed(dec, f, 0, _log_probs(0.001, 0.15, 0.849))
            if dec._beliefs[0].committed_label == "B":
                return f
        return None

    vacuous_at = frames_to_revise(vacuous)
    strict_at = frames_to_revise(strict)
    assert vacuous_at is not None
    assert strict_at is None or strict_at > vacuous_at, (
        f"a 0.95 margin blocked nothing: strict revised at {strict_at}, "
        f"vacuous at {vacuous_at}"
    )
