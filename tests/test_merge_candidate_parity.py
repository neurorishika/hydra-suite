"""The two merge-candidate implementations must agree on identical input.

`_find_merge_candidates` dispatches to a Numba kernel or to
`_find_merge_candidates_python`, and treats them as interchangeable: the Python
one runs for mixed-DetectionID coverage AND as the fallback when the JIT raises.
They were not interchangeable.

The Numba path implements a deliberate relaxation -- >=2 matching DetectionIDs
is strong identity evidence, so only `max(2, min_overlap // 2)` agreeing frames
are required instead of `min_overlap`. The Python path nullified it twice: an
early `len(common_frames) < min_overlap` gate ahead of the DetectionID logic,
and a closing `agreeing_frames >= min_overlap` gate applied after the relaxed
branch had already accepted the pair. The relaxation was dead code there, so the
Python path returned a strict subset of the Numba path's candidates -- fewer
merges, more surviving fragments.

This mattered in practice because which path runs is an accident of whether
Numba's on-disk cache (`@jit(cache=True)`) loads: a poisoned cache entry made
the JIT raise, `except Exception` swapped the algorithm, and tracking output
changed with no verdict anywhere noticing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post import processing as P

AGREEMENT_DISTANCE = 20.0
MIN_OVERLAP = 10


def _traj(frames, x, y, detection_ids=None):
    data = {
        "FrameID": list(frames),
        "X": list(x),
        "Y": list(y),
        "Theta": [0.0] * len(list(frames)),
    }
    if detection_ids is not None:
        data["DetectionID"] = list(detection_ids)
    return pd.DataFrame(data)


def _pairs(candidates):
    return sorted((int(c[0]), int(c[1])) for c in candidates)


def _random_trajectories(n, rng, with_detection_id):
    out = []
    for _ in range(n):
        start = int(rng.integers(0, 40))
        length = int(rng.integers(3, 30))
        frames = list(range(start, start + length))
        data = {
            "FrameID": frames,
            "X": rng.normal(rng.integers(0, 200), 5.0, length),
            "Y": rng.normal(rng.integers(0, 200), 5.0, length),
            "Theta": np.zeros(length),
        }
        if with_detection_id:
            base = int(rng.integers(0, 4)) * 1000
            data["DetectionID"] = [
                float(base + f) if rng.random() < 0.8 else np.nan for f in frames
            ]
        out.append(pd.DataFrame(data))
    return out


def test_short_overlap_with_identity_evidence_is_a_candidate():
    """6 common frames, 6 DetectionID matches, min_overlap 10.

    The relaxed requirement is `max(2, 10 // 2)` = 5, and 6 >= 5, so the pair
    qualifies on identity evidence despite the overlap being under min_overlap.
    The early gate used to drop it before the relaxation was ever consulted.
    """
    ids = [100.0 + i for i in range(6)]
    fwd = [
        _traj(range(0, 6), [10.0] * 6, [10.0] * 6, ids),
        _traj(
            range(50, 70), [900.0] * 20, [900.0] * 20, [500.0 + i for i in range(20)]
        ),
    ]
    bwd = [
        _traj(range(0, 6), [10.0] * 6, [10.0] * 6, ids),
        _traj(
            range(50, 70), [900.0] * 20, [900.0] * 20, [500.0 + i for i in range(20)]
        ),
    ]

    got = _pairs(
        P._find_merge_candidates_python(fwd, bwd, AGREEMENT_DISTANCE, MIN_OVERLAP)
    )
    assert (0, 0) in got, f"identity-confirmed short overlap dropped; got {got}"


def test_relaxed_agreement_survives_the_closing_gate():
    """12 common frames but only 9 agreeing, all identity-confirmed.

    9 >= the relaxed 5, so the pair qualifies -- but 9 < min_overlap 10, so the
    closing `agreeing_frames >= min_overlap` gate used to discard it after the
    relaxed branch had already accepted it.
    """
    frames = list(range(0, 12))
    ids = [200.0 + f for f in frames]
    # First 9 frames coincide; the last 3 are far apart and share no DetectionID.
    fx = [10.0] * 9 + [10.0, 10.0, 10.0]
    bx = [10.0] * 9 + [800.0, 800.0, 800.0]
    b_ids = ids[:9] + [np.nan, np.nan, np.nan]

    fwd = [
        _traj(frames, fx, [10.0] * 12, ids),
        _traj(
            range(50, 70), [900.0] * 20, [900.0] * 20, [500.0 + i for i in range(20)]
        ),
    ]
    bwd = [
        _traj(frames, bx, [10.0] * 12, b_ids),
        _traj(
            range(50, 70), [900.0] * 20, [900.0] * 20, [500.0 + i for i in range(20)]
        ),
    ]

    got = _pairs(
        P._find_merge_candidates_python(fwd, bwd, AGREEMENT_DISTANCE, MIN_OVERLAP)
    )
    assert (0, 0) in got, f"relaxed-agreement pair dropped by closing gate; got {got}"


@pytest.mark.skipif(not P.NUMBA_AVAILABLE, reason="requires numba")
@pytest.mark.parametrize("with_detection_id", [True, False])
def test_both_paths_agree_on_random_inputs(with_detection_id):
    """Differential test: the dispatcher and the fallback must return the same set.

    Without DetectionID columns this always held (the strict gates are
    equivalent there, since `agreeing <= common`); with them it failed on
    roughly 40% of random inputs.
    """
    rng = np.random.default_rng(0)
    divergent = []
    for trial in range(150):
        fwd = _random_trajectories(int(rng.integers(2, 6)), rng, with_detection_id)
        bwd = _random_trajectories(int(rng.integers(2, 6)), rng, with_detection_id)
        dispatched = _pairs(
            P._find_merge_candidates(fwd, bwd, AGREEMENT_DISTANCE, MIN_OVERLAP)
        )
        fallback = _pairs(
            P._find_merge_candidates_python(fwd, bwd, AGREEMENT_DISTANCE, MIN_OVERLAP)
        )
        if dispatched != fallback:
            divergent.append((trial, dispatched, fallback))

    assert not divergent, (
        f"{len(divergent)}/150 trials diverged; first: trial {divergent[0][0]} "
        f"numba={divergent[0][1]} python={divergent[0][2]}"
    )


@pytest.mark.skipif(not P.NUMBA_AVAILABLE, reason="requires numba")
def test_a_jit_failure_is_reported_loudly(monkeypatch, caplog):
    """A cache-load failure must not silently swap the algorithm.

    The bare `except Exception` turned a numba caching bug into a different
    tracking result with only a debug-grade breadcrumb. The fallback is still
    taken -- it now produces the same answer -- but it must be loud enough to
    find in a log.
    """

    def _boom(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named '<dynamic>'")

    monkeypatch.setattr(P, "_compute_all_merge_candidates_numba", _boom)

    ids = [300.0 + i for i in range(12)]
    fwd = [_traj(range(12), [10.0] * 12, [10.0] * 12, ids)] * 2
    bwd = [_traj(range(12), [10.0] * 12, [10.0] * 12, ids)] * 2

    with caplog.at_level("WARNING"):
        got = P._find_merge_candidates(fwd, bwd, AGREEMENT_DISTANCE, MIN_OVERLAP)

    assert got, "fallback produced no candidates"
    messages = " ".join(r.message for r in caplog.records)
    assert "Numba" in messages and "<dynamic>" in messages
