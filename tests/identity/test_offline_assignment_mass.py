"""Task 4: mass-first seeding + exact-objective multi-blocker displacement.

2026-08-27 identity-final-consistency: replaces the dead component-Hungarian
base assignment (`_base_assignment_via_substrate`) with mass-first seeding
(descending duration x top support) and a genuine multi-blocker displacement
move gated by an exact-objective margin.
"""

import math

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity.offline import _iterative_assign

LABELS = ["a", "b", "c"]


def _frag(tid, s, e, x0, y0, x1, y1, probs, stability=1.0):
    log = {l: math.log(max(probs.get(l, 1e-6), 1e-6)) for l in LABELS}
    return {
        "TrajectoryID": tid,
        "StartFrame": s,
        "EndFrame": e,
        "StartX": x0,
        "StartY": y0,
        "EndX": x1,
        "EndY": y1,
        "MeanCNNProbs": probs,
        "MeanTagProbs": {},
        "CNNLogEvidence": log,
        "TagLogEvidence": {},
        "Stability": stability,
        "OnlineLabel": "unknown",
        "OnlineConfidence": 0.0,
    }


PARAMS = {
    "FRAGMENT_CNN_WEIGHT": 0.4,
    "FRAGMENT_TAG_WEIGHT": 0.0,
    "ONLINE_PRIOR_WEIGHT": 0.0,
    "FRAGMENT_LENGTH_WEIGHT": 0.6,
    "MAX_VELOCITY_BREAK": 50.0,
    "MAX_BRIDGE_GAP_FRAMES": 30,
    "SPATIAL_NO_NEIGHBOR_SCORE": 0.3,
    "FRAGMENT_SPATIAL_VETO_THRESHOLD": 0.05,
    "ASSIGNMENT_MARGIN_THRESHOLD": 0.0,
    "FRAGMENT_TOP_K": 3,
    "FRAGMENT_MAX_PASSES": 10,
    "FRAGMENT_MIN_SUPPORT": 0.5,
    "FRAGMENT_MAX_BLOCKERS": 4,
}


def test_long_consistent_track_beats_short_fragments_for_its_label():
    long = _frag(0, 0, 700, 0, 0, 0, 700, {"a": 0.999, "b": 0.0005, "c": 0.0005})
    shorts = [
        _frag(
            i,
            50 * i + 10,
            50 * i + 20,
            500,
            50 * i,
            500,
            50 * i + 10,
            {"a": 0.6, "b": 0.3, "c": 0.1},
        )
        for i in range(1, 5)
    ]
    frags = pd.DataFrame([long, *shorts])
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[0] == "a"
    assert all(out[i] != "a" for i in range(1, 5))


def test_fragment_below_support_floor_stays_unknown():
    frags = pd.DataFrame(
        [_frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.3, "b": 0.3, "c": 0.4})]
    )
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[0] is None


def test_two_disjoint_fragments_may_share_a_label():
    f1 = _frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.99, "b": 0.005, "c": 0.005})
    f2 = _frag(1, 110, 200, 0, 105, 0, 200, {"a": 0.99, "b": 0.005, "c": 0.005})
    out = _iterative_assign(pd.DataFrame([f1, f2]), LABELS, PARAMS)
    assert out == {0: "a", 1: "a"}


def test_overlapping_fragments_never_share_a_label():
    f1 = _frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.99, "b": 0.005, "c": 0.005})
    f2 = _frag(1, 50, 150, 300, 0, 300, 150, {"a": 0.99, "b": 0.005, "c": 0.005})
    out = _iterative_assign(pd.DataFrame([f1, f2]), LABELS, PARAMS)
    assert not (out[0] == "a" and out[1] == "a")
    assert out[0] == "a"  # the longer/heavier one wins the tie on mass


def test_displacement_moves_multiple_blockers_when_it_raises_objective():
    long = _frag(0, 0, 700, 0, 0, 0, 700, {"a": 0.999, "b": 0.0005, "c": 0.0005})
    b1 = _frag(1, 100, 110, 400, 100, 400, 110, {"a": 0.55, "b": 0.45, "c": 0.0})
    b2 = _frag(2, 300, 310, 400, 300, 400, 310, {"a": 0.55, "c": 0.45, "b": 0.0})
    frags = pd.DataFrame([b1, b2, long])  # blockers first in index order on purpose
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[2] == "a"
    assert out[0] != "a" and out[1] != "a"


def test_terminates_with_zero_margin_threshold():
    rng = np.random.default_rng(0)
    frags = []
    for i in range(40):
        s = int(rng.integers(0, 600))
        e = s + int(rng.integers(5, 120))
        p = rng.dirichlet([1, 1, 1])
        probs = dict(zip(LABELS, p))
        frags.append(
            _frag(
                i,
                s,
                e,
                float(rng.uniform(0, 500)),
                float(rng.uniform(0, 500)),
                float(rng.uniform(0, 500)),
                float(rng.uniform(0, 500)),
                probs,
            )
        )
    out = _iterative_assign(
        pd.DataFrame(frags), LABELS, {**PARAMS, "ASSIGNMENT_MARGIN_THRESHOLD": 0.0}
    )
    assert len(out) == 40


# --- extra edge cases (own judgment, per brief instructions) ---


def test_empty_catalog_returns_all_none():
    frags = pd.DataFrame([_frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.99})])
    out = _iterative_assign(frags, [], PARAMS)
    assert out == {0: None}


def test_empty_fragments_returns_empty_dict():
    frags = pd.DataFrame(
        columns=[
            "TrajectoryID",
            "StartFrame",
            "EndFrame",
            "StartX",
            "StartY",
            "EndX",
            "EndY",
            "MeanCNNProbs",
            "MeanTagProbs",
            "CNNLogEvidence",
            "TagLogEvidence",
            "Stability",
            "OnlineLabel",
            "OnlineConfidence",
        ]
    )
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out == {}


def test_single_fragment_gets_its_top_label():
    frags = pd.DataFrame(
        [_frag(0, 0, 50, 0, 0, 0, 50, {"a": 0.9, "b": 0.05, "c": 0.05})]
    )
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[0] == "a"


def test_all_fragments_unknown_when_all_below_floor():
    frags = pd.DataFrame(
        [
            _frag(0, 0, 50, 0, 0, 0, 50, {"a": 0.4, "b": 0.35, "c": 0.25}),
            _frag(1, 60, 110, 0, 0, 0, 50, {"a": 0.34, "b": 0.33, "c": 0.33}),
        ]
    )
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out == {0: None, 1: None}
