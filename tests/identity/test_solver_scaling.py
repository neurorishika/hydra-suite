"""The fragment solver scales to thousands of fragments without changing answers.

Regression: every displacement attempt re-scored the whole schedule, written
for "at most a few dozen" fragments. A crowded full-arena clip produced 8,307
fragments and the solver ran for hours on one core. Moves are now scored over
the fragments whose labels they touch -- exact, because a fragment's score
depends only on its own label's schedule.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.individual.identity.offline import _iterative_assign

PARAMS = {
    "FRAGMENT_CNN_WEIGHT": 0.4,
    "FRAGMENT_TAG_WEIGHT": 0.0,
    "ONLINE_PRIOR_WEIGHT": 0.0,
    "FRAGMENT_LENGTH_WEIGHT": 0.6,
    "MAX_VELOCITY_BREAK": 50.0,
    "MAX_BRIDGE_GAP_FRAMES": 30,
    "SPATIAL_NO_NEIGHBOR_SCORE": 0.3,
    "FRAGMENT_SPATIAL_VETO_THRESHOLD": 0.05,
    "ASSIGNMENT_MARGIN_THRESHOLD": 0.01,
    "FRAGMENT_TOP_K": 3,
    "FRAGMENT_MAX_PASSES": 10,
    "FRAGMENT_MIN_SUPPORT": 0.3,
    "FRAGMENT_MAX_BLOCKERS": 4,
    "FPS": 10.0,
}


def _problem(seed, n, n_labels, span):
    rng = np.random.default_rng(seed)
    labels = [f"l{k}" for k in range(n_labels)]
    rows, detected = [], []
    for t in range(n):
        s = int(rng.integers(0, span))
        e = s + int(rng.integers(5, 400))
        true = labels[int(rng.integers(n_labels))]
        noise = rng.dirichlet(np.ones(n_labels) * 0.5)
        probs = {lbl: 0.3 * float(noise[k]) for k, lbl in enumerate(labels)}
        probs[true] += 0.7
        x0, y0 = rng.uniform(0, 2000, 2)
        x1, y1 = x0 + rng.normal(0, 200), y0 + rng.normal(0, 200)
        rows.append(
            {
                "TrajectoryID": t,
                "StartFrame": s,
                "EndFrame": e,
                "StartX": x0,
                "StartY": y0,
                "EndX": x1,
                "EndY": y1,
                "MeanCNNProbs": probs,
                "MeanTagProbs": {},
                "CNNLogEvidence": {k: math.log(max(v, 1e-6)) for k, v in probs.items()},
                "TagLogEvidence": {},
                "Stability": float(rng.uniform(0.3, 1.0)),
                "OnlineLabel": "unknown",
                "OnlineConfidence": 0.0,
            }
        )
        coast = int(rng.integers(0, 6))
        detected.append(frozenset(range(s, e + 1 - coast)))
    return pd.DataFrame(rows), labels, detected


@pytest.mark.parametrize("seed", range(12))
def test_local_objective_matches_whole_schedule(seed):
    frags, labels, detected = _problem(seed, n=150, n_labels=6, span=3000)
    for det in (None, detected):
        local_counts, whole_counts = {}, {}
        local = _iterative_assign(
            frags, labels, PARAMS, _debug_counts=local_counts, detected_frames=det
        )
        whole = _iterative_assign(
            frags,
            labels,
            PARAMS,
            _debug_counts=whole_counts,
            detected_frames=det,
            _whole_schedule_objective=True,
        )
        assert local == whole
        assert local_counts.get("displacements", 0) == whole_counts.get(
            "displacements", 0
        )


def test_thousands_of_fragments_solve_in_reasonable_time():
    frags, labels, detected = _problem(1, n=3000, n_labels=25, span=40000)
    t0 = time.perf_counter()
    out = _iterative_assign(frags, labels, PARAMS, detected_frames=detected)
    assert len(out) == 3000
    assert time.perf_counter() - t0 < 120.0
