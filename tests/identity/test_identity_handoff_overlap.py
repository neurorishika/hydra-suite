"""Brief hand-off overlaps between same-label fragments are not conflicts.

Regression: a track that lost its animal coasts on predictions for a few frames
while a new track already holds the animal. The fragment solver treated any
frame overlap between two same-label fragments as two animals, so a 357-705
frame track with a 92-100% consistent label lost it to its successor and was
left unknown.
"""

from __future__ import annotations

import math

import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import (
    _iterative_assign,
    _unlabel_shared_label_rows,
    handoff_limits,
)

LABELS = ["a", "b", "c"]
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
    "FPS": 10.0,
}
A = {"a": 0.99, "b": 0.005, "c": 0.005}


def _frag(tid, s, e, probs=A):
    log = {l: math.log(max(probs.get(l, 1e-6), 1e-6)) for l in LABELS}
    return {
        "TrajectoryID": tid,
        "StartFrame": s,
        "EndFrame": e,
        "StartX": 0.0,
        "StartY": 0.0,
        "EndX": 0.0,
        "EndY": 0.0,
        "MeanCNNProbs": probs,
        "MeanTagProbs": {},
        "CNNLogEvidence": log,
        "TagLogEvidence": {},
        "Stability": 1.0,
        "OnlineLabel": "unknown",
        "OnlineConfidence": 0.0,
    }


def _span(a, b):
    return frozenset(range(a, b + 1))


def test_handoff_overlap_lets_both_fragments_keep_the_label():
    frags = pd.DataFrame([_frag(0, 0, 300), _frag(1, 290, 700)])
    # Track 0 coasts on predictions over 296-300; 6 detection-backed shared frames.
    detected = [_span(0, 295), _span(290, 700)]
    out = _iterative_assign(frags, LABELS, PARAMS, detected_frames=detected)
    assert out == {0: "a", 1: "a"}
    # Without detection-backed frames any overlap conflicts (old behaviour).
    old = _iterative_assign(frags, LABELS, PARAMS)
    assert sorted(v for v in old.values() if v) == ["a"]


def test_sustained_simultaneous_claim_still_conflicts():
    frags = pd.DataFrame([_frag(0, 0, 300), _frag(1, 100, 200)])
    detected = [_span(0, 300), _span(100, 200)]
    out = _iterative_assign(frags, LABELS, PARAMS, detected_frames=detected)
    assert sorted(v for v in out.values() if v == "a") == ["a"]


def test_short_fragment_mostly_inside_a_long_one_still_conflicts():
    # 7 shared detection-backed frames are within 1.5 s but cover > 25% of
    # a 15-frame fragment: that is a rival claim, and the long track wins.
    frags = pd.DataFrame([_frag(0, 0, 400), _frag(1, 395, 409)])
    detected = [_span(0, 401), _span(395, 409)]
    out = _iterative_assign(frags, LABELS, PARAMS, detected_frames=detected)
    assert out[0] == "a" and out[1] != "a"


def test_handoff_limits_are_time_based_and_configurable():
    assert handoff_limits({"FPS": 10.0}) == (15, 0.25)
    assert handoff_limits({"FPS": 30.0}) == (45, 0.25)
    assert handoff_limits({}) == (15, 0.25)
    custom = {
        "FPS": 10.0,
        "IDENTITY_HANDOFF_MAX_SECONDS": 0.5,
        "IDENTITY_HANDOFF_MAX_FRACTION": 0.1,
    }
    assert handoff_limits(custom) == (5, 0.1)


class _Catalog:
    unknown_index = 0


def test_shared_frames_keep_the_label_on_one_row_only():
    rows = []
    for f in range(0, 10):
        rows.append(
            dict(
                TrajectoryID=0,
                FrameID=f,
                DetectionID=float(f) if f < 7 else float("nan"),
            )
        )
    for f in range(6, 20):
        rows.append(dict(TrajectoryID=1, FrameID=f, DetectionID=100.0 + f))
    df = pd.DataFrame(rows)
    df[C.FINAL_LABEL] = "a"
    df[C.FINAL_ID] = 1
    df[C.FINAL_CONFIDENCE] = 0.9
    df[C.FINAL_FRAGMENT_SCORE] = [0.5] * 10 + [0.8] * 14
    df[C.FINAL_SOURCE] = "offline"
    out = _unlabel_shared_label_rows(df, _Catalog())
    labelled = out[out[C.FINAL_LABEL] == "a"]
    assert not labelled.duplicated(["FrameID", C.FINAL_LABEL]).any()
    # Frames 7-9: track 0 only coasts there, so the detection row keeps "a".
    assert (
        out[(out.TrajectoryID == 0) & (out.FrameID >= 7)][C.FINAL_LABEL] == "unknown"
    ).all()
    # Frame 6: both detected; the higher fragment score (track 1) keeps it.
    assert (
        out[(out.TrajectoryID == 1) & (out.FrameID == 6)][C.FINAL_LABEL].item() == "a"
    )
    assert out[(out.TrajectoryID == 0) & (out.FrameID < 6)][C.FINAL_LABEL].eq("a").all()
