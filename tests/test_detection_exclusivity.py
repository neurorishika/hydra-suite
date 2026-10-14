"""One trajectory per detection per frame after forward/backward merging.

Regression: forward and backward tracking are each one-to-one per frame, but
when the two passes swap animals at different frames of a crossing, the merge
kept both versions, so two final trajectories shared the same detections for a
few frames. The identity solver then treats them as rivals for one label and
drops a whole long trajectory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.post.detection_exclusivity import enforce_detection_exclusivity

N = 60


def _ant(frames, y, det_offset):
    frames = np.asarray(list(frames))
    return pd.DataFrame(
        {
            "FrameID": frames,
            "X": frames * 10.0,
            "Y": np.full(len(frames), float(y)),
            "Theta": 0.0,
            "State": "active",
            "DetectionID": frames * 100.0 + det_offset,
        }
    )


def _shared_rows(trajs):
    rows = pd.concat(
        [t.assign(_t=i) for i, t in enumerate(trajs)], ignore_index=True
    ).dropna(subset=["DetectionID"])
    g = rows.groupby(["FrameID", "DetectionID"])._t.nunique()
    return int(g[g > 1].sum())


def _max_step(traj):
    t = traj.sort_values("FrameID")
    gap = t.FrameID.diff()
    step = np.hypot(t.X.diff(), t.Y.diff())
    return float((step[gap == 1]).max()) if len(t) > 1 else 0.0


def _crossing():
    # Ant A at y=0, ant B at y=120. The forward-derived track follows A then
    # swaps onto B at frame 33; the backward-derived track follows B then swaps
    # onto A at frame 30. Both hold A for frames 30-32.
    a, b = _ant(range(N), 0, 1), _ant(range(N), 120, 2)
    t1 = pd.concat([a[a.FrameID <= 32], b[b.FrameID >= 33]], ignore_index=True)
    t2 = pd.concat([b[b.FrameID <= 29], a[a.FrameID >= 30]], ignore_index=True)
    return t1, t2


def test_crossing_with_swaps_in_both_passes_is_recut_per_animal():
    out = enforce_detection_exclusivity(list(_crossing()), join_step_px=50.0)
    assert _shared_rows(out) == 0
    # Ant A comes back as one continuous trajectory over all frames.
    a_tracks = [t for t in out if (t.Y == 0).all()]
    assert any(len(t) == N for t in a_tracks)
    # No output trajectory still contains a swap.
    assert all(_max_step(t) <= 50.0 for t in out)


def test_short_duplicate_at_the_end_of_a_track_is_absorbed():
    a = _ant(range(N), 0, 1)
    long, tail = a.copy(), a[a.FrameID >= 50].copy()
    out = enforce_detection_exclusivity([long, tail], join_step_px=50.0)
    assert _shared_rows(out) == 0
    assert sum(len(t) for t in out) == N


def test_trajectories_without_shared_detections_pass_through_unchanged():
    a, b = _ant(range(N), 0, 1), _ant(range(N), 120, 2)
    out = enforce_detection_exclusivity([a, b], join_step_px=50.0)
    assert len(out) == 2
    assert out[0] is a and out[1] is b


def test_identical_interpolated_rows_inside_a_shared_run_are_deduplicated():
    t1, t2 = _crossing()
    for t in (t1, t2):
        t.loc[t.FrameID == 31, ["DetectionID", "State"]] = [np.nan, "occluded"]
    out = enforce_detection_exclusivity([t1, t2], join_step_px=50.0)
    frames_31 = [t for t in out if (t.FrameID == 31).any()]
    assert len(frames_31) == 1


def test_implausible_attachment_is_not_made_without_a_continuous_neighbour():
    # Both tracks jump onto the shared run: neither side may be attached.
    a, b, c = _ant(range(N), 0, 1), _ant(range(N), 120, 2), _ant(range(N), 240, 3)
    t1 = pd.concat([b[b.FrameID < 30], a[(a.FrameID >= 30) & (a.FrameID <= 32)]])
    t2 = pd.concat([c[c.FrameID < 30], a[(a.FrameID >= 30) & (a.FrameID <= 32)]])
    out = enforce_detection_exclusivity([t1, t2], join_step_px=50.0)
    assert _shared_rows(out) == 0
    assert all(_max_step(t) <= 50.0 for t in out)
