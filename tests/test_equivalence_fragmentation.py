"""Fragmentation metrics and gates for the equivalence harness.

Byte-level equivalence alone missed an 8x fragmentation regression: enabling
pose silently switched on an association gate, so X1_d1 went from 298 raw
tracks to 2446 and ~400 track ends were followed next frame by a new track a
few px away (same ant, new ID). These gates catch that class of change.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "equivalence" / "fragmentation.py"
)
_spec = importlib.util.spec_from_file_location("equiv_fragmentation", _PATH)
frag = importlib.util.module_from_spec(_spec)
sys.modules["equiv_fragmentation"] = frag
_spec.loader.exec_module(frag)


def _track(tid, frames, x0, y0, step=2.0):
    return [
        {"TrajectoryID": tid, "FrameID": f, "X": x0 + step * i, "Y": y0, "Theta": 0.0}
        for i, f in enumerate(frames)
    ]


def _df(*tracks):
    return pd.DataFrame([row for t in tracks for row in t])


def test_continuous_tracks_have_no_handoffs():
    df = _df(_track(1, range(0, 50), 0, 0), _track(2, range(0, 50), 500, 500))
    m = frag.fragmentation_metrics(df, body_px=20.0)
    assert (m.tracks, m.handoffs, m.median_len) == (2, 0, 50)


def test_same_animal_restarted_under_new_id_is_a_handoff():
    # One ant: track 1 ends at frame 9, track 2 starts at frame 10, 2 px on.
    df = _df(_track(1, range(0, 10), 0, 0), _track(2, range(10, 20), 20, 0))
    m = frag.fragmentation_metrics(df, body_px=20.0)
    assert m.tracks == 2 and m.handoffs == 1


def test_far_away_new_track_is_not_a_handoff():
    df = _df(_track(1, range(0, 10), 0, 0), _track(2, range(10, 20), 400, 400))
    assert frag.fragmentation_metrics(df, body_px=20.0).handoffs == 0


def test_lost_rows_without_positions_are_ignored():
    rows = _track(1, range(0, 10), 0, 0) + [
        {"TrajectoryID": 1, "FrameID": 10, "X": float("nan"), "Y": float("nan")}
    ]
    m = frag.fragmentation_metrics(pd.DataFrame(rows), body_px=20.0)
    assert m.rows == 10 and m.tracks == 1


def _metrics(tracks, handoffs):
    return frag.FragmentationMetrics(
        rows=1000, tracks=tracks, median_len=10.0, handoffs=handoffs
    )


@pytest.mark.parametrize(
    "base,new,ok",
    [
        ((300, 0), (303, 0), True),  # X1_d1 with the fix: noise
        ((298, 0), (2446, 400), False),  # X1_d1 regression
        ((10, 0), (14, 3), True),  # small clips get absolute slack
        ((100, 10), (100, 40), False),  # handoffs alone regress
    ],
)
def test_regression_gate(base, new, ok):
    assert frag.fragmentation_regressed(_metrics(*base), _metrics(*new)) is (not ok)


def test_positional_match_ignores_ids_and_heading():
    a = _df(_track(1, range(0, 20), 0, 0), _track(2, range(0, 20), 300, 300))
    b = a.copy()
    b["TrajectoryID"] = b["TrajectoryID"].map({1: 7, 2: 9})
    b["Theta"] = 3.14159
    assert frag.positional_match_fraction(a, b, tol_px=0.5) == pytest.approx(1.0)


def test_positional_match_counts_moved_and_missing_points():
    a = _df(_track(1, range(0, 10), 0, 0))
    b = a.copy()
    b.loc[b.FrameID == 3, "X"] += 5.0  # moved
    b = b[b.FrameID != 4]  # missing
    assert frag.positional_match_fraction(a, b, tol_px=0.5) == pytest.approx(0.8)


def test_cli_exit_codes(tmp_path):
    a = _df(_track(1, range(0, 50), 0, 0))
    bad = _df(*[_track(i, range(5 * i, 5 * i + 5), 10 * i, 0) for i in range(10)])
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    bad.to_csv(pb, index=False)
    assert frag.main(["regression", str(pa), str(pa), "--body-px", "20"]) == 0
    assert frag.main(["regression", str(pa), str(pb), "--body-px", "20"]) == 1
    assert frag.main(["invariance", str(pa), str(pa), "--body-px", "20"]) == 0
    assert frag.main(["invariance", str(pa), str(pb), "--body-px", "20"]) == 1
    assert (
        frag.main(["regression", str(pa), str(tmp_path / "x.csv"), "--body-px", "20"])
        == 2
    )
