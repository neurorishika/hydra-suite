"""Track-fragmentation metrics and gates for the equivalence harness.

Byte-level comparison says *that* tracking changed, not whether it got worse.
An 8x fragmentation regression (pose extraction silently enabling an
association gate: X1_d1 298 -> 2446 raw tracks) needed these numbers to show
up. Two modes, both on raw forward CSVs:

``regression A B``  B must not fragment noticeably more than A
                    (track count and same-animal handoffs).
``invariance A B``  A and B come from the same video with a feature that must
                    not affect association (e.g. pose extraction on vs off):
                    the same positions must be tracked (IDs and heading are
                    ignored) and fragmentation must agree.

A *handoff* is a track ending at frame f followed, at f + 1, by a new track
starting within one body length of where it ended -- the same animal re-born
under a new ID. Exit codes match compare.py: 0 ok, 1 differences, 2 no data.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Regression gate: allow 25% growth plus absolute slack for short clips.
REGRESSION_RATIO = 1.25
REGRESSION_SLACK = 5
# Invariance gate: positions tracked must agree almost exactly, counts closely.
INVARIANCE_MIN_POSITIONAL_MATCH = 0.995
INVARIANCE_RATIO = 1.10
INVARIANCE_SLACK = 3


@dataclass(frozen=True)
class FragmentationMetrics:
    rows: int
    tracks: int
    median_len: float
    handoffs: int

    def __str__(self) -> str:
        return (
            f"rows={self.rows} tracks={self.tracks} "
            f"median_len={self.median_len:.0f} handoffs={self.handoffs}"
        )


def _positioned(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["X", "Y"])


def fragmentation_metrics(df: pd.DataFrame, body_px: float) -> FragmentationMetrics:
    v = _positioned(df)
    if v.empty:
        return FragmentationMetrics(0, 0, 0.0, 0)
    v = v.sort_values(["TrajectoryID", "FrameID"])
    grouped = v.groupby("TrajectoryID")
    lengths = grouped.size()
    ends = grouped.agg(f=("FrameID", "max"), x=("X", "last"), y=("Y", "last"))
    starts = grouped.agg(f=("FrameID", "min"), x=("X", "first"), y=("Y", "first"))
    starts_by_frame = {f: s[["x", "y"]].to_numpy() for f, s in starts.groupby("f")}
    handoffs = 0
    for tid, end in ends.iterrows():
        nxt = starts_by_frame.get(end.f + 1)
        if nxt is None:
            continue
        d = np.hypot(nxt[:, 0] - end.x, nxt[:, 1] - end.y)
        handoffs += int(d.min() <= body_px)
    return FragmentationMetrics(
        rows=int(len(v)),
        tracks=int(lengths.size),
        median_len=float(lengths.median()),
        handoffs=handoffs,
    )


def _grew(base: int, new: int, ratio: float, slack: int) -> bool:
    return new > base * ratio + slack


def fragmentation_regressed(
    base: FragmentationMetrics, new: FragmentationMetrics
) -> bool:
    return _grew(base.tracks, new.tracks, REGRESSION_RATIO, REGRESSION_SLACK) or _grew(
        base.handoffs, new.handoffs, REGRESSION_RATIO, REGRESSION_SLACK
    )


def positional_match_fraction(a: pd.DataFrame, b: pd.DataFrame, tol_px: float) -> float:
    """Fraction of tracked points (of the larger side) that both runs place
    within *tol_px* of each other in the same frame -- ID- and heading-blind."""
    a, b = _positioned(a), _positioned(b)
    total = max(len(a), len(b))
    if total == 0:
        return 1.0
    b_by_frame = {f: g[["X", "Y"]].to_numpy() for f, g in b.groupby("FrameID")}
    matched = 0
    for f, ga in a.groupby("FrameID"):
        gb = b_by_frame.get(f)
        if gb is None:
            continue
        pa = ga[["X", "Y"]].to_numpy()
        d = np.hypot(pa[:, None, 0] - gb[None, :, 0], pa[:, None, 1] - gb[None, :, 1])
        rows, cols = linear_sum_assignment(d)
        matched += int((d[rows, cols] <= tol_px).sum())
    return matched / total


def invariance_violated(
    a: pd.DataFrame, b: pd.DataFrame, body_px: float, tol_px: float = 0.5
) -> tuple[bool, float, FragmentationMetrics, FragmentationMetrics]:
    ma, mb = fragmentation_metrics(a, body_px), fragmentation_metrics(b, body_px)
    frac = positional_match_fraction(a, b, tol_px)

    def count_gap(x: int, y: int) -> bool:
        return _grew(min(x, y), max(x, y), INVARIANCE_RATIO, INVARIANCE_SLACK)

    bad = (
        frac < INVARIANCE_MIN_POSITIONAL_MATCH
        or count_gap(ma.tracks, mb.tracks)
        or count_gap(ma.handoffs, mb.handoffs)
    )
    return bad, frac, ma, mb


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.is_file():
        return None
    df = pd.read_csv(p, low_memory=False)
    return None if df.empty else df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["regression", "invariance"])
    ap.add_argument("a", help="baseline raw forward CSV")
    ap.add_argument("b", help="candidate raw forward CSV")
    ap.add_argument(
        "--body-px", type=float, required=True, help="one body length in px"
    )
    args = ap.parse_args(argv)

    a, b = _load(args.a), _load(args.b)
    if a is None or b is None:
        print("  NO DATA (missing or empty CSV)")
        return 2
    if args.mode == "regression":
        ma = fragmentation_metrics(a, args.body_px)
        mb = fragmentation_metrics(b, args.body_px)
        print(f"  baseline : {ma}\n  candidate: {mb}")
        bad = fragmentation_regressed(ma, mb)
        print(
            "VERDICT: "
            + (
                "FRAGMENTATION REGRESSED ❌"
                if bad
                else "no fragmentation regression ✅"
            )
            + f"  (tracks/handoffs <= {REGRESSION_RATIO}x + {REGRESSION_SLACK})"
        )
        return 1 if bad else 0
    bad, frac, ma, mb = invariance_violated(a, b, args.body_px)
    print(f"  feature on : {ma}\n  feature off: {mb}\n  positions shared: {frac:.2%}")
    print(
        "VERDICT: "
        + ("ASSOCIATION CHANGED ❌" if bad else "association invariant ✅")
        + f"  (positions >= {INVARIANCE_MIN_POSITIONAL_MATCH:.1%}, "
        f"tracks/handoffs within {INVARIANCE_RATIO}x + {INVARIANCE_SLACK})"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
