"""Enforce one trajectory per detection per frame after forward/backward merging.

Forward and backward tracking each assign a detection to at most one track per
frame, but when the two passes swap animals at different frames of a crossing,
resolving them can keep both versions: two final trajectories then hold the
same detections for a few frames, each continuing onto a different animal on
one side. Downstream, the identity solver treats such trajectories as rivals
for one label and leaves a whole trajectory unlabelled.

:func:`enforce_detection_exclusivity` resolves every shared run by re-cutting:
each trajectory in the run splits into the part before and the part after it;
the run joins the most continuous before-piece and after-piece (only when the
step is physically plausible), and the remaining pieces stand alone. A clean
swap therefore comes back as one trajectory per animal, and no trajectory keeps
the jump of a swap.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def _real_rows(traj: pd.DataFrame) -> pd.DataFrame:
    if "DetectionID" not in traj.columns:
        return traj.iloc[0:0]
    return traj[traj["DetectionID"].notna()]


def _shared_index(trajs: Sequence[pd.DataFrame]) -> dict:
    """(frame, detection id) -> sorted trajectory indices holding it, if > 1."""
    parts = []
    for i, traj in enumerate(trajs):
        real = _real_rows(traj)
        if not real.empty:
            parts.append(
                pd.DataFrame(
                    {
                        "f": real["FrameID"].to_numpy(dtype=np.int64),
                        "d": real["DetectionID"].to_numpy(dtype=np.float64),
                        "t": i,
                    }
                )
            )
    if not parts:
        return {}
    rows = pd.concat(parts, ignore_index=True)
    dup = rows[rows.duplicated(["f", "d"], keep=False)]
    if dup.empty:
        return {}
    return {
        (int(f), float(d)): sorted(set(g["t"].tolist()))
        for (f, d), g in dup.groupby(["f", "d"], sort=False)
        if g["t"].nunique() > 1
    }


def _row_at(traj: pd.DataFrame, frame: int):
    rows = traj[traj["FrameID"] == frame]
    return None if rows.empty else rows.iloc[0]


def _same_place(rows) -> bool:
    """All rows hold one detection, or are identical position-only rows."""
    dets = [r["DetectionID"] for r in rows]
    if all(pd.notna(d) for d in dets):
        return len({float(d) for d in dets}) == 1
    if all(pd.isna(d) for d in dets):
        xs = {(round(float(r["X"]), 3), round(float(r["Y"]), 3)) for r in rows}
        return len(xs) == 1 and not any(math.isnan(x) for x, _ in xs)
    return False


def _run_end(trajs, members, start, shared) -> int:
    """Last frame of the run of frames ``members`` share, starting at ``start``."""
    frame = start
    while True:
        nxt = frame + 1
        rows = [_row_at(trajs[i], nxt) for i in members]
        if any(r is None for r in rows) or not _same_place(rows):
            return frame
        det = rows[0]["DetectionID"]
        if pd.notna(det) and shared.get((nxt, float(det))) != members:
            return frame  # another trajectory joins or leaves: a new run
        frame = nxt


def _endpoint(piece: pd.DataFrame, last: bool):
    real = _real_rows(piece)
    if real.empty:
        real = piece[piece["X"].notna()]
    if real.empty:
        return None
    row = real.iloc[-1] if last else real.iloc[0]
    return int(row["FrameID"]), float(row["X"]), float(row["Y"])


def _speed(a, b) -> float:
    gap = abs(b[0] - a[0])
    return math.hypot(b[1] - a[1], b[2] - a[2]) / max(gap, 1)


def _best(pieces, anchor, last, join_step_px):
    """Index of the piece most continuous with ``anchor``, or None."""
    best, best_speed = None, math.inf
    for idx, piece in pieces:
        end = _endpoint(piece, last=last)
        if end is None:
            continue
        speed = _speed(end, anchor)
        if speed < best_speed:
            best, best_speed = idx, speed
    if best is not None and join_step_px is not None and best_speed > join_step_px:
        return None
    return best


def _resolve_run(trajs, members, start, end, join_step_px):
    frames = trajs[members[0]]["FrameID"]
    pre = [(i, trajs[i][trajs[i]["FrameID"] < start]) for i in members]
    post = [(i, trajs[i][trajs[i]["FrameID"] > end]) for i in members]
    run_first = _endpoint(trajs[members[0]][frames == start], last=False)
    run_last = _endpoint(trajs[members[0]][frames == end], last=True)
    pre_i = _best([p for p in pre if not p[1].empty], run_first, True, join_step_px)
    post_i = _best([p for p in post if not p[1].empty], run_last, False, join_step_px)

    owner = (
        pre_i if pre_i is not None else (post_i if post_i is not None else members[0])
    )
    src = trajs[owner]
    run_rows = src[(src["FrameID"] >= start) & (src["FrameID"] <= end)]
    chain = [run_rows]
    if pre_i is not None:
        chain.insert(0, dict(pre)[pre_i])
    if post_i is not None:
        chain.append(dict(post)[post_i])
    out = [pd.concat(chain, ignore_index=True)]
    for i, piece in pre:
        if i != pre_i and not piece.empty:
            out.append(piece.reset_index(drop=True))
    for i, piece in post:
        if i != post_i and not piece.empty:
            out.append(piece.reset_index(drop=True))
    return out


def enforce_detection_exclusivity(
    trajs: Sequence[pd.DataFrame], join_step_px: float | None = None
) -> list[pd.DataFrame]:
    """Return trajectories in which no detection belongs to two trajectories.

    Parameters
    ----------
    trajs : sequence of DataFrame
        Merged trajectories with ``FrameID``, ``X``, ``Y`` and ``DetectionID``
        (NaN on rows with no detection, e.g. occluded predictions).
    join_step_px : float, optional
        Largest plausible per-frame step for re-attaching a piece to a shared
        run; a less continuous piece is left standing alone. ``None`` attaches
        the most continuous piece regardless.

    Returns
    -------
    list of DataFrame
        The input list unchanged (same objects) when nothing is shared;
        otherwise the re-cut trajectories, sorted by frame within each.
    """
    trajs = list(trajs)
    shared = _shared_index(trajs)
    if not shared:
        return trajs
    trajs = [
        t.sort_values("FrameID", kind="stable").reset_index(drop=True) for t in trajs
    ]
    limit = sum(len(t) for t in trajs) + 1
    for _ in range(limit):
        if not shared:
            return trajs
        frame, det = min(shared)
        members = shared[(frame, det)]
        end = _run_end(trajs, members, frame, shared)
        replacement = _resolve_run(trajs, members, frame, end, join_step_px)
        keep = [t for i, t in enumerate(trajs) if i not in set(members)]
        trajs = keep[: min(members)] + replacement + keep[min(members) :]
        shared = _shared_index(trajs)
    raise RuntimeError("detection exclusivity did not converge")
