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

import heapq
import math
from typing import Sequence

import numpy as np
import pandas as pd


def _real_rows(traj: pd.DataFrame) -> pd.DataFrame:
    if "DetectionID" not in traj.columns:
        return traj.iloc[0:0]
    return traj[traj["DetectionID"].notna()]


def _keys(traj: pd.DataFrame) -> list[tuple[int, float]]:
    real = _real_rows(traj)
    return list(
        zip(
            real["FrameID"].to_numpy(dtype=np.int64).tolist(),
            real["DetectionID"].to_numpy(dtype=np.float64).tolist(),
        )
    )


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


def _run_end(trajs, members, start, owners) -> int:
    """Last frame of the run of frames ``members`` share, starting at ``start``."""
    frame = start
    member_set = set(members)
    while True:
        nxt = frame + 1
        rows = [_row_at(trajs[i], nxt) for i in members]
        if any(r is None for r in rows) or not _same_place(rows):
            return frame
        det = rows[0]["DetectionID"]
        if pd.notna(det) and owners.get((nxt, float(det))) != member_set:
            return frame  # another trajectory joins or leaves: a new run
        frame = nxt


def _detection_endpoint(piece: pd.DataFrame, last: bool):
    """Last/first detection-backed (frame, x, y) of a piece, else None.

    Predicted rows cannot decide continuity: a coasting prediction next to a
    shared run is not evidence that the animal continues there.
    """
    real = _real_rows(piece)
    real = real[real["X"].notna()]
    if real.empty:
        return None
    row = real.iloc[-1] if last else real.iloc[0]
    return int(row["FrameID"]), float(row["X"]), float(row["Y"])


def _run_endpoint(rows: pd.DataFrame, last: bool):
    rows = rows[rows["X"].notna()]
    row = rows.iloc[-1] if last else rows.iloc[0]
    return int(row["FrameID"]), float(row["X"]), float(row["Y"])


def _speed(a, b) -> float:
    gap = abs(b[0] - a[0])
    return math.hypot(b[1] - a[1], b[2] - a[2]) / max(gap, 1)


def _best(pieces, anchor, last, join_step_px):
    """Key of the piece most continuous with ``anchor``, or None."""
    best, best_speed = None, math.inf
    for key, piece in pieces:
        end = _detection_endpoint(piece, last=last)
        if end is None:
            continue
        speed = _speed(end, anchor)
        if speed < best_speed:
            best, best_speed = key, speed
    if best is not None and join_step_px is not None and best_speed > join_step_px:
        return None
    return best


def _resolve_run(trajs, members, start, end, join_step_px):
    first = trajs[members[0]]
    run0 = first[(first["FrameID"] >= start) & (first["FrameID"] <= end)]
    pre = [(i, trajs[i][trajs[i]["FrameID"] < start]) for i in members]
    post = [(i, trajs[i][trajs[i]["FrameID"] > end]) for i in members]
    pre_i = _best(pre, _run_endpoint(run0, last=False), True, join_step_px)
    post_i = _best(post, _run_endpoint(run0, last=True), False, join_step_px)

    owner = (
        pre_i if pre_i is not None else (post_i if post_i is not None else members[0])
    )
    src = trajs[owner]
    chain = [src[(src["FrameID"] >= start) & (src["FrameID"] <= end)]]
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
    owners: dict[tuple[int, float], set[int]] = {}
    for i, traj in enumerate(trajs):
        for key in _keys(traj):
            owners.setdefault(key, set()).add(i)
    heap = [key for key, held in owners.items() if len(held) > 1]
    if not heap:
        return trajs

    # Stable integer handles: replaced trajectories get fresh handles, and the
    # output keeps each replacement at its parents' position.
    live: dict[int, pd.DataFrame] = {
        i: t.sort_values("FrameID", kind="stable").reset_index(drop=True)
        for i, t in enumerate(trajs)
    }
    rank: dict[int, tuple] = {i: (i,) for i in live}
    next_handle = len(trajs)
    heapq.heapify(heap)
    while heap:
        key = heapq.heappop(heap)
        held = owners.get(key)
        if held is None or len(held) < 2:
            continue  # resolved by an earlier re-cut
        members = sorted(held, key=lambda h: rank[h])
        end = _run_end(live, members, key[0], owners)
        replacement = _resolve_run(live, members, key[0], end, join_step_px)
        base = min(rank[h] for h in members)
        for h in members:
            for k in _keys(live.pop(h)):
                owners[k].discard(h)
            del rank[h]
        for n, piece in enumerate(replacement):
            handle, next_handle = next_handle, next_handle + 1
            live[handle] = piece
            rank[handle] = base + (n,)
            for k in _keys(piece):
                held_k = owners.setdefault(k, set())
                held_k.add(handle)
                if len(held_k) == 2:
                    heapq.heappush(heap, k)
    return [live[h] for h in sorted(live, key=lambda h: rank[h])]
