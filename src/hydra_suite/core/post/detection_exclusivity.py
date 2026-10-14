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
    if "DetectionID" not in traj.columns:
        return []
    real = _real_rows(traj)
    return list(
        zip(
            real["FrameID"].to_numpy(dtype=np.int64).tolist(),
            real["DetectionID"].to_numpy(dtype=np.float64).tolist(),
        )
    )


def _frames(traj: pd.DataFrame) -> np.ndarray:
    return traj["FrameID"].to_numpy()


def _slice(traj: pd.DataFrame, lo=None, hi=None) -> pd.DataFrame:
    """Rows with ``lo <= FrameID <= hi`` of a frame-sorted trajectory."""
    frames = _frames(traj)
    a = 0 if lo is None else int(np.searchsorted(frames, lo, side="left"))
    b = len(frames) if hi is None else int(np.searchsorted(frames, hi, side="right"))
    return traj.iloc[a:b]


def _row_at(traj: pd.DataFrame, frame: int):
    rows = _slice(traj, frame, frame)
    return None if rows.empty else rows.iloc[0]


def _same_place(rows) -> bool:
    """All rows hold one detection, or are identical position-only rows."""
    dets = [r.get("DetectionID", np.nan) for r in rows]
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
        det = rows[0].get("DetectionID", np.nan)
        if pd.notna(det) and owners.get((nxt, float(det))) != member_set:
            return frame  # another trajectory joins or leaves: a new run
        frame = nxt


def _detection_endpoint(piece: pd.DataFrame, last: bool):
    """Last/first detection-backed (frame, x, y) of a piece, else None.

    Predicted rows cannot decide continuity: a coasting prediction next to a
    shared run is not evidence that the animal continues there.
    """
    real = _real_rows(piece)
    if real.empty:
        return None
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
    # Trajectories are frame-sorted here, so position slicing replaces the
    # boolean filters that dominated this pass on crowded clips.
    run0 = _slice(trajs[members[0]], start, end)
    pre = [(i, _slice(trajs[i], None, start - 1)) for i in members]
    post = [(i, _slice(trajs[i], end + 1, None)) for i in members]
    pre_i = _best(pre, _run_endpoint(run0, last=False), True, join_step_px)
    post_i = _best(post, _run_endpoint(run0, last=True), False, join_step_px)

    owner = (
        pre_i if pre_i is not None else (post_i if post_i is not None else members[0])
    )
    chain = [_slice(trajs[owner], start, end)]
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


def trim_coasting_ends(trajs: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Drop each trajectory's predicted rows before its first and after its
    last detection.

    When a track loses its animal the tracker keeps predicting ("coasting")
    for a few frames before giving up, and a new track often already holds the
    animal by then. Those predicted rows are not observations: they sit on
    other animals' detections, overlap the successor track, and make the
    identity solver see two animals where there is one. Gaps between
    detections are kept (interpolation fills them). Trajectories without a
    ``DetectionID`` column, or with no detection at all, are returned
    unchanged.
    """
    out = []
    for traj in trajs:
        if "DetectionID" not in traj.columns or traj.empty:
            out.append(traj)
            continue
        real = traj["DetectionID"].notna().to_numpy()
        order = np.argsort(traj["FrameID"].to_numpy(), kind="stable")
        real_sorted = real[order]
        if not real_sorted.any() or (real_sorted[0] and real_sorted[-1]):
            out.append(traj)
            continue
        first = int(np.argmax(real_sorted))
        last = len(real_sorted) - 1 - int(np.argmax(real_sorted[::-1]))
        keep = order[first : last + 1]
        out.append(traj.iloc[np.sort(keep)].reset_index(drop=True))
    return out


def drop_predictions_on_other_detections(
    df: pd.DataFrame, tolerance_px: float = 1.0
) -> pd.DataFrame:
    """Drop rows without their own detection that sit on another trajectory's
    detection in the same frame.

    A predicted, interpolated or copied position cannot occupy an animal that
    another trajectory is detecting at that moment. Such rows appear after
    merging (gap fills, relinked fragments, position-only copies of another
    trajectory) and made two trajectories claim one animal. Rows backed by a
    detection are never dropped, so two detections of one animal (a detector
    double box) are left alone. ``df`` needs ``TrajectoryID``, ``FrameID``,
    ``X``, ``Y`` and ``DetectionID``; otherwise it is returned unchanged.
    """
    needed = {"TrajectoryID", "FrameID", "X", "Y", "DetectionID"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return df
    real = df["DetectionID"].notna()
    guess = ~real & df["X"].notna() & df["Y"].notna()
    if not guess.any() or not real.any():
        return df
    dets = df.loc[real, ["FrameID", "TrajectoryID", "X", "Y"]]
    cand = df.loc[guess, ["FrameID", "TrajectoryID", "X", "Y"]].reset_index()
    pairs = cand.merge(dets, on="FrameID", suffixes=("", "_det"))
    pairs = pairs[pairs["TrajectoryID"] != pairs["TrajectoryID_det"]]
    close = np.hypot(pairs["X"] - pairs["X_det"], pairs["Y"] - pairs["Y_det"])
    drop = pd.Index(pairs.loc[close <= tolerance_px, "index"].unique())
    if drop.empty:
        return df
    return df.drop(index=drop)
