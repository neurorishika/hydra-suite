"""
Post-processing tag identity utilities.

Functions to:
1. Assign a majority-vote TagID to each trajectory segment.
2. Detect identity swaps based on tag observations.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Majority-vote tag identity per trajectory
# ---------------------------------------------------------------------------


def resolve_tag_identities(
    trajectories_df: pd.DataFrame,
    tag_cache: Any,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Add ``TagID`` and ``TagVotes`` columns to *trajectories_df*.

    For each trajectory segment the tag observation cache is queried on every
    frame where the trajectory has a valid position.  The tag whose detection
    index is spatially closest to the trajectory's XY is assigned.  A majority
    vote across all frames determines the final ``TagID``.

    Parameters
    ----------
    trajectories_df:
        Standard MAT trajectory DataFrame (columns: ``FrameID``,
        ``TrajectoryID``, ``X``, ``Y``, …).
    tag_cache:
        An open :class:`TagObservationCache` in read mode.
    params:
        Tracking parameters dict.  Relevant keys:

        - ``TAG_ASSOCIATION_RADIUS`` (float, default 50.0) – max distance
          (pixels) between a trajectory position and a tag centre to count
          as an association.

    Returns
    -------
    A *copy* of *trajectories_df* with ``TagID`` (int, -1 = no tag) and
    ``TagVotes`` (int, number of frames that voted for the winning tag).
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if tag_cache is None:
        df = trajectories_df.copy()
        df["TagID"] = -1
        df["TagVotes"] = 0
        return df

    assoc_radius = float(params.get("TAG_ASSOCIATION_RADIUS", 50.0))
    df = trajectories_df.copy()
    df["TagID"] = -1
    df["TagVotes"] = 0

    for traj_id in df["TrajectoryID"].unique():
        mask = df["TrajectoryID"] == traj_id
        traj = df.loc[mask]
        votes: List[int] = []

        for _, row in traj.iterrows():
            fid = int(row["FrameID"])
            x, y = row["X"], row["Y"]
            if pd.isna(x) or pd.isna(y):
                continue
            obs = tag_cache.get_frame(fid)
            tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
            centers = obs.get("centers_xy", np.zeros((0, 2), dtype=np.float32))
            if len(tag_ids) == 0:
                continue
            # Find the closest tag centre
            dists = np.sqrt(
                (centers[:, 0] - float(x)) ** 2 + (centers[:, 1] - float(y)) ** 2
            )
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= assoc_radius:
                votes.append(int(tag_ids[best_idx]))

        if votes:
            counter = Counter(votes)
            winner, cnt = counter.most_common(1)[0]
            df.loc[mask, "TagID"] = winner
            df.loc[mask, "TagVotes"] = cnt

    return df


# ---------------------------------------------------------------------------
# 2.  Detect tag-based identity swaps
# ---------------------------------------------------------------------------


def _build_frame_trajs(
    trajectories_df: pd.DataFrame,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Build per-frame trajectory positions lookup."""
    frame_trajs: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for _, row in trajectories_df.iterrows():
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            continue
        frame_trajs[int(row["FrameID"])].append(
            (int(row["TrajectoryID"]), float(row["X"]), float(row["Y"]))
        )
    return frame_trajs


def _process_tag_observation(
    tid: int,
    assoc_traj: int,
    tag_last_traj: Dict[int, int],
    tag_streak: Dict[int, int],
    min_streak: int,
    fid: int,
    swaps: List[Dict[str, Any]],
) -> None:
    """Update tag tracking state and record swap if detected."""
    prev = tag_last_traj.get(tid)
    if prev is not None and prev != assoc_traj:
        streak = tag_streak.get(tid, 0)
        if streak >= min_streak:
            confidence = min(1.0, streak / (2.0 * min_streak))
            swaps.append(
                {
                    "frame": fid,
                    "tag_id": tid,
                    "from_traj": prev,
                    "to_traj": assoc_traj,
                    "confidence": confidence,
                }
            )
        tag_streak[tid] = 1
    else:
        tag_streak[tid] = tag_streak.get(tid, 0) + 1

    tag_last_traj[tid] = assoc_traj


def detect_tag_swaps(
    trajectories_df: pd.DataFrame,
    tag_cache: Any,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return a list of suspected identity swap events.

    A swap is reported when a tag that was consistently associated with
    trajectory *A* suddenly appears closest to trajectory *B*.

    Each swap event is a dict::

        {
            "frame": int,
            "tag_id": int,
            "from_traj": int,
            "to_traj": int,
            "confidence": float,   # 0–1
        }
    """
    if trajectories_df is None or trajectories_df.empty or tag_cache is None:
        return []

    assoc_radius = float(params.get("TAG_ASSOCIATION_RADIUS", 50.0))
    min_streak = int(params.get("TAG_SWAP_MIN_STREAK", 3))

    frame_trajs = _build_frame_trajs(trajectories_df)

    tag_last_traj: Dict[int, int] = {}
    tag_streak: Dict[int, int] = {}
    swaps: List[Dict[str, Any]] = []

    for fid in sorted(frame_trajs.keys()):
        obs = tag_cache.get_frame(fid)
        tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
        centers = obs.get("centers_xy", np.zeros((0, 2), dtype=np.float32))
        if len(tag_ids) == 0:
            continue

        trajs = frame_trajs.get(fid, [])
        if not trajs:
            continue

        traj_ids_arr = np.array([t[0] for t in trajs])
        traj_xy = np.array([[t[1], t[2]] for t in trajs], dtype=np.float32)

        for ti in range(len(tag_ids)):
            tid = int(tag_ids[ti])
            tc = centers[ti]
            dists = np.sqrt(np.sum((traj_xy - tc) ** 2, axis=1))
            best = int(np.argmin(dists))
            if dists[best] > assoc_radius:
                continue
            assoc_traj = int(traj_ids_arr[best])
            _process_tag_observation(
                tid, assoc_traj, tag_last_traj, tag_streak, min_streak, fid, swaps
            )

    return swaps
