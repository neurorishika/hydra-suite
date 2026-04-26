"""Identity fragment utilities.

Identity Phase 4 helpers for building, iterating, and writing fragment-level
identity outputs back into the trajectory DataFrame.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def iter_fragments(
    fragments_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
) -> Iterator[tuple[int, pd.Series, pd.DataFrame]]:
    """Yield ``(fragment_id, fragment_row, traj_slice)`` for each fragment.

    Parameters
    ----------
    fragments_df:
        Output of ``build_identity_fragments`` or
        ``solve_fragment_identity_assignment``.
    trajectories_df:
        The full trajectory DataFrame.

    Yields
    ------
    fragment_id:
        Integer fragment ID.
    fragment_row:
        Single-row Series from *fragments_df*.
    traj_slice:
        Rows of *trajectories_df* belonging to this fragment's trajectory and
        within ``[StartFrame, EndFrame]`` inclusive.
    """
    for _, frag_row in fragments_df.iterrows():
        frag_id = int(frag_row["FragmentID"])
        traj_id = frag_row["TrajectoryID"]
        start = int(frag_row["StartFrame"])
        end = int(frag_row["EndFrame"])
        mask = (
            (trajectories_df["TrajectoryID"] == traj_id)
            & (trajectories_df["FrameID"] >= start)
            & (trajectories_df["FrameID"] <= end)
        )
        yield frag_id, frag_row, trajectories_df[mask]


def fragment_overlap_matrix(fragments_df: pd.DataFrame) -> np.ndarray:
    """Return a boolean overlap matrix ``O[i, j]`` for all fragment pairs.

    ``O[i, j]`` is ``True`` if fragments *i* and *j* are temporally
    overlapping (same or different trajectories).  The matrix is symmetric
    and has False on the diagonal.

    Used by the global uniqueness-constraint solver to build the conflict
    graph.

    Parameters
    ----------
    fragments_df:
        Output of ``build_identity_fragments``.

    Returns
    -------
    np.ndarray
        Shape ``(N, N)`` bool where N = ``len(fragments_df)``.
    """
    n = len(fragments_df)
    starts = fragments_df["StartFrame"].to_numpy(dtype=np.int64)
    ends = fragments_df["EndFrame"].to_numpy(dtype=np.int64)

    O = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if not (ends[i] < starts[j] or ends[j] < starts[i]):
                O[i, j] = True
                O[j, i] = True
    return O


def apply_fragment_labels_to_trajectories(
    trajectories_df: pd.DataFrame,
    fragments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write the ``AssignedLabel`` from each fragment back into trajectories.

    Parameters
    ----------
    trajectories_df:
        Full trajectory DataFrame.
    fragments_df:
        Output of ``solve_fragment_identity_assignment`` (must have
        ``AssignedLabel`` column).

    Returns
    -------
    pd.DataFrame
        Copy of *trajectories_df* with a new ``IdentityOfflineLabel`` column.
        Rows not covered by any fragment receive ``None``.
    """
    df = trajectories_df.copy()
    df["IdentityOfflineLabel"] = None

    for _, frag_row in fragments_df.iterrows():
        assigned = frag_row.get("AssignedLabel")
        if assigned is None or (
            isinstance(assigned, float) and np.isnan(assigned)
        ):
            continue
        traj_id = frag_row["TrajectoryID"]
        start = int(frag_row["StartFrame"])
        end = int(frag_row["EndFrame"])
        mask = (
            (df["TrajectoryID"] == traj_id)
            & (df["FrameID"] >= start)
            & (df["FrameID"] <= end)
        )
        df.loc[mask, "IdentityOfflineLabel"] = assigned

    return df


def fragment_summary_row(
    fragment_id: int,
    traj_id: object,
    frames: list[int],
    labels: list[str | None],
    confidences: list[float],
) -> dict:
    """Build a summary row dict suitable for appending to a fragments DataFrame.

    Parameters
    ----------
    fragment_id:
        Integer ID for this fragment.
    traj_id:
        Trajectory ID.
    frames:
        Sorted list of frame indices belonging to this fragment.
    labels:
        Smoothed identity labels aligned with *frames*.
    confidences:
        Smoothed identity confidences aligned with *frames*.

    Returns
    -------
    dict
        Keys: ``FragmentID``, ``TrajectoryID``, ``StartFrame``,
        ``EndFrame``, ``DominantLabel``, ``FragmentConf``,
        ``FragmentLength``.
    """
    from collections import Counter

    non_none = [l for l in labels if l is not None]
    dominant = Counter(non_none).most_common(1)[0][0] if non_none else None
    valid_confs = [c for c in confidences if c > 0]
    mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0

    return {
        "FragmentID": fragment_id,
        "TrajectoryID": traj_id,
        "StartFrame": frames[0] if frames else 0,
        "EndFrame": frames[-1] if frames else 0,
        "DominantLabel": dominant,
        "FragmentConf": mean_conf,
        "FragmentLength": len(frames),
    }
