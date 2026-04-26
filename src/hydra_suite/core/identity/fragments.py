"""Identity fragment utilities.

Identity Phase 4 helpers for writing fragment-level identity outputs back
into the trajectory DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        if assigned is None or (isinstance(assigned, float) and np.isnan(assigned)):
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
