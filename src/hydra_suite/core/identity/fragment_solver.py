"""Global identity fragment solver.

Replaces the HMM-based offline decoder with:
1. PELT changepoint detection on per-trajectory CNN probability matrices.
2. Fragment building from detected changepoints.
3. Global MILP assignment: maximises spatial continuity + CNN/tag evidence
   with a confidence-weighted online-label prior and a margin threshold.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from hydra_suite.core.identity.catalog import IdentityCatalog

log = logging.getLogger(__name__)

_LABEL_COL = "IdentityAssignedLabel"
_CONF_COL = "IdentityAssignedConfidence"
_UNKNOWN_VALUES = frozenset({"", "unknown"})


def detect_identity_changepoints(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> dict[Any, list[int]]:
    """Return {traj_id: [split_frame_indices]} using PELT on CNN prob matrix.

    Each split_frame_index is the *inclusive end* (last FrameID) of a segment.
    ``build_fragments`` treats these as inclusive boundaries: segment k spans
    FrameIDs [split_indices[k-1]+1, split_indices[k]].
    Trajectories with no CNN evidence or fewer than min_fragment_frames*2
    rows are returned with no splits.
    """
    try:
        import ruptures as rpt
    except ImportError:
        log.warning(
            "ruptures not installed; changepoint detection skipped — install ruptures>=1.1"
        )
        return {}

    penalty = float(params.get("CHANGEPOINT_PENALTY", 3.0))
    min_frames = int(params.get("MIN_FRAGMENT_FRAMES", 5))
    known_labels = list(catalog.labels[1:])

    # Find CNN_*_Prob columns for known labels only.
    prob_cols: list[str] = []
    for label in known_labels:
        suffix = f"_{label}_Prob"
        for col in df.columns:
            if str(col).endswith(suffix):
                prob_cols.append(col)
                break

    if not prob_cols:
        return {}

    result: dict[Any, list[int]] = {}

    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        grp_sorted = grp.sort_values("FrameID")
        if len(grp_sorted) < min_frames * 2:
            continue

        signal = (
            grp_sorted[prob_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.5)
            .values
        )
        # z-score per column to suppress magnitude drift.
        col_std = signal.std(axis=0)
        col_std[col_std < 1e-8] = 1.0
        signal = (signal - signal.mean(axis=0)) / col_std

        try:
            splits = (
                rpt.Pelt(model="rbf", min_size=min_frames, jump=1)
                .fit(signal)
                .predict(pen=penalty)
            )
        except Exception as exc:
            log.warning("PELT failed for traj %s: %s", traj_id, exc)
            continue

        # ruptures returns end-of-segment indices (1-indexed frame position in grp_sorted).
        # Convert to FrameID values (drop the final sentinel which equals len).
        frame_ids = grp_sorted["FrameID"].values
        split_frames = [
            int(frame_ids[s - 1]) for s in splits[:-1] if s < len(frame_ids)
        ]
        if split_frames:
            result[traj_id] = split_frames

    return result


def build_fragments(
    df: pd.DataFrame,
    changepoints: dict[Any, list[int]],
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Return a fragments DataFrame with one row per (traj_id, segment).

    Columns: TrajectoryID, FragmentID, StartFrame, EndFrame,
    StartX, StartY, EndX, EndY, MeanCNNProbs (dict serialised as object),
    OnlineLabel, OnlineConfidence.
    """
    known_labels = list(catalog.labels[1:])

    rows: list[dict] = []
    frag_counter = 0

    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        grp_sorted = grp.sort_values("FrameID").reset_index(drop=True)
        frames = grp_sorted["FrameID"].values
        split_frames = sorted(changepoints.get(traj_id, []))

        # Build segment boundaries: list of (start_frame, end_frame) inclusive.
        boundaries: list[tuple[int, int]] = []
        prev = int(frames[0])
        for sf in split_frames:
            if prev <= sf < int(frames[-1]):
                boundaries.append((prev, sf))
                prev = sf + 1
        boundaries.append((prev, int(frames[-1])))

        for start_f, end_f in boundaries:
            mask = (grp_sorted["FrameID"] >= start_f) & (grp_sorted["FrameID"] <= end_f)
            seg = grp_sorted[mask]
            if seg.empty:
                continue

            # Spatial endpoints.
            valid_xy = (
                seg[seg["X"].notna() & seg["Y"].notna()].sort_values("FrameID")
                if "X" in seg.columns and "Y" in seg.columns
                else pd.DataFrame()
            )
            if not valid_xy.empty:
                sx, sy = float(valid_xy.iloc[0]["X"]), float(valid_xy.iloc[0]["Y"])
                ex, ey = float(valid_xy.iloc[-1]["X"]), float(valid_xy.iloc[-1]["Y"])
            else:
                sx = sy = ex = ey = math.nan

            # CNN mean probabilities.
            mean_probs: dict[str, float] = {}
            for label in known_labels:
                suffix = f"_{label}_Prob"
                prob_col = next(
                    (c for c in seg.columns if str(c).endswith(suffix)), None
                )
                if prob_col is not None:
                    vals = pd.to_numeric(seg[prob_col], errors="coerce")
                    if vals.notna().any():
                        mean_probs[label] = float(np.nanmean(vals.values))

            # Online label: dominant non-unknown label (or "unknown" if all unknown).
            label_col = seg.get(_LABEL_COL, pd.Series(dtype=object))
            unknown_mask = label_col.isna() | label_col.astype(str).str.strip().isin(
                _UNKNOWN_VALUES
            )
            known_rows = seg[~unknown_mask]
            if not known_rows.empty:
                online_label = str(known_rows[_LABEL_COL].astype(str).mode().iloc[0])
                conf_vals = pd.to_numeric(
                    seg.get(_CONF_COL, pd.Series(dtype=float)), errors="coerce"
                )
                online_conf = (
                    float(np.nanmean(conf_vals.values))
                    if conf_vals.notna().any()
                    else 0.0
                )
            else:
                online_label = "unknown"
                online_conf = 0.0

            rows.append(
                {
                    "TrajectoryID": traj_id,
                    "FragmentID": frag_counter,
                    "StartFrame": start_f,
                    "EndFrame": end_f,
                    "StartX": sx,
                    "StartY": sy,
                    "EndX": ex,
                    "EndY": ey,
                    "MeanCNNProbs": mean_probs,
                    "OnlineLabel": online_label,
                    "OnlineConfidence": online_conf,
                }
            )
            frag_counter += 1

    if not rows:
        return pd.DataFrame(
            columns=[
                "TrajectoryID",
                "FragmentID",
                "StartFrame",
                "EndFrame",
                "StartX",
                "StartY",
                "EndX",
                "EndY",
                "MeanCNNProbs",
                "OnlineLabel",
                "OnlineConfidence",
            ]
        )
    return pd.DataFrame(rows)


def solve_global_assignment(
    fragments_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Return fragments_df with an added AssignedLabel column.

    Uses MILP with:
    - Spatial continuity score between consecutive fragments of the same identity.
    - CNN evidence score from MeanCNNProbs.
    - Online label prior: online_prior_weight * OnlineConfidence bonus for the
      online label column.
    - Margin threshold: only re-assign when best_score - second_best > threshold;
      otherwise keep OnlineLabel.
    - Uniqueness: at most one fragment per identity per overlapping time window.
    """
    raise NotImplementedError


def apply_fragment_labels(
    df: pd.DataFrame,
    fragments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write AssignedLabel from fragments back into trajectories.

    Updates IdentityAssignedLabel, IdentityAssignedConfidence,
    IdentityAssignedID, IdentityCommitted in the trajectory DataFrame.
    Rows not covered by any fragment are unchanged.
    Returns a copy.
    """
    raise NotImplementedError


def run_fragment_solver(
    trajectories_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """End-to-end fragment solver: detect -> build -> assign -> apply.

    Parameters
    ----------
    trajectories_df : post-augmentation trajectory DataFrame.
    catalog : IdentityCatalog for the run.
    params : optional overrides. Keys:
        CHANGEPOINT_PENALTY          float  default 3.0
        MIN_FRAGMENT_FRAMES          int    default 5
        FRAGMENT_CNN_WEIGHT          float  default 0.40
        FRAGMENT_SPATIAL_WEIGHT      float  default 0.35
        ONLINE_PRIOR_WEIGHT          float  default 0.25
        ASSIGNMENT_MARGIN_THRESHOLD  float  default 0.10
        MAX_VELOCITY_BREAK           float  default 50.0
        TAG_IDENTITY_LABELS          list   default []
        FRAGMENT_SOLVER_ILP_TIME_LIMIT float default 30.0
    """
    raise NotImplementedError
