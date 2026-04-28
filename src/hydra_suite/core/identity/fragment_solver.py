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
            if _LABEL_COL in seg.columns:  # noqa: SIM401
                label_col = seg[_LABEL_COL]
            else:
                label_col = pd.Series("unknown", index=seg.index, dtype=object)
            unknown_mask = label_col.isna() | label_col.astype(str).str.strip().isin(
                _UNKNOWN_VALUES
            )
            known_rows = seg[~unknown_mask]
            if not known_rows.empty:
                online_label = str(known_rows[_LABEL_COL].astype(str).mode().iloc[0])
                if _CONF_COL in seg.columns:
                    conf_vals = pd.to_numeric(seg[_CONF_COL], errors="coerce")
                    online_conf = (
                        float(np.nanmean(conf_vals.values))
                        if conf_vals.notna().any()
                        else 0.0
                    )
                else:
                    online_conf = 0.0
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


def _fragments_overlap(a: pd.Series, b: pd.Series) -> bool:
    return int(a["StartFrame"]) <= int(b["EndFrame"]) and int(b["StartFrame"]) <= int(
        a["EndFrame"]
    )


def _spatial_score_for_fragment(
    frag: pd.Series,
    identity: str,
    schedule: dict[str, list[dict]],
    max_velocity: float,
) -> float:
    """Gaussian spatial continuity score against nearest prior/following fragment of identity."""
    t0 = int(frag["StartFrame"])
    t1 = int(frag["EndFrame"])
    x0, y0 = float(frag["StartX"]), float(frag["StartY"])
    x1, y1 = float(frag["EndX"]), float(frag["EndY"])
    segs = schedule.get(identity, [])
    term_scores: list[float] = []

    prior = max(
        (s for s in segs if s["end_frame"] < t0),
        key=lambda s: s["end_frame"],
        default=None,
    )
    if prior and all(
        math.isfinite(v) for v in [x0, y0, prior["end_X"], prior["end_Y"]]
    ):
        gap = max(1, t0 - prior["end_frame"])
        sigma = max_velocity * gap
        dist = math.hypot(x0 - prior["end_X"], y0 - prior["end_Y"])
        term_scores.append(math.exp(-(dist**2) / (2.0 * sigma**2)))

    following = min(
        (s for s in segs if s["start_frame"] > t1),
        key=lambda s: s["start_frame"],
        default=None,
    )
    if following and all(
        math.isfinite(v) for v in [x1, y1, following["start_X"], following["start_Y"]]
    ):
        gap = max(1, following["start_frame"] - t1)
        sigma = max_velocity * gap
        dist = math.hypot(x1 - following["start_X"], y1 - following["start_Y"])
        term_scores.append(math.exp(-(dist**2) / (2.0 * sigma**2)))

    return float(np.mean(term_scores)) if term_scores else 0.5


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
    from itertools import combinations

    from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

    known_labels = list(catalog.labels[1:])
    if not known_labels or fragments_df.empty:
        out = fragments_df.copy()
        out["AssignedLabel"] = fragments_df["OnlineLabel"]
        return out

    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    spatial_w = float(params.get("FRAGMENT_SPATIAL_WEIGHT", 0.35))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    margin_thresh = float(params.get("ASSIGNMENT_MARGIN_THRESHOLD", 0.10))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))

    n_frags = len(fragments_df)
    n_labels = len(known_labels)
    frags = fragments_df.reset_index(drop=True)

    # Seed schedule from online labels so spatial scores are informed.
    schedule: dict[str, list[dict]] = {}
    for _, row in frags.iterrows():
        lbl = str(row["OnlineLabel"])
        if lbl in _UNKNOWN_VALUES or lbl not in known_labels:
            continue
        schedule.setdefault(lbl, []).append(
            {
                "start_frame": int(row["StartFrame"]),
                "end_frame": int(row["EndFrame"]),
                "start_X": float(row["StartX"]),
                "start_Y": float(row["StartY"]),
                "end_X": float(row["EndX"]),
                "end_Y": float(row["EndY"]),
            }
        )
    for lbl in schedule:
        schedule[lbl].sort(key=lambda s: s["start_frame"])

    # Build score matrix (n_frags x n_labels). -1 means ineligible.
    score_mat = np.full((n_frags, n_labels), -1.0, dtype=np.float64)
    for i, frag_row in frags.iterrows():
        mean_probs: dict[str, float] = frag_row["MeanCNNProbs"] or {}
        online_lbl = str(frag_row["OnlineLabel"])
        online_conf = float(frag_row["OnlineConfidence"])

        for j, label in enumerate(known_labels):
            cnn_s = float(mean_probs.get(label, 0.0))
            spatial_s = _spatial_score_for_fragment(frag_row, label, schedule, max_vel)
            prior_bonus = prior_w * online_conf if label == online_lbl else 0.0
            score_mat[i, j] = cnn_w * cnn_s + spatial_w * spatial_s + prior_bonus

    # Collect eligible (fragment, label) pairs for MILP.
    pairs = [
        (i, j) for i in range(n_frags) for j in range(n_labels) if score_mat[i, j] >= 0
    ]
    if not pairs:
        out = frags.copy()
        out["AssignedLabel"] = frags["OnlineLabel"]
        return out

    n_vars = len(pairs)
    pair_idx = {p: k for k, p in enumerate(pairs)}
    c_vec = np.array([-score_mat[i, j] for i, j in pairs], dtype=np.float64)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars, dtype=np.int8)

    A_rows: list[np.ndarray] = []
    lb_list: list[float] = []
    ub_list: list[float] = []

    # Each fragment assigned at most 1 label.
    for i in range(n_frags):
        idxs = [pair_idx[(i, j)] for j in range(n_labels) if (i, j) in pair_idx]
        if len(idxs) < 2:
            continue
        row = np.zeros(n_vars)
        row[idxs] = 1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(1.0)

    # Each label assigned at most 1 fragment globally.
    for j in range(n_labels):
        idxs = [pair_idx[(i, j)] for i in range(n_frags) if (i, j) in pair_idx]
        if len(idxs) < 2:
            continue
        row = np.zeros(n_vars)
        row[idxs] = 1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(1.0)

    # Pairwise: overlapping fragments cannot share a label.
    for a, b in combinations(range(n_frags), 2):
        if not _fragments_overlap(frags.iloc[a], frags.iloc[b]):
            continue
        for j in range(n_labels):
            ka = pair_idx.get((a, j))
            kb = pair_idx.get((b, j))
            if ka is None or kb is None:
                continue
            row = np.zeros(n_vars)
            row[ka] = 1.0
            row[kb] = 1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1.0)

    assigned: dict[int, str | None] = {i: None for i in range(n_frags)}
    try:
        if A_rows:
            A = np.vstack(A_rows)
            constraints = LinearConstraint(
                A, lb=np.array(lb_list), ub=np.array(ub_list)
            )
            opt = milp(
                c_vec, constraints=constraints, integrality=integrality, bounds=bounds
            )
            if opt.success:
                for k, (i, j) in enumerate(pairs):
                    if opt.x[k] > 0.5:
                        assigned[i] = known_labels[j]
            else:
                log.warning(
                    "MILP returned status %s (%s); falling back to online labels.",
                    opt.status,
                    opt.message,
                )
        else:
            row_ind, col_ind = linear_sum_assignment(
                np.where(score_mat < 0, 1e6, -score_mat)
            )
            for r, c in zip(row_ind, col_ind):
                if score_mat[r, c] >= 0:
                    assigned[r] = known_labels[c]
    except Exception as exc:
        log.warning("MILP solve failed (%s); falling back to online labels.", exc)

    # Apply margin threshold: only accept re-assignments that beat second-best by margin_thresh.
    labels_out: list[str] = []
    for i, frag_row in frags.iterrows():
        online_lbl = str(frag_row["OnlineLabel"])
        milp_label = assigned.get(i)

        if milp_label is None or milp_label == online_lbl:
            labels_out.append(online_lbl)
            continue

        row_scores = score_mat[i]
        valid_scores = sorted(row_scores[row_scores >= 0], reverse=True)
        if (
            len(valid_scores) >= 2
            and (valid_scores[0] - valid_scores[1]) >= margin_thresh
        ):
            labels_out.append(milp_label)
        else:
            labels_out.append(online_lbl)

    out = frags.copy()
    out["AssignedLabel"] = labels_out
    return out


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
