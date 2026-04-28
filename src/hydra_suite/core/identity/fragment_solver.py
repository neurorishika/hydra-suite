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

    PELT model is read from params["PELT_MODEL"] (l1 / l2 / rbf; default rbf).
    Z-scoring is skipped for l1 since l1 is already median-based.
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
    pelt_model = str(params.get("PELT_MODEL", "rbf")).lower()
    if pelt_model not in ("l1", "l2", "rbf"):
        pelt_model = "rbf"
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
        # Z-score per column to suppress magnitude drift.
        # Skipped for l1 which is already median-based and scale-insensitive.
        if pelt_model != "l1":
            col_std = signal.std(axis=0)
            col_std[col_std < 1e-8] = 1.0
            signal = (signal - signal.mean(axis=0)) / col_std

        try:
            splits = (
                rpt.Pelt(model=pelt_model, min_size=min_frames, jump=1)
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


def split_trajectories_at_changepoints(
    df: pd.DataFrame,
    changepoints: dict[Any, list[int]],
    params: dict[str, Any],
) -> pd.DataFrame:
    """Split trajectories at PELT-detected changepoints, assigning new TrajectoryIDs.

    Each value in changepoints is a list of FrameID values that are the
    *inclusive end* of a segment (same convention as detect_identity_changepoints).
    Sub-segments shorter than MIN_FRAGMENT_FRAMES rows are dropped.
    OriginalTrajectoryID is set to the pre-split TrajectoryID on all rows.
    Trajectories with no changepoints pass through unchanged.
    """
    min_frames = int(params.get("MIN_FRAGMENT_FRAMES", 5))

    to_split = {tid: sorted(sfs) for tid, sfs in changepoints.items() if sfs}
    if not to_split:
        out = df.copy()
        if "OriginalTrajectoryID" not in out.columns:
            out["OriginalTrajectoryID"] = out["TrajectoryID"]
        return out

    out = df.copy()
    if "OriginalTrajectoryID" not in out.columns:
        out["OriginalTrajectoryID"] = out["TrajectoryID"]

    next_id = int(out["TrajectoryID"].max()) + 1

    unchanged = out[~out["TrajectoryID"].isin(to_split)].copy()
    parts: list[pd.DataFrame] = [unchanged]

    for traj_id, split_frames in to_split.items():
        grp = out[out["TrajectoryID"] == traj_id].sort_values("FrameID")
        if grp.empty:
            continue

        first_frame = int(grp["FrameID"].min())
        last_frame = int(grp["FrameID"].max())

        # Build inclusive (start, end) boundaries from split_frames.
        boundaries: list[tuple[int, int]] = []
        prev = first_frame
        for sf in split_frames:
            if prev <= sf < last_frame:
                boundaries.append((prev, sf))
                prev = sf + 1
        boundaries.append((prev, last_frame))

        for start_f, end_f in boundaries:
            seg = grp[(grp["FrameID"] >= start_f) & (grp["FrameID"] <= end_f)].copy()
            if len(seg) < min_frames:
                continue
            seg["TrajectoryID"] = next_id
            seg["OriginalTrajectoryID"] = traj_id
            next_id += 1
            parts.append(seg)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["TrajectoryID", "FrameID"], kind="stable").reset_index(
        drop=True
    )
    return result


def build_fragments(
    df: pd.DataFrame,
    changepoints: dict[Any, list[int]],
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Return a fragments DataFrame with one row per (traj_id, segment).

    Columns: TrajectoryID, FragmentID, StartFrame, EndFrame,
    StartX, StartY, EndX, EndY, MeanCNNProbs (dict), MeanTagProbs (dict),
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

            # Tag evidence: fraction of frames with each known label's AprilTag.
            tag_probs: dict[str, float] = {}
            if "DetectedTagLabel" in seg.columns:
                tag_vals = seg["DetectedTagLabel"].dropna().astype(str).str.strip()
                tag_known = tag_vals[~tag_vals.isin(_UNKNOWN_VALUES)]
                n_rows = len(seg)
                if len(tag_known) > 0 and n_rows > 0:
                    for label in known_labels:
                        frac = float((tag_known == str(label)).sum()) / n_rows
                        if frac > 0.0:
                            tag_probs[label] = frac

            # Online label: dominant non-unknown label.
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
                # Confidence averaged over known-label rows only.
                if _CONF_COL in known_rows.columns:
                    conf_vals = pd.to_numeric(known_rows[_CONF_COL], errors="coerce")
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
                    "MeanTagProbs": tag_probs,
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
                "MeanTagProbs",
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


def _build_schedule(frags: pd.DataFrame, label_col: str) -> dict[str, list[dict]]:
    """Build a spatial schedule dict from a fragments DataFrame column."""
    schedule: dict[str, list[dict]] = {}
    for _, row in frags.iterrows():
        lbl = str(row[label_col])
        if lbl in _UNKNOWN_VALUES:
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
    return schedule


def _build_score_matrix(
    frags: pd.DataFrame,
    known_labels: list[str],
    schedule: dict[str, list[dict]],
    params: dict[str, Any],
) -> np.ndarray:
    """Build (n_frags x n_labels) score matrix.  -1 means ineligible."""
    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    spatial_w = float(params.get("FRAGMENT_SPATIAL_WEIGHT", 0.35))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    tag_w = float(params.get("FRAGMENT_TAG_WEIGHT", 0.15))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    # Normalize to sum 1 so the score is on a stable [0,1] scale regardless
    # of how users set the individual weights.
    total_w = cnn_w + spatial_w + prior_w + tag_w
    if total_w > 1e-9:
        cnn_w /= total_w
        spatial_w /= total_w
        prior_w /= total_w
        tag_w /= total_w

    n_frags = len(frags)
    n_labels = len(known_labels)
    score_mat = np.full((n_frags, n_labels), -1.0, dtype=np.float64)

    for i, frag_row in frags.iterrows():
        mean_probs: dict[str, float] = frag_row.get("MeanCNNProbs") or {}
        raw_tag = frag_row.get("MeanTagProbs")
        mean_tag_probs: dict[str, float] = raw_tag if isinstance(raw_tag, dict) else {}
        online_lbl = str(frag_row["OnlineLabel"])
        online_conf = float(frag_row["OnlineConfidence"])

        for j, label in enumerate(known_labels):
            cnn_s = float(mean_probs.get(label, 0.0))
            tag_s = float(mean_tag_probs.get(label, 0.0))
            spatial_s = _spatial_score_for_fragment(frag_row, label, schedule, max_vel)
            prior_bonus = prior_w * online_conf if label == online_lbl else 0.0
            score_mat[i, j] = (
                cnn_w * cnn_s + tag_w * tag_s + spatial_w * spatial_s + prior_bonus
            )

    return score_mat


def _milp_solve(
    frags: pd.DataFrame,
    known_labels: list[str],
    score_mat: np.ndarray,
    params: dict[str, Any],
) -> dict[int, str | None]:
    """Run the MILP and return {frag_index: assigned_label_or_None}.

    Uses scipy.sparse constraint matrix for memory efficiency.
    """
    from itertools import combinations

    from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
    from scipy.sparse import csr_matrix, lil_matrix

    n_frags = len(frags)
    n_labels = len(known_labels)
    pairs = [
        (i, j) for i in range(n_frags) for j in range(n_labels) if score_mat[i, j] >= 0
    ]
    if not pairs:
        return {i: None for i in range(n_frags)}

    n_vars = len(pairs)
    pair_idx = {p: k for k, p in enumerate(pairs)}
    c_vec = np.array([-score_mat[i, j] for i, j in pairs], dtype=np.float64)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars, dtype=np.int8)

    # Collect all constraints as (col_indices, lb, ub) before allocating matrix.
    constraint_specs: list[tuple[list[int], float, float]] = []

    # Each fragment assigned at most 1 label.
    for i in range(n_frags):
        idxs = [pair_idx[(i, j)] for j in range(n_labels) if (i, j) in pair_idx]
        if len(idxs) >= 2:
            constraint_specs.append((idxs, -np.inf, 1.0))

    # Pairwise: overlapping fragments cannot share a label.
    for a, b in combinations(range(n_frags), 2):
        if not _fragments_overlap(frags.iloc[a], frags.iloc[b]):
            continue
        for j in range(n_labels):
            ka = pair_idx.get((a, j))
            kb = pair_idx.get((b, j))
            if ka is None or kb is None:
                continue
            constraint_specs.append(([ka, kb], -np.inf, 1.0))

    assigned: dict[int, str | None] = {i: None for i in range(n_frags)}
    try:
        if constraint_specs:
            n_constraints = len(constraint_specs)
            A = lil_matrix((n_constraints, n_vars))
            lb_arr = np.empty(n_constraints)
            ub_arr = np.empty(n_constraints)
            for k, (col_idxs, lb_val, ub_val) in enumerate(constraint_specs):
                for ci in col_idxs:
                    A[k, ci] = 1.0
                lb_arr[k] = lb_val
                ub_arr[k] = ub_val
            time_limit = float(params.get("FRAGMENT_SOLVER_ILP_TIME_LIMIT", 30.0))
            constraints = LinearConstraint(csr_matrix(A), lb=lb_arr, ub=ub_arr)
            opt = milp(
                c_vec,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_limit, "disp": False},
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

    return assigned


def solve_global_assignment(
    fragments_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Return fragments_df with added AssignedLabel and AssignedScore columns.

    Two-pass strategy removes circular bias from the online-label schedule seed:
    - Pass 1: schedule seeded from OnlineLabel → MILP.
    - Pass 2: schedule re-seeded from Pass-1 labels → MILP again.

    Margin threshold applied to Pass-2 results: only re-assign when
    score(milp_label) - score(second_best) >= ASSIGNMENT_MARGIN_THRESHOLD.
    """
    known_labels = list(catalog.labels[1:])
    if not known_labels or fragments_df.empty:
        out = fragments_df.copy()
        out["AssignedLabel"] = fragments_df["OnlineLabel"]
        out["AssignedScore"] = 0.0
        return out

    margin_thresh = float(params.get("ASSIGNMENT_MARGIN_THRESHOLD", 0.10))
    frags = fragments_df.reset_index(drop=True)
    n_frags = len(frags)
    n_labels = len(known_labels)
    known_label_set = set(known_labels)

    def _catalog_label_or_unknown(lbl: str) -> str:
        # Non-catalog strings (e.g. AprilTag family names) become "unknown" so they
        # don't propagate into AssignedLabel or pollute the spatial schedule.
        return lbl if lbl in known_label_set else "unknown"

    # Pass 1: seed schedule from OnlineLabel.
    schedule1 = _build_schedule(frags, "OnlineLabel")
    score_mat1 = _build_score_matrix(frags, known_labels, schedule1, params)
    assigned1 = _milp_solve(frags, known_labels, score_mat1, params)

    # Build intermediate label list for schedule re-seeding (no margin threshold yet).
    # Filter to catalog labels so non-catalog online labels don't pollute the schedule.
    pass1_labels = [
        (
            str(assigned1.get(i))
            if assigned1.get(i) is not None
            else _catalog_label_or_unknown(str(frags.iloc[i]["OnlineLabel"]))
        )
        for i in range(n_frags)
    ]

    # Pass 2: re-seed schedule from Pass-1 labels to remove online-label bias.
    frags_pass1 = frags.copy()
    frags_pass1["_pass1_label"] = pass1_labels
    schedule2 = _build_schedule(frags_pass1, "_pass1_label")
    score_mat2 = _build_score_matrix(frags, known_labels, schedule2, params)
    assigned2 = _milp_solve(frags, known_labels, score_mat2, params)

    # Apply margin threshold: accept re-assignment only when it beats second-best.
    # OnlineLabel fallbacks are filtered to known catalog labels — non-catalog values
    # (e.g. AprilTag family strings written by the online decoder) become "unknown"
    # and are skipped by apply_fragment_labels rather than polluting the output.
    labels_out: list[str] = []
    for i in range(n_frags):
        online_lbl = _catalog_label_or_unknown(str(frags.iloc[i]["OnlineLabel"]))
        milp_label = assigned2.get(i)

        if milp_label is None or milp_label == online_lbl:
            labels_out.append(online_lbl)
            continue

        try:
            milp_j = known_labels.index(milp_label)
        except ValueError:
            labels_out.append(online_lbl)
            continue

        milp_score = score_mat2[i, milp_j]
        other_scores = [
            score_mat2[i, j]
            for j in range(n_labels)
            if j != milp_j and score_mat2[i, j] >= 0
        ]
        second_best = max(other_scores) if other_scores else 0.0

        if milp_score - second_best >= margin_thresh:
            labels_out.append(milp_label)
        else:
            labels_out.append(online_lbl)

    # AssignedScore from pass-2 matrix for the accepted label.
    assigned_scores: list[float] = []
    for i in range(n_frags):
        label = labels_out[i]
        if label in known_labels:
            j = known_labels.index(label)
            s = score_mat2[i, j]
            assigned_scores.append(float(s) if s >= 0 else 0.0)
        else:
            assigned_scores.append(0.0)

    out = frags.copy()
    out["AssignedLabel"] = labels_out
    out["AssignedScore"] = assigned_scores
    return out


def apply_fragment_labels(
    df: pd.DataFrame,
    fragments_df: pd.DataFrame,
    catalog: IdentityCatalog | None = None,
) -> pd.DataFrame:
    """Write AssignedLabel from fragments back into trajectories.

    Updates IdentityAssignedLabel, IdentityFragmentScore, IdentityCommitted.
    IdentityAssignedConfidence is preserved (not overwritten — it comes from
    the online decoder and is probabilistic; AssignedScore is an MILP objective
    value with different semantics).
    IdentityAssignedID is updated from catalog when catalog is provided.
    Rows not covered by any fragment are unchanged.
    Returns a copy.
    """
    original_index = df.index
    out = df.copy().reset_index(drop=True)

    if "IdentityAssignedLabel" not in out.columns:
        out["IdentityAssignedLabel"] = np.nan
    if "IdentityAssignedConfidence" not in out.columns:
        out["IdentityAssignedConfidence"] = np.nan
    if "IdentityCommitted" not in out.columns:
        out["IdentityCommitted"] = False
    if "IdentityFragmentScore" not in out.columns:
        out["IdentityFragmentScore"] = np.nan

    if "AssignedLabel" not in fragments_df.columns:
        out.index = original_index
        return out

    frag_cols = ["TrajectoryID", "StartFrame", "EndFrame", "AssignedLabel"]
    if "AssignedScore" in fragments_df.columns:
        frag_cols.append("AssignedScore")

    valid_frags = fragments_df[
        fragments_df["AssignedLabel"].notna()
        & ~fragments_df["AssignedLabel"].astype(str).str.strip().isin(_UNKNOWN_VALUES)
    ][frag_cols].copy()

    if "AssignedScore" not in valid_frags.columns:
        valid_frags["AssignedScore"] = np.nan

    if valid_frags.empty:
        out.index = original_index
        return out

    # Vectorized range-join: cross-join on TrajectoryID, then filter by frame range.
    tmp = (
        out[["TrajectoryID", "FrameID"]]
        .assign(_idx=np.arange(len(out)))
        .merge(valid_frags, on="TrajectoryID", how="inner")
    )
    in_range = (tmp["FrameID"] >= tmp["StartFrame"]) & (
        tmp["FrameID"] <= tmp["EndFrame"]
    )
    matched = tmp[in_range].drop_duplicates(subset=["_idx"])

    if matched.empty:
        out.index = original_index
        return out

    row_positions = matched["_idx"].values
    out.loc[row_positions, "IdentityAssignedLabel"] = matched["AssignedLabel"].values
    out.loc[row_positions, "IdentityFragmentScore"] = matched["AssignedScore"].values
    out.loc[row_positions, "IdentityCommitted"] = True

    if catalog is not None and "IdentityAssignedID" in out.columns:
        for label_val in matched["AssignedLabel"].unique():
            label_str = str(label_val)
            if catalog.contains(label_str):
                lbl_mask = matched["AssignedLabel"].astype(str) == label_str
                out.loc[matched.loc[lbl_mask, "_idx"].values, "IdentityAssignedID"] = (
                    catalog.index_of(label_str)
                )

    # Sanitize non-catalog values in IdentityAssignedLabel (e.g. AprilTag family
    # strings written by the online decoder that never mapped to a real identity).
    # Any value that is neither a catalog label nor already "unknown"/"" is cleared.
    if catalog is not None and "IdentityAssignedLabel" in out.columns:
        valid_labels = {str(l) for l in catalog.labels} | _UNKNOWN_VALUES
        bad_mask = out["IdentityAssignedLabel"].notna() & ~out[
            "IdentityAssignedLabel"
        ].astype(str).str.strip().isin(valid_labels)
        if bad_mask.any():
            out.loc[bad_mask, "IdentityAssignedLabel"] = "unknown"
            if "IdentityAssignedID" in out.columns:
                out.loc[bad_mask, "IdentityAssignedID"] = np.nan

    out.index = original_index
    return out


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
        CHANGEPOINT_PENALTY              float  default 3.0
        MIN_FRAGMENT_FRAMES              int    default 5
        PELT_MODEL                       str    default "rbf" (l1 / l2 / rbf)
        FRAGMENT_CNN_WEIGHT              float  default 0.40
        FRAGMENT_TAG_WEIGHT              float  default 0.15
        FRAGMENT_SPATIAL_WEIGHT          float  default 0.35
        ONLINE_PRIOR_WEIGHT              float  default 0.25
        ASSIGNMENT_MARGIN_THRESHOLD      float  default 0.10
        MAX_VELOCITY_BREAK               float  default 50.0
        FRAGMENT_SOLVER_ILP_TIME_LIMIT   float  default 30.0
        ENABLE_FRAGMENT_SCORING          bool   default True
    """
    params = params or {}

    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df if trajectories_df is not None else pd.DataFrame()

    known_labels = list(catalog.labels[1:])
    if not known_labels:
        return trajectories_df

    enable_scoring = bool(params.get("ENABLE_FRAGMENT_SCORING", True))

    changepoints = detect_identity_changepoints(trajectories_df, catalog, params)
    fragments_df = build_fragments(trajectories_df, changepoints, catalog, params)

    if fragments_df.empty:
        log.debug("fragment_solver: no fragments built; returning unchanged.")
        return trajectories_df

    log.info(
        "fragment_solver: %d fragments across %d trajectories; %d changepoints detected.",
        len(fragments_df),
        int(fragments_df["TrajectoryID"].nunique()),
        sum(len(v) for v in changepoints.values()),
    )

    if not enable_scoring:
        log.debug("fragment_solver: scoring disabled; labels not updated.")
        return trajectories_df

    fragments_df = solve_global_assignment(fragments_df, catalog, params)
    result = apply_fragment_labels(trajectories_df, fragments_df, catalog)

    return result
