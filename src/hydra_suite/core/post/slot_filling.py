"""Vacancy-aware pre-offline identity slot filling.

Runs after augmentation (CNN_*_Prob columns present, DetectedTagID set),
before the offline HMM decoder. The online decoder has already committed
identities to confident tracks; this pass resolves the remaining wholly-unknown
trajectories by asking: which identity is spatially and evidentially consistent
with this track, given the vacancies in this time window?
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from hydra_suite.core.identity.catalog import IdentityCatalog

log = logging.getLogger(__name__)

_LABEL_COL = "IdentityAssignedLabel"
_CONF_COL = "IdentityAssignedConfidence"
_UNKNOWN_VALUES = frozenset({"", "unknown"})
_SLOT_FILL_SOURCE = "slot_fill"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _is_unknown_label(series: pd.Series) -> pd.Series:
    """Return boolean mask: True where value is NaN, empty string, or 'unknown'."""
    return series.isna() | series.astype(str).str.strip().isin(_UNKNOWN_VALUES)


def _build_claimed_schedule(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Build {label -> [segment_dicts]} from trajectories with a real label.

    A trajectory contributes to a label's schedule when its dominant
    ``IdentityAssignedLabel`` is a non-unknown value.  One segment is emitted
    per (TrajectoryID, label) combination.  Each segment dict contains:
    ``traj_id, start_frame, end_frame, start_X, start_Y, end_X, end_Y``.
    """
    schedule: dict[str, list[dict]] = {}

    if _LABEL_COL not in df.columns:
        return schedule

    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        label_col = grp[_LABEL_COL]
        unknown_mask = _is_unknown_label(label_col)
        known_rows = grp[~unknown_mask]
        if known_rows.empty:
            continue

        # Use the dominant label for this trajectory.
        dominant = known_rows[_LABEL_COL].astype(str).mode()
        if dominant.empty:
            continue
        label = str(dominant.iloc[0])
        if label in _UNKNOWN_VALUES:
            continue

        start_frame = int(grp["FrameID"].min())
        end_frame = int(grp["FrameID"].max())

        # Use last/first valid X/Y positions.
        valid_xy = grp[grp["X"].notna() & grp["Y"].notna()].sort_values("FrameID")
        if not valid_xy.empty:
            start_X = float(valid_xy.iloc[0]["X"])
            start_Y = float(valid_xy.iloc[0]["Y"])
            end_X = float(valid_xy.iloc[-1]["X"])
            end_Y = float(valid_xy.iloc[-1]["Y"])
        else:
            start_X = start_Y = end_X = end_Y = math.nan

        seg = {
            "traj_id": traj_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_X": start_X,
            "start_Y": start_Y,
            "end_X": end_X,
            "end_Y": end_Y,
        }
        schedule.setdefault(label, []).append(seg)

    # Sort each label's list by start_frame.
    for label in schedule:
        schedule[label].sort(key=lambda s: s["start_frame"])

    return schedule


def _find_unassigned_segments(df: pd.DataFrame) -> list[dict]:
    """Return one entry per wholly-unknown trajectory.

    A trajectory is unassigned if ``_is_unknown_label`` is True for **all**
    rows.  Each entry is a dict with keys:
    ``traj_id, start_frame, end_frame, start_X, start_Y, end_X, end_Y, rows``.
    """
    segments: list[dict] = []

    if _LABEL_COL not in df.columns:
        # Treat every trajectory as unassigned when the column is absent.
        for traj_id, grp in df.groupby("TrajectoryID", sort=False):
            _append_unassigned(segments, traj_id, grp)
        return segments

    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        if not _is_unknown_label(grp[_LABEL_COL]).all():
            continue
        _append_unassigned(segments, traj_id, grp)

    return segments


def _append_unassigned(segments: list[dict], traj_id: Any, grp: pd.DataFrame) -> None:
    """Append an unassigned-segment dict for *traj_id* / *grp* to *segments*."""
    start_frame = int(grp["FrameID"].min())
    end_frame = int(grp["FrameID"].max())

    valid_xy = (
        grp[grp["X"].notna() & grp["Y"].notna()].sort_values("FrameID")
        if "X" in grp.columns and "Y" in grp.columns
        else pd.DataFrame()
    )
    if not valid_xy.empty:
        start_X = float(valid_xy.iloc[0]["X"])
        start_Y = float(valid_xy.iloc[0]["Y"])
        end_X = float(valid_xy.iloc[-1]["X"])
        end_Y = float(valid_xy.iloc[-1]["Y"])
    else:
        start_X = start_Y = end_X = end_Y = math.nan

    segments.append(
        {
            "traj_id": traj_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_X": start_X,
            "start_Y": start_Y,
            "end_X": end_X,
            "end_Y": end_Y,
            "rows": grp.sort_values("FrameID"),
        }
    )


def _vacancies(
    t0: int,
    t1: int,
    schedule: dict[str, list[dict]],
    known_labels: list[str],
) -> list[str]:
    """Return labels with no claimed segment overlapping ``[t0, t1]``."""
    vacant = []
    for label in known_labels:
        segs = schedule.get(label, [])
        occupied = any(s["start_frame"] <= t1 and s["end_frame"] >= t0 for s in segs)
        if not occupied:
            vacant.append(label)
    return vacant


def _prob_col_for_label(df: pd.DataFrame, label: str) -> str | None:
    """Find column ending in ``_{label}_Prob``.  Returns first match or None."""
    suffix = f"_{label}_Prob"
    for col in df.columns:
        if str(col).endswith(suffix):
            return col
    return None


def _cnn_evidence_score(
    rows: pd.DataFrame, identity: str, full_df: pd.DataFrame
) -> float:
    """Return CNN evidence score for *identity* over *rows*.

    1. Try ``*_{identity}_Prob`` column — return nanmean of that column.
    2. Fallback: scan ``*_Class`` columns; collect mean ``*_Conf`` for rows
       where the class matches *identity*; return nanmean of those means.
    3. Return 0.0 if no evidence found.
    """
    # Strategy 1: probability vector column.
    prob_col = _prob_col_for_label(full_df, identity)
    if prob_col is not None and prob_col in rows.columns:
        vals = pd.to_numeric(rows[prob_col], errors="coerce")
        if vals.notna().any():
            return float(np.nanmean(vals.values))

    # Strategy 2: class/confidence columns.
    class_cols = [c for c in rows.columns if str(c).endswith("_Class")]
    per_source_means: list[float] = []
    for class_col in class_cols:
        prefix = str(class_col)[: -len("_Class")]
        conf_col = f"{prefix}_Conf"
        if conf_col not in rows.columns:
            continue
        match_mask = rows[class_col].astype(str) == identity
        if not match_mask.any():
            continue
        conf_vals = pd.to_numeric(rows.loc[match_mask, conf_col], errors="coerce")
        if conf_vals.notna().any():
            per_source_means.append(float(np.nanmean(conf_vals.values)))

    if per_source_means:
        return float(np.nanmean(per_source_means))

    return 0.0


def _spatial_score(
    segment: dict,
    identity: str,
    schedule: dict[str, list[dict]],
    max_velocity: float,
) -> float:
    """Gaussian falloff spatial score using prior and following claimed segments.

    ``sigma = max_velocity * max(1, gap_frames)``

    Returns mean of available terms; 0.5 if no spatial anchor is available.
    """
    t0 = segment["start_frame"]
    t1 = segment["end_frame"]
    x0 = segment["start_X"]
    y0 = segment["start_Y"]
    x1 = segment["end_X"]
    y1 = segment["end_Y"]

    segs = schedule.get(identity, [])
    term_scores: list[float] = []

    # Prior: most recent segment ending before t0.
    prior_candidates = [s for s in segs if s["end_frame"] < t0]
    if prior_candidates:
        prior = max(prior_candidates, key=lambda s: s["end_frame"])
        prior_ex = prior["end_X"]
        prior_ey = prior["end_Y"]
        if (
            math.isfinite(x0)
            and math.isfinite(y0)
            and math.isfinite(prior_ex)
            and math.isfinite(prior_ey)
        ):
            gap = max(1, t0 - prior["end_frame"])
            sigma = max_velocity * gap
            dist = math.hypot(x0 - prior_ex, y0 - prior_ey)
            score = math.exp(-(dist**2) / (2.0 * sigma**2))
            term_scores.append(score)

    # Following: earliest segment starting after t1.
    following_candidates = [s for s in segs if s["start_frame"] > t1]
    if following_candidates:
        following = min(following_candidates, key=lambda s: s["start_frame"])
        fol_sx = following["start_X"]
        fol_sy = following["start_Y"]
        if (
            math.isfinite(x1)
            and math.isfinite(y1)
            and math.isfinite(fol_sx)
            and math.isfinite(fol_sy)
        ):
            gap = max(1, following["start_frame"] - t1)
            sigma = max_velocity * gap
            dist = math.hypot(x1 - fol_sx, y1 - fol_sy)
            score = math.exp(-(dist**2) / (2.0 * sigma**2))
            term_scores.append(score)

    if term_scores:
        return float(np.mean(term_scores))
    return 0.5  # uninformative


def _tag_override(
    rows: pd.DataFrame, identity: str, tag_labels: list[str]
) -> float | None:
    """Return tag-based override score or None if no tag information available.

    Returns:
        1.0  — identity is the only label mapped from observed tags.
        0.7  — identity is one of several mapped labels.
        0.0  — tags present but none map to identity.
        None — no DetectedTagID column or all values are NaN.
    """
    if "DetectedTagID" not in rows.columns:
        return None
    tag_col = pd.to_numeric(rows["DetectedTagID"], errors="coerce")
    valid_tags = tag_col.dropna()
    if valid_tags.empty:
        return None

    if not tag_labels:
        return None

    mapped_labels: set[str] = set()
    for raw_tag in valid_tags:
        idx = int(raw_tag)
        if 0 <= idx < len(tag_labels):
            lbl = tag_labels[idx]
            if lbl and lbl not in _UNKNOWN_VALUES:
                mapped_labels.add(lbl)

    if not mapped_labels:
        return 0.0  # tags present but none mapped to a known label — exclude

    if identity not in mapped_labels:
        return 0.0

    if len(mapped_labels) == 1:
        return 1.0

    return 0.7


def _segments_overlap(a: dict, b: dict) -> bool:
    """True when segments *a* and *b* share at least one frame."""
    return a["start_frame"] <= b["end_frame"] and b["start_frame"] <= a["end_frame"]


def _score_matrix(
    segments: list[dict],
    schedule: dict[str, list[dict]],
    known_labels: list[str],
    full_df: pd.DataFrame,
    params: dict[str, Any],
) -> np.ndarray:
    """Return shape ``(n_segments, n_labels)`` score matrix.

    -1.0 means ineligible (not a vacancy or hard tag conflict).
    """
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    cnn_w = float(params.get("SLOT_FILL_CNN_WEIGHT", 0.55))
    spatial_w = float(params.get("SLOT_FILL_SPATIAL_WEIGHT", 0.45))
    tag_labels: list[str] = list(params.get("TAG_IDENTITY_LABELS", []) or [])

    n_seg = len(segments)
    n_lbl = len(known_labels)
    mat = np.full((n_seg, n_lbl), -1.0, dtype=np.float64)

    for i, seg in enumerate(segments):
        t0 = seg["start_frame"]
        t1 = seg["end_frame"]
        vacant = set(_vacancies(t0, t1, schedule, known_labels))

        for j, label in enumerate(known_labels):
            # Gate 1: must be a vacancy.
            if label not in vacant:
                continue

            # Gate 2: tag evidence must not flatly exclude this identity.
            rows = seg["rows"]
            tag_ev = _tag_override(rows, label, tag_labels)
            if tag_ev is not None and tag_ev == 0.0:
                continue  # hard tag conflict

            cnn_s = _cnn_evidence_score(rows, label, full_df)
            spatial_s = _spatial_score(seg, label, schedule, max_vel)

            if tag_ev is not None:
                combined = 0.5 * tag_ev + 0.3 * cnn_s + 0.2 * spatial_s
            else:
                combined = cnn_w * cnn_s + spatial_w * spatial_s

            mat[i, j] = combined

    return mat


# ---------------------------------------------------------------------------
# Assignment solvers
# ---------------------------------------------------------------------------


def _solve_hungarian(
    score_matrix: np.ndarray,
    known_labels: list[str],
    min_score: float,
) -> dict[int, str | None]:
    """Assign segments to labels using the Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment

    n = score_matrix.shape[0]
    result: dict[int, str | None] = {i: None for i in range(n)}

    # Rows that have at least one eligible label.
    eligible_rows = [i for i in range(n) if np.any(score_matrix[i] >= 0)]
    if not eligible_rows:
        return result

    sub = score_matrix[np.ix_(eligible_rows, range(score_matrix.shape[1]))]
    # Convert to cost matrix (minimise negative score); ineligible cells get large cost.
    cost = np.where(sub < 0, 1e6, -sub)

    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        orig_row = eligible_rows[r]
        score = score_matrix[orig_row, c]
        if score >= min_score:
            result[orig_row] = known_labels[c]

    return result


def _solve_milp(
    segments: list[dict],
    known_labels: list[str],
    score_matrix: np.ndarray,
    min_score: float,
) -> dict[int, str | None]:
    """Assign segments to labels using MILP (handles overlapping segments).

    Falls back to :func:`_solve_hungarian` if scipy MILP is unavailable.
    """
    try:
        from scipy.optimize import LinearConstraint, milp  # type: ignore[attr-defined]
    except ImportError:
        log.debug("scipy MILP not available; falling back to Hungarian assignment.")
        return _solve_hungarian(score_matrix, known_labels, min_score)

    n_seg = len(segments)
    n_lbl = len(known_labels)

    # Enumerate valid (i, j) pairs.
    pairs: list[tuple[int, int]] = [
        (i, j) for i in range(n_seg) for j in range(n_lbl) if score_matrix[i, j] >= 0
    ]

    if not pairs:
        return {i: None for i in range(n_seg)}

    n_vars = len(pairs)
    pair_index: dict[tuple[int, int], int] = {p: k for k, p in enumerate(pairs)}

    # Objective: minimise negative score (i.e. maximise score).
    c = np.array([-score_matrix[i, j] for i, j in pairs], dtype=np.float64)

    # Variable bounds: [0, 1] (continuous relaxation is fine for these constraints,
    # but we request integer via integrality).
    from scipy.optimize import Bounds  # type: ignore[attr-defined]

    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars, dtype=np.int8)

    # Build constraint matrix rows.
    A_rows: list[np.ndarray] = []
    lb_list: list[float] = []
    ub_list: list[float] = []

    # C1: Each segment assigned at most 1 label.
    for i in range(n_seg):
        seg_pair_indices = [
            pair_index[(i, j)] for j in range(n_lbl) if (i, j) in pair_index
        ]
        if not seg_pair_indices:
            continue
        row = np.zeros(n_vars)
        row[seg_pair_indices] = 1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(1.0)

    # C2: Each label assigned at most 1 segment (per non-overlapping assumption for
    #     overlapping segments we add pairwise exclusion).
    # First, a global "at most 1 per label" constraint.
    for j in range(n_lbl):
        lbl_pair_indices = [
            pair_index[(i, j)] for i in range(n_seg) if (i, j) in pair_index
        ]
        if len(lbl_pair_indices) < 2:
            continue
        row = np.zeros(n_vars)
        row[lbl_pair_indices] = 1.0
        A_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(1.0)

    # C3: Pairwise exclusion for overlapping segments sharing a label.
    for a_idx, b_idx in combinations(range(n_seg), 2):
        if not _segments_overlap(segments[a_idx], segments[b_idx]):
            continue
        for j in range(n_lbl):
            k_a = pair_index.get((a_idx, j))
            k_b = pair_index.get((b_idx, j))
            if k_a is None or k_b is None:
                continue
            row = np.zeros(n_vars)
            row[k_a] = 1.0
            row[k_b] = 1.0
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1.0)

    result: dict[int, str | None] = {i: None for i in range(n_seg)}

    if not A_rows:
        # No constraints needed; just pick the best score per segment.
        return _solve_hungarian(score_matrix, known_labels, min_score)

    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lb=np.array(lb_list), ub=np.array(ub_list))

    try:
        opt = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if opt.success:
            x = opt.x
            for k, (i, j) in enumerate(pairs):
                if x[k] > 0.5:
                    score = score_matrix[i, j]
                    if score >= min_score:
                        result[i] = known_labels[j]
        else:
            log.debug(
                "MILP solver did not find an optimal solution; falling back to Hungarian."
            )
            return _solve_hungarian(score_matrix, known_labels, min_score)
    except Exception as exc:
        log.warning("MILP solve failed (%s); falling back to Hungarian.", exc)
        return _solve_hungarian(score_matrix, known_labels, min_score)

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_vacancy_aware_slot_filling(
    trajectories_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Pre-offline vacancy-aware identity slot filling.

    Resolves wholly-unknown trajectories by identifying which identity is
    vacant during each segment's time window and scoring candidates on CNN
    evidence plus spatial continuity.

    Parameters
    ----------
    trajectories_df:
        Full trajectories dataframe (post-augmentation).
    catalog:
        Identity catalog for the run; ``catalog.labels[1:]`` are the known
        identities.
    params:
        Optional parameter overrides.  Recognised keys:

        - ``MAX_VELOCITY_BREAK`` (float, default 50.0)
        - ``SLOT_FILL_CNN_WEIGHT`` (float, default 0.55)
        - ``SLOT_FILL_SPATIAL_WEIGHT`` (float, default 0.45)
        - ``SLOT_FILL_MIN_SCORE`` (float, default 0.1)
        - ``TAG_IDENTITY_LABELS`` (list[str], default [])

    Returns
    -------
    pd.DataFrame
        Modified copy of *trajectories_df* with ``IdentityAssignedLabel`` and
        ``IdentityAssignedConfidence`` updated for resolved segments.
    """
    params = params or {}

    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df

    known_labels: list[str] = list(catalog.labels[1:])
    if not known_labels:
        return trajectories_df

    df = trajectories_df.copy()

    # Ensure required columns exist.
    if _LABEL_COL not in df.columns:
        df[_LABEL_COL] = np.nan
    if _CONF_COL not in df.columns:
        df[_CONF_COL] = np.nan

    schedule = _build_claimed_schedule(df)
    segments = _find_unassigned_segments(df)

    if not segments:
        log.debug("slot_filling: no unassigned segments found.")
        return df

    min_score = float(params.get("SLOT_FILL_MIN_SCORE", 0.1))

    scores = _score_matrix(segments, schedule, known_labels, df, params)

    has_overlap = any(_segments_overlap(a, b) for a, b in combinations(segments, 2))

    if has_overlap:
        assignment = _solve_milp(segments, known_labels, scores, min_score)
    else:
        assignment = _solve_hungarian(scores, known_labels, min_score)

    n_resolved = 0
    for seg_idx, label in assignment.items():
        if label is None:
            continue

        seg = segments[seg_idx]
        traj_id = seg["traj_id"]
        mask = df["TrajectoryID"] == traj_id

        # Retrieve the winning score.
        lbl_idx = known_labels.index(label)
        winning_score = float(scores[seg_idx, lbl_idx])

        df.loc[mask, _LABEL_COL] = label
        df.loc[mask, _CONF_COL] = winning_score

        # Append slot_fill source to IdentityEvidenceSources if present.
        if "IdentityEvidenceSources" in df.columns:
            current = df.loc[mask, "IdentityEvidenceSources"]
            df.loc[mask, "IdentityEvidenceSources"] = current.apply(
                lambda v: (
                    f"{v},{_SLOT_FILL_SOURCE}"
                    if (isinstance(v, str) and v.strip())
                    else _SLOT_FILL_SOURCE
                )
            )

        n_resolved += 1
        log.debug(
            "slot_filling: traj %s -> '%s' (score=%.3f)",
            traj_id,
            label,
            winning_score,
        )

    log.info(
        "slot_filling: resolved %d / %d unassigned segments.",
        n_resolved,
        len(segments),
    )

    return df
