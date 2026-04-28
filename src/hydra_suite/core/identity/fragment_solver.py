"""Global identity fragment solver.

Current identity post-processing pipeline:
1. PELT changepoint detection on per-trajectory CNN probability matrices.
2. Fragment building from detected changepoints.
3. Global MILP assignment: maximises spatial continuity + CNN/tag evidence
   with a confidence-weighted online-label prior and a margin threshold.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from hydra_suite.core.identity.catalog import IdentityCatalog

log = logging.getLogger(__name__)

_LABEL_COL = "IdentityAssignedLabel"
_CONF_COL = "IdentityAssignedConfidence"
_UNKNOWN_VALUES = frozenset({"", "unknown"})
_CNN_CLASS_SUFFIX = "_Class"
_CNN_PROB_SUFFIX = "_Prob"


def _sanitize_probability_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return token


def _build_cnn_probability_prefix_map(columns: pd.Index) -> dict[str, dict[str, str]]:
    """Return ``prefix -> class_token -> prob_column`` for exported CNN columns.

    Exported wide CSVs always include ``CNN_*_Class`` columns that identify the
    phase/factor prefix. Probability columns share that prefix and append the
    sanitized class token before ``_Prob``.
    """
    prefixes = sorted(
        {
            str(col)[: -len(_CNN_CLASS_SUFFIX)]
            for col in columns
            if str(col).startswith("CNN_") and str(col).endswith(_CNN_CLASS_SUFFIX)
        },
        key=len,
        reverse=True,
    )
    if not prefixes:
        return {}

    result: dict[str, dict[str, str]] = {}
    for col in columns:
        token = str(col)
        if not token.startswith("CNN_") or not token.endswith(_CNN_PROB_SUFFIX):
            continue
        for prefix in prefixes:
            prefix_token = f"{prefix}_"
            if not token.startswith(prefix_token):
                continue
            class_token = token[len(prefix_token) : -len(_CNN_PROB_SUFFIX)]
            if class_token:
                result.setdefault(prefix, {})[class_token] = token
            break
    return result


def _build_cnn_label_specs(
    known_labels: list[str],
    prefix_prob_cols: dict[str, dict[str, str]],
    columns: pd.Index,
) -> dict[str, dict[str, Any]]:
    """Precompute how each downstream label is reconstructed from CNN columns."""
    specs: dict[str, dict[str, Any]] = {}
    for label in known_labels:
        label_token = _sanitize_probability_token(label)
        if not label_token:
            specs[label] = {"direct_cols": [], "part_cols": {}, "parts": set()}
            continue

        direct_cols = [
            str(col)
            for col in columns
            if str(col).startswith("CNN_")
            and str(col).endswith(f"_{label_token}{_CNN_PROB_SUFFIX}")
        ]
        part_cols: dict[str, list[str]] = {}
        parts = tuple(part for part in label_token.split("_") if part)
        for class_token_map in prefix_prob_cols.values():
            direct_col = class_token_map.get(label_token)
            if direct_col is not None and direct_col not in direct_cols:
                direct_cols.append(direct_col)
                continue

            matched_parts = [part for part in parts if part in class_token_map]
            if len(matched_parts) == 1:
                part = matched_parts[0]
                part_cols.setdefault(part, []).append(class_token_map[part])

        specs[label] = {
            "direct_cols": direct_cols,
            "part_cols": part_cols,
            "parts": set(parts),
        }
    return specs


def _trajectory_mean_cnn_probs(
    grp_sorted: pd.DataFrame,
    known_labels: list[str],
    label_specs: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Return mean downstream-label probabilities reconstructed from CNN exports."""
    mean_probs: dict[str, float] = {}
    for label in known_labels:
        spec = label_specs.get(label, {})
        direct_cols = [col for col in spec.get("direct_cols", []) if col in grp_sorted]
        part_cols = {
            part: [col for col in cols if col in grp_sorted]
            for part, cols in (spec.get("part_cols", {}) or {}).items()
        }
        parts = set(spec.get("parts", set()))

        direct_vals: list[np.ndarray] = []
        for col in direct_cols:
            vals = pd.to_numeric(grp_sorted[col], errors="coerce").to_numpy(
                dtype=np.float64
            )
            direct_vals.append(np.clip(vals, 1e-9, 1.0))

        part_vals: list[np.ndarray] = []
        part_mask = np.ones(len(grp_sorted), dtype=bool)
        use_parts = bool(parts)
        if use_parts:
            for part in parts:
                cols = part_cols.get(part, [])
                if not cols:
                    use_parts = False
                    break
                part_present = np.zeros(len(grp_sorted), dtype=bool)
                for col in cols:
                    vals = pd.to_numeric(grp_sorted[col], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    part_present |= np.isfinite(vals)
                    part_vals.append(np.clip(vals, 1e-9, 1.0))
                part_mask &= part_present

        row_prob = np.full(len(grp_sorted), np.nan, dtype=np.float64)
        any_used = False
        if direct_vals:
            any_used = True
            direct_stack = np.stack(direct_vals, axis=0)
            direct_mask = np.all(np.isfinite(direct_stack), axis=0)
            if np.any(direct_mask):
                row_prob[direct_mask] = np.prod(direct_stack[:, direct_mask], axis=0)

        if use_parts and part_vals:
            any_used = True
            part_stack = np.stack(part_vals, axis=0)
            valid_part_mask = part_mask & np.all(np.isfinite(part_stack), axis=0)
            if np.any(valid_part_mask):
                part_prod = np.full(len(grp_sorted), np.nan, dtype=np.float64)
                part_prod[valid_part_mask] = np.prod(
                    part_stack[:, valid_part_mask], axis=0
                )
                missing_mask = valid_part_mask & np.isnan(row_prob)
                overlap_mask = valid_part_mask & np.isfinite(row_prob)
                if np.any(missing_mask):
                    row_prob[missing_mask] = part_prod[missing_mask]
                if np.any(overlap_mask):
                    row_prob[overlap_mask] *= part_prod[overlap_mask]

        if any_used and np.isfinite(row_prob).any():
            mean_probs[label] = float(np.nanmean(row_prob))

    return mean_probs


def _trajectory_cnn_log_evidence(
    grp_sorted: pd.DataFrame,
    known_labels: list[str],
    label_specs: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Return mean log-evidence per downstream label from exported CNN columns.

    Rows without CNN evidence are treated as neutral for that row so sparse
    detections do not dominate long fragments simply because only observed rows
    contribute to the average.
    """
    log_scores: dict[str, float] = {}
    n_rows = len(grp_sorted)
    if n_rows == 0:
        return log_scores

    for label in known_labels:
        spec = label_specs.get(label, {})
        direct_cols = [col for col in spec.get("direct_cols", []) if col in grp_sorted]
        part_cols = {
            part: [col for col in cols if col in grp_sorted]
            for part, cols in (spec.get("part_cols", {}) or {}).items()
        }
        parts = set(spec.get("parts", set()))

        direct_vals: list[np.ndarray] = []
        for col in direct_cols:
            vals = pd.to_numeric(grp_sorted[col], errors="coerce").to_numpy(
                dtype=np.float64
            )
            direct_vals.append(np.clip(vals, 1e-9, 1.0))

        part_vals: list[np.ndarray] = []
        part_mask = np.ones(n_rows, dtype=bool)
        use_parts = bool(parts)
        if use_parts:
            for part in parts:
                cols = part_cols.get(part, [])
                if not cols:
                    use_parts = False
                    break
                part_present = np.zeros(n_rows, dtype=bool)
                for col in cols:
                    vals = pd.to_numeric(grp_sorted[col], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    part_present |= np.isfinite(vals)
                    part_vals.append(np.clip(vals, 1e-9, 1.0))
                part_mask &= part_present

        row_prob = np.full(n_rows, np.nan, dtype=np.float64)
        any_used = False
        if direct_vals:
            any_used = True
            direct_stack = np.stack(direct_vals, axis=0)
            direct_mask = np.all(np.isfinite(direct_stack), axis=0)
            if np.any(direct_mask):
                row_prob[direct_mask] = np.prod(direct_stack[:, direct_mask], axis=0)

        if use_parts and part_vals:
            any_used = True
            part_stack = np.stack(part_vals, axis=0)
            valid_part_mask = part_mask & np.all(np.isfinite(part_stack), axis=0)
            if np.any(valid_part_mask):
                part_prod = np.full(n_rows, np.nan, dtype=np.float64)
                part_prod[valid_part_mask] = np.prod(
                    part_stack[:, valid_part_mask], axis=0
                )
                missing_mask = valid_part_mask & np.isnan(row_prob)
                overlap_mask = valid_part_mask & np.isfinite(row_prob)
                if np.any(missing_mask):
                    row_prob[missing_mask] = part_prod[missing_mask]
                if np.any(overlap_mask):
                    row_prob[overlap_mask] *= part_prod[overlap_mask]

        if any_used:
            row_log = np.zeros(n_rows, dtype=np.float64)
            finite_mask = np.isfinite(row_prob)
            if np.any(finite_mask):
                row_log[finite_mask] = np.log(
                    np.clip(row_prob[finite_mask], 1e-12, 1.0)
                )
                log_scores[label] = float(np.mean(row_log))

    return log_scores


def _trajectory_tag_evidence(
    grp_sorted: pd.DataFrame,
    known_labels: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return mean tag probabilities and mean tag log-evidence per label."""
    mean_probs: dict[str, float] = {}
    log_scores: dict[str, float] = {}
    n_rows = len(grp_sorted)
    if n_rows == 0 or "DetectedTagLabel" not in grp_sorted.columns:
        return mean_probs, log_scores

    tag_vals = grp_sorted["DetectedTagLabel"].astype(object)
    tag_labels = tag_vals.where(tag_vals.notna(), np.nan).astype(str).str.strip()
    detected_mask = (~tag_vals.isna()) & (~tag_labels.isin(_UNKNOWN_VALUES))
    if not detected_mask.any():
        return mean_probs, log_scores

    if "DetectedTagConf" in grp_sorted.columns:
        conf_vals = pd.to_numeric(
            grp_sorted["DetectedTagConf"], errors="coerce"
        ).to_numpy(dtype=np.float64)
    elif "DetectedTagHamming" in grp_sorted.columns:
        hammings = pd.to_numeric(
            grp_sorted["DetectedTagHamming"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        conf_vals = 1.0 / (1.0 + np.clip(hammings, 0.0, None))
    else:
        conf_vals = np.ones(n_rows, dtype=np.float64)

    conf_vals = np.clip(conf_vals, 1e-4, 1.0 - 1e-4)
    detected_mask_arr = detected_mask.to_numpy(dtype=bool)
    tag_labels_arr = tag_labels.to_numpy(dtype=object)
    n_known = len(known_labels)

    for label in known_labels:
        row_prob = np.full(n_rows, np.nan, dtype=np.float64)
        match_mask = detected_mask_arr & (tag_labels_arr == label)
        other_mask = detected_mask_arr & (tag_labels_arr != label)
        if np.any(match_mask):
            row_prob[match_mask] = conf_vals[match_mask]
        if np.any(other_mask):
            if n_known > 1:
                row_prob[other_mask] = (1.0 - conf_vals[other_mask]) / (n_known - 1)
            else:
                row_prob[other_mask] = 1e-4
        finite_mask = np.isfinite(row_prob)
        if np.any(finite_mask):
            mean_probs[label] = float(np.nanmean(row_prob))
            row_log = np.zeros(n_rows, dtype=np.float64)
            row_log[finite_mask] = np.log(np.clip(row_prob[finite_mask], 1e-12, 1.0))
            log_scores[label] = float(np.mean(row_log))

    return mean_probs, log_scores


def _normalize_support_scores(
    known_labels: list[str],
    log_scores: dict[str, float],
) -> dict[str, float]:
    """Convert log-supports into a normalized per-label score distribution."""
    raw = np.array(
        [
            math.exp(float(log_scores[label])) if label in log_scores else 0.0
            for label in known_labels
        ],
        dtype=np.float64,
    )
    raw[~np.isfinite(raw)] = 0.0
    total = float(raw.sum())
    if total <= 1e-12:
        if not known_labels:
            return {}
        uniform = 1.0 / len(known_labels)
        return {label: uniform for label in known_labels}
    raw /= total
    return {label: float(raw[idx]) for idx, label in enumerate(known_labels)}


def _build_prior_log_scores(
    known_labels: list[str],
    online_label: str,
    online_confidence: float,
) -> dict[str, float]:
    """Build a soft prior over labels from the online label/confidence pair."""
    if online_label not in known_labels or not np.isfinite(online_confidence):
        return {label: 0.0 for label in known_labels}

    conf = float(np.clip(online_confidence, 1e-4, 1.0 - 1e-4))
    n_labels = len(known_labels)
    if n_labels <= 1:
        return {known_labels[0]: math.log(conf)} if known_labels else {}
    other = max((1.0 - conf) / (n_labels - 1), 1e-6)
    return {
        label: math.log(conf if label == online_label else other)
        for label in known_labels
    }


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


def _fragments_overlap(a: pd.Series, b: pd.Series) -> bool:
    return int(a["StartFrame"]) <= int(b["EndFrame"]) and int(b["StartFrame"]) <= int(
        a["EndFrame"]
    )


def _spatial_score_for_fragment(
    frag: pd.Series,
    identity: str,
    schedule: dict[str, list[dict]],
    max_velocity: float,
    no_neighbor_score: float = 0.3,
) -> tuple[float, bool]:
    """Velocity-based spatial continuity score against nearest neighboring segment.

    Returns (score, has_neighbors).

    Scoring:
    - Implied velocity = dist / gap (pixels per frame).
    - Hard veto: if velocity > max_velocity for any neighbor, return (0.0, True)
      immediately — physically impossible jump, caller should mark ineligible.
    - Otherwise: score = exp(-2 * (velocity / max_velocity)^2).
      This is a velocity-space Gaussian where max_velocity is one sigma; scores
      range from ~1.0 (stationary) through ~0.61 (half max) to ~0.14 (at max).
    - When has_neighbors is False the spatial score cannot be trusted;
      the caller should use evidence-only scoring.
    """
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
        dist = math.hypot(x0 - prior["end_X"], y0 - prior["end_Y"])
        velocity = dist / gap
        if velocity > max_velocity:
            return 0.0, True  # physically impossible — hard veto
        term_scores.append(math.exp(-2.0 * (velocity / max_velocity) ** 2))

    following = min(
        (s for s in segs if s["start_frame"] > t1),
        key=lambda s: s["start_frame"],
        default=None,
    )
    if following and all(
        math.isfinite(v) for v in [x1, y1, following["start_X"], following["start_Y"]]
    ):
        gap = max(1, following["start_frame"] - t1)
        dist = math.hypot(x1 - following["start_X"], y1 - following["start_Y"])
        velocity = dist / gap
        if velocity > max_velocity:
            return 0.0, True  # physically impossible — hard veto
        term_scores.append(math.exp(-2.0 * (velocity / max_velocity) ** 2))

    if term_scores:
        return float(np.mean(term_scores)), True
    return no_neighbor_score, False


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
    """Build (n_frags x n_labels) score matrix.  -1 means ineligible.

    Scoring strategy:
        - Reconstruct per-fragment per-label unary evidence from all CNN phases and
            tag detections as weighted average log-evidence, then normalize that into a
            candidate score distribution.
        - Fuse the online label/confidence as a soft prior in the same log domain.
        - When temporal neighbors exist (has_neighbors=True):
                score = unary_score × spatial_s × length_factor
        - When no neighbors exist (has_neighbors=False):
                score = unary_score × length_factor

    Spatial veto: if has_neighbors and spatial_s < FRAGMENT_SPATIAL_VETO_THRESHOLD
    (default 0.05), the cell is marked -1 (ineligible). Combined with the hard
    velocity veto in _spatial_score_for_fragment (any neighbor implying
    velocity > MAX_VELOCITY_BREAK returns score=0.0), this guarantees that
    physically impossible jumps are never assigned.

    Length weighting: short fragments are discounted relative to long ones so a
    tiny high-confidence fragment cannot outbid a large spatially-consistent track.
    """
    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    tag_w = float(params.get("FRAGMENT_TAG_WEIGHT", 0.15))
    # FRAGMENT_LENGTH_WEIGHT: multiplicative blend coefficient in [0, 1].
    # At 0.0, length has no effect. At 1.0, the shortest fragment scores 0.
    # Default 0.60 is chosen so that a fragment with a good online prior (conf≥0.75)
    # survives length discounting while a tiny (≤5 frame) confident-but-isolated
    # fragment cannot displace it through CNN/tag strength alone.
    length_w = min(1.0, max(0.0, float(params.get("FRAGMENT_LENGTH_WEIGHT", 0.60))))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    no_neighbor_score = float(params.get("SPATIAL_NO_NEIGHBOR_SCORE", 0.3))
    # Score below this threshold (when neighbors exist) → fragment is spatially
    # incompatible with the identity and is marked ineligible.
    spatial_veto = float(params.get("FRAGMENT_SPATIAL_VETO_THRESHOLD", 0.05))
    n_frags = len(frags)
    n_labels = len(known_labels)
    score_mat = np.full((n_frags, n_labels), -1.0, dtype=np.float64)

    # Pre-compute log-normalised durations for the multiplicative length factor.
    durations = [
        max(1, int(r["EndFrame"]) - int(r["StartFrame"]) + 1)
        for _, r in frags.iterrows()
    ]
    max_duration = max(durations) if durations else 1
    log_max = math.log1p(max_duration)

    combined_supports: list[dict[str, float]] = []
    source_supports: list[dict[str, float]] = []

    for _, frag_row in frags.iterrows():
        cnn_log_scores = frag_row.get("CNNLogEvidence") or {}
        tag_log_scores = frag_row.get("TagLogEvidence") or {}
        online_lbl = str(frag_row["OnlineLabel"])
        online_conf = float(frag_row["OnlineConfidence"])
        prior_log_scores = _build_prior_log_scores(
            known_labels, online_lbl, online_conf
        )

        source_log_scores = {
            label: (
                cnn_w * float(cnn_log_scores.get(label, 0.0))
                + tag_w * float(tag_log_scores.get(label, 0.0))
            )
            for label in known_labels
        }
        combined_log_scores = {
            label: source_log_scores[label] + prior_w * float(prior_log_scores[label])
            for label in known_labels
        }

        source_supports.append(
            _normalize_support_scores(known_labels, source_log_scores)
        )
        combined_supports.append(
            _normalize_support_scores(known_labels, combined_log_scores)
        )

    for (i, frag_row), duration in zip(frags.iterrows(), durations):
        # Multiplicative length factor: 1.0 for the longest fragment, smaller for
        # shorter ones.  Discounts the entire evidence bundle so a tiny high-confidence
        # fragment cannot outbid a long spatially-consistent track on CNN alone.
        length_scale = math.log1p(duration) / log_max if log_max > 1e-9 else 1.0
        length_factor = 1.0 - length_w * (1.0 - length_scale)
        unary_support = combined_supports[i]

        for j, label in enumerate(known_labels):
            spatial_s, has_neighbors = _spatial_score_for_fragment(
                frag_row, label, schedule, max_vel, no_neighbor_score
            )
            # Veto: fragment is at the wrong place for this identity.
            if has_neighbors and spatial_s < spatial_veto:
                continue  # score_mat[i, j] stays -1 (ineligible)
            evidence = float(unary_support.get(label, 0.0))
            if has_neighbors:
                # Spatial gates all evidence: high-velocity jumps suppress the
                # score even when CNN is strong, so spatial continuity wins.
                raw_score = evidence * spatial_s
            else:
                # No temporal neighbors — spatial assessment impossible.
                raw_score = evidence
            score_mat[i, j] = raw_score * length_factor

    # Cross-trajectory spatial veto: if a shorter fragment overlaps a longer one
    # and is spatially inconsistent with it at their temporal overlap midpoint, the
    # shorter fragment is ineligible for labels where the longer one has meaningful
    # CNN evidence.  This catches the case where the schedule-based veto above cannot
    # fire (neither fragment has temporal neighbors yet).
    if spatial_veto > 0.0:
        frag_list = [
            (idx, row, dur) for (idx, row), dur in zip(frags.iterrows(), durations)
        ]
        for idx_a, row_a, dur_a in frag_list:
            for idx_b, row_b, dur_b in frag_list:
                if dur_b <= dur_a or not _fragments_overlap(row_a, row_b):
                    continue
                os_ = max(int(row_a["StartFrame"]), int(row_b["StartFrame"]))
                oe_ = min(int(row_a["EndFrame"]), int(row_b["EndFrame"]))
                mid = 0.5 * (os_ + oe_)
                sf_a, ef_a = int(row_a["StartFrame"]), int(row_a["EndFrame"])
                alpha_a = (mid - sf_a) / max(1, ef_a - sf_a)
                ax = float(row_a["StartX"]) + alpha_a * (
                    float(row_a["EndX"]) - float(row_a["StartX"])
                )
                ay = float(row_a["StartY"]) + alpha_a * (
                    float(row_a["EndY"]) - float(row_a["StartY"])
                )
                sf_b, ef_b = int(row_b["StartFrame"]), int(row_b["EndFrame"])
                alpha_b = (mid - sf_b) / max(1, ef_b - sf_b)
                bx = float(row_b["StartX"]) + alpha_b * (
                    float(row_b["EndX"]) - float(row_b["StartX"])
                )
                by = float(row_b["StartY"]) + alpha_b * (
                    float(row_b["EndY"]) - float(row_b["StartY"])
                )
                if not all(math.isfinite(v) for v in [ax, ay, bx, by]):
                    continue
                dist = math.hypot(ax - bx, ay - by)
                cross_s = math.exp(-(dist**2) / (2.0 * max_vel**2))
                if cross_s >= spatial_veto:
                    continue
                b_probs = source_supports[idx_b]
                for j, label in enumerate(known_labels):
                    if (
                        float(b_probs.get(label, 0.0)) > 0.3
                        and score_mat[idx_a, j] >= 0
                    ):
                        score_mat[idx_a, j] = -1.0

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


def _build_traj_summaries(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
) -> pd.DataFrame:
    """Build a per-trajectory summary DataFrame for the MILP.

    Columns: TrajectoryID, StartFrame, EndFrame, StartX, StartY, EndX, EndY,
    MeanCNNProbs (dict), MeanTagProbs (dict), CNNLogEvidence (dict),
    TagLogEvidence (dict), OnlineLabel, OnlineConfidence.
    """
    known_labels = list(catalog.labels[1:])
    prefix_prob_cols = _build_cnn_probability_prefix_map(df.columns)
    label_specs = _build_cnn_label_specs(known_labels, prefix_prob_cols, df.columns)
    rows: list[dict] = []

    for traj_id, grp in df.groupby("TrajectoryID", sort=False):
        grp_sorted = grp.sort_values("FrameID").reset_index(drop=True)
        start_f = int(grp_sorted["FrameID"].iloc[0])
        end_f = int(grp_sorted["FrameID"].iloc[-1])

        valid_xy = (
            grp_sorted[grp_sorted["X"].notna() & grp_sorted["Y"].notna()].sort_values(
                "FrameID"
            )
            if "X" in grp_sorted.columns and "Y" in grp_sorted.columns
            else pd.DataFrame()
        )
        if not valid_xy.empty:
            sx = float(valid_xy.iloc[0]["X"])
            sy = float(valid_xy.iloc[0]["Y"])
            ex = float(valid_xy.iloc[-1]["X"])
            ey = float(valid_xy.iloc[-1]["Y"])
        else:
            sx = sy = ex = ey = math.nan

        mean_probs = _trajectory_mean_cnn_probs(grp_sorted, known_labels, label_specs)
        cnn_log_scores = _trajectory_cnn_log_evidence(
            grp_sorted, known_labels, label_specs
        )
        tag_probs, tag_log_scores = _trajectory_tag_evidence(grp_sorted, known_labels)

        label_col = grp_sorted.get(
            _LABEL_COL, pd.Series("unknown", index=grp_sorted.index, dtype=object)
        )
        unknown_mask = label_col.isna() | label_col.astype(str).str.strip().isin(
            _UNKNOWN_VALUES
        )
        known_rows = grp_sorted[~unknown_mask]
        if not known_rows.empty:
            online_label = str(known_rows[_LABEL_COL].astype(str).mode().iloc[0])
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
                "StartFrame": start_f,
                "EndFrame": end_f,
                "StartX": sx,
                "StartY": sy,
                "EndX": ex,
                "EndY": ey,
                "MeanCNNProbs": mean_probs,
                "MeanTagProbs": tag_probs,
                "CNNLogEvidence": cnn_log_scores,
                "TagLogEvidence": tag_log_scores,
                "OnlineLabel": online_label,
                "OnlineConfidence": online_conf,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "TrajectoryID",
                "StartFrame",
                "EndFrame",
                "StartX",
                "StartY",
                "EndX",
                "EndY",
                "MeanCNNProbs",
                "MeanTagProbs",
                "CNNLogEvidence",
                "TagLogEvidence",
                "OnlineLabel",
                "OnlineConfidence",
            ]
        )
    return pd.DataFrame(rows)


def solve_global_assignment(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Assign one identity label per trajectory using a two-pass MILP.

    Builds per-trajectory summaries internally, runs the MILP with spatial
    continuity + CNN evidence + online-label prior, then writes
    IdentityAssignedLabel, IdentityFragmentScore, and IdentityCommitted back
    into every row of each trajectory.  Returns a modified copy of df.

    Two-pass strategy:
    - Pass 1: spatial schedule seeded from online labels.
    - Pass 2: schedule re-seeded from Pass-1 results to remove online-label bias.
    Margin threshold applied to Pass-2: only accept re-assignment when
    score(milp_label) - score(second_best) >= ASSIGNMENT_MARGIN_THRESHOLD.
    """
    known_labels = list(catalog.labels[1:])
    if not known_labels or df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    known_label_set = set(known_labels)
    margin_thresh = float(params.get("ASSIGNMENT_MARGIN_THRESHOLD", 0.10))

    def _catalog_label_or_unknown(lbl: str) -> str:
        return lbl if lbl in known_label_set else "unknown"

    traj_summaries = _build_traj_summaries(df, catalog)
    if traj_summaries.empty:
        return df

    summaries = traj_summaries.reset_index(drop=True)
    n_trajs = len(summaries)
    n_labels = len(known_labels)

    # Pass 1: seed schedule from OnlineLabel.
    schedule1 = _build_schedule(summaries, "OnlineLabel")
    score_mat1 = _build_score_matrix(summaries, known_labels, schedule1, params)
    assigned1 = _milp_solve(summaries, known_labels, score_mat1, params)

    pass1_labels = [
        (
            str(assigned1.get(i))
            if assigned1.get(i) is not None
            else _catalog_label_or_unknown(str(summaries.iloc[i]["OnlineLabel"]))
        )
        for i in range(n_trajs)
    ]

    # Pass 2: re-seed schedule from Pass-1 labels to remove online-label bias.
    summaries_pass1 = summaries.copy()
    summaries_pass1["_pass1_label"] = pass1_labels
    schedule2 = _build_schedule(summaries_pass1, "_pass1_label")
    score_mat2 = _build_score_matrix(summaries, known_labels, schedule2, params)
    assigned2 = _milp_solve(summaries, known_labels, score_mat2, params)

    # Trajectories where every label was spatially vetoed: prefer "unknown" over
    # an online label that may itself represent a spatial jump.
    all_vetoed = frozenset(
        i for i in range(n_trajs) if all(score_mat2[i, j] < 0 for j in range(n_labels))
    )

    # Apply margin threshold.
    labels_out: list[str] = []
    for i in range(n_trajs):
        online_lbl = _catalog_label_or_unknown(str(summaries.iloc[i]["OnlineLabel"]))
        milp_label = assigned2.get(i)

        if milp_label is None:
            # Use "unknown" when spatial constraints ruled out every assignment,
            # so we don't silently keep an online label that implies a jump.
            labels_out.append("unknown" if i in all_vetoed else online_lbl)
            continue

        if milp_label == online_lbl:
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

    assigned_scores: list[float] = []
    for i in range(n_trajs):
        label = labels_out[i]
        if label in known_labels:
            j = known_labels.index(label)
            s = score_mat2[i, j]
            assigned_scores.append(float(s) if s >= 0 else 0.0)
        else:
            assigned_scores.append(0.0)

    # Write one label per trajectory back to every row.
    out = df.copy()
    if "IdentityAssignedLabel" not in out.columns:
        out["IdentityAssignedLabel"] = np.nan
    if "IdentityCommitted" not in out.columns:
        out["IdentityCommitted"] = False
    out["IdentityCommitted"] = out["IdentityCommitted"].fillna(False).astype(bool)
    if "IdentityFragmentScore" not in out.columns:
        out["IdentityFragmentScore"] = np.nan

    for i in range(n_trajs):
        label = labels_out[i]
        if not label or label in _UNKNOWN_VALUES:
            continue
        traj_id = summaries.iloc[i]["TrajectoryID"]
        mask = out["TrajectoryID"] == traj_id
        out.loc[mask, "IdentityAssignedLabel"] = label
        out.loc[mask, "IdentityFragmentScore"] = assigned_scores[i]
        out.loc[mask, "IdentityCommitted"] = True

    return out


def run_fragment_solver(
    trajectories_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """End-to-end fragment solver: (optional PELT split) -> MILP assign.

    Parameters
    ----------
    trajectories_df : post-augmentation trajectory DataFrame.
    catalog : IdentityCatalog for the run.
    params : optional overrides. Keys:
        ENABLE_PELT_SPLITTING            bool   default False
        CHANGEPOINT_PENALTY              float  default 3.0
        MIN_FRAGMENT_FRAMES              int    default 5
        PELT_MODEL                       str    default "rbf" (l1 / l2 / rbf)
        FRAGMENT_CNN_WEIGHT              float  default 0.40
        FRAGMENT_TAG_WEIGHT              float  default 0.15
        FRAGMENT_SPATIAL_WEIGHT          float  default 0.35 (legacy; currently unused)
        ONLINE_PRIOR_WEIGHT              float  default 0.25
        FRAGMENT_LENGTH_WEIGHT           float  default 0.60
            Multiplicative blend [0,1]: discounts short fragments' evidence relative
            to the longest fragment in the pool.  Prevents a tiny high-confidence
            fragment from overriding a long spatially-consistent track on CNN alone.
        SPATIAL_NO_NEIGHBOR_SCORE        float  default 0.3
        FRAGMENT_SPATIAL_VETO_THRESHOLD  float  default 0.05
            Minimum acceptable spatial score when neighbors exist; fragments below
            this are marked ineligible for that identity (spatially incompatible).
        ASSIGNMENT_MARGIN_THRESHOLD      float  default 0.10
        MAX_VELOCITY_BREAK               float  default 50.0
        FRAGMENT_SOLVER_ILP_TIME_LIMIT   float  default 30.0
    """
    params = params or {}

    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df if trajectories_df is not None else pd.DataFrame()

    known_labels = list(catalog.labels[1:])
    if not known_labels:
        return trajectories_df

    if params.get("ENABLE_PELT_SPLITTING", False):
        changepoints = detect_identity_changepoints(trajectories_df, catalog, params)
        split_df = split_trajectories_at_changepoints(
            trajectories_df, changepoints, params
        )
        n_splits = sum(len(v) for v in changepoints.values())
        log.info(
            "fragment_solver: PELT found %d changepoints; %d → %d trajectories after splitting.",
            n_splits,
            trajectories_df["TrajectoryID"].nunique(),
            split_df["TrajectoryID"].nunique(),
        )
    else:
        split_df = trajectories_df
        log.info(
            "fragment_solver: PELT splitting disabled; MILP assigning labels to %d existing trajectories.",
            trajectories_df["TrajectoryID"].nunique(),
        )

    return solve_global_assignment(split_df, catalog, params)
