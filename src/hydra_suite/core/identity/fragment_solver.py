"""Global identity fragment solver.

Identity post-processing pipeline:
1. PELT changepoint detection on per-trajectory CNN probability matrices.
2. Fragment building from detected changepoints.
3. Iterative greedy label refinement: walks fragments in order of doubt score
   (low CNN stability × short length × poor spatial fit + Unknown bonus),
   evaluates the top-K candidate labels for each, and commits a flip only if
   it strictly increases a global objective (sum of evidence × spatial × length
   over all fragments). Iterates to a fixed point. Long fragments with stable
   per-frame CNN agreement settle to the bottom of the queue and act as
   anchors; short or jittery fragments yield to the schedule formed by them.
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


def _trajectory_per_row_probs(
    grp_sorted: pd.DataFrame,
    known_labels: list[str],
    label_specs: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Return ``(n_rows, n_labels)`` per-row reconstructed CNN probabilities.

    Cells where no CNN evidence is present remain NaN. Direct probability
    columns and part-product columns are combined identically to the mean-prob
    helper so the per-row matrix stays consistent with the fragment-level
    aggregates.
    """
    n_rows = len(grp_sorted)
    n_labels = len(known_labels)
    out = np.full((n_rows, n_labels), np.nan, dtype=np.float64)
    if n_rows == 0 or n_labels == 0:
        return out

    for j, label in enumerate(known_labels):
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
        if direct_vals:
            direct_stack = np.stack(direct_vals, axis=0)
            direct_mask = np.all(np.isfinite(direct_stack), axis=0)
            if np.any(direct_mask):
                row_prob[direct_mask] = np.prod(direct_stack[:, direct_mask], axis=0)

        if use_parts and part_vals:
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

        out[:, j] = row_prob

    return out


def _fragment_stability(per_row_probs: np.ndarray) -> float:
    """Combined agreement × mean-margin stability score in [0, 1].

    Stability is high for fragments whose per-frame argmax is consistently the
    same label (high agreement) *and* whose top-1 / top-2 separation is wide
    (high mean margin). A long fragment with jittery per-frame predictions or
    a small margin scores low even though it has many rows; a short fragment
    that is internally consistent and confident scores high.

    Returns 0.0 when no per-frame CNN evidence is present.
    """
    if per_row_probs.size == 0:
        return 0.0
    valid_mask = np.isfinite(per_row_probs).any(axis=1)
    if not valid_mask.any():
        return 0.0
    valid = per_row_probs[valid_mask]
    valid = np.where(np.isfinite(valid), valid, 0.0)
    n = valid.shape[0]
    n_labels = valid.shape[1]

    top1_idx = np.argmax(valid, axis=1)
    if n_labels >= 2:
        sorted_desc = np.sort(valid, axis=1)[:, ::-1]
        top1 = sorted_desc[:, 0]
        top2 = sorted_desc[:, 1]
    else:
        top1 = valid[:, 0]
        top2 = np.zeros(n, dtype=np.float64)

    counts = np.bincount(top1_idx, minlength=n_labels)
    agreement = float(counts.max()) / float(n)
    margin = float(np.mean(top1 - top2))
    return float(agreement * max(0.0, margin))


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


def _spatial_score_for_fragment(
    frag: pd.Series,
    identity: str,
    schedule: dict[str, list[dict]],
    max_velocity: float,
    no_neighbor_score: float = 0.3,
    max_bridge_gap: int = 30,
) -> tuple[float, bool]:
    """Velocity-based spatial continuity score against nearest neighboring segment.

    Returns (score, has_neighbors).

    Scoring:
    - Effective gap = min(actual gap, ``max_bridge_gap``). Clamping prevents
      arbitrarily long temporal gaps from excusing arbitrarily large spatial
      jumps: beyond ``max_bridge_gap`` frames we have no evidence of the
      animal's path, so the bridge must still be explainable as if the gap
      were no longer than this window.
    - Implied velocity = dist / effective_gap (pixels per frame).
    - Hard veto: if velocity > max_velocity for any neighbor, return (0.0, True)
      immediately — physically implausible jump, caller should mark ineligible.
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
    cap = max(1, int(max_bridge_gap))

    prior = max(
        (s for s in segs if s["end_frame"] < t0),
        key=lambda s: s["end_frame"],
        default=None,
    )
    if prior and all(
        math.isfinite(v) for v in [x0, y0, prior["end_X"], prior["end_Y"]]
    ):
        gap = max(1, t0 - prior["end_frame"])
        effective_gap = min(gap, cap)
        dist = math.hypot(x0 - prior["end_X"], y0 - prior["end_Y"])
        velocity = dist / effective_gap
        if velocity > max_velocity:
            return 0.0, True  # physically implausible — hard veto
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
        effective_gap = min(gap, cap)
        dist = math.hypot(x1 - following["start_X"], y1 - following["start_Y"])
        velocity = dist / effective_gap
        if velocity > max_velocity:
            return 0.0, True  # physically implausible — hard veto
        term_scores.append(math.exp(-2.0 * (velocity / max_velocity) ** 2))

    if term_scores:
        return float(np.mean(term_scores)), True
    return no_neighbor_score, False


def _seg_from_row(row: pd.Series) -> dict:
    return {
        "start_frame": int(row["StartFrame"]),
        "end_frame": int(row["EndFrame"]),
        "start_X": float(row["StartX"]),
        "start_Y": float(row["StartY"]),
        "end_X": float(row["EndX"]),
        "end_Y": float(row["EndY"]),
    }


def _iterative_assign(
    frags: pd.DataFrame,
    known_labels: list[str],
    params: dict[str, Any],
) -> dict[int, str | None]:
    """Iteratively refine fragment-to-label assignment.

    Walks fragments by descending doubt score, evaluates the top-K candidate
    labels (∪ Unknown ∪ current) for each, and commits a flip only if the
    delta against the current per-fragment score is at least
    ``ASSIGNMENT_MARGIN_THRESHOLD``. Iterates to a fixed point (no flips in a
    pass) or until ``FRAGMENT_MAX_PASSES`` is hit.

    The schedule of committed labels is updated incrementally after every
    accepted flip; collision (same-label time overlap) and physical-velocity
    vetoes are enforced per-evaluation. A final Unknown-rescue pass assigns
    Unknown fragments their best feasible label even when the score is below
    the monotone gate, since Unknown contributes 0 to the global objective.

    Returns ``{frag_index: assigned_label_or_None}`` (None means Unknown).
    """
    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    tag_w = float(params.get("FRAGMENT_TAG_WEIGHT", 0.15))
    length_w = min(1.0, max(0.0, float(params.get("FRAGMENT_LENGTH_WEIGHT", 0.60))))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    max_bridge_gap = max(1, int(params.get("MAX_BRIDGE_GAP_FRAMES", 30)))
    no_neighbor_score = float(params.get("SPATIAL_NO_NEIGHBOR_SCORE", 0.3))
    spatial_veto = float(params.get("FRAGMENT_SPATIAL_VETO_THRESHOLD", 0.05))
    monotone_eps = float(params.get("ASSIGNMENT_MARGIN_THRESHOLD", 0.10))
    top_k = max(1, int(params.get("FRAGMENT_TOP_K", 3)))
    max_passes = max(1, int(params.get("FRAGMENT_MAX_PASSES", 10)))
    unknown_doubt_bonus = float(params.get("FRAGMENT_UNKNOWN_DOUBT_BONUS", 0.5))

    n_frags = len(frags)
    if n_frags == 0:
        return {}

    known_label_set = set(known_labels)

    # Pre-compute durations and length factors.
    durations = np.array(
        [
            max(1, int(r["EndFrame"]) - int(r["StartFrame"]) + 1)
            for _, r in frags.iterrows()
        ],
        dtype=np.float64,
    )
    max_duration = float(durations.max())
    log_max = math.log1p(max_duration)
    length_scales = (
        np.log1p(durations) / log_max
        if log_max > 1e-9
        else np.ones(n_frags, dtype=np.float64)
    )
    length_factors = 1.0 - length_w * (1.0 - length_scales)

    # Pre-compute combined evidence supports (used for candidate ranking + scoring).
    combined_supports: list[dict[str, float]] = []
    for _, frag_row in frags.iterrows():
        cnn_log = frag_row.get("CNNLogEvidence") or {}
        tag_log = frag_row.get("TagLogEvidence") or {}
        online_lbl = str(frag_row["OnlineLabel"])
        online_conf = float(frag_row["OnlineConfidence"])
        prior_log = _build_prior_log_scores(known_labels, online_lbl, online_conf)
        combined_log = {
            label: (
                cnn_w * float(cnn_log.get(label, 0.0))
                + tag_w * float(tag_log.get(label, 0.0))
                + prior_w * float(prior_log[label])
            )
            for label in known_labels
        }
        combined_supports.append(_normalize_support_scores(known_labels, combined_log))

    stabilities = np.array(
        [float(r.get("Stability", 0.0)) for _, r in frags.iterrows()],
        dtype=np.float64,
    )

    def _sanitize(lbl: Any) -> str | None:
        s = str(lbl)
        if s in _UNKNOWN_VALUES or s not in known_label_set:
            return None
        return s

    current: list[str | None] = [
        _sanitize(frags.iloc[i]["OnlineLabel"]) for i in range(n_frags)
    ]

    # Schedule: dict[label] -> list of (frag_idx, segment_dict), unordered;
    # spatial helpers don't require sortedness.
    schedule: dict[str, list[tuple[int, dict]]] = {lbl: [] for lbl in known_labels}
    for i in range(n_frags):
        lbl = current[i]
        if lbl is not None:
            schedule[lbl].append((i, _seg_from_row(frags.iloc[i])))

    def _spatial_for(
        label: str, exclude_idx: int, also_exclude: int | None = None
    ) -> dict[str, list[dict]]:
        excludes = {exclude_idx}
        if also_exclude is not None:
            excludes.add(also_exclude)
        return {
            label: [
                seg for (idx, seg) in schedule.get(label, []) if idx not in excludes
            ]
        }

    def _has_collision(
        label: str,
        exclude_idx: int,
        t0: int,
        t1: int,
        also_exclude: int | None = None,
    ) -> bool:
        for idx, seg in schedule.get(label, []):
            if idx in (exclude_idx, also_exclude):
                continue
            if seg["start_frame"] <= t1 and t0 <= seg["end_frame"]:
                return True
        return False

    def _candidate_score(
        i: int, label: str, also_exclude: int | None = None
    ) -> tuple[float, bool]:
        """Evaluate placing fragment i under ``label``. Returns (score, vetoed).

        ``also_exclude`` lets callers temporarily ignore another fragment in the
        schedule (used by the swap move to score i under ``label`` as if the
        blocking neighbor had already been displaced).
        """
        row = frags.iloc[i]
        t0 = int(row["StartFrame"])
        t1 = int(row["EndFrame"])
        if _has_collision(label, i, t0, t1, also_exclude):
            return 0.0, True
        spatial_s, has_neighbors = _spatial_score_for_fragment(
            row,
            label,
            _spatial_for(label, i, also_exclude),
            max_vel,
            no_neighbor_score,
            max_bridge_gap,
        )
        if has_neighbors and spatial_s < spatial_veto:
            return 0.0, True
        evidence = float(combined_supports[i].get(label, 0.0))
        raw = evidence * spatial_s if has_neighbors else evidence
        return float(raw * float(length_factors[i])), False

    def _find_blocker(i: int, label: str) -> int | None:
        """Identify the schedule entry whose presence vetoes placing i under
        ``label``. Returns the blocking fragment index or None if no single
        blocker is responsible (e.g. when the no-neighbor score is below
        ``spatial_veto``, which cannot be resolved by displacing one segment).

        Mirrors the veto checks in ``_has_collision`` and
        ``_spatial_score_for_fragment``.
        """
        row = frags.iloc[i]
        t0 = int(row["StartFrame"])
        t1 = int(row["EndFrame"])
        segs_with_idx = [
            (idx, seg) for (idx, seg) in schedule.get(label, []) if idx != i
        ]
        for idx, seg in segs_with_idx:
            if seg["start_frame"] <= t1 and t0 <= seg["end_frame"]:
                return idx
        x0, y0 = float(row["StartX"]), float(row["StartY"])
        x1, y1 = float(row["EndX"]), float(row["EndY"])
        cap = max(1, int(max_bridge_gap))
        prior = max(
            ((idx, seg) for (idx, seg) in segs_with_idx if seg["end_frame"] < t0),
            key=lambda x: x[1]["end_frame"],
            default=None,
        )
        if prior is not None:
            idx, seg = prior
            if all(math.isfinite(v) for v in [x0, y0, seg["end_X"], seg["end_Y"]]):
                gap = max(1, t0 - seg["end_frame"])
                if (
                    math.hypot(x0 - seg["end_X"], y0 - seg["end_Y"]) / min(gap, cap)
                    > max_vel
                ):
                    return idx
        following = min(
            ((idx, seg) for (idx, seg) in segs_with_idx if seg["start_frame"] > t1),
            key=lambda x: x[1]["start_frame"],
            default=None,
        )
        if following is not None:
            idx, seg = following
            if all(math.isfinite(v) for v in [x1, y1, seg["start_X"], seg["start_Y"]]):
                gap = max(1, seg["start_frame"] - t1)
                if (
                    math.hypot(x1 - seg["start_X"], y1 - seg["start_Y"]) / min(gap, cap)
                    > max_vel
                ):
                    return idx
        return None

    def _best_alt_for(
        j: int, exclude_label: str, partner: int
    ) -> tuple[float, str | None]:
        """Best feasible label for fragment j excluding ``exclude_label``,
        scored as if ``partner`` were already absent from the schedule.

        Used by the swap move: when i takes ``exclude_label`` and j is the
        displaced occupant, j must find an alternative — and j scores its
        candidates as if i had already vacated j's eventual destination
        (``partner`` was the segment whose presence forced the displacement).
        Returns (score, label_or_None).
        """
        sup = combined_supports[j]
        ranked = sorted(known_labels, key=lambda lbl: -sup.get(lbl, 0.0))[:top_k]
        best_s = 0.0  # Unknown is the floor (score 0, never vetoed).
        best_l: str | None = None
        for jc in ranked:
            if jc == exclude_label:
                continue
            s, vetoed = _candidate_score(j, jc, also_exclude=partner)
            if vetoed:
                continue
            if s > best_s:
                best_s = s
                best_l = jc
        return best_s, best_l

    def _current_score(i: int) -> float:
        cur = current[i]
        if cur is None:
            return 0.0
        score, vetoed = _candidate_score(i, cur)
        return -math.inf if vetoed else score

    def _doubt_score(i: int) -> float:
        s_norm = 1.0 - float(stabilities[i])
        l_norm = 1.0 - float(length_scales[i])
        cur = current[i]
        if cur is None:
            return s_norm * l_norm + unknown_doubt_bonus
        spatial_s, has_neighbors = _spatial_score_for_fragment(
            frags.iloc[i],
            cur,
            _spatial_for(cur, i),
            max_vel,
            no_neighbor_score,
            max_bridge_gap,
        )
        fit = float(spatial_s) if has_neighbors else no_neighbor_score
        return s_norm * l_norm * (1.0 - fit)

    def _commit(i: int, new_label: str | None) -> None:
        cur = current[i]
        if cur is not None:
            schedule[cur] = [(idx, seg) for (idx, seg) in schedule[cur] if idx != i]
        if new_label is not None:
            schedule[new_label].append((i, _seg_from_row(frags.iloc[i])))
        current[i] = new_label

    for pass_idx in range(max_passes):
        order = sorted(range(n_frags), key=lambda i: -_doubt_score(i))
        flips = 0
        for i in order:
            cur = current[i]
            cur_score = _current_score(i)

            sup = combined_supports[i]
            top_evidence = sorted(known_labels, key=lambda lbl: -sup.get(lbl, 0.0))[
                :top_k
            ]
            seen: set[str | None] = set()
            candidates: list[str | None] = [None]  # Unknown is always a candidate.
            seen.add(None)
            for c in top_evidence:
                if c not in seen:
                    candidates.append(c)
                    seen.add(c)

            best_score = cur_score
            best_label: str | None = cur
            best_swap: tuple[int, str | None] | None = None
            best_delta = 0.0
            for c in candidates:
                if c == cur:
                    continue
                if c is None:
                    score = 0.0
                    vetoed = False
                else:
                    score, vetoed = _candidate_score(i, c)
                if not vetoed:
                    delta = score - cur_score
                    if score > best_score and delta > best_delta:
                        best_score = score
                        best_label = c
                        best_swap = None
                        best_delta = delta
                    continue
                # Vetoed simple flip — try a one-step swap with the blocker.
                if c is None:
                    continue
                blocker = _find_blocker(i, c)
                if blocker is None or blocker == i or current[blocker] != c:
                    continue
                score_i_clean, vetoed_i = _candidate_score(i, c, also_exclude=blocker)
                if vetoed_i:
                    continue
                cur_score_j = _current_score(blocker)
                if not math.isfinite(cur_score_j):
                    cur_score_j = 0.0
                alt_score_j, alt_label_j = _best_alt_for(blocker, c, partner=i)
                # Both fragments must individually be no worse off under the
                # swap.  Without this guard a high-evidence short fragment
                # could displace a long anchor whenever the short fragment's
                # gain exceeds the anchor's loss in joint sum — re-introducing
                # the very behavior the length factor exists to prevent.
                cur_score_i_finite = cur_score if math.isfinite(cur_score) else 0.0
                if score_i_clean < cur_score_i_finite or alt_score_j < cur_score_j:
                    continue
                swap_delta = (score_i_clean + alt_score_j) - (
                    cur_score_i_finite + cur_score_j
                )
                if swap_delta < monotone_eps:
                    continue
                if swap_delta > best_delta:
                    best_score = score_i_clean
                    best_label = c
                    best_swap = (blocker, alt_label_j)
                    best_delta = swap_delta

            if best_label == cur and best_swap is None:
                continue
            if best_delta < monotone_eps:
                continue
            if best_swap is not None:
                blocker, alt_label_j = best_swap
                # Free the blocker first so the schedule is consistent when
                # i's commit re-adds it under best_label.
                _commit(blocker, alt_label_j)
                _commit(i, best_label)
                flips += 2
            else:
                _commit(i, best_label)
                flips += 1

        log.debug("iterative fragment solver pass %d: %d flips", pass_idx + 1, flips)
        if flips == 0:
            break
    else:
        log.warning(
            "Iterative fragment solver hit FRAGMENT_MAX_PASSES (%d) without convergence.",
            max_passes,
        )

    # Unknown-rescue pass: assign any Unknown fragment its best feasible label
    # even if the score is below the monotone gate (Unknown contributes 0, any
    # feasible label strictly improves the objective).
    for i in range(n_frags):
        if current[i] is not None:
            continue
        sup = combined_supports[i]
        top_evidence = sorted(known_labels, key=lambda lbl: -sup.get(lbl, 0.0))[:top_k]
        best_score = 0.0
        best_label: str | None = None
        for c in top_evidence:
            score, vetoed = _candidate_score(i, c)
            if vetoed:
                continue
            if score > best_score:
                best_score = score
                best_label = c
        if best_label is not None:
            _commit(i, best_label)

    return {i: current[i] for i in range(n_frags)}


def _build_traj_summaries(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
) -> pd.DataFrame:
    """Build a per-trajectory summary DataFrame consumed by the iterative solver.

    Columns: TrajectoryID, StartFrame, EndFrame, StartX, StartY, EndX, EndY,
    MeanCNNProbs (dict), MeanTagProbs (dict), CNNLogEvidence (dict),
    TagLogEvidence (dict), Stability (float), OnlineLabel, OnlineConfidence.
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
        per_row_probs = _trajectory_per_row_probs(grp_sorted, known_labels, label_specs)
        stability = _fragment_stability(per_row_probs)
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
                "Stability": stability,
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
                "Stability",
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
    """Assign one identity label per trajectory via the iterative solver.

    Builds per-trajectory summaries internally, runs ``_iterative_assign`` with
    spatial continuity + CNN/tag evidence + online-label prior + per-fragment
    stability, then writes IdentityAssignedLabel, IdentityFragmentScore, and
    IdentityCommitted back into every row of each trajectory. Returns a
    modified copy of df.
    """
    known_labels = list(catalog.labels[1:])
    if not known_labels or df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    traj_summaries = _build_traj_summaries(df, catalog)
    if traj_summaries.empty:
        return df

    summaries = traj_summaries.reset_index(drop=True)
    n_trajs = len(summaries)

    assigned = _iterative_assign(summaries, known_labels, params)

    # Per-fragment final score for committed labels (recomputed with the final
    # schedule so the value reflects the converged spatial configuration).
    final_schedule: dict[str, list[dict]] = {}
    for i in range(n_trajs):
        lbl = assigned.get(i)
        if lbl is None:
            continue
        final_schedule.setdefault(lbl, []).append(_seg_from_row(summaries.iloc[i]))

    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    tag_w = float(params.get("FRAGMENT_TAG_WEIGHT", 0.15))
    length_w = min(1.0, max(0.0, float(params.get("FRAGMENT_LENGTH_WEIGHT", 0.60))))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    max_bridge_gap = max(1, int(params.get("MAX_BRIDGE_GAP_FRAMES", 30)))
    no_neighbor_score = float(params.get("SPATIAL_NO_NEIGHBOR_SCORE", 0.3))

    durations = np.array(
        [
            max(1, int(r["EndFrame"]) - int(r["StartFrame"]) + 1)
            for _, r in summaries.iterrows()
        ],
        dtype=np.float64,
    )
    log_max = math.log1p(float(durations.max()))
    length_scales = (
        np.log1p(durations) / log_max
        if log_max > 1e-9
        else np.ones(n_trajs, dtype=np.float64)
    )
    length_factors = 1.0 - length_w * (1.0 - length_scales)

    assigned_scores: list[float] = []
    for i in range(n_trajs):
        lbl = assigned.get(i)
        if lbl is None or lbl not in known_labels:
            assigned_scores.append(0.0)
            continue
        cnn_log = summaries.iloc[i].get("CNNLogEvidence") or {}
        tag_log = summaries.iloc[i].get("TagLogEvidence") or {}
        online_lbl = str(summaries.iloc[i]["OnlineLabel"])
        online_conf = float(summaries.iloc[i]["OnlineConfidence"])
        prior_log = _build_prior_log_scores(known_labels, online_lbl, online_conf)
        combined_log = {
            label: (
                cnn_w * float(cnn_log.get(label, 0.0))
                + tag_w * float(tag_log.get(label, 0.0))
                + prior_w * float(prior_log[label])
            )
            for label in known_labels
        }
        support = _normalize_support_scores(known_labels, combined_log)
        sched_minus_self = {
            lbl: [
                seg
                for j, seg in enumerate(final_schedule.get(lbl, []))
                if not (
                    seg["start_frame"] == int(summaries.iloc[i]["StartFrame"])
                    and seg["end_frame"] == int(summaries.iloc[i]["EndFrame"])
                )
            ]
        }
        spatial_s, has_neighbors = _spatial_score_for_fragment(
            summaries.iloc[i],
            lbl,
            sched_minus_self,
            max_vel,
            no_neighbor_score,
            max_bridge_gap,
        )
        evidence = float(support.get(lbl, 0.0))
        raw = evidence * spatial_s if has_neighbors else evidence
        assigned_scores.append(float(raw * float(length_factors[i])))

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
        label = assigned.get(i)
        traj_id = summaries.iloc[i]["TrajectoryID"]
        mask = out["TrajectoryID"] == traj_id
        if label is None or label in _UNKNOWN_VALUES:
            # The solver explicitly chose Unknown for this fragment (e.g. its
            # spatial fit under every feasible label fails the veto). Clear the
            # online label so the user sees the solver's decision.
            out.loc[mask, "IdentityAssignedLabel"] = "unknown"
            out.loc[mask, "IdentityFragmentScore"] = 0.0
            out.loc[mask, "IdentityCommitted"] = False
            continue
        out.loc[mask, "IdentityAssignedLabel"] = label
        out.loc[mask, "IdentityFragmentScore"] = assigned_scores[i]
        out.loc[mask, "IdentityCommitted"] = True

    return out


def run_fragment_solver(
    trajectories_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """End-to-end fragment solver: (optional PELT split) → iterative assign.

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
            Minimum global-objective delta required to accept a fragment relabel
            during iterative refinement (monotone gate epsilon).
        MAX_VELOCITY_BREAK               float  default 50.0
        MAX_BRIDGE_GAP_FRAMES            int    default 30
            Cap on the temporal gap (in frames) used when computing the implied
            velocity between two same-identity segments.  Without this cap an
            arbitrarily long temporal gap would excuse an arbitrarily large
            spatial jump (``dist / gap`` shrinks with gap), so the same identity
            could be assigned to two trajectories at far-apart positions
            separated by a long pause.  Beyond this window we have no evidence
            of the animal's path; the bridge must remain plausible as if the
            gap were no longer than this many frames.
        FRAGMENT_TOP_K                   int    default 3
            Number of top-evidence candidate labels evaluated per fragment per pass.
        FRAGMENT_MAX_PASSES              int    default 10
            Hard cap on iterative-refinement passes.
        FRAGMENT_UNKNOWN_DOUBT_BONUS     float  default 0.5
            Additive doubt bonus for currently-Unknown fragments so they get
            re-evaluated early in each pass.
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
            "fragment_solver: PELT splitting disabled; iteratively assigning labels to %d existing trajectories.",
            trajectories_df["TrajectoryID"].nunique(),
        )

    return solve_global_assignment(split_df, catalog, params)
