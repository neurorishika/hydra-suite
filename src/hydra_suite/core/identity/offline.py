"""Offline identity decoder.

Identity Phases 3 & 4: runs after full tracking + post-processing on the final
trajectory dataframe, using the persisted evidence sidecar to improve identity
assignments with future evidence (forward-backward smoothing) and to enforce
global uniqueness across overlapping trajectory fragments.

Entry points
------------
``smooth_trajectory_identity_posteriors``
    Forward-backward pass per trajectory; adds ``IdentitySmoothed*`` columns.
``split_mixed_identity_trajectories``
    Posterior-driven splitter for stitched trajectories before fragmenting.
``build_identity_fragments``
    Segments smoothed trajectories into stable fragments.
``solve_fragment_identity_assignment``
    Globally unique assignment across fragments.
``run_identity_residual_assignment``
    Secondary pass to fill unassigned fragments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from hydra_suite.core.identity.cache import IdentityEvidenceCache
from hydra_suite.core.identity.catalog import IdentityCatalog

log = logging.getLogger(__name__)


def _fragments_overlap(row_a: pd.Series, row_b: pd.Series) -> bool:
    return not (
        int(row_a["EndFrame"]) < int(row_b["StartFrame"])
        or int(row_b["EndFrame"]) < int(row_a["StartFrame"])
    )


def _fragment_label_scores(fragment_df: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for row in fragment_df.itertuples():
        label = getattr(row, "IdentitySmoothedLabel", None)
        conf = getattr(row, "IdentitySmoothedConf", np.nan)
        if label is None or pd.isna(label):
            continue
        try:
            conf_value = float(conf)
        except Exception:
            conf_value = 0.0
        scores.setdefault(str(label), []).append(conf_value)
    return {
        label: float(np.mean(values))
        for label, values in scores.items()
        if values
    }


def _normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    out = np.asarray(log_probs, dtype=np.float64)
    out -= np.logaddexp.reduce(out)
    return out


def _remap_cache_log_probs_to_catalog(
    log_probs: np.ndarray,
    cache_labels: Optional[Sequence[str]],
    catalog: IdentityCatalog,
) -> np.ndarray:
    """Map cache-local evidence vectors into the run-wide catalog order."""
    arr = np.asarray(log_probs, dtype=np.float64)
    if cache_labels is None:
        if len(arr) == catalog.size:
            return _normalize_log_probs(arr)
        return catalog.known_uniform_log_prior()

    labels = tuple(str(label) for label in cache_labels)
    if len(labels) != len(arr):
        return catalog.known_uniform_log_prior()

    probs = np.exp(arr - arr.max())
    probs /= np.clip(probs.sum(), 1e-300, None)
    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)

    for src_idx, label in enumerate(labels):
        if not catalog.contains(label):
            continue
        remapped[catalog.index_of(label)] += float(probs[src_idx])

    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))


def _coerce_evidence_caches(
    evidence_cache: IdentityEvidenceCache | Sequence[IdentityEvidenceCache],
) -> list[IdentityEvidenceCache]:
    if isinstance(evidence_cache, IdentityEvidenceCache):
        return [evidence_cache]
    return [cache for cache in evidence_cache if cache is not None]


def _build_detection_evidence_lookup(
    evidence_cache: IdentityEvidenceCache | Sequence[IdentityEvidenceCache],
    catalog: IdentityCatalog,
) -> dict[tuple[int, int], np.ndarray]:
    """Aggregate all evidence into a detection-keyed log-posterior lookup."""
    caches = _coerce_evidence_caches(evidence_cache)
    combined: dict[tuple[int, int], list[np.ndarray]] = {}

    for cache in caches:
        cache_labels = cache.catalog_labels
        for frame_idx in cache.get_cached_frames():
            for evidence in cache.load_frame(int(frame_idx)):
                mapped = _remap_cache_log_probs_to_catalog(
                    evidence.log_probs,
                    cache_labels,
                    catalog,
                )
                key = (int(frame_idx), int(evidence.detection_id))
                combined.setdefault(key, []).append(mapped)

    lookup: dict[tuple[int, int], np.ndarray] = {}
    for key, mapped_rows in combined.items():
        total = np.sum(np.stack(mapped_rows, axis=0), axis=0)
        lookup[key] = _normalize_log_probs(total)
    return lookup


# ---------------------------------------------------------------------------
# Phase 3 – forward-backward smoothing
# ---------------------------------------------------------------------------


def smooth_trajectory_identity_posteriors(
    trajectories_df: pd.DataFrame,
    evidence_cache: IdentityEvidenceCache | Sequence[IdentityEvidenceCache],
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run forward-backward smoothing over per-trajectory identity posteriors.

    For each trajectory, loads stored evidence per frame, runs a Bayesian
    forward-backward (HMM) pass to compute the smoothed posterior at each
    frame, then appends summary columns.

    New columns added
    -----------------
    ``IdentitySmoothedLabel``    Most likely known identity after smoothing.
    ``IdentitySmoothedConf``     Smoothed posterior probability for that label.
    ``IdentitySmoothedEntropy``  Shannon entropy of the smoothed posterior.
    ``IdentitySmoothedMargin``   Gap between the top-1 and top-2 known labels.

    Parameters
    ----------
    trajectories_df:
        Final trajectory DataFrame with at least ``TrajectoryID`` and
        ``FrameID`` columns.
    evidence_cache:
        One or more opened-for-read ``IdentityEvidenceCache`` sidecars.
    catalog:
        Identity catalog for this run.
    params:
        Runtime parameter dict.  Reads ``IDENTITY_TRANSITION_EPSILON``.

    Returns
    -------
    pd.DataFrame
        Copy of *trajectories_df* with the three new columns appended.
    """
    if trajectories_df.empty:
        return trajectories_df.copy()

    transition_epsilon = float(params.get("IDENTITY_TRANSITION_EPSILON", 0.02))
    C = catalog.size
    default_log_probs = catalog.known_uniform_log_prior()
    evidence_lookup = _build_detection_evidence_lookup(evidence_cache, catalog)

    T = np.full((C, C), transition_epsilon / max(C - 1, 1), dtype=np.float64)
    np.fill_diagonal(T, 1.0 - transition_epsilon)
    log_T = np.log(np.clip(T, 1e-300, None))

    df = trajectories_df.copy()
    df["IdentitySmoothedLabel"] = None
    df["IdentitySmoothedConf"] = np.nan
    df["IdentitySmoothedEntropy"] = np.nan
    df["IdentitySmoothedMargin"] = np.nan

    if "DetectionID" not in df.columns:
        return df

    traj_ids = df["TrajectoryID"].unique()

    for traj_id in traj_ids:
        traj_df = (
            df.loc[df["TrajectoryID"] == traj_id]
            .sort_values("FrameID")
            .copy()
        )
        if traj_df.empty:
            continue

        row_indices = list(traj_df.index)
        frame_log_probs: list[np.ndarray] = []
        for row in traj_df.itertuples():
            try:
                frame_idx = int(getattr(row, "FrameID"))
            except Exception:
                frame_log_probs.append(default_log_probs.copy())
                continue

            det_value = getattr(row, "DetectionID", np.nan)
            if pd.isna(det_value):
                frame_log_probs.append(default_log_probs.copy())
                continue

            try:
                det_id = int(det_value)
            except Exception:
                frame_log_probs.append(default_log_probs.copy())
                continue

            frame_log_probs.append(
                evidence_lookup.get((frame_idx, det_id), default_log_probs.copy())
            )

        # Forward pass
        alpha = catalog.known_uniform_log_prior()  # shape (C,)
        alphas: list[np.ndarray] = []
        for lp in frame_log_probs:
            # Predict: alpha_pred[j] = logsumexp_i (alpha[i] + log_T[i, j])
            pred = np.empty(C, dtype=np.float64)
            for j in range(C):
                pred[j] = np.logaddexp.reduce(alpha + log_T[:, j])
            # Update
            alpha = pred + lp
            alpha -= np.logaddexp.reduce(alpha)
            alphas.append(alpha.copy())

        # Backward pass: beta is the log-likelihood of observations *after* frame t
        beta = np.zeros(C, dtype=np.float64)  # uniform in log-space
        betas: list[np.ndarray] = [beta.copy()]
        for i in range(len(row_indices) - 1, 0, -1):
            lp = frame_log_probs[i]
            pred = np.empty(C, dtype=np.float64)
            for k in range(C):
                pred[k] = np.logaddexp.reduce(log_T[k, :] + beta + lp)
            beta = pred
            beta -= np.logaddexp.reduce(beta)
            betas.insert(0, beta.copy())

        # Smoothed = alpha * beta (normalised)
        for t, row_idx in enumerate(row_indices):
            smooth_log = alphas[t] + betas[t]
            smooth_log -= np.logaddexp.reduce(smooth_log)
            smooth_p = np.exp(smooth_log - smooth_log.max())
            smooth_p /= smooth_p.sum()

            known_p = smooth_p[1:]
            best_k = int(np.argmax(known_p))
            best_idx = best_k + 1  # catalog index
            best_label = catalog.label_of(best_idx)
            best_conf = float(smooth_p[best_idx])
            ent = float(-np.sum(smooth_p * np.log(np.clip(smooth_p, 1e-300, None))))
            if len(known_p) >= 2:
                top2 = np.partition(known_p, -2)[-2:]
                margin = float(top2[1] - top2[0])
            elif len(known_p) == 1:
                margin = float(known_p[0])
            else:
                margin = 0.0

            df.at[row_idx, "IdentitySmoothedLabel"] = best_label
            df.at[row_idx, "IdentitySmoothedConf"] = best_conf
            df.at[row_idx, "IdentitySmoothedEntropy"] = ent
            df.at[row_idx, "IdentitySmoothedMargin"] = margin

    return df


def _posterior_split_runs(
    traj_df: pd.DataFrame,
    *,
    min_conf: float,
    min_margin: float,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    sorted_df = traj_df.sort_values("FrameID", kind="stable")
    for row in sorted_df.itertuples():
        label = getattr(row, "IdentitySmoothedLabel", None)
        conf = float(pd.to_numeric(getattr(row, "IdentitySmoothedConf", np.nan), errors="coerce"))
        margin = float(pd.to_numeric(getattr(row, "IdentitySmoothedMargin", np.nan), errors="coerce"))
        stable_label = None
        if label is not None and not pd.isna(label):
            if np.isfinite(conf) and conf >= min_conf and np.isfinite(margin) and margin >= min_margin:
                stable_label = str(label)
        frame_id = int(getattr(row, "FrameID"))
        if not runs or runs[-1]["label"] != stable_label:
            runs.append(
                {
                    "label": stable_label,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "length": 1,
                }
            )
            continue
        runs[-1]["end_frame"] = frame_id
        runs[-1]["length"] += 1
    return runs


def _posterior_split_frames(
    traj_df: pd.DataFrame,
    params: dict[str, Any],
) -> list[int]:
    min_conf = float(params.get("IDENTITY_OFFLINE_SPLIT_MIN_CONF", 0.75))
    min_margin = float(params.get("IDENTITY_OFFLINE_SPLIT_MIN_MARGIN", 0.2))
    min_seg_frames = int(params.get("IDENTITY_OFFLINE_SPLIT_MIN_FRAMES", 3))
    max_bridge_frames = int(params.get("IDENTITY_OFFLINE_SPLIT_MAX_BRIDGE_FRAMES", 6))
    runs = _posterior_split_runs(
        traj_df,
        min_conf=min_conf,
        min_margin=min_margin,
    )
    if len(runs) < 2:
        return []

    stable_positions = [idx for idx, run in enumerate(runs) if run["label"] is not None]
    split_frames: list[int] = []
    for left_pos, right_pos in zip(stable_positions[:-1], stable_positions[1:]):
        left_run = runs[left_pos]
        right_run = runs[right_pos]
        if left_run["label"] == right_run["label"]:
            continue
        if left_run["length"] < min_seg_frames or right_run["length"] < min_seg_frames:
            continue
        bridge_frames = sum(runs[pos]["length"] for pos in range(left_pos + 1, right_pos))
        if bridge_frames > max_bridge_frames:
            continue
        split_frames.append(int(right_run["start_frame"]))
    return sorted(set(split_frames))


def _split_dataframe_by_frames(
    traj_df: pd.DataFrame,
    split_frames: list[int],
) -> list[pd.DataFrame]:
    if not split_frames:
        return [traj_df.copy()]
    ordered = traj_df.sort_values("FrameID", kind="stable").copy()
    start_frame = int(ordered["FrameID"].min())
    end_frame = int(ordered["FrameID"].max())
    boundaries = [frame for frame in split_frames if start_frame < frame <= end_frame]
    prev_frame = start_frame
    parts: list[pd.DataFrame] = []
    for boundary in boundaries:
        part = ordered[
            (ordered["FrameID"] >= prev_frame) & (ordered["FrameID"] < boundary)
        ].copy()
        if not part.empty:
            parts.append(part)
        prev_frame = boundary
    tail = ordered[ordered["FrameID"] >= prev_frame].copy()
    if not tail.empty:
        parts.append(tail)
    return parts or [ordered]


def _next_numeric_trajectory_id(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return 0
    return int(finite.max()) + 1


def split_mixed_identity_trajectories(
    smoothed_df: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Split trajectories when the smoothed identity posterior shows a sustained switch.

    This runs before fragment construction so stitched trajectories do not force
    the fragment assigner to explain incompatible identity regimes inside one
    trajectory.
    """
    if smoothed_df.empty:
        return smoothed_df.copy()
    required_cols = {
        "TrajectoryID",
        "FrameID",
        "IdentitySmoothedLabel",
        "IdentitySmoothedConf",
    }
    if not required_cols.issubset(smoothed_df.columns):
        return smoothed_df.copy()

    out_parts: list[pd.DataFrame] = []
    next_traj_id = _next_numeric_trajectory_id(smoothed_df["TrajectoryID"])
    split_count = 0

    for traj_id, traj_df in smoothed_df.groupby("TrajectoryID", sort=False):
        split_frames = _posterior_split_frames(traj_df, params)
        parts = _split_dataframe_by_frames(traj_df, split_frames)
        if len(parts) > 1:
            split_count += len(parts) - 1
        for part_index, part in enumerate(parts):
            updated = part.copy()
            updated["OriginalTrajectoryID"] = updated.get(
                "OriginalTrajectoryID",
                pd.Series(index=updated.index, dtype=object),
            )
            updated["OriginalTrajectoryID"] = updated["OriginalTrajectoryID"].where(
                updated["OriginalTrajectoryID"].notna(),
                traj_id,
            )
            if part_index == 0:
                updated["TrajectoryID"] = traj_id
            else:
                updated["TrajectoryID"] = next_traj_id
                next_traj_id += 1
            out_parts.append(updated)

    if not out_parts:
        return smoothed_df.copy()

    out = pd.concat(out_parts, ignore_index=True, sort=False)
    out = out.sort_values(["TrajectoryID", "FrameID"], kind="stable").reset_index(drop=True)
    if split_count > 0:
        log.info("Split %d mixed-identity trajectory segment(s) before fragment assignment.", split_count)
    return out


# ---------------------------------------------------------------------------
# Phase 4 – fragment segmentation
# ---------------------------------------------------------------------------


def build_identity_fragments(
    smoothed_df: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Segment trajectories into fragments with stable smoothed identity.

    A new fragment begins whenever the smoothed label changes or the smoothed
    confidence drops below ``IDENTITY_OFFLINE_FRAGMENT_MIN_CONF``.

    Parameters
    ----------
    smoothed_df:
        DataFrame produced by ``smooth_trajectory_identity_posteriors``.
    params:
        Runtime parameter dict.  Reads:
        ``IDENTITY_OFFLINE_FRAGMENT_MIN_FRAMES`` (int, default 10),
        ``IDENTITY_OFFLINE_FRAGMENT_MIN_CONF`` (float, default 0.5).

    Returns
    -------
    pd.DataFrame
        Columns: ``FragmentID``, ``TrajectoryID``, ``StartFrame``,
        ``EndFrame``, ``DominantLabel``, ``FragmentConf``,
        ``FragmentLength``.
    """
    min_frames = int(params.get("IDENTITY_OFFLINE_FRAGMENT_MIN_FRAMES", 10))
    min_conf = float(params.get("IDENTITY_OFFLINE_FRAGMENT_MIN_CONF", 0.5))

    rows: list[dict] = []
    frag_id = 0

    for traj_id, traj_df in smoothed_df.groupby("TrajectoryID"):
        sorted_df = traj_df.sort_values("FrameID").reset_index(drop=True)

        cur_label: Optional[str] = None
        cur_confs: list[float] = []
        cur_frames: list[int] = []

        for _, row in sorted_df.iterrows():
            label = row.get("IdentitySmoothedLabel")
            conf_val = row.get("IdentitySmoothedConf", 0.0)
            fi = int(row["FrameID"])
            conf = 0.0 if (conf_val is None or (isinstance(conf_val, float) and np.isnan(conf_val))) else float(conf_val)

            label_changed = label != cur_label
            below_threshold = conf < min_conf

            if label_changed or below_threshold:
                # Flush accumulated fragment
                if cur_label is not None and len(cur_frames) >= min_frames:
                    fragment_slice = sorted_df[
                        (sorted_df["FrameID"] >= cur_frames[0])
                        & (sorted_df["FrameID"] <= cur_frames[-1])
                    ]
                    rows.append(
                        {
                            "FragmentID": frag_id,
                            "TrajectoryID": traj_id,
                            "StartFrame": cur_frames[0],
                            "EndFrame": cur_frames[-1],
                            "DominantLabel": cur_label,
                            "FragmentConf": float(np.mean(cur_confs)),
                            "FragmentLength": len(cur_frames),
                            "LabelScores": _fragment_label_scores(fragment_slice),
                        }
                    )
                    frag_id += 1
                # Start new fragment only if above threshold
                cur_label = label if not below_threshold else None
                cur_confs = [conf] if (not below_threshold and label is not None) else []
                cur_frames = [fi] if (not below_threshold and label is not None) else []
            else:
                cur_confs.append(conf)
                cur_frames.append(fi)

        # Flush final fragment
        if cur_label is not None and len(cur_frames) >= min_frames:
            fragment_slice = sorted_df[
                (sorted_df["FrameID"] >= cur_frames[0])
                & (sorted_df["FrameID"] <= cur_frames[-1])
            ]
            rows.append(
                {
                    "FragmentID": frag_id,
                    "TrajectoryID": traj_id,
                    "StartFrame": cur_frames[0],
                    "EndFrame": cur_frames[-1],
                    "DominantLabel": cur_label,
                    "FragmentConf": float(np.mean(cur_confs)),
                    "FragmentLength": len(cur_frames),
                    "LabelScores": _fragment_label_scores(fragment_slice),
                }
            )
            frag_id += 1

    if not rows:
        return pd.DataFrame(
            columns=[
                "FragmentID",
                "TrajectoryID",
                "StartFrame",
                "EndFrame",
                "DominantLabel",
                "FragmentConf",
                "FragmentLength",
                "LabelScores",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 4 – global fragment assignment
# ---------------------------------------------------------------------------


def solve_fragment_identity_assignment(
    fragments_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Solve a globally unique assignment of labels to fragments.

    Two overlapping fragments (concurrent time range, different trajectories)
    cannot share the same known identity.

    Uses a global integer linear program over all fragment/label candidates,
    with a fallback to the legacy component-wise search only if MILP support is
    unavailable.

    Parameters
    ----------
    fragments_df:
        Output of ``build_identity_fragments``.
    catalog:
        Identity catalog.
    params:
        Runtime parameter dict.  Reads ``IDENTITY_OFFLINE_AMBIGUITY_MARGIN``.

    Returns
    -------
    pd.DataFrame
        Copy of *fragments_df* with ``AssignedLabel`` column added.
    """
    if fragments_df.empty:
        df = fragments_df.copy()
        df["AssignedLabel"] = None
        return df

    df = fragments_df.copy().reset_index(drop=True)
    df["AssignedLabel"] = None

    ambiguity_margin = float(params.get("IDENTITY_OFFLINE_AMBIGUITY_MARGIN", 0.15))
    candidate_options: dict[int, list[tuple[str | None, float]]] = {}
    for idx, row in df.iterrows():
        candidates = _candidate_labels_for_fragment(row, catalog)
        if candidates:
            best_score = float(candidates[0][1])
            candidates = [
                (label, float(score))
                for label, score in candidates
                if best_score - float(score) <= ambiguity_margin
            ]
        candidate_options[idx] = [(None, 0.0), *candidates]

    overlap_pairs: list[tuple[int, int]] = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if _fragments_overlap(df.iloc[i], df.iloc[j]):
                overlap_pairs.append((i, j))

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp

        var_specs: list[tuple[int, str | None, float]] = []
        for frag_idx, options in candidate_options.items():
            for label, score in options:
                var_specs.append((frag_idx, label, float(score)))

        if not var_specs:
            return df.sort_values("FragmentID").reset_index(drop=True)

        n_vars = len(var_specs)
        frag_to_vars: dict[int, list[int]] = {}
        label_to_vars: dict[str, list[int]] = {}
        for var_idx, (frag_idx, label, _score) in enumerate(var_specs):
            frag_to_vars.setdefault(frag_idx, []).append(var_idx)
            if label is not None:
                label_to_vars.setdefault(label, []).append(var_idx)

        c = np.asarray([-score for _frag_idx, _label, score in var_specs], dtype=np.float64)
        integrality = np.ones(n_vars, dtype=np.int8)
        bounds = Bounds(np.zeros(n_vars, dtype=np.float64), np.ones(n_vars, dtype=np.float64))

        constraint_rows: list[np.ndarray] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        for frag_idx, variable_indices in frag_to_vars.items():
            row = np.zeros(n_vars, dtype=np.float64)
            row[variable_indices] = 1.0
            constraint_rows.append(row)
            lower_bounds.append(1.0)
            upper_bounds.append(1.0)

        overlap_lookup = {tuple(sorted(pair)) for pair in overlap_pairs}
        for label, variable_indices in label_to_vars.items():
            if len(variable_indices) < 2:
                continue
            for left_idx, left_var in enumerate(variable_indices[:-1]):
                left_frag = var_specs[left_var][0]
                for right_var in variable_indices[left_idx + 1 :]:
                    right_frag = var_specs[right_var][0]
                    if tuple(sorted((left_frag, right_frag))) not in overlap_lookup:
                        continue
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[left_var] = 1.0
                    row[right_var] = 1.0
                    constraint_rows.append(row)
                    lower_bounds.append(-np.inf)
                    upper_bounds.append(1.0)

        constraints = []
        if constraint_rows:
            A = np.vstack(constraint_rows)
            constraints.append(
                LinearConstraint(
                    A,
                    np.asarray(lower_bounds, dtype=np.float64),
                    np.asarray(upper_bounds, dtype=np.float64),
                )
            )

        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": float(params.get("IDENTITY_OFFLINE_ILP_TIME_LIMIT", 30.0)),
                "mip_rel_gap": float(params.get("IDENTITY_OFFLINE_ILP_REL_GAP", 1e-6)),
            },
        )
        if result.success and result.x is not None:
            chosen = np.asarray(result.x, dtype=np.float64) > 0.5
            for var_idx, is_selected in enumerate(chosen.tolist()):
                if not is_selected:
                    continue
                frag_idx, label, _score = var_specs[var_idx]
                df.at[frag_idx, "AssignedLabel"] = label
            return df.sort_values("FragmentID").reset_index(drop=True)
        log.warning(
            "Global identity MILP did not converge (%s); falling back to legacy solver.",
            getattr(result, "message", "unknown failure"),
        )
    except Exception:
        log.debug("Global identity MILP unavailable; falling back to legacy solver.", exc_info=True)

    return _solve_fragment_identity_assignment_legacy(df, candidate_options)


def _solve_fragment_identity_assignment_legacy(
    df: pd.DataFrame,
    candidate_options: dict[int, list[tuple[str | None, float]]],
) -> pd.DataFrame:
    """Component-wise exact search with greedy fallback used as a MILP fallback."""
    max_exact_component = 14
    n = len(df)
    overlap = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if _fragments_overlap(df.iloc[i], df.iloc[j]):
                overlap[i, j] = True
                overlap[j, i] = True

    visited: set[int] = set()

    def _component_nodes(start_idx: int) -> list[int]:
        stack = [start_idx]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            neighbors = np.where(overlap[node])[0].tolist()
            stack.extend(neighbors)
        return component

    def _solve_component_greedy(component: list[int]) -> dict[int, str | None]:
        assigned: dict[int, str | None] = {}
        claimed: dict[str, list[int]] = {}
        ordered = sorted(
            component,
            key=lambda idx: float(df.iloc[idx].get("FragmentConf", 0.0)),
            reverse=True,
        )
        for idx in ordered:
            label = None
            for candidate, _score in candidate_options[idx][1:]:
                if candidate is None:
                    continue
                if any(overlap[idx, other_idx] for other_idx in claimed.get(candidate, [])):
                    continue
                label = candidate
                claimed.setdefault(candidate, []).append(idx)
                break
            assigned[idx] = label
        return assigned

    def _solve_component_exact(component: list[int]) -> dict[int, str | None]:
        order = sorted(
            component,
            key=lambda idx: (
                int(np.count_nonzero(overlap[idx, component])),
                float(df.iloc[idx].get("FragmentConf", 0.0)),
            ),
            reverse=True,
        )
        best_score = -np.inf
        best_assignment: dict[int, str | None] = {idx: None for idx in component}
        max_remaining = [
            max((score for _label, score in candidate_options[idx]), default=0.0)
            for idx in order
        ]

        def _search(pos: int, score: float, current: dict[int, str | None]) -> None:
            nonlocal best_score, best_assignment
            if pos >= len(order):
                if score > best_score:
                    best_score = score
                    best_assignment = dict(current)
                return

            bound = score + sum(max_remaining[pos:])
            if bound < best_score:
                return

            idx = order[pos]
            for label, label_score in sorted(
                candidate_options[idx], key=lambda item: float(item[1]), reverse=True
            ):
                if label is not None:
                    conflict = any(
                        other_label == label and overlap[idx, other_idx]
                        for other_idx, other_label in current.items()
                    )
                    if conflict:
                        continue
                current[idx] = label
                _search(pos + 1, score + float(label_score), current)
                current.pop(idx, None)

        _search(0, 0.0, {})
        return best_assignment

    for idx in range(n):
        if idx in visited:
            continue
        component = _component_nodes(idx)
        if len(component) > max_exact_component:
            component_assignment = _solve_component_greedy(component)
        else:
            component_assignment = _solve_component_exact(component)
        for comp_idx, label in component_assignment.items():
            df.at[comp_idx, "AssignedLabel"] = label

    return df.sort_values("FragmentID").reset_index(drop=True)


def _candidate_labels_for_fragment(
    row: pd.Series,
    catalog: IdentityCatalog,
) -> list[tuple[str, float]]:
    label_scores = row.get("LabelScores")
    if isinstance(label_scores, dict) and label_scores:
        candidates = []
        for label, score in label_scores.items():
            label_text = str(label)
            if not catalog.contains(label_text):
                continue
            if catalog.is_unknown(catalog.index_of(label_text)):
                continue
            candidates.append((label_text, float(score)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if candidates:
            return candidates

    label = row.get("DominantLabel")
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return []
    label_text = str(label)
    if not catalog.contains(label_text):
        return []
    if catalog.is_unknown(catalog.index_of(label_text)):
        return []
    return [(label_text, float(row.get("FragmentConf", 0.0)))]


def run_identity_residual_assignment(
    fragments_df: pd.DataFrame,
    params: dict[str, Any],
    catalog: IdentityCatalog | None = None,
) -> pd.DataFrame:
    """Assign identities to fragments left unassigned in the primary pass.

    Secondary pass: for each unassigned fragment, try lower-ranked candidate
    labels derived from its in-fragment smoothed label scores while respecting
    temporal overlap constraints with already-assigned fragments.

    Returns
    -------
    pd.DataFrame
        Copy of *fragments_df* with additional assignments filled when a
        conflict-free alternative label exists.
    """
    df = fragments_df.copy()
    if df.empty or "AssignedLabel" not in df.columns:
        return df

    if catalog is None:
        log.debug("Residual identity pass skipped: no catalog provided.")
        return df

    ambiguity_margin = float(
        params.get("IDENTITY_OFFLINE_AMBIGUITY_MARGIN", 0.15)
    )

    for idx, row in df[df["AssignedLabel"].isna()].iterrows():
        candidates = _candidate_labels_for_fragment(row, catalog)
        if not candidates:
            continue
        best_score = float(candidates[0][1])
        overlapping = df[
            (df.index != idx)
            & df["AssignedLabel"].notna()
            & df.apply(lambda other: _fragments_overlap(row, other), axis=1)
        ]
        used_labels = {str(label) for label in overlapping["AssignedLabel"].tolist()}

        for label, score in candidates:
            if best_score - float(score) > ambiguity_margin:
                continue
            if label in used_labels:
                continue
            df.at[idx, "AssignedLabel"] = label
            break

    unassigned_count = int(df["AssignedLabel"].isna().sum())
    if unassigned_count:
        log.debug(
            "%d fragment(s) remain unassigned after residual identity pass",
            unassigned_count,
        )
    return df
