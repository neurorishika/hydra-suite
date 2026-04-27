"""Identity-aware post-processing for stable trajectory repair.

This stage runs on the rich, augmented trajectory dataframe after detected and
interpolated identity signals have been merged in. It performs three tasks:

1. Annotate per-row AprilTag observations from the detected tag cache.
2. Split trajectories when stable unique-identity evidence changes.
3. Re-chain resulting fragments by compatible unique identities, optionally
   filling short gaps with synthetic rows.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

IDENTITY_AUDIT_COLUMNS = [
    "DetectedTagID",
    "DetectedTagLabel",
    "DetectedTagHamming",
    "DetectedTagConf",
    "OriginalTrajectoryID",
    "IdentityFragmentID",
    "UniqueIdentityKey",
    "UniqueIdentityConfidence",
    "UniqueIdentitySources",
    "UniqueIdentitySourceCount",
    "IdentityInterpolated",
]

_KEY_SEP = "|"
_PAIR_SEP = "="
_TAG_SOURCE = "apriltag"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_string(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _safe_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(round(number))


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    return float(np.clip(_safe_float(value, default), 0.0, 1.0))


def _ensure_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in IDENTITY_AUDIT_COLUMNS:
        if col == "IdentityInterpolated":
            if col not in out.columns:
                out[col] = False
        elif col not in out.columns:
            out[col] = np.nan
    return out


def _apply_offline_identity_compat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Back-fill legacy identity audit columns from offline decoder outputs."""
    out = _ensure_identity_columns(df)
    if "OriginalTrajectoryID" in out.columns and "TrajectoryID" in out.columns:
        out["OriginalTrajectoryID"] = out["OriginalTrajectoryID"].where(
            out["OriginalTrajectoryID"].notna(),
            out["TrajectoryID"],
        )
    if "IdentityFragmentID" in out.columns and "TrajectoryID" in out.columns:
        out["IdentityFragmentID"] = out["IdentityFragmentID"].where(
            out["IdentityFragmentID"].notna(),
            out["TrajectoryID"],
        )

    assigned = out.get("IdentityAssignedLabel")
    if assigned is None:
        assigned = out.get("IdentityOfflineLabel")
    if assigned is None:
        assigned = pd.Series(index=out.index, dtype=object)

    assigned_conf = pd.to_numeric(
        out.get("IdentityAssignedConfidence", out.get("IdentitySmoothedConf")),
        errors="coerce",
    )

    keys = []
    sources = []
    source_counts = []
    for label in assigned:
        if _is_missing(label):
            keys.append(np.nan)
            sources.append(np.nan)
            source_counts.append(0)
            continue
        source_map = {"offline": str(label)}
        keys.append(format_identity_key(source_map) or np.nan)
        sources.append("offline")
        source_counts.append(1)

    out["UniqueIdentityKey"] = keys
    out["UniqueIdentitySources"] = sources
    out["UniqueIdentitySourceCount"] = source_counts
    out["UniqueIdentityConfidence"] = assigned_conf
    out["IdentityInterpolated"] = (
        out["IdentityInterpolated"] if "IdentityInterpolated" in out.columns else False
    )
    return out


def format_identity_key(sources: dict[str, str]) -> str:
    """Serialize a source-keyed identity dict into a stable string."""
    if not sources:
        return ""
    parts = []
    for source in sorted(sources):
        value = _normalize_string(sources[source])
        if not value:
            continue
        parts.append(f"{source}{_PAIR_SEP}{value}")
    return _KEY_SEP.join(parts)


def parse_identity_key(identity_key: Any) -> dict[str, str]:
    """Parse a serialized identity key back into a source-keyed dict."""
    token = _normalize_string(identity_key)
    if not token:
        return {}
    parsed: dict[str, str] = {}
    for item in token.split(_KEY_SEP):
        if _PAIR_SEP not in item:
            continue
        source, value = item.split(_PAIR_SEP, 1)
        source = _normalize_string(source)
        value = _normalize_string(value)
        if source and value:
            parsed[source] = value
    return parsed


def _parse_cnn_factor_source(source: str) -> tuple[str, str] | None:
    token = _normalize_string(source)
    if not token.startswith("cnn:"):
        return None
    parts = token.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def _partition_identity_sources(
    sources: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    direct_sources: dict[str, str] = {}
    grouped_sources: dict[str, dict[str, str]] = defaultdict(dict)
    for source, value in sources.items():
        parsed = _parse_cnn_factor_source(source)
        if parsed is None:
            direct_sources[source] = value
            continue
        label, factor = parsed
        grouped_sources[f"cnn:{label}"][factor] = value
    return direct_sources, dict(grouped_sources)


def _compare_identity_sources(
    lhs: dict[str, str],
    rhs: dict[str, str],
) -> dict[str, Any]:
    lhs_direct, lhs_grouped = _partition_identity_sources(lhs)
    rhs_direct, rhs_grouped = _partition_identity_sources(rhs)

    shared_direct = set(lhs_direct).intersection(rhs_direct)
    direct_agreements = sum(
        1 for source in shared_direct if lhs_direct[source] == rhs_direct[source]
    )
    direct_conflicts = len(shared_direct) - direct_agreements

    grouped_results: list[tuple[int, int]] = []
    for group_key in set(lhs_grouped).intersection(rhs_grouped):
        shared_factors = set(lhs_grouped[group_key]).intersection(
            rhs_grouped[group_key]
        )
        if not shared_factors:
            continue
        agreements = sum(
            1
            for factor in shared_factors
            if lhs_grouped[group_key][factor] == rhs_grouped[group_key][factor]
        )
        conflicts = len(shared_factors) - agreements
        grouped_results.append((agreements, conflicts))

    return {
        "direct_agreements": direct_agreements,
        "direct_conflicts": direct_conflicts,
        "grouped_results": grouped_results,
        "has_shared": bool(shared_direct or grouped_results),
    }


def _shared_identity_sources(lhs: dict[str, str], rhs: dict[str, str]) -> set[str]:
    return set(lhs).intersection(rhs)


def identity_sources_compatible(lhs: dict[str, str], rhs: dict[str, str]) -> bool:
    """Return True when all overlapping identity sources agree."""
    comparison = _compare_identity_sources(lhs, rhs)
    if not comparison["has_shared"]:
        return False
    if comparison["direct_conflicts"] > 0:
        return False
    has_agreement = comparison["direct_agreements"] > 0
    for agreements, conflicts in comparison["grouped_results"]:
        if conflicts > agreements:
            return False
        if agreements > conflicts and agreements > 0:
            has_agreement = True
    return has_agreement


def identity_sources_conflict(lhs: dict[str, str], rhs: dict[str, str]) -> bool:
    """Return True when overlapping identity sources disagree."""
    comparison = _compare_identity_sources(lhs, rhs)
    if not comparison["has_shared"]:
        return False
    if comparison["direct_conflicts"] > 0:
        return True
    return any(
        conflicts > agreements
        for agreements, conflicts in comparison["grouped_results"]
    )


def _cnn_column_specs(
    df: pd.DataFrame, label: str
) -> list[tuple[str | None, str, str]]:
    single_class_col = f"CNN_{label}_Class"
    single_conf_col = f"CNN_{label}_Conf"
    if single_class_col in df.columns or single_conf_col in df.columns:
        return [(None, single_class_col, single_conf_col)]

    pattern = re.compile(rf"^CNN_{re.escape(label)}_(.+)_Class$")
    specs = []
    for col in df.columns:
        match = pattern.match(str(col))
        if match is None:
            continue
        factor = match.group(1)
        specs.append((factor, str(col), f"CNN_{label}_{factor}_Conf"))
    specs.sort(key=lambda item: item[0] or "")
    return specs


def _unique_cnn_sources(
    trajectories_df: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    configs = params.get("CNN_CLASSIFIERS", []) or []
    unique_sources: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        label = _normalize_string(cfg.get("label") or "cnn_identity")
        if not label or not bool(cfg.get("unique_identifier", False)):
            continue
        unique_sources[label] = {
            "confidence": float(cfg.get("confidence", 0.5)),
            "scoring_mode": _normalize_string(cfg.get("scoring_mode") or "atomic"),
            "specs": _cnn_column_specs(trajectories_df, label),
        }
    return unique_sources


def _row_identity_sources(
    row: pd.Series,
    unique_cnn_sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
    sources: dict[str, str] = {}
    confidences: dict[str, float] = {}
    weights: dict[str, float] = {}

    detected_tag = _safe_int(row.get("DetectedTagID"))
    interp_tag = _safe_int(row.get("InterpTagID"))
    if detected_tag is not None and detected_tag >= 0:
        sources[_TAG_SOURCE] = str(detected_tag)
        conf = _clamp_confidence(row.get("DetectedTagConf"), 1.0)
        confidences[_TAG_SOURCE] = conf
        weights[_TAG_SOURCE] = max(0.85, conf)
    elif interp_tag is not None and interp_tag >= 0:
        sources[_TAG_SOURCE] = str(interp_tag)
        conf = _clamp_confidence(row.get("InterpTagConf"), 0.65)
        confidences[_TAG_SOURCE] = conf
        weights[_TAG_SOURCE] = 0.75 * max(0.35, conf)

    for label, cfg in unique_cnn_sources.items():
        specs = list(cfg.get("specs") or [])
        if not specs:
            continue
        threshold = float(cfg.get("confidence", 0.5))
        scoring_mode = _normalize_string(cfg.get("scoring_mode") or "atomic")
        if len(specs) > 1 and scoring_mode == "per_head_average":
            for factor, class_col, conf_col in specs:
                class_name = _normalize_string(row.get(class_col))
                conf = _safe_float(row.get(conf_col), math.nan)
                if not class_name or not np.isfinite(conf) or conf < threshold:
                    continue
                factor_token = _normalize_string(factor) or "flat"
                source_name = f"cnn:{label}:{factor_token}"
                confidence = float(np.clip(conf, 0.0, 1.0))
                sources[source_name] = class_name
                confidences[source_name] = confidence
                weights[source_name] = confidence
            continue

        classes = []
        per_factor_conf = []
        valid = True
        for factor, class_col, conf_col in specs:
            class_name = _normalize_string(row.get(class_col))
            conf = _safe_float(row.get(conf_col), math.nan)
            if not class_name or not np.isfinite(conf) or conf < threshold:
                valid = False
                break
            if factor is None:
                classes.append(class_name)
            else:
                classes.append(f"{factor}:{class_name}")
            per_factor_conf.append(float(conf))
        if not valid or not classes:
            continue
        source_name = f"cnn:{label}"
        sources[source_name] = "+".join(classes)
        confidence = float(np.clip(min(per_factor_conf), 0.0, 1.0))
        confidences[source_name] = confidence
        weights[source_name] = confidence

    return sources, confidences, weights


def _aggregate_identity_evidence(
    evidences: list[dict[str, Any]],
    *,
    min_weight: float = 0.8,
    dominance_ratio: float = 0.6,
) -> tuple[dict[str, str], dict[str, float]]:
    source_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for evidence in evidences:
        for source, value in evidence["sources"].items():
            source_votes[source][value] += float(
                evidence["weights"].get(
                    source, evidence["confidences"].get(source, 0.0)
                )
            )

    aggregated: dict[str, str] = {}
    confidences: dict[str, float] = {}
    for source, votes in source_votes.items():
        if not votes:
            continue
        total = float(sum(votes.values()))
        value, top_weight = max(votes.items(), key=lambda item: item[1])
        if total < min_weight:
            continue
        dominance = float(top_weight / max(total, 1e-6))
        if dominance < dominance_ratio:
            continue
        aggregated[source] = value
        confidences[source] = float(np.clip(dominance * min(1.0, total), 0.0, 1.0))
    return aggregated, confidences


def _build_identity_observations(
    fragment_df: pd.DataFrame,
    unique_cnn_sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    observations = []
    for row_index, row in fragment_df.iterrows():
        sources, confidences, weights = _row_identity_sources(row, unique_cnn_sources)
        if not sources:
            continue
        observations.append(
            {
                "row_index": row_index,
                "frame_id": int(row["FrameID"]),
                "sources": sources,
                "confidences": confidences,
                "weights": weights,
            }
        )
    return observations


def _identity_support(evidences: list[dict[str, Any]]) -> float:
    return float(
        sum(
            sum(float(weight) for weight in evidence["weights"].values())
            for evidence in evidences
        )
    )


def _stable_identity_groups(
    observations: list[dict[str, Any]],
    *,
    min_streak: int = 2,
    min_group_weight: float = 1.1,
) -> list[list[dict[str, Any]]]:
    if not observations:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [observations[0]]
    pending: list[dict[str, Any]] = []

    for observation in observations[1:]:
        current_identity, _ = _aggregate_identity_evidence(current, min_weight=0.0)
        if current_identity and identity_sources_compatible(
            current_identity, observation["sources"]
        ):
            if pending:
                current.extend(pending)
                pending = []
            current.append(observation)
            continue

        if current_identity and identity_sources_conflict(
            current_identity, observation["sources"]
        ):
            pending.append(observation)
            pending_identity, _ = _aggregate_identity_evidence(pending, min_weight=0.0)
            if (
                pending_identity
                and len(pending) >= min_streak
                and _identity_support(pending) >= min_group_weight
                and identity_sources_conflict(current_identity, pending_identity)
            ):
                groups.append(current)
                current = list(pending)
                pending = []
            continue

        if pending:
            current.extend(pending)
            pending = []
        current.append(observation)

    if pending:
        current.extend(pending)
    groups.append(current)
    return groups


def _trajectory_split_frames(
    traj_df: pd.DataFrame,
    unique_cnn_sources: dict[str, dict[str, Any]],
) -> list[int]:
    observations = _build_identity_observations(traj_df, unique_cnn_sources)
    if len(observations) < 2:
        return []

    split_frames: list[int] = []
    groups = _stable_identity_groups(observations)
    if len(groups) < 2:
        return split_frames

    for left_group, right_group in zip(groups[:-1], groups[1:]):
        left_identity, left_conf = _aggregate_identity_evidence(left_group)
        right_identity, right_conf = _aggregate_identity_evidence(right_group)
        if not left_identity or not right_identity:
            continue
        if not identity_sources_conflict(left_identity, right_identity):
            continue
        if max(left_conf.values(), default=0.0) < 0.65:
            continue
        if max(right_conf.values(), default=0.0) < 0.65:
            continue
        split_frame = int(right_group[0]["frame_id"])
        if split_frame not in split_frames:
            split_frames.append(split_frame)
    return sorted(split_frames)


def _split_trajectory_dataframe(
    traj_df: pd.DataFrame,
    split_frames: list[int],
) -> list[pd.DataFrame]:
    if not split_frames:
        return [traj_df.copy()]
    remaining = traj_df.sort_values("FrameID", kind="stable").copy()
    parts: list[pd.DataFrame] = []
    start_frame = int(remaining["FrameID"].min())
    end_frame = int(remaining["FrameID"].max())
    boundaries = [frame for frame in split_frames if start_frame < frame <= end_frame]
    prev_frame = start_frame
    for boundary in boundaries:
        part = remaining[
            (remaining["FrameID"] >= prev_frame) & (remaining["FrameID"] < boundary)
        ].copy()
        if not part.empty:
            parts.append(part)
        prev_frame = boundary
    tail = remaining[remaining["FrameID"] >= prev_frame].copy()
    if not tail.empty:
        parts.append(tail)
    return parts or [traj_df.copy()]


def _fragment_summary(
    fragment_id: int,
    fragment_df: pd.DataFrame,
    unique_cnn_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    observations = _build_identity_observations(fragment_df, unique_cnn_sources)
    identity_sources, identity_conf = _aggregate_identity_evidence(observations)
    valid_df = fragment_df[fragment_df["X"].notna() & fragment_df["Y"].notna()].copy()
    start_row = valid_df.iloc[0] if not valid_df.empty else fragment_df.iloc[0]
    end_row = valid_df.iloc[-1] if not valid_df.empty else fragment_df.iloc[-1]
    return {
        "fragment_id": fragment_id,
        "original_trajectory_id": int(fragment_df["TrajectoryID"].iloc[0]),
        "start_frame": int(fragment_df["FrameID"].min()),
        "end_frame": int(fragment_df["FrameID"].max()),
        "start_row": start_row,
        "end_row": end_row,
        "identity_sources": identity_sources,
        "identity_confidences": identity_conf,
        "identity_key": format_identity_key(identity_sources),
        "identity_source_count": len(identity_sources),
        "identity_confidence": (
            float(np.mean(list(identity_conf.values()))) if identity_conf else math.nan
        ),
        "df": fragment_df.copy(),
    }


def _merge_identity_sources(lhs: dict[str, str], rhs: dict[str, str]) -> dict[str, str]:
    if not lhs:
        return dict(rhs)
    if not rhs:
        return dict(lhs)
    merged = dict(lhs)
    for source, value in rhs.items():
        if source not in merged:
            merged[source] = value
    return merged


def _chain_assignment_score(
    chain: dict[str, Any],
    fragment: dict[str, Any],
) -> tuple[int, int, int]:
    shared = _shared_identity_sources(
        chain["identity_sources"], fragment["identity_sources"]
    )
    gap = int(fragment["start_frame"] - chain["fragments"][-1]["end_frame"] - 1)
    same_origin = int(
        fragment["original_trajectory_id"]
        == chain["fragments"][-1]["original_trajectory_id"]
    )
    return (len(shared), -max(gap, 0), same_origin)


def _build_identity_chains(
    fragments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for fragment in sorted(
        fragments, key=lambda item: (item["start_frame"], item["fragment_id"])
    ):
        identity_sources = fragment["identity_sources"]
        if not identity_sources:
            chains.append(
                {
                    "identity_sources": {},
                    "fragments": [fragment],
                }
            )
            continue

        best_index = None
        best_score = None
        for idx, chain in enumerate(chains):
            chain_identity = chain["identity_sources"]
            if not chain_identity:
                continue
            last_fragment = chain["fragments"][-1]
            if fragment["start_frame"] <= last_fragment["end_frame"]:
                continue
            if not identity_sources_compatible(chain_identity, identity_sources):
                continue
            score = _chain_assignment_score(chain, fragment)
            if best_score is None or score > best_score:
                best_index = idx
                best_score = score
        if best_index is None:
            chains.append(
                {
                    "identity_sources": dict(identity_sources),
                    "fragments": [fragment],
                }
            )
            continue
        chains[best_index]["fragments"].append(fragment)
        chains[best_index]["identity_sources"] = _merge_identity_sources(
            chains[best_index]["identity_sources"], identity_sources
        )
    return chains


def _circular_interp(theta_a: float, theta_b: float, alpha: float) -> float:
    if not np.isfinite(theta_a) or not np.isfinite(theta_b):
        return math.nan
    delta = ((theta_b - theta_a + math.pi) % (2.0 * math.pi)) - math.pi
    return float(theta_a + alpha * delta)


def _motion_allows_identity_fill(
    left_fragment: dict[str, Any],
    right_fragment: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    left_row = left_fragment["end_row"]
    right_row = right_fragment["start_row"]
    left_x = _safe_float(left_row.get("X"))
    left_y = _safe_float(left_row.get("Y"))
    right_x = _safe_float(right_row.get("X"))
    right_y = _safe_float(right_row.get("Y"))
    if not all(np.isfinite(v) for v in (left_x, left_y, right_x, right_y)):
        return False
    delta_frames = max(
        1, int(right_fragment["start_frame"] - left_fragment["end_frame"])
    )
    jump = float(math.hypot(right_x - left_x, right_y - left_y))
    max_velocity_break = float(params.get("MAX_VELOCITY_BREAK", 100.0))
    agreement_distance = float(params.get("AGREEMENT_DISTANCE", 15.0))
    return jump <= max(max_velocity_break * delta_frames, agreement_distance * 2.0)


def _identity_interpolation_rows(
    left_fragment: dict[str, Any],
    right_fragment: dict[str, Any],
    chain_identity_key: str,
    chain_identity_sources: dict[str, str],
) -> list[dict[str, Any]]:
    gap = int(right_fragment["start_frame"] - left_fragment["end_frame"] - 1)
    if gap <= 0:
        return []

    left_row = left_fragment["end_row"]
    right_row = right_fragment["start_row"]
    rows = []
    numeric_linear_cols = ["X", "Y", "Width", "Height"]
    all_columns = sorted(
        set(left_fragment["df"].columns).union(right_fragment["df"].columns)
    )
    for step in range(1, gap + 1):
        alpha = float(step / (gap + 1))
        frame_id = int(left_fragment["end_frame"] + step)
        row = {column: np.nan for column in all_columns}
        row["FrameID"] = frame_id
        row["TrajectoryID"] = left_fragment["original_trajectory_id"]
        row["State"] = _normalize_string(left_row.get("State")) or "active"
        row["IdentityInterpolated"] = True
        row["OriginalTrajectoryID"] = left_fragment["original_trajectory_id"]
        row["IdentityFragmentID"] = left_fragment["fragment_id"]
        row["UniqueIdentityKey"] = chain_identity_key or np.nan
        row["UniqueIdentitySources"] = (
            ", ".join(sorted(chain_identity_sources)) or np.nan
        )
        row["UniqueIdentitySourceCount"] = int(len(chain_identity_sources))
        row["UniqueIdentityConfidence"] = (
            float(
                np.mean(
                    list(left_fragment["identity_confidences"].values())
                    + list(right_fragment["identity_confidences"].values())
                )
            )
            if (
                left_fragment["identity_confidences"]
                or right_fragment["identity_confidences"]
            )
            else math.nan
        )
        for column in numeric_linear_cols:
            left_value = _safe_float(left_row.get(column))
            right_value = _safe_float(right_row.get(column))
            if np.isfinite(left_value) and np.isfinite(right_value):
                row[column] = float(left_value + alpha * (right_value - left_value))
        row["Theta"] = _circular_interp(
            _safe_float(left_row.get("Theta")),
            _safe_float(right_row.get("Theta")),
            alpha,
        )
        rows.append(row)
    return rows


def _apply_chain_identity_metadata(
    fragment: dict[str, Any],
    chain_identity_sources: dict[str, str],
) -> pd.DataFrame:
    part = _ensure_identity_columns(fragment["df"])
    part["OriginalTrajectoryID"] = fragment["original_trajectory_id"]
    part["IdentityFragmentID"] = fragment["fragment_id"]
    identity_key = format_identity_key(chain_identity_sources)
    part["UniqueIdentityKey"] = identity_key or np.nan
    part["UniqueIdentitySources"] = (
        ", ".join(sorted(chain_identity_sources)) if chain_identity_sources else np.nan
    )
    part["UniqueIdentitySourceCount"] = int(len(chain_identity_sources))
    part["UniqueIdentityConfidence"] = (
        fragment["identity_confidence"] if chain_identity_sources else math.nan
    )
    part["IdentityInterpolated"] = False
    return part


def apply_identity_postprocessing(
    trajectories_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Split and re-chain trajectories using stable unique-identity evidence.

    .. note::
        **Identity Phase 5 shim** — When ``ENABLE_IDENTITY_OFFLINE_DECODER`` is
        ``True`` the offline Bayesian decoder has already been applied upstream
        (in the tracking orchestrator's ``_apply_identity_postprocessing_to_df``
        method) and written ``IdentityOfflineLabel`` / ``IdentitySmoothedLabel``
        columns.  In that case this function skips the legacy heuristic
        split/re-chain pass to avoid double-processing.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df

    params = params or {}

    # === Identity Phase 5 compatibility shim ===
    # If the offline probabilistic decoder has already run, the dataframe
    # already carries ``IdentityOfflineLabel`` rows written by
    # ``apply_fragment_labels_to_trajectories``.  The legacy heuristic pass
    # below would overwrite those results, so we skip it.
    if str(params.get("IDENTITY_POSTPROCESS_MODE", "Heuristic")) == "Offline Decoder":
        if "IdentityOfflineLabel" in trajectories_df.columns:
            return _apply_offline_identity_compat_columns(trajectories_df)
        # Offline decoder was requested but did not produce output (evidence
        # cache missing?).  Fall through to heuristic processing so the user
        # still gets *some* identity assignment rather than an empty result.

    out = _ensure_identity_columns(trajectories_df)
    if "TrajectoryID" not in out.columns or "FrameID" not in out.columns:
        return out

    unique_cnn_sources = _unique_cnn_sources(out, params)
    use_apriltags = bool(params.get("USE_APRILTAGS", False))
    if not unique_cnn_sources and not use_apriltags:
        return out

    fragments: list[dict[str, Any]] = []
    next_fragment_id = 0
    for _traj_id, traj_df in out.groupby("TrajectoryID", sort=False):
        sorted_traj = traj_df.sort_values("FrameID", kind="stable").copy()
        split_frames = _trajectory_split_frames(sorted_traj, unique_cnn_sources)
        for part in _split_trajectory_dataframe(sorted_traj, split_frames):
            if part.empty:
                continue
            fragments.append(
                _fragment_summary(next_fragment_id, part, unique_cnn_sources)
            )
            next_fragment_id += 1

    if not fragments:
        return out

    chains = _build_identity_chains(fragments)
    max_interp_gap = int(max(0, params.get("IDENTITY_INTERPOLATION_MAX_GAP", 0)))

    result_parts: list[pd.DataFrame] = []
    next_traj_id = 0
    for chain in chains:
        chain_identity_sources = chain["identity_sources"]
        chain_identity_key = format_identity_key(chain_identity_sources)
        assembled_rows: list[pd.DataFrame] = []
        ordered_fragments = sorted(
            chain["fragments"],
            key=lambda item: (item["start_frame"], item["fragment_id"]),
        )

        def _flush_chain_rows() -> None:
            nonlocal assembled_rows, next_traj_id
            if not assembled_rows:
                return
            chain_df = pd.concat(assembled_rows, ignore_index=True, sort=False)
            chain_df = _ensure_identity_columns(chain_df)
            chain_df = chain_df.sort_values("FrameID", kind="stable").reset_index(
                drop=True
            )
            chain_df["TrajectoryID"] = next_traj_id
            result_parts.append(chain_df)
            next_traj_id += 1
            assembled_rows = []

        for frag_index, fragment in enumerate(ordered_fragments):
            part = _apply_chain_identity_metadata(fragment, chain_identity_sources)
            assembled_rows.append(part)

            bridge_to_next = False
            if frag_index == len(ordered_fragments) - 1:
                _flush_chain_rows()
                continue

            next_fragment = ordered_fragments[frag_index + 1]
            gap = int(next_fragment["start_frame"] - fragment["end_frame"] - 1)
            motion_ok = _motion_allows_identity_fill(fragment, next_fragment, params)

            if gap == 0 and motion_ok:
                bridge_to_next = True
            elif 0 < gap <= max_interp_gap and motion_ok:
                interp_rows = _identity_interpolation_rows(
                    fragment,
                    next_fragment,
                    chain_identity_key,
                    chain_identity_sources,
                )
                if interp_rows:
                    assembled_rows.append(pd.DataFrame(interp_rows))
                bridge_to_next = True

            if not bridge_to_next:
                _flush_chain_rows()

    if not result_parts:
        return out

    result = pd.concat(result_parts, ignore_index=True, sort=False)
    result = _ensure_identity_columns(result)
    result = result.sort_values(["TrajectoryID", "FrameID"], kind="stable")
    result = result.reset_index(drop=True)
    return result


_UNKNOWN_LABEL = "unknown"


def fill_identity_nans_with_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN identity columns using per-trajectory majority label.

    Strategy per column:
    - ``IdentityAssignedLabel``: trajectory consensus; ``"unknown"`` when the
      entire trajectory has no label evidence.
    - ``IdentityAssignedID``: catalog index inferred from existing label→ID
      pairs in the data; 0 for rows whose label resolved to ``"unknown"``.
    - ``IdentityAssignedConfidence``: 0.0 for every filled/unknown row.
    - ``IdentitySlotLockLabel``: trajectory consensus; ``"unknown"`` fallback.
    - ``IdentityPosteriorMargin``: 0.0 (no detection → no discriminating
      information between any two identities).
    - ``IdentityEntropy``: forward-fill then backward-fill within each
      trajectory (belief state persists between frames); 0.0 for trajectories
      that never had a detection.
    """
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return df
    if "IdentityAssignedLabel" not in df.columns:
        return df

    df = df.copy()
    df["IdentityAssignedLabel"] = df["IdentityAssignedLabel"].astype(object)
    if "IdentityAssignedConfidence" not in df.columns:
        df["IdentityAssignedConfidence"] = np.nan

    # --- IdentityAssignedLabel + IdentityAssignedConfidence ---
    label_missing = df["IdentityAssignedLabel"].isna() | (
        df["IdentityAssignedLabel"].astype(str).str.strip() == ""
    )
    for _traj_id, group in df.groupby("TrajectoryID", sort=False):
        grp_missing = label_missing.loc[group.index]
        if not grp_missing.any():
            continue
        present = group.loc[~grp_missing, "IdentityAssignedLabel"]
        consensus = present.mode().iloc[0] if not present.empty else _UNKNOWN_LABEL
        fill_idx = group.index[grp_missing]
        df.loc[fill_idx, "IdentityAssignedLabel"] = consensus
        df.loc[fill_idx, "IdentityAssignedConfidence"] = 0.0

    # --- IdentityAssignedID ---
    if "IdentityAssignedID" in df.columns:
        # Build label→ID mapping from rows where both are already valid.
        valid = (
            df["IdentityAssignedLabel"].notna()
            & (df["IdentityAssignedLabel"].astype(str).str.strip() != "")
            & df["IdentityAssignedID"].notna()
        )
        label_to_id: dict[str, float] = {}
        for lbl, idx in zip(
            df.loc[valid, "IdentityAssignedLabel"].astype(str),
            df.loc[valid, "IdentityAssignedID"],
        ):
            label_to_id.setdefault(lbl, float(idx))
        label_to_id[_UNKNOWN_LABEL] = 0.0

        id_missing = df["IdentityAssignedID"].isna()
        if id_missing.any():
            df.loc[id_missing, "IdentityAssignedID"] = (
                df.loc[id_missing, "IdentityAssignedLabel"]
                .astype(str)
                .map(label_to_id)
                .fillna(0.0)
            )

    # --- IdentitySlotLockLabel ---
    if "IdentitySlotLockLabel" in df.columns:
        df["IdentitySlotLockLabel"] = df["IdentitySlotLockLabel"].astype(object)
        slot_missing = df["IdentitySlotLockLabel"].isna() | (
            df["IdentitySlotLockLabel"].astype(str).str.strip() == ""
        )
        for _traj_id, group in df.groupby("TrajectoryID", sort=False):
            grp_missing = slot_missing.loc[group.index]
            if not grp_missing.any():
                continue
            present = group.loc[~grp_missing, "IdentitySlotLockLabel"]
            consensus = present.mode().iloc[0] if not present.empty else _UNKNOWN_LABEL
            df.loc[group.index[grp_missing], "IdentitySlotLockLabel"] = consensus

    # --- IdentityPosteriorMargin ---
    if "IdentityPosteriorMargin" in df.columns:
        df["IdentityPosteriorMargin"] = df["IdentityPosteriorMargin"].fillna(0.0)

    # --- IdentityEntropy ---
    if "IdentityEntropy" in df.columns:
        df["IdentityEntropy"] = (
            df.groupby("TrajectoryID", sort=False)["IdentityEntropy"]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0.0)
        )

    return df


def sort_trajectories_by_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber TrajectoryIDs so same-identity fragments are consecutive.

    Fragments are ordered by (consensus_identity_label, first_frame) so all
    trajectories belonging to the same animal get adjacent IDs.  New IDs start
    at 0 and are strictly sequential; existing values are fully replaced.
    """
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return df

    identity_col = next(
        (c for c in ("IdentityAssignedLabel", "UniqueIdentityKey") if c in df.columns),
        None,
    )
    frame_col = "FrameID" if "FrameID" in df.columns else None

    traj_info: list[tuple] = []
    for traj_id in df["TrajectoryID"].unique():
        mask = df["TrajectoryID"] == traj_id
        consensus = ""
        if identity_col is not None:
            vals = df.loc[mask, identity_col].dropna()
            vals = vals[vals.astype(str).str.strip() != ""]
            if not vals.empty:
                consensus = str(vals.mode().iloc[0])
        min_frame = float(df.loc[mask, frame_col].min()) if frame_col else 0.0
        traj_info.append((traj_id, consensus, min_frame))

    traj_info.sort(key=lambda x: (x[1], x[2]))
    id_mapping = {old: new for new, (old, _, _) in enumerate(traj_info)}

    df = df.copy()
    df["TrajectoryID"] = df["TrajectoryID"].map(id_mapping)
    sort_cols = ["TrajectoryID", frame_col] if frame_col else ["TrajectoryID"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df
