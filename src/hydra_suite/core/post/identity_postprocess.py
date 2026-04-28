"""Identity post-processing utilities.

Provides three functions used by the tracking orchestrator after the MILP
fragment solver has assigned labels:

- ``identity_sources_conflict`` — detect conflicting identity evidence
- ``parse_identity_key``        — deserialise a source-keyed identity string
- ``fill_identity_nans_with_consensus`` — fill missing labels per trajectory
- ``sort_trajectories_by_identity``     — renumber IDs by identity then time
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

_KEY_SEP = "|"
_PAIR_SEP = "="
_UNKNOWN_LABEL = "unknown"


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

    if "IdentityAssignedID" in df.columns:
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

    if "IdentityPosteriorMargin" in df.columns:
        df["IdentityPosteriorMargin"] = df["IdentityPosteriorMargin"].fillna(0.0)

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
