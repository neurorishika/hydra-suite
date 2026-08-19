"""Identity post-processing utilities.

Provides three functions used by the tracking orchestrator after the MILP
fragment solver has assigned labels:

- ``identity_sources_conflict`` — detect conflicting identity evidence
- ``parse_identity_key``        — deserialise a source-keyed identity string
- ``fill_identity_nans_with_consensus`` — fill missing labels per trajectory
- ``sort_trajectories_by_identity``     — renumber IDs by identity then time
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C

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


def format_identity_key(sources: dict[str, str]) -> str:
    """Serialize a source-keyed identity dict into a ``UniqueIdentityKey`` token.

    Inverse of :func:`parse_identity_key`. Tokens are sorted ascending by
    source key and joined by ``_KEY_SEP``. Sources with an empty/missing
    value are omitted. Returns ``""`` when there is nothing to serialize --
    callers writing a dataframe column should map that to ``np.nan``.
    """
    parts = []
    for source in sorted(sources):
        value = _normalize_string(sources[source])
        source_token = _normalize_string(source)
        if not source_token or not value:
            continue
        parts.append(f"{source_token}{_PAIR_SEP}{value}")
    return _KEY_SEP.join(parts)


_CNN_CLASS_COLUMN_RE = re.compile(r"^CNN_(.+)_Class$")


def _cnn_identity_sources_for_row(row: "pd.Series", cnn_class_columns: list) -> dict:
    """Build ``cnn:<head>``/``cnn:<head>:<factor>`` tokens from CNN_* columns.

    A column is a CNN class column iff it matches ``^CNN_(.+)_Class$``.
    Within the captured middle: no further ``_`` -> 2-part head-only source
    (``cnn:<head>``); otherwise split on the FIRST ``_`` -> 3-part factor
    source (``cnn:<head>:<factor>``), matching the old serializer's
    convention. ``CNN_<head>_Conf`` sibling columns are never class columns.
    """
    sources: dict[str, str] = {}
    for col in cnn_class_columns:
        match = _CNN_CLASS_COLUMN_RE.match(str(col))
        if not match:
            continue
        value = _normalize_string(row.get(col))
        if not value:
            continue
        middle = match.group(1)
        if "_" in middle:
            head, factor = middle.split("_", 1)
            source = f"cnn:{head}:{factor}"
        else:
            source = f"cnn:{middle}"
        sources[source] = value
    return sources


def derive_unique_identity_key_series(
    df: pd.DataFrame, identity_heads=None, all_classifier_labels=()
) -> pd.Series:
    """Re-derive the ``UniqueIdentityKey`` column from per-row evidence columns.

    Builds, per row, a ``source -> value`` dict from ``DetectedTagLabel``
    (preferred) / ``DetectedTagID`` for the ``apriltag`` source and the
    ``CNN_<head>_Class`` / ``CNN_<head>_<factor>_Class`` columns of the
    *identity heads* for the CNN sources, then serializes it with
    :func:`format_identity_key`. Rows with no evidence get ``np.nan``
    (never an empty string or a bare label).

    ``identity_heads`` is the tuple of classifier labels marked
    ``unique_identifier`` (see ``identity.heads.identity_head_labels``).
    Classifiers that are not identity heads -- behavior, sex, caste -- must
    never enter this key: it feeds the relink identity veto
    (``processing.py:_score_relink_candidate``), where a mere behavior change
    across an occlusion gap would otherwise read as an identity conflict and
    refuse a legitimate relink. ``None`` preserves the legacy
    every-CNN-column behavior for callers with no classifier config.

    ``all_classifier_labels`` is the full classifier roster's labels
    (identity and non-identity alike); it is passed straight through to
    ``identity_class_columns`` to disambiguate prefix collisions between an
    identity head and a differently-named non-identity classifier (e.g.
    head ``tag`` vs. classifier ``tag_v2``).
    """
    if df is None or df.empty:
        return pd.Series([], index=getattr(df, "index", None), dtype=object)

    if identity_heads is None:
        cnn_class_columns = [
            col for col in df.columns if _CNN_CLASS_COLUMN_RE.match(str(col))
        ]
    else:
        from hydra_suite.core.individual.identity.heads import identity_class_columns

        cnn_class_columns = identity_class_columns(
            df.columns, identity_heads, all_classifier_labels
        )
    has_tag_label = "DetectedTagLabel" in df.columns
    has_tag_id = "DetectedTagID" in df.columns

    def _row_key(row: "pd.Series") -> Any:
        sources: dict[str, str] = {}
        tag_value = ""
        if has_tag_label:
            tag_value = _normalize_string(row.get("DetectedTagLabel"))
        if not tag_value and has_tag_id:
            tag_value = _normalize_string(row.get("DetectedTagID"))
        if tag_value:
            sources["apriltag"] = tag_value
        sources.update(_cnn_identity_sources_for_row(row, cnn_class_columns))
        key = format_identity_key(sources)
        return key if key else np.nan

    result = df.apply(_row_key, axis=1)
    return result.astype(object)


def fill_identity_nans_with_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN ``IdentityFinal*`` columns using per-trajectory majority label.

    Strategy per column:
    - ``C.FINAL_LABEL``: trajectory consensus; ``"unknown"`` when the entire
      trajectory has no label evidence.
    - ``C.FINAL_ID``: catalog index inferred from existing label->ID pairs in
      the data; 0 for rows whose label resolved to ``"unknown"``.
    - ``C.FINAL_CONFIDENCE``: 0.0 for every filled/unknown row.

    This is a Final-family (resolved identity) operation only. It never
    reads or writes ``IdentityRealtime*`` columns -- ``IdentityRealtimeMargin``
    /``IdentityRealtimeEntropy``/``IdentityRealtimeSlotLock`` are realtime
    inputs and stay untouched here.
    """
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return df
    if C.FINAL_LABEL not in df.columns:
        return df

    df = df.copy()
    df[C.FINAL_LABEL] = df[C.FINAL_LABEL].astype(object)
    if C.FINAL_CONFIDENCE not in df.columns:
        df[C.FINAL_CONFIDENCE] = np.nan

    label_missing = df[C.FINAL_LABEL].isna() | (
        df[C.FINAL_LABEL].astype(str).str.strip() == ""
    )
    for _traj_id, group in df.groupby("TrajectoryID", sort=False):
        grp_missing = label_missing.loc[group.index]
        if not grp_missing.any():
            continue
        present = group.loc[~grp_missing, C.FINAL_LABEL]
        consensus = present.mode().iloc[0] if not present.empty else _UNKNOWN_LABEL
        fill_idx = group.index[grp_missing]
        df.loc[fill_idx, C.FINAL_LABEL] = consensus
        df.loc[fill_idx, C.FINAL_CONFIDENCE] = 0.0

    if C.FINAL_ID in df.columns:
        valid = (
            df[C.FINAL_LABEL].notna()
            & (df[C.FINAL_LABEL].astype(str).str.strip() != "")
            & df[C.FINAL_ID].notna()
        )
        label_to_id: dict[str, float] = {}
        for lbl, idx in zip(
            df.loc[valid, C.FINAL_LABEL].astype(str),
            df.loc[valid, C.FINAL_ID],
        ):
            label_to_id.setdefault(lbl, float(idx))
        label_to_id[_UNKNOWN_LABEL] = 0.0

        id_missing = df[C.FINAL_ID].isna()
        if id_missing.any():
            df.loc[id_missing, C.FINAL_ID] = (
                df.loc[id_missing, C.FINAL_LABEL]
                .astype(str)
                .map(label_to_id)
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
        (c for c in (C.FINAL_LABEL, C.UNIQUE_IDENTITY_KEY) if c in df.columns),
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
