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
from hydra_suite.core.individual.identity.columns import (  # noqa: F401
    normalize_final_source_series,
)

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


def _cnn_identity_sources_for_row(
    row: "pd.Series", cnn_class_columns: list, non_identifying_values_by_column=None
) -> dict:
    """Build ``cnn:<head>``/``cnn:<head>:<factor>`` tokens from CNN_* columns.

    A column is a CNN class column iff it matches ``^CNN_(.+)_Class$``.
    Within the captured middle: no further ``_`` -> 2-part head-only source
    (``cnn:<head>``); otherwise split on the FIRST ``_`` -> 3-part factor
    source (``cnn:<head>:<factor>``), matching the old serializer's
    convention. ``CNN_<head>_Conf`` sibling columns are never class columns.

    ``non_identifying_values_by_column`` (``{class_column: {values}}``, as
    resolved by ``resolve.non_identifying_axis_values``) drops a column
    whose value is declared non-identifying on that axis, exactly like a
    missing value. Whether a row contributes identity evidence at ALL is a
    separate, caller-owned decision (see
    ``derive_unique_identity_key_series``'s ``non_identifying_rows``); no
    mark semantics are re-derived here.
    """
    sources: dict[str, str] = {}
    for col in cnn_class_columns:
        match = _CNN_CLASS_COLUMN_RE.match(str(col))
        if not match:
            continue
        value = _normalize_string(row.get(col))
        if not value:
            continue
        if value in (non_identifying_values_by_column or {}).get(str(col), ()):
            # A non-identifying class carries no identity information. Left
            # in, `notag == notag` would count as AGREEMENT in
            # `_compare_identity_sources`' grouped tally and could out-vote
            # a genuine conflict on another axis.
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
    df: pd.DataFrame,
    identity_heads=None,
    all_classifier_labels=(),
    non_identifying_rows=None,
    non_identifying_values_by_column=None,
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

    ``non_identifying_rows`` is an optional boolean mask (aligned to
    ``df.index``) marking rows whose observed composite was declared
    non-identifying *as a whole* (``resolve.whole_composite_excluded_labels``
    -- the ``"notag_notag"`` mark form). Those rows contribute no CNN
    identity evidence at all, exactly as if every identity-head column were
    missing: two untagged fragments would otherwise produce identical keys
    and register as *agreement* in ``_compare_identity_sources``, which is
    precisely the relink-veto failure the exclusion exists to prevent.

    It must NOT be widened to every composite any mark excludes: a bare or
    scoped mark excludes composites in which only ONE axis reads the marked
    value, and such a row can still carry a genuine tag on another axis.
    Dropping it whole would turn a real conflict into "no evidence" and stop
    the veto from firing -- see ``non_identifying_values_by_column``, which
    is the right granularity for those forms. A non-CNN source (a detected
    AprilTag) is a genuinely different identity source and is NOT dropped.
    ``None`` by default, which reproduces the legacy every-row-counts
    behavior byte-for-bit.

    ``non_identifying_values_by_column`` is the per-axis half of the same
    exclusion: ``{class_column: {declared non-identifying class values}}``
    (as resolved by ``resolve.non_identifying_axis_values``). A column
    carrying one of its own axis's values contributes nothing, exactly as
    if it were missing, so a shared ``notag`` on one axis cannot register
    as agreement and out-vote a genuine conflict on another. ``None`` by
    default, which reproduces the legacy every-value-counts behavior
    byte-for-bit.
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
    if non_identifying_rows is None:
        drop_cnn_index = frozenset()
    else:
        mask = pd.Series(non_identifying_rows, index=df.index).fillna(False)
        drop_cnn_index = frozenset(df.index[mask.astype(bool)])

    def _row_key(row: "pd.Series") -> Any:
        sources: dict[str, str] = {}
        tag_value = ""
        if has_tag_label:
            tag_value = _normalize_string(row.get("DetectedTagLabel"))
        if not tag_value and has_tag_id:
            tag_value = _normalize_string(row.get("DetectedTagID"))
        if tag_value:
            sources["apriltag"] = tag_value
        if row.name not in drop_cnn_index:
            sources.update(
                _cnn_identity_sources_for_row(
                    row, cnn_class_columns, non_identifying_values_by_column or {}
                )
            )
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


def _trajectory_arena_sort_key(values) -> tuple:
    """Return a stable, orderable sort key for one trajectory's ``arena_id`` values.

    A trajectory cannot legitimately span arenas (slot -> arena is a static,
    per-track-slot property, Tasks 1-6): raises if this trajectory's
    non-null ``arena_id`` values disagree, consistent with
    ``processing._trajectory_arena``'s handling of the same invariant.

    Missing/NaN arena info sorts deterministically before any real arena
    (rather than via a raw NaN, whose comparisons are unstable -- ``nan <
    nan`` is always False -- which would make Python's sort silently
    arbitrary instead of merely surprising). Raises on a non-numeric
    ``arena_id`` rather than propagating a NaN sort key.
    """
    unique = pd.unique(pd.Series(values).dropna())
    if len(unique) > 1:
        raise ValueError(
            f"Trajectory has a non-constant arena_id ({sorted(unique.tolist())}); "
            "arena assignment must be static per slot (Tasks 1-6)."
        )
    if len(unique) == 0:
        return (0, 0.0)
    raw = unique[0]
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"arena_id must be numeric, got {raw!r}") from exc
    return (1, numeric)


def sort_trajectories_by_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber TrajectoryIDs so same-identity fragments are consecutive.

    Fragments are ordered by (arena, consensus_identity_label, first_frame) so
    all trajectories belonging to the same animal get adjacent IDs.  New IDs
    start at 0 and are strictly sequential; existing values are fully
    replaced.

    The arena id leads the sort key so identity labels that repeat across
    arenas (e.g. arena 0's "ant A" and arena 7's "ant A") never interleave --
    each arena's trajectories form one contiguous run of new ids, and arena
    0's numbering never depends on what arena 7 contains. With no
    ``arena_id`` column (or a single arena), the key's leading component is
    constant and the ordering is bit-identical to before arena support.
    """
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return df

    identity_col = next(
        (c for c in (C.FINAL_LABEL, C.UNIQUE_IDENTITY_KEY) if c in df.columns),
        None,
    )
    frame_col = "FrameID" if "FrameID" in df.columns else None
    arena_col = "arena_id" if "arena_id" in df.columns else None

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
        arena_key = (
            _trajectory_arena_sort_key(df.loc[mask, arena_col])
            if arena_col
            else (0, 0.0)
        )
        traj_info.append((traj_id, arena_key, consensus, min_frame))

    traj_info.sort(key=lambda x: (x[1], x[2], x[3]))
    id_mapping = {old: new for new, (old, _, _, _) in enumerate(traj_info)}

    df = df.copy()
    df["TrajectoryID"] = df["TrajectoryID"].map(id_mapping)
    sort_cols = ["TrajectoryID", frame_col] if frame_col else ["TrajectoryID"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df


_IDENTITY_INVARIANT_COLUMNS = (C.FINAL_LABEL, C.FINAL_ID, C.FINAL_SOURCE)


def assert_one_identity_per_trajectory(df: pd.DataFrame) -> list:
    """Return the sorted ``TrajectoryID``s that carry more than one identity.

    A ``TrajectoryID`` is an offender when either ``IdentityFinalLabel`` or
    ``IdentityFinalID`` takes more than one distinct value (``dropna=False``
    -- a mix of a real value and a missing one is also a violation) within
    that trajectory, OR its non-``IdentityFinalSource.NONE`` rows disagree
    on ``IdentityFinalSource``. Missing Final-family columns (identity never
    resolved at all) or a missing ``TrajectoryID`` column both mean nothing
    to check, so both return ``[]``.

    ``IdentityFinalSource.NONE`` rows are excluded from the source check
    (but NOT from the label/ID checks) because ``NONE`` is not itself a
    provenance value -- it is the "no source was ever recorded for this
    row" sentinel written by ``fill_identity_nans_with_consensus`` for rows
    that had no realtime/tag/offline evidence at all. A trajectory where
    every row agrees on ``IdentityFinalLabel`` and every *evidenced* row
    agrees on ``IdentityFinalSource`` has never actually had conflicting
    identity information, purely varying source PROVENANCE (some rows
    resolved, some consensus-filled) is not a conflict, and must not trip
    this guard or trigger ``collapse_to_majority_identity``'s rewrite.

    This is the last-line invariant guard, called right before every rich
    -export write (Task 6): relink runs BEFORE identity resolution now, but
    a bug in either stage -- or a future caller that reorders them again --
    could still stitch two solver-labelled fragments into one trajectory.
    """
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return []
    present_cols = [c for c in _IDENTITY_INVARIANT_COLUMNS if c in df.columns]
    if not present_cols:
        return []

    offenders = set()
    for col in present_cols:
        if col == C.FINAL_SOURCE:
            source = normalize_final_source_series(df[col])
            evidenced = source != C.IdentityFinalSource.NONE
            if not evidenced.any():
                continue
            nunique = (
                source[evidenced].groupby(df.loc[evidenced, "TrajectoryID"]).nunique()
            )
            offenders.update(nunique[nunique > 1].index.tolist())
            continue
        nunique = df.groupby("TrajectoryID")[col].nunique(dropna=False)
        offenders.update(nunique[nunique > 1].index.tolist())
    return sorted(offenders)


def collapse_to_majority_identity(df: pd.DataFrame, offenders) -> pd.DataFrame:
    """Force each offending trajectory onto its majority identity.

    For every ``TrajectoryID`` in *offenders*: pick the ``IdentityFinalLabel``
    with the most rows (ties broken by first appearance in ``FrameID``
    order), take ``IdentityFinalID`` from that label's first row, and set
    ``IdentityFinalConfidence`` to the MINIMUM confidence among the rows
    that actually carried a source -- a forced collapse is exactly the
    situation where the trajectory's identity is least trustworthy, and
    reporting the min (not the majority label's own confidence) keeps that
    honest. Trajectories not in *offenders* are returned unchanged.

    ``IdentityFinalSource`` is rewritten to the majority row's source only
    on rows that already carried a real (non-``NONE``) source of their own;
    rows whose source was ``IdentityFinalSource.NONE`` (never resolved by
    realtime/tag/offline evidence, only consensus-filled) are left at
    ``NONE`` -- this mechanism must never fabricate provenance a row never
    had. For the same reason, the MINIMUM confidence is taken only over
    rows that carried a real source: consensus-filled rows are always
    confidence 0.0 by construction (``fill_identity_nans_with_consensus``)
    and were never actually in conflict, so including them would flatten
    every genuinely-evidenced row's confidence to 0.0 for free. If a
    trajectory has no evidenced rows at all (a pure ID/label mismatch with
    no source column, or every row is source-less), fall back to the
    whole-trajectory minimum.
    """
    if df is None or df.empty or not offenders:
        return df

    offenders_set = set(offenders)
    out = df.copy()
    if C.FINAL_LABEL not in out.columns:
        return out

    has_source_col = C.FINAL_SOURCE in out.columns
    sort_col = "FrameID" if "FrameID" in out.columns else None
    for traj_id in offenders_set:
        mask = out["TrajectoryID"] == traj_id
        group = out.loc[mask]
        if sort_col:
            group = group.sort_values(sort_col, kind="stable")
        labels = group[C.FINAL_LABEL]
        counts = labels.value_counts(dropna=False)
        if counts.empty:
            continue
        max_count = counts.max()
        top_labels = counts[counts == max_count].index.tolist()
        # Tie-break by first appearance (FrameID order) among the tied labels.
        majority_label = next(lbl for lbl in labels.tolist() if lbl in top_labels)
        first_row = group[labels == majority_label].iloc[0]

        out.loc[mask, C.FINAL_LABEL] = majority_label
        if C.FINAL_ID in out.columns:
            out.loc[mask, C.FINAL_ID] = first_row[C.FINAL_ID]

        evidenced_idx = group.index
        if has_source_col:
            existing_source = normalize_final_source_series(
                out.loc[group.index, C.FINAL_SOURCE]
            )
            evidenced_idx = group.index[
                (existing_source != C.IdentityFinalSource.NONE).to_numpy()
            ]
            # Never fabricate provenance on rows that never carried a
            # source of their own -- only rows that genuinely had one get
            # overwritten with the majority row's source.
            out.loc[evidenced_idx, C.FINAL_SOURCE] = first_row[C.FINAL_SOURCE]

        if C.FINAL_CONFIDENCE in out.columns:
            conf_pool_idx = evidenced_idx if len(evidenced_idx) else group.index
            min_conf = pd.to_numeric(
                out.loc[conf_pool_idx, C.FINAL_CONFIDENCE], errors="coerce"
            ).min()
            if pd.isna(min_conf):
                min_conf = pd.to_numeric(
                    group[C.FINAL_CONFIDENCE], errors="coerce"
                ).min()
            out.loc[mask, C.FINAL_CONFIDENCE] = min_conf

    return out
