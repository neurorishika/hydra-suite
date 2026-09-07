"""Correctness gates for calibration outputs and determinism floors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .models import EquivalenceVerdict

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is an optional runtime dependency
    linear_sum_assignment = None


@dataclass(frozen=True, slots=True)
class CalibrationOutputs:
    forward: pd.DataFrame
    final: pd.DataFrame

    @classmethod
    def from_csvs(cls, forward: str | Path, final: str | Path) -> "CalibrationOutputs":
        return cls(pd.read_csv(forward), pd.read_csv(final))


@dataclass(frozen=True, slots=True)
class EquivalencePolicy:
    """Correctness tolerances for a candidate-vs-reference comparison.

    ``exact_categorical`` governs the *heuristic* categorical columns only.
    The columns in :data:`_MANDATORY_EXACT_COLUMNS` are always compared
    exactly -- no construction site may switch track-identity checking off.
    """

    position_p99_tolerance: float = 0.5
    angle_max_tolerance: float = 0.05
    match_gate: float = 2.0
    exact_categorical: bool = True


# Always compared exactly, independent of the token heuristics below: a
# Hungarian identity swap keeps row counts, keys and XY identical, so track
# identity is only caught by an exact comparison against the positionally
# matched row.
_MANDATORY_EXACT_COLUMNS = ("TrackID", "TrajectoryID", "State", "ArenaID")


_CATEGORICAL_TOKENS = (
    "uniqueidentity",
    "unique_identity",
    "classlabel",
    "class_label",
    "classname",
    "class_name",
    "directed",
)

_IDENTITY_CATEGORICAL_SUFFIXES = (
    "id",
    "label",
    "source",
    "sources",
    "committed",
    "conflictflag",
    "conflictresolved",
    "slotlock",
)


def _is_headtail_column(column: str) -> bool:
    normalized = column.lower().replace("_", "")
    return "headtail" in normalized and "confidence" not in normalized


def _categorical_columns(columns: Iterable[str]) -> tuple[str, ...]:
    output = []
    for column in columns:
        normalized = column.lower().replace("_", "")
        categorical = column in _MANDATORY_EXACT_COLUMNS or any(
            token.replace("_", "") in normalized for token in _CATEGORICAL_TOKENS
        )
        if "identity" in normalized and normalized.endswith(
            _IDENTITY_CATEGORICAL_SUFFIXES
        ):
            categorical = True
        if _is_headtail_column(column):
            categorical = True
        if categorical:
            output.append(column)
    return tuple(output)


def _row_key(frame: pd.DataFrame) -> list[str] | None:
    if "FrameID" not in frame.columns:
        return None
    for identifier in ("DetectionID", "TrackID", "TrajectoryID", "det_index"):
        if identifier in frame.columns:
            return ["FrameID", identifier]
    return None


def _key_strings(frame: pd.DataFrame, key: list[str]) -> np.ndarray:
    """Key columns rendered NaN-safely as strings.

    ``NaN == NaN`` is False, so comparing raw key values made the aligner
    reject a frame against *itself* whenever a row key was NaN -- and a lost
    track carries a NaN ``DetectionID``, which is ordinary output. Rendering
    through ``string`` + ``fillna`` (the convention already used by the exact
    comparisons below) makes two missing keys agree.
    """

    return np.column_stack(
        [frame[column].astype("string").fillna("<nan>").to_numpy() for column in key]
    )


def _aligned(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    key = _row_key(reference)
    if key is None or any(column not in candidate.columns for column in key):
        return None
    # ``kind="stable"`` so rows sharing a key (three lost tracks in one frame
    # all key on ``(frame, NaN)``) keep their input order on both sides rather
    # than relying on pandas' default sort happening to be stable.
    left = reference.sort_values(key, kind="stable").reset_index(drop=True)
    right = candidate.sort_values(key, kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        return None
    if not (_key_strings(left, key) == _key_strings(right, key)).all():
        return None
    return left, right


def _wrapped_abs_delta(old: float, new: float) -> float:
    """Absolute angular difference wrapped to ``[0, pi]``.

    Wrapping first means 0 rad and 2*pi rad agree instead of reading as a
    2*pi-sized divergence.
    """

    delta = (float(old) - float(new) + np.pi) % (2 * np.pi) - np.pi
    return float(abs(delta))


def _nan_position_pairs(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[list[tuple[object, object]], int]:
    """Pair NaN-position rows by KEY instead of by distance.

    A lost or coasting track exports a row with NaN ``X``/``Y``. Those rows are
    ordinary output, but they cannot take part in a distance-based Hungarian
    pairing, so the gate used to count every one of them as *unmatched* on both
    sides -- which made two byte-identical files compare as different and made
    the A-vs-A determinism floor unreachable on any real project.

    Semantics chosen here:

    * rows are grouped by the row key (``FrameID`` + the row identifier),
      rendered NaN-safely;
    * within a key group each side is sorted by the stringified tuple of all
      shared columns and paired positionally. Two NaN-position rows in the same
      frame with identical content are genuinely indistinguishable -- the CSV
      carries no order -- so multiset equality is the strongest observable
      semantics, and because a sorted elementwise comparison of two differing
      multisets must differ somewhere, nothing real is masked;
    * leftovers on either side are **unmatched**. A row that is NaN on one side
      and positioned on the other therefore fails: its NaN half has no partner
      here, and its positioned half is a surplus row in the Hungarian pass;
    * the returned pairs feed :func:`_mandatory_exact_mismatches` (so
      ``TrackID``/``TrajectoryID``/``State``/``ArenaID`` are still compared
      exactly on these rows) but never feed ``distances``, so the p99
      population is unchanged by this pairing.

    Without a usable row key the rows stay unmatched, as before -- this
    function only ever narrows the unmatched count when it can prove a
    key-level correspondence.
    """

    if reference.empty and candidate.empty:
        return [], 0
    key = _row_key(reference)
    if key is None or any(column not in candidate.columns for column in key):
        return [], len(reference) + len(candidate)

    shared = [column for column in reference.columns if column in candidate.columns]

    def _grouped(frame: pd.DataFrame) -> dict[tuple[str, ...], list[object]]:
        if frame.empty:
            return {}
        keys = _key_strings(frame, key)
        order = (
            frame[shared].astype("string").fillna("<nan>").agg("\x1f".join, axis=1)
            if shared
            else pd.Series("", index=frame.index)
        )
        groups: dict[tuple[str, ...], list[tuple[str, object]]] = {}
        for position, index in enumerate(frame.index):
            groups.setdefault(tuple(keys[position]), []).append(
                (str(order.iloc[position]), index)
            )
        return {
            group: [index for _token, index in sorted(rows, key=lambda item: item[0])]
            for group, rows in groups.items()
        }

    left_groups = _grouped(reference)
    right_groups = _grouped(candidate)
    pairs: list[tuple[object, object]] = []
    unmatched = 0
    for group in set(left_groups) | set(right_groups):
        lefts = left_groups.get(group, [])
        rights = right_groups.get(group, [])
        pairs.extend(zip(lefts, rights))
        unmatched += abs(len(lefts) - len(rights))
    return pairs, unmatched


def _positional(
    reference: pd.DataFrame, candidate: pd.DataFrame, gate: float
) -> dict[str, Any]:
    required = {"FrameID", "X", "Y"}
    if not required.issubset(reference.columns) or not required.issubset(
        candidate.columns
    ):
        return {
            "matched": min(len(reference), len(candidate)),
            "unmatched": abs(len(reference) - len(candidate)),
            "position_p99": 0.0,
            "angle_max": 0.0,
            "angle_samples": (),
            "pairs": (),
        }
    left_positioned = reference[["X", "Y"]].notna().all(axis=1).to_numpy()
    right_positioned = candidate[["X", "Y"]].notna().all(axis=1).to_numpy()
    left = reference[left_positioned]
    right = candidate[right_positioned]
    distances: list[float] = []
    angle_samples: list[tuple[object, object, float]] = []
    nan_pairs, unmatched = _nan_position_pairs(
        reference[~left_positioned], candidate[~right_positioned]
    )
    matched = len(nan_pairs)
    pairs: list[tuple[object, object]] = list(nan_pairs)
    for frame_id in sorted(set(left["FrameID"]) | set(right["FrameID"])):
        a = left[left["FrameID"] == frame_id]
        b = right[right["FrameID"] == frame_id]
        pa = a[["X", "Y"]].to_numpy(float)
        pb = b[["X", "Y"]].to_numpy(float)
        if not len(pa) or not len(pb):
            unmatched += len(pa) + len(pb)
            continue
        cost = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
        else:  # deterministic greedy fallback
            rows, cols, used_rows, used_cols = [], [], set(), set()
            for flat in np.argsort(cost, axis=None):
                row, col = np.unravel_index(flat, cost.shape)
                if int(row) in used_rows or int(col) in used_cols:
                    continue
                used_rows.add(int(row))
                used_cols.add(int(col))
                rows.append(int(row))
                cols.append(int(col))
        accepted = 0
        theta_a = a["Theta"].to_numpy(float) if "Theta" in a else None
        theta_b = b["Theta"].to_numpy(float) if "Theta" in b else None
        for row, col in zip(rows, cols):
            distance = float(cost[row, col])
            if distance > gate:
                continue
            distances.append(distance)
            accepted += 1
            pairs.append((a.index[row], b.index[col]))
            if theta_a is not None and theta_b is not None:
                old, new = theta_a[row], theta_b[col]
                if not (np.isnan(old) or np.isnan(new)):
                    angle_samples.append(
                        (a.index[row], b.index[col], _wrapped_abs_delta(old, new))
                    )
        matched += accepted
        unmatched += len(pa) + len(pb) - 2 * accepted
    return {
        "matched": matched,
        "unmatched": unmatched,
        "position_p99": float(np.percentile(distances, 99)) if distances else 0.0,
        "angle_max": (
            float(max(sample[2] for sample in angle_samples)) if angle_samples else 0.0
        ),
        "angle_samples": tuple(angle_samples),
        "pairs": tuple(pairs),
    }


def _mandatory_exact_mismatches(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    pairs: tuple[tuple[object, object], ...],
    aligned: tuple[pd.DataFrame, pd.DataFrame] | None,
    name: str,
) -> tuple[int, list[str]]:
    """Compare the always-exact identity columns on positionally matched rows.

    The positional (Hungarian XY) pairing is preferred over the keyed
    alignment: when ``TrackID`` is itself the row key the keyed alignment makes
    it equal by construction, which is exactly the identity swap this check
    exists to catch. The keyed alignment is the fallback for outputs without
    X/Y columns.
    """

    if pairs:
        left = reference.loc[[old for old, _new in pairs]].reset_index(drop=True)
        right = candidate.loc[[new for _old, new in pairs]].reset_index(drop=True)
    elif aligned is not None:
        left, right = aligned
    else:
        return 0, []

    mismatches = 0
    details: list[str] = []
    for column in _MANDATORY_EXACT_COLUMNS:
        if column not in left.columns:
            continue
        if column not in right.columns:
            mismatches += len(left)
            details.append(f"{name}: mandatory column {column} missing from candidate")
            continue
        old = left[column].astype("string").fillna("<nan>")
        new = right[column].astype("string").fillna("<nan>")
        differing = (old != new).to_numpy()
        count = int(np.count_nonzero(differing))
        if not count:
            continue
        mismatches += count
        first = int(np.flatnonzero(differing)[0])
        frame_id = (
            left["FrameID"].iloc[first] if "FrameID" in left.columns else "<unknown>"
        )
        details.append(
            f"{name}: {column} differs on {count} row(s); "
            f"first differing FrameID={frame_id}"
        )
    return mismatches, details


def _headtail_flip_pairs(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    pairs: tuple[tuple[object, object], ...],
) -> frozenset[tuple[object, object]]:
    """Pairs whose head/tail categorical column differs between the two runs.

    These rows are the repository's documented bistable head/tail pi-flip noise
    floor. They are already rejected by the exact categorical check, so removing
    them from the *measured determinism floor's* angular statistic loses no
    coverage -- and it stops one bistable row from setting the floor to pi and
    thereby disarming the angle gate for every later candidate.
    """

    columns = [
        column
        for column in reference.columns
        if _is_headtail_column(column) and column in candidate.columns
    ]
    if not columns or not pairs:
        return frozenset()
    flipped = set()
    for old_index, new_index in pairs:
        for column in columns:
            old = reference.at[old_index, column]
            new = candidate.at[new_index, column]
            if pd.isna(old) and pd.isna(new):
                continue
            if pd.isna(old) or pd.isna(new) or old != new:
                flipped.add((old_index, new_index))
                break
    return frozenset(flipped)


def _angle_max(
    samples: tuple[tuple[object, object, float], ...],
    excluded: frozenset[tuple[object, object]],
) -> float:
    kept = [
        delta
        for old_index, new_index, delta in samples
        if (old_index, new_index) not in excluded
    ]
    return float(max(kept)) if kept else 0.0


def _compare_one(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    policy: EquivalencePolicy,
    position_limit: float,
    angle_limit: float,
    name: str,
    exclude_headtail_flip_rows: bool = False,
) -> EquivalenceVerdict:
    details = []
    nonzero = len(reference) > 0 and len(candidate) > 0
    counts_match = len(reference) == len(candidate)
    columns_match = set(reference.columns) == set(candidate.columns)
    metrics = _positional(reference, candidate, policy.match_gate)
    aligned = _aligned(reference, candidate)
    nan_mismatches = 0
    categorical_mismatches = 0
    if aligned is None:
        pairs = metrics["pairs"]
        if len(pairs) == len(reference) == len(candidate):
            left = reference.loc[[old for old, _new in pairs]].reset_index(drop=True)
            right = candidate.loc[[new for _old, new in pairs]].reset_index(drop=True)
            aligned = (left, right)
        elif counts_match:
            details.append(f"{name}: keyed rows are not aligned")
    if aligned is not None:
        left, right = aligned
        for column in set(left.columns) & set(right.columns):
            left_nan = pd.isna(left[column]).to_numpy()
            right_nan = pd.isna(right[column]).to_numpy()
            nan_mismatches += int(np.count_nonzero(left_nan != right_nan))
        if policy.exact_categorical:
            for column in _categorical_columns(left.columns):
                if column in _MANDATORY_EXACT_COLUMNS:
                    # Owned by _mandatory_exact_mismatches, which compares them
                    # against the positional pairing; counting them here too
                    # would double-count and muddy the rejection reason.
                    continue
                if column not in right.columns:
                    categorical_mismatches += len(left)
                    continue
                old = left[column].astype("string").fillna("<nan>")
                new = right[column].astype("string").fillna("<nan>")
                categorical_mismatches += int((old != new).sum())
    mandatory_mismatches, mandatory_details = _mandatory_exact_mismatches(
        reference, candidate, pairs=metrics["pairs"], aligned=aligned, name=name
    )
    categorical_mismatches += mandatory_mismatches
    details.extend(mandatory_details)
    excluded_pairs = (
        _headtail_flip_pairs(reference, candidate, metrics["pairs"])
        if exclude_headtail_flip_rows
        else frozenset()
    )
    angle_max = _angle_max(metrics["angle_samples"], excluded_pairs)
    if not nonzero:
        details.append(f"{name}: empty output")
    if not counts_match:
        details.append(
            f"{name}: row counts differ ({len(reference)} != {len(candidate)})"
        )
    if not columns_match:
        missing = sorted(set(reference.columns) - set(candidate.columns))
        extra = sorted(set(candidate.columns) - set(reference.columns))
        details.append(
            f"{name}: output columns differ (missing={missing}, extra={extra})"[:1024]
        )
    if metrics["unmatched"]:
        details.append(f"{name}: {metrics['unmatched']} unmatched positional rows")
    if nan_mismatches:
        details.append(f"{name}: {nan_mismatches} NaN-pattern mismatches")
    if categorical_mismatches:
        details.append(f"{name}: {categorical_mismatches} categorical mismatches")
    if angle_max > angle_limit:
        details.append(
            f"{name}: angular per-row max {angle_max:.6f} rad "
            f"exceeds limit {angle_limit:.6f} rad"
        )
    passed = bool(
        nonzero
        and counts_match
        and columns_match
        and int(metrics["matched"]) > 0
        and int(metrics["unmatched"]) == 0
        and float(metrics["position_p99"]) <= position_limit
        and angle_max <= angle_limit
        and nan_mismatches == 0
        and categorical_mismatches == 0
    )
    return EquivalenceVerdict(
        passed=passed,
        nonzero_rows=nonzero,
        row_counts_match=counts_match,
        unmatched_rows=int(metrics["unmatched"]),
        position_p99=float(metrics["position_p99"]),
        angle_max=angle_max,
        nan_pattern_mismatches=nan_mismatches,
        categorical_mismatches=categorical_mismatches,
        details=tuple(details),
    )


def compare_outputs(
    reference: CalibrationOutputs,
    candidate: CalibrationOutputs,
    *,
    determinism_floor: EquivalenceVerdict | None = None,
    policy: EquivalencePolicy | None = None,
    for_determinism_floor: bool = False,
) -> EquivalenceVerdict:
    """Gate both forward-rich and final tracking outputs.

    Set ``for_determinism_floor`` for the A-vs-A repeat comparison whose metrics
    become the measured floor: it excludes known-bistable head/tail rows from
    the angular statistic (see :func:`_headtail_flip_pairs`).
    """

    policy = policy or EquivalencePolicy()
    position_limit = max(
        policy.position_p99_tolerance,
        determinism_floor.position_p99 if determinism_floor is not None else 0.0,
    )
    angle_limit = max(
        policy.angle_max_tolerance,
        determinism_floor.angle_max if determinism_floor is not None else 0.0,
    )
    verdicts = (
        _compare_one(
            reference.forward,
            candidate.forward,
            policy=policy,
            position_limit=position_limit,
            angle_limit=angle_limit,
            name="forward",
            exclude_headtail_flip_rows=for_determinism_floor,
        ),
        _compare_one(
            reference.final,
            candidate.final,
            policy=policy,
            position_limit=position_limit,
            angle_limit=angle_limit,
            name="final",
            exclude_headtail_flip_rows=for_determinism_floor,
        ),
    )
    return EquivalenceVerdict(
        passed=all(item.passed for item in verdicts),
        nonzero_rows=all(item.nonzero_rows for item in verdicts),
        row_counts_match=all(item.row_counts_match for item in verdicts),
        unmatched_rows=sum(item.unmatched_rows for item in verdicts),
        position_p99=max(item.position_p99 for item in verdicts),
        angle_max=max(item.angle_max for item in verdicts),
        nan_pattern_mismatches=sum(item.nan_pattern_mismatches for item in verdicts),
        categorical_mismatches=sum(item.categorical_mismatches for item in verdicts),
        details=tuple(detail for item in verdicts for detail in item.details),
    )
