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
    position_p99_tolerance: float = 0.5
    angle_mean_tolerance: float = 0.05
    match_gate: float = 2.0
    exact_categorical: bool = True


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


def _categorical_columns(columns: Iterable[str]) -> tuple[str, ...]:
    output = []
    for column in columns:
        normalized = column.lower().replace("_", "")
        categorical = any(
            token.replace("_", "") in normalized for token in _CATEGORICAL_TOKENS
        )
        if "identity" in normalized and normalized.endswith(
            _IDENTITY_CATEGORICAL_SUFFIXES
        ):
            categorical = True
        if "headtail" in normalized and "confidence" not in normalized:
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


def _aligned(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    key = _row_key(reference)
    if key is None or any(column not in candidate.columns for column in key):
        return None
    left = reference.sort_values(key).reset_index(drop=True)
    right = candidate.sort_values(key).reset_index(drop=True)
    if (
        len(left) != len(right)
        or not (left[key].to_numpy() == right[key].to_numpy()).all()
    ):
        return None
    return left, right


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
            "angle_mean": 0.0,
            "pairs": (),
        }
    left = reference.dropna(subset=["X", "Y"])
    right = candidate.dropna(subset=["X", "Y"])
    distances: list[float] = []
    angles: list[float] = []
    unmatched = (len(reference) - len(left)) + (len(candidate) - len(right))
    matched = 0
    pairs: list[tuple[object, object]] = []
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
                    delta = abs(old - new) % (2 * np.pi)
                    angles.append(float(min(delta, 2 * np.pi - delta)))
        matched += accepted
        unmatched += len(pa) + len(pb) - 2 * accepted
    return {
        "matched": matched,
        "unmatched": unmatched,
        "position_p99": float(np.percentile(distances, 99)) if distances else 0.0,
        "angle_mean": float(np.mean(angles)) if angles else 0.0,
        "pairs": tuple(pairs),
    }


def _compare_one(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    policy: EquivalencePolicy,
    position_limit: float,
    angle_limit: float,
    name: str,
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
                if column not in right.columns:
                    categorical_mismatches += len(left)
                    continue
                old = left[column].astype("string").fillna("<nan>")
                new = right[column].astype("string").fillna("<nan>")
                categorical_mismatches += int((old != new).sum())
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
    passed = bool(
        nonzero
        and counts_match
        and columns_match
        and int(metrics["matched"]) > 0
        and int(metrics["unmatched"]) == 0
        and float(metrics["position_p99"]) <= position_limit
        and float(metrics["angle_mean"]) <= angle_limit
        and nan_mismatches == 0
        and categorical_mismatches == 0
    )
    return EquivalenceVerdict(
        passed=passed,
        nonzero_rows=nonzero,
        row_counts_match=counts_match,
        unmatched_rows=int(metrics["unmatched"]),
        position_p99=float(metrics["position_p99"]),
        angle_mean=float(metrics["angle_mean"]),
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
) -> EquivalenceVerdict:
    """Gate both forward-rich and final tracking outputs."""

    policy = policy or EquivalencePolicy()
    position_limit = max(
        policy.position_p99_tolerance,
        determinism_floor.position_p99 if determinism_floor is not None else 0.0,
    )
    angle_limit = max(
        policy.angle_mean_tolerance,
        determinism_floor.angle_mean if determinism_floor is not None else 0.0,
    )
    verdicts = (
        _compare_one(
            reference.forward,
            candidate.forward,
            policy=policy,
            position_limit=position_limit,
            angle_limit=angle_limit,
            name="forward",
        ),
        _compare_one(
            reference.final,
            candidate.final,
            policy=policy,
            position_limit=position_limit,
            angle_limit=angle_limit,
            name="final",
        ),
    )
    return EquivalenceVerdict(
        passed=all(item.passed for item in verdicts),
        nonzero_rows=all(item.nonzero_rows for item in verdicts),
        row_counts_match=all(item.row_counts_match for item in verdicts),
        unmatched_rows=sum(item.unmatched_rows for item in verdicts),
        position_p99=max(item.position_p99 for item in verdicts),
        angle_mean=max(item.angle_mean for item in verdicts),
        nan_pattern_mismatches=sum(item.nan_pattern_mismatches for item in verdicts),
        categorical_mismatches=sum(item.categorical_mismatches for item in verdicts),
        details=tuple(detail for item in verdicts for detail in item.details),
    )
