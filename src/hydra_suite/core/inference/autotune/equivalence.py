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
    # Keypoints are positions in the same pixel units as X/Y, so they carry
    # the body-position budget verbatim -- the same p99 and the same hard
    # ``match_gate``.
    keypoint_p99_tolerance: float = 0.5
    # Every other tolerance-compared numeric column (confidences, quality
    # scores, per-row counters). MEASURED FLOOR: an A-vs-A repeat of
    # ``ant_pose_headtail`` on MPS (12500 forward + 9096 final rows) has
    # ``max|delta| == 0.0`` exactly on every numeric column, so this is a
    # CSV-round-trip guard rather than a substantive budget. A platform whose
    # own repeat is noisier widens it through ``determinism_floor``.
    numeric_max_tolerance: float = 1e-6


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


# Columns whose comparison is owned by the positional (Hungarian) pass and
# must NOT be double-counted by the numeric families below.
_POSITIONAL_COLUMNS = ("X", "Y", "Theta")


# --- reported-only columns: NaN/presence-gated, not value-compared -----------
#
# These carry no decision. They are written once, exported, max-merged and
# zero-filled for interpolated rows -- nothing downstream branches on their
# value. Every decision they could conceivably influence is caught
# INDEPENDENTLY and EXACTLY by their categorical shadow plus the structural
# checks: row counts, ``unmatched_rows == 0``, and exact TrackID / TrajectoryID
# / State / IdentityRealtimeCommitted / IdentityFinalLabel / PoseQualityState.
# A threshold crossing or an assignment flip therefore cannot hide here.
#
# This is an EXPLICIT, ENUMERATED exemption, not a return to the pre-B1 hole:
# the default is still "a shared column is compared", and every member is
# listed by name with a reason. Membership is never inferred from a name token
# -- that heuristic is exactly how ``HeadTailClassifierConf`` ended up
# string-compared.
#
# WHY NOT A TOLERANCE: the alternative was to derive a budget from a
# batch-VARYING determinism floor. Measured on courtship, the batch-induced
# drift of DetectionConfidence is NON-MONOTONE in batch -- det=2: 0.00766,
# det=4: 0.00894, det=8: 0.00811, det=16: 0.00486. A floor taken from the
# natural first perturbation (det=2) would reject det=8 (the fastest vector)
# and admit det=16 (slower), purely on which perturbation happened to measure
# it. It is also circular: the tuned coordinate would be certifying itself.
# And ``numeric_max`` is a single scalar spanning pixels, counts and
# probabilities, so one budget cannot be right for all of them.
_REPORTED_ONLY_COLUMNS = frozenset(
    {
        # A per-frame SLOT INDEX (frame_idx * STRIDE + slot), not a measurement.
        # Every consumer tests equality only WITHIN a run, and forward/backward
        # share a batch size, so a global slot shift preserves every equality.
        # tools/equivalence/compare.py -- the repository's certified
        # byte-identity gate -- uses it purely as a grouping id and never
        # compares its value.
        "DetectionID",
        # Detector/assigner scores. The decisions they feed (which detections
        # survive, which track takes which detection) are already pinned by row
        # counts, 0 unmatched, and exact TrackID/TrajectoryID/State.
        "DetectionConfidence",
        "AssignmentConfidence",
        "PositionUncertainty",
        # Realtime identity scores. The decision is IdentityRealtimeCommitted,
        # which is compared EXACTLY.
        "IdentityRealtimeConfidence",
        "IdentityRealtimeMargin",
        "IdentityRealtimeEntropy",
        # Pose quality score. Its decision is PoseQualityState, exact.
        "PoseQualityScore",
        # Pure aggregates of the per-keypoint confidences below. Their integer
        # shadow PoseNumValid (and PoseNumKeypoints) stays value-compared, so a
        # pose run that actually lost keypoints is still rejected.
        "PoseMeanConf",
        "PoseValidFraction",
    }
)


def _is_reported_only(column: str) -> bool:
    """Reported-only columns: per-keypoint confidences plus the named set.

    ``PoseKpt_<name>_Conf`` is matched structurally (exporter-owned prefix and
    suffix), not by a substring token.
    """

    if column in _REPORTED_ONLY_COLUMNS:
        return True
    return column.startswith(_KEYPOINT_PREFIX) and column.endswith("_Conf")


_KEYPOINT_PREFIX = "PoseKpt_"


def _keypoint_pairs(columns: Iterable[str]) -> tuple[tuple[str, str, str], ...]:
    """``(name, x_column, y_column)`` for every exported pose keypoint.

    The exporter writes ``PoseKpt_<keypoint>_X`` / ``_Y`` / ``_Conf``. Only
    complete X/Y pairs are returned: a keypoint present on one axis only is
    a schema difference, which the column-set check already rejects.
    """

    available = set(columns)
    output = []
    for column in columns:
        if not column.startswith(_KEYPOINT_PREFIX) or not column.endswith("_X"):
            continue
        partner = column[: -len("_X")] + "_Y"
        if partner in available:
            output.append((column[len(_KEYPOINT_PREFIX) : -len("_X")], column, partner))
    return tuple(sorted(output))


def _is_angular_column(column: str) -> bool:
    """Columns carrying an angle in radians, compared with pi-wrapping.

    ``Theta`` is excluded: the positional pass already compares it against
    the Hungarian-matched partner. ``HeadTail*`` columns are excluded because
    ``_categorical_columns`` claims them for an exact comparison, which is
    strictly stronger than a tolerance.
    """

    normalized = column.lower().replace("_", "")
    if normalized == "theta" or _is_headtail_column(column):
        return False
    return normalized.endswith("rad") or normalized.startswith("heading")


def _numeric_families(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[tuple[tuple[str, str, str], ...], tuple[str, ...], tuple[str, ...]]:
    """Split the shared numeric columns into (keypoints, angular, scalar).

    The default is INVERTED relative to the pre-B1 gate: a shared numeric
    column is compared unless something else already owns it. Previously a
    column was compared only if a name heuristic opted it in, which left
    every ``PoseKpt_*`` and every confidence ungated while
    ``pose_batch_size`` was a tuned coordinate.
    """

    shared = [column for column in reference.columns if column in candidate.columns]
    numeric = [
        column
        for column in shared
        if pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(candidate[column])
    ]
    owned = set(_POSITIONAL_COLUMNS) | set(_MANDATORY_EXACT_COLUMNS)
    owned |= set(_categorical_columns(shared))
    keypoints = tuple(item for item in _keypoint_pairs(numeric) if item[1] not in owned)
    keypoint_columns = {column for _name, x, y in keypoints for column in (x, y)}
    angular = tuple(
        column
        for column in numeric
        if column not in owned
        and column not in keypoint_columns
        and _is_angular_column(column)
    )
    scalar = tuple(
        column
        for column in numeric
        if column not in owned
        and column not in keypoint_columns
        and column not in angular
        and not _is_reported_only(column)
    )
    return keypoints, angular, scalar


def _exactly_compared_columns(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[str, ...]:
    """Shared columns compared as exact strings: categoricals + every non-numeric.

    Sorted for a deterministic ``details`` order.
    """

    shared = [column for column in left.columns if column in right.columns]
    selected = set(_categorical_columns(shared))
    for column in shared:
        if not (
            pd.api.types.is_numeric_dtype(left[column])
            and pd.api.types.is_numeric_dtype(right[column])
        ):
            selected.add(column)
    return tuple(sorted(selected))


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


def _stable_key_sort(frame: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """``frame`` sorted by ``key``, stably, with a NaN-safe composite ordering.

    The ordering is lexicographic on the rendered key rather than numeric --
    irrelevant here, because both sides are permuted by the same rule and the
    aligner only needs a deterministic correspondence, not a meaningful order.
    """

    tokens = pd.Series(
        ["\x1f".join(row) for row in _key_strings(frame, key)], index=frame.index
    )
    order = tokens.sort_values(kind="stable").index
    return frame.loc[order].reset_index(drop=True)


def _aligned(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    key = _row_key(reference)
    if key is None or any(column not in candidate.columns for column in key):
        return None
    # Sort on ONE composite key column: ``sort_values(kind=...)`` is documented
    # to apply only when sorting by a single label, so a multi-column sort could
    # not be pinned to a stable algorithm. Rows sharing a key (three lost tracks
    # in one frame all key on ``(frame, NaN)``) must keep their input order on
    # both sides. The NaN-position pairing below does not rely on this -- it
    # sorts by content explicitly.
    left = _stable_key_sort(reference, key)
    right = _stable_key_sort(candidate, key)
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


def _paired_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    pairs: tuple[tuple[object, object], ...],
    aligned: tuple[pd.DataFrame, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The two frames row-aligned by the positional pairing, else the keyed one.

    Same preference order as :func:`_mandatory_exact_mismatches`: the
    Hungarian pairing is authoritative because a key-based alignment can make
    the very column under test equal by construction.
    """

    if pairs:
        left = reference.loc[[old for old, _new in pairs]].reset_index(drop=True)
        right = candidate.loc[[new for _old, new in pairs]].reset_index(drop=True)
        return left, right
    return aligned


def _numeric_mismatches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    policy: EquivalencePolicy,
    keypoint_limit: float,
    angle_limit: float,
    numeric_limit: float,
    name: str,
) -> tuple[float, int, float, list[str]]:
    """Compare the pose/confidence product on already-paired rows.

    Returns ``(keypoint_p99, keypoints_over_gate, numeric_max, details)``.
    Rows where either side is NaN are skipped: the NaN-pattern check already
    rejects a one-sided NaN, and two NaNs carry no numeric information.
    """

    keypoints, angular, scalar = _numeric_families(left, right)
    details: list[str] = []
    keypoint_samples: list[float] = []
    over_gate = 0
    numeric_max = 0.0

    for keypoint, x_column, y_column in keypoints:
        old_x = left[x_column].to_numpy(float)
        old_y = left[y_column].to_numpy(float)
        new_x = right[x_column].to_numpy(float)
        new_y = right[y_column].to_numpy(float)
        usable = ~(
            np.isnan(old_x) | np.isnan(old_y) | np.isnan(new_x) | np.isnan(new_y)
        )
        if not usable.any():
            continue
        distance = np.hypot(
            old_x[usable] - new_x[usable], old_y[usable] - new_y[usable]
        )
        keypoint_samples.extend(distance.tolist())
        beyond = int(np.count_nonzero(distance > policy.match_gate))
        if beyond:
            over_gate += beyond
            details.append(
                f"{name}: keypoint {keypoint} moved beyond the {policy.match_gate:g}px "
                f"match gate on {beyond} row(s) (max {float(distance.max()):.6g}px)"
            )

    keypoint_p99 = (
        float(np.percentile(keypoint_samples, 99)) if keypoint_samples else 0.0
    )
    if keypoint_p99 > keypoint_limit:
        details.append(
            f"{name}: keypoint positional p99 {keypoint_p99:.6g}px exceeds limit "
            f"{keypoint_limit:.6g}px"
        )

    for column in angular:
        old = left[column].to_numpy(float)
        new = right[column].to_numpy(float)
        usable = ~(np.isnan(old) | np.isnan(new))
        if not usable.any():
            continue
        # (-pi, pi] wrapping, so 0 rad and 2*pi rad agree.
        delta = np.abs((old[usable] - new[usable] + np.pi) % (2 * np.pi) - np.pi)
        worst = float(delta.max())
        if worst > angle_limit:
            details.append(
                f"{name}: {column} angular per-row max {worst:.6g} rad exceeds limit "
                f"{angle_limit:.6g} rad"
            )

    for column in scalar:
        old = left[column].to_numpy(float)
        new = right[column].to_numpy(float)
        usable = ~(np.isnan(old) | np.isnan(new))
        if not usable.any():
            continue
        worst = float(np.abs(old[usable] - new[usable]).max())
        numeric_max = max(numeric_max, worst)
        if worst > numeric_limit:
            details.append(
                f"{name}: {column} per-row max |delta| {worst:.6g} exceeds limit "
                f"{numeric_limit:.6g}"
            )

    return keypoint_p99, over_gate, numeric_max, details


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
    keypoint_limit: float,
    numeric_limit: float,
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
            # Every shared NON-numeric column is compared exactly, on top of
            # the token-matched categorical set. Before B1 the string product
            # was opt-in by name heuristic, which left PoseQualityState,
            # PoseQualityFlags, PoseSource and HeadingMethod free to change
            # while pose_batch_size was tuned. Numeric columns get a
            # tolerance (see _numeric_mismatches); non-numeric ones have no
            # meaningful tolerance, so exact is the only honest rule.
            for column in _exactly_compared_columns(left, right):
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
    keypoint_p99 = 0.0
    keypoints_over_gate = 0
    numeric_max = 0.0
    product_details: list[str] = []
    # The measured determinism floor must exclude the repository's documented
    # bistable head/tail pi-flip rows from the NUMERIC families too, not just
    # from the angular statistic. A flipped row relabels which keypoint is the
    # head, so its keypoint XY and its PoseKpt_*_Conf legitimately swap -- and
    # a floor computed over those rows would come back with
    # numeric_max ~ 1.0, which
    # ``numeric_limit = max(policy, floor.numeric_max)`` would then hand to
    # EVERY later candidate as a budget. That is exactly the disarm bug
    # already fixed once for the angle gate
    # (test_determinism_floor_is_not_disarmed_by_a_bistable_pi_flip), one
    # family over.
    product_pairs = tuple(
        pair for pair in metrics["pairs"] if pair not in excluded_pairs
    )
    # If EVERY positional pair was a head/tail flip, ``product_pairs`` is
    # empty and _paired_frames would fall back to the KEYED alignment -- which
    # re-admits exactly the rows just excluded and re-inflates the floor. That
    # is the same shape as the disarm bug above. The keyed fallback is only
    # legitimate when there was no positional pairing to begin with.
    product = (
        _paired_frames(reference, candidate, pairs=product_pairs, aligned=aligned)
        if (product_pairs or not metrics["pairs"])
        else None
    )
    if product is not None:
        keypoint_p99, keypoints_over_gate, numeric_max, product_details = (
            _numeric_mismatches(
                product[0],
                product[1],
                policy=policy,
                keypoint_limit=keypoint_limit,
                angle_limit=angle_limit,
                numeric_limit=numeric_limit,
                name=name,
            )
        )
        details.extend(product_details)
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
        # Every pose-product failure -- keypoint p99, keypoint match gate,
        # angular columns, scalar columns -- records a detail line naming the
        # column, so an empty list is exactly "the product agreed".
        and not product_details
    )
    return EquivalenceVerdict(
        passed=passed,
        nonzero_rows=nonzero,
        row_counts_match=counts_match,
        unmatched_rows=int(metrics["unmatched"]),
        position_p99=float(metrics["position_p99"]),
        angle_max=angle_max,
        keypoint_p99=keypoint_p99,
        keypoints_over_gate=keypoints_over_gate,
        numeric_max=numeric_max,
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
    # The pose product widens with the measured floor exactly as positions and
    # angles do: a platform whose own A-vs-A repeat moves a keypoint must not
    # then reject itself.
    keypoint_limit = max(
        policy.keypoint_p99_tolerance,
        determinism_floor.keypoint_p99 if determinism_floor is not None else 0.0,
    )
    numeric_limit = max(
        policy.numeric_max_tolerance,
        determinism_floor.numeric_max if determinism_floor is not None else 0.0,
    )
    verdicts = (
        _compare_one(
            reference.forward,
            candidate.forward,
            policy=policy,
            position_limit=position_limit,
            angle_limit=angle_limit,
            keypoint_limit=keypoint_limit,
            numeric_limit=numeric_limit,
            name="forward",
            exclude_headtail_flip_rows=for_determinism_floor,
        ),
        _compare_one(
            reference.final,
            candidate.final,
            policy=policy,
            position_limit=position_limit,
            angle_limit=angle_limit,
            keypoint_limit=keypoint_limit,
            numeric_limit=numeric_limit,
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
        keypoint_p99=max(item.keypoint_p99 for item in verdicts),
        keypoints_over_gate=sum(item.keypoints_over_gate for item in verdicts),
        numeric_max=max(item.numeric_max for item in verdicts),
        nan_pattern_mismatches=sum(item.nan_pattern_mismatches for item in verdicts),
        categorical_mismatches=sum(item.categorical_mismatches for item in verdicts),
        details=tuple(detail for item in verdicts for detail in item.details),
    )
