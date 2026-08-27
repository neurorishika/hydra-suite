#!/usr/bin/env python3
"""Acceptance checker for a `..._tracking_final_with_individual.csv` export.

Audits the final identity-annotated tracking CSV for the structural and
identity-consistency invariants the identity-final-consistency branch is
supposed to guarantee, and prints a labeled report. See
`docs/superpowers/specs/2026-08-27-identity-final-consistency-design.md` for
the acceptance criteria this checker encodes.

Usage:
    python scripts/audit_final_csv.py <csv> [--tracks ID [ID ...]]

Exit code 1 if any of the following is non-zero: NaN-position row count,
leading/trailing-NaN-run count, missing-interior-frame count, empty
IdentityFinalSource count, NaN IdentityFinalConflictResolved count, or
multi-identity-trajectory count. Exit code 0 otherwise.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

UNKNOWN_LABELS = {"unknown", "", None}


def _is_unknown(label) -> bool:
    if pd.isna(label):
        return True
    return str(label).strip().lower() in ("unknown", "")


def audit(df: pd.DataFrame, tracks: list[int] | None) -> int:
    lines: list[str] = []

    def emit(msg: str = "") -> None:
        lines.append(msg)

    total_rows = len(df)
    emit(f"Total rows: {total_rows}")
    emit(f"Total trajectories: {df['TrajectoryID'].nunique()}")
    emit("")

    # --- 1. NaN X/Y/Theta ------------------------------------------------
    nan_pos_mask = df["X"].isna() | df["Y"].isna() | df["Theta"].isna()
    nan_pos_count = int(nan_pos_mask.sum())
    emit(f"[1] Rows with NaN X, Y, or Theta: {nan_pos_count}")

    # --- 2. Leading/trailing NaN-position runs per trajectory ------------
    leading_trailing = []
    for tid, g in df.groupby("TrajectoryID", sort=False):
        g = g.sort_values("FrameID")
        pos_nan = (g["X"].isna() | g["Y"].isna() | g["Theta"].isna()).to_numpy()
        n = len(pos_nan)
        if n == 0:
            continue
        lead = 0
        while lead < n and pos_nan[lead]:
            lead += 1
        trail = 0
        while trail < n and pos_nan[n - 1 - trail]:
            trail += 1
        if lead > 0 or trail > 0:
            leading_trailing.append((tid, lead, trail))
    emit(
        f"[2] Trajectories with a leading/trailing NaN-position run: "
        f"{len(leading_trailing)}"
    )
    for tid, lead, trail in leading_trailing:
        emit(f"      t{tid}: leading={lead} trailing={trail}")

    # --- 3. Missing interior frames ---------------------------------------
    missing_interior = 0
    missing_interior_detail = []
    for tid, g in df.groupby("TrajectoryID", sort=False):
        g = g.sort_values("FrameID")
        frame_diffs = g["FrameID"].diff().dropna()
        gaps = frame_diffs[frame_diffs > 1]
        if len(gaps) > 0:
            missing_interior += len(gaps)
            missing_interior_detail.append((tid, len(gaps)))
    emit(f"[3] Missing interior frames (gaps inside a trajectory): {missing_interior}")
    for tid, n_gaps in missing_interior_detail:
        emit(f"      t{tid}: {n_gaps} gap(s)")

    # --- 4. Empty IdentityFinalSource -------------------------------------
    if "IdentityFinalSource" in df.columns:
        empty_source_mask = df["IdentityFinalSource"].isna() | (
            df["IdentityFinalSource"].astype(str).str.strip() == ""
        )
        empty_source_count = int(empty_source_mask.sum())
    else:
        empty_source_count = -1
    emit(f"[4] Rows with empty/NaN IdentityFinalSource: {empty_source_count}")
    if empty_source_count > 0:
        emit(
            "      NOTE: after this branch's fixes, blank IdentityFinalSource means "
            "legacy data — the branch never writes blank. A nonzero count on a "
            "rerun of current code is a real regression."
        )

    # --- 5. NaN IdentityFinalConflictResolved ------------------------------
    if "IdentityFinalConflictResolved" in df.columns:
        nan_conflict_count = int(df["IdentityFinalConflictResolved"].isna().sum())
    else:
        nan_conflict_count = -1
    emit(f"[5] Rows with NaN IdentityFinalConflictResolved: {nan_conflict_count}")

    # --- 6. Trajectories with >1 distinct identity label --------------------
    multi_identity_trajs = []
    for col in ("IdentityFinalLabel", "IdentityFinalID", "IdentityFinalSource"):
        if col not in df.columns:
            continue
        for tid, g in df.groupby("TrajectoryID", sort=False):
            n_distinct = g[col].dropna().nunique()
            if n_distinct > 1:
                multi_identity_trajs.append((tid, col, n_distinct))
    multi_identity_traj_ids = {tid for tid, _, _ in multi_identity_trajs}
    emit(
        f"[6] Trajectories with >1 distinct identity value (any of "
        f"IdentityFinalLabel/ID/Source): {len(multi_identity_traj_ids)}"
    )
    for tid, col, n_distinct in multi_identity_trajs:
        emit(f"      t{tid}: {col} has {n_distinct} distinct values")

    # --- 7. Labelled vs unknown trajectory/row split ------------------------
    if "IdentityFinalLabel" in df.columns:
        traj_label = df.groupby("TrajectoryID")["IdentityFinalLabel"].agg(
            lambda s: s.dropna().iloc[0] if s.dropna().size else None
        )
        labelled_traj_ids = {
            tid for tid, lab in traj_label.items() if not _is_unknown(lab)
        }
        unknown_traj_ids = set(traj_label.index) - labelled_traj_ids
        labelled_rows = int(df["TrajectoryID"].isin(labelled_traj_ids).sum())
        unknown_rows = int(df["TrajectoryID"].isin(unknown_traj_ids).sum())
        emit(
            f"[7] Labelled trajectories: {len(labelled_traj_ids)} "
            f"({labelled_rows} rows) | Unknown trajectories: "
            f"{len(unknown_traj_ids)} ({unknown_rows} rows)"
        )
    else:
        labelled_traj_ids = set()
        emit("[7] IdentityFinalLabel column not present")

    # --- 8. Label vs mode of own smoothed label -----------------------------
    if {"IdentityFinalLabel", "IdentityFinalSmoothedLabel"}.issubset(df.columns):
        disagree = 0
        checked = 0
        for tid in labelled_traj_ids:
            g = df[df["TrajectoryID"] == tid]
            own_label = g["IdentityFinalLabel"].dropna()
            own_label = own_label.iloc[0] if len(own_label) else None
            smoothed = g["IdentityFinalSmoothedLabel"]
            smoothed_present = smoothed[~smoothed.apply(_is_unknown)]
            if smoothed_present.empty or own_label is None:
                continue
            checked += 1
            mode_smoothed = smoothed_present.mode()
            if mode_smoothed.empty:
                continue
            if str(mode_smoothed.iloc[0]) != str(own_label):
                disagree += 1
        emit(
            f"[8] Labelled trajectories checked against own smoothed-label mode: "
            f"{checked}; disagreements: {disagree}"
        )
    else:
        emit("[8] IdentityFinalSmoothedLabel column not present; skipped")

    emit("")

    # --- --tracks -----------------------------------------------------------
    if tracks:
        emit("Requested track labels:")
        for tid in tracks:
            g = df[df["TrajectoryID"] == tid]
            if g.empty:
                emit(f"  t{tid}: not found")
                continue
            labels = (
                g["IdentityFinalLabel"].dropna().unique()
                if "IdentityFinalLabel" in df.columns
                else []
            )
            if len(labels) == 0:
                emit(f"  t{tid}: no IdentityFinalLabel value present")
            elif len(labels) == 1:
                emit(f"  t{tid}: {labels[0]}")
            else:
                emit(f"  t{tid}: MULTIPLE LABELS {list(labels)}")

    print("\n".join(lines))

    fail = any(
        [
            nan_pos_count > 0,
            len(leading_trailing) > 0,
            missing_interior > 0,
            empty_source_count > 0,
            nan_conflict_count > 0,
            len(multi_identity_traj_ids) > 0,
        ]
    )
    return 1 if fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", help="Path to the *_tracking_final_with_individual.csv file"
    )
    parser.add_argument(
        "--tracks",
        type=int,
        nargs="+",
        default=None,
        help="TrajectoryID(s) to print the IdentityFinalLabel for",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    return audit(df, args.tracks)


if __name__ == "__main__":
    sys.exit(main())
