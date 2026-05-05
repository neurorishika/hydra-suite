"""
Atomic correction writer for _proofread.csv.

Applies identity corrections to the proofread copy.
Supports: split+swap, merge fragments, delete track, erase flicker,
reassign chain (N-way relabeling), and fragment-level edit ops from the
track editor.
Never touches the original CSV.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from hydra_suite.core.post.processing import interpolate_trajectories
from hydra_suite.refinekit.core.track_editor_model import EditOp, OpKind

logger = logging.getLogger(__name__)

_NEW_ID_OFFSET = 100_000
_TRACKING_CSV_SUFFIXES = (
    "_tracking_final_with_individual",
    "_tracking_final_with_pose",
    "_tracking_final",
    "_tracking_forward_processed",
    "_tracking_forward",
    "_tracking_backward_processed",
    "_tracking_backward",
)


def _config_path_for_csv(source_csv: Path) -> Path:
    stem = source_csv.stem
    for suffix in _TRACKING_CSV_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return source_csv.with_name(f"{stem}_config.json")


def _config_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


# ---------------------------------------------------------------------------
# Atomic operations (pure functions on DataFrames)
# ---------------------------------------------------------------------------


def apply_split_and_swap(
    df: pd.DataFrame,
    track_a: int,
    track_b: int,
    split_frame: int,
    swap_post: bool,
) -> pd.DataFrame:
    """Split two tracks at ``split_frame``, optionally swapping post-split IDs."""
    df = df.copy()
    mask_a_post = (df["TrajectoryID"] == track_a) & (df["FrameID"] >= split_frame)
    mask_b_post = (df["TrajectoryID"] == track_b) & (df["FrameID"] >= split_frame)

    if swap_post:
        df.loc[mask_a_post, "TrajectoryID"] = track_b + _NEW_ID_OFFSET
        df.loc[mask_b_post, "TrajectoryID"] = track_a + _NEW_ID_OFFSET
    else:
        df.loc[mask_a_post, "TrajectoryID"] = track_a + _NEW_ID_OFFSET
        df.loc[mask_b_post, "TrajectoryID"] = track_b + _NEW_ID_OFFSET

    return df


def merge_fragments(
    df: pd.DataFrame,
    track_ids: List[int],
) -> pd.DataFrame:
    """Merge multiple trajectory IDs into the lowest ID.

    All rows whose TrajectoryID is in *track_ids* are relabeled to
    ``min(track_ids)``.
    """
    if len(track_ids) < 2:
        return df
    df = df.copy()
    target = min(track_ids)
    for tid in track_ids:
        if tid != target:
            df.loc[df["TrajectoryID"] == tid, "TrajectoryID"] = target
    return df


class CorrectionWriter:
    """
    Manages the _proofread.csv lifecycle: open -> apply corrections -> close.

    Creates a proofread copy once from the original CSV. Subsequent opens
    load the existing proofread copy without overwriting.
    """

    def __init__(self, source_csv: Path | str):
        self.source_csv = Path(source_csv)
        stem = self.source_csv.stem
        self.proofread_path = self.source_csv.with_name(f"{stem}_proofread.csv")
        self._df: pd.DataFrame | None = None
        self._interpolation_settings = self._load_interpolation_settings()

    def open(self) -> None:
        """Create proofread copy if needed, then load it into memory."""
        if not self.proofread_path.exists():
            shutil.copy2(self.source_csv, self.proofread_path)
            logger.info("Created proofread copy: %s", self.proofread_path)
        loaded_df = pd.read_csv(self.proofread_path)
        self._df = self._normalize_df(loaded_df)
        if not loaded_df.equals(self._df):
            self._write_atomic()

    def apply_merge(self, track_ids: List[int]) -> None:
        """Merge fragment track IDs into one and persist."""
        if self._df is None:
            raise RuntimeError("Call open() before apply_merge()")
        self._commit_df(merge_fragments(self._df, track_ids))

    def apply_correction(
        self,
        track_a: int,
        track_b: int,
        split_frame: int,
        swap_post: bool,
    ) -> None:
        """Apply the legacy split-and-swap correction workflow and persist."""
        if self._df is None:
            raise RuntimeError("Call open() before apply_correction()")
        self._commit_df(
            apply_split_and_swap(
                self._df,
                track_a=track_a,
                track_b=track_b,
                split_frame=split_frame,
                swap_post=swap_post,
            )
        )

    def apply_swap_merge(
        self,
        source_id: int,
        target_id: int,
        swap_frame: int,
    ) -> None:
        """Fix an identity swap by relabeling *target*'s post-swap rows.

        After this operation:

        * *source_id* has a continuous trajectory (original pre-swap data
          plus *target*'s post-swap data).
        * *target_id* ends at ``swap_frame - 1`` (becomes a dead fragment).
        * Any (wrong) detections attributed to *source* at or after
          *swap_frame* are removed.
        """
        if self._df is None:
            raise RuntimeError("Call open() before apply_swap_merge()")
        df = self._df.copy()

        # Remove source's rows at/after swap_frame (they were wrong)
        mask_remove = (df["TrajectoryID"] == source_id) & (df["FrameID"] >= swap_frame)
        df = df[~mask_remove]

        # Relabel target's post-swap rows to source_id
        mask_relabel = (df["TrajectoryID"] == target_id) & (df["FrameID"] >= swap_frame)
        df.loc[mask_relabel, "TrajectoryID"] = source_id

        self._commit_df(df)

    def _load_interpolation_settings(self) -> Optional[dict[str, Any]]:
        config_path = _config_path_for_csv(self.source_csv)
        if not config_path.is_file():
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            logger.warning("Failed to read tracking config from %s", config_path)
            return None

        method_raw = _config_value(data, "interpolation_method", "INTERPOLATION_METHOD")
        method = str(method_raw or "none").strip().lower()
        if method in {"", "none", "null"}:
            return None

        gap_raw = _config_value(
            data,
            "interpolation_max_gap_seconds",
            "interpolation_max_gap",
            "INTERPOLATION_MAX_GAP_SECONDS",
            "INTERPOLATION_MAX_GAP",
        )
        try:
            gap_seconds = float(gap_raw)
        except (TypeError, ValueError):
            return None
        if gap_seconds <= 0:
            return None

        fps_raw = _config_value(data, "fps", "FPS")
        try:
            fps = float(fps_raw)
        except (TypeError, ValueError):
            fps = 0.0
        if fps > 0:
            max_gap = max(1, int(round(gap_seconds * fps)))
        else:
            max_gap = max(1, int(round(gap_seconds)))

        burst_raw = _config_value(
            data,
            "heading_flip_max_burst",
            "HEADING_FLIP_MAX_BURST",
        )
        try:
            heading_flip_max_burst = max(1, int(burst_raw))
        except (TypeError, ValueError):
            heading_flip_max_burst = 5

        directed_heading_posthoc = bool(
            _config_value(
                data,
                "DIRECTED_ORIENT_POSTHOC_CONSISTENCY",
                "directed_orient_posthoc_consistency",
            )
        )

        return {
            "method": method,
            "max_gap": max_gap,
            "heading_flip_max_burst": heading_flip_max_burst,
            "directed_heading_posthoc": directed_heading_posthoc,
        }

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        settings = self._interpolation_settings
        normalized = df.reset_index(drop=True)
        if settings is None or normalized.empty:
            return normalized

        required_columns = {"FrameID", "TrajectoryID", "X", "Y"}
        if not required_columns.issubset(normalized.columns):
            return normalized

        original_columns = list(normalized.columns)
        added_theta = False
        if "Theta" not in normalized.columns:
            normalized = normalized.copy()
            normalized["Theta"] = float("nan")
            added_theta = True

        try:
            normalized = interpolate_trajectories(
                normalized,
                method=settings["method"],
                max_gap=settings["max_gap"],
                heading_flip_max_burst=settings["heading_flip_max_burst"],
                directed_heading_posthoc=settings["directed_heading_posthoc"],
            )
        except Exception:
            logger.exception(
                "Failed to interpolate RefineKit proofread trajectories using tracking config"
            )
            return df.reset_index(drop=True)

        if added_theta:
            normalized = normalized.drop(columns=["Theta"], errors="ignore")
        normalized = normalized.reindex(columns=original_columns)
        return normalized.reset_index(drop=True)

    def _commit_df(self, df: pd.DataFrame) -> None:
        self._df = self._normalize_df(df)
        self._write_atomic()

    def _write_atomic(self) -> None:
        """Write to .tmp then atomically replace the proofread file."""
        tmp = self.proofread_path.with_suffix(".tmp")
        self._df.to_csv(tmp, index=False)
        os.replace(tmp, self.proofread_path)

    def apply_edit_ops(self, ops: List[EditOp]) -> None:
        """Apply a batch of fragment-level edit operations and persist.

        Operations are applied in order: DELETEs first, then REASSIGNs.
        """
        if self._df is None:
            raise RuntimeError("Call open() before apply_edit_ops()")
        df = self._df.copy()

        # Deletes first (so reassign doesn't move rows we want to remove)
        for op in ops:
            if op.kind == OpKind.DELETE:
                mask = (
                    (df["TrajectoryID"] == op.track_id)
                    & (df["FrameID"] >= op.frame_start)
                    & (df["FrameID"] <= op.frame_end)
                )
                df = df[~mask]

        # Reassigns — use temp offset to avoid collision
        offset = _NEW_ID_OFFSET
        for op in ops:
            if op.kind == OpKind.REASSIGN and op.new_track_id is not None:
                mask = (
                    (df["TrajectoryID"] == op.track_id)
                    & (df["FrameID"] >= op.frame_start)
                    & (df["FrameID"] <= op.frame_end)
                )
                df.loc[mask, "TrajectoryID"] = op.new_track_id + offset

        # Resolve temp IDs
        df.loc[
            df["TrajectoryID"] >= offset,
            "TrajectoryID",
        ] = (
            df.loc[df["TrajectoryID"] >= offset, "TrajectoryID"] - offset
        )

        self._commit_df(df.reset_index(drop=True))

    def close(self) -> None:
        """Release the in-memory DataFrame."""
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Return the current in-memory DataFrame."""
        if self._df is None:
            raise RuntimeError("Call open() first")
        return self._df
