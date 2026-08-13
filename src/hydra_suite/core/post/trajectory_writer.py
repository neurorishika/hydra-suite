"""Mode-aware terminal trajectory writers (User-mode clean CSV + Debug base-final CSV)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C

_POSE_PREFIX = "PoseKpt_"
_POSE_X_SUFFIX = "_X"
_FINAL_SUFFIXES = ("_final", "_forward_processed")


def _is_empty_label(series: pd.Series) -> pd.Series:
    """True where a label is missing or an empty/whitespace string."""
    s = series.astype("string")
    return s.isna() | (s.str.strip().str.len() == 0)


def project_user_tracks(
    df: pd.DataFrame, *, fps: float | None, identity_ran: bool = True
) -> pd.DataFrame:
    """Project the full trajectory DataFrame to the clean User-mode schema.

    ``identity_ran`` gates the identity block (``identity``/
    ``identity_confidence``/``identity_source``): even when a resolved-final
    label column is present in ``df``, it is only emitted when an
    identity/tag method actually ran. Rich-export's identity postprocessing
    unconditionally resolves ``C.FINAL_LABEL`` to a placeholder (``"unknown"``)
    whenever the identity-postprocessing pipeline is enabled, regardless of
    whether a real method (CNN classifier, AprilTags, ...) executed, so
    column presence alone is not a reliable signal.
    """
    out = pd.DataFrame(index=df.index)
    out["id"] = df["TrajectoryID"]
    out["frame"] = pd.to_numeric(df["FrameID"], errors="coerce").round().astype("Int64")
    if fps and float(fps) > 0:
        out["time_s"] = out["frame"].astype("Float64") / float(fps)
    else:
        out["time_s"] = pd.array([np.nan] * len(df), dtype="Float64")
    out["x"] = df["X"]
    out["y"] = df["Y"]
    theta = pd.to_numeric(df["Theta"], errors="coerce")
    out["heading_deg"] = np.mod(np.degrees(theta), 360.0)
    out["state"] = df["State"]
    out["detection_confidence"] = df.get("DetectionConfidence")

    # Identity — only when an identity/tag method actually ran AND the
    # resolved-final label column is present.
    if identity_ran and C.FINAL_LABEL in df.columns:
        label = df[C.FINAL_LABEL].astype("string")
        if C.FINAL_SMOOTHED_LABEL in df.columns:
            label = label.mask(
                _is_empty_label(label), df[C.FINAL_SMOOTHED_LABEL].astype("string")
            )
        out["identity"] = label
        if C.FINAL_CONFIDENCE in df.columns:
            out["identity_confidence"] = df[C.FINAL_CONFIDENCE]
        if C.FINAL_SOURCE in df.columns:
            out["identity_source"] = df[C.FINAL_SOURCE]

    # Pose — one <kpt>_x/_y/_conf triple per PoseKpt_<name>_X column present.
    pose_x_cols = [
        c
        for c in df.columns
        if c.startswith(_POSE_PREFIX) and c.endswith(_POSE_X_SUFFIX)
    ]
    for xcol in pose_x_cols:
        name = xcol[len(_POSE_PREFIX) : -len(_POSE_X_SUFFIX)]
        out[f"{name}_x"] = df.get(f"{_POSE_PREFIX}{name}_X")
        out[f"{name}_y"] = df.get(f"{_POSE_PREFIX}{name}_Y")
        out[f"{name}_conf"] = df.get(f"{_POSE_PREFIX}{name}_Conf")

    return out


def user_tracks_path(final_csv_path: str) -> str:
    """Derive the clean `<stem>_tracks.csv` path from a debug final-CSV path."""
    base, ext = os.path.splitext(final_csv_path)
    for suffix in _FINAL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}_tracks{ext}"


def write_final_trajectories(
    rich_df: pd.DataFrame,
    final_csv_path: str,
    *,
    debug_mode: bool,
    fps: float | None,
    identity_ran: bool = True,
) -> str | None:
    """Terminal trajectory writer. Debug → `_with_individual.csv`; User → `<stem>_tracks.csv`.

    ``identity_ran`` is forwarded to `project_user_tracks` for the User
    branch only; the Debug branch's rich export always includes every
    resolved column regardless of whether identity actually ran.
    """
    if debug_mode:
        from hydra_suite.core.post.rich_export import write_rich_export_csv

        return write_rich_export_csv(rich_df, final_csv_path)
    clean = project_user_tracks(rich_df, fps=fps, identity_ran=identity_ran)
    out_path = user_tracks_path(final_csv_path)
    clean.to_csv(out_path, index=False)
    return out_path


def write_base_final_csv(df: pd.DataFrame, output_path: str) -> bool:
    """Write the debug base-final CSV: round X/Y/FrameID, drop TrackID/Index, reorder."""
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected post-processed trajectories as a pandas DataFrame.")
    if df.empty:
        return False

    df_to_save = df.copy()
    for column in ["X", "Y", "FrameID"]:
        if column in df_to_save.columns:
            df_to_save[column] = pd.to_numeric(df_to_save[column], errors="coerce")
            df_to_save[column] = df_to_save[column].round().astype("Int64")

    df_to_save = df_to_save.drop(
        columns=[c for c in ["TrackID", "Index"] if c in df_to_save.columns],
        errors="ignore",
    )
    base_columns = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    ordered_columns = base_columns + [
        c for c in df_to_save.columns if c not in base_columns
    ]
    df_to_save[ordered_columns].to_csv(output_path, index=False)
    return True
