"""Mode-aware terminal trajectory writers (User-mode clean CSV + Debug base-final CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C

_POSE_PREFIX = "PoseKpt_"
_POSE_X_SUFFIX = "_X"


def _is_empty_label(series: pd.Series) -> pd.Series:
    """True where a label is missing or an empty/whitespace string."""
    s = series.astype("string")
    return s.isna() | (s.str.strip().str.len() == 0)


def project_user_tracks(df: pd.DataFrame, *, fps: float | None) -> pd.DataFrame:
    """Project the full trajectory DataFrame to the clean User-mode schema."""
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

    # Identity — only when the resolved-final label column is present.
    if C.FINAL_LABEL in df.columns:
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
