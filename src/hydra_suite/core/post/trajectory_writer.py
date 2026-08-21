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


def _non_identity_classifier_columns(df: pd.DataFrame, cnn_classifiers) -> list:
    """``(source_col, user_col)`` for every NON-identity classifier column.

    An identity classifier's contribution to the User export is the resolved
    ``identity``; its per-frame calls are evidence and belong in the Debug
    export. A behavior/sex/caste classifier has no such channel -- it is
    output, not identity -- so without this its results never reach a
    User-mode user at all: the classifier runs, costs inference time, and is
    discarded at export.

    Names are lowercased into the clean schema's style
    (``CNN_behavior_Class`` -> ``behavior_class``), matching how pose becomes
    ``<kpt>_x``/``_y``/``_conf``.
    """
    from hydra_suite.core.individual.identity.heads import (
        identity_class_columns,
        identity_head_labels,
    )

    if not cnn_classifiers:
        return []
    identity_labels = set(identity_head_labels(cnn_classifiers))
    all_labels = tuple(
        str(cfg.get("label", "") or "").strip() or "cnn" for cfg in cnn_classifiers
    )
    output_labels = tuple(
        lbl for lbl in all_labels if lbl and lbl not in identity_labels
    )
    if not output_labels:
        return []

    pairs = []
    taken = set()
    for class_col in identity_class_columns(df.columns, output_labels, all_labels):
        # One stem for both columns: `_Class` and `_Conf` are different
        # lengths, so slicing each by its own suffix silently truncates the
        # confidence name by a character.
        stem = class_col[len("CNN_") : -len("_Class")].lower()
        conf_col = f"{class_col[: -len('_Class')]}_Conf"
        for src, user_col in ((class_col, f"{stem}_class"), (conf_col, f"{stem}_conf")):
            if src not in df.columns or user_col in taken:
                continue
            taken.add(user_col)
            pairs.append((src, user_col))
    return pairs


def _directed_flag(series: pd.Series) -> pd.Series:
    """Coerce a ``HeadingIsDirected`` column to a nullable boolean.

    Rows with no detection carry no head-tail evidence at all, and those
    must stay ``<NA>`` rather than collapse to ``False`` -- "no direction
    was resolved here" and "we never looked" are different claims. A plain
    ``astype(bool)`` cannot express that, and on a CSV round-trip (where the
    column arrives as the strings ``"True"``/``"False"``) it would report
    every row as directed, since any non-empty string is truthy.
    """
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    known = series.notna()
    if not known.any():
        return out
    values = series[known]
    if values.dtype == object or pd.api.types.is_string_dtype(values):
        text = values.astype("string").str.strip().str.lower()
        out[known] = text.map({"true": True, "false": False}).astype("boolean")
    else:
        out[known] = values.astype(bool)
    return out


def project_user_tracks(
    df: pd.DataFrame,
    *,
    fps: float | None,
    identity_ran: bool = True,
    cnn_classifiers=None,
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
    # Arena — present only on multi-arena runs (the engine appends the column
    # when n_arenas > 1). Without it, every per-arena grouping a user wants
    # from this file is impossible: trajectory ids are globally unique but
    # carry no arena, so a 24-arena plate exports as one undifferentiated
    # pool. Single-arena runs have no such column, so their schema is
    # unchanged.
    if "arena_id" in df.columns:
        out["arena_id"] = pd.to_numeric(df["arena_id"], errors="coerce").astype("Int64")
    out["frame"] = pd.to_numeric(df["FrameID"], errors="coerce").round().astype("Int64")
    if fps and float(fps) > 0:
        out["time_s"] = out["frame"].astype("Float64") / float(fps)
    else:
        out["time_s"] = pd.array([np.nan] * len(df), dtype="Float64")
    out["x"] = df["X"]
    out["y"] = df["Y"]
    theta = pd.to_numeric(df["Theta"], errors="coerce")
    out["heading_deg"] = np.mod(np.degrees(theta), 360.0)
    # Is `heading_deg` a real head-forward direction, or only a body axis?
    # `Theta` already carries the pose/head-tail resolution when one was
    # available (the directed angle is substituted for the OBB axis before
    # the Kalman update), but where no model resolved a direction it stays
    # the axis -- meaningful mod 180, not 360. Both kinds land in the same
    # column, indistinguishable, so a turning-rate or mean-heading
    # calculation over the undirected rows silently averages nonsense.
    #
    # Per-row semantics: "a model resolved head-vs-tail on THIS row".
    # Post-processing's global flip-fixer propagates direction along a whole
    # trajectory, so an undirected row can still inherit a correct heading
    # -- this flag therefore under-reports rather than over-reports. Group
    # by `id` for a per-track answer.
    if "HeadingIsDirected" in df.columns:
        out["heading_is_directed"] = _directed_flag(df["HeadingIsDirected"])
    out["state"] = df["State"]
    out["detection_confidence"] = df.get("DetectionConfidence")

    # Identity — only when an identity/tag method actually ran AND the
    # resolved-final label column is present.
    if identity_ran and C.FINAL_LABEL in df.columns:
        label = df[C.FINAL_LABEL].astype("string")
        from_smoothed = pd.Series(False, index=df.index)
        if C.FINAL_SMOOTHED_LABEL in df.columns:
            smoothed = df[C.FINAL_SMOOTHED_LABEL].astype("string")
            # Only rows that actually take the smoothed label count as
            # smoothed-sourced: an empty Final *and* an empty Smoothed stays
            # empty and must keep its own (Final) confidence.
            from_smoothed = _is_empty_label(label) & ~_is_empty_label(smoothed)
            label = label.mask(_is_empty_label(label), smoothed)
        out["identity"] = label
        # ``identity`` is a *display* label and is not unique on its own: a
        # class declared non-identifying (an untagged animal, an unreadable
        # tag) is deliberately shared by every track carrying it. The resolved
        # identity *slot* is what disambiguates -- ``0`` is the unknown slot,
        # never a real individual -- so it travels with the label instead of
        # living only in the rich export. Without it, a reader who groups this
        # file by ``identity`` silently merges every untagged animal into one.
        if C.FINAL_ID in df.columns:
            out["identity_id"] = pd.to_numeric(df[C.FINAL_ID], errors="coerce").astype(
                "Int64"
            )
        if C.FINAL_CONFIDENCE in df.columns:
            confidence = df[C.FINAL_CONFIDENCE]
            # A row whose label came from the smoothed column must report the
            # smoothed confidence with it. Reading FINAL_CONFIDENCE for those
            # rows pairs one estimator's label with another's score -- and
            # FINAL_CONFIDENCE is exactly the field that is empty/0 there,
            # so the smoothed labels looked like the least trustworthy rows.
            if C.FINAL_SMOOTHED_CONFIDENCE in df.columns and from_smoothed.any():
                confidence = confidence.mask(
                    from_smoothed, df[C.FINAL_SMOOTHED_CONFIDENCE]
                )
            out["identity_confidence"] = confidence
        if C.FINAL_SOURCE in df.columns:
            out["identity_source"] = df[C.FINAL_SOURCE]

    # Non-identity classifier output (behavior, sex, caste, ...). Identity
    # heads are excluded: `identity` above already carries their result.
    for src_col, user_col in _non_identity_classifier_columns(df, cnn_classifiers):
        out[user_col] = df[src_col]

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
    cnn_classifiers=None,
) -> str | None:
    """Terminal trajectory writer. Debug → `_with_individual.csv`; User → `<stem>_tracks.csv`.

    ``identity_ran`` is forwarded to `project_user_tracks` for the User
    branch only; the Debug branch's rich export always includes every
    resolved column regardless of whether identity actually ran.
    """
    if debug_mode:
        from hydra_suite.core.post.rich_export import write_rich_export_csv

        return write_rich_export_csv(rich_df, final_csv_path)
    clean = project_user_tracks(
        rich_df,
        fps=fps,
        identity_ran=identity_ran,
        cnn_classifiers=cnn_classifiers,
    )
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
