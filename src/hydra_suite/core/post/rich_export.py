"""Rich-export CSV builders/writers (Qt-free).

Moved out of ``trackerkit/gui/orchestrators/tracking.py`` as part of the
headless Qt-free refactor. These compose the Task 3 (``pose_merge``) and
Task 4 (``postprocess_df``) pure functions into the rich-export pipeline:
build the merged/quality-gated/identity-postprocessed dataframe, write it
to a suffixed CSV, and relink+export in one shot.
"""

import logging
import os
import re

import pandas as pd

from hydra_suite.core.identity.postprocess_df import apply_identity_postprocessing_to_df
from hydra_suite.core.identity.properties.export import DETECTED_HEADING_COLUMNS
from hydra_suite.core.post.pose_merge import (
    apply_pose_quality_postprocessing,
    check_pose_export_sources,
    merge_pose_sources_into_df,
)

logger = logging.getLogger(__name__)

RICH_EXPORT_SUFFIX = "_with_individual"
LEGACY_RICH_EXPORT_SUFFIX = "_with_pose"


def rich_export_path(final_csv_path: str, *, legacy: bool = False) -> str:
    """Return the rich-export CSV path next to *final_csv_path*."""
    base, ext = os.path.splitext(final_csv_path)
    suffix = LEGACY_RICH_EXPORT_SUFFIX if legacy else RICH_EXPORT_SUFFIX
    return f"{base}{suffix}{ext or '.csv'}"


def write_rich_export_csv(rich_df: pd.DataFrame, final_csv_path: str) -> str | None:
    """Write the canonical rich export and remove any stale legacy alias."""
    rich_path = rich_export_path(final_csv_path)
    legacy_path = rich_export_path(final_csv_path, legacy=True)
    try:
        cleaned_df = drop_empty_rich_export_columns(rich_df)
        cleaned_df.to_csv(rich_path, index=False)
        if legacy_path != rich_path and os.path.exists(legacy_path):
            os.remove(legacy_path)
    except Exception:
        logger.exception("Failed to save rich export CSV to: %s", rich_path)
        return None

    logger.info("Rich trajectories saved to: %s", rich_path)
    if legacy_path != rich_path:
        logger.info("Legacy rich-export alias removed: %s", legacy_path)
    return rich_path


def drop_empty_rich_export_columns(rich_df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that carry no information in the current export."""
    keep_columns: list[str] = []
    for column in rich_df.columns:
        series = rich_df[column]
        if series.isna().all():
            continue
        non_null = series.dropna()
        if not non_null.empty and non_null.astype(str).str.strip().eq("").all():
            continue
        keep_columns.append(column)
    return rich_df.loc[:, keep_columns].copy()


def remove_legacy_rich_exports(final_csv_path: str) -> None:
    """Remove any stale rich-export CSV variants next to *final_csv_path*."""
    for legacy in (False, True):
        candidate = rich_export_path(final_csv_path, legacy=legacy)
        if not os.path.exists(candidate):
            continue
        try:
            os.remove(candidate)
        except Exception:
            logger.warning("Failed to remove stale rich-export CSV: %s", candidate)


def count_augmented_pose_rows(with_pose_df):
    pose_cols = [col for col in with_pose_df.columns if str(col).startswith("Pose")]
    if not pose_cols:
        return 0, 0
    pose_present = with_pose_df[pose_cols].notna().any(axis=1)
    detection_present = pd.to_numeric(
        with_pose_df.get("DetectionID"), errors="coerce"
    ).notna()
    detection_rows = int((detection_present & pose_present).sum())
    interpolated_rows = int(((~detection_present) & pose_present).sum())
    return detection_rows, interpolated_rows


def count_interpolated_cnn_rows(with_pose_df):
    labels = []
    for col in with_pose_df.columns:
        match = re.match(r"^CNN_(.+)_Class$", str(col))
        if match:
            labels.append(match.group(1))
    labels = sorted(set(labels))
    parts = []
    for label in labels:
        class_col = f"CNN_{label}_Class"
        conf_col = f"CNN_{label}_Conf"
        present = pd.Series(False, index=with_pose_df.index)
        if class_col in with_pose_df.columns:
            present = present | with_pose_df[class_col].fillna("").astype(str).ne("")
        if conf_col in with_pose_df.columns:
            present = present | with_pose_df[conf_col].notna()
        count = int(present.sum())
        if count > 0:
            parts.append(f"{label}={count}")
    return ", ".join(parts) if parts else "none"


def log_rich_export_summary(df: pd.DataFrame) -> None:
    """Log a structured per-source fill-rate summary for the rich export CSV."""
    total = len(df)
    if total == 0:
        logger.info("Rich export summary: 0 rows — nothing to summarize.")
        return

    def fill(col: str) -> int:
        return int(df[col].notna().sum()) if col in df.columns else 0

    def fill_any(cols: list) -> int:
        present = [c for c in cols if c in df.columns]
        if not present:
            return 0
        return int(df[present].notna().any(axis=1).sum())

    def pct(n: int) -> str:
        return f"{100.0 * n / total:.1f}%" if total > 0 else "—"

    lines = [f"Rich export summary — {total:,} rows"]

    # --- pose (detection-keyed vs interpolated) ---
    pose_cols = [c for c in df.columns if str(c).startswith("Pose")]
    kpt_x_cols = [
        c for c in df.columns if str(c).startswith("PoseKpt_") and str(c).endswith("_X")
    ]
    if pose_cols:
        det_present = pd.to_numeric(df.get("DetectionID"), errors="coerce").notna()
        pose_any = df[pose_cols].notna().any(axis=1)
        det_pose = int((det_present & pose_any).sum())
        interp_pose = int((~det_present & pose_any).sum())
        lines.append(
            f"  Pose (detection-keyed)   : {det_pose:>6,} / {total:,}  ({pct(det_pose)})"
        )
        if interp_pose:
            lines.append(
                f"  Pose (interpolated)      : {interp_pose:>6,} / {total:,}  ({pct(interp_pose)})"
            )

    # --- detected heading ---
    heading_cols = [c for c in DETECTED_HEADING_COLUMNS if c in df.columns]
    if heading_cols:
        h_fill = fill_any(heading_cols)
        lines.append(
            f"  Detected heading         : {h_fill:>6,} / {total:,}  ({pct(h_fill)})"
        )

    # --- detected CNN per label ---
    cnn_class_cols = [c for c in df.columns if re.match(r"^CNN_.+_Class$", str(c))]
    cnn_labels = sorted(
        {re.match(r"^CNN_(.+)_Class$", str(c)).group(1) for c in cnn_class_cols}
    )
    for lbl in cnn_labels:
        n = fill_any([f"CNN_{lbl}_Class", f"CNN_{lbl}_Conf"])
        lines.append(f"  CNN [{lbl}]               : {n:>6,} / {total:,}  ({pct(n)})")

    # --- interpolated AprilTag ---
    if "InterpTagID" in df.columns:
        n = fill("InterpTagID")
        if n:
            lines.append(
                f"  AprilTag (interpolated)  : {n:>6,} / {total:,}  ({pct(n)})"
            )

    # --- interpolated head-tail ---
    if "InterpHeadingRad" in df.columns:
        n = fill("InterpHeadingRad")
        if n:
            lines.append(
                f"  Head-tail (interpolated) : {n:>6,} / {total:,}  ({pct(n)})"
            )

    # --- per-keypoint fill rates (grouped 4 per line) ---
    if kpt_x_cols:
        _kpt_re = re.compile(r"^PoseKpt_(.+)_X$")
        kpt_entries = []
        for col in kpt_x_cols:
            m = _kpt_re.match(str(col))
            if m:
                kpt_entries.append(f"{m.group(1)}: {pct(fill(col))}")
        if kpt_entries:
            lines.append("  Per-keypoint fill:")
            for i in range(0, len(kpt_entries), 4):
                lines.append(
                    "    " + "   ".join(f"{s:<22}" for s in kpt_entries[i : i + 4])
                )

    # --- trajectory count ---
    if "TrajectoryID" in df.columns:
        n_tracks = int(df["TrajectoryID"].nunique())
        lines.append(f"  Unique trajectories      : {n_tracks:,}")

    logger.info("\n".join(lines))


def build_rich_export_dataframe(
    final_csv_path, state, *, params, min_valid_conf, ignore_keypoints
):
    """Load final CSV and merge all available analysis sources into a rich export dataframe."""
    if not final_csv_path or not os.path.exists(final_csv_path):
        return None

    sources = check_pose_export_sources(state)
    (
        _has_other_analyses,
        cache_path,
        cache_available,
        interp_pose_path,
        interp_available,
        interp_pose_df_mem,
        interp_mem_available,
    ) = sources

    if (
        not cache_available
        and not interp_available
        and not interp_mem_available
        and not _has_other_analyses
    ):
        logger.warning(
            "Rich export skipped: no analysis sources found (pose_cache=%s, interp=%s, in_memory=%s).",
            cache_path or "<empty>",
            interp_pose_path or "<empty>",
            bool(interp_mem_available),
        )
        return None

    try:
        trajectories_df = pd.read_csv(final_csv_path)
    except Exception:
        logger.exception(
            "Rich export skipped: failed to load trajectories CSV: %s",
            final_csv_path,
        )
        return None

    try:
        with_pose_df = merge_pose_sources_into_df(
            trajectories_df,
            sources,
            state,
            params=params,
            min_valid_conf=min_valid_conf,
            ignore_keypoints=ignore_keypoints,
        )
    except Exception:
        logger.exception(
            "Rich export skipped: failed while merging sources (pose_cache=%s, interp=%s)",
            cache_path or "<empty>",
            interp_pose_path or "<empty>",
        )
        return None

    if with_pose_df is None or with_pose_df.empty:
        logger.warning(
            "Rich export skipped: merged dataframe is empty for %s",
            final_csv_path,
        )
        return None

    _kpt_re = re.compile(r"^PoseKpt_(.+)_X$")
    pose_labels = [
        m.group(1) for col in with_pose_df.columns if (m := _kpt_re.match(str(col)))
    ]

    if pose_labels:
        with_pose_df = apply_pose_quality_postprocessing(
            with_pose_df,
            pose_labels,
            params,
            individual_properties_cache_path=state.individual_properties_cache_path,
        )

    with_pose_df = apply_identity_postprocessing_to_df(with_pose_df, params)

    log_rich_export_summary(with_pose_df)

    return with_pose_df


def export_rich_csv(final_csv_path, state, *, params, min_valid_conf, ignore_keypoints):
    """Write the rich individual-analysis CSV next to the final CSV."""
    with_pose_df = build_rich_export_dataframe(
        final_csv_path,
        state,
        params=params,
        min_valid_conf=min_valid_conf,
        ignore_keypoints=ignore_keypoints,
    )
    if with_pose_df is None or with_pose_df.empty:
        return None

    return write_rich_export_csv(with_pose_df, final_csv_path)


def relink_and_export_rich_csv(
    final_csv_path, state, *, params, min_valid_conf, ignore_keypoints
):
    """Rewrite final CSV IDs after pose-aware relinking and regenerate the rich export CSV."""
    if not final_csv_path or not os.path.exists(final_csv_path):
        return None

    with_pose_df = build_rich_export_dataframe(
        final_csv_path,
        state,
        params=params,
        min_valid_conf=min_valid_conf,
        ignore_keypoints=ignore_keypoints,
    )

    try:
        base_df = pd.read_csv(final_csv_path)
    except Exception:
        logger.exception(
            "Relinking skipped: failed to reload final CSV: %s", final_csv_path
        )
        return export_rich_csv(
            final_csv_path,
            state,
            params=params,
            min_valid_conf=min_valid_conf,
            ignore_keypoints=ignore_keypoints,
        )

    relink_input_df = (
        with_pose_df if with_pose_df is not None and not with_pose_df.empty else base_df
    )
    from hydra_suite.core.post.processing import relink_trajectories_with_pose

    relinked_with_pose = relink_trajectories_with_pose(relink_input_df, params)
    if relinked_with_pose is None or relinked_with_pose.empty:
        relinked_with_pose = relink_input_df

    common_cols = [col for col in base_df.columns if col in relinked_with_pose.columns]
    relinked_base = relinked_with_pose.loc[:, common_cols].copy()
    relinked_base = relinked_base.sort_values(
        ["TrajectoryID", "FrameID"], kind="stable"
    ).reset_index(drop=True)
    relinked_with_pose = relinked_with_pose.sort_values(
        ["TrajectoryID", "FrameID"], kind="stable"
    ).reset_index(drop=True)

    try:
        relinked_base.to_csv(final_csv_path, index=False)
    except Exception:
        logger.exception("Failed to rewrite relinked final CSV: %s", final_csv_path)
        return None

    if with_pose_df is not None and not with_pose_df.empty:
        rich_path = write_rich_export_csv(relinked_with_pose, final_csv_path)
        if not rich_path:
            return None
    else:
        remove_legacy_rich_exports(final_csv_path)

    logger.info(
        "Relinked final CSV rewritten: %s (%d trajectories)",
        final_csv_path,
        (
            int(relinked_base["TrajectoryID"].nunique())
            if "TrajectoryID" in relinked_base.columns
            else 0
        ),
    )
    if with_pose_df is not None and not with_pose_df.empty:
        rich_path = rich_export_path(final_csv_path)
        logger.info("Relinked rich-export CSV saved: %s", rich_path)
        return rich_path
    return final_csv_path
