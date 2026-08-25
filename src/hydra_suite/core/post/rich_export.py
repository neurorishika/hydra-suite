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

from hydra_suite.core.individual.postprocess_df import (
    apply_identity_postprocessing_to_df,
)
from hydra_suite.core.individual.properties.export import DETECTED_HEADING_COLUMNS
from hydra_suite.core.post.pose_merge import (
    apply_pose_quality_postprocessing,
    check_pose_export_sources,
    merge_pose_sources_into_df,
)
from hydra_suite.utils import profiling_names as N
from hydra_suite.utils.profiling import span

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


def count_by_source(df: pd.DataFrame, source_col: str) -> dict:
    """Real-vs-interpolated row counts for one ``*Source`` provenance column.

    Replaces the dead ``count_augmented_pose_rows``/``count_interpolated_cnn_rows``
    (zero callers) with one generic counter used uniformly for all four
    signal types (Pose/CNN/AprilTag/head-tail) now that they share the same
    coalesce-into-original-columns + explicit ``*Source`` provenance
    convention (design spec, "Provenance").
    """
    if source_col not in df.columns:
        return {"real": 0, "interp": 0}
    counts = df[source_col].value_counts()
    return {
        "real": int(counts.get("real", 0)),
        "interp": int(counts.get("interp", 0)),
    }


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

    # --- AprilTag: real vs interpolated ---
    if "TagSource" in df.columns:
        counts = count_by_source(df, "TagSource")
        if counts["interp"]:
            lines.append(
                f"  AprilTag (interpolated)  : {counts['interp']:>6,} / {total:,}  "
                f"({pct(counts['interp'])})"
            )

    # --- head-tail: real vs interpolated ---
    if "HeadingSource" in df.columns:
        counts = count_by_source(df, "HeadingSource")
        if counts["interp"]:
            lines.append(
                f"  Head-tail (interpolated) : {counts['interp']:>6,} / {total:,}  "
                f"({pct(counts['interp'])})"
            )

    # --- CNN: real vs interpolated, per label ---
    for lbl in cnn_labels:
        source_col = f"CNN_{lbl}_Source"
        if source_col in df.columns:
            counts = count_by_source(df, source_col)
            if counts["interp"]:
                lines.append(
                    f"  CNN [{lbl}] (interpolated): {counts['interp']:>6,} / {total:,}  "
                    f"({pct(counts['interp'])})"
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
    final_csv_path,
    state,
    *,
    params,
    min_valid_conf,
    ignore_keypoints,
    identity_evidence_cache_path=None,
):
    """Load final CSV and merge all available analysis sources into a rich export dataframe.

    ``identity_evidence_cache_path`` (Identity Phase 5): the Phase-3 evidence
    sidecar path for this run, forwarded to
    ``apply_identity_postprocessing_to_df`` so the offline fragment solver is
    self-sufficient from the realtime decoder. ``None`` when the caller
    couldn't resolve one (identity post-processing then falls back to
    whatever CSV columns are present).
    """
    with span(N.BUILD_DATAFRAME):
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

        with_pose_df = apply_identity_postprocessing_to_df(
            with_pose_df,
            params,
            identity_evidence_cache_path=identity_evidence_cache_path,
        )

        log_rich_export_summary(with_pose_df)

        return with_pose_df


def export_rich_csv(
    final_csv_path,
    state,
    *,
    params,
    min_valid_conf,
    ignore_keypoints,
    identity_evidence_cache_path=None,
    debug_mode=True,
    fps=None,
    identity_ran=True,
):
    """Write the rich individual-analysis CSV next to the final CSV."""
    with_pose_df = build_rich_export_dataframe(
        final_csv_path,
        state,
        params=params,
        min_valid_conf=min_valid_conf,
        ignore_keypoints=ignore_keypoints,
        identity_evidence_cache_path=identity_evidence_cache_path,
    )
    if with_pose_df is None or with_pose_df.empty:
        return None

    from hydra_suite.core.post.trajectory_writer import write_final_trajectories

    return write_final_trajectories(
        with_pose_df,
        final_csv_path,
        debug_mode=debug_mode,
        fps=fps,
        identity_ran=identity_ran,
        cnn_classifiers=params.get("CNN_CLASSIFIERS"),
    )


def relink_and_export_rich_csv(
    final_csv_path,
    state,
    *,
    params,
    min_valid_conf,
    ignore_keypoints,
    identity_evidence_cache_path=None,
    debug_mode=True,
    fps=None,
    identity_ran=True,
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
        identity_evidence_cache_path=identity_evidence_cache_path,
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
            debug_mode=debug_mode,
            fps=fps,
            identity_ran=identity_ran,
        )

    relink_input_df = (
        with_pose_df if with_pose_df is not None and not with_pose_df.empty else base_df
    )
    # relink_trajectories_with_pose matches fragments via UniqueIdentityKey,
    # which repeats across arenas (arena 0 and arena 7 can both hold an
    # "ant A"), so an ungrouped relink is a genuine cross-arena merge risk.
    # This path is reached from relink_and_export_rich_csv, not from
    # process_trajectories_from_csv, so the grouping there does not cover it.
    # relink_trajectories_with_pose_by_arena groups by arena_id (falling
    # straight through to one unchanged call when there is no arena_id
    # column, or a single arena).
    from hydra_suite.core.post.processing import relink_trajectories_with_pose_by_arena

    relinked_with_pose = relink_trajectories_with_pose_by_arena(relink_input_df, params)
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
        # Intentionally NOT routed through write_base_final_csv: the equivalence
        # gate compares this exact `_tracking_final.csv` byte-for-byte against
        # legacy when relinking runs, and the shared writer's Int64 rounding of
        # X/Y/FrameID can print differently than this bare float write for
        # NaN-containing columns (e.g. "123.0" vs "123"). Keep this a plain
        # to_csv to preserve byte-identity; folding relink formatting into the
        # shared writer is deferred to a separate, gate-validated change.
        relinked_base.to_csv(final_csv_path, index=False)
    except Exception:
        logger.exception("Failed to rewrite relinked final CSV: %s", final_csv_path)
        return None

    if with_pose_df is not None and not with_pose_df.empty:
        from hydra_suite.core.post.trajectory_writer import write_final_trajectories

        rich_path = write_final_trajectories(
            relinked_with_pose,
            final_csv_path,
            debug_mode=debug_mode,
            fps=fps,
            identity_ran=identity_ran,
            cnn_classifiers=params.get("CNN_CLASSIFIERS"),
        )
        if not rich_path:
            return None
    else:
        remove_legacy_rich_exports(final_csv_path)
        if not debug_mode:
            # No pose-augmented frame to relink from, but the clean
            # `<stem>_tracks.csv` was already written (pre-relink IDs) by the
            # earlier export_rich_csv() call. relinked_base above *is* the
            # just-rewritten final CSV content -- refresh tracks.csv from it
            # so User mode doesn't ship stale IDs that disagree with the
            # rewritten final CSV.
            from hydra_suite.core.post.trajectory_writer import write_final_trajectories

            write_final_trajectories(
                relinked_base,
                final_csv_path,
                debug_mode=False,
                fps=fps,
                identity_ran=identity_ran,
                cnn_classifiers=params.get("CNN_CLASSIFIERS"),
            )

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
