import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _open_identity_evidence_cache(identity_evidence_cache_path):
    """Open the Phase-3 evidence sidecar for reading, or return ``None``.

    Graceful by construction: a missing path, a nonexistent file, or a
    corrupt/unreadable sidecar all resolve to ``None`` (identity
    post-processing then falls back to whatever CSV columns are present,
    same as before Phase 5) -- this function must never raise.
    """
    if not identity_evidence_cache_path:
        return None
    try:
        from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache

        return IdentityEvidenceCache(identity_evidence_cache_path, mode="r")
    except Exception:
        logger.warning(
            "Identity evidence cache at %s could not be opened; offline "
            "identity post-processing will proceed without it.",
            identity_evidence_cache_path,
            exc_info=True,
        )
        return None


def apply_identity_postprocessing_to_df(
    with_pose_df, params, identity_evidence_cache_path=None
):
    """Run identity-aware split/join processing on the augmented dataframe.

    ``identity_evidence_cache_path`` (Identity Phase 5, the honesty fix):
    path to the always-written Phase-3 ``IdentityEvidenceCache`` sidecar for
    this run (see ``core.individual.identity.cache.
    find_identity_evidence_cache_path`` / ``core.tracking.session.
    TrackingSessionCore``, which resolves and threads it through from the
    tracking worker's cache directory). When given and openable, the
    fragment solver sources identity evidence from it directly -- making
    post-hoc identity self-sufficient from realtime (``IdentityAssignedLabel``/
    ``IdentityAssignedConfidence`` written by the realtime decoder are no
    longer required for the offline solver to produce real identities).
    ``None``/unopenable degrades gracefully: the solver falls back to
    whatever CSV columns (if any) are present, matching pre-Phase-5
    behavior.
    """
    if with_pose_df is None or with_pose_df.empty:
        return with_pose_df

    def _annotate_identity_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cnn_class_columns = [
            col
            for col in out.columns
            if str(col).startswith("CNN_") and str(col).endswith("_Class")
        ]

        def _row_sources(row: pd.Series) -> object:
            sources = []
            if pd.notna(row.get("DetectedTagID")) or pd.notna(row.get("InterpTagID")):
                sources.append("apriltag")
            if any(pd.notna(row.get(col)) for col in cnn_class_columns):
                sources.append("cnn")
            if pd.notna(row.get("IdentityOfflineLabel")) or pd.notna(
                row.get("IdentitySmoothedLabel")
            ):
                sources.append("offline")
            if pd.notna(row.get("IdentityAssignedLabel")) and not sources:
                sources.append("online")
            if not sources:
                return np.nan
            return ",".join(sorted(set(sources)))

        def _row_conflict(row: pd.Series) -> int:
            assigned = row.get("IdentityAssignedLabel")
            observed = set()
            detected_tag_label = row.get("DetectedTagLabel")
            if pd.notna(detected_tag_label):
                observed.add(str(detected_tag_label))
            for col in cnn_class_columns:
                value = row.get(col)
                if pd.notna(value):
                    observed.add(str(value))
            if pd.notna(assigned):
                assigned_label = str(assigned)
                if any(label != assigned_label for label in observed):
                    return 1
            return 1 if len(observed) > 1 else 0

        out["IdentityEvidenceSources"] = out.apply(_row_sources, axis=1)
        out["IdentityConflictFlag"] = out.apply(_row_conflict, axis=1).astype(int)
        return out

    try:
        # Identity Phase 5: catalog resolution is the single shared resolver
        # (also used by the tracking worker to build the SAME catalog for the
        # realtime decoder + the Phase-3 evidence-cache writer) rather than a
        # second, hand-maintained inline duplicate of that assembly logic.
        from hydra_suite.core.individual.identity.catalog import IdentityCatalog
        from hydra_suite.core.individual.identity.offline import run_fragment_solver
        from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
        from hydra_suite.core.post.identity_postprocess import (
            fill_identity_nans_with_consensus,
            sort_trajectories_by_identity,
        )

        catalog_spec = resolve_catalog_spec(
            params.get("CNN_CLASSIFIERS", []) or [],
            params.get("TAG_IDENTITY_LABELS", []) or [],
        )

        if (
            params.get("ENABLE_IDENTITY_FRAGMENT_SOLVER", False)
            and catalog_spec.entries
        ):
            try:
                catalog = IdentityCatalog.from_spec(catalog_spec)
                cache = _open_identity_evidence_cache(identity_evidence_cache_path)
                with_pose_df = run_fragment_solver(
                    with_pose_df, catalog, params, cache=cache
                )
                with_pose_df = _annotate_identity_summary_columns(with_pose_df)
                logger.info("Fragment solver complete.")
            except Exception:
                logger.exception("Fragment solver failed; results unchanged.")

        with_pose_df = fill_identity_nans_with_consensus(with_pose_df)
        with_pose_df = sort_trajectories_by_identity(with_pose_df)
    except Exception:
        logger.exception(
            "Identity-aware post-processing failed; using unmodified rich dataframe."
        )
    return _annotate_identity_summary_columns(with_pose_df)
