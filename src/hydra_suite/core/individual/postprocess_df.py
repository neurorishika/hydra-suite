import logging

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C

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
    post-hoc identity self-sufficient from realtime (``IdentityRealtimeLabel``/
    ``IdentityRealtimeConfidence`` written by the realtime decoder are no
    longer required for the offline solver to produce real identities).
    ``None``/unopenable degrades gracefully: the solver falls back to
    whatever CSV columns (if any) are present, matching pre-Phase-5
    behavior.
    """
    if with_pose_df is None or with_pose_df.empty:
        return with_pose_df

    def _annotate_identity_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Single owner of the ``IdentityEvidence*`` family (rich-export summary).

        Reads the Final family (``C.FINAL_LABEL``/``C.FINAL_SMOOTHED_LABEL``,
        Task 3) for the ``offline`` source signal and the Realtime family
        (``C.REALTIME_LABEL``) for the ``realtime`` source signal. Never
        writes any ``IdentityRealtime*``/``IdentityFinal*`` column -- this
        function only summarizes evidence already present on the row.
        """
        out = df.copy()
        cnn_class_columns = [
            col
            for col in out.columns
            if str(col).startswith("CNN_") and str(col).endswith("_Class")
        ]
        cnn_conf_columns = {
            col: f"{str(col)[: -len('_Class')]}_Conf" for col in cnn_class_columns
        }

        def _row_sources(row: pd.Series) -> object:
            sources = []
            if pd.notna(row.get("DetectedTagID")) or pd.notna(row.get("InterpTagID")):
                sources.append("apriltag")
            if any(pd.notna(row.get(col)) for col in cnn_class_columns):
                sources.append("cnn")
            final_label_present = pd.notna(row.get(C.FINAL_LABEL)) or pd.notna(
                row.get(C.FINAL_SMOOTHED_LABEL)
            )
            if final_label_present:
                # Prefer the explicit IdentityFinalSource when present -- it
                # is the authoritative provenance signal (Task 5's
                # realtime/tag mirror leaves it "realtime"/"tag" so a
                # merely-mirrored row is not misreported as "offline"). Rows
                # with a Final label but no source recorded (legacy/synthetic
                # data written directly by the offline solver's own tests)
                # fall back to "offline", the pre-mirror default.
                final_source = row.get(C.FINAL_SOURCE)
                final_source_token = (
                    str(final_source).strip() if pd.notna(final_source) else ""
                )
                if final_source_token == C.IdentityFinalSource.REALTIME:
                    pass  # already covered by the C.REALTIME_LABEL check below
                elif final_source_token == C.IdentityFinalSource.TAG:
                    if "apriltag" not in sources:
                        sources.append("apriltag")
                else:
                    sources.append("offline")
            if pd.notna(row.get(C.REALTIME_LABEL)) and not sources:
                sources.append("realtime")
            if not sources:
                return np.nan
            return ",".join(sorted(set(sources)))

        def _row_conflict(row: pd.Series) -> int:
            assigned = row.get(C.REALTIME_LABEL)
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

        def _row_top_evidence(row: pd.Series) -> tuple:
            """Per-row top calibrated evidence: (top_label, confidence).

            Prefers the CNN classifier head with the highest reported
            confidence (``CNN_*_Class``/``CNN_*_Conf`` pairs); falls back to
            a detected AprilTag label/confidence when no CNN evidence is
            present on the row.
            """
            best_label = np.nan
            best_conf = np.nan
            for class_col, conf_col in cnn_conf_columns.items():
                label = row.get(class_col)
                conf = row.get(conf_col)
                if pd.isna(label) or pd.isna(conf):
                    continue
                conf = float(conf)
                if pd.isna(best_conf) or conf > best_conf:
                    best_label = label
                    best_conf = conf
            if pd.isna(best_label):
                tag_label = row.get("DetectedTagLabel")
                if pd.notna(tag_label):
                    tag_conf = row.get("DetectedTagConf")
                    best_label = tag_label
                    best_conf = float(tag_conf) if pd.notna(tag_conf) else np.nan
            return best_label, best_conf

        out[C.EVIDENCE_SOURCES] = out.apply(_row_sources, axis=1)
        out[C.EVIDENCE_CONFLICT_FLAG] = out.apply(_row_conflict, axis=1).astype(int)

        top_evidence = out.apply(_row_top_evidence, axis=1, result_type="expand")
        if top_evidence.empty:
            out[C.EVIDENCE_TOPLABEL] = pd.Series(
                [np.nan] * len(out), index=out.index, dtype=object
            )
            out[C.EVIDENCE_CONFIDENCE] = pd.Series(
                [np.nan] * len(out), index=out.index, dtype=float
            )
        else:
            out[C.EVIDENCE_TOPLABEL] = top_evidence[0].astype(object)
            out[C.EVIDENCE_CONFIDENCE] = pd.to_numeric(top_evidence[1], errors="coerce")
        return out

    def _mirror_realtime_and_tag_into_final(df: pd.DataFrame) -> pd.DataFrame:
        """Non-destructive ``IdentityRealtime*``/tag -> ``IdentityFinal*`` mirror.

        For every row where ``C.FINAL_LABEL`` is still empty (the fragment
        solver either did not run, or catalog_spec had no entries, or the
        row fell outside the solver's scope), fall back to a cheaper already
        -resolved identity: first the realtime decoder's per-frame decision
        (``C.REALTIME_LABEL``), then a detected AprilTag (``DetectedTagLabel``).
        Rows the offline solver already resolved (``C.FINAL_SOURCE`` already
        non-empty) are never touched, and ``IdentityRealtime*``/tag columns
        are read-only here -- this function writes only the Final family.

        A true no-op (no columns created/touched) when there is nothing to
        mirror from and no prior stage wrote ``C.FINAL_LABEL`` -- keeps the
        "no Final family without evidence" invariant for rows/runs with
        neither realtime nor tag nor offline identity data.
        """
        if (
            C.FINAL_LABEL not in df.columns
            and C.REALTIME_LABEL not in df.columns
            and "DetectedTagLabel" not in df.columns
        ):
            return df

        out = df.copy()

        if C.FINAL_LABEL not in out.columns:
            out[C.FINAL_LABEL] = pd.Series(
                [np.nan] * len(out), index=out.index, dtype=object
            )
        elif out[C.FINAL_LABEL].dtype != object:
            out[C.FINAL_LABEL] = out[C.FINAL_LABEL].astype(object)
        if C.FINAL_SOURCE not in out.columns:
            out[C.FINAL_SOURCE] = pd.Series(
                [C.IdentityFinalSource.NONE] * len(out), index=out.index, dtype=object
            )
        elif out[C.FINAL_SOURCE].dtype != object:
            out[C.FINAL_SOURCE] = out[C.FINAL_SOURCE].astype(object)
        if C.FINAL_ID not in out.columns:
            out[C.FINAL_ID] = np.nan
        if C.FINAL_CONFIDENCE not in out.columns:
            out[C.FINAL_CONFIDENCE] = np.nan

        empty_final = out[C.FINAL_LABEL].isna() | (
            out[C.FINAL_LABEL].astype(str).str.strip() == ""
        )
        if not empty_final.any():
            return out

        if C.REALTIME_LABEL in out.columns:
            realtime_label = out[C.REALTIME_LABEL]
            has_realtime = (
                empty_final
                & realtime_label.notna()
                & (realtime_label.astype(str).str.strip() != "")
            )
            if has_realtime.any():
                out.loc[has_realtime, C.FINAL_LABEL] = realtime_label.loc[has_realtime]
                if C.REALTIME_ID in out.columns:
                    out.loc[has_realtime, C.FINAL_ID] = out.loc[
                        has_realtime, C.REALTIME_ID
                    ]
                if C.REALTIME_CONFIDENCE in out.columns:
                    out.loc[has_realtime, C.FINAL_CONFIDENCE] = out.loc[
                        has_realtime, C.REALTIME_CONFIDENCE
                    ]
                out.loc[has_realtime, C.FINAL_SOURCE] = C.IdentityFinalSource.REALTIME
                empty_final = empty_final & ~has_realtime

        if "DetectedTagLabel" in out.columns and empty_final.any():
            tag_label = out["DetectedTagLabel"]
            has_tag = (
                empty_final
                & tag_label.notna()
                & (tag_label.astype(str).str.strip() != "")
            )
            if has_tag.any():
                out.loc[has_tag, C.FINAL_LABEL] = tag_label.loc[has_tag]
                if "DetectedTagConf" in out.columns:
                    out.loc[has_tag, C.FINAL_CONFIDENCE] = pd.to_numeric(
                        out.loc[has_tag, "DetectedTagConf"], errors="coerce"
                    )
                out.loc[has_tag, C.FINAL_SOURCE] = C.IdentityFinalSource.TAG

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
            params.get("IDENTITY_POSTHOC_ENABLED", True)
            and params.get("ENABLE_IDENTITY_FRAGMENT_SOLVER", False)
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

        with_pose_df = _mirror_realtime_and_tag_into_final(with_pose_df)
        with_pose_df = fill_identity_nans_with_consensus(with_pose_df)
        with_pose_df = sort_trajectories_by_identity(with_pose_df)
    except Exception:
        logger.exception(
            "Identity-aware post-processing failed; using unmodified rich dataframe."
        )
    with_pose_df = _annotate_identity_summary_columns(with_pose_df)
    try:
        from hydra_suite.core.post.identity_postprocess import (
            derive_unique_identity_key_series,
        )

        with_pose_df[C.UNIQUE_IDENTITY_KEY] = derive_unique_identity_key_series(
            with_pose_df
        )
    except Exception:
        logger.exception("UniqueIdentityKey derivation failed; column left unset.")
    return with_pose_df
