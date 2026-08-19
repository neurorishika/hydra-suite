import logging

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C

logger = logging.getLogger(__name__)


def _classifier_label_roster(params) -> tuple:
    """Normalized ``CNN_CLASSIFIERS`` label roster (identity + non-identity).

    Shared by every call site that needs ``identity_class_columns``'s
    ``all_labels`` argument or ``derive_unique_identity_key_series``'s
    ``all_classifier_labels`` argument -- one place owns the
    whitespace-stripping / falsy-label normalization rule so it cannot drift
    between call sites.
    """
    return tuple(
        str(cfg.get("label", "") or "").strip()
        for cfg in (params.get("CNN_CLASSIFIERS") or [])
    )


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


def _stamp_non_identifying_labels(df, params):
    """Give trajectories whose observed composite is non-identifying a name.

    Excluded classes are absent from the identity catalog (by design -- that
    is what frees them from exclusivity), so those tracks resolve to unknown.
    "Unknown" is the wrong report: "the classifier saw no tag" and "the
    classifier could not tell" are different findings. This stamps the
    observed composite as the final label while pinning ``IdentityFinalID``
    to the unknown slot, so the label is descriptive and no consumer keying
    on a resolved identity slot can mistake it for one.

    Only trajectories with no already-resolved identity are touched. Every
    qualifying trajectory is found BEFORE any column is created or the frame
    is copied: declaring marks that match no observed track must leave the
    dataframe byte-identical (no invented ``IdentityFinal*`` family), the
    same "no Final family without evidence" invariant
    ``_mirror_realtime_and_tag_into_final`` documents.
    """
    from hydra_suite.core.individual.identity.heads import identity_axis_columns
    from hydra_suite.core.individual.identity.resolve import excluded_display_labels

    classifiers = params.get("CNN_CLASSIFIERS") or []
    excluded = excluded_display_labels(classifiers)
    if df is None or df.empty or not excluded or "TrajectoryID" not in df.columns:
        return df

    axis_cols = identity_axis_columns(df.columns, classifiers)
    if not axis_cols:
        return df

    class_cols = [c for c, _ in axis_cols]
    conf_cols = [c for _, c in axis_cols if c in df.columns]

    # A row missing any axis value contributes no vote: astype(str) alone
    # would turn a missing value into the literal string "nan", which then
    # silently fails the `in excluded` check instead of being ignored.
    valid_row = df[class_cols].notna().all(axis=1)
    composite = df[class_cols[0]].astype(str)
    for col in class_cols[1:]:
        composite = composite + "_" + df[col].astype(str)
    composite = composite.where(valid_row)

    if C.FINAL_LABEL in df.columns:
        final_label = df[C.FINAL_LABEL]
        unresolved = final_label.isna() | (final_label.astype(str).str.strip() == "")
    else:
        unresolved = pd.Series(True, index=df.index)

    # Phase 1: decide which trajectories qualify, touching nothing.
    stamps: dict = {}
    for traj_id, group in df.groupby("TrajectoryID", sort=False):
        if not unresolved.loc[group.index].all():
            continue  # partially or fully resolved: leave it alone
        observed = composite.loc[group.index].dropna()
        observed = observed[observed != ""]
        if observed.empty:
            continue
        modal = str(observed.mode().iloc[0])
        if modal not in excluded:
            continue
        conf = 0.0
        if conf_cols:
            per_frame = df.loc[group.index, conf_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            # Weakest axis per frame: a composite is only as trustworthy as
            # its least confident tag call.
            mean_conf = per_frame.min(axis=1).mean()
            if pd.notna(mean_conf):
                conf = float(mean_conf)
        stamps[traj_id] = (group.index, modal, conf)

    if not stamps:
        return df

    # Phase 2: at least one trajectory qualifies -- now create the Final
    # family scaffolding and write only the qualifying rows.
    out = df.copy()
    if C.FINAL_LABEL not in out.columns:
        out[C.FINAL_LABEL] = pd.Series(
            [np.nan] * len(out), index=out.index, dtype=object
        )
    else:
        out[C.FINAL_LABEL] = out[C.FINAL_LABEL].astype(object)
    if C.FINAL_SOURCE not in out.columns:
        out[C.FINAL_SOURCE] = pd.Series(
            [C.IdentityFinalSource.NONE] * len(out), index=out.index, dtype=object
        )
    if C.FINAL_ID not in out.columns:
        out[C.FINAL_ID] = np.nan
    if C.FINAL_CONFIDENCE not in out.columns:
        out[C.FINAL_CONFIDENCE] = np.nan

    for _traj_id, (idx, modal, conf) in stamps.items():
        out.loc[idx, C.FINAL_LABEL] = modal
        out.loc[idx, C.FINAL_ID] = 0
        out.loc[idx, C.FINAL_SOURCE] = C.IdentityFinalSource.NON_IDENTIFYING
        out.loc[idx, C.FINAL_CONFIDENCE] = conf

    return out


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
        from hydra_suite.core.individual.identity.heads import (
            HEADS_UNKNOWN,
            identity_class_columns,
            resolve_identity_heads,
        )

        # Only identity heads (`unique_identifier=True`) may feed the identity
        # evidence summary. A behavior/sex/caste classifier is output, not
        # identity: unscoped, its class wins IdentityEvidenceTopLabel whenever
        # it is more confident than the identity classifier.
        _heads = resolve_identity_heads(params)
        if _heads is HEADS_UNKNOWN:
            cnn_class_columns = [
                col
                for col in out.columns
                if str(col).startswith("CNN_") and str(col).endswith("_Class")
            ]
        else:
            cnn_class_columns = identity_class_columns(
                out.columns, _heads, all_labels=_classifier_label_roster(params)
            )
        cnn_conf_columns = {
            col: f"{str(col)[: -len('_Class')]}_Conf" for col in cnn_class_columns
        }

        def _column_or_nan(frame: pd.DataFrame, name: str) -> pd.Series:
            if name in frame.columns:
                return frame[name]
            return pd.Series(np.nan, index=frame.index)

        def _sources_and_conflict_columns(
            frame: pd.DataFrame,
        ) -> tuple:
            """Vectorized (column-wise) equivalent of ``_row_sources`` /
            ``_row_conflict``.

            Loops only over the small, fixed set of evidence columns
            (AprilTag/CNN/Final/Realtime), never over rows -- semantics
            (including the sorted-set join order, the tag/apriltag
            precedence, and the "falls through to the length check even
            when an assigned label is present and matches" conflict
            behavior) are preserved exactly.
            """
            n = len(frame)
            idx = frame.index

            tag_id = _column_or_nan(frame, "DetectedTagID")
            interp_id = _column_or_nan(frame, "InterpTagID")
            has_apriltag = tag_id.notna() | interp_id.notna()

            has_cnn = pd.Series(False, index=idx)
            for col in cnn_class_columns:
                has_cnn = has_cnn | frame[col].notna()

            final_label = _column_or_nan(frame, C.FINAL_LABEL)
            final_smoothed = _column_or_nan(frame, C.FINAL_SMOOTHED_LABEL)
            final_label_present = final_label.notna() | final_smoothed.notna()

            final_source = _column_or_nan(frame, C.FINAL_SOURCE)
            final_source_token = pd.Series("", index=idx, dtype=object)
            final_source_notna = final_source.notna()
            final_source_token = final_source_token.mask(
                final_source_notna, final_source.astype(str).str.strip()
            )

            is_realtime_source = final_label_present & (
                final_source_token == C.IdentityFinalSource.REALTIME
            )
            is_tag_source = final_label_present & (
                final_source_token == C.IdentityFinalSource.TAG
            )
            offline_flag = final_label_present & ~is_realtime_source & ~is_tag_source

            # The tag/apriltag precedence: a tag-sourced Final label folds
            # into "apriltag" (matching the original's
            # `if "apriltag" not in sources: sources.append("apriltag")`).
            has_apriltag_final = has_apriltag | is_tag_source

            realtime_label = _column_or_nan(frame, C.REALTIME_LABEL)
            realtime_notna = realtime_label.notna()
            not_sources_before_realtime = ~(has_apriltag_final | has_cnn | offline_flag)
            has_realtime_final = realtime_notna & not_sources_before_realtime

            have_any = np.zeros(n, dtype=bool)
            sources_arr = np.full(n, "", dtype=object)
            for token, flag in (
                ("apriltag", has_apriltag_final),
                ("cnn", has_cnn),
                ("offline", offline_flag),
                ("realtime", has_realtime_final),
            ):
                flag_arr = flag.to_numpy(dtype=bool, na_value=False)
                addition = np.where(
                    flag_arr, np.where(have_any, "," + token, token), ""
                )
                sources_arr = sources_arr + addition
                have_any = have_any | flag_arr
            sources_final = np.where(have_any, sources_arr, np.nan)
            sources_series = pd.Series(list(sources_final), index=idx)

            # -- conflict flag --
            assigned = realtime_label
            assigned_notna = realtime_notna
            assigned_str = assigned.astype(str)

            observed_cols = []
            if "DetectedTagLabel" in frame.columns:
                observed_cols.append("DetectedTagLabel")
            observed_cols.extend(cnn_class_columns)

            any_mismatch = pd.Series(False, index=idx)
            running_first = pd.Series(np.nan, index=idx, dtype=object)
            multi_distinct = pd.Series(False, index=idx)
            for col in observed_cols:
                value = frame[col]
                present = value.notna()
                value_str = value.astype(str)
                any_mismatch = any_mismatch | (
                    present & assigned_notna & (value_str != assigned_str)
                )
                set_first_mask = present & running_first.isna()
                differs_mask = (
                    present & running_first.notna() & (value_str != running_first)
                )
                multi_distinct = multi_distinct | differs_mask
                running_first = running_first.where(~set_first_mask, value_str)

            conflict_bool = (assigned_notna & any_mismatch) | multi_distinct
            conflict_series = conflict_bool.astype(int)

            return sources_series, conflict_series

        def _top_evidence_columns(frame: pd.DataFrame) -> tuple:
            """Vectorized (column-wise) equivalent of ``_row_top_evidence``.

            Loops only over the CNN classifier head columns (a running
            argmax-by-confidence scan that keeps the first head on exact
            ties, matching the original's strict ``conf > best_conf``),
            then falls back to a detected AprilTag label/confidence.
            """
            idx = frame.index
            best_label = pd.Series(np.nan, index=idx, dtype=object)
            best_conf = pd.Series(np.nan, index=idx, dtype=float)
            for class_col, conf_col in cnn_conf_columns.items():
                label = frame[class_col]
                conf = frame[conf_col]
                valid = label.notna() & conf.notna()
                conf_f = conf.astype(float)
                update = valid & (best_conf.isna() | (conf_f > best_conf))
                best_label = best_label.where(~update, label)
                best_conf = best_conf.where(~update, conf_f)

            tag_label = _column_or_nan(frame, "DetectedTagLabel")
            tag_conf = _column_or_nan(frame, "DetectedTagConf")
            need_fallback = best_label.isna() & tag_label.notna()
            best_label = best_label.where(~need_fallback, tag_label)
            best_conf = best_conf.where(
                ~need_fallback, pd.to_numeric(tag_conf, errors="coerce")
            )

            top_label = best_label.astype(object)
            top_confidence = pd.to_numeric(best_conf, errors="coerce")
            return top_label, top_confidence

        sources_series, conflict_series = _sources_and_conflict_columns(out)
        out[C.EVIDENCE_SOURCES] = sources_series
        out[C.EVIDENCE_CONFLICT_FLAG] = conflict_series

        top_label, top_confidence = _top_evidence_columns(out)
        out[C.EVIDENCE_TOPLABEL] = top_label
        out[C.EVIDENCE_CONFIDENCE] = top_confidence
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

        with_pose_df = _stamp_non_identifying_labels(with_pose_df, params)
        with_pose_df = _mirror_realtime_and_tag_into_final(with_pose_df)
        with_pose_df = fill_identity_nans_with_consensus(with_pose_df)
        with_pose_df = sort_trajectories_by_identity(with_pose_df)
    except Exception:
        logger.exception(
            "Identity-aware post-processing failed; using unmodified rich dataframe."
        )
    with_pose_df = _annotate_identity_summary_columns(with_pose_df)
    try:
        from hydra_suite.core.individual.identity.heads import (
            HEADS_UNKNOWN,
            resolve_identity_heads,
        )
        from hydra_suite.core.individual.identity.resolve import non_identifying_marks
        from hydra_suite.core.post.identity_postprocess import (
            derive_unique_identity_key_series,
        )

        _heads = resolve_identity_heads(params)
        _marks = non_identifying_marks(params.get("CNN_CLASSIFIERS") or [])
        _bare_marks = frozenset(
            m.partition(":")[2] or m
            for marks in _marks.values()
            for m in marks
            if "_" not in m or ":" in m
        )
        with_pose_df[C.UNIQUE_IDENTITY_KEY] = derive_unique_identity_key_series(
            with_pose_df,
            identity_heads=None if _heads is HEADS_UNKNOWN else _heads,
            all_classifier_labels=_classifier_label_roster(params),
            non_identifying_values=_bare_marks,
        )
    except Exception:
        logger.exception("UniqueIdentityKey derivation failed; column left unset.")
    return with_pose_df
