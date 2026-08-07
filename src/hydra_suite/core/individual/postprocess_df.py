import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_identity_postprocessing_to_df(with_pose_df, params):
    """Run identity-aware split/join processing on the augmented dataframe."""
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
        # Build catalog the same way as the online decoder: CNN composite
        # class labels (cartesian product for multi-factor models) followed
        # by tag labels that match CNN classes.  Using the CNN *phase name*
        # (e.g. "test") instead of class names was the previous bug — phase
        # names are model identifiers, not individual animal identities.
        import itertools as _itertools

        from hydra_suite.core.individual.catalog import IdentityCatalog
        from hydra_suite.core.individual.fragment_solver import run_fragment_solver
        from hydra_suite.core.post.identity_postprocess import (
            fill_identity_nans_with_consensus,
            sort_trajectories_by_identity,
        )

        _raw_labels: list[str] = []
        for _cnn_cfg in params.get("CNN_CLASSIFIERS", []) or []:
            if not bool(_cnn_cfg.get("unique_identifier", False)):
                continue
            _cnpf = list(_cnn_cfg.get("class_names_per_factor") or [])
            _non_empty = [fl for fl in _cnpf if fl]
            if len(_non_empty) > 1:
                for _combo in _itertools.product(*_non_empty):
                    _c = "_".join(str(x) for x in _combo if x)
                    if _c and _c not in _raw_labels:
                        _raw_labels.append(_c)
            elif len(_non_empty) == 1:
                for _l in _non_empty[0]:
                    if _l and str(_l) not in _raw_labels:
                        _raw_labels.append(str(_l))
            else:
                for _l in _cnn_cfg.get("labels", []) or []:
                    if _l and str(_l) not in _raw_labels:
                        _raw_labels.append(str(_l))

        _cnn_label_set = set(_raw_labels)
        for _lbl in params.get("TAG_IDENTITY_LABELS", []) or []:
            _s = str(_lbl).strip()
            if not _s:
                continue
            # When CNN classes are known, only accept tag labels that
            # match them — prevents garbage composites from entering.
            if _cnn_label_set and _s not in _cnn_label_set:
                continue
            if _s not in _raw_labels:
                _raw_labels.append(_s)

        # Tag-only config (no CNN): accept all tag labels.
        if not _raw_labels:
            for _lbl in params.get("TAG_IDENTITY_LABELS", []) or []:
                _s = str(_lbl).strip()
                if _s and _s not in _raw_labels:
                    _raw_labels.append(_s)

        if params.get("ENABLE_IDENTITY_FRAGMENT_SOLVER", False) and _raw_labels:
            try:
                catalog = IdentityCatalog.from_labels(_raw_labels)
                with_pose_df = run_fragment_solver(with_pose_df, catalog, params)
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
