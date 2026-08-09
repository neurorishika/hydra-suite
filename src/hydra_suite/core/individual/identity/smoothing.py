"""Offline evidence sourcing + forward-backward smoothing for post-hoc identity.

Identity Phase 5 (the honesty fix): the offline/post-hoc identity decoder
must be self-sufficient from the always-written Phase-3
``IdentityEvidenceCache`` sidecar, not the decoder-populated CSV columns
(``CNN_*_Prob``/``DetectedTag*``/``IdentityAssignedLabel``) which are only
populated when ``ENABLE_IDENTITY_IN_TRACKING`` is on. This module is the
seam: given the tracking-output ``DataFrame`` and the evidence cache, pull
each trajectory's ordered per-frame calibrated evidence, on the *global*
catalog basis regardless of what basis each source's evidence was written
against.

This module is Core: it must only depend on Core/numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.substrate import fuse_log_evidence

_TRAJECTORY_COL = "TrajectoryID"
_FRAME_COL = "FrameID"
_DETECTION_COL = "DetectionID"


def _remap_source_log_probs_to_catalog(
    log_probs: np.ndarray,
    source_labels: tuple[str, ...] | None,
    catalog: IdentityCatalog,
) -> np.ndarray:
    """Remap one evidence item's log-probs from its source's own catalog
    basis into ``catalog``'s basis.

    Mirrors the tracking worker's ``_remap_source_log_probs_to_catalog``
    nested closure (``core/tracking/worker.py``): when the source's basis is
    unknown/absent and already the right size, treat it as already on the
    global basis (renormalized in place); otherwise probability-mass-sum
    each source label into its matching global catalog slot (labels the
    global catalog does not contain are dropped), then renormalize and
    return to log-space. A size-mismatched, unlabeled source with no way to
    align falls back to a flat known-uniform prior (uninformative, not
    fabricated certainty).
    """
    arr = np.asarray(log_probs, dtype=np.float64)

    if source_labels is None:
        if len(arr) == catalog.size:
            out = arr.copy()
            out -= np.logaddexp.reduce(out)
            return out
        return catalog.known_uniform_log_prior()

    labels = tuple(str(label) for label in source_labels)
    if len(labels) != len(arr):
        return catalog.known_uniform_log_prior()

    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)
    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)
    for src_idx, label in enumerate(labels):
        if not catalog.contains(label):
            continue
        remapped[catalog.index_of(label)] += float(probs[src_idx])
    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))


def load_trajectory_evidence(
    df: pd.DataFrame,
    cache: IdentityEvidenceCache,
    catalog: IdentityCatalog,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Per-``TrajectoryID`` ordered ``[(FrameID, catalog_log_probs)]`` pulled
    straight from the evidence cache.

    Joins trajectory rows to cached evidence on
    ``(FrameID, DetectionID) -> IdentityEvidence.detection_id``. Rows with a
    missing/NaN ``DetectionID``, or whose ``(FrameID, DetectionID)`` has no
    matching evidence in the cache, are omitted. When a detection has
    evidence from multiple sources in the same frame (e.g. a CNN phase plus
    AprilTag), each source's evidence is first remapped to ``catalog``'s
    basis (via ``cache.catalog_labels_for_source``, mirroring the tracking
    worker's ``_remap_source_log_probs_to_catalog``), then fused into one
    catalog log-vector for that frame via
    ``substrate.fuse_log_evidence`` (starting from a flat log-prior so the
    result is the sources' evidence alone, not a Bayesian update against an
    external prior). Each trajectory's sequence is ordered by ``FrameID``.

    Args:
        df: tracking-output rows with ``TrajectoryID``, ``FrameID``, and
            ``DetectionID`` columns (the ``ln``-style stable per-frame
            detection id the tracking worker writes to CSV).
        cache: an open (``mode="r"``) ``IdentityEvidenceCache``.
        catalog: the target (global) identity catalog to remap/fuse onto.

    Returns:
        ``{TrajectoryID: [(FrameID, log_probs), ...]}``, frame-ordered.
        Trajectories with no cached evidence for any of their rows are
        simply absent from the returned dict (not an empty list).
    """
    for col in (_TRAJECTORY_COL, _FRAME_COL, _DETECTION_COL):
        if col not in df.columns:
            raise KeyError(f"load_trajectory_evidence: df is missing column {col!r}")

    if df.empty:
        return {}

    rows = df[[_TRAJECTORY_COL, _FRAME_COL, _DETECTION_COL]].copy()
    rows = rows[pd.notna(rows[_DETECTION_COL])]
    if rows.empty:
        return {}

    frame_cache: dict[int, dict[int, np.ndarray]] = {}

    def _evidence_for(frame_idx: int, detection_id: int) -> np.ndarray | None:
        if frame_idx not in frame_cache:
            per_detection: dict[int, list[np.ndarray]] = {}
            for ev in cache.load_frame(frame_idx):
                source_labels = cache.catalog_labels_for_source(ev.source_name)
                remapped = _remap_source_log_probs_to_catalog(
                    ev.log_probs, source_labels, catalog
                )
                per_detection.setdefault(ev.detection_id, []).append(remapped)

            fused: dict[int, np.ndarray] = {}
            for det_id, log_prob_list in per_detection.items():
                acc = catalog.uniform_log_prior()
                for lp in log_prob_list:
                    acc = fuse_log_evidence(acc, lp)
                fused[det_id] = acc
            frame_cache[frame_idx] = fused

        return frame_cache[frame_idx].get(detection_id)

    result: dict[int, list[tuple[int, np.ndarray]]] = {}
    for traj_id, grp in rows.groupby(_TRAJECTORY_COL, sort=False):
        grp_sorted = grp.sort_values(_FRAME_COL, kind="stable")
        sequence: list[tuple[int, np.ndarray]] = []
        for frame_id, detection_id in zip(
            grp_sorted[_FRAME_COL], grp_sorted[_DETECTION_COL]
        ):
            log_probs = _evidence_for(int(frame_id), int(detection_id))
            if log_probs is None:
                continue
            sequence.append((int(frame_id), log_probs))
        if sequence:
            result[int(traj_id)] = sequence

    return result
