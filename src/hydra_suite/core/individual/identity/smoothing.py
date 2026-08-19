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
from hydra_suite.core.individual.identity.phase_remap import remap_phase_log_probs
from hydra_suite.core.individual.identity.substrate import fuse_log_evidence

_TRAJECTORY_COL = "TrajectoryID"
_FRAME_COL = "FrameID"
_DETECTION_COL = "DetectionID"


def _remap_source_log_probs_to_catalog(
    log_probs: np.ndarray,
    source_labels: tuple[str, ...] | None,
    catalog: IdentityCatalog,
    phase_label_map: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """Remap one evidence item's log-probs from its source's own phase
    catalog basis into ``catalog``'s basis.

    The offline twin of the tracking worker's
    ``_remap_source_log_probs_to_catalog`` nested closure
    (``core/tracking/worker.py``), and delegating to the same
    ``phase_remap.remap_phase_log_probs`` so the two cannot diverge: when
    the source's basis is unknown/absent and already the right size, treat
    it as already on the global basis (renormalized in place); otherwise
    distribute each phase label's probability mass across every global
    catalog entry it names (via ``phase_label_map``; a label with no map
    entry still resolves by direct lookup, and one the global catalog does
    not contain at all is dropped), then renormalize and return to
    log-space. A size-mismatched, unlabeled source with no way to align
    falls back to a flat known-uniform prior (uninformative, not fabricated
    certainty).

    An empty/absent ``phase_label_map`` reduces ``remap_phase_log_probs``
    exactly to the historical exact-label-match implementation, so a single
    identity model (whose phase basis *is* the global catalog) is
    bit-identical either way.
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

    return remap_phase_log_probs(arr, labels, catalog, phase_label_map or {})


def load_trajectory_evidence(
    df: pd.DataFrame,
    cache: IdentityEvidenceCache,
    catalog: IdentityCatalog,
    phase_label_maps: dict[str, dict[str, list[int]]] | None = None,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Per-``TrajectoryID`` ordered ``[(FrameID, catalog_log_probs)]`` pulled
    straight from the evidence cache.

    Joins trajectory rows to cached evidence on
    ``(FrameID, DetectionID) -> IdentityEvidence.detection_id``. Rows with a
    missing/NaN ``DetectionID``, or whose ``(FrameID, DetectionID)`` has no
    matching evidence in the cache, are omitted. When a detection has
    evidence from multiple sources in the same frame (e.g. a CNN phase plus
    AprilTag), each source's evidence is first remapped to ``catalog``'s
    basis (via ``cache.catalog_labels_for_source`` plus this source's entry
    in ``phase_label_maps``, mirroring the tracking worker's
    ``_remap_source_log_probs_to_catalog``), then fused into one
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
        phase_label_maps: ``source_name -> phase label map`` as built by
            ``phase_remap.build_phase_label_maps`` (the same maps the
            tracking worker builds for the live path). Required whenever
            the global catalog is a cross-product of more than one identity
            model: without them every phase label misses, all evidence
            floors, and renormalization then fabricates certainty on
            ``unknown``. ``None``/empty is exact-label matching, which is
            correct (and bit-identical to the historical behavior) for a
            single identity model.

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
                    ev.log_probs,
                    source_labels,
                    catalog,
                    (phase_label_maps or {}).get(str(ev.source_name)),
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


def _normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """Renormalize a log-probability vector via log-sum-exp."""
    out = np.asarray(log_probs, dtype=np.float64).copy()
    out -= np.logaddexp.reduce(out)
    return out


def _build_log_transition(catalog_size: int, transition_epsilon: float) -> np.ndarray:
    """Sticky-Markov log-transition of shape ``(catalog_size, catalog_size)``.

    Mirrors ``online.py``'s ``TrackIdentityDecoder._build_log_transition``
    exactly: off-diagonal mass ``eps / (C - 1)`` spread evenly, diagonal
    (stay-in-state) mass ``1 - eps``, then logged with a numerical floor.
    """
    eps = float(transition_epsilon)
    c = int(catalog_size)
    transition = np.full((c, c), eps / max(c - 1, 1), dtype=np.float64)
    np.fill_diagonal(transition, 1.0 - eps)
    return np.log(np.clip(transition, 1e-300, None))


def _predict(log_posterior: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Apply the sticky log-transition to a log-posterior: ``T^T . posterior``.

    Mirrors ``online.py``'s ``TrackIdentityDecoder._predict_belief`` exactly
    (pure function instead of an in-place mutation): for each destination
    state ``j``, ``new[j] = logsumexp_i(log_posterior[i] + log_transition[i, j])``.
    """
    c = log_posterior.shape[0]
    new_log = np.empty_like(log_posterior)
    for j in range(c):
        new_log[j] = np.logaddexp.reduce(log_posterior + log_transition[:, j])
    return new_log


def smooth_trajectory_posteriors(
    frame_log_probs: list[np.ndarray],
    transition_epsilon: float,
) -> list[np.ndarray]:
    """Forward-backward (two-filter) smoothing over one trajectory's evidence.

    Each element of ``frame_log_probs`` is a per-frame, already-normalized
    log-posterior over the catalog (as produced by
    :func:`load_trajectory_evidence`). This combines each frame's evidence
    with both the frames *before* it and the frames *after* it (propagated
    through a sticky-Markov transition), so a confident late burst of
    evidence corrects ambiguous early frames -- and vice versa -- instead of
    the online decoder's forward-only, causal-only belief.

    Two-filter combination (no double-counting)
    ---------------------------------------------
    FORWARD pass (causal, mirrors the online decoder exactly):
        ``alpha_0 = evidence_0``
        ``alpha_t = fuse(predict(alpha_{t-1}), evidence_t)``    for t > 0

    BACKWARD pass (anti-causal, same recursion run in reverse time):
        ``beta_T = evidence_T``
        ``beta_t = fuse(predict(beta_{t+1}), evidence_t)``      for t < T

    Both ``alpha_t`` and ``beta_t`` independently fold in ``evidence_t``
    itself (each is a self-sufficient one-sided posterior), so summing them
    directly in log-space would double-count frame ``t``'s own evidence.
    Instead:

        ``smoothed_t = normalize(alpha_t + beta_t - evidence_t)``

    Subtracting one copy of ``evidence_t`` (already normalized, so this is
    a plain log-space cancellation, not a division-by-near-zero hazard)
    leaves exactly the desired mixture: ``predict(alpha_{t-1})
    + predict(beta_{t+1}) + evidence_t``, i.e. left-context prediction +
    right-context prediction + the frame's own evidence, each counted once.

    This reduces to the fused evidence at the two properties this module is
    tested against:
      * A single-frame trajectory has no left/right context: ``alpha_0 ==
        beta_0 == evidence_0``, so ``smoothed_0 == evidence_0`` exactly.
      * At the trajectory's own boundaries (``t=0``: no left context since
        ``alpha_0 = evidence_0`` exactly; ``t=T``: no right context since
        ``beta_T = evidence_T`` exactly), ``smoothed_0 == beta_0`` and
        ``smoothed_T == alpha_T`` -- i.e. exactly the one-sided filter that
        actually has context to offer.

    Args:
        frame_log_probs: ordered per-frame normalized log-posteriors over
            the catalog (one trajectory's sequence, as returned by
            :func:`load_trajectory_evidence`, e.g. ``[lp for _, lp in
            sequence]``).
        transition_epsilon: sticky-Markov transition leak probability (same
            semantics/knob as the online decoder's
            ``IDENTITY_TRANSITION_EPSILON``); the total probability mass
            per step that a state is allowed to have moved away from its
            previous frame's identity. Smaller values propagate confident
            evidence further along the trajectory before it decays.

    Returns:
        Per-frame normalized log-posteriors, same length and order as
        ``frame_log_probs``. Empty input returns ``[]``.
    """
    n = len(frame_log_probs)
    if n == 0:
        return []

    evidence = [_normalize_log_probs(lp) for lp in frame_log_probs]
    catalog_size = evidence[0].shape[0]
    log_transition = _build_log_transition(catalog_size, transition_epsilon)

    alpha: list[np.ndarray] = [evidence[0]]
    for t in range(1, n):
        predicted = _predict(alpha[t - 1], log_transition)
        alpha.append(fuse_log_evidence(predicted, evidence[t]))

    beta: list[np.ndarray] = [np.empty(0)] * n
    beta[n - 1] = evidence[n - 1]
    for t in range(n - 2, -1, -1):
        predicted = _predict(beta[t + 1], log_transition)
        beta[t] = fuse_log_evidence(predicted, evidence[t])

    smoothed: list[np.ndarray] = []
    for t in range(n):
        combined = alpha[t] + beta[t] - evidence[t]
        smoothed.append(_normalize_log_probs(combined))

    return smoothed


def smoothed_label_and_conf(
    smoothed: list[np.ndarray],
    catalog: IdentityCatalog,
    display_threshold: float,
) -> list[tuple[str, float]]:
    """Per-frame ``(label, confidence)`` from smoothed catalog posteriors.

    For each frame's smoothed log-posterior, picks the argmax over the
    *known* identities (catalog index >= 1, i.e. excluding ``unknown`` at
    index 0) and reports its probability as the confidence. When that
    probability is below ``display_threshold`` the label is reported as
    ``''`` (unknown/undecided) and the confidence is zeroed -- matching the
    online decoder's display-threshold convention exactly
    (``substrate.solve_unique_assignment`` / ``TrackIdentityDecoder.
    _display_threshold``, which both report 0.0 confidence alongside an
    unassigned/blank label rather than a raw sub-threshold probability).

    Args:
        smoothed: per-frame normalized log-posteriors (e.g. the output of
            :func:`smooth_trajectory_posteriors`).
        catalog: the catalog the posteriors are indexed against.
        display_threshold: minimum best-known probability required to
            report a label instead of ``''``.

    Returns:
        A list aligned to ``smoothed`` of ``(label, confidence)`` pairs.
        Empty input returns ``[]``.
    """
    results: list[tuple[str, float]] = []
    for log_probs in smoothed:
        shifted = np.asarray(log_probs, dtype=np.float64)
        shifted = shifted - np.max(shifted)
        probs = np.exp(shifted)
        probs /= np.clip(probs.sum(), 1e-300, None)

        known_probs = probs[1:]
        if known_probs.size == 0:
            results.append(("", 0.0))
            continue

        best_known_idx = int(np.argmax(known_probs)) + 1
        best_conf = float(probs[best_known_idx])
        if best_conf >= display_threshold:
            label = catalog.label_of(best_known_idx)
        else:
            label = ""
            best_conf = 0.0
        results.append((label, best_conf))

    return results
