"""Shared, Qt-free per-factor posterior to catalog-evidence builder.

Identity Phase 3: this class lifts the structured ``(factor_index,
class_name) -> catalog_index`` mapping, per-factor calibration, and
``IdentityEvidence`` construction out of
``core.tracking.identity.evidence_emitter.IdentityEvidenceEmitter`` so both
the live tracking-time emitter and the offline evidence stage share one exact
implementation of the math. ``IdentityEvidenceEmitter`` delegates to this
class; a parity test
(``tests/identity/test_evidence_builder_parity.py``) proves the two agree
exactly for identical inputs.

Unlike the emitter, ``EvidenceBuilder`` does not construct its own internal
catalog: it is handed an already-built :class:`IdentityCatalog` (typically
the composite/cartesian catalog for multi-factor sources) and only maps
per-factor posteriors into that catalog's index space.

This module is Core: no Qt, no app-layer imports.

Scope note
----------
This builder is **posteriors-only**. The emitter's top-1-prediction fallback
(``_build_log_probs_from_prediction``, used only when a source provides no
per-factor posteriors) is intentionally *not* lifted here -- it remains a
degraded-mode path unique to the streaming emitter.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Optional

import numpy as np

from hydra_suite.core.individual.identity.evidence import IdentityEvidence

if TYPE_CHECKING:
    from hydra_suite.core.individual.identity.calibration import CalibrationModel
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog


def build_phase_catalog_labels(
    class_labels_per_factor: list[list[str]],
) -> tuple[str, ...]:
    """Build one CNN phase's own cartesian catalog label tuple.

    This is the exact label-construction algorithm
    ``IdentityEvidenceEmitter.__init__`` used to build its per-source
    (phase-scoped) catalog: for composite (multi-factor) sources, the
    cartesian product of the non-empty factors' labels, joined with ``"_"``,
    deduplicated in first-seen order; for a single (or atomic) factor, the
    flat union of that factor's labels. ``"unknown"`` is always the leading
    entry.

    Extracted as a standalone, shared function (Identity Phase 3 final-fix
    wave) so both the emitter and the inference-time
    ``_build_identity_evidence_stage`` build a CNN phase's ``EvidenceBuilder``
    against the *same* phase-local catalog basis -- required for the tracking
    worker's ``_remap_source_log_probs_to_catalog`` to reproduce the old
    emitter's phase-basis -> global-catalog remap byte-identically.
    """
    import itertools

    non_empty_factors = [fl for fl in class_labels_per_factor if fl]
    is_composite = len(non_empty_factors) > 1

    if is_composite:
        catalog_labels: list[str] = ["unknown"]
        for combo in itertools.product(*non_empty_factors):
            label = "_".join(str(c) for c in combo if c)
            if label and label not in catalog_labels:
                catalog_labels.append(label)
    else:
        catalog_labels = ["unknown"]
        for factor_labels in class_labels_per_factor:
            for lbl in factor_labels:
                if lbl and lbl not in catalog_labels:
                    catalog_labels.append(lbl)

    return tuple(catalog_labels)


class EvidenceBuilder:
    """Map per-detection per-factor raw posteriors to catalog-level evidence.

    Parameters
    ----------
    catalog:
        The :class:`IdentityCatalog` whose index space evidence log-probs are
        built against. Not constructed internally -- the caller owns catalog
        construction (e.g. the emitter's cartesian composite catalog, or a
        catalog shared across sources).
    source_name:
        Human-readable source name stored in each ``IdentityEvidence``.
    class_labels_per_factor:
        List of class label lists, one per raw-posterior factor. Used to
        resolve each factor's per-class probability into the catalog's index
        space. May contain empty ``[]`` entries for "gap" factors that carry
        no classes; the composite index space is always **compacted**: only
        non-empty factors participate, in their relative (gap-skipping)
        order (matching the original emitter's ``_factor_class_to_catalog``
        semantics exactly -- this is bug-for-bug parity, not a new
        contract). Correspondingly, ``per_det_factor_probs`` passed to
        :meth:`build_frame_evidences` must be aligned to the *non-empty*
        factors only (i.e. gap factors contribute no entry), matching how
        real per-factor posteriors are produced upstream (a model never
        emits a probability vector for a factor with no classes).
    calibration:
        Optional ``CalibrationModel``. When provided, raw per-factor softmax
        posteriors are temperature-scaled before being mapped to catalog
        log-priors.
    calibration_signature, runtime_signature:
        Provenance strings written into each evidence item.
    """

    def __init__(
        self,
        catalog: "IdentityCatalog",
        source_name: str,
        class_labels_per_factor: list[list[str]],
        calibration: "CalibrationModel | None" = None,
        calibration_signature: str = "",
        runtime_signature: str = "",
    ) -> None:
        self._catalog = catalog
        self._catalog_labels: tuple[str, ...] = catalog.labels
        self._source_name = source_name
        self._class_labels_per_factor = class_labels_per_factor
        self._calibration = calibration
        self._calibration_signature = calibration_signature
        self._runtime_signature = runtime_signature

        non_empty_factors = [fl for fl in class_labels_per_factor if fl]
        self._is_composite = len(non_empty_factors) > 1

        # Build (factor_index, class_name) -> [catalog_indices] lookup for the
        # composite case by reconstructing the same cartesian-product labels
        # the caller used to build `catalog`, then resolving each combo's
        # label to its *actual* index in `catalog` (never assumed positional
        # alignment, never split a composite label back apart -- composite
        # class names may themselves contain "_").
        #
        # Two semantics must match the original emitter EXACTLY (bug-for-bug
        # parity, since Task 3's evidence stage calls this builder directly
        # and must be a faithful drop-in):
        #   1. Index compaction: `fi` below is the position within `combo`
        #      (i.e. within the non-empty factors only, gap-skipping) --
        #      NOT the raw index into `class_labels_per_factor`.
        #   2. Collision dedup: only the FIRST combo (in
        #      itertools.product traversal order) that produces a given
        #      joined label is registered; later combos joining to an
        #      already-seen label are skipped entirely, even if that label
        #      is present in `catalog`.
        self._factor_class_to_catalog: dict[tuple[int, str], list[int]] = {}
        if self._is_composite:
            seen_labels: set[str] = set()
            for combo in itertools.product(*non_empty_factors):
                label = "_".join(str(c) for c in combo if c)
                if not label or label in seen_labels:
                    continue
                seen_labels.add(label)
                if not catalog.contains(label):
                    continue
                cat_idx = catalog.index_of(label)
                for fi, cls in enumerate(combo):
                    key = (fi, cls)
                    self._factor_class_to_catalog.setdefault(key, []).append(cat_idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_composite(self) -> bool:
        """True when more than one non-empty class-label factor was given."""
        return self._is_composite

    @property
    def catalog_labels(self) -> tuple[str, ...]:
        return self._catalog_labels

    @property
    def source_name(self) -> str:
        return self._source_name

    def build_frame_evidences(
        self,
        frame_idx: int,
        det_ids: list[int],
        per_det_factor_probs: list[list[np.ndarray]],
    ) -> list[IdentityEvidence]:
        """Convert one frame's per-detection per-factor posteriors to evidence.

        Parameters
        ----------
        frame_idx:
            Absolute frame index.
        det_ids:
            Stable detection IDs, one per detection, aligned with
            ``per_det_factor_probs``.
        per_det_factor_probs:
            One entry per detection; each entry is a list of raw per-factor
            softmax probability vectors (one per class-label factor).

        Returns
        -------
        list[IdentityEvidence]
        """
        evidences: list[IdentityEvidence] = []
        for det_id, det_posteriors in zip(det_ids, per_det_factor_probs):
            log_p, observed_mask = self._build_log_probs_from_posteriors(det_posteriors)
            evidences.append(
                IdentityEvidence.from_cnn(
                    frame_idx=frame_idx,
                    detection_id=int(det_id),
                    source_name=self._source_name,
                    log_probs=log_p,
                    calibration_signature=self._calibration_signature,
                    runtime_signature=self._runtime_signature,
                    observed_mask=observed_mask,
                )
            )
        return evidences

    # ------------------------------------------------------------------
    # Internals (lifted verbatim from IdentityEvidenceEmitter)
    # ------------------------------------------------------------------

    def _factor_log_prob(
        self,
        factor_index: int,
        factor_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map one factor posterior to the catalog label space.

        For composite catalogs each factor's probabilities are distributed to
        all composite entries that contain that factor's class, so the sum
        over factors (in log space) gives the joint probability. For flat
        catalogs the original direct lookup is used.
        """
        C = len(self._catalog_labels)
        label_map = []
        if 0 <= factor_index < len(self._class_labels_per_factor):
            label_map = list(self._class_labels_per_factor[factor_index] or [])

        floor = 1e-6
        probs = np.full(C, floor, dtype=np.float64)
        observed = np.zeros(C, dtype=bool)
        observed[0] = True

        factor_arr = np.asarray(factor_probs, dtype=np.float64)

        if self._is_composite:
            for class_idx, cls in enumerate(label_map):
                if class_idx >= len(factor_arr):
                    break
                if not cls:
                    continue
                prob = max(float(factor_arr[class_idx]), floor)
                for cat_idx in self._factor_class_to_catalog.get(
                    (factor_index, cls), []
                ):
                    probs[cat_idx] = prob
                    observed[cat_idx] = True
        else:
            for class_idx, label in enumerate(label_map):
                if class_idx >= len(factor_arr):
                    break
                if not label:
                    continue
                try:
                    catalog_idx = self._catalog.index_of(str(label))
                except KeyError:
                    continue
                probs[catalog_idx] = max(float(factor_arr[class_idx]), floor)
                observed[catalog_idx] = True

        probs /= probs.sum()
        return np.log(np.clip(probs, 1e-300, None)), observed

    def _calibrate_posterior(self, factor_probs: np.ndarray) -> np.ndarray:
        """Temperature-scale a raw softmax posterior, returning a probability
        vector. No-op when no calibration model is configured.

        Mirrors legacy ``predict_batch_posteriors`` (cnn.py): apply
        ``calibrate_probs`` (log-softmax temperature scaling), then
        exponentiate and renormalise back to probabilities so the downstream
        catalog mapping (which expects probabilities) is unchanged.
        """
        arr = np.asarray(factor_probs, dtype=np.float64)
        if self._calibration is None or arr.size == 0:
            return arr
        log_p = self._calibration.calibrate_probs(arr[None, :])[0]
        cal = np.exp(log_p - log_p.max())
        total = cal.sum()
        return cal / total if total > 0 else cal

    def _build_log_probs_from_posteriors(
        self,
        det_posteriors: Optional[list[np.ndarray]],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        C = len(self._catalog_labels)
        if not det_posteriors:
            return np.full(C, -np.log(C), dtype=np.float64), None

        combined = np.zeros(C, dtype=np.float64)
        observed_mask = np.zeros(C, dtype=bool)
        for factor_index, factor_probs in enumerate(det_posteriors):
            factor_log, factor_observed = self._factor_log_prob(
                factor_index,
                self._calibrate_posterior(factor_probs),
            )
            combined += factor_log
            observed_mask |= factor_observed

        combined -= np.logaddexp.reduce(combined)
        return combined, observed_mask
