"""Shared, Qt-free per-factor posterior to catalog-evidence builder.

Identity Phase 3: this class lifted the structured ``(factor_index,
class_name) -> catalog_index`` mapping, per-factor calibration, and
``IdentityEvidence`` construction out of the (now-retired, Identity Phase 7)
tracking-time ``IdentityEvidenceEmitter`` so both the live emitter and the
inference-time ``IdentityEvidenceStage`` shared one exact implementation of
the math. The emitter has since been deleted -- ``IdentityEvidenceStage`` is
now the sole production consumer -- and a parity test
(``tests/identity/test_evidence_builder_parity.py``) checks this builder's
output against a committed golden snapshot of the emitter's former output.

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

from hydra_suite.core.individual.identity import substrate
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
    unknown_prior:
        Prior probability mass forced onto the catalog's "unknown" slot
        after per-factor fusion (spec R6). Default 0.0 is a strict no-op
        (today's behavior); see ``substrate.map_cnn_to_catalog``.
    """

    def __init__(
        self,
        catalog: "IdentityCatalog",
        source_name: str,
        class_labels_per_factor: list[list[str]],
        calibration: "CalibrationModel | None" = None,
        calibration_signature: str = "",
        runtime_signature: str = "",
        unknown_prior: float = 0.0,
    ) -> None:
        self._catalog = catalog
        self._catalog_labels: tuple[str, ...] = catalog.labels
        self._source_name = source_name
        self._class_labels_per_factor = class_labels_per_factor
        self._calibration = calibration
        self._calibration_signature = calibration_signature
        self._runtime_signature = runtime_signature
        self._unknown_prior = unknown_prior

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

        Delegates to the shared ``substrate._factor_log_prob`` (Identity
        Phase 4, Task 4) so this builder and any future offline consumer
        share one implementation of the math; kept as a builder method
        (rather than inlined) because ``IdentityEvidenceEmitter``'s top-1
        prediction fallback also calls it directly per factor.
        """
        return substrate._factor_log_prob(
            factor_index,
            factor_probs,
            class_labels_per_factor=self._class_labels_per_factor,
            factor_class_to_catalog=self._factor_class_to_catalog,
            is_composite=self._is_composite,
            catalog_size=len(self._catalog_labels),
            catalog=self._catalog,
        )

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
        """Product-over-factors combination of calibrated per-factor
        posteriors into one catalog log-prob vector.

        Delegates to the shared ``substrate.map_cnn_to_catalog`` (Identity
        Phase 4, Task 4): calibration (this builder's own concern) is
        applied here per factor, then the calibrated posteriors are handed
        to the substrate for the factor->catalog mapping + product-over-
        factors combination, so Layer 2 (this builder) and any future
        offline consumer (Phase 5) share one implementation of that math.
        """
        calibrated = (
            [self._calibrate_posterior(factor_probs) for factor_probs in det_posteriors]
            if det_posteriors
            else det_posteriors
        )
        return substrate.map_cnn_to_catalog(
            calibrated,
            class_labels_per_factor=self._class_labels_per_factor,
            factor_class_to_catalog=self._factor_class_to_catalog,
            is_composite=self._is_composite,
            catalog_size=len(self._catalog_labels),
            catalog=self._catalog,
            unknown_prior=self._unknown_prior,
        )
