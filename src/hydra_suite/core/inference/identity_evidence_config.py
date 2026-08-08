"""Resolved identity-evidence configuration threaded into `InferenceRunner`.

Identity Phase 3, Task 4: carries everything the runner needs to build an
`IdentityEvidenceStage` and write the evidence sidecar during the inference
pass -- the resolved catalog domain, one `EvidenceBuilder` config per CNN
phase (class labels + optional per-phase calibration), and the AprilTag
tag-id -> catalog-label mapping. Resolved once, in the worker, from the same
`resolve_catalog_spec` / calibration-construction logic already used to build
the tracking-time emitter (`worker.py`), and passed to the runner unchanged.

When `identity_evidence` is `None` on `InferenceRunner`, the stage is never
built and neither pass writes a sidecar -- zero behavior change for runs with
no identity configuration.

This module is Core: no Qt, no app-layer imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from hydra_suite.core.individual.identity.calibration import CalibrationModel
    from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec


@dataclass(frozen=True)
class IdentityEvidenceCNNPhaseConfig:
    """One CNN phase's evidence-building inputs.

    Parameters
    ----------
    label:
        The CNN phase label -- MUST match the key used for both
        ``_CacheSet.cnn`` entries (``CNNCacheHandle.label``) and the
        ``cnn_reads`` dict passed to ``IdentityEvidenceStage.evidences_for_frame``,
        so Task 3's "unmatched key -> skip" behavior never silently drops a
        configured phase.
    class_names_per_factor:
        Per-factor class label lists, aligned with the raw per-factor
        probabilities stored in the CNN cache (same contract as
        ``EvidenceBuilder``).
    calibration:
        Optional per-phase ``CalibrationModel`` (temperature scaling).
        ``None`` means no calibration (raw softmax used as-is).
    calibration_signature:
        Provenance string stamped onto every evidence row from this phase.
    """

    label: str
    class_names_per_factor: list[list[str]]
    calibration: "CalibrationModel | None" = None
    calibration_signature: str = ""


@dataclass(frozen=True)
class IdentityEvidenceRunConfig:
    """Resolved identity-evidence inputs for one `InferenceRunner` pass.

    Parameters
    ----------
    catalog_spec:
        The resolved, ordered identity domain (``resolve_catalog_spec``
        output) -- rebuilt into an ``IdentityCatalog`` once, at stage
        construction.
    cnn_phases:
        One entry per configured CNN phase.
    tag_to_label:
        AprilTag ID -> catalog label mapping (from ``TAG_IDENTITY_LABELS``).
    """

    catalog_spec: "IdentityCatalogSpec"
    cnn_phases: tuple[IdentityEvidenceCNNPhaseConfig, ...] = ()
    tag_to_label: dict = field(default_factory=dict)

    def per_factor_temps(self) -> Mapping[str, tuple[float, ...]]:
        """Per-phase calibration temperature(s), for the sidecar cache key.

        Today's ``CalibrationModel`` is single-temperature (applied uniformly
        across all factors of a phase), so each phase contributes a 1-tuple;
        the key schema (``identity_evidence_key.py``) accepts a tuple per
        phase so a future per-factor calibration model can widen this without
        changing the key's shape.
        """
        return {
            phase.label: (
                phase.calibration.temperature if phase.calibration is not None else 1.0,
            )
            for phase in self.cnn_phases
        }
