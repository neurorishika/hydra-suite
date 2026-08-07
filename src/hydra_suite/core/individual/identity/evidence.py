"""Identity evidence contract.

Identity Phase 0: every source contributing to identity inference (AprilTag,
CNN) emits ``IdentityEvidence`` objects with a calibrated log-posterior over
the full catalog including the unknown slot.

Design rules
------------
- Missing evidence is *explicit*: use ``IdentityEvidence.missing()`` rather
  than dropping the detection.
- The ``log_probs`` vector must cover every catalog index including index 0
  (unknown).  Downstream components rely on this for log-space fusion.
- Evidence objects are frozen (``frozen=True``) to prevent accidental mutation
  during multi-source fusion passes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class EvidenceSource(str, enum.Enum):
    """Registered evidence source type tags."""

    APRILTAG = "apriltag"
    CNN = "cnn"
    MISSING = "missing"  # Explicit placeholder for absent evidence

    def __str__(self) -> str:  # noqa: D105
        return self.value


@dataclass(frozen=True)
class IdentityEvidence:
    """Per-detection identity evidence at one frame.

    ``log_probs`` is a calibrated log-posterior (log-scale, not log-softmax)
    over the full catalog, **including the unknown slot at index 0**.  It does
    not need to be normalised on construction; downstream fused posteriors will
    be renormalised after all evidence is combined.

    The observation is *missing* when ``source == EvidenceSource.MISSING``;
    in that case ``log_probs`` is a uniform distribution.

    Parameters
    ----------
    frame_idx:
        Absolute frame index.
    detection_id:
        Stable detection slot ID within this frame (matches
        ``StreamingAnalysisPayload.detection_ids``).
    source:
        Enum tag identifying the evidence type.
    source_name:
        Human-readable name, e.g. ``'apriltag'``, ``'cnn_mouse_color'``.
    log_probs:
        Shape ``(catalog_size,)`` float64.  Calibrated evidence log-posterior.
    catalog_size:
        Size of the catalog this evidence was produced against.  Used for
        validation on load.
    calibration_signature:
        Hash / identifier of the calibration model used to produce
        ``log_probs``.  Empty string means no explicit calibration was applied.
    runtime_signature:
        Runtime string that produced the model outputs, e.g. ``'cuda'``.
    observed_mask:
        Optional shape ``(catalog_size,)`` bool.  True where the model had
        valid observations.  ``None`` means all entries are observed.
    metadata:
        Arbitrary provenance metadata for audit purposes.  Not persisted to
        the evidence cache.
    """

    frame_idx: int
    detection_id: int
    source: EvidenceSource
    source_name: str
    log_probs: np.ndarray
    catalog_size: int
    calibration_signature: str = ""
    runtime_signature: str = ""
    observed_mask: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.log_probs) != self.catalog_size:
            raise ValueError(
                f"log_probs length {len(self.log_probs)} != "
                f"catalog_size {self.catalog_size}"
            )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def missing(
        frame_idx: int,
        detection_id: int,
        source_name: str,
        catalog_size: int,
    ) -> "IdentityEvidence":
        """Construct a *missing* evidence placeholder with a uniform log-prior.

        Missing evidence is distinct from absent evidence: it signals that the
        source ran but produced no useful output for this detection.  Downstream
        fused posteriors will treat this as a weak, uninformative update.
        """
        log_p = np.full(catalog_size, -np.log(catalog_size), dtype=np.float64)
        return IdentityEvidence(
            frame_idx=frame_idx,
            detection_id=detection_id,
            source=EvidenceSource.MISSING,
            source_name=source_name,
            log_probs=log_p,
            catalog_size=catalog_size,
        )

    @staticmethod
    def from_apriltag(
        frame_idx: int,
        detection_id: int,
        log_probs: np.ndarray,
        runtime_signature: str = "",
    ) -> "IdentityEvidence":
        """Construct AprilTag evidence from a pre-computed log-prior vector."""
        return IdentityEvidence(
            frame_idx=frame_idx,
            detection_id=detection_id,
            source=EvidenceSource.APRILTAG,
            source_name="apriltag",
            log_probs=np.asarray(log_probs, dtype=np.float64),
            catalog_size=len(log_probs),
            runtime_signature=runtime_signature,
        )

    @staticmethod
    def from_cnn(
        frame_idx: int,
        detection_id: int,
        source_name: str,
        log_probs: np.ndarray,
        calibration_signature: str = "",
        runtime_signature: str = "",
        observed_mask: Optional[np.ndarray] = None,
    ) -> "IdentityEvidence":
        """Construct CNN evidence from a calibrated log-posterior vector."""
        return IdentityEvidence(
            frame_idx=frame_idx,
            detection_id=detection_id,
            source=EvidenceSource.CNN,
            source_name=source_name,
            log_probs=np.asarray(log_probs, dtype=np.float64),
            catalog_size=len(log_probs),
            calibration_signature=calibration_signature,
            runtime_signature=runtime_signature,
            observed_mask=observed_mask,
        )
