"""Inference-time `IdentityEvidence` producer over raw cache reads.

Identity Phase 3, Task 3: turns one frame's raw CNN/AprilTag cache reads
(``list[CNNDetectionPrediction]`` per configured CNN phase, plus an optional
``AprilTagResult``) into a flat ``list[IdentityEvidence]``, using the shared
``EvidenceBuilder`` for CNN phases and ``IdentityCatalog.apriltag_log_prior``
for AprilTag observations.

This module is Core: no Qt, no app-layer imports. Purely additive -- nothing
in the runner calls this stage yet (Task 4 wires it in).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra_suite.core.individual.identity.evidence import IdentityEvidence

if TYPE_CHECKING:
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog
    from hydra_suite.core.individual.identity.evidence_builder import EvidenceBuilder
    from hydra_suite.core.inference.result import AprilTagResult, CNNDetectionPrediction


class IdentityEvidenceStage:
    """Convert one frame's raw CNN + AprilTag cache reads into evidence.

    Missing-observation contract
    -----------------------------
    A detection with no CNN prediction for a given phase, and no AprilTag
    read, produces **no** evidence row from that source -- it is simply
    absent from the returned list, not represented as an
    ``IdentityEvidence.missing()`` placeholder. This matches
    ``IdentityEvidenceEmitter.build_frame_evidences``, which only ever
    iterates ``predictions`` (i.e. detections a source actually reported on)
    and never synthesizes a placeholder for detections a source stayed
    silent on. Downstream fusion is expected to treat "no evidence from
    source X this frame" as "source X abstained", not as an explicit
    uninformative observation.

    Parameters
    ----------
    catalog:
        The shared :class:`IdentityCatalog` evidence is built against.
    cnn_builders:
        Per-CNN-phase :class:`EvidenceBuilder`, keyed by the same phase name
        used as the key in ``cnn_reads`` passed to
        :meth:`evidences_for_frame`.
    tag_to_label:
        AprilTag ID -> catalog label mapping, owned by the caller (built
        once from the identity config's tag assignments).
    tag_source_name:
        Stored for provenance/API symmetry with the CNN builders; note
        ``IdentityEvidence.from_apriltag`` itself always stamps
        ``source_name="apriltag"`` on the evidence it constructs.
    """

    def __init__(
        self,
        catalog: "IdentityCatalog",
        cnn_builders: dict[str, "EvidenceBuilder"],
        tag_to_label: dict[int, str],
        tag_source_name: str = "apriltag",
    ) -> None:
        self._catalog = catalog
        self._cnn_builders = cnn_builders
        self._tag_to_label = tag_to_label
        self._tag_source_name = tag_source_name

    def evidences_for_frame(
        self,
        frame_idx: int,
        det_ids: list[int],
        cnn_reads: dict[str, list["CNNDetectionPrediction"]],
        tag_read: "AprilTagResult | None",
    ) -> list[IdentityEvidence]:
        """Build one frame's merged CNN + AprilTag evidence list.

        Parameters
        ----------
        frame_idx:
            Absolute frame index.
        det_ids:
            Stable detection IDs for this frame, indexed by per-frame
            detection-slot position -- i.e. ``det_ids[det_index]`` resolves
            a raw ``det_index`` (as carried by ``CNNDetectionPrediction`` and
            ``AprilTagResult``) to its stable ``DetectionID``.
        cnn_reads:
            One raw cache read per configured CNN phase, keyed by the same
            phase name as ``cnn_builders``.
        tag_read:
            The frame's AprilTag cache read, or ``None`` if AprilTag is not
            configured / produced nothing this frame.

        Returns
        -------
        list[IdentityEvidence]
            CNN evidence for every phase, followed by AprilTag evidence,
            each in the source read's original detection order.
        """
        evidences: list[IdentityEvidence] = []

        for phase_name, predictions in cnn_reads.items():
            builder = self._cnn_builders.get(phase_name)
            if builder is None or not predictions:
                continue
            evidences.extend(
                self._cnn_evidences_for_phase(frame_idx, det_ids, builder, predictions)
            )

        if tag_read is not None:
            evidences.extend(self._apriltag_evidences(frame_idx, det_ids, tag_read))

        return evidences

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _cnn_evidences_for_phase(
        frame_idx: int,
        det_ids: list[int],
        builder: "EvidenceBuilder",
        predictions: list["CNNDetectionPrediction"],
    ) -> list[IdentityEvidence]:
        phase_det_ids: list[int] = []
        per_det_factor_probs: list[list] = []
        for pred in predictions:
            det_index = int(pred.det_index)
            if not (0 <= det_index < len(det_ids)):
                continue
            phase_det_ids.append(det_ids[det_index])
            # Compact to the non-empty factors only -- matching
            # EvidenceBuilder's documented contract that per_det_factor_probs
            # is aligned to the non-empty (class-bearing) factors, in their
            # relative gap-skipping order.
            per_det_factor_probs.append(
                [
                    factor.raw_probabilities
                    for factor in pred.factors
                    if factor.class_names
                ]
            )

        if not phase_det_ids:
            return []
        return builder.build_frame_evidences(
            frame_idx, phase_det_ids, per_det_factor_probs
        )

    def _apriltag_evidences(
        self,
        frame_idx: int,
        det_ids: list[int],
        tag_read: "AprilTagResult",
    ) -> list[IdentityEvidence]:
        evidences: list[IdentityEvidence] = []
        for tag_id, det_index in zip(tag_read.tag_ids, tag_read.det_indices):
            det_index = int(det_index)
            if not (0 <= det_index < len(det_ids)):
                continue
            log_probs = self._catalog.apriltag_log_prior(
                int(tag_id), self._tag_to_label
            )
            evidences.append(
                IdentityEvidence.from_apriltag(
                    frame_idx=frame_idx,
                    detection_id=det_ids[det_index],
                    log_probs=log_probs,
                )
            )
        return evidences
