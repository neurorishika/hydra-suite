"""Identity evidence building: calibration, scoring-mode aggregation, AprilTag priority.

This is the single place where temperature calibration, scoring mode aggregation,
and AprilTag->CNN priority resolution live. Nothing else performs these operations.

Per Correction 17: the OnlineIdentityDecoder consumes the FULL calibrated probability
vector (cnn_factors[i].calibrated_probabilities), NOT the top-1 resolved_label /
resolved_confidence. The top-1 fields are CSV/visualization-only convenience fields.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from hydra_suite.core.inference.config import CNNConfig
from hydra_suite.core.inference.result import AprilTagResult, CNNResult, FrameResult


@dataclass
class CNNFactorEvidence:
    factor_name: str
    class_names: list[str]
    calibrated_probabilities: np.ndarray  # (num_classes,) post-temperature softmax
    winning_class: str
    confidence: float


@dataclass
class DetectionIdentityEvidence:
    det_index: int
    cnn_factors: (
        list[CNNFactorEvidence] | None
    )  # None if CNN phase index is out of range
    apriltag_label: str | None
    apriltag_tag_id: int | None
    resolved_label: str | None  # apriltag overrides cnn when both present
    resolved_confidence: float
    is_authoritative: bool  # True = label came from AprilTag


@dataclass
class FrameIdentityEvidence:
    frame_idx: int
    phase_label: str
    detections: list[DetectionIdentityEvidence]


class IdentityEvidenceBuilder:
    """Converts raw FrameResult into FrameIdentityEvidence for one CNN phase.

    Applies temperature calibration, scoring_mode aggregation, and AprilTag
    priority resolution. One instance per CNN phase; worker.py creates one
    per enabled phase in config.cnn_phases and calls build() once per frame.

    Important (Correction 17): the full calibrated_probabilities vector is
    preserved in each CNNFactorEvidence so the OnlineIdentityDecoder can do
    catalog remapping on the full distribution. Do NOT collapse to argmax here.
    """

    def __init__(self, config: CNNConfig, catalog: Any, phase_index: int) -> None:
        self.config = config
        self._catalog = catalog
        self._phase_index = phase_index

    def build(self, frame_result: FrameResult) -> FrameIdentityEvidence:
        cnn_result: CNNResult | None = (
            frame_result.cnn[self._phase_index]
            if self._phase_index < len(frame_result.cnn)
            else None
        )
        n_dets = frame_result.obb.num_detections if frame_result.obb is not None else 0
        detections = [
            self._build_detection(det_idx, cnn_result, frame_result.apriltag)
            for det_idx in range(n_dets)
        ]
        return FrameIdentityEvidence(
            frame_idx=frame_result.frame_idx,
            phase_label=self.config.label,
            detections=detections,
        )

    def _build_detection(
        self,
        det_idx: int,
        cnn_result: CNNResult | None,
        apriltag_result: AprilTagResult | None,
    ) -> DetectionIdentityEvidence:
        cnn_factors = self._build_cnn_factors(det_idx, cnn_result)
        apriltag_label, apriltag_tag_id = self._lookup_apriltag(
            det_idx, apriltag_result
        )

        if apriltag_label is not None:
            return DetectionIdentityEvidence(
                det_index=det_idx,
                cnn_factors=cnn_factors,
                apriltag_label=apriltag_label,
                apriltag_tag_id=apriltag_tag_id,
                resolved_label=apriltag_label,
                resolved_confidence=1.0,
                is_authoritative=True,
            )

        resolved_label, resolved_confidence = self._resolve_from_cnn(cnn_factors)
        return DetectionIdentityEvidence(
            det_index=det_idx,
            cnn_factors=cnn_factors,
            apriltag_label=None,
            apriltag_tag_id=None,
            resolved_label=resolved_label,
            resolved_confidence=resolved_confidence,
            is_authoritative=False,
        )

    def _build_cnn_factors(
        self, det_idx: int, cnn_result: CNNResult | None
    ) -> list[CNNFactorEvidence] | None:
        if cnn_result is None:
            return None
        # Find the prediction for this detection index using Phase 1 format.
        # CNNResult.predictions is list[CNNDetectionPrediction], one per detection.
        pred = next((p for p in cnn_result.predictions if p.det_index == det_idx), None)
        if pred is None:
            return None
        factors: list[CNNFactorEvidence] = []
        for factor in pred.factors:
            calibrated = self._calibrate(factor.raw_probabilities)
            winning_idx = int(np.argmax(calibrated))
            factors.append(
                CNNFactorEvidence(
                    factor_name=factor.factor_name,
                    class_names=factor.class_names,
                    calibrated_probabilities=calibrated,
                    winning_class=factor.class_names[winning_idx],
                    confidence=float(calibrated[winning_idx]),
                )
            )
        return factors

    def _calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling. t=1 is identity (renormalize only)."""
        t = self.config.calibration_temperature
        if abs(t - 1.0) < 1e-6:
            s = raw_probs.sum()
            return raw_probs / s if s > 0 else raw_probs
        log_probs = np.log(raw_probs + 1e-10)
        scaled = log_probs / t
        scaled -= scaled.max()
        out = np.exp(scaled)
        return (out / out.sum()).astype(np.float32)

    def _lookup_apriltag(
        self,
        det_idx: int,
        apriltag_result: AprilTagResult | None,
    ) -> tuple[str | None, int | None]:
        if apriltag_result is None:
            return None, None
        for i, di in enumerate(apriltag_result.det_indices):
            if di == det_idx:
                tag_id = apriltag_result.tag_ids[i]
                label = self._catalog.get_label(tag_id)
                return label, tag_id
        return None, None

    def _resolve_from_cnn(
        self, factors: list[CNNFactorEvidence] | None
    ) -> tuple[str | None, float]:
        if not factors:
            return None, 0.0
        if self.config.scoring_mode == "atomic" or len(factors) == 1:
            best = max(factors, key=lambda f: f.confidence)
            return best.winning_class, best.confidence
        # per_head_average: pick the class that wins in the most factors;
        # break ties by summed calibrated probability for that class.
        votes: Counter = Counter(f.winning_class for f in factors)
        top_class, _ = votes.most_common(1)[0]
        avg_conf = float(
            np.mean(
                [
                    f.calibrated_probabilities[f.class_names.index(top_class)]
                    for f in factors
                    if top_class in f.class_names
                ]
            )
        )
        return top_class, avg_conf
