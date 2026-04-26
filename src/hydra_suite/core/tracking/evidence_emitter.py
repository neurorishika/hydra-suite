"""Identity evidence emitter for the streaming and precompute paths.

Streaming Phases 3 & 4 / Identity Phase 0: converts CNN ``ClassPrediction``
outputs to ``IdentityEvidence`` objects and accumulates them into an
``IdentityEvidenceCache`` sidecar file.

The emitter is wired as a frame-result callback on ``CNNPrecomputePhase``
(via ``set_frame_result_callback``) and on the live ``LiveCNNIdentityStore``
equivalent in streaming mode.  It emits the same artifact whether the run
used batch precompute or streaming live analysis, satisfying the parity
requirement from the streaming plan.

The emitter does not require a full ``IdentityCatalog`` to be available at
construction time.  It uses the CNN model's own class labels as the initial
label space and persists them in the sidecar so the offline decoder can map
them to catalog indices later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hydra_suite.core.identity.cache import IdentityEvidenceCache
from hydra_suite.core.identity.classification.cnn import ClassPrediction
from hydra_suite.core.identity.evidence import IdentityEvidence

log = logging.getLogger(__name__)


class IdentityEvidenceEmitter:
    """Convert CNN frame predictions to evidence and write to sidecar cache.

    Intended to be called as a frame-result callback::

        emitter = IdentityEvidenceEmitter(
            cache_path=...,
            source_name="cnn_mouse_color",
            class_labels_per_factor=[["white", "black", "brown"]],
            runtime_signature="cpu",
        )
        cnn_phase.set_frame_result_callback(emitter)
        ...
        emitter.flush()

    Parameters
    ----------
    cache_path:
        Path to the output ``.npz`` sidecar file.
    source_name:
        Human-readable source name stored in each ``IdentityEvidence``.
    class_labels_per_factor:
        List of class label lists, one per CNN factor.  These become the
        ``catalog_labels`` stored in the sidecar.
    runtime_signature:
        Runtime string written into each evidence item for provenance.
    calibration_signature:
        Optional calibration model signature.
    """

    def __init__(
        self,
        cache_path: str | Path,
        source_name: str,
        class_labels_per_factor: list[list[str]],
        runtime_signature: str = "",
        calibration_signature: str = "",
    ) -> None:
        self._source_name = source_name
        self._runtime_signature = runtime_signature
        self._calibration_signature = calibration_signature
        self._class_labels_per_factor = class_labels_per_factor

        # Build a flat catalog label list from all factor labels
        catalog_labels: list[str] = ["unknown"]
        for factor_labels in class_labels_per_factor:
            for lbl in factor_labels:
                if lbl and lbl not in catalog_labels:
                    catalog_labels.append(lbl)

        self._catalog_labels = tuple(catalog_labels)
        self._cache = IdentityEvidenceCache(
            cache_path,
            catalog_labels=self._catalog_labels,
            mode="w",
        )
        self._flushed = False

    # ------------------------------------------------------------------
    # Callable frame-result callback interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        frame_idx: int,
        predictions: list[ClassPrediction],
        posteriors: Optional[list[Optional[list[np.ndarray]]]] = None,
        detection_ids: Optional[list[int]] = None,
    ) -> None:
        """Process one frame's CNN predictions and store evidence.

        This is the callback signature expected by
        ``CNNPrecomputePhase.set_frame_result_callback()``.
        """
        self.emit_frame(
            frame_idx,
            predictions,
            posteriors=posteriors,
            detection_ids=detection_ids,
        )

    @staticmethod
    def _resolve_detection_id(
        pred: ClassPrediction,
        detection_ids: Optional[list[int]] = None,
    ) -> int:
        """Resolve a stable DetectionID for persisted evidence rows.

        CNN caches and live stores intentionally keep ``ClassPrediction.det_index``
        aligned to the per-frame detection-slot index for association-time use.
        The evidence sidecar, however, must be keyed by the stable
        ``DetectionID`` so online/offline identity decoders can join against the
        trajectory dataframe.
        """
        det_index = int(pred.det_index)
        if detection_ids is None:
            return det_index
        if 0 <= det_index < len(detection_ids):
            try:
                return int(detection_ids[det_index])
            except Exception:
                log.debug(
                    "Failed to map detection-slot index %d to stable DetectionID",
                    det_index,
                    exc_info=True,
                )
        return det_index

    def build_frame_evidences(
        self,
        frame_idx: int,
        predictions: list[ClassPrediction],
        posteriors: Optional[list[Optional[list[np.ndarray]]]] = None,
        detection_ids: Optional[list[int]] = None,
    ) -> list[IdentityEvidence]:
        """Convert one frame of predictions to in-memory evidence rows."""
        if not predictions:
            return []

        evidences: list[IdentityEvidence] = []
        if posteriors is None:
            posteriors = [None] * len(predictions)

        for pred, det_posteriors in zip(predictions, posteriors):
            if det_posteriors is not None:
                log_p, observed_mask = self._build_log_probs_from_posteriors(
                    det_posteriors
                )
            else:
                log_p, observed_mask = self._build_log_probs_from_prediction(pred)

            resolved_detection_id = self._resolve_detection_id(
                pred,
                detection_ids=detection_ids,
            )

            evidences.append(
                IdentityEvidence.from_cnn(
                    frame_idx=frame_idx,
                    detection_id=resolved_detection_id,
                    source_name=self._source_name,
                    log_probs=log_p,
                    calibration_signature=self._calibration_signature,
                    runtime_signature=self._runtime_signature,
                    observed_mask=observed_mask,
                )
            )
        return evidences

    def emit_evidences(self, frame_idx: int, evidences: list[IdentityEvidence]) -> None:
        """Persist pre-built evidence rows for one frame."""
        if evidences:
            self._cache.save_frame(frame_idx, evidences)

    def _factor_log_prob(
        self,
        factor_index: int,
        factor_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map one factor posterior to the emitter's flat label space."""
        C = len(self._catalog_labels)
        label_map = []
        if 0 <= factor_index < len(self._class_labels_per_factor):
            label_map = list(self._class_labels_per_factor[factor_index] or [])

        floor = 1e-6
        probs = np.full(C, floor, dtype=np.float64)
        observed = np.zeros(C, dtype=bool)
        observed[0] = True

        factor_arr = np.asarray(factor_probs, dtype=np.float64)
        for class_idx, label in enumerate(label_map):
            if class_idx >= len(factor_arr):
                break
            if not label:
                continue
            try:
                catalog_idx = self._catalog_labels.index(str(label))
            except ValueError:
                continue
            probs[catalog_idx] = max(float(factor_arr[class_idx]), floor)
            observed[catalog_idx] = True

        probs /= probs.sum()
        return np.log(np.clip(probs, 1e-300, None)), observed

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
                np.asarray(factor_probs, dtype=np.float64),
            )
            combined += factor_log
            observed_mask |= factor_observed

        combined -= np.logaddexp.reduce(combined)
        return combined, observed_mask

    def _build_log_probs_from_prediction(
        self,
        pred: ClassPrediction,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        C = len(self._catalog_labels)
        combined = np.zeros(C, dtype=np.float64)
        observed_mask = np.zeros(C, dtype=bool)
        used_any_factor = False

        for factor_index, (class_name, conf) in enumerate(
            zip(pred.class_names, pred.confidences)
        ):
            if class_name is None:
                continue
            used_any_factor = True
            try:
                catalog_idx = self._catalog_labels.index(class_name)
            except ValueError:
                continue
            floor = max(1e-6, (1.0 - float(conf)) / max(C - 1, 1))
            probs = np.full(C, floor, dtype=np.float64)
            probs[catalog_idx] = max(float(conf), floor)
            probs /= probs.sum()
            combined += np.log(np.clip(probs, 1e-300, None))
            observed_mask[catalog_idx] = True
            observed_mask[0] = True

        if not used_any_factor:
            return np.full(C, -np.log(C), dtype=np.float64), None

        combined -= np.logaddexp.reduce(combined)
        return combined, observed_mask

    def emit_frame(
        self,
        frame_idx: int,
        predictions: list[ClassPrediction],
        posteriors: Optional[list[Optional[list[np.ndarray]]]] = None,
        detection_ids: Optional[list[int]] = None,
    ) -> None:
        """Convert predictions to evidence and accumulate in the sidecar."""
        evidences = self.build_frame_evidences(
            frame_idx,
            predictions,
            posteriors=posteriors,
            detection_ids=detection_ids,
        )
        self.emit_evidences(frame_idx, evidences)

    def flush(self) -> None:
        """Write accumulated evidence sidecar to disk."""
        if not self._flushed:
            self._cache.flush()
            self._flushed = True

    @property
    def catalog_labels(self) -> tuple[str, ...]:
        """Flat catalog label tuple stored in the sidecar."""
        return self._catalog_labels


def build_evidence_cache_path(
    base_cache_path: str,
    source_name: str,
    signature: str,
) -> Path:
    """Derive the evidence sidecar path from the detection cache base path.

    Convention::

        <base>_identity_evidence_<source>_<signature>.npz
    """
    p = Path(base_cache_path)
    stem = p.stem.replace("_detections", "").replace("_detection", "")
    sidecar_name = f"{stem}_identity_evidence_{source_name}_{signature}.npz"
    return p.parent / sidecar_name
