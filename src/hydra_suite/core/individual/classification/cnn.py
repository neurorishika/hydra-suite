"""CNN identity backend for MAT: config, predictions, cache, and inference backend.

Pure Python — no Qt dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hydra_suite.runtime.resolver import ResolvedBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CNNIdentityConfig:
    """Configuration for CNN Classifier identity method."""

    model_path: str = ""
    confidence: float = 0.5
    label: str = ""
    batch_size: int = 64
    match_bonus: float = 0.5
    mismatch_penalty: float = 1.0
    window: int = 10
    scoring_mode: str = "atomic"


@dataclass(frozen=True)
class ClassPrediction:
    """Single detection's classifier output.

    For flat models ``factor_names`` has length 1 and the ``class_name`` /
    ``confidence`` properties give the scalar view. For multi-head models
    each tuple index is a distinct factor.
    """

    det_index: int
    factor_names: tuple[str, ...]
    class_names: tuple[str | None, ...]
    confidences: tuple[float, ...]

    @property
    def is_unknown(self) -> tuple[bool, ...]:
        return tuple(name == "unknown" for name in self.class_names)

    @property
    def class_name(self) -> str | None:
        if len(self.factor_names) != 1:
            raise ValueError(
                "ClassPrediction.class_name is only defined for flat (K=1) "
                "predictions; use class_names tuple for multi-factor"
            )
        return self.class_names[0]

    @property
    def confidence(self) -> float:
        if len(self.factor_names) != 1:
            raise ValueError(
                "ClassPrediction.confidence is only defined for flat (K=1) "
                "predictions; use confidences tuple for multi-factor"
            )
        return self.confidences[0]


# ---------------------------------------------------------------------------
# CNNIdentityBackend
# ---------------------------------------------------------------------------


class CNNIdentityBackend:
    """High-level wrapper around ``ClassifierBackend`` that adds CNN identity
    semantics: per-factor confidence thresholding, class-name lookup, and
    scoring-mode validation.
    """

    def __init__(
        self,
        config: CNNIdentityConfig,
        model_path: str | None = None,
        resolved: "ResolvedBackend | None" = None,
    ) -> None:
        from hydra_suite.core.individual.classification.backend import ClassifierBackend
        from hydra_suite.core.individual.classification.errors import (
            ClassifierConfigError,
        )
        from hydra_suite.runtime.resolver import ResolvedBackend

        self._config = config
        resolved_path = str(model_path or config.model_path or "")
        if not resolved_path:
            raise ClassifierConfigError("CNN identity backend requires a model_path")
        self._backend = ClassifierBackend(
            resolved_path,
            (
                resolved
                if resolved is not None
                else ResolvedBackend("torch", "cpu", False)
            ),
        )
        meta = self._backend.metadata
        if meta.is_multihead and config.scoring_mode not in (
            "atomic",
            "per_head_average",
        ):
            raise ClassifierConfigError(
                f"multi-head CNN identity model {resolved_path!r} requires "
                f"scoring_mode in {{atomic, per_head_average}}; got "
                f"{config.scoring_mode!r}"
            )

    @property
    def metadata(self):
        return self._backend.metadata

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(self._backend.metadata.factor_names)

    def predict_batch(self, crops: list[np.ndarray]) -> list[ClassPrediction]:
        """Run inference and return per-crop ``ClassPrediction`` instances with
        per-factor confidence thresholding applied.
        """
        if not crops:
            return []
        raw = self._backend.predict_batch(crops)
        meta = self._backend.metadata
        factor_names = tuple(meta.factor_names)
        threshold = float(self._config.confidence)
        results: list[ClassPrediction] = []
        for det_idx, per_factor in enumerate(raw):
            names: list[str | None] = []
            confs: list[float] = []
            for k, probs in enumerate(per_factor):
                best_idx = int(np.argmax(probs))
                best_conf = float(probs[best_idx])
                class_list = meta.class_names_per_factor[k]
                if best_conf >= threshold and 0 <= best_idx < len(class_list):
                    names.append(class_list[best_idx])
                else:
                    names.append(None)
                confs.append(best_conf)
            results.append(
                ClassPrediction(
                    det_index=det_idx,
                    factor_names=factor_names,
                    class_names=tuple(names),
                    confidences=tuple(confs),
                )
            )
        return results

    def predict_batch_cuda(self, crops) -> list["ClassPrediction"]:
        """GPU-native batch prediction path (Streaming Phase 2).

        Delegates to ``ClassifierBackend.predict_batch_cuda()`` when the
        underlying backend supports it.  Falls back transparently to the CPU
        path when GPU execution is not available.

        Parameters
        ----------
        crops:
            Either a list of CPU ``np.ndarray`` crops or a stacked CUDA tensor
            ``(B, C, H, W)``.  The underlying backend selects the appropriate
            execution path based on input type and the configured runtime.

        Returns
        -------
        list[ClassPrediction]
            Same contract as ``predict_batch()``.
        """
        if crops is None:
            return []
        if hasattr(crops, "__len__") and len(crops) == 0:
            return []
        # Delegate to the GPU-capable backend method; fall back to CPU batch
        try:
            raw = self._backend.predict_batch_cuda(crops)
        except (AttributeError, NotImplementedError):
            # Backend does not support CUDA crops — convert to list and use CPU path

            if hasattr(crops, "cpu"):
                raw_np = crops.cpu().numpy()
                cpu_crops = [raw_np[i].transpose(1, 2, 0) for i in range(len(raw_np))]
            else:
                cpu_crops = list(crops)
            raw = self._backend.predict_batch(cpu_crops)

        meta = self._backend.metadata
        factor_names = tuple(meta.factor_names)
        threshold = float(self._config.confidence)
        results: list[ClassPrediction] = []
        for det_idx, per_factor in enumerate(raw):
            names: list[str | None] = []
            confs: list[float] = []
            for k, probs in enumerate(per_factor):
                probs_arr = np.asarray(probs, dtype=np.float32)
                best_idx = int(np.argmax(probs_arr))
                best_conf = float(probs_arr[best_idx])
                class_list = meta.class_names_per_factor[k]
                if best_conf >= threshold and 0 <= best_idx < len(class_list):
                    names.append(class_list[best_idx])
                else:
                    names.append(None)
                confs.append(best_conf)
            results.append(
                ClassPrediction(
                    det_index=det_idx,
                    factor_names=factor_names,
                    class_names=tuple(names),
                    confidences=tuple(confs),
                )
            )
        return results

    def predict_batch_posteriors(
        self,
        crops: list[np.ndarray],
        calibration=None,
    ) -> tuple[list["ClassPrediction"], list[list[np.ndarray]]]:
        """Calibrated posterior output hook (Streaming Phase 2 / Identity Phase 0).

        Runs the same batch inference as ``predict_batch()`` but additionally
        returns the full calibrated probability distribution over every class in
        every factor, enabling the identity overhaul to build
        ``IdentityEvidence`` objects without re-running inference.

        Parameters
        ----------
        crops:
            List of ``np.ndarray`` crops (same contract as ``predict_batch``).
        calibration:
            Optional ``CalibrationModel`` from ``identity.calibration``.
            When ``None``, raw softmax probabilities are returned as-is.

        Returns
        -------
        predictions: list[ClassPrediction]
            Hard predictions (same as ``predict_batch()``).
        posteriors: list[list[np.ndarray]]
            ``posteriors[det_index][factor_index]`` is a shape ``(K_f,)``
            float64 array of calibrated probabilities over the factor's
            class list.  The caller maps these to catalog log-priors via
            ``IdentityCatalog.cnn_log_prior()``.
        """
        if not crops:
            return [], []
        raw = self._backend.predict_batch(crops)
        meta = self._backend.metadata
        factor_names = tuple(meta.factor_names)
        threshold = float(self._config.confidence)

        predictions: list[ClassPrediction] = []
        posteriors: list[list[np.ndarray]] = []

        for det_idx, per_factor in enumerate(raw):
            names: list[str | None] = []
            confs: list[float] = []
            det_posteriors: list[np.ndarray] = []

            for k, probs in enumerate(per_factor):
                probs_arr = np.asarray(probs, dtype=np.float64)

                # Apply calibration if provided
                if calibration is not None:
                    # calibrate_probs expects shape (..., K); add batch dim
                    log_p = calibration.calibrate_probs(probs_arr[None, :])[0]
                    cal_probs = np.exp(log_p - log_p.max())
                    cal_probs /= cal_probs.sum()
                else:
                    cal_probs = (
                        probs_arr / probs_arr.sum()
                        if probs_arr.sum() > 0
                        else probs_arr
                    )

                det_posteriors.append(cal_probs)

                best_idx = int(np.argmax(cal_probs))
                best_conf = float(cal_probs[best_idx])
                class_list = meta.class_names_per_factor[k]
                if best_conf >= threshold and 0 <= best_idx < len(class_list):
                    names.append(class_list[best_idx])
                else:
                    names.append(None)
                confs.append(best_conf)

            predictions.append(
                ClassPrediction(
                    det_index=det_idx,
                    factor_names=factor_names,
                    class_names=tuple(names),
                    confidences=tuple(confs),
                )
            )
            posteriors.append(det_posteriors)

        return predictions, posteriors

    def close(self) -> None:
        self._backend.close()


# ---------------------------------------------------------------------------
# Hungarian cost helper
# ---------------------------------------------------------------------------


def apply_cnn_identity_cost(
    *,
    track_identity: tuple[str | None, ...] | None,
    det: ClassPrediction | None,
    match_bonus: float,
    mismatch_penalty: float,
    scoring_mode: str,
) -> float:
    """Compute the cost delta contributed by a CNN identity classifier for a
    (track, detection) pair under the given scoring mode.
    """
    if track_identity is None or det is None:
        return 0.0
    det_tuple = tuple(det.class_names)
    if scoring_mode == "atomic":
        return cost_atomic(
            track_identity,
            det_tuple,
            match_bonus=match_bonus,
            mismatch_penalty=mismatch_penalty,
        )
    if scoring_mode == "per_head_average":
        return cost_per_head_average(
            track_identity,
            det_tuple,
            match_bonus=match_bonus,
            mismatch_penalty=mismatch_penalty,
            K=len(det_tuple),
        )
    raise ValueError(f"unknown scoring_mode {scoring_mode!r}")


def cost_atomic(
    track: tuple[str | None, ...],
    det: tuple[str | None, ...],
    *,
    match_bonus: float,
    mismatch_penalty: float,
) -> float:
    """Atomic tuple compare: any ``None`` or ``"unknown"`` in either side -> no signal."""
    for x in (*track, *det):
        if x is None or x == "unknown":
            return 0.0
    return -float(match_bonus) if track == det else +float(mismatch_penalty)


def cost_per_head_average(
    track: tuple[str | None, ...],
    det: tuple[str | None, ...],
    *,
    match_bonus: float,
    mismatch_penalty: float,
    K: int,
) -> float:
    """Per-head average cost. Divisor is always K (not the number of comparable heads)."""
    if K <= 0:
        return 0.0
    contributions = 0.0
    for k in range(K):
        tk = track[k] if k < len(track) else None
        dk = det[k] if k < len(det) else None
        if tk is None or tk == "unknown":
            continue
        if dk is None or dk == "unknown":
            continue
        contributions += -float(match_bonus) if tk == dk else +float(mismatch_penalty)
    return contributions / float(K)
