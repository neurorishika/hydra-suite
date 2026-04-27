"""CNN identity backend for MAT: config, predictions, cache, and inference backend.

Pure Python — no Qt dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
# CNNIdentityCache
# ---------------------------------------------------------------------------

_SENTINEL_NONE = "__NONE__"  # stored in npz when class_name is None
_CACHE_SCHEMA_V2 = 2  # multi-factor format marker
_CACHE_SCHEMA_V3 = 3  # adds per-detection per-factor full probability vectors


class CNNIdentityCache:
    """Persistent .npz cache of per-frame CNN identity predictions.

    Data is accumulated in memory via ``save()`` and written to disk in a
    single compressed write via ``flush()``.  Call ``load()`` during the
    tracking loop to retrieve per-frame predictions.

    Supports two on-disk formats:

    - **Legacy** (no ``factor_names`` key in the .npz): flat single-factor,
      keys ``f{N}_det``, ``f{N}_cls``, ``f{N}_conf``.  Reconstructed on
      load as ``factor_names=("flat",)``.
    - **v2** (``cache_schema_version == 2`` and ``factor_names`` present):
      multi-factor, keys ``f{N}_det``, ``f{N}_cls_k{K}``,
      ``f{N}_conf_k{K}`` for each factor index K.

    The ``factor_names`` constructor argument is used only when writing new
    caches. When loading an existing file the stored factor_names always wins.
    """

    def __init__(
        self,
        cache_path: str | Path,
        factor_names: tuple[str, ...] | None = None,
        class_names_per_factor: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        self._path = Path(cache_path)
        self._factor_names: tuple[str, ...] = (
            tuple(factor_names) if factor_names is not None else ("flat",)
        )
        self._class_names_per_factor: tuple[tuple[str, ...], ...] | None = (
            tuple(tuple(names) for names in class_names_per_factor)
            if class_names_per_factor is not None
            else None
        )
        self._data: dict[str, Any] = {}
        self._is_legacy = False
        if self._path.exists():
            raw = np.load(str(self._path), allow_pickle=False)
            self._data = dict(raw)
            if "factor_names" not in self._data:
                self._is_legacy = True
                self._factor_names = ("flat",)
            else:
                stored = self._data["factor_names"]
                self._factor_names = tuple(str(n) for n in stored)
            # Load class names per factor if present (v3+)
            if self._class_names_per_factor is None:
                loaded_names: list[tuple[str, ...]] = []
                for k in range(len(self._factor_names)):
                    key = f"class_names_k{k}"
                    if key in self._data:
                        loaded_names.append(tuple(str(n) for n in self._data[key]))
                    else:
                        loaded_names.append(())
                if any(loaded_names):
                    self._class_names_per_factor = tuple(loaded_names)

    def exists(self) -> bool:
        """Return True if the cache file exists on disk."""
        return self._path.exists()

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self._factor_names

    @property
    def class_names_per_factor(self) -> tuple[tuple[str, ...], ...] | None:
        """Per-factor class name lists, or None when not stored."""
        return self._class_names_per_factor

    def save(
        self,
        frame_idx: int,
        predictions: list[ClassPrediction],
        posteriors: list[list[np.ndarray] | None] | None = None,
    ) -> None:
        """Update in-memory cache for *frame_idx*. Call flush() when done.

        ``posteriors`` is an optional per-detection list of per-factor probability
        vectors (one np.ndarray of shape (n_classes,) per factor).  When provided
        the full distributions are persisted alongside the top-1 predictions so
        that augmented exports can include per-class probability columns.
        """
        K = len(self._factor_names)
        if not predictions:
            self._data[f"f{frame_idx}_det"] = np.array([], dtype=np.int32)
            for k in range(K):
                self._data[f"f{frame_idx}_cls_k{k}"] = np.array([], dtype=object)
                self._data[f"f{frame_idx}_conf_k{k}"] = np.array([], dtype=np.float32)
        else:
            det_arr = np.array([p.det_index for p in predictions], dtype=np.int32)
            self._data[f"f{frame_idx}_det"] = det_arr
            for k in range(K):
                cls_col = []
                conf_col = []
                for p in predictions:
                    raw = p.class_names[k] if k < len(p.class_names) else None
                    cls_col.append(raw if raw is not None else _SENTINEL_NONE)
                    conf_col.append(
                        float(p.confidences[k]) if k < len(p.confidences) else 0.0
                    )
                self._data[f"f{frame_idx}_cls_k{k}"] = np.array(cls_col, dtype=object)
                self._data[f"f{frame_idx}_conf_k{k}"] = np.array(
                    conf_col, dtype=np.float32
                )

            if posteriors is not None and len(posteriors) == len(predictions):
                for k in range(K):
                    prob_rows = []
                    n_classes = 0
                    for per_det_probs in posteriors:
                        if per_det_probs is not None and k < len(per_det_probs):
                            vec = np.asarray(per_det_probs[k], dtype=np.float32)
                            n_classes = max(n_classes, len(vec))
                            prob_rows.append(vec)
                        else:
                            prob_rows.append(None)
                    if n_classes > 0:
                        mat = np.full(
                            (len(predictions), n_classes), np.nan, dtype=np.float32
                        )
                        for i, row in enumerate(prob_rows):
                            if row is not None:
                                mat[i, : len(row)] = row
                        self._data[f"f{frame_idx}_probs_k{k}"] = mat

    def flush(self) -> None:
        """Write all in-memory predictions to disk (v3 format when probs present)."""
        if not self._data:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = dict(self._data)
        out["factor_names"] = np.array(list(self._factor_names), dtype=object)
        has_probs = any(k.endswith("_probs_k0") for k in out)
        if has_probs:
            out["cache_schema_version"] = np.array(_CACHE_SCHEMA_V3, dtype=np.int32)
            if self._class_names_per_factor is not None:
                for k, names in enumerate(self._class_names_per_factor):
                    out[f"class_names_k{k}"] = np.array(list(names), dtype=object)
        else:
            out["cache_schema_version"] = np.array(_CACHE_SCHEMA_V2, dtype=np.int32)
        np.savez_compressed(str(self._path), **out)

    def load_probs(self, frame_idx: int) -> list[list[np.ndarray | None] | None] | None:
        """Return per-detection per-factor probability vectors for *frame_idx*.

        Returns ``None`` when no probability data was stored.  Otherwise returns
        a list aligned with ``load(frame_idx)``: each entry is either ``None``
        (no probs for that detection) or a list of K np.ndarray prob vectors.
        """
        key_det = f"f{frame_idx}_det"
        if key_det not in self._data:
            return None
        det_arr = self._data[key_det]
        n = len(det_arr)
        K = len(self._factor_names)

        has_any = any(f"f{frame_idx}_probs_k{k}" in self._data for k in range(K))
        if not has_any:
            return None

        results: list[list[np.ndarray | None] | None] = []
        for i in range(n):
            per_factor: list[np.ndarray | None] = []
            found = False
            for k in range(K):
                mat = self._data.get(f"f{frame_idx}_probs_k{k}")
                if mat is not None and i < len(mat):
                    row = np.asarray(mat[i], dtype=np.float32)
                    if not np.all(np.isnan(row)):
                        per_factor.append(row)
                        found = True
                        continue
                per_factor.append(None)
            results.append(per_factor if found else None)
        return results

    def load(self, frame_idx: int) -> list[ClassPrediction]:
        """Return saved predictions for *frame_idx*, or [] if not found."""
        key_det = f"f{frame_idx}_det"
        if key_det not in self._data:
            return []
        det_arr = self._data[key_det]
        if len(det_arr) == 0:
            return []

        if self._is_legacy:
            # Legacy single-factor format: keys f{N}_cls and f{N}_conf
            cls_arr = self._data.get(f"f{frame_idx}_cls", np.array([], dtype=object))
            conf_arr = self._data.get(
                f"f{frame_idx}_conf", np.zeros(len(det_arr), dtype=np.float32)
            )
            results = []
            for i in range(len(det_arr)):
                raw_cls = str(cls_arr[i]) if i < len(cls_arr) else _SENTINEL_NONE
                class_name = None if raw_cls == _SENTINEL_NONE else raw_cls
                results.append(
                    ClassPrediction(
                        det_index=int(det_arr[i]),
                        factor_names=("flat",),
                        class_names=(class_name,),
                        confidences=(float(conf_arr[i]) if i < len(conf_arr) else 0.0,),
                    )
                )
            return results

        # v2 multi-factor format: keys f{N}_cls_k{K} and f{N}_conf_k{K}
        K = len(self._factor_names)
        per_factor_cls: list[Any] = []
        per_factor_conf: list[Any] = []
        for k in range(K):
            per_factor_cls.append(
                self._data.get(
                    f"f{frame_idx}_cls_k{k}",
                    np.full(len(det_arr), _SENTINEL_NONE),
                )
            )
            per_factor_conf.append(
                self._data.get(
                    f"f{frame_idx}_conf_k{k}",
                    np.zeros(len(det_arr), dtype=np.float32),
                )
            )
        results = []
        for i in range(len(det_arr)):
            names: list[str | None] = []
            confs: list[float] = []
            for k in range(K):
                raw = str(per_factor_cls[k][i])
                names.append(None if raw == _SENTINEL_NONE else raw)
                confs.append(float(per_factor_conf[k][i]))
            results.append(
                ClassPrediction(
                    det_index=int(det_arr[i]),
                    factor_names=self._factor_names,
                    class_names=tuple(names),
                    confidences=tuple(confs),
                )
            )
        return results

    def get_cached_frames(self) -> list[int]:
        """Return sorted frame indices present in the cache."""
        frames = []
        for key in self._data:
            if not key.startswith("f") or not key.endswith("_det"):
                continue
            try:
                frames.append(int(str(key)[1:-4]))
            except ValueError:
                continue
        return sorted(set(frames))


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
        compute_runtime: str = "cpu",
    ) -> None:
        from hydra_suite.core.identity.classification.backend import ClassifierBackend
        from hydra_suite.core.identity.classification.errors import (
            ClassifierConfigError,
        )

        self._config = config
        resolved_path = str(model_path or config.model_path or "")
        if not resolved_path:
            raise ClassifierConfigError("CNN identity backend requires a model_path")
        self._backend = ClassifierBackend(
            resolved_path, compute_runtime=compute_runtime
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
# TrackCNNHistory
# ---------------------------------------------------------------------------


class TrackCNNHistory:
    """Sliding-window per-track history of multi-factor classifier predictions.

    Per-factor majority vote excludes ``None`` and ``"unknown"`` observations.
    Ties return ``None`` for that factor.
    """

    def __init__(self, *, window: int, factor_names: tuple[str, ...]) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not factor_names:
            raise ValueError("factor_names must be non-empty")
        self._window = int(window)
        self._factor_names = tuple(str(n) for n in factor_names)
        # per-track deque of (class_names_tuple, confidences_tuple)
        from collections import deque

        self._deque_cls = deque
        self._history: dict[
            int, "deque[tuple[tuple[str | None, ...], tuple[float, ...]]]"
        ] = {}

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self._factor_names

    def record(
        self,
        *,
        track_id: int,
        class_names: tuple[str | None, ...],
        confidences: tuple[float, ...],
    ) -> None:
        if len(class_names) != len(self._factor_names):
            raise ValueError(
                f"class_names length {len(class_names)} does not match "
                f"factor_names length {len(self._factor_names)}"
            )
        if len(confidences) != len(self._factor_names):
            raise ValueError(
                f"confidences length {len(confidences)} does not match "
                f"factor_names length {len(self._factor_names)}"
            )
        buf = self._history.get(track_id)
        if buf is None:
            buf = self._deque_cls(maxlen=self._window)
            self._history[track_id] = buf
        buf.append((tuple(class_names), tuple(confidences)))

    def majority_class(self, track_id: int) -> tuple[str | None, ...]:
        buf = self._history.get(track_id)
        if not buf:
            return tuple(None for _ in self._factor_names)
        result: list[str | None] = []
        for k in range(len(self._factor_names)):
            counts: dict[str, int] = {}
            for names_tuple, _confs in buf:
                name = names_tuple[k]
                if name is None or name == "unknown":
                    continue
                counts[name] = counts.get(name, 0) + 1
            if not counts:
                result.append(None)
                continue
            max_count = max(counts.values())
            winners = [name for name, n in counts.items() if n == max_count]
            result.append(winners[0] if len(winners) == 1 else None)
        return tuple(result)

    def clear_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def build_track_identity_list(self) -> dict[int, tuple[str | None, ...]]:
        return {tid: self.majority_class(tid) for tid in self._history}


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
