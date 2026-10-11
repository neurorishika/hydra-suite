"""The Smart Select job: cluster, score pose signals, build the batch.

Runs off the GUI thread (``SmartSelectWorker``) but is plain Python so it can
be tested and scripted without Qt.  Inputs are per *instance* (one image /
crop); selection runs on units (instances, or source frames in Frame Mode,
see :mod:`hydra_suite.posekit.core.selection`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Sequence

import numpy as np

from hydra_suite.core.selection.clustering import SKLearnClusterer, auto_cluster_count

from .selection import (
    PoseBatchConfig,
    PoseBatchResult,
    build_units,
    pose_error_norm,
    pose_uncertainty,
    select_pose_batch,
    unit_any,
    unit_embeddings,
    unit_max,
)

Predictions = Dict[str, List[Sequence[float]]]


class SmartSelectCanceled(Exception):
    """Raised when the job is canceled between steps."""


@dataclass
class SmartSelectRequest:
    """Everything the job needs, captured on the GUI thread."""

    embeddings: np.ndarray  # (N, D), one row per image_paths entry
    image_paths: List[Path]
    candidate_mask: np.ndarray  # (N,) instances that may be picked
    labeled_mask: np.ndarray  # (N,) instances with a saved label
    config: PoseBatchConfig = field(default_factory=PoseBatchConfig)
    unit_keys: Optional[Sequence[Hashable]] = None  # Frame Mode grouping
    n_clusters: int = 0  # 0 = auto (sqrt of the unit count)
    cluster_method: str = "kmeans"
    predictions: Optional[Predictions] = None  # path -> [(x, y, conf), ...] px
    label_paths: Dict[int, Path] = field(default_factory=dict)  # labeled idx -> txt
    num_keypoints: int = 0
    # Instances without a usable embedding (unreadable images).  They take no
    # part in clustering or selection; a frame keeps its other individuals.
    unusable_mask: Optional[np.ndarray] = None


@dataclass
class SmartSelectOutcome:
    """The batch plus what the dialog needs to explain it."""

    units: List[List[int]]
    unit_clusters: np.ndarray  # -1 for units with no usable embedding
    instance_clusters: np.ndarray
    unit_embeddings: np.ndarray
    unit_labeled: np.ndarray
    result: PoseBatchResult
    unit_uncertainty: Optional[np.ndarray]
    unit_error: Optional[np.ndarray]
    n_clusters: int
    n_predicted: int = 0  # instances with a cached prediction
    n_error_scored: int = 0  # labeled instances compared to a prediction

    unusable: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def usable_units(self) -> np.ndarray:
        """Units that took part in clustering and selection."""
        return self.unit_clusters >= 0

    def batch_instances(self) -> List[int]:
        """Instance indices of the batch, unit by unit in pick order.

        Unreadable individuals of a picked frame are left out.
        """
        bad = self.unusable
        return [
            i
            for u in self.result.indices
            for i in self.units[u]
            if not (len(bad) and bad[i])
        ]


def lookup_prediction(predictions: Optional[Predictions], path: Path):
    """Cached prediction for ``path`` (stored under the path or its resolution)."""
    if not predictions:
        return None
    key = str(path)
    if key in predictions:
        return predictions[key]
    try:
        return predictions.get(str(Path(path).resolve()))
    except OSError:
        return None


def _image_size(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _load_label(label_path: Path, k: int):
    from .extensions import load_yolo_pose_label

    try:
        loaded = load_yolo_pose_label(Path(label_path), k)
    except (OSError, ValueError):
        return None
    if not loaded:
        return None
    _cls, kpts, bbox = loaded
    return [(kp.x, kp.y, kp.v) for kp in kpts], bbox


def score_instances(req: SmartSelectRequest, check: Callable[[], None] = lambda: None):
    """Per-instance uncertainty (candidates) and pose error (labeled).

    Returns ``(uncertainty, error, n_predicted, n_error_scored)``; arrays are
    ``None`` when there are no predictions at all.
    """
    n = len(req.image_paths)
    if not req.predictions:
        return None, None, 0, 0
    unc = np.full(n, np.nan)
    err = np.full(n, np.nan)
    n_pred = n_scored = 0
    labeled = np.asarray(req.labeled_mask, dtype=bool)
    for i, path in enumerate(req.image_paths):
        if i % 256 == 0:
            check()
        preds = lookup_prediction(req.predictions, path)
        if not preds:
            continue
        n_pred += 1
        unc[i] = pose_uncertainty(preds)
        if labeled[i] and i in req.label_paths:
            loaded = _load_label(req.label_paths[i], req.num_keypoints)
            if loaded is None:
                continue
            gt, bbox = loaded
            try:
                w, h = _image_size(path)
            except OSError:
                continue
            e = pose_error_norm(gt, preds, w, h, bbox=bbox)
            if np.isfinite(e):
                err[i] = e
                n_scored += 1
    return unc, err, n_pred, n_scored


def run_smart_select(
    req: SmartSelectRequest,
    status: Callable[[str], None] = lambda s: None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> SmartSelectOutcome:
    """Cluster all units, score model signals and pick the batch."""

    def check() -> None:
        if is_canceled():
            raise SmartSelectCanceled()

    n = len(req.image_paths)
    if len(req.embeddings) != n:
        raise ValueError("embeddings must have one row per image")

    units = build_units(req.unit_keys, n)
    bad = (
        np.zeros(n, dtype=bool)
        if req.unusable_mask is None
        else np.asarray(req.unusable_mask, dtype=bool)
    )
    # Units with at least one readable member take part; a frame's embedding
    # comes from its readable individuals only.
    use_units = [u for u, m in enumerate(units) if not bad[m].all()]
    sub_units = [[i for i in units[u] if not bad[i]] for u in use_units]
    emb_all = np.asarray(req.embeddings)
    unit_emb = np.zeros((len(units), emb_all.shape[1]), dtype=np.float32)
    unit_clusters = np.full(len(units), -1, dtype=int)
    unit_labeled = unit_any(req.labeled_mask, units)
    if not sub_units:
        raise ValueError("None of the images could be read.")
    sub_emb = unit_embeddings(emb_all, sub_units)
    unit_emb[use_units] = sub_emb
    sub_labeled = unit_labeled[use_units]
    sub_cand = unit_any(np.asarray(req.candidate_mask) & ~bad, sub_units)

    k = int(req.n_clusters) or auto_cluster_count(len(sub_units))
    k = max(1, min(k, len(sub_units)))
    status(f"Clustering {len(sub_units):,} items into {k} clusters…")
    check()
    clusterer = SKLearnClusterer(n_clusters=k, method=req.cluster_method)
    sub_clusters = np.asarray(clusterer.fit(sub_emb), dtype=int)
    unit_clusters[use_units] = sub_clusters
    instance_clusters = np.full(n, -1, dtype=int)
    for u, members in enumerate(units):
        instance_clusters[members] = unit_clusters[u]

    check()
    if req.predictions:
        status("Scoring model predictions…")
    unc, err, n_pred, n_scored = score_instances(req, check)
    if unc is not None:  # stale predictions of unreadable images don't count
        unc[bad] = np.nan
        err[bad] = np.nan
    unit_unc = unit_max(unc, units)
    unit_err = unit_max(err, units)

    check()
    status("Selecting the batch…")
    sub_result = select_pose_batch(
        sub_emb,
        sub_cand,
        sub_labeled,
        sub_clusters,
        req.config,
        uncertainty=None if unit_unc is None else unit_unc[use_units],
        error_scores=None if unit_err is None else unit_err[use_units],
    )
    result = _remap_result(sub_result, use_units)
    return SmartSelectOutcome(
        units=units,
        unit_clusters=unit_clusters,
        instance_clusters=instance_clusters,
        unit_embeddings=unit_emb,
        unit_labeled=unit_labeled,
        result=result,
        unit_uncertainty=unit_unc,
        unit_error=unit_err,
        n_clusters=int(sub_clusters.max()) + 1 if len(sub_clusters) else 0,
        n_predicted=n_pred,
        n_error_scored=n_scored,
        unusable=bad,
    )


def _remap_result(res: PoseBatchResult, use_units: List[int]) -> PoseBatchResult:
    """Map a result computed on the usable units back to all unit indices."""
    m = list(use_units)

    def g(i: int) -> int:
        return int(m[int(i)])

    info = dict(res.info)
    if "error_sources" in info:
        info["error_sources"] = {g(k): g(v) for k, v in info["error_sources"].items()}
    if "error_severity" in info:
        info["error_severity"] = {g(k): v for k, v in info["error_severity"].items()}
    if "uncertainty" in info:
        info["uncertainty"] = {g(k): v for k, v in info["uncertainty"].items()}
    return PoseBatchResult(
        indices=[g(i) for i in res.indices],
        reasons={g(k): v for k, v in res.reasons.items()},
        breakdown={r: [g(i) for i in v] for r, v in res.breakdown.items()},
        info=info,
    )
