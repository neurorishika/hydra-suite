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


@dataclass
class SmartSelectOutcome:
    """The batch plus what the dialog needs to explain it."""

    units: List[List[int]]
    unit_clusters: np.ndarray
    instance_clusters: np.ndarray
    unit_embeddings: np.ndarray
    unit_labeled: np.ndarray
    result: PoseBatchResult
    unit_uncertainty: Optional[np.ndarray]
    unit_error: Optional[np.ndarray]
    n_clusters: int
    n_predicted: int = 0  # instances with a cached prediction
    n_error_scored: int = 0  # labeled instances compared to a prediction

    def batch_instances(self) -> List[int]:
        """Instance indices of the batch, unit by unit in pick order."""
        return [i for u in self.result.indices for i in self.units[u]]


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
    unit_emb = unit_embeddings(np.asarray(req.embeddings), units)
    unit_labeled = unit_any(req.labeled_mask, units)
    unit_cand = unit_any(req.candidate_mask, units)

    k = int(req.n_clusters) or auto_cluster_count(len(units))
    k = max(1, min(k, len(units)))
    status(f"Clustering {len(units):,} items into {k} clusters…")
    check()
    clusterer = SKLearnClusterer(n_clusters=k, method=req.cluster_method)
    unit_clusters = np.asarray(clusterer.fit(unit_emb), dtype=int)
    instance_clusters = np.zeros(n, dtype=int)
    for u, members in enumerate(units):
        instance_clusters[members] = unit_clusters[u]

    check()
    if req.predictions:
        status("Scoring model predictions…")
    unc, err, n_pred, n_scored = score_instances(req, check)
    unit_unc = unit_max(unc, units)
    unit_err = unit_max(err, units)

    check()
    status("Selecting the batch…")
    result = select_pose_batch(
        unit_emb,
        unit_cand,
        unit_labeled,
        unit_clusters,
        req.config,
        uncertainty=unit_unc,
        error_scores=unit_err,
    )
    return SmartSelectOutcome(
        units=units,
        unit_clusters=unit_clusters,
        instance_clusters=instance_clusters,
        unit_embeddings=unit_emb,
        unit_labeled=unit_labeled,
        result=result,
        unit_uncertainty=unit_unc,
        unit_error=unit_err,
        n_clusters=int(unit_clusters.max()) + 1 if len(unit_clusters) else 0,
        n_predicted=n_pred,
        n_error_scored=n_scored,
    )
