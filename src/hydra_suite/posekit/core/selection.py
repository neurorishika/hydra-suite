"""Smart Select batch building for PoseKit (pure numpy, no Qt).

Built from the same selectors as ClassKit's active-learning batch
(:mod:`hydra_suite.core.selection`), with pose-specific signals:

* **Pose error** — on frames that have both a human label and a model
  prediction, the mean keypoint error over visible ground-truth keypoints,
  divided by the object's scale (the Evaluation dialog's ``mean_error_norm``,
  except that a keypoint the model missed counts as ``miss_penalty``).  Frames above
  ``error_threshold`` are *known failures*; unlabeled frames that look like
  them are picked by the error-neighbour slot.
* **Uncertainty** — ``1 - mean keypoint confidence`` of the model's
  prediction (a missed keypoint has confidence 0).

Selection works on *units*: individual crops, or whole source frames in Frame
Mode.  A frame's embedding is the normalised mean of its instances', it counts
as labeled when any instance is, and its error / uncertainty is its worst
instance's, so one code path serves both modes.

Slots, in order (``PoseBatchConfig`` defaults): error neighbours 20%,
uncertainty 25%, rare clusters 10%, diversity 30% (greedy coverage: each pick
most reduces the distance from unlabeled frames to their nearest labeled one,
so it fills the gaps labeling has left), representative 10%, and
the remainder as random audits.  Slots without a signal (no predictions, no
known errors yet) leave their budget to a diversity top-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from hydra_suite.core.selection.batch import SlotBatch, slot_budgets
from hydra_suite.core.selection.clustering import label_coverage
from hydra_suite.core.selection.density import (
    compute_cluster_densities,
    select_coverage_samples,
)
from hydra_suite.core.selection.selectors import (
    AuditSelector,
    ErrorNeighborSelector,
    RareClusterSelector,
    RepresentativeSelector,
    errors_from_scores,
    l2_normalize,
    top_scores,
)

# Reasons, in slot order (also the order the dialog explains them in).
REASON_ERROR = "error_neighbor"
REASON_UNCERTAINTY = "uncertainty"
REASON_RARE = "rare_cluster"
REASON_DIVERSITY = "diversity"
REASON_REPRESENTATIVE = "representative"
REASON_AUDIT = "audit"
REASONS = (
    REASON_ERROR,
    REASON_UNCERTAINTY,
    REASON_RARE,
    REASON_DIVERSITY,
    REASON_REPRESENTATIVE,
    REASON_AUDIT,
)
# Uncertainty at or below this is "fully confident": no signal for the slot.
UNCERTAINTY_FLOOR = 1e-6

REASON_LABELS = {
    REASON_ERROR: "Near a model error",
    REASON_UNCERTAINTY: "Model unsure",
    REASON_RARE: "Rare cluster",
    REASON_DIVERSITY: "Fills a gap",
    REASON_REPRESENTATIVE: "Typical, under-labeled",
    REASON_AUDIT: "Random check",
}


# ── per-frame pose signals ───────────────────────────────────────────────────


def object_scale(
    bbox: Optional[Tuple[float, float, float, float]], width: float, height: float
) -> float:
    """Object scale in pixels: the longer side of the (normalised) bbox."""
    if bbox is not None:
        _, _, bw, bh = bbox
        scale = max(float(bw) * width, float(bh) * height)
    else:
        scale = max(width, height)
    if not scale or scale <= 0 or not math.isfinite(scale):
        scale = max(width, height, 1.0)
    return float(scale)


def pose_error_norm(
    gt_kpts: Sequence[Tuple[float, float, float]],
    preds: Optional[Sequence[Sequence[float]]],
    width: float,
    height: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    miss_penalty: float = 1.0,
) -> float:
    """Mean keypoint error of a prediction against a label, per object scale.

    ``gt_kpts`` are ``(x, y, v)`` in normalised image coordinates (YOLO pose
    labels); ``preds`` are ``(x, y, conf)`` in pixels.  Only visible
    ground-truth keypoints (``v > 0``) count.  This is the Evaluation dialog's
    ``mean_error_norm`` with one difference: a visible keypoint the model did
    not predict (no entry, confidence ``<= 0`` or non-finite position) counts
    as ``miss_penalty`` instead of being skipped, since a miss is an error.
    Returns NaN when nothing is visible.
    """
    if width <= 0 or height <= 0:
        return float("nan")
    scale = object_scale(bbox, width, height)
    errs: List[float] = []
    preds = list(preds or [])
    for k, kp in enumerate(gt_kpts):
        gx, gy, gv = float(kp[0]), float(kp[1]), float(kp[2])
        if gv <= 0:
            continue
        if k >= len(preds) or len(preds[k]) < 2:
            errs.append(miss_penalty)
            continue
        px, py = float(preds[k][0]), float(preds[k][1])
        conf = float(preds[k][2]) if len(preds[k]) > 2 else 1.0
        if not (math.isfinite(px) and math.isfinite(py)) or not conf > 0:
            errs.append(miss_penalty)
            continue
        errs.append(math.hypot(px - gx * width, py - gy * height) / scale)
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def pose_uncertainty(preds: Optional[Sequence[Sequence[float]]]) -> float:
    """``1 - mean keypoint confidence``; NaN when there is no prediction."""
    if not preds:
        return float("nan")
    confs = []
    for p in preds:
        c = float(p[2]) if len(p) > 2 else 0.0
        if not math.isfinite(c):
            c = 0.0
        confs.append(min(max(c, 0.0), 1.0))
    return float(1.0 - np.mean(confs))


# ── units (crops, or frames in Frame Mode) ───────────────────────────────────


def build_units(keys: Optional[Sequence[Hashable]], n: int) -> List[List[int]]:
    """Group ``n`` instances into units by ``keys`` (``None`` = one per instance).

    Units are ordered by their first instance, and members keep index order.
    """
    if keys is None:
        return [[i] for i in range(n)]
    groups: Dict[Hashable, List[int]] = {}
    for i in range(n):
        groups.setdefault(keys[i], []).append(i)
    return sorted(groups.values(), key=lambda m: m[0])


def unit_embeddings(embeddings: np.ndarray, units: List[List[int]]) -> np.ndarray:
    """Normalised mean of each unit's (normalised) member embeddings."""
    emb = l2_normalize(embeddings)
    if all(len(m) == 1 for m in units):
        return emb[[m[0] for m in units]]
    out = np.stack([emb[m].mean(axis=0) for m in units])
    return l2_normalize(out)


def unit_any(mask: np.ndarray, units: List[List[int]]) -> np.ndarray:
    """True for a unit when any member is True."""
    m = np.asarray(mask, dtype=bool)
    return np.array([bool(m[u].any()) for u in units], dtype=bool)


def unit_max(
    values: Optional[np.ndarray], units: List[List[int]]
) -> Optional[np.ndarray]:
    """Worst (largest finite) member value per unit; NaN when none is finite."""
    if values is None:
        return None
    v = np.asarray(values, dtype=np.float64)
    out = np.full(len(units), np.nan)
    for j, u in enumerate(units):
        vals = v[u]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[j] = float(vals.max())
    return out


# ── batch ────────────────────────────────────────────────────────────────────


@dataclass
class PoseBatchConfig:
    """Mix and knobs for a Smart Select batch."""

    batch_size: int = 50
    error_fraction: float = 0.20
    uncertainty_fraction: float = 0.25
    rare_fraction: float = 0.10
    rare_alpha: float = 0.75  # 0 = proportional to cluster size, 1 = equal per cluster
    diversity_fraction: float = 0.30
    representative_fraction: float = 0.10
    error_threshold: float = 0.05  # mean error / object scale above this = failure
    dedup_threshold: Optional[float] = None  # cosine; drop near-duplicates of picks
    seed: int = 0


@dataclass
class PoseBatchResult:
    """An ordered batch of units with the reason each was picked."""

    indices: List[int]  # unit indices, pick order
    reasons: Dict[int, str]
    breakdown: Dict[str, List[int]]
    info: Dict[str, object] = field(default_factory=dict)


def select_pose_batch(
    embeddings: np.ndarray,
    candidate_mask: np.ndarray,
    labeled_mask: np.ndarray,
    cluster_ids: np.ndarray,
    config: Optional[PoseBatchConfig] = None,
    uncertainty: Optional[np.ndarray] = None,
    error_scores: Optional[np.ndarray] = None,
) -> PoseBatchResult:
    """Pick an ordered batch of units to label next.

    Args:
        embeddings: (N, D) unit embeddings (normalised inside)
        candidate_mask: (N,) units that may be picked
        labeled_mask: (N,) units that already carry a label (coverage,
            diversity reference, error sources)
        cluster_ids: (N,) cluster of every unit (labeled ones too)
        config: batch mix
        uncertainty: (N,) model uncertainty per unit, NaN where unknown
        error_scores: (N,) pose error per labeled unit, NaN where unknown

    ``info`` carries ``n_errors``, ``error_sources`` (pick -> error unit),
    ``error_severity`` (error unit -> error), ``uncertainty`` (pick -> value),
    ``coverage`` (cluster -> labeled fraction), ``topped_up`` and
    ``suppressed`` (near-duplicates dropped).
    """
    cfg = config or PoseBatchConfig()
    emb = l2_normalize(embeddings)
    n = len(emb)
    cand = np.asarray(candidate_mask, dtype=bool)
    labeled = np.asarray(labeled_mask, dtype=bool)
    ids = np.asarray(cluster_ids)
    if not (len(cand) == len(labeled) == len(ids) == n):
        raise ValueError("embeddings, masks and cluster_ids must have equal length")
    rng = np.random.default_rng(cfg.seed)
    info: Dict[str, object] = {}

    sb = SlotBatch(cand, emb, cfg.dedup_threshold)
    avail = sb.available
    batch_size = min(int(cfg.batch_size), sb.remaining)
    budgets = slot_budgets(
        batch_size,
        {
            REASON_ERROR: cfg.error_fraction,
            REASON_UNCERTAINTY: cfg.uncertainty_fraction,
            REASON_RARE: cfg.rare_fraction,
            REASON_DIVERSITY: cfg.diversity_fraction,
            REASON_REPRESENTATIVE: cfg.representative_fraction,
        },
        remainder=REASON_AUDIT,
    )
    coverage = label_coverage(ids, labeled)
    info["coverage"] = coverage

    # 1. Error neighbours: unlabeled frames that look like known failures.
    if error_scores is not None and budgets[REASON_ERROR] > 0:
        scores = np.where(labeled, np.asarray(error_scores, dtype=np.float64), np.nan)
        errors = errors_from_scores(scores, cfg.error_threshold)
        info["n_errors"] = len(errors)
        info["error_severity"] = {
            int(i): float(s) for i, s in zip(errors.indices, errors.severity)
        }
        if len(errors):
            # Never the failures themselves (or other labeled frames, which
            # can be candidates under "All frames" or in Frame Mode).
            res = ErrorNeighborSelector().select(
                emb, errors, avail & ~labeled, budgets[REASON_ERROR]
            )
            sb.claim(REASON_ERROR, res.indices)
            info["error_sources"] = {
                int(i): int(res.source_error[int(i)])
                for i in res.indices
                if int(i) in res.source_error
            }

    # 2. Uncertainty: the model's least confident predictions.
    if uncertainty is not None and budgets[REASON_UNCERTAINTY] > 0:
        unc = np.asarray(uncertainty, dtype=np.float64)
        # A fully confident prediction carries no signal; ties are broken at
        # random so equal scores don't all come from the start of the list.
        unc = np.where(unc > UNCERTAINTY_FLOOR, unc, np.nan)
        picked = top_scores(unc, budgets[REASON_UNCERTAINTY], avail, rng=rng)
        taken = sb.claim(REASON_UNCERTAINTY, picked)
        info["uncertainty"] = {int(i): float(unc[i]) for i in taken}

    # 3. Rare clusters: small or under-labeled clusters.
    if budgets[REASON_RARE] > 0 and avail.any():
        sb.claim(
            REASON_RARE,
            RareClusterSelector(cfg.rare_alpha).select(
                ids, avail, budgets[REASON_RARE], coverage, rng=rng
            ),
        )

    def diverse(n_pick: int, reason: str) -> None:
        if n_pick <= 0 or not avail.any():
            return
        avail_idx = np.where(avail)[0]
        covered = labeled.copy()
        covered[sb.chosen] = True
        local = select_coverage_samples(
            emb[avail_idx],
            n_pick,
            points=emb[~covered],
            reference=emb[covered],
            rng=rng,
        )
        sb.claim(reason, avail_idx[np.asarray(local, dtype=int)])

    # 4. Diversity: fill the biggest gaps the labeled set (and earlier picks)
    # leave, weighted by how many unlabeled frames each pick covers.
    diverse(budgets[REASON_DIVERSITY], REASON_DIVERSITY)

    # 5. Representative: dense clusters with low label coverage.
    if budgets[REASON_REPRESENTATIVE] > 0 and avail.any() and n >= 2:
        densities = compute_cluster_densities(emb, ids)
        sb.claim(
            REASON_REPRESENTATIVE,
            RepresentativeSelector().select(
                emb,
                ids,
                coverage,
                densities,
                budgets[REASON_REPRESENTATIVE],
                avail,
                rng=rng,
            ),
        )

    # 6. Audits: random spot checks.
    if budgets[REASON_AUDIT] > 0 and avail.any():
        sb.claim(
            REASON_AUDIT, AuditSelector().select(budgets[REASON_AUDIT], avail, rng=rng)
        )

    # 7. Top-up: slots with no signal hand their budget to diversity.  Loop:
    # with near-duplicate suppression a pick can knock out later picks of
    # the same call, so one pass may still leave the batch short.
    before = len(sb)
    while len(sb) < batch_size and avail.any():
        n_before = len(sb)
        diverse(batch_size - len(sb), REASON_DIVERSITY)
        if len(sb) == n_before:
            break
    if len(sb) > before:
        info["topped_up"] = len(sb) - before
    info["suppressed"] = sb.suppressed

    return PoseBatchResult(
        indices=list(sb.chosen),
        reasons=dict(sb.reason_of),
        breakdown=sb.breakdown,
        info=info,
    )
