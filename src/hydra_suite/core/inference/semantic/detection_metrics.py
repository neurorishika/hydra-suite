"""Detection-quality arithmetic: PR curve and average precision.

Pure numpy over plain counts -- no cv2, no torch, no SAM3, no matcher. The
matcher lives next door in ``calibration.match_one_to_one``; keeping the
arithmetic separate is what lets both the offline parity tool and the
training loop use the SAME AP definition without either dragging the other's
dependencies in.

WHY IT LIVES HERE. ``precision_recall_curve`` / ``average_precision`` were
written for ``tools/sam3_parity/compare_models.py``. A per-epoch training
metric cannot import a tools script (tools are not installed with the
package, and `core`/`training` may not depend on them), and a second copy
would be a second definition of AP that could silently drift from the one
every published measurement was made with. So the arithmetic moved down here
and the tool imports it back. Any change to these functions changes both the
stored parity baselines and the training series; treat them as frozen.

AP IS NOT COMPARABLE ACROSS CORPORA. Measured, same models: AP 0.96 on
1766-px tiles versus 0.61-0.69 on 971-px tiles. It is not comparable across
tile geometries, across datasets, or between tile-space and frame-merged
scoring. Read any AP series as a WITHIN-RUN trend only; two runs' numbers
are not on the same scale and differencing them means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    """One confidence-sweep sample: recall and cost at that confidence."""

    confidence: float
    recall: float
    extra_per_frame: float


@dataclass(frozen=True)
class SweepCounts:
    """Corpus-summed instance counts at one confidence.

    ``matched`` / ``missed`` are ground-truth instances (their sum is the
    ground-truth total, so recall is ``matched / (matched + missed)``);
    ``extras`` are predictions matched to nothing.
    """

    confidence: float
    matched: int
    extras: int
    missed: int


def precision_recall_curve(
    points: Sequence[OperatingPoint],
    *,
    missed_per_frame: Sequence[float] | None = None,
    matched_per_frame: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recall/precision arrays sorted by ascending recall.

    ``precision`` at a point is ``matched / (matched + extra)`` in
    per-frame-normalised units (the arbitrary per-frame scale cancels out of
    the ratio). There are three ways to supply ``matched``, in decreasing
    order of preference:

    * ``matched_per_frame`` -- the EXACT matched instance counts, per frame,
      aligned 1:1 with ``points``. No inversion, no singularity. Callers that
      hold raw counts (``average_precision_from_counts``) must use this.
    * ``missed_per_frame`` -- the legacy path, which recovers the scale by
      inverting ``total = missed / (1 - recall)``. That inversion is SINGULAR
      at ``recall == 1`` (``missed == 0``), which is precisely where a
      detector found everything. Because ``total`` (the ground-truth count
      per frame) is a property of the corpus and not of the confidence
      threshold, it is constant across the sweep, so the full-recall points
      borrow the total recovered from the sweep's other points. Only if NO
      point has ``recall < 1`` does the scale stay unrecoverable; precision
      is then NaN there and ``average_precision`` drops it.
    * neither -- precision is monotone-correct but not on an absolute
      per-frame instance scale (``total`` is fixed at 1.0).

    HISTORY. Before the ``matched_per_frame`` path existed, a full-recall
    point produced ``total = 0/0 = NaN``, hence ``denom = NaN``, and
    ``np.divide(..., where=denom > 0)`` left the ``out=ones`` default in
    place because ``NaN > 0`` is False. Precision was therefore silently
    **1.0** exactly where the detector was at its most over-predicting; the
    monotone envelope then pinned the whole curve at 1.0 and AP degenerated
    to ``max(recall) - min(recall)``. That is a real measured defect, not a
    hypothetical: a 10-epoch SAM3 run scored its worst epoch 0.986.
    """
    if missed_per_frame is not None and len(missed_per_frame) != len(points):
        raise ValueError("missed_per_frame must align 1:1 with points")
    if matched_per_frame is not None and len(matched_per_frame) != len(points):
        raise ValueError("matched_per_frame must align 1:1 with points")
    indexed = sorted(enumerate(points), key=lambda ip: ip[1].recall)
    order = [i for i, _p in indexed]
    ordered = [p for _i, p in indexed]
    recalls = np.array([p.recall for p in ordered], dtype=np.float64)
    extras = np.array([p.extra_per_frame for p in ordered], dtype=np.float64)
    if matched_per_frame is not None:
        matched = np.array([matched_per_frame[i] for i in order], dtype=np.float64)
    elif missed_per_frame is not None:
        missed = np.array([missed_per_frame[i] for i in order], dtype=np.float64)
        # matched = recall * total, and total = matched + missed, so
        # total = missed / (1 - recall) when recall < 1.
        with np.errstate(divide="ignore", invalid="ignore"):
            total = np.where(recalls < 1.0, missed / (1.0 - recalls), np.nan)
        # `total` is the ground-truth count per frame -- a corpus property,
        # identical at every confidence -- so the singular (recall == 1)
        # entries take the value the non-singular ones agree on.
        finite = np.isfinite(total)
        if finite.any():
            total = np.where(finite, total, float(np.median(total[finite])))
        matched = recalls * total
    else:
        matched = recalls
    denom = matched + extras
    with np.errstate(invalid="ignore"):
        precisions = np.divide(
            matched, denom, out=np.ones_like(matched), where=denom > 0
        )
    # Never let an unrecoverable scale masquerade as perfect precision: the
    # `where=` mask above skips NaN denominators and would leave the 1.0
    # default behind. NaN is the honest answer, and `average_precision`
    # drops it.
    precisions = np.where(np.isnan(matched), np.nan, precisions)
    return recalls, precisions


# ``np.trapezoid`` is the numpy >= 2.0 spelling; ``np.trapz`` is the 1.x one
# (removed in 2.0). The GPU sidecar envs this code actually runs in are not
# guaranteed to be on numpy 2 -- courtship's ``hydra-sam3`` is numpy 1.26.4 --
# so bind whichever exists at import time rather than dying at AP time.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """PASCAL-VOC-style AP: integrate the monotone-decreasing precision
    envelope (each precision replaced by the max precision at that recall or
    higher) over recall via the trapezoidal rule.

    Recall/precision need not be pre-sorted or de-duplicated; this sorts by
    recall internally. NaN precision values (recall == 1.0 with no total
    derivable) are dropped before integrating.
    """
    r = np.asarray(recalls, dtype=np.float64)
    p = np.asarray(precisions, dtype=np.float64)
    mask = ~np.isnan(p)
    r, p = r[mask], p[mask]
    if r.size == 0:
        return 0.0
    order = np.argsort(r)
    r, p = r[order], p[order]
    # Monotone precision envelope: precision(r) := max(precision(r' >= r)).
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    if r.size == 1:
        return float(envelope[0] * r[0])
    return float(_trapezoid(envelope, r))


def average_precision_from_counts(
    counts: Sequence[SweepCounts], n_frames: int
) -> float:
    """AP from raw summed instance counts over a confidence sweep.

    The same computation ``compare_models`` performs, entered from counts
    rather than from ``CalibrationPoint``s: it builds the ``OperatingPoint``
    sweep and the aligned ``missed_per_frame`` series, then integrates.
    Returns 0.0 when the corpus has no ground truth at all -- recall is
    undefined there, and inventing a value would be worse than reporting
    that nothing was measurable.

    Reminder from the module docstring: the result is a WITHIN-RUN trend
    number, not comparable across corpora or tile geometries.
    """
    frames = max(1, int(n_frames))
    usable = [c for c in counts if (c.matched + c.missed) > 0]
    if not usable:
        return 0.0
    points = [
        OperatingPoint(
            confidence=float(c.confidence),
            recall=c.matched / (c.matched + c.missed),
            extra_per_frame=c.extras / frames,
        )
        for c in usable
    ]
    # EXACT counts, straight through: precision = matched / (matched +
    # extras) and recall = matched / (matched + missed) are both directly
    # available here, so there is nothing to invert and nothing to be
    # singular at full recall.
    matched_per_frame = [c.matched / frames for c in usable]
    recalls, precisions = precision_recall_curve(
        points, matched_per_frame=matched_per_frame
    )
    return average_precision(recalls, precisions)
