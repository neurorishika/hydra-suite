"""A size-and-shape prior fitted from the user's own labels.

Why this exists: calibration used to admit a prediction/label pair on
CONTAINMENT alone. An arena-sized blob contains dozens of label centroids,
so it earned recall credit for one of them; a leg-sized fragment inside a
label earned credit too. Because ``recommend`` is recall-first, calibration
therefore actively SELECTED for the configurations that produce blobs and
subparts. Nothing downstream filtered by size either.

Two things live here, and the split is deliberate:

* ``AreaBand`` -- a hard admissibility gate, derived from the labelled
  frames. It is purely geometric, so it is threshold-INDEPENDENT and can be
  applied both during calibration and at inference/re-threshold time
  against cached candidates.
* ``match_quality`` -- a graded score over overlap, area agreement and
  aspect agreement. It needs a ground-truth partner, so it exists at
  CALIBRATION time only. That asymmetry is intentional: at inference there
  is nothing to compare against.

Overlap enters as a SCORE, never as a hard threshold. SAM3 masks trace legs
and antennae at ~1.7x the labelled body-core area, so IoU-gating would
reject correct detections for a purely conventional reason -- but a
systematic IoU offset shifts every configuration equally and so does not
distort the ranking a score is used for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from hydra_suite.core.inference.masks import polygon_iou

# Band = [LOW x median label area, HIGH x median label area], then widened
# to contain every observed label (see fit_area_band). HIGH is generous on
# purpose: it must clear the ~1.7x appendage overshoot with room to spare,
# while still rejecting the multi-animal merges and arena chunks that are
# an order of magnitude out.
LOW_MULTIPLIER = 0.3
HIGH_MULTIPLIER = 3.5
# Slack applied when widening to the observed extremes, so a label sitting
# exactly at the boundary is not admitted by a hair.
OBSERVED_SLACK_LOW = 0.8
OBSERVED_SLACK_HIGH = 1.25
# A pair scoring below this is garbage even though it is admissible; it must
# not be counted as a find.
MIN_MATCH_QUALITY = 0.1


@dataclass(frozen=True)
class AreaBand:
    """Admissible polygon-area range, in square pixels of frame space."""

    min_px2: float
    max_px2: float
    median_px2: float
    n_labels: int


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a (P, 2) polygon; 0.0 for degenerate input."""
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def fit_area_band(label_polys: Sequence[np.ndarray]) -> AreaBand | None:
    """Fit the admissible area range to *label_polys*. None if unfittable.

    The multiplier band is widened until it contains every label with slack.
    That invariant -- no ground-truth label is ever out of band -- is what
    makes a HARD gate safe: without it, a dataset with one unusually large
    animal would have calibration score the user's own labels as impossible.
    """
    areas = [a for a in (polygon_area(p) for p in label_polys) if a > 0.0]
    if not areas:
        return None
    arr = np.asarray(areas, dtype=np.float64)
    median = float(np.median(arr))
    lo = min(LOW_MULTIPLIER * median, OBSERVED_SLACK_LOW * float(arr.min()))
    hi = max(HIGH_MULTIPLIER * median, OBSERVED_SLACK_HIGH * float(arr.max()))
    return AreaBand(min_px2=lo, max_px2=hi, median_px2=median, n_labels=int(arr.size))


def in_band(poly: np.ndarray, band: AreaBand | None) -> bool:
    """True if *poly*'s area is admissible. A None band admits everything."""
    if band is None:
        return True
    area = polygon_area(poly)
    return band.min_px2 <= area <= band.max_px2


def aspect_ratio(poly: np.ndarray) -> float:
    """Elongation (long edge / short edge) of the minimum-area rectangle.

    Rotation- and scale-invariant, and >= 1 by construction, so two shapes
    can be compared by a plain min/max ratio.
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    if pts.shape[0] < 3:
        return 1.0
    _c, (w, h), _a = cv2.minAreaRect(pts)
    long_edge, short_edge = max(float(w), float(h)), min(float(w), float(h))
    if short_edge <= 0.0:
        return float("inf")
    return long_edge / short_edge


def _ratio(a: float, b: float) -> float:
    """Symmetric agreement of two positive quantities, in [0, 1]."""
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
        return 0.0
    return float(min(a, b) / max(a, b))


def match_quality(pred_poly: np.ndarray, label_poly: np.ndarray) -> float:
    """How well *pred_poly* targets *label_poly*, in [0, 1].

    The geometric mean of three agreements -- overlap (IoU), area, and
    aspect ratio. The geometric mean, not the arithmetic one: any single
    term collapsing to zero must collapse the score, because a prediction
    that overlaps perfectly but is 50x the area is not half-right.
    """
    overlap = polygon_iou(pred_poly, label_poly)
    if overlap <= 0.0:
        return 0.0
    area = _ratio(polygon_area(pred_poly), polygon_area(label_poly))
    shape = _ratio(aspect_ratio(pred_poly), aspect_ratio(label_poly))
    return float((overlap * area * shape) ** (1.0 / 3.0))
