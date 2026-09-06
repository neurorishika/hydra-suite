"""Matcher geometry shared by the direct and semantic calibration paths.

Moved out of ``core/inference/semantic/calibration.py`` (2026-09-06) so the
direct-calibration path can adopt the same matcher without importing under
``core/inference/semantic/`` -- see
``docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md``
Task 1. This is a pure move: no scoring behaviour changed. See
``semantic/calibration.py`` for the containment-gate design rationale these
functions implement (representative-point fix, quality ranking, the removed
IoU admissibility route) -- that history is preserved there, not duplicated
here.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from hydra_suite.core.inference.shape_prior import (
    MIN_MATCH_QUALITY,
    AreaBand,
    in_band,
    match_quality,
)

# `cv2.moments` m00 below this is treated as no area at all, so the area
# centroid is not computed from a near-zero denominator.
_MIN_MOMENT_AREA = 1e-9
# Raster budget for the pole-of-inaccessibility fallback, in pixels. A
# pathological polygon (a label spanning most of a 4500 px frame) must not
# turn one containment test into a 20 M px allocation; above the budget the
# fallback declines and `representative_point` drops to its last stage.
_MAX_RASTER_PX = 4_000_000


def _contains(poly: np.ndarray, point: np.ndarray) -> bool:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def _is_finite(poly: np.ndarray) -> bool:
    """True when every vertex coordinate is finite. See ``match_one_to_one``:
    ``polygon_iou`` and ``cv2`` both raise on NaN/inf, so malformed geometry
    is filtered rather than allowed to crash a calibration run."""
    return bool(np.isfinite(np.asarray(poly, dtype=np.float64)).all())


def _vertex_mean(poly: np.ndarray) -> np.ndarray:
    """Mean of the polygon's vertices. NOT inside-guaranteed -- see
    ``representative_point``; retained only as the last-resort branch and
    for the deterministic distance tie-break, where insideness is
    irrelevant."""
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def _pole_of_inaccessibility(pts: np.ndarray) -> np.ndarray | None:
    """The interior point furthest from the boundary, or None if unavailable.

    Rasterise-and-distance-transform, rather than an analytic construction:
    it is exact-enough (one pixel), needs only ``cv2``/``numpy`` (``shapely``
    is not a dependency of this project), and is correct for crescents, U and
    S curves, and self-intersecting outlines alike, because ``fillPoly``
    settles the interior question the same way ``pointPolygonTest`` does.

    DEVIATION worth stating: this rasterises in the polygon's OWN bounding
    box, not in frame space. Frames here are ~4500 px square while a labelled
    animal is ~50 px, so a frame-space raster would cost ~20 M px per polygon
    and per call. The box is also padded by one pixel so the boundary is not
    clipped by the raster edge, which would otherwise make edge pixels look
    interior to the distance transform.
    """
    if pts.shape[0] < 3 or not np.isfinite(pts).all():
        # A non-finite vertex would reach `np.round(...).astype(np.int32)` and
        # raise (ValueError / OverflowError). Declining here makes a malformed
        # polygon a NON-MATCH instead.
        #
        # Measured against the pre-fix module, NOT assumed: the old vertex-mean
        # code was silent only for a LABEL-side NaN. A prediction-side NaN
        # vertex already raised there, and +/-inf raised on both sides. So this
        # guard is strictly BETTER than the behaviour it replaces, not a
        # restoration of it -- do not describe it as one.
        return None
    lo = np.floor(pts.min(axis=0)) - 1.0
    hi = np.ceil(pts.max(axis=0)) + 1.0
    w, h = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1
    if w < 3 or h < 3 or w * h > _MAX_RASTER_PX:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = np.round(pts - lo).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [shifted], 255)
    if not mask.any():
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return np.array([float(ix) + lo[0], float(iy) + lo[1]], dtype=np.float64)


def representative_point(poly: np.ndarray) -> np.ndarray:
    """An INSIDE-GUARANTEED point standing in for *poly* in containment tests.

    Replaces a measured defect. The previous implementation returned the
    MEAN OF POLYGON VERTICES, which for a curved, elongated, densely sampled
    outline routinely falls OUTSIDE the polygon: measured on a real ant
    validation split, 128 of 805 (15.9 %) ground-truth outlines did not
    contain their own vertex-mean, so ``match_one_to_one``'s containment gate
    vetoed near-perfect masks (verified case: IoU 0.904, matching area,
    claimed by no other label -- scored a MISS). See
    docs/superpowers/specs/2026-09-05-sam3-spike-parity-measurement-findings.md

    Three stages, each VERIFIED with the same ``_contains`` predicate the gate
    itself uses, so the point and the test can never disagree:

    1. the AREA centroid (``cv2.moments``). Correct and cheap for the convex
       and mildly concave majority. It is *not* sufficient alone -- for a
       crescent or a U the area centroid also lies outside -- which is why it
       is verified rather than trusted.
    2. the pole of inaccessibility (see ``_pole_of_inaccessibility``): the
       interior point maximising distance to the boundary. Chosen over a
       scanline midpoint because it is the most robustly interior point
       available, so it degrades gracefully on thin arcs, and because it is
       stable under vertex resampling -- a scanline midpoint jumps between
       lobes when the sampling changes, which would make matching depend on
       how densely a user's annotation tool emitted points.
    3. the first vertex, for shapes with no interior at all (zero area, a
       duplicate-vertex degenerate, a sub-pixel-thin sliver). ``_contains``
       accepts on-boundary points (``>= 0``), so this is still "inside" by
       the gate's own definition and never returns a point the gate would
       then veto.
    """
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros(2, dtype=np.float64)
    if not np.isfinite(pts).all():
        # Same contract as `_pole_of_inaccessibility`'s guard: a polygon with
        # a NaN/inf vertex has no meaningful interior, so hand back the plain
        # vertex mean (itself non-finite) and let the containment test fail
        # the pair, exactly as it did before this function existed. Never
        # raise: this is reachable from GUI calibration on malformed labels.
        return _vertex_mean(pts)
    if pts.shape[0] >= 3:
        m = cv2.moments(pts.astype(np.float32).reshape(-1, 1, 2))
        if abs(m["m00"]) > _MIN_MOMENT_AREA:
            centroid = np.array(
                [m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float64
            )
            if _contains(pts, centroid):
                return centroid
        pole = _pole_of_inaccessibility(pts)
        if pole is not None and _contains(pts, pole):
            return pole
    return pts[0].copy()


def match_one_to_one(
    pred_polys: Sequence[np.ndarray],
    label_polys: Sequence[np.ndarray],
    *,
    area_band: AreaBand | None = None,
    min_quality: float = MIN_MATCH_QUALITY,
    return_admissible: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Greedy one-to-one pairing by descending match QUALITY.

    A pair is admissible when it clears the area band and ``min_quality``
    and is CONTAINED -- the prediction's representative point falls inside
    the label, or the label's inside the prediction. Containment is what
    stops one oversized blob from claiming its neighbour's label in a dense
    cluster, so the gate is kept, not deleted.

    What was FIXED is the point it tests. It used to be the mean of polygon
    vertices, which lies outside 15.9 % (128/805) of real ant outlines and so
    vetoed near-perfect masks; it is now ``representative_point``, which is
    inside by construction. On the held-out split, predictions held fixed,
    that change alone moved recall 0.867 -> 0.988 and extras/tile
    0.203 -> 0.035.

    An IoU second admissibility route was tried and then REMOVED after it was
    measured -- read the comment at the containment test in the body before
    re-adding one.

    Ranking by quality rather than by centroid distance also fixes cluster
    pairing: the nearest centroid is not always the better fit, and a
    distance-first greedy pass can hand a prediction to the wrong label and
    strand the right one.

    Measured effect of the two changes together, predictions held fixed:
    see ``tools/sam3_parity/matcher_gate_results.json`` and
    docs/superpowers/specs/2026-09-05-sam3-spike-parity-measurement-findings.md

    ``return_admissible`` additionally returns every ADMISSIBLE pair (the
    pairs that cleared the band, containment and ``min_quality``), not only
    the greedy one-to-one winners. The direct-calibration path needs this to
    count cross-tile DUPLICATES -- a prediction that was a legitimate
    candidate for some label but lost the one-to-one race. Exposing the
    matcher's own admissibility set is deliberate: recomposing band +
    containment + quality at the call site would fork the definition, which
    is the exact failure this module was extracted to prevent. The default
    return value is unchanged.
    """
    pred_c = [representative_point(p) for p in pred_polys]
    label_c = [representative_point(g) for g in label_polys]
    # The tie-break below is a DISTANCE, not a containment test, so the plain
    # vertex mean is fine for it and is kept: it is cheap, and changing the
    # tie-break would perturb pairings for no stated reason.
    pred_m = [_vertex_mean(p) for p in pred_polys]
    label_m = [_vertex_mean(g) for g in label_polys]
    # Non-finite vertices are dropped up front, on BOTH sides. This is KEPT
    # after the IoU route's removal on purpose: it is correct defensive
    # behaviour. Note the pre-fix code was NOT uniformly silent on malformed
    # geometry -- measured, it raised for a prediction-side NaN and for +/-inf
    # on either side, and was silent only for a label-side NaN. Malformed geometry had TWO crash entrances, not one --
    # `cv2` raises inside the representative point, and `polygon_iou` raises
    # independently (`utils/polygon_iou.py:59`,
    # "cannot convert float NaN to integer"), which the retired IoU route
    # reached for pairs containment never scored. `match_quality` still calls
    # `polygon_iou` for every admissible pair, so that second entrance is
    # live regardless. A polygon with no finite geometry cannot be matched to
    # anything, which is exactly what dropping it means.
    finite_label = [_is_finite(g) for g in label_polys]
    admissible = [
        i for i, p in enumerate(pred_polys) if _is_finite(p) and in_band(p, area_band)
    ]
    pairs: list[tuple[float, int, int]] = []
    for pi in admissible:
        pc = pred_c[pi]
        for gi, gc in enumerate(label_c):
            if not finite_label[gi]:
                continue
            # A SECOND admissibility route lived here and was REMOVED after
            # measurement: `or polygon_iou(pred, label) >= 0.5`, added
            # alongside the point fix. Measured against it on the 16 held-out
            # frames, predictions held fixed (`--score-only` over the cached
            # candidates on courtship; see
            # tools/sam3_parity/matcher_gate_results.json): containment alone,
            # under `representative_point`, scored recall 0.9876 -- with the
            # IoU route, 0.9888. ONE instance in 805.
            #
            # Deleted because its safety rested entirely on the threshold
            # being exactly 0.5: IoU >= 0.5 implies the two areas are within
            # 2x of each other, and that is what makes a region spanning two
            # labels geometrically unable to use the route at all. At 0.3 the
            # bound becomes 3.3x and a two-label blob gets in. A knob that is
            # safe at exactly one value, that nobody has a reason to tune, and
            # that buys 0.1 % recall is a liability, not a feature.
            #
            # Its original justification was also unsound, which is worth
            # recording so it is not rediscovered: it rested on an
            # AREA-CENTROID measurement (recall 0.929) that never transferred,
            # because `representative_point` is materially stronger than an
            # area centroid -- 0/805 of these labels fail to contain it,
            # against 128/805 for the vertex mean. Re-add a route like this
            # only with a measurement that actually says it is needed.
            if not (_contains(label_polys[gi], pc) or _contains(pred_polys[pi], gc)):
                continue
            # Only AFTER admissibility: `match_quality` is the expensive term
            # (it computes IoU, area and minAreaRect), and the overwhelming
            # majority of pred x label pairs are inadmissible.
            quality = match_quality(pred_polys[pi], label_polys[gi])
            if quality < min_quality:
                continue
            # Negated so a plain ascending sort puts the BEST pair first,
            # with the centroid distance as a deterministic tie-break.
            pairs.append((-quality, pi, gi))
    pairs.sort(key=lambda t: (t[0], float(np.hypot(*(pred_m[t[1]] - label_m[t[2]])))))
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[tuple[int, int]] = []
    for _neg_q, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi))
    if return_admissible:
        return out, [(pi, gi) for _neg_q, pi, gi in pairs]
    return out


# Public alias: callers outside this module should use `contains`, not the
# underscore-prefixed name kept for the semantic-package re-export.
contains = _contains
