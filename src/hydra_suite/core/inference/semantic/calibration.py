"""Fit the operating point to the user's own labelled frames.

Three design commitments:

* TWO parameters are fitted, tile fraction AND confidence. There is no
  defensible dev-side value for the tile fraction: the seed in ``tiling``
  was back-derived from one measured configuration on one dataset. Tile
  geometry is baked into the candidates, so it costs one inference pass per
  fraction (outer loop); confidence is swept offline from the cache (inner).
* The objective is the MISSED-vs-TO-DELETE frontier, not F1. Deleting a
  spurious polygon is one click; a missed animal must be found by eye. The
  F1-optimal threshold missed 4.7 animals/frame where a recall-first one
  missed 1.0.
* Matching is one-to-one, gated by containment AND by a size/shape prior
  fitted to the user's own labels (``shape_prior``), and ranked by a graded
  quality score rather than by centroid distance. The gate is the fix for a
  real mistargeting bug: containment alone let an arena-sized blob or a
  leg-sized fragment earn recall credit, and since ``recommend`` is
  recall-first, calibration then SELECTED for whatever produced them. IoU
  is still not a hard gate -- SAM3 masks trace legs and antennae at ~1.7x
  the labelled body-core area -- but it enters the quality score, where a
  systematic offset shifts every configuration equally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from hydra_suite.core.inference.masks import polygon_iou
from hydra_suite.data.al.escalation import LabelRecord

from .base import SemanticLabeler
from .shape_prior import (
    MIN_MATCH_QUALITY,
    AreaBand,
    fit_area_band,
    in_band,
    match_quality,
    polygon_area,
)
from .tiling import (
    DEFAULT_OVERLAP,
    TILE_FRACTION_GRID,
    TileCandidate,
    TileCollectionCancelled,
    TileProgressReporter,
    candidate_tile_plans,
    collect_candidates,
    merge_candidates,
)

# Refuse to recommend a threshold fitted on fewer matched instances than this.
MIN_MATCHED_INSTANCES = 20
# Recall floor a point must clear to be recommendable. It was previously
# annotated as possibly unreachable for SCORING rather than detection
# reasons; the matcher defect responsible has now been fixed (see
# `representative_point` and `match_one_to_one` below) and the before/after
# measurement is in tools/sam3_parity/matcher_gate_results.json. The value is
# UNCHANGED: a measurement is not a reason to move a product requirement, and
# baking one into a constant is exactly the coupling that hid the defect.
MIN_RECALL = 0.90
# Mean match quality a point must clear to be recommendable. An ELIGIBILITY
# filter in the same spirit as MIN_MATCHED_INSTANCES, not a new objective:
# recall bought with mistargeted masks is not recall.
MIN_MEAN_QUALITY = 0.35
CONFIDENCE_GRID: tuple[float, ...] = tuple(
    round(float(c), 2) for c in np.arange(0.05, 0.96, 0.05)
)
# IoU at which a pair is admissible REGARDLESS of containment -- see
# `match_one_to_one`. Not a hard gate (nothing is rejected for being below
# it); a second, independent route to admissibility.
ADMISSIBLE_IOU = 0.5
# `cv2.moments` m00 below this is treated as no area at all, so the area
# centroid is not computed from a near-zero denominator.
_MIN_MOMENT_AREA = 1e-9
# Raster budget for the pole-of-inaccessibility fallback, in pixels. A
# pathological polygon (a label spanning most of a 4500 px frame) must not
# turn one containment test into a 20 M px allocation; above the budget the
# fallback declines and `representative_point` drops to its last stage.
_MAX_RASTER_PX = 4_000_000


@dataclass(frozen=True)
class CalibrationPoint:
    """One (tile fraction, confidence) cell of the calibration frontier."""

    tile_fraction: float | None  # None = full frame, no tiling
    tile_px: int | None
    tiles_per_frame: int
    seconds_per_frame: float  # MEASURED on this machine and this data
    confidence: float
    missed_per_frame: float
    extra_per_frame: float
    recall: float
    n_matched: int
    # The label-derived area gate this point was scored under. Persisted
    # with the operating point so inference applies the SAME gate.
    area_min_px2: float = 0.0
    area_max_px2: float = 0.0
    # Graded match quality over the matched pairs (0 when nothing matched).
    mean_quality: float = 0.0
    median_iou: float = 0.0
    median_area_ratio: float = 0.0


@dataclass(frozen=True)
class CalibrationGroundTruth:
    """One labelled polygon retained for visual calibration inspection."""

    class_id: int
    polygon_px: np.ndarray


@dataclass(frozen=True)
class CalibrationPreviewFrame:
    """Reusable inference evidence for one labelled calibration frame.

    Candidates are retained at the sweep's confidence floor and grouped by
    tile fraction.  The results dialog can therefore render any table row by
    re-thresholding and merging this cache; selecting rows never reruns SAM3.
    """

    image_path: Path
    ground_truth: tuple[CalibrationGroundTruth, ...]
    candidates_by_fraction: dict[float | None, tuple[TileCandidate, ...]]


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
        # raise (ValueError / OverflowError). Declining here keeps a malformed
        # polygon a NON-MATCH, which is what the previous vertex-mean
        # implementation did silently -- introducing a hard crash on a
        # GUI-reachable path would be a regression, not a fix.
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
) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing by descending match QUALITY.

    A pair is admissible when it clears the area band and ``min_quality``
    AND takes either of two independent routes:

    * **containment** -- the prediction's representative point falls inside
      the label, or the label's inside the prediction. This is what stops one
      oversized blob from claiming its neighbour's label in a dense cluster,
      so it is kept, not deleted. What was FIXED is the point it tests: it
      used to be the mean of polygon vertices, which lies outside 15.9 %
      (128/805) of real ant outlines and so vetoed near-perfect masks. It is
      now ``representative_point``, which is inside by construction.
    * **overlap** -- IoU >= ``ADMISSIBLE_IOU``. Added as a SECOND route
      rather than a replacement. The motivating evidence is INDIRECT and the
      limit is worth stating: on the held-out split an AREA-CENTROID-only
      gate reached recall 0.929 where a plain IoU >= 0.5 rule reached 0.962.
      ``representative_point`` is strictly stronger than an area centroid
      (0/805 of those labels fail to contain it, against 128/805 for the
      vertex mean), so that 0.929 is a LOWER bound on what containment alone
      would now achieve, not a measurement of it. See
      ``tools/sam3_parity/matcher_gate_results.json`` for the arm that
      measures containment-only under the shipped point directly. The
      mechanism the route covers is real geometry, not a bug -- SAM3 traces
      legs and antennae, so a correct mask's interior point can sit outside a
      body-core quad and vice versa, at IoU 0.9. Note what this does NOT do:
      it never REJECTS a pair for low IoU, so ``shape_prior``'s commitment
      that overlap enters as a score and not as a hard threshold stands. It
      only widens admissibility, and widening is safe here because the
      anti-blob protection comes from the area band and ``min_quality``, not
      from containment being narrow -- a blob at IoU >= 0.5 with a label is
      not a blob.

    Ranking by quality rather than by centroid distance also fixes cluster
    pairing: the nearest centroid is not always the better fit, and a
    distance-first greedy pass can hand a prediction to the wrong label and
    strand the right one.

    Measured effect of the two changes together, predictions held fixed:
    see ``tools/sam3_parity/matcher_gate_results.json`` and
    docs/superpowers/specs/2026-09-05-sam3-spike-parity-measurement-findings.md
    """
    pred_c = [representative_point(p) for p in pred_polys]
    label_c = [representative_point(g) for g in label_polys]
    # The tie-break below is a DISTANCE, not a containment test, so the plain
    # vertex mean is fine for it and is kept: it is cheap, and changing the
    # tie-break would perturb pairings for no stated reason.
    pred_m = [_vertex_mean(p) for p in pred_polys]
    label_m = [_vertex_mean(g) for g in label_polys]
    # Non-finite vertices are dropped up front, on BOTH sides. `polygon_iou`
    # and `cv2` both raise on NaN/inf, and the IoU route below reaches
    # `polygon_iou` for pairs the old containment-only gate never scored -- so
    # without this a malformed label would turn a silent non-match into a hard
    # crash on a GUI-reachable path. A polygon with no finite geometry cannot
    # be matched to anything, which is exactly what dropping it means.
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
            contained = _contains(label_polys[gi], pc) or _contains(pred_polys[pi], gc)
            if (
                not contained
                and polygon_iou(pred_polys[pi], label_polys[gi]) < ADMISSIBLE_IOU
            ):
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
    return out


def calibrate(
    labeler: SemanticLabeler,
    frames: Sequence[tuple[Path, list[LabelRecord]]],
    prompt: str,
    *,
    reference_body_px: float | None,
    tile_fractions: Sequence[float | None] = TILE_FRACTION_GRID,
    overlap: float = DEFAULT_OVERLAP,
    seam_margin_px: float,
    merge_iou: float,
    max_instances: int = 0,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    preview_sink: Callable[[list[CalibrationPreviewFrame]], None] | None = None,
) -> list[CalibrationPoint]:
    """Sweep tile fraction x confidence against *frames*' existing labels.

    One inference pass per (frame, tile fraction); the confidence grid is
    then swept offline by re-merging that pass's cached candidates. Wall
    time per fraction is measured and reported, because it is the only
    run-time projection the UI is permitted to show.

    ``frames`` carries image paths and ``LabelRecord``s (not an app-layer
    source type) so this module stays inside the Core -> Data direction.
    Frames of differing size are handled by keeping only fractions that
    resolved on EVERY frame -- a fraction measured on a subset would have
    incomparable cost and error rates.

    When ``preview_sink`` is supplied it receives the same low-threshold
    candidates used by the metric sweep, grouped by labelled frame and tile
    fraction. This adds no model calls and retains no decoded image arrays.
    """
    floor = CONFIDENCE_GRID[0]
    # fraction -> (per-frame candidates+labels, total seconds, total tiles, tile_px)
    acc: dict[float | None, dict] = {}
    # Precompute per-frame tile-plan options before running any inference, so
    # the total tile count -- and hence the progress percentage and ETA -- is
    # known upfront. Frame dimensions (not just reference_body_px) affect
    # tiles_per_frame, so this cannot be estimated from the first frame alone
    # without either overshooting or undershooting 100% under mixed sizes.
    #
    # Only the frame's PATH and dimensions are retained here, never the
    # decoded pixels: a 4512^2 BGR frame is ~61 MB, so holding a 50-frame
    # labelled set decoded would cost ~3 GB before a single inference pass.
    # The image is re-read inside the fraction loop below, where exactly one
    # frame is resident at a time.
    if progress is not None:
        progress(0, "Planning calibration tile work…")
    per_frame: list[tuple[Path, tuple[int, int], list, tuple, list]] = []
    total_tiles = 0
    for img_path, records in frames:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        del image
        ground_truth = tuple(
            CalibrationGroundTruth(
                class_id=int(r.class_id),
                polygon_px=np.asarray(r.points, dtype=np.float32).reshape(-1, 2),
            )
            for r in records
        )
        label_polys = [item.polygon_px for item in ground_truth]
        options = candidate_tile_plans(
            (h, w), reference_body_px, fractions=tile_fractions, overlap=overlap
        )
        per_frame.append((Path(img_path), (h, w), label_polys, ground_truth, options))
        total_tiles += sum(option.tiles_per_frame for option in options)
    tile_progress = TileProgressReporter(total_tiles)

    # One band for the whole sweep, pooled over every labelled frame: the
    # gate must not differ between the configurations being compared, or the
    # frontier's rows would not be comparable. None (no labels) leaves every
    # prediction admissible, exactly as before this gate existed.
    area_band = fit_area_band(
        [
            g
            for _p, _hw, label_polys, _ground_truth, _o in per_frame
            for g in label_polys
        ]
    )

    done_tiles = 0
    cancelled = False
    completed_frames: set[int] = set()
    for fi, (img_path, _hw, label_polys, ground_truth, options) in enumerate(per_frame):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        if not options:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        for opt in options:
            if should_stop is not None and should_stop():
                cancelled = True
                break
            started = time.perf_counter()
            tiles_before_pass = done_tiles

            def _report_tile(
                done: int,
                total: int,
                *,
                tiles_before_pass: int = tiles_before_pass,
                frame_number: int = fi + 1,
                tile_px: int | None = opt.tile_px,
            ) -> None:
                if progress is None:
                    return
                pct, message = tile_progress.report(
                    tiles_before_pass + done,
                    f"Calibrating frame {frame_number}/{len(per_frame)}, "
                    f"tile {done}/{total} ({tile_px or 'full frame'})",
                )
                progress(pct, message)

            try:
                candidates = collect_candidates(
                    labeler,
                    image,
                    opt.plan,
                    prompt,
                    confidence_threshold=floor,
                    max_instances=max_instances,
                    seam_margin_px=seam_margin_px,
                    should_stop=should_stop,
                    progress=_report_tile,
                )
            except TileCollectionCancelled:
                # F6: a cancelled frame is not a measured frame. Appending it
                # (and bumping entry["n"]) let a cancel on the LAST frame slip
                # past the `entry["n"] < seen_frames` completeness filter, so
                # a fraction whose final frame was only part-inferred was
                # reported with full standing -- understating its misses in
                # exactly the frontier the user chooses an operating point
                # from.
                cancelled = True
                break
            elapsed = time.perf_counter() - started
            entry = acc.setdefault(
                opt.fraction,
                {
                    "frames": [],
                    "seconds": 0.0,
                    "tiles_total": 0,
                    "tile_px": opt.tile_px,
                    "n": 0,
                },
            )
            entry["frames"].append((img_path, candidates, ground_truth))
            entry["seconds"] += elapsed
            # tile_px is genuinely constant across frames for a given
            # fraction (it depends only on reference_body_px and the
            # fraction), so capturing it once above is safe -- but
            # tiles_per_frame depends on frame dimensions too, so it is
            # accumulated here and averaged below rather than captured once.
            entry["tiles_total"] += opt.tiles_per_frame
            entry["n"] += 1
            done_tiles += opt.tiles_per_frame
            completed_frames.add(fi)
        del image
        if cancelled:
            break

    # F6: only frames that ran to completion for at least one fraction count
    # towards the completeness denominator, so a cancel cannot inflate a
    # fraction's standing.
    seen_frames = len(completed_frames)
    points: list[CalibrationPoint] = []
    for fraction, entry in acc.items():
        if seen_frames and entry["n"] < seen_frames:
            # Resolved on only some frames (mixed frame sizes) -- dropping it
            # is honest; reporting a partial average would not be comparable.
            continue
        n_frames = max(entry["n"], 1)
        seconds_per_frame = entry["seconds"] / n_frames
        tiles_per_frame = round(entry["tiles_total"] / n_frames)
        for conf in CONFIDENCE_GRID:
            matched = missed = extra = total_labels = 0
            qualities: list[float] = []
            ious: list[float] = []
            area_ratios: list[float] = []
            for _img_path, candidates, ground_truth in entry["frames"]:
                label_polys = [item.polygon_px for item in ground_truth]
                merged = merge_candidates(
                    candidates,
                    confidence_threshold=conf,
                    iou_threshold=merge_iou,
                    area_band=area_band,
                )
                preds = [m.polygon_px for m in merged]
                pairs = match_one_to_one(preds, label_polys, area_band=area_band)
                matched += len(pairs)
                missed += len(label_polys) - len(pairs)
                extra += len(preds) - len(pairs)
                total_labels += len(label_polys)
                for pi, gi in pairs:
                    qualities.append(match_quality(preds[pi], label_polys[gi]))
                    ious.append(polygon_iou(preds[pi], label_polys[gi]))
                    pa = polygon_area(preds[pi])
                    ga = polygon_area(label_polys[gi])
                    if pa > 0 and ga > 0:
                        area_ratios.append(min(pa, ga) / max(pa, ga))
            points.append(
                CalibrationPoint(
                    tile_fraction=fraction,
                    tile_px=entry["tile_px"],
                    tiles_per_frame=tiles_per_frame,
                    seconds_per_frame=seconds_per_frame,
                    confidence=conf,
                    missed_per_frame=missed / n_frames,
                    extra_per_frame=extra / n_frames,
                    recall=(matched / total_labels) if total_labels else 0.0,
                    n_matched=matched,
                    area_min_px2=(area_band.min_px2 if area_band else 0.0),
                    area_max_px2=(area_band.max_px2 if area_band else 0.0),
                    mean_quality=(float(np.mean(qualities)) if qualities else 0.0),
                    median_iou=(float(np.median(ious)) if ious else 0.0),
                    median_area_ratio=(
                        float(np.median(area_ratios)) if area_ratios else 0.0
                    ),
                )
            )

    if preview_sink is not None:
        by_path: dict[Path, CalibrationPreviewFrame] = {}
        for fraction, entry in acc.items():
            if seen_frames and entry["n"] < seen_frames:
                continue
            for img_path, candidates, ground_truth in entry["frames"]:
                frame = by_path.get(img_path)
                if frame is None:
                    frame = CalibrationPreviewFrame(
                        image_path=img_path,
                        ground_truth=ground_truth,
                        candidates_by_fraction={},
                    )
                    by_path[img_path] = frame
                frame.candidates_by_fraction[fraction] = tuple(candidates)
        preview_sink(list(by_path.values()))
    return points


def recommend(
    points: Sequence[CalibrationPoint],
    *,
    min_matched: int = MIN_MATCHED_INSTANCES,
    min_recall: float = MIN_RECALL,
    min_quality: float = MIN_MEAN_QUALITY,
) -> tuple[CalibrationPoint | None, str]:
    """The cheapest tiling that clears the recall floor, or a refusal.

    Lexicographic, and stated as such in the UI: among points clearing
    *min_recall*, fewest ``tiles_per_frame`` wins -- inference cost over a
    whole project is roughly linear in it and a full run is hours -- with
    ties broken by the highest confidence (fewest polygons to delete).
    Deliberately not the F1 maximum.

    The ``min_matched`` and ``min_quality`` floors are ELIGIBILITY filters,
    not vetoes on the winner: a cheap configuration that finds almost
    nothing would otherwise post a perfect recall on four matches and win on
    cost, and one whose "finds" are mistargeted masks would post a high
    recall it has not earned.
    """
    if not points:
        return None, "No calibration points; nothing to recommend."
    on_recall = [p for p in points if p.recall >= min_recall]
    if not on_recall:
        return None, (
            f"No configuration reached {min_recall:.0%} recall on these frames. "
            "Try a different prompt, or a finer tile fraction."
        )
    on_quality = [p for p in on_recall if p.mean_quality >= min_quality]
    if not on_quality:
        best_quality = max(p.mean_quality for p in on_recall)
        return None, (
            f"Mistargeted: every configuration reaching {min_recall:.0%} recall "
            f"did so with masks that match the labels poorly (best mean "
            f"quality {best_quality:.2f}, need {min_quality:.2f}). The masks "
            "are probably covering the wrong thing -- whole regions, or parts "
            "of an animal. Try a more specific prompt, or a different tile "
            "fraction."
        )
    eligible = [p for p in on_quality if p.n_matched >= min_matched]
    if not eligible:
        best_matched = max(p.n_matched for p in on_quality)
        return None, (
            f"Insufficient data: the best configuration reaching "
            f"{min_recall:.0%} recall matched only {best_matched} instance(s) "
            f"across the labelled frames (need {min_matched}). The frontier "
            "below is shown for inspection, but no operating point is "
            "recommended -- label a few more frames."
        )
    return min(eligible, key=lambda p: (p.tiles_per_frame, -p.confidence)), ""
