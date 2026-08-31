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
    candidate_tile_plans,
    collect_candidates,
    merge_candidates,
)

# Refuse to recommend a threshold fitted on fewer matched instances than this.
MIN_MATCHED_INSTANCES = 20
# Recall floor a point must clear to be recommendable.
MIN_RECALL = 0.90
# Mean match quality a point must clear to be recommendable. An ELIGIBILITY
# filter in the same spirit as MIN_MATCHED_INSTANCES, not a new objective:
# recall bought with mistargeted masks is not recall.
MIN_MEAN_QUALITY = 0.35
CONFIDENCE_GRID: tuple[float, ...] = tuple(
    round(float(c), 2) for c in np.arange(0.05, 0.96, 0.05)
)


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


def _centroid(poly: np.ndarray) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def _contains(poly: np.ndarray, point: np.ndarray) -> bool:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def match_one_to_one(
    pred_polys: Sequence[np.ndarray],
    label_polys: Sequence[np.ndarray],
    *,
    area_band: AreaBand | None = None,
    min_quality: float = MIN_MATCH_QUALITY,
) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing by descending match QUALITY.

    Three conditions make a pair admissible:

    * containment -- the prediction's centroid falls inside the label, or
      the label's centroid inside the prediction. Stops one oversized blob
      from claiming its neighbour's label in a dense cluster.
    * the area band, when one is supplied -- see ``shape_prior``.
    * ``min_quality`` -- a floor on the graded score, so a pair that is
      technically admissible but plainly not the same object (a mask ~40x
      the label's area still contains its centroid) is not counted a find.

    Ranking by quality rather than by centroid distance also fixes cluster
    pairing: the nearest centroid is not always the better fit, and a
    distance-first greedy pass can hand a prediction to the wrong label and
    strand the right one.
    """
    pred_c = [_centroid(p) for p in pred_polys]
    label_c = [_centroid(g) for g in label_polys]
    admissible = [i for i, p in enumerate(pred_polys) if in_band(p, area_band)]
    pairs: list[tuple[float, int, int]] = []
    for pi in admissible:
        pc = pred_c[pi]
        for gi, gc in enumerate(label_c):
            if not (_contains(label_polys[gi], pc) or _contains(pred_polys[pi], gc)):
                continue
            quality = match_quality(pred_polys[pi], label_polys[gi])
            if quality < min_quality:
                continue
            # Negated so a plain ascending sort puts the BEST pair first,
            # with the centroid distance as a deterministic tie-break.
            pairs.append((-quality, pi, gi))
    pairs.sort(key=lambda t: (t[0], float(np.hypot(*(pred_c[t[1]] - label_c[t[2]])))))
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
    n_frames_total = max(len(frames), 1)

    # Precompute per-frame tile-plan options before running any inference, so
    # the total inference-pass count -- and hence the progress percentage --
    # is known upfront. Frame dimensions (not just reference_body_px) affect
    # tiles_per_frame, so this cannot be estimated from the first frame alone
    # without either overshooting or undershooting 100% under mixed sizes.
    #
    # Only the frame's PATH and dimensions are retained here, never the
    # decoded pixels: a 4512^2 BGR frame is ~61 MB, so holding a 50-frame
    # labelled set decoded would cost ~3 GB before a single inference pass.
    # The image is re-read inside the fraction loop below, where exactly one
    # frame is resident at a time.
    per_frame: list[tuple[Path, tuple[int, int], list, tuple, list]] = []
    total_passes = 0
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
        total_passes += len(options)
    total_passes = max(total_passes, 1)

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

    done_passes = 0
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
            done_passes += 1
            completed_frames.add(fi)
            if progress is not None:
                progress(
                    min(100, int(100 * done_passes / total_passes)),
                    f"Calibrating frame {fi + 1}/{n_frames_total}, "
                    f"tile {opt.tile_px or 'full frame'}",
                )
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
