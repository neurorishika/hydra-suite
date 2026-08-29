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
* Matching is one-to-one nearest-centroid gated by containment, not IoU.
  SAM3 masks trace legs and antennae (~1.7x the labelled body-core area),
  so IoU penalises correct detections for a purely conventional reason --
  but centroid distance alone lets one blob claim two labels in a cluster.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .base import SemanticLabeler
from .tiling import (
    DEFAULT_OVERLAP,
    TILE_FRACTION_GRID,
    candidate_tile_plans,
    collect_candidates,
    merge_candidates,
)

# Refuse to recommend a threshold fitted on fewer matched instances than this.
MIN_MATCHED_INSTANCES = 20
# Recall floor a point must clear to be recommendable.
MIN_RECALL = 0.90
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


def _centroid(poly: np.ndarray) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def _contains(poly: np.ndarray, point: np.ndarray) -> bool:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def match_one_to_one(
    pred_polys: Sequence[np.ndarray], label_polys: Sequence[np.ndarray]
) -> list[tuple[int, int]]:
    """Greedy nearest-centroid pairing, each side used at most once.

    A pair is admissible only if the prediction's centroid falls inside the
    label, or the label's centroid falls inside the prediction -- the
    containment gate that stops one oversized blob from claiming its
    neighbour's label in a dense cluster.
    """
    pred_c = [_centroid(p) for p in pred_polys]
    label_c = [_centroid(g) for g in label_polys]
    pairs: list[tuple[float, int, int]] = []
    for pi, pc in enumerate(pred_c):
        for gi, gc in enumerate(label_c):
            if not (_contains(label_polys[gi], pc) or _contains(pred_polys[pi], gc)):
                continue
            pairs.append((float(np.hypot(*(pc - gc))), pi, gi))
    pairs.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[tuple[int, int]] = []
    for _dist, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi))
    return out


def calibrate(
    labeler: SemanticLabeler,
    frames: Sequence[tuple[Path, list]],
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
    """
    floor = CONFIDENCE_GRID[0]
    # fraction -> (per-frame candidates+labels, total seconds, tiles, tile_px)
    acc: dict[float | None, dict] = {}
    seen_frames = 0
    n_frames_total = max(len(frames), 1)
    for fi, (img_path, records) in enumerate(frames):
        if should_stop is not None and should_stop():
            break
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        seen_frames += 1
        h, w = image.shape[:2]
        label_polys = [
            np.asarray(r.points, dtype=np.float32).reshape(-1, 2) for r in records
        ]
        options = candidate_tile_plans(
            (h, w), reference_body_px, fractions=tile_fractions, overlap=overlap
        )
        for oi, opt in enumerate(options):
            if should_stop is not None and should_stop():
                break
            started = time.perf_counter()
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
            elapsed = time.perf_counter() - started
            entry = acc.setdefault(
                opt.fraction,
                {
                    "frames": [],
                    "seconds": 0.0,
                    "tiles": opt.tiles_per_frame,
                    "tile_px": opt.tile_px,
                    "n": 0,
                },
            )
            entry["frames"].append((candidates, label_polys))
            entry["seconds"] += elapsed
            entry["n"] += 1
            if progress is not None:
                done = fi * len(options) + oi + 1
                progress(
                    int(100 * done / (n_frames_total * max(len(options), 1))),
                    f"Calibrating frame {fi + 1}/{n_frames_total}, "
                    f"tile {opt.tile_px or 'full frame'}",
                )

    points: list[CalibrationPoint] = []
    for fraction, entry in acc.items():
        if seen_frames and entry["n"] < seen_frames:
            # Resolved on only some frames (mixed frame sizes) -- dropping it
            # is honest; reporting a partial average would not be comparable.
            continue
        n_frames = max(entry["n"], 1)
        seconds_per_frame = entry["seconds"] / n_frames
        for conf in CONFIDENCE_GRID:
            matched = missed = extra = total_labels = 0
            for candidates, label_polys in entry["frames"]:
                merged = merge_candidates(
                    candidates, confidence_threshold=conf, iou_threshold=merge_iou
                )
                preds = [m.polygon_px for m in merged]
                pairs = match_one_to_one(preds, label_polys)
                matched += len(pairs)
                missed += len(label_polys) - len(pairs)
                extra += len(preds) - len(pairs)
                total_labels += len(label_polys)
            points.append(
                CalibrationPoint(
                    tile_fraction=fraction,
                    tile_px=entry["tile_px"],
                    tiles_per_frame=entry["tiles"],
                    seconds_per_frame=seconds_per_frame,
                    confidence=conf,
                    missed_per_frame=missed / n_frames,
                    extra_per_frame=extra / n_frames,
                    recall=(matched / total_labels) if total_labels else 0.0,
                    n_matched=matched,
                )
            )
    return points


def recommend(
    points: Sequence[CalibrationPoint],
    *,
    min_matched: int = MIN_MATCHED_INSTANCES,
    min_recall: float = MIN_RECALL,
) -> tuple[CalibrationPoint | None, str]:
    """The cheapest tiling that clears the recall floor, or a refusal.

    Lexicographic, and stated as such in the UI: among points clearing
    *min_recall*, fewest ``tiles_per_frame`` wins -- inference cost over a
    whole project is roughly linear in it and a full run is hours -- with
    ties broken by the highest confidence (fewest polygons to delete).
    Deliberately not the F1 maximum.

    The ``min_matched`` floor is an ELIGIBILITY filter, not a veto on the
    winner: a cheap configuration that finds almost nothing would otherwise
    post a perfect recall on four matches and win on cost.
    """
    if not points:
        return None, "No calibration points; nothing to recommend."
    on_recall = [p for p in points if p.recall >= min_recall]
    if not on_recall:
        return None, (
            f"No configuration reached {min_recall:.0%} recall on these frames. "
            "Try a different prompt, or a finer tile fraction."
        )
    eligible = [p for p in on_recall if p.n_matched >= min_matched]
    if not eligible:
        best_matched = max(p.n_matched for p in on_recall)
        return None, (
            f"Insufficient data: the best configuration reaching "
            f"{min_recall:.0%} recall matched only {best_matched} instance(s) "
            f"across the labelled frames (need {min_matched}). The frontier "
            "below is shown for inspection, but no operating point is "
            "recommended -- label a few more frames."
        )
    return min(eligible, key=lambda p: (p.tiles_per_frame, -p.confidence)), ""
