"""Fit SAM2 geometry escalation's tile fraction to the user's polygon labels.

The ground truth is the user-reviewed polygon labels a project already has.
Each polygon is reduced to the oriented box a box-only source would carry,
SAM2 is prompted with that box exactly as escalation prompts it, and the
resulting mask is scored against the polygon it came from.

Three things differ from SAM3 calibration (``semantic/calibration.py``):

* Pairing is one-to-one BY CONSTRUCTION -- one box in, one mask out -- so
  there is no matcher and no false-positive count.
* The sweep is ONE-dimensional (tile fraction). SAM2 has no confidence
  threshold to sweep; every box is segmented.
* Cost is OWNED tiles per frame: tiles that hold no box are never encoded,
  so the cost of a fraction depends on where the animals are, not only on
  the grid.

``IOU_FLOOR`` and ``FALLBACK_CEIL`` are measured, not chosen: see the
"Measurement" section of
``docs/superpowers/specs/2026-10-03-geometry-escalation-sahi-calibration-design.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from hydra_suite.core.inference.masks import (
    clip_mask_to_polygon,
    mask_to_contour,
    polygon_iou,
)
from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_OVERLAP,
    TILE_FRACTION_GRID,
    candidate_tile_plans,
)

from .tiling import segment_boxes

# PROVISIONAL until the mehek measurement replaces them (plan Task 11).
IOU_FLOOR = 0.5
FALLBACK_CEIL = 0.10
# Refuse to recommend from fewer scored instances than this.
MIN_INSTANCES = 20


@dataclass(frozen=True)
class GeometryCalibrationPoint:
    """One tile fraction, measured against polygon ground truth."""

    tile_fraction: float | None  # None = full frame
    tile_px: int | None
    owned_tiles_per_frame: float  # encoder passes per frame (the cost axis)
    seconds_per_frame: float  # MEASURED on this machine and this data
    median_iou: float  # fallbacks count as IoU 0
    p10_iou: float
    fallback_rate: float  # masks that produced no contour
    seam_fallback_rate: float  # boxes no tile contained (full-frame pass)
    n_instances: int


@dataclass(frozen=True)
class _Prompt:
    box_xyxy: tuple[float, float, float, float]
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]


def _overlaps(a, b) -> bool:
    return (
        min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0
    )


def prompts_from_ground_truth(
    polygons: Sequence[np.ndarray],
) -> tuple[list[_Prompt], list[np.ndarray]]:
    """The prompts a box-only source would give for *polygons*, and their OBBs.

    Mirrors ``detectkit/jobs/sam2_prompts.build_prompts`` (core cannot import
    it): the box is the OBB's axis-aligned extent, the positive point is the
    OBB's corner mean, and the centres of overlapping neighbours are
    negatives. The OBB is the minimum-area rectangle, as ``derive_down``
    computes it.
    """
    obbs = [
        cv2.boxPoints(
            cv2.minAreaRect(np.asarray(p, dtype=np.float32).reshape(-1, 1, 2))
        ).astype(np.float32)
        for p in polygons
    ]
    aabbs = [
        (
            float(o[:, 0].min()),
            float(o[:, 1].min()),
            float(o[:, 0].max()),
            float(o[:, 1].max()),
        )
        for o in obbs
    ]
    centers = [(float(o[:, 0].mean()), float(o[:, 1].mean())) for o in obbs]
    prompts = [
        _Prompt(
            box_xyxy=aabbs[i],
            positive_points=[centers[i]],
            negative_points=[
                centers[j]
                for j in range(len(obbs))
                if j != i and _overlaps(aabbs[i], aabbs[j])
            ],
        )
        for i in range(len(obbs))
    ]
    return prompts, obbs


def calibrate_geometry(
    executor,
    frames: Sequence[tuple[Path, Sequence[np.ndarray]]],
    *,
    reference_body_px: float,
    tile_fractions: Sequence[float | None] = TILE_FRACTION_GRID,
    overlap: float = DEFAULT_OVERLAP,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[GeometryCalibrationPoint]:
    """Sweep *tile_fractions* over *frames* (image path, GT polygons in px).

    Only fractions that resolved on EVERY completed frame are reported, as in
    SAM3 calibration: a fraction measured on a subset has incomparable cost.
    A (frame, fraction) pass interrupted by *should_stop* is not counted.
    """
    plans: list[tuple[Path, list, list]] = []
    for img_path, polygons in frames:
        image = cv2.imread(str(img_path))
        if image is None or not len(polygons):
            continue
        h, w = image.shape[:2]
        del image
        options = candidate_tile_plans(
            (h, w), reference_body_px, fractions=tile_fractions, overlap=overlap
        )
        plans.append((Path(img_path), list(polygons), options))
    total_passes = max(1, sum(len(opts) for _p, _g, opts in plans))

    acc: dict[float | None, dict] = {}
    completed: set[int] = set()
    done = 0
    cancelled = False
    for fi, (img_path, polygons, options) in enumerate(plans):
        if cancelled:
            break
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        prompts, obbs = prompts_from_ground_truth(polygons)
        for opt in options:
            if should_stop is not None and should_stop():
                cancelled = True
                break
            started = time.perf_counter()
            outcomes = segment_boxes(executor, image, prompts, opt.plan.tiles)
            elapsed = time.perf_counter() - started
            entry = acc.setdefault(
                opt.fraction,
                {
                    "tile_px": opt.tile_px,
                    "seconds": 0.0,
                    "encodes": 0,
                    "frames": 0,
                    "ious": [],
                    "fallbacks": 0,
                    "seam": 0,
                },
            )
            owners = {o.owner_tile for o in outcomes if o.owner_tile is not None}
            seam = sum(1 for o in outcomes if o.owner_tile is None)
            entry["encodes"] += len(owners) + (1 if seam else 0)
            entry["seam"] += seam
            entry["seconds"] += elapsed
            entry["frames"] += 1
            for outcome, obb, gt in zip(outcomes, obbs, polygons):
                contour = None
                if outcome.mask is not None:
                    contour = mask_to_contour(
                        clip_mask_to_polygon(outcome.mask, obb.tolist())
                    )
                if contour is None:
                    entry["fallbacks"] += 1
                    entry["ious"].append(0.0)
                else:
                    entry["ious"].append(float(polygon_iou(contour, gt)))
            completed.add(fi)
            done += 1
            if progress is not None:
                progress(
                    int(100 * done / total_passes),
                    f"Calibrating frame {fi + 1}/{len(plans)} "
                    f"({opt.tile_px or 'full frame'} px tiles)",
                )
        del image

    n_frames = len(completed)
    points: list[GeometryCalibrationPoint] = []
    for fraction, entry in acc.items():
        if entry["frames"] < n_frames or not entry["ious"]:
            continue
        ious = np.asarray(entry["ious"], dtype=np.float64)
        n = len(ious)
        points.append(
            GeometryCalibrationPoint(
                tile_fraction=fraction,
                tile_px=entry["tile_px"],
                owned_tiles_per_frame=entry["encodes"] / entry["frames"],
                seconds_per_frame=entry["seconds"] / entry["frames"],
                median_iou=float(np.median(ious)),
                p10_iou=float(np.percentile(ious, 10)),
                fallback_rate=entry["fallbacks"] / n,
                seam_fallback_rate=entry["seam"] / n,
                n_instances=n,
            )
        )
    if cancelled:
        return []
    return points


def recommend_geometry(
    points: Sequence[GeometryCalibrationPoint],
    *,
    min_instances: int = MIN_INSTANCES,
    iou_floor: float = IOU_FLOOR,
    fallback_ceil: float = FALLBACK_CEIL,
) -> tuple[GeometryCalibrationPoint | None, str]:
    """The cheapest tiling whose masks match the polygons, or a refusal.

    Among points clearing the IoU floor and the fallback ceiling, fewest
    owned tiles per frame wins (a whole-project run is roughly linear in
    it), ties broken by the higher median IoU.
    """
    if not points:
        return None, "No calibration points; nothing to recommend."
    on_iou = [p for p in points if p.median_iou >= iou_floor]
    if not on_iou:
        best = max(p.median_iou for p in points)
        return None, (
            f"No tile size reached a median mask IoU of {iou_floor:.2f} against "
            f"your polygons (best {best:.2f}). SAM2 is not segmenting these "
            "animals well at any tested scale."
        )
    on_fallback = [p for p in on_iou if p.fallback_rate <= fallback_ceil]
    if not on_fallback:
        best = min(p.fallback_rate for p in on_iou)
        return None, (
            f"Every tile size with good masks fell back to the original box on "
            f"at least {best:.0%} of animals (limit {fallback_ceil:.0%})."
        )
    eligible = [p for p in on_fallback if p.n_instances >= min_instances]
    if not eligible:
        best = max(p.n_instances for p in on_fallback)
        return None, (
            f"Insufficient data: only {best} labelled instances were scored "
            f"(need {min_instances}). Label polygons on a few more frames."
        )
    return min(eligible, key=lambda p: (p.owned_tiles_per_frame, -p.median_iou)), ""
