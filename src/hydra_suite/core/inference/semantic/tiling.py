"""Tiled semantic inference: seam handling and cross-tile merge.

The tile GRID comes from ``utils/slice_geometry.py`` -- the same planner
training and the DetectKit preview use -- so this module never invents a
second tiling convention. What it adds is what that module does not cover:
dropping detections cut by an interior tile seam, offsetting tile-local
polygons into frame space, and merging duplicates across overlapping tiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from hydra_suite.core.inference.masks import polygon_iou
from hydra_suite.utils.slice_geometry import SlicePlan, plan_tiles

from .base import SemanticInstance, SemanticLabeler

logger = logging.getLogger(__name__)

# Tile edge = reference_body_px / fraction. The fraction is a CALIBRATED
# parameter, not a tuned constant: the seed below was back-derived from a
# single measured-good configuration (1504 px tile at body 80 px) on a single
# dataset, and has no independent grounding. It exists to prefill the dialog
# when the user skips calibration. Calibration sweeps TILE_FRACTION_GRID
# against the user's own labelled frames and the user picks the point.
SEMANTIC_TILE_FRACTION_SEED = 0.05
# None means "no tiling, one full-frame pass" -- the right answer on a rig
# where animals are already large at native resolution, where tiling HURTS.
TILE_FRACTION_GRID: tuple[float | None, ...] = (0.03, 0.05, 0.10, None)

DEFAULT_OVERLAP = 0.5
DEFAULT_SEAM_MARGIN_PX = 4
DEFAULT_MERGE_IOU = 0.5


@dataclass(frozen=True)
class TileCandidate:
    """One surviving detection from one tile, in FRAME pixel space."""

    polygon_px: np.ndarray  # (P, 2) float32
    confidence: float
    tile_index: int


def resolve_tile_px(
    reference_body_px: float | None,
    fraction: float | None,
) -> int | None:
    """Tile edge in pixels; None means "full frame, no tiling".

    None is returned for an unknown object scale AND for ``fraction=None``,
    which is the grid's explicit no-tiling option. Deliberately never reads
    ``SliceTrainingSettings.object_tile_fraction``: the sliced-training
    optimum and the SAM3 optimum differ by ~3x, so one persisted fraction
    cannot serve both.
    """
    if fraction is None:
        return None
    if reference_body_px is None or float(reference_body_px) <= 0:
        return None
    frac = max(0.01, min(0.9, float(fraction)))
    return int(max(64, min(4096, round(float(reference_body_px) / frac))))


@dataclass(frozen=True)
class TilePlanOption:
    """One tiling configuration to calibrate: a fraction and its concrete plan."""

    fraction: float | None  # None = full frame
    tile_px: int | None
    plan: SlicePlan
    tiles_per_frame: int


def candidate_tile_plans(
    frame_hw,
    reference_body_px: float | None,
    *,
    fractions: Sequence[float | None] = TILE_FRACTION_GRID,
    overlap: float = DEFAULT_OVERLAP,
) -> list[TilePlanOption]:
    """Resolve the calibration grid into concrete, non-degenerate tile plans.

    Skips (never raises for) fractions that cannot apply on this frame: an
    unknown object scale, a tile at least as large as the frame -- which
    would merely duplicate the full-frame pass -- and any geometry breaching
    ``MAX_TILES_PER_FRAME``. Full frame is always present, so the grid can
    always answer "do not tile".
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    out: list[TilePlanOption] = []
    for frac in fractions:
        tile_px = resolve_tile_px(reference_body_px, frac)
        if frac is not None and tile_px is None:
            continue
        if tile_px is not None and tile_px >= min(frame_w, frame_h):
            logger.info(
                "Skipping tile fraction %s: tile %d px covers the %dx%d frame; "
                "the full-frame option already measures this.",
                frac,
                tile_px,
                frame_w,
                frame_h,
            )
            continue
        try:
            plan = (
                full_frame_plan((frame_h, frame_w))
                if tile_px is None
                else plan_for_frame((frame_h, frame_w), tile_px, overlap)
            )
        except ValueError as exc:
            logger.info("Skipping tile fraction %s: %s", frac, exc)
            continue
        out.append(TilePlanOption(frac, tile_px, plan, len(plan.tiles)))
    return out


def plan_for_frame(frame_hw, tile_px: int, overlap: float) -> SlicePlan:
    """Tile plan for one frame. Raises ValueError above the tile ceiling."""
    return plan_tiles(
        frame_hw, int(tile_px), int(tile_px), float(overlap), float(overlap)
    )


def full_frame_plan(frame_hw) -> SlicePlan:
    """The no-tiling plan: one tile covering the frame, hence no interior seams.

    Kept here so the job, calibration and the grid all express "do not tile"
    the same way rather than each open-coding a degenerate plan.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    return plan_tiles((frame_h, frame_w), frame_w, frame_h, 0.0, 0.0)


def _touches_interior_seam(
    polygon_px: np.ndarray,
    tile: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    margin_px: float,
) -> bool:
    """True if the polygon comes within margin_px of a NON-frame tile edge.

    With overlap, an object clipped by an interior seam is interior to some
    other tile, so dropping the fragment loses nothing and avoids merging a
    partial contour with a whole one.
    """
    x0, y0, x1, y1 = tile
    fw, fh = frame_wh
    px_min, py_min = polygon_px[:, 0].min(), polygon_px[:, 1].min()
    px_max, py_max = polygon_px[:, 0].max(), polygon_px[:, 1].max()
    if x0 > 0 and px_min <= x0 + margin_px:
        return True
    if y0 > 0 and py_min <= y0 + margin_px:
        return True
    if x1 < fw and px_max >= x1 - margin_px:
        return True
    if y1 < fh and py_max >= y1 - margin_px:
        return True
    return False


def collect_candidates(
    labeler: SemanticLabeler,
    image_bgr: np.ndarray,
    plan: SlicePlan,
    prompt: str,
    *,
    confidence_threshold: float,
    max_instances: int,
    seam_margin_px: float,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[TileCandidate]:
    """Run *labeler* over every tile; return frame-space, seam-clean candidates.

    Seam drop is purely geometric and so threshold-INDEPENDENT: applying it
    here once is exact, and lets the confidence sweep re-merge cached
    candidates without re-running inference.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    out: list[TileCandidate] = []
    for ti, (x0, y0, x1, y1) in enumerate(plan.tiles):
        if should_stop is not None and should_stop():
            break
        tile_img = image_bgr[y0:y1, x0:x1]
        instances = labeler.label_image(
            tile_img,
            prompt,
            confidence_threshold=confidence_threshold,
            max_instances=max_instances,
        )
        offset = np.array([x0, y0], dtype=np.float32)
        for inst in instances:
            poly = np.asarray(inst.polygon_px, dtype=np.float32).reshape(-1, 2) + offset
            if poly.shape[0] < 3:
                continue
            if _touches_interior_seam(
                poly, (x0, y0, x1, y1), (frame_w, frame_h), seam_margin_px
            ):
                continue
            out.append(TileCandidate(poly, float(inst.confidence), ti))
        if progress is not None:
            progress(ti + 1, len(plan.tiles))
    return out


def merge_candidates(
    candidates: Sequence[TileCandidate],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[SemanticInstance]:
    """Threshold then greedy-NMS the candidates into final instances.

    NMS is survivor-dependent, so this MUST be re-run for each swept
    confidence -- post-filtering an already merged set gives a different
    (wrong) answer, because a suppressor removed by the higher threshold
    should resurrect whatever it suppressed.
    """
    kept = sorted(
        (c for c in candidates if c.confidence >= confidence_threshold),
        key=lambda c: -c.confidence,
    )
    survivors: list[TileCandidate] = []
    for cand in kept:
        if any(
            polygon_iou(cand.polygon_px, s.polygon_px) >= iou_threshold
            for s in survivors
        ):
            continue
        survivors.append(cand)
    return [SemanticInstance(s.polygon_px, s.confidence) for s in survivors]
