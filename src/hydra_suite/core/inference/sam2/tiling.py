"""Owner-tile SAHI for box-prompted SAM2 segmentation (Qt-free, torch-free).

SAM2 resizes whatever image it is given to 1024 px, so on a large frame a
small animal's box prompt collapses to a few pixels and the mask is poor or
empty. Segmenting inside a tile keeps the animal at a usable scale.

Unlike SAM3 there is nothing to detect and nothing to merge: every box is
already known. Each box is segmented ONCE, inside the tile that fully
contains it with the largest margin, and tiles that own no box are never
encoded. A box that no tile fully contains (it straddles every seam) is
segmented on the full frame, which is exactly what happened before tiling.
The tile GRID itself comes from ``core/inference/semantic/tiling.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SegmentOutcome:
    """One box's segmentation, in FRAME coordinates."""

    mask: np.ndarray | None  # bool mask the size of the frame
    iou: float  # SAM2's own predicted IoU for the chosen mask
    owner_tile: int | None  # None = segmented on the full frame (seam fallback)


def _margin(box, tile) -> float:
    x1, y1, x2, y2 = box
    tx0, ty0, tx1, ty1 = tile
    return min(x1 - tx0, y1 - ty0, tx1 - x2, ty1 - y2)


def assign_owner_tiles(
    boxes_xyxy: Sequence[Sequence[float]],
    tiles: Sequence[Sequence[int]],
) -> tuple[dict[int, list[int]], list[int]]:
    """(tile index -> owned box indices, unowned box indices).

    A box is owned by the tile that FULLY contains it with the largest
    minimum margin; ties go to the lowest tile index, so the assignment is
    deterministic. Box indices within a tile stay ascending.
    """
    owned: dict[int, list[int]] = {}
    unowned: list[int] = []
    for bi, box in enumerate(boxes_xyxy):
        best, best_margin = None, -1.0
        for ti, tile in enumerate(tiles):
            margin = _margin(box, tile)
            if margin >= 0 and margin > best_margin:
                best, best_margin = ti, margin
        if best is None:
            unowned.append(bi)
        else:
            owned.setdefault(best, []).append(bi)
    return owned, unowned


def _is_full_frame(tiles, frame_hw) -> bool:
    h, w = frame_hw
    return len(tiles) == 1 and tuple(tiles[0]) == (0, 0, w, h)


def _shift(points, dx: float, dy: float) -> list[tuple[float, float]]:
    return [(float(x) - dx, float(y) - dy) for x, y in points]


def segment_boxes(executor, image, prompts, tiles) -> list[SegmentOutcome]:
    """Segment every prompt, each inside its owner tile; results in prompt order.

    *prompts* need ``box_xyxy``, ``positive_points`` and ``negative_points``
    in frame pixels; *executor* is anything with SAM2's ``set_image`` /
    ``segment`` pair. A plan that is one full-frame tile reproduces the
    pre-tiling call sequence exactly (one ``set_image``, then one
    ``segment`` per prompt with untouched coordinates), which is what keeps
    an uncalibrated run byte-identical.
    """
    h, w = image.shape[:2]
    out: list[SegmentOutcome | None] = [None] * len(prompts)
    if _is_full_frame(tiles, (h, w)):
        executor.set_image(image)
        for i, p in enumerate(prompts):
            mask, iou = executor.segment(
                p.box_xyxy, p.positive_points, p.negative_points
            )
            out[i] = SegmentOutcome(mask, float(iou), 0)
        return out  # type: ignore[return-value]

    owned, unowned = assign_owner_tiles([p.box_xyxy for p in prompts], tiles)
    for ti in sorted(owned):
        x0, y0, x1, y1 = (int(v) for v in tiles[ti])
        executor.set_image(image[y0:y1, x0:x1])
        for bi in owned[ti]:
            p = prompts[bi]
            bx1, by1, bx2, by2 = p.box_xyxy
            # A negative point outside the tile is outside SAM2's image;
            # it cannot be used, only dropped.
            negatives = [
                (x, y) for x, y in p.negative_points if x0 <= x < x1 and y0 <= y < y1
            ]
            mask, iou = executor.segment(
                (bx1 - x0, by1 - y0, bx2 - x0, by2 - y0),
                _shift(p.positive_points, x0, y0),
                _shift(negatives, x0, y0),
            )
            full = np.zeros((h, w), dtype=bool)
            if mask is not None:
                full[y0:y1, x0:x1] = np.asarray(mask, dtype=bool)
            out[bi] = SegmentOutcome(full, float(iou), ti)
    if unowned:
        # Straddles every seam: one full-frame pass for all such boxes,
        # which is the pre-tiling behaviour for them.
        executor.set_image(image)
        for bi in unowned:
            p = prompts[bi]
            mask, iou = executor.segment(
                p.box_xyxy, p.positive_points, p.negative_points
            )
            out[bi] = SegmentOutcome(mask, float(iou), None)
    return out  # type: ignore[return-value]
