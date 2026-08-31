"""Merging staged label records into a frame's existing ones.

Every writer in this codebase truncates -- `write_label_file` opens "w",
escalation accept did rmtree+copytree, the X-AnyLabeling sync-back does
rmtree+copy. "Add these instances to what is already there" had no
expression at all, which is why review was all-or-nothing. This is that
missing primitive.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Sequence

from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.utils.polygon_iou import polygon_iou

from .escalation import LabelRecord, derive_down


class MergeMode(Enum):
    """How a staged frame's records combine with the existing ones."""

    OVERWRITE = auto()  # staged replaces existing for this frame
    ADD_NEW = auto()  # existing kept verbatim; non-overlapping staged appended


def merge_records(
    existing: Sequence[LabelRecord],
    staged: Sequence[LabelRecord],
    *,
    mode: MergeMode,
    iou_threshold: float,
    level: GeometryLevel,
) -> list[LabelRecord]:
    """Combine `staged` into `existing` at `level`.

    OVERWRITE returns the staged records alone, derived to `level`.

    ADD_NEW keeps every existing record -- by identity, in order, unmutated
    -- and appends each staged record whose IoU against EVERY existing
    record is below `iou_threshold`. The result is exactly
    ``list(existing) + survivors``; callers rely on that positional
    contract to know which records are new.

    That "a merge can only add" invariant is what makes applying a merge
    immediately (rather than accumulating a pending set) safe: no accept
    can silently degrade labels the user already curated. It is asserted in
    tests rather than trusted.

    IoU uses the rasterised `utils.polygon_iou`, not the convex quad clip in
    `utils/rotated_iou.py`, because staged contours are arbitrary non-convex
    polygons and the convex clip returns wrong areas for them silently.
    Comparison happens at `level`, after derivation, so an OBB source
    compares quads against quads.

    Raises ValueError (via `derive_down`) if a staged record is BELOW
    `level`: deriving upward would invent information.
    """
    staged_at_level = derive_down(list(staged), level)
    if mode is MergeMode.OVERWRITE:
        return staged_at_level

    existing_at_level = derive_down(list(existing), level)
    out = list(existing)
    for candidate in staged_at_level:
        if any(
            polygon_iou(candidate.points, prior.points) >= iou_threshold
            for prior in existing_at_level
        ):
            continue
        out.append(candidate)
    return out
