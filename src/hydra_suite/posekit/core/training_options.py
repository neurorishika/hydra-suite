"""Training choices that depend on the project, not on the backend.

Pure (no Qt): the Training Runner calls these for both YOLO-pose and ViTPose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from hydra_suite.core.pose_flip import (
    flip_index,
    normalize_flip_pairs,
    suggest_flip_pairs,
)


@dataclass(frozen=True)
class FlipPlan:
    """What a run does about mirror augmentation."""

    fliplr: float
    flipud: float
    flip_idx: Optional[List[int]]  # keypoint permutation; None when flips are off
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    note: str = ""
    # True when flips were asked for but switched off because the skeleton
    # names left/right keypoints without saying which ones pair up.
    blocked: bool = False

    @property
    def enabled(self) -> bool:
        return self.fliplr > 0 or self.flipud > 0


def plan_flips(
    keypoint_names: Sequence[str],
    flip_pairs: Sequence[Sequence[int]],
    fliplr: float,
    flipud: float,
) -> FlipPlan:
    """Decide the flips for a run.

    A mirror must swap left and right keypoints.  With pairs set, flips run and
    swap them.  With no pairs and no left/right names (a midline-only skeleton)
    flips run with every keypoint kept in place.  With left/right names but no
    pairs, flips would teach the model to confuse the sides, so they are off.
    """
    k = len(keypoint_names)
    fliplr = max(0.0, min(1.0, float(fliplr)))
    flipud = max(0.0, min(1.0, float(flipud)))
    pairs = normalize_flip_pairs(flip_pairs, k)
    if fliplr <= 0 and flipud <= 0:
        return FlipPlan(0.0, 0.0, None, pairs, "Flips off.")
    if pairs:
        names = ", ".join(
            f"{keypoint_names[a]} ↔ {keypoint_names[b]}" for a, b in pairs
        )
        return FlipPlan(
            fliplr,
            flipud,
            flip_index(pairs, k),
            pairs,
            f"Flips swap {len(pairs)} left/right pair(s): {names}.",
        )
    if suggest_flip_pairs(keypoint_names):
        return FlipPlan(
            0.0,
            0.0,
            None,
            [],
            "Flips off: the keypoint names have left and right sides but the "
            "skeleton has no left/right pairs. Set them in the Skeleton Editor "
            "(Suggest from names).",
            blocked=True,
        )
    return FlipPlan(
        fliplr,
        flipud,
        list(range(k)),
        [],
        "No left/right keypoints: flips keep every keypoint in place.",
    )
