"""Training choices that depend on the project, not on the backend.

Pure (no Qt): the Training Runner calls these for both YOLO-pose and ViTPose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from hydra_suite.core.pose_flip import (
    flip_index,
    normalize_flip_pairs,
    unpaired_side_keypoints,
)


def is_vitpose_run(weights) -> bool:
    """Whether ``weights`` came from a PoseKit ViTPose run.

    ViTPose runs are written under ``<training runs>/vitpose/`` (YOLO runs go
    under the project's ``runs/yolo_pose/``), so the location says which
    backend a ``best.pt`` belongs to.  A ViTPose model copied elsewhere (into a
    project, say) keeps its run's ``run.json`` beside it, which names the
    ViTPose variant.
    """
    import json

    from hydra_suite.paths import get_training_runs_dir

    if not weights:
        return False
    try:
        path = Path(weights).expanduser().resolve()
        root = (get_training_runs_dir() / "vitpose").resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if path.is_relative_to(root):
        return True
    run_json = path.with_name("run.json")
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("variant") in ("S", "B", "L", "H")


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
    allow_unpaired: bool = False,
) -> FlipPlan:
    """Decide the flips for a run.

    A mirror must swap left and right keypoints, so flips run only when every
    side-named keypoint (``left_*``, ``l_*``, ``antL``, ``L1``, ...) is in a
    pair; any left unpaired would teach the model to confuse the sides, so
    flips are off until the Skeleton Editor pairs them.  A skeleton with only
    midline keypoints flips with every keypoint kept in place.
    ``allow_unpaired`` overrides the check for names that only look like sides
    (the user says they are midline keypoints).
    """
    k = len(keypoint_names)
    fliplr = max(0.0, min(1.0, float(fliplr)))
    flipud = max(0.0, min(1.0, float(flipud)))
    pairs = normalize_flip_pairs(flip_pairs, k)
    if fliplr <= 0 and flipud <= 0:
        return FlipPlan(0.0, 0.0, None, pairs, "Flips off.")
    flagged = unpaired_side_keypoints(keypoint_names, pairs)
    overridden = ""
    if allow_unpaired and flagged:
        overridden = (
            " Kept in place as midline (override): "
            + ", ".join(str(keypoint_names[i]) for i in flagged)
            + "."
        )
    unpaired = [] if allow_unpaired else flagged
    if unpaired:
        listed = ", ".join(str(keypoint_names[i]) for i in unpaired)
        return FlipPlan(
            0.0,
            0.0,
            None,
            pairs,
            f"Flips off: {listed} look like left/right keypoints but are not in "
            "a left/right pair. Pair them in the Skeleton Editor (Suggest from "
            "names), then flips swap them. If they are not left/right keypoints, "
            "tick 'Flip with unpaired side-like names'.",
            blocked=True,
        )
    if pairs:
        names = ", ".join(
            f"{keypoint_names[a]} ↔ {keypoint_names[b]}" for a, b in pairs
        )
        return FlipPlan(
            fliplr,
            flipud,
            flip_index(pairs, k),
            pairs,
            f"Flips swap {len(pairs)} left/right pair(s): {names}.{overridden}",
        )
    return FlipPlan(
        fliplr,
        flipud,
        list(range(k)),
        [],
        ("No left/right pairs: flips keep every keypoint in place." + overridden),
    )
