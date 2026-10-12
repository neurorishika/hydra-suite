"""Choose which predicted pose belongs to a crop's own animal.

Pose crops are centred on one animal, but neighbours often reach into the
frame and a YOLO-pose model can return a pose for each of them.  The crop's
animal is the one nearest the crop centre; confidence only breaks ties.
(Picking the most confident pose instead reports a neighbour whenever the
neighbour happens to be clearer.)

numpy only: also imported by PoseKit's prediction subprocess.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Poses a YOLO-pose model may return per crop.  More than one is needed for the
# centre rule to choose; a crop rarely holds more than a few animals.
MAX_POSES_PER_CROP = 10


def instance_centers(
    xy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    boxes_xywh: Optional[np.ndarray] = None,
    min_conf: float = 0.0,
) -> np.ndarray:
    """``(N, 2)`` centre of each instance: its box centre when boxes are given,
    else the mean of its keypoints with confidence above ``min_conf`` (all
    finite keypoints when none pass).  Rows that have neither are NaN."""
    xy = np.asarray(xy, dtype=np.float64)
    n = int(xy.shape[0]) if xy.ndim == 3 else 0
    centers = np.full((n, 2), np.nan, dtype=np.float64)
    if boxes_xywh is not None:
        boxes = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
        if boxes.shape[0] == n:
            centers[:] = boxes[:, :2]
    for i in range(n):
        if np.all(np.isfinite(centers[i])):
            continue
        pts = xy[i]
        finite = np.all(np.isfinite(pts), axis=1)
        keep = finite
        if conf is not None:
            c = np.asarray(conf, dtype=np.float64).reshape(n, -1)[i]
            if c.shape[0] == pts.shape[0]:
                confident = finite & (c > min_conf)
                if confident.any():
                    keep = confident
        if keep.any():
            centers[i] = pts[keep].mean(axis=0)
    return centers


def central_instance_index(
    centers: np.ndarray,
    image_wh: Sequence[float],
    scores: Optional[np.ndarray] = None,
) -> int:
    """Index of the instance nearest the image centre.

    Distances within a pixel count as a tie, won by the higher score.  With no
    usable centre at all, the highest score wins (0 without scores).
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    n = centers.shape[0]
    if n == 0:
        return 0
    s = (
        np.zeros(n)
        if scores is None
        else np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1)[:n])
    )
    if s.shape[0] < n:
        s = np.pad(s, (0, n - s.shape[0]))
    mid = np.array([float(image_wh[0]) / 2.0, float(image_wh[1]) / 2.0])
    dist = np.linalg.norm(centers - mid, axis=1)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    if not np.isfinite(dist).any():
        return int(np.argmax(s))
    near = np.flatnonzero(dist <= dist.min() + 1.0)
    return int(near[np.argmax(s[near])])
