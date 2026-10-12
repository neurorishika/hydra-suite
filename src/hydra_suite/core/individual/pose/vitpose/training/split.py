"""Train / validation split that keeps related crops together.

Crops cut from one video frame (``<video>__f000644__003.png``,
``<video>__frame_07790__019.png``) share lighting, background and often the
same neighbours.  Splitting them across train and validation inflates the
validation score, so the split can move whole frames or whole videos.
"""

from __future__ import annotations

import random
import re
from pathlib import PurePosixPath

# <video>__<frame token>__<crop index>.<ext>, optionally with the
# _<sha8> suffix the dataset builders add to colliding names
_CROP = re.compile(r"^(?P<frame>(?P<video>.+?)__.+)__\d+(?:_[0-9a-f]{8})?$")


def group_key(file_name: str, split_by: str) -> str:
    """The unit a crop is split with: itself, its source frame or its video.

    Names that don't follow the crop pattern are their own group.
    """
    stem = PurePosixPath(str(file_name)).stem
    if split_by == "image":
        return stem
    m = _CROP.match(stem)
    if not m:
        return stem
    return m.group("video") if split_by == "video" else m.group("frame")


def split_ids(
    ids: list[int],
    file_names: dict[int, str],
    val_fraction: float,
    seed: int,
    split_by: str = "image",
) -> tuple[list[int], list[int]]:
    """``(train_ids, val_ids)`` with whole groups on one side.

    Groups are visited in a ``seed``-shuffled order and a group joins
    validation when that brings the validation size closer to
    ``val_fraction`` (never past half of the crops).  With fewer than two
    groups the split falls back to single images.
    """
    groups: dict[str, list[int]] = {}
    for i in ids:
        groups.setdefault(group_key(file_names[i], split_by), []).append(i)
    if len(groups) < 2:
        groups = {str(i): [i] for i in ids}
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(ids)
    target = max(1, int(round(n * val_fraction)))
    cap = max(1, n // 2)
    val: list[int] = []
    for k in keys:
        size = len(groups[k])
        if len(val) + size > cap:
            continue
        if abs(len(val) + size - target) < abs(len(val) - target):
            val.extend(groups[k])
    if not val:  # every group alone overshoots: take the smallest
        val = list(min(groups.values(), key=len))
    val_set = set(val)
    if len(val_set) == n:  # keep something to train on
        val_set = set(val[: n // 2])
    train = [i for i in ids if i not in val_set]
    return train, [i for i in ids if i in val_set]
