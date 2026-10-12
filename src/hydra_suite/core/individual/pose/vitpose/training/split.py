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

# <video>__<frame token>__<crop index>.<ext>
_CROP = re.compile(r"^(?P<frame>(?P<video>.+?)__.+)__\d+$")


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

    Groups are shuffled with ``seed`` and taken into validation until about
    ``val_fraction`` of the crops are there (at least one group, and never
    all of them when there are two or more groups).
    """
    groups: dict[str, list[int]] = {}
    for i in ids:
        groups.setdefault(group_key(file_names[i], split_by), []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, int(round(len(ids) * val_fraction)))
    val: list[int] = []
    last: list[int] = []
    for k in keys:
        if len(val) >= target:
            break
        last = groups[k]
        val.extend(last)
    if len(keys) > 1 and len(val) == len(ids):  # keep something to train on
        val = val[: len(val) - len(last)]
    val_set = set(val)
    train = [i for i in ids if i not in val_set]
    return train, [i for i in ids if i in val_set]
