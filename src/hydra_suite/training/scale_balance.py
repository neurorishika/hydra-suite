"""Framework-agnostic scale-grouped batching primitives.

Promoted out of ``ultralytics_scale_balance`` (Task 5 of the multi-scale SAM3
port): grouping a sliced dataset by tile scale, weighting the groups by inverse
frequency, and emitting scale-HOMOGENEOUS batches are all pure functions of a
list of names. Only the *installer* -- which monkeypatches an Ultralytics
trainer -- is framework-specific, and it stays where it was.

**Stdlib only, on purpose.** The SAM3 sidecar (`sam3-lora`) is a slim env; a
module-scope import of a heavy subtree here would break it at epoch 0, exactly
as an eager sklearn import once did.

Note on the two consumers: the Ultralytics side must PARSE the tile name
(`scale_group_for_path`), because that is the only channel Ultralytics gives
it. The SAM3 side does not -- `dataset_build` writes `scale_group` as a field
on the COCO image record, so its adapter reads data and only reuses the
sampler-shaped logic here.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

_TILE_NAME = re.compile(r"_t(?P<width>\d+)x(?P<height>\d+)_\d+$")


def scale_group_for_path(path: str | Path) -> str:
    """Return the emitted sliced-dataset scale group for one image path."""

    stem = Path(path).stem
    if stem.endswith("_full"):
        return "full"
    match = _TILE_NAME.search(stem)
    if match:
        return f"tile:{match.group('width')}x{match.group('height')}"
    return "other"


def scale_group_weights(
    image_paths: Sequence[str | Path], *, power: float
) -> tuple[list[str], dict[str, float]]:
    """Return each image's group and normalized inverse-count group weights.

    Full-frame examples remain at weight one, preserving their configured mix.
    Tile groups use ``(mean_count / group_count) ** power``; ``power=1``
    gives exact group balance and ``0.5`` gives square-root balancing.
    """

    bounded_power = min(1.0, max(0.0, float(power)))
    groups = [scale_group_for_path(path) for path in image_paths]
    counts: dict[str, int] = defaultdict(int)
    for group in groups:
        if group.startswith("tile:"):
            counts[group] += 1
    if not counts:
        return groups, {group: 1.0 for group in set(groups)}
    mean_count = sum(counts.values()) / len(counts)
    weights = {
        group: (mean_count / count) ** bounded_power for group, count in counts.items()
    }
    weights.update({"full": 1.0, "other": 1.0})
    return groups, weights


class ScaleGroupedBatchSampler:
    """Shuffle homogeneous scale batches while yielding every index once/epoch."""

    def __init__(
        self,
        groups: Sequence[str],
        *,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self._groups = list(groups)
        self._batch_size = max(1, int(batch_size))
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set a deterministic shuffle epoch (matching PyTorch samplers)."""

        self._epoch = int(epoch)

    def __len__(self) -> int:
        by_group: dict[str, int] = defaultdict(int)
        for group in self._groups:
            by_group[group] += 1
        if self._drop_last:
            return sum(count // self._batch_size for count in by_group.values())
        return sum(math.ceil(count / self._batch_size) for count in by_group.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._epoch)
        by_group: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(self._groups):
            by_group[group].append(index)
        batches: list[list[int]] = []
        for indices in by_group.values():
            rng.shuffle(indices)
            stop = len(indices) - (len(indices) % self._batch_size)
            if not self._drop_last:
                stop = len(indices)
            for start in range(0, stop, self._batch_size):
                batch = indices[start : start + self._batch_size]
                if len(batch) == self._batch_size or not self._drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches
