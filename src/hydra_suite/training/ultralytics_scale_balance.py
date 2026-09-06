"""No-data-reduction SAHI multi-scale loss balancing for Ultralytics trainers.

The sliced dataset contains every emitted tile. Smaller source tiles naturally
produce more images, which otherwise makes their scale dominate the detector
loss. This module groups training batches by emitted tile size and applies an
inverse-frequency loss multiplier. No tile is removed or replaced.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

_TILE_NAME = re.compile(r"_t(?P<width>\d+)x(?P<height>\d+)_\d+$")
_WEIGHT_KEY = "hydra_sahi_scale_loss_weight"
_PATCH_ATTR = "_hydra_sahi_scale_balance_original"


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


def _balance_settings_from_argv(argv: Sequence[str]) -> dict[str, float] | None:
    """Read enabled balancing settings from the sliced dataset manifest."""

    data_arg = next(
        (str(arg).split("=", 1)[1] for arg in argv if str(arg).startswith("data=")),
        "",
    )
    dataset_yaml = Path(data_arg).expanduser()
    if not dataset_yaml.is_file():
        return None
    try:
        manifest = json.loads((dataset_yaml.parent / "manifest.json").read_text())
        geometry = manifest.get("slice_geometry")
        settings = (
            geometry.get("multiscale_loss_balance")
            if isinstance(geometry, dict)
            else None
        )
        if manifest.get("type") != "sliced_obb" or not isinstance(settings, dict):
            return None
        if not bool(settings.get("enabled", False)):
            return None
        return {"power": min(1.0, max(0.0, float(settings.get("power", 0.5))))}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _grouped_loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    device: Any,
    rank: int,
    seed: int,
    power: float,
):
    """Build an Ultralytics-compatible infinite loader with scale batches."""

    import torch
    from ultralytics.data.build import RANK, InfiniteDataLoader, seed_worker

    groups, weights = scale_group_weights(dataset.im_files, power=power)
    dataset._hydra_sahi_scale_weights = [weights[group] for group in groups]
    sampler = ScaleGroupedBatchSampler(
        groups,
        batch_size=batch_size,
        seed=seed,
        drop_last=False,
    )
    device_type = getattr(device, "type", str(device).split(":")[0])
    worker_count = min(max(0, int(workers)), len(sampler))
    generator = torch.Generator()
    generator.manual_seed((6148914691236517205 + RANK + int(seed)) % (1 << 64))
    loader = InfiniteDataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=worker_count,
        pin_memory=device_type not in {"cpu", "mps"},
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        prefetch_factor=4 if worker_count > 0 else None,
    )
    # BaseTrainer reads ``batch_size`` while checking the final batch. PyTorch
    # sets it to None when ``batch_sampler`` is supplied, although this loader
    # still has a concrete fixed maximum batch size.
    object.__setattr__(loader, "batch_size", max(1, int(batch_size)))
    return loader


def install_sahi_multiscale_loss_balance(argv: Sequence[str] | None = None) -> bool:
    """Install the optional trainer patches selected by a sliced-data manifest.

    Returns whether balance mode was enabled. DDP is deliberately left on the
    unmodified loader: a single-process sampler cannot safely partition every
    scale group across ranks without changing the epoch's data exposure.
    """

    settings = _balance_settings_from_argv(argv or sys.argv[1:])
    if settings is None:
        return False

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel, OBBModel, SegmentationModel
    from ultralytics.utils import LOGGER

    original_getitem = getattr(
        YOLODataset.__getitem__, _PATCH_ATTR, YOLODataset.__getitem__
    )
    if not getattr(YOLODataset.__getitem__, _PATCH_ATTR, False):
        original_collate = YOLODataset.collate_fn

        def weighted_getitem(self: Any, index: int) -> dict[str, Any]:
            item = original_getitem(self, index)
            weights = getattr(self, "_hydra_sahi_scale_weights", None)
            if weights is not None:
                item[_WEIGHT_KEY] = float(weights[index])
            return item

        def weighted_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
            result = original_collate(batch)
            if _WEIGHT_KEY in result:
                import torch

                result[_WEIGHT_KEY] = torch.as_tensor(
                    result[_WEIGHT_KEY], dtype=torch.float32
                )
            return result

        setattr(weighted_getitem, _PATCH_ATTR, original_getitem)
        YOLODataset.__getitem__ = weighted_getitem
        YOLODataset.collate_fn = staticmethod(weighted_collate)

    original_loader = getattr(
        DetectionTrainer.get_dataloader, _PATCH_ATTR, DetectionTrainer.get_dataloader
    )
    if not getattr(DetectionTrainer.get_dataloader, _PATCH_ATTR, False):

        def grouped_get_dataloader(
            self: Any,
            dataset_path: str,
            batch_size: int = 16,
            rank: int = 0,
            mode: str = "train",
        ):
            loader = original_loader(self, dataset_path, batch_size, rank, mode)
            if mode != "train":
                return loader
            if rank != -1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
                LOGGER.warning(
                    "SAHI multi-scale loss balance is disabled for DDP training."
                )
                return loader
            loader.close()
            return _grouped_loader(
                loader.dataset,
                batch_size=batch_size,
                workers=self.args.workers,
                device=self.device,
                rank=rank,
                seed=self.args.seed,
                power=settings["power"],
            )

        setattr(grouped_get_dataloader, _PATCH_ATTR, original_loader)
        DetectionTrainer.get_dataloader = grouped_get_dataloader

    for model_cls in (DetectionModel, OBBModel, SegmentationModel):
        original_loss = getattr(model_cls.loss, _PATCH_ATTR, model_cls.loss)
        if getattr(model_cls.loss, _PATCH_ATTR, False):
            continue

        def weighted_loss(
            self: Any, batch: dict[str, Any], preds: Any = None, _original=original_loss
        ):
            loss, items = _original(self, batch, preds)
            weight = batch.get(_WEIGHT_KEY)
            if weight is None:
                return loss, items
            factor = weight.to(loss.device).mean()
            return loss * factor, {
                name: value * factor.detach() for name, value in items.items()
            }

        setattr(weighted_loss, _PATCH_ATTR, original_loss)
        model_cls.loss = weighted_loss

    LOGGER.info(
        "Hydra SAHI multi-scale loss balance enabled: every tile is retained; "
        f"inverse-frequency power={settings['power']:.2f}."
    )
    return True
