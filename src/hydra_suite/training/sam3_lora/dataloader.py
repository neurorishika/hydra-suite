"""Reads the COCO tile dataset built by `dataset_build.py` and turns it into
batches of Meta Datapoints for `train.py`.

Qt-free; no `sam3` import at module scope -- `datapoints.py`'s helpers own
that lazy import, this module only calls them.

`_default_transform` here only converts the PIL tile to a float tensor;
`datapoints.build_datapoint` applies SAM3's own `NormalizeAPI` (mean/std,
box XYXY -> normalized CxCyWH) at the `Datapoint` level afterwards -- see
that module for the confirmed values and citation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

from .datapoints import build_shared_query_datapoints, collate_datapoints


@dataclass(frozen=True, slots=True)
class InstanceDescriptor:
    """Serializable polygon metadata; never owns pixels, masks, or tensors."""

    polygon: tuple[tuple[float, float], ...]
    is_crowd: bool


@dataclass(frozen=True, slots=True)
class TileDescriptor:
    """All lightweight information needed to load one tile on demand."""

    image_id: int
    image_path: str
    positive_prompt: str
    negative_prompts: tuple[str, ...]
    instances: tuple[InstanceDescriptor, ...]


def _default_transform():
    """Plain float tensor conversion. Mean/std normalization and box
    XYXY -> normalized CxCyWH conversion happen afterwards, at the
    `Datapoint` level, via `datapoints.build_datapoint`'s `NormalizeAPI`
    call -- do not add per-tile normalization here."""
    import torchvision.transforms.functional as tvf

    return tvf.to_tensor


def _load_coco_split(
    dataset_dir: Path, split: str
) -> tuple[Path, dict, dict[int, list[dict]]]:
    split_dir = dataset_dir / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"SAM3 dataset split {split!r} not found at {ann_path}; run "
            "build_sam3_coco_dataset before training."
        )
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)
    return split_dir, coco, by_image


def _negative_prompts_for(dataset_dir: Path, params: Any) -> list[str]:
    """Prefer the manifest's already-resolved negatives (Task 7b's
    `resolve_negative_prompts`, stamped at dataset-build time) so training
    uses exactly what the dataset was built against; fall back to the spec's
    own list only if the manifest is missing.

    `num_negatives == 0` is a valid "no negatives wanted" configuration and
    must not require any negative prompts to exist.
    """
    if int(params.num_negatives) <= 0:
        return []
    manifest_path = dataset_dir / "build_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        negatives = manifest.get("negative_prompts") or []
        if negatives:
            return list(negatives)
    if params.negative_prompts:
        return list(params.negative_prompts)
    raise RuntimeError(
        f"No negative prompts available for {dataset_dir}: build_manifest.json "
        "has none and params.negative_prompts is empty, but "
        f"params.num_negatives={params.num_negatives} wants some."
    )


def _segmentation_to_polygons(annotations: list[dict]) -> list[tuple[np.ndarray, bool]]:
    """Instance polygons, in tile-pixel space (scaled to RES space later,
    inside `build_datapoint`), paired with their `iscrowd` flag.

    Earlier revisions dropped `iscrowd` instances from the object list
    entirely and forced the tile's query to `is_exhaustive=False` to
    compensate (COCO's own meaning of `iscrowd` is "present but
    unannotated", not "absent", so silently omitting it would have taught
    the model those pixels were background). `Object` does have a
    first-class `is_crowd` field, so these instances can stay present with
    `is_crowd=True` instead of being dropped -- but grepping the installed
    `sam3.train.{loss,matcher,data}` shows nothing in the loss, the matcher,
    or the collator actually *reads* `is_crowd`; only the COCO loaders and
    the dataclass definition reference it. It is metadata sam3's own
    training path does not consume, so these instances behave as ordinary
    positives, not specially-weighted ones. That is still the right
    behaviour here (the builder's `MIN_RETAINED_AREA_FRAC` logic only marks
    tile-clipped instances as crowd, not something that needs loss-level
    special-casing), it just is not "the loss handles it" -- nothing does.
    """
    polygons: list[tuple[np.ndarray, bool]] = []
    for ann in annotations:
        is_crowd = bool(ann.get("iscrowd"))
        for seg in ann.get("segmentation") or []:
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                polygons.append((pts, is_crowd))
    return polygons


def build_descriptors(
    dataset_dir: str | Path, params: Any, split: str, *, seed: int = 0
) -> list[TileDescriptor]:
    """Read COCO metadata without decoding or transforming any tile.

    The returned list scales with annotation metadata only. In particular it
    contains no OpenCV/PIL images, float tensors, dense masks, or collated
    batches, making it safe to shuffle and retain for the whole run.
    """
    dataset_path = Path(dataset_dir).expanduser().resolve()
    split_dir, coco, by_image = _load_coco_split(dataset_path, split)
    negatives = _negative_prompts_for(dataset_path, params)
    rng = random.Random(seed)
    descriptors: list[TileDescriptor] = []

    for image_meta in coco.get("images", []):
        img_path = split_dir / image_meta["file_name"]
        anns = by_image.get(image_meta["id"], [])
        instances = _segmentation_to_polygons(anns)
        n_neg = min(max(0, int(params.num_negatives)), len(negatives))
        sampled_negatives = rng.sample(negatives, n_neg) if n_neg else []
        descriptors.append(
            TileDescriptor(
                image_id=int(image_meta["id"]),
                image_path=str(img_path),
                positive_prompt=str(params.prompt),
                negative_prompts=tuple(sampled_negatives),
                instances=tuple(
                    InstanceDescriptor(
                        polygon=tuple((float(x), float(y)) for x, y in polygon),
                        is_crowd=is_crowd,
                    )
                    for polygon, is_crowd in instances
                ),
            )
        )

    if not descriptors:
        raise RuntimeError(
            f"SAM3 {split!r} split at {dataset_path} produced zero tiles; "
            "refusing to build an empty dataloader."
        )
    return descriptors


def load_datapoints(descriptor: TileDescriptor, transform: Any) -> list[Any]:
    """Decode one tile into query-level Datapoints sharing one image tensor."""
    tile_bgr = cv2.imread(descriptor.image_path)
    if tile_bgr is None:
        raise RuntimeError(
            f"Could not read SAM3 training tile: {descriptor.image_path}"
        )
    instances = [
        (np.asarray(instance.polygon, dtype=np.float32), instance.is_crowd)
        for instance in descriptor.instances
    ]
    return build_shared_query_datapoints(
        tile_bgr,
        descriptor.positive_prompt,
        instances,
        descriptor.negative_prompts,
        transform,
    )


def iter_split_datapoints(
    dataset_dir: Path, params: Any, split: str, *, seed: int = 0
) -> Iterator[Any]:
    """Compatibility iterator over query Datapoints, grouped lazily by tile."""
    descriptors = build_descriptors(dataset_dir, params, split, seed=seed)
    transform = _default_transform()
    return (
        datapoint
        for descriptor in descriptors
        for datapoint in load_datapoints(descriptor, transform)
    )


def build_datapoints(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> Iterator[Any]:
    """Return a lazy compatibility iterator over multi-query Datapoints.

    New training code retains ``build_descriptors`` instead so epochs can
    shuffle lightweight metadata and decode each tile only for its batch.
    """
    dataset_path = Path(dataset_dir).expanduser().resolve()
    return iter_split_datapoints(dataset_path, params, split, seed=seed)


def try_build_datapoints(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> Iterator[Any] | None:
    """Like `build_datapoints`, but returns None instead of raising when the
    split's annotation file itself is absent (e.g. no validation frames were
    held out -- see `dataset_build.py`'s `validation: "none"` case). A split
    whose file EXISTS but is empty/unreadable/misconfigured still raises:
    that is a broken dataset, not "no validation configured", and round 1's
    "never fake success" rule applies here too.
    """
    try:
        return build_datapoints(dataset_dir, params, split, seed=seed)
    except FileNotFoundError:
        return None


def try_build_descriptors(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> list[TileDescriptor] | None:
    """Build lightweight descriptors, or ``None`` for an absent split."""
    try:
        return build_descriptors(dataset_dir, params, split, seed=seed)
    except FileNotFoundError:
        return None


def batch_count(item_count: int, batch_size: int) -> int:
    """Number of batches including a final incomplete batch."""
    batch_size = max(1, int(batch_size))
    return -(-int(item_count) // batch_size)


def query_count(descriptors: Sequence[TileDescriptor]) -> int:
    """Number of logical query Datapoints represented by tile descriptors."""
    return sum(1 + len(descriptor.negative_prompts) for descriptor in descriptors)


def collate_batches(
    descriptors: Sequence[TileDescriptor], batch_size: int
) -> Iterator[Any]:
    """Fixed dataset-order batching (no shuffle) -- used for validation,
    where reproducible order across runs is preferable to decorrelation."""
    batch_size = max(1, int(batch_size))
    transform = _default_transform()
    pending: list[Any] = []
    for descriptor in descriptors:
        # The group remains adjacent so its query Datapoints can share one
        # transformed Image without a dataset-sized image cache.
        for datapoint in load_datapoints(descriptor, transform):
            pending.append(datapoint)
            if len(pending) == batch_size:
                batch = collate_datapoints(pending)
                pending.clear()
                yield batch
                del batch
    if pending:
        batch = collate_datapoints(pending)
        pending.clear()
        yield batch
        del batch


def collate_epoch_batches(
    descriptors: Sequence[TileDescriptor], batch_size: int, *, seed: int
) -> Iterator[Any]:
    """Shuffle lightweight descriptor indices and lazily yield each batch.

    A tile's positive and negatives deliberately remain adjacent query-level
    Datapoints so they can share one transformed Image owner without changing
    the established query-batch semantics.
    """
    order = list(range(len(descriptors)))
    random.Random(seed).shuffle(order)
    shuffled = [descriptors[i] for i in order]
    return collate_batches(shuffled, batch_size)
