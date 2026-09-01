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
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .datapoints import build_datapoint, build_negative_datapoint, collate_datapoints


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


def iter_split_datapoints(dataset_dir: Path, params: Any, split: str, *, seed: int = 0):
    """Yield one positive Datapoint per tile, plus its sampled negatives.

    Raises (never silently yields nothing) if the split's annotation file is
    missing, an image cannot be read, or negatives are wanted but
    unavailable.
    """
    split_dir, coco, by_image = _load_coco_split(dataset_dir, split)
    negatives = _negative_prompts_for(dataset_dir, params)
    transform = _default_transform()
    rng = random.Random(seed)

    for image_meta in coco.get("images", []):
        img_path = split_dir / image_meta["file_name"]
        tile_bgr = cv2.imread(str(img_path))
        if tile_bgr is None:
            raise RuntimeError(f"Could not read SAM3 training tile: {img_path}")
        anns = by_image.get(image_meta["id"], [])
        instances = _segmentation_to_polygons(anns)
        # Every instance in the tile is now represented in `instances`
        # (crowd or not -- see `_segmentation_to_polygons`), so the query
        # over this tile is genuinely exhaustive.
        yield build_datapoint(tile_bgr, params.prompt, instances, transform)

        n_neg = min(max(0, int(params.num_negatives)), len(negatives))
        for neg_prompt in (rng.sample(negatives, n_neg) if n_neg else []):
            yield build_negative_datapoint(tile_bgr, neg_prompt, transform)


def build_datapoints(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> list[Any]:
    """Read the built COCO split and return its (positive + negative)
    Datapoints, in dataset order (unshuffled -- see `collate_epoch_batches`
    for the per-epoch shuffle).

    Raises -- never returns `[]` -- when the split is missing or produces
    zero datapoints; the caller (`train.py`) is responsible for refusing to
    train on an empty set, not this function silently pretending there was
    nothing to do.
    """
    dataset_path = Path(dataset_dir).expanduser().resolve()
    datapoints = list(iter_split_datapoints(dataset_path, params, split, seed=seed))
    if not datapoints:
        raise RuntimeError(
            f"SAM3 {split!r} split at {dataset_path} produced zero datapoints; "
            "refusing to build an empty dataloader."
        )
    return datapoints


def try_build_datapoints(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> list[Any] | None:
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


def collate_batches(datapoints: list, batch_size: int) -> list[Any]:
    """Fixed dataset-order batching (no shuffle) -- used for validation,
    where reproducible order across runs is preferable to decorrelation."""
    batch_size = max(1, int(batch_size))
    return [
        collate_datapoints(datapoints[i : i + batch_size])
        for i in range(0, len(datapoints), batch_size)
    ]


def collate_epoch_batches(datapoints: list, batch_size: int, *, seed: int) -> list[Any]:
    """Shuffle a *copy* of `datapoints` deterministically from `seed`, then
    batch it. Call once per epoch with a seed that varies by epoch (e.g.
    `spec.seed + epoch`) so every epoch sees a different order and a tile's
    negatives are not always glued to its positive in the same accumulation
    window, while staying fully reproducible for a given (seed, epoch).
    """
    order = list(range(len(datapoints)))
    random.Random(seed).shuffle(order)
    shuffled = [datapoints[i] for i in order]
    return collate_batches(shuffled, batch_size)
