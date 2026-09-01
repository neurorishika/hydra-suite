"""Reads the COCO tile dataset built by `dataset_build.py` and turns it into
batches of Meta Datapoints for `train.py`.

Qt-free; no `sam3` import at module scope -- `datapoints.py`'s helpers own
that lazy import, this module only calls them.

UNVERIFIED (flag alongside the loss/matcher import paths in train.py): the
per-tile transform (`_default_transform`, plain `to_tensor`) and the shape
handed to `build_datapoint`'s `polygons` argument (a list of `(N, 2)`
float32 pixel-coordinate arrays, one per COCO instance) are inferred from
the COCO tile builder's own output format in `dataset_build.py`, not
confirmed against a live SAM3 install. Verify on the CUDA box.
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
    """Plain float tensor conversion; SAM3's own normalization is unknown
    without the package installed -- verify against `sam3`'s reference
    transform on the CUDA box before relying on this for real training."""
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
    own list only if the manifest is missing."""
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
        "has none and params.negative_prompts is empty."
    )


def _segmentation_to_polygons(annotations: list[dict]) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for ann in annotations:
        if ann.get("iscrowd"):
            continue
        for seg in ann.get("segmentation") or []:
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 3:
                polygons.append(pts)
    return polygons


def iter_split_datapoints(dataset_dir: Path, params: Any, split: str, *, seed: int = 0):
    """Yield one positive Datapoint per tile, plus its sampled negatives.

    Raises (never silently yields nothing) if the split's annotation file is
    missing or if there are no negative prompts to sample from.
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
        polygons = _segmentation_to_polygons(anns)
        yield build_datapoint(tile_bgr, params.prompt, polygons, transform)

        n_neg = min(max(0, int(params.num_negatives)), len(negatives))
        for neg_prompt in (rng.sample(negatives, n_neg) if n_neg else []):
            yield build_negative_datapoint(tile_bgr, neg_prompt, transform)


def build_batches(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> list[Any]:
    """Read the built COCO split and batch it into `params.batch`-sized batches.

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
    batch_size = max(1, int(params.batch))
    return [
        collate_datapoints(datapoints[i : i + batch_size])
        for i in range(0, len(datapoints), batch_size)
    ]


def try_build_batches(
    dataset_dir: str, params: Any, split: str, *, seed: int = 0
) -> list[Any] | None:
    """Like `build_batches`, but returns None instead of raising when the
    split simply does not exist (e.g. no validation frames were held out) --
    used for the optional validation split only. Training callers must use
    `build_batches` directly so a genuinely broken train split still raises.
    """
    try:
        return build_batches(dataset_dir, params, split, seed=seed)
    except (FileNotFoundError, RuntimeError):
        return None
