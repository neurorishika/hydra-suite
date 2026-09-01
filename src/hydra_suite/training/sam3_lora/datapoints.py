"""COCO tile -> SAM3 Datapoint adapter.

Qt-free; no ``sam3`` import at module scope -- that package is training-only
and lazily imported here, inside function bodies, so this module (and
anything that imports it) loads cleanly on a machine without ``sam3``
installed.

Training and serving must agree on input resolution or the sidecar's imgsz
guard fires; `RES` is imported from the same predictor module the inference
path uses, rather than redefined here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hydra_suite.core.inference.semantic.sam3 import PREDICTOR_IMGSZ

# ONE definition of SAM3's input size, shared with the predictor overrides.
RES = PREDICTOR_IMGSZ

# Confirmed against every reference config under sam3/train/configs/ (e.g.
# eval_base.yaml's train_norm_mean/train_norm_std/val_norm_mean/val_norm_std):
# SAM3's image path normalizes with (0.5, 0.5, 0.5) mean/std, not ImageNet's.
SAM3_NORM_MEAN = (0.5, 0.5, 0.5)
SAM3_NORM_STD = (0.5, 0.5, 0.5)


def _scale_polygons_to_res(
    polygons: list[np.ndarray], w: int, h: int
) -> list[np.ndarray]:
    """Scale tile-pixel-space polygons into RES x RES output space.

    `dataset_build.py`'s `tile_size_for_mode` does not always emit exactly
    RES x RES tiles (`auto_object` mode picks a tile size from the measured
    object scale, and right/bottom edge tiles in any mode come back smaller
    than a full tile), so the tile image is resized to (RES, RES) in
    `build_datapoint`. If the polygons were left in the original tile's
    pixel space while the image moved to RES space, every mask/box target
    would silently mistrain against the wrong geometry -- no error, just a
    worse model. A no-op when the tile is already RES x RES, since no resize
    occurs in that case either.
    """
    if (w, h) == (RES, RES):
        return polygons
    scale = np.asarray([RES / float(w), RES / float(h)], dtype=np.float32)
    return [(poly.astype(np.float32) * scale) for poly in polygons]


def _polygon_to_object(polygon: np.ndarray, is_crowd: bool) -> Any:
    """Build one `sam3.train.data.sam3_image_dataset.Object` from a single
    RES-space polygon (already scaled by `_scale_polygons_to_res`).

    `Object.segment` accepts either an RLE dict or a mask; polygons rasterize
    cleanly to a binary mask at RES x RES, so that is what is passed here
    (no lazy RLE decode needed, since it is already a dense mask). `area` is
    the mask's own pixel area (not the bbox area), matching COCO's
    `annotation["area"]` semantics for a polygon instance.
    """
    import cv2
    import torch
    from sam3.train.data.sam3_image_dataset import Object

    mask = np.zeros((RES, RES), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], color=1)
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    bbox = torch.as_tensor(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=torch.float32,
    )
    return Object(
        bbox=bbox,
        area=float(mask.sum()),
        segment=torch.from_numpy(mask),
        is_crowd=is_crowd,
    )


def build_datapoint(
    tile_bgr: Any,
    prompt: str,
    instances: list[tuple[np.ndarray, bool]],
    transform: Any,
) -> Any:
    """One COCO tile -> one Datapoint carrying a single text query.

    `instances` are `(polygon, is_crowd)` pairs, in the tile's original pixel
    space -- polygons are scaled to RES space here, alongside the image
    resize, and turned into `sam3` `Object`s (see `_polygon_to_object`).
    Pass an empty list for a negative query (a prompt that must return
    nothing).

    The query is always exhaustive: every instance in the tile -- crowd or
    not -- is represented in `instances` (see `dataloader._segmentation_to_
    polygons`), with crowd instances carrying `Object(is_crowd=True)` rather
    than being omitted, so there is nothing left unaccounted for that would
    make an exhaustive claim false.
    """
    import cv2
    from PIL import Image as PILImage
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
    )
    from sam3.train.transforms.basic_for_api import NormalizeAPI

    h, w = tile_bgr.shape[:2]
    pil = PILImage.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
    if (w, h) != (RES, RES):
        pil = pil.resize((RES, RES), PILImage.BILINEAR)
    polygons = [polygon for polygon, _is_crowd in instances]
    crowd_flags = [is_crowd for _polygon, is_crowd in instances]
    scaled_polygons = _scale_polygons_to_res(polygons, w, h)
    objects = [
        _polygon_to_object(polygon, is_crowd)
        for polygon, is_crowd in zip(scaled_polygons, crowd_flags)
    ]
    query = FindQueryLoaded(
        query_text=prompt,
        image_id=0,
        # `collate_fn_api` (sam3/train/data/collator.py) builds every find
        # target EXCLUSIVELY from `object_ids_output`, using each entry as a
        # positional index into `Image.objects` -- `Image.objects` itself is
        # never consulted any other way. Leaving this `[]` for a positive
        # query silently collates to num_boxes=0 (indistinguishable from a
        # negative), so it must enumerate every object's position here.
        object_ids_output=list(range(len(objects))),
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=0,
            original_size=(h, w),
            object_id=-1,
            frame_index=-1,
        ),
    )
    datapoint = Datapoint(
        find_queries=[query],
        images=[Image(data=transform(pil), objects=objects, size=(RES, RES))],
        raw_images=[pil],
    )
    # `Object.bbox` is documented (sam3/train/data/sam3_image_dataset.py) as
    # denormalized XYXY on construction, converted to normalized CxCyWH by
    # `NormalizeAPI` -- which also applies the image mean/std the pretrained
    # checkpoint expects. Both jobs happen together, at the Datapoint level;
    # use Meta's own transform rather than reimplementing its maths so the
    # two paths cannot drift. mean/std=(0.5, 0.5, 0.5) is what every SAM3
    # reference train/eval config under `sam3/train/configs/` sets for
    # `train_norm_mean`/`train_norm_std`/`val_norm_mean`/`val_norm_std`.
    normalize = NormalizeAPI(mean=SAM3_NORM_MEAN, std=SAM3_NORM_STD)
    return normalize(datapoint)


def build_negative_datapoint(
    tile_bgr: Any, negative_prompt: str, transform: Any
) -> Any:
    """A negative query datapoint: same structure, no objects, exhaustive."""
    return build_datapoint(tile_bgr, negative_prompt, [], transform)


def collate_datapoints(datapoints: list) -> Any:
    """Batch a list of Datapoints via Meta's own collator."""
    from sam3.train.data.collator import collate_fn_api

    return collate_fn_api(datapoints, dict_key="input", with_seg_masks=True)
