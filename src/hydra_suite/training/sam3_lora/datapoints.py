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


def build_datapoint(
    tile_bgr: Any,
    prompt: str,
    polygons: list,
    transform: Any,
    *,
    is_exhaustive: bool = True,
) -> Any:
    """One COCO tile -> one Datapoint carrying a single text query.

    `polygons` are the objects (already in Meta's expected object format) for
    a positive query, in the tile's original pixel space -- they are scaled
    to RES space here, alongside the image resize. Pass an empty list for a
    negative query (a prompt that must return nothing).

    `is_exhaustive` must be False when the tile contains at least one
    `iscrowd` (seam-clipped, partially-retained) instance: those instances
    are excluded from `polygons` but are still physically present in the
    tile, so an exhaustive query would teach the model that a partial animal
    is background. COCO's own meaning of `iscrowd` is "present but
    unannotated", not "absent" -- `is_exhaustive=False` matches that.
    """
    import cv2
    from PIL import Image as PILImage
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
    )

    h, w = tile_bgr.shape[:2]
    pil = PILImage.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
    if (w, h) != (RES, RES):
        pil = pil.resize((RES, RES), PILImage.BILINEAR)
    scaled_polygons = _scale_polygons_to_res(polygons, w, h)
    query = FindQueryLoaded(
        query_text=prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=is_exhaustive,
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
    return Datapoint(
        find_queries=[query],
        images=[Image(data=transform(pil), objects=scaled_polygons, size=(RES, RES))],
        raw_images=[pil],
    )


def build_negative_datapoint(
    tile_bgr: Any, negative_prompt: str, transform: Any
) -> Any:
    """A negative query datapoint: same structure, no objects.

    Always exhaustive -- a negative prompt asks for a different concept than
    whatever `iscrowd` instances might be in the tile, so the crowd caveat
    that forces `is_exhaustive=False` on the positive query does not apply.
    """
    return build_datapoint(tile_bgr, negative_prompt, [], transform)


def collate_datapoints(datapoints: list) -> Any:
    """Batch a list of Datapoints via Meta's own collator."""
    from sam3.train.data.collator import collate_fn_api

    return collate_fn_api(datapoints, dict_key="input", with_seg_masks=True)
