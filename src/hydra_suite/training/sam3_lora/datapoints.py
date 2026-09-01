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

from hydra_suite.core.inference.semantic.sam3 import PREDICTOR_IMGSZ

# ONE definition of SAM3's input size, shared with the predictor overrides.
RES = PREDICTOR_IMGSZ


def build_datapoint(tile_bgr: Any, prompt: str, polygons: list, transform: Any) -> Any:
    """One COCO tile -> one Datapoint carrying a single text query.

    `polygons` are the objects (already in Meta's expected object format) for
    a positive query; pass an empty list for a negative query (a prompt that
    must return nothing).
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
    query = FindQueryLoaded(
        query_text=prompt,
        image_id=0,
        object_ids_output=[],
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
    return Datapoint(
        find_queries=[query],
        images=[Image(data=transform(pil), objects=polygons, size=(RES, RES))],
        raw_images=[pil],
    )


def build_negative_datapoint(
    tile_bgr: Any, negative_prompt: str, transform: Any
) -> Any:
    """A negative query datapoint: same structure, no objects."""
    return build_datapoint(tile_bgr, negative_prompt, [], transform)


def collate_datapoints(datapoints: list) -> Any:
    """Batch a list of Datapoints via Meta's own collator."""
    from sam3.train.data.collator import collate_fn_api

    return collate_fn_api(datapoints, dict_key="input", with_seg_masks=True)
