"""COCO instance-segmentation tile dataset builder for SAM3 LoRA finetuning.

The source is a single raw DetectKit source (``images/`` + ``labels/`` +
``classes.txt``), not the merged multi-source OBB dataset -- concept training
is per source (see the design's breakage row 5). Tiling reuses
``hydra_suite.utils.slice_geometry`` so the trained tile grid matches the one
inference plans at escalation time (Approach B). Qt-free; no ``sam3`` import
at module scope -- that package is training-only and lazily imported by the
runner (Task 8), never here.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.utils.slice_geometry import (
    clip_polygon_to_tile,
    plan_tiles,
    polygon_area,
    tile_size_for_mode,
)

from ..class_mapping import resolve_dataset_class_names
from ..contracts import Sam3LoraParams, SplitConfig
from ..dataset_builders import IMAGE_EXTS, _find_label_for_obb_image
from ..dataset_builders import _parse_geometry_label_lines as _parse_labels
from ..sliced_dataset import measure_reference_body_px

logger = logging.getLogger(__name__)

# Explicit negative-prompt tiers (see resolve_negative_prompts): curated last
# resort when the source declares only one class and the caller gave none.
CURATED_NEGATIVES = ("background", "shadow", "debris")

# A clipped instance retaining less than half its original area is still a
# visible, real object -- SAM3 must not be taught it is background. It is kept
# as `iscrowd=1` rather than dropped.
MIN_RETAINED_AREA_FRAC = 0.5

# SAM3's native training resolution (see the design's "1008 px OOMs at batch
# 2" note); used only as the `imgsz` fallback for auto_model / auto_object
# geometry modes when no explicit custom tile size is given.
_SAM3_IMGSZ = 1008


def resolve_negative_prompts(
    params: Sam3LoraParams,
    source_class_names: list[str],
    selected_class: str,
) -> list[str]:
    """Negatives are NAMED, not inferred.

    SAM3 trains with prompts that must return nothing so the tuned model keeps
    discriminating concepts. The spike's third-party trainer sampled these
    from other COCO categories -- impossible here, because this builder emits
    a single category by construction. Hence three explicit tiers:
      1. Explicit ``params.negative_prompts``, verbatim.
      2. The source's OTHER class names -- the confusable concepts a
         multi-class DetectKit project already distinguishes.
      3. Curated generic negatives, minus any that share a word with the
         positive prompt (a negative literally naming part of the prompt
         would be self-defeating).
    """
    if params.negative_prompts:
        return list(params.negative_prompts)
    others = [c for c in source_class_names if c != selected_class]
    if others:
        return others
    prompt_words = {w for w in params.prompt.lower().split() if w}
    return [n for n in CURATED_NEGATIVES if not (set(n.lower().split()) & prompt_words)]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _collect_frames(source: Path) -> list[Path]:
    images_dir = source / "images"
    if not images_dir.exists():
        raise RuntimeError(f"Missing images/ directory in SAM3 source {source}")
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _labels_for_frame(
    img_path: Path, images_dir: Path, labels_dir: Path
) -> list[tuple[int, np.ndarray]]:
    lbl_path = _find_label_for_obb_image(img_path, images_dir, labels_dir)
    if lbl_path is None:
        return []
    return _parse_labels(lbl_path)


def _split_frame_stems(
    stems: list[str], split: SplitConfig, seed: int
) -> tuple[list[str], list[str]]:
    ordered = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    if n < 2:
        return ordered, []
    n_val = max(1, round(n * split.val))
    n_val = min(n_val, n - 1)
    if n == 2:
        logger.warning(
            "SAM3 dataset has only 2 frames; validation split is a single frame."
        )
    return ordered[: n - n_val], ordered[n - n_val :]


def _bbox_for_poly(poly: np.ndarray) -> list[float]:
    x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _write_coco_split(
    split_dir: Path,
    category_name: str,
    images: list[dict],
    annotations: list[dict],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name, "supercategory": "object"}],
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )


def _tile_frame(
    img: np.ndarray,
    labels_px: list[np.ndarray],
    tile_w: int,
    tile_h: int,
    overlap: float,
    keep_empty_tiles: bool,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, list[tuple[np.ndarray, bool]]]]:
    """Plan tiles for one frame and clip the selected-class polygons into each.

    Returns a list of (tile_rect, tile_image, [(tile_local_poly, is_crowd)]).
    Tiles with zero instances are omitted unless ``keep_empty_tiles``.
    """
    frame_h, frame_w = img.shape[:2]
    plan = plan_tiles((frame_h, frame_w), tile_w, tile_h, overlap, overlap)
    out = []
    for x0, y0, x1, y1 in plan.tiles:
        xi0, yi0 = max(0, int(x0)), max(0, int(y0))
        xi1, yi1 = min(frame_w, int(x1)), min(frame_h, int(y1))
        crop = img[yi0:yi1, xi0:xi1]
        if crop.size == 0:
            continue
        instances: list[tuple[np.ndarray, bool]] = []
        for poly_px in labels_px:
            full_area = polygon_area(poly_px)
            if full_area <= 1e-6:
                continue
            clipped = clip_polygon_to_tile(poly_px, (xi0, yi0, xi1, yi1))
            if clipped is None:
                continue
            local = clipped.copy()
            local[:, 0] -= xi0
            local[:, 1] -= yi0
            retained_frac = polygon_area(clipped) / full_area
            is_crowd = retained_frac < MIN_RETAINED_AREA_FRAC
            instances.append((local, is_crowd))
        if instances or keep_empty_tiles:
            out.append(((xi0, yi0, xi1, yi1), crop, instances))
    return out


def build_sam3_coco_dataset(
    source_dir: str,
    out_dir: str,
    params: Sam3LoraParams,
    *,
    class_name: str | None = None,
    seed: int = 42,
    split: SplitConfig | None = None,
) -> dict:
    """Build a COCO instance-segmentation tile dataset from one raw source."""
    source = Path(source_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    split_cfg = split or SplitConfig()

    class_names = resolve_dataset_class_names(source)
    selected_class = class_name if class_name in class_names else class_names[0]
    selected_idx = class_names.index(selected_class)

    negatives = resolve_negative_prompts(params, class_names, selected_class)

    images_dir = source / "images"
    labels_dir = source / "labels"
    frames = _collect_frames(source)
    if not frames:
        raise RuntimeError(f"No images found under {images_dir}")

    frame_labels: dict[str, list[np.ndarray]] = {}
    frame_paths: dict[str, Path] = {}
    frame_wh: dict[str, tuple[int, int]] = {}
    for img_path in frames:
        stem = img_path.stem
        frame_paths[stem] = img_path
        raw_labels = _labels_for_frame(img_path, images_dir, labels_dir)
        img_shape = cv2.imread(str(img_path))
        if img_shape is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        h, w = img_shape.shape[:2]
        frame_wh[stem] = (w, h)
        selected_norm = [pts for cls_id, pts in raw_labels if cls_id == selected_idx]
        polys_px = []
        for pts in selected_norm:
            px = np.asarray(pts, dtype=np.float32).copy()
            px[:, 0] *= w
            px[:, 1] *= h
            polys_px.append(px)
        frame_labels[stem] = polys_px

    # Measure reference body size in pixels across all frames' selected-class
    # objects (normalized-point labels expected by measure_reference_body_px).
    per_frame_norm_labels = []
    for img_path in frames:
        stem = img_path.stem
        raw_labels = _labels_for_frame(img_path, images_dir, labels_dir)
        norm_selected = [
            (cls_id, pts) for cls_id, pts in raw_labels if cls_id == selected_idx
        ]
        per_frame_norm_labels.append((norm_selected, frame_wh[stem]))
    majors = []
    for norm_selected, wh in per_frame_norm_labels:
        rbp = measure_reference_body_px(norm_selected, wh)
        if rbp > 0:
            majors.append(rbp)
    reference_body_px = float(np.median(majors)) if majors else 0.0

    tile_w, tile_h = tile_size_for_mode(
        geometry_mode=params.geometry_mode,
        imgsz=_SAM3_IMGSZ,
        reference_body_px=reference_body_px,
        object_tile_fraction=params.object_tile_fraction,
        slice_width=params.slice_width,
        slice_height=params.slice_height,
    )

    train_stems, val_stems = _split_frame_stems(
        list(frame_paths.keys()), split_cfg, seed
    )

    def _build_split(stems: list[str], split_name: str) -> tuple[int, int, int]:
        images: list[dict] = []
        annotations: list[dict] = []
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        image_id = 0
        ann_id = 0
        crowd_count = 0
        for stem in stems:
            img_path = frame_paths[stem]
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"Could not read image: {img_path}")
            tiles = _tile_frame(
                img,
                frame_labels[stem],
                tile_w,
                tile_h,
                params.tile_overlap,
                params.keep_empty_tiles,
            )
            for tile_idx, (_rect, crop, instances) in enumerate(tiles):
                image_id += 1
                file_name = f"{stem}_tile{tile_idx:03d}.jpg"
                cv2.imwrite(str(split_dir / file_name), crop)
                th, tw = crop.shape[:2]
                images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": int(tw),
                        "height": int(th),
                    }
                )
                for local_poly, is_crowd in instances:
                    ann_id += 1
                    if is_crowd:
                        crowd_count += 1
                    bbox = _bbox_for_poly(local_poly)
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "segmentation": [
                                [float(v) for v in local_poly.reshape(-1)]
                            ],
                            "bbox": bbox,
                            "area": float(polygon_area(local_poly)),
                            "iscrowd": 1 if is_crowd else 0,
                        }
                    )
        _write_coco_split(split_dir, params.prompt, images, annotations)
        return image_id, ann_id, crowd_count

    train_images, train_annotations, train_crowd = _build_split(train_stems, "train")
    if val_stems:
        val_images, val_annotations, val_crowd = _build_split(val_stems, "valid")
        validation = "ok"
    else:
        val_images = val_annotations = val_crowd = 0
        validation = "none"

    stats = {
        "train_images": train_images,
        "train_annotations": train_annotations,
        "crowd_annotations": train_crowd + val_crowd,
        "tile_px": [int(tile_w), int(tile_h)],
        "negative_prompts": negatives,
        "validation": validation,
        "selected_class": selected_class,
        "val_images": val_images,
        "val_annotations": val_annotations,
    }

    manifest = {
        "type": "sam3_coco_tiles",
        "source": str(source),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tile_px": [int(tile_w), int(tile_h)],
        "reference_body_px": reference_body_px,
        "object_tile_fraction": params.object_tile_fraction,
        "geometry_mode": params.geometry_mode,
        "tile_overlap": params.tile_overlap,
        "prompt": params.prompt,
        "negative_prompts": negatives,
        "selected_class": selected_class,
        "frame_split": {"train": train_stems, "valid": val_stems},
        "seed": seed,
    }
    (out_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return stats
