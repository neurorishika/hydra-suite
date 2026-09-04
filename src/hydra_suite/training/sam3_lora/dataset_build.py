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
import os
import random
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from hydra_suite.utils.slice_geometry import (
    clip_polygon_to_tile,
    plan_tiles,
    polygon_area,
    tile_size_for_mode,
)

from ..class_mapping import resolve_dataset_class_names
from ..contracts import Sam3LoraParams, SplitConfig, sam3_prompt_pool_error
from ..dataset_builders import IMAGE_EXTS, _find_label_for_obb_image
from ..dataset_io import (
    DEFAULT_DATASET_IO_LIMITS,
    DatasetIOLimits,
    DatasetLimitError,
    atomic_output_directory,
    iter_bounded_text_lines,
    iter_indexed_paths,
    sorted_file_index,
)
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
    """Legacy compatibility helper; production building uses a disk index."""
    images_dir = source / "images"
    if not images_dir.exists():
        raise RuntimeError(f"Missing images/ directory in SAM3 source {source}")
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _labels_for_frame(
    img_path: Path,
    images_dir: Path,
    labels_dir: Path,
    *,
    limits: DatasetIOLimits = DEFAULT_DATASET_IO_LIMITS,
) -> list[tuple[int, np.ndarray]]:
    lbl_path = _find_label_for_obb_image(img_path, images_dir, labels_dir)
    if lbl_path is None:
        return []
    out: list[tuple[int, np.ndarray]] = []
    for raw in iter_bounded_text_lines(lbl_path, limits=limits):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 5:
            cls_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:])
            points = np.asarray(
                [
                    [cx - width / 2, cy - height / 2],
                    [cx + width / 2, cy - height / 2],
                    [cx + width / 2, cy + height / 2],
                    [cx - width / 2, cy + height / 2],
                ],
                dtype=np.float32,
            )
        elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            point_count = (len(parts) - 1) // 2
            if point_count > limits.max_points_per_object:
                raise DatasetLimitError(
                    f"Label object exceeds {limits.max_points_per_object} points: {lbl_path}"
                )
            cls_id = int(float(parts[0]))
            points = np.asarray(
                [float(value) for value in parts[1:]], dtype=np.float32
            ).reshape(-1, 2)
        else:
            raise RuntimeError(f"Invalid geometry label line in {lbl_path}: {line}")
        out.append((cls_id, points))
    return out


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


def _assemble_coco_split(
    split_dir: Path,
    category_name: str,
    images_spool: Path,
    annotations_spool: Path,
) -> None:
    """Assemble COCO JSON without retaining its arrays or encoded bytes."""

    destination = split_dir / "_annotations.coco.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write('{"images":[')
        for spool in (images_spool,):
            first = True
            with spool.open("r", encoding="utf-8") as records:
                for record in records:
                    if not first:
                        output.write(",")
                    output.write(record.rstrip("\n"))
                    first = False
        output.write('],"annotations":[')
        first = True
        with annotations_spool.open("r", encoding="utf-8") as records:
            for record in records:
                if not first:
                    output.write(",")
                output.write(record.rstrip("\n"))
                first = False
        output.write('],"categories":')
        json.dump(
            [{"id": 1, "name": category_name, "supercategory": "object"}],
            output,
            ensure_ascii=False,
        )
        output.write("}")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)


def _tile_frame(
    img: np.ndarray,
    labels_px: list[np.ndarray],
    tile_w: int,
    tile_h: int,
    overlap: float,
    keep_empty_tiles: bool,
) -> Iterator[
    tuple[tuple[int, int, int, int], np.ndarray, list[tuple[np.ndarray, bool]]]
]:
    """Plan tiles for one frame and clip the selected-class polygons into each.

    Returns a list of (tile_rect, tile_image, [(tile_local_poly, is_crowd)]).
    Tiles with zero instances are omitted unless ``keep_empty_tiles``.
    """
    frame_h, frame_w = img.shape[:2]
    plan = plan_tiles((frame_h, frame_w), tile_w, tile_h, overlap, overlap)
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
            yield (xi0, yi0, xi1, yi1), crop, instances


def build_sam3_coco_dataset(
    source_dir: str,
    out_dir: str,
    params: Sam3LoraParams,
    *,
    class_name: str | None = None,
    seed: int = 42,
    split: SplitConfig | None = None,
    io_limits: DatasetIOLimits = DEFAULT_DATASET_IO_LIMITS,
) -> dict:
    """Build a COCO tile dataset with source-independent Python heap use."""
    source = Path(source_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    split_cfg = split or SplitConfig()

    prompt_error = sam3_prompt_pool_error(params.prompt, params.negative_prompts)
    if prompt_error is not None:
        raise ValueError(f"Invalid SAM3 prompt configuration: {prompt_error}")

    class_names = resolve_dataset_class_names(source)
    selected_class = class_name if class_name in class_names else class_names[0]
    selected_idx = class_names.index(selected_class)

    negatives = resolve_negative_prompts(params, class_names, selected_class)

    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.is_dir():
        raise RuntimeError(f"Missing images/ directory in SAM3 source {source}")

    db_fd, db_name = tempfile.mkstemp(prefix="hydra-sam3-frames-", suffix=".sqlite3")
    os.close(db_fd)
    database_path = Path(db_name)
    database = sqlite3.connect(database_path)
    try:
        database.execute(
            "CREATE TABLE frames ("
            "stem TEXT PRIMARY KEY, path TEXT NOT NULL, width INTEGER NOT NULL, "
            "height INTEGER NOT NULL, reference REAL NOT NULL, position INTEGER, "
            "split TEXT)"
        )
        with sorted_file_index(
            images_dir, suffixes=IMAGE_EXTS, limits=io_limits
        ) as file_index:
            frame_count = 0
            for img_path in iter_indexed_paths(file_index, images_dir):
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    raise RuntimeError(f"Could not read image: {img_path}")
                height, width = image.shape[:2]
                if int(height) * int(width) > io_limits.max_image_pixels:
                    raise DatasetLimitError(
                        f"Image exceeds {io_limits.max_image_pixels} pixels: {img_path}"
                    )
                labels = _labels_for_frame(
                    img_path, images_dir, labels_dir, limits=io_limits
                )
                selected = [entry for entry in labels if entry[0] == selected_idx]
                reference = float(measure_reference_body_px(selected, (width, height)))
                try:
                    database.execute(
                        "INSERT INTO frames(stem,path,width,height,reference) VALUES (?,?,?,?,?)",
                        (img_path.stem, str(img_path), width, height, reference),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError(
                        "SAM3 source contains duplicate image stems; output tile names "
                        f"would collide: {img_path.stem}"
                    ) from exc
                frame_count += 1
                del image, labels, selected
        database.commit()
        if frame_count == 0:
            raise RuntimeError(f"No images found under {images_dir}")

        positive_references = int(
            database.execute(
                "SELECT COUNT(*) FROM frames WHERE reference > 0"
            ).fetchone()[0]
        )
        if positive_references:
            middle = (positive_references - 1) // 2
            count = 2 if positive_references % 2 == 0 else 1
            values = [
                float(row[0])
                for row in database.execute(
                    "SELECT reference FROM frames WHERE reference > 0 "
                    "ORDER BY reference LIMIT ? OFFSET ?",
                    (count, middle),
                )
            ]
            reference_body_px = float(sum(values) / len(values))
        else:
            reference_body_px = 0.0

        tile_w, tile_h = tile_size_for_mode(
            geometry_mode=params.geometry_mode,
            imgsz=_SAM3_IMGSZ,
            reference_body_px=reference_body_px,
            object_tile_fraction=params.object_tile_fraction,
            slice_width=params.slice_width,
            slice_height=params.slice_height,
        )

        # Reproduce ``random.shuffle(sorted(stems))`` exactly, but keep the
        # mutable permutation in SQLite rather than a source-sized list.
        for position, (stem,) in enumerate(
            database.execute("SELECT stem FROM frames ORDER BY stem")
        ):
            database.execute(
                "UPDATE frames SET position=? WHERE stem=?", (position, stem)
            )
        database.execute("CREATE UNIQUE INDEX frames_position ON frames(position)")
        rng = random.Random(seed)
        for index in range(frame_count - 1, 0, -1):
            other = rng.randrange(index + 1)
            if other == index:
                continue
            stem_a = database.execute(
                "SELECT stem FROM frames WHERE position=?", (index,)
            ).fetchone()[0]
            stem_b = database.execute(
                "SELECT stem FROM frames WHERE position=?", (other,)
            ).fetchone()[0]
            database.execute("UPDATE frames SET position=-1 WHERE stem=?", (stem_a,))
            database.execute(
                "UPDATE frames SET position=? WHERE stem=?", (index, stem_b)
            )
            database.execute(
                "UPDATE frames SET position=? WHERE stem=?", (other, stem_a)
            )
        if frame_count < 2:
            train_count = frame_count
        else:
            validation_count = max(1, round(frame_count * split_cfg.val))
            validation_count = min(validation_count, frame_count - 1)
            train_count = frame_count - validation_count
        database.execute(
            "UPDATE frames SET split=CASE WHEN position < ? THEN 'train' ELSE 'valid' END",
            (train_count,),
        )
        database.commit()

        def _build_split(build_root: Path, split_name: str) -> tuple[int, int, int]:
            split_dir = build_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            images_spool = split_dir / ".images.jsonl"
            annotations_spool = split_dir / ".annotations.jsonl"
            image_id = 0
            ann_id = 0
            crowd_count = 0
            with (
                images_spool.open("w", encoding="utf-8") as image_records,
                annotations_spool.open("w", encoding="utf-8") as annotation_records,
            ):
                rows = database.execute(
                    "SELECT stem,path,width,height FROM frames WHERE split=? ORDER BY position",
                    (split_name,),
                )
                for stem, stored_path, width, height in rows:
                    img_path = Path(str(stored_path))
                    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0:
                        raise RuntimeError(f"Could not read image: {img_path}")
                    raw_labels = _labels_for_frame(
                        img_path, images_dir, labels_dir, limits=io_limits
                    )
                    labels_px: list[np.ndarray] = []
                    for class_id, points in raw_labels:
                        if class_id != selected_idx:
                            continue
                        pixels = np.asarray(points, dtype=np.float32).copy()
                        pixels[:, 0] *= int(width)
                        pixels[:, 1] *= int(height)
                        labels_px.append(pixels)
                    for tile_idx, (_rect, crop, instances) in enumerate(
                        _tile_frame(
                            image,
                            labels_px,
                            tile_w,
                            tile_h,
                            params.tile_overlap,
                            params.keep_empty_tiles,
                        )
                    ):
                        image_id += 1
                        file_name = f"{stem}_tile{tile_idx:03d}.jpg"
                        if not cv2.imwrite(str(split_dir / file_name), crop):
                            raise RuntimeError(
                                f"Could not write tile image: {file_name}"
                            )
                        tile_height, tile_width = crop.shape[:2]
                        json.dump(
                            {
                                "id": image_id,
                                "file_name": file_name,
                                "width": int(tile_width),
                                "height": int(tile_height),
                            },
                            image_records,
                            separators=(",", ":"),
                        )
                        image_records.write("\n")
                        for local_poly, is_crowd in instances:
                            ann_id += 1
                            crowd_count += int(is_crowd)
                            json.dump(
                                {
                                    "id": ann_id,
                                    "image_id": image_id,
                                    "category_id": 1,
                                    "segmentation": [
                                        [
                                            float(value)
                                            for value in local_poly.reshape(-1)
                                        ]
                                    ],
                                    "bbox": _bbox_for_poly(local_poly),
                                    "area": float(polygon_area(local_poly)),
                                    "iscrowd": 1 if is_crowd else 0,
                                },
                                annotation_records,
                                separators=(",", ":"),
                            )
                            annotation_records.write("\n")
                    del image, raw_labels, labels_px
                image_records.flush()
                annotation_records.flush()
                os.fsync(image_records.fileno())
                os.fsync(annotation_records.fileno())
            _assemble_coco_split(
                split_dir, params.prompt, images_spool, annotations_spool
            )
            images_spool.unlink()
            annotations_spool.unlink()
            return image_id, ann_id, crowd_count

        with atomic_output_directory(out_root) as build_root:
            train_images, train_annotations, train_crowd = _build_split(
                build_root, "train"
            )
            if train_count < frame_count:
                val_images, val_annotations, val_crowd = _build_split(
                    build_root, "valid"
                )
                validation = "ok"
            else:
                val_images = val_annotations = val_crowd = 0
                validation = "none"

            manifest_path = build_root / "build_manifest.json"
            fields = {
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
            }
            with manifest_path.open("w", encoding="utf-8") as manifest:
                manifest.write("{")
                first_field = True
                for key, value in fields.items():
                    if not first_field:
                        manifest.write(",")
                    json.dump(key, manifest)
                    manifest.write(":")
                    json.dump(value, manifest, ensure_ascii=False)
                    first_field = False
                manifest.write(',"frame_split":{')
                for split_index, split_name in enumerate(("train", "valid")):
                    if split_index:
                        manifest.write(",")
                    json.dump(split_name, manifest)
                    manifest.write(":[")
                    first_stem = True
                    for (stem,) in database.execute(
                        "SELECT stem FROM frames WHERE split=? ORDER BY position",
                        (split_name,),
                    ):
                        if not first_stem:
                            manifest.write(",")
                        json.dump(stem, manifest, ensure_ascii=False)
                        first_stem = False
                    manifest.write("]")
                manifest.write('},"seed":')
                json.dump(seed, manifest)
                manifest.write("}")
                manifest.flush()
                os.fsync(manifest.fileno())

        return {
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
    finally:
        database.close()
        database_path.unlink(missing_ok=True)
