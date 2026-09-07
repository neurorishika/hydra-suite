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

import dataclasses
import json
import logging
import os
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

from hydra_suite.core.inference.geometry_drift import (
    GeometrySource,
    log_drift_verdicts,
    log_effective_geometry,
    sidecar_drift_verdicts,
)
from hydra_suite.core.inference.semantic.checkpoints import sidecar_for
from hydra_suite.utils.slice_geometry import (
    DEFAULT_MIN_AREA_RATIO,
    clip_polygon_to_tile,
    plan_tiles,
    polygon_area,
    resolve_scales,
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
    make_dataset_index_path,
    sorted_file_index,
)
from ..sliced_dataset import measure_reference_body_px

logger = logging.getLogger(__name__)


class SplitCounts(NamedTuple):
    """Per-split tallies, including how much precision pressure was given up.

    ``downgraded_tiles`` / ``fragment_only_tiles`` exist so the cost documented
    at ``MIN_RETAINED_AREA_FRAC`` is measurable from a built dataset instead of
    being discovered after a training run.
    """

    tiles: int
    annotations: int
    fragment_annotations: int
    downgraded_tiles: int
    fragment_only_tiles: int


_SCALE_COUNTER_KEYS = (
    "tiles",
    "annotations",
    "fragment_annotations",
    "downgraded_tiles",
    "fragment_only_tiles",
    "non_square_tiles",
)


def _log_scale_table(
    per_scale: dict,
    non_square_totals: dict,
    totals: dict,
) -> None:
    """Print the realised per-scale table before any GPU time is spent.

    Single-scale builds have no per-scale buckets; they still get the split
    line, so the anisotropic-edge-tile count (R2) and the seam-downgrade count
    (R3b) are reported on EVERY path rather than only on the opt-in one.
    """

    for split_name in ("train", "valid"):
        buckets = per_scale.get(split_name) or {}
        split_counts = totals.get(split_name)
        logger.info(
            "SAM3 dataset build [%s]: tiles=%d annotations=%d "
            "downgraded_tiles=%d fragment_only_tiles=%d non_square_tiles=%d",
            split_name,
            getattr(split_counts, "tiles", 0),
            getattr(split_counts, "annotations", 0),
            getattr(split_counts, "downgraded_tiles", 0),
            getattr(split_counts, "fragment_only_tiles", 0),
            int(non_square_totals.get(split_name, 0)),
        )
        for group in sorted(buckets):
            bucket = buckets[group]
            logger.info(
                "SAM3 dataset build [%s] %s: %s",
                split_name,
                group,
                " ".join(
                    f"{key}={int(bucket.get(key, 0))}" for key in _SCALE_COUNTER_KEYS
                ),
            )


# Explicit negative-prompt tiers (see resolve_negative_prompts): curated last
# resort when the source declares only one class and the caller gave none.
CURATED_NEGATIVES = ("background", "shadow", "debris")

# A tile-clipped instance retaining less than this fraction of its original
# area is a FRAGMENT: still a visible, real object, so it is kept as
# `iscrowd=1` rather than dropped (SAM3 must never be taught it is
# background), but it is excluded from the positive query's supervised
# instance list downstream, and the tile is marked non-exhaustive in the same
# step (`datapoints.select_output_objects`).
#
# Deviation from the research spike, and what it costs. The spike (and this
# builder before Task 5) trains every truncated instance as a full exhaustive
# positive, because nothing in `sam3.train.{loss,matcher,data}` reads
# `is_crowd` -- so a 5 %-visible sliver became a full-quality mask target.
# Downgrading instead nullifies the downgraded tile's no-object BCE and
# excludes it from false-positive penalties, i.e. it REMOVES precision
# pressure from the seam-adjacent tiles where false positives are most likely
# -- and extras/frame is the metric this programme optimises. The floor is
# therefore set as high as the fragments are small and no higher: 0.25 (was
# 0.5, which classified even a cleanly halved animal as a fragment and would
# have downgraded roughly half the annotated tiles). At 0.25 a mostly-visible
# animal stays a full positive and only genuinely unreconstructable slivers
# cost a tile its precision pressure. The build manifest reports how many
# tiles were actually downgraded so the size of that cost is visible before
# anyone trains.
#
# D18: this used to be the sole source of truth (a module constant with no
# per-project override); it is now just this field's DEFAULT. The live value
# for any given build is ``Sam3LoraParams.min_area_ratio`` -- both builders
# read a per-build field, and this module constant and the YOLO builder's
# ``SliceBuildParams.min_area_ratio`` share one upstream default,
# ``utils.slice_geometry.DEFAULT_MIN_AREA_RATIO``, so they cannot silently
# drift apart. Kept as a module-level name (rather than deleted) because it
# is still a meaningful "the shipped default" constant referenced by tests,
# docstrings, and the manifest's historical field name.
MIN_RETAINED_AREA_FRAC = DEFAULT_MIN_AREA_RATIO

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
    min_area_ratio: float = MIN_RETAINED_AREA_FRAC,
) -> Iterator[
    tuple[tuple[int, int, int, int], np.ndarray, list[tuple[np.ndarray, bool]]]
]:
    """Plan tiles for one frame and clip the selected-class polygons into each.

    Returns a list of (tile_rect, tile_image, [(tile_local_poly, is_crowd)]).
    Tiles with zero instances are omitted unless ``keep_empty_tiles``.
    ``min_area_ratio`` is the D18 fragment floor (per-build, see
    ``Sam3LoraParams.min_area_ratio``); it defaults to the module constant
    for callers that predate threading it through explicitly.
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
            is_crowd = retained_frac < min_area_ratio
            instances.append((local, is_crowd))
        if instances or keep_empty_tiles:
            yield (xi0, yi0, xi1, yi1), crop, instances


def _median(values: list[float] | tuple[float, ...]) -> float:
    """Lower-interpolated median, matching ``np.median`` without importing it."""
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _median_scale(scale_set: list[tuple[int, int]]) -> tuple[int, int]:
    """The set's median tile size, chosen as an ACTUAL member of the set.

    Averaging two tile sizes would name a scale the build never used; the
    prefill must be a geometry the artifact was really trained at.
    """
    ordered = sorted(scale_set, key=lambda pair: (pair[0] * pair[1], pair))
    return ordered[(len(ordered) - 1) // 2]


def _full_frame_instances(
    img: np.ndarray, labels_px: list[np.ndarray]
) -> list[tuple[np.ndarray, bool]]:
    """The un-tiled arm's instances: every polygon, whole, never a fragment."""
    instances: list[tuple[np.ndarray, bool]] = []
    for poly_px in labels_px:
        if polygon_area(poly_px) <= 1e-6:
            continue
        instances.append((np.asarray(poly_px, dtype=np.float32).copy(), False))
    return instances


def _scaled_frame_jobs(
    img: np.ndarray,
    labels_px: list[np.ndarray],
    stem: str,
    scale_set: list[tuple[int, int]],
    *,
    overlap: float,
    keep_empty_tiles: bool,
    full_frame_mix: bool,
    min_area_ratio: float = MIN_RETAINED_AREA_FRAC,
) -> Iterator[tuple[str, str, tuple[int, int] | None, np.ndarray, list]]:
    """Yield ``(file_name, scale_group, tile_px, crop, instances)`` per emission.

    The multi-scale arm only. ``scale_group`` is emitted as DATA on the COCO
    record (D19); the filename token exists so a filename-only consumer still
    agrees, and both are produced here from the same tuple so they cannot
    drift apart.

    ``MAX_TILES_PER_FRAME`` is enforced by ``plan_tiles`` PER SCALE. The YOLO
    builder swallows that ``ValueError`` and drops the offending scale; SAM3
    re-raises with the scale named. Silently truncating one scale would leave
    a dataset that claims a scale set it does not contain -- corrupting
    exactly the cross-scale comparison this fan-out exists to enable.
    """
    for tile_w, tile_h in scale_set:
        group = f"tile:{tile_w}x{tile_h}"
        # Stepped by hand rather than a for-loop so the ceiling's ValueError
        # is re-raised with the scale named while the tiles stay STREAMED
        # (materializing a scale's tiles would undo this builder's
        # source-independent heap discipline).
        tiles = _tile_frame(
            img,
            labels_px,
            tile_w,
            tile_h,
            overlap,
            keep_empty_tiles,
            min_area_ratio=min_area_ratio,
        )
        tile_idx = 0
        while True:
            try:
                _rect, crop, instances = next(tiles)
            except StopIteration:
                break
            except ValueError as exc:
                raise ValueError(
                    f"SAM3 tiling refused at scale {tile_w}x{tile_h} for frame "
                    f"{stem!r}: {exc}"
                ) from exc
            yield (
                f"{stem}_t{tile_w}x{tile_h}_{tile_idx:04d}.jpg",
                group,
                (tile_w, tile_h),
                crop,
                instances,
            )
            tile_idx += 1
    if full_frame_mix:
        instances = _full_frame_instances(img, labels_px)
        if instances or keep_empty_tiles:
            height, width = img.shape[:2]
            yield (
                f"{stem}_full.jpg",
                "full",
                (int(width), int(height)),
                img,
                instances,
            )


def build_sam3_coco_dataset(
    source_dir: str,
    out_dir: str,
    params: Sam3LoraParams,
    *,
    class_name: str | None = None,
    seed: int = 42,
    split: SplitConfig | None = None,
    io_limits: DatasetIOLimits = DEFAULT_DATASET_IO_LIMITS,
    baseline_model_key: str | None = None,
) -> dict:
    """Build a COCO tile dataset with source-independent Python heap use.

    ``baseline_model_key`` names a published SAM3 artifact this run exists to
    be compared against. Its stamped geometry is read and compared with this
    build's effective geometry, and any divergence is WARNED about -- never
    refused, since a deliberate re-scale is legitimate. This is the exact miss
    that produced the 2026-09-06 confound, where a build at the 0.055 contract
    default was compared against a checkpoint served at 0.10.
    """
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

    database_path = make_dataset_index_path("hydra-sam3-frames-")
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

        # The scale SET. ``resolve_scales`` is fed the contract's fractions
        # DIRECTLY: computing a pixel ratio here and dividing it back out by
        # an imgsz would reintroduce the 640-vs-1008 denominator confusion
        # that the 0.055 incident is an instance of.
        scale_set = resolve_scales(
            geometry_mode=params.geometry_mode,
            imgsz=_SAM3_IMGSZ,
            reference_body_px=reference_body_px,
            fractions=tuple(params.object_tile_fractions),
            object_tile_fraction=params.object_tile_fraction,
            slice_width=params.slice_width,
            slice_height=params.slice_height,
        )
        # Fork on set-emptiness, not on len(scale_set): a one-element set that
        # a user asked for still gets the scale-tagged names, and today's
        # default takes literally today's path (the Task 1 tree-hash golden is
        # what proves that claim rather than asserting it).
        multiscale = bool(params.object_tile_fractions) or bool(params.full_frame_mix)
        tile_w, tile_h = scale_set[0]
        # The median, kept ONLY under an explicitly named `prefill_*` key. A
        # bare collapsed scalar would read as a measurement of "the training
        # tile size" at every consumer that takes one number.
        _effective_fractions = tuple(params.object_tile_fractions) or (
            float(params.object_tile_fraction),
        )
        _prefill_fraction = float(_median(sorted(_effective_fractions)))
        _prefill_tile = _median_scale(scale_set)

        # Provenance + drift guard, before a single tile is written. The
        # 0.055 incident was undetectable because the effective geometry's
        # SOURCE was never printed: a contract default and a deliberate
        # choice looked identical in every artifact this build produced.
        # ``Sam3LoraParams`` is a slots dataclass, so the CLASS attribute is a
        # member descriptor, not the default -- read the default off the field.
        _default_fraction = next(
            field.default
            for field in dataclasses.fields(Sam3LoraParams)
            if field.name == "object_tile_fraction"
        )
        _fraction_source = (
            GeometrySource.CONTRACT_DEFAULT
            if abs(float(params.object_tile_fraction) - float(_default_fraction))
            <= 1e-12
            else GeometrySource.EXPLICIT
        )
        _geometry_values: dict[str, object] = {
            "geometry_mode": params.geometry_mode,
            "object_tile_fraction": float(params.object_tile_fraction),
            "reference_body_px": reference_body_px,
            "tile_px": [int(tile_w), int(tile_h)],
            "imgsz": _SAM3_IMGSZ,
        }
        _geometry_sources = {
            "geometry_mode": GeometrySource.EXPLICIT,
            "object_tile_fraction": _fraction_source,
            # Measured from this project's own labels, above.
            "reference_body_px": GeometrySource.CORPUS_DERIVED,
            "tile_px": GeometrySource.CORPUS_DERIVED,
            "imgsz": GeometrySource.CONTRACT_DEFAULT,
        }
        if multiscale:
            # The whole set, before a single tile is written: a build that
            # fans out N-fold must say so where a reader looks first.
            _geometry_values["object_tile_fractions"] = [
                float(value) for value in params.object_tile_fractions
            ]
            _geometry_values["tile_px_set"] = [[int(w), int(h)] for w, h in scale_set]
            _geometry_values["full_frame_mix"] = bool(params.full_frame_mix)
            _geometry_sources["object_tile_fractions"] = GeometrySource.EXPLICIT
            _geometry_sources["tile_px_set"] = GeometrySource.CORPUS_DERIVED
            _geometry_sources["full_frame_mix"] = GeometrySource.EXPLICIT
        log_effective_geometry(
            logger, "SAM3 dataset build", _geometry_values, _geometry_sources
        )
        if baseline_model_key:
            # Report only. A PREFILL verdict is deliberately NOT adopted here:
            # silently taking the baseline's value would change what this run
            # trains, which is the opposite of an observability guard.
            log_drift_verdicts(
                logger,
                sidecar_drift_verdicts(
                    sidecar_for(baseline_model_key),
                    {
                        "reference_body_px": reference_body_px,
                        "object_tile_fraction": float(params.object_tile_fraction),
                        # STALE-COMMENT FIX: `publish.py:_request_payload`
                        # COLLAPSES a square `tile_px` pair to a scalar before
                        # the child ever writes the sidecar, so a published
                        # single-scale sidecar carries `train_tile_px: 971`, a
                        # number -- not the [w, h] pair the old comment here
                        # claimed. The guard reads both shapes (and a set), so
                        # the pair is still the honest thing to send: it is
                        # what THIS build used.
                        "train_tile_px": (
                            [[int(w), int(h)] for w, h in scale_set]
                            if multiscale
                            else [int(tile_w), int(tile_h)]
                        ),
                    },
                    baseline_label=baseline_model_key,
                ),
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

        def _build_split(
            build_root: Path, split_name: str
        ) -> tuple[SplitCounts, dict[str, dict[str, int]], int]:
            # Realised counts PER SCALE. Empty on the single-scale path so the
            # default manifest gains no key -- the Task 1 golden hashes
            # `build_manifest.json`, so an unconditional key would break it.
            per_scale: dict[str, dict[str, int]] = {}
            split_dir = build_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            images_spool = split_dir / ".images.jsonl"
            annotations_spool = split_dir / ".annotations.jsonl"
            image_id = 0
            ann_id = 0
            crowd_count = 0
            downgraded_tiles = 0
            fragment_only_tiles = 0
            # R2: a non-square tile gets stretched anisotropically to RESxRES
            # by `datapoints.py`. Edge tiles are NOT the cause -- tiling is
            # edge-flushed to full tile size (`slice_geometry._axis_starts`),
            # so every tile this builder cuts is already tile_w x tile_h.
            # Non-square tiles instead come from: a requested tile size
            # clamped down to a smaller frame dimension, a custom
            # (non-square) slice geometry, or the full-frame arm (whose
            # "tile" is the frame itself, of arbitrary aspect). Counted on
            # every path (a single-scale build can hit the clamp too); it
            # rides on the RETURN summary there, because `build_manifest.json`
            # is byte-frozen by the Task 1 golden.
            non_square_tiles = 0
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
                    if multiscale:
                        jobs = _scaled_frame_jobs(
                            image,
                            labels_px,
                            str(stem),
                            scale_set,
                            overlap=params.tile_overlap,
                            keep_empty_tiles=params.keep_empty_tiles,
                            full_frame_mix=params.full_frame_mix,
                            min_area_ratio=params.min_area_ratio,
                        )
                    else:
                        jobs = (
                            (f"{stem}_tile{tile_idx:03d}.jpg", None, None, crop, insts)
                            for tile_idx, (_rect, crop, insts) in enumerate(
                                _tile_frame(
                                    image,
                                    labels_px,
                                    tile_w,
                                    tile_h,
                                    params.tile_overlap,
                                    params.keep_empty_tiles,
                                    min_area_ratio=params.min_area_ratio,
                                )
                            )
                        )
                    for file_name, scale_group, tile_px, crop, instances in jobs:
                        image_id += 1
                        if not cv2.imwrite(str(split_dir / file_name), crop):
                            raise RuntimeError(
                                f"Could not write tile image: {file_name}"
                            )
                        tile_height, tile_width = crop.shape[:2]
                        record = {
                            "id": image_id,
                            "file_name": file_name,
                            "width": int(tile_width),
                            "height": int(tile_height),
                        }
                        if scale_group is not None:
                            # DATA, not a filename to be parsed back out (D19).
                            record["scale_group"] = scale_group
                            record["tile_px"] = [int(tile_px[0]), int(tile_px[1])]
                        json.dump(
                            record,
                            image_records,
                            separators=(",", ":"),
                        )
                        image_records.write("\n")
                        is_non_square = int(tile_width) != int(tile_height)
                        non_square_tiles += int(is_non_square)
                        fragments = sum(1 for _poly, crowd in instances if crowd)
                        if fragments:
                            downgraded_tiles += 1
                            if fragments == len(instances):
                                fragment_only_tiles += 1
                        if scale_group is not None:
                            bucket = per_scale.setdefault(
                                scale_group,
                                {
                                    "tiles": 0,
                                    "annotations": 0,
                                    "fragment_annotations": 0,
                                    "downgraded_tiles": 0,
                                    "fragment_only_tiles": 0,
                                    "non_square_tiles": 0,
                                },
                            )
                            bucket["tiles"] += 1
                            bucket["non_square_tiles"] += int(is_non_square)
                            bucket["annotations"] += len(instances)
                            bucket["fragment_annotations"] += fragments
                            if fragments:
                                bucket["downgraded_tiles"] += 1
                                if fragments == len(instances):
                                    bucket["fragment_only_tiles"] += 1
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
            return (
                SplitCounts(
                    tiles=image_id,
                    annotations=ann_id,
                    fragment_annotations=crowd_count,
                    downgraded_tiles=downgraded_tiles,
                    fragment_only_tiles=fragment_only_tiles,
                ),
                per_scale,
                non_square_tiles,
            )

        with atomic_output_directory(out_root) as build_root:
            train_counts, train_per_scale, train_non_square = _build_split(
                build_root, "train"
            )
            if train_count < frame_count:
                valid_counts, valid_per_scale, valid_non_square = _build_split(
                    build_root, "valid"
                )
                validation = "ok"
            else:
                valid_counts = SplitCounts(0, 0, 0, 0, 0)
                valid_per_scale = {}
                valid_non_square = 0
                validation = "none"
            # BEFORE any GPU time is spent: what the fan-out actually cost, per
            # scale. Anisotropic edge tiles and seam downgrades appear in
            # neither the loss nor any existing artifact, so if this table is
            # not printed the cost is invisible until after a training run.
            _log_scale_table(
                {"train": train_per_scale, "valid": valid_per_scale},
                {"train": train_non_square, "valid": valid_non_square},
                {"train": train_counts, "valid": valid_counts},
            )

            manifest_path = build_root / "build_manifest.json"
            fields = {
                "type": "sam3_coco_tiles",
                "source": str(source),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            if multiscale:
                # The WHOLE set. The legacy scalars are OMITTED rather than
                # filled with the median: a median under a measurement's name
                # reads as "the training tile size" at every downstream
                # surface. The median still travels, under names that say what
                # it is.
                fields.update(
                    {
                        "tile_px_set": [[int(w), int(h)] for w, h in scale_set],
                        "object_tile_fractions": [
                            float(value) for value in _effective_fractions
                        ],
                        "full_frame_mix": bool(params.full_frame_mix),
                        "scale_range_px": [
                            int(min(min(pair) for pair in scale_set)),
                            int(max(max(pair) for pair in scale_set)),
                        ],
                        "prefill_tile_px": [
                            int(_prefill_tile[0]),
                            int(_prefill_tile[1]),
                        ],
                        "prefill_object_tile_fraction": _prefill_fraction,
                        # Realised, not planned: what the build ACTUALLY wrote
                        # at each scale, so the fan-out's cost is visible
                        # before any GPU time is spent.
                        "scale_counts": {
                            "train": train_per_scale,
                            "valid": valid_per_scale,
                        },
                    }
                )
            else:
                fields.update(
                    {
                        "tile_px": [int(tile_w), int(tile_h)],
                        "object_tile_fraction": params.object_tile_fraction,
                    }
                )
            fields.update(
                {
                    "reference_body_px": reference_body_px,
                    "geometry_mode": params.geometry_mode,
                    "tile_overlap": params.tile_overlap,
                    "prompt": params.prompt,
                    "negative_prompts": negatives,
                    "selected_class": selected_class,
                    "min_retained_area_frac": params.min_area_ratio,
                    # Makes the M2 cost of the fragment policy auditable before
                    # any GPU time is spent: how many tiles gave up their
                    # no-object BCE and false-positive penalty, and how many
                    # kept no supervised positive at all.
                    "fragment_counts": {
                        "train": train_counts._asdict(),
                        "valid": valid_counts._asdict(),
                    },
                }
            )
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
            "train_images": train_counts.tiles,
            "train_annotations": train_counts.annotations,
            "crowd_annotations": (
                train_counts.fragment_annotations + valid_counts.fragment_annotations
            ),
            "fragment_annotations": (
                train_counts.fragment_annotations + valid_counts.fragment_annotations
            ),
            "downgraded_tiles": (
                train_counts.downgraded_tiles + valid_counts.downgraded_tiles
            ),
            "fragment_only_tiles": (
                train_counts.fragment_only_tiles + valid_counts.fragment_only_tiles
            ),
            "non_square_tiles": int(train_non_square + valid_non_square),
            "min_retained_area_frac": params.min_area_ratio,
            **(
                {
                    "tile_px_set": [[int(w), int(h)] for w, h in scale_set],
                    "prefill_tile_px": [int(_prefill_tile[0]), int(_prefill_tile[1])],
                }
                if multiscale
                else {"tile_px": [int(tile_w), int(tile_h)]}
            ),
            "negative_prompts": negatives,
            "validation": validation,
            "selected_class": selected_class,
            "val_images": valid_counts.tiles,
            "val_annotations": valid_counts.annotations,
        }
    finally:
        database.close()
        database_path.unlink(missing_ok=True)
