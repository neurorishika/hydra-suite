"""Measure a PoseKit label set and suggest a ViTPose input geometry.

Pure measurement: no Qt, and no imports from any app layer -- Training must not
depend on PoseKit. That rule is a benefit here rather than a cost: PoseKit's own
`load_yolo_pose_label` parses only the FIRST line of a label file, so it silently
under-counts multi-animal frames. This module parses every line.

Sizing targets the bare keypoint extent on purpose. Inference already pads by
PADDING_FACTOR = 1.25 inside `box2cs` before warping, so the model sees more than
the animal; the `detail` multiplier is for operator preference, not to compensate
for that padding.

Known divergence from the trainer: this estimator measures the visible-keypoint
extent, but the trainer actually crops the label's STORED bbox, which PoseKit
derives from those same keypoints plus an ISOTROPIC pad. The stored box is
therefore both larger and closer to aspect 1 than what we measure here, so the
suggestion this module produces runs slightly small and slightly more elongated
than the crop the trainer will really use. The `detail` knob absorbs the scale
part of that gap (make the suggestion bigger) but it cannot absorb the aspect
part -- that would require measuring the stored bbox instead of the keypoint
extent, which is a design decision reserved for the user, not fixed here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

SIZE_MULTIPLE = 32
MIN_SIZE = 64
# ViT attention cost tracks token count: 192x256 is 192 tokens, 256x256 is 256,
# 384x384 is 576, 512x512 is 1024. Cap the SUGGESTION so the tool never quietly
# proposes a model that trains several times slower; typing a larger value by
# hand stays available.
MAX_SUGGESTED_SIZE = 384


@dataclass(frozen=True)
class PoseSizeStats:
    sample_count: int  # instances measured, not files read
    frames_scanned: int
    frames_skipped: int
    median_aspect: float  # width / height of the keypoint bounding box
    median_long_px: float
    p90_long_px: float
    suggested_hw: List[int]  # [H, W]
    clamped: bool  # suggestion was constrained (cap and/or floor); may not
    # match the measured aspect -- see `_reduce_pair`.


def _snap_to_grid(value: float) -> int:
    """Round to the nearest multiple of 32 (never below one multiple)."""
    return max(SIZE_MULTIPLE, int(round(value / SIZE_MULTIPLE)) * SIZE_MULTIPLE)


def _reduce_pair(raw_w: float, raw_h: float) -> Tuple[int, int, bool]:
    """Constrain a raw (width, height) pair to the size cap/floor and snap it.

    The cap and the floor are applied to the PAIR, not to each dimension
    independently -- clamping width and height separately discards the aspect
    ratio exactly when it matters most (a 900x100px worm would collapse to a
    384x384 square). Order of operations:

      1. Scale the whole pair down (uniformly) so the long side respects
         MAX_SUGGESTED_SIZE. This step always preserves the aspect ratio.
      2. If the short side is still below MIN_SIZE after that scale, raise
         just that side to MIN_SIZE. At an extreme aspect ratio (long side
         already capped, short side still under the floor) both constraints
         cannot be satisfied at once; the aspect genuinely cannot be
         honoured here, and that is reported via `clamped`, not hidden.
      3. Snap both sides to the nearest multiple of 32 last, so the final
         suggestion stays on-grid.

    `clamped` is True whenever step 1 scaled the pair down OR step 2 had to
    override a side -- i.e. whenever the returned pair may not match the
    measured aspect ratio.
    """
    long_side = max(raw_w, raw_h)
    scale = MAX_SUGGESTED_SIZE / long_side if long_side > MAX_SUGGESTED_SIZE else 1.0
    w = raw_w * scale
    h = raw_h * scale
    clamped = scale < 1.0

    if w < MIN_SIZE:
        w = MIN_SIZE
        clamped = True
    if h < MIN_SIZE:
        h = MIN_SIZE
        clamped = True

    return _snap_to_grid(w), _snap_to_grid(h), clamped


def _instance_extents(
    text: str, num_keypoints: int, w_px: int, h_px: int
) -> List[Tuple[float, float]]:
    """Per-instance (width, height) in pixels of the VISIBLE keypoints' box.

    Invisible keypoints (v == 0) are excluded: they carry no position
    information, and counting them at their stored coordinates would bias every
    box toward the origin.
    """
    out: List[Tuple[float, float]] = []
    need = 5 + 3 * num_keypoints
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < need:
            continue
        try:
            vals = [float(v) for v in parts[1:need]]
        except ValueError:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for i in range(num_keypoints):
            x, y, v = vals[4 + 3 * i], vals[5 + 3 * i], vals[6 + 3 * i]
            if v > 0:
                xs.append(x * w_px)
                ys.append(y * h_px)
        if len(xs) < 2:
            continue
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw <= 0 or bh <= 0:
            continue
        out.append((bw, bh))
    return out


def measure_pose_geometry(
    image_paths: Sequence[Path],
    labels_dir: Path,
    num_keypoints: int,
    *,
    detail: float = 1.0,
    max_images: int = 500,
    seed: int = 0,
) -> PoseSizeStats:
    """Suggest a ViTPose input geometry from a PoseKit label store."""
    if detail <= 0:
        raise ValueError(f"detail must be positive; got {detail}")
    labels_dir = Path(labels_dir)

    labelled: List[Tuple[Path, Path]] = []
    for raw in image_paths:
        img_path = Path(raw)
        label_path = labels_dir / f"{img_path.stem}.txt"
        try:
            if label_path.exists() and label_path.stat().st_size > 0:
                labelled.append((img_path, label_path))
        except OSError:
            continue
    if not labelled:
        raise ValueError(f"no labelled frames found under {labels_dir}")

    if len(labelled) > max_images:
        labelled = random.Random(seed).sample(labelled, max_images)
        labelled.sort()  # order-independent of the sample draw

    widths: List[float] = []
    heights: List[float] = []
    scanned = 0
    skipped = 0
    for img_path, label_path in labelled:
        scanned += 1
        try:
            with Image.open(img_path) as im:
                w_px, h_px = im.size
            text = label_path.read_text(encoding="utf-8")
        except (OSError, ValueError, UnidentifiedImageError):
            skipped += 1
            continue
        extents = _instance_extents(text, num_keypoints, int(w_px), int(h_px))
        if not extents:
            skipped += 1
            continue
        for bw, bh in extents:
            widths.append(bw)
            heights.append(bh)

    if not widths:
        raise ValueError(
            f"found {scanned} labelled frame(s) under {labels_dir} but none held a "
            "usable instance (an instance needs at least 2 visible keypoints)"
        )

    w_arr = np.asarray(widths, dtype=np.float64)
    h_arr = np.asarray(heights, dtype=np.float64)
    long_arr = np.maximum(w_arr, h_arr)
    median_aspect = float(np.median(w_arr / h_arr))
    median_long = float(np.median(long_arr))
    p90_long = float(np.percentile(long_arr, 90))

    # Reconstruct a coherent (W, H) from one length and one aspect. Taking
    # independent medians of width and height could describe an animal that
    # does not exist in the data.
    long_side = median_long * detail
    if median_aspect >= 1.0:
        raw_w, raw_h = long_side, long_side / median_aspect
    else:
        raw_h, raw_w = long_side, long_side * median_aspect

    snapped_w, snapped_h, clamped = _reduce_pair(raw_w, raw_h)
    return PoseSizeStats(
        sample_count=len(widths),
        frames_scanned=scanned,
        frames_skipped=skipped,
        median_aspect=median_aspect,
        median_long_px=median_long,
        p90_long_px=p90_long,
        suggested_hw=[snapped_h, snapped_w],
        clamped=clamped,
    )


# ── crop-size sizing ────────────────────────────────────────────────────────
#
# PoseKit trains and predicts on whole crops: ViTPose fits the crop's full
# extent (times its box padding) into the model input, and YOLO letterboxes the
# crop to ``imgsz``.  So the natural input size follows the crop size, not the
# animal: at "detail 1" each crop pixel maps to about one input pixel.

# Mirrors ``core.individual.pose.vitpose.config.PADDING_FACTOR`` (box2cs scales
# the box by it); copied because importing the ViTPose package pulls in torch.
VITPOSE_BOX_PADDING = 1.25
YOLO_STRIDE = 32
YOLO_MIN_IMGSZ = 64
# Ultralytics' own default; larger crops are downscaled rather than trained at
# a size several times slower.
YOLO_MAX_SUGGESTED_IMGSZ = 640


@dataclass(frozen=True)
class CropSizeStats:
    images_measured: int
    images_skipped: int
    median_w: float
    median_h: float
    p90_long_px: float
    vitpose_hw: List[int]  # [H, W] ViTPose input
    vitpose_clamped: bool
    yolo_imgsz: int


def _even_sample(items: Sequence[Path], limit: int) -> List[Path]:
    """Up to ``limit`` items spread evenly over the list (deterministic)."""
    items = list(items)
    if len(items) <= limit:
        return items
    step = len(items) / float(limit)
    return [items[int(i * step)] for i in range(limit)]


def suggest_yolo_imgsz(long_px: float, detail: float = 1.0) -> int:
    """Smallest stride multiple holding a crop's long side, within YOLO's range."""
    size = int(np.ceil(long_px * detail / YOLO_STRIDE)) * YOLO_STRIDE
    return int(min(YOLO_MAX_SUGGESTED_IMGSZ, max(YOLO_MIN_IMGSZ, size)))


def suggest_vitpose_hw(w_px: float, h_px: float, detail: float = 1.0):
    """ViTPose ``[H, W]`` for a ``w_px`` x ``h_px`` crop, and whether it was clamped."""
    raw_w = w_px * VITPOSE_BOX_PADDING * detail
    raw_h = h_px * VITPOSE_BOX_PADDING * detail
    snapped_w, snapped_h, clamped = _reduce_pair(raw_w, raw_h)
    return [snapped_h, snapped_w], clamped


def measure_crop_geometry(
    image_paths: Sequence[Path],
    *,
    detail: float = 1.0,
    max_images: int = 150,
) -> CropSizeStats:
    """Suggest YOLO ``imgsz`` and a ViTPose input from the crops' pixel sizes.

    Reads image headers only.  The ViTPose input keeps the median crop's aspect
    ratio; YOLO's square ``imgsz`` covers the 90th-percentile long side.
    """
    if detail <= 0:
        raise ValueError(f"detail must be positive; got {detail}")
    sample = _even_sample([Path(p) for p in image_paths], max_images)
    widths: List[float] = []
    heights: List[float] = []
    skipped = 0
    for path in sample:
        try:
            with Image.open(path) as im:
                w_px, h_px = im.size
        except (OSError, ValueError, UnidentifiedImageError):
            skipped += 1
            continue
        if w_px <= 0 or h_px <= 0:
            skipped += 1
            continue
        widths.append(float(w_px))
        heights.append(float(h_px))
    if not widths:
        raise ValueError(
            f"none of the {len(sample)} image(s) sampled could be read"
            if sample
            else "no images"
        )
    w_arr = np.asarray(widths)
    h_arr = np.asarray(heights)
    median_w = float(np.median(w_arr))
    median_h = float(np.median(h_arr))
    p90_long = float(np.percentile(np.maximum(w_arr, h_arr), 90))
    hw, clamped = suggest_vitpose_hw(median_w, median_h, detail)
    return CropSizeStats(
        images_measured=len(widths),
        images_skipped=skipped,
        median_w=median_w,
        median_h=median_h,
        p90_long_px=p90_long,
        vitpose_hw=hw,
        vitpose_clamped=clamped,
        yolo_imgsz=suggest_yolo_imgsz(p90_long),
    )
